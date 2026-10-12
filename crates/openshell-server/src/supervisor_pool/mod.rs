// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Gateway supervisor topology (prototype).
//!
//! With `[openshell.gateway] supervisor_topology = "gateway"`, one shared
//! supervisor Deployment serves every sandbox of the installation. This module
//! is the gateway side of that topology:
//!
//! - Pool replicas authenticate with their platform credential, verified by
//!   the compute driver's [`SupervisorPoolBackend`].
//! - [`assignment`] records which replica owns each sandbox runtime
//!   generation and at which ownership epoch, durably and with CAS.
//! - One gateway process at a time is the elected assignment coordinator
//!   ([`coordinator`]): it alone places sandboxes, reserves fenced recovery
//!   after an owner's enforced lease lapses, fails recoveries that miss their
//!   deadline, drives drain/rebalance handoffs, and lets stop/delete win.
//!   Every coordinator write is fenced by its lease term in the same database
//!   transaction. Its progress is recorded as durable [`operations`].
//! - Pool sync, assignment material, grants, activations, and owner lease
//!   renewals are served by EVERY gateway with per-assignment CAS, so a
//!   coordinator change never interrupts healthy supervision.
//! - Recovery grants, activations, and leases are gateway-signed
//!   `supervisor-recovery.v1` authority the sandbox runtime verifies itself.

pub mod assignment;
pub mod coordinator;
pub mod operations;
pub mod operator;
#[cfg(test)]
mod test_support;

use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use openshell_core::jwt::{
    RECOVERY_RECEIPT_JWT_TYPE, RecoveryReceipt, SUPERVISOR_RECOVERY_PROTOCOL, SandboxId,
    SupervisorAuthBundle, SupervisorAuthority, SupervisorAuthorityKind, recovery_receipt_digest,
    verify_ed25519_document,
};
use openshell_core::proto::{
    ActivateSupervisorAssignmentRequest, ActivateSupervisorAssignmentResponse,
    GetSupervisorAssignmentRequest, GetSupervisorAssignmentResponse, RenewSupervisorLeaseRequest,
    RenewSupervisorLeaseResponse, RequestSupervisorRecoveryGrantRequest,
    RequestSupervisorRecoveryGrantResponse, Sandbox, SandboxPhase, SupervisorAssignmentAction,
    SupervisorPoolAssignment, SyncSupervisorPoolRequest, SyncSupervisorPoolResponse,
};
use openshell_core::sandbox_generation::SandboxGenerationId;
use openshell_core::time::now_ms;
use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};
use tonic::Status;
use tracing::info;

use crate::auth::sandbox_jwt::SandboxSessionJwtAuthority;
use crate::persistence::{PersistenceError, Store, WriteCondition};
pub use assignment::{Assignment, AssignmentState, AssignmentStore, AssignmentTiming};

/// Internal prototype default: two shared supervisor replicas per installation.
pub const DEFAULT_POOL_REPLICAS: i32 = 2;
const MEMBER_OBJECT_TYPE: &str = "supervisor_pool_member";
const MEMBER_TTL_MS: i64 = 6_000;
const SYNC_INTERVAL_MS: u32 = 1_000;
const MEMBER_AUTH_CACHE_TTL: Duration = Duration::from_mins(1);
const GRANT_TTL: Duration = Duration::from_secs(30);

/// Platform credential checks and bootstrap hand-off provided by the
/// compute driver that implements the shared pool.
#[tonic::async_trait]
pub trait SupervisorPoolBackend: Send + Sync + fmt::Debug {
    /// Authenticate a pool replica's platform credential. Must verify that the
    /// caller is a member of this installation's shared supervisor pool.
    async fn authenticate_member(&self, credential: &str) -> Result<PoolMember, Status>;

    /// Read the driver-written supervisor bootstrap for the sandbox's current
    /// runtime generation.
    async fn supervisor_bootstrap(
        &self,
        sandbox_id: &str,
        runtime_generation: &str,
    ) -> Result<SupervisorBootstrap, Status>;
}

/// Authenticated identity of one shared supervisor replica.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PoolMember {
    /// Stable replica name (Kubernetes Pod name).
    pub member_id: String,
    /// Immutable replica UID (Kubernetes Pod UID).
    pub member_uid: String,
}

/// Driver-written supervisor launch material for one runtime generation.
#[derive(Clone)]
pub struct SupervisorBootstrap {
    pub runtime_descriptor_json: Vec<u8>,
    pub auth_bundle_json: Vec<u8>,
    pub main_process_spec_json: String,
    pub proxy_ca_cert_pem: String,
    pub proxy_ca_key_pem: String,
    pub upstream_proxy_ca_pem: String,
}

impl fmt::Debug for SupervisorBootstrap {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("SupervisorBootstrap")
            .finish_non_exhaustive()
    }
}

/// Where a compute factory registers its pool backend at build time.
#[derive(Clone, Default)]
pub struct SupervisorPoolSlot(Arc<Mutex<Option<Arc<dyn SupervisorPoolBackend>>>>);

impl fmt::Debug for SupervisorPoolSlot {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_struct("SupervisorPoolSlot").finish()
    }
}

impl SupervisorPoolSlot {
    pub fn register(&self, backend: Arc<dyn SupervisorPoolBackend>) {
        *self
            .0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(backend);
    }

    #[must_use]
    pub fn take(&self) -> Option<Arc<dyn SupervisorPoolBackend>> {
        self.0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take()
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct MemberRecord {
    member_id: String,
    member_uid: String,
    last_seen_ms: i64,
    /// Sandboxes whose sessions the member reported running at
    /// `last_seen_ms`. A planned handoff's release is confirmed when a report
    /// newer than the handoff no longer lists the sandbox.
    #[serde(default)]
    hosted: Vec<String>,
}

/// A pool member as last reported through sync.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MemberView {
    pub member_id: String,
    pub last_seen_ms: i64,
    pub hosted: std::collections::BTreeSet<String>,
    pub live: bool,
}

/// Gateway-side coordinator for the shared supervisor pool.
pub struct SupervisorPool {
    store: Arc<Store>,
    assignments: AssignmentStore,
    operations: operations::OperationStore,
    backend: Arc<dyn SupervisorPoolBackend>,
    authority: Arc<SandboxSessionJwtAuthority>,
    compute: crate::compute::ComputeRuntime,
    timing: AssignmentTiming,
    member_cache: Mutex<HashMap<[u8; 32], (PoolMember, Instant)>>,
}

impl fmt::Debug for SupervisorPool {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("SupervisorPool")
            .field("timing", &self.timing)
            .finish_non_exhaustive()
    }
}

fn assignment_status(error: assignment::AssignmentError) -> Status {
    use assignment::AssignmentError;
    match error {
        AssignmentError::Terminated | AssignmentError::Failed(_) => {
            Status::failed_precondition(error.to_string())
        }
        AssignmentError::NotOwner
        | AssignmentError::InstanceMismatch
        | AssignmentError::LeaseExpired
        | AssignmentError::DeadlinePassed => Status::permission_denied(error.to_string()),
        AssignmentError::WrongState
        | AssignmentError::LeaseValid
        | AssignmentError::GrantAlreadyIssued
        | AssignmentError::HandoffInProgress
        | AssignmentError::NoHandoff => Status::failed_precondition(error.to_string()),
    }
}

fn store_status(error: PersistenceError) -> Status {
    match error {
        PersistenceError::Conflict { .. } => {
            Status::aborted("supervisor assignment changed concurrently; retry")
        }
        PersistenceError::Fenced => Status::aborted("supervisor pool coordinator changed; retry"),
        other => Status::unavailable(format!("supervisor assignment store: {other}")),
    }
}

/// Phases in which a sandbox runtime generation may be supervised.
fn wants_supervision(phase: SandboxPhase) -> bool {
    matches!(
        phase,
        SandboxPhase::Provisioning | SandboxPhase::Ready | SandboxPhase::Starting
    )
}

fn sandbox_generation(sandbox: &Sandbox) -> Option<String> {
    sandbox.metadata.as_ref().and_then(|metadata| {
        metadata
            .annotations
            .get(crate::auth::sandbox_session::RUNTIME_GENERATION_ANNOTATION)
            .cloned()
    })
}

fn sandbox_credential_epoch(sandbox: &Sandbox) -> u64 {
    sandbox
        .metadata
        .as_ref()
        .and_then(|metadata| {
            metadata
                .annotations
                .get(crate::auth::sandbox_session::AUTH_EPOCH_ANNOTATION)
        })
        .and_then(|value| value.parse().ok())
        .unwrap_or(0)
}

fn has_runtime_binding(sandbox: &Sandbox) -> bool {
    sandbox.metadata.as_ref().is_some_and(|metadata| {
        metadata
            .annotations
            .contains_key(crate::compute::COMPUTE_RUNTIME_IDENTITY_ANNOTATION)
    })
}

impl SupervisorPool {
    pub fn new(
        store: Arc<Store>,
        backend: Arc<dyn SupervisorPoolBackend>,
        authority: Arc<SandboxSessionJwtAuthority>,
        compute: crate::compute::ComputeRuntime,
        timing: AssignmentTiming,
    ) -> Result<Self, String> {
        timing.validate()?;
        Ok(Self {
            assignments: AssignmentStore::new(store.clone()),
            operations: operations::OperationStore::new(store.clone()),
            store,
            backend,
            authority,
            compute,
            timing,
            member_cache: Mutex::new(HashMap::new()),
        })
    }

    #[must_use]
    pub const fn timing(&self) -> AssignmentTiming {
        self.timing
    }

    /// Ownership epoch of the current enforcement owner, used to let a
    /// fenced replacement supersede the stale routing session immediately.
    pub async fn ownership_epoch(&self, sandbox_id: &str) -> Option<u64> {
        self.assignments
            .get(sandbox_id)
            .await
            .ok()
            .flatten()
            .filter(|assignment| {
                matches!(
                    assignment.state,
                    AssignmentState::Active | AssignmentState::Recovering
                )
            })
            .map(|assignment| assignment.ownership_epoch)
    }

    /// Authenticate the pool member that sent `metadata`.
    pub async fn authenticate(
        &self,
        metadata: &tonic::metadata::MetadataMap,
    ) -> Result<PoolMember, Status> {
        let credential = metadata
            .get("authorization")
            .and_then(|value| value.to_str().ok())
            .and_then(openshell_core::auth::strip_bearer_scheme)
            .filter(|credential| !credential.is_empty())
            .ok_or_else(|| Status::unauthenticated("missing supervisor pool credential"))?;
        let key: [u8; 32] = Sha256::digest(credential.as_bytes()).into();
        if let Some((member, verified_at)) = self
            .member_cache
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get(&key)
            .cloned()
            && verified_at.elapsed() < MEMBER_AUTH_CACHE_TTL
        {
            return Ok(member);
        }
        let member = self.backend.authenticate_member(credential).await?;
        let mut cache = self
            .member_cache
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        cache.retain(|_, (_, verified_at)| verified_at.elapsed() < MEMBER_AUTH_CACHE_TTL);
        cache.insert(key, (member.clone(), Instant::now()));
        Ok(member)
    }

    async fn heartbeat(
        &self,
        member: &PoolMember,
        hosted: &[String],
    ) -> Result<(), PersistenceError> {
        let mut hosted = hosted.to_vec();
        hosted.sort();
        hosted.dedup();
        let record = MemberRecord {
            member_id: member.member_id.clone(),
            member_uid: member.member_uid.clone(),
            last_seen_ms: now_ms(),
            hosted,
        };
        let payload = serde_json::to_vec(&record)
            .map_err(|error| PersistenceError::Encode(error.to_string()))?;
        self.store
            .put_if(
                MEMBER_OBJECT_TYPE,
                &format!("supervisor-pool-member:{}", member.member_id),
                &member.member_id,
                "",
                &payload,
                None,
                WriteCondition::Unconditional,
            )
            .await
            .map(|_| ())
    }

    /// Every known member with its latest report. Long-dead records are
    /// garbage collected.
    pub(crate) async fn members(&self, now: i64) -> Result<Vec<MemberView>, PersistenceError> {
        let records = self.store.list_by_type(MEMBER_OBJECT_TYPE, 1000, 0).await?;
        let mut members = Vec::new();
        for record in records {
            let Ok(member) = serde_json::from_slice::<MemberRecord>(&record.payload) else {
                continue;
            };
            let age = now.saturating_sub(member.last_seen_ms);
            if age > MEMBER_TTL_MS * 100 {
                let _ = self
                    .store
                    .delete_if(MEMBER_OBJECT_TYPE, &record.id, record.resource_version)
                    .await;
                continue;
            }
            members.push(MemberView {
                member_id: member.member_id,
                last_seen_ms: member.last_seen_ms,
                hosted: member.hosted.into_iter().collect(),
                live: age < MEMBER_TTL_MS,
            });
        }
        members.sort_by(|a, b| a.member_id.cmp(&b.member_id));
        Ok(members)
    }

    async fn sandbox(&self, sandbox_id: &str) -> Result<Option<Sandbox>, Status> {
        self.store
            .get_message::<Sandbox>(sandbox_id)
            .await
            .map_err(|error| Status::unavailable(format!("load sandbox: {error}")))
    }

    /// Load an assignment and require that the sandbox still wants supervision.
    /// Stop or delete wins: a sandbox that no longer wants supervision has its
    /// assignment terminated before any authority is issued.
    async fn live_assignment(&self, sandbox_id: &str) -> Result<Assignment, Status> {
        let assignment = self
            .assignments
            .get(sandbox_id)
            .await
            .map_err(store_status)?
            .ok_or_else(|| Status::not_found("sandbox has no supervisor assignment"))?;
        let sandbox = self.sandbox(sandbox_id).await?;
        let supervisable = sandbox.as_ref().is_some_and(|sandbox| {
            wants_supervision(
                SandboxPhase::try_from(sandbox.phase()).unwrap_or(SandboxPhase::Unknown),
            ) && sandbox_generation(sandbox).as_deref()
                == Some(assignment.runtime_generation.as_str())
                && sandbox_credential_epoch(sandbox) == assignment.credential_epoch
        });
        if !supervisable {
            if !matches!(
                assignment.state,
                AssignmentState::Terminated | AssignmentState::Failed
            ) {
                let _ = self
                    .assignments
                    .write(Some(&assignment), &assignment.terminate())
                    .await;
            }
            return Err(Status::failed_precondition(
                "sandbox supervision was stopped or deleted",
            ));
        }
        Ok(assignment)
    }

    pub async fn sync(
        &self,
        member: &PoolMember,
        request: SyncSupervisorPoolRequest,
    ) -> Result<SyncSupervisorPoolResponse, Status> {
        self.heartbeat(member, &request.active_sandbox_ids)
            .await
            .map_err(store_status)?;
        let assignments = self.assignments.list().await.map_err(store_status)?;
        let reported: std::collections::HashSet<&str> = request
            .active_sandbox_ids
            .iter()
            .map(String::as_str)
            .collect();
        let assignments = assignments
            .into_iter()
            .filter_map(|assignment| {
                let owned = assignment.owner_member.as_deref() == Some(member.member_id.as_str());
                let action = match assignment.state {
                    AssignmentState::Launching if owned => SupervisorAssignmentAction::Launch,
                    // A planned handoff asks the owner to release ownership
                    // without stopping the workload.
                    AssignmentState::Active if owned && assignment.handoff.is_some() => {
                        SupervisorAssignmentAction::Relinquish
                    }
                    AssignmentState::Active if owned => SupervisorAssignmentAction::Supervise,
                    AssignmentState::Recovering if owned => SupervisorAssignmentAction::Recover,
                    AssignmentState::Terminated | AssignmentState::Failed
                        if reported.contains(assignment.sandbox_id.as_str()) =>
                    {
                        SupervisorAssignmentAction::Release
                    }
                    _ => return None,
                };
                Some(SupervisorPoolAssignment {
                    sandbox_id: assignment.sandbox_id,
                    sandbox_name: assignment.sandbox_name,
                    runtime_generation: assignment.runtime_generation,
                    ownership_epoch: assignment.ownership_epoch,
                    action: action.into(),
                    credential_epoch: assignment.credential_epoch,
                })
            })
            .collect();
        Ok(SyncSupervisorPoolResponse {
            member_id: member.member_id.clone(),
            assignments,
            sync_interval: Some(prost_duration_ms(i64::from(SYNC_INTERVAL_MS))),
        })
    }

    pub async fn get_assignment(
        &self,
        member: &PoolMember,
        request: GetSupervisorAssignmentRequest,
    ) -> Result<GetSupervisorAssignmentResponse, Status> {
        let assignment = self.live_assignment(&request.sandbox_id).await?;
        if assignment.owner_member.as_deref() != Some(member.member_id.as_str())
            || assignment.ownership_epoch != request.ownership_epoch
        {
            return Err(Status::permission_denied(
                "caller is not the assigned supervisor for this epoch",
            ));
        }
        let action = match assignment.state {
            AssignmentState::Launching => SupervisorAssignmentAction::Launch,
            AssignmentState::Recovering => SupervisorAssignmentAction::Recover,
            AssignmentState::Active => SupervisorAssignmentAction::Supervise,
            _ => return Err(Status::failed_precondition("assignment is not runnable")),
        };
        let bootstrap = self
            .backend
            .supervisor_bootstrap(&assignment.sandbox_id, &assignment.runtime_generation)
            .await?;
        let auth_bundle_json = self.fresh_auth_bundle(&assignment, &bootstrap).await?;
        Ok(GetSupervisorAssignmentResponse {
            sandbox_id: assignment.sandbox_id,
            sandbox_name: assignment.sandbox_name,
            ownership_epoch: assignment.ownership_epoch,
            action: action.into(),
            runtime_descriptor_json: bootstrap.runtime_descriptor_json,
            auth_bundle_json,
            main_process_spec_json: bootstrap.main_process_spec_json,
            proxy_ca_cert_pem: bootstrap.proxy_ca_cert_pem,
            proxy_ca_key_pem: bootstrap.proxy_ca_key_pem,
            upstream_proxy_ca_pem: bootstrap.upstream_proxy_ca_pem,
            lease_ttl: Some(prost_duration_ms(self.timing.lease_ttl_ms)),
            lease_interval: Some(prost_duration_ms(self.timing.lease_interval_ms)),
        })
    }

    /// Replace the driver-staged launch tokens with ones minted from the
    /// sandbox's current durable identity. After a recovery reservation the
    /// gateway token lineage was rotated, so the previous owner's gateway
    /// credential no longer authorizes any RPC.
    async fn fresh_auth_bundle(
        &self,
        assignment: &Assignment,
        bootstrap: &SupervisorBootstrap,
    ) -> Result<Vec<u8>, Status> {
        let mut bundle: SupervisorAuthBundle = serde_json::from_slice(&bootstrap.auth_bundle_json)
            .map_err(|_| Status::internal("driver supervisor auth bundle is invalid"))?;
        let sandbox = self
            .sandbox(&assignment.sandbox_id)
            .await?
            .ok_or_else(|| Status::not_found("sandbox not found"))?;
        let metadata = sandbox
            .metadata
            .as_ref()
            .ok_or_else(|| Status::internal("sandbox identity metadata is missing"))?;
        let identity =
            crate::auth::sandbox_session::PersistedSandboxIdentity::read(&metadata.annotations)
                .map_err(|_| Status::internal("sandbox runtime identity is invalid"))?;
        if identity.runtime_generation.as_str() != bundle.runtime_generation.as_str()
            || identity.auth_epoch != bundle.auth_epoch
            || identity.auth_epoch.get() != assignment.credential_epoch
        {
            return Err(Status::failed_precondition(
                "driver bootstrap does not match the sandbox runtime launch",
            ));
        }
        let minted = self
            .authority
            .mint_persisted_launch(&assignment.sandbox_id, &identity)?;
        bundle.gateway_token = minted.supervisor.gateway_token;
        bundle.gateway_expires_at = minted.supervisor.gateway_expires_at;
        bundle.sandbox_token = minted.supervisor.sandbox_token;
        bundle.sandbox_expires_at = minted.supervisor.sandbox_expires_at;
        bundle.auth_epoch = minted.supervisor.auth_epoch;
        serde_json::to_vec(&bundle).map_err(|_| Status::internal("encode supervisor auth bundle"))
    }

    fn authority_claims(
        assignment: &Assignment,
        kind: SupervisorAuthorityKind,
        supervisor_instance: &str,
    ) -> Result<SupervisorAuthority, Status> {
        Ok(SupervisorAuthority {
            kind,
            sandbox_id: SandboxId::parse(assignment.sandbox_id.clone())
                .map_err(|_| Status::internal("assignment sandbox ID is invalid"))?,
            runtime_generation: SandboxGenerationId::parse(assignment.runtime_generation.clone())
                .map_err(|_| {
                Status::internal("assignment runtime generation is invalid")
            })?,
            ownership_epoch: assignment.ownership_epoch,
            supervisor_instance: supervisor_instance.to_string(),
            candidate_key_pem: None,
            challenge: None,
            receipt_digest: None,
        })
    }

    pub async fn request_grant(
        &self,
        member: &PoolMember,
        request: RequestSupervisorRecoveryGrantRequest,
    ) -> Result<RequestSupervisorRecoveryGrantResponse, Status> {
        if request.candidate_key_pem.is_empty() || request.challenge.is_empty() {
            return Err(Status::invalid_argument(
                "recovery grant requires a candidate key and runtime challenge",
            ));
        }
        let assignment = self.live_assignment(&request.sandbox_id).await?;
        let mut claims = Self::authority_claims(
            &assignment,
            SupervisorAuthorityKind::RecoveryGrant,
            &request.supervisor_instance,
        )?;
        claims.candidate_key_pem = Some(request.candidate_key_pem);
        claims.challenge = Some(request.challenge);
        let minted = self
            .authority
            .mint_supervisor_authority(&claims, GRANT_TTL)?;
        let next = assignment
            .issue_grant(
                &member.member_id,
                request.ownership_epoch,
                &minted.token_id.to_string(),
                now_ms(),
            )
            .map_err(assignment_status)?;
        self.assignments
            .write(Some(&assignment), &next)
            .await
            .map_err(store_status)?;
        info!(
            sandbox_id = %assignment.sandbox_id,
            ownership_epoch = assignment.ownership_epoch,
            member = %member.member_id,
            runtime_epoch = request.runtime_ownership_epoch,
            "Issued supervisor recovery grant"
        );
        Ok(RequestSupervisorRecoveryGrantResponse {
            grant: minted.token.expose_secret().to_string(),
            grant_id: minted.token_id.to_string(),
        })
    }

    pub async fn activate(
        &self,
        member: &PoolMember,
        request: ActivateSupervisorAssignmentRequest,
    ) -> Result<ActivateSupervisorAssignmentResponse, Status> {
        let assignment = self.live_assignment(&request.sandbox_id).await?;
        let bootstrap = self
            .backend
            .supervisor_bootstrap(&assignment.sandbox_id, &assignment.runtime_generation)
            .await?;
        let receipt_key = receipt_public_key(&bootstrap.runtime_descriptor_json)?;
        let receipt: RecoveryReceipt = verify_ed25519_document(
            receipt_key.as_bytes(),
            RECOVERY_RECEIPT_JWT_TYPE,
            &request.signed_receipt,
        )
        .map_err(|error| Status::permission_denied(format!("invalid fencing receipt: {error}")))?;
        if receipt.protocol != SUPERVISOR_RECOVERY_PROTOCOL
            || receipt.sandbox_id != assignment.sandbox_id
            || receipt.runtime_generation != assignment.runtime_generation
            || receipt.ownership_epoch != request.ownership_epoch
            || receipt.supervisor_instance != request.supervisor_instance
            || assignment.grant_id.as_deref() != Some(receipt.grant_id.as_str())
        {
            return Err(Status::permission_denied(
                "fencing receipt does not match the reserved recovery",
            ));
        }
        let next = assignment
            .activate(
                &member.member_id,
                request.ownership_epoch,
                &request.supervisor_instance,
                now_ms(),
                &self.timing,
            )
            .map_err(assignment_status)?;
        self.assignments
            .write(Some(&assignment), &next)
            .await
            .map_err(store_status)?;
        let mut claims = Self::authority_claims(
            &next,
            SupervisorAuthorityKind::Activation,
            &request.supervisor_instance,
        )?;
        claims.receipt_digest = Some(recovery_receipt_digest(&request.signed_receipt));
        let minted = self
            .authority
            .mint_supervisor_authority(&claims, GRANT_TTL)?;
        info!(
            sandbox_id = %next.sandbox_id,
            ownership_epoch = next.ownership_epoch,
            fenced_previous_epoch = receipt.fenced_previous_epoch,
            member = %member.member_id,
            "Activated supervisor recovery"
        );
        Ok(ActivateSupervisorAssignmentResponse {
            activation: minted.token.expose_secret().to_string(),
        })
    }

    pub async fn renew(
        &self,
        member: &PoolMember,
        request: RenewSupervisorLeaseRequest,
    ) -> Result<RenewSupervisorLeaseResponse, Status> {
        let assignment = self.live_assignment(&request.sandbox_id).await?;
        let next = assignment
            .renew(
                &member.member_id,
                request.ownership_epoch,
                &request.supervisor_instance,
                now_ms(),
                &self.timing,
            )
            .map_err(assignment_status)?;
        self.assignments
            .write(Some(&assignment), &next)
            .await
            .map_err(store_status)?;
        let claims = Self::authority_claims(
            &next,
            SupervisorAuthorityKind::Lease,
            &request.supervisor_instance,
        )?;
        let lease_ttl = Duration::from_millis(u64::try_from(self.timing.lease_ttl_ms).unwrap_or(0));
        let minted = self
            .authority
            .mint_supervisor_authority(&claims, lease_ttl)?;
        Ok(RenewSupervisorLeaseResponse {
            lease: minted.token.expose_secret().to_string(),
            lease_ttl: Some(prost_duration_ms(self.timing.lease_ttl_ms)),
        })
    }
}

pub(crate) fn prost_duration_ms(milliseconds: i64) -> prost_types::Duration {
    prost_types::Duration {
        seconds: milliseconds / 1000,
        nanos: i32::try_from((milliseconds % 1000) * 1_000_000).unwrap_or(0),
    }
}

/// Extract the runtime's receipt-verification key from the driver-written
/// runtime descriptor. Absence means the runtime cannot be recovered.
fn receipt_public_key(runtime_descriptor_json: &[u8]) -> Result<String, Status> {
    let descriptor: serde_json::Value = serde_json::from_slice(runtime_descriptor_json)
        .map_err(|_| Status::internal("runtime descriptor is invalid"))?;
    let recovery = descriptor.get("supervisor_recovery").ok_or_else(|| {
        Status::failed_precondition("runtime does not support supervisor recovery")
    })?;
    if recovery.get("protocol").and_then(serde_json::Value::as_str)
        != Some(SUPERVISOR_RECOVERY_PROTOCOL)
    {
        return Err(Status::failed_precondition(
            "runtime supervisor recovery protocol is unsupported",
        ));
    }
    recovery
        .get("receipt_public_key_pem")
        .and_then(serde_json::Value::as_str)
        .map(str::to_string)
        .ok_or_else(|| Status::failed_precondition("runtime receipt key is missing"))
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::auth::sandbox_session::PersistedSandboxIdentity;
    use openshell_core::jwt::{
        CredentialEpoch, RecoveryReceipt, SessionJwtVerifier, SessionTokenProfile,
        SessionVerificationKey, SystemJwtClock, sign_ed25519_document,
    };
    use openshell_core::proto::SandboxStatus;
    use openshell_core::proto::datamodel::v1::ObjectMeta;
    use rcgen::{KeyPair, PKCS_ED25519};

    #[derive(Debug)]
    struct FakeBackend {
        runtime_key: String,
        auth_bundle: Vec<u8>,
    }

    #[tonic::async_trait]
    impl SupervisorPoolBackend for FakeBackend {
        async fn authenticate_member(&self, credential: &str) -> Result<PoolMember, Status> {
            Ok(PoolMember {
                member_id: credential.to_string(),
                member_uid: format!("uid-{credential}"),
            })
        }

        async fn supervisor_bootstrap(
            &self,
            _sandbox_id: &str,
            _runtime_generation: &str,
        ) -> Result<SupervisorBootstrap, Status> {
            Ok(SupervisorBootstrap {
                runtime_descriptor_json: serde_json::to_vec(&serde_json::json!({
                    "supervisor_recovery": {
                        "protocol": SUPERVISOR_RECOVERY_PROTOCOL,
                        "receipt_public_key_pem": self.runtime_key,
                    }
                }))
                .unwrap(),
                auth_bundle_json: self.auth_bundle.clone(),
                main_process_spec_json: "{}".to_string(),
                proxy_ca_cert_pem: String::new(),
                proxy_ca_key_pem: String::new(),
                upstream_proxy_ca_pem: String::new(),
            })
        }
    }

    const FAST: AssignmentTiming = AssignmentTiming {
        lease_ttl_ms: 300,
        lease_interval_ms: 100,
        launch_grace_ms: 5_000,
        recovery_budget_ms: 2_000,
        attempt_budget_ms: 1_000,
        runtime_recovery_window_ms: 30_000,
        release_budget_ms: 1_000,
    };

    impl SupervisorPool {
        /// Run one coordinator pass as a test coordinator ("test-gateway"),
        /// acquiring the lease if it is free.
        async fn reconcile_once(
            &self,
        ) -> Result<coordinator::PassReport, coordinator::CoordinateError> {
            let now = now_ms();
            let lease = self
                .store
                .acquire_coordinator_lease(
                    coordinator::COORDINATOR_LEASE,
                    "test-gateway/1",
                    now,
                    coordinator::COORDINATOR_TTL_MS,
                )
                .await?
                .expect("test coordinator lease is free");
            let fence = crate::persistence::coordinator::CoordinatorFence {
                lease: coordinator::COORDINATOR_LEASE.to_string(),
                holder: lease.holder,
                term: lease.term,
            };
            self.coordinate_once(&fence, now).await
        }
    }

    async fn set_phase(store: &Store, phase: SandboxPhase) {
        let mut sandbox = store
            .get_message::<Sandbox>("sandbox-a")
            .await
            .unwrap()
            .unwrap();
        sandbox.status.get_or_insert_with(Default::default).phase = phase as i32;
        store.put_message(&sandbox).await.unwrap();
    }

    fn gateway_token_id(sandbox: &Sandbox) -> uuid::Uuid {
        PersistedSandboxIdentity::read(&sandbox.metadata.as_ref().unwrap().annotations)
            .unwrap()
            .gateway_token_id
    }

    #[tokio::test]
    async fn fenced_takeover_flow_through_the_gateway() {
        let store = Arc::new(crate::persistence::test_store().await);
        let gateway_key = KeyPair::generate_for(&PKCS_ED25519).unwrap();
        let authority = Arc::new(
            SandboxSessionJwtAuthority::from_pem(
                gateway_key.serialize_pem().as_bytes(),
                gateway_key.public_key_pem().as_bytes(),
                "k".to_string(),
                "gw",
                None,
            )
            .unwrap(),
        );
        let identity = PersistedSandboxIdentity {
            runtime_generation: SandboxGenerationId::parse("generation-a").unwrap(),
            auth_epoch: CredentialEpoch::new(1).unwrap(),
            gateway_token_id: uuid::Uuid::from_u128(1),
            refresh_replay: None,
        };
        let mut metadata = ObjectMeta {
            id: "sandbox-a".to_string(),
            name: "sandbox-a".to_string(),
            workspace: "default".to_string(),
            ..Default::default()
        };
        identity.write(&mut metadata.annotations);
        metadata.annotations.insert(
            crate::compute::COMPUTE_RUNTIME_IDENTITY_ANNOTATION.to_string(),
            "kubernetes://ns/sandbox/pool".to_string(),
        );
        store
            .put_message(&Sandbox {
                metadata: Some(metadata),
                status: Some(SandboxStatus {
                    phase: SandboxPhase::Ready as i32,
                    ..Default::default()
                }),
                ..Default::default()
            })
            .await
            .unwrap();
        let launch = authority
            .mint_persisted_launch("sandbox-a", &identity)
            .unwrap();
        let runtime_key = KeyPair::generate_for(&PKCS_ED25519).unwrap();
        let pool = SupervisorPool::new(
            store.clone(),
            Arc::new(FakeBackend {
                runtime_key: runtime_key.public_key_pem(),
                auth_bundle: serde_json::to_vec(&launch.supervisor).unwrap(),
            }),
            authority.clone(),
            crate::compute::new_test_runtime(store.clone()).await,
            FAST,
        )
        .unwrap();
        let verifier = SessionJwtVerifier::new(
            "gw",
            SessionTokenProfile::Sandbox,
            [SessionVerificationKey {
                key_id: "k".to_string(),
                public_key_pem: gateway_key.public_key_pem().into_bytes(),
            }],
            Arc::new(SystemJwtClock),
        )
        .unwrap();
        let replica_a = PoolMember {
            member_id: "replica-a".to_string(),
            member_uid: "uid-a".to_string(),
        };
        let replica_b = PoolMember {
            member_id: "replica-b".to_string(),
            member_uid: "uid-b".to_string(),
        };
        let sync = |member: PoolMember| {
            let pool = &pool;
            async move {
                pool.sync(&member, SyncSupervisorPoolRequest::default())
                    .await
                    .unwrap()
                    .assignments
            }
        };

        // Launch: both replicas are live; the sandbox goes to replica-a.
        sync(replica_a.clone()).await;
        sync(replica_b.clone()).await;
        pool.reconcile_once().await.unwrap();
        let assigned = sync(replica_a.clone()).await;
        assert_eq!(assigned.len(), 1);
        assert_eq!(
            assigned[0].action,
            SupervisorAssignmentAction::Launch as i32
        );
        assert!(sync(replica_b.clone()).await.is_empty());
        let material = pool
            .get_assignment(
                &replica_a,
                GetSupervisorAssignmentRequest {
                    sandbox_id: "sandbox-a".to_string(),
                    ownership_epoch: 1,
                },
            )
            .await
            .unwrap();
        assert!(!material.auth_bundle_json.is_empty());
        let lease = pool
            .renew(
                &replica_a,
                RenewSupervisorLeaseRequest {
                    sandbox_id: "sandbox-a".to_string(),
                    ownership_epoch: 1,
                    supervisor_instance: "instance-a".to_string(),
                },
            )
            .await
            .unwrap();
        let lease = verifier.verify_supervisor_authority(&lease.lease).unwrap();
        assert_eq!(lease.authority.kind, SupervisorAuthorityKind::Lease);
        assert_eq!(lease.authority.ownership_epoch, 1);

        // replica-a stops renewing (crash or partition). After the lease
        // lapses, a reconciler reserves epoch 2 for replica-b and rotates the
        // gateway token lineage so replica-a's gateway credential dies.
        let before = gateway_token_id(
            &store
                .get_message::<Sandbox>("sandbox-a")
                .await
                .unwrap()
                .unwrap(),
        );
        tokio::time::sleep(Duration::from_millis(350)).await;
        sync(replica_a.clone()).await;
        sync(replica_b.clone()).await;
        pool.reconcile_once().await.unwrap();
        let after = gateway_token_id(
            &store
                .get_message::<Sandbox>("sandbox-a")
                .await
                .unwrap()
                .unwrap(),
        );
        assert_ne!(
            before, after,
            "recovery must revoke the old gateway token lineage"
        );
        let assigned = sync(replica_b.clone()).await;
        assert_eq!(
            assigned[0].action,
            SupervisorAssignmentAction::Recover as i32
        );
        assert_eq!(assigned[0].ownership_epoch, 2);
        assert!(
            sync(replica_a.clone()).await.is_empty(),
            "the stale owner is no longer assigned and must abandon its session"
        );

        // The returning stale owner cannot renew or obtain recovery authority.
        assert!(
            pool.renew(
                &replica_a,
                RenewSupervisorLeaseRequest {
                    sandbox_id: "sandbox-a".to_string(),
                    ownership_epoch: 1,
                    supervisor_instance: "instance-a".to_string(),
                },
            )
            .await
            .is_err()
        );
        let grant_request = |epoch: u64| RequestSupervisorRecoveryGrantRequest {
            sandbox_id: "sandbox-a".to_string(),
            ownership_epoch: epoch,
            supervisor_instance: "instance-b".to_string(),
            candidate_key_pem: "candidate-key".to_string(),
            challenge: "runtime-challenge".to_string(),
            runtime_ownership_epoch: 1,
        };
        assert!(
            pool.request_grant(&replica_a, grant_request(2))
                .await
                .is_err()
        );
        let grant = pool
            .request_grant(&replica_b, grant_request(2))
            .await
            .unwrap();
        let verified_grant = verifier.verify_supervisor_authority(&grant.grant).unwrap();
        assert_eq!(verified_grant.authority.ownership_epoch, 2);
        assert_eq!(
            verified_grant.authority.challenge.as_deref(),
            Some("runtime-challenge")
        );
        assert!(
            pool.request_grant(&replica_b, grant_request(2))
                .await
                .is_err(),
            "one grant per reserved epoch"
        );

        // Activation requires a receipt signed by the runtime key for exactly
        // the issued grant.
        let receipt = |key: &KeyPair, grant_id: &str| {
            sign_ed25519_document(
                key.serialize_pem().as_bytes(),
                RECOVERY_RECEIPT_JWT_TYPE,
                &RecoveryReceipt {
                    protocol: SUPERVISOR_RECOVERY_PROTOCOL.to_string(),
                    sandbox_id: "sandbox-a".to_string(),
                    runtime_generation: "generation-a".to_string(),
                    ownership_epoch: 2,
                    supervisor_instance: "instance-b".to_string(),
                    challenge: "runtime-challenge".to_string(),
                    grant_id: grant_id.to_string(),
                    fenced_previous_epoch: 1,
                    main_process_id: Some("generation-a:main:0".to_string()),
                    issued_at_unix_ms: 0,
                },
            )
            .unwrap()
        };
        let activate = |signed_receipt: String| ActivateSupervisorAssignmentRequest {
            sandbox_id: "sandbox-a".to_string(),
            ownership_epoch: 2,
            supervisor_instance: "instance-b".to_string(),
            signed_receipt,
        };
        let forged_key = KeyPair::generate_for(&PKCS_ED25519).unwrap();
        assert!(
            pool.activate(&replica_b, activate(receipt(&forged_key, &grant.grant_id)))
                .await
                .is_err(),
            "a receipt not signed by the runtime is rejected"
        );
        assert!(
            pool.activate(&replica_b, activate(receipt(&runtime_key, "other-grant")))
                .await
                .is_err(),
            "a receipt for another grant is rejected"
        );
        let signed = receipt(&runtime_key, &grant.grant_id);
        let activation = pool
            .activate(&replica_b, activate(signed.clone()))
            .await
            .unwrap();
        let activation = verifier
            .verify_supervisor_authority(&activation.activation)
            .unwrap();
        assert_eq!(
            activation.authority.kind,
            SupervisorAuthorityKind::Activation
        );
        assert_eq!(
            activation.authority.receipt_digest,
            Some(recovery_receipt_digest(&signed))
        );
        assert_eq!(
            sync(replica_b.clone()).await[0].action,
            SupervisorAssignmentAction::Supervise as i32
        );

        // Delete wins: once the sandbox leaves its running phases, the owner
        // can no longer renew and the assignment is terminated.
        set_phase(&store, SandboxPhase::Deleting).await;
        assert!(
            pool.renew(
                &replica_b,
                RenewSupervisorLeaseRequest {
                    sandbox_id: "sandbox-a".to_string(),
                    ownership_epoch: 2,
                    supervisor_instance: "instance-b".to_string(),
                },
            )
            .await
            .is_err()
        );
        let released = pool
            .sync(
                &replica_b,
                SyncSupervisorPoolRequest {
                    active_sandbox_ids: vec!["sandbox-a".to_string()],
                },
            )
            .await
            .unwrap()
            .assignments;
        assert_eq!(
            released[0].action,
            SupervisorAssignmentAction::Release as i32
        );
    }

    #[tokio::test]
    async fn restarting_a_sandbox_relaunches_under_a_new_credential_epoch() {
        let store = Arc::new(crate::persistence::test_store().await);
        let gateway_key = KeyPair::generate_for(&PKCS_ED25519).unwrap();
        let authority = Arc::new(
            SandboxSessionJwtAuthority::from_pem(
                gateway_key.serialize_pem().as_bytes(),
                gateway_key.public_key_pem().as_bytes(),
                "k".to_string(),
                "gw",
                None,
            )
            .unwrap(),
        );
        let write_sandbox = |epoch: u64, phase: SandboxPhase| {
            let store = store.clone();
            async move {
                let identity = PersistedSandboxIdentity {
                    runtime_generation: SandboxGenerationId::parse("generation-a").unwrap(),
                    auth_epoch: CredentialEpoch::new(epoch).unwrap(),
                    gateway_token_id: uuid::Uuid::from_u128(u128::from(epoch)),
                    refresh_replay: None,
                };
                let mut metadata = ObjectMeta {
                    id: "sandbox-a".to_string(),
                    name: "sandbox-a".to_string(),
                    workspace: "default".to_string(),
                    ..Default::default()
                };
                identity.write(&mut metadata.annotations);
                metadata.annotations.insert(
                    crate::compute::COMPUTE_RUNTIME_IDENTITY_ANNOTATION.to_string(),
                    "kubernetes://ns/sandbox/pool".to_string(),
                );
                store
                    .put_message(&Sandbox {
                        metadata: Some(metadata),
                        status: Some(SandboxStatus {
                            phase: phase as i32,
                            ..Default::default()
                        }),
                        ..Default::default()
                    })
                    .await
                    .unwrap();
            }
        };
        write_sandbox(1, SandboxPhase::Ready).await;
        let pool = SupervisorPool::new(
            store.clone(),
            Arc::new(FakeBackend {
                runtime_key: String::new(),
                auth_bundle: Vec::new(),
            }),
            authority,
            crate::compute::new_test_runtime(store.clone()).await,
            FAST,
        )
        .unwrap();
        let member = PoolMember {
            member_id: "replica-a".to_string(),
            member_uid: "uid-a".to_string(),
        };
        let assigned = || async {
            pool.sync(&member, SyncSupervisorPoolRequest::default())
                .await
                .unwrap()
                .assignments
        };
        assigned().await;
        pool.reconcile_once().await.unwrap();
        let first = assigned().await;
        assert_eq!(first[0].credential_epoch, 1);

        // Stop terminates the launch.
        write_sandbox(1, SandboxPhase::Stopped).await;
        pool.reconcile_once().await.unwrap();
        assert!(assigned().await.is_empty());

        // Start keeps the generation but advances the credential epoch: a
        // fresh launch must be assigned instead of staying terminated.
        write_sandbox(2, SandboxPhase::Starting).await;
        pool.reconcile_once().await.unwrap();
        pool.reconcile_once().await.unwrap();
        let relaunched = assigned().await;
        assert_eq!(relaunched.len(), 1);
        assert_eq!(relaunched[0].credential_epoch, 2);
        assert_eq!(relaunched[0].ownership_epoch, 1);
        assert_eq!(
            relaunched[0].action,
            SupervisorAssignmentAction::Launch as i32
        );
    }

    #[tokio::test]
    async fn missed_recovery_deadline_fails_closed() {
        let store = Arc::new(crate::persistence::test_store().await);
        let assignments = AssignmentStore::new(store);
        let now = now_ms();
        let pending = assignments
            .write(None, &Assignment::pending("sb", "sb", "gen"))
            .await
            .unwrap();
        let active = pending
            .assign_launch("replica-a", now - 10_000, &FAST)
            .unwrap()
            .renew("replica-a", 1, "instance-a", now - 10_000, &FAST)
            .unwrap();
        let reserved = active
            .reserve_recovery("replica-b", now - 9_000, &FAST)
            .unwrap();
        let failed = reserved
            .fail_if_overdue(now, &FAST)
            .expect("deadline passed");
        assert_eq!(failed.state, AssignmentState::Failed);
        assert!(failed.failure_reason.unwrap().contains("deadline"));
    }

    #[test]
    fn receipt_key_requires_the_versioned_extension() {
        assert!(receipt_public_key(br#"{"boundary_id":"a"}"#).is_err());
        assert!(
            receipt_public_key(
                br#"{"supervisor_recovery":{"protocol":"other","receipt_public_key_pem":"k"}}"#
            )
            .is_err()
        );
        assert_eq!(
            receipt_public_key(
                br#"{"supervisor_recovery":{"protocol":"supervisor-recovery.v1","receipt_public_key_pem":"k"}}"#
            )
            .unwrap(),
            "k"
        );
    }

    #[test]
    fn only_running_phases_want_supervision() {
        assert!(wants_supervision(SandboxPhase::Ready));
        assert!(wants_supervision(SandboxPhase::Provisioning));
        for phase in [
            SandboxPhase::Stopping,
            SandboxPhase::Stopped,
            SandboxPhase::Deleting,
            SandboxPhase::Completed,
            SandboxPhase::Error,
        ] {
            assert!(!wants_supervision(phase), "{phase:?}");
        }
    }
}
