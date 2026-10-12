// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Test harness: several gateway processes (`SupervisorPool` instances) on
//! one store, pool members that sync, and a runtime key that signs fencing
//! receipts, so coordinator behavior is exercised through the real RPC
//! handlers and fenced store writes.

use std::sync::Arc;

use openshell_core::jwt::{
    CredentialEpoch, RECOVERY_RECEIPT_JWT_TYPE, RecoveryReceipt, SUPERVISOR_RECOVERY_PROTOCOL,
    sign_ed25519_document,
};
use openshell_core::proto::datamodel::v1::ObjectMeta;
use openshell_core::proto::{
    ActivateSupervisorAssignmentRequest, RenewSupervisorLeaseRequest,
    RequestSupervisorRecoveryGrantRequest, Sandbox, SandboxPhase, SandboxStatus,
    SupervisorAssignmentAction, SyncSupervisorPoolRequest,
};
use openshell_core::sandbox_generation::SandboxGenerationId;
use rcgen::{KeyPair, PKCS_ED25519};
use tonic::Status;

use super::coordinator::{COORDINATOR_LEASE, COORDINATOR_TTL_MS, CoordinateError, PassReport};
use super::{
    AssignmentTiming, PoolMember, SupervisorBootstrap, SupervisorPool, SupervisorPoolBackend,
};
use crate::auth::sandbox_jwt::SandboxSessionJwtAuthority;
use crate::auth::sandbox_session::PersistedSandboxIdentity;
use crate::persistence::Store;
use crate::persistence::coordinator::CoordinatorFence;

#[derive(Debug)]
pub struct FakeBackend {
    pub runtime_key: String,
}

#[tonic::async_trait]
impl SupervisorPoolBackend for FakeBackend {
    async fn authenticate_member(&self, credential: &str) -> Result<PoolMember, Status> {
        Ok(member(credential))
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
            auth_bundle_json: Vec::new(),
            main_process_spec_json: "{}".to_string(),
            proxy_ca_cert_pem: String::new(),
            proxy_ca_key_pem: String::new(),
            upstream_proxy_ca_pem: String::new(),
        })
    }
}

/// Timing for fast tests: short owner leases; budgets as validated.
pub const FAST: AssignmentTiming = AssignmentTiming {
    lease_ttl_ms: 300,
    lease_interval_ms: 100,
    launch_grace_ms: 5_000,
    recovery_budget_ms: 2_000,
    attempt_budget_ms: 1_000,
    runtime_recovery_window_ms: 30_000,
    release_budget_ms: 1_000,
};

#[must_use]
pub fn member(name: &str) -> PoolMember {
    PoolMember {
        member_id: name.to_string(),
        member_uid: format!("uid-{name}"),
    }
}

pub struct Harness {
    pub store: Arc<Store>,
    pub authority: Arc<SandboxSessionJwtAuthority>,
    runtime_key: KeyPair,
    timing: AssignmentTiming,
}

impl Harness {
    pub async fn new(timing: AssignmentTiming) -> Self {
        Self::with_store(Arc::new(crate::persistence::test_store().await), timing)
    }

    pub fn with_store(store: Arc<Store>, timing: AssignmentTiming) -> Self {
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
        Self {
            store,
            authority,
            runtime_key: KeyPair::generate_for(&PKCS_ED25519).unwrap(),
            timing,
        }
    }

    /// One gateway process on the shared store.
    pub async fn gateway(&self) -> Arc<SupervisorPool> {
        Arc::new(
            SupervisorPool::new(
                self.store.clone(),
                Arc::new(FakeBackend {
                    runtime_key: self.runtime_key.public_key_pem(),
                }),
                self.authority.clone(),
                crate::compute::new_test_runtime(self.store.clone()).await,
                self.timing,
            )
            .unwrap(),
        )
    }

    pub async fn put_sandbox(&self, id: &str, credential_epoch: u64, phase: SandboxPhase) {
        let identity = PersistedSandboxIdentity {
            runtime_generation: SandboxGenerationId::parse(format!("generation-{id}")).unwrap(),
            auth_epoch: CredentialEpoch::new(credential_epoch).unwrap(),
            gateway_token_id: uuid::Uuid::new_v4(),
            refresh_replay: None,
        };
        let mut metadata = ObjectMeta {
            id: id.to_string(),
            name: id.to_string(),
            workspace: "default".to_string(),
            ..Default::default()
        };
        identity.write(&mut metadata.annotations);
        metadata.annotations.insert(
            crate::compute::COMPUTE_RUNTIME_IDENTITY_ANNOTATION.to_string(),
            "kubernetes://ns/sandbox/pool".to_string(),
        );
        self.store
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

    pub async fn set_phase(&self, id: &str, phase: SandboxPhase) {
        let mut sandbox = self
            .store
            .get_message::<Sandbox>(id)
            .await
            .unwrap()
            .unwrap();
        sandbox.status.get_or_insert_with(Default::default).phase = phase as i32;
        self.store.put_message(&sandbox).await.unwrap();
    }

    /// Acquire the coordinator lease for `holder` at `now`, if free.
    pub async fn elect(&self, holder: &str, now: i64) -> Option<CoordinatorFence> {
        self.store
            .acquire_coordinator_lease(COORDINATOR_LEASE, holder, now, COORDINATOR_TTL_MS)
            .await
            .unwrap()
            .map(|lease| CoordinatorFence {
                lease: COORDINATOR_LEASE.to_string(),
                holder: lease.holder,
                term: lease.term,
            })
    }

    /// Sign the runtime's fencing receipt for a reserved epoch.
    pub fn receipt(&self, sandbox_id: &str, epoch: u64, instance: &str, grant_id: &str) -> String {
        sign_ed25519_document(
            self.runtime_key.serialize_pem().as_bytes(),
            RECOVERY_RECEIPT_JWT_TYPE,
            &RecoveryReceipt {
                protocol: SUPERVISOR_RECOVERY_PROTOCOL.to_string(),
                sandbox_id: sandbox_id.to_string(),
                runtime_generation: format!("generation-{sandbox_id}"),
                ownership_epoch: epoch,
                supervisor_instance: instance.to_string(),
                challenge: "challenge".to_string(),
                grant_id: grant_id.to_string(),
                fenced_previous_epoch: epoch - 1,
                main_process_id: Some(format!("generation-{sandbox_id}:main:0")),
                issued_at_unix_ms: 0,
            },
        )
        .unwrap()
    }
}

/// Sync `member` reporting `hosted` sessions; returns (sandbox, action, epoch).
pub async fn sync(
    gateway: &SupervisorPool,
    member_name: &str,
    hosted: &[&str],
) -> Vec<(String, SupervisorAssignmentAction, u64)> {
    gateway
        .sync(
            &member(member_name),
            SyncSupervisorPoolRequest {
                active_sandbox_ids: hosted.iter().map(|s| (*s).to_string()).collect(),
            },
        )
        .await
        .unwrap()
        .assignments
        .into_iter()
        .map(|a| {
            (
                a.sandbox_id,
                SupervisorAssignmentAction::try_from(a.action).unwrap(),
                a.ownership_epoch,
            )
        })
        .collect()
}

pub async fn renew(
    gateway: &SupervisorPool,
    member_name: &str,
    sandbox_id: &str,
    epoch: u64,
) -> Result<(), Status> {
    gateway
        .renew(
            &member(member_name),
            RenewSupervisorLeaseRequest {
                sandbox_id: sandbox_id.to_string(),
                ownership_epoch: epoch,
                supervisor_instance: format!("instance-{member_name}"),
            },
        )
        .await
        .map(|_| ())
}

/// Run the candidate side of the recovery protocol through `gateway`: grant,
/// runtime receipt, activation.
pub async fn take_over(
    harness: &Harness,
    gateway: &SupervisorPool,
    member_name: &str,
    sandbox_id: &str,
    epoch: u64,
) -> Result<(), Status> {
    let instance = format!("instance-{member_name}");
    let grant = gateway
        .request_grant(
            &member(member_name),
            RequestSupervisorRecoveryGrantRequest {
                sandbox_id: sandbox_id.to_string(),
                ownership_epoch: epoch,
                supervisor_instance: instance.clone(),
                candidate_key_pem: "candidate-key".to_string(),
                challenge: "challenge".to_string(),
                runtime_ownership_epoch: epoch - 1,
            },
        )
        .await?;
    gateway
        .activate(
            &member(member_name),
            ActivateSupervisorAssignmentRequest {
                sandbox_id: sandbox_id.to_string(),
                ownership_epoch: epoch,
                supervisor_instance: instance.clone(),
                signed_receipt: harness.receipt(sandbox_id, epoch, &instance, &grant.grant_id),
            },
        )
        .await
        .map(|_| ())
}

pub async fn pass(
    gateway: &SupervisorPool,
    fence: &CoordinatorFence,
) -> Result<PassReport, CoordinateError> {
    gateway
        .coordinate_once(fence, openshell_core::time::now_ms())
        .await
}
