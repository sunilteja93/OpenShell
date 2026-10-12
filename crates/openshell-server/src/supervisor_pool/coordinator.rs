// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The elected supervisor pool assignment coordinator.
//!
//! Exactly one gateway process holds the coordinator lease (see
//! [`crate::persistence::coordinator`]). Only that process places sandboxes,
//! schedules failed-owner recovery, fails overdue work, and drives drain and
//! rebalance handoffs. Followers run no reconciliation at all; they keep
//! serving pool sync, assignment material, grants, activations, and owner
//! lease renewals, so a coordinator change only pauses new placement and
//! handoffs for at most one coordinator TTL and never freezes a healthy agent.
//!
//! Every coordinator decision is one fenced batch: the assignment change, the
//! durable operation record that describes it, and (for a reservation) the
//! gateway-token rotation that revokes the previous owner. The batch commits
//! only while the coordinator still holds an unexpired term, so a coordinator
//! that paused mid-decision cannot apply stale work after a takeover.
//! Decisions are level-triggered from durable state, so a new coordinator
//! adopts pending operations idempotently: deterministic operation IDs, no
//! second launch for a Launching assignment, and recovery deadlines copied
//! from the assignment, never extended.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::Arc;
use std::time::Duration;

use openshell_core::proto::{Sandbox, SandboxPhase};
use openshell_core::time::now_ms;
use openshell_core::{GetResourceVersion as _, ObjectId as _, ObjectName as _};
use prost::Message as _;
use tonic::Status;
use tracing::{debug, info, warn};

use super::assignment::{Assignment, AssignmentState};
use super::operations::{MoveState, OperationKind, OperationPhase, PoolOperation, SessionMove};
use super::{
    MemberView, SupervisorPool, has_runtime_binding, sandbox_credential_epoch, sandbox_generation,
    wants_supervision,
};
use crate::persistence::coordinator::{CoordinatorFence, FencedOp};
use crate::persistence::{
    ObjectLabels as _, ObjectType as _, ObjectWorkspace as _, PersistenceError,
};

/// Name of the single coordinator lease of an installation.
pub const COORDINATOR_LEASE: &str = "supervisor-pool-coordinator";
/// Coordinator lease TTL. Kept well inside the recovery budget so a
/// coordinator change cannot by itself consume a lost owner's recovery window.
pub const COORDINATOR_TTL_MS: i64 = 5_000;
/// How often the coordinator renews its lease and runs one pass, and how
/// often a follower tries to acquire the lease.
pub const COORDINATOR_TICK: Duration = Duration::from_secs(1);
/// Terminal assign/recover operations are kept this long for status.
const SHORT_RETENTION_MS: i64 = 10 * 60_000;
/// Terminal drain/rebalance operations are kept this long for status.
const LONG_RETENTION_MS: i64 = 60 * 60_000;

/// Unique identity of this gateway process for coordinator election. A
/// restarted process with the same Pod name is a different holder.
#[must_use]
pub fn process_holder_id(replica_id: &str) -> String {
    format!("{replica_id}/{}", uuid::Uuid::new_v4())
}

#[derive(Debug, thiserror::Error)]
pub enum CoordinateError {
    #[error("supervisor pool coordinator lease was lost")]
    Fenced,
    #[error("supervisor pool coordinator pass failed: {0}")]
    Store(String),
}

impl From<PersistenceError> for CoordinateError {
    fn from(error: PersistenceError) -> Self {
        match error {
            PersistenceError::Fenced => Self::Fenced,
            other => Self::Store(other.to_string()),
        }
    }
}

impl From<Status> for CoordinateError {
    fn from(status: Status) -> Self {
        Self::Store(status.message().to_string())
    }
}

/// What one coordinator pass did.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct PassReport {
    /// Fenced batches committed.
    pub committed: usize,
    /// Batches skipped because a non-coordinator write (renewal, grant,
    /// activation, cancel) changed their target first; retried next pass.
    pub conflicts: usize,
}

/// One fenced decision: an assignment change, the operations it touches, an
/// optional sandbox write (gateway-token rotation), and deletes.
#[derive(Default)]
struct Batch {
    assignment: Option<(Option<Assignment>, Assignment)>,
    operations: Vec<(Option<PoolOperation>, PoolOperation)>,
    sandbox: Option<FencedOp>,
    deletes: Vec<FencedOp>,
    /// Compute side effect to apply after the commit: (sandbox, generation,
    /// credential epoch, reason) of a failed recovery.
    recovery_failed: Option<(String, String, u64, String)>,
}

impl Batch {
    fn is_empty(&self) -> bool {
        self.assignment.is_none()
            && self.operations.is_empty()
            && self.sandbox.is_none()
            && self.deletes.is_empty()
    }
}

/// Durable state observed at the start of one pass, updated in memory as
/// batches commit.
struct Pass<'a> {
    pool: &'a SupervisorPool,
    fence: &'a CoordinatorFence,
    now: i64,
    members: Vec<MemberView>,
    cordoned: BTreeSet<String>,
    sandboxes: HashMap<String, Sandbox>,
    assignments: BTreeMap<String, Assignment>,
    operations: BTreeMap<String, PoolOperation>,
    report: PassReport,
}

fn operation_changed(expected: Option<&PoolOperation>, next: &PoolOperation) -> bool {
    let Some(expected) = expected else {
        return true;
    };
    let mut a = expected.clone();
    let mut b = next.clone();
    a.updated_at_ms = 0;
    b.updated_at_ms = 0;
    a.coordinator_term = 0;
    b.coordinator_term = 0;
    a != b
}

impl Pass<'_> {
    fn member(&self, member_id: &str) -> Option<&MemberView> {
        self.members.iter().find(|m| m.member_id == member_id)
    }

    fn is_live(&self, member_id: &str) -> bool {
        self.member(member_id).is_some_and(|m| m.live)
    }

    /// Live and not cordoned by a drain: may receive new placements,
    /// recoveries, and planned handoffs.
    fn is_eligible(&self, member_id: &str) -> bool {
        self.is_live(member_id) && !self.cordoned.contains(member_id)
    }

    /// Projected sessions per eligible member: owners of occupying
    /// assignments, with pending handoffs already counted at their target.
    fn load(&self) -> BTreeMap<String, usize> {
        let mut load: BTreeMap<String, usize> = self
            .members
            .iter()
            .filter(|m| self.is_eligible(&m.member_id))
            .map(|m| (m.member_id.clone(), 0))
            .collect();
        for assignment in self.assignments.values() {
            if !assignment.occupies_owner() {
                continue;
            }
            let owner = assignment
                .handoff
                .as_ref()
                .map(|handoff| handoff.target_member.as_str())
                .or(assignment.owner_member.as_deref());
            if let Some(count) = owner.and_then(|owner| load.get_mut(owner)) {
                *count += 1;
            }
        }
        load
    }

    fn least_loaded(&self, exclude: &[&str]) -> Option<String> {
        self.load()
            .into_iter()
            .filter(|(member, _)| !exclude.contains(&member.as_str()))
            .min_by(|(a_member, a), (b_member, b)| a.cmp(b).then(a_member.cmp(b_member)))
            .map(|(member, _)| member)
    }

    fn operation(&self, id: Option<&str>) -> Option<PoolOperation> {
        id.and_then(|id| self.operations.get(id)).cloned()
    }

    /// Commit one batch fenced by the coordinator term. Returns `false` on a
    /// CAS conflict (skipped, retried next pass) and `Err(Fenced)` once the
    /// term is lost.
    async fn commit(&mut self, mut batch: Batch) -> Result<bool, CoordinateError> {
        if batch.is_empty() {
            return Ok(true);
        }
        for (_, operation) in &mut batch.operations {
            operation.coordinator_term = self.fence.term;
        }
        let mut ops = Vec::new();
        if let Some((expected, next)) = &batch.assignment {
            ops.push(next.fenced_write(expected.as_ref())?);
        }
        for (expected, next) in &batch.operations {
            ops.push(next.fenced_write(expected.as_ref())?);
        }
        if let Some(sandbox) = batch.sandbox.take() {
            ops.push(sandbox);
        }
        ops.extend(batch.deletes.iter().cloned());
        let results = match self
            .pool
            .store
            .write_fenced(self.fence, self.now, &ops)
            .await
        {
            Ok(results) => results,
            Err(PersistenceError::Conflict { .. }) => {
                self.report.conflicts += 1;
                return Ok(false);
            }
            Err(error) => return Err(error.into()),
        };
        self.report.committed += 1;
        let mut results = results.into_iter();
        if let Some((_, mut next)) = batch.assignment {
            if let Some(Some(written)) = results.next() {
                next.resource_version = written.resource_version;
            }
            self.assignments.insert(next.sandbox_id.clone(), next);
        }
        for (_, mut next) in batch.operations {
            if let Some(Some(written)) = results.next() {
                next.resource_version = written.resource_version;
            }
            self.operations.insert(next.id.clone(), next);
        }
        for op in &batch.deletes {
            if let FencedOp::Delete {
                object_type, id, ..
            } = op
            {
                if object_type == super::assignment::ASSIGNMENT_OBJECT_TYPE {
                    self.assignments.retain(|_, a| {
                        super::assignment::assignment_object_id(&a.sandbox_id) != *id
                    });
                } else {
                    self.operations.remove(id);
                }
            }
        }
        if let Some((sandbox_id, generation, credential_epoch, reason)) = batch.recovery_failed {
            // Delayed side effect, fenced against a newer launch: applied with
            // a sandbox CAS only if the sandbox still runs the generation and
            // credential epoch whose recovery failed.
            self.pool
                .compute
                .mark_supervisor_recovery_failed_for_launch(
                    &sandbox_id,
                    &generation,
                    credential_epoch,
                    &reason,
                )
                .await;
        }
        Ok(true)
    }

    /// Stage `next` for `operation` if it changed.
    fn stage_operation(&self, batch: &mut Batch, mut next: PoolOperation) {
        if next.sandbox_name.is_none()
            && let Some(assignment) = next
                .sandbox_id
                .as_ref()
                .and_then(|id| self.assignments.get(id))
        {
            next.sandbox_name = Some(assignment.sandbox_name.clone());
        }
        let expected = self.operations.get(&next.id).cloned();
        if operation_changed(expected.as_ref(), &next) {
            next.updated_at_ms = self.now;
            batch.operations.push((expected, next));
        }
    }
}

/// Build the gateway-token rotation for `sandbox` as a fenced CAS write, so
/// it commits atomically with the reservation that revokes the old owner.
fn rotation_op(sandbox: &Sandbox) -> Result<FencedOp, CoordinateError> {
    let mut next = sandbox.clone();
    let metadata = next
        .metadata
        .as_mut()
        .ok_or_else(|| CoordinateError::Store("sandbox identity metadata is missing".into()))?;
    let mut identity =
        crate::auth::sandbox_session::PersistedSandboxIdentity::read(&metadata.annotations)
            .map_err(|_| CoordinateError::Store("sandbox runtime identity is invalid".into()))?;
    identity.gateway_token_id = uuid::Uuid::new_v4();
    identity.refresh_replay = None;
    identity.write(&mut metadata.annotations);
    let labels = next.object_labels();
    let labels = if labels.as_ref().is_none_or(HashMap::is_empty) {
        None
    } else {
        Some(
            serde_json::to_string(&labels)
                .map_err(|error| CoordinateError::Store(error.to_string()))?,
        )
    };
    Ok(FencedOp::Put {
        object_type: Sandbox::object_type().to_string(),
        id: next.object_id().to_string(),
        name: next.object_name().to_string(),
        workspace: next.object_workspace().to_string(),
        payload: next.encode_to_vec(),
        labels,
        condition: crate::persistence::WriteCondition::MatchResourceVersion(
            sandbox.get_resource_version(),
        ),
    })
}

impl SupervisorPool {
    /// Run the coordinator election loop for this gateway process until
    /// shutdown, releasing the lease on the way out.
    pub fn spawn_coordinator(
        self: &Arc<Self>,
        holder: String,
        mut shutdown: tokio::sync::watch::Receiver<bool>,
    ) {
        let pool = self.clone();
        tokio::spawn(async move {
            let mut held: Option<CoordinatorFence> = None;
            let mut interval = tokio::time::interval(COORDINATOR_TICK);
            interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
            loop {
                tokio::select! {
                    _ = interval.tick() => {}
                    _ = shutdown.changed() => break,
                }
                held = pool.coordinator_tick(&holder, held).await;
            }
            if let Some(fence) = held {
                match pool.store.release_coordinator_lease(&fence, now_ms()).await {
                    Ok(_) => info!(
                        term = fence.term,
                        "Released supervisor pool coordinator lease"
                    ),
                    Err(error) => {
                        warn!(%error, "Failed to release supervisor pool coordinator lease");
                    }
                }
            }
        });
    }

    /// One election step plus, while leading, one coordination pass.
    pub(crate) async fn coordinator_tick(
        &self,
        holder: &str,
        held: Option<CoordinatorFence>,
    ) -> Option<CoordinatorFence> {
        let now = now_ms();
        let held = match held {
            Some(fence) => match self
                .store
                .renew_coordinator_lease(&fence, now, COORDINATOR_TTL_MS)
                .await
            {
                Ok(true) => Some(fence),
                Ok(false) => {
                    warn!(
                        term = fence.term,
                        "Supervisor pool coordinator lease lost; stepping down"
                    );
                    None
                }
                Err(error) => {
                    // Keep the fence: writes stay guarded by the stored term
                    // and expiry, so a store blip cannot cause split brain.
                    warn!(%error, "Supervisor pool coordinator lease renewal failed");
                    Some(fence)
                }
            },
            None => match self
                .store
                .acquire_coordinator_lease(COORDINATOR_LEASE, holder, now, COORDINATOR_TTL_MS)
                .await
            {
                Ok(Some(lease)) => {
                    info!(
                        holder,
                        term = lease.term,
                        "Elected supervisor pool assignment coordinator"
                    );
                    Some(CoordinatorFence {
                        lease: COORDINATOR_LEASE.to_string(),
                        holder: holder.to_string(),
                        term: lease.term,
                    })
                }
                Ok(None) => None,
                Err(error) => {
                    debug!(%error, "Supervisor pool coordinator election failed");
                    None
                }
            },
        };
        let fence = held?;
        match self.coordinate_once(&fence, now_ms()).await {
            Ok(_) => Some(fence),
            Err(CoordinateError::Fenced) => {
                warn!(
                    term = fence.term,
                    "Supervisor pool coordinator was fenced; stepping down"
                );
                None
            }
            Err(error) => {
                warn!(%error, "Supervisor pool coordination pass failed");
                Some(fence)
            }
        }
    }

    /// One coordination pass under `fence`. Public within the crate for
    /// tests that drive explicit coordinator terms.
    pub(crate) async fn coordinate_once(
        &self,
        fence: &CoordinatorFence,
        now: i64,
    ) -> Result<PassReport, CoordinateError> {
        // The pass keeps a snapshot of the pool state; box it so callers'
        // futures stay small.
        Box::pin(self.coordinate_pass(fence, now)).await
    }

    async fn coordinate_pass(
        &self,
        fence: &CoordinatorFence,
        now: i64,
    ) -> Result<PassReport, CoordinateError> {
        let members = self.members(now).await?;
        let sandboxes: HashMap<String, Sandbox> = self
            .store
            .list_all_messages::<Sandbox>(10_000, 0)
            .await
            .map_err(|error| CoordinateError::Store(format!("list sandboxes: {error}")))?
            .into_iter()
            .map(|sandbox| (sandbox.object_id().to_string(), sandbox))
            .collect();
        let assignments: BTreeMap<String, Assignment> = self
            .assignments
            .list()
            .await?
            .into_iter()
            .map(|assignment| (assignment.sandbox_id.clone(), assignment))
            .collect();
        let operations: BTreeMap<String, PoolOperation> = self
            .operations
            .list()
            .await?
            .into_iter()
            .map(|operation| (operation.id.clone(), operation))
            .collect();
        let cordoned = operations
            .values()
            .filter_map(PoolOperation::cordoned_member)
            .map(str::to_string)
            .collect();
        let mut pass = Pass {
            pool: self,
            fence,
            now,
            members,
            cordoned,
            sandboxes,
            assignments,
            operations,
            report: PassReport::default(),
        };
        pass.ensure_desired_assignments().await?;
        let ids: Vec<String> = pass.assignments.keys().cloned().collect();
        for sandbox_id in ids {
            pass.advance_assignment(&sandbox_id).await?;
        }
        pass.advance_movement().await?;
        pass.collect_garbage().await?;
        Ok(pass.report)
    }
}

impl Pass<'_> {
    fn desired(sandbox: &Sandbox) -> Option<(String, u64)> {
        let phase = SandboxPhase::try_from(sandbox.phase()).unwrap_or(SandboxPhase::Unknown);
        if !wants_supervision(phase) || !has_runtime_binding(sandbox) {
            return None;
        }
        Some((
            sandbox_generation(sandbox)?,
            sandbox_credential_epoch(sandbox),
        ))
    }

    /// Create (or reset for a new launch) the assignment of every sandbox
    /// runtime that wants supervision.
    async fn ensure_desired_assignments(&mut self) -> Result<(), CoordinateError> {
        let mut wanted = Vec::new();
        for sandbox in self.sandboxes.values() {
            let Some((generation, credential_epoch)) = Self::desired(sandbox) else {
                continue;
            };
            let sandbox_id = sandbox.object_id().to_string();
            let existing = self.assignments.get(&sandbox_id);
            if existing.is_some_and(|assignment| {
                assignment.runtime_generation == generation
                    && assignment.credential_epoch == credential_epoch
            }) {
                continue;
            }
            wanted.push((
                sandbox_id,
                sandbox.object_name().to_string(),
                generation,
                credential_epoch,
                existing.cloned(),
            ));
        }
        for (sandbox_id, name, generation, credential_epoch, existing) in wanted {
            let mut pending = Assignment::pending(&sandbox_id, &name, &generation);
            pending.credential_epoch = credential_epoch;
            let mut batch = Batch {
                assignment: Some((existing, pending)),
                ..Batch::default()
            };
            let mut operation =
                PoolOperation::assign(&sandbox_id, &generation, credential_epoch, self.now);
            operation.set_phase(OperationPhase::Pending, "waiting for placement", self.now);
            if !self.operations.contains_key(&operation.id) {
                self.stage_operation(&mut batch, operation);
            }
            if self.commit(batch).await? {
                info!(%sandbox_id, generation, credential_epoch, "Created supervisor assignment");
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_lines)]
    async fn advance_assignment(&mut self, sandbox_id: &str) -> Result<(), CoordinateError> {
        let Some(assignment) = self.assignments.get(sandbox_id).cloned() else {
            return Ok(());
        };
        let sandbox = self.sandboxes.get(sandbox_id).cloned();
        let wanted = sandbox.as_ref().and_then(Self::desired).is_some_and(
            |(generation, credential_epoch)| {
                generation == assignment.runtime_generation
                    && credential_epoch == assignment.credential_epoch
            },
        );
        let linked = self.operation(assignment.operation_id.as_deref());
        let now = self.now;
        let timing = self.pool.timing;
        let mut batch = Batch::default();

        // Stop or delete wins over everything else.
        if !wanted {
            if matches!(
                assignment.state,
                AssignmentState::Terminated | AssignmentState::Failed
            ) {
                if sandbox.is_none() {
                    batch.deletes.push(assignment.fenced_delete());
                    self.commit(batch).await?;
                }
                return Ok(());
            }
            if let Some(mut operation) = linked
                && matches!(
                    operation.kind,
                    OperationKind::Assign | OperationKind::Recover
                )
                && !operation.phase.is_terminal()
            {
                operation.set_phase(
                    OperationPhase::Cancelled,
                    "sandbox stopped or was deleted",
                    now,
                );
                self.stage_operation(&mut batch, operation);
            }
            batch.assignment = Some((Some(assignment.clone()), assignment.terminate()));
            if self.commit(batch).await? {
                info!(%sandbox_id, "Sandbox no longer wants supervision; terminated assignment");
            }
            return Ok(());
        }

        if let Some(failed) = assignment.fail_if_overdue(now, &timing) {
            let reason = failed.failure_reason.clone().unwrap_or_default();
            let mut operation = match linked {
                Some(operation)
                    if matches!(
                        operation.kind,
                        OperationKind::Assign | OperationKind::Recover
                    ) =>
                {
                    operation
                }
                // The owner lapsed and no candidate was ever reserved.
                _ if assignment.state == AssignmentState::Active => self
                    .operation(Some(&PoolOperation::recover_id(
                        sandbox_id,
                        &assignment.runtime_generation,
                        assignment.credential_epoch,
                        assignment.ownership_epoch,
                    )))
                    .unwrap_or_else(|| {
                        PoolOperation::recover(
                            sandbox_id,
                            &assignment.runtime_generation,
                            assignment.credential_epoch,
                            assignment.ownership_epoch,
                            now,
                        )
                    }),
                _ => PoolOperation::assign(
                    sandbox_id,
                    &assignment.runtime_generation,
                    assignment.credential_epoch,
                    now,
                ),
            };
            operation.set_phase(OperationPhase::Failed, reason.clone(), now);
            self.stage_operation(&mut batch, operation);
            batch.recovery_failed = (assignment.state != AssignmentState::Launching).then(|| {
                (
                    sandbox_id.to_string(),
                    assignment.runtime_generation.clone(),
                    assignment.credential_epoch,
                    reason.clone(),
                )
            });
            batch.assignment = Some((Some(assignment.clone()), failed));
            if self.commit(batch).await? {
                warn!(%sandbox_id, reason, "Supervisor recovery failed");
            }
            return Ok(());
        }

        match assignment.state {
            AssignmentState::Pending => {
                let mut operation = linked.unwrap_or_else(|| {
                    self.operation(Some(&PoolOperation::assign_id(
                        sandbox_id,
                        &assignment.runtime_generation,
                        assignment.credential_epoch,
                    )))
                    .unwrap_or_else(|| {
                        PoolOperation::assign(
                            sandbox_id,
                            &assignment.runtime_generation,
                            assignment.credential_epoch,
                            now,
                        )
                    })
                });
                let Some(member) = self.least_loaded(&[]) else {
                    operation.set_phase(
                        OperationPhase::Blocked,
                        "no eligible supervisor replica (none live, or all draining)",
                        now,
                    );
                    self.stage_operation(&mut batch, operation);
                    self.commit(batch).await?;
                    return Ok(());
                };
                let mut next = assignment
                    .assign_launch(&member, now, &timing)
                    .map_err(|error| CoordinateError::Store(error.to_string()))?;
                next.operation_id = Some(operation.id.clone());
                operation.target_member = Some(member.clone());
                operation.ownership_epoch = Some(next.ownership_epoch);
                operation.set_phase(
                    OperationPhase::Running,
                    format!("launching on {member}"),
                    now,
                );
                self.stage_operation(&mut batch, operation);
                batch.assignment = Some((Some(assignment), next));
                if self.commit(batch).await? {
                    info!(%sandbox_id, %member, "Assigned sandbox to shared supervisor");
                }
            }
            AssignmentState::Active if assignment.needs_recovery(now) => {
                let lost_owner = assignment.owner_member.clone().unwrap_or_default();
                let mut operation = self
                    .operation(Some(&PoolOperation::recover_id(
                        sandbox_id,
                        &assignment.runtime_generation,
                        assignment.credential_epoch,
                        assignment.ownership_epoch,
                    )))
                    .unwrap_or_else(|| {
                        PoolOperation::recover(
                            sandbox_id,
                            &assignment.runtime_generation,
                            assignment.credential_epoch,
                            assignment.ownership_epoch,
                            now,
                        )
                    });
                // Prefer a pending handoff's target, if still eligible.
                let preferred = assignment
                    .handoff
                    .as_ref()
                    .map(|handoff| handoff.target_member.clone())
                    .filter(|target| self.is_eligible(target) && *target != lost_owner);
                let Some(candidate) = preferred.or_else(|| self.least_loaded(&[&lost_owner]))
                else {
                    operation.deadline_ms =
                        Some(assignment.lease_expires_at_ms + timing.recovery_budget_ms);
                    operation.set_phase(
                        OperationPhase::Blocked,
                        format!(
                            "owner {lost_owner} lost; no eligible supervisor replica (draining replicas are excluded)"
                        ),
                        now,
                    );
                    self.stage_operation(&mut batch, operation);
                    self.commit(batch).await?;
                    return Ok(());
                };
                let Ok(mut next) = assignment.reserve_recovery(&candidate, now, &timing) else {
                    return Ok(());
                };
                self.rotate_gateway_token(&mut batch, sandbox_id).await?;
                let handoff_operation = assignment.handoff.as_ref().map(|h| h.operation_id.clone());
                operation.attempts += 1;
                operation.target_member = Some(candidate.clone());
                operation.ownership_epoch = Some(next.ownership_epoch);
                operation.deadline_ms = next.recovery_deadline_ms;
                operation.set_phase(
                    OperationPhase::Running,
                    format!(
                        "owner {lost_owner} lost; fenced recovery reserved for {candidate} at epoch {}",
                        next.ownership_epoch
                    ),
                    now,
                );
                next.operation_id = Some(operation.id.clone());
                self.stage_operation(&mut batch, operation);
                if let Some(handoff_operation) = handoff_operation {
                    // The handoff turned into failure recovery.
                    self.stage_move_update(
                        &mut batch,
                        &handoff_operation,
                        sandbox_id,
                        |m| {
                            m.state = MoveState::Reserved;
                            m.to_member.clone_from(&candidate);
                            m.message = "source lost during handoff; recovering".to_string();
                        },
                        Some(next.ownership_epoch),
                    );
                }
                let epoch = next.ownership_epoch;
                batch.assignment = Some((Some(assignment), next));
                if self.commit(batch).await? {
                    warn!(
                        %sandbox_id,
                        previous_owner = lost_owner,
                        %candidate,
                        ownership_epoch = epoch,
                        "Reserved fenced supervisor recovery"
                    );
                }
            }
            AssignmentState::Active if assignment.handoff.is_some() => {
                self.advance_handoff(assignment).await?;
            }
            AssignmentState::Active => {
                // A launch or recovery finished.
                if let Some(mut operation) = linked
                    && matches!(
                        operation.kind,
                        OperationKind::Assign | OperationKind::Recover
                    )
                    && !operation.phase.is_terminal()
                {
                    let owner = assignment.owner_member.clone().unwrap_or_default();
                    operation.set_phase(
                        OperationPhase::Succeeded,
                        format!("active on {owner} at epoch {}", assignment.ownership_epoch),
                        now,
                    );
                    self.stage_operation(&mut batch, operation);
                    self.commit(batch).await?;
                }
            }
            AssignmentState::Recovering if assignment.needs_recovery(now) => {
                // The reserved candidate did not activate within its attempt
                // budget: reserve another epoch under the SAME deadline.
                let failed_candidate = assignment.owner_member.clone().unwrap_or_default();
                let mut operation = linked;
                let Some(candidate) = self.least_loaded(&[&failed_candidate]) else {
                    if let Some(mut operation) = operation
                        && operation.kind == OperationKind::Recover
                    {
                        operation.set_phase(
                            OperationPhase::Blocked,
                            format!(
                                "candidate {failed_candidate} did not activate; no other eligible supervisor replica"
                            ),
                            now,
                        );
                        self.stage_operation(&mut batch, operation);
                        self.commit(batch).await?;
                    }
                    return Ok(());
                };
                let Ok(mut next) = assignment.reserve_recovery(&candidate, now, &timing) else {
                    return Ok(());
                };
                next.planned = assignment.planned;
                next.operation_id.clone_from(&assignment.operation_id);
                self.rotate_gateway_token(&mut batch, sandbox_id).await?;
                match operation.as_mut() {
                    Some(operation) if operation.kind == OperationKind::Recover => {
                        operation.attempts += 1;
                        operation.target_member = Some(candidate.clone());
                        operation.ownership_epoch = Some(next.ownership_epoch);
                        operation.set_phase(
                            OperationPhase::Running,
                            format!(
                                "candidate {failed_candidate} did not activate; reserved {candidate} at epoch {}",
                                next.ownership_epoch
                            ),
                            now,
                        );
                        self.stage_operation(&mut batch, operation.clone());
                    }
                    Some(operation) if operation.kind.moves_sessions() => {
                        let id = operation.id.clone();
                        self.stage_move_update(
                            &mut batch,
                            &id,
                            sandbox_id,
                            |m| {
                                m.to_member.clone_from(&candidate);
                                m.message =
                                    format!("target {failed_candidate} did not activate; retrying");
                            },
                            Some(next.ownership_epoch),
                        );
                    }
                    _ => {}
                }
                let epoch = next.ownership_epoch;
                batch.assignment = Some((Some(assignment), next));
                if self.commit(batch).await? {
                    warn!(%sandbox_id, %candidate, ownership_epoch = epoch, "Reserved another recovery candidate");
                }
            }
            _ => {}
        }
        Ok(())
    }

    /// Add the gateway-token rotation that revokes the previous owner's
    /// gateway credential to a reservation batch. Reads the sandbox fresh so
    /// the CAS rarely conflicts with unrelated status updates.
    async fn rotate_gateway_token(
        &self,
        batch: &mut Batch,
        sandbox_id: &str,
    ) -> Result<(), CoordinateError> {
        if let Some(sandbox) = self.pool.sandbox(sandbox_id).await? {
            batch.sandbox = Some(rotation_op(&sandbox)?);
        }
        Ok(())
    }

    /// Release stage of a planned handoff: once the owner released ownership
    /// (its newest report no longer lists the sandbox), the owner is gone,
    /// or the release budget passed, reserve the target at the next epoch.
    async fn advance_handoff(&mut self, assignment: Assignment) -> Result<(), CoordinateError> {
        let Some(handoff) = assignment.handoff.clone() else {
            return Ok(());
        };
        let now = self.now;
        let timing = self.pool.timing;
        let owner = assignment.owner_member.clone().unwrap_or_default();
        let released = match self.member(&owner) {
            None => true,
            Some(member) if !member.live => true,
            Some(member) => {
                member.last_seen_ms > handoff.started_at_ms
                    && !member.hosted.contains(&assignment.sandbox_id)
            }
        };
        if !released && now < handoff.release_deadline_ms {
            return Ok(());
        }
        let target = Some(handoff.target_member.clone())
            .filter(|target| self.is_eligible(target) && *target != owner)
            .or_else(|| self.least_loaded(&[&owner]));
        let mut batch = Batch::default();
        let Some(target) = target else {
            // Nothing eligible to hand off to: keep the workload with its
            // current owner and report the move as failed.
            let next = assignment
                .cancel_handoff()
                .map_err(|error| CoordinateError::Store(error.to_string()))?;
            self.stage_move_update(
                &mut batch,
                &handoff.operation_id,
                &assignment.sandbox_id,
                |m| {
                    m.state = MoveState::Failed;
                    m.message = "no eligible target replica".to_string();
                },
                None,
            );
            batch.assignment = Some((Some(assignment), next));
            self.commit(batch).await?;
            return Ok(());
        };
        let mut next = assignment
            .reserve_planned(&target, now, &timing)
            .map_err(|error| CoordinateError::Store(error.to_string()))?;
        next.operation_id = Some(handoff.operation_id.clone());
        self.rotate_gateway_token(&mut batch, &assignment.sandbox_id)
            .await?;
        let released_note = if released {
            "owner released"
        } else {
            "release budget passed; the runtime fences the previous owner"
        };
        self.stage_move_update(
            &mut batch,
            &handoff.operation_id,
            &assignment.sandbox_id,
            |m| {
                m.state = MoveState::Reserved;
                m.to_member.clone_from(&target);
                m.message = format!("{released_note}; target reserved");
            },
            Some(next.ownership_epoch),
        );
        let sandbox_id = assignment.sandbox_id.clone();
        let epoch = next.ownership_epoch;
        batch.assignment = Some((Some(assignment), next));
        if self.commit(batch).await? {
            info!(%sandbox_id, from = owner, to = target, ownership_epoch = epoch, released, "Reserved planned supervisor handoff");
        }
        Ok(())
    }

    /// Stage an update of one move of a movement operation.
    fn stage_move_update(
        &self,
        batch: &mut Batch,
        operation_id: &str,
        sandbox_id: &str,
        update: impl FnOnce(&mut SessionMove),
        to_epoch: Option<u64>,
    ) {
        // Reuse an operation already staged in this batch.
        let staged = batch
            .operations
            .iter()
            .position(|(_, operation)| operation.id == operation_id);
        let mut operation = match staged {
            Some(index) => batch.operations.remove(index).1,
            None => match self.operations.get(operation_id) {
                Some(operation) => operation.clone(),
                None => return,
            },
        };
        if let Some(entry) = operation
            .moves
            .iter_mut()
            .rev()
            .find(|m| m.sandbox_id == sandbox_id)
        {
            update(entry);
            if to_epoch.is_some() {
                entry.to_epoch = to_epoch;
            }
        }
        operation.updated_at_ms = self.now;
        let expected = self.operations.get(operation_id).cloned();
        batch.operations.push((expected, operation));
    }

    /// Advance the oldest non-terminal drain or rebalance operation; others
    /// wait. Bounded by the operation's `max_parallel` in-flight handoffs.
    #[allow(clippy::too_many_lines)]
    async fn advance_movement(&mut self) -> Result<(), CoordinateError> {
        let mut active: Vec<PoolOperation> = self
            .operations
            .values()
            .filter(|operation| operation.kind.moves_sessions() && !operation.phase.is_terminal())
            .cloned()
            .collect();
        active.sort_by(|a, b| a.created_at_ms.cmp(&b.created_at_ms).then(a.id.cmp(&b.id)));
        let mut active = active.into_iter();
        let Some(mut operation) = active.next() else {
            return Ok(());
        };
        let now = self.now;
        for mut waiting in active {
            if waiting.phase == OperationPhase::Pending {
                let message = format!("waiting for operation {}", operation.id);
                if waiting.message != message {
                    waiting.set_phase(OperationPhase::Pending, message, now);
                    let mut batch = Batch::default();
                    self.stage_operation(&mut batch, waiting);
                    self.commit(batch).await?;
                }
            }
        }

        // Admission: the oldest queued operation becomes the single active
        // one in a fenced write. Queued operations were inert until now.
        if !operation.admitted {
            let mut admitted = operation.clone();
            if operation.cancel_requested {
                admitted.set_phase(OperationPhase::Cancelled, "cancelled before admission", now);
            } else if operation.kind == OperationKind::Drain {
                let member = operation.member_id.clone().unwrap_or_default();
                let remaining = self
                    .members
                    .iter()
                    .filter(|m| m.member_id != member && self.is_eligible(&m.member_id))
                    .count();
                if self.cordoned.contains(&member) {
                    admitted.set_phase(
                        OperationPhase::Failed,
                        format!("{member} is already drained"),
                        now,
                    );
                } else if remaining == 0 {
                    admitted.set_phase(
                        OperationPhase::Failed,
                        format!(
                            "draining {member} would leave no eligible supervisor replica; not admitted"
                        ),
                        now,
                    );
                } else {
                    admitted.admitted = true;
                    admitted.set_phase(OperationPhase::Running, "admitted", now);
                }
            } else {
                admitted.admitted = true;
                admitted.set_phase(OperationPhase::Running, "admitted", now);
            }
            let mut batch = Batch::default();
            self.stage_operation(&mut batch, admitted);
            if self.commit(batch).await? {
                info!(operation = %operation.id, "Admitted supervisor pool operation");
            }
            // Execution starts next pass, with eligibility recomputed from
            // durable state.
            return Ok(());
        }

        // Refresh in-flight moves from the assignments they track.
        for entry in &mut operation.moves {
            if !entry.state.in_flight() {
                continue;
            }
            let Some(assignment) = self.assignments.get(&entry.sandbox_id) else {
                entry.state = MoveState::Skipped;
                entry.message = "sandbox deleted".to_string();
                continue;
            };
            if assignment.runtime_generation != entry.runtime_generation
                || assignment.credential_epoch != entry.credential_epoch
                || assignment.state == AssignmentState::Terminated
            {
                entry.state = MoveState::Skipped;
                entry.message = "sandbox stopped or relaunched".to_string();
                continue;
            }
            match assignment.state {
                AssignmentState::Failed => {
                    entry.state = MoveState::Failed;
                    entry.message = assignment.failure_reason.clone().unwrap_or_default();
                }
                AssignmentState::Active if assignment.ownership_epoch > entry.from_epoch => {
                    entry.state = MoveState::Done;
                    entry.to_member = assignment.owner_member.clone().unwrap_or_default();
                    entry.to_epoch = Some(assignment.ownership_epoch);
                    entry.message = format!(
                        "active on {} at epoch {}",
                        entry.to_member, assignment.ownership_epoch
                    );
                }
                AssignmentState::Active
                    if entry.state == MoveState::Releasing && assignment.handoff.is_none() =>
                {
                    // Handoff cancelled (e.g. no eligible target).
                    entry.state = MoveState::Failed;
                    if entry.message.is_empty() {
                        entry.message = "handoff abandoned".to_string();
                    }
                }
                _ => {}
            }
        }

        let in_flight = operation.in_flight_moves();
        let max_parallel = usize::try_from(operation.max_parallel.max(1)).unwrap_or(1);
        let mut batch_moves: Vec<(Assignment, Assignment)> = Vec::new();
        let next_phase: (OperationPhase, String);

        if operation.cancel_requested {
            // Stop starting moves; abandon handoffs that have not reserved a
            // target yet; let reserved ones finish.
            for entry in &mut operation.moves {
                if entry.state == MoveState::Releasing
                    && let Some(assignment) = self.assignments.get(&entry.sandbox_id)
                    && assignment
                        .handoff
                        .as_ref()
                        .is_some_and(|h| h.operation_id == operation.id)
                    && let Ok(next) = assignment.cancel_handoff()
                {
                    batch_moves.push((assignment.clone(), next));
                    entry.state = MoveState::Failed;
                    entry.message = "operation cancelled".to_string();
                }
            }
            next_phase = if operation.in_flight_moves() == 0 {
                (OperationPhase::Cancelled, "cancelled".to_string())
            } else {
                (
                    OperationPhase::Running,
                    format!(
                        "cancelling; {} handoffs finishing",
                        operation.in_flight_moves()
                    ),
                )
            };
        } else {
            let mut started = 0usize;
            let moved: BTreeSet<String> = operation
                .moves
                .iter()
                .map(|m| m.sandbox_id.clone())
                .collect();
            // Projected load including reservations of moves started below.
            let mut load = self.load();
            loop {
                if in_flight + started >= max_parallel {
                    next_phase = (
                        OperationPhase::Running,
                        format!("{} handoffs in flight", in_flight + started),
                    );
                    break;
                }
                let (candidate, target) = match operation.kind {
                    OperationKind::Drain => {
                        let member = operation.member_id.clone().unwrap_or_default();
                        let candidate = self
                            .assignments
                            .values()
                            .find(|a| {
                                a.state == AssignmentState::Active
                                    && a.owner_member.as_deref() == Some(member.as_str())
                                    && a.handoff.is_none()
                                    && now < a.lease_expires_at_ms
                                    && !batch_moves
                                        .iter()
                                        .any(|(c, _)| c.sandbox_id == a.sandbox_id)
                            })
                            .cloned();
                        let Some(candidate) = candidate else {
                            let remaining = self
                                .assignments
                                .values()
                                .filter(|a| {
                                    a.occupies_owner()
                                        && a.owner_member.as_deref() == Some(member.as_str())
                                })
                                .count();
                            next_phase = if remaining == 0 && in_flight + started == 0 {
                                (OperationPhase::Succeeded, format!("{member} drained"))
                            } else {
                                (
                                    OperationPhase::Running,
                                    format!(
                                        "waiting for {remaining} sessions on {member} ({} handoffs in flight)",
                                        in_flight + started
                                    ),
                                )
                            };
                            break;
                        };
                        let target = load
                            .iter()
                            .filter(|(m, _)| **m != member)
                            .min_by(|(am, a), (bm, b)| a.cmp(b).then(am.cmp(bm)))
                            .map(|(m, _)| m.clone());
                        let Some(target) = target else {
                            next_phase = (
                                OperationPhase::Blocked,
                                format!(
                                    "no eligible replica to receive sessions from {member}; waiting for capacity"
                                ),
                            );
                            break;
                        };
                        (candidate, target)
                    }
                    OperationKind::Rebalance => {
                        if load.len() < 2 {
                            next_phase = if in_flight + started == 0 {
                                (
                                    OperationPhase::Succeeded,
                                    "fewer than two eligible replicas; nothing to balance"
                                        .to_string(),
                                )
                            } else {
                                (OperationPhase::Running, "handoffs finishing".to_string())
                            };
                            break;
                        }
                        let (max_member, max) = load
                            .iter()
                            .max_by(|(am, a), (bm, b)| a.cmp(b).then(bm.cmp(am)))
                            .map(|(m, c)| (m.clone(), *c))
                            .unwrap_or_default();
                        let (min_member, min) = load
                            .iter()
                            .min_by(|(am, a), (bm, b)| a.cmp(b).then(am.cmp(bm)))
                            .map(|(m, c)| (m.clone(), *c))
                            .unwrap_or_default();
                        if max <= min + 1 {
                            next_phase = if in_flight + started == 0 {
                                (
                                    OperationPhase::Succeeded,
                                    format!("balanced (max {max}, min {min})"),
                                )
                            } else {
                                (OperationPhase::Running, "handoffs finishing".to_string())
                            };
                            break;
                        }
                        let candidate = self
                            .assignments
                            .values()
                            .find(|a| {
                                a.state == AssignmentState::Active
                                    && a.owner_member.as_deref() == Some(max_member.as_str())
                                    && a.handoff.is_none()
                                    && now < a.lease_expires_at_ms
                                    && !moved.contains(&a.sandbox_id)
                                    && !batch_moves
                                        .iter()
                                        .any(|(c, _)| c.sandbox_id == a.sandbox_id)
                            })
                            .cloned();
                        let Some(candidate) = candidate else {
                            next_phase = if in_flight + started == 0 {
                                (
                                    OperationPhase::Blocked,
                                    format!(
                                        "imbalanced (max {max} on {max_member}, min {min}) but no movable session; waiting"
                                    ),
                                )
                            } else {
                                (OperationPhase::Running, "handoffs finishing".to_string())
                            };
                            break;
                        };
                        (candidate, min_member)
                    }
                    OperationKind::Assign | OperationKind::Recover => {
                        next_phase = (OperationPhase::Failed, "not a movement operation".into());
                        break;
                    }
                };
                let Ok(mut next) =
                    candidate.begin_handoff(&target, &operation.id, now, &self.pool.timing)
                else {
                    next_phase = (
                        OperationPhase::Running,
                        format!(
                            "could not start a handoff of {}; retrying",
                            candidate.sandbox_id
                        ),
                    );
                    break;
                };
                next.operation_id.clone_from(&candidate.operation_id);
                if let Some(count) = candidate
                    .owner_member
                    .as_ref()
                    .and_then(|owner| load.get_mut(owner))
                {
                    *count = count.saturating_sub(1);
                }
                *load.entry(target.clone()).or_default() += 1;
                operation.moves.push(SessionMove {
                    sandbox_id: candidate.sandbox_id.clone(),
                    sandbox_name: candidate.sandbox_name.clone(),
                    from_member: candidate.owner_member.clone().unwrap_or_default(),
                    to_member: target,
                    runtime_generation: candidate.runtime_generation.clone(),
                    credential_epoch: candidate.credential_epoch,
                    from_epoch: candidate.ownership_epoch,
                    to_epoch: None,
                    state: MoveState::Releasing,
                    message: "owner asked to release ownership".to_string(),
                });
                batch_moves.push((candidate, next));
                started += 1;
            }
        }

        operation.set_phase(next_phase.0, next_phase.1, now);
        // Commit the operation together with each started/cancelled handoff;
        // on conflict the operation's moves are rebuilt next pass.
        if batch_moves.is_empty() {
            let mut batch = Batch::default();
            self.stage_operation(&mut batch, operation);
            self.commit(batch).await?;
            return Ok(());
        }
        // All handoffs of one pass commit atomically with the operation.
        let mut ops = Vec::new();
        let expected_operation = self.operations.get(&operation.id).cloned();
        operation.coordinator_term = self.fence.term;
        operation.updated_at_ms = now;
        for (expected, next) in &batch_moves {
            ops.push(next.fenced_write(Some(expected))?);
        }
        ops.push(operation.fenced_write(expected_operation.as_ref())?);
        match self.pool.store.write_fenced(self.fence, now, &ops).await {
            Ok(results) => {
                self.report.committed += 1;
                let mut results = results.into_iter();
                for (_, mut next) in batch_moves {
                    if let Some(Some(written)) = results.next() {
                        next.resource_version = written.resource_version;
                    }
                    info!(
                        sandbox_id = %next.sandbox_id,
                        target = next.handoff.as_ref().map(|h| h.target_member.as_str()).unwrap_or_default(),
                        operation = %operation.id,
                        "Started planned supervisor handoff"
                    );
                    self.assignments.insert(next.sandbox_id.clone(), next);
                }
                if let Some(Some(written)) = results.next() {
                    operation.resource_version = written.resource_version;
                }
                self.operations.insert(operation.id.clone(), operation);
            }
            Err(PersistenceError::Conflict { .. }) => self.report.conflicts += 1,
            Err(error) => return Err(error.into()),
        }
        Ok(())
    }

    /// Drop terminal operations past retention. A succeeded drain stays (and
    /// keeps its member cordoned) until it is cancelled or its member is
    /// gone.
    async fn collect_garbage(&mut self) -> Result<(), CoordinateError> {
        let now = self.now;
        let known: BTreeSet<&str> = self.members.iter().map(|m| m.member_id.as_str()).collect();
        let expired: Vec<PoolOperation> = self
            .operations
            .values()
            .filter(|operation| {
                let Some(finished) = operation.finished_at_ms else {
                    return false;
                };
                if !operation.phase.is_terminal() {
                    return false;
                }
                if operation.cordoned_member().is_some() {
                    return operation
                        .member_id
                        .as_deref()
                        .is_some_and(|member| !known.contains(member));
                }
                let retention = if operation.kind.moves_sessions() {
                    LONG_RETENTION_MS
                } else {
                    SHORT_RETENTION_MS
                };
                now.saturating_sub(finished) > retention
            })
            .cloned()
            .collect();
        for operation in expired {
            let batch = Batch {
                deletes: vec![operation.fenced_delete()],
                ..Batch::default()
            };
            self.commit(batch).await?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::time::Duration;

    use openshell_core::proto::{SandboxPhase, SupervisorAssignmentAction as Action};

    use super::super::assignment::{AssignmentState, AssignmentTiming};
    use super::super::operations::{MoveState, OperationKind, OperationPhase, PoolOperation};
    use super::super::test_support::{FAST, Harness, pass, renew, sync, take_over};
    use super::*;

    /// Long owner leases so healthy owners never lapse during a test.
    const SLOW: AssignmentTiming = AssignmentTiming {
        lease_ttl_ms: 20_000,
        lease_interval_ms: 5_000,
        launch_grace_ms: 60_000,
        recovery_budget_ms: 30_000,
        attempt_budget_ms: 10_000,
        runtime_recovery_window_ms: 60_000,
        release_budget_ms: 5_000,
    };

    fn later(ms: i64) -> i64 {
        now_ms() + ms
    }

    async fn assignment(gateway: &SupervisorPool, sandbox_id: &str) -> Assignment {
        gateway.assignments.get(sandbox_id).await.unwrap().unwrap()
    }

    async fn owners(gateway: &SupervisorPool) -> BTreeMap<String, usize> {
        let mut owners = BTreeMap::new();
        for a in gateway.assignments.list().await.unwrap() {
            if a.occupies_owner() {
                *owners
                    .entry(a.owner_member.unwrap_or_default())
                    .or_default() += 1;
            }
        }
        owners
    }

    async fn operations(gateway: &SupervisorPool) -> Vec<PoolOperation> {
        gateway.operations.list().await.unwrap()
    }

    /// Launch every pending sandbox: members sync, the coordinator places,
    /// owners renew (Launching -> Active).
    async fn launch_all(gateway: &SupervisorPool, fence: &CoordinatorFence, members: &[&str]) {
        for m in members {
            sync(gateway, m, &[]).await;
        }
        pass(gateway, fence).await.unwrap();
        for m in members {
            for (sandbox, action, epoch) in sync(gateway, m, &[]).await {
                if action == Action::Launch {
                    renew(gateway, m, &sandbox, epoch).await.unwrap();
                }
            }
        }
        pass(gateway, fence).await.unwrap();
    }

    /// Play every pool member: report hosted sessions (relinquished ones are
    /// gone), take over reserved recoveries, renew active ones. Then pass.
    async fn step(
        harness: &Harness,
        gateway: &SupervisorPool,
        fence: &CoordinatorFence,
        members: &[&str],
        actions_seen: &mut Vec<Action>,
    ) {
        for m in members {
            // First learn the actions, then report the post-action state.
            let actions = sync(gateway, m, &[]).await;
            let mut hosted = Vec::new();
            for (sandbox, action, epoch) in &actions {
                actions_seen.push(*action);
                match action {
                    Action::Recover => {
                        take_over(harness, gateway, m, sandbox, *epoch)
                            .await
                            .unwrap();
                        renew(gateway, m, sandbox, *epoch).await.unwrap();
                        hosted.push(sandbox.clone());
                    }
                    Action::Supervise | Action::Launch => {
                        renew(gateway, m, sandbox, *epoch).await.unwrap();
                        hosted.push(sandbox.clone());
                    }
                    // Released ownership: no longer hosted, workload untouched.
                    Action::Relinquish | Action::Release | Action::Unspecified => {}
                }
            }
            let hosted: Vec<&str> = hosted.iter().map(String::as_str).collect();
            sync(gateway, m, &hosted).await;
        }
        pass(gateway, fence).await.unwrap();
    }

    async fn run_operation(
        harness: &Harness,
        gateway: &SupervisorPool,
        fence: &CoordinatorFence,
        members: &[&str],
        id: &str,
    ) -> (PoolOperation, Vec<Action>, usize) {
        let mut actions = Vec::new();
        let mut max_in_flight = 0;
        for _ in 0..40 {
            step(harness, gateway, fence, members, &mut actions).await;
            let operation = gateway.operations.get(id).await.unwrap().unwrap();
            max_in_flight = max_in_flight.max(operation.in_flight_moves());
            if operation.phase.is_terminal() {
                return (operation, actions, max_in_flight);
            }
        }
        panic!("operation {id} did not finish");
    }

    #[tokio::test]
    async fn exactly_one_coordinator_places_and_a_takeover_fences_the_old_one() {
        let harness = Harness::new(SLOW).await;
        let (gw_a, gw_b) = (harness.gateway().await, harness.gateway().await);
        harness.put_sandbox("s1", 1, SandboxPhase::Ready).await;
        sync(&gw_a, "m1", &[]).await;

        let a = harness
            .elect("gw-a/1", later(0))
            .await
            .expect("first gateway elected");
        assert!(
            harness.elect("gw-b/1", later(0)).await.is_none(),
            "one coordinator"
        );
        // The standby runs no reconciliation: nothing is placed.
        assert!(gw_b.coordinator_tick("gw-b/1", None).await.is_none());
        assert!(gw_b.assignments.get("s1").await.unwrap().is_none());

        pass(&gw_a, &a).await.unwrap();
        assert_eq!(
            assignment(&gw_a, "s1").await.state,
            AssignmentState::Launching
        );

        // The coordinator stops renewing; after its TTL another gateway takes
        // over with a higher term, and the old one is fenced.
        let b = harness
            .elect("gw-b/1", later(COORDINATOR_TTL_MS + 1))
            .await
            .expect("standby takes over after expiry");
        assert_eq!(b.term, a.term + 1);
        harness.put_sandbox("s2", 1, SandboxPhase::Ready).await;
        assert!(matches!(
            pass(&gw_a, &a).await,
            Err(CoordinateError::Fenced)
        ));
        assert!(
            gw_a.assignments.get("s2").await.unwrap().is_none(),
            "a fenced coordinator writes nothing"
        );
        assert!(
            gw_a.coordinator_tick("gw-a/1", Some(a)).await.is_none(),
            "steps down"
        );
        pass(&gw_b, &b).await.unwrap();
        assert_eq!(
            assignment(&gw_b, "s2").await.state,
            AssignmentState::Launching
        );
    }

    #[tokio::test]
    async fn paused_coordinator_cannot_reserve_recovery_after_takeover() {
        let harness = Harness::new(FAST).await;
        let gw_a = harness.gateway().await;
        let gw_b = harness.gateway().await;
        harness.put_sandbox("s1", 1, SandboxPhase::Ready).await;
        let a = harness.elect("gw-a/1", later(0)).await.unwrap();
        launch_all(&gw_a, &a, &["m1"]).await;
        sync(&gw_a, "m2", &[]).await;
        let before = assignment(&gw_a, "s1").await;
        let token_before = gw_a.sandbox("s1").await.unwrap().unwrap();

        // m1 is lost; gateway A pauses before acting on the lapse while B
        // takes over the coordinator lease.
        tokio::time::sleep(Duration::from_millis(350)).await;
        sync(&gw_a, "m2", &[]).await;
        let b = harness
            .elect("gw-b/1", later(COORDINATOR_TTL_MS + 1))
            .await
            .unwrap();
        assert!(matches!(
            pass(&gw_a, &a).await,
            Err(CoordinateError::Fenced)
        ));
        let unchanged = assignment(&gw_a, "s1").await;
        assert_eq!(unchanged.ownership_epoch, before.ownership_epoch);
        assert_eq!(
            gw_a.sandbox("s1").await.unwrap().unwrap(),
            token_before,
            "no token rotation from the stale coordinator"
        );
        pass(&gw_b, &b).await.unwrap();
        let reserved = assignment(&gw_b, "s1").await;
        assert_eq!(reserved.state, AssignmentState::Recovering);
        assert_eq!(reserved.owner_member.as_deref(), Some("m2"));
    }

    #[tokio::test]
    async fn healthy_supervision_continues_through_any_gateway_while_the_coordinator_changes() {
        let harness = Harness::new(FAST).await;
        let gw_a = harness.gateway().await;
        let gw_b = harness.gateway().await;
        let gw_c = harness.gateway().await; // never coordinates
        harness.put_sandbox("s1", 1, SandboxPhase::Ready).await;
        harness.put_sandbox("s2", 1, SandboxPhase::Ready).await;
        let a = harness.elect("gw-a/1", later(0)).await.unwrap();
        launch_all(&gw_a, &a, &["m1", "m2"]).await;
        let s1 = assignment(&gw_a, "s1").await;
        let s2 = assignment(&gw_a, "s2").await;
        let (s1_owner, s2_owner) = (
            s1.owner_member.clone().unwrap(),
            s2.owner_member.clone().unwrap(),
        );
        assert_ne!(s1_owner, s2_owner);

        // Gateway A stops coordinating (crash). For ~1 s nobody coordinates,
        // then B takes over. s1's owner renews through C the whole time;
        // s2's owner is lost and must be recovered through C.
        let mut b = None;
        for tick in 0..12 {
            renew(&gw_c, &s1_owner, "s1", 1)
                .await
                .expect("renewal through a non-coordinator");
            sync(&gw_c, &s1_owner, &["s1"]).await;
            if tick == 6 {
                b = harness.elect("gw-b/1", later(COORDINATOR_TTL_MS + 1)).await;
            }
            if let Some(b) = &b {
                pass(&gw_b, b).await.unwrap();
            }
            tokio::time::sleep(Duration::from_millis(80)).await;
        }
        let b = b.expect("B elected");
        let s1_after = assignment(&gw_c, "s1").await;
        assert_eq!(s1_after.state, AssignmentState::Active);
        assert_eq!(
            s1_after.ownership_epoch, 1,
            "a healthy owner is never moved or recovered"
        );
        assert_eq!(s1_after.recoveries, 0);

        // s2 was recovered onto s1's owner by the new coordinator; the
        // candidate's grant and activation go through gateway C.
        let s2_after = assignment(&gw_c, "s2").await;
        assert_eq!(s2_after.state, AssignmentState::Recovering);
        assert_eq!(s2_after.owner_member.as_deref(), Some(s1_owner.as_str()));
        take_over(&harness, &gw_c, &s1_owner, "s2", s2_after.ownership_epoch)
            .await
            .unwrap();
        renew(&gw_c, &s1_owner, "s2", s2_after.ownership_epoch)
            .await
            .unwrap();
        pass(&gw_b, &b).await.unwrap();
        assert_eq!(assignment(&gw_c, "s2").await.state, AssignmentState::Active);
    }

    #[tokio::test]
    async fn new_coordinator_adopts_pending_operations_without_duplicates_or_extensions() {
        let harness = Harness::new(FAST).await;
        let gw_a = harness.gateway().await;
        let gw_b = harness.gateway().await;
        let gw_c = harness.gateway().await;
        harness.put_sandbox("s1", 1, SandboxPhase::Ready).await;
        sync(&gw_a, "m1", &[]).await;
        let a = harness.elect("gw-a/1", later(0)).await.unwrap();
        pass(&gw_a, &a).await.unwrap();
        let launching = assignment(&gw_a, "s1").await;
        assert_eq!(launching.state, AssignmentState::Launching);

        // A dies mid-launch. B adopts: no second launch, no new operation.
        sync(&gw_a, "m2", &[]).await;
        let b = harness
            .elect("gw-b/1", later(COORDINATOR_TTL_MS + 1))
            .await
            .unwrap();
        pass(&gw_b, &b).await.unwrap();
        let adopted = assignment(&gw_b, "s1").await;
        assert_eq!(adopted.state, AssignmentState::Launching);
        assert_eq!(adopted.owner_member, launching.owner_member);
        assert_eq!(adopted.ownership_epoch, 1);
        let assigns: Vec<_> = operations(&gw_b)
            .await
            .into_iter()
            .filter(|o| o.kind == OperationKind::Assign)
            .collect();
        assert_eq!(assigns.len(), 1);
        assert_eq!(assigns[0].phase, OperationPhase::Running);
        renew(&gw_b, "m1", "s1", 1).await.unwrap();
        pass(&gw_b, &b).await.unwrap();
        assert_eq!(
            gw_b.operations
                .get(&assigns[0].id)
                .await
                .unwrap()
                .unwrap()
                .phase,
            OperationPhase::Succeeded
        );

        // m1 is lost; B reserves m2 and records the recovery operation.
        tokio::time::sleep(Duration::from_millis(350)).await;
        sync(&gw_b, "m2", &[]).await;
        sync(&gw_b, "m1", &[]).await; // m1 is reachable again but lost its lease
        pass(&gw_b, &b).await.unwrap();
        let reserved = assignment(&gw_b, "s1").await;
        assert_eq!(reserved.state, AssignmentState::Recovering);
        let deadline = reserved.recovery_deadline_ms.unwrap();
        let recover_id = reserved.operation_id.clone().unwrap();

        // B dies; m2 never activates. C adopts the recovery after the attempt
        // budget: another epoch, the SAME operation and deadline.
        let c = harness
            .elect("gw-c/1", later(2 * COORDINATOR_TTL_MS + 2))
            .await
            .unwrap();
        tokio::time::sleep(Duration::from_millis(1_050)).await;
        sync(&gw_c, "m1", &[]).await;
        sync(&gw_c, "m2", &[]).await;
        pass(&gw_c, &c).await.unwrap();
        let retried = assignment(&gw_c, "s1").await;
        assert_eq!(retried.state, AssignmentState::Recovering);
        assert_eq!(retried.ownership_epoch, reserved.ownership_epoch + 1);
        assert_eq!(
            retried.recovery_deadline_ms,
            Some(deadline),
            "deadline never extended"
        );
        assert_eq!(retried.operation_id.as_deref(), Some(recover_id.as_str()));
        let recover = gw_c.operations.get(&recover_id).await.unwrap().unwrap();
        assert_eq!(recover.attempts, 2);
        assert_eq!(recover.deadline_ms, Some(deadline));
        assert_eq!(
            operations(&gw_c)
                .await
                .iter()
                .filter(|o| o.kind == OperationKind::Recover)
                .count(),
            1
        );
    }

    #[tokio::test]
    async fn drain_hands_off_in_bounded_batches_and_cordons_the_member() {
        let harness = Harness::new(SLOW).await;
        let gateway = harness.gateway().await;
        for id in ["s1", "s2", "s3"] {
            harness.put_sandbox(id, 1, SandboxPhase::Ready).await;
        }
        let fence = harness.elect("gw-a/1", later(0)).await.unwrap();
        // Only m1 is up during placement.
        launch_all(&gateway, &fence, &["m1"]).await;
        assert_eq!(
            owners(&gateway).await,
            BTreeMap::from([("m1".to_string(), 3)])
        );
        sync(&gateway, "m2", &[]).await;

        let drain = gateway
            .operations
            .create(&PoolOperation::drain("m1", 1, None, later(0)))
            .await
            .unwrap();
        let (done, actions, max_in_flight) = Box::pin(run_operation(
            &harness,
            &gateway,
            &fence,
            &["m1", "m2"],
            &drain.id,
        ))
        .await;
        assert_eq!(done.phase, OperationPhase::Succeeded, "{}", done.message);
        assert_eq!(max_in_flight, 1, "at most max_parallel handoffs in flight");
        assert!(
            !actions.contains(&Action::Release),
            "planned movement never terminates a session"
        );
        assert!(actions.contains(&Action::Relinquish));
        assert_eq!(done.moves.len(), 3);
        for m in &done.moves {
            assert_eq!(m.state, MoveState::Done);
            assert_eq!((m.from_member.as_str(), m.to_member.as_str()), ("m1", "m2"));
        }
        for id in ["s1", "s2", "s3"] {
            let a = assignment(&gateway, id).await;
            assert_eq!(a.state, AssignmentState::Active);
            assert_eq!(a.owner_member.as_deref(), Some("m2"));
            assert_eq!(a.ownership_epoch, 2, "one fenced handoff, same generation");
            assert_eq!(a.runtime_generation, format!("generation-{id}"));
        }

        // The drained member stays cordoned: new placement avoids it.
        harness.put_sandbox("s4", 1, SandboxPhase::Ready).await;
        launch_all(&gateway, &fence, &["m1", "m2"]).await;
        assert_eq!(
            assignment(&gateway, "s4").await.owner_member.as_deref(),
            Some("m2")
        );

        // A second drain (of the only eligible member) is refused at
        // admission instead of cordoning the whole pool.
        let blocked = gateway
            .operations
            .create(&PoolOperation::drain("m2", 2, None, later(1)))
            .await
            .unwrap();
        step(&harness, &gateway, &fence, &["m1", "m2"], &mut Vec::new()).await;
        let blocked = gateway.operations.get(&blocked.id).await.unwrap().unwrap();
        assert_eq!(blocked.phase, OperationPhase::Failed, "{}", blocked.message);
        assert!(blocked.message.contains("no eligible supervisor replica"));
        assert!(!blocked.admitted);
        assert!(blocked.moves.is_empty());
    }

    #[tokio::test]
    async fn recovery_never_targets_a_draining_member() {
        let harness = Harness::new(FAST).await;
        let gateway = harness.gateway().await;
        harness.put_sandbox("s1", 1, SandboxPhase::Ready).await;
        let fence = harness.elect("gw-a/1", later(0)).await.unwrap();
        launch_all(&gateway, &fence, &["m2"]).await;
        sync(&gateway, "m1", &[]).await;
        gateway
            .operations
            .create(&PoolOperation::drain("m1", 1, None, later(0)))
            .await
            .unwrap();
        pass(&gateway, &fence).await.unwrap();
        // m2 is lost; the only other member is draining.
        tokio::time::sleep(Duration::from_millis(350)).await;
        sync(&gateway, "m1", &[]).await;
        pass(&gateway, &fence).await.unwrap();
        let a = assignment(&gateway, "s1").await;
        assert_eq!(
            a.state,
            AssignmentState::Active,
            "no reservation onto a draining member"
        );
        let recover = operations(&gateway)
            .await
            .into_iter()
            .find(|o| o.kind == OperationKind::Recover)
            .expect("recovery recorded");
        assert_eq!(recover.phase, OperationPhase::Blocked);
        assert!(recover.message.contains("no eligible supervisor replica"));
        // Past the recovery deadline the sandbox fails closed.
        tokio::time::sleep(Duration::from_millis(
            u64::try_from(FAST.recovery_budget_ms).unwrap(),
        ))
        .await;
        sync(&gateway, "m1", &[]).await;
        pass(&gateway, &fence).await.unwrap();
        assert_eq!(
            assignment(&gateway, "s1").await.state,
            AssignmentState::Failed
        );
    }

    #[tokio::test]
    async fn delete_wins_over_a_planned_handoff() {
        let harness = Harness::new(SLOW).await;
        let gateway = harness.gateway().await;
        harness.put_sandbox("s1", 1, SandboxPhase::Ready).await;
        harness.put_sandbox("s2", 1, SandboxPhase::Ready).await;
        let fence = harness.elect("gw-a/1", later(0)).await.unwrap();
        launch_all(&gateway, &fence, &["m1"]).await;
        sync(&gateway, "m2", &[]).await;
        let drain = gateway
            .operations
            .create(&PoolOperation::drain("m1", 2, None, later(0)))
            .await
            .unwrap();
        pass(&gateway, &fence).await.unwrap(); // admission
        pass(&gateway, &fence).await.unwrap(); // handoffs start
        assert!(assignment(&gateway, "s1").await.handoff.is_some());

        harness.set_phase("s1", SandboxPhase::Deleting).await;
        // The relinquishing owner reports s1 gone; the coordinator must not
        // reserve a target for a deleted sandbox.
        sync(&gateway, "m1", &["s2"]).await;
        pass(&gateway, &fence).await.unwrap();
        let s1 = assignment(&gateway, "s1").await;
        assert_eq!(s1.state, AssignmentState::Terminated);
        assert!(s1.handoff.is_none());
        assert_eq!(s1.ownership_epoch, 1, "no target reserved");
        let (done, _, _) = Box::pin(run_operation(
            &harness,
            &gateway,
            &fence,
            &["m1", "m2"],
            &drain.id,
        ))
        .await;
        let s1_move = done.moves.iter().find(|m| m.sandbox_id == "s1").unwrap();
        assert_eq!(s1_move.state, MoveState::Skipped);
        assert_eq!(done.phase, OperationPhase::Succeeded);
    }

    async fn rebalance_case(initial: &[(&str, usize)], expected_sorted: &[usize]) {
        let harness = Harness::new(SLOW).await;
        let gateway = harness.gateway().await;
        let fence = harness.elect("gw-a/1", later(0)).await.unwrap();
        let members: Vec<&str> = initial.iter().map(|(m, _)| *m).collect();
        let mut n = 0;
        for (m, count) in initial {
            for _ in 0..*count {
                n += 1;
                harness
                    .put_sandbox(&format!("s{n}"), 1, SandboxPhase::Ready)
                    .await;
            }
            // Place this batch while only `m` is eligible.
            launch_all(&gateway, &fence, &[m]).await;
        }
        for m in &members {
            sync(&gateway, m, &[]).await;
        }
        let rebalance = gateway
            .operations
            .create(&PoolOperation::rebalance(1, None, later(0)))
            .await
            .unwrap();
        let (done, actions, max_in_flight) = Box::pin(run_operation(
            &harness,
            &gateway,
            &fence,
            &members,
            &rebalance.id,
        ))
        .await;
        assert_eq!(done.phase, OperationPhase::Succeeded, "{}", done.message);
        assert!(!actions.contains(&Action::Release));
        assert!(max_in_flight <= 1);
        let mut counts: Vec<usize> = members
            .iter()
            .map(|m| owners_sync(&gateway, m))
            .collect::<futures::future::JoinAll<_>>()
            .await;
        counts.sort_unstable_by(|a, b| b.cmp(a));
        assert_eq!(counts, expected_sorted, "moves: {:?}", done.moves);
        // Every moved session moved exactly once.
        let mut moved: Vec<&str> = done.moves.iter().map(|m| m.sandbox_id.as_str()).collect();
        moved.sort_unstable();
        moved.dedup();
        assert_eq!(moved.len(), done.moves.len());
        // Further passes never move a healthy, balanced assignment again.
        let before = gateway.assignments.list().await.unwrap();
        for _ in 0..3 {
            step(&harness, &gateway, &fence, &members, &mut Vec::new()).await;
        }
        let after = gateway.assignments.list().await.unwrap();
        let epochs = |v: &[Assignment]| {
            v.iter()
                .map(|a| {
                    (
                        a.sandbox_id.clone(),
                        a.ownership_epoch,
                        a.owner_member.clone(),
                    )
                })
                .collect::<Vec<_>>()
        };
        assert_eq!(epochs(&before), epochs(&after));
    }

    async fn owners_sync(gateway: &SupervisorPool, member: &str) -> usize {
        owners(gateway).await.get(member).copied().unwrap_or(0)
    }

    #[tokio::test]
    async fn rebalance_converges_for_skewed_three_member_pools() {
        Box::pin(rebalance_case(
            &[("m1", 3), ("m2", 1), ("m3", 1)],
            &[2, 2, 1],
        ))
        .await;
    }

    #[tokio::test]
    async fn rebalance_spreads_two_sessions_off_one_member() {
        Box::pin(rebalance_case(
            &[("m1", 2), ("m2", 0), ("m3", 0)],
            &[1, 1, 0],
        ))
        .await;
    }

    #[tokio::test]
    async fn healthy_assignments_are_stable_and_placement_is_accounted_within_a_pass() {
        let harness = Harness::new(SLOW).await;
        let gateway = harness.gateway().await;
        for n in 1..=4 {
            harness
                .put_sandbox(&format!("s{n}"), 1, SandboxPhase::Ready)
                .await;
        }
        let fence = harness.elect("gw-a/1", later(0)).await.unwrap();
        launch_all(&gateway, &fence, &["m1", "m2"]).await;
        assert_eq!(
            owners(&gateway).await,
            BTreeMap::from([("m1".to_string(), 2), ("m2".to_string(), 2)]),
            "one pass places with up-to-date load accounting"
        );
        // Settle operation records, then no pass writes anything.
        step(&harness, &gateway, &fence, &["m1", "m2"], &mut Vec::new()).await;
        for _ in 0..3 {
            for m in ["m1", "m2"] {
                for (sandbox, _, epoch) in sync(&gateway, m, &[]).await {
                    renew(&gateway, m, &sandbox, epoch).await.unwrap();
                }
            }
            let report = pass(&gateway, &fence).await.unwrap();
            assert_eq!(
                report,
                PassReport::default(),
                "healthy assignments are stable"
            );
        }
    }
}
