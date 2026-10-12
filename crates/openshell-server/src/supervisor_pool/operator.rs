// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Operator surface for the shared supervisor pool: status, and durable
//! drain / rebalance requests.
//!
//! Any gateway records a request as a durable operation; only the elected
//! coordinator executes it. Authorization (bearer, `platform_admin`, and
//! `config:read` / `config:write`) is enforced by the RPC descriptor.

use std::collections::BTreeMap;

use openshell_core::proto::{
    CancelSupervisorPoolOperationRequest, CancelSupervisorPoolOperationResponse,
    GetSupervisorPoolStatusResponse, StartSupervisorPoolOperationRequest,
    StartSupervisorPoolOperationResponse, SupervisorPoolCoordinator, SupervisorPoolMember,
    SupervisorPoolOperation, SupervisorPoolOperationKind, SupervisorPoolOperationPhase,
    SupervisorPoolSessionMove,
};
use openshell_core::time::now_ms;
use tonic::Status;

use super::coordinator::COORDINATOR_LEASE;
use super::operations::{
    MAX_MAX_PARALLEL, MoveState, OperationKind, OperationPhase, PoolOperation,
};
use super::{SupervisorPool, store_status};
use crate::persistence::PersistenceError;

fn timestamp(ms: i64) -> Option<prost_types::Timestamp> {
    openshell_core::time::timestamp_from_millis(ms).ok()
}

fn public_kind(kind: OperationKind) -> SupervisorPoolOperationKind {
    match kind {
        OperationKind::Assign => SupervisorPoolOperationKind::Assign,
        OperationKind::Recover => SupervisorPoolOperationKind::Recover,
        OperationKind::Drain => SupervisorPoolOperationKind::Drain,
        OperationKind::Rebalance => SupervisorPoolOperationKind::Rebalance,
    }
}

fn public_phase(phase: OperationPhase) -> SupervisorPoolOperationPhase {
    match phase {
        OperationPhase::Pending => SupervisorPoolOperationPhase::Pending,
        OperationPhase::Running => SupervisorPoolOperationPhase::Running,
        OperationPhase::Blocked => SupervisorPoolOperationPhase::Blocked,
        OperationPhase::Succeeded => SupervisorPoolOperationPhase::Succeeded,
        OperationPhase::Failed => SupervisorPoolOperationPhase::Failed,
        OperationPhase::Cancelled => SupervisorPoolOperationPhase::Cancelled,
    }
}

const fn move_state(state: MoveState) -> &'static str {
    match state {
        MoveState::Releasing => "releasing",
        MoveState::Reserved => "reserved",
        MoveState::Done => "done",
        MoveState::Skipped => "skipped",
        MoveState::Failed => "failed",
    }
}

/// Public view of an operation. Sandboxes are referenced by name.
#[must_use]
pub fn public_operation(operation: &PoolOperation) -> SupervisorPoolOperation {
    SupervisorPoolOperation {
        name: operation.id.clone(),
        kind: public_kind(operation.kind).into(),
        phase: public_phase(operation.phase).into(),
        message: operation.message.clone(),
        created_time: timestamp(operation.created_at_ms),
        updated_time: timestamp(operation.updated_at_ms),
        finished_time: operation.finished_at_ms.and_then(timestamp),
        sandbox: operation.sandbox_name.clone().unwrap_or_default(),
        member: operation
            .member_id
            .clone()
            .or_else(|| operation.target_member.clone())
            .unwrap_or_default(),
        max_parallel: operation.max_parallel,
        moves: operation
            .moves
            .iter()
            .map(|entry| SupervisorPoolSessionMove {
                sandbox: entry.sandbox_name.clone(),
                from_member: entry.from_member.clone(),
                to_member: entry.to_member.clone(),
                state: move_state(entry.state).to_string(),
                message: entry.message.clone(),
            })
            .collect(),
        cancel_requested: operation.cancel_requested,
        coordinator_term: operation.coordinator_term,
        attempts: operation.attempts,
        deadline: operation.deadline_ms.and_then(timestamp),
        requested_by: operation.requested_by.clone().unwrap_or_default(),
    }
}

impl SupervisorPool {
    pub async fn status(
        &self,
        serving_gateway: &str,
    ) -> Result<GetSupervisorPoolStatusResponse, Status> {
        let now = now_ms();
        let coordinator = self
            .store
            .get_coordinator_lease(COORDINATOR_LEASE)
            .await
            .map_err(store_status)?
            .map(|lease| SupervisorPoolCoordinator {
                expired: lease.expired(now),
                holder: lease.holder,
                term: lease.term,
                expiration_time: timestamp(lease.expires_at_ms),
            });
        let operations = self.operations.list().await.map_err(store_status)?;
        let cordoned: std::collections::BTreeSet<&str> = operations
            .iter()
            .filter_map(PoolOperation::cordoned_member)
            .collect();
        let mut sessions: BTreeMap<String, u32> = BTreeMap::new();
        for assignment in self.assignments.list().await.map_err(store_status)? {
            if assignment.occupies_owner()
                && let Some(owner) = assignment.owner_member
            {
                *sessions.entry(owner).or_default() += 1;
            }
        }
        let members = self
            .members(now)
            .await
            .map_err(store_status)?
            .into_iter()
            .map(|member| SupervisorPoolMember {
                draining: cordoned.contains(member.member_id.as_str()),
                sessions: sessions.get(&member.member_id).copied().unwrap_or(0),
                last_seen_time: timestamp(member.last_seen_ms),
                live: member.live,
                name: member.member_id,
            })
            .collect();
        Ok(GetSupervisorPoolStatusResponse {
            coordinator,
            members,
            operations: operations.iter().map(public_operation).collect(),
            serving_gateway: serving_gateway.to_string(),
        })
    }

    pub async fn start_operation(
        &self,
        request: StartSupervisorPoolOperationRequest,
        requested_by: Option<String>,
    ) -> Result<StartSupervisorPoolOperationResponse, Status> {
        let max_parallel = request.max_parallel.unwrap_or(1);
        if !(1..=MAX_MAX_PARALLEL).contains(&max_parallel) {
            return Err(Status::invalid_argument(format!(
                "max_parallel must be between 1 and {MAX_MAX_PARALLEL}"
            )));
        }
        let now = now_ms();
        let operations = self.operations.list().await.map_err(store_status)?;
        if let Some(active) = operations
            .iter()
            .find(|op| op.kind.moves_sessions() && !op.phase.is_terminal())
        {
            return Err(Status::failed_precondition(format!(
                "supervisor pool operation {} is still in progress; wait for it or cancel it",
                active.id
            )));
        }
        let operation = match SupervisorPoolOperationKind::try_from(request.kind) {
            Ok(SupervisorPoolOperationKind::Drain) => {
                let member = request.member.trim();
                if member.is_empty() {
                    return Err(Status::invalid_argument("drain requires a member"));
                }
                if !self
                    .members(now)
                    .await
                    .map_err(store_status)?
                    .iter()
                    .any(|known| known.member_id == member)
                {
                    return Err(Status::not_found(format!(
                        "supervisor pool member {member} not found"
                    )));
                }
                if operations
                    .iter()
                    .any(|op| op.cordoned_member() == Some(member))
                {
                    return Err(Status::already_exists(format!(
                        "supervisor pool member {member} is already drained; cancel that drain to return it to service"
                    )));
                }
                PoolOperation::drain(member, max_parallel, requested_by, now)
            }
            Ok(SupervisorPoolOperationKind::Rebalance) => {
                if !request.member.is_empty() {
                    return Err(Status::invalid_argument(
                        "rebalance applies to the whole pool; member must be empty",
                    ));
                }
                PoolOperation::rebalance(max_parallel, requested_by, now)
            }
            _ => {
                return Err(Status::invalid_argument(
                    "kind must be SUPERVISOR_POOL_OPERATION_KIND_DRAIN or _REBALANCE",
                ));
            }
        };
        let created = self
            .operations
            .create(&operation)
            .await
            .map_err(store_status)?;
        tracing::info!(
            operation = %created.id,
            kind = ?created.kind,
            member = created.member_id.as_deref().unwrap_or_default(),
            "Recorded supervisor pool operation"
        );
        Ok(StartSupervisorPoolOperationResponse {
            operation: Some(public_operation(&created)),
        })
    }

    pub async fn cancel_operation(
        &self,
        request: CancelSupervisorPoolOperationRequest,
    ) -> Result<CancelSupervisorPoolOperationResponse, Status> {
        for _ in 0..5 {
            let current = self
                .operations
                .get(&request.name)
                .await
                .map_err(store_status)?
                .ok_or_else(|| Status::not_found("supervisor pool operation not found"))?;
            if !current.kind.moves_sessions() {
                return Err(Status::failed_precondition(
                    "only drain and rebalance operations can be cancelled",
                ));
            }
            let mut next = current.clone();
            if current.phase == OperationPhase::Succeeded && current.cordoned_member().is_some() {
                // A finished drain only keeps its member cordoned: cancelling
                // returns the member to service immediately.
                next.cancel_requested = true;
                next.set_phase(
                    OperationPhase::Cancelled,
                    "member returned to service",
                    now_ms(),
                );
            } else if current.phase.is_terminal() {
                return Err(Status::failed_precondition("operation already finished"));
            } else {
                // The coordinator stops starting handoffs and lets reserved
                // ones finish.
                next.cancel_requested = true;
                next.updated_at_ms = now_ms();
            }
            match self.operations.write(Some(&current), &next).await {
                Ok(written) => {
                    return Ok(CancelSupervisorPoolOperationResponse {
                        operation: Some(public_operation(&written)),
                    });
                }
                Err(PersistenceError::Conflict { .. }) => {}
                Err(error) => return Err(store_status(error)),
            }
        }
        Err(Status::aborted(
            "supervisor pool operation changed concurrently; retry",
        ))
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use openshell_core::proto::{
        CancelSupervisorPoolOperationRequest, SandboxPhase, StartSupervisorPoolOperationRequest,
        SupervisorPoolOperationKind, SupervisorPoolOperationPhase,
    };

    use super::super::SupervisorPool;
    use super::super::assignment::AssignmentTiming;
    use super::super::test_support::{Harness, pass, renew, sync};

    const SLOW: AssignmentTiming = AssignmentTiming {
        lease_ttl_ms: 20_000,
        lease_interval_ms: 5_000,
        launch_grace_ms: 60_000,
        recovery_budget_ms: 30_000,
        attempt_budget_ms: 10_000,
        runtime_recovery_window_ms: 60_000,
        release_budget_ms: 5_000,
    };

    fn drain(member: &str) -> StartSupervisorPoolOperationRequest {
        StartSupervisorPoolOperationRequest {
            kind: SupervisorPoolOperationKind::Drain.into(),
            member: member.to_string(),
            max_parallel: None,
        }
    }

    async fn draining(gateway: &SupervisorPool) -> BTreeSet<String> {
        gateway
            .status("gw")
            .await
            .unwrap()
            .members
            .into_iter()
            .filter(|m| m.draining)
            .map(|m| m.name)
            .collect()
    }

    #[tokio::test]
    async fn concurrent_drain_requests_never_cordon_the_whole_pool() {
        let harness = Harness::new(SLOW).await;
        let (gw_a, gw_b) = (harness.gateway().await, harness.gateway().await);
        let fence = harness
            .elect("gw-a/1", openshell_core::time::now_ms())
            .await
            .unwrap();
        sync(&gw_a, "m1", &[]).await;
        sync(&gw_a, "m2", &[]).await;

        // Two operators race through two gateways. Both pre-checks may pass.
        let (first, second) = tokio::join!(
            gw_a.start_operation(drain("m1"), Some("alice".into())),
            gw_b.start_operation(drain("m2"), Some("bob".into())),
        );
        let accepted: Vec<_> = <[_; 2]>::from((first, second))
            .into_iter()
            .filter_map(Result::ok)
            .filter_map(|r| r.operation)
            .collect();
        assert!(!accepted.is_empty());

        // Recorded but not admitted: nothing is cordoned, placement still
        // uses both members.
        assert!(draining(&gw_a).await.is_empty(), "queued drains are inert");
        harness.put_sandbox("s1", 1, SandboxPhase::Ready).await;
        harness.put_sandbox("s2", 1, SandboxPhase::Ready).await;
        pass(&gw_a, &fence).await.unwrap();
        let placed: BTreeSet<_> = gw_a
            .assignments
            .list()
            .await
            .unwrap()
            .into_iter()
            .filter_map(|a| a.owner_member)
            .collect();
        assert_eq!(placed.len(), 2, "both members took placements: {placed:?}");

        // The coordinator admits exactly one; the other never cordons. If both
        // were accepted, the second is refused at admission because it would
        // leave no eligible replica.
        for _ in 0..12 {
            for m in ["m1", "m2"] {
                let mut hosted = Vec::new();
                for (sandbox, _, epoch) in sync(&gw_a, m, &[]).await {
                    if renew(&gw_a, m, &sandbox, epoch).await.is_ok() {
                        hosted.push(sandbox);
                    }
                }
                let hosted: Vec<&str> = hosted.iter().map(String::as_str).collect();
                sync(&gw_a, m, &hosted).await;
            }
            pass(&gw_a, &fence).await.unwrap();
            let cordoned = draining(&gw_a).await;
            assert!(
                cordoned.len() <= 1,
                "at most one member cordoned: {cordoned:?}"
            );
        }
        let operations = gw_b.operations.list().await.unwrap();
        let drains: Vec<_> = operations
            .iter()
            .filter(|op| op.kind == super::OperationKind::Drain)
            .collect();
        let admitted: Vec<_> = drains.iter().filter(|op| op.admitted).collect();
        assert_eq!(admitted.len(), 1, "exactly one drain admitted: {drains:?}");
        let cordoned = draining(&gw_b).await;
        assert_eq!(
            cordoned,
            BTreeSet::from([admitted[0].member_id.clone().unwrap()]),
            "only the admitted drain affects eligibility"
        );
        for queued in drains.iter().filter(|op| !op.admitted) {
            // Inert while queued behind the admitted one (or refused).
            assert!(
                matches!(
                    queued.phase,
                    super::OperationPhase::Pending | super::OperationPhase::Failed
                ),
                "{queued:?}"
            );
            assert!(queued.moves.is_empty());
        }
        assert_eq!(draining(&gw_b).await.len(), 1);
    }

    #[tokio::test]
    async fn operator_requests_are_validated_and_cancel_returns_a_member_to_service() {
        let harness = Harness::new(SLOW).await;
        let gateway = harness.gateway().await;
        let fence = harness
            .elect("gw-a/1", openshell_core::time::now_ms())
            .await
            .unwrap();
        sync(&gateway, "m1", &[]).await;
        sync(&gateway, "m2", &[]).await;
        let bad = |kind: SupervisorPoolOperationKind, member: &str, max_parallel| {
            StartSupervisorPoolOperationRequest {
                kind: kind.into(),
                member: member.to_string(),
                max_parallel,
            }
        };
        for (request, code) in [
            (
                bad(SupervisorPoolOperationKind::Assign, "", None),
                tonic::Code::InvalidArgument,
            ),
            (
                bad(SupervisorPoolOperationKind::Drain, "", None),
                tonic::Code::InvalidArgument,
            ),
            (
                bad(SupervisorPoolOperationKind::Drain, "nope", None),
                tonic::Code::NotFound,
            ),
            (
                bad(SupervisorPoolOperationKind::Drain, "m1", Some(0)),
                tonic::Code::InvalidArgument,
            ),
            (
                bad(SupervisorPoolOperationKind::Drain, "m1", Some(9)),
                tonic::Code::InvalidArgument,
            ),
            (
                bad(SupervisorPoolOperationKind::Rebalance, "m1", None),
                tonic::Code::InvalidArgument,
            ),
        ] {
            assert_eq!(
                gateway
                    .start_operation(request, None)
                    .await
                    .unwrap_err()
                    .code(),
                code
            );
        }
        let started = gateway
            .start_operation(drain("m1"), Some("alice".into()))
            .await
            .unwrap()
            .operation
            .unwrap();
        assert_eq!(started.requested_by, "alice");
        assert_eq!(
            gateway
                .start_operation(drain("m2"), None)
                .await
                .unwrap_err()
                .code(),
            tonic::Code::FailedPrecondition,
            "one movement operation at a time"
        );
        // Admit and finish (no sessions to move).
        for _ in 0..3 {
            pass(&gateway, &fence).await.unwrap();
        }
        assert_eq!(draining(&gateway).await, BTreeSet::from(["m1".to_string()]));
        harness.put_sandbox("s1", 1, SandboxPhase::Ready).await;
        pass(&gateway, &fence).await.unwrap();
        assert_eq!(
            gateway
                .assignments
                .get("s1")
                .await
                .unwrap()
                .unwrap()
                .owner_member
                .as_deref(),
            Some("m2"),
            "a drained member receives no placements"
        );
        let cancelled = gateway
            .cancel_operation(CancelSupervisorPoolOperationRequest {
                name: started.name.clone(),
            })
            .await
            .unwrap()
            .operation
            .unwrap();
        assert_eq!(
            cancelled.phase,
            SupervisorPoolOperationPhase::Cancelled as i32
        );
        assert!(
            draining(&gateway).await.is_empty(),
            "cancel returns the member to service"
        );
        assert_eq!(
            gateway
                .cancel_operation(CancelSupervisorPoolOperationRequest { name: started.name })
                .await
                .unwrap_err()
                .code(),
            tonic::Code::FailedPrecondition
        );
    }
}
