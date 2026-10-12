// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Durable supervisor pool operations.
//!
//! Every coordinator decision that takes more than one step is recorded as an
//! operation: assigning a sandbox, recovering a lost owner, draining a
//! replica, and an explicit rebalance. The elected coordinator advances
//! operations with fenced writes in the same transaction as the assignment
//! changes they describe, so a newly elected coordinator adopts them exactly
//! where the previous one stopped.
//!
//! Assign and recover operations have deterministic IDs derived from the
//! assignment they track, so re-running a step after failover finds the
//! existing record instead of creating a duplicate. Drain and rebalance
//! operations are requested by an operator through any gateway and executed
//! only by the coordinator.

use serde::{Deserialize, Serialize};

use crate::persistence::coordinator::FencedOp;
use crate::persistence::{PersistenceError, Store, WriteCondition};

pub(crate) const OPERATION_OBJECT_TYPE: &str = "supervisor_pool_operation";

/// Bounds on planned movement concurrency.
pub const DEFAULT_MAX_PARALLEL: u32 = 1;
pub const MAX_MAX_PARALLEL: u32 = 8;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OperationKind {
    Assign,
    Recover,
    Drain,
    Rebalance,
}

impl OperationKind {
    #[must_use]
    pub const fn moves_sessions(self) -> bool {
        matches!(self, Self::Drain | Self::Rebalance)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OperationPhase {
    /// Recorded; waiting for the coordinator (or for another operation).
    Pending,
    /// Being advanced by the coordinator.
    Running,
    /// Cannot progress right now (e.g. no eligible capacity); retried.
    Blocked,
    Succeeded,
    Failed,
    Cancelled,
}

impl OperationPhase {
    #[must_use]
    pub const fn is_terminal(self) -> bool {
        matches!(self, Self::Succeeded | Self::Failed | Self::Cancelled)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MoveState {
    /// The owner was asked to release ownership (handoff recorded).
    Releasing,
    /// The target was reserved at a new epoch and is taking over.
    Reserved,
    /// The target activated the assignment.
    Done,
    /// The sandbox stopped or changed generation; nothing to move.
    Skipped,
    /// The handoff could not complete (recovery failed or was abandoned).
    Failed,
}

impl MoveState {
    #[must_use]
    pub const fn in_flight(self) -> bool {
        matches!(self, Self::Releasing | Self::Reserved)
    }
}

/// One planned session move inside a drain or rebalance operation.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionMove {
    pub sandbox_id: String,
    #[serde(default)]
    pub sandbox_name: String,
    pub from_member: String,
    pub to_member: String,
    pub runtime_generation: String,
    pub credential_epoch: u64,
    /// Ownership epoch of the source owner when the move started.
    pub from_epoch: u64,
    /// Ownership epoch reserved for the target, once reserved.
    #[serde(default)]
    pub to_epoch: Option<u64>,
    pub state: MoveState,
    #[serde(default)]
    pub message: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PoolOperation {
    pub id: String,
    pub kind: OperationKind,
    pub phase: OperationPhase,
    /// Human-readable progress or the reason for Blocked / Failed.
    #[serde(default)]
    pub message: String,
    pub created_at_ms: i64,
    pub updated_at_ms: i64,
    #[serde(default)]
    pub finished_at_ms: Option<i64>,
    /// Coordinator term that last advanced the operation.
    #[serde(default)]
    pub coordinator_term: u64,
    /// Requesting principal for operator operations.
    #[serde(default)]
    pub requested_by: Option<String>,

    // Assign / recover.
    #[serde(default)]
    pub sandbox_id: Option<String>,
    #[serde(default)]
    pub sandbox_name: Option<String>,
    #[serde(default)]
    pub runtime_generation: Option<String>,
    #[serde(default)]
    pub credential_epoch: Option<u64>,
    /// Latest ownership epoch this operation reserved.
    #[serde(default)]
    pub ownership_epoch: Option<u64>,
    /// Recovery deadline, copied from the assignment and never extended.
    #[serde(default)]
    pub deadline_ms: Option<i64>,
    /// Candidates reserved so far (recover).
    #[serde(default)]
    pub attempts: u32,
    #[serde(default)]
    pub target_member: Option<String>,

    // Drain / rebalance.
    /// Member being drained.
    #[serde(default)]
    pub member_id: Option<String>,
    #[serde(default = "default_max_parallel")]
    pub max_parallel: u32,
    #[serde(default)]
    pub moves: Vec<SessionMove>,
    #[serde(default)]
    pub cancel_requested: bool,
    /// Set by the elected coordinator (fenced) when it admits this movement
    /// operation. A recorded but not yet admitted operation is inert: it
    /// cordons nothing and moves nothing, so concurrent requests accepted by
    /// different gateways cannot jointly cordon the pool.
    #[serde(default)]
    pub admitted: bool,

    /// Store version for compare-and-swap; not serialized.
    #[serde(skip)]
    pub resource_version: u64,
}

const fn default_max_parallel() -> u32 {
    DEFAULT_MAX_PARALLEL
}

impl PoolOperation {
    fn base(id: String, kind: OperationKind, now_ms: i64) -> Self {
        Self {
            id,
            kind,
            phase: OperationPhase::Pending,
            message: String::new(),
            created_at_ms: now_ms,
            updated_at_ms: now_ms,
            finished_at_ms: None,
            coordinator_term: 0,
            requested_by: None,
            sandbox_id: None,
            sandbox_name: None,
            runtime_generation: None,
            credential_epoch: None,
            ownership_epoch: None,
            deadline_ms: None,
            attempts: 0,
            target_member: None,
            member_id: None,
            max_parallel: DEFAULT_MAX_PARALLEL,
            moves: Vec::new(),
            cancel_requested: false,
            admitted: false,
            resource_version: 0,
        }
    }

    /// Deterministic ID of the assign operation for one launch.
    #[must_use]
    pub fn assign_id(sandbox_id: &str, generation: &str, credential_epoch: u64) -> String {
        format!("assign:{sandbox_id}:{generation}:{credential_epoch}")
    }

    /// Deterministic ID of the recover operation for one owner loss, keyed by
    /// the lost owner's epoch (every later candidate shares the record).
    #[must_use]
    pub fn recover_id(
        sandbox_id: &str,
        generation: &str,
        credential_epoch: u64,
        lost_epoch: u64,
    ) -> String {
        format!("recover:{sandbox_id}:{generation}:{credential_epoch}:{lost_epoch}")
    }

    #[must_use]
    pub fn assign(sandbox_id: &str, generation: &str, credential_epoch: u64, now_ms: i64) -> Self {
        let mut operation = Self::base(
            Self::assign_id(sandbox_id, generation, credential_epoch),
            OperationKind::Assign,
            now_ms,
        );
        operation.sandbox_id = Some(sandbox_id.to_string());
        operation.runtime_generation = Some(generation.to_string());
        operation.credential_epoch = Some(credential_epoch);
        operation
    }

    #[must_use]
    pub fn recover(
        sandbox_id: &str,
        generation: &str,
        credential_epoch: u64,
        lost_epoch: u64,
        now_ms: i64,
    ) -> Self {
        let mut operation = Self::base(
            Self::recover_id(sandbox_id, generation, credential_epoch, lost_epoch),
            OperationKind::Recover,
            now_ms,
        );
        operation.sandbox_id = Some(sandbox_id.to_string());
        operation.runtime_generation = Some(generation.to_string());
        operation.credential_epoch = Some(credential_epoch);
        operation
    }

    #[must_use]
    pub fn drain(
        member_id: &str,
        max_parallel: u32,
        requested_by: Option<String>,
        now_ms: i64,
    ) -> Self {
        let mut operation = Self::base(
            format!("drain:{}", uuid::Uuid::new_v4()),
            OperationKind::Drain,
            now_ms,
        );
        operation.member_id = Some(member_id.to_string());
        operation.max_parallel = max_parallel;
        operation.requested_by = requested_by;
        operation
    }

    #[must_use]
    pub fn rebalance(max_parallel: u32, requested_by: Option<String>, now_ms: i64) -> Self {
        let mut operation = Self::base(
            format!("rebalance:{}", uuid::Uuid::new_v4()),
            OperationKind::Rebalance,
            now_ms,
        );
        operation.max_parallel = max_parallel;
        operation.requested_by = requested_by;
        operation
    }

    /// Move to `phase` with `message`, stamping finish time for terminal phases.
    pub fn set_phase(&mut self, phase: OperationPhase, message: impl Into<String>, now_ms: i64) {
        self.phase = phase;
        self.message = message.into();
        self.updated_at_ms = now_ms;
        if phase.is_terminal() && self.finished_at_ms.is_none() {
            self.finished_at_ms = Some(now_ms);
        }
    }

    /// Member cordoned by this operation: an ADMITTED, non-cancelled drain
    /// keeps its member out of placement, recovery, and planned targets,
    /// including after it succeeded (cancel the drain to return the member to
    /// service). Queued (not yet admitted) drains cordon nothing.
    #[must_use]
    pub fn cordoned_member(&self) -> Option<&str> {
        (self.kind == OperationKind::Drain
            && self.admitted
            && !self.cancel_requested
            && !matches!(
                self.phase,
                OperationPhase::Failed | OperationPhase::Cancelled
            ))
        .then_some(self.member_id.as_deref())
        .flatten()
    }

    #[must_use]
    pub fn in_flight_moves(&self) -> usize {
        self.moves.iter().filter(|m| m.state.in_flight()).count()
    }

    pub fn fenced_write(&self, expected: Option<&Self>) -> Result<FencedOp, PersistenceError> {
        Ok(FencedOp::Put {
            object_type: OPERATION_OBJECT_TYPE.to_string(),
            id: self.id.clone(),
            name: self.id.clone(),
            workspace: String::new(),
            payload: serde_json::to_vec(self)
                .map_err(|error| PersistenceError::Encode(error.to_string()))?,
            labels: None,
            condition: expected.map_or(WriteCondition::MustCreate, |expected| {
                WriteCondition::MatchResourceVersion(expected.resource_version)
            }),
        })
    }

    #[must_use]
    pub fn fenced_delete(&self) -> FencedOp {
        FencedOp::Delete {
            object_type: OPERATION_OBJECT_TYPE.to_string(),
            id: self.id.clone(),
            resource_version: self.resource_version,
        }
    }
}

/// Durable operation storage. Operator requests (create, cancel) use plain
/// CAS writes; coordinator progress uses fenced writes.
#[derive(Clone, Debug)]
pub struct OperationStore {
    store: std::sync::Arc<Store>,
}

impl OperationStore {
    #[must_use]
    pub fn new(store: std::sync::Arc<Store>) -> Self {
        Self { store }
    }

    fn decode(payload: &[u8], resource_version: u64) -> Result<PoolOperation, PersistenceError> {
        let mut operation: PoolOperation = serde_json::from_slice(payload)
            .map_err(|error| PersistenceError::Decode(error.to_string()))?;
        operation.resource_version = resource_version;
        Ok(operation)
    }

    pub async fn get(&self, id: &str) -> Result<Option<PoolOperation>, PersistenceError> {
        self.store
            .get(OPERATION_OBJECT_TYPE, id)
            .await?
            .map(|record| Self::decode(&record.payload, record.resource_version))
            .transpose()
    }

    /// All operations, oldest first.
    pub async fn list(&self) -> Result<Vec<PoolOperation>, PersistenceError> {
        let mut operations = Vec::new();
        let mut offset = 0;
        loop {
            let page = self
                .store
                .list_by_type(OPERATION_OBJECT_TYPE, 500, offset)
                .await?;
            let count = page.len();
            for record in page {
                operations.push(Self::decode(&record.payload, record.resource_version)?);
            }
            if count < 500 {
                break;
            }
            offset += 500;
        }
        operations.sort_by(|a, b| {
            a.created_at_ms
                .cmp(&b.created_at_ms)
                .then_with(|| a.id.cmp(&b.id))
        });
        Ok(operations)
    }

    /// Operator request: record a new operation.
    pub async fn create(
        &self,
        operation: &PoolOperation,
    ) -> Result<PoolOperation, PersistenceError> {
        self.write(None, operation).await
    }

    /// Operator request: CAS update (e.g. cancellation).
    pub async fn write(
        &self,
        expected: Option<&PoolOperation>,
        next: &PoolOperation,
    ) -> Result<PoolOperation, PersistenceError> {
        let payload = serde_json::to_vec(next)
            .map_err(|error| PersistenceError::Encode(error.to_string()))?;
        let condition = expected.map_or(WriteCondition::MustCreate, |expected| {
            WriteCondition::MatchResourceVersion(expected.resource_version)
        });
        let result = self
            .store
            .put_if(
                OPERATION_OBJECT_TYPE,
                &next.id,
                &next.id,
                "",
                &payload,
                None,
                condition,
            )
            .await
            .map_err(|error| match error {
                PersistenceError::UniqueViolation { .. } => PersistenceError::Conflict {
                    current_resource_version: None,
                },
                other => other,
            })?;
        let mut written = next.clone();
        written.resource_version = result.resource_version;
        Ok(written)
    }
}
