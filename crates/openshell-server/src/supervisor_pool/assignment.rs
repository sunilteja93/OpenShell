// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Durable per-sandbox enforcement ownership for the gateway supervisor topology.
//!
//! An assignment says which shared supervisor replica may enforce one sandbox
//! runtime generation and at which ownership epoch. It is separate from the
//! supervisor-session routing owner in `supervisor_owner`: routing decides
//! which gateway replica relays traffic, while this record decides which
//! supervisor may act on the boundary at all.
//!
//! Every transition is a pure function checked by a compare-and-swap write,
//! so concurrent gateways cannot both reserve the same epoch.

use serde::{Deserialize, Serialize};

use crate::persistence::{PersistenceError, Store, WriteCondition};

const ASSIGNMENT_OBJECT_TYPE: &str = "supervisor_assignment";

fn assignment_object_id(sandbox_id: &str) -> String {
    format!("supervisor-assignment:{sandbox_id}")
}

/// Timing for one installation. Validated so the worst-case recovery path
/// fits inside the runtime's authenticated reconnect window.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AssignmentTiming {
    /// Enforced owner lease; the runtime uses the same TTL.
    pub lease_ttl_ms: i64,
    /// How often the owner renews its lease.
    pub lease_interval_ms: i64,
    /// Lease granted to a launch owner until its first confirmed renewal.
    pub launch_grace_ms: i64,
    /// Bound from owner lease expiry to completed activation.
    pub recovery_budget_ms: i64,
    /// Time one candidate has to activate before another may be reserved.
    pub attempt_budget_ms: i64,
    /// The runtime's authenticated reconnect window.
    pub runtime_recovery_window_ms: i64,
}

impl AssignmentTiming {
    pub const DEFAULT: Self = Self {
        lease_ttl_ms: 10_000,
        lease_interval_ms: 2_500,
        launch_grace_ms: 300_000,
        recovery_budget_ms: 16_000,
        attempt_budget_ms: 8_000,
        runtime_recovery_window_ms: 30_000,
    };

    /// Reject timing that could leave a frozen workload without a usable
    /// recovery window, or renew too rarely to keep a healthy lease.
    pub fn validate(&self) -> Result<(), String> {
        if self.lease_interval_ms <= 0 || self.lease_interval_ms * 3 > self.lease_ttl_ms {
            return Err("supervisor lease must be renewable at least three times per TTL".into());
        }
        // The runtime freezes when the owner's transport drops, at most one
        // full lease before the gateway can observe expiry. The candidate
        // must activate before the runtime terminates the workload.
        if self.lease_ttl_ms + self.recovery_budget_ms >= self.runtime_recovery_window_ms {
            return Err(format!(
                "lease TTL ({} ms) plus recovery budget ({} ms) must be below the runtime recovery window ({} ms)",
                self.lease_ttl_ms, self.recovery_budget_ms, self.runtime_recovery_window_ms
            ));
        }
        if self.attempt_budget_ms <= 0 || self.attempt_budget_ms > self.recovery_budget_ms {
            return Err("recovery attempt budget must fit inside the recovery budget".into());
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AssignmentState {
    /// A runtime generation exists but no replica owns it yet.
    Pending,
    /// A replica was asked to launch; it owns epoch 1 under launch grace.
    Launching,
    /// The owner renews an enforced lease.
    Active,
    /// Epoch was advanced for a candidate that must fence and activate.
    Recovering,
    /// Recovery could not finish safely in time; the workload is stopped.
    Failed,
    /// Stop or delete won; no supervisor may act for this generation.
    Terminated,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Assignment {
    pub sandbox_id: String,
    pub sandbox_name: String,
    pub runtime_generation: String,
    /// Credential epoch of the launch; a restart keeps the generation but
    /// advances this epoch.
    #[serde(default)]
    pub credential_epoch: u64,
    pub ownership_epoch: u64,
    pub state: AssignmentState,
    pub owner_member: Option<String>,
    pub owner_instance: Option<String>,
    pub lease_expires_at_ms: i64,
    #[serde(default)]
    pub recovery_deadline_ms: Option<i64>,
    #[serde(default)]
    pub attempt_expires_at_ms: Option<i64>,
    /// Recovery grant issued for the reserved epoch; activation must
    /// present a receipt for exactly this grant.
    #[serde(default)]
    pub grant_id: Option<String>,
    #[serde(default)]
    pub failure_reason: Option<String>,
    #[serde(default)]
    pub recoveries: u64,
    /// Store version for compare-and-swap; not serialized.
    #[serde(skip)]
    pub resource_version: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum AssignmentError {
    #[error("sandbox supervision was terminated")]
    Terminated,
    #[error("sandbox supervision failed: {0}")]
    Failed(String),
    #[error("caller is not the assigned supervisor for this epoch")]
    NotOwner,
    #[error("assignment is not in a state that permits this operation")]
    WrongState,
    #[error("owner lease has expired")]
    LeaseExpired,
    #[error("owner lease is still valid")]
    LeaseValid,
    #[error("recovery deadline has passed")]
    DeadlinePassed,
    #[error("a recovery grant was already issued for this epoch")]
    GrantAlreadyIssued,
    #[error("supervisor instance does not match the activated owner")]
    InstanceMismatch,
}

impl Assignment {
    #[must_use]
    pub fn pending(sandbox_id: &str, sandbox_name: &str, runtime_generation: &str) -> Self {
        Self {
            sandbox_id: sandbox_id.to_string(),
            sandbox_name: sandbox_name.to_string(),
            runtime_generation: runtime_generation.to_string(),
            credential_epoch: 0,
            ownership_epoch: 0,
            state: AssignmentState::Pending,
            owner_member: None,
            owner_instance: None,
            lease_expires_at_ms: 0,
            recovery_deadline_ms: None,
            attempt_expires_at_ms: None,
            grant_id: None,
            failure_reason: None,
            recoveries: 0,
            resource_version: 0,
        }
    }

    fn check_live(&self) -> Result<(), AssignmentError> {
        match self.state {
            AssignmentState::Terminated => Err(AssignmentError::Terminated),
            AssignmentState::Failed => Err(AssignmentError::Failed(
                self.failure_reason.clone().unwrap_or_default(),
            )),
            _ => Ok(()),
        }
    }

    fn check_owner(&self, member: &str, epoch: u64) -> Result<(), AssignmentError> {
        self.check_live()?;
        if self.owner_member.as_deref() != Some(member) || self.ownership_epoch != epoch {
            return Err(AssignmentError::NotOwner);
        }
        Ok(())
    }

    /// Assign a fresh generation to a replica. The launch owner holds epoch 1,
    /// which the runtime also starts with.
    pub fn assign_launch(
        &self,
        member: &str,
        now_ms: i64,
        timing: &AssignmentTiming,
    ) -> Result<Self, AssignmentError> {
        self.check_live()?;
        if self.state != AssignmentState::Pending {
            return Err(AssignmentError::WrongState);
        }
        let mut next = self.clone();
        next.state = AssignmentState::Launching;
        next.ownership_epoch = 1;
        next.owner_member = Some(member.to_string());
        next.lease_expires_at_ms = now_ms + timing.launch_grace_ms;
        Ok(next)
    }

    /// Renew the owner's lease. The first renewal binds the supervisor
    /// instance; a lapsed lease is never revived because the runtime and the
    /// owner both treat lapse as loss of authority.
    pub fn renew(
        &self,
        member: &str,
        epoch: u64,
        instance: &str,
        now_ms: i64,
        timing: &AssignmentTiming,
    ) -> Result<Self, AssignmentError> {
        self.check_owner(member, epoch)?;
        if !matches!(
            self.state,
            AssignmentState::Launching | AssignmentState::Active
        ) {
            return Err(AssignmentError::WrongState);
        }
        if now_ms >= self.lease_expires_at_ms {
            return Err(AssignmentError::LeaseExpired);
        }
        if self
            .owner_instance
            .as_deref()
            .is_some_and(|bound| bound != instance)
        {
            return Err(AssignmentError::InstanceMismatch);
        }
        let mut next = self.clone();
        next.state = AssignmentState::Active;
        next.owner_instance = Some(instance.to_string());
        next.lease_expires_at_ms = now_ms + timing.lease_ttl_ms;
        Ok(next)
    }

    /// Reserve the next epoch for a candidate after the owner's lease lapsed,
    /// or after a previous candidate failed to activate in its attempt
    /// budget. The recovery deadline is anchored to the original lease
    /// expiry and never extended by a later candidate.
    pub fn reserve_recovery(
        &self,
        candidate: &str,
        now_ms: i64,
        timing: &AssignmentTiming,
    ) -> Result<Self, AssignmentError> {
        self.check_live()?;
        let deadline = match self.state {
            AssignmentState::Active => {
                if now_ms < self.lease_expires_at_ms {
                    return Err(AssignmentError::LeaseValid);
                }
                self.lease_expires_at_ms + timing.recovery_budget_ms
            }
            AssignmentState::Recovering => {
                if self
                    .attempt_expires_at_ms
                    .is_some_and(|expires| now_ms < expires)
                {
                    return Err(AssignmentError::LeaseValid);
                }
                self.recovery_deadline_ms
                    .ok_or(AssignmentError::WrongState)?
            }
            _ => return Err(AssignmentError::WrongState),
        };
        if now_ms >= deadline {
            return Err(AssignmentError::DeadlinePassed);
        }
        let mut next = self.clone();
        next.state = AssignmentState::Recovering;
        next.ownership_epoch += 1;
        next.owner_member = Some(candidate.to_string());
        next.owner_instance = None;
        next.grant_id = None;
        next.recovery_deadline_ms = Some(deadline);
        next.attempt_expires_at_ms = Some((now_ms + timing.attempt_budget_ms).min(deadline));
        Ok(next)
    }

    /// Record the single recovery grant issued for the reserved epoch.
    pub fn issue_grant(
        &self,
        member: &str,
        epoch: u64,
        grant_id: &str,
        now_ms: i64,
    ) -> Result<Self, AssignmentError> {
        self.check_owner(member, epoch)?;
        if self.state != AssignmentState::Recovering {
            return Err(AssignmentError::WrongState);
        }
        if self
            .recovery_deadline_ms
            .is_some_and(|deadline| now_ms >= deadline)
        {
            return Err(AssignmentError::DeadlinePassed);
        }
        if self.grant_id.is_some() {
            return Err(AssignmentError::GrantAlreadyIssued);
        }
        let mut next = self.clone();
        next.grant_id = Some(grant_id.to_string());
        Ok(next)
    }

    /// Activate a candidate that presented a verified runtime fencing receipt.
    pub fn activate(
        &self,
        member: &str,
        epoch: u64,
        instance: &str,
        now_ms: i64,
        timing: &AssignmentTiming,
    ) -> Result<Self, AssignmentError> {
        self.check_owner(member, epoch)?;
        if self.state != AssignmentState::Recovering {
            return Err(AssignmentError::WrongState);
        }
        if self
            .recovery_deadline_ms
            .is_some_and(|deadline| now_ms >= deadline)
        {
            return Err(AssignmentError::DeadlinePassed);
        }
        let mut next = self.clone();
        next.state = AssignmentState::Active;
        next.owner_instance = Some(instance.to_string());
        next.lease_expires_at_ms = now_ms + timing.lease_ttl_ms;
        next.recovery_deadline_ms = None;
        next.attempt_expires_at_ms = None;
        next.recoveries += 1;
        Ok(next)
    }

    /// Fail a recovery that missed its deadline, or a launch whose owner
    /// disappeared before it ever confirmed the boundary.
    #[must_use]
    pub fn fail_if_overdue(&self, now_ms: i64) -> Option<Self> {
        let reason = match self.state {
            AssignmentState::Recovering
                if self
                    .recovery_deadline_ms
                    .is_some_and(|deadline| now_ms >= deadline) =>
            {
                "supervisor recovery did not complete before its deadline"
            }
            AssignmentState::Launching if now_ms >= self.lease_expires_at_ms => {
                "launch supervisor lost before it confirmed the boundary"
            }
            _ => return None,
        };
        let mut next = self.clone();
        next.state = AssignmentState::Failed;
        next.failure_reason = Some(reason.to_string());
        Some(next)
    }

    /// Stop or delete wins over any pending launch or recovery.
    #[must_use]
    pub fn terminate(&self) -> Self {
        let mut next = self.clone();
        next.state = AssignmentState::Terminated;
        next
    }

    /// The owner's lease lapsed and nobody reserved recovery yet.
    #[must_use]
    pub fn needs_recovery(&self, now_ms: i64) -> bool {
        match self.state {
            AssignmentState::Active => now_ms >= self.lease_expires_at_ms,
            AssignmentState::Recovering => self
                .attempt_expires_at_ms
                .is_some_and(|expires| now_ms >= expires),
            _ => false,
        }
    }
}

/// Durable assignment storage with compare-and-swap writes.
#[derive(Clone, Debug)]
pub struct AssignmentStore {
    store: std::sync::Arc<Store>,
}

impl AssignmentStore {
    #[must_use]
    pub fn new(store: std::sync::Arc<Store>) -> Self {
        Self { store }
    }

    pub async fn get(&self, sandbox_id: &str) -> Result<Option<Assignment>, PersistenceError> {
        let Some(record) = self
            .store
            .get(ASSIGNMENT_OBJECT_TYPE, &assignment_object_id(sandbox_id))
            .await?
        else {
            return Ok(None);
        };
        let mut assignment: Assignment = serde_json::from_slice(&record.payload)
            .map_err(|error| PersistenceError::Decode(error.to_string()))?;
        assignment.resource_version = record.resource_version;
        Ok(Some(assignment))
    }

    pub async fn list(&self) -> Result<Vec<Assignment>, PersistenceError> {
        let mut assignments = Vec::new();
        let mut offset = 0;
        loop {
            let page = self
                .store
                .list_by_type(ASSIGNMENT_OBJECT_TYPE, 500, offset)
                .await?;
            let count = page.len();
            for record in page {
                let mut assignment: Assignment = serde_json::from_slice(&record.payload)
                    .map_err(|error| PersistenceError::Decode(error.to_string()))?;
                assignment.resource_version = record.resource_version;
                assignments.push(assignment);
            }
            if count < 500 {
                return Ok(assignments);
            }
            offset += 500;
        }
    }

    /// Write `next` only if the stored record still has `expected`'s version
    /// (or does not exist when `expected` is `None`).
    pub async fn write(
        &self,
        expected: Option<&Assignment>,
        next: &Assignment,
    ) -> Result<Assignment, PersistenceError> {
        let payload = serde_json::to_vec(next)
            .map_err(|error| PersistenceError::Encode(error.to_string()))?;
        let condition = expected.map_or(WriteCondition::MustCreate, |expected| {
            WriteCondition::MatchResourceVersion(expected.resource_version)
        });
        let result = self
            .store
            .put_if(
                ASSIGNMENT_OBJECT_TYPE,
                &assignment_object_id(&next.sandbox_id),
                &next.sandbox_id,
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

    pub async fn delete(&self, assignment: &Assignment) -> Result<bool, PersistenceError> {
        self.store
            .delete_if(
                ASSIGNMENT_OBJECT_TYPE,
                &assignment_object_id(&assignment.sandbox_id),
                assignment.resource_version,
            )
            .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const T: AssignmentTiming = AssignmentTiming::DEFAULT;

    fn active(now: i64) -> Assignment {
        Assignment::pending("sb", "name", "gen")
            .assign_launch("replica-a", now, &T)
            .unwrap()
            .renew("replica-a", 1, "instance-a", now, &T)
            .unwrap()
    }

    #[test]
    fn default_timing_fits_the_runtime_window() {
        T.validate().unwrap();
        let mut unsafe_timing = T;
        unsafe_timing.recovery_budget_ms = 25_000;
        assert!(unsafe_timing.validate().is_err());
    }

    #[test]
    fn launch_renew_binds_instance_and_rejects_lapsed_or_foreign_owners() {
        let assignment = active(0);
        assert_eq!(assignment.state, AssignmentState::Active);
        assert_eq!(assignment.ownership_epoch, 1);
        assert_eq!(
            assignment.renew("replica-b", 1, "instance-a", 1, &T),
            Err(AssignmentError::NotOwner)
        );
        assert_eq!(
            assignment.renew("replica-a", 1, "instance-other", 1, &T),
            Err(AssignmentError::InstanceMismatch)
        );
        assert_eq!(
            assignment.renew("replica-a", 1, "instance-a", T.lease_ttl_ms, &T),
            Err(AssignmentError::LeaseExpired),
            "a lapsed lease is never revived"
        );
    }

    #[test]
    fn recovery_requires_lapsed_lease_and_advances_epoch() {
        let assignment = active(0);
        assert_eq!(
            assignment.reserve_recovery("replica-b", 1, &T),
            Err(AssignmentError::LeaseValid)
        );
        let reserved = assignment
            .reserve_recovery("replica-b", T.lease_ttl_ms, &T)
            .unwrap();
        assert_eq!(reserved.state, AssignmentState::Recovering);
        assert_eq!(reserved.ownership_epoch, 2);
        assert_eq!(reserved.owner_member.as_deref(), Some("replica-b"));
        // The stale owner can no longer renew or claim the new epoch.
        assert_eq!(
            reserved.renew("replica-a", 1, "instance-a", T.lease_ttl_ms, &T),
            Err(AssignmentError::NotOwner)
        );
        assert_eq!(
            reserved.activate("replica-a", 2, "instance-a", T.lease_ttl_ms, &T),
            Err(AssignmentError::NotOwner)
        );
        let granted = reserved
            .issue_grant("replica-b", 2, "grant-1", T.lease_ttl_ms)
            .unwrap();
        assert_eq!(
            granted.issue_grant("replica-b", 2, "grant-2", T.lease_ttl_ms),
            Err(AssignmentError::GrantAlreadyIssued)
        );
        let activated = granted
            .activate("replica-b", 2, "instance-b", T.lease_ttl_ms + 1, &T)
            .unwrap();
        assert_eq!(activated.state, AssignmentState::Active);
        assert_eq!(activated.recoveries, 1);
    }

    #[test]
    fn failed_candidates_never_extend_the_recovery_deadline() {
        let reserved = active(0)
            .reserve_recovery("replica-b", T.lease_ttl_ms, &T)
            .unwrap();
        let deadline = reserved.recovery_deadline_ms.unwrap();
        assert_eq!(deadline, T.lease_ttl_ms + T.recovery_budget_ms);
        let attempt = reserved.attempt_expires_at_ms.unwrap();
        assert_eq!(
            reserved.reserve_recovery("replica-c", attempt - 1, &T),
            Err(AssignmentError::LeaseValid)
        );
        let retried = reserved.reserve_recovery("replica-c", attempt, &T).unwrap();
        assert_eq!(retried.ownership_epoch, 3);
        assert_eq!(retried.recovery_deadline_ms, Some(deadline));
        assert!(retried.attempt_expires_at_ms.unwrap() <= deadline);
        let failed = retried.fail_if_overdue(deadline).unwrap();
        assert_eq!(failed.state, AssignmentState::Failed);
        assert!(matches!(
            failed.activate("replica-c", 3, "instance-c", deadline, &T),
            Err(AssignmentError::Failed(_))
        ));
    }

    #[test]
    fn terminate_wins_over_launch_and_recovery() {
        let reserved = active(0)
            .reserve_recovery("replica-b", T.lease_ttl_ms, &T)
            .unwrap();
        let terminated = reserved.terminate();
        assert_eq!(
            terminated.activate("replica-b", 2, "instance-b", T.lease_ttl_ms, &T),
            Err(AssignmentError::Terminated)
        );
        assert_eq!(
            terminated.issue_grant("replica-b", 2, "grant", T.lease_ttl_ms),
            Err(AssignmentError::Terminated)
        );
        assert_eq!(
            terminated.reserve_recovery("replica-c", i64::MAX / 2, &T),
            Err(AssignmentError::Terminated)
        );
    }

    #[tokio::test]
    async fn concurrent_reservations_cannot_both_win() {
        let store = std::sync::Arc::new(crate::persistence::test_store().await);
        let assignments = AssignmentStore::new(store);
        let created = assignments
            .write(None, &Assignment::pending("sb-race", "race", "gen"))
            .await
            .unwrap();
        let launched = assignments
            .write(
                Some(&created),
                &created.assign_launch("replica-a", 0, &T).unwrap(),
            )
            .await
            .unwrap();
        let active = assignments
            .write(
                Some(&launched),
                &launched.renew("replica-a", 1, "instance-a", 0, &T).unwrap(),
            )
            .await
            .unwrap();
        // Two gateways observe the same lapsed lease and pick different candidates.
        let now = T.lease_ttl_ms;
        let first = active.reserve_recovery("replica-b", now, &T).unwrap();
        let second = active.reserve_recovery("replica-c", now, &T).unwrap();
        let first_write = assignments.write(Some(&active), &first).await;
        let second_write = assignments.write(Some(&active), &second).await;
        assert!(first_write.is_ok());
        assert!(matches!(
            second_write,
            Err(PersistenceError::Conflict { .. })
        ));
        let stored = assignments.get("sb-race").await.unwrap().unwrap();
        assert_eq!(stored.owner_member.as_deref(), Some("replica-b"));
        assert_eq!(stored.ownership_epoch, 2);
    }
}
