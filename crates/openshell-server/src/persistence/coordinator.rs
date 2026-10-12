// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Elected coordinator leases and fenced multi-object writes.
//!
//! A coordinator lease is one row in the `coordinator_leases` table per lease
//! name: the holder (a unique per-process identity), a monotonic term, and a
//! persisted expiry deadline. Rows are never deleted, so a term is never
//! reused.
//!
//! [`Store::write_fenced`] applies compare-and-swap writes in the same
//! database transaction that verifies the writer still holds an UNEXPIRED
//! lease at its term. A takeover updates the lease row, which serializes
//! against in-flight fenced writes: a coordinator that pauses after deciding
//! and resumes after its lease expired, or after another process took over,
//! is rejected with [`PersistenceError::Fenced`], and none of its writes apply.
//!
//! Clock: every call takes `now_ms` explicitly (gateways pass
//! `openshell_core::time::now_ms`; tests pass synthetic times anchored to the
//! real epoch). Inside the transaction, after the lease row lock is granted,
//! expiry is judged against `max(now_ms, database clock)`. A caller timestamp
//! captured before a pause can therefore never make an expired lease valid;
//! a caller clock running ahead can only expire a lease early, which is safe.
//! Assumption: the database clock and the gateway clocks are synchronized
//! (NTP) well within the lease TTL; a database clock far behind real time
//! weakens the bound back to the caller's clock.

use super::{PersistenceError, PersistenceResult, Store, WriteCondition, WriteResult};

/// Observed state of a coordinator lease row.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CoordinatorLease {
    pub holder: String,
    pub term: u64,
    pub expires_at_ms: i64,
}

impl CoordinatorLease {
    #[must_use]
    pub const fn expired(&self, now_ms: i64) -> bool {
        self.expires_at_ms <= now_ms
    }
}

/// Proof a writer must present for a fenced write.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CoordinatorFence {
    pub lease: String,
    pub holder: String,
    pub term: u64,
}

/// One write inside a fenced transaction. Unconditional writes are not
/// allowed: every coordinator decision is also guarded by the target's CAS.
#[derive(Clone, Debug)]
pub enum FencedOp {
    Put {
        object_type: String,
        id: String,
        name: String,
        workspace: String,
        payload: Vec<u8>,
        labels: Option<String>,
        condition: WriteCondition,
    },
    Delete {
        object_type: String,
        id: String,
        resource_version: u64,
    },
}

impl FencedOp {
    pub(super) fn validate(ops: &[Self]) -> PersistenceResult<()> {
        if ops.iter().any(|op| {
            matches!(
                op,
                Self::Put {
                    condition: WriteCondition::Unconditional,
                    ..
                }
            )
        }) {
            return Err(PersistenceError::Encode(
                "fenced writes must be conditional".to_string(),
            ));
        }
        Ok(())
    }
}

/// Outcome of one fenced op: the new version of a put, or `None` for a delete.
pub type FencedResult = Option<WriteResult>;

impl Store {
    /// Acquire `lease` for `holder` if it is free or expired. Returns the new
    /// lease (with a strictly higher term on takeover), or `None` while
    /// another holder's lease is valid.
    pub async fn acquire_coordinator_lease(
        &self,
        lease: &str,
        holder: &str,
        now_ms: i64,
        ttl_ms: i64,
    ) -> PersistenceResult<Option<CoordinatorLease>> {
        match self {
            Self::Postgres(store) => {
                store
                    .acquire_coordinator_lease(lease, holder, now_ms, ttl_ms)
                    .await
            }
            Self::Sqlite(store) => {
                store
                    .acquire_coordinator_lease(lease, holder, now_ms, ttl_ms)
                    .await
            }
        }
    }

    /// Extend a valid lease held by `fence` to `now_ms + ttl_ms`. Returns
    /// `false` once the lease expired or another process took it over.
    pub async fn renew_coordinator_lease(
        &self,
        fence: &CoordinatorFence,
        now_ms: i64,
        ttl_ms: i64,
    ) -> PersistenceResult<bool> {
        match self {
            Self::Postgres(store) => store.renew_coordinator_lease(fence, now_ms, ttl_ms).await,
            Self::Sqlite(store) => store.renew_coordinator_lease(fence, now_ms, ttl_ms).await,
        }
    }

    /// Give up a held lease so another process can take over immediately.
    /// The row is expired in place, never deleted, so terms keep increasing.
    pub async fn release_coordinator_lease(
        &self,
        fence: &CoordinatorFence,
        now_ms: i64,
    ) -> PersistenceResult<bool> {
        match self {
            Self::Postgres(store) => store.release_coordinator_lease(fence, now_ms).await,
            Self::Sqlite(store) => store.release_coordinator_lease(fence, now_ms).await,
        }
    }

    pub async fn get_coordinator_lease(
        &self,
        lease: &str,
    ) -> PersistenceResult<Option<CoordinatorLease>> {
        match self {
            Self::Postgres(store) => store.get_coordinator_lease(lease).await,
            Self::Sqlite(store) => store.get_coordinator_lease(lease).await,
        }
    }

    /// Apply `ops` atomically, and only while `fence` still holds its term
    /// and the lease has not expired at `now_ms`.
    ///
    /// Returns [`PersistenceError::Fenced`] if the term was lost or expired and
    /// [`PersistenceError::Conflict`] if any op's own condition fails; in both
    /// cases nothing is written.
    pub async fn write_fenced(
        &self,
        fence: &CoordinatorFence,
        now_ms: i64,
        ops: &[FencedOp],
    ) -> PersistenceResult<Vec<FencedResult>> {
        FencedOp::validate(ops)?;
        match self {
            Self::Postgres(store) => store.write_fenced(fence, now_ms, ops).await,
            Self::Sqlite(store) => store.write_fenced(fence, now_ms, ops).await,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};

    /// Long enough that a scenario anchored at the real clock finishes well
    /// inside one lease, since the database clock is a lower bound. The
    /// database samples its clock slightly after `t0`, so "expired" checks
    /// use `t0 + 2 * TTL` rather than the exact boundary.
    const TTL: i64 = 60_000;
    const OBJECT_TYPE: &str = "coordinator_test_object";

    fn fence(lease: &CoordinatorLease, name: &str) -> CoordinatorFence {
        CoordinatorFence {
            lease: name.to_string(),
            holder: lease.holder.clone(),
            term: lease.term,
        }
    }

    fn put(id: &str, payload: &str, condition: WriteCondition) -> FencedOp {
        FencedOp::Put {
            object_type: OBJECT_TYPE.to_string(),
            id: id.to_string(),
            name: id.to_string(),
            workspace: String::new(),
            payload: payload.as_bytes().to_vec(),
            labels: None,
            condition,
        }
    }

    async fn payload(store: &Store, id: &str) -> Option<(String, u64)> {
        store.get(OBJECT_TYPE, id).await.unwrap().map(|record| {
            (
                String::from_utf8(record.payload).unwrap(),
                record.resource_version,
            )
        })
    }

    /// Exactly one of several competing processes holds the lease; others get
    /// it only after expiry or release, always with a strictly higher term.
    async fn election_and_takeover(store: &Store, lease: &str) {
        let t0 = openshell_core::time::now_ms();
        let first = store
            .acquire_coordinator_lease(lease, "gw-a/1", t0, TTL)
            .await
            .unwrap()
            .expect("first acquirer wins");
        assert_eq!(first.term, 1);
        for contender in ["gw-b/1", "gw-c/1"] {
            assert!(
                store
                    .acquire_coordinator_lease(lease, contender, t0 + 1_000, TTL)
                    .await
                    .unwrap()
                    .is_none(),
                "a valid lease has exactly one holder"
            );
        }
        // Renewal keeps the term and moves the deadline.
        assert!(
            store
                .renew_coordinator_lease(&fence(&first, lease), t0 + 40_000, TTL)
                .await
                .unwrap()
        );
        assert!(
            store
                .acquire_coordinator_lease(lease, "gw-b/1", t0 + 99_000, TTL)
                .await
                .unwrap()
                .is_none(),
            "renewal extended the lease to t0 + 100000"
        );
        let second = store
            .acquire_coordinator_lease(lease, "gw-b/1", t0 + 100_000, TTL)
            .await
            .unwrap()
            .expect("expired lease is taken over");
        assert_eq!(second.term, 2);
        assert!(
            !store
                .renew_coordinator_lease(&fence(&first, lease), t0 + 100_001, TTL)
                .await
                .unwrap(),
            "the previous holder cannot renew after a takeover"
        );
        // Graceful release lets a standby take over at once, with a new term.
        assert!(
            store
                .release_coordinator_lease(&fence(&second, lease), t0 + 100_500)
                .await
                .unwrap()
        );
        let third = store
            .acquire_coordinator_lease(lease, "gw-c/1", t0 + 100_600, TTL)
            .await
            .unwrap()
            .expect("released lease is free");
        assert_eq!(third.term, 3, "terms never repeat");
        assert_eq!(
            store.get_coordinator_lease(lease).await.unwrap(),
            Some(third)
        );
    }

    /// A coordinator observes state, pauses, and resumes after a takeover:
    /// its decision is rejected atomically and nothing in the batch applies.
    async fn paused_coordinator_is_fenced(store: &Store, lease: &str, id: &str) {
        let t0 = openshell_core::time::now_ms();
        let old = store
            .acquire_coordinator_lease(lease, "gw-a/1", t0, TTL)
            .await
            .unwrap()
            .unwrap();
        let created = store
            .write_fenced(
                &fence(&old, lease),
                t0 + 10,
                &[put(id, "v1", WriteCondition::MustCreate)],
            )
            .await
            .unwrap();
        let observed_version = created[0].as_ref().unwrap().resource_version;

        // -- pause: the old coordinator stops here after reading v1 --
        let new = store
            .acquire_coordinator_lease(lease, "gw-b/1", t0 + 2 * TTL, TTL)
            .await
            .unwrap()
            .expect("standby takes over after expiry");

        let stale = store
            .write_fenced(
                &fence(&old, lease),
                t0 + 2 * TTL + 1,
                &[
                    put(
                        id,
                        "stale",
                        WriteCondition::MatchResourceVersion(observed_version),
                    ),
                    put(&format!("{id}-extra"), "stale", WriteCondition::MustCreate),
                ],
            )
            .await;
        assert!(matches!(stale, Err(PersistenceError::Fenced)));
        assert_eq!(payload(store, id).await.unwrap().0, "v1");
        assert!(payload(store, &format!("{id}-extra")).await.is_none());

        store
            .write_fenced(
                &fence(&new, lease),
                t0 + 2 * TTL + 2,
                &[put(
                    id,
                    "v2",
                    WriteCondition::MatchResourceVersion(observed_version),
                )],
            )
            .await
            .unwrap();
        assert_eq!(payload(store, id).await.unwrap().0, "v2");
    }

    /// A current-but-expired term authorizes nothing, even with no standby.
    async fn expired_term_without_takeover_is_fenced(store: &Store, lease: &str, id: &str) {
        let t0 = openshell_core::time::now_ms();
        let held = store
            .acquire_coordinator_lease(lease, "gw-a/1", t0, TTL)
            .await
            .unwrap()
            .unwrap();
        assert!(
            store
                .write_fenced(
                    &fence(&held, lease),
                    t0 + TTL - 1,
                    &[put(id, "inside", WriteCondition::MustCreate)],
                )
                .await
                .is_ok()
        );
        let version = payload(store, id).await.unwrap().1;
        let expired = store
            .write_fenced(
                &fence(&held, lease),
                t0 + 2 * TTL,
                &[put(
                    id,
                    "after-expiry",
                    WriteCondition::MatchResourceVersion(version),
                )],
            )
            .await;
        assert!(matches!(expired, Err(PersistenceError::Fenced)));
        assert_eq!(payload(store, id).await.unwrap().0, "inside");
        assert!(
            !store
                .renew_coordinator_lease(&fence(&held, lease), t0 + 2 * TTL, TTL)
                .await
                .unwrap(),
            "an expired lease is not revived by renewal"
        );
        assert_eq!(
            store
                .get_coordinator_lease(lease)
                .await
                .unwrap()
                .unwrap()
                .term,
            1,
            "no takeover happened"
        );
    }

    /// A timestamp captured before a pause must not make a lease that has
    /// expired by the database clock look valid.
    async fn stale_caller_clock_cannot_revive_an_expired_lease(
        store: &Store,
        lease: &str,
        id: &str,
    ) {
        // The lease was taken (and expired) in the past by real time.
        let past = openshell_core::time::now_ms() - 10 * TTL;
        let held = store
            .acquire_coordinator_lease(lease, "gw-a/1", past, TTL)
            .await
            .unwrap()
            .unwrap();
        assert!(
            held.expires_at_ms > openshell_core::time::now_ms(),
            "acquisition uses the database clock as a lower bound"
        );
        // Force the deadline into the past, as if real time passed a pause.
        store
            .release_coordinator_lease(&fence(&held, lease), past)
            .await
            .unwrap();
        let stale = store
            .write_fenced(
                &fence(&held, lease),
                past + 1,
                &[put(id, "stale-clock", WriteCondition::MustCreate)],
            )
            .await;
        assert!(matches!(stale, Err(PersistenceError::Fenced)));
        assert!(payload(store, id).await.is_none());
        assert!(
            !store
                .renew_coordinator_lease(&fence(&held, lease), past + 1, TTL)
                .await
                .unwrap(),
            "a stale caller clock cannot renew an expired lease"
        );
        // A takeover with a stale clock is judged by the database clock too.
        let taken = store
            .acquire_coordinator_lease(lease, "gw-b/1", past + 2, TTL)
            .await
            .unwrap()
            .expect("expired lease is free");
        assert_eq!(taken.term, 2);
        assert!(taken.expires_at_ms > openshell_core::time::now_ms());
    }

    /// One failing op aborts the whole fenced batch.
    async fn batch_is_all_or_nothing(store: &Store, lease: &str, id: &str) {
        let t0 = openshell_core::time::now_ms();
        let held = store
            .acquire_coordinator_lease(lease, "gw-a/1", t0, TTL)
            .await
            .unwrap()
            .unwrap();
        let result = store
            .write_fenced(
                &fence(&held, lease),
                t0 + 1,
                &[
                    put(id, "first", WriteCondition::MustCreate),
                    put("missing", "x", WriteCondition::MatchResourceVersion(7)),
                ],
            )
            .await;
        assert!(matches!(result, Err(PersistenceError::Conflict { .. })));
        assert!(payload(store, id).await.is_none());
        assert!(matches!(
            store
                .write_fenced(
                    &fence(&held, lease),
                    t0 + 1,
                    &[put(id, "x", WriteCondition::Unconditional)],
                )
                .await,
            Err(PersistenceError::Encode(_))
        ));
    }

    /// The old coordinator keeps issuing CAS writes while a standby takes
    /// over. After the standby's first write, no stale write ever lands.
    async fn no_stale_write_lands_after_takeover(store: Arc<Store>, lease: &str, id: &str) {
        let t0 = openshell_core::time::now_ms();
        let old = store
            .acquire_coordinator_lease(lease, "gw-a/1", t0, TTL)
            .await
            .unwrap()
            .unwrap();
        store
            .write_fenced(
                &fence(&old, lease),
                t0,
                &[put(id, "old-0", WriteCondition::MustCreate)],
            )
            .await
            .unwrap();
        let stop = Arc::new(AtomicBool::new(false));
        // The old coordinator's clock never advances: it believes its lease
        // is still valid throughout.
        let old_clock = Arc::new(AtomicI64::new(t0));
        let writer = {
            let (store, stop, old_clock, id) = (
                store.clone(),
                stop.clone(),
                old_clock.clone(),
                id.to_string(),
            );
            let fence = fence(&old, lease);
            tokio::spawn(async move {
                let mut n = 0;
                while !stop.load(Ordering::SeqCst) {
                    n += 1;
                    let Some((_, version)) = payload(&store, &id).await else {
                        continue;
                    };
                    let _ = store
                        .write_fenced(
                            &fence,
                            old_clock.load(Ordering::SeqCst),
                            &[put(
                                &id,
                                &format!("old-{n}"),
                                WriteCondition::MatchResourceVersion(version),
                            )],
                        )
                        .await;
                }
            })
        };
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        let new = store
            .acquire_coordinator_lease(lease, "gw-b/1", t0 + 2 * TTL, TTL)
            .await
            .unwrap()
            .expect("takeover");
        let mut written = false;
        for _ in 0..50 {
            let (_, version) = payload(&store, id).await.unwrap();
            if store
                .write_fenced(
                    &fence(&new, lease),
                    t0 + 2 * TTL + 1,
                    &[put(
                        id,
                        "new",
                        WriteCondition::MatchResourceVersion(version),
                    )],
                )
                .await
                .is_ok()
            {
                written = true;
                break;
            }
        }
        assert!(
            written,
            "stale writes kept invalidating the new coordinator"
        );
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        stop.store(true, Ordering::SeqCst);
        writer.await.unwrap();
        assert_eq!(
            payload(&store, id).await.unwrap().0,
            "new",
            "a stale coordinator write landed after the takeover"
        );
    }

    async fn run_all(store: Arc<Store>, prefix: &str) {
        election_and_takeover(&store, &format!("{prefix}-election")).await;
        paused_coordinator_is_fenced(&store, &format!("{prefix}-pause"), &format!("{prefix}-p"))
            .await;
        expired_term_without_takeover_is_fenced(
            &store,
            &format!("{prefix}-expiry"),
            &format!("{prefix}-e"),
        )
        .await;
        stale_caller_clock_cannot_revive_an_expired_lease(
            &store,
            &format!("{prefix}-stale-clock"),
            &format!("{prefix}-s"),
        )
        .await;
        batch_is_all_or_nothing(&store, &format!("{prefix}-batch"), &format!("{prefix}-b")).await;
        no_stale_write_lands_after_takeover(
            store.clone(),
            &format!("{prefix}-race"),
            &format!("{prefix}-r"),
        )
        .await;
    }

    #[tokio::test]
    async fn sqlite_coordinator_lease_fences_stale_and_expired_writers() {
        run_all(Arc::new(crate::persistence::test_store().await), "sqlite").await;
    }

    /// PostgreSQL-only: the lease row is locked (but never changed) by another
    /// transaction; the lease expires while a fenced write and a renewal wait
    /// for the lock. After the lock is released both must be rejected: the
    /// expiry check uses a clock sampled after the lock wait, not before.
    async fn lock_wait_judges_expiry_after_the_wait(store: &Store, lease: &str, id: &str) {
        const SHORT_TTL: i64 = 1_500;
        let Store::Postgres(postgres) = store else {
            panic!("PostgreSQL store required");
        };
        let t0 = openshell_core::time::now_ms();
        let held = store
            .acquire_coordinator_lease(lease, "gw-a/1", t0, SHORT_TTL)
            .await
            .unwrap()
            .unwrap();
        let blocker = postgres.lock_coordinator_lease_row(lease).await.unwrap();
        let write = {
            let (store, fence) = (store.clone(), fence(&held, lease));
            let id = id.to_string();
            // The caller's clock is captured before the wait and stays stale.
            tokio::spawn(async move {
                store
                    .write_fenced(
                        &fence,
                        t0,
                        &[put(&id, "after-wait", WriteCondition::MustCreate)],
                    )
                    .await
            })
        };
        let renew = {
            let (store, fence) = (store.clone(), fence(&held, lease));
            tokio::spawn(async move { store.renew_coordinator_lease(&fence, t0, SHORT_TTL).await })
        };
        tokio::time::sleep(std::time::Duration::from_millis(
            u64::try_from(2 * SHORT_TTL).unwrap(),
        ))
        .await;
        assert!(
            !write.is_finished(),
            "fenced write must wait for the row lock"
        );
        assert!(!renew.is_finished(), "renewal must wait for the row lock");
        blocker.rollback().await.unwrap();
        assert!(matches!(
            write.await.unwrap(),
            Err(PersistenceError::Fenced)
        ));
        assert!(
            !renew.await.unwrap().unwrap(),
            "an expired lease is not renewed"
        );
        assert!(payload(store, id).await.is_none());
        assert_eq!(
            store
                .get_coordinator_lease(lease)
                .await
                .unwrap()
                .unwrap()
                .term,
            held.term
        );
    }

    #[tokio::test]
    #[ignore = "requires a disposable PostgreSQL database in OPENSHELL_COORDINATOR_TEST_DATABASE_URL"]
    async fn postgres_coordinator_lease_fences_stale_and_expired_writers() {
        let url = std::env::var("OPENSHELL_COORDINATOR_TEST_DATABASE_URL")
            .expect("disposable database URL");
        let store = Arc::new(Store::connect(&url).await.unwrap());
        let prefix = format!("pg-{}", uuid::Uuid::new_v4());
        run_all(store.clone(), &prefix).await;
        lock_wait_judges_expiry_after_the_wait(
            &store,
            &format!("{prefix}-lock-wait"),
            &format!("{prefix}-w"),
        )
        .await;
        store.close().await;
    }
}
