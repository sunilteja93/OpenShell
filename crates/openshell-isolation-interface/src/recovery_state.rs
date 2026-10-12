// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Supervisor enforcement state that must survive a fenced supervisor replacement.
//!
//! Trust-on-first-use executable pins are learned by the supervisor that owns a
//! boundary. A replacement must restore them from runtime-held state instead
//! of relearning them from a workload that kept running. A durable journal
//! makes a newly learned pin authoritative only after the backend reports it
//! checkpointed, so a crash cannot lose a pin that already authorized traffic.

use std::collections::BTreeMap;
use std::sync::Mutex;
use std::time::Duration;

/// Shared journal between the network supervisor and a recovery-capable backend.
#[derive(Debug)]
pub struct EnforcementPinJournal {
    pins: Mutex<BTreeMap<String, String>>,
    learned: tokio::sync::watch::Sender<u64>,
    durable: tokio::sync::watch::Sender<u64>,
}

impl EnforcementPinJournal {
    /// Start a journal from pins restored by a recovery handshake (or empty).
    #[must_use]
    pub fn new(restored: BTreeMap<String, String>) -> Self {
        Self {
            pins: Mutex::new(restored),
            learned: tokio::sync::watch::channel(0).0,
            durable: tokio::sync::watch::channel(0).0,
        }
    }

    /// Replace the pin set with pins restored from authenticated runtime state.
    /// Only valid before any network mediation consumed the journal.
    pub fn restore(&self, restored: BTreeMap<String, String>) {
        *self
            .pins
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = restored;
    }

    /// Pins currently known to the journal.
    #[must_use]
    pub fn pins(&self) -> BTreeMap<String, String> {
        self.pins
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }

    /// Record newly learned pins and return the revision that must become
    /// durable before they authorize traffic.
    pub fn record(&self, new_pins: impl IntoIterator<Item = (String, String)>) -> u64 {
        let mut pins = self
            .pins
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut changed = false;
        for (path, digest) in new_pins {
            if pins.insert(path, digest).is_none() {
                changed = true;
            }
        }
        drop(pins);
        if changed {
            self.learned.send_modify(|revision| *revision += 1);
        }
        *self.learned.borrow()
    }

    /// Snapshot for checkpointing: the learned revision and its pins.
    #[must_use]
    pub fn snapshot(&self) -> (u64, BTreeMap<String, String>) {
        let pins = self
            .pins
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        (*self.learned.borrow(), pins.clone())
    }

    /// Wake when newly learned pins need a checkpoint.
    #[must_use]
    pub fn subscribe_learned(&self) -> tokio::sync::watch::Receiver<u64> {
        self.learned.subscribe()
    }

    /// Mark a learned revision checkpointed by the backend.
    pub fn mark_durable(&self, revision: u64) {
        self.durable.send_if_modified(|durable| {
            if revision > *durable {
                *durable = revision;
                true
            } else {
                false
            }
        });
    }

    /// Wait until `revision` is durable. Returns `false` on timeout, in which
    /// case the caller must not authorize traffic with the new pin.
    pub async fn wait_durable(&self, revision: u64, timeout: Duration) -> bool {
        let mut durable = self.durable.subscribe();
        tokio::time::timeout(timeout, durable.wait_for(|value| *value >= revision))
            .await
            .is_ok_and(|result| result.is_ok())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn only_new_pins_require_a_checkpoint() {
        let journal =
            EnforcementPinJournal::new(BTreeMap::from([("/bin/a".to_string(), "aa".to_string())]));
        assert_eq!(
            journal.record([("/bin/a".to_string(), "aa".to_string())]),
            0
        );
        assert!(journal.wait_durable(0, Duration::from_millis(1)).await);
        let revision = journal.record([("/bin/b".to_string(), "bb".to_string())]);
        assert_eq!(revision, 1);
        assert!(
            !journal
                .wait_durable(revision, Duration::from_millis(10))
                .await
        );
        let (snapshot_revision, pins) = journal.snapshot();
        assert_eq!(snapshot_revision, 1);
        assert_eq!(pins.len(), 2);
        journal.mark_durable(snapshot_revision);
        assert!(
            journal
                .wait_durable(revision, Duration::from_millis(10))
                .await
        );
    }
}
