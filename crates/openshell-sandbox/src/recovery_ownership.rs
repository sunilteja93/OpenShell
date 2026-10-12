// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Runtime-side state for the `supervisor-recovery.v1` Sandbox Protocol extension.
//!
//! The runtime is the last line of fencing: a database row cannot close a
//! socket. This state machine accepts a replacement supervisor only with
//! gateway-signed authority for a strictly higher ownership epoch, a fresh
//! runtime challenge, and a proof that the candidate holds the key named by
//! its grant. A prepared candidate cannot resume the workload until the
//! gateway activates the exact receipt the runtime signed, and the active
//! owner must keep renewing an enforced lease.

use std::collections::{HashSet, VecDeque};
use std::time::{Duration, Instant};

use openshell_core::jwt::{
    RECOVERY_PROOF_JWT_TYPE, RecoveryKeyProof, SupervisorAuthorityKind,
    VerifiedSupervisorAuthority, verify_ed25519_document,
};
use openshell_sandbox_backend::boundary_protocol::{
    MAX_RECOVERY_STATE_PINS, SupervisorRecoveryStateWire,
};

/// A challenge is useful only for one grant round trip through the gateway.
pub const RECOVERY_CHALLENGE_TTL: Duration = Duration::from_mins(1);
/// Bound outstanding challenges so authenticated callers cannot grow state.
const MAX_OUTSTANDING_CHALLENGES: usize = 8;
/// Bound remembered grant IDs; epochs are monotonic, so older grants are
/// rejected by epoch anyway.
const MAX_USED_GRANTS: usize = 64;

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum RecoveryError {
    #[error("recovery authority has the wrong purpose")]
    WrongKind,
    #[error("recovery epoch {offered} does not advance current ownership epoch {current}")]
    StaleEpoch { offered: u64, current: u64 },
    #[error("recovery challenge is unknown, expired, or already used")]
    UnknownChallenge,
    #[error("recovery grant was already used")]
    GrantReplayed,
    #[error("recovery grant does not name a candidate key")]
    MissingCandidateKey,
    #[error("candidate key proof is invalid")]
    InvalidProof,
    #[error("no recovery is prepared")]
    NotPrepared,
    #[error("activation does not match the prepared recovery")]
    ActivationMismatch,
    #[error("recovery is prepared; activation is required before confirm")]
    ActivationRequired,
    #[error("lease authority does not match the current owner")]
    LeaseMismatch,
    #[error("recovery state revision does not advance")]
    StaleState,
    #[error("recovery state exceeds {MAX_RECOVERY_STATE_PINS} pins")]
    StateTooLarge,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct PreparedRecovery {
    epoch: u64,
    supervisor_instance: String,
    receipt_digest: Option<String>,
}

/// Facts the runtime puts into its signed fencing receipt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FencedOwnership {
    pub previous_epoch: u64,
    pub epoch: u64,
    pub supervisor_instance: String,
    pub challenge: String,
    pub grant_id: String,
}

#[derive(Debug)]
pub struct RecoveryOwnership {
    epoch: u64,
    lease_ttl: Duration,
    lease_deadline: Option<Instant>,
    challenges: VecDeque<(String, Instant)>,
    used_grants: VecDeque<String>,
    used_grant_set: HashSet<String>,
    prepared: Option<PreparedRecovery>,
    state: Option<SupervisorRecoveryStateWire>,
}

impl RecoveryOwnership {
    /// The launch owner holds epoch 1. Every replacement must advance it.
    #[must_use]
    pub fn new(lease_ttl: Duration) -> Self {
        Self {
            epoch: 1,
            lease_ttl,
            lease_deadline: None,
            challenges: VecDeque::new(),
            used_grants: VecDeque::new(),
            used_grant_set: HashSet::new(),
            prepared: None,
            state: None,
        }
    }

    #[must_use]
    pub const fn epoch(&self) -> u64 {
        self.epoch
    }

    /// Issue a fresh, single-use challenge for one candidate's grant request.
    pub fn issue_challenge(&mut self, now: Instant) -> String {
        self.challenges
            .retain(|(_, issued)| now.duration_since(*issued) < RECOVERY_CHALLENGE_TTL);
        while self.challenges.len() >= MAX_OUTSTANDING_CHALLENGES {
            self.challenges.pop_front();
        }
        let challenge = uuid::Uuid::new_v4().to_string();
        self.challenges.push_back((challenge.clone(), now));
        challenge
    }

    /// Validate a verified recovery grant and the candidate's key proof, then
    /// advance the ownership epoch. The caller fences connections and keeps the
    /// workload frozen while recovery is prepared.
    pub fn begin_recovery(
        &mut self,
        grant: &VerifiedSupervisorAuthority,
        proof: &str,
        now: Instant,
    ) -> Result<FencedOwnership, RecoveryError> {
        let authority = &grant.authority;
        if authority.kind != SupervisorAuthorityKind::RecoveryGrant {
            return Err(RecoveryError::WrongKind);
        }
        if authority.ownership_epoch <= self.epoch {
            return Err(RecoveryError::StaleEpoch {
                offered: authority.ownership_epoch,
                current: self.epoch,
            });
        }
        let grant_id = grant.token_id.to_string();
        if self.used_grant_set.contains(&grant_id) {
            return Err(RecoveryError::GrantReplayed);
        }
        let challenge = authority
            .challenge
            .as_deref()
            .ok_or(RecoveryError::UnknownChallenge)?;
        let challenge_index = self
            .challenges
            .iter()
            .position(|(value, issued)| {
                value == challenge && now.duration_since(*issued) < RECOVERY_CHALLENGE_TTL
            })
            .ok_or(RecoveryError::UnknownChallenge)?;
        let key_pem = authority
            .candidate_key_pem
            .as_deref()
            .ok_or(RecoveryError::MissingCandidateKey)?;
        let proof: RecoveryKeyProof =
            verify_ed25519_document(key_pem.as_bytes(), RECOVERY_PROOF_JWT_TYPE, proof)
                .map_err(|_| RecoveryError::InvalidProof)?;
        if proof.challenge != challenge
            || proof.grant_id != grant_id
            || proof.supervisor_instance != authority.supervisor_instance
        {
            return Err(RecoveryError::InvalidProof);
        }

        // Every check passed; commit atomically.
        self.challenges.remove(challenge_index);
        self.used_grant_set.insert(grant_id.clone());
        self.used_grants.push_back(grant_id.clone());
        while self.used_grants.len() > MAX_USED_GRANTS {
            if let Some(oldest) = self.used_grants.pop_front() {
                self.used_grant_set.remove(&oldest);
            }
        }
        let previous_epoch = self.epoch;
        self.epoch = authority.ownership_epoch;
        self.lease_deadline = None;
        self.prepared = Some(PreparedRecovery {
            epoch: self.epoch,
            supervisor_instance: authority.supervisor_instance.clone(),
            receipt_digest: None,
        });
        Ok(FencedOwnership {
            previous_epoch,
            epoch: self.epoch,
            supervisor_instance: authority.supervisor_instance.clone(),
            challenge: challenge.to_string(),
            grant_id,
        })
    }

    /// Bind the prepared recovery to the exact receipt the runtime signed.
    pub fn bind_receipt(&mut self, digest: String) -> Result<(), RecoveryError> {
        let prepared = self.prepared.as_mut().ok_or(RecoveryError::NotPrepared)?;
        prepared.receipt_digest = Some(digest);
        Ok(())
    }

    /// Check gateway activation for the prepared candidate without committing.
    #[allow(
        clippy::suspicious_operation_groupings,
        reason = "the prepared recovery stores its epoch as `epoch`"
    )]
    pub fn check_activation(
        &self,
        activation: &VerifiedSupervisorAuthority,
    ) -> Result<(), RecoveryError> {
        let authority = &activation.authority;
        if authority.kind != SupervisorAuthorityKind::Activation {
            return Err(RecoveryError::WrongKind);
        }
        let prepared = self.prepared.as_ref().ok_or(RecoveryError::NotPrepared)?;
        if authority.ownership_epoch != prepared.epoch
            || authority.supervisor_instance != prepared.supervisor_instance
            || prepared.receipt_digest.is_none()
            || authority.receipt_digest != prepared.receipt_digest
        {
            return Err(RecoveryError::ActivationMismatch);
        }
        Ok(())
    }

    /// Commit activation after the workload resumed; the lease starts now.
    pub fn complete_activation(&mut self, now: Instant) {
        self.prepared = None;
        self.lease_deadline = Some(now + self.lease_ttl);
    }

    /// A plain confirm is valid only when no replacement is pending activation.
    pub fn confirm_allowed(&self) -> Result<(), RecoveryError> {
        if self.prepared.is_some() {
            Err(RecoveryError::ActivationRequired)
        } else {
            Ok(())
        }
    }

    /// The launch owner's first confirm starts its enforced lease.
    pub fn start_lease_if_unset(&mut self, now: Instant) {
        if self.prepared.is_none() && self.lease_deadline.is_none() {
            self.lease_deadline = Some(now + self.lease_ttl);
        }
    }

    /// Extend the current owner's lease. `remaining` is the token's remaining
    /// wall-clock lifetime; the runtime never extends beyond its own TTL.
    pub fn renew_lease(
        &mut self,
        lease: &VerifiedSupervisorAuthority,
        owner_instance: &str,
        remaining: Duration,
        now: Instant,
    ) -> Result<Duration, RecoveryError> {
        let authority = &lease.authority;
        if authority.kind != SupervisorAuthorityKind::Lease {
            return Err(RecoveryError::WrongKind);
        }
        if self.prepared.is_some()
            || authority.ownership_epoch != self.epoch
            || authority.supervisor_instance != owner_instance
        {
            return Err(RecoveryError::LeaseMismatch);
        }
        let extension = remaining.min(self.lease_ttl);
        let deadline = now + extension;
        // Never shorten an already granted lease with an older token.
        self.lease_deadline = Some(
            self.lease_deadline
                .map_or(deadline, |current| current.max(deadline)),
        );
        Ok(extension)
    }

    /// True once the owner's enforced lease lapsed.
    #[must_use]
    pub fn lease_expired(&self, now: Instant) -> bool {
        self.prepared.is_none() && self.lease_deadline.is_some_and(|deadline| now >= deadline)
    }

    /// Stop lease enforcement after the owner lost its connection; recovery
    /// is then governed by the runtime's reconnect window.
    pub fn suspend_lease(&mut self) {
        self.lease_deadline = None;
    }

    pub fn store_state(
        &mut self,
        state: SupervisorRecoveryStateWire,
    ) -> Result<u64, RecoveryError> {
        if state.binary_pins.len() > MAX_RECOVERY_STATE_PINS {
            return Err(RecoveryError::StateTooLarge);
        }
        if self
            .state
            .as_ref()
            .is_some_and(|current| state.revision <= current.revision)
        {
            return Err(RecoveryError::StaleState);
        }
        let revision = state.revision;
        self.state = Some(state);
        Ok(revision)
    }

    #[must_use]
    pub fn state(&self) -> Option<&SupervisorRecoveryStateWire> {
        self.state.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use openshell_core::jwt::{
        RecoveryKeyProof, SandboxId, SupervisorAuthority, recovery_receipt_digest,
        sign_ed25519_document,
    };
    use openshell_core::sandbox_generation::SandboxGenerationId;
    use rcgen::{KeyPair, PKCS_ED25519};

    use super::*;

    const INSTANCE: &str = "6a8f1f4e-8c0e-4d55-9d8c-1d6e2f3a4b5c";

    fn verified(
        kind: SupervisorAuthorityKind,
        epoch: u64,
        challenge: Option<String>,
        key: Option<&KeyPair>,
        receipt_digest: Option<String>,
    ) -> VerifiedSupervisorAuthority {
        VerifiedSupervisorAuthority {
            authority: SupervisorAuthority {
                kind,
                sandbox_id: SandboxId::parse("sandbox-a").unwrap(),
                runtime_generation: SandboxGenerationId::parse("generation-1").unwrap(),
                ownership_epoch: epoch,
                supervisor_instance: INSTANCE.to_string(),
                candidate_key_pem: key.map(KeyPair::public_key_pem),
                challenge,
                receipt_digest,
            },
            token_id: uuid::Uuid::new_v4(),
            issued_at: 0,
            expires_at: 0,
        }
    }

    fn proof(key: &KeyPair, challenge: &str, grant: &VerifiedSupervisorAuthority) -> String {
        sign_ed25519_document(
            key.serialize_pem().as_bytes(),
            RECOVERY_PROOF_JWT_TYPE,
            &RecoveryKeyProof {
                challenge: challenge.to_string(),
                grant_id: grant.token_id.to_string(),
                supervisor_instance: INSTANCE.to_string(),
            },
        )
        .unwrap()
    }

    fn prepared(ownership: &mut RecoveryOwnership, epoch: u64) -> FencedOwnership {
        let now = Instant::now();
        let key = KeyPair::generate_for(&PKCS_ED25519).unwrap();
        let challenge = ownership.issue_challenge(now);
        let grant = verified(
            SupervisorAuthorityKind::RecoveryGrant,
            epoch,
            Some(challenge.clone()),
            Some(&key),
            None,
        );
        ownership
            .begin_recovery(&grant, &proof(&key, &challenge, &grant), now)
            .unwrap()
    }

    #[test]
    fn full_recovery_advances_epoch_and_requires_exact_activation() {
        let mut ownership = RecoveryOwnership::new(Duration::from_secs(15));
        let fenced = prepared(&mut ownership, 2);
        assert_eq!(fenced.previous_epoch, 1);
        assert_eq!(ownership.epoch(), 2);
        assert_eq!(
            ownership.confirm_allowed(),
            Err(RecoveryError::ActivationRequired)
        );
        // Activation before the receipt is bound, or for another receipt, fails.
        let activation = verified(
            SupervisorAuthorityKind::Activation,
            2,
            None,
            None,
            Some(recovery_receipt_digest("receipt")),
        );
        assert_eq!(
            ownership.check_activation(&activation),
            Err(RecoveryError::ActivationMismatch)
        );
        ownership
            .bind_receipt(recovery_receipt_digest("other"))
            .unwrap();
        assert_eq!(
            ownership.check_activation(&activation),
            Err(RecoveryError::ActivationMismatch)
        );
        ownership
            .bind_receipt(recovery_receipt_digest("receipt"))
            .unwrap();
        ownership.check_activation(&activation).unwrap();
        let now = Instant::now();
        ownership.complete_activation(now);
        assert!(ownership.confirm_allowed().is_ok());
        assert!(!ownership.lease_expired(now));
        assert!(ownership.lease_expired(now + Duration::from_secs(16)));
    }

    #[test]
    fn stale_and_equal_epochs_cannot_fence() {
        let mut ownership = RecoveryOwnership::new(Duration::from_secs(15));
        prepared(&mut ownership, 3);
        let now = Instant::now();
        let key = KeyPair::generate_for(&PKCS_ED25519).unwrap();
        for epoch in [2, 3] {
            let challenge = ownership.issue_challenge(now);
            let grant = verified(
                SupervisorAuthorityKind::RecoveryGrant,
                epoch,
                Some(challenge.clone()),
                Some(&key),
                None,
            );
            assert_eq!(
                ownership.begin_recovery(&grant, &proof(&key, &challenge, &grant), now),
                Err(RecoveryError::StaleEpoch {
                    offered: epoch,
                    current: 3
                })
            );
        }
    }

    #[test]
    fn concurrent_candidates_cannot_both_prepare_the_same_epoch() {
        let mut ownership = RecoveryOwnership::new(Duration::from_secs(15));
        let now = Instant::now();
        let first_key = KeyPair::generate_for(&PKCS_ED25519).unwrap();
        let second_key = KeyPair::generate_for(&PKCS_ED25519).unwrap();
        let first_challenge = ownership.issue_challenge(now);
        let second_challenge = ownership.issue_challenge(now);
        let first = verified(
            SupervisorAuthorityKind::RecoveryGrant,
            2,
            Some(first_challenge.clone()),
            Some(&first_key),
            None,
        );
        let second = verified(
            SupervisorAuthorityKind::RecoveryGrant,
            2,
            Some(second_challenge.clone()),
            Some(&second_key),
            None,
        );
        ownership
            .begin_recovery(&first, &proof(&first_key, &first_challenge, &first), now)
            .unwrap();
        assert!(matches!(
            ownership.begin_recovery(
                &second,
                &proof(&second_key, &second_challenge, &second),
                now
            ),
            Err(RecoveryError::StaleEpoch { .. })
        ));
    }

    #[test]
    fn proofs_bind_key_challenge_and_single_use() {
        let mut ownership = RecoveryOwnership::new(Duration::from_secs(15));
        let now = Instant::now();
        let key = KeyPair::generate_for(&PKCS_ED25519).unwrap();
        let attacker = KeyPair::generate_for(&PKCS_ED25519).unwrap();
        let challenge = ownership.issue_challenge(now);
        let grant = verified(
            SupervisorAuthorityKind::RecoveryGrant,
            2,
            Some(challenge.clone()),
            Some(&key),
            None,
        );
        // A grant without possession of its key is useless.
        assert_eq!(
            ownership.begin_recovery(&grant, &proof(&attacker, &challenge, &grant), now),
            Err(RecoveryError::InvalidProof)
        );
        // An expired challenge cannot be used.
        assert_eq!(
            ownership.begin_recovery(
                &grant,
                &proof(&key, &challenge, &grant),
                now + RECOVERY_CHALLENGE_TTL,
            ),
            Err(RecoveryError::UnknownChallenge)
        );
        ownership
            .begin_recovery(&grant, &proof(&key, &challenge, &grant), now)
            .unwrap();
        // The challenge and grant are consumed.
        assert_eq!(
            ownership.begin_recovery(&grant, &proof(&key, &challenge, &grant), now),
            Err(RecoveryError::StaleEpoch {
                offered: 2,
                current: 2
            })
        );
    }

    #[test]
    fn leases_bind_epoch_and_instance_and_never_exceed_ttl() {
        let mut ownership = RecoveryOwnership::new(Duration::from_secs(15));
        let now = Instant::now();
        ownership.start_lease_if_unset(now);
        let lease = verified(SupervisorAuthorityKind::Lease, 1, None, None, None);
        assert_eq!(
            ownership.renew_lease(&lease, "another-instance", Duration::from_secs(10), now),
            Err(RecoveryError::LeaseMismatch)
        );
        let stale = verified(SupervisorAuthorityKind::Lease, 0, None, None, None);
        assert_eq!(
            ownership.renew_lease(&stale, INSTANCE, Duration::from_secs(10), now),
            Err(RecoveryError::LeaseMismatch)
        );
        let granted = ownership
            .renew_lease(&lease, INSTANCE, Duration::from_mins(5), now)
            .unwrap();
        assert_eq!(granted, Duration::from_secs(15));
        assert!(ownership.lease_expired(now + Duration::from_secs(15)));
        // After fencing, the previous epoch's lease is rejected.
        prepared(&mut ownership, 2);
        assert_eq!(
            ownership.renew_lease(&lease, INSTANCE, Duration::from_secs(10), now),
            Err(RecoveryError::LeaseMismatch)
        );
    }

    #[test]
    fn recovery_state_is_monotonic_and_bounded() {
        let mut ownership = RecoveryOwnership::new(Duration::from_secs(15));
        let mut state = SupervisorRecoveryStateWire {
            revision: 2,
            ..Default::default()
        };
        state
            .binary_pins
            .insert("/usr/bin/curl".to_string(), "aa".to_string());
        assert_eq!(ownership.store_state(state.clone()), Ok(2));
        assert_eq!(
            ownership.store_state(state.clone()),
            Err(RecoveryError::StaleState)
        );
        for index in 0..=MAX_RECOVERY_STATE_PINS {
            state
                .binary_pins
                .insert(format!("/bin/{index}"), "bb".to_string());
        }
        state.revision = 3;
        assert_eq!(
            ownership.store_state(state),
            Err(RecoveryError::StateTooLarge)
        );
        assert_eq!(ownership.state().map(|state| state.revision), Some(2));
    }
}
