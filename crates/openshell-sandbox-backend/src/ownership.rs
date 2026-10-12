// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Supervisor-side binding to gateway-managed enforcement ownership.
//!
//! A pooled supervisor proves its authority to the runtime with gateway-signed
//! tokens for the `supervisor-recovery.v1` Sandbox Protocol extension. The
//! gateway, not the supervisor, decides which candidate may own a sandbox and
//! at which ownership epoch.

use std::fmt;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use openshell_isolation_interface::contract::BackendError;
use openshell_isolation_interface::recovery_state::EnforcementPinJournal;

/// Request for a recovery grant bound to one runtime challenge and candidate key.
#[derive(Clone, Debug)]
pub struct RecoveryGrantRequest {
    pub supervisor_instance: String,
    /// Candidate SPKI PEM public key the runtime verifies a proof against.
    pub candidate_key_pem: String,
    pub challenge: String,
    /// Ownership epoch the runtime reported, for gateway diagnostics only.
    pub runtime_ownership_epoch: u64,
}

/// A gateway-signed recovery grant and its token ID.
#[derive(Clone)]
pub struct IssuedRecoveryGrant {
    pub grant: String,
    pub grant_id: String,
}

impl fmt::Debug for IssuedRecoveryGrant {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("IssuedRecoveryGrant")
            .field("grant_id", &self.grant_id)
            .finish_non_exhaustive()
    }
}

/// Gateway authority used by a supervisor session in gateway topology.
#[async_trait]
pub trait SupervisorOwnershipAuthority: Send + Sync + fmt::Debug {
    /// Reserve or confirm the candidate's ownership epoch and sign a grant.
    async fn recovery_grant(
        &self,
        request: RecoveryGrantRequest,
    ) -> Result<IssuedRecoveryGrant, BackendError>;

    /// Submit the runtime-signed fencing receipt and obtain activation.
    async fn activate(
        &self,
        supervisor_instance: &str,
        signed_receipt: &str,
    ) -> Result<String, BackendError>;

    /// Renew the current owner's enforcement lease.
    async fn lease(&self, supervisor_instance: &str) -> Result<String, BackendError>;

    /// The owner lost authority (lease lapsed or fenced). The session must
    /// stop acting for this sandbox.
    fn fenced(&self, reason: &str);
}

/// How a session obtains ownership of its boundary.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OwnershipMode {
    /// First owner of a fresh runtime generation (epoch 1).
    Launch,
    /// Replacement that fences the previous owner and resumes the workload.
    Recover,
}

/// Ownership binding passed to [`crate::OpenShellRuntimeBackend`].
#[derive(Clone, Debug)]
pub struct SupervisorOwnership {
    pub authority: Arc<dyn SupervisorOwnershipAuthority>,
    pub mode: OwnershipMode,
    pub pin_journal: Arc<EnforcementPinJournal>,
    /// How often the owner renews its runtime lease.
    pub lease_interval: Duration,
    /// Lease TTL the runtime enforces; the owner self-fences before it lapses.
    pub lease_ttl: Duration,
}
