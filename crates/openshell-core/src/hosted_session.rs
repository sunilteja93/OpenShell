// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Identity of the supervisor session that owns the current thread.
//!
//! A shared supervisor process hosts each sandbox session on its own Tokio
//! runtime. The runtime's thread-start hook binds the session identity to
//! every worker and blocking thread, so code that previously read
//! process-wide sandbox environment resolves the owning session instead.
//! Threads outside a hosted session see `None` and keep the dedicated
//! supervisor behavior.

use std::cell::RefCell;
use std::sync::Arc;

/// Per-session identity visible to code running on the session's threads.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HostedSessionIdentity {
    pub sandbox_id: String,
    pub sandbox_name: String,
    /// Session-scoped gateway endpoint (see
    /// [`crate::grpc_client::session_scoped_endpoint`]).
    pub endpoint: String,
}

thread_local! {
    static CURRENT: RefCell<Option<Arc<HostedSessionIdentity>>> = const { RefCell::new(None) };
}

/// Bind `identity` to the calling thread. Called from a session runtime's
/// thread-start hook.
pub fn bind_current_thread(identity: Option<Arc<HostedSessionIdentity>>) {
    CURRENT.with(|current| *current.borrow_mut() = identity);
}

/// The hosted session that owns the calling thread, if any.
#[must_use]
pub fn current() -> Option<Arc<HostedSessionIdentity>> {
    CURRENT.with(|current| current.borrow().clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_is_thread_scoped() {
        let identity = Arc::new(HostedSessionIdentity {
            sandbox_id: "a".to_string(),
            sandbox_name: "alpha".to_string(),
            endpoint: "http://gateway/?openshell-supervisor-session=a".to_string(),
        });
        bind_current_thread(Some(identity.clone()));
        assert_eq!(current(), Some(identity));
        std::thread::spawn(|| assert_eq!(current(), None))
            .join()
            .unwrap();
        bind_current_thread(None);
        assert_eq!(current(), None);
    }
}
