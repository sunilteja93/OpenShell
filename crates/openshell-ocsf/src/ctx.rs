// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Process-wide [`EventContext`] singleton.
//!
//! Initialised once via [`set_ctx`] during sandbox start; read by every event
//! builder via [`ctx`]. Falls back to a default context when the singleton has
//! not been set (e.g. unit tests that exercise builders without booting the
//! sandbox).

use crate::{EventContext, EventOrigin};
use std::cell::Cell;
use std::collections::HashMap;
use std::sync::{LazyLock, Mutex, OnceLock};

static OCSF_CTX: OnceLock<EventContext> = OnceLock::new();

thread_local! {
    /// Context of the hosted supervisor session that owns this thread.
    static THREAD_CTX: Cell<Option<&'static EventContext>> = const { Cell::new(None) };
}

/// Interned per-session contexts. Each distinct context is allocated once
/// for the process lifetime so `ctx()` can keep returning `&'static`.
static SESSION_CTXS: LazyLock<Mutex<HashMap<String, &'static EventContext>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Intern the event context of one hosted supervisor session.
pub fn intern_session_ctx(ctx: EventContext) -> &'static EventContext {
    let key = format!("{}\u{1f}{}", ctx.sandbox_id, ctx.sandbox_name);
    let mut contexts = SESSION_CTXS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    contexts
        .entry(key)
        .or_insert_with(|| Box::leak(Box::new(ctx)))
}

/// Attribute every event emitted on the calling thread to a hosted session.
pub fn bind_thread_ctx(ctx: Option<&'static EventContext>) {
    THREAD_CTX.with(|current| current.set(ctx));
}

static OCSF_CTX_FALLBACK: LazyLock<EventContext> = LazyLock::new(|| EventContext {
    sandbox_id: String::new(),
    sandbox_name: String::new(),
    container_image: String::new(),
    hostname: "test".to_string(),
    product_version: env!("CARGO_PKG_VERSION").to_string(),
    proxy_ip: std::net::IpAddr::from([127, 0, 0, 1]),
    proxy_port: 3128,
    origin: EventOrigin::Supervisor,
});

/// Initialise the process-wide OCSF sandbox context.
///
/// Returns `false` if the context was already set; the caller may log and
/// continue. Intended to be called exactly once during sandbox startup.
pub fn set_ctx(ctx: EventContext) -> bool {
    OCSF_CTX.set(ctx).is_ok()
}

/// Return a reference to the process-wide [`EventContext`].
///
/// Falls back to a default context if [`set_ctx`] has not been called (e.g.
/// during unit tests that exercise individual builders).
#[must_use]
pub fn ctx() -> &'static EventContext {
    THREAD_CTX
        .with(Cell::get)
        .or_else(|| OCSF_CTX.get())
        .unwrap_or(&OCSF_CTX_FALLBACK)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builders::test_sandbox_context;
    use crate::{ActionId, ActivityId, Endpoint, NetworkActivityBuilder};

    fn session_ctx(sandbox_id: &str, sandbox_name: &str) -> &'static EventContext {
        intern_session_ctx(EventContext {
            sandbox_id: sandbox_id.to_string(),
            sandbox_name: sandbox_name.to_string(),
            ..test_sandbox_context()
        })
    }

    fn attributed_sandbox() -> (String, String) {
        let json = NetworkActivityBuilder::new(ctx())
            .activity(ActivityId::Open)
            .action(ActionId::Denied)
            .dst_endpoint(Endpoint::from_domain("api.example.com", 443))
            .message("CONNECT denied")
            .build()
            .to_json()
            .unwrap();
        (
            json["container"]["uid"]
                .as_str()
                .unwrap_or_default()
                .to_string(),
            json["container"]["name"]
                .as_str()
                .unwrap_or_default()
                .to_string(),
        )
    }

    #[test]
    fn hosted_session_threads_attribute_events_to_their_own_sandbox() {
        let alpha = session_ctx("sandbox-alpha", "alpha");
        let beta = session_ctx("sandbox-beta", "beta");

        let run = |bound: &'static EventContext| {
            std::thread::spawn(move || {
                bind_thread_ctx(Some(bound));
                attributed_sandbox()
            })
        };
        let (alpha_events, beta_events) = (run(alpha), run(beta));
        assert_eq!(
            alpha_events.join().unwrap(),
            ("sandbox-alpha".to_string(), "alpha".to_string())
        );
        assert_eq!(
            beta_events.join().unwrap(),
            ("sandbox-beta".to_string(), "beta".to_string())
        );

        // Unbound threads keep the dedicated (process-wide) attribution.
        let unbound = std::thread::spawn(attributed_sandbox).join().unwrap();
        assert_ne!(unbound.0, "sandbox-alpha");
        assert_ne!(unbound.0, "sandbox-beta");

        // Unbinding a thread drops the session attribution.
        bind_thread_ctx(Some(alpha));
        bind_thread_ctx(None);
        assert_ne!(ctx().sandbox_id, "sandbox-alpha");
    }

    #[test]
    fn interning_reuses_one_context_per_session() {
        let first = session_ctx("sandbox-intern", "intern");
        let second = session_ctx("sandbox-intern", "intern");
        assert!(std::ptr::eq(first, second));
        assert!(!std::ptr::eq(first, session_ctx("sandbox-other", "intern")));
    }
}
