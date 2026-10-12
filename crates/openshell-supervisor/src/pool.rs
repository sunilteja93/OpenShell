// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared supervisor replica for the gateway supervisor topology.
//!
//! One process hosts many sandbox sessions. Each session runs on its own Tokio
//! runtime whose threads are bound to that session's identity, OCSF context,
//! and log sink, so attribution, tasks, and cleanup stay per sandbox. The
//! gateway decides which replica owns which sandbox at which ownership epoch;
//! this controller only follows assignments:
//!
//! - `LAUNCH`: own a fresh runtime generation.
//! - `RECOVER`: fence the previous owner and resume its running workload.
//! - `RELEASE`: the sandbox stopped or was deleted; terminate the session.
//! - `RELINQUISH`: planned handoff (drain/rebalance); release ownership
//!   without stopping the workload. The session is abandoned and keeps being
//!   reported until its thread has fully stopped, which is how the gateway
//!   learns the release completed.
//! - Not listed: ownership moved away; abandon the session without touching
//!   the workload so the new owner can resume it.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use miette::{IntoDiagnostic as _, Result, WrapErr as _};
use openshell_core::grpc_client::{PoolMemberClient, connect_supervisor_pool};
use openshell_core::hosted_session::HostedSessionIdentity;
use openshell_core::proto::{
    ActivateSupervisorAssignmentRequest, GetSupervisorAssignmentRequest,
    RenewSupervisorLeaseRequest, RequestSupervisorRecoveryGrantRequest, SupervisorAssignmentAction,
    SyncSupervisorPoolRequest,
};
use openshell_isolation_interface::contract::{BackendDescriptor, BackendError};
use openshell_isolation_interface::recovery_state::EnforcementPinJournal;
use openshell_sandbox_backend::ownership::{
    IssuedRecoveryGrant, OwnershipMode, RecoveryGrantRequest, SupervisorOwnership,
    SupervisorOwnershipAuthority,
};
use tokio::sync::watch;
use tracing::{debug, error, info, warn};

/// Worker threads per hosted session runtime.
const SESSION_WORKER_THREADS: usize = 2;
/// How long a terminated session may take to stop its workload cleanly.
const SESSION_TERMINATE_GRACE: Duration = Duration::from_secs(20);

/// Static inputs for one shared supervisor replica.
#[derive(Clone, Debug)]
pub struct PoolConfig {
    /// Gateway endpoint shared by every session.
    pub endpoint: String,
    /// Projected platform token that authenticates this replica.
    pub token_file: PathBuf,
    /// Per-session state root (sockets, CA material).
    pub session_root: PathBuf,
    /// Upstream proxy settings shared by every session.
    pub upstream_proxy_args: openshell_supervisor_network::upstream_proxy::UpstreamProxyArgs,
    /// Ready-signal sender: set once the replica has registered.
    pub ready: Option<watch::Sender<bool>>,
}

/// Per-attempt deadline for pool RPCs. A hung (not dead) gateway must not
/// stall an owner's lease renewal until HTTP/2 keepalive notices.
const POOL_RPC_ATTEMPT_TIMEOUT: Duration = Duration::from_millis(1_500);
/// Total budget for one pool RPC across reconnecting retries.
const POOL_RPC_BUDGET: Duration = Duration::from_secs(4);

/// Pool RPC client with per-attempt deadlines.
///
/// Idempotent calls (sync, get assignment, lease renewal) reconnect and retry
/// on timeouts and transport failures within [`POOL_RPC_BUDGET`]; a new
/// connection is routed by the gateway Service again, so they reach any
/// healthy gateway. Grant and activation commit single-use durable state and
/// are NOT retried: after a lost response a retry could turn a committed
/// grant into "already issued" and strand a valid candidate. They get one
/// bounded attempt; on failure the coordinator reserves a new epoch for
/// another attempt within the original recovery deadline.
#[derive(Clone, Debug)]
pub(crate) struct PoolChannel {
    endpoint: String,
    token_file: PathBuf,
    client: Arc<tokio::sync::RwLock<PoolMemberClient>>,
}

impl PoolChannel {
    async fn connect(endpoint: &str, token_file: &Path) -> Result<Self> {
        let client = connect_supervisor_pool(endpoint, token_file).await?;
        Ok(Self {
            endpoint: endpoint.to_string(),
            token_file: token_file.to_path_buf(),
            client: Arc::new(tokio::sync::RwLock::new(client)),
        })
    }

    /// Replace the connection, bounded by `deadline` so a hanging connect
    /// cannot outlive the caller's RPC budget.
    async fn reconnect(&self, deadline: tokio::time::Instant) {
        match tokio::time::timeout_at(
            deadline,
            connect_supervisor_pool(&self.endpoint, &self.token_file),
        )
        .await
        {
            Ok(Ok(client)) => *self.client.write().await = client,
            Ok(Err(error)) => warn!(error = %error, "Supervisor pool reconnect failed"),
            Err(_) => warn!("Supervisor pool reconnect timed out"),
        }
    }

    /// One bounded attempt, for calls that commit single-use durable state.
    /// On a transport failure the connection is replaced for later calls,
    /// but the call itself is not repeated.
    async fn call_once<T, F, Fut>(
        &self,
        what: &str,
        call: F,
    ) -> std::result::Result<T, tonic::Status>
    where
        F: FnOnce(PoolMemberClient) -> Fut,
        Fut: Future<Output = std::result::Result<tonic::Response<T>, tonic::Status>>,
    {
        let client = self.client.read().await.clone();
        run_once(what, POOL_RPC_ATTEMPT_TIMEOUT, call(client), |deadline| {
            self.reconnect(deadline)
        })
        .await
    }

    /// Run an idempotent `call` with a per-attempt timeout, reconnecting and
    /// retrying on timeouts and transport errors until the budget is spent.
    /// Application errors (denied, failed precondition, ...) return
    /// immediately.
    async fn call<T, F, Fut>(&self, what: &str, call: F) -> std::result::Result<T, tonic::Status>
    where
        T: Send,
        F: Fn(PoolMemberClient) -> Fut + Sync,
        Fut: Future<Output = std::result::Result<tonic::Response<T>, tonic::Status>> + Send,
    {
        run_idempotent(
            what,
            POOL_RPC_BUDGET,
            POOL_RPC_ATTEMPT_TIMEOUT,
            || async { call(self.client.read().await.clone()).await },
            |deadline| self.reconnect(deadline),
        )
        .await
    }
}

/// Single bounded attempt; on a transport failure, reconnect (bounded) for
/// later calls and return the failure without repeating the call.
async fn run_once<T, Fut, R, RFut>(
    what: &str,
    attempt_timeout: Duration,
    attempt: Fut,
    reconnect: R,
) -> std::result::Result<T, tonic::Status>
where
    Fut: Future<Output = std::result::Result<tonic::Response<T>, tonic::Status>>,
    R: FnOnce(tokio::time::Instant) -> RFut,
    RFut: Future<Output = ()>,
{
    let status = match tokio::time::timeout(attempt_timeout, attempt).await {
        Ok(Ok(response)) => return Ok(response.into_inner()),
        Ok(Err(status)) if !is_transport_failure(&status) => return Err(status),
        Ok(Err(status)) => status,
        Err(_) => tonic::Status::deadline_exceeded(format!("{what} timed out")),
    };
    // The reconnect is bounded too: a hanging connect must not stall the
    // caller (and with it the recovery attempt) indefinitely.
    let deadline = tokio::time::Instant::now() + attempt_timeout;
    let _ = tokio::time::timeout_at(deadline, reconnect(deadline)).await;
    Err(status)
}

/// Retry an idempotent call within `budget`: each attempt and each
/// reconnect is bounded by the remaining budget.
async fn run_idempotent<T, A, Fut, R, RFut>(
    what: &str,
    budget: Duration,
    attempt_timeout: Duration,
    attempt: A,
    reconnect: R,
) -> std::result::Result<T, tonic::Status>
where
    A: Fn() -> Fut,
    Fut: Future<Output = std::result::Result<tonic::Response<T>, tonic::Status>>,
    R: Fn(tokio::time::Instant) -> RFut,
    RFut: Future<Output = ()>,
{
    let deadline = tokio::time::Instant::now() + budget;
    loop {
        let attempt_deadline = (tokio::time::Instant::now() + attempt_timeout).min(deadline);
        let status = match tokio::time::timeout_at(attempt_deadline, attempt()).await {
            Ok(Ok(response)) => return Ok(response.into_inner()),
            Ok(Err(status)) if !is_transport_failure(&status) => return Err(status),
            Ok(Err(status)) => status,
            Err(_) => tonic::Status::deadline_exceeded(format!("{what} timed out")),
        };
        if tokio::time::Instant::now() >= deadline {
            return Err(status);
        }
        debug!(what, error = %status, "Supervisor pool RPC failed; reconnecting");
        // A hanging connect must not outlive the budget either.
        let _ = tokio::time::timeout_at(deadline, reconnect(deadline)).await;
        if tokio::time::Instant::now() >= deadline {
            return Err(status);
        }
    }
}

fn is_transport_failure(status: &tonic::Status) -> bool {
    matches!(
        status.code(),
        tonic::Code::Unavailable | tonic::Code::Unknown | tonic::Code::Cancelled
    )
}

/// Ownership authority for one hosted session, backed by pool RPCs.
#[derive(Debug)]
struct PoolOwnershipAuthority {
    client: PoolChannel,
    sandbox_id: String,
    ownership_epoch: u64,
    fenced: watch::Sender<bool>,
}

fn rpc_error(context: &str, status: &tonic::Status) -> BackendError {
    match status.code() {
        tonic::Code::PermissionDenied | tonic::Code::FailedPrecondition => {
            BackendError::Denied(format!("{context}: {}", status.message()))
        }
        _ => BackendError::Unavailable(format!("{context}: {}", status.message())),
    }
}

#[tonic::async_trait]
impl SupervisorOwnershipAuthority for PoolOwnershipAuthority {
    async fn recovery_grant(
        &self,
        request: RecoveryGrantRequest,
    ) -> std::result::Result<IssuedRecoveryGrant, BackendError> {
        let grant_request = RequestSupervisorRecoveryGrantRequest {
            sandbox_id: self.sandbox_id.clone(),
            ownership_epoch: self.ownership_epoch,
            supervisor_instance: request.supervisor_instance,
            candidate_key_pem: request.candidate_key_pem,
            challenge: request.challenge,
            runtime_ownership_epoch: request.runtime_ownership_epoch,
        };
        let response = self
            .client
            .call_once("request recovery grant", |mut client| async move {
                client
                    .request_supervisor_recovery_grant(grant_request)
                    .await
            })
            .await
            .map_err(|status| rpc_error("request recovery grant", &status))?;
        Ok(IssuedRecoveryGrant {
            grant: response.grant,
            grant_id: response.grant_id,
        })
    }

    async fn activate(
        &self,
        supervisor_instance: &str,
        signed_receipt: &str,
    ) -> std::result::Result<String, BackendError> {
        let activate = ActivateSupervisorAssignmentRequest {
            sandbox_id: self.sandbox_id.clone(),
            ownership_epoch: self.ownership_epoch,
            supervisor_instance: supervisor_instance.to_string(),
            signed_receipt: signed_receipt.to_string(),
        };
        Ok(self
            .client
            .call_once("activate recovery", |mut client| async move {
                client.activate_supervisor_assignment(activate).await
            })
            .await
            .map_err(|status| rpc_error("activate recovery", &status))?
            .activation)
    }

    async fn lease(&self, supervisor_instance: &str) -> std::result::Result<String, BackendError> {
        let renew = RenewSupervisorLeaseRequest {
            sandbox_id: self.sandbox_id.clone(),
            ownership_epoch: self.ownership_epoch,
            supervisor_instance: supervisor_instance.to_string(),
        };
        Ok(self
            .client
            .call("renew lease", |mut client| {
                let renew = renew.clone();
                async move { client.renew_supervisor_lease(renew).await }
            })
            .await
            .map_err(|status| rpc_error("renew lease", &status))?
            .lease)
    }

    fn fenced(&self, reason: &str) {
        warn!(
            "Hosted supervisor session lost ownership; abandoning sandbox_id={} ownership_epoch={} reason={reason}",
            self.sandbox_id, self.ownership_epoch
        );
        let _ = self.fenced.send(true);
    }
}

/// One hosted session: its own runtime thread plus stop controls.
struct HostedSession {
    /// A restarted sandbox gets a new launch (same or new runtime generation,
    /// new credential epoch) whose owner starts again at ownership epoch 1,
    /// so all three identify the session.
    runtime_generation: String,
    credential_epoch: u64,
    ownership_epoch: u64,
    /// `true` asks the session to stop its workload (sandbox stop/delete).
    terminate: watch::Sender<bool>,
    /// `true` drops the session immediately without touching the workload.
    abandon: watch::Sender<bool>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl HostedSession {
    fn finished(&self) -> bool {
        self.thread
            .as_ref()
            .is_none_or(std::thread::JoinHandle::is_finished)
    }
}

fn write_private(path: &Path, contents: &[u8]) -> Result<()> {
    use std::io::Write as _;
    #[cfg(unix)]
    use std::os::unix::fs::OpenOptionsExt as _;
    let mut options = std::fs::OpenOptions::new();
    options.write(true).create(true).truncate(true);
    #[cfg(unix)]
    options.mode(0o600);
    let mut file = options
        .open(path)
        .into_diagnostic()
        .wrap_err_with(|| format!("open {}", path.display()))?;
    file.write_all(contents).into_diagnostic()
}

/// Run the replica until the process receives SIGTERM/SIGINT.
pub async fn run_pool(config: PoolConfig) -> Result<()> {
    std::fs::create_dir_all(&config.session_root)
        .into_diagnostic()
        .wrap_err("create supervisor session root")?;
    // The gateway may still be starting; keep retrying instead of exiting.
    let client = loop {
        match PoolChannel::connect(&config.endpoint, &config.token_file).await {
            Ok(client) => break client,
            Err(error) => {
                warn!(error = %error, "Supervisor pool cannot reach the gateway yet; retrying");
                tokio::time::sleep(Duration::from_secs(2)).await;
            }
        }
    };
    let mut sessions: HashMap<String, HostedSession> = HashMap::new();
    // A recovery attempt consumes its epoch's single grant; never retry it.
    // The gateway reserves a new epoch for another candidate instead.
    let mut attempted_recoveries = std::collections::HashSet::new();
    let mut interval = Duration::from_secs(1);
    let shutdown = shutdown_signal();
    tokio::pin!(shutdown);
    let mut registered = false;
    loop {
        tokio::select! {
            () = &mut shutdown => break,
            () = tokio::time::sleep(interval) => {}
        }
        sessions.retain(|sandbox_id, session| {
            if session.finished() {
                info!("Hosted supervisor session ended sandbox_id={sandbox_id}");
                false
            } else {
                true
            }
        });
        let sync = SyncSupervisorPoolRequest {
            active_sandbox_ids: sessions.keys().cloned().collect(),
        };
        let response = match client
            .call("sync supervisor pool", |mut client| {
                let sync = sync.clone();
                async move { client.sync_supervisor_pool(sync).await }
            })
            .await
        {
            Ok(response) => response,
            Err(status) => {
                warn!(error = %status, "Supervisor pool sync failed; retrying");
                continue;
            }
        };
        if !registered {
            registered = true;
            info!(member = %response.member_id, "Shared supervisor replica registered with the gateway");
            if let Some(ready) = config.ready.as_ref() {
                let _ = ready.send(true);
            }
        }
        interval = response
            .sync_interval
            .as_ref()
            .and_then(|interval| openshell_core::time::duration_to_std(interval).ok())
            .unwrap_or(Duration::from_secs(1))
            .max(Duration::from_millis(200));
        let mut listed = std::collections::HashSet::new();
        for assignment in response.assignments {
            listed.insert(assignment.sandbox_id.clone());
            let action = SupervisorAssignmentAction::try_from(assignment.action)
                .unwrap_or(SupervisorAssignmentAction::Unspecified);
            match action {
                SupervisorAssignmentAction::Launch | SupervisorAssignmentAction::Recover => {
                    let current = sessions.get(&assignment.sandbox_id);
                    if current.is_some_and(|session| {
                        session.runtime_generation == assignment.runtime_generation
                            && session.credential_epoch == assignment.credential_epoch
                            && session.ownership_epoch == assignment.ownership_epoch
                    }) {
                        continue;
                    }
                    let mode = if action == SupervisorAssignmentAction::Recover {
                        OwnershipMode::Recover
                    } else {
                        OwnershipMode::Launch
                    };
                    if mode == OwnershipMode::Recover
                        && !attempted_recoveries.insert((
                            assignment.sandbox_id.clone(),
                            assignment.runtime_generation.clone(),
                            assignment.credential_epoch,
                            assignment.ownership_epoch,
                        ))
                    {
                        continue;
                    }
                    if let Some(stale) = sessions.remove(&assignment.sandbox_id) {
                        let _ = stale.abandon.send(true);
                    }
                    info!(
                        "Starting hosted supervisor session sandbox_id={} ownership_epoch={} mode={mode:?}",
                        assignment.sandbox_id, assignment.ownership_epoch
                    );
                    let session = spawn_session(
                        &config,
                        client.clone(),
                        assignment.sandbox_id.clone(),
                        assignment.sandbox_name,
                        assignment.runtime_generation,
                        assignment.credential_epoch,
                        assignment.ownership_epoch,
                        mode,
                    );
                    sessions.insert(assignment.sandbox_id, session);
                }
                SupervisorAssignmentAction::Release => {
                    if let Some(session) = sessions.get(&assignment.sandbox_id) {
                        info!(
                            "Gateway stopped sandbox; terminating hosted session sandbox_id={}",
                            assignment.sandbox_id
                        );
                        let _ = session.terminate.send(true);
                    }
                }
                SupervisorAssignmentAction::Relinquish => {
                    if let Some(session) = sessions.get(&assignment.sandbox_id)
                        && !*session.abandon.borrow()
                    {
                        info!(
                            "Planned handoff; releasing ownership without stopping the workload sandbox_id={}",
                            assignment.sandbox_id
                        );
                        let _ = session.abandon.send(true);
                    }
                }
                SupervisorAssignmentAction::Supervise | SupervisorAssignmentAction::Unspecified => {
                }
            }
        }
        // Ownership moved away (or the gateway no longer knows this replica
        // as owner): stop acting immediately, never touch the workload.
        for (sandbox_id, session) in &sessions {
            if !listed.contains(sandbox_id) && !*session.terminate.borrow() {
                warn!(
                    "Sandbox no longer assigned to this replica; abandoning session sandbox_id={sandbox_id}"
                );
                let _ = session.abandon.send(true);
            }
        }
    }
    info!(
        sessions = sessions.len(),
        "Shared supervisor replica stopping; abandoning hosted sessions for takeover"
    );
    for session in sessions.values() {
        let _ = session.abandon.send(true);
    }
    for (_, mut session) in sessions {
        if let Some(thread) = session.thread.take() {
            let _ = tokio::task::spawn_blocking(move || thread.join()).await;
        }
    }
    Ok(())
}

async fn shutdown_signal() {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{SignalKind, signal};
        let mut sigterm = signal(SignalKind::terminate()).expect("install SIGTERM handler");
        let mut sigint = signal(SignalKind::interrupt()).expect("install SIGINT handler");
        tokio::select! {
            _ = sigterm.recv() => {}
            _ = sigint.recv() => {}
        }
    }
    #[cfg(not(unix))]
    {
        let _ = tokio::signal::ctrl_c().await;
    }
}

#[allow(clippy::too_many_arguments)]
fn spawn_session(
    config: &PoolConfig,
    client: PoolChannel,
    sandbox_id: String,
    sandbox_name: String,
    runtime_generation: String,
    credential_epoch: u64,
    ownership_epoch: u64,
    mode: OwnershipMode,
) -> HostedSession {
    let (terminate_tx, terminate_rx) = watch::channel(false);
    let (abandon_tx, abandon_rx) = watch::channel(false);
    let config = config.clone();
    let abandon_for_fence = abandon_tx.clone();
    let thread = std::thread::Builder::new()
        .name(format!(
            "session-{}",
            &sandbox_id[..sandbox_id.len().min(8)]
        ))
        .spawn(move || {
            run_session_thread(
                config,
                client,
                sandbox_id,
                sandbox_name,
                ownership_epoch,
                mode,
                terminate_rx,
                abandon_rx,
                abandon_for_fence,
            );
        })
        .expect("spawn hosted session thread");
    HostedSession {
        runtime_generation,
        credential_epoch,
        ownership_epoch,
        terminate: terminate_tx,
        abandon: abandon_tx,
        thread: Some(thread),
    }
}

#[allow(clippy::too_many_arguments, clippy::needless_pass_by_value)]
fn run_session_thread(
    config: PoolConfig,
    client: PoolChannel,
    sandbox_id: String,
    sandbox_name: String,
    ownership_epoch: u64,
    mode: OwnershipMode,
    terminate: watch::Receiver<bool>,
    abandon: watch::Receiver<bool>,
    fence: watch::Sender<bool>,
) {
    let base_endpoint = config.endpoint.clone();
    host_on_session_runtime(
        &base_endpoint,
        sandbox_id.clone(),
        sandbox_name.clone(),
        abandon,
        move |endpoint, log_rx| async move {
            let result = run_hosted(
                &config,
                client,
                &sandbox_id,
                &sandbox_name,
                ownership_epoch,
                mode,
                endpoint,
                terminate,
                fence,
                log_rx,
            )
            .await;
            match result {
                Ok(exit_code) => {
                    info!(
                        "Hosted supervisor session finished sandbox_id={sandbox_id} exit_code={exit_code}"
                    );
                }
                Err(error) => {
                    error!("Hosted supervisor session failed sandbox_id={sandbox_id}: {error}");
                }
            }
        },
    );
}

/// Run one hosted session to completion on its own Tokio runtime.
///
/// Every runtime thread (and the calling thread) is bound to the session's
/// identity, OCSF context and log sink. The session ends when its future
/// completes (terminate path: tasks get a grace period to stop the workload)
/// or when `abandon` fires (ownership moved: tasks are dropped without running
/// shutdown paths so the workload stays frozen for the next owner). Either
/// way the session's gateway credentials are removed and the calling thread
/// is unbound.
fn host_on_session_runtime<F, Fut>(
    base_endpoint: &str,
    sandbox_id: String,
    sandbox_name: String,
    mut abandon: watch::Receiver<bool>,
    session: F,
) where
    F: FnOnce(String, tokio::sync::mpsc::Receiver<openshell_core::proto::SandboxLogLine>) -> Fut,
    Fut: Future<Output = ()>,
{
    let endpoint = openshell_core::grpc_client::session_scoped_endpoint(base_endpoint, &sandbox_id);
    let identity = Arc::new(HostedSessionIdentity {
        sandbox_id: sandbox_id.clone(),
        sandbox_name: sandbox_name.clone(),
        endpoint: endpoint.clone(),
    });
    let event_context = openshell_ocsf::ctx::intern_session_ctx(openshell_ocsf::EventContext {
        sandbox_id: sandbox_id.clone(),
        sandbox_name,
        container_image: String::new(),
        hostname: std::env::var("HOSTNAME").unwrap_or_else(|_| "openshell-supervisor".to_string()),
        product_version: openshell_core::VERSION.to_string(),
        proxy_ip: std::net::IpAddr::from([127, 0, 0, 1]),
        proxy_port: 3128,
        origin: openshell_ocsf::EventOrigin::Supervisor,
    });
    let (log_sink, log_rx) =
        openshell_supervisor_process::log_push::session_sink(sandbox_id.clone());
    let bind = {
        move || {
            openshell_core::hosted_session::bind_current_thread(Some(identity.clone()));
            openshell_ocsf::ctx::bind_thread_ctx(Some(event_context));
            openshell_supervisor_process::log_push::bind_thread_sink(Some(log_sink.clone()));
        }
    };
    let runtime = match tokio::runtime::Builder::new_multi_thread()
        .worker_threads(SESSION_WORKER_THREADS)
        .thread_name(format!("sb-{}", &sandbox_id[..sandbox_id.len().min(8)]))
        .enable_all()
        .on_thread_start(bind.clone())
        .build()
    {
        Ok(runtime) => runtime,
        Err(error) => {
            error!(%sandbox_id, %error, "Failed to build hosted session runtime");
            return;
        }
    };
    // The controlling thread also runs session futures via block_on.
    bind();
    let session = session(endpoint.clone(), log_rx);
    let abandoned = runtime.block_on(async {
        tokio::select! {
            () = session => false,
            _ = abandon.wait_for(|abandon| *abandon) => true,
        }
    });
    openshell_core::grpc_client::remove_session_credentials(&endpoint);
    if abandoned {
        // Drop every task without running shutdown paths: the workload stays
        // frozen under the runtime's recovery window for the next owner.
        runtime.shutdown_background();
    } else {
        runtime.shutdown_timeout(SESSION_TERMINATE_GRACE);
    }
    openshell_core::hosted_session::bind_current_thread(None);
    openshell_ocsf::ctx::bind_thread_ctx(None);
    openshell_supervisor_process::log_push::bind_thread_sink(None);
}

#[allow(clippy::too_many_arguments)]
async fn run_hosted(
    config: &PoolConfig,
    client: PoolChannel,
    sandbox_id: &str,
    sandbox_name: &str,
    ownership_epoch: u64,
    mode: OwnershipMode,
    endpoint: String,
    terminate: watch::Receiver<bool>,
    fence: watch::Sender<bool>,
    log_rx: tokio::sync::mpsc::Receiver<openshell_core::proto::SandboxLogLine>,
) -> Result<i32> {
    let get = GetSupervisorAssignmentRequest {
        sandbox_id: sandbox_id.to_string(),
        ownership_epoch,
    };
    let material = client
        .call("get supervisor assignment", |mut client| {
            let get = get.clone();
            async move { client.get_supervisor_assignment(get).await }
        })
        .await
        .map_err(|status| miette::miette!("get supervisor assignment: {}", status.message()))?;
    let auth_bundle: openshell_core::jwt::SupervisorAuthBundle =
        serde_json::from_slice(&material.auth_bundle_json)
            .into_diagnostic()
            .wrap_err("decode assigned auth bundle")?;
    auth_bundle
        .validate()
        .map_err(|error| miette::miette!("validate assigned auth bundle: {error}"))?;
    // Install this session's gateway credentials before any gateway client
    // (including log push) dials with the session-scoped endpoint.
    openshell_core::grpc_client::install_supervisor_auth_bundle_for_endpoint(
        &endpoint,
        &auth_bundle,
    )?;
    tokio::spawn(openshell_supervisor_process::log_push::run_session_push(
        endpoint.clone(),
        sandbox_id.to_string(),
        log_rx,
    ));
    let main_process = if material.main_process_spec_json.is_empty() {
        openshell_core::sandbox_env::MainProcessConfig::scratch()
    } else {
        openshell_core::sandbox_env::MainProcessConfig::decode(&material.main_process_spec_json)
            .map_err(|error| miette::miette!("decode main process spec: {error}"))?
    };

    let session_dir = config.session_root.join(sandbox_id);
    std::fs::create_dir_all(&session_dir)
        .into_diagnostic()
        .wrap_err("create session directory")?;
    let proxy_ca_paths = if material.proxy_ca_cert_pem.is_empty() {
        None
    } else {
        let certificate = session_dir.join("proxy-ca.crt");
        let key = session_dir.join("proxy-ca.key");
        write_private(&certificate, material.proxy_ca_cert_pem.as_bytes())?;
        write_private(&key, material.proxy_ca_key_pem.as_bytes())?;
        Some((certificate, key))
    };
    let mut upstream_proxy_args = config.upstream_proxy_args.clone();
    if !material.upstream_proxy_ca_pem.is_empty() {
        let bundle = session_dir.join("upstream-proxy-ca.pem");
        write_private(&bundle, material.upstream_proxy_ca_pem.as_bytes())?;
        upstream_proxy_args.proxy_ca_bundle = Some(bundle.to_string_lossy().into_owned());
    }
    let proxy_tls_dir = session_dir.join("proxy-tls");
    std::fs::create_dir_all(&proxy_tls_dir).into_diagnostic()?;
    let ssh_socket_path = session_dir.join("ssh.sock");
    let _ = std::fs::remove_file(&ssh_socket_path);

    let duration = |value: Option<&prost_types::Duration>, minimum: Duration| {
        value
            .and_then(|value| openshell_core::time::duration_to_std(value).ok())
            .unwrap_or(minimum)
            .max(minimum)
    };
    let lease_ttl = duration(material.lease_ttl.as_ref(), Duration::from_secs(1));
    let lease_interval = duration(material.lease_interval.as_ref(), Duration::from_millis(250));
    let ownership = SupervisorOwnership {
        authority: Arc::new(PoolOwnershipAuthority {
            client,
            sandbox_id: sandbox_id.to_string(),
            ownership_epoch,
            fenced: fence,
        }),
        mode,
        pin_journal: Arc::new(EnforcementPinJournal::new(std::collections::BTreeMap::new())),
        lease_interval,
        lease_ttl,
    };
    crate::run_hosted_session(crate::HostedSessionLaunch {
        sandbox_id: sandbox_id.to_string(),
        sandbox_name: sandbox_name.to_string(),
        endpoint,
        main_process,
        workdir: Some("/sandbox".to_string()),
        ssh_socket_path: ssh_socket_path.to_string_lossy().into_owned(),
        upstream_proxy_args,
        backend_descriptor: BackendDescriptor {
            backend_name: openshell_sandbox_backend::BACKEND_NAME.to_string(),
            payload: material.runtime_descriptor_json,
        },
        auth_bundle,
        ownership,
        proxy_ca_paths,
        proxy_tls_dir,
        terminate,
    })
    .await
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

    /// What one hosted session observed about its own threads.
    #[derive(Default)]
    struct Observed {
        ticks: AtomicU64,
        misattributed: AtomicBool,
        task_dropped: AtomicBool,
        shutdown_path_ran: AtomicBool,
    }

    struct DropFlag(Arc<Observed>);
    impl Drop for DropFlag {
        fn drop(&mut self) {
            self.0.task_dropped.store(true, Ordering::SeqCst);
        }
    }

    fn check_binding(sandbox_id: &str, observed: &Observed) {
        let identity_ok = openshell_core::hosted_session::current()
            .is_some_and(|identity| identity.sandbox_id == sandbox_id);
        let ocsf_ok = openshell_ocsf::ctx::ctx().sandbox_id == sandbox_id;
        if !(identity_ok && ocsf_ok) {
            observed.misattributed.store(true, Ordering::SeqCst);
        }
    }

    /// Host a fake session whose spawned and blocking tasks keep checking
    /// that they are attributed to their own sandbox. The session future
    /// completes only on `terminate`.
    fn host(
        sandbox_id: &'static str,
        observed: Arc<Observed>,
        mut terminate: watch::Receiver<bool>,
        abandon: watch::Receiver<bool>,
    ) -> std::thread::JoinHandle<()> {
        std::thread::spawn(move || {
            host_on_session_runtime(
                "https://gateway.example:8080",
                sandbox_id.to_string(),
                sandbox_id.to_string(),
                abandon,
                move |endpoint, _logs| async move {
                    assert!(endpoint.ends_with(&format!("session={sandbox_id}")));
                    let worker = {
                        let observed = observed.clone();
                        tokio::spawn(async move {
                            let _flag = DropFlag(observed.clone());
                            loop {
                                check_binding(sandbox_id, &observed);
                                let blocking = observed.clone();
                                tokio::task::spawn_blocking(move || {
                                    check_binding(sandbox_id, &blocking);
                                })
                                .await
                                .unwrap();
                                observed.ticks.fetch_add(1, Ordering::SeqCst);
                                tokio::time::sleep(Duration::from_millis(10)).await;
                            }
                        })
                    };
                    let _ = terminate.wait_for(|terminate| *terminate).await;
                    // Terminate path: the session gets to run its own cleanup.
                    observed.shutdown_path_ran.store(true, Ordering::SeqCst);
                    worker.abort();
                },
            );
            // The hosting thread is unbound once the session ends.
            assert!(openshell_core::hosted_session::current().is_none());
        })
    }

    fn wait_until(what: &str, condition: impl Fn() -> bool) {
        let deadline = std::time::Instant::now() + Duration::from_secs(10);
        while !condition() {
            assert!(std::time::Instant::now() < deadline, "timed out: {what}");
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    #[test]
    fn stopping_one_hosted_session_leaves_the_other_running_and_isolated() {
        let alpha = Arc::new(Observed::default());
        let beta = Arc::new(Observed::default());
        let (alpha_terminate, alpha_terminate_rx) = watch::channel(false);
        let (_alpha_abandon, alpha_abandon_rx) = watch::channel(false);
        let (_beta_terminate, beta_terminate_rx) = watch::channel(false);
        let (beta_abandon, beta_abandon_rx) = watch::channel(false);
        let alpha_thread = host(
            "sandbox-alpha",
            alpha.clone(),
            alpha_terminate_rx,
            alpha_abandon_rx,
        );
        let beta_thread = host(
            "sandbox-beta",
            beta.clone(),
            beta_terminate_rx,
            beta_abandon_rx,
        );

        wait_until("both sessions running", || {
            alpha.ticks.load(Ordering::SeqCst) > 3 && beta.ticks.load(Ordering::SeqCst) > 3
        });

        // Stop A (sandbox stop/delete): A runs its shutdown path and its
        // runtime, tasks and thread end.
        alpha_terminate.send(true).unwrap();
        wait_until("alpha thread exits", || alpha_thread.is_finished());
        alpha_thread.join().unwrap();
        assert!(alpha.shutdown_path_ran.load(Ordering::SeqCst));
        assert!(alpha.task_dropped.load(Ordering::SeqCst));

        // B is unaffected and keeps making progress.
        let beta_before = beta.ticks.load(Ordering::SeqCst);
        wait_until("beta keeps running", || {
            beta.ticks.load(Ordering::SeqCst) > beta_before + 3
        });
        assert!(!beta_thread.is_finished());
        assert!(!beta.task_dropped.load(Ordering::SeqCst));

        // Ownership of B moves away: B is dropped without running its
        // shutdown path (the workload stays frozen for the next owner).
        beta_abandon.send(true).unwrap();
        wait_until("beta thread exits", || beta_thread.is_finished());
        beta_thread.join().unwrap();
        assert!(!beta.shutdown_path_ran.load(Ordering::SeqCst));
        wait_until("abandoned tasks are dropped", || {
            beta.task_dropped.load(Ordering::SeqCst)
        });

        assert!(
            !alpha.misattributed.load(Ordering::SeqCst),
            "alpha work observed another session's identity or OCSF context"
        );
        assert!(
            !beta.misattributed.load(Ordering::SeqCst),
            "beta work observed another session's identity or OCSF context"
        );
    }

    mod pool_rpc {
        use super::super::{run_idempotent, run_once};
        use std::sync::Arc;
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::time::Duration;

        const ATTEMPT: Duration = Duration::from_millis(1_500);
        const BUDGET: Duration = Duration::from_secs(4);

        #[allow(clippy::unnecessary_wraps)]
        fn ok() -> Result<tonic::Response<u32>, tonic::Status> {
            Ok(tonic::Response::new(7))
        }

        #[tokio::test(start_paused = true)]
        async fn idempotent_calls_retry_past_a_hung_gateway_on_a_new_connection() {
            let attempts = Arc::new(AtomicUsize::new(0));
            let reconnects = Arc::new(AtomicUsize::new(0));
            let result = run_idempotent(
                "renew lease",
                BUDGET,
                ATTEMPT,
                || {
                    let n = attempts.fetch_add(1, Ordering::SeqCst);
                    async move {
                        if n == 0 {
                            // The first gateway hangs (paused, not dead).
                            std::future::pending::<()>().await;
                        }
                        ok()
                    }
                },
                |_| {
                    reconnects.fetch_add(1, Ordering::SeqCst);
                    async {}
                },
            )
            .await;
            assert_eq!(result.unwrap(), 7);
            assert_eq!(attempts.load(Ordering::SeqCst), 2);
            assert_eq!(reconnects.load(Ordering::SeqCst), 1);
        }

        #[tokio::test(start_paused = true)]
        async fn application_errors_are_not_retried() {
            let attempts = AtomicUsize::new(0);
            let result: Result<u32, _> = run_idempotent(
                "renew lease",
                BUDGET,
                ATTEMPT,
                || {
                    attempts.fetch_add(1, Ordering::SeqCst);
                    async { Err(tonic::Status::permission_denied("not the owner")) }
                },
                |_| async {},
            )
            .await;
            assert_eq!(result.unwrap_err().code(), tonic::Code::PermissionDenied);
            assert_eq!(attempts.load(Ordering::SeqCst), 1);
        }

        #[tokio::test(start_paused = true)]
        async fn a_hanging_reconnect_cannot_outlive_the_budget() {
            let started = tokio::time::Instant::now();
            let result: Result<u32, _> = run_idempotent(
                "renew lease",
                BUDGET,
                ATTEMPT,
                || async { Err(tonic::Status::unavailable("connection reset")) },
                |_| std::future::pending::<()>(),
            )
            .await;
            assert_eq!(result.unwrap_err().code(), tonic::Code::Unavailable);
            assert!(
                started.elapsed() <= BUDGET,
                "elapsed {:?}",
                started.elapsed()
            );
        }

        #[tokio::test(start_paused = true)]
        async fn grant_and_activation_are_attempted_once() {
            // A lost response (hang) is NOT retried: the next attempt could
            // see "grant already issued" for a grant that was committed.
            let reconnects = AtomicUsize::new(0);
            let started = tokio::time::Instant::now();
            let result: Result<u32, _> = run_once(
                "request recovery grant",
                ATTEMPT,
                async {
                    std::future::pending::<()>().await;
                    ok()
                },
                |_| {
                    reconnects.fetch_add(1, Ordering::SeqCst);
                    std::future::pending::<()>()
                },
            )
            .await;
            assert_eq!(result.unwrap_err().code(), tonic::Code::DeadlineExceeded);
            assert_eq!(
                reconnects.load(Ordering::SeqCst),
                1,
                "connection replaced for later calls"
            );
            assert!(started.elapsed() <= ATTEMPT * 2);
        }
    }
}
