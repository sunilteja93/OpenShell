// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Gateway supervisor topology prototype (`supervisor_topology = "gateway"`).
//!
//! Runs against a gateway installed with
//! `deploy/helm/openshell/ci/values-supervisor-gateway-topology.yaml`: two
//! gateway replicas on `PostgreSQL` and one shared supervisor Deployment with two
//! replicas for every sandbox. The test proves, with real Pods:
//!
//! - several sandboxes run through the shared pool with different policies and
//!   credentials;
//! - killing the supervisor Pod that owns a sandbox freezes and then resumes
//!   the *same* agent process (same PID, same workload Pod, counter continues)
//!   on the surviving replica, with policy and credential isolation intact;
//! - pausing the elected assignment coordinator gateway hands coordination
//!   to the other gateway with a higher term while every healthy agent keeps
//!   running; the resumed gateway stays a follower;
//! - a drain and an explicit rebalance through the operator API move
//!   sessions with fenced handoffs that keep the same agent processes;
//! - deleting a sandbox while its recovery is pending wins, and never removes
//!   the shared Deployment or disturbs other sandboxes.

#![cfg(feature = "e2e-kubernetes-supervisor-topology")]

use std::io::Write as _;
use std::process::Stdio;
use std::time::{Duration, Instant};

use openshell_e2e::harness::binary::openshell_cmd;
use openshell_e2e::harness::output::strip_ansi;
use openshell_e2e::harness::sandbox::{E2E_WORKLOAD_IMAGE, SandboxGuard, unique_sandbox_name};
use serde_json::Value;
use tempfile::NamedTempFile;
use tokio::io::{AsyncReadExt as _, AsyncWriteExt as _};
use tokio::net::TcpListener;
use tokio::process::Command;

const POOL_DEPLOYMENT: &str = "openshell-supervisor-pool";
const POOL_SELECTOR: &str = "openshell.ai/component=supervisor-pool";
const PROVIDER_A: &str = "e2e-topology-provider-a";
const SECRET_A: &str = "e2e-topology-secret-a";
const RECOVERY_TIMEOUT: Duration = Duration::from_secs(90);

const COUNTER_AGENT: &str = r#"echo $$ > /tmp/agent.pid
i=0
while :; do
  i=$((i+1))
  echo "$i" > /tmp/agent.counter
  sleep 1
done"#;

struct Kube {
    context: String,
    namespace: String,
}

impl Kube {
    fn from_env() -> Self {
        Self {
            context: std::env::var("OPENSHELL_E2E_KUBE_CONTEXT")
                .expect("OPENSHELL_E2E_KUBE_CONTEXT must be set"),
            namespace: std::env::var("OPENSHELL_E2E_KUBE_NAMESPACE")
                .unwrap_or_else(|_| "openshell".to_string()),
        }
    }

    async fn run(&self, args: &[&str]) -> Result<String, String> {
        let output = Command::new("kubectl")
            .arg("--context")
            .arg(&self.context)
            .arg("-n")
            .arg(&self.namespace)
            .args(args)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .await
            .map_err(|error| format!("spawn kubectl {args:?}: {error}"))?;
        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        if !output.status.success() {
            return Err(format!(
                "kubectl {args:?} failed: {stdout}{}",
                String::from_utf8_lossy(&output.stderr)
            ));
        }
        Ok(stdout)
    }

    async fn json(&self, args: &[&str]) -> Result<Value, String> {
        let mut full = args.to_vec();
        full.extend(["-o", "json"]);
        let output = self.run(&full).await?;
        serde_json::from_str(&output).map_err(|error| format!("parse kubectl JSON: {error}"))
    }

    async fn pool_pods(&self) -> Result<Vec<(String, bool)>, String> {
        let pods = self.json(&["get", "pods", "-l", POOL_SELECTOR]).await?;
        Ok(pods["items"]
            .as_array()
            .cloned()
            .unwrap_or_default()
            .into_iter()
            .filter(|pod| pod["metadata"]["deletionTimestamp"].is_null())
            .map(|pod| {
                let ready = pod["status"]["conditions"]
                    .as_array()
                    .is_some_and(|conditions| {
                        conditions.iter().any(|condition| {
                            condition["type"] == "Ready" && condition["status"] == "True"
                        })
                    });
                (
                    pod["metadata"]["name"]
                        .as_str()
                        .unwrap_or_default()
                        .to_string(),
                    ready,
                )
            })
            .collect())
    }

    /// The live pool Pod whose logs show it hosting `sandbox_id` most recently.
    async fn owner_of(&self, sandbox_id: &str) -> Result<String, String> {
        let mut best: Option<(String, String)> = None;
        for (pod, _) in self.pool_pods().await? {
            let logs = self
                .run(&["logs", &pod, "--timestamps"])
                .await
                .unwrap_or_default();
            for line in logs.lines() {
                if line.contains("Starting hosted supervisor session") && line.contains(sandbox_id)
                {
                    let timestamp = line
                        .split_whitespace()
                        .next()
                        .unwrap_or_default()
                        .to_string();
                    if best.as_ref().is_none_or(|(_, seen)| timestamp > *seen) {
                        best = Some((pod.clone(), timestamp));
                    }
                }
            }
        }
        best.map(|(pod, _)| pod)
            .ok_or_else(|| format!("no pool replica hosts sandbox {sandbox_id}"))
    }

    async fn workload_pod_uid(&self, sandbox_id: &str) -> Result<String, String> {
        let pods = self
            .json(&[
                "get",
                "pods",
                "-l",
                &format!(
                    "openshell.ai/boundary-pair={},openshell.ai/boundary-role=workload",
                    sandbox_id.to_ascii_lowercase()
                ),
            ])
            .await?;
        pods["items"][0]["metadata"]["uid"]
            .as_str()
            .map(str::to_string)
            .ok_or_else(|| format!("workload Pod for {sandbox_id} not found"))
    }
}

/// Docker container of the k3d node that runs the pool Pods. Used only to
/// pause and resume one supervisor process, simulating a partitioned owner.
fn node_container(context: &str) -> String {
    std::env::var("OPENSHELL_E2E_KUBE_NODE_CONTAINER")
        .unwrap_or_else(|_| format!("{}-server-0", context.trim_start_matches("kind-")))
}

async fn node_shell(node: &str, script: &str) -> Result<String, String> {
    let output = Command::new("docker")
        .args(["exec", node, "sh", "-c", script])
        .output()
        .await
        .map_err(|error| format!("docker exec {node}: {error}"))?;
    if !output.status.success() {
        return Err(format!(
            "node command failed: {}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    Ok(String::from_utf8_lossy(&output.stdout).to_string())
}

/// Host PID (in the node) of the supervisor process in `pod`.
async fn supervisor_host_pid(node: &str, pod: &str) -> Result<u32, String> {
    let output = node_shell(
        node,
        &format!(
            "crictl inspect -o go-template --template '{{{{.info.pid}}}}' $(crictl ps --name supervisor --label io.kubernetes.pod.name={pod} -q)"
        ),
    )
    .await?;
    let pid: u32 = output
        .trim()
        .parse()
        .map_err(|error| format!("parse supervisor pid from {output:?}: {error}"))?;
    if pid <= 1 {
        return Err(format!("refusing to signal node pid {pid}"));
    }
    Ok(pid)
}

async fn cli(args: &[&str]) -> Result<String, String> {
    let output = openshell_cmd()
        .args(args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .await
        .map_err(|error| format!("spawn openshell {args:?}: {error}"))?;
    let combined = strip_ansi(&format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    ));
    if !output.status.success() {
        return Err(format!("openshell {args:?} failed: {combined}"));
    }
    Ok(combined)
}

async fn sandbox_id(name: &str) -> Result<String, String> {
    let output = cli(&["sandbox", "get", name, "--output", "json"]).await?;
    let start = output.find('{').ok_or("sandbox get printed no JSON")?;
    let value: Value = serde_json::from_str(&output[start..])
        .map_err(|error| format!("parse sandbox: {error}"))?;
    value["id"]
        .as_str()
        .or_else(|| value["metadata"]["id"].as_str())
        .or_else(|| value["sandbox"]["metadata"]["id"].as_str())
        .map(str::to_string)
        .ok_or_else(|| format!("sandbox get JSON has no id: {value}"))
}

async fn exec(name: &str, script: &str) -> Result<String, String> {
    cli(&[
        "sandbox",
        "exec",
        "--name",
        name,
        "--no-tty",
        "--",
        "/usr/bin/bash",
        "-c",
        script,
    ])
    .await
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct AgentState {
    pid: u64,
    counter: u64,
}

async fn agent_state(name: &str) -> Result<AgentState, String> {
    let output = exec(
        name,
        "echo PID=$(cat /tmp/agent.pid) COUNTER=$(cat /tmp/agent.counter)",
    )
    .await?;
    let field = |key: &str| {
        output
            .split_whitespace()
            .find_map(|token| token.strip_prefix(key))
            .and_then(|value| value.trim().parse::<u64>().ok())
    };
    match (field("PID="), field("COUNTER=")) {
        (Some(pid), Some(counter)) => Ok(AgentState { pid, counter }),
        _ => Err(format!("agent state unavailable: {output}")),
    }
}

async fn wait_for_agent(name: &str, timeout: Duration) -> Result<(AgentState, Duration), String> {
    let started = Instant::now();
    loop {
        match agent_state(name).await {
            Ok(state) if state.counter >= 1 => return Ok((state, started.elapsed())),
            result => {
                if started.elapsed() > timeout {
                    return Err(format!("agent {name} not reachable: {result:?}"));
                }
            }
        }
        tokio::time::sleep(Duration::from_millis(250)).await;
    }
}

fn policy_file(allowed_port: Option<u16>) -> NamedTempFile {
    let network = allowed_port.map_or_else(String::new, |port| {
        format!(
            r"
network_policies:
  host_echo:
    name: host_echo
    endpoints:
      - host: host.openshell.internal
        port: {port}
        path: /**
        protocol: rest
        access: full
        enforcement: enforce
    binaries:
      - path: /usr/bin/bash
"
        )
    });
    let mut file = NamedTempFile::new().expect("policy file");
    write!(
        file,
        r"version: 1

filesystem_policy:
  include_workdir: true
  read_only: [/bin, /usr, /lib, /proc, /dev/urandom, /app, /etc, /var/log]
  read_write: [/sandbox, /tmp, /dev/null]

landlock:
  compatibility: best_effort

process:
  run_as_user: sandbox
  run_as_group: sandbox
{network}"
    )
    .expect("write policy");
    file
}

async fn start_host_server() -> (u16, tokio::task::JoinHandle<()>) {
    let listener = TcpListener::bind(("0.0.0.0", 0))
        .await
        .expect("bind host server");
    let port = listener.local_addr().expect("host server address").port();
    let task = tokio::spawn(async move {
        while let Ok((mut stream, _)) = listener.accept().await {
            tokio::spawn(async move {
                let mut buffer = [0_u8; 2048];
                let mut request = Vec::new();
                while let Ok(read) = stream.read(&mut buffer).await {
                    if read == 0 {
                        return;
                    }
                    request.extend_from_slice(&buffer[..read]);
                    if request.windows(4).any(|window| window == b"\r\n\r\n") {
                        break;
                    }
                }
                let body = "topology-policy-allowed";
                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                );
                let _ = stream.write_all(response.as_bytes()).await;
                let _ = stream.shutdown().await;
            });
        }
    });
    (port, task)
}

async fn http_probe(name: &str, port: u16) -> String {
    let script = format!(
        r#"if exec 3<>/dev/tcp/host.openshell.internal/{port}; then
  printf 'GET /check HTTP/1.1\r\nHost: host.openshell.internal:{port}\r\nConnection: close\r\n\r\n' >&3
  out=$(cat <&3)
  case "$out" in *topology-policy-allowed*) echo PROBE=allowed ;; *) echo "PROBE=denied:$(printf '%s' "$out" | head -1)" ;; esac
else
  echo PROBE=connect-denied
fi"#
    );
    exec(name, &script)
        .await
        .unwrap_or_else(|error| format!("PROBE=error {error}"))
}

/// Check policy enforcement and credential isolation for both sandboxes.
async fn assert_isolation(phase: &str, allowed: &str, denied: &str, port: u16) {
    let allowed_probe = http_probe(allowed, port).await;
    assert!(
        allowed_probe.contains("PROBE=allowed"),
        "[{phase}] policy must allow the host echo from {allowed}: {allowed_probe}"
    );
    let denied_probe = http_probe(denied, port).await;
    assert!(
        !denied_probe.contains("PROBE=allowed"),
        "[{phase}] policy must deny the host echo from {denied}: {denied_probe}"
    );
    let allowed_env = exec(allowed, "printenv OPENAI_API_KEY || echo NO_KEY")
        .await
        .unwrap();
    let denied_env = exec(denied, "printenv OPENAI_API_KEY || echo NO_KEY")
        .await
        .unwrap();
    assert!(
        allowed_env.contains("openshell:resolve:env:"),
        "[{phase}] {allowed} must receive its provider placeholder: {allowed_env}"
    );
    assert!(
        denied_env.contains("NO_KEY"),
        "[{phase}] {denied} must not receive another sandbox's provider: {denied_env}"
    );
    assert!(
        !allowed_env.contains(SECRET_A) && !denied_env.contains(SECRET_A),
        "[{phase}] provider secret values must never enter the workload"
    );
    eprintln!("METRIC isolation[{phase}] allowed={allowed_probe:?} denied={denied_probe:?}");
}

async fn create_agent(create_args: &[&str]) -> SandboxGuard {
    let name = unique_sandbox_name();
    let mut args = vec![
        "sandbox",
        "create",
        "--detach",
        "--name",
        &name,
        "--from",
        E2E_WORKLOAD_IMAGE,
        "--no-auto-providers",
    ];
    args.extend_from_slice(create_args);
    args.extend(["--", "/usr/bin/bash", "-c", COUNTER_AGENT]);
    cli(&args).await.expect("create counter sandbox");
    SandboxGuard::manage_existing(name)
}

async fn delete_provider(name: &str) {
    let _ = cli(&["provider", "delete", name]).await;
}

/// Minimal client for the supervisor pool operator API. Messages mirror the
/// fields of `proto/openshell.proto` the test reads; unknown fields are
/// ignored by prost.
mod pool_api {
    use prost::Message;

    #[derive(Clone, PartialEq, Message)]
    pub struct Timestamp {
        #[prost(int64, tag = "1")]
        pub seconds: i64,
        #[prost(int32, tag = "2")]
        pub nanos: i32,
    }

    #[derive(Clone, PartialEq, Message)]
    pub struct Empty {}

    #[derive(Clone, PartialEq, Message)]
    pub struct SessionMove {
        #[prost(string, tag = "1")]
        pub sandbox: String,
        #[prost(string, tag = "2")]
        pub from_member: String,
        #[prost(string, tag = "3")]
        pub to_member: String,
        #[prost(string, tag = "4")]
        pub state: String,
        #[prost(string, tag = "5")]
        pub message: String,
    }

    #[derive(Clone, PartialEq, Message)]
    pub struct Operation {
        #[prost(string, tag = "1")]
        pub name: String,
        #[prost(int32, tag = "2")]
        pub kind: i32,
        #[prost(int32, tag = "3")]
        pub phase: i32,
        #[prost(string, tag = "4")]
        pub message: String,
        #[prost(string, tag = "9")]
        pub member: String,
        #[prost(message, repeated, tag = "11")]
        pub moves: Vec<SessionMove>,
        #[prost(uint64, tag = "13")]
        pub coordinator_term: u64,
    }

    #[derive(Clone, PartialEq, Message)]
    pub struct Coordinator {
        #[prost(string, tag = "1")]
        pub holder: String,
        #[prost(uint64, tag = "2")]
        pub term: u64,
        #[prost(message, optional, tag = "3")]
        pub expiration_time: Option<Timestamp>,
        #[prost(bool, tag = "4")]
        pub expired: bool,
    }

    #[derive(Clone, PartialEq, Message)]
    pub struct Member {
        #[prost(string, tag = "1")]
        pub name: String,
        #[prost(bool, tag = "2")]
        pub live: bool,
        #[prost(bool, tag = "3")]
        pub draining: bool,
        #[prost(uint32, tag = "5")]
        pub sessions: u32,
    }

    #[derive(Clone, PartialEq, Message)]
    pub struct Status {
        #[prost(message, optional, tag = "1")]
        pub coordinator: Option<Coordinator>,
        #[prost(message, repeated, tag = "2")]
        pub members: Vec<Member>,
        #[prost(message, repeated, tag = "3")]
        pub operations: Vec<Operation>,
        #[prost(string, tag = "4")]
        pub serving_gateway: String,
    }

    #[derive(Clone, PartialEq, Message)]
    pub struct StartRequest {
        #[prost(int32, tag = "1")]
        pub kind: i32,
        #[prost(string, tag = "2")]
        pub member: String,
        #[prost(uint32, optional, tag = "3")]
        pub max_parallel: Option<u32>,
    }

    #[derive(Clone, PartialEq, Message)]
    pub struct OperationResponse {
        #[prost(message, optional, tag = "1")]
        pub operation: Option<Operation>,
    }

    #[derive(Clone, PartialEq, Message)]
    pub struct CancelRequest {
        #[prost(string, tag = "1")]
        pub name: String,
    }

    pub const KIND_DRAIN: i32 = 3;
    pub const KIND_REBALANCE: i32 = 4;
    pub const PHASE_SUCCEEDED: i32 = 4;
    pub const PHASE_FAILED: i32 = 5;
    pub const PHASE_CANCELLED: i32 = 6;

    async fn unary<Req, Resp>(
        endpoint: &str,
        method: &'static str,
        request: Req,
    ) -> Result<Resp, String>
    where
        Req: Message + Send + Sync + 'static,
        Resp: Message + Default + Send + Sync + 'static,
    {
        let channel = tonic::transport::Endpoint::from_shared(endpoint.to_string())
            .map_err(|error| error.to_string())?
            .connect_timeout(std::time::Duration::from_secs(3))
            .timeout(std::time::Duration::from_secs(5))
            .connect()
            .await
            .map_err(|error| format!("connect {endpoint}: {error}"))?;
        let mut grpc = tonic::client::Grpc::new(channel);
        grpc.ready().await.map_err(|error| error.to_string())?;
        let path = tonic::codegen::http::uri::PathAndQuery::from_static(method);
        grpc.unary(
            tonic::Request::new(request),
            path,
            tonic_prost::ProstCodec::<Req, Resp>::default(),
        )
        .await
        .map(tonic::Response::into_inner)
        .map_err(|status| format!("{method}: {status}"))
    }

    pub async fn status(endpoint: &str) -> Result<Status, String> {
        unary(
            endpoint,
            "/openshell.v1.OpenShell/GetSupervisorPoolStatus",
            Empty {},
        )
        .await
    }

    pub async fn start(endpoint: &str, kind: i32, member: &str) -> Result<Operation, String> {
        let response: OperationResponse = unary(
            endpoint,
            "/openshell.v1.OpenShell/StartSupervisorPoolOperation",
            StartRequest {
                kind,
                member: member.to_string(),
                max_parallel: Some(1),
            },
        )
        .await?;
        response
            .operation
            .ok_or_else(|| "no operation returned".to_string())
    }

    pub async fn cancel(endpoint: &str, name: &str) -> Result<Operation, String> {
        let response: OperationResponse = unary(
            endpoint,
            "/openshell.v1.OpenShell/CancelSupervisorPoolOperation",
            CancelRequest {
                name: name.to_string(),
            },
        )
        .await?;
        response
            .operation
            .ok_or_else(|| "no operation returned".to_string())
    }
}

/// `kubectl port-forward` to one gateway Pod; killed on drop.
struct PortForward {
    _child: tokio::process::Child,
    endpoint: String,
}

impl Kube {
    async fn gateway_pods(&self) -> Result<Vec<String>, String> {
        let pods = self
            .json(&["get", "pods", "-l", "app.kubernetes.io/name=openshell"])
            .await?;
        Ok(pods["items"]
            .as_array()
            .cloned()
            .unwrap_or_default()
            .into_iter()
            .filter(|pod| pod["metadata"]["deletionTimestamp"].is_null())
            .filter_map(|pod| pod["metadata"]["name"].as_str().map(str::to_string))
            .collect())
    }

    async fn port_forward(&self, pod: &str) -> Result<PortForward, String> {
        use tokio::io::AsyncBufReadExt as _;
        let mut child = Command::new("kubectl")
            .args(["--context", &self.context, "-n", &self.namespace])
            .args(["port-forward", &format!("pod/{pod}"), ":8080"])
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .kill_on_drop(true)
            .spawn()
            .map_err(|error| format!("spawn port-forward: {error}"))?;
        let stdout = child.stdout.take().ok_or("port-forward stdout")?;
        let mut lines = tokio::io::BufReader::new(stdout).lines();
        let line = tokio::time::timeout(Duration::from_secs(20), lines.next_line())
            .await
            .map_err(|_| "port-forward did not start".to_string())?
            .map_err(|error| error.to_string())?
            .ok_or("port-forward exited")?;
        // "Forwarding from 127.0.0.1:43123 -> 8080"
        let port = line
            .split("127.0.0.1:")
            .nth(1)
            .and_then(|rest| rest.split_whitespace().next())
            .ok_or_else(|| format!("unexpected port-forward output: {line}"))?
            .to_string();
        tokio::spawn(async move { while let Ok(Some(_)) = lines.next_line().await {} });
        Ok(PortForward {
            _child: child,
            endpoint: format!("http://127.0.0.1:{port}"),
        })
    }

    async fn workload_pod(&self, sandbox_id: &str) -> Result<String, String> {
        let pods = self
            .json(&[
                "get",
                "pods",
                "-l",
                &format!(
                    "openshell.ai/boundary-pair={},openshell.ai/boundary-role=workload",
                    sandbox_id.to_ascii_lowercase()
                ),
            ])
            .await?;
        pods["items"][0]["metadata"]["name"]
            .as_str()
            .map(str::to_string)
            .ok_or_else(|| format!("workload Pod for {sandbox_id} not found"))
    }

    /// Read the agent's PID and counter directly in its workload Pod, without
    /// any gateway or supervisor on the path.
    async fn agent_state_direct(&self, sandbox_id: &str) -> Result<AgentState, String> {
        let pod = self.workload_pod(sandbox_id).await?;
        let output = self
            .run(&[
                "exec",
                &pod,
                "--",
                "sh",
                "-c",
                "echo PID=$(cat /tmp/agent.pid) COUNTER=$(cat /tmp/agent.counter)",
            ])
            .await?;
        let field = |key: &str| {
            output
                .split_whitespace()
                .find_map(|token| token.strip_prefix(key))
                .and_then(|value| value.trim().parse::<u64>().ok())
        };
        match (field("PID="), field("COUNTER=")) {
            (Some(pid), Some(counter)) => Ok(AgentState { pid, counter }),
            _ => Err(format!("agent state unavailable in {pod}: {output}")),
        }
    }
}

async fn wait_for_operation(
    endpoint: &str,
    name: &str,
    timeout: Duration,
) -> Result<(pool_api::Operation, Duration), String> {
    let started = Instant::now();
    loop {
        let status = pool_api::status(endpoint).await;
        if let Ok(status) = &status
            && let Some(operation) = status.operations.iter().find(|op| op.name == name)
            && matches!(
                operation.phase,
                pool_api::PHASE_SUCCEEDED | pool_api::PHASE_FAILED | pool_api::PHASE_CANCELLED
            )
        {
            return Ok((operation.clone(), started.elapsed()));
        }
        if started.elapsed() > timeout {
            return Err(format!("operation {name} did not finish: {status:?}"));
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
}

/// Host PID (in the node) of the gateway process in `pod`.
async fn gateway_host_pid(node: &str, pod: &str) -> Result<u32, String> {
    let output = node_shell(
        node,
        &format!(
            "crictl inspect -o go-template --template '{{{{.info.pid}}}}' $(crictl ps --name openshell-gateway --label io.kubernetes.pod.name={pod} -q)"
        ),
    )
    .await?;
    let pid: u32 = output
        .trim()
        .parse()
        .map_err(|error| format!("parse gateway pid from {output:?}: {error}"))?;
    if pid <= 1 {
        return Err(format!("refusing to signal node pid {pid}"));
    }
    Ok(pid)
}

/// Pause the elected coordinator gateway and prove that (1) another gateway
/// takes over with a higher term, (2) every healthy agent keeps running
/// without a freeze or an ownership change, (3) the resumed gateway stays a
/// follower.
#[allow(clippy::too_many_lines)]
async fn coordinator_pause_phase(kube: &Kube, sandbox_ids: &[&str]) {
    let gateways = kube.gateway_pods().await.unwrap();
    assert_eq!(gateways.len(), 2, "two gateway replicas: {gateways:?}");
    let probe = kube.port_forward(&gateways[0]).await.unwrap();
    let status = pool_api::status(&probe.endpoint).await.unwrap();
    let coordinator = status.coordinator.expect("a coordinator is elected");
    let coordinator_pod = coordinator.holder.split('/').next().unwrap().to_string();
    assert!(gateways.contains(&coordinator_pod), "{coordinator:?}");
    let follower = gateways
        .iter()
        .find(|pod| **pod != coordinator_pod)
        .unwrap();
    drop(probe);
    let api = kube.port_forward(follower).await.unwrap();
    let mut owners = Vec::new();
    let mut before = Vec::new();
    for id in sandbox_ids {
        owners.push(kube.owner_of(id).await.unwrap());
        before.push(kube.agent_state_direct(id).await.unwrap());
    }

    let node = node_container(&kube.context);
    let pid = gateway_host_pid(&node, &coordinator_pod).await.unwrap();
    let paused_at = Instant::now();
    node_shell(&node, &format!("kill -STOP {pid}"))
        .await
        .unwrap();
    let mut failover = None;
    let mut last = before.clone();
    let mut max_gap = Duration::ZERO;
    let mut last_progress = vec![Instant::now(); sandbox_ids.len()];
    // Longer than the 5 s coordinator TTL, shorter than the gateway liveness
    // probe budget (3 x 5 s) so the paused process is not restarted.
    while paused_at.elapsed() < Duration::from_secs(11) {
        if failover.is_none()
            && let Ok(status) = pool_api::status(&api.endpoint).await
            && let Some(current) = status.coordinator
            && current.term > coordinator.term
            && !current.holder.starts_with(&coordinator_pod)
        {
            failover = Some((paused_at.elapsed(), current));
        }
        for (index, id) in sandbox_ids.iter().enumerate() {
            let state = kube.agent_state_direct(id).await.unwrap();
            assert_eq!(
                state.pid, before[index].pid,
                "the agent process never changed"
            );
            if state.counter > last[index].counter {
                max_gap = max_gap.max(last_progress[index].elapsed());
                last_progress[index] = Instant::now();
                last[index] = state;
            }
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
    for progress in &last_progress {
        max_gap = max_gap.max(progress.elapsed());
    }
    node_shell(&node, &format!("kill -CONT {pid}"))
        .await
        .unwrap();
    let (failover_after, new_coordinator) =
        failover.expect("another gateway became coordinator while the old one was paused");
    eprintln!(
        "METRIC coordinator_pause old={} term={} new={} term={} failover_ms={} max_agent_counter_gap_ms={} counters_before={:?} counters_after={:?}",
        coordinator.holder,
        coordinator.term,
        new_coordinator.holder,
        new_coordinator.term,
        failover_after.as_millis(),
        max_gap.as_millis(),
        before.iter().map(|s| s.counter).collect::<Vec<_>>(),
        last.iter().map(|s| s.counter).collect::<Vec<_>>()
    );
    assert!(
        max_gap < Duration::from_secs(4),
        "healthy agents must not freeze while the coordinator changes (max gap {max_gap:?})"
    );
    for (index, id) in sandbox_ids.iter().enumerate() {
        assert!(
            last[index].counter >= before[index].counter + 7,
            "{id} kept running"
        );
        assert_eq!(
            kube.owner_of(id).await.unwrap(),
            owners[index],
            "a coordinator change never moves a healthy assignment"
        );
    }
    // The resumed gateway discovers it lost the lease and stays a follower.
    tokio::time::sleep(Duration::from_secs(5)).await;
    let status = pool_api::status(&api.endpoint).await.unwrap();
    let current = status.coordinator.unwrap();
    assert_eq!(current.holder, new_coordinator.holder);
    let mut logs = kube
        .run(&["logs", &coordinator_pod])
        .await
        .unwrap_or_default();
    logs.push_str(
        &kube
            .run(&["logs", &coordinator_pod, "--previous"])
            .await
            .unwrap_or_default(),
    );
    let stepped_down = logs
        .lines()
        .rev()
        .find(|line| line.contains("stepping down") && line.contains("coordinator"));
    eprintln!("METRIC coordinator_pause old_coordinator_reaction={stepped_down:?}");
    assert!(
        stepped_down.is_some(),
        "the paused coordinator stepped down when it resumed"
    );
}

/// Drain the busiest replica, then cancel the drain and rebalance; every
/// handoff preserves the agent process.
#[allow(clippy::too_many_lines)]
async fn drain_and_rebalance_phase(kube: &Kube, sandboxes: &[(&str, &str)], port: u16) {
    let gateways = kube.gateway_pods().await.unwrap();
    let api = kube.port_forward(&gateways[0]).await.unwrap();
    let mut before = Vec::new();
    let mut workload_uids = Vec::new();
    for (_, id) in sandboxes {
        before.push(kube.agent_state_direct(id).await.unwrap());
        workload_uids.push(kube.workload_pod_uid(id).await.unwrap());
    }
    let status = pool_api::status(&api.endpoint).await.unwrap();
    let busiest = status
        .members
        .iter()
        .filter(|m| m.live)
        .max_by_key(|m| m.sessions)
        .unwrap()
        .clone();
    eprintln!("METRIC drain members_before={:?}", status.members);

    let drain = pool_api::start(&api.endpoint, pool_api::KIND_DRAIN, &busiest.name)
        .await
        .unwrap();
    let (drained, took) = wait_for_operation(&api.endpoint, &drain.name, Duration::from_secs(180))
        .await
        .unwrap();
    eprintln!(
        "METRIC drain member={} sessions={} phase={} took_ms={} moves={:?}",
        busiest.name,
        busiest.sessions,
        drained.phase,
        took.as_millis(),
        drained.moves
    );
    assert_eq!(
        drained.phase,
        pool_api::PHASE_SUCCEEDED,
        "{}",
        drained.message
    );
    assert_eq!(drained.moves.len(), busiest.sessions as usize);
    assert!(
        drained.moves.iter().all(|m| m.state == "done"),
        "{:?}",
        drained.moves
    );
    let status = pool_api::status(&api.endpoint).await.unwrap();
    let drained_member = status
        .members
        .iter()
        .find(|m| m.name == busiest.name)
        .unwrap();
    assert!(drained_member.draining);
    assert_eq!(drained_member.sessions, 0);
    for (index, (name, id)) in sandboxes.iter().enumerate() {
        assert_ne!(
            kube.owner_of(id).await.unwrap(),
            busiest.name,
            "{name} moved off"
        );
        let (state, _) = wait_for_agent(name, RECOVERY_TIMEOUT).await.unwrap();
        assert_eq!(
            state.pid, before[index].pid,
            "{name}: same agent process after drain"
        );
        assert!(state.counter > before[index].counter);
        assert_eq!(
            kube.workload_pod_uid(id).await.unwrap(),
            workload_uids[index]
        );
    }
    assert_isolation("after-drain", sandboxes[0].0, sandboxes[1].0, port).await;

    // Return the member to service and rebalance.
    let cancelled = pool_api::cancel(&api.endpoint, &drain.name).await.unwrap();
    assert_eq!(cancelled.phase, pool_api::PHASE_CANCELLED);
    let rebalance = pool_api::start(&api.endpoint, pool_api::KIND_REBALANCE, "")
        .await
        .unwrap();
    let (balanced, took) =
        wait_for_operation(&api.endpoint, &rebalance.name, Duration::from_secs(180))
            .await
            .unwrap();
    let status = pool_api::status(&api.endpoint).await.unwrap();
    let mut loads: Vec<u32> = status
        .members
        .iter()
        .filter(|m| m.live)
        .map(|m| m.sessions)
        .collect();
    loads.sort_unstable();
    eprintln!(
        "METRIC rebalance phase={} took_ms={} moves={:?} loads={loads:?}",
        balanced.phase,
        took.as_millis(),
        balanced.moves
    );
    assert_eq!(
        balanced.phase,
        pool_api::PHASE_SUCCEEDED,
        "{}",
        balanced.message
    );
    assert!(
        loads.last().unwrap() - loads.first().unwrap() <= 1,
        "balanced: {loads:?}"
    );
    for (index, (name, id)) in sandboxes.iter().enumerate() {
        let (state, _) = wait_for_agent(name, RECOVERY_TIMEOUT).await.unwrap();
        assert_eq!(
            state.pid, before[index].pid,
            "{name}: same agent process after rebalance"
        );
        assert_eq!(
            kube.workload_pod_uid(id).await.unwrap(),
            workload_uids[index]
        );
    }
    assert_isolation("after-rebalance", sandboxes[0].0, sandboxes[1].0, port).await;
}

#[tokio::test]
#[allow(clippy::too_many_lines)]
async fn shared_supervisor_pool_resumes_running_agents_after_owner_loss() {
    let kube = Kube::from_env();

    // The installation runs one shared, two-replica supervisor Deployment.
    let deployment = kube
        .json(&["get", "deployment", POOL_DEPLOYMENT])
        .await
        .expect("shared supervisor Deployment exists");
    assert_eq!(deployment["spec"]["replicas"], 2);
    assert!(
        deployment["metadata"]["ownerReferences"].is_null(),
        "the shared Deployment must not be owned by a sandbox"
    );
    let deployment_uid = deployment["metadata"]["uid"].as_str().unwrap().to_string();

    let (port, _server) = start_host_server().await;
    let allow_policy = policy_file(Some(port));
    let deny_policy = policy_file(None);
    delete_provider(PROVIDER_A).await;
    cli(&[
        "provider",
        "create",
        "--name",
        PROVIDER_A,
        "--type",
        "openai",
        "--credential",
        &format!("OPENAI_API_KEY={SECRET_A}"),
    ])
    .await
    .expect("create provider A");

    let allow_path = allow_policy.path().to_str().unwrap();
    let deny_path = deny_policy.path().to_str().unwrap();
    let mut sandbox_a = create_agent(&["--policy", allow_path, "--provider", PROVIDER_A]).await;
    let mut sandbox_b = create_agent(&["--policy", deny_path]).await;
    let mut sandbox_c = create_agent(&["--policy", deny_path]).await;
    let names = [&sandbox_a.name, &sandbox_b.name, &sandbox_c.name];
    for name in names {
        wait_for_agent(name, Duration::from_secs(300))
            .await
            .expect("agent running through the shared pool");
    }
    let id_a = sandbox_id(&sandbox_a.name).await.unwrap();
    let id_b = sandbox_id(&sandbox_b.name).await.unwrap();
    let id_c = sandbox_id(&sandbox_c.name).await.unwrap();
    let owners = [
        kube.owner_of(&id_a).await.unwrap(),
        kube.owner_of(&id_b).await.unwrap(),
        kube.owner_of(&id_c).await.unwrap(),
    ];
    eprintln!(
        "METRIC owners a={} b={} c={}",
        owners[0], owners[1], owners[2]
    );
    let pool = kube.pool_pods().await.unwrap();
    assert_eq!(pool.len(), 2, "two shared replicas: {pool:?}");
    assert!(
        owners
            .iter()
            .all(|owner| pool.iter().any(|(pod, _)| pod == owner)),
        "every sandbox is hosted by a shared replica, none by a dedicated Pod"
    );

    assert_isolation("before-takeover", &sandbox_a.name, &sandbox_b.name, port).await;

    // The elected assignment coordinator pauses: another gateway takes over
    // and no healthy agent freezes.
    coordinator_pause_phase(&kube, &[&id_a, &id_b, &id_c]).await;

    // Planned movement: drain the busiest replica, then rebalance.
    drain_and_rebalance_phase(
        &kube,
        &[
            (&sandbox_a.name, &id_a),
            (&sandbox_b.name, &id_b),
            (&sandbox_c.name, &id_c),
        ],
        port,
    )
    .await;

    // Drain and rebalance moved sessions: re-read who owns what.
    let owners = [
        kube.owner_of(&id_a).await.unwrap(),
        kube.owner_of(&id_b).await.unwrap(),
        kube.owner_of(&id_c).await.unwrap(),
    ];
    eprintln!(
        "METRIC owners_after_rebalance a={} b={} c={}",
        owners[0], owners[1], owners[2]
    );

    // Kill the replica that owns sandbox A, as a crash would.
    let victim = owners[0].clone();
    let survivors_unaffected: Vec<&String> = names
        .iter()
        .zip(owners.iter())
        .filter(|(_, owner)| **owner != victim)
        .map(|(name, _)| *name)
        .collect();
    let before_a = agent_state(&sandbox_a.name).await.unwrap();
    let workload_uid_a = kube.workload_pod_uid(&id_a).await.unwrap();
    let mut before_others = Vec::new();
    for name in &survivors_unaffected {
        before_others.push(agent_state(name).await.unwrap());
    }
    let killed_at = Instant::now();
    kube.run(&[
        "delete",
        "pod",
        &victim,
        "--grace-period=0",
        "--force",
        "--wait=false",
    ])
    .await
    .expect("kill owning supervisor Pod");

    // Sandboxes owned by the surviving replica keep running throughout.
    for (name, before) in survivors_unaffected.iter().zip(before_others) {
        let state = agent_state(name)
            .await
            .expect("unaffected sandbox stays reachable");
        assert_eq!(state.pid, before.pid);
    }

    let (after_a, recovery) = wait_for_agent(&sandbox_a.name, RECOVERY_TIMEOUT)
        .await
        .expect("sandbox A resumes after fenced takeover");
    let recovered_after = killed_at.elapsed();
    let new_owner = kube.owner_of(&id_a).await.unwrap();
    // The counter advanced by less than wall time: the agent was frozen,
    // not restarted, while ownership moved.
    let progressed = after_a.counter.saturating_sub(before_a.counter);
    eprintln!(
        "METRIC takeover victim={victim} new_owner={new_owner} exec_restored_after_ms={} poll_ms={} counter_before={} counter_after={} counter_progress={} wall_s={}",
        recovered_after.as_millis(),
        recovery.as_millis(),
        before_a.counter,
        after_a.counter,
        progressed,
        recovered_after.as_secs()
    );
    assert_ne!(new_owner, victim, "a different replica took over");
    assert_eq!(after_a.pid, before_a.pid, "the same agent process resumed");
    assert!(
        after_a.counter >= before_a.counter,
        "in-memory counter continued instead of restarting"
    );
    assert_eq!(
        kube.workload_pod_uid(&id_a).await.unwrap(),
        workload_uid_a,
        "the workload Pod was not replaced"
    );
    tokio::time::sleep(Duration::from_secs(3)).await;
    let later = agent_state(&sandbox_a.name).await.unwrap();
    assert!(
        later.counter > after_a.counter,
        "resumed agent keeps running"
    );

    assert_isolation("after-takeover", &sandbox_a.name, &sandbox_b.name, port).await;

    // Partition / stale owner: freeze B's owner process without killing it.
    // Its gateway lease and its runtime lease lapse, another replica fences
    // it and resumes B, and when the old owner wakes it must stay fenced.
    let node = node_container(&kube.context);
    let stale_owner = kube.owner_of(&id_b).await.unwrap();
    let stale_pid = supervisor_host_pid(&node, &stale_owner).await.unwrap();
    let before_b = agent_state(&sandbox_b.name).await.unwrap();
    let paused_at = Instant::now();
    node_shell(&node, &format!("kill -STOP {stale_pid}"))
        .await
        .unwrap();
    let (after_b, _) = wait_for_agent(&sandbox_b.name, RECOVERY_TIMEOUT)
        .await
        .expect("sandbox B resumes while its old owner is partitioned");
    let partition_recovery = paused_at.elapsed();
    let b_owner = kube.owner_of(&id_b).await.unwrap();
    eprintln!(
        "METRIC partition stale_owner={stale_owner} new_owner={b_owner} exec_restored_after_ms={} counter_before={} counter_after={}",
        partition_recovery.as_millis(),
        before_b.counter,
        after_b.counter
    );
    assert_ne!(
        b_owner, stale_owner,
        "another replica fenced the stale owner"
    );
    assert_eq!(
        after_b.pid, before_b.pid,
        "the partitioned sandbox's agent survived"
    );
    node_shell(&node, &format!("kill -CONT {stale_pid}"))
        .await
        .unwrap();
    // Give the returning owner time to try to act with its stale authority.
    tokio::time::sleep(Duration::from_secs(12)).await;
    let stale_logs = kube.run(&["logs", &stale_owner]).await.unwrap_or_default();
    let fenced_line = stale_logs.lines().rev().find(|line| {
        line.contains(&id_b)
            && (line.contains("lost ownership")
                || line.contains("no longer assigned")
                || line.contains("self-fenc"))
    });
    eprintln!("METRIC partition stale_owner_reaction={fenced_line:?}");
    assert!(
        fenced_line.is_some() || stale_logs.contains("self-fencing"),
        "the returning stale owner must observe that it was fenced"
    );
    let resumed_b = agent_state(&sandbox_b.name).await.unwrap();
    assert_eq!(resumed_b.pid, before_b.pid);
    assert!(
        resumed_b.counter > after_b.counter,
        "B keeps running under its new owner"
    );
    assert_eq!(
        kube.owner_of(&id_b).await.unwrap(),
        b_owner,
        "the stale owner did not reclaim the sandbox"
    );
    assert_isolation("after-partition", &sandbox_a.name, &sandbox_b.name, port).await;

    // Delete wins over a pending recovery: kill C's owner and delete C at once.
    let owner_c = kube.owner_of(&id_c).await.unwrap();
    kube.run(&[
        "delete",
        "pod",
        &owner_c,
        "--grace-period=0",
        "--force",
        "--wait=false",
    ])
    .await
    .expect("kill sandbox C owner");
    sandbox_c.cleanup().await;
    let remaining = cli(&["sandbox", "list"]).await.unwrap_or_default();
    assert!(
        !remaining.contains(&sandbox_c.name),
        "deleted sandbox is gone: {remaining}"
    );
    let deployment_after = kube
        .json(&["get", "deployment", POOL_DEPLOYMENT])
        .await
        .expect("shared Deployment survives sandbox deletion");
    assert_eq!(deployment_after["metadata"]["uid"], deployment_uid.as_str());

    // Other sandboxes recover or keep running and remain isolated.
    let (a_final, _) = wait_for_agent(&sandbox_a.name, RECOVERY_TIMEOUT)
        .await
        .unwrap();
    let (b_final, _) = wait_for_agent(&sandbox_b.name, RECOVERY_TIMEOUT)
        .await
        .unwrap();
    assert_eq!(a_final.pid, before_a.pid);
    eprintln!("METRIC final a={a_final:?} b={b_final:?}");
    assert_isolation("after-delete", &sandbox_a.name, &sandbox_b.name, port).await;

    sandbox_a.cleanup().await;
    sandbox_b.cleanup().await;
    delete_provider(PROVIDER_A).await;
    assert!(
        kube.json(&["get", "deployment", POOL_DEPLOYMENT])
            .await
            .is_ok(),
        "shared Deployment survives deleting every sandbox"
    );
}
