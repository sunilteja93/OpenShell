# OpenShell supervisor

The supervisor loads and reconciles policy, maintains provider credentials, applies network and MCP inspection, and drives the admitted isolation backend through attachment, confirmation, and workload start.

## Backend startup

The public `run_sandbox` entry point selects the OpenShell Sandbox Protocol backend and collects its startup inputs into a private `SandboxRunConfig`. Shared startup receives that config and the trusted backend setup separately. The `backend_setup` module owns that backend's launch-data decoder, workload policy discovery, and client construction. Descriptor contents cannot select an implementation.

Shared startup checks the admitted backend name before passing the opaque payload to its decoder. It then compares the decoded sandbox, session, and runtime generation with the trusted launch inputs before installing credentials or discovering workload policy. A mismatch stops startup.

The built-in decoder also carries the VM driver's fixed workload identity into shared policy validation. Startup and later policy updates must reject selectors that conflict with that identity. Other launch descriptors do not enable this VM-specific check.

The supervisor admits policy and prepares credentials before constructing and attaching the selected client. It uses the isolation contract's `BoundBoundary` and `ConfirmedBoundary` directly: confirm the attached boundary, prepare network mediation, then start the workload. Backend implementations remain responsible for validating their native enforcement evidence through the isolation contract.

The client receives the supervisor's live provider state, bearer-token slot, and CA-path slot. Provider refresh, token rotation, and later CA publication must remain visible through those shared handles. Startup does not create independent copies of their current values.

The setup interface stays private to the supervisor. It adds no runtime backend registration, endpoint configuration, or public factory API. The public `run_sandbox` signature and standard backend selection remain unchanged.

## Console logging

Diagnostics and OCSF shorthand share a bounded, nonblocking stderr writer. With `ocsf_json_enabled=true`, the same writer also receives timestamped `OCSF-JSON` records. Each formatter submits a complete line in one write so concurrent producers cannot interleave records in the queue. The 1,024-line queue drops new lines when full rather than waiting for stderr.

Console JSON is installed independently of optional file appenders and uses an INFO filter independent of the diagnostic filter. Its runtime enabled flag and target schema version are shared with the existing JSONL file layer. The supervisor retains the writer guards until shutdown. The gateway log push layer continues to emit shorthand only.

## Shared pool mode (prototype)

`openshell-supervisor pool` runs one replica of the gateway supervisor
topology. It authenticates pool RPCs with its projected token, follows
gateway assignments, and hosts each assigned sandbox through
`run_hosted_session` on its own Tokio runtime. The runtime's thread-start hook
binds the session identity, OCSF context, and log-push sink, so attribution,
tasks, and cleanup stay per sandbox. Gateway clients use a session-scoped
endpoint whose credentials live only in that session; dropping the session
revokes them. A `RELEASE` assignment (stop or delete) terminates the session.
A `RELINQUISH` assignment (planned drain or rebalance handoff) and losing the
assignment both abandon the session without touching the workload so the next
owner can resume it; a relinquished session keeps being reported in sync until
its thread has stopped, which is how the gateway learns the release finished.
Process signals stop the replica, never a hosted workload.

Pool RPCs go through `PoolChannel`: each attempt has a 1.5-second deadline.
Idempotent calls (sync, get assignment, lease renewal) reconnect and retry
within a 4-second budget, with the reconnect itself bounded by that budget, so
a hung gateway cannot starve an owner's lease renewal; a new connection is
routed by the gateway Service to any healthy gateway. Recovery grants and
activations are single bounded attempts because they commit single-use state:
on failure the coordinator reserves a new epoch within the same recovery
deadline. A recovering candidate retries `RecoveryHello` for up to 4 seconds
before it requests a grant.
