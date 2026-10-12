<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# OpenShell gateway server

The server owns control-plane RPCs, authentication, provider credential delivery,
and sandbox lifecycle orchestration. The `openshell-gateway` crate composes the
server with selected compute drivers.

## Operator credential export

`GetProviderCredentials` exports selected profile-declared runtime environment
keys from one provider in an explicitly named workspace. It requires the distinct
`Principal::Operator`, established only from a verified TLS leaf certificate with
`OU=operator` when `mtls_auth.operator_enabled` is enabled. The TLS listener supplies
its certificate fingerprint separately from user identity; the ordinary
identity-only serving path cannot establish an operator.

Operators explicitly inherit user/platform-admin permissions, but not sandbox or
peer identities. Descriptor authorization and a handler guard both protect the
export RPC. Operator mutation admission uses a separate caller namespace from
ordinary mTLS users with the same certificate subject. Credential responses never
enter durable mutation replay or generic interceptor routing.

The export resolver validates the entire selection before minting, resolves only
selected credential handles, and fences provider revisions, refresh epochs, and
profile declarations before delivery. Driver expiration and provider expiration
combine using the earlier timestamp. OAuth minting inputs, broker-only keys, and
supervisor-bound dynamic grants are not exportable. A failed response can still
have committed refresh effects. Missing refresh-owned stored outputs trigger
refresh-on-read; missing static outputs and other credential-backend errors fail
closed without falling back to inline values.

## Refresh coordination

Explicit rotation, background refresh, and operator refresh-on-read share the
same refresh coordination:

- A process-local weak keyed mutex coalesces waiters without retaining deleted
  providers indefinitely.
- A bounded admission queue and executor pool limit detached refresh work.
- PostgreSQL deployments also take a dedicated per-provider/per-credential
  advisory lock; SQLite uses the local mutex. Worker secret cleanup takes the
  same locks and reloads state before changing its resource version.
- A durable `refresh_in_progress` marker precedes upstream minting. A
  `refresh_committing` marker protects the interval between grant persistence
  and provider-handle persistence. Only a completed provider commit publishes
  `refreshed` state, with a unique completed-mint identity in metadata
  annotations. Waiters coalesce against that identity, not bookkeeping changes
  to the resource version.
- Automatic callers recheck live recovery and retry deadlines under both locks.
  A failed mint blocks queued automatic exchanges without disabling scheduled
  worker retries or a separately requested manual rotation.
- Detached refresh tasks complete despite request cancellation, preserving
  replacement refresh tokens returned by the issuer.
- Existing generation/CAS fencing remains active for deletion and
  reconfiguration. An ambiguous rotating-token exchange is parked for recovery
  rather than automatically consuming the grant again.

A crash or lost database-lock session can leave a parked marker. Reconfigure or
reauthorize the refresh grant after investigating its issuer state. Changing a
marker directly is not a supported recovery workflow.

## Supervisor pool coordinator (prototype)

With `supervisor_topology = "gateway"`, `supervisor_pool::coordinator` elects
one gateway process as the assignment coordinator through a lease row in
`coordinator_leases` (migration 009). The holder is `<replica>/<uuid>`, unique
per process; the lease lasts 5 seconds and is renewed every second. The row is
never deleted, so terms only increase, and shutdown releases it in place.
Followers run no reconciliation; every gateway keeps serving pool sync,
assignment material, grants, activations, and owner lease renewals with
per-assignment CAS.

Every coordinator decision is one `Store::write_fenced` transaction: the
assignment CAS, the durable operation record, and, for a reservation, the
gateway-token rotation that revokes the previous owner. The transaction locks
the lease row (`FOR SHARE` on PostgreSQL, `BEGIN IMMEDIATE` on SQLite), then
samples the database clock and requires the same holder, the same term, and an
unexpired lease, judged at `max(caller clock, database clock)`. A paused
coordinator that resumes after its lease expired or after a takeover therefore
writes nothing. Gateway and database clocks are assumed to be synchronized well
within the lease TTL.

`supervisor_pool::operations` records assign, recover, drain, and rebalance
operations with a phase and a message. Assign and recover IDs are
deterministic, so a new coordinator adopts them; a recovery's deadline comes
from the assignment and is never extended. Drain and rebalance requests are
recorded by any gateway through the operator RPCs in `supervisor_pool::operator`
(bearer, `platform_admin`), but they are inert until the coordinator admits
them, one at a time. A drain that would leave no eligible replica is refused at
admission. An admitted drain cordons its member from placement, recovery, and
handoff targets until it is cancelled. Planned moves record a release stage on
the assignment: the owner receives `RELINQUISH`, and once its sync no longer
reports the sandbox (or it is gone, or 10 seconds passed) the coordinator
reserves the next epoch for the target and the normal fenced recovery protocol
completes the move. Rebalance moves from the most- to the least-loaded eligible
member while their loads differ by more than one, counting in-flight
reservations, and moves each sandbox at most once per operation.

## Focused verification

```shell
cargo test -p openshell-server --features test-support,prebuilt-z3
cargo clippy -p openshell-server --all-targets --features test-support,prebuilt-z3 -- -D warnings
```

The ignored `refresh_locks_coordinate_independent_replica_sessions` test requires
`OPENSHELL_REFRESH_TEST_DATABASE_URL` pointing to a disposable PostgreSQL database.
It exercises independent sessions without the local mutex. Do not point this
fixture at an existing deployment database.
