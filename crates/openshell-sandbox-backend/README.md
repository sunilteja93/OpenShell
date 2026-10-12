# OpenShell sandbox backend

This crate implements the supervisor-side isolation backend and the private
control protocol shared with `openshell-sandbox`.

## Exec recovery

An exec envelope carries a UUID request ID and an absolute expiration time in
Unix milliseconds, set to 30 seconds after creation. The payload digest includes
the expiration time, and transport recovery preserves the entire envelope.
The supervisor and sandbox need synchronized wall clocks.

The sandbox checks expiration before starting or reattaching an exec. It keeps
each admitted request ID until that deadline, independently of process and I/O
retention. An expired request, including a delayed first attempt, is rejected
even after its ID has been discarded. The sandbox never moves its observed
admission clock backwards, so a clock adjustment cannot resurrect discarded
requests. Expired IDs are reclaimed when the next exec request arrives.

There is no lifetime exec request count limit. The existing concurrent process
retention limit still applies. The deadline governs admission and recovery;
it does not terminate an already running command. A recovery timeout can leave
the execution outcome unknown.

Exec envelopes without an expiration time are rejected. Update the supervisor
and sandbox runtime together when deploying this protocol change.

## Supervisor recovery extension (prototype)

`supervisor-recovery.v1` is an opt-in, versioned extension to the Sandbox
Protocol used by the gateway supervisor topology. A runtime enables it only
when its `BoundaryConfig.supervisor_recovery` is set; otherwise it keeps the
RFC 0012 rule that only the same supervisor process can reconnect.

- The launch owner holds ownership epoch 1. A replacement asks the frozen
  runtime for a single-use challenge (`RecoveryHello`), obtains a
  gateway-signed recovery grant bound to the sandbox, generation, next epoch,
  its supervisor instance, its ephemeral Ed25519 key, and the challenge, and
  presents it with a key proof (`Recover`).
- The runtime accepts only a strictly higher epoch, re-pins the supervisor
  instance, closes every previous connection, keeps the workload frozen, and
  returns a fencing receipt signed with its TLS key plus the attached policy,
  main process, provider publication generation, and checkpointed TOFU pins.
- The workload resumes only on `ConfirmRecovery` with a gateway activation
  bound to that receipt's digest. Recovery never calls `StartAgent`.
- The active owner must forward gateway-signed leases (`RenewLease`). A lapsed
  lease freezes the workload and starts the bounded recovery window, which a
  later candidate never extends. Terminate wins over recovery.
- The owner checkpoints new TOFU pins (`StoreRecoveryState`); a new pin does
  not authorize traffic until the runtime acknowledged it.
