// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared supervisor Deployment for the gateway supervisor topology.
//!
//! One Deployment serves every sandbox of a gateway installation. It is owned
//! by the driver, not by any Sandbox resource, so deleting a sandbox can never
//! delete it. Its Pods carry the supervisor boundary role, so the existing
//! namespace-wide workload fence admits them to sandbox control ports, and
//! they derive the same unprivileged security context as dedicated supervisor
//! Pods.

use std::collections::BTreeMap;

use k8s_openapi::api::apps::v1::{
    Deployment, DeploymentSpec, DeploymentStrategy, RollingUpdateDeployment,
};
use k8s_openapi::api::core::v1::{
    Affinity, EnvVar, EnvVarSource, ObjectFieldSelector, Pod, PodAffinityTerm, PodAntiAffinity,
    PodTemplateSpec, WeightedPodAffinityTerm,
};
use k8s_openapi::apimachinery::pkg::apis::meta::v1::{LabelSelector, OwnerReference};
use k8s_openapi::apimachinery::pkg::util::intstr::IntOrString;
use kube::core::ObjectMeta;

use crate::isolation::BOUNDARY_ROLE_LABEL;
use crate::sandbox_runtime::{
    CONTROL_HEALTH_PORT, SandboxRuntimeNames, SupervisorClientTls, supervisor_pod,
};

/// Name of the shared supervisor Deployment in the sandbox namespace.
pub const SUPERVISOR_POOL_DEPLOYMENT: &str = "openshell-supervisor-pool";
/// Component label value that identifies shared supervisor Pods.
pub const SUPERVISOR_POOL_COMPONENT: &str = "supervisor-pool";
/// Internal prototype default for the number of shared replicas.
pub const SUPERVISOR_POOL_REPLICAS: i32 = 2;
/// Enforced owner lease the runtime applies in gateway topology. Must match
/// the gateway's assignment timing.
pub const SUPERVISOR_POOL_LEASE_TTL_MS: u64 = 10_000;
/// Per-generation supervisor bootstrap key carrying the main-process spec,
/// which dedicated supervisors receive as Pod environment.
pub const MAIN_PROCESS_SPEC_KEY: &str = "main-process-spec.json";
/// Resource claim that binds a runtime to the shared pool rather than to a
/// dedicated supervisor Pod.
pub const SUPERVISOR_POOL_UID_CLAIM: &str = "kubernetes.supervisor_pool_uid";
/// Session state root inside each shared supervisor Pod.
pub const SUPERVISOR_POOL_SESSION_ROOT: &str = "/run/openshell/sessions";

const COMPONENT_LABEL: &str = "openshell.ai/component";
const MANAGED_BY_LABEL: &str = "openshell.ai/managed-by";
const GATEWAY_ID_LABEL: &str = "openshell.ai/gateway-id";

/// Labels shared by the Deployment selector and its Pods.
#[must_use]
pub fn supervisor_pool_labels(gateway_id: &str) -> BTreeMap<String, String> {
    BTreeMap::from([
        (MANAGED_BY_LABEL.to_string(), "openshell".to_string()),
        (
            COMPONENT_LABEL.to_string(),
            SUPERVISOR_POOL_COMPONENT.to_string(),
        ),
        (GATEWAY_ID_LABEL.to_string(), gateway_id.to_string()),
    ])
}

/// Whether `pod` carries the shared supervisor identity of this gateway.
#[must_use]
pub fn is_supervisor_pool_pod(pod: &Pod, gateway_id: &str) -> bool {
    let Some(labels) = pod.metadata.labels.as_ref() else {
        return false;
    };
    labels.get(COMPONENT_LABEL).map(String::as_str) == Some(SUPERVISOR_POOL_COMPONENT)
        && labels.get(GATEWAY_ID_LABEL).map(String::as_str) == Some(gateway_id)
        && labels.get(BOUNDARY_ROLE_LABEL).map(String::as_str) == Some("supervisor")
}

/// Inputs for the shared supervisor Deployment.
pub struct SupervisorPoolSpec<'a> {
    pub namespace: &'a str,
    pub gateway_id: &'a str,
    pub supervisor_image: &'a str,
    pub supervisor_pull_policy: Option<crate::KubernetesImagePullPolicy>,
    pub service_account_name: &'a str,
    pub control_uid: u32,
    pub control_gid: u32,
    pub image_pull_secrets: &'a [String],
    pub grpc_endpoint: &'a str,
    pub client_tls: SupervisorClientTls<'a>,
    pub log_level: &'a str,
    pub sa_token_ttl_secs: i64,
    pub https_proxy: Option<&'a str>,
    pub no_proxy: Option<&'a str>,
    pub replicas: i32,
    pub tracing_environment: &'a [(&'a str, String)],
}

/// Render the shared supervisor Deployment.
///
/// The Pod template starts from the dedicated supervisor Pod so the security
/// context, gateway TLS trust, and projected token stay identical, then drops
/// every sandbox-specific input. Sessions receive their launch material from
/// the gateway after it authenticated the replica.
pub fn supervisor_pool_deployment(spec: &SupervisorPoolSpec<'_>) -> Result<Deployment, String> {
    let names = SandboxRuntimeNames::new("supervisor-pool");
    let template_pod = supervisor_pod(
        spec.namespace,
        &names,
        "supervisor-pool",
        "supervisor-pool",
        spec.gateway_id,
        spec.supervisor_image,
        spec.supervisor_pull_policy,
        spec.service_account_name,
        spec.control_uid,
        spec.control_gid,
        spec.image_pull_secrets,
        spec.grpc_endpoint,
        spec.client_tls,
        "",
        spec.log_level,
        spec.sa_token_ttl_secs,
        spec.https_proxy,
        spec.no_proxy,
        None,
        false,
        false,
        false,
        None,
        OwnerReference::default(),
        spec.tracing_environment,
    )?;
    let mut pod_spec = template_pod
        .spec
        .ok_or_else(|| "supervisor Pod template has no spec".to_string())?;
    pod_spec.scheduling_gates = None;
    pod_spec.restart_policy = Some("Always".to_string());
    pod_spec.volumes = pod_spec.volumes.map(|volumes| {
        volumes
            .into_iter()
            .filter(|volume| volume.name != "bootstrap")
            .collect()
    });
    let labels = {
        let mut labels = supervisor_pool_labels(spec.gateway_id);
        labels.insert(BOUNDARY_ROLE_LABEL.to_string(), "supervisor".to_string());
        labels
    };
    pod_spec.affinity = Some(Affinity {
        pod_anti_affinity: Some(PodAntiAffinity {
            preferred_during_scheduling_ignored_during_execution: Some(vec![
                WeightedPodAffinityTerm {
                    weight: 100,
                    pod_affinity_term: PodAffinityTerm {
                        label_selector: Some(LabelSelector {
                            match_labels: Some(supervisor_pool_labels(spec.gateway_id)),
                            ..Default::default()
                        }),
                        topology_key: "kubernetes.io/hostname".to_string(),
                        ..Default::default()
                    },
                },
            ]),
            ..Default::default()
        }),
        ..Default::default()
    });
    let container = pod_spec
        .containers
        .first_mut()
        .ok_or_else(|| "supervisor Pod template has no container".to_string())?;
    let sandbox_specific = [
        "OPENSHELL_SANDBOX_ID",
        "OPENSHELL_SANDBOX",
        "OPENSHELL_MAIN_PROCESS_SPEC",
        "OPENSHELL_SSH_SOCKET_PATH",
        "OPENSHELL_PROXY_TLS_DIR",
        openshell_core::sandbox_env::PROXY_CA_CERT,
        openshell_core::sandbox_env::PROXY_CA_KEY,
        "TRACEPARENT",
        "TRACESTATE",
    ];
    let mut environment: Vec<EnvVar> = container
        .env
        .take()
        .unwrap_or_default()
        .into_iter()
        .filter(|variable| !sandbox_specific.contains(&variable.name.as_str()))
        .collect();
    environment.push(EnvVar {
        name: "OPENSHELL_SUPERVISOR_POOL_MEMBER".to_string(),
        value_from: Some(EnvVarSource {
            field_ref: Some(ObjectFieldSelector {
                field_path: "metadata.name".to_string(),
                ..Default::default()
            }),
            ..Default::default()
        }),
        ..Default::default()
    });
    container.env = Some(environment);
    container.volume_mounts = container.volume_mounts.take().map(|mounts| {
        mounts
            .into_iter()
            .filter(|mount| mount.name != "bootstrap")
            .collect()
    });
    let mut command = vec![
        "/openshell-supervisor".to_string(),
        "pool".to_string(),
        "--session-root".to_string(),
        SUPERVISOR_POOL_SESSION_ROOT.to_string(),
        "--health-port".to_string(),
        CONTROL_HEALTH_PORT.to_string(),
    ];
    if let Some(url) = spec.https_proxy {
        command.extend(["--upstream-proxy".to_string(), url.to_string()]);
    }
    if let Some(hosts) = spec.no_proxy {
        command.extend(["--upstream-no-proxy".to_string(), hosts.to_string()]);
    }
    container.command = Some(command);

    Ok(Deployment {
        metadata: ObjectMeta {
            name: Some(SUPERVISOR_POOL_DEPLOYMENT.to_string()),
            namespace: Some(spec.namespace.to_string()),
            labels: Some(supervisor_pool_labels(spec.gateway_id)),
            ..Default::default()
        },
        spec: Some(DeploymentSpec {
            replicas: Some(spec.replicas),
            selector: LabelSelector {
                match_labels: Some(supervisor_pool_labels(spec.gateway_id)),
                ..Default::default()
            },
            // Replace one replica at a time and never drop below the desired
            // count: a rollout must leave capacity for fenced takeover.
            strategy: Some(DeploymentStrategy {
                type_: Some("RollingUpdate".to_string()),
                rolling_update: Some(RollingUpdateDeployment {
                    max_surge: Some(IntOrString::Int(1)),
                    max_unavailable: Some(IntOrString::Int(0)),
                }),
            }),
            template: PodTemplateSpec {
                metadata: Some(ObjectMeta {
                    labels: Some(labels),
                    ..Default::default()
                }),
                spec: Some(pod_spec),
            },
            ..Default::default()
        }),
        ..Default::default()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec() -> SupervisorPoolSpec<'static> {
        SupervisorPoolSpec {
            namespace: "openshell",
            gateway_id: "gw",
            supervisor_image: "supervisor:test",
            supervisor_pull_policy: None,
            service_account_name: "openshell-sandbox",
            control_uid: 1000,
            control_gid: 1000,
            image_pull_secrets: &[],
            grpc_endpoint: "https://openshell.openshell.svc:8080",
            client_tls: SupervisorClientTls::Secret("openshell-client-tls"),
            log_level: "info",
            sa_token_ttl_secs: 3600,
            https_proxy: None,
            no_proxy: None,
            replicas: SUPERVISOR_POOL_REPLICAS,
            tracing_environment: &[],
        }
    }

    #[test]
    fn pool_is_unowned_shared_and_carries_no_sandbox_material() {
        let deployment = supervisor_pool_deployment(&spec()).unwrap();
        assert!(
            deployment.metadata.owner_references.is_none(),
            "no Sandbox resource may own the shared pool"
        );
        let spec = deployment.spec.unwrap();
        assert_eq!(spec.replicas, Some(2));
        let template = spec.template;
        let labels = template.metadata.unwrap().labels.unwrap();
        assert_eq!(
            labels.get(BOUNDARY_ROLE_LABEL).map(String::as_str),
            Some("supervisor")
        );
        assert_eq!(
            labels.get(COMPONENT_LABEL).map(String::as_str),
            Some(SUPERVISOR_POOL_COMPONENT)
        );
        assert!(!labels.contains_key("openshell.ai/sandbox-id"));
        let pod = template.spec.unwrap();
        assert!(pod.scheduling_gates.is_none());
        assert_eq!(pod.automount_service_account_token, Some(false));
        assert!(
            pod.volumes
                .as_ref()
                .unwrap()
                .iter()
                .all(|volume| volume.name != "bootstrap"),
            "no per-sandbox bootstrap Secret is mounted"
        );
        let container = &pod.containers[0];
        let env = container.env.as_ref().unwrap();
        for forbidden in ["OPENSHELL_SANDBOX_ID", "OPENSHELL_MAIN_PROCESS_SPEC"] {
            assert!(env.iter().all(|variable| variable.name != forbidden));
        }
        assert!(
            env.iter()
                .any(|variable| variable.name == "OPENSHELL_TLS_CA")
        );
        let security = container.security_context.as_ref().unwrap();
        assert_eq!(security.allow_privilege_escalation, Some(false));
        assert_eq!(security.run_as_non_root, Some(true));
        assert_eq!(
            security.capabilities.as_ref().unwrap().drop.as_deref(),
            Some(&["ALL".to_string()][..])
        );
        assert_eq!(
            container.command.as_ref().unwrap()[..2],
            ["/openshell-supervisor".to_string(), "pool".to_string()]
        );
    }
}
