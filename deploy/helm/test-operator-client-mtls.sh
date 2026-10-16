#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Live test for skaffold run -p cert-manager-operator and a direct Service
# port-forward. Creates and removes a synthetic provider/profile when operator
# auth is enabled. Set OPENSHELL_OPERATOR_AUTH_ENABLED=false after disabling
# gateway operator auth to verify the negative case without changing fixtures.
set -euo pipefail
umask 077

namespace="${OPENSHELL_OPERATOR_NAMESPACE:-openshell}"
release="${OPENSHELL_OPERATOR_RELEASE:-openshell}"
address="${OPENSHELL_OPERATOR_ADDRESS:-localhost:8090}"
enabled="${OPENSHELL_OPERATOR_AUTH_ENABLED:-true}"
operator_secret="${OPENSHELL_OPERATOR_SECRET:-${release}-operator-client-tls}"
ordinary_secret="${OPENSHELL_ORDINARY_CLIENT_SECRET:-openshell-client-tls}"
server_secret="${OPENSHELL_SERVER_TLS_SECRET:-openshell-server-tls}"
work_dir="$(mktemp -d)"
fixture="operator-e2e-$(date +%s)-$$"
created_fixture=false

rpc() {
  local identity="$1" method="$2" body="$3"
  grpcurl -max-time 15 -cacert "${work_dir}/ca.crt" \
    -cert "${work_dir}/${identity}.crt" -key "${work_dir}/${identity}.key" \
    -d "${body}" "${address}" "openshell.v1.OpenShell/${method}"
}

cleanup() {
  if [[ "${created_fixture}" == true ]]; then
    rpc operator DeleteProvider "{\"workspaceScope\":{\"workspace\":\"default\"},\"name\":\"${fixture}\",\"allowMissing\":true}" >/dev/null || true
    rpc operator DeleteProviderProfile "{\"id\":\"${fixture}\",\"allowMissing\":true}" >/dev/null || true
  fi
  rm -rf "${work_dir}"
}
trap cleanup EXIT

kubectl -n "${namespace}" wait "certificate/${release}-operator-client" \
  --for=condition=Ready --timeout=120s
kubectl -n "${namespace}" get secret "${server_secret}" \
  -o jsonpath='{.data.ca\.crt}' | base64 -d >"${work_dir}/ca.crt"
for identity in operator ordinary; do
  secret="${operator_secret}"
  if [[ "${identity}" == ordinary ]]; then secret="${ordinary_secret}"; fi
  kubectl -n "${namespace}" get secret "${secret}" \
    -o jsonpath='{.data.tls\.crt}' | base64 -d >"${work_dir}/${identity}.crt"
  kubectl -n "${namespace}" get secret "${secret}" \
    -o jsonpath='{.data.tls\.key}' | base64 -d >"${work_dir}/${identity}.key"
done
openssl verify -CAfile "${work_dir}/ca.crt" -purpose sslclient "${work_dir}/operator.crt"
openssl x509 -in "${work_dir}/operator.crt" -noout -subject -nameopt RFC2253 \
  | grep -Eq '(^|[, ])OU=operator(,|$)'
if openssl x509 -in "${work_dir}/ordinary.crt" -noout -subject -nameopt RFC2253 \
  | grep -Eq '(^|[, ])OU=operator(,|$)'; then
  echo "ordinary client must not have the operator OU" >&2
  exit 1
fi

request="{\"workspaceScope\":{\"workspace\":\"default\"},\"name\":\"${fixture}\",\"credentialKeys\":[\"OPERATOR_TEST_KEY\"]}"
expect_denied() {
  local identity="$1"
  if rpc "${identity}" GetProviderCredentials "${request}" >"${work_dir}/denied" 2>&1; then
    echo "${identity} unexpectedly retrieved operator-only credentials" >&2
    exit 1
  fi
  if ! grep -Eq 'Code: (PermissionDenied|Unauthenticated)' "${work_dir}/denied"; then
    cat "${work_dir}/denied" >&2
    exit 1
  fi
}

expect_denied ordinary
if [[ "${enabled}" == false ]]; then
  expect_denied operator
  echo "operator and ordinary clients denied with operator auth disabled"
  exit 0
elif [[ "${enabled}" != true ]]; then
  echo "OPENSHELL_OPERATOR_AUTH_ENABLED must be true or false" >&2
  exit 1
fi

created_fixture=true
rpc operator ImportProviderProfiles "{\"profiles\":[{\"profile\":{\"id\":\"${fixture}\",\"credentials\":[{\"name\":\"test_key\",\"envVars\":[\"OPERATOR_TEST_KEY\"]}]}}]}" >"${work_dir}/import.json"
jq -e '.imported == true' "${work_dir}/import.json" >/dev/null
rpc operator CreateProvider "{\"workspaceScope\":{\"workspace\":\"default\"},\"provider\":{\"metadata\":{\"name\":\"${fixture}\"},\"type\":\"${fixture}\",\"credentials\":{\"OPERATOR_TEST_KEY\":\"synthetic-test-value\"}}}" >/dev/null
rpc operator GetProviderCredentials "${request}" >"${work_dir}/credentials.json"
jq -e '.credentials.OPERATOR_TEST_KEY.value == "synthetic-test-value"' "${work_dir}/credentials.json" >/dev/null
expect_denied ordinary
echo "cert-manager operator retrieved synthetic credentials; ordinary client denied"
