#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
. "$SCRIPT_DIR/lib/common.sh"
ensure_runtime
if test -f "$RUNTIME_DIR/fake-gpu-incompatible"; then warn 'Skipping Fake GPU Operator: required image architecture is incompatible. KWOK remains available.'; exit 0; fi
run helm_cmd upgrade --install fake-gpu-operator "$FAKE_GPU_CHART" --version "$FAKE_GPU_OPERATOR_VERSION" --namespace "$FAKE_GPU_NAMESPACE" --create-namespace --values "$PROJECT_DIR/config/fake-gpu-values.yaml" --wait --timeout "${WAIT_TIMEOUT_SECONDS}s"
run kube -n "$FAKE_GPU_NAMESPACE" wait --for=condition=Available deployment --all --timeout="${WAIT_TIMEOUT_SECONDS}s"
run kube -n "$FAKE_GPU_NAMESPACE" rollout status daemonset --all --timeout="${WAIT_TIMEOUT_SECONDS}s"
