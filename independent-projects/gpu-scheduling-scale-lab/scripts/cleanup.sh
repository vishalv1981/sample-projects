#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
. "$SCRIPT_DIR/lib/common.sh"
validate_cluster_name "$CLUSTER_NAME"; validate_runtime_target "$RUNTIME_DIR"
if kind_cmd get clusters 2>/dev/null | grep -Fxq "$CLUSTER_NAME"; then
  if test -f "$KUBECONFIG_FILE"; then
    if test -f "$RUNTIME_DIR/latest-experiment"; then
      experiment_id=$(cat "$RUNTIME_DIR/latest-experiment")
      experiment_id_valid "$experiment_id" || die 'Refusing cleanup for invalid experiment ID'
      run kube delete pods -n "$EXPERIMENT_NAMESPACE" -l "gpu-simulation.example/experiment-id=$experiment_id" --ignore-not-found=true
    fi
    run kube delete nodes -l 'kwok.x-k8s.io/node=fake,gpu-simulation.example/project=gpu-scheduling-scale-lab' --ignore-not-found=true
    for ns in "$FUNCTIONAL_NAMESPACE" "$EXPERIMENT_NAMESPACE" "$KWOK_NAMESPACE" "$FAKE_GPU_NAMESPACE"; do case "$ns" in gpu-functional|gpu-scale-lab|kwok-system|fake-gpu-operator) run kube delete namespace "$ns" --ignore-not-found=true --wait=false;; *) die "Refusing unexpected namespace target: $ns";; esac; done
  fi
  run kind_cmd delete cluster --name "$CLUSTER_NAME"
fi
if test "$DRY_RUN" != true; then rm -f "$KUBECONFIG_FILE" "$RUNTIME_DIR/latest-experiment" "$RUNTIME_DIR/fake-gpu-incompatible"; fi
log 'Scoped cleanup complete; downloaded tools and diagnostic results were preserved under .runtime.'
