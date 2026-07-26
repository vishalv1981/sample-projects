#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
. "$SCRIPT_DIR/lib/common.sh"
validate_cluster_name "$CLUSTER_NAME"
kind_cmd get clusters | grep -Fxq "$CLUSTER_NAME" || { log "Cluster $CLUSTER_NAME is absent"; exit 0; }
kube get nodes -L run.ai/simulated-gpu-node-pool,kwok.x-k8s.io/node,gpu-simulation.example/type,gpu-simulation.example/gpu-model
kube get pods -A -l "$PROJECT_LABEL_KEY=$PROJECT_LABEL_VALUE" -o wide
if test -f "$RUNTIME_DIR/latest-experiment"; then id=$(cat "$RUNTIME_DIR/latest-experiment"); experiment_id_valid "$id" && test -f "$RESULTS_DIR/$id/summary.json" && cat "$RESULTS_DIR/$id/summary.json"; fi
