#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
. "$SCRIPT_DIR/lib/common.sh"
ensure_runtime; validate_cluster_name "$CLUSTER_NAME"
if kind_cmd get clusters 2>/dev/null | grep -Fxq "$CLUSTER_NAME"; then log "Cluster $CLUSTER_NAME already exists"; exit 0; fi
args=(create cluster --name "$CLUSTER_NAME" --config "$PROJECT_DIR/config/kind-cluster.yaml" --image "$KIND_NODE_IMAGE" --kubeconfig "$KUBECONFIG_FILE" --wait "${WAIT_TIMEOUT_SECONDS}s")
if test "$PLATFORM_MODE" = emulated-amd64; then warn 'Creating an emulated AMD64 kind cluster'; DOCKER_DEFAULT_PLATFORM=linux/amd64 run "$(tool kind)" "${args[@]}"; else run "$(tool kind)" "${args[@]}"; fi
test "$DRY_RUN" = true || chmod 0600 "$KUBECONFIG_FILE"
worker="$CLUSTER_NAME-worker"; run kube label node "$worker" "$REAL_WORKER_LABEL" --overwrite
run kube wait --for=condition=Ready "node/$worker" --timeout="${WAIT_TIMEOUT_SECONDS}s"
