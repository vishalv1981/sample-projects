#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
. "$SCRIPT_DIR/lib/common.sh"
ensure_runtime
profile=${1:-smoke}; profile_file="$PROJECT_DIR/experiments/profiles/$profile.env"; test -f "$profile_file" || die "Unknown profile: $profile"
override_nodes=${KWOK_NODES:-}; . "$profile_file"; test -z "$override_nodes" || KWOK_NODES=$override_nodes; require_positive_integer KWOK_NODES "$KWOK_NODES"; test "$KWOK_NODES" -le "$KWOK_NODE_COUNT_LIMIT" || die 'KWOK node limit exceeded'
case "$profile" in medium|large) test "$CONFIRM_LARGE" = true || die "$profile requires --confirm-large";; esac
log "Impact estimate: create $KWOK_NODES API-only nodes; they run no containers or kubelets."
if test "$DRY_RUN" = true; then run kube create namespace "$EXPERIMENT_NAMESPACE"; else kube create namespace "$EXPERIMENT_NAMESPACE" --dry-run=client -o yaml | kube apply -f -; fi

render_node() {
  index=$1; case $((index%3)) in 1) model=T4; gpus=4; pool=t4;; 2) model=A100; gpus=8; pool=a100;; 0) model=H100; gpus=8; pool=h100;; esac
  cat <<EOF
apiVersion: v1
kind: Node
metadata:
  name: kwok-gpu-$index
  annotations:
    kwok.x-k8s.io/node: fake
  labels:
    kwok.x-k8s.io/node: fake
    gpu-simulation.example/type: kwok
    gpu-simulation.example/project: gpu-scheduling-scale-lab
    gpu-simulation.example/gpu-model: $model
    gpu-simulation.example/node-pool: $pool
    kubernetes.io/arch: amd64
spec:
  taints:
    - key: kwok.x-k8s.io/node
      value: fake
      effect: NoSchedule
---
EOF
}

while read -r first last; do
  manifest="$RUNTIME_DIR/nodes-$first-$last.yaml"; : > "$manifest"; i=$first; while test "$i" -le "$last"; do render_node "$i" >> "$manifest"; i=$((i+1)); done
  run kube apply -f "$manifest"; rm -f "$manifest"
  if test "$DRY_RUN" != true; then
    i=$first; while test "$i" -le "$last"; do case $((i%3)) in 1) gpus=4;; *) gpus=8;; esac; kube patch node "kwok-gpu-$i" --subresource=status --type=merge -p "{\"status\":{\"capacity\":{\"cpu\":\"32\",\"memory\":\"128Gi\",\"pods\":\"110\",\"nvidia.com/gpu\":\"$gpus\"},\"allocatable\":{\"cpu\":\"32\",\"memory\":\"128Gi\",\"pods\":\"110\",\"nvidia.com/gpu\":\"$gpus\"}}}" >/dev/null; i=$((i+1)); done
  fi
done <<EOF
$(batch_ranges "$KWOK_NODES" "$BATCH_SIZE")
EOF
test "$DRY_RUN" = true || wait_for 'KWOK nodes to become Ready' "$WAIT_TIMEOUT_SECONDS" sh -c "test \"\$(KUBECONFIG='$KUBECONFIG_FILE' '$(tool kubectl)' get nodes -l kwok.x-k8s.io/node=fake --no-headers 2>/dev/null | awk '\$2==\"Ready\"{n++} END{print n+0}')\" -eq '$KWOK_NODES'"
log 'KWOK nodes are API simulations; no containers or GPU code execute on them.'
