#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
. "$SCRIPT_DIR/lib/common.sh"
ensure_runtime
if test -f "$RUNTIME_DIR/fake-gpu-incompatible"; then warn 'Functional validation SKIPPED: Fake GPU Operator images are architecture-incompatible.'; exit 0; fi
worker="$CLUSTER_NAME-worker"
capacity=$(kube get node "$worker" -o jsonpath='{.status.allocatable.nvidia\.com/gpu}')
test -n "$capacity" && test "$capacity" -ge 1 || die "$worker does not advertise nvidia.com/gpu"
if test "$DRY_RUN" = true; then run kube create namespace "$FUNCTIONAL_NAMESPACE"; else kube create namespace "$FUNCTIONAL_NAMESPACE" --dry-run=client -o yaml | kube apply -f -; fi
run kube delete pod fake-gpu-validation -n "$FUNCTIONAL_NAMESPACE" --ignore-not-found
run kube apply -f "$PROJECT_DIR/manifests/functional/gpu-validation-pod.yaml"
run kube wait -n "$FUNCTIONAL_NAMESPACE" --for=condition=PodScheduled pod/fake-gpu-validation --timeout="${WAIT_TIMEOUT_SECONDS}s"
scheduled=$(kube get pod -n "$FUNCTIONAL_NAMESPACE" fake-gpu-validation -o jsonpath='{.spec.nodeName}')
test "$scheduled" = "$worker" || die "Functional pod scheduled on $scheduled instead of $worker"
result="$RESULTS_DIR/functional-$(date -u +%Y%m%dT%H%M%SZ).txt"
{ printf 'simulation=fake-gpu-operator\nnode=%s\ngpu_allocatable=%s\n' "$scheduled" "$capacity"; kube logs -n "$FUNCTIONAL_NAMESPACE" fake-gpu-validation 2>&1 || true; } > "$result"
log 'Functional scheduling passed; this is simulated GPU allocation, not CUDA performance.'
