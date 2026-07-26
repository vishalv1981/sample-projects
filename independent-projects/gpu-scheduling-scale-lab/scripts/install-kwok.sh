#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
. "$SCRIPT_DIR/lib/common.sh"
ensure_runtime
chart="$RUNTIME_DIR/kwok-chart-$KWOK_CHART_VERSION.tgz"
if ! test -f "$chart"; then run curl --fail --location --retry 3 --output "$chart" "https://github.com/kubernetes-sigs/kwok/releases/download/$KWOK_VERSION/kwok-chart-$KWOK_CHART_VERSION.tgz" || die 'KWOK chart download failed'; fi
test "$DRY_RUN" = true || verify_sha256 "$chart" "$KWOK_CHART_SHA256"
run helm_cmd upgrade --install kwok "$chart" --namespace "$KWOK_NAMESPACE" --create-namespace --values "$PROJECT_DIR/config/kwok-values.yaml" --wait --timeout "${WAIT_TIMEOUT_SECONDS}s"
stage_chart="$RUNTIME_DIR/kwok-stage-fast-chart-$KWOK_STAGE_CHART_VERSION.tgz"
if ! test -f "$stage_chart"; then run curl --fail --location --retry 3 --output "$stage_chart" "https://github.com/kubernetes-sigs/kwok/releases/download/$KWOK_VERSION/kwok-stage-fast-chart-$KWOK_STAGE_CHART_VERSION.tgz" || die 'KWOK stage chart download failed'; fi
test "$DRY_RUN" = true || verify_sha256 "$stage_chart" "$KWOK_STAGE_CHART_SHA256"
run helm_cmd upgrade --install kwok-stage-fast "$stage_chart" --namespace "$KWOK_NAMESPACE" --wait --timeout "${WAIT_TIMEOUT_SECONDS}s"
# The official configuration defaults to manageAllNodes=false and this exact annotation selector.
run kube -n "$KWOK_NAMESPACE" rollout status deployment/kwok-controller --timeout="${WAIT_TIMEOUT_SECONDS}s"
