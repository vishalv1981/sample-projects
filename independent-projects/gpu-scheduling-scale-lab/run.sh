#!/usr/bin/env bash
set -Eeuo pipefail
PROJECT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck disable=SC1091
. "$PROJECT_DIR/scripts/lib/common.sh"

usage() {
  cat <<'USAGE'
Usage: ./run.sh COMMAND [options]
Commands: help, preflight, bootstrap, up, functional, scale, status, down, all
Options:
  --profile NAME                 smoke|small|medium|large (default: smoke)
  --dry-run                      print mutating commands
  --platform-mode MODE           auto|native|emulated-amd64
  --confirm-large                acknowledge medium/large impact
  --batch-size N                 API creation batch size
  --timeout N                    wait timeout in seconds
  --install-docker               Linux only; print guarded installation guidance
Environment overrides: CLUSTER_NAME, BATCH_SIZE, WAIT_TIMEOUT_SECONDS,
POLL_INTERVAL_SECONDS, PRESERVE_ON_FAILURE and profile variables documented in README.
USAGE
}

command_name=${1:-help}; test "$#" -eq 0 || shift
profile=$DEFAULT_PROFILE
while test "$#" -gt 0; do
  case "$1" in
    --profile) test "$#" -ge 2 || die '--profile requires a value'; profile=$2; shift 2;;
    --dry-run) DRY_RUN=true; shift;;
    --platform-mode) test "$#" -ge 2 || die '--platform-mode requires a value'; PLATFORM_MODE=$2; shift 2;;
    --confirm-large) CONFIRM_LARGE=true; shift;;
    --batch-size) test "$#" -ge 2 || die '--batch-size requires a value'; BATCH_SIZE=$2; shift 2;;
    --timeout) test "$#" -ge 2 || die '--timeout requires a value'; WAIT_TIMEOUT_SECONDS=$2; shift 2;;
    --install-docker) INSTALL_DOCKER=true; shift;;
    -h|--help) usage; exit 0;;
    *) die "Unknown option: $1";;
  esac
done
export DRY_RUN PLATFORM_MODE CONFIRM_LARGE BATCH_SIZE WAIT_TIMEOUT_SECONDS INSTALL_DOCKER
validate_platform_mode "$PLATFORM_MODE"; require_positive_integer batch-size "$BATCH_SIZE"; require_positive_integer timeout "$WAIT_TIMEOUT_SECONDS"

case "$command_name" in
  help) usage;;
  preflight|bootstrap|up|status|down) "$PROJECT_DIR/scripts/${command_name}.sh";;
  functional) "$PROJECT_DIR/scripts/validate-functional.sh";;
  scale) "$PROJECT_DIR/scripts/create-kwok-nodes.sh" "$profile"; "$PROJECT_DIR/scripts/run-scale-test.sh" "$profile";;
  all) "$PROJECT_DIR/scripts/preflight.sh"; "$PROJECT_DIR/scripts/bootstrap.sh"; "$PROJECT_DIR/scripts/create-cluster.sh"; "$PROJECT_DIR/scripts/install-kwok.sh"; "$PROJECT_DIR/scripts/install-fake-gpu.sh"; "$PROJECT_DIR/scripts/validate-functional.sh"; "$PROJECT_DIR/scripts/create-kwok-nodes.sh" "$profile"; "$PROJECT_DIR/scripts/run-scale-test.sh" "$profile";;
  *) die "Unknown command: $command_name";;
esac
