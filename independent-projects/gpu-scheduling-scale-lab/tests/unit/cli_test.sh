#!/usr/bin/env bash
set -Eeuo pipefail
TEST_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_DIR="$(CDPATH= cd -- "$TEST_DIR/../.." && pwd -P)"
"$PROJECT_DIR/run.sh" help | grep -q 'emulated-amd64'
if "$PROJECT_DIR/run.sh" unknown-command >/dev/null 2>&1; then exit 1; fi
if "$PROJECT_DIR/run.sh" help --platform-mode invalid >/dev/null 2>&1; then exit 1; fi
if "$PROJECT_DIR/run.sh" help --batch-size 0 >/dev/null 2>&1; then exit 1; fi
output=$("$PROJECT_DIR/run.sh" down --dry-run)
printf '%s\n' "$output" | grep -qv 'delete cluster' || true
"$PROJECT_DIR/run.sh" scale --profile smoke --dry-run >/dev/null
override_output=$(KWOK_NODES=4 SYNTHETIC_PODS=12 "$PROJECT_DIR/run.sh" scale --profile smoke --dry-run 2>&1)
printf '%s\n' "$override_output" | grep -q 'create 4 API-only nodes'
printf '%s\n' "$override_output" | grep -q '12 synthetic pods'
printf 'ok - CLI validation and dry-run cleanup behavior\n'
