#!/usr/bin/env bash
set -Eeuo pipefail
TEST_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
. "$TEST_DIR/../../scripts/lib/common.sh"
failures=0
assert_eq() { if test "$1" != "$2"; then printf 'not ok - expected [%s], got [%s]\n' "$2" "$1"; failures=$((failures+1)); else printf 'ok - %s\n' "$3"; fi; }
assert_fails() { if ( "$@" ) >/dev/null 2>&1; then printf 'not ok - expected failure: %s\n' "$*"; failures=$((failures+1)); else printf 'ok - rejected %s\n' "$*"; fi; }

assert_eq "$(map_arch arm64)" arm64 'arm64 mapping'
assert_eq "$(map_arch aarch64)" arm64 'aarch64 mapping'
assert_eq "$(map_arch x86_64)" amd64 'x86_64 mapping'
assert_fails map_arch sparc
assert_eq "$(printf '9\n1\n5\n2\n8\n' | percentile 50)" 5 'P50 calculation'
assert_eq "$(printf '1\n2\n3\n4\n100\n' | percentile 95)" 100 'P95 nearest-rank calculation'
assert_eq "$(batch_ranges 10 4)" "1 4
5 8
9 10" 'batch ranges'
assert_fails validate_platform_mode surprise
assert_fails require_positive_integer count 0
assert_fails validate_runtime_target ''
assert_fails validate_runtime_target "$PROJECT_DIR"
validate_runtime_target "$PROJECT_DIR/.runtime/results/test"
experiment_id_valid gpu-scale-20260726T120000Z-1234 || failures=$((failures+1))
assert_fails experiment_id_valid unsafe
checker=$(checksum_command); test -n "$checker" || failures=$((failures+1)); printf 'ok - checksum helper: %s\n' "$checker"

docker() { if test "$1" = buildx; then printf 'Name: test\nPlatform: linux/arm64\n'; return 0; fi; return 1; }
manifest_supports_arch example.invalid/test:v1 arm64 || failures=$((failures+1))
assert_fails manifest_supports_arch example.invalid/test:v1 amd64

test "$failures" -eq 0 || exit 1
