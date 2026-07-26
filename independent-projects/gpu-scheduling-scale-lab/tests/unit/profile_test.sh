#!/usr/bin/env bash
set -Eeuo pipefail
TEST_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_DIR="$(CDPATH= cd -- "$TEST_DIR/../.." && pwd -P)"
expected='smoke:3:10 small:50:500 medium:500:5000 large:1000:10000'
actual=''
for profile in smoke small medium large; do
  unset KWOK_NODES SYNTHETIC_PODS GPU_REQUEST_MAX
  . "$PROJECT_DIR/experiments/profiles/$profile.env"
  case "$KWOK_NODES:$SYNTHETIC_PODS:$GPU_REQUEST_MAX" in *[!0-9:]*) exit 1;; esac
  actual="$actual $profile:$KWOK_NODES:$SYNTHETIC_PODS"
done
test "${actual# }" = "$expected"
printf 'ok - profile parsing and conservative defaults\n'
