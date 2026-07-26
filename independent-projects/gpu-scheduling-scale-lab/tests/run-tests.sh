#!/usr/bin/env bash
set -Eeuo pipefail
TEST_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
for test_file in "$TEST_DIR"/unit/*_test.sh; do /bin/bash "$test_file"; done
printf 'All unit tests passed.\n'
