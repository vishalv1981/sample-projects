#!/usr/bin/env bash
set -Eeuo pipefail
TEST_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_DIR="$(CDPATH= cd -- "$TEST_DIR/../.." && pwd -P)"
manifest="$PROJECT_DIR/manifests/functional/gpu-validation-pod.yaml"
grep -q 'nvidia.com/gpu: 1' "$manifest"
grep -q 'run.ai/simulated-gpu-node-pool: default' "$manifest"
grep -q 'restartPolicy: Never' "$manifest"
readme="$PROJECT_DIR/README.md"
test "$(grep -c '^```mermaid$' "$readme")" -eq 1
grep -q '^flowchart TD$' "$readme"
grep -Fq 'W --> F["Simulated GPU nodes and pods"]' "$readme"
printf 'ok - functional manifest rendering invariants\n'
