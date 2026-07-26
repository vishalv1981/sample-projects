#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
"$SCRIPT_DIR/preflight.sh"
"$SCRIPT_DIR/create-cluster.sh"
"$SCRIPT_DIR/install-kwok.sh"
"$SCRIPT_DIR/install-fake-gpu.sh"
