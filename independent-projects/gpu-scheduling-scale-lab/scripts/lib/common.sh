#!/usr/bin/env bash
set -Eeuo pipefail

COMMON_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_DIR="$(CDPATH= cd -- "$COMMON_DIR/../.." && pwd -P)"
RUNTIME_DIR="$PROJECT_DIR/.runtime"
BIN_DIR="$RUNTIME_DIR/bin"
KUBECONFIG_FILE="$RUNTIME_DIR/kubeconfig"
RESULTS_DIR="$RUNTIME_DIR/results"

# shellcheck disable=SC1091
. "$PROJECT_DIR/versions.env"
# shellcheck disable=SC1091
. "$PROJECT_DIR/config/defaults.env"

DRY_RUN=${DRY_RUN:-false}
PLATFORM_MODE=${PLATFORM_MODE:-auto}
BATCH_SIZE=${BATCH_SIZE:-100}
WAIT_TIMEOUT_SECONDS=${WAIT_TIMEOUT_SECONDS:-300}
CONFIRM_LARGE=${CONFIRM_LARGE:-false}
INSTALL_DOCKER=${INSTALL_DOCKER:-false}

log() { printf '[gpu-scale-lab] %s\n' "$*" >&2; }
warn() { printf '[gpu-scale-lab] WARNING: %s\n' "$*" >&2; }
die() { printf '[gpu-scale-lab] ERROR: %s\n' "$*" >&2; exit 1; }
command_exists() { command -v "$1" >/dev/null 2>&1; }
run() { if test "$DRY_RUN" = true; then printf '+ '; printf '%q ' "$@"; printf '\n'; else "$@"; fi; }

ensure_runtime() { mkdir -p "$BIN_DIR" "$RESULTS_DIR"; chmod 700 "$RUNTIME_DIR"; }
tool() { if test -x "$BIN_DIR/$1"; then printf '%s\n' "$BIN_DIR/$1"; else command -v "$1" 2>/dev/null || return 1; fi; }
kube() { KUBECONFIG="$KUBECONFIG_FILE" "$(tool kubectl)" "$@"; }
helm_cmd() { KUBECONFIG="$KUBECONFIG_FILE" "$(tool helm)" "$@"; }
kind_cmd() { "$(tool kind)" "$@"; }

host_os() { case "$(uname -s)" in Darwin) printf darwin;; Linux) printf linux;; *) die "Unsupported host OS: $(uname -s)";; esac; }
map_arch() { case "$1" in arm64|aarch64) printf arm64;; x86_64|amd64) printf amd64;; *) return 1;; esac; }
host_arch() { map_arch "$(uname -m)" || die "Unsupported architecture: $(uname -m)"; }
cpu_count() { if test "$(host_os)" = darwin; then sysctl -n hw.ncpu; elif command_exists nproc; then nproc; else getconf _NPROCESSORS_ONLN 2>/dev/null || printf 1; fi; }
checksum_command() { if test "$(host_os)" = darwin && command_exists shasum; then printf 'shasum -a 256'; elif command_exists sha256sum; then printf sha256sum; elif command_exists shasum; then printf 'shasum -a 256'; else return 1; fi; }
verify_sha256() { file=$1; expected=$2; checker=$(checksum_command) || die 'Neither shasum nor sha256sum is available'; actual=$($checker "$file" | awk '{print $1}'); test "$actual" = "$expected" || die "Checksum mismatch for $file"; }

require_positive_integer() { case "$2" in ''|*[!0-9]*) die "$1 must be a positive integer";; 0) die "$1 must be greater than zero";; esac; }
validate_platform_mode() { case "$1" in auto|native|emulated-amd64) :;; *) die "Invalid platform mode: $1";; esac; }
validate_runtime_target() { case "$1" in "$PROJECT_DIR/.runtime"|"$PROJECT_DIR/.runtime/"*) :;; *) die "Refusing destructive runtime target outside $PROJECT_DIR/.runtime: $1";; esac; }
validate_cluster_name() { case "$1" in ''|*[!a-zA-Z0-9._-]*) die 'Unsafe or empty cluster name';; esac; }
experiment_id_valid() { case "$1" in gpu-scale-[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]T[0-9][0-9][0-9][0-9][0-9][0-9]Z-[0-9][0-9][0-9][0-9]*) return 0;; *) return 1;; esac; }

wait_for() { description=$1; timeout=$2; shift 2; start=$(date +%s); while ! "$@"; do now=$(date +%s); test $((now-start)) -lt "$timeout" || die "Timed out waiting for $description after ${timeout}s"; sleep "$POLL_INTERVAL_SECONDS"; done; }
percentile() { p=$1; sort -n | awk -v p="$p" 'NF {a[++n]=$1} END {if (!n) {print 0; exit}; idx=int((p*n+99)/100); if(idx<1)idx=1; if(idx>n)idx=n; print a[idx]}'; }
batch_ranges() { total=$1; size=$2; require_positive_integer total "$total"; require_positive_integer batch-size "$size"; start=1; while test "$start" -le "$total"; do end=$((start+size-1)); test "$end" -le "$total" || end=$total; printf '%s %s\n' "$start" "$end"; start=$((end+1)); done; }

docker_arch() { docker info --format '{{.Architecture}}' 2>/dev/null | sed 's/aarch64/arm64/;s/x86_64/amd64/'; }
manifest_supports_arch() { image=$1; arch=$2; output=$(docker buildx imagetools inspect "$image" 2>/dev/null || docker manifest inspect "$image" 2>/dev/null || return 2); printf '%s\n' "$output" | grep -Eq "linux[/ ]$arch|architecture.*$arch|Platform:.*linux/$arch"; }
sanitize_line() { printf '%s' "$1" | sed -E 's#(token|password|secret|key)=[^[:space:]]+#\1=[REDACTED]#Ig'; }
