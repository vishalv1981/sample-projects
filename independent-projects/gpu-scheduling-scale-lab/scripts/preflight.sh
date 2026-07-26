#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
. "$SCRIPT_DIR/lib/common.sh"
ensure_runtime

test "${BASH_VERSINFO[0]}" -ge 3 || die 'Bash 3.2 or newer is required'
for utility in curl awk sed grep sort tar chmod uname date; do command_exists "$utility" || die "Required utility missing: $utility"; done
checksum_command >/dev/null || die 'Install shasum or sha256sum'
command_exists docker || {
  if test "$(host_os)" = darwin; then die 'Docker Desktop is required. Install it from https://docs.docker.com/desktop/setup/install/mac-install/ and start it.'; fi
  if test "$INSTALL_DOCKER" = true; then die 'Automatic Docker installation is intentionally not implemented; follow https://docs.docker.com/engine/install/ubuntu/ on this existing VM.'; fi
  die 'Docker Engine is required; rerun with --install-docker for official installation guidance'
}
docker info >/dev/null 2>&1 || die 'Docker is installed but its daemon is unavailable. Start Docker Desktop or Docker Engine.'

engine_arch=$(docker_arch); test -n "$engine_arch" || die 'Docker did not report an architecture'
cpus=$(docker info --format '{{.NCPU}}'); memory=$(docker info --format '{{.MemTotal}}'); memory_gib=$((memory/1073741824))
disk=$(docker system df --format '{{json .}}' 2>/dev/null | head -1 || true)
log "Host=$(host_os)/$(host_arch) Docker=$engine_arch CPUs=$cpus Memory=${memory_gib}GiB"
test "$cpus" -ge 4 || warn 'Docker has fewer than 4 CPUs; smoke may work but scheduling tests can be slow.'
test "$memory_gib" -ge 6 || warn 'Docker has less than 6 GiB; smoke may work but operator pods may be constrained.'
test -n "$disk" || warn 'Docker disk availability could not be determined.'

compat_file="$RUNTIME_DIR/image-compatibility.txt"
: > "$compat_file"
check_image() {
  image=$1; required_arch=$2
  if manifest_supports_arch "$image" "$required_arch"; then printf '%s %s supported\n' "$image" "$required_arch" >> "$compat_file"; return 0; fi
  status=$?
  test "$status" -ne 2 || die "Unable to inspect image manifest: $image"
  printf '%s %s unsupported\n' "$image" "$required_arch" >> "$compat_file"
  return 1
}

target_arch=$engine_arch
if test "$PLATFORM_MODE" = emulated-amd64; then target_arch=amd64; warn 'AMD64 emulation is explicit, potentially slow, and not performance-representative.'; fi
check_image "$KIND_NODE_IMAGE" "$target_arch" || die "Pinned kind image lacks linux/$target_arch"
check_image "$KWOK_IMAGE" "$target_arch" || die "Pinned KWOK image lacks linux/$target_arch"
fake_ok=true
for component in device-plugin status-updater topology-server status-exporter; do
  check_image "$FAKE_GPU_IMAGE_PREFIX/$component:$FAKE_GPU_IMAGE_TAG" "$target_arch" || fake_ok=false
done
check_image "$FAKE_GPU_UBUNTU_IMAGE" "$target_arch" || fake_ok=false
check_image "$FAKE_GPU_PLACEHOLDER_IMAGE" "$target_arch" || fake_ok=false
if test "$fake_ok" != true; then
  : > "$RUNTIME_DIR/fake-gpu-incompatible"
  if test "$PLATFORM_MODE" = emulated-amd64; then die 'A required Fake GPU Operator image lacks AMD64 support'; fi
  warn "Fake GPU functional path is incompatible with linux/$target_arch; KWOK scale testing remains available. Use an AMD64 VM or explicit emulated-amd64 mode."
else
  rm -f "$RUNTIME_DIR/fake-gpu-incompatible"
fi
log "Sanitized compatibility results: $compat_file"
