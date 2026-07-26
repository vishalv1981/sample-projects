#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
. "$SCRIPT_DIR/lib/common.sh"
ensure_runtime
os=$(host_os); arch=$(host_arch)

download() { url=$1; output=$2; run curl --fail --location --retry 3 --connect-timeout 15 --output "$output" "$url" || die "Download failed: $url"; }
install_binary() { name=$1; version=$2; url=$3; checksum_url=$4; target="$BIN_DIR/$name"; if test -x "$target"; then log "$name already present in project runtime"; return; fi; if test "$DRY_RUN" = true; then run curl "$url"; return; fi; tmp="$RUNTIME_DIR/$name.download"; sum="$RUNTIME_DIR/$name.sha256"; download "$url" "$tmp"; download "$checksum_url" "$sum"; expected=$(awk '{print $1}' "$sum"); verify_sha256 "$tmp" "$expected"; mv "$tmp" "$target"; chmod 0755 "$target"; rm -f "$sum"; log "Installed $name $version"; }

install_binary kind "$KIND_VERSION" "https://github.com/kubernetes-sigs/kind/releases/download/$KIND_VERSION/kind-$os-$arch" "https://github.com/kubernetes-sigs/kind/releases/download/$KIND_VERSION/kind-$os-$arch.sha256sum"
install_binary kubectl "$KUBECTL_VERSION" "https://dl.k8s.io/release/$KUBECTL_VERSION/bin/$os/$arch/kubectl" "https://dl.k8s.io/release/$KUBECTL_VERSION/bin/$os/$arch/kubectl.sha256"

if ! test -x "$BIN_DIR/helm"; then
  archive="$RUNTIME_DIR/helm.tar.gz"; checksum_file="$RUNTIME_DIR/helm.sha256sum"
  helm_os=$os; download "https://get.helm.sh/helm-$HELM_VERSION-$helm_os-$arch.tar.gz" "$archive"
  download "https://get.helm.sh/helm-$HELM_VERSION-$helm_os-$arch.tar.gz.sha256sum" "$checksum_file"
  if test "$DRY_RUN" != true; then expected=$(awk '{print $1}' "$checksum_file"); verify_sha256 "$archive" "$expected"; tar -xzf "$archive" -C "$RUNTIME_DIR"; mv "$RUNTIME_DIR/$helm_os-$arch/helm" "$BIN_DIR/helm"; chmod 0755 "$BIN_DIR/helm"; validate_runtime_target "$RUNTIME_DIR/$helm_os-$arch"; rm -rf "$RUNTIME_DIR/$helm_os-$arch"; rm -f "$archive" "$checksum_file"; fi
fi
if test "$DRY_RUN" = true; then log 'Dry run complete; no tools downloaded.'; exit 0; fi
"$(tool kind)" version; "$(tool kubectl)" version --client; "$(tool helm)" version --short
