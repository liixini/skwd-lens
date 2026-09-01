#!/bin/sh
set -eu

root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
payload=${1:?usage: scripts/build-default-portable.sh PAYLOAD ONNX_RUNTIME [OUTPUT]}
runtime=${2:?usage: scripts/build-default-portable.sh PAYLOAD ONNX_RUNTIME [OUTPUT]}
output=${3:-$root/dist}
version=$(sed -n 's/^version = "\([^"]*\)"/\1/p' "$root/Cargo.toml" | head -n 1)
case "$version" in
    *[!0-9.a-zA-Z+-]*|"")
        echo "invalid Lens version: $version" >&2
        exit 1
        ;;
esac
case "$(uname -m)" in
    x86_64) architecture=x86_64 ;;
    aarch64|arm64) architecture=aarch64 ;;
    *) echo "unsupported architecture: $(uname -m)" >&2; exit 1 ;;
esac
case "$output" in
    /*) ;;
    *) output="$root/$output" ;;
esac

temporary=$(mktemp -d)
cleanup() {
    [ -n "${temporary:-}" ] || return 0
    case "$temporary" in
        "${TMPDIR:-/tmp}"/tmp.*) ;;
        *) echo "refusing to remove unexpected temporary path: $temporary" >&2; return 0 ;;
    esac
    [ ! -L "$temporary" ] || return 0
    [ ! -e "$temporary" ] || rm -rf -- "$temporary"
}
trap cleanup EXIT HUP INT TERM

"$root/scripts/package-default-stage.sh" "$temporary/stage" "$payload" "$runtime"
mkdir -p "$output"
archive="skwd-lens_${version}_linux-${architecture}.tar.xz"
tar -C "$temporary/stage" -cJf "$output/$archive" usr
(cd "$output" && sha256sum "$archive" > SHA256SUMS)
echo "built $output/$archive"
