#!/bin/sh
set -eu

root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
destination=${1:?usage: scripts/package-stage.sh DESTINATION}
case "$destination" in
    /)
        echo "refusing to stage directly into /" >&2
        exit 2
        ;;
    /*) ;;
    *) destination="$root/$destination" ;;
esac

if [ -L "$destination" ] || { [ -e "$destination" ] && [ ! -d "$destination" ]; }; then
    echo "package destination must be a directory: $destination" >&2
    exit 2
fi
if [ -d "$destination" ] && [ -n "$(find "$destination" -mindepth 1 -print -quit)" ]; then
    echo "package destination is not empty: $destination" >&2
    exit 2
fi

binary_directory=${SKWD_LENS_RELEASE_BIN_DIR:-$root/target/release}
case "$binary_directory" in
    /*) ;;
    *) binary_directory="$root/$binary_directory" ;;
esac

for binary in skwd-lens; do
    path="$binary_directory/$binary"
    if [ ! -f "$path" ] || [ ! -x "$path" ]; then
        echo "missing executable Lens release binary: $path" >&2
        exit 1
    fi
done

for path in \
    "$root/LICENSE" \
    "$root/LICENSES/Apache-2.0.txt" \
    "$root/LICENSES/CC-BY-4.0.txt" \
    "$root/LICENSES/MIT.txt"
do
    if [ ! -f "$path" ]; then
        echo "missing Lens license material: $path" >&2
        exit 1
    fi
done

umask 022
license_directory="$destination/usr/share/licenses/skwd-lens"
third_party_directory="$license_directory/third-party"
mkdir -p "$destination/usr/bin" "$third_party_directory"

for binary in skwd-lens; do
    install -m755 "$binary_directory/$binary" "$destination/usr/bin/$binary"
done

install -m644 "$root/LICENSE" "$license_directory/LICENSE"
install -m644 "$root/LICENSES/Apache-2.0.txt" "$third_party_directory/Apache-2.0.txt"
install -m644 "$root/LICENSES/CC-BY-4.0.txt" "$third_party_directory/CC-BY-4.0.txt"
install -m644 "$root/LICENSES/MIT.txt" "$third_party_directory/MIT.txt"
