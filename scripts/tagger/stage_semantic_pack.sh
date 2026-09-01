#!/bin/sh
set -eu

root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
exec python3 "$root/scripts/tagger/stage_semantic_pack.py" "$@"
