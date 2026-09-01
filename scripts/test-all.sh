#!/bin/sh
set -eu

cd "$(dirname "$0")/.."

heavy=0
for argument in "$@"; do
    case "$argument" in
        --heavy) heavy=1 ;;
        -h|--help)
            echo "usage: scripts/test-all.sh [--heavy]"
            exit 0
            ;;
        *)
            echo "unknown argument: $argument" >&2
            exit 2
            ;;
    esac
done

environment=${SKWD_LENS_TEST_ENV:-.venv-lens}
if [ ! -x "$environment/bin/python" ]; then
    echo "$environment is missing; run scripts/setup-lens-test-env.sh" >&2
    exit 1
fi

VERIFY_ROOT="${SKWD_VERIFY_ROOT:-../skwd-verify}"
if [ ! -f "$VERIFY_ROOT/scripts/python_suite.py" ]; then
    echo "missing skwd-verify checkout at $VERIFY_ROOT (set SKWD_VERIFY_ROOT)" >&2
    exit 1
fi

cargo fmt --check
cargo clippy --workspace --all-targets -- -D warnings
"$VERIFY_ROOT/scripts/check-unsafe-code.sh"
cargo test --release --workspace
cargo build --release --workspace
nvidia_library_path=$("$environment/bin/python" -c '
import site
from pathlib import Path

paths = (
    path
    for root in site.getsitepackages()
    for path in (Path(root) / "nvidia").glob("*/lib")
    if path.is_dir()
)
print(":".join(sorted(map(str, paths))))
')
if [ -n "${LD_LIBRARY_PATH:-}" ]; then
    nvidia_library_path="${nvidia_library_path:+$nvidia_library_path:}$LD_LIBRARY_PATH"
fi

env LD_LIBRARY_PATH="$nvidia_library_path" PYTHONDONTWRITEBYTECODE=1 \
    "$environment/bin/python" \
    "$VERIFY_ROOT/scripts/python_suite.py" python-lens

if [ "$heavy" -eq 1 ]; then
    env LD_LIBRARY_PATH="$nvidia_library_path" PYTHONDONTWRITEBYTECODE=1 \
        "$environment/bin/python" \
        "$VERIFY_ROOT/scripts/python_suite.py" python-lens-heavy
fi
