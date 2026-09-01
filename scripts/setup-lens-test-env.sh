#!/bin/sh
set -eu

cd "$(dirname "$0")/.."

python=${SKWD_LENS_BASE_PYTHON:-python3}
environment=${SKWD_LENS_TEST_ENV:-.venv-lens}
requirements=scripts/tagger/requirements-ci.txt
index=https://pypi.org/simple

version=$($python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
case "$version" in
    3.12) ;;
    *) echo "The Lens CI wheel lock requires Python 3.12, got $version" >&2; exit 1 ;;
esac
system=$(uname -s)
machine=$(uname -m)
if [ "$system" != Linux ] || [ "$machine" != x86_64 ]; then
    echo "The Lens CI wheel lock requires Linux x86_64, got $system $machine" >&2
    exit 1
fi

"$python" scripts/tagger/python_lock.py check

if [ ! -x "$environment/bin/python" ]; then
    "$python" -m venv "$environment"
fi

"$environment/bin/python" -m pip "--isolated" install \
    "--disable-pip-version-check" \
    "--index-url" "$index" \
    "--require-hashes" \
    "--only-binary=:all:" \
    "--force-reinstall" \
    --requirement "$requirements"
