#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import subprocess
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOCK = ROOT / "packaging/default-semantic.lock.json"
DEFAULT_MANIFEST = ROOT / "scripts/tagger/model-packs/maximum/semantic-pack.json"
PACKAGE_MANIFEST = ROOT / "packaging/default-manifest.txt"
SEMANTIC_ROOT = Path("usr/share/skwd-lens/models/semantic")


class PackageError(RuntimeError):
    pass


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PackageError(f"cannot read default semantic lock {path}: {error}") from error
    if not isinstance(value, dict):
        raise PackageError("default semantic lock must contain a JSON object")
    return value


def safe_relative_path(value: object) -> Path:
    if not isinstance(value, str) or not value:
        raise PackageError("default semantic lock contains an invalid path")
    pure = PurePosixPath(value)
    if pure.is_absolute() or any(part in ("", ".", "..") for part in pure.parts):
        raise PackageError(f"default semantic lock contains an unsafe path: {value}")
    return Path(*pure.parts)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def regular_source(root: Path, relative: Path, label: str) -> Path:
    candidate = root / relative
    cursor = root
    for part in relative.parts:
        cursor /= part
        if cursor.is_symlink():
            raise PackageError(f"{label} must not traverse a symlink: {relative.as_posix()}")
    try:
        status = candidate.stat()
    except OSError as error:
        raise PackageError(f"{label} is missing: {candidate} ({error})") from error
    if not stat.S_ISREG(status.st_mode):
        raise PackageError(f"{label} is not a regular file: {candidate}")
    return candidate


def checked_files(lock_path: Path, payload: Path, runtime: Path, manifest: Path):
    lock = load_json(lock_path)
    if lock.get("format") != 1:
        raise PackageError("default semantic lock format must be 1")
    entries = lock.get("files")
    if not isinstance(entries, list) or not entries:
        raise PackageError("default semantic lock must contain files")
    resolved = []
    paths = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise PackageError("default semantic lock file entries must be objects")
        relative = safe_relative_path(entry.get("path"))
        source_kind = entry.get("source")
        if source_kind == "manifest":
            if relative != Path("semantic-pack.json"):
                raise PackageError("the manifest source may only provide semantic-pack.json")
            source = regular_source(manifest.parent, Path(manifest.name), "default manifest")
        elif source_kind == "payload":
            if relative.parts[0] == "runtime":
                raise PackageError(f"payload entry is inside runtime: {relative.as_posix()}")
            source = regular_source(payload, relative, "default payload")
        elif source_kind == "runtime":
            if relative.parts[0] != "runtime" or len(relative.parts) < 2:
                raise PackageError(f"runtime entry has the wrong destination: {relative.as_posix()}")
            source = regular_source(runtime, Path(*relative.parts[1:]), "ONNX Runtime payload")
        else:
            raise PackageError(f"unknown default semantic source: {source_kind}")
        expected_size = entry.get("size")
        expected_digest = entry.get("sha256")
        if not isinstance(expected_size, int) or expected_size < 0:
            raise PackageError(f"invalid size for {relative.as_posix()}")
        if not isinstance(expected_digest, str) or len(expected_digest) != 64:
            raise PackageError(f"invalid SHA-256 for {relative.as_posix()}")
        actual_size = source.stat().st_size
        if actual_size != expected_size:
            raise PackageError(
                f"size mismatch for {relative.as_posix()}: expected {expected_size}, got {actual_size}"
            )
        actual_digest = sha256(source)
        if actual_digest != expected_digest.lower():
            raise PackageError(
                f"SHA-256 mismatch for {relative.as_posix()}: expected {expected_digest}, got {actual_digest}"
            )
        paths.append(relative.as_posix())
        resolved.append((relative, source))
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise PackageError("default semantic lock paths must be unique and sorted")
    return lock, resolved


def validate_destination(destination: Path) -> Path:
    expanded = destination.expanduser().absolute()
    if expanded.name in ("", ".", "..") or expanded == Path("/"):
        raise PackageError(f"package destination must name a dedicated directory: {expanded}")
    if expanded.is_symlink() or (expanded.exists() and not expanded.is_dir()):
        raise PackageError(f"package destination must be a directory: {expanded}")
    if expanded.exists() and any(expanded.iterdir()):
        raise PackageError(f"package destination is not empty: {expanded}")
    expanded.parent.mkdir(parents=True, exist_ok=True)
    return expanded.parent.resolve(strict=True) / expanded.name


def expected_manifest(path: Path) -> tuple[str, ...]:
    entries = tuple(path.read_text(encoding="utf-8").splitlines())
    if entries != tuple(sorted(entries)) or not entries:
        raise PackageError(f"package manifest must be non-empty and sorted: {path}")
    return entries


def staged_files(root: Path) -> tuple[str, ...]:
    return tuple(
        sorted(
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file()
        )
    )


def stage(
    destination: Path,
    payload: Path,
    runtime: Path,
    lock_path: Path,
    manifest: Path,
    package_manifest: Path,
) -> tuple[Path, dict[str, Any]]:
    target = validate_destination(destination)
    payload_root = payload.expanduser().resolve(strict=True)
    runtime_root = runtime.expanduser().resolve(strict=True)
    lock, files = checked_files(lock_path, payload_root, runtime_root, manifest)
    expected = expected_manifest(package_manifest)
    with tempfile.TemporaryDirectory(prefix=".skwd-lens-package-", dir=target.parent) as temporary:
        candidate = Path(temporary) / "root"
        subprocess.run(
            ["sh", str(ROOT / "scripts/package-stage.sh"), str(candidate)],
            cwd=ROOT,
            check=True,
        )
        semantic = candidate / SEMANTIC_ROOT
        for relative, source in files:
            output = semantic / relative
            output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, output)
            output.chmod(0o644)
        if staged_files(candidate) != expected:
            raise PackageError("staged default package does not match packaging/default-manifest.txt")
        from tagger.stage_semantic_pack import validate_product

        validate_product(semantic, require_root=True)
        if target.exists():
            target.rmdir()
        os.rename(candidate, target)
    return target, lock


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage the complete default skwd-lens package.",
        usage="package-default-stage.sh DESTINATION PAYLOAD ONNX_RUNTIME",
    )
    parser.add_argument("destination", type=Path)
    parser.add_argument("payload", type=Path)
    parser.add_argument("runtime", type=Path)
    parser.add_argument(
        "--lock",
        type=Path,
        default=Path(os.environ.get("SKWD_LENS_DEFAULT_LOCK", DEFAULT_LOCK)),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(os.environ.get("SKWD_LENS_DEFAULT_MANIFEST", DEFAULT_MANIFEST)),
    )
    parser.add_argument(
        "--package-manifest",
        type=Path,
        default=Path(os.environ.get("SKWD_LENS_PACKAGE_MANIFEST", PACKAGE_MANIFEST)),
    )
    arguments = parser.parse_args()
    try:
        destination, lock = stage(
            arguments.destination,
            arguments.payload,
            arguments.runtime,
            arguments.lock,
            arguments.manifest,
            arguments.package_manifest,
        )
    except (OSError, PackageError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"default Lens packaging failed: {error}\n")
    print(f"{destination} ({lock['product']}, ONNX Runtime {lock['runtimeVersion']})")


if __name__ == "__main__":
    main()
