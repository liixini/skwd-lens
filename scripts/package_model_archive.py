#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import subprocess
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOCK = ROOT / "packaging/default-semantic.lock.json"
DEFAULT_PIN = ROOT / "packaging/skwd-lens-model-1.0.0.tar.xz.sha256"
SEMANTIC_ROOT = PurePosixPath("usr/share/skwd-lens/models/semantic")
PACKAGE_NAME = "skwd-lens-model"
PINNED_PRODUCT = "siglip2-base-p16-224@google-image-int8-attention-text-int8-stretch-v4"
MODEL_VERSION = "1.0.0"
PACK_PREFIX = "semantic"
LICENSE_PREFIX = "licenses"
LICENSE_FILES = ("Apache-2.0.txt", "CC-BY-4.0.txt", "MIT.txt")


class PackageError(RuntimeError):
    pass


def safe_relative_path(value: object) -> PurePosixPath:
    if not isinstance(value, str) or not value:
        raise PackageError("default semantic lock contains an invalid path")
    pure = PurePosixPath(value)
    if pure.is_absolute() or any(part in ("", ".", "..") for part in pure.parts):
        raise PackageError(f"default semantic lock contains an unsafe path: {value}")
    return pure


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_lock(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PackageError(f"cannot read default semantic lock {path}: {error}") from error
    if not isinstance(value, dict) or value.get("format") != 1:
        raise PackageError("default semantic lock must be a format 1 object")
    if value.get("product") != PINNED_PRODUCT:
        raise PackageError(
            f"default semantic lock ships {value.get('product')!r}, not the pinned "
            f"{PINNED_PRODUCT!r}; bump MODEL_VERSION deliberately when the pack changes"
        )
    entries = value.get("files")
    if not isinstance(entries, list) or not entries:
        raise PackageError("default semantic lock must contain files")
    return value


def extract_pack(source: Path, workspace: Path) -> Path:
    member_root = str(SEMANTIC_ROOT)
    with tarfile.open(source, "r:xz") as archive:
        selected = []
        for member in archive:
            name = PurePosixPath(member.name)
            if name.is_absolute() or ".." in name.parts:
                raise PackageError(f"release asset contains an unsafe path: {member.name}")
            if member.issym() or member.islnk():
                raise PackageError(f"release asset contains a link: {member.name}")
            if member.name == member_root or member.name.startswith(member_root + "/"):
                selected.append(member)
        if not selected:
            raise PackageError(f"release asset has no {member_root} tree: {source}")
        archive.extractall(workspace, members=selected, filter="data")
    return workspace / member_root


def verified_files(lock: dict[str, Any], pack: Path) -> list[tuple[str, Path]]:
    resolved = []
    names = []
    for entry in lock["files"]:
        if not isinstance(entry, dict):
            raise PackageError("default semantic lock entries must be objects")
        relative = safe_relative_path(entry.get("path"))
        expected_digest = entry.get("sha256")
        expected_size = entry.get("size")
        if not isinstance(expected_digest, str) or not isinstance(expected_size, int):
            raise PackageError(f"lock entry lacks size and sha256: {relative}")
        candidate = pack / Path(*relative.parts)
        cursor = pack
        for part in relative.parts:
            cursor /= part
            if cursor.is_symlink():
                raise PackageError(f"pack file must not traverse a symlink: {relative}")
        try:
            status = candidate.stat()
        except OSError as error:
            raise PackageError(f"pack file is missing: {relative} ({error})") from error
        if not stat.S_ISREG(status.st_mode):
            raise PackageError(f"pack file is not a regular file: {relative}")
        if status.st_size != expected_size:
            raise PackageError(
                f"size mismatch for {relative}: expected {expected_size}, got {status.st_size}"
            )
        actual_digest = sha256(candidate)
        if actual_digest != expected_digest.lower():
            raise PackageError(
                f"SHA-256 mismatch for {relative}: expected {expected_digest}, got {actual_digest}"
            )
        names.append(str(relative))
        resolved.append((str(relative), candidate))
    if names != sorted(names) or len(names) != len(set(names)):
        raise PackageError("default semantic lock paths must be unique and sorted")
    present = sorted(
        str(path.relative_to(pack).as_posix()) for path in pack.rglob("*") if path.is_file()
    )
    if present != sorted(names):
        raise PackageError(
            "pack tree does not match the lock exactly; "
            f"unexpected {sorted(set(present) - set(names))}"
        )
    return resolved


def license_files() -> list[tuple[str, Path]]:
    resolved = []
    for name in LICENSE_FILES:
        source = ROOT / "LICENSES" / name
        if not source.is_file():
            raise PackageError(f"missing Lens license material: {source}")
        resolved.append((f"{LICENSE_PREFIX}/{name}", source))
    return resolved


def write_archive(files: list[tuple[str, Path]], output: Path, prefix: str, epoch: int) -> None:
    directories = sorted({str(PurePosixPath(name).parent) for name, _ in files} - {"."})
    xz = shutil.which("xz")
    if xz is None:
        raise PackageError("xz is required to write the model archive")
    with output.open("wb") as raw:
        process = subprocess.Popen(
            [xz, "-6", "--threads=4", "--block-size=64MiB", "--no-adjust", "-c"],
            stdin=subprocess.PIPE,
            stdout=raw,
        )
        try:
            with tarfile.open(fileobj=process.stdin, mode="w|", format=tarfile.PAX_FORMAT) as archive:
                for name in [""] + directories:
                    info = tarfile.TarInfo(f"{prefix}/{name}".rstrip("/"))
                    info.type = tarfile.DIRTYPE
                    info.mode = 0o755
                    info.mtime = epoch
                    info.uid = info.gid = 0
                    info.uname = info.gname = "root"
                    archive.addfile(info)
                for name, source in files:
                    info = tarfile.TarInfo(f"{prefix}/{name}")
                    info.size = source.stat().st_size
                    info.mode = 0o644
                    info.mtime = epoch
                    info.uid = info.gid = 0
                    info.uname = info.gname = "root"
                    with source.open("rb") as handle:
                        archive.addfile(info, handle)
        finally:
            process.stdin.close()
        if process.wait() != 0:
            raise PackageError(f"xz failed with exit status {process.returncode}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the redistributable skwd-lens-model archive from a release asset."
    )
    parser.add_argument("asset", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--pin", type=Path, default=DEFAULT_PIN)
    arguments = parser.parse_args()

    epoch = int(os.environ.get("SOURCE_DATE_EPOCH", "0"))
    output_directory = arguments.output.expanduser().absolute()
    output_directory.mkdir(parents=True, exist_ok=True)
    prefix = f"{PACKAGE_NAME}-{MODEL_VERSION}"
    archive_path = output_directory / f"{prefix}.tar.xz"
    if archive_path.exists():
        raise PackageError(f"refusing to overwrite an existing model archive: {archive_path}")

    lock = load_lock(arguments.lock)
    with tempfile.TemporaryDirectory(prefix=".skwd-lens-model-") as temporary:
        pack = extract_pack(arguments.asset.expanduser().resolve(strict=True), Path(temporary))
        verified = verified_files(lock, pack)
        files = [(f"{PACK_PREFIX}/{name}", source) for name, source in verified]
        files.extend(license_files())
        files.sort(key=lambda entry: entry[0])
        write_archive(files, archive_path, prefix, epoch)
    digest = sha256(archive_path)
    if arguments.pin.exists():
        expected = arguments.pin.read_text(encoding="utf-8").split()[0]
        if digest != expected:
            archive_path.unlink()
            raise PackageError(
                f"model archive digest {digest} does not match the pinned {expected}"
            )

    print(
        json.dumps(
            {
                "archive": str(archive_path),
                "sha256": digest,
                "bytes": archive_path.stat().st_size,
                "package": PACKAGE_NAME,
                "version": MODEL_VERSION,
                "product": lock["product"],
                "files": len(verified),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except PackageError as error:
        raise SystemExit(f"error: {error}") from error
