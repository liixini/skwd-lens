#!/usr/bin/env python3

from __future__ import annotations

import argparse
import ctypes
import errno
import filecmp
import hashlib
import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any


class StageError(RuntimeError):
    pass


REQUIRED_REFERENCES = (
    ("image.model", ("image", "model")),
    ("text.model", ("text", "model")),
    ("text.tokenizer", ("text", "tokenizer")),
)

OPTIONAL_REFERENCES = (
    ("text.tokenEmbeddings.table", ("text", "tokenEmbeddings", "table")),
    ("text.projection.path", ("text", "projection", "path")),
)


def load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise StageError(f"cannot read {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise StageError(f"{label} must contain a JSON object: {path}")
    return value


def nested_value(value: dict[str, Any], keys: tuple[str, ...]) -> Any:
    current: Any = value
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def set_nested(value: dict[str, Any], keys: tuple[str, ...], replacement: str) -> None:
    current: dict[str, Any] = value
    for key in keys[:-1]:
        child = current.get(key)
        if not isinstance(child, dict):
            raise StageError(f"semantic manifest field {'.'.join(keys)} has an invalid parent")
        current = child
    current[keys[-1]] = replacement


def relative_reference(origin: Path, target: Path) -> str:
    return Path(os.path.relpath(target, origin)).as_posix()


def resolve_reference(origin: Path, reference: str, label: str) -> Path:
    path = Path(reference)
    if path.is_absolute():
        raise StageError(f"{label} must be relative, got absolute path {reference}")
    cursor = origin
    for component in path.parts:
        if component in ("", "."):
            continue
        if component == "..":
            cursor = cursor.parent
            continue
        cursor /= component
        if cursor.is_symlink():
            raise StageError(f"{label} must not traverse a symlink: {reference}")
    try:
        resolved = (origin / path).resolve(strict=True)
    except OSError as error:
        raise StageError(f"{label} is missing: {reference} ({error})") from error
    if not resolved.is_file():
        raise StageError(f"{label} is not a file: {reference}")
    return resolved


def ensure_inside(path: Path, root: Path, label: str, reference: str) -> None:
    if not path.is_relative_to(root):
        raise StageError(f"{label} escapes the staged semantic product: {reference}")


def tokenizer_target(prepared_root: Path, manifest_dir: Path, source: Path) -> Path:
    candidates = (
        prepared_root / "tokenizer" / source.name,
        manifest_dir / "tokenizer" / source.name,
    )
    for candidate in candidates:
        if candidate.is_symlink():
            raise StageError(f"tokenizer destination must not be a symlink: {candidate}")
        if candidate.is_file() and filecmp.cmp(source, candidate, shallow=False):
            return candidate
    target = candidates[-1]
    if target.exists():
        digest = hashlib.sha256(source.read_bytes()).hexdigest()[:12]
        target = target.with_name(f"{target.stem}-{digest}{target.suffix}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink():
        raise StageError(f"tokenizer destination must not be a symlink: {target}")
    shutil.copy2(source, target)
    return target


def normalize_manifest(
    source_root: Path,
    prepared_root: Path,
    relative_manifest: Path,
) -> None:
    source_manifest = source_root / relative_manifest
    prepared_manifest = prepared_root / relative_manifest
    manifest = load_json(prepared_manifest, "semantic manifest")
    fields = list(REQUIRED_REFERENCES)
    fields.extend(
        (label, keys)
        for label, keys in OPTIONAL_REFERENCES
        if nested_value(manifest, keys) is not None
    )
    for label, keys in fields:
        reference = nested_value(manifest, keys)
        if not isinstance(reference, str) or not reference:
            raise StageError(f"semantic manifest field {label} must be a non-empty path")
        source = resolve_reference(source_manifest.parent, reference, label)
        if source.is_relative_to(source_root):
            prepared = resolve_reference(prepared_manifest.parent, reference, label)
            ensure_inside(prepared, prepared_root, label, reference)
            replacement = relative_reference(prepared_manifest.parent, prepared)
        elif label == "text.tokenizer":
            prepared = tokenizer_target(prepared_root, prepared_manifest.parent, source)
            replacement = relative_reference(prepared_manifest.parent, prepared)
        else:
            raise StageError(
                f"{label} escapes the source pack: {reference}; copy the asset into the pack"
            )
        set_nested(manifest, keys, replacement)
    prepared_manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def validate_manifest(stage_root: Path, manifest_path: Path) -> None:
    manifest = load_json(manifest_path, "semantic manifest")
    fields = list(REQUIRED_REFERENCES)
    fields.extend(
        (label, keys)
        for label, keys in OPTIONAL_REFERENCES
        if nested_value(manifest, keys) is not None
    )
    for label, keys in fields:
        reference = nested_value(manifest, keys)
        if not isinstance(reference, str) or not reference:
            raise StageError(f"semantic manifest field {label} must be a non-empty path")
        resolved = resolve_reference(manifest_path.parent, reference, label)
        ensure_inside(resolved, stage_root, label, reference)


def validate_payload_tree(
    root: Path,
    label: str,
    *,
    allow_internal_file_symlinks: bool = False,
) -> None:
    resolved_root = root.resolve()
    pending = [root]
    while pending:
        directory = pending.pop()
        try:
            entries = list(os.scandir(directory))
        except OSError as error:
            raise StageError(f"cannot inspect {label} {directory}: {error}") from error
        for entry in entries:
            path = Path(entry.path)
            if entry.is_symlink():
                if allow_internal_file_symlinks:
                    try:
                        target = path.resolve(strict=True)
                    except OSError as error:
                        raise StageError(f"{label} contains a broken symlink: {path}") from error
                    if target.is_relative_to(resolved_root) and target.is_file():
                        continue
                raise StageError(f"{label} must not contain symlinks: {path}")
            if entry.is_dir(follow_symlinks=False):
                pending.append(path)
            elif not entry.is_file(follow_symlinks=False):
                raise StageError(f"{label} contains an unsupported filesystem entry: {path}")


def validate_product(stage_root: Path, require_root: bool) -> None:
    validate_payload_tree(stage_root, "semantic product")
    root = stage_root.resolve()
    root_manifest = stage_root / "semantic-pack.json"
    if require_root and not root_manifest.is_file():
        raise StageError(f"semantic pack input is missing: {root_manifest}")
    for manifest in sorted(stage_root.rglob("semantic-pack.json")):
        validate_manifest(root, manifest)


def validate_directory(path: Path, label: str) -> Path:
    try:
        resolved = path.expanduser().resolve(strict=True)
    except OSError as error:
        raise StageError(f"{label} does not exist: {path} ({error})") from error
    if not resolved.is_dir():
        raise StageError(f"{label} is not a directory: {path}")
    return resolved


def output_path(path: Path) -> Path:
    expanded = path.expanduser()
    if expanded.name in ("", ".", ".."):
        raise StageError(f"semantic output must name a dedicated directory: {expanded}")
    if expanded.is_symlink() or (expanded.exists() and not expanded.is_dir()):
        raise StageError(f"semantic output must be a directory, not a file or symlink: {expanded}")
    parent = expanded.absolute().parent
    parent.mkdir(parents=True, exist_ok=True)
    return parent.resolve(strict=True) / expanded.name


def remove_candidate_path(path: Path, candidate_root: Path) -> None:
    if not path.absolute().is_relative_to(candidate_root.absolute()):
        raise StageError(f"refusing to replace path outside staging candidate: {path}")
    if path.is_symlink():
        raise StageError(f"staging candidate unexpectedly contains a symlink: {path}")
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def copy_entry(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        shutil.copytree(source, destination)
    else:
        shutil.copy2(source, destination)


def replace_entry(source: Path, destination: Path, candidate_root: Path) -> None:
    remove_candidate_path(destination, candidate_root)
    copy_entry(source, destination)


def compose_candidate(
    existing: Path,
    candidate: Path,
    prepared_pack: Path,
    prepared_runtime: Path,
    name: str,
) -> None:
    if existing.exists():
        shutil.copytree(existing, candidate)
    else:
        candidate.mkdir()
    if name:
        replace_entry(prepared_pack, candidate / "packs" / name, candidate)
    else:
        for source in prepared_pack.iterdir():
            if source.name == "packs" and source.is_dir():
                for pack in source.iterdir():
                    replace_entry(pack, candidate / "packs" / pack.name, candidate)
            else:
                replace_entry(source, candidate / source.name, candidate)
    replace_entry(prepared_runtime, candidate / "runtime", candidate)


def exchange_directories(first: Path, second: Path) -> bool:
    renameat2 = getattr(ctypes.CDLL(None, use_errno=True), "renameat2", None)
    if renameat2 is None:
        return False
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100,
        os.fsencode(first),
        -100,
        os.fsencode(second),
        2,
    )
    if result == 0:
        return True
    error = ctypes.get_errno()
    if error in (errno.ENOSYS, errno.EINVAL, errno.EOPNOTSUPP):
        return False
    raise OSError(error, os.strerror(error), first, second)


def publish_candidate(candidate: Path, destination: Path) -> None:
    if not destination.exists():
        os.rename(candidate, destination)
        return
    try:
        exchanged = exchange_directories(candidate, destination)
    except OSError as error:
        raise StageError(f"cannot atomically replace semantic output: {error}") from error
    if not exchanged:
        raise StageError(
            "cannot atomically replace semantic output: directory exchange is unsupported"
        )


def stage(pack: Path, runtime: Path, root: Path, name: str) -> Path:
    source_root = validate_directory(pack, "semantic pack")
    runtime_root = validate_directory(runtime, "ONNX Runtime")
    if name and re.fullmatch(r"[a-z0-9-]+", name) is None:
        raise StageError("semantic pack name must use lowercase letters, digits, and dashes")
    destination_root = output_path(root)
    for source, label in ((source_root, "semantic pack"), (runtime_root, "ONNX Runtime")):
        if (
            destination_root == source
            or destination_root.is_relative_to(source)
            or source.is_relative_to(destination_root)
        ):
            raise StageError(f"semantic output and {label} must be separate directory trees")
    validate_payload_tree(source_root, "semantic pack")
    validate_payload_tree(
        runtime_root,
        "ONNX Runtime",
        allow_internal_file_symlinks=True,
    )
    runtime_library = runtime_root / "libonnxruntime.so.1.27.0"
    if not runtime_library.is_file():
        raise StageError(f"ONNX Runtime input is missing: {runtime_library}")
    if destination_root.exists():
        validate_product(destination_root, require_root=False)
    with tempfile.TemporaryDirectory(
        prefix=".skwd-lens-stage-", dir=destination_root.parent
    ) as temporary:
        temporary_root = Path(temporary)
        prepared_pack = temporary_root / "pack"
        prepared_runtime = temporary_root / "runtime"
        shutil.copytree(source_root, prepared_pack)
        shutil.copytree(runtime_root, prepared_runtime)
        manifests = sorted(prepared_pack.rglob("semantic-pack.json"))
        if prepared_pack / "semantic-pack.json" not in manifests:
            raise StageError(f"semantic pack input is missing: {source_root / 'semantic-pack.json'}")
        for manifest in manifests:
            normalize_manifest(source_root, prepared_pack, manifest.relative_to(prepared_pack))
        validate_product(prepared_pack, require_root=True)
        prepared_runtime_library = prepared_runtime / "libonnxruntime.so.1.27.0"
        if not prepared_runtime_library.is_file():
            raise StageError(f"staged ONNX Runtime is missing: {prepared_runtime_library}")
        candidate = temporary_root / "candidate"
        compose_candidate(
            destination_root,
            candidate,
            prepared_pack,
            prepared_runtime,
            name,
        )
        validate_product(candidate, require_root=not name)
        publish_candidate(candidate, destination_root)
    destination = destination_root if not name else destination_root / "packs" / name
    return destination / "semantic-pack.json"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage a self-contained semantic model product.",
        usage="stage_semantic_pack.sh PACK RUNTIME [OUTPUT] [NAME]",
    )
    parser.add_argument("pack", type=Path)
    parser.add_argument("runtime", type=Path)
    parser.add_argument("output", type=Path, nargs="?", default=Path("target/release/lens"))
    parser.add_argument("name", nargs="?", default="")
    arguments = parser.parse_args()
    try:
        result = stage(arguments.pack, arguments.runtime, arguments.output, arguments.name)
    except StageError as error:
        parser.exit(1, f"semantic pack staging failed: {error}\n")
    print(result)


if __name__ == "__main__":
    main()
