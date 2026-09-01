#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def copy(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return destination.name


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-manifest", type=Path, required=True)
    parser.add_argument("--student-manifest", type=Path, required=True)
    parser.add_argument("--projection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    teacher = load(args.teacher_manifest)
    student = load(args.student_manifest)
    args.output.mkdir(parents=True, exist_ok=True)
    teacher_root = args.teacher_manifest.parent
    student_root = args.student_manifest.parent
    manifest = json.loads(json.dumps(teacher))
    manifest["contextLength"] = student["contextLength"]
    manifest["image"]["model"] = copy(
        teacher_root / teacher["image"]["model"], args.output / "vision.onnx"
    )
    manifest["text"] = json.loads(json.dumps(student["text"]))
    manifest["text"]["model"] = copy(
        student_root / student["text"]["model"], args.output / "text.onnx"
    )
    tokenizer = student_root / student["text"]["tokenizer"]
    manifest["text"]["tokenizer"] = copy(tokenizer, args.output / tokenizer.name)
    manifest["text"]["projection"] = {
        "path": copy(args.projection, args.output / "projection.bin"),
        "inputDimensions": student["dimensions"],
        "outputDimensions": teacher["dimensions"],
    }
    manifest_path = args.output / "semantic-pack.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    files = [
        path
        for path in sorted(args.output.iterdir())
        if path.is_file() and path.name != "provenance.json"
    ]
    provenance = {
        "format": 1,
        "generator": "scripts/tagger/stage_distilled_query_pack.py",
        "method": "ridge-linear-query-tower-distillation",
        "teacher": f"{teacher['id']}@{teacher['version']}",
        "student": f"{student['id']}@{student['version']}",
        "projection": {"bytes": args.projection.stat().st_size, "sha256": sha256(args.projection)},
        "outputs": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in files
        },
    }
    (args.output / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    print(manifest_path)


if __name__ == "__main__":
    main()
