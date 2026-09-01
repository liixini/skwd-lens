#!/usr/bin/env python3
"""Stage the exact SigLIP2 semantic search product pack."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load_manifest(root: Path) -> dict:
    return json.loads((root / "semantic-pack.json").read_text(encoding="utf-8"))


def copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def copy_model(root: Path, relative: str, destination: Path) -> str:
    source = root / relative
    target = destination / source.name
    copy_file(source, target)
    external = source.with_name(source.name + ".data")
    if external.is_file():
        copy_file(external, destination / external.name)
    return target.name


def copy_tokenizer(root: Path, relative: str, destination: Path) -> str:
    source = root / relative
    target = destination / "tokenizer" / source.name
    copy_file(source, target)
    return str(Path("tokenizer") / source.name)


def write_manifest(destination: Path, manifest: dict) -> None:
    (destination / "semantic-pack.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-pack", type=Path, required=True)
    parser.add_argument("--text-pack", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    image_manifest = load_manifest(args.image_pack)
    text_manifest = load_manifest(args.text_pack)
    if (image_manifest["id"], image_manifest["version"]) != (
        text_manifest["id"],
        text_manifest["version"],
    ):
        raise RuntimeError("text and image towers do not share an embedding space")

    args.output.mkdir(parents=True, exist_ok=True)
    product = json.loads(json.dumps(text_manifest))
    product["image"] = json.loads(json.dumps(image_manifest["image"]))
    product["image"]["model"] = copy_model(
        args.image_pack,
        image_manifest["image"]["model"],
        args.output,
    )
    product["text"]["model"] = copy_model(
        args.text_pack,
        text_manifest["text"]["model"],
        args.output,
    )
    product["text"]["tokenizer"] = copy_tokenizer(
        args.text_pack,
        text_manifest["text"]["tokenizer"],
        args.output,
    )
    table = text_manifest["text"]["tokenEmbeddings"]["table"]
    product["text"]["tokenEmbeddings"]["table"] = copy_model(
        args.text_pack,
        table,
        args.output,
    )
    write_manifest(args.output, product)
    print(args.output / "semantic-pack.json")


if __name__ == "__main__":
    main()
