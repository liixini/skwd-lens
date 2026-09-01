#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from PIL import Image


VIEW_CODES = {"full": 0, "center": 1, "leftThird": 2, "rightThird": 3}


def patch_grid(height: int, width: int, patch_size: int, max_patches: int) -> tuple[int, int]:
    minimum = 0.000001
    maximum = 100.0

    def scaled(size: int, scale: float) -> int:
        return max(1, math.ceil(size * scale / patch_size))

    while maximum - minimum >= 0.00001:
        scale = (minimum + maximum) / 2
        patch_height = scaled(height, scale)
        patch_width = scaled(width, scale)
        if patch_height * patch_width <= max_patches:
            minimum = scale
        else:
            maximum = scale
    return scaled(height, minimum), scaled(width, minimum)


def view_dimensions(width: int, height: int, view: str) -> tuple[int, int]:
    if view == "center":
        edge = min(width, height)
        return edge, edge
    if view in ("leftThird", "rightThird"):
        return max(1, width // 3), height
    return width, height


def fingerprint(path: Path, view: str) -> int:
    stat = path.stat()
    digest = hashlib.blake2b(digest_size=8, person=b"skwdview")
    digest.update(str(path).encode())
    digest.update(stat.st_size.to_bytes(8, "little"))
    digest.update(stat.st_mtime_ns.to_bytes(8, "little"))
    digest.update(VIEW_CODES[view].to_bytes(1, "little"))
    return int.from_bytes(digest.digest(), "little")


def catalog_fingerprint(entries: list[dict]) -> int:
    digest = hashlib.blake2b(digest_size=8, person=b"skwdcatv")
    for entry in entries:
        digest.update(entry["key"].encode())
        digest.update(entry["fingerprint"].to_bytes(8, "little"))
    return int.from_bytes(digest.digest(), "little")


def source_path(entry: dict, static_root: Path) -> Path:
    if entry["key"].startswith("static:"):
        original = static_root / entry["key"].removeprefix("static:")
        if not original.is_file():
            raise FileNotFoundError(original)
        return original.resolve()
    path = Path(entry["path"])
    if not path.is_file():
        raise FileNotFoundError(path)
    return path.resolve()


def prepare(
    source: dict,
    static_root: Path,
    ultrawide_ratio: float,
    patch_size: int,
    max_patches: int,
) -> tuple[dict, dict, list[list[int]]]:
    full_entries = []
    multiview_entries = []
    shapes = set()
    for source_entry in source["entries"]:
        path = source_path(source_entry, static_root)
        with Image.open(path) as image:
            width, height = image.size
        views = ["full", "center"]
        if source_entry["key"].startswith("static:") and width / height >= ultrawide_ratio:
            views.extend(("leftThird", "rightThird"))
        for view in views:
            entry = {
                "key": source_entry["key"],
                "path": str(path),
                "fingerprint": fingerprint(path, view),
            }
            if view != "full":
                entry["view"] = view
            multiview_entries.append(entry)
            view_width, view_height = view_dimensions(width, height, view)
            shapes.add(patch_grid(view_height, view_width, patch_size, max_patches))
            if view == "full":
                full_entries.append(entry)
    full = {"fingerprint": catalog_fingerprint(full_entries), "entries": full_entries}
    multiview = {
        "fingerprint": catalog_fingerprint(multiview_entries),
        "entries": multiview_entries,
    }
    return full, multiview, [list(shape) for shape in sorted(shapes)]


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, separators=(",", ":")) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument("--static-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--ultrawide-ratio", type=float, default=2.0)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--max-patches", type=int, default=256)
    args = parser.parse_args()
    if args.ultrawide_ratio <= 1 or args.patch_size <= 0 or args.max_patches <= 0:
        raise ValueError("invalid semantic view configuration")
    source = json.loads(args.catalog.read_text(encoding="utf-8"))
    full, multiview, shapes = prepare(
        source,
        args.static_root,
        args.ultrawide_ratio,
        args.patch_size,
        args.max_patches,
    )
    write_json(args.output / "full.json", full)
    write_json(args.output / "multiview.json", multiview)
    write_json(
        args.output / "naflex-shapes.json",
        {
            "patchSize": args.patch_size,
            "maxNumPatches": args.max_patches,
            "shapes": shapes,
        },
    )
    print(
        json.dumps(
            {
                "items": len(full["entries"]),
                "views": len(multiview["entries"]),
                "extraViews": len(multiview["entries"]) - len(full["entries"]),
                "naflexShapes": len(shapes),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
