#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import resource
import struct
import time
from pathlib import Path

import numpy as np
import onnxruntime
from PIL import Image


def index_embedding(path: Path, wanted: str) -> tuple[str, np.ndarray]:
    with path.open("rb") as file:
        if file.read(8) != b"SKWDSEM3":
            raise RuntimeError("unsupported semantic index")
        dimensions, model_length = struct.unpack("<II", file.read(8))
        model = file.read(model_length).decode("utf-8")
        file.read(8)
        count = struct.unpack("<Q", file.read(8))[0]
        for _ in range(count):
            key_length = struct.unpack("<I", file.read(4))[0]
            key = file.read(key_length).decode("utf-8")
            file.read(8)
            values = file.read(dimensions * 4)
            if len(values) != dimensions * 4:
                raise RuntimeError("truncated semantic index")
            if key == wanted:
                return model, np.frombuffer(values, dtype="<f4").copy()
    raise RuntimeError(f"key not present in index: {wanted}")


def preprocess(
    path: Path,
    width: int,
    height: int,
    mean: tuple[float, float, float] = (0.0, 0.0, 0.0),
    std: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> np.ndarray:
    with Image.open(path) as source:
        image = source.convert("RGB")
        scale = max(width / image.width, height / image.height)
        resized = image.resize(
            (round(image.width * scale), round(image.height * scale)),
            Image.Resampling.BICUBIC,
        )
    left = (resized.width - width) // 2
    top = (resized.height - height) // 2
    cropped = resized.crop((left, top, left + width, top + height))
    values = np.asarray(cropped, dtype=np.float32) / 255.0
    values = (values - np.asarray(mean, dtype=np.float32)) / np.asarray(
        std,
        dtype=np.float32,
    )
    return np.transpose(values, (2, 0, 1))[None]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image", type=Path, action="append", required=True)
    parser.add_argument("--index", type=Path)
    parser.add_argument("--key")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    if (args.index is None) != (args.key is None):
        parser.error("--index and --key must be used together")

    manifest_path = args.manifest.resolve()
    root = manifest_path.parent
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    image_config = manifest["image"]
    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = args.threads
    options.inter_op_num_threads = 1
    options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
    started = time.perf_counter()
    session = onnxruntime.InferenceSession(
        str(root / image_config["model"]),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    load_seconds = time.perf_counter() - started
    encode_seconds = []
    embeddings = []
    for path in args.image:
        values = preprocess(
            path.resolve(),
            int(image_config["width"]),
            int(image_config["height"]),
            tuple(image_config["mean"]),
            tuple(image_config["std"]),
        )
        started = time.perf_counter()
        embedding = session.run(
            ["image_embedding"],
            {"images": values},
        )[0][0]
        encode_seconds.append(time.perf_counter() - started)
        embeddings.append(embedding)
    reference = None
    if args.index is not None and args.key is not None:
        index_model, expected = index_embedding(args.index.resolve(), args.key)
        measured = embeddings[0]
        reference = {
            "indexModel": index_model,
            "key": args.key,
            "cosine": float(measured @ expected),
            "maximumAbsoluteDifference": float(np.max(np.abs(measured - expected))),
        }
    model = root / image_config["model"]
    model_bytes = model.stat().st_size
    data = Path(f"{model}.data")
    if data.exists():
        model_bytes += data.stat().st_size
    report = json.dumps(
        {
            "format": 1,
            "model": manifest["id"],
            "images": len(args.image),
            "imageModelBytes": model_bytes,
            "modelLoadSeconds": load_seconds,
            "encodeSeconds": encode_seconds,
            "peakResidentMiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            / 1024,
            "reference": reference,
        },
        indent=2,
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report + "\n", encoding="utf-8")
        print(args.output)
    else:
        print(report)


if __name__ == "__main__":
    main()
