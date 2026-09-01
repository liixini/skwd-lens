#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import torch


CHECKPOINT_SHA256 = "0cdab5b338cbaa1e7a5dcd1b2fb4c9f4d5df1abd289564658edbab64a650e7e8"
MODEL_NAME = "PE-Core-L14-336"
SOURCE_REVISION = "3e352cca660658d4b5c90f42a7808b11469e4c66"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


class ImageTower(torch.nn.Module):
    def __init__(self, visual: torch.nn.Module) -> None:
        super().__init__()
        self.visual = visual

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.normalize(self.visual(pixel_values), dim=-1)


class TextTower(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.normalize(self.model.encode_text(input_ids), dim=-1)


def export(module: torch.nn.Module, inputs: torch.Tensor, path: Path, input_name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        module.eval(),
        inputs,
        str(path),
        input_names=[input_name],
        output_names=["embedding"],
        dynamic_axes={input_name: {0: "batch"}, "embedding": {0: "batch"}},
        opset_version=17,
        dynamo=False,
        external_data=True,
    )


def manifest() -> dict:
    return {
        "format": 1,
        "id": "pe-core-l14-336",
        "version": "facebook-bafb0f76-fp32-multiview-v1",
        "dimensions": 1024,
        "contextLength": 32,
        "image": {
            "model": "vision.onnx",
            "input": "pixel_values",
            "output": "embedding",
            "resizeMode": "stretch",
            "resizeFilter": "bilinear",
            "width": 336,
            "height": 336,
            "mean": [0.5, 0.5, 0.5],
            "std": [0.5, 0.5, 0.5],
        },
        "text": {
            "model": "text.onnx",
            "tokenizer": "tokenizer.json",
            "input": "input_ids",
            "output": "embedding",
            "eosToken": 49407,
            "padToken": 0,
        },
        "provenance": {
            "repository": "facebook/PE-Core-L14-336",
            "revision": "bafb0f76541d399057e980a25947f67acec76575",
            "checkpointSha256": CHECKPOINT_SHA256,
            "sourceRevision": SOURCE_REVISION,
            "license": "Apache-2.0",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--tower", choices=("image", "text"), required=True)
    args = parser.parse_args()
    if sha256(args.checkpoint) != CHECKPOINT_SHA256:
        raise ValueError("unexpected PE-Core-L14-336 checkpoint SHA-256")
    sys.path.insert(0, str(args.source))
    from core.vision_encoder.pe import CLIP

    model = CLIP.from_config(
        MODEL_NAME,
        pretrained=True,
        checkpoint_path=str(args.checkpoint),
    ).eval()
    if args.tower == "image":
        export(
            ImageTower(model.visual),
            torch.zeros((1, 3, 336, 336), dtype=torch.float32),
            args.output / "vision.onnx",
            "pixel_values",
        )
    else:
        model.visual = None
        export(
            TextTower(model),
            torch.zeros((1, 32), dtype=torch.int64),
            args.output / "text.onnx",
            "input_ids",
        )
    args.output.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.tokenizer, args.output / "tokenizer.json")
    (args.output / "semantic-pack.json").write_text(
        json.dumps(manifest(), indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
