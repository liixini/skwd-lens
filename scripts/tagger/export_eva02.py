#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import torch

EVA02_B_SHA256 = "4aab21dd652dc19aab7ff2302e5a07582d7838e8a3b26c80d697115fb0c6ddf0"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while block := file.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


class ImageTower(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.model.encode_image(images, normalize=True)


class TextTower(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.encode_text(input_ids, normalize=True)


def export(
    module: torch.nn.Module,
    inputs: torch.Tensor,
    output: Path,
    input_name: str,
    output_name: str,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        module,
        inputs,
        output,
        input_names=[input_name],
        output_names=[output_name],
        dynamic_axes={input_name: {0: "batch"}, output_name: {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eva-source", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--tower", choices=("both", "image", "text"), default="both")
    args = parser.parse_args()
    actual_hash = sha256(args.checkpoint)
    if actual_hash != EVA02_B_SHA256:
        raise ValueError(f"unexpected EVA02-B checkpoint SHA-256: {actual_hash}")

    sys.path.insert(0, str(args.eva_source / "EVA-CLIP" / "rei"))
    from eva_clip import create_model_and_transforms, factory

    model_name = "EVA02-CLIP-B-16"
    config = factory._MODEL_CONFIGS[model_name]
    config["vision_cfg"]["xattn"] = False
    config["vision_cfg"]["fusedLN"] = False
    config["text_cfg"]["xattn"] = False
    config["text_cfg"]["fusedLN"] = False
    model, _, _ = create_model_and_transforms(
        model_name,
        str(args.checkpoint),
        force_custom_clip=True,
    )
    model.eval()

    if args.tower in ("both", "image"):
        export(
            ImageTower(model),
            torch.zeros((1, 3, 224, 224), dtype=torch.float32),
            args.output / "vision.onnx",
            "images",
            "image_embedding",
        )
    if args.tower in ("both", "text"):
        export(
            TextTower(model),
            torch.zeros((1, 77), dtype=torch.int64),
            args.output / "text.onnx",
            "input_ids",
            "text_embedding",
        )


if __name__ == "__main__":
    main()
