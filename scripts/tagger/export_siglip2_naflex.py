#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import torch
from transformers import Siglip2Model


CHECKPOINT_SHA256 = "11a61a2068800d5f4f35cb041c1fea25de86ce87725e4c97a9ed046d0b22c076"
TOKENIZER_SHA256 = "61a7b147390c64585d6c3543dd6fc636906c9af3865a5548f27f31aee1d4c8e2"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


class TableVisionEmbeddings(torch.nn.Module):
    def __init__(self, embeddings: torch.nn.Module, shapes: list[list[int]]) -> None:
        super().__init__()
        self.patch_embedding = embeddings.patch_embedding
        positional = embeddings.position_embedding.weight.reshape(
            embeddings.position_embedding_size,
            embeddings.position_embedding_size,
            -1,
        )
        table = [
            embeddings.resize_positional_embeddings(
                positional,
                torch.tensor([shape], dtype=torch.int64),
                256,
            )[0]
            for shape in shapes
        ]
        self.register_buffer("position_table", torch.stack(table))
        self.register_buffer("supported_shapes", torch.tensor(shapes, dtype=torch.int64))

    def forward(
        self, pixel_values: torch.Tensor, spatial_shapes: torch.Tensor
    ) -> torch.Tensor:
        patch_embeds = self.patch_embedding(pixel_values.to(self.patch_embedding.weight.dtype))
        matches = torch.all(
            self.supported_shapes.unsqueeze(0) == spatial_shapes.unsqueeze(1), dim=-1
        )
        positions = self.position_table[torch.argmax(matches.to(torch.int64), dim=1)]
        return patch_embeds + positions


class ImageTower(torch.nn.Module):
    def __init__(self, vision: torch.nn.Module) -> None:
        super().__init__()
        self.vision = vision

    def forward(
        self,
        pixel_values: torch.Tensor,
        pixel_attention_mask: torch.Tensor,
        spatial_shapes: torch.Tensor,
    ) -> torch.Tensor:
        pooled = self.vision(
            pixel_values=pixel_values,
            pixel_attention_mask=pixel_attention_mask,
            spatial_shapes=spatial_shapes,
            return_dict=False,
        )[1]
        return torch.nn.functional.normalize(pooled, dim=-1)


class TextTower(torch.nn.Module):
    def __init__(self, text: torch.nn.Module) -> None:
        super().__init__()
        self.text = text

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        pooled = self.text(input_ids=input_ids, return_dict=False)[1]
        return torch.nn.functional.normalize(pooled, dim=-1)


def export_image(module: torch.nn.Module, path: Path, shape: list[int]) -> None:
    patch_count = 256
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        module.eval(),
        (
            torch.zeros((1, patch_count, 16 * 16 * 3), dtype=torch.float32),
            torch.ones((1, patch_count), dtype=torch.int64),
            torch.tensor([shape], dtype=torch.int64),
        ),
        str(path),
        input_names=["pixel_values", "pixel_attention_mask", "spatial_shapes"],
        output_names=["embedding"],
        opset_version=17,
        dynamo=False,
        external_data=True,
    )


def export_text(module: torch.nn.Module, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        module.eval(),
        torch.zeros((1, 64), dtype=torch.int64),
        str(path),
        input_names=["input_ids"],
        output_names=["embedding"],
        dynamic_axes={"input_ids": {0: "batch"}, "embedding": {0: "batch"}},
        opset_version=17,
        dynamo=False,
        external_data=True,
    )


def manifest(shapes: list[list[int]]) -> dict:
    return {
        "format": 1,
        "id": "siglip2-so400m-p16-naflex",
        "version": "google-cc24074f-fp32-multiview-v1",
        "dimensions": 1152,
        "contextLength": 64,
        "image": {
            "model": "vision.onnx",
            "input": "pixel_values",
            "pixelAttentionMask": "pixel_attention_mask",
            "spatialShapes": "spatial_shapes",
            "output": "embedding",
            "width": 0,
            "height": 0,
            "patchSize": 16,
            "maxNumPatches": 256,
            "supportedShapes": shapes,
            "mean": [0.5, 0.5, 0.5],
            "std": [0.5, 0.5, 0.5],
        },
        "text": {
            "model": "text.onnx",
            "tokenizer": "tokenizer.model",
            "input": "input_ids",
            "output": "embedding",
            "lowercase": True,
            "eosToken": 1,
            "padToken": 0,
        },
        "provenance": {
            "repository": "google/siglip2-so400m-patch16-naflex",
            "revision": "cc24074f717b612951c2dead130904ab9b65a81e",
            "checkpointSha256": CHECKPOINT_SHA256,
            "tokenizerSha256": TOKENIZER_SHA256,
            "transformers": "5.15.1",
            "license": "Apache-2.0",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--shapes", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--tower", choices=("image", "text"), required=True)
    args = parser.parse_args()
    if sha256(args.checkpoint / "model.safetensors") != CHECKPOINT_SHA256:
        raise ValueError("unexpected SigLIP2 SO400M checkpoint SHA-256")
    if sha256(args.checkpoint / "tokenizer.model") != TOKENIZER_SHA256:
        raise ValueError("unexpected SigLIP2 SO400M tokenizer SHA-256")
    shapes = json.loads(args.shapes.read_text(encoding="utf-8"))["shapes"]
    model = Siglip2Model.from_pretrained(
        args.checkpoint,
        local_files_only=True,
        attn_implementation="eager",
    ).eval()
    if args.tower == "image":
        model.vision_model.embeddings = TableVisionEmbeddings(
            model.vision_model.embeddings, shapes
        )
        export_image(ImageTower(model.vision_model), args.output / "vision.onnx", shapes[0])
    else:
        export_text(TextTower(model.text_model), args.output / "text.onnx")
    args.output.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.checkpoint / "tokenizer.model", args.output / "tokenizer.model")
    (args.output / "semantic-pack.json").write_text(
        json.dumps(manifest(shapes), indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
