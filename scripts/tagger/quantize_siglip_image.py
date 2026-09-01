#!/usr/bin/env python3
"""Build selectively quantized SigLIP image-tower experiment packs."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import onnx
from onnxruntime.quantization import QuantType, quantize_dynamic


def selected_nodes(model_path: Path, profile: str) -> list[str]:
    model = onnx.load(model_path, load_external_data=False)
    initializers = {value.name for value in model.graph.initializer}
    weighted = [
        node
        for node in model.graph.node
        if node.op_type == "MatMul" and len(node.input) > 1 and node.input[1] in initializers
    ]

    def layer(node: onnx.NodeProto) -> int | None:
        match = re.search(r"/encoder/layers\.(\d+)/", node.name)
        return int(match.group(1)) if match else None

    if profile == "attention":
        chosen = [node for node in weighted if "/self_attn/" in node.name]
    elif profile == "mlp":
        chosen = [node for node in weighted if "/mlp/" in node.name]
    elif profile == "early-half":
        chosen = [node for node in weighted if layer(node) is not None and layer(node) < 6]
    elif profile == "early-three-quarters":
        chosen = [node for node in weighted if layer(node) is not None and layer(node) < 9]
    elif profile == "backbone":
        chosen = [node for node in weighted if layer(node) is not None]
    else:
        raise ValueError(f"unknown quantization profile: {profile}")
    if not chosen:
        raise RuntimeError(f"profile {profile} selected no weighted MatMul nodes")
    return [node.name for node in chosen]


def link(source: Path, destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    destination.symlink_to(source.resolve(), target_is_directory=source.is_dir())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--profile",
        choices=("attention", "mlp", "early-half", "early-three-quarters", "backbone"),
        required=True,
    )
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    source = manifest_path.parent
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_image = source / manifest["image"]["model"]
    args.output.mkdir(parents=True, exist_ok=True)
    output_image = args.output / f"siglip2-image-int8-{args.profile}.onnx"
    nodes = selected_nodes(source_image, args.profile)
    quantize_dynamic(
        source_image,
        output_image,
        op_types_to_quantize=["MatMul"],
        nodes_to_quantize=nodes,
        per_channel=True,
        weight_type=QuantType.QInt8,
        use_external_data_format=True,
    )

    manifest["version"] = f'{manifest["version"]}-image-int8-{args.profile}'
    manifest["image"]["model"] = output_image.name
    for relative in (
        manifest["text"]["model"],
        manifest["text"]["tokenEmbeddings"]["table"],
    ):
        link(source / relative, args.output / Path(relative).name)
    link(source / "tokenizer", args.output / "tokenizer")
    (args.output / "semantic-pack.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "manifest": str(args.output / "semantic-pack.json"),
                "profile": args.profile,
                "quantizedNodes": len(nodes),
                "imageModelBytes": output_image.stat().st_size,
            }
        )
    )


if __name__ == "__main__":
    main()
