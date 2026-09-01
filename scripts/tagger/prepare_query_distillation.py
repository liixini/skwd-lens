#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import struct
from pathlib import Path

import onnx
from onnx import TensorProto, helper


SIGLIP_REVISION = "ba1f3b0843f24bc5417d38e19c37b287d719b2f4"
TINYCLIP_REVISION = "737108a175dc6c043d9e64cf738baa91a272f7cb"
EXPECTED = {
    "siglip-text.onnx": "3a0603d3a00c05a80a6ded4743c16aaac7b1e62cdcc7e362e7ce418659b96400",
    "siglip-vision.onnx": "0dd31785a2713f1113ef2272472165c69d580473dae38d7b47568ac587795e70",
    "siglip-tokenizer.model": "61a7b147390c64585d6c3543dd6fc636906c9af3865a5548f27f31aee1d4c8e2",
    "tinyclip.onnx": "0843578fa3e1c386a3be5250ccb13cc8836d062876a34756f380a2e371e86a75",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(files: dict[str, Path]) -> None:
    for name, expected in EXPECTED.items():
        actual = sha256(files[name])
        if actual != expected:
            raise RuntimeError(f"{name} checksum is {actual}, expected {expected}")


def normalized_vision(source: Path, destination: Path) -> None:
    model = onnx.load(source, load_external_data=True)
    pooled = next(output for output in model.graph.output if output.name == "pooler_output")
    model.graph.node.extend(
        [
            helper.make_node(
                "ReduceL2",
                [pooled.name],
                ["pooler_norm"],
                axes=[1],
                keepdims=1,
            ),
            helper.make_node("Div", [pooled.name, "pooler_norm"], ["embedding"]),
        ]
    )
    del model.graph.output[:]
    model.graph.output.extend(
        [helper.make_tensor_value_info("embedding", TensorProto.FLOAT, [None, 768])]
    )
    onnx.checker.check_model(model)
    onnx.save(model, destination)


def extract_student(source: Path, destination: Path) -> None:
    onnx.utils.extract_model(
        source,
        destination,
        ["input_ids", "attention_mask"],
        ["text_embeds"],
        check_model=True,
    )
    model = onnx.load(destination)
    for value in model.graph.input:
        dimensions = value.type.tensor_type.shape.dim
        dimensions[0].ClearField("dim_param")
        dimensions[0].dim_value = 1
        dimensions[1].ClearField("dim_param")
        dimensions[1].dim_value = 77
    model = onnx.shape_inference.infer_shapes(model, data_prop=True)
    onnx.checker.check_model(model)
    onnx.save(model, destination)


def write_index(path: Path, dimensions: int, identity: str) -> None:
    encoded = identity.encode("utf-8")
    path.write_bytes(
        b"SKWDSEM3"
        + struct.pack("<II", dimensions, len(encoded))
        + encoded
        + struct.pack("<QQI", 0, 1, 0)
        + struct.pack("<Q", 0)
        + bytes(dimensions * 4)
    )


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--siglip", type=Path, required=True)
    parser.add_argument("--tinyclip", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    files = {
        "siglip-text.onnx": args.siglip / "onnx/text_model_int8.onnx",
        "siglip-vision.onnx": args.siglip / "onnx/vision_model_int8.onnx",
        "siglip-tokenizer.model": args.siglip / "tokenizer.model",
        "tinyclip.onnx": args.tinyclip / "onnx/model_int8.onnx",
    }
    verify(files)
    teacher = args.output / "teacher"
    student = args.output / "student"
    teacher.mkdir(parents=True, exist_ok=True)
    student.mkdir(parents=True, exist_ok=True)
    shutil.copy2(files["siglip-text.onnx"], teacher / "text.onnx")
    shutil.copy2(files["siglip-tokenizer.model"], teacher / "tokenizer.model")
    normalized_vision(files["siglip-vision.onnx"], teacher / "vision.onnx")
    extract_student(files["tinyclip.onnx"], student / "text.onnx")
    for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        shutil.copy2(args.tinyclip / name, student / name)

    teacher_id = "siglip2-base-p16-224-community-int8"
    teacher_version = SIGLIP_REVISION
    teacher_manifest = {
        "format": 1,
        "id": teacher_id,
        "version": teacher_version,
        "dimensions": 768,
        "contextLength": 64,
        "image": {
            "model": "vision.onnx",
            "input": "pixel_values",
            "output": "embedding",
            "resizeMode": "stretch",
            "width": 224,
            "height": 224,
            "mean": [0.5, 0.5, 0.5],
            "std": [0.5, 0.5, 0.5],
        },
        "text": {
            "model": "text.onnx",
            "tokenizer": "tokenizer.model",
            "input": "input_ids",
            "maskPadding": False,
            "lowercase": True,
            "output": "pooler_output",
            "eosToken": 1,
            "padToken": 0,
        },
    }
    student_id = "tinyclip-19m-text-int8"
    student_version = TINYCLIP_REVISION
    student_manifest = {
        "format": 1,
        "id": student_id,
        "version": student_version,
        "dimensions": 512,
        "contextLength": 77,
        "image": {
            "model": "text.onnx",
            "input": "pixel_values",
            "output": "text_embeds",
            "resizeMode": "stretch",
            "width": 224,
            "height": 224,
            "mean": [0.48145466, 0.4578275, 0.40821073],
            "std": [0.26862954, 0.26130258, 0.27577711],
        },
        "text": {
            "model": "text.onnx",
            "tokenizer": "tokenizer.json",
            "input": "input_ids",
            "attentionMask": "attention_mask",
            "maskPadding": True,
            "lowercase": True,
            "output": "text_embeds",
            "eosToken": 49407,
            "padToken": 49407,
        },
    }
    write_json(teacher / "semantic-pack.json", teacher_manifest)
    write_json(student / "semantic-pack.json", student_manifest)
    write_index(teacher / "empty.sidx", 768, f"{teacher_id}@{teacher_version}")
    write_index(student / "empty.sidx", 512, f"{student_id}@{student_version}")
    provenance = {
        "format": 1,
        "generator": "scripts/tagger/prepare_query_distillation.py",
        "environment": {
            "requirements": "scripts/tagger/requirements-ci.txt",
            "sha256": sha256(
                Path(__file__).resolve().parent / "requirements-ci.txt"
            ),
        },
        "runtime": {"format": "ONNX", "providers": ["CPUExecutionProvider"]},
        "teacher": {
            "repository": "onnx-community/siglip2-base-patch16-224-ONNX",
            "revision": SIGLIP_REVISION,
            "license": "Apache-2.0",
        },
        "student": {
            "repository": "onnx-community/TinyCLIP-ViT-40M-32-Text-19M-LAION400M-ONNX",
            "revision": TINYCLIP_REVISION,
            "upstreamRepository": "wkcn/TinyCLIP-ViT-40M-32-Text-19M-LAION400M",
            "upstreamRevision": "95ec8197b3f2fe7f747865c61ca556cf0768b2f7",
            "license": "MIT",
        },
        "inputs": {
            name: {"bytes": files[name].stat().st_size, "sha256": EXPECTED[name]}
            for name in sorted(EXPECTED)
        },
        "outputs": {
            str(path.relative_to(args.output)): {
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in sorted([*teacher.rglob("*"), *student.rglob("*")])
            if path.is_file()
        },
    }
    write_json(args.output / "provenance.json", provenance)
    print(args.output / "provenance.json")


if __name__ == "__main__":
    main()
