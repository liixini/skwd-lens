#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import statistics
import struct
import subprocess
import time
from pathlib import Path

import numpy as np
import onnx


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def proportional_mib(pid: int) -> float:
    status = Path(f"/proc/{pid}/smaps_rollup").read_text(encoding="utf-8")
    for line in status.splitlines():
        if line.startswith("Pss:"):
            return int(line.split()[1]) / 1024
    raise RuntimeError("Pss missing")


def index_header(path: Path) -> dict:
    with path.open("rb") as source:
        if source.read(8) != b"SKWDSEM3":
            raise RuntimeError(f"unsupported index: {path}")
        dimensions, model_length = struct.unpack("<II", source.read(8))
        model = source.read(model_length).decode("utf-8")
        fingerprint = struct.unpack("<Q", source.read(8))[0]
        records = struct.unpack("<Q", source.read(8))[0]
        keys = set()
        norms = []
        for _ in range(records):
            key_length = struct.unpack("<I", source.read(4))[0]
            keys.add(source.read(key_length).decode("utf-8"))
            source.read(8)
            values = np.frombuffer(source.read(dimensions * 4), dtype="<f4")
            if len(values) != dimensions:
                raise RuntimeError(f"truncated index: {path}")
            norms.append(float(np.linalg.norm(values)))
    return {
        "dimensions": dimensions,
        "model": model,
        "fingerprint": fingerprint,
        "records": records,
        "uniqueKeys": len(keys),
        "embeddingNormMin": min(norms),
        "embeddingNormMedian": statistics.median(norms),
        "embeddingNormMax": max(norms),
        "bytes": path.stat().st_size,
    }


def onnx_assets(path: Path) -> set[Path]:
    assets = {path.resolve()}
    model = onnx.load(path, load_external_data=False)
    for initializer in model.graph.initializer:
        if initializer.data_location != onnx.TensorProto.EXTERNAL:
            continue
        fields = {item.key: item.value for item in initializer.external_data}
        assets.add((path.parent / fields["location"]).resolve())
    return assets


def model_bytes(manifest_path: Path) -> int:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assets = {manifest_path.resolve()}
    for tower in ("image", "text"):
        assets.update(onnx_assets(manifest_path.parent / manifest[tower]["model"]))
    assets.add((manifest_path.parent / manifest["text"]["tokenizer"]).resolve())
    table = manifest["text"].get("tokenEmbeddings", {}).get("table")
    if table:
        assets.add((manifest_path.parent / table).resolve())
    projection = manifest["text"].get("projection", {}).get("path")
    if projection:
        assets.add((manifest_path.parent / projection).resolve())
    return sum(path.stat().st_size for path in assets)


class Searcher:
    def __init__(
        self,
        helper: Path,
        runtime: Path,
        manifest: Path,
        index: Path,
        threads: int,
    ) -> None:
        self.process = subprocess.Popen(
            [
                str(helper),
                "--serve",
                "--manifest",
                str(manifest),
                "--index",
                str(index),
                "--runtime",
                str(runtime),
                "--threads",
                str(threads),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

    def search(self, request: dict) -> dict:
        started = time.perf_counter()
        self.process.stdin.write(json.dumps(request) + "\n")
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        wall_ms = (time.perf_counter() - started) * 1000
        if not line:
            raise RuntimeError(self.process.stderr.read())
        response = json.loads(line)
        if response.get("error"):
            raise RuntimeError(response["error"])
        response["wallMs"] = wall_ms
        return response

    def close(self) -> None:
        self.process.stdin.close()
        self.process.wait(timeout=30)
        if self.process.returncode:
            raise RuntimeError(self.process.stderr.read())


def run_model(
    helper: Path,
    runtime: Path,
    config: dict,
    prompts: list[str],
    threads: int,
) -> dict:
    header = index_header(config["index"])
    searcher = Searcher(
        helper, runtime, config["manifest"], config["index"], threads
    )
    rows = []
    try:
        generation = 0
        for prompt in prompts:
            generation += 1
            raw = searcher.search(
                {"generation": generation, "query": prompt, "topK": 10}
            )
            generation += 1
            product = searcher.search(
                {
                    "generation": generation,
                    "query": prompt,
                    "topK": header["records"],
                    "scoreWindow": 0.022,
                    "minScoreProminence": 0.015,
                    "maxResults": 256,
                    "minResults": 0,
                }
            )
            rows.append(
                {
                    "query": prompt,
                    "raw": {
                        "queryMs": raw["queryMs"],
                        "searchMs": raw["searchMs"],
                        "wallMs": raw["wallMs"],
                        "matches": raw["matches"],
                    },
                    "product": {
                        "queryMs": product["queryMs"],
                        "searchMs": product["searchMs"],
                        "wallMs": product["wallMs"],
                        "count": len(product["matches"]),
                        "matches": product["matches"][:10],
                    },
                }
            )
        pss_mib = proportional_mib(searcher.process.pid)
    finally:
        searcher.close()
    warm = rows[1:]
    query_ms = [row["raw"]["queryMs"] for row in warm]
    search_ms = [row["raw"]["searchMs"] for row in warm]
    wall_ms = [row["raw"]["wallMs"] for row in warm]
    counts = [row["product"]["count"] for row in rows]
    return {
        "manifest": str(config["manifest"]),
        "index": str(config["index"]),
        "modelBytes": model_bytes(config["manifest"]),
        "indexHeader": header,
        "pssMiB": pss_mib,
        "coldEndToEndMs": rows[0]["raw"]["wallMs"],
        "warmQueryMedianMs": statistics.median(query_ms),
        "warmQueryP95Ms": percentile(query_ms, 0.95),
        "warmSearchMedianMs": statistics.median(search_ms),
        "warmWallMedianMs": statistics.median(wall_ms),
        "warmWallP95Ms": percentile(wall_ms, 0.95),
        "productZeroResults": sum(count == 0 for count in counts),
        "productResultMedian": statistics.median(counts),
        "productResultP95": percentile(counts, 0.95),
        "details": rows,
    }


def overlap(left: list[dict], right: list[dict], count: int) -> float:
    left_keys = [item["key"] for item in left[:count]]
    right_keys = [item["key"] for item in right[:count]]
    return len(set(left_keys) & set(right_keys)) / count


def comparisons(models: dict) -> dict:
    names = list(models)
    result = {}
    for left_index, left_name in enumerate(names):
        for right_name in names[left_index + 1 :]:
            left = models[left_name]["details"]
            right = models[right_name]["details"]
            result[f"{left_name}:{right_name}"] = {
                f"top{count}": statistics.fmean(
                    overlap(a["raw"]["matches"], b["raw"]["matches"], count)
                    for a, b in zip(left, right, strict=True)
                )
                for count in (1, 3, 10)
            }
    return result


def parse_model(value: str) -> tuple[str, dict]:
    name, paths = value.split("=", 1)
    manifest, index = paths.split(",", 1)
    return name, {"manifest": Path(manifest).resolve(), "index": Path(index).resolve()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--helper", required=True, type=Path)
    parser.add_argument("--runtime", required=True, type=Path)
    parser.add_argument("--evaluation", required=True, type=Path)
    parser.add_argument("--model", action="append", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    prompts = json.loads(args.evaluation.read_text(encoding="utf-8"))["queries"]
    configs = dict(parse_model(value) for value in args.model)
    models = {
        name: run_model(args.helper, args.runtime, config, prompts, args.threads)
        for name, config in configs.items()
    }
    report = {
        "format": 1,
        "queries": len(prompts),
        "models": models,
        "agreement": comparisons(models),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "queries": len(prompts),
                "models": {
                    name: {key: value for key, value in model.items() if key != "details"}
                    for name, model in models.items()
                },
                "agreement": report["agreement"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
