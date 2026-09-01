#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import resource
import statistics
import struct
import time
from pathlib import Path
from typing import Callable

import numpy as np
import onnxruntime
from tokenizers import Tokenizer


def load_sidx(path: Path) -> tuple[str, list[str], np.ndarray]:
    with path.open("rb") as file:
        if file.read(8) != b"SKWDSEM3":
            raise RuntimeError("unsupported semantic index")
        dimensions, model_length = struct.unpack("<II", file.read(8))
        model = file.read(model_length).decode("utf-8")
        file.read(8)
        count = struct.unpack("<Q", file.read(8))[0]
        keys = []
        embeddings = np.empty((count, dimensions), dtype=np.float32)
        for index in range(count):
            key_length = struct.unpack("<I", file.read(4))[0]
            keys.append(file.read(key_length).decode("utf-8"))
            file.read(8)
            values = file.read(dimensions * 4)
            if len(values) != dimensions * 4:
                raise RuntimeError("truncated semantic index")
            embeddings[index] = np.frombuffer(values, dtype="<f4")
    return model, keys, embeddings


def load_npz(path: Path) -> tuple[str, list[str], np.ndarray]:
    data = np.load(path)
    model = str(data["architecture"])
    return model, data["keys"].tolist(), data["embeddings"].astype(np.float32)


def clip_tokens(tokenizer: Tokenizer, query: str, context: int) -> dict[str, np.ndarray]:
    values = tokenizer.encode(query).ids
    if len(values) > context:
        values = values[:context]
        values[-1] = 49407
    values.extend([0] * (context - len(values)))
    return {"tokens": np.asarray([values], dtype=np.int64)}


def siglip_tokens(tokenizer: Tokenizer, query: str, context: int) -> dict[str, np.ndarray]:
    encoding = tokenizer.encode(query)
    values = encoding.ids[:context]
    if len(values) == context:
        values[-1] = 1
    mask = [1] * len(values)
    values.extend([0] * (context - len(values)))
    mask.extend([0] * (context - len(mask)))
    return {
        "input_ids": np.asarray([values], dtype=np.int64),
        "attention_mask": np.asarray([mask], dtype=np.int64),
    }


def hf_clip_tokens(tokenizer: Tokenizer, query: str, context: int) -> dict[str, np.ndarray]:
    values = tokenizer.encode(query).ids[:context]
    if len(values) == context:
        values[-1] = 49407
    mask = [1] * len(values)
    values.extend([0] * (context - len(values)))
    mask.extend([0] * (context - len(mask)))
    return {
        "input_ids": np.asarray([values], dtype=np.int64),
        "attention_mask": np.asarray([mask], dtype=np.int64),
    }


def session_options(threads: int) -> onnxruntime.SessionOptions:
    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    return options


def load_encoder(args: argparse.Namespace) -> tuple[str, Callable[[str], np.ndarray], int]:
    started = time.perf_counter()
    if args.manifest:
        manifest_path = args.manifest.resolve()
        root = manifest_path.parent
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        model_path = root / manifest["text"]["model"]
        tokenizer_path = root / manifest["text"]["tokenizer"]
        context = int(manifest["contextLength"])
        inputs = lambda tokenizer, query: clip_tokens(tokenizer, query, context)
        output = "text_embedding"
        name = f'{manifest["id"]}:{manifest.get("version", "unversioned")}'
    else:
        config = json.loads(args.model_config.read_text(encoding="utf-8"))
        root = args.model_config.resolve().parent
        model_path = root / args.text_model if args.text_model else root / config["text"]["path"]
        tokenizer_path = args.tokenizer.resolve()
        context = int(config["text"]["length"])
        if config["architecture"].startswith("siglip"):
            inputs = lambda tokenizer, query: siglip_tokens(tokenizer, query, context)
        else:
            inputs = lambda tokenizer, query: hf_clip_tokens(tokenizer, query, context)
        output = config["text"]["output"]
        name = config["architecture"] + (f":{model_path.stem}" if args.text_model else "")
    session = onnxruntime.InferenceSession(
        str(model_path),
        sess_options=session_options(args.threads),
        providers=["CPUExecutionProvider"],
    )
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    load_milliseconds = round((time.perf_counter() - started) * 1000, 3)

    def encode(query: str) -> np.ndarray:
        embedding = session.run([output], inputs(tokenizer, query))[0][0].astype(np.float32)
        norm = np.linalg.norm(embedding)
        return embedding / norm if norm else embedding

    return name, encode, load_milliseconds


def matches_required(tags: set[str], query: dict) -> bool:
    if not all(tag in tags for tag in query.get("requiredAll", [])):
        return False
    return all(tags.intersection(group) for group in query.get("requiredAny", []))


def is_positive(tags: set[str], query: dict) -> bool:
    return matches_required(tags, query) and not tags.intersection(query.get("forbiddenAny", []))


def average_precision(relevant: list[bool], total_positives: int) -> float:
    if total_positives == 0:
        return 0.0
    hits = 0
    score = 0.0
    for rank, positive in enumerate(relevant, start=1):
        if positive:
            hits += 1
            score += hits / rank
    return score / min(total_positives, len(relevant))


def ndcg(relevant: list[bool], total_positives: int) -> float:
    gain = sum(value / math.log2(rank + 1) for rank, value in enumerate(relevant, start=1))
    ideal_count = min(total_positives, len(relevant))
    ideal = sum(1 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))
    return gain / ideal if ideal else 0.0


def metrics(order: np.ndarray, tags: list[set[str]], query: dict, top_k: int) -> dict:
    positives = [is_positive(value, query) for value in tags]
    selected = order[:top_k].tolist()
    relevant = [positives[index] for index in selected]
    forbidden = set(query.get("forbiddenAny", []))
    forbidden_hits = sum(bool(tags[index].intersection(forbidden)) for index in selected)
    required_hits = sum(matches_required(tags[index], query) for index in selected)
    first = next((rank for rank, value in enumerate(relevant, start=1) if value), None)
    positive_count = sum(positives)
    hits = sum(relevant)
    return {
        "positiveCount": positive_count,
        "precisionAtK": hits / top_k,
        "recallAtK": hits / positive_count if positive_count else 0.0,
        "averagePrecisionAtK": average_precision(relevant, positive_count),
        "ndcgAtK": ndcg(relevant, positive_count),
        "reciprocalRank": 1 / first if first else 0.0,
        "requiredAtK": required_hits / top_k,
        "forbiddenAtK": forbidden_hits / top_k,
    }


def exclude_forbidden(
    order: np.ndarray, tags: list[set[str]], query: dict
) -> np.ndarray:
    forbidden = set(query.get("forbiddenAny", []))
    if not forbidden:
        return order
    return np.asarray(
        [index for index in order if not tags[index].intersection(forbidden)],
        dtype=order.dtype,
    )


def aggregate(results: list[dict]) -> dict:
    fields = [
        "precisionAtK",
        "recallAtK",
        "averagePrecisionAtK",
        "ndcgAtK",
        "reciprocalRank",
        "requiredAtK",
        "forbiddenAtK",
    ]
    return ({
        field: statistics.fmean(result[field] for result in results)
        for field in fields
    } if results else {})


def aggregate_by_kind(results: list[tuple[str, dict]]) -> dict:
    kinds = sorted({kind for kind, _ in results})
    return {
        kind: aggregate([result for candidate, result in results if candidate == kind])
        for kind in kinds
    }


def percentile(values: list[float], proportion: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(len(ordered) * proportion) - 1)
    return ordered[index]


def benchmark_queries(
    embeddings: np.ndarray,
    keys: list[str],
    tags: list[set[str]],
    queries: list[dict],
    encode: Callable[[str], np.ndarray],
    top_k: int,
    negative_weight: float,
) -> dict:
    raw_results: list[tuple[str, dict]] = []
    filtered_results: list[tuple[str, dict]] = []
    composed_results: list[tuple[str, dict]] = []
    composed_filtered_results: list[tuple[str, dict]] = []
    raw_latencies = []
    composed_latencies = []
    details = []
    for query in queries:
        started = time.perf_counter()
        raw_embedding = encode(query["text"])
        raw_latencies.append((time.perf_counter() - started) * 1000)
        raw_order = np.argsort(embeddings @ raw_embedding)[::-1]
        raw = metrics(raw_order, tags, query, top_k)
        filtered_order = exclude_forbidden(raw_order, tags, query)
        filtered = metrics(filtered_order, tags, query, top_k)
        kind = query.get("kind", "negation" if query.get("forbiddenAny") else "other")
        if raw["positiveCount"]:
            raw_results.append((kind, raw))
            filtered_results.append((kind, filtered))
        detail = {
            "id": query["id"],
            "text": query["text"],
            "kind": kind,
            "raw": raw,
            "rawTopKeys": [keys[index] for index in raw_order[:top_k]],
            "filtered": filtered,
            "filteredTopKeys": [keys[index] for index in filtered_order[:top_k]],
        }
        if "positiveText" in query and "negativeText" in query:
            started = time.perf_counter()
            positive = encode(query["positiveText"])
            negative = encode(query["negativeText"])
            composed_embedding = positive - negative_weight * negative
            composed_embedding /= np.linalg.norm(composed_embedding)
            composed_latencies.append((time.perf_counter() - started) * 1000)
            composed_order = np.argsort(embeddings @ composed_embedding)[::-1]
            composed = metrics(composed_order, tags, query, top_k)
            composed_filtered_order = exclude_forbidden(composed_order, tags, query)
            composed_filtered = metrics(composed_filtered_order, tags, query, top_k)
            if composed["positiveCount"]:
                composed_results.append((kind, composed))
                composed_filtered_results.append((kind, composed_filtered))
            detail.update(
                {
                    "positiveText": query["positiveText"],
                    "negativeText": query["negativeText"],
                    "composed": composed,
                    "composedTopKeys": [keys[index] for index in composed_order[:top_k]],
                    "composedFiltered": composed_filtered,
                    "composedFilteredTopKeys": [
                        keys[index] for index in composed_filtered_order[:top_k]
                    ],
                }
            )
        details.append(detail)
    return {
        "queries": len(details),
        "evaluatedRawQueries": len(raw_results),
        "evaluatedComposedQueries": len(composed_results),
        "rawLatencyMilliseconds": {
            "mean": statistics.fmean(raw_latencies),
            "p50": statistics.median(raw_latencies),
            "p95": percentile(raw_latencies, 0.95),
        },
        "composedLatencyMilliseconds": {
            "mean": statistics.fmean(composed_latencies) if composed_latencies else 0.0,
            "p50": statistics.median(composed_latencies) if composed_latencies else 0.0,
            "p95": percentile(composed_latencies, 0.95),
        },
        "raw": aggregate([result for _, result in raw_results]),
        "rawByKind": aggregate_by_kind(raw_results),
        "filtered": aggregate([result for _, result in filtered_results]),
        "filteredByKind": aggregate_by_kind(filtered_results),
        "composed": aggregate([result for _, result in composed_results]),
        "composedByKind": aggregate_by_kind(composed_results),
        "composedFiltered": aggregate(
            [result for _, result in composed_filtered_results]
        ),
        "composedFilteredByKind": aggregate_by_kind(composed_filtered_results),
        "details": details,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    encoder = parser.add_mutually_exclusive_group(required=True)
    encoder.add_argument("--manifest", type=Path)
    encoder.add_argument("--model-config", type=Path)
    parser.add_argument("--text-model", type=Path)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--negative-weight", type=float, default=0.5)
    args = parser.parse_args()
    if args.model_config and not args.tokenizer:
        parser.error("--tokenizer is required with --model-config")
    if args.threads < 1 or args.top_k < 1 or args.negative_weight < 0:
        parser.error("threads and top-k must be positive, negative-weight cannot be negative")

    sample = json.loads(args.sample.read_text(encoding="utf-8"))
    query_data = json.loads(args.queries.read_text(encoding="utf-8"))
    if args.index.suffix == ".npz":
        index_model, index_keys, index_embeddings = load_npz(args.index)
    else:
        index_model, index_keys, index_embeddings = load_sidx(args.index)
    index_lookup = {key: index for index, key in enumerate(index_keys)}
    items = [item for item in sample["items"] if item["key"] in index_lookup]
    keys = [item["key"] for item in items]
    tags = [set(item["tags"]) for item in items]
    embeddings = np.stack([index_embeddings[index_lookup[key]] for key in keys])
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = np.divide(embeddings, norms, out=np.zeros_like(embeddings), where=norms != 0)
    model, encode, load_milliseconds = load_encoder(args)
    encode("wallpaper")
    benchmark = benchmark_queries(
        embeddings,
        keys,
        tags,
        query_data["queries"],
        encode,
        args.top_k,
        args.negative_weight,
    )
    report = {
        "format": 1,
        "model": model,
        "indexModel": index_model,
        "items": len(items),
        "topK": args.top_k,
        "negativeWeight": args.negative_weight,
        "threads": args.threads,
        "loadMilliseconds": load_milliseconds,
        "peakResidentMiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        **benchmark,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
