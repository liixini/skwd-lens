#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import statistics
from pathlib import Path

from benchmark_model_lab import Searcher, parse_model


TAGS = [
    "woman",
    "anime",
    "ocean",
    "mountain",
    "peaceful",
    "eerie",
    "dreamy",
    "dark",
    "photo",
    "3d",
    "bright",
    "cold",
    "cozy",
    "colorful",
    "city",
    "building",
    "pixel-art",
    "cartoon",
    "oil-painting",
    "flower",
    "minimalist",
    "sketch",
    "abstract",
    "watercolor",
    "comic",
    "illustration",
    "forest",
    "car",
    "cat",
    "fox",
    "dragon",
    "bedroom",
    "astronaut",
    "dock",
]


def labels(database: Path) -> dict[str, set[str]]:
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    relevant = {tag: set() for tag in TAGS}
    try:
        rows = connection.execute(
            "SELECT key, generated_tags FROM meta WHERE generated_tags IS NOT NULL"
        )
        for key, raw in rows:
            item_tags = {tag.strip().lower() for tag in raw.split(",") if tag.strip()}
            for tag in TAGS:
                if tag in item_tags:
                    relevant[tag].add(key)
    finally:
        connection.close()
    return relevant


def metrics(keys: list[str], relevant: set[str], count: int = 10) -> dict:
    ranked = keys[:count]
    hits = [key in relevant for key in ranked]
    precision = []
    found = 0
    for rank, hit in enumerate(hits, start=1):
        if hit:
            found += 1
            precision.append(found / rank)
    average_precision = sum(precision) / min(len(relevant), count)
    dcg = sum((1.0 / math.log2(rank + 1)) for rank, hit in enumerate(hits, 1) if hit)
    ideal = sum(
        1.0 / math.log2(rank + 1) for rank in range(1, min(len(relevant), count) + 1)
    )
    return {
        "precisionAt1": float(hits[0]),
        "hitAt5": float(any(hits[:5])),
        "precisionAt5": sum(hits[:5]) / 5,
        "averagePrecisionAt10": average_precision,
        "ndcgAt10": dcg / ideal if ideal else 0.0,
        "relevant": len(relevant),
        "top10": keys[:10],
    }


def run_model(
    helper: Path,
    runtime: Path,
    config: dict,
    relevant: dict[str, set[str]],
    threads: int,
) -> dict:
    searcher = Searcher(
        helper, runtime, config["manifest"], config["index"], threads
    )
    rows = []
    try:
        for generation, tag in enumerate(TAGS, start=1):
            response = searcher.search(
                {"generation": generation, "query": tag.replace("-", " "), "topK": 50}
            )
            keys = [match["key"] for match in response["matches"]]
            rows.append({"query": tag, **metrics(keys, relevant[tag])})
    finally:
        searcher.close()
    fields = (
        "precisionAt1",
        "hitAt5",
        "precisionAt5",
        "averagePrecisionAt10",
        "ndcgAt10",
    )
    return {
        "summary": {
            field: statistics.fmean(row[field] for row in rows) for field in fields
        },
        "details": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--helper", required=True, type=Path)
    parser.add_argument("--runtime", required=True, type=Path)
    parser.add_argument("--database", required=True, type=Path)
    parser.add_argument("--model", action="append", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    relevant = labels(args.database.resolve())
    configs = dict(parse_model(value) for value in args.model)
    models = {
        name: run_model(args.helper, args.runtime, config, relevant, args.threads)
        for name, config in configs.items()
    }
    report = {
        "format": 1,
        "labels": "legacy generated_tags; approximate visual relevance proxy",
        "queries": len(TAGS),
        "models": models,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {name: value["summary"] for name, value in models.items()}, indent=2
        )
    )


if __name__ == "__main__":
    main()
