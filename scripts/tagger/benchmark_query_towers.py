#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import time
from pathlib import Path

import numpy as np


def proportional_mib(pid: int) -> float:
    for line in Path(f"/proc/{pid}/smaps_rollup").read_text(encoding="utf-8").splitlines():
        if line.startswith("Pss:"):
            return int(line.split()[1]) / 1024
    raise RuntimeError(f"Pss is missing for process {pid}")


class Searcher:
    def __init__(
        self,
        helper: Path,
        manifest: Path,
        index: Path,
        runtime: Path,
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

    def search(self, query: str, generation: int, top_k: int) -> dict:
        if self.process.stdin is None or self.process.stdout is None:
            raise RuntimeError("semantic helper pipes are unavailable")
        started = time.perf_counter()
        self.process.stdin.write(
            json.dumps({"generation": generation, "query": query, "topK": top_k})
            + "\n"
        )
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        elapsed = (time.perf_counter() - started) * 1000
        if not line:
            error = self.process.stderr.read() if self.process.stderr else ""
            raise RuntimeError(f"semantic helper exited: {error}")
        response = json.loads(line)
        if response.get("error"):
            raise RuntimeError(response["error"])
        response["wallMs"] = elapsed
        return response

    def close(self) -> None:
        self.process.terminate()
        self.process.wait(timeout=10)


def run(
    helper: Path,
    manifest: Path,
    index: Path,
    runtime: Path,
    threads: int,
    prompts: list[str],
    top_k: int,
) -> tuple[dict, list[list[str]]]:
    searcher = Searcher(helper, manifest, index, runtime, threads)
    responses = []
    try:
        for generation, prompt in enumerate(prompts, start=1):
            responses.append(searcher.search(prompt, generation, top_k))
        pss_mib = proportional_mib(searcher.process.pid)
    finally:
        searcher.close()
    warm = responses[1:]
    return (
        {
            "manifest": str(manifest),
            "pssMiB": pss_mib,
            "coldEndToEndMs": responses[0]["wallMs"],
            "warmQueryMedianMs": statistics.median(item["queryMs"] for item in warm),
            "warmQueryP95Ms": float(
                np.quantile([item["queryMs"] for item in warm], 0.95)
            ),
            "warmWallMedianMs": statistics.median(item["wallMs"] for item in warm),
            "warmWallP95Ms": float(
                np.quantile([item["wallMs"] for item in warm], 0.95)
            ),
        },
        [[match["key"] for match in item["matches"]] for item in responses],
    )


def overlap(expected: list[list[str]], actual: list[list[str]], count: int) -> float:
    return statistics.mean(
        len(set(left[:count]) & set(right[:count])) / min(count, len(left), len(right))
        for left, right in zip(expected, actual, strict=True)
    )


def recall(
    expected: list[list[str]], actual: list[list[str]], expected_count: int, actual_count: int
) -> float:
    return statistics.mean(
        len(set(left[:expected_count]) & set(right[:actual_count]))
        / min(expected_count, len(left))
        for left, right in zip(expected, actual, strict=True)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--helper", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--baseline-manifest", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    evaluation = json.loads(args.evaluation.read_text(encoding="utf-8"))
    prompts = [
        item if isinstance(item, str) else item["text"]
        for item in evaluation["queries"]
    ]
    baseline, baseline_rankings = run(
        args.helper,
        args.baseline_manifest,
        args.index,
        args.runtime,
        args.threads,
        prompts,
        args.top_k,
    )
    candidate, candidate_rankings = run(
        args.helper,
        args.candidate_manifest,
        args.index,
        args.runtime,
        args.threads,
        prompts,
        args.top_k,
    )
    narrow = min(3, args.top_k)
    report = {
        "format": 1,
        "queries": len(prompts),
        "topK": args.top_k,
        "baseline": baseline,
        "candidate": candidate,
        "candidatePssSavedMiB": baseline["pssMiB"] - candidate["pssMiB"],
        "retrieval": {
            "top1Agreement": overlap(baseline_rankings, candidate_rankings, 1),
            f"top{narrow}ExactOverlap": overlap(
                baseline_rankings, candidate_rankings, narrow
            ),
            "teacherTop3RecallAt5": recall(
                baseline_rankings, candidate_rankings, min(3, args.top_k), min(5, args.top_k)
            ),
            f"top{args.top_k}ExactOverlap": overlap(
                baseline_rankings, candidate_rankings, args.top_k
            ),
        },
    }
    encoded = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
