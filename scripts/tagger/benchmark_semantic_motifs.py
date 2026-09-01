#!/usr/bin/env python3

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import statistics
import subprocess
import tempfile
import time
from pathlib import Path


def deterministic_pool(items: list[dict], anchor_keys: set[str], size: int) -> list[dict]:
    if size < len(anchor_keys):
        raise ValueError("pool size cannot be smaller than the anchor count")
    by_key = {item["key"]: item for item in items}
    missing = sorted(anchor_keys - by_key.keys())
    if missing:
        raise ValueError(f"anchors missing from corpus metadata: {', '.join(missing)}")
    distractors = [item for item in items if item["key"] not in anchor_keys]
    distractors.sort(
        key=lambda item: hashlib.sha256(item["key"].encode("utf-8")).digest()
    )
    return [by_key[key] for key in sorted(anchor_keys)] + distractors[: size - len(anchor_keys)]


def model_bytes(manifest_path: Path) -> int:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    total = 0
    for tower in ("image", "text"):
        model = manifest_path.parent / manifest[tower]["model"]
        total += model.stat().st_size
        external = Path(f"{model}.data")
        if external.exists():
            total += external.stat().st_size
    token_table = manifest["text"].get("tokenEmbeddings", {}).get("table")
    if token_table:
        total += (manifest_path.parent / token_table).stat().st_size
    return total


def build_index(
    helper: Path,
    runtime: Path,
    manifest: Path,
    index: Path,
    entries: list[dict],
    threads: int,
) -> tuple[float, str, float]:
    request = {
        "fingerprint": 1,
        "entries": [
            {
                "key": item["key"],
                "path": str(item["absolutePath"]),
                "fingerprint": 1,
            }
            for item in entries
        ],
    }
    command = [
        str(helper),
        "--manifest",
        str(manifest),
        "--index",
        str(index),
        "--runtime",
        str(runtime),
        "--threads",
        str(threads),
        "--build-index",
    ]
    started = time.perf_counter()
    with tempfile.TemporaryFile(mode="w+") as stdin, tempfile.TemporaryFile(
        mode="w+"
    ) as stderr:
        stdin.write(json.dumps(request))
        stdin.seek(0)
        process = subprocess.Popen(
            command,
            stdin=stdin,
            stdout=subprocess.DEVNULL,
            stderr=stderr,
        )
        peak_mib = 0.0
        while process.poll() is None:
            try:
                peak_mib = max(peak_mib, process_memory_mib(process.pid))
            except (FileNotFoundError, RuntimeError):
                pass
            time.sleep(0.01)
        stderr.seek(0)
        build_log = stderr.read().strip()
        if process.returncode:
            raise RuntimeError(build_log or f"index builder exited {process.returncode}")
    return time.perf_counter() - started, build_log, peak_mib


def process_memory_mib(pid: int, field: str = "VmRSS:") -> float:
    for line in Path(f"/proc/{pid}/status").read_text(encoding="utf-8").splitlines():
        if line.startswith(field):
            return int(line.split()[1]) / 1024
    raise RuntimeError(f"{field} missing from process status")


def query_index(
    helper: Path,
    runtime: Path,
    manifest: Path,
    index: Path,
    anchors: list[dict],
    pool_size: int,
    threads: int,
) -> tuple[list[dict], float, float, float]:
    command = [
        str(helper),
        "--manifest",
        str(manifest),
        "--index",
        str(index),
        "--runtime",
        str(runtime),
        "--threads",
        str(threads),
        "--serve",
    ]
    started = time.perf_counter()
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert process.stdin is not None and process.stdout is not None
    results = []
    first_result_milliseconds = None
    try:
        generation = 0
        for anchor in anchors:
            for query in anchor["queries"]:
                generation += 1
                request = {
                    "generation": generation,
                    "query": query,
                    "topK": pool_size,
                }
                process.stdin.write(json.dumps(request) + "\n")
                process.stdin.flush()
                response = json.loads(process.stdout.readline())
                if first_result_milliseconds is None:
                    first_result_milliseconds = (time.perf_counter() - started) * 1000
                if response.get("error"):
                    raise RuntimeError(response["error"])
                keys = [match["key"] for match in response["matches"]]
                rank = keys.index(anchor["key"]) + 1
                results.append(
                    {
                        "anchor": anchor["id"],
                        "query": query,
                        "rank": rank,
                        "reciprocalRank": 1 / rank,
                        "queryMilliseconds": response["queryMs"],
                        "searchMilliseconds": response["searchMs"],
                        "topFive": keys[:5],
                    }
                )
        loaded_mib = process_memory_mib(process.pid)
        startup_seconds = time.perf_counter() - started
    finally:
        process.stdin.close()
        process.wait(timeout=10)
        if process.returncode:
            assert process.stderr is not None
            raise RuntimeError(process.stderr.read())
    assert first_result_milliseconds is not None
    return results, first_result_milliseconds, startup_seconds, loaded_mib


def summarize(results: list[dict]) -> dict:
    ranks = [result["rank"] for result in results]
    return {
        "queries": len(results),
        "hitAt1": sum(rank <= 1 for rank in ranks) / len(ranks),
        "hitAt5": sum(rank <= 5 for rank in ranks) / len(ranks),
        "hitAt10": sum(rank <= 10 for rank in ranks) / len(ranks),
        "meanReciprocalRank": statistics.fmean(1 / rank for rank in ranks),
        "medianRank": statistics.median(ranks),
        "worstRank": max(ranks),
        "meanQueryMilliseconds": statistics.fmean(
            result["queryMilliseconds"] for result in results
        ),
    }


def parse_model(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("model must be NAME=MANIFEST")
    name, path = value.split("=", 1)
    if not name or not path:
        raise argparse.ArgumentTypeError("model must be NAME=MANIFEST")
    return name, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", type=parse_model, required=True)
    parser.add_argument("--corpus", type=Path, default=Path("target/wallpaper-corpus"))
    parser.add_argument(
        "--anchors",
        type=Path,
        default=Path("scripts/tagger/semantic-motif-benchmark-v1.json"),
    )
    parser.add_argument("--helper", type=Path, default=Path("target/release/skwd-lens"))
    parser.add_argument(
        "--runtime",
        type=Path,
        default=Path("target/release/lens/runtime/libonnxruntime.so.1.27.0"),
    )
    parser.add_argument("--pool-size", type=int, default=200)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--index-dir",
        type=Path,
        help="Keep generated indexes here for embedding-level comparisons.",
    )
    args = parser.parse_args()

    anchors = json.loads(args.anchors.read_text(encoding="utf-8"))["anchors"]
    items = [
        json.loads(line)
        for line in (args.corpus / "items.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    pool = deterministic_pool(items, {anchor["key"] for anchor in anchors}, args.pool_size)
    for item in pool:
        item["absolutePath"] = (args.corpus / item["path"]).resolve()
        if not item["absolutePath"].is_file():
            raise FileNotFoundError(item["absolutePath"])

    reports = []
    index_context = (
        contextlib.nullcontext(args.index_dir)
        if args.index_dir
        else tempfile.TemporaryDirectory(prefix="skwd-semantic-motifs-")
    )
    with index_context as temporary:
        root = Path(temporary)
        root.mkdir(parents=True, exist_ok=True)
        for name, manifest_value in args.model:
            manifest = manifest_value.resolve()
            index = root / f"{name}.sidx"
            build_seconds, build_log, build_peak_mib = build_index(
                args.helper.resolve(),
                args.runtime.resolve(),
                manifest,
                index,
                pool,
                args.threads,
            )
            results, first_result_ms, query_session_seconds, query_resident_mib = query_index(
                args.helper.resolve(),
                args.runtime.resolve(),
                manifest,
                index,
                anchors,
                args.pool_size,
                args.threads,
            )
            reports.append(
                {
                    "name": name,
                    "manifest": str(manifest),
                    "modelMiB": model_bytes(manifest) / 1024 / 1024,
                    "buildSeconds": build_seconds,
                    "buildPeakResidentMiB": build_peak_mib,
                    "buildLog": build_log,
                    "firstResultMilliseconds": first_result_ms,
                    "querySessionSeconds": query_session_seconds,
                    "queryResidentMiB": query_resident_mib,
                    **summarize(results),
                    "details": results,
                }
            )

    report = {
        "format": 1,
        "poolSize": args.pool_size,
        "anchors": [anchor["id"] for anchor in anchors],
        "threads": args.threads,
        "models": reports,
    }
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
        print(args.output)
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
