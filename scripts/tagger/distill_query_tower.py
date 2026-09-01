#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import random
import struct
import subprocess
from pathlib import Path

import numpy as np
import onnxruntime
import sentencepiece
from tokenizers import Tokenizer


class Encoder:
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

    def encode(self, query: str, generation: int) -> np.ndarray:
        if self.process.stdin is None or self.process.stdout is None:
            raise RuntimeError("semantic helper pipes are unavailable")
        request = {
            "generation": generation,
            "query": query,
            "topK": 1,
            "embeddingOnly": True,
        }
        self.process.stdin.write(json.dumps(request) + "\n")
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            error = self.process.stderr.read() if self.process.stderr else ""
            raise RuntimeError(f"semantic helper exited: {error}")
        response = json.loads(line)
        if response.get("error"):
            raise RuntimeError(response["error"])
        return np.asarray(response["embedding"], dtype=np.float32)

    def close(self) -> None:
        self.process.terminate()
        self.process.wait(timeout=10)


def prompt_corpus(path: Path, limit: int, seed: int) -> list[str]:
    value = json.loads(path.read_text(encoding="utf-8"))
    rng = random.Random(seed)
    prompts = set(value["fixed"])
    while len(prompts) < limit * 2:
        selected = {
            "subject": rng.choice(value["subjects"]),
            "setting": rng.choice(value["settings"]),
            "style": rng.choice(value["styles"]),
            "quality": rng.choice(value["qualities"]),
            "color": rng.choice(value["colors"]),
        }
        prompts.add(rng.choice(value["templates"]).format(**selected))
    result = sorted(prompts)
    rng.shuffle(result)
    return result[:limit]


def evaluation_corpus(path: Path) -> tuple[list[str], dict[str, float]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    queries = [
        item if isinstance(item, str) else item["text"]
        for item in value["queries"]
    ]
    if not queries or len(set(queries)) != len(queries):
        raise RuntimeError("evaluation queries must be non-empty and unique")
    return queries, value["acceptance"]


def encode_pairs(
    prompts: list[str], student: Encoder, teacher: Encoder
) -> tuple[np.ndarray, np.ndarray]:
    source = []
    target = []
    for generation, prompt in enumerate(prompts, start=1):
        source.append(student.encode(prompt, generation))
        target.append(teacher.encode(prompt, generation))
        if generation % 250 == 0:
            print(f"encoded {generation}/{len(prompts)}", flush=True)
    return np.stack(source), np.stack(target)


def encode_pairs_lens(
    args: argparse.Namespace, prompts: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    required = {
        "helper": args.helper,
        "runtime": args.runtime,
        "student manifest": args.student_manifest,
        "student index": args.student_index,
        "teacher manifest": args.teacher_manifest,
        "teacher index": args.teacher_index,
    }
    missing = [name for name, path in required.items() if path is None]
    if missing:
        raise RuntimeError(f"Lens encoder requires {', '.join(missing)}")
    student = Encoder(
        args.helper,
        args.student_manifest,
        args.student_index,
        args.runtime,
        args.threads,
    )
    teacher = Encoder(
        args.helper,
        args.teacher_manifest,
        args.teacher_index,
        args.runtime,
        args.threads,
    )
    try:
        return encode_pairs(prompts, student, teacher)
    finally:
        student.close()
        teacher.close()


def batch_encoder(
    model: Path,
    tokenizer_path: Path,
    prompts: list[str],
    context_length: int,
    pad_token: int,
    output: str,
    threads: int,
    batch_size: int,
    attention_mask: bool,
) -> np.ndarray:
    if tokenizer_path.suffix == ".model":
        tokenizer = sentencepiece.SentencePieceProcessor(model_file=str(tokenizer_path))

        def encode(values: list[str]) -> tuple[np.ndarray, np.ndarray]:
            ids = []
            masks = []
            for value in values:
                tokens = tokenizer.encode(value.lower(), out_type=int) + [tokenizer.eos_id()]
                tokens = tokens[:context_length]
                tokens[-1] = tokenizer.eos_id()
                mask = [1] * len(tokens)
                tokens.extend([pad_token] * (context_length - len(tokens)))
                mask.extend([0] * (context_length - len(mask)))
                ids.append(tokens)
                masks.append(mask)
            return np.asarray(ids, dtype=np.int64), np.asarray(masks, dtype=np.int64)

    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        tokenizer.enable_truncation(max_length=context_length)
        tokenizer.enable_padding(length=context_length, pad_id=pad_token)

        def encode(values: list[str]) -> tuple[np.ndarray, np.ndarray]:
            encodings = tokenizer.encode_batch([value.lower() for value in values])
            return (
                np.asarray([value.ids for value in encodings], dtype=np.int64),
                np.asarray(
                    [value.attention_mask for value in encodings], dtype=np.int64
                ),
            )

    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
    session = onnxruntime.InferenceSession(
        str(model), sess_options=options, providers=["CPUExecutionProvider"]
    )
    results = []
    for start in range(0, len(prompts), batch_size):
        input_ids, mask = encode(prompts[start : start + batch_size])
        feed = {"input_ids": input_ids}
        if attention_mask:
            feed["attention_mask"] = mask
        results.append(session.run([output], feed)[0])
        if start and start % 512 == 0:
            print(f"encoded {min(start + batch_size, len(prompts))}/{len(prompts)}", flush=True)
    return normalize(np.concatenate(results))


def encode_pairs_batched(args: argparse.Namespace, prompts: list[str]) -> tuple[np.ndarray, np.ndarray]:
    required = {
        "student model": args.student_model,
        "student tokenizer": args.student_tokenizer,
        "teacher model": args.teacher_model,
        "teacher tokenizer": args.teacher_tokenizer,
    }
    missing = [name for name, path in required.items() if path is None]
    if missing:
        raise RuntimeError(f"batch encoder requires {', '.join(missing)}")
    source = batch_encoder(
        args.student_model,
        args.student_tokenizer,
        prompts,
        77,
        49407,
        "text_embeds",
        args.threads,
        args.batch_size,
        True,
    )
    target = batch_encoder(
        args.teacher_model,
        args.teacher_tokenizer,
        prompts,
        64,
        0,
        "pooler_output",
        args.threads,
        args.batch_size,
        False,
    )
    return source, target


def encode_selected(
    args: argparse.Namespace, prompts: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    if args.batch_size:
        return encode_pairs_batched(args, prompts)
    return encode_pairs_lens(args, prompts)


def normalize(values: np.ndarray) -> np.ndarray:
    lengths = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(lengths, 1e-8)


def train_linear(source: np.ndarray, target: np.ndarray, ridge: float) -> np.ndarray:
    gram = source.T @ source
    gram.flat[:: gram.shape[0] + 1] += ridge
    return np.linalg.solve(gram, source.T @ target).astype(np.float32)


def top_indices(scores: np.ndarray, count: int) -> np.ndarray:
    count = min(count, len(scores))
    selected = np.argpartition(scores, -count)[-count:]
    return selected[np.argsort(scores[selected])[::-1]]


def evaluate_embeddings(predicted: np.ndarray, target: np.ndarray) -> dict:
    predicted = normalize(predicted)
    target = normalize(target)
    cosine = np.sum(predicted * target, axis=1)
    return {
        "embeddingCosineMean": float(np.mean(cosine)),
        "embeddingCosineP05": float(np.quantile(cosine, 0.05)),
        "embeddingCosineMinimum": float(np.min(cosine)),
    }


def evaluate_retrieval(
    predicted: np.ndarray, target: np.ndarray, image_embeddings: np.ndarray
) -> dict:
    predicted = normalize(predicted)
    target = normalize(target)
    images = normalize(image_embeddings)
    top1_agreements = []
    overlaps_3 = []
    recalls_5 = []
    overlaps_10 = []
    recalls_50 = []
    for student_query, teacher_query in zip(predicted, target, strict=True):
        student_1 = top_indices(images @ student_query, 1)
        teacher_1 = top_indices(images @ teacher_query, 1)
        student_3 = set(top_indices(images @ student_query, 3).tolist())
        teacher_3 = set(top_indices(images @ teacher_query, 3).tolist())
        student_5 = set(top_indices(images @ student_query, 5).tolist())
        student_10 = set(top_indices(images @ student_query, 10).tolist())
        student_50 = set(top_indices(images @ student_query, 50).tolist())
        teacher_10 = set(top_indices(images @ teacher_query, 10).tolist())
        top1_agreements.append(bool(student_1[0] == teacher_1[0]))
        overlaps_3.append(len(student_3 & teacher_3) / min(3, len(images)))
        recalls_5.append(len(student_5 & teacher_3) / min(3, len(images)))
        overlaps_10.append(len(student_10 & teacher_10) / min(10, len(images)))
        recalls_50.append(len(student_50 & teacher_10) / min(10, len(images)))
    return {
        "top1Agreement": float(np.mean(top1_agreements)),
        "top3ExactOverlap": float(np.mean(overlaps_3)),
        "teacherTop3RecallAt5": float(np.mean(recalls_5)),
        "top10ExactOverlap": float(np.mean(overlaps_10)),
        "teacherTop10RecallAt50": float(np.mean(recalls_50)),
        "images": len(images),
    }


def acceptance(metrics: dict, thresholds: dict[str, float]) -> dict:
    failures = {
        name: {"actual": float(metrics[name]), "minimum": float(minimum)}
        for name, minimum in thresholds.items()
        if float(metrics[name]) < float(minimum)
    }
    return {"passed": not failures, "thresholds": thresholds, "failures": failures}


def load_index(path: Path) -> np.ndarray:
    with path.open("rb") as source:
        if source.read(8) != b"SKWDSEM3":
            raise RuntimeError(f"unsupported index: {path}")
        dimensions, model_length = struct.unpack("<II", source.read(8))
        source.read(model_length + 8)
        count = struct.unpack("<Q", source.read(8))[0]
        embeddings = np.empty((count, dimensions), dtype=np.float32)
        for row in range(count):
            key_length = struct.unpack("<I", source.read(4))[0]
            source.read(key_length + 8)
            embeddings[row] = np.frombuffer(source.read(dimensions * 4), dtype="<f4")
    return embeddings


def write_projection(path: Path, values: np.ndarray) -> None:
    with path.open("wb") as destination:
        destination.write(b"SKWDPRJ1")
        destination.write(struct.pack("<II", values.shape[0], values.shape[1]))
        destination.write(values.astype("<f4", copy=False).tobytes(order="C"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--helper", type=Path)
    parser.add_argument("--runtime", type=Path)
    parser.add_argument("--student-manifest", type=Path)
    parser.add_argument("--student-index", type=Path)
    parser.add_argument("--teacher-manifest", type=Path)
    parser.add_argument("--teacher-index", type=Path)
    parser.add_argument("--student-model", type=Path)
    parser.add_argument("--student-tokenizer", type=Path)
    parser.add_argument("--teacher-model", type=Path)
    parser.add_argument("--teacher-tokenizer", type=Path)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--evaluation-prompts", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--queries", type=int, default=3000)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--image-index", type=Path)
    parser.add_argument("--pairs", type=Path)
    args = parser.parse_args()
    if args.batch_size not in (0, 1):
        raise RuntimeError("dynamic-int8 distillation requires batch size one")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.pairs:
        pairs = np.load(args.pairs)
        prompts = pairs["prompts"].tolist()
        source = pairs["source"]
        target = pairs["target"]
    else:
        prompts = prompt_corpus(args.prompts, args.queries, args.seed)
    split = max(1, int(len(prompts) * 0.8))
    if args.pairs:
        pass
    else:
        source, target = encode_selected(args, prompts)
    linear = train_linear(source[:split], target[:split], args.ridge)
    prediction = source[split:] @ linear
    result = {
        "format": 1,
        "seed": args.seed,
        "ridge": args.ridge,
        "queries": len(prompts),
        "trainingQueries": split,
        "validationQueries": len(prompts) - split,
        "sourceDimensions": source.shape[1],
        "targetDimensions": target.shape[1],
        "linearBytes": linear.nbytes,
        "encoder": (
            "reused-pairs"
            if args.pairs
            else "onnx-single"
            if args.batch_size
            else "lens"
        ),
        "promptSource": {
            "path": str(args.prompts),
            "sha256": sha256(args.prompts),
        },
        "validation": evaluate_embeddings(prediction, target[split:]),
    }
    if args.image_index:
        result["retrieval"] = evaluate_retrieval(
            prediction, target[split:], load_index(args.image_index)
        )
    if args.evaluation_prompts:
        evaluation_prompts, thresholds = evaluation_corpus(args.evaluation_prompts)
        leaked = sorted(set(prompts).intersection(evaluation_prompts))
        if leaked:
            raise RuntimeError(f"evaluation queries leaked into training: {leaked[:3]}")
        evaluation_source, evaluation_target = encode_selected(args, evaluation_prompts)
        evaluation_prediction = evaluation_source @ linear
        independent = evaluate_embeddings(evaluation_prediction, evaluation_target)
        if args.image_index:
            independent.update(
                evaluate_retrieval(
                    evaluation_prediction,
                    evaluation_target,
                    load_index(args.image_index),
                )
            )
        result["independentEvaluation"] = {
            "path": str(args.evaluation_prompts),
            "sha256": sha256(args.evaluation_prompts),
            "queries": len(evaluation_prompts),
            "metrics": independent,
            "acceptance": acceptance(independent, thresholds),
        }
    if args.student_model and args.teacher_model:
        result["models"] = {
            "studentSha256": sha256(args.student_model),
            "teacherSha256": sha256(args.teacher_model),
        }
    np.savez_compressed(
        args.output / "embedding-pairs.npz",
        prompts=np.asarray(prompts),
        source=source,
        target=target,
    )
    numpy_projection = args.output / "projector-linear.npy"
    binary_projection = args.output / "projector-linear.bin"
    np.save(numpy_projection, linear)
    write_projection(binary_projection, linear)
    result["projection"] = {
        "bytes": binary_projection.stat().st_size,
        "sha256": sha256(binary_projection),
    }
    (args.output / "report.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
