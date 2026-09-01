from __future__ import annotations

import json
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
TAGGER = ROOT / "scripts/tagger"
sys.path.insert(0, str(TAGGER))

from distill_query_tower import (  # noqa: E402
    acceptance,
    evaluate_embeddings,
    evaluation_corpus,
    prompt_corpus,
    train_linear,
    write_projection,
)
from prepare_query_distillation import write_index  # noqa: E402


class QueryDistillationTests(unittest.TestCase):
    def test_prompt_corpus_is_unique_and_deterministic(self) -> None:
        source = TAGGER / "semantic-distillation-prompts-v2.json"

        first = prompt_corpus(source, 250, 19)
        second = prompt_corpus(source, 250, 19)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 250)
        self.assertEqual(len(set(first)), 250)

    def test_evaluation_queries_are_independent_from_generated_training(self) -> None:
        training = prompt_corpus(TAGGER / "semantic-distillation-prompts-v2.json", 6000, 1337)
        queries, thresholds = evaluation_corpus(
            TAGGER / "semantic-query-evaluation-v1.json"
        )

        self.assertFalse(set(training).intersection(queries))
        self.assertGreaterEqual(len(queries), 80)
        self.assertIn("top1Agreement", thresholds)

    def test_acceptance_reports_every_failed_quality_floor(self) -> None:
        result = acceptance(
            {"mean": 0.99, "minimum": 0.8},
            {"mean": 0.98, "minimum": 0.9},
        )

        self.assertFalse(result["passed"])
        self.assertEqual(set(result["failures"]), {"minimum"})

    def test_linear_projection_recovers_known_mapping(self) -> None:
        source = np.asarray(
            [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, -1.0]],
            dtype=np.float32,
        )
        expected = np.asarray([[2.0, 3.0, 4.0], [-1.0, 5.0, 2.0]], dtype=np.float32)
        target = source @ expected

        actual = train_linear(source, target, 0.0)

        np.testing.assert_allclose(actual, expected, atol=1e-5)
        metrics = evaluate_embeddings(source @ actual, target)
        self.assertAlmostEqual(metrics["embeddingCosineMean"], 1.0, places=6)

    def test_projection_and_identity_index_have_canonical_headers(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            projection = root / "projection.bin"
            index = root / "index.sidx"
            values = np.arange(6, dtype=np.float32).reshape(2, 3)

            write_projection(projection, values)
            write_index(index, 3, "teacher@revision")

            self.assertEqual(projection.read_bytes()[:16], b"SKWDPRJ1" + struct.pack("<II", 2, 3))
            encoded = index.read_bytes()
            self.assertEqual(encoded[:8], b"SKWDSEM3")
            dimensions, model_length = struct.unpack("<II", encoded[8:16])
            self.assertEqual(dimensions, 3)
            self.assertEqual(encoded[16 : 16 + model_length], b"teacher@revision")

    def test_staged_pack_keeps_teacher_identity_and_adds_projection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            teacher = root / "teacher"
            student = root / "student"
            output = root / "output"
            teacher.mkdir()
            student.mkdir()
            (teacher / "vision.onnx").write_bytes(b"vision")
            (student / "text.onnx").write_bytes(b"text")
            (student / "tokenizer.json").write_text("{}", encoding="utf-8")
            projection = root / "projection.bin"
            projection.write_bytes(b"projection")
            teacher_manifest = {
                "format": 1,
                "id": "teacher",
                "version": "revision",
                "dimensions": 3,
                "contextLength": 4,
                "image": {"model": "vision.onnx"},
                "text": {"model": "unused", "tokenizer": "unused"},
            }
            student_manifest = {
                "format": 1,
                "id": "student",
                "version": "revision",
                "dimensions": 2,
                "contextLength": 7,
                "image": {"model": "unused"},
                "text": {"model": "text.onnx", "tokenizer": "tokenizer.json"},
            }
            teacher_path = teacher / "semantic-pack.json"
            student_path = student / "semantic-pack.json"
            teacher_path.write_text(json.dumps(teacher_manifest), encoding="utf-8")
            student_path.write_text(json.dumps(student_manifest), encoding="utf-8")

            subprocess.run(
                [
                    sys.executable,
                    str(TAGGER / "stage_distilled_query_pack.py"),
                    "--teacher-manifest",
                    str(teacher_path),
                    "--student-manifest",
                    str(student_path),
                    "--projection",
                    str(projection),
                    "--output",
                    str(output),
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            staged = json.loads((output / "semantic-pack.json").read_text(encoding="utf-8"))
            self.assertEqual((staged["id"], staged["version"]), ("teacher", "revision"))
            self.assertEqual(staged["dimensions"], 3)
            self.assertEqual(staged["contextLength"], 7)
            self.assertEqual(
                staged["text"]["projection"],
                {"path": "projection.bin", "inputDimensions": 2, "outputDimensions": 3},
            )
            provenance = json.loads(
                (output / "provenance.json").read_text(encoding="utf-8")
            )
            self.assertNotIn("provenance.json", provenance["outputs"])


if __name__ == "__main__":
    unittest.main()
