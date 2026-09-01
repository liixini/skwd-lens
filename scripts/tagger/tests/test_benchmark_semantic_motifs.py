from __future__ import annotations

import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark_semantic_motifs import deterministic_pool, summarize  # noqa: E402


class BenchmarkSemanticMotifTests(unittest.TestCase):
    def test_pool_keeps_anchors_and_is_independent_of_input_order(self) -> None:
        items = [{"key": key} for key in ["c", "anchor", "a", "b"]]

        first = deterministic_pool(items, {"anchor"}, 3)
        second = deterministic_pool(list(reversed(items)), {"anchor"}, 3)

        self.assertEqual([item["key"] for item in first], [item["key"] for item in second])
        self.assertEqual(first[0]["key"], "anchor")

    def test_summary_reports_retrieval_quality(self) -> None:
        report = summarize(
            [
                {"rank": 1, "queryMilliseconds": 2.0},
                {"rank": 4, "queryMilliseconds": 4.0},
                {"rank": 20, "queryMilliseconds": 6.0},
            ]
        )

        self.assertAlmostEqual(report["hitAt1"], 1 / 3)
        self.assertAlmostEqual(report["hitAt5"], 2 / 3)
        self.assertEqual(report["medianRank"], 4)
        self.assertEqual(report["worstRank"], 20)
        self.assertEqual(report["meanQueryMilliseconds"], 4)


if __name__ == "__main__":
    unittest.main()
