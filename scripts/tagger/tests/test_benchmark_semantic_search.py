from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark_semantic_search import (  # noqa: E402
    average_precision,
    exclude_forbidden,
    is_positive,
    metrics,
    ndcg,
    percentile,
)


class BenchmarkSemanticSearchTests(unittest.TestCase):
    def test_positive_requires_groups_and_rejects_forbidden_tags(self) -> None:
        query = {
            "requiredAny": [["city", "urban"], ["night"]],
            "forbiddenAny": ["car"],
        }

        self.assertTrue(is_positive({"city", "night"}, query))
        self.assertFalse(is_positive({"city"}, query))
        self.assertFalse(is_positive({"urban", "night", "car"}, query))

    def test_metrics_count_forbidden_intrusions(self) -> None:
        query = {"requiredAny": [["road"]], "forbiddenAny": ["car"]}
        tags = [{"road"}, {"road", "car"}, {"tree"}]

        result = metrics(np.asarray([0, 1, 2]), tags, query, 3)

        self.assertEqual(result["positiveCount"], 1)
        self.assertAlmostEqual(result["precisionAtK"], 1 / 3)
        self.assertAlmostEqual(result["requiredAtK"], 2 / 3)
        self.assertAlmostEqual(result["forbiddenAtK"], 1 / 3)

    def test_exclude_forbidden_preserves_rank_of_allowed_items(self) -> None:
        query = {"forbiddenAny": ["car", "truck"]}
        tags = [{"road", "car"}, {"road"}, {"truck"}, {"forest"}]

        result = exclude_forbidden(np.asarray([2, 0, 3, 1]), tags, query)

        np.testing.assert_array_equal(result, np.asarray([3, 1]))

    def test_ranking_helpers(self) -> None:
        relevant = [False, True, True, False]

        self.assertAlmostEqual(average_precision(relevant, 2), (0.5 + 2 / 3) / 2)
        self.assertGreater(ndcg(relevant, 2), 0)
        self.assertEqual(percentile([4, 1, 3, 2], 0.95), 4)


if __name__ == "__main__":
    unittest.main()
