import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prepare_semantic_views import patch_grid, prepare  # noqa: E402


class PrepareSemanticViewsTests(unittest.TestCase):
    def test_builds_full_center_and_ultrawide_third_views(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            Image.new("RGB", (600, 100)).save(root / "wide.png")
            Image.new("RGB", (160, 90)).save(root / "normal.png")
            source = {
                "fingerprint": 1,
                "entries": [
                    {"key": "static:wide.png", "path": "/ignored", "fingerprint": 1},
                    {"key": "static:normal.png", "path": "/ignored", "fingerprint": 2},
                ],
            }

            full, multiview, shapes = prepare(source, root, 2.0, 16, 256)

            self.assertEqual(len(full["entries"]), 2)
            self.assertEqual(len(multiview["entries"]), 6)
            self.assertEqual(
                [entry.get("view", "full") for entry in multiview["entries"]],
                ["full", "center", "leftThird", "rightThird", "full", "center"],
            )
            self.assertEqual(len({entry["fingerprint"] for entry in multiview["entries"]}), 6)
            self.assertIn(list(patch_grid(100, 600, 16, 256)), shapes)
            self.assertIn([16, 16], shapes)

    def test_rejects_missing_original_static_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = {
                "fingerprint": 1,
                "entries": [
                    {"key": "static:missing.png", "path": "/ignored", "fingerprint": 1}
                ],
            }

            with self.assertRaises(FileNotFoundError):
                prepare(source, Path(temporary), 2.0, 16, 256)


if __name__ == "__main__":
    unittest.main()
