from __future__ import annotations

import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quantize_siglip_image import selected_nodes  # noqa: E402


class SelectiveSiglipQuantizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model = Path("target/release/lens/siglip2-base-p16-224-image-fp32-b1.onnx")
        if not cls.model.is_file():
            raise unittest.SkipTest("staged SigLIP2 image model is unavailable")

    def test_profiles_select_distinct_backbone_subsets(self) -> None:
        attention = selected_nodes(self.model, "attention")
        mlp = selected_nodes(self.model, "mlp")
        early = selected_nodes(self.model, "early-half")
        backbone = selected_nodes(self.model, "backbone")

        self.assertTrue(attention)
        self.assertTrue(mlp)
        self.assertTrue(early)
        self.assertTrue(set(attention).isdisjoint(mlp))
        self.assertTrue(set(early) < set(backbone))


if __name__ == "__main__":
    unittest.main()
