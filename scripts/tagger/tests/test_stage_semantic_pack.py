import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts/tagger/stage_semantic_pack.sh"


def write(path: Path, value: bytes = b"fixture") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)


def manifest(tokenizer: str = "tokenizer/tokenizer.model") -> dict:
    return {
        "format": 1,
        "id": "fixture",
        "version": "1",
        "dimensions": 2,
        "contextLength": 4,
        "image": {"model": "image.onnx"},
        "text": {
            "model": "text.onnx",
            "tokenizer": tokenizer,
            "tokenEmbeddings": {"table": "tokens.bin"},
            "projection": {"path": "projection.bin"},
        },
    }


def write_manifest(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def write_pack(root: Path, tokenizer: str = "tokenizer/tokenizer.model") -> None:
    write_manifest(root / "semantic-pack.json", manifest(tokenizer))
    for name in ("image.onnx", "text.onnx", "tokens.bin", "projection.bin"):
        write(root / name)
    if not Path(tokenizer).is_absolute() and ".." not in Path(tokenizer).parts:
        write(root / tokenizer, b"tokenizer")


class StageSemanticPackTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.runtime = self.root / "runtime-input"
        write(self.runtime / "libonnxruntime.so.1.27.0", b"runtime")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def run_stage(self, pack: Path, output: Path, name: str = "") -> subprocess.CompletedProcess:
        command = [str(SCRIPT), str(pack), str(self.runtime), str(output)]
        if name:
            command.append(name)
        return subprocess.run(command, text=True, capture_output=True, check=False)

    def test_stages_all_supported_manifest_assets(self) -> None:
        pack = self.root / "pack"
        output = self.root / "output"
        write_pack(pack)
        result = self.run_stage(pack, output)
        self.assertEqual(result.returncode, 0, result.stderr)
        staged = json.loads((output / "semantic-pack.json").read_text(encoding="utf-8"))
        references = [
            staged["image"]["model"],
            staged["text"]["model"],
            staged["text"]["tokenizer"],
            staged["text"]["tokenEmbeddings"]["table"],
            staged["text"]["projection"]["path"],
        ]
        for reference in references:
            resolved = (output / reference).resolve()
            self.assertTrue(resolved.is_relative_to(output.resolve()))
            self.assertTrue(resolved.is_file())
        self.assertTrue((output / "runtime/libonnxruntime.so.1.27.0").is_file())

    def test_replaces_existing_product_from_a_validated_candidate(self) -> None:
        pack = self.root / "pack"
        output = self.root / "output"
        write_pack(pack)
        first = self.run_stage(pack, output)
        self.assertEqual(first.returncode, 0, first.stderr)
        first_inode = output.stat().st_ino
        write(output / "preserved.bin", b"preserved")
        write(pack / "image.onnx", b"updated")

        second = self.run_stage(pack, output)

        self.assertEqual(second.returncode, 0, second.stderr)
        self.assertNotEqual(output.stat().st_ino, first_inode)
        self.assertEqual((output / "image.onnx").read_bytes(), b"updated")
        self.assertEqual((output / "preserved.bin").read_bytes(), b"preserved")

    def test_repairs_external_nested_tokenizer_with_identical_root_asset(self) -> None:
        pack = self.root / "semantic"
        write_pack(pack)
        shared = self.root / "semantic-model-research/tokenizer/tokenizer.model"
        write(shared, b"tokenizer")
        nested = pack / "packs/maximum"
        reference = os.path.relpath(shared, nested)
        write_pack(nested, reference)
        output = self.root / "output"
        result = self.run_stage(pack, output)
        self.assertEqual(result.returncode, 0, result.stderr)
        staged_path = output / "packs/maximum/semantic-pack.json"
        staged = json.loads(staged_path.read_text(encoding="utf-8"))
        self.assertEqual(staged["text"]["tokenizer"], "../../tokenizer/tokenizer.model")
        self.assertEqual(
            (staged_path.parent / staged["text"]["tokenizer"]).read_bytes(), b"tokenizer"
        )

    def test_copies_external_tokenizer_when_pack_has_no_internal_equivalent(self) -> None:
        shared = self.root / "research/tokenizer.model"
        write(shared, b"external-tokenizer")
        pack = self.root / "pack"
        reference = os.path.relpath(shared, pack)
        write_pack(pack, reference)
        output = self.root / "output"
        result = self.run_stage(pack, output, "maximum")
        self.assertEqual(result.returncode, 0, result.stderr)
        staged_path = output / "packs/maximum/semantic-pack.json"
        staged = json.loads(staged_path.read_text(encoding="utf-8"))
        self.assertEqual(staged["text"]["tokenizer"], "tokenizer/tokenizer.model")
        self.assertEqual(
            (staged_path.parent / staged["text"]["tokenizer"]).read_bytes(),
            b"external-tokenizer",
        )

    def test_materializes_internal_runtime_file_symlink(self) -> None:
        pack = self.root / "pack"
        write_pack(pack)
        versioned = self.runtime / "libonnxruntime.so.1.27.0"
        versioned.unlink()
        write(self.runtime / "libonnxruntime-real.so", b"runtime-target")
        versioned.symlink_to("libonnxruntime-real.so")
        output = self.root / "output"
        result = self.run_stage(pack, output)
        self.assertEqual(result.returncode, 0, result.stderr)
        staged = output / "runtime/libonnxruntime.so.1.27.0"
        self.assertFalse(staged.is_symlink())
        self.assertEqual(staged.read_bytes(), b"runtime-target")

    def test_rejects_source_tokenizer_symlink_before_publishing(self) -> None:
        pack = self.root / "pack"
        write_pack(pack)
        external = self.root / "external/tokenizer.model"
        write(external, b"external-tokenizer")
        tokenizer = pack / "tokenizer/tokenizer.model"
        tokenizer.unlink()
        tokenizer.symlink_to(external)
        output = self.root / "output"
        result = self.run_stage(pack, output)
        self.assertEqual(result.returncode, 1)
        self.assertIn("semantic pack must not contain symlinks", result.stderr)
        self.assertFalse(output.exists())

    def test_rejects_existing_destination_child_symlink_without_mutation(self) -> None:
        pack = self.root / "pack"
        output = self.root / "output"
        write_pack(pack)
        first = self.run_stage(pack, output)
        self.assertEqual(first.returncode, 0, first.stderr)
        original_manifest = (output / "semantic-pack.json").read_bytes()
        external = self.root / "external"
        write(external / "sentinel", b"untouched")
        hostile = output / "hostile"
        hostile.symlink_to(external, target_is_directory=True)

        second = self.run_stage(pack, output, "maximum")

        self.assertEqual(second.returncode, 1)
        self.assertIn("semantic product must not contain symlinks", second.stderr)
        self.assertEqual((output / "semantic-pack.json").read_bytes(), original_manifest)
        self.assertTrue(hostile.is_symlink())
        self.assertEqual((external / "sentinel").read_bytes(), b"untouched")

    def test_rejects_external_model_before_publishing(self) -> None:
        pack = self.root / "pack"
        write_pack(pack)
        external = self.root / "external/image.onnx"
        write(external)
        value = manifest()
        value["image"]["model"] = os.path.relpath(external, pack)
        write_manifest(pack / "semantic-pack.json", value)
        output = self.root / "output"
        result = self.run_stage(pack, output)
        self.assertEqual(result.returncode, 1)
        self.assertIn("image.model escapes the source pack", result.stderr)
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
