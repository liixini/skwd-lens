import os
import hashlib
import json
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts/package-stage.sh"
DEFAULT_SCRIPT = ROOT / "scripts/package-default-stage.sh"
BASE_MANIFEST = ROOT / "packaging/manifest.txt"
DEFAULT_MANIFEST = ROOT / "packaging/default-manifest.txt"
DEFAULT_LOCK = ROOT / "packaging/default-semantic.lock.json"
BASE_EXPECTED = (
    "usr/bin/skwd-lens",
    "usr/share/licenses/skwd-lens/LICENSE",
    "usr/share/licenses/skwd-lens/third-party/Apache-2.0.txt",
    "usr/share/licenses/skwd-lens/third-party/CC-BY-4.0.txt",
    "usr/share/licenses/skwd-lens/third-party/MIT.txt",
)
DEFAULT_SEMANTIC = (
    "runtime/libonnxruntime.so.1.27.0",
    "runtime/libonnxruntime_providers_shared.so",
    "semantic-pack.json",
    "siglip2-base-p16-224-text-int8-split-b1.onnx",
    "siglip2-image-int8-attention.onnx",
    "siglip2-image-int8-attention.onnx.data",
    "siglip2-token-embedding-int8.bin",
    "tokenizer/tokenizer.model",
)


def release_binaries(directory: Path, names=BASE_EXPECTED[:1]) -> None:
    directory.mkdir(parents=True)
    for entry in names:
        path = directory / Path(entry).name
        path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        path.chmod(0o755)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def default_fixture(root: Path):
    payload = root / "payload"
    runtime = root / "runtime"
    manifest = root / "semantic-pack.json"
    payload.mkdir()
    runtime.mkdir()
    assets = {
        "image.onnx": b"image",
        "image.onnx.data": b"external-data",
        "text.onnx": b"text",
        "tokens.bin": b"tokens",
        "tokenizer/tokenizer.model": b"tokenizer",
    }
    for relative, content in assets.items():
        path = payload / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    runtime_assets = {
        "libonnxruntime.so.1.27.0": b"runtime",
        "libonnxruntime_providers_shared.so": b"providers",
    }
    for relative, content in runtime_assets.items():
        (runtime / relative).write_bytes(content)
    manifest.write_text(
        json.dumps(
            {
                "format": 1,
                "id": "fixture",
                "version": "one",
                "dimensions": 2,
                "contextLength": 4,
                "image": {
                    "model": "image.onnx",
                    "width": 1,
                    "height": 1,
                    "mean": [0.5, 0.5, 0.5],
                    "std": [0.5, 0.5, 0.5],
                },
                "text": {
                    "model": "text.onnx",
                    "tokenizer": "tokenizer/tokenizer.model",
                    "tokenEmbeddings": {"table": "tokens.bin"},
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    sources = {
        "image.onnx": ("payload", payload / "image.onnx"),
        "image.onnx.data": ("payload", payload / "image.onnx.data"),
        "runtime/libonnxruntime.so.1.27.0": (
            "runtime",
            runtime / "libonnxruntime.so.1.27.0",
        ),
        "runtime/libonnxruntime_providers_shared.so": (
            "runtime",
            runtime / "libonnxruntime_providers_shared.so",
        ),
        "semantic-pack.json": ("manifest", manifest),
        "text.onnx": ("payload", payload / "text.onnx"),
        "tokenizer/tokenizer.model": (
            "payload",
            payload / "tokenizer/tokenizer.model",
        ),
        "tokens.bin": ("payload", payload / "tokens.bin"),
    }
    lock = root / "default.lock.json"
    lock.write_text(
        json.dumps(
            {
                "format": 1,
                "product": "fixture@one",
                "runtimeVersion": "1.27.0",
                "files": [
                    {
                        "path": relative,
                        "source": source,
                        "size": path.stat().st_size,
                        "sha256": digest(path),
                    }
                    for relative, (source, path) in sorted(sources.items())
                ],
            }
        ),
        encoding="utf-8",
    )
    package_manifest = root / "package-manifest.txt"
    semantic_prefix = "usr/share/skwd-lens/models/semantic/"
    package_manifest.write_text(
        "\n".join(sorted(BASE_EXPECTED + tuple(semantic_prefix + path for path in sources)))
        + "\n",
        encoding="utf-8",
    )
    return payload, runtime, manifest, lock, package_manifest


class PackagingTests(unittest.TestCase):
    def test_manifest_is_the_exact_sorted_model_free_base_contract(self):
        entries = tuple(BASE_MANIFEST.read_text(encoding="utf-8").splitlines())
        self.assertEqual(entries, BASE_EXPECTED)
        self.assertEqual(entries, tuple(sorted(entries)))
        self.assertFalse(any("models/" in entry for entry in entries))
        self.assertFalse(any("skwd-wall-semantic" in entry for entry in entries))
        self.assertNotIn("model-packs", SCRIPT.read_text(encoding="utf-8"))

    def test_stage_contains_exact_manifest_with_canonical_names_and_modes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            binaries = root / "release"
            destination = root / "stage"
            release_binaries(binaries)
            environment = {**os.environ, "SKWD_LENS_RELEASE_BIN_DIR": str(binaries)}
            subprocess.run(
                ["sh", str(SCRIPT), str(destination)],
                cwd=root,
                env=environment,
                check=True,
                capture_output=True,
                text=True,
            )

            files = tuple(
                sorted(
                    str(path.relative_to(destination))
                    for path in destination.rglob("*")
                    if path.is_file()
                )
            )
            self.assertEqual(files, BASE_EXPECTED)
            for entry in BASE_EXPECTED[:1]:
                mode = stat.S_IMODE((destination / entry).stat().st_mode)
                self.assertEqual(mode, 0o755)
            for entry in BASE_EXPECTED[1:]:
                mode = stat.S_IMODE((destination / entry).stat().st_mode)
                self.assertEqual(mode, 0o644)

    def test_stage_fails_before_writing_when_a_helper_is_missing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            binaries = root / "release"
            destination = root / "stage"
            binaries.mkdir(parents=True)
            environment = {**os.environ, "SKWD_LENS_RELEASE_BIN_DIR": str(binaries)}
            result = subprocess.run(
                ["sh", str(SCRIPT), str(destination)],
                cwd=root,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("skwd-lens", result.stderr)
            self.assertFalse(destination.exists())

    def test_stage_refuses_a_nonempty_destination_without_mutation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            binaries = root / "release"
            destination = root / "stage"
            destination.mkdir()
            sentinel = destination / "unowned-file"
            sentinel.write_text("keep\n", encoding="utf-8")
            release_binaries(binaries)
            environment = {**os.environ, "SKWD_LENS_RELEASE_BIN_DIR": str(binaries)}
            result = subprocess.run(
                ["sh", str(SCRIPT), str(destination)],
                cwd=root,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("not empty", result.stderr)
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep\n")
            self.assertEqual(tuple(destination.iterdir()), (sentinel,))

    def test_default_contract_pins_only_the_curated_product(self):
        lock = json.loads(DEFAULT_LOCK.read_text(encoding="utf-8"))
        self.assertEqual(lock["format"], 1)
        self.assertEqual(
            lock["product"],
            "siglip2-base-p16-224@google-image-int8-attention-text-int8-stretch-v4",
        )
        self.assertEqual(lock["runtimeVersion"], "1.27.0")
        paths = tuple(entry["path"] for entry in lock["files"])
        self.assertEqual(paths, DEFAULT_SEMANTIC)
        self.assertEqual(paths, tuple(sorted(paths)))
        self.assertEqual(len(paths), len(set(paths)))
        self.assertEqual(sum(entry["size"] for entry in lock["files"]), 598_469_965)
        expected = tuple(DEFAULT_MANIFEST.read_text(encoding="utf-8").splitlines())
        self.assertEqual(
            expected,
            tuple(sorted(BASE_EXPECTED + tuple(f"usr/share/skwd-lens/models/semantic/{path}" for path in paths))),
        )

    def test_default_stage_contains_helper_runtime_and_model(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            binaries = root / "release"
            destination = root / "stage"
            release_binaries(binaries)
            payload, runtime, manifest, lock, package_manifest = default_fixture(root)
            environment = {**os.environ, "SKWD_LENS_RELEASE_BIN_DIR": str(binaries)}
            subprocess.run(
                [
                    "sh",
                    str(DEFAULT_SCRIPT),
                    str(destination),
                    str(payload),
                    str(runtime),
                    "--lock",
                    str(lock),
                    "--manifest",
                    str(manifest),
                    "--package-manifest",
                    str(package_manifest),
                ],
                cwd=root,
                env=environment,
                check=True,
                capture_output=True,
                text=True,
            )
            expected = tuple(package_manifest.read_text(encoding="utf-8").splitlines())
            files = tuple(
                sorted(
                    path.relative_to(destination).as_posix()
                    for path in destination.rglob("*")
                    if path.is_file()
                )
            )
            self.assertEqual(files, expected)
            self.assertEqual(
                stat.S_IMODE((destination / "usr/bin/skwd-lens").stat().st_mode),
                0o755,
            )
            for relative in DEFAULT_SEMANTIC:
                fixture_relative = relative.replace(
                    "siglip2-base-p16-224-text-int8-split-b1.onnx", "text.onnx"
                ).replace("siglip2-image-int8-attention.onnx.data", "image.onnx.data").replace(
                    "siglip2-image-int8-attention.onnx", "image.onnx"
                ).replace("siglip2-token-embedding-int8.bin", "tokens.bin")
                path = destination / "usr/share/skwd-lens/models/semantic" / fixture_relative
                if path.exists():
                    self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o644)

    def test_default_stage_fails_closed_on_a_payload_digest_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            binaries = root / "release"
            destination = root / "stage"
            release_binaries(binaries)
            payload, runtime, manifest, lock, package_manifest = default_fixture(root)
            (payload / "image.onnx").write_bytes(b"changed")
            environment = {**os.environ, "SKWD_LENS_RELEASE_BIN_DIR": str(binaries)}
            result = subprocess.run(
                [
                    "sh",
                    str(DEFAULT_SCRIPT),
                    str(destination),
                    str(payload),
                    str(runtime),
                    "--lock",
                    str(lock),
                    "--manifest",
                    str(manifest),
                    "--package-manifest",
                    str(package_manifest),
                ],
                cwd=root,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("mismatch", result.stderr)
            self.assertFalse(destination.exists())


if __name__ == "__main__":
    unittest.main()
