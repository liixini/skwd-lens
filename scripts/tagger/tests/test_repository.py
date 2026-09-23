import json
import os
import re
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
WORKFLOW = ROOT / ".forgejo/workflows/verify.yml"
TEST_ALL = ROOT / "scripts/test-all.sh"
DEFAULT_ASSET = "release-assets/skwd-lens_0.1.0_linux-x86_64.tar.xz"
DEFAULT_ASSET_SHA256 = "3cea269214cad0892340201bb2717996fa190bfcb336fb25fd4420eb0b3dc243"
DEFAULT_ASSET_SIZE = 449226684


class RepositoryTests(unittest.TestCase):
    def test_large_model_payloads_are_lfs_release_assets(self):
        tracked = subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=ROOT,
            check=True,
            capture_output=True,
        ).stdout.split(b"\0")
        large = sorted(
            str(path.relative_to(ROOT))
            for raw in tracked
            if raw
            for path in [ROOT / os.fsdecode(raw)]
            if int(
                subprocess.run(
                    ["git", "cat-file", "-s", f":{path.relative_to(ROOT)}"],
                    cwd=ROOT,
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout
            )
            > 5 * 1024 * 1024
        )
        self.assertEqual(large, [])

    def test_default_release_archive_is_an_exact_lfs_pointer(self):
        pointer = subprocess.run(
            ["git", "show", f":{DEFAULT_ASSET}"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        self.assertEqual(
            pointer,
            "version https://git-lfs.github.com/spec/v1\n"
            f"oid sha256:{DEFAULT_ASSET_SHA256}\n"
            f"size {DEFAULT_ASSET_SIZE}\n",
        )
        checksum = (ROOT / "release-assets/SHA256SUMS").read_text(encoding="utf-8")
        self.assertEqual(checksum, f"{DEFAULT_ASSET_SHA256}  {Path(DEFAULT_ASSET).name}\n")
        attribute = subprocess.run(
            ["git", "check-attr", "--cached", "filter", "--", DEFAULT_ASSET],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        self.assertEqual(attribute, f"{DEFAULT_ASSET}: filter: lfs\n")

    def test_private_checkout_uses_restricted_token_without_persisting_it(self):
        workflow = WORKFLOW.read_text(encoding="utf-8")
        checkout = """          repository: liixini/skwd-verify
          ref: 6c8fc9023e41a2d71fed44de4f7b7313d0f7fb17
          path: .skwd-verify
          token: ${{ secrets.SKWD_SUITE_READ_TOKEN }}
          persist-credentials: false
"""
        self.assertEqual(workflow.count(checkout), 1)
        self.assertEqual(workflow.count("repository: liixini/"), 1)
        self.assertEqual(workflow.count("token: ${{ secrets.SKWD_SUITE_READ_TOKEN }}"), 1)
        self.assertEqual(workflow.count("persist-credentials: false"), 2)

    def test_required_gate_emits_the_exact_catalog_and_retains_provenance(self):
        workflow = WORKFLOW.read_text(encoding="utf-8")
        catalog = json.loads(
            (ROOT / "scripts/ci_suite_catalog.json").read_text(encoding="utf-8")
        )
        suites = [value["id"] for value in catalog["suites"]]
        self.assertEqual(len(suites), len(set(suites)))
        self.assertEqual(workflow.count("name: Forgejo / Lens required"), 1)
        for suite in suites:
            self.assertEqual(
                len(re.findall(rf"--suite {re.escape(suite)}(?:\s|$)", workflow)),
                1,
                suite,
            )
        self.assertIn(
            'if: always()\n        run: python3 "$SKWD_VERIFY_ROOT/scripts/ci-report.py" aggregate',
            workflow,
        )
        self.assertIn("retention-days: 14", workflow)
        self.assertIn(
            "actions/upload-artifact@c6a3b2bd78b3985e4b2f15397fec357f0fd808de",
            workflow,
        )

    def test_python_gate_loads_cuda_libraries_from_its_virtual_environment(self):
        script = TEST_ALL.read_text(encoding="utf-8")
        self.assertIn("site.getsitepackages()", script)
        self.assertIn('.glob("*/lib")', script)
        self.assertEqual(script.count('LD_LIBRARY_PATH="$nvidia_library_path"'), 2)
        self.assertNotIn("/usr/local/cuda", script)


if __name__ == "__main__":
    unittest.main()
