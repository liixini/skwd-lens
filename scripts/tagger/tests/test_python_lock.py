import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts/tagger/python_lock.py"
DIRECT = ROOT / "scripts/tagger/requirements-ci.in"
LOCK = ROOT / "scripts/tagger/requirements-ci.txt"
SETUP = ROOT / "scripts/setup-lens-test-env.sh"
WORKFLOW = ROOT / ".forgejo/workflows/verify.yml"
DIGEST = "a" * 64


def report(url="https://files.pythonhosted.org/packages/a/alpha-1.0-py3-none-any.whl"):
    return {
        "version": "1",
        "pip_version": "25.0.1",
        "environment": {
            "implementation_name": "cpython",
            "implementation_version": "3.12.11",
            "platform_machine": "x86_64",
            "platform_system": "Linux",
            "python_full_version": "3.12.11",
            "platform_python_implementation": "CPython",
            "python_version": "3.12",
            "sys_platform": "linux",
        },
        "install": [
            {
                "download_info": {
                    "url": url,
                    "archive_info": {
                        "hash": f"sha256={DIGEST}",
                        "hashes": {"sha256": DIGEST},
                    },
                },
                "is_yanked": False,
                "requested": True,
                "metadata": {"name": "Alpha", "version": "1.0"},
            }
        ],
    }


class PythonLockTests(unittest.TestCase):
    def run_lock(self, *arguments):
        return subprocess.run(
            [sys.executable, str(SCRIPT), *map(str, arguments)],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )

    def render_fixture(self, root: Path, payload=None):
        direct = root / "requirements.in"
        lock = root / "requirements.txt"
        report_path = root / "report.json"
        direct.write_text("alpha==1.0\n", encoding="utf-8")
        report_path.write_text(json.dumps(payload or report()), encoding="utf-8")
        result = self.run_lock(
            "render-report",
            "--input",
            direct,
            "--report",
            report_path,
            "--lock",
            lock,
        )
        return direct, lock, result

    def test_repository_lock_is_current_and_complete(self):
        result = self.run_lock("check", "--input", DIRECT, "--lock", LOCK)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_rendered_lock_is_deterministic_and_checks_offline(self):
        with tempfile.TemporaryDirectory() as temporary:
            direct, lock, result = self.render_fixture(Path(temporary))
            self.assertEqual(result.returncode, 0, result.stderr)
            first = lock.read_text(encoding="utf-8")
            self.assertIn("alpha==1.0 \\\n    --hash=sha256:" + DIGEST, first)
            result = self.run_lock("check", "--input", direct, "--lock", lock)
            self.assertEqual(result.returncode, 0, result.stderr)
            result = self.render_fixture(Path(temporary))[2]
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(lock.read_text(encoding="utf-8"), first)

    def test_check_rejects_changed_direct_input_and_duplicate_lock_entry(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            direct, lock, result = self.render_fixture(root)
            self.assertEqual(result.returncode, 0, result.stderr)
            direct.write_text("alpha==2.0\n", encoding="utf-8")
            result = self.run_lock("check", "--input", direct, "--lock", lock)
            self.assertNotEqual(result.returncode, 0)
            direct.write_text("alpha==1.0\n", encoding="utf-8")
            lock.write_text(
                lock.read_text(encoding="utf-8")
                + f"alpha==1.0 \\\n    --hash=sha256:{DIGEST}\n",
                encoding="utf-8",
            )
            result = self.run_lock("check", "--input", direct, "--lock", lock)
            self.assertNotEqual(result.returncode, 0)

    def test_render_rejects_non_pypi_missing_hash_and_yanked_artifacts(self):
        cases = []
        non_pypi = report("https://example.invalid/alpha-1.0-py3-none-any.whl")
        cases.append(non_pypi)
        missing_hash = report()
        del missing_hash["install"][0]["download_info"]["archive_info"]["hashes"]
        cases.append(missing_hash)
        yanked = report()
        yanked["install"][0]["is_yanked"] = True
        cases.append(yanked)
        for index, payload in enumerate(cases):
            with self.subTest(index=index), tempfile.TemporaryDirectory() as temporary:
                _, _, result = self.render_fixture(Path(temporary), copy.deepcopy(payload))
                self.assertNotEqual(result.returncode, 0)

    def test_setup_forces_hash_checked_binary_reinstallation(self):
        setup = SETUP.read_text(encoding="utf-8")
        for required in (
            '"--isolated"',
            '"--require-hashes"',
            '"--only-binary=:all:"',
            '"--force-reinstall"',
            '"--index-url" "$index"',
        ):
            self.assertIn(required, setup)
        self.assertIn('"$python" scripts/tagger/python_lock.py check', setup)

    def test_ci_never_executes_a_restored_virtual_environment(self):
        workflow = WORKFLOW.read_text(encoding="utf-8")
        self.assertNotIn("path: .venv-lens", workflow)
        self.assertNotIn("lens-python-3.12-", workflow)
        self.assertIn(
            "SKWD_LENS_TEST_ENV: /tmp/skwd-lens-venv-${{ github.run_id }}-${{ github.run_attempt }}",
            workflow,
        )
        self.assertEqual(workflow.count('"${SKWD_LENS_TEST_ENV}/bin/python"'), 2)


if __name__ == "__main__":
    unittest.main()
