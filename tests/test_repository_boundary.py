from __future__ import annotations

import subprocess
import unittest
from pathlib import Path

from alma3.release import RELEASE_LICENSE_HEADER

ROOT = Path(__file__).resolve().parents[1]


class RepositoryBoundaryTests(unittest.TestCase):
    def test_checked_in_license_matches_release_contract(self) -> None:
        license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")
        self.assertTrue(license_text.startswith(RELEASE_LICENSE_HEADER))

    def test_source_tree_is_inference_only(self) -> None:
        modules = {
            path.name
            for path in (ROOT / "src/alma3").glob("*.py")
        }
        self.assertEqual(
            modules,
            {
                "__init__.py",
                "__main__.py",
                "cli.py",
                "clinical_result.py",
                "config.py",
                "data.py",
                "download.py",
                "dx.py",
                "hashes.py",
                "infer.py",
                "model.py",
                "release.py",
                "runtime.py",
                "sitewise.py",
            },
        )
        pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8").lower()
        self.assertNotIn("pacmap", pyproject)
        self.assertNotIn("h5py", pyproject)
        cli = (ROOT / "src/alma3/cli.py").read_text(encoding="utf-8")
        self.assertNotIn("build-map-assets", cli)
        self.assertNotIn("finalize", cli)
        self.assertNotIn("evaluate", cli)

    def test_repository_contains_no_private_paths_or_binary_artifacts(self) -> None:
        tracked = subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=ROOT,
            check=True,
            capture_output=True,
        ).stdout.split(b"\0")
        files = [ROOT / path.decode() for path in tracked if path]
        forbidden_suffixes = {".safetensors", ".pt", ".pth", ".h5", ".hdf5"}
        self.assertFalse([path for path in files if path.suffix.lower() in forbidden_suffixes])
        searchable = [
            path
            for path in files
            if "tests" not in path.parts
            and (path.suffix in {".py", ".md", ".toml", ".sh"} or path.name == "Dockerfile")
        ]
        joined = "\n".join(path.read_text(encoding="utf-8") for path in searchable).lower()
        for forbidden in ("/home/", "/data/", "pacmap", "raw-beta", "schema-v3", "schema-v5"):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, joined)


if __name__ == "__main__":
    unittest.main()
