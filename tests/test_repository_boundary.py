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

    def test_software_citation_matches_public_release(self) -> None:
        citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
        for required in (
            "cff-version: 1.2.0",
            'type: software',
            'title: "ALMA3"',
            'version: "3.0.0"',
            'date-released: "2026-09-04"',
            '- name: "ALMA Genomics Inc."',
            'repository-code: "https://github.com/almagx/alma3"',
            'url: "https://almagx.com/atlas"',
        ):
            with self.subTest(required=required):
                self.assertIn(required, citation)

        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn(
            "ALMA Genomics Inc. (2026). ALMA3 (Version 3.0.0) [Computer software].",
            readme,
        )
        self.assertIn("Manuscript in preparation.", readme)

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
                "bedmethyl_target.py",
                "cli.py",
                "clinical_result.py",
                "config.py",
                "data.py",
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
            if path.is_file()
            and "tests" not in path.parts
            and (path.suffix in {".py", ".md", ".toml", ".sh"} or path.name == "Dockerfile")
        ]
        joined = "\n".join(path.read_text(encoding="utf-8") for path in searchable).lower()
        for forbidden in ("/home/", "/data/", "pacmap", "raw-beta", "schema-v3", "schema-v5"):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, joined)

    def test_readme_pins_the_ont_combined_projection(self) -> None:
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        ont = readme.split("<summary><strong>Oxford Nanopore</strong></summary>", 1)[1].split(
            "<summary><strong>PacBio HiFi</strong></summary>", 1
        )[0]
        for required in (
            "--modified-bases 5mC 5hmC",
            "--combine-mods",
            "--cpg",
            "--combine-strands",
            "--no-filtering",
            "--include-bed",
            "ALMA3 detects automatically",
        ):
            with self.subTest(required=required):
                self.assertIn(required, ont)

    def test_readme_pins_the_pacbio_5mc_projection(self) -> None:
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        pacbio = readme.split("<summary><strong>PacBio HiFi</strong></summary>", 1)[1].split(
            "<summary><strong>Results</strong></summary>", 1
        )[0]
        for required in (
            "--pileup-mode model",
            "--modsites-mode denovo",
            "--min-coverage 4",
            "--min-mapq 1",
            '$5 == "Total"',
            '$6, $4',
            "Do not use count mode or `discretized_mod_score`",
            "never add or normalize their probabilities",
            "ALMA3 detects 5mC automatically",
        ):
            with self.subTest(required=required):
                self.assertIn(required, pacbio)
        self.assertNotIn("--bedmethyl-modification-mode", readme)

    def test_readme_collapses_optional_sections(self) -> None:
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        for section in (
            "Inputs",
            "Oxford Nanopore",
            "PacBio HiFi",
            "Results",
            "Citation",
            "Support",
            "License",
        ):
            with self.subTest(section=section):
                self.assertIn(f"<summary><strong>{section}</strong></summary>", readme)
        self.assertIn("### Key features", readme)
        self.assertNotIn("<summary><strong>Key features</strong></summary>", readme)
        inputs = readme.index("<summary><strong>Inputs</strong></summary>")
        ont = readme.index("<summary><strong>Oxford Nanopore</strong></summary>")
        pacbio = readme.index("<summary><strong>PacBio HiFi</strong></summary>")
        self.assertLess(readme.index("</details>", inputs), ont)
        self.assertLess(readme.index("</details>", ont), pacbio)
        self.assertNotIn("<summary><strong>Python</strong></summary>", readme)
        self.assertNotIn("<summary><strong>Docker</strong></summary>", readme)
        self.assertNotIn("<summary><strong>Performance</strong></summary>", readme)
        self.assertIn("ghcr.io/almagx/alma3:3.0.0", readme)
        self.assertNotIn("pip install alma3", readme)
        self.assertIn(
            "Use our platform, [app.almagx.com](https://app.almagx.com/), for automated "
            "preprocessing, ALMA3 classification, diagnostic maps, and integrated reports.",
            readme,
        )

    def test_readme_uses_plain_linked_license_copy_without_a_badge(self) -> None:
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn(
            "ALMA3 models are free to explore and use for research, education, and other "
            "noncommercial or internal projects. For commercial use, see "
            "[ALMA3 License 1.0](LICENSE).",
            readme,
        )
        self.assertNotIn('alt="ALMA3 License 1.0"', readme)

    def test_repository_has_no_public_model_downloader(self) -> None:
        tracked = subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=ROOT,
            check=True,
            capture_output=True,
        ).stdout.split(b"\0")
        text = "\n".join(
            path.read_text(encoding="utf-8")
            for raw_path in tracked
            if raw_path
            for path in (ROOT / raw_path.decode(),)
            if path.is_file()
            and path.suffix in {".py", ".md", ".toml", ".json", ".sh"}
            and "tests" not in path.parts
        ).lower()
        for removed in ("models.almagx.com", "release_catalog.json", "alma3 download"):
            with self.subTest(removed=removed):
                self.assertNotIn(removed, text)


if __name__ == "__main__":
    unittest.main()
