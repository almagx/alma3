from __future__ import annotations

import gzip
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from alma3.bedmethyl_target import (
    EXCLUDED_CPG_IDS,
    TARGET_BED_GZIP_SHA256,
    TARGET_BED_SHA256,
    TARGET_CPG_COUNT,
    BedMethylTargetError,
    _packaged_target,
    _parse_bed3,
    _read_fai,
    _release_target_rows,
    _validate_reference,
    export_bedmethyl_target,
    main,
)
from alma3.release import load_release


def _small_reference(root: Path, sequence: bytes = b"AACGAACGAA") -> Path:
    reference = root / "GRCh38.fa"
    reference.write_bytes(b">chr1\n" + sequence + b"\n")
    Path(f"{reference}.fai").write_text("chr1\t10\t6\t10\t11\n", encoding="ascii")
    return reference


def _small_release(root: Path) -> Path:
    release = root / "release"
    release.mkdir()
    (release / "cpg_manifest.json").write_text(
        json.dumps(
            {
                "cpg_ids": ["cg1", EXCLUDED_CPG_IDS[0], "cg2"],
                "chrom": ["chr1", "chr1", "chr1"],
                "start": [2, 4, 6],
                "end": [4, 6, 8],
            }
        ),
        encoding="utf-8",
    )
    return release


class BedMethylTargetTests(unittest.TestCase):
    def test_packaged_target_is_exact_deterministic_bed3(self) -> None:
        compressed, payload = _packaged_target()
        self.assertEqual(hashlib.sha256(compressed).hexdigest(), TARGET_BED_GZIP_SHA256)
        self.assertEqual(hashlib.sha256(payload).hexdigest(), TARGET_BED_SHA256)
        self.assertEqual(gzip.decompress(compressed), payload)
        rows = _parse_bed3(payload)
        self.assertEqual(len(rows), TARGET_CPG_COUNT)
        self.assertEqual(len(set(rows)), TARGET_CPG_COUNT)

    def test_bed3_rejects_duplicate_reordered_malformed_and_wrong_count(self) -> None:
        cases = (
            (b"chr1\t2\t4\nchr1\t2\t4\n", "duplicate"),
            (b"chr1\t6\t8\nchr1\t2\t4\n", "ordered"),
            (b"chr1\t2\t3\n", "interval"),
            (b"chr1\t2\t4\textra\n", "BED3"),
            (b"chr1\t2\t4", "serialization"),
            (b"chr1\t2\t4\n", "row count"),
        )
        for payload, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(BedMethylTargetError, message):
                _parse_bed3(payload, expected_count=2)

    def test_release_rows_exclude_only_the_declared_probe(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            release = _small_release(Path(raw))
            rows, excluded = _release_target_rows(release)
        self.assertEqual(rows, (("chr1", 2, 4), ("chr1", 6, 8)))
        self.assertEqual(excluded, {EXCLUDED_CPG_IDS[0]: ("chr1", 4, 6)})

    def test_reference_validation_requires_grch38_lengths_and_cpg_sequence(self) -> None:
        rows = (("chr1", 2, 4), ("chr1", 6, 8))
        excluded = {EXCLUDED_CPG_IDS[0]: ("chr1", 4, 6)}
        with tempfile.TemporaryDirectory() as raw, patch(
            "alma3.bedmethyl_target._GRCH38_CANONICAL_LENGTHS", {"chr1": 10}
        ):
            root = Path(raw)
            reference = _small_reference(root)
            fai, reference_sha, fai_sha = _validate_reference(reference, rows, excluded)
            self.assertEqual(reference_sha, hashlib.sha256(reference.read_bytes()).hexdigest())
            self.assertEqual(fai_sha, hashlib.sha256(fai.read_bytes()).hexdigest())

            Path(f"{reference}.fai").write_text("chr1\t9\t6\t10\t11\n", encoding="ascii")
            with self.assertRaisesRegex(BedMethylTargetError, "canonical contig lengths"):
                _read_fai(Path(f"{reference}.fai"))

            _small_reference(root, b"AACAATCGAA")
            with self.assertRaisesRegex(BedMethylTargetError, "not CpG"):
                _validate_reference(reference, rows, excluded)

            Path(f"{reference}.fai").unlink()
            with self.assertRaisesRegex(BedMethylTargetError, "index is unreadable"):
                _validate_reference(reference, rows, excluded)

    def test_export_is_release_bound_atomic_and_refuses_existing_outputs(self) -> None:
        target = b"chr1\t2\t4\nchr1\t6\t8\n"
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            release = _small_release(root)
            reference = _small_reference(root)
            output = root / "target.bed"
            validated = {
                "root": release,
                "release": {"version": "3.0.0"},
                "hashes": {"cpg_manifest.json": "404f4fa5eaf6bc26eea2231d83b0463430fbbfba31f5a6d11f698a2f9ca05935"},
                "manifest_sha256": "a" * 64,
            }
            with (
                patch("alma3.bedmethyl_target.load_release", return_value=validated) as loader,
                patch("alma3.bedmethyl_target.revalidate_release_identity"),
                patch("alma3.bedmethyl_target._packaged_target", return_value=(gzip.compress(target), target)),
                patch("alma3.bedmethyl_target.TARGET_CPG_COUNT", 2),
                patch("alma3.bedmethyl_target.TARGET_BED_SHA256", hashlib.sha256(target).hexdigest()),
                patch("alma3.bedmethyl_target._GRCH38_CANONICAL_LENGTHS", {"chr1": 10}),
            ):
                receipt = export_bedmethyl_target(reference, output, artifact=release)
                loader.assert_called_once_with(release, device="cpu", load_model=False)
                self.assertEqual(output.read_bytes(), target)
                self.assertEqual(receipt["target_cpg_count"], 2)
                self.assertEqual(
                    json.loads(Path(f"{output}.receipt.json").read_text(encoding="utf-8")),
                    receipt,
                )
                with self.assertRaises(FileExistsError):
                    export_bedmethyl_target(reference, output, artifact=release)

    def test_export_rejects_a_different_release(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            release = _small_release(root)
            reference = _small_reference(root)
            validated = {
                "root": release,
                "release": {"version": "3.0.0"},
                "hashes": {"cpg_manifest.json": "0" * 64},
                "manifest_sha256": "a" * 64,
            }
            with patch("alma3.bedmethyl_target.load_release", return_value=validated), self.assertRaisesRegex(
                BedMethylTargetError, "does not match"
            ):
                export_bedmethyl_target(reference, root / "target.bed", artifact=release)

    def test_load_release_can_skip_model_loading(self) -> None:
        with patch("alma3.release.validate_release", return_value={}) as validate:
            load_release("release", load_model=False)
        validate.assert_called_once_with(Path("release"), device="cpu", load_model=False)

    def test_cli_passes_explicit_paths_to_exporter(self) -> None:
        with patch("alma3.bedmethyl_target.export_bedmethyl_target", return_value={}) as export:
            self.assertEqual(
                main(["--artifact", "release", "--reference", "GRCh38.fa", "--output", "target.bed"]),
                0,
            )
        export.assert_called_once_with("GRCh38.fa", "target.bed", artifact="release")


if __name__ == "__main__":
    unittest.main()
