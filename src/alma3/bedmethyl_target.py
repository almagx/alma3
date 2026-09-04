from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import tempfile
from importlib import resources
from pathlib import Path
from typing import Any

from .hashes import publish_new_file, sha256_file, validate_new_external_outputs
from .release import load_release, revalidate_release_identity

TARGET_RESOURCE = "alma3-3.0.0-grch38-projectable-cpgs.bed.gz"
TARGET_RELEASE_VERSION = "3.0.0"
TARGET_CPG_MANIFEST_SHA256 = "404f4fa5eaf6bc26eea2231d83b0463430fbbfba31f5a6d11f698a2f9ca05935"
TARGET_BED_GZIP_SHA256 = "c79f88affdb2848a85741d416b7b723b5c08077ee8948962114a666cfd23b7ac"
TARGET_BED_SHA256 = "3e6feef17fc4813caf4186c63d63165e7ddb77a0c3b31c93a2a33ddf6eec916a"
TARGET_CPG_COUNT = 65_535
EXCLUDED_CPG_IDS = ("cg05280794",)
_RECEIPT_KIND = "alma3_bedmethyl_target"
_RECEIPT_SCHEMA_VERSION = 1
_GRCH38_CANONICAL_LENGTHS = {
    "chr1": 248_956_422,
    "chr2": 242_193_529,
    "chr3": 198_295_559,
    "chr4": 190_214_555,
    "chr5": 181_538_259,
    "chr6": 170_805_979,
    "chr7": 159_345_973,
    "chr8": 145_138_636,
    "chr9": 138_394_717,
    "chr10": 133_797_422,
    "chr11": 135_086_622,
    "chr12": 133_275_309,
    "chr13": 114_364_328,
    "chr14": 107_043_718,
    "chr15": 101_991_189,
    "chr16": 90_338_345,
    "chr17": 83_257_441,
    "chr18": 80_373_285,
    "chr19": 58_617_616,
    "chr20": 64_444_167,
    "chr21": 46_709_983,
    "chr22": 50_818_468,
    "chrX": 156_040_895,
}


class BedMethylTargetError(ValueError):
    """Raised when the release-bound BedMethyl target cannot be exported safely."""


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _packaged_target() -> tuple[bytes, bytes]:
    compressed = resources.files("alma3").joinpath("assets", TARGET_RESOURCE).read_bytes()
    if _sha256_bytes(compressed) != TARGET_BED_GZIP_SHA256:
        raise BedMethylTargetError("packaged BedMethyl target gzip SHA-256 mismatch")
    try:
        payload = gzip.decompress(compressed)
    except (OSError, EOFError) as error:
        raise BedMethylTargetError("packaged BedMethyl target is not valid gzip") from error
    if _sha256_bytes(payload) != TARGET_BED_SHA256:
        raise BedMethylTargetError("packaged BedMethyl target SHA-256 mismatch")
    return compressed, payload


def _parse_bed3(
    payload: bytes,
    *,
    expected_count: int = TARGET_CPG_COUNT,
) -> tuple[tuple[str, int, int], ...]:
    try:
        text = payload.decode("ascii")
    except UnicodeDecodeError as error:
        raise BedMethylTargetError("BedMethyl target must be ASCII BED3") from error
    if not text.endswith("\n") or "\r" in text:
        raise BedMethylTargetError("BedMethyl target serialization is not canonical")
    rows: list[tuple[str, int, int]] = []
    seen: set[tuple[str, int, int]] = set()
    for line_number, line in enumerate(text.splitlines(), start=1):
        fields = line.split("\t")
        if len(fields) != 3:
            raise BedMethylTargetError(f"BedMethyl target is not BED3 at line {line_number}")
        try:
            row = fields[0], int(fields[1]), int(fields[2])
        except ValueError as error:
            raise BedMethylTargetError(f"BedMethyl target coordinate is invalid at line {line_number}") from error
        if row[0] not in _GRCH38_CANONICAL_LENGTHS or row[1] < 0 or row[2] != row[1] + 2:
            raise BedMethylTargetError(f"BedMethyl target interval is invalid at line {line_number}")
        if row in seen:
            raise BedMethylTargetError(f"BedMethyl target contains a duplicate at line {line_number}")
        if rows and (_chromosome_order(row[0]), row[1], row[2]) <= (
            _chromosome_order(rows[-1][0]),
            rows[-1][1],
            rows[-1][2],
        ):
            raise BedMethylTargetError(f"BedMethyl target is not strictly coordinate ordered at line {line_number}")
        seen.add(row)
        rows.append(row)
    if len(rows) != expected_count:
        raise BedMethylTargetError(
            f"BedMethyl target row count is {len(rows):,}, expected {expected_count:,}"
        )
    return tuple(rows)


def _chromosome_order(chromosome: str) -> int:
    suffix = chromosome.removeprefix("chr")
    return 23 if suffix == "X" else int(suffix)


def _release_target_rows(root: Path) -> tuple[tuple[tuple[str, int, int], ...], dict[str, tuple[str, int, int]]]:
    try:
        payload = json.loads((root / "cpg_manifest.json").read_text(encoding="utf-8"))
        iterator = zip(
            payload["cpg_ids"],
            payload["chrom"],
            payload["start"],
            payload["end"],
            strict=True,
        )
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise BedMethylTargetError("release CpG coordinates are invalid") from error
    rows: list[tuple[str, int, int]] = []
    excluded: dict[str, tuple[str, int, int]] = {}
    try:
        for raw_cpg_id, raw_chromosome, raw_start, raw_end in iterator:
            cpg_id = str(raw_cpg_id)
            row = str(raw_chromosome), int(raw_start), int(raw_end)
            if cpg_id in EXCLUDED_CPG_IDS:
                excluded[cpg_id] = row
            else:
                rows.append(row)
    except (TypeError, ValueError) as error:
        raise BedMethylTargetError("release CpG coordinates are invalid") from error
    if set(excluded) != set(EXCLUDED_CPG_IDS):
        raise BedMethylTargetError("release CpG manifest does not contain the required excluded probe")
    return tuple(rows), excluded


def _read_fai(path: Path) -> dict[str, tuple[int, int, int, int]]:
    try:
        lines = path.read_text(encoding="ascii").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise BedMethylTargetError(f"reference FASTA index is unreadable: {path}") from error
    index: dict[str, tuple[int, int, int, int]] = {}
    for line_number, line in enumerate(lines, start=1):
        fields = line.split("\t")
        if len(fields) < 5 or not fields[0] or fields[0] in index:
            raise BedMethylTargetError(f"reference FASTA index is invalid at line {line_number}")
        try:
            values = tuple(int(value) for value in fields[1:5])
        except ValueError as error:
            raise BedMethylTargetError(f"reference FASTA index is invalid at line {line_number}") from error
        if min(values) < 0 or values[2] <= 0 or values[3] < values[2]:
            raise BedMethylTargetError(f"reference FASTA index is invalid at line {line_number}")
        index[fields[0]] = values
    observed = {chromosome: index.get(chromosome, (None,))[0] for chromosome in _GRCH38_CANONICAL_LENGTHS}
    if observed != _GRCH38_CANONICAL_LENGTHS:
        raise BedMethylTargetError("reference FASTA does not have chr-prefixed GRCh38 canonical contig lengths")
    return index


def _read_fasta_interval(
    handle: Any,
    entry: tuple[int, int, int, int],
    chromosome: str,
    start: int,
    end: int,
) -> bytes:
    length, offset, line_bases, line_width = entry
    if start < 0 or end <= start or end > length:
        raise BedMethylTargetError(f"target interval is outside the reference: {chromosome}:{start}-{end}")
    pieces: list[bytes] = []
    position = start
    while position < end:
        line_index, column = divmod(position, line_bases)
        take = min(end - position, line_bases - column)
        handle.seek(offset + line_index * line_width + column)
        chunk = handle.read(take)
        if len(chunk) != take:
            raise BedMethylTargetError(
                f"reference FASTA and index disagree at {chromosome}:{start}-{end}"
            )
        pieces.append(chunk)
        position += take
    return b"".join(pieces).upper()


def _validate_reference(
    reference: Path,
    rows: tuple[tuple[str, int, int], ...],
    excluded: dict[str, tuple[str, int, int]],
) -> tuple[Path, str, str]:
    if not reference.is_file():
        raise FileNotFoundError(f"reference FASTA is not a file: {reference}")
    fai = Path(f"{reference}.fai")
    index = _read_fai(fai)
    try:
        with reference.open("rb") as handle:
            for chromosome, start, end in rows:
                if _read_fasta_interval(handle, index[chromosome], chromosome, start, end) != b"CG":
                    raise BedMethylTargetError(
                        f"ALMA3 target is not CpG in the reference: {chromosome}:{start}-{end}"
                    )
            for cpg_id, (chromosome, start, end) in excluded.items():
                if _read_fasta_interval(handle, index[chromosome], chromosome, start, end) == b"CG":
                    raise BedMethylTargetError(
                        f"excluded release probe unexpectedly resolves to CpG: {cpg_id}"
                    )
    except OSError as error:
        raise BedMethylTargetError(f"reference FASTA is unreadable: {reference}") from error
    return fai, sha256_file(reference), sha256_file(fai)


def _temporary_file(output: Path, payload: bytes) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_path = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
    temporary = Path(raw_path)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return temporary


def _unlink_owned(path: Path, identity: tuple[int, int]) -> None:
    try:
        stat = path.stat()
    except FileNotFoundError:
        return
    if (stat.st_dev, stat.st_ino) == identity:
        path.unlink()


def export_bedmethyl_target(
    reference: str | Path,
    output: str | Path,
    *,
    artifact: str | Path | None = None,
) -> dict[str, Any]:
    output_path = Path(output)
    if not output_path.name.endswith(".bed"):
        raise BedMethylTargetError("BedMethyl target output must end in .bed")
    receipt_path = Path(f"{output_path}.receipt.json")
    validated = load_release(artifact, device="cpu", load_model=False)
    root = Path(validated["root"])
    outputs = validate_new_external_outputs(
        root,
        {"BedMethyl target": output_path, "BedMethyl target receipt": receipt_path},
        inputs=(reference,),
    )
    output_path = outputs["BedMethyl target"]
    receipt_path = outputs["BedMethyl target receipt"]
    if (
        validated["release"]["version"] != TARGET_RELEASE_VERSION
        or validated["hashes"]["cpg_manifest.json"] != TARGET_CPG_MANIFEST_SHA256
    ):
        raise BedMethylTargetError("release does not match the ALMA3 3.0.0 BedMethyl target")
    _, target_payload = _packaged_target()
    rows = _parse_bed3(target_payload, expected_count=TARGET_CPG_COUNT)
    release_rows, excluded = _release_target_rows(root)
    if release_rows != rows:
        raise BedMethylTargetError("packaged BedMethyl target does not match the release CpG manifest")
    reference_path = Path(reference).expanduser().resolve()
    fai, reference_sha256, fai_sha256 = _validate_reference(reference_path, rows, excluded)
    receipt = {
        "kind": _RECEIPT_KIND,
        "schema_version": _RECEIPT_SCHEMA_VERSION,
        "release_version": validated["release"]["version"],
        "release_manifest_sha256": validated["manifest_sha256"],
        "cpg_manifest_sha256": validated["hashes"]["cpg_manifest.json"],
        "reference_fasta": str(reference_path),
        "reference_fasta_sha256": reference_sha256,
        "reference_fasta_index": str(fai.resolve()),
        "reference_fasta_index_sha256": fai_sha256,
        "target_bed_sha256": TARGET_BED_SHA256,
        "target_bed_gzip_sha256": TARGET_BED_GZIP_SHA256,
        "target_cpg_count": TARGET_CPG_COUNT,
        "excluded_cpg_ids": list(EXCLUDED_CPG_IDS),
    }
    receipt_payload = (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode("utf-8")
    target_temporary = _temporary_file(output_path, target_payload)
    receipt_temporary = _temporary_file(receipt_path, receipt_payload)
    target_identity = None
    try:
        revalidate_release_identity(
            root,
            manifest_sha256=validated["manifest_sha256"],
            hashes=validated["hashes"],
        )
        target_identity = publish_new_file(target_temporary, output_path)
        try:
            publish_new_file(receipt_temporary, receipt_path)
        except BaseException:
            _unlink_owned(output_path, target_identity)
            raise
    finally:
        target_temporary.unlink(missing_ok=True)
        receipt_temporary.unlink(missing_ok=True)
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="alma3 export-bedmethyl-target",
        description="Export and verify the ALMA3 3.0.0 GRCh38 Modkit target.",
    )
    parser.add_argument(
        "--artifact",
        help="release directory; otherwise use ALMA3_RELEASE",
    )
    parser.add_argument("--reference", required=True, help="chr-prefixed GRCh38 FASTA with adjacent .fai")
    parser.add_argument("--output", required=True, help="new uncompressed BED3 output ending in .bed")
    args = parser.parse_args(argv)
    receipt = export_bedmethyl_target(args.reference, args.output, artifact=args.artifact)
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
