from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
import tempfile
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import torch

from .clinical_result import serialize_result
from .data import CpGManifest
from .dx import (
    DX_REPRESENTATION_DIMENSIONS,
    DX_REPRESENTATION_DTYPE,
    DX_REPRESENTATION_NAME,
    DX_REPRESENTATION_VERSION,
)
from .hashes import publish_new_file, validate_new_external_outputs
from .release import revalidate_release_identity
from .runtime import ALMA3
from .sitewise import real_coverage_presentation


class InputContractError(ValueError):
    """Raised when an inference input does not match the ALMA3 contract."""


DEFAULT_INFERENCE_BATCH_SIZE = 2
EMBEDDING_SIDECAR_KIND = "alma3_embedding_sidecar"
EMBEDDING_SIDECAR_SCHEMA_VERSION = 1


def embedding_sidecar_schema_path() -> Path:
    return Path(__file__).with_name("schemas") / "embedding_sidecar.schema.json"


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and all(char in "0123456789abcdef" for char in value)
    )


def validate_embedding_sidecar(payload: Any) -> None:
    fields = {
        "kind",
        "schema_version",
        "release",
        "representation",
        "minimum_observed_cpgs",
        "samples",
    }
    if not isinstance(payload, dict) or set(payload) != fields:
        raise InputContractError("embedding sidecar fields are invalid")
    if payload["kind"] != EMBEDDING_SIDECAR_KIND or payload["schema_version"] != EMBEDDING_SIDECAR_SCHEMA_VERSION:
        raise InputContractError("embedding sidecar kind or schema version is invalid")
    release_fields = {
        "manifest_sha256",
        "model_sha256",
        "taxonomy_sha256",
        "cpg_manifest_sha256",
        "thresholds_sha256",
    }
    release = payload["release"]
    if not isinstance(release, dict) or set(release) != release_fields:
        raise InputContractError("embedding sidecar release fields are invalid")
    if not all(_is_sha256(release[field]) for field in release_fields):
        raise InputContractError("embedding sidecar release hashes are invalid")
    representation = payload["representation"]
    representation_fields = {"name", "version", "dtype", "dimensions"}
    if not isinstance(representation, dict) or set(representation) != representation_fields:
        raise InputContractError("embedding sidecar representation fields are invalid")
    if (
        representation["name"] != DX_REPRESENTATION_NAME
        or representation["version"] != DX_REPRESENTATION_VERSION
        or representation["dtype"] != DX_REPRESENTATION_DTYPE
        or representation["dimensions"] != DX_REPRESENTATION_DIMENSIONS
    ):
        raise InputContractError("embedding sidecar representation is invalid")
    minimum = payload["minimum_observed_cpgs"]
    if type(minimum) is not int or minimum <= 0:
        raise InputContractError("embedding sidecar observed-CpG floor is invalid")
    samples = payload["samples"]
    if not isinstance(samples, list) or not samples:
        raise InputContractError("embedding sidecar samples must be a nonempty list")
    seen: set[str] = set()
    dimensions = representation["dimensions"]
    for sample in samples:
        if not isinstance(sample, dict) or set(sample) != {"sample_id", "observed_cpg_count", "embedding"}:
            raise InputContractError("embedding sidecar sample fields are invalid")
        sample_id = sample["sample_id"]
        if not isinstance(sample_id, str) or not sample_id or sample_id in seen:
            raise InputContractError("embedding sidecar sample IDs must be nonempty and unique")
        seen.add(sample_id)
        observed_count = sample["observed_cpg_count"]
        if type(observed_count) is not int or observed_count < minimum:
            raise InputContractError("embedding sidecar sample is below the observed-CpG floor")
        embedding = sample["embedding"]
        if not isinstance(embedding, list) or len(embedding) != dimensions:
            raise InputContractError("embedding sidecar vector dimensions are invalid")
        if any(
            isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
            for value in embedding
        ):
            raise InputContractError("embedding sidecar vectors must be finite numbers")


def _json_temporary(output: Path, payload: Any) -> Path:
    descriptor, raw_temporary = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
    temporary = Path(raw_temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return temporary


def _unlink_owned(path: Path, identity: tuple[int, int]) -> None:
    try:
        current = path.stat(follow_symlinks=False)
    except FileNotFoundError:
        return
    if (current.st_dev, current.st_ino) == identity:
        path.unlink()


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int]:
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns


def _array_csv_batches(
    path: str | Path,
    cpg: CpGManifest,
    *,
    batch_size: int = DEFAULT_INFERENCE_BATCH_SIZE,
) -> Iterator[tuple[list[str], torch.Tensor, torch.Tensor]]:
    if type(batch_size) is not int or batch_size <= 0:
        raise InputContractError("batch_size must be a positive integer")
    input_path = Path(path)
    opener = gzip.open if input_path.name.endswith(".gz") else open
    with opener(input_path, "rt", encoding="utf-8", newline="") as handle:
        initial_stat = os.fstat(handle.fileno())
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration:
            raise InputContractError("array CSV is empty") from None
        if len(header) < 2:
            raise InputContractError("array CSV requires sample id column plus CpG columns")
        columns = header[1:]
        seen: set[str] = set()
        duplicate_seen: set[str] = set()
        duplicates: list[str] = []
        for name in columns:
            if name in seen and name not in duplicate_seen:
                duplicates.append(name)
                duplicate_seen.add(name)
            seen.add(name)
        if duplicates:
            raise InputContractError(f"array CSV has duplicate CpG columns: {duplicates[:5]}")
        column_index = {name: idx for idx, name in enumerate(columns)}
        cpg_columns = [column_index.get(cpg_id) for cpg_id in cpg.cpg_ids]
        sample_ids: list[str] = []
        seen_sample_ids: set[str] = set()
        values: list[list[float]] = []
        observed_rows: list[list[bool]] = []
        for line_no, row in enumerate(reader, start=2):
            if len(row) != len(header):
                raise InputContractError(f"array CSV row {line_no} has {len(row)} fields; expected {len(header)}")
            sample_id = row[0]
            if not sample_id.strip():
                raise InputContractError(f"array CSV row {line_no} has blank sample id")
            if sample_id in seen_sample_ids:
                raise InputContractError(f"array CSV has duplicate sample id: {sample_id}")
            seen_sample_ids.add(sample_id)
            sample_values: list[float] = []
            sample_observed: list[bool] = []
            for column in cpg_columns:
                if column is None:
                    sample_values.append(0.0)
                    sample_observed.append(False)
                    continue
                raw = row[1 + column].strip()
                if raw == "" or raw.lower() == "nan":
                    sample_values.append(0.0)
                    sample_observed.append(False)
                    continue
                value = float(raw)
                if not math.isfinite(value) or value < 0 or value > 1:
                    raise InputContractError("array CSV observed beta values must be finite and in [0, 1]")
                sample_values.append(value)
                sample_observed.append(True)
            if not any(sample_observed):
                raise InputContractError(f"array CSV row {line_no} has no observed CpGs")
            sample_ids.append(sample_id)
            values.append(sample_values)
            observed_rows.append(sample_observed)
            if len(values) == batch_size:
                yield (
                    sample_ids,
                    torch.tensor(values, dtype=torch.float32),
                    torch.tensor(observed_rows, dtype=torch.bool),
                )
                sample_ids, values, observed_rows = [], [], []
        if values:
            yield (
                sample_ids,
                torch.tensor(values, dtype=torch.float32),
                torch.tensor(observed_rows, dtype=torch.bool),
            )
        try:
            path_stat = input_path.stat()
        except FileNotFoundError:
            raise InputContractError("array CSV changed while inference was running") from None
        if _stat_identity(os.fstat(handle.fileno())) != _stat_identity(initial_stat) or _stat_identity(
            path_stat
        ) != _stat_identity(initial_stat):
            raise InputContractError("array CSV changed while inference was running")
        if not seen_sample_ids:
            raise InputContractError("array CSV contains no samples")


def load_array_csv(path: str | Path, cpg: CpGManifest) -> tuple[list[str], torch.Tensor, torch.Tensor]:
    batches = list(_array_csv_batches(path, cpg))
    return (
        [sample_id for sample_ids, _, _ in batches for sample_id in sample_ids],
        torch.cat([values for _, values, _ in batches]),
        torch.cat([observed for _, _, observed in batches]),
    )


def load_bed_methyl_with_manifest(
    path: str | Path, cpg: CpGManifest, sample_id: str | None = None
) -> tuple[list[str], torch.Tensor, torch.Tensor, torch.Tensor]:
    input_path = Path(path)
    if cpg.chrom is None or cpg.start is None:
        raise InputContractError("bedMethyl inference requires chrom and genomic start in CpG manifest")
    coord_to_indices = cpg.coordinate_index
    values = torch.zeros(len(cpg.cpg_ids), dtype=torch.float32)
    observed = torch.zeros(len(cpg.cpg_ids), dtype=torch.bool)
    coverage_by_cpg = torch.zeros(len(cpg.cpg_ids), dtype=torch.int64)
    seen_coords: set[tuple[str, int]] = set()
    opener = gzip.open if input_path.name.endswith(".gz") else open
    with opener(input_path, "rt", encoding="utf-8") as handle:
        initial_stat = os.fstat(handle.fileno())
        reader = csv.reader(handle, delimiter="\t")
        for row in reader:
            if len(row) < 11:
                raise InputContractError("bedMethyl rows must have at least 11 columns")
            key = (row[0], int(row[1]))
            indices = coord_to_indices.get(key)
            if indices is None:
                continue
            if key in seen_coords:
                raise InputContractError(f"bedMethyl has duplicate release CpG coordinate: {key[0]}:{key[1]}")
            seen_coords.add(key)
            coverage = float(row[9])
            if (
                not math.isfinite(coverage)
                or coverage < 0
                or not coverage.is_integer()
                or coverage > torch.iinfo(torch.int64).max
            ):
                raise InputContractError("bedMethyl coverage must be a finite non-negative integer")
            coverage_count = int(coverage)
            if coverage_count == 0:
                continue
            fraction = float(row[10])
            if not math.isfinite(fraction) or fraction < 0 or fraction > 100:
                raise InputContractError("bedMethyl fraction_modified must be in [0, 100]")
            values[indices] = fraction / 100.0
            observed[indices] = True
            coverage_by_cpg[indices] = coverage_count
        try:
            path_stat = input_path.stat()
        except FileNotFoundError:
            raise InputContractError("bedMethyl input changed while inference was running") from None
        if _stat_identity(os.fstat(handle.fileno())) != _stat_identity(initial_stat) or _stat_identity(
            path_stat
        ) != _stat_identity(initial_stat):
            raise InputContractError("bedMethyl input changed while inference was running")
    if int(observed.sum().item()) == 0:
        raise InputContractError("bedMethyl input did not match any release CpGs")
    sid = sample_id or input_path.name.replace(".bed.gz", "").replace(".bed", "")
    if not isinstance(sid, str) or not sid.strip():
        raise InputContractError("bedMethyl sample id must be nonempty")
    return [sid], values[None, :], observed[None, :], coverage_by_cpg[None, :]


def _input_paths(value: str | Path | Sequence[str | Path]) -> list[Path]:
    paths = [Path(value)] if isinstance(value, (str, Path)) else [Path(item) for item in value]
    if not paths:
        raise InputContractError("at least one inference input is required")
    invalid = [str(path) for path in paths if not path.is_file()]
    if invalid:
        raise InputContractError(f"inference input is not a file: {invalid[:5]}")
    return paths


def _bedmethyl_batches(
    paths: Sequence[Path],
    cpg: CpGManifest,
    *,
    batch_size: int,
) -> Iterator[tuple[list[str], torch.Tensor, torch.Tensor, torch.Tensor]]:
    sample_ids: list[str] = []
    beta_rows: list[torch.Tensor] = []
    observed_rows: list[torch.Tensor] = []
    uncertainty_rows: list[torch.Tensor] = []
    seen: set[str] = set()
    for path in paths:
        ids, beta, observed, coverage = load_bed_methyl_with_manifest(path, cpg)
        sample_id = ids[0]
        if sample_id in seen:
            raise InputContractError(f"duplicate BedMethyl sample ID: {sample_id}")
        seen.add(sample_id)
        presentation = real_coverage_presentation(beta, observed, coverage)
        sample_ids.append(sample_id)
        beta_rows.append(presentation.beta_input)
        observed_rows.append(presentation.input_observed)
        uncertainty_rows.append(presentation.uncertainty)
        if len(sample_ids) == batch_size:
            yield (
                sample_ids,
                torch.cat(beta_rows),
                torch.cat(observed_rows),
                torch.cat(uncertainty_rows),
            )
            sample_ids, beta_rows, observed_rows, uncertainty_rows = [], [], [], []
    if sample_ids:
        yield (
            sample_ids,
            torch.cat(beta_rows),
            torch.cat(observed_rows),
            torch.cat(uncertainty_rows),
        )


def run_inference(
    artifact: str | Path | None,
    input_path: str | Path | Sequence[str | Path],
    input_format: str,
    output: str | Path,
    *,
    device: str = "auto",
    embedding_sidecar: str | Path | None = None,
    batch_size: int = DEFAULT_INFERENCE_BATCH_SIZE,
) -> Path:
    if type(batch_size) is not int or batch_size <= 0:
        raise InputContractError("batch_size must be a positive integer")
    inputs = _input_paths(input_path)
    if input_format == "array-csv" and len(inputs) != 1:
        raise InputContractError("array-csv accepts exactly one input file")
    if input_format not in {"array-csv", "bedmethyl"}:
        raise InputContractError("input_format must be array-csv or bedmethyl")
    validate_new_external_outputs(
        None,
        {"inference output": output, "embedding sidecar": embedding_sidecar},
        inputs=inputs,
    )
    runtime = ALMA3(artifact, device=device)
    outputs = validate_new_external_outputs(
        runtime.artifact,
        {"inference output": output, "embedding sidecar": embedding_sidecar},
        inputs=inputs,
    )
    out = outputs["inference output"]
    sidecar_out = outputs.get("embedding sidecar")
    validated = runtime.validated_release
    hashes = validated["hashes"]
    if input_format == "array-csv":
        batches = (
            (sample_ids, x, observed, torch.zeros_like(x))
            for sample_ids, x, observed in _array_csv_batches(
                inputs[0],
                runtime.cpg,
                batch_size=batch_size,
            )
        )
    else:
        batches = _bedmethyl_batches(inputs, runtime.cpg, batch_size=batch_size)
    minimum_observed = runtime.minimum_observed_cpgs
    out.parent.mkdir(parents=True, exist_ok=True)
    sidecar_samples: list[dict[str, Any]] = []

    descriptor, raw_temporary = tempfile.mkstemp(prefix=f".{out.name}.", suffix=".tmp", dir=out.parent)
    temporary = Path(raw_temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            for sample_ids, x, observed, uncertainty in batches:
                clinical_results, embedding, observed_counts = runtime._predict_tensors(
                    sample_ids,
                    x,
                    observed,
                    uncertainty,
                )
                if sidecar_out is not None:
                    vectors = embedding.detach().to(device="cpu", dtype=torch.float32).tolist()
                    sidecar_samples.extend(
                        {
                            "sample_id": str(sample_id),
                            "observed_cpg_count": observed_count,
                            "embedding": vector,
                        }
                        for sample_id, observed_count, vector in zip(
                            sample_ids,
                            observed_counts,
                            vectors,
                            strict=True,
                        )
                    )
                for result in clinical_results:
                    handle.write(serialize_result(result) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        sidecar_temporary = None
        sidecar_identity = None
        try:
            if sidecar_out is not None:
                sidecar_payload = {
                    "kind": EMBEDDING_SIDECAR_KIND,
                    "schema_version": EMBEDDING_SIDECAR_SCHEMA_VERSION,
                    "release": runtime.sidecar_release_identity,
                    "representation": {
                        "name": DX_REPRESENTATION_NAME,
                        "version": DX_REPRESENTATION_VERSION,
                        "dtype": DX_REPRESENTATION_DTYPE,
                        "dimensions": DX_REPRESENTATION_DIMENSIONS,
                    },
                    "minimum_observed_cpgs": minimum_observed,
                    "samples": sidecar_samples,
                }
                validate_embedding_sidecar(sidecar_payload)
                sidecar_out.parent.mkdir(parents=True, exist_ok=True)
                sidecar_temporary = _json_temporary(sidecar_out, sidecar_payload)
            revalidate_release_identity(
                runtime.artifact,
                manifest_sha256=validated["manifest_sha256"],
                hashes=hashes,
            )
            if sidecar_out is not None:
                sidecar_identity = publish_new_file(sidecar_temporary, sidecar_out)
            try:
                publish_new_file(temporary, out)
            except BaseException:
                if sidecar_out is not None and sidecar_identity is not None:
                    _unlink_owned(sidecar_out, sidecar_identity)
                raise
        finally:
            if sidecar_temporary is not None:
                sidecar_temporary.unlink(missing_ok=True)
    finally:
        temporary.unlink(missing_ok=True)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="alma3 infer",
        description="Run ALMA3-Dx inference.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--artifact",
        help="release directory; otherwise use ALMA3_RELEASE, the verified cache, or automatic download",
    )
    parser.add_argument(
        "--input",
        required=True,
        action="append",
        help="input file; repeat for multiple BedMethyl samples",
    )
    parser.add_argument(
        "--format",
        required=True,
        choices=["array-csv", "bedmethyl"],
        help="input data format",
    )
    parser.add_argument("--output", required=True, help="new canonical ALMA3 JSONL output file")
    parser.add_argument(
        "--embedding-sidecar",
        help="optional new JSON file containing same-pass diagnostic embeddings",
    )
    parser.add_argument("--device", default="auto", help="inference device: auto, cpu, cuda, or cuda:<index>")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_INFERENCE_BATCH_SIZE,
        help="samples evaluated together",
    )
    args = parser.parse_args(argv)
    run_inference(
        args.artifact,
        args.input,
        args.format,
        args.output,
        device=args.device,
        embedding_sidecar=args.embedding_sidecar,
        batch_size=args.batch_size,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
