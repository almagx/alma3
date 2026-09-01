from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
import sys
import tempfile
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import torch

from .clinical_result import serialize_result, validate_result, validate_sample_id
from .data import CpGManifest
from .dx import (
    DX_REPRESENTATION_DIMENSIONS,
    DX_REPRESENTATION_DTYPE,
    DX_REPRESENTATION_NAME,
    DX_REPRESENTATION_VERSION,
)
from .hashes import publish_new_file, validate_new_external_outputs
from .release import revalidate_release_identity
from .runtime import (
    ALMA3,
    DEFAULT_BATCH_SIZE,
    _adjustment_message,
    _ArrayValueSummary,
    _prepare_array_values,
)
from .sitewise import real_coverage_presentation


class InputContractError(ValueError):
    """Raised when an inference input does not match the ALMA3 contract."""


DEFAULT_INFERENCE_BATCH_SIZE = DEFAULT_BATCH_SIZE
EMBEDDING_SIDECAR_KIND = "alma3_embedding_sidecar"
EMBEDDING_SIDECAR_SCHEMA_VERSION = 1
_RESULT_CSV_FIELDS = (
    "sample_id",
    "result_summary",
    "result_status",
    "resolved_level",
    "resolved_classification",
    "resolved_basis",
    "unresolved_level",
    "unresolved_reporting_cutoff",
    "differential_1_classification",
    "differential_1_model_score",
    "differential_2_classification",
    "differential_2_model_score",
    "observed_cpg_count",
    "minimum_observed_cpgs",
    "input_format",
    "input_value_mode",
    "input_clipped_value_count",
    "tumor_presence",
    "tumor_presence_status",
    "tumor_presence_model_score",
    "tumor_presence_reporting_cutoff",
    "lineage",
    "lineage_status",
    "lineage_model_score",
    "lineage_reporting_cutoff",
    "family",
    "family_status",
    "family_model_score",
    "family_reporting_cutoff",
    "type",
    "type_status",
    "type_model_score",
    "type_reporting_cutoff",
    "subtype",
    "subtype_status",
    "subtype_model_score",
    "subtype_reporting_cutoff",
    "result_kind",
    "result_schema_version",
    "release_version",
    "release_manifest_sha256",
    "runtime_package_version",
    "runtime_contract_sha256",
    "inference_device",
)


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
        try:
            validate_sample_id(sample_id)
        except ValueError as error:
            raise InputContractError(f"embedding sidecar {error}") from error
        if sample_id in seen:
            raise InputContractError("embedding sidecar sample IDs must be unique")
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


def _result_csv_row(result: dict[str, Any]) -> dict[str, Any]:
    validate_result(result)
    accepted = result["accepted"] or {}
    decision = result["decision"] or {}
    nodes = {node["level"]: node for node in result["path"]}
    accepted_node = result["path"][-1] if result["path"] else {}
    differential = decision.get("differential", [])
    row: dict[str, Any] = {
        "sample_id": result["sample_id"],
        "result_summary": result["result_summary"],
        "result_status": result["status"],
        "result_kind": result["kind"],
        "result_schema_version": result["schema_version"],
        "release_version": result["release"]["version"],
        "release_manifest_sha256": result["release"]["manifest_sha256"],
        "runtime_package_version": result["runtime"]["package_version"],
        "runtime_contract_sha256": result["runtime"]["contract_sha256"],
        "inference_device": result["runtime"]["device"],
        "input_format": result["input"]["format"],
        "input_value_mode": result["input"]["value_mode"],
        "input_clipped_value_count": result["input"]["clipped_value_count"],
        "observed_cpg_count": result["observed_cpg_count"],
        "minimum_observed_cpgs": result["minimum_observed_cpgs"],
        "resolved_level": accepted.get("level", ""),
        "resolved_classification": accepted.get("classification", ""),
        "resolved_basis": (
            "implied_by_hierarchy"
            if accepted_node.get("status") == "implied"
            else "scored"
            if accepted_node
            else ""
        ),
        "unresolved_level": decision.get("level", ""),
        "unresolved_reporting_cutoff": decision.get("reporting_cutoff", ""),
        "differential_1_classification": (
            differential[0]["classification"] if differential else ""
        ),
        "differential_1_model_score": (
            differential[0]["model_score"] if differential else ""
        ),
        "differential_2_classification": (
            differential[1]["classification"] if differential else ""
        ),
        "differential_2_model_score": (
            differential[1]["model_score"] if differential else ""
        ),
    }
    for level, prefix in (
        ("presence", "tumor_presence"),
        ("lineage", "lineage"),
        ("family", "family"),
        ("type", "type"),
        ("subtype", "subtype"),
    ):
        node = nodes.get(level, {})
        row[prefix] = node.get("classification", "")
        row[f"{prefix}_status"] = node.get("status", "")
        score = node.get("model_score")
        cutoff = node.get("reporting_cutoff")
        row[f"{prefix}_model_score"] = "" if score is None else score
        row[f"{prefix}_reporting_cutoff"] = "" if cutoff is None else cutoff
    if set(row) != set(_RESULT_CSV_FIELDS):
        raise InputContractError("CSV result projection fields are invalid")
    return row


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int]:
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns


def _array_csv_batches(
    path: str | Path,
    cpg: CpGManifest,
    *,
    batch_size: int = DEFAULT_INFERENCE_BATCH_SIZE,
    input_values: str = "beta",
    minimum_observed_cpgs: int = 1,
) -> Iterator[tuple[list[str], torch.Tensor, torch.Tensor, _ArrayValueSummary]]:
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
        recognized = sum(column is not None for column in cpg_columns)
        if recognized < minimum_observed_cpgs:
            raise InputContractError(
                f"found {recognized:,} ALMA3 CpGs; at least {minimum_observed_cpgs:,} are required. "
                "Gene-expression matrices are not supported."
            )
        sample_ids: list[str] = []
        seen_sample_ids: set[str] = set()
        values: list[list[float]] = []

        def prepared_batch() -> tuple[list[str], torch.Tensor, torch.Tensor, _ArrayValueSummary]:
            raw_values = torch.tensor(values, dtype=torch.float32)
            prepared, observed, summary = _prepare_array_values(
                raw_values,
                sample_ids,
                input_values=input_values,
            )
            return sample_ids, torch.where(observed, prepared, 0.0), observed, summary

        for line_no, row in enumerate(reader, start=2):
            if len(row) != len(header):
                raise InputContractError(f"array CSV row {line_no} has {len(row)} fields; expected {len(header)}")
            sample_id = row[0]
            try:
                validate_sample_id(sample_id)
            except ValueError as error:
                raise InputContractError(f"array CSV row {line_no} {error}") from error
            if sample_id in seen_sample_ids:
                raise InputContractError(f"array CSV has duplicate sample id: {sample_id}")
            seen_sample_ids.add(sample_id)
            sample_values: list[float] = []
            for cpg_id, column in zip(cpg.cpg_ids, cpg_columns, strict=True):
                if column is None:
                    sample_values.append(math.nan)
                    continue
                raw = row[1 + column].strip()
                if raw == "" or raw.lower() == "nan":
                    sample_values.append(math.nan)
                    continue
                try:
                    value = float(raw)
                except ValueError:
                    raise InputContractError(
                        f"array CSV row {line_no} has non-numeric value for CpG {cpg_id}: {raw!r}"
                    ) from None
                if not math.isfinite(value):
                    raise InputContractError(
                        f"array CSV row {line_no} has nonfinite value for CpG {cpg_id}: {raw!r}"
                    )
                sample_values.append(value)
            if not any(math.isfinite(value) for value in sample_values):
                raise InputContractError(f"array CSV row {line_no} has no observed CpGs")
            sample_ids.append(sample_id)
            values.append(sample_values)
            if len(values) == batch_size:
                yield prepared_batch()
                sample_ids, values = [], []
        if values:
            yield prepared_batch()
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


def load_array_csv(
    path: str | Path,
    cpg: CpGManifest,
    *,
    input_values: str = "beta",
) -> tuple[list[str], torch.Tensor, torch.Tensor]:
    batches = list(_array_csv_batches(path, cpg, input_values=input_values))
    return (
        [sample_id for sample_ids, _, _, _ in batches for sample_id in sample_ids],
        torch.cat([values for _, values, _, _ in batches]),
        torch.cat([observed for _, _, observed, _ in batches]),
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
    sid = (
        input_path.name.replace(".bed.gz", "").replace(".bed", "")
        if sample_id is None
        else sample_id
    )
    try:
        validate_sample_id(sid)
    except ValueError as error:
        raise InputContractError(f"bedMethyl {error}") from error
    return [sid], values[None, :], observed[None, :], coverage_by_cpg[None, :]


def _input_paths(value: str | Path | Sequence[str | Path]) -> list[Path]:
    paths = [Path(value)] if isinstance(value, (str, Path)) else [Path(item) for item in value]
    if not paths:
        raise InputContractError("at least one inference input is required")
    invalid = [str(path) for path in paths if not path.is_file()]
    if invalid:
        raise InputContractError(f"inference input is not a file: {invalid[:5]}")
    return paths


def infer_input_format(value: str | Path | Sequence[str | Path]) -> str:
    paths = [Path(value)] if isinstance(value, (str, Path)) else [Path(item) for item in value]
    formats: set[str] = set()
    invalid: list[str] = []
    for path in paths:
        name = path.name.lower()
        if name.endswith((".bed", ".bed.gz")):
            formats.add("bedmethyl")
        elif name.endswith((".csv", ".csv.gz")):
            formats.add("array-csv")
        else:
            invalid.append(str(path))
    if invalid:
        raise InputContractError(
            f"cannot infer input format from filename: {invalid[:5]}; use --format array-csv or bedmethyl"
        )
    if len(formats) != 1:
        raise InputContractError("inference inputs use mixed formats; provide one format per command")
    return formats.pop()


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
    input_values: str = "beta",
    progress: bool = False,
) -> Path:
    if type(batch_size) is not int or batch_size <= 0:
        raise InputContractError("batch_size must be a positive integer")
    output_suffix = Path(output).suffix.lower()
    if output_suffix not in {".csv", ".jsonl"}:
        raise InputContractError("output filename must end in .jsonl or .csv")
    inputs = _input_paths(input_path)
    if input_format == "array-csv" and len(inputs) != 1:
        raise InputContractError("array-csv accepts exactly one input file")
    if input_format not in {"array-csv", "bedmethyl"}:
        raise InputContractError("input_format must be array-csv or bedmethyl")
    if input_values not in {"beta", "mvalue"}:
        raise InputContractError("input_values must be beta or mvalue")
    if input_format != "array-csv" and input_values != "beta":
        raise InputContractError("--input-values applies only to array-csv input")
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
            (sample_ids, x, observed, torch.zeros_like(x), summary)
            for sample_ids, x, observed, summary in _array_csv_batches(
                inputs[0],
                runtime.cpg,
                batch_size=batch_size,
                input_values=input_values,
                minimum_observed_cpgs=runtime.minimum_observed_cpgs,
            )
        )
    else:
        batches = (
            (*batch, _ArrayValueSummary(0, 0, (0,) * len(batch[0])))
            for batch in _bedmethyl_batches(inputs, runtime.cpg, batch_size=batch_size)
        )
    minimum_observed = runtime.minimum_observed_cpgs
    out.parent.mkdir(parents=True, exist_ok=True)
    sidecar_samples: list[dict[str, Any]] = []
    adjustment_observed = 0
    adjustment_clipped = 0
    processed = 0

    descriptor, raw_temporary = tempfile.mkstemp(prefix=f".{out.name}.", suffix=".tmp", dir=out.parent)
    temporary = Path(raw_temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            csv_writer = None
            if output_suffix == ".csv":
                csv_writer = csv.DictWriter(handle, fieldnames=_RESULT_CSV_FIELDS, lineterminator="\n")
                csv_writer.writeheader()
            for sample_ids, x, observed, uncertainty, adjustment in batches:
                adjustment_observed += adjustment.observed
                adjustment_clipped += adjustment.clipped
                input_metadata = [
                    {
                        "format": input_format,
                        "value_mode": (
                            input_values if input_format == "array-csv" else "fraction_modified"
                        ),
                        "clipped_value_count": clipped,
                    }
                    for clipped in adjustment.clipped_by_sample
                ]
                clinical_results, embedding, observed_counts = runtime._predict_tensors(
                    sample_ids,
                    x,
                    observed,
                    uncertainty,
                    input_metadata,
                )
                if sidecar_out is not None:
                    vectors = embedding.detach().to(device="cpu", dtype=torch.float32).tolist()
                    sidecar_samples.extend(
                        {
                            "sample_id": sample_id,
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
                    if csv_writer is None:
                        handle.write(serialize_result(result) + "\n")
                    else:
                        csv_writer.writerow(_result_csv_row(result))
                processed += len(sample_ids)
                if progress:
                    print(f"Processed {processed:,} samples.", file=sys.stderr)
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
    if adjustment_clipped:
        print(
            _adjustment_message(
                _ArrayValueSummary(adjustment_observed, adjustment_clipped, ())
            ),
            file=sys.stderr,
        )
    if progress:
        print(f"Results saved to {out}", file=sys.stderr)
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
        "-i",
        "--input",
        required=True,
        action="append",
        help="input file; repeat for multiple BedMethyl samples",
    )
    parser.add_argument(
        "--format",
        choices=["array-csv", "bedmethyl"],
        help="input data format; inferred from .bed(.gz) or .csv(.gz) when omitted",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="new ALMA3 output ending in .jsonl or .csv",
    )
    parser.add_argument(
        "--input-values",
        default="beta",
        choices=["beta", "mvalue"],
        help="array values supplied as beta values or explicit M-values",
    )
    parser.add_argument(
        "--embedding-sidecar",
        help="optional new JSON file containing same-pass diagnostic embeddings",
    )
    parser.add_argument("--device", default="auto", help="inference device: auto, cpu, cuda, or cuda:<index>")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_INFERENCE_BATCH_SIZE,
        help="advanced override for samples evaluated together",
    )
    args = parser.parse_args(argv)
    input_format = args.format or infer_input_format(args.input)
    progress = sys.stderr.isatty()
    if progress:
        print("Loading ALMA3...", file=sys.stderr)
    run_inference(
        args.artifact,
        args.input,
        input_format,
        args.output,
        device=args.device,
        embedding_sidecar=args.embedding_sidecar,
        batch_size=args.batch_size,
        input_values=args.input_values,
        progress=progress,
    )
    return 0


def demo_path() -> Path:
    return Path(__file__).with_name("examples") / "example_dataset.csv.gz"


def demo_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="alma3 demo",
        description="Run the packaged ALMA3 example.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--artifact",
        help="release directory; otherwise use ALMA3_RELEASE, the verified cache, or automatic download",
    )
    parser.add_argument("-o", "--output", default="alma3-demo.jsonl", help="new demo JSONL output file")
    parser.add_argument("--device", default="auto", help="inference device: auto, cpu, cuda, or cuda:<index>")
    args = parser.parse_args(argv)
    progress = sys.stderr.isatty()
    if progress:
        print("Loading ALMA3...", file=sys.stderr)
    run_inference(
        args.artifact,
        demo_path(),
        "array-csv",
        args.output,
        device=args.device,
        batch_size=DEFAULT_INFERENCE_BATCH_SIZE,
        input_values="beta",
        progress=progress,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
