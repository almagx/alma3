from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from .config import DxConfig
from .data import CpGManifest
from .dx import (
    DX_REPRESENTATION_DIMENSIONS,
    DX_TARGETS,
    DxContractError,
    load_dx,
    load_taxonomy,
    load_thresholds,
)
from .hashes import sha256_file, verify_sha256_manifest
from .model import validate_chromosome_layout

RELEASE_PAYLOADS = frozenset(
    {
        "release.json",
        "config.json",
        "model.safetensors",
        "taxonomy.json",
        "cpg_manifest.json",
        "thresholds.json",
        "LICENSE",
    }
)
RELEASE_FILES = RELEASE_PAYLOADS | {"SHA256SUMS.json", "RELEASE_COMPLETE"}
RELEASE_KIND = "alma3_release"
RELEASE_SCHEMA_VERSION = 1
RELEASE_VERSION = "3.0.0"
RELEASE_ENV = "ALMA3_RELEASE"
RELEASE_LICENSE_HEADER = "ALMA3 LICENSE 1.0\n"
RELEASE_METADATA_FIELDS = frozenset(
    {
        "kind",
        "schema_version",
        "version",
        "runtime_git_commit",
        "source_release_manifest_sha256",
    }
)
CPG_MANIFEST_SCHEMA_VERSION = 1
CPG_MANIFEST_REQUIRED_FIELDS = frozenset(
    {
        "kind",
        "schema_version",
        "bundle_fingerprint",
        "cpg_manifest_sha256",
        "source_cpg_manifest_sha256",
        "selected_cpg_count",
        "selection_algorithm",
        "cpg_ids",
        "indices",
        "pos",
        "chrom",
        "start",
        "end",
    }
)
CPG_MANIFEST_OPTIONAL_FIELDS = frozenset({"chr_id"})


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_revision(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 40
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


def _read_object(path: Path, description: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DxContractError(f"invalid {description}: {path}") from error
    if not isinstance(payload, dict):
        raise DxContractError(f"{description} must be a JSON object: {path}")
    return payload


def _artifact_files(root: Path) -> set[str]:
    if not root.is_dir() or root.is_symlink():
        raise DxContractError(f"release artifact must be a real directory: {root}")
    paths = list(root.rglob("*"))
    if any(path.is_symlink() for path in paths):
        raise DxContractError(f"release artifact must not contain symbolic links: {root}")
    if any(path.is_dir() for path in paths):
        raise DxContractError(f"release artifact must not contain directories: {root}")
    return {path.relative_to(root).as_posix() for path in paths if path.is_file()}


def _validate_cpg_release_contract(payload: dict[str, Any], cpg: CpGManifest, config: DxConfig) -> None:
    fields = set(payload)
    if not CPG_MANIFEST_REQUIRED_FIELDS.issubset(fields) or not fields.issubset(
        CPG_MANIFEST_REQUIRED_FIELDS | CPG_MANIFEST_OPTIONAL_FIELDS
    ):
        raise DxContractError("release CpG manifest fields are invalid")
    row_count = len(cpg.cpg_ids)
    indices = payload.get("indices")
    selection_algorithm = payload.get("selection_algorithm")
    if (
        payload.get("kind") != "alma3_cpg_manifest"
        or payload.get("schema_version") != CPG_MANIFEST_SCHEMA_VERSION
        or not _is_sha256(payload.get("bundle_fingerprint"))
        or not _is_sha256(payload.get("cpg_manifest_sha256"))
        or not _is_sha256(payload.get("source_cpg_manifest_sha256"))
        or payload.get("selected_cpg_count") != row_count
        or config.foundation.n_cpgs != row_count
        or not isinstance(selection_algorithm, str)
        or not selection_algorithm
        or not isinstance(indices, list)
        or len(indices) != row_count
        or any(type(index) is not int for index in indices)
        or any(index < 0 for index in indices)
        or len(set(indices)) != row_count
        or cpg.chrom is None
        or cpg.start is None
        or cpg.arm_id is None
    ):
        raise DxContractError("release CpG manifest contract is invalid")


def revalidate_release_identity(
    artifact: str | Path,
    *,
    manifest_sha256: str,
    hashes: Mapping[str, str],
) -> None:
    root = Path(artifact).resolve()
    if _artifact_files(root) != RELEASE_FILES:
        raise DxContractError("release artifact changed during inference")
    if (root / "RELEASE_COMPLETE").read_bytes() != b"complete\n":
        raise DxContractError("release artifact changed during inference")
    if sha256_file(root / "SHA256SUMS.json") != manifest_sha256:
        raise DxContractError("release manifest changed during inference")
    if verify_sha256_manifest(root, required=RELEASE_PAYLOADS) != dict(hashes):
        raise DxContractError("release payloads changed during inference")


def validate_release(
    artifact: str | Path,
    *,
    device: torch.device | str = "cpu",
    load_model: bool = True,
) -> dict[str, Any]:
    artifact_path = Path(artifact)
    if artifact_path.is_symlink():
        raise DxContractError(f"release artifact must not be a symbolic link: {artifact_path}")
    root = artifact_path.resolve()
    files = _artifact_files(root)
    if files != RELEASE_FILES:
        missing = sorted(RELEASE_FILES - files)
        extra = sorted(files - RELEASE_FILES)
        raise DxContractError(f"release artifact file set is invalid; missing={missing}, extra={extra}")
    if (root / "RELEASE_COMPLETE").read_bytes() != b"complete\n":
        raise DxContractError(f"release is incomplete: {root / 'RELEASE_COMPLETE'}")
    hashes = verify_sha256_manifest(root, required=RELEASE_PAYLOADS)
    if set(hashes) != RELEASE_PAYLOADS:
        raise DxContractError("release SHA256 manifest entries are invalid")

    release_metadata = _read_object(root / "release.json", "release metadata")
    if (
        set(release_metadata) != RELEASE_METADATA_FIELDS
        or release_metadata.get("kind") != RELEASE_KIND
        or release_metadata.get("schema_version") != RELEASE_SCHEMA_VERSION
        or release_metadata.get("version") != RELEASE_VERSION
        or not _is_revision(release_metadata.get("runtime_git_commit"))
        or not _is_sha256(release_metadata.get("source_release_manifest_sha256"))
    ):
        raise DxContractError("release metadata contract is invalid")
    try:
        license_text = (root / "LICENSE").read_text(encoding="utf-8")
    except OSError as error:
        raise DxContractError("release license is unreadable") from error
    if not license_text.startswith(RELEASE_LICENSE_HEADER):
        raise DxContractError("release license must be ALMA3 License 1.0")

    config_payload = _read_object(root / "config.json", "release model config")
    if set(config_payload) != {"foundation", "targets", "hidden_dim", "dropout"}:
        raise DxContractError("release model config fields are invalid")
    config = DxConfig.from_dict(config_payload)
    if config.foundation.d_model != DX_REPRESENTATION_DIMENSIONS:
        raise DxContractError(
            f"released diagnostic embedding dimension must be {DX_REPRESENTATION_DIMENSIONS}"
        )

    taxonomy_payload = _read_object(root / "taxonomy.json", "release taxonomy")
    if set(taxonomy_payload) != {
        "kind",
        "schema_version",
        "generated_at",
        "bundle_fingerprint",
        "levels",
        "family_by_lineage",
        "type_by_family",
        "subtype_by_type",
    } or (
        taxonomy_payload["schema_version"] != 1
        or not isinstance(taxonomy_payload["generated_at"], str)
        or not taxonomy_payload["generated_at"]
        or not _is_sha256(taxonomy_payload["bundle_fingerprint"])
    ):
        raise DxContractError("release taxonomy fields are invalid")
    levels = taxonomy_payload.get("levels")
    if not isinstance(levels, dict) or set(levels) != set(DX_TARGETS):
        raise DxContractError("release taxonomy levels are invalid")
    taxonomy = load_taxonomy(root / "taxonomy.json")
    taxonomy.validate_sizes(config.targets)
    if taxonomy.classes["hematolymphoid_tumor_presence"] != ("absent", "present"):
        raise DxContractError("release tumor-presence taxonomy must be exactly absent, present")

    cpg_payload = _read_object(root / "cpg_manifest.json", "release CpG manifest")
    cpg = CpGManifest.load(root / "cpg_manifest.json")
    _validate_cpg_release_contract(cpg_payload, cpg, config)
    validate_chromosome_layout(config.foundation, cpg.chr_id, cpg.arm_id)

    thresholds = load_thresholds(root / "thresholds.json")
    expected_threshold_bindings = {
        "model_sha256": hashes["model.safetensors"],
        "taxonomy_sha256": hashes["taxonomy.json"],
        "cpg_manifest_sha256": hashes["cpg_manifest.json"],
    }
    mismatches = [
        field for field, expected in expected_threshold_bindings.items() if thresholds.get(field) != expected
    ]
    if (
        mismatches
        or thresholds.get("operating_policy_met") is not True
        or thresholds.get("evaluation_git_clean") is not True
    ):
        raise DxContractError(
            "release threshold bindings are invalid"
            + (f": {', '.join(sorted(mismatches))}" if mismatches else "")
        )

    validated = {
        "root": root,
        "hashes": hashes,
        "manifest_sha256": sha256_file(root / "SHA256SUMS.json"),
        "config": config,
        "taxonomy": taxonomy,
        "cpg": cpg,
        "thresholds": thresholds,
        "release": release_metadata,
    }
    if load_model:
        model = load_dx(root, device=device)
        taxonomy.validate_sizes(model.config.targets)
        if model.config != config:
            raise DxContractError("loaded model configuration differs from the validated release config")
        validated["model"] = model
    return validated


def load_release(
    explicit: str | Path | None = None,
    *,
    device: torch.device | str = "cpu",
    load_model: bool = True,
) -> dict[str, Any]:
    """Load an explicit release or the release configured by ALMA3_RELEASE."""

    raw_path = explicit if explicit is not None else os.environ.get(RELEASE_ENV)
    if not raw_path:
        raise DxContractError("no ALMA3 release configured; use --artifact or set ALMA3_RELEASE")
    return validate_release(
        Path(raw_path).expanduser(),
        device=device,
        load_model=load_model,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="alma3 verify-release",
        description="Verify an ALMA3 release and print its machine-readable identity.",
    )
    parser.add_argument(
        "--artifact",
        help="release directory; otherwise use ALMA3_RELEASE",
    )
    args = parser.parse_args(argv)
    validated = load_release(args.artifact, device="cpu")
    summary = {
        "kind": "alma3_release_verification",
        "schema_version": 1,
        "release_version": validated["release"]["version"],
        "manifest_sha256": validated["manifest_sha256"],
        "model_sha256": validated["hashes"]["model.safetensors"],
        "taxonomy_sha256": validated["hashes"]["taxonomy.json"],
        "cpg_manifest_sha256": validated["hashes"]["cpg_manifest.json"],
        "thresholds_sha256": validated["hashes"]["thresholds.json"],
        "minimum_observed_cpgs": validated["thresholds"]["minimum_observed_cpgs"],
        "representation_dimensions": validated["config"].foundation.d_model,
    }
    print(json.dumps(summary, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
