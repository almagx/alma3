from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from .config import DxConfig
from .data import CpGManifest
from .dx import DX_TARGETS, DxContractError, load_dx, load_taxonomy, load_thresholds
from .hashes import sha256_file, verify_sha256_manifest
from .model import validate_chromosome_layout


RELEASE_PAYLOADS = frozenset(
    {
        "config.json",
        "model.safetensors",
        "taxonomy.json",
        "cpg_manifest.json",
        "thresholds.json",
        "release_provenance.json",
    }
)
RELEASE_FILES = RELEASE_PAYLOADS | {"SHA256SUMS.json", "RELEASE_COMPLETE"}
RELEASE_PROVENANCE_KIND = "alma3_dx_release_provenance"
RELEASE_PROVENANCE_SCHEMA_VERSIONS = frozenset({7, 8})
RELEASE_PROVENANCE_CORE_FIELDS = frozenset(
    {
        "kind",
        "schema_version",
        "generated_at",
        "evaluation_git_commit",
        "training_manifest_sha256",
        "evaluation_freeze_sha256",
        "thresholds_sha256",
        "selection_calibration_sha256",
        "input_contract",
    }
)
PROVENANCE_SHA256_FIELDS = frozenset(
    {
        "training_manifest_sha256",
        "evaluation_freeze_sha256",
        "thresholds_sha256",
        "selection_calibration_sha256",
    }
)


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
        and len(value) in {40, 64}
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


def validate_release(
    artifact: str | Path,
    *,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    root = Path(artifact).resolve()
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

    config_payload = _read_object(root / "config.json", "release model config")
    if set(config_payload) != {"foundation", "targets", "hidden_dim", "dropout"}:
        raise DxContractError("release model config fields are invalid")
    config = DxConfig.from_dict(config_payload)

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

    cpg = CpGManifest.load(root / "cpg_manifest.json")
    validate_chromosome_layout(config.foundation, cpg.chr_id, cpg.arm_id)

    thresholds = load_thresholds(root / "thresholds.json")
    provenance = _read_object(root / "release_provenance.json", "release provenance")
    if (
        not RELEASE_PROVENANCE_CORE_FIELDS.issubset(provenance)
        or provenance.get("kind") != RELEASE_PROVENANCE_KIND
        or provenance.get("schema_version") not in RELEASE_PROVENANCE_SCHEMA_VERSIONS
        or not _is_revision(provenance.get("evaluation_git_commit"))
        or any(not _is_sha256(provenance.get(field)) for field in PROVENANCE_SHA256_FIELDS)
        or provenance.get("thresholds_sha256") != hashes["thresholds.json"]
        or provenance.get("input_contract") != "per_cpg_uncertainty_v1"
    ):
        raise DxContractError("release provenance contract is invalid")
    if provenance["schema_version"] == 8 and (
        not _is_revision(provenance.get("training_git_commit"))
        or not _is_revision(provenance.get("runtime_git_commit"))
        or len(provenance["training_git_commit"]) != 40
        or len(provenance["runtime_git_commit"]) != 40
        or provenance["training_git_commit"] != provenance["evaluation_git_commit"]
    ):
        raise DxContractError("release source provenance is invalid")
    expected_threshold_bindings = {
        "model_sha256": hashes["model.safetensors"],
        "taxonomy_sha256": hashes["taxonomy.json"],
        "cpg_manifest_sha256": hashes["cpg_manifest.json"],
        "training_manifest_sha256": provenance["training_manifest_sha256"],
        "selection_calibration_sha256": provenance["selection_calibration_sha256"],
    }
    mismatches = [
        field for field, expected in expected_threshold_bindings.items() if thresholds.get(field) != expected
    ]
    if (
        mismatches
        or thresholds.get("operating_policy_met") is not True
        or thresholds.get("evaluation_git_clean") is not True
        or (
            provenance["schema_version"] == 8
            and thresholds.get("evaluation_git_commit") != provenance["training_git_commit"]
        )
    ):
        raise DxContractError(
            "release threshold bindings are invalid"
            + (f": {', '.join(sorted(mismatches))}" if mismatches else "")
        )

    model = load_dx(root, device=device)
    taxonomy.validate_sizes(model.config.targets)
    if model.config != config:
        raise DxContractError("loaded model configuration differs from the validated release config")
    return {
        "root": root,
        "hashes": hashes,
        "manifest_sha256": sha256_file(root / "SHA256SUMS.json"),
        "config": config,
        "taxonomy": taxonomy,
        "cpg": cpg,
        "thresholds": thresholds,
        "provenance": provenance,
        "model": model,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="alma3 verify-release",
        description="Verify a complete, checksum-bound ALMA3 release artifact.",
    )
    parser.add_argument("--artifact", required=True)
    args = parser.parse_args(argv)
    validated = validate_release(args.artifact, device="cpu")
    summary = {
        "kind": "alma3_release_verification",
        "schema_version": 1,
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
