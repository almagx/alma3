from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file
from torch import nn

from alma3.config import CHROMOSOME_ARM_NAMES, DxConfig, FoundationConfig
from alma3.data import CpGManifest
from alma3.dx import (
    CALIBRATION_PRECISION_FLOOR,
    CALIBRATION_RULE,
    DX_TARGETS,
    THRESHOLDS_SCHEMA_VERSION,
    DiagnosticModel,
    load_taxonomy,
    load_thresholds,
)
from alma3.hashes import sha256_file
from alma3.release import RELEASE_LICENSE_HEADER
from alma3.sitewise import CANONICAL_BASE_SEED, CANONICAL_CONDITIONS, MINIMUM_RUNTIME_INPUT_CPGS


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def foundation_config() -> FoundationConfig:
    arm_counts = {name: 0 for name in CHROMOSOME_ARM_NAMES}
    arm_counts["chr1p"] = MINIMUM_RUNTIME_INPUT_CPGS
    return FoundationConfig(
        architecture_version=5,
        n_cpgs=MINIMUM_RUNTIME_INPUT_CPGS,
        chromosome_cpg_counts=[MINIMUM_RUNTIME_INPUT_CPGS] + [0] * 23,
        arm_cpg_counts=arm_counts,
        d_model=8,
        n_layers=1,
        n_heads=2,
        mlp_ratio=2,
        patch_size=64,
        pos_bands=12,
        chr_dim=32,
        cpg_id_dim=32,
        dropout=0.0,
    )


def release_foundation_config() -> FoundationConfig:
    base = foundation_config()
    return FoundationConfig(
        **{
            **base.to_dict(),
            "d_model": 1536,
            "n_heads": 24,
            "mlp_ratio": 1,
        }
    )


class FixtureFoundation(nn.Module):
    def __init__(self, config: FoundationConfig):
        super().__init__()
        self.config = config

    def embed(
        self,
        beta: torch.Tensor,
        observed: torch.Tensor,
        uncertainty: torch.Tensor,
        chr_id: torch.Tensor,
        pos: torch.Tensor,
    ) -> torch.Tensor:
        del uncertainty, chr_id, pos
        pooled = (beta * observed).sum(dim=1, keepdim=True) / observed.sum(dim=1, keepdim=True).clamp_min(1)
        return pooled.expand(-1, self.config.d_model).contiguous()


def taxonomy_payload() -> dict[str, object]:
    return {
        "kind": "alma3_taxonomy",
        "schema_version": 1,
        "generated_at": "20260827T000000Z",
        "bundle_fingerprint": "3" * 64,
        "levels": {
            "hematolymphoid_tumor_presence": ["absent", "present"],
            "lineage": ["myeloid", "lymphoid"],
            "family": ["myeloid_family", "lymphoid_family"],
            "type": ["myeloid_type", "lymphoid_type"],
            "subtype": ["myeloid_subtype", "lymphoid_subtype"],
        },
        "family_by_lineage": {
            "myeloid": ["myeloid_family"],
            "lymphoid": ["lymphoid_family"],
        },
        "type_by_family": {
            "myeloid_family": ["myeloid_type"],
            "lymphoid_family": ["lymphoid_type"],
        },
        "subtype_by_type": {
            "myeloid_type": ["myeloid_subtype"],
            "lymphoid_type": ["lymphoid_subtype"],
        },
    }


def _wilson_bounds(correct: int, total: int) -> tuple[float, float]:
    z = 1.959963984540054
    proportion = correct / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    margin = z * math.sqrt(
        proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)
    ) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def threshold_payload(bindings: dict[str, str]) -> dict[str, object]:
    condition_names = [name for name, _ in CANONICAL_CONDITIONS]

    def condition_metric(target: str) -> dict[str, object]:
        lower, upper = _wilson_bounds(30, 30)
        return {
            "threshold": 0.0,
            "precision": 1.0,
            "wilson_lower_95": lower,
            "wilson_upper_95": upper,
            "recall": 1.0,
            "coverage": 1.0,
            "called": 30,
            "correct": 30,
            "false_positive": 0,
            "labeled": 30,
            "precision_floor": CALIBRATION_PRECISION_FLOOR[target],
            "minimum_calls": 30,
            "precision_met": True,
        }

    def aggregate_metric(target: str) -> dict[str, object]:
        conditions = {name: condition_metric(target) for name in condition_names}
        return {
            "threshold": 0.0,
            "precision_floor": CALIBRATION_PRECISION_FLOOR[target],
            "minimum_calls_per_condition": 30,
            "precision_met": True,
            "worst_condition_coverage": 1.0,
            "total_called": 30 * len(condition_names),
            "total_correct": 30 * len(condition_names),
            "total_false_positive": 0,
            "total_labeled": 30 * len(condition_names),
            "conditions": conditions,
        }

    return {
        "kind": "alma3_dx_thresholds",
        "schema_version": THRESHOLDS_SCHEMA_VERSION,
        "generated_at": "20260827T000000Z",
        "status": "calibrated",
        "calibration_rule": CALIBRATION_RULE,
        "conditions": condition_names,
        "calibration_precision_floor": dict(CALIBRATION_PRECISION_FLOOR),
        "operating_policy_met": True,
        "availability": {target: True for target in DX_TARGETS},
        "thresholds": {target: 0.0 for target in DX_TARGETS},
        "temperatures": {target: 1.0 for target in DX_TARGETS},
        "temperature_metrics": {
            target: {
                "examples": len(condition_names),
                "examples_by_condition": {name: 1 for name in condition_names},
                "nll_before": 1.0,
                "nll_after": 1.0,
            }
            for target in DX_TARGETS
        },
        "metrics": {target: aggregate_metric(target) for target in DX_TARGETS},
        "calibration_seed": CANONICAL_BASE_SEED,
        "minimum_observed_cpgs": MINIMUM_RUNTIME_INPUT_CPGS,
        "model_sha256": bindings["model_sha256"],
        "taxonomy_sha256": bindings["taxonomy_sha256"],
        "cpg_manifest_sha256": bindings["cpg_manifest_sha256"],
        "dx_view_sha256": "4" * 64,
        "training_manifest_sha256": "5" * 64,
        "canonical_evaluator_sha256": "6" * 64,
        "selection_calibration_sha256": "7" * 64,
        "evaluation_git_commit": "8" * 40,
        "evaluation_git_clean": True,
    }


def create_release(root: Path) -> tuple[Path, DiagnosticModel]:
    root.mkdir()
    torch.manual_seed(27)
    foundation = release_foundation_config()
    config = DxConfig(
        foundation=foundation,
        targets={target: 2 for target in DX_TARGETS},
        hidden_dim=16,
        dropout=0.0,
    )
    model = DiagnosticModel(FixtureFoundation(foundation), config, freeze_foundation=False).eval()
    write_json(root / "config.json", config.to_dict())
    save_file(model.state_dict(), str(root / "model.safetensors"))
    write_json(root / "taxonomy.json", taxonomy_payload())
    cpg_ids = [f"cg{index:07d}" for index in range(MINIMUM_RUNTIME_INPUT_CPGS)]
    write_json(
        root / "cpg_manifest.json",
        {
            "kind": "alma3_cpg_manifest",
            "schema_version": 1,
            "bundle_fingerprint": "3" * 64,
            "cpg_ids": cpg_ids,
            "cpg_manifest_sha256": hashlib.sha256(("\n".join(cpg_ids) + "\n").encode()).hexdigest(),
            "source_cpg_manifest_sha256": hashlib.sha256(
                ("\n".join(cpg_ids) + "\n").encode()
            ).hexdigest(),
            "selected_cpg_count": MINIMUM_RUNTIME_INPUT_CPGS,
            "selection_algorithm": "parent_conditioned_tumor_dataset_stability_v3",
            "indices": list(range(MINIMUM_RUNTIME_INPUT_CPGS)),
            "chr_id": [0] * MINIMUM_RUNTIME_INPUT_CPGS,
            "chrom": ["chr1"] * MINIMUM_RUNTIME_INPUT_CPGS,
            "start": list(range(100, 100 + MINIMUM_RUNTIME_INPUT_CPGS)),
            "end": list(range(101, 101 + MINIMUM_RUNTIME_INPUT_CPGS)),
            "pos": [index / (MINIMUM_RUNTIME_INPUT_CPGS - 1) for index in range(MINIMUM_RUNTIME_INPUT_CPGS)],
        },
    )
    bindings = {
        "model_sha256": sha256(root / "model.safetensors"),
        "taxonomy_sha256": sha256(root / "taxonomy.json"),
        "cpg_manifest_sha256": sha256(root / "cpg_manifest.json"),
    }
    thresholds = threshold_payload(bindings)
    write_json(root / "thresholds.json", thresholds)
    write_json(
        root / "release.json",
        {
            "kind": "alma3_release",
            "schema_version": 1,
            "version": "3.0.0",
            "runtime_git_commit": "a" * 40,
            "source_release_manifest_sha256": "b" * 64,
        },
    )
    (root / "LICENSE").write_text(
        f"{RELEASE_LICENSE_HEADER}\nSynthetic test license\n",
        encoding="utf-8",
    )
    names = (
        "config.json",
        "model.safetensors",
        "taxonomy.json",
        "cpg_manifest.json",
        "thresholds.json",
        "release.json",
        "LICENSE",
    )
    write_json(root / "SHA256SUMS.json", {name: sha256(root / name) for name in names})
    (root / "RELEASE_COMPLETE").write_text("complete\n", encoding="utf-8")
    return root, model


def validated_release_fixture(root: Path, model: DiagnosticModel) -> dict[str, Any]:
    hashes = json.loads((root / "SHA256SUMS.json").read_text(encoding="utf-8"))
    return {
        "root": root.resolve(),
        "hashes": hashes,
        "manifest_sha256": sha256_file(root / "SHA256SUMS.json"),
        "config": model.config,
        "taxonomy": load_taxonomy(root / "taxonomy.json"),
        "cpg": CpGManifest.load(root / "cpg_manifest.json"),
        "thresholds": load_thresholds(root / "thresholds.json"),
        "release": json.loads((root / "release.json").read_text(encoding="utf-8")),
        "model": model,
    }


def write_array_csv(
    path: Path,
    *,
    sample_count: int = 3,
    observed: int = MINIMUM_RUNTIME_INPUT_CPGS,
    omit_unobserved_columns: bool = False,
) -> None:
    cpg_ids = [f"cg{index:07d}" for index in range(MINIMUM_RUNTIME_INPUT_CPGS)]
    if omit_unobserved_columns:
        cpg_ids = cpg_ids[:observed]
    rows = [",".join(["sample_id", *cpg_ids])]
    for sample_index in range(sample_count):
        values = ["0.5" if index < observed else "" for index in range(len(cpg_ids))]
        rows.append(",".join([f"sample-{sample_index + 1}", *values]))
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def write_bedmethyl(
    path: Path,
    *,
    fraction_modified: float = 50.0,
    modification_code: str = "C",
) -> None:
    rows = [
        f"chr1\t{start}\t{start + 1}\t{modification_code}\t0\t.\t{start}\t{start + 1}\t0\t10\t{fraction_modified}"
        for start in range(100, 100 + MINIMUM_RUNTIME_INPUT_CPGS)
    ]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
