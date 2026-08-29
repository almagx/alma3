from __future__ import annotations

import json
import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch

from .dx import DX_TARGETS, PRESENT_LABEL, DxContractError, Taxonomy, apply_temperatures
from .release import RELEASE_VERSION

RESULT_KIND = "alma3_dx_result"
RESULT_SCHEMA_VERSION = 1
RESULT_STATUSES = ("no_call", "tumor_not_detected", "classified", "partially_resolved")
RESULT_LEVELS = ("presence", "lineage", "family", "type", "subtype")
TARGET_BY_LEVEL = {
    "presence": "hematolymphoid_tumor_presence",
    "lineage": "lineage",
    "family": "family",
    "type": "type",
    "subtype": "subtype",
}
LEVEL_BY_TARGET = {target: level for level, target in TARGET_BY_LEVEL.items()}
_RELEASE_HASH_FIELDS = {
    "manifest_sha256",
    "model_sha256",
    "taxonomy_sha256",
    "cpg_manifest_sha256",
    "thresholds_sha256",
}
_RELEASE_FIELDS = {"version", *_RELEASE_HASH_FIELDS}


def result_schema_path() -> Path:
    return Path(__file__).with_name("schemas") / "dx_result.schema.json"


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def _rank_candidates(
    probabilities: torch.Tensor,
    indices: list[int],
    *,
    limit: int,
) -> list[tuple[int, float]]:
    return sorted(
        ((idx, float(probabilities[idx].detach().cpu())) for idx in indices),
        key=lambda item: (-item[1], item[0]),
    )[:limit]


def _child_indices(level: str, parent_idx: int | None, taxonomy: Taxonomy) -> list[int]:
    if level == "lineage":
        return list(range(len(taxonomy.classes["lineage"])))
    if parent_idx is None:
        raise DxContractError(f"{level} decision requires an accepted parent")
    if level == "family":
        mask = taxonomy.valid_family_mask(parent_idx)
    elif level == "type":
        mask = taxonomy.valid_type_mask(parent_idx)
    elif level == "subtype":
        mask = taxonomy.valid_subtype_mask(parent_idx)
    else:
        raise DxContractError(f"unsupported result level: {level}")
    return mask.nonzero(as_tuple=False).flatten().tolist()


def _path_node(
    level: str,
    status: str,
    index: int,
    score: float | None,
    threshold: float | None,
    taxonomy: Taxonomy,
) -> dict[str, Any]:
    return {
        "level": level,
        "status": status,
        "index": index,
        "classification": taxonomy.classes[TARGET_BY_LEVEL[level]][index],
        "model_score": score,
        "reporting_cutoff": threshold,
    }


def _unresolved_decision(
    level: str,
    threshold: float,
    ranked: list[tuple[int, float]],
    taxonomy: Taxonomy,
) -> dict[str, Any]:
    if len(ranked) != 2:
        raise DxContractError(f"{level} unresolved decision requires a two-class differential")
    labels = taxonomy.classes[TARGET_BY_LEVEL[level]]
    return {
        "level": level,
        "status": "unresolved",
        "reporting_cutoff": threshold,
        "differential": [
            {"index": index, "classification": labels[index], "model_score": score}
            for index, score in ranked
        ],
    }


def _result(
    sample_id: str,
    release: dict[str, str],
    observed_cpg_count: int,
    minimum_observed_cpgs: int,
    status: str,
    path: list[dict[str, Any]],
    decision: dict[str, Any] | None,
) -> dict[str, Any]:
    accepted = (
        None
        if not path
        else {key: path[-1][key] for key in ("level", "index", "classification")}
    )
    result = {
        "kind": RESULT_KIND,
        "schema_version": RESULT_SCHEMA_VERSION,
        "sample_id": sample_id,
        "release": dict(release),
        "observed_cpg_count": observed_cpg_count,
        "minimum_observed_cpgs": minimum_observed_cpgs,
        "status": status,
        "accepted": accepted,
        "path": path,
        "decision": decision,
    }
    validate_result(result)
    return result


def results_from_logits(
    sample_ids: Iterable[str],
    logits: dict[str, torch.Tensor],
    thresholds: dict[str, Any],
    taxonomy: Taxonomy,
    release: dict[str, str],
    *,
    observed_cpg_counts: Iterable[int],
    minimum_observed_cpgs: int,
) -> list[dict[str, Any]]:
    _validate_release(release)
    scaled = apply_temperatures(logits, thresholds, taxonomy)
    ids = [str(sample_id) for sample_id in sample_ids]
    observed_counts = list(observed_cpg_counts)
    batch = next(iter(scaled.values())).shape[0]
    if (
        len(ids) != batch
        or any(not sample_id for sample_id in ids)
        or len(ids) != len(set(ids))
        or len(observed_counts) != batch
        or type(minimum_observed_cpgs) is not int
        or minimum_observed_cpgs <= 0
        or any(type(count) is not int or count < minimum_observed_cpgs for count in observed_counts)
    ):
        raise DxContractError("result sample IDs must be non-empty, unique, and aligned with logits")

    stops = {LEVEL_BY_TARGET[target]: float(thresholds["thresholds"][target]) for target in DX_TARGETS}
    present_idx = taxonomy.classes["hematolymphoid_tumor_presence"].index(PRESENT_LABEL)
    results = []
    for row_idx, (sample_id, observed_cpg_count) in enumerate(
        zip(ids, observed_counts, strict=True)
    ):
        path: list[dict[str, Any]] = []
        presence = torch.softmax(scaled["hematolymphoid_tumor_presence"][row_idx].float(), dim=-1)
        presence_ranked = _rank_candidates(presence, list(range(len(presence))), limit=2)
        presence_pred, presence_score = presence_ranked[0]
        if presence_score < stops["presence"]:
            results.append(
                _result(
                    sample_id,
                    release,
                    observed_cpg_count,
                    minimum_observed_cpgs,
                    "no_call",
                    path,
                    _unresolved_decision(
                        "presence",
                        stops["presence"],
                        presence_ranked,
                        taxonomy,
                    ),
                )
            )
            continue
        path.append(
            _path_node(
                "presence",
                "resolved",
                presence_pred,
                presence_score,
                stops["presence"],
                taxonomy,
            )
        )
        if presence_pred != present_idx:
            results.append(
                _result(
                    sample_id,
                    release,
                    observed_cpg_count,
                    minimum_observed_cpgs,
                    "tumor_not_detected",
                    path,
                    None,
                )
            )
            continue

        parent_idx: int | None = presence_pred
        terminal = False
        for level in RESULT_LEVELS[1:]:
            indices = _child_indices(level, None if level == "lineage" else parent_idx, taxonomy)
            if not indices:
                terminal = True
                break
            if len(indices) == 1:
                parent_idx = indices[0]
                path.append(_path_node(level, "implied", parent_idx, None, None, taxonomy))
                continue

            target = TARGET_BY_LEVEL[level]
            mask = torch.zeros_like(scaled[target][row_idx], dtype=torch.bool)
            mask[indices] = True
            probabilities = torch.softmax(
                scaled[target][row_idx].float().masked_fill(~mask, -torch.inf), dim=-1
            )
            ranked = _rank_candidates(probabilities, indices, limit=2)
            pred, score = ranked[0]
            if score < stops[level]:
                results.append(
                    _result(
                        sample_id,
                        release,
                        observed_cpg_count,
                        minimum_observed_cpgs,
                        "partially_resolved",
                        path,
                        _unresolved_decision(level, stops[level], ranked, taxonomy),
                    )
                )
                break
            path.append(_path_node(level, "resolved", pred, score, stops[level], taxonomy))
            parent_idx = pred
        else:
            terminal = True
        if terminal:
            results.append(
                _result(
                    sample_id,
                    release,
                    observed_cpg_count,
                    minimum_observed_cpgs,
                    "classified",
                    path,
                    None,
                )
            )
    return results


def _validate_release(release: Any) -> None:
    if not isinstance(release, dict) or set(release) != _RELEASE_FIELDS:
        raise DxContractError("result release fields are invalid")
    if release["version"] != RELEASE_VERSION:
        raise DxContractError("result release version is invalid")
    for field in _RELEASE_HASH_FIELDS:
        if not _is_sha256(release[field]):
            raise DxContractError(f"result release {field} must be a lowercase SHA-256")


def _validate_score(value: Any, description: str, *, nullable: bool = False) -> None:
    if nullable and value is None:
        return
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) <= 1.0
    ):
        raise DxContractError(f"{description} must be finite in [0, 1]")


def _validate_identity_node(node: Any, description: str) -> None:
    if not isinstance(node, dict) or set(node) != {"level", "index", "classification"}:
        raise DxContractError(f"{description} fields are invalid")
    if node["level"] not in RESULT_LEVELS:
        raise DxContractError(f"{description} level is invalid")
    if type(node["index"]) is not int or node["index"] < 0:
        raise DxContractError(f"{description} index is invalid")
    if not isinstance(node["classification"], str) or not node["classification"]:
        raise DxContractError(f"{description} classification is invalid")


def _validate_differential_entry(entry: Any, description: str) -> None:
    if not isinstance(entry, dict) or set(entry) != {"index", "classification", "model_score"}:
        raise DxContractError(f"{description} fields are invalid")
    if type(entry["index"]) is not int or entry["index"] < 0:
        raise DxContractError(f"{description} index is invalid")
    if not isinstance(entry["classification"], str) or not entry["classification"]:
        raise DxContractError(f"{description} classification is invalid")
    _validate_score(entry["model_score"], f"{description} model score")


def validate_result(result: Any) -> None:
    fields = {
        "kind",
        "schema_version",
        "sample_id",
        "release",
        "observed_cpg_count",
        "minimum_observed_cpgs",
        "status",
        "accepted",
        "path",
        "decision",
    }
    if not isinstance(result, dict) or set(result) != fields:
        raise DxContractError("Dx result fields are invalid")
    if result["kind"] != RESULT_KIND or result["schema_version"] != RESULT_SCHEMA_VERSION:
        raise DxContractError("Dx result kind or schema version is invalid")
    if not isinstance(result["sample_id"], str) or not result["sample_id"]:
        raise DxContractError("Dx result sample_id is invalid")
    _validate_release(result["release"])
    observed_count = result["observed_cpg_count"]
    minimum_observed = result["minimum_observed_cpgs"]
    if (
        type(observed_count) is not int
        or type(minimum_observed) is not int
        or minimum_observed <= 0
        or observed_count < minimum_observed
    ):
        raise DxContractError("Dx result observed-CpG support is invalid")
    if result["status"] not in RESULT_STATUSES:
        raise DxContractError("Dx result status is invalid")

    path = result["path"]
    if not isinstance(path, list) or len(path) > len(RESULT_LEVELS):
        raise DxContractError("Dx result path is invalid")
    for position, node in enumerate(path):
        node_fields = {
            "level",
            "status",
            "index",
            "classification",
            "model_score",
            "reporting_cutoff",
        }
        if not isinstance(node, dict) or set(node) != node_fields:
            raise DxContractError("Dx result path node fields are invalid")
        if node["level"] != RESULT_LEVELS[position]:
            raise DxContractError("Dx result path must be contiguous and ordered")
        if node["status"] not in ("resolved", "implied"):
            raise DxContractError("Dx result path nodes must be resolved or implied")
        if position == 0 and node["status"] != "resolved":
            raise DxContractError("tumor presence cannot be implied")
        if type(node["index"]) is not int or node["index"] < 0:
            raise DxContractError("Dx result path node index is invalid")
        if not isinstance(node["classification"], str) or not node["classification"]:
            raise DxContractError("Dx result path node classification is invalid")
        if node["status"] == "implied":
            if node["model_score"] is not None or node["reporting_cutoff"] is not None:
                raise DxContractError(
                    "implied Dx result nodes must have null model score and reporting cutoff"
                )
        else:
            _validate_score(node["model_score"], "Dx result path model score")
            _validate_score(node["reporting_cutoff"], "Dx result path reporting cutoff")

    accepted = result["accepted"]
    if accepted is None:
        if path:
            raise DxContractError("Dx result with a path requires accepted")
    else:
        _validate_identity_node(accepted, "Dx result accepted")
        if not path or accepted != {
            key: path[-1][key] for key in ("level", "index", "classification")
        }:
            raise DxContractError("Dx result accepted must equal the last path node")

    decision = result["decision"]
    if decision is not None:
        if not isinstance(decision, dict) or set(decision) != {
            "level",
            "status",
            "reporting_cutoff",
            "differential",
        }:
            raise DxContractError("Dx result decision fields are invalid")
        if decision["status"] != "unresolved":
            raise DxContractError("Dx result decision status is invalid")
        next_position = len(path)
        if next_position >= len(RESULT_LEVELS) or decision["level"] != RESULT_LEVELS[next_position]:
            raise DxContractError("Dx result decision must be the first unresolved level")
        _validate_score(decision["reporting_cutoff"], "Dx result decision reporting cutoff")
        differential = decision["differential"]
        if not isinstance(differential, list) or len(differential) != 2:
            raise DxContractError("Dx result decision must contain a two-class differential")
        for position, entry in enumerate(differential, start=1):
            _validate_differential_entry(entry, f"Dx result differential entry {position}")
        if len({entry["index"] for entry in differential}) != 2:
            raise DxContractError("Dx result differential indices must be unique")
        expected_order = sorted(
            differential,
            key=lambda entry: (-float(entry["model_score"]), entry["index"]),
        )
        if differential != expected_order:
            raise DxContractError("Dx result differential is not ranked deterministically")
        if float(differential[0]["model_score"]) >= float(decision["reporting_cutoff"]):
            raise DxContractError(
                "Dx result unresolved classification meets its reporting cutoff"
            )

    expected_decision = result["status"] in ("no_call", "partially_resolved")
    if (decision is not None) is not expected_decision:
        raise DxContractError("Dx result status and decision are inconsistent")
    if result["status"] == "no_call" and (
        accepted is not None or path or decision["level"] != "presence"
    ):
        raise DxContractError("no_call must stop at tumor presence")
    if result["status"] == "no_call" and {
        entry["classification"] for entry in decision["differential"]
    } != {"absent", PRESENT_LABEL}:
        raise DxContractError("no_call must report absent and present in its differential")
    if result["status"] != "no_call" and accepted is None:
        raise DxContractError("issued Dx results require an accepted node")
    if result["status"] == "tumor_not_detected" and (
        len(path) != 1 or path[0]["classification"] != "absent"
    ):
        raise DxContractError("tumor_not_detected must stop after tumor presence")
    if result["status"] in ("classified", "partially_resolved") and (
        not path or path[0]["classification"] != PRESENT_LABEL
    ):
        raise DxContractError("classified Dx results require an issued present tumor signal")


def serialize_result(result: dict[str, Any]) -> str:
    validate_result(result)
    return json.dumps(result, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
