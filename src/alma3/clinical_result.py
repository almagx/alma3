from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch

from .dx import DX_TARGETS, PRESENT_LABEL, DxContractError, Taxonomy
from .release import RELEASE_VERSION
from .terminal import SUBTYPE_CUTOFF

SUBTYPE_SCALED_CUTOFF = 85.0

RESULT_STATUSES = (
    "no_call",
    "heme_tumor_not_detected",
    "fully_resolved",
    "partially_resolved",
)
RESULT_LEVELS = ("presence", "lineage", "family", "type", "subtype")
BEDMETHYL_MODIFICATION_MODES = ("5mc", "5mc_plus_5hmc")
TARGET_BY_LEVEL = {
    "presence": "hematolymphoid_tumor_presence",
    "lineage": "lineage",
    "family": "family",
    "type": "type",
    "subtype": "subtype",
}
LEVEL_BY_TARGET = {target: level for level, target in TARGET_BY_LEVEL.items()}
_RELEASE_HASH_FIELDS = {
    "ensemble_sha256",
    "taxonomy_sha256",
    "cpg_manifest_sha256",
    "thresholds_sha256",
}
_RELEASE_FIELDS = {"version", *_RELEASE_HASH_FIELDS}
_RUNTIME_FIELDS = {"device"}
_INPUT_FIELDS = {"format", "value_mode", "clipped_value_count"}
_LEVEL_DISPLAY = {
    "presence": "hematolymphoid tumor presence",
    "lineage": "lineage",
    "family": "family",
    "type": "type",
    "subtype": "subtype",
}


def result_schema_path() -> Path:
    return Path(__file__).with_name("schemas") / "dx_result.schema.json"


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def validate_sample_id(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("sample ID must be a nonempty string")
    if value != value.strip():
        raise ValueError("sample ID must not have leading or trailing whitespace")
    if any(ord(character) <= 0x1F or ord(character) == 0x7F for character in value):
        raise ValueError("sample ID must not contain ASCII control characters")
    if value[0] in "=+-@":
        raise ValueError("sample ID must not begin with =, +, -, or @")
    return value


def scaled_subtype_score(probability: float) -> float:
    """Display-only 0–100 scale; raw probability remains the issuance criterion."""
    if (isinstance(probability, bool) or not isinstance(probability, (int, float))
            or not math.isfinite(probability) or not 0 <= probability <= 1):
        raise ValueError("subtype probability must be a finite number in [0, 1]")
    if probability <= SUBTYPE_CUTOFF:
        return SUBTYPE_SCALED_CUTOFF * (probability / SUBTYPE_CUTOFF)
    return SUBTYPE_SCALED_CUTOFF + (100 - SUBTYPE_SCALED_CUTOFF) * (
        (probability - SUBTYPE_CUTOFF) / (1 - SUBTYPE_CUTOFF)
    )


def _result_summary(result: dict[str, Any]) -> str:
    status = result["status"]
    leading_candidate = None
    if status in ("no_call", "partially_resolved"):
        decision = result["decision"]
        candidate = decision["differential"][0]
        leading_candidate = (
            f"Leading candidate: {candidate['classification']} "
            f"({scaled_subtype_score(result['prediction']['probabilities'][candidate['index'] + 1]):.1f}%; "
            f"threshold {SUBTYPE_SCALED_CUTOFF:.1f}%)."
        )
    if status == "no_call":
        return f"Hematolymphoid tumor presence unresolved. {leading_candidate}"

    accepted = result["accepted"]
    accepted_level = result["path"][-1]["level"]
    scored_node = next(node for node in reversed(result["path"]) if node["status"] == "resolved")
    score = (result["prediction"]["probabilities"][result["prediction"]["terminal_index"]]
             if scored_node["level"] == "subtype" else float(scored_node["model_score"]))
    scored_confidence = score * 100.0
    if status == "heme_tumor_not_detected":
        return f"Hematolymphoid tumor not detected ({scored_confidence:.1f}%)."

    evidence = f"{scored_confidence:.1f}%"
    if scored_node["level"] == "subtype":
        evidence = f"{scaled_subtype_score(score):.1f}%"
    if accepted_level == "presence":
        summary = f"Hematolymphoid tumor detected ({evidence})."
    else:
        level = _LEVEL_DISPLAY[accepted_level].capitalize()
        summary = f"{level}: {accepted['classification']} ({evidence})."
    if status == "partially_resolved":
        unresolved = _LEVEL_DISPLAY[RESULT_LEVELS[len(result["path"])]].capitalize()
        return f"{summary} {unresolved} unresolved. {leading_candidate}"
    return summary


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
    taxonomy: Taxonomy,
) -> dict[str, Any]:
    return {
        "level": level,
        "status": status,
        "index": index,
        "classification": taxonomy.classes[TARGET_BY_LEVEL[level]][index],
        **({"model_score": score} if level != "subtype" else {}),
    }


def _unresolved_decision(
    level: str,
    ranked: list[tuple[int, float]],
    taxonomy: Taxonomy,
) -> dict[str, Any]:
    if not 1 <= len(ranked) <= 2:
        raise DxContractError(f"{level} unresolved decision requires one or two differential diagnoses")
    labels = taxonomy.classes[TARGET_BY_LEVEL[level]]
    return {
        "status": "unresolved",
        "differential": [
            {"index": index, "classification": labels[index]}
            for index, _ in ranked
        ],
    }


def _result(
    sample_id: str,
    release: dict[str, str],
    runtime: dict[str, str],
    input_metadata: dict[str, Any],
    observed_cpg_count: int,
    status: str,
    path: list[dict[str, Any]],
    decision: dict[str, Any] | None,
    prediction: dict[str, Any],
) -> dict[str, Any]:
    accepted = (
        None
        if not path
        else {key: path[-1][key] for key in ("index", "classification")}
    )
    result = {
        "sample_id": sample_id,
        "result_summary": "",
        "release": dict(release),
        "runtime": dict(runtime),
        "input": dict(input_metadata),
        "observed_cpg_count": observed_cpg_count,
        "status": status,
        "accepted": accepted,
        "path": path,
        "decision": decision,
        "prediction": prediction,
    }
    result["result_summary"] = _result_summary(result)
    validate_result(result)
    return result


def results_from_probabilities(
    sample_ids, probabilities, thresholds, taxonomy, release, *, runtime,
    input_metadata, observed_cpg_counts, minimum_observed_cpgs,
):
    """Apply reporting gates to the frozen selected path without changing the decoder."""
    from .terminal import selected_path_scores
    release = {key: value for key, value in release.items() if key in _RELEASE_FIELDS}
    runtime = {"device": runtime["device"]}
    _validate_release(release)
    _validate_runtime(runtime)
    selected, chosen, scores, distributions = selected_path_scores(probabilities, taxonomy)
    ids, inputs, counts = list(sample_ids), list(input_metadata), list(observed_cpg_counts)
    if (not ids or len(ids) != len(set(ids)) or len(ids) != len(probabilities)
            or len(inputs) != len(ids) or len(counts) != len(ids)
            or minimum_observed_cpgs != 5000 or any(type(n) is not int or n < 5000 for n in counts)):
        raise DxContractError('result identities and observed CpG counts must match the prediction batch')
    cuts = [float(thresholds['thresholds'][key]) for key in DX_TARGETS]
    if cuts != [0., 0., 0., 0., SUBTYPE_CUTOFF]:
        raise DxContractError('reporting uses only the frozen subtype cutoff')
    results = []
    for i, (sample_id, metadata, count) in enumerate(zip(ids, inputs, counts, strict=True)):
        validate_sample_id(sample_id)
        _validate_input(metadata, count)
        path, decision = [], None
        terminal = int(selected[i])
        prediction = dict(decision_rule='type_first_v1', terminal_index=terminal,
                          path=chosen[i].tolist(), probabilities=probabilities[i].tolist(),
                          subtype_cutoff_passed=bool(chosen[i, 4] >= 0 and probabilities[i, terminal] >= SUBTYPE_CUTOFF))
        for j, level in enumerate(RESULT_LEVELS):
            index = int(chosen[i, j])
            if index < 0:
                break
            parent = None if j < 2 else int(chosen[i, j - 1])
            indices = list(range(2)) if j == 0 else _child_indices(level, parent, taxonomy)
            # A subtype always uses the joint probability gate, including singleton branches.
            implied = j in (1, 2, 3) and len(indices) == 1
            score = float(scores[i, j])
            if not implied and score < cuts[j]:
                ranked = _rank_candidates(distributions[j][i], indices, limit=2)
                decision = _unresolved_decision(level, ranked, taxonomy)
                decision.update(selected_index=index, selected_score=score)
                break
            path.append(_path_node(level, 'implied' if implied else 'resolved', index,
                                   None if implied else score, taxonomy))
        status = ('no_call' if not path else 'partially_resolved') if decision else (
            'heme_tumor_not_detected' if terminal == 0 else 'fully_resolved')
        results.append(_result(sample_id, release, runtime, metadata, count,
                               status, path, decision, prediction))
    return results


def _validate_release(release: Any) -> None:
    if not isinstance(release, dict) or set(release) != _RELEASE_FIELDS:
        raise DxContractError("result release fields are invalid")
    if release["version"] != RELEASE_VERSION:
        raise DxContractError("result release version is invalid")
    for field in _RELEASE_HASH_FIELDS:
        if not _is_sha256(release[field]):
            raise DxContractError(f"result release {field} must be a lowercase SHA-256")


def _validate_runtime(runtime: Any) -> None:
    if not isinstance(runtime, dict) or set(runtime) != _RUNTIME_FIELDS:
        raise DxContractError("result runtime fields are invalid")
    device = runtime["device"]
    if device == "cpu":
        return
    if not isinstance(device, str) or not device.startswith("cuda:"):
        raise DxContractError("result runtime device is invalid")
    index = device.removeprefix("cuda:")
    if not index.isdigit() or str(int(index)) != index:
        raise DxContractError("result runtime device is invalid")


def _validate_input(metadata: Any, observed_cpg_count: int) -> None:
    if not isinstance(metadata, dict) or "format" not in metadata:
        raise DxContractError("result input fields are invalid")
    input_format = metadata["format"]
    expected_fields = (
        {*_INPUT_FIELDS, "modification_mode"}
        if input_format == "bedmethyl"
        else _INPUT_FIELDS
    )
    if set(metadata) != expected_fields:
        raise DxContractError("result input fields are invalid")
    value_mode = metadata["value_mode"]
    clipped = metadata["clipped_value_count"]
    if not isinstance(input_format, str) or input_format not in {
        "array",
        "array-csv",
        "bedmethyl",
    }:
        raise DxContractError("result input format is invalid")
    if input_format in {"array", "array-csv"}:
        if not isinstance(value_mode, str) or value_mode not in {"beta", "mvalue"}:
            raise DxContractError("result array input value mode is invalid")
    else:
        if value_mode != "fraction_modified":
            raise DxContractError("result bedMethyl input value mode is invalid")
        if metadata["modification_mode"] not in BEDMETHYL_MODIFICATION_MODES:
            raise DxContractError("result bedMethyl modification mode is invalid")
    if type(clipped) is not int or clipped < 0 or clipped > observed_cpg_count:
        raise DxContractError("result input clipped value count is invalid")
    if (input_format == "bedmethyl" or value_mode == "mvalue") and clipped != 0:
        raise DxContractError("result input mode cannot report clipped values")


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
    if not isinstance(node, dict) or set(node) != {"index", "classification"}:
        raise DxContractError(f"{description} fields are invalid")
    if type(node["index"]) is not int or node["index"] < 0:
        raise DxContractError(f"{description} index is invalid")
    if not isinstance(node["classification"], str) or not node["classification"]:
        raise DxContractError(f"{description} classification is invalid")


def _validate_differential_entry(entry: Any, description: str) -> None:
    _validate_identity_node(entry, description)
    if entry["index"] >= 107:
        raise DxContractError(f"{description} index is invalid")


def validate_result(result: Any) -> None:
    fields = {
        "sample_id",
        "result_summary",
        "release",
        "runtime",
        "input",
        "observed_cpg_count",
        "status",
        "accepted",
        "path",
        "decision",
        "prediction",
    }
    if not isinstance(result, dict) or set(result) != fields:
        raise DxContractError("Dx result fields are invalid")
    try:
        validate_sample_id(result["sample_id"])
    except ValueError as error:
        raise DxContractError(f"Dx result {error}") from error
    _validate_release(result["release"])
    _validate_runtime(result["runtime"])
    observed_count = result["observed_cpg_count"]
    if type(observed_count) is not int or observed_count < 5000:
        raise DxContractError("Dx result observed-CpG support is invalid")
    _validate_input(result["input"], observed_count)
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
        }
        if position != 4:
            node_fields.add("model_score")
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
            if position == 4 or node["model_score"] is not None:
                raise DxContractError(
                    "implied ancestor nodes must have null model score"
                )
        elif position != 4:
            _validate_score(node["model_score"], "Dx result path model score")

    accepted = result["accepted"]
    if accepted is None:
        if path:
            raise DxContractError("Dx result with a path requires accepted")
    else:
        _validate_identity_node(accepted, "Dx result accepted")
        if not path or accepted != {
            key: path[-1][key] for key in ("index", "classification")
        }:
            raise DxContractError("Dx result accepted must equal the last path node")

    prediction = result['prediction']
    if (not isinstance(prediction, dict) or set(prediction) != {'decision_rule', 'terminal_index', 'path', 'probabilities', 'subtype_cutoff_passed'}
            or prediction['decision_rule'] != 'type_first_v1' or type(prediction['terminal_index']) is not int
            or not 0 <= prediction['terminal_index'] < 109
            or not isinstance(prediction['path'], list) or len(prediction['path']) != 5
            or any(type(i) is not int or i < -1 for i in prediction['path'])
            or not isinstance(prediction['probabilities'], list) or len(prediction['probabilities']) != 109
            or type(prediction['subtype_cutoff_passed']) is not bool):
        raise DxContractError('invalid selected diagnosis record')
    for probability in prediction['probabilities']:
        _validate_score(probability, 'terminal probability')
    if abs(sum(prediction['probabilities']) - 1.) > 3e-6:
        raise DxContractError('terminal probabilities must sum to one')

    decision = result["decision"]
    if decision is not None:
        if not isinstance(decision, dict) or set(decision) != {
            "status",
            "differential",
            "selected_index",
            "selected_score",
        }:
            raise DxContractError("Dx result decision fields are invalid")
        if decision["status"] != "unresolved":
            raise DxContractError("Dx result decision status is invalid")
        next_position = len(path)
        if next_position != 4:
            raise DxContractError("only a subtype may be withheld")
        _validate_score(decision['selected_score'], 'selected path score')
        differential = decision["differential"]
        if not isinstance(differential, list) or not 1 <= len(differential) <= 2:
            raise DxContractError("Dx result decision must contain one or two differential diagnoses")
        for position, entry in enumerate(differential, start=1):
            _validate_differential_entry(entry, f"Dx result differential entry {position}")
        if len({entry["index"] for entry in differential}) != len(differential):
            raise DxContractError("Dx result differential indices must be unique")
        expected_order = sorted(
            differential,
            key=lambda entry: (-result['prediction']['probabilities'][entry['index'] + 1], entry['index']),
        )
        if differential != expected_order:
            raise DxContractError("Dx result differential is not ranked deterministically")
        if float(decision["selected_score"]) >= SUBTYPE_CUTOFF:
            raise DxContractError(
                "Dx result unresolved classification meets its reporting cutoff"
            )

    eligible = prediction['path'][4] >= 0 and prediction['probabilities'][prediction['terminal_index']] >= SUBTYPE_CUTOFF
    if prediction['subtype_cutoff_passed'] != eligible:
        raise DxContractError('subtype cutoff eligibility differs from the selected terminal probability')
    if any(node['index'] != prediction['path'][i] for i, node in enumerate(path)):
        raise DxContractError('reporting must preserve the selected diagnostic path')
    if decision and (decision['selected_index'] != prediction['path'][len(path)]):
        raise DxContractError('unresolved decision must preserve the selected diagnostic path')
    expected_decision = result["status"] in ("no_call", "partially_resolved")
    if (decision is not None) is not expected_decision:
        raise DxContractError("Dx result status and decision are inconsistent")
    if result["status"] == "no_call" and (
        accepted is not None or path
    ):
        raise DxContractError("no_call must stop at tumor presence")
    if result["status"] == "no_call" and {
        entry["classification"] for entry in decision["differential"]
    } != {"absent", PRESENT_LABEL}:
        raise DxContractError("no_call must report absent and present in its differential")
    if result["status"] != "no_call" and accepted is None:
        raise DxContractError("issued Dx results require an accepted node")
    if result["status"] == "heme_tumor_not_detected" and (
        len(path) != 1 or path[0]["classification"] != "absent"
    ):
        raise DxContractError("heme_tumor_not_detected must stop after tumor presence")
    if result["status"] in ("fully_resolved", "partially_resolved") and (
        not path or path[0]["classification"] != PRESENT_LABEL
    ):
        raise DxContractError("resolved Dx results require an issued present tumor signal")
    if not isinstance(result["result_summary"], str) or result["result_summary"] != _result_summary(result):
        raise DxContractError("Dx result summary is invalid")


def serialize_result(result: dict[str, Any]) -> str:
    validate_result(result)
    return json.dumps(result, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
