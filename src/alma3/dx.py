from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file
from torch import nn

from .config import DxConfig
from .model import FoundationModel
from .sitewise import CANONICAL_BASE_SEED, CANONICAL_CONDITIONS, MINIMUM_RUNTIME_INPUT_CPGS

DX_TARGETS = (
    "hematolymphoid_tumor_presence",
    "lineage",
    "family",
    "type",
    "subtype",
)
DX_REPRESENTATION_NAME = "diagnostic_trunk_embedding"
DX_REPRESENTATION_VERSION = 1
DX_REPRESENTATION_DTYPE = "float32"
DX_REPRESENTATION_DIMENSIONS = 1536
CALIBRATION_RULE = "fixed_1500_canonical_worst_condition_point_precision_source_bound_v5"
THRESHOLDS_SCHEMA_VERSION = 9
CALIBRATION_PRECISION_FLOOR = {
    "hematolymphoid_tumor_presence": 0.99,
    "lineage": 0.95,
    "family": 0.95,
    "type": 0.95,
    "subtype": 0.90,
}
MINIMUM_OBSERVED_CPGS = MINIMUM_RUNTIME_INPUT_CPGS
NO_CALL_THRESHOLD = 1.000001
PRESENT_LABEL = "present"


class DxContractError(ValueError):
    """Raised when a Dx artifact violates the runtime contract."""


def _head_key(target: str) -> str:
    return f"{target}_head"


@dataclass(frozen=True)
class Taxonomy:
    classes: dict[str, tuple[str, ...]]
    family_by_lineage: dict[str, tuple[str, ...]]
    type_by_family: dict[str, tuple[str, ...]]
    subtype_by_type: dict[str, tuple[str, ...]]

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Taxonomy:
        if payload.get("kind") != "alma3_taxonomy":
            raise DxContractError("taxonomy kind must be alma3_taxonomy")
        levels = payload.get("levels")
        if not isinstance(levels, dict):
            raise DxContractError("taxonomy requires levels object")
        missing = [target for target in DX_TARGETS if target not in levels]
        if missing:
            raise DxContractError(f"taxonomy missing level(s): {', '.join(missing)}")
        classes = {}
        for target in DX_TARGETS:
            values = levels[target]
            if isinstance(values, (str, bytes)) or not isinstance(values, list):
                raise DxContractError(f"taxonomy level {target} must be a list")
            classes[target] = tuple(str(value) for value in values)
        for target, values in classes.items():
            if not values:
                raise DxContractError(f"taxonomy level {target} is empty")
        taxonomy = cls(
            classes=classes,
            family_by_lineage=_tuple_mapping(payload.get("family_by_lineage") or {}),
            type_by_family=_tuple_mapping(payload.get("type_by_family") or {}),
            subtype_by_type=_tuple_mapping(payload.get("subtype_by_type") or {}),
        )
        _validate_taxonomy(taxonomy)
        return taxonomy

    def validate_sizes(self, sizes: dict[str, int]) -> None:
        mismatch = [
            f"{target}: taxonomy={len(self.classes[target])} config={sizes.get(target)}"
            for target in DX_TARGETS
            if int(sizes.get(target, -1)) != len(self.classes[target])
        ]
        if mismatch:
            raise DxContractError("Dx target size mismatch: " + "; ".join(mismatch))

    def valid_family_mask(self, lineage_idx: int, device: torch.device | None = None) -> torch.Tensor:
        lineage = self.classes["lineage"][int(lineage_idx)]
        return _label_mask(self.classes["family"], self.family_by_lineage.get(lineage, ()), device)

    def valid_type_mask(self, family_idx: int, device: torch.device | None = None) -> torch.Tensor:
        family = self.classes["family"][int(family_idx)]
        return _label_mask(self.classes["type"], self.type_by_family.get(family, ()), device)

    def valid_subtype_mask(self, type_idx: int, device: torch.device | None = None) -> torch.Tensor:
        type_label = self.classes["type"][int(type_idx)]
        return _label_mask(self.classes["subtype"], self.subtype_by_type.get(type_label, ()), device)

    def has_subtype_children(self, type_idx: int) -> bool:
        type_label = self.classes["type"][int(type_idx)]
        return bool(self.subtype_by_type.get(type_label, ()))


def _tuple_mapping(raw: dict[str, Any]) -> dict[str, tuple[str, ...]]:
    if not isinstance(raw, dict):
        raise DxContractError("taxonomy hierarchy mappings must be objects")
    result: dict[str, tuple[str, ...]] = {}
    for parent, children in raw.items():
        if isinstance(children, (str, bytes)) or not isinstance(children, list):
            raise DxContractError(f"taxonomy hierarchy children for {parent!r} must be a list")
        result[str(parent)] = tuple(str(child) for child in children)
    return result


def _validate_unique(level: str, values: tuple[str, ...]) -> None:
    seen: set[str] = set()
    duplicates: list[str] = []
    for value in values:
        if value in seen and value not in duplicates:
            duplicates.append(value)
        seen.add(value)
    if duplicates:
        raise DxContractError(f"taxonomy level {level} has duplicate labels: {duplicates[:5]}")


def _validate_mapping(
    name: str,
    mapping: dict[str, tuple[str, ...]],
    parent_classes: tuple[str, ...],
    child_classes: tuple[str, ...],
    *,
    require_all_parents: bool,
) -> None:
    parent_set = set(parent_classes)
    child_set = set(child_classes)
    unknown_parents = sorted(set(mapping) - parent_set)
    if unknown_parents:
        raise DxContractError(f"taxonomy {name} has unknown parent label: {unknown_parents[0]}")
    empty_parents = sorted(parent for parent in parent_classes if not mapping.get(parent))
    if require_all_parents and empty_parents:
        raise DxContractError(f"taxonomy {name} leaves parent labels without children: {empty_parents[:5]}")
    attached: set[str] = set()
    for parent, children in mapping.items():
        child_seen: set[str] = set()
        for child in children:
            if child in child_seen:
                raise DxContractError(f"taxonomy {name} repeats child label under {parent}: {child}")
            child_seen.add(child)
            if child not in child_set:
                raise DxContractError(f"taxonomy {name} has unknown child label: {child}")
            if child in attached:
                raise DxContractError(f"taxonomy {name} attaches child label to multiple parents: {child}")
            attached.add(child)
    missing = sorted(child_set - attached)
    if missing:
        raise DxContractError(f"taxonomy {name} leaves child labels unattached: {missing[:5]}")


def _validate_taxonomy(taxonomy: Taxonomy) -> None:
    for target, values in taxonomy.classes.items():
        _validate_unique(target, values)
    if PRESENT_LABEL not in taxonomy.classes["hematolymphoid_tumor_presence"]:
        raise DxContractError(f"taxonomy hematolymphoid_tumor_presence must include {PRESENT_LABEL!r}")
    _validate_mapping(
        "family_by_lineage",
        taxonomy.family_by_lineage,
        taxonomy.classes["lineage"],
        taxonomy.classes["family"],
        require_all_parents=True,
    )
    _validate_mapping(
        "type_by_family",
        taxonomy.type_by_family,
        taxonomy.classes["family"],
        taxonomy.classes["type"],
        require_all_parents=True,
    )
    _validate_mapping(
        "subtype_by_type",
        taxonomy.subtype_by_type,
        taxonomy.classes["type"],
        taxonomy.classes["subtype"],
        require_all_parents=False,
    )


def _label_mask(classes: tuple[str, ...], labels: tuple[str, ...], device: torch.device | None) -> torch.Tensor:
    allowed = set(labels)
    return torch.tensor([label in allowed for label in classes], dtype=torch.bool, device=device)


def load_taxonomy(path: str | Path) -> Taxonomy:
    return Taxonomy.from_dict(json.loads(Path(path).read_text()))


class DiagnosticModel(nn.Module):
    def __init__(self, foundation: FoundationModel, config: DxConfig, *, freeze_foundation: bool = True):
        super().__init__()
        self.foundation = foundation
        self.config = config
        if freeze_foundation:
            for param in self.foundation.parameters():
                param.requires_grad = False
        d_model = config.foundation.d_model
        self.trunk = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, d_model),
            nn.GELU(),
        )
        self.heads = nn.ModuleDict(
            {_head_key(target): nn.Linear(d_model, config.targets[target]) for target in DX_TARGETS}
        )

    def forward(
        self,
        beta: torch.Tensor,
        observed: torch.Tensor,
        uncertainty: torch.Tensor,
        chr_id: torch.Tensor,
        pos: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        return self.logits_from_embedding(self.embed(beta, observed, uncertainty, chr_id, pos))

    def embed(
        self,
        beta: torch.Tensor,
        observed: torch.Tensor,
        uncertainty: torch.Tensor,
        chr_id: torch.Tensor,
        pos: torch.Tensor,
    ) -> torch.Tensor:
        """Return the versioned post-trunk representation consumed by every Dx head."""
        return self.trunk(self.foundation.embed(beta, observed, uncertainty, chr_id, pos)).to(dtype=torch.float32)

    def logits_from_embedding(self, embedding: torch.Tensor) -> dict[str, torch.Tensor]:
        if embedding.ndim != 2 or embedding.shape[1] != self.config.foundation.d_model:
            raise DxContractError(
                "Dx embedding must have shape "
                f"[batch, {self.config.foundation.d_model}], found {tuple(embedding.shape)}"
            )
        if embedding.dtype != torch.float32:
            raise DxContractError("Dx embedding must use float32")
        return {target: self.heads[_head_key(target)](embedding) for target in DX_TARGETS}


def _wilson_bounds(correct: int, total: int) -> tuple[float, float] | None:
    if total == 0:
        return None
    z = 1.959963984540054
    proportion = correct / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    margin = z * math.sqrt(proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


_CONDITION_METRIC_FIELDS = {
    "threshold",
    "precision",
    "wilson_lower_95",
    "wilson_upper_95",
    "recall",
    "coverage",
    "called",
    "correct",
    "false_positive",
    "labeled",
    "precision_floor",
    "minimum_calls",
    "precision_met",
}


def validate_threshold_condition_metric(
    metric: Any,
    *,
    target: str,
    threshold: float,
    floor: float,
    description: str,
    expected_labeled: int | None = None,
) -> bool:
    """Validate one fixed-threshold condition and return whether its policy is met."""

    if not isinstance(metric, dict) or set(metric) != _CONDITION_METRIC_FIELDS:
        raise DxContractError(f"{description} metric fields are invalid for {target}")
    called, correct, false_positive, labeled = (
        metric[name] for name in ("called", "correct", "false_positive", "labeled")
    )
    precision = (
        correct / called
        if type(called) is int and type(correct) is int and called > 0
        else None
    )
    expected_met = type(called) is int and called >= 30 and precision is not None and precision >= floor
    if (
        any(type(value) is not int for value in (called, correct, false_positive, labeled))
        or not 0 <= correct <= called <= labeled
        or false_positive != called - correct
        or (expected_labeled is not None and labeled != expected_labeled)
        or metric["minimum_calls"] != 30
        or type(metric["precision_met"]) is not bool
        or metric["precision_met"] is not expected_met
        or not math.isclose(float(metric["threshold"]), threshold, rel_tol=0.0, abs_tol=1e-12)
        or not math.isclose(float(metric["precision_floor"]), floor, rel_tol=0.0, abs_tol=1e-12)
    ):
        raise DxContractError(f"{description} metric accounting is invalid for {target}")
    interval = _wilson_bounds(correct, called)
    if precision is None:
        if any(metric[name] is not None for name in ("precision", "wilson_lower_95", "wilson_upper_95")):
            raise DxContractError(f"{description} empty-call precision is invalid for {target}")
    elif any(
        isinstance(metric[name], bool)
        or not isinstance(metric[name], (int, float))
        or not math.isfinite(float(metric[name]))
        for name in ("precision", "wilson_lower_95", "wilson_upper_95")
    ) or not all(
        math.isclose(float(metric[name]), expected, rel_tol=0.0, abs_tol=1e-12)
        for name, expected in zip(
            ("precision", "wilson_lower_95", "wilson_upper_95"),
            (precision, *(interval or (0.0, 0.0))),
            strict=True,
        )
    ):
        raise DxContractError(f"{description} precision evidence is invalid for {target}")
    for name, expected in (
        ("recall", correct / labeled if labeled else 0.0),
        ("coverage", called / labeled if labeled else 0.0),
    ):
        value = metric[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not math.isclose(float(value), expected, rel_tol=0.0, abs_tol=1e-12)
        ):
            raise DxContractError(f"{description} {name} is invalid for {target}")
    return expected_met


def load_thresholds(path: str | Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text())
    fields = {
        "kind",
        "schema_version",
        "generated_at",
        "status",
        "calibration_rule",
        "conditions",
        "calibration_precision_floor",
        "operating_policy_met",
        "availability",
        "thresholds",
        "temperatures",
        "temperature_metrics",
        "metrics",
        "calibration_seed",
        "minimum_observed_cpgs",
        "model_sha256",
        "taxonomy_sha256",
        "cpg_manifest_sha256",
        "dx_view_sha256",
        "training_manifest_sha256",
        "canonical_evaluator_sha256",
        "selection_calibration_sha256",
        "evaluation_git_commit",
        "evaluation_git_clean",
    }
    if not isinstance(data, dict) or set(data) != fields:
        raise DxContractError("thresholds file fields are invalid")
    if data.get("kind") != "alma3_dx_thresholds":
        raise DxContractError("thresholds file has wrong kind")
    if data.get("status") != "calibrated":
        raise DxContractError("thresholds file must have status='calibrated'")
    if int(data.get("schema_version", 0)) != THRESHOLDS_SCHEMA_VERSION:
        raise DxContractError(f"thresholds schema_version must be {THRESHOLDS_SCHEMA_VERSION}")
    if data.get("calibration_rule") != CALIBRATION_RULE:
        raise DxContractError(f"thresholds file must use calibration_rule={CALIBRATION_RULE!r}")
    condition_names = [name for name, _ in CANONICAL_CONDITIONS]
    if data.get("conditions") != condition_names:
        raise DxContractError("thresholds conditions must exactly match the canonical evaluator")
    for key in ("thresholds", "temperatures", "metrics", "calibration_precision_floor", "availability"):
        if not isinstance(data[key], dict) or set(data[key]) != set(DX_TARGETS):
            raise DxContractError(f"thresholds file {key} must define exactly the Dx targets")
    if {target: float(data["calibration_precision_floor"][target]) for target in DX_TARGETS} != CALIBRATION_PRECISION_FLOOR:
        raise DxContractError("thresholds file must use the fixed 99/95/95/95/90 precision policy")
    if any(
        not math.isfinite(float(data["temperatures"][target])) or float(data["temperatures"][target]) <= 0
        for target in DX_TARGETS
    ):
        raise DxContractError("all calibration temperatures must be positive")
    if not isinstance(data["temperature_metrics"], dict) or set(data["temperature_metrics"]) != set(DX_TARGETS):
        raise DxContractError("temperature_metrics must define exactly the Dx targets")
    for target, metric in data["temperature_metrics"].items():
        if (
            not isinstance(metric, dict)
            or set(metric) != {"examples", "examples_by_condition", "nll_before", "nll_after"}
            or type(metric["examples"]) is not int
            or metric["examples"] < 0
            or not isinstance(metric["examples_by_condition"], dict)
            or set(metric["examples_by_condition"]) != set(condition_names)
            or any(
                type(metric["examples_by_condition"][condition]) is not int
                or metric["examples_by_condition"][condition] < 0
                for condition in condition_names
            )
            or metric["examples"]
            != sum(metric["examples_by_condition"][condition] for condition in condition_names)
            or any(
                isinstance(metric[name], bool)
                or not isinstance(metric[name], (int, float))
                or not math.isfinite(float(metric[name]))
                or float(metric[name]) < 0.0
                for name in ("nll_before", "nll_after")
            )
        ):
            raise DxContractError(f"temperature metrics are invalid for {target}")
    if data["calibration_seed"] != CANONICAL_BASE_SEED:
        raise DxContractError(f"calibration_seed must be the canonical value {CANONICAL_BASE_SEED}")
    if type(data["operating_policy_met"]) is not bool:
        raise DxContractError("operating_policy_met must be boolean")
    if any(type(data["availability"][target]) is not bool for target in DX_TARGETS):
        raise DxContractError("threshold availability must be boolean")
    if data["operating_policy_met"] is not all(data["availability"].values()):
        raise DxContractError("operating_policy_met does not match target availability")
    if data["minimum_observed_cpgs"] != MINIMUM_OBSERVED_CPGS:
        raise DxContractError(f"minimum_observed_cpgs must be {MINIMUM_OBSERVED_CPGS}")

    metric_fields = {
        "threshold",
        "precision_floor",
        "minimum_calls_per_condition",
        "precision_met",
        "worst_condition_coverage",
        "total_called",
        "total_correct",
        "total_false_positive",
        "total_labeled",
        "conditions",
    }

    for name in (
        "model_sha256",
        "taxonomy_sha256",
        "cpg_manifest_sha256",
        "dx_view_sha256",
        "training_manifest_sha256",
        "canonical_evaluator_sha256",
        "selection_calibration_sha256",
    ):
        value = str(data[name])
        if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            raise DxContractError(f"thresholds {name} must be a lowercase SHA256")
    revision = str(data["evaluation_git_commit"])
    if (
        len(revision) not in {40, 64}
        or revision != revision.lower()
        or any(char not in "0123456789abcdef" for char in revision)
        or data["evaluation_git_clean"] is not True
    ):
        raise DxContractError("thresholds evaluation source binding is invalid")
    unavailable_seen = False
    for target in DX_TARGETS:
        threshold = float(data["thresholds"][target])
        floor = float(data["calibration_precision_floor"][target])
        metric = data["metrics"].get(target)
        available = data["availability"][target]
        if unavailable_seen and available:
            raise DxContractError("threshold availability must form one contiguous hierarchy prefix")
        unavailable_seen = unavailable_seen or not available
        if not math.isfinite(threshold) or (
            available and not 0.0 <= threshold <= 1.0
        ) or (not available and threshold != NO_CALL_THRESHOLD):
            raise DxContractError(f"threshold for {target} is incompatible with its availability")
        if not math.isfinite(floor) or not 0.0 < floor <= 1.0:
            raise DxContractError(f"precision floor for {target} must be in (0, 1]")
        if not isinstance(metric, dict) or set(metric) != metric_fields:
            raise DxContractError(f"calibration metric fields are invalid for {target}")
        condition_metrics = metric["conditions"]
        if (
            not isinstance(condition_metrics, dict)
            or set(condition_metrics) != set(condition_names)
        ):
            raise DxContractError(f"calibration conditions are invalid for {target}")
        expected_met = True
        totals = {name: 0 for name in ("called", "correct", "false_positive", "labeled")}
        coverages: list[float] = []
        for condition in condition_names:
            condition_metric = condition_metrics[condition]
            precision = (
                None if not isinstance(condition_metric, dict) else condition_metric.get("precision")
            )
            condition_met = (
                isinstance(condition_metric, dict)
                and type(condition_metric.get("called")) is int
                and condition_metric["called"] >= 30
                and isinstance(precision, (int, float))
                and not isinstance(precision, bool)
                and math.isfinite(float(precision))
                and float(precision) >= floor
            )
            validated_met = validate_threshold_condition_metric(
                condition_metric,
                target=target,
                threshold=threshold,
                floor=floor,
                description=f"calibration {condition}",
            )
            if validated_met is not condition_met:
                raise DxContractError(f"calibration {condition} policy evidence is invalid for {target}")
            expected_met = expected_met and condition_met
            for name in totals:
                totals[name] += int(condition_metric[name])
            coverages.append(float(condition_metric["coverage"]))
        if (
            metric["threshold"] != threshold
            or metric["precision_floor"] != floor
            or metric["minimum_calls_per_condition"] != 30
            or type(metric["precision_met"]) is not bool
            or metric["precision_met"] is not expected_met
            or any(type(metric[f"total_{name}"]) is not int for name in totals)
            or any(metric[f"total_{name}"] != value for name, value in totals.items())
            or not isinstance(metric["worst_condition_coverage"], (int, float))
            or isinstance(metric["worst_condition_coverage"], bool)
            or not math.isfinite(float(metric["worst_condition_coverage"]))
            or not math.isclose(
                float(metric["worst_condition_coverage"]),
                min(coverages),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise DxContractError(f"calibration aggregate evidence is invalid for {target}")
        if available is not expected_met:
            raise DxContractError(f"threshold availability does not match calibration evidence for {target}")
    return data
def load_dx(path: str | Path, device: torch.device | str = "cpu") -> DiagnosticModel:
    root = Path(path)
    config_path = root / "config.json"
    weights_path = root / "model.safetensors"
    if not config_path.exists():
        raise FileNotFoundError(f"Dx config missing: {config_path}")
    if not weights_path.exists():
        raise FileNotFoundError(f"Dx weights missing: {weights_path}")
    config = DxConfig.from_dict(json.loads(config_path.read_text()))
    foundation = FoundationModel(config.foundation)
    model = DiagnosticModel(foundation, config, freeze_foundation=False)
    model.load_state_dict(load_file(str(weights_path), device=str(device)))
    return model.to(device).eval()


def require_finite_logits(logit: torch.Tensor, *, target: str) -> None:
    if not bool(torch.isfinite(logit).all().item()):
        raise DxContractError(f"non-finite logits for {target}")


def validate_dx_logits(logits: dict[str, torch.Tensor], taxonomy: Taxonomy) -> None:
    if set(logits) != set(DX_TARGETS):
        raise DxContractError("logits must define exactly the Dx targets")
    batch_size: int | None = None
    for target in DX_TARGETS:
        logit = logits[target]
        if not isinstance(logit, torch.Tensor) or logit.ndim != 2:
            raise DxContractError(f"logits for {target} must have rank 2")
        if batch_size is None:
            batch_size = int(logit.shape[0])
            if batch_size < 1:
                raise DxContractError("logits batch size must be positive")
        elif int(logit.shape[0]) != batch_size:
            raise DxContractError("logits must share one batch size across Dx targets")
        if int(logit.shape[1]) != len(taxonomy.classes[target]):
            raise DxContractError(f"logits class width does not match taxonomy for {target}")
        require_finite_logits(logit, target=target)


def _masked_top_prediction(logit: torch.Tensor, mask: torch.Tensor | None) -> tuple[float, int | None]:
    if mask is not None:
        if int(mask.sum().item()) == 0:
            return 0.0, None
        logit = logit.masked_fill(~mask.to(device=logit.device), -torch.inf)
    probs = torch.softmax(logit.float(), dim=-1)
    conf, pred = probs.max(dim=-1)
    return float(conf.cpu()), int(pred.cpu())


def apply_temperatures(
    logits: dict[str, torch.Tensor], thresholds: dict[str, Any], taxonomy: Taxonomy
) -> dict[str, torch.Tensor]:
    validate_dx_logits(logits, taxonomy)
    temperatures = thresholds.get("temperatures")
    if not isinstance(temperatures, dict) or set(temperatures) != set(DX_TARGETS):
        raise DxContractError("threshold temperatures must define exactly the Dx targets")
    scaled = {target: logits[target] / float(temperatures[target]) for target in DX_TARGETS}
    validate_dx_logits(scaled, taxonomy)
    return scaled


def _masked_probabilities(logit: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is not None:
        if int(mask.sum().item()) == 0:
            raise DxContractError("terminal hierarchy probability encountered an empty taxonomy mask")
        logit = logit.masked_fill(~mask.to(dtype=torch.bool), -torch.inf)
    return torch.softmax(logit.float(), dim=-1)


def terminal_path_probabilities(
    logits: dict[str, torch.Tensor], thresholds: dict[str, Any], taxonomy: Taxonomy
) -> list[dict[str, float]]:
    """Return the normalized absence plus terminal hierarchy distribution."""
    scaled = apply_temperatures(logits, thresholds, taxonomy)
    present_idx = taxonomy.classes["hematolymphoid_tumor_presence"].index(PRESENT_LABEL)
    result: list[dict[str, float]] = []
    for sample_idx in range(next(iter(scaled.values())).shape[0]):
        sample = {target: scaled[target][sample_idx].float().detach().cpu() for target in DX_TARGETS}
        presence = _masked_probabilities(sample["hematolymphoid_tumor_presence"], None)
        present_probability = float(presence[present_idx])
        row = {"absence": float(presence.sum() - presence[present_idx])}
        lineage_probabilities = _masked_probabilities(sample["lineage"], None)
        for lineage_idx, lineage in enumerate(taxonomy.classes["lineage"]):
            family_probabilities = _masked_probabilities(
                sample["family"], taxonomy.valid_family_mask(lineage_idx)
            )
            for family in taxonomy.family_by_lineage[lineage]:
                family_idx = taxonomy.classes["family"].index(family)
                type_probabilities = _masked_probabilities(sample["type"], taxonomy.valid_type_mask(family_idx))
                for type_label in taxonomy.type_by_family[family]:
                    type_idx = taxonomy.classes["type"].index(type_label)
                    edge_probabilities = (
                        present_probability,
                        float(lineage_probabilities[lineage_idx]),
                        float(family_probabilities[family_idx]),
                        float(type_probabilities[type_idx]),
                    )
                    subtypes = taxonomy.subtype_by_type.get(type_label, ())
                    if not subtypes:
                        key = f"type:{type_idx}"
                        scores = edge_probabilities
                        if key in row:
                            raise DxContractError(f"duplicate terminal hierarchy path: {key}")
                        row[key] = math.prod(scores)
                        continue
                    subtype_probabilities = _masked_probabilities(
                        sample["subtype"], taxonomy.valid_subtype_mask(type_idx)
                    )
                    for subtype in subtypes:
                        subtype_idx = taxonomy.classes["subtype"].index(subtype)
                        key = f"subtype:{subtype_idx}"
                        if key in row:
                            raise DxContractError(f"duplicate terminal hierarchy path: {key}")
                        row[key] = math.prod((*edge_probabilities, float(subtype_probabilities[subtype_idx])))
        total = sum(row.values())
        if not math.isclose(total, 1.0, rel_tol=1e-5, abs_tol=1e-6):
            raise DxContractError(f"terminal hierarchy probabilities sum to {total}, expected 1")
        result.append(row)
    return result


def predictions_from_logits(
    logits: dict[str, torch.Tensor], thresholds: dict[str, Any], taxonomy: Taxonomy
) -> list[dict[str, Any]]:
    logits = apply_temperatures(logits, thresholds, taxonomy)
    stops = {target: float(thresholds["thresholds"][target]) for target in DX_TARGETS}
    batch = next(iter(logits.values())).shape[0]
    rows: list[dict[str, Any]] = []
    for idx in range(batch):
        row: dict[str, Any] = {f"{target}_index": None for target in DX_TARGETS}
        row.update({f"{target}_confidence": 0.0 for target in DX_TARGETS})

        conf, pred = _masked_top_prediction(logits["hematolymphoid_tumor_presence"][idx], None)
        row["hematolymphoid_tumor_presence_confidence"] = conf
        if pred is None or conf < stops["hematolymphoid_tumor_presence"]:
            rows.append(row)
            continue
        row["hematolymphoid_tumor_presence_index"] = pred
        if taxonomy.classes["hematolymphoid_tumor_presence"][pred] != PRESENT_LABEL:
            rows.append(row)
            continue

        parent: dict[str, int] = {}
        conf, pred = _masked_top_prediction(logits["lineage"][idx], None)
        row["lineage_confidence"] = conf
        if pred is None or conf < stops["lineage"]:
            rows.append(row)
            continue
        row["lineage_index"] = pred
        parent["lineage"] = pred

        conf, pred = _masked_top_prediction(
            logits["family"][idx], taxonomy.valid_family_mask(parent["lineage"], logits["family"].device)
        )
        row["family_confidence"] = conf
        if pred is None or conf < stops["family"]:
            rows.append(row)
            continue
        row["family_index"] = pred
        parent["family"] = pred

        conf, pred = _masked_top_prediction(
            logits["type"][idx], taxonomy.valid_type_mask(parent["family"], logits["type"].device)
        )
        row["type_confidence"] = conf
        if pred is None or conf < stops["type"]:
            rows.append(row)
            continue
        row["type_index"] = pred
        parent["type"] = pred
        if not taxonomy.has_subtype_children(pred):
            rows.append(row)
            continue

        conf, pred = _masked_top_prediction(
            logits["subtype"][idx], taxonomy.valid_subtype_mask(parent["type"], logits["subtype"].device)
        )
        row["subtype_confidence"] = conf
        if pred is not None and conf >= stops["subtype"]:
            row["subtype_index"] = pred
        rows.append(row)
    return rows
