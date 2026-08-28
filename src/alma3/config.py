from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any


class ConfigError(ValueError):
    """Raised when an ALMA3 model configuration is invalid."""


CHROMOSOME_ARM_NAMES = tuple(
    f"chr{chromosome}{arm}"
    for chromosome in (*range(1, 23), "X")
    for arm in ("p", "q")
)


def _require_bool(value: Any, name: str) -> None:
    if type(value) is not bool:
        raise ConfigError(f"{name} must be a boolean")


def _require_probability(value: Any, name: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) < 1.0
    ):
        raise ConfigError(f"{name} must be a finite number in [0, 1)")


@dataclass(frozen=True)
class FoundationConfig:
    architecture_version: int
    n_cpgs: int
    chromosome_cpg_counts: list[int]
    arm_cpg_counts: dict[str, int]
    d_model: int = 1536
    n_layers: int = 36
    n_heads: int = 24
    mlp_ratio: int = 4
    patch_size: int = 64
    pos_bands: int = 12
    n_chromosomes: int = 24
    chr_dim: int = 32
    cpg_id_dim: int = 32
    dropout: float = 0.0
    activation_checkpointing: bool = False

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> FoundationConfig:
        if type(raw.get("architecture_version")) is not int or raw["architecture_version"] != 5:
            raise ConfigError("architecture_version must be 5")
        if "arm_cpg_counts" not in raw:
            raise ConfigError("foundation config requires arm_cpg_counts")
        unknown = sorted(set(raw) - set(cls.__dataclass_fields__))
        if unknown:
            raise ConfigError(f"unknown foundation config field(s): {', '.join(unknown)}")
        config = cls(**raw)
        config.validate()
        return config

    def validate(self) -> None:
        if type(self.architecture_version) is not int or self.architecture_version != 5:
            raise ConfigError("architecture_version must be 5")
        dimensions = {
            "n_cpgs": self.n_cpgs,
            "d_model": self.d_model,
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "mlp_ratio": self.mlp_ratio,
            "patch_size": self.patch_size,
            "pos_bands": self.pos_bands,
            "n_chromosomes": self.n_chromosomes,
            "chr_dim": self.chr_dim,
            "cpg_id_dim": self.cpg_id_dim,
        }
        invalid = [name for name, value in dimensions.items() if type(value) is not int or value < 1]
        if invalid:
            raise ConfigError(f"foundation dimensions must be positive integers: {', '.join(invalid)}")
        if not isinstance(self.chromosome_cpg_counts, list):
            raise ConfigError("chromosome_cpg_counts must be a list")
        if len(self.chromosome_cpg_counts) != self.n_chromosomes:
            raise ConfigError("chromosome_cpg_counts length must equal n_chromosomes")
        if any(type(count) is not int or count < 0 for count in self.chromosome_cpg_counts):
            raise ConfigError("chromosome_cpg_counts must contain non-negative integers")
        if sum(self.chromosome_cpg_counts) != self.n_cpgs:
            raise ConfigError("chromosome_cpg_counts must sum to n_cpgs")
        if self.n_chromosomes != 24:
            raise ConfigError("n_chromosomes must be 24")
        if self.chromosome_cpg_counts[23] != 0:
            raise ConfigError("chromosome_cpg_counts must exclude chrY")
        if not isinstance(self.arm_cpg_counts, dict):
            raise ConfigError("arm_cpg_counts must be an object")
        missing = sorted(set(CHROMOSOME_ARM_NAMES) - set(self.arm_cpg_counts))
        extra = sorted(set(self.arm_cpg_counts) - set(CHROMOSOME_ARM_NAMES))
        if missing or extra:
            details = []
            if missing:
                details.append(f"missing: {', '.join(missing)}")
            if extra:
                details.append(f"unexpected: {', '.join(extra)}")
            raise ConfigError(
                "arm_cpg_counts must contain exactly chr1p through chrXq "
                f"({'; '.join(details)})"
            )
        if any(
            type(self.arm_cpg_counts[name]) is not int or self.arm_cpg_counts[name] < 0
            for name in CHROMOSOME_ARM_NAMES
        ):
            raise ConfigError("arm_cpg_counts must contain non-negative integers")
        ordered = self.ordered_arm_cpg_counts
        if sum(ordered) != self.n_cpgs:
            raise ConfigError("arm_cpg_counts must sum to n_cpgs")
        for chromosome_index, chromosome_count in enumerate(self.chromosome_cpg_counts[:23]):
            arm_offset = 2 * chromosome_index
            if sum(ordered[arm_offset : arm_offset + 2]) != chromosome_count:
                chromosome = chromosome_index + 1 if chromosome_index < 22 else "X"
                raise ConfigError(f"chr{chromosome} arm counts must sum to its chromosome count")
        if self.d_model % self.n_heads != 0:
            raise ConfigError("d_model must be divisible by n_heads")
        if self.patch_size != 64:
            raise ConfigError("architecture 5 requires patch_size=64")
        if self.pos_bands != 12 or self.chr_dim != 32 or self.cpg_id_dim != 32:
            raise ConfigError("architecture 5 requires pos_bands=12, chr_dim=32, and cpg_id_dim=32")
        _require_probability(self.dropout, "dropout")
        _require_bool(self.activation_checkpointing, "activation_checkpointing")

    @property
    def ordered_arm_cpg_counts(self) -> tuple[int, ...]:
        return tuple(self.arm_cpg_counts[name] for name in CHROMOSOME_ARM_NAMES)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DxConfig:
    foundation: FoundationConfig
    targets: dict[str, int]
    hidden_dim: int = 2048
    dropout: float = 0.1

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> DxConfig:
        unknown = sorted(set(raw) - set(cls.__dataclass_fields__))
        if unknown:
            raise ConfigError(f"unknown Dx config field(s): {', '.join(unknown)}")
        foundation = raw.get("foundation")
        targets = raw.get("targets")
        if not isinstance(foundation, dict):
            raise ConfigError("Dx config requires foundation object")
        if not isinstance(targets, dict):
            raise ConfigError("Dx config requires targets object")
        config = cls(
            foundation=FoundationConfig.from_dict(foundation),
            targets=dict(targets),
            hidden_dim=raw.get("hidden_dim", 2048),
            dropout=raw.get("dropout", 0.1),
        )
        config.validate()
        return config

    def validate(self) -> None:
        required = {
            "hematolymphoid_tumor_presence",
            "lineage",
            "family",
            "type",
            "subtype",
        }
        if set(self.targets) != required:
            missing = sorted(required - set(self.targets))
            extra = sorted(set(self.targets) - required)
            raise ConfigError(f"Dx targets must match exactly; missing={missing}, extra={extra}")
        invalid = [name for name, size in self.targets.items() if type(size) is not int or size < 1]
        if invalid:
            raise ConfigError(f"Dx target sizes must be positive integers: {', '.join(sorted(invalid))}")
        if type(self.hidden_dim) is not int or self.hidden_dim < 1:
            raise ConfigError("hidden_dim must be positive")
        _require_probability(self.dropout, "dropout")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["foundation"] = self.foundation.to_dict()
        return payload
