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
    foundation: FoundationConfig | None
    targets: dict[str, int]
    hidden_dim: int = 384
    dropout: float = 0.1
    direct_cpg_branch: str = 'quality256_v1'
    n_cpgs: int = 387849
    output_mode: str = 'joint_terminal_v1'
    terminal_taxonomy: dict[str, Any] | None = None
    refinement_mode: str | None = None
    terminal_geometry: str | None = None
    direct_logit_correction: str | None = None

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> DxConfig:
        unknown = sorted(set(raw) - set(cls.__dataclass_fields__))
        if unknown:
            raise ConfigError(f"unknown Dx config field(s): {', '.join(unknown)}")
        config = cls(**raw)
        config.validate()
        return config

    def validate(self) -> None:
        from .terminal import terminal_paths
        from .dx import Taxonomy
        if (self.foundation is not None or self.hidden_dim != 384 or self.n_cpgs != 387849
                or self.direct_cpg_branch not in ('quality256_v1', 'quality256_linear_v1', 'quality256_neighbor_v1')
                or self.output_mode not in ('joint_terminal_v1', 'conditional_heads_v1', 'type_subtype_v1')
                or self.refinement_mode not in (None, 'feature_attention2_v1')
                or self.terminal_geometry not in (None, 'cosine_v1')
                or self.direct_logit_correction not in (None, 'cpg_linear_v1')):
            raise ConfigError('unsupported ALMA3 3.1.0 diagnostic architecture')
        if self.direct_cpg_branch == 'quality256_linear_v1' and self.refinement_mode is not None:
            raise ConfigError('additive encoding requires no attention')
        if self.direct_cpg_branch == 'quality256_neighbor_v1' and (
                self.refinement_mode != 'feature_attention2_v1' or self.output_mode != 'conditional_heads_v1'):
            raise ConfigError('neighbor evidence requires conditional feature attention')
        if self.direct_logit_correction and (self.direct_cpg_branch != 'quality256_v1'
                or self.output_mode != 'conditional_heads_v1' or self.refinement_mode != 'feature_attention2_v1'):
            raise ConfigError('direct correction requires conditional feature attention')
        if self.terminal_geometry and self.output_mode == 'conditional_heads_v1':
            raise ConfigError('cosine geometry requires terminal or type/subtype heads')
        _require_probability(self.dropout, 'dropout')
        taxonomy = Taxonomy.from_dict(self.terminal_taxonomy)
        taxonomy.validate_sizes(self.targets)
        if len(terminal_paths(taxonomy)) != 109:
            raise ConfigError('3.1.0 requires 109 ordered terminal diagnoses')

    @property
    def input_cpg_count(self) -> int:
        return self.n_cpgs

    @property
    def embedding_dim(self) -> int:
        return 1024

    @property
    def has_branch_ablations(self) -> bool:
        return False

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if self.refinement_mode is None:
            del payload['refinement_mode']
        if self.terminal_geometry is None:
            del payload['terminal_geometry']
        if self.direct_logit_correction is None:
            del payload['direct_logit_correction']
        return payload
