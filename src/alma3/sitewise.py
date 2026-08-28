from __future__ import annotations

from dataclasses import dataclass

import torch

CANONICAL_CONDITIONS: tuple[tuple[str, float | None], ...] = (
    ("clean", None),
    ("poisson_0p5", 0.5),
    ("poisson_4", 4.0),
    ("poisson_32", 32.0),
)
CANONICAL_BASE_SEED = 42
MINIMUM_RUNTIME_INPUT_CPGS = 1500


@dataclass(frozen=True)
class SitewisePresentation:
    beta_input: torch.Tensor
    input_observed: torch.Tensor
    uncertainty: torch.Tensor
    target_mask: torch.Tensor | None
    source_observed: torch.Tensor
    available_observed: torch.Tensor


def _validate_inputs(beta: torch.Tensor, observed: torch.Tensor) -> None:
    if beta.ndim != 2 or observed.shape != beta.shape:
        raise ValueError("beta and observed must have the same [B, N] shape")
    if not beta.is_floating_point() or observed.dtype != torch.bool:
        raise ValueError("beta must be floating point and observed must be bool")
    if not bool(torch.isfinite(beta).all().item()) or bool(((beta < 0) | (beta > 1)).any().item()):
        raise ValueError("beta must contain finite values in [0, 1]")
    if bool((beta[~observed] != 0).any().item()):
        raise ValueError("beta must be zero wherever observed is false")
    if bool((observed.sum(dim=1) == 0).any().item()):
        raise ValueError("every source sample must contain at least one observed CpG")


def real_coverage_presentation(
    beta: torch.Tensor,
    observed: torch.Tensor,
    coverage: torch.Tensor,
) -> SitewisePresentation:
    """Use measured beta and coverage without adding simulated sampling noise."""

    _validate_inputs(beta, observed)
    integer_dtypes = {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
        torch.uint16,
        torch.uint32,
    }
    if coverage.shape != beta.shape or coverage.dtype not in integer_dtypes:
        raise ValueError("coverage must be an integer tensor with shape [B, N]")
    if coverage.dtype in {torch.int8, torch.int16, torch.int32, torch.int64} and bool(
        (coverage < 0).any().item()
    ):
        raise ValueError("coverage must be nonnegative")
    count_integer = coverage.to(device=beta.device, dtype=torch.int64)
    input_observed = count_integer > 0
    if not torch.equal(input_observed, observed):
        raise ValueError("positive coverage must exactly match observed CpGs")
    count_float = count_integer.to(dtype=torch.float32)
    return SitewisePresentation(
        beta_input=torch.where(input_observed, beta, torch.zeros_like(beta)),
        input_observed=input_observed,
        uncertainty=torch.where(
            input_observed,
            count_float.clamp_min(1).rsqrt(),
            torch.zeros_like(count_float),
        ).to(dtype=beta.dtype),
        target_mask=None,
        source_observed=observed.clone(),
        available_observed=input_observed.clone(),
    )
