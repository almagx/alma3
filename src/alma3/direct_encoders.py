"""Selected CpG evidence and diagnostic trunk."""

import math

import torch
from torch import nn
from torch.nn import functional as F


def high_precision(value):
    return value.float() if value.dtype in (torch.float16, torch.bfloat16) else value


class DirectDiagnosisEvidence(nn.Module):
    """Zero-initialized observed-CpG correction for each conditional diagnosis logit."""

    def __init__(self, n_cpgs, targets, *, unit_energy=False):
        super().__init__()
        from .dx import DX_TARGETS

        self.targets = DX_TARGETS
        self.widths = tuple(targets[k] for k in self.targets)
        self.unit_energy = unit_energy
        self.projection = nn.Linear(n_cpgs, sum(self.widths), bias=False)
        nn.init.zeros_(self.projection.weight)

    def forward(self, beta, observed):
        signed = torch.where(observed, 2 * high_precision(beta) - 1, 0.0)
        if self.unit_energy:
            signed = F.normalize(signed, p=2, dim=-1, eps=1.0)
        value = high_precision(self.projection(signed))
        return dict(zip(self.targets, value.split(self.widths, dim=-1)))


def quality_summaries(beta, observed, uncertainty):
    beta, uncertainty = high_precision(beta), high_precision(uncertainty)
    count = observed.sum(1, keepdim=True).to(beta.dtype)
    x = torch.where(observed, 2 * beta - 1, 0)
    denominator = count.clamp_min(1)
    return torch.cat(
        (
            count.log1p() / math.log1p(beta.shape[1]),
            torch.where(observed, uncertainty, 0).sum(1, keepdim=True) / denominator,
            x.sum(1, keepdim=True) / denominator,
            (x.square().sum(1, keepdim=True) / denominator).sqrt(),
        ),
        1,
    )


class QualityTrunk(nn.Sequential):
    def forward(self, value, quality):
        value = self[0](high_precision(value))
        value = torch.cat((value, quality.to(value.dtype)), -1)
        for layer in list(self)[1:]:
            value = layer(value)
        return value
