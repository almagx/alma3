"""Frozen observed-neighbor CpG interactions."""

import hashlib

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def genomic_groups(chromosome, position, *, maximum_sites=32, maximum_span=100000):
    chromosome = np.asarray(chromosome)
    position = np.asarray(position)
    if position.ndim != 1 or not np.issubdtype(position.dtype, np.integer) or len(position) != len(chromosome):
        raise ValueError("genomic grouping requires exact integer base-pair starts")
    order = np.lexsort((position, chromosome))
    groups = []
    current = []
    for i in order.tolist():
        if current and (
            chromosome[i] != chromosome[current[0]]
            or len(current) == maximum_sites
            or position[i] - position[current[0]] > maximum_span
        ):
            groups.append(current)
            current = []
        current.append(i)
    if current:
        groups.append(current)
    indices = np.full((len(groups), maximum_sites), len(order), dtype=np.int64)
    relative = np.zeros(indices.shape, dtype=np.float32)
    for j, group in enumerate(groups):
        indices[j, : len(group)] = group
        relative[j, : len(group)] = (position[group] - position[group[0]]) / float(maximum_span)
    return indices, relative


class NeighborEvidence(nn.Module):
    """Per-CpG agreement with other observed sites in its fixed genomic group."""

    def __init__(self, n_cpgs, width=256, groups=22289):
        super().__init__()
        self.n_cpgs = n_cpgs
        self._bound = False
        self.projection = nn.Linear(n_cpgs, width, bias=False)
        nn.init.zeros_(self.projection.weight)
        self.register_buffer("indices", torch.full((groups, 32), n_cpgs, dtype=torch.long))
        self.register_buffer("order", torch.zeros(n_cpgs, dtype=torch.long))
        self.register_buffer("reference_digest", torch.zeros(32, dtype=torch.uint8))

    def bind_reference(self, chromosome, position):
        chromosome = np.asarray(chromosome)
        position = np.asarray(position)
        indices, _ = genomic_groups(chromosome, position)
        if indices.shape != tuple(self.indices.shape) or len(chromosome) != self.n_cpgs:
            raise ValueError("neighbor evidence differs from the complete CpG partition")
        flat = indices.flatten()
        locations = np.flatnonzero(flat < self.n_cpgs)
        order = np.empty(self.n_cpgs, dtype=np.int64)
        order[flat[locations]] = locations
        with torch.no_grad():
            self.indices.copy_(torch.from_numpy(indices))
            self.order.copy_(torch.from_numpy(order))
            digest = hashlib.sha256(chromosome.tobytes() + position.tobytes()).digest()
            self.reference_digest.copy_(torch.tensor(list(digest), dtype=torch.uint8))
        self._bound = True

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        ix = state_dict.get(prefix + "indices")
        order = state_dict.get(prefix + "order")
        digest = state_dict.get(prefix + "reference_digest")
        if ix is not None and order is not None and digest is not None:
            flat = ix.cpu().flatten()
            order = order.cpu()
            valid = (
                bool((flat >= 0).all())
                and bool((flat <= self.n_cpgs).all())
                and torch.equal(flat[flat < self.n_cpgs].sort().values, torch.arange(self.n_cpgs))
                and bool((order >= 0).all())
                and bool((order < len(flat)).all())
                and torch.equal(flat[order], torch.arange(self.n_cpgs))
                and bool(digest.any())
            )
            if not valid:
                error_msgs.append("neighbor evidence checkpoint has an invalid CpG partition")
            else:
                self._bound = True
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def features(self, beta, observed):
        if not self._bound:
            raise ValueError("neighbor evidence requires its verified reference partition")
        beta = beta.float() if beta.dtype in (torch.float16, torch.bfloat16) else beta
        x = F.pad(torch.where(observed, 2 * beta - 1, 0.0), (0, 1))[:, self.indices]
        mask = F.pad(observed, (0, 1))[:, self.indices]
        count = mask.sum(-1, keepdim=True)
        q = x * (x.sum(-1, keepdim=True) - x) / (count - 1).clamp_min(1)
        return torch.where(mask & (count > 1), q, 0.0).flatten(1)[:, self.order]

    def forward(self, beta, observed):
        return self.projection(self.features(beta, observed))
