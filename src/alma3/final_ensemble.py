"""Inference-only implementation of the frozen final-ten selection."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

from alma3.dx import load_dx, load_taxonomy
from alma3.hashes import sha256_file
from alma3.terminal import terminal_log_probabilities, terminal_paths

from .terminal import terminal_indices

MINIMUM_OBSERVED_CPGS = 5000
PROBABILITY_REPRESENTATION = dict(name="final_ten_member_probabilities", version=1, dtype="float32", dimensions=1090)


class DiagnosisResidual(nn.Module):
    def __init__(self, state):
        super().__init__()
        self.register_buffer("mean", state["mean"])
        self.register_buffer("scale", state["scale"])
        self.linear = nn.Linear(len(self.mean), 109, dtype=self.mean.dtype)
        self.load_state_dict(state)

    def forward(self, embedding, native):
        features = (embedding - self.mean) / self.scale / math.sqrt(len(self.mean))
        return (native + self.linear(features)).log_softmax(-1)


class SparseTemplates(nn.Module):
    """The frozen DF likelihood-ratio classifier; no fitting at inference."""

    def __init__(self, means, groups, sites, temperature):
        super().__init__()
        classes, n = means.shape
        regions = int(groups.max()) + 1
        strength = (means[1:] - means[:1]).abs()
        group = groups[None].expand(classes - 1, -1)
        best = torch.zeros(classes - 1, regions).scatter_reduce_(1, group, strength, reduce="amax")
        candidate = torch.where(strength == best.gather(1, group), torch.arange(n)[None], n)
        index = torch.full((classes - 1, regions), n, dtype=torch.long).scatter_reduce_(
            1, group, candidate, reduce="amin"
        )
        selected = torch.argsort(best, dim=1, descending=True, stable=True)[:, : min(sites, regions)]
        index = index.gather(1, selected)
        disease = means[1:].gather(1, index)
        normal = means[:1].expand(classes - 1, -1).gather(1, index)
        self.register_buffer("index", index)
        self.register_buffer("coefficient", torch.logit(disease) - torch.logit(normal))
        self.register_buffer("offset", torch.log1p(-disease) - torch.log1p(-normal))
        self.temperature = float(temperature)

    def forward(self, beta, observed):
        mask = observed[:, self.index].bool()
        value = torch.where(mask, beta[:, self.index].float(), 0.0)
        scores = torch.where(mask, value * self.coefficient + self.offset, 0.0).sum(-1)
        scores /= mask.sum(-1).clamp_min(1)
        scores = torch.cat([torch.zeros_like(scores[:, :1]), scores], 1)
        prior = torch.full((109,), 0.5 / 108, device=scores.device, dtype=scores.dtype)
        prior[0] = 0.5
        return (scores / self.temperature + prior.log()).log_softmax(-1)


def mix_stages(native, calibrated, native_roles, paths):
    """Reproduce the selected presence/type/subtype factorization in FP64."""
    types = torch.unique(paths[:, 3])
    parents = torch.searchsorted(types, paths[1:, 3])
    parts = {}
    for lp, roles in ((native, set(native_roles)), (calibrated, {"presence", "type", "subtype"} - set(native_roles))):
        lp = lp.double()
        mass = torch.stack([lp[:, paths[:, 3] == t].logsumexp(1) for t in types], 1)
        if "presence" in roles:
            parts["presence"] = torch.stack((mass[:, 0], mass[:, 1:].logsumexp(1)), 1).log_softmax(1)
        if "type" in roles:
            parts["type"] = mass[:, 1:].log_softmax(1)
        if "subtype" in roles:
            parts["subtype"] = lp[:, 1:] - mass[:, parents]
    presence = parts["presence"]
    return torch.cat((presence[:, :1], presence[:, 1:] + parts["type"][:, parents - 1] + parts["subtype"]), 1)


def combine_members(parts, selection):
    values = torch.stack([parts[k].double().exp() for k in selection["team_a"]])
    trimmed = values.sort(dim=0).values[1:-1].mean(0)
    trimmed = trimmed / trimmed.sum(1, keepdim=True)
    fraction = selection["team_a_transform_weight"]
    probability = selection["team_a_weight"] * (fraction * trimmed + (1 - fraction) * values.mean(0))
    for key, weight in selection["team_b_member_weights"].items():
        probability = probability + weight * parts[key].double().exp()
    return probability.log()


class FinalEnsemble:
    """Load every inference dependency from a relocatable, checksummed bundle."""

    def __init__(self, root, *, device="cpu", verify=True, precision="bf16", batch_size=64):
        self.root = Path(root).resolve()
        self.manifest = json.loads((self.root / "ensemble.json").read_text())
        if self.manifest["kind"] != "alma3_final_ten_v1":
            raise ValueError("unsupported ensemble release")
        if verify:
            hashes = json.loads((self.root / "SHA256SUMS.json").read_text())
            for relative, expected in hashes.items():
                if sha256_file(self.path(relative)) != expected:
                    raise ValueError(f"ensemble file changed: {relative}")
        if precision not in ("bf16", "fp32") or batch_size < 1:
            raise ValueError("invalid explicit inference policy")
        policy = self.manifest["execution_policy"]
        if (precision, batch_size) != (policy["precision"], policy["batch_size"]):
            raise ValueError("execution must use the certified precision and batch size")
        self.precision, self.batch_size = precision, int(batch_size)
        self.device = torch.device(device)
        self.taxonomy = load_taxonomy(self.path(self.manifest["taxonomy"]))
        self.paths = torch.tensor(terminal_paths(self.taxonomy), device=self.device)
        self.models, self.heads, self.partitions = {}, {}, {}
        torch.backends.cuda.matmul.allow_tf32 = False
        for row in self.manifest["members"]:
            key = row["key"]
            if row["kind"] == "sparse_templates":
                state = torch.load(self.path(row["templates"]), map_location="cpu", weights_only=True)
                model = SparseTemplates(state["means"], state["group_index"], row["sites"], row["temperature"])
            else:
                model = load_dx(self.path(row["export"]), device=self.device)
            self.models[key] = model.to(self.device).eval().requires_grad_(False)
            if "residual" in row:
                state = torch.load(self.path(row["residual"]), map_location="cpu", weights_only=True)["state"]
                self.heads[key] = DiagnosisResidual(state).to(self.device).eval().requires_grad_(False)
            if "partition" in row:
                self.partitions[key] = torch.tensor(np.load(self.path(row["partition"])), device=self.device)

    def path(self, relative):
        candidate = self.root / relative
        if Path(relative).is_absolute() or not candidate.resolve().is_relative_to(self.root):
            raise ValueError("ensemble dependency must remain inside the bundle")
        return candidate

    def _raw(self, model, inputs, *, embedding=False):
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.precision == "bf16"):
            value = model(*inputs, **({"_return_embedding": True} if embedding else {}))
        if embedding:
            value, embedding_value = value
            return {k: v.float() for k, v in value.items()}, embedding_value.double()
        return {k: v.float() for k, v in value.items()}

    def _member(self, row, inputs, *, embeddings=None):
        key, model = row["key"], self.models[row["key"]]
        if row["kind"] == "sparse_templates":
            return model(inputs[0], inputs[1])
        capture = key in self.heads or embeddings is not None
        value = self._raw(model, inputs, embedding=capture)
        raw, embedding = value if capture else (value, None)
        if embeddings is not None:
            embeddings[key] = embedding.float()
        native = terminal_log_probabilities(raw, self.taxonomy, output_mode=model.config.output_mode)
        if key in self.heads:
            return self.heads[key](embedding, native.double())
        if "temperatures" in row:
            calibrated = terminal_log_probabilities(
                {k: v / float(row["temperatures"][j]) for j, (k, v) in enumerate(raw.items())},
                self.taxonomy,
                output_mode="conditional_heads_v1",
            )
            return (
                mix_stages(native, calibrated, row["native_roles"], self.paths) if "native_roles" in row else calibrated
            )
        if key in self.partitions:
            partition = self.partitions[key]
            beta, observed, uncertainty, chrom, positions = inputs
            masks = [observed & (partition != j)[None] for j in range(2)]
            eligible = torch.stack([m.sum(1) >= MINIMUM_OBSERVED_CPGS for m in masks], 1).all(1)
            views = [native]
            for mask in masks:
                raw = self._raw(
                    model, (beta, torch.where(eligible[:, None], mask, observed), uncertainty, chrom, positions)
                )
                views.append(terminal_log_probabilities(raw, self.taxonomy, output_mode=model.config.output_mode))
            averaged = torch.logsumexp(torch.stack(views), 0) - math.log(3)
            return torch.where(eligible[:, None], averaged, native)
        return native

    def predict(self, beta, observed, uncertainty, chrom, positions, *, return_members=False):
        return self._predict(beta, observed, uncertainty, chrom, positions, return_members=return_members)

    def predict_with_map_features(self, beta, observed, uncertainty, chrom, positions):
        """Capture the exact ten member probabilities used to fit the tumor map."""
        logp, parts = self.predict(beta, observed, uncertainty, chrom, positions, return_members=True)
        keys = [row["key"] for row in self.manifest["members"]]
        if len(keys) != 10 or len(set(keys)) != 10 or list(parts) != keys:
            raise ValueError("probability maps require ten members in manifest order")
        values = torch.stack([parts[key].exp() for key in keys], dim=1).float()
        if (
            values.shape != (len(beta), 10, 109)
            or not torch.isfinite(values).all()
            or (values < 0).any()
            or (values > 1).any()
            or not torch.allclose(values.sum(2), torch.ones_like(values[:, :, 0]), rtol=0, atol=3e-6)
        ):
            raise ValueError("invalid member probability blocks")
        return logp, values.reshape(len(beta), 1090)

    def predict_with_embedding(self, beta, observed, uncertainty, chrom, positions):
        """Return the unchanged diagnosis and nine ordered full-input trunk blocks."""
        embeddings = {}
        logp = self._predict(beta, observed, uncertainty, chrom, positions, embeddings=embeddings)
        keys = [row["key"] for row in self.manifest["members"] if row["kind"] != "sparse_templates"]
        if len(keys) != 9 or list(embeddings) != keys:
            raise ValueError("map representation requires nine ordered neural members")
        blocks = [embeddings[key][: len(beta)] for key in keys]
        if any(v.shape != (len(beta), 1024) or v.dtype != torch.float32 for v in blocks):
            raise ValueError("map representation requires FP32 1024-feature blocks")
        result = torch.cat(blocks, dim=1)
        if not torch.isfinite(result).all():
            raise ValueError("nonfinite map representation")
        return logp, result

    @torch.no_grad()
    def _predict(self, beta, observed, uncertainty, chrom, positions, *, return_members=False, embeddings=None):
        rows = len(beta)
        if not 0 < rows <= self.batch_size:
            raise ValueError("prediction chunk exceeds the bound execution batch")
        if beta.dtype != torch.float32 or uncertainty.dtype != torch.float32 or observed.dtype != torch.bool:
            raise ValueError("inputs require FP32 methylation/uncertainty and a boolean observation mask")
        counts = observed.sum(1)
        below_minimum = counts < MINIMUM_OBSERVED_CPGS
        if below_minimum.any():
            sparse = dict(zip(below_minimum.nonzero().flatten().tolist(), counts[below_minimum].tolist()))
            raise ValueError(
                f"each sample requires at least {MINIMUM_OBSERVED_CPGS:,} observed CpGs; "
                f"below-minimum batch rows (zero-based) and counts: {sparse}"
            )
        inputs = [value.to(self.device) for value in (beta, observed, uncertainty, chrom, positions)]
        if rows < self.batch_size:
            inputs = [torch.cat((v, v[-1:].expand(self.batch_size - rows, *v.shape[1:]))) for v in inputs]
        options = {} if embeddings is None else {"embeddings": embeddings}
        parts = {row["key"]: self._member(row, inputs, **options)[:rows] for row in self.manifest["members"]}
        logp = combine_members(parts, self.manifest)
        if not torch.isfinite(logp).all():
            raise ValueError("nonfinite ensemble probability")
        return (logp, parts) if return_members else logp

    def decode(self, logp):
        return terminal_indices(logp, self.taxonomy, rule=self.manifest["decision_rule"])
