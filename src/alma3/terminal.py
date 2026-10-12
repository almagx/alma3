"""Ordered terminal distributions for the selected 3.1.0 classifiers."""

from __future__ import annotations
import hashlib
import json
import torch
from torch import nn
from .dx import DX_TARGETS, DxContractError

MODE = "joint_terminal_v1"
SUBTYPE_CUTOFF = 0.3295999742967636


def selected_path_scores(probabilities, taxonomy):
    """Frozen type-first paths, conditional ancestor scores and joint subtype scores."""
    paths = torch.tensor(terminal_paths(taxonomy), device=probabilities.device)
    if (
        probabilities.ndim != 2
        or probabilities.shape[1] != len(paths)
        or not torch.isfinite(probabilities).all()
        or (probabilities < 0).any()
        or not torch.allclose(probabilities.sum(1), torch.ones_like(probabilities[:, 0]), atol=3e-6, rtol=0)
    ):
        raise DxContractError("expected normalized ordered terminal probabilities")
    selected = terminal_indices(
        probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny).log(), taxonomy, rule="type_first_v1"
    )
    chosen = paths[selected]
    marginal = [
        torch.stack([probabilities[:, paths[:, j] == k].sum(1) for k in range(len(taxonomy.classes[target]))], 1)
        for j, target in enumerate(DX_TARGETS)
    ]
    conditional = [marginal[0]]
    rows = torch.arange(len(probabilities), device=probabilities.device)
    for j in range(1, 4):
        denominator = marginal[j - 1][rows, chosen[:, j - 1].clamp_min(0)]
        conditional.append(marginal[j] / denominator[:, None].clamp_min(torch.finfo(probabilities.dtype).tiny))
    conditional.append(marginal[4])
    scores = torch.stack([value[rows, chosen[:, j].clamp_min(0)] for j, value in enumerate(conditional)], 1)
    scores = scores.masked_fill(chosen < 0, 0.0)
    return selected, chosen, scores, conditional


def taxonomy_payload(taxonomy):
    return dict(
        kind="alma3_taxonomy",
        schema_version=1,
        levels={k: list(v) for k, v in taxonomy.classes.items()},
        **{
            k: {p: list(c) for p, c in getattr(taxonomy, k).items()}
            for k in ("family_by_lineage", "type_by_family", "subtype_by_type")
        },
    )


def terminal_paths(taxonomy):
    if taxonomy.classes[DX_TARGETS[0]] != ("absent", "present"):
        raise DxContractError("terminal distribution requires ordered absent/present classes")
    family_parent = {c: p for p, cs in taxonomy.family_by_lineage.items() for c in cs}
    type_parent = {c: p for p, cs in taxonomy.type_by_family.items() for c in cs}
    subtype_parent = {c: p for p, cs in taxonomy.subtype_by_type.items() for c in cs}
    paths = [(0, -1, -1, -1, -1)]
    leaves = [(subtype_parent[s], i) for i, s in enumerate(taxonomy.classes["subtype"])]
    leaves += [(t, -1) for i, t in enumerate(taxonomy.classes["type"]) if not taxonomy.has_subtype_children(i)]
    for kind, subtype in leaves:
        family = type_parent[kind]
        paths.append(
            (
                1,
                taxonomy.classes["lineage"].index(family_parent[family]),
                taxonomy.classes["family"].index(family),
                taxonomy.classes["type"].index(kind),
                subtype,
            )
        )
    return tuple(paths)


class CosineProjection(nn.Linear):
    """Direction-based diagnosis logits with one shared learned positive scale."""

    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features)
        self.log_scale = nn.Parameter(torch.tensor(20.0).log())

    def forward(self, embedding):
        from .direct_encoders import high_precision

        value = torch.nn.functional.normalize(high_precision(embedding), dim=-1, eps=1e-12)
        weight = torch.nn.functional.normalize(high_precision(self.weight), dim=-1, eps=1e-12)
        similarity = torch.nn.functional.linear(value, weight)
        return high_precision(similarity) * self.log_scale.exp() + self.bias


class TerminalHead(nn.Module):
    def __init__(self, embedding_dim, taxonomy):
        super().__init__()
        paths = torch.tensor(terminal_paths(taxonomy))
        self.sizes = tuple(len(taxonomy.classes[k]) for k in DX_TARGETS)
        self.projection = nn.Linear(embedding_dim, len(paths))
        membership = torch.cat([paths[:, j][None, :] == torch.arange(n)[:, None] for j, n in enumerate(self.sizes)])
        if not membership.any(1).all():
            raise DxContractError("every taxonomy node requires a terminal descendant")
        self.register_buffer("membership", membership, persistent=False)
        digest = hashlib.sha256(
            json.dumps(taxonomy_payload(taxonomy), sort_keys=True, separators=(",", ":")).encode()
        ).digest()
        self.register_buffer("taxonomy_digest", torch.tensor(list(digest), dtype=torch.uint8))

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        value = state_dict.get(prefix + "taxonomy_digest")
        if value is not None and not torch.equal(value.cpu(), self.taxonomy_digest.cpu()):
            error_msgs.append("terminal weights are bound to a different ordered taxonomy")
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def forward(self, embedding):
        terminal = self.projection(embedding).float()
        values = terminal[:, None, :].masked_fill(~self.membership[None], -torch.inf).logsumexp(-1)
        return dict(zip(DX_TARGETS, values.split(self.sizes, dim=1), strict=True))


class ConditionalHeads(nn.ModuleDict):
    """The five independent linear heads from ALMA 3.0.0."""

    def __init__(self, embedding_dim, taxonomy):
        from .dx import _head_key

        super().__init__({_head_key(k): nn.Linear(embedding_dim, len(taxonomy.classes[k])) for k in DX_TARGETS})
        digest = hashlib.sha256(
            json.dumps(taxonomy_payload(taxonomy), sort_keys=True, separators=(",", ":")).encode()
        ).digest()
        self.register_buffer("taxonomy_digest", torch.tensor(list(digest), dtype=torch.uint8))

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        value = state_dict.get(prefix + "taxonomy_digest")
        if value is not None and not torch.equal(value.cpu(), self.taxonomy_digest.cpu()):
            error_msgs.append("conditional weights are bound to a different ordered taxonomy")
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def forward(self, embedding):
        from .dx import _head_key

        return {k: self[_head_key(k)](embedding).float() for k in DX_TARGETS}


class TypeSubtypeHead(nn.Module):
    """Direct type/absence competition followed by within-type subtype evidence."""

    def __init__(self, embedding_dim, taxonomy):
        super().__init__()
        paths = torch.tensor(terminal_paths(taxonomy))
        self.type_projection = nn.Linear(embedding_dim, len(taxonomy.classes["type"]) + 1)
        self.subtype_projection = nn.Linear(embedding_dim, len(taxonomy.classes["subtype"]))
        self.sizes = tuple(len(taxonomy.classes[k]) for k in DX_TARGETS)
        self.register_buffer("paths", paths)
        self.register_buffer(
            "siblings", paths[1 : 1 + self.sizes[-1], 3, None] == paths[1 : 1 + self.sizes[-1], 3][None, :]
        )
        self.register_buffer(
            "membership",
            torch.cat([paths[:, j][None, :] == torch.arange(n)[:, None] for j, n in enumerate(self.sizes)]),
        )

    def forward(self, embedding):
        types = self.type_projection(embedding)
        subtypes = self.subtype_projection(embedding)
        types = types if types.dtype == torch.float64 else types.float()
        subtypes = subtypes if subtypes.dtype == torch.float64 else subtypes.float()
        type_logp = types.log_softmax(-1)
        normalizer = subtypes[:, None, :].masked_fill(~self.siblings[None], -torch.inf).logsumexp(-1)
        conditional = subtypes - normalizer
        terminal = type_logp[:, (self.paths[:, 3] + 1).clamp_min(0)]
        positive = torch.cat(
            (
                torch.zeros_like(conditional[:, :1]),
                conditional,
                torch.zeros(
                    len(embedding),
                    len(self.paths) - len(conditional[0]) - 1,
                    device=embedding.device,
                    dtype=conditional.dtype,
                ),
            ),
            1,
        )
        terminal = terminal + positive
        values = terminal[:, None, :].masked_fill(~self.membership[None], -torch.inf).logsumexp(-1)
        return dict(zip(DX_TARGETS, values.split(self.sizes, dim=1), strict=True))


def terminal_log_probabilities(logits, taxonomy, *, output_mode=None):
    paths = terminal_paths(taxonomy)
    if output_mode in (MODE, "type_subtype_v1"):
        leaves = [logits[DX_TARGETS[0]][:, 0]]
        leaves += [logits["subtype"][:, i] for i in range(len(taxonomy.classes["subtype"]))]
        leaves += [logits["type"][:, p[3]] for p in paths[1:] if p[4] < 0]
        return torch.stack(leaves, 1).float().log_softmax(-1)
    if output_mode not in (None, "conditional_heads_v1"):
        raise DxContractError("unsupported terminal output mode")
    conditional = {k: v.float().log_softmax(-1) for k, v in logits.items()}
    for j, key in ((2, "family"), (3, "type"), (4, "subtype")):
        values = torch.full_like(logits[key].float(), -torch.inf)
        for parent in range(len(taxonomy.classes[DX_TARGETS[j - 1]])):
            mask = getattr(taxonomy, f"valid_{key}_mask")(parent, values.device)
            if mask.any():
                values[:, mask] = logits[key][:, mask].float().log_softmax(-1)
        conditional[key] = values
    return torch.stack([sum(conditional[k][:, p[j]] for j, k in enumerate(DX_TARGETS) if p[j] >= 0) for p in paths], 1)


def terminal_loss_terms(logits, targets, taxonomy):
    """Marginal NLL for all known labels; ignored padding contributes zero."""
    logp = terminal_log_probabilities(logits, taxonomy, output_mode=MODE)
    truth = torch.stack([targets[k] for k in DX_TARGETS], 1)
    sizes = truth.new_tensor([len(taxonomy.classes[k]) for k in DX_TARGETS])
    if bool(((truth < -1) | (truth >= sizes)).any()):
        raise DxContractError("terminal target index is outside the bound taxonomy")
    paths = truth.new_tensor(terminal_paths(taxonomy))
    compatible = ((truth[:, None, :] < 0) | (truth[:, None, :] == paths[None])).all(-1)
    if not bool(compatible.any(1).all()):
        raise DxContractError("known diagnostic labels have no compatible terminal diagnosis")
    known = (truth >= 0).any(1)
    losses = -logp.masked_fill(~compatible, -torch.inf).logsumexp(1)
    return (losses * known).sum(), known.sum().to(logp.dtype)


def terminal_indices(logp, taxonomy, *, rule="terminal_map_v1"):
    """Choose a complete diagnosis with an explicitly bound decision rule."""
    paths = torch.tensor(terminal_paths(taxonomy), device=logp.device)
    if logp.ndim != 2 or logp.shape[1] != len(paths) or not torch.isfinite(logp).all():
        raise ValueError("decoding requires finite ordered terminal log probabilities")
    if rule == "terminal_map_v1":
        return logp.argmax(1)
    if rule != "type_first_v1":
        raise ValueError("unsupported terminal decision rule")
    types = paths[:, 3]
    groups = torch.unique(types)
    mass = torch.stack([logp[:, types == kind].logsumexp(1) for kind in groups], 1)
    selected = groups[mass.argmax(1)]
    return logp.masked_fill(types[None] != selected[:, None], -torch.inf).argmax(1)
