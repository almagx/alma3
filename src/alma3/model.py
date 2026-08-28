from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from safetensors.torch import load_file
from torch import nn
from torch.utils.checkpoint import checkpoint

from .config import FoundationConfig


def positional_features(pos: torch.Tensor, bands: int) -> torch.Tensor:
    if bands <= 0:
        return pos[..., None]
    freqs = torch.arange(1, bands + 1, device=pos.device, dtype=pos.dtype) * torch.pi
    phase = pos[..., None] * freqs
    return torch.cat([pos[..., None], torch.sin(phase), torch.cos(phase)], dim=-1)


def local_context_features(
    beta: torch.Tensor,
    observed: torch.Tensor,
    uncertainty: torch.Tensor,
    chr_id: torch.Tensor,
    pos: torch.Tensor,
) -> torch.Tensor:
    batch, n_cpgs = beta.shape
    ids = torch.arange(n_cpgs, device=beta.device)[None].expand(batch, -1)
    available = observed.bool()
    candidates = torch.where(available, ids, torch.full_like(ids, -1))
    previous = torch.cat([torch.full_like(candidates[:, :1], -1), candidates[:, :-1]], dim=1)
    left_idx = torch.cummax(previous, dim=1).values
    candidates = torch.where(available, ids, torch.full_like(ids, n_cpgs))
    following = torch.cat([candidates[:, 1:], torch.full_like(candidates[:, :1], n_cpgs)], dim=1)
    right_idx = torch.flip(torch.cummin(torch.flip(following, dims=(1,)), dim=1).values, dims=(1,))
    safe_left = left_idx.clamp_min(0)
    safe_right = right_idx.clamp_max(n_cpgs - 1)
    left_chr = chr_id.gather(1, safe_left)
    right_chr = chr_id.gather(1, safe_right)
    left_valid = (left_idx >= 0) & (left_chr == chr_id)
    right_valid = (right_idx < n_cpgs) & (right_chr == chr_id)
    left_beta = beta.gather(1, safe_left).masked_fill(~left_valid, 0)
    right_beta = beta.gather(1, safe_right).masked_fill(~right_valid, 0)
    left_pos = pos.gather(1, safe_left)
    right_pos = pos.gather(1, safe_right)
    left_dist = (pos - left_pos).abs().masked_fill(~left_valid, 0)
    right_dist = (right_pos - pos).abs().masked_fill(~right_valid, 0)
    left_uncertainty = uncertainty.gather(1, safe_left).masked_fill(~left_valid, 0)
    right_uncertainty = uncertainty.gather(1, safe_right).masked_fill(~right_valid, 0)
    return torch.stack(
        [
            left_beta,
            right_beta,
            left_dist,
            right_dist,
            (~left_valid).to(dtype=beta.dtype),
            (~right_valid).to(dtype=beta.dtype),
            left_uncertainty,
            right_uncertainty,
        ],
        dim=-1,
    )


def _balanced_arm_patch_groups(count: int, patch_size: int) -> tuple[tuple[int, int], ...]:
    if count == 0:
        return ()
    patch_count = math.ceil(count / patch_size)
    base_size, larger_count = divmod(count, patch_count)
    groups = []
    if larger_count:
        groups.append((larger_count, base_size + 1))
    if patch_count > larger_count:
        groups.append((patch_count - larger_count, base_size))
    return tuple(groups)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, mlp_ratio: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.drop1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * mlp_ratio, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        y = self.norm1(x)
        y, _ = self.attn(y, y, y, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + self.drop1(y)
        return x + self.ff(self.norm2(x))


class FoundationModel(nn.Module):
    def __init__(self, config: FoundationConfig):
        super().__init__()
        config.validate()
        self.config = config
        self.patch_size = int(config.patch_size)
        self.chromosome_cpg_counts = tuple(int(count) for count in config.chromosome_cpg_counts)
        self.arm_cpg_counts = config.ordered_arm_cpg_counts
        self.arm_patch_groups = tuple(
            _balanced_arm_patch_groups(count, self.patch_size) for count in self.arm_cpg_counts
        )
        self.arm_patch_counts = tuple(
            sum(group_count for group_count, _ in groups) for groups in self.arm_patch_groups
        )
        self.chr_emb = nn.Embedding(config.n_chromosomes, config.chr_dim)
        self.cpg_emb = nn.Embedding(config.n_cpgs, config.cpg_id_dim)
        in_dim = 3 + config.chr_dim + config.cpg_id_dim + 1 + 2 * config.pos_bands + 8
        self.token = nn.Sequential(nn.Linear(in_dim, config.d_model), nn.GELU(), nn.Linear(config.d_model, config.d_model))
        self.patch_metadata = nn.Sequential(
            nn.Linear(2, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model),
        )
        self.patch_position = nn.Embedding(sum(self.arm_patch_counts), config.d_model)
        self.blocks = nn.ModuleList(
            TransformerBlock(config.d_model, config.n_heads, config.mlp_ratio, config.dropout)
            for _ in range(config.n_layers)
        )
        self.patch_norm = nn.LayerNorm(config.d_model)
        self.context_fuse = nn.Sequential(
            nn.LayerNorm(2 * config.d_model),
            nn.Linear(2 * config.d_model, config.d_model),
            nn.GELU(),
        )
        self.global_fuse = nn.Sequential(
            nn.LayerNorm(2 * config.d_model),
            nn.Linear(2 * config.d_model, config.d_model),
            nn.GELU(),
        )
        self.head = nn.Linear(config.d_model, 1)
        self.apply(self._init)

    @staticmethod
    def _init(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def _cpg_features(self, batch: int, n_cpgs: int, device: torch.device) -> torch.Tensor:
        if n_cpgs != self.config.n_cpgs:
            raise ValueError(f"input has {n_cpgs} CpGs but model requires {self.config.n_cpgs}")
        cpg_ids = torch.arange(n_cpgs, device=device)
        return self.cpg_emb(cpg_ids)[None].expand(batch, -1, -1)

    def _validate_inputs(
        self,
        beta: torch.Tensor,
        observed: torch.Tensor,
        uncertainty: torch.Tensor,
        chr_id: torch.Tensor,
        pos: torch.Tensor,
    ) -> None:
        if beta.ndim != 2:
            raise ValueError("beta must have shape [B, N]")
        if observed.shape != beta.shape or uncertainty.shape != beta.shape:
            raise ValueError("observed and uncertainty must match beta shape [B, N]")
        if chr_id.shape != beta.shape or pos.shape != beta.shape:
            raise ValueError("chr_id and pos must match beta shape [B, N]")
        if not beta.is_floating_point() or not uncertainty.is_floating_point() or not pos.is_floating_point():
            raise ValueError("beta, uncertainty, and pos must be floating-point tensors")
        if observed.dtype != torch.bool:
            raise ValueError("observed must be a bool tensor")
        if chr_id.is_floating_point() or chr_id.dtype == torch.bool:
            raise ValueError("chr_id must be an integer tensor")
        tensors = (beta, observed, uncertainty, chr_id, pos)
        if any(tensor.device != beta.device for tensor in tensors):
            raise ValueError("all foundation inputs must be on the same device")

    def _token_features(
        self,
        beta: torch.Tensor,
        observed: torch.Tensor,
        uncertainty: torch.Tensor,
        chr_id: torch.Tensor,
        pos: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_inputs(beta, observed, uncertainty, chr_id, pos)
        batch, n_cpgs = beta.shape
        chr_id_long = chr_id.long()
        features = [
            beta[..., None],
            observed.to(dtype=beta.dtype)[..., None],
            uncertainty.to(dtype=beta.dtype)[..., None],
            self.chr_emb(chr_id_long),
            positional_features(pos.to(dtype=beta.dtype), self.config.pos_bands),
            self._cpg_features(batch, n_cpgs, beta.device),
            local_context_features(beta, observed, uncertainty, chr_id, pos),
        ]
        return self.token(torch.cat(features, dim=-1))

    def _patch_context(self, h: torch.Tensor, observed: torch.Tensor, uncertainty: torch.Tensor) -> torch.Tensor:
        batch, n_cpgs, d_model = h.shape
        if n_cpgs != sum(self.arm_cpg_counts):
            raise ValueError("input CpG count does not match arm_cpg_counts")
        pooled_parts = []
        metadata_parts = []
        nonempty_parts = []
        offset = 0
        for count, groups in zip(self.arm_cpg_counts, self.arm_patch_groups, strict=True):
            if not count:
                continue
            end = offset + count
            h_part = h[:, offset:end]
            observed_part = observed[:, offset:end].to(dtype=h.dtype)
            uncertainty_part = uncertainty[:, offset:end].to(dtype=h.dtype)
            group_offset = 0
            for group_count, group_size in groups:
                group_end = group_offset + group_count * group_size
                h_group = h_part[:, group_offset:group_end].reshape(batch, group_count, group_size, d_model)
                observed_group = observed_part[:, group_offset:group_end].reshape(batch, group_count, group_size)
                uncertainty_group = uncertainty_part[:, group_offset:group_end].reshape(
                    batch, group_count, group_size
                )
                denom = observed_group.sum(dim=2, keepdim=True).clamp_min(1)
                pooled_parts.append((h_group * observed_group[..., None]).sum(dim=2) / denom)
                metadata_parts.append(
                    torch.cat(
                        [
                            observed_group.mean(dim=2, keepdim=True),
                            uncertainty_group.sum(dim=2, keepdim=True) / denom,
                        ],
                        dim=-1,
                    )
                )
                nonempty_parts.append(observed_group.bool().any(dim=2))
                group_offset = group_end
            offset = end
        pooled = torch.cat(pooled_parts, dim=1)
        metadata = torch.cat(metadata_parts, dim=1)
        nonempty = torch.cat(nonempty_parts, dim=1)
        patch_ids = torch.arange(pooled.shape[1], device=h.device)
        z = pooled + self.patch_metadata(metadata) + self.patch_position(patch_ids)[None]
        key_padding_mask = ~nonempty
        for block in self.blocks:
            if self.config.activation_checkpointing and self.training:
                z = checkpoint(block, z, key_padding_mask, use_reentrant=False)
            else:
                z = block(z, key_padding_mask)
        z = self.patch_norm(z)
        expanded = []
        patch_offset = 0
        for count, groups in zip(self.arm_cpg_counts, self.arm_patch_groups, strict=True):
            if not count:
                continue
            arm_parts = []
            for group_count, group_size in groups:
                group_end = patch_offset + group_count
                part = z[:, patch_offset:group_end, None, :]
                arm_parts.append(part.expand(batch, group_count, group_size, d_model).reshape(batch, -1, d_model))
                patch_offset = group_end
            expanded.append(torch.cat(arm_parts, dim=1))
        return torch.cat(expanded, dim=1)

    def encode_tokens(
        self,
        beta: torch.Tensor,
        observed: torch.Tensor,
        uncertainty: torch.Tensor,
        chr_id: torch.Tensor,
        pos: torch.Tensor,
    ) -> torch.Tensor:
        h = self._token_features(beta, observed, uncertainty, chr_id, pos)
        context = self._patch_context(h, observed, uncertainty)
        fused = self.context_fuse(torch.cat([h, context], dim=-1))
        observed_f = observed.to(dtype=fused.dtype)
        pooled = (fused * observed_f[..., None]).sum(dim=1) / observed_f.sum(dim=1, keepdim=True).clamp_min(1)
        return fused + self.global_fuse(torch.cat([fused, pooled[:, None, :].expand_as(fused)], dim=-1))

    def embed(
        self,
        beta: torch.Tensor,
        observed: torch.Tensor,
        uncertainty: torch.Tensor,
        chr_id: torch.Tensor,
        pos: torch.Tensor,
    ) -> torch.Tensor:
        tokens = self.encode_tokens(beta, observed, uncertainty, chr_id, pos)
        return self.embedding_from_tokens(tokens, observed)

    @staticmethod
    def embedding_from_tokens(tokens: torch.Tensor, observed: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3 or observed.shape != tokens.shape[:2]:
            raise ValueError("foundation tokens and observed mask must have shapes [B, N, D] and [B, N]")
        observed_f = observed.to(dtype=tokens.dtype)
        return (tokens * observed_f[..., None]).sum(dim=1) / observed_f.sum(dim=1, keepdim=True).clamp_min(1)

    def forward(
        self,
        beta: torch.Tensor,
        observed: torch.Tensor,
        uncertainty: torch.Tensor,
        chr_id: torch.Tensor,
        pos: torch.Tensor,
    ) -> torch.Tensor:
        return torch.sigmoid(self.predict_logits(beta, observed, uncertainty, chr_id, pos))

    def predict_logits(
        self,
        beta: torch.Tensor,
        observed: torch.Tensor,
        uncertainty: torch.Tensor,
        chr_id: torch.Tensor,
        pos: torch.Tensor,
    ) -> torch.Tensor:
        tokens = self.encode_tokens(beta, observed, uncertainty, chr_id, pos)
        return self.head(tokens).squeeze(-1)

def validate_chromosome_layout(
    config: FoundationConfig,
    chr_id: torch.Tensor,
    arm_id: torch.Tensor | None,
) -> None:
    if chr_id.ndim != 1 or int(chr_id.numel()) != config.n_cpgs:
        raise ValueError("CpG chromosome layout must be a one-dimensional n_cpgs vector")
    if chr_id.numel() > 1 and bool((chr_id[1:] < chr_id[:-1]).any().item()):
        raise ValueError("CpG chromosome layout must be genomically grouped by chromosome")
    counts = torch.bincount(chr_id.long(), minlength=config.n_chromosomes)
    if counts.numel() > config.n_chromosomes or counts.tolist() != config.chromosome_cpg_counts:
        raise ValueError("CpG chromosome layout does not match chromosome_cpg_counts")
    if arm_id is None:
        raise ValueError("CpG manifest requires chrom, start, and end arrays for architecture-5 arm validation")
    if arm_id.ndim != 1 or int(arm_id.numel()) != config.n_cpgs:
        raise ValueError("CpG arm layout must be a one-dimensional n_cpgs vector")
    expected_arm_id = torch.repeat_interleave(
        torch.arange(len(config.ordered_arm_cpg_counts), device=arm_id.device),
        torch.as_tensor(config.ordered_arm_cpg_counts, device=arm_id.device),
    )
    if not torch.equal(arm_id.long(), expected_arm_id):
        raise ValueError("CpG arm layout does not match arm_cpg_counts")


def load_foundation(path: str | Path, device: torch.device | str = "cpu") -> FoundationModel:
    root = Path(path)
    config_path = root / "config.json"
    weights_path = root / "model.safetensors"
    if not config_path.exists():
        raise FileNotFoundError(f"foundation config missing: {config_path}")
    if not weights_path.exists():
        raise FileNotFoundError(f"foundation weights missing: {weights_path}")
    config = FoundationConfig.from_dict(json.loads(config_path.read_text()))
    model = FoundationModel(config)
    model.load_state_dict(load_file(str(weights_path), device=str(device)))
    return model.to(device).eval()
