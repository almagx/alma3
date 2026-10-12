"""The selected two-layer feature attention mechanism."""

import torch
from torch import nn

from .direct_encoders import high_precision


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, mlp_ratio, dropout):
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


class _FP32LayerNorm(nn.LayerNorm):
    def forward(self, value):
        return super().forward(high_precision(value))


class SignatureRefinement(nn.Module):
    def __init__(self, mode):
        super().__init__()
        if mode != "feature_attention2_v1":
            raise ValueError("unsupported feature refinement")
        self.mode, self.scale = mode, 0.1
        self.token_weight = nn.Parameter(torch.empty(256, 64))
        self.token_bias = nn.Parameter(torch.empty(256, 64))
        nn.init.xavier_uniform_(self.token_weight)
        nn.init.xavier_uniform_(self.token_bias)
        self.blocks = nn.ModuleList(TransformerBlock(64, 4, 4, 0.0) for _ in range(2))
        for block in self.blocks:
            block.norm1, block.norm2 = _FP32LayerNorm(64), _FP32LayerNorm(64)
            nn.init.xavier_uniform_(block.attn.in_proj_weight)
            nn.init.zeros_(block.attn.in_proj_bias)
        self.norm = _FP32LayerNorm(64)
        self.readout = nn.Linear(64, 1)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, value):
        tokens = value.unsqueeze(-1) * self.token_weight + self.token_bias
        for block in self.blocks:
            tokens = block(tokens)
        return value + self.scale * self.readout(self.norm(tokens)).squeeze(-1)
