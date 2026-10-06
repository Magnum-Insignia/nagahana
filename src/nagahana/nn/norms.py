"""RMSNorm: the normalisation of every pre-norm block (AS-32).

    RMSNorm(x) = g ⊙ x / sqrt(mean(x²) + ε)

Zhang & Sennrich, "Root Mean Square Layer Normalization", NeurIPS 2019 (arXiv:1910.07467). It drops
LayerNorm's mean-centring, which costs nothing in quality in large transformers and keeps the
statistic computed in float32 even under bf16 autocast (the cast below), so tiny residual streams do
not underflow.
"""

from __future__ import annotations

import torch
from torch import nn


class RMSNorm(nn.Module):
    """Root-mean-square normalisation over the last dimension, with a learned gain."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Statistic in float32 whatever the input dtype; result cast back to the input dtype.
        x32 = x.float()
        y = x32 * torch.rsqrt(x32.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (y * self.weight.float()).to(x.dtype)
