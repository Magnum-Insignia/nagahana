"""Numeric value encodings for continuous and count fields (AS-31).

Network magnitudes span many orders (a flow carries 40 bytes or 4 GB; a gap lasts 10 µs or a day).
Two steps make them learnable:

1. **Signed log compression**  x̃ = sign(x)·log(1 + |x|). Monotone, invertible
   (`signed_expm1`), linear near 0 and logarithmic in the tails. The same transform is the Decoder's
   likelihood space, so reconstruction errors are relative, not absolute.
2. **Periodic embedding**  e(x̃) = W·[sin(2π·w·x̃), cos(2π·w·x̃)] with learned frequencies w
   (Gorishniy, Rubachev & Babenko, "On Embeddings for Numerical Features in Tabular Deep Learning",
   NeurIPS 2022, arXiv:2203.05556). Their experiments show periodic embeddings let MLPs and
   transformers resolve fine differences in a scalar that a single linear projection blurs.

Frequencies are initialised from 𝒩(0, σ²); σ is a scale hyperparameter (their main knob).
"""

from __future__ import annotations

import math

import torch
from torch import nn


def signed_log1p(x: torch.Tensor) -> torch.Tensor:
    """sign(x)·log(1 + |x|)."""
    return torch.sign(x) * torch.log1p(torch.abs(x))


def signed_expm1(y: torch.Tensor) -> torch.Tensor:
    """Inverse of `signed_log1p`: sign(y)·(exp(|y|) − 1)."""
    return torch.sign(y) * torch.expm1(torch.abs(y))


class PeriodicEmbedding(nn.Module):
    """Embed scalars [...] → [..., out_dim] with learned periodic features.

    Parameters
    ----------
    n_features: how many independent scalars are embedded (each gets its own frequencies), e.g. the
        number of continuous columns. Input shape is then [..., n_features].
    n_frequencies: k frequencies per scalar (2k periodic features).
    out_dim: output width per scalar.
    sigma: initial frequency scale.
    """

    def __init__(self, n_features: int, n_frequencies: int, out_dim: int, *, sigma: float = 1.0) -> None:
        super().__init__()
        self.freq = nn.Parameter(torch.randn(n_features, n_frequencies) * sigma)
        # Per-feature linear map from the 2k periodic features to out_dim (einsum below).
        self.weight = nn.Parameter(torch.randn(n_features, 2 * n_frequencies, out_dim) / math.sqrt(2 * n_frequencies))
        self.bias = nn.Parameter(torch.zeros(n_features, out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., F] → angles [..., F, k] → periodic [..., F, 2k] → [..., F, out_dim]
        ang = 2 * math.pi * x.unsqueeze(-1) * self.freq
        per = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)
        return torch.einsum("...fk,fko->...fo", per, self.weight) + self.bias
