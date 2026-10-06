"""SwiGLU feed-forward (AS-32).

    SwiGLU(x) = W_out ( SiLU(W_gate x) ⊙ (W_up x) )

Shazeer, "GLU Variants Improve Transformer", 2020 (arXiv:2002.05202). With hidden width ≈ 8/3·d the
parameter count (3·d·hidden) matches a 4·d GELU MLP (2·d·4d) while the gated form trains better in
Shazeer's comparisons. `swiglu_hidden` rounds 8/3·d up to a multiple of 256 for efficient kernels
(1024 → 2816). No biases: none of the blocks use them (AS-32).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def swiglu_hidden(dim: int, multiple: int = 256) -> int:
    """Hidden width ⌈(8/3)·dim / multiple⌉·multiple (e.g. 1024 → 2816)."""
    h = int(8 * dim / 3)
    return multiple * ((h + multiple - 1) // multiple)


class SwiGLU(nn.Module):
    """Gated feed-forward block. `hidden` defaults to `swiglu_hidden(dim)`."""

    def __init__(self, dim: int, hidden: int | None = None, out_dim: int | None = None) -> None:
        super().__init__()
        hidden = hidden or swiglu_hidden(dim)
        self.gate = nn.Linear(dim, hidden, bias=False)
        self.up = nn.Linear(dim, hidden, bias=False)
        self.down = nn.Linear(hidden, out_dim or dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))
