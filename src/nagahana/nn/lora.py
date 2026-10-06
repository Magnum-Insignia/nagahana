"""Low-rank adapters for site calibration (assumption AS-26, standing in for held D-13).

    W' x = W x + (α / r) · B A x,     A ∈ ℝ^{r×d_in},  B ∈ ℝ^{d_out×r},  B initialised to 0

Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models", ICLR 2022 (arXiv:2106.09685). With
B = 0 at start the adapted model is exactly the base model, so attaching adapters changes nothing
until they are trained. The base weight is frozen; only A and B train, so a site's calibration is a
small, separable, auditable delta that can be removed (`enabled = False`) at any time.

Applying adapters changes the model, so it happens only on a human command (D-21); the gate is in
the training code (`training/stage6`), not here.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class LoRALinear(nn.Module):
    """Wrap an `nn.Linear` with a low-rank, initially-zero update."""

    def __init__(self, base: nn.Linear, *, rank: int, alpha: float) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive")
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.scale = alpha / rank
        self.a = nn.Parameter(torch.empty(rank, base.in_features))
        self.b = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))
        self.enabled = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        if self.enabled:
            y = y + self.scale * ((x @ self.a.t()) @ self.b.t())
        return y


def attach_lora(module: nn.Module, *, names: tuple[str, ...], rank: int, alpha: float) -> list[str]:
    """Replace every `nn.Linear` child whose attribute name is in `names` (e.g. "q_proj", "v_proj")
    by a `LoRALinear`. Returns the dotted paths replaced."""
    replaced: list[str] = []
    for parent_name, parent in list(module.named_modules()):
        for child_name, child in list(parent.named_children()):
            if child_name in names and isinstance(child, nn.Linear):
                setattr(parent, child_name, LoRALinear(child, rank=rank, alpha=alpha))
                replaced.append(f"{parent_name}.{child_name}" if parent_name else child_name)
    return replaced


def lora_parameters(module: nn.Module) -> list[nn.Parameter]:
    """The trainable adapter parameters (A and B of every LoRALinear)."""
    return [p for m in module.modules() if isinstance(m, LoRALinear) for p in (m.a, m.b)]
