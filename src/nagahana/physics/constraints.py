"""Hard limits C: outputs that are physical by construction (D-18).

Two ways keep a model inside the physical boundary:
- soft: add lambda_phys * Phi_phys to the loss (`term.py`), counting a residual only where its fields are
  observed;
- hard: parameterise the output so it cannot leave the allowed set C, whatever the network computes.
  Counts are non-negative, ratios lie in [0, 1], a maximum is at least its mean.

Hard limits never need a weight and never trade off against accuracy. They are used wherever a constraint is
exact and cheap to parameterise; the soft term covers constraints that are checkable only where data are
observed (conservation, queueing, propagation).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def nonnegative(raw: torch.Tensor) -> torch.Tensor:
    """Map any real output to (0, inf) smoothly (softplus). For counts, bytes and durations."""
    return F.softplus(raw)


def unit_interval(raw: torch.Tensor) -> torch.Tensor:
    """Map any real output to (0, 1) (sigmoid). For ratios and probabilities."""
    return torch.sigmoid(raw)


def ordered_pair(low_raw: torch.Tensor, gap_raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (low, high) with high >= low >= 0 by construction.

    Example: (iat_mean, iat_max). The mean of the gaps cannot exceed their maximum, so the model predicts
    the mean and a non-negative gap to the maximum.
    """
    low = nonnegative(low_raw)
    return low, low + nonnegative(gap_raw)
