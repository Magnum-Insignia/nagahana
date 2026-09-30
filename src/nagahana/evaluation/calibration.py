"""Calibration: do forecast probabilities match how often things happen? (Verifier; RLCD; ARCH §6.3)

Reliability analysis with equal-width bins over predicted probability:

    ECE = Σ_b (n_b / n) · | acc_b − conf_b |

conf_b is the mean predicted probability in bin b, and acc_b is the observed frequency of the event
in bin b. The bin count is required. It changes the value of ECE, so it is a reported choice, not a
hidden default.

Responded-to cases (the SOC acted, so the attack did not complete) are excluded by the caller, as in
`roles/verifier.py` [Q-38].
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ReliabilityBin:
    """One reliability-diagram bin."""

    lower: float
    upper: float
    count: int
    mean_confidence: float
    frequency: float


def reliability(p: torch.Tensor, y: torch.Tensor, *, bins: int) -> list[ReliabilityBin]:
    """Equal-width reliability bins over [0, 1] (the last bin includes 1.0)."""
    if bins < 1:
        raise ValueError("bins must be >= 1")
    p, yf = p.flatten().double(), y.flatten().double()
    edges = torch.linspace(0, 1, bins + 1, dtype=torch.float64)
    idx = torch.clamp(torch.bucketize(p, edges, right=True) - 1, 0, bins - 1)
    out: list[ReliabilityBin] = []
    for b in range(bins):
        m = idx == b
        n = int(m.sum())
        out.append(ReliabilityBin(float(edges[b]), float(edges[b + 1]), n,
                                  float(p[m].mean()) if n else float("nan"),
                                  float(yf[m].mean()) if n else float("nan")))
    return out


def ece(p: torch.Tensor, y: torch.Tensor, *, bins: int) -> float:
    """Expected calibration error with equal-width bins."""
    rel = reliability(p, y, bins=bins)
    n = sum(r.count for r in rel)
    if n == 0:
        return float("nan")
    return sum(r.count / n * abs(r.frequency - r.mean_confidence) for r in rel if r.count)
