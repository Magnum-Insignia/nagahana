"""Proper scoring rules: the shared term S and the forecast-verification metrics (ARCH §6.2, #26.6).

A scoring rule is *proper* when the forecaster minimises its expected score by reporting its true
belief. That is exactly what calibration needs: honest probabilities are rewarded, and inflated or
deflated ones are not. Meteorology's forecast-verification discipline builds on these.

    Brier:      BS(p, y)  = (p − y)²                               y ∈ {0, 1}
    Log score:  LS(p, y)  = −[ y·log p + (1 − y)·log(1 − p) ]
    CRPS:       CRPS(F, y) = E|X − y| − ½·E|X − X'|,  X, X' ~ F     (ensemble estimator below)

The CRPS generalises absolute error to a whole predictive distribution. For the N imagined samples
of a quantity (e.g. time to first infiltration), the empirical ensemble estimator is

    CRPS_N = (1/N) Σ_i |x_i − y|  −  (1/(2N²)) Σ_i Σ_j |x_i − x_j|

All functions return per-item values; the caller reduces them. Log score clamps p to [ε, 1 − ε], with
ε required (a modelling choice, so no default).
"""

from __future__ import annotations

import torch


def brier(p: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Per-item Brier score (p − y)²."""
    return (p - y.to(p.dtype)) ** 2


def log_score(p: torch.Tensor, y: torch.Tensor, *, eps: float) -> torch.Tensor:
    """Per-item negative log-likelihood of a Bernoulli forecast (lower is better)."""
    if not 0 < eps < 0.5:
        raise ValueError("eps must be in (0, 0.5)")
    q = p.clamp(eps, 1.0 - eps)
    yf = y.to(p.dtype)
    return -(yf * torch.log(q) + (1.0 - yf) * torch.log(1.0 - q))


def crps_ensemble(samples: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Empirical CRPS of an ensemble. samples: [..., N]; y: [...]. Returns [...]."""
    n = samples.shape[-1]
    term1 = (samples - y[..., None]).abs().mean(dim=-1)
    term2 = (samples[..., :, None] - samples[..., None, :]).abs().sum(dim=(-1, -2)) / (2.0 * n * n)
    return term1 - term2
