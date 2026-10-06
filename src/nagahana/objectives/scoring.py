"""Proper scoring rules: the shared term S of the loss template and of the Forecaster's process reward.

A scoring rule is proper when the forecaster minimises its expected score by reporting its true belief, and
strictly proper when that minimum is unique (Gneiting and Raftery, "Strictly Proper Scoring Rules, Prediction,
and Estimation", JASA 102(477), 2007). That is what calibration needs: honest probabilities are rewarded, and
inflated or deflated ones are not.

    Brier (binary):        BS(p, y)  = (p - y)^2                                   y in {0, 1}
    Log score (binary):    LS(p, y)  = -[ y log p + (1 - y) log(1 - p) ]
    Brier (categorical):   BS(p, c)  = sum_k (p_k - [k = c])^2                     (Brier, Monthly Weather Review 78(1), 1950)
    Log score (categorical): LS(p, c) = -log p_c
    CRPS:                  CRPS(F, y) = E|X - y| - 1/2 E|X - X'|,  X, X' ~ F      (ensemble estimator below)

The CRPS generalises absolute error to a whole predictive distribution. For N samples of a quantity (e.g. the
time to first infiltration), the empirical ensemble estimator is

    CRPS_N = (1 / N) sum_i |x_i - y|  -  (1 / (2 N^2)) sum_i sum_j |x_i - x_j|

All functions return per-item values; the caller reduces them. The log scores clamp p to [eps, 1 - eps], with
eps required (a modelling choice, so no default).
"""

from __future__ import annotations

import torch


def brier(p: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Per-item Brier score (p - y)^2."""
    return (p - y.to(p.dtype)) ** 2


def log_score(p: torch.Tensor, y: torch.Tensor, *, eps: float) -> torch.Tensor:
    """Per-item negative log-likelihood of a Bernoulli forecast (lower is better)."""
    if not 0 < eps < 0.5:
        raise ValueError("eps must be in (0, 0.5)")
    q = p.clamp(eps, 1.0 - eps)
    yf = y.to(p.dtype)
    return -(yf * torch.log(q) + (1.0 - yf) * torch.log(1.0 - q))


def _check_categorical(probs: torch.Tensor, outcome: torch.Tensor) -> None:
    if probs.shape[:-1] != outcome.shape:
        raise ValueError(f"probs {tuple(probs.shape)} must be [..., K] for outcomes {tuple(outcome.shape)}")
    if outcome.numel() and (int(outcome.min()) < 0 or int(outcome.max()) >= probs.shape[-1]):
        raise ValueError("every outcome must be a class index in [0, K)")


def brier_categorical(probs: torch.Tensor, outcome: torch.Tensor) -> torch.Tensor:
    """Per-item multi-class Brier score sum_k (p_k - [k = c])^2: probs [..., K], outcome long [...] in [0, K)."""
    _check_categorical(probs, outcome)
    onehot = torch.nn.functional.one_hot(outcome.long(), probs.shape[-1]).to(probs.dtype)
    return ((probs - onehot) ** 2).sum(-1)


def log_score_categorical(probs: torch.Tensor, outcome: torch.Tensor, *, eps: float) -> torch.Tensor:
    """Per-item -log p_c of a categorical forecast: probs [..., K], outcome long [...] in [0, K)."""
    if not 0 < eps < 0.5:
        raise ValueError("eps must be in (0, 0.5)")
    _check_categorical(probs, outcome)
    p_c = torch.gather(probs, -1, outcome.long().unsqueeze(-1)).squeeze(-1)
    return -torch.log(p_c.clamp(eps, 1.0))


def crps_ensemble(samples: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Empirical CRPS of an ensemble. samples: [..., N]; y: [...]. Returns [...]."""
    n = samples.shape[-1]
    term1 = (samples - y[..., None]).abs().mean(dim=-1)
    term2 = (samples[..., :, None] - samples[..., None, :]).abs().sum(dim=(-1, -2)) / (2.0 * n * n)
    return term1 - term2
