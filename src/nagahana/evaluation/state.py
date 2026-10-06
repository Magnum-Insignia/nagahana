"""Forecasts of the network state itself: masked errors per horizon and feature, energy score, Gaussian CRPS.

The world model's P(S_t+k | S_t) is scored directly on the observable window features (counts of
flows, packets and bytes, distinct destination hosts and ports, shares of SYN, RST and failed
connections), standardised on the training split (thesis section on next-state forecasts). Only cells
where the observed value exists are scored (`StateForecastPredictions.mask`; absence is not zero, D-41).

Errors at horizon h over the scored cells (i, d) with unit weights w_i:

    MAE_h  = sum w_i m_ihd |xhat_ihd - x_ihd| / sum w_i m_ihd,
    MSE_h  = sum w_i m_ihd (xhat_ihd - x_ihd)^2 / sum w_i m_ihd,   RMSE_h = sqrt(MSE_h),

and the same per feature. MAE and RMSE are reported together because the squared error weighs large
errors more. The skill against persistence (the window at the origin repeated) is
MSESS = 1 - MSE / MSE_pers (thesis equation for MSESS), computed on the cells scored by both forecasts.

Energy score. For predictive samples X_1..X_S of the vector of scored features and the outcome y,

    ES = (1/S) sum_s ||X_s - y|| - (1 / (2 S^2)) sum_{s,s'} ||X_s - X_s'||

(the negative of Gneiting and Raftery's energy score with beta = 1, JASA 102:359-378, 2007), strictly
proper relative to distributions with a finite first moment; it reduces to the CRPS for one feature.
The fair version divides the second sum by 2 S (S - 1) instead (Ferro 2014). Norms run over the
features scored for that origin and horizon.

Gaussian CRPS. When a predictive variance sigma^2 is supplied, the CRPS of N(mu, sigma^2) at y is

    CRPS = sigma [ z (2 Phi(z) - 1) + 2 phi(z) - 1 / sqrt(pi) ],   z = (y - mu) / sigma

(Gneiting, Raftery, Westveld and Goldman, Monthly Weather Review 133:1098-1118, 2005), and |y - mu| when
sigma = 0. The coefficient of determination is not reported: on a non-stationary series it reflects
the variance of the evaluation window as much as the quality of the forecast.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy import stats

from nagahana.core.errors import InvariantViolation
from nagahana.evaluation._arrays import safe_ratio, weight_matrix
from nagahana.evaluation.predictions import StateForecastPredictions


def persistence_forecast(pred: StateForecastPredictions) -> tuple[np.ndarray, np.ndarray]:
    """(forecast [n, H, D], mask [n, H, D]) of persistence: the origin's features at every horizon."""
    if pred.current is None or pred.current_mask is None:
        raise InvariantViolation("persistence needs StateForecastPredictions.current (the window at the origin)")
    h = pred.predicted.shape[1]
    fc = np.repeat(pred.current[:, None, :], h, axis=1)
    mask = pred.mask & np.repeat(pred.current_mask[:, None, :], h, axis=1)
    return fc, mask


def errors_by_horizon(predicted: np.ndarray, observed: np.ndarray, mask: np.ndarray, weights: Any = None
                      ) -> dict[str, Any]:
    """MAE, MSE, RMSE and scored-cell weight per horizon: arrays [B, H] ([H] unbatched)."""
    n = predicted.shape[0]
    w, batched = weight_matrix(weights, n)
    m = mask.astype(np.float64)
    err = np.where(mask, predicted - observed, 0.0)
    cells = np.einsum("bn,nh->bh", w, m.sum(axis=2))
    abs_sum = np.einsum("bn,nh->bh", w, np.abs(err).sum(axis=2))
    sq_sum = np.einsum("bn,nh->bh", w, (err ** 2).sum(axis=2))
    mse = safe_ratio(sq_sum, cells)
    out = {"mae": safe_ratio(abs_sum, cells), "mse": mse, "rmse": np.sqrt(mse), "cells": cells}
    return out if batched else {k: v[0] for k, v in out.items()}


def errors_by_feature(predicted: np.ndarray, observed: np.ndarray, mask: np.ndarray, weights: Any = None
                      ) -> dict[str, Any]:
    """MAE, RMSE and scored-cell weight per (horizon, feature): arrays [B, H, D] ([H, D] unbatched)."""
    w, batched = weight_matrix(weights, predicted.shape[0])
    m = mask.astype(np.float64)
    err = np.where(mask, predicted - observed, 0.0)
    cells = np.einsum("bn,nhd->bhd", w, m)
    mae = safe_ratio(np.einsum("bn,nhd->bhd", w, np.abs(err)), cells)
    rmse = np.sqrt(safe_ratio(np.einsum("bn,nhd->bhd", w, err ** 2), cells))
    out = {"mae": mae, "rmse": rmse, "cells": cells}
    return out if batched else {k: v[0] for k, v in out.items()}


def msess(model_mse: Any, reference_mse: Any) -> Any:
    """Mean-squared-error skill score 1 - MSE / MSE_ref (NaN where the reference error is 0)."""
    out = 1.0 - safe_ratio(model_mse, reference_mse)
    return float(out) if np.ndim(out) == 0 else out


def energy_score(pred: StateForecastPredictions, *, fair: bool = False, max_elements: int = 20_000_000) -> np.ndarray:
    """Per-origin energy score [n, H] over the scored features (NaN where none is scored).

    Origins are processed in chunks so that the pairwise sample differences [c, S, S, H, D] hold at most
    `max_elements` values.
    """
    if pred.samples is None:
        raise InvariantViolation("the energy score needs StateForecastPredictions.samples")
    smp, obs, mask = pred.samples, pred.observed, pred.mask
    n, s, h, d = smp.shape
    if fair and s < 2:
        raise InvariantViolation("the fair energy score needs at least two samples")
    out = np.full((n, h), np.nan)
    chunk = max(1, max_elements // max(1, s * s * h * d))
    for a in range(0, n, chunk):
        z = slice(a, min(a + chunk, n))
        m = mask[z].astype(np.float64)                                  # [c, H, D]
        x = smp[z]                                                      # [c, S, H, D]
        y = obs[z]
        to_obs = np.sqrt((((x - y[:, None]) ** 2) * m[:, None]).sum(axis=3))   # [c, S, H]
        diff = x[:, :, None] - x[:, None, :]                            # [c, S, S, H, D]
        pair = np.sqrt(((diff ** 2) * m[:, None, None]).sum(axis=4))    # [c, S, S, H]
        spread = pair.sum(axis=(1, 2)) / (2.0 * s * (s - 1) if fair else 2.0 * s * s)
        es = to_obs.mean(axis=1) - spread
        out[z] = np.where(m.sum(axis=2) > 0, es, np.nan)
    return out


def gaussian_crps(mu: Any, variance: Any, y: Any) -> np.ndarray:
    """Closed-form CRPS of N(mu, variance) at y (elementwise); |y - mu| where the variance is 0."""
    m = np.asarray(mu, dtype=np.float64)
    v = np.asarray(variance, dtype=np.float64)
    o = np.asarray(y, dtype=np.float64)
    if np.any(v < 0):
        raise InvariantViolation("variance must be non-negative")
    sigma = np.sqrt(v)
    z = safe_ratio(o - m, sigma)
    val = sigma * (z * (2.0 * stats.norm.cdf(z) - 1.0) + 2.0 * stats.norm.pdf(z) - 1.0 / math.sqrt(math.pi))
    return np.where(sigma > 0, val, np.abs(o - m))


def gaussian_crps_by_horizon(pred: StateForecastPredictions, weights: Any = None) -> Any:
    """Weighted mean Gaussian CRPS over the scored cells, per horizon [B, H] ([H] unbatched)."""
    if pred.variance is None:
        raise InvariantViolation("the Gaussian CRPS needs StateForecastPredictions.variance")
    w, batched = weight_matrix(weights, pred.predicted.shape[0])
    crps = np.where(pred.mask, gaussian_crps(pred.predicted, pred.variance, pred.observed), 0.0)
    cells = np.einsum("bn,nh->bh", w, pred.mask.astype(np.float64).sum(axis=2))
    val = safe_ratio(np.einsum("bn,nh->bh", w, crps.sum(axis=2)), cells)
    return val if batched else val[0]


def energy_by_horizon(pred: StateForecastPredictions, weights: Any = None, *, fair: bool = False) -> Any:
    """Weighted mean energy score per horizon over origins with scored features [B, H] ([H] unbatched)."""
    es = energy_score(pred, fair=fair)                                  # [n, H]
    w, batched = weight_matrix(weights, es.shape[0])
    ok = np.isfinite(es)
    val = safe_ratio(np.einsum("bn,nh->bh", w, np.where(ok, es, 0.0)), np.einsum("bn,nh->bh", w, ok.astype(np.float64)))
    return val if batched else val[0]
