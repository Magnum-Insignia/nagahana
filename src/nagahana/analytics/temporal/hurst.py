"""Long-range dependence: Hurst exponent by detrended fluctuation analysis and by aggregated variance.

Detrended fluctuation analysis (Peng, Buldyrev, Havlin, Simons, Stanley and Goldberger, Phys. Rev. E
49:1685-1689, 1994)
--------------------------------------------------------------------------------------------------
Profile Y(i) = sum_{k<=i} (x_k - xbar). For a box size s the profile is cut into floor(N / s) boxes
from the start and as many from the end (so no data are left out), a polynomial of order q is fitted
in each box (DFA-q), and F(s)^2 is the mean over the 2 floor(N / s) boxes of the mean squared residual.
F(s) ~ s^alpha; alpha is the least-squares slope of log F against log s over geometric box sizes. For
fractional Gaussian noise alpha = H, for its cumulative sum (fractional Brownian motion) alpha = H + 1;
white noise gives 0.5. Box residuals come from sufficient statistics: with Q the orthonormal basis of
the polynomial design on one box, RSS = sum Y^2 - ||Q' Y||^2, O(N (q + 1)) per box size.

Aggregated variance (AS-36 uses this estimator inside the information lens)
---------------------------------------------------------------------------
X^(m)_k the mean of block k of m consecutive values; Var(X^(m)) ~ m^(2H - 2), so H = 1 + slope / 2
(Taqqu, Teverovsky and Willinger, Fractals 3(4):785-798, 1995).

Fractional Gaussian noise (Davies and Harte, Biometrika 74(1):95-101, 1987)
-------------------------------------------------------------------------
Exact simulation by circulant embedding of gamma(k) = (|k + 1|^{2H} - 2 |k|^{2H} + |k - 1|^{2H}) / 2:
the eigenvalues of the 2n circulant are non-negative for 0 < H < 1, and a complex Gaussian vector
scaled by their square roots and transformed by the FFT gives a series with exactly that covariance.
Used as the known-answer input of the tests.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Fluctuation:
    """DFA result: exponent, its standard error from the log-log fit, the scales and F(s)."""

    alpha: float
    alpha_se: float
    scales: np.ndarray
    fluctuation: np.ndarray
    order: int


def _fit_slope(lx: np.ndarray, ly: np.ndarray) -> tuple[float, float]:
    """(slope, standard error) of the least-squares line through (lx, ly)."""
    if lx.size < 3:
        return float("nan"), float("nan")
    x = np.column_stack([np.ones_like(lx), lx])
    beta, *_ = np.linalg.lstsq(x, ly, rcond=None)
    resid = ly - x @ beta
    s2 = float(resid @ resid) / (lx.size - 2)
    cov = s2 * np.linalg.inv(x.T @ x)
    return float(beta[1]), float(np.sqrt(cov[1, 1]))


def dfa(x: np.ndarray, *, order: int = 1, min_scale: int = 8, max_scale_share: float = 0.25, scales: int = 24) -> Fluctuation:
    """Detrended fluctuation analysis (module docstring)."""
    a = np.asarray(x, dtype=np.float64).reshape(-1)
    if not np.isfinite(a).all():
        raise ValueError("the series must be finite")
    n = a.size
    max_scale = int(np.floor(max_scale_share * n))
    if max_scale < max(min_scale, order + 2) or scales < 3:
        return Fluctuation(float("nan"), float("nan"), np.zeros(0, dtype=np.int64), np.zeros(0), order)
    s_all = np.unique(np.round(np.geomspace(max(min_scale, order + 2), max_scale, scales)).astype(np.int64))
    prof = np.cumsum(a - a.mean())
    f = np.empty(s_all.size)
    for i, s in enumerate(s_all.tolist()):
        nb = n // s
        q, _ = np.linalg.qr(np.vander(np.arange(s, dtype=np.float64) / s, order + 1, increasing=True))   # [s, q + 1]
        boxes = np.concatenate([prof[: nb * s].reshape(nb, s), prof[n - nb * s:].reshape(nb, s)])      # [2 nb, s]
        proj = boxes @ q
        rss = (boxes * boxes).sum(axis=1) - (proj * proj).sum(axis=1)
        f[i] = np.sqrt(max(float(rss.mean()) / s, 0.0))
    ok = f > 0
    slope, se = _fit_slope(np.log(s_all[ok].astype(np.float64)), np.log(f[ok]))
    return Fluctuation(alpha=slope, alpha_se=se, scales=s_all, fluctuation=f, order=order)


def aggregated_variance(x: np.ndarray, *, min_block: int = 2, max_block_share: float = 0.1, blocks: int = 20) -> tuple[float, float]:
    """(H, standard error) by the aggregated-variance method (module docstring)."""
    a = np.asarray(x, dtype=np.float64).reshape(-1)
    n = a.size
    max_block = int(np.floor(max_block_share * n))
    if max_block < min_block:
        return float("nan"), float("nan")
    ms = np.unique(np.round(np.geomspace(min_block, max_block, blocks)).astype(np.int64))
    v = []
    for m in ms.tolist():
        k = n // m
        means = a[: k * m].reshape(k, m).mean(axis=1)
        v.append(means.var(ddof=1) if k > 1 else np.nan)
    va = np.asarray(v)
    ok = np.isfinite(va) & (va > 0)
    slope, se = _fit_slope(np.log(ms[ok].astype(np.float64)), np.log(va[ok]))
    return 1.0 + slope / 2.0, se / 2.0


def simulate_fgn(n: int, hurst: float, *, seed: int = 0) -> np.ndarray:
    """Exact fractional Gaussian noise of length n with unit variance (Davies-Harte; module docstring)."""
    if not 0.0 < hurst < 1.0:
        raise ValueError("hurst must be in (0, 1)")
    k = np.arange(n + 1, dtype=np.float64)
    gamma = 0.5 * (np.abs(k + 1) ** (2 * hurst) - 2 * k ** (2 * hurst) + np.abs(k - 1) ** (2 * hurst))
    row = np.concatenate([gamma, gamma[-2:0:-1]])                         # first row of the 2n circulant
    lam = np.fft.fft(row).real
    if (lam < -1e-10 * np.abs(lam).max()).any():
        raise ValueError("circulant embedding failed (negative eigenvalue)")
    lam = np.clip(lam, 0.0, None)
    m = row.size
    rng = np.random.default_rng(seed)
    w = rng.standard_normal(m) + 1j * rng.standard_normal(m)
    z = np.fft.fft(np.sqrt(lam / m) * w)
    return z.real[:n]


__all__ = ["Fluctuation", "aggregated_variance", "dfa", "simulate_fgn"]
