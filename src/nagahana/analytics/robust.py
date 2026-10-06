"""Robust location, scale, shape and tail statistics of one sample.

Network telemetry is heavy-tailed (bytes, durations and inter-arrival times span many orders of
magnitude), so moment statistics are reported only beside order statistics, and the tail itself is
measured.

Location and scale
------------------
    median, quantiles      `numpy.quantile` with the "linear" rule (Hyndman and Fan, The American
                           Statistician 50(4), 1996: their definition 7)
    MAD                    median(|x - median(x)|) * c, c = 1 / Phi^-1(3/4) = 1.482602218505602, so that
                           MAD estimates sigma for normal data
    IQR                    Q(0.75) - Q(0.25)
    trimmed mean           mean of the central (1 - 2 * proportion) share (scipy.stats.trim_mean)

Shape (quantile based, bounded influence)
-----------------------------------------
    Bowley skewness        (Q3 + Q1 - 2 * Q2) / (Q3 - Q1), in [-1, 1]; 0 for symmetric laws
                           (Bowley, "Elements of Statistics", 1901)
    Moors kurtosis         ((E7 - E5) + (E3 - E1)) / (E6 - E2), E_i the i/8 quantile; 1.2331 for the
                           normal law (Moors, The Statistician 37(1):25-32, 1988)

Tail index
----------
For the k largest of n positive values X(1) >= X(2) >= ... >= X(n) and L(i) = log X(i):

    Hill (Hill, Annals of Statistics 3(5):1163-1174, 1975)
        gamma_H(k) = (1/k) sum_{i<=k} L(i) - L(k+1),       alpha_H(k) = 1 / gamma_H(k)
        sqrt(k) (gamma_H(k) - gamma) -> N(0, gamma^2) under the usual second-order conditions, so the
        level-(1 - a) interval for gamma is gamma_H * (1 +- z_{a/2} / sqrt(k)); the alpha interval is
        its reciprocal.
    Moment estimator (Dekkers, Einmahl and de Haan, Annals of Statistics 17(4):1833-1855, 1989)
        M1 = gamma_H(k),  M2 = (1/k) sum_{i<=k} (L(i) - L(k+1))^2
        gamma_M(k) = M1 + 1 - 0.5 / (1 - M1^2 / M2)
        valid for every real extreme-value index, so it also shows light tails (gamma <= 0).
    Choice of k (Clauset, Shalizi and Newman, SIAM Review 51(4):661-703, 2009): the k whose fitted
    Pareto tail P(X > x | X > u) = (x / u)^(-alpha_H(k)), u = X(k+1), is closest to the empirical
    tail of the k largest values in Kolmogorov-Smirnov distance
        D(k) = max_{i<=k} max(|(k-i+1)/k - F_k(X(i))|, |(k-i)/k - F_k(X(i))|),   F_k = 1 - (x/u)^-alpha.

The "stability plot" is returned as data: k, alpha_H with its interval, gamma_M and D(k) on a
geometric grid of at most `grid` values of k in [k_min, k_max]. D(k) costs O(k), so evaluating every
k would cost O(k_max^2); on the grid the cost is O(grid * k_max) and every reported value is exact.
k_max is bounded by `k_cap` (explicit parameter) so that one field of a large corpus stays cheap.
Ties make some log spacings zero; for count data (bytes, packets) the Hill path is therefore
stepwise and the estimate is reported with the count of distinct tail values.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import special, stats

#: 1 / Phi^-1(3/4): makes the MAD a consistent estimator of sigma under normality.
MAD_NORMAL = 1.482602218505602
#: Quantile levels of the standard robust summary.
SUMMARY_PROBS: tuple[float, ...] = (0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0)


def _clean(x: np.ndarray) -> np.ndarray:
    """1-D float64 view of the finite entries of `x` (NaN marks absent cells and is never a value)."""
    a = np.asarray(x, dtype=np.float64).reshape(-1)
    return a[np.isfinite(a)]


def mad(x: np.ndarray, *, normal: bool = True) -> float:
    """Median absolute deviation of the finite entries; scaled by `MAD_NORMAL` when `normal`."""
    a = _clean(x)
    if a.size == 0:
        return float("nan")
    raw = float(np.median(np.abs(a - np.median(a))))
    return raw * MAD_NORMAL if normal else raw


def bowley_skewness(x: np.ndarray) -> float:
    """Quartile skewness (Q3 + Q1 - 2 Q2) / (Q3 - Q1); NaN when the IQR is 0 or the sample is empty."""
    a = _clean(x)
    if a.size == 0:
        return float("nan")
    q1, q2, q3 = np.quantile(a, [0.25, 0.5, 0.75])
    den = q3 - q1
    return float((q3 + q1 - 2.0 * q2) / den) if den > 0 else float("nan")


def moors_kurtosis(x: np.ndarray) -> float:
    """Octile kurtosis ((E7 - E5) + (E3 - E1)) / (E6 - E2); NaN when E6 = E2."""
    a = _clean(x)
    if a.size == 0:
        return float("nan")
    e = np.quantile(a, np.arange(1, 8) / 8.0)                     # E1 ... E7
    den = e[5] - e[1]
    return float(((e[6] - e[4]) + (e[2] - e[0])) / den) if den > 0 else float("nan")


def robust_summary(x: np.ndarray, *, trim: float = 0.1) -> dict[str, float]:
    """Order and moment statistics of the finite entries of `x` (module docstring).

    Returns a flat dict: n, n_distinct, mean, std, the quantiles of `SUMMARY_PROBS` (q000 ... q100),
    median, mad, iqr, trimmed_mean, bowley_skewness, moors_kurtosis, share_zero, share_negative.
    An empty sample gives n = 0 and NaN elsewhere.
    """
    if not 0.0 <= trim < 0.5:
        raise ValueError("trim must be in [0, 0.5)")
    a = _clean(x)
    out: dict[str, float] = {"n": float(a.size)}
    keys = [f"q{round(p * 100):03d}" for p in SUMMARY_PROBS]
    if a.size == 0:
        for k in ("n_distinct", "mean", "std", *keys, "median", "mad", "iqr", "trimmed_mean", "bowley_skewness",
                  "moors_kurtosis", "share_zero", "share_negative"):
            out[k] = float("nan")
        out["n_distinct"] = 0.0
        return out
    qs = np.quantile(a, SUMMARY_PROBS)
    out["n_distinct"] = float(np.unique(a).size)
    out["mean"] = float(a.mean())
    out["std"] = float(a.std(ddof=1)) if a.size > 1 else float("nan")
    out.update({k: float(v) for k, v in zip(keys, qs, strict=True)})
    out["median"] = float(qs[SUMMARY_PROBS.index(0.5)])
    out["mad"] = mad(a)
    out["iqr"] = float(qs[SUMMARY_PROBS.index(0.75)] - qs[SUMMARY_PROBS.index(0.25)])
    out["trimmed_mean"] = float(stats.trim_mean(a, trim))
    out["bowley_skewness"] = bowley_skewness(a)
    out["moors_kurtosis"] = moors_kurtosis(a)
    out["share_zero"] = float(np.mean(a == 0.0))
    out["share_negative"] = float(np.mean(a < 0.0))
    return out


@dataclass(frozen=True)
class TailIndex:
    """Tail-index estimates and the stability path (module docstring).

    Attributes
    ----------
    n_positive : int
        Number of strictly positive finite values used.
    k : numpy.ndarray
        int64 [K] numbers of upper order statistics on the path (k_min ... k_max).
    alpha_hill, alpha_lower, alpha_upper : numpy.ndarray
        float64 [K] Hill tail index and its interval at `level`.
    gamma_moment : numpy.ndarray
        float64 [K] moment estimator of the extreme-value index (gamma = 1 / alpha when positive).
    ks_distance : numpy.ndarray
        float64 [K] Kolmogorov-Smirnov distance of the fitted Pareto tail.
    k_star : int
        The k minimising `ks_distance` (0 when no k could be evaluated).
    alpha_star, alpha_star_lower, alpha_star_upper, gamma_moment_star, ks_star : float
        Estimates at k_star.
    threshold : float
        u = X(k_star + 1), the value above which the tail model holds.
    distinct_tail_values : int
        Distinct values among the k_star largest (small numbers flag tie-dominated tails).
    level : float
        Confidence level of the intervals.
    """

    n_positive: int
    k: np.ndarray
    alpha_hill: np.ndarray
    alpha_lower: np.ndarray
    alpha_upper: np.ndarray
    gamma_moment: np.ndarray
    ks_distance: np.ndarray
    k_star: int
    alpha_star: float
    alpha_star_lower: float
    alpha_star_upper: float
    gamma_moment_star: float
    ks_star: float
    threshold: float
    distinct_tail_values: int
    level: float


def tail_index(
    x: np.ndarray,
    *,
    k_min: int = 10,
    k_max: int | None = None,
    max_share: float = 0.5,
    k_cap: int = 100_000,
    grid: int = 256,
    level: float = 0.95,
) -> TailIndex:
    """Hill and moment tail-index paths with the Clauset choice of k (module docstring).

    Parameters
    ----------
    x : array
        Sample; non-finite and non-positive entries are ignored (a tail index is defined on the
        positive half-line).
    k_min : int
        Smallest number of upper order statistics on the path (very small tails are noise).
    k_max : int or None
        Largest; default `floor(max_share * n_positive)`. Never more than n_positive - 1 or `k_cap`.
    max_share : float
        Share of the positive sample the path may reach when `k_max` is None.
    k_cap : int
        Hard bound on k_max (module docstring: cost of the Kolmogorov-Smirnov scan).
    grid : int
        Maximum number of k values on the path (geometric spacing; every value of k up to `grid`
        is used when the range is that small).
    level : float
        Confidence level of the Hill intervals.
    """
    if k_min < 2:
        raise ValueError("k_min must be >= 2")
    if not 0.0 < max_share < 1.0:
        raise ValueError("max_share must be in (0, 1)")
    if not 0.0 < level < 1.0:
        raise ValueError("level must be in (0, 1)")
    if grid < 2 or k_cap < k_min:
        raise ValueError("grid must be >= 2 and k_cap >= k_min")
    a = _clean(x)
    a = a[a > 0.0]
    n = int(a.size)
    hi = min(n - 1, k_cap, int(np.floor(max_share * n)) if k_max is None else int(k_max))
    empty = np.zeros(0)
    if hi < k_min:
        return TailIndex(n, np.zeros(0, dtype=np.int64), empty, empty, empty, empty, empty, 0, float("nan"),
                         float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), 0, level)
    # Upper order statistics: partial sort is enough, only the hi + 1 largest values are needed.
    top = np.sort(np.partition(a, n - hi - 1)[n - hi - 1:])[::-1] if hi + 1 < n else np.sort(a)[::-1]
    logs = np.log(top)                                               # L(1) >= ... >= L(hi + 1)
    ks_all = np.arange(1, hi + 1, dtype=np.int64)                    # k = 1 ... hi
    c1 = np.cumsum(logs[:hi])                                        # sum_{i<=k} L(i)
    c2 = np.cumsum(logs[:hi] ** 2)
    lk1 = logs[1: hi + 1]                                            # L(k+1)
    m1 = c1 / ks_all - lk1                                           # Hill gamma, [hi]
    m2 = (c2 - 2.0 * lk1 * c1 + ks_all * lk1 ** 2) / ks_all          # second log moment, [hi]
    # Path grid: geometric between k_min and hi (exact integers, unique).
    path_k = np.unique(np.round(np.geomspace(k_min, hi, num=min(grid, hi - k_min + 1))).astype(np.int64))
    j = path_k - 1                                                   # positions into the [hi] arrays
    with np.errstate(divide="ignore", invalid="ignore"):
        g = m1[j]
        alpha = np.where(g > 0, 1.0 / g, np.inf)
        z = float(special.ndtri(0.5 + level / 2.0))
        g_lo = g * (1.0 - z / np.sqrt(path_k))
        g_hi = g * (1.0 + z / np.sqrt(path_k))
        a_lo = np.where(g_hi > 0, 1.0 / g_hi, np.nan)
        a_hi = np.where(g_lo > 0, 1.0 / g_lo, np.inf)
        ratio = np.where(m2[j] > 0, g ** 2 / m2[j], np.nan)
        gamma_m = g + 1.0 - 0.5 / (1.0 - ratio)
    # Kolmogorov-Smirnov distance of the fitted Pareto tail at every grid k (Clauset et al. 2009).
    # For k the tail sample is X(1..k) and the model CDF is F(x) = 1 - (x/u)^(-alpha), u = X(k+1);
    # the empirical CDF jumps from (k-i)/k to (k-i+1)/k at X(i) (sorted descending).
    ksd = np.full(path_k.size, np.nan)
    for t, k in enumerate(path_k.tolist()):
        if not np.isfinite(alpha[t]):
            continue
        spacing = logs[:k] - logs[k]                                 # log(X(i) / u) >= 0, [k]
        f = 1.0 - np.exp(-alpha[t] * spacing)                        # model CDF at X(i)
        i = np.arange(1, k + 1)
        ksd[t] = float(np.max(np.maximum(np.abs((k - i + 1) / k - f), np.abs((k - i) / k - f))))
    if not np.isfinite(ksd).any():
        star = (0, float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), 0)
    else:
        t_s = int(np.argmin(np.where(np.isfinite(ksd), ksd, np.inf)))
        k_s = int(path_k[t_s])
        star = (k_s, float(alpha[t_s]), float(a_lo[t_s]), float(a_hi[t_s]), float(gamma_m[t_s]),
                float(ksd[t_s]), float(np.exp(logs[k_s])), int(np.unique(logs[:k_s]).size))
    return TailIndex(
        n_positive=n, k=path_k, alpha_hill=alpha, alpha_lower=a_lo, alpha_upper=a_hi,
        gamma_moment=gamma_m, ks_distance=ksd, k_star=star[0], alpha_star=star[1],
        alpha_star_lower=star[2], alpha_star_upper=star[3], gamma_moment_star=star[4], ks_star=star[5],
        threshold=star[6], distinct_tail_values=star[7], level=level,
    )


def freedman_diaconis_edges(x: np.ndarray, *, max_bins: int, min_bins: int = 1) -> np.ndarray:
    """Histogram edges with the Freedman-Diaconis width h = 2 IQR n^(-1/3), clipped to [min_bins, max_bins].

    Freedman and Diaconis, Z. Wahrscheinlichkeitstheorie verw. Gebiete 57:453-476, 1981. A sample with
    IQR 0 (or a single distinct value) falls back to `min_bins` equal bins over its range.
    """
    if max_bins < 1 or min_bins < 1 or min_bins > max_bins:
        raise ValueError("need 1 <= min_bins <= max_bins")
    a = _clean(x)
    if a.size == 0:
        return np.zeros(0)
    lo, hi = float(a.min()), float(a.max())
    if hi <= lo:
        return np.array([lo - 0.5, hi + 0.5])
    q1, q3 = np.quantile(a, [0.25, 0.75])
    h = 2.0 * (q3 - q1) * a.size ** (-1.0 / 3.0)
    bins = int(np.ceil((hi - lo) / h)) if h > 0 else min_bins
    bins = int(np.clip(bins, min_bins, max_bins))
    return np.linspace(lo, hi, bins + 1)


__all__ = [
    "MAD_NORMAL", "SUMMARY_PROBS", "TailIndex", "bowley_skewness", "freedman_diaconis_edges", "mad",
    "moors_kurtosis", "robust_summary", "tail_index",
]
