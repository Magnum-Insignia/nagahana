"""Pairwise dependence: Pearson, Spearman, distance correlation, mutual information, Cramer's V.

Every measure is computed on pairwise-complete records: a record enters the pair (i, j) only when both
fields carry a value (D-41: an absent cell is never imputed). Heavy-tailed magnitudes enter Pearson
correlation after the signed log transform slog1p(x) = sign(x) log(1 + |x|), the transform of the
model's numeric encoder (AS-31); Spearman and distance correlation are rank- and scale-free in the
senses noted below.

Pearson and Spearman
--------------------
r = cov(x, y) / (sd(x) sd(y)); the interval uses Fisher's z = atanh(r) with standard error
1 / sqrt(n - 3) (Fisher, Biometrika 10(4):507-521, 1915). Spearman's rho is Pearson's r of the
mid-ranks (ties share their mean rank). The p-values use t = r sqrt((n - 2) / (1 - r^2)) with n - 2
degrees of freedom (exact for Pearson under bivariate normality, the usual approximation for Spearman).

Distance correlation (Szekely, Rizzo and Bakirov, Annals of Statistics 35(6):2769-2794, 2007)
-----------------------------------------------------------------------------------------
With a_ij = |x_i - x_j|, b_ij = |y_i - y_j|, row sums a_i., b_i. and totals a.., b..:
    S = sum_{i,j} a_ij b_ij,   T = sum_i a_i. b_i.
    V-statistic   dCov^2_n = S / n^2 - 2 T / n^3 + a.. b.. / n^4      (the 2007 definition, >= 0)
    U-statistic   Omega_n  = S / (n(n-3)) - 2 T / (n(n-2)(n-3)) + a.. b.. / (n(n-1)(n-2)(n-3))
                  (unbiased; Szekely and Rizzo, Annals of Statistics 42(6):2382-2412, 2014)
    dCor = dCov(x, y) / sqrt(dCov(x, x) dCov(y, y)); bias-corrected R* = Omega(x, y) / sqrt(Omega(x, x) Omega(y, y)).
dCor = 0 exactly when x and y are independent (population version), unlike Pearson's r.

Fast exact algorithm (univariate x and y). Row sums come from sorted prefix sums:
a_i. = x_i (2 r_i - n) + sum(x) - 2 P_i with r_i the number of values <= x_i and P_i their sum.
For S, sort the records by x (ascending); for j < i, |x_i - x_j| = x_i - x_j and |y_i - y_j| =
s_ij (y_i - y_j), s_ij = sign(y_i - y_j), so
    sum_{j<i} a_ij b_ij = x_i y_i G_1(i) - x_i G_y(i) - y_i G_x(i) + G_xy(i),
    G_c(i) = sum_{j<i} s_ij c_j = L_c(i) + LE_c(i) - P_c(i),
with L_c, LE_c the sums of c_j over earlier records with y_j < y_i (y_j <= y_i) and P_c the prefix sum.
This decomposition is that of Huo and Szekely (Technometrics 58(4):435-447, 2016). The dominance sums
L and LE are computed level by level over the dyadic blocks of the x-order (each pair j < i meets in
exactly one level, as the two halves of one block): at every level one sort and three binary searches
over all records, O(n log n) per level, O(n log^2 n) in total and fully vectorised. Both variables are
centred and scaled first; distance correlation is invariant to that, and it keeps the products of
large magnitudes well inside float64 precision. `distance_correlation_naive` evaluates the double-
centred definition in O(n^2) and is used by the tests to check the fast path.

Mutual information and Cramer's V
---------------------------------
Numeric pairs use the KSG estimator (`information.mutual_information_ksg`), reported with Linfoot's
coefficient sqrt(1 - exp(-2 I)) on the |rho| scale. Categorical pairs use the plug-in estimator with
the Miller-Madow correction and the bias-corrected Cramer's V of Bergsma (Journal of the Korean
Statistical Society 42(3):323-328, 2013):
    phi2 = chi2 / n,  phi2c = max(0, phi2 - (r-1)(c-1)/(n-1)),  rc = r - (r-1)^2/(n-1),  cc = c - (c-1)^2/(n-1)
    V~ = sqrt(phi2c / min(rc - 1, cc - 1)).
Mixed (categorical, numeric) pairs use the Ross estimator (`information.mutual_information_mixed`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import special, stats

from nagahana.analytics import information


def slog1p(x: np.ndarray) -> np.ndarray:
    """Signed log transform sign(x) log(1 + |x|) (AS-31); NaN stays NaN."""
    a = np.asarray(x, dtype=np.float64)
    return np.sign(a) * np.log1p(np.abs(a))


@dataclass(frozen=True)
class Correlation:
    """A correlation coefficient with its interval and p-value (n pairwise-complete records)."""

    r: float
    lower: float
    upper: float
    p_value: float
    n: int


def _complete(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pairwise-complete float64 views of x and y (both finite)."""
    a = np.asarray(x, dtype=np.float64).reshape(-1)
    b = np.asarray(y, dtype=np.float64).reshape(-1)
    if a.shape != b.shape:
        raise ValueError("x and y must have the same length")
    ok = np.isfinite(a) & np.isfinite(b)
    return a[ok], b[ok]


def _correlation(a: np.ndarray, b: np.ndarray, level: float) -> Correlation:
    """Pearson r of complete samples with the Fisher-z interval and the t-test p-value."""
    n = int(a.size)
    if n < 3 or a.std() == 0 or b.std() == 0:
        return Correlation(float("nan"), float("nan"), float("nan"), float("nan"), n)
    r = float(np.clip(np.corrcoef(a, b)[0, 1], -1.0, 1.0))
    if n > 3:
        z = np.arctanh(np.clip(r, -1 + 1e-15, 1 - 1e-15))
        h = float(special.ndtri(0.5 + level / 2.0)) / np.sqrt(n - 3)
        lo, hi = float(np.tanh(z - h)), float(np.tanh(z + h))
    else:
        lo, hi = float("nan"), float("nan")
    if abs(r) >= 1.0:
        p = 0.0
    else:
        t = r * np.sqrt((n - 2) / (1.0 - r * r))
        p = float(2.0 * stats.t.sf(abs(t), df=n - 2))
    return Correlation(r, lo, hi, p, n)


def pearson(x: np.ndarray, y: np.ndarray, *, transform: str = "none", level: float = 0.95) -> Correlation:
    """Pearson correlation on pairwise-complete records; transform "none" or "slog1p" (AS-31)."""
    a, b = _complete(x, y)
    if transform == "slog1p":
        a, b = slog1p(a), slog1p(b)
    elif transform != "none":
        raise ValueError(f"unknown transform {transform!r}")
    return _correlation(a, b, level)


def spearman(x: np.ndarray, y: np.ndarray, *, level: float = 0.95) -> Correlation:
    """Spearman's rho (Pearson r of mid-ranks) on pairwise-complete records."""
    a, b = _complete(x, y)
    return _correlation(stats.rankdata(a), stats.rankdata(b), level)


def _row_sums(x: np.ndarray) -> np.ndarray:
    """a_i. = sum_j |x_i - x_j| for all i in O(n log n) (module docstring)."""
    n = x.size
    order = np.argsort(x, kind="stable")
    xs = x[order]
    csum = np.cumsum(xs)
    # r_i: number of values <= x_i, P_i: their sum (ties included; their distance is 0 anyway).
    r = np.searchsorted(xs, x, side="right")
    p = np.where(r > 0, csum[np.maximum(r - 1, 0)], 0.0)
    return x * (2.0 * r - n) + csum[-1] - 2.0 * p


def _dominance_sums(y_rank: np.ndarray, w: np.ndarray, n_ranks: int) -> tuple[np.ndarray, np.ndarray]:
    """For records in x-order: sums of w over earlier records with smaller (lt) and not larger (le) y.

    y_rank: int64 [n] dense ranks of y (ties share a rank); w: float64 [n, m] weights.
    Returns (lt, le), both [n, m]. Level-by-level over dyadic blocks (module docstring).
    """
    n = y_rank.shape[0]
    lt = np.zeros_like(w)
    le = np.zeros_like(w)
    pos = np.arange(n, dtype=np.int64)
    level = 0
    while (1 << level) < n:
        seg = pos >> (level + 1)                                       # block of size 2^(level+1)
        right = ((pos >> level) & 1).astype(bool)                      # second half of its block
        left_idx = np.flatnonzero(~right)
        right_idx = np.flatnonzero(right)
        if right_idx.size:
            key_left = seg[left_idx] * n_ranks + y_rank[left_idx]      # (block, y-rank) of the first halves
            order = np.argsort(key_left, kind="stable")
            kl = key_left[order]
            csum = np.vstack([np.zeros((1, w.shape[1])), np.cumsum(w[left_idx][order], axis=0)])
            seg_r = seg[right_idx]
            k_r = seg_r * n_ranks + y_rank[right_idx]
            start = np.searchsorted(kl, seg_r * n_ranks, side="left")  # first record of the same block
            lo = np.searchsorted(kl, k_r, side="left")                 # strictly smaller y in the block
            hi = np.searchsorted(kl, k_r, side="right")                # smaller or equal y in the block
            lt[right_idx] += csum[lo] - csum[start]
            le[right_idx] += csum[hi] - csum[start]
        level += 1
    return lt, le


def _dcov_terms(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float]:
    """(S, T, a.., b..) of the module docstring for univariate x, y in O(n log^2 n)."""
    n = x.size
    order = np.argsort(x, kind="stable")
    xs, ys = x[order], y[order]
    _, y_rank = np.unique(ys, return_inverse=True)
    y_rank = y_rank.reshape(-1).astype(np.int64)
    w = np.stack([np.ones(n), ys, xs, xs * ys], axis=1)                # weights 1, y_j, x_j, x_j y_j, [n, 4]
    lt, le = _dominance_sums(y_rank, w, int(y_rank.max()) + 1)
    prefix = np.vstack([np.zeros((1, 4)), np.cumsum(w, axis=0)[:-1]])  # sum over j < i, [n, 4]
    g = lt + le - prefix                                               # sum_{j<i} s_ij w_j
    half = float(np.sum(xs * ys * g[:, 0] - xs * g[:, 1] - ys * g[:, 2] + g[:, 3]))
    a_row = _row_sums(x)
    b_row = _row_sums(y)
    return 2.0 * half, float(np.dot(a_row, b_row)), float(a_row.sum()), float(b_row.sum())


def _dvar_terms(x: np.ndarray) -> tuple[float, float, float]:
    """(S_xx, T_xx, a..) with S_xx = sum_{i,j} (x_i - x_j)^2 = 2 (n sum x^2 - (sum x)^2) in closed form."""
    n = x.size
    s = 2.0 * (n * float(np.dot(x, x)) - float(x.sum()) ** 2)
    a_row = _row_sums(x)
    return s, float(np.dot(a_row, a_row)), float(a_row.sum())


@dataclass(frozen=True)
class DistanceCorrelation:
    """Distance covariance and correlation of one pair (module docstring).

    dcor: V-statistic distance correlation in [0, 1]. dcor_unbiased: bias-corrected R* (may be < 0).
    dcov2, dcov2_unbiased: squared distance covariances (of the standardised variables).
    p_value: permutation p-value of R* (NaN when no permutations were requested). n: records used.
    """

    dcor: float
    dcor_unbiased: float
    dcov2: float
    dcov2_unbiased: float
    p_value: float
    n: int


def _standardise(a: np.ndarray) -> np.ndarray:
    sd = a.std()
    return (a - a.mean()) / (sd if sd > 0 else 1.0)


def _dcor_from_terms(n: int, sxy: tuple[float, float, float, float], sxx: tuple[float, float, float],
                     syy: tuple[float, float, float]) -> tuple[float, float, float, float]:
    """(dcor, R*, dcov2_V, Omega) from the sufficient sums."""
    s, t, ax, by = sxy

    def v_stat(s_: float, t_: float, a_: float, b_: float) -> float:
        return s_ / n**2 - 2.0 * t_ / n**3 + a_ * b_ / n**4

    def u_stat(s_: float, t_: float, a_: float, b_: float) -> float:
        return s_ / (n * (n - 3)) - 2.0 * t_ / (n * (n - 2) * (n - 3)) + a_ * b_ / (n * (n - 1) * (n - 2) * (n - 3))

    vxy, vxx, vyy = v_stat(s, t, ax, by), v_stat(sxx[0], sxx[1], sxx[2], sxx[2]), v_stat(syy[0], syy[1], syy[2], syy[2])
    uxy, uxx, uyy = u_stat(s, t, ax, by), u_stat(sxx[0], sxx[1], sxx[2], sxx[2]), u_stat(syy[0], syy[1], syy[2], syy[2])
    den_v = np.sqrt(vxx * vyy)
    dcor = float(np.sqrt(max(vxy, 0.0) / den_v)) if den_v > 0 else float("nan")
    den_u = np.sqrt(uxx * uyy) if uxx > 0 and uyy > 0 else 0.0
    r_star = float(uxy / den_u) if den_u > 0 else float("nan")
    return dcor, r_star, float(vxy), float(uxy)


def distance_correlation(
    x: np.ndarray,
    y: np.ndarray,
    *,
    permutations: int = 0,
    seed: int = 0,
) -> DistanceCorrelation:
    """Exact distance correlation of univariate x and y in O(n log^2 n) (module docstring).

    Pairwise-complete records only. With `permutations` > 0 the p-value of R* is
    (1 + #{R*_perm >= R*}) / (1 + permutations), y permuted by a seeded generator.
    """
    a, b = _complete(x, y)
    n = int(a.size)
    if n < 4 or a.std() == 0 or b.std() == 0:
        return DistanceCorrelation(float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), n)
    a, b = _standardise(a), _standardise(b)
    sxx, syy = _dvar_terms(a), _dvar_terms(b)
    dcor, r_star, v, u = _dcor_from_terms(n, _dcov_terms(a, b), sxx, syy)
    p = float("nan")
    if permutations > 0:
        rng = np.random.default_rng(seed)
        exceed = 0
        for _ in range(permutations):
            bp = b[rng.permutation(n)]
            _, r_perm, _, _ = _dcor_from_terms(n, _dcov_terms(a, bp), sxx, syy)
            exceed += int(r_perm >= r_star)
        p = (1.0 + exceed) / (1.0 + permutations)
    return DistanceCorrelation(dcor, r_star, v, u, p, n)


def distance_correlation_naive(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """(dcor, R*) from the O(n^2) double-centred and U-centred definitions (reference for tests)."""
    a, b = _complete(x, y)
    n = a.size
    a, b = _standardise(a), _standardise(b)
    da = np.abs(a[:, None] - a[None, :])
    db = np.abs(b[:, None] - b[None, :])

    def dc(m: np.ndarray) -> np.ndarray:                               # double centring
        return m - m.mean(axis=0, keepdims=True) - m.mean(axis=1, keepdims=True) + m.mean()

    def uc(m: np.ndarray) -> np.ndarray:                               # U-centring (diagonal set to 0)
        out = (m - m.sum(axis=0, keepdims=True) / (n - 2) - m.sum(axis=1, keepdims=True) / (n - 2)
               + m.sum() / ((n - 1) * (n - 2)))
        np.fill_diagonal(out, 0.0)
        return out

    aa, bb = dc(da), dc(db)
    v = (aa * bb).mean()
    dcor = float(np.sqrt(max(v, 0.0) / np.sqrt((aa * aa).mean() * (bb * bb).mean())))
    ua, ub = uc(da), uc(db)
    uxy = (ua * ub).sum() / (n * (n - 3))
    uxx = (ua * ua).sum() / (n * (n - 3))
    uyy = (ub * ub).sum() / (n * (n - 3))
    return dcor, float(uxy / np.sqrt(uxx * uyy))


def cramers_v(x: np.ndarray, y: np.ndarray, *, bias_corrected: bool = True) -> float:
    """Cramer's V of two categorical samples; Bergsma's bias correction when `bias_corrected`."""
    cx, kx = information.factorize(x)
    cy, ky = information.factorize(y)
    n = cx.size
    if n < 2 or kx < 2 or ky < 2:
        return float("nan")
    table = np.zeros((kx, ky))
    np.add.at(table, (cx, cy), 1.0)
    expected = table.sum(axis=1, keepdims=True) * table.sum(axis=0, keepdims=True) / n
    chi2 = float(((table - expected) ** 2 / expected).sum())
    phi2 = chi2 / n
    if not bias_corrected:
        return float(np.sqrt(phi2 / min(kx - 1, ky - 1)))
    phi2c = max(0.0, phi2 - (kx - 1) * (ky - 1) / (n - 1))
    rc = kx - (kx - 1) ** 2 / (n - 1)
    cc = ky - (ky - 1) ** 2 / (n - 1)
    den = min(rc - 1, cc - 1)
    return float(np.sqrt(phi2c / den)) if den > 0 else float("nan")


__all__ = [
    "Correlation", "DistanceCorrelation", "cramers_v", "distance_correlation", "distance_correlation_naive",
    "pearson", "slog1p", "spearman",
]
