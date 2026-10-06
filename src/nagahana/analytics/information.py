"""Entropy and mutual-information estimators for discrete, continuous, mixed and partially observed data.

All quantities are in nats. The estimators are the building blocks of the EDA dependence tables and of
the information audit (`nagahana.lab.info_audit`, P-15).

Discrete variables
------------------
Plug-in entropy H = -sum_x p(x) log p(x) with empirical frequencies is biased downward; the
Miller-Madow correction adds (m - 1) / (2n), m the number of occupied cells (Miller, "Note on the
bias of information estimates", in Quastler (ed.), Information Theory in Psychology, Free Press 1955,
pp. 95-100). Mutual information I(X; Y) = H(X) + H(Y) - H(X, Y); with the correction applied to each
entropy the MI correction is -(m_XY - m_X - m_Y + 1) / (2n).

Continuous variables: KSG (Kraskov, Stoegbauer and Grassberger, Phys. Rev. E 69, 066138, 2004)
-------------------------------------------------------------------------------------------
Distances are max-norms. For point i let eps_i be the distance to its k-th nearest neighbour in the
joint space (x, y).
    Algorithm 1:  I = psi(k) + psi(N) - < psi(n_x + 1) + psi(n_y + 1) >,
                  n_x(i) = #{j != i : ||x_j - x_i|| < eps_i}   (strict), n_y likewise.
    Algorithm 2:  I = psi(k) - 1/k + psi(N) - < psi(n_x) + psi(n_y) >,
                  eps_x(i) = max over the k joint neighbours of ||x_j - x_i||,
                  n_x(i) = #{j != i : ||x_j - x_i|| <= eps_x(i)}   (closed), n_y likewise.
psi is the digamma function. Each column is scaled to unit standard deviation first (MI is invariant
under per-variable invertible maps; the k-NN geometry is not, so the scales are made comparable).

Continuous observables, discrete target (Ross, PLoS ONE 9(2):e87357, 2014)
------------------------------------------------------------------------
For point i with label c: d_i = distance to its k_i-th nearest neighbour among points with the same
label (k_i = min(k, N_c - 1)); m_i = #{j != i : ||x_j - x_i|| <= d_i} over all points;
    I = psi(N) - < psi(N_c(i)) > + < psi(k_i) > - < psi(m_i) >.
Labels with a single member carry no within-label distance and are left out (N counts the rest).

Ties
----
k-NN estimators assume continuous laws: a tie makes eps_i = 0 and the counts undefined. Every
continuous column receives independent Gaussian noise of standard deviation `jitter` (in units of the
column's standard deviation, default 1e-10) from a seeded generator, which breaks ties without moving
any value by a measurable amount. Strongly discrete columns belong in the discrete estimator.

Partially observed records (D-41)
---------------------------------
A record is (status pattern, discrete fields, continuous fields). Absent cells (NaN) are never
imputed. With D = (observation pattern, values of the observed discrete fields) and C = the observed
continuous fields, which are a fixed set inside each pattern, the chain rule gives exactly
    I(O; S) = I(D; S) + sum_d P(D = d) I(C; S | D = d).
The first term uses the discrete estimator, the second the Ross estimator inside each stratum. A
stratum too small for a k-NN estimate (fewer than `min_stratum` records) while S varies inside it
contributes nothing and its probability mass is reported as `unestimated_mass`; because conditional
MI is non-negative, the total is then a lower bound up to estimation error.

Intervals
---------
`bootstrap_interval` resamples records with replacement (valid for the discrete plug-in estimators).
Resampling with replacement duplicates points and breaks k-NN estimators, so `subsample_interval`
draws subsamples of size r = n - d without replacement and uses the delete-d jackknife (Shao and Wu,
Annals of Statistics 17(3):1176-1197, 1989): the deviations theta_b - mean(theta) scaled by sqrt(r / d)
estimate the sampling deviations of the full-sample estimate (for half-samples, r = d and the scale is 1).
The interval is the estimate plus the scaled deviation quantiles. Its coverage is checked by the tests
on data with known mutual information.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import special
from scipy.spatial import cKDTree


def _column_codes(col: np.ndarray) -> tuple[np.ndarray, int]:
    """Integer codes of one column: numeric columns by `np.unique` (all NaN equal), others by pandas."""
    a = np.asarray(col)
    if a.dtype.kind in "biuf":
        uniq, inv = np.unique(a, return_inverse=True)
        return inv.reshape(-1).astype(np.int64), int(uniq.size)
    c, uniq = pd.factorize(pd.Series(a.astype(object)), use_na_sentinel=False)
    return c.astype(np.int64), int(len(uniq))


def factorize(*columns: np.ndarray) -> tuple[np.ndarray, int]:
    """Joint integer codes of one or more aligned label columns (any hashable values; NaN is a value).

    Returns (codes int64 [n], number of distinct joint values); codes are 0 ... count - 1.
    """
    if not columns:
        raise ValueError("factorize needs at least one column")
    n = len(columns[0])
    codes = np.zeros(n, dtype=np.int64)
    card = 1
    for col in columns:
        if len(col) != n:
            raise ValueError("all columns must have the same length")
        c, k = _column_codes(col)
        k = max(k, 1)
        if card * k > 2**62:                                          # re-compact before the product overflows
            uniq, inv = np.unique(codes, return_inverse=True)
            codes, card = inv.astype(np.int64), int(uniq.size)
        codes = codes * k + c
        card *= k
    uniq_codes, inv = np.unique(codes, return_inverse=True)
    return inv.reshape(-1).astype(np.int64), int(uniq_codes.size)


def entropy_from_counts(counts: np.ndarray, *, correction: str = "none") -> float:
    """Entropy (nats) of a histogram; correction "none" (plug-in) or "miller_madow"."""
    c = np.asarray(counts, dtype=np.float64)
    c = c[c > 0]
    n = float(c.sum())
    if n <= 0:
        return float("nan")
    p = c / n
    h = float(-(p * np.log(p)).sum())
    if correction == "none":
        return h
    if correction == "miller_madow":
        return h + (c.size - 1) / (2.0 * n)
    raise ValueError(f"unknown entropy correction {correction!r}")


def entropy_discrete(x: np.ndarray, *, correction: str = "none") -> float:
    """Entropy (nats) of a discrete sample."""
    codes, k = factorize(x)
    return entropy_from_counts(np.bincount(codes, minlength=k), correction=correction)


def mutual_information_discrete(x: np.ndarray, y: np.ndarray, *, correction: str = "none") -> float:
    """I(X; Y) in nats for discrete samples (module docstring); correction "none" or "miller_madow"."""
    if len(x) != len(y) or len(x) == 0:
        raise ValueError("x and y must be non-empty and of equal length")
    cx, kx = factorize(x)
    cy, ky = factorize(y)
    cxy, kxy = factorize(cx, cy)
    hx = entropy_from_counts(np.bincount(cx, minlength=kx))
    hy = entropy_from_counts(np.bincount(cy, minlength=ky))
    hxy = entropy_from_counts(np.bincount(cxy, minlength=kxy))
    mi = hx + hy - hxy
    if correction == "none":
        return float(mi)
    if correction == "miller_madow":
        return float(mi - (kxy - kx - ky + 1) / (2.0 * len(x)))
    raise ValueError(f"unknown MI correction {correction!r}")


def _prepare_continuous(a: np.ndarray, *, jitter: float, rng: np.random.Generator) -> np.ndarray:
    """[n, d] float64, columns scaled to unit standard deviation, plus tie-breaking noise (module docstring)."""
    z = np.asarray(a, dtype=np.float64)
    z = z.reshape(-1, 1) if z.ndim == 1 else z.copy()
    if not np.isfinite(z).all():
        raise ValueError("continuous inputs must be finite (absent cells are handled by the caller, D-41)")
    sd = z.std(axis=0)
    sd[sd == 0] = 1.0                                                  # a constant column stays constant
    z = (z - z.mean(axis=0)) / sd
    if jitter > 0:
        z = z + rng.normal(0.0, jitter, size=z.shape)
    return z


def _count_within(points: np.ndarray, radius: np.ndarray, *, strict: bool) -> np.ndarray:
    """For each point i: #{j != i : ||p_j - p_i||_inf < r_i} (strict) or <= r_i; points [n, d], radius [n]."""
    n, d = points.shape
    if d == 1:
        # One dimension: exact counts by binary search on the sorted coordinate.
        x = points[:, 0]
        xs = np.sort(x)
        if strict:
            hi = np.searchsorted(xs, x + radius, side="left")
            lo = np.searchsorted(xs, x - radius, side="right")
        else:
            hi = np.searchsorted(xs, x + radius, side="right")
            lo = np.searchsorted(xs, x - radius, side="left")
        return (hi - lo - 1).astype(np.int64)                           # minus the point itself
    tree = cKDTree(points)
    r = np.nextafter(radius, 0.0) if strict else radius                # open ball as the largest closed one inside
    r = np.maximum(r, 0.0)
    counts = tree.query_ball_point(points, r=r, p=np.inf, return_length=True, workers=-1)
    return np.asarray(counts, dtype=np.int64) - 1


def mutual_information_ksg(
    x: np.ndarray,
    y: np.ndarray,
    *,
    k: int = 3,
    algorithm: int = 1,
    jitter: float = 1e-10,
    seed: int = 0,
) -> float:
    """KSG estimate of I(X; Y) in nats for continuous X [n, dx] and Y [n, dy] (module docstring)."""
    if algorithm not in (1, 2):
        raise ValueError("algorithm must be 1 or 2")
    rng = np.random.default_rng(seed)
    xs = _prepare_continuous(x, jitter=jitter, rng=rng)
    ys = _prepare_continuous(y, jitter=jitter, rng=rng)
    n = xs.shape[0]
    if ys.shape[0] != n:
        raise ValueError("x and y must have the same number of rows")
    if not 1 <= k < n:
        raise ValueError(f"k must be in [1, n-1]; got k={k}, n={n}")
    joint = np.hstack([xs, ys])                                        # [n, dx + dy]
    dist, idx = cKDTree(joint).query(joint, k=k + 1, p=np.inf, workers=-1)
    if algorithm == 1:
        eps = dist[:, -1]                                              # distance to the k-th neighbour
        nx = _count_within(xs, eps, strict=True)
        ny = _count_within(ys, eps, strict=True)
        val = special.digamma(k) + special.digamma(n) - np.mean(special.digamma(nx + 1) + special.digamma(ny + 1))
        return float(val)
    nb = idx[:, 1:]                                                    # [n, k] neighbours (self excluded)
    eps_x = np.abs(xs[nb] - xs[:, None, :]).max(axis=(1, 2))
    eps_y = np.abs(ys[nb] - ys[:, None, :]).max(axis=(1, 2))
    nx = _count_within(xs, eps_x, strict=False)
    ny = _count_within(ys, eps_y, strict=False)
    val = special.digamma(k) - 1.0 / k + special.digamma(n) - np.mean(special.digamma(nx) + special.digamma(ny))
    return float(val)


def mutual_information_mixed(
    x: np.ndarray,
    labels: np.ndarray,
    *,
    k: int = 3,
    jitter: float = 1e-10,
    seed: int = 0,
) -> float:
    """Ross (2014) estimate of I(X; L) in nats for continuous X [n, d] and discrete labels L [n]."""
    if k < 1:
        raise ValueError("k must be >= 1")
    rng = np.random.default_rng(seed)
    xs = _prepare_continuous(x, jitter=jitter, rng=rng)
    if xs.shape[0] != len(labels):
        raise ValueError("x and labels must have the same number of rows")
    codes, n_codes = factorize(labels)
    counts = np.bincount(codes, minlength=n_codes)
    keep = counts[codes] >= 2                                          # singletons have no same-label neighbour
    xs, codes = xs[keep], codes[keep]
    n = int(codes.size)
    if n < 2 or np.unique(codes).size < 1:
        return float("nan")
    counts = np.bincount(codes, minlength=n_codes)
    radius = np.empty(n)
    k_all = np.empty(n)
    for c in np.unique(codes).tolist():
        m = codes == c
        kc = int(min(k, counts[c] - 1))
        dist, _ = cKDTree(xs[m]).query(xs[m], k=kc + 1, p=np.inf, workers=-1)
        radius[m] = dist[:, -1] if dist.ndim == 2 else dist
        k_all[m] = kc
    m_all = _count_within(xs, radius, strict=False)                    # includes the k_i same-label neighbours
    val = (special.digamma(n) - np.mean(special.digamma(counts[codes])) + np.mean(special.digamma(k_all))
           - np.mean(special.digamma(np.maximum(m_all, 1))))
    return float(val)


@dataclass(frozen=True)
class ObservablesMI:
    """I(O; S) for partially observed records (module docstring, "Partially observed records").

    Attributes
    ----------
    total : float
        discrete + continuous, in nats.
    discrete : float
        I(D; S): observation pattern and observed discrete fields.
    continuous : float
        sum_d P(d) I(C; S | D = d).
    unestimated_mass : float
        Probability mass of strata that were too small for a k-NN estimate while S varied in them.
    n_strata : int
        Distinct values of D.
    n : int
        Records used.
    """

    total: float
    discrete: float
    continuous: float
    unestimated_mass: float
    n_strata: int
    n: int


def mutual_information_observables(
    values: np.ndarray,
    hidden: np.ndarray,
    *,
    discrete: np.ndarray,
    k: int = 3,
    min_stratum: int = 20,
    correction: str = "miller_madow",
    jitter: float = 1e-10,
    seed: int = 0,
) -> ObservablesMI:
    """I(O; S) for records with absent cells (NaN) by the chain rule (module docstring).

    Parameters
    ----------
    values : array [n, d]
        Observables; NaN where a field carries no value (never imputed).
    hidden : array [n]
        Discrete hidden quantity (for example the ATT&CK stage code).
    discrete : bool array [d]
        True for discrete fields (categorical, bitmask), False for continuous ones.
    k, min_stratum, correction, jitter, seed
        Estimator settings (module docstring).
    """
    v = np.asarray(values, dtype=np.float64)
    v = v.reshape(-1, 1) if v.ndim == 1 else v
    disc = np.asarray(discrete, dtype=bool).reshape(-1)
    n, d = v.shape
    if disc.shape != (d,):
        raise ValueError(f"discrete must have one flag per column ({d})")
    if len(hidden) != n or n == 0:
        raise ValueError("values and hidden must be non-empty with the same number of rows")
    observed = np.isfinite(v)                                          # [n, d]
    # D = (pattern bits, observed discrete values); absent discrete cells are part of the pattern only.
    parts: list[np.ndarray] = [observed[:, j] for j in range(d)]
    parts += [np.where(observed[:, j], v[:, j], -np.inf) for j in np.nonzero(disc)[0]]
    dcode, n_strata = factorize(*parts) if parts else (np.zeros(n, dtype=np.int64), 1)
    s_codes, _ = factorize(hidden)
    i_d = mutual_information_discrete(dcode, s_codes, correction=correction)
    cont = 0.0
    unest = 0.0
    order = np.argsort(dcode, kind="stable")
    bounds = np.flatnonzero(np.diff(dcode[order])) + 1
    for rows in np.split(order, bounds):
        cols = np.nonzero(~disc & observed[rows[0]])[0]                 # continuous fields observed in this pattern
        if cols.size == 0:
            continue
        s_r = s_codes[rows]
        if np.unique(s_r).size < 2:
            continue                                                   # S constant in the stratum: exactly 0
        if rows.size < min_stratum:
            unest += rows.size / n
            continue
        mi = mutual_information_mixed(v[np.ix_(rows, cols)], s_r, k=k, jitter=jitter, seed=seed)
        if np.isfinite(mi):
            cont += rows.size / n * mi
        else:
            unest += rows.size / n
    return ObservablesMI(total=float(i_d + cont), discrete=float(i_d), continuous=float(cont),
                         unestimated_mass=float(unest), n_strata=int(n_strata), n=int(n))


def information_coefficient(mi: float) -> float:
    """Linfoot's informational coefficient of correlation sqrt(1 - exp(-2 I)) (|rho| for Gaussians).

    Linfoot, Information and Control 1(1):85-89, 1957. Negative estimates (estimation noise) map to 0.
    """
    if not np.isfinite(mi):
        return float("nan")
    return float(np.sqrt(1.0 - np.exp(-2.0 * max(mi, 0.0))))


def bootstrap_interval(
    statistic: Callable[[np.ndarray], float],
    n: int,
    *,
    n_boot: int,
    level: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, np.ndarray]:
    """Percentile bootstrap interval of `statistic(index_array)` over resampled row indices.

    Returns (lower, upper, replicate values [n_boot]). For estimators that tolerate duplicated rows
    (the discrete plug-in estimators).
    """
    if n_boot < 2:
        raise ValueError("n_boot must be >= 2")
    rng = np.random.default_rng(seed)
    reps = np.array([statistic(rng.integers(0, n, size=n)) for _ in range(n_boot)])
    a = (1.0 - level) / 2.0
    finite = reps[np.isfinite(reps)]
    if finite.size == 0:
        return float("nan"), float("nan"), reps
    lo, hi = np.quantile(finite, [a, 1.0 - a])
    return float(lo), float(hi), reps


def subsample_interval(
    statistic: Callable[[np.ndarray], float],
    n: int,
    *,
    estimate: float,
    n_sub: int,
    level: float = 0.95,
    seed: int = 0,
    size: int | None = None,
) -> tuple[float, float, np.ndarray]:
    """Delete-d jackknife interval for k-NN estimators (module docstring, "Intervals").

    Subsamples of size r (default n // 2) without replacement give replicates theta_b; with d = n - r,
    interval = estimate + quantiles((theta_b - mean(theta)) * sqrt(r / d)).
    """
    if n_sub < 2:
        raise ValueError("n_sub must be >= 2")
    r = n // 2 if size is None else int(size)
    d = n - r
    if not 1 <= r < n:
        raise ValueError("the subsample size must be in [1, n - 1]")
    rng = np.random.default_rng(seed)
    reps = np.array([statistic(rng.choice(n, size=r, replace=False)) for _ in range(n_sub)])
    finite = reps[np.isfinite(reps)]
    if finite.size < 2:
        return float("nan"), float("nan"), reps
    dev = (finite - finite.mean()) * np.sqrt(r / d)
    a = (1.0 - level) / 2.0
    lo, hi = np.quantile(dev, [a, 1.0 - a])
    return float(estimate + lo), float(estimate + hi), reps


__all__ = [
    "ObservablesMI", "bootstrap_interval", "entropy_discrete", "entropy_from_counts", "factorize",
    "information_coefficient", "mutual_information_discrete", "mutual_information_ksg", "mutual_information_mixed",
    "mutual_information_observables", "subsample_interval",
]
