"""Bootstrap confidence intervals: resampling schemes as weight matrices, block length, percentile and BCa.

Resamples as weights

A bootstrap resample is represented by the number of times each unit was drawn, a count vector
c [n] with sum_i c_i equal to the resample size. Every metric of this package accepts such counts as
unit weights (`_arrays`), so evaluating a metric on a resample is one weighted evaluation and a chunk
of resamples is one weight matrix [b, n]. The result is identical to resampling the data explicitly.

Schemes

    iid         n draws with replacement from all units (Efron, Annals of Statistics 7:1-26, 1979).
    stratified  draws with replacement inside each stratum, keeping every stratum's size (for example
                positives and negatives separately, so that a rare class is present in every resample).
    stationary  the stationary bootstrap of Politis and Romano (JASA 89:1303-1313, 1994) inside each
                series (one series per dataset and network, units in time order): blocks start at a
                uniform position, have geometric lengths with mean L and wrap around the end of the
                series, so the resampled series is stationary and keeps the dependence within blocks.
    cluster     draws whole clusters (attack episodes, networks) with replacement, keeping all units
                of a drawn cluster (Davison and Hinkley, Bootstrap Methods and their Application,
                Cambridge 1997, section 3.8; Field and Welsh, JRSS-B 69:369-390, 2007). Optionally
                stratified over clusters.

Block length

With L = "auto", the mean block length of each series is the estimate of Politis and White
(Econometric Reviews 23:53-70, 2004) with the correction of Patton, Politis and White (Econometric
Reviews 28:372-375, 2009), computed on a per-unit loss series of the metric (the series whose mean
the metric depends on). With R(k) the sample autocovariance and rho(k) = R(k) / R(0):

    m = the smallest m >= 0 with |rho(m + j)| < 2 sqrt(log10(N) / N) for j = 1 .. K_N,
        K_N = max(5, ceil(sqrt(log10 N))),   M = max(2 m, 1),
    lambda(t) = 1 for |t| <= 1/2, 2 (1 - |t|) for 1/2 < |t| <= 1, 0 otherwise (flat-top window),
    G = sum_{|k| <= M} lambda(k / M) |k| R(k),     g = sum_{|k| <= M} lambda(k / M) R(k),
    L_SB = (2 G^2 / D_SB)^(1/3) N^(1/3),          D_SB = 2 g^2,

capped at b_max = ceil(min(3 sqrt(N), N / 3)) and floored at 1. A series shorter than 8 units uses
L = 1. The thesis evaluation chapter asks for block lengths matched to the measured autocorrelation.

Intervals

Percentile: the (1 - c)/2 and (1 + c)/2 quantiles of the replicates. BCa (Efron, JASA 82:171-185,
1987; DiCiccio and Efron, Statistical Science 11:189-228, 1996): with z0 = Phi^-1(share of replicates
below the estimate, ties counting 1/2) and the acceleration

    a = sum_g (tbar - t_(g))^3 / (6 (sum_g (tbar - t_(g))^2)^(3/2))

from a grouped jackknife (delete-a-group values t_(g): random groups for iid and stratified schemes,
contiguous blocks of each series for the stationary scheme as in the delete-a-block jackknife of
Kunsch, Annals of Statistics 17:1217-1241, 1989, and whole clusters for the cluster scheme), the
interval is the replicate quantiles at Phi(z0 + (z0 + z) / (1 - a (z0 + z))) for z = Phi^-1((1 -+ c)/2).
When z0 is not finite (every replicate on one side of the estimate) or an adjusted level leaves
(0, 1), the percentile interval is returned and the method is reported as "percentile".

Replicates that are undefined (NaN, for example a precision in a resample where nothing was flagged)
are excluded from the interval and counted (`n_valid`). Infinite replicates are values (a median lead
time of -inf when most episodes were missed) and stay in; when any is present, the interval ends are
the exact order statistics of the replicates (inverted empirical distribution) instead of linear
interpolations, which are undefined between an infinite and a finite value.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import stats

from nagahana.core.errors import InvariantViolation
from nagahana.evaluation._arrays import codes

#: A statistic evaluated on a weight matrix [b, n], returning [b] or [b, q].
Statistic = Callable[[np.ndarray], np.ndarray]


def optimal_block_length(x: Any) -> float:
    """Mean block length of the stationary bootstrap for series x (Politis-White, Patton correction)."""
    v = np.asarray(x, dtype=np.float64)
    n = v.size
    if n < 8:
        return 1.0
    e = v - v.mean()
    r0 = float(e @ e) / n
    if r0 <= 0.0:
        return 1.0                                                      # a constant series: no dependence
    kn = max(5, math.ceil(math.sqrt(math.log10(n))))
    m_max = min(n - 1, math.ceil(math.sqrt(n)) + kn)
    # Autocovariances R(k) = (1/n) sum_t e_t e_{t+k}, k = 0 .. m_max.
    acov = np.array([float(e[: n - k] @ e[k:]) / n for k in range(m_max + 1)])
    rho = acov / r0
    crit = 2.0 * math.sqrt(math.log10(n) / n)
    m_hat = None
    for m in range(0, m_max - kn + 1):
        if np.all(np.abs(rho[m + 1 : m + 1 + kn]) < crit):
            m_hat = m
            break
    big_m = min(max(2 * m_hat, 1), m_max) if m_hat is not None else m_max
    k = np.arange(1, big_m + 1)
    t = k / big_m
    lam = np.where(t <= 0.5, 1.0, 2.0 * (1.0 - t))
    g_sum = 2.0 * float(np.sum(lam * k * acov[1 : big_m + 1]))         # sum over +-k of lambda |k| R(k)
    g0 = acov[0] + 2.0 * float(np.sum(lam * acov[1 : big_m + 1]))       # sum over |k| <= M of lambda R(k)
    if g0 <= 0.0:
        return 1.0
    d_sb = 2.0 * g0 * g0
    b = (2.0 * g_sum * g_sum / d_sb) ** (1.0 / 3.0) * n ** (1.0 / 3.0)
    b_max = math.ceil(min(3.0 * math.sqrt(n), n / 3.0))
    return float(min(max(b, 1.0), b_max))


def stationary_indices(length: int, mean_block: float, size: int, rng: np.random.Generator) -> np.ndarray:
    """Positions [size, length] of `size` stationary-bootstrap resamples of a series of `length` units."""
    if length <= 0:
        return np.zeros((size, 0), dtype=np.int64)
    if not mean_block >= 1.0:
        raise ValueError("mean_block must be >= 1")
    p = 1.0 / mean_block
    new = rng.random((size, length)) < p                               # a new block starts here
    new[:, 0] = True
    starts = rng.integers(0, length, size=(size, length))              # where a new block would start
    pos = np.arange(length)
    last = np.maximum.accumulate(np.where(new, pos[None, :], 0), axis=1)   # start of the current block
    offset = pos[None, :] - last
    begin = np.take_along_axis(starts, last, axis=1)
    return (begin + offset) % length                                   # circular wrap


@dataclass(frozen=True)
class Scheme:
    """A resampling scheme over n units (construct with the factory functions below)."""

    kind: str
    n: int
    strata: np.ndarray | None = None
    series: tuple[np.ndarray, ...] = ()
    mean_block: tuple[float, ...] = ()
    clusters: np.ndarray | None = None
    cluster_strata: np.ndarray | None = None
    note: str = ""

    def draw(self, size: int, rng: np.random.Generator) -> np.ndarray:
        """Count matrix [size, n] of `size` resamples."""
        n = self.n
        if self.kind == "iid":
            return rng.multinomial(n, np.full(n, 1.0 / n), size=size).astype(np.float64) if n else np.zeros((size, 0))
        if self.kind == "stratified":
            assert self.strata is not None
            out = np.zeros((size, n))
            for s in np.unique(self.strata):
                idx = np.flatnonzero(self.strata == s)
                out[:, idx] = rng.multinomial(idx.size, np.full(idx.size, 1.0 / idx.size), size=size)
            return out
        if self.kind == "stationary":
            out = np.zeros((size, n))
            for idx, block in zip(self.series, self.mean_block, strict=True):
                length = idx.size
                pos = stationary_indices(length, block, size, rng)       # [size, length]
                flat = (np.arange(size)[:, None] * length + pos).ravel()
                cnt = np.bincount(flat, minlength=size * length).reshape(size, length)
                out[:, idx] = cnt
            return out
        if self.kind == "cluster":
            assert self.clusters is not None
            n_c = int(self.clusters.max()) + 1 if n else 0
            if self.cluster_strata is None:
                cc = rng.multinomial(n_c, np.full(n_c, 1.0 / n_c), size=size) if n_c else np.zeros((size, 0))
            else:
                cc = np.zeros((size, n_c))
                for s in np.unique(self.cluster_strata):
                    members = np.flatnonzero(self.cluster_strata == s)
                    cc[:, members] = rng.multinomial(members.size, np.full(members.size, 1.0 / members.size), size=size)
            return np.asarray(cc, dtype=np.float64)[:, self.clusters]
        raise InvariantViolation(f"unknown resampling scheme {self.kind!r}")

    def jackknife(self, max_groups: int, rng: np.random.Generator) -> np.ndarray:
        """Delete-a-group weights [G, n]: row g is 1 everywhere except 0 on group g (module docstring)."""
        n = self.n
        if n == 0:
            return np.ones((0, 0))
        if self.kind == "stationary":
            # Contiguous blocks of each series, the number of blocks per series proportional to its length.
            groups: list[np.ndarray] = []
            total = sum(idx.size for idx in self.series)
            for idx in self.series:
                g = max(1, round(max_groups * idx.size / total))
                groups.extend(part for part in np.array_split(idx, min(g, idx.size)) if part.size)
        elif self.kind == "cluster":
            assert self.clusters is not None
            n_c = int(self.clusters.max()) + 1
            perm = rng.permutation(n_c)
            parts = np.array_split(perm, min(max_groups, n_c))
            groups = [np.flatnonzero(np.isin(self.clusters, part)) for part in parts if part.size]
        else:
            perm = rng.permutation(n)
            groups = [part for part in np.array_split(perm, min(max_groups, n)) if part.size]
        out = np.ones((len(groups), n))
        for g, idx in enumerate(groups):
            out[g, idx] = 0.0
        return out


def iid_scheme(n: int) -> Scheme:
    """Ordinary bootstrap over n units."""
    return Scheme("iid", n)


def stratified_scheme(strata: Any) -> Scheme:
    """Bootstrap inside strata (any labels [n]); every stratum keeps its size."""
    c, _ = codes(strata)
    return Scheme("stratified", c.size, strata=c)


def stationary_scheme(series: Any, time: Any, *, mean_block: float | str = "auto", loss: Any = None) -> Scheme:
    """Stationary bootstrap inside each series (labels [n]), units ordered by `time` within a series.

    mean_block: a fixed mean block length (>= 1) or "auto" (Politis-White on `loss`, required then).
    """
    c, _ = codes(series)
    t = np.asarray(time, dtype=np.float64)
    if t.shape != c.shape:
        raise InvariantViolation("series and time must have the same shape")
    lv = None if loss is None else np.asarray(loss, dtype=np.float64)
    if mean_block == "auto" and lv is None:
        raise InvariantViolation("an automatic block length needs the per-unit loss series")
    idx_list: list[np.ndarray] = []
    blocks: list[float] = []
    for s in np.unique(c):
        idx = np.flatnonzero(c == s)
        idx = idx[np.argsort(t[idx], kind="stable")]                    # time order inside the series
        idx_list.append(idx)
        if mean_block == "auto":
            assert lv is not None
            blocks.append(optimal_block_length(lv[idx]))
        else:
            blocks.append(float(min(max(float(mean_block), 1.0), max(idx.size, 1))))
    return Scheme("stationary", c.size, series=tuple(idx_list), mean_block=tuple(blocks))


def cluster_scheme(clusters: Any, strata: Any = None) -> Scheme:
    """Cluster bootstrap over cluster labels [n]; `strata` gives one stratum label per unit (constant per cluster)."""
    c, _ = codes(clusters)
    cs = None
    if strata is not None:
        sc, _ = codes(strata)
        n_c = int(c.max()) + 1 if c.size else 0
        cs = np.zeros(n_c, dtype=np.int64)
        for k in range(n_c):
            vals = np.unique(sc[c == k])
            if vals.size != 1:
                raise InvariantViolation("cluster strata must be constant within a cluster")
            cs[k] = vals[0]
    return Scheme("cluster", c.size, clusters=c, cluster_strata=cs)


def replicates(stat: Statistic, scheme: Scheme, n_resamples: int, rng: np.random.Generator, *,
               chunk_elements: int = 20_000_000) -> np.ndarray:
    """Evaluate `stat` on `n_resamples` resamples in chunks of weight matrices -> [B, q]."""
    if n_resamples < 1:
        raise ValueError("n_resamples must be >= 1")
    chunk = max(1, min(n_resamples, chunk_elements // max(1, scheme.n)))
    parts: list[np.ndarray] = []
    done = 0
    while done < n_resamples:
        b = min(chunk, n_resamples - done)
        w = scheme.draw(b, rng)
        parts.append(np.asarray(stat(w), dtype=np.float64).reshape(b, -1))
        done += b
    return np.concatenate(parts, axis=0)


def _quantiles(values: np.ndarray, levels: Any) -> np.ndarray:
    # Linear quantiles of finite values; exact order statistics when infinities are present.
    method = "linear" if np.all(np.isfinite(values)) else "inverted_cdf"
    return np.quantile(values, levels, method=method)


def percentile_interval(reps: np.ndarray, confidence: float) -> tuple[float, float, int]:
    """(low, high, number of defined replicates) from replicate quantiles."""
    valid = reps[~np.isnan(reps)]
    if valid.size < 2:
        return math.nan, math.nan, int(valid.size)
    lo, hi = _quantiles(valid, [(1.0 - confidence) / 2.0, (1.0 + confidence) / 2.0])
    return float(lo), float(hi), int(valid.size)


def bca_interval(reps: np.ndarray, estimate: float, jackknife_values: np.ndarray, confidence: float
                 ) -> tuple[float, float, int, str]:
    """(low, high, number of defined replicates, method actually used) of the BCa interval."""
    valid = reps[~np.isnan(reps)]
    if valid.size < 2 or not math.isfinite(estimate):
        lo, hi, nv = percentile_interval(reps, confidence)
        return lo, hi, nv, "percentile"
    share = (np.sum(valid < estimate) + 0.5 * np.sum(valid == estimate)) / valid.size
    if not 0.0 < share < 1.0:
        lo, hi, nv = percentile_interval(reps, confidence)
        return lo, hi, nv, "percentile"
    z0 = float(stats.norm.ppf(share))
    jk = jackknife_values[np.isfinite(jackknife_values)]
    accel = 0.0
    if jk.size >= 2:
        d = jk.mean() - jk
        den = 6.0 * float(np.sum(d * d)) ** 1.5
        accel = float(np.sum(d ** 3)) / den if den > 0 else 0.0
    levels = []
    for z in stats.norm.ppf([(1.0 - confidence) / 2.0, (1.0 + confidence) / 2.0]):
        den = 1.0 - accel * (z0 + z)
        if den <= 0:
            lo, hi, nv = percentile_interval(reps, confidence)
            return lo, hi, nv, "percentile"
        levels.append(float(stats.norm.cdf(z0 + (z0 + z) / den)))
    if not 0.0 < levels[0] < levels[1] < 1.0:
        lo, hi, nv = percentile_interval(reps, confidence)
        return lo, hi, nv, "percentile"
    lo, hi = _quantiles(valid, levels)
    return float(lo), float(hi), int(valid.size), "bca"


@dataclass(frozen=True)
class Estimate:
    """Point estimates and confidence intervals of q statistics computed on the same resamples."""

    names: tuple[str, ...]
    value: np.ndarray
    low: np.ndarray
    high: np.ndarray
    method: tuple[str, ...]
    confidence: float
    n_resamples: int
    n_valid: np.ndarray
    scheme: str
    replicates: np.ndarray = field(repr=False)

    def get(self, name: str) -> tuple[float, float, float]:
        """(value, low, high) of one statistic."""
        i = self.names.index(name)
        return float(self.value[i]), float(self.low[i]), float(self.high[i])


@dataclass(frozen=True)
class BootstrapSettings:
    """How intervals are computed (mirrors conf/evaluation/evaluation.yaml, section bootstrap)."""

    n_resamples: int = 2000
    confidence: float = 0.95
    interval: str = "bca"
    jackknife_groups: int = 100
    chunk_elements: int = 20_000_000

    def __post_init__(self) -> None:
        if self.n_resamples < 1:
            raise ValueError("n_resamples must be >= 1")
        if not 0.0 < self.confidence < 1.0:
            raise ValueError("confidence must lie in (0, 1)")
        if self.interval not in ("percentile", "bca"):
            raise ValueError("interval must be 'percentile' or 'bca'")
        if self.jackknife_groups < 2:
            raise ValueError("jackknife_groups must be >= 2")


def estimate(stat: Statistic, names: Sequence[str], scheme: Scheme, settings: BootstrapSettings,
             rng: np.random.Generator) -> Estimate:
    """Point values (unit weights) and bootstrap intervals of the statistics computed by `stat`."""
    q = len(names)
    point = np.asarray(stat(np.ones((1, scheme.n))), dtype=np.float64).reshape(1, -1)[0]
    if point.size != q:
        raise InvariantViolation(f"statistic returned {point.size} values for {q} names")
    reps = replicates(stat, scheme, settings.n_resamples, rng, chunk_elements=settings.chunk_elements)
    jack = None
    if settings.interval == "bca":
        jw = scheme.jackknife(settings.jackknife_groups, rng)
        jack = np.asarray(stat(jw), dtype=np.float64).reshape(jw.shape[0], -1) if jw.size else np.zeros((0, q))
    low, high, valid, methods = np.full(q, np.nan), np.full(q, np.nan), np.zeros(q, dtype=np.int64), []
    for i in range(q):
        if settings.interval == "bca":
            assert jack is not None
            lo, hi, nv, used = bca_interval(reps[:, i], float(point[i]), jack[:, i], settings.confidence)
        else:
            lo, hi, nv = percentile_interval(reps[:, i], settings.confidence)
            used = "percentile"
        low[i], high[i], valid[i] = lo, hi, nv
        methods.append(used)
    return Estimate(tuple(names), point, low, high, tuple(methods), settings.confidence, settings.n_resamples,
                    valid, scheme.kind, reps)
