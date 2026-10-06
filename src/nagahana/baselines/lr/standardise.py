"""Location and scale of every design column, fitted on training rows only and stored with the model.

The raw design (features.py) is float32 with NaN marking a value that does not exist (a field that
was not supplied or not observable, a share over an empty window). Standardisation maps a column x to

    z = (x - location) / scale,        z = 0 where x is NaN,

so an absent value contributes w * 0 to a linear predictor (the neutral fill of the missing-indicator
method) while an explicit indicator column carries the absence itself (D-41; AS-502). Ridge-type
penalties are not equivariant under rescaling of the inputs, which is why every column is put on a
common scale before fitting (Hastie, Tibshirani and Friedman, The Elements of Statistical Learning,
2nd ed., Springer 2009, Section 3.4.1).

Two estimators (AS-504)

    zscore    location = mean, scale = population standard deviation (ddof = 0)
    robust    location = median, scale = IQR / 1.3489795 ("iqr") or 1.4826 * median |x - median|
              ("mad"); both are consistent for the standard deviation of a normal distribution

Indicator columns (status indicators, category indicators, bits) always use zscore: the median and the
IQR of 0/1 data are degenerate. Numeric columns use the configured estimator. When a robust scale is
zero (a zero-inflated count, for example) the column falls back to its standard deviation, and a column
whose valid training values are all equal (or that has none) is constant: scale 1, flagged, and dropped
by the models when `FeatureConfig.drop_constant` is on (AS-505).

Exactness and out-of-core fitting

All statistics are computed from a re-iterable sequence of float32 chunks, so a design that does not fit
in memory is fitted in passes over its chunks and an in-memory design is the one-chunk case of the same
code. Means and variances use the pairwise update of Chan, Golub and LeVeque ("Algorithms for computing
the sample variance: analysis and recommendations", The American Statistician 37(3), 1983). Quantiles
are exact order statistics found by a two-pass radix selection on the order-preserving 32-bit keys of
the float32 values: pass one counts the high 16 bits of every key, which locates the bucket holding each
requested rank; pass two counts the low 16 bits inside those buckets, which gives the exact key. The
quantile of order p is then the linear interpolation between the order statistics x_(floor(h)) and
x_(floor(h)+1) with h = (n - 1) p (Hyndman and Fan, "Sample quantiles in statistical packages", The
American Statistician 50(4), 1996, definition 7; NumPy's default).
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass

import numpy as np

from nagahana.core.errors import InvariantViolation

from .config import StandardiserConfig

ChunkFactory = Callable[[], Iterator[np.ndarray]]

#: Scale of the IQR of a standard normal distribution: Phi^-1(0.75) - Phi^-1(0.25).
IQR_NORMAL = 1.3489795003921634
#: 1 / Phi^-1(0.75): the MAD of a standard normal distribution is 1 / 1.4826.
MAD_NORMAL = 1.482602218505602
#: Kinds of a column's estimator.
KIND_ZSCORE, KIND_ROBUST_IQR, KIND_ROBUST_MAD = 0, 1, 2
_TINY = 1e-12
_BUCKETS = 1 << 16


def order_keys(x: np.ndarray) -> np.ndarray:
    """Order-preserving uint32 keys of float32 values (NaN excluded by the caller).

    For a non-negative float the sign bit is set; for a negative float every bit is inverted, so key
    order equals numeric order (with -0.0 just below +0.0).
    """
    if x.dtype != np.float32:
        raise InvariantViolation("order keys are defined on float32 values")
    b = x.view(np.uint32)
    neg = (b >> np.uint32(31)).astype(bool)
    return np.where(neg, ~b, b | np.uint32(0x80000000)).astype(np.uint32)


def from_order_keys(k: np.ndarray) -> np.ndarray:
    """Inverse of `order_keys`."""
    k = np.asarray(k, dtype=np.uint32)
    pos = (k >> np.uint32(31)).astype(bool)
    b = np.where(pos, k & np.uint32(0x7FFFFFFF), ~k).astype(np.uint32)
    return b.view(np.float32)


def exact_order_statistics(chunks: ChunkFactory, columns: np.ndarray, ranks: np.ndarray, *,
                           transform: Callable[[np.ndarray, int], np.ndarray] | None = None,
                           group_size: int = 64) -> np.ndarray:
    """Exact order statistics of selected columns over all chunks (module docstring).

    chunks: returns a fresh iterator of float32 [rows, D] arrays each call.
    columns: int [G] the columns to select from. ranks: int64 [G, R], 0-based ranks among the valid
    (non-NaN) values of each column, -1 where unused. transform(values, column) may map a column's
    float32 values to other float32 values first (used for absolute deviations from the median).
    Returns float64 [G, R] (NaN where the rank is unused).
    """
    columns = np.asarray(columns, dtype=np.int64)
    ranks = np.asarray(ranks, dtype=np.int64)
    if ranks.ndim != 2 or ranks.shape[0] != columns.shape[0]:
        raise InvariantViolation("ranks must be [len(columns), R]")
    out = np.full(ranks.shape, np.nan, dtype=np.float64)
    for g0 in range(0, columns.shape[0], group_size):
        cols = columns[g0:g0 + group_size]
        rk = ranks[g0:g0 + group_size]
        out[g0:g0 + group_size] = _select_group(chunks, cols, rk, transform)
    return out


def _column_values(x: np.ndarray, col: int, transform: Callable[[np.ndarray, int], np.ndarray] | None) -> np.ndarray:
    # One column's valid float32 values of a chunk, optionally transformed.
    v = x[:, col]
    v = v[~np.isnan(v)]
    if transform is not None:
        v = transform(v, col)
        if v.dtype != np.float32:
            raise InvariantViolation("transform must return float32 values")
    return v


def _select_group(chunks: ChunkFactory, cols: np.ndarray, ranks: np.ndarray,
                  transform: Callable[[np.ndarray, int], np.ndarray] | None) -> np.ndarray:
    g, r = ranks.shape
    # Pass 1: counts of the high 16 bits of every key, per column.   [G, 65536]
    high = np.zeros((g, _BUCKETS), dtype=np.int64)
    for x in chunks():
        for i, c in enumerate(cols.tolist()):
            v = _column_values(x, c, transform)
            if v.size:
                high[i] += np.bincount((order_keys(v) >> np.uint32(16)).astype(np.int64), minlength=_BUCKETS)
    cum = np.cumsum(high, axis=1)                                            # [G, 65536]
    total = cum[:, -1]
    if np.any((ranks >= 0) & (ranks >= total[:, None])):
        raise InvariantViolation("a requested rank exceeds the number of valid values")
    # Bucket of each rank and its rank inside the bucket.
    bucket = np.full((g, r), -1, dtype=np.int64)
    inner = np.full((g, r), -1, dtype=np.int64)
    for i in range(g):
        ok = ranks[i] >= 0
        if ok.any():
            b = np.searchsorted(cum[i], ranks[i][ok], side="right")
            before = np.where(b > 0, cum[i][np.maximum(b - 1, 0)], 0)
            bucket[i, ok] = b
            inner[i, ok] = ranks[i][ok] - before
    # Pass 2: counts of the low 16 bits inside the target buckets, per (column, bucket).
    targets: list[np.ndarray] = [np.unique(bucket[i][bucket[i] >= 0]) for i in range(g)]
    low = [np.zeros((t.size, _BUCKETS), dtype=np.int64) for t in targets]
    for x in chunks():
        for i, c in enumerate(cols.tolist()):
            if targets[i].size == 0:
                continue
            v = _column_values(x, c, transform)
            if not v.size:
                continue
            k = order_keys(v)
            hb = (k >> np.uint32(16)).astype(np.int64)
            hit = np.isin(hb, targets[i])
            if hit.any():
                t_idx = np.searchsorted(targets[i], hb[hit])
                lb = (k[hit] & np.uint32(0xFFFF)).astype(np.int64)
                low[i] += np.bincount(t_idx * _BUCKETS + lb, minlength=targets[i].size * _BUCKETS).reshape(-1, _BUCKETS)
    res = np.full((g, r), np.nan, dtype=np.float64)
    for i in range(g):
        for j in range(r):
            if bucket[i, j] < 0:
                continue
            t = int(np.searchsorted(targets[i], bucket[i, j]))
            cl = np.cumsum(low[i][t])
            lo_bits = int(np.searchsorted(cl, inner[i, j], side="right"))
            key = np.uint32((int(bucket[i, j]) << 16) | lo_bits)
            res[i, j] = float(from_order_keys(np.asarray([key]))[0])
    return res


def _quantile_ranks(n: np.ndarray, p: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For n valid values: lower rank floor(h), upper rank min(floor(h) + 1, n - 1) and weight h - floor(h)."""
    h = (n.astype(np.float64) - 1.0) * p
    lo = np.floor(h).astype(np.int64)
    hi = np.minimum(lo + 1, n - 1)
    return lo, hi, h - lo


def column_moments(chunks: ChunkFactory, d: int) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Rows seen, and per column over valid values: count, mean, M2 = sum (x - mean)^2, min, max (Chan et al. 1983)."""
    n = np.zeros(d, dtype=np.int64)
    mean = np.zeros(d, dtype=np.float64)
    m2 = np.zeros(d, dtype=np.float64)
    vmin = np.full(d, np.inf)
    vmax = np.full(d, -np.inf)
    rows = 0
    for x in chunks():
        if x.ndim != 2 or x.shape[1] != d or x.dtype != np.float32:
            raise InvariantViolation(f"chunks must be float32 [rows, {d}], got {x.dtype} {x.shape}")
        rows += x.shape[0]
        xv = x.astype(np.float64)
        valid = ~np.isnan(xv)
        nb = valid.sum(axis=0).astype(np.int64)
        has = nb > 0
        s = np.where(valid, xv, 0.0).sum(axis=0)
        mb = np.divide(s, nb, out=np.zeros(d), where=has)
        m2b = np.where(valid, (xv - mb) ** 2, 0.0).sum(axis=0)
        # pairwise combination of (n_a, mean_a, M2_a) with (n_b, mean_b, M2_b)
        tot = n + nb
        delta = mb - mean
        frac = np.divide(nb, tot, out=np.zeros(d), where=tot > 0)
        mean = np.where(has, mean + delta * frac, mean)
        m2 = np.where(has, m2 + m2b + delta ** 2 * n * frac, m2)
        n = tot
        vmin = np.minimum(vmin, np.where(valid, xv, np.inf).min(axis=0))
        vmax = np.maximum(vmax, np.where(valid, xv, -np.inf).max(axis=0))
    if rows == 0:
        raise InvariantViolation("a standardiser cannot be fitted on zero rows")
    return rows, n, mean, m2, vmin, vmax


@dataclass
class Standardiser:
    """Fitted per-column location and scale (module docstring).

    location, scale: float64 [D]; kind: int8 [D] (KIND_*); constant: bool [D]; fallback: bool [D]
    (a robust scale of zero replaced by the standard deviation); n_valid: int64 [D]; n_rows: training
    rows seen; mean: float64 [D] mean of the valid training values.
    """

    location: np.ndarray
    scale: np.ndarray
    kind: np.ndarray
    constant: np.ndarray
    fallback: np.ndarray
    n_valid: np.ndarray
    n_rows: int
    mean: np.ndarray

    def standardised_mean(self) -> np.ndarray:
        """Mean of each standardised column over the training rows (absent values count as 0)."""
        frac = self.n_valid / max(self.n_rows, 1)
        return frac * (self.mean - self.location) / self.scale

    @property
    def n_cols(self) -> int:
        return int(self.location.shape[0])

    @classmethod
    def fit(cls, chunks: ChunkFactory, binary: np.ndarray, cfg: StandardiserConfig) -> Standardiser:
        """Fit on the rows the chunks yield (training rows only; the caller selects them)."""
        binary = np.asarray(binary, dtype=bool)
        d = int(binary.shape[0])
        rows, n, mean, m2, vmin, vmax = column_moments(chunks, d)
        std = np.sqrt(np.divide(m2, n, out=np.zeros(d), where=n > 0))
        constant = (n == 0) | (vmax <= vmin)
        kind = np.full(d, KIND_ZSCORE, dtype=np.int8)
        location = mean.copy()
        scale = std.copy()
        fallback = np.zeros(d, dtype=bool)
        robust = (~binary) & (~constant) & (cfg.numeric == "robust")
        if robust.any():
            cols = np.nonzero(robust)[0]
            nv = n[cols]
            lo50, hi50, f50 = _quantile_ranks(nv, 0.5)
            ranks = [lo50, hi50]
            if cfg.robust_scale == "iqr":
                lo25, hi25, f25 = _quantile_ranks(nv, 0.25)
                lo75, hi75, f75 = _quantile_ranks(nv, 0.75)
                ranks += [lo25, hi25, lo75, hi75]
            st = exact_order_statistics(chunks, cols, np.stack(ranks, axis=1))
            med = st[:, 0] + f50 * (st[:, 1] - st[:, 0])
            if cfg.robust_scale == "iqr":
                q25 = st[:, 2] + f25 * (st[:, 3] - st[:, 2])
                q75 = st[:, 4] + f75 * (st[:, 5] - st[:, 4])
                rscale = (q75 - q25) / IQR_NORMAL
                k = KIND_ROBUST_IQR
            else:
                med32 = {int(c): float(m) for c, m in zip(cols.tolist(), med.tolist(), strict=True)}

                def absdev(v: np.ndarray, c: int) -> np.ndarray:
                    # |x - median| rounded to float32 (the same arithmetic on every path)
                    return np.abs(v.astype(np.float64) - med32[c]).astype(np.float32)

                st2 = exact_order_statistics(chunks, cols, np.stack([lo50, hi50], axis=1), transform=absdev)
                rscale = MAD_NORMAL * (st2[:, 0] + f50 * (st2[:, 1] - st2[:, 0]))
                k = KIND_ROBUST_MAD
            location[cols] = med
            fb = rscale <= _TINY
            scale[cols] = np.where(fb, std[cols], rscale)
            fallback[cols] = fb
            kind[cols] = k
        scale = np.where(constant | (scale <= _TINY), 1.0, scale)
        location = np.where(n == 0, 0.0, location)
        return cls(location=location, scale=scale, kind=kind, constant=constant, fallback=fallback,
                   n_valid=n, n_rows=rows, mean=np.where(n > 0, mean, 0.0))

    def transform(self, x: np.ndarray, columns: np.ndarray | None = None) -> np.ndarray:
        """float32 raw design [n, D] -> float64 standardised [n, D'] (D' = len(columns)); NaN -> 0."""
        if x.ndim != 2 or x.shape[1] != self.n_cols:
            raise InvariantViolation(f"expected [n, {self.n_cols}] raw design, got {x.shape}")
        cols = np.arange(self.n_cols) if columns is None else np.asarray(columns, dtype=np.int64)
        z = (x[:, cols].astype(np.float64) - self.location[cols]) / self.scale[cols]
        return np.where(np.isnan(z), 0.0, z)

    def state(self) -> dict[str, np.ndarray]:
        """Arrays for serialisation."""
        return {"location": self.location, "scale": self.scale, "kind": self.kind, "constant": self.constant,
                "fallback": self.fallback, "n_valid": self.n_valid, "n_rows": np.asarray(self.n_rows, dtype=np.int64),
                "mean": self.mean}

    @classmethod
    def from_state(cls, s: dict[str, np.ndarray]) -> Standardiser:
        return cls(location=np.asarray(s["location"], dtype=np.float64), scale=np.asarray(s["scale"], dtype=np.float64),
                   kind=np.asarray(s["kind"], dtype=np.int8), constant=np.asarray(s["constant"], dtype=bool),
                   fallback=np.asarray(s["fallback"], dtype=bool), n_valid=np.asarray(s["n_valid"], dtype=np.int64),
                   n_rows=int(np.asarray(s["n_rows"])), mean=np.asarray(s["mean"], dtype=np.float64))


__all__ = ["IQR_NORMAL", "MAD_NORMAL", "Standardiser", "column_moments", "exact_order_statistics", "from_order_keys",
           "order_keys"]
