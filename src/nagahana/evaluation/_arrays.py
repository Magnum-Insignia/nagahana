"""Array validation and the unit-weight convention shared by every evaluation metric.

Weights

Every metric in this package accepts optional unit weights. A weight vector w of shape [n] multiplies
each unit's contribution to the metric. A weight matrix W of shape [B, n] evaluates the metric once per
row and returns B values. The bootstrap uses this form: row b holds the number of times each unit was
drawn in resample b, so the weighted metric of that row equals the metric of the explicitly resampled
data exactly (a unit drawn c times contributes c copies). A leave-one-group-out jackknife is the same
mechanism with zero weights on the deleted group. Unit weights of 1 give the ordinary estimate.

Undefined values

A ratio whose denominator is zero is undefined and is returned as NaN, never as 0: a detector that
never fires has an undefined precision, not a precision of 0.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from nagahana.core.errors import InvariantViolation


def as_float(name: str, x: Any, ndim: int | None = None, *, finite: bool = True) -> np.ndarray:
    """Convert to a float64 array, checking the rank and (optionally) finiteness."""
    a = np.asarray(x, dtype=np.float64)
    if ndim is not None and a.ndim != ndim:
        raise InvariantViolation(f"{name} must have {ndim} dimensions, got shape {a.shape}")
    if finite and not np.all(np.isfinite(a)):
        raise InvariantViolation(f"{name} contains NaN or infinite values")
    return a


def as_binary(name: str, y: Any) -> np.ndarray:
    """Convert to an int64 vector of 0/1 labels (no unknown labels allowed here)."""
    a = np.asarray(y)
    if a.ndim != 1:
        raise InvariantViolation(f"{name} must be one-dimensional, got shape {a.shape}")
    if a.dtype == bool:
        return a.astype(np.int64)
    out = a.astype(np.int64)
    if not np.array_equal(out, a) or not np.isin(out, (0, 1)).all():
        raise InvariantViolation(f"{name} must hold only 0 and 1")
    return out


def as_probability(name: str, p: Any, ndim: int | None = None, *, tol: float = 1e-9) -> np.ndarray:
    """Convert to float64 probabilities in [0, 1] (values within tol of the bounds are clipped)."""
    a = as_float(name, p, ndim)
    if a.size and (a.min() < -tol or a.max() > 1.0 + tol):
        raise InvariantViolation(f"{name} must lie in [0, 1]; got range [{a.min()}, {a.max()}]")
    return np.clip(a, 0.0, 1.0)


def weight_matrix(weights: Any, n: int) -> tuple[np.ndarray, bool]:
    """Return weights as a float64 matrix [B, n] and whether the input was batched.

    None gives a single row of ones. A vector [n] gives one row. A matrix [B, n] is returned as is.
    Weights must be finite and non-negative.
    """
    if weights is None:
        return np.ones((1, n), dtype=np.float64), False
    w = np.asarray(weights, dtype=np.float64)
    batched = w.ndim == 2
    if w.ndim == 1:
        w = w[None, :]
    if w.ndim != 2 or w.shape[1] != n:
        raise InvariantViolation(f"weights must be [n] or [B, n] with n = {n}, got shape {np.shape(weights)}")
    if not np.all(np.isfinite(w)) or np.any(w < 0):
        raise InvariantViolation("weights must be finite and non-negative")
    return w, batched


def unbatch(values: np.ndarray, batched: bool) -> Any:
    """Return values [B, ...] as is when batched, or the single row (a float for a scalar) otherwise."""
    if batched:
        return values
    row = values[0]
    return float(row) if np.ndim(row) == 0 else row


def safe_ratio(num: Any, den: Any) -> np.ndarray:
    """Elementwise num / den with NaN where den == 0 (undefined, never 0)."""
    num_a = np.asarray(num, dtype=np.float64)
    den_a = np.asarray(den, dtype=np.float64)
    out = np.full(np.broadcast(num_a, den_a).shape, np.nan)
    np.divide(num_a, den_a, out=out, where=den_a != 0)
    return out


def descending_groups(score: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sort by descending score and locate groups of tied scores.

    Returns (order, starts, values): `order` sorts the units by descending score (stable), `starts`
    are the first sorted positions of each group of equal scores, and `values` are the group scores.
    Every threshold-based curve in this package moves through these groups, so tied scores are always
    crossed together (a threshold cannot split a tie).
    """
    order = np.argsort(-score, kind="stable")
    s = score[order]
    if s.size == 0:
        return order, np.zeros(0, dtype=np.int64), np.zeros(0)
    starts = np.flatnonzero(np.r_[True, s[1:] != s[:-1]])
    return order, starts, s[starts]


def group_sums(w_sorted: np.ndarray, starts: np.ndarray) -> np.ndarray:
    """Sum the columns of w_sorted [B, n] over consecutive groups beginning at `starts` -> [B, G]."""
    if starts.size == 0:
        return np.zeros((w_sorted.shape[0], 0))
    return np.add.reduceat(w_sorted, starts, axis=1)


def codes(values: Any) -> tuple[np.ndarray, np.ndarray]:
    """Integer codes of arbitrary labels: (codes [n], unique labels), labels in sorted order."""
    arr = np.asarray(values)
    uniq, inv = np.unique(arr, return_inverse=True)
    return inv.astype(np.int64).reshape(-1), uniq


def rowwise_searchsorted(sorted_rows: np.ndarray, queries: np.ndarray, *, side: str) -> np.ndarray:
    """numpy.searchsorted applied row by row: sorted_rows [B, n] (each row non-decreasing), queries [B, m].

    The rows are shifted apart by multiples of a span larger than every row's range, which turns the
    B searches into one search on a single sorted array; the result is [B, m] positions in 0 .. n.
    """
    a = np.asarray(sorted_rows, dtype=np.float64)
    q = np.asarray(queries, dtype=np.float64)
    b, n = a.shape
    if b == 0 or n == 0:
        return np.zeros(q.shape, dtype=np.int64)
    finite_a = a[np.isfinite(a)]
    finite_q = q[np.isfinite(q)]
    lo = min(finite_a.min() if finite_a.size else 0.0, finite_q.min() if finite_q.size else 0.0)
    hi = max(finite_a.max() if finite_a.size else 0.0, finite_q.max() if finite_q.size else 0.0)
    span = (hi - lo) + 1.0
    shift = span * np.arange(b, dtype=np.float64)[:, None]
    # Infinite entries keep their order inside a row by mapping them just outside the row's finite range.
    a_s = np.where(np.isposinf(a), hi + 0.5, np.where(np.isneginf(a), lo - 0.25, a)) - lo + shift
    q_s = np.where(np.isposinf(q), hi + 0.5, np.where(np.isneginf(q), lo - 0.25, q)) - lo + shift
    pos = np.searchsorted(a_s.ravel(), q_s.ravel(), side=side).reshape(q.shape)
    return pos - n * np.arange(b, dtype=np.int64)[:, None]


def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> np.ndarray:
    """Quantile q of values [n] under each row of weights [B, n] (linear interpolation, type 7).

    For integer weights (bootstrap counts) the result equals numpy.quantile of the expanded sample
    in which unit i is repeated w_i times: with N = sum_i w_i expanded items, the quantile sits at the
    0-based expanded position h = (N - 1) q and interpolates linearly between positions floor(h) and
    ceil(h). Non-integer weights use the same formula with N = sum_i w_i. Values may contain -inf or
    +inf (for example lead times of instances that were never alerted); an interpolation towards an
    infinite end is that infinity, and between opposite infinities it is undefined (NaN). Rows with
    zero total weight give NaN.
    """
    if not 0.0 <= q <= 1.0:
        raise ValueError("q must lie in [0, 1]")
    v = np.asarray(values, dtype=np.float64)
    w, _ = weight_matrix(weights, v.size)
    out = np.full(w.shape[0], np.nan)
    if v.size == 0:
        return out
    order = np.argsort(v, kind="stable")
    vs = v[order]
    cum = np.cumsum(w[:, order], axis=1)                                # [B, n] expanded positions covered
    total = cum[:, -1]
    h = (np.maximum(total, 1.0) - 1.0) * q
    lo_pos, hi_pos = np.floor(h), np.ceil(h)
    # The unit holding expanded position p is the first sorted unit with cum > p.
    i_lo = np.minimum(rowwise_searchsorted(cum, lo_pos[:, None], side="right")[:, 0], v.size - 1)
    i_hi = np.minimum(rowwise_searchsorted(cum, hi_pos[:, None], side="right")[:, 0], v.size - 1)
    a, c = vs[i_lo], vs[i_hi]
    frac = h - lo_pos
    with np.errstate(invalid="ignore"):
        interp = a + frac * (c - a)
    either_inf = np.isinf(a) | np.isinf(c)
    inf_value = np.where(np.isinf(a) & np.isinf(c) & (a != c), np.nan, np.where(np.isinf(a), a, c))
    res = np.where((a == c) | (frac == 0.0), a, np.where(either_inf, inf_value, interp))
    return np.where(total > 0, res, out)
