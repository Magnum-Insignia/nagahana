"""Time-to-event evaluation with right-censoring: concordance, time-dependent AUC, IPCW Brier, D-calibration.

Data. For trigger i: an observed time T_i (to the event, or to censoring), an event indicator d_i, a
risk score r_i (higher means sooner) and a predicted survival curve S_i(t) on a time grid.

Ties and censoring. At a tied time, events precede censorings (the Kaplan-Meier convention), and a
unit censored at time c is known to be event-free through c. This is exactly the rule of
ForecastPredictions.outcome: a trigger with no event and s observed steps has a known outcome "no
infiltration within k steps" for every k <= s. Consequently a unit is a survivor at t when T_i > t, or
when T_i = t and it was censored.

Concordance (Harrell et al., JAMA 247:2543-2546, 1982; Statistics in Medicine 15:361-387, 1996)

    C = sum_{i,j} d_i 1[T_i < T_j] (1[r_i > r_j] + 1/2 1[r_i = r_j]) / sum_{i,j} d_i 1[T_i < T_j]

(the thesis equation with ties in risk counted as one half). Optionally a pair with T_i = T_j, d_i = 1
and d_j = 0 is comparable as well (the censored unit outlived the event). Uno et al. (Statistics in
Medicine 30:1105-1117, 2011, doi:10.1002/sim.4154) weight each comparable pair by G(T_i-)^-2, with G
the Kaplan-Meier estimate of the censoring survival function, and restrict T_i < tau; the result
estimates a concordance that does not depend on the censoring distribution.

Algorithm. Sorting by time and sweeping from the latest time down, every unit with a strictly later
time has already been inserted into a Fenwick tree (binary indexed tree; Fenwick, Software: Practice
and Experience 24:327-336, 1994) indexed by risk rank, so the weighted counts of later units with
lower and equal risk are two prefix sums of O(log n) each. Units are processed in blocks of whole
time groups: pairs inside a block are counted directly (a small dense comparison), pairs across
blocks through the tree. The total work is O(n log n + n s) for a block size s, with the Python loop
over blocks only. The tree stores one column per weight row, so B bootstrap resamples are counted in
the same sweep. A brute-force O(n^2) count is the reference in the tests.

Kaplan-Meier (Kaplan and Meier, JASA 53:457-481, 1958). S(t) = prod_{t_k <= t} (1 - e_k / n_k) with
n_k the weight at risk (T >= t_k) and e_k the event weight at t_k. The censoring distribution G uses
the censorings as events; because events precede censorings at a tie, the weight at risk of
censoring at t_k excludes the events at t_k. G(t-) is the left limit.

IPCW Brier score (Graf, Schmoor, Sauerbrei and Schumacher, Statistics in Medicine 18:2529-2545, 1999;
Gerds and Schumacher, Biometrical Journal 48:1029-1040, 2006)

    BS(t) = (1/n) sum_i [ 1(T_i <= t, d_i = 1) S_i(t)^2 / G(T_i-) + 1(survivor at t) (1 - S_i(t))^2 / G(t-) ],

and the integrated Brier score IBS = (1/t_max) integral_0^{t_max} BS(t) dt (the thesis equation),
computed by the trapezoidal rule over 0 and the evaluation times. Under the convention above, a
survivor at t is observed with probability P(C >= t) = G(t-), an event at T_i with probability
G(T_i-). Times with G(t-) = 0 cannot be reweighted and give NaN.

Time-dependent AUC (cumulative cases, dynamic controls; Heagerty, Lumley and Pepe, Biometrics
56:337-344, 2000) with the IPCW estimator (Uno, Cai, Tian and Wei, JASA 102:527-537, 2007; Hung and
Chiang, Canadian Journal of Statistics 38:8-26, 2010):

    AUC(t) = sum_{i,j} w_i 1(T_i <= t, d_i = 1) 1(j survivor at t) psi(m_i(t), m_j(t))
             / (sum_i w_i 1(T_i <= t, d_i = 1) * sum_j 1(j survivor at t)),   w_i = 1 / G(T_i-),

with the marker m_i(t) = 1 - S_i(t) (or the risk score). The summary over times is the mean AUC
weighted by the Kaplan-Meier decrements of the event distribution, sum_k AUC(t_k) (S(t_{k-1}) - S(t_k))
/ (1 - S(t_K)) with S(t_0) = 1 (Lambert and Chevret, Statistical Methods in Medical Research
25:2088-2102, 2016).

D-calibration (Haider, Hoehn, Davis and Greiner, JMLR 21(85):1-63, 2020). If the curves are calibrated,
S_i(T_i) is uniform on [0, 1] for uncensored units. A censored unit with s = S_i(c_i) is known to have
S_i(T_i) < s, so it spreads one unit of mass uniformly over [0, s]. With B equal bins of [0, 1] and n
units, Pearson's statistic sum_b (O_b - n/B)^2 / (n/B) is compared with chi-square on B - 1 degrees
of freedom. S_i(T_i) is read from the curve by linear interpolation (a constant event density inside a
grid step), so that S_i(T_i) is continuous as the uniformity argument requires.

Restricted mean survival time. RMST(tau) = integral_0^tau S(t) dt (Royston and Parmar, BMC Medical
Research Methodology 13:152, 2013); -RMST is the risk score of a curve when no separate score is given.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import stats

from nagahana.core.errors import InvariantViolation
from nagahana.evaluation._arrays import as_float, safe_ratio, unbatch, weight_matrix
from nagahana.evaluation.calibration import bin_sums
from nagahana.evaluation.predictions import ForecastPredictions, TimeToEventPredictions
from nagahana.evaluation.ranking import auroc


class FenwickTree:
    """Binary indexed tree over positions 1 .. size whose entries are rows of `width` values.

    `add` and `prefix` take arrays of positions and work on all of them at once; each takes O(log size)
    vectorised steps. Row r of the tree is a vector, so B weight rows (bootstrap resamples) share one
    tree.
    """

    def __init__(self, size: int, width: int) -> None:
        if size < 0 or width < 1:
            raise ValueError("size must be >= 0 and width >= 1")
        self.size = size
        self.tree = np.zeros((size + 1, width))

    def add(self, positions: np.ndarray, values: np.ndarray) -> None:
        """Add values [k, width] at positions [k] (1-based; repeated positions accumulate)."""
        pos = np.asarray(positions, dtype=np.int64).copy()
        vals = np.asarray(values, dtype=np.float64)
        if pos.size and (pos.min() < 1 or pos.max() > self.size):
            raise IndexError("Fenwick positions must lie in 1 .. size")
        while pos.size:
            np.add.at(self.tree, pos, vals)
            pos = pos + (pos & -pos)                                   # next node covering this position
            keep = pos <= self.size
            pos, vals = pos[keep], vals[keep]

    def prefix(self, positions: np.ndarray) -> np.ndarray:
        """Sums over positions 1 .. p for each p in positions [k] -> [k, width] (p = 0 gives 0)."""
        pos = np.asarray(positions, dtype=np.int64).copy()
        out = np.zeros((pos.size, self.tree.shape[1]))
        active = pos > 0
        while active.any():
            out[active] += self.tree[pos[active]]
            pos[active] -= pos[active] & -pos[active]                  # drop the lowest set bit
            active = pos > 0
        return out


def _validate_tte(time: Any, event: Any, risk: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t = as_float("time", time, 1)
    d = np.asarray(event).astype(bool)
    r = as_float("risk", risk, 1)
    if d.shape != t.shape or r.shape != t.shape:
        raise InvariantViolation("time, event and risk must have the same shape")
    return t, d, r


def concordance_sums(time: Any, event: Any, risk: Any, weights: Any = None, *, multiplier: Any = None,
                     tied_times: str = "exclude", block: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """Weighted concordant and comparable pair sums [B] (module docstring, Algorithm).

    numerator   = sum_i m_i w_i d_i sum_{j comparable to i} w_j (1[r_j < r_i] + 1/2 1[r_j = r_i])
    denominator = sum_i m_i w_i d_i sum_{j comparable to i} w_j
    with comparable meaning T_j > T_i (and T_j = T_i, d_j = 0 when tied_times = "censored_later"),
    m_i the optional per-unit multiplier (Uno's weights; [n], or [B, n] for one multiplier row per weight
    row), w the unit weights ([n] or [B, n]).
    """
    if tied_times not in ("exclude", "censored_later"):
        raise ValueError("tied_times must be 'exclude' or 'censored_later'")
    t, d, r = _validate_tte(time, event, risk)
    n = t.size
    w, _ = weight_matrix(weights, n)
    b = w.shape[0]
    mult = np.ones((1, n)) if multiplier is None else np.atleast_2d(as_float("multiplier", multiplier))
    if mult.shape[1] != n or mult.shape[0] not in (1, b):
        raise InvariantViolation("multiplier must be [n] or [B, n]")
    m = mult * d[None, :]                                              # only events start comparable pairs [1 or B, n]
    num, den = np.zeros(b), np.zeros(b)
    if n == 0:
        return num, den
    uniq = np.unique(r)
    rank = np.searchsorted(uniq, r) + 1                                # dense risk rank 1 .. R
    order = np.argsort(-t, kind="stable")                              # latest time first
    ts = t[order]
    group_starts = np.flatnonzero(np.r_[True, ts[1:] != ts[:-1]])
    # Blocks of whole time groups with at least `block` units (a long tie group is one block).
    cuts = [0]
    for g in group_starts[1:]:
        if g - cuts[-1] >= block:
            cuts.append(int(g))
    cuts.append(n)
    tree = FenwickTree(int(uniq.size), b)
    inserted = np.zeros(b)                                             # weight of all later units
    for a, z in zip(cuts[:-1], cuts[1:], strict=True):
        idx = order[a:z]
        q = idx[d[idx]]
        if q.size:
            wq = w[:, q].T * m[:, q].T                                 # [q, B] i-side weights
            less = tree.prefix(rank[q] - 1)                            # later units with lower risk [q, B]
            leq = tree.prefix(rank[q])                                 # later units with risk <= r_i
            num += (wq * (less + 0.5 * (leq - less))).sum(axis=0)
            den += (wq * inserted[None, :]).sum(axis=0)
            # Pairs inside the block: j later than i (strictly), or tied and censored when allowed.
            later = t[idx][None, :] > t[q][:, None]                    # [q, s]
            if tied_times == "censored_later":
                later |= (t[idx][None, :] == t[q][:, None]) & ~d[idx][None, :]
            psi = later * ((r[idx][None, :] < r[q][:, None]) + 0.5 * (r[idx][None, :] == r[q][:, None]))
            wb = w[:, idx].T                                           # [s, B]
            num += (wq * (psi @ wb)).sum(axis=0)
            den += (wq * (later.astype(np.float64) @ wb)).sum(axis=0)
        tree.add(rank[idx], w[:, idx].T)
        inserted += w[:, idx].sum(axis=1)
    return num, den


def harrell_c(risk: Any, time: Any, event: Any, weights: Any = None, *, tied_times: str = "exclude") -> Any:
    """Harrell's concordance index (ties in risk count 1/2); NaN without comparable pairs."""
    w, batched = weight_matrix(weights, np.asarray(time).size)
    num, den = concordance_sums(time, event, risk, w, tied_times=tied_times)
    return unbatch(safe_ratio(num, den), batched)


@dataclass(frozen=True)
class StepFunction:
    """A right-continuous step function with jumps at `times`; `values` [B, D] (one row per weight row)."""

    times: np.ndarray
    values: np.ndarray

    def at(self, t: Any) -> np.ndarray:
        """F(t) [B, len(t)]: value at the last jump <= t (1 before the first jump)."""
        tt = np.atleast_1d(np.asarray(t, dtype=np.float64))
        k = np.searchsorted(self.times, tt, side="right") - 1
        full = np.c_[np.ones(self.values.shape[0]), self.values]
        return full[:, k + 1]

    def left(self, t: Any) -> np.ndarray:
        """F(t-) [B, len(t)]: value at the last jump strictly before t (1 before the first jump)."""
        tt = np.atleast_1d(np.asarray(t, dtype=np.float64))
        k = np.searchsorted(self.times, tt, side="left") - 1
        full = np.c_[np.ones(self.values.shape[0]), self.values]
        return full[:, k + 1]


def kaplan_meier(time: Any, event: Any, weights: Any = None, *, censoring: bool = False) -> StepFunction:
    """Weighted Kaplan-Meier estimate of the event (or, with censoring=True, the censoring) survival.

    Events precede censorings at a tie (module docstring). A hazard 0/0 (nobody at risk) is 0.
    """
    t = as_float("time", time, 1)
    d = np.asarray(event).astype(bool)
    if d.shape != t.shape:
        raise InvariantViolation("time and event must have the same shape")
    w, _ = weight_matrix(weights, t.size)
    uniq = np.unique(t)
    idx = np.searchsorted(uniq, t)
    ev, ce = bin_sums(idx, uniq.size, w, d.astype(np.float64), (~d).astype(np.float64))
    total = ev + ce
    at_risk = np.cumsum(total[:, ::-1], axis=1)[:, ::-1]               # weight with T >= t_k
    # For the censoring distribution, the events at t_k leave the risk set before the censorings at t_k.
    hazard = safe_ratio(ce, at_risk - ev) if censoring else safe_ratio(ev, at_risk)
    surv = np.cumprod(1.0 - np.nan_to_num(hazard), axis=1)
    return StepFunction(uniq, surv)


def _inverse(g: np.ndarray) -> np.ndarray:
    # 1/G where G > 0, else 0 (such units cannot be reweighted).
    out = np.zeros_like(g)
    np.divide(1.0, g, out=out, where=g > 0)
    return out


def uno_c(risk: Any, time: Any, event: Any, weights: Any = None, *, horizon: float | None = None) -> Any:
    """Uno's IPCW concordance restricted to event times T_i < horizon.

    Without a horizon every event counts whose censoring weight is defined (G(T_i-) > 0); events where
    the censoring survival has reached 0 receive weight 0, which is the truncation Uno et al. require.
    """
    t, d, r = _validate_tte(time, event, risk)
    w, batched = weight_matrix(weights, t.size)
    g = kaplan_meier(t, d, w, censoring=True)
    g_left = g.left(t)                                                 # [B, n]
    tau = math.inf if horizon is None else float(horizon)
    mult = _inverse(g_left) ** 2 * (t < tau)[None, :]                 # [B, n]: one censoring estimate per row
    num, den = concordance_sums(t, d, r, w, multiplier=mult)
    out = safe_ratio(num, den)
    return unbatch(out, batched)


def survival_at(survival: np.ndarray, time_grid: np.ndarray, t: Any, *, interpolation: str = "step") -> np.ndarray:
    """S_i(t) [n, len(t)] from curves [n, T] on an increasing grid.

    "step": the value at the last grid time <= t (1 before the first grid time); "linear": linear between
    (0, 1) and the grid points (no extrapolation: the last value is held after the grid).
    """
    tt = np.atleast_1d(np.asarray(t, dtype=np.float64))
    if interpolation == "step":
        k = np.searchsorted(time_grid, tt, side="right") - 1
        full = np.c_[np.ones(survival.shape[0]), survival]
        return full[:, k + 1]
    _, curves, k, frac = _linear_position(survival, time_grid, tt, interpolation)
    return curves[:, k] + frac[None, :] * (curves[:, k + 1] - curves[:, k])


def _linear_position(survival: np.ndarray, time_grid: np.ndarray, tt: np.ndarray, interpolation: str
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Grid extended by (0, 1) when it starts after 0 and by a copy of the last point (to hold the last value);
    # returns the left knot index and the interpolation fraction of every time in tt.
    if interpolation != "linear":
        raise ValueError("interpolation must be 'step' or 'linear'")
    if time_grid.size and time_grid[0] > 0:
        grid = np.r_[0.0, time_grid]
        curves = np.c_[np.ones(survival.shape[0]), survival]
    else:
        grid, curves = time_grid, survival
    grid = np.r_[grid, grid[-1] + 1.0]
    curves = np.c_[curves, curves[:, -1]]
    k = np.clip(np.searchsorted(grid, tt, side="right") - 1, 0, grid.size - 2)
    span = grid[k + 1] - grid[k]
    frac = np.clip((np.minimum(tt, grid[-2]) - grid[k]) / span, 0.0, 1.0)
    return grid, curves, k, frac


def survival_at_own_time(survival: np.ndarray, time_grid: np.ndarray, t: np.ndarray, *, interpolation: str) -> np.ndarray:
    """S_i(t_i) [n]: each unit's curve at its own time (no [n, n] intermediate)."""
    tt = np.asarray(t, dtype=np.float64)
    rows = np.arange(tt.size)
    if interpolation == "step":
        k = np.searchsorted(time_grid, tt, side="right") - 1
        full = np.c_[np.ones(survival.shape[0]), survival]
        return full[rows, k + 1]
    _, curves, k, frac = _linear_position(survival, time_grid, tt, interpolation)
    return curves[rows, k] + frac * (curves[rows, k + 1] - curves[rows, k])


def _tte_arrays(pred: TimeToEventPredictions) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return pred.survival, pred.time_grid, pred.event_time, pred.event_observed, pred.risk


def survivors(event_time: np.ndarray, event_observed: np.ndarray, t: np.ndarray) -> np.ndarray:
    """1 where unit i is known event-free through t (T_i > t, or T_i = t and censored) [n, len(t)]."""
    tt = np.atleast_1d(t)
    return (event_time[:, None] > tt[None, :]) | ((event_time[:, None] == tt[None, :]) & ~event_observed[:, None])


def brier_ipcw(pred: TimeToEventPredictions, times: Any, weights: Any = None, *, interpolation: str = "step") -> Any:
    """IPCW Brier score BS(t) at each time [B, T] (or [T] unbatched); NaN where G(t-) = 0."""
    surv, grid, et, eo, _ = _tte_arrays(pred)
    tt = np.atleast_1d(as_float("times", times, None))
    w, batched = weight_matrix(weights, et.size)
    g = kaplan_meier(et, eo, w, censoring=True)
    inv_ti = _inverse(g.left(et))                                      # [B, n]
    g_t = g.left(tt)                                                   # [B, T]
    s_it = survival_at(surv, grid, tt, interpolation=interpolation)    # [n, T]
    died = (et[:, None] <= tt[None, :]) & eo[:, None]
    alive = survivors(et, eo, tt)
    num = (w * inv_ti) @ (s_it ** 2 * died) + (w @ ((1.0 - s_it) ** 2 * alive)) * _inverse(g_t)
    bs = safe_ratio(num, w.sum(axis=1)[:, None])
    bs = np.where(g_t > 0, bs, np.nan)
    return bs if batched else bs[0]


def integrated_brier(pred: TimeToEventPredictions, t_max: float | None = None, weights: Any = None, *,
                     interpolation: str = "step") -> Any:
    """IBS = (1/t_max) integral_0^{t_max} BS(t) dt over 0 and the grid times <= t_max (trapezoidal rule)."""
    grid = pred.time_grid
    top = float(grid[-1]) if t_max is None else float(t_max)
    if not top > 0:
        raise ValueError("t_max must be positive")
    times = np.unique(np.r_[0.0, grid[grid <= top], top])
    bs = np.atleast_2d(brier_ipcw(pred, times, weights if weights is not None else None, interpolation=interpolation))
    ok = np.all(np.isfinite(bs), axis=0)
    out = np.full(bs.shape[0], np.nan)
    if ok.sum() >= 2:
        tt = times[ok]
        out = np.trapezoid(bs[:, ok], tt, axis=1) / (tt[-1] - tt[0])
    batched = weights is not None and np.ndim(weights) == 2
    return out if batched else float(out[0])


def cumulative_dynamic_auc(pred: TimeToEventPredictions, times: Any, weights: Any = None, *, marker: str = "survival",
                           interpolation: str = "step") -> tuple[Any, Any]:
    """(AUC(t) per time, mean AUC) with IPCW case weights; marker "survival" (1 - S_i(t)) or "risk"."""
    if marker not in ("survival", "risk"):
        raise ValueError("marker must be 'survival' or 'risk'")
    surv, grid, et, eo, risk = _tte_arrays(pred)
    tt = np.atleast_1d(as_float("times", times, None))
    w, batched = weight_matrix(weights, et.size)
    g = kaplan_meier(et, eo, w, censoring=True)
    inv_ti = _inverse(g.left(et))                                      # [B, n]
    s_it = survival_at(surv, grid, tt, interpolation=interpolation)    # [n, T]
    aucs = np.full((w.shape[0], tt.size), np.nan)
    for k, t in enumerate(tt):
        case = (et <= t) & eo
        ctrl = survivors(et, eo, np.array([t]))[:, 0]
        if not case.any() or not ctrl.any():
            continue
        m = 1.0 - s_it[:, k] if marker == "survival" else risk
        comb = np.where(case[None, :], w * inv_ti, 0.0) + np.where(ctrl[None, :], w, 0.0)
        aucs[:, k] = np.asarray(auroc(m, case.astype(np.int64), comb), dtype=np.float64).reshape(-1)
    km = kaplan_meier(et, eo, w)
    s = km.at(tt)                                                      # [B, T]
    dec = -np.diff(np.c_[np.ones(w.shape[0]), s], axis=1)              # KM decrements
    ok = np.isfinite(aucs)
    num = np.where(ok, aucs * dec, 0.0).sum(axis=1)
    den = np.where(ok, dec, 0.0).sum(axis=1)
    mean = safe_ratio(num, den)
    if tt.size == 1:
        mean = aucs[:, 0]
    return (aucs if batched else aucs[0]), unbatch(mean, batched)


@dataclass(frozen=True)
class DCalibration:
    """D-calibration statistic, its chi-square p-value and the bin masses (observed, expected)."""

    statistic: float
    p_value: float
    observed: np.ndarray
    expected: float


def d_calibration_mass(pred: TimeToEventPredictions, *, bins: int = 10) -> np.ndarray:
    """Each unit's mass over the D-calibration bins [n, bins] (rows sum to 1; module docstring)."""
    if bins < 2:
        raise ValueError("bins must be >= 2")
    surv, grid, et, eo, _ = _tte_arrays(pred)
    s_t = survival_at_own_time(surv, grid, et, interpolation="linear") if et.size else np.zeros(0)
    s_t = np.clip(s_t, 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, bins + 1)
    mass = np.zeros((et.size, bins))
    # Uncensored: one unit of mass in the bin of S_i(T_i) (1.0 falls in the last bin).
    bidx = np.clip(np.searchsorted(edges, s_t, side="right") - 1, 0, bins - 1)
    mass[np.flatnonzero(eo), bidx[eo]] = 1.0
    # Censored with s = S_i(c_i): mass spread uniformly over [0, s] (all in the lowest bin when s = 0).
    cen = np.flatnonzero(~eo)
    s_c = s_t[cen]
    lo, hi = edges[:-1][None, :], edges[1:][None, :]
    overlap = np.clip(np.minimum(hi, s_c[:, None]) - lo, 0.0, None)    # |[lo, hi] intersect [0, s]|
    share = np.where(s_c[:, None] > 0, overlap / np.where(s_c[:, None] > 0, s_c[:, None], 1.0), 0.0)
    share[s_c <= 0, 0] = 1.0
    mass[cen] = share
    return mass


def d_calibration_statistic(pred: TimeToEventPredictions, *, bins: int = 10, weights: Any = None) -> Any:
    """Pearson statistic of D-calibration per weight row ([B] batched, a float otherwise)."""
    mass = d_calibration_mass(pred, bins=bins)
    w, batched = weight_matrix(weights, mass.shape[0])
    obs = w @ mass                                                     # [B, bins]
    expected = w.sum(axis=1, keepdims=True) / bins
    stat = safe_ratio(((obs - expected) ** 2).sum(axis=1), expected[:, 0])
    return unbatch(stat, batched)


def d_calibration(pred: TimeToEventPredictions, *, bins: int = 10, weights: Any = None) -> DCalibration:
    """Haider et al.'s D-calibration test (module docstring); weights act as unit multiplicities."""
    mass = d_calibration_mass(pred, bins=bins)
    w = np.ones(mass.shape[0]) if weights is None else as_float("weights", weights, 1)
    obs = w @ mass
    expected = float(w.sum()) / bins
    if expected <= 0:
        return DCalibration(math.nan, math.nan, obs, expected)
    statistic = float(np.sum((obs - expected) ** 2) / expected)
    return DCalibration(statistic, float(stats.chi2.sf(statistic, bins - 1)), obs, expected)


def subset(pred: TimeToEventPredictions, idx: np.ndarray) -> TimeToEventPredictions:
    """The record restricted to units idx (in that order)."""
    return TimeToEventPredictions(risk=pred.risk[idx], survival=pred.survival[idx], time_grid=pred.time_grid,
                                  event_time=pred.event_time[idx], event_observed=pred.event_observed[idx],
                                  meta=pred.meta.iloc[idx].reset_index(drop=True))


def rmst(survival: np.ndarray, time_grid: np.ndarray, tau: float) -> np.ndarray:
    """Restricted mean survival time integral_0^tau S(t) dt of step curves [n, T] -> [n]."""
    if not tau > 0:
        raise ValueError("tau must be positive")
    knots = np.unique(np.r_[0.0, time_grid[(time_grid > 0) & (time_grid < tau)], tau])
    vals = survival_at(survival, time_grid, knots[:-1], interpolation="step")   # S on [knot_k, knot_k+1)
    return vals @ np.diff(knots)


def time_to_event_from_forecast(f: ForecastPredictions) -> TimeToEventPredictions:
    """Time-to-event view of a forecast record, with censoring exactly as ForecastPredictions.outcome.

    Grid t_k = k w (k = 1 .. K) with S(t_k) = 1 - P_inf(k): the event of step e is placed at the end of
    its step, e w. A trigger with an event at step e has T = e w and d = 1; a trigger without an event
    and s observed steps is censored at s w (known event-free through step s, as `outcome` defines).
    The risk score is -RMST over the horizon K w, which ranks triggers by their expected time to
    infiltration within the horizon.
    """
    w = f.window_seconds
    k = f.horizon
    grid = w * np.arange(1, k + 1, dtype=np.float64)
    surv = np.clip(1.0 - f.p_inf, 0.0, 1.0)
    observed = f.event_step > 0
    t = np.where(observed, f.event_step * w, f.observed_steps * w).astype(np.float64)
    risk = -rmst(surv, grid, float(grid[-1]))
    return TimeToEventPredictions(risk=risk, survival=surv, time_grid=grid, event_time=t,
                                  event_observed=observed, meta=f.meta.reset_index(drop=True))
