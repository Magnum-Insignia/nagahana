"""Calibration: do forecast probabilities match how often the events happen?

Reliability analysis bins the forecast probabilities. In bin b with weight share n_b / n, mean forecast
conf_b and observed frequency acc_b, the expected calibration error is

    ECE   = sum_b (n_b / n) |acc_b - conf_b|                       (Guo et al., ICML 2017, arXiv:1706.04599)
    ECE_2 = ( sum_b (n_b / n) (acc_b - conf_b)^2 )^(1/2)           (the L2 calibration error)

with equal-width bins over [0, 1] (the last bin includes 1.0) or equal-mass bins whose edges are the
j/B quantiles of the forecasts (ties always share a bin, so empty bins can occur and are skipped). The
bin count changes the value, so it is a reported setting, never a hidden default.

Debiased estimator. The plug-in squared error is biased upward by the sampling variance of acc_b.
Kumar, Liang and Ma (NeurIPS 2019, "Verified Uncertainty Calibration", arXiv:1909.10155) subtract it:

    CE_2^2 (debiased) = sum_b (n_b / n) [ (acc_b - conf_b)^2 - acc_b (1 - acc_b) / (n_b - 1) ],

an unbiased estimate that can be slightly negative; bins with n_b < 2 keep the plug-in term.

ECE sweep. Roelofs, Cain, Shlens and Mozer (AISTATS 2022, arXiv:2012.08668) choose the largest number of
equal-mass bins for which the bin frequencies are still monotone in the forecast, which reduces the
bias of a fixed bin count.

Adaptive calibration error. Nixon et al. (CVPR Workshops 2019, arXiv:1904.01685) average |acc - conf|
over R equal-mass ranges of every class's predicted probability, ACE = (1 / (K R)) sum_k sum_r
|acc(r, k) - conf(r, k)|. Classwise ECE (Kull et al., NeurIPS 2019, arXiv:1910.12656) averages the
one-vs-rest ECE over classes, and the top-label ECE bins the confidence max_k p_k against the accuracy
of the predicted class.

Brier score and its decomposition. BS = (1/n) sum_i (f_i - o_i)^2. Murphy (Journal of Applied
Meteorology 12:595-600, 1973) splits it over bins of forecast values into reliability minus resolution
plus uncertainty; with continuous forecasts, Stephenson, Coelho and Jolliffe (Weather and Forecasting
23:752-757, 2008, doi:10.1175/2007WAF2006116.1) add a within-bin variance and covariance so that

    BS = REL - RES + UNC + WBV - WBC,
    REL = (1/n) sum_b n_b (fbar_b - obar_b)^2,     RES = (1/n) sum_b n_b (obar_b - obar)^2,
    UNC = obar (1 - obar),                         WBV = (1/n) sum_b sum_{i in b} (f_i - fbar_b)^2,
    WBC = (2/n) sum_b sum_{i in b} (o_i - obar_b)(f_i - fbar_b)

holds exactly. By default the bins are the distinct forecast values (Murphy's exact partition, where
WBV = WBC = 0) when there are at most `max_distinct` of them, and equal-width bins otherwise.

Log score. LS = -(1/n) sum_i [o_i log f_i + (1 - o_i) log(1 - f_i)]. It is unbounded: a forecast of 0
or 1 for an outcome that then does not occur gives +inf, which is reported as such (the thesis reports
the log score of persistence as unbounded). An explicit eps clips forecasts to [eps, 1 - eps].

Temperature effects. Temperature scaling p_T = sigmoid(logit(p) / T) (Guo et al. 2017) leaves the
ranking (and AUROC) unchanged and moves calibration; `temperature_effects` tabulates ECE, Brier and log
score over temperatures together with the maximum-likelihood temperature, found by bisection on the
convex negative log-likelihood in beta = 1 / T (the method of models/verifier/calibration.py).
`calibration_change` reports the same metrics before and after a calibration step (the Verifier's
"ECE and Brier before and after calibration").

Responded-to cases (the defenders acted, so the attack did not complete) are excluded by the caller,
as in roles/verifier.py.

Interfaces. `reliability` and `ece` keep their PyTorch form for the Verifier and the training code.
The NumPy functions accept unit weights ([n] or [B, n], see `_arrays`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy import stats

from nagahana.core.errors import InvariantViolation
from nagahana.evaluation._arrays import (
    as_binary,
    as_probability,
    rowwise_searchsorted,
    safe_ratio,
    unbatch,
    weight_matrix,
)


@dataclass(frozen=True)
class ReliabilityBin:
    """One reliability-diagram bin."""

    lower: float
    upper: float
    count: int
    mean_confidence: float
    frequency: float


def reliability(p: torch.Tensor, y: torch.Tensor, *, bins: int) -> list[ReliabilityBin]:
    """Equal-width reliability bins over [0, 1] (the last bin includes 1.0)."""
    if bins < 1:
        raise ValueError("bins must be >= 1")
    p, yf = p.flatten().double(), y.flatten().double()
    edges = torch.linspace(0, 1, bins + 1, dtype=torch.float64)
    idx = torch.clamp(torch.bucketize(p, edges, right=True) - 1, 0, bins - 1)
    out: list[ReliabilityBin] = []
    for b in range(bins):
        m = idx == b
        n = int(m.sum())
        out.append(ReliabilityBin(float(edges[b]), float(edges[b + 1]), n,
                                  float(p[m].mean()) if n else float("nan"),
                                  float(yf[m].mean()) if n else float("nan")))
    return out


def ece(p: torch.Tensor, y: torch.Tensor, *, bins: int) -> float:
    """Expected calibration error with equal-width bins (PyTorch interface)."""
    rel = reliability(p, y, bins=bins)
    n = sum(r.count for r in rel)
    if n == 0:
        return float("nan")
    return sum(r.count / n * abs(r.frequency - r.mean_confidence) for r in rel if r.count)


def _inputs(p: Any, y: Any, weights: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    pp = as_probability("p", p, 1)
    yy = as_binary("y", y).astype(np.float64)
    if pp.shape != yy.shape:
        raise InvariantViolation("p and y must have the same shape")
    w, batched = weight_matrix(weights, pp.size)
    return pp, yy, w, batched


def width_bins(p: np.ndarray, bins: int) -> np.ndarray:
    """Equal-width bin index in 0 .. bins-1 of each probability (1.0 falls in the last bin)."""
    if bins < 1:
        raise ValueError("bins must be >= 1")
    edges = np.linspace(0.0, 1.0, bins + 1)
    return np.clip(np.searchsorted(edges, p, side="right") - 1, 0, bins - 1)


def mass_bins(p: np.ndarray, bins: int, weights: np.ndarray | None = None) -> np.ndarray:
    """Equal-mass bin index [n]: edges at the j/bins (weighted) quantiles; ties share a bin."""
    w = np.ones((1, p.size)) if weights is None else np.asarray(weights, dtype=np.float64).reshape(1, -1)
    return mass_bin_index(p, w, bins)[0]


def mass_bin_index(p: np.ndarray, w: np.ndarray, bins: int) -> np.ndarray:
    """Equal-mass bin index of every unit under each weight row: [B, n] for weights [B, n].

    Edge j of row b is the forecast value at which row b's cumulative weight (forecasts in increasing
    order) first reaches j/bins of its total; a unit's bin is the number of edges at or below its
    forecast, so tied forecasts always share a bin. All rows are processed together.
    """
    if bins < 1:
        raise ValueError("bins must be >= 1")
    b, n = w.shape
    if n == 0:
        return np.zeros((b, 0), dtype=np.int64)
    if bins == 1:
        return np.zeros((b, n), dtype=np.int64)
    order = np.argsort(p, kind="stable")
    ps = p[order]
    cum = np.cumsum(w[:, order], axis=1)                                # [B, n]
    targets = cum[:, -1:] * (np.arange(1, bins) / bins)[None, :]        # [B, bins - 1]
    pos = np.clip(rowwise_searchsorted(cum, targets, side="left"), 0, n - 1)
    edges = ps[pos]                                                     # [B, bins - 1], non-decreasing per row
    return rowwise_searchsorted(edges, np.broadcast_to(p[None, :], (b, n)), side="right").astype(np.int64)


def row_bin_sums(idx: np.ndarray, n_bins: int, w: np.ndarray, *columns: np.ndarray | None) -> list[np.ndarray]:
    """Weighted per-bin sums [B, n_bins] of each column when every weight row has its own bins idx [B, n]."""
    b = w.shape[0]
    flat = (np.arange(b, dtype=np.int64)[:, None] * n_bins + idx).ravel()
    out = []
    for col in columns:
        vals = w if col is None else w * col[None, :]
        out.append(np.bincount(flat, weights=vals.ravel(), minlength=b * n_bins).reshape(b, n_bins))
    return out


def bin_sums(idx: np.ndarray, n_bins: int, w: np.ndarray, *columns: np.ndarray | None) -> list[np.ndarray]:
    """Weighted per-bin sums [B, n_bins] of each column (None sums the weights themselves).

    Units are sorted by bin once and summed with numpy.add.reduceat, so memory stays O(B n) whatever
    the number of bins.
    """
    b = w.shape[0]
    if idx.size == 0:
        return [np.zeros((b, n_bins)) for _ in columns]
    order = np.argsort(idx, kind="stable")
    sidx = idx[order]
    starts = np.flatnonzero(np.r_[True, sidx[1:] != sidx[:-1]])
    present = sidx[starts]
    ws = w[:, order]
    out = []
    for col in columns:
        vals = ws if col is None else ws * col[order][None, :]
        full = np.zeros((b, n_bins))
        full[:, present] = np.add.reduceat(vals, starts, axis=1)
        out.append(full)
    return out


def _bin_stats(idx: np.ndarray, p: np.ndarray, y: np.ndarray, w: np.ndarray, n_bins: int
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Weighted count, sum of forecasts, sum of outcomes and sum of squared forecasts per bin: [B, n_bins].
    cnt, sp, sy, spp = bin_sums(idx, n_bins, w, None, p, y, p * p)
    return cnt, sp, sy, spp


def calibration_error(p: Any, y: Any, *, bins: int, strategy: str = "width", norm: str = "l1",
                      debias: bool = False, weights: Any = None) -> Any:
    """Binned calibration error: L1 ECE, or the L2 calibration error (optionally debiased).

    strategy: "width" (equal-width) or "mass" (equal-mass, edges recomputed per weight row).
    norm: "l1" (ECE) or "l2" (square root of the weighted mean squared gap). debias (l2 only): the
    Kumar-Liang-Ma estimator; the returned value is sqrt(max(estimate, 0)).
    """
    if norm not in ("l1", "l2"):
        raise ValueError("norm must be 'l1' or 'l2'")
    if debias and norm != "l2":
        raise ValueError("the debiased estimator is defined for the squared (l2) calibration error")
    sq = squared_or_abs_gap(p, y, bins=bins, strategy=strategy, norm=norm, debias=debias, weights=weights)
    if norm == "l2":
        sq = np.sqrt(np.maximum(np.asarray(sq, dtype=np.float64), 0.0))
    return sq


def squared_or_abs_gap(p: Any, y: Any, *, bins: int, strategy: str, norm: str, debias: bool, weights: Any
                       ) -> Any:
    """L1 ECE (norm "l1") or the (possibly debiased, possibly negative) squared calibration error (norm "l2")."""
    pp, yy, w, batched = _inputs(p, y, weights)
    if strategy not in ("width", "mass"):
        raise ValueError("strategy must be 'width' or 'mass'")
    if strategy == "width":
        cnt, sp, sy, _ = _bin_stats(width_bins(pp, bins), pp, yy, w, bins)
    else:
        cnt, sp, sy = row_bin_sums(mass_bin_index(pp, w, bins), bins, w, None, pp, yy)
    total = cnt.sum(axis=1)
    conf, acc = safe_ratio(sp, cnt), safe_ratio(sy, cnt)
    share = safe_ratio(cnt, total[:, None])
    if norm == "l1":
        terms = np.where(cnt > 0, share * np.abs(acc - conf), 0.0)
    else:
        gap = np.where(cnt > 0, (acc - conf) ** 2, 0.0)
        if debias:
            corr = np.where(cnt >= 2, acc * (1.0 - acc) / np.where(cnt >= 2, cnt - 1.0, 1.0), 0.0)
            gap = gap - np.nan_to_num(corr)
        terms = np.where(cnt > 0, share * gap, 0.0)
    out = np.where(total > 0, terms.sum(axis=1), np.nan)
    return unbatch(out, batched)


def ece_sweep(p: Any, y: Any, *, norm: str = "l1", max_bins: int | None = None, weights: Any = None) -> Any:
    """ECE with the largest number of equal-mass bins whose frequencies are monotone (Roelofs et al. 2022)."""
    pp, yy, w, batched = _inputs(p, y, weights)
    out = np.full(w.shape[0], np.nan)
    cap = pp.size if max_bins is None else min(max_bins, pp.size)
    for b in range(w.shape[0]):
        if w[b].sum() <= 0:
            continue
        best = 1
        for nb in range(2, cap + 1):
            idx = mass_bins(pp, nb, w[b])
            cnt, _, sy, _ = _bin_stats(idx, pp, yy, w[b : b + 1], nb)
            freq = safe_ratio(sy[0], cnt[0])[cnt[0] > 0]
            if freq.size < 2 or np.any(np.diff(freq) < 0):
                break
            best = nb
        val = squared_or_abs_gap(pp, yy, bins=best, strategy="mass", norm=norm, debias=False, weights=w[b])
        out[b] = math.sqrt(max(float(val), 0.0)) if norm == "l2" else float(val)
    return unbatch(out, batched)


def reliability_table(p: Any, y: Any, *, bins: int, strategy: str = "width", weights: Any = None,
                      confidence: float = 0.95) -> pd.DataFrame:
    """Reliability-diagram data: one row per bin with Wilson score intervals of the observed frequency.

    Columns: bin, lower, upper, weight, mean_forecast, frequency, frequency_low, frequency_high. The
    Wilson interval (Wilson, JASA 22:209-212, 1927) uses the bin weight as the number of trials.
    """
    pp, yy, w, batched = _inputs(p, y, weights)
    if batched:
        raise ValueError("reliability_table takes a single weight vector")
    idx = width_bins(pp, bins) if strategy == "width" else mass_bins(pp, bins, w[0])
    cnt, sp, sy, _ = _bin_stats(idx, pp, yy, w, bins)
    cnt, sp, sy = cnt[0], sp[0], sy[0]
    if strategy == "width":
        edges = np.linspace(0.0, 1.0, bins + 1)
        lower, upper = edges[:-1], edges[1:]
    else:
        lower = np.array([pp[idx == b].min() if np.any(idx == b) else np.nan for b in range(bins)])
        upper = np.array([pp[idx == b].max() if np.any(idx == b) else np.nan for b in range(bins)])
    freq = safe_ratio(sy, cnt)
    # Wilson score interval: centre (f + z^2 / 2n) / (1 + z^2 / n), half-width
    # z sqrt(f (1 - f) / n + z^2 / 4n^2) / (1 + z^2 / n); empty bins have no interval.
    z = float(stats.norm.ppf(0.5 + confidence / 2.0))
    occupied = cnt > 0
    nn = np.where(occupied, cnt, 1.0)
    f = np.where(occupied, freq, 0.0)
    denom = 1.0 + z * z / nn
    centre = (f + z * z / (2.0 * nn)) / denom
    half = z * np.sqrt(np.maximum(f * (1.0 - f) / nn + z * z / (4.0 * nn * nn), 0.0)) / denom
    return pd.DataFrame({"bin": np.arange(bins), "lower": lower, "upper": upper, "weight": cnt,
                         "mean_forecast": safe_ratio(sp, cnt), "frequency": freq,
                         "frequency_low": np.where(occupied, centre - half, np.nan),
                         "frequency_high": np.where(occupied, centre + half, np.nan)})


def brier_score(p: Any, y: Any, weights: Any = None) -> Any:
    """Weighted mean of (p - y)^2."""
    pp, yy, w, batched = _inputs(p, y, weights)
    return unbatch(safe_ratio(w @ (pp - yy) ** 2, w.sum(axis=1)), batched)


def log_loss(p: Any, y: Any, *, eps: float | None = None, weights: Any = None) -> Any:
    """Weighted mean negative log-likelihood; +inf when a 0 or 1 forecast meets the other outcome (eps=None)."""
    pp, yy, w, batched = _inputs(p, y, weights)
    if eps is not None:
        if not 0.0 < eps < 0.5:
            raise ValueError("eps must lie in (0, 0.5)")
        pp = np.clip(pp, eps, 1.0 - eps)
    with np.errstate(divide="ignore"):
        ll = np.where(yy > 0.5, np.log(pp), np.log1p(-pp))               # -inf where the forecast was certain and wrong
    loss = -ll
    total = w.sum(axis=1)
    # Units of zero weight must not turn an infinite loss into NaN (0 * inf); mask them out first.
    contrib = np.where(w > 0, np.where(w > 0, w, 1.0) * loss[None, :], 0.0)   # no 0 * inf
    return unbatch(safe_ratio(contrib.sum(axis=1), total), batched)


def brier_decomposition(p: Any, y: Any, *, bins: int | None = None, max_distinct: int = 100,
                        weights: Any = None) -> dict[str, Any]:
    """Murphy decomposition with the Stephenson-Coelho-Jolliffe terms: BS = REL - RES + UNC + WBV - WBC.

    bins=None uses the distinct forecast values when there are at most `max_distinct` of them, and
    otherwise `max_distinct` equal-width bins. Returns the five terms and the Brier score itself.
    """
    pp, yy, w, batched = _inputs(p, y, weights)
    if bins is None:
        uniq, inv = np.unique(pp, return_inverse=True)
        if uniq.size <= max_distinct:
            idx, n_bins = inv.reshape(-1).astype(np.int64), int(uniq.size)
        else:
            idx, n_bins = width_bins(pp, max_distinct), max_distinct
    else:
        idx, n_bins = width_bins(pp, bins), bins
    cnt, sp, sy, spp = _bin_stats(idx, pp, yy, w, max(n_bins, 1))
    total = cnt.sum(axis=1)
    fbar, obar_b = safe_ratio(sp, cnt), safe_ratio(sy, cnt)
    obar = safe_ratio(sy.sum(axis=1), total)
    occupied = cnt > 0
    rel = safe_ratio(np.where(occupied, cnt * (fbar - obar_b) ** 2, 0.0).sum(axis=1), total)
    res = safe_ratio(np.where(occupied, cnt * (obar_b - obar[:, None]) ** 2, 0.0).sum(axis=1), total)
    unc = obar * (1.0 - obar)
    # Within-bin variance of forecasts: sum_i w (f - fbar_b)^2 = sum w f^2 - n_b fbar_b^2.
    wbv = safe_ratio(np.where(occupied, spp - cnt * fbar ** 2, 0.0).sum(axis=1), total)
    # Within-bin covariance: (2/n) sum_i w (o - obar_b)(f - fbar_b) = (2/n) sum_b (sum w o f - n_b obar_b fbar_b).
    (syf,) = bin_sums(idx, max(n_bins, 1), w, yy * pp)
    wbc = 2.0 * safe_ratio(np.where(occupied, syf - cnt * obar_b * fbar, 0.0).sum(axis=1), total)
    bs = safe_ratio(w @ (pp - yy) ** 2, total)
    out = {"reliability": rel, "resolution": res, "uncertainty": unc, "within_bin_variance": wbv,
           "within_bin_covariance": wbc, "brier": bs}
    return {k: unbatch(np.asarray(v), batched) for k, v in out.items()}


def _check_multiclass(probs: Any, label: Any) -> tuple[np.ndarray, np.ndarray]:
    pr = as_probability("probs", probs, 2)
    lab = np.asarray(label, dtype=np.int64)
    if lab.shape != (pr.shape[0],) or np.any((lab < 0) | (lab >= pr.shape[1])):
        raise InvariantViolation("label must be [n] with values in 0 .. K-1")
    return pr, lab


def top_label_ece(probs: Any, label: Any, *, bins: int, weights: Any = None) -> Any:
    """ECE of the confidence max_k p_k against the correctness of the predicted class (Guo et al. 2017)."""
    pr, lab = _check_multiclass(probs, label)
    conf = pr.max(axis=1)
    correct = (pr.argmax(axis=1) == lab).astype(np.int64)
    return calibration_error(conf, correct, bins=bins, weights=weights)


def classwise_ece(probs: Any, label: Any, *, bins: int, weights: Any = None) -> Any:
    """Mean over classes of the one-vs-rest ECE (Kull et al. 2019); classes absent and never predicted skip."""
    pr, lab = _check_multiclass(probs, label)
    w, batched = weight_matrix(weights, lab.size)
    vals = []
    for k in range(pr.shape[1]):
        yk = (lab == k).astype(np.int64)
        if yk.sum() == 0 and pr[:, k].max() == 0:
            continue
        vals.append(np.asarray(calibration_error(pr[:, k], yk, bins=bins, weights=w), dtype=np.float64))
    if not vals:
        return unbatch(np.full(w.shape[0], np.nan), batched)
    return unbatch(np.nanmean(np.vstack(vals), axis=0), batched)


def adaptive_calibration_error(probs: Any, label: Any, *, ranges: int, weights: Any = None) -> Any:
    """ACE: mean |acc - conf| over `ranges` equal-mass ranges of each class's probability (Nixon et al. 2019)."""
    pr, lab = _check_multiclass(probs, label)
    w, batched = weight_matrix(weights, lab.size)
    out = np.zeros(w.shape[0])
    cells = np.zeros(w.shape[0])
    for k in range(pr.shape[1]):
        yk = (lab == k).astype(np.float64)
        # Equal-mass ranges of class k's probability, recomputed for every weight row at once.
        cnt, sp, sy = row_bin_sums(mass_bin_index(pr[:, k], w, ranges), ranges, w, None, pr[:, k], yk)
        occ = cnt > 0
        gap = np.abs(safe_ratio(sy, cnt) - safe_ratio(sp, cnt))
        out += np.where(occ, gap, 0.0).sum(axis=1)
        cells += occ.sum(axis=1)
    return unbatch(safe_ratio(out, cells), batched)


def multiclass_brier(probs: Any, label: Any, weights: Any = None) -> Any:
    """Weighted mean of sum_k (p_k - 1[y = k])^2 (Brier, Monthly Weather Review 78:1-3, 1950)."""
    pr, lab = _check_multiclass(probs, label)
    w, batched = weight_matrix(weights, lab.size)
    onehot = np.zeros_like(pr)
    onehot[np.arange(lab.size), lab] = 1.0
    per = ((pr - onehot) ** 2).sum(axis=1)
    return unbatch(safe_ratio(w @ per, w.sum(axis=1)), batched)


def multiclass_log_loss(probs: Any, label: Any, *, eps: float | None = None, weights: Any = None) -> Any:
    """Weighted mean of -log p_y; +inf where the true class had probability 0 (eps=None)."""
    pr, lab = _check_multiclass(probs, label)
    w, batched = weight_matrix(weights, lab.size)
    py = pr[np.arange(lab.size), lab]
    if eps is not None:
        py = np.clip(py, eps, 1.0)
    with np.errstate(divide="ignore"):
        loss = -np.log(py)
    contrib = np.where(w > 0, np.where(w > 0, w, 1.0) * loss[None, :], 0.0)   # no 0 * inf
    return unbatch(safe_ratio(contrib.sum(axis=1), w.sum(axis=1)), batched)


def logit(p: Any, *, clip: float = 1e-12) -> np.ndarray:
    """log p - log(1 - p) with p clipped to [clip, 1 - clip] (forecasts of exactly 0 or 1 stay extreme)."""
    q = np.clip(np.asarray(p, dtype=np.float64), clip, 1.0 - clip)
    return np.log(q) - np.log1p(-q)


def apply_temperature(p: Any, temperature: float) -> np.ndarray:
    """p_T = sigmoid(logit(p) / T)."""
    if not temperature > 0:
        raise ValueError("temperature must be > 0")
    return 1.0 / (1.0 + np.exp(-logit(p) / temperature))


def ml_temperature(p: Any, y: Any, *, t_min: float = 0.05, t_max: float = 20.0, tol: float = 1e-10,
                   max_iter: int = 200) -> float:
    """Maximum-likelihood temperature in [t_min, t_max] by bisection on NLL'(beta), beta = 1/T (convex in beta)."""
    if not 0 < t_min < t_max:
        raise ValueError("need 0 < t_min < t_max")
    z = logit(as_probability("p", p, 1))
    yy = as_binary("y", y).astype(np.float64)
    if z.size == 0:
        raise ValueError("no forecasts")

    def grad(beta: float) -> float:
        return float(np.sum(z * (1.0 / (1.0 + np.exp(-beta * z)) - yy)))

    lo, hi = 1.0 / t_max, 1.0 / t_min
    if grad(lo) >= 0.0:
        return t_max
    if grad(hi) <= 0.0:
        return t_min
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if grad(mid) > 0.0:
            hi = mid
        else:
            lo = mid
        if hi - lo < tol:
            break
    return 1.0 / (0.5 * (lo + hi))


def temperature_effects(p: Any, y: Any, temperatures: Any, *, bins: int) -> pd.DataFrame:
    """ECE, L2 calibration error, Brier and log score after temperature scaling, per temperature.

    The last row is the maximum-likelihood temperature (column `ml` True). The ranking, hence AUROC,
    is the same at every temperature.
    """
    pp = as_probability("p", p, 1)
    yy = as_binary("y", y)
    temps = [float(t) for t in np.atleast_1d(np.asarray(temperatures, dtype=np.float64))]
    t_ml = ml_temperature(pp, yy)
    rows = []
    for t, is_ml in [(t, False) for t in temps] + [(t_ml, True)]:
        q = apply_temperature(pp, t)
        rows.append({"temperature": t, "ml": is_ml,
                     "ece": calibration_error(q, yy, bins=bins),
                     "ce_l2": calibration_error(q, yy, bins=bins, norm="l2"),
                     "brier": brier_score(q, yy), "log_score": log_loss(q, yy)})
    return pd.DataFrame(rows)


def calibration_change(p_before: Any, p_after: Any, y: Any, *, bins: int, weights: Any = None) -> dict[str, Any]:
    """ECE and Brier before and after a calibration step, and the change after - before."""
    out: dict[str, Any] = {}
    for tag, p in (("before", p_before), ("after", p_after)):
        out[f"ece_{tag}"] = calibration_error(p, y, bins=bins, weights=weights)
        out[f"brier_{tag}"] = brier_score(p, y, weights=weights)
    out["ece_change"] = np.asarray(out["ece_after"]) - np.asarray(out["ece_before"])
    out["brier_change"] = np.asarray(out["brier_after"]) - np.asarray(out["brier_before"])
    if np.ndim(out["ece_change"]) == 0:
        out["ece_change"], out["brier_change"] = float(out["ece_change"]), float(out["brier_change"])
    return out
