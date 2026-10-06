"""Threshold-free discrimination and operating points: ROC, AUROC with DeLong variance, AUPRC, pAUROC.

Every function here moves a threshold through the groups of tied scores in descending order
(`_arrays.descending_groups`), so a tie is always crossed as a whole and every result is invariant to
the order in which tied units are stored. All functions accept unit weights ([n] or [B, n], see
`_arrays`); a weight matrix evaluates B bootstrap resamples at once.

AUROC

With positives X_1..X_m and negatives Y_1..Y_n and the kernel psi(x, y) = 1 if x > y, 1/2 if x = y,
0 otherwise, the area under the ROC curve is the Mann-Whitney statistic

    AUROC = (1 / (m n)) sum_i sum_j psi(X_i, Y_j)

(Hanley and McNeil, Radiology 143:29-36, 1982, doi:10.1148/radiology.143.1.7063747). With weights it
is sum_i sum_j a_i b_j psi(X_i, Y_j) / (sum_i a_i sum_j b_j), computed in O(n log n) from cumulative
negative weight below each group of tied scores.

DeLong variance

DeLong, DeLong and Clarke-Pearson (Biometrics 44:837-845, 1988, doi:10.2307/2531595) write the AUROC
as the mean of structural components V10(X_i) = (1/n) sum_j psi(X_i, Y_j) and
V01(Y_j) = (1/m) sum_i psi(X_i, Y_j). For k scores on the same units, with S10 and S01 the k x k
sample covariance matrices of these components (denominators m - 1 and n - 1),

    Cov(AUROC) = S10 / m + S01 / n.

Sun and Xu (IEEE Signal Processing Letters 21:1389-1393, 2014, doi:10.1109/LSP.2014.2337313) compute
the components from midranks in O((m + n) log(m + n)): with T_Z the midranks in the pooled sample, T_X
the midranks among positives and T_Y among negatives,

    V10(X_i) = (T_Z(X_i) - T_X(X_i)) / n,        V01(Y_j) = 1 - (T_Z(Y_j) - T_Y(Y_j)) / m,
    AUROC = (sum_i T_Z(X_i) / m - (m + 1) / 2) / n.

AUPRC as average precision

AUPRC is reported as average precision, AP = sum_g (R_g - R_{g-1}) P_g over the groups g of tied
scores in descending order, with P_g and R_g the precision and recall after group g. Davis and
Goadrich (ICML 2006, doi:10.1145/1143844.1143874) show that linear interpolation between PR points
overestimates the area; the step-wise AP never interpolates. Their correct interpolation is also
provided (`auprc_interpolated`): between two achievable points A and B, true positives grow by x from
TP_A while false positives grow by s x with s = (FP_B - FP_A) / (TP_B - TP_A), so that

    precision(x) = (TP_A + x) / (TP_A + FP_A + (1 + s) x),   recall(x) = (TP_A + x) / P,

whose integral over each segment has the closed form used below.

Partial AUROC

The area under the ROC curve for FPR in [0, e] (with the curve linearly interpolated at FPR = e, the
ROC of the corresponding randomised rule) is standardised as McClish (Medical Decision Making
9:190-195, 1989, doi:10.1177/0272989X8900900307) proposes:

    pAUROC_std = (1/2) (1 + (pAUROC - e^2 / 2) / (e - e^2 / 2)),

so that 1/2 is chance and 1 is perfect within the region.

Operating points

Every decision rule here alerts when score >= threshold. `operating_point_at_fpr` chooses, among the
achievable thresholds, the one with the largest TPR subject to FPR <= alpha on the evaluated units (an
oracle operating point used to compare rankings at a matched false-positive rate).
`conformal_threshold` fits a threshold on benign calibration scores with the split-conformal rule of
the Verifier (Vovk, Gammerman and Shafer, Algorithmic Learning in a Random World, Springer 2005 and
2022 edition; Angelopoulos and Bates, arXiv:2107.07511): with k = ceil((n + 1)(1 - alpha)) and q the
k-th smallest benign score, alerting when s > q gives P(s_new > q) <= alpha for an exchangeable benign
unit. The returned threshold is the next float above q, so the inclusive rule s >= threshold is
exactly s > q.

Alert volume

False alerts per day at the evaluation base rate are FP / (observed days), with the observed days the
sum over (dataset, network) of the time spanned by the evaluated units. `precision_at_base_rate`
re-expresses precision at another base rate pi by Bayes' rule,
P = pi TPR / (pi TPR + (1 - pi) FPR), as Arp et al. (USENIX Security 2022) recommend.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from nagahana.core.errors import InvariantViolation
from nagahana.evaluation._arrays import (
    as_binary,
    as_float,
    descending_groups,
    group_sums,
    safe_ratio,
    unbatch,
    weight_matrix,
)


def _prepare(score: Any, label: Any, weights: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    # Scores may be any finite reals (probabilities, energies, logits); labels 0/1.
    s = as_float("score", score, 1)
    y = as_binary("label", label)
    if s.shape != y.shape:
        raise InvariantViolation("score and label must have the same shape")
    w, batched = weight_matrix(weights, s.size)
    return s, y, w, batched


@dataclass(frozen=True)
class Grouped:
    """Units sorted by descending score, with the groups of tied scores (one sort serves every resample).

    order     permutation sorting the units by descending score (stable)
    starts    first sorted position of each group of tied scores
    values    the score of each group
    y_sorted  float64 labels in sorted order
    """

    order: np.ndarray
    starts: np.ndarray
    values: np.ndarray
    y_sorted: np.ndarray

    def class_weights(self, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Positive and negative weight per group [B, G] for weight rows w [B, n]."""
        ws = w[:, self.order]
        return group_sums(ws * self.y_sorted, self.starts), group_sums(ws * (1.0 - self.y_sorted), self.starts)


def grouped(score: Any, label: Any) -> Grouped:
    """Sort once by descending score and locate the groups of ties."""
    s = as_float("score", score, 1)
    y = as_binary("label", label)
    if s.shape != y.shape:
        raise InvariantViolation("score and label must have the same shape")
    order, starts, values = descending_groups(s)
    return Grouped(order, starts, values, y[order].astype(np.float64))


def _grouped(s: np.ndarray, y: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Positive and negative weight per group of tied scores, groups in descending score order.
    g = grouped(s, y)
    pos, neg = g.class_weights(w)
    return pos, neg, g.values


def auroc_from_groups(pos: np.ndarray, neg: np.ndarray) -> np.ndarray:
    """AUROC [B] from group weights: sum_g pos_g (negatives strictly below + neg_g / 2) / (P N)."""
    p_tot, n_tot = pos.sum(axis=1), neg.sum(axis=1)
    cum_neg = np.cumsum(neg, axis=1)                                     # negatives scoring >= group
    below = n_tot[:, None] - cum_neg                                     # negatives scoring strictly lower
    num = (pos * (below + 0.5 * neg)).sum(axis=1)
    return safe_ratio(num, p_tot * n_tot)


def average_precision_from_groups(pos: np.ndarray, neg: np.ndarray) -> np.ndarray:
    """Average precision [B] from group weights: sum_g (pos_g / P) precision_g."""
    tp, fp = np.cumsum(pos, axis=1), np.cumsum(neg, axis=1)
    prec = np.where(pos > 0, safe_ratio(tp, tp + fp), 0.0)              # P_g only where recall moves
    return safe_ratio((pos * prec).sum(axis=1), pos.sum(axis=1))


def partial_auroc_from_groups(pos: np.ndarray, neg: np.ndarray, max_fpr: float, *, standardized: bool = True) -> np.ndarray:
    """Partial AUROC over FPR in [0, max_fpr] [B] from group weights (McClish-standardised by default)."""
    if not 0.0 < max_fpr <= 1.0:
        raise ValueError("max_fpr must lie in (0, 1]")
    p_tot, n_tot = pos.sum(axis=1), neg.sum(axis=1)
    b = pos.shape[0]
    tpr = np.c_[np.zeros(b), np.cumsum(pos, axis=1) / np.where(p_tot > 0, p_tot, 1.0)[:, None]]
    fpr = np.c_[np.zeros(b), np.cumsum(neg, axis=1) / np.where(n_tot > 0, n_tot, 1.0)[:, None]]
    f0, f1 = fpr[:, :-1], fpr[:, 1:]                                     # segment ends [B, G]
    t0, t1 = tpr[:, :-1], tpr[:, 1:]
    f_hi = np.minimum(f1, max_fpr)
    width = np.clip(f_hi - f0, 0.0, None)                                # part of the segment inside [0, e]
    frac = safe_ratio(width, f1 - f0)
    t_hi = t0 + np.nan_to_num(frac) * (t1 - t0)
    area = (width * 0.5 * (t0 + t_hi)).sum(axis=1)
    if standardized and max_fpr < 1.0:
        lo, hi = 0.5 * max_fpr * max_fpr, max_fpr
        area = 0.5 * (1.0 + (area - lo) / (hi - lo))
    return np.where((p_tot > 0) & (n_tot > 0), area, np.nan)


@dataclass(frozen=True)
class OperatingPoint:
    """An operating point: alert when score >= threshold; rates are arrays [B] when batched."""

    threshold: Any
    tpr: Any
    fpr: Any
    precision: Any


def operating_point_from_groups(pos: np.ndarray, neg: np.ndarray, values: np.ndarray, alpha: float
                                ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(threshold, TPR, FPR, precision) [B] of the largest TPR with FPR <= alpha (+inf and TPR 0 if none)."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must lie in [0, 1]")
    p_tot, n_tot = pos.sum(axis=1), neg.sum(axis=1)
    tp, fp = np.cumsum(pos, axis=1), np.cumsum(neg, axis=1)
    fpr_g = safe_ratio(fp, n_tot[:, None])
    # FPR is non-decreasing over the groups, so the feasible groups form a prefix.
    feasible = np.nan_to_num(fpr_g, nan=np.inf) <= alpha + 1e-15
    idx = feasible.sum(axis=1) - 1                                       # last feasible group, -1 if none
    rows = np.arange(pos.shape[0])
    safe_idx = np.clip(idx, 0, None)
    if values.size == 0:
        nan = np.full(pos.shape[0], np.nan)
        return np.full(pos.shape[0], np.inf), nan, nan, nan
    thr = np.where(idx >= 0, values[safe_idx], np.inf)
    tpr = np.where(idx >= 0, safe_ratio(tp[rows, safe_idx], p_tot), np.where(p_tot > 0, 0.0, np.nan))
    fpr = np.where(idx >= 0, fpr_g[rows, safe_idx], np.where(n_tot > 0, 0.0, np.nan))
    prec = np.where(idx >= 0, safe_ratio(tp[rows, safe_idx], tp[rows, safe_idx] + fp[rows, safe_idx]), np.nan)
    return thr, tpr, fpr, prec


def auroc(score: Any, label: Any, weights: Any = None) -> Any:
    """Area under the ROC curve (Mann-Whitney form, ties count 1/2); NaN without both classes."""
    s, y, w, batched = _prepare(score, label, weights)
    pos, neg, _ = _grouped(s, y, w)
    return unbatch(auroc_from_groups(pos, neg), batched)


def roc_curve(score: Any, label: Any, weights: Any = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(fpr, tpr, thresholds) from the point (0, 0) at threshold +inf through every group of ties."""
    s, y, w, batched = _prepare(score, label, weights)
    if batched:
        raise ValueError("roc_curve takes a single weight vector")
    pos, neg, values = _grouped(s, y, w)
    tpr = np.r_[0.0, safe_ratio(np.cumsum(pos[0]), pos[0].sum())]
    fpr = np.r_[0.0, safe_ratio(np.cumsum(neg[0]), neg[0].sum())]
    return fpr, tpr, np.r_[np.inf, values]


def pr_curve(score: Any, label: Any, weights: Any = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(precision, recall, thresholds) after each group of ties in descending score order."""
    s, y, w, batched = _prepare(score, label, weights)
    if batched:
        raise ValueError("pr_curve takes a single weight vector")
    pos, neg, values = _grouped(s, y, w)
    tp, fp = np.cumsum(pos[0]), np.cumsum(neg[0])
    return safe_ratio(tp, tp + fp), safe_ratio(tp, pos[0].sum()), values


def average_precision(score: Any, label: Any, weights: Any = None) -> Any:
    """AUPRC as average precision, sum_g (R_g - R_{g-1}) P_g (no interpolation); NaN without positives."""
    s, y, w, batched = _prepare(score, label, weights)
    pos, neg, _ = _grouped(s, y, w)
    return unbatch(average_precision_from_groups(pos, neg), batched)


def auprc_interpolated(score: Any, label: Any, weights: Any = None) -> float:
    """Area under the PR curve with the Davis-Goadrich interpolation between achievable points.

    The curve starts at recall 0 with the precision of the first group of ties (the limit of the
    interpolation from the empty prediction). Each segment A -> B contributes
    (1/P) integral_0^{dTP} (a + x) / (c + k x) dx with a = TP_A, c = TP_A + FP_A, k = 1 + s, which is
    (1/P) [dTP / k + (a - c / k) (1 / k) log((c + k dTP) / c)] for c > 0 and (1/P) dTP / k for c = 0.
    """
    s, y, w, batched = _prepare(score, label, weights)
    if batched:
        raise ValueError("auprc_interpolated takes a single weight vector")
    pos, neg, _ = _grouped(s, y, w)
    pos, neg = pos[0], neg[0]
    p_tot = pos.sum()
    if p_tot <= 0:
        return math.nan
    tp = np.r_[0.0, np.cumsum(pos)]
    fp = np.r_[0.0, np.cumsum(neg)]
    area = 0.0
    for i in range(1, tp.size):
        d_tp, d_fp = tp[i] - tp[i - 1], fp[i] - fp[i - 1]
        if d_tp <= 0:
            continue                                                     # recall does not move: no area
        k = 1.0 + d_fp / d_tp
        a, c = tp[i - 1], tp[i - 1] + fp[i - 1]
        if c <= 0:
            area += d_tp / k
        else:
            area += d_tp / k + (a - c / k) / k * math.log((c + k * d_tp) / c)
    return float(area / p_tot)


def partial_auroc(score: Any, label: Any, max_fpr: float, *, standardized: bool = True, weights: Any = None) -> Any:
    """Area under the ROC curve over FPR in [0, max_fpr], McClish-standardised by default.

    The ROC is linearly interpolated at FPR = max_fpr (a convex combination of two achievable rules).
    """
    if not 0.0 < max_fpr <= 1.0:
        raise ValueError("max_fpr must lie in (0, 1]")
    s, y, w, batched = _prepare(score, label, weights)
    pos, neg, _ = _grouped(s, y, w)
    return unbatch(partial_auroc_from_groups(pos, neg, max_fpr, standardized=standardized), batched)


def operating_point_at_fpr(score: Any, label: Any, alpha: float, weights: Any = None) -> OperatingPoint:
    """The achievable threshold with the largest TPR subject to FPR <= alpha on the evaluated units.

    When no threshold satisfies the bound except "never alert", the threshold is +inf with TPR 0.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must lie in [0, 1]")
    s, y, w, batched = _prepare(score, label, weights)
    pos, neg, values = _grouped(s, y, w)
    thr, tpr, fpr, prec = operating_point_from_groups(pos, neg, values, alpha)
    return OperatingPoint(unbatch(thr, batched), unbatch(tpr, batched), unbatch(fpr, batched), unbatch(prec, batched))


def conformal_threshold(benign_scores: Any, alpha: float) -> float:
    """Split-conformal alert threshold on benign calibration scores (alert when score >= threshold).

    k = ceil((n + 1)(1 - alpha)) is computed exactly from the decimal reading of alpha (a rational
    number), so that values such as alpha = 0.001 do not lose the guarantee to rounding. Returns +inf
    when k > n (too few calibration scores for this alpha: never alert by this rule).
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    s = np.sort(as_float("benign_scores", benign_scores, 1))
    n = s.size
    k = math.ceil(Fraction(n + 1) * (1 - Fraction(repr(float(alpha)))))
    if n == 0 or k > n:
        return math.inf
    return float(np.nextafter(s[k - 1], np.inf))


def delong_components(scores: Any, label: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """AUROC [k] and structural components V10 [k, m] (positives), V01 [k, n] (negatives) by midranks.

    scores: [k, N] (k scores of the same N units) or [N]; label [N] with both classes present.
    """
    s = as_float("scores", scores)
    if s.ndim == 1:
        s = s[None, :]
    if s.ndim != 2:
        raise InvariantViolation("scores must be [N] or [k, N]")
    y = as_binary("label", label)
    if s.shape[1] != y.size:
        raise InvariantViolation("scores and label must cover the same units")
    xs, ys = s[:, y == 1], s[:, y == 0]
    m, n = xs.shape[1], ys.shape[1]
    if m == 0 or n == 0:
        raise InvariantViolation("DeLong needs at least one positive and one negative unit")
    tx = stats.rankdata(xs, axis=1)                                      # midranks among positives
    ty = stats.rankdata(ys, axis=1)                                      # midranks among negatives
    tz = stats.rankdata(np.concatenate([xs, ys], axis=1), axis=1)        # midranks in the pooled sample
    auc = (tz[:, :m].sum(axis=1) / m - (m + 1) / 2.0) / n
    v10 = (tz[:, :m] - tx) / n
    v01 = 1.0 - (tz[:, m:] - ty) / m
    return auc, v10, v01


def delong_covariance(scores: Any, label: Any) -> tuple[np.ndarray, np.ndarray]:
    """AUROC [k] and their DeLong covariance matrix [k, k] = S10 / m + S01 / n."""
    auc, v10, v01 = delong_components(scores, label)
    m, n = v10.shape[1], v01.shape[1]
    s10 = np.atleast_2d(np.cov(v10)) if m > 1 else np.zeros((auc.size, auc.size))
    s01 = np.atleast_2d(np.cov(v01)) if n > 1 else np.zeros((auc.size, auc.size))
    return auc, s10 / m + s01 / n


def auroc_confidence_interval(score: Any, label: Any, *, confidence: float = 0.95, method: str = "logit") -> tuple[float, float, float]:
    """(AUROC, low, high) from the DeLong standard error.

    method "wald": AUROC +- z se. method "logit": the interval on logit(AUROC) with the delta-method
    standard error se / (A (1 - A)), back-transformed, which stays inside [0, 1] (Pepe, The Statistical
    Evaluation of Medical Tests for Classification and Prediction, Oxford 2003, section 5.2).
    """
    auc, cov = delong_covariance(score, label)
    a, se = float(auc[0]), math.sqrt(max(float(cov[0, 0]), 0.0))
    z = float(stats.norm.ppf(0.5 + confidence / 2.0))
    if method == "wald":
        return a, max(0.0, a - z * se), min(1.0, a + z * se)
    if method != "logit":
        raise ValueError("method must be 'wald' or 'logit'")
    if a <= 0.0 or a >= 1.0 or se == 0.0:
        return a, a, a
    lg, se_lg = math.log(a / (1.0 - a)), se / (a * (1.0 - a))
    lo, hi = lg - z * se_lg, lg + z * se_lg
    return a, 1.0 / (1.0 + math.exp(-lo)), 1.0 / (1.0 + math.exp(-hi))


def observed_days(meta: pd.DataFrame) -> float:
    """Days of observation covered by units: sum over (dataset, network) of (max time - min time) / 86400."""
    if len(meta) == 0:
        return 0.0
    t = meta["time"].to_numpy(dtype=np.float64)
    key = meta["dataset"].astype(str).to_numpy() + "\x1f" + meta["network"].astype(str).to_numpy()
    frame = pd.DataFrame({"key": key, "t": t})
    span = frame.groupby("key")["t"].agg(lambda x: float(x.max() - x.min()))
    return float(span.sum()) / 86400.0


def alerts_per_day(n_alerts: Any, days: float) -> Any:
    """Alerts per day of observation; NaN when no time was observed."""
    return safe_ratio(n_alerts, days) if days > 0 else np.full(np.shape(n_alerts), np.nan)


def precision_at_base_rate(tpr: Any, fpr: Any, base_rate: float) -> Any:
    """Precision re-expressed at base rate pi: pi TPR / (pi TPR + (1 - pi) FPR) (Bayes' rule)."""
    if not 0.0 <= base_rate <= 1.0:
        raise ValueError("base_rate must lie in [0, 1]")
    t, f = np.asarray(tpr, dtype=np.float64), np.asarray(fpr, dtype=np.float64)
    return safe_ratio(base_rate * t, base_rate * t + (1.0 - base_rate) * f)
