"""Scores used to choose hyperparameters on held-out folds (not the reported metrics, which the evaluation
package computes from the prediction records). The default criterion of each model is recorded in AS-513.

Binary (y in {0, 1}, score s or probability p):

    average_precision   sum_n (R_n - R_{n-1}) P_n over the distinct thresholds, highest first (the step
                        interpolation of the precision-recall curve; Davis and Goadrich, ICML 2006, on why
                        it is the area to read for rare positives)
    auroc               P(s_pos > s_neg) + 0.5 P(s_pos = s_neg) (Mann-Whitney U / (n_pos n_neg))
    log_loss            -(1/n) sum [y log p + (1 - y) log(1 - p)]
    balanced_log_loss   the mean of the per-class mean log losses, a proper score under the class-balanced
                        distribution and therefore comparable across training class weights
    brier               (1/n) sum (p - y)^2

Multiclass (labels in 0 ... S-1, probabilities [n, S]):

    log_loss, balanced_log_loss (mean over the classes present of the per-class mean -log p_y),
    macro_f1 (top-1 decisions, mean F1 over the classes present in the labels), top1 (accuracy)

Discrete-time survival:

    survival_nll        mean over units of -[sum_{j < e} log(1 - h_j) + log h_e] for an event at step e and
                        -sum_{j <= c} log(1 - h_j) for a unit censored after c event-free steps (the
                        likelihood the hazard model is fitted with; Singer and Willett, Journal of
                        Educational Statistics 18(2), 1993)
    brier_k             mean over k of the Brier score of P(T <= k) on units whose outcome at k is known

Every probability entering a logarithm is clipped to [1e-15, 1 - 1e-15].
"""

from __future__ import annotations

import numpy as np

from nagahana.core.errors import InvariantViolation

_EPS = 1e-15
LOWER_IS_BETTER: dict[str, bool] = {
    "average_precision": False, "auroc": False, "log_loss": True, "balanced_log_loss": True, "brier": True,
    "macro_f1": False, "top1": False, "survival_nll": True, "brier_k": True, "mse": True,
}


def _binary_inputs(y: np.ndarray, s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=np.float64).ravel()
    s = np.asarray(s, dtype=np.float64).ravel()
    if y.shape != s.shape:
        raise InvariantViolation("labels and scores must have the same length")
    if np.any((y != 0) & (y != 1)):
        raise InvariantViolation("binary labels must be 0 or 1")
    return y, s


def average_precision(y: np.ndarray, s: np.ndarray) -> float:
    """Step-interpolated area under the precision-recall curve (module docstring); NaN without positives."""
    y, s = _binary_inputs(y, s)
    p = y.sum()
    if p == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    ys, ss = y[order], s[order]
    tp = np.cumsum(ys)
    fp = np.cumsum(1.0 - ys)
    last = np.r_[np.flatnonzero(np.diff(ss) != 0), ys.size - 1]          # last index of each distinct score
    tp, fp = tp[last], fp[last]
    precision = tp / (tp + fp)
    recall = tp / p
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def auroc(y: np.ndarray, s: np.ndarray) -> float:
    """Area under the ROC curve by the rank-sum identity with mid-ranks for ties; NaN with one class."""
    y, s = _binary_inputs(y, s)
    n1 = y.sum()
    n0 = y.size - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ss = s[order]
    ranks = np.empty(s.size, dtype=np.float64)
    # mid-ranks: positions 1 ... n, averaged over runs of equal scores
    boundaries = np.r_[0, np.flatnonzero(np.diff(ss) != 0) + 1, s.size]
    for a, b in zip(boundaries[:-1], boundaries[1:], strict=True):
        ranks[order[a:b]] = 0.5 * (a + 1 + b)
    u = ranks[y == 1].sum() - n1 * (n1 + 1) / 2.0
    return float(u / (n1 * n0))


def log_loss(y: np.ndarray, p: np.ndarray) -> float:
    y, p = _binary_inputs(y, p)
    p = np.clip(p, _EPS, 1.0 - _EPS)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log1p(-p)))


def balanced_log_loss(y: np.ndarray, p: np.ndarray) -> float:
    y, p = _binary_inputs(y, p)
    p = np.clip(p, _EPS, 1.0 - _EPS)
    parts = [float(-np.mean(np.log(p[y == 1]))) if (y == 1).any() else np.nan,
             float(-np.mean(np.log1p(-p[y == 0]))) if (y == 0).any() else np.nan]
    return float(np.nanmean(parts))


def brier(y: np.ndarray, p: np.ndarray) -> float:
    y, p = _binary_inputs(y, p)
    return float(np.mean((p - y) ** 2))


def binary_score(name: str, y: np.ndarray, p: np.ndarray) -> float:
    """Dispatch a binary criterion by name."""
    fns = {"average_precision": average_precision, "auroc": auroc, "log_loss": log_loss,
           "balanced_log_loss": balanced_log_loss, "brier": brier}
    if name not in fns:
        raise InvariantViolation(f"unknown binary criterion {name!r}")
    return fns[name](y, p)


def multiclass_score(name: str, y: np.ndarray, probs: np.ndarray) -> float:
    """Multiclass criteria of the module docstring; y int [n] in 0 ... S-1, probs [n, S]."""
    y = np.asarray(y, dtype=np.int64)
    probs = np.asarray(probs, dtype=np.float64)
    if probs.ndim != 2 or y.shape != (probs.shape[0],) or np.any((y < 0) | (y >= probs.shape[1])):
        raise InvariantViolation("multiclass scoring needs y [n] in 0 ... S-1 and probs [n, S]")
    if y.size == 0:
        return float("nan")
    py = np.clip(probs[np.arange(y.size), y], _EPS, 1.0)
    if name == "log_loss":
        return float(-np.mean(np.log(py)))
    if name == "balanced_log_loss":
        return float(np.mean([-np.mean(np.log(py[y == c])) for c in np.unique(y)]))
    pred = np.argmax(probs, axis=1)
    if name == "top1":
        return float(np.mean(pred == y))
    if name == "macro_f1":
        f1s = []
        for c in np.unique(y):
            tp = float(np.sum((pred == c) & (y == c)))
            fp = float(np.sum((pred == c) & (y != c)))
            fn = float(np.sum((pred != c) & (y == c)))
            f1s.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
        return float(np.mean(f1s))
    raise InvariantViolation(f"unknown multiclass criterion {name!r}")


def survival_nll(hazard: np.ndarray, event_step: np.ndarray, observed_steps: np.ndarray) -> float:
    """Mean discrete-time survival negative log-likelihood (module docstring).

    hazard [m, K]; event_step [m] (1 ... K, or 0 when no event was observed); observed_steps [m].
    Units with neither an event nor an observed step carry no information and are left out.
    """
    h = np.clip(np.asarray(hazard, dtype=np.float64), _EPS, 1.0 - _EPS)
    e = np.asarray(event_step, dtype=np.int64)
    c = np.asarray(observed_steps, dtype=np.int64)
    m, k = h.shape
    steps = np.arange(1, k + 1)[None, :]
    event = e > 0
    survived = np.where(event[:, None], steps < e[:, None], steps <= c[:, None])
    nll = -(np.log1p(-h) * survived).sum(axis=1)
    hit = event[:, None] & (steps == e[:, None])
    nll -= (np.log(h) * hit).sum(axis=1)
    informative = event | (c > 0)
    if not informative.any():
        return float("nan")
    return float(np.mean(nll[informative]))


def brier_k(p_inf: np.ndarray, event_step: np.ndarray, observed_steps: np.ndarray) -> float:
    """Mean over k of the Brier score of P(T <= k) on the units whose outcome at k is known."""
    p = np.asarray(p_inf, dtype=np.float64)
    e = np.asarray(event_step, dtype=np.int64)
    c = np.asarray(observed_steps, dtype=np.int64)
    out = []
    for k in range(1, p.shape[1] + 1):
        happened = (e > 0) & (e <= k)
        known = happened | (c >= k)
        if known.any():
            out.append(float(np.mean((p[known, k - 1] - happened[known]) ** 2)))
    return float(np.mean(out)) if out else float("nan")


__all__ = ["LOWER_IS_BETTER", "auroc", "average_precision", "balanced_log_loss", "binary_score", "brier", "brier_k",
           "log_loss", "multiclass_score", "survival_nll"]
