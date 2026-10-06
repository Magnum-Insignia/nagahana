"""Operating thresholds chosen on validation units only. A unit alerts when its score is >= the threshold.

Rules

    max_f1      the cut that maximises F1 = 2 TP / (2 TP + FP + FN) over every distinct validation score,
                ties broken towards the higher threshold (fewer alerts); the threshold is the midpoint
                between the chosen score and the next lower distinct score, so the validation decisions are
                exactly those of the chosen cut
    youden      the cut that maximises Youden's J = TPR - FPR (Youden, "Index for rating diagnostic
                tests", Cancer 3(1), 1950), with the same tie rule and midpoint
    fixed_fpr   the split-conformal threshold for a false-positive rate alpha on benign validation units:
                with benign scores s_1 ... s_n and k = ceil((n + 1)(1 - alpha)), alert when s > s_(k), the
                k-th smallest benign score. For benign units exchangeable with the calibration ones,
                P(s_new > s_(k)) <= alpha (Vovk, Gammerman and Shafer, Algorithmic Learning in a Random World,
                Springer 2005; Angelopoulos and Bates, arXiv:2107.07511, Section 1). The stored threshold is
                the next float above s_(k), so the rule "score >= threshold" is the rule "score > s_(k)".
                When k > n the guarantee cannot be met with n benign units; the threshold is then 1.0 and the
                choice is flagged as not attainable.

The model's own operating point (ThresholdConfig.rule, max_f1 by default) and the conformal threshold at
the configured alpha are both kept by every detector, so results can be reported at each method's own
threshold and at a common false-positive rate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from nagahana.core.errors import InvariantViolation


@dataclass(frozen=True)
class ThresholdChoice:
    """A chosen threshold and what it achieved on the validation units."""

    rule: str
    value: float
    criterion: float        # F1, J, or the achieved validation false-positive rate
    n_pos: int
    n_neg: int
    attainable: bool = True
    alpha: float | None = None

    def as_dict(self) -> dict[str, object]:
        return {"rule": self.rule, "value": self.value, "criterion": self.criterion, "n_pos": self.n_pos,
                "n_neg": self.n_neg, "attainable": self.attainable, "alpha": self.alpha}


def _check(score: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    s = np.asarray(score, dtype=np.float64).ravel()
    yy = np.asarray(y, dtype=np.int64).ravel()
    if s.shape != yy.shape:
        raise InvariantViolation("scores and labels must have equal lengths")
    if np.any((yy != 0) & (yy != 1)):
        raise InvariantViolation("threshold selection needs labels 0 or 1 (drop unknown units first)")
    if not np.all(np.isfinite(s)) or (s.size and (s.min() < 0.0 or s.max() > 1.0)):
        raise InvariantViolation("threshold selection needs probabilities in [0, 1]")
    return s, yy


def _sweep(s: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(distinct scores descending, TP, FP) of the cuts 'score >= distinct score'."""
    order = np.argsort(-s, kind="mergesort")
    ss, ys = s[order], y[order]
    tp = np.cumsum(ys)
    fp = np.cumsum(1 - ys)
    last = np.r_[np.flatnonzero(np.diff(ss) != 0), ss.size - 1]
    return ss[last], tp[last].astype(np.float64), fp[last].astype(np.float64)


def _midpoint(values: np.ndarray, j: int) -> float:
    return float(values[j]) if j + 1 >= values.size else 0.5 * float(values[j] + values[j + 1])


def max_f1_threshold(score: np.ndarray, y: np.ndarray) -> ThresholdChoice:
    s, yy = _check(score, y)
    p = int(yy.sum())
    n = int(yy.size - p)
    if p == 0:
        raise InvariantViolation("max-F1 needs at least one positive validation unit")
    v, tp, fp = _sweep(s, yy)
    f1 = 2 * tp / (2 * tp + fp + (p - tp))
    j = int(np.argmax(f1))                       # first maximum: the highest threshold among ties
    return ThresholdChoice("max_f1", _midpoint(v, j), float(f1[j]), p, n)


def youden_threshold(score: np.ndarray, y: np.ndarray) -> ThresholdChoice:
    s, yy = _check(score, y)
    p = int(yy.sum())
    n = int(yy.size - p)
    if p == 0 or n == 0:
        raise InvariantViolation("Youden's J needs both classes among the validation units")
    v, tp, fp = _sweep(s, yy)
    j_stat = tp / p - fp / n
    j = int(np.argmax(j_stat))
    return ThresholdChoice("youden", _midpoint(v, j), float(j_stat[j]), p, n)


def conformal_fpr_threshold(score: np.ndarray, y: np.ndarray, alpha: float) -> ThresholdChoice:
    s, yy = _check(score, y)
    if not 0.0 < alpha < 1.0:
        raise InvariantViolation("alpha must lie in (0, 1)")
    benign = np.sort(s[yy == 0])
    n = int(benign.size)
    p = int(yy.sum())
    if n == 0:
        raise InvariantViolation("the conformal threshold needs benign validation units")
    k = math.ceil((n + 1) * (1.0 - alpha))
    if k > n:
        return ThresholdChoice("fixed_fpr", 1.0, float(np.mean(benign >= 1.0)), p, n, attainable=False, alpha=alpha)
    q = float(benign[k - 1])
    value = float(np.nextafter(q, np.inf))
    attainable = value <= 1.0
    value = min(value, 1.0)
    return ThresholdChoice("fixed_fpr", value, float(np.mean(benign >= value)), p, n, attainable=attainable, alpha=alpha)


def select_threshold(score: np.ndarray, y: np.ndarray, rule: str, *, alpha: float) -> ThresholdChoice:
    """Dispatch by rule name (module docstring)."""
    if rule == "max_f1":
        return max_f1_threshold(score, y)
    if rule == "youden":
        return youden_threshold(score, y)
    if rule == "fixed_fpr":
        return conformal_fpr_threshold(score, y, alpha)
    raise InvariantViolation(f"unknown threshold rule {rule!r}")


__all__ = ["ThresholdChoice", "conformal_fpr_threshold", "max_f1_threshold", "select_threshold", "youden_threshold"]
