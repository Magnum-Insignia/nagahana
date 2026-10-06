"""Threshold-dependent detection metrics: the problem-statement set and the operational error rates.

The problem statement asks for F1 score, precision, recall and false-positive rate against a
logistic-regression baseline trained on the same features. Operations pay for missed intrusions and
false alarms alike, so the false-negative rate, the detection error and the base rate are reported
with them (docs/architecture.md section 7). From the confusion counts TP, FP, TN, FN:

    precision   = TP / (TP + FP)              recall (TPR) = TP / (TP + FN)
    specificity = TN / (TN + FP)              NPV          = TN / (TN + FN)
    F1          = 2 P R / (P + R)             F_beta       = (1 + beta^2) P R / (beta^2 P + R)
    FPR         = FP / (FP + TN)              FNR          = FN / (FN + TP) = 1 - recall
    DE          = FP + FN                     DE rate      = (FP + FN) / (TP + FP + TN + FN)
    accuracy    = (TP + TN) / N               balanced accuracy = (recall + specificity) / 2
    MCC         = (TP TN - FP FN) / sqrt((TP + FP)(TP + FN)(TN + FP)(TN + FN))
    base rate   = (TP + FN) / N               alert rate   = (TP + FP) / N

A zero denominator gives NaN (undefined), never 0. A detector that never fires has an undefined
precision, and a split without positives has an undefined recall. MCC is undefined when any margin is
zero; Chicco and Jurman (BMC Genomics 21:6, 2020, doi:10.1186/s12864-019-6413-7) discuss MCC as the
summary that stays informative under class imbalance. Balanced accuracy is reported instead of plain
accuracy where attacks are rare, because accuracy at a base rate of a few per cent rewards a detector
that flags nothing.

At realistic base rates a small FPR still produces many false alerts (Sommer and Paxson, IEEE S&P
2010, doi:10.1109/SP.2010.25; Arp et al., USENIX Security 2022, "Dos and Don'ts of Machine Learning in
Computer Security"), so the FPR is always reported with the base rate of its split.

Two interfaces are provided. `Confusion` and the functions on it work on PyTorch 0/1 tensors and are
used by the training code. `counts` and `rates` work on NumPy arrays with unit weights (a weight vector
or a weight matrix [B, n], see `_arrays`), which is how every confidence interval of this package is
computed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from nagahana.evaluation._arrays import as_binary, as_float, safe_ratio, unbatch, weight_matrix


@dataclass(frozen=True)
class Confusion:
    """Confusion counts for a binary decision (positive = malicious or infiltration)."""

    tp: int
    fp: int
    tn: int
    fn: int

    @classmethod
    def from_predictions(cls, y_true: torch.Tensor, y_pred: torch.Tensor) -> Confusion:
        """Counts from 0/1 tensors of equal shape."""
        if y_true.shape != y_pred.shape:
            raise ValueError("y_true and y_pred must have the same shape")
        t, p = y_true.bool(), y_pred.bool()
        return cls(int((t & p).sum()), int((~t & p).sum()), int((~t & ~p).sum()), int((t & ~p).sum()))

    @property
    def n(self) -> int:
        """Total number of items."""
        return self.tp + self.fp + self.tn + self.fn


def _ratio(a: float, b: float) -> float:
    return a / b if b else math.nan


def precision(c: Confusion) -> float:
    """TP / (TP + FP); NaN when nothing was flagged."""
    return _ratio(c.tp, c.tp + c.fp)


def recall(c: Confusion) -> float:
    """TP / (TP + FN); NaN when there are no positives."""
    return _ratio(c.tp, c.tp + c.fn)


def f1(c: Confusion) -> float:
    """2 P R / (P + R); NaN when precision or recall is undefined or both are 0."""
    p, r = precision(c), recall(c)
    if math.isnan(p) or math.isnan(r) or p + r == 0:
        return math.nan
    return 2 * p * r / (p + r)


def fpr(c: Confusion) -> float:
    """FP / (FP + TN); NaN when there are no negatives."""
    return _ratio(c.fp, c.fp + c.tn)


def fnr(c: Confusion) -> float:
    """FN / (FN + TP); NaN when there are no positives."""
    return _ratio(c.fn, c.fn + c.tp)


def detection_error(c: Confusion) -> float:
    """(FP + FN) / N, the detection-error rate."""
    return _ratio(c.fp + c.fn, c.n)


def base_rate(c: Confusion) -> float:
    """Fraction of truly positive items: reported next to every FPR."""
    return _ratio(c.tp + c.fn, c.n)


def report(c: Confusion) -> dict[str, float]:
    """The problem-statement metrics plus FNR, detection-error rate and base rate."""
    return {
        "precision": precision(c), "recall": recall(c), "f1": f1(c), "fpr": fpr(c), "fnr": fnr(c),
        "detection_error": detection_error(c), "base_rate": base_rate(c),
    }


@dataclass(frozen=True)
class Counts:
    """Weighted confusion counts; each field is an array [B] (one entry per weight row)."""

    tp: np.ndarray
    fp: np.ndarray
    tn: np.ndarray
    fn: np.ndarray

    @property
    def n(self) -> np.ndarray:
        """Total weight per row."""
        return self.tp + self.fp + self.tn + self.fn


def counts(label: Any, decision: Any, weights: Any = None) -> Counts:
    """Weighted confusion counts of 0/1 decisions against 0/1 labels.

    label, decision: [n] in {0, 1}; weights: None, [n] or [B, n]. Returns a `Counts` whose fields are
    arrays [B] (B = 1 when weights is None or a vector).
    """
    y = as_binary("label", label)
    d = as_binary("decision", decision)
    if y.shape != d.shape:
        raise ValueError("label and decision must have the same shape")
    w, _ = weight_matrix(weights, y.size)
    yf, df = y.astype(np.float64), d.astype(np.float64)
    # Each count is a weighted sum of a per-unit indicator: [B, n] @ [n] -> [B].
    tp = w @ (yf * df)
    fp = w @ ((1.0 - yf) * df)
    tn = w @ ((1.0 - yf) * (1.0 - df))
    fn = w @ (yf * (1.0 - df))
    return Counts(tp=tp, fp=fp, tn=tn, fn=fn)


#: Names returned by `rates`, in report order.
RATE_NAMES: tuple[str, ...] = (
    "precision", "recall", "f1", "fpr", "fnr", "specificity", "npv", "accuracy", "balanced_accuracy",
    "mcc", "detection_error", "detection_error_rate", "base_rate", "alert_rate",
)


def rates(c: Counts, *, beta: float | None = None) -> dict[str, np.ndarray]:
    """All ratio metrics of weighted counts, each an array [B]; NaN where undefined.

    `detection_error` is the count FP + FN (thesis eq. for DE); `detection_error_rate` divides it by N.
    With `beta`, the key "f_beta" is added.
    """
    tp, fp, tn, fn = c.tp, c.fp, c.tn, c.fn
    n = tp + fp + tn + fn
    prec = safe_ratio(tp, tp + fp)
    rec = safe_ratio(tp, tp + fn)
    spec = safe_ratio(tn, tn + fp)
    # F1 = 2TP / (2TP + FP + FN) equals 2PR/(P+R) where both are defined; it is undefined when either
    # precision or recall is undefined, and when TP = 0 with both defined it is 0.
    f1_v = np.where(np.isnan(prec) | np.isnan(rec), np.nan, safe_ratio(2.0 * tp, 2.0 * tp + fp + fn))
    denom = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    mcc = safe_ratio(tp * tn - fp * fn, np.sqrt(denom))
    out: dict[str, np.ndarray] = {
        "precision": prec,
        "recall": rec,
        "f1": f1_v,
        "fpr": safe_ratio(fp, fp + tn),
        "fnr": safe_ratio(fn, fn + tp),
        "specificity": spec,
        "npv": safe_ratio(tn, tn + fn),
        "accuracy": safe_ratio(tp + tn, n),
        "balanced_accuracy": 0.5 * (rec + spec),
        "mcc": mcc,
        "detection_error": fp + fn,
        "detection_error_rate": safe_ratio(fp + fn, n),
        "base_rate": safe_ratio(tp + fn, n),
        "alert_rate": safe_ratio(tp + fp, n),
    }
    if beta is not None:
        if not beta > 0:
            raise ValueError("beta must be positive")
        b2 = beta * beta
        fb = safe_ratio((1.0 + b2) * tp, (1.0 + b2) * tp + b2 * fn + fp)
        out["f_beta"] = np.where(np.isnan(prec) | np.isnan(rec), np.nan, fb)
    return out


def decisions(score: Any, threshold: float) -> np.ndarray:
    """0/1 decisions: alert when score >= threshold (the convention of every operating point here)."""
    s = as_float("score", score, 1, finite=False)
    if np.isnan(s).any():
        raise ValueError("score contains NaN")
    return (s >= threshold).astype(np.int64)


def binary_report(score: Any, label: Any, threshold: float, weights: Any = None) -> dict[str, Any]:
    """Every rate of `rates` for the decisions score >= threshold (floats, or arrays [B] when batched)."""
    y = as_binary("label", label)
    w, batched = weight_matrix(weights, y.size)
    c = counts(y, decisions(score, threshold), w)
    return {k: unbatch(np.asarray(v), batched) for k, v in rates(c).items()}
