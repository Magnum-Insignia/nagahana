"""Detection metrics required by the problem statement, plus the error rates the owner emphasises.

The problem statement asks for "F1 score, precision, recall, false positive rate" against a
logistic-regression baseline trained on the same features. The owner also targets *both* false
positives and false negatives in operational settings, so FNR and the total detection error are
reported too.

From the confusion counts TP, FP, TN, FN:

    precision = TP / (TP + FP)          recall (TPR) = TP / (TP + FN)
    F1        = 2·P·R / (P + R)         FPR          = FP / (FP + TN)
    FNR       = FN / (FN + TP) = 1 − recall
    detection error = (FP + FN) / (TP + FP + TN + FN)

Zero denominators give NaN (undefined), never 0. A detector that never fires has *undefined*
precision, not "precision 0". Reporting NaN keeps that visible.

Operational note: at realistic base rates, a small FPR still means many false alerts (Sommer & Paxson,
IEEE S&P 2010; Arp et al., USENIX Security 2022 "base rate fallacy"). Always report FPR together
with the base rate of the evaluation data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Confusion:
    """Confusion counts for a binary decision (positive = malicious / infiltration)."""

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
    return _ratio(c.tp, c.tp + c.fp)


def recall(c: Confusion) -> float:
    return _ratio(c.tp, c.tp + c.fn)


def f1(c: Confusion) -> float:
    p, r = precision(c), recall(c)
    if math.isnan(p) or math.isnan(r) or p + r == 0:
        return math.nan
    return 2 * p * r / (p + r)


def fpr(c: Confusion) -> float:
    return _ratio(c.fp, c.fp + c.tn)


def fnr(c: Confusion) -> float:
    return _ratio(c.fn, c.fn + c.tp)


def detection_error(c: Confusion) -> float:
    return _ratio(c.fp + c.fn, c.n)


def base_rate(c: Confusion) -> float:
    """Fraction of truly positive items: report it next to FPR."""
    return _ratio(c.tp + c.fn, c.n)


def report(c: Confusion) -> dict[str, float]:
    """All the problem-statement metrics plus FNR, detection error and base rate."""
    return {
        "precision": precision(c), "recall": recall(c), "f1": f1(c), "fpr": fpr(c), "fnr": fnr(c),
        "detection_error": detection_error(c), "base_rate": base_rate(c),
    }
