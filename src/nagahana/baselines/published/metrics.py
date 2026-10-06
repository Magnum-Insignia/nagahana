"""Metric hooks: the reproduced papers' metrics computed on ModelOutputs, beside the printed values.

Definitions (positive class = malicious; unknown labels, code -1, are excluded)

    precision = TP / (TP + FP)        recall = DR = TPR = TP / (TP + FN)
    F1 = 2 P R / (P + R)              FPR = FAR = FP / (FP + TN)          FNR = FN / (FN + TP)
    accuracy = (TP + TN) / N          balanced accuracy = (TPR + TNR) / 2
    MCC = (TP TN - FP FN) / sqrt((TP + FP)(TP + FN)(TN + FP)(TN + FN))
    macro_x = mean of x over the classes (for binary: over benign and malicious, as averaged by
              Caville et al. 2022 and, by the notes' reading, Neto et al. 2023)
    weighted_x = support-weighted mean of x over the classes
    AUROC = P(score of a random positive > score of a random negative), ties counted 1/2
            (the Mann-Whitney statistic, computed from average ranks)
    AUPRC = average precision, sum_k (R_k - R_{k-1}) P_k over distinct score thresholds (the step
            interpolation of scikit-learn's average_precision_score)
    top-k accuracy = fraction of units whose true class is among the k most probable

A ratio with a zero denominator is NaN (undefined), never 0, the convention of evaluation/metrics.py.

Comparison with the printed values (AS-531)
-------------------------------------------
`compare_with_reported` evaluates every numeric metric of a `ReportedResult` on a model's outputs with
the definition above and returns one row per metric: the value as printed, its parsed fraction, the
reproduced value and their difference. Metric names in a ReportedResult may use the papers' own
abbreviations; METRIC_ALIASES maps them (DR -> recall, FAR -> fpr, AUC -> auroc, AP -> auprc).
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
import pandas as pd

from nagahana.baselines.published.base import ReportedResult
from nagahana.core.errors import InvariantViolation
from nagahana.evaluation.predictions import ModelOutputs

METRIC_ALIASES: dict[str, str] = {
    "dr": "recall", "tpr": "recall", "sensitivity": "recall", "r": "recall",
    "far": "fpr", "fp_rate": "fpr",
    "p": "precision", "auc": "auroc", "roc_auc": "auroc", "ap": "auprc", "average_precision": "auprc",
    "f_score": "f1", "f1_score": "f1", "micro_precision": "top1_next_event",
    # The ROC area of hard 0/1 decisions is (TPR + TNR) / 2, the balanced accuracy (Sarhan et al. 2022).
    "auc_of_decisions": "balanced_accuracy",
}


def canonical_metric(name: str) -> str:
    """Canonical metric name for a paper's abbreviation (case-insensitive)."""
    key = name.strip().lower()
    return METRIC_ALIASES.get(key, key)


def _ratio(a: float, b: float) -> float:
    return a / b if b else math.nan


def confusion_counts(label: np.ndarray, decision: np.ndarray) -> tuple[int, int, int, int]:
    """(TP, FP, TN, FN) of 0/1 arrays."""
    t = np.asarray(label).astype(bool)
    p = np.asarray(decision).astype(bool)
    return int((t & p).sum()), int((~t & p).sum()), int((~t & ~p).sum()), int((t & ~p).sum())


def average_ranks(x: np.ndarray) -> np.ndarray:
    """Ranks 1 ... n with ties given their average rank (as scipy.stats.rankdata(method="average"))."""
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    ranks = np.empty(x.shape[0], dtype=np.float64)
    # Boundaries of runs of equal values in the sorted array.
    starts = np.r_[0, np.nonzero(np.diff(xs))[0] + 1]
    ends = np.r_[starts[1:], xs.shape[0]]
    avg = (starts + ends + 1) / 2.0                     # mean of ranks start+1 ... end
    ranks[order] = np.repeat(avg, ends - starts)
    return ranks


def roc_auc(score: np.ndarray, label: np.ndarray) -> float:
    """AUROC by the Mann-Whitney statistic; NaN when one class is absent."""
    y = np.asarray(label).astype(bool)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return math.nan
    r = average_ranks(score)
    return float((r[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def average_precision(score: np.ndarray, label: np.ndarray) -> float:
    """Average precision (step interpolation over distinct thresholds); NaN without positives."""
    y = np.asarray(label).astype(bool)
    n_pos = int(y.sum())
    if n_pos == 0:
        return math.nan
    order = np.argsort(-np.asarray(score, dtype=np.float64), kind="mergesort")
    s, t = np.asarray(score, dtype=np.float64)[order], y[order]
    tp = np.cumsum(t)
    fp = np.cumsum(~t)
    # Keep the last index of each run of equal scores: one point per distinct threshold.
    last = np.r_[np.nonzero(np.diff(s))[0], s.shape[0] - 1]
    tp, fp = tp[last].astype(np.float64), fp[last].astype(np.float64)
    precision = tp / (tp + fp)
    recall = tp / n_pos
    prev = np.r_[0.0, recall[:-1]]
    return float(np.sum((recall - prev) * precision))


def binary_metrics(score: np.ndarray, label: np.ndarray, *, threshold: float) -> dict[str, float]:
    """Every binary metric of the module docstring at decision rule score >= threshold."""
    score = np.asarray(score, dtype=np.float64)
    label = np.asarray(label, dtype=np.int64)
    if score.shape != label.shape:
        raise InvariantViolation("score and label must have the same shape")
    keep = label >= 0
    s, y = score[keep], label[keep].astype(bool)
    decision = s >= threshold
    tp, fp, tn, fn = confusion_counts(y, decision)
    n = tp + fp + tn + fn
    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    f1 = math.nan if (math.isnan(precision) or math.isnan(recall) or precision + recall == 0) else 2 * precision * recall / (precision + recall)
    npv = _ratio(tn, tn + fn)
    tnr = _ratio(tn, tn + fp)
    f1_neg = math.nan if (math.isnan(npv) or math.isnan(tnr) or npv + tnr == 0) else 2 * npv * tnr / (npv + tnr)
    den = math.sqrt(float(tp + fp) * float(tp + fn) * float(tn + fp) * float(tn + fn))
    mcc = (tp * tn - fp * fn) / den if den else math.nan
    pos_share = _ratio(tp + fn, n)
    out = {
        "n": float(n), "tp": float(tp), "fp": float(fp), "tn": float(tn), "fn": float(fn),
        "accuracy": _ratio(tp + tn, n), "precision": precision, "recall": recall, "f1": f1,
        "fpr": _ratio(fp, fp + tn), "fnr": _ratio(fn, fn + tp), "tnr": tnr,
        "balanced_accuracy": (recall + tnr) / 2.0 if not (math.isnan(recall) or math.isnan(tnr)) else math.nan,
        "mcc": mcc, "detection_error": _ratio(fp + fn, n), "base_rate": pos_share,
        "macro_precision": float(np.mean([precision, npv])), "macro_recall": float(np.mean([recall, tnr])),
        "macro_f1": float(np.mean([f1, f1_neg])),
        "auroc": roc_auc(s, y), "auprc": average_precision(s, y),
    }
    if not math.isnan(pos_share):
        out["weighted_f1"] = pos_share * f1 + (1.0 - pos_share) * f1_neg if not (math.isnan(f1) or math.isnan(f1_neg)) else math.nan
    return out


def multiclass_metrics(probs: np.ndarray, label: np.ndarray) -> dict[str, float]:
    """Accuracy and macro / weighted precision, recall and F1 of the argmax decision.

    The averaging follows scikit-learn's precision_recall_fscore_support, with which the reproduced
    papers computed their values: classes are those present in the true or the predicted labels, and a
    ratio with a zero denominator counts as 0 (scikit-learn's zero_division default). Macro means are
    unweighted over those classes; weighted means use the true support.
    """
    probs = np.asarray(probs, dtype=np.float64)
    label = np.asarray(label, dtype=np.int64)
    keep = label >= 0
    p, y = probs[keep], label[keep]
    if p.ndim != 2 or y.shape != (p.shape[0],):
        raise InvariantViolation("probs must be [n, C] and label [n]")
    if y.size == 0:
        return {k: math.nan for k in ("n", "accuracy", "macro_precision", "macro_recall", "macro_f1",
                                      "weighted_precision", "weighted_recall", "weighted_f1")}
    c = p.shape[1]
    pred = p.argmax(axis=1)
    support = np.bincount(y, minlength=c).astype(np.float64)
    predicted = np.bincount(pred, minlength=c).astype(np.float64)
    tp = np.bincount(y[pred == y], minlength=c).astype(np.float64)
    prec = np.divide(tp, predicted, out=np.zeros(c), where=predicted > 0)
    rec = np.divide(tp, support, out=np.zeros(c), where=support > 0)
    f1 = np.divide(2.0 * prec * rec, prec + rec, out=np.zeros(c), where=(prec + rec) > 0)
    present = (support > 0) | (predicted > 0)
    w = support / support.sum()
    return {
        "n": float(y.shape[0]), "accuracy": float((pred == y).mean()),
        "macro_precision": float(prec[present].mean()), "macro_recall": float(rec[present].mean()),
        "macro_f1": float(f1[present].mean()),
        "weighted_precision": float(np.sum(w * prec)), "weighted_recall": float(np.sum(w * rec)),
        "weighted_f1": float(np.sum(w * f1)),
    }


def topk_accuracy(probs: np.ndarray, label: np.ndarray, k: int) -> float:
    """Fraction of units (label >= 0) whose true class is among the k most probable classes."""
    probs = np.asarray(probs, dtype=np.float64)
    label = np.asarray(label, dtype=np.int64)
    keep = label >= 0
    if not keep.any():
        return math.nan
    p, y = probs[keep], label[keep]
    k = min(int(k), p.shape[1])
    # Stable ranking: a class with a higher score, or an equal score and a lower index, ranks first.
    top = np.argsort(-p, axis=1, kind="mergesort")[:, :k]
    return float(np.mean(np.any(top == y[:, None], axis=1)))


def topk_codes_accuracy(codes: np.ndarray, label: np.ndarray, k: int) -> float:
    """Top-k accuracy from precomputed ranked class codes [n, >= k] (for vocabularies too large to store)."""
    codes = np.asarray(codes, dtype=np.int64)
    label = np.asarray(label, dtype=np.int64)
    keep = label >= 0
    if not keep.any():
        return math.nan
    return float(np.mean(np.any(codes[keep, : int(k)] == label[keep, None], axis=1)))


def outputs_metrics(outputs: ModelOutputs, *, threshold: float | None = None) -> dict[str, float]:
    """All metrics computable from a ModelOutputs bundle.

    Detection metrics use the record's operating threshold (or `threshold`, or 0.5). Multi-class metrics
    use component["class_probs"] and component["class_label"]. Next-stage, next-observation and
    next-event metrics use the components the forecasting reproductions write
    (`<target>_probs` / `<target>_label`, or `next_event_topk` / `next_event_label`).
    """
    out: dict[str, float] = {}
    det = outputs.detection
    if det is not None:
        thr = threshold if threshold is not None else (det.threshold if det.threshold is not None else 0.5)
        out.update(binary_metrics(det.score, det.label, threshold=float(thr)))
    comp = outputs.component
    if "class_probs" in comp and "class_label" in comp:
        mc = multiclass_metrics(comp["class_probs"], comp["class_label"])
        out.update({f"multiclass_{k}": v for k, v in mc.items()})
    for target in ("next_stage", "next_observation"):
        if f"{target}_probs" in comp and f"{target}_label" in comp:
            for k in (1, 2, 3):
                out[f"top{k}_{target}"] = topk_accuracy(comp[f"{target}_probs"], comp[f"{target}_label"], k)
            if "position" in comp:
                # Accuracy after the first n observations of each sequence (position n - 1), n = 1 ... 5.
                pos = np.asarray(comp["position"])
                for n_obs in range(1, 6):
                    sel = pos == n_obs - 1
                    if sel.any():
                        for k in (1, 2, 3):
                            out[f"top{k}_{target}_after_{n_obs}"] = topk_accuracy(
                                comp[f"{target}_probs"][sel], comp[f"{target}_label"][sel], k)
    if "decoded_prefix_match" in comp:
        # Share of sequences whose Viterbi decoding of the first n observations equals the true stages.
        match = np.asarray(comp["decoded_prefix_match"])
        for n_obs in range(1, match.shape[1] + 1):
            col = match[:, n_obs - 1]
            ok = col >= 0
            out[f"decoded_sequence_accuracy_after_{n_obs}"] = float(col[ok].mean()) if ok.any() else math.nan
    if "next_event_topk" in comp and "next_event_label" in comp:
        width = comp["next_event_topk"].shape[1]
        for k in range(1, min(width, 10) + 1):
            out[f"top{k}_next_event"] = topk_codes_accuracy(comp["next_event_topk"], comp["next_event_label"], k)
    if outputs.stage is not None:
        st = outputs.stage
        for k in (1, 2, 3):
            out[f"top{k}_stage"] = topk_accuracy(st.probs, st.label, k)
        sm = multiclass_metrics(st.probs, st.label)
        out["stage_macro_f1"] = sm["macro_f1"]
    return out


def compare_with_reported(outputs: ModelOutputs, reported: ReportedResult, *, threshold: float | None = None,
                          multiclass: bool | None = None) -> pd.DataFrame:
    """One row per numeric reported metric: printed text, parsed value, reproduced value, difference.

    With `multiclass` (default: True when the outputs carry class posteriors and the reported task is
    multi-class), accuracy, precision, recall and F1 names refer to the multi-class metrics.
    """
    ours = outputs_metrics(outputs, threshold=threshold)
    use_mc = multiclass if multiclass is not None else (reported.task.startswith("multiclass") and "multiclass_accuracy" in ours)
    rows = []
    for metric in reported.numeric_metrics():
        name = canonical_metric(metric)
        key = f"multiclass_{name}" if use_mc and f"multiclass_{name}" in ours else name
        value = ours.get(key, math.nan)
        rep = reported.value(metric)
        rows.append({"metric": metric, "canonical": key, "printed": reported.values[metric], "reported": rep,
                     "reproduced": value, "difference": value - rep if not math.isnan(value) else math.nan})
    return pd.DataFrame(rows, columns=["metric", "canonical", "printed", "reported", "reproduced", "difference"])


def mean_and_spread(per_split: list[Mapping[str, float]]) -> dict[str, dict[str, float]]:
    """Mean, sample standard deviation, minimum and maximum of each metric over splits or repeats."""
    keys = sorted({k for m in per_split for k in m})
    out: dict[str, dict[str, float]] = {}
    for k in keys:
        v = np.asarray([m.get(k, math.nan) for m in per_split], dtype=np.float64)
        v = v[~np.isnan(v)]
        if v.size == 0:
            out[k] = {"mean": math.nan, "std": math.nan, "min": math.nan, "max": math.nan, "count": 0.0}
            continue
        out[k] = {"mean": float(v.mean()), "std": float(v.std(ddof=1)) if v.size > 1 else 0.0,
                  "min": float(v.min()), "max": float(v.max()), "count": float(v.size)}
    return out
