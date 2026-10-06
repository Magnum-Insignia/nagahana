"""ATT&CK stage prediction: top-k accuracy, per-stage and averaged F1, confusion, ordinal errors, calibration.

Units are scored where the true stage is known (label >= 0). The predicted stage is the class of
largest probability (the first such class in code order when probabilities tie).

Top-k accuracy. With g the number of classes whose probability exceeds that of the true class and t
the number of classes tied with it (the true class included), a uniformly random order of the tie puts
the true class among the first k with probability clip((k - g) / t, 0, 1); this is the credit of the
unit, so the metric does not depend on how ties happen to be ordered.

F1. From the weighted confusion matrix C (rows true, columns predicted), per class c:
TP_c = C[c, c], FP_c = column sum - TP_c, FN_c = row sum - TP_c and F1_c = 2 TP_c / (2 TP_c + FP_c + FN_c),
which is 0 for a class that is predicted but never true or true but never predicted. Macro-F1 averages
F1_c over the evaluated classes with equal weight, so rare stages count as much as common ones
(thesis section on stages and paths); weighted F1 weights F1_c by support; micro-F1 equals accuracy for
single-label prediction. The evaluated classes are, by default, those that occur as a true or a
predicted class on the evaluated units ("union", the convention of scikit-learn's f1_score, Pedregosa et
al., JMLR 12:2825-2830, 2011); "present" uses the true classes only and "all" every class. The set is
fixed on the full sample and kept for every bootstrap resample, so the metric's definition does not
change between resamples.

Ordinal errors over the kill chain. Stages are ordered by their position in models/vocab.py STAGES (the
order of the ATT&CK matrix columns, "none" first, the order behind STAGE_PROGRESS). The ordinal mean
absolute error is the mean |pos(predicted) - pos(true)| in stage steps, split into the share of
predictions ahead of and behind the true stage. The ranked probability score over the ordered classes
(Epstein 1969; Murphy, Journal of Applied Meteorology 10:155-156, 1971),

    RPS = (1 / (S - 1)) sum_{c=0}^{S-2} (P(stage <= c) - 1[y <= c])^2,

is a proper score that penalises probability placed far from the true stage more than probability
placed near it.

Calibration of the stage posterior: top-label ECE, classwise ECE and the adaptive calibration error
(calibration.py), multiclass Brier and log score.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from nagahana.core.errors import InvariantViolation
from nagahana.evaluation._arrays import safe_ratio, unbatch, weight_matrix
from nagahana.evaluation.calibration import bin_sums
from nagahana.evaluation.predictions import StagePredictions


def scored(pred: StagePredictions) -> np.ndarray:
    """Boolean mask [n] of units with a known true stage."""
    return pred.label >= 0


def top_k_credit(probs: np.ndarray, label: np.ndarray, k: int) -> np.ndarray:
    """Expected top-k correctness [n] under uniformly random ordering of tied probabilities."""
    if k < 1:
        raise ValueError("k must be >= 1")
    py = probs[np.arange(label.size), label][:, None]
    greater = (probs > py).sum(axis=1)
    tied = (probs == py).sum(axis=1)
    return np.clip((k - greater) / tied, 0.0, 1.0)


def predicted_class(probs: np.ndarray) -> np.ndarray:
    """Class of largest probability (first in code order on ties) [n]."""
    return probs.argmax(axis=1)


def confusion(label: np.ndarray, predicted: np.ndarray, n_classes: int, weights: Any = None) -> np.ndarray:
    """Weighted confusion matrices [B, S, S] (rows true, columns predicted)."""
    w, _ = weight_matrix(weights, label.size)
    (flat,) = bin_sums(label * n_classes + predicted, n_classes * n_classes, w, None)
    return flat.reshape(w.shape[0], n_classes, n_classes)


def evaluated_classes(label: np.ndarray, predicted: np.ndarray, n_classes: int, rule: str) -> np.ndarray:
    """Class indices entering macro averages: rule "union", "present" or "all"."""
    if rule == "all":
        return np.arange(n_classes)
    present = np.zeros(n_classes, dtype=bool)
    present[np.unique(label)] = True
    if rule == "union":
        present[np.unique(predicted)] = True
    elif rule != "present":
        raise ValueError("rule must be 'union', 'present' or 'all'")
    return np.flatnonzero(present)


def kill_chain_positions(stage_names: tuple[str, ...]) -> np.ndarray:
    """Position of each class on the kill chain: its vocab code when all names are vocab stages, else its index."""
    # Imported here so that the evaluation package does not load the model vocabulary (and the data model
    # behind it) until stage metrics are computed.
    from nagahana.models.vocab import STAGE_CODE

    if all(name in STAGE_CODE for name in stage_names):
        return np.array([STAGE_CODE[name] for name in stage_names], dtype=np.float64)
    return np.arange(len(stage_names), dtype=np.float64)


def stage_metrics(pred: StagePredictions, weights: Any = None, *, top_k: tuple[int, ...] = (1, 3),
                  classes: str = "union", class_set: np.ndarray | None = None) -> dict[str, Any]:
    """Accuracy, top-k, macro/weighted/micro F1, ordinal MAE (ahead, behind), RPS, multiclass Brier.

    `weights` are over all n units; units without a true stage get weight 0. `class_set` fixes the
    classes of the macro average (computed with `classes` from the full sample when None).
    """
    return stage_metrics_arrays(pred.probs, pred.label, pred.stage_names, weights, top_k=top_k, classes=classes,
                                class_set=class_set)


def stage_metrics_arrays(probs: np.ndarray, label: np.ndarray, stage_names: tuple[str, ...], weights: Any = None, *,
                         top_k: tuple[int, ...] = (1, 3), classes: str = "union", class_set: np.ndarray | None = None
                         ) -> dict[str, Any]:
    """`stage_metrics` on arrays: probs [n, S], label [n] (-1 unknown), stage names of the S classes."""
    ok = label >= 0
    w, batched = weight_matrix(weights, ok.size)
    w = w * ok[None, :]
    n_cls = probs.shape[1]
    lab = np.where(ok, label, 0)
    yhat = predicted_class(probs)
    total = w.sum(axis=1)
    out: dict[str, np.ndarray] = {"n_units": total}
    for k in top_k:
        out[f"top{k}"] = safe_ratio(w @ top_k_credit(probs, lab, k), total)
    cm = confusion(lab, yhat, n_cls, w)                                 # [B, S, S]
    tp = np.einsum("bcc->bc", cm)
    fp = cm.sum(axis=1) - tp
    fn = cm.sum(axis=2) - tp
    f1c = safe_ratio(2.0 * tp, 2.0 * tp + fp + fn)                     # [B, S]
    cls = evaluated_classes(lab[ok], yhat[ok], n_cls, classes) if class_set is None else np.asarray(class_set)
    # A class of the fixed set that is neither true nor predicted in a resample has an undefined F1 there
    # and is left out of that resample's average (on the full sample every class of the set is defined).
    sub = f1c[:, cls]
    defined = np.isfinite(sub)
    out["macro_f1"] = safe_ratio(np.where(defined, sub, 0.0).sum(axis=1), defined.sum(axis=1))
    support = cm.sum(axis=2)[:, cls]
    out["weighted_f1"] = safe_ratio((np.nan_to_num(f1c[:, cls]) * support).sum(axis=1), support.sum(axis=1))
    out["accuracy"] = safe_ratio(tp.sum(axis=1), total)
    out["micro_f1"] = out["accuracy"]
    pos = kill_chain_positions(stage_names)
    gap = pos[yhat] - pos[lab]
    out["ordinal_mae"] = safe_ratio(w @ np.abs(gap), total)
    out["ahead_share"] = safe_ratio(w @ (gap > 0).astype(np.float64), total)
    out["behind_share"] = safe_ratio(w @ (gap < 0).astype(np.float64), total)
    order = np.argsort(pos, kind="stable")                              # classes in kill-chain order
    cum = np.cumsum(probs[:, order], axis=1)[:, :-1]                    # P(stage <= c), c = 0 .. S-2
    rank_true = np.argsort(order)[lab]                                  # true class position in that order
    obs = (np.arange(n_cls - 1)[None, :] >= rank_true[:, None]).astype(np.float64)
    rps = ((cum - obs) ** 2).sum(axis=1) / max(n_cls - 1, 1)
    out["rps"] = safe_ratio(w @ rps, total)
    onehot = np.zeros_like(probs)
    onehot[np.arange(lab.size), lab] = 1.0
    out["brier"] = safe_ratio(w @ ((probs - onehot) ** 2).sum(axis=1), total)
    return {name: unbatch(np.atleast_1d(v), batched) for name, v in out.items()}


def per_class_table(pred: StagePredictions, weights: Any = None) -> dict[str, np.ndarray]:
    """Per-class precision, recall, F1 and support on the scored units (single weight vector)."""
    ok = scored(pred)
    w, batched = weight_matrix(weights, ok.size)
    if batched:
        raise InvariantViolation("per_class_table takes a single weight vector")
    w = w * ok[None, :]
    lab = np.where(ok, pred.label, 0)
    cm = confusion(lab, predicted_class(pred.probs), pred.probs.shape[1], w)[0]
    tp = np.diag(cm)
    return {"precision": safe_ratio(tp, cm.sum(axis=0)), "recall": safe_ratio(tp, cm.sum(axis=1)),
            "f1": safe_ratio(2.0 * tp, cm.sum(axis=0) + cm.sum(axis=1)), "support": cm.sum(axis=1),
            "predicted": cm.sum(axis=0)}
