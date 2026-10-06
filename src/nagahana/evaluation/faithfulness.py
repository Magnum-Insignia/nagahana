"""Faithfulness of explanations: deletion and insertion curves, their areas and the gain over random orders.

An explanation ranks the input features of one prediction (field columns of the data model, explain/).
The deletion curve replaces the features with their baseline in that order, most important first, and
records the model output after each fraction t of the features is removed; the insertion curve starts
from the baseline input and restores the features in the same order (Petsiuk, Das and Saenko, "RISE:
Randomized Input Sampling for Explanation of Black-box Models", BMVC 2018, arXiv:1806.07421). A faithful
explanation makes the output fall fast under deletion and rise fast under insertion:

    AUC_del = integral_0^1 f(x_del(t)) dt          (lower is better)
    AUC_ins = integral_0^1 f(x_ins(t)) dt          (higher is better)

by the trapezoidal rule over the recorded fractions. The area over the perturbation curve (Samek,
Binder, Montavon, Lapuschkin and Muller, IEEE TNNLS 28:2660-2673, 2017),

    AOPC = (1 / L) sum_{l=1}^{L} (f(x) - f(x_del(t_l))),

averages the drop of the output over the L deletion steps after the first. Areas are also reported
relative to the unperturbed output f(x) = f(x_del(0)), which makes units with different outputs
comparable, and against random feature orders (the curves of the same deletion and insertion with
features in random order, averaged over orders):

    deletion gain  = AUC_del(random) - AUC_del,       insertion gain = AUC_ins - AUC_ins(random),

both positive when the explanation is better than chance. Units are weighted ([n] or [B, n]) for the
bootstrap like every metric of this package, and two explanation methods are compared on the same
units with paired differences.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from nagahana.evaluation._arrays import safe_ratio, unbatch, weight_matrix
from nagahana.evaluation.predictions import ExplanationCurves

FAITHFULNESS_NAMES: tuple[str, ...] = (
    "deletion_auc", "insertion_auc", "deletion_auc_relative", "insertion_auc_relative", "aopc", "deletion_gain",
    "insertion_gain")


def subset(e: ExplanationCurves, idx: np.ndarray) -> ExplanationCurves:
    """The record restricted to explained predictions idx (in that order)."""
    sel = np.asarray(idx, dtype=np.int64)
    rd = None if e.random_deletion is None else e.random_deletion[sel]
    ri = None if e.random_insertion is None else e.random_insertion[sel]
    return ExplanationCurves(method=e.method, fractions=e.fractions, deletion=e.deletion[sel], insertion=e.insertion[sel],
                             meta=e.meta.iloc[sel].reset_index(drop=True), random_deletion=rd, random_insertion=ri)


def curve_area(fractions: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Trapezoidal area under each row of values [n, F] over fractions [F]."""
    return np.trapezoid(values, fractions, axis=-1)


def unit_scores(e: ExplanationCurves) -> dict[str, np.ndarray]:
    """Per-unit areas, relative areas, AOPC and gains over random orders (NaN where not defined) [n]."""
    fr = e.fractions
    d_auc = curve_area(fr, e.deletion)
    i_auc = curve_area(fr, e.insertion)
    full = e.deletion[:, 0]
    out = {"deletion_auc": d_auc, "insertion_auc": i_auc,
           "deletion_auc_relative": safe_ratio(d_auc, full), "insertion_auc_relative": safe_ratio(i_auc, full),
           "aopc": (full[:, None] - e.deletion[:, 1:]).mean(axis=1)}
    if e.random_deletion is not None and e.random_insertion is not None:
        out["deletion_gain"] = curve_area(fr, e.random_deletion) - d_auc
        out["insertion_gain"] = i_auc - curve_area(fr, e.random_insertion)
    else:
        nan = np.full(d_auc.shape, np.nan)
        out["deletion_gain"], out["insertion_gain"] = nan, nan
    return out


def faithfulness_metrics(e: ExplanationCurves, weights: Any = None) -> dict[str, Any]:
    """Weighted means of the per-unit scores over the explained units (arrays [B] when batched)."""
    per = unit_scores(e)
    w, batched = weight_matrix(weights, len(e.meta))
    out = {}
    for name in FAITHFULNESS_NAMES:
        v = per[name]
        ok = np.isfinite(v)
        out[name] = unbatch(safe_ratio((w * ok) @ np.where(ok, v, 0.0), (w * ok).sum(axis=1)), batched)
    return out


def mean_curves(e: ExplanationCurves) -> dict[str, np.ndarray]:
    """Mean deletion and insertion curves over units (and the random-order curves when present) [F]."""
    out = {"fraction": e.fractions, "deletion": e.deletion.mean(axis=0), "insertion": e.insertion.mean(axis=0)}
    if e.random_deletion is not None and e.random_insertion is not None:
        out["random_deletion"] = e.random_deletion.mean(axis=0)
        out["random_insertion"] = e.random_insertion.mean(axis=0)
    return out
