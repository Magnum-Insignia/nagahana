"""Forensic backtesting on labelled incidents (protocol P7): onset timing, patient zero, narrative steps.

For each incident the forensic replay reports when each stage began, a ranking of candidate first
compromised entities and the steps of a reconstructed narrative (ForensicPredictions). The thesis
measures

    onset error        |t_hat_onset - t_onset| per (incident, stage) present in both, summarised by its
                       median; the share of annotated stages that received an onset (coverage)
    patient zero       accuracy at top-1 and top-k: the annotated first compromised entity is the first,
                       or among the first k, entities of the ranking
    narrative steps    precision and recall of the reconstructed steps against the annotated steps

Narrative matching. A reconstructed step and an annotated step of the same incident can match when
their stages are equal, their entities match (equal, or an annotated entity of -1 that the annotation
leaves open) and their times differ by at most a tolerance. Each step matches at most once: the
matching is a maximum-cardinality assignment that, among maximum matchings, minimises the total time
difference (the Hungarian algorithm, Kuhn, Naval Research Logistics Quarterly 2:83-97, 1955, through
scipy.optimize.linear_sum_assignment, with a cost above every feasible total for infeasible pairs).
Precision = matched / reconstructed and recall = matched / annotated, pooled over incidents.

The resampling unit of every forensic metric is the incident: weights are per incident ([I] or [B, I]).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from nagahana.evaluation._arrays import safe_ratio, unbatch, weight_matrix, weighted_quantile
from nagahana.evaluation.predictions import ForensicPredictions


def subset(z: ForensicPredictions, idx: np.ndarray) -> ForensicPredictions:
    """The record restricted to incidents idx (in that order); narrative rows are renumbered."""
    sel = np.asarray(idx, dtype=np.int64)
    remap = np.full(z.onset_pred.shape[0], -1, dtype=np.int64)
    remap[sel] = np.arange(sel.size)

    def rows(arr: np.ndarray) -> np.ndarray:
        keep = remap[arr[:, 0].astype(np.int64)] >= 0 if arr.size else np.zeros(0, dtype=bool)
        out = arr[keep].copy()
        if out.size:
            out[:, 0] = remap[out[:, 0].astype(np.int64)]
        return out.reshape(-1, 4)

    return ForensicPredictions(onset_pred=z.onset_pred[sel], onset_true=z.onset_true[sel], stage_names=z.stage_names,
                               patient_zero=z.patient_zero[sel], patient_zero_true=z.patient_zero_true[sel],
                               narrative_pred=rows(z.narrative_pred), narrative_true=rows(z.narrative_true),
                               meta=z.meta.iloc[sel].reset_index(drop=True))


def onset_error_pairs(z: ForensicPredictions) -> tuple[np.ndarray, np.ndarray]:
    """(incident index, absolute onset error in seconds) of every (incident, stage) present in both."""
    both = np.isfinite(z.onset_pred) & np.isfinite(z.onset_true)
    inc, _ = np.nonzero(both)
    return inc, np.abs(z.onset_pred - z.onset_true)[both]


def narrative_matches(z: ForensicPredictions, *, tolerance_seconds: float, rule: str = "exact") -> np.ndarray:
    """Matched step count per incident [I] (maximum matching, minimal total time difference)."""
    if tolerance_seconds < 0:
        raise ValueError("tolerance_seconds must be >= 0")
    if rule not in ("exact", "stage"):
        raise ValueError("rule must be 'exact' or 'stage'")
    n_inc = z.onset_pred.shape[0]
    out = np.zeros(n_inc)
    pred, true = z.narrative_pred, z.narrative_true
    for i in range(n_inc):
        p, t = pred[pred[:, 0] == i], true[true[:, 0] == i]
        if p.shape[0] == 0 or t.shape[0] == 0:
            continue
        dt = np.abs(p[:, 3][:, None] - t[:, 3][None, :])
        feasible = (p[:, 1][:, None] == t[:, 1][None, :]) & (dt <= tolerance_seconds)
        if rule == "exact":
            feasible &= (p[:, 2][:, None] == t[:, 2][None, :]) | (t[:, 2][None, :] == -1)
        if not feasible.any():
            continue
        big = float(dt[feasible].sum()) + 1.0                          # above any total of feasible costs
        cost = np.where(feasible, dt, big)
        r, c = linear_sum_assignment(cost)
        out[i] = float(feasible[r, c].sum())
    return out


def forensic_metrics(z: ForensicPredictions, weights: Any = None, *, top_k: int, tolerance_seconds: float,
                     rule: str = "exact") -> dict[str, Any]:
    """Median onset error (s), onset coverage, patient zero top-1 and top-k, narrative precision and recall."""
    n_inc = z.onset_pred.shape[0]
    w, batched = weight_matrix(weights, n_inc)
    inc, err = onset_error_pairs(z)
    out: dict[str, np.ndarray] = {"n_incidents": w.sum(axis=1)}
    out["median_onset_error_s"] = weighted_quantile(err, w[:, inc], 0.5) if err.size else np.full(w.shape[0], np.nan)
    annotated = np.isfinite(z.onset_true).sum(axis=1).astype(np.float64)
    covered = (np.isfinite(z.onset_true) & np.isfinite(z.onset_pred)).sum(axis=1).astype(np.float64)
    out["onset_coverage"] = safe_ratio(w @ covered, w @ annotated)
    known = z.patient_zero_true >= 0
    rank = np.full(n_inc, np.inf)
    for i in np.flatnonzero(known):
        hit = np.flatnonzero(z.patient_zero[i] == z.patient_zero_true[i])
        if hit.size:
            rank[i] = hit[0] + 1
    wk = w * known[None, :]
    out["patient_zero_top1"] = safe_ratio(wk @ (rank <= 1).astype(np.float64), wk.sum(axis=1))
    out[f"patient_zero_top{top_k}"] = safe_ratio(wk @ (rank <= top_k).astype(np.float64), wk.sum(axis=1))
    matched = narrative_matches(z, tolerance_seconds=tolerance_seconds, rule=rule)
    n_pred = np.bincount(z.narrative_pred[:, 0].astype(np.int64), minlength=n_inc).astype(np.float64)
    n_true = np.bincount(z.narrative_true[:, 0].astype(np.int64), minlength=n_inc).astype(np.float64)
    out["narrative_precision"] = safe_ratio(w @ matched, w @ n_pred)
    out["narrative_recall"] = safe_ratio(w @ matched, w @ n_true)
    return {name: unbatch(np.atleast_1d(v), batched) for name, v in out.items()}
