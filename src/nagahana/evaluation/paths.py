"""Imagined attack paths against realised paths: edit distance, precision and recall at N, ranking quality.

A path is a sequence of (tactic, entity) steps (PathPredictions). Two steps match when their stages are
equal and their entities match: equal entities, or an unknown realised entity (-1), which the truth does
not constrain ("exact" rule); the "stage" rule compares stages only.

Edit distance. Lev is the Levenshtein distance (unit cost for insertion, deletion and substitution;
Levenshtein, Soviet Physics Doklady 10:707-710, 1966) and the normalised distance is
d(p, q) = Lev(p, q) / max(|p|, |q|) (0 for two empty paths). A predicted path matches the realised path
pi* when d <= epsilon (thesis section on stages and paths). The distances of all (trigger, route) pairs
are computed together by dynamic programming over padded arrays, one table row at a time.

Metrics over triggers that have a realised path (an attack followed):

    precision@N   share of the top-N predicted routes (over all triggers) that match the realised path
    recall@N      share of realised paths matched by at least one top-N route (hit@N)
    hit@k         recall with the top-k routes
    exact@1       the most probable route equals the realised path step for step
    prefix@1      the fraction of the realised path reproduced from its first step by the most probable route
    best distance the normalised distance of the best-matching top-N route, per realised path (its
                  median and its distribution are reported)

Ranking quality. Kendall's tau-b (Kendall, Biometrika 33:239-251, 1945) between the route probabilities
and the routes' true values, per trigger, averaged over triggers where it is defined; NDCG@N (Jarvelin
and Kekalainen, ACM TOIS 20:422-446, 2002) with gain = true value in [0, 1] and discount log2(rank + 1).
On real data a route's true value is its similarity to the realised path, 1 - d; in simulated worlds it
is the value under the true dynamics (PathPredictions.true_value, protocol P6).

Ordinal safety (thesis equation). Over the pairs P of candidates of a trigger whose true values differ,

    OS = (1 / |P|) sum_{(i,j) in P} 1[sign(Jhat_i - Jhat_j) = sign(J_i - J_j)],

pooled over the pairs of all triggers (pairs with tied true values carry no order and are excluded
unless requested). It is reported as a function of horizon when values per horizon are supplied.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from nagahana.core.errors import InvariantViolation
from nagahana.evaluation._arrays import safe_ratio, unbatch, weight_matrix, weighted_quantile
from nagahana.evaluation.predictions import PathPredictions, PathStep


def subset(pred: PathPredictions, idx: np.ndarray) -> PathPredictions:
    """The record restricted to triggers idx (in that order)."""
    sel = [int(i) for i in np.asarray(idx, dtype=np.int64)]
    tv = None if pred.true_value is None else [pred.true_value[i] for i in sel]
    return PathPredictions(predicted=[pred.predicted[i] for i in sel], realised=[pred.realised[i] for i in sel],
                           meta=pred.meta.iloc[sel].reset_index(drop=True), true_value=tv)


def _pad(paths: list[tuple[PathStep, ...]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # (stage [P, L], entity [P, L], length [P]) with -2 padding.
    length = np.array([len(p) for p in paths], dtype=np.int64)
    width = max(1, int(length.max()) if length.size else 1)
    stage = np.full((len(paths), width), -2, dtype=np.int64)
    ent = np.full((len(paths), width), -2, dtype=np.int64)
    for i, p in enumerate(paths):
        if p:
            stage[i, : len(p)] = [s.stage for s in p]
            ent[i, : len(p)] = [s.entity for s in p]
    return stage, ent, length


def levenshtein(pred: list[tuple[PathStep, ...]], true: list[tuple[PathStep, ...]], *, rule: str = "exact") -> np.ndarray:
    """Levenshtein distances [P] between pred[p] and true[p] (step match by `rule`)."""
    if len(pred) != len(true):
        raise InvariantViolation("pred and true must pair up")
    if rule not in ("exact", "stage"):
        raise ValueError("rule must be 'exact' or 'stage'")
    n_pairs = len(pred)
    if n_pairs == 0:
        return np.zeros(0, dtype=np.int64)
    a_s, a_e, a_len = _pad(pred)
    b_s, b_e, b_len = _pad(true)
    la, lb = a_s.shape[1], b_s.shape[1]
    out = np.where(a_len == 0, b_len, 0)                                # row 0 of the table: j insertions
    prev = np.repeat(np.arange(lb + 1)[None, :], n_pairs, axis=0)       # D[0, j] = j
    for i in range(1, la + 1):
        cur = np.empty_like(prev)
        cur[:, 0] = i
        for j in range(1, lb + 1):
            same = a_s[:, i - 1] == b_s[:, j - 1]
            if rule == "exact":
                same &= (a_e[:, i - 1] == b_e[:, j - 1]) | (b_e[:, j - 1] == -1)
            cur[:, j] = np.minimum(np.minimum(prev[:, j] + 1, cur[:, j - 1] + 1), prev[:, j - 1] + (~same))
        done = a_len == i
        out[done] = cur[done, b_len[done]]
        prev = cur
    return out


def normalised_distance(pred: list[tuple[PathStep, ...]], true: list[tuple[PathStep, ...]], *, rule: str = "exact") -> np.ndarray:
    """Lev(p, q) / max(|p|, |q|) per pair [P] (0 for two empty paths)."""
    lev = levenshtein(pred, true, rule=rule).astype(np.float64)
    longest = np.array([max(len(a), len(b)) for a, b in zip(pred, true, strict=True)], dtype=np.float64)
    return np.where(longest > 0, lev / np.where(longest > 0, longest, 1.0), 0.0)


def _prefix_fraction(pred: tuple[PathStep, ...], true: tuple[PathStep, ...], rule: str) -> float:
    # Share of the realised path reproduced from its first step.
    k = 0
    for a, b in zip(pred, true, strict=False):
        ok = a.stage == b.stage and (rule == "stage" or a.entity == b.entity or b.entity == -1)
        if not ok:
            break
        k += 1
    return k / len(true)


def kendall_tau_b(x: np.ndarray, y: np.ndarray, valid: np.ndarray, *, chunk: int = 4096) -> np.ndarray:
    """Kendall's tau-b per row of x, y [m, C] over the valid entries; NaN where undefined."""
    m, c = x.shape
    out = np.full(m, np.nan)
    iu, ju = np.triu_indices(c, k=1)
    for a in range(0, m, chunk):
        z = slice(a, min(a + chunk, m))
        both = valid[z][:, iu] & valid[z][:, ju]                       # [r, pairs]
        sx = np.sign(x[z][:, iu] - x[z][:, ju])
        sy = np.sign(y[z][:, iu] - y[z][:, ju])
        n0 = both.sum(axis=1)
        conc = (both & (sx * sy > 0)).sum(axis=1)
        disc = (both & (sx * sy < 0)).sum(axis=1)
        n1 = (both & (sx == 0)).sum(axis=1)
        n2 = (both & (sy == 0)).sum(axis=1)
        den = np.sqrt((n0 - n1).astype(np.float64) * (n0 - n2))
        out[z] = safe_ratio(conc - disc, den)
    return out


def ordinal_pairs(model_values: np.ndarray, true_values: np.ndarray, valid: np.ndarray, *,
                  include_ties: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Per row (agreeing pairs, counted pairs) of candidates [m, C] (or [m, C, H] -> [m, H] each)."""
    mv, tv, ok = np.asarray(model_values, dtype=np.float64), np.asarray(true_values, dtype=np.float64), np.asarray(valid, dtype=bool)
    if mv.shape != tv.shape or ok.shape != mv.shape:
        raise InvariantViolation("model values, true values and the mask must share one shape")
    squeeze = mv.ndim == 2
    if squeeze:
        mv, tv, ok = mv[..., None], tv[..., None], ok[..., None]
    c = mv.shape[1]
    iu, ju = np.triu_indices(c, k=1)
    both = ok[:, iu] & ok[:, ju]                                       # [m, pairs, H]
    sm = np.sign(mv[:, iu] - mv[:, ju])
    st = np.sign(tv[:, iu] - tv[:, ju])
    counted = both if include_ties else both & (st != 0)
    agree = (counted & (sm == st)).sum(axis=1).astype(np.float64)
    total = counted.sum(axis=1).astype(np.float64)
    return (agree[:, 0], total[:, 0]) if squeeze else (agree, total)


def path_metrics(pred: PathPredictions, weights: Any = None, *, top_n: int, tolerance: float, rule: str = "exact",
                 include_empty: bool = False) -> dict[str, Any]:
    """Path precision/recall at N, hit@1, exact@1, prefix@1, best distances, Kendall tau, NDCG, ordinal safety.

    weights: over all triggers ([m] or [B, m]); triggers without a realised path get weight 0 unless
    include_empty (then their routes count as unmatched).
    """
    if top_n < 1 or not 0.0 <= tolerance <= 1.0:
        raise ValueError("top_n must be >= 1 and tolerance in [0, 1]")
    m = len(pred.predicted)
    w, batched = weight_matrix(weights, m)
    has_true = np.array([len(r) > 0 for r in pred.realised])
    use = has_true | include_empty
    w = w * use[None, :]
    # All (trigger, route) pairs among the top N, with their normalised distance to the realised path.
    owner, rank, routes, truths, probs = [], [], [], [], []
    for i, rts in enumerate(pred.predicted):
        for r, (p, path) in enumerate(rts[:top_n]):
            owner.append(i)
            rank.append(r)
            routes.append(path)
            truths.append(pred.realised[i])
            probs.append(p)
    owner_a, rank_a = np.asarray(owner, dtype=np.int64), np.asarray(rank, dtype=np.int64)
    dist = normalised_distance(routes, truths, rule=rule) if routes else np.zeros(0)
    dist = np.where(has_true[owner_a], dist, 1.0) if routes else dist   # no realised path: no match possible
    match = dist <= tolerance + 1e-12
    n_routes = np.bincount(owner_a, minlength=m).astype(np.float64)
    n_match = np.bincount(owner_a, weights=match.astype(np.float64), minlength=m)
    # Best-matching route per trigger with a realised path (NaN without routes or without a realised path).
    lowest = np.full(m, np.inf)
    if routes:
        np.minimum.at(lowest, owner_a, dist)
    best = np.where(np.isfinite(lowest) & has_true, lowest, np.nan)
    top1 = rank_a == 0
    hit1 = np.zeros(m)
    exact1 = np.zeros(m)
    prefix1 = np.full(m, np.nan)
    hit1[owner_a[top1]] = match[top1]
    exact1[owner_a[top1 & (dist == 0)]] = 1.0
    for i in np.flatnonzero(has_true):
        rts = pred.predicted[i]
        prefix1[i] = _prefix_fraction(rts[0][1], pred.realised[i], rule) if rts else 0.0
    hit_n = (np.nan_to_num(best, nan=np.inf) <= tolerance + 1e-12).astype(np.float64)
    # Ranking: true value per route (simulated truth when given, else similarity to the realised path).
    c = min(top_n, max((len(r) for r in pred.predicted), default=0))
    out: dict[str, np.ndarray] = {}
    total = w.sum(axis=1)
    out["n_triggers"] = total
    out["precision_at_n"] = safe_ratio(w @ n_match, w @ n_routes)
    out["recall_at_n"] = safe_ratio(w @ hit_n, total)
    out["hit_at_1"] = safe_ratio(w @ hit1, total)
    out["exact_at_1"] = safe_ratio(w @ exact1, total)
    pv = np.isfinite(prefix1)
    out["prefix_at_1"] = safe_ratio(w @ np.where(pv, prefix1, 0.0), w @ pv.astype(np.float64))
    bv = np.isfinite(best)
    out["mean_best_distance"] = safe_ratio(w @ np.where(bv, best, 0.0), w @ bv.astype(np.float64))
    out["median_best_distance"] = weighted_quantile(np.where(bv, best, np.inf), w * bv[None, :], 0.5) if bv.any() else np.full(w.shape[0], np.nan)
    if c >= 1:
        x = np.zeros((m, c))
        y = np.zeros((m, c))
        ok = np.zeros((m, c), dtype=bool)
        x[owner_a[rank_a < c], rank_a[rank_a < c]] = np.asarray(probs, dtype=np.float64)[rank_a < c]
        sim = 1.0 - dist
        y[owner_a[rank_a < c], rank_a[rank_a < c]] = sim[rank_a < c]
        ok[owner_a[rank_a < c], rank_a[rank_a < c]] = True
        if pred.true_value is not None:
            for i, vals in enumerate(pred.true_value):
                k = min(len(vals), c)
                y[i, :k] = vals[:k]
        tau = kendall_tau_b(x, y, ok)
        tv = np.isfinite(tau)
        out["kendall_tau"] = safe_ratio(w @ np.where(tv, tau, 0.0), w @ tv.astype(np.float64))
        gain = np.clip(y, 0.0, 1.0) * ok
        disc = 1.0 / np.log2(np.arange(2, c + 2))
        dcg = (gain * disc[None, :]).sum(axis=1)
        idcg = (-np.sort(-gain, axis=1) * disc[None, :]).sum(axis=1)
        ndcg = safe_ratio(dcg, idcg)
        nv = np.isfinite(ndcg)
        out["ndcg_at_n"] = safe_ratio(w @ np.where(nv, ndcg, 0.0), w @ nv.astype(np.float64))
        agree, pairs = ordinal_pairs(x, y, ok)
        out["ordinal_safety"] = safe_ratio(w @ agree, w @ pairs)
    else:
        for name in ("kendall_tau", "ndcg_at_n", "ordinal_safety"):
            out[name] = np.full(w.shape[0], np.nan)
    return {name: unbatch(np.atleast_1d(v), batched) for name, v in out.items()}


def best_distances(pred: PathPredictions, *, top_n: int, rule: str = "exact") -> np.ndarray:
    """Normalised distance of the best-matching top-N route per trigger with a realised path [m'] (figure data)."""
    vals = []
    for routes, true in zip(pred.predicted, pred.realised, strict=True):
        if not true:
            continue
        cand = [p for _, p in routes[:top_n]]
        vals.append(float(normalised_distance(cand, [true] * len(cand), rule=rule).min()) if cand else 1.0)
    return np.asarray(vals, dtype=np.float64)


def ordinal_safety(model_values: Any, true_values: Any, valid: Any = None, weights: Any = None, *,
                   include_ties: bool = False) -> Any:
    """Pooled ordinal safety over triggers: [m, C] -> scalar, or [m, C, H] -> per horizon [H] ([B, H] batched)."""
    mv = np.asarray(model_values, dtype=np.float64)
    ok = np.ones(mv.shape, dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
    agree, pairs = ordinal_pairs(mv, true_values, ok, include_ties=include_ties)
    w, batched = weight_matrix(weights, mv.shape[0])
    if agree.ndim == 1:
        return unbatch(safe_ratio(w @ agree, w @ pairs), batched)
    val = safe_ratio(w @ agree, w @ pairs)                              # [B, H]
    return val if batched else val[0]
