"""Blocked temporal cross-validation (forward chaining with purging and an embargo) and grid selection.

Folds

Units are grouped (by network or by source, `SelectionConfig.group_by`): different networks are separate
time lines and are never ordered against each other. Inside each group the units are cut, in time order,
into `n_blocks` blocks of (nearly) equal counts, with every boundary moved to a change of time so that
units at the same instant share a block. Fold j (j = min_train_blocks ... n_blocks - 1) validates on block
j of every group and trains on the earlier blocks of that group (all of them, or the last
`max_train_blocks`), so no fold trains on a block later than the one it validates on (forward chaining;
Tashman, "Out-of-sample tests of forecasting accuracy", Int. J. Forecasting 16(4), 2000).

Each unit i carries the interval [t_i, e_i] of information its label depends on: e_i = t_i for a state
update, e_i = tau + K w for a trigger whose target looks K cadence windows ahead. Following Lopez de
Prado (Advances in Financial Machine Learning, Wiley 2018, Chapter 7):

    purging   a training unit is dropped when e_i >= v_0, the first time of its group's validation block,
              because its label is determined by information from the validation period
    embargo   a training unit is dropped when t_i > v_0 - h, a gap of h seconds before the validation
              block, against serial correlation between neighbouring units (h = embargo_windows * w);
              with forward chaining no training block follows a validation block, so the embargo after
              the block that Lopez de Prado also prescribes never removes a unit here, and it is applied
              for completeness when blocks are given out of order

Selection

Every (lambda, class-weight power) pair of the grid is fitted on the training part of every fold and
scored on its validation part with the configured criterion. The chosen pair has the best mean score
over the folds ("best") or, under the one-standard-error rule, the largest lambda whose mean is within one
standard error of the best mean (Hastie, Tibshirani and Friedman, The Elements of Statistical Learning,
2nd ed., Springer 2009, Section 7.10). The final model is then refitted on all training units.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

from nagahana.core.errors import InvariantViolation

from .config import SelectionConfig


@dataclass(frozen=True)
class Fold:
    """One fold: training and validation unit indices (into the arrays given to `forward_chaining_folds`)."""

    index: int
    block: int
    train: np.ndarray
    val: np.ndarray


def _blocks(times: np.ndarray, n_blocks: int) -> np.ndarray:
    """Block id 0 ... n_blocks-1 of each unit of one group (equal counts, boundaries at time changes)."""
    n = times.size
    order = np.argsort(times, kind="mergesort")
    ts = times[order]
    block_sorted = np.zeros(n, dtype=np.int64)
    if n == 0:
        return block_sorted
    cuts = []
    for b in range(1, n_blocks):
        c = int(round(b * n / n_blocks))
        c = int(np.searchsorted(ts, ts[min(c, n - 1)], side="left")) if c < n else n
        cuts.append(c)
    edges = np.maximum.accumulate(np.asarray([0, *cuts, n], dtype=np.int64))
    for b in range(n_blocks):
        block_sorted[edges[b]:edges[b + 1]] = b
    out = np.empty(n, dtype=np.int64)
    out[order] = block_sorted
    return out


def forward_chaining_folds(times: np.ndarray, label_end: np.ndarray, groups: np.ndarray, cfg: SelectionConfig, *,
                           embargo_seconds: float) -> list[Fold]:
    """Purged, embargoed forward-chaining folds (module docstring). Folds with no training or no validation unit are skipped."""
    t = np.asarray(times, dtype=np.float64)
    e = np.asarray(label_end, dtype=np.float64)
    g = np.asarray(groups)
    if t.shape != e.shape or t.shape != g.shape:
        raise InvariantViolation("times, label ends and groups must have equal lengths")
    if np.any(e < t):
        raise InvariantViolation("a label cannot be determined before its unit's time")
    if embargo_seconds < 0:
        raise InvariantViolation("the embargo must be >= 0")
    block = np.full(t.size, -1, dtype=np.int64)
    members: dict[object, np.ndarray] = {}
    for key in np.unique(g):
        idx = np.flatnonzero(g == key)
        members[key] = idx
        block[idx] = _blocks(t[idx], cfg.n_blocks)
    folds: list[Fold] = []
    for j in range(cfg.min_train_blocks, cfg.n_blocks):
        tr_parts: list[np.ndarray] = []
        va_parts: list[np.ndarray] = []
        for idx in members.values():
            bj = block[idx]
            val = idx[bj == j]
            if val.size == 0:
                continue
            v0, v1 = float(t[val].min()), float(t[val].max())
            lo = j - cfg.max_train_blocks if cfg.max_train_blocks is not None else 0
            cand = idx[(bj < j) & (bj >= lo)]
            keep = (e[cand] < v0) & (t[cand] <= v0 - embargo_seconds)
            after = (t[cand] > v1) & (t[cand] <= v1 + embargo_seconds)
            tr_parts.append(cand[keep & ~after])
            va_parts.append(val)
        train = np.sort(np.concatenate(tr_parts)) if tr_parts else np.zeros(0, dtype=np.int64)
        val = np.sort(np.concatenate(va_parts)) if va_parts else np.zeros(0, dtype=np.int64)
        if train.size and val.size:
            folds.append(Fold(index=len(folds), block=j, train=train, val=val))
    return folds


#: fit_and_score(train_idx, val_idx, grid) -> scores [len(grid)] (NaN where a grid point cannot be scored).
FitAndScore = Callable[[np.ndarray, np.ndarray, list[tuple[float, float]]], np.ndarray]


@dataclass
class SelectionResult:
    """Scores of every grid point on every fold and the chosen point."""

    method: str
    criterion: str
    lower_is_better: bool
    grid: list[tuple[float, float]]
    scores: np.ndarray                      # [G, F]
    mean: np.ndarray                        # [G]
    se: np.ndarray                          # [G]
    chosen: tuple[float, float]
    chosen_index: int
    rule: str
    folds: list[dict[str, int]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {"method": self.method, "criterion": self.criterion, "lower_is_better": self.lower_is_better,
                "grid": [list(p) for p in self.grid], "scores": self.scores.tolist(), "mean": self.mean.tolist(),
                "se": self.se.tolist(), "chosen": list(self.chosen), "chosen_index": self.chosen_index,
                "rule": self.rule, "folds": self.folds, "notes": self.notes}


def choose(grid: Sequence[tuple[float, float]], scores: np.ndarray, *, lower_is_better: bool, rule: str) -> tuple[int, np.ndarray, np.ndarray]:
    """Index of the chosen grid point, with the mean and standard error of every point over the folds."""
    s = np.asarray(scores, dtype=np.float64)
    ok = np.isfinite(s)
    cnt = ok.sum(axis=1)
    usable = cnt > 0
    if not usable.any():
        raise InvariantViolation("no grid point could be scored on any fold")
    mean = np.full(s.shape[0], np.nan)
    se = np.full(s.shape[0], np.nan)
    for i in np.flatnonzero(usable):
        v = s[i, ok[i]]
        mean[i] = float(v.mean())
        se[i] = float(v.std(ddof=1) / np.sqrt(v.size)) if v.size > 1 else 0.0
    # points scored on fewer folds than the best-covered ones are not compared with them
    full = usable & (cnt == cnt.max())
    signed = np.where(full, mean if lower_is_better else -mean, np.inf)
    best = int(np.argmin(signed))
    if rule == "best":
        return best, mean, se
    if rule == "one_se":
        bound = signed[best] + se[best]
        cand = np.flatnonzero(full & (signed <= bound))
        lams = np.asarray([grid[int(i)][0] for i in cand])
        top = cand[lams == lams.max()]
        return int(top[np.argmin(signed[top])]), mean, se
    raise InvariantViolation(f"unknown selection rule {rule!r}")


def select(grid: Sequence[tuple[float, float]], folds: Sequence[Fold], fit_and_score: FitAndScore, *, criterion: str,
           lower_is_better: bool, rule: str, method: str) -> SelectionResult:
    """Score every grid point on every fold and choose one (module docstring)."""
    if not folds:
        raise InvariantViolation("model selection needs at least one fold")
    scores = np.full((len(grid), len(folds)), np.nan)
    notes: list[str] = []
    for f_i, fold in enumerate(folds):
        try:
            scores[:, f_i] = np.asarray(fit_and_score(fold.train, fold.val, list(grid)), dtype=np.float64)
        except InvariantViolation as exc:
            notes.append(f"fold {fold.index} (block {fold.block}) not scored: {exc}")
    best, mean, se = choose(grid, scores, lower_is_better=lower_is_better, rule=rule)
    return SelectionResult(method=method, criterion=criterion, lower_is_better=lower_is_better, grid=list(grid),
                           scores=scores, mean=mean, se=se, chosen=tuple(grid[best]), chosen_index=best, rule=rule,  # type: ignore[arg-type]
                           folds=[{"index": f.index, "block": f.block, "train": int(f.train.size), "val": int(f.val.size)}
                                  for f in folds], notes=notes)


def select_hyperparameters(cfg: SelectionConfig, *, tr: np.ndarray, va: np.ndarray, times: np.ndarray,
                           label_end: np.ndarray, groups: np.ndarray, fit_and_score: FitAndScore, criterion: str,
                           lower_is_better: bool, embargo_seconds: float,
                           grid: Sequence[tuple[float, float]] | None = None) -> SelectionResult | None:
    """The selection of `cfg.method` over positions of one unit set (module docstring).

    tr, va: positions of the training and validation units; times, label_end, groups: arrays over all
    positions. fit_and_score receives positions. `grid` replaces cfg.grid() (the ridge forecaster pairs
    lambda with a lag count instead of a class-weight power). Returns None for method "fixed".
    """
    if cfg.method == "fixed":
        return None
    grid = list(grid) if grid is not None else cfg.grid()
    tr = np.asarray(tr, dtype=np.int64)
    va = np.asarray(va, dtype=np.int64)
    if cfg.method == "validation":
        if va.size == 0:
            raise InvariantViolation("selection on the validation split needs validation units")
        return select(grid, [Fold(index=0, block=-1, train=tr, val=va)], fit_and_score, criterion=criterion,
                      lower_is_better=lower_is_better, rule=cfg.rule, method="validation")
    t = np.asarray(times, dtype=np.float64)
    e = np.maximum(np.asarray(label_end, dtype=np.float64), t)
    folds = forward_chaining_folds(t[tr], e[tr], np.asarray(groups)[tr], cfg, embargo_seconds=embargo_seconds)
    if not folds:
        raise InvariantViolation("blocked temporal cross-validation produced no fold; use more training data, fewer "
                                 "blocks, or the selection method 'validation'")
    mapped = [Fold(index=f.index, block=f.block, train=tr[f.train], val=tr[f.val]) for f in folds]
    return select(grid, mapped, fit_and_score, criterion=criterion, lower_is_better=lower_is_better, rule=cfg.rule,
                  method="temporal_cv")


def selection_from_dict(d: dict[str, object] | None) -> SelectionResult | None:
    """Inverse of SelectionResult.as_dict (for stored models)."""
    if d is None:
        return None
    g = d["grid"]
    ch = d["chosen"]
    assert isinstance(g, list) and isinstance(ch, list)
    return SelectionResult(method=str(d["method"]), criterion=str(d["criterion"]), lower_is_better=bool(d["lower_is_better"]),
                           grid=[(float(a), float(b)) for a, b in g], scores=np.asarray(d["scores"], dtype=np.float64),
                           mean=np.asarray(d["mean"], dtype=np.float64), se=np.asarray(d["se"], dtype=np.float64),
                           chosen=(float(ch[0]), float(ch[1])), chosen_index=int(d["chosen_index"]),  # type: ignore[call-overload]
                           rule=str(d["rule"]), folds=list(d["folds"]), notes=list(d["notes"]))  # type: ignore[call-overload]


__all__ = ["Fold", "SelectionResult", "choose", "forward_chaining_folds", "select", "select_hyperparameters",
           "selection_from_dict"]
