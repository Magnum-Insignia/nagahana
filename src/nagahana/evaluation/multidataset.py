"""Comparison of several models over many datasets: Friedman test, Nemenyi post-hoc, critical-difference data.

Following Demsar ("Statistical comparisons of classifiers over multiple data sets", JMLR 7:1-30, 2006),
models are ranked within every block (one dataset, or one dataset under one protocol, novelty group or
horizon), rank 1 being the best and ties sharing the average rank. With N blocks, k models and the
average ranks R_j,

    chi2_F = 12 N / (k (k + 1)) [ sum_j R_j^2 - k (k + 1)^2 / 4 ]                (Friedman, JASA 32:675-701, 1937)
    F_F    = (N - 1) chi2_F / (N (k - 1) - chi2_F) ~ F(k - 1, (k - 1)(N - 1))     (Iman and Davenport, Communications
                                                                                in Statistics A 9:571-595, 1980)

F_F is less conservative than chi2_F and is the test Demsar recommends. When the null hypothesis of
equal performance is rejected, the Nemenyi test (Nemenyi, PhD thesis, Princeton University, 1963)
declares two models different when their average ranks differ by at least the critical difference

    CD = q_alpha sqrt(k (k + 1) / (6 N)),   q_alpha = q(1 - alpha; k, infinity) / sqrt(2),

with q the studentized range distribution (scipy.stats.studentized_range). Pairwise Nemenyi p-values are
P(Q > |R_i - R_j| sqrt(2) / sqrt(k (k + 1) / (6 N))). The critical-difference diagram shows the average
ranks on an axis and joins by a bar every maximal group of models whose average ranks differ by less
than CD (a clique: no pair inside it is significantly different).

Because the mean-ranks test makes the result for two models depend on the other models compared,
Benavoli, Corani and Mangili (JMLR 17(5):1-10, 2016) recommend pairwise Wilcoxon signed-rank tests over
the blocks with Holm's correction; `wilcoxon_holm` provides them as a complement.

Blocks must be complete (every model scored on every block); incomplete blocks are dropped and
counted. Blocks are treated as independent; the pooled group "all" is never a block, because it repeats
the units of the per-dataset blocks.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from nagahana.core.errors import InvariantViolation
from nagahana.evaluation.significance import adjust_pvalues, wilcoxon_signed_rank


@dataclass(frozen=True)
class FriedmanResult:
    """Friedman and Iman-Davenport statistics, Nemenyi critical difference and the data of a CD diagram."""

    models: tuple[str, ...]
    blocks: int
    dropped_blocks: int
    average_ranks: np.ndarray
    chi2: float
    chi2_p: float
    iman_davenport: float
    iman_davenport_p: float
    critical_difference: float
    alpha: float
    nemenyi_p: np.ndarray
    cliques: tuple[tuple[str, ...], ...]
    ranks: pd.DataFrame

    def diagram(self) -> dict[str, Any]:
        """Plain data of the critical-difference diagram (models sorted by average rank)."""
        order = np.argsort(self.average_ranks, kind="stable")
        return {"models": [self.models[i] for i in order],
                "average_ranks": [float(self.average_ranks[i]) for i in order],
                "critical_difference": self.critical_difference, "alpha": self.alpha, "blocks": self.blocks,
                "cliques": [list(c) for c in self.cliques], "friedman_chi2": self.chi2, "friedman_p": self.chi2_p,
                "iman_davenport_f": self.iman_davenport, "iman_davenport_p": self.iman_davenport_p}


def block_ranks(scores: pd.DataFrame, *, higher_is_better: bool) -> pd.DataFrame:
    """Ranks within each row (block) of scores [blocks x models]; 1 is best, ties share the average rank."""
    vals = scores.to_numpy(dtype=np.float64)
    signed = -vals if higher_is_better else vals
    ranks = stats.rankdata(signed, axis=1, method="average")
    return pd.DataFrame(ranks, index=scores.index, columns=scores.columns)


def nemenyi_critical_difference(k: int, n: int, alpha: float = 0.05) -> float:
    """CD = q(1 - alpha; k, inf) / sqrt(2) * sqrt(k (k + 1) / (6 N))."""
    if k < 2 or n < 1:
        raise InvariantViolation("the Nemenyi test needs k >= 2 models and N >= 1 blocks")
    q = float(stats.studentized_range.ppf(1.0 - alpha, k, np.inf)) / math.sqrt(2.0)
    return q * math.sqrt(k * (k + 1) / (6.0 * n))


def cliques(models: list[str], average_ranks: np.ndarray, cd: float) -> tuple[tuple[str, ...], ...]:
    """Maximal groups of models (consecutive in rank order) whose average ranks span less than CD."""
    order = np.argsort(average_ranks, kind="stable")
    r = average_ranks[order]
    groups: list[tuple[int, int]] = []
    for i in range(r.size):
        j = i
        while j + 1 < r.size and r[j + 1] - r[i] < cd:
            j += 1
        if j > i and not any(a <= i and j <= b for a, b in groups):
            groups.append((i, j))
    return tuple(tuple(models[order[t]] for t in range(a, b + 1)) for a, b in groups)


def friedman_nemenyi(scores: pd.DataFrame, *, higher_is_better: bool, alpha: float = 0.05) -> FriedmanResult:
    """Friedman (with Iman-Davenport) test and Nemenyi post-hoc over blocks x models (complete blocks only)."""
    complete = scores.dropna(axis=0, how="any")
    dropped = len(scores) - len(complete)
    n, k = complete.shape
    if k < 2 or n < 2:
        raise InvariantViolation(f"need at least 2 complete blocks and 2 models (got {n} blocks, {k} models)")
    ranks = block_ranks(complete, higher_is_better=higher_is_better)
    avg = ranks.mean(axis=0).to_numpy(dtype=np.float64)
    chi2 = 12.0 * n / (k * (k + 1)) * (float(np.sum(avg ** 2)) - k * (k + 1) ** 2 / 4.0)
    chi2_p = float(stats.chi2.sf(chi2, k - 1))
    den = n * (k - 1) - chi2
    ff = (n - 1) * chi2 / den if den > 0 else math.inf
    ff_p = float(stats.f.sf(ff, k - 1, (k - 1) * (n - 1))) if math.isfinite(ff) else 0.0
    cd = nemenyi_critical_difference(k, n, alpha)
    se = math.sqrt(k * (k + 1) / (6.0 * n))
    pmat = np.ones((k, k))
    for i, j in itertools.combinations(range(k), 2):
        z = abs(avg[i] - avg[j]) / se
        pmat[i, j] = pmat[j, i] = float(stats.studentized_range.sf(z * math.sqrt(2.0), k, np.inf))
    models = [str(m) for m in complete.columns]
    return FriedmanResult(tuple(models), n, dropped, avg, chi2, chi2_p, ff, ff_p, cd, alpha, pmat,
                          cliques(models, avg, cd), ranks)


def wilcoxon_holm(scores: pd.DataFrame, *, method: str = "holm") -> pd.DataFrame:
    """Pairwise Wilcoxon signed-rank tests over complete blocks with Holm-adjusted p-values."""
    complete = scores.dropna(axis=0, how="any")
    rows = []
    for a, b in itertools.combinations(list(complete.columns), 2):
        res = wilcoxon_signed_rank(complete[a].to_numpy(dtype=np.float64), complete[b].to_numpy(dtype=np.float64))
        rows.append({"model_a": str(a), "model_b": str(b), "median_difference": res.estimate, "statistic": res.statistic,
                     "p_value": res.p_value, "rank_biserial": res.detail.get("rank_biserial", math.nan),
                     "blocks": len(complete)})
    out = pd.DataFrame(rows)
    if len(out):
        out["p_adjusted"] = adjust_pvalues(out["p_value"].to_numpy(dtype=np.float64), method)
    return out


def score_matrix(metrics: pd.DataFrame, *, metric: str, task: str, models: list[str] | None = None,
                 horizon: str | None = None) -> pd.DataFrame:
    """Blocks x models matrix of one metric from scorer metric rows (the pooled group "all" is excluded).

    A block is (protocol, variant, group, novelty, split, horizon); its value per model is the seed-mean
    point estimate of the row.
    """
    sel = metrics[(metrics["metric"] == metric) & (metrics["task"] == task) & (metrics["group"] != "all")]
    if horizon is not None:
        sel = sel[sel["horizon"] == horizon]
    if models is not None:
        sel = sel[sel["model"].isin(models)]
    if sel.empty:
        return pd.DataFrame()
    key = sel["protocol"].astype(str) + "|" + sel["variant"].astype(str) + "|" + sel["group"].astype(str) + "|" \
        + sel["novelty"].astype(str) + "|" + sel["split"].astype(str) + "|" + sel["horizon"].astype(str)
    frame = pd.DataFrame({"block": key.to_numpy(), "model": sel["model"].astype(str).to_numpy(),
                          "value": sel["value"].to_numpy(dtype=np.float64)})
    if frame.duplicated(["block", "model"]).any():
        raise InvariantViolation("several rows per block and model; restrict the metric rows further")
    return frame.pivot(index="block", columns="model", values="value")
