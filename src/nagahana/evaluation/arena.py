"""Defender agents in a cyber-defence arena (protocol P-CW): sample efficiency, returns and generalisation.

The arena is a simulated defended network such as CyberWheel, in which defender agents (NagaHana's
Advisor, CyberWorld's world-model agents, model-free agents) are trained against attacker strategies
and compared with a control policy. One ArenaRun record holds one agent trained with one seed: its
learning curve (evaluation returns at increasing environment steps), the returns of the control, the
returns of its final policy and its returns under generalisation conditions (a held-out attacker
strategy, network sizes beyond training, degraded observations).

Statistics over seeds. Reinforcement-learning results come from few independent runs, so every
aggregate here follows Agarwal, Schwarzer, Castro, Courville and Bellemare ("Deep Reinforcement Learning
at the Edge of the Statistical Precipice", NeurIPS 2021, arXiv:2108.13264):

    per-run score     the mean return of the run's evaluation episodes
    IQM               the interquartile mean of the per-run scores, scipy.stats.trim_mean with proportion
                      0.25 as in the authors' reference implementation (rliable): floor(n / 4) runs are
                      dropped at each end, so with three seeds the IQM equals the mean; robust to outlier
                      runs and less noisy than the median
    intervals         percentile intervals of a bootstrap over runs (a stratified bootstrap when several
                      tasks or conditions are pooled: runs are resampled within each stratum)
    P(X > Y)          the probability of improvement, the Mann-Whitney statistic of the per-run scores
                      of two agents, (1/(m n)) sum_i sum_j [1(x_i > y_j) + 1/2 1(x_i = y_j)]
    profiles          the fraction of runs whose score exceeds tau, over a grid of tau

At least three seeds per agent are required (`MIN_SEEDS`).

Sample efficiency. A run exceeds the control at the first evaluation point at which its mean return
is above the control's mean return; it exceeds it in a sustained way from the first point after which
every later point is above it. A run that never does is censored at its last evaluated step and counts
as +inf, so the median over runs is an order statistic that remains defined while fewer than half the
runs are censored (the same convention as lead times in episodes.py). The interval is the bootstrap
percentile interval of that median over runs.

Learning curves. Runs of one agent must share their evaluation steps; at every step the IQM and the
mean of the per-run mean returns are reported with bootstrap intervals over runs (the data of the
learning-curve plots). Agents are compared at their final policies by the difference of IQMs (with a
bootstrap interval that resamples the two agents' runs independently), the probability of improvement
and an exact two-sample permutation test of the difference of mean per-run scores (all relabellings
when there are at most 10^5, otherwise Monte Carlo with the (1 + count) / (1 + R) correction).
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import stats

from nagahana.core.errors import InvariantViolation
from nagahana.evaluation.predictions import ArenaRun

MIN_SEEDS = 3


def iqm(scores: np.ndarray) -> Any:
    """Interquartile mean along the last axis: scipy.stats.trim_mean with proportion 0.25."""
    x = np.asarray(scores, dtype=np.float64)
    if x.shape[-1] == 0:
        return np.full(x.shape[:-1], np.nan) if x.ndim > 1 else math.nan
    return stats.trim_mean(x, 0.25, axis=-1)


def run_scores(runs: Sequence[np.ndarray]) -> np.ndarray:
    """Per-run scores: the mean of each run's evaluation returns."""
    return np.array([float(np.mean(r)) for r in runs])


def bootstrap_runs(scores: np.ndarray, stat: Any, *, n_resamples: int, confidence: float, rng: np.random.Generator,
                   strata: np.ndarray | None = None) -> tuple[float, float, float]:
    """(statistic, low, high) with the percentile interval of a (stratified) bootstrap over runs."""
    s = np.asarray(scores, dtype=np.float64)
    point = float(stat(s))
    if s.size < 2:
        return point, math.nan, math.nan
    if strata is None:
        idx = rng.integers(0, s.size, size=(n_resamples, s.size))
    else:
        st = np.asarray(strata)
        idx = np.empty((n_resamples, s.size), dtype=np.int64)
        for v in np.unique(st):
            members = np.flatnonzero(st == v)
            idx[:, members] = members[rng.integers(0, members.size, size=(n_resamples, members.size))]
    reps = np.array([float(stat(s[row])) for row in idx])
    fin = reps[np.isfinite(reps)]
    if fin.size == 0:
        return point, math.nan, math.nan
    lo, hi = np.quantile(fin, [(1 - confidence) / 2, (1 + confidence) / 2])
    return point, float(lo), float(hi)


def probability_of_improvement(x: np.ndarray, y: np.ndarray) -> float:
    """P(X > Y) + 1/2 P(X = Y) over all pairs of per-run scores."""
    xa, ya = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if xa.size == 0 or ya.size == 0:
        return math.nan
    return float(np.mean((xa[:, None] > ya[None, :]) + 0.5 * (xa[:, None] == ya[None, :])))


def permutation_test(x: np.ndarray, y: np.ndarray, *, rng: np.random.Generator, max_exact: int = 100_000,
                     n_random: int = 20_000) -> float:
    """Two-sided permutation test of mean(x) = mean(y) for independent runs (exact when feasible)."""
    xa, ya = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    pool = np.r_[xa, ya]
    m, n = xa.size, pool.size
    if m == 0 or ya.size == 0:
        return math.nan
    observed = abs(xa.mean() - ya.mean())
    total = math.comb(n, m)
    if total <= max_exact:
        count = 0
        for combo in itertools.combinations(range(n), m):
            mask = np.zeros(n, dtype=bool)
            mask[list(combo)] = True
            if abs(pool[mask].mean() - pool[~mask].mean()) >= observed - 1e-12:
                count += 1
        return count / total
    count = 0
    for _ in range(n_random):
        perm = rng.permutation(n)
        if abs(pool[perm[:m]].mean() - pool[perm[m:]].mean()) >= observed - 1e-12:
            count += 1
    return (1 + count) / (1 + n_random)


def steps_to_exceed(run: ArenaRun, *, sustained: bool) -> float:
    """First evaluated step at which the mean return exceeds the control's mean return (+inf if never)."""
    curve = run.returns.mean(axis=1)
    control = float(run.control_returns.mean())
    above = curve > control
    if sustained:
        # The first index after which every later point is above the control.
        tail_all = np.flip(np.logical_and.accumulate(np.flip(above)))
        hits = np.flatnonzero(tail_all)
    else:
        hits = np.flatnonzero(above)
    return float(run.steps[hits[0]]) if hits.size else math.inf


def _median(x: np.ndarray) -> float:
    # Median allowing +inf entries (censored runs rank last).
    s = np.sort(np.asarray(x, dtype=np.float64))
    n = s.size
    if n == 0:
        return math.nan
    a, b = s[(n - 1) // 2], s[n // 2]
    if a == b:
        return float(a)
    return math.inf if math.isinf(b) else float(0.5 * (a + b))


@dataclass(frozen=True)
class AgentSummary:
    """Aggregate results of one agent over its seeds (each value a (point, low, high) triple)."""

    agent: str
    seeds: int
    final_iqm: tuple[float, float, float]
    final_mean: tuple[float, float, float]
    improvement_over_control: tuple[float, float, float]
    steps_to_exceed: tuple[float, float, float]
    steps_to_exceed_sustained: tuple[float, float, float]
    exceeded_share: float
    conditions: dict[str, dict[str, tuple[float, float, float]]]


def check_runs(agent: str, runs: Sequence[ArenaRun]) -> None:
    """Refuse an agent with fewer than MIN_SEEDS runs or runs that disagree on environment or steps."""
    if len(runs) < MIN_SEEDS:
        raise InvariantViolation(f"agent {agent}: the arena needs at least {MIN_SEEDS} seeds, got {len(runs)}")
    env = {r.environment for r in runs}
    if len(env) != 1:
        raise InvariantViolation(f"agent {agent}: runs come from different environments {sorted(env)}")
    if any(not np.array_equal(r.steps, runs[0].steps) for r in runs):
        raise InvariantViolation(f"agent {agent}: runs must share their evaluation steps")
    if any(set(r.conditions) != set(runs[0].conditions) for r in runs):
        raise InvariantViolation(f"agent {agent}: runs must share their generalisation conditions")


def summarise(agent: str, runs: Sequence[ArenaRun], *, n_resamples: int, confidence: float,
              rng: np.random.Generator) -> AgentSummary:
    """Final return, improvement over the control, steps to exceed the control and generalisation."""
    check_runs(agent, runs)
    final = run_scores([r.final_returns for r in runs])
    control = run_scores([r.control_returns for r in runs])
    gain = final - control
    exceed = np.array([steps_to_exceed(r, sustained=False) for r in runs])
    exceed_s = np.array([steps_to_exceed(r, sustained=True) for r in runs])
    boot = {"n_resamples": n_resamples, "confidence": confidence, "rng": rng}
    conds: dict[str, dict[str, tuple[float, float, float]]] = {}
    for name in runs[0].conditions:
        sc = run_scores([r.conditions[name] for r in runs])
        cc = run_scores([r.control_conditions[name] for r in runs])
        conds[name] = {"iqm": bootstrap_runs(sc, iqm, **boot), "mean": bootstrap_runs(sc, np.mean, **boot),
                       "improvement_over_control": bootstrap_runs(sc - cc, np.mean, **boot),
                       "change_from_training_conditions": bootstrap_runs(sc - final, np.mean, **boot)}
    return AgentSummary(agent, len(runs), bootstrap_runs(final, iqm, **boot), bootstrap_runs(final, np.mean, **boot),
                        bootstrap_runs(gain, np.mean, **boot), bootstrap_runs(exceed, _median, **boot),
                        bootstrap_runs(exceed_s, _median, **boot), float(np.isfinite(exceed).mean()), conds)


def learning_curve(agent: str, runs: Sequence[ArenaRun], *, n_resamples: int, confidence: float,
                   rng: np.random.Generator) -> list[dict[str, float]]:
    """IQM and mean of the per-run mean returns at every evaluation step, with bootstrap intervals."""
    check_runs(agent, runs)
    mat = np.stack([r.returns.mean(axis=1) for r in runs])               # [runs, T]
    control = float(np.mean([r.control_returns.mean() for r in runs]))
    out = []
    for t, step in enumerate(runs[0].steps):
        v, lo, hi = bootstrap_runs(mat[:, t], iqm, n_resamples=n_resamples, confidence=confidence, rng=rng)
        m, mlo, mhi = bootstrap_runs(mat[:, t], np.mean, n_resamples=n_resamples, confidence=confidence, rng=rng)
        out.append({"agent": agent, "step": float(step), "iqm": v, "iqm_low": lo, "iqm_high": hi, "mean": m,
                    "mean_low": mlo, "mean_high": mhi, "control_mean": control})
    return out


def performance_profile(scores: np.ndarray, taus: np.ndarray) -> np.ndarray:
    """Fraction of runs with a score above each tau."""
    s = np.asarray(scores, dtype=np.float64)
    return (s[None, :] > np.asarray(taus, dtype=np.float64)[:, None]).mean(axis=1)


def compare_agents(x_runs: Sequence[ArenaRun], y_runs: Sequence[ArenaRun], *, n_resamples: int, confidence: float,
                   rng: np.random.Generator) -> dict[str, float]:
    """Final-policy comparison of agent X against agent Y (independent runs)."""
    x = run_scores([r.final_returns for r in x_runs])
    y = run_scores([r.final_returns for r in y_runs])
    reps_d, reps_p = [], []
    for _ in range(n_resamples):
        xs = x[rng.integers(0, x.size, x.size)]
        ys = y[rng.integers(0, y.size, y.size)]
        reps_d.append(float(iqm(xs) - iqm(ys)))
        reps_p.append(probability_of_improvement(xs, ys))
    q = [(1 - confidence) / 2, (1 + confidence) / 2]
    d_lo, d_hi = np.quantile(reps_d, q)
    p_lo, p_hi = np.quantile(reps_p, q)
    return {"iqm_difference": float(iqm(x) - iqm(y)), "iqm_difference_low": float(d_lo), "iqm_difference_high": float(d_hi),
            "probability_of_improvement": probability_of_improvement(x, y), "poi_low": float(p_lo), "poi_high": float(p_hi),
            "p_value": permutation_test(x, y, rng=rng), "runs_x": float(x.size), "runs_y": float(y.size)}
