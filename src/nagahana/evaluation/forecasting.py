"""Verification of the infiltration forecast P_inf(k): proper scores per horizon, CRPS, references, skill.

Scores are penalties (lower is better), as in the thesis. For horizon k the outcome of a trigger is
"infiltration within k steps" with the censoring rule of ForecastPredictions.outcome: known when the
event fell at a step <= k or when at least k steps were observed; triggers censored earlier are
excluded at that horizon. Metrics take unit weights over all m triggers (`_arrays`); the weights of
triggers with an unknown outcome at horizon k are set to zero there, so one bootstrap resample serves
every horizon.

Brier and log score at horizon k: the weighted means of (P_inf(k) - y_k)^2 and -log of the probability
given to the outcome (calibration.py). ECE at horizon k likewise.

CRPS of the time to infiltration. P_inf(.) is the predictive CDF of the discrete time T to the first
infiltration, F(j) = P(T <= j), j = 1 .. K. For an integer-valued outcome the continuous ranked
probability score is the ranked probability score (Epstein, Journal of Applied Meteorology 8:985-987,
1969; Czado, Gneiting and Held, Biometrics 65:1254-1261, 2009); restricted to the first k steps
(censored at the horizon, a threshold-weighted CRPS in the sense of Gneiting and Ranjan, JBES
29:411-422, 2011) it is, in windows,

    CRPS_k = sum_{j=1}^{k} (F(j) - 1[T <= j])^2,

known under the same rule as the outcome at k. It equals the sum of the Brier scores of the k
cumulative events, so CRPS_1 = BS_1.

Ensemble CRPS with route weights. With route curves F_n(j) and weights w_n (sum 1), the mixture
F = sum_n w_n F_n scores sum_j (F(j) - O(j))^2 = sum_j [sum_n w_n (F_n(j) - O(j))^2 - Var_w(F_.(j))].
The fair score of Ferro (Meteorological Applications 21:7-13, 2014; Ferro, Richardson and Weigel,
Meteorological Applications 15:19-24, 2008) estimates the score of the infinite ensemble from which
the routes are drawn by dividing the variance term by 1 - sum_n w_n^2,

    CRPS_fair = sum_j [ sum_n w_n (F_n(j) - O(j))^2 - Var_w(F_.(j)) / (1 - sum_n w_n^2) ],

which for equal weights 1/N is Ferro's fair CRPS (variance with denominator N - 1). It is undefined for
a single effective route (sum_n w_n^2 = 1).

Reference forecasts (thesis section on baselines). Persistence predicts that the current state
continues: P_inf(k) = 1 for every k when the infiltration state holds at the trigger
(`ForecastPredictions.infiltrated_now`) and 0 otherwise. Climatology predicts the base rate of
infiltration within k steps estimated on the training split: the Kaplan-Meier estimate
1 - S(k w) of the training triggers' times to infiltration, censored as `outcome` defines, which is
non-decreasing in k by construction and consistent under censoring; it is estimated per group (for
example per dataset) and falls back to the pooled training estimate for a group without training
triggers.

Skill. SS = 1 - S_model / S_ref (Gneiting and Raftery, JASA 102:359-378, 2007): 1 for a perfect
forecast, 0 for the reference, negative when worse. Skill scores are generally improper, so they are
reported beside the proper scores, never instead of them.

Lead time and concordance on PyTorch tensors (`lead_time`, `concordance_index`) are kept for the
training code; the evaluation of episodes is in episodes.py and of time-to-event curves in survival.py.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from nagahana.core.errors import InvariantViolation
from nagahana.evaluation._arrays import codes, safe_ratio, unbatch, weight_matrix
from nagahana.evaluation.calibration import brier_score, calibration_error, log_loss
from nagahana.evaluation.predictions import ForecastPredictions
from nagahana.evaluation.survival import harrell_c, kaplan_meier


def skill_score(model_score: float, reference_score: float) -> float:
    """1 - S_model / S_reference (lower-is-better scores). NaN if the reference score is 0."""
    if reference_score == 0:
        return math.nan
    return 1.0 - model_score / reference_score


def skill(model_score: Any, reference_score: Any) -> Any:
    """Elementwise 1 - S_model / S_reference; NaN where the reference score is 0 or undefined."""
    m = np.asarray(model_score, dtype=np.float64)
    r = np.asarray(reference_score, dtype=np.float64)
    out = 1.0 - safe_ratio(m, r)
    return float(out) if out.ndim == 0 else out


def lead_time(times: torch.Tensor, p_inf: torch.Tensor, *, threshold: float, completion_time: float) -> float:
    """t_c - (first time p_inf >= threshold before t_c); NaN if no timely alert."""
    before = times < completion_time
    hits = torch.nonzero(before & (p_inf >= threshold)).flatten()
    if hits.numel() == 0:
        return math.nan
    return float(completion_time - times[hits[0]])


def concordance_index(risk: torch.Tensor, time: torch.Tensor, event: torch.Tensor) -> float:
    """Harrell's C-index for right-censored data (higher risk should mean earlier event).

    Computed in O(n log n) by the Fenwick-tree sweep of survival.concordance_sums: pairs (i, j) with
    T_i < T_j and an observed event at T_i are comparable, ties in risk count 1/2.
    """
    r = risk.detach().double().cpu().numpy().reshape(-1)
    t = time.detach().double().cpu().numpy().reshape(-1)
    e = event.detach().cpu().numpy().reshape(-1).astype(bool)
    return float(harrell_c(r, t, e))


def outcome_weights(f: ForecastPredictions, k: int, weights: Any = None) -> tuple[np.ndarray, np.ndarray, bool]:
    """(y_k [m], weights [B, m] zeroed where the outcome at k is unknown, batched flag)."""
    y, known = f.outcome(k)
    w, batched = weight_matrix(weights, y.size)
    return y, w * known[None, :], batched


def horizon_scores(f: ForecastPredictions, k: int, weights: Any = None, *, bins: int, eps: float | None = None
                   ) -> dict[str, Any]:
    """Brier, log score, ECE, CRPS_k, outcome count and event count at horizon k (arrays [B] when batched)."""
    y, w, batched = outcome_weights(f, k, weights)
    p = f.p_inf[:, k - 1]
    crps = crps_cumulative(f, k)
    out = {
        "brier": brier_score(p, y, w),
        "log_score": log_loss(p, y, eps=eps, weights=w),
        "ece": calibration_error(p, y, bins=bins, weights=w),
        "crps": safe_ratio(w @ np.nan_to_num(crps), w.sum(axis=1)),
        "n_known": w.sum(axis=1),
        "n_events": w @ y.astype(np.float64),
    }
    return {name: unbatch(np.atleast_1d(np.asarray(v, dtype=np.float64)), batched) for name, v in out.items()}


def _step_outcomes(f: ForecastPredictions, k: int) -> np.ndarray:
    # O(j) = 1[event at a step <= j] for j = 1 .. k: [m, k].
    j = np.arange(1, k + 1)
    return ((f.event_step[:, None] > 0) & (f.event_step[:, None] <= j[None, :])).astype(np.float64)


def crps_cumulative(f: ForecastPredictions, k: int) -> np.ndarray:
    """Per-trigger CRPS_k = sum_{j<=k} (P_inf(j) - 1[T <= j])^2 [m]; NaN where the outcome at k is unknown."""
    if not 1 <= k <= f.horizon:
        raise ValueError(f"k must lie in 1 ... {f.horizon}")
    _, known = f.outcome(k)
    val = ((f.p_inf[:, :k] - _step_outcomes(f, k)) ** 2).sum(axis=1)
    return np.where(known, val, np.nan)


def crps_ensemble(f: ForecastPredictions, k: int, *, fair: bool = True) -> np.ndarray:
    """Per-trigger ensemble CRPS_k of the route curves with their weights (fair version by default) [m]."""
    if f.ensemble is None or f.ensemble_weight is None:
        raise InvariantViolation("the forecast record has no route ensemble")
    if not 1 <= k <= f.horizon:
        raise ValueError(f"k must lie in 1 ... {f.horizon}")
    _, known = f.outcome(k)
    ens = f.ensemble[:, :, :k]                                         # [m, N, k]
    wr = f.ensemble_weight                                             # [m, N]
    obs = _step_outcomes(f, k)[:, None, :]                             # [m, 1, k]
    term1 = np.einsum("mn,mnk->m", wr, (ens - obs) ** 2)
    mean = np.einsum("mn,mnk->mk", wr, ens)
    var = np.einsum("mn,mnk->mk", wr, ens ** 2) - mean ** 2            # Var_w per step [m, k]
    var = np.clip(var, 0.0, None).sum(axis=1)
    if fair:
        s2 = (wr ** 2).sum(axis=1)
        corr = safe_ratio(var, 1.0 - s2)
        val = np.where(s2 < 1.0 - 1e-12, term1 - corr, np.nan)
    else:
        val = term1 - var
    return np.where(known, val, np.nan)


class MissingReference(InvariantViolation):
    """A reference forecast cannot be built from the record (the reason is in the message)."""


def persistence(f: ForecastPredictions) -> np.ndarray:
    """Persistence forecast [m, K]: 1 at every step where the infiltration state holds at the trigger."""
    if f.infiltrated_now is None:
        raise MissingReference("persistence needs ForecastPredictions.infiltrated_now (the current state)")
    return np.repeat(f.infiltrated_now.astype(np.float64)[:, None], f.horizon, axis=1)


@dataclass(frozen=True)
class Climatology:
    """Climatological forecast per unit [m, K] and how each group's curve was estimated."""

    p: np.ndarray
    source: dict[str, str]


def climatology(f: ForecastPredictions, *, train: np.ndarray, group: Any = None) -> Climatology:
    """Kaplan-Meier base rate of infiltration within k steps on the training triggers, per group.

    train: boolean [m], the training triggers of the record; group: labels [m] (None: one group).
    """
    tr = np.asarray(train, dtype=bool)
    if tr.shape != (f.p_inf.shape[0],):
        raise InvariantViolation("train must flag every trigger")
    if not tr.any():
        raise MissingReference("climatology needs training triggers (split 'train') in the record")
    gcodes, labels = codes(np.zeros(tr.size, dtype=np.int64) if group is None else np.asarray(group).astype(str))
    w = f.window_seconds
    grid = w * np.arange(1, f.horizon + 1, dtype=np.float64)
    observed = f.event_step > 0
    t = np.where(observed, f.event_step * w, f.observed_steps * w).astype(np.float64)

    def curve(mask: np.ndarray) -> np.ndarray:
        km = kaplan_meier(t[mask], observed[mask])
        return 1.0 - km.at(grid)[0]

    pooled = curve(tr)
    p = np.empty_like(f.p_inf)
    source: dict[str, str] = {}
    for g, name in enumerate(labels):
        members = gcodes == g
        sel = members & tr
        if sel.any():
            p[members] = curve(sel)
            source[str(name)] = f"training triggers of the group ({int(sel.sum())})"
        else:
            p[members] = pooled
            source[str(name)] = f"pooled training triggers ({int(tr.sum())}); the group has none"
    return Climatology(np.clip(p, 0.0, 1.0), source)


def reference_record(f: ForecastPredictions, p: np.ndarray) -> ForecastPredictions:
    """A forecast record with the same triggers, outcomes and metadata as f and forecasts p [m, K]."""
    return ForecastPredictions(p_inf=np.maximum.accumulate(np.clip(p, 0.0, 1.0), axis=1),
                               window_seconds=f.window_seconds, event_step=f.event_step,
                               observed_steps=f.observed_steps, meta=f.meta, infiltrated_now=f.infiltrated_now)


def reporting_horizons(spec: Any, k_max: int) -> tuple[int, ...]:
    """Resolve horizon names: integers, "K" (the full horizon) and "mid" (ceil(K / 2)); sorted, unique."""
    out: list[int] = []
    for item in spec:
        if isinstance(item, str):
            if item == "K":
                out.append(k_max)
            elif item == "mid":
                out.append(math.ceil(k_max / 2))
            else:
                out.append(int(item))
        else:
            out.append(int(item))
    if any(not 1 <= h <= k_max for h in out):
        raise InvariantViolation(f"reporting horizons {out} must lie in 1 ... {k_max}")
    return tuple(sorted(set(out)))
