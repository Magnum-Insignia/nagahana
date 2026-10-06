"""Verifiable rewards from confirmed outcomes: proper scores, censoring and responded-to outcomes (AS-839, AS-840).

RLVR rewards a route by how well its own forecast scored once the horizon has elapsed; no learned
reward model is involved. The outcome contract is the one of `evaluation/predictions.py`: for a forecast
of K steps, `event_step` is the first infiltration step (1 ... K) or 0, and `observed_steps` (0 ... K)
says how many steps were observed; event_step = 0 with observed_steps = c < K is right-censored after c.

Proper scores of a binary event with forecast p and outcome y (higher is better):

    log score    S_log(p, y)   = y log p + (1 - y) log(1 - p)        (Good, JRSS-B 14, 1952)
    Brier score  S_brier(p, y) = -(p - y)^2                          (Brier, Monthly Weather Review 78, 1950)

Under y ~ Bernoulli(q) the expected scores are q log p + (1 - q) log(1 - p) and -(p - q)^2 - q(1 - q);
both have their unique maximum at p = q (Gibbs' inequality; completing the square), so neither can be
gamed by inflating or deflating p (Gneiting and Raftery, JASA 102, 2007).

Right-censored survival log score of a route (the default infiltration reward). With hazards
h_j = P(first infiltration at j | none before j):

    event at e:            log h_e + sum_(j<e) log(1 - h_j)
    censored after c:      sum_(j<=c) log(1 - h_j)          (c = K: no infiltration within the horizon)

This is the log-likelihood of exactly what was observed, i.e. the log score of the censored observation,
and the same quantity the Forecaster's survival loss minimises (`forecaster.losses.hazard_nll`, reused
here). Under non-informative censoring its expectation decomposes into one Bernoulli log score per step
j, weighted by P(at risk and observed at j), so it is maximised exactly at the true hazards on every
step that can be observed: the right-censored log-likelihood is a proper scoring rule (Rindt, Hu,
Steinsaltz and Sejdinovic, "Survival regression with proper scoring rules and monotonic neural networks",
AISTATS 2022, arXiv:2103.14755). A trigger with no observed step carries no information (reward 0) and
is not used. For a route mixture P(k) the same score is computed from the mixture's hazards
(`routes.mixture_hazard`): event at e gives log(P(e) - P(e - 1)), censoring after c gives log(1 - P(c)).

Brier score at the horizon. -(F(K) - y)^2 with y = 1[infiltration within K]. The outcome is known only
when infiltration was observed or all K steps were observed; the others are excluded, exactly as in
`ForecastPredictions.outcome(K)`. Because whether an outcome is known depends on the outcome itself (an
event is seen however early censoring comes, a survival only if observation lasts K steps), plain
exclusion favours events. With "ipcw" each known outcome is weighted by the inverse probability of
remaining uncensored (Graf, Schmoor, Sauerbrei and Schumacher, Statistics in Medicine 18, 1999):

    event at T <= K:     w = 1 / G(T-)          survived all K steps:   w = 1 / G(K-)

with G the Kaplan-Meier estimate of the censoring survival function (`evaluation.survival.kaplan_meier`,
events before censorings at ties), truncated at `max_ipw_weight`.

Responded-to outcomes (AS-840). When the SOC acted on a forecast, its outcome is counterfactual (what
would have happened without the response is not observed) and it is never scored as right or wrong
[Q-38], exactly as `calibration.calibration_report`, `monitor.Monitor` and `rlcd.brier_reward` exclude
responded-to pairs. Exclusion alone selects the cases the response policy did not act on; when that
policy's propensity rho_i = P(respond | forecast i) is recorded, "inverse-propensity" weights every scored
case by 1 / (1 - rho_i) (Horvitz and Thompson, JASA 47, 1952), truncated at `max_ipw_weight` (Ionides,
JCGS 17, 2008) and self-normalised in every objective, which makes the objective an estimate of its value
on all cases under the no-response counterfactual (consistent when the response depends on recorded
information only and rho < 1). A scored case without a recorded propensity is excluded in that mode.

Stage rewards at confirmed steps (route stage = argmax of its stage posterior at the step):

    correct   1[argmax_s p_k(s) = s*]       consistent for the mode (Gneiting, JASA 106, 2011)
    log       log p_k(s*)                    strictly proper
    brier     -sum_s (p_k(s) - 1[s = s*])^2  strictly proper (multi-class Brier)

averaged over the confirmed steps. A route's reward is S_inf + stage_weight * mean stage score.

Precision (D-54): every score is float64.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np
import torch

from nagahana.core.errors import InvariantViolation
from nagahana.evaluation.survival import kaplan_meier
from nagahana.models.forecaster.losses import hazard_nll
from nagahana.models.forecaster.routes import mixture_hazard, route_cumulative
from nagahana.models.verifier.config import OutcomeRewardConfig
from nagahana.models.verifier.feedback import OutcomeConfirmation


def binary_score(p: torch.Tensor, y: torch.Tensor, rule: str, *, eps: float = 1e-6) -> torch.Tensor:
    """S_log or S_brier of probabilities p against outcomes y in {0, 1} (module docstring), float64."""
    pp, yy = p.double(), y.double()
    if rule == "log":
        q = pp.clamp(eps, 1.0 - eps)
        return yy * torch.log(q) + (1.0 - yy) * torch.log1p(-q)
    if rule == "brier":
        return -((pp - yy) ** 2)
    raise InvariantViolation(f"unknown binary score {rule!r}; use 'log' or 'brier'")


def survival_log_score(hazard: torch.Tensor, event_step: torch.Tensor, observed_steps: torch.Tensor, *,
                       eps: float = 1e-6) -> torch.Tensor:
    """Right-censored discrete-time log score of hazards [..., K] (module docstring), float64 [...].

    event_step: 1 ... K for an observed infiltration, 0 for none; observed_steps: 0 ... K.
    """
    censored = event_step <= 0
    steps = torch.where(censored, observed_steps, event_step).long()
    return -hazard_nll(hazard.double(), steps, censored, eps=eps)


def curve_log_score(p_inf: torch.Tensor, event_step: torch.Tensor, observed_steps: torch.Tensor, *,
                    eps: float = 1e-6) -> torch.Tensor:
    """The same score for a cumulative curve P(k) [..., K] through its hazards (`routes.mixture_hazard`)."""
    return survival_log_score(mixture_hazard(p_inf.double()), event_step, observed_steps, eps=eps)


def outcome_at_horizon(event_step: torch.Tensor, observed_steps: torch.Tensor, horizon_k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """(y, known) of 'infiltration within K' (ForecastPredictions.outcome semantics)."""
    happened = (event_step > 0) & (event_step <= horizon_k)
    known = happened | (observed_steps >= horizon_k)
    return happened.double(), known


def brier_at_horizon(cumulative: torch.Tensor, event_step: torch.Tensor, observed_steps: torch.Tensor
                     ) -> tuple[torch.Tensor, torch.Tensor]:
    """(-(F(K) - y)^2, known) for curves [..., K]; the score is 0 where the outcome is unknown."""
    k = cumulative.shape[-1]
    y, known = outcome_at_horizon(event_step, observed_steps, k)
    s = -((cumulative[..., -1].double() - y) ** 2)
    return torch.where(known, s, torch.zeros_like(s)), known


def ipcw_weights(event_step: Sequence[int], observed_steps: Sequence[int], horizon_k: Sequence[int], *,
                 max_weight: float) -> np.ndarray:
    """Inverse-probability-of-censoring weights of 'infiltration within K' outcomes (module docstring).

    Times are in imagined steps: T = event step for events, observed steps otherwise (censored at T).
    Known outcomes get 1 / G(T-) (events) or 1 / G(K-) (all K steps observed); unknown ones get 0.
    """
    e = np.asarray(event_step, dtype=np.int64)
    c = np.asarray(observed_steps, dtype=np.int64)
    k = np.asarray(horizon_k, dtype=np.int64)
    if not (e.shape == c.shape == k.shape):
        raise InvariantViolation("event_step, observed_steps and horizon_k must have the same length")
    if e.size == 0:
        return np.zeros(0)
    t = np.where(e > 0, e, c).astype(np.float64)
    d = e > 0
    g = kaplan_meier(t, d, censoring=True)
    g_event = g.left(t)[0]                                              # G(T-) of each unit [n]
    g_k = g.left(k.astype(np.float64))[0]                               # G(K-) of each unit [n]
    w = np.zeros(e.size)
    event_known = d & (e <= k)
    survived = ~d & (c >= k)
    with np.errstate(divide="ignore"):
        w = np.where(event_known & (g_event > 0), 1.0 / np.where(g_event > 0, g_event, 1.0), w)
        w = np.where(survived & (g_k > 0), 1.0 / np.where(g_k > 0, g_k, 1.0), w)
    return np.minimum(w, float(max_weight))


def response_weights(outcomes: Sequence[OutcomeConfirmation], *, mode: str, max_weight: float
                     ) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """(weight [n], keep [n], reason per excluded outcome) for responded-to handling (module docstring)."""
    n = len(outcomes)
    w = np.ones(n)
    keep = np.ones(n, dtype=bool)
    reasons = [""] * n
    for i, o in enumerate(outcomes):
        if o.responded_to:
            keep[i], reasons[i] = False, "responded-to: counterfactual outcome, not scored [Q-38]"
            continue
        if mode == "exclude":
            continue
        if mode != "inverse-propensity":
            raise InvariantViolation(f"unknown responded-to mode {mode!r}")
        if o.response_propensity is None:
            keep[i], reasons[i] = False, "no recorded response propensity (inverse-propensity mode)"
            continue
        w[i] = min(1.0 / (1.0 - float(o.response_propensity)), float(max_weight))
    return w, keep, reasons


def stage_scores(stage_logits: torch.Tensor, confirmed: Mapping[int, int], rule: str) -> torch.Tensor:
    """Mean stage score over confirmed steps for R routes: stage_logits [R, K, S] -> [R] float64.

    confirmed maps a forecast step (1 ... K) to its confirmed stage code. Steps outside 1 ... K are
    ignored; with no confirmed step the score is 0.
    """
    r, k, s = stage_logits.shape
    steps = sorted(step for step in confirmed if 1 <= step <= k)
    if not steps:
        return torch.zeros(r, dtype=torch.float64, device=stage_logits.device)
    idx = torch.tensor([step - 1 for step in steps], dtype=torch.long, device=stage_logits.device)
    codes = torch.tensor([confirmed[step] for step in steps], dtype=torch.long, device=stage_logits.device)
    if bool((codes < 0).any()) or bool((codes >= s).any()):
        raise InvariantViolation(f"confirmed stage codes must lie in 0 ... {s - 1}")
    logp = torch.log_softmax(stage_logits.double()[:, idx], dim=-1)                 # [R, n, S]
    return stage_scores_from_log_probs(logp, codes, rule)


def stage_scores_from_log_probs(logp: torch.Tensor, codes: torch.Tensor, rule: str) -> torch.Tensor:
    """Mean over the n confirmed steps of the stage score, from log-probabilities [R, n, S] and codes [n]."""
    tgt = codes.view(1, -1, 1).expand(logp.shape[0], -1, 1)
    if rule == "correct":
        per = (logp.argmax(-1) == codes.view(1, -1)).double()
    elif rule == "log":
        per = logp.gather(-1, tgt).squeeze(-1)
    elif rule == "brier":
        p = logp.exp()
        onehot = torch.zeros_like(p).scatter_(-1, tgt, 1.0)
        per = -((p - onehot) ** 2).sum(-1)
    else:
        raise InvariantViolation(f"unknown stage score {rule!r}; use 'correct', 'log' or 'brier'")
    return per.mean(-1)


def route_rewards(hazard: torch.Tensor, stage_logits: torch.Tensor | None, outcome: OutcomeConfirmation,
                  cfg: OutcomeRewardConfig) -> tuple[torch.Tensor | None, str]:
    """Rewards [R] of R routes of one trigger against its confirmed outcome, or (None, reason) when unusable.

    hazard [R, K] are the routes' own hazards; stage_logits [R, K, S] their stage logits (None skips the
    stage term). K must equal the outcome's horizon.
    """
    r, k = hazard.shape
    if k != outcome.horizon_k:
        return None, f"route horizon {k} differs from the outcome's horizon {outcome.horizon_k}"
    e = torch.full((r,), int(outcome.event_step), dtype=torch.long, device=hazard.device)
    c = torch.full((r,), int(outcome.observed_steps), dtype=torch.long, device=hazard.device)
    if cfg.p_inf_score == "log":
        if outcome.event_step == 0 and outcome.observed_steps == 0:
            return None, "no observed step: the outcome carries no information"
        reward = survival_log_score(hazard, e, c, eps=cfg.log_eps)
    else:
        s, known = brier_at_horizon(route_cumulative(hazard.double()), e, c)
        if not bool(known.all()):
            return None, "outcome unknown at the horizon (censored before K)"
        reward = s
    confirmed = outcome.stage_codes
    if stage_logits is not None and confirmed and cfg.stage_weight > 0:
        reward = reward + cfg.stage_weight * stage_scores(stage_logits, confirmed, cfg.stage_score)
    if not bool(torch.isfinite(reward).all()):
        raise InvariantViolation("a verifiable reward is not finite")
    return reward, ""


def forecast_scores(p_inf: torch.Tensor, stage: torch.Tensor | None, outcome: OutcomeConfirmation, *,
                    eps: float = 1e-6) -> dict[str, float]:
    """Proper scores of one trigger's mixture forecast: P_inf [K] and stage posterior [K, S] (or None).

    Keys: "log" (censored log score; NaN without an observed step), "brier" (at K; NaN when unknown),
    "stage_log" and "stage_correct" (means over confirmed steps; NaN without one).
    """
    k = p_inf.shape[-1]
    if k != outcome.horizon_k:
        raise InvariantViolation(f"forecast horizon {k} differs from the outcome's horizon {outcome.horizon_k}")
    e = torch.tensor([outcome.event_step])
    c = torch.tensor([outcome.observed_steps])
    out: dict[str, float] = {}
    has_info = outcome.event_step > 0 or outcome.observed_steps > 0
    out["log"] = float(curve_log_score(p_inf.double()[None], e, c, eps=eps)[0]) if has_info else math.nan
    s, known = brier_at_horizon(p_inf.double()[None], e, c)
    out["brier"] = float(s[0]) if bool(known[0]) else math.nan
    confirmed = {step: code for step, code in outcome.stage_codes.items() if 1 <= step <= k}
    if stage is not None and confirmed:
        steps = sorted(confirmed)
        idx = torch.tensor([st - 1 for st in steps], dtype=torch.long)
        codes = torch.tensor([confirmed[st] for st in steps], dtype=torch.long)
        logp = torch.log(stage.double()[idx].clamp_min(1e-300))[None]               # [1, n, S]
        out["stage_log"] = float(stage_scores_from_log_probs(logp, codes, "log")[0])
        out["stage_correct"] = float(stage_scores_from_log_probs(logp, codes, "correct")[0])
    else:
        out["stage_log"] = math.nan
        out["stage_correct"] = math.nan
    return out


__all__ = ["binary_score", "brier_at_horizon", "curve_log_score", "forecast_scores", "ipcw_weights", "outcome_at_horizon",
           "response_weights", "route_rewards", "stage_scores", "stage_scores_from_log_probs", "survival_log_score"]
