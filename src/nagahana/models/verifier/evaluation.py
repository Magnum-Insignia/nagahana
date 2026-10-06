"""Held-out evaluation of a candidate update: proper scores before and after, drift, KL, and the promotion gates (AS-842, AS-843).

A candidate is judged on feedback that arrived after everything it was fitted on (learning.split_batch).
Every comparison is paired: the same held-out unit is scored under the reference (the deployed weights
and temperatures) and under the candidate, and the per-unit improvement d_i (positive = the candidate is
better) is summarised by its weighted mean and a bootstrap interval (BCa, Efron, JASA 82, 1987; the
weight-matrix bootstrap of `evaluation.resampling`).

What is compared, per target

    forecaster   held-out confirmed outcomes: the trigger is re-imagined by the reference and by the
                 candidate with the same random numbers (one generator seed per situation, so both draw
                 from identical uniforms; the Advisor's common-random-numbers device), and each mixture
                 forecast is scored by the right-censored log score, the Brier score at the horizon and
                 the stage log score at confirmed steps (rewards.forecast_scores), all under the p_inf
                 temperature in force. RLHF candidates are also scored on held-out preferences.
    advisor      held-out advisory preferences.
    verifier     held-out pairs of every family with a proposed temperature (Brier and log score of
                 sigma(logit(p) / T) under the temperature in force and the proposed one) and held-out
                 decisions for the trust head (Brier of its confidence).

Preferences: under the reference the implicit reward margin is 0 and the DPO loss is log 2 for every
label; the candidate's held-out DPO loss, its improvement over log 2, the share of decided preferences
whose margin has the analyst's sign, and the Bradley-Terry reward model's held-out accuracy are reported.

Drift (D-21: the Monitor raises alerts for human review): the held-out (P_inf(K), outcome) pairs, in time
order, are fed to one fresh `monitor.Monitor` per side; an alert kind the candidate's pairs raise and the
reference's do not is a new drift alert.

KL: the mean per-step KL(candidate || reference) on routes drawn from the reference at the held-out
situations (policies.analytic_step_kl with pi_old = pi_ref: the average KL on the states the deployed
policy visits); for the Advisor on the steps of the held-out advisories.

Gates (all must pass; `EvaluationReport.passed`):

    data:<unit>              at least the configured number of held-out units
    improvement:<metric>     the mean improvement of the method's own objective is >= 0 (RLHF: DPO loss;
                             RLVR: the configured P_inf score; RLCD: each family's RLCD score and the trust Brier)
    non-inferiority:<metric> the lower confidence bound of the improvement of every other proper score is
                             >= -margin (the non-inferiority test of a two-sided interval at `confidence`)
    preference-accuracy      RLHF: share of correctly ordered held-out preferences >= the configured minimum
    kl                       policy targets: mean per-step KL <= max_kl
    drift                    no new Monitor alert kind (when required)

A failed gate blocks promotion unless the promoting command's holder explicitly accepts the failure,
which the ledger records (AS-845).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from nagahana.core.errors import InvariantViolation
from nagahana.evaluation.resampling import BootstrapSettings, estimate, iid_scheme
from nagahana.models.advisor.model import Advisor
from nagahana.models.batch import ForecastOut
from nagahana.models.config.components import VerifierConfig
from nagahana.models.forecaster.model import Forecaster
from nagahana.models.verifier.calibration import apply_temperature, logit
from nagahana.models.verifier.candidates import CandidateUpdate
from nagahana.models.verifier.canonical import digest_of
from nagahana.models.verifier.config import FeedbackLearningConfig
from nagahana.models.verifier.feedback import OutcomeConfirmation, PreferenceFeedback
from nagahana.models.verifier.heads import N_FORECAST_FEATURES
from nagahana.models.verifier.learning import Exclusion, Exclusions, FeedbackBatch, derive_seed, resolve_situation
from nagahana.models.verifier.model import VerifierNet
from nagahana.models.verifier.monitor import Monitor
from nagahana.models.verifier.params import call_with, overrides_from
from nagahana.models.verifier.policies import ForecasterPolicy, analytic_step_kl
from nagahana.models.verifier.reports import TemperatureState
from nagahana.models.verifier.rewards import binary_score, forecast_scores
from nagahana.models.verifier.rlcd import RLCDLearner, rlcr_trust_loss
from nagahana.models.verifier.rlhf import RLHFLearner, dpo_loss
from nagahana.models.verifier.rlvr import usable_outcomes
from nagahana.roles.contracts import OutcomeForecastPair

MarginalEnergy = Callable[[torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class Comparison:
    """A paired before/after comparison: improvement d = (after - before) for scores, (before - after) for losses."""

    name: str
    n: int
    before: float
    after: float
    improvement: float
    low: float
    high: float
    method: str

    def to_record(self) -> dict[str, Any]:
        return {"name": self.name, "n": self.n, "before": self.before, "after": self.after, "improvement": self.improvement,
                "low": self.low, "high": self.high, "method": self.method}


@dataclass(frozen=True)
class Gate:
    """One promotion gate and why it passed or failed."""

    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class EvaluationReport:
    """The held-out evaluation of one candidate (module docstring)."""

    candidate_id: str
    method: str
    target: str
    comparisons: tuple[Comparison, ...]
    gates: tuple[Gate, ...]
    kl: float
    drift_before: tuple[str, ...]
    drift_after: tuple[str, ...]
    n_held_out: int
    excluded: tuple[Exclusion, ...]
    stats: Mapping[str, float]

    @property
    def passed(self) -> bool:
        """True when every gate passed."""
        return all(g.passed for g in self.gates)

    def failed(self) -> tuple[str, ...]:
        """Names of the failed gates."""
        return tuple(g.name for g in self.gates if not g.passed)

    def to_record(self) -> dict[str, Any]:
        return {"candidate_id": self.candidate_id, "method": self.method, "target": self.target,
                "comparisons": [c.to_record() for c in self.comparisons],
                "gates": [{"name": g.name, "passed": g.passed, "detail": g.detail} for g in self.gates],
                "kl": self.kl, "drift_before": list(self.drift_before), "drift_after": list(self.drift_after),
                "n_held_out": self.n_held_out,
                "excluded": [{"index": e.index, "event_id": e.event_id, "reason": e.reason} for e in self.excluded],
                "stats": {k: float(v) for k, v in sorted(self.stats.items())}, "passed": self.passed}

    @property
    def evaluation_id(self) -> str:
        """SHA-256 of the report's record."""
        return digest_of(self.to_record())


def paired(name: str, before: Sequence[float], after: Sequence[float], weights: Sequence[float] | None, *, higher_is_better: bool,
           cfg: FeedbackLearningConfig, seed_parts: Sequence[object]) -> Comparison:
    """Weighted mean improvement with a bootstrap interval (module docstring); NaN interval for fewer than 2 units."""
    b = np.asarray(before, dtype=np.float64)
    a = np.asarray(after, dtype=np.float64)
    w = np.ones_like(b) if weights is None else np.asarray(weights, dtype=np.float64)
    keep = np.isfinite(a) & np.isfinite(b) & (w > 0)
    b, a, w = b[keep], a[keep], w[keep]
    n = int(b.size)
    if n == 0:
        return Comparison(name, 0, math.nan, math.nan, math.nan, math.nan, math.nan, "none")
    d = (a - b) if higher_is_better else (b - a)
    mean_b = float((w * b).sum() / w.sum())
    mean_a = float((w * a).sum() / w.sum())
    mean_d = float((w * d).sum() / w.sum())
    if n < 2 or float(np.ptp(d)) == 0.0:
        return Comparison(name, n, mean_b, mean_a, mean_d, mean_d if n >= 2 else math.nan, mean_d if n >= 2 else math.nan,
                          "degenerate" if n >= 2 else "none")
    ev = cfg.evaluation

    def stat(weights_matrix: np.ndarray) -> np.ndarray:
        ww = weights_matrix * w[None, :]
        den = ww.sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            return ((ww @ d) / den)[:, None]

    rng = np.random.default_rng(derive_seed(cfg.seed, "bootstrap", name, *seed_parts))
    est = estimate(stat, [name], iid_scheme(n), BootstrapSettings(n_resamples=ev.bootstrap_resamples,
                                                                  confidence=ev.confidence, interval="bca"), rng)
    _, lo, hi = est.get(name)
    return Comparison(name, n, mean_b, mean_a, mean_d, lo, hi, est.method[0])


def _non_inferior(c: Comparison, margin: float) -> Gate:
    ok = c.n >= 2 and math.isfinite(c.low) and c.low >= -margin
    return Gate(f"non-inferiority:{c.name}", ok, f"lower bound {c.low:.6g} vs -margin {-margin:g} (n = {c.n})")


def _improves(c: Comparison) -> Gate:
    ok = c.n >= 1 and math.isfinite(c.improvement) and c.improvement >= 0.0
    return Gate(f"improvement:{c.name}", ok, f"mean improvement {c.improvement:.6g} (n = {c.n})")


def _drift(pairs_before: Sequence[tuple[str, float, bool]], pairs_after: Sequence[tuple[str, float, bool]],
           vcfg: VerifierConfig) -> tuple[tuple[str, ...], tuple[str, ...]]:
    # Feed both sides' held-out pairs through fresh Monitors; return the alert kinds each raised.
    kinds: list[tuple[str, ...]] = []
    for pairs in (pairs_before, pairs_after):
        mon = Monitor(vcfg, region_dims={})
        for fid, p, y in pairs:
            mon.record_resolution(OutcomeForecastPair(fid, float(min(max(p, 0.0), 1.0)), bool(y)))
        kinds.append(tuple(sorted({a.kind for a in mon.report().alerts})))
    return kinds[0], kinds[1]


def _forecaster_outcomes(cand: CandidateUpdate, held: FeedbackBatch, cfg: FeedbackLearningConfig, forecaster: Forecaster,
                         vcfg: VerifierConfig, t_inf: float, exposure: MarginalEnergy | None, exclusions: Exclusions,
                         comparisons: list[Comparison], gates: list[Gate], stats: dict[str, float]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    # Re-imagine every usable held-out outcome under the reference and the candidate (common random numbers).
    ev = cfg.evaluation
    oc = cfg.outcome_reward
    triggers = usable_outcomes(held, oc, exclusions)
    overrides = overrides_from(forecaster, cand.deltas)
    n_routes = ev.routes_n if ev.routes_n > 0 else forecaster.cfg.routes_n
    rows: dict[str, list[float]] = {f"{k}_{side}": [] for k in ("log", "brier", "stage_log") for side in ("before", "after")}
    weights: list[float] = []
    pairs_b: list[tuple[str, float, bool]] = []
    pairs_a: list[tuple[str, float, bool]] = []
    for trig in sorted(triggers, key=lambda t: (t.outcome.provenance.time, t.index)):
        an1 = trig.situation.analysis
        k = trig.outcome.horizon_k
        seed = derive_seed(cfg.seed, "evaluate", trig.situation.situation_id, trig.index)
        with torch.no_grad():
            fo_b: ForecastOut = forecaster.imagine(an1, horizon_k=k, routes_n=n_routes, generator=torch.Generator().manual_seed(seed),
                                                   exposure=exposure)
            fo_a: ForecastOut = call_with(forecaster, overrides, Forecaster.imagine, an1, horizon_k=k, routes_n=n_routes,
                                          generator=torch.Generator().manual_seed(seed), exposure=exposure)
        for side, fo in (("before", fo_b), ("after", fo_a)):
            p = torch.cummax(apply_temperature(fo.p_inf[0, 0], t_inf), dim=-1).values if t_inf != 1.0 else fo.p_inf[0, 0].double()
            sc = forecast_scores(p, fo.stage[0, 0], trig.outcome, eps=oc.log_eps)
            for key in ("log", "brier", "stage_log"):
                rows[f"{key}_{side}"].append(sc[key])
            if trig.outcome.known_at_horizon:
                (pairs_b if side == "before" else pairs_a).append((trig.situation.situation_id, float(p[-1]), trig.outcome.occurred))
        weights.append(trig.weight)
    n = len(weights)
    stats["held_out_outcomes"] = float(n)
    gates.append(Gate("data:outcomes", n >= ev.min_outcomes, f"{n} held-out confirmed outcomes (need {ev.min_outcomes})"))
    margins = {"log": ev.margin_log, "brier": ev.margin_brier, "stage_log": ev.margin_stage}
    primary = oc.p_inf_score if cand.method == "rlvr" else None
    for key in ("log", "brier", "stage_log"):
        c = paired(f"outcome_{key}", rows[f"{key}_before"], rows[f"{key}_after"], weights, higher_is_better=True, cfg=cfg,
                   seed_parts=(cand.candidate_id,))
        if c.n == 0:
            continue
        comparisons.append(c)
        gates.append(_non_inferior(c, margins[key]))
        if key == primary:
            gates.append(_improves(c))
    return _drift(pairs_b, pairs_a, vcfg)


def _preferences(cand: CandidateUpdate, held: FeedbackBatch, cfg: FeedbackLearningConfig, module: Forecaster | Advisor,
                 exclusions: Exclusions, comparisons: list[Comparison], gates: list[Gate], stats: dict[str, float]) -> list[float]:
    # Held-out DPO loss, preference accuracy and reward-model accuracy; returns per-step KLs on the items' steps.
    learner = RLHFLearner(cfg, target=cand.target, module=module)
    examples = learner.examples(held, exclusions)
    overrides = overrides_from(module, cand.deltas)
    beta = cfg.dpo.beta
    before, after, correct, rm_correct, decided = [], [], 0, 0, 0
    kls: list[float] = []
    for ex in examples:
        with torch.no_grad():
            lp1, lp2 = learner.item_logps(ex, overrides)
            loss, h = dpo_loss(lp1[None], lp2[None], torch.tensor([ex.ref_first]), torch.tensor([ex.ref_second]),
                               torch.tensor([ex.event.label]), beta=beta)
            before.append(math.log(2.0))
            after.append(float(loss[0]))
            if ex.event.preferred != "tie":
                decided += 1
                correct += int((float(h[0]) > 0) == (ex.event.preferred == "first"))
                if cand.reward_model is not None:
                    rm_correct += int((cand.reward_model.prob_first(ex.event) > 0.5) == (ex.event.preferred == "first"))
            kls.extend(_item_kl(learner, ex, overrides))
    n = len(examples)
    stats["held_out_preferences"] = float(n)
    gates.append(Gate("data:preferences", n >= cfg.evaluation.min_preferences,
                      f"{n} held-out preferences (need {cfg.evaluation.min_preferences})"))
    c = paired("dpo_loss", before, after, None, higher_is_better=False, cfg=cfg, seed_parts=(cand.candidate_id,))
    if c.n:
        comparisons.append(c)
        gates.append(_improves(c))
    acc = correct / decided if decided else math.nan
    stats["held_out_preference_accuracy"] = acc
    if cand.reward_model is not None:
        stats["held_out_reward_model_accuracy"] = rm_correct / decided if decided else math.nan
    gates.append(Gate("preference-accuracy", decided > 0 and acc >= cfg.evaluation.min_preference_accuracy,
                      f"{correct} of {decided} decided preferences ordered correctly (need {cfg.evaluation.min_preference_accuracy:g})"))
    return kls


def _item_kl(learner: RLHFLearner, ex: Any, overrides: Mapping[str, torch.Tensor]) -> list[float]:
    # Per-step KL(candidate || reference) on the steps of a preference's items (given prefixes, w = 1).
    if isinstance(learner.policy, ForecasterPolicy):
        cand = learner.policy.step_log_probs(ex.analysis, ex.tech, ex.tgt, overrides=overrides)
        ref = learner.policy.step_log_probs(ex.analysis, ex.tech, ex.tgt, overrides=None)
    else:
        cand = learner.policy.step_log_probs(ex.analysis, list(ex.plans), overrides=overrides)
        ref = learner.policy.step_log_probs(ex.analysis, list(ex.plans), overrides=None)
    d = analytic_step_kl(cand, ref, None)
    return [float(x) for x in d[cand.mask]]


def _forecaster_kl(cand: CandidateUpdate, held: FeedbackBatch, cfg: FeedbackLearningConfig, forecaster: Forecaster) -> list[float]:
    # KL on routes drawn from the reference at every held-out situation with a usable outcome or preference.
    policy = ForecasterPolicy(forecaster)
    overrides = overrides_from(forecaster, cand.deltas)
    seen: set[str] = set()
    out: list[float] = []
    for _, ev in held.events:
        if not isinstance(ev, OutcomeConfirmation | PreferenceFeedback):
            continue
        s, _ = resolve_situation(held, ev)
        if s is None or s.situation_id in seen:
            continue
        seen.add(s.situation_id)
        k = ev.horizon_k if isinstance(ev, OutcomeConfirmation) else (ev.first.routes[0].horizon if ev.first.routes else 0)
        if k < 1:
            continue
        gen = torch.Generator().manual_seed(derive_seed(cfg.seed, "evaluate-kl", s.situation_id))
        tech, tgt = policy.sample(s.analysis, cfg.evaluation.kl_routes, k, gen, overrides=None)
        with torch.no_grad():
            ref = policy.step_log_probs(s.analysis, tech, tgt, overrides=None)
            new = policy.step_log_probs(s.analysis, tech, tgt, overrides=overrides)
            d = analytic_step_kl(new, ref, ref.first_logp)
        out.extend(float(x) for x in d[new.mask])
    return out


def _verifier(cand: CandidateUpdate, held: FeedbackBatch, cfg: FeedbackLearningConfig, verifier: VerifierNet, vcfg: VerifierConfig,
              temps: TemperatureState, exclusions: Exclusions, comparisons: list[Comparison], gates: list[Gate],
              stats: dict[str, float]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    # Held-out pairs per proposed family and held-out trust decisions.
    ev = cfg.evaluation
    learner = RLCDLearner(cfg, verifier=verifier, verifier_cfg=vcfg)
    fam, dec = learner.collect(held, exclusions)
    drift: tuple[tuple[str, ...], tuple[str, ...]] = ((), ())
    score = cfg.rlcd.score
    other = "log" if score == "brier" else "brier"
    margin_other = ev.margin_log if other == "log" else ev.margin_brier
    proposed = {} if cand.temperatures is None else dict(cand.temperatures.temperatures)
    for f, t_new in sorted(proposed.items()):
        pairs = fam[f]
        n = len(pairs)
        stats[f"held_out_pairs_{f}"] = float(n)
        gates.append(Gate(f"data:pairs_{f}", n >= ev.min_pairs, f"{n} held-out {f} pairs (need {ev.min_pairs})"))
        if n == 0:
            continue
        p, y, w = pairs.tensors()
        t_old = float(temps.temperatures.get(f, 1.0))
        p_b, p_a = apply_temperature(p, t_old), apply_temperature(p, t_new)
        for rule in (score, other):
            c = paired(f"{f}_{rule}", binary_score(p_b, y, rule).tolist(), binary_score(p_a, y, rule).tolist(), w.tolist(),
                       higher_is_better=True, cfg=cfg, seed_parts=(cand.candidate_id,))
            comparisons.append(c)
            gates.append(_improves(c) if rule == score else _non_inferior(c, margin_other))
        if f == "p_inf":
            order = range(n)
            drift = _drift([(str(pairs.index[i]), float(p_b[i]), bool(y[i] > 0.5)) for i in order],
                           [(str(pairs.index[i]), float(p_a[i]), bool(y[i] > 0.5)) for i in order], vcfg)
    trust_names = [n for n in cand.deltas if n.startswith("trust.")]
    if trust_names:
        n = len(dec)
        stats["held_out_decisions"] = float(n)
        gates.append(Gate("data:decisions", n >= ev.min_pairs, f"{n} held-out decisions (need {ev.min_pairs})"))
        if n:
            x = torch.stack(dec.features)
            c_ = torch.tensor(dec.correct, dtype=torch.float64)
            ff, mf = x[:, :N_FORECAST_FEATURES], x[:, N_FORECAST_FEATURES:]
            overrides = overrides_from(verifier, cand.deltas)
            with torch.no_grad():
                q_b = torch.sigmoid(verifier.trust(ff, mf).double())
                q_a = torch.sigmoid(call_with(verifier, overrides, lambda v, a, b: v.trust(a, b), ff, mf).double())
            c = paired("trust_brier", (-(q_b - c_) ** 2).tolist(), (-(q_a - c_) ** 2).tolist(), dec.w, higher_is_better=True,
                       cfg=cfg, seed_parts=(cand.candidate_id,))
            comparisons.append(c)
            gates.append(_improves(c))
            stats["held_out_trust_brier_after"] = float(rlcr_trust_loss(logit(q_a), c_, torch.tensor(dec.w)))
    return drift


def evaluate_candidate(candidate: CandidateUpdate, held_out: FeedbackBatch, *, cfg: FeedbackLearningConfig,
                       verifier_cfg: VerifierConfig, forecaster: Forecaster | None = None, advisor: Advisor | None = None,
                       verifier: VerifierNet | None = None, temperatures: TemperatureState | None = None,
                       exposure: MarginalEnergy | None = None) -> EvaluationReport:
    """The held-out evaluation and gates of one candidate (module docstring). No weight or temperature changes."""
    temps = temperatures if temperatures is not None else TemperatureState()
    exclusions = Exclusions()
    comparisons: list[Comparison] = []
    gates: list[Gate] = []
    stats: dict[str, float] = {}
    drift: tuple[tuple[str, ...], tuple[str, ...]] = ((), ())
    kls: list[float] = []
    if candidate.target == "forecaster":
        if forecaster is None:
            raise InvariantViolation("a forecaster candidate is evaluated with the deployed Forecaster")
        drift = _forecaster_outcomes(candidate, held_out, cfg, forecaster, verifier_cfg, float(temps.temperatures.get("p_inf", 1.0)),
                                     exposure, exclusions, comparisons, gates, stats)
        if candidate.method == "rlhf":
            kls += _preferences(candidate, held_out, cfg, forecaster, exclusions, comparisons, gates, stats)
        kls += _forecaster_kl(candidate, held_out, cfg, forecaster)
    elif candidate.target == "advisor":
        if advisor is None:
            raise InvariantViolation("an advisor candidate is evaluated with the deployed Advisor")
        kls += _preferences(candidate, held_out, cfg, advisor, exclusions, comparisons, gates, stats)
    else:
        if verifier is None:
            raise InvariantViolation("a verifier candidate is evaluated with the deployed VerifierNet")
        drift = _verifier(candidate, held_out, cfg, verifier, verifier_cfg, temps, exclusions, comparisons, gates, stats)
    kl = float(np.mean(kls)) if kls else math.nan
    if candidate.target in ("forecaster", "advisor"):
        gates.append(Gate("kl", bool(kls) and kl <= cfg.evaluation.max_kl,
                          f"mean per-step KL {kl:.6g} nats over {len(kls)} steps (max {cfg.evaluation.max_kl:g})"))
    if cfg.evaluation.require_no_new_drift_alerts:
        new = sorted(set(drift[1]) - set(drift[0]))
        gates.append(Gate("drift", not new, "no new Monitor alert" if not new else f"new Monitor alerts: {new}"))
    return EvaluationReport(candidate_id=candidate.candidate_id, method=candidate.method, target=candidate.target,
                            comparisons=tuple(comparisons), gates=tuple(gates), kl=kl, drift_before=drift[0], drift_after=drift[1],
                            n_held_out=len(held_out), excluded=exclusions.as_tuple(), stats=stats)


__all__ = ["Comparison", "EvaluationReport", "Gate", "evaluate_candidate", "paired"]
