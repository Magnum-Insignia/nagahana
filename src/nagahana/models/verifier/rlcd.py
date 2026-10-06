"""RLCD: reinforcement learning for calibrated decisions, the Verifier's calibration objective (P-10, AS-25, D-65; AS-841).

Sources: [Q-34] (RLCD, the calibrated-decisions standard), [Q-38] (responded-to outcomes are not wrong),
proposal P-10 (Brier reward), assumed in AS-25; Damani et al., "Beyond Binary Rewards: Training LMs to
Reason About Their Uncertainty" (RLCR), arXiv:2507.16806, which adds a Brier term on the stated
confidence to the correctness reward.

The reward

    r_i = -(p_i - y_i)^2    for scored pairs; responded-to pairs are excluded (mask False, reward 0)

The Brier score is strictly proper (Brier 1950; Gneiting and Raftery, JASA 2007): its expectation
E_y[r] = -(p - P(y = 1))^2 - Var(y) is maximised exactly at the honest probability, so a policy trained on
it cannot gain by inflating or deflating its forecasts. Precision (D-54): rewards are float64.

The calibrated-decision objective. For an output family f (D-45: P_inf, stage, compromise, advice) with
scored pairs (p_i, y_i) and case weights omega_i (1, or the inverse response propensity and inverse
censoring weights of rewards.py), the Verifier's calibration policy is the temperature map
p -> sigma(logit(p) / T_f) and RLCD maximises the weighted mean reward

    J_f(T) = sum_i omega_i S(sigma(z_i / T), y_i) / sum_i omega_i,      z_i = logit(p_i)

over T in [T_min, T_max], with S the Brier reward (default) or the log score (`RLCDConfig.score`).
Why this calibrates: for a strictly monotone recalibration g the Brier score decomposes as
E[(g(p) - E[y | p])^2] + E[Var(y | p)] (Murphy, J. Applied Meteorology 12, 1973; Broecker, QJRMS 135, 2009);
the second (refinement) term does not depend on g, so maximising J over the temperature family minimises
exactly the calibration term within the family. The log-score optimum is the weighted maximum-likelihood
temperature, found by bisection on the derivative of the convex NLL in beta = 1 / T
(sum_i omega_i z_i (sigma(beta z_i) - y_i), non-decreasing in beta; with unit weights this is
`calibration.ml_temperature`). The Brier objective is not convex in T, so it is maximised globally: a
grid of `grid` points uniform in log T, then golden-section refinement in the bracket around the best
grid point (Kiefer, Proc. AMS 4, 1953) to `tol` in log T.

Pairs per family (AS-841):

    p_inf       outcomes known at the horizon: (raw P_inf(K) of the deployed forecast, 1[infiltration
                within K]); responded-to excluded or inverse-propensity weighted, censoring by IPCW
    stage       top-label pairs (confidence of the predicted stage, 1[it was the confirmed stage]) from
                stage corrections, confirmed outcome stages and alert edits (Gupta and Ramdas, ICLR 2022,
                top-label calibration)
    compromise  alert verdicts on the alerted entity: accept 1, reject 0, an entity edit 0 for the named
                entity and 1 for the corrected one, a stage-only edit 1
    advice      no verifiable outcome exists in the feedback kinds; reported as no pairs

The value head (D-45) learns the RLCR calibration term. The trust head states the confidence
q = sigma(trust(phi_f, phi_m)) that a decision is right; with correctness c in {0, 1} (an alert decision
P_inf(K) >= threshold against the outcome, or an analyst's verdict on a raised alert) its loss is the
weighted Brier score sum_i omega_i (q_i - c_i)^2 / sum_i omega_i (`rlcr_trust_loss`). The calibration
policy head is trained toward log T*_f from the family's reliability features (heads.CalibrationPolicyHead.loss).

`RLCDLearner.fit(batch)` returns one candidate: the temperatures T*_f of the families with at least
`min_pairs` pairs (a CalibrationProposal, applied only by `gate.apply_calibration` under an
"apply-calibration" command) and the deltas of the trust and calibration heads (promoted only under an
"update-weights" command).

Invariants (tested): the rewards exclude responded-to pairs; `rlcd_temperature` recovers a planted
temperature with both scores and equals `ml_temperature` for the log score with unit weights; the
golden-section optimum is a maximum of J; the reference Verifier weights are unchanged by a fit.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import torch

from nagahana.core.config import to_mapping
from nagahana.core.errors import InvariantViolation
from nagahana.governance.assumptions import assume
from nagahana.models.config.components import VerifierConfig
from nagahana.models.forecaster.context import stage_probabilities
from nagahana.models.verifier.calibration import logit
from nagahana.models.verifier.candidates import CandidateUpdate, FitReport
from nagahana.models.verifier.canonical import digest_of
from nagahana.models.verifier.config import FeedbackLearningConfig
from nagahana.models.verifier.feedback import AlertFeedback, OutcomeConfirmation, StageCorrection
from nagahana.models.verifier.heads import N_FORECAST_FEATURES, N_MONITOR_FEATURES, forecast_features, reliability_features
from nagahana.models.verifier.learning import (
    Exclusions,
    FeedbackBatch,
    generator_for,
    make_optimiser,
    minibatches,
    optimiser_step,
    resolve_situation,
)
from nagahana.models.verifier.model import VerifierNet
from nagahana.models.verifier.params import ParameterDelta, call_with, select_parameters, state_hash
from nagahana.models.verifier.reports import OUTPUT_FAMILIES, CalibrationProposal
from nagahana.models.verifier.rewards import ipcw_weights, response_weights
from nagahana.models.verifier.situations import Situation
from nagahana.models.vocab import STAGE_CODE
from nagahana.roles.contracts import OutcomeForecastPair

_GOLDEN = (math.sqrt(5.0) - 1.0) / 2.0


def brier_reward(p: torch.Tensor, y: torch.Tensor, responded_to: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """(reward float64, scored) with reward = -(p - y)^2 on scored pairs and 0 on responded-to pairs."""
    assume("AS-25", by=__name__)
    scored = torch.ones_like(p, dtype=torch.bool) if responded_to is None else ~responded_to.bool()
    # float64 (D-54): an exact promotion of float32 inputs; the squared error is then not rounded to float32.
    r = -(p.double() - y.double()) ** 2
    return torch.where(scored, r, torch.zeros_like(r)), scored


def rewards_from_pairs(pairs: Sequence[OutcomeForecastPair]) -> tuple[torch.Tensor, torch.Tensor]:
    """The same reward over the Verifier ledger's outcome-forecast pairs."""
    p = torch.tensor([x.predicted for x in pairs], dtype=torch.float64)
    y = torch.tensor([float(x.occurred) for x in pairs], dtype=torch.float64)
    resp = torch.tensor([x.responded_to for x in pairs], dtype=torch.bool)
    return brier_reward(p, y, resp)


def rlcd_objective(z: torch.Tensor, y: torch.Tensor, weights: torch.Tensor | None, temperature: float, score: str) -> float:
    """J(T): weighted mean reward of tempered probabilities sigma(z / T) (module docstring), float64."""
    if not temperature > 0:
        raise ValueError("temperature must be > 0")
    zz, yy = z.double(), y.double()
    w = torch.ones_like(zz) if weights is None else weights.double()
    q = torch.sigmoid(zz / temperature)
    if score == "brier":
        s = -((q - yy) ** 2)
    elif score == "log":
        # log sigma(u) = -softplus(-u) and log(1 - sigma(u)) = -softplus(u): exact for saturated logits.
        u = zz / temperature
        s = -(yy * torch.nn.functional.softplus(-u) + (1.0 - yy) * torch.nn.functional.softplus(u))
    else:
        raise InvariantViolation(f"unknown RLCD score {score!r}")
    return float((w * s).sum() / w.sum())


def _weighted_ml_temperature(z: torch.Tensor, y: torch.Tensor, w: torch.Tensor, t_min: float, t_max: float, tol: float) -> float:
    # Bisection on the derivative of the weighted NLL in beta = 1 / T (module docstring).
    def grad(beta: float) -> float:
        return float((w * z * (torch.sigmoid(beta * z) - y)).sum())

    lo, hi = 1.0 / t_max, 1.0 / t_min
    if grad(lo) >= 0.0:
        return t_max
    if grad(hi) <= 0.0:
        return t_min
    for _ in range(400):
        mid = 0.5 * (lo + hi)
        if grad(mid) > 0.0:
            hi = mid
        else:
            lo = mid
        if hi - lo < tol:
            break
    return 1.0 / (0.5 * (lo + hi))


def rlcd_temperature(p: torch.Tensor, y: torch.Tensor, weights: torch.Tensor | None = None, *, score: str, t_min: float,
                     t_max: float, grid: int = 401, tol: float = 1e-10) -> float:
    """T* = argmax_T J(T) on [t_min, t_max] (module docstring). p are probabilities, y outcomes in {0, 1}."""
    if not 0 < t_min < t_max:
        raise ValueError("need 0 < t_min < t_max")
    z = logit(p.double().flatten())
    yy = y.double().flatten()
    if z.numel() == 0 or z.shape != yy.shape:
        raise InvariantViolation("RLCD needs at least one pair and one outcome per probability")
    w = torch.ones_like(z) if weights is None else weights.double().flatten()
    if w.shape != z.shape or bool((w < 0).any()) or not float(w.sum()) > 0:
        raise InvariantViolation("RLCD weights must be non-negative, one per pair, with a positive sum")
    if score == "log":
        return _weighted_ml_temperature(z, yy, w, t_min, t_max, tol)
    if score != "brier":
        raise InvariantViolation(f"unknown RLCD score {score!r}")
    # Global search on a grid in u = log T, then golden-section refinement in the best grid bracket.
    us = np.linspace(math.log(t_min), math.log(t_max), int(grid))
    vals = np.array([rlcd_objective(z, yy, w, math.exp(u), "brier") for u in us])
    best = int(np.argmax(vals))
    a = us[max(best - 1, 0)]
    b = us[min(best + 1, len(us) - 1)]
    c = b - _GOLDEN * (b - a)
    d = a + _GOLDEN * (b - a)
    fc = rlcd_objective(z, yy, w, math.exp(c), "brier")
    fd = rlcd_objective(z, yy, w, math.exp(d), "brier")
    while b - a > tol:
        if fc >= fd:
            b, d, fd = d, c, fc
            c = b - _GOLDEN * (b - a)
            fc = rlcd_objective(z, yy, w, math.exp(c), "brier")
        else:
            a, c, fc = c, d, fd
            d = a + _GOLDEN * (b - a)
            fd = rlcd_objective(z, yy, w, math.exp(d), "brier")
    u_star = 0.5 * (a + b)
    # Keep the best of the refined point and the grid point (the bracket ends are grid points).
    t_star = math.exp(u_star)
    if rlcd_objective(z, yy, w, t_star, "brier") < vals[best]:
        t_star = math.exp(us[best])
    return min(max(t_star, t_min), t_max)


def rlcr_trust_loss(trust_logit: torch.Tensor, correct: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
    """Weighted Brier score of the trust confidence sigma(logit) against decision correctness (float64)."""
    q = torch.sigmoid(trust_logit.double())
    c = correct.double()
    w = torch.ones_like(q) if weights is None else weights.double().to(q.device)
    return (w * (q - c) ** 2).sum() / w.sum().clamp_min(1e-300)


@dataclass
class FamilyPairs:
    """Scored (probability, outcome, weight) pairs of one output family, with the ledger records behind them."""

    p: list[float] = field(default_factory=list)
    y: list[float] = field(default_factory=list)
    w: list[float] = field(default_factory=list)
    index: list[int] = field(default_factory=list)

    def add(self, p: float, y: float, w: float, index: int) -> None:
        self.p.append(float(min(max(p, 0.0), 1.0)))
        self.y.append(float(y))
        self.w.append(float(w))
        self.index.append(int(index))

    def tensors(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (torch.tensor(self.p, dtype=torch.float64), torch.tensor(self.y, dtype=torch.float64),
                torch.tensor(self.w, dtype=torch.float64))

    def __len__(self) -> int:
        return len(self.p)


@dataclass
class Decisions:
    """Resolved decisions for the trust head: features [n, 12 + 6], correctness [n], weights [n], ledger records."""

    features: list[torch.Tensor] = field(default_factory=list)
    correct: list[float] = field(default_factory=list)
    w: list[float] = field(default_factory=list)
    index: list[int] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.correct)


def _top_label(probs: torch.Tensor, code: int) -> tuple[float, float]:
    # (confidence of the predicted class, 1 if it is the confirmed class)
    pr = probs.double()
    return float(pr.max()), 1.0 if int(pr.argmax()) == int(code) else 0.0


def _outcome_weights(batch: FeedbackBatch, cfg: FeedbackLearningConfig, exclusions: Exclusions
                     ) -> dict[int, tuple[OutcomeConfirmation, float]]:
    # Scored outcomes with their response and censoring weights (rewards.py), keyed by ledger index.
    oc = cfg.outcome_reward
    events = [(i, e) for i, e in batch.of_kind("outcome") if isinstance(e, OutcomeConfirmation)]
    w_resp, keep, reasons = response_weights([e for _, e in events], mode=oc.responded_to, max_weight=oc.max_ipw_weight)
    out: dict[int, tuple[OutcomeConfirmation, float]] = {}
    scored = []
    for (i, e), w, k, why in zip(events, w_resp, keep, reasons, strict=True):
        if not k:
            exclusions.add(i, e, why)
            continue
        scored.append((i, e, float(w)))
    censor = {i: 1.0 for i, _, _ in scored}
    if oc.brier_censoring == "ipcw" and scored:
        g = ipcw_weights([e.event_step for _, e, _ in scored], [e.observed_steps for _, e, _ in scored],
                         [e.horizon_k for _, e, _ in scored], max_weight=oc.max_ipw_weight)
        censor = {i: float(x) for (i, _, _), x in zip(scored, g, strict=True)}
    for i, e, w in scored:
        out[i] = (e, w * censor[i])
    return out


class RLCDLearner:
    """Temperatures and trust/calibration-head deltas from calibrated-decision feedback (module docstring).

    Parameters
    ----------
    cfg:
        The feedback-learning configuration (rlcd, outcome_reward, adapter, seed).
    verifier:
        The deployed VerifierNet: its PRM supplies the trust features' plausibility, its heads are the
        reference of the deltas, and its weights are never written.
    verifier_cfg:
        The Verifier's configuration (temperature interval, reliability bins).
    """

    method = "rlcd"
    target = "verifier"

    def __init__(self, cfg: FeedbackLearningConfig, *, verifier: VerifierNet, verifier_cfg: VerifierConfig) -> None:
        self.cfg, self.verifier, self.vcfg = cfg, verifier, verifier_cfg
        names = [n for n, _ in select_parameters(verifier, cfg.adapter.verifier_targets)]
        self.trust_names = [n for n in names if n.startswith("trust.")]
        self.cal_names = [n for n in names if n.startswith("calibration.")]
        other = sorted(set(names) - set(self.trust_names) - set(self.cal_names))
        if other:
            raise InvariantViolation(f"RLCD trains the trust and calibration heads only; the targets also select {other[:4]}")

    # ---- data
    def trust_features(self, s: Situation) -> torch.Tensor | None:
        """phi_f [12] and phi_m [6] of the situation, concatenated [18], or None without forecast or Monitor statistics."""
        if s.forecast is None or s.monitor_features is None:
            return None
        with torch.no_grad():
            plaus = torch.sigmoid(self.verifier.prm(s.analysis, s.forecast))
            ff = forecast_features(s.forecast, plaus)[0, 0]                       # [12]
        return torch.cat([ff.float(), s.monitor_features.float()])

    def collect(self, batch: FeedbackBatch, exclusions: Exclusions) -> tuple[dict[str, FamilyPairs], Decisions]:
        """Pairs per family and trust-head decisions from the batch (module docstring)."""
        fam = {f: FamilyPairs() for f in OUTPUT_FAMILIES}
        dec = Decisions()
        theta = self.cfg.rlcd.decision_threshold
        for idx, (o, w) in _outcome_weights(batch, self.cfg, exclusions).items():
            s, why = resolve_situation(batch, o)
            if s is None:
                exclusions.add(idx, o, why)
                continue
            if s.forecast is None or s.forecast.horizon_k != o.horizon_k:
                exclusions.add(idx, o, "the situation has no deployed forecast of the outcome's horizon")
                continue
            p_k = float(s.forecast.p_inf[0, 0, o.horizon_k - 1])
            if o.known_at_horizon and w > 0:
                y = 1.0 if o.occurred else 0.0
                fam["p_inf"].add(p_k, y, w, idx)
                feats = self.trust_features(s)
                if feats is not None:
                    dec.features.append(feats)
                    dec.correct.append(1.0 if (p_k >= theta) == (y > 0.5) else 0.0)
                    dec.w.append(w)
                    dec.index.append(idx)
            for step, code in o.stage_codes.items():
                if step <= s.forecast.horizon_k:
                    conf, ok = _top_label(s.forecast.stage[0, 0, step - 1], code)
                    fam["stage"].add(conf, ok, 1.0, idx)
        for idx, ev in batch.of_kind("stage"):
            assert isinstance(ev, StageCorrection)
            s, why = resolve_situation(batch, ev)
            if s is None:
                exclusions.add(idx, ev, why)
                continue
            if ev.step == 0:
                assert ev.entity is not None
                if ev.entity >= s.n_entities:
                    exclusions.add(idx, ev, "entity outside the situation's entity table")
                    continue
                probs = stage_probabilities(s.analysis.readouts["stage"])[0, 0, ev.entity]
            else:
                if s.forecast is None or ev.step > s.forecast.horizon_k:
                    exclusions.add(idx, ev, "no deployed forecast covering the corrected step")
                    continue
                probs = s.forecast.stage[0, 0, ev.step - 1]
            conf, ok = _top_label(probs, ev.stage_code)
            fam["stage"].add(conf, ok, 1.0, idx)
        for idx, ev in batch.of_kind("alert"):
            assert isinstance(ev, AlertFeedback)
            s, why = resolve_situation(batch, ev)
            if s is None:
                exclusions.add(idx, ev, why)
                continue
            comp = s.analysis.readouts["compromise"][0, 0]
            v = s.n_entities
            if ev.entity is not None and ev.entity < v:
                y_named = 0.0 if (ev.verdict == "reject" or ev.corrected_entity is not None) else 1.0
                fam["compromise"].add(float(comp[ev.entity]), y_named, 1.0, idx)
                if ev.corrected_entity is not None and ev.corrected_entity < v:
                    fam["compromise"].add(float(comp[ev.corrected_entity]), 1.0, 1.0, idx)
                if ev.corrected_stage is not None:
                    probs = stage_probabilities(s.analysis.readouts["stage"])[0, 0, ev.entity]
                    conf, ok = _top_label(probs, STAGE_CODE[ev.corrected_stage])
                    fam["stage"].add(conf, ok, 1.0, idx)
            elif ev.entity is not None:
                exclusions.add(idx, ev, "entity outside the situation's entity table")
            feats = self.trust_features(s)
            if feats is not None:
                dec.features.append(feats)
                dec.correct.append(1.0 if ev.decision_correct else 0.0)
                dec.w.append(1.0)
                dec.index.append(idx)
        return fam, dec

    # ---- fit
    def fit(self, batch: FeedbackBatch) -> CandidateUpdate:
        """RLCD temperatures + trust and calibration head deltas (module docstring)."""
        rc, vc = self.cfg.rlcd, self.vcfg
        exclusions = Exclusions()
        fam, dec = self.collect(batch, exclusions)
        temps: dict[str, float] = {}
        n_pairs: dict[str, int] = {}
        stats: dict[str, float] = {}
        for f in rc.families:
            pairs = fam[f]
            stats[f"pairs_{f}"] = float(len(pairs))
            if len(pairs) < rc.min_pairs:
                continue
            p, y, w = pairs.tensors()
            if not (0.0 < float((w * y).sum()) < float(w.sum())):
                stats[f"single_outcome_{f}"] = 1.0                               # only one outcome: T is not identified
                continue
            t = rlcd_temperature(p, y, w, score=rc.score, t_min=vc.temperature_min, t_max=vc.temperature_max, grid=rc.grid,
                                 tol=rc.tol)
            temps[f], n_pairs[f] = t, len(pairs)
            stats[f"temperature_{f}"] = t
            stats[f"objective_before_{f}"] = rlcd_objective(logit(p), y, w, 1.0, rc.score)
            stats[f"objective_after_{f}"] = rlcd_objective(logit(p), y, w, t, rc.score)
        train_trust = len(dec) >= rc.min_decisions and len(set(dec.correct)) == 2
        stats["decisions"] = float(len(dec))
        trained_names = (self.trust_names if train_trust else []) + (self.cal_names if temps else [])
        if not temps and not train_trust:
            raise InvariantViolation(f"RLCD has nothing to fit: fewer than {rc.min_pairs} scored pairs in every family with both "
                                     f"outcomes, and {len(dec)} resolved decisions (need {rc.min_decisions} with both verdicts)")
        reference = state_hash(self.verifier)
        losses: list[float] = []
        deltas: dict[str, torch.Tensor] = {}
        if trained_names:
            delta = ParameterDelta(self.verifier, trained_names, rank=self.cfg.adapter.rank, alpha=self.cfg.adapter.alpha)
            if train_trust:
                losses += self._train_trust(delta, dec, stats)
            if temps:
                losses += self._train_calibration(delta, fam, temps, stats)
            deltas = delta.frozen()
        if state_hash(self.verifier) != reference:
            raise InvariantViolation("the reference Verifier weights changed during an RLCD fit")
        proposal = (CalibrationProposal(temperatures=temps, source="rlcd", n_pairs=n_pairs, t_min=vc.temperature_min,
                                        t_max=vc.temperature_max) if temps else None)
        used = sorted({i for f in OUTPUT_FAMILIES for i in fam[f].index} | set(dec.index))
        report = FitReport(method=self.method, target=self.target, used=tuple(used), excluded=exclusions.as_tuple(),
                           losses=tuple(losses), stats=stats, config_digest=digest_of(to_mapping(self.cfg)),
                           ledger_head=batch.ledger_head, seed=self.cfg.seed)
        return CandidateUpdate(method=self.method, target=self.target, reference_hash=reference, deltas=deltas,
                               temperatures=proposal, reward_model=None, report=report)

    def _train_trust(self, delta: ParameterDelta, dec: Decisions, stats: dict[str, float]) -> list[float]:
        # Weighted Brier of the trust confidence against decision correctness (RLCR calibration term).
        oc = self.cfg.rlcd.trust_optimiser
        params = delta.parameters_of(self.trust_names)
        opt = make_optimiser(params, oc)
        x = torch.stack(dec.features)                                             # [n, 18]
        c = torch.tensor(dec.correct, dtype=torch.float64)
        w = torch.tensor(dec.w, dtype=torch.float64)
        ff, mf = x[:, :N_FORECAST_FEATURES], x[:, N_FORECAST_FEATURES:N_FORECAST_FEATURES + N_MONITOR_FEATURES]
        gen = generator_for(self.cfg.seed, "rlcd", "trust")
        with torch.no_grad():
            stats["trust_brier_before"] = float(rlcr_trust_loss(self.verifier.trust(ff, mf), c, w))
        losses: list[float] = []
        order: list[list[int]] = []
        for _ in range(oc.steps):
            if not order:
                order = minibatches(len(dec), oc.batch_size, gen)
            mb = order.pop(0)
            logit_q = call_with(self.verifier, delta.overrides(self.verifier), lambda v, a, b: v.trust(a, b), ff[mb], mf[mb])
            loss = rlcr_trust_loss(logit_q, c[mb], w[mb])
            optimiser_step(opt, params, loss, oc)
            losses.append(float(loss.detach()))
        with torch.no_grad():
            after = call_with(self.verifier, delta.overrides(self.verifier), lambda v, a, b: v.trust(a, b), ff, mf)
            stats["trust_brier_after"] = float(rlcr_trust_loss(after, c, w))
        return losses

    def _train_calibration(self, delta: ParameterDelta, fam: dict[str, FamilyPairs], temps: dict[str, float],
                           stats: dict[str, float]) -> list[float]:
        # The policy head learns to propose log T* from each family's reliability features (D-45).
        oc = self.cfg.rlcd.calibration_optimiser
        params = delta.parameters_of(self.cal_names)
        opt = make_optimiser(params, oc)
        fams = sorted(temps)
        feats = torch.stack([reliability_features(*fam[f].tensors()[:2], bins=self.vcfg.reliability_bins) for f in fams])
        codes = torch.tensor([OUTPUT_FAMILIES.index(f) for f in fams])
        target = torch.tensor([temps[f] for f in fams], dtype=torch.float64)
        gen = generator_for(self.cfg.seed, "rlcd", "calibration")
        losses: list[float] = []
        order: list[list[int]] = []
        for _ in range(oc.steps):
            if not order:
                order = minibatches(len(fams), oc.batch_size, gen)
            mb = order.pop(0)
            log_t = call_with(self.verifier, delta.overrides(self.verifier), lambda v, a, b: v.calibration(a, b), feats[mb], codes[mb])
            loss = ((log_t.double() - torch.log(target[mb])) ** 2).mean()
            optimiser_step(opt, params, loss, oc)
            losses.append(float(loss.detach()))
        with torch.no_grad():
            log_t = call_with(self.verifier, delta.overrides(self.verifier), lambda v, a, b: v.calibration(a, b), feats, codes)
            stats["calibration_head_log_error"] = float((log_t.double() - torch.log(target)).abs().max())
        return losses


__all__ = ["Decisions", "FamilyPairs", "RLCDLearner", "brier_reward", "rewards_from_pairs", "rlcd_objective",
           "rlcd_temperature", "rlcr_trust_loss"]
