"""RLVR: verifiable rewards from confirmed outcomes, optimised by group-relative policy optimisation (D-65).

After a forecast's horizon has elapsed, the confirmed outcome (forensics or the analyst; feedback.
OutcomeConfirmation) scores every route the Forecaster can imagine for that trigger with a proper score
of the route's own forecast (rewards.route_rewards: the right-censored survival log score or the Brier
score of P_inf, plus stage correctness where stages are confirmed). No learned reward model is involved.

GRPO (Shao et al., "DeepSeekMath", arXiv:2402.03300, section 4.1), with routes in place of sampled
answers and imagined steps in place of output positions. For each trigger x in an iteration, a group of
G routes o_1 ... o_G is drawn from pi_old (exact ancestral sampling, policies.ForecasterPolicy.sample,
AS-836), scored r_1 ... r_G, and every step of route i gets the group-relative advantage

    A_i = (r_i - mean_j r_j) / std_j r_j          (population standard deviation, AS-838)

A group whose rewards are all equal (std below `advantage_eps`) carries no relative information and gets
A = 0. With per-step ratios rho_(i,k) = pi_theta(a_(i,k) | x, a_(i,<k)) / pi_old(a_(i,k) | x, a_(i,<k)) the
objective maximised is

    J(theta) = sum_x omega_x / sum_x omega_x * (1/G) sum_i (1/K_i) sum_k [ min(rho_(i,k) A_i, clip(rho_(i,k), 1 - eps, 1 + eps) A_i)
                                                                          - beta_KL D_(i,k) ]

the clipped surrogate of PPO (Schulman et al., arXiv:1707.06347) with the KL penalty to the reference
policy pi_ref (the deployed weights): D is the exact per-step KL on the prefixes visited by pi_old
(policies.analytic_step_kl, AS-837) or Shao et al.'s sampled k3 estimator. omega_x is the trigger's
weight: 1, the inverse response propensity of a scored outcome (AS-840) and, for the Brier reward under
"ipcw", the inverse probability of censoring (rewards.ipcw_weights).

Why the penalty bounds the change. With the analytic KL, a stationary point of J at rho = 1 satisfies,
per state, A(a) - beta_KL log(pi_theta(a)/pi_ref(a)) = constant, i.e. pi_theta is proportional to
pi_ref exp(A / beta_KL), the KL-regularised optimum (the same form as DPO's pi*), and then
KL(pi_theta || pi_ref) <= (max_a A(a) - min_a A(a)) / beta_KL by Jensen's inequality
(log Z >= E_ref[A] / beta_KL). The clip additionally bounds each iteration's step: an update stops
earning advantage once rho leaves [1 - eps, 1 + eps].

Iterations: pi_old is the candidate at the start of each outer iteration (theta_ref + Delta, frozen), so
the first groups are drawn from the deployed policy; within an iteration the groups are reused for
`inner_epochs` passes, where the ratios and the clip matter.

Invariants (tested): advantages have mean 0 and unit population variance within every live group and
are 0 in a degenerate group; on a bandit with exact expectations the fixed point is pi_ref exp(A/beta)/Z
and its KL obeys the bound above for several beta; the reference weights are unchanged by a fit.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch

from nagahana.core.config import to_mapping
from nagahana.core.errors import InvariantViolation
from nagahana.models.forecaster.model import Forecaster
from nagahana.models.verifier.candidates import CandidateUpdate, FitReport
from nagahana.models.verifier.canonical import digest_of
from nagahana.models.verifier.config import FeedbackLearningConfig, GRPOConfig, OutcomeRewardConfig
from nagahana.models.verifier.feedback import OutcomeConfirmation, Provenance
from nagahana.models.verifier.learning import (
    Exclusions,
    FeedbackBatch,
    generator_for,
    make_optimiser,
    minibatches,
    optimiser_step,
    resolve_situation,
)
from nagahana.models.verifier.params import ParameterDelta, state_hash
from nagahana.models.verifier.policies import ForecasterPolicy, Overrides, StepLogProbs, analytic_step_kl, k3_kl
from nagahana.models.verifier.rewards import ipcw_weights, response_weights, route_rewards
from nagahana.models.verifier.situations import Situation
from nagahana.models.vocab import STAGES


def group_advantages(rewards: torch.Tensor, *, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    """(A [X, G], live [X]) from rewards [X, G]: (r - mean) / population std per group; 0 for degenerate groups."""
    r = rewards.double()
    if r.dim() != 2 or r.shape[1] < 2:
        raise InvariantViolation("rewards must be [groups, G] with G >= 2")
    mu = r.mean(-1, keepdim=True)
    sd = ((r - mu) ** 2).mean(-1, keepdim=True).sqrt()
    live = sd.squeeze(-1) > eps
    adv = torch.where(live[:, None], (r - mu) / torch.where(live[:, None], sd, torch.ones_like(sd)), torch.zeros_like(r))
    return adv, live


def clipped_surrogate(logp_new: torch.Tensor, logp_old: torch.Tensor, advantage: torch.Tensor, mask: torch.Tensor,
                      clip_eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    """(per-route mean over steps of min(rho A, clip(rho) A) [R], clipped-step indicator [R, K])."""
    ratio = torch.exp(logp_new.double() - logp_old.double())
    a = advantage.double()[:, None]
    s1 = ratio * a
    s2 = ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps) * a
    per = torch.minimum(s1, s2)
    m = mask.double()
    clipped = (s2 < s1) & mask
    return (per * m).sum(-1) / m.sum(-1).clamp_min(1.0), clipped


def grpo_loss(new: StepLogProbs, old: StepLogProbs, ref: StepLogProbs, advantage: torch.Tensor, cfg: GRPOConfig
              ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Negative GRPO objective of one group [R routes] (module docstring) and its diagnostics (detached).

    The step mask is the routes' own mask; old and ref must be evaluated on the same routes.
    """
    mask = new.mask & old.mask & ref.mask
    surr, clipped = clipped_surrogate(new.logp, old.logp, advantage, mask, cfg.clip_eps)
    # Analytic per-step KL on the visited prefixes (AS-837), or the sampled k3 estimator of Shao et al.
    kl_steps = analytic_step_kl(new, ref, old.first_logp) if cfg.kl_estimator == "analytic" else k3_kl(new, ref)
    m = mask.double()
    kl_route = (kl_steps * m).sum(-1) / m.sum(-1).clamp_min(1.0)
    per_route = -(surr - cfg.kl_beta * kl_route)
    ratio = torch.exp(new.logp.double() - old.logp.double())
    stats = {"kl": kl_route.mean().detach(), "clip_fraction": (clipped.double().sum() / m.sum().clamp_min(1.0)).detach(),
             "ratio_mean": ((ratio * m).sum() / m.sum().clamp_min(1.0)).detach(), "surrogate": surr.mean().detach()}
    return per_route.mean(), stats


@dataclass
class _Group:
    """One trigger's group of routes for an iteration."""

    index: int
    situation: Situation
    tech: torch.Tensor
    tgt: torch.Tensor
    old: StepLogProbs
    ref: StepLogProbs
    rewards: torch.Tensor
    weight: float
    advantage: torch.Tensor | None = None


@dataclass
class _Trigger:
    """A usable confirmed outcome with its situation and objective weight."""

    index: int
    outcome: OutcomeConfirmation
    situation: Situation
    weight: float


def usable_outcomes(batch: FeedbackBatch, cfg: OutcomeRewardConfig, exclusions: Exclusions) -> list[_Trigger]:
    """Confirmed outcomes the verifiable reward can score, with their weights (AS-839, AS-840)."""
    events = [(i, e) for i, e in batch.of_kind("outcome") if isinstance(e, OutcomeConfirmation)]
    outs = [e for _, e in events]
    w_resp, keep, reasons = response_weights(outs, mode=cfg.responded_to, max_weight=cfg.max_ipw_weight)
    # Inverse-probability-of-censoring weights for the Brier reward, estimated over every scored outcome of the
    # batch (censored ones included), so the Kaplan-Meier estimate of G sees every censoring time.
    censor_w: dict[int, float] = {}
    if cfg.p_inf_score == "brier" and cfg.brier_censoring == "ipcw":
        scored = [(i, e) for (i, e), k in zip(events, keep, strict=True) if k]
        if scored:
            g = ipcw_weights([e.event_step for _, e in scored], [e.observed_steps for _, e in scored],
                             [e.horizon_k for _, e in scored], max_weight=cfg.max_ipw_weight)
            censor_w = {i: float(w) for (i, _), w in zip(scored, g, strict=True)}
    out: list[_Trigger] = []
    for (idx, ev), w, k, why in zip(events, w_resp, keep, reasons, strict=True):
        if not k:
            exclusions.add(idx, ev, why)
            continue
        if cfg.p_inf_score == "log" and ev.event_step == 0 and ev.observed_steps == 0:
            exclusions.add(idx, ev, "no observed step: the outcome carries no information")
            continue
        if cfg.p_inf_score == "brier" and not ev.known_at_horizon:
            exclusions.add(idx, ev, "outcome unknown at the horizon (censored before K)")
            continue
        s, why_s = resolve_situation(batch, ev)
        if s is None:
            exclusions.add(idx, ev, why_s)
            continue
        weight = float(w) * censor_w.get(idx, 1.0)
        if not weight > 0:
            exclusions.add(idx, ev, "zero inverse-probability weight (censoring distribution exhausted)")
            continue
        out.append(_Trigger(idx, ev, s, weight))
    return out


def draw_group(policy: ForecasterPolicy, trig: _Trigger, *, cfg: FeedbackLearningConfig, iteration: int,
               old_overrides: Overrides, ref_overrides: Overrides) -> _Group | None:
    """One group: G routes drawn from pi_old, their pi_old and pi_ref terms, and their verifiable rewards.

    old_overrides / ref_overrides are parameter views (params.call_with): None is the module's own weights.
    Returns None when the outcome cannot score the routes (rewards.route_rewards).
    """
    an1 = trig.situation.analysis
    gen = generator_for(cfg.seed, "rlvr", "group", trig.situation.situation_id, trig.index, iteration)
    tech, tgt = policy.sample(an1, cfg.grpo.group_size, trig.outcome.horizon_k, gen, overrides=old_overrides)
    with torch.no_grad():
        old = policy.step_log_probs(an1, tech, tgt, overrides=old_overrides).detached()
        ref = policy.step_log_probs(an1, tech, tgt, overrides=ref_overrides).detached()
    assert old.hazard is not None
    rewards, _why = route_rewards(old.hazard, old.stage_logits, trig.outcome, cfg.outcome_reward)
    if rewards is None:
        return None
    return _Group(trig.index, trig.situation, tech, tgt, old, ref, rewards, trig.weight)


class RLVRLearner:
    """GRPO candidate of the Forecaster's policy from confirmed outcomes (module docstring)."""

    method = "rlvr"
    target = "forecaster"

    def __init__(self, cfg: FeedbackLearningConfig, *, forecaster: Forecaster) -> None:
        self.cfg, self.module = cfg, forecaster
        self.policy = ForecasterPolicy(forecaster)

    def fit(self, batch: FeedbackBatch) -> CandidateUpdate:
        """GRPO over the batch's usable confirmed outcomes (module docstring)."""
        exclusions = Exclusions()
        triggers = usable_outcomes(batch, self.cfg.outcome_reward, exclusions)
        if not triggers:
            raise InvariantViolation("no usable confirmed outcome for RLVR: "
                                     + ("; ".join(sorted({e.reason for e in exclusions.items})) or "none in the batch"))
        g = self.cfg.grpo
        delta = ParameterDelta(self.module, self.cfg.adapter.forecaster_targets, rank=self.cfg.adapter.rank,
                               alpha=self.cfg.adapter.alpha)
        params = list(delta.parameters())
        opt = make_optimiser(params, g.optimiser)
        pick = generator_for(self.cfg.seed, "rlvr", "triggers", batch.ledger_head)
        losses: list[float] = []
        live_groups = dead_groups = 0
        diag: dict[str, list[float]] = {"kl": [], "clip_fraction": [], "ratio_mean": [], "reward": []}
        for it in range(g.iterations):
            chosen = torch.randperm(len(triggers), generator=pick)[: g.triggers_per_iteration].tolist()
            old_overrides = {n: t.detach() for n, t in delta.overrides(self.module).items()}
            drawn = (draw_group(self.policy, triggers[i], cfg=self.cfg, iteration=it, old_overrides=old_overrides, ref_overrides=None)
                     for i in chosen)
            groups = [grp for grp in drawn if grp is not None]
            if not groups:
                continue
            adv, live = group_advantages(torch.stack([grp.rewards for grp in groups]), eps=g.advantage_eps)
            live_groups += int(live.sum())
            dead_groups += int((~live).sum())
            for grp, a in zip(groups, adv, strict=True):
                grp.advantage = a
                diag["reward"].append(float(grp.rewards.mean()))
            active = [grp for grp, lv in zip(groups, live.tolist(), strict=True) if lv]
            if not active:
                continue
            mb_gen = generator_for(self.cfg.seed, "rlvr", "minibatches", it)
            for _epoch in range(g.inner_epochs):
                for mb in minibatches(len(active), g.optimiser.batch_size, mb_gen):
                    overrides = delta.overrides(self.module)
                    total = torch.zeros((), dtype=torch.float64)
                    wsum = 0.0
                    for j in mb:
                        grp = active[j]
                        assert grp.advantage is not None
                        new = self.policy.step_log_probs(grp.situation.analysis, grp.tech, grp.tgt, overrides=overrides)
                        loss_g, st = grpo_loss(new, grp.old, grp.ref, grp.advantage.to(new.logp.device), g)
                        total = total + grp.weight * loss_g.to(total.device)
                        wsum += grp.weight
                        for key in ("kl", "clip_fraction", "ratio_mean"):
                            diag[key].append(float(st[key]))
                    loss = total / wsum
                    optimiser_step(opt, params, loss, g.optimiser)
                    losses.append(float(loss.detach()))
        if state_hash(self.module) != delta.reference_hash:
            raise InvariantViolation("the reference weights changed during an RLVR fit")
        if not losses:
            raise InvariantViolation("RLVR made no update: every group had equal rewards for all its routes "
                                     f"({dead_groups} degenerate groups); the outcomes do not separate the routes")
        stats = {"live_groups": float(live_groups), "degenerate_groups": float(dead_groups),
                 "n_outcomes": float(len(triggers)), "delta_norm": float(delta.sq_norm()) ** 0.5,
                 **{f"mean_{k}": float(np.mean(v)) if v else float("nan") for k, v in diag.items()},
                 **{f"last_{k}": v[-1] for k, v in diag.items() if v}}
        report = FitReport(method=self.method, target=self.target, used=tuple(t.index for t in triggers),
                           excluded=exclusions.as_tuple(), losses=tuple(losses), stats=stats,
                           config_digest=digest_of(to_mapping(self.cfg)), ledger_head=batch.ledger_head, seed=self.cfg.seed)
        return CandidateUpdate(method=self.method, target=self.target, reference_hash=delta.reference_hash,
                               deltas=delta.frozen(), temperatures=None, reward_model=None, report=report)


def outcome_from_survival(*, refers_to: str, event_step: int, censored: bool, horizon_k: int, stage_labels: Sequence[int],
                          analyst: str, time: float, command_id: str = "") -> OutcomeConfirmation | None:
    """An OutcomeConfirmation from the Forecaster's survival targets (forecaster.losses.survival_targets).

    event_step / censored follow `hazard_nll`: for an event the 1-based step, for a censored trigger the
    number of event-free observed steps. stage_labels [K + 1] are `step_labels`' stages (index 0 is the
    step before the trigger, -1 unknown); steps 1 ... K with a label become confirmed stages. A dataset's
    annotations count as human-supplied truth (AS-25, D-17). For an event the observed steps are set to the
    event step (the least the label horizon must cover); the log score and the outcome at the horizon do
    not depend on observation after an event.
    """
    e = 0 if censored else int(event_step)
    obs = int(event_step)
    if e > horizon_k or obs > horizon_k or obs < 0:
        return None
    stages = tuple((k, STAGES[int(code)][0]) for k, code in enumerate(stage_labels) if 1 <= k <= horizon_k and int(code) >= 0)
    return OutcomeConfirmation(provenance=Provenance(analyst=analyst, time=float(time), refers_to=refers_to, command_id=command_id),
                               horizon_k=int(horizon_k), event_step=e, observed_steps=obs, source="forensics", stages=stages)


__all__ = ["RLVRLearner", "clipped_surrogate", "draw_group", "group_advantages", "grpo_loss", "outcome_from_survival",
           "usable_outcomes"]
