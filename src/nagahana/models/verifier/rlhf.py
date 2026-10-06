"""RLHF: a Bradley-Terry reward model of analyst preferences and direct preference optimisation of the policies (D-65).

From preferences between two forecasts, two route sets or two advisories of the same situation, the
RLHF learner produces one candidate update of the policy those items came from (route sets: the
Forecaster's adversary policy pi_A; advisories: the Advisor's defender policy pi_D), plus the explicit
reward model of the analysts' preferences (bradley_terry.py, AS-833).

KL-regularised preference optimisation. RLHF maximises E_(y ~ pi)[r(x, y)] - beta KL(pi(. | x) || pi_ref(. | x))
(Christiano et al., NeurIPS 2017, arXiv:1706.03741; Ouyang et al., NeurIPS 2022, arXiv:2203.02155), whose
maximiser is

    pi*(y | x) = pi_ref(y | x) exp(r(x, y) / beta) / Z(x)

(Rafailov et al., "Direct Preference Optimization", NeurIPS 2023, arXiv:2305.18290, eq. 4). Solving for r
gives r = beta log(pi*/pi_ref) + beta log Z(x), and Z cancels in the Bradley-Terry difference, so the
preference likelihood can be maximised directly in the policy. With soft label s = P(first preferred)
(feedback.PreferenceFeedback.label) and the implicit reward margin

    h = beta [ (log pi_theta(y_1 | x) - log pi_ref(y_1 | x)) - (log pi_theta(y_2 | x) - log pi_ref(y_2 | x)) ]

the DPO loss per preference is the cross-entropy of sigma(h) against s:

    L = -s log sigma(h) - (1 - s) log sigma(-h) = softplus(h) - s h

    dL/dh = sigma(h) - s,   dL/d log pi_theta(y_1) = beta (sigma(h) - s),   dL/d log pi_theta(y_2) = -beta (sigma(h) - s)

For s = 1 this is Rafailov et al.'s loss -log sigma(h) and gradient -beta sigma(-h) [grad log pi(y_w) -
grad log pi(y_l)] (eq. 7); s in (1/2, 1) is the conservative form with label smoothing 1 - s; a tie (s = 1/2)
pulls the two implicit rewards together. The reference pi_ref is the deployed policy (the module's own
weights at fit time; params.ParameterDelta keeps them untouched) and the candidate is theta_ref + Delta.

Sequence probabilities. A route set is a multiset of routes drawn from the policy; with merge counts
c_r its log-probability is log(multinomial coefficient) + sum_r c_r log pi(r | x), and the coefficient
cancels in log pi_theta - log pi_ref, so

    log pi_theta(set) - log pi_ref(set) = sum_r c_r [ log pi_theta(r | x) - log pi_ref(r | x) ]

exactly (AS-834). An advisory's log-probability is the sum of its steps' log pi_D. Items the reference
policy cannot express (a target outside its context, a masked option) have no likelihood and are
excluded with the reason.

Optimisation: AdamW on Delta with decoupled decay toward the reference (config.OptimiserConfig), random
minibatches of preferences (learning.minibatches), global-norm clipping; float64 losses (D-54).

Invariants (tested): `dpo_loss` and its gradient equal the closed forms above and autograd; on a toy
categorical policy with exact Bradley-Terry labels the minimiser is pi_ref exp(r / beta) / Z; the
reference weights are bit-identical after a fit.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from nagahana.core.config import to_mapping
from nagahana.core.errors import InvariantViolation
from nagahana.models.advisor.model import Advisor
from nagahana.models.batch import AnalysisOut
from nagahana.models.forecaster.model import Forecaster
from nagahana.models.verifier.bradley_terry import BradleyTerryModel, fit_reward_model
from nagahana.models.verifier.candidates import CandidateUpdate, FitReport
from nagahana.models.verifier.canonical import digest_of
from nagahana.models.verifier.config import POLICY_TARGETS, FeedbackLearningConfig
from nagahana.models.verifier.feedback import PreferenceFeedback
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
from nagahana.models.verifier.policies import AdvisorPolicy, ForecasterPolicy, Overrides, route_codes


def dpo_loss(logp_first: torch.Tensor, logp_second: torch.Tensor, ref_first: torch.Tensor, ref_second: torch.Tensor,
             label: torch.Tensor, *, beta: float) -> tuple[torch.Tensor, torch.Tensor]:
    """(loss [n], margin h [n]) of the module docstring, float64: L = softplus(h) - s h."""
    if beta <= 0:
        raise ValueError("beta must be > 0")
    h = beta * ((logp_first.double() - ref_first.double()) - (logp_second.double() - ref_second.double()))
    s = label.double()
    return torch.nn.functional.softplus(h) - s * h, h


def dpo_grad(logp_first: torch.Tensor, logp_second: torch.Tensor, ref_first: torch.Tensor, ref_second: torch.Tensor,
             label: torch.Tensor, *, beta: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Closed-form (dL/d log pi(y_1), dL/d log pi(y_2)) = (beta (sigma(h) - s), -beta (sigma(h) - s))."""
    _, h = dpo_loss(logp_first, logp_second, ref_first, ref_second, label, beta=beta)
    g = beta * (torch.sigmoid(h) - label.double())
    return g, -g


@dataclass
class _Example:
    """One usable preference: its situation, the items' action sequences, reference log-probabilities."""

    index: int
    event: PreferenceFeedback
    analysis: AnalysisOut
    tech: torch.Tensor | None = None             # [R1 + R2, K] (Forecaster)
    tgt: torch.Tensor | None = None
    counts: torch.Tensor | None = None           # [R1 + R2] merge counts
    n_first: int = 0
    plans: tuple[tuple[tuple[int, int], ...], ...] = ()      # (first, second) (Advisor)
    ref_first: float = 0.0
    ref_second: float = 0.0


class RLHFLearner:
    """Bradley-Terry reward model + DPO candidate for one policy (module docstring).

    Parameters
    ----------
    cfg:
        The feedback-learning configuration (bradley_terry, dpo, adapter, seed).
    target:
        "forecaster" (preferences between forecasts and route sets) or "advisor" (between advisories).
    module:
        The deployed Forecaster or Advisor: its weights are the reference policy and are never written.
    """

    method = "rlhf"

    def __init__(self, cfg: FeedbackLearningConfig, *, target: str, module: nn.Module) -> None:
        if target not in POLICY_TARGETS:
            raise InvariantViolation(f"RLHF trains a policy: target must be one of {list(POLICY_TARGETS)}")
        if target == "forecaster" and not isinstance(module, Forecaster):
            raise InvariantViolation("the forecaster target needs a Forecaster")
        if target == "advisor" and not isinstance(module, Advisor):
            raise InvariantViolation("the advisor target needs an Advisor")
        self.cfg, self.target, self.module = cfg, target, module
        self.policy: ForecasterPolicy | AdvisorPolicy = (ForecasterPolicy(module) if isinstance(module, Forecaster)
                                                          else AdvisorPolicy(module))  # type: ignore[arg-type]
        self.patterns = cfg.adapter.forecaster_targets if target == "forecaster" else cfg.adapter.advisor_targets

    # ---- examples
    def examples(self, batch: FeedbackBatch, exclusions: Exclusions, *, reference: Overrides = None) -> list[_Example]:
        """Usable preferences of this learner's policy, with reference log-probabilities (no gradient).

        reference: the parameter view of pi_ref (None: the module's own weights, the deployed policy).
        """
        out: list[_Example] = []
        for idx, ev in batch.of_kind("preference"):
            assert isinstance(ev, PreferenceFeedback)
            if ev.target != self.target:
                continue
            s, why = resolve_situation(batch, ev)
            if s is None:
                exclusions.add(idx, ev, why)
                continue
            ex = _Example(index=idx, event=ev, analysis=s.analysis)
            try:
                if isinstance(self.policy, ForecasterPolicy):
                    routes = (*ev.first.routes, *ev.second.routes)
                    tech, tgt = route_codes(routes)
                    if bool((tech >= self.policy.module.cfg.n_techniques).any()):
                        exclusions.add(idx, ev, "technique slot outside the Forecaster's action space")
                        continue
                    ex.tech, ex.tgt = tech, tgt
                    ex.counts = torch.tensor([float(r.count) for r in routes], dtype=torch.float64)
                    ex.n_first = len(ev.first.routes)
                else:
                    assert ev.first.plan is not None and ev.second.plan is not None
                    ex.plans = (tuple(ev.first.plan.steps), tuple(ev.second.plan.steps))
            except InvariantViolation as exc:
                exclusions.add(idx, ev, f"item cannot be evaluated: {exc}")
                continue
            with torch.no_grad():
                lp1, lp2 = self.item_logps(ex, reference)
            if not (torch.isfinite(lp1) and torch.isfinite(lp2)):
                exclusions.add(idx, ev, "an item has no likelihood under the reference policy (target outside its context "
                                        "or a masked option)")
                continue
            ex.ref_first, ex.ref_second = float(lp1), float(lp2)
            out.append(ex)
        return out

    def item_logps(self, ex: _Example, overrides: Overrides) -> tuple[torch.Tensor, torch.Tensor]:
        """(log pi(first | x), log pi(second | x)) under a parameter view (module docstring, Sequence probabilities)."""
        if isinstance(self.policy, ForecasterPolicy):
            assert ex.tech is not None and ex.tgt is not None and ex.counts is not None
            steps = self.policy.step_log_probs(ex.analysis, ex.tech, ex.tgt, overrides=overrides)
            seq = steps.sequence_logp()                                          # [R1 + R2]
            c = ex.counts.to(seq.device)
            n1 = ex.n_first
            return (c[:n1] * seq[:n1]).sum(), (c[n1:] * seq[n1:]).sum()
        steps = self.policy.step_log_probs(ex.analysis, list(ex.plans), overrides=overrides)
        seq = steps.sequence_logp()                                              # [2]
        return seq[0], seq[1]

    # ---- fit
    def fit(self, batch: FeedbackBatch) -> CandidateUpdate:
        """Reward model + DPO candidate from the batch's preferences (module docstring)."""
        exclusions = Exclusions()
        examples = self.examples(batch, exclusions)
        if not examples:
            raise InvariantViolation(f"no usable preference for the {self.target}: "
                                     + "; ".join(sorted({e.reason for e in exclusions.items})) if exclusions.items
                                     else f"no preference for the {self.target} in the batch")
        reward_model: BradleyTerryModel = fit_reward_model([ex.event for ex in examples], self.cfg.bradley_terry)
        dcfg = self.cfg.dpo
        delta = ParameterDelta(self.module, self.patterns, rank=self.cfg.adapter.rank, alpha=self.cfg.adapter.alpha)
        params = list(delta.parameters())
        opt = make_optimiser(params, dcfg.optimiser)
        gen = generator_for(self.cfg.seed, "rlhf", self.target, "minibatches", batch.ledger_head)
        losses: list[float] = []
        order: list[list[int]] = []
        for _step in range(dcfg.optimiser.steps):
            if not order:
                order = minibatches(len(examples), dcfg.optimiser.batch_size, gen)
            mb = [examples[i] for i in order.pop(0)]
            overrides = delta.overrides(self.module)
            terms = [self.item_logps(ex, overrides) for ex in mb]
            lp1 = torch.stack([t[0] for t in terms])
            lp2 = torch.stack([t[1] for t in terms])
            ref1 = torch.tensor([ex.ref_first for ex in mb], dtype=torch.float64)
            ref2 = torch.tensor([ex.ref_second for ex in mb], dtype=torch.float64)
            lab = torch.tensor([ex.event.label for ex in mb], dtype=torch.float64)
            loss, _h = dpo_loss(lp1, lp2, ref1.to(lp1.device), ref2.to(lp1.device), lab.to(lp1.device), beta=dcfg.beta)
            total = loss.mean()
            optimiser_step(opt, params, total, dcfg.optimiser)
            losses.append(float(total.detach()))
        stats = self._final_stats(examples, delta, reward_model)
        if state_hash(self.module) != delta.reference_hash:
            raise InvariantViolation("the reference weights changed during an RLHF fit")
        report = FitReport(method=self.method, target=self.target, used=tuple(ex.index for ex in examples),
                           excluded=exclusions.as_tuple(), losses=tuple(losses), stats=stats,
                           config_digest=digest_of(to_mapping(self.cfg)), ledger_head=batch.ledger_head, seed=self.cfg.seed)
        return CandidateUpdate(method=self.method, target=self.target, reference_hash=delta.reference_hash,
                               deltas=delta.frozen(), temperatures=None, reward_model=reward_model, report=report)

    @torch.no_grad()
    def _final_stats(self, examples: list[_Example], delta: ParameterDelta, rm: BradleyTerryModel) -> dict[str, float]:
        # Training-set summary of the candidate: DPO loss, implicit-reward accuracy and agreement with the reward model.
        overrides = delta.overrides(self.module)
        beta = self.cfg.dpo.beta
        losses, correct, agree, decided = [], 0, 0, 0
        for ex in examples:
            lp1, lp2 = self.item_logps(ex, overrides)
            loss, h = dpo_loss(lp1[None], lp2[None], torch.tensor([ex.ref_first]), torch.tensor([ex.ref_second]),
                               torch.tensor([ex.event.label]), beta=beta)
            losses.append(float(loss[0]))
            if ex.event.preferred != "tie":
                decided += 1
                want_first = ex.event.preferred == "first"
                correct += int((float(h[0]) > 0) == want_first)
                agree += int((float(h[0]) > 0) == (rm.prob_first(ex.event) > 0.5))
        n = len(examples)
        return {"train_dpo_loss": sum(losses) / n, "train_preference_accuracy": correct / decided if decided else float("nan"),
                "train_reward_model_agreement": agree / decided if decided else float("nan"),
                "reward_model_nll": rm.nll, "delta_norm": float(delta.sq_norm()) ** 0.5, "n_preferences": float(n),
                "beta": beta}


__all__ = ["RLHFLearner", "dpo_grad", "dpo_loss"]
