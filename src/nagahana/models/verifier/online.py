"""Feedback objectives on the live weights, for full training under the run's HumanCommand (D-21, D-65; AS-847).

In full training (Stage 3) the whole model trains on dataset annotations, which count as human-supplied
truth (AS-25, D-17). The Verifier's feedback learning joins it as additional loss terms whose optimiser
steps run only through `gate.train_on_feedback` under the run's "update-weights" command:

    RLVR   GRPO of the Forecaster's policy with the confirmed outcomes of the window's triggers as
           verifiable rewards (rlvr.py); one step per batch, so pi_old is the live policy at the step
           (every ratio is 1 and the clip is inactive; the objective is the group-relative policy
           gradient with the KL penalty)
    RLHF   DPO of the Forecaster's or Advisor's policy on analyst preferences, when a ledger is supplied
    RLCD   the trust head's Brier loss (rlcd.rlcr_trust_loss) and the RLCD temperature as the
           calibration head's target (rlcd.rlcd_temperature)

The reference policy pi_ref is a snapshot of the policy's parameters taken when the objectives are
created (the start of the stage): `ReferenceSnapshot` keeps detached copies and evaluates the module at
them through `params.call_with`, so the KL penalty and DPO's implicit reward are anchored to the stage's
starting policy while the live weights move. Precision: losses are float64 (D-54).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import nn

from nagahana.core.errors import InvariantViolation
from nagahana.models.advisor.model import Advisor
from nagahana.models.forecaster.model import Forecaster
from nagahana.models.verifier.config import FeedbackLearningConfig
from nagahana.models.verifier.feedback import OutcomeConfirmation, PreferenceFeedback
from nagahana.models.verifier.learning import Exclusions, FeedbackBatch
from nagahana.models.verifier.params import state_hash
from nagahana.models.verifier.policies import ForecasterPolicy
from nagahana.models.verifier.rlhf import RLHFLearner, dpo_loss
from nagahana.models.verifier.rlvr import draw_group, group_advantages, grpo_loss, usable_outcomes
from nagahana.models.verifier.situations import Situation


class ReferenceSnapshot:
    """Detached copies of a module's parameters: the reference policy of in-loop objectives."""

    def __init__(self, module: nn.Module) -> None:
        self.hash = state_hash(module)
        self.tensors = {n: p.detach().clone() for n, p in module.named_parameters()}

    def overrides(self) -> Mapping[str, torch.Tensor]:
        """The parameter view of the snapshot (params.call_with)."""
        return self.tensors


class OnlineFeedback:
    """RLVR and RLHF loss terms on the live Forecaster and Advisor (module docstring).

    Parameters
    ----------
    cfg:
        Feedback-learning configuration (grpo, dpo, outcome_reward, seed).
    forecaster, advisor:
        The live modules; their parameters at construction are the reference policies.
    """

    def __init__(self, cfg: FeedbackLearningConfig, *, forecaster: Forecaster, advisor: Advisor | None = None) -> None:
        self.cfg = cfg
        self.forecaster, self.advisor = forecaster, advisor
        self.policy = ForecasterPolicy(forecaster)
        self.ref_forecaster = ReferenceSnapshot(forecaster)
        self.ref_advisor = None if advisor is None else ReferenceSnapshot(advisor)

    def rlvr_loss(self, situations: Sequence[Situation], outcomes: Sequence[OutcomeConfirmation], *, step: int
                  ) -> tuple[torch.Tensor, dict[str, float]]:
        """GRPO loss of one batch of triggers with confirmed outcomes (situation i pairs with outcome i).

        Outcomes refer to their situation by id (provenance.refers_to). Returns a zero loss with a note
        when no trigger yields a live group (all routes scored equally), so the caller's step is a no-op.
        """
        if len(situations) != len(outcomes):
            raise InvariantViolation("rlvr_loss pairs each situation with one outcome")
        batch = FeedbackBatch(tuple(enumerate(outcomes)), {s.situation_id: s for s in situations})
        exclusions = Exclusions()
        triggers = usable_outcomes(batch, self.cfg.outcome_reward, exclusions)
        groups = [g for g in (draw_group(self.policy, t, cfg=self.cfg, iteration=step, old_overrides=None,
                                         ref_overrides=self.ref_forecaster.overrides()) for t in triggers) if g is not None]
        zero = torch.zeros((), dtype=torch.float64, requires_grad=False)
        stats = {"rlvr_triggers": float(len(triggers)), "rlvr_excluded": float(len(exclusions.items)), "rlvr_live_groups": 0.0}
        if not groups:
            return zero, stats
        adv, live = group_advantages(torch.stack([g.rewards for g in groups]), eps=self.cfg.grpo.advantage_eps)
        stats["rlvr_live_groups"] = float(live.sum())
        total: torch.Tensor | None = None
        wsum = 0.0
        kl = []
        for g, a, lv in zip(groups, adv, live.tolist(), strict=True):
            if not lv:
                continue
            new = self.policy.step_log_probs(g.situation.analysis, g.tech, g.tgt, overrides=None)
            loss_g, st = grpo_loss(new, g.old, g.ref, a.to(new.logp.device), self.cfg.grpo)
            total = g.weight * loss_g if total is None else total + g.weight * loss_g
            wsum += g.weight
            kl.append(float(st["kl"]))
            stats["rlvr_reward_mean"] = stats.get("rlvr_reward_mean", 0.0) + float(g.rewards.mean()) / len(groups)
        if total is None:
            return zero, stats
        stats["rlvr_kl"] = sum(kl) / len(kl)
        return total / wsum, stats

    def dpo_loss(self, preferences: Sequence[PreferenceFeedback], situations: Mapping[str, Situation], *, target: str
                 ) -> tuple[torch.Tensor, dict[str, float]]:
        """DPO loss of analyst preferences for the live Forecaster ("forecaster") or Advisor ("advisor")."""
        module: nn.Module | None = self.forecaster if target == "forecaster" else self.advisor
        snap = self.ref_forecaster if target == "forecaster" else self.ref_advisor
        if module is None or snap is None:
            raise InvariantViolation(f"no live {target} was given to the online objectives")
        learner = RLHFLearner(self.cfg, target=target, module=module)
        exclusions = Exclusions()
        batch = FeedbackBatch(tuple(enumerate(preferences)), dict(situations))
        examples = learner.examples(batch, exclusions, reference=snap.overrides())
        stats = {"dpo_preferences": float(len(examples)), "dpo_excluded": float(len(exclusions.items))}
        if not examples:
            return torch.zeros((), dtype=torch.float64), stats
        terms = [learner.item_logps(ex, None) for ex in examples]
        lp1 = torch.stack([t[0] for t in terms])
        lp2 = torch.stack([t[1] for t in terms])
        ref1 = torch.tensor([ex.ref_first for ex in examples], dtype=torch.float64, device=lp1.device)
        ref2 = torch.tensor([ex.ref_second for ex in examples], dtype=torch.float64, device=lp1.device)
        lab = torch.tensor([ex.event.label for ex in examples], dtype=torch.float64, device=lp1.device)
        loss, h = dpo_loss(lp1, lp2, ref1, ref2, lab, beta=self.cfg.dpo.beta)
        stats["dpo_margin_mean"] = float(h.detach().mean())
        return loss.mean(), stats


__all__ = ["OnlineFeedback", "ReferenceSnapshot"]
