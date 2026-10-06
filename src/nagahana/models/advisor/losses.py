"""Advisor training: model-based policy improvement on −ΔP_inf − κ·cost under re-imagination (build-spec §3, stage 5).

Purpose
-------
The Advisor learns π_D and V_D from the Forecaster's model of the adversary, never from acting on a
network (D-33). For a trigger:

1. propose candidate single counters from π_D (top actions; a target drawn from the pointer);
2. re-imagine each with the Forecaster (no gradient; the Forecaster is not changed, [Q-38]) and score
   J_c = −(CVaR_α(ΔP_inf) + κ·cost) (`objectives.rewards.advisor_objective`, AS-23/AS-24);
3. form the improved target over the candidate set (an MPO-style E-step; Abdolmaleki et al.,
   "Maximum a Posteriori Policy Optimisation", ICLR 2018, arXiv:1806.06920):

       q_c ∝ sg π̂_D(c) · exp(J_c / η)       (rule-infeasible candidates: q_c = 0)

   and fit π_D to it (M-step):  L_π = −Σ_c q_c log π̂_D(c),  with π̂_D the policy renormalised over
   the candidate set;
4. regress the plan value: L_V = ½ Σ_c (V_D([start; c]) − J_c)².

Assumptions: AS-23, AS-24, AS-257 (η = `cfg.improvement_temperature`). Because effects are structural
(`effects.py`), the policy cannot improve its score by changing how counters act, only by choosing.

Invariants (tested): gradients reach every trainable Advisor parameter; the target q is a
distribution over feasible candidates.
"""

from __future__ import annotations

import torch

from nagahana.models.advisor.model import Advisor, slice_trigger
from nagahana.models.batch import AnalysisOut
from nagahana.models.forecaster.model import Forecaster, MarginalEnergy
from nagahana.models.forecaster.routes import gumbel_argmax
from nagahana.objectives.rewards import advisor_objective


def policy_improvement_loss(advisor: Advisor, analysis: AnalysisOut, forecaster: Forecaster, *, b: int, m: int,
                            rollouts: int, generator: torch.Generator | None = None, horizon_k: int | None = None,
                            exposure: MarginalEnergy | None = None, entity_kind: torch.Tensor | None = None,
                            entity_internal: torch.Tensor | None = None,
                            technique_plane: torch.Tensor | None = None) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """L = L_π + L_V for trigger (b, m) (module docstring). Returns (total, parts incl. the target q and J)."""
    cfg = advisor.cfg
    an1 = slice_trigger(analysis, b, m)
    k = forecaster.cfg.horizon_k if horizon_k is None else horizon_k
    seed = int(torch.randint(0, 2**62, (1,), generator=generator))
    base_f = advisor.baseline(an1, forecaster, horizon_k=k, rollouts=rollouts, seed=seed, exposure=exposure)
    mem = advisor.memory(an1)

    # 1. candidates from π_D at the empty plan: top actions, each with a target drawn from the pointer.
    h0 = advisor.plan_forward(mem, [()])[:, -1]                                     # [1, dim]
    log_a = advisor.action_log_probs(h0)[0]                                         # [A]
    n = min(cfg.train_candidates, int(advisor.assigned.sum()))
    _, top_a = torch.topk(log_a.detach(), n)                                        # [n]
    log_v = advisor.target_log_probs(mem, h0, top_a[None])[0]                       # [n, V]
    u = torch.rand(log_v.shape, generator=generator).to(log_v.device)
    tgt = gumbel_argmax(log_v.detach(), u)                                          # [n]
    log_pi = log_a[top_a] + log_v.gather(-1, tgt[:, None]).squeeze(-1)              # log π̂_D(c), with gradient
    plans = [((int(a), int(v)),) for a, v in zip(top_a, tgt, strict=True)]

    # 2. re-imagine and score each candidate (no gradient through the Forecaster).
    with torch.no_grad():
        evs = [advisor.evaluate(an1, forecaster, p, mem=mem, baseline_f=base_f, horizon_k=k, rollouts=rollouts, seed=seed,
                                exposure=exposure, entity_kind=entity_kind, entity_internal=entity_internal,
                                technique_plane=technique_plane) for p in plans]
    j = torch.stack([advisor_objective(torch.tensor(e.delta_cvar), torch.tensor(e.cost), kappa=cfg.cost_weight)
                     for e in evs]).float().to(log_pi.device)                       # [n]
    feasible = torch.tensor([e.feasible for e in evs], device=log_pi.device)

    # 3. improved target q ∝ sg π̂ · exp(J/η) over feasible candidates; M-step cross-entropy.
    log_pi_hat = torch.log_softmax(log_pi, dim=0)
    if bool(feasible.any()):
        logits_q = (log_pi_hat.detach() + j / cfg.improvement_temperature).masked_fill(~feasible, float("-inf"))
        q = torch.softmax(logits_q, dim=0)
        loss_pi = -(q * log_pi_hat.masked_fill(~feasible, 0.0)).sum()
    else:
        q = torch.zeros_like(log_pi_hat)
        loss_pi = log_pi_hat.sum() * 0.0

    # 4. plan value regression on every evaluated candidate.
    v_hat = advisor.plan_value(advisor.plan_forward(mem, plans)[:, -1])             # [n]
    loss_v = 0.5 * ((v_hat.float() - j) ** 2).mean()
    total = loss_pi + loss_v
    return total, {"policy": loss_pi.detach(), "value": loss_v.detach(), "target_q": q.detach(), "objective": j}
