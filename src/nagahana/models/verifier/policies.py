"""Policy views for feedback learning: exact per-step log-probabilities, ancestral sampling and KL (AS-836, AS-837).

Two policies learn from feedback, each a product of two factors per step:

    Forecaster (adversary policy pi_A):  pi(a_k | s_(k-1)) = pi(tech_k | s_(k-1)) * pi(tgt_k | s_(k-1), tech_k)
                                         tech over n_techniques slots, tgt a pointer over the C context entities
                                         plus "no target" (build-spec section 2.8)
    Advisor (defender policy pi_D):      pi(c_j | h_(j-1)) = pi(act_j | h_(j-1)) * pi(v_j | h_(j-1), act_j)
                                         act over the D3FEND slots (unassigned slots masked), v a pointer over
                                         the active entities (build-spec section 2.9)

A route (or plan) o = (a_1 ... a_K) of a situation x has log pi(o | x) = sum_k log pi(a_k | x, a_(<k)).
`StepLogProbs` holds, for R routes of one situation under one parameter view, the log-probability of
every taken step, the full log-distribution of the first factor and the second factor's
log-distribution given the taken first factor. The Forecaster's terms come from one teacher-forced pass
(`Forecaster.teacher_forced`, which is exactly imagination's computation with given actions, AS-250); the
Advisor's from `Advisor.plan_forward` and its two heads. Every log-probability is float64 (D-54).

Parameter views. Each method takes `overrides` (params.call_with): None evaluates the module's own
weights (the reference policy pi_ref), a mapping theta_ref + Delta evaluates a candidate.

Exact ancestral sampling (AS-836). GRPO needs routes drawn from pi_old with full support. Step k is drawn
from the teacher-forced pass over the already drawn prefix (later steps marked unknown; the causal mask
makes s_k depend on a_(<=k) only): tech_k by Gumbel-max over all n_techniques slots, then tgt_k by
Gumbel-max over the pointer distribution given tech_k. That is K passes for K steps, each an exact draw
from pi(. | s_k); the deployed imagination's MPPI tilt and top-B truncation (AS-21) are not used, because
the clipped-ratio trust region is defined relative to the sampling policy.

KL to the reference (AS-837). The per-state KL of a factorised step distribution is

    KL(pi_theta(. | s) || pi_ref(. | s)) = KL_1(s) + E_(f ~ pi_theta(. | s)) [ KL_2(s, f) ]

with KL_1 the KL of the first factors (computed exactly over all options) and KL_2 the KL of the second
factors given first factor f. Its expectation is estimated without bias from the first factor f_k taken
by pi_old:

    D_k = KL_1(s_k) + w_k KL_2(s_k, f_k),   w_k = pi_theta(f_k | s_k) / pi_old(f_k | s_k)

and the mean of D_k over the steps of routes drawn from pi_old is the average KL on the states pi_old
visits, the trust-region measure of TRPO (Schulman et al., ICML 2015, arXiv:1502.05477). Its gradient
is exact for that surrogate. `k3_kl` is the sampled estimator of Shao et al. (arXiv:2402.03300, eq. 4),
pi_ref/pi_theta - log(pi_ref/pi_theta) - 1 at the taken action.

Invariants (tested): log-probabilities of sampled routes are finite; the route log-probability equals
the sum of its factors; the KL of a view with itself is zero; sampling is deterministic under a seed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch

from nagahana.core.errors import InvariantViolation
from nagahana.models.advisor.model import Advisor
from nagahana.models.batch import AnalysisOut
from nagahana.models.forecaster.losses import NO_TARGET
from nagahana.models.forecaster.model import Forecaster
from nagahana.models.forecaster.routes import gumbel_argmax
from nagahana.models.verifier.feedback import ROUTE_NO_TARGET, RouteSpec
from nagahana.models.verifier.params import call_with
from nagahana.models.verifier.situations import repeat_trigger

Overrides = Mapping[str, torch.Tensor] | None


@dataclass
class StepLogProbs:
    """Per-step log-probabilities of R action sequences of one situation under one parameter view.

    logp [R, K]: log pi(a_k | prefix) of the taken step (first plus second factor); first_logp [R, K];
    first_dist [R, K, A1]: log-distribution of the first factor; second_dist [R, K, A2]: log-distribution
    of the second factor given the taken first factor; present [R, K]: steps that exist (a shorter plan
    is padded); mask [R, K]: steps that exist and that the policy can express (a target inside the
    policy's context). Forecaster only: hazard [R, K] (float64) and stage_logits [R, K, S].
    """

    logp: torch.Tensor
    first_logp: torch.Tensor
    first_dist: torch.Tensor
    second_dist: torch.Tensor
    mask: torch.Tensor
    present: torch.Tensor
    hazard: torch.Tensor | None = None
    stage_logits: torch.Tensor | None = None

    def expressible(self) -> torch.Tensor:
        """bool [R]: every present step can be expressed by the policy and has a finite log-probability."""
        ok = torch.where(self.present, self.mask & torch.isfinite(self.logp), torch.ones_like(self.mask))
        return ok.all(-1)

    def sequence_logp(self) -> torch.Tensor:
        """log pi(o) = sum over present steps [R] (float64); -inf for a sequence the policy cannot express."""
        lp = torch.where(self.mask, self.logp, torch.zeros_like(self.logp)).sum(-1)
        return torch.where(self.expressible(), lp, torch.full_like(lp, float("-inf")))

    def detached(self) -> StepLogProbs:
        """A copy without autograd history (old and reference views)."""
        return StepLogProbs(self.logp.detach(), self.first_logp.detach(), self.first_dist.detach(), self.second_dist.detach(),
                            self.mask, self.present, None if self.hazard is None else self.hazard.detach(),
                            None if self.stage_logits is None else self.stage_logits.detach())


def categorical_kl(logp: torch.Tensor, logq: torch.Tensor) -> torch.Tensor:
    """KL(p || q) along the last axis from log-probabilities (float64); options with p = 0 contribute 0.

    Inputs are sanitised before the products so that masked options (log p = -inf) carry neither NaN
    values nor NaN gradients.
    """
    lp, lq = logp.double(), logq.double()
    p = lp.exp()
    live = p > 0
    lp_s = torch.where(live, lp, torch.zeros_like(lp))
    lq_s = torch.where(live, lq, torch.zeros_like(lq))
    return (p * (lp_s - lq_s)).sum(-1)


def analytic_step_kl(cand: StepLogProbs, ref: StepLogProbs, old_first_logp: torch.Tensor | None = None) -> torch.Tensor:
    """D_k = KL_1 + w_k KL_2 per step [R, K] (module docstring); zero on masked steps.

    old_first_logp: log pi_old of the taken first factor when the steps were drawn from pi_old (GRPO);
    None when they were not drawn from any policy (given items): then w_k = 1, the KL at the taken first
    factor, a diagnostic of the given prefixes.
    """
    kl1 = categorical_kl(cand.first_dist, ref.first_dist)
    kl2 = categorical_kl(cand.second_dist, ref.second_dist)
    w = torch.ones_like(kl2) if old_first_logp is None else torch.exp(cand.first_logp - old_first_logp.double())
    d = kl1 + w * kl2
    return torch.where(cand.mask, d, torch.zeros_like(d))


def k3_kl(cand: StepLogProbs, ref: StepLogProbs) -> torch.Tensor:
    """pi_ref/pi_theta - log(pi_ref/pi_theta) - 1 at every taken step [R, K] (Shao et al., eq. 4); zero on masked steps."""
    r = torch.where(cand.mask, ref.logp.double() - cand.logp.double(), torch.zeros_like(cand.logp, dtype=torch.float64))
    return torch.where(cand.mask, torch.exp(r) - r - 1.0, torch.zeros_like(r))


def route_codes(routes: Sequence[RouteSpec]) -> tuple[torch.Tensor, torch.Tensor]:
    """(techniques [R, K], teacher-forcing target codes [R, K]) of routes: -1 (no target) becomes NO_TARGET."""
    if not routes:
        raise InvariantViolation("no routes")
    k = routes[0].horizon
    if any(r.horizon != k for r in routes):
        raise InvariantViolation("routes of one evaluation share one horizon")
    tech = torch.tensor([list(r.techniques) for r in routes], dtype=torch.long)
    tgt = torch.tensor([list(r.targets) for r in routes], dtype=torch.long)
    tgt = torch.where(tgt == ROUTE_NO_TARGET, torch.full_like(tgt, NO_TARGET), tgt)
    return tech, tgt


def routes_from_codes(techniques: torch.Tensor, targets: torch.Tensor) -> tuple[RouteSpec, ...]:
    """Route specs of teacher-forcing codes (NO_TARGET becomes -1); the inverse of `route_codes`."""
    out = []
    for t_row, v_row in zip(techniques.tolist(), targets.tolist(), strict=True):
        out.append(RouteSpec(techniques=tuple(int(t) for t in t_row),
                             targets=tuple(ROUTE_NO_TARGET if int(v) == NO_TARGET else int(v) for v in v_row)))
    return tuple(out)


def _forecaster_terms(fc: Forecaster, an1: AnalysisOut, tech: torch.Tensor, tgt: torch.Tensor) -> StepLogProbs:
    # One teacher-forced pass over R routes of one situation (rows of M = R).
    r, k = tech.shape
    out = fc.teacher_forced(repeat_trigger(an1, r), tech[None].to(an1.context.device), tgt[None].to(an1.context.device))
    log_pi = out["log_pi"]                                                       # [R, K, T] float64
    ptr = out["pointer"]                                                         # [R, K, C+1] float64
    slot = out["slot"]                                                           # [R, K]
    valid = (slot >= 0) & (tech.to(slot.device) >= 0)
    t_idx = tech.to(log_pi.device).clamp_min(0).unsqueeze(-1)
    first = log_pi.gather(-1, t_idx).squeeze(-1)                                 # [R, K]
    second = ptr.gather(-1, slot.clamp_min(0).unsqueeze(-1)).squeeze(-1)         # [R, K]
    heads = out["heads"]
    return StepLogProbs(logp=first + second, first_logp=first, first_dist=log_pi, second_dist=ptr, mask=valid,
                        present=torch.ones_like(valid), hazard=heads["hazard"], stage_logits=heads["stage_logits"])


def _sample_step(fc: Forecaster, an_rep: AnalysisOut, tech: torch.Tensor, tgt: torch.Tensor, k: int,
                 u_tech: torch.Tensor, u_tgt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # Draw step k of every route: technique from pi(tech | s_k), then target from pi(tgt | s_k, tech).
    out = fc.teacher_forced(an_rep, tech[None], tgt[None])
    log_pi = out["log_pi"][:, k]                                                 # [n, T]
    t_k = gumbel_argmax(log_pi, u_tech)                                          # [n]
    h_k = out["h"][:, k]                                                         # [n, dim] state after k actions
    rows = torch.arange(t_k.shape[0], device=t_k.device)
    ptr = fc.pointer_log_probs(out["enc"], rows, h_k, t_k[:, None])[:, 0]       # [n, C+1]
    slot = gumbel_argmax(ptr, u_tgt)                                             # [n]
    c = ptr.shape[-1] - 1
    ent = out["enc"].gathered.entity_index[rows, slot.clamp(max=c - 1)]
    code = torch.where(slot >= c, torch.full_like(slot, NO_TARGET), ent)
    return t_k, code


class ForecasterPolicy:
    """The adversary policy pi_A of a Forecaster, under any parameter view (module docstring)."""

    target = "forecaster"

    def __init__(self, forecaster: Forecaster) -> None:
        self.module = forecaster

    def step_log_probs(self, an1: AnalysisOut, techniques: torch.Tensor, targets: torch.Tensor, *,
                       overrides: Overrides = None) -> StepLogProbs:
        """Terms of R routes [R, K] (teacher-forcing target codes) of one situation [1, 1, ...]."""
        if techniques.dim() != 2 or techniques.shape != targets.shape:
            raise InvariantViolation("techniques and targets must be [R, K] of the same shape")
        n_tech = int(self.module.cfg.n_techniques)
        if bool((techniques >= n_tech).any()) or bool((techniques < 0).any()):
            raise InvariantViolation(f"technique slots must lie in 0 ... {n_tech - 1}")
        return call_with(self.module, overrides, _forecaster_terms, an1, techniques, targets)

    @torch.no_grad()
    def sample(self, an1: AnalysisOut, n: int, horizon_k: int, generator: torch.Generator, *,
               overrides: Overrides = None) -> tuple[torch.Tensor, torch.Tensor]:
        """n routes of K steps drawn exactly from the policy (module docstring): (techniques, target codes) [n, K]."""
        if n < 1 or horizon_k < 1:
            raise ValueError("n and horizon_k must be >= 1")
        dev = an1.context.device
        an_rep = repeat_trigger(an1, n)
        tech = torch.full((n, horizon_k), -1, dtype=torch.long, device=dev)
        tgt = torch.full((n, horizon_k), -1, dtype=torch.long, device=dev)
        n_tech = self.module.cfg.n_techniques
        c = self.module.cfg.context_entities
        # Uniforms drawn up front in a fixed order: the same seed gives the same routes for any parameter view.
        u_tech = torch.rand(horizon_k, n, n_tech, generator=generator, dtype=torch.float64).to(dev)
        u_tgt = torch.rand(horizon_k, n, c + 1, generator=generator, dtype=torch.float64).to(dev)
        for k in range(horizon_k):
            t_k, code = call_with(self.module, overrides, _sample_step, an_rep, tech, tgt, k, u_tech[k], u_tgt[k])
            tech[:, k] = t_k
            tgt[:, k] = code
        return tech, tgt


def _advisor_terms(adv: Advisor, an1: AnalysisOut, plans: Sequence[Sequence[tuple[int, int]]]) -> StepLogProbs:
    # Plans of different lengths are run in groups of equal length; results are padded to the longest.
    mem = adv.memory(an1)
    r = len(plans)
    length = max(len(p) for p in plans)
    a_slots = int(adv.cfg.n_actions)
    v = int(mem.active.shape[0])
    dev = mem.tokens.device
    logp = torch.zeros(r, length, dtype=torch.float64, device=dev)
    first = torch.zeros(r, length, dtype=torch.float64, device=dev)
    first_dist = torch.zeros(r, length, a_slots, dtype=torch.float64, device=dev)
    second_dist = torch.zeros(r, length, v, dtype=torch.float64, device=dev)
    mask = torch.zeros(r, length, dtype=torch.bool, device=dev)
    present = torch.zeros(r, length, dtype=torch.bool, device=dev)
    for size in sorted({len(p) for p in plans}):
        idx = [i for i, p in enumerate(plans) if len(p) == size]
        group = [tuple((int(a), int(e)) for a, e in plans[i]) for i in idx]
        h = adv.plan_forward(mem, group)[:, :size]                               # [g, l, dim]: state before each step
        g = len(group)
        acts = torch.tensor([[a for a, _ in p] for p in group], dtype=torch.long, device=dev)      # [g, l]
        ents = torch.tensor([[e for _, e in p] for p in group], dtype=torch.long, device=dev)      # [g, l]
        la = adv.action_log_probs(h).double()                                     # [g, l, A]
        lv = adv.target_log_probs(mem, h.reshape(g * size, -1), acts.reshape(-1, 1)).double().reshape(g, size, v)
        f = la.gather(-1, acts.unsqueeze(-1)).squeeze(-1)
        s = lv.gather(-1, ents.clamp(0, v - 1).unsqueeze(-1)).squeeze(-1)
        ok = (ents < v) & (acts < a_slots)
        rows = torch.tensor(idx, dtype=torch.long, device=dev)
        logp[rows, :size] = f + s
        first[rows, :size] = f
        first_dist[rows, :size] = la
        second_dist[rows, :size] = lv
        mask[rows, :size] = ok
        present[rows, :size] = True
    return StepLogProbs(logp=logp, first_logp=first, first_dist=first_dist, second_dist=second_dist, mask=mask, present=present)


class AdvisorPolicy:
    """The defender policy pi_D of an Advisor, under any parameter view (module docstring)."""

    target = "advisor"

    def __init__(self, advisor: Advisor) -> None:
        self.module = advisor

    def step_log_probs(self, an1: AnalysisOut, plans: Sequence[Sequence[tuple[int, int]]], *,
                       overrides: Overrides = None) -> StepLogProbs:
        """Terms of R plans of (slot, entity) steps of one situation [1, 1, ...], padded to the longest plan."""
        if not plans or any(len(p) == 0 for p in plans):
            raise InvariantViolation("plans must be non-empty sequences of (slot, entity) steps")
        return call_with(self.module, overrides, _advisor_terms, an1, plans)


__all__ = ["AdvisorPolicy", "ForecasterPolicy", "StepLogProbs", "analytic_step_kl", "categorical_kl", "k3_kl", "route_codes",
           "routes_from_codes"]
