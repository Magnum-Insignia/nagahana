"""Advisor: D3FEND counter-measure sequences re-imagined against the Forecaster's adversary (build-spec §2.9).

Purpose
-------
"The advisor is another agent/policy-value pair that directly works on both env & imagination to
produce counters" [A-15]; "only advisory" [Q-30]. Given one trigger's analysis and the Forecaster, it
proposes counter sequences (π_D), re-imagines the future under each (the effect model,
`effects.py`), ranks them by risk reduction, cost and feasibility, and returns an `AdvisoryBundle`
whose `advisory_only` is always True (D-33; the contract refuses anything else).

Owner sources: [A-15], [A-20], [Q-30], [Q-38] (GAN-like in function only: the Advisor never changes the
Forecaster). Decisions: D-01 (reads both caches), D-33, D-35 (memory-less). Assumptions: AS-22
(STAGED read of TAAFT), AS-23 (pricing), AS-24 (CVaR_0.2 ranking), AS-254, AS-255, AS-256 (counters
act at the trigger, all at once), AS-257 (policy-improvement temperature), AS-261 (ranking key).

Architecture
------------
- Memory: the trigger's Imagination positions (all V entity states + G adversary slots of TAAFT),
  encoded as m_i = RMSNorm(W_c c_i + W_r [p_i ; π^stage_i ; t_i] + e_type).
- Plan positions ("candidate counter positions"): x_0 = e_start; x_j = E_act[a_j] + E_level[ℓ(a_j)] +
  W_tgt m_{v_j}. `cfg.blocks` CrossBlocks: causal self-attention over plan positions (no positional
  code: a plan step is not a time, D-49), cross-attention to the memory, SwiGLU.
- π_D(a, v | plan) = π(a | h_L) · π(v | h_L, a): action over `n_actions` slots (unassigned slots
  masked), pointer over active entity positions. The level ℓ is a property of the action in the
  D3FEND table (every starting entry acts at exactly one level), so the level factor of
  slots × target × level is deterministic given the slot.
- V_D(plan) = w·h_L: the predicted objective J of the plan (trained on re-imagined outcomes).

Search and ranking
------------------
Beam search of width W over sequences of up to L_seq steps. At each depth every beam proposes its
top-W (action, best target) continuations by π_D; plans with the same set of steps are evaluated once
(counters act at once, AS-256); the W most probable are re-imagined with `rollouts` routes and common
random numbers (the same seed as the no-counter baseline, Gumbel draws in `routes.py`):

    Δ_n = F_n^{counter}(K) − F_n^{none}(K),   F_n(K) = 1 − Π_{j≤K}(1 − h_{n,j})   (paired routes, weight 1/N)
    expected = mean_n Δ_n,  CVaR_α = mean of the worst α share (largest Δ),  worst = max_n Δ_n

Ranking key (AS-261: AS-24 with AS-23's κ): feasible first, then CVaR_α + κ·cost (lower is better), then
cost. The expected and worst cases are reported beside CVaR in every `CounterSequence`.
Feasibility = rule checks (`effects.rule_check`) and, when a physics callable is supplied, the
physics term of the re-imagined effects: max Φ_phys(ẑ) ≤ τ (`cfg.physics_tau`).

Invariants (tested): advisory_only is True for every output; CVaR matches the closed form on known
distributions; rule-infeasible plans never rank above feasible ones.

Extension points: richer effects (new `EffectType` + `ForecastIntervention` field); time-staggered
counters (AS-256 relaxed) by intervening per imagined step.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from nagahana.governance.assumptions import assume
from nagahana.models.advisor.d3fend import LEVELS, D3FENDAction, action_table
from nagahana.models.advisor.effects import Step, build_intervention, disruption_cost, information_value, rule_check
from nagahana.models.batch import AnalysisOut
from nagahana.models.config.components import AdvisorConfig
from nagahana.models.forecaster.context import stage_probabilities
from nagahana.models.forecaster.model import Forecaster, MarginalEnergy
from nagahana.models.forecaster.routes import cvar_upper, route_cumulative
from nagahana.models.heads.policy_value import assumed_coupling, head_input
from nagahana.models.vocab import N_STAGES
from nagahana.nn.blocks import AttnContext, CrossBlock
from nagahana.nn.loop import TwoStreamStack
from nagahana.nn.norms import RMSNorm
from nagahana.objectives.rewards import advisor_objective
from nagahana.roles.contracts import AdvisoryBundle, CounterSequence, CounterStep

#: Physics on predicted effects: back-projected latents [..., dz] → Φ_phys ≥ 0 [...] (Decoder ∘ physics term).
PhysicsCheck = Callable[[torch.Tensor], torch.Tensor]


def slice_trigger(analysis: AnalysisOut, b: int, m: int) -> AnalysisOut:
    """One trigger of an `AnalysisOut` as a [1, 1, …] analysis (fields the Forecaster/Advisor read)."""
    def one(x: torch.Tensor) -> torch.Tensor:
        return x[b:b + 1, m:m + 1]

    return AnalysisOut(
        context=one(analysis.context), token_mask=one(analysis.token_mask), imagination_kv=[],
        y0=one(analysis.y0), y=one(analysis.y), energy_trace=analysis.energy_trace,
        lens_energy={k: one(v) for k, v in analysis.lens_energy.items()},
        lens_share={k: one(v) for k, v in analysis.lens_share.items()},
        readouts={k: one(v) for k, v in analysis.readouts.items()},
        passes=analysis.passes, descent_steps=analysis.descent_steps,
    )


@dataclass
class TriggerMemory:
    """The Advisor's view of one trigger: encoded positions and the readouts it needs."""

    tokens: torch.Tensor        # [1, V+G, dim]
    valid: torch.Tensor         # [1, V+G]
    active: torch.Tensor        # [V] active entities (pointer support)
    stage_probs: torch.Tensor   # [V, S]
    trust: torch.Tensor | None  # [V]


@dataclass
class Evaluation:
    """A re-imagined counter plan (one row of the ranking)."""

    plan: tuple[Step, ...]
    log_prob: float
    delta_expected: float
    delta_cvar: float
    delta_worst: float
    cost: float
    info_value: float
    feasible: bool
    reasons: tuple[str, ...]
    physics_max: float | None

    def rank_key(self, kappa: float) -> tuple[bool, float, float]:
        return (not self.feasible, self.delta_cvar + kappa * self.cost, self.cost)


class Advisor(nn.Module):
    """Defender policy/value over D3FEND counters. See the module docstring."""

    def __init__(self, cfg: AdvisorConfig, *, d_context: int, latent_dim: int, n_stages: int = N_STAGES) -> None:
        super().__init__()
        self.cfg = cfg
        self.latent_dim = latent_dim
        self.table: tuple[D3FENDAction | None, ...] = action_table(cfg.n_actions)
        dim = cfg.dim
        # --- memory (Imagination positions)
        self.mem_proj = nn.Linear(d_context, dim, bias=False)
        self.read_proj = nn.Linear(2 + n_stages, dim, bias=False)       # [compromise ; stage ; trust]
        self.type_emb = nn.Embedding(2, dim)                            # 0 adversary slot, 1 entity
        self.mem_norm = RMSNorm(dim)
        # --- plan positions
        self.start = nn.Parameter(torch.randn(dim) * 0.02)
        self.act_emb = nn.Embedding(cfg.n_actions, dim)
        self.level_emb = nn.Embedding(len(LEVELS), dim)
        self.tgt_proj = nn.Linear(dim, dim, bias=False)
        self.stack = TwoStreamStack([CrossBlock(dim, cfg.heads, mlp_hidden=cfg.mlp_hidden) for _ in range(cfg.blocks)])
        self.norm = RMSNorm(dim)
        # --- π_D and V_D
        self.action_head = nn.Linear(dim, cfg.n_actions)
        self.ptr_q = nn.Linear(dim, dim, bias=False)
        self.ptr_k = nn.Linear(dim, dim, bias=False)
        self.value_head = nn.Linear(dim, 1)
        #: AS-22 STAGED: stop-gradient into TAAFT until the joint phase.
        self.joint_phase = False
        level_of = [LEVELS.index(a.level) if a is not None else 0 for a in self.table]
        self.level_of: torch.Tensor
        self.register_buffer("level_of", torch.tensor(level_of, dtype=torch.long), persistent=False)
        self.assigned: torch.Tensor
        self.register_buffer("assigned", torch.tensor([a is not None for a in self.table]), persistent=False)

    # ================================================================== network
    def memory(self, an1: AnalysisOut) -> TriggerMemory:
        """Encode the positions of a single-trigger analysis ([1, 1, …])."""
        coupling = assumed_coupling()

        def rd(x: torch.Tensor) -> torch.Tensor:
            return head_input(x, coupling, joint_phase=self.joint_phase)

        ctx = rd(an1.context)[0, 0].float()                              # [V+G, d_context]
        comp = rd(an1.readouts["compromise"])[0, 0].float()               # [V]
        stage = stage_probabilities(rd(an1.readouts["stage"]))[0, 0]      # [V, S]
        trust_t = an1.readouts.get("trust")
        trust = None if trust_t is None else rd(trust_t)[0, 0].float()
        v = comp.shape[0]
        g = ctx.shape[0] - v
        read_e = torch.cat([comp[:, None], stage, (trust if trust is not None else torch.zeros_like(comp))[:, None]], dim=-1)
        read = torch.cat([read_e, read_e.new_zeros(g, read_e.shape[-1])], dim=0)          # [V+G, 2+S]
        is_ent = torch.cat([torch.ones(v, dtype=torch.long), torch.zeros(g, dtype=torch.long)]).to(ctx.device)
        tok = self.mem_norm(self.mem_proj(ctx) + self.read_proj(read) + self.type_emb(is_ent))
        valid = an1.token_mask[0, 0]
        return TriggerMemory(tokens=tok[None], valid=valid[None], active=valid[:v].clone(), stage_probs=stage, trust=trust)

    def plan_forward(self, mem: TriggerMemory, plans: Sequence[Sequence[Step]]) -> torch.Tensor:
        """Plans of equal length L → outputs [R, L+1, dim] (position 0 = start, j = step j)."""
        r = len(plans)
        length = len(plans[0]) if plans else 0
        if any(len(p) != length for p in plans):
            raise ValueError("plan_forward needs plans of equal length")
        dim = self.cfg.dim
        x0 = self.start.expand(r, 1, dim)
        if length:
            slots = torch.tensor([[s for s, _ in p] for p in plans], dtype=torch.long, device=x0.device)     # [R, L]
            ents = torch.tensor([[v for _, v in p] for p in plans], dtype=torch.long, device=x0.device)
            steps = self.act_emb(slots) + self.level_emb(self.level_of[slots]) + self.tgt_proj(mem.tokens[0, ents])
            x = torch.cat([x0, steps], dim=1)                                                                    # [R, L+1, dim]
        else:
            x = x0
        n = length + 1
        ctx = AttnContext(
            allowed=torch.ones(n, n, dtype=torch.bool, device=x.device).tril()[None, None],   # causal over plan steps
            cross_source=mem.tokens.expand(r, -1, -1),
            cross_allowed=mem.valid[:, None, None, :].expand(r, 1, 1, -1),
        )
        out, _, _ = self.stack.memory(x, ctx)
        return out

    def action_log_probs(self, h_last: torch.Tensor) -> torch.Tensor:
        """log π(a | plan) [R, n_actions]; unassigned slots −∞."""
        logits = self.action_head(self.norm(h_last)).float().masked_fill(~self.assigned, float("-inf"))
        return torch.log_softmax(logits, dim=-1)

    def target_log_probs(self, mem: TriggerMemory, h_last: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """log π(v | plan, a) over the V entity positions: h_last [R, dim], actions [R, A'] → [R, A', V]."""
        v = mem.active.shape[0]
        q = self.ptr_q(self.norm(h_last).unsqueeze(1) + self.act_emb(actions))            # [R, A', dim]
        k = self.ptr_k(mem.tokens[0, :v])                                                  # [V, dim]
        logits = torch.einsum("rad,vd->rav", q.float(), k.float()) / math.sqrt(q.shape[-1])
        logits = logits.masked_fill(~mem.active[None, None], float("-inf"))
        return torch.log_softmax(logits, dim=-1)

    def plan_value(self, h_last: torch.Tensor) -> torch.Tensor:
        """V_D(plan) [R]: predicted objective J of the plan."""
        return self.value_head(self.norm(h_last)).squeeze(-1)

    # ================================================================== evaluation by re-imagination
    def evaluate(self, an1: AnalysisOut, forecaster: Forecaster, plan: tuple[Step, ...], *, mem: TriggerMemory,
                 baseline_f: torch.Tensor, horizon_k: int, rollouts: int, seed: int, log_prob: float = 0.0,
                 exposure: MarginalEnergy | None = None, entity_kind: torch.Tensor | None = None,
                 entity_internal: torch.Tensor | None = None, technique_plane: torch.Tensor | None = None,
                 physics: PhysicsCheck | None = None) -> Evaluation:
        """Re-imagine one plan with common random numbers and score it (module docstring)."""
        cfg = self.cfg
        v = mem.active.shape[0]
        reasons = rule_check(plan, self.table, mem.active, entity_internal, technique_plane)
        inter = build_intervention(plan, self.table, v, technique_plane, device=an1.context.device)
        out = forecaster.imagine(an1, horizon_k=horizon_k, routes_n=rollouts, generator=torch.Generator().manual_seed(seed),
                                 exposure=exposure, intervention=inter)
        f_c = route_cumulative(out.hazard[0, 0].double())[:, -1]                           # [N]
        delta = f_c - baseline_f                                                           # paired routes
        w = torch.ones_like(delta)
        cvar = float(cvar_upper(delta, w, cfg.cvar_alpha))
        phys_max = None
        if physics is not None:
            phys_max = float(physics(out.step_latent[0, 0]).max())
            if phys_max > cfg.physics_tau:
                reasons = (*reasons, f"physics: max Φ_phys of predicted effects {phys_max:.3g} > τ = {cfg.physics_tau:g}")
        cost, _ = disruption_cost(plan, self.table, entity_kind, cfg.criticality)
        iv = information_value(plan, self.table, mem.stage_probs, mem.trust)
        return Evaluation(plan=plan, log_prob=log_prob, delta_expected=float(delta.mean()), delta_cvar=cvar,
                          delta_worst=float(delta.max()), cost=cost, info_value=iv, feasible=not reasons,
                          reasons=reasons, physics_max=phys_max)

    def baseline(self, an1: AnalysisOut, forecaster: Forecaster, *, horizon_k: int, rollouts: int, seed: int,
                 exposure: MarginalEnergy | None = None) -> torch.Tensor:
        """Per-route F_n^{none}(K) [N] under the same random numbers as every evaluation."""
        out = forecaster.imagine(an1, horizon_k=horizon_k, routes_n=rollouts, generator=torch.Generator().manual_seed(seed),
                                 exposure=exposure)
        return route_cumulative(out.hazard[0, 0].double())[:, -1]

    def propose(self, mem: TriggerMemory, plans: Sequence[tuple[Step, ...]], width: int) -> list[tuple[tuple[Step, ...], float]]:
        """Top-`width` continuations (action, its most likely target) of each plan, with their log π_D."""
        h = self.plan_forward(mem, plans)[:, -1]                                          # [R, dim]
        log_a = self.action_log_probs(h)                                                   # [R, A]
        n_assigned = int(self.assigned.sum())
        top_lp, top_a = torch.topk(log_a, min(width, n_assigned), dim=-1)                  # [R, W]
        log_v = self.target_log_probs(mem, h, top_a)                                       # [R, W, V]
        best_lp, best_v = log_v.max(-1)                                                    # [R, W]
        out: list[tuple[tuple[Step, ...], float]] = []
        for i, p in enumerate(plans):
            for j in range(top_a.shape[1]):
                lp = float(top_lp[i, j] + best_lp[i, j])
                if math.isfinite(lp):
                    out.append(((*p, (int(top_a[i, j]), int(best_v[i, j]))), lp))
        return out

    # ================================================================== the advisory search
    @torch.no_grad()
    def advise(self, analysis: AnalysisOut, forecaster: Forecaster, *, beam_width: int, max_steps: int, rollouts: int,
               generator: torch.Generator | None = None, b: int = 0, m: int | None = None, horizon_k: int | None = None,
               exposure: MarginalEnergy | None = None, entity_kind: torch.Tensor | None = None,
               entity_internal: torch.Tensor | None = None, entity_names: Sequence[str] | None = None,
               technique_plane: torch.Tensor | None = None, physics: PhysicsCheck | None = None,
               ) -> tuple[AdvisoryBundle, dict[str, Any]]:
        """Beam search → ranked, re-imagined counter sequences for trigger (b, m) (default: the window's last trigger).

        horizon_k: K of the re-imagination (default: the Forecaster's run-time default, reported in the info).
        entity_kind [V], entity_internal [V]: the trigger's entity table, if known (pricing and sensor rule).
        Returns (AdvisoryBundle — always advisory_only=True —, info dict with the evaluation record).
        """
        assume("AS-24", by=__name__)
        assume("AS-23", by=__name__)
        cfg = self.cfg
        mm = analysis.context.shape[1] - 1 if m is None else m
        an1 = slice_trigger(analysis, b, mm)
        k = forecaster.cfg.horizon_k if horizon_k is None else horizon_k
        seed = int(torch.randint(0, 2**62, (1,), generator=generator))
        base_f = self.baseline(an1, forecaster, horizon_k=k, rollouts=rollouts, seed=seed, exposure=exposure)
        mem = self.memory(an1)
        evaluated: dict[frozenset[Step], Evaluation] = {}
        beams: list[tuple[Step, ...]] = [()]
        for _depth in range(max_steps):
            if not beams:
                break
            cands = self.propose(mem, beams, beam_width)
            # Same set of steps = same intervention (AS-256): evaluate each set once, keep the most probable order.
            best: dict[frozenset[Step], tuple[tuple[Step, ...], float]] = {}
            for plan, lp in cands:
                key = frozenset(plan)
                if len(key) < len(plan) or key in evaluated:
                    continue
                if key not in best or lp > best[key][1]:
                    best[key] = (plan, lp)
            chosen = sorted(best.values(), key=lambda t: -t[1])[:beam_width]
            new: list[Evaluation] = []
            for plan, lp in chosen:
                ev = self.evaluate(an1, forecaster, plan, mem=mem, baseline_f=base_f, horizon_k=k, rollouts=rollouts,
                                   seed=seed, log_prob=lp, exposure=exposure, entity_kind=entity_kind,
                                   entity_internal=entity_internal, technique_plane=technique_plane, physics=physics)
                evaluated[frozenset(plan)] = ev
                new.append(ev)
            beams = [e.plan for e in sorted(new, key=lambda e: e.rank_key(cfg.cost_weight))[:beam_width]]

        ranked = sorted(evaluated.values(), key=lambda e: e.rank_key(cfg.cost_weight))
        sequences = tuple(self._to_sequence(e, entity_names) for e in ranked)
        bundle = AdvisoryBundle(sequences=sequences)            # advisory_only stays True (D-33)
        info: dict[str, Any] = {
            "trigger": (b, mm), "horizon_k": k, "rollouts": rollouts, "seed": seed, "cvar_alpha": cfg.cvar_alpha,
            "baseline_p_inf_K": float(base_f.mean()), "evaluated": len(evaluated),
            "entity_kinds_known": entity_kind is not None, "physics_checked": physics is not None,
            "objective": [float(advisor_objective(torch.tensor(e.delta_cvar), torch.tensor(e.cost), kappa=cfg.cost_weight))
                          for e in ranked],
            "evaluations": ranked,
            "notes": ([] if entity_kind is not None else ["entity kinds unknown: every target priced at the most critical kind"])
            + ([] if physics is not None else ["physics of predicted effects not checked (no physics callable supplied)"]),
        }
        return bundle, info

    def _to_sequence(self, e: Evaluation, entity_names: Sequence[str] | None) -> CounterSequence:
        steps = []
        for slot, v in e.plan:
            act = self.table[slot]
            assert act is not None
            name = entity_names[v] if entity_names is not None else f"entity:{v}"
            steps.append(CounterStep(tactic=act.tactic, action=f"{act.d3fend_id} {act.name}", target=name, level=act.level))
        return CounterSequence(steps=tuple(steps), delta_p_inf=e.delta_expected, disruption_cost=e.cost, feasible=e.feasible,
                               information_value=e.info_value, delta_p_inf_cvar=e.delta_cvar, delta_p_inf_worst=e.delta_worst,
                               cvar_alpha=self.cfg.cvar_alpha, infeasible_reasons=e.reasons)


__all__ = ["Advisor", "Evaluation", "PhysicsCheck", "TriggerMemory", "slice_trigger"]
