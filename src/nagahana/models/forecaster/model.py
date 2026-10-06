"""Forecaster: the adversary's imagined futures over K steps along at most N routes (build-spec §2.8).

Purpose
-------
"Policy and value functions, MPC-guided model-based deep RL [Q-28], plan as the adversary would. They
imagine K future states along at most N routes" (architecture §3.8). The Forecaster reads TAAFT's
analysis at a trigger (Imagination, read-only for this purpose), imagines routes, and returns
`ForecastOut` (models/batch.py) → `ForecastBundle` (roles/contracts.py) per trigger.

Owner sources: [Q-28], [A-06], [A-14], [Q-24] (process reward per imagined step).
Decisions: D-30/D-44 (K and N set at run time and recorded), D-46 (≤ N distinct routes), D-49
(imagined step time t = k·window_seconds; no index positions), D-12 held → AS-22 (STAGED coupling).
Assumptions: AS-17 (adversary reward), AS-18 (infiltration state), AS-20 (700 technique slots), AS-21
(MPPI), AS-22, AS-250 (steps carry actions; the state is the step's output), AS-251 (route
estimator), AS-252 (exposure = positive marginal-energy rise over the trigger's baseline).

Architecture
------------
Sequence per route: [context (T = G_adv + C positions, t = 0) ; step 0 (t = 0) ; step 1 … step K
(t = k·w)], through `cfg.blocks` pre-norm SelfBlocks (`sequence.RouteStack`), causal over steps.

- Context positions: `context.ContextEncoder` over [adversary slots ; top-C entities by compromise].
- Step 0 input: x_0 = W_σ σ + e_start, σ the trigger summary (`context.SummaryPool`). Its output
  h_0 is the current state s_0.
- Step k ≥ 1 input: x_k = E_tech[a^tech_k] + W_tgt x_{tgt(k)} (the target's context encoding, or a
  learned "no target" vector). Its output h_k is the imagined state s_k after action a_k (AS-250: the
  input carries the action, the residual stream carries the state; feeding a predicted state back as
  input would make teacher forcing and imagination different computations — exposure bias, Bengio
  et al., "Scheduled Sampling", NeurIPS 2015, arXiv:1506.03099).

Heads (on RMSNorm(h)):
- policy π_A(a | s_{k−1}) = π(tech | s) · π(tgt | s, tech) — technique over n_techniques slots;
  target a pointer over the C entity positions plus "no target":
      logit_j = ⟨W_q(h + E_tech[tech]), W_k x_j⟩ / √d   (j valid, not blocked)
- per step k ≥ 1: stage posterior softmax(W_s h_k) [n_stages]; hazard h_k = σ(w_h·h_k) ∈ (0, 1);
  value V(s_k); task reward r̂_k (Δ progress + β·infiltration part of AS-17); back-projected latent
  ẑ_k ∈ ℝ^{dz} of the target; imagined hypothesis ŷ_k ∈ ℝ^{d_y} of the target (for the exposure term);
  predicted next trigger summary σ̂_k (latent consistency, losses.py).

Imagination (`imagine`, AS-21)
------------------------------
For each step k and route n (all routes in parallel, prefix K/V cached):
1. candidates: the top-B techniques of π(·|s_{k−1}); for each, a target drawn from π(tgt | s, tech);
2. one-step lookahead per candidate: run the stack one step → s′; Q = r̂(s′) − κ·exposure(s′) + γV(s′);
3. π̃ = MPPI re-weighting over the B candidates (routes.mppi_log_weights, temperature η);
4. draw a_k ~ π̃ (Gumbel-max with pre-drawn uniforms: common random numbers across interventions);
   s_k = s′ of the chosen candidate (its K/V appended to the cache).
Route log-probability: log q_n = Σ_k [log π̃(tech_k) + log π(tgt_k | tech_k)].
Then merge identical sequences, weight routes (AS-251), and summarise:

    P_inf(k) = Σ_n w_n (1 − Π_{j≤k}(1 − h_{n,j}))          (non-decreasing by construction)
    band = weighted 10 % / 90 % quantiles of F_n(k) over routes; median = weighted 50 % quantile
    stage(k) = Σ_n w_n softmax(stage logits)_{n,k};  mode route = argmax_n w_n

Exposure (AS-17, AS-252): with TAAFT's marginal energy E(∅, y) supplied as a callable on hypotheses
[..., d_y] → [...], exposure_k = max(0, E(∅, ŷ_k) − Ē_0), Ē_0 the mean marginal energy of the
trigger's selected entity hypotheses; exposure = 0 when no callable is given.

Interventions (Advisor's effect model, D-33 advisory): `ForecastIntervention` edits *structure*, never
learned values, so the Advisor cannot learn effects that flatter itself:
- no_target: the entity cannot be targeted (inbound blocked / isolated);
- no_outbound: the entity's state cannot be read by any other position (outbound blocked / isolated);
- blocked_plane + technique_plane: techniques of a plane cannot target the entity.

Precision (D-54: weights and compute fp32, outputs fp64)
-------------------------------------------------------
The route transformer, its K/V cache and every head projection run in float32 (weights are stored and
served in float32). The per-step head *logits* are cast to float64 before their link function, and
everything downstream of a link function is float64:
- hazard h_k = σ(logit.double()) → survival, P_inf(k), band, median (`routes.py`);
- log π(tech | s) and log π(target | s, tech) (log-softmax of float64 logits) → MPPI log-weights →
  route log-probability log q(r) → route weights (softmax over distinct routes);
- stage posterior per route and step = softmax(stage_logits.double()); the route mixture of stages.
Casting a float32 logit to float64 is exact and differentiable (its backward casts the gradient back
to float32), so teacher forcing trains the same float64 link functions that imagination reports.
Not converted (not D-54 outputs, documented in AS-450): imagined states, values, task rewards,
back-projected latents and hypotheses — learned float32 estimates that feed networks or MSE losses.

Invariants (tested): P_inf ∈ [0, 1], non-decreasing; ≤ N distinct routes; weights sum to 1; masked
targets never chosen; dense ≡ incremental computation; `to_bundle` satisfies ForecastBundle; P_inf,
band, median, stage, hazard and route weights are float64 (tests/test_precision_outputs.py).

Extension points: other route summaries in `routes.py`; richer interventions as new mask fields.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from nagahana.governance.assumptions import assume
from nagahana.models.batch import AnalysisOut, ForecastOut
from nagahana.models.config.components import ForecasterConfig
from nagahana.models.forecaster.context import ContextEncoder, GatheredContext, SummaryPool, gather_context
from nagahana.models.forecaster.routes import (
    gumbel_argmax,
    merge_routes,
    mixture_hazard,
    mppi_log_weights,
    p_inf_from_hazards,
    route_cumulative,
    route_weights,
    weighted_quantile,
)
from nagahana.models.forecaster.sequence import RouteStack
from nagahana.models.heads.policy_value import assumed_coupling, head_input
from nagahana.nn.blocks import KV
from nagahana.nn.norms import RMSNorm
from nagahana.roles.contracts import (
    AttackStage,
    ComputeRecord,
    DrivingFeature,
    ForecastBundle,
    ImaginedPath,
)

#: Marginal-energy callable from TAAFT: hypotheses [..., d_y] → E(∅, y) [...] (AS-16 reading).
MarginalEnergy = Callable[[torch.Tensor], torch.Tensor]


@dataclass
class ForecastIntervention:
    """Structural edits of the imagined state (the Advisor's effect model). Shapes per window entity.

    no_target: bool [B, M, V] — entity cannot be an action target (inbound blocked or isolated).
    no_outbound: bool [B, M, V] — no other position may read the entity's state (outbound blocked or isolated).
    blocked_plane: bool [B, M, V, n_planes] — techniques of plane p cannot target the entity.
    technique_plane: long [n_techniques] — plane code of each technique slot (−1 unknown); needed by blocked_plane.
    """

    no_target: torch.Tensor | None = None
    no_outbound: torch.Tensor | None = None
    blocked_plane: torch.Tensor | None = None
    technique_plane: torch.Tensor | None = None


@dataclass
class EncodedContext:
    """Everything the route stack needs for one batch of triggers (Bt = B·M rows)."""

    gathered: GatheredContext
    tokens: torch.Tensor          # [Bt, T, dim] encoded context positions
    ent_tokens: torch.Tensor      # [Bt, C, dim] the entity part (pointer keys / target inputs)
    ctx_allowed: torch.Tensor     # [Bt, 1, T, T] context ↔ context pattern
    step_ctx_allowed: torch.Tensor  # [Bt, T] context keys a step may read
    target_ok: torch.Tensor       # [Bt, C] entity positions that can be targeted
    blocked_plane: torch.Tensor | None  # [Bt, C, n_planes]
    technique_plane: torch.Tensor | None  # [n_techniques]
    summary: torch.Tensor         # [Bt, dim] σ
    energy_baseline: torch.Tensor | None  # [Bt] Ē_0 (AS-252) or None


class Forecaster(nn.Module):
    """The adversary's policy/value world model over imagined steps. See the module docstring."""

    def __init__(self, cfg: ForecasterConfig, *, d_context: int, d_hyp: int, latent_dim: int, n_stages: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.n_stages, self.latent_dim, self.d_hyp = n_stages, latent_dim, d_hyp
        dim = cfg.dim
        # --- context and summary
        self.ctx_enc = ContextEncoder(d_context, d_hyp, n_stages, dim)
        self.summary = SummaryPool(dim, cfg.heads)
        self.sum_in = nn.Linear(dim, dim, bias=False)
        self.start = nn.Parameter(torch.randn(dim) * 0.02)
        # --- action inputs: n_techniques slots + one "unknown technique" row used only in teacher forcing
        self.tech_emb = nn.Embedding(cfg.n_techniques + 1, dim)
        self.tgt_in = nn.Linear(dim, dim, bias=False)
        self.null_target = nn.Parameter(torch.randn(dim) * 0.02)      # "no target" input
        self.unknown_target = nn.Parameter(torch.randn(dim) * 0.02)   # label unknown (teacher forcing only)
        # --- dynamics
        self.stack = RouteStack(dim, cfg.heads, cfg.blocks, cfg.mlp_hidden, p_min=cfg.rotary_p_min, p_max=cfg.rotary_p_max)
        self.norm = RMSNorm(dim)
        # --- policy π_A
        self.tech_head = nn.Linear(dim, cfg.n_techniques)
        self.ptr_q = nn.Linear(dim, dim, bias=False)
        self.ptr_k = nn.Linear(dim, dim, bias=False)
        self.null_target_logit = nn.Parameter(torch.zeros(()))
        # --- per-step heads
        self.stage_head = nn.Linear(dim, n_stages)
        self.hazard_head = nn.Linear(dim, 1)
        self.value_head = nn.Linear(dim, 1)
        self.reward_head = nn.Linear(dim, 1)
        self.latent_head = nn.Linear(dim, latent_dim)
        self.hyp_head = nn.Linear(dim, d_hyp)
        self.summary_pred = nn.Linear(dim, dim)
        #: AS-22 STAGED: False = stop-gradient into TAAFT (first part of stage 5); True = joint fine-tune.
        self.joint_phase = False

    # ================================================================== context
    def encode(self, analysis: AnalysisOut, intervention: ForecastIntervention | None = None,
               exposure: MarginalEnergy | None = None) -> EncodedContext:
        """Select and encode the context of every trigger, with intervention masks (module docstring)."""
        coupling = assumed_coupling()                                   # AS-22 (D-12 held)
        joint = self.joint_phase
        g = gather_context(analysis, self.cfg.context_entities,
                           read=lambda x: head_input(x, coupling, joint_phase=joint))
        bt, t = g.valid.shape
        c, ga = g.n_entities, g.n_adv
        tokens = self.ctx_enc(g)                                        # [Bt, T, dim]
        summary = self.summary(tokens, g.valid)                         # [Bt, dim]

        # Gather per-entity intervention flags onto the C entity positions.
        def per_entity(flag: torch.Tensor | None) -> torch.Tensor:
            out = torch.zeros(bt, c, dtype=torch.bool, device=tokens.device)
            if flag is None:
                return out
            f = flag.reshape(bt, -1)                                    # [Bt, V]
            got = torch.gather(f, 1, g.entity_index.clamp_min(0))
            return got & (g.entity_index >= 0)

        no_tgt = per_entity(None if intervention is None else intervention.no_target)
        no_out = per_entity(None if intervention is None else intervention.no_outbound)
        blocked_plane = None
        if intervention is not None and intervention.blocked_plane is not None:
            bp = intervention.blocked_plane.reshape(bt, -1, intervention.blocked_plane.shape[-1])   # [Bt, V, P]
            blocked_plane = torch.gather(bp, 1, g.entity_index.clamp_min(0).unsqueeze(-1).expand(-1, -1, bp.shape[-1]))
            blocked_plane = blocked_plane & (g.entity_index >= 0).unsqueeze(-1)

        # Outbound blocking: an entity's key is hidden from every other position (edges removed).
        hidden = torch.cat([torch.zeros(bt, ga, dtype=torch.bool, device=tokens.device), no_out], dim=1)   # [Bt, T]
        key_ok = g.valid & ~hidden                                      # keys others may read
        eye = torch.eye(t, dtype=torch.bool, device=tokens.device)
        ctx_allowed = (key_ok[:, None, :] | (eye[None] & g.valid[:, None, :]))   # [Bt, T, T]
        target_ok = g.valid[:, ga:] & ~no_tgt                           # [Bt, C]

        baseline = None
        if exposure is not None:
            # Ē_0: mean marginal energy of the selected entity hypotheses at the trigger (AS-252).
            e_ent = exposure(g.hyp[:, ga:].float())                     # [Bt, C]
            w = g.valid[:, ga:].float()
            baseline = (e_ent * w).sum(-1) / w.sum(-1).clamp_min(1.0)
        return EncodedContext(
            gathered=g, tokens=tokens, ent_tokens=tokens[:, ga:], ctx_allowed=ctx_allowed.unsqueeze(1),
            step_ctx_allowed=key_ok, target_ok=target_ok, blocked_plane=blocked_plane,
            technique_plane=None if intervention is None else intervention.technique_plane,
            summary=summary, energy_baseline=baseline,
        )

    # ================================================================== small pieces
    def _start_input(self, enc: EncodedContext) -> torch.Tensor:
        return self.sum_in(enc.summary) + self.start                    # [Bt, dim]

    def _action_input(self, enc: EncodedContext, row: torch.Tensor, tech: torch.Tensor, slot: torch.Tensor,
                      *, unknown_target: torch.Tensor | None = None) -> torch.Tensor:
        """x_k = E_tech[tech] + W_tgt x_tgt. row [R] (Bt row of each item), tech [R], slot [R] in 0…C (C = no target)."""
        c = enc.ent_tokens.shape[1]
        ent = enc.ent_tokens[row, slot.clamp(max=c - 1)]                # [R, dim]
        tgt = torch.where((slot >= c).unsqueeze(-1), self.null_target.expand_as(ent), self.tgt_in(ent))
        if unknown_target is not None:
            tgt = torch.where(unknown_target.unsqueeze(-1), self.unknown_target.expand_as(tgt), tgt)
        return self.tech_emb(tech) + tgt

    def tech_log_probs(self, h: torch.Tensor) -> torch.Tensor:
        """log π(tech | s) from raw stack outputs h [..., dim] → float64 [..., n_techniques] (D-54).

        The head runs in float32; its logits are cast to float64 before the log-softmax, because these
        log-probabilities are summed into the route log-probability whose softmax is a reported weight.
        """
        return torch.log_softmax(self.tech_head(self.norm(h)).double(), dim=-1)

    def pointer_log_probs(self, enc: EncodedContext, row: torch.Tensor, h: torch.Tensor, tech: torch.Tensor) -> torch.Tensor:
        """log π(target | s, tech) over [C entity positions ; no target].

        row [R], h [R, dim], tech [R, B'] (technique per candidate; the unknown row allowed) → [R, B', C+1].
        Masked: invalid or untargetable entities, and (entity, plane-of-technique) pairs that are blocked.
        """
        d = h.shape[-1]
        q = self.ptr_q(self.norm(h).unsqueeze(1) + self.tech_emb(tech))         # [R, B', dim]
        k = self.ptr_k(enc.ent_tokens[row])                                      # [R, C, dim]
        # Pointer scores in float32 (the matmul is compute, D-54), then float64 for the link function.
        logits = (torch.einsum("rbd,rcd->rbc", q.float(), k.float()) / math.sqrt(d)).double()   # [R, B', C]
        ok = enc.target_ok[row].unsqueeze(1).expand_as(logits)                   # [R, B', C]
        if enc.blocked_plane is not None and enc.technique_plane is not None:
            plane = enc.technique_plane.to(tech.device)[tech.clamp(max=self.cfg.n_techniques - 1)]   # [R, B']
            bp = enc.blocked_plane[row]                                          # [R, C, P]
            hit = torch.gather(bp.unsqueeze(1).expand(-1, tech.shape[1], -1, -1), 3,
                               plane.clamp_min(0)[..., None, None].expand(-1, -1, bp.shape[1], 1)).squeeze(-1)
            ok = ok & ~(hit & (plane >= 0).unsqueeze(-1))
        logits = logits.masked_fill(~ok, float("-inf"))
        null = self.null_target_logit.double().expand(*logits.shape[:2], 1)
        return torch.log_softmax(torch.cat([logits, null], dim=-1), dim=-1)       # float64 [R, B', C+1]

    def step_heads(self, h: torch.Tensor) -> dict[str, torch.Tensor]:
        """Per-step heads on raw stack outputs h [..., dim] (see module docstring).

        Precision (D-54): "hazard" is float64 — σ of the float64-cast logit, so the survival product
        and P_inf are float64 from the link function on. "stage_logits" stay in the head's float32;
        every consumer applies softmax after casting them to float64 (`imagine`, the stage loss).
        The other heads are float32 learned estimates (AS-450).
        """
        n = self.norm(h)
        return {
            "stage_logits": self.stage_head(n),
            "hazard": torch.sigmoid(self.hazard_head(n).squeeze(-1).double()),
            "value": self.value_head(n).squeeze(-1),
            "reward_task": self.reward_head(n).squeeze(-1),
            "latent": self.latent_head(n),
            "hyp": self.hyp_head(n),
            "summary_pred": self.summary_pred(n),
        }

    def exposure_of(self, enc: EncodedContext, row: torch.Tensor, hyp: torch.Tensor,
                    exposure: MarginalEnergy | None) -> torch.Tensor:
        """exposure = max(0, E(∅, ŷ) − Ē_0[row]) (AS-252); zeros without a callable. hyp [R, ..., d_y] → [R, ...]."""
        if exposure is None or enc.energy_baseline is None:
            return hyp.new_zeros(hyp.shape[:-1])
        # AS-252 (new; registry entry requested in the build report)
        e = exposure(hyp.float())
        base = enc.energy_baseline[row].reshape(-1, *([1] * (e.dim() - 1)))
        return (e - base).clamp_min(0.0)

    # ================================================================== prefix (context + step 0)
    def prefix(self, enc: EncodedContext) -> tuple[torch.Tensor, list[KV], torch.Tensor]:
        """Run [context ; step 0] densely. → (h_0 [Bt, dim], per-block K/V [Bt, H, T+1, d_h], allowed keys [Bt, T+1])."""
        bt, t, _ = enc.tokens.shape
        x = torch.cat([enc.tokens, self._start_input(enc).unsqueeze(1)], dim=1)          # [Bt, T+1, dim]
        times = torch.zeros(bt, t + 1, dtype=torch.float64, device=x.device)
        allowed = torch.zeros(bt, 1, t + 1, t + 1, dtype=torch.bool, device=x.device)
        allowed[:, :, :t, :t] = enc.ctx_allowed
        allowed[:, 0, t, :t] = enc.step_ctx_allowed
        allowed[:, 0, t, t] = True
        out, kvs, _ = self.stack.dense(x, times, allowed)
        keys_ok = torch.cat([enc.step_ctx_allowed, torch.ones(bt, 1, dtype=torch.bool, device=x.device)], dim=1)
        return out[:, t], kvs, keys_ok

    # ================================================================== teacher forcing (training)
    def teacher_forced(self, analysis: AnalysisOut, technique: torch.Tensor, target_entity: torch.Tensor,
                       *, exposure: MarginalEnergy | None = None) -> dict[str, Any]:
        """Dense pass on given action sequences (labels where known).

        technique: long [B, M, K] (−1 unknown); target_entity: long [B, M, K] window entity index,
        −1 unknown, −2 "no target" (AS-260).

        Returns per-trigger-row outputs (Bt = B·M): h [Bt, K+1, dim] (steps 0…K), heads on steps 1…K,
        value on steps 0…K, log π(tech) [Bt, K, n_tech] from s_{k−1}, pointer log-probs [Bt, K, C+1]
        conditioned on the labelled technique, the target slot of each label (−1 if not in context),
        exposure [Bt, K], and the encoded context.
        """
        enc = self.encode(analysis, exposure=exposure)
        bt, t, dim = enc.tokens.shape
        k_steps = technique.shape[-1]
        tech = technique.reshape(bt, k_steps)
        tgt_ent = target_entity.reshape(bt, k_steps)
        c = enc.ent_tokens.shape[1]
        # Map labelled target entity → context entity slot (−1 if the entity is not among the C positions).
        match = (enc.gathered.entity_index.unsqueeze(1) == tgt_ent.unsqueeze(-1)) & (tgt_ent.unsqueeze(-1) >= 0)   # [Bt, K, C]
        slot = torch.where(match.any(-1), match.float().argmax(-1), torch.full_like(tgt_ent, -1))
        slot = torch.where(tgt_ent == -2, torch.full_like(slot, c), slot)      # −2 = "no target" (losses.NO_TARGET)
        tech_in = torch.where(tech >= 0, tech, torch.full_like(tech, self.cfg.n_techniques))   # unknown row
        unknown_tgt = slot < 0
        row = torch.arange(bt, device=tech.device).repeat_interleave(k_steps)
        x_steps = self._action_input(enc, row, tech_in.reshape(-1), slot.clamp_min(0).reshape(-1),
                                     unknown_target=unknown_tgt.reshape(-1)).reshape(bt, k_steps, dim)
        x = torch.cat([enc.tokens, self._start_input(enc).unsqueeze(1), x_steps], dim=1)      # [Bt, T+1+K, dim]
        w = self.cfg.window_seconds
        step_t = torch.arange(k_steps + 1, dtype=torch.float64, device=x.device) * w
        times = torch.cat([torch.zeros(bt, t, dtype=torch.float64, device=x.device), step_t.expand(bt, -1)], dim=1)
        n_all = t + 1 + k_steps
        allowed = torch.zeros(bt, 1, n_all, n_all, dtype=torch.bool, device=x.device)
        allowed[:, :, :t, :t] = enc.ctx_allowed
        allowed[:, 0, t:, :t] = enc.step_ctx_allowed.unsqueeze(1)
        allowed[:, 0, t:, t:] = torch.ones(k_steps + 1, k_steps + 1, dtype=torch.bool, device=x.device).tril()
        out, _, _ = self.stack.dense(x, times, allowed)
        h = out[:, t:]                                                    # [Bt, K+1, dim]
        heads = self.step_heads(h[:, 1:])
        value = self.value_head(self.norm(h)).squeeze(-1)                 # [Bt, K+1]
        log_pi = self.tech_log_probs(h[:, :-1])                          # [Bt, K, n_tech] from s_{k−1}
        ptr = self.pointer_log_probs(enc, torch.arange(bt, device=x.device).repeat_interleave(k_steps),
                                     h[:, :-1].reshape(bt * k_steps, dim), tech_in.reshape(-1, 1)).reshape(bt, k_steps, c + 1)
        expo = self.exposure_of(enc, torch.arange(bt, device=x.device), heads["hyp"].detach(), exposure)
        return {"enc": enc, "h": h, "heads": heads, "value": value, "log_pi": log_pi, "pointer": ptr,
                "slot": slot, "exposure": expo}

    # ================================================================== imagination (inference)
    @torch.no_grad()
    def imagine(self, analysis: AnalysisOut, *, horizon_k: int, routes_n: int, generator: torch.Generator | None = None,
                exposure: MarginalEnergy | None = None, intervention: ForecastIntervention | None = None,
                temperature: float | None = None) -> ForecastOut:
        """Imagine `routes_n` routes of `horizon_k` steps per trigger with MPPI-guided sampling (module docstring).

        `temperature`: MPPI η (default `cfg.mppi_temperature`; `math.inf` = plain policy sampling).
        Runs without gradients: imagination is inference; training uses `teacher_forced`.
        """
        if horizon_k < 1 or routes_n < 1:
            raise ValueError("horizon_k and routes_n must be >= 1")
        assume("AS-21", by=__name__)
        assume("AS-17", by=__name__)
        assume("AS-450", by=__name__)                                   # D-54 outputs float64, states float32
        cfg = self.cfg
        eta = cfg.mppi_temperature if temperature is None else temperature
        b, m = analysis.context.shape[:2]
        enc = self.encode(analysis, intervention, exposure)
        bt = b * m
        c = enc.ent_tokens.shape[1]
        n_cand = min(cfg.mppi_top_b, cfg.n_techniques)
        h0, kvs0, keys0 = self.prefix(enc)
        dev = h0.device
        # Pre-drawn uniforms: identical draws whatever the chunking and whatever the intervention (CRN).
        u_tgt = torch.rand(horizon_k, bt, routes_n, n_cand, c + 1, generator=generator).to(dev)
        u_sel = torch.rand(horizon_k, bt, routes_n, n_cand, generator=generator).to(dev)

        chunk = routes_n if cfg.route_chunk <= 0 else min(cfg.route_chunk, routes_n)
        pieces: list[dict[str, torch.Tensor]] = []
        for start in range(0, routes_n, chunk):
            n_c = min(chunk, routes_n - start)
            pieces.append(self._roll(enc, h0, kvs0, keys0, n_c, horizon_k, n_cand, eta,
                                     u_tgt[:, :, start:start + n_c], u_sel[:, :, start:start + n_c], exposure))
        r = {key: torch.cat([p[key] for p in pieces], dim=1) for key in pieces[0]}   # [Bt, N, K, ...]

        # ---------------------------------------------------------- merge, weight, summarise
        first, count = merge_routes(r["actions"])                                      # [Bt, N]
        weight = route_weights(r["logq"], first, count, estimator=cfg.route_estimator)  # [Bt, N]
        # AS-251 (new; registry entry requested in the build report): route-estimator reading
        hazard = r["hazard"]                                                           # [Bt, N, K]
        p_inf = p_inf_from_hazards(hazard, weight)                                     # [Bt, K] float64
        f = route_cumulative(hazard)                                                   # [Bt, N, K]
        lo_q, hi_q = cfg.band_quantiles
        band = torch.stack([weighted_quantile(f, weight, lo_q), weighted_quantile(f, weight, hi_q)], dim=-1)
        median = weighted_quantile(f, weight, 0.5)
        # Route mixture of the per-step stage posteriors, softmax of float64-cast logits (D-54).
        stage = (weight[..., None, None].double() * torch.softmax(r["stage_logits"].double(), dim=-1)).sum(1)   # [Bt, K, S]

        def bm(x: torch.Tensor) -> torch.Tensor:
            return x.reshape(b, m, *x.shape[1:])

        return ForecastOut(
            p_inf=bm(p_inf), p_inf_band=bm(band), p_inf_median=bm(median), stage=bm(stage), hazard=bm(hazard),
            route_weight=bm(weight.double()), route_actions=bm(r["actions"]), route_distinct=bm(first.sum(-1)),
            mode_route=bm(weight.argmax(-1)), step_state=bm(r["state"]), step_value=bm(r["value"]),
            step_reward=bm(r["reward"]), step_latent=bm(r["latent"]), horizon_k=horizon_k, routes_n=routes_n,
        )

    def _roll(self, enc: EncodedContext, h0: torch.Tensor, kvs0: Sequence[KV], keys0: torch.Tensor, n_c: int,
              horizon_k: int, n_cand: int, eta: float, u_tgt: torch.Tensor, u_sel: torch.Tensor,
              exposure: MarginalEnergy | None) -> dict[str, torch.Tensor]:
        """Roll `n_c` routes per trigger for K steps (one chunk). Uniforms: [K, Bt, n_c, B(, C+1)]."""
        cfg = self.cfg
        bt, dim = h0.shape
        rows = bt * n_c
        row = torch.arange(bt, device=h0.device).repeat_interleave(n_c)               # [R] Bt row of each route
        h = h0[row]                                                                    # [R, dim]
        kvs = [(k[row], v[row]) for k, v in kvs0]                                      # [R, H, T+1, d_h]
        keys = keys0[row]                                                              # [R, T+1]
        logq = torch.zeros(rows, dtype=torch.float64, device=h0.device)
        rec: dict[str, list[torch.Tensor]] = {k: [] for k in ("actions", "hazard", "stage_logits", "value", "reward", "state", "latent")}
        c = enc.ent_tokens.shape[1]
        for k in range(1, horizon_k + 1):
            # 1. candidate techniques: top-B of π(tech | s_{k−1})                        [R, B]
            log_pi = self.tech_log_probs(h)
            lp_top, cand = torch.topk(log_pi, n_cand, dim=-1)
            # 2. a target per candidate from π(tgt | s, tech) (Gumbel-max, pre-drawn uniforms)   [R, B]
            log_ptr = self.pointer_log_probs(enc, row, h, cand)                          # [R, B, C+1]
            slot = gumbel_argmax(log_ptr, u_tgt[k - 1].reshape(rows, n_cand, c + 1))
            # 3. one-step lookahead of every candidate through the stack                  [R·B, 1, dim]
            x = self._action_input(enc, row.repeat_interleave(n_cand), cand.reshape(-1), slot.reshape(-1))
            rep = torch.arange(rows, device=h.device).repeat_interleave(n_cand)
            t_k = torch.full((rows * n_cand, 1), k * cfg.window_seconds, dtype=torch.float64, device=h.device)
            h_c, kv_new = self.stack.step(x.unsqueeze(1), t_k, [(kk[rep], vv[rep]) for kk, vv in kvs], keys[rep])
            h_c = h_c[:, 0].reshape(rows, n_cand, dim)
            heads = self.step_heads(h_c)
            expo = self.exposure_of(enc, row, heads["hyp"], exposure)                    # [R, B]
            reward = heads["reward_task"].float() - cfg.exposure_cost * expo.float()
            q = reward + cfg.gamma * heads["value"].float()
            # 4. MPPI re-weighting and the draw                                           [R]
            log_tilde = mppi_log_weights(lp_top, q, eta)
            choice = gumbel_argmax(log_tilde, u_sel[k - 1].reshape(rows, n_cand))
            pick = torch.arange(rows, device=h.device)
            chosen_slot = slot[pick, choice]
            logq += (log_tilde[pick, choice] + log_ptr[pick, choice, chosen_slot]).double()
            # 5. advance the cache with the chosen candidate's state
            h = h_c[pick, choice]
            sel = pick * n_cand + choice
            kvs = [(torch.cat([kk, kn[sel]], dim=2), torch.cat([vv, vn[sel]], dim=2)) for (kk, vv), (kn, vn) in zip(kvs, kv_new, strict=True)]
            keys = torch.cat([keys, torch.ones(rows, 1, dtype=torch.bool, device=h.device)], dim=1)
            # 6. record (target as window entity index, −1 = no target)
            ent = torch.where(chosen_slot < c, enc.gathered.entity_index[row, chosen_slot.clamp(max=c - 1)],
                              torch.full_like(chosen_slot, -1))
            rec["actions"].append(torch.stack([cand[pick, choice], ent], dim=-1))
            for key, src in (("hazard", heads["hazard"]), ("stage_logits", heads["stage_logits"]), ("value", heads["value"]),
                             ("latent", heads["latent"])):
                rec[key].append(src[pick, choice])
            rec["reward"].append(reward[pick, choice])
            rec["state"].append(h)
        out = {key: torch.stack(v, dim=1).reshape(bt, n_c, horizon_k, *v[0].shape[1:]) for key, v in rec.items()}
        out["logq"] = logq.reshape(bt, n_c)
        return out

    # ================================================================== contract conversion
    def to_bundle(self, out: ForecastOut, b: int, m: int, *, compute: ComputeRecord, stage_names: Sequence[str],
                  driving_features: Sequence[DrivingFeature], entity_names: Sequence[str] | None = None,
                  max_paths: int | None = None) -> ForecastBundle:
        """One trigger's `ForecastOut` → `roles.contracts.ForecastBundle` (validates its invariants).

        stage_names: the model's stage axis as `vocab.STAGES` names (each must be an `AttackStage` value).
        entity_names: window entity names for the paths (default "entity:<index>"; "-" = no target).
        Paths: the distinct routes ordered by weight (at most `max_paths`, default all, ≤ N by D-46).
        """
        if compute.horizon_k != out.horizon_k:
            raise ValueError("compute.horizon_k must equal the forecast's horizon")
        stages = tuple(AttackStage(n) for n in stage_names)
        p = out.p_inf[b, m].double().clamp(0.0, 1.0)
        p = torch.cummax(p, dim=0).values                                       # exact guard, see routes.py
        rows = out.stage[b, m].double()
        rows = rows / rows.sum(-1, keepdim=True)
        band = out.p_inf_band[b, m].double().clamp(0.0, 1.0)
        hz = mixture_hazard(p)                                                  # hazard of the route mixture
        w = out.route_weight[b, m].double()
        order = [int(i) for i in torch.argsort(w, descending=True) if float(w[i]) > 0.0]   # distinct routes only
        if max_paths is not None:
            order = order[:max_paths]
        paths = []
        for n in order:
            acts = out.route_actions[b, m, n]                                    # [K, 2] (technique, entity)
            # Per-route stage per step: the stage head re-applied to the stored imagined states (the same
            # computation as during imagination; ForecastOut keeps only the route mixture of stages).
            with torch.no_grad():
                st = self.stage_head(self.norm(out.step_state[b, m, n].to(self.stage_head.weight.dtype))).argmax(-1)
            stage_seq = tuple(stages[int(i)] for i in st)
            ents = tuple("-" if int(e) < 0 else (entity_names[int(e)] if entity_names is not None else f"entity:{int(e)}")
                         for e in acts[:, 1])
            paths.append(ImaginedPath(probability=float(w[n].clamp(0.0, 1.0)), stages=stage_seq, entities=ents))
        return ForecastBundle(
            p_inf=tuple(float(x) for x in p),
            stage_probs=tuple(tuple(float(x) for x in r) for r in rows),
            stages=stages,
            top_paths=tuple(paths),
            driving_features=tuple(driving_features),
            compute=compute,
            hazard=tuple(float(x) for x in hz),
            p_inf_interval=tuple((float(lo), float(hi)) for lo, hi in band),
        )
