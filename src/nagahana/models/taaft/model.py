"""TAAFT: Topological Anti-Adversary Foundation Transformer — the analyser that writes Imagination.

"now coming to the TAAFT, it is basically the crux of this model being adversary foundation model,
cuz here's where everything converges that others fail over" [A-19].

Purpose and data flow (build-spec §1, §2.7; ADR-0007, ADR-0008)
---------------------------------------------------------------
    Environment (TSTCT memory-stream K/V, refined states, causal gates)   ── read only (D-35) ──┐
    WindowBatch (positions, contacts, triggers)                                                 │
                                                                                                ▼
    for each trigger τ_m, in time order (m = 0 … M−1):
      positions/slots: V entity tokens (latest refined TSTCT state as of τ_m) + G adversary slots
      memory stream (1 pass)  ── its self-attention K/V = Imagination of trigger m (written)
      thinking stream (R passes, weight-tied, reads the memory K/V; D-43, D-44)  ── context c
      y₀ = W_y c  ── energy descent on E_total (S steps, learned α, annealed noise) ── ŷ
      readouts(ŷ): compromise (floored), stage, malignity, trust, goal, type, next latent
    no loop back into TSTCT: beliefs never overwrite observed facts (ADR-0008).

Attention of block b at trigger τ_m (TAAFTBlock, blocks.py)
-----------------------------------------------------------
- **Self-attention = belief recursion.** Keys: the current trigger's tokens (bias from contacts as
  of τ: relation type self / hop-1 / hop-2 / none / entity↔slot, plus per-plane contact bias), and
  each token's **own** memory-stream K/V of the last M_im = `imagination_triggers` triggers with a
  log-Δτ bias. A designated share of heads (`self_rotary_fraction`, the last heads) rotates q and k
  by trigger time (D-49), so ⟨R(τ_m)q, R(τ_m′)k⟩ depends on τ_m − τ_m′ only; keys are rotated once
  when written, so Imagination entries stay valid (cache-friendly, as in TSTCT).
- **Cross-attention = reading the Environment.** TAAFT block b reads TSTCT block
  ⌊b·L_TSTCT/L_TAAFT⌋'s cached K/V with TAAFT's own query and output projections (AS-13). Only the
  head width must match (d_h = dim/heads in both); TAAFT's width may differ from TSTCT's because
  only TAAFT's q and o projections touch the cache. An entity token reads its own last
  `own_states` positions and the latest positions (≤ τ) of up to `neighbour_states` hop-1/hop-2
  neighbours (AS-37, AS-201); adversary slots read only the null key (they see entities through
  self-attention). Queries of the heads TSTCT rotates (temporal and causal: TSTCT's head order is
  spatial, temporal, causal) are rotated by τ_m, so the score depends on τ_m − t_j; a log-Δt bucket
  bias, a key-kind bias (own / hop-1 / hop-2) and a plane bias complete it.
- Every attention has the learned null key/value (no all-masked rows).

Energy and refinement (D-42; lenses.py, energy.py)
--------------------------------------------------
    E_total(c, y) = E_belief-trust + E_game + E_information + E_topology + E_temporal + E_causal + λ_phys Φ_phys
    ŷ = descend(E_total, y₀ = W_y c, steps = S, α = softplus(ρ), σ_i = σ₀(1 − i/S))
- `lens_energy[ℓ]` [B, M]: E_ℓ at ŷ; `lens_share[ℓ]` [B, M]: projection share of the last step
  (Σ_ℓ = 1, AS-211); `readouts["energy_rel/ℓ"]`: E_ℓ − reference_ℓ (ADR-0007: shares and values
  are read against each term's own reference, a bias-corrected EMA over real training windows, AS-210).
- ŷ is not fed back into the memory stream: Imagination K/V depend neither on R nor on S, so the
  cache is identical whatever the run-time budgets (the TSTCT argument of nn/loop.py, extended).
- Precision (D-54, AS-451): the transformer, the hypotheses y and the descent steps are float32;
  E_total, `lens_energy`, `lens_share`, `energy_trace`, `readouts["token_energy"]`, the
  `energy_rel/ℓ` readings and the posterior readouts (readouts.py) are float64.
- Thermodynamic readouts (D-56): at every trigger the model adds `readouts["thermo/total_energy"]`
  (E_total at y_hat) and, through `Readouts.thermodynamics`, the Gibbs ensembles of the active entity
  tokens and of the active adversary slots over their per-token energies (`thermo/entities/*`,
  `thermo/slots/*`, `thermo/occupation`), next to the per-entity stage ensembles computed in
  `Readouts.forward` (`thermo/stage/*`). Stacked over the triggers they are the energy, entropy and
  free-energy trajectories of the analysis; statphys reads their growth across calls. No parameter or
  buffer is involved, so the parameter count and the state dict are unchanged.

Long-term memory as extra cross keys (AS-220 … AS-222)
------------------------------------------------------
TAAFT owns X = `memory_probes` learned probe queries p_x and its own maps of the read:
    r_x = M_k(normalise(W_Q p_x))       (`memory.longterm.LongTermMemory.read`, the Titans memory)
    K_mem = RMSNorm_h(W_K^mem r),  V_mem = W_V^mem r            [B, H, X, d_h], shared by all blocks
Every token (entities and adversary slots, AS-221) reads them in its cross-attention, in the same
softmax as the Environment keys, with a learned per-head bias by token type. Contract (AS-222): the
memory state read at trigger τ_m holds only writes of triggers strictly before τ_m — either one state
for the call (written before its first trigger) or one per trigger (`memory_kv` [B, M, H, X, d_h]).

Imagination across calls (AS-223, AS-224)
-----------------------------------------
`past_imagination` / `past_time` / `past_mask` / `past_y` pre-fill the trigger history with carried
triggers (oldest first, one slot per trigger, right-aligned), so "the last M_im triggers" are
counted over earlier calls and this call alike: one call over triggers m₁ … m₂ equals two calls
with the carry (tested to 1e-5, also through the ImaginationStore). Carried entries are read only
when their trigger time is < τ (as-of guard), and E_time takes each token's latest belief among
the last M_im triggers before τ. Helpers: `models/taaft/imagination.py`.

One network, two readings (P-09, AS-16, AS-214)
-----------------------------------------------
`drop_context=True` (and, in training, each window with probability `context_dropout`) replaces the
context by a learned null context per token type and removes the evidence-derived structure (contact
weights → a learned uniform coherence weight, causal gates → 0, noise features → missing). Then
E(∅, y) is the marginal "is this normal" energy. `marginal_energy(y, …)` evaluates it for any y
without running the transformer (novelty and exposure for the Forecaster reward AS-17; Generator
acceptance AS-28).

No future leakage (tested, tests/test_taaft_model.py)
-----------------------------------------------------
Trigger m reads: positions ≤ τ_m (latest/own/neighbour indices are as of τ_m by the TriggerBatch
contract; noise and causal statistics are prefix sums read at the latest position; causal gates are
masked to t_j < t_i), contacts ≤ τ_m, and Imagination of triggers m′ < m only. The test perturbs
every later position, contact and trigger and checks earlier triggers are unchanged (R > 1).

Decisions: D-19, D-35, D-42, D-43, D-44, D-49. Assumptions: AS-06, AS-07, AS-13, AS-14, AS-15,
AS-16, AS-36, AS-37, AS-200 … AS-224 (docs/assumptions/taaft.md).
Held, untouched: D-12 (coupling with the policy/value heads), D-24 and D-26 (inside lenses), D-11b.

Extension points: lenses by name (`TAAFTConfig.lenses`, registry `LENSES`); `cross_layout`
"gathered" (inference layout) or "dense" (same function, masked; cheaper in memory when P is small
relative to N·T_k); readouts in `readouts.py`; stage-4 objectives in `objectives.py`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from nagahana.core.registry import Registry
from nagahana.governance.assumptions import assume
from nagahana.memory.longterm import LongTermMemory, NeuralMemoryState
from nagahana.models.batch import AnalysisOut, EnvironmentOut, WindowBatch
from nagahana.models.config.components import TAAFTConfig, TSTCTConfig
from nagahana.models.taaft.blocks import (
    MEMORY_BIAS,
    MEMORY_KV,
    PAST_ALLOWED,
    PAST_BIAS,
    PAST_KV,
    DenseMappedKV,
    LazyGatheredKV,
    block_map,
    stack_blocks,
)
from nagahana.models.taaft.energy import annealed_noise, descend, lens_step_shares
from nagahana.models.taaft.lenses import LENSES, DecoderLike, LensInputs, LensSpec, PhysicsLike, TotalEnergy
from nagahana.models.taaft.noise import NOISE_FEATURES, NoiseSpec, noise_features
from nagahana.models.taaft.readouts import Readouts
from nagahana.models.taaft.structure import (
    aggregate_causal_gates,
    causal_prefix,
    contacts_as_of,
    gather_rows,
    prev_positions,
    select_cross_keys,
)
from nagahana.models.vocab import NODE_KINDS
from nagahana.nn.attention import rotate_heads
from nagahana.nn.blocks import KV, AttnContext
from nagahana.nn.loop import TwoStreamStack
from nagahana.nn.norms import RMSNorm
from nagahana.nn.positional import LogDeltaBias, TimeRotary
from nagahana.physics.term import PhysicsTerm

ANALYSERS: Registry[type] = Registry("TAAFT")

#: Self-attention relation types (index into `rel_bias`).
REL_SELF, REL_HOP1, REL_HOP2, REL_NONE, REL_ENT_TO_SLOT, REL_SLOT_TO_ENT, REL_SLOT_TO_SLOT = range(7)


def _inv_softplus(x: float) -> float:
    return x + math.log(-math.expm1(-x))


@ANALYSERS.register("taaft", summary="trigger-sequential decoder over the Environment; energy of lens terms → Imagination")
class TAAFT(nn.Module):
    """The TAAFT analyser. See the module docstring.

    Parameters
    ----------
    cfg: TAAFT configuration. tstct: TSTCT configuration (head split, rotary periods, blocks).
    latent_dim: dz of the shared latent space. n_planes: relation planes (AS-01).
    n_stages: stage classes (15, AS-19).
    latent_split: (Dc, G_z, C) with Dc + G_z·C = latent_dim; None treats the latent as all
        continuous (no categorical groups). Integration passes (cvgae.cont_dim, disc_groups, disc_classes).
    memory_input_dim, memory_dim: the long-term memory's query width (`LongTermMemory(input_dim=…)`,
        = TSTCT width in NagaHana) and its width D (`MemoryConfig.longterm_dim`). Both given → TAAFT
        builds its probes and memory maps (AS-220) and accepts `memory_kv`; None → no memory read.
    """

    def __init__(
        self,
        cfg: TAAFTConfig,
        *,
        tstct: TSTCTConfig,
        latent_dim: int,
        n_planes: int,
        n_stages: int,
        latent_split: tuple[int, int, int] | None = None,
        memory_input_dim: int | None = None,
        memory_dim: int | None = None,
    ) -> None:
        super().__init__()
        # ---- compatibility with TSTCT's cache (AS-13): same heads, same head width.
        if cfg.heads != tstct.heads:
            raise ValueError(f"TAAFT heads ({cfg.heads}) must equal TSTCT heads ({tstct.heads}): cross-attention reads its K/V per head")
        if cfg.dim % cfg.heads or tstct.dim % tstct.heads or cfg.dim // cfg.heads != tstct.dim // tstct.heads:
            raise ValueError("TAAFT and TSTCT must have the same head width d_h = dim / heads (AS-13)")
        if cfg.cross_layout not in ("gathered", "dense"):
            raise ValueError("cross_layout must be 'gathered' or 'dense'")
        split = latent_split if latent_split is not None else (latent_dim, 0, 0)
        if split[0] + split[1] * split[2] != latent_dim:
            raise ValueError(f"latent_split {split} does not add up to latent_dim {latent_dim}")
        assume("AS-13", by=__name__)
        assume("AS-14", by=__name__)
        self.cfg, self.tstct_cfg = cfg, tstct
        self.latent_dim, self.latent_split, self.n_planes, self.n_stages = latent_dim, split, n_planes, n_stages
        d, h = cfg.dim, cfg.heads
        self.head_dim = d // h

        # ---- position (token) initialisation (D-49: kinds and time buckets, never indices)
        self.w_in = nn.Linear(tstct.dim, d, bias=False)                  # refined TSTCT state → TAAFT width
        self.kind_emb = nn.Embedding(len(NODE_KINDS), d)
        self.internal_emb = nn.Embedding(2, d)
        self.age_emb = LogDeltaBias(d, n_buckets=cfg.delta_buckets)      # age of the latest state, by log bucket
        self.slot_emb = nn.Parameter(torch.randn(cfg.adversary_slots, d) * 0.02)   # D-49 slot embeddings
        self.hidden_emb = nn.Parameter(torch.randn(d) * 0.02)            # masked-entity objective (stage 4)

        # ---- attention biases and time rotation
        self.rel_bias = nn.Parameter(torch.zeros(7, h))
        self.self_plane_bias = nn.Parameter(torch.zeros(n_planes, h))
        self.past_dt_bias = LogDeltaBias(h, n_buckets=cfg.delta_buckets)
        self.cross_kind_bias = nn.Parameter(torch.zeros(3, h))
        self.cross_plane_bias = nn.Parameter(torch.zeros(n_planes, h))
        self.cross_dt_bias = LogDeltaBias(h, n_buckets=cfg.delta_buckets)
        self.rotary = TimeRotary(self.head_dim, p_min=tstct.rotary_p_min, p_max=tstct.rotary_p_max)
        n_rot = int(round(h * cfg.self_rotary_fraction))
        self.self_rot_heads: torch.Tensor
        self.cross_rot_heads: torch.Tensor
        self.register_buffer("self_rot_heads", torch.arange(h) >= h - n_rot, persistent=False)
        self.register_buffer("cross_rot_heads", torch.arange(h) >= tstct.spatial_heads, persistent=False)

        # ---- the weight-tied block stack (two-stream loop, AS-06)
        self.block_map = block_map(cfg.blocks, tstct.blocks)
        self.stack = TwoStreamStack(stack_blocks(d, h, cfg.blocks, cfg.mlp_hidden))
        self.final_norm = RMSNorm(d)

        # ---- hypotheses, energy, readouts
        self.null_context = nn.Parameter(torch.randn(2, d) * 0.02)       # [entity, slot] (P-09 reading)
        self.w_y = nn.Linear(d, cfg.d_hyp)                                # amortised guess y₀ = W_y c
        self.readouts = Readouts(cfg, n_stages=n_stages, latent_split=split)
        self.spec = LensSpec(cfg=cfg, d_ctx=d, n_planes=n_planes)
        self.energy: TotalEnergy = LENSES.build("energy", self.spec, cfg.lenses)
        self.step_raw = nn.Parameter(torch.tensor(_inv_softplus(cfg.descent_step_init)))
        n_terms = len(self.energy.names)
        self.energy_reference: torch.Tensor
        self.reference_updates: torch.Tensor
        self.register_buffer("energy_reference", torch.zeros(n_terms))
        self.register_buffer("reference_updates", torch.zeros(()))
        self.noise_spec = NoiseSpec(cfg.noise_frequencies, cfg.noise_period_min, cfg.noise_period_max,
                                    cfg.noise_scales, cfg.noise_min_events, cfg.noise_min_blocks)

        # ---- long-term memory read as extra cross keys (AS-220): X learned probes, TAAFT's own K/V maps.
        self.has_memory = memory_input_dim is not None and memory_dim is not None
        if self.has_memory:
            assert memory_input_dim is not None and memory_dim is not None
            self.memory_probes = nn.Parameter(torch.randn(cfg.memory_probes, memory_input_dim) * 0.02)  # [X, d_in]
            self.memory_k = nn.Linear(memory_dim, d, bias=False)                  # D → H·d_h
            self.memory_v = nn.Linear(memory_dim, d, bias=False)
            self.memory_k_norm = RMSNorm(self.head_dim)                            # QK-norm, as every TAAFT key
            self.memory_bias = nn.Parameter(torch.zeros(2, h))                     # [entity, slot] per head (AS-221)

    # ================================================================== long-term memory (AS-220, AS-222)
    def read_longterm(self, memory: LongTermMemory, state: NeuralMemoryState) -> KV:
        """K, V [B, H, X, d_h] of the long-term memory as seen by TAAFT's X probe queries.

        r_x = M(normalise(W_Q p_x)) (`LongTermMemory.read`), K = RMSNorm_h(W_K^mem r), V = W_V^mem r.
        `state` must contain only writes of triggers strictly before the trigger(s) that read it
        (AS-222); see `forward(memory_kv=…)`. Differentiable in the probes, TAAFT's maps and the memory.
        """
        if not self.has_memory:
            raise ValueError("this TAAFT was built without memory_input_dim / memory_dim (no long-term read)")
        assume("AS-220", by=__name__)
        b = state.w1.shape[0]
        r = memory.read(state, self.memory_probes[None].expand(b, -1, -1).to(state.w1.dtype))   # [B, X, D]
        h, dh = self.cfg.heads, self.head_dim
        k = self.memory_k(r.to(self.memory_k.weight.dtype)).view(b, -1, h, dh).transpose(1, 2)  # [B, H, X, d_h]
        v = self.memory_v(r.to(self.memory_v.weight.dtype)).view(b, -1, h, dh).transpose(1, 2)
        return self.memory_k_norm(k), v

    def read_longterm_per_trigger(self, memory: LongTermMemory, states: list[NeuralMemoryState]) -> KV:
        """Per-trigger memory K/V [B, M, H, X, d_h]: states[m] = memory written by triggers before τ_m."""
        kvs = [self.read_longterm(memory, s) for s in states]
        return torch.stack([k for k, _ in kvs], dim=1), torch.stack([v for _, v in kvs], dim=1)

    # ================================================================== helpers
    @property
    def step_size(self) -> torch.Tensor:
        """α = softplus(ρ) > 0 (learned, AS-16)."""
        return F.softplus(self.step_raw)

    def _dropped(self, b: int, drop_context: bool, generator: torch.Generator | None, device: torch.device) -> torch.Tensor:
        # Which windows use the null-context reading (P-09): all if asked, else context dropout in training.
        if drop_context:
            return torch.ones(b, dtype=torch.bool, device=device)
        if self.training and self.cfg.context_dropout > 0:
            assume("AS-16", by=__name__)
            return (torch.rand(b, generator=generator) < self.cfg.context_dropout).to(device)
        return torch.zeros(b, dtype=torch.bool, device=device)

    def _null_context(self, b: int, v: int) -> torch.Tensor:
        g = self.cfg.adversary_slots
        return torch.cat([self.null_context[0].expand(b, v, -1), self.null_context[1].expand(b, g, -1)], dim=1)

    def _init_tokens(self, env: EnvironmentOut, window: WindowBatch, latest: torch.Tensor, active: torch.Tensor,
                     hidden: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        """x₀ [B, N, d]: entity tokens from the latest refined state as of τ, then adversary slots."""
        b, v = latest.shape
        x = self.w_in(gather_rows(env.refined, latest))                          # [B, V, d]
        age = (tau[:, None] - gather_rows(window.positions.time, latest)).clamp_min(0.0)
        x = x + self.age_emb(age).to(x.dtype)
        # Hidden entities (masked-entity objective): their states are replaced by a learned vector.
        x = torch.where(hidden[..., None], self.hidden_emb.expand_as(x), x)
        kind = window.entity_kind.clamp(0, len(NODE_KINDS) - 1)
        x = x + self.kind_emb(kind) + self.internal_emb(window.entity_internal.long())
        x = x * active[..., None].to(x.dtype)                                    # inactive tokens: zeros
        slots = self.slot_emb[None].expand(b, -1, -1)
        return torch.cat([x, slots], dim=1)

    @staticmethod
    def _relations(hop1: torch.Tensor, hop2: torch.Tensor, g: int) -> torch.Tensor:
        """Relation type of every (query, key) pair [B, N, N] (long), for the self-attention bias."""
        b, v, _ = hop1.shape
        n = v + g
        rel = torch.full((b, n, n), REL_SLOT_TO_SLOT, dtype=torch.long, device=hop1.device)
        ee = torch.full((b, v, v), REL_NONE, dtype=torch.long, device=hop1.device)
        ee = torch.where(hop2, torch.full_like(ee, REL_HOP2), ee)
        ee = torch.where(hop1, torch.full_like(ee, REL_HOP1), ee)
        rel[:, :v, :v] = ee
        rel[:, :v, v:] = REL_ENT_TO_SLOT
        rel[:, v:, :v] = REL_SLOT_TO_ENT
        idx = torch.arange(n, device=hop1.device)
        rel[:, idx, idx] = REL_SELF
        return rel

    # ================================================================== attention context per trigger
    def _attention_context(
        self,
        env: EnvironmentOut,
        window: WindowBatch,
        prev: torch.Tensor,
        m: int,
        tau: torch.Tensor,
        latest: torch.Tensor,
        active: torch.Tensor,
        readable: torch.Tensor,
        token_mask: torch.Tensor,
        hop1: torch.Tensor,
        hop2: torch.Tensor,
        planes: torch.Tensor,
        hist_kv: list[list[KV]],
        hist_mask: list[torch.Tensor],
        hist_time: list[torch.Tensor],
        need_weights: bool,
        memory_kv: KV | None = None,
    ) -> tuple[AttnContext, torch.Tensor]:
        """Masks, biases, rotary hooks, Imagination past and Environment keys of trigger m.

        `m` is the *global* trigger index into the history (carried triggers first, then this call's).
        """
        cfg, pos = self.cfg, window.positions
        b, v = latest.shape
        g = cfg.adversary_slots
        n = v + g
        p = pos.entity.shape[1]
        # ---- self-attention over current tokens: keys must be active; bias by relation and planes.
        allowed = token_mask[:, None, None, :]                                          # [B, 1, 1, N]
        bias = self.rel_bias[self._relations(hop1, hop2, g)].permute(0, 3, 1, 2)        # [B, H, N, N]
        ee_plane = torch.einsum("buvp,ph->bhuv", planes.to(bias.dtype), self.self_plane_bias)
        bias = bias + F.pad(ee_plane, (0, g, 0, g))
        # ---- time rotation by τ_m (self: designated heads; cross: TSTCT's temporal + causal heads).
        cos, sin = self.rotary.angles(tau)                                              # [B, d_h]
        cos, sin = cos[:, None, :], sin[:, None, :]                                     # [B, 1, d_h]
        self_rot, cross_rot = self.self_rot_heads, self.cross_rot_heads

        def q_hook(x: torch.Tensor) -> torch.Tensor:
            return rotate_heads(x, cos, sin, self_rot)

        def cross_q_hook(x: torch.Tensor) -> torch.Tensor:
            return rotate_heads(x, cos, sin, cross_rot)

        # ---- belief recursion: own Imagination K/V of the last M_im triggers (global m' = m−1 … m−M_im),
        # carried ones (earlier calls) and this call's alike. As-of guard (AS-223): an entry is read only
        # if its trigger time is strictly before τ, whatever the caller passed.
        extras: dict[str, Any] = {}
        m_im = cfg.imagination_triggers
        past_ids = [m - 1 - s for s in range(m_im) if m - 1 - s >= 0]
        if past_ids:
            n_blocks = len(self.stack.blocks)
            past_kv: list[KV] = []
            for blk in range(n_blocks):
                ks = torch.stack([hist_kv[mp][blk][0] for mp in past_ids], dim=3)       # [B, H, N, M', d_h]
                vs = torch.stack([hist_kv[mp][blk][1] for mp in past_ids], dim=3)
                past_kv.append((ks, vs))
            extras[PAST_KV] = past_kv
            t_past = torch.stack([hist_time[mp].to(torch.float64) for mp in past_ids], dim=-1)   # [B, M']
            before = t_past < tau[:, None]
            allowed_p = torch.stack([hist_mask[mp] for mp in past_ids], dim=-1) & before[:, None, :]
            extras[PAST_ALLOWED] = allowed_p[:, None]                                    # [B, 1, N, M']
            dt_p = (tau[:, None] - t_past).clamp_min(0.0)
            extras[PAST_BIAS] = self.past_dt_bias(dt_p).permute(0, 2, 1)[:, :, None, :]  # [B, H, 1, M']
            extras["past_triggers"] = past_ids
        # ---- long-term memory keys (AS-220): every token reads them, bias by token type (AS-221).
        if memory_kv is not None:
            extras[MEMORY_KV] = memory_kv
            tok_type = torch.cat([torch.zeros(v, dtype=torch.long), torch.ones(g, dtype=torch.long)]).to(tau.device)
            extras[MEMORY_BIAS] = self.memory_bias[tok_type].t()[None, :, :, None]       # [1, H, N, 1]
        # ---- Environment keys per entity token (own chain + neighbours' latest), slots none.
        idx, kind, nbr = select_cross_keys(latest, prev, pos.time, hop1, hop2, active, readable,
                                           own_states=cfg.own_states, neighbour_states=cfg.neighbour_states)
        tk = idx.shape[-1]
        idx = torch.cat([idx, idx.new_full((b, g, tk), -1)], dim=1)                     # [B, N, T_k]
        kind = torch.cat([kind, kind.new_zeros((b, g, tk))], dim=1)
        nbr = torch.cat([nbr, nbr.new_full((b, g, tk), -1)], dim=1)
        valid = idx >= 0
        dt = (tau[:, None, None] - gather_rows(pos.time, idx)).clamp_min(0.0)           # [B, N, T_k] float64
        cb = self.cross_kind_bias[kind] + self.cross_dt_bias(dt).to(self.cross_kind_bias.dtype)   # [B, N, T_k, H]
        # Plane bias for hop-1 neighbours: planes[b, v, u, :] with u = neighbour entity.
        np_ = planes.shape[-1]
        pl = planes.gather(2, nbr[:, :v].clamp_min(0)[..., None].expand(b, v, tk, np_))   # [B, V, T_k, n_planes]
        pl = pl & (kind[:, :v] == 1)[..., None] & valid[:, :v, :, None]
        pl = F.pad(pl.to(cb.dtype), (0, 0, 0, 0, 0, g))                                 # [B, N, T_k, n_planes]
        cb = cb + torch.einsum("bntp,ph->bnth", pl, self.cross_plane_bias)
        cross_bias = cb.permute(0, 3, 1, 2)                                             # [B, H, N, T_k]
        cross_allowed = valid[:, None]                                                  # [B, 1, N, T_k]
        if cfg.cross_layout == "gathered":
            ckv: Any = LazyGatheredKV(env.kv, self.block_map, idx)
            c_allowed, c_bias, gathered = cross_allowed, cross_bias, True
        else:
            # Dense layout: the same key sets scattered into the full position axis (one dummy column).
            col = torch.where(valid, idx, torch.full_like(idx, p))                      # [B, N, T_k]
            c_allowed = torch.zeros(b, 1, n, p + 1, dtype=torch.bool, device=idx.device)
            c_allowed.scatter_(3, col[:, None], valid[:, None])
            c_allowed = c_allowed[..., :p]
            h = cross_bias.shape[1]
            c_bias = cross_bias.new_zeros(b, h, n, p + 1)
            c_bias.scatter_(3, col[:, None].expand(b, h, n, tk), cross_bias)
            c_bias = c_bias[..., :p]
            ckv = DenseMappedKV(env.kv, self.block_map)
            gathered = False
        ctx = AttnContext(
            q_hook=q_hook, k_hook=q_hook, allowed=allowed, bias=bias, gathered=False,
            cross_kv=ckv, cross_q_hook=cross_q_hook, cross_allowed=c_allowed, cross_bias=c_bias,
            cross_gathered=gathered, need_weights=need_weights, extras=extras,
        )
        return ctx, idx

    def _previous_belief(
        self,
        hist_y: list[torch.Tensor | None],
        hist_mask: list[torch.Tensor],
        hist_time: list[torch.Tensor],
        gi: int,
        tau: torch.Tensor,
        token_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Each token's refined ŷ at its latest trigger among the last M_im with time < τ (AS-207, AS-224).

        Returns (y_prev [B, N, d_y], prev_time float64 [B, N], prev_ok bool [B, N]).
        """
        b, n = token_mask.shape
        dev = token_mask.device
        y_prev = torch.zeros(b, n, self.cfg.d_hyp, device=dev)
        prev_time = torch.zeros(b, n, dtype=torch.float64, device=dev)
        found = torch.zeros(b, n, dtype=torch.bool, device=dev)
        for j in range(gi - 1, max(-1, gi - 1 - self.cfg.imagination_triggers), -1):   # newest first
            yj = hist_y[j]
            if yj is None:
                continue
            take = hist_mask[j] & (hist_time[j] < tau)[:, None] & ~found
            y_prev = torch.where(take[..., None], yj.to(y_prev.dtype), y_prev)
            prev_time = torch.where(take, hist_time[j][:, None].to(torch.float64), prev_time)
            found = found | take
        return y_prev, prev_time, found & token_mask

    # ================================================================== forward
    def forward(
        self,
        env: EnvironmentOut,
        window: WindowBatch,
        *,
        passes: int,
        descent_steps: int,
        physics: PhysicsTerm | PhysicsLike | None = None,
        decoder: DecoderLike | None = None,
        create_graph: bool = False,
        need_weights: bool = False,
        generator: torch.Generator | None = None,
        drop_context: bool = False,
        hidden_entities: torch.Tensor | None = None,
        memory_kv: KV | None = None,
        past_imagination: Sequence[KV] | None = None,
        past_time: torch.Tensor | None = None,
        past_mask: torch.Tensor | None = None,
        past_y: torch.Tensor | None = None,
    ) -> AnalysisOut:
        """Analyse the triggers of a window batch. See the module docstring.

        passes: R thinking passes (≥ 1, D-44). descent_steps: S (≥ 0, D-44). physics + decoder: the
        physics term on decoded beliefs (0 when either is None). create_graph: keep the unrolled
        descent differentiable (stage-4 EBT training, AS-16). need_weights: attention weights into
        `readouts["attention/…"]`. generator: context dropout and descent noise (noise is used in
        eval only when a generator is given). drop_context: the null-context reading E(∅, y).
        hidden_entities: bool [B, V] entities whose states are hidden (masked-entity objective).

        Long-term memory (AS-220, AS-222) — `memory_kv` from `read_longterm`:
          (K, V) [B, H, X, d_h]: one memory state for every trigger of the call; it must contain only
            writes of triggers strictly before the call's first trigger;
          (K, V) [B, M, H, X, d_h]: one state per trigger; entry m must contain only writes of triggers
            strictly before τ_m (never τ_m's own write).
        Cross-call Imagination (AS-223, AS-224) — carried triggers, oldest first, right-aligned per
        window (padding slots at the front, mask False), aligned with this call's token axis
        (V entity tokens of this window's entity table, then G adversary slots):
          past_imagination: per block (K, V) [B, H, N, M_c, d_h] (memory-stream K/V of those triggers);
          past_time: float64 [B, M_c] their trigger times relative to *this* window's origin (< τ_0);
          past_mask: bool [B, N, M_c] token present at that trigger;
          past_y: [B, N, M_c, d_y] or None, the refined ŷ there (needed for E_time continuity).
        Build these with `models.taaft.imagination.past_from_analysis` / `past_from_store`.
        """
        cfg = self.cfg
        assume("AS-450", by=__name__)            # D-54: posterior readouts float64 (readouts.py)
        assume("AS-451", by=__name__)            # D-54: energies reduced in float64, descent on y in float32
        pos, trig = window.positions, window.triggers
        b, _ = pos.entity.shape
        m_tr, v = trig.entity_latest.shape[1], trig.entity_latest.shape[2]
        g = cfg.adversary_slots
        n = v + g
        dev = env.refined.device
        hidden = hidden_entities.to(torch.bool) if hidden_entities is not None else torch.zeros(b, v, dtype=torch.bool, device=dev)

        # ---- window-level as-of statistics (computed once; each trigger reads its prefix).
        prev = prev_positions(pos.next_index)                                           # [B, P]
        nf = noise_features(pos.entity, pos.time, pos.mask, trig.entity_latest, self.noise_spec)
        noise_valid = nf.valid & ~hidden[:, None, :, None]
        noise_vals = torch.where(noise_valid, nf.values, torch.zeros_like(nf.values))
        cmass, ccount = causal_prefix(env.causal_gate, pos.entity, pos.time, pos.mask, v)
        dropped = self._dropped(b, drop_context, generator, dev)
        alpha = self.step_size
        use_noise = self.training or generator is not None
        sigma = annealed_noise(cfg.descent_noise if use_noise else 0.0, descent_steps)

        # ---- long-term memory K/V: one state for the call, or one per trigger (AS-222).
        if memory_kv is not None:
            assume("AS-222", by=__name__)
            if not self.has_memory:
                raise ValueError("memory_kv given but this TAAFT has no memory maps (build with memory dims)")
            if memory_kv[0].dim() == 5 and memory_kv[0].shape[1] != m_tr:
                raise ValueError(f"per-trigger memory_kv needs M = {m_tr} entries, got {memory_kv[0].shape[1]}")
            if memory_kv[0].dim() not in (4, 5) or memory_kv[0].shape[0] != b:
                raise ValueError("memory_kv must be (K, V) [B, H, X, d_h] or [B, M, H, X, d_h]")

        # ---- state carried across triggers: Imagination of earlier calls (prefill) then of this call.
        hist_kv: list[list[KV]] = []
        hist_mask: list[torch.Tensor] = []
        hist_time: list[torch.Tensor] = []
        hist_y: list[torch.Tensor | None] = []                                          # ŷ per trigger (E_time)
        m_c = 0
        if past_imagination is not None:
            assume("AS-223", by=__name__)
            if past_time is None or past_mask is None:
                raise ValueError("past_imagination needs past_time and past_mask")
            if len(past_imagination) != len(self.stack.blocks):
                raise ValueError(f"past_imagination needs K/V for {len(self.stack.blocks)} blocks")
            m_c = int(past_time.shape[1])
            if past_mask.shape != (b, n, m_c):
                raise ValueError(f"past_mask must be [B, N, M_c] = {(b, n, m_c)}")
            for j in range(m_c):
                hist_kv.append([(k[:, :, :, j], vv[:, :, :, j]) for k, vv in past_imagination])
                hist_mask.append(past_mask[:, :, j].to(torch.bool))
                hist_time.append(past_time[:, j].to(torch.float64))
                hist_y.append(None if past_y is None else past_y[:, :, j].detach().float())

        outs: dict[str, list[torch.Tensor]] = {k: [] for k in ("context", "token_mask", "y0", "y")}
        lens_e: dict[str, list[torch.Tensor]] = {nm: [] for nm in self.energy.names}
        lens_s: dict[str, list[torch.Tensor]] = {nm: [] for nm in self.energy.names}
        ro_lists: dict[str, list[torch.Tensor]] = {}
        trace_sum = torch.zeros(descent_steps + 1, dtype=torch.float64)
        n_valid = 0

        for m in range(m_tr):
            gi = m_c + m                                                                # global trigger index
            tau = trig.time[:, m].to(torch.float64)                                     # [B]
            tvalid = trig.mask[:, m]                                                    # [B]
            latest = trig.entity_latest[:, m]                                           # [B, V]
            active = (latest >= 0) & window.entity_mask & tvalid[:, None]
            readable = active & ~hidden
            token_mask = torch.cat([active, tvalid[:, None].expand(b, g)], dim=1)       # [B, N]
            hop1, hop2, planes = contacts_as_of(window.contact1, window.contact2, window.contact_planes, tau)

            # ---- transformer: memory stream (writes Imagination) + R thinking passes.
            x0 = self._init_tokens(env, window, latest, active, hidden, tau)            # [B, N, d]
            mem_m: KV | None = None
            if memory_kv is not None:
                mem_m = (memory_kv[0][:, m], memory_kv[1][:, m]) if memory_kv[0].dim() == 5 else memory_kv
            ctx, cross_index = self._attention_context(env, window, prev, gi, tau, latest, active, readable, token_mask,
                                                       hop1, hop2, planes, hist_kv, hist_mask, hist_time, need_weights,
                                                       memory_kv=mem_m)
            _, kvs, _ = self.stack.memory(x0, ctx)
            h, aux = self.stack.think(x0, kvs, ctx, passes=passes,
                                      grad_passes=cfg.grad_passes if self.training else None)
            c = self.final_norm(h)                                                      # [B, N, d]
            c = torch.where(dropped[:, None, None], self._null_context(b, v).to(c.dtype), c)

            # ---- lens inputs as of τ_m (evidence removed on dropped windows, AS-214).
            keep = ~dropped[:, None, None]
            cause = aggregate_causal_gates(cmass, ccount, latest)                       # [B, u, v]
            cause = cause * (readable[:, :, None] & readable[:, None, :]).to(cause.dtype) * keep.to(cause.dtype)
            role = gather_rows(pos.role, latest)                                        # [B, V]
            upd = gather_rows(pos.update, latest)
            ent_planes = gather_rows(window.update_planes, upd) & (latest >= 0)[..., None]
            y_prev, prev_time, prev_ok = self._previous_belief(hist_y, hist_mask, hist_time, gi, tau, token_mask)
            inp = LensInputs(
                context=c, token_mask=token_mask, n_entities=v,
                hop1=hop1 & keep, hop2=hop2 & keep, planes=planes & keep[..., None], cause=cause,
                noise=noise_vals[:, m], noise_valid=noise_valid[:, m] & keep,
                y_prev=y_prev if bool(prev_ok.any()) else None, prev_valid=prev_ok,
                prev_dt=(tau[:, None] - prev_time).clamp_min(0.0), dropped=dropped,
                heads=self.readouts, entity_role=role, entity_planes=ent_planes,
                decoder=decoder, physics=physics,
            )
            preps = self.energy.prepare(inp)
            tv = tvalid.to(c.dtype)

            def e_total(y: torch.Tensor, inp: LensInputs = inp, preps: Any = preps, tv: torch.Tensor = tv) -> torch.Tensor:
                return self.energy(y, inp, preps)[0] * tv

            def e_terms(y: torch.Tensor, inp: LensInputs = inp, preps: Any = preps, tv: torch.Tensor = tv) -> dict[str, torch.Tensor]:
                return {nm: t.energy * tv for nm, t in self.energy(y, inp, preps)[1].items()}

            # ---- refinement: ŷ = descent on E_total from y₀ = W_y c.
            y0 = self.w_y(c)                                                            # [B, N, d_y]
            res = descend(e_total, y0, steps=descent_steps, step_size=alpha if create_graph else alpha.detach(),
                          noise=sigma, generator=generator, create_graph=create_graph)
            y_hat = res.y
            _, terms = self.energy(y_hat, inp, preps)
            shares = lens_step_shares(e_terms, res.y_last_input, alpha.detach())
            trace_sum += torch.tensor(res.trace, dtype=torch.float64)
            n_valid += int(tvalid.sum())

            # ---- readouts and explanations of this trigger.
            ro = self.readouts(y_hat, n_entities=v, token_mask=token_mask)
            # Energy per token: float32 per-lens maps summed over lenses in float64 (D-54).
            tok_e = torch.zeros(b, n, device=dev, dtype=torch.float64)
            for nm, t in terms.items():
                lens_e[nm].append(t.energy.double() * tv.double())                    # [B] float64
                lens_s[nm].append(shares[nm])                                          # [B] float64
                if t.per_token is not None:
                    tok_e = tok_e + t.per_token.double()
                for ak, av in t.aux.items():
                    ro[f"{nm}/{ak}"] = av
            ro["token_energy"] = tok_e * token_mask.to(tok_e.dtype)
            # Thermodynamic readouts of the trigger (D-56, readouts.py): Gibbs ensembles of the entity and
            # slot tokens over their energies, and E_total at y_hat; float64, no parameters involved.
            ro.update(self.readouts.thermodynamics(ro["token_energy"], token_mask, n_entities=v))
            ro["thermo/total_energy"] = torch.stack([t.energy.double() for t in terms.values()]).sum(0) * tv.double()
            for fi, fname in enumerate(NOISE_FEATURES):
                ro[f"noise/{fname}"] = noise_vals[:, m, :, fi]
                ro[f"noise_valid/{fname}"] = noise_valid[:, m, :, fi]
            if need_weights:
                # Self weights are [current N | past M' | null]; pad the past part to M_im slots
                # (slot s = trigger m−1−s) so every trigger has the same width N + M_im + 1.
                ro["attention/cross_index"] = cross_index
                n_past = len(ctx.extras.get("past_triggers", []))
                for bi, a in enumerate(aux):
                    for ak, av in a.items():
                        if ak == "self_weights" and n_past < cfg.imagination_triggers:
                            cur, rest = av[..., : n + n_past], av[..., n + n_past :]
                            pad = av.new_zeros(*av.shape[:-1], cfg.imagination_triggers - n_past)
                            av = torch.cat([cur, pad, rest], dim=-1)
                        ro[f"attention/{ak}/{bi}"] = av
            for k, val in ro.items():
                ro_lists.setdefault(k, []).append(val)
            outs["context"].append(c)
            outs["token_mask"].append(token_mask)
            outs["y0"].append(y0)
            outs["y"].append(y_hat)

            # ---- carry Imagination forward (K/V written by the memory stream; ŷ for E_time).
            hist_kv.append(kvs)
            hist_mask.append(token_mask)
            hist_time.append(tau)
            hist_y.append(y_hat.detach())

        # ---- assemble [B, M, …] outputs.
        lens_energy = {nm: torch.stack(vals, dim=1) for nm, vals in lens_e.items()}
        lens_share = {nm: torch.stack(vals, dim=1) for nm, vals in lens_s.items()}
        readouts = {k: torch.stack(vals, dim=1) for k, vals in ro_lists.items()}
        if self.training:
            self._update_reference(lens_energy, trig.mask & ~dropped[:, None])
        for nm, rel in self.relative_energy(lens_energy).items():
            readouts[f"energy_rel/{nm}"] = rel
        n_blocks = len(self.stack.blocks)
        imagination = [
            (torch.cat([hist_kv[m_c + mm][blk][0] for mm in range(m_tr)], dim=2),
             torch.cat([hist_kv[m_c + mm][blk][1] for mm in range(m_tr)], dim=2))
            for blk in range(n_blocks)
        ] if m_tr else []
        trace = trace_sum / max(1, n_valid)                                             # [S+1] float64 (D-54)
        return AnalysisOut(
            context=torch.stack(outs["context"], dim=1), token_mask=torch.stack(outs["token_mask"], dim=1),
            imagination_kv=imagination, y0=torch.stack(outs["y0"], dim=1), y=torch.stack(outs["y"], dim=1),
            energy_trace=trace, lens_energy=lens_energy, lens_share=lens_share, readouts=readouts,
            passes=passes, descent_steps=descent_steps,
        )

    # ================================================================== energy references and readings
    @torch.no_grad()
    def _update_reference(self, lens_energy: dict[str, torch.Tensor], valid: torch.Tensor) -> None:
        """Bias-corrected EMA of each term's mean per-trigger energy over real windows (AS-210)."""
        if not bool(valid.any()):
            return
        mu = self.cfg.reference_momentum
        w = valid.to(torch.float32)
        cur = torch.stack([(lens_energy[nm].detach().float() * w).sum() / w.sum() for nm in self.energy.names])
        self.energy_reference.mul_(mu).add_((1.0 - mu) * cur)
        self.reference_updates.add_(1.0)

    def reference(self) -> dict[str, torch.Tensor]:
        """Each term's reference level (bias-corrected EMA; 0 before any training update)."""
        k = float(self.reference_updates)
        if k <= 0:
            return {nm: self.energy_reference.new_zeros(()) for nm in self.energy.names}
        corr = 1.0 - self.cfg.reference_momentum**k
        return {nm: self.energy_reference[i] / corr for i, nm in enumerate(self.energy.names)}

    def relative_energy(self, lens_energy: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """E_ℓ − reference_ℓ: per-lens energy read against the term's own reference (ADR-0007, AS-210)."""
        ref = self.reference()
        return {nm: e - ref[nm].to(e.dtype) for nm, e in lens_energy.items() if nm in ref}

    def marginal_energy(
        self,
        y: torch.Tensor,
        token_mask: torch.Tensor,
        *,
        n_entities: int,
        y_prev: torch.Tensor | None = None,
        prev_dt: torch.Tensor | None = None,
        decoder: DecoderLike | None = None,
        physics: PhysicsTerm | PhysicsLike | None = None,
        entity_role: torch.Tensor | None = None,
        entity_planes: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """E(∅, y): the null-context ("is this normal") energy of hypotheses y (P-09, AS-16, AS-214).

        y [B, M, N, d_y]; token_mask bool [B, M, N]; y_prev / prev_dt [B, M, N, d_y] / [B, M, N]
        (optional: the temporal term is 0 without them). For the physics term pass decoder, physics,
        entity_role [B, M, V] and entity_planes [B, M, V, n_planes]. Returns (E_total [B, M],
        {term: [B, M]}). Used for novelty / exposure (AS-17) and Generator acceptance (AS-28).
        """
        b, m_tr, n, d_y = y.shape
        v = n_entities
        bm = b * m_tr
        dev = y.device
        flat = y.reshape(bm, n, d_y)
        mask = token_mask.reshape(bm, n)
        no_pairs = torch.zeros(bm, v, v, dtype=torch.bool, device=dev)
        nf = len(NOISE_FEATURES)
        prev_valid = mask if y_prev is not None else torch.zeros_like(mask)
        inp = LensInputs(
            context=self._null_context(bm, v).to(flat.dtype), token_mask=mask, n_entities=v,
            hop1=no_pairs, hop2=no_pairs, planes=torch.zeros(bm, v, v, self.n_planes, dtype=torch.bool, device=dev),
            cause=torch.zeros(bm, v, v, device=dev), noise=torch.zeros(bm, v, nf, device=dev),
            noise_valid=torch.zeros(bm, v, nf, dtype=torch.bool, device=dev),
            y_prev=None if y_prev is None else y_prev.reshape(bm, n, d_y).detach(), prev_valid=prev_valid,
            prev_dt=(torch.zeros(bm, n, dtype=torch.float64, device=dev) if prev_dt is None
                     else prev_dt.reshape(bm, n).to(torch.float64)),
            dropped=torch.ones(bm, dtype=torch.bool, device=dev), heads=self.readouts,
            entity_role=(torch.full((bm, v), 3, dtype=torch.long, device=dev) if entity_role is None
                         else entity_role.reshape(bm, v)),
            entity_planes=(torch.zeros(bm, v, self.n_planes, dtype=torch.bool, device=dev) if entity_planes is None
                           else entity_planes.reshape(bm, v, -1)),
            decoder=decoder, physics=physics,
        )
        # E_total and the terms are float64 (D-54: lens reductions and the lens sum in float64).
        total, terms = self.energy(flat, inp, self.energy.prepare(inp))
        return total.reshape(b, m_tr), {nm: t.energy.reshape(b, m_tr) for nm, t in terms.items()}


def count_parameters(module: nn.Module) -> int:
    """Number of parameters (works on the meta device: no memory is allocated for them)."""
    return sum(p.numel() for p in module.parameters())
