"""TSTCT: Topological Spatio-Temporal Causal Transformer, the Environment builder (build-spec §2.5).

Role
----
"Next TSTCT will run by taking this state as input, to generate the spatio-temporal-causal kv cache;
this basically stores the state model & updates of it … which makes up the environment" [A-12].
With CVG-AE it is a *perceptor, not an analyser* [A-19]: it organises the latent entity states
across space (topology), time (each entity's own past) and lagged influence (cause → effect), and it
holds the world model's transition prior p(z_next | Environment).

Owner sources: [A-10], [A-12], [A-13], [A-14], [A-19]; diagram 03. Decisions: D-19 (name), D-35
(Environment access), D-43/D-44 (weight-tied looping, run-time R), D-49 (time, never index).
Assumptions: AS-05 (transition prior, DreamerV3 KL), AS-06 (two-stream loop), AS-07 (R sampling),
AS-09 (head split), AS-10 (Granger-style causal gates), AS-41 (2 positions per update), and the new
AS-150 … AS-161 of `docs/assumptions/tstct-memory.md` (D-51 carry: AS-159 … AS-161).

Maths
-----
Input. e⁰_j = W_in z_j  (z: CVG-AE posterior sample in training, posterior mean at inference, AS-05).

Blocks. `nn.blocks.SelfBlock` × L (pre-norm RMSNorm, QK-norm, SwiGLU; AS-32). For head h of group
G(h) ∈ {spatial, temporal, causal} (AS-09) and query i, key j:

    a_ij^h = ⟨R(t_i) q̂_i, R(t_j) k̂_j⟩/√d_h  +  β_h(i, j)  +  log n_j  +  M_h(i, j)

- R(t): continuous-time rotary on temporal and causal heads only (D-49, `nn.positional.TimeRotary`),
  so ⟨R(t_i)q, R(t_j)k⟩ = ⟨q, R(t_j − t_i)k⟩ depends on real elapsed time only. A key is rotated once,
  by its own time, when it is written; it stays valid in the cache forever.
- β_h, spatial heads:   b_hop[hop_ij] + Σ_p w_p·Π_p(i, j) + b_age[bucket(t_i − t_j)].
- β_h, temporal heads:  b_Δ[bucket(t_i − t_j)]  (learned per bucket, so recall need not decay with
  distance as plain RoPE's long-range decay does: build-spec §4b.1).
- β_h, causal heads:    b_Δ[bucket(t_i − t_j)] + log g_ij^h on the top-k candidates (below).
- log n_j: log of the number of states merged into slot j (0 in training; Environment merges, AS-11).
- M_h: 0 where the head's pattern allows j (masks.py), −∞ elsewhere; plus a learned null key.
- The bias tables are shared by all blocks (as T5 shares its relative-position bias across layers;
  Raffel et al., JMLR 2020, arXiv:1910.10683), which keeps the bias identical across blocks and passes.

Causal gate (AS-10, AS-152). For causal head h, candidate j of query i (masks.py):

    g_ij^h = σ( w_hᵀ SiLU( A_h u_i + B_h u_j + C_h φ(Δt_ij) ) + c_h ),   u = V⁰ = W_V⁰ RMSNorm⁰(e⁰)

i.e. an MLP on the concatenation [q^g_i ; k^g_j ; φ(Δt)] whose first layer is split by input. The
gate's query/key are dedicated projections of u, the block-0 memory-stream *values* (a learned
linear map of the normalised input states), not of a block's rotated q/k:
- one gate map per causal head serves every block and every pass (masks and biases identical across
  passes; EnvironmentOut.causal_gate is one map; TAAFT's E_cause reads one ḡ);
- it is origin-free (rotated block keys would make an MLP of them depend on absolute time phase);
- V⁰ of every state is already in the Environment cache, so stored and carried states (D-51) are gated
  without re-encoding and without extra storage, and since A, B are linear, the gate key of a merged
  slot (count-weighted mean V⁰) is exactly the count-weighted mean of its members' gate keys.
Selection: the top `causal_keys` candidates by g per (query, head) are kept and enter as log g;
the rest are masked. Hard top-k with differentiable values on the kept set (as in sparsely-gated
mixture-of-experts routing, Shazeer et al., ICLR 2017, arXiv:1701.06538): the kept gates receive
gradient through log g, all candidates receive the L1 penalty λ_gate·mean(g) (`gate_l1`). No
straight-through estimator is used. This is Granger-style predictive influence (Tank et al., TPAMI
2021, arXiv:1802.05842), never a claim of identified causal structure (AS-10).

Two-stream weight-tied loop (AS-06, `nn/loop.py`).
    memory stream (1 pass):    m = Stack(e⁰), caching (K_b, V_b) of every block   → Environment
    thinking stream (R passes): h⁰ = 0, h^{r+1} = Stack_read(h^r + e⁰)             → refined e = h^R
Why the cache is R-independent: the memory stream's K/V at position j depend only on e⁰ of positions
≤ j through an ordinary masked transformer; the thinking stream only *reads* them. So the cache is
the same function whatever R is, training (all positions in parallel) and inference (appended one by
one, `step`) compute exactly the same thing, and raising R at run time changes only the thinking
stream (tested: `test_tstct_equivalence.py`, `test_cache_is_independent_of_passes`).

Transition prior (the world model, AS-05). On the memory stream at position j, with the time to the
same entity's next state Δ_j as an input:

    [μ_p, log σ_p², ℓ_p] = MLP( RMSNorm(m_j) + φ_p(Δ_j) ),    p(z_next(j) | Environment ≤ t_j) = 𝒩(μ_p, σ_p²) × Cat(ℓ_p)

This is the problem statement's P(S_{t+1} | S_t), trained by the KL terms of build-spec §3.
The same head applied to the thinking stream (`refined_prior`) gives the R passes a stage-3
objective of their own: without it the thinking stream receives no gradient in stage 3 (the memory
stream's prior does not depend on R) and is frozen in stage 4, so "more passes" would be trained only
in stage 5. The memory-stream prior stays the cache-consistent world model; the refined prior asks
"does thinking longer predict the next state better", and is measurable per R (AS-159).

Training across windows (D-51; Transformer-XL, Dai et al., ACL 2019, arXiv:1901.02860). `forward`
takes an optional `CarriedEnvironment`: the earlier windows' memory-stream K/V (stop-gradient,
read-only), exported from one `EnvironmentStore` per window stream (`carry_out`), so the carried
memory has the store's volume invariance and capacity bound. The key axis becomes [C carried ;
P current]; the masks apply the same rules across the boundary (masks.py), the thinking stream reads
the same carried K/V, and carried keys are re-based from the store origin to the window origin by one
rotary rotation R(o_store − o_window) (rotations compose exactly). With a store that keeps everything,
windows w1, w2 with carry compute exactly what one dense window w1 + w2 computes for w2's positions
(tested: `tests/test_tstct_carry.py`). This is the training-time image of `step` reading the store.

Precision of time angles. Times are float64 seconds relative to the window (or store) origin. The
rotary reduces ω·t mod 2π in float64 before the float32 cast: for t up to 4 weeks (2.4·10⁶ s) and
the fastest ω = 2π/1 ms, ω·t ≈ 1.5·10¹⁰ rad, whose float64 rounding is ≈ 2·10⁻⁶ rad; the float64
resolution of t itself (≈ 5·10⁻¹⁰ s) adds ≈ 3·10⁻⁶ rad. That is larger than float32 rounding
(≈ 6·10⁻⁸) but changes a unit-scale attention logit by ≈ 10⁻⁵ at most, and only on the 1 ms
frequency pair; it grows linearly with store age (≈ 4·10⁻⁵ rad after one year), so long-lived stores
should re-base their origin yearly (an extension point). Float32 epoch seconds would have had a
resolution of 128 s, i.e. every frequency faster than a few minutes would be pure noise.

Invariants (tested)
-------------------
- No future leakage, for any R (tests/test_tstct_model.py::test_no_future_leak).
- Dense path ≡ cached incremental path (`step`), R ∈ {1, 3}, atol 1e-4 (tests/test_tstct_equivalence.py).
- Padded positions do not change real outputs; gradients reach every parameter.
- Windows with carry ≡ one dense window; carried slots never receive gradient (tests/test_tstct_carry.py).

Extension points
----------------
- Further spatial bias features (hyperedge kinds) add tables beside `hop_bias`/`plane_bias`.
- A soft (Gumbel/sparsemax) causal selection can replace `_top_k`.
- `step` builds key sets with Python loops over the new states; a vectorised/paged gather can
  replace `_key_sets` without changing the maths.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, cast

import torch
import torch.nn.functional as F
from torch import nn

from nagahana.core.errors import InvariantViolation
from nagahana.core.registry import Registry
from nagahana.core.roles import Role
from nagahana.governance.assumptions import assume
from nagahana.memory.access import Region
from nagahana.memory.environment import EnvironmentStore
from nagahana.memory.kvcache import CACHE_DTYPE, KVCacheMeta
from nagahana.models.batch import EnvironmentOut, LatentPrior, WindowBatch
from nagahana.models.config import NagaHanaConfig
from nagahana.models.config.components import MemoryConfig, TSTCTConfig
from nagahana.models.tstct.masks import CAUSAL, SPATIAL, TEMPORAL, CarriedKeys, build_dense_masks, head_groups
from nagahana.nn.attention import rotate_heads
from nagahana.nn.blocks import KV, AttnContext, SelfBlock
from nagahana.nn.loop import TwoStreamStack
from nagahana.nn.mlp import SwiGLU
from nagahana.nn.norms import RMSNorm
from nagahana.nn.numeric import PeriodicEmbedding
from nagahana.nn.positional import LogDeltaBias, TimeRotary

TRANSFORMERS: Registry[type] = Registry("TSTCT")

#: log1p(Δt/δ₀) for Δt = 1 year at δ₀ = 1 ms is ≈ 24.2; dividing the trend input by 20 keeps it O(1).
_LOG_TIME_SCALE = 20.0


# ===================================================================================== Δt encoding
class TimeDeltaEncoding(nn.Module):
    """φ(Δt): x̃ = log1p(Δt/δ₀) → periodic embedding (learned frequencies) + a linear trend (AS-153).

    The log compresses 1 ms … months into ≈ 0 … 25; periodic features resolve fine differences in
    x̃ (Gorishniy et al., NeurIPS 2022, arXiv:2203.05556); the trend term keeps a monotone direction.
    Δt is float64 seconds; negative Δt is clamped to 0 (masks exclude those pairs anyway).
    """

    def __init__(self, out_dim: int, *, n_frequencies: int, delta0: float) -> None:
        super().__init__()
        self.delta0 = delta0
        self.periodic = PeriodicEmbedding(1, n_frequencies, out_dim, sigma=0.5)
        self.trend = nn.Linear(1, out_dim, bias=False)

    def forward(self, dt: torch.Tensor) -> torch.Tensor:
        """Δt [...] (float64) → [..., out_dim]."""
        x = torch.log1p(dt.to(torch.float64).clamp_min(0.0) / self.delta0).to(self.trend.weight.dtype).unsqueeze(-1)
        return self.periodic(x)[..., 0, :] + self.trend(x / _LOG_TIME_SCALE)


# ===================================================================================== step contracts
@dataclass
class StepContext:
    """What the data pipeline / inference engine supplies for n new states (`TSTCT.step`).

    For new state k (entity e_k at time t_k, float64 seconds since the store origin), the entities v
    within two hops of e_k *as of t_k* (AS-156):

    neighbour_entity: long [n, N]: neighbour entity ids (store ids), −1 = padding.
    neighbour_hop: long [n, N]: 1 if C¹[e_k, v] ≤ t_k (direct contact), else 2 (C²[e_k, v] ≤ t_k).
        Only hop-1 neighbours are causal sources (contact by then, AS-10).
    neighbour_planes: bool [n, N, n_planes]: planes p with C¹_p[e_k, v] ≤ t_k.
    horizon_dt: float64 [n] or None: Δt ahead at which to evaluate the transition prior (the next
        state's gap is unknown at inference; the caller picks the horizon, e.g. one window).
    """

    neighbour_entity: torch.Tensor
    neighbour_hop: torch.Tensor
    neighbour_planes: torch.Tensor
    horizon_dt: torch.Tensor | None = None

    @classmethod
    def from_contacts(
        cls,
        entity: torch.Tensor,
        time: torch.Tensor,
        contact1: torch.Tensor,
        contact2: torch.Tensor,
        contact_planes: torch.Tensor,
        *,
        horizon_dt: torch.Tensor | None = None,
        tie_s: float,
    ) -> StepContext:
        """Build the context from contact matrices C¹, C² [V, V] and C¹_p [V, V, n_planes] (+inf = never).

        Entity ids index the matrices' rows. Neighbours are listed in ascending entity id. `tie_s`: times within
        it count as simultaneous (TSTCTConfig.time_tie_s, AS-160); `TSTCT.step_context` passes it.
        """
        c1, c2, cp = contact1.to(torch.float64), contact2.to(torch.float64), contact_planes.to(torch.float64)
        rows: list[list[int]] = []
        for e, t in zip(entity.tolist(), time.to(torch.float64).tolist(), strict=True):
            near = ((c1[e] <= t + tie_s) | (c2[e] <= t + tie_s)).clone()
            near[e] = False
            rows.append(torch.nonzero(near).flatten().tolist())
        width = max([1, *(len(r) for r in rows)])
        n, n_planes = len(rows), cp.shape[-1]
        ne = torch.full((n, width), -1, dtype=torch.long)
        nh = torch.zeros((n, width), dtype=torch.long)
        npl = torch.zeros((n, width, n_planes), dtype=torch.bool)
        for k, (r, e, t) in enumerate(zip(rows, entity.tolist(), time.to(torch.float64).tolist(), strict=True)):
            for c, v in enumerate(r):
                ne[k, c] = v
                nh[k, c] = 1 if float(c1[e, v]) <= t + tie_s else 2
                npl[k, c] = cp[e, v] <= t + tie_s
        return cls(neighbour_entity=ne, neighbour_hop=nh, neighbour_planes=npl, horizon_dt=horizon_dt)


@dataclass
class StepOut:
    """Result of `TSTCT.step` for n new states.

    memory, refined: [n, d] memory-stream and thinking-stream outputs; kv: per block (K, V) [n, H, d_h]
    written to the store (keys rotated); slots: long [n] store slot ids; prior: transition prior at
    `context.horizon_dt` (None if no horizon given); causal_gate: [n, H_c, T_c] gate values of the
    causal candidates (0 at padding); causal_entity: long [n, T_c] source entity of each candidate
    (−1 padding); causal_time: float64 [n, T_c] its time; causal_kept: bool [n, H_c, T_c] top-k
    selection; passes: R used; attention: per block {"memory_weights", "thinking_weights"} when requested.
    """

    memory: torch.Tensor
    refined: torch.Tensor
    kv: list[KV]
    slots: torch.Tensor
    prior: LatentPrior | None
    causal_gate: torch.Tensor
    causal_entity: torch.Tensor
    causal_time: torch.Tensor
    causal_kept: torch.Tensor
    passes: int
    attention: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class CarriedEnvironment:
    """The Environment of earlier windows, carried into the next window's dense pass (D-51).

    Exported from one `EnvironmentStore` per window stream (`TSTCT.carry_out` / `export_carry`), so it is
    bounded per entity by the store's cells (P4) and has its volume invariance (P3). All tensors are
    data (no gradient); `forward` detaches them again.

    k, v: per block [B, H, C, d_h]: memory-stream K/V; keys rotated relative to `origin` (the store
        origin), re-based to the window origin inside `forward`.
    origin: float64 [B] epoch seconds of the time zero of `time` and of the key rotation.
    time: float64 [B, C] slot time relative to `origin` (merged slots: count-weighted mean).
    key: long [B, C] stable entity key (the store's entity id), −1 = padding.
    count: float32 [B, C] states merged into the slot (log n is added to its logits).
    bucket: bool [B, C] cell slot (temporal and causal heads); latest: bool [B, C] the entity's
        latest-state register (spatial heads only, AS-154).
    mask: bool [B, C] real slots.
    entity: long [B, C] index in the window's entity table, set by `align` (None = not aligned yet).

    Entity mapping between windows (what the data pipeline must provide)
    --------------------------------------------------------------------
    1. A stable entity key per window entity (`entity_keys`, long [B, V], −1 for padding): the same
       machine (D-48) gets the same key in every window of a stream. Keys are the store's entity ids.
    2. Every carried entity that the next window should be able to read must be in that window's
       entity table (it may have no update in the window); `align` maps keys to indices, and slots of
       entities absent from the table get entity −1 and are masked.
    3. The window's contact matrices C¹, C², C¹_p must be cumulative over the stream: first contact
       ever, relative to this window's origin (negative before it), +inf for never, including pairs
       with carried entities. Hop and plane bits "as of t_i" for carried keys come from them.
    4. Windows of one stream arrive in time order, and every carried time ≤ the window's first time.
    """

    k: list[torch.Tensor]
    v: list[torch.Tensor]
    origin: torch.Tensor
    time: torch.Tensor
    key: torch.Tensor
    count: torch.Tensor
    bucket: torch.Tensor
    latest: torch.Tensor
    mask: torch.Tensor
    entity: torch.Tensor | None = None

    @property
    def num_slots(self) -> int:
        """C, the padded number of carried slots per window."""
        return int(self.key.shape[1])

    def align(self, entity_keys: torch.Tensor) -> CarriedEnvironment:
        """Map stable keys to the next window's entity indices: entity_keys long [B, V] (−1 = padding)."""
        ent = torch.full_like(self.key, -1)
        for b in range(self.key.shape[0]):
            index = {int(k): i for i, k in enumerate(entity_keys[b].tolist()) if k >= 0}
            for c, k in enumerate(self.key[b].tolist()):
                if k >= 0 and k in index:
                    ent[b, c] = index[k]
        return CarriedEnvironment(k=self.k, v=self.v, origin=self.origin, time=self.time, key=self.key,
                                  count=self.count, bucket=self.bucket, latest=self.latest, mask=self.mask, entity=ent)


@dataclass
class CarryPolicy:
    """What `carry_out` keeps and where (D-51).

    stores: one `EnvironmentStore` per batch row (window stream), persistent across that stream's
        windows; its MemoryConfig (cells, quota) is the bound on what is carried. Create with
        `TSTCT.new_store(..., origin=<first window origin>)`.
    entity_keys: long [B, V]: stable entity key of each entity of the window being written (−1 pad).
    """

    stores: Sequence[EnvironmentStore]
    entity_keys: torch.Tensor


@dataclass
class _KeySet:
    """Gathered key set of one head group for n queries (pool ids: < S store slots, ≥ S new states)."""

    pool: torch.Tensor      # long [n, T] (−1 padding)
    mask: torch.Tensor      # bool [n, T]
    dt: torch.Tensor        # float64 [n, T] t_query − t_key (0 at padding)
    count: torch.Tensor     # float32 [n, T] merged-state counts (1 at padding, so log n = 0)
    entity: torch.Tensor    # long [n, T] source entity (−1 padding)
    hop: torch.Tensor       # long [n, T] (spatial only; 0 elsewhere)
    planes: torch.Tensor    # bool [n, T, n_planes] (spatial only)


# ===================================================================================== the model
@TRANSFORMERS.register("tstct", requires=(), summary="typed spatial/temporal/causal heads, two-stream loop, Environment cache")
class TSTCT(nn.Module):
    """The Environment builder. See the module docstring.

    Parameters
    ----------
    cfg: TSTCTConfig.
    latent_dim: dz = Dc + G·C of the shared latent space.
    n_planes: number of relation planes (AS-01) for the plane-bit bias.
    disc_groups, disc_classes: G and C of the categorical latent (the prior head predicts G×C logits;
        Dc = latent_dim − G·C). Required: the prior must split z exactly as CVG-AE does.
    """

    def __init__(self, cfg: TSTCTConfig, *, latent_dim: int, n_planes: int, disc_groups: int, disc_classes: int) -> None:
        super().__init__()
        for key in ("AS-05", "AS-06", "AS-09", "AS-10"):
            assume(key, by=__name__)
        if cfg.dim % cfg.heads:
            raise InvariantViolation("TSTCT dim must be divisible by heads")
        self.cfg = cfg
        self.dim, self.heads = cfg.dim, cfg.heads
        self.head_dim = cfg.dim // cfg.heads
        self.n_planes = n_planes
        self.disc_groups, self.disc_classes = disc_groups, disc_classes
        self.cont_dim = latent_dim - disc_groups * disc_classes
        if self.cont_dim <= 0:
            raise InvariantViolation("latent_dim must exceed G·C (Dc > 0)")
        hs, ht, hc = cfg.spatial_heads, cfg.temporal_heads, cfg.causal_heads
        groups = head_groups(cfg)
        self.groups: torch.Tensor
        self.rot_heads: torch.Tensor
        self.register_buffer("groups", groups, persistent=False)
        self.register_buffer("rot_heads", groups != SPATIAL, persistent=False)   # temporal + causal rotate

        # Input projection and the weight-tied block stack (AS-06).
        self.w_in = nn.Linear(latent_dim, cfg.dim, bias=False)
        self.stack = TwoStreamStack([SelfBlock(cfg.dim, cfg.heads, mlp_hidden=cfg.mlp_hidden) for _ in range(cfg.blocks)])
        self.rotary = TimeRotary(self.head_dim, p_min=cfg.rotary_p_min, p_max=cfg.rotary_p_max)

        # Spatial bias tables: hop (0 self, 1, 2), plane bits, log-age bucket.
        self.hop_bias = nn.Parameter(torch.zeros(3, hs))
        self.plane_bias = nn.Parameter(torch.zeros(n_planes, hs))
        self.age_bias = LogDeltaBias(hs, n_buckets=cfg.delta_buckets, delta0=cfg.delta0_s)
        # Log-Δt bucket bias of temporal and causal heads (one table, columns = those heads in order).
        self.delta_bias = LogDeltaBias(ht + hc, n_buckets=cfg.delta_buckets, delta0=cfg.delta0_s)

        # Causal gate MLP per causal head (AS-152), reading block-0 memory-stream values V⁰.
        gh = cfg.gate_hidden
        self.gate_q = nn.Linear(cfg.dim, hc * gh, bias=False)
        self.gate_k = nn.Linear(cfg.dim, hc * gh, bias=False)
        self.gate_dt = TimeDeltaEncoding(hc * gh, n_frequencies=cfg.dt_frequencies, delta0=cfg.delta0_s)
        self.gate_out = nn.Parameter(torch.randn(hc, gh) / gh**0.5)
        self.gate_out_bias = nn.Parameter(torch.full((hc,), cfg.gate_bias_init))

        # Transition prior head on the memory stream (AS-05).
        self.prior_norm = RMSNorm(cfg.dim)
        self.prior_dt = TimeDeltaEncoding(cfg.dim, n_frequencies=cfg.dt_frequencies, delta0=cfg.delta0_s)
        self.prior_mlp = SwiGLU(cfg.dim, cfg.mlp_hidden, out_dim=2 * self.cont_dim + disc_groups * disc_classes)

    @classmethod
    def from_config(cls, cfg: NagaHanaConfig) -> TSTCT:
        """Build from the whole model configuration (latent split from CVG-AE, planes from the graph)."""
        return cls(cfg.tstct, latent_dim=cfg.latent_dim, n_planes=len(cfg.graph.planes),
                   disc_groups=cfg.cvgae.disc_groups, disc_classes=cfg.cvgae.disc_classes)

    # ================================================================== budgets
    def sample_passes(self, generator: torch.Generator | None = None) -> int:
        """R = 1 + Poisson(train_passes_mean), clipped to [1, max_passes] (AS-07)."""
        assume("AS-07", by=__name__)
        lam = torch.tensor([self.cfg.train_passes_mean], dtype=torch.float64)
        r = 1 + int(torch.poisson(lam, generator=generator).item())
        return max(1, min(r, self.cfg.max_passes))

    # ================================================================== shared pieces
    def _rotation_hook(self, time: torch.Tensor) -> Any:
        # cos/sin [B, T, d_h] from float64 times; rotates temporal and causal heads of [B, H, T, d_h].
        cos, sin = self.rotary.angles(time)

        def hook(x: torch.Tensor) -> torch.Tensor:
            return rotate_heads(x, cos, sin, self.rot_heads)

        return hook

    def _gate_logits(self, gq: torch.Tensor, gk: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        """Gate logits: gq, gk [..., H_c, g_h], dt [...] (float64) → [..., H_c]."""
        hc, gh = self.cfg.causal_heads, self.cfg.gate_hidden
        c = self.gate_dt(dt).view(*dt.shape, hc, gh)                 # φ(Δt) through C_h
        h = F.silu(gq + gk + c)                                      # first layer of the MLP, split by input
        return (h * self.gate_out).sum(-1) + self.gate_out_bias

    def _values0(self, e0: torch.Tensor) -> torch.Tensor:
        """V⁰ = W_V⁰ RMSNorm⁰(e⁰): block 0's memory-stream values, flattened over heads [..., d].

        Identical to the V that block 0 writes into the cache (`SelfBlock.kv`), so stored and carried
        slots hold it already (as `values(0)` / `carry.v[0]`).
        """
        block = cast(SelfBlock, self.stack.blocks[0])
        return block.attn.v_proj(block.norm1(e0))

    def _gate_q(self, u: torch.Tensor) -> torch.Tensor:
        """Gate queries from V⁰ [..., d] → [..., H_c, g_h]."""
        return self.gate_q(u).view(*u.shape[:-1], self.cfg.causal_heads, self.cfg.gate_hidden)

    def _gate_k(self, u: torch.Tensor) -> torch.Tensor:
        """Gate keys from V⁰ [..., d] → [..., H_c, g_h] (linear: merging commutes with it)."""
        return self.gate_k(u).view(*u.shape[:-1], self.cfg.causal_heads, self.cfg.gate_hidden)

    def _rebase_keys(self, k: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
        """Rotate the time-rotary heads of keys k [B, H, T, d_h] by angle ω·shift_b (float64 [B]).

        A key rotated relative to origin o (angle ω(t − o)) becomes relative to o − shift; rotations of
        each frequency pair compose exactly, so only float32 rounding (≈ 1e-7 relative) is added.
        """
        b, _, t, _ = k.shape
        cos, sin = self.rotary.angles(shift.to(torch.float64)[:, None].expand(b, t))   # [B, T, d_h]
        return rotate_heads(k, cos, sin, self.rot_heads)

    def _top_k(self, gate: torch.Tensor, cand: torch.Tensor) -> torch.Tensor:
        """Keep the `causal_keys` largest gates among candidates along the last axis. bool like `gate`."""
        k = self.cfg.causal_keys
        if k >= gate.shape[-1]:
            return cand.clone()
        score = torch.where(cand, gate.detach(), torch.full_like(gate, -1.0))
        top = torch.topk(score, k, dim=-1).indices
        keep = torch.zeros_like(cand)
        keep.scatter_(-1, top, True)
        return keep & cand

    def transition_prior(self, h: torch.Tensor, dt: torch.Tensor) -> LatentPrior:
        """p(z_next | Environment): h [..., d] memory-stream states, dt [...] float64 gap ahead → LatentPrior.

        Also applied to the thinking stream's output by `refined_prior` (AS-159).
        """
        x = self.prior_norm(h) + self.prior_dt(dt)
        out = self.prior_mlp(x)
        dc, g, c = self.cont_dim, self.disc_groups, self.disc_classes
        mean, logvar, logits = out.split([dc, dc, g * c], dim=-1)
        return LatentPrior(mean=mean, logvar=logvar, logits=logits.reshape(*logits.shape[:-1], g, c))

    def gate_l1(self, out: EnvironmentOut) -> torch.Tensor:
        """Sparsity penalty: mean gate value over causal candidates (g > 0 exactly on candidates).

        Gates are ≥ 0, so this is the L1 norm normalised by the number of candidates; the training
        loss multiplies it by λ_gate (TrainingConfig.lambda_gate, AS-10).
        """
        g = out.causal_gate
        n = (g > 0).sum().clamp_min(1)
        return g.sum() / n

    def refined_prior(self, out: EnvironmentOut, window: WindowBatch) -> LatentPrior:
        """The transition prior read from the thinking stream: transition_prior(out.refined, next_dt) (AS-159).

        For the stage-3 KL on both streams: the memory-stream prior (`out.prior`) is the cache-consistent
        world model; this one gives the R thinking passes their own stage-3 objective.
        """
        return self.transition_prior(out.refined, window.positions.next_dt)

    # ================================================================== dense (training) path
    def forward(
        self,
        z: torch.Tensor,
        window: WindowBatch,
        *,
        passes: int,
        grad_passes: int | None = None,
        need_weights: bool = False,
        carry: CarriedEnvironment | None = None,
    ) -> EnvironmentOut:
        """Dense masked path over B windows: z [B, P, dz] → EnvironmentOut (see the module docstring).

        carry: the earlier windows' Environment (D-51), aligned to this window (`CarriedEnvironment.align`).
        With carry, attention keys are [C carried ; P current]: `causal_gate` is [B, H_c, P, C + P] (carried
        candidates first) and attention weights are [B, H, P, C + P + 1]; `kv` holds this window's
        entries only ([B, H, P, d_h]). Carried tensors are detached: they never receive gradient.
        """
        cfg = self.cfg
        pos = window.positions
        b_, p_, _ = z.shape
        hs, ht, hc = cfg.spatial_heads, cfg.temporal_heads, cfg.causal_heads
        e0 = self.w_in(z)                                                       # [B, P, d]
        u_cur = self._values0(e0)                                               # [B, P, d] block-0 values

        # ---------------------------------------------------------- carried keys (D-51), stop-gradient
        carried_keys: CarriedKeys | None = None
        ck: list[torch.Tensor] = []
        cv: list[torch.Tensor] = []
        count = e0.new_ones(b_, p_)                                             # [B, K] merged-state counts
        gk = self._gate_k(u_cur)                                                # [B, K, H_c, g_h]
        if carry is not None:
            if carry.entity is None:
                raise InvariantViolation("align the carry to this window first (CarriedEnvironment.align)")
            shift = window.origin.to(torch.float64) - carry.origin.to(torch.float64)        # [B] o_w − o_c
            carried_keys = CarriedKeys(entity=carry.entity, time=carry.time.to(torch.float64) - shift[:, None],
                                       mask=carry.mask, bucket=carry.bucket, latest=carry.latest)
            ck = [self._rebase_keys(k.detach().to(e0.dtype), -shift) for k in carry.k]       # angle ω(t − o_w)
            cv = [v.detach().to(e0.dtype) for v in carry.v]
            c_ = carry.num_slots
            u_car = cv[0].permute(0, 2, 1, 3).reshape(b_, c_, -1)                # [B, C, d] carried V⁰
            gk = torch.cat([self._gate_k(u_car), gk], dim=1)
            count = torch.cat([torch.where(carry.mask, carry.count.to(e0.dtype), torch.ones_like(carry.count, dtype=e0.dtype)),
                               count], dim=1)
        masks = build_dense_masks(pos, window.contact1, window.contact2, window.contact_planes, cfg, carried_keys)
        k_ = masks.dt.shape[-1]                                                 # K = C + P

        # ---------------------------------------------------------- causal gates on candidates only
        cand = masks.causal_candidates                                          # [B, P, K]
        bi, ii, jj = torch.nonzero(cand, as_tuple=True)
        gq = self._gate_q(u_cur)                                                # [B, P, H_c, g_h]
        g = torch.sigmoid(self._gate_logits(gq[bi, ii], gk[bi, jj], masks.dt[bi, ii, jj]))  # [nnz, H_c]
        gate_bpkh = e0.new_zeros(b_, p_, k_, hc)
        gate_bpkh = gate_bpkh.index_put((bi, ii, jj), g)                       # zeros off candidates
        gate = gate_bpkh.permute(0, 3, 1, 2)                                    # [B, H_c, P, K]
        kept = self._top_k(gate, cand[:, None].expand_as(gate))                 # [B, H_c, P, K]

        # ---------------------------------------------------------- patterns per head
        allowed = masks.allowed.clone()                                         # [B, H, P, K]
        allowed[:, hs + ht:] = kept

        # ---------------------------------------------------------- additive biases per head group
        sp = (self.hop_bias[masks.hop.clamp_min(0)]                             # [B, P, K, H_s]
              + masks.planes.to(e0.dtype) @ self.plane_bias
              + self.age_bias(masks.dt))
        sp = sp.masked_fill(~masks.spatial[..., None], 0.0).permute(0, 3, 1, 2)
        tc = self.delta_bias(masks.dt).permute(0, 3, 1, 2)                      # [B, H_t + H_c, P, K]
        tm = tc[:, :ht].masked_fill(~masks.temporal[:, None], 0.0)
        log_g = torch.log(torch.where(kept, gate, torch.ones_like(gate)))       # 0 where not kept (no log 0)
        ca = torch.where(kept, tc[:, ht:] + log_g, torch.zeros_like(log_g))
        bias = torch.cat([sp, tm, ca], dim=1)                                   # [B, H, P, K]
        log_n = count.clamp_min(1.0).log()[:, None, None, :]                   # [B, 1, 1, K] (0 in-window)
        bias = bias + torch.where(allowed, log_n, torch.zeros_like(bias))

        # ---------------------------------------------------------- rotary hooks (padded times → 0)
        t = torch.where(pos.mask, pos.time.to(torch.float64), torch.zeros_like(pos.time, dtype=torch.float64))
        hook = self._rotation_hook(t)
        ctx = AttnContext(q_hook=hook, k_hook=hook, allowed=allowed, bias=bias, need_weights=need_weights)

        # ---------------------------------------------------------- two streams (AS-06)
        if carry is None:
            memory, kvs, aux_m = self.stack.memory(e0, ctx)
            reads: list[KV] = list(kvs)
        else:
            # Memory stream with the carried K/V prepended in every block (read-only, no gradient).
            kvs, reads, aux_m = [], [], []
            x = e0
            for b, module in enumerate(self.stack.blocks):
                block = cast(SelfBlock, module)
                k, v = block.kv(x, ctx)                                         # [B, H, P, d_h] this window
                kvs.append((k, v))
                full = (torch.cat([ck[b], k], dim=2), torch.cat([cv[b], v], dim=2))   # [B, H, K, d_h]
                reads.append(full)
                x, aux = block(x, full, ctx, b)
                aux_m.append(aux)
            memory = x
        refined, aux_t = self.stack.think(e0, reads, ctx, passes=passes, grad_passes=grad_passes)
        prior = self.transition_prior(memory, pos.next_dt)
        attention = ([{"memory_weights": am.get("self_weights"), "thinking_weights": at.get("self_weights"),
                       "groups": self.groups} for am, at in zip(aux_m, aux_t, strict=True)] if need_weights else [])
        return EnvironmentOut(memory=memory, kv=kvs, refined=refined, prior=prior, causal_gate=gate,
                              passes=passes, attention=attention)

    # ================================================================== cached (inference) path
    def step_context(self, entity: torch.Tensor, time: torch.Tensor, contact1: torch.Tensor, contact2: torch.Tensor,
                     contact_planes: torch.Tensor, *, horizon_dt: torch.Tensor | None = None) -> StepContext:
        """`StepContext.from_contacts` with this model's tie tolerance (AS-160)."""
        return StepContext.from_contacts(entity, time, contact1, contact2, contact_planes, horizon_dt=horizon_dt,
                                         tie_s=self.cfg.time_tie_s)

    def new_store(self, mem_cfg: MemoryConfig, *, model_hash: str, latent_space: str, schema_version: str,
                  origin: float) -> EnvironmentStore:
        """An empty Environment store shaped for this model (one per inference site or training stream).

        K/V are stored in fp32 (`CACHE_DTYPE`, D-54), independent of the compute precision of the
        forward pass that writes them (bf16 under training autocast, AS-39)."""
        meta = KVCacheMeta(Region.ENVIRONMENT, "tstct", model_hash, latent_space, schema_version,
                           self.cfg.blocks, self.heads, self.head_dim)
        return EnvironmentStore(mem_cfg, meta, origin=origin, device=self.w_in.weight.device, dtype=CACHE_DTYPE)

    # ================================================================== carry across windows (D-51)
    def carry_out(self, out: EnvironmentOut, window: WindowBatch, *, keep: CarryPolicy) -> CarriedEnvironment:
        """Write this window's memory-stream K/V into the per-stream stores and export the next carry.

        For row b: the window's real positions are appended to `keep.stores[b]` with entity ids
        `keep.entity_keys[b][entity]`, times re-based to the store origin and keys re-rotated by
        ω·(o_window − o_store), so every slot of a store shares one rotation origin (merging needs it).
        The returned carry holds every cell slot and register of the stores (bounded, P3/P4) and must
        be `align`-ed to the next window's entity table before `forward(carry=...)`.
        """
        pos = window.positions
        if len(keep.stores) != pos.entity.shape[0]:
            raise InvariantViolation("CarryPolicy needs one store per batch row (window stream)")
        with torch.no_grad():
            for b, store in enumerate(keep.stores):
                idx = torch.nonzero(pos.mask[b]).flatten()                       # real positions, in order
                if idx.numel() == 0:
                    continue
                shift = float(window.origin[b]) - store.origin                   # o_w − o_s
                ents = keep.entity_keys[b][pos.entity[b, idx]]
                if bool((ents < 0).any()):
                    raise InvariantViolation("every written entity needs a stable key (entity_keys ≥ 0)")
                times = pos.time[b, idx].to(torch.float64) + shift
                kv_rows: list[KV] = []
                for k, v in out.kv:
                    kb = self._rebase_keys(k[b:b + 1, :, idx].detach(), torch.tensor([shift], dtype=torch.float64))
                    kv_rows.append((kb[0].transpose(0, 1), v[b, :, idx].detach().transpose(0, 1)))   # [n, H, d_h]
                store.append(ents, times, kv_rows, role=Role.SIMULATOR)
                store.compact(store.clock)                                       # coarsen every entity (P1)
        return self.export_carry(keep.stores)

    def export_carry(self, stores: Sequence[EnvironmentStore]) -> CarriedEnvironment:
        """All cell slots and registers of each stream's store, sorted by (time, write order), padded to C."""
        rows = [store.export_slots(role=Role.SIMULATOR) for store in stores]
        b_, c_ = len(stores), max([1, *(len(ids) for ids, _ in rows)])
        dev, dt = self.w_in.weight.device, self.w_in.weight.dtype
        h, dh = self.heads, self.head_dim
        k = [torch.zeros(b_, h, c_, dh, device=dev, dtype=dt) for _ in range(self.cfg.blocks)]
        v = [torch.zeros(b_, h, c_, dh, device=dev, dtype=dt) for _ in range(self.cfg.blocks)]
        time = torch.zeros(b_, c_, dtype=torch.float64)
        key = torch.full((b_, c_), -1, dtype=torch.long)
        count = torch.ones(b_, c_, dtype=torch.float32)
        latest = torch.zeros(b_, c_, dtype=torch.bool)
        mask = torch.zeros(b_, c_, dtype=torch.bool)
        for b, (store, (ids, reg)) in enumerate(zip(stores, rows, strict=True)):
            n = len(ids)
            if not n:
                continue
            sel = torch.tensor(ids, dtype=torch.long, device=store.device)
            for blk in range(self.cfg.blocks):
                k[blk][b, :, :n] = store.keys(blk)[sel].transpose(0, 1).to(dev, dt)
                v[blk][b, :, :n] = store.values(blk)[sel].transpose(0, 1).to(dev, dt)
            time[b, :n] = torch.tensor(store.slot_time(ids), dtype=torch.float64)
            key[b, :n] = torch.tensor(store.slot_entity(ids), dtype=torch.long)
            count[b, :n] = torch.tensor(store.slot_count(ids), dtype=torch.float32)
            latest[b, :n] = torch.tensor(reg, dtype=torch.bool)
            mask[b, :n] = True
        origin = torch.tensor([store.origin for store in stores], dtype=torch.float64)
        return CarriedEnvironment(k=k, v=v, origin=origin, time=time.to(dev), key=key.to(dev), count=count.to(dev),
                                  bucket=(mask & ~latest).to(dev), latest=latest.to(dev), mask=mask.to(dev))

    def step(
        self,
        z_new: torch.Tensor,
        entity: torch.Tensor,
        time: torch.Tensor,
        store: EnvironmentStore,
        context: StepContext,
        *,
        passes: int,
        need_weights: bool = False,
    ) -> StepOut:
        """Process n new states against the store, then append their memory-stream K/V.

        z_new [n, dz]; entity long [n] (store ids); time float64 [n] (seconds since the store origin,
        non-decreasing, ≥ store.clock). New state k sees the store plus the new states before it in
        this call (AS-156), so any split of a stream into calls gives the same result.
        """
        cfg = self.cfg
        n = int(z_new.shape[0])
        if n == 0:
            raise InvariantViolation("step needs at least one new state")
        times = time.to(torch.float64)
        if float(times[0]) < store.clock or bool((times[1:] < times[:-1]).any()):
            raise InvariantViolation("new states must be in non-decreasing time order, at or after the store clock")
        hs, ht, hc = cfg.spatial_heads, cfg.temporal_heads, cfg.causal_heads
        e0 = self.w_in(z_new)[None]                                            # [1, n, d]
        u_new = self._values0(e0[0])                                            # [n, d] block-0 values
        gq, gk_new = self._gate_q(u_new), self._gate_k(u_new)                   # [n, H_c, g_h]

        # ---------------------------------------------------------- key sets (shared by all blocks)
        s_rows = store.buffer_rows
        sp, tm, ca = self._key_sets(entity, times, store, context, s_rows)

        # ---------------------------------------------------------- causal gates and top-k
        # Gate keys from block-0 values V⁰: stored slots read the store's values(0), new states their own.
        u_store = store.values(0)[ca.pool.clamp(0, max(s_rows - 1, 0))].reshape(n, ca.pool.shape[1], -1)
        ca_store = self._gate_k(u_store.to(e0.dtype))                          # [n, T_c, H_c, g_h]
        ca_new = gk_new[(ca.pool - s_rows).clamp(0, n - 1)]                    # [n, T_c, H_c, g_h]
        gk = torch.where((ca.pool >= s_rows)[..., None, None], ca_new, ca_store)
        g = torch.sigmoid(self._gate_logits(gq[:, None], gk, ca.dt)).permute(0, 2, 1)   # [n, H_c, T_c]
        cmask = ca.mask[:, None].expand_as(g)
        g = torch.where(cmask, g, torch.zeros_like(g))
        kept = self._top_k(g, cmask)

        # ---------------------------------------------------------- per-head patterns and biases
        t_k = max(sp.pool.shape[1], tm.pool.shape[1], ca.pool.shape[1])

        def pad(x: torch.Tensor, value: float | bool) -> torch.Tensor:
            # [..., T] → [..., T_k]
            extra = t_k - x.shape[-1]
            return F.pad(x, (0, extra), value=value) if extra else x

        sp_bias = (self.hop_bias[sp.hop] + sp.planes.to(e0.dtype) @ self.plane_bias + self.age_bias(sp.dt)
                   + sp.count.log()[..., None])                                 # [n, T_s, H_s]
        tc_tm = self.delta_bias(tm.dt)[..., :ht] + tm.count.log()[..., None]    # [n, T_t, H_t]
        tc_ca = (self.delta_bias(ca.dt)[..., ht:] + ca.count.log()[..., None]).permute(0, 2, 1)  # [n, H_c, T_c]
        log_g = torch.log(torch.where(kept, g, torch.ones_like(g)))
        ca_bias = torch.where(kept, tc_ca + log_g, torch.zeros_like(tc_ca))
        bias = torch.cat([
            pad(sp_bias.masked_fill(~sp.mask[..., None], 0.0).permute(2, 0, 1), 0.0),     # [H_s, n, T_k]
            pad(tc_tm.masked_fill(~tm.mask[..., None], 0.0).permute(2, 0, 1), 0.0),       # [H_t, n, T_k]
            pad(ca_bias.permute(1, 0, 2), 0.0),                                             # [H_c, n, T_k]
        ], dim=0)[None]                                                                   # [1, H, n, T_k]
        allowed = torch.cat([
            pad(sp.mask[None].expand(hs, -1, -1), False),
            pad(tm.mask[None].expand(ht, -1, -1), False),
            pad(kept.permute(1, 0, 2), False),
        ], dim=0)[None]                                                                   # [1, H, n, T_k]

        hook = self._rotation_hook(times[None])                                 # new states' own times
        ctx = AttnContext(q_hook=hook, k_hook=hook, allowed=allowed, bias=bias, gathered=True,
                          need_weights=need_weights)

        # ---------------------------------------------------------- memory stream, block by block
        groups_pools = [(sp.pool, slice(0, hs)), (tm.pool, slice(hs, hs + ht)), (ca.pool, slice(hs + ht, hs + ht + hc))]
        gathered: list[KV] = []
        new_kv: list[KV] = []
        aux_m: list[dict[str, Any]] = []
        x = e0
        for b, module in enumerate(self.stack.blocks):
            block = cast(SelfBlock, module)
            k_new, v_new = block.kv(x, ctx)                                     # [1, H, n, d_h] (keys rotated)
            kn, vn = k_new[0].transpose(0, 1), v_new[0].transpose(0, 1)         # [n, H, d_h]
            kg = self._assemble(store.keys(b), kn, groups_pools, s_rows, t_k)
            vg = self._assemble(store.values(b), vn, groups_pools, s_rows, t_k)
            gathered.append((kg, vg))
            new_kv.append((kn, vn))
            x, aux = block(x, (kg, vg), ctx, b)
            aux_m.append(aux)
        memory = x

        # ---------------------------------------------------------- thinking stream reads the same keys
        refined, aux_t = self.stack.think(e0, gathered, ctx, passes=passes)

        # ---------------------------------------------------------- write the memory stream to the store
        slots = store.append(entity, times, new_kv, role=Role.SIMULATOR)
        prior = self.transition_prior(memory[0], context.horizon_dt) if context.horizon_dt is not None else None
        attention = ([{"memory_weights": am.get("self_weights"), "thinking_weights": at.get("self_weights"),
                       "groups": self.groups} for am, at in zip(aux_m, aux_t, strict=True)] if need_weights else [])
        ca_time = torch.where(ca.mask, times[:, None] - ca.dt, torch.zeros_like(ca.dt))
        return StepOut(memory=memory[0], refined=refined[0], kv=new_kv, slots=slots, prior=prior, causal_gate=g,
                       causal_entity=ca.entity, causal_time=ca_time, causal_kept=kept, passes=passes, attention=attention)

    @staticmethod
    def _assemble(store_buf: torch.Tensor, new_buf: torch.Tensor,
                  groups_pools: Sequence[tuple[torch.Tensor, slice]], s_rows: int, t_k: int) -> torch.Tensor:
        """Per-head gathered keys (or values): [1, H, n, T_k, d_h] from store rows and new rows.

        Pool id < s_rows reads the store buffer [S, H, d_h]; ≥ s_rows reads new_buf [n, H, d_h].
        Each head group reads its own key set; padding rows are zero (and masked by `allowed`).
        """
        n = new_buf.shape[0]
        parts = []
        for pool, heads in groups_pools:
            from_store = store_buf[pool.clamp(0, max(s_rows - 1, 0))]                    # [n, T, H, d_h]
            from_new = new_buf[(pool - s_rows).clamp(0, n - 1)]                         # [n, T, H, d_h]
            rows = torch.where((pool >= s_rows)[..., None, None], from_new, from_store)
            rows = torch.where((pool >= 0)[..., None, None], rows, torch.zeros_like(rows))
            rows = rows[:, :, heads].permute(2, 0, 1, 3)                                # [H_g, n, T, d_h]
            extra = t_k - rows.shape[2]
            if extra:
                rows = F.pad(rows, (0, 0, 0, extra))
            parts.append(rows)
        return torch.cat(parts, dim=0)[None]

    def _key_sets(self, entity: torch.Tensor, times: torch.Tensor, store: EnvironmentStore, context: StepContext,
                  s_rows: int) -> tuple[_KeySet, _KeySet, _KeySet]:
        """Spatial, temporal and causal key sets of the new states (same definitions as masks.py).

        Ordering "most recent" is by (time, write sequence), which equals position order in the
        dense path (AS-156). New state k gets pool id s_rows + k and sequence store.next_seq + k.
        """
        cfg = self.cfg
        role = Role.SIMULATOR
        ents, ts = entity.tolist(), times.tolist()
        n = len(ents)
        seq0 = store.next_seq
        n_planes = self.n_planes
        no_planes = (False,) * n_planes
        # Entry: (pool id, time, count, seq, entity, hop, planes)
        Entry = tuple[int, float, float, int, int, int, tuple[bool, ...]]
        sp_rows: list[list[Entry]] = []
        tm_rows: list[list[Entry]] = []
        ca_rows: list[list[Entry]] = []

        def from_store(slot: int, e: int, hop: int = 0, planes: tuple[bool, ...] = no_planes) -> Entry:
            return (slot, store.slot_time([slot])[0], store.slot_count([slot])[0], store.slot_seq([slot])[0], e, hop, planes)

        def from_new(k: int, hop: int = 0, planes: tuple[bool, ...] = no_planes) -> Entry:
            return (s_rows + k, ts[k], 1.0, seq0 + k, ents[k], hop, planes)

        def recency(entry: Entry) -> tuple[float, int]:
            return (entry[1], entry[3])

        for k in range(n):
            e, t = ents[k], ts[k]
            # Temporal: own last W states up to and including k.
            own = [from_store(s, e) for s in store.temporal_slots(e, t, cfg.temporal_window, role=role)]
            own += [from_new(j) for j in range(k + 1) if ents[j] == e]
            tm_rows.append(own[-cfg.temporal_window:])
            # Spatial: self + neighbours' latest state as of k, most recent `spatial_keys`.
            nbrs: list[Entry] = []
            causal: list[Entry] = []
            for c in range(context.neighbour_entity.shape[1]):
                v = int(context.neighbour_entity[k, c])
                if v < 0 or v == e:
                    continue
                hop = int(context.neighbour_hop[k, c])
                planes = tuple(bool(x) for x in context.neighbour_planes[k, c].tolist())
                earlier = [j for j in range(k) if ents[j] == v]
                if earlier:
                    nbrs.append(from_new(earlier[-1], hop, planes))
                else:
                    slot = store.latest_slot(v, t, role=role)
                    if slot >= 0:
                        nbrs.append(from_store(slot, v, hop, planes))
                # Causal candidates: hop-1 sources, 0 < t − t_j ≤ Λ_lag.
                if hop == 1:
                    causal += [from_store(s, v) for s in store.lagged_slots(v, t, cfg.causal_lag_s, role=role, tie=cfg.time_tie_s)]
                    causal += [from_new(j) for j in earlier if cfg.time_tie_s < t - ts[j] <= cfg.causal_lag_s]
            nbrs.sort(key=recency, reverse=True)
            sp_rows.append([from_new(k), *nbrs[: cfg.spatial_keys]])
            causal.sort(key=recency, reverse=True)
            ca_rows.append(causal[: cfg.causal_candidate_cap])

        def tensors(rows: list[list[Entry]]) -> _KeySet:
            width = max([1, *(len(r) for r in rows)])
            pool = torch.full((n, width), -1, dtype=torch.long)
            dt = torch.zeros((n, width), dtype=torch.float64)
            count = torch.ones((n, width), dtype=torch.float32)
            ent = torch.full((n, width), -1, dtype=torch.long)
            hop = torch.zeros((n, width), dtype=torch.long)
            planes = torch.zeros((n, width, n_planes), dtype=torch.bool)
            for k, r in enumerate(rows):
                for c, (pid, tj, cnt, _seq, ej, hj, pl) in enumerate(r):
                    pool[k, c], dt[k, c], count[k, c], ent[k, c], hop[k, c] = pid, ts[k] - tj, cnt, ej, hj
                    planes[k, c] = torch.tensor(pl, dtype=torch.bool) if n_planes else planes[k, c]
            dev = self.w_in.weight.device
            return _KeySet(pool=pool.to(dev), mask=(pool >= 0).to(dev), dt=dt.to(dev), count=count.to(dev),
                           entity=ent.to(dev), hop=hop.to(dev), planes=planes.to(dev))

        return tensors(sp_rows), tensors(tm_rows), tensors(ca_rows)


# ===================================================================================== utilities
def weights_hash(module: nn.Module) -> str:
    """SHA-256 over the module's state dict (names, dtypes, shapes, bytes): the cache's `model_hash` (P-18)."""
    h = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        t = tensor.detach().cpu().contiguous()
        h.update(name.encode())
        h.update(str(t.dtype).encode())
        h.update(str(tuple(t.shape)).encode())
        h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes() if t.numel() else b"")
    return h.hexdigest()


__all__ = ["CAUSAL", "SPATIAL", "TEMPORAL", "TRANSFORMERS", "CarriedEnvironment", "CarryPolicy", "StepContext", "StepOut", "TSTCT",
           "TimeDeltaEncoding",
           "weights_hash"]
