"""Compute profile of L, the NagaHana model (Generator excluded), from the built modules.

The profile covers FLOPs, activation memory, weights, the context window and the working memory at an
operating point between the light and the heavy case: 4,096 active entities with 512 states each, the
critical-infrastructure point of the site sizing (`lab/sizing.py`, D-63).

Where every number comes from
-----------------------------
- Parameters: `models.nagahana.count_parameters(preset("L"))`, a meta-device build of the real modules (no
  memory is allocated). The count follows the data model's columns (D-64).
- Weights memory (D-54): weights are stored and served in float32, 4 bytes per parameter. The half-precision
  size (2 bytes per parameter) is kept only as a labelled reference figure (`weights_fp16_unused_bytes`): no
  NagaHana code stores or serves 16-bit weights. Outputs are computed in float64 (D-54) but are per-trigger
  tensors of a few KiB, not counted here.
- Cache memory (D-54): the Environment and Imagination K/V caches are stored in float32, 4 bytes per element
  (`CACHE`, the same constant as `memory.kvcache.CACHE_ELEMENT_BYTES`). Inference runs without autocast, so the
  K/V working sets derived from them (TAAFT's gathered view of the Environment cache, the Forecaster's route
  caches) are float32 too. Only training activations stay at 2 bytes (bf16 autocast, AS-39). The memory
  formulas (`environment_bytes`, `imagination_store_bytes`, `taaft_view_bytes`, `taaft_scores_bytes`,
  `route_cache_bytes`) are shared with the site budget of `lab/sizing.py`.
- Accelerators: the inference memory sets the server: `accelerators_needed` is the number of 80 GB
  accelerators that hold it (ceil(inference bytes / 80 GB) = 4). The effective compute of the server
  (1.2e14 FLOP/s, an order-of-magnitude assumption of the architecture chapter) is kept as stated; four
  accelerators exceed it, so the compute figures (trigger time, server load) are conservative.
- Shapes: `preset("L")` (models/config). Small per-row maps (heads, decoders, lens maps, time encodings) are
  read from the built modules themselves: `_mm(module)` is the cost of pushing one row through every
  `nn.Linear` of the module once, 2 x its weight count.
- Schedule: what the inference engine runs (`inference/engine.py`): per state update the FieldEncoder, the
  CVG-AE of each position's local subgraph and one `TSTCT.step` (memory stream plus R thinking passes); per
  Forecaster trigger TAAFT (memory stream plus R thinking passes, then S descent steps on the lens energies),
  the Forecaster's imagination (N routes of K steps, each step a one-step lookahead of the top-B candidates),
  the Verifier's process reward of every imagined step and the long-term memory write; the Advisor on demand.
  Explanations (attributions) are not part of the profile.
- Operating point: `OperatingPoint` below; the budgets (K, N, R, S, B, W, rollouts) equal the configuration's
  run-time defaults (tested).

FLOPs (a multiply-add counts as two operations). The block formulas are checked against PyTorch's own count
(`torch.utils.flop_counter.FlopCounterMode`, meta device) on the real modules in `tests/test_lab_compute.py`.
For a block of width d with SwiGLU hidden width f:
  - `SelfBlock` (TSTCT, Forecaster, Verifier), one position attending to S keys: 8 d^2 (Q, K, V, O) + 6 d f
    (SwiGLU) + attention. Attention is 4 (S + 1) d in the dense layout (the learned null key is one more key)
    and 4 S d + 2 d in the gathered layout (the null key's score only). A thinking pass reads the memory
    stream's keys and values: 4 d^2 instead of 8 d^2.
  - `TAAFTBlock`, one position with S_self self keys and S_cross cross keys: 12 d^2 (self Q, K, V, O; cross
    Q, O; the cross keys and values are TSTCT's cached ones, never projected) + 6 d f + 4 (S_self + S_cross) d
    + 4 d (two null-key scores). A thinking pass: 8 d^2.
  - CVG-AE plane layer: 4 d^2 per incidence pair (typed keys and values), 2 d^2 for the query and the typed
    MLP, 2 (3 d + K_rwse) m d + 2 m d d, per output row.
TSTCT's inference step gathers one key set per head group and pads it to the longest group, so every head
scores T_k = max(spatial, temporal, causal candidates) = 512 keys, while the keys a query can read are 608
(64 spatial + 512 temporal + 32 causal kept by the gate).

Rules from the literature
-------------------------
- Training compute: forward plus backward is about three forwards per position, C about 6 N per position for N
  parameters (Kaplan et al. 2020, arXiv:2001.08361). Thinking passes before the last `grad_passes` run without
  gradient (one forward each, AS-07).
- Activations: the accounting of Korthikanti et al. 2022 (arXiv:2205.05198), adapted to the built blocks
  (RMSNorm, QK-norm, SwiGLU, no dropout): a self-attention block stores 10 d + 3 f values per position in half
  precision (norm inputs and outputs, Q and K before and after QK-norm, V, the attention output, the SwiGLU
  gate, up and product), a TAAFT block 15 d + 3 f (a third norm and the cross-attention query, normalised query
  and output). Attention scores are recomputed (fused kernels), not stored. With full recomputation only each
  block's input is kept. Truncated backpropagation keeps the memory stream and `grad_passes` thinking passes.
- Training state: mixed-precision Adam, 16 bytes per parameter (Rajbhandari et al. 2020, ZeRO,
  arXiv:1910.02054).
- Environment cache: 2 L d b bytes per cached state (2 L n_h d_h T b for T states): 128 KiB at L = 16,
  d = 1024, b = 4 (fp32, D-54).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import torch
from torch import nn

from nagahana.memory.kvcache import CACHE_ELEMENT_BYTES
from nagahana.models.config import NagaHanaConfig, preset
from nagahana.models.nagahana import NagaHana, count_parameters
from nagahana.nn.numeric import PeriodicEmbedding

KIB, MIB, GIB = 1024, 1024**2, 1024**3
HALF = 2                                   # bytes per value in half precision (training activations, bf16, AS-39)
FP32 = 4                                   # bytes per value in single precision (stored and served weights, D-54)
CACHE = CACHE_ELEMENT_BYTES                # bytes per stored K/V element: fp32 caches (D-54), = 4


@dataclass(frozen=True)
class OperatingPoint:
    """The operating point of the profile: the critical-infrastructure site class (`lab/sizing.py`)."""

    # working context (the TSTCT's attention pattern)
    entities: int = 4096                # active entities in the working context (TAAFT positions: + adversary slots)
    states_per_entity: int = 512        # temporal heads: an entity's own past states (= MemoryConfig.slots_per_entity)
    spatial_keys: int = 64              # spatial heads: neighbours sharing a hyperedge, at their latest state
    causal_keys: int = 32               # causal heads: the top-k kept by the learned gate
    causal_candidates: int = 256        # candidates the gate scores per query (the cap; reached at the sustained rate)
    # per state update
    positions_per_update: int = 2       # new latent states written: one per endpoint entity (AS-41)
    nodes_per_update: int = 32          # CVG-AE nodes re-encoded: the two endpoints and their sampled neighbourhood
    incidences: int = 4                 # hyperedges per node and plane in a local subgraph (one per hyperedge kind)
    # per Forecaster trigger (budgets N, K, R, S, B, W; the configuration's run-time defaults)
    k: int = 12                         # imagined horizon K, in windows
    n: int = 200                        # imagined routes N
    r: int = 4                          # thinking passes R of TSTCT and of TAAFT
    descent_steps: int = 8              # energy-descent steps S
    lookahead: int = 8                  # candidates B evaluated one step ahead per imagined step (MPPI top-B)
    advisor_width: int = 64             # W: beam width of the Advisor's counter-sequence search
    advisor_depth: int = 3              # counter-sequence length (beam-search depth)
    advisor_rollouts: int = 50          # re-imagined routes per evaluated sequence
    # streaming
    rate: float = 18_400.0              # sustained state updates per second (server, full telemetry)
    window_s: float = 60.0              # one Forecaster trigger per window (CSE-CIC-IDS2018: 60 s)
    server_flops: float = 1.2e14        # effective compute assumed for the server (order of magnitude)
    accelerators: int = 4               # 80 GB accelerators of the server: what the fp32 inference memory needs
    accelerator_bytes: float = 80e9
    retained_bytes_per_update: int = 24  # retained Environment per state update (server, full telemetry)
    # training
    train_seq: int = 2048               # latent states per training sequence (TrainingConfig: 1,024 updates x 2)
    micro_batch: int = 8                # sequences per micro-batch
    corpus_updates: float = 4e9         # state updates in the public training corpus (record level)
    generator_updates: float = 4e9      # state updates of Generator variants
    stage3_passes: float = 2.0          # passes over corpus + variants: perceptor pretraining (Stage 1 of D-62)
    stage4_passes: float = 1.0          # TAAFT pretraining, perceptors frozen (Stage 2 of D-62)
    stage5_passes: float = 1.0          # full training (Stage 3 of D-62)
    stage5_imagined: float = 4e9        # imagined states for the policy, value and process-reward heads in full training
    gpu_peak: float = 989e12            # dense BF16 peak of one H100 SXM (NVIDIA datasheet)
    mfu: float = 0.40                   # model FLOPs utilisation in training


OP = OperatingPoint()
CFG: NagaHanaConfig = preset("L")


@lru_cache(maxsize=4)
def built(name: str = "L") -> NagaHana:
    """The built model on the meta device (shapes only; no memory is allocated)."""
    with torch.device("meta"):
        return NagaHana(preset(name))


@lru_cache(maxsize=4)
def parameters(name: str = "L") -> dict[str, int]:
    """Parameters per component and in total (`count_parameters`, meta device)."""
    return count_parameters(preset(name))


def _mm(*modules: Any) -> int:
    """FLOPs to push one row through every linear map of the modules once: 2 x their weight counts
    (`nn.Linear`, and the per-feature map of a `PeriodicEmbedding`, an einsum over its weight)."""
    return sum(2 * m.weight.numel() for mod in modules if isinstance(mod, nn.Module) for m in mod.modules()
               if isinstance(m, (nn.Linear, PeriodicEmbedding)))


def attention_flops(d: int, keys: int, *, gathered: bool, rotary: bool = False) -> int:
    """Scores and weighted sum of one query over `keys` keys plus the learned null key. A dense layout with
    time-rotated queries adds the un-rotated null-key correction (2 d, `MultiHeadAttention.attend`)."""
    if gathered:
        return 4 * keys * d + 2 * d
    return 4 * (keys + 1) * d + (2 * d if rotary else 0)


def self_block_flops(d: int, f: int, keys: int, *, kv: bool = True, gathered: bool = False,
                     rotary: bool = False) -> int:
    """One SelfBlock position; kv=False is a thinking pass (keys and values read, not projected)."""
    return (8 if kv else 4) * d * d + 6 * d * f + attention_flops(d, keys, gathered=gathered, rotary=rotary)


def taaft_block_flops(d: int, f: int, self_keys: int, cross_keys: int, *, kv: bool = True) -> int:
    """One TAAFTBlock position: joint self-attention (current positions and own past) and cross-attention
    (gathered Environment keys and long-term memory keys), each with a null-key score."""
    return (12 if kv else 8) * d * d + 6 * d * f + 4 * (self_keys + cross_keys) * d + 4 * d


def plane_layer_flops(d: int, mlp_mult: int, rwse: int, incidences: int, rows: int) -> int:
    """One CVG-AE plane layer: `incidences` typed key/value maps, `rows` output rows (query and typed MLP)."""
    return incidences * 4 * d * d + rows * (2 * d * d + 2 * (3 * d + rwse) * mlp_mult * d + 2 * mlp_mult * d * d)


def input_layer_flops(cfg: NagaHanaConfig = CFG, kinds: Sequence[int] | None = None) -> int:
    """The FieldEncoder for one update: numeric and bitmask value maps, attention pooling over all columns.
    kinds: column kind codes (default: the canonical column layout of the data windows)."""
    from nagahana.data.windows import CANONICAL_KIND_CODES
    from nagahana.models.inputs.encoder import _KIND_BIT, NUMERIC_KINDS

    c = cfg.inputs
    kinds = CANONICAL_KIND_CODES if kinds is None else kinds
    n_cols = len(kinds)
    n_num = sum(1 for k in kinds if k in NUMERIC_KINDS)
    n_bit = sum(1 for k in kinds if k == _KIND_BIT)
    values = n_num * 2 * (2 * c.n_frequencies) * c.d_field + n_bit * 2 * c.max_bits * c.d_field
    pool = n_cols * 4 * c.d_field * c.d_update + 4 * c.d_update * c.d_update + 4 * n_cols * c.d_update
    return values + pool


def cvgae_flops(cfg: NagaHanaConfig, nodes: int, incidences: Sequence[int], centres: int) -> int:
    """The CVG-AE over a batch of local subgraphs: `nodes` nodes in all, `incidences[p]` incidence pairs on plane
    p, `centres` positions. Node inputs, P planes x L layers (the last layer updates only the centre rows,
    `out_rows`), the cross-plane coupling of every layer and the variational head."""
    m = built(cfg.name)
    c, g = cfg.cvgae, cfg.graph
    d, planes = c.dim, len(g.planes)
    inputs = nodes * _mm(m.cvgae.in_update, m.cvgae.age_periodic)          # update map, periodic age
    layers = sum((c.layers - 1) * plane_layer_flops(d, c.mlp_mult, g.rwse_steps, inc, nodes)
                 + plane_layer_flops(d, c.mlp_mult, g.rwse_steps, inc, centres) for inc in incidences)
    coupling = c.layers * 2 * planes * planes * nodes * d
    head = centres * _mm(m.cvgae.head_mean, m.cvgae.head_logvar, m.cvgae.head_logits)
    return inputs + layers + coupling + head


def cvgae_position_flops(cfg: NagaHanaConfig = CFG, op: OperatingPoint = OP) -> int:
    """One position's local subgraph through the CVG-AE at the operating point."""
    nodes = op.nodes_per_update // op.positions_per_update
    return cvgae_flops(cfg, nodes, [nodes * op.incidences] * len(cfg.graph.planes), 1)


def tstct_keys(op: OperatingPoint = OP) -> int:
    """Keys a TSTCT query can read: spatial + temporal + causal (kept)."""
    return op.spatial_keys + op.states_per_entity + op.causal_keys


def tstct_padded_keys(op: OperatingPoint = OP) -> int:
    """Keys every head scores in the gathered step layout: the longest head-group set."""
    return max(op.spatial_keys, op.states_per_entity, op.causal_candidates)


def tstct_step_parts(cfg: NagaHanaConfig, keys: int, candidates: int, spatial: int, passes: int) -> dict[str, int]:
    """One new state through `TSTCT.step` with `keys` gathered keys per head, `candidates` causal candidates and
    `spatial` spatial keys: input map, causal gate and biases, memory stream, `passes` thinking passes,
    transition prior."""
    m = built(cfg.name)
    t = cfg.tstct
    d, f = t.dim, t.mlp_hidden
    gate_w = t.causal_heads * t.gate_hidden
    gate = (2 * d * d                                   # block-0 values V^0 of the new state
            + 2 * (2 * d * gate_w)                      # gate query and the new state's gate key
            + candidates * (2 * d * gate_w + _mm(m.tstct.gate_dt))     # candidates' gate keys and phi(delta t)
            + 2 * spatial * len(cfg.graph.planes) * t.spatial_heads)  # plane bias of the spatial keys
    return {
        "input": _mm(m.tstct.w_in),
        "gate": gate,
        "memory": t.blocks * self_block_flops(d, f, keys, kv=True, gathered=True),
        "thinking": passes * t.blocks * self_block_flops(d, f, keys, kv=False, gathered=True),
        "prior": _mm(m.tstct.prior_mlp, m.tstct.prior_dt),
    }


def tstct_position_parts(cfg: NagaHanaConfig = CFG, op: OperatingPoint = OP) -> dict[str, int]:
    """One new state through `TSTCT.step` at the operating point."""
    return tstct_step_parts(cfg, tstct_padded_keys(op), op.causal_candidates, op.spatial_keys, op.r)


def update_parts(cfg: NagaHanaConfig = CFG, op: OperatingPoint = OP) -> dict[str, int]:
    """One state update: input layer, CVG-AE and TSTCT of its positions."""
    tst = tstct_position_parts(cfg, op)
    p = op.positions_per_update
    return {"Input layer": input_layer_flops(cfg),
            "CVG-AE": p * cvgae_position_flops(cfg, op),
            "TSTCT": p * sum(tst.values()),
            "TSTCT thinking": p * tst["thinking"]}      # included in "TSTCT"; reported separately


def update_flops(cfg: NagaHanaConfig = CFG, op: OperatingPoint = OP) -> int:
    u = update_parts(cfg, op)
    return u["Input layer"] + u["CVG-AE"] + u["TSTCT"]


def taaft_tokens(cfg: NagaHanaConfig = CFG, op: OperatingPoint = OP) -> int:
    """TAAFT positions at a trigger: the active entities' states and the adversary slots."""
    return op.entities + cfg.taaft.adversary_slots


def taaft_self_keys(cfg: NagaHanaConfig = CFG, op: OperatingPoint = OP) -> int:
    """Self keys of a TAAFT position: every current position (dense) and its own last M_im triggers."""
    return taaft_tokens(cfg, op) + cfg.taaft.imagination_triggers


def taaft_cross_keys(cfg: NagaHanaConfig = CFG) -> int:
    """Cross keys of an entity position: own states + neighbours' latest + long-term memory probes."""
    a = cfg.taaft
    return a.own_states + a.neighbour_states + a.memory_probes


def decoder_row_flops(cfg: NagaHanaConfig = CFG) -> int:
    """The Decoder's field head for one row (trunk and value heads), as run for the physics lens."""
    dec = built(cfg.name).decoder
    return _mm(dec.trunk, dec.num_head, dec.bit_head, dec.cat_head)


def lens_flops(cfg: NagaHanaConfig = CFG, op: OperatingPoint = OP) -> dict[str, int]:
    """The lens energies at one trigger: context-dependent preparation (once) and one evaluation in y.

    The game lens (`models/taaft/lenses.GameLens`) has two bilinear interactions of rank r' = r // 2 between the
    G slots and the V entities (the valuation and the claimed targeting), so its maps cost g (w_s + w_a) and
    v (w_t + w_e), and its score matrices 2 x 2 g v r'.
    """
    m = built(cfg.name).taaft
    lens = m.energy.terms
    v, g = op.entities, cfg.taaft.adversary_slots
    n = v + g
    r = cfg.taaft.d_hyp // cfg.taaft.lens_rank_divisor
    bt, info, game, top, temp, cau = (lens[k] for k in ("belief-trust", "information", "game", "topology",
                                                        "temporal", "causal"))
    r_game = game.r_game
    prepare = (v * _mm(bt.w_c, bt.trust_prior) + n * _mm(bt.prior)
               + v * _mm(info.w_c, info.w_nu, info.w_o) + v * _mm(game.ctx)
               + n * _mm(temp.mlp_in, temp.mlp_out, temp.log_prec))
    pairwise = 4 * g * v * r_game + 2 * v * v * r + 2 * v * v * r      # game (two matrices), topology, causal
    maps = (v * (_mm(bt.w_y, bt.w_o) + _mm(m.readouts.trust))          # belief-trust: evidence and trust
            + v * _mm(info.w_i)                                        # information
            + g * _mm(game.w_s, game.w_a) + v * _mm(game.w_t, game.w_e)  # game: slot and entity maps
            + v * _mm(top.a) + v * _mm(cau.a, cau.b))                  # topology, causal maps
    physics = v * (_mm(m.readouts.next_latent_head) + decoder_row_flops(cfg))   # decoded believed next state
    dec = built(cfg.name).decoder
    # the input gradient: every map once more, every pair matrix twice, and only the decoder layers that feed
    # Phi_phys (the trunk and the numeric head; categorical and bitmask heads do not)
    gradient = maps + 2 * pairwise + v * (_mm(m.readouts.next_latent_head) + _mm(dec.trunk, dec.num_head))
    return {"prepare": prepare, "evaluation": maps + pairwise + physics, "gradient": gradient}


def taaft_parts(cfg: NagaHanaConfig = CFG, op: OperatingPoint = OP) -> dict[str, int]:
    """TAAFT once per trigger over every position: blocks (memory stream + R thinking passes), the long-term
    memory read, the amortised guess, the lens energies over the descent, the readouts."""
    m = built(cfg.name)
    a, mem = cfg.taaft, cfg.memory
    d, f = a.dim, a.mlp_hidden
    n = taaft_tokens(cfg, op)
    s_self, s_cross = taaft_self_keys(cfg, op), taaft_cross_keys(cfg)
    blocks = n * a.blocks * (taaft_block_flops(d, f, s_self, s_cross, kv=True)
                             + op.r * taaft_block_flops(d, f, s_self, s_cross, kv=False))
    read = a.memory_probes * (_mm(m.longterm.w_q) + 4 * mem.longterm_dim * mem.longterm_hidden
                              + _mm(m.taaft.memory_k, m.taaft.memory_v))
    lens = lens_flops(cfg, op)
    # S descent steps (value and input gradient), the value at y_S, the terms at y_hat, the per-lens shares
    # (value and one gradient per term): S + 3 evaluations and S + 1 gradients (`energy.descend`, AS-211)
    descent = (op.descent_steps + 3) * lens["evaluation"] + (op.descent_steps + 1) * lens["gradient"]
    ro = m.taaft.readouts
    g = a.adversary_slots
    readouts = (op.entities * _mm(ro.compromise, ro.stage, ro.malignity_entity, ro.trust, ro.next_latent_head,
                                  ro.latent_head)
                + g * _mm(ro.malignity_slot, ro.slot_weight, ro.goal_head, ro.type_head) + 2 * g * (a.n_goals + a.n_types))
    # per-plane contact biases of the self-attention (every entity pair) and of the cross keys
    n_planes = len(cfg.graph.planes)
    biases = 2 * op.entities**2 * n_planes * a.heads + 2 * n * (a.own_states + a.neighbour_states) * n_planes * a.heads
    return {
        "Blocks": blocks,
        "Inputs, biases and readouts": (op.entities * _mm(m.taaft.w_in) + read + n * _mm(m.taaft.w_y) + biases
                                        + readouts),
        "Lens energies and descent": lens["prepare"] + descent,
    }


def exposure_flops(cfg: NagaHanaConfig = CFG) -> int:
    """TAAFT's marginal energy E(empty, y) of one imagined hypothesis (null context, a one-entity set; AS-252)."""
    m = built(cfg.name).taaft
    bt, info, game, top, cau = (m.energy.terms[k] for k in ("belief-trust", "information", "game", "topology",
                                                            "causal"))
    r = cfg.taaft.d_hyp // cfg.taaft.lens_rank_divisor
    # the null context has the entity and the G adversary slots; the prior reads every one of them
    prepare = (_mm(bt.w_c, bt.trust_prior, info.w_c, info.w_nu, info.w_o, game.ctx)
               + (1 + cfg.taaft.adversary_slots) * _mm(bt.prior))
    # one entity, no slot: the game's entity maps only; the topology and causal 1 x 1 score matrices
    evaluate = _mm(bt.w_y, bt.w_o, m.readouts.trust, info.w_i, game.w_t, game.w_e, top.a, cau.a, cau.b) + 2 * 2 * r
    return prepare + evaluate


def context_positions(cfg: NagaHanaConfig = CFG) -> int:
    """T: the Forecaster's context positions (adversary slots + top-C entities)."""
    return cfg.taaft.adversary_slots + cfg.forecaster.context_entities


def imagination_flops(cfg: NagaHanaConfig = CFG, op: OperatingPoint = OP, *, routes: int | None = None,
                      horizon: int | None = None) -> dict[str, int]:
    """`Forecaster.imagine` for one trigger: context, prefix, then per route and step the policy, the pointer, the
    B candidates through the route stack (one step each) with their heads and exposure."""
    m = built(cfg.name).forecaster
    fc = cfg.forecaster
    d, f, b = fc.dim, fc.mlp_hidden, op.lookahead
    n = op.n if routes is None else routes
    k_h = op.k if horizon is None else horizon
    t = context_positions(cfg)
    c = fc.context_entities
    s = m.summary.attn
    encode = (t * _mm(m.ctx_enc)                                     # context positions
              + t * _mm(s.k_proj, s.v_proj) + _mm(s.q_proj, s.o_proj) + 4 * (t + 1) * d   # summary pooling
              + c * exposure_flops(cfg))                             # baseline marginal energy E_0 (AS-252)
    prefix = (t + 1) * fc.blocks * self_block_flops(d, f, t + 1, rotary=True) + _mm(m.sum_in)
    heads = _mm(m.stage_head, m.hazard_head, m.value_head, m.reward_head, m.latent_head, m.hyp_head, m.summary_pred)
    steps = 0
    for k in range(1, k_h + 1):
        steps += (_mm(m.tech_head)                                   # pi(tech | s) on the route's state
                  + b * _mm(m.ptr_q) + c * _mm(m.ptr_k) + 2 * b * c * d  # pointer over the C entities
                  + b * (_mm(m.tgt_in)                               # action input of each candidate
                         + fc.blocks * self_block_flops(d, f, t + k + 1, rotary=True)  # prefix, k - 1 steps, itself
                         + heads + exposure_flops(cfg)))
    return {"Context and prefix": encode + prefix, "Routes": n * steps}


def verifier_flops(cfg: NagaHanaConfig = CFG, op: OperatingPoint = OP) -> int:
    """The Verifier's process reward of every imagined step: per route, [context ; K steps] densely."""
    m = built(cfg.name).verifier.prm
    v = cfg.verifier
    t = context_positions(cfg)
    per_route = ((t + op.k) * v.blocks * self_block_flops(v.dim, v.mlp_hidden, t + op.k, rotary=True)
                 + op.k * (_mm(m.state_in, m.tgt_in) + _mm(m.head)))
    return t * _mm(m.ctx_enc) + op.n * per_route


def memory_write_flops(cfg: NagaHanaConfig = CFG, op: OperatingPoint = OP) -> int:
    """The long-term memory write of one trigger: every active entity's memory-stream state projected to a key
    and value, the memory MLP forward and its closed-form gradient (Titans form, AS-155)."""
    m = built(cfg.name).longterm
    mem = cfg.memory
    return op.entities * (_mm(m.w_k, m.w_v) + 10 * mem.longterm_dim * mem.longterm_hidden)


def trigger_parts(cfg: NagaHanaConfig = CFG, op: OperatingPoint = OP) -> dict[str, int]:
    """One complete Forecaster trigger at the operating point."""
    return {"TAAFT": sum(taaft_parts(cfg, op).values()),
            "Imagination": sum(imagination_flops(cfg, op).values()),
            "Verifier": verifier_flops(cfg, op),
            "Memory write": memory_write_flops(cfg, op)}


def advisor_flops(cfg: NagaHanaConfig = CFG, op: OperatingPoint = OP) -> dict[str, int]:
    """One Advisor solve: the baseline and every evaluated sequence re-imagined with `advisor_rollouts` routes
    (1 + depth x W imaginations), plus the Advisor's own network over the beam search."""
    m = built(cfg.name).advisor
    a = cfg.advisor
    d, f = a.dim, a.mlp_hidden
    n_tok = taaft_tokens(cfg, op)
    per_imagination = sum(imagination_flops(cfg, op, routes=op.advisor_rollouts).values())
    imaginations = 1 + op.advisor_depth * op.advisor_width
    network = n_tok * _mm(m.mem_proj, m.read_proj) + op.entities * _mm(m.ptr_k) * op.advisor_depth
    for j in range(op.advisor_depth):
        rows = 1 if j == 0 else op.advisor_width
        length = j + 1
        block = (12 * d * d + 6 * d * f + 4 * (length + 1) * d + 4 * (n_tok + 1) * d)
        network += rows * (a.blocks * (n_tok * 4 * d * d + length * block) + _mm(m.action_head)
                           + op.advisor_width * _mm(m.ptr_q) + 2 * op.advisor_width * op.entities * d)
    return {"Re-imagination": imaginations * per_imagination, "Advisor network": network}


def cache_bytes_per_state(cfg: NagaHanaConfig = CFG) -> int:
    """Keys and values of every TSTCT block for one cached state (fp32, D-54)."""
    return 2 * cfg.tstct.blocks * cfg.tstct.dim * CACHE


def imagination_bytes_per_token(cfg: NagaHanaConfig = CFG) -> int:
    """Keys and values of every TAAFT block for one position at one trigger (the Imagination store; fp32, D-54)."""
    return 2 * cfg.taaft.blocks * cfg.taaft.dim * CACHE


def route_bytes_per_position(cfg: NagaHanaConfig = CFG) -> int:
    """Keys and values of every Forecaster block for one route position (fp32: inference runs without autocast)."""
    return 2 * cfg.forecaster.blocks * cfg.forecaster.dim * CACHE


def environment_bytes(cfg: NagaHanaConfig, slots: int) -> int:
    """The Environment cache of `slots` stored states (cell slots and latest-state registers), fp32."""
    return slots * cache_bytes_per_state(cfg)


def imagination_store_bytes(cfg: NagaHanaConfig, entities: int, triggers: int | None = None) -> int:
    """The Imagination store: the kept triggers x (active entities + adversary slots) x one position's K/V."""
    m_im = cfg.taaft.imagination_triggers if triggers is None else triggers
    return (entities + cfg.taaft.adversary_slots) * m_im * imagination_bytes_per_token(cfg)


def taaft_view_bytes(cfg: NagaHanaConfig, entities: int) -> int:
    """TAAFT's gathered cross keys of one block: every position's own and neighbour Environment rows (fp32)."""
    a = cfg.taaft
    return (entities + a.adversary_slots) * (a.own_states + a.neighbour_states) * 2 * a.dim * CACHE


def taaft_scores_bytes(cfg: NagaHanaConfig, entities: int) -> int:
    """TAAFT's self-attention scores of one block (three score maps of every head over the self keys, fp32)."""
    a = cfg.taaft
    n = entities + a.adversary_slots
    return 3 * a.heads * n * (n + a.imagination_triggers + 1) * 4


def route_cache_bytes(cfg: NagaHanaConfig, routes: int, horizon: int, lookahead: int) -> int:
    """The Forecaster's route caches with the lookahead copies: (1 + B) N (T + 1 + K) route positions (fp32)."""
    return (1 + lookahead) * routes * (context_positions(cfg) + 1 + horizon) * route_bytes_per_position(cfg)


def longterm_state_bytes(cfg: NagaHanaConfig) -> int:
    """The long-term memory's fast state for one site: W_1, W_2 and their momenta, fp32 (`memory/longterm.py`)."""
    return 4 * cfg.memory.longterm_dim * cfg.memory.longterm_hidden * FP32


def training_passes(cfg: NagaHanaConfig = CFG) -> tuple[float, float]:
    """(E[passes with gradient], E[passes without]) of a thinking stream in training: R = 1 + Poisson(mean),
    clipped to [1, max_passes]; gradients through the last `grad_passes` (AS-07)."""
    t = cfg.tstct
    lam, cap, g = t.train_passes_mean, t.max_passes, t.grad_passes
    with_grad = without = 0.0
    for j in range(0, 80):                             # Poisson mass beyond 80 is below 1e-60 at mean 3
        p = math.exp(-lam) * lam**j / math.factorial(j)
        r = min(1 + j, cap)
        with_grad += p * min(r, g)
        without += p * max(r - g, 0)
    return with_grad, without


def block_activation_bytes(seq: int, batch: int, d: int, f: int, *, taaft: bool = False) -> int:
    """Activations one block stores per pass in half precision (see the module docstring)."""
    return HALF * seq * batch * ((15 if taaft else 10) * d + 3 * f)


def activation_bytes(seq: int, batch: int, d: int, f: int, blocks: int, passes: int, *, taaft: bool = False,
                     recompute: bool = False, scores_keys: int = 0, heads: int = 0) -> int:
    """Training activations of a looped block stack: `passes` stored passes (memory stream + gradient passes).
    recompute: keep each block's input per pass and one block's activations. scores_keys: also store the
    attention probabilities (half precision) of every block over that many keys."""
    per_block = block_activation_bytes(seq, batch, d, f, taaft=taaft)
    scores = HALF * heads * seq * scores_keys * batch
    if recompute:
        return passes * blocks * HALF * seq * batch * d + per_block
    return passes * blocks * (per_block + scores)


def profile(cfg: NagaHanaConfig = CFG, op: OperatingPoint = OP) -> dict[str, float]:
    n_params = parameters(cfg.name)["total"]
    upd = update_parts(cfg, op)
    per_update = update_flops(cfg, op)
    trig = trigger_parts(cfg, op)
    per_trigger = sum(trig.values())
    adv = advisor_flops(cfg, op)
    states = op.entities * op.states_per_entity
    n_tok = taaft_tokens(cfg, op)
    a, fc = cfg.taaft, cfg.forecaster
    p: dict[str, float] = {
        "params": n_params,
        # Weights are stored and served in float32 (D-54); the half-precision size is a reference only.
        "weights_bytes": FP32 * n_params,
        "weights_fp32_bytes": FP32 * n_params,
        "weights_fp16_unused_bytes": HALF * n_params,
        "train_state_bytes": 16 * n_params,
        # context window
        "tstct_keys": tstct_keys(op),
        "tstct_padded_keys": tstct_padded_keys(op),
        "cached_states": states,
        "cache_bytes_per_state": cache_bytes_per_state(cfg),
        "working_cache_bytes": environment_bytes(cfg, states),
        "context_seconds_per_entity": states / (op.positions_per_update * op.rate),
        "retained_bytes_per_day": op.retained_bytes_per_update * op.rate * 86_400,
        "taaft_tokens": n_tok,
        "taaft_self_keys": taaft_self_keys(cfg, op),
        "taaft_cross_keys": taaft_cross_keys(cfg),
        "imagination_store_bytes": imagination_store_bytes(cfg, op.entities),
        "taaft_view_bytes": taaft_view_bytes(cfg, op.entities),
        "taaft_scores_bytes": taaft_scores_bytes(cfg, op.entities),
        "route_cache_bytes": route_cache_bytes(cfg, op.n, op.k, op.lookahead),
        # per state update
        "update_input_flops": upd["Input layer"],
        "update_cvgae_flops": upd["CVG-AE"],
        "update_tstct_flops": upd["TSTCT"],
        "update_thinking_flops": upd["TSTCT thinking"],
        "update_flops": per_update,
        "update_flops_per_param": per_update / n_params,
        "sustained_flops": per_update * op.rate,
        # per Forecaster trigger
        "trigger_taaft_flops": trig["TAAFT"],
        "trigger_imagination_flops": trig["Imagination"],
        "trigger_verifier_flops": trig["Verifier"],
        "trigger_memory_flops": trig["Memory write"],
        "trigger_flops": per_trigger,
        "taaft_pass_flops": n_tok * a.blocks * taaft_block_flops(a.dim, a.mlp_hidden, taaft_self_keys(cfg, op),
                                                                 taaft_cross_keys(cfg), kv=False),
        "trigger_compute_s": per_trigger / op.server_flops,
        "advisor_flops": sum(adv.values()),
        "advisor_reimagination_flops": adv["Re-imagination"],
        "server_load": (per_update * op.rate + per_trigger / op.window_s) / op.server_flops,
    }
    p["inference_bytes"] = (p["weights_bytes"] + p["working_cache_bytes"] + p["imagination_store_bytes"]
                            + p["taaft_view_bytes"] + p["taaft_scores_bytes"] + p["route_cache_bytes"])
    # The accelerators the inference memory needs (memory, not compute, sets the server).
    p["accelerators_needed"] = math.ceil(p["inference_bytes"] / op.accelerator_bytes)
    # Training: activations per micro-batch (memory stream + gradient passes stored).
    s, b = op.train_seq, op.micro_batch
    t = cfg.tstct
    stored = 1 + t.grad_passes
    p["act_tstct_bytes"] = activation_bytes(s, b, t.dim, t.mlp_hidden, t.blocks, stored)
    p["act_tstct_scores_bytes"] = activation_bytes(s, b, t.dim, t.mlp_hidden, t.blocks, stored, scores_keys=s,
                                                   heads=t.heads)
    p["act_tstct_recompute_bytes"] = activation_bytes(s, b, t.dim, t.mlp_hidden, t.blocks, stored, recompute=True)
    p["act_taaft_bytes"] = activation_bytes(s, b, a.dim, a.mlp_hidden, a.blocks, 1 + a.grad_passes, taaft=True)
    p["act_taaft_recompute_bytes"] = activation_bytes(s, b, a.dim, a.mlp_hidden, a.blocks, 1 + a.grad_passes,
                                                      taaft=True, recompute=True)
    # Training compute per stage (forward + backward = 3 forwards; thinking passes without gradient: 1).
    g_with, g_without = training_passes(cfg)
    tst = tstct_position_parts(cfg, op)
    # training runs TSTCT densely over the window (masked attention over all its states; carry left out)
    tst_pass = t.blocks * self_block_flops(t.dim, t.mlp_hidden, s, kv=False, rotary=True)
    tst_mem = t.blocks * self_block_flops(t.dim, t.mlp_hidden, s, kv=True, rotary=True)
    tst_fixed = tst["input"] + tst["gate"] + tst["prior"]
    pos = op.positions_per_update
    perceive_train = (3 * (upd["Input layer"] + upd["CVG-AE"] + pos * (tst_fixed + tst_mem + decoder_row_flops(cfg)))
                      + pos * (3 * g_with + g_without) * tst_pass)
    perceive_fwd = upd["Input layer"] + upd["CVG-AE"] + pos * (tst_fixed + tst_mem + (g_with + g_without) * tst_pass)
    s_self, s_cross = taaft_self_keys(cfg, op), taaft_cross_keys(cfg)
    tp = taaft_parts(cfg, op)
    per_token_other = (tp["Inputs, biases and readouts"] + tp["Lens energies and descent"]) / n_tok
    taaft_token = (3 * a.blocks * taaft_block_flops(a.dim, a.mlp_hidden, s_self, s_cross, kv=True)
                   + (3 * g_with + g_without) * a.blocks * taaft_block_flops(a.dim, a.mlp_hidden, s_self, s_cross,
                                                                              kv=False)
                   + 3 * per_token_other)
    data = op.corpus_updates + op.generator_updates
    tpos = context_positions(cfg) + 1 + op.k
    # an imagined state in full training: one teacher-forced Forecaster position and one process-reward position
    imagined = 3 * (fc.blocks * self_block_flops(fc.dim, fc.mlp_hidden, tpos, rotary=True)
                    + cfg.verifier.blocks * self_block_flops(cfg.verifier.dim, cfg.verifier.mlp_hidden, tpos, rotary=True))
    stage3 = perceive_train * data * op.stage3_passes
    stage4 = (perceive_fwd + pos * taaft_token) * data * op.stage4_passes
    stage5 = (perceive_train + pos * taaft_token) * data * op.stage5_passes + imagined * op.stage5_imagined
    p.update(train_stage3_flops=stage3, train_stage4_flops=stage4, train_stage5_flops=stage5,
             train_flops=stage3 + stage4 + stage5,
             train_positions=data * (op.stage3_passes + op.stage4_passes + op.stage5_passes),
             train_passes_with_grad=g_with, train_passes_without=g_without)
    p["train_gpu_hours"] = p["train_flops"] / (op.gpu_peak * op.mfu) / 3600
    return p


def _gib(x: float) -> str:
    return f"{x / GIB:,.2f} GiB" if x >= GIB else f"{x / MIB:,.0f} MiB"


def _flops(x: float) -> str:
    for unit, scale in (("PFLOP", 1e15), ("TFLOP", 1e12), ("GFLOP", 1e9), ("MFLOP", 1e6)):
        if x >= scale:
            return f"{x / scale:,.2f} {unit}"
    return f"{x:,.0f} FLOP"


def markdown(cfg: NagaHanaConfig = CFG, op: OperatingPoint = OP) -> str:
    """The profile as the Markdown table of docs/sizing.md."""
    p = profile(cfg, op)
    a = cfg.taaft
    rows = [
        ("**Weights and state**", ""),
        ("Parameters (Generator excluded; built model, meta device)", f"{int(p['params']):,}"),
        ("Weights, stored and served in single precision (fp32, 4 B per parameter, D-54)",
         f"{_gib(p['weights_bytes'])} ({p['weights_bytes'] / 1e9:.2f} GB)"),
        ("Half precision for reference only (not used: no 16-bit weights, D-54)", _gib(p["weights_fp16_unused_bytes"])),
        ("Training state (mixed-precision Adam, 16 B per parameter)", _gib(p["train_state_bytes"])),
        ("**Context window**", ""),
        ("TSTCT keys per query (spatial + temporal + causal)",
         f"{p['tstct_keys']} ({op.spatial_keys} + {op.states_per_entity} + {op.causal_keys})"),
        ("Working context", f"{op.entities:,} entities x {op.states_per_entity} states = {p['cached_states']:,} cached states"),
        ("Environment cache per cached state (fp32, D-54)", f"{p['cache_bytes_per_state'] // KIB} KiB"),
        ("Working Environment cache", _gib(p["working_cache_bytes"])),
        ("Time an average entity's 512 states cover at the sustained rate", f"{p['context_seconds_per_entity']:,.0f} s"),
        (f"TAAFT positions; self keys; cross keys ({a.own_states} own + {a.neighbour_states} neighbours + {a.memory_probes} memory)",
         f"{p['taaft_tokens']:,}; {p['taaft_self_keys']:,}; {p['taaft_cross_keys']}"),
        (f"Imagination store ({a.imagination_triggers} triggers x {p['taaft_tokens']:,} positions, "
         f"{imagination_bytes_per_token(cfg) // KIB} KiB each)", _gib(p["imagination_store_bytes"])),
        ("TAAFT working set of one block (gathered cross keys; self-attention scores)",
         f"{_gib(p['taaft_view_bytes'])}; {_gib(p['taaft_scores_bytes'])}"),
        (f"Forecaster route caches with the {op.lookahead} lookahead copies", _gib(p["route_cache_bytes"])),
        ("Inference memory in all", f"{_gib(p['inference_bytes'])} ({p['inference_bytes'] / 1e9:.1f} GB)"),
        (f"{op.accelerator_bytes / 1e9:.0f} GB accelerators it needs (server)",
         f"{p['accelerators_needed']} ({p['accelerators_needed'] * op.accelerator_bytes / 1e9:.0f} GB)"),
        ("Retained Environment per day (24 B per update)", _gib(p["retained_bytes_per_day"])),
        ("**Per state update**", ""),
        ("Input layer (canonical columns)", _flops(p["update_input_flops"])),
        (f"CVG-AE ({op.nodes_per_update} nodes, {op.incidences} incidences per node and plane)", _flops(p["update_cvgae_flops"])),
        (f"TSTCT (2 positions x 16 blocks, memory stream + R {op.r} thinking passes)", _flops(p["update_tstct_flops"])),
        ("Of which the thinking passes", _flops(p["update_thinking_flops"])),
        ("Total, and per parameter", f"{_flops(p['update_flops'])} ({p['update_flops_per_param']:.2f} FLOP per parameter)"),
        (f"At {op.rate:,.0f} updates/s", f"{p['sustained_flops'] / 1e12:,.1f} TFLOP/s"),
        (f"**Per Forecaster trigger** (K {op.k}, N {op.n}, R {op.r}, S {op.descent_steps}, B {op.lookahead})", ""),
        (f"TAAFT over {p['taaft_tokens']:,} positions (34 blocks, memory stream + R thinking passes, descent)",
         _flops(p["trigger_taaft_flops"])),
        (f"Imagination: N K = {op.n * op.k:,} route steps, {op.lookahead} candidates each", _flops(p["trigger_imagination_flops"])),
        ("Verifier (process reward of every imagined step)", _flops(p["trigger_verifier_flops"])),
        ("Long-term memory write", _flops(p["trigger_memory_flops"])),
        ("Total, and compute time at 1.2e14 FLOP/s", f"{_flops(p['trigger_flops'])} ({p['trigger_compute_s']:.2f} s)"),
        (f"Advisor solve, on demand (W {op.advisor_width}, depth {op.advisor_depth}, {op.advisor_rollouts} rollouts)",
         _flops(p["advisor_flops"])),
        ("Server load: updates plus one trigger per 60 s", f"{100 * p['server_load']:.0f} % of 1.2e14 FLOP/s"),
        ("**Training**", ""),
        ("Activations per micro-batch (8 x 2,048 states, 3 stored passes): TSTCT / TAAFT",
         f"{_gib(p['act_tstct_bytes'])} / {_gib(p['act_taaft_bytes'])}"),
        ("With attention scores stored (TSTCT, dense over the window)", _gib(p["act_tstct_scores_bytes"])),
        ("With full recomputation: TSTCT / TAAFT",
         f"{_gib(p['act_tstct_recompute_bytes'])} / {_gib(p['act_taaft_recompute_bytes'])}"),
        ("Stage 1 / 2 / 3 compute (D-62 numbering)",
         f"{p['train_stage3_flops']:.2e} / {p['train_stage4_flops']:.2e} / {p['train_stage5_flops']:.2e} FLOP"),
        ("Total training compute", f"{p['train_flops']:.2e} FLOP over {p['train_positions']:.1e} state updates"),
        ("GPU time at 40 % of an H100's 989 TFLOP/s", f"{p['train_gpu_hours']:,.0f} GPU-hours"),
    ]
    out = ["| Figure | Value |", "|---|---:|"]
    out += [f"| {a_} | {b_} |" if b_ else f"| {a_} | |" for a_, b_ in rows]
    return "\n".join(out)


def spec_values(cfg: NagaHanaConfig = CFG, op: OperatingPoint = OP) -> dict[str, str]:
    """The integers the thesis generator copies (latexdocs/full-docs/tools/results_spec.py, COMPUTE)."""
    p = profile(cfg, op)
    keys = ("params", "taaft_view_bytes", "imagination_store_bytes", "taaft_scores_bytes", "route_cache_bytes",
            "update_input_flops", "update_cvgae_flops", "update_tstct_flops", "update_thinking_flops",
            "trigger_taaft_flops", "taaft_pass_flops", "trigger_imagination_flops", "trigger_verifier_flops",
            "trigger_memory_flops",
            "advisor_flops", "act_tstct_bytes", "act_tstct_scores_bytes", "act_tstct_recompute_bytes",
            "act_taaft_bytes", "act_taaft_recompute_bytes", "train_stage3_flops", "train_stage4_flops",
            "train_stage5_flops", "train_positions")
    return {k: str(round(p[k])) for k in keys}


if __name__ == "__main__":
    import io
    import sys

    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8")
    if sys.argv[1:] == ["--spec"]:
        for key, val in spec_values().items():
            print(f'"{key}": "{val}",')
    else:
        print(markdown())
