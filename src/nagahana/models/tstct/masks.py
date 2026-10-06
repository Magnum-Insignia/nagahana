"""Attention patterns of TSTCT's three head types, as of each state's own time (build-spec §2.5).

Purpose
-------
Every TSTCT head reads only what existed when its query state happened. This module turns a window's
positions and contact matrices into the boolean patterns and the bias *features* of each head group.
The learned biases themselves (hop, plane, age, log-Δt, causal gate) are parameters of the model
(`model.py`); keeping them out of here keeps this module parameter-free and testable against a
brute-force reference (`tests/test_tstct_masks.py`).

Owner sources: diagram 03 (spatial, temporal, causal heads; corrected 2026-09-29); [A-10] (the
"Topological" in TSTCT). Decisions: D-49 (time, never index, sets positions; Δt buckets). Assumptions:
AS-09 (head split), AS-10 (causal candidates and gates), AS-151 (causal candidate pre-cap), AS-160
(times within `time_tie_s` = 100 ns are ties in every as-of and strict-order test),
AS-156 ("as of" = position order, below).

"As of" (AS-156)
----------------
Positions are sorted by (time, update, role) (`models/batch.PositionBatch`), so position order is a
valid time order. "State j exists as of state i" is defined as **j ≤ i in position order** (hence
t_j ≤ t_i). With ties in time (the initiator and responder states of one update share t) this is the
order in which the incremental path (`TSTCT.step`) writes states, which is what makes the dense
training path and the cached inference path compute the same function.

The patterns (for a query position i with entity e_i at time t_i)
-----------------------------------------------------------------
- **temporal** (same entity, own past):  e_j = e_i, j ≤ i, and j among the last `temporal_window`
  states of e_i up to i (self included).
- **spatial** (topology):  i itself (hop 0), plus for every other entity v with
      hop(e_i, v; t_i) = 1 if C¹[e_i, v] ≤ t_i,  2 if C²[e_i, v] ≤ t_i (and not 1),
  the *latest* state of v as of i (the j ≤ i with e_j = v and no later state of v up to i), capped
  to the `spatial_keys` most recent such neighbours. Features: hop, plane bits
  Π_p = [C¹_p[e_i, v] ≤ t_i] (all False for the self key: hop 0 is its own category), age t_i − t_j.
- **causal candidates** (Granger-style lagged influence, AS-10):  e_j ≠ e_i, t_j < t_i strictly,
  t_i − t_j ≤ Λ_lag, C¹[e_i, e_j] ≤ t_i (direct contact by then), capped to the
  `causal_candidate_cap` most recent (AS-151). The gate and the top-`causal_keys` selection are
  learned and applied by the model; here only the candidate set is built.

Carried memory (D-51)
---------------------
With `carried` keys (states of earlier windows, Transformer-XL style, Dai et al., ACL 2019,
arXiv:1901.02860), the key axis is [C carried ; P current] and the same rules apply across the
boundary: temporal heads read the entity's carried cell slots; spatial heads read a neighbour's
carried latest-state register while it has no state in this window before t_i; causal heads read
carried cell slots within Λ_lag. Carried slots precede every position in key order.

No future leakage: every pattern requires j ≤ i (temporal, spatial) or t_j < t_i (causal), and every
contact test is "as of t_i". Padded positions (mask False) are neither queries nor keys; their rows
allow nothing (the attention's learned null key absorbs them, so no NaN).

Invariants
----------
- Masks are identical for every loop pass and every block (built once per forward).
- `dt` is t_i − t_j in float64 and 0 wherever i or j is padding (mask before compute).

Extension points
----------------
- Further spatial features (hyperedge kind of the contact) add fields to `TSTCTMasks`.
- Sparse (CSR) patterns for very long windows can replace the dense [B, P, P] tensors.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from nagahana.core.errors import InvariantViolation
from nagahana.models.batch import PositionBatch
from nagahana.models.config.components import TSTCTConfig

#: Head-group codes (order of the heads: spatial first, then temporal, then causal; AS-09).
SPATIAL, TEMPORAL, CAUSAL = 0, 1, 2


def head_groups(cfg: TSTCTConfig) -> torch.Tensor:
    """Group code per head, long [H]: spatial heads first, then temporal, then causal (AS-09)."""
    if cfg.spatial_heads + cfg.temporal_heads + cfg.causal_heads != cfg.heads:
        raise InvariantViolation("spatial + temporal + causal heads must equal heads")
    return torch.tensor([SPATIAL] * cfg.spatial_heads + [TEMPORAL] * cfg.temporal_heads + [CAUSAL] * cfg.causal_heads)


@dataclass
class CarriedKeys:
    """Key-side description of carried slots (D-51): states of earlier windows read as memory.

    C carried slots per window, sorted by (time, write order) and all no later than the window's first
    position (stream order; enforced by the sortedness check).
    entity: long [B, C]: index in the *current* window's entity table (−1 = entity not in this window:
        the slot is unusable). time: float64 [B, C] relative to the current window origin (≤ 0 usually).
    mask: bool [B, C] real slots. bucket: bool [B, C] the slot is a (possibly merged) cell slot: usable
        by temporal and causal heads. latest: bool [B, C] the slot is an entity's latest-state register
        (AS-154): usable by spatial heads only (it duplicates the newest cell slot, so the other head
        types must not read it twice).
    """

    entity: torch.Tensor
    time: torch.Tensor
    mask: torch.Tensor
    bucket: torch.Tensor
    latest: torch.Tensor


@dataclass
class TSTCTMasks:
    """Patterns and bias features of one batch of windows (see the module docstring).

    The key axis has K = C + P entries: the C carried slots first (D-51; C = 0 without carry), then the
    P positions of the window. Queries are the P positions (query i is key C + i).
    allowed: bool [B, H, P, K] per head (causal heads: the *candidates*, before the gate's top-k).
    spatial, temporal, causal_candidates: bool [B, P, K].
    hop: long [B, P, K]: 0 self, 1 or 2 for spatial neighbours, −1 elsewhere.
    planes: bool [B, P, K, n_planes]: contact planes as of t_i (spatial keys only).
    dt: float64 [B, P, K]: t_i − t_j (0 where i or j is padding).
    groups: long [H] head-group codes. carried: C.
    """

    allowed: torch.Tensor
    spatial: torch.Tensor
    temporal: torch.Tensor
    causal_candidates: torch.Tensor
    hop: torch.Tensor
    planes: torch.Tensor
    dt: torch.Tensor
    groups: torch.Tensor
    carried: int = 0


def _keep_most_recent(cand: torch.Tensor, k: int) -> torch.Tensor:
    """Keep, per row i, the k candidates with the largest key index j (most recent). cand [B, P, K]."""
    p = cand.shape[-1]
    if k >= p:
        return cand
    idx = torch.arange(p, device=cand.device)
    score = torch.where(cand, idx.expand_as(cand), torch.full_like(cand, -1, dtype=torch.long))
    top = torch.topk(score, k, dim=-1).indices                     # [B, P, k]
    keep = torch.zeros_like(cand)
    keep.scatter_(-1, top, True)
    return keep & cand


def build_dense_masks(
    positions: PositionBatch,
    contact1: torch.Tensor,
    contact2: torch.Tensor,
    contact_planes: torch.Tensor,
    cfg: TSTCTConfig,
    carried: CarriedKeys | None = None,
) -> TSTCTMasks:
    """Patterns and features for B windows (see the module docstring), optionally over carried keys.

    positions: entity/time/mask [B, P]; contact1, contact2: float64 [B, V, V]; contact_planes:
    float64 [B, V, V, n_planes] (+inf = never). With carry, the contact matrices must be *cumulative
    over the stream* (first contact ever, relative to this window's origin, negative if before it) and
    cover every carried entity (D-51; see `model.CarriedEnvironment`).
    """
    q_ent, q_time, q_valid = positions.entity, positions.time.to(torch.float64), positions.mask.bool()
    b_, p_ = q_ent.shape
    dev = q_ent.device
    # ---------------------------------------------------------------- key table [C carried ; P current]
    ones = torch.ones_like(q_valid)
    if carried is None:
        c_ = 0
        k_ent, k_time, k_valid = q_ent, q_time, q_valid
        temp_ok = spat_ok = ones
    else:
        c_ = int(carried.entity.shape[1])
        k_ent = torch.cat([carried.entity, q_ent], dim=1)
        k_time = torch.cat([carried.time.to(torch.float64), q_time], dim=1)
        k_valid = torch.cat([carried.mask.bool() & (carried.entity >= 0), q_valid], dim=1)
        temp_ok = torch.cat([carried.bucket.bool(), ones], dim=1)    # temporal and causal heads
        spat_ok = torch.cat([carried.latest.bool(), ones], dim=1)    # spatial heads
    k_ = c_ + p_
    # Sortedness is what makes key order a time order (AS-156); refuse anything else.
    t_real = torch.where(k_valid, k_time, torch.full_like(k_time, -torch.inf))
    run_max = torch.cummax(t_real, dim=1).values
    if bool(((t_real < run_max - cfg.time_tie_s) & k_valid).any()):
        raise InvariantViolation("positions (after carried slots) must be sorted by time (PositionBatch contract, D-51)")

    e_k = k_ent.clamp_min(0)                                         # padded entity −1 → 0 (masked below)
    t_k = torch.where(k_valid, k_time, torch.zeros_like(k_time))     # padded time → 0 (masked below)
    e_q, t_q = e_k[:, c_:], t_k[:, c_:]
    pair = q_valid[:, :, None] & k_valid[:, None, :]                 # [B, P, K] both real
    kidx = torch.arange(k_, device=dev)
    qidx = c_ + torch.arange(p_, device=dev)
    prec = (kidx[None, :] <= qidx[:, None])[None]                    # [1, P, K] key j ≤ query i
    same = (e_q[:, :, None] == e_k[:, None, :]) & pair               # [B, P, K] same entity
    dt = torch.where(pair, t_q[:, :, None] - t_k[:, None, :], torch.zeros((), dtype=torch.float64, device=dev))
    same_kk = (e_k[:, :, None] == e_k[:, None, :]) & k_valid[:, :, None] & k_valid[:, None, :]   # [B, K, K]

    # ---------------------------------------------------------------- temporal: own last W states
    # rank_j = temporal-eligible same-entity keys up to and including j; keep rank_i − rank_j < W.
    tri = (kidx[None, :] <= kidx[:, None])[None]                     # [1, K, K] j' ≤ j
    rank = (same_kk & tri & temp_ok[:, None, :]).sum(dim=-1)         # [B, K]
    q_rank = rank[:, c_:]
    temporal = same & prec & temp_ok[:, None, :] & ((q_rank[:, :, None] - rank[:, None, :]) < cfg.temporal_window)

    # ---------------------------------------------------------------- contacts as of t_i
    bidx = torch.arange(b_, device=dev)[:, None, None]
    c1 = contact1.to(torch.float64)[bidx, e_q[:, :, None], e_k[:, None, :]]   # [B, P, K] C¹[e_i, e_j]
    c2 = contact2.to(torch.float64)[bidx, e_q[:, :, None], e_k[:, None, :]]   # [B, P, K] C²[e_i, e_j]
    ti = t_q[:, :, None]
    hop1 = c1 <= ti + cfg.time_tie_s                                 # as of t_i (ties within time_tie_s, AS-160)
    hop2 = (c2 <= ti + cfg.time_tie_s) & ~hop1
    other = pair & ~same

    # ---------------------------------------------------------------- spatial: neighbours' latest state
    # Among spatial-eligible keys, next_same[j] = the next one of the same entity after j (K if none);
    # j is "latest as of i" iff j ≤ i < next_same[j]. A carried register is therefore read only while its
    # entity has no state in this window before i.
    after = same_kk & (kidx[None, None, :] > kidx[None, :, None]) & spat_ok[:, None, :]
    next_same = torch.where(after, kidx.expand(b_, k_, k_), torch.full_like(after, k_, dtype=torch.long)).min(dim=-1).values
    latest = prec & (next_same[:, None, :] > qidx[None, :, None]) & spat_ok[:, None, :]
    neighbours = other & latest & (hop1 | hop2)
    neighbours = _keep_most_recent(neighbours, cfg.spatial_keys)
    eye = (kidx[None, :] == qidx[:, None])[None] & q_valid[:, :, None]
    spatial = neighbours | eye
    hop = torch.full((b_, p_, k_), -1, dtype=torch.long, device=dev)
    hop = torch.where(neighbours & hop1, torch.ones_like(hop), hop)
    hop = torch.where(neighbours & hop2, torch.full_like(hop, 2), hop)
    hop = torch.where(eye, torch.zeros_like(hop), hop)
    cp = contact_planes.to(torch.float64)[bidx, e_q[:, :, None], e_k[:, None, :]]  # [B, P, K, n_planes]
    planes = (cp <= ti[..., None] + cfg.time_tie_s) & neighbours[..., None]

    # ---------------------------------------------------------------- causal candidates
    causal = other & hop1 & (dt > cfg.time_tie_s) & (dt <= cfg.causal_lag_s) & temp_ok[:, None, :]   # strictly earlier
    causal = _keep_most_recent(causal, cfg.causal_candidate_cap)

    # ---------------------------------------------------------------- per-head patterns
    groups = head_groups(cfg).to(dev)
    stacked = torch.stack([spatial, temporal, causal], dim=1)        # [B, 3, P, K]
    allowed = stacked[:, groups]                                     # [B, H, P, K]
    return TSTCTMasks(allowed=allowed, spatial=spatial, temporal=temporal, causal_candidates=causal,
                      hop=hop, planes=planes, dt=dt, groups=groups, carried=c_)


# ===================================================================================== 1-D helpers
def temporal_allowed(entity: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    """allowed[i, j] = (entity_i == entity_j) ∧ (t_j ≤ t_i). Shapes: [T] → [T, T]."""
    if entity.shape != time.shape or entity.dim() != 1:
        raise InvariantViolation("entity and time must be 1-D and of equal length")
    same = entity[:, None] == entity[None, :]
    past_or_now = time[None, :] <= time[:, None]
    return same & past_or_now


def spatial_allowed(adjacency: torch.Tensor) -> torch.Tensor:
    """allowed = adjacency ∨ I, for entity states as of one time t. Shape [V, V] (bool)."""
    if adjacency.dim() != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise InvariantViolation("adjacency must be square")
    eye = torch.eye(adjacency.shape[0], dtype=torch.bool, device=adjacency.device)
    return adjacency.bool() | eye


def causal_allowed(entity: torch.Tensor, time: torch.Tensor, contact1: torch.Tensor, lag_s: float) -> torch.Tensor:
    """Causal candidates of one window, uncapped: [T] entity/time + C¹ [V, V] → bool [T, T].

    allowed[i, j] = e_j ≠ e_i ∧ t_j < t_i ∧ t_i − t_j ≤ Λ_lag ∧ C¹[e_i, e_j] ≤ t_i (AS-10). The gate
    and its top-k are learned in `model.py`; this is the set they choose from.
    """
    if entity.shape != time.shape or entity.dim() != 1:
        raise InvariantViolation("entity and time must be 1-D and of equal length")
    t = time.to(torch.float64)
    dt = t[:, None] - t[None, :]
    contact = contact1.to(torch.float64)[entity[:, None], entity[None, :]] <= t[:, None]
    return (entity[:, None] != entity[None, :]) & (dt > 0) & (dt <= lag_s) & contact


def to_additive(allowed: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """0 where allowed, −∞ elsewhere. Every row must allow at least one position."""
    if not bool(allowed.any(dim=-1).all()):
        raise InvariantViolation("a query row allows no key; softmax would be NaN")
    out = torch.zeros(allowed.shape, dtype=dtype, device=allowed.device)
    return out.masked_fill(~allowed, float("-inf"))
