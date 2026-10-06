"""As-of structure for TAAFT: what each trigger may read, built only from facts with time ≤ τ.

Purpose
-------
TAAFT analyses the network at a trigger time τ (build-spec §2.7). Everything it reads must be as of
τ (build-spec §2.2 "as-of discipline"; batch.py "no future leakage is the builder's duty"). This
module turns the window's time-stamped facts into per-trigger structure:

- `contacts_as_of`: hop-1 / hop-2 contact and per-plane contact between entities as of τ;
- `select_cross_keys`: which TSTCT positions each entity token reads in cross-attention (its own
  last states and its neighbours' latest states, AS-13, AS-37, AS-201);
- `aggregate_causal_gates`: TSTCT's Granger-style gates g_ij summed into a directed entity-pair
  weight ḡ_{u→v} as of τ (the cause lens's weights, AS-208);
- `segmented_cumsum` / `group_prev`: per-entity prefix sums over positions, the primitive that makes
  every "as of τ" statistic a single gather at the entity's latest position ≤ τ.

Owner sources: [A-12], [A-14], [A-19]. Decisions: D-35 (Environment holds facts; TAAFT only reads it),
D-49 (time, never index). Assumptions: AS-10, AS-13, AS-37, AS-201, AS-208.

Why prefix sums at the latest position are exactly "as of τ"
----------------------------------------------------------
Positions are sorted by time within a window (batch.py PositionBatch). For an entity v, its
positions in that order are its state history. `TriggerBatch.entity_latest[b, m, v]` is the index of
v's latest position with time ≤ τ_m. Any statistic S that is a sum over v's events,

    S_v(τ) = Σ_{p : entity(p) = v, t_p ≤ τ} s_p,

is therefore the inclusive prefix sum of s over v's positions, read at `entity_latest`. Nothing at a
later position can enter it. The no-future-leak tests perturb every later position and check that
earlier triggers are bit-for-bit unchanged (tests/test_taaft_model.py).

Invariants
----------
- No function here reads a position with time > τ for trigger τ, or a contact time > τ.
- Masked positions (mask False or entity < 0) never contribute (they form their own dummy group).
- Causal gates are additionally masked to strict t_j < t_i (the Granger candidate set), so a gate
  that broke TSTCT's contract still cannot leak a later position into an earlier trigger.

Extension points
----------------
- Neighbour ranking (hop-1 first, then hop-2, each by recency of the neighbour's latest state) is one
  function; a learned or plane-aware ranking can replace it.
"""

from __future__ import annotations

import torch

#: Offset that ranks every hop-1 neighbour before every hop-2 neighbour (seconds ≫ any window span).
_HOP_RANK = 1.0e12


# ============================================================================ grouping primitives
def group_sort(group: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Stable order that groups positions by `group` while keeping time order inside a group.

    group: long [B, P] (non-negative; masked positions should carry a dummy group id).
    Returns (order [B, P], inverse [B, P]) with x_sorted = x.gather(1, order), x = x_sorted.gather(1, inverse).
    """
    p = group.shape[1]
    # key = group·P + index: distinct, so a plain sort is stable in index order within a group.
    key = group * p + torch.arange(p, device=group.device)[None, :]
    order = torch.argsort(key, dim=1)
    inverse = torch.argsort(order, dim=1)
    return order, inverse


def _expand_index(index: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    # index [B, P] → [B, P, *rest] matching `like` [B, P, *rest] for gather along dim 1.
    shape = index.shape + (1,) * (like.dim() - 2)
    return index.view(shape).expand(*index.shape, *like.shape[2:])


def segmented_cumsum(x: torch.Tensor, group: torch.Tensor) -> torch.Tensor:
    """Inclusive prefix sums of `x` within each group, in position (time) order.

    x: [B, P, ...] any float dtype; group: long [B, P]. out[b, p] = Σ_{q ≤ p, group(q) = group(p)} x[b, q].

    Maths: sort positions by (group, index); take the running sum cs; subtract the running sum just
    before the group's first element (found with a cummax over group-start indices).
    """
    order, inverse = group_sort(group)
    xs = x.gather(1, _expand_index(order, x))                                  # [B, P, ...] grouped
    gs = group.gather(1, order)                                                # [B, P]
    cs = xs.cumsum(dim=1)
    # Start flags of each group in sorted order, and the index of the current group's start.
    start = torch.ones_like(gs, dtype=torch.bool)
    start[:, 1:] = gs[:, 1:] != gs[:, :-1]
    ar = torch.arange(gs.shape[1], device=gs.device)[None, :].expand_as(gs)
    first = torch.where(start, ar, torch.zeros_like(ar)).cummax(dim=1).values  # [B, P]
    base = (cs - xs).gather(1, _expand_index(first, cs))                       # sum before the group
    return (cs - base).gather(1, _expand_index(inverse, xs))


def group_prev(group: torch.Tensor) -> torch.Tensor:
    """Index of the previous position of the same group (−1 for the first). group: long [B, P] → [B, P]."""
    order, inverse = group_sort(group)
    gs = group.gather(1, order)
    prev_sorted = torch.full_like(order, -1)
    same = gs[:, 1:] == gs[:, :-1]
    prev_sorted[:, 1:] = torch.where(same, order[:, :-1], torch.full_like(order[:, :-1], -1))
    return prev_sorted.gather(1, inverse)


def event_groups(entity: torch.Tensor, mask: torch.Tensor, n_entities: int) -> torch.Tensor:
    """Group id per position: the entity, or `n_entities` (a dummy group) for masked positions."""
    ok = mask & (entity >= 0)
    return torch.where(ok, entity, torch.full_like(entity, n_entities))


def gather_rows(table: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """Read rows of a per-position table at window-local indices.

    table: [B, P, ...]; index: long [B, *I] with −1 = none (reads row 0; caller masks).
    Returns [B, *I, ...].
    """
    b = table.shape[0]
    rest = table.shape[2:]
    flat = index.clamp_min(0).reshape(b, -1)                                   # [B, I]
    idx = flat.view(*flat.shape, *(1,) * len(rest)).expand(*flat.shape, *rest)
    return table.gather(1, idx).reshape(*index.shape, *rest)


# ============================================================================ contacts as of τ
def contacts_as_of(
    contact1: torch.Tensor, contact2: torch.Tensor, contact_planes: torch.Tensor, tau: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Contact structure as of τ (build-spec §2.2 contact matrices).

    contact1, contact2: float64 [B, V, V]; contact_planes: float64 [B, V, V, n_planes]; tau: float64 [B].
    Returns
      hop1 [B, V, V] bool: C¹[u, v] ≤ τ, u ≠ v;
      hop2 [B, V, V] bool: C²[u, v] ≤ τ, not hop1, u ≠ v;
      planes [B, V, V, n_planes] bool: first contact on plane p ≤ τ, u ≠ v.
    +inf ("never") compares False, so never-contacted pairs are excluded without special cases.
    """
    v = contact1.shape[-1]
    off = ~torch.eye(v, dtype=torch.bool, device=contact1.device)[None]       # [1, V, V]
    t = tau.to(torch.float64)[:, None, None]
    hop1 = (contact1 <= t) & off
    hop2 = (contact2 <= t) & off & ~hop1
    planes = (contact_planes <= t[..., None]) & off[..., None]
    return hop1, hop2, planes


# ============================================================================ cross-attention keys
def select_cross_keys(
    latest: torch.Tensor,
    prev: torch.Tensor,
    pos_time: torch.Tensor,
    hop1: torch.Tensor,
    hop2: torch.Tensor,
    active: torch.Tensor,
    readable: torch.Tensor,
    *,
    own_states: int,
    neighbour_states: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Which Environment positions each entity token reads at one trigger (AS-13, AS-37, AS-201).

    latest: long [B, V] entity's latest position ≤ τ (−1 none); prev: long [B, P] same entity's
    previous position (−1 none); pos_time: float64 [B, P]; hop1, hop2: bool [B, V, V] as of τ;
    active: bool [B, V] entity tokens active at this trigger; readable: bool [B, V] entities whose
    states may be read (False for entities hidden by the masked-entity objective).

    Returns (index [B, V, T_k] position or −1, kind [B, V, T_k] 0 = own, 1 = hop-1 neighbour,
    2 = hop-2 neighbour, neighbour [B, V, T_k] neighbour entity or −1), T_k = own + neighbour states.

    Own states: the chain latest → prev → prev … (all ≤ τ because latest ≤ τ and prev goes back).
    Neighbours: entities u ≠ v with hop-1 or hop-2 contact as of τ, readable and active, ranked
    hop-1 before hop-2 and, within a hop, by the time of u's latest state (most recent first). Each
    contributes its latest position ≤ τ.
    """
    b, v = latest.shape
    dev = latest.device
    own_ok = active & readable                                                     # [B, V]
    # ---- own chain: k = 0 is the latest state, k = 1 the one before, …
    own = torch.full((b, v, own_states), -1, dtype=torch.long, device=dev)
    cur = torch.where(own_ok, latest, torch.full_like(latest, -1))
    for k in range(own_states):
        own[:, :, k] = cur
        nxt = prev.gather(1, cur.clamp_min(0))
        cur = torch.where(cur >= 0, nxt, cur)
    # ---- neighbours: candidates, then top-k by (hop, recency)
    cand = (hop1 | hop2) & own_ok[:, None, :] & active[:, :, None]                 # [B, V(query), V(nbr)]
    t_latest = gather_rows(pos_time, latest)                                       # [B, V] float64
    score = t_latest[:, None, :] - hop2.to(torch.float64) * _HOP_RANK
    score = torch.where(cand, score, torch.full_like(score, float("-inf")))
    k_n = min(neighbour_states, v)
    top_val, top_u = torch.topk(score, k_n, dim=-1)                                # [B, V, k_n]
    ok = torch.isfinite(top_val)
    nbr = torch.where(ok, top_u, torch.full_like(top_u, -1))
    nbr_pos = torch.where(ok, latest.gather(1, top_u.reshape(b, -1)).reshape(b, v, k_n), torch.full_like(top_u, -1))
    nbr_hop2 = hop2.gather(2, top_u)
    nbr_kind = torch.where(nbr_hop2, torch.full_like(top_u, 2), torch.ones_like(top_u))
    if k_n < neighbour_states:                                                     # pad to the fixed width
        pad = neighbour_states - k_n
        nbr = torch.cat([nbr, nbr.new_full((b, v, pad), -1)], dim=-1)
        nbr_pos = torch.cat([nbr_pos, nbr_pos.new_full((b, v, pad), -1)], dim=-1)
        nbr_kind = torch.cat([nbr_kind, nbr_kind.new_ones((b, v, pad))], dim=-1)
    index = torch.cat([own, nbr_pos], dim=-1)                                      # [B, V, T_k]
    kind = torch.cat([torch.zeros_like(own), nbr_kind], dim=-1)
    neighbour = torch.cat([torch.full_like(own, -1), nbr], dim=-1)
    return index, kind, neighbour


# ============================================================================ causal gates → entity pairs
def causal_prefix(
    gate: torch.Tensor, entity: torch.Tensor, time: torch.Tensor, mask: torch.Tensor, n_entities: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-position prefix sums of causal gate mass by cause entity (window-level, computed once).

    gate: [B, H_c, P, P] TSTCT gate values g_ij (i = effect query, j = cause key, zero where not a
    candidate); entity: long [B, P]; time: float64 [B, P]; mask: bool [B, P].
    Returns (mass [B, P, V], count [B, P, V]): for position i of effect entity v, the sums over v's
    positions i' ≤ i of Σ_{j: entity(j) = u} g_{i'j} and of the number of candidate pairs (g > 0).

    Granger mask (AS-10, AS-208): only pairs with t_j < t_i strictly, both real positions.
    """
    g = gate.float().mean(dim=1)                                                   # [B, P, P] head mean
    real = mask & (entity >= 0)
    granger = (time[:, None, :] < time[:, :, None]) & real[:, :, None] & real[:, None, :]
    g = torch.where(granger, g, torch.zeros_like(g))
    cnt = (g > 0).to(g.dtype)
    cause_ent = entity.clamp_min(0)[:, None, :].expand_as(g)                       # [B, P(i), P(j)] → u
    b, p, _ = g.shape
    mass = g.new_zeros(b, p, n_entities).scatter_add_(2, cause_ent, g)            # [B, P, V(u)]
    count = g.new_zeros(b, p, n_entities).scatter_add_(2, cause_ent, cnt)
    groups = event_groups(entity, mask, n_entities)
    return segmented_cumsum(mass, groups), segmented_cumsum(count, groups)


def aggregate_causal_gates(
    mass: torch.Tensor, count: torch.Tensor, latest: torch.Tensor
) -> torch.Tensor:
    """ḡ_{u→v} as of one trigger: mean candidate gate from cause u to effect v over v's states ≤ τ.

    mass, count: [B, P, V] from `causal_prefix`; latest: long [B, V] (−1 none).
    Returns [B, V(u cause), V(v effect)] in [0, 1], zero on the diagonal and for unseen effects.

        ḡ_{u→v}(τ) = Σ_{i: entity(i)=v, t_i ≤ τ} Σ_{j: entity(j)=u, t_j < t_i} g_ij  /  max(1, #candidates)
    """
    m = gather_rows(mass, latest)                                                  # [B, V(v), V(u)]
    c = gather_rows(count, latest)
    seen = (latest >= 0)[:, :, None]
    gbar = torch.where(seen, m / c.clamp_min(1.0), torch.zeros_like(m)).transpose(1, 2)  # [B, u, v]
    eye = torch.eye(gbar.shape[-1], dtype=torch.bool, device=gbar.device)[None]
    return gbar.masked_fill(eye, 0.0)


def prev_positions(next_index: torch.Tensor) -> torch.Tensor:
    """Invert `PositionBatch.next_index` [B, P] into the same entity's previous position (−1 none)."""
    b, p = next_index.shape
    src = torch.arange(p, device=next_index.device)[None, :].expand(b, p)
    ok = next_index >= 0
    # Scatter i into prev[next[i]]; rows without a successor write into a dummy column P.
    tgt = torch.where(ok, next_index, torch.full_like(next_index, p))
    buf = torch.full((b, p + 1), -1, dtype=next_index.dtype, device=next_index.device)
    buf.scatter_(1, tgt, torch.where(ok, src, torch.full_like(src, -1)))
    return buf[:, :p]
