"""Window structure builder: planes, hyperedges, as-of local subgraphs, RWSE and contact matrices.

Purpose
-------
For one window of state updates, build every structure the model reads about who touched whom and
when (build-spec section 2.2), with no future leakage:

- per update: its relation (base hyperedge) id and its planes (AS-01);
- per window: the multiplex hypergraph itself with as-of bookkeeping (`build_window_hypergraph`), the
  one construction of relations and fans that every consumer shares (the local subgraphs below and the
  event-driven snapshots of `graph/builder.py`);
- per window: the contact matrices C1, C2 and Pi (first contact, first 2-hop path, first contact per
  plane) used by the TSTCT and TAAFT masks;
- per position (entity state): the local hypergraph as of the position's time, with node inputs, hop
  distances and per-plane RWSE, as an unbatched `GraphBatch`.

`collate_structures` batches several windows into the `WindowBatch` fields and one union `GraphBatch`
with flat indices.

Decisions and assumptions
-------------------------
- D-39 (heterogeneous multiplex hypergraph), D-47 (`multicast` group entities), D-49 (RWSE, hop
  distance; no index-based encodings), D-41 (unknown port or protocol stays unknown: NaN, never 0),
  D-52 (star semantics: contact means communication; fan and group co-members are not in contact).
- AS-01 planes (working option of the held D-04; rules in `graph.planes`), AS-02 hyperedge kinds.
- AS-101 (fan hyperedges: episodes, as-of `since`), AS-102 (local subgraph: radius, cap, recency,
  time-to-live), AS-103 (RWSE on the weighted clique expansion), AS-104 (relation identity, kinds,
  plane matching), AS-105 (contact semantics and the visibility order).

Objects and mathematics
-----------------------
Visibility order (AS-105). Updates are ranked by (time, index): rank(j) < rank(k) iff t_j < t_k, or
t_j = t_k and j < k. A position p sees exactly the updates with rank <= R_p, where
    R_p = rank(pos_update[p])                      if the position's own update is given,
    R_p = max{ rank(j) : t_j <= t_p }              otherwise (pure time; ties all visible).
Every derived event (a hyperedge appearing on a plane, a member joining a fan, a contact) carries the
rank of the update that created it, so "as of p" is the single integer test rank <= R_p.

Base hyperedges (AS-02, AS-104). The members of update j are its distinct non-negative entities
(initiator, responder, service). A relation is the pair (member set, kind) with
    kind = group     if any member is a `multicast` entity (D-47),
           session   else if protocol = 6 (TCP),
           exchange  otherwise (UDP, other, unknown).
Relation r is on plane p from since_p(r) = min{ rank(j) : j in r, j on plane p }.

Fan hyperedges (AS-101). Per plane p and initiator i, the updates of i on p are split into episodes at
gaps > fan_window_s. Inside an episode the fan qualifies at the first update k where
    |{ responder(j) : j in the episode, t_k - fan_window_s <= t_j <= t_k }| >= fan_min_responders.
The fan's `since` is that update (only then is a sweep evident, so earlier positions never see it: no
leakage). Member y joins at max(qualification, first touch of y in the episode); the initiator joins at
qualification. Every touch of the episode from qualification on is an activity. The time of the first
touch of the episode is kept as the sweep's start.

Local subgraph of position p (AS-102). A hyperedge is alive at p if since <= R_p and its last activity
with rank <= R_p is at most `hyperedge_ttl_s` before t_p (time-based validity, never count-based, in the
spirit of D-36). BFS from the centre over alive hyperedges up to `max_hops`, stepping only along
communication (D-52, the star rule of the contacts below), so a node's hop in the subgraph is its contact
distance and the subgraph is the neighbourhood the hop <= 2 spatial heads of TSTCT see; at each hop the
candidates are ordered by recency (the latest activity rank of an alive hyperedge joining them to the
frontier, larger first; ties by entity index) and the subgraph is capped at `max_nodes`. The subgraph
holds every alive hyperedge with >= 2 selected members (the induced sub-hypergraph). Node inputs are the
entity's latest visible update (index, role, age).

RWSE (D-49, AS-103). Per plane, the weighted clique expansion of the local subgraph,
    A_uv = sum over e containing u and v, u != v, of 1 / (|e| - 1)
(|e| counted inside the subgraph), and per node the return probabilities diag(M^k), M = D^-1 A,
k = 1 ... K, exactly as `nn.positional.random_walk_se` (isolated nodes get a self-loop). The weights make
each hyperedge spread one unit of walk mass: the hypergraph random walk of Zhou, Huang and Schoelkopf
("Learning with Hypergraphs", NeurIPS 2006) without the step that stays at the same node. RWSE keeps the
clique expansion (D-52 note): it encodes the structure a node sits in (for example being one of thirty
hosts swept together), not who communicated with whom; contact semantics govern hops, C1 and C2 only.

Contact matrices (AS-105, D-52). Contact means communication (star semantics). In a hyperedge e the hubs
are: every member for a session or exchange (one flow), the initiator for a fan, the multicast member(s)
for a group; u and v communicate through e iff u != v are members and at least one is a hub. So a fan
joins its initiator with each responder, a group joins each sender with the group entity, and two
co-members of a fan or group are not in contact (they are 2 hops apart, linked in C2 through the hub).
Over all events of the window (no time-to-live: first contact is monotone):
    C1_p[u, v] = time of the first update after which u and v communicate through a hyperedge on
                 plane p (+inf never; the later of the two members' joins),
    C1[u, v]   = min_p C1_p[u, v],     C1[u, u] = C1_p[u, u] = 0,
    C2[u, v]   = min_w max(C1[u, w], C1[w, v])        (a min-max "tropical" product).
`contact_planes[u, v, p]` = C1_p[u, v].

Invariants (tests/test_perception_graph.py)
-------------------------------------------
- No future leakage: the subgraph of position p is identical when the window is truncated to the
  updates with rank <= R_p; every node update and hyperedge `since` has rank <= R_p.
- C1, C2 and Pi equal a brute-force evaluation of their definitions.
- `collate_structures` offsets: node_update -> b*U + u, node_owner and center -> b*P + p, node and
  hyperedge indices shifted by the preceding windows' counts.

Complexity
----------
Event construction is O(U * planes + sum over fans of k^2); C2 is O(sum over w of deg(w)^2); each position
costs a constant number of vectorised NumPy calls over the frontier's incident hyperedges (CSR plus
searchsorted), i.e. O(touched entities), never a rebuild of the window graph.

Extension points
----------------
- Plane rules: `graph.planes.DECLARED_PLANE_PORTS`.
- New hyperedge kinds: extend `vocab.HYPEREDGE_KINDS` (append only) and `build_window_hypergraph`.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np
import torch

from nagahana.core.errors import InvariantViolation
from nagahana.governance.assumptions import assume
from nagahana.graph.planes import declared_update_planes
from nagahana.models.batch import GraphBatch
from nagahana.models.config.components import GraphConfig
from nagahana.models.vocab import HYPEREDGE_KIND_CODE, NODE_KIND_CODE, ROLE_CODE

INF = float("inf")
_KIND_SESSION = HYPEREDGE_KIND_CODE["session"]
_KIND_EXCHANGE = HYPEREDGE_KIND_CODE["exchange"]
_KIND_GROUP = HYPEREDGE_KIND_CODE["group"]
_KIND_FAN = HYPEREDGE_KIND_CODE["fan"]
_ROLE_NONE = ROLE_CODE["none"]
_TCP = 6


@dataclass
class WindowStructure:
    """Everything structural about one window (see the module docstring).

    update_relation: long [U] base-hyperedge (relation) id of each update, -1 if it joins < 2 entities.
    update_planes: bool [U, n_planes] planes of the update itself (AS-01). An update's planes are its own
        evidence; the planes of its relation accumulate as of each time through `contact_planes`.
    contact1, contact2: float64 [V, V]; contact_planes: float64 [V, V, n_planes] (+inf = never).
    graph: unbatched `GraphBatch`: node_update holds the window-local update index (-1 none);
        node_owner holds the window-local position index; center [P] the node index of each centre.
    node_entity: long [N] window entity index of each node (provenance, Decoder view, tests).
    hyperedge_since: plane -> float64 [E_p] time from which the local hyperedge existed on that plane.
    hyperedge_global: plane -> long [E_p] window-level hyperedge id (relation id r, or R + fan id).
    """

    update_relation: torch.Tensor
    update_planes: torch.Tensor
    contact1: torch.Tensor
    contact2: torch.Tensor
    contact_planes: torch.Tensor
    graph: GraphBatch
    node_entity: torch.Tensor = field(default_factory=lambda: torch.zeros(0, dtype=torch.long))
    hyperedge_since: dict[str, torch.Tensor] = field(default_factory=dict)
    hyperedge_global: dict[str, torch.Tensor] = field(default_factory=dict)


class CollatedStructures(NamedTuple):
    """Return value of `collate_structures` (a tuple; fields by name).

    update_relation: long [B, U] (-1 pad); update_planes: bool [B, U, n_planes] (False pad);
    contact1, contact2: float64 [B, V, V]; contact_planes: float64 [B, V, V, n_planes] (+inf pad);
    graph: union `GraphBatch` with flat indices (node_update b*U + u, node_owner b*P + p,
        center [B*P] node index or -1 for padded positions);
    node_entity: long [N] window-local entity index; node_window: long [N] window b of each node;
    hyperedge_since: plane -> float64 [E_p]; hyperedge_window: plane -> long [E_p] window b of each
        hyperedge (to gather member latents: entities in `graph.hyperedge_entities` are window-local).
    """

    update_relation: torch.Tensor
    update_planes: torch.Tensor
    contact1: torch.Tensor
    contact2: torch.Tensor
    contact_planes: torch.Tensor
    graph: GraphBatch
    node_entity: torch.Tensor
    node_window: torch.Tensor
    hyperedge_since: dict[str, torch.Tensor]
    hyperedge_window: dict[str, torch.Tensor]


def _segments(starts: np.ndarray, ends: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Flat indices of the half-open segments [starts_i, ends_i) and the segment id of each index."""
    lengths = np.maximum(ends - starts, 0)
    total = int(lengths.sum())
    if total == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    seg = np.repeat(np.arange(len(starts), dtype=np.int64), lengths)
    offsets = np.cumsum(lengths) - lengths                                  # exclusive prefix sum
    idx = np.arange(total, dtype=np.int64) - np.repeat(offsets, lengths) + np.repeat(starts, lengths)
    return idx, seg


def batched_rwse(adjacency: torch.Tensor, steps: int) -> torch.Tensor:
    """`nn.positional.random_walk_se` for a batch: [K, M, M] (weights >= 0) -> [K, M, steps] float32.

    Same semantics (diagonal ignored, isolated nodes get a self-loop, M = D^-1 A, row v = diag(M^k)_v),
    computed with batched float64 matrix products. Equality with the shared primitive is tested.
    """
    a = adjacency.to(torch.float64).clone()
    k, m, _ = a.shape
    eye = torch.eye(m, dtype=torch.bool).expand(k, m, m)
    a = a.masked_fill(eye, 0.0)
    deg = a.sum(dim=-1)                                                      # [K, M]
    a = a + torch.diag_embed((deg == 0).to(torch.float64))                   # self-loop only where isolated
    mat = a / a.sum(dim=-1, keepdim=True)
    out = torch.empty(k, m, steps, dtype=torch.float64)
    p = mat
    for s in range(steps):
        out[:, :, s] = torch.diagonal(p, dim1=-2, dim2=-1)
        p = p @ mat
    return out.to(torch.float32)


@dataclass
class _Hyperedges:
    """Window-level hyperedges over all planes (index g), in CSR layouts keyed by rank."""

    plane: np.ndarray           # [G] plane index
    kind: np.ndarray            # [G] hyperedge kind code
    since_rank: np.ndarray      # [G]
    gid: np.ndarray             # [G] relation id r, or R + fan id
    mem_key: np.ndarray         # members sorted by (g, join, entity): key g*(U+1) + join
    mem_entity: np.ndarray
    mem_join: np.ndarray
    mem_off: np.ndarray         # [G + 1]
    act_key: np.ndarray         # activity ranks sorted by (g, rank): key g*(U+1) + rank
    act_rank: np.ndarray
    inc_key: np.ndarray         # incidence sorted by (entity, join): key entity*(U+1) + join
    inc_g: np.ndarray
    inc_off: np.ndarray         # [V + 1]


@dataclass
class WindowHypergraph:
    """The window-level multiplex hypergraph with as-of bookkeeping (module docstring).

    order: long [U] rank -> update index; rank: long [U] update index -> rank; times_sorted: float64 [U]
    time of each rank; update_relation: long [U]; update_planes: bool [U, n_planes]; planes: plane names;
    hyperedges: the CSR tables of every window-level hyperedge g; hub: long [G] the fan initiator of g
    (-1 for base hyperedges); first_time: float64 [G] the sweep start of a fan (NaN for base hyperedges);
    n_relations: number of base relations; entity_kind: long [V]; stride: U + 1 (the rank key stride).
    """

    order: np.ndarray
    rank: np.ndarray
    times_sorted: np.ndarray
    update_relation: np.ndarray
    update_planes: np.ndarray
    planes: tuple[str, ...]
    hyperedges: _Hyperedges
    hub: np.ndarray
    first_time: np.ndarray
    n_relations: int
    entity_kind: np.ndarray
    stride: int

    @property
    def n_hyperedges(self) -> int:
        """G, the number of window-level hyperedges over all planes."""
        return int(self.hyperedges.plane.shape[0])

    def visible_rank(self, t: float) -> int:
        """R_t = max{ rank(j) : t_j <= t } (ties all visible), -1 when nothing is visible yet."""
        return int(np.searchsorted(self.times_sorted, t, side="right")) - 1

    def is_hub(self, entity: np.ndarray, g: np.ndarray) -> np.ndarray:
        """D-52: is `entity` a hub of hyperedge g? Fan: its initiator; group: its multicast member(s);
        session or exchange (one flow): every member. Two members communicate iff one is a hub."""
        kind = self.hyperedges.kind[g]
        is_mc = self.entity_kind == NODE_KIND_CODE["multicast"]
        return np.where(kind == _KIND_FAN, entity == self.hub[g], np.where(kind == _KIND_GROUP, is_mc[entity], True))

    def last_activity(self, g: np.ndarray, r: int) -> np.ndarray:
        """Rank of the latest activity of each hyperedge g with rank <= r (requires since(g) <= r)."""
        hx = self.hyperedges
        pos = np.searchsorted(hx.act_key, g * self.stride + r, side="right") - 1
        return hx.act_rank[pos]

    def alive(self, r: int, t: float, ttl: float) -> np.ndarray:
        """bool [G]: hyperedges that exist as of rank r and were active within `ttl` seconds before t."""
        hx = self.hyperedges
        out = np.zeros(self.n_hyperedges, dtype=bool)
        if r < 0 or self.n_hyperedges == 0:
            return out
        exists = hx.since_rank <= r
        g = np.nonzero(exists)[0]
        if g.size:
            last = self.last_activity(g, r)
            out[g] = self.times_sorted[last] >= t - ttl
        return out

    def members_as_of(self, g: int, r: int) -> np.ndarray:
        """Entities of hyperedge g that joined with rank <= r, in join order."""
        hx = self.hyperedges
        start = hx.mem_off[g]
        end = np.searchsorted(hx.mem_key, g * self.stride + r, side="right")
        return hx.mem_entity[start:end]


def _relation_kind(members: tuple[int, ...], protocol: float, entity_kind: np.ndarray) -> int:
    """AS-02 / AS-104 kind of a base hyperedge: group > session (TCP) > exchange."""
    if any(int(entity_kind[m]) == NODE_KIND_CODE["multicast"] for m in members):
        return _KIND_GROUP
    if np.isfinite(protocol) and int(protocol) == _TCP:
        return _KIND_SESSION
    return _KIND_EXCHANGE


def _detect_fans(
    order: np.ndarray, times_sorted: np.ndarray, initiator: np.ndarray, responder: np.ndarray,
    on_plane: np.ndarray, *, min_responders: int, window_s: float,
) -> list[tuple[int, list[int], list[int], int, list[int], float]]:
    """Fans of one plane (AS-101): list of (initiator, members, joins, since_rank, activity, first_time).

    Updates are visited in rank order; `order[rank]` is the update index of that rank.
    """
    per_init: dict[int, list[tuple[int, float, int]]] = {}
    for rk, j in enumerate(order.tolist()):
        i, y = int(initiator[j]), int(responder[j])
        if on_plane[j] and i >= 0 and y >= 0 and y != i:
            per_init.setdefault(i, []).append((rk, float(times_sorted[rk]), y))
    fans: list[tuple[int, list[int], list[int], int, list[int], float]] = []
    for i, touches in per_init.items():
        episode: list[tuple[int, float, int]] = []

        def close(ep: list[tuple[int, float, int]], init: int = i) -> None:
            # Sliding window over the episode; qualification at the first update with >= f responders.
            win: deque[tuple[float, int]] = deque()
            counts: dict[int, int] = {}
            first: dict[int, int] = {}
            qual = -1
            activity: list[int] = []
            for rk, t, y in ep:
                first.setdefault(y, rk)
                win.append((t, y))
                counts[y] = counts.get(y, 0) + 1
                while win and t - win[0][0] > window_s:          # keep t - t_j <= fan_window_s (closed)
                    _, y_old = win.popleft()
                    counts[y_old] -= 1
                    if counts[y_old] == 0:
                        del counts[y_old]
                if qual < 0 and len(counts) >= min_responders:
                    qual = rk
                if qual >= 0:
                    activity.append(rk)
            if qual < 0:
                return
            members = [init, *first.keys()]
            joins = [qual, *(max(qual, r) for r in first.values())]
            fans.append((init, members, joins, qual, activity, ep[0][1]))

        for touch in touches:
            if episode and touch[1] - episode[-1][1] > window_s:   # a gap > fan_window_s closes the episode
                close(episode)
                episode = []
            episode.append(touch)
        if episode:
            close(episode)
    return fans


def _check_update_arrays(t_upd: np.ndarray, ents: np.ndarray, dst_port: np.ndarray, protocol: np.ndarray,
                         ekind: np.ndarray) -> None:
    """Contract checks shared by every builder entry point."""
    u_n, v_n = t_upd.shape[0], ekind.shape[0]
    if ents.shape != (u_n, 3) or np.shape(dst_port) != (u_n,) or np.shape(protocol) != (u_n,):
        raise InvariantViolation("update arrays must be [U] / [U, 3]")
    if not np.isfinite(t_upd).all():
        raise InvariantViolation("event times must be finite")
    if ents.size and int(ents.max(initial=-1)) >= v_n:
        raise InvariantViolation("update entity index >= V")


def build_window_hypergraph(
    *,
    update_time: np.ndarray,
    update_entities: np.ndarray,
    dst_port: np.ndarray,
    protocol: np.ndarray,
    entity_kind: np.ndarray,
    cfg: GraphConfig,
) -> WindowHypergraph:
    """The window-level multiplex hypergraph: visibility ranks, planes, relations and fans (module docstring).

    update_time float64 [U] (any order); update_entities long [U, 3] (initiator, responder, service; -1 none);
    dst_port, protocol float [U] (NaN where unknown, D-41); entity_kind long [V] codes of `vocab.NODE_KINDS`.
    """
    for a in ("AS-01", "AS-02"):
        assume(a, by=__name__)
    t_upd = np.asarray(update_time, dtype=np.float64)
    ents = np.asarray(update_entities, dtype=np.int64)
    ekind = np.asarray(entity_kind, dtype=np.int64)
    _check_update_arrays(t_upd, ents, np.asarray(dst_port), np.asarray(protocol), ekind)
    u_n, v_n = t_upd.shape[0], ekind.shape[0]
    planes = tuple(cfg.planes)
    n_pl = len(planes)

    # Visibility ranks: (time, index) order (AS-105).
    order = np.lexsort((np.arange(u_n), t_upd))                              # rank -> update
    rank = np.empty(u_n, dtype=np.int64)
    rank[order] = np.arange(u_n)
    times_sorted = t_upd[order]                                              # time of each rank

    # Planes of each update (AS-01) and base relations (AS-02, AS-104), in order of first appearance.
    upl = declared_update_planes(planes, dst_port=np.asarray(dst_port, dtype=np.float64),
                                 protocol=np.asarray(protocol, dtype=np.float64), has_service=ents[:, 2] >= 0)
    proto = np.asarray(protocol, dtype=np.float64)
    rel_of: dict[tuple[int, tuple[int, ...]], int] = {}
    rel_members: list[tuple[int, ...]] = []
    rel_kind: list[int] = []
    update_relation = np.full(u_n, -1, dtype=np.int64)
    for j in order.tolist():
        members = tuple(sorted({int(x) for x in ents[j] if x >= 0}))
        if len(members) < 2:
            upl[j] = False                                                   # no relation: no plane
            continue
        kind = _relation_kind(members, float(proto[j]), ekind)
        key = (kind, members)
        if key not in rel_of:
            rel_of[key] = len(rel_members)
            rel_members.append(members)
            rel_kind.append(kind)
        update_relation[j] = rel_of[key]
    n_rel = len(rel_members)

    # Window-level hyperedges g over all planes: base relations then fans (AS-101).
    g_plane: list[int] = []
    g_kind: list[int] = []
    g_since: list[int] = []
    g_gid: list[int] = []
    g_hub: list[int] = []                          # fan initiator (the star's hub, D-52); -1 otherwise
    g_first: list[float] = []                      # sweep start of a fan; NaN otherwise
    mem_rows: list[tuple[int, int, int]] = []      # (g, entity, join rank)
    act_rows: list[tuple[int, int]] = []           # (g, activity rank)
    n_fans = 0                                     # fans over all planes (window-level ids R + n)
    for pi in range(n_pl):
        on = upl[:, pi]
        # Base hyperedges: relation r on plane pi from its first update on pi; every update is activity.
        first_rank: dict[int, int] = {}
        acts: dict[int, list[int]] = {}
        for rk, j in enumerate(order.tolist()):
            r = int(update_relation[j])
            if r >= 0 and on[j]:
                first_rank.setdefault(r, rk)
                acts.setdefault(r, []).append(rk)
        for r in sorted(first_rank):
            g = len(g_plane)
            g_plane.append(pi)
            g_kind.append(rel_kind[r])
            g_since.append(first_rank[r])
            g_gid.append(r)
            g_hub.append(-1)
            g_first.append(float("nan"))
            mem_rows.extend((g, m, first_rank[r]) for m in rel_members[r])
            act_rows.extend((g, a) for a in acts[r])
        # Fan hyperedges of this plane.
        fans = _detect_fans(order, times_sorted, ents[:, 0], ents[:, 1], on,
                            min_responders=cfg.fan_min_responders, window_s=cfg.fan_window_s)
        for init_f, members_f, joins_f, since_f, activity_f, first_t in fans:
            g = len(g_plane)
            g_plane.append(pi)
            g_kind.append(_KIND_FAN)
            g_since.append(since_f)
            g_gid.append(n_rel + n_fans)                     # unique window-level id per (plane, fan)
            g_hub.append(init_f)
            g_first.append(first_t)
            n_fans += 1
            mem_rows.extend((g, m, jn) for m, jn in zip(members_f, joins_f, strict=True))
            act_rows.extend((g, a) for a in activity_f)
    hx = _index_hyperedges(np.array(g_plane, dtype=np.int64), np.array(g_kind, dtype=np.int64),
                           np.array(g_since, dtype=np.int64), np.array(g_gid, dtype=np.int64),
                           mem_rows, act_rows, u_n, v_n)
    return WindowHypergraph(order=order, rank=rank, times_sorted=times_sorted, update_relation=update_relation,
                            update_planes=upl, planes=planes, hyperedges=hx, hub=np.array(g_hub, dtype=np.int64),
                            first_time=np.array(g_first, dtype=np.float64), n_relations=n_rel, entity_kind=ekind,
                            stride=u_n + 1)


def build_window_structure(
    *,
    update_time: np.ndarray,
    update_entities: np.ndarray,
    dst_port: np.ndarray,
    protocol: np.ndarray,
    entity_kind: np.ndarray,
    pos_entity: np.ndarray,
    pos_time: np.ndarray,
    cfg: GraphConfig,
    max_members: int = 8,
    pos_update: np.ndarray | None = None,
) -> WindowStructure:
    """Build the structure of one window. See the module docstring for every definition.

    Parameters
    ----------
    update_time: float64 [U] seconds relative to the window origin (any order).
    update_entities: long [U, 3] (initiator, responder, service), -1 = none.
    dst_port, protocol: float [U], NaN where unknown (D-41).
    entity_kind: long [V] codes of `vocab.NODE_KINDS`.
    pos_entity: long [P]; pos_time: float64 [P] (each position's event time).
    cfg: `GraphConfig` (planes, max_nodes, max_hops, rwse_steps, fan rules, hyperedge_ttl_s).
    max_members: width of `hyperedge_entities` (members beyond it are truncated there only).
    pos_update: long [P] optional, each position's own update; when given, visibility is the strict
        (time, index) order and pos_time must equal that update's time (AS-105).
    """
    t_upd = np.asarray(update_time, dtype=np.float64)
    ents = np.asarray(update_entities, dtype=np.int64)
    ekind = np.asarray(entity_kind, dtype=np.int64)
    p_ent = np.asarray(pos_entity, dtype=np.int64)
    p_time = np.asarray(pos_time, dtype=np.float64)
    u_n, v_n, p_n = t_upd.shape[0], ekind.shape[0], p_ent.shape[0]
    # Contract checks of the position arrays (the update arrays are checked by the hypergraph builder).
    _check_update_arrays(t_upd, ents, np.asarray(dst_port), np.asarray(protocol), ekind)
    if p_time.shape != (p_n,):
        raise InvariantViolation("pos_entity and pos_time must both be [P]")
    if not np.isfinite(p_time).all():
        raise InvariantViolation("event times must be finite")
    if p_n and (int(p_ent.min()) < 0 or int(p_ent.max()) >= v_n):
        raise InvariantViolation("position entity outside [0, V)")

    wh = build_window_hypergraph(update_time=t_upd, update_entities=ents, dst_port=dst_port, protocol=protocol,
                                 entity_kind=ekind, cfg=cfg)
    order, rank, times_sorted = wh.order, wh.rank, wh.times_sorted
    hx = wh.hyperedges
    planes, n_pl, n_g = wh.planes, len(wh.planes), wh.n_hyperedges
    upl, update_relation = wh.update_planes, wh.update_relation

    # Visibility rank of each position (AS-105).
    if pos_update is not None:
        p_upd = np.asarray(pos_update, dtype=np.int64)
        if p_upd.shape != (p_n,) or (p_n and (p_upd.min() < 0 or p_upd.max() >= u_n)):
            raise InvariantViolation("pos_update must be [P] update indices")
        if not np.array_equal(t_upd[p_upd], p_time):
            raise InvariantViolation("pos_time must equal the time of pos_update")
        r_pos = rank[p_upd]
    else:
        r_pos = np.searchsorted(times_sorted, p_time, side="right") - 1      # -1: nothing visible yet

    # Contact matrices (AS-105, D-52): the first time two entities communicated through a hyperedge, per
    # plane. Each ordered member pair (a, b) of g with a or b a hub of g is one event at the later of the
    # two joins: all pairs of a session or exchange, initiator-responder pairs of a fan, member-group pairs
    # of a group. Co-members of a fan or group are not in contact (they meet in C2 through the hub).
    cpl = np.full((v_n, v_n, n_pl), INF)
    if hx.mem_entity.size:
        g_row = np.repeat(np.arange(n_g), np.diff(hx.mem_off))                 # hyperedge of each member row
        idx_b, row_a = _segments(hx.mem_off[g_row], hx.mem_off[g_row + 1])     # all member pairs, per hyperedge
        g_pair = g_row[row_a]
        star = wh.is_hub(hx.mem_entity[row_a], g_pair) | wh.is_hub(hx.mem_entity[idx_b], g_pair)
        row_a, idx_b, g_pair = row_a[star], idx_b[star], g_pair[star]
        tt = times_sorted[np.maximum(hx.mem_join[row_a], hx.mem_join[idx_b])]
        np.minimum.at(cpl, (hx.mem_entity[row_a], hx.mem_entity[idx_b], hx.plane[g_pair]), tt)
    idx_v = np.arange(v_n)
    cpl[idx_v, idx_v, :] = 0.0
    c1 = cpl.min(axis=2) if n_pl else np.full((v_n, v_n), INF)
    c1[idx_v, idx_v] = 0.0
    c2 = c1.copy()
    for w in range(v_n):                                                     # min-max product, sparse in w
        nb = np.nonzero(np.isfinite(c1[w]) & (idx_v != w))[0]
        if nb.size:
            cw = c1[w, nb]
            c2[np.ix_(nb, nb)] = np.minimum(c2[np.ix_(nb, nb)], np.maximum.outer(cw, cw))

    # Per-entity update index (for node inputs): sorted by (entity, rank).
    ue_rows = [(int(ents[j, role]), int(rank[j]), j, role) for j in range(u_n) for role in range(3) if ents[j, role] >= 0]
    ue = np.array(ue_rows, dtype=np.int64).reshape(-1, 4)
    if ue.size:
        # One row per (entity, update): keep the first role if an entity fills two roles of one update.
        ue = ue[np.lexsort((ue[:, 3], ue[:, 1], ue[:, 0]))]
        keep = np.ones(len(ue), dtype=bool)
        keep[1:] = (ue[1:, 0] != ue[:-1, 0]) | (ue[1:, 1] != ue[:-1, 1])
        ue = ue[keep]
    ue_key = ue[:, 0] * (u_n + 1) + ue[:, 1] if ue.size else np.zeros(0, dtype=np.int64)

    # Local subgraphs, one per position.
    stride = wh.stride
    node_update: list[np.ndarray] = []
    node_role: list[np.ndarray] = []
    node_kind: list[np.ndarray] = []
    node_age: list[np.ndarray] = []
    node_hop: list[np.ndarray] = []
    node_owner: list[np.ndarray] = []
    node_entity: list[np.ndarray] = []
    center = np.full(p_n, -1, dtype=np.int64)
    # Local hyperedges over all positions (index h): window hyperedge g and owning position p; and the
    # incidence pairs (union node, h). Per-plane ids are assigned after the loop.
    loc_g: list[np.ndarray] = []
    loc_p: list[np.ndarray] = []
    pair_node: list[np.ndarray] = []
    pair_h: list[np.ndarray] = []
    n_local = 0
    onehot_plane = np.eye(n_pl)[hx.plane] if n_g else np.zeros((0, n_pl))      # [G, n_pl]
    m_cap = cfg.max_nodes
    adj_all = np.zeros((p_n, n_pl, m_cap, m_cap), dtype=np.float64)
    n_nodes = 0

    def alive_incident(xs: np.ndarray, r: int, t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(entity position in xs, g, last activity rank) for alive hyperedges incident to xs as of r."""
        starts = hx.inc_off[xs]
        ends = np.searchsorted(hx.inc_key, xs * stride + r, side="right")
        idx, seg = _segments(starts, ends)
        g = hx.inc_g[idx]
        if g.size == 0:
            return seg, g, g
        last = wh.last_activity(g, r)                                        # since <= r: an activity <= r exists
        alive = times_sorted[last] >= t - cfg.hyperedge_ttl_s
        return seg[alive], g[alive], last[alive]

    for p in range(p_n):
        r, t, e0 = int(r_pos[p]), float(p_time[p]), int(p_ent[p])
        sel = [e0]
        hops = [0]
        chosen = {e0}
        frontier = np.array([e0], dtype=np.int64)
        # BFS over alive hyperedges, capped by recency (AS-102).
        for h in range(1, cfg.max_hops + 1):
            if len(sel) >= m_cap or frontier.size == 0 or r < 0:
                break
            fseg, ga, last = alive_incident(frontier, r, t)
            if ga.size == 0:
                break
            ms, me = hx.mem_off[ga], np.searchsorted(hx.mem_key, ga * stride + r, side="right")
            idx, seg = _segments(ms, me)
            cand, rec = hx.mem_entity[idx], last[seg]
            # D-52: step only along communication (star): a hub reaches every member, a non-hub only the hubs.
            g_seg = ga[seg]
            star = wh.is_hub(frontier[fseg[seg]], g_seg) | wh.is_hub(cand, g_seg)
            cand, rec = cand[star], rec[star]
            fresh = ~np.isin(cand, list(chosen))
            cand, rec = cand[fresh], rec[fresh]
            if cand.size == 0:
                break
            # Recency per candidate = max over joining hyperedges; order: recency descending, entity ascending.
            o = np.lexsort((-rec, cand))
            cand, rec = cand[o], rec[o]
            first = np.ones(cand.size, dtype=bool)
            first[1:] = cand[1:] != cand[:-1]
            cand, rec = cand[first], rec[first]
            o = np.lexsort((cand, -rec))
            take = cand[o][: m_cap - len(sel)]
            sel.extend(int(x) for x in take)
            hops.extend([h] * take.size)
            chosen.update(int(x) for x in take)
            frontier = take
        sel_arr = np.array(sel, dtype=np.int64)
        n = sel_arr.size
        # Node inputs: latest visible update of each node (index, role, age).
        if ue.size and r >= 0:
            pos = np.searchsorted(ue_key, sel_arr * stride + r, side="right") - 1
            ok = (pos >= 0) & (ue[np.maximum(pos, 0), 0] == sel_arr)
            row = ue[np.maximum(pos, 0)]
            nu = np.where(ok, row[:, 2], -1)
            nr = np.where(ok, row[:, 3], _ROLE_NONE)
            na = np.where(ok, t - t_upd[np.maximum(nu, 0)], 0.0)
        else:
            nu = np.full(n, -1, dtype=np.int64)
            nr = np.full(n, _ROLE_NONE, dtype=np.int64)
            na = np.zeros(n)
        center[p] = n_nodes
        node_update.append(nu)
        node_role.append(nr)
        node_kind.append(ekind[sel_arr])
        node_age.append(na.astype(np.float32))
        node_hop.append(np.array(hops, dtype=np.int64))
        node_owner.append(np.full(n, p, dtype=np.int64))
        node_entity.append(sel_arr)
        # Induced alive hyperedges with >= 2 selected members (vectorised over hyperedges).
        if r >= 0:
            loc, gi, _ = alive_incident(sel_arr, r, t)
            if gi.size:
                gs, inv, cnt = np.unique(gi, return_inverse=True, return_counts=True)
                kept = np.nonzero(cnt >= 2)[0]                               # [K] kept among the unique g
                if kept.size:
                    remap = np.full(gs.size, -1, dtype=np.int64)
                    remap[kept] = np.arange(kept.size)
                    in_kept = remap[inv] >= 0
                    k_of_pair = remap[inv[in_kept]]                          # local hyperedge of each pair
                    l_of_pair = loc[in_kept]                                 # node position in sel_arr
                    loc_g.append(gs[kept])
                    loc_p.append(np.full(kept.size, p, dtype=np.int64))
                    pair_node.append(n_nodes + l_of_pair)
                    pair_h.append(n_local + k_of_pair)
                    n_local += kept.size
                    # Weighted clique expansion for RWSE (AS-103): A_p = H diag(w * [plane = p]) H^T.
                    hm = np.zeros((n, kept.size))
                    hm[l_of_pair, k_of_pair] = 1.0
                    w_k = 1.0 / (cnt[kept] - 1.0)
                    hw = hm[None, :, :] * (onehot_plane[gs[kept]] * w_k[:, None]).T[:, None, :]   # [n_pl, n, K]
                    adj_all[p, :, :n, :n] = hw @ hm.T                                              # [n_pl, n, n]
        n_nodes += n

    # Per-plane local hyperedges: ids in order of appearance; members as of the owning position.
    h_g = np.concatenate(loc_g) if loc_g else np.zeros(0, dtype=np.int64)
    h_p = np.concatenate(loc_p) if loc_p else np.zeros(0, dtype=np.int64)
    pr_node = np.concatenate(pair_node) if pair_node else np.zeros(0, dtype=np.int64)
    pr_h = np.concatenate(pair_h) if pair_h else np.zeros(0, dtype=np.int64)
    h_plane = hx.plane[h_g] if h_g.size else np.zeros(0, dtype=np.int64)
    h_local = np.zeros(h_g.size, dtype=np.int64)
    for pi in range(n_pl):
        on_pl = h_plane == pi
        h_local[on_pl] = np.arange(int(on_pl.sum()))
    m_start = hx.mem_off[h_g]
    m_end = np.minimum(np.searchsorted(hx.mem_key, h_g * stride + r_pos[h_p], side="right"), m_start + max_members)
    m_idx, m_seg = _segments(m_start, m_end)
    mem_tab = np.full((h_g.size, max_members), -1, dtype=np.int64)
    mem_tab[m_seg, m_idx - m_start[m_seg]] = hx.mem_entity[m_idx]          # join order, truncated
    pair_plane = h_plane[pr_h] if pr_h.size else np.zeros(0, dtype=np.int64)

    # RWSE per position and plane on the local clique expansions (D-49, AS-103).
    rwse_parts: list[torch.Tensor] = []
    chunk = 256                                                              # bounds the float64 working set
    for p0 in range(0, p_n if n_pl else 0, chunk):
        blk = adj_all[p0: p0 + chunk]
        rw = batched_rwse(torch.from_numpy(blk.reshape(-1, m_cap, m_cap)), cfg.rwse_steps)
        rw = rw.view(blk.shape[0], n_pl, m_cap, cfg.rwse_steps).permute(0, 2, 1, 3)   # [chunk, M, n_pl, K]
        for p in range(p0, p0 + blk.shape[0]):
            rwse_parts.append(rw[p - p0, : node_entity[p].size])
    rwse = (torch.cat(rwse_parts) if rwse_parts else torch.zeros(0, n_pl, cfg.rwse_steps))

    def cat_long(parts: list[np.ndarray]) -> torch.Tensor:
        return torch.from_numpy(np.concatenate(parts).astype(np.int64)) if parts else torch.zeros(0, dtype=torch.long)

    graph = GraphBatch(
        node_update=cat_long(node_update),
        node_role=cat_long(node_role),
        node_kind=cat_long(node_kind),
        node_age=torch.from_numpy(np.concatenate(node_age).astype(np.float32)) if node_age else torch.zeros(0),
        node_hop=cat_long(node_hop),
        node_owner=cat_long(node_owner),
        center=torch.from_numpy(center),
        incidence={planes[pi]: torch.from_numpy(np.stack([pr_node[pair_plane == pi], h_local[pr_h[pair_plane == pi]]]).astype(np.int64))
                   for pi in range(n_pl)},
        hyperedge_kind={planes[pi]: torch.from_numpy(hx.kind[h_g[h_plane == pi]].astype(np.int64)) for pi in range(n_pl)},
        hyperedge_entities={planes[pi]: torch.from_numpy(mem_tab[h_plane == pi]) for pi in range(n_pl)},
        rwse=rwse,
    )
    return WindowStructure(
        update_relation=torch.from_numpy(update_relation),
        update_planes=torch.from_numpy(upl),
        contact1=torch.from_numpy(c1),
        contact2=torch.from_numpy(c2),
        contact_planes=torch.from_numpy(cpl),
        graph=graph,
        node_entity=cat_long(node_entity),
        hyperedge_since={planes[pi]: torch.from_numpy(times_sorted[hx.since_rank[h_g[h_plane == pi]]].astype(np.float64))
                         for pi in range(n_pl)},
        hyperedge_global={planes[pi]: torch.from_numpy(hx.gid[h_g[h_plane == pi]].astype(np.int64)) for pi in range(n_pl)},
    )


def _index_hyperedges(
    plane: np.ndarray, kind: np.ndarray, since: np.ndarray, gid: np.ndarray,
    mem_rows: list[tuple[int, int, int]], act_rows: list[tuple[int, int]], u_n: int, v_n: int,
) -> _Hyperedges:
    """Sort membership, activity and incidence rows into rank-keyed CSR layouts."""
    stride = u_n + 1
    n_g = plane.shape[0]
    mem = np.array(mem_rows, dtype=np.int64).reshape(-1, 3)
    mem = mem[np.lexsort((mem[:, 1], mem[:, 2], mem[:, 0]))] if mem.size else mem     # (g, join, entity)
    act = np.array(act_rows, dtype=np.int64).reshape(-1, 2)
    act = act[np.lexsort((act[:, 1], act[:, 0]))] if act.size else act                # (g, rank)
    inc = mem[np.lexsort((mem[:, 0], mem[:, 2], mem[:, 1]))] if mem.size else mem     # (entity, join, g)
    mem_off = np.searchsorted(mem[:, 0], np.arange(n_g + 1)) if mem.size else np.zeros(n_g + 1, dtype=np.int64)
    inc_off = np.searchsorted(inc[:, 1], np.arange(v_n + 1)) if inc.size else np.zeros(v_n + 1, dtype=np.int64)
    return _Hyperedges(
        plane=plane, kind=kind, since_rank=since, gid=gid,
        mem_key=mem[:, 0] * stride + mem[:, 2], mem_entity=mem[:, 1], mem_join=mem[:, 2], mem_off=mem_off,
        act_key=act[:, 0] * stride + act[:, 1], act_rank=act[:, 1],
        inc_key=inc[:, 1] * stride + inc[:, 2], inc_g=inc[:, 0], inc_off=inc_off,
    )


def collate_structures(
    items: list[WindowStructure],
    *,
    updates_per_window: int,
    positions_per_window: int,
    entities_per_window: int | None = None,
) -> CollatedStructures:
    """Batch window structures into the `WindowBatch` fields and one union `GraphBatch`.

    Padding: update_relation -1, update_planes False, contacts +inf (padded entities never touch
    anything). `entities_per_window` defaults to the largest V among the items.
    Flat indices: node_update u -> b*U + u (-1 stays -1); node_owner p -> b*P + p; center[b*P + p] =
    node offset of window b + local centre index (-1 for padded positions); incidence node and
    hyperedge indices are shifted by the node and per-plane hyperedge counts of preceding windows.
    """
    if not items:
        raise InvariantViolation("collate_structures needs at least one window")
    u_w, p_w = updates_per_window, positions_per_window
    v_w = entities_per_window if entities_per_window is not None else max(int(it.contact1.shape[0]) for it in items)
    planes = tuple(items[0].graph.incidence)
    n_pl = len(planes)
    b_n = len(items)
    rel = torch.full((b_n, u_w), -1, dtype=torch.long)
    upl = torch.zeros(b_n, u_w, n_pl, dtype=torch.bool)
    c1 = torch.full((b_n, v_w, v_w), INF, dtype=torch.float64)
    c2 = torch.full((b_n, v_w, v_w), INF, dtype=torch.float64)
    cpl = torch.full((b_n, v_w, v_w, n_pl), INF, dtype=torch.float64)
    center = torch.full((b_n * p_w,), -1, dtype=torch.long)
    parts: dict[str, list[torch.Tensor]] = {k: [] for k in ("upd", "role", "kind", "age", "hop", "owner", "ent", "win", "rwse")}
    inc: dict[str, list[torch.Tensor]] = {pl: [] for pl in planes}
    hk: dict[str, list[torch.Tensor]] = {pl: [] for pl in planes}
    he: dict[str, list[torch.Tensor]] = {pl: [] for pl in planes}
    hs: dict[str, list[torch.Tensor]] = {pl: [] for pl in planes}
    hw: dict[str, list[torch.Tensor]] = {pl: [] for pl in planes}
    node_off = 0
    edge_off = dict.fromkeys(planes, 0)
    for b, it in enumerate(items):
        g = it.graph
        u_n, p_n, v_n = int(it.update_relation.shape[0]), int(g.center.shape[0]), int(it.contact1.shape[0])
        if u_n > u_w or p_n > p_w or v_n > v_w:
            raise InvariantViolation(f"window {b}: U={u_n}, P={p_n}, V={v_n} exceed the batch shape {u_w, p_w, v_w}")
        if tuple(g.incidence) != planes:
            raise InvariantViolation("all windows must have the same planes in the same order")
        # Window-level fields, padded.
        rel[b, :u_n] = it.update_relation
        upl[b, :u_n] = it.update_planes
        c1[b, :v_n, :v_n] = it.contact1
        c2[b, :v_n, :v_n] = it.contact2
        cpl[b, :v_n, :v_n] = it.contact_planes
        # Graph: flat indices.
        n = g.num_nodes
        parts["upd"].append(torch.where(g.node_update >= 0, g.node_update + b * u_w, g.node_update))
        parts["role"].append(g.node_role)
        parts["kind"].append(g.node_kind)
        parts["age"].append(g.node_age)
        parts["hop"].append(g.node_hop)
        parts["owner"].append(g.node_owner + b * p_w)
        parts["ent"].append(it.node_entity)
        parts["win"].append(torch.full((n,), b, dtype=torch.long))
        parts["rwse"].append(g.rwse)
        center[b * p_w: b * p_w + p_n] = torch.where(g.center >= 0, g.center + node_off, g.center)
        for pl in planes:
            i_pl = g.incidence[pl]
            inc[pl].append(i_pl + torch.tensor([[node_off], [edge_off[pl]]], dtype=torch.long))
            hk[pl].append(g.hyperedge_kind[pl])
            he[pl].append(g.hyperedge_entities[pl])
            e_n = int(g.hyperedge_kind[pl].shape[0])
            hs[pl].append(it.hyperedge_since.get(pl, torch.full((e_n,), float("nan"), dtype=torch.float64)))
            hw[pl].append(torch.full((e_n,), b, dtype=torch.long))
            edge_off[pl] += e_n
        node_off += n
    graph = GraphBatch(
        node_update=torch.cat(parts["upd"]), node_role=torch.cat(parts["role"]), node_kind=torch.cat(parts["kind"]),
        node_age=torch.cat(parts["age"]), node_hop=torch.cat(parts["hop"]), node_owner=torch.cat(parts["owner"]),
        center=center,
        incidence={pl: torch.cat(inc[pl], dim=1) for pl in planes},
        hyperedge_kind={pl: torch.cat(hk[pl]) for pl in planes},
        hyperedge_entities={pl: torch.cat(he[pl]) for pl in planes},
        rwse=torch.cat(parts["rwse"]),
    )
    return CollatedStructures(
        update_relation=rel, update_planes=upl, contact1=c1, contact2=c2, contact_planes=cpl, graph=graph,
        node_entity=torch.cat(parts["ent"]), node_window=torch.cat(parts["win"]),
        hyperedge_since={pl: torch.cat(hs[pl]) for pl in planes},
        hyperedge_window={pl: torch.cat(hw[pl]) for pl in planes},
    )
