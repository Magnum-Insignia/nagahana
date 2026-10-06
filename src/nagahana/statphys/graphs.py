"""Multiplex graph states of the network at a trigger, from the window contract or from the stream (D-56).

Purpose
-------
`spectral.py` reads entropies of graphs; this module decides which graph is "the network state at
tau". The state is a multiplex: one layer per relation plane of CVG-AE (AS-01: connectivity,
services, identity, remote_admin, name_resolution, ot_control), all layers on one node set.

Activity state (default, AS-767): the state updates of the state window (tau - Delta, tau] (AS-764)
- nodes: every entity that is a member (initiator, responder or service) of one of these updates;
- layer p: an edge between two members of an update that lies on plane p (`update_planes`, AS-01)
  and that communicate under the star semantics of D-52 (graph/window.py, AS-105): one update is one
  flow; the hubs of an update are all its members, unless a member is a multicast group entity
  (D-47), in which case the group members are the hubs; two members communicate iff at least one of
  them is a hub. So a sweep (a fan of flows from one scanner) joins the scanner to every target and
  never the targets to each other, and a sender joins the group entity, not the other receivers;
- weights (`SpectralConfig.weight`): binary (the pair communicated in the window), count (number of
  updates) or log1p_count. Binary is the default: the structure then does not move with traffic
  volume, which an attacker controls, while the traffic entropies read the volume side.
Contact state: the first-contact matrices of the window as of tau (models/batch.py `contact_planes`):
- nodes: the entities seen by tau (`TriggerBatch.entity_latest >= 0`);
- layer p: an edge (u, v) iff the first contact of u and v on plane p happened at or before tau.
It is cumulative within a window (first contact never expires), so it is offered as a second view.

Alignment: two states over different node sets are compared on the union of their nodes, an entity
absent from one state being an isolated node of it (AS-770). Isolated nodes do not change a von
Neumann entropy; they are part of the diffusion ensemble (one zero mode each).

Streaming: `SlidingMultiplex` keeps the activity state of the sliding window with O(1) work per
state update (each update adds and later removes its pair counts once) and equals `activity_state`
on the same updates (tested).
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
import torch

from nagahana.models.batch import WindowBatch
from nagahana.models.vocab import NODE_KIND_CODE
from nagahana.statphys.config import EDGE_WEIGHTS, SpectralConfig
from nagahana.statphys.spectral import (
    Estimate,
    Graph,
    SpectralThermo,
    combine,
    graph_entropies,
    laplacian_jsd,
    multiplex_entropy,
    quantum_jsd,
    von_neumann_entropy,
)

#: Code of the multicast (group) entity kind (D-47).
MULTICAST_CODE: int = NODE_KIND_CODE["multicast"]
_PAIRS: tuple[tuple[int, int], ...] = ((0, 1), (0, 2), (1, 2))


@dataclass(frozen=True)
class MultiplexState:
    """A multiplex graph state: node identities [n] (sorted) and one layer per plane over 0 ... n-1."""

    planes: tuple[str, ...]
    nodes: np.ndarray
    layers: tuple[Graph, ...]

    def __post_init__(self) -> None:
        if len(self.layers) != len(self.planes):
            raise ValueError("one layer per plane")
        if any(g.n != self.nodes.shape[0] for g in self.layers):
            raise ValueError("every layer must span the state's node set")
        if self.nodes.shape[0] > 1 and not bool(np.all(np.diff(self.nodes) > 0)):
            raise ValueError("node identities must be sorted and unique")

    @property
    def n(self) -> int:
        return int(self.nodes.shape[0])

    def aggregate(self) -> Graph:
        """The aggregate graph: the sum of the layers (De Domenico et al. 2015)."""
        if not any(g.num_edges for g in self.layers):
            return Graph.from_edges(self.n, np.zeros(0, np.int64), np.zeros(0, np.int64))
        return combine(list(self.layers))


def _weights(counts: np.ndarray, weight: str) -> np.ndarray:
    if weight not in EDGE_WEIGHTS:
        raise ValueError(f"weight must be one of {EDGE_WEIGHTS}")
    c = counts.astype(np.float64)
    if weight == "binary":
        return np.ones_like(c)
    if weight == "count":
        return c
    return np.log1p(c)


def pair_planes(members: np.ndarray, multicast: np.ndarray, planes: np.ndarray
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Communicating member pairs of state updates on their planes (module docstring, star semantics).

    members long [n, 3] (-1 none), multicast bool [n, 3] (the member is a group entity), planes bool
    [n, P]. Returns (update row, plane, u, v) arrays with u < v, one entry per (update, pair, plane).
    """
    m = np.asarray(members, dtype=np.int64)
    mc = np.asarray(multicast, dtype=bool) & (m >= 0)
    pl = np.asarray(planes, dtype=bool)
    any_group = np.asarray(mc.any(axis=1), dtype=bool)                        # [n]
    hub = np.where(any_group[:, None], mc, m >= 0)                            # [n, 3]
    rows, pls, us, vs = [], [], [], []
    for a, b in _PAIRS:
        ok = (m[:, a] >= 0) & (m[:, b] >= 0) & (m[:, a] != m[:, b]) & (hub[:, a] | hub[:, b])
        r_idx, p_idx = np.nonzero(ok[:, None] & pl)
        rows.append(r_idx)
        pls.append(p_idx)
        us.append(np.minimum(m[r_idx, a], m[r_idx, b]))
        vs.append(np.maximum(m[r_idx, a], m[r_idx, b]))
    return (np.concatenate(rows), np.concatenate(pls), np.concatenate(us), np.concatenate(vs))


def _state_from_pairs(planes: tuple[str, ...], node_ids: np.ndarray, pl: np.ndarray, u: np.ndarray, v: np.ndarray,
                      count: np.ndarray, weight: str) -> MultiplexState:
    # Layers over the local index of the sorted node identities.
    nodes = np.unique(node_ids.astype(np.int64))
    lu, lv = np.searchsorted(nodes, u), np.searchsorted(nodes, v)
    w = _weights(count, weight)
    layers = tuple(Graph.from_edges(nodes.shape[0], lu[pl == p], lv[pl == p], w[pl == p]) for p in range(len(planes)))
    return MultiplexState(planes=planes, nodes=nodes, layers=layers)


def activity_state(window: WindowBatch, b: int, tau: float, *, window_seconds: float, weight: str,
                   planes: tuple[str, ...]) -> MultiplexState:
    """Activity multiplex of window b at relative time tau over the state updates of (tau - window_seconds, tau]."""
    if not (math.isfinite(window_seconds) and window_seconds > 0):
        raise ValueError("window_seconds must be finite and > 0")
    t = window.update_time[b].to(torch.float64).cpu().numpy()
    sel = window.update_mask[b].cpu().numpy().astype(bool) & (t > tau - window_seconds) & (t <= tau)
    members = window.update_entities[b].cpu().numpy().astype(np.int64)[sel]
    kind = window.entity_kind[b].cpu().numpy().astype(np.int64)
    multicast = np.where(members >= 0, kind[np.maximum(members, 0)] == MULTICAST_CODE, False)
    upl = window.update_planes[b].cpu().numpy().astype(bool)[sel]
    if upl.shape[1] != len(planes):
        raise ValueError("planes must name every column of update_planes")
    row, pl, u, v = pair_planes(members, multicast, upl)
    key = (pl * window.entity_mask.shape[1] + u) * window.entity_mask.shape[1] + v
    uniq, counts = np.unique(key, return_counts=True)
    vn = window.entity_mask.shape[1]
    p_u, rem = uniq // (vn * vn), uniq % (vn * vn)
    node_ids = members[members >= 0]
    return _state_from_pairs(planes, node_ids, p_u, rem // vn, rem % vn, counts, weight)


def contact_state(window: WindowBatch, b: int, m: int, *, planes: tuple[str, ...]) -> MultiplexState:
    """Contact multiplex of window b at trigger m: first contacts per plane as of the trigger time."""
    tau = float(window.triggers.time[b, m])
    seen = (window.triggers.entity_latest[b, m] >= 0) & window.entity_mask[b]
    nodes = torch.nonzero(seen).flatten().cpu().numpy().astype(np.int64)
    cp = window.contact_planes[b].to(torch.float64).cpu().numpy()                 # [V, V, P]
    if cp.shape[-1] != len(planes):
        raise ValueError("planes must name every plane of contact_planes")
    layers = []
    sub = cp[np.ix_(nodes, nodes)] if nodes.size else np.zeros((0, 0, len(planes)))
    iu, iv = np.triu_indices(nodes.shape[0], k=1)
    for p in range(len(planes)):
        on = sub[iu, iv, p] <= tau if iu.size else np.zeros(0, dtype=bool)
        layers.append(Graph.from_edges(nodes.shape[0], iu[on], iv[on]))
    return MultiplexState(planes=planes, nodes=nodes, layers=tuple(layers))


def align_states(a: MultiplexState, b: MultiplexState) -> tuple[MultiplexState, MultiplexState]:
    """Both states on the union of their node identities (absent nodes become isolated)."""
    if a.planes != b.planes:
        raise ValueError("states must have the same planes")
    nodes = np.union1d(a.nodes, b.nodes).astype(np.int64)

    def lift(s: MultiplexState) -> MultiplexState:
        idx = np.searchsorted(nodes, s.nodes)
        layers = tuple(Graph.from_edges(nodes.shape[0], idx[g.row], idx[g.col], g.weight) for g in s.layers)
        return MultiplexState(planes=s.planes, nodes=nodes, layers=layers)

    return lift(a), lift(b)


class SlidingMultiplex:
    """Activity multiplex of the sliding window (now - window_seconds, now], streamed (module docstring)."""

    def __init__(self, planes: tuple[str, ...], *, window_seconds: float, weight: str) -> None:
        if not (math.isfinite(window_seconds) and window_seconds > 0):
            raise ValueError("window_seconds must be finite and > 0")
        if weight not in EDGE_WEIGHTS:
            raise ValueError(f"weight must be one of {EDGE_WEIGHTS}")
        self.planes = planes
        self.window = float(window_seconds)
        self.weight = weight
        self._pending: deque[tuple[float, tuple[tuple[int, int, int], ...], tuple[int, ...]]] = deque()
        self._live: deque[tuple[float, tuple[tuple[int, int, int], ...], tuple[int, ...]]] = deque()
        self._pairs: dict[tuple[int, int, int], int] = {}
        self._nodes: dict[int, int] = {}
        self._last = -math.inf

    def observe(self, times: np.ndarray, members: np.ndarray, multicast: np.ndarray, planes: np.ndarray) -> None:
        """Queue state updates (time-sorted, after every update observed before): members [n, 3] long ids,
        multicast [n, 3] bool, planes [n, P] bool."""
        t = np.asarray(times, dtype=np.float64)
        if t.shape[0] and (float(t[0]) < self._last or not bool(np.all(np.diff(t) >= 0))):
            raise ValueError("observe needs time-sorted state updates, after the ones observed before")
        m = np.asarray(members, dtype=np.int64)
        if np.asarray(planes).shape[1] != len(self.planes):
            raise ValueError("planes must have one column per plane")
        row, pl, u, v = pair_planes(m, multicast, planes)
        per_row: dict[int, list[tuple[int, int, int]]] = {}
        for r, p, a, b in zip(row.tolist(), pl.tolist(), u.tolist(), v.tolist(), strict=True):
            per_row.setdefault(r, []).append((p, a, b))
        for r in range(t.shape[0]):
            ents = tuple(sorted({int(x) for x in m[r] if x >= 0}))
            self._pending.append((float(t[r]), tuple(per_row.get(r, [])), ents))
        if t.shape[0]:
            self._last = float(t[-1])

    def _apply(self, entry: tuple[float, tuple[tuple[int, int, int], ...], tuple[int, ...]], sign: int) -> None:
        for key in entry[1]:
            c = self._pairs.get(key, 0) + sign
            if c > 0:
                self._pairs[key] = c
            else:
                self._pairs.pop(key, None)
        for e in entry[2]:
            c = self._nodes.get(e, 0) + sign
            if c > 0:
                self._nodes[e] = c
            else:
                self._nodes.pop(e, None)

    def state(self, now: float) -> MultiplexState:
        """The activity state at `now` (updates after `now` stay queued)."""
        while self._pending and self._pending[0][0] <= now:
            entry = self._pending.popleft()
            self._live.append(entry)
            self._apply(entry, +1)
        cut = now - self.window
        while self._live and self._live[0][0] <= cut:
            self._apply(self._live.popleft(), -1)
        keys = sorted(self._pairs)
        pl = np.array([k[0] for k in keys], dtype=np.int64)
        u = np.array([k[1] for k in keys], dtype=np.int64)
        v = np.array([k[2] for k in keys], dtype=np.int64)
        cnt = np.array([self._pairs[k] for k in keys], dtype=np.int64)
        node_ids = np.array(sorted(self._nodes), dtype=np.int64)
        return _state_from_pairs(self.planes, node_ids, pl, u, v, cnt, self.weight)


@dataclass(frozen=True)
class GraphReading:
    """Entropies of one multiplex state (and its divergence from the previous one).

    nodes; edges[name] per plane and "aggregate"; von_neumann[name] and von_neumann_stderr[name] (NaN:
    no edge); spectral: the aggregate's diffusion ensemble, one entry per tau; relative_entropy: q of
    the unreduced multiplex; best_relative_entropy and best_partition (plane names per group) of the
    q-maximising level; reduced_layers: groups in that partition; jsd_previous[name]: D_QJS to the
    previous state per plane and "aggregate" (NaN when there is no previous state or no edge);
    estimates[name]: the von Neumann estimates themselves (passed to the next reading).
    """

    nodes: int
    edges: dict[str, int]
    von_neumann: dict[str, float]
    von_neumann_stderr: dict[str, float]
    spectral: tuple[SpectralThermo, ...]
    relative_entropy: float
    best_relative_entropy: float
    reduced_layers: int
    best_partition: tuple[tuple[str, ...], ...]
    jsd_previous: dict[str, float]
    estimates: dict[str, Estimate] = field(default_factory=dict)


def graph_reading(state: MultiplexState, *, config: SpectralConfig, previous: MultiplexState | None = None,
                  previous_entropy: Mapping[str, Estimate] | None = None) -> GraphReading:
    """Read a multiplex state: per-plane and aggregate entropies, reducibility, divergence from `previous`.

    previous_entropy: the von Neumann estimates of `previous` per plane and "aggregate" (the `estimates`
    of its own reading). Alignment adds isolated nodes only, which leave a von Neumann entropy unchanged,
    so with the laplacian kind each divergence then needs only its mixture's entropy.
    """
    names = state.planes
    agg = state.aggregate()
    ge = graph_entropies(agg, config=config)
    edges = {p: g.num_edges for p, g in zip(names, state.layers, strict=True)}
    edges["aggregate"] = agg.num_edges
    estimates: dict[str, Estimate] = {}
    if config.multiplex:
        mx = multiplex_entropy(list(state.layers), list(names), config=config, aggregate=ge.von_neumann)
        for i, p in enumerate(names):
            se = float(mx.layer_stderr[i])                                     # 0 exactly when the spectrum was exact
            estimates[p] = Estimate(float(mx.layer_entropy[i]), se, se == 0.0, 0, 0)
        q0, qb = mx.relative_entropy, mx.best_relative_entropy
        part = tuple(tuple(names[i] for i in group) for group in mx.best_partition)
    else:
        for p, g in zip(names, state.layers, strict=True):
            estimates[p] = von_neumann_entropy(g, config=config)
        q0, qb, part = math.nan, math.nan, ()
    estimates["aggregate"] = ge.von_neumann
    jsd: dict[str, float] = {p: math.nan for p in (*names, "aggregate")}
    if previous is not None:
        prev, cur = align_states(previous, state)
        pairs = [*((p, prev.layers[i], cur.layers[i]) for i, p in enumerate(names)), ("aggregate", prev.aggregate(), cur.aggregate())]
        for p, a, b in pairs:
            if config.jsd_kind == "laplacian":
                ha = previous_entropy[p] if previous_entropy is not None and p in previous_entropy else                     von_neumann_entropy(a, config=config)
                jsd[p] = laplacian_jsd(a, b, ha, estimates[p], config).value
            else:
                jsd[p] = quantum_jsd(a, b, config=config).value
    return GraphReading(nodes=state.n, edges=edges, von_neumann={k: e.value for k, e in estimates.items()},
                        von_neumann_stderr={k: e.stderr for k, e in estimates.items()}, spectral=ge.spectral,
                        relative_entropy=q0, best_relative_entropy=qb, reduced_layers=len(part), best_partition=part,
                        jsd_previous=jsd, estimates=estimates)


__all__ = ["MULTICAST_CODE", "GraphReading", "MultiplexState", "SlidingMultiplex", "activity_state", "align_states",
           "contact_state", "graph_reading", "pair_planes"]
