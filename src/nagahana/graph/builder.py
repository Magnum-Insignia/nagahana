"""Snapshot builder: state updates in, hypergraph snapshots out (event-driven, D-30).

Contract
--------
`apply(update)` folds one `StateUpdate` into the builder's view of the world; `snapshot(as_of)` returns
G_t with every entity at its latest state as of t, as a `HypergraphSnapshot`.

One construction of the hypergraph
----------------------------------
The builder does not construct relations or fans itself. `apply` records the update (its entities, its
event time, its destination port and protocol) in O(touched entities); `snapshot` runs the one
construction of the window-level multiplex hypergraph, `graph.window.build_window_hypergraph`, over the
recorded updates and reads it as of t. Training windows, inference windows and these snapshots therefore
share every rule: the planes of AS-01, the hyperedge kinds of AS-02, the fan episodes of AS-101 and the
visibility order of AS-105.

As-of reading (no future leakage, AS-102, AS-105)
-------------------------------------------------
With R_t = max{ rank(j) : t_j <= t }:
- nodes: the entities first seen in an update with rank <= R_t;
- hyperedge g of plane p is in the snapshot iff since(g) <= R_t and its latest activity with rank <= R_t
  is at most `hyperedge_ttl_s` before t (time-based validity: volatile connections expire by time, never
  by a count of updates an attacker could drive, in the spirit of D-36);
- its members are those that joined with rank <= R_t (a fan grows as the sweep goes on), and it is kept
  when at least two of them are present.
Every hyperedge remembers the update that created it (`hyperedge_origin`, for provenance in the
Decoder view) and its creation time (`hyperedge_since`).

Ordering
--------
Temporal ordering is resolved in the data model before the model [Q-20], so updates arrive in
non-decreasing event time; an update older than the latest one folded is refused rather than silently
re-ordered (the same rule as the engine's event log, `inference/buffer.py`).

Complexity
----------
`apply` is O(touched entities). `snapshot` is O(U log U + G) for U recorded updates and G hyperedges;
the construction is cached and reused by later snapshots until the next `apply`.
"""

from __future__ import annotations

import math
from typing import Protocol

import numpy as np
import torch

from nagahana.core.errors import InvariantViolation
from nagahana.datamodel.records import EntityRef, StateUpdate
from nagahana.graph.hypergraph import HypergraphSnapshot
from nagahana.graph.window import WindowHypergraph, build_window_hypergraph
from nagahana.models.config.components import GraphConfig
from nagahana.models.vocab import HYPEREDGE_KINDS, NODE_KIND_CODE, NODE_KINDS

#: Entity order of a state update by the flow-adapter convention (`datamodel.columnar`): 0 initiator,
#: 1 responder, 2 service.
MAX_ENTITIES_PER_UPDATE = 3


class SnapshotBuilder(Protocol):
    """What any builder must provide."""

    def apply(self, update: StateUpdate) -> None:
        """Fold one state update into the current world view."""
        ...

    def snapshot(self, as_of: float) -> HypergraphSnapshot:
        """The world as of time `as_of`."""
        ...


def _field_value(update: StateUpdate, field_id: str) -> float:
    """The value of a contributing field as a float, NaN when it does not contribute (absence is not zero, D-41)."""
    fv = update.fields.get(field_id)
    if fv is None or not fv.contributes:
        return math.nan
    return float(fv.value)


class EventDrivenBuilder:
    """The event-driven builder of the world hypergraph (module docstring).

    Parameters
    ----------
    cfg: `GraphConfig` (planes, fan rules, hyperedge_ttl_s). The planes are formed by the declared rules of
        AS-01; learned planes (D-04 options with learning) live inside the encoder, over the connectivity
        plane of these snapshots.
    """

    def __init__(self, cfg: GraphConfig) -> None:
        self.cfg = cfg
        self._entity_index: dict[tuple[str, str], int] = {}
        self._entities: list[EntityRef] = []
        self._kind: list[int] = []
        self._time: list[float] = []
        self._ents: list[tuple[int, int, int]] = []
        self._port: list[float] = []
        self._proto: list[float] = []
        self._update_id: list[str] = []
        self._cache: WindowHypergraph | None = None

    def __len__(self) -> int:
        return len(self._time)

    @property
    def entities(self) -> tuple[EntityRef, ...]:
        """Every entity folded so far, in first-seen order (row i of the builder's entity table)."""
        return tuple(self._entities)

    def _row(self, ref: EntityRef) -> int:
        """Stable row of an entity (append-only, keyed by kind and id)."""
        key = (ref.kind, ref.id)
        row = self._entity_index.get(key)
        if row is None:
            row = len(self._entities)
            self._entity_index[key] = row
            self._entities.append(ref)
            self._kind.append(NODE_KIND_CODE[ref.kind])
        return row

    def apply(self, update: StateUpdate) -> None:
        """Record one state update: O(touched entities). See the module docstring for the ordering rule."""
        t = float(update.ordering.event_time)
        if not math.isfinite(t):
            raise InvariantViolation(f"{update.update_id}: the event time must be finite")
        if self._time and t < self._time[-1]:
            raise InvariantViolation(
                f"{update.update_id}: event time {t} precedes the latest folded update ({self._time[-1]}); "
                "ordering is resolved in the data model before the model ([Q-20])"
            )
        if len(update.entities) > MAX_ENTITIES_PER_UPDATE:
            raise InvariantViolation(
                f"{update.update_id}: {len(update.entities)} entities; the flow-adapter convention has at most "
                "initiator, responder and service"
            )
        rows = [self._row(ref) for ref in update.entities]
        rows += [-1] * (MAX_ENTITIES_PER_UPDATE - len(rows))
        self._time.append(t)
        self._ents.append((rows[0], rows[1], rows[2]))
        self._port.append(_field_value(update, "flow.dst_port"))
        self._proto.append(_field_value(update, "flow.protocol"))
        self._update_id.append(update.update_id)
        self._cache = None

    def hypergraph(self) -> WindowHypergraph:
        """The window-level hypergraph over every recorded update (built once per batch of `apply` calls)."""
        if self._cache is None:
            self._cache = build_window_hypergraph(
                update_time=np.asarray(self._time, dtype=np.float64),
                update_entities=np.asarray(self._ents, dtype=np.int64).reshape(-1, 3),
                dst_port=np.asarray(self._port, dtype=np.float64),
                protocol=np.asarray(self._proto, dtype=np.float64),
                entity_kind=np.asarray(self._kind, dtype=np.int64),
                cfg=self.cfg,
            )
        return self._cache

    def snapshot(self, as_of: float) -> HypergraphSnapshot:
        """G_t as of `as_of` (module docstring): seen entities, alive hyperedges, members joined by then."""
        if not math.isfinite(as_of):
            raise InvariantViolation("as_of must be a finite time")
        planes = tuple(self.cfg.planes)
        empty = {p: torch.zeros(2, 0, dtype=torch.long) for p in planes}
        if not self._time:
            return HypergraphSnapshot(float(as_of), (), planes, empty, {p: () for p in planes}, (),
                                      {p: () for p in planes}, {p: () for p in planes})
        wh = self.hypergraph()
        r = wh.visible_rank(float(as_of))
        # Nodes: entities whose first update is visible as of t, numbered in first-seen order.
        ents = np.asarray(self._ents, dtype=np.int64)
        visible_updates = wh.order[: r + 1] if r >= 0 else np.zeros(0, dtype=np.int64)
        seen_entities = ents[visible_updates].reshape(-1)
        seen_entities = seen_entities[seen_entities >= 0]
        node_of = np.full(len(self._entities), -1, dtype=np.int64)
        seen_sorted = np.unique(seen_entities)                    # rows are appended in first-seen order
        node_of[seen_sorted] = np.arange(seen_sorted.size)
        node_kinds = tuple(NODE_KINDS[self._kind[e]] for e in seen_sorted.tolist())
        node_ids = tuple(f"{self._entities[e].kind}:{self._entities[e].id}" for e in seen_sorted.tolist())
        alive = wh.alive(r, float(as_of), self.cfg.hyperedge_ttl_s)
        hx = wh.hyperedges
        incidence: dict[str, torch.Tensor] = {}
        kinds: dict[str, tuple[str, ...]] = {}
        since: dict[str, tuple[float, ...]] = {}
        origin: dict[str, tuple[str, ...]] = {}
        for pi, plane in enumerate(planes):
            rows_n: list[int] = []
            rows_e: list[int] = []
            k_p: list[str] = []
            s_p: list[float] = []
            o_p: list[str] = []
            for g in np.nonzero(alive & (hx.plane == pi))[0].tolist():
                members = node_of[wh.members_as_of(g, r)]
                members = np.unique(members[members >= 0])
                if members.size < 2:
                    continue
                e = len(k_p)
                rows_n.extend(members.tolist())
                rows_e.extend([e] * members.size)
                k_p.append(HYPEREDGE_KINDS[int(hx.kind[g])])
                s_p.append(float(wh.times_sorted[hx.since_rank[g]]))
                o_p.append(self._update_id[int(wh.order[hx.since_rank[g]])])
            incidence[plane] = torch.tensor([rows_n, rows_e], dtype=torch.long).reshape(2, -1)
            kinds[plane], since[plane], origin[plane] = tuple(k_p), tuple(s_p), tuple(o_p)
        return HypergraphSnapshot(float(as_of), node_kinds, planes, incidence, kinds, node_ids, since, origin)
