"""The world state as a heterogeneous multiplex hypergraph (D-39, [Q-43], [A-03]).

Structure
---------
    G_t = ( V, tau_V, { (E_p, tau_E_p) } for p in P )

- V: entities, each with a kind tau_V(v) (host, service, account, OT device, ...): heterogeneous.
- P: relation planes (connectivity, services, identity, remote_admin, name_resolution, ot_control under
  AS-01; how planes are formed is the held D-04, configured in governance/decisions.py): multiplex.
- E_p: hyperedges of plane p. Each is a set of >= 2 entities with a kind tau_E_p(e) (a scan touching
  many hosts, a client-service-server session, an OT polling group). A pairwise edge is a hyperedge of
  size 2: hypergraph.
- t: the snapshot is the state as of time t. Every entity holds its latest state as of t, because with
  event-driven updates two entities rarely share a timestamp (see the TSTCT spatial heads).

Representation
--------------
Incidence lists per plane: a LongTensor [2, nnz] whose rows are (node index, hyperedge index). This is
the layout of PyTorch Geometric's `HypergraphConv` (`hyperedge_index`). `dense_incidence` builds the
V x E_p matrix H_p for the reference layer and tests.

Provenance (optional fields)
----------------------------
- `node_ids`: the entity key of each node (for views and explanations; never a model input).
- `hyperedge_since`: plane -> event time from which each hyperedge existed on that plane.
- `hyperedge_origin`: plane -> the update id of the state update that created each hyperedge, so a view
  can show which observation backs a relation.

Macrostate, not microstate
--------------------------
Raw identities (addresses, ports) are volatile and high-entropy [Q-09], so they key nodes and are not
features. What the model learns from is the structure plus typed, status-tagged features.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

import torch

from nagahana.core.errors import InvariantViolation


@dataclass(frozen=True)
class HypergraphSnapshot:
    """One snapshot G_t. See the module docstring.

    Attributes
    ----------
    as_of: snapshot time t (epoch seconds).
    node_kinds: kind of each node; index = node id.
    planes: plane names in a fixed order (the order of the model's parallel branches).
    incidence: plane -> LongTensor[2, nnz] of (node, hyperedge) pairs.
    hyperedge_kinds: plane -> kind of each hyperedge; index = hyperedge id.
    node_ids: entity key of each node (empty when not tracked).
    hyperedge_since: plane -> creation time of each hyperedge (empty mapping when not tracked).
    hyperedge_origin: plane -> id of the state update that created each hyperedge (empty when not tracked).
    """

    as_of: float
    node_kinds: tuple[str, ...]
    planes: tuple[str, ...]
    incidence: Mapping[str, torch.Tensor]
    hyperedge_kinds: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    node_ids: tuple[str, ...] = ()
    hyperedge_since: Mapping[str, tuple[float, ...]] = field(default_factory=dict)
    hyperedge_origin: Mapping[str, tuple[str, ...]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if set(self.planes) != set(self.incidence):
            raise InvariantViolation(
                f"Planes {self.planes} and incidence keys {tuple(self.incidence)} differ."
            )
        n = len(self.node_kinds)
        if self.node_ids and len(self.node_ids) != n:
            raise InvariantViolation(f"{len(self.node_ids)} node ids for {n} nodes")
        for p in self.planes:
            inc = self.incidence[p]
            if inc.dtype != torch.long or inc.dim() != 2 or inc.shape[0] != 2:
                raise InvariantViolation(f"plane {p!r}: incidence must be LongTensor[2, nnz]")
            e = self.num_hyperedges(p)
            for name, table in (("hyperedge kinds", self.hyperedge_kinds), ("hyperedge since", self.hyperedge_since),
                                ("hyperedge origins", self.hyperedge_origin)):
                col = table.get(p)
                if col is not None and len(col) != e:
                    raise InvariantViolation(f"plane {p!r}: {len(col)} {name} for {e} hyperedges")
            if inc.numel() == 0:
                continue
            if int(inc[0].min()) < 0 or int(inc[0].max()) >= n:
                raise InvariantViolation(f"plane {p!r}: node index out of range [0, {n})")
            # Each hyperedge must join at least two distinct entities.
            pairs = torch.unique(inc, dim=1)
            sizes = torch.bincount(pairs[1], minlength=e)
            if bool((sizes < 2).any()):
                bad = torch.nonzero(sizes < 2).flatten().tolist()
                raise InvariantViolation(f"plane {p!r}: hyperedges {bad} join fewer than 2 entities")
        object.__setattr__(self, "incidence", MappingProxyType(dict(self.incidence)))
        object.__setattr__(self, "hyperedge_kinds", MappingProxyType(dict(self.hyperedge_kinds)))
        object.__setattr__(self, "hyperedge_since", MappingProxyType(dict(self.hyperedge_since)))
        object.__setattr__(self, "hyperedge_origin", MappingProxyType(dict(self.hyperedge_origin)))

    @property
    def num_nodes(self) -> int:
        """|V|."""
        return len(self.node_kinds)

    def num_hyperedges(self, plane: str) -> int:
        """|E_p|, taken as 1 + the largest hyperedge index (hyperedge ids are dense)."""
        inc = self.incidence[plane]
        return 0 if inc.numel() == 0 else int(inc[1].max()) + 1

    def dense_incidence(self, plane: str, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """H_p in {0, 1}^(V x E_p), with H[v, e] = 1 iff v is a member of e. For the reference layer and tests."""
        inc = self.incidence[plane]
        h = torch.zeros(self.num_nodes, self.num_hyperedges(plane), dtype=dtype)
        if inc.numel():
            h[inc[0], inc[1]] = 1.0
        return h
