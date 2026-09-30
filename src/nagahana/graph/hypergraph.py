"""The world state as a heterogeneous multiplex hypergraph (D-39, [Q-43], [A-03]).

Structure
---------
    𝒢_t = ( V, τ_V, { (ℰ_p, τ_E,p) }_{p ∈ 𝒫} )

- V: entities, each with a kind τ_V(v) (host, service, account, OT device …): *heterogeneous*.
- 𝒫: relation planes (connectivity, services, identity & auth, OT control … are examples;
  how planes are formed is held, D-04): *multiplex*.
- ℰ_p: hyperedges of plane p. Each is a set of ≥ 2 entities with a kind τ_E,p(e) (a scan
  touching many hosts, a client–service–server session, an OT polling group). A pairwise edge is a
  hyperedge of size 2: *hypergraph*.
- t: the snapshot is the state *as of* time t. Every entity holds its latest state as of t, because
  with event-driven updates two entities rarely share a timestamp (see TSTCT spatial heads).

Representation
--------------
Incidence lists per plane: a LongTensor of shape [2, nnz] whose rows are (node index, hyperedge
index). This is the layout PyTorch Geometric's `HypergraphConv` uses (`hyperedge_index`), so the
production path can pass it straight through. `dense_incidence` builds the V × E_p matrix H_p for the
small reference layers and tests.

Macrostate, not microstate
--------------------------
Raw identities (IPs, ports) are volatile and high-entropy [Q-09], so they key *nodes*. They are not
features. What the model learns from is the structure plus typed, status-tagged features
(ARCH §3.2).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

import torch

from nagahana.core.errors import InvariantViolation


@dataclass(frozen=True)
class HypergraphSnapshot:
    """One snapshot 𝒢_t. See the module docstring.

    Attributes
    ----------
    as_of: snapshot time t (epoch seconds).
    node_kinds: kind of each node; index = node id.
    planes: plane names in a fixed order (the order of the model's parallel branches).
    incidence: plane → LongTensor[2, nnz] of (node, hyperedge) pairs.
    hyperedge_kinds: plane → kind of each hyperedge; index = hyperedge id.
    """

    as_of: float
    node_kinds: tuple[str, ...]
    planes: tuple[str, ...]
    incidence: Mapping[str, torch.Tensor]
    hyperedge_kinds: Mapping[str, tuple[str, ...]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if set(self.planes) != set(self.incidence):
            raise InvariantViolation(
                f"Planes {self.planes} and incidence keys {tuple(self.incidence)} differ."
            )
        n = len(self.node_kinds)
        for p in self.planes:
            inc = self.incidence[p]
            if inc.dtype != torch.long or inc.dim() != 2 or inc.shape[0] != 2:
                raise InvariantViolation(f"plane {p!r}: incidence must be LongTensor[2, nnz]")
            if inc.numel() == 0:
                continue
            if int(inc[0].min()) < 0 or int(inc[0].max()) >= n:
                raise InvariantViolation(f"plane {p!r}: node index out of range [0, {n})")
            e = self.num_hyperedges(p)
            kinds = self.hyperedge_kinds.get(p)
            if kinds is not None and len(kinds) != e:
                raise InvariantViolation(f"plane {p!r}: {len(kinds)} hyperedge kinds for {e} hyperedges")
            # Each hyperedge must join at least two *distinct* entities.
            pairs = torch.unique(inc, dim=1)
            sizes = torch.bincount(pairs[1], minlength=e)
            if bool((sizes < 2).any()):
                bad = torch.nonzero(sizes < 2).flatten().tolist()
                raise InvariantViolation(f"plane {p!r}: hyperedges {bad} join fewer than 2 entities")
        object.__setattr__(self, "incidence", MappingProxyType(dict(self.incidence)))
        object.__setattr__(self, "hyperedge_kinds", MappingProxyType(dict(self.hyperedge_kinds)))

    @property
    def num_nodes(self) -> int:
        """|V|."""
        return len(self.node_kinds)

    def num_hyperedges(self, plane: str) -> int:
        """|ℰ_p|, taken as 1 + the largest hyperedge index (hyperedge ids are dense)."""
        inc = self.incidence[plane]
        return 0 if inc.numel() == 0 else int(inc[1].max()) + 1

    def dense_incidence(self, plane: str, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """H_p ∈ {0,1}^{V × E_p}, with H[v, e] = 1 iff v ∈ e. For reference layers and tests only."""
        inc = self.incidence[plane]
        h = torch.zeros(self.num_nodes, self.num_hyperedges(plane), dtype=dtype)
        if inc.numel():
            h[inc[0], inc[1]] = 1.0
        return h
