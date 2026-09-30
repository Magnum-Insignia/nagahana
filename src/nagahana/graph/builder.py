"""Snapshot builder: state updates in, hypergraph snapshots out (template).

Contract
--------
`apply(update)` folds one `StateUpdate` into the builder's view of the world (event-driven, D-30).
`snapshot(as_of)` returns 𝒢_t with every entity at its latest state as of t.

Why this is a template
----------------------
Two open items decide its internals:
- D-04 (how planes are formed), which says which relations become which plane's hyperedges;
- stage-1 analysis [Q-02], which says how each source's records map to entities and relations.
Until both are settled, `EventDrivenBuilder` raises `NotBuiltYet` instead of guessing.

Implementation notes for whoever builds it
------------------------------------------
- Keep the fold O(touched entities) per update; never rebuild the whole graph per update.
- Volatility [Q-09]: temporary connections and topology shifts are normal. Expire hyperedges by an
  explicit, *time-based* validity (not by count of updates, which an attacker could drive; cf. D-36).
- Every hyperedge should remember which updates created it (for provenance in the Decoder view).
"""

from __future__ import annotations

from typing import Protocol

from nagahana.core.errors import NotBuiltYet
from nagahana.datamodel.records import StateUpdate
from nagahana.graph.hypergraph import HypergraphSnapshot


class SnapshotBuilder(Protocol):
    """What any builder must provide."""

    def apply(self, update: StateUpdate) -> None:
        """Fold one state update into the current world view."""
        ...

    def snapshot(self, as_of: float) -> HypergraphSnapshot:
        """The world as of time `as_of`."""
        ...


class EventDrivenBuilder:
    """Template for the production builder. See the module docstring for why it raises."""

    def apply(self, update: StateUpdate) -> None:
        raise NotBuiltYet("event-driven hypergraph fold", waiting_on=("D-04", "stage-1 analysis"))

    def snapshot(self, as_of: float) -> HypergraphSnapshot:
        raise NotBuiltYet("hypergraph snapshot", waiting_on=("D-04", "stage-1 analysis"))
