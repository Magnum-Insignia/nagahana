"""Component registries: how implementations are swapped without touching their callers.

What this is
------------
A `Registry` maps a name ("hgnn-reference", "jem", "mtu_bound", …) to a factory. Callers ask the
registry for a component by the name found in config, never by importing a concrete class. This is
the mechanism behind the owner's request for a codebase that is "highly decoupled, transformable
further, modular … for multiple changes over and over" [Q-45]:
- replacing an implementation = registering a new name + changing one config value;
- running two variants side by side (an ablation, P-17) = two config files.

Governance built in
-------------------
An entry can declare
- `requires=("D-12", ...)`: decisions that must be DECIDED before it can be built. Building it
  while they are held raises `DecisionHeld` (no silent defaults [Q-39]);
- `proposal="P-18"`: the proposal it implements. Building it requires that ID in
  `enabled_proposals`.
Registration itself is always allowed, so templates and experiments can live in the tree without
running by accident.

Why not only Hydra's `_target_`?
--------------------------------
Hydra (`hydra.utils.instantiate`) builds objects from import paths. That is convenient, but it
bypasses the governance checks above and ties configs to module paths. The registry keeps configs
stable when code moves. Hydra configs select registry names, so the two combine.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Iterable
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from nagahana.governance import decisions

T = TypeVar("T")


@dataclass(frozen=True)
class Entry(Generic[T]):
    """One registered implementation.

    Attributes
    ----------
    name: registry key used in config.
    factory: class or function that builds the component.
    requires: decisions that must be DECIDED before building.
    proposal: proposal ID this entry implements (None if it implements decided design).
    summary: one line shown by the CLI.
    """

    name: str
    factory: T
    requires: tuple[str, ...]
    proposal: str | None
    summary: str


class Registry(Generic[T]):
    """A named collection of interchangeable implementations of one kind of component."""

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._entries: dict[str, Entry[T]] = {}

    def register(
        self,
        name: str,
        *,
        requires: Iterable[str] = (),
        proposal: str | None = None,
        summary: str = "",
    ) -> Callable[[T], T]:
        """Decorator: register `factory` under `name`.

        The decision and proposal IDs are validated at registration time. A typo in an ID fails
        at import, not later at run time.
        """
        req = tuple(requires)
        for key in req:
            decisions.get(key)  # raises KeyError on unknown IDs
        if proposal is not None:
            decisions.get(proposal)

        def deco(factory: T) -> T:
            if name in self._entries:
                raise ValueError(f"{self.kind} registry already has an entry named {name!r}")
            self._entries[name] = Entry(name, factory, req, proposal, summary)
            return factory

        return deco

    def entry(self, name: str) -> Entry[T]:
        """Look up an entry, with a helpful error listing what exists."""
        try:
            return self._entries[name]
        except KeyError:
            known = ", ".join(sorted(self._entries)) or "(none registered)"
            raise KeyError(f"No {self.kind} named {name!r}. Registered: {known}") from None

    def build(
        self,
        name: str,
        /,
        *args: Any,
        enabled_proposals: Collection[str] = (),
        **kwargs: Any,
    ) -> Any:
        """Build a component after checking its governance gates.

        Order of checks: required decisions first (they are the owner's), then the proposal gate.
        """
        e = self.entry(name)
        for key in e.requires:
            decisions.require(key)
        if e.proposal is not None:
            decisions.require_proposal(e.proposal, enabled_proposals)
        factory: Any = e.factory
        return factory(*args, **kwargs)

    def names(self) -> tuple[str, ...]:
        """Registered names, sorted."""
        return tuple(sorted(self._entries))

    def entries(self) -> tuple[Entry[T], ...]:
        """Registered entries, sorted by name."""
        return tuple(self._entries[n] for n in self.names())

    def __contains__(self, name: object) -> bool:
        return name in self._entries

    def __len__(self) -> int:
        return len(self._entries)
