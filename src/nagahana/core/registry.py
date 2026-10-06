"""Component registries: how implementations are swapped without touching their callers.

What this is
------------
A `Registry` maps a name ("hgnn-attn", "jem", "mtu_bound", ...) to a factory. Callers ask the
registry for a component by the name found in configuration, never by importing a concrete class:

- replacing an implementation means registering a new name and changing one configuration value;
- running two variants side by side (an ablation, P-17) means two configurations.

Governance built in
-------------------
An entry can declare
- `requires`: decisions the entry depends on. Given as a sequence of IDs, any option in force is
  accepted; given as a mapping ID -> options, the entry exists only under those options. At build
  time every required decision is resolved with `governance.decisions.require`: a decided entry
  passes, a held entry resolves to its option in force (the run's configured option, else the
  working option of its assumption), and an option the entry does not implement raises
  `InvalidOption`;
- `proposal`: the proposal it implements. Building it requires that ID in `enabled_proposals`.

Registration itself is always allowed and validates the IDs, so a typo fails at import time.

Why not only Hydra's `_target_`?
--------------------------------
Hydra (`hydra.utils.instantiate`) builds objects from import paths. That bypasses the governance
checks above and ties configuration files to module paths. The registry keeps configuration stable
when code moves; Hydra configuration selects registry names, so the two combine.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Generic, TypeVar

from nagahana.core.errors import InvalidOption
from nagahana.governance import decisions

T = TypeVar("T")


@dataclass(frozen=True)
class Entry(Generic[T]):
    """One registered implementation.

    Attributes
    ----------
    name: registry key used in configuration.
    factory: class or function that builds the component.
    requires: IDs of the decisions the entry depends on.
    proposal: proposal ID this entry implements (None if it implements decided or configured design).
    summary: one line shown by the CLI.
    options: decision ID -> the options under which the entry exists (absent: any option in force).
    """

    name: str
    factory: T
    requires: tuple[str, ...]
    proposal: str | None
    summary: str
    options: Mapping[str, frozenset[str]] = field(default_factory=dict)


class Registry(Generic[T]):
    """A named collection of interchangeable implementations of one kind of component."""

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._entries: dict[str, Entry[T]] = {}

    def register(
        self,
        name: str,
        *,
        requires: Iterable[str] | Mapping[str, Iterable[str]] = (),
        proposal: str | None = None,
        summary: str = "",
    ) -> Callable[[T], T]:
        """Decorator: register `factory` under `name`.

        Decision and proposal IDs are validated here, and so is every option named in a `requires`
        mapping (it must be admissible for a held decision), so a typo fails at import, not at run time.
        """
        req: tuple[str, ...]
        opts: dict[str, frozenset[str]] = {}
        if isinstance(requires, Mapping):
            req = tuple(requires)
            for key, allowed in requires.items():
                d = decisions.get(key)                        # KeyError on unknown IDs
                if d.status is not decisions.Status.HELD:
                    raise InvalidOption(f"{name!r}: option restrictions apply to held decisions only, {d.id} is {d.status.value}")
                allowed_set = frozenset(allowed)
                bad = sorted(allowed_set - set(d.admissible))
                if not allowed_set or bad:
                    raise InvalidOption(f"{name!r}: options {bad or '(none)'} are not admissible for {d.id}")
                opts[d.id] = allowed_set
        else:
            req = tuple(requires)
            for key in req:
                decisions.get(key)                           # KeyError on unknown IDs
        if proposal is not None:
            decisions.get(proposal)

        def deco(factory: T) -> T:
            if name in self._entries:
                raise ValueError(f"{self.kind} registry already has an entry named {name!r}")
            self._entries[name] = Entry(name, factory, req, proposal, summary, MappingProxyType(opts))
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

        Order of checks: required decisions first (each resolved to its decided value or to the option
        in force), then the option restrictions of the entry, then the proposal gate.
        """
        e = self.entry(name)
        for key in e.requires:
            d = decisions.require(key, by=f"{self.kind}:{name}")
            allowed = e.options.get(d.id)
            if allowed is not None and d.value not in allowed:
                raise InvalidOption(
                    f"{self.kind} {name!r} exists only under {d.id} options {sorted(allowed)}; "
                    f"the option in force is {d.value!r} (configure it with governance.decisions.configure)."
                )
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
