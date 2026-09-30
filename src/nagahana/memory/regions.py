"""Memory regions with enforced ownership (D-35).

`MemoryRegion` wraps a key–value store and checks `memory/access.py` on every read and write. The
access rule is enforced where memory is touched, not at call sites that could forget it.

`DictStore` is a plain in-memory store for tests and small lab experiments. Production stores
(KV-cache stores, the event log, drift statistics) plug in behind the same two methods.
"""

from __future__ import annotations

from collections.abc import Collection
from typing import Any, Protocol

from nagahana.core.roles import Role
from nagahana.memory.access import Op, Region, check


class Store(Protocol):
    """Minimal storage contract."""

    def get(self, key: str) -> Any:
        """Value stored under `key` (KeyError if absent)."""
        ...

    def put(self, key: str, value: Any) -> None:
        """Store `value` under `key`."""
        ...


class DictStore:
    """In-memory store (tests and lab only)."""

    def __init__(self) -> None:
        self._d: dict[str, Any] = {}

    def get(self, key: str) -> Any:
        return self._d[key]

    def put(self, key: str, value: Any) -> None:
        self._d[key] = value


class MemoryRegion:
    """One region, with access checked on every operation.

    Parameters
    ----------
    region: which region this is.
    store: the backing store.
    enabled_proposals: proposal IDs enabled in this run (some rights exist only under proposals).
    """

    def __init__(self, region: Region, store: Store, *, enabled_proposals: Collection[str] = ()) -> None:
        self.region = region
        self._store = store
        self._enabled = frozenset(enabled_proposals)

    def read(self, role: Role, key: str) -> Any:
        check(role, self.region, Op.READ, self._enabled)
        return self._store.get(key)

    def write(self, role: Role, key: str, value: Any) -> None:
        check(role, self.region, Op.WRITE, self._enabled)
        self._store.put(key, value)
