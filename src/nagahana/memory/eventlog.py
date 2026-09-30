"""Append-only, hash-chained event log (proposal P-02; relates to D-15 and P-18).

Why propose it
--------------
- **Durable Environment.** KV caches grow with history and are tied to one set of weights
  (`kvcache.py`). An append-only log of state updates is model-independent. Caches, snapshots and
  any future memory can be rebuilt from it after retraining (P-18). This answers the months-lossless
  constraint [Q-19] without making a model's working memory the only copy of the facts
  (ARCH §5.2 "memory-centric, not processing-centric").
- **Chain of custody.** Each entry commits to the previous one:

      h_i = SHA-256( h_{i−1} ‖ SHA-256(payload_i) ),      h_{−1} = 0^{256}

  so any edit, deletion or reordering breaks every later hash. That gives tamper evidence for
  forensic replay and legal admissibility (ARCH §7, §10.3), and it makes memory-drift and poisoning
  analysis (the Verifier, [A-16]) replayable.

This module only chains payload bytes. Storage (files, object store, Kafka compacted topic) is an
adapter concern. The class refuses to run until P-02 is enabled, because it is a proposal.
"""

from __future__ import annotations

import hashlib
from collections.abc import Collection
from dataclasses import dataclass

from nagahana.core.errors import InvariantViolation
from nagahana.governance import decisions

_GENESIS = "0" * 64


@dataclass(frozen=True)
class LogEntry:
    """One committed entry."""

    index: int
    prev_hash: str
    payload_hash: str
    entry_hash: str
    payload: bytes


def _h(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class HashChainLog:
    """In-memory hash chain (reference implementation of P-02)."""

    def __init__(self, *, enabled_proposals: Collection[str]) -> None:
        decisions.require_proposal("environment-event-sourcing", enabled_proposals)
        self._entries: list[LogEntry] = []

    @property
    def head(self) -> str:
        """Hash of the last entry (the value to anchor or sign)."""
        return self._entries[-1].entry_hash if self._entries else _GENESIS

    def append(self, payload: bytes) -> LogEntry:
        """Commit one payload and return its entry."""
        prev = self.head
        ph = _h(payload)
        entry = LogEntry(len(self._entries), prev, ph, _h((prev + ph).encode("ascii")), payload)
        self._entries.append(entry)
        return entry

    def verify(self) -> None:
        """Recompute the chain; raise `InvariantViolation` at the first broken link."""
        prev = _GENESIS
        for e in self._entries:
            if e.prev_hash != prev or e.payload_hash != _h(e.payload):
                raise InvariantViolation(f"log entry {e.index} does not match its predecessor or payload")
            if e.entry_hash != _h((prev + e.payload_hash).encode("ascii")):
                raise InvariantViolation(f"log entry {e.index} has a wrong entry hash")
            prev = e.entry_hash

    def __len__(self) -> int:
        return len(self._entries)

    def entries(self) -> tuple[LogEntry, ...]:
        """All entries in order (read-only copy)."""
        return tuple(self._entries)
