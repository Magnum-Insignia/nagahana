"""Schema versions and stable field slots (P-22).

Every field has a stable slot: a row of an embedding table keyed by field ID, plus reserved spare
rows. The input layer embeds each contributing field as

    e_i = E_field[slot(i)] + g(value_i) + E_status[m_i]

and pools over the set of contributing fields (a set encoder, so order and count are free). Then
adding a field takes a reserved row (existing rows and weights are untouched; only the new row is
fine-tuned), retiring a field means never emitting it again (its row stays, so old checkpoints still
load), and absent fields contribute nothing (D-41).

Version rules
-------------
- Adding a field, or adding codes to a categorical code space: minor version bump; a new matrix field
  takes the next reserved slot (`data.windows.COLUMN_SLOTS` is append-only).
- Changing a field's meaning, unit or the meaning of an existing code: major bump (old data must be
  migrated or re-derived).
- Removing a field: major bump; the slot is retired, never reused.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from nagahana.core.errors import InvariantViolation


@dataclass(frozen=True, order=True)
class SchemaVersion:
    """Semantic version of the state model (see the version rules above)."""

    major: int
    minor: int

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}"

    @classmethod
    def parse(cls, text: str) -> SchemaVersion:
        """Parse "<major>.<minor>"."""
        try:
            major, minor = (int(x) for x in str(text).split("."))
        except ValueError:
            raise InvariantViolation(f"Not a schema version: {text!r}") from None
        return cls(major, minor)

    def reads(self, other: SchemaVersion) -> bool:
        """True if data written under `other` can be read by this version (same major, not newer)."""
        return self.major == other.major and other.minor <= self.minor


#: Changes of the data model, oldest first.
CHANGES: tuple[tuple[SchemaVersion, str], ...] = (
    (SchemaVersion(0, 1), "Initial state model: flow-level, packet-level and observable-region example fields; "
                          "five observation statuses; state updates with entities, ordering and provenance."),
    (SchemaVersion(0, 2), "Superset of every supported telemetry format: shared fields of the new formats (64 new "
                          "matrix fields appended to the column slots), source-native fields generated from the "
                          "mapping tables, field layers, dtypes and admissible statuses, the attribute kind, "
                          "flow.end_reason codes 5 to 8, entity roles, retained attributes, record locations in "
                          "provenance and exact nanosecond event times."),
)

#: Version of the data model this code writes.
DATA_MODEL_VERSION: SchemaVersion = CHANGES[-1][0]


class FieldIndex:
    """Stable mapping field ID -> slot, with reserved capacity (P-22).

    Parameters
    ----------
    field_ids:
        Fields in slot order. The order is frozen once a model is trained on it.
    reserved:
        Number of spare slots. There is no default: the number is a sizing decision recorded in config.
    """

    def __init__(self, field_ids: Iterable[str], *, reserved: int) -> None:
        ids = list(field_ids)
        if len(set(ids)) != len(ids):
            raise InvariantViolation("Duplicate field IDs in FieldIndex")
        if reserved < 0:
            raise ValueError("reserved must be >= 0")
        self._slots: dict[str, int] = {f: i for i, f in enumerate(ids)}
        self._retired: set[str] = set()
        self._capacity = len(ids) + reserved

    @property
    def capacity(self) -> int:
        """Total rows the embedding table must have (used + reserved)."""
        return self._capacity

    @property
    def used(self) -> int:
        """Rows taken, including retired ones (retired rows are never reused)."""
        return len(self._slots)

    def slot(self, field_id: str) -> int:
        """Slot of an active field."""
        if field_id in self._retired:
            raise InvariantViolation(f"{field_id} is retired; its slot is kept but must not be emitted")
        return self._slots[field_id]

    def add(self, field_id: str) -> int:
        """Take the next reserved slot for a new field (a minor schema bump)."""
        if field_id in self._slots:
            raise InvariantViolation(f"{field_id} already has slot {self._slots[field_id]}")
        if self.used >= self._capacity:
            raise InvariantViolation(
                "No reserved slots left; growing capacity is a major change (new embedding rows)."
            )
        self._slots[field_id] = self.used
        return self._slots[field_id]

    def retire(self, field_id: str) -> None:
        """Stop emitting a field; keep its slot so older checkpoints still load."""
        if field_id not in self._slots:
            raise KeyError(field_id)
        self._retired.add(field_id)
