"""State updates: one telemetry record = one state update (D-30, [Q-25]).

The unit of input
-----------------
"we have to input as the state (each input record is a state update)" [Q-25]. NagaHana is
event-driven: there are no fixed time windows at the input. Windows such as "the next K time
windows" in the problem statement are *derived views* over forecast event times (DESIGN_LOG,
2026-09-28).

Vocabulary (D-31, [Q-25])
-------------------------
- **state model**: the schema (this module plus `fields.py`);
- **state**: an instance of it, i.e. the world at a time;
- **state update**: one record that changes part of the state;
- **transition**: the change a state update causes.

Never "token": "a token is like an instance of a state while a state is a data model" [Q-25].

What a StateUpdate carries
--------------------------
- `fields`: `FieldValue`s keyed by catalogue ID. Each value carries its observation status
  (`status.py`). Absence is a status, never a zero (D-41).
- `entities`: the typed entities the record touches (host, service, account, OT device …). They
  become nodes of the heterogeneous multiplex hypergraph (D-39).
- `ordering`: event time, ingest time and ordering quality. Temporal ordering is resolved *before*
  the model, in the data model [Q-20]. Residual uncertainty is tagged here, not hidden (ARCH §9.2).
- `provenance`: which source and adapter produced it, plus the hash of the raw record (L0). This
  supports audit and forensic chain of custody (ARCH §7, §10).

Invariants (checked on construction)
------------------------------------
1. An excluded status (NOT_SUPPLIED, NOT_OBSERVABLE) carries no value (`value is None`).
2. A contributing status carries a value.
3. LOW_RELIABILITY carries a reliability in (0, 1]; STALE carries an age >= 0.
4. Field keys equal the field IDs they hold; unknown IDs are rejected unless `strict=False`
   (used by stage-1 analysis while the catalogue is still being extended).
5. At least one entity is touched.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from nagahana.core.errors import InvariantViolation
from nagahana.datamodel.fields import CATALOGUE
from nagahana.datamodel.status import CONTRIBUTING, EXCLUDED, ObservationStatus

#: Entity kinds of the heterogeneous hypergraph (diagram 01 legend). Extend when D-04 settles planes.
ENTITY_KINDS: frozenset[str] = frozenset(
    {"host", "service", "account", "ot_device", "external", "subnet", "application"}
)


@dataclass(frozen=True, slots=True)
class EntityRef:
    """A typed entity touched by a state update.

    `id` is an opaque, stable key produced by the adapter (it may be pseudonymised). Raw identity is
    not a model feature (see `fields.py`).
    """

    kind: str
    id: str

    def __post_init__(self) -> None:
        if self.kind not in ENTITY_KINDS:
            raise InvariantViolation(f"Unknown entity kind {self.kind!r}; known: {sorted(ENTITY_KINDS)}")
        if not self.id:
            raise InvariantViolation("EntityRef.id must be non-empty")


@dataclass(frozen=True, slots=True)
class FieldValue:
    """One field of one state update, with its evidence status.

    Attributes
    ----------
    field_id: catalogue ID (`fields.CATALOGUE`).
    value: the measured value, or None when the status excludes it.
    status: `ObservationStatus`.
    source: adapter/source that produced it (e.g. "netflow", "zeek", "pcap").
    reliability: in (0, 1]; required for LOW_RELIABILITY, optional otherwise.
    age_s: seconds since observation; required for STALE.
    """

    field_id: str
    value: Any
    status: ObservationStatus
    source: str
    reliability: float | None = None
    age_s: float | None = None

    def __post_init__(self) -> None:
        if self.status in EXCLUDED and self.value is not None:
            raise InvariantViolation(
                f"{self.field_id}: status {self.status.value!r} must carry no value (absence ≠ zero)."
            )
        if self.status in CONTRIBUTING and self.value is None:
            raise InvariantViolation(
                f"{self.field_id}: status {self.status.value!r} requires a value; use NOT_SUPPLIED "
                "or NOT_OBSERVABLE if there is none."
            )
        if self.reliability is not None and not 0.0 < self.reliability <= 1.0:
            raise InvariantViolation(f"{self.field_id}: reliability must be in (0, 1].")
        if self.status is ObservationStatus.LOW_RELIABILITY and self.reliability is None:
            raise InvariantViolation(f"{self.field_id}: LOW_RELIABILITY requires a reliability.")
        if self.status is ObservationStatus.STALE and (self.age_s is None or self.age_s < 0):
            raise InvariantViolation(f"{self.field_id}: STALE requires age_s >= 0.")

    @property
    def contributes(self) -> bool:
        """True if this value counts as evidence."""
        return self.status in CONTRIBUTING


@dataclass(frozen=True, slots=True)
class OrderingInfo:
    """Time and ordering quality of a record, resolved before the model [Q-20].

    Attributes
    ----------
    event_time: when it happened, per the source (epoch seconds).
    ingest_time: when NagaHana received it.
    watermark: event-time watermark at emission (records later than it arrived out of order).
    reorder_uncertainty_s: residual ordering uncertainty, tagged rather than hidden.
    clock_quality: e.g. "ntp-synced", "skewed", "unknown". Clock manipulation is itself a signal
        (ARCH §9.2), so it is recorded, not silently corrected.
    """

    event_time: float
    ingest_time: float
    watermark: float | None = None
    reorder_uncertainty_s: float | None = None
    clock_quality: str | None = None


@dataclass(frozen=True, slots=True)
class Provenance:
    """Where a record came from (audit and chain of custody).

    `raw_hash` is the hash of the immutable L0 raw record (proposal P-04 layering; P-02 event log).
    """

    source_id: str
    adapter: str
    adapter_version: str
    raw_hash: str | None = None


@dataclass(frozen=True)
class StateUpdate:
    """One telemetry record, as one update to the world state. See the module docstring."""

    update_id: str
    ordering: OrderingInfo
    entities: tuple[EntityRef, ...]
    fields: Mapping[str, FieldValue]
    provenance: Provenance
    strict: bool = field(default=True, compare=False)

    def __post_init__(self) -> None:
        if not self.entities:
            raise InvariantViolation(f"{self.update_id}: a state update must touch at least one entity.")
        for key, fv in self.fields.items():
            if key != fv.field_id:
                raise InvariantViolation(f"{self.update_id}: key {key!r} holds field {fv.field_id!r}.")
            if self.strict and key not in CATALOGUE:
                raise InvariantViolation(
                    f"{self.update_id}: unknown field {key!r} (add it to datamodel/fields.py, or "
                    "use strict=False during stage-1 analysis)."
                )
        # Freeze the mapping so the update is immutable end to end.
        object.__setattr__(self, "fields", MappingProxyType(dict(self.fields)))

    def contributing(self) -> dict[str, FieldValue]:
        """Fields that count as evidence (status in CONTRIBUTING)."""
        return {k: v for k, v in self.fields.items() if v.contributes}

    def status_of(self, field_id: str) -> ObservationStatus:
        """Status of a field. A field absent from the record is NOT_SUPPLIED by definition."""
        fv = self.fields.get(field_id)
        return fv.status if fv is not None else ObservationStatus.NOT_SUPPLIED
