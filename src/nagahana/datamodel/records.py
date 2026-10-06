"""State updates: one telemetry record is one state update (D-30).

The unit of input. Each input record is a state update; NagaHana is event-driven and has no fixed
input windows. Windows such as "the next K time windows" are derived views over forecast event times.

Vocabulary (D-31): the state model is the schema (this module, `fields.py`, `native.py`); a state is
an instance of it, the world at a time; a state update is one record that changes part of the state;
a transition is the change a state update causes.

What a StateUpdate carries
--------------------------
- `fields`: `FieldValue`s keyed by catalogue ID, each with its observation status (`status.py`).
  Absence is a status, never a zero (D-41).
- `attributes`: values of source fields the catalogue does not list (a newer source version, a vendor
  extension), keyed "<source namespace>.<name>", each with its status. Retained, never dropped, and
  never model inputs.
- `entities` and `roles`: the typed entities the record touches and the role each plays
  (`ENTITY_ROLES`). They become nodes of the heterogeneous multiplex hypergraph (D-39).
- `ordering`: event time (float64 seconds and, where the source has it, exact integer nanoseconds),
  ingest time, watermark and ordering quality. Temporal order is resolved before the model; residual
  uncertainty is tagged, not hidden.
- `provenance`: source, adapter, record type, where the raw record sits (file, byte offset, line,
  partition offset), the sensor, and the SHA-256 of the raw record (layer L0, chain of custody).

Invariants (checked on construction)
------------------------------------
1. An excluded status (NOT_SUPPLIED, NOT_OBSERVABLE) carries no value; a contributing status carries
   one. LOW_RELIABILITY carries a reliability in (0, 1]; STALE carries an age >= 0.
2. Field keys equal the field IDs they hold. With `strict=True` (the default) every field is in the
   catalogue, its status is admissible for it, and a contributing value has the field's dtype.
3. Attribute keys are namespaced ("<namespace>.<name>") and are not catalogue IDs.
4. At least one entity is touched; `roles` has one role per entity, at most one initiator and at most
   one responder.
5. `ordering.event_time_ns`, when given, agrees with `event_time` to within a microsecond.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from nagahana.core.errors import InvariantViolation
from nagahana.datamodel.fields import CATALOGUE
from nagahana.datamodel.spec import check_value
from nagahana.datamodel.status import CONTRIBUTING, EXCLUDED, ObservationStatus

#: Entity kinds of the heterogeneous hypergraph (`models.vocab.NODE_KINDS` holds the same list).
#: `multicast`: a destination address that names a group of machines, multicast or broadcast (D-47).
ENTITY_KINDS: frozenset[str] = frozenset(
    {"host", "service", "account", "ot_device", "external", "subnet", "application", "multicast"}
)

#: Role an entity plays in a state update.
#:   initiator       the party that started the activity (flow initiator, logon source, alert source)
#:   responder       the party that answered or was targeted (flow responder, logon target host)
#:   service         the responder's service ("<address>:<port>/<protocol>")
#:   account         the account that authenticated or acted
#:   target_account  the account or principal acted upon (a service principal, explicit credentials)
#:   subject         the one entity a record is about when it has no parties (device counters, a host's
#:                   software inventory, a sensor's statistics)
ENTITY_ROLES: tuple[str, ...] = ("initiator", "responder", "service", "account", "target_account", "subject")
ROLE_CODE: dict[str, int] = {r: i for i, r in enumerate(ENTITY_ROLES)}


def default_roles(n: int) -> tuple[str, ...]:
    """Roles of `n` entities given without roles: initiator, responder, then services (flow convention)."""
    return tuple(("initiator", "responder")[i] if i < 2 else "service" for i in range(n))


@dataclass(frozen=True, slots=True)
class EntityRef:
    """A typed entity touched by a state update.

    `id` is an opaque, stable key produced by the adapter (it may be pseudonymised). Raw identity is not
    a model feature (see `fields.py`).
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
    field_id: catalogue ID (`fields.CATALOGUE`), or the key of a retained attribute.
    value: the measured value, or None when the status excludes it.
    status: `ObservationStatus`.
    source: adapter or source that produced it ("netflow", "zeek", "pcap").
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
                f"{self.field_id}: status {self.status.value!r} must carry no value (absence is not zero)."
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
    """Time and ordering quality of a record, resolved before the model.

    Attributes
    ----------
    event_time: when it happened, per the source (UTC epoch seconds, float64).
    ingest_time: when NagaHana received it.
    watermark: event-time watermark at emission (a record older than it arrived out of order).
    reorder_uncertainty_s: residual ordering uncertainty, tagged rather than hidden.
    clock_quality: e.g. "ntp-synced", "skewed", "unknown", "timezone-unverified". Clock manipulation
        is itself a signal, so it is recorded, never silently corrected.
    event_time_ns: the event time as integer nanoseconds since the epoch when the source carries
        sub-microsecond precision (a float64 holds about 0.24 microseconds at present epoch values).
    """

    event_time: float
    ingest_time: float
    watermark: float | None = None
    reorder_uncertainty_s: float | None = None
    clock_quality: str | None = None
    event_time_ns: int | None = None

    def __post_init__(self) -> None:
        ns = self.event_time_ns
        if ns is not None and math.isfinite(self.event_time) and abs(ns / 1e9 - self.event_time) > 1e-6:
            raise InvariantViolation(
                f"event_time_ns {ns} disagrees with event_time {self.event_time!r} by more than a microsecond."
            )


@dataclass(frozen=True, slots=True)
class Provenance:
    """Where a record came from (audit and chain of custody, layer L0).

    Attributes
    ----------
    source_id: name of the source (a file name, a sensor stream).
    adapter, adapter_version: the adapter that built the update.
    raw_hash: SHA-256 (hex) of the raw record's bytes as stored in the source.
    source_type: the format ("zeek-tsv", "suricata-eve", "netflow-v9", "pcap").
    record_type: the record map it was mapped with ("zeek.conn", "windows.4624").
    location: file path, URI or "kafka://<topic>/<partition>".
    offset, length: byte range of the raw record in `location` (the Kafka offset for Kafka).
    line: 1-based line number for line-oriented sources.
    record_index: ordinal of the raw record in the source (0-based).
    sub_index: ordinal of the record inside its container (a flow record inside a datagram).
    sensor_id: the sensor that observed the traffic or produced the log.
    exporter: address of the exporting device (flow export, syslog sender).
    original_time: the source's timestamp text, verbatim.
    """

    source_id: str
    adapter: str
    adapter_version: str
    raw_hash: str | None = None
    source_type: str | None = None
    record_type: str | None = None
    location: str | None = None
    offset: int | None = None
    length: int | None = None
    line: int | None = None
    record_index: int | None = None
    sub_index: int | None = None
    sensor_id: str | None = None
    exporter: str | None = None
    original_time: str | None = None


@dataclass(frozen=True)
class StateUpdate:
    """One telemetry record, as one update to the world state. See the module docstring."""

    update_id: str
    ordering: OrderingInfo
    entities: tuple[EntityRef, ...]
    fields: Mapping[str, FieldValue]
    provenance: Provenance
    strict: bool = field(default=True, compare=False)
    roles: tuple[str, ...] | None = None
    attributes: Mapping[str, FieldValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.entities:
            raise InvariantViolation(f"{self.update_id}: a state update must touch at least one entity.")
        roles = self.roles if self.roles is not None else default_roles(len(self.entities))
        if len(roles) != len(self.entities):
            raise InvariantViolation(f"{self.update_id}: {len(roles)} roles for {len(self.entities)} entities.")
        for r in roles:
            if r not in ROLE_CODE:
                raise InvariantViolation(f"{self.update_id}: unknown role {r!r}; known: {ENTITY_ROLES}")
        if roles.count("initiator") > 1 or roles.count("responder") > 1:
            raise InvariantViolation(f"{self.update_id}: at most one initiator and one responder.")
        object.__setattr__(self, "roles", tuple(roles))
        for key, fv in self.fields.items():
            if key != fv.field_id:
                raise InvariantViolation(f"{self.update_id}: key {key!r} holds field {fv.field_id!r}.")
            if not self.strict:
                continue
            spec = CATALOGUE.get(key)
            if spec is None:
                raise InvariantViolation(
                    f"{self.update_id}: unknown field {key!r} (add it to the catalogue, keep it as an "
                    "attribute, or use strict=False during stage-1 analysis)."
                )
            if not spec.admits(fv.status):
                raise InvariantViolation(
                    f"{self.update_id}: status {fv.status.value!r} is not admissible for {key}."
                )
            if fv.status in CONTRIBUTING:
                why = check_value(spec, fv.value)
                if why is not None:
                    raise InvariantViolation(f"{self.update_id}: {key}: {why}.")
        for key, fv in self.attributes.items():
            if key != fv.field_id:
                raise InvariantViolation(f"{self.update_id}: attribute key {key!r} holds {fv.field_id!r}.")
            if "." not in key:
                raise InvariantViolation(f"{self.update_id}: attribute {key!r} must be namespaced '<ns>.<name>'.")
            if self.strict and key in CATALOGUE:
                raise InvariantViolation(f"{self.update_id}: {key!r} is a catalogue field; put it in `fields`.")
        # Freeze the mappings so the update is immutable end to end.
        object.__setattr__(self, "fields", MappingProxyType(dict(self.fields)))
        object.__setattr__(self, "attributes", MappingProxyType(dict(self.attributes)))

    def contributing(self) -> dict[str, FieldValue]:
        """Fields that count as evidence (status in CONTRIBUTING)."""
        return {k: v for k, v in self.fields.items() if v.contributes}

    def status_of(self, field_id: str) -> ObservationStatus:
        """Status of a field. A field absent from the record is NOT_SUPPLIED by definition."""
        fv = self.fields.get(field_id)
        return fv.status if fv is not None else ObservationStatus.NOT_SUPPLIED

    def entity(self, role: str) -> EntityRef | None:
        """The entity playing `role`, or None (the first one when a role repeats)."""
        assert self.roles is not None
        for e, r in zip(self.entities, self.roles, strict=True):
            if r == role:
                return e
        return None

    def role_pairs(self) -> tuple[tuple[EntityRef, str], ...]:
        """(entity, role) pairs in entity order."""
        assert self.roles is not None
        return tuple(zip(self.entities, self.roles, strict=True))


def placement(u: StateUpdate) -> list[tuple[EntityRef, str] | None]:
    """Entities of an update by column slot (the columnar form's "entity placement").

    Slot 0 holds the initiator, or the subject when the record has no initiator; slot 1 the responder;
    slots 2, 3, ... the remaining entities in the update's order. A slot without an entity is None.
    """
    first: tuple[EntityRef, str] | None = None
    second: tuple[EntityRef, str] | None = None
    rest: list[tuple[EntityRef, str]] = []
    for e, r in u.role_pairs():
        if r == "initiator" and first is None:
            first = (e, r)
        elif r == "responder" and second is None:
            second = (e, r)
        else:
            rest.append((e, r))
    if first is None:
        for k, (e, r) in enumerate(rest):
            if r == "subject":
                first = (e, r)
                rest.pop(k)
                break
    return [first, second, *rest]


def _same_value(a: Any, b: Any) -> bool:
    if isinstance(a, float) or isinstance(b, float):
        try:
            return math.isclose(float(a), float(b), rel_tol=1e-12, abs_tol=1e-12)
        except (TypeError, ValueError):
            return False
    if isinstance(a, tuple | list) and isinstance(b, tuple | list):
        return len(a) == len(b) and all(_same_value(x, y) for x, y in zip(a, b, strict=True))
    if isinstance(a, Mapping) and isinstance(b, Mapping):
        return set(a) == set(b) and all(_same_value(a[k], b[k]) for k in a)
    return bool(a == b)


def _same_field(a: FieldValue | None, b: FieldValue | None) -> bool:
    sa = a.status if a is not None else ObservationStatus.NOT_SUPPLIED
    sb = b.status if b is not None else ObservationStatus.NOT_SUPPLIED
    if sa is not sb:
        return False
    if sa not in CONTRIBUTING:
        return True
    assert a is not None and b is not None
    return _same_value(a.value, b.value) and a.reliability == b.reliability and a.age_s == b.age_s


def differences(a: StateUpdate, b: StateUpdate) -> list[str]:
    """Semantic differences between two state updates (empty when they are equivalent).

    Equivalent means: the same status for every field (a missing field counts as NOT_SUPPLIED), the same
    contributing values, reliabilities and ages, the same attributes, entities with roles, ordering and
    provenance. A field written explicitly as NOT_SUPPLIED equals a field left out (D-41 by definition).
    """
    out: list[str] = []
    for fid in sorted(set(a.fields) | set(b.fields)):
        if not _same_field(a.fields.get(fid), b.fields.get(fid)):
            out.append(f"field {fid}: {a.fields.get(fid)} != {b.fields.get(fid)}")
    for key in sorted(set(a.attributes) | set(b.attributes)):
        if not _same_field(a.attributes.get(key), b.attributes.get(key)):
            out.append(f"attribute {key}: {a.attributes.get(key)} != {b.attributes.get(key)}")
    if placement(a) != placement(b):
        out.append(f"entities: {placement(a)} != {placement(b)}")
    oa, ob = a.ordering, b.ordering
    for name in ("event_time", "ingest_time", "watermark", "reorder_uncertainty_s"):
        va, vb = getattr(oa, name), getattr(ob, name)
        if (va is None) != (vb is None) or (va is not None and not _same_value(va, vb)):
            out.append(f"ordering.{name}: {va!r} != {vb!r}")
    if oa.clock_quality != ob.clock_quality or oa.event_time_ns != ob.event_time_ns:
        out.append(f"ordering: {oa.clock_quality!r}/{oa.event_time_ns!r} != {ob.clock_quality!r}/{ob.event_time_ns!r}")
    if a.provenance != b.provenance:
        out.append(f"provenance: {a.provenance} != {b.provenance}")
    return out


__all__ = [
    "ENTITY_KINDS", "ENTITY_ROLES", "ROLE_CODE", "EntityRef", "FieldValue", "OrderingInfo", "Provenance", "StateUpdate",
    "default_roles", "differences", "placement",
]
