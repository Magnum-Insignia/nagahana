"""Source-native catalogue and field-by-field mapping tables.

Every supported telemetry format describes its records with a `RecordMap`: one `Row` per source field,
in the source's own order. A row says where the field goes:

    target = a shared catalogue ID     the field is mapped onto a shared field (fields.py), through the
                                        named converter (`conv`, implemented in ingest/convert.py)
    target = None                      the field is kept as a source-native catalogue field
                                        "<prefix>.<source field>" with the row's dtype, unit and kind
    target = "@time"                   the field is the record's event time (ordering, not a field)
    also = "native" or a shared ID     the source value is copied there unchanged as well, where the
                                        mapping onto `target` is not one-to-one (code tables, label lists)

So every field of every source is mapped onto exactly one catalogue field (plus an optional verbatim
copy), and the tables double as the documentation (`ingest.docs` renders them into docs/ingest.md).
Fields a record carries that its table does not list (a newer source version, a vendor extension)
are retained as uncatalogued attributes of the state update (`records.StateUpdate.attributes`), never
dropped.

This module holds data and catalogue generation only; parsing and conversion live in `ingest`.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from nagahana.core.errors import InvariantViolation
from nagahana.datamodel.layers import Layer
from nagahana.datamodel.spec import Dtype, FieldSpec, Kind, Level
from nagahana.datamodel.status import IDENTITY_STATUSES, MEASUREMENT_STATUSES, ObservationStatus

#: Short dtype codes used by the compact tables.
DTYPE_CODES: dict[str, Dtype] = {
    "i": Dtype.INT, "f": Dtype.FLOAT, "b": Dtype.BOOL, "s": Dtype.STR, "t": Dtype.TIME, "a": Dtype.ADDRESS,
    "m": Dtype.MAC, "y": Dtype.BYTES, "I": Dtype.INT_LIST, "F": Dtype.FLOAT_LIST, "S": Dtype.STR_LIST,
    "A": Dtype.ADDRESS_LIST, "T": Dtype.TIME_LIST, "M": Dtype.MAP,
}
#: Short kind codes for native fields.
KIND_CODES: dict[str, Kind] = {"attr": Kind.ATTRIBUTE, "id": Kind.IDENTIFIER, "fp": Kind.FINGERPRINT}

#: Special targets (not catalogue fields).
TIME_TARGET = "@time"
SPECIAL_TARGETS: frozenset[str] = frozenset({TIME_TARGET})


@dataclass(frozen=True)
class Row:
    """One source field of a record type and where it goes (see the module docstring)."""

    source: str
    dtype: Dtype
    target: str | None = None
    conv: str = ""
    unit: str | None = None
    kind: Kind = Kind.ATTRIBUTE
    also: str | None = None
    note: str = ""
    native: str | None = None

    @property
    def is_native(self) -> bool:
        """True if the row creates a source-native catalogue field."""
        return self.target is None or self.also == "native"


def R(source: str, dtype: str, target: str | None = None, conv: str = "", *, unit: str | None = None,
      kind: str = "attr", also: str | None = None, note: str = "", native: str | None = None) -> Row:
    """Compact row constructor: dtype and kind by their short codes (`DTYPE_CODES`, `KIND_CODES`).

    `native` overrides the native field's ID (fields shared by every record type of one source, such as
    the envelope of Suricata EVE events, keep one native ID across the record types).
    """
    if dtype not in DTYPE_CODES:
        raise InvariantViolation(f"Unknown dtype code {dtype!r} for {source!r}")
    if kind not in KIND_CODES:
        raise InvariantViolation(f"Unknown kind code {kind!r} for {source!r}")
    return Row(source, DTYPE_CODES[dtype], target, conv, unit, KIND_CODES[kind], also, note, native)


@dataclass(frozen=True)
class RecordMap:
    """The field-by-field mapping of one record type of one source format.

    Attributes
    ----------
    source, record:
        Format and record type; `record_type` is "<source>.<record>".
    title:
        Human name of the record type (for the documentation).
    level:
        Level of the native fields.
    rows:
        The source fields, in source order.
    reference:
        Where the record layout is defined (specification or documentation).
    layer:
        Layer of the native fields (L1 by default).
    native_prefix:
        Prefix of native field IDs; default "<source>.<record>".
    native_statuses:
        Admissible statuses of the native fields; default by kind (identity or measurement).
    ocsf_class:
        OCSF class the record type exports to (ingest/ocsf.py), 0 for the OCSF base event.
    notes:
        Record-level notes for the documentation (derived fields, entities, event time).
    """

    source: str
    record: str
    title: str
    level: Level
    rows: tuple[Row, ...]
    reference: str
    layer: Layer = Layer.EVENT
    native_prefix: str | None = None
    native_statuses: frozenset[ObservationStatus] | None = None
    ocsf_class: int = 0
    notes: tuple[str, ...] = field(default=())

    def __post_init__(self) -> None:
        seen: set[str] = set()
        for r in self.rows:
            if r.source in seen:
                raise InvariantViolation(f"{self.record_type}: source field {r.source!r} listed twice")
            seen.add(r.source)

    @property
    def record_type(self) -> str:
        return f"{self.source}.{self.record}"

    @property
    def prefix(self) -> str:
        return self.native_prefix if self.native_prefix is not None else self.record_type

    def native_id(self, row: Row) -> str:
        """Catalogue ID of a row's native field."""
        return row.native if row.native is not None else f"{self.prefix}.{row.source}"

    def row(self, source: str) -> Row:
        """The row of a source field (KeyError when the table does not list it)."""
        for r in self.rows:
            if r.source == source:
                return r
        raise KeyError(f"{self.record_type} has no source field {source!r}")

    def sources(self) -> tuple[str, ...]:
        return tuple(r.source for r in self.rows)

    def targets(self) -> tuple[str, ...]:
        """Every catalogue field this record type can carry (shared targets, verbatim copies, natives)."""
        out: list[str] = []
        for r in self.rows:
            for t in self.row_targets(r):
                if t not in out:
                    out.append(t)
        return tuple(out)

    def row_targets(self, r: Row) -> tuple[str, ...]:
        """Catalogue fields a row writes, in order (target first, then the verbatim copy)."""
        out: list[str] = []
        if r.target is None:
            out.append(self.native_id(r))
        elif r.target not in SPECIAL_TARGETS:
            out.append(r.target)
        if r.also == "native":
            out.append(self.native_id(r))
        elif r.also is not None:
            out.append(r.also)
        return tuple(out)

    def native_specs(self) -> tuple[FieldSpec, ...]:
        """FieldSpecs of this record type's native fields."""
        out: list[FieldSpec] = []
        for r in self.rows:
            if not r.is_native:
                continue
            # A verbatim copy keeps the source text, so its dtype is the source's.
            kind = r.kind if r.target is None else (Kind.FINGERPRINT if r.dtype is Dtype.STR else Kind.ATTRIBUTE)
            statuses = self.native_statuses or (IDENTITY_STATUSES if kind is Kind.IDENTIFIER else MEASUREMENT_STATUSES)
            dtype = r.dtype if kind is Kind.ATTRIBUTE else (Dtype.STR if r.dtype not in (Dtype.STR, Dtype.ADDRESS, Dtype.MAC) else r.dtype)
            if kind in (Kind.IDENTIFIER, Kind.FINGERPRINT) and r.dtype not in (Dtype.STR, Dtype.ADDRESS, Dtype.MAC):
                # Identifiers and fingerprints are text; a structured value stays an attribute.
                kind, dtype = Kind.ATTRIBUTE, r.dtype
            desc = f"{self.title} field '{r.source}'." + (f" {r.note}" if r.note else "")
            out.append(FieldSpec(self.native_id(r), self.level, kind, r.unit, desc,
                                 f"native field of {self.title} ({self.reference})", False, self.layer, dtype,
                                 statuses, None, "0.2"))
        return tuple(out)


_REGISTRY: dict[str, RecordMap] = {}


def register(maps: Iterable[RecordMap]) -> None:
    """Add record maps to the registry (each record type once)."""
    for m in maps:
        if m.record_type in _REGISTRY and _REGISTRY[m.record_type] is not m:
            raise InvariantViolation(f"Record type {m.record_type} registered twice")
        _REGISTRY[m.record_type] = m


def _load() -> None:
    """Import the mapping tables once (they register themselves)."""
    if _REGISTRY:
        return
    from nagahana.datamodel.maps import ALL_MAPS

    register(ALL_MAPS)


def record_maps() -> tuple[RecordMap, ...]:
    """Every registered record map, in registration order."""
    _load()
    return tuple(_REGISTRY.values())


def record_map(record_type: str) -> RecordMap:
    """The record map of a record type ("zeek.conn")."""
    _load()
    try:
        return _REGISTRY[record_type]
    except KeyError:
        raise KeyError(f"No mapping table for record type {record_type!r}") from None


def native_specs() -> tuple[FieldSpec, ...]:
    """Native FieldSpecs of every record map; a native ID shared by several record types must agree."""
    out: dict[str, FieldSpec] = {}
    for m in record_maps():
        for s in m.native_specs():
            prev = out.get(s.id)
            if prev is None:
                out[s.id] = s
            elif (prev.dtype, prev.kind, prev.unit) != (s.dtype, s.kind, s.unit):
                raise InvariantViolation(f"Native field {s.id} is defined differently by two record types")
    return tuple(out.values())


__all__ = ["DTYPE_CODES", "KIND_CODES", "R", "SPECIAL_TARGETS", "TIME_TARGET", "RecordMap", "Row", "native_specs",
           "record_map", "record_maps", "register"]
