"""The mapping engine: one record's source values -> catalogue fields with statuses, through its table.

`Mapper(record_map).apply(values, draft, ctx)` walks the table's rows in order:

    raw value missing          the field is not written (absent means NOT_SUPPLIED, D-41)
    UNSET                      the source states "no value" (Zeek "-", Windows "-"): NOT_SUPPLIED,
                               written explicitly so the record shows the source said so
    EMPTY                      the source states "empty" (Zeek "(empty)"): OBSERVED with an empty
                               string or list where the field's type allows it, refused otherwise
    a value                    converted (ingest/convert.py) and checked against the target's dtype;
                               OBSERVED, or NOT_SUPPLIED with the refusal reason counted per field

Precedence: when several rows target one field, the first row (in table order) that yields a
contributing value wins; a later row may still fill a field the earlier ones left without value.
Bit rows ("bit:<mask>", "zbits") contribute to one bitmask field together: it is OBSERVED only when
every bit row of the field is present, and the bits are OR-ed.

`also`: the source value is also written verbatim to the row's native field ("native") or to a named
shared field, so a mapping that is not one-to-one (a code table, a first-of-list) loses nothing.

Source keys the table does not list are retained as attributes "<record type>.<key>" with status
OBSERVED (NOT_SUPPLIED for UNSET), so every field of every source is mapped or retained.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from nagahana.datamodel.fields import CATALOGUE
from nagahana.datamodel.native import SPECIAL_TARGETS, TIME_TARGET, RecordMap, Row
from nagahana.datamodel.records import FieldValue
from nagahana.datamodel.spec import Dtype, FieldSpec, check_value
from nagahana.datamodel.status import ObservationStatus
from nagahana.ingest import convert as CV
from nagahana.ingest.core import IngestStats, UpdateDraft, json_safe
from nagahana.ingest.timeparse import Instant

OBS = ObservationStatus.OBSERVED
NS_ = ObservationStatus.NOT_SUPPLIED


class _Marker:
    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:
        return self.name


#: The source states that the field has no value.
UNSET = _Marker("UNSET")
#: The source states that the field is empty (an empty string or set).
EMPTY = _Marker("EMPTY")
_MISSING = _Marker("MISSING")

_LIST_DTYPES = frozenset({Dtype.INT_LIST, Dtype.FLOAT_LIST, Dtype.STR_LIST, Dtype.ADDRESS_LIST, Dtype.TIME_LIST})
_TEXT_DTYPES = frozenset({Dtype.STR, Dtype.ADDRESS, Dtype.MAC})

#: Default converter of a native row without one, by dtype.
_DEFAULT_CONV: dict[Dtype, str] = {
    Dtype.INT: "int", Dtype.FLOAT: "double", Dtype.BOOL: "bool", Dtype.STR: "string", Dtype.TIME: "time",
    Dtype.ADDRESS: "addr", Dtype.MAC: "mac", Dtype.INT_LIST: "list_int", Dtype.FLOAT_LIST: "list_float",
    Dtype.STR_LIST: "list", Dtype.ADDRESS_LIST: "list_addr", Dtype.TIME_LIST: "list_float", Dtype.MAP: "map",
}


def coerce(spec: FieldSpec, value: Any) -> Any:
    """`value` in the dtype of `spec`; raises `Refusal` when it cannot be."""
    d = spec.dtype
    if isinstance(value, Instant):
        if d in (Dtype.TIME, Dtype.FLOAT):
            value = value.seconds
        else:
            raise CV.Refusal(f"a time cannot fill a {d.value if d else '?'} field")
    if d is Dtype.INT:
        if isinstance(value, bool):
            value = int(value)
        elif isinstance(value, float):
            if not math.isfinite(value) or value != int(value):
                raise CV.Refusal(f"not an integer: {value!r}")
            value = int(value)
    elif d in (Dtype.FLOAT, Dtype.TIME):
        if isinstance(value, int | float) and not isinstance(value, bool):
            value = float(value)
    elif d in _TEXT_DTYPES:
        if isinstance(value, int) and not isinstance(value, bool):
            value = str(value)
    elif d is Dtype.BYTES:
        if isinstance(value, str):
            value = value.encode("utf-8")
    elif d in _LIST_DTYPES:
        if isinstance(value, list):
            value = tuple(value)
        elif not isinstance(value, tuple):
            value = (value,)
        if d is Dtype.TIME_LIST:
            value = tuple(v.seconds if isinstance(v, Instant) else float(v) for v in value)
        elif d is Dtype.FLOAT_LIST:
            value = tuple(float(v) for v in value)
    elif d is Dtype.MAP:
        value = json_safe(value)
    assert spec is not None
    why = check_value(spec, value)
    if why is not None:
        raise CV.Refusal(why)
    return value


@dataclass
class _Step:
    """A prepared row: its converter, target spec and copy target."""

    row: Row
    target: str | None
    spec: FieldSpec | None
    conv: CV.Converter
    is_bit: bool
    also: str | None
    also_spec: FieldSpec | None
    also_conv: CV.Converter | None


class Mapper:
    """The prepared mapping of one record map (see the module docstring)."""

    def __init__(self, rmap: RecordMap) -> None:
        self.map = rmap
        self.steps: list[_Step] = []
        self.sources: frozenset[str] = frozenset(rmap.sources())
        bit_rows: dict[str, int] = {}
        for r in rmap.rows:
            target = rmap.native_id(r) if r.target is None else r.target
            spec = None if target in SPECIAL_TARGETS else CATALOGUE[target]
            if r.conv:
                conv = CV.get(r.conv)
            else:
                assert spec is not None and spec.dtype is not None
                conv = CV.get(_DEFAULT_CONV.get(spec.dtype, "string")) if spec.dtype is not Dtype.BYTES else _bytes_conv
            is_bit = r.conv.startswith("bit:") or r.conv == "zbits"
            if is_bit:
                bit_rows[target] = bit_rows.get(target, 0) + 1
            also = also_spec = None
            also_conv = None
            if r.also is not None:
                also = rmap.native_id(r) if r.also == "native" else r.also
                also_spec = CATALOGUE[also]
                assert also_spec.dtype is not None
                also_conv = _bytes_conv if also_spec.dtype is Dtype.BYTES else CV.get(_DEFAULT_CONV.get(also_spec.dtype, "string"))
            self.steps.append(_Step(r, target, spec, conv, is_bit, also, also_spec, also_conv))
        self.bit_rows = bit_rows
        self.extra_prefix = rmap.record_type

    def apply(self, values: Mapping[str, Any], draft: UpdateDraft, ctx: CV.ConvContext, stats: IngestStats, *,
              source: str, keep_unmapped: bool = True) -> None:
        """Map `values` (source field -> raw value or marker) into `draft` (module docstring)."""
        fields = draft.fields
        bits: dict[str, list[int | None]] = {}
        for st in self.steps:
            raw = values.get(st.row.source, _MISSING)
            if raw is _MISSING:
                if st.is_bit:
                    bits.setdefault(st.target or "", []).append(None)
                continue
            target = st.target
            assert target is not None
            if raw is UNSET:
                if st.is_bit:
                    bits.setdefault(target, []).append(None)
                elif target != TIME_TARGET and not _has_value(fields, target):
                    fields[target] = FieldValue(target, None, NS_, source)
                if st.also is not None and st.also not in fields:
                    fields[st.also] = FieldValue(st.also, None, NS_, source)
                continue
            if raw is EMPTY:
                if st.is_bit:
                    bits.setdefault(target, []).append(None)
                    continue
                self._empty(st.spec, target, fields, stats, source)
                if st.also is not None:
                    self._empty(st.also_spec, st.also, fields, stats, source)
                continue
            try:
                value = st.conv(raw, ctx)
            except CV.Refusal as exc:
                if st.is_bit:
                    bits.setdefault(target, []).append(None)
                    stats.refuse(target, str(exc))
                elif target == TIME_TARGET:
                    stats.refuse(TIME_TARGET, str(exc))
                else:
                    stats.refuse(target, str(exc))
                    if not _has_value(fields, target):
                        fields[target] = FieldValue(target, None, NS_, source)
                value = None
            if value is not None:
                if st.is_bit:
                    bits.setdefault(target, []).append(int(value))
                elif target == TIME_TARGET:
                    if draft.time is None:
                        draft.time = value if isinstance(value, Instant) else Instant(float(value), None, 1e-6)
                        draft.original_time = raw if isinstance(raw, str) else None
                elif not _has_value(fields, target):
                    assert st.spec is not None
                    try:
                        fields[target] = FieldValue(target, coerce(st.spec, value), OBS, source)
                    except CV.Refusal as exc:
                        stats.refuse(target, str(exc))
                        fields[target] = FieldValue(target, None, NS_, source)
            if st.also is not None and st.also_conv is not None and not _has_value(fields, st.also):
                assert st.also_spec is not None
                try:
                    fields[st.also] = FieldValue(st.also, coerce(st.also_spec, st.also_conv(raw, ctx)), OBS, source)
                except CV.Refusal as exc:
                    stats.refuse(st.also, str(exc))
                    fields[st.also] = FieldValue(st.also, None, NS_, source)
        for target, parts in bits.items():
            if _has_value(fields, target):
                continue
            if len(parts) == self.bit_rows.get(target, 0) and all(p is not None for p in parts):
                v = 0
                for p in parts:
                    v |= int(p or 0)
                fields[target] = FieldValue(target, v, OBS, source)
            elif any(p is not None for p in parts):
                stats.refuse(target, "incomplete bit set")
                fields[target] = FieldValue(target, None, NS_, source)
        if keep_unmapped:
            for key, raw in values.items():
                if key in self.sources or key.startswith("__"):
                    continue
                akey = f"{self.extra_prefix}.{key}"
                if akey in CATALOGUE:
                    akey = f"x.{akey}"
                stats.retain(akey)
                if raw is UNSET:
                    draft.attributes[akey] = FieldValue(akey, None, NS_, source)
                elif raw is EMPTY:
                    draft.attributes[akey] = FieldValue(akey, "", OBS, source)
                elif raw is None:
                    draft.attributes[akey] = FieldValue(akey, None, NS_, source)
                else:
                    draft.attributes[akey] = FieldValue(akey, json_safe(raw), OBS, source)

    @staticmethod
    def _empty(spec: FieldSpec | None, target: str, fields: dict[str, FieldValue], stats: IngestStats,
               source: str) -> None:
        if spec is None or _has_value(fields, target):
            return
        if spec.dtype in _LIST_DTYPES:
            fields[target] = FieldValue(target, (), OBS, source)
        elif spec.dtype in (Dtype.STR,):
            fields[target] = FieldValue(target, "", OBS, source)
        elif spec.dtype is Dtype.MAP:
            fields[target] = FieldValue(target, {}, OBS, source)
        else:
            stats.refuse(target, "empty value for a non-text field")
            fields[target] = FieldValue(target, None, NS_, source)


def _bytes_conv(raw: Any, ctx: CV.ConvContext) -> bytes:
    if isinstance(raw, bytes | bytearray):
        return bytes(raw)
    if isinstance(raw, str):
        return CV.get("b64")(raw, ctx)
    raise CV.Refusal(f"expected bytes, got {type(raw).__name__}")


def _has_value(fields: Mapping[str, FieldValue], target: str) -> bool:
    fv = fields.get(target)
    return fv is not None and fv.contributes


_MAPPERS: dict[str, Mapper] = {}


def mapper(record_type: str) -> Mapper:
    """The cached `Mapper` of a record type."""
    m = _MAPPERS.get(record_type)
    if m is None:
        from nagahana.datamodel.native import record_map

        m = _MAPPERS[record_type] = Mapper(record_map(record_type))
    return m


def set_field(draft: UpdateDraft, field_id: str, value: Any, source: str, stats: IngestStats | None = None, *,
              status: ObservationStatus = OBS, reliability: float | None = None, overwrite: bool = False) -> bool:
    """Write a derived field (checked and coerced); a refusal leaves it NOT_SUPPLIED and is counted.

    Returns True when the value was written. Without `overwrite`, an existing contributing value is
    kept (the source's own value outranks a derived one).
    """
    if not overwrite and _has_value(draft.fields, field_id):
        return False
    spec = CATALOGUE[field_id]
    try:
        v = coerce(spec, value)
    except CV.Refusal as exc:
        if stats is not None:
            stats.refuse(field_id, str(exc))
        draft.fields.setdefault(field_id, FieldValue(field_id, None, NS_, source))
        return False
    if not spec.admits(status):
        status = OBS if spec.admits(OBS) else status
    draft.fields[field_id] = FieldValue(field_id, v, status, source, reliability)
    return True


def mark(draft: UpdateDraft, field_id: str, status: ObservationStatus, source: str) -> None:
    """Write an excluded status (NOT_SUPPLIED or NOT_OBSERVABLE) unless the field has a value."""
    if not _has_value(draft.fields, field_id):
        draft.fields[field_id] = FieldValue(field_id, None, status, source)


Deriver = Callable[[UpdateDraft, IngestStats, str], None]

__all__ = ["EMPTY", "UNSET", "Mapper", "coerce", "mapper", "mark", "set_field"]
