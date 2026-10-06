"""Columnar form of state updates: the model-ready, human-auditable tables.

`records.StateUpdate` is the contract: one immutable object per telemetry record, the right shape for
checking invariants and auditing one record, the wrong shape for feeding a model millions of records.
This module holds the same information as pandas and NumPy structures:

    updates     one row per state update: ordering, the entities it touches (by role), provenance
    entities    one row per typed entity (host, external, service, multicast, account ...)
    relations   one row per distinct set of entities touched together (a hyperedge of the graph)
    values      float64 matrix [updates x columns]; NaN where a field carries no value
    status      uint8 matrix of the same shape; the observation status of every cell
    quality     optional float64 matrix of the same shape: the reliability of LOW_RELIABILITY cells and
                the age (seconds) of STALE cells, NaN elsewhere (absent when no cell needs it)
    rest        optional per-row dicts of everything that is neither a matrix column nor a side field:
                catalogued attributes, identifiers, fingerprints and retained uncatalogued attributes
    aliases, roles, names
                optional tables of time-stamped facts about entities (`FACT_TABLES`): every row carries
                `since`, the event time from which the fact was known (the PCAP adapter infers them;
                D-47, D-48)

Absence is never zero (D-41)
----------------------------
A cell whose status is NOT_SUPPLIED or NOT_OBSERVABLE holds NaN in `values`. NaN is a guard, not a
value: a consumer selects cells through `status` (`contributing_cells`) and never imputes the NaN.
`validate()` checks the pairing in both directions and that every status is admissible for its field.

Entity placement
----------------
Column `entity_0` holds the initiator (or, for a record without parties, its subject), `entity_1` the
responder, and `entity_2`, `entity_3`, ... the remaining entities in the order of the update. -1 means
none. When some update's roles differ from the positional defaults (initiator, responder, service),
columns `role_k` hold the role codes (`records.ROLE_CODE`, -1 none). The graph builder reads
`entity_0` to `entity_2` (initiator, responder and the third member: a service, or an account for an
authentication record).

What goes where
---------------
- Matrix kinds (`spec.MATRIX_KINDS`) are matrix columns; a histogram takes one column per bin. The
  `kind` of each column says how a model treats it (a port is a category, not a magnitude).
- `side_fields` names non-matrix fields stored as columns of `updates` (addresses, payload digest,
  SNI); open-vocabulary fields (`Kind.FINGERPRINT`) are stored as an index into `vocab[field_id]`.
- Everything else goes to `rest`.

Round trip
----------
`ColumnarUpdates.update(i)` rebuilds the `StateUpdate` of row i and `to_columnar` builds the tables from
any iterable of state updates, so the object form and the columnar form can be checked against each
other (`records.differences` is empty).
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from nagahana.core.errors import InvariantViolation
from nagahana.datamodel.fields import CATALOGUE, Kind
from nagahana.datamodel.records import (
    ENTITY_ROLES,
    ROLE_CODE,
    EntityRef,
    FieldValue,
    OrderingInfo,
    Provenance,
    StateUpdate,
    placement,
)
from nagahana.datamodel.spec import MATRIX_KINDS
from nagahana.datamodel.status import CONTRIBUTING, EXCLUDED, ObservationStatus

#: Status codes of the `status` matrix, in the order of `ObservationStatus`.
STATUS_ORDER: tuple[ObservationStatus, ...] = tuple(ObservationStatus)
STATUS_CODE: dict[ObservationStatus, int] = {s: i for i, s in enumerate(STATUS_ORDER)}
CODE_OBSERVED = STATUS_CODE[ObservationStatus.OBSERVED]
CODE_STALE = STATUS_CODE[ObservationStatus.STALE]
CODE_LOW_RELIABILITY = STATUS_CODE[ObservationStatus.LOW_RELIABILITY]
CODE_NOT_SUPPLIED = STATUS_CODE[ObservationStatus.NOT_SUPPLIED]
CODE_NOT_OBSERVABLE = STATUS_CODE[ObservationStatus.NOT_OBSERVABLE]
_CONTRIBUTING_CODES = np.array([STATUS_CODE[s] for s in STATUS_ORDER if s in CONTRIBUTING], dtype=np.uint8)
_EXCLUDED_CODES = np.array([STATUS_CODE[s] for s in STATUS_ORDER if s in EXCLUDED], dtype=np.uint8)

#: Default role of an entity column when no role column is written.
_SLOT_DEFAULT_ROLE = ("initiator", "responder")

#: The optional fact tables: table -> ((column, dtype), ...). `entity` indexes `entities`; `since` is
#: the event time from which the fact was known. A table holds facts from the whole source, so a
#: consumer showing the state at time t keeps only rows with `since <= t`.
#:   aliases  another address of a host entity (an IPv6 link-local address) and the link-layer
#:            (MAC) address that ties the two
#:   roles    a role inferred from traffic; `evidence` is a fixed description of the rule, never a count
#:   names    a name seen for an address (`source` "dns" or "tls"); `entity` is -1 if the address
#:            never became an entity
FACT_TABLES: dict[str, tuple[tuple[str, Any], ...]] = {
    "aliases": (("entity", np.int32), ("address", str), ("mac", str), ("since", np.float64)),
    "roles": (("entity", np.int32), ("role", str), ("since", np.float64), ("evidence", str)),
    "names": (("entity", np.int32), ("address", str), ("name", str), ("source", str), ("since", np.float64)),
}

#: Optional provenance columns of `updates` (written when some update carries them).
PROVENANCE_COLUMNS: tuple[str, ...] = (
    "source_type", "record_type", "location", "line", "record_index", "sub_index", "sensor_id", "exporter",
    "original_time",
)
_INT_PROVENANCE = frozenset({"line", "record_index", "sub_index"})


def fact_table(name: str, rows: Iterable[Sequence[Any]] = ()) -> pd.DataFrame:
    """The fact table `name` (see `FACT_TABLES`) with its columns and dtypes, from rows in column order."""
    spec = FACT_TABLES[name]
    columns = list(zip(*rows, strict=True)) or [()] * len(spec)
    return pd.DataFrame({c: pd.Series(list(v), dtype=t) for (c, t), v in zip(spec, columns, strict=True)})


@dataclass(frozen=True)
class Column:
    """One column of the value/status matrices.

    Attributes
    ----------
    name: unique column name; the field ID, or "<field_id>[<component>]" for histogram bins.
    field_id: catalogue ID.
    component: bin index for a histogram field, else None.
    kind, unit: copied from the catalogue so the matrix is self-describing.
    label: what the component means (the byte range of a histogram bin), or "".
    """

    name: str
    field_id: str
    component: int | None
    kind: Kind
    unit: str | None
    label: str = ""


def columns_for(field_ids: Sequence[str], *, histogram_bins: Mapping[str, Sequence[str]] | None = None) -> tuple[Column, ...]:
    """Matrix columns for the given fields, in order. Histogram fields need their bin labels."""
    bins = histogram_bins or {}
    out: list[Column] = []
    for fid in field_ids:
        spec = CATALOGUE[fid]
        if spec.kind not in MATRIX_KINDS:
            raise InvariantViolation(f"{fid} is {spec.kind.value}; it cannot be a matrix column.")
        if spec.kind is Kind.HISTOGRAM:
            if fid not in bins:
                raise InvariantViolation(f"{fid} is a histogram; its bin labels must be given.")
            out.extend(Column(f"{fid}[{i}]", fid, i, spec.kind, spec.unit, lab) for i, lab in enumerate(bins[fid]))
        else:
            out.append(Column(fid, fid, None, spec.kind, spec.unit))
    return tuple(out)


def _admissible_table(columns: Sequence[Column]) -> np.ndarray:
    """[C, n_statuses] bool: True where the status is admissible for the column's field."""
    out = np.zeros((len(columns), len(STATUS_ORDER)), dtype=bool)
    for j, c in enumerate(columns):
        spec = CATALOGUE.get(c.field_id)
        if spec is None:
            out[j, :] = True
            continue
        for s in STATUS_ORDER:
            out[j, STATUS_CODE[s]] = spec.admits(s)
    return out


@dataclass
class ColumnarUpdates:
    """State updates as tables and matrices. See the module docstring.

    `updates` columns
        seq                   position in the stream (row number)
        record                record number in the source (sources skip records they cannot decode)
        event_time, ingest_time, watermark, reorder_uncertainty_s      `records.OrderingInfo`
        event_time_ns         int64 nanoseconds, -1 when the source has no such precision (optional)
        entity_0 ... entity_k indices into `entities` (-1 none); see "Entity placement"
        role_0 ... role_k     role codes (optional; see "Entity placement")
        relation              index into `relations`
        direction             0 initiator to responder, 1 responder to initiator, -1 not applicable
        raw_offset, raw_len   where the raw record sits in the source file (-1 when unknown)
    plus the optional `PROVENANCE_COLUMNS`, side-field columns and adapter audit columns.

    `entities` columns: kind, key, first_seen, last_seen, updates, plus adapter-specific columns.
    `raw_hash` holds the SHA-256 digest of each raw record (L0), one row of 32 bytes per update.
    """

    source_id: str
    adapter: str
    adapter_version: str
    columns: tuple[Column, ...]
    updates: pd.DataFrame
    entities: pd.DataFrame
    relations: pd.DataFrame
    values: np.ndarray
    status: np.ndarray
    raw_hash: np.ndarray
    clock_quality: str | None = None
    vocab: dict[str, list[str]] = field(default_factory=dict)
    #: Non-matrix fields: field ID -> (value column of `updates`, status column of `updates`). A value
    #: column written "@entity_k" means "the key of that entity" (addresses are entity keys); an
    #: integer value column that is not a vocabulary index is a digest, shown as 16 hex digits.
    side_fields: dict[str, tuple[str | None, str]] = field(default_factory=dict)
    #: Fields the adapter declares as explicitly present (others are NOT_SUPPLIED by absence).
    explicit_fields: tuple[str, ...] = ()
    aliases: pd.DataFrame = field(default_factory=lambda: fact_table("aliases"))
    roles: pd.DataFrame = field(default_factory=lambda: fact_table("roles"))
    names: pd.DataFrame = field(default_factory=lambda: fact_table("names"))
    #: Reliability (LOW_RELIABILITY) or age (STALE) per cell, NaN elsewhere; None when no cell needs it.
    quality: np.ndarray | None = None
    #: Per-row dicts: key -> (value, status code, reliability, age); None when no row has any.
    rest: list[dict[str, tuple[Any, int, float | None, float | None]] | None] | None = None
    #: Clock quality per row when it differs between rows (else `clock_quality` holds the common one).
    row_clock_quality: list[str | None] | None = None

    def __len__(self) -> int:
        return int(self.values.shape[0])

    def validate(self) -> None:
        """Check the invariants that make the matrices safe to feed to a model."""
        n = len(self.updates)
        if self.values.shape != (n, len(self.columns)) or self.status.shape != self.values.shape:
            raise InvariantViolation("values/status shape does not match updates x columns.")
        if self.values.dtype != np.float64 or self.status.dtype != np.uint8:
            raise InvariantViolation("values must be float64 and status uint8.")
        if self.raw_hash.shape != (n, 32):
            raise InvariantViolation("raw_hash must hold one 32-byte digest per update.")
        if int(self.status.max(initial=0)) >= len(STATUS_ORDER):
            raise InvariantViolation("Unknown status code in the status matrix.")
        excluded = np.isin(self.status, _EXCLUDED_CODES)
        nan = np.isnan(self.values)
        if (excluded & ~nan).any():
            raise InvariantViolation("An excluded cell carries a value (absence is not zero).")
        if (~excluded & nan).any():
            raise InvariantViolation("A contributing cell carries no value.")
        if n and len(self.columns):
            ok = _admissible_table(self.columns)[np.arange(len(self.columns))[None, :], self.status]   # [n, C]
            if not ok.all():
                i, j = (int(x[0]) for x in np.nonzero(~ok))
                raise InvariantViolation(
                    f"Row {i}: status {STATUS_ORDER[int(self.status[i, j])].value!r} is not admissible for "
                    f"{self.columns[j].field_id}."
                )
        if self.quality is not None:
            if self.quality.shape != self.values.shape:
                raise InvariantViolation("quality must have the shape of values.")
            needs = (self.status == CODE_LOW_RELIABILITY) | (self.status == CODE_STALE)
            q = self.quality
            if (needs & np.isnan(q)).any() or (~needs & ~np.isnan(q)).any():
                raise InvariantViolation("quality must be set exactly on LOW_RELIABILITY and STALE cells.")
        elif ((self.status == CODE_LOW_RELIABILITY) | (self.status == CODE_STALE)).any():
            raise InvariantViolation("LOW_RELIABILITY and STALE cells need a quality matrix.")
        if self.rest is not None and len(self.rest) != n:
            raise InvariantViolation("rest must hold one entry per update.")

    def contributing_cells(self) -> np.ndarray:
        """Boolean mask of the cells that count as evidence (the set O_t of datamodel/status.py)."""
        return np.isin(self.status, _CONTRIBUTING_CODES)

    def memory_bytes(self) -> dict[str, int]:
        """Memory held by each structure, in bytes."""
        return {
            "values": int(self.values.nbytes),
            "status": int(self.status.nbytes),
            "quality": int(self.quality.nbytes) if self.quality is not None else 0,
            "raw_hash": int(self.raw_hash.nbytes),
            "updates": int(self.updates.memory_usage(deep=True).sum()),
            "entities": int(self.entities.memory_usage(deep=True).sum()),
            "relations": int(self.relations.memory_usage(deep=True).sum()),
            "aliases": int(self.aliases.memory_usage(deep=True).sum()),
            "roles": int(self.roles.memory_usage(deep=True).sum()),
            "names": int(self.names.memory_usage(deep=True).sum()),
        }

    def entity_ref(self, index: int) -> EntityRef:
        row = self.entities.iloc[index]
        return EntityRef(str(row["kind"]), str(row["key"]))

    def update(self, i: int) -> StateUpdate:
        """Rebuild the `StateUpdate` of row i (audit of a single record)."""
        if not 0 <= i < len(self):
            raise IndexError(i)
        frame = self.updates

        def cell(column: str) -> Any:
            # column-wise access keeps each column's dtype (a row Series would turn integers into floats)
            return frame[column].iloc[i]

        ent_cols = sorted((c for c in frame.columns if c.startswith("entity_")), key=lambda c: int(c.split("_")[1]))
        entities: list[EntityRef] = []
        roles: list[str] = []
        for c in ent_cols:
            k = int(c.split("_")[1])
            idx = int(cell(c))
            if idx < 0:
                continue
            entities.append(self.entity_ref(idx))
            rc = f"role_{k}"
            if rc in frame.columns and int(cell(rc)) >= 0:
                roles.append(ENTITY_ROLES[int(cell(rc))])
            else:
                roles.append(_SLOT_DEFAULT_ROLE[k] if k < 2 else "service")
        fields: dict[str, FieldValue] = {}
        attributes: dict[str, FieldValue] = {}
        row_v, row_s = self.values[i], self.status[i]
        row_q = self.quality[i] if self.quality is not None else None
        explicit = set(self.explicit_fields)

        done: set[str] = set()
        for j, col in enumerate(self.columns):
            if col.field_id in done:
                continue
            done.add(col.field_id)
            st = STATUS_ORDER[int(row_s[j])]
            if st in EXCLUDED:
                if col.field_id in explicit or st is ObservationStatus.NOT_OBSERVABLE:
                    fields[col.field_id] = FieldValue(col.field_id, None, st, self.adapter)
                continue
            value: Any
            if col.kind is Kind.HISTOGRAM:
                idx_h = [k for k, c in enumerate(self.columns) if c.field_id == col.field_id]
                value = tuple(int(row_v[k]) for k in idx_h)
            elif col.kind in (Kind.COUNT, Kind.CATEGORICAL, Kind.BITMASK):
                value = int(row_v[j])
            else:
                value = float(row_v[j])
            rel = age = None
            if row_q is not None and st is ObservationStatus.LOW_RELIABILITY:
                rel = float(row_q[j])
            if row_q is not None and st is ObservationStatus.STALE:
                age = float(row_q[j])
            fields[col.field_id] = FieldValue(col.field_id, value, st, self.adapter, rel, age)

        for fid, (value_col, status_col) in self.side_fields.items():
            st = STATUS_ORDER[int(cell(status_col))]
            if st in EXCLUDED:
                if fid in explicit or st is ObservationStatus.NOT_OBSERVABLE:
                    fields[fid] = FieldValue(fid, None, st, self.adapter)
                continue
            if value_col is None:
                raise InvariantViolation(f"{fid}: a contributing side field needs a value column.")
            if value_col.startswith("@"):
                text = str(self.entities["key"].iloc[int(cell(value_col[1:]))])
            elif fid in self.vocab:
                text = self.vocab[fid][int(cell(value_col))]
            elif isinstance(cell(value_col), int | np.integer):
                text = f"{int(cell(value_col)):016x}"
            else:
                text = str(cell(value_col))
            fields[fid] = FieldValue(fid, text, st, self.adapter)

        if self.rest is not None and self.rest[i]:
            for key, (value, code, rel, age) in self.rest[i].items():
                fv = FieldValue(key, value, STATUS_ORDER[code], self.adapter, rel, age)
                (fields if key in CATALOGUE else attributes)[key] = fv

        wm, ru = float(cell("watermark")), float(cell("reorder_uncertainty_s"))
        ns = int(cell("event_time_ns")) if "event_time_ns" in frame.columns else -1
        prov: dict[str, Any] = {}
        for name in PROVENANCE_COLUMNS:
            if name in frame.columns:
                v = cell(name)
                if name in _INT_PROVENANCE:
                    prov[name] = None if int(v) < 0 else int(v)
                else:
                    prov[name] = None if v is None or (isinstance(v, float) and math.isnan(v)) else str(v)
        off, ln = int(cell("raw_offset")), int(cell("raw_len"))
        clock = self.row_clock_quality[i] if self.row_clock_quality is not None else self.clock_quality
        return StateUpdate(
            update_id=f"{self.source_id}:{int(cell('seq'))}",
            ordering=OrderingInfo(
                event_time=float(cell("event_time")),
                ingest_time=float(cell("ingest_time")),
                watermark=None if math.isnan(wm) else wm,
                reorder_uncertainty_s=None if math.isnan(ru) else ru,
                clock_quality=clock,
                event_time_ns=None if ns < 0 else ns,
            ),
            entities=tuple(entities),
            fields=fields,
            provenance=Provenance(
                self.source_id, self.adapter, self.adapter_version, bytes(self.raw_hash[i]).hex(),
                offset=None if off < 0 else off, length=None if ln < 0 else ln, **prov,
            ),
            roles=tuple(roles),
            attributes=attributes,
        )


class ColumnarBuilder:
    """Accumulates state updates into the columnar form, one update at a time.

    Rows are written into preallocated arrays that double when full (amortised O(1) per update), so the
    builder serves both the reference path (`to_columnar`) and adapters that stream millions of
    records. Statuses and values follow the update exactly; nothing is imputed.

    Parameters
    ----------
    columns: matrix columns (`columns_for`, or a canonical layout).
    side_fields: non-matrix fields stored as `updates` columns (see `ColumnarUpdates.side_fields`).
    keep_rest: keep every other field and attribute in `rest` (False drops nothing silently: it counts
        the values left out in `rest_dropped`, for consumers that only need the matrices).
    """

    def __init__(self, columns: Sequence[Column], *, side_fields: Mapping[str, tuple[str | None, str]] | None = None,
                 keep_rest: bool = True) -> None:
        self.columns = tuple(columns)
        self.side = dict(side_fields or {})
        self.keep_rest = keep_rest
        self.rest_dropped = 0
        self._col_index: dict[str, list[int]] = {}
        for j, c in enumerate(self.columns):
            self._col_index.setdefault(c.field_id, []).append(j)
        self._hist = {fid for fid, js in self._col_index.items() if CATALOGUE[fid].kind is Kind.HISTOGRAM}
        cap = 1024
        c_n = len(self.columns)
        self._values = np.empty((cap, c_n), dtype=np.float64)
        self._status = np.empty((cap, c_n), dtype=np.uint8)
        self._quality: np.ndarray | None = None
        self._n = 0
        self._hashes = bytearray()
        self._meta: dict[str, list[Any]] = {k: [] for k in (
            "event_time", "ingest_time", "watermark", "reorder_uncertainty_s", "event_time_ns", "raw_offset", "raw_len",
            "relation")}
        self._prov: dict[str, list[Any]] = {k: [] for k in PROVENANCE_COLUMNS}
        self._prov_seen: set[str] = set()
        self._ents: list[list[int]] = []
        self._roles: list[list[int]] = []
        self._nondefault_roles = False
        self._side_rows: dict[str, list[Any]] = {}
        self._status_cols: set[str] = set()
        for _fid, (value_col, status_col) in self.side.items():
            self._side_rows[status_col] = []
            self._status_cols.add(status_col)
            if value_col is not None and not value_col.startswith("@"):
                self._side_rows[value_col] = []
        self._rest: list[dict[str, tuple[Any, int, float | None, float | None]] | None] = []
        self._any_rest = False
        self._clock: list[str | None] = []
        self._entity_index: dict[tuple[str, str], int] = {}
        self._relation_index: dict[tuple[int, ...], int] = {}
        self._vocab: dict[str, list[str]] = {}
        self._vocab_index: dict[str, dict[str, int]] = {}
        self._explicit: set[str] | None = None
        self._first: StateUpdate | None = None
        self._has_ns = False

    def __len__(self) -> int:
        return self._n

    def _grow(self) -> None:
        cap = self._values.shape[0] * 2
        self._values = np.resize(self._values, (cap, self._values.shape[1]))
        self._status = np.resize(self._status, (cap, self._status.shape[1]))
        if self._quality is not None:
            self._quality = np.resize(self._quality, (cap, self._quality.shape[1]))

    def _place(self, u: StateUpdate) -> tuple[list[int], list[int]]:
        """Entity indices and role codes per slot (module docstring, "Entity placement")."""
        slots: list[int] = []
        roles: list[int] = []
        for k, item in enumerate(placement(u)):
            if item is None:
                slots.append(-1)
                roles.append(-1)
                continue
            e, r = item
            slots.append(self._entity(e))
            roles.append(ROLE_CODE[r])
            if r != (_SLOT_DEFAULT_ROLE[k] if k < 2 else "service"):
                self._nondefault_roles = True
        return slots, roles

    def _entity(self, e: EntityRef) -> int:
        key = (e.kind, e.id)
        idx = self._entity_index.get(key)
        if idx is None:
            idx = self._entity_index[key] = len(self._entity_index)
        return idx

    def add(self, u: StateUpdate) -> int:
        """Append one state update; returns its row."""
        if self._n == self._values.shape[0]:
            self._grow()
        i = self._n
        self._first = self._first or u
        v = self._values[i]
        s = self._status[i]
        v.fill(np.nan)
        s.fill(CODE_NOT_SUPPLIED)
        if self._quality is not None:
            self._quality[i].fill(np.nan)
        self._explicit = set(u.fields) if self._explicit is None else self._explicit & set(u.fields)
        side_row: dict[str, Any] = {}
        rest: dict[str, tuple[Any, int, float | None, float | None]] = {}
        for fid, fv in u.fields.items():
            idx = self._col_index.get(fid)
            code = STATUS_CODE[fv.status]
            if idx is not None:
                s[idx] = code
                if fv.status in CONTRIBUTING:
                    if fid in self._hist:
                        v[idx] = [float(x) for x in fv.value]
                    else:
                        v[idx[0]] = float(fv.value)
                    if fv.status is not ObservationStatus.OBSERVED:
                        if self._quality is None:
                            self._quality = np.full(self._values.shape, np.nan, dtype=np.float64)
                        self._quality[i, idx] = fv.reliability if fv.status is ObservationStatus.LOW_RELIABILITY else fv.age_s
            elif fid in self.side:
                value_col, status_col = self.side[fid]
                side_row[status_col] = code
                if value_col is not None and not value_col.startswith("@") and fv.status in CONTRIBUTING:
                    if CATALOGUE[fid].kind is Kind.FINGERPRINT:
                        table = self._vocab_index.setdefault(fid, {})
                        if fv.value not in table:
                            table[fv.value] = len(table)
                            self._vocab.setdefault(fid, []).append(str(fv.value))
                        side_row[value_col] = table[fv.value]
                    else:
                        side_row[value_col] = fv.value
            else:
                rest[fid] = (fv.value, code, fv.reliability, fv.age_s)
        for key, fv in u.attributes.items():
            rest[key] = (fv.value, STATUS_CODE[fv.status], fv.reliability, fv.age_s)
        if rest and not self.keep_rest:
            self.rest_dropped += len(rest)
            rest = {}
        self._rest.append(rest or None)
        self._any_rest = self._any_rest or bool(rest)
        for col, values in self._side_rows.items():
            values.append(side_row.get(col, CODE_NOT_SUPPLIED if col in self._status_cols else None))
        slots, roles = self._place(u)
        self._ents.append(slots)
        self._roles.append(roles)
        rel = self._relation_index.setdefault(tuple(x for x in slots if x >= 0), len(self._relation_index))
        o = u.ordering
        m = self._meta
        m["event_time"].append(o.event_time)
        m["ingest_time"].append(o.ingest_time)
        m["watermark"].append(np.nan if o.watermark is None else o.watermark)
        m["reorder_uncertainty_s"].append(np.nan if o.reorder_uncertainty_s is None else o.reorder_uncertainty_s)
        m["event_time_ns"].append(-1 if o.event_time_ns is None else int(o.event_time_ns))
        self._has_ns = self._has_ns or o.event_time_ns is not None
        p = u.provenance
        m["raw_offset"].append(-1 if p.offset is None else int(p.offset))
        m["raw_len"].append(-1 if p.length is None else int(p.length))
        m["relation"].append(rel)
        for name in PROVENANCE_COLUMNS:
            val = getattr(p, name)
            if val is not None:
                self._prov_seen.add(name)
            self._prov[name].append(val)
        self._clock.append(o.clock_quality)
        self._hashes += bytes.fromhex(p.raw_hash) if p.raw_hash else bytes(32)
        self._n += 1
        return i

    def build(self, *, source_id: str | None = None, adapter: str | None = None,
              adapter_version: str | None = None) -> ColumnarUpdates:
        """The columnar form of every update added so far."""
        if self._first is None:
            raise InvariantViolation("No state updates to convert.")
        n = self._n
        width = max(len(x) for x in self._ents)
        frame: dict[str, Any] = {"seq": np.arange(n, dtype=np.int64), "record": np.arange(n, dtype=np.int64)}
        m = self._meta
        frame["event_time"] = np.asarray(m["event_time"], dtype=np.float64)
        frame["ingest_time"] = np.asarray(m["ingest_time"], dtype=np.float64)
        frame["watermark"] = np.asarray(m["watermark"], dtype=np.float64)
        frame["reorder_uncertainty_s"] = np.asarray(m["reorder_uncertainty_s"], dtype=np.float64)
        if self._has_ns:
            frame["event_time_ns"] = np.asarray(m["event_time_ns"], dtype=np.int64)
        ents = np.full((n, width), -1, dtype=np.int32)
        roles = np.full((n, width), -1, dtype=np.int8)
        for r, (e_row, r_row) in enumerate(zip(self._ents, self._roles, strict=True)):
            ents[r, :len(e_row)] = e_row
            roles[r, :len(r_row)] = r_row
        for k in range(width):
            frame[f"entity_{k}"] = ents[:, k]
        if self._nondefault_roles:
            for k in range(width):
                frame[f"role_{k}"] = roles[:, k]
        frame["relation"] = np.asarray(m["relation"], dtype=np.int32)
        frame["direction"] = np.full(n, -1, dtype=np.int8)
        frame["raw_offset"] = np.asarray(m["raw_offset"], dtype=np.int64)
        frame["raw_len"] = np.asarray(m["raw_len"], dtype=np.int64)
        for name in PROVENANCE_COLUMNS:
            if name not in self._prov_seen:
                continue
            col = self._prov[name]
            if name in _INT_PROVENANCE:
                frame[name] = np.asarray([-1 if x is None else int(x) for x in col], dtype=np.int64)
            else:
                frame[name] = pd.Series(col, dtype=object)
        for col, values in self._side_rows.items():
            if col in self._status_cols:
                frame[col] = np.asarray(values, dtype=np.uint8)
            else:
                frame[col] = pd.Series(values, dtype=object)
        updates = pd.DataFrame(frame)
        entities = pd.DataFrame({"kind": [k for k, _ in self._entity_index], "key": [i for _, i in self._entity_index]})
        clocks = set(self._clock)
        first = self._first
        quality = self._quality[:n].copy() if self._quality is not None else None
        out = ColumnarUpdates(
            source_id=source_id or first.provenance.source_id, adapter=adapter or first.provenance.adapter,
            adapter_version=adapter_version or first.provenance.adapter_version, columns=self.columns,
            updates=updates, entities=entities, relations=pd.DataFrame(),
            values=np.ascontiguousarray(self._values[:n]), status=np.ascontiguousarray(self._status[:n]),
            raw_hash=np.frombuffer(bytes(self._hashes), dtype=np.uint8).reshape(n, 32).copy(),
            clock_quality=first.ordering.clock_quality if len(clocks) == 1 else None, vocab=dict(self._vocab),
            side_fields=dict(self.side), explicit_fields=tuple(sorted(self._explicit or ())), quality=quality,
            rest=list(self._rest) if self._any_rest else None,
            row_clock_quality=None if len(clocks) == 1 else list(self._clock),
        )
        with np.errstate(invalid="ignore"):
            finalise_tables(out, relation_entities={v: k for k, v in self._relation_index.items()})
        return out


def to_columnar(
    updates: Iterable[StateUpdate],
    columns: Sequence[Column],
    *,
    side_fields: Mapping[str, tuple[str | None, str]] | None = None,
) -> ColumnarUpdates:
    """Build the columnar form from state-update objects (any adapter), through `ColumnarBuilder`."""
    b = ColumnarBuilder(columns, side_fields=side_fields)
    for u in updates:
        b.add(u)
    return b.build()


def finalise_tables(cu: ColumnarUpdates, *, relation_entities: Mapping[int, Sequence[int]]) -> None:
    """Fill the per-entity and per-relation summary columns from the update table (vectorised)."""
    u = cu.updates
    n_ent = len(cu.entities)
    ent_cols = [c for c in u.columns if c.startswith("entity_")]
    t = u["event_time"].to_numpy()
    first = np.full(n_ent, np.inf)
    last = np.full(n_ent, -np.inf)
    count = np.zeros(n_ent, dtype=np.int64)
    for c in ent_cols:
        idx = u[c].to_numpy()
        ok = idx >= 0
        np.minimum.at(first, idx[ok], t[ok])
        np.maximum.at(last, idx[ok], t[ok])
        count += np.bincount(idx[ok], minlength=n_ent)
    cu.entities["first_seen"] = first
    cu.entities["last_seen"] = last
    cu.entities["updates"] = count

    rel = u["relation"].to_numpy()
    n_rel = int(rel.max()) + 1 if len(rel) else 0
    r_first = np.full(n_rel, np.inf)
    r_last = np.full(n_rel, -np.inf)
    np.minimum.at(r_first, rel, t)
    np.maximum.at(r_last, rel, t)
    width = max((len(v) for v in relation_entities.values()), default=0)
    data: dict[str, Any] = {}
    for k in range(width):
        data[f"entity_{k}"] = [relation_entities[r][k] if k < len(relation_entities[r]) else -1 for r in range(n_rel)]
    data["first_seen"] = r_first
    data["last_seen"] = r_last
    data["updates"] = np.bincount(rel, minlength=n_rel)
    direction = u["direction"].to_numpy()
    data["updates_fwd"] = np.bincount(rel[direction == 0], minlength=n_rel)
    data["updates_bwd"] = np.bincount(rel[direction == 1], minlength=n_rel)
    if "ip_len" in u:
        ip_len = u["ip_len"].to_numpy().astype(float)
        data["bytes_fwd"] = np.bincount(rel[direction == 0], weights=ip_len[direction == 0], minlength=n_rel)
        data["bytes_bwd"] = np.bincount(rel[direction == 1], weights=ip_len[direction == 1], minlength=n_rel)
    cu.relations = pd.DataFrame(data)


__all__ = [
    "CODE_LOW_RELIABILITY", "CODE_NOT_OBSERVABLE", "CODE_NOT_SUPPLIED", "CODE_OBSERVED", "CODE_STALE", "FACT_TABLES",
    "MATRIX_KINDS", "PROVENANCE_COLUMNS", "STATUS_CODE", "STATUS_ORDER", "Column", "ColumnarBuilder", "ColumnarUpdates",
    "columns_for", "fact_table", "finalise_tables", "to_columnar",
]
