"""Field specification: the vocabulary every catalogue entry is written in.

A catalogue entry (`FieldSpec`) states, for one field of the superset data model:

    id          stable identifier "<group>.<name>" (model inputs key on it, P-22)
    level       where the field comes from (flow, packet, protocol, OT, authentication, alert,
                device, event envelope, derived)
    kind        how the model treats the value; the matrix kinds become input columns, the others are
                entity keys, open-vocabulary fingerprints, or attributes carried for audit and export
    dtype       the value's type in the data model (int, float, bool, str, time, address, lists ...)
    unit        physical unit (SI where possible) or None
    layer       data-model layer L0 to L5 (`layers.Layer`)
    statuses    the observation statuses a value of this field may carry (`status.py`)
    codes       for categorical and bitmask fields: what each code or bit means
    since       the data-model version that added the field (`versioning.py`)

This module holds only the vocabulary and the value checks; the catalogue itself is `fields.py` (the
shared fields) and `native.py` (source-native attributes).
"""

from __future__ import annotations

import enum
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import numpy as np

from nagahana.core.errors import InvariantViolation
from nagahana.datamodel.layers import Layer
from nagahana.datamodel.status import IDENTITY_STATUSES, MEASUREMENT_STATUSES, ObservationStatus


class Level(enum.Enum):
    """Where a field comes from."""

    FLOW = "flow"            # aggregates over one flow (NetFlow, IPFIX, Zeek conn, Suricata flow)
    PACKET = "packet"        # derived from packets (PCAP, sampled headers, alert packets)
    PROTOCOL = "protocol"    # application-protocol metadata: DNS, HTTP, TLS, SSH, SMB, Kerberos ...
    OT = "ot"                # industrial protocols: Modbus, DNP3, IEC 60870-5-104
    DERIVED = "derived"      # computed across several records (port-access evidence)
    AUTH = "auth"            # authentication and account activity (Windows Security, sshd, sudo)
    ALERT = "alert"          # detector outputs: IDS alerts, notices, anomaly events, findings
    DEVICE = "device"        # device and sensor telemetry: interface counters, capture statistics
    EVENT = "event"          # the event envelope: severity, action, message, logging host, vendor


class Kind(enum.Enum):
    """How the model treats a field's value.

    CONTINUOUS, COUNT, CATEGORICAL, BITMASK and HISTOGRAM are the matrix kinds: each such field has a
    stable input column (`data.windows.COLUMN_SLOTS`). IDENTIFIER values name entities or objects and
    build the graph, never a numeric feature. FINGERPRINT values are categorical with an open
    vocabulary (JA4, user agents, software banners). ATTRIBUTE values are carried by the data model
    for audit, export and analysis and are not model inputs.
    """

    CONTINUOUS = "continuous"
    COUNT = "count"
    CATEGORICAL = "categorical"
    BITMASK = "bitmask"
    IDENTIFIER = "identifier"
    FINGERPRINT = "fingerprint"
    HISTOGRAM = "histogram"
    ATTRIBUTE = "attribute"


class Dtype(enum.Enum):
    """Type of a value in the data model (what `FieldValue.value` holds when it carries one)."""

    INT = "int"
    FLOAT = "float"
    BOOL = "bool"
    STR = "str"
    BYTES = "bytes"
    TIME = "time"                     # epoch seconds (UTC), float or int
    ADDRESS = "address"               # IPv4 or IPv6 address, text form
    MAC = "mac"                       # link-layer address, "aa:bb:cc:dd:ee:ff"
    INT_LIST = "list[int]"
    FLOAT_LIST = "list[float]"
    STR_LIST = "list[str]"
    ADDRESS_LIST = "list[address]"
    TIME_LIST = "list[time]"
    MAP = "map"                       # a JSON-compatible mapping (structured source data)


#: Matrix kinds: fields of these kinds become input columns.
MATRIX_KINDS: frozenset[Kind] = frozenset({Kind.CONTINUOUS, Kind.COUNT, Kind.CATEGORICAL, Kind.BITMASK, Kind.HISTOGRAM})

#: Default dtype of each kind (ATTRIBUTE has none: its dtype is always stated).
_DEFAULT_DTYPE: dict[Kind, Dtype] = {
    Kind.CONTINUOUS: Dtype.FLOAT,
    Kind.COUNT: Dtype.INT,
    Kind.CATEGORICAL: Dtype.INT,
    Kind.BITMASK: Dtype.INT,
    Kind.HISTOGRAM: Dtype.INT_LIST,
    Kind.IDENTIFIER: Dtype.STR,
    Kind.FINGERPRINT: Dtype.STR,
}

#: Dtypes a matrix kind may have (a matrix cell is a float64 number; a histogram is a tuple of counts).
_MATRIX_DTYPES: dict[Kind, frozenset[Dtype]] = {
    Kind.CONTINUOUS: frozenset({Dtype.FLOAT}),
    Kind.COUNT: frozenset({Dtype.INT}),
    Kind.CATEGORICAL: frozenset({Dtype.INT}),
    Kind.BITMASK: frozenset({Dtype.INT}),
    Kind.HISTOGRAM: frozenset({Dtype.INT_LIST}),
}


@dataclass(frozen=True)
class FieldSpec:
    """Definition of one field of the superset data model. See the module docstring.

    The first seven attributes keep their historical positional order, so entries written as
    `FieldSpec(id, level, kind, unit, description, required_by)` stay valid.
    """

    id: str
    level: Level
    kind: Kind
    unit: str | None
    description: str
    required_by: str
    example: bool = False
    layer: Layer = Layer.STATE
    dtype: Dtype | None = None
    statuses: frozenset[ObservationStatus] | None = None
    codes: Mapping[int, str] | None = field(default=None, compare=False)
    since: str = "0.1"

    def __post_init__(self) -> None:
        if not self.id or "." not in self.id or self.id != self.id.strip():
            raise InvariantViolation(f"Field id {self.id!r} must be '<group>.<name>' without surrounding spaces.")
        dtype = self.dtype
        if dtype is None:
            if self.kind not in _DEFAULT_DTYPE:
                raise InvariantViolation(f"{self.id}: an {self.kind.value} field must state its dtype.")
            dtype = _DEFAULT_DTYPE[self.kind]
            object.__setattr__(self, "dtype", dtype)
        if self.kind in _MATRIX_DTYPES and dtype not in _MATRIX_DTYPES[self.kind]:
            raise InvariantViolation(f"{self.id}: kind {self.kind.value} cannot have dtype {dtype.value}.")
        if self.statuses is None:
            default = IDENTITY_STATUSES if self.kind is Kind.IDENTIFIER else MEASUREMENT_STATUSES
            object.__setattr__(self, "statuses", default)
        assert self.statuses is not None
        if ObservationStatus.NOT_SUPPLIED not in self.statuses:
            # D-41: any source may lack any field, so "not supplied" is always admissible.
            raise InvariantViolation(f"{self.id}: NOT_SUPPLIED must be an admissible status (D-41).")
        if self.codes is not None:
            if self.kind not in (Kind.CATEGORICAL, Kind.BITMASK, Kind.ATTRIBUTE):
                raise InvariantViolation(f"{self.id}: only categorical, bitmask or attribute fields carry codes.")
            object.__setattr__(self, "codes", MappingProxyType(dict(self.codes)))

    @property
    def is_matrix(self) -> bool:
        """True if the field is an input column (a matrix kind)."""
        return self.kind in MATRIX_KINDS

    def admits(self, status: ObservationStatus) -> bool:
        """True if `status` is admissible for this field."""
        assert self.statuses is not None
        return status in self.statuses


def _is_int(value: Any) -> bool:
    return (isinstance(value, int | np.integer)) and not isinstance(value, bool | np.bool_)


def _is_number(value: Any) -> bool:
    if _is_int(value):
        return True
    return isinstance(value, float | np.floating) and math.isfinite(float(value))


def _is_seq(value: Any) -> bool:
    return isinstance(value, tuple | list)


def check_value(spec: FieldSpec, value: Any) -> str | None:
    """Why `value` is not a valid value of `spec`, or None when it is.

    Numbers must be finite (a NaN or infinity is absence, which has a status, D-41). Lists may be
    tuples or lists; their elements are checked one by one.
    """
    d = spec.dtype
    if d is Dtype.INT:
        return None if _is_int(value) else f"expected int, got {type(value).__name__}"
    if d is Dtype.FLOAT or d is Dtype.TIME:
        return None if _is_number(value) else f"expected a finite number, got {value!r}"
    if d is Dtype.BOOL:
        return None if isinstance(value, bool | np.bool_) else f"expected bool, got {type(value).__name__}"
    if d in (Dtype.STR, Dtype.ADDRESS, Dtype.MAC):
        return None if isinstance(value, str) else f"expected str, got {type(value).__name__}"
    if d is Dtype.BYTES:
        return None if isinstance(value, bytes) else f"expected bytes, got {type(value).__name__}"
    if d is Dtype.MAP:
        return None if isinstance(value, Mapping) else f"expected a mapping, got {type(value).__name__}"
    if not _is_seq(value):
        return f"expected a list, got {type(value).__name__}"
    if d is Dtype.INT_LIST:
        ok = all(_is_int(x) for x in value)
    elif d in (Dtype.FLOAT_LIST, Dtype.TIME_LIST):
        ok = all(_is_number(x) for x in value)
    else:  # STR_LIST, ADDRESS_LIST
        ok = all(isinstance(x, str) for x in value)
    return None if ok else f"list elements are not all of type {d.value}"


__all__ = ["MATRIX_KINDS", "Dtype", "FieldSpec", "Kind", "Level", "check_value"]
