"""Router and switch telemetry: gNMI notifications with OpenConfig interface state -> state updates.

Input: one JSON object per line, a gNMI `Notification` or a `SubscribeResponse` holding one
("update": {...}); `sync_response` messages are counted and skipped. The protobuf JSON mapping writes
64-bit integers as strings and bytes as base64. A path is a list of elements {"name", "key"}; the full
path of an update is the notification's prefix followed by the update's path.

TypedValue (gnmi.proto): stringVal, intVal, uintVal, boolVal, bytesVal, floatVal, doubleVal,
decimalVal {digits, precision}, leaflistVal {element: [...]}, asciiVal, jsonVal and jsonIetfVal
(base64 JSON: an object is spread into leaves under the update's path).

Every leaf under interfaces/interface[name=X] is grouped by interface; one state update per interface
per notification, with the leaf paths relative to the interface (datamodel/maps/devtelemetry.py). Leaves
outside interfaces are retained as attributes "gnmi.<path>".

Counters are cumulative; deltas between consecutive notifications of the same (target, interface) give
the dev.* fields (a decrease is a counter reset: NOT_SUPPLIED and counted). dev.if_status combines
admin-status and oper-status; a component carried from an earlier notification makes the value STALE
with the age of that component (AS-702).
"""

from __future__ import annotations

import base64
import binascii
import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any

from nagahana.datamodel.records import FieldValue
from nagahana.datamodel.status import ObservationStatus
from nagahana.ingest.config import GnmiConfig
from nagahana.ingest.convert import ConvContext
from nagahana.ingest.core import BoundedLRU, MalformedRecord, RawRecord, StreamAdapter, UpdateDraft, iter_lines, open_source
from nagahana.ingest.mapping import mapper, set_field
from nagahana.ingest.timeparse import epoch_integer

ADAPTER_VERSION = "1.0.0"
RECORD_TYPE = "gnmi.interface"
_STATUS_UP = "UP"
#: dev.* delta fields and the counters (relative leaf paths) they sum; the first alternative present wins.
_DELTAS: dict[str, tuple[tuple[str, ...], ...]] = {
    "dev.if_in_octets": (("state/counters/in-octets",),),
    "dev.if_out_octets": (("state/counters/out-octets",),),
    "dev.if_in_packets": (("state/counters/in-pkts",),
                          ("state/counters/in-unicast-pkts", "state/counters/in-multicast-pkts",
                           "state/counters/in-broadcast-pkts")),
    "dev.if_out_packets": (("state/counters/out-pkts",),
                           ("state/counters/out-unicast-pkts", "state/counters/out-multicast-pkts",
                            "state/counters/out-broadcast-pkts")),
    "dev.if_errors": (("state/counters/in-errors", "state/counters/out-errors"),),
    "dev.if_discards": (("state/counters/in-discards", "state/counters/out-discards"),),
    "dev.if_in_errors": (("state/counters/in-errors",),),
    "dev.if_out_errors": (("state/counters/out-errors",),),
    "dev.if_in_discards": (("state/counters/in-discards",),),
    "dev.if_out_discards": (("state/counters/out-discards",),),
    "dev.if_in_unicast": (("state/counters/in-unicast-pkts",),),
    "dev.if_out_unicast": (("state/counters/out-unicast-pkts",),),
    "dev.if_in_multicast": (("state/counters/in-multicast-pkts",),),
    "dev.if_out_multicast": (("state/counters/out-multicast-pkts",),),
    "dev.if_in_broadcast": (("state/counters/in-broadcast-pkts",),),
    "dev.if_out_broadcast": (("state/counters/out-broadcast-pkts",),),
    "dev.if_in_unknown_protos": (("state/counters/in-unknown-protos",),),
}


def typed_value(tv: Any) -> Any:
    """The Python value of a gNMI TypedValue in the protobuf JSON mapping."""
    if not isinstance(tv, dict) or len(tv) != 1:
        raise MalformedRecord("bad-typed-value", str(tv)[:80])
    kind, v = next(iter(tv.items()))
    if kind in ("intVal", "uintVal"):
        return int(v)
    if kind in ("stringVal", "asciiVal"):
        return str(v)
    if kind == "boolVal":
        return bool(v)
    if kind in ("floatVal", "doubleVal"):
        return float(v)
    if kind == "decimalVal":
        digits, precision = int(v.get("digits", 0)), int(v.get("precision", 0))
        return digits / (10 ** precision)
    if kind == "bytesVal":
        return base64.b64decode(v)
    if kind == "leaflistVal":
        return [typed_value(e) for e in v.get("element", [])]
    if kind in ("jsonVal", "jsonIetfVal"):
        try:
            return json.loads(base64.b64decode(v))
        except (binascii.Error, ValueError) as exc:
            raise MalformedRecord("bad-json-value", str(exc)) from None
    raise MalformedRecord("unknown-typed-value", kind)


def path_elems(path: Any) -> list[tuple[str, dict[str, str]]]:
    """[(name, keys)] of a gNMI Path (elem form; the deprecated string "element" form is read too)."""
    if path is None:
        return []
    if not isinstance(path, dict):
        raise MalformedRecord("bad-path", str(path)[:80])
    elems = path.get("elem")
    if elems is not None:
        return [(str(e.get("name", "")), {str(k): str(x) for k, x in (e.get("key") or {}).items()}) for e in elems]
    return [(str(e), {}) for e in path.get("element", [])]


def _leaves(prefix: list[str], value: Any) -> Iterator[tuple[str, Any]]:
    """(relative path, leaf value) pairs of a JSON value (objects spread, module prefixes dropped)."""
    if isinstance(value, dict):
        for k, v in value.items():
            name = k.split(":", 1)[-1]
            yield from _leaves([*prefix, name], v)
    else:
        yield "/".join(prefix), value


@dataclass
class _Interface:
    """What one notification says about one interface."""

    name: str
    leaves: dict[str, Any] = field(default_factory=dict)


class GnmiSource(StreamAdapter):
    """gNMI interface telemetry (JSON lines) -> state updates. See the module docstring."""

    name = "device-telemetry"
    source_type = "gnmi"
    version = ADAPTER_VERSION

    def __init__(self, source: str | Path | bytes | IO[bytes], *, config: GnmiConfig | None = None, **kw: Any) -> None:
        self.config = config or GnmiConfig()
        super().__init__(source, self.config.common, **kw)
        self.prev: BoundedLRU[tuple[str, str], tuple[float, dict[str, int]]] = BoundedLRU(
            self.config.max_interfaces, "gnmi_counters", self.stats)
        self.status: BoundedLRU[tuple[str, str], dict[str, tuple[str, float]]] = BoundedLRU(
            self.config.max_interfaces, "gnmi_status", self.stats)
        self._ctx = ConvContext()

    def raw_records(self) -> Iterator[RawRecord | tuple[str, RawRecord]]:
        stream, location, close = open_source(self.source)
        try:
            for item in iter_lines(stream, location, max_bytes=self.config.common.max_record_bytes):
                if isinstance(item, tuple) or item.data.strip():
                    yield item
        finally:
            if close:
                stream.close()

    def decode(self, raw: RawRecord) -> Iterable[UpdateDraft]:
        try:
            msg = json.loads(raw.data)
        except (ValueError, UnicodeDecodeError) as exc:
            raise MalformedRecord("json-decode", str(exc)) from None
        if not isinstance(msg, dict):
            raise MalformedRecord("json-not-object", type(msg).__name__)
        if msg.get("sync_response") is not None and "update" not in msg:
            self.stats.counters["sync_responses"] += 1
            return []
        note = msg.get("update") if isinstance(msg.get("update"), dict) else msg
        if "timestamp" not in note:
            raise MalformedRecord("no-timestamp", "notification without timestamp")
        try:
            ts_ns = int(str(note["timestamp"]))
        except ValueError:
            raise MalformedRecord("bad-timestamp", str(note["timestamp"])[:40]) from None
        prefix = note.get("prefix") or {}
        target = str(prefix.get("target") or note.get("target") or "")
        origin = str(prefix.get("origin") or "")
        base = path_elems(prefix)
        interfaces: dict[str, _Interface] = {}
        other: dict[str, Any] = {}
        updates = note.get("update") or []
        if not isinstance(updates, list):
            raise MalformedRecord("bad-update-list", type(updates).__name__)
        for upd in updates:
            if not isinstance(upd, dict):
                raise MalformedRecord("bad-update", str(upd)[:80])
            elems = base + path_elems(upd.get("path"))
            value = typed_value(upd.get("val")) if "val" in upd else None
            names = [n.split(":", 1)[-1] for n, _ in elems]
            if len(names) >= 2 and names[0] == "interfaces" and names[1] == "interface" and elems[1][1].get("name"):
                ifname = elems[1][1]["name"]
                itf = interfaces.setdefault(ifname, _Interface(ifname))
                for rel, leaf in _leaves(names[2:], value):
                    itf.leaves[rel] = leaf
            else:
                for rel, leaf in _leaves(names, value):
                    other[rel] = leaf
        out = []
        for k, itf in enumerate(interfaces.values()):
            out.append(self._interface(raw, k, target, origin, ts_ns, itf, other if k == 0 else {}))
        if not interfaces:
            d = UpdateDraft(RECORD_TYPE, raw)
            d.time = epoch_integer(ts_ns, "ns")
            if target:
                set_field(d, "event.hostname", target, RECORD_TYPE, self.stats)
            self._retain(d, other)
            d.add_entity(self.resolver.host_name(target) if target else None, "subject")
            out.append(d)
        return out

    def _retain(self, d: UpdateDraft, other: dict[str, Any]) -> None:
        for rel, leaf in other.items():
            key = f"gnmi.{rel.replace('/', '.')}"
            self.stats.retain(key)
            d.attributes[key] = FieldValue(key, leaf if not isinstance(leaf, bytes) else {"b64": base64.b64encode(leaf).decode()},
                                           ObservationStatus.OBSERVED, RECORD_TYPE)

    def _interface(self, raw: RawRecord, k: int, target: str, origin: str, ts_ns: int, itf: _Interface,
                   other: dict[str, Any]) -> UpdateDraft:
        sub = RawRecord(raw.data, raw.location, raw.index, offset=raw.offset, line=raw.line, sub_index=k)
        d = UpdateDraft(RECORD_TYPE, sub)
        src = RECORD_TYPE
        values: dict[str, Any] = {"timestamp": ts_ns, "target": target, "name": itf.name, **itf.leaves}
        if origin:
            values["origin"] = origin
        self._ctx.values = values
        mapper(RECORD_TYPE).apply(values, d, self._ctx, self.stats, source=src,
                                  keep_unmapped=self.config.common.keep_unmapped)
        self._retain(d, other)
        t = ts_ns / 1e9
        key = (target, itf.name)
        self._status(d, key, itf.leaves, t, src)
        now = {p: int(v) for p, v in itf.leaves.items() if p.startswith("state/counters/") and isinstance(v, int)}
        prev = self.prev.get(key)
        if now:
            merged = dict(prev[1]) if prev is not None else {}
            merged.update(now)
            self.prev[key] = (t, merged)
        if prev is not None and now:
            t0, before = prev
            if t > t0:
                set_field(d, "dev.interval", t - t0, src, self.stats)
                deltas: dict[str, int] = {}
                for p, v in now.items():
                    if p in before:
                        if v < before[p]:
                            self.stats.refuse(f"gnmi.{p}", "counter reset")
                        else:
                            deltas[p] = v - before[p]
                for fid, options in _DELTAS.items():
                    for parts in options:
                        if all(p in deltas for p in parts):
                            set_field(d, fid, sum(deltas[p] for p in parts), src, self.stats)
                            break
            else:
                self.stats.refuse("dev.interval", "notifications out of order")
        d.add_entity(self.resolver.host_name(target) if target else self.resolver.sensor(), "subject")
        return d

    def _status(self, d: UpdateDraft, key: tuple[str, str], leaves: dict[str, Any], t: float, src: str) -> None:
        known = self.status.get(key) or {}
        for leaf, name in (("state/admin-status", "admin"), ("state/oper-status", "oper")):
            v = leaves.get(leaf)
            if isinstance(v, str):
                known[name] = (v.split(":")[-1].upper(), t)
        self.status[key] = known
        if "admin" not in known or "oper" not in known:
            return
        bits = (1 if known["admin"][0] == _STATUS_UP else 0) | (2 if known["oper"][0] == _STATUS_UP else 0)
        oldest = min(known["admin"][1], known["oper"][1])
        if oldest < t:
            fv = FieldValue("dev.if_status", bits, ObservationStatus.STALE, src, age_s=t - oldest)
            if d.fields.get("dev.if_status") is None or not d.fields["dev.if_status"].contributes:
                d.fields["dev.if_status"] = fv
        else:
            set_field(d, "dev.if_status", bits, src, self.stats)


__all__ = ["ADAPTER_VERSION", "GnmiSource", "path_elems", "typed_value"]
