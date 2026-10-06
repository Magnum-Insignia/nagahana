"""Streaming ingest core shared by every adapter.

An adapter is two functions: framing (`raw_records`: the source -> `RawRecord`s, each with its bytes
and where they sit) and decoding (`decode`: one raw record -> `UpdateDraft`s through the mapping
tables). `StreamAdapter` wraps them into the `Source` protocol (`updates()`), the columnar form
(`columnar()`) and single-record decoding for streams (`decode_bytes`, used by the Kafka consumer).

Guarantees, for every adapter
-----------------------------
- Provenance per record: source type, record type, location (file or URI), byte offset and length,
  line, record ordinal, ordinal inside a container, sensor, exporter, original time text, and the
  SHA-256 of the record's bytes.
- Malformed records are quarantined with a reason (`Quarantine`), never dropped silently; a value that
  cannot be a measurement makes its field NOT_SUPPLIED and is counted per field and reason
  (`IngestStats.refused`).
- Bounded memory: framing reads in chunks and refuses records above `max_record_bytes`; every
  per-key table is a `BoundedLRU` whose evictions are counted; the reorder buffer is bounded.
- Event time is UTC epoch float64 seconds, with exact nanoseconds kept where the source has them;
  records leave in event-time order within a bounded window, and lateness beyond it is tagged as
  reorder uncertainty (`ReorderBuffer`).
"""

from __future__ import annotations

import abc
import base64
import hashlib
import heapq
import io
import ipaddress
import json
import math
import re
import time
from collections import Counter, OrderedDict, deque
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any, ClassVar, Generic, TypeVar

from nagahana.core.errors import InvariantViolation
from nagahana.datamodel.columnar import Column, ColumnarBuilder, ColumnarUpdates
from nagahana.datamodel.records import EntityRef, FieldValue, OrderingInfo, Provenance, StateUpdate
from nagahana.datamodel.status import ObservationStatus
from nagahana.ingest.config import CommonConfig, QuarantineConfig, ReorderConfig
from nagahana.ingest.convert import JsonNumber
from nagahana.ingest.timeparse import Instant

K = TypeVar("K")
V = TypeVar("V")


@dataclass(slots=True)
class RawRecord:
    """One raw record of a source and where it sits.

    Attributes
    ----------
    data: the record's bytes as stored (a line without its terminator, a binary record).
    location: file path, URI or stream name.
    index: ordinal of the record in the source (0-based).
    offset: byte offset of `data` in `location` (None when the source is not addressable).
    line: 1-based line number for line-oriented sources.
    sub_index: ordinal inside a container (a flow record inside a datagram).
    received_time: when the collector received it (live sources).
    exporter: address of the device that sent it (flow export, syslog).
    meta: framing context (Zeek header, Kafka headers, capture interface).
    """

    data: bytes
    location: str
    index: int
    offset: int | None = None
    line: int | None = None
    sub_index: int | None = None
    received_time: float | None = None
    exporter: str | None = None
    meta: dict[str, Any] | None = None

    @property
    def length(self) -> int:
        return len(self.data)

    def sha256(self) -> str:
        return hashlib.sha256(self.data).hexdigest()


class MalformedRecord(Exception):
    """A raw record that cannot be decoded; `reason` is a short stable code, `detail` free text."""

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason
        self.detail = detail


@dataclass
class QuarantineEntry:
    """One quarantined record (bytes truncated to the configured maximum)."""

    reason: str
    detail: str
    location: str
    index: int
    offset: int | None
    line: int | None
    sub_index: int | None
    length: int
    sha256: str
    data: bytes

    def to_json(self) -> dict[str, Any]:
        return {"reason": self.reason, "detail": self.detail, "location": self.location, "index": self.index,
                "offset": self.offset, "line": self.line, "sub_index": self.sub_index, "length": self.length,
                "sha256": self.sha256, "data_b64": base64.b64encode(self.data).decode("ascii")}


class Quarantine:
    """Malformed records with their reasons: counts for all, bounded samples, optional JSON Lines file."""

    def __init__(self, cfg: QuarantineConfig | None = None) -> None:
        self.cfg = cfg or QuarantineConfig()
        self.counts: Counter[str] = Counter()
        self.samples: deque[QuarantineEntry] = deque(maxlen=max(self.cfg.max_samples, 0) or None)
        self._fh: IO[str] | None = None
        if self.cfg.max_samples == 0:
            self.samples = deque(maxlen=1)

    @property
    def total(self) -> int:
        return sum(self.counts.values())

    def add(self, raw: RawRecord, reason: str, detail: str = "") -> QuarantineEntry:
        """Quarantine `raw` for `reason`."""
        entry = QuarantineEntry(reason, detail[:500], raw.location, raw.index, raw.offset, raw.line, raw.sub_index,
                                raw.length, raw.sha256(), raw.data[: self.cfg.max_raw_bytes])
        self.counts[reason] += 1
        if self.cfg.max_samples > 0:
            self.samples.append(entry)
        if self.cfg.path is not None:
            if self._fh is None:
                self._fh = open(self.cfg.path, "a", encoding="utf-8", newline="\n")  # noqa: SIM115
            self._fh.write(json.dumps(entry.to_json(), sort_keys=True) + "\n")
            self._fh.flush()
        return entry

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None


@dataclass
class IngestStats:
    """What an adapter read, emitted, refused, retained and evicted."""

    records: int = 0
    updates: int = 0
    refused: dict[str, Counter[str]] = field(default_factory=dict)
    retained_attributes: int = 0
    retained_keys: set[str] = field(default_factory=set)
    evictions: Counter[str] = field(default_factory=Counter)
    late_records: int = 0
    max_lateness_s: float = 0.0
    fallback_subject: int = 0
    no_time: int = 0
    counters: Counter[str] = field(default_factory=Counter)
    _MAX_KEYS: ClassVar[int] = 10_000

    def refuse(self, field_id: str, reason: str) -> None:
        self.refused.setdefault(field_id, Counter())[reason] += 1

    def retain(self, key: str) -> None:
        self.retained_attributes += 1
        if len(self.retained_keys) < self._MAX_KEYS:
            self.retained_keys.add(key)

    def as_dict(self) -> dict[str, Any]:
        return {
            "records": self.records, "updates": self.updates,
            "refused": {k: dict(v) for k, v in sorted(self.refused.items())},
            "retained_attributes": self.retained_attributes, "retained_keys": sorted(self.retained_keys),
            "evictions": dict(self.evictions), "late_records": self.late_records, "max_lateness_s": self.max_lateness_s,
            "fallback_subject": self.fallback_subject, "no_time": self.no_time, "counters": dict(self.counters),
        }


class BoundedLRU(Generic[K, V]):
    """A mapping with at most `maxsize` keys; the least recently used is evicted and counted.

    `on_evict(key, value)` lets the owner act on an eviction (emit a pending update, mark a loss).
    """

    def __init__(self, maxsize: int, name: str, stats: IngestStats | None = None,
                 on_evict: Callable[[K, V], None] | None = None) -> None:
        if maxsize < 1:
            raise InvariantViolation(f"{name}: maxsize must be >= 1")
        self.maxsize = maxsize
        self.name = name
        self.stats = stats
        self.on_evict = on_evict
        self.evictions = 0
        self._d: OrderedDict[K, V] = OrderedDict()

    def __len__(self) -> int:
        return len(self._d)

    def __contains__(self, key: object) -> bool:
        return key in self._d

    def get(self, key: K, default: V | None = None) -> V | None:
        v = self._d.get(key)
        if v is None:
            return default
        self._d.move_to_end(key)
        return v

    def peek(self, key: K) -> V | None:
        """The value without marking it as used."""
        return self._d.get(key)

    def __setitem__(self, key: K, value: V) -> None:
        if key in self._d:
            self._d.move_to_end(key)
        self._d[key] = value
        while len(self._d) > self.maxsize:
            old_k, old_v = self._d.popitem(last=False)
            self.evictions += 1
            if self.stats is not None:
                self.stats.evictions[self.name] += 1
            if self.on_evict is not None:
                self.on_evict(old_k, old_v)

    def setdefault(self, key: K, default: V) -> V:
        v = self.get(key)
        if v is None:
            self[key] = default
            return default
        return v

    def pop(self, key: K, default: V | None = None) -> V | None:
        return self._d.pop(key, default)

    def items(self) -> list[tuple[K, V]]:
        return list(self._d.items())

    def values(self) -> list[V]:
        return list(self._d.values())

    def keys(self) -> list[K]:
        return list(self._d.keys())

    def clear(self) -> None:
        self._d.clear()


class ReorderBuffer(Generic[V]):
    """Bounded event-time reordering (AS-671).

    Items wait until the newest time seen is `window_s` past theirs, or until more than `max_items` wait;
    they leave in (time, arrival) order. The watermark of an item is the latest time emitted before it;
    an item older than that watermark is late: it leaves at once and its lateness is its reorder
    uncertainty.
    """

    def __init__(self, cfg: ReorderConfig, stats: IngestStats | None = None) -> None:
        if cfg.window_s < 0 or cfg.max_records < 0:
            raise InvariantViolation("reorder window and size must be >= 0")
        self.window = cfg.window_s
        self.max_items = cfg.max_records
        self.stats = stats
        self._heap: list[tuple[float, int, V]] = []
        self._seq = 0
        self._newest = -math.inf
        self.emitted_max = math.nan

    def __len__(self) -> int:
        return len(self._heap)

    def push(self, t: float, item: V) -> list[tuple[V, float, float]]:
        """Add one item; returns the items that may leave now as (item, watermark, lateness)."""
        out: list[tuple[V, float, float]] = []
        if not math.isnan(self.emitted_max) and t < self.emitted_max:
            lateness = self.emitted_max - t
            if self.stats is not None:
                self.stats.late_records += 1
                self.stats.max_lateness_s = max(self.stats.max_lateness_s, lateness)
            out.append((item, self.emitted_max, lateness))
            return out
        heapq.heappush(self._heap, (t, self._seq, item))
        self._seq += 1
        self._newest = max(self._newest, t)
        while self._heap and (self._heap[0][0] <= self._newest - self.window or len(self._heap) > self.max_items):
            out.append(self._pop())
        return out

    def _pop(self) -> tuple[V, float, float]:
        t, _, item = heapq.heappop(self._heap)
        wm = self.emitted_max
        if math.isnan(wm) or t > wm:
            self.emitted_max = t
        return item, wm, 0.0

    def flush(self) -> list[tuple[V, float, float]]:
        out = []
        while self._heap:
            out.append(self._pop())
        return out


def open_source(source: str | Path | bytes | IO[bytes]) -> tuple[IO[bytes], str, bool]:
    """A binary stream for `source`: a path, raw bytes, or an open binary stream. Returns (stream,
    location, close_after)."""
    if isinstance(source, bytes | bytearray):
        return io.BytesIO(bytes(source)), "<bytes>", True
    if isinstance(source, str | Path):
        p = Path(source)
        return open(p, "rb"), str(p), True                       # noqa: SIM115
    name = getattr(source, "name", "<stream>")
    return source, str(name), False


def iter_lines(stream: IO[bytes], location: str, *, max_bytes: int, chunk: int = 1 << 20,
               start_index: int = 0) -> Iterator[RawRecord | tuple[str, RawRecord]]:
    """Lines of a binary stream with byte offsets and 1-based line numbers (terminator removed).

    Reads in chunks, so memory is bounded by `max_bytes` plus one chunk. A line longer than
    `max_bytes` is yielded as ("oversize", record) with its first `max_bytes` bytes, for the caller to
    quarantine; the rest of that line is skipped.
    """
    buf = b""
    pos = 0                       # file offset of buf[0]
    line_no = 0
    index = start_index
    oversize: RawRecord | None = None
    while True:
        block = stream.read(chunk)
        if not block:
            break
        buf += block
        start = 0
        while True:
            nl = buf.find(b"\n", start)
            if nl < 0:
                break
            line_no += 1
            body = buf[start:nl]
            if body.endswith(b"\r"):
                body = body[:-1]
            if oversize is not None:
                yield ("oversize", oversize)
                oversize = None
                index += 1
            elif len(body) > max_bytes:
                yield ("oversize", RawRecord(body[:max_bytes], location, index, offset=pos + start, line=line_no))
                index += 1
            else:
                yield RawRecord(body, location, index, offset=pos + start, line=line_no)
                index += 1
            start = nl + 1
        pos += start
        buf = buf[start:]
        if len(buf) > max_bytes and oversize is None:
            # an over-long line in progress: keep its head, drop the rest until its newline
            oversize = RawRecord(buf[:max_bytes], location, index, offset=pos, line=line_no + 1)
        if oversize is not None:
            pos += len(buf)
            buf = b""
    if oversize is not None:
        yield ("oversize", oversize)
    elif buf:
        line_no += 1
        body = buf[:-1] if buf.endswith(b"\r") else buf
        yield RawRecord(body, location, index, offset=pos, line=line_no)


_WS = frozenset(b" \t\r\n")
_STRUCT = re.compile(rb'[\[\]{}"]')
_IN_STR = re.compile(rb'["\\]')
_SCALAR_END = re.compile(rb"[,\]\s]")


def iter_json_array(stream: IO[bytes], location: str, *, max_bytes: int,
                    chunk: int = 1 << 20) -> Iterator[RawRecord | tuple[str, RawRecord]]:
    """Top-level elements of a JSON array, one at a time, with their byte ranges.

    A scanner jumps between structural bytes (brackets, braces, quotes, backslashes) with a regular
    expression and copies the runs between them in bulk, tracking nesting depth, strings and escapes, so
    an element is cut out without parsing the whole array (tshark -T json writes one array holding every
    packet). An element above `max_bytes` is yielded as ("oversize", record) with its head; scanning goes
    on to its end without keeping the rest.
    """
    depth = 0
    in_str = esc = scalar = oversize = started = False
    elem = bytearray()
    elem_start = -1                     # file offset of the current element; -1 between elements
    index = 0
    pos = 0

    def take(part: bytes) -> None:
        nonlocal oversize
        if oversize:
            return
        elem.extend(part)
        if len(elem) > max_bytes:
            oversize = True
            del elem[max_bytes:]

    def finished() -> RawRecord | tuple[str, RawRecord]:
        nonlocal elem_start, index
        rec = RawRecord(bytes(elem), location, index, offset=elem_start)
        index += 1
        elem_start = -1
        return ("oversize", rec) if oversize else rec

    while True:
        block = stream.read(chunk)
        if not block:
            break
        n = len(block)
        i = 0
        while i < n:
            if not started:
                c = block[i]
                if c in _WS:
                    i += 1
                    continue
                if c != 0x5B:                                      # "["
                    raise MalformedRecord("not-a-json-array", f"byte {chr(c)!r} at offset {pos + i}")
                started = True
                i += 1
                continue
            if elem_start < 0:
                c = block[i]
                if c in _WS or c == 0x2C:                          # whitespace or ","
                    i += 1
                    continue
                if c == 0x5D:                                      # "]": end of the array
                    return
                elem_start, depth, oversize = pos + i, 0, False
                elem.clear()
                scalar = c not in (0x7B, 0x5B, 0x22)
            if esc:                                                # the byte after a backslash
                take(block[i:i + 1])
                i += 1
                esc = False
                continue
            if in_str:
                m = _IN_STR.search(block, i)
                if m is None:
                    take(block[i:])
                    i = n
                    continue
                j = m.start()
                take(block[i:j + 1])
                i = j + 1
                if block[j] == 0x5C:
                    if i < n:
                        take(block[i:i + 1])
                        i += 1
                    else:
                        esc = True
                else:
                    in_str = False
                    if depth == 0:                                 # a top-level string ended
                        yield finished()
                continue
            if scalar:
                m = _SCALAR_END.search(block, i)
                if m is None:
                    take(block[i:])
                    i = n
                    continue
                take(block[i:m.start()])
                i = m.start()
                yield finished()
                continue
            m = _STRUCT.search(block, i)
            if m is None:
                take(block[i:])
                i = n
                continue
            j = m.start()
            c = block[j]
            take(block[i:j + 1])
            i = j + 1
            if c == 0x22:
                in_str = True
            elif c in (0x7B, 0x5B):
                depth += 1
            else:
                depth -= 1
                if depth == 0:
                    yield finished()
        pos += n
    if scalar and elem_start >= 0:
        yield finished()
    if not started or elem_start >= 0 or in_str:
        raise MalformedRecord("truncated-json-array", f"array not closed at offset {pos}")
    raise MalformedRecord("truncated-json-array", f"no closing bracket at offset {pos}")


class EntityResolver:
    """Typed entities from source identifiers (D-47, D-48).

    An IP address is a `multicast` entity when it names a group (IPv4 224.0.0.0/4 and 255.255.255.255,
    IPv6 ff00::/8), a `host` inside the monitored network and `external` otherwise. A service is
    "<address>:<port>/<protocol>" for TCP, UDP and SCTP. A host known only by name is keyed by its name
    in lower case (DNS names are case-insensitive, RFC 4343). An account is "<scope>\\<name>"; Windows
    accounts are case-insensitive, Unix accounts are not (`case_insensitive`).
    """

    PROTO_NAMES: ClassVar[dict[int, str]] = {6: "tcp", 17: "udp", 132: "sctp"}

    def __init__(self, common: CommonConfig, stats: IngestStats | None = None) -> None:
        self.nets = [ipaddress.ip_network(c, strict=False) for c in common.internal_networks]
        self.cache: BoundedLRU[str, tuple[str, str]] = BoundedLRU(common.max_entity_cache, "entity_kinds", stats)
        self.sensor_key = f"sensor:{common.sensor_id or common.source_id or 'unnamed'}"

    def address(self, text: str | None) -> EntityRef | None:
        """The entity of an IP address (canonical text), or None when `text` is not an address."""
        if not text:
            return None
        hit = self.cache.get(text)
        if hit is None:
            try:
                ip = ipaddress.ip_address(text.strip("[]"))
            except ValueError:
                return None
            if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
                ip = ip.ipv4_mapped
            key = str(ip)
            if ip.is_multicast or key == "255.255.255.255":
                hit = ("multicast", key)
            elif any(ip.version == n.version and ip in n for n in self.nets):
                hit = ("host", key)
            else:
                hit = ("external", key)
            self.cache[text] = hit
        return EntityRef(*hit)

    def is_internal(self, ref: EntityRef) -> bool:
        return ref.kind == "host"

    def service(self, responder: EntityRef | None, port: int | None, proto: int | None) -> EntityRef | None:
        """The responder's service entity, for TCP, UDP and SCTP with a known port."""
        if responder is None or port is None or proto not in self.PROTO_NAMES:
            return None
        assert proto is not None
        return EntityRef("service", f"{responder.id}:{int(port)}/{self.PROTO_NAMES[proto]}")

    def host_name(self, name: str | None) -> EntityRef | None:
        """A host known by name (an address-shaped name resolves as an address)."""
        if not name or name in ("-", ""):
            return None
        ref = self.address(name)
        if ref is not None:
            return ref
        return EntityRef("host", name.strip().rstrip(".").lower())

    def account(self, name: str | None, scope: str | None = None, *, case_insensitive: bool = True) -> EntityRef | None:
        """An account entity "<scope>\\<name>" (scope: domain, realm or host)."""
        if not name or name in ("-", ""):
            return None
        n = name.strip()
        s = (scope or "").strip()
        if "@" in n and not s:                         # user@REALM (Kerberos, UPN)
            n, s = n.split("@", 1)
        key = f"{s}\\{n}" if s and s != "-" else n
        return EntityRef("account", key.lower() if case_insensitive else key)

    def sensor(self) -> EntityRef:
        """The sensor (or source) itself, as the subject of records without parties."""
        return EntityRef("host", self.sensor_key)


@dataclass(slots=True)
class UpdateDraft:
    """A state update being built; it becomes a `StateUpdate` when it leaves the reorder buffer."""

    record_type: str
    raw: RawRecord
    time: Instant | None = None
    fields: dict[str, FieldValue] = field(default_factory=dict)
    attributes: dict[str, FieldValue] = field(default_factory=dict)
    entities: list[tuple[EntityRef, str]] = field(default_factory=list)
    raw_hash: str | None = None
    original_time: str | None = None
    clock_quality: str | None = None
    exporter: str | None = None

    def add_entity(self, ref: EntityRef | None, role: str) -> None:
        """Add an entity with its role (None is ignored; a repeated pair is added once)."""
        if ref is None:
            return
        if (ref, role) not in self.entities:
            self.entities.append((ref, role))

    def value(self, field_id: str) -> Any:
        """The contributing value of a field, or None."""
        fv = self.fields.get(field_id)
        return fv.value if fv is not None and fv.contributes else None

    def status(self, field_id: str) -> ObservationStatus:
        fv = self.fields.get(field_id)
        return fv.status if fv is not None else ObservationStatus.NOT_SUPPLIED


class StreamAdapter(abc.ABC):
    """Base class of the streaming adapters (module docstring).

    Subclasses set `name` (registry name), `source_type`, `version`, and implement `raw_records` and
    `decode`; `finish` flushes state held across records (a pending join or event).
    """

    name: ClassVar[str] = ""
    source_type: ClassVar[str] = ""
    version: ClassVar[str] = "1.0.0"
    implemented: ClassVar[bool] = True

    def __init__(self, source: str | Path | bytes | IO[bytes] | None, common: CommonConfig | None = None,
                 *, ingest_clock: Callable[[], float] | None = None) -> None:
        self.source = source
        self.common = common or CommonConfig()
        self.ingest_clock = ingest_clock
        self.stats = IngestStats()
        self.quarantine = Quarantine(self.common.quarantine)
        self.resolver = EntityResolver(self.common, self.stats)
        if isinstance(source, str | Path):
            default_id = Path(source).name
        else:
            default_id = getattr(source, "name", None) or self.name or "stream"
        self.source_id = self.common.source_id or str(default_id)
        if self.common.source_id is None and self.resolver.sensor_key == "sensor:unnamed":
            self.resolver.sensor_key = f"sensor:{self.source_id}"
        self._seq = 0

    @abc.abstractmethod
    def raw_records(self) -> Iterator[RawRecord | tuple[str, RawRecord]]:
        """Frame the source into raw records (or ("<reason>", record) for framing failures)."""

    @abc.abstractmethod
    def decode(self, raw: RawRecord) -> Iterable[UpdateDraft]:
        """Map one raw record to drafts; raise `MalformedRecord` when it cannot be decoded."""

    def finish(self) -> Iterable[UpdateDraft]:
        """Drafts held across records, at the end of the source."""
        return ()

    def drafts(self) -> Iterator[UpdateDraft]:
        """Every draft of the source, in source order, with malformed records quarantined."""
        for item in self.raw_records():
            if isinstance(item, tuple):
                reason, raw = item
                self.stats.records += 1
                self.quarantine.add(raw, reason, "framing")
                continue
            self.stats.records += 1
            try:
                yield from self.decode(item)
            except MalformedRecord as exc:
                self.quarantine.add(item, exc.reason, exc.detail)
        yield from self.finish()

    def _finalise(self, d: UpdateDraft, watermark: float, lateness: float) -> StateUpdate:
        """The state update of a draft leaving the reorder buffer."""
        assert d.time is not None
        t = d.time
        seq = self._seq
        self._seq += 1
        raw = d.raw
        clock = d.clock_quality or ("timezone-unverified" if not t.zone_known else None)
        entities = d.entities
        if not entities:
            entities = [(self.resolver.sensor(), "subject")]
            self.stats.fallback_subject += 1
        wm = None if math.isnan(watermark) else watermark
        ordering = OrderingInfo(
            event_time=t.seconds,
            ingest_time=self.ingest_clock() if self.ingest_clock is not None else t.seconds,
            watermark=wm,
            reorder_uncertainty_s=max(t.resolution, lateness),
            clock_quality=clock,
            event_time_ns=t.ns,
        )
        prov = Provenance(
            source_id=self.source_id, adapter=self.name, adapter_version=self.version,
            raw_hash=d.raw_hash or raw.sha256(), source_type=self.source_type, record_type=d.record_type,
            location=raw.location, offset=raw.offset, length=raw.length, line=raw.line, record_index=raw.index,
            sub_index=raw.sub_index, sensor_id=self.common.sensor_id, exporter=d.exporter or raw.exporter,
            original_time=d.original_time,
        )
        return StateUpdate(
            update_id=f"{self.source_id}:{seq}", ordering=ordering, entities=tuple(e for e, _ in entities),
            fields=d.fields, provenance=prov, roles=tuple(r for _, r in entities), attributes=d.attributes,
        )

    def updates(self) -> Iterator[StateUpdate]:
        """State updates in event-time order within the reorder window (the `Source` protocol)."""
        buf: ReorderBuffer[UpdateDraft] = ReorderBuffer(self.common.reorder, self.stats)
        try:
            for d in self.drafts():
                if d.time is None:
                    self.stats.no_time += 1
                    self.quarantine.add(d.raw, "no-event-time", d.record_type)
                    continue
                for item, wm, late in buf.push(d.time.seconds, d):
                    self.stats.updates += 1
                    yield self._finalise(item, wm, late)
            for item, wm, late in buf.flush():
                self.stats.updates += 1
                yield self._finalise(item, wm, late)
        finally:
            self.quarantine.close()

    def columnar(self, columns: Sequence[Column] | None = None, *, keep_rest: bool = True) -> ColumnarUpdates:
        """The columnar form, in the canonical column layout unless `columns` is given."""
        if columns is None:
            from nagahana.data.windows import CANONICAL_COLUMNS

            columns = CANONICAL_COLUMNS
        b = ColumnarBuilder(columns, keep_rest=keep_rest)
        for u in self.updates():
            b.add(u)
        return b.build(source_id=self.source_id, adapter=self.name, adapter_version=self.version)

    def decode_bytes(self, data: bytes, *, location: str, index: int, offset: int | None = None,
                     received_time: float | None = None, exporter: str | None = None,
                     meta: dict[str, Any] | None = None) -> list[StateUpdate]:
        """Decode one record received as bytes (a Kafka message): no reordering, ingest time = now.

        Malformed records are quarantined and give an empty list; refusals are counted as usual.
        """
        raw = RawRecord(data, location, index, offset=offset, received_time=received_time, exporter=exporter,
                        meta=meta)
        self.stats.records += 1
        if len(data) > self.common.max_record_bytes:
            self.quarantine.add(raw, "oversize", f"{len(data)} bytes")
            return []
        out: list[StateUpdate] = []
        try:
            drafts = list(self.decode(raw))
        except MalformedRecord as exc:
            self.quarantine.add(raw, exc.reason, exc.detail)
            return []
        clock = self.ingest_clock
        self.ingest_clock = clock or time.time
        try:
            for d in drafts:
                if d.time is None:
                    self.stats.no_time += 1
                    self.quarantine.add(raw, "no-event-time", d.record_type)
                    continue
                self.stats.updates += 1
                out.append(self._finalise(d, math.nan, 0.0))
        finally:
            self.ingest_clock = clock
        return out


def json_loads_exact(data: bytes | str) -> Any:
    """JSON with numbers that have a fraction or exponent kept as `JsonNumber` text."""
    return json.loads(data, parse_float=JsonNumber)


def json_safe(value: Any) -> Any:
    """`value` as JSON-compatible data (bytes as {"b64": ...}, tuples as lists, non-finite floats as text)."""
    if isinstance(value, JsonNumber):
        return float(value)
    if isinstance(value, bytes | bytearray):
        return {"b64": base64.b64encode(bytes(value)).decode("ascii")}
    if isinstance(value, Mapping):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)
    return value


__all__ = [
    "BoundedLRU", "EntityResolver", "IngestStats", "JsonNumber", "MalformedRecord", "json_loads_exact", "Quarantine", "QuarantineEntry", "RawRecord",
    "ReorderBuffer", "StreamAdapter", "UpdateDraft", "iter_json_array", "iter_lines", "json_safe", "open_source",
]
