"""Flow export: NetFlow v5, NetFlow v9 (RFC 3954) and IPFIX (RFC 7011) -> state updates.

Decoding
--------
v5      24-byte header (version, count, sys_uptime, unix_secs, unix_nsecs, flow_sequence, engine type
        and id, sampling interval) and `count` fixed records of 48 bytes.
v9      20-byte header (version, count, sys_uptime, unix_secs, sequence, source id) and FlowSets:
        id 0 templates, id 1 options templates (scope and option field lists by byte length),
        id >= 256 data records of the template with that id, padded to 32 bits.
IPFIX   16-byte message header (version 10, length, export time, sequence, observation domain) and
        Sets: id 2 templates, id 3 options templates (scope field count), id >= 256 data. Field
        specifiers carry an enterprise bit and a 4-byte private enterprise number; length 65535 marks
        a variable-length field, whose length is 1 byte, or 255 followed by 2 bytes (RFC 7011 section
        7). A template with field count 0 withdraws the template (template id 2 or 3: all of them).
        Integers may use reduced-size encoding (section 6.2). basicList, subTemplateList and
        subTemplateMultiList are decoded per RFC 6313. Reverse elements (RFC 5103) are enterprise
        29305 with the forward element's number.

Template state
--------------
Templates are cached per (exporter address, source id or observation domain, template id), with at
most `max_templates` entries (LRU, evictions counted) and expiry `template_lifetime_s` after their
last refresh in export time (RFC 7011 section 8.4). A data set whose template is unknown waits in a
bounded buffer until the template arrives (RFC 3954 section 9); when the buffer overflows, when it
waits longer than `pending_max_age_s` of the exporter's export time, or when the source ends, it is
quarantined with the reason. Options records fill a sampler table per exporter and domain.

Sequence numbers are checked per exporter and domain (v5: flows, v9: export packets, IPFIX: data
records); gaps are counted as lost records in `stats.counters`.

Inputs: a packet capture holding the export datagrams (UDP to `udp_ports`), an IPFIX file (RFC 5655:
a sequence of messages), or an iterable of `Datagram`s from a live collector.
"""

from __future__ import annotations

import ipaddress
import struct
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any

from nagahana.datamodel.maps.flowexport import IE_BY_ID, REVERSE_IES, REVERSE_PEN
from nagahana.datamodel.status import ObservationStatus
from nagahana.ingest import derive as DV
from nagahana.ingest.capture import CapturedPacket, is_capture, read_capture, udp_payload
from nagahana.ingest.config import FlowExportConfig
from nagahana.ingest.convert import ConvContext
from nagahana.ingest.core import BoundedLRU, IngestStats, MalformedRecord, RawRecord, StreamAdapter, UpdateDraft, open_source
from nagahana.ingest.mapping import mapper, set_field
from nagahana.ingest.timeparse import NS, Instant, epoch_integer, ntp64

ADAPTER_VERSION = "1.0.0"
LOW = ObservationStatus.LOW_RELIABILITY
V5_HEADER = struct.Struct(">HHIIIIBBH")
V5_RECORD = struct.Struct(">4s4s4sHHIIIIHHBBBBHHBBH")
V9_HEADER = struct.Struct(">HHIIII")
IPFIX_HEADER = struct.Struct(">HHIII")
#: Delta counters accumulated into running totals: element name -> target field.
_DELTA_TARGETS: tuple[tuple[str, str], ...] = (
    ("octetDeltaCount", "flow.bytes_fwd"), ("packetDeltaCount", "flow.packets_fwd"),
    ("reverse.octetDeltaCount", "flow.bytes_bwd"), ("reverse.packetDeltaCount", "flow.packets_bwd"),
    ("initiatorOctets", "flow.payload_bytes_fwd"), ("responderOctets", "flow.payload_bytes_bwd"),
    ("initiatorPackets", "flow.packets_fwd"), ("responderPackets", "flow.packets_bwd"),
    ("dOctets", "flow.bytes_fwd"), ("dPkts", "flow.packets_fwd"),
)
_TOTAL_TARGETS: tuple[tuple[str, str], ...] = (
    ("octetTotalCount", "flow.bytes_fwd"), ("packetTotalCount", "flow.packets_fwd"),
    ("reverse.octetTotalCount", "flow.bytes_bwd"), ("reverse.packetTotalCount", "flow.packets_bwd"),
)
#: Flow end reasons (RFC 5102 flowEndReason) after which a flow's accumulator is released.
_FLOW_ENDED = frozenset({1, 3, 4, 5})
#: Active timeout assumed when the exporter does not report flowActiveTimeout (AS-696).
DEFAULT_ACTIVE_TIMEOUT_S = 1800.0
#: Reliability floor of a running total built from a small share of a flow's lifetime.
MIN_COMPLETENESS = 0.01
#: Reliability of a running total whose flow has no start time to judge its completeness by.
UNKNOWN_COMPLETENESS = 0.5


@dataclass(frozen=True)
class Datagram:
    """One export datagram from a live collector."""

    data: bytes
    exporter: str
    received_time: float | None = None


@dataclass
class Template:
    """A (possibly options) template: field specifiers (element id, length, enterprise)."""

    template_id: int
    fields: tuple[tuple[int, int, int], ...]
    scope_count: int
    refreshed: float

    @property
    def options(self) -> bool:
        return self.scope_count > 0

    def min_length(self) -> int:
        return sum(1 if ln == 0xFFFF else ln for _, ln, _ in self.fields)


@dataclass
class DecodedRecord:
    """One data record and its context."""

    kind: str                                   # "v5", "v9", "v9_options", "ipfix", "ipfix_options"
    values: dict[str, Any]
    offset: int                                 # offset of the record bytes in the datagram
    length: int
    export_time: Instant
    boot_ms: int | None = None                  # exporter boot time (ms since epoch) for sysUpTime fields
    scope: tuple[str, ...] = ()


@dataclass
class _Pending:
    data: bytes
    set_offset: int
    header: dict[str, Any]
    export_time: Instant
    boot_ms: int | None
    raw: RawRecord


@dataclass
class _Accumulator:
    totals: dict[str, int] = field(default_factory=dict)
    created: float = 0.0


def ie_name(ie: int, pen: int) -> str:
    """Name of an element: the IANA name, "reverse.<name>", or "e<pen>.<id>" for an unknown one."""
    if pen == 0:
        entry = IE_BY_ID.get(ie)
        return entry[0] if entry is not None else f"ie{ie}"
    if pen == REVERSE_PEN:
        entry = IE_BY_ID.get(ie)
        return f"reverse.{entry[0]}" if entry is not None else f"reverse.ie{ie}"
    return f"e{pen}.{ie}"


def decode_value(ie: int, pen: int, data: bytes, decoder: FlowExportDecoder | None = None,
                 ctx: tuple[str, int] | None = None) -> Any:
    """The value of an element's bytes, per its abstract data type (RFC 7011 section 6; RFC 6313)."""
    entry = IE_BY_ID.get(ie) if pen in (0, REVERSE_PEN) else None
    if entry is None:
        return bytes(data)
    _, abstract, _ = entry
    n = len(data)
    if abstract.startswith("unsigned"):
        return int.from_bytes(data, "big")
    if abstract.startswith("signed"):
        return int.from_bytes(data, "big", signed=True)
    if abstract == "float64":
        return struct.unpack(">d", data)[0] if n == 8 else (struct.unpack(">f", data)[0] if n == 4 else bytes(data))
    if abstract == "float32":
        return struct.unpack(">f", data)[0] if n == 4 else bytes(data)
    if abstract == "boolean":
        return {1: True, 2: False}.get(data[0], bytes(data)) if n == 1 else bytes(data)
    if abstract == "macAddress":
        return data.hex(":") if n == 6 else bytes(data)
    if abstract == "string":
        return data.decode("utf-8", "backslashreplace")
    if abstract == "ipv4Address":
        return str(ipaddress.IPv4Address(data)) if n == 4 else bytes(data)
    if abstract == "ipv6Address":
        return str(ipaddress.IPv6Address(data)) if n == 16 else bytes(data)
    if abstract == "dateTimeSeconds":
        return epoch_integer(int.from_bytes(data, "big"), "s")
    if abstract == "dateTimeMilliseconds":
        return epoch_integer(int.from_bytes(data, "big"), "ms")
    if abstract in ("dateTimeMicroseconds", "dateTimeNanoseconds"):
        if n != 8:
            return bytes(data)
        sec, frac = struct.unpack(">II", data)
        if abstract == "dateTimeMicroseconds":
            frac &= ~0x7FF                      # the low 11 bits of a microsecond time are unused (6.1.9)
        return ntp64(sec, frac)
    if abstract in ("basicList", "subTemplateList", "subTemplateMultiList"):
        return _structured(abstract, data, decoder, ctx)
    return bytes(data)


def _structured(abstract: str, data: bytes, decoder: FlowExportDecoder | None, ctx: tuple[str, int] | None) -> Any:
    """RFC 6313 structured data, as JSON-compatible values."""
    if not data:
        return {"semantic": None, "elements": []}
    semantic = data[0]
    if abstract == "basicList":
        if len(data) < 5:
            return {"semantic": semantic, "raw": data[1:].hex()}
        ie_raw, length = struct.unpack_from(">HH", data, 1)
        pen = 0
        o = 5
        if ie_raw & 0x8000:
            pen = struct.unpack_from(">I", data, o)[0]
            o += 4
        ie = ie_raw & 0x7FFF
        elements = []
        while o < len(data):
            if length == 0xFFFF:
                ln, o = _varlen(data, o)
            else:
                ln = length
            elements.append(_jsonable(decode_value(ie, pen, data[o:o + ln])))
            o += ln
        return {"semantic": semantic, "element": ie_name(ie, pen), "elements": elements}
    if abstract == "subTemplateList":
        if len(data) < 3 or decoder is None or ctx is None:
            return {"semantic": semantic, "raw": data[1:].hex()}
        tid = struct.unpack_from(">H", data, 1)[0]
        return {"semantic": semantic, "template_id": tid, "records": decoder.sub_records(ctx, tid, data[3:])}
    out: list[Any] = []
    o = 1
    while o + 4 <= len(data) and decoder is not None and ctx is not None:
        tid, ln = struct.unpack_from(">HH", data, o)
        if ln < 4:
            break
        out.append({"template_id": tid, "records": decoder.sub_records(ctx, tid, data[o + 4:o + ln])})
        o += ln
    return {"semantic": semantic, "lists": out}


def _jsonable(v: Any) -> Any:
    if isinstance(v, Instant):
        return v.seconds
    if isinstance(v, bytes):
        return v.hex()
    return v


def _varlen(data: bytes, o: int) -> tuple[int, int]:
    """A variable-length field's length and the offset of its value (RFC 7011 section 7)."""
    if o >= len(data):
        raise MalformedRecord("truncated-varlen", f"at {o}")
    ln = data[o]
    if ln < 255:
        return ln, o + 1
    if o + 3 > len(data):
        raise MalformedRecord("truncated-varlen", f"at {o}")
    return struct.unpack_from(">H", data, o + 1)[0], o + 3


class FlowExportDecoder:
    """Stateful decoder of export datagrams for one collector (module docstring)."""

    def __init__(self, cfg: FlowExportConfig, stats: IngestStats) -> None:
        self.cfg = cfg
        self.stats = stats
        self.templates: BoundedLRU[tuple[str, int, int], Template] = BoundedLRU(cfg.max_templates, "templates", stats)
        self.pending: BoundedLRU[tuple[str, int, int], list[_Pending]] = BoundedLRU(
            max(cfg.pending_sets, 1), "pending_sets", stats, on_evict=self._evicted_pending)
        self.pending_count = 0
        self.quarantined: list[tuple[RawRecord, str, str]] = []
        self.sequence: dict[tuple[str, int, str], int] = {}
        self.active_timeout: dict[tuple[str, int], float] = {}
        self._pending_in_message = False
        self.samplers: BoundedLRU[tuple[str, int], dict[int | None, int]] = BoundedLRU(
            cfg.max_sampler_entries, "samplers", stats)
        self.boot: dict[tuple[str, int], int] = {}
        self.first_export: dict[tuple[str, int], float] = {}
        self.latest_export: dict[tuple[str, int], float] = {}

    def _evicted_pending(self, key: tuple[str, int, int], items: list[_Pending]) -> None:
        self.pending_count -= len(items)
        for p in items:
            self.quarantined.append((p.raw, "unknown-template-evicted", f"template {key[2]} of {key[0]}/{key[1]}"))

    def decode(self, raw: RawRecord) -> list[DecodedRecord]:
        data = raw.data
        if len(data) < 2:
            raise MalformedRecord("truncated-header", f"{len(data)} bytes")
        version = struct.unpack_from(">H", data, 0)[0]
        exporter = raw.exporter or "unknown"
        if version == 5:
            return self._v5(data, exporter)
        if version == 9:
            return self._v9(data, exporter, raw)
        if version == 10:
            return self._ipfix(data, exporter, raw)
        raise MalformedRecord("unsupported-version", f"export version {version}")

    def _check_sequence(self, exporter: str, domain: int, kind: str, seq: int, advance: int | None) -> None:
        """Count a gap between the expected and the received sequence number; `advance` None re-anchors
        (a message whose data sets wait for their template cannot be counted)."""
        key = (exporter, domain, kind)
        expected = self.sequence.get(key)
        if expected is not None and seq != expected:
            gap = (seq - expected) % (1 << 32)
            if gap < (1 << 31):
                self.stats.counters["sequence_gaps"] += 1
                self.stats.counters["records_lost_estimate"] += gap
            else:
                self.stats.counters["sequence_out_of_order"] += 1
        if advance is None:
            self.sequence.pop(key, None)
        else:
            self.sequence[key] = (seq + advance) % (1 << 32)

    def _v5(self, data: bytes, exporter: str) -> list[DecodedRecord]:
        if len(data) < 24:
            raise MalformedRecord("truncated-header", f"v5 header of {len(data)} bytes")
        version, count, uptime, secs, nsecs, seq, etype, eid, sampling = V5_HEADER.unpack_from(data, 0)
        if not 1 <= count <= 30:
            raise MalformedRecord("bad-count", f"v5 count {count}")
        if len(data) < 24 + 48 * count:
            raise MalformedRecord("truncated-datagram", f"{len(data)} bytes for {count} v5 records")
        self._check_sequence(exporter, eid, "v5", seq, count)
        export = Instant.from_ns(secs * NS + nsecs, 1e-9 if nsecs else 1.0)
        boot_ms = secs * 1000 + nsecs // 1_000_000 - uptime
        self._note_export(exporter, eid, export.seconds)
        header = {"exporter": exporter, "version": version, "count": count, "sys_uptime": uptime, "unix_secs": secs,
                  "unix_nsecs": nsecs, "flow_sequence": seq, "engine_type": etype, "engine_id": eid,
                  "sampling_interval": sampling}
        out = []
        for k in range(count):
            o = 24 + 48 * k
            (src, dst, nh, inp, outp, pkts, octs, first, last, sport, dport, pad1, flags, prot, tos, sas, das, smask,
             dmask, pad2) = V5_RECORD.unpack_from(data, o)
            v = dict(header)
            v.update(srcaddr=str(ipaddress.IPv4Address(src)), dstaddr=str(ipaddress.IPv4Address(dst)),
                     nexthop=str(ipaddress.IPv4Address(nh)), input=inp, output=outp, dPkts=pkts, dOctets=octs,
                     first=first, last=last, srcport=sport, dstport=dport, pad1=pad1, tcp_flags=flags, prot=prot,
                     tos=tos, src_as=sas, dst_as=das, src_mask=smask, dst_mask=dmask, pad2=pad2)
            out.append(DecodedRecord("v5", v, o, 48, export, boot_ms))
        return out

    def _v9(self, data: bytes, exporter: str, raw: RawRecord) -> list[DecodedRecord]:
        if len(data) < 20:
            raise MalformedRecord("truncated-header", f"v9 header of {len(data)} bytes")
        version, count, uptime, secs, seq, source = V9_HEADER.unpack_from(data, 0)
        self._check_sequence(exporter, source, "v9", seq, 1)
        export = epoch_integer(secs, "s")
        boot_ms = secs * 1000 - uptime
        self.boot[(exporter, source)] = boot_ms
        self._note_export(exporter, source, export.seconds)
        header = {"exporter": exporter, "version": version, "count": count, "sys_uptime": uptime, "unix_secs": secs,
                  "sequence": seq, "source_id": source}
        out: list[DecodedRecord] = []
        o = 20
        while o + 4 <= len(data):
            set_id, length = struct.unpack_from(">HH", data, o)
            if length < 4 or o + length > len(data):
                raise MalformedRecord("bad-flowset-length", f"flowset {set_id} length {length} at {o}")
            body = data[o + 4:o + length]
            if set_id == 0:
                self._v9_templates(body, exporter, source, export.seconds)
            elif set_id == 1:
                self._v9_options_templates(body, exporter, source, export.seconds)
            elif set_id >= 256:
                out.extend(self._data_set("v9", data, o, length, set_id, exporter, source, header, export, boot_ms, raw))
            else:
                self.stats.counters["reserved_flowset_ids"] += 1
            o += length
        return out

    def _v9_templates(self, body: bytes, exporter: str, source: int, t: float) -> None:
        i = 0
        while i + 4 <= len(body):
            tid, n = struct.unpack_from(">HH", body, i)
            i += 4
            if tid < 256 or i + 4 * n > len(body):
                if tid == 0 and n == 0:
                    break                                          # padding
                raise MalformedRecord("bad-template", f"v9 template {tid} with {n} fields")
            fields = tuple((struct.unpack_from(">H", body, i + 4 * k)[0], struct.unpack_from(">H", body, i + 4 * k + 2)[0], 0)
                           for k in range(n))
            i += 4 * n
            self._add_template(exporter, source, Template(tid, fields, 0, t))

    def _v9_options_templates(self, body: bytes, exporter: str, source: int, t: float) -> None:
        i = 0
        while i + 6 <= len(body):
            tid, scope_len, opt_len = struct.unpack_from(">HHH", body, i)
            i += 6
            if tid < 256 or scope_len % 4 or opt_len % 4 or i + scope_len + opt_len > len(body):
                if tid == 0:
                    break
                raise MalformedRecord("bad-options-template", f"v9 options template {tid}")
            n_scope = scope_len // 4
            n = n_scope + opt_len // 4
            fields = tuple((struct.unpack_from(">H", body, i + 4 * k)[0], struct.unpack_from(">H", body, i + 4 * k + 2)[0], 0)
                           for k in range(n))
            i += scope_len + opt_len
            # v9 scope field types (1 system, 2 interface, 3 line card, 4 cache, 5 template) are not
            # information elements: they are named "scope_<type>" when the data is decoded.
            self._add_template(exporter, source, Template(tid, fields, max(n_scope, 1), t))

    def _ipfix(self, data: bytes, exporter: str, raw: RawRecord) -> list[DecodedRecord]:
        if len(data) < 16:
            raise MalformedRecord("truncated-header", f"IPFIX header of {len(data)} bytes")
        version, length, export_s, seq, domain = IPFIX_HEADER.unpack_from(data, 0)
        if length != len(data):
            raise MalformedRecord("bad-message-length", f"header {length}, datagram {len(data)}")
        export = epoch_integer(export_s, "s")
        self._note_export(exporter, domain, export.seconds)
        header = {"exporter": exporter, "version": version, "length": length, "export_time": export_s,
                  "sequence": seq, "observation_domain_id": domain}
        out: list[DecodedRecord] = []
        self._pending_in_message = False
        o = 16
        while o + 4 <= len(data):
            set_id, slen = struct.unpack_from(">HH", data, o)
            if slen < 4 or o + slen > len(data):
                raise MalformedRecord("bad-set-length", f"set {set_id} length {slen} at {o}")
            body = data[o + 4:o + slen]
            if set_id in (2, 3):
                self._ipfix_templates(body, exporter, domain, export.seconds, options=set_id == 3)
            elif set_id >= 256:
                out.extend(self._data_set("ipfix", data, o, slen, set_id, exporter, domain, header, export,
                                          self.boot.get((exporter, domain)), raw))
            else:
                self.stats.counters["reserved_set_ids"] += 1
            o += slen
        self._check_sequence(exporter, domain, "ipfix", seq, None if self._pending_in_message else len(out))
        return out

    def _ipfix_templates(self, body: bytes, exporter: str, domain: int, t: float, *, options: bool) -> None:
        i = 0
        hdr = 6 if options else 4
        while i + 4 <= len(body):
            tid, n = struct.unpack_from(">HH", body, i)
            if n == 0:
                # Template withdrawal (RFC 7011 section 8.1): id 2 / 3 withdraws every template of the kind.
                if tid in (2, 3):
                    for key in [k for k in self.templates.keys() if k[0] == exporter and k[1] == domain]:
                        tpl = self.templates.peek(key)
                        if tpl is not None and tpl.options == (tid == 3):
                            self.templates.pop(key)
                elif tid >= 256:
                    self.templates.pop((exporter, domain, tid))
                    self.stats.counters["template_withdrawals"] += 1
                i += 4
                if tid == 0:
                    break
                continue
            if i + hdr > len(body):
                break
            scope = struct.unpack_from(">H", body, i + 4)[0] if options else 0
            i += hdr
            fields: list[tuple[int, int, int]] = []
            for _ in range(n):
                if i + 4 > len(body):
                    raise MalformedRecord("bad-template", f"IPFIX template {tid} truncated")
                ie_raw, ln = struct.unpack_from(">HH", body, i)
                i += 4
                pen = 0
                if ie_raw & 0x8000:
                    if i + 4 > len(body):
                        raise MalformedRecord("bad-template", f"IPFIX template {tid} truncated enterprise number")
                    pen = struct.unpack_from(">I", body, i)[0]
                    i += 4
                fields.append((ie_raw & 0x7FFF, ln, pen))
            if tid < 256 or (options and not 1 <= scope <= n):
                raise MalformedRecord("bad-template", f"IPFIX template {tid} (scope {scope} of {n})")
            self._add_template(exporter, domain, Template(tid, tuple(fields), scope, t))

    def _add_template(self, exporter: str, domain: int, tpl: Template) -> None:
        key = (exporter, domain, tpl.template_id)
        old = self.templates.peek(key)
        if old is not None and old.fields != tpl.fields:
            self.stats.counters["template_redefinitions"] += 1
        self.templates[key] = tpl
        self.stats.counters["templates_received"] += 1

    def _note_export(self, exporter: str, domain: int, t: float) -> None:
        key = (exporter, domain)
        self.first_export.setdefault(key, t)
        self.latest_export[key] = max(self.latest_export.get(key, t), t)

    def _template(self, exporter: str, domain: int, tid: int, now: float) -> Template | None:
        key = (exporter, domain, tid)
        tpl = self.templates.get(key)
        if tpl is not None and now - tpl.refreshed > self.cfg.template_lifetime_s:
            self.templates.pop(key)
            self.stats.counters["templates_expired"] += 1
            return None
        return tpl

    def _data_set(self, kind: str, data: bytes, set_offset: int, length: int, tid: int, exporter: str, domain: int,
                  header: dict[str, Any], export: Instant, boot_ms: int | None, raw: RawRecord) -> list[DecodedRecord]:
        tpl = self._template(exporter, domain, tid, export.seconds)
        if tpl is None:
            key = (exporter, domain, tid)
            items = self.pending.get(key) or []
            items.append(_Pending(data[set_offset:set_offset + length], set_offset, dict(header), export, boot_ms,
                                  RawRecord(data[set_offset:set_offset + length], raw.location, raw.index,
                                            offset=None if raw.offset is None else raw.offset + set_offset,
                                            exporter=exporter)))
            self.pending[key] = items
            self.pending_count += 1
            self._pending_in_message = True
            self.stats.counters["data_sets_pending"] += 1
            return []
        return self._records(kind, data, set_offset, length, tpl, header, export, boot_ms)

    def _records(self, kind: str, data: bytes, set_offset: int, length: int, tpl: Template, header: dict[str, Any],
                 export: Instant, boot_ms: int | None) -> list[DecodedRecord]:
        out = []
        o = set_offset + 4
        end = set_offset + length
        min_len = max(tpl.min_length(), 1)
        exporter = str(header.get("exporter"))
        domain = int(header.get("observation_domain_id", header.get("source_id", 0)))
        ctx = (exporter, domain)
        while end - o >= min_len:
            start = o
            values: dict[str, Any] = dict(header)
            values["template_id"] = tpl.template_id
            scope_names: list[str] = []
            try:
                for k, (ie, ln, pen) in enumerate(tpl.fields):
                    if ln == 0xFFFF:
                        ln, o = _varlen(data, o)
                    if o + ln > end:
                        raise MalformedRecord("truncated-record", f"template {tpl.template_id} field {ie}")
                    chunk = data[o:o + ln]
                    o += ln
                    if kind == "v9" and tpl.options and k < tpl.scope_count:
                        name = f"scope_{ie}"                         # v9 scope types are not elements
                        value: Any = int.from_bytes(chunk, "big") if ln <= 8 else chunk
                    else:
                        name = ie_name(ie, pen)
                        value = decode_value(ie, pen, chunk, self, ctx)
                    if tpl.options and k < tpl.scope_count:
                        scope_names.append(name)
                    if name in values and name not in header:
                        name = f"{name}#{k}"                         # a repeated element keeps every value
                    values[name] = value
            except MalformedRecord:
                raise
            except (struct.error, ValueError, IndexError) as exc:
                raise MalformedRecord("bad-record", f"template {tpl.template_id}: {exc}") from None
            if tpl.options:
                values["scope_fields"] = scope_names
                self._note_options(exporter, domain, values)
            rec_kind = f"{kind}_options" if tpl.options else kind
            out.append(DecodedRecord(rec_kind, values, start, o - start, export, boot_ms, tuple(scope_names)))
        return out

    def _note_options(self, exporter: str, domain: int, values: dict[str, Any]) -> None:
        """Options records: sampler table and exporter boot time."""
        interval = values.get("samplingInterval", values.get("samplerRandomInterval", values.get("samplingPacketInterval")))
        if isinstance(interval, int) and interval > 0:
            sid = values.get("samplerId", values.get("selectorId"))
            table = self.samplers.get((exporter, domain)) or {}
            table[sid if isinstance(sid, int) else None] = interval
            self.samplers[(exporter, domain)] = table
        boot = values.get("systemInitTimeMilliseconds")
        if isinstance(boot, Instant) and boot.ns is not None:
            self.boot[(exporter, domain)] = boot.ns // 1_000_000
        active = values.get("flowActiveTimeout")
        if isinstance(active, int) and active > 0:
            self.active_timeout[(exporter, domain)] = float(active)

    def sub_records(self, ctx: tuple[str, int], tid: int, data: bytes) -> list[dict[str, Any]]:
        """Records of template `tid` inside a subTemplateList (RFC 6313)."""
        tpl = self.templates.peek((ctx[0], ctx[1], tid))
        if tpl is None:
            return [{"undecoded": data.hex()}]
        out: list[dict[str, Any]] = []
        o = 0
        min_len = max(tpl.min_length(), 1)
        while len(data) - o >= min_len:
            rec: dict[str, Any] = {}
            for ie, ln, pen in tpl.fields:
                if ln == 0xFFFF:
                    ln, o = _varlen(data, o)
                rec[ie_name(ie, pen)] = _jsonable(decode_value(ie, pen, data[o:o + ln], self, ctx))
                o += ln
            out.append(rec)
        return out

    def resolve_pending(self, exporter: str, domain: int) -> list[tuple[_Pending, list[DecodedRecord]]]:
        """Pending sets whose template is now known, decoded."""
        out = []
        for key in [k for k in self.pending.keys() if k[0] == exporter and k[1] == domain]:
            items = self.pending.peek(key) or []
            latest = self.latest_export.get((exporter, domain), 0.0)
            tpl = self._template(exporter, domain, key[2], latest)
            keep: list[_Pending] = []
            for p in items:
                if tpl is not None:
                    kind = "ipfix" if p.header.get("version") == 10 else "v9"
                    recs = self._records(kind, p.data, 0, len(p.data), tpl, p.header, p.export_time, p.boot_ms)
                    out.append((p, recs))
                elif latest - p.export_time.seconds > self.cfg.pending_max_age_s:
                    self.quarantined.append((p.raw, "unknown-template-expired", f"template {key[2]}"))
                else:
                    keep.append(p)
            self.pending_count -= len(items) - len(keep)
            if keep:
                self.pending[key] = keep
            else:
                self.pending.pop(key)
        return out

    def drain_pending(self) -> None:
        """At the end of the source: every pending set is quarantined (its template never arrived)."""
        for key, items in self.pending.items():
            for p in items:
                self.quarantined.append((p.raw, "unknown-template", f"template {key[2]} of {key[0]}/{key[1]}"))
        self.pending.clear()
        self.pending_count = 0


class FlowExportSource(StreamAdapter):
    """NetFlow v5 / v9 and IPFIX -> state updates. See the module docstring.

    Parameters
    ----------
    source: a packet capture of export datagrams, an IPFIX file, or an iterable of `Datagram`s.
    config: `FlowExportConfig`.
    """

    name = "netflow-ipfix"
    source_type = "flow-export"
    version = ADAPTER_VERSION

    def __init__(self, source: str | Path | bytes | IO[bytes] | Iterable[Datagram], *,
                 config: FlowExportConfig | None = None, **kw: Any) -> None:
        self.config = config or FlowExportConfig()
        self.datagrams: Iterable[Datagram] | None = None
        if not isinstance(source, str | Path | bytes | bytearray) and not hasattr(source, "read"):
            self.datagrams = source                         # type: ignore[assignment]
            super().__init__(None, self.config.common, **kw)
        else:
            super().__init__(source, self.config.common, **kw)   # type: ignore[arg-type]
        self.decoder = FlowExportDecoder(self.config, self.stats)
        self.flows: BoundedLRU[tuple[Any, ...], _Accumulator] = BoundedLRU(self.config.max_flows, "flow_accumulators",
                                                                            self.stats)
        self._ctx = ConvContext()

    def raw_records(self) -> Iterator[RawRecord | tuple[str, RawRecord]]:
        if self.datagrams is not None:
            for i, dg in enumerate(self.datagrams):
                yield RawRecord(dg.data, "datagrams", i, received_time=dg.received_time, exporter=dg.exporter)
            return
        stream, location, close = open_source(self.source)        # type: ignore[arg-type]
        try:
            head = stream.read(4)
            stream.seek(0)
            if is_capture(head):
                yield from self._from_capture(stream, location)
            elif head[:2] == b"\x00\x0a":
                yield from self._ipfix_file(stream, location)
            else:
                raise MalformedRecord("unknown-input", f"neither a capture nor an IPFIX file (first bytes {head.hex()})")
        finally:
            if close:
                stream.close()

    def _from_capture(self, stream: IO[bytes], location: str) -> Iterator[RawRecord | tuple[str, RawRecord]]:
        ports = set(self.config.udp_ports)
        for item in read_capture(stream):
            if not isinstance(item, CapturedPacket):
                continue
            udp = udp_payload(item)
            if udp is None or udp[3] not in ports:
                self.stats.counters["capture_packets_skipped"] += 1
                continue
            src, _dst, _sp, _dp, payload, off = udp
            t = None if item.ts_ns is None else item.ts_ns / NS
            yield RawRecord(payload, location, item.index, offset=off, received_time=t, exporter=src)

    def _ipfix_file(self, stream: IO[bytes], location: str) -> Iterator[RawRecord | tuple[str, RawRecord]]:
        offset = 0
        index = 0
        while True:
            head = stream.read(16)
            if not head:
                return
            if len(head) < 16:
                yield ("truncated-message", RawRecord(head, location, index, offset=offset))
                return
            version, length = struct.unpack_from(">HH", head, 0)
            if version != 10 or length < 16:
                yield ("bad-message-header", RawRecord(head, location, index, offset=offset))
                return
            body = stream.read(length - 16)
            msg = head + body
            if len(msg) < length:
                yield ("truncated-message", RawRecord(msg, location, index, offset=offset))
                return
            yield RawRecord(msg, location, index, offset=offset, exporter=f"file:{location}")
            offset += length
            index += 1

    def decode(self, raw: RawRecord) -> Iterable[UpdateDraft]:
        recs = self.decoder.decode(raw)
        out = [self._draft(raw, r, k) for k, r in enumerate(recs)]
        exporter = raw.exporter or "unknown"
        version = struct.unpack_from(">H", raw.data, 0)[0]
        if version in (9, 10):
            domain = struct.unpack_from(">I", raw.data, 16 if version == 9 else 12)[0]
            for p, prs in self.decoder.resolve_pending(exporter, domain):
                for k, r in enumerate(prs):
                    out.append(self._draft(p.raw, r, k))
        for qraw, reason, detail in self.decoder.quarantined:
            self.quarantine.add(qraw, reason, detail)
        self.decoder.quarantined.clear()
        return out

    def finish(self) -> Iterable[UpdateDraft]:
        self.decoder.drain_pending()
        for qraw, reason, detail in self.decoder.quarantined:
            self.quarantine.add(qraw, reason, detail)
        self.decoder.quarantined.clear()
        return ()

    def _draft(self, raw: RawRecord, rec: DecodedRecord, k: int) -> UpdateDraft:
        start = rec.offset
        data = raw.data[start:start + rec.length]
        off = None if raw.offset is None else raw.offset + start
        sub = RawRecord(data, raw.location, raw.index, offset=off, sub_index=k, received_time=raw.received_time,
                        exporter=raw.exporter)
        record_type = {"v5": "netflow.v5", "v9": "netflow.v9", "v9_options": "netflow.v9_options",
                       "ipfix": "ipfix.data", "ipfix_options": "ipfix.options"}[rec.kind]
        d = UpdateDraft(record_type, sub, exporter=raw.exporter)
        src = record_type
        values = rec.values
        self._ctx.values = values
        mapper(record_type).apply(values, d, self._ctx, self.stats, source=src,
                                  keep_unmapped=self.config.common.keep_unmapped)
        if rec.kind.endswith("options"):
            d.time = rec.export_time
            d.add_entity(self.resolver.address(raw.exporter), "subject")
            return d
        self._times(d, rec, values, src)
        self._icmp(d, values, src)
        self._counters(d, rec, values, src)
        f, b = d.value("flow.tcp_flags_fwd"), d.value("flow.tcp_flags_bwd")
        if f is not None or b is not None:
            set_field(d, "flow.tcp_flags", (int(f or 0) | int(b or 0)) & 0x3F, src, self.stats)
        er = d.value("flow.end_reason")
        if er == 8:
            bits_f, bits_b = int(f or 0), int(b or 0)
            if (bits_f | bits_b) & 0x04:
                set_field(d, "flow.end_reason", 2, src, self.stats, overwrite=True)
            elif bits_f & 0x01 and bits_b & 0x01:
                set_field(d, "flow.end_reason", 1, src, self.stats, overwrite=True)
        if d.value("flow.sampling_rate") is None:
            table = self.decoder.samplers.get((str(raw.exporter), int(values.get("observation_domain_id",
                                                                                 values.get("source_id", 0)))))
            if table:
                sid = values.get("samplerId", values.get("selectorId"))
                rate = table.get(sid if isinstance(sid, int) else None)
                if rate is None and len(table) == 1:
                    rate = next(iter(table.values()))
                if rate is not None:
                    set_field(d, "flow.sampling_rate", rate, src, self.stats)
        DV.flow_totals(d, self.stats, src)
        DV.flow_entities(d, self.resolver)
        return d

    def _times(self, d: UpdateDraft, rec: DecodedRecord, v: dict[str, Any], src: str) -> None:
        export = rec.export_time
        boot = rec.boot_ms

        def pick(names: tuple[str, ...]) -> Instant | None:
            for n in names:
                x = v.get(n)
                if isinstance(x, Instant):
                    return x
            return None

        start = pick(("flowStartNanoseconds", "flowStartMicroseconds", "flowStartMilliseconds", "flowStartSeconds"))
        end = pick(("flowEndNanoseconds", "flowEndMicroseconds", "flowEndMilliseconds", "flowEndSeconds"))
        if start is None and isinstance(v.get("flowStartDeltaMicroseconds"), int):
            start = Instant.from_ns(export.ns - v["flowStartDeltaMicroseconds"] * 1000 if export.ns is not None else 0, 1e-6)
        if end is None and isinstance(v.get("flowEndDeltaMicroseconds"), int):
            end = Instant.from_ns(export.ns - v["flowEndDeltaMicroseconds"] * 1000 if export.ns is not None else 0, 1e-6)
        if rec.kind == "v5":
            up = int(v["sys_uptime"])
            for key, slot in (("first", "start"), ("last", "end")):
                delta_ms = (up - int(v[key])) % (1 << 32)
                assert export.ns is not None
                inst = Instant.from_ns(export.ns - delta_ms * 1_000_000, 1e-3)
                if slot == "start":
                    start = inst
                else:
                    end = inst
        elif (start is None or end is None) and boot is not None:
            for key, slot in (("flowStartSysUpTime", "start"), ("flowEndSysUpTime", "end")):
                ms = v.get(key)
                if isinstance(ms, int):
                    inst = Instant.from_ns((boot + ms) * 1_000_000, 1e-3)
                    if slot == "start" and start is None:
                        start = inst
                    elif slot == "end" and end is None:
                        end = inst
        if start is not None:
            set_field(d, "flow.start_time", start.seconds, src, self.stats)
        if end is not None:
            set_field(d, "flow.end_time", end.seconds, src, self.stats)
        if start is not None and end is not None:
            DV.times_and_duration(d, self.stats, src)
        else:
            for key, scale in (("flowDurationMicroseconds", 1e-6), ("flowDurationMilliseconds", 1e-3)):
                if isinstance(v.get(key), int):
                    set_field(d, "flow.duration", v[key] * scale, src, self.stats)
                    break
        d.time = end if end is not None else export

    def _icmp(self, d: UpdateDraft, v: dict[str, Any], src: str) -> None:
        proto = d.value("flow.protocol")
        if proto not in (1, 58):
            return
        code = v.get("icmpTypeCodeIPv4", v.get("icmpTypeCodeIPv6"))
        if not isinstance(code, int):
            port = v.get("destinationTransportPort", v.get("dstport"))
            code = port if isinstance(port, int) else None       # exporters put type * 256 + code there (AS-708)
        if isinstance(code, int):
            set_field(d, "proto.icmp.type", code >> 8, src, self.stats)
            set_field(d, "proto.icmp.code", code & 0xFF, src, self.stats)
        for fid in ("flow.src_port", "flow.dst_port"):
            d.fields.pop(fid, None)

    def _counters(self, d: UpdateDraft, rec: DecodedRecord, v: dict[str, Any], src: str) -> None:
        """Running totals from total counters, or from delta counters accumulated per flow (AS-696).

        A flow is exported again every active timeout T_a while it lasts. Collection from an exporter
        and domain began at C (its first export seen); a flow that started after C - T_a can have no
        export before C, so the sum of its deltas is its total (OBSERVED). Otherwise the sum covers the
        share (end - (C - T_a)) / (end - start) of its lifetime at least: LOW_RELIABILITY with that
        reliability (floor MIN_COMPLETENESS). T_a is flowActiveTimeout when the exporter reports it.
        """
        for name, target in _TOTAL_TARGETS:
            if isinstance(v.get(name), int):
                set_field(d, target, v[name], src, self.stats)
        deltas = [(name, target) for name, target in _DELTA_TARGETS if isinstance(v.get(name), int)]
        if not deltas:
            return
        exporter = str(v.get("exporter"))
        domain = int(v.get("observation_domain_id", v.get("source_id", v.get("engine_id", 0))))
        key = (exporter, domain, d.value("flow.src_ip"), d.value("flow.dst_ip"), d.value("flow.src_port"),
               d.value("flow.dst_port"), d.value("flow.protocol"), d.value("flow.start_time"))
        if isinstance(v.get("flowActiveTimeout"), int) and v["flowActiveTimeout"] > 0:
            self.decoder.active_timeout[(exporter, domain)] = float(v["flowActiveTimeout"])
        t_a = self.decoder.active_timeout.get((exporter, domain), DEFAULT_ACTIVE_TIMEOUT_S)
        acc = self.flows.get(key)
        if acc is None:
            evicted_before = self.flows.evictions > 0
            acc = _Accumulator(created=self.decoder.first_export.get((exporter, domain), rec.export_time.seconds))
            if evicted_before:
                # An evicted flow that reappears has lost its earlier deltas: its history starts now.
                acc.created = max(acc.created, rec.export_time.seconds)
            self.flows[key] = acc
        for name, target in deltas:
            acc.totals[target] = acc.totals.get(target, 0) + int(v[name])
        start, end = d.value("flow.start_time"), d.value("flow.end_time")
        horizon = acc.created - t_a
        status: ObservationStatus = ObservationStatus.OBSERVED
        rel: float | None = None
        if start is None:
            status, rel = LOW, UNKNOWN_COMPLETENESS
        elif float(start) <= horizon:
            span = (float(end) - float(start)) if end is not None else 0.0
            share = (float(end) - horizon) / span if span > 0 else 0.0
            status, rel = LOW, max(MIN_COMPLETENESS, min(share, 1.0))
        for target, total in acc.totals.items():
            set_field(d, target, total, src, self.stats, status=status, reliability=rel)
        if v.get("flowEndReason") in _FLOW_ENDED or d.value("flow.end_reason") in (1, 2, 3, 6, 7, 8):
            self.flows.pop(key)


def iter_datagrams(items: Iterable[tuple[bytes, str] | tuple[bytes, str, float]]) -> Iterator[Datagram]:
    """Datagrams from (bytes, exporter[, received time]) tuples."""
    for it in items:
        yield Datagram(it[0], it[1], it[2] if len(it) > 2 else None)        # type: ignore[misc]


__all__ = ["ADAPTER_VERSION", "Datagram", "DecodedRecord", "FlowExportDecoder", "FlowExportSource", "Template",
           "decode_value", "ie_name", "iter_datagrams"]
