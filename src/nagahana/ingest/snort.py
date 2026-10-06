"""Snort adapter: Snort 3 alert_json, Snort 2 unified2, alert_fast and alert_full -> state updates.

Formats (tables in datamodel/maps/snort.py)
-------------------------------------------
alert_json   one JSON object per line (Snort 3 module alert_json).
unified2     binary records: a header of record type (u32, big-endian) and length (u32), then the body
             (Snort 2.9 src/sfutil/Unified2_common.h):
                 7 / 72      IDS event v1, IPv4 / IPv6         (52 / 76 bytes)
                 104 / 105   IDS event v2, IPv4 / IPv6         (60 / 84 bytes: + MPLS label, VLAN id, pad)
                 111 / 112   IDS event v2 with an application name of 16 bytes (76 / 100 bytes)
                 2           packet: sensor_id, event_id, event_second, packet_second,
                             packet_microsecond, linktype, packet_length, then the packet bytes
                 110         extra data: event_type, event_length, then sensor_id, event_id,
                             event_second, type, data_type, blob_length, data (blob_length - 8 bytes)
             Packets and extra data that follow an event with the same sensor, event id and second are
             folded into that event's state update (the first packet is decoded for packet-level
             fields; the rest are listed by offset and SHA-256). A record of another type is
             quarantined as "unsupported-unified2-type" with its bytes kept.
alert_fast   TIME [**] [gid:sid:rev] MSG [**] [Classification: C] [Priority: P] {PROTO} SRC -> DST
alert_full   a block of lines ended by a blank line: header, classification, time and endpoints, then
             the IP and transport header lines and optional cross-references.

Snort's text times carry no year unless Snort ran with -y; alert_json's "seconds" (UTC epoch) gives the
year and the integer second, the text the microseconds. Text times are local unless Snort ran with -U:
the configured UTC offset applies (AS-698).
"""

from __future__ import annotations

import re
import struct
import time
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import IO, Any

from nagahana.datamodel.records import FieldValue
from nagahana.datamodel.status import ObservationStatus
from nagahana.ingest import derive as DV
from nagahana.ingest.config import SnortConfig
from nagahana.ingest.convert import ConvContext
from nagahana.ingest.core import (
    MalformedRecord,
    RawRecord,
    StreamAdapter,
    UpdateDraft,
    iter_lines,
    json_loads_exact,
    open_source,
)
from nagahana.ingest.mapping import mapper, set_field
from nagahana.ingest.packets import apply_packet, decode_packet
from nagahana.ingest.timeparse import NS, Instant, TimeParseError, epoch_integer, snort

ADAPTER_VERSION = "1.0.0"
FORMATS: tuple[str, ...] = ("alert_json", "unified2", "alert_fast", "alert_full")

U2_PACKET, U2_EVENT_V1, U2_EVENT_V1_6, U2_EVENT_V2, U2_EVENT_V2_6 = 2, 7, 72, 104, 105
U2_EXTRA, U2_EVENT_APPID, U2_EVENT_APPID_6 = 110, 111, 112
#: Event body sizes (bytes) by record type.
U2_EVENT_SIZES: dict[int, int] = {U2_EVENT_V1: 52, U2_EVENT_V1_6: 76, U2_EVENT_V2: 60, U2_EVENT_V2_6: 84,
                                  U2_EVENT_APPID: 76, U2_EVENT_APPID_6: 100}
U2_TYPES = frozenset({U2_PACKET, U2_EXTRA, *U2_EVENT_SIZES})
#: Extra-data `type` -> row of snort.unified2_extra that receives the data (Unified2_common.h EVENT_INFO_*).
U2_EXTRA_ROWS: dict[int, str] = {
    1: "xff_ipv4", 2: "xff_ipv6", 3: "reviewed_by", 4: "gzip_data", 5: "smtp_filename", 6: "smtp_mailfrom",
    7: "smtp_rcptto", 8: "smtp_headers", 9: "http_uri", 10: "http_hostname", 11: "ipv6_source",
    12: "ipv6_destination", 13: "jsnorm_data",
}
_U2_MAX_BODY = 1 << 24

_FAST = re.compile(
    r"^\s*(?P<ts>\d{2}/\d{2}(?:/\d{2})?-\d{2}:\d{2}:\d{2}(?:\.\d+)?)\s+"
    r"(?:\[(?P<action>[A-Za-z_]+)\]\s+)?\[\*\*\]\s+\[(?P<gid>\d+):(?P<sid>\d+):(?P<rev>\d+)\]\s+"
    r"(?P<msg>.*?)\s+\[\*\*\]\s*"
    r"(?:\[Classification:\s*(?P<cls>[^\]]*)\]\s*)?(?:\[Priority:\s*(?P<pri>\d+)\]\s*)?"
    r"(?:\[[^\]]*\]\s*)*"
    r"\{(?P<proto>[^}]+)\}\s+(?P<src>\S+)\s+->\s+(?P<dst>\S+)\s*$"
)
_FULL_HEAD = re.compile(r"^\[\*\*\]\s+\[(?P<gid>\d+):(?P<sid>\d+):(?P<rev>\d+)\]\s+(?P<msg>.*?)\s+\[\*\*\]\s*$")
_FULL_CLASS = re.compile(r"(?:\[Classification:\s*(?P<cls>[^\]]*)\]\s*)?(?:\[Priority:\s*(?P<pri>\d+)\])?")
_FULL_ENDS = re.compile(r"^(?P<ts>\d{2}/\d{2}(?:/\d{2})?-\d{2}:\d{2}:\d{2}(?:\.\d+)?)\s+(?P<src>\S+)\s+->\s+(?P<dst>\S+)")
_FULL_IP = re.compile(r"^(?P<proto>[A-Z0-9-]+)\s+TTL:(?P<ttl>\d+)\s+TOS:(?P<tos>0x[0-9A-Fa-f]+)\s+ID:(?P<id>\d+)\s+"
                      r"IpLen:(?P<iplen>\d+)\s+DgmLen:(?P<dgm>\d+)(?P<flags>.*)$")
_FULL_TCP = re.compile(r"^(?P<flags>[*12UAPRSFCE]{8})\s+Seq:\s*(?P<seq>0x[0-9A-Fa-f]+)\s+Ack:\s*(?P<ack>0x[0-9A-Fa-f]+)\s+"
                       r"Win:\s*(?P<win>0x[0-9A-Fa-f]+)\s+TcpLen:\s*(?P<len>\d+)")
_FULL_UDP = re.compile(r"^Len:\s*(?P<len>\d+)")
_FULL_ICMP = re.compile(r"^Type:\s*(?P<type>\d+)\s+Code:\s*(?P<code>\d+)(?:\s+ID:\s*(?P<id>\d+))?(?:\s+Seq:\s*(?P<seq>\d+))?")


def split_endpoint(text: str, *, has_port: bool) -> tuple[str, int | None]:
    """"addr:port", "[v6]:port", "v6:port" or a bare address -> (address, port or None).

    Snort prints an IPv6 endpoint without brackets, so "2001:db8::1:80" is ambiguous; the protocol
    decides: for TCP, UDP and SCTP the last group is the port, otherwise there is none.
    """
    t = text.strip()
    if t.startswith("["):
        addr, _, rest = t[1:].partition("]")
        return addr, int(rest[1:]) if rest.startswith(":") and rest[1:].isdigit() else None
    if has_port:
        addr, sep, port = t.rpartition(":")
        if sep and port.isdigit():
            return addr, int(port)
    return t, None


def detect_format(head: bytes) -> str:
    """The Snort output format of a file from its first bytes."""
    if len(head) >= 8:
        typ, length = struct.unpack(">II", head[:8])
        if typ in U2_TYPES and 0 < length < _U2_MAX_BODY:
            return "unified2"
    text = head.lstrip()
    if text.startswith(b"{"):
        return "alert_json"
    if text.startswith(b"[**]"):
        return "alert_full"
    return "alert_fast"


class SnortSource(StreamAdapter):
    """Snort output -> state updates. See the module docstring.

    Parameters
    ----------
    source: a Snort output file, bytes or a binary stream.
    format: one of `FORMATS`; None detects it from the first bytes.
    config: `SnortConfig`.
    """

    name = "snort"
    source_type = "snort"
    version = ADAPTER_VERSION

    def __init__(self, source: str | Path | bytes | IO[bytes], *, format: str | None = None,
                 config: SnortConfig | None = None, **kw: Any) -> None:
        self.config = config or SnortConfig()
        super().__init__(source, self.config.common, **kw)
        if format is not None and format not in FORMATS:
            raise MalformedRecord("unknown-format", f"{format!r}; known: {FORMATS}")
        self.format = format
        clock = self.config.common.clock
        self._ctx = ConvContext(year=clock.assumed_year,
                                utc_offset_s=None if clock.utc_offset_hours is None else clock.utc_offset_hours * 3600.0)
        self._pending: UpdateDraft | None = None
        self._pending_key: tuple[int, int, int] | None = None
        self._pending_packets: list[dict[str, Any]] = []

    def _detect(self, stream: IO[bytes]) -> str:
        if self.format is not None:
            return self.format
        pos = stream.tell()
        head = stream.read(64)
        stream.seek(pos)
        self.format = detect_format(head)
        return self.format

    def raw_records(self) -> Iterator[RawRecord | tuple[str, RawRecord]]:
        stream, location, close = open_source(self.source)
        try:
            fmt = self._detect(stream)
            if fmt == "unified2":
                yield from self._u2_records(stream, location)
            elif fmt == "alert_full":
                yield from self._blocks(stream, location)
            else:
                for item in iter_lines(stream, location, max_bytes=self.config.common.max_record_bytes):
                    if isinstance(item, tuple) or item.data.strip():
                        yield item
        finally:
            if close:
                stream.close()

    def _u2_records(self, stream: IO[bytes], location: str) -> Iterator[RawRecord | tuple[str, RawRecord]]:
        offset = 0
        index = 0
        while True:
            head = stream.read(8)
            if not head:
                return
            if len(head) < 8:
                yield ("truncated-record-header", RawRecord(head, location, index, offset=offset))
                return
            typ, length = struct.unpack(">II", head)
            if length > min(_U2_MAX_BODY, self.config.common.max_record_bytes):
                yield ("oversize", RawRecord(head, location, index, offset=offset, meta={"type": typ}))
                return                                    # the framing cannot be trusted past this point
            body = stream.read(length)
            rec = RawRecord(head + body, location, index, offset=offset, meta={"type": typ})
            if len(body) < length:
                yield ("truncated-record", rec)
                return
            yield rec
            offset += 8 + length
            index += 1

    def _blocks(self, stream: IO[bytes], location: str) -> Iterator[RawRecord | tuple[str, RawRecord]]:
        """alert_full: records are blocks of lines ended by a blank line."""
        lines: list[RawRecord] = []
        index = 0
        for item in iter_lines(stream, location, max_bytes=self.config.common.max_record_bytes):
            if isinstance(item, tuple):
                yield item
                continue
            if item.data.strip():
                lines.append(item)
                continue
            if lines:
                yield _join_block(lines, location, index)
                index += 1
                lines = []
        if lines:
            yield _join_block(lines, location, index)

    def decode(self, raw: RawRecord) -> Iterable[UpdateDraft]:
        fmt = self.format or "alert_fast"
        if fmt == "unified2":
            return self._u2(raw)
        if fmt == "alert_json":
            return [self._json(raw)]
        if fmt == "alert_full":
            return [self._full(raw)]
        return [self._fast(raw)]

    def finish(self) -> Iterable[UpdateDraft]:
        if self._pending is not None:
            yield self._close_pending()

    # alert_json
    def _json(self, raw: RawRecord) -> UpdateDraft:
        try:
            obj = json_loads_exact(raw.data)
        except (ValueError, UnicodeDecodeError) as exc:
            raise MalformedRecord("json-decode", str(exc)) from None
        if not isinstance(obj, dict):
            raise MalformedRecord("json-not-object", type(obj).__name__)
        d = UpdateDraft("snort.alert_json", raw)
        src = "snort.alert_json"
        values = dict(obj)
        rule = values.get("rule")
        if isinstance(rule, str) and rule.count(":") == 2:
            g, s, r = rule.split(":")
            for k, v in (("gid", g), ("sid", s), ("rev", r)):
                if k not in values and v.isdigit():
                    values[k] = int(v)
        self._ctx.values = values
        self._ctx.ipv6 = ":" in str(values.get("src_addr", ""))
        mapper("snort.alert_json").apply(values, d, self._ctx, self.stats, source=src,
                                         keep_unmapped=self.config.common.keep_unmapped)
        secs, text = values.get("seconds"), values.get("timestamp")
        if isinstance(secs, int):
            d.time = _seconds_with_fraction(secs, text if isinstance(text, str) else None, self._ctx.utc_offset_s)
            d.original_time = text if isinstance(text, str) else str(secs)
        elif isinstance(text, str):
            year = self._ctx.year
            try:
                d.time = snort(text, year=year, offset_s=self._ctx.utc_offset_s)
                d.original_time = text
            except TimeParseError as exc:
                self.stats.refuse("@time", str(exc))
        self._single_packet(d, values.get("ip_len"), src)
        flags = d.value("flow.tcp_flags_fwd")
        if flags is not None:
            set_field(d, "flow.tcp_flags", int(flags) & 0x3F, src, self.stats)
        DV.alert_identity(d, self.stats, src)
        if d.value("proto.icmp.type") is not None:
            for fid in ("flow.src_port", "flow.dst_port"):
                d.fields.pop(fid, None)
        DV.flow_entities(d, self.resolver)
        return d

    def _single_packet(self, d: UpdateDraft, ip_len: Any, src: str) -> None:
        ttl = d.value("pkt.ttl_mean")
        if ttl is not None:
            set_field(d, "pkt.ttl_min", int(ttl), src, self.stats)
            set_field(d, "pkt.ttl_max", int(ttl), src, self.stats)
        if isinstance(ip_len, int) and ip_len >= 0:
            set_field(d, "pkt.ip_len_min", ip_len, src, self.stats)
            set_field(d, "pkt.ip_len_max", ip_len, src, self.stats)
            set_field(d, "flow.bytes_fwd", ip_len, src, self.stats)
            set_field(d, "flow.packets_fwd", 1, src, self.stats)

    # unified2
    def _u2(self, raw: RawRecord) -> list[UpdateDraft]:
        typ = int((raw.meta or {}).get("type", -1))
        body = raw.data[8:]
        out: list[UpdateDraft] = []
        if typ in U2_EVENT_SIZES:
            if self._pending is not None:
                out.append(self._close_pending())
            self._event(raw, typ, body)
        elif typ == U2_PACKET:
            out.extend(self._packet(raw, body))
        elif typ == U2_EXTRA:
            out.extend(self._extra(raw, body))
        else:
            raise MalformedRecord("unsupported-unified2-type", f"record type {typ}")
        return out

    def _event(self, raw: RawRecord, typ: int, body: bytes) -> None:
        need = U2_EVENT_SIZES[typ]
        if len(body) != need:
            raise MalformedRecord("bad-event-length", f"type {typ}: {len(body)} bytes, expected {need}")
        v6 = typ in (U2_EVENT_V1_6, U2_EVENT_V2_6, U2_EVENT_APPID_6)
        alen = 16 if v6 else 4
        (sensor, event_id, sec, usec, sig, gen, rev, cls, pri) = struct.unpack_from(">9I", body, 0)
        o = 36
        src_ip, dst_ip = body[o:o + alen], body[o + alen:o + 2 * alen]
        o += 2 * alen
        sport, dport, proto, impact_flag, impact, blocked = struct.unpack_from(">HHBBBB", body, o)
        o += 8
        values: dict[str, Any] = {
            "sensor_id": sensor, "event_id": event_id, "event_second": sec, "event_microsecond": usec,
            "signature_id": sig, "generator_id": gen, "signature_revision": rev, "classification_id": cls,
            "priority_id": pri, "ip_source": _ip(src_ip), "ip_destination": _ip(dst_ip), "protocol": proto,
            "impact_flag": impact_flag, "impact": impact, "blocked": blocked,
        }
        icmp = proto in (1, 58)
        if not icmp:
            values["sport_itype"], values["dport_icode"] = sport, dport
        if typ in (U2_EVENT_V2, U2_EVENT_V2_6, U2_EVENT_APPID, U2_EVENT_APPID_6):
            mpls, vlan, pad = struct.unpack_from(">IHH", body, o)
            o += 8
            values.update(mpls_label=mpls, vlanId=vlan, pad2=pad)
            if vlan == 0:
                del values["vlanId"]                      # 0: the packet had no VLAN tag
        d = UpdateDraft("snort.unified2_event", raw)
        src = "snort.unified2_event"
        self._ctx.values = values
        mapper("snort.unified2_event").apply(values, d, self._ctx, self.stats, source=src,
                                             keep_unmapped=self.config.common.keep_unmapped)
        d.time = Instant.from_ns(sec * NS + usec * 1000, 1e-6)
        d.original_time = f"{sec}.{usec:06d}"
        if icmp:
            set_field(d, "proto.icmp.type", sport, src, self.stats)
            set_field(d, "proto.icmp.code", dport, src, self.stats)
        if typ in (U2_EVENT_APPID, U2_EVENT_APPID_6):
            app = body[o:o + 16].split(b"\x00", 1)[0].decode("ascii", "replace")
            if app:
                d.attributes["snort.unified2_event.app_name"] = FieldValue(
                    "snort.unified2_event.app_name", app, ObservationStatus.OBSERVED, src)
        DV.alert_identity(d, self.stats, src)
        DV.flow_entities(d, self.resolver)
        self._pending, self._pending_key, self._pending_packets = d, (sensor, event_id, sec), []

    def _packet(self, raw: RawRecord, body: bytes) -> list[UpdateDraft]:
        if len(body) < 28:
            raise MalformedRecord("bad-packet-length", f"{len(body)} bytes")
        sensor, event_id, ev_sec, psec, pusec, link, plen = struct.unpack_from(">7I", body, 0)
        data = body[28:28 + plen]
        if len(data) != plen:
            raise MalformedRecord("bad-packet-length", f"packet_length {plen}, {len(body) - 28} bytes present")
        info = {"offset": raw.offset, "sha256": raw.sha256(), "packet_second": psec, "packet_microsecond": pusec,
                "linktype": link, "packet_length": plen}
        if self._pending is not None and self._pending_key == (sensor, event_id, ev_sec):
            if not self._pending_packets:
                apply_packet(self._pending, decode_packet(data, link), self.stats, "snort.unified2_packet",
                             addresses=False)
            if len(self._pending_packets) < self.config.pending_packets:
                self._pending_packets.append(info)
            else:
                self.stats.counters["unified2_packets_not_listed"] += 1
            return []
        # A packet without its event: its own state update.
        d = UpdateDraft("snort.unified2_packet", raw)
        src = "snort.unified2_packet"
        values = {"sensor_id": sensor, "event_id": event_id, "event_second": ev_sec, "packet_second": psec,
                  "packet_microsecond": pusec, "linktype": link, "packet_length": plen, "packet_data": data}
        self._ctx.values = values
        mapper("snort.unified2_packet").apply(values, d, self._ctx, self.stats, source=src,
                                              keep_unmapped=self.config.common.keep_unmapped)
        d.time = Instant.from_ns(psec * NS + pusec * 1000, 1e-6)
        apply_packet(d, decode_packet(data, link), self.stats, src)
        DV.flow_entities(d, self.resolver)
        self.stats.counters["unified2_orphan_packets"] += 1
        return [d]

    def _extra(self, raw: RawRecord, body: bytes) -> list[UpdateDraft]:
        if len(body) < 32:
            raise MalformedRecord("bad-extra-length", f"{len(body)} bytes")
        ev_type, ev_len, sensor, event_id, ev_sec, typ, data_type, blob_len = struct.unpack_from(">8I", body, 0)
        data = body[32:32 + max(blob_len - 8, 0)]
        if blob_len < 8 or len(data) != blob_len - 8:
            raise MalformedRecord("bad-extra-length", f"blob_length {blob_len}, {len(body) - 32} data bytes")
        values: dict[str, Any] = {"event_type": ev_type, "event_length": ev_len, "sensor_id": sensor,
                                  "event_id": event_id, "event_second": ev_sec, "type": typ, "data_type": data_type,
                                  "blob_length": blob_len}
        row = U2_EXTRA_ROWS.get(typ)
        if row is None:
            values[f"type_{typ}"] = data
        elif row in ("xff_ipv4", "ipv6_source", "ipv6_destination", "xff_ipv6"):
            values[row] = _ip(data) if len(data) in (4, 16) else data.decode("ascii", "replace")
        elif row == "gzip_data":
            values[row] = data
        else:
            values[row] = data.decode("utf-8", "backslashreplace")
        target = self._pending if (self._pending is not None and self._pending_key == (sensor, event_id, ev_sec)) else None
        d = target or UpdateDraft("snort.unified2_extra", raw)
        self._ctx.values = values
        mapper("snort.unified2_extra").apply(values, d, self._ctx, self.stats, source="snort.unified2_extra",
                                             keep_unmapped=self.config.common.keep_unmapped)
        if target is not None:
            return []
        d.time = epoch_integer(ev_sec, "s")
        if row in ("xff_ipv4", "xff_ipv6"):
            d.add_entity(self.resolver.address(values[row] if isinstance(values[row], str) else None), "initiator")
        self.stats.counters["unified2_orphan_extra"] += 1
        return [d]

    def _close_pending(self) -> UpdateDraft:
        d = self._pending
        assert d is not None
        if self._pending_packets:
            key = "snort.unified2_event.packet_records"
            d.attributes[key] = FieldValue(key, list(self._pending_packets), ObservationStatus.OBSERVED,
                                           "snort.unified2_packet")
        self._pending, self._pending_key, self._pending_packets = None, None, []
        return d

    # alert_fast and alert_full
    def _text_time(self, d: UpdateDraft, text: str) -> None:
        try:
            d.time = snort(text, year=self._ctx.year, offset_s=self._ctx.utc_offset_s)
            d.original_time = text
        except TimeParseError as exc:
            self.stats.refuse("@time", str(exc))

    def _fast(self, raw: RawRecord) -> UpdateDraft:
        line = raw.data.decode("utf-8", "backslashreplace")
        m = _FAST.match(line)
        if m is None:
            raise MalformedRecord("alert-fast-syntax", line[:200])
        return self._text_alert("snort.alert_fast", raw, m.groupdict(), {})

    def _text_alert(self, record_type: str, raw: RawRecord, g: dict[str, Any], extra: dict[str, Any]) -> UpdateDraft:
        d = UpdateDraft(record_type, raw)
        has_port = str(g["proto"]).strip().upper() in ("TCP", "UDP", "SCTP")
        src_addr, sport = split_endpoint(g["src"], has_port=has_port)
        dst_addr, dport = split_endpoint(g["dst"], has_port=has_port)
        msg = g["msg"].strip()
        if len(msg) >= 2 and msg[0] == msg[-1] == '"':
            msg = msg[1:-1]                                   # Snort 3 quotes the message
        values: dict[str, Any] = {
            "gid": int(g["gid"]), "sid": int(g["sid"]), "rev": int(g["rev"]), "msg": msg, "proto": g["proto"],
            "src_addr": src_addr, "dst_addr": dst_addr,
        }
        if g.get("action"):
            values["action"] = g["action"]
        if g.get("cls"):
            values["classification"] = g["cls"].strip()
        if g.get("pri"):
            values["priority"] = int(g["pri"])
        if sport is not None:
            values["src_port"] = sport
        if dport is not None:
            values["dst_port"] = dport
        values.update(extra)
        self._ctx.values = values
        self._ctx.ipv6 = ":" in src_addr
        mapper(record_type).apply(values, d, self._ctx, self.stats, source=record_type,
                                  keep_unmapped=self.config.common.keep_unmapped)
        self._text_time(d, g["ts"])
        DV.alert_identity(d, self.stats, record_type)
        DV.flow_entities(d, self.resolver)
        return d

    def _full(self, raw: RawRecord) -> UpdateDraft:
        lines = raw.data.decode("utf-8", "backslashreplace").split("\n")
        head = _FULL_HEAD.match(lines[0].strip()) if lines else None
        if head is None or len(lines) < 3:
            raise MalformedRecord("alert-full-syntax", lines[0][:200] if lines else "")
        g: dict[str, Any] = dict(head.groupdict())
        k = 1
        cm = _FULL_CLASS.match(lines[1].strip())
        if cm is not None and (cm.group("cls") or cm.group("pri")):
            g.update({"cls": cm.group("cls"), "pri": cm.group("pri")})
            k = 2
        ends = _FULL_ENDS.match(lines[k].strip()) if k < len(lines) else None
        if ends is None:
            raise MalformedRecord("alert-full-syntax", "no time and endpoint line")
        g.update(ends.groupdict())
        extra: dict[str, Any] = {}
        xrefs: list[str] = []
        proto = "IP"
        for line in lines[k + 1:]:
            t = line.strip()
            if (m := _FULL_IP.match(t)) is not None:
                proto = m.group("proto")
                extra.update(ttl=int(m.group("ttl")), tos=m.group("tos"), ip_id=int(m.group("id")),
                             ip_len=int(m.group("iplen")), dgm_len=int(m.group("dgm")), ip_flags=m.group("flags").strip())
            elif (m := _FULL_TCP.match(t)) is not None:
                extra.update(tcp_flags=m.group("flags"), tcp_seq=m.group("seq"), tcp_ack=m.group("ack"),
                             tcp_win=m.group("win"), tcp_len=int(m.group("len")))
            elif (m := _FULL_ICMP.match(t)) is not None:
                extra.update(icmp_type=int(m.group("type")), icmp_code=int(m.group("code")))
                if m.group("id"):
                    extra["icmp_id"] = int(m.group("id"))
                if m.group("seq"):
                    extra["icmp_seq"] = int(m.group("seq"))
            elif (m := _FULL_UDP.match(t)) is not None:
                extra["udp_len"] = int(m.group("len"))
            elif t.startswith("TCP Options"):
                extra["tcp_options"] = t
            elif t.startswith("[Xref =>"):
                xrefs.extend(x.strip() for x in re.findall(r"\[Xref =>\s*([^\]]+)\]", t))
        if xrefs:
            extra["xref"] = xrefs
        g["proto"] = proto
        d = self._text_alert("snort.alert_full", raw, g, extra)
        src = "snort.alert_full"
        self._single_packet(d, extra.get("dgm_len"), src)
        flags = extra.get("ip_flags", "")
        if "dgm_len" in extra:
            set_field(d, "pkt.ip_df_count", 1 if "DF" in flags.split() else 0, src, self.stats)
            set_field(d, "pkt.ip_mf_count", 1 if "MF" in flags.split() else 0, src, self.stats)
        tf = d.value("flow.tcp_flags_fwd")
        if tf is not None:
            set_field(d, "flow.tcp_flags", int(tf) & 0x3F, src, self.stats)
        return d


def _join_block(lines: list[RawRecord], location: str, index: int) -> RawRecord:
    first = lines[0]
    data = b"\n".join(x.data for x in lines)
    return RawRecord(data, location, index, offset=first.offset, line=first.line)


def _ip(b: bytes) -> str:
    import ipaddress

    return str(ipaddress.ip_address(b))


def _seconds_with_fraction(seconds: int, text: str | None, offset_s: float | None) -> Instant:
    """Epoch seconds from alert_json "seconds", with the microseconds of its "timestamp" text.

    The text's year comes from `seconds` (UTC); its clock may be local, so only the fraction is taken,
    and only when the text's minutes and seconds agree with `seconds` (zones differ by whole quarter
    hours at most, which leaves seconds unchanged).
    """
    base = seconds * NS
    if text:
        year = time_year(seconds)
        try:
            t = snort(text, year=year, offset_s=offset_s)
        except TimeParseError:
            return Instant.from_ns(base, 1.0)
        assert t.ns is not None
        frac = t.ns % NS
        if (t.ns // NS) % 60 == seconds % 60:
            return Instant.from_ns(base + frac, t.resolution)
    return Instant.from_ns(base, 1.0)


def time_year(seconds: int) -> int:
    """UTC calendar year of an epoch second."""
    return time.gmtime(seconds).tm_year


__all__ = ["ADAPTER_VERSION", "FORMATS", "SnortSource", "detect_format", "split_endpoint"]
