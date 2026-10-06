"""sFlow v5 adapter: datagrams -> one state update per flow sample and per counter sample.

Layouts follow "sFlow Version 5" (sflow.org, July 2004), all big-endian 32-bit words (XDR):

    datagram        version 5, agent address (type 1 IPv4 / 2 IPv6, address), sub-agent id, sequence
                    number, uptime (ms), number of samples, samples
    sample          data format (enterprise << 12 | format), length, data
    flow sample (1) sequence, source id (type << 24 | index), sampling rate, sample pool, drops, input,
                    output, number of flow records, flow records
    expanded (3)    sequence, source id type, source id index, sampling rate, sample pool, drops, input
                    format and value, output format and value, flow records
    counter (2)     sequence, source id, number of counter records, counter records
    expanded (4)    sequence, source id type, source id index, counter records
    flow records    1 sampled header, 2 Ethernet frame, 3 IPv4, 4 IPv6, 1001 extended switch, 1002
                    extended router, 1003 extended gateway, 1004 extended user, 1005 extended URL
    counter records 1 generic interface, 2 Ethernet interface, 5 VLAN, 1001 processor

Records of other formats are retained as attributes with their bytes (hex). A datagram carries the
agent's uptime but no absolute time: the event time is the time the collector received it (the
capture time of the datagram, or the `received_time` of a live `Datagram`).

Sampling (AS-697): a flow sample stands for `sampling_rate` packets. flow.packets_fwd = sampling_rate and
flow.bytes_fwd = IP length x sampling_rate are estimates, LOW_RELIABILITY with reliability
1 / (1 + 1.96): with c samples the estimate's relative error is within 196 x sqrt(1/c) percent at 95 %
(sflow.org, "Packet Sampling Basics"), and one sample is c = 1. The sampled packet's own values (its
TTL, flags, ports) are exact (OBSERVED).
"""

from __future__ import annotations

import ipaddress
import struct
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any

from nagahana.datamodel.status import ObservationStatus
from nagahana.ingest import derive as DV
from nagahana.ingest.capture import CapturedPacket, is_capture, read_capture, udp_payload
from nagahana.ingest.config import SFlowConfig
from nagahana.ingest.convert import ConvContext
from nagahana.ingest.core import BoundedLRU, MalformedRecord, RawRecord, StreamAdapter, UpdateDraft, open_source
from nagahana.ingest.flowexport import Datagram
from nagahana.ingest.mapping import mapper, set_field
from nagahana.ingest.packets import SFLOW_HEADER_LINK, apply_packet, decode_packet
from nagahana.ingest.timeparse import NS, Instant

ADAPTER_VERSION = "1.0.0"
LOW = ObservationStatus.LOW_RELIABILITY
#: Reliability of a value scaled from one sample (module docstring, AS-697).
SAMPLE_RELIABILITY = 1.0 / (1.0 + 1.96)
_GENERIC_IF = struct.Struct(">IIQIIQIIIIIIQIIIIII")
_ETHERNET_IF = struct.Struct(">13I")
_VLAN = struct.Struct(">IQIIII")
_PROCESSOR = struct.Struct(">IIIQQ")
_GENERIC_NAMES = ("ifIndex", "ifType", "ifSpeed", "ifDirection", "ifStatus", "ifInOctets", "ifInUcastPkts",
                  "ifInMulticastPkts", "ifInBroadcastPkts", "ifInDiscards", "ifInErrors", "ifInUnknownProtos",
                  "ifOutOctets", "ifOutUcastPkts", "ifOutMulticastPkts", "ifOutBroadcastPkts", "ifOutDiscards",
                  "ifOutErrors", "ifPromiscuousMode")
_ETHERNET_NAMES = ("dot3StatsAlignmentErrors", "dot3StatsFCSErrors", "dot3StatsSingleCollisionFrames",
                   "dot3StatsMultipleCollisionFrames", "dot3StatsSQETestErrors", "dot3StatsDeferredTransmissions",
                   "dot3StatsLateCollisions", "dot3StatsExcessiveCollisions", "dot3StatsInternalMacTransmitErrors",
                   "dot3StatsCarrierSenseErrors", "dot3StatsFrameTooLongs", "dot3StatsInternalMacReceiveErrors",
                   "dot3StatsSymbolErrors")
#: Counters of 64 bits (no wrap correction); the rest are 32-bit.
_COUNTERS_64 = frozenset({"ifInOctets", "ifOutOctets"})


class _Reader:
    """XDR reader over a byte string with bounds checks."""

    def __init__(self, data: bytes, pos: int = 0, end: int | None = None) -> None:
        self.data = data
        self.pos = pos
        self.end = len(data) if end is None else end

    def need(self, n: int) -> None:
        if self.pos + n > self.end:
            raise MalformedRecord("truncated-sflow", f"need {n} bytes at {self.pos}, have {self.end - self.pos}")

    def u32(self) -> int:
        self.need(4)
        v = struct.unpack_from(">I", self.data, self.pos)[0]
        self.pos += 4
        return int(v)

    def u64(self) -> int:
        self.need(8)
        v = struct.unpack_from(">Q", self.data, self.pos)[0]
        self.pos += 8
        return int(v)

    def raw(self, n: int) -> bytes:
        self.need(n)
        v = self.data[self.pos:self.pos + n]
        self.pos += n
        return v

    def opaque(self) -> bytes:
        n = self.u32()
        v = self.raw(n)
        self.pos += (-n) % 4
        if self.pos > self.end:
            raise MalformedRecord("truncated-sflow", "opaque padding past the end")
        return v

    def address(self) -> str | None:
        kind = self.u32()
        if kind == 1:
            return str(ipaddress.IPv4Address(self.raw(4)))
        if kind == 2:
            return str(ipaddress.IPv6Address(self.raw(16)))
        if kind == 0:
            return None
        raise MalformedRecord("bad-sflow-address", f"address type {kind}")

    def mac(self) -> str:
        b = self.raw(8)                                   # opaque<6> padded to 8 bytes
        return b[:6].hex(":")


@dataclass
class Sample:
    """One decoded sample with its values and byte range in the datagram."""

    kind: str                                         # "flow" or "counter"
    values: dict[str, Any]
    offset: int
    length: int
    header: bytes | None = None
    header_protocol: int | None = None


def decode_datagram(data: bytes) -> tuple[dict[str, Any], list[Sample], list[tuple[str, int, int]]]:
    """(datagram header values, samples, retained unknown records (name, offset, length))."""
    r = _Reader(data)
    version = r.u32()
    if version != 5:
        raise MalformedRecord("unsupported-version", f"sFlow version {version}")
    agent = r.address()
    head = {"agent_address": agent, "sub_agent_id": r.u32(), "datagram_sequence": r.u32(), "uptime": r.u32()}
    n = r.u32()
    samples: list[Sample] = []
    unknown: list[tuple[str, int, int]] = []
    for _ in range(n):
        fmt = r.u32()
        length = r.u32()
        start = r.pos
        r.need(length)
        enterprise, sformat = fmt >> 12, fmt & 0xFFF
        sub = _Reader(data, start, start + length)
        if enterprise == 0 and sformat in (1, 3):
            samples.append(_flow_sample(sub, sformat, head, start, length))
        elif enterprise == 0 and sformat in (2, 4):
            samples.append(_counter_sample(sub, sformat, head, start, length, unknown))
        else:
            unknown.append((f"sample_{enterprise}_{sformat}", start, length))
        r.pos = start + length
    return head, samples, unknown


def _flow_sample(r: _Reader, fmt: int, head: dict[str, Any], start: int, length: int) -> Sample:
    v: dict[str, Any] = dict(head)
    v["sample_sequence"] = r.u32()
    if fmt == 1:
        sid = r.u32()
        v["source_id_type"], v["source_id_index"] = sid >> 24, sid & 0xFFFFFF
    else:
        v["source_id_type"], v["source_id_index"] = r.u32(), r.u32()
    v["sampling_rate"], v["sample_pool"], v["drops"] = r.u32(), r.u32(), r.u32()
    if fmt == 1:
        v["input"], v["output"] = r.u32(), r.u32()
    else:
        in_fmt, in_val, out_fmt, out_val = r.u32(), r.u32(), r.u32(), r.u32()
        v["input"] = (in_fmt << 30) | (in_val & 0x3FFFFFFF)
        v["output"] = (out_fmt << 30) | (out_val & 0x3FFFFFFF)
    n = r.u32()
    sample = Sample("flow", v, start, length)
    for _ in range(n):
        fmt_rec = r.u32()
        rlen = r.u32()
        rs = r.pos
        r.need(rlen)
        sub = _Reader(r.data, rs, rs + rlen)
        ent, rf = fmt_rec >> 12, fmt_rec & 0xFFF
        if ent == 0 and rf == 1:
            v["header_protocol"], v["frame_length"], v["stripped"] = sub.u32(), sub.u32(), sub.u32()
            sample.header = sub.opaque()
            sample.header_protocol = v["header_protocol"]
            v["header"] = sample.header
        elif ent == 0 and rf == 2:
            v["eth_length"], v["eth_src"], v["eth_dst"], v["eth_type"] = sub.u32(), sub.mac(), sub.mac(), sub.u32()
        elif ent == 0 and rf in (3, 4):
            alen = 4 if rf == 3 else 16
            v["ip_length"], v["ip_protocol"] = sub.u32(), sub.u32()
            v["ip_src"] = str(ipaddress.ip_address(sub.raw(alen)))
            v["ip_dst"] = str(ipaddress.ip_address(sub.raw(alen)))
            v["ip_src_port"], v["ip_dst_port"], v["ip_tcp_flags"], v["ip_tos"] = sub.u32(), sub.u32(), sub.u32(), sub.u32()
        elif ent == 0 and rf == 1001:
            v["src_vlan"], v["src_priority"], v["dst_vlan"], v["dst_priority"] = sub.u32(), sub.u32(), sub.u32(), sub.u32()
        elif ent == 0 and rf == 1002:
            v["nexthop"] = sub.address()
            v["src_mask_len"], v["dst_mask_len"] = sub.u32(), sub.u32()
        elif ent == 0 and rf == 1003:
            v["nexthop"] = sub.address()
            v["gateway_as"], v["src_as"], v["src_peer_as"] = sub.u32(), sub.u32(), sub.u32()
            path: list[int] = []
            for _seg in range(sub.u32()):
                _seg_type = sub.u32()
                path.extend(sub.u32() for _ in range(sub.u32()))
            v["dst_as_path"] = path
            v["communities"] = [sub.u32() for _ in range(sub.u32())]
            v["localpref"] = sub.u32()
        elif ent == 0 and rf == 1004:
            sub.u32()
            v["src_user"] = sub.opaque().decode("utf-8", "replace")
            sub.u32()
            v["dst_user"] = sub.opaque().decode("utf-8", "replace")
        elif ent == 0 and rf == 1005:
            v["url_direction"] = sub.u32()
            v["url"] = sub.opaque().decode("utf-8", "replace")
            v["url_host"] = sub.opaque().decode("utf-8", "replace")
        else:
            v[f"flow_record_{ent}_{rf}"] = r.data[rs:rs + rlen].hex()
        r.pos = rs + rlen
    return sample


def _counter_sample(r: _Reader, fmt: int, head: dict[str, Any], start: int, length: int,
                    unknown: list[tuple[str, int, int]]) -> Sample:
    v: dict[str, Any] = dict(head)
    v["sample_sequence"] = r.u32()
    if fmt == 2:
        sid = r.u32()
        v["source_id_type"], v["source_id_index"] = sid >> 24, sid & 0xFFFFFF
    else:
        v["source_id_type"], v["source_id_index"] = r.u32(), r.u32()
    for _ in range(r.u32()):
        fmt_rec = r.u32()
        rlen = r.u32()
        rs = r.pos
        r.need(rlen)
        ent, rf = fmt_rec >> 12, fmt_rec & 0xFFF
        body = r.data[rs:rs + rlen]
        if ent == 0 and rf == 1 and rlen >= _GENERIC_IF.size:
            v.update(zip(_GENERIC_NAMES, _GENERIC_IF.unpack_from(body, 0), strict=True))
        elif ent == 0 and rf == 2 and rlen >= _ETHERNET_IF.size:
            v.update(zip(_ETHERNET_NAMES, _ETHERNET_IF.unpack_from(body, 0), strict=True))
        elif ent == 0 and rf == 5 and rlen >= _VLAN.size:
            v.update(zip(("vlan_id", "vlan_octets", "vlan_ucastPkts", "vlan_multicastPkts", "vlan_broadcastPkts",
                          "vlan_discards"), _VLAN.unpack_from(body, 0), strict=True))
        elif ent == 0 and rf == 1001 and rlen >= _PROCESSOR.size:
            v.update(zip(("cpu_5s", "cpu_1m", "cpu_5m", "total_memory", "free_memory"), _PROCESSOR.unpack_from(body, 0),
                         strict=True))
        else:
            v[f"counter_record_{ent}_{rf}"] = body.hex()
        r.pos = rs + rlen
    return Sample("counter", v, start, length)


class SFlowSource(StreamAdapter):
    """sFlow v5 -> state updates. See the module docstring.

    Parameters
    ----------
    source: a packet capture of sFlow datagrams (UDP to `udp_ports`), or an iterable of `Datagram`s.
    config: `SFlowConfig`.
    """

    name = "sflow"
    source_type = "sflow-v5"
    version = ADAPTER_VERSION

    def __init__(self, source: str | Path | bytes | IO[bytes] | Iterable[Datagram], *, config: SFlowConfig | None = None,
                 **kw: Any) -> None:
        self.config = config or SFlowConfig()
        self.datagrams: Iterable[Datagram] | None = None
        if not isinstance(source, str | Path | bytes | bytearray) and not hasattr(source, "read"):
            self.datagrams = source                         # type: ignore[assignment]
            super().__init__(None, self.config.common, **kw)
        else:
            super().__init__(source, self.config.common, **kw)   # type: ignore[arg-type]
        self.prev: BoundedLRU[tuple[Any, ...], dict[str, int]] = BoundedLRU(self.config.max_sources, "sflow_sources",
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
            if not is_capture(head):
                raise MalformedRecord("unknown-input", "sFlow input must be a capture or datagrams")
            ports = set(self.config.udp_ports)
            for item in read_capture(stream):
                if not isinstance(item, CapturedPacket):
                    continue
                udp = udp_payload(item)
                if udp is None or udp[3] not in ports:
                    self.stats.counters["capture_packets_skipped"] += 1
                    continue
                t = None if item.ts_ns is None else item.ts_ns / NS
                yield RawRecord(udp[4], location, item.index, offset=udp[5], received_time=t, exporter=udp[0],
                                meta={"ts_ns": item.ts_ns})
        finally:
            if close:
                stream.close()

    def decode(self, raw: RawRecord) -> Iterable[UpdateDraft]:
        head, samples, unknown = decode_datagram(raw.data)
        for name, off, ln in unknown:
            self.stats.counters[f"unknown_{name}"] += 1
            self.stats.retain(f"sflow.{name}")
        if raw.received_time is None:
            raise MalformedRecord("no-receive-time", "an sFlow datagram has no absolute time without its receipt time")
        ts_ns = (raw.meta or {}).get("ts_ns")
        t = Instant.from_ns(int(ts_ns), 1e-6) if isinstance(ts_ns, int) else Instant(float(raw.received_time), None, 1e-6)
        out = []
        for k, s in enumerate(samples):
            sub = RawRecord(raw.data[s.offset:s.offset + s.length], raw.location, raw.index,
                            offset=None if raw.offset is None else raw.offset + s.offset, sub_index=k,
                            received_time=raw.received_time, exporter=raw.exporter)
            out.append(self._flow(sub, s, t) if s.kind == "flow" else self._counters(sub, s, t))
        return out

    def _flow(self, raw: RawRecord, s: Sample, t: Instant) -> UpdateDraft:
        d = UpdateDraft("sflow.flow_sample", raw, exporter=raw.exporter)
        src = "sflow.flow_sample"
        v = s.values
        self._ctx.values = v
        mapper("sflow.flow_sample").apply(v, d, self._ctx, self.stats, source=src,
                                          keep_unmapped=self.config.common.keep_unmapped)
        d.time = t
        if s.header is not None:
            link = SFLOW_HEADER_LINK.get(int(s.header_protocol or 0))
            if link is None:
                self.stats.counters["sflow_header_protocol_unsupported"] += 1
            else:
                apply_packet(d, decode_packet(s.header, link), self.stats, src)
        rate = d.value("flow.sampling_rate")
        ip_len = d.value("pkt.ip_len_max")
        if rate is not None and int(rate) > 0:
            set_field(d, "flow.packets_fwd", int(rate), src, self.stats, status=LOW, reliability=SAMPLE_RELIABILITY)
            if ip_len is not None:
                set_field(d, "flow.bytes_fwd", int(ip_len) * int(rate), src, self.stats, status=LOW,
                          reliability=SAMPLE_RELIABILITY)
            frame = v.get("frame_length")
            if isinstance(frame, int):
                set_field(d, "flow.frame_bytes_fwd", frame * int(rate), src, self.stats, status=LOW,
                          reliability=SAMPLE_RELIABILITY)
        key = (raw.exporter, v.get("sub_agent_id"), v.get("source_id_type"), v.get("source_id_index"), "flow")
        prev = self.prev.get(key)
        drops = v.get("drops")
        if isinstance(drops, int):
            self.prev[key] = {"drops": drops, "uptime": int(v.get("uptime", 0))}
            if prev is not None:
                delta = _delta32(drops, prev["drops"])
                if delta is not None:
                    set_field(d, "dev.capture_dropped", delta, src, self.stats)
                else:
                    self.stats.refuse("dev.capture_dropped", "counter reset")
        tf = d.value("flow.tcp_flags_fwd")
        if tf is not None:
            set_field(d, "flow.tcp_flags", int(tf) & 0x3F, src, self.stats)
        DV.flow_entities(d, self.resolver)
        return d

    def _counters(self, raw: RawRecord, s: Sample, t: Instant) -> UpdateDraft:
        d = UpdateDraft("sflow.counter_sample", raw, exporter=raw.exporter)
        src = "sflow.counter_sample"
        v = s.values
        self._ctx.values = v
        mapper("sflow.counter_sample").apply(v, d, self._ctx, self.stats, source=src,
                                             keep_unmapped=self.config.common.keep_unmapped)
        d.time = t
        d.add_entity(self.resolver.address(v.get("agent_address")), "subject")
        key = (raw.exporter, v.get("sub_agent_id"), v.get("source_id_type"), v.get("source_id_index"), "counters")
        now = {k: int(x) for k, x in v.items() if k.startswith("if") and isinstance(x, int)}
        now["uptime"] = int(v.get("uptime", 0))
        prev = self.prev.get(key)
        self.prev[key] = now
        if prev is None:
            return d
        up = (now["uptime"] - prev["uptime"]) % (1 << 32)
        if up <= 0:
            self.stats.refuse("dev.interval", "agent uptime did not advance")
            return d
        set_field(d, "dev.interval", up / 1000.0, src, self.stats)
        deltas: dict[str, int] = {}
        for name in _GENERIC_NAMES[5:18]:
            if name in now and name in prev:
                dv = (now[name] - prev[name]) if name in _COUNTERS_64 else _delta32(now[name], prev[name])
                if dv is None or dv < 0:
                    self.stats.refuse(f"sflow.counter_sample.{name}", "counter reset")
                    continue
                deltas[name] = dv
        pairs = {
            "dev.if_in_octets": ("ifInOctets",), "dev.if_out_octets": ("ifOutOctets",),
            "dev.if_in_packets": ("ifInUcastPkts", "ifInMulticastPkts", "ifInBroadcastPkts"),
            "dev.if_out_packets": ("ifOutUcastPkts", "ifOutMulticastPkts", "ifOutBroadcastPkts"),
            "dev.if_errors": ("ifInErrors", "ifOutErrors"), "dev.if_discards": ("ifInDiscards", "ifOutDiscards"),
            "dev.if_in_errors": ("ifInErrors",), "dev.if_out_errors": ("ifOutErrors",),
            "dev.if_in_discards": ("ifInDiscards",), "dev.if_out_discards": ("ifOutDiscards",),
            "dev.if_in_unicast": ("ifInUcastPkts",), "dev.if_out_unicast": ("ifOutUcastPkts",),
            "dev.if_in_multicast": ("ifInMulticastPkts",), "dev.if_out_multicast": ("ifOutMulticastPkts",),
            "dev.if_in_broadcast": ("ifInBroadcastPkts",), "dev.if_out_broadcast": ("ifOutBroadcastPkts",),
            "dev.if_in_unknown_protos": ("ifInUnknownProtos",),
        }
        for fid, parts in pairs.items():
            if all(p in deltas for p in parts):
                set_field(d, fid, sum(deltas[p] for p in parts), src, self.stats)
        return d


def _delta32(now: int, before: int) -> int | None:
    """Delta of a 32-bit counter: one wrap is corrected; a larger decrease is a reset (None)."""
    if now >= before:
        return now - before
    wrapped = now + (1 << 32) - before
    return wrapped if wrapped < (1 << 31) else None


__all__ = ["ADAPTER_VERSION", "SAMPLE_RELIABILITY", "SFlowSource", "decode_datagram"]
