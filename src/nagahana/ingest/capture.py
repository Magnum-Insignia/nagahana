"""Packet-capture files: classic pcap and pcapng, read block by block with byte offsets and exact times.

Classic pcap (IETF draft-ietf-opsawg-pcap): a 24-byte file header (magic, version, zone, accuracy,
snap length, link type) and 16-byte record headers (seconds, fraction, captured length, original
length). The magic gives byte order and fraction unit: a1b2c3d4 microseconds, a1b23c4d nanoseconds
(and their byte-swapped forms).

pcapng (IETF draft-ietf-opsawg-pcapng): blocks of type, total length, body, total length.
    Section Header Block (0x0A0D0D0A)    byte-order magic 0x1A2B3C4D; starts a section, which resets the
                                         interfaces
    Interface Description Block (1)      link type, snap length; options if_name (2), if_description
                                         (3), if_tsresol (9: 10^-n, or 2^-n when the high bit is set;
                                         default 10^-6) and if_tsoffset (14: seconds added to every time)
    Enhanced Packet Block (6)            interface, 64-bit time in if_tsresol units, captured and
                                         original lengths, the packet, options
    Simple Packet Block (3)              original length and the packet (interface 0, no time)
    Interface Statistics Block (5)       isb_ifrecv (4) and isb_ifdrop (5) counters among its options
    Name Resolution Block (4)            IPv4 (1) and IPv6 (2) address-to-name records
Other blocks are counted and skipped. Times are integer nanoseconds: ts_ns = (units x 10^9) / resolution
+ offset x 10^9, rounded to the nanosecond for binary resolutions.
"""

from __future__ import annotations

import socket
import struct
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import IO

from nagahana.ingest.core import MalformedRecord

PCAP_MAGICS: dict[bytes, tuple[str, int]] = {
    b"\xa1\xb2\xc3\xd4": (">", 1_000), b"\xd4\xc3\xb2\xa1": ("<", 1_000),
    b"\xa1\xb2\x3c\x4d": (">", 1), b"\x4d\x3c\xb2\xa1": ("<", 1),
}
PCAPNG_MAGIC = b"\x0a\x0d\x0d\x0a"
BT_SHB, BT_IDB, BT_SPB, BT_NRB, BT_ISB, BT_EPB = 0x0A0D0D0A, 1, 3, 4, 5, 6
_MAX_BLOCK = 1 << 28


@dataclass(slots=True)
class CapturedPacket:
    """One packet of a capture.

    ts_ns: integer nanoseconds since the epoch; None for a simple packet block (no time).
    offset: file offset of the packet bytes; record_offset: of the record or block holding them.
    """

    index: int
    ts_ns: int | None
    data: bytes
    linktype: int
    interface: int
    offset: int
    record_offset: int
    orig_len: int
    resolution: float


@dataclass(slots=True)
class InterfaceStats:
    """An Interface Statistics Block: counters of one interface at a time (pcapng options 4 and 5)."""

    interface: int
    ts_ns: int
    received: int | None
    dropped: int | None
    offset: int


@dataclass(slots=True)
class NameRecord:
    """A Name Resolution Block entry: an address and its names."""

    address: str
    names: tuple[str, ...]
    offset: int


@dataclass
class _Interface:
    linktype: int
    snaplen: int
    tsresol_num: int = 1                     # resolution = num / den seconds
    tsresol_den: int = 1_000_000
    tsoffset: int = 0
    name: str | None = None


@dataclass
class CaptureInfo:
    """What a reader learned about the file (format, link types, skipped blocks)."""

    format: str = ""
    linktypes: list[int] = field(default_factory=list)
    skipped_blocks: dict[int, int] = field(default_factory=dict)
    interfaces: list[str | None] = field(default_factory=list)


def is_capture(head: bytes) -> bool:
    return head[:4] in PCAP_MAGICS or head[:4] == PCAPNG_MAGIC


def read_capture(stream: IO[bytes], info: CaptureInfo | None = None, *,
                 max_packet: int = 1 << 26) -> Iterator[CapturedPacket | InterfaceStats | NameRecord]:
    """Packets (and statistics and name records) of a pcap or pcapng stream, in file order."""
    info = info if info is not None else CaptureInfo()
    head = stream.read(4)
    if head in PCAP_MAGICS:
        yield from _classic(stream, head, info, max_packet)
    elif head == PCAPNG_MAGIC:
        yield from _pcapng(stream, head, info, max_packet)
    else:
        raise MalformedRecord("not-a-capture", f"magic {head.hex()}")


def _classic(stream: IO[bytes], magic: bytes, info: CaptureInfo,
             max_packet: int) -> Iterator[CapturedPacket]:
    order, scale = PCAP_MAGICS[magic]
    rest = stream.read(20)
    if len(rest) < 20:
        raise MalformedRecord("truncated-pcap-header", f"{4 + len(rest)} of 24 bytes")
    _vmaj, _vmin, _zone, _sig, snaplen, network = struct.unpack(order + "HHiIII", rest)
    linktype = network & 0xFFFF
    info.format = "pcap"
    info.linktypes = [linktype]
    rec_hdr = struct.Struct(order + "IIII")
    offset = 24
    index = 0
    resolution = 1e-6 if scale == 1_000 else 1e-9
    while True:
        h = stream.read(16)
        if not h:
            return
        if len(h) < 16:
            raise MalformedRecord("truncated-pcap-record", f"record header of {len(h)} bytes at offset {offset}")
        sec, frac, incl, orig = rec_hdr.unpack(h)
        if incl > max(max_packet, snaplen or 0) or incl > max_packet:
            raise MalformedRecord("pcap-record-too-large", f"captured length {incl} at offset {offset}")
        data = stream.read(incl)
        if len(data) < incl:
            raise MalformedRecord("truncated-pcap-record", f"{len(data)} of {incl} bytes at offset {offset}")
        yield CapturedPacket(index, sec * 1_000_000_000 + frac * scale, data, linktype, 0, offset + 16, offset, orig,
                             resolution)
        offset += 16 + incl
        index += 1


def _options(body: bytes, order: str) -> Iterator[tuple[int, bytes]]:
    i = 0
    while i + 4 <= len(body):
        code, length = struct.unpack_from(order + "HH", body, i)
        i += 4
        if code == 0:
            return
        value = body[i:i + length]
        i += (length + 3) & ~3
        yield code, value


def _pcapng(stream: IO[bytes], first: bytes, info: CaptureInfo,
            max_packet: int) -> Iterator[CapturedPacket | InterfaceStats | NameRecord]:
    info.format = "pcapng"
    order = "<"
    interfaces: list[_Interface] = []
    offset = 0
    index = 0
    pending = first
    while True:
        bt_raw = pending if pending else stream.read(4)
        pending = b""
        if not bt_raw:
            return
        if len(bt_raw) < 4:
            raise MalformedRecord("truncated-pcapng-block", f"at offset {offset}")
        if bt_raw == PCAPNG_MAGIC:
            # Section header: the byte-order magic follows the length, so read both before trusting order.
            tl_raw = stream.read(4)
            bom = stream.read(4)
            if len(tl_raw) < 4 or len(bom) < 4:
                raise MalformedRecord("truncated-pcapng-block", f"section header at offset {offset}")
            if bom == b"\x4d\x3c\x2b\x1a":
                order = "<"
            elif bom == b"\x1a\x2b\x3c\x4d":
                order = ">"
            else:
                raise MalformedRecord("bad-pcapng-byte-order", bom.hex())
            total = struct.unpack(order + "I", tl_raw)[0]
            if total < 28 or total > _MAX_BLOCK or total % 4:
                raise MalformedRecord("bad-pcapng-block-length", f"{total} at offset {offset}")
            rest = stream.read(total - 12)
            if len(rest) < total - 12:
                raise MalformedRecord("truncated-pcapng-block", f"section header at offset {offset}")
            interfaces = []
            info.linktypes = []
            info.interfaces = []
            offset += total
            continue
        btype = struct.unpack(order + "I", bt_raw)[0]
        tl_raw = stream.read(4)
        if len(tl_raw) < 4:
            raise MalformedRecord("truncated-pcapng-block", f"at offset {offset}")
        total = struct.unpack(order + "I", tl_raw)[0]
        if total < 12 or total > _MAX_BLOCK or total % 4:
            raise MalformedRecord("bad-pcapng-block-length", f"{total} at offset {offset}")
        body_trailer = stream.read(total - 8)
        if len(body_trailer) < total - 8:
            raise MalformedRecord("truncated-pcapng-block", f"type {btype} at offset {offset}")
        body = body_trailer[:-4]
        if struct.unpack(order + "I", body_trailer[-4:])[0] != total:
            raise MalformedRecord("pcapng-length-mismatch", f"type {btype} at offset {offset}")
        if btype == BT_IDB:
            linktype, _res, snaplen = struct.unpack_from(order + "HHI", body, 0)
            itf = _Interface(linktype, snaplen)
            for code, value in _options(body[8:], order):
                if code == 9 and value:
                    r = value[0]
                    if r & 0x80:
                        itf.tsresol_num, itf.tsresol_den = 1, 1 << (r & 0x7F)
                    else:
                        itf.tsresol_num, itf.tsresol_den = 1, 10 ** r
                elif code == 14 and len(value) >= 8:
                    itf.tsoffset = struct.unpack(order + "q", value[:8])[0]
                elif code == 2:
                    itf.name = value.decode("utf-8", "replace").rstrip("\x00")
            interfaces.append(itf)
            info.linktypes.append(linktype)
            info.interfaces.append(itf.name)
        elif btype == BT_EPB:
            if len(body) < 20:
                raise MalformedRecord("bad-pcapng-epb", f"at offset {offset}")
            iface, hi, lo, cap, orig = struct.unpack_from(order + "IIIII", body, 0)
            if iface >= len(interfaces):
                raise MalformedRecord("pcapng-unknown-interface", f"interface {iface} at offset {offset}")
            if cap > max_packet or 20 + cap > len(body):
                raise MalformedRecord("bad-pcapng-epb", f"captured length {cap} at offset {offset}")
            itf = interfaces[iface]
            units = (hi << 32) | lo
            ts_ns = (units * 1_000_000_000 * itf.tsresol_num + itf.tsresol_den // 2) // itf.tsresol_den + itf.tsoffset * 1_000_000_000
            yield CapturedPacket(index, ts_ns, body[20:20 + cap], itf.linktype, iface, offset + 28, offset, orig,
                                 itf.tsresol_num / itf.tsresol_den)
            index += 1
        elif btype == BT_SPB:
            if not interfaces:
                raise MalformedRecord("pcapng-unknown-interface", f"simple packet block at offset {offset}")
            orig = struct.unpack_from(order + "I", body, 0)[0]
            itf = interfaces[0]
            cap = min(orig, itf.snaplen or orig, len(body) - 4)
            yield CapturedPacket(index, None, body[4:4 + cap], itf.linktype, 0, offset + 12, offset, orig,
                                 itf.tsresol_num / itf.tsresol_den)
            index += 1
        elif btype == BT_ISB:
            iface, hi, lo = struct.unpack_from(order + "III", body, 0)
            recv = drop = None
            for code, value in _options(body[12:], order):
                if code == 4 and len(value) >= 8:
                    recv = struct.unpack(order + "Q", value[:8])[0]
                elif code == 5 and len(value) >= 8:
                    drop = struct.unpack(order + "Q", value[:8])[0]
            if iface < len(interfaces):
                itf = interfaces[iface]
                units = (hi << 32) | lo
                ts_ns = (units * 1_000_000_000 * itf.tsresol_num + itf.tsresol_den // 2) // itf.tsresol_den + itf.tsoffset * 1_000_000_000
                yield InterfaceStats(iface, ts_ns, recv, drop, offset)
        elif btype == BT_NRB:
            i = 0
            while i + 4 <= len(body):
                rtype, rlen = struct.unpack_from(order + "HH", body, i)
                i += 4
                if rtype == 0:
                    break
                value = body[i:i + rlen]
                i += (rlen + 3) & ~3
                alen = {1: 4, 2: 16}.get(rtype)
                if alen is None or len(value) <= alen:
                    continue
                addr = socket.inet_ntop(socket.AF_INET if alen == 4 else socket.AF_INET6, value[:alen])
                names = tuple(n.decode("utf-8", "replace") for n in value[alen:].split(b"\x00") if n)
                yield NameRecord(addr, names, offset)
        else:
            info.skipped_blocks[btype] = info.skipped_blocks.get(btype, 0) + 1
        offset += total


def udp_payload(pkt: CapturedPacket) -> tuple[str, str, int, int, bytes, int] | None:
    """(source address, destination address, source port, destination port, payload, payload file offset)
    of a UDP packet, or None (used to read flow export and sFlow datagrams from captures)."""
    import dpkt

    from nagahana.ingest.packets import LINK_ETHERNET, LINK_IPV4, LINK_IPV6, LINK_RAW, LINK_RAW_ALT, LINK_SLL

    data = pkt.data
    try:
        if pkt.linktype == LINK_ETHERNET:
            eth = dpkt.ethernet.Ethernet(data)
            net = eth.data
        elif pkt.linktype == LINK_SLL:
            net = dpkt.sll.SLL(data).data
        elif pkt.linktype in (LINK_RAW, LINK_RAW_ALT):
            net = dpkt.ip.IP(data) if data[:1] and data[0] >> 4 == 4 else dpkt.ip6.IP6(data)
        elif pkt.linktype == LINK_IPV4:
            net = dpkt.ip.IP(data)
        elif pkt.linktype == LINK_IPV6:
            net = dpkt.ip6.IP6(data)
        else:
            return None
    except (dpkt.UnpackError, dpkt.NeedData, IndexError, ValueError):
        return None
    if isinstance(net, dpkt.ip.IP):
        src, dst = socket.inet_ntoa(net.src), socket.inet_ntoa(net.dst)
    elif isinstance(net, dpkt.ip6.IP6):
        src, dst = socket.inet_ntop(socket.AF_INET6, net.src), socket.inet_ntop(socket.AF_INET6, net.dst)
    else:
        return None
    udp = net.data
    if not isinstance(udp, dpkt.udp.UDP):
        return None
    payload = bytes(udp.data)
    # The payload starts right after the 8-byte UDP header; locating that exact header (ports, length,
    # checksum) gives the offset whatever link header, VLAN tags, IP options or frame padding precede it.
    header = struct.pack(">HHHH", int(udp.sport), int(udp.dport), int(udp.ulen), int(udp.sum))
    at = data.find(header)
    start = at + 8 if at >= 0 else len(data) - len(payload)
    return src, dst, int(udp.sport), int(udp.dport), payload, pkt.offset + start


__all__ = ["CaptureInfo", "CapturedPacket", "InterfaceStats", "NameRecord", "PCAPNG_MAGIC", "PCAP_MAGICS", "is_capture",
           "read_capture", "udp_payload"]
