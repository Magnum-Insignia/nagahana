"""Decoding of one captured packet (dpkt) into the fields a single packet can supply.

Used where a record carries packet bytes: the triggering packet of a Suricata alert, unified2 packet
records, sFlow sampled headers. A header may be truncated (sFlow samples the first 128 bytes by
default): whatever the available bytes decode to is used, and what they do not reach stays unknown.

Link types (tcpdump.org LINKTYPE values): 1 Ethernet, 12 and 101 raw IP, 113 Linux cooked (SLL), 228
raw IPv4, 229 raw IPv6. sFlow header protocols map onto them (1 Ethernet, 11 IPv4, 12 IPv6).
"""

from __future__ import annotations

import socket
from dataclasses import dataclass
from typing import Any

from nagahana.ingest.core import UpdateDraft
from nagahana.ingest.mapping import set_field

LINK_ETHERNET, LINK_RAW, LINK_RAW_ALT, LINK_SLL, LINK_IPV4, LINK_IPV6 = 1, 12, 101, 113, 228, 229
#: sFlow v5 header_protocol -> link type (sFlow v5, sampled_header).
SFLOW_HEADER_LINK: dict[int, int] = {1: LINK_ETHERNET, 11: LINK_IPV4, 12: LINK_IPV6}


@dataclass
class PacketFacts:
    """What one packet shows. None: not present in the bytes (truncated, or not that protocol)."""

    src_mac: str | None = None
    dst_mac: str | None = None
    vlan: int | None = None
    ip_version: int | None = None
    src: str | None = None
    dst: str | None = None
    proto: int | None = None
    ttl: int | None = None
    ip_len: int | None = None
    tos: int | None = None
    df: int | None = None
    mf: int | None = None
    sport: int | None = None
    dport: int | None = None
    tcp_flags: int | None = None
    tcp_window: int | None = None
    icmp_type: int | None = None
    icmp_code: int | None = None
    payload_len: int | None = None
    truncated: bool = False


def _ip_text(b: bytes) -> str:
    return socket.inet_ntoa(b) if len(b) == 4 else socket.inet_ntop(socket.AF_INET6, b)


def decode_packet(data: bytes, linktype: int) -> PacketFacts:
    """Decode `data` of link type `linktype` as far as it goes (never raises)."""
    import dpkt

    f = PacketFacts()
    net: Any = None
    try:
        if linktype == LINK_ETHERNET:
            eth = dpkt.ethernet.Ethernet(data)
            f.src_mac, f.dst_mac = eth.src.hex(":"), eth.dst.hex(":")
            tags = getattr(eth, "vlan_tags", None)
            if tags:
                f.vlan = int(tags[0].id)
            net = eth.data
        elif linktype == LINK_SLL:
            sll = dpkt.sll.SLL(data)
            if sll.hlen == 6:
                f.src_mac = sll.hdr[:6].hex(":")
            net = sll.data
        elif linktype in (LINK_RAW, LINK_RAW_ALT):
            net = dpkt.ip.IP(data) if data[:1] and data[0] >> 4 == 4 else dpkt.ip6.IP6(data)
        elif linktype == LINK_IPV4:
            net = dpkt.ip.IP(data)
        elif linktype == LINK_IPV6:
            net = dpkt.ip6.IP6(data)
        else:
            f.truncated = True
            return f
    except (dpkt.UnpackError, dpkt.NeedData, IndexError, ValueError):
        f.truncated = True
        return f
    l4: Any = None
    try:
        if isinstance(net, dpkt.ip.IP):
            f.ip_version, f.src, f.dst = 4, _ip_text(net.src), _ip_text(net.dst)
            f.proto, f.ttl, f.ip_len, f.tos = int(net.p), int(net.ttl), int(net.len), int(net.tos)
            f.df, f.mf = (1 if net.df else 0), (1 if net.mf else 0)
            l4 = net.data
        elif isinstance(net, dpkt.ip6.IP6):
            f.ip_version, f.src, f.dst = 6, _ip_text(net.src), _ip_text(net.dst)
            f.proto, f.ttl, f.ip_len = int(net.p), int(net.hlim), int(net.plen) + 40
            f.tos = int(getattr(net, "fc", 0)) & 0xFF
            l4 = net.data
        else:
            f.truncated = True
            return f
    except (AttributeError, ValueError):
        f.truncated = True
        return f
    try:
        if isinstance(l4, dpkt.tcp.TCP):
            f.sport, f.dport, f.tcp_flags, f.tcp_window = int(l4.sport), int(l4.dport), int(l4.flags), int(l4.win)
            f.payload_len = len(l4.data)
        elif isinstance(l4, dpkt.udp.UDP):
            f.sport, f.dport = int(l4.sport), int(l4.dport)
            f.payload_len = len(l4.data)
        elif isinstance(l4, dpkt.icmp.ICMP | dpkt.icmp6.ICMP6):
            f.icmp_type, f.icmp_code = int(l4.type), int(l4.code)
        elif isinstance(l4, bytes | bytearray):
            f.truncated = f.proto in (6, 17)
    except (AttributeError, ValueError):
        f.truncated = True
    return f


def apply_packet(d: UpdateDraft, f: PacketFacts, stats: Any, source: str, *, direction_fwd: bool = True,
                 addresses: bool = True) -> None:
    """Write the fields one packet supplies into a draft (source values win; nothing is overwritten).

    For the single packet: pkt.ttl_mean = ttl_min = ttl_max = its TTL; pkt.ip_len_min = ip_len_max =
    its IP length; its TCP flags as the sender's direction and in flow.tcp_flags (six bits); the
    don't-fragment and more-fragments flags as counts of 0 or 1.
    """
    if addresses:
        if f.src is not None:
            set_field(d, "flow.src_ip", f.src, source, stats)
        if f.dst is not None:
            set_field(d, "flow.dst_ip", f.dst, source, stats)
        if f.proto is not None:
            set_field(d, "flow.protocol", f.proto, source, stats)
        if f.sport is not None:
            set_field(d, "flow.src_port", f.sport, source, stats)
        if f.dport is not None:
            set_field(d, "flow.dst_port", f.dport, source, stats)
    if f.src_mac is not None:
        set_field(d, "flow.src_mac", f.src_mac, source, stats)
    if f.dst_mac is not None:
        set_field(d, "flow.dst_mac", f.dst_mac, source, stats)
    if f.vlan is not None:
        set_field(d, "flow.vlan", f.vlan, source, stats)
    if f.ip_version is not None:
        set_field(d, "flow.ip_version", f.ip_version, source, stats)
    if f.ttl is not None:
        for fid in ("pkt.ttl_mean", "pkt.ttl_min", "pkt.ttl_max"):
            set_field(d, fid, f.ttl, source, stats)
    if f.ip_len is not None:
        set_field(d, "pkt.ip_len_min", f.ip_len, source, stats)
        set_field(d, "pkt.ip_len_max", f.ip_len, source, stats)
    if f.tos is not None:
        set_field(d, "flow.ip_tos", f.tos, source, stats)
    if f.df is not None:
        set_field(d, "pkt.ip_df_count", f.df, source, stats)
        set_field(d, "pkt.ip_mf_count", f.mf, source, stats)
    if f.tcp_flags is not None:
        set_field(d, "flow.tcp_flags_fwd" if direction_fwd else "flow.tcp_flags_bwd", f.tcp_flags & 0xFF, source, stats)
        set_field(d, "flow.tcp_flags", f.tcp_flags & 0x3F, source, stats)
    if f.icmp_type is not None:
        set_field(d, "proto.icmp.type", f.icmp_type, source, stats)
        set_field(d, "proto.icmp.code", f.icmp_code, source, stats)


__all__ = ["LINK_ETHERNET", "LINK_RAW", "LINK_SLL", "SFLOW_HEADER_LINK", "PacketFacts", "apply_packet", "decode_packet"]
