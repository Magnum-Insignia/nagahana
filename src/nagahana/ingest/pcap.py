"""PCAP adapter: packet captures → state updates of the superset data model.

One record = one state update (D-30). Since D-51 (owner, 2026-10-02) the record of a capture is a
**change of a flow's state**, not every packet. Every update carries the *running state of its flow*
at that moment: what is known about the conversation after the packet that triggered it and nothing
later. Replaying the updates in order therefore reproduces what a live flow sensor would have
delivered.

Emission modes (D-51; `emit=`)
------------------------------
- `"flow-state"` (default). A flow emits a state update when its state changes:
      flow-start      its first packet;
      flag-change     a packet carries a TCP flag (SYN ACK FIN RST PSH URG ECE CWR) not yet seen in that
                      packet's direction;
      active-timeout  a packet arrives at least `active_timeout_s` after the flow's last update, while
                      the flow is open (checked on packet arrival, as flow caches do);
      flow-end        the packet that closes the flow (RST, or FIN from both sides);
      idle-end        no packet for `idle_timeout_s`: an update at t_last + idle_timeout_s (also for a
                      closed flow that received packets after its flow-end update);
      capture-end     the file ends while packets of a flow are not yet in an update: an update at the
                      time of the capture's last record.
  The first trigger that applies names the update (`emit_reason`, audit column; `EMIT_REASONS`).
  Each update carries the flow-level running state **and** the packet-level features accumulated
  since the flow started (TTL statistics, windows, fragment flags, payload-size histogram,
  retransmissions, scan evidence): exactly the packet mode's state at the triggering packet (tested).
  Provenance: the raw record of the triggering packet (for idle-end and capture-end, the flow's last
  packet); `first_record` and `packets` say which packets the update covers since the flow's previous
  update. `app.payload` aggregates those packets: OBSERVED (digest of the last readable payload) if
  any was readable, else NOT_OBSERVABLE if any was encrypted, else NOT_SUPPLIED.
- `"packet"`: one update per packet (exact replay; the adapter's only behaviour before D-51).
End of a flow (D-53; AS-337, AS-338): `flow.end_reason` (1 fin = FIN from both sides, 2 rst, 3 idle,
4 capture_end) and `flow.unanswered` (1 if the responder never sent a packet, else 0) are OBSERVED on
updates of an ended flow and NOT_SUPPLIED while it is open (an end reason does not exist yet). The
first cause is kept (a flow reset after one FIN is "rst"; late packets after a close keep it). In
packet mode only FIN and RST can appear, from the closing packet on: idle and capture ends have no
packet, so packet mode has no update to carry them.
Identity facts (aliases, roles, names, first_sent, ttl_initial) are computed per packet in both
modes, so they are identical across modes (tested).

What each update carries
------------------------
- **Entities** (nodes of the hypergraph): the initiator and the responder of the flow and, for
  TCP/UDP/SCTP, the responder's `service` (`"<address>:<port>/<protocol>"`). An address is a
  `multicast` entity when it names a group of machines (multicast or broadcast), a `host` when it is
  inside the monitored network and `external` otherwise. An IPv6 link-local address that ARP ties to
  a host stands for that host: one entity per machine (see *Group address* and *Alias* below).
- **Flow-level fields** (problem statement, flow list): the 5-tuple (addresses as entity keys, ports
  and protocol as categories), the OR of TCP flags and a count per flag, bytes and packets per
  direction, transport payload bytes per direction and total packets (AS-303), duration,
  inter-arrival mean / variance / maximum, and the bidirectional ratio.
- **Packet-level fields** (problem statement, packet list): TTL mean and variance, initial TCP window
  per side, IP fragment-flag counts, payload-size histogram, retransmissions, and evidence of
  sequential and of randomised port access.
- **Observable-region extras** this adapter can read: DNS query type and response code, TLS server
  name, ICMP type, ARP operation, and the application payload's status.
- **Observation status** on every field (D-41, absence is never zero):
    OBSERVED         read from the packets seen so far;
    NOT_OBSERVABLE   present on the wire but unreadable passively (an encrypted payload);
    NOT_SUPPLIED     not in the record: a field that does not apply (TCP flags of a UDP flow), a
                     statistic that does not exist yet (a variance before the second packet), or
                     anything a capture cannot contain.
- **Ordering**: event time from the capture clock; the watermark (latest event time seen before this
  record) and the reorder uncertainty when a record is older than the watermark. A capture says
  nothing about clock synchronisation, so `clock_quality` is "unknown".
- **Provenance**: source, adapter name and version, and the SHA-256 of the raw record (L0): the
  8-byte big-endian capture time in microseconds followed by the captured frame bytes.

Definitions this adapter fixes (stage-1 items, recorded here as the catalogue asks)
----------------------------------------------------------------------------------
- *Flow*: packets sharing addresses, ports and protocol in either direction, split after
  `FLOW_IDLE_TIMEOUT_S` of silence, after a TCP close (RST, or FIN from both sides) or by a new SYN.
- *Initiator*: the sender of the first packet. For a capture that starts mid-conversation: the
  sender of a bare SYN, the receiver of a SYN+ACK, or the side using the higher port when only the
  other port is below 1024.
- *Bytes*: IP-layer bytes (header + payload), per direction.
- *Bidirectional ratio*: responder → initiator bytes divided by initiator → responder bytes.
- *Payload-size histogram*: bins of `PAYLOAD_BIN_LABELS`, counted over both directions.
- *Retransmission*: a TCP segment with payload, or a bare SYN, whose (direction, sequence number,
  length) was already seen on the flow.
- *Port-access evidence*: per (source, destination) pair, the last `SCAN_WINDOW` distinct destination
  ports the source opened, in order of first use. With at least `SCAN_MIN_PORTS` of them,
      sequential = share of consecutive pairs that differ by exactly one port,
      random     = min(1, ports / SCAN_WINDOW) × (1 − sequential).
  The memory is counted in ports, not seconds, so a slow scan accumulates the same evidence.
- *Encrypted payload*: a TLS record header or an SSH banner was seen on the flow, or the service
  port is one of `ENCRYPTED_PORTS`. From then on `app.payload` is NOT_OBSERVABLE.

Identity (D-47, D-48): every fact below is inferred from the traffic alone, never from a known
address, port or time, and carries `since`, the event time from which it was known.

- *Link addresses*: the Ethernet source and destination of each frame. A Linux cooked capture
  records only the source link address and a packet type, of which type 1 (PACKET_BROADCAST, Linux
  packet(7)) stands for the broadcast destination. A raw-IP capture records neither; the rules below
  that need a link address do not apply to it.
- *Sending*: an address sends when it is the source of an IP packet or the sender of an ARP packet.
- *Group address* (entity kind `multicast`): IPv4 224.0.0.0/4, IPv6 ff00::/8, IPv4 255.255.255.255,
  and a directed broadcast, i.e. an IPv4 destination of a frame sent to the Ethernet broadcast
  address ff:ff:ff:ff:ff:ff that has not been seen sending. The value of an address alone never
  marks a directed broadcast: which address is a subnet's broadcast depends on its prefix length,
  which a capture does not carry (x.x.x.255 is an ordinary host in any subnet larger than /24). If
  an address that is a group only by the Ethernet rule later sends, flows that start from then on
  classify it by the normal rule (a new entity row).
- *MAC binding*: learned from ARP only (IPv4 over 6-byte link addresses): the sender pair (sha, spa)
  of requests and replies and the target pair (tha, tpa) of replies. Never bound: IPv4 0.0.0.0, a
  null or group link address (neither is one interface's own address), and an address that is not
  a `host` at that moment.
- *Alias*: an IPv6 link-local source (fe80::/10) is remembered with the source link address of the
  first frame it sent. From the moment that link address is bound to exactly one IPv4 address, the
  link-local address is an alias of that host (`since`): flows that start from then on resolve it,
  as source and as destination, to the host's entity (kind `host`, key = the IPv4 address), and its
  services are keyed by that address. Flows started earlier keep the separate entity. An alias, once
  made, is kept.
- *Roles* (`evidence` is the fixed rule text of `ROLE_EVIDENCE`, never a count):
      dns-server  an address that sent DNS responses (UDP source port 53, QR = 1) to at least
                  `DNS_SERVER_MIN_CLIENTS` distinct clients; `since` = the response to the last of them;
      gateway     each IPv4 address that ARP binds to a link address which is the Ethernet source of
                  IP packets from, or the Ethernet destination of IP packets to, at least
                  `GATEWAY_MIN_EXTERNAL` distinct `external` addresses; `since` = the later of the
                  two facts.
- *Names*: each A or AAAA answer of a DNS response (UDP source port 53) names the answered address
  with the queried name (source "dns"); a TLS ClientHello names the packet's destination address
  with its server name (source "tls"). First occurrence per (address, name), at most
  `MAX_NAMES_PER_ADDRESS` names per address, kept as they appear on the wire.
- *What an entity sent* (entity columns of `columnar()`): the initiator sends the forward packets of
  its flows; the responder, and its service, send the backward ones.
      first_sent   the first packet it sent (NaN if none);
      ttl_initial  the smallest of `TTL_INITIAL_VALUES` that is >= the largest TTL or hop limit it
                   sent (-1 if none); ttl_since: the first packet it sent that carried one (NaN).
                   Packets whose TTL is fixed by their protocol, whatever the system, are left
                   out: packets to a group address (LLMNR and IGMP use 1, mDNS 255), IGMP, and
                   ICMPv6 (Neighbor Discovery requires 255, RFC 4861; MLD uses 1, RFC 3810);
      mac          the first link address ARP bound to a host, or the one remembered for a
                   link-local address; "" if none. It has no time of its own.
- *Fact tables* `aliases`, `roles`, `names` of `columnar()` (`datamodel/columnar.py`, `FACT_TABLES`):
  `entity` is the entity the address stands for at the end of the capture. `names` holds -1 for an
  address that never became an entity; alias and role rows whose entity never appeared are left
  out. The tables hold facts from the whole file: a consumer that shows the state at time t keeps
  only rows with `since <= t`.
- *Counters* in `stats`: `arp_bindings` (distinct link address–IPv4 pairs), `aliases`,
  `gateway_macs`, `group_addresses_inferred` (addresses made groups by the Ethernet rule) and
  `reclassified_group` (of those, the ones that later sent).

Two outputs, one core
---------------------
`updates()` yields `StateUpdate` objects (the contract; convenient for audit and tests).
`columnar()` fills the pandas/NumPy form directly (`datamodel/columnar.py`), which is what a model
consumes; it is about an order of magnitude faster. Both run the same per-packet core, and the tests
check that they agree cell by cell.

Safety
------
An uploaded capture is attacker-influenced input and packet parsers have a long vulnerability
history (ARCH §7). `sandboxed` must be passed explicitly so the caller states where parsing runs.
This adapter only reads: undecodable records are counted in `stats`, never guessed at.

Library: dpkt (pure Python, fast header decoding), imported lazily.
"""

from __future__ import annotations

import hashlib
import ipaddress
import math
import socket
import struct
from array import array
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from nagahana.core.errors import InvariantViolation
from nagahana.datamodel.columnar import (
    CODE_NOT_OBSERVABLE,
    CODE_NOT_SUPPLIED,
    CODE_OBSERVED,
    STATUS_ORDER,
    Column,
    ColumnarUpdates,
    columns_for,
    fact_table,
    finalise_tables,
)
from nagahana.datamodel.fields import Kind
from nagahana.datamodel.records import EntityRef, FieldValue, OrderingInfo, Provenance, StateUpdate
from nagahana.datamodel.status import ObservationStatus

ADAPTER_NAME = "pcap"
ADAPTER_VERSION = "2.0.0"            # 2.0.0: flow-state emission is the default (D-51)

FLOW_IDLE_TIMEOUT_S = 120.0
#: D-51 active timeout (s) of the flow-state mode; why this value: `docs/assumptions/data.md` AS-332.
ACTIVE_TIMEOUT_S = 1.0
EMIT_MODES: tuple[str, ...] = ("flow-state", "packet")
#: `emit_reason` codes of the update table (module docstring, "Emission modes").
EMIT_REASONS: tuple[str, ...] = ("packet", "flow-start", "flag-change", "active-timeout", "flow-end", "idle-end", "capture-end")
_R_PACKET, _R_START, _R_FLAG, _R_ACTIVE, _R_END, _R_IDLE, _R_CAPTURE = range(len(EMIT_REASONS))
#: `flow.end_reason` codes (D-53, AS-337); 0 is never written: an open flow has no end reason (NOT_SUPPLIED).
END_REASON_CODES: dict[str, int] = {"fin": 1, "rst": 2, "idle": 3, "capture_end": 4}
_END_FIN, _END_RST, _END_IDLE, _END_CAPTURE = 1, 2, 3, 4
SCAN_WINDOW = 32
SCAN_MIN_PORTS = 4
PAYLOAD_BIN_EDGES: tuple[int, ...] = (0, 1, 64, 128, 256, 512, 1024, 1460)   # lower edges; the last bin is open
PAYLOAD_BIN_LABELS: tuple[str, ...] = ("0", "1-63", "64-127", "128-255", "256-511", "512-1023", "1024-1459", ">=1460")
# Registered (>= 1024) ports that are server ports in practice (IANA service-name registry), so a flow
# whose first packet comes *from* one of them started before the capture or after an idle timeout.
SERVICE_PORTS: frozenset[int] = frozenset({1433, 1521, 1723, 2049, 3128, 3306, 3389, 5432, 5900, 5985, 5986, 6379,
                                           8000, 8080, 8443, 9200, 11211, 27017})
ENCRYPTED_PORTS: frozenset[int] = frozenset({22, 443, 465, 563, 636, 853, 989, 990, 992, 993, 995, 3389, 5986, 8443})
PROTOCOL_NAMES: dict[int, str] = {1: "icmp", 2: "igmp", 6: "tcp", 17: "udp", 47: "gre", 50: "esp", 58: "icmpv6", 132: "sctp"}

# identity rules (module docstring, "Identity")
DNS_SERVER_MIN_CLIENTS = 2
GATEWAY_MIN_EXTERNAL = 2
MAX_NAMES_PER_ADDRESS = 16
TTL_INITIAL_VALUES: tuple[int, ...] = (32, 64, 128, 255)
#: Fixed description of each role's rule: the `evidence` column of the roles table (never a count).
ROLE_EVIDENCE: dict[str, str] = {
    "dns-server": f"sent DNS responses (UDP source port 53, QR = 1) to at least {DNS_SERVER_MIN_CLIENTS} distinct clients",
    "gateway": "ARP binds it to a MAC that is the Ethernet source of packets from, or the Ethernet destination of "
               f"packets to, at least {GATEWAY_MIN_EXTERNAL} distinct external addresses",
}

#: Fields that become matrix columns, in column order.
MATRIX_FIELDS: tuple[str, ...] = (
    "flow.src_port", "flow.dst_port", "flow.protocol", "flow.tcp_flags",
    "flow.flag_count.syn", "flow.flag_count.ack", "flow.flag_count.fin", "flow.flag_count.rst",
    "flow.flag_count.psh", "flow.flag_count.urg",
    "flow.bytes_fwd", "flow.bytes_bwd", "flow.packets_fwd", "flow.packets_bwd",
    "flow.duration", "flow.iat_mean", "flow.iat_var", "flow.iat_max", "flow.bidir_ratio",
    "pkt.ttl_mean", "pkt.ttl_var", "pkt.tcp_window_init_fwd", "pkt.tcp_window_init_bwd",
    "pkt.ip_df_count", "pkt.ip_mf_count", "pkt.payload_size_hist", "pkt.retransmissions",
    "derived.portscan_sequential", "derived.portscan_random",
    "proto.dns.qtype", "proto.dns.rcode", "proto.icmp.type", "proto.arp.opcode",
    # appended 2026-10-02 (column positions of earlier fields unchanged)
    "flow.payload_bytes_fwd", "flow.payload_bytes_bwd", "flow.packets_total",
    "flow.end_reason", "flow.unanswered",                 # D-53
)
COLUMNS: tuple[Column, ...] = columns_for(MATRIX_FIELDS, histogram_bins={"pkt.payload_size_hist": PAYLOAD_BIN_LABELS})
_COL: dict[str, int] = {c.name: j for j, c in enumerate(COLUMNS)}
_N = len(COLUMNS)

#: Fields present in every update (NOT_SUPPLIED when they do not apply). Extras appear only when they
#: carry a value or are NOT_OBSERVABLE.
EXPLICIT_FIELDS: tuple[str, ...] = (
    "flow.src_ip", "flow.dst_ip",
    *(f for f in MATRIX_FIELDS if not f.startswith("proto.")),
)
#: Non-matrix fields: field → (value column, status column) of the update table.
SIDE_FIELDS: dict[str, tuple[str | None, str]] = {
    "flow.src_ip": ("@entity_0", "src_ip_status"),
    "flow.dst_ip": ("@entity_1", "dst_ip_status"),
    "app.payload": ("payload_digest", "payload_status"),
    "proto.tls.sni": ("tls_sni", "tls_sni_status"),
}

# column positions used in the hot loop
_C_SPORT, _C_DPORT, _C_PROTO, _C_FLAGS = _COL["flow.src_port"], _COL["flow.dst_port"], _COL["flow.protocol"], _COL["flow.tcp_flags"]
_C_FC = _COL["flow.flag_count.syn"]
_C_BF, _C_BB, _C_PF, _C_PB = _COL["flow.bytes_fwd"], _COL["flow.bytes_bwd"], _COL["flow.packets_fwd"], _COL["flow.packets_bwd"]
_C_DUR, _C_IM, _C_IV, _C_IX, _C_RATIO = (
    _COL["flow.duration"], _COL["flow.iat_mean"], _COL["flow.iat_var"], _COL["flow.iat_max"], _COL["flow.bidir_ratio"],
)
_C_TM, _C_TV, _C_WF, _C_WB = _COL["pkt.ttl_mean"], _COL["pkt.ttl_var"], _COL["pkt.tcp_window_init_fwd"], _COL["pkt.tcp_window_init_bwd"]
_C_DF, _C_MF, _C_HIST, _C_RETX = _COL["pkt.ip_df_count"], _COL["pkt.ip_mf_count"], _COL["pkt.payload_size_hist[0]"], _COL["pkt.retransmissions"]
_C_SEQ, _C_RND = _COL["derived.portscan_sequential"], _COL["derived.portscan_random"]
_C_QT, _C_RC, _C_ICMP, _C_ARP = _COL["proto.dns.qtype"], _COL["proto.dns.rcode"], _COL["proto.icmp.type"], _COL["proto.arp.opcode"]
_C_PBF, _C_PBB, _C_PT = _COL["flow.payload_bytes_fwd"], _COL["flow.payload_bytes_bwd"], _COL["flow.packets_total"]
_C_END, _C_UNANS = _COL["flow.end_reason"], _COL["flow.unanswered"]

_TH_FIN, _TH_SYN, _TH_RST, _TH_PSH, _TH_ACK, _TH_URG = 0x01, 0x02, 0x04, 0x08, 0x10, 0x20
_FLAG_BITS = (_TH_SYN, _TH_ACK, _TH_FIN, _TH_RST, _TH_PSH, _TH_URG)      # order of flow.flag_count.*
_NAN = math.nan
_DLT_EN10MB, _DLT_RAW, _DLT_RAW_ALT, _DLT_LINUX_SLL = 1, 12, 101, 113
_PCAP_MAGICS = {b"\xd4\xc3\xb2\xa1", b"\xa1\xb2\xc3\xd4", b"\x4d\x3c\xb2\xa1", b"\xa1\xb2\x3c\x4d"}
_PCAPNG_MAGIC = b"\x0a\x0d\x0d\x0a"
_PACKET_BROADCAST = 1                     # Linux cooked capture packet type: sent to the link broadcast address
_BROADCAST_MAC, _NULL_MAC = b"\xff" * 6, b"\x00" * 6
_LIMITED_BROADCAST, _UNSPECIFIED_V4 = b"\xff" * 4, b"\x00" * 4
_ETH_P_IP = 0x0800


def is_capture(path: str | Path) -> bool:
    """True if the file starts with a pcap or pcapng magic number."""
    try:
        with open(path, "rb") as fh:
            magic = fh.read(4)
    except OSError:
        return False
    return magic in _PCAP_MAGICS or magic == _PCAPNG_MAGIC


def _payload_bin(size: int) -> int:
    if size <= 0:
        return 0
    if size < 64:
        return 1
    if size < 128:
        return 2
    if size < 256:
        return 3
    if size < 512:
        return 4
    if size < 1024:
        return 5
    if size < 1460:
        return 6
    return 7


#: Entity keys (kind, key) of a flow: initiator, responder, responder's service.
_EntityKeys = tuple[tuple[str, str], tuple[str, str], tuple[str, str] | None]


def _is_station(mac: bytes) -> bool:
    """True for a 6-byte individual link address: one interface's own, never a group or the null address."""
    return len(mac) == 6 and not mac[0] & 1 and mac != _NULL_MAC


class _Flow:
    """Running state of one conversation. Plain attributes with slots: this is the hot path."""

    __slots__ = (
        "src", "dst", "sport", "dport", "proto", "keys", "ents", "rel", "index", "t0", "t_last",
        "n_fwd", "n_bwd", "b_fwd", "b_bwd", "pb_fwd", "pb_bwd", "flags", "fc", "iat_n", "iat_s", "iat_ss", "iat_max",
        "ttl_n", "ttl_s", "ttl_ss", "win_fwd", "win_bwd", "df", "mf", "has_frag_flags", "hist", "retx",
        "seen", "syn_seq", "fin_fwd", "fin_bwd", "closed", "end", "scan_seq", "scan_rnd", "encrypted", "sni",
        "qtype", "rcode", "icmp_type", "arp_op",
    )

    def __init__(self, src: bytes, dst: bytes, sport: int, dport: int, proto: int, t: float) -> None:
        self.src, self.dst, self.sport, self.dport, self.proto = src, dst, sport, dport, proto
        self.keys: _EntityKeys | None = None           # fixed by `PcapSource._entity_keys`
        self.ents: tuple[int, int, int] = (-1, -1, -1)
        self.rel = -1
        self.index = -1
        self.t0 = self.t_last = t
        self.n_fwd = self.n_bwd = self.b_fwd = self.b_bwd = 0
        self.pb_fwd = self.pb_bwd = 0                      # transport payload bytes per direction
        self.flags = 0
        self.fc = [0, 0, 0, 0, 0, 0]
        self.iat_n = 0
        self.iat_s = self.iat_ss = self.iat_max = 0.0
        self.ttl_n = 0
        self.ttl_s = self.ttl_ss = 0.0
        self.win_fwd = self.win_bwd = -1
        self.df = self.mf = 0
        self.has_frag_flags = False
        self.hist = [0, 0, 0, 0, 0, 0, 0, 0]
        self.retx = 0
        self.seen: set[tuple[int, int, int]] = set()
        self.syn_seq = -1
        self.fin_fwd = self.fin_bwd = False
        self.closed = False
        self.end = 0                                       # flow.end_reason code once the flow has ended (D-53)
        self.scan_seq = self.scan_rnd = -1.0
        self.encrypted = False
        self.sni = -1
        self.qtype = self.rcode = self.icmp_type = self.arp_op = -1

    def fill(self, v: list[float], s: list[int]) -> None:
        """Write the flow's running state into a value row and a status row."""
        obs = CODE_OBSERVED
        proto = self.proto
        if proto >= 0:
            v[_C_PROTO] = proto
            s[_C_PROTO] = obs
        if self.sport >= 0:
            v[_C_SPORT], v[_C_DPORT] = self.sport, self.dport
            s[_C_SPORT] = s[_C_DPORT] = obs
        if proto == 6:
            v[_C_FLAGS] = self.flags
            s[_C_FLAGS] = obs
            fc = self.fc
            for k in range(6):
                v[_C_FC + k] = fc[k]
                s[_C_FC + k] = obs
            v[_C_RETX] = self.retx
            s[_C_RETX] = obs
            if self.win_fwd >= 0:
                v[_C_WF] = self.win_fwd
                s[_C_WF] = obs
            if self.win_bwd >= 0:
                v[_C_WB] = self.win_bwd
                s[_C_WB] = obs
        if proto >= 0:
            v[_C_BF], v[_C_BB], v[_C_PF], v[_C_PB] = self.b_fwd, self.b_bwd, self.n_fwd, self.n_bwd
            v[_C_DUR] = self.t_last - self.t0
            s[_C_BF] = s[_C_BB] = s[_C_PF] = s[_C_PB] = s[_C_DUR] = obs
            v[_C_PBF], v[_C_PBB], v[_C_PT] = self.pb_fwd, self.pb_bwd, self.n_fwd + self.n_bwd
            s[_C_PBF] = s[_C_PBB] = s[_C_PT] = obs
            if self.b_fwd > 0:
                v[_C_RATIO] = self.b_bwd / self.b_fwd
                s[_C_RATIO] = obs
            n = self.iat_n
            if n >= 1:
                mean = self.iat_s / n
                v[_C_IM], v[_C_IX] = mean, self.iat_max
                s[_C_IM] = s[_C_IX] = obs
                if n >= 2:
                    v[_C_IV] = max(self.iat_ss / n - mean * mean, 0.0)
                    s[_C_IV] = obs
            n = self.ttl_n
            mean = self.ttl_s / n
            v[_C_TM] = mean
            s[_C_TM] = obs
            if n >= 2:
                v[_C_TV] = max(self.ttl_ss / n - mean * mean, 0.0)
                s[_C_TV] = obs
            if self.has_frag_flags:
                v[_C_DF], v[_C_MF] = self.df, self.mf
                s[_C_DF] = s[_C_MF] = obs
            hist = self.hist
            for k in range(8):
                v[_C_HIST + k] = hist[k]
                s[_C_HIST + k] = obs
            if self.scan_seq >= 0.0:
                v[_C_SEQ], v[_C_RND] = self.scan_seq, self.scan_rnd
                s[_C_SEQ] = s[_C_RND] = obs
        if self.qtype >= 0:
            v[_C_QT] = self.qtype
            s[_C_QT] = obs
        if self.rcode >= 0:
            v[_C_RC] = self.rcode
            s[_C_RC] = obs
        if self.icmp_type >= 0:
            v[_C_ICMP] = self.icmp_type
            s[_C_ICMP] = obs
        if self.arp_op >= 0:
            v[_C_ARP] = self.arp_op
            s[_C_ARP] = obs
        if self.end:                                       # D-53: only an ended flow has an end reason
            v[_C_END], v[_C_UNANS] = self.end, (1 if self.n_bwd == 0 else 0)
            s[_C_END] = s[_C_UNANS] = obs


class _Facts:
    """Identity evidence learned so far in one pass over a capture (module docstring, "Identity").

    The per-packet checks are inline set and dict look-ups in `PcapSource._iter`; only rare events
    (an ARP pair, a new link-local source, a new external address behind a link address, a DNS
    response, a name) call these methods.
    """

    __slots__ = ("stats", "sent", "group_l2", "alias", "aliases", "ll_mac", "ll_pending", "mac_ips", "ip_mac", "bound",
                 "ext_src", "ext_dst", "gw_since", "dns_clients", "roles", "role_seen", "names", "names_of")

    def __init__(self, stats: dict[str, Any]) -> None:
        self.stats = stats
        self.sent: set[bytes] = set()                        # addresses seen sending
        self.group_l2: set[bytes] = set()                    # groups by the Ethernet-broadcast rule, until they send
        self.alias: dict[bytes, bytes] = {}                  # link-local address → the host's IPv4 address
        self.aliases: list[tuple[bytes, bytes, bytes, float]] = []     # (link-local, IPv4, link address, since)
        self.ll_mac: dict[bytes, bytes] = {}                 # link-local address → its remembered link address
        self.ll_pending: dict[bytes, list[bytes]] = {}       # link address → link-local addresses awaiting a binding
        self.mac_ips: dict[bytes, list[bytes]] = {}          # link address → IPv4 addresses ARP bound to it
        self.ip_mac: dict[bytes, bytes] = {}                 # IPv4 address → the first link address bound to it
        self.bound: set[tuple[bytes, bytes]] = set()
        self.ext_src: dict[bytes, set[bytes]] = {}           # link address → external sources of its frames
        self.ext_dst: dict[bytes, set[bytes]] = {}           # link address → external destinations of frames to it
        self.gw_since: dict[bytes, float] = {}               # gateway link addresses → since
        self.dns_clients: dict[bytes, set[bytes]] = {}
        self.roles: list[tuple[bytes, str, float]] = []      # (address, role, since)
        self.role_seen: set[tuple[bytes, str]] = set()
        self.names: list[tuple[bytes, str, str, float]] = []           # (address, name, source, since)
        self.names_of: dict[bytes, set[str]] = {}

    def bind(self, mac: bytes, ip: bytes, kind: str, t: float) -> None:
        """ARP showed the pair (mac, ip) at t; `kind` is the kind of `ip` at that moment."""
        if kind != "host" or ip == _UNSPECIFIED_V4 or not _is_station(mac) or (mac, ip) in self.bound:
            return
        self.bound.add((mac, ip))
        self.stats["arp_bindings"] += 1
        self.ip_mac.setdefault(ip, mac)
        ips = self.mac_ips.setdefault(mac, [])
        ips.append(ip)
        if mac in self.gw_since:                             # the gateway evidence came first
            self.role(ip, "gateway", t)
        waiting = self.ll_pending.pop(mac, [])
        if len(ips) == 1:                                    # otherwise the link address is ambiguous
            for addr in waiting:
                self.make_alias(addr, ip, mac, t)

    def link_local(self, addr: bytes, mac: bytes, t: float) -> None:
        """The IPv6 link-local address `addr` sent its first frame, from link address `mac`."""
        if not _is_station(mac):
            return
        self.ll_mac[addr] = mac
        ips = self.mac_ips.get(mac)
        if ips is None:
            self.ll_pending.setdefault(mac, []).append(addr)
        elif len(ips) == 1:
            self.make_alias(addr, ips[0], mac, t)

    def make_alias(self, addr: bytes, host: bytes, mac: bytes, t: float) -> None:
        self.alias[addr] = host
        self.aliases.append((addr, host, mac, t))
        self.stats["aliases"] += 1

    def external(self, table: dict[bytes, set[bytes]], mac: bytes, addr: bytes, t: float) -> None:
        """Link address `mac` carried an IP packet from (table `ext_src`) or to (`ext_dst`) external `addr`."""
        seen = table.get(mac)
        if seen is None:
            if not _is_station(mac):
                return
            seen = table[mac] = set()
        if addr in seen:
            return
        seen.add(addr)
        if len(seen) >= GATEWAY_MIN_EXTERNAL and mac not in self.gw_since:
            self.gw_since[mac] = t
            self.stats["gateway_macs"] += 1
            for ip in self.mac_ips.get(mac, []):             # the bindings came first
                self.role(ip, "gateway", t)

    def dns_response(self, server: bytes, client: bytes, t: float) -> None:
        clients = self.dns_clients.get(server)
        if clients is None:
            clients = self.dns_clients[server] = set()
        if len(clients) < DNS_SERVER_MIN_CLIENTS and client not in clients:
            clients.add(client)
            if len(clients) == DNS_SERVER_MIN_CLIENTS:
                self.role(server, "dns-server", t)

    def role(self, addr: bytes, role: str, t: float) -> None:
        if (addr, role) not in self.role_seen:
            self.role_seen.add((addr, role))
            self.roles.append((addr, role, t))

    def name(self, addr: bytes, name: str, source: str, t: float) -> None:
        known = self.names_of.get(addr)
        if known is None:
            known = self.names_of[addr] = set()
        if name not in known and len(known) < MAX_NAMES_PER_ADDRESS:
            known.add(name)
            self.names.append((addr, name, source, t))


class _Emission:
    """Flow-state emission bookkeeping of one flow (D-51; module docstring, "Emission modes")."""

    __slots__ = ("flow", "dir_flags", "last_emit", "pending", "first_record", "last_pkt", "pay_obs", "pay_no",
                 "digest", "closed_emitted")

    def __init__(self, flow: _Flow, t: float) -> None:
        self.flow = flow
        self.dir_flags = [0, 0]           # TCP flags seen so far per direction
        self.last_emit = t                # time of the flow's previous update
        self.pending = 0                  # packets not yet covered by an update
        self.first_record = -1            # first of those packets
        self.last_pkt: tuple[int, float, int, int, bytes] = (-1, t, -1, 0, bytes(32))
        self.pay_obs = self.pay_no = False
        self.digest: bytes | None = None
        self.closed_emitted = False

    def add(self, record: int, ts: float, off: int, length: int, raw_hash: bytes, pstatus: int, digest: bytes | None) -> None:
        """Account one packet of the flow."""
        if self.pending == 0:
            self.first_record = record
        self.pending += 1
        self.last_pkt = (record, ts, off, length, raw_hash)
        if pstatus == CODE_OBSERVED:
            self.pay_obs, self.digest = True, digest
        elif pstatus == CODE_NOT_OBSERVABLE:
            self.pay_no = True

    def reset(self, t: float) -> None:
        """After an update: nothing pending; the active timeout restarts at t."""
        self.pending, self.first_record, self.last_emit = 0, -1, t
        self.pay_obs = self.pay_no = False
        self.digest = None


class PcapSource:
    """PCAP / PCAPNG file → state updates.

    Parameters
    ----------
    path:
        The capture file.
    sandboxed:
        Whether the caller runs this in a sandboxed process (ARCH §7). Required, no default, so the
        choice is always explicit; it is recorded in `stats`.
    source_id:
        Name recorded in provenance; defaults to the file name.
    internal_networks:
        CIDR blocks of the monitored network. When None: private, link-local and loopback ranges.
    max_records:
        Stop after this many records of the file (None = all). `stats["truncated"]` says if it did.
    ingest_clock:
        Returns the ingest time of a record. When None the ingest time equals the event time: a file
        is replayed on its own clock, which keeps the output a pure function of the file.
    emit:
        "flow-state" (default, D-51) or "packet" (one update per packet, exact replay).
    active_timeout_s:
        Flow-state mode: an open flow emits an update on the first packet at least this long after its
        previous update (AS-332).
    idle_timeout_s:
        A flow ends after this much silence (both modes; a later packet starts a new flow).
    """

    name = ADAPTER_NAME
    implemented = True

    def __init__(
        self,
        path: str | Path,
        *,
        sandboxed: bool,
        source_id: str | None = None,
        internal_networks: Sequence[str] | None = None,
        max_records: int | None = None,
        ingest_clock: Callable[[], float] | None = None,
        emit: str = "flow-state",
        active_timeout_s: float = ACTIVE_TIMEOUT_S,
        idle_timeout_s: float = FLOW_IDLE_TIMEOUT_S,
    ) -> None:
        if emit not in EMIT_MODES:
            raise InvariantViolation(f"emit must be one of {EMIT_MODES}, got {emit!r}")
        if not (active_timeout_s > 0 and idle_timeout_s > 0):
            raise InvariantViolation("active_timeout_s and idle_timeout_s must be > 0")
        self.emit = emit
        self.active_timeout_s = float(active_timeout_s)
        self.idle_timeout_s = float(idle_timeout_s)
        self.path = Path(path)
        self.sandboxed = sandboxed
        self.source_id = source_id or self.path.name
        self.max_records = max_records
        self.ingest_clock = ingest_clock
        self._nets = [ipaddress.ip_network(n) for n in internal_networks] if internal_networks is not None else None
        self._addr_kind: dict[bytes, str] = {}
        self._addr_text: dict[bytes, str] = {}
        self._sni: list[str] = []
        self.stats: dict[str, Any] = {}
        self._facts = _Facts(self.stats)
        #: per decodable packet: flow index, direction, time, TTL (identity tables, both modes)
        self._packets: tuple[array[int], array[int], array[float], array[int]] = (array("q"), array("b"), array("d"), array("h"))

    # ------------------------------------------------------------------ addresses and entities
    def _kind_of(self, addr: bytes) -> str:
        """Kind of an address by its value alone (the normal rule); see `_identity` for what is learned."""
        kind = self._addr_kind.get(addr)
        if kind is None:
            ip = ipaddress.ip_address(addr)
            if ip.is_multicast or addr == _LIMITED_BROADCAST:
                kind = "multicast"         # names a group of machines, not one machine (D-47)
            elif self._nets is not None:
                kind = "host" if any(ip in n for n in self._nets) else "external"
            elif ip.is_private or ip.is_link_local or ip.is_loopback or ip.is_unspecified:
                kind = "host"
            else:
                kind = "external"
            self._addr_kind[addr] = kind
        return kind

    # ------------------------------------------------------------------ the per-packet core
    def _iter(self, progress: Callable[[float], None] | None = None) -> Iterator[tuple[Any, ...]]:
        """Decode the file and yield one tuple per state update.

        Tuple: (record, ts, raw_offset, raw_len, frame, flow, direction, ip_len, ttl, payload_status,
                payload, watermark, tcp_flags). `flow` holds the running state *after* this packet; `ttl`
                is the packet's TTL or hop limit (-1 for ARP); `tcp_flags` is -1 for a non-TCP packet.
        One tuple per decodable packet; `_records` turns them into state updates.
        """
        import dpkt

        size = max(self.path.stat().st_size, 1)
        flows: dict[tuple[bytes, bytes, int, int, int], _Flow] = {}
        scan: dict[tuple[bytes, bytes], tuple[list[int], set[int]]] = {}
        sni_index: dict[str, int] = {}
        self._sni = []
        stats: dict[str, Any] = {"records": 0, "updates": 0, "skipped_not_ip": 0, "skipped_undecodable": 0, "flows": 0,
                 "arp": 0, "ipv6": 0, "truncated": False, "sandboxed": self.sandboxed,
                 "arp_bindings": 0, "aliases": 0, "gateway_macs": 0, "group_addresses_inferred": 0, "reclassified_group": 0}
        self.stats = stats
        facts = self._facts = _Facts(stats)
        sent, group_l2, gw_since, ll_mac = facts.sent, facts.group_l2, facts.gw_since, facts.ll_mac
        ext_src, ext_dst = facts.ext_src, facts.ext_dst
        akind, kind_of = self._addr_kind, self._kind_of
        smac: bytes | None
        dmac: bytes | None
        n_flows = 0
        idle_timeout = self.idle_timeout_s

        with open(self.path, "rb") as fh:
            magic = fh.read(4)
            fh.seek(0)
            classic = magic in _PCAP_MAGICS
            reader: Any = dpkt.pcap.Reader(fh) if classic else dpkt.pcapng.Reader(fh)
            link = reader.datalink()
            stats["link_type"] = int(link)
            offset = 24                       # classic pcap: global header, then 16-byte record headers
            watermark = _NAN
            record = -1
            Ethernet, IP, IP6, ARP, SLL = dpkt.ethernet.Ethernet, dpkt.ip.IP, dpkt.ip6.IP6, dpkt.arp.ARP, dpkt.sll.SLL
            TCP, UDP, ICMP = dpkt.tcp.TCP, dpkt.udp.UDP, dpkt.icmp.ICMP

            for ts, frame in reader:
                record += 1
                raw_offset = offset if classic else -1
                offset += 16 + len(frame)
                if self.max_records is not None and record >= self.max_records:
                    stats["truncated"] = True
                    break
                if progress is not None and record % 20000 == 0:
                    progress(min(fh.tell() / size, 1.0))
                stats["records"] = record + 1
                ts = float(ts)

                # ---- link layer
                try:
                    if link == _DLT_EN10MB:
                        eth = Ethernet(frame)
                        net, smac, dmac = eth.data, eth.src, eth.dst
                    elif link == _DLT_LINUX_SLL:
                        sll = SLL(frame)
                        net = sll.data
                        smac = sll.hdr[:6] if sll.hlen == 6 else None
                        dmac = _BROADCAST_MAC if sll.type == _PACKET_BROADCAST else None
                    elif link in (_DLT_RAW, _DLT_RAW_ALT):
                        net = IP(frame) if frame[:1] and frame[0] >> 4 == 4 else IP6(frame)
                        smac = dmac = None
                    else:
                        stats["skipped_not_ip"] += 1
                        continue
                except Exception:
                    stats["skipped_undecodable"] += 1
                    continue

                direction = 0
                payload = b""
                payload_len = 0
                tcp_flags = -1
                tcp: Any = None

                # ---- network layer
                if isinstance(net, IP):
                    src, dst, proto, ttl, ip_len = net.src, net.dst, int(net.p), int(net.ttl), int(net.len)
                    frag_flags: tuple[int, int] | None = (1 if net.df else 0, 1 if net.mf else 0)
                    l4 = net.data
                elif isinstance(net, IP6):
                    src, dst, proto, ttl, ip_len = net.src, net.dst, int(net.p), int(net.hlim), int(net.plen) + 40
                    frag_flags = None
                    l4 = net.data
                    stats["ipv6"] += 1
                elif isinstance(net, ARP):
                    src, dst, proto, ttl, ip_len = net.spa, net.tpa, -1, -1, 0
                    frag_flags = None
                    l4 = None
                    stats["arp"] += 1
                    if len(src) != 4 or len(dst) != 4:
                        stats["skipped_undecodable"] += 1
                        continue
                else:
                    stats["skipped_not_ip"] += 1
                    continue

                # ---- identity evidence (module docstring, "Identity"): only set and dict look-ups per packet
                if src not in sent:
                    sent.add(src)
                    if src in group_l2:                 # a group by the Ethernet rule sends: it is a machine
                        group_l2.discard(src)
                        stats["reclassified_group"] += 1
                if proto >= 0:                          # an IP packet
                    if dmac is not None:
                        if (dmac == _BROADCAST_MAC and len(dst) == 4 and dst not in sent and dst not in group_l2
                                and (akind.get(dst) or kind_of(dst)) != "multicast"):
                            group_l2.add(dst)           # a directed broadcast
                            stats["group_addresses_inferred"] += 1
                        if dmac not in gw_since and (akind.get(dst) or kind_of(dst)) == "external":
                            facts.external(ext_dst, dmac, dst, ts)
                    if smac is not None:
                        if smac not in gw_since and (akind.get(src) or kind_of(src)) == "external":
                            facts.external(ext_src, smac, src, ts)
                        if len(src) == 16 and src[0] == 0xFE and src[1] & 0xC0 == 0x80 and src not in ll_mac:
                            facts.link_local(src, smac, ts)
                elif net.op in (1, 2) and net.pro == _ETH_P_IP and net.hln == 6 and net.pln == 4:
                    facts.bind(net.sha, src, self._identity(src)[0], ts)          # sender of a request or reply
                    if net.op == 2:
                        facts.bind(net.tha, dst, self._identity(dst)[0], ts)      # target of a reply

                # ---- transport layer
                sport = dport = -1
                if isinstance(l4, TCP):
                    tcp = l4
                    sport, dport, tcp_flags = int(tcp.sport), int(tcp.dport), int(tcp.flags)
                    payload = bytes(tcp.data)
                elif isinstance(l4, UDP):
                    sport, dport = int(l4.sport), int(l4.dport)
                    payload = bytes(l4.data)
                elif isinstance(l4, ICMP):
                    payload = bytes(l4.data.data) if hasattr(l4.data, "data") and isinstance(l4.data.data, bytes) else b""
                elif isinstance(l4, bytes | bytearray):
                    payload = bytes(l4)          # a later fragment or an unparsed protocol: no ports
                payload_len = len(payload)

                # ---- find or start the flow
                key = (src, dst, sport, dport, proto)
                flow = flows.get(key)
                if flow is None:
                    rkey = (dst, src, dport, sport, proto)
                    flow = flows.get(rkey)
                    if flow is not None:
                        key, direction = rkey, 1
                if flow is not None:
                    bare_syn = tcp_flags >= 0 and (tcp_flags & (_TH_SYN | _TH_ACK)) == _TH_SYN
                    new_syn = bare_syn and direction == 0 and int(tcp.seq) != flow.syn_seq
                    if ts - flow.t_last > idle_timeout or (flow.closed and bare_syn) or new_syn:
                        del flows[key]
                        flow = None
                        key, direction = (src, dst, sport, dport, proto), 0
                if flow is None:
                    # initiator = sender of the first packet, corrected for mid-conversation starts
                    sport_service = 0 <= sport < 1024 or sport in SERVICE_PORTS
                    dport_service = 0 <= dport < 1024 or dport in SERVICE_PORTS
                    low_port_sender = sport_service and not dport_service and dport >= 0
                    if tcp_flags >= 0:
                        syn_ack = (tcp_flags & (_TH_SYN | _TH_ACK)) == (_TH_SYN | _TH_ACK)
                        swap = syn_ack or (not tcp_flags & _TH_SYN and low_port_sender)
                    else:
                        swap = low_port_sender
                    if swap:
                        flow = _Flow(dst, src, dport, sport, proto, ts)
                        key, direction = (dst, src, dport, sport, proto), 1
                    else:
                        flow = _Flow(src, dst, sport, dport, proto, ts)
                    flows[key] = flow
                    flow.index = n_flows
                    n_flows += 1
                    # port-access evidence of this source towards this destination
                    if flow.dport >= 0 and proto in (6, 17):
                        pair = (flow.src, flow.dst)
                        mem = scan.get(pair)
                        if mem is None:
                            mem = scan[pair] = ([], set())
                        order, known = mem
                        if flow.dport not in known:
                            known.add(flow.dport)
                            order.append(flow.dport)
                            if len(order) > SCAN_WINDOW:
                                known.discard(order.pop(0))
                        if len(order) >= SCAN_MIN_PORTS:
                            steps = sum(1 for a, b in zip(order, order[1:], strict=False) if abs(a - b) == 1)
                            seq_share = steps / (len(order) - 1)
                            flow.scan_seq = seq_share
                            flow.scan_rnd = min(1.0, len(order) / SCAN_WINDOW) * (1.0 - seq_share)
                        if flow.dport in ENCRYPTED_PORTS:
                            flow.encrypted = True
                else:
                    gap = ts - flow.t_last
                    if gap < 0.0:
                        gap = 0.0
                    flow.iat_n += 1
                    flow.iat_s += gap
                    flow.iat_ss += gap * gap
                    if gap > flow.iat_max:
                        flow.iat_max = gap
                    if ts > flow.t_last:
                        flow.t_last = ts

                # ---- update the running state
                if direction == 0:
                    flow.n_fwd += 1
                    flow.b_fwd += ip_len
                    flow.pb_fwd += payload_len
                else:
                    flow.n_bwd += 1
                    flow.b_bwd += ip_len
                    flow.pb_bwd += payload_len
                if ttl >= 0:
                    flow.ttl_n += 1
                    flow.ttl_s += ttl
                    flow.ttl_ss += ttl * ttl
                    flow.hist[_payload_bin(payload_len)] += 1
                if frag_flags is not None:
                    flow.has_frag_flags = True
                    flow.df += frag_flags[0]
                    flow.mf += frag_flags[1]
                if tcp is not None:
                    flow.flags |= tcp_flags & 0x3F
                    fc = flow.fc
                    for k in range(6):
                        if tcp_flags & _FLAG_BITS[k]:
                            fc[k] += 1
                    win = int(tcp.win)
                    if direction == 0:
                        if flow.win_fwd < 0:
                            flow.win_fwd = win
                    elif flow.win_bwd < 0:
                        flow.win_bwd = win
                    seq = int(tcp.seq)
                    if (tcp_flags & (_TH_SYN | _TH_ACK)) == _TH_SYN:
                        if flow.syn_seq == seq and direction == 0 and flow.fc[0] > 1:
                            flow.retx += 1
                        flow.syn_seq = seq if direction == 0 else flow.syn_seq
                    if payload_len > 0:
                        sig = (direction, seq, payload_len)
                        if sig in flow.seen:
                            flow.retx += 1
                        elif len(flow.seen) < 2048:
                            flow.seen.add(sig)
                    if tcp_flags & _TH_RST:
                        flow.closed = True
                        if not flow.end:
                            flow.end = _END_RST
                    if tcp_flags & _TH_FIN:
                        if direction == 0:
                            flow.fin_fwd = True
                        else:
                            flow.fin_bwd = True
                        if flow.fin_fwd and flow.fin_bwd:
                            flow.closed = True
                            if not flow.end:
                                flow.end = _END_FIN
                    if payload_len >= 5 and not flow.encrypted:
                        b0 = payload[0]
                        if (0x14 <= b0 <= 0x17 and payload[1] == 3 and payload[2] <= 4) or payload[:4] == b"SSH-":
                            flow.encrypted = True
                    if payload_len > 43 and payload[0] == 0x16 and payload[5] == 0x01 and flow.sni < 0:
                        name = _client_hello_sni(payload)
                        if name:
                            idx = sni_index.get(name)
                            if idx is None:
                                idx = sni_index[name] = len(self._sni)
                                self._sni.append(name)
                            flow.sni = idx
                            facts.name(dst, name, "tls", ts)
                elif proto == 17 and payload_len >= 12 and (flow.dport == 53 or flow.sport == 53):
                    qr = payload[2] >> 7
                    if qr:
                        flow.rcode = payload[3] & 0x0F
                        if sport == 53:                 # a DNS response: role and name evidence
                            facts.dns_response(src, dst, ts)
                            answered = _dns_answers(payload)
                            if answered is not None:
                                for addr in answered[1]:
                                    facts.name(addr, answered[0], "dns", ts)
                    qtype = _dns_qtype(payload)
                    if qtype >= 0:
                        flow.qtype = qtype
                elif isinstance(l4, ICMP):
                    flow.icmp_type = int(l4.type)
                elif proto == -1:
                    flow.arp_op = int(net.op)

                # ---- payload status of this record
                if payload_len == 0:
                    payload_status = CODE_NOT_SUPPLIED
                elif flow.encrypted:
                    payload_status = CODE_NOT_OBSERVABLE
                else:
                    payload_status = CODE_OBSERVED

                stats["updates"] += 1
                yield (record, ts, raw_offset, len(frame), frame, flow, direction, ip_len, ttl, payload_status, payload,
                       watermark, tcp_flags)
                if not ts <= watermark:          # also true while the watermark is still NaN
                    watermark = ts

                if record % 50000 == 0 and len(flows) > 20000:       # bound memory: drop finished flows
                    dead = [fk for fk, f in flows.items() if f.closed or ts - f.t_last > idle_timeout]
                    for fk in dead:
                        del flows[fk]
        stats["flows"] = n_flows
        if progress is not None:
            progress(1.0)

    def _text(self, addr: bytes) -> str:
        text = self._addr_text.get(addr)
        if text is None:
            text = socket.inet_ntoa(addr) if len(addr) == 4 else socket.inet_ntop(socket.AF_INET6, addr)
            self._addr_text[addr] = text
        return text

    def _identity(self, addr: bytes) -> tuple[str, str]:
        """(kind, key) of the entity an address stands for, given what has been learned so far."""
        facts = self._facts
        host = facts.alias.get(addr)
        if host is not None:
            return "host", self._text(host)
        if addr in facts.group_l2:
            return "multicast", self._text(addr)
        return self._kind_of(addr), self._text(addr)

    def _entity_keys(self, flow: _Flow) -> _EntityKeys:
        """Entity keys of a flow: initiator, responder and the responder's service (keyed by its canonical address).

        Resolved once, on the first call, and kept on the flow. `updates()` and `columnar()` both make
        that call on the flow's first state update, so both see identity (aliases, group addresses) as
        it was known when the flow started, and they stay identical.
        """
        keys = flow.keys
        if keys is None:
            a, b = self._identity(flow.src), self._identity(flow.dst)
            service = None
            if flow.dport >= 0 and flow.proto in (6, 17, 132):
                service = ("service", f"{b[1]}:{flow.dport}/{PROTOCOL_NAMES.get(flow.proto, str(flow.proto))}")
            keys = flow.keys = (a, b, service)
        return keys

    @staticmethod
    def raw_hash(ts: float, frame: bytes) -> bytes:
        """SHA-256 of a raw record: capture time in microseconds (8 bytes, big-endian) + frame."""
        return hashlib.sha256(struct.pack(">q", int(round(ts * 1e6))) + frame).digest()

    # ------------------------------------------------------------------ packets → state updates (D-51)
    def _records(self, progress: Callable[[float], None] | None = None) -> Iterator[tuple[Any, ...]]:
        """Turn the per-packet core into state updates (module docstring, "Emission modes").

        Tuple: (record, event_time, raw_offset, raw_len, raw_hash, flow, direction, ip_len, payload_status,
        payload_digest, watermark, emit_reason, first_record, packets). `payload_digest` is the first 8
        bytes of the SHA-256 of the payload, or None. `flow` holds the running state the update carries;
        the consumer reads it before asking for the next tuple.

        Side effect: `self._packets` collects, per decodable packet, (flow index, direction, time, TTL),
        from which the identity tables are computed identically in both modes.
        """
        import heapq

        sha256, pack = hashlib.sha256, struct.pack
        pk_flow, pk_dir, pk_t, pk_ttl = array("q"), array("b"), array("d"), array("h")
        self._packets = (pk_flow, pk_dir, pk_t, pk_ttl)
        stats_updates = 0

        def digest(payload: bytes, status: int) -> bytes | None:
            return sha256(payload).digest()[:8] if status == CODE_OBSERVED else None

        if self.emit == "packet":
            for record, ts, off, length, frame, flow, direction, ip_len, ttl, pstatus, payload, wm, _tf in self._iter(progress):
                pk_flow.append(flow.index)
                pk_dir.append(direction)
                pk_t.append(ts)
                pk_ttl.append(ttl)
                stats_updates += 1
                yield (record, ts, off, length, sha256(pack(">q", int(round(ts * 1e6))) + frame).digest(), flow, direction,
                       ip_len, pstatus, digest(payload, pstatus), wm, _R_PACKET, record, 1)
            self.stats["updates"] = stats_updates
            return

        active, idle = self.active_timeout_s, self.idle_timeout_s
        states: dict[int, _Emission] = {}
        heap: list[tuple[float, int]] = []               # (expiry time, flow index), lazily refreshed
        wm_now = _NAN
        last_ts = _NAN

        def emit(st: _Emission, t: float, reason: int, direction: int, wm: float) -> tuple[Any, ...]:
            # one state update covering the flow's packets since its previous update
            rec, _ts, off, length, h = st.last_pkt
            status = CODE_OBSERVED if st.pay_obs else (CODE_NOT_OBSERVABLE if st.pay_no else CODE_NOT_SUPPLIED)
            out = (rec, t, off, length, h, st.flow, direction, -1, status, st.digest if st.pay_obs else None, wm,
                   reason, st.first_record, st.pending)
            st.reset(t)
            return out

        def expire(before: float, wm: float) -> Iterator[tuple[Any, ...]]:
            # idle-end updates of every flow silent for longer than idle_timeout_s before `before`
            while heap and heap[0][0] < before:
                exp, idx = heapq.heappop(heap)
                st = states.get(idx)
                if st is None:
                    continue
                real = st.flow.t_last + idle
                if real > exp:                           # the flow saw packets since: refresh its entry
                    heapq.heappush(heap, (real, idx))
                    continue
                del states[idx]
                if not st.flow.end:                      # a closed flow keeps its FIN / RST reason
                    st.flow.end = _END_IDLE
                if st.pending or not st.flow.closed:
                    yield emit(st, real, _R_IDLE, -1, wm)

        for record, ts, off, length, frame, flow, direction, _ip_len, ttl, pstatus, payload, wm, tflags in self._iter(progress):
            wm_now, last_ts = wm, ts
            for upd in expire(ts, wm):
                stats_updates += 1
                yield upd
            pk_flow.append(flow.index)
            pk_dir.append(direction)
            pk_t.append(ts)
            pk_ttl.append(ttl)
            st = states.get(flow.index)
            reason = -1
            if st is None:                                # flow-start
                st = states[flow.index] = _Emission(flow, ts)
                heapq.heappush(heap, (flow.t_last + idle, flow.index))
                reason = _R_START
            st.add(record, ts, off, length, sha256(pack(">q", int(round(ts * 1e6))) + frame).digest(), pstatus,
                   digest(payload, pstatus))
            new_flags = tflags >= 0 and (tflags & 0xFF & ~st.dir_flags[direction]) != 0
            if tflags >= 0:
                st.dir_flags[direction] |= tflags & 0xFF
            if reason < 0:
                if flow.closed and not st.closed_emitted:
                    reason = _R_END
                elif new_flags:
                    reason = _R_FLAG
                elif not flow.closed and ts - st.last_emit >= active:
                    reason = _R_ACTIVE
            if reason >= 0:
                if flow.closed:
                    st.closed_emitted = True
                stats_updates += 1
                yield emit(st, ts, reason, direction, wm)
        # end of the capture: open flows with packets not yet in an update
        for upd in expire(last_ts, wm_now):
            stats_updates += 1
            yield upd
        for idx in sorted(states):
            st = states[idx]
            if st.pending:
                if not st.flow.end:
                    st.flow.end = _END_CAPTURE
                stats_updates += 1
                yield emit(st, last_ts, _R_CAPTURE, -1, wm_now)
        self.stats["updates"] = stats_updates
        self.stats["packets_in_updates"] = len(pk_flow)

    # ------------------------------------------------------------------ object form
    def updates(self) -> Iterator[StateUpdate]:
        """Yield one `StateUpdate` per emitted update (`emit` mode), in event-time order."""
        adapter = ADAPTER_NAME
        ns, no = ObservationStatus.NOT_SUPPLIED, ObservationStatus.NOT_OBSERVABLE
        obs = ObservationStatus.OBSERVED
        entity_cache: dict[int, tuple[EntityRef, ...]] = {}
        seq = -1
        for (_record, ts, _off, _len, raw_hash, flow, _direction, _ip_len, payload_status, pdigest, watermark,
             _reason, _first, _n) in self._records():
            seq += 1
            v = [_NAN] * _N
            s = [CODE_NOT_SUPPLIED] * _N
            flow.fill(v, s)
            ents = entity_cache.get(flow.index)
            if ents is None:
                a, b, svc = self._entity_keys(flow)
                ents = (EntityRef(*a), EntityRef(*b)) + ((EntityRef(*svc),) if svc else ())
                entity_cache[flow.index] = ents
                if len(entity_cache) > 50000:
                    entity_cache.clear()
                    entity_cache[flow.index] = ents
            fields: dict[str, FieldValue] = {
                "flow.src_ip": FieldValue("flow.src_ip", ents[0].id, obs, adapter),
                "flow.dst_ip": FieldValue("flow.dst_ip", ents[1].id, obs, adapter),
            }
            j = 0
            while j < _N:
                col = COLUMNS[j]
                fid = col.field_id
                code = s[j]
                if col.kind is Kind.HISTOGRAM:
                    if code == CODE_OBSERVED:
                        fields[fid] = FieldValue(fid, tuple(int(x) for x in v[j:j + 8]), obs, adapter)
                    else:
                        fields[fid] = FieldValue(fid, None, ns, adapter)
                    j += 8
                    continue
                if code == CODE_OBSERVED:
                    value: Any = float(v[j]) if col.kind is Kind.CONTINUOUS else int(v[j])
                    fields[fid] = FieldValue(fid, value, obs, adapter)
                elif not fid.startswith("proto."):
                    fields[fid] = FieldValue(fid, None, ns, adapter)
                j += 1
            if payload_status == CODE_OBSERVED:
                fields["app.payload"] = FieldValue("app.payload", pdigest.hex(), obs, adapter)
            elif payload_status == CODE_NOT_OBSERVABLE:
                fields["app.payload"] = FieldValue("app.payload", None, no, adapter)
            if flow.sni >= 0:
                fields["proto.tls.sni"] = FieldValue("proto.tls.sni", self._sni[flow.sni], obs, adapter)
            wm = None if math.isnan(watermark) else watermark
            yield StateUpdate(
                update_id=f"{self.source_id}:{seq}",
                ordering=OrderingInfo(
                    event_time=ts,
                    ingest_time=self.ingest_clock() if self.ingest_clock else ts,
                    watermark=wm,
                    reorder_uncertainty_s=None if wm is None else max(0.0, wm - ts),
                    clock_quality="unknown",
                ),
                entities=ents,
                fields=fields,
                provenance=Provenance(self.source_id, adapter, ADAPTER_VERSION, raw_hash.hex()),
            )

    # ------------------------------------------------------------------ columnar form
    def columnar(self, progress: Callable[[float], None] | None = None) -> ColumnarUpdates:
        """Fill the pandas/NumPy form directly. Same content as `to_columnar(self.updates(), COLUMNS)`."""
        cap = 1 << 16
        values = np.empty((cap, _N), dtype=np.float64)
        status = np.empty((cap, _N), dtype=np.uint8)
        hashes = bytearray()
        rec: list[int] = []
        t_ev: list[float] = []
        t_in: list[float] = []
        wmk: list[float] = []
        e0: list[int] = []
        e1: list[int] = []
        e2: list[int] = []
        rel: list[int] = []
        flw: list[int] = []
        dirn: list[int] = []
        roff: list[int] = []
        rlen: list[int] = []
        iplen: list[int] = []
        pst: list[int] = []
        pdig: list[int] = []
        sni: list[int] = []
        why: list[int] = []
        first: list[int] = []
        npk: list[int] = []

        entity_index: dict[tuple[str, str], int] = {}
        relation_index: dict[tuple[int, int, int], int] = {}
        flow_ents: dict[int, tuple[int, int, int]] = {}
        flow_proto: dict[int, int] = {}
        rel_proto: list[int] = []
        rel_port: list[int] = []
        n = 0
        for (record, ts, off, length, raw_hash, flow, direction, ip_len, payload_status, pdigest, watermark,
             reason, first_record, packets) in self._records(progress):
            if n == cap:
                cap *= 2
                values = np.resize(values, (cap, _N))
                status = np.resize(status, (cap, _N))
            if flow.rel < 0:
                a, b, svc = self._entity_keys(flow)
                ia = entity_index.setdefault(a, len(entity_index))
                ib = entity_index.setdefault(b, len(entity_index))
                ic = entity_index.setdefault(svc, len(entity_index)) if svc else -1
                flow.ents = (ia, ib, ic)
                flow_ents[flow.index] = flow.ents
                flow_proto[flow.index] = flow.proto
                r = relation_index.get(flow.ents)
                if r is None:
                    r = relation_index[flow.ents] = len(relation_index)
                    rel_proto.append(flow.proto)
                    rel_port.append(flow.dport)
                flow.rel = r
            v = [_NAN] * _N
            s = [CODE_NOT_SUPPLIED] * _N
            flow.fill(v, s)
            values[n] = v
            status[n] = s
            hashes += raw_hash
            rec.append(record)
            t_ev.append(ts)
            t_in.append(self.ingest_clock() if self.ingest_clock else ts)
            wmk.append(watermark)
            ents = flow.ents
            e0.append(ents[0])
            e1.append(ents[1])
            e2.append(ents[2])
            rel.append(flow.rel)
            flw.append(flow.index)
            dirn.append(direction)
            roff.append(off)
            rlen.append(length)
            iplen.append(ip_len)
            pst.append(payload_status)
            pdig.append(int.from_bytes(pdigest, "big") if pdigest is not None else 0)
            sni.append(flow.sni)
            why.append(reason)
            first.append(first_record)
            npk.append(packets)
            n += 1
        if n == 0:
            raise InvariantViolation(f"{self.path.name}: no decodable IP or ARP record in the capture.")

        import pandas as pd

        ev = np.asarray(t_ev, dtype=np.float64)
        wm = np.asarray(wmk, dtype=np.float64)
        sni_arr = np.asarray(sni, dtype=np.int32)
        columns: dict[str, Any] = {
            "seq": np.arange(n, dtype=np.int64),
            "record": np.asarray(rec, dtype=np.int64),
            "event_time": ev,
            "ingest_time": np.asarray(t_in, dtype=np.float64),
            "watermark": wm,
            "reorder_uncertainty_s": np.where(np.isnan(wm), np.nan, np.maximum(0.0, wm - ev)),
            "entity_0": np.asarray(e0, dtype=np.int32),
            "entity_1": np.asarray(e1, dtype=np.int32),
            "entity_2": np.asarray(e2, dtype=np.int32),
            "relation": np.asarray(rel, dtype=np.int32),
            "flow": np.asarray(flw, dtype=np.int32),
            "direction": np.asarray(dirn, dtype=np.int8),
            "raw_offset": np.asarray(roff, dtype=np.int64),
            "raw_len": np.asarray(rlen, dtype=np.int32),
        }
        if self.emit == "packet":                         # per-packet IP length (relation byte totals)
            columns["ip_len"] = np.asarray(iplen, dtype=np.int32)
        columns.update({
            "src_ip_status": np.full(n, CODE_OBSERVED, dtype=np.uint8),
            "dst_ip_status": np.full(n, CODE_OBSERVED, dtype=np.uint8),
            "payload_status": np.asarray(pst, dtype=np.uint8),
            "payload_digest": np.asarray(pdig, dtype=np.uint64),
            "tls_sni": sni_arr,
            "tls_sni_status": np.where(sni_arr >= 0, CODE_OBSERVED, CODE_NOT_SUPPLIED).astype(np.uint8),
            "emit_reason": np.asarray(why, dtype=np.int8),
            "first_record": np.asarray(first, dtype=np.int64),
            "packets": np.asarray(npk, dtype=np.int32),
        })
        frame_ = pd.DataFrame(columns)
        entities = pd.DataFrame({"kind": [k for k, _ in entity_index], "key": [i for _, i in entity_index]})
        out = ColumnarUpdates(
            source_id=self.source_id, adapter=ADAPTER_NAME, adapter_version=ADAPTER_VERSION, columns=COLUMNS,
            updates=frame_, entities=entities, relations=pd.DataFrame(),
            values=np.ascontiguousarray(values[:n]), status=np.ascontiguousarray(status[:n]),
            raw_hash=np.frombuffer(bytes(hashes), dtype=np.uint8).reshape(n, 32).copy(),
            clock_quality="unknown", vocab={"proto.tls.sni": list(self._sni)}, side_fields=dict(SIDE_FIELDS),
            explicit_fields=EXPLICIT_FIELDS,
        )
        finalise_tables(out, relation_entities={v: k for k, v in relation_index.items()})
        out.relations["protocol"] = np.asarray(rel_proto, dtype=np.int32)
        out.relations["dst_port"] = np.asarray(rel_port, dtype=np.int32)
        if self.emit != "packet":
            # relation byte totals from each flow's last update (its running totals), summed per relation
            last = frame_.groupby("flow").tail(1).index.to_numpy()
            n_rel = len(out.relations)
            for col, j in (("bytes_fwd", _C_BF), ("bytes_bwd", _C_BB)):
                vals = np.nan_to_num(out.values[last, j])
                out.relations[col] = np.bincount(frame_["relation"].to_numpy()[last], weights=vals, minlength=n_rel)
        self._identity_tables(out, entity_index, flow_ents, flow_proto)
        return out

    def _identity_tables(
        self, out: ColumnarUpdates, entity_index: dict[tuple[str, str], int],
        flow_ents: dict[int, tuple[int, int, int]], flow_proto: dict[int, int],
    ) -> None:
        """Entity columns first_sent, mac, ttl_initial, ttl_since and the fact tables (module docstring).

        Computed from every decodable packet (`self._packets`), not from the emitted updates, so the
        result is the same in both emission modes.
        """
        import pandas as pd

        ent, facts, text = out.entities, self._facts, self._text
        n_ent = len(ent)
        def arr(buf: array[Any], dtype: Any) -> np.ndarray:
            return np.frombuffer(buf, dtype=dtype) if len(buf) else np.zeros(0, dtype=dtype)

        a_flow, a_dir, a_t, a_ttl = self._packets
        pk_flow, pk_dir, pk_t, pk_ttl = arr(a_flow, np.int64), arr(a_dir, np.int8), arr(a_t, np.float64), arr(a_ttl, np.int16)
        ents = np.array([flow_ents[int(i)] for i in pk_flow], dtype=np.int64).reshape(-1, 3)
        proto = np.array([flow_proto[int(i)] for i in pk_flow], dtype=np.int64)
        t = pk_t
        e0, e1, e2 = ents[:, 0], ents[:, 1], ents[:, 2]
        back = pk_dir == 1
        svc = back & (e2 >= 0)
        # a TTL fixed by the protocol says nothing about the sender's system (module docstring)
        receiver = np.where(back, e0, e1)
        to_group = (receiver >= 0) & (ent["kind"].to_numpy()[np.maximum(receiver, 0)] == "multicast")
        ttl = np.where(to_group | (proto == 2) | (proto == 58), -1, pk_ttl.astype(np.int64))
        # who sent each packet: the initiator forward; the responder, and its service, backward
        who = np.concatenate([np.where(back, e1, e0), e2[svc]])
        when = np.concatenate([t, t[svc]])
        hop = np.concatenate([ttl, ttl[svc]]).astype(np.int32)
        has = hop >= 0
        first_sent = np.full(n_ent, np.inf)
        np.minimum.at(first_sent, who, when)
        ttl_since = np.full(n_ent, np.inf)
        np.minimum.at(ttl_since, who[has], when[has])
        ttl_max = np.full(n_ent, -1, dtype=np.int32)
        np.maximum.at(ttl_max, who[has], hop[has])
        ttl_initial = np.full(n_ent, -1, dtype=np.int16)
        for value in reversed(TTL_INITIAL_VALUES):          # the smallest value >= the largest TTL is written last
            ttl_initial[(ttl_max >= 0) & (ttl_max <= value)] = value
        mac_of = {text(ip): mac.hex(":") for ip, mac in facts.ip_mac.items()}
        mac_of.update((text(addr), mac.hex(":")) for addr, mac in facts.ll_mac.items())
        ent["first_sent"] = np.where(np.isinf(first_sent), np.nan, first_sent)
        ent["mac"] = pd.Series([mac_of.get(key, "") if kind in ("host", "external") else ""
                                for kind, key in zip(ent["kind"], ent["key"], strict=True)], dtype=str)
        ent["ttl_initial"] = ttl_initial
        ent["ttl_since"] = np.where(np.isinf(ttl_since), np.nan, ttl_since)

        def entity(addr: bytes) -> int:
            return entity_index.get(self._identity(addr), -1)

        aliases = [(entity_index.get(("host", text(host)), -1), text(addr), mac.hex(":"), since)
                   for addr, host, mac, since in facts.aliases]
        out.aliases = fact_table("aliases", [row for row in aliases if row[0] >= 0])
        roles: list[tuple[int, str, float, str]] = []
        seen: set[tuple[int, str]] = set()
        for addr, role, since in facts.roles:
            e = entity(addr)
            if e >= 0 and (e, role) not in seen:
                seen.add((e, role))
                roles.append((e, role, since, ROLE_EVIDENCE[role]))
        out.roles = fact_table("roles", roles)
        out.names = fact_table("names", [(entity(addr), text(addr), name, source, since)
                                         for addr, name, source, since in facts.names])

    # ------------------------------------------------------------------ one raw record, for audit
    def read_raw(self, raw_offset: int) -> tuple[float, bytes]:
        """Return (capture time, frame bytes) of the record at `raw_offset` of a classic pcap file."""
        with open(self.path, "rb") as fh:
            magic = fh.read(4)
            if magic not in _PCAP_MAGICS:
                raise InvariantViolation("Raw records can be read by offset from classic pcap files only.")
            little = magic in (b"\xd4\xc3\xb2\xa1", b"\x4d\x3c\xb2\xa1")
            nano = magic in (b"\x4d\x3c\xb2\xa1", b"\xa1\xb2\x3c\x4d")
            fh.seek(raw_offset)
            sec, frac, caplen, _wire = struct.unpack("<IIII" if little else ">IIII", fh.read(16))
            return sec + frac / (1e9 if nano else 1e6), fh.read(caplen)


def _dns_qtype(payload: bytes) -> int:
    """Query type of the first question of a DNS message, or -1."""
    try:
        if int.from_bytes(payload[4:6], "big") < 1:
            return -1
        i = 12
        while i < len(payload):
            length = payload[i]
            if length == 0:
                return int.from_bytes(payload[i + 1:i + 3], "big") if i + 3 <= len(payload) else -1
            if length >= 0xC0:                      # a compression pointer ends the name
                return int.from_bytes(payload[i + 2:i + 4], "big") if i + 4 <= len(payload) else -1
            i += 1 + length
    except Exception:
        return -1
    return -1


def _dns_name(payload: bytes, i: int) -> tuple[str, int]:
    """Domain name at offset i (RFC 1035 §4.1.4, with compression) and the offset after it.

    Labels are kept as they appear on the wire; bytes outside ASCII are escaped. Raises ValueError
    on a malformed name (a reserved label type, a pointer loop, or a name past the message).
    """
    labels: list[str] = []
    end = -1
    jumps = 0
    while True:
        length = payload[i]
        if length == 0:
            return ".".join(labels), (i + 1 if end < 0 else end)
        if length >= 0xC0:                              # a pointer to an earlier name
            jumps += 1
            if jumps > 16:
                raise ValueError("DNS name pointer loop")
            if end < 0:
                end = i + 2
            i = ((length & 0x3F) << 8) | payload[i + 1]
            continue
        if length > 63 or i + 1 + length > len(payload):
            raise ValueError("malformed DNS label")
        labels.append(payload[i + 1:i + 1 + length].decode("ascii", "backslashreplace"))
        i += 1 + length


def _dns_skip_name(payload: bytes, i: int) -> int:
    """Offset after the domain name at offset i, without decoding it."""
    while True:
        length = payload[i]
        if length == 0:
            return i + 1
        if length >= 0xC0:
            return i + 2
        if length > 63:
            raise ValueError("malformed DNS label")
        i += 1 + length


def _dns_answers(payload: bytes) -> tuple[str, list[bytes]] | None:
    """Queried name and the A / AAAA addresses answered by a DNS response, or None.

    The name is that of the first question (the name the client asked for, even when the answers
    follow a CNAME chain). A malformed message gives None: nothing is guessed from part of it.
    """
    try:
        qdcount, ancount = int.from_bytes(payload[4:6], "big"), int.from_bytes(payload[6:8], "big")
        if qdcount < 1 or ancount < 1:
            return None
        qname, i = _dns_name(payload, 12)
        i += 4                                          # question type and class
        for _ in range(qdcount - 1):
            i = _dns_skip_name(payload, i) + 4
        n = len(payload)
        out: list[bytes] = []
        for _ in range(ancount):
            i = _dns_skip_name(payload, i)
            if i + 10 > n:
                return None
            rtype = (payload[i] << 8) | payload[i + 1]
            rdlength = (payload[i + 8] << 8) | payload[i + 9]
            i += 10
            if i + rdlength > n:
                return None
            if (rtype == 1 and rdlength == 4) or (rtype == 28 and rdlength == 16):
                out.append(payload[i:i + rdlength])
            i += rdlength
    except (IndexError, ValueError):
        return None
    return (qname, out) if qname and out else None


def _client_hello_sni(payload: bytes) -> str | None:
    """Server name of a TLS ClientHello that starts at the beginning of `payload`, or None."""
    try:
        i = 43                                       # record header 5, handshake header 4, version 2, random 32
        i += 1 + payload[i]                          # session id
        i += 2 + int.from_bytes(payload[i:i + 2], "big")      # cipher suites
        i += 1 + payload[i]                          # compression methods
        end = i + 2 + int.from_bytes(payload[i:i + 2], "big")
        i += 2
        while i + 4 <= min(end, len(payload)):
            ext_type = int.from_bytes(payload[i:i + 2], "big")
            ext_len = int.from_bytes(payload[i + 2:i + 4], "big")
            if ext_type == 0 and ext_len >= 5:
                name_len = int.from_bytes(payload[i + 7:i + 9], "big")
                name = payload[i + 9:i + 9 + name_len]
                return name.decode("ascii") if name and len(name) == name_len else None
            i += 4 + ext_len
    except Exception:
        return None
    return None


__all__ = [
    "ADAPTER_NAME", "ADAPTER_VERSION", "COLUMNS", "DNS_SERVER_MIN_CLIENTS", "ENCRYPTED_PORTS", "EXPLICIT_FIELDS",
    "FLOW_IDLE_TIMEOUT_S", "GATEWAY_MIN_EXTERNAL", "MATRIX_FIELDS", "MAX_NAMES_PER_ADDRESS", "PAYLOAD_BIN_EDGES",
    "PAYLOAD_BIN_LABELS", "PROTOCOL_NAMES", "ROLE_EVIDENCE", "SCAN_MIN_PORTS", "SCAN_WINDOW", "SIDE_FIELDS",
    "STATUS_ORDER", "TTL_INITIAL_VALUES", "PcapSource", "is_capture",
]
