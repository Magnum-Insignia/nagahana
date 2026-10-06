"""Constants of IP, TCP, UDP, Ethernet and signal propagation used by the physics laws.

Every constant is a fact of a standard or of physics, never a site parameter, so the laws that use them
hold at every site and at every capture point (D-18, D-59).

IP sizes
--------
- IPv4 header: at least 20 bytes (RFC 791 section 3.1: Internet Header Length >= 5 32-bit words).
- IPv6 fixed header: 40 bytes (RFC 8200 section 3).
- Largest IP datagram: IPv4 Total Length is a 16-bit field counting the header (<= 65,535 bytes,
  RFC 791 section 3.1); IPv6 Payload Length is a 16-bit field counting the bytes after the 40-byte fixed
  header (<= 65,535 + 40 = 65,575 bytes, RFC 8200 section 3). Jumbograms (RFC 2675) need links whose MTU
  exceeds 65,575 bytes and a hop-by-hop option, and are outside the data model. The ceiling below is the
  larger of the two, so it bounds every IP datagram whatever its version, on a wire or in a host-side
  capture with segmentation offload (TSO/GSO coalesce segments into datagrams of at most this size; GRO
  and LRO likewise build at most 64 KiB packets).

Transport headers
-----------------
- TCP header: at least 20 bytes (RFC 9293 section 3.1: Data Offset >= 5).
- UDP header: 8 bytes (RFC 768).
- TCP sequence space: 2^32 (RFC 9293 section 3.4); SYN and FIN each consume one sequence number.

Ethernet
--------
- Standard Ethernet MTU: 1,500 bytes, the largest IP datagram in a standard Ethernet frame (RFC 894).
  An IP datagram larger than this in a capture means either jumbo frames on that link or a host-side
  capture after segmentation offload; `network.offload_status` reports which rows show it.
- Smallest frame on the wire: 64-byte minimum frame + 7-byte preamble + 1-byte start-of-frame delimiter
  + 12-byte (96-bit) inter-frame gap = 84 bytes (IEEE 802.3, clause 4 and clause 3.2).
- Fastest standardised single-interface line rate: 800 Gb/s (IEEE Std 802.3df-2024). It bounds the
  rate of any one flow direction on any one link without knowing the site's links.

Propagation
-----------
- Speed of light in vacuum: c = 299,792,458 m/s, exact by the definition of the metre (SI).
- Light in silica fibre travels at about 2.0e8 m/s. As a lower bound on delay, the bound must use the
  fastest light any standard fibre carries: the group index of silica single-mode fibre at 1310 nm and
  1550 nm is about 1.467-1.468 (citation to verify: Corning SMF-28 product information), and no silica
  fibre has a group index below the refractive index of fused silica at 1550 nm, about 1.444
  (Malitson, JOSA 55(10), 1965). The fibre bound therefore uses v = c / 1.444 (about 2.08e8 m/s), which
  stays below the true delay of every silica fibre path. Paths with free-space segments (microwave,
  hollow-core fibre) need the vacuum speed c.

TCP throughput
--------------
- Mathis, Semke, Mahdavi and Ott, "The macroscopic behavior of the TCP congestion avoidance algorithm",
  ACM SIGCOMM Computer Communication Review 27(3), 1997: BW = (MSS / RTT) * C / sqrt(p), with
  C = sqrt(3 / (2 b)) for b packets acknowledged per ACK under periodic loss; b = 1 gives the largest
  constant, C = sqrt(3 / 2), so it is used for the soft bound.
"""

from __future__ import annotations

import math

IPV4_MIN_HEADER: float = 20.0
IPV6_HEADER: float = 40.0
#: Largest IP datagram without jumbograms (IPv6: 65,535 payload + 40 header bytes).
IP_DATAGRAM_CEILING: float = 65_575.0
TCP_MIN_HEADER: float = 20.0
UDP_HEADER: float = 8.0
TCP_SEQUENCE_SPACE: int = 2**32
ETHERNET_MTU: float = 1_500.0
ETHERNET_MIN_FRAME_ON_WIRE: float = 84.0
MAX_STANDARD_LINE_RATE_BPS: float = 800e9
SPEED_OF_LIGHT: float = 299_792_458.0
FUSED_SILICA_INDEX_1550NM: float = 1.444
#: Fastest light in silica fibre (m/s): the propagation-delay bound of fibre paths.
FIBRE_SPEED_BOUND: float = SPEED_OF_LIGHT / FUSED_SILICA_INDEX_1550NM
#: Nominal speed of light in fibre (m/s), for reports only: about 2.0e8.
FIBRE_SPEED_NOMINAL: float = 2.0e8
#: Mathis et al. 1997 constant for one packet per ACK under periodic loss.
MATHIS_C: float = math.sqrt(1.5)

#: IANA IP protocol numbers.
PROTO_TCP: int = 6
PROTO_UDP: int = 17
#: IANA ifType of Ethernet interfaces (ethernetCsmacd).
IFTYPE_ETHERNET: int = 6
