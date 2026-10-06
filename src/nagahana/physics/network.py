"""Network-physics laws: offload-aware IP sizes, link capacity, propagation delay, conservation (D-18, D-59).

Each law holds for every flow at every capture point, attack traffic included, so it bounds what the model
may produce without forbidding anything an attacker can do. Constants are in `constants.py`.

IP sizes, offload-aware
-----------------------
A host-side capture with segmentation offload records coalesced datagrams: TSO/GSO hand the NIC segments of
up to 64 KiB that it cuts into MSS-sized wire segments after the capture point, and GRO/LRO merge received
segments before it. Such captures show datagrams far above the wire MTU, so the wire bound
bytes <= packets x MTU (`residuals.MTUBound`) is false for them. The laws below are exact at any capture
point:

- `DatagramCeilingBound`: bytes_d <= packets_d x C_IP, with C_IP = 65,575 bytes, the largest IP datagram
  of either version without jumbograms (RFC 791, RFC 8200). It needs no length evidence.
- `IPLengthBound`: with the flow's own smallest and largest IP datagram (`pkt.ip_len_min`,
  `pkt.ip_len_max`), packets_d x l_min <= bytes_d <= packets_d x l_max, and 20 <= l_min <= l_max <= C_IP.
  Both directions are inside the flow's range, so the law holds per direction.
- `offload_status`: a per-row status flag, 1 where datagrams above the Ethernet MTU (RFC 894) were seen
  (host-side offload or jumbo frames: the wire MTU bound must not be applied), 0 where every datagram fit a
  standard Ethernet frame, -1 where the record carries no length evidence. Consumers apply
  `MTUBound` only on rows with status 0.
- `wire_segment_bounds`: the wire segments a host-side capture corresponds to. A super-segment of length l
  becomes ceil(l / MSS) wire segments (one when l = 0), so for P captured packets carrying B payload bytes
      max(P, ceil(B / MSS)) <= wire segments <= P + floor(B / MSS)
  (the upper bound because ceil(l / M) <= floor(l / M) + 1 and the sum of floors is at most the floor of
  the sum). Mechanism: the Linux kernel documentation, "Segmentation Offloads"
  (Documentation/networking/segmentation-offloads.rst).

Capacity
--------
- `LinkCapacityBound`: a direction of a flow crosses one link at the observation point, at most at its line
  rate R, and the duration runs from the first to the last packet, so everything after the first datagram
  (at most C_IP bytes) was serialised within it:
      bytes_d <= (R / 8) x duration + C_IP
  R defaults to the fastest standardised single interface, 800 Gb/s (IEEE Std 802.3df-2024), which bounds
  every site; a known site line rate gives a tighter bound. For sampled exports whose counts are rescaled
  by the sampling interval n (`flow.sampling_rate`), the law holds for bytes / n (`sampled=True`).
- `InterfaceCapacityBound`: an interface counter over an interval: octets <= (speed / 8) x interval plus one
  maximal frame at the interval boundary (C_IP + 84 bytes). `ifInOctets` counts framing octets too
  (RFC 2863), and the allowance covers them.
- `InterfacePacketRateBound`: on an Ethernet interface (IANA ifType 6) a frame occupies at least 84 bytes of
  wire time (64-byte minimum frame, preamble, delimiter, inter-frame gap; IEEE 802.3), so
  packets <= speed x interval / (8 x 84) + 1.

Propagation (`PropagationDelayBound`, registry name "rtt_floor")
---------------------------------------------------------------
A response cannot reach the observation point before a signal has covered the path to the responder and
back: rtt >= 2 d / v, with d the path distance from the observation point to the responder (site
topology data, per row, `DISTANCE_FIELD`) and v the fastest signal on the medium: c / 1.444 for silica
fibre (no silica fibre carries light faster; nominally about 2.0e8 m/s), c for paths with free-space
segments. The default timing field is the DNS response time (`proto.dns.rtt`).

Conservation (`FlowConservation`)
---------------------------------
At a store-and-forward node over an interval, bytes are conserved: in - out - dropped = Q(t1) - Q(t0),
the change of the queued bytes, so |in - out - dropped| <= B with B the node's buffer bytes. The
aggregates come from `node_balance` (flow legs entering and leaving each node, site forwarding data) or from
interface counters summed per device, and are passed under the field names `NODE_*`.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch

from nagahana.core.errors import InvariantViolation
from nagahana.physics.constants import (
    ETHERNET_MIN_FRAME_ON_WIRE,
    ETHERNET_MTU,
    FIBRE_SPEED_BOUND,
    IFTYPE_ETHERNET,
    IP_DATAGRAM_CEILING,
    IPV4_MIN_HEADER,
    MAX_STANDARD_LINE_RATE_BPS,
    SPEED_OF_LIGHT,
)
from nagahana.physics.residuals import RESIDUALS

#: Per-row path distance (m) from the observation point to the responder: site topology data.
DISTANCE_FIELD = "site.path_distance_m"
#: Per-node aggregates over an interval (bytes), for `FlowConservation` (produced by `node_balance`).
NODE_IN, NODE_OUT, NODE_DROPPED, NODE_BUFFER = "node.bytes_in", "node.bytes_out", "node.bytes_dropped", "node.buffer_bytes"

_DIRECTIONS = ("fwd", "bwd")


def _direction(direction: str) -> str:
    if direction not in _DIRECTIONS:
        raise ValueError(f"direction must be one of {_DIRECTIONS}")
    return direction


@RESIDUALS.register("datagram_ceiling", summary="IP bytes of a direction <= packets x the largest IP datagram (any capture point)")
class DatagramCeilingBound:
    """r = relu(bytes_d - packets_d x C_IP), C_IP = 65,575 bytes (module docstring)."""

    def __init__(self, direction: str) -> None:
        self.direction = _direction(direction)
        self.name: str = f"datagram_ceiling.{direction}"
        self.fields: tuple[str, ...] = (f"flow.bytes_{direction}", f"flow.packets_{direction}")

    def __call__(self, x: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return torch.relu(x[f"flow.bytes_{self.direction}"] - IP_DATAGRAM_CEILING * x[f"flow.packets_{self.direction}"])

    def bound(self, x: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return IP_DATAGRAM_CEILING * x[f"flow.packets_{self.direction}"]


@RESIDUALS.register("ip_length_bound", summary="IP bytes of a direction within packets x the flow's datagram length range")
class IPLengthBound:
    """r = relu(bytes_d - P_d l_max) + relu(P_d l_min - bytes_d) + relu(l_min - l_max) + relu(l_max - C_IP) + relu(20 - l_min).

    P_d: packets of direction d; l_min, l_max: the flow's smallest and largest IP total length. Exact at any
    capture point (module docstring); the last three terms keep the length range itself possible.
    """

    def __init__(self, direction: str) -> None:
        self.direction = _direction(direction)
        self.name: str = f"ip_length_bound.{direction}"
        self.fields: tuple[str, ...] = (f"flow.bytes_{direction}", f"flow.packets_{direction}", "pkt.ip_len_min",
                                        "pkt.ip_len_max")

    def __call__(self, x: Mapping[str, torch.Tensor]) -> torch.Tensor:
        b, p = x[f"flow.bytes_{self.direction}"], x[f"flow.packets_{self.direction}"]
        lo, hi = x["pkt.ip_len_min"], x["pkt.ip_len_max"]
        return (torch.relu(b - p * hi) + torch.relu(p * lo - b) + torch.relu(lo - hi)
                + torch.relu(hi - IP_DATAGRAM_CEILING) + torch.relu(IPV4_MIN_HEADER - lo))

    def bound(self, x: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return x[f"flow.packets_{self.direction}"] * x["pkt.ip_len_max"]


def offload_status(values: Mapping[str, torch.Tensor], contributing: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """Per-row status int8 [N]: 1 datagrams above the Ethernet MTU seen, 0 none above, -1 no length evidence.

    The largest datagram (`pkt.ip_len_max`) decides when it contributes; otherwise a direction whose mean
    datagram (bytes / packets) exceeds the Ethernet MTU proves datagrams above it (the mean is at most the
    maximum), and a record without either stays unknown (absence is not evidence, D-41).
    """
    ref = next(iter(values.values()))
    n = int(ref.shape[0])
    status = torch.full((n,), -1, dtype=torch.int8, device=ref.device)

    def has(f: str) -> torch.Tensor:
        m = contributing.get(f)
        return m.bool() if (m is not None and f in values) else torch.zeros(n, dtype=torch.bool, device=ref.device)

    for d in _DIRECTIONS:
        b, p = f"flow.bytes_{d}", f"flow.packets_{d}"
        ok = has(b) & has(p)
        if bool(ok.any()):
            mean = values[b] / values[p].clamp_min(1.0)
            status = torch.where(ok & (values[p] > 0) & (mean > ETHERNET_MTU), torch.ones_like(status), status)
    lmax_ok = has("pkt.ip_len_max")
    if bool(lmax_ok.any()):
        above = values["pkt.ip_len_max"] > ETHERNET_MTU
        status = torch.where(lmax_ok, above.to(torch.int8), status)
    return status


def wire_segment_bounds(payload_bytes: torch.Tensor, packets: torch.Tensor, mss: float) -> tuple[torch.Tensor, torch.Tensor]:
    """(lower, upper) bounds on the wire segments of `packets` captured super-segments carrying `payload_bytes`.

        lower = max(P, ceil(B / MSS)),   upper = P + floor(B / MSS)       (module docstring)
    """
    if not mss > 0:
        raise ValueError("mss must be positive")
    lo = torch.maximum(packets, torch.ceil(payload_bytes / mss))
    hi = packets + torch.floor(payload_bytes / mss)
    return lo, hi


@RESIDUALS.register("link_capacity", summary="bytes of a flow direction <= line rate x duration + one datagram")
class LinkCapacityBound:
    """r = relu(bytes_d / n - (R / 8) x duration - C_IP) (module docstring; n = 1 unless `sampled`).

    Parameters
    ----------
    direction: "fwd" or "bwd".
    line_rate_bps: R in bit/s; the default is the fastest standardised interface (no site parameter).
    sampled: read `flow.sampling_rate` and bound the unscaled count bytes / n (exports rescaled by 1-in-n).
    """

    def __init__(self, direction: str, line_rate_bps: float = MAX_STANDARD_LINE_RATE_BPS, *, sampled: bool = False) -> None:
        self.direction = _direction(direction)
        if not line_rate_bps > 0:
            raise ValueError("line_rate_bps must be positive")
        self.rate, self.sampled = float(line_rate_bps), bool(sampled)
        self.name: str = f"link_capacity.{direction}" + (".sampled" if sampled else "")
        base = (f"flow.bytes_{direction}", "flow.duration")
        self.fields: tuple[str, ...] = (*base, "flow.sampling_rate") if sampled else base

    def _bytes(self, x: Mapping[str, torch.Tensor]) -> torch.Tensor:
        b = x[f"flow.bytes_{self.direction}"]
        return b / x["flow.sampling_rate"].clamp_min(1.0) if self.sampled else b

    def __call__(self, x: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return torch.relu(self._bytes(x) - self.rate / 8.0 * x["flow.duration"] - IP_DATAGRAM_CEILING)

    def bound(self, x: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return self.rate / 8.0 * x["flow.duration"] + IP_DATAGRAM_CEILING


@RESIDUALS.register("interface_capacity", summary="interface octets over an interval <= speed x interval + one frame")
class InterfaceCapacityBound:
    """r = relu(octets - (speed / 8) x interval - (C_IP + 84)) for the in or out counter of an interface."""

    def __init__(self, direction: str) -> None:
        if direction not in ("in", "out"):
            raise ValueError("direction must be 'in' or 'out'")
        self.direction = direction
        self.name: str = f"interface_capacity.{direction}"
        self.fields: tuple[str, ...] = (f"dev.if_{direction}_octets", "dev.interval", "dev.if_speed")

    def __call__(self, x: Mapping[str, torch.Tensor]) -> torch.Tensor:
        allowance = IP_DATAGRAM_CEILING + ETHERNET_MIN_FRAME_ON_WIRE
        return torch.relu(x[f"dev.if_{self.direction}_octets"] - x["dev.if_speed"] / 8.0 * x["dev.interval"] - allowance)

    def bound(self, x: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return x["dev.if_speed"] / 8.0 * x["dev.interval"] + IP_DATAGRAM_CEILING + ETHERNET_MIN_FRAME_ON_WIRE


@RESIDUALS.register("interface_packet_rate", summary="frames on an Ethernet interface <= speed x interval / (8 x 84) + 1")
class InterfacePacketRateBound:
    """r = relu(packets - speed x interval / (8 x 84) - 1) on Ethernet interfaces (ifType 6); 0 on others."""

    def __init__(self, direction: str) -> None:
        if direction not in ("in", "out"):
            raise ValueError("direction must be 'in' or 'out'")
        self.direction = direction
        self.name: str = f"interface_packet_rate.{direction}"
        self.fields: tuple[str, ...] = (f"dev.if_{direction}_packets", "dev.interval", "dev.if_speed", "dev.if_type")

    def _cap(self, x: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return x["dev.if_speed"] * x["dev.interval"] / (8.0 * ETHERNET_MIN_FRAME_ON_WIRE) + 1.0

    def __call__(self, x: Mapping[str, torch.Tensor]) -> torch.Tensor:
        r = torch.relu(x[f"dev.if_{self.direction}_packets"] - self._cap(x))
        return torch.where(x["dev.if_type"] == IFTYPE_ETHERNET, r, torch.zeros_like(r))

    def bound(self, x: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return self._cap(x)


@RESIDUALS.register("rtt_floor", summary="request-response time >= 2 x path distance / signal speed")
class PropagationDelayBound:
    """r = relu(2 d / v - rtt) (module docstring).

    Parameters
    ----------
    rtt_field: the response-time field (seconds), default the DNS response time.
    distance_field: the per-row path distance in metres (site topology data).
    medium: "fibre" (v = c / 1.444) or "free-space" (v = c).
    """

    MEDIA = {"fibre": FIBRE_SPEED_BOUND, "free-space": SPEED_OF_LIGHT}

    def __init__(self, rtt_field: str = "proto.dns.rtt", distance_field: str = DISTANCE_FIELD, medium: str = "fibre") -> None:
        if medium not in self.MEDIA:
            raise ValueError(f"medium must be one of {sorted(self.MEDIA)}")
        self.rtt_field, self.distance_field, self.speed = rtt_field, distance_field, float(self.MEDIA[medium])
        self.name: str = f"rtt_floor.{medium}"
        self.fields: tuple[str, ...] = (rtt_field, distance_field)

    def floor_seconds(self, distance_m: torch.Tensor) -> torch.Tensor:
        """2 d / v: the shortest possible round trip over the path."""
        return 2.0 * distance_m / self.speed

    def __call__(self, x: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return torch.relu(self.floor_seconds(x[self.distance_field]) - x[self.rtt_field])

    def bound(self, x: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return self.floor_seconds(x[self.distance_field])


@RESIDUALS.register("flow_conservation", summary="bytes conserved at a forwarding node within its buffer")
class FlowConservation:
    """r = relu(|in - out - dropped| - buffer) over node-interval aggregates (module docstring)."""

    name: str = "flow_conservation"
    fields: tuple[str, ...] = (NODE_IN, NODE_OUT, NODE_DROPPED, NODE_BUFFER)

    def __call__(self, x: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return torch.relu(torch.abs(x[NODE_IN] - x[NODE_OUT] - x[NODE_DROPPED]) - x[NODE_BUFFER])

    def bound(self, x: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return x[NODE_IN] + x[NODE_BUFFER]


def node_balance(leg_bytes: torch.Tensor, ingress: torch.Tensor, egress: torch.Tensor, n_nodes: int) -> dict[str, torch.Tensor]:
    """Per-node bytes in and out from flow legs: {NODE_IN: [n], NODE_OUT: [n]} (differentiable in leg_bytes).

    leg_bytes [L]: bytes of each observed flow leg; ingress [L]: the node the leg enters (-1: none, e.g. it
    starts at an end host); egress [L]: the node the leg leaves (-1: none). The site's forwarding data says
    which legs traverse which node (for example the inside and outside legs of a translated flow at a
    gateway, observed by two taps).
    """
    if not (leg_bytes.shape == ingress.shape == egress.shape):
        raise InvariantViolation("leg_bytes, ingress and egress must share a shape [L]")
    nodes = torch.cat([ingress, egress])
    if n_nodes < 0 or (nodes.numel() and int(nodes.max()) >= n_nodes):
        raise InvariantViolation("node index outside [0, n_nodes)")
    zeros = torch.zeros(n_nodes, dtype=leg_bytes.dtype, device=leg_bytes.device)
    in_ok, out_ok = ingress >= 0, egress >= 0
    total_in = zeros.index_add(0, ingress[in_ok].long(), leg_bytes[in_ok])
    total_out = zeros.index_add(0, egress[out_ok].long(), leg_bytes[out_ok])
    return {NODE_IN: total_in, NODE_OUT: total_out}
