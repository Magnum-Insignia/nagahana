"""Physics residuals r_c: the residual contract, the registry, and the laws of flow accounting.

What a residual is for (D-18)
-----------------------------
A residual measures how far a model output lies outside what is physically possible. Model outputs are
reconstructions (CVG-AE and the Decoder), imagined states (Forecaster), generated variants (Generator)
and the predicted effects of counters (Advisor). A residual is zero inside the boundary and grows with
the violation. Residuals are not attack detectors: attackers obey physics too, so every law here holds
for crafted traffic as well (a packet without an ACK flag still has headers; a scan still serialises at
line rate). Whether residuals on incoming telemetry may inform trust is the held D-25 (`term.py`).

Conventions
-----------
- A residual reads named fields from a mapping of tensors of shape [N] (N rows) and returns r >= 0 of
  shape [N]. Zero means "inside the boundary".
- It declares the fields it needs. The shared term (`term.py`) counts it only on rows where all of those
  fields contribute (the mask m_c), so absent data never creates a fake violation (D-41). Masked rows reach
  the residual as zeros, so every residual is finite, with finite gradients, on all-zero inputs.
- It exposes `bound(x)`, the limit it compares against, used by `physics.normalise.RelativeResidual`
  (AS-15: residuals normalised by their field scale).
- Constants of standards and physics live in `constants.py`; a residual has no site parameter unless the
  law is a site fact (the wire MTU of `MTUBound`).

The laws, by module
-------------------
- here, flow accounting (unconditional facts of IP/TCP flow records): flag counts and per-packet tallies
  <= packets; IP bytes >= packets x the minimum IP header; inter-arrival gaps within the duration; the
  variance of gaps bounded by Popoviciu's inequality; the wire MTU bound for wire captures;
- `network.py`: offload-aware IP size laws (the IP datagram ceiling, the flow's own length range, the
  offload status flag, wire-segment reconstruction), link capacity per flow and per interface,
  propagation delay of request-response timing, conservation at a forwarding node;
- `protocol.py`: segment-header accounting, TCP sequence-space consistency, the TCP state machine within a
  calibrated tolerance, the Mathis throughput relation as a soft bound, OT process limits;
- `queueing.py`: Little's law with its exact finite-window allowance.
Every law is registered in `RESIDUALS` (the package `__init__` imports the law modules).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Protocol

import torch

from nagahana.core.registry import Registry
from nagahana.physics.constants import IPV4_MIN_HEADER


class Residual(Protocol):
    """A physics residual r_c. See the module docstring for the conventions."""

    name: str
    fields: tuple[str, ...]

    def __call__(self, x: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Return r >= 0 of shape [N]; zero inside the physical boundary."""
        ...


RESIDUALS: Registry[Callable[..., Residual]] = Registry("physics residual")

_FLAGS = ("syn", "ack", "fin", "rst", "psh", "urg")


@RESIDUALS.register("flag_count_bound", summary="packets carrying a flag <= packets in the flow")
class FlagCountBound:
    """r = relu(count_f - (packets_fwd + packets_bwd)) for one TCP flag f.

    A packet either carries a flag or not, so the number of packets carrying flag f cannot exceed the
    number of packets. Example hallucination this blocks: a generated SYN flood with more SYN packets than
    packets.
    """

    def __init__(self, flag: str) -> None:
        if flag not in _FLAGS:
            raise ValueError(f"flag must be one of {_FLAGS}")
        self.flag = flag
        self.name: str = f"flag_count_bound.{flag}"
        self.fields: tuple[str, ...] = (f"flow.flag_count.{flag}", "flow.packets_fwd", "flow.packets_bwd")

    def __call__(self, x: Mapping[str, torch.Tensor]) -> torch.Tensor:
        total = x["flow.packets_fwd"] + x["flow.packets_bwd"]
        return torch.relu(x[f"flow.flag_count.{self.flag}"] - total)

    def bound(self, x: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """The limit the residual compares against (packets in the flow), for relative scaling."""
        return x["flow.packets_fwd"] + x["flow.packets_bwd"]


@RESIDUALS.register("mtu_bound", summary="IP bytes in one direction <= packets x the wire MTU (wire captures)")
class MTUBound:
    """r = relu(bytes_d - packets_d * MTU) for direction d in {fwd, bwd}: a wire-capture law.

    On a wire, no IP datagram exceeds the link MTU, so bytes_d <= packets_d * MTU. A capture taken on a host
    with segmentation offload (TSO/GSO on send, GRO/LRO on receive) records coalesced datagrams of up to
    64 KiB, so this bound does not hold there: use it only for wire captures, and use the offload-aware
    laws of `network.py` (the IP datagram ceiling and the flow's own length range, exact at any capture
    point) where the capture point is not known; `network.offload_status` flags the rows that show
    datagrams above the Ethernet MTU. `mtu` is the largest IP datagram on the path, a site fact (1500 on
    standard Ethernet, up to 9000 with jumbo frames), so it is required, not defaulted.
    """

    def __init__(self, direction: str, mtu: float) -> None:
        if direction not in ("fwd", "bwd"):
            raise ValueError("direction must be 'fwd' or 'bwd'")
        if mtu <= 0:
            raise ValueError("mtu must be positive")
        self.direction, self.mtu = direction, float(mtu)
        self.name: str = f"mtu_bound.{direction}"
        self.fields: tuple[str, ...] = (f"flow.bytes_{direction}", f"flow.packets_{direction}")

    def __call__(self, x: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return torch.relu(x[f"flow.bytes_{self.direction}"] - x[f"flow.packets_{self.direction}"] * self.mtu)

    def bound(self, x: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """packets x MTU, for relative scaling."""
        return x[f"flow.packets_{self.direction}"] * self.mtu


@RESIDUALS.register("iat_max_bound", summary="largest inter-arrival gap <= flow duration")
class IATMaxBound:
    """r = relu(iat_max - duration).

    Every inter-arrival gap lies between the first and the last packet, so none can exceed the duration.
    """

    name: str = "iat_max_bound"
    fields: tuple[str, ...] = ("flow.iat_max", "flow.duration")

    def __call__(self, x: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return torch.relu(x["flow.iat_max"] - x["flow.duration"])

    def bound(self, x: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """The flow duration, for relative scaling."""
        return x["flow.duration"]


#: Packet-count fields bounded by the flow's packet count (each counts a subset of the flow's packets).
_PACKET_SUBSET_COUNTS = ("pkt.ip_df_count", "pkt.ip_mf_count", "pkt.retransmissions")


@RESIDUALS.register("count_within_packets", summary="a per-packet count <= packets in the flow")
class CountWithinPackets:
    """r = relu(count - (packets_fwd + packets_bwd)) for DF-flagged, MF-flagged or retransmitted packets.

    Each of these counts a subset of the flow's packets (a retransmitted segment is itself a packet of the
    flow), so none can exceed the number of packets. Unconditional, like `FlagCountBound` (AS-114).
    """

    def __init__(self, field: str) -> None:
        if field not in _PACKET_SUBSET_COUNTS:
            raise ValueError(f"field must be one of {_PACKET_SUBSET_COUNTS}")
        self.field = field
        self.name: str = f"count_within_packets.{field}"
        self.fields: tuple[str, ...] = (field, "flow.packets_fwd", "flow.packets_bwd")

    def __call__(self, x: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return torch.relu(x[self.field] - (x["flow.packets_fwd"] + x["flow.packets_bwd"]))

    def bound(self, x: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return x["flow.packets_fwd"] + x["flow.packets_bwd"]


@RESIDUALS.register("iat_mean_bound", summary="mean inter-arrival gap <= flow duration")
class IATMeanBound:
    """r = relu(iat_mean - duration): the mean of the gaps is at most their maximum, which is at most the
    duration (`IATMaxBound`). Holds for any gap definition that averages gaps inside the flow (AS-114)."""

    name: str = "iat_mean_bound"
    fields: tuple[str, ...] = ("flow.iat_mean", "flow.duration")

    def __call__(self, x: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return torch.relu(x["flow.iat_mean"] - x["flow.duration"])

    def bound(self, x: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return x["flow.duration"]


@RESIDUALS.register("iat_var_bound", summary="variance of gaps <= iat_max^2 / 2")
class IATVarianceBound:
    """r = relu(iat_var - iat_max^2 / 2).

    Gaps g_1 ... g_n lie in [0, M] with M = iat_max. Their population variance is at most M^2 / 4
    (Popoviciu's inequality; the sharper Bhatia-Davis bound sigma^2 <= (M - mu)(mu - m) <= (M - m)^2 / 4,
    Bhatia and Davis, "A Better Bound on the Variance", American Mathematical Monthly 107(4), 2000). The
    sample variance is n / (n - 1) times the population variance, at most 2x for n >= 2, so
    iat_var <= M^2 / 2 holds for both definitions an adapter may use (AS-114).
    """

    name: str = "iat_var_bound"
    fields: tuple[str, ...] = ("flow.iat_var", "flow.iat_max")

    def __call__(self, x: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return torch.relu(x["flow.iat_var"] - 0.5 * x["flow.iat_max"] ** 2)

    def bound(self, x: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return 0.5 * x["flow.iat_max"] ** 2


@RESIDUALS.register("min_header_bound", summary="IP bytes in one direction >= packets x the minimum IP header")
class MinHeaderBound:
    """r = relu(packets_d * h_min - bytes_d) for direction d.

    Every IP packet carries at least its header: 20 bytes for IPv4 (RFC 791), 40 for IPv6 (RFC 8200).
    `flow.bytes_*` are IP-layer bytes, so bytes_d >= h_min * packets_d. The default h_min = 20 is the
    smallest IP header of any version, so the law holds for every flow at every capture point; a site
    known to carry only IPv6 may pass 40.
    """

    def __init__(self, direction: str, min_header: float = IPV4_MIN_HEADER) -> None:
        if direction not in ("fwd", "bwd"):
            raise ValueError("direction must be 'fwd' or 'bwd'")
        if min_header <= 0:
            raise ValueError("min_header must be positive")
        self.direction, self.min_header = direction, float(min_header)
        self.name: str = f"min_header_bound.{direction}"
        self.fields: tuple[str, ...] = (f"flow.bytes_{direction}", f"flow.packets_{direction}")

    def __call__(self, x: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return torch.relu(x[f"flow.packets_{self.direction}"] * self.min_header - x[f"flow.bytes_{self.direction}"])

    def bound(self, x: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return x[f"flow.packets_{self.direction}"] * self.min_header
