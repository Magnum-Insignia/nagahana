"""Protocol laws: header accounting, TCP sequence space, the TCP state machine, the Mathis relation, OT limits.

Every law holds for any traffic at any capture point, crafted traffic included, or is stated with the
tolerance that makes it so (the state machine). Constants are in `constants.py`.

Segment-header accounting (`SegmentHeaderBound`)
------------------------------------------------
Every packet of a direction carries at least an IP header and, for TCP and UDP, a transport header, so the
payload of a direction fits in its IP bytes after those headers:
    payload_d <= bytes_d - h x packets_d,   h = 40 for TCP (IPv4 20 + TCP 20), 28 for UDP (20 + 8), 20 otherwise
(RFC 791, RFC 9293 section 3.1, RFC 768). IPv6 headers are larger, so the IPv4 minima keep the law true for
both versions.

TCP sequence space (`SequenceSpaceBound`)
----------------------------------------
A TCP flow's payload counted in sequence space (as sequence-number-based sensors report it) consists of the
bytes the sensor saw and the bytes it missed in content gaps (`flow.missed_bytes`). The seen part was carried
in IP payload, so
    (payload_fwd + payload_bwd) - missed <= (bytes_fwd + bytes_bwd) - 40 (packets_fwd + packets_bwd),
and the missed bytes lie inside the sequence space: missed <= payload_fwd + payload_bwd. For sensors that
sum the payload of the packets they saw (retransmissions included) the first inequality is the header
accounting of the whole flow and still holds. SYN and FIN consume sequence numbers but carry no payload
(RFC 9293 section 3.4), so they do not enter either side.

TCP state machine (`TCPStateMachine`)
-------------------------------------
In a connection of a conforming stack (RFC 9293):
- at most two segments carry SYN (the SYN and the SYN-ACK) besides retransmissions:
      v_syn = relu(n_SYN - 2 - n_retx);
- at most two carry FIN (one per direction) besides retransmissions:
      v_fin = relu(n_FIN - 2 - n_retx);
- once synchronised every segment carries ACK, so only the initial SYN (and its retransmissions) and RST
  segments may lack it (RFC 9293 sections 3.5, 3.10.7):
      v_ack = relu(n_packets - n_ACK - n_SYN - n_RST).
Crafted traffic breaks these rules on purpose (NULL, FIN and Xmas scans send segments without ACK), and
the physics boundary must not forbid imagining it (D-18). The law is therefore stated within a calibrated
tolerance: r = sum_k relu(v_k - tau_k), with tau_k the empirical quantile (default 0.999) of v_k over the
real TCP records of the training data (`TCPStateMachine.calibrate`, AS-722). Inside the envelope of real
traffic, attacks included, the term is zero; it acts only on outputs beyond anything observed.

Mathis throughput relation (`MathisThroughputBound`, a soft bound)
-----------------------------------------------------------------
For a loss-limited TCP flow in congestion avoidance, Mathis, Semke, Mahdavi and Ott (ACM SIGCOMM CCR
27(3), 1997) give BW = (MSS / RTT) x C / sqrt(p), with C = sqrt(3/2) for one packet per ACK (the largest
constant of their periodic-loss model). The residual is the log-ratio by which a direction's throughput
B = payload / duration exceeds it:
    r = relu( log B - log( MSS C / (RTT sqrt(p)) ) ),   MSS = l_max - 40,   p = n_retx / n_packets
on TCP rows with p > 0 and a positive duration (0 elsewhere: without loss evidence the relation bounds
nothing). It is a soft bound: congestion controls that do not follow the AIMD model (CUBIC, RFC 9438; BBR)
can exceed it, so its weight in Phi_phys is chosen small and it is never a hard limit. RTT comes from a
per-flow round-trip-time field the adapter supplies (`rtt_field`).

OT process limits (`OTProcessLimits`, registry name "ot_process_limits")
-----------------------------------------------------------------------
A physical process obeys its asset's engineering range, its maximum rate of change, and the device's
scan cycle (D-34): for asset limits supplied per row (site OT asset data),
    range:  r = relu(low - x) + relu(x - high)
    rate:   r = relu(|delta x| - rate_max x interval)
    period: r = relu(period_min - period).
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch

from nagahana.core.errors import ConfigMissing, InvariantViolation
from nagahana.physics.constants import IPV4_MIN_HEADER, MATHIS_C, PROTO_TCP, PROTO_UDP, TCP_MIN_HEADER, UDP_HEADER
from nagahana.physics.residuals import RESIDUALS

_DIRECTIONS = ("fwd", "bwd")
_TCP_HEADERS = IPV4_MIN_HEADER + TCP_MIN_HEADER          # 40
_UDP_HEADERS = IPV4_MIN_HEADER + UDP_HEADER              # 28


def _direction(direction: str) -> str:
    if direction not in _DIRECTIONS:
        raise ValueError(f"direction must be one of {_DIRECTIONS}")
    return direction


def _headers(protocol: torch.Tensor) -> torch.Tensor:
    """Minimum header bytes per packet by IP protocol: 40 TCP, 28 UDP, 20 otherwise."""
    dtype = protocol.dtype if protocol.is_floating_point() else torch.float32
    h = torch.full(protocol.shape, IPV4_MIN_HEADER, dtype=dtype, device=protocol.device)
    h = torch.where(protocol == PROTO_TCP, torch.full_like(h, _TCP_HEADERS), h)
    return torch.where(protocol == PROTO_UDP, torch.full_like(h, _UDP_HEADERS), h)


@RESIDUALS.register("segment_header_bound", summary="payload of a direction <= IP bytes - headers x packets")
class SegmentHeaderBound:
    """r = relu(payload_d - (bytes_d - h(protocol) x packets_d)) (module docstring)."""

    def __init__(self, direction: str) -> None:
        self.direction = _direction(direction)
        self.name: str = f"segment_header_bound.{direction}"
        self.fields: tuple[str, ...] = (f"flow.payload_bytes_{direction}", f"flow.bytes_{direction}",
                                        f"flow.packets_{direction}", "flow.protocol")

    def __call__(self, x: Mapping[str, torch.Tensor]) -> torch.Tensor:
        d = self.direction
        room = x[f"flow.bytes_{d}"] - _headers(x["flow.protocol"]) * x[f"flow.packets_{d}"]
        return torch.relu(x[f"flow.payload_bytes_{d}"] - room)

    def bound(self, x: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return x[f"flow.bytes_{self.direction}"]


@RESIDUALS.register("sequence_space", summary="TCP payload in sequence space minus missed bytes fits the seen IP payload")
class SequenceSpaceBound:
    """r = relu((S - m) - (B - 40 P)) + relu(m - S) on TCP rows (module docstring), 0 on other protocols.

    S = payload_fwd + payload_bwd, m = missed bytes, B = bytes_fwd + bytes_bwd, P = packets_fwd + packets_bwd.
    """

    name: str = "sequence_space"
    fields: tuple[str, ...] = ("flow.payload_bytes_fwd", "flow.payload_bytes_bwd", "flow.missed_bytes",
                               "flow.bytes_fwd", "flow.bytes_bwd", "flow.packets_fwd", "flow.packets_bwd",
                               "flow.protocol")

    def __call__(self, x: Mapping[str, torch.Tensor]) -> torch.Tensor:
        seq = x["flow.payload_bytes_fwd"] + x["flow.payload_bytes_bwd"]
        missed = x["flow.missed_bytes"]
        seen = x["flow.bytes_fwd"] + x["flow.bytes_bwd"] - _TCP_HEADERS * (x["flow.packets_fwd"] + x["flow.packets_bwd"])
        r = torch.relu(seq - missed - seen) + torch.relu(missed - seq)
        return torch.where(x["flow.protocol"] == PROTO_TCP, r, torch.zeros_like(r))

    def bound(self, x: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return x["flow.bytes_fwd"] + x["flow.bytes_bwd"]


@RESIDUALS.register("tcp_state_machine", summary="RFC 9293 segment rules within a calibrated tolerance")
class TCPStateMachine:
    """r = sum_k relu(v_k - tau_k) over the measures `MEASURES` on TCP rows (module docstring).

    Parameters
    ----------
    tolerance: measure -> tau_k >= 0, calibrated on real data (`calibrate`); every measure needs one.
    """

    MEASURES: tuple[str, ...] = ("syn_excess", "fin_excess", "unacknowledged")
    name: str = "tcp_state_machine"
    fields: tuple[str, ...] = ("flow.flag_count.syn", "flow.flag_count.ack", "flow.flag_count.fin",
                               "flow.flag_count.rst", "flow.packets_fwd", "flow.packets_bwd", "pkt.retransmissions",
                               "flow.protocol")

    def __init__(self, tolerance: Mapping[str, float]) -> None:
        missing = [m for m in self.MEASURES if m not in tolerance]
        if missing:
            raise ConfigMissing(f"no calibrated tolerance for {missing}; use TCPStateMachine.calibrate on real data")
        extra = sorted(set(tolerance) - set(self.MEASURES))
        if extra:
            raise InvariantViolation(f"unknown measures {extra}; known: {self.MEASURES}")
        for k, v in tolerance.items():
            if not (v >= 0 and math.isfinite(v)):
                raise InvariantViolation(f"tolerance of {k} must be finite and >= 0")
        self.tolerance = {k: float(tolerance[k]) for k in self.MEASURES}

    @staticmethod
    def violations(x: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """The measures v_k [N] of the module docstring (computed on every row; protocol not applied)."""
        retx = x["pkt.retransmissions"]
        packets = x["flow.packets_fwd"] + x["flow.packets_bwd"]
        return {
            "syn_excess": torch.relu(x["flow.flag_count.syn"] - 2.0 - retx),
            "fin_excess": torch.relu(x["flow.flag_count.fin"] - 2.0 - retx),
            "unacknowledged": torch.relu(packets - x["flow.flag_count.ack"] - x["flow.flag_count.syn"]
                                         - x["flow.flag_count.rst"]),
        }

    def __call__(self, x: Mapping[str, torch.Tensor]) -> torch.Tensor:
        v = self.violations(x)
        r = sum(torch.relu(v[k] - self.tolerance[k]) for k in self.MEASURES)
        assert isinstance(r, torch.Tensor)
        return torch.where(x["flow.protocol"] == PROTO_TCP, r, torch.zeros_like(r))

    def bound(self, x: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return x["flow.packets_fwd"] + x["flow.packets_bwd"]

    @classmethod
    def calibrate(cls, values: Mapping[str, torch.Tensor], contributing: Mapping[str, torch.Tensor], *,
                  quantile: float = 0.999) -> TCPStateMachine:
        """tau_k = the `quantile` of v_k over the TCP rows where every field contributes (real training data).

        Rows of other protocols and rows with an absent field never enter the quantile (D-41). With no such
        row the tolerance cannot be calibrated and `ConfigMissing` is raised.
        """
        if not 0.0 < quantile <= 1.0:
            raise ValueError("quantile must lie in (0, 1]")
        ref = next(iter(values.values()))
        mask = torch.ones(ref.shape[0], dtype=torch.bool, device=ref.device)
        for f in cls.fields:
            m = contributing.get(f)
            if m is None or f not in values:
                raise ConfigMissing(f"calibration needs the field {f!r} on the calibration rows")
            mask = mask & m.bool()
        mask = mask & (values["flow.protocol"] == PROTO_TCP)
        if not bool(mask.any()):
            raise ConfigMissing("no contributing TCP rows to calibrate the tolerance on")
        safe = {f: values[f][mask].double() for f in cls.fields}
        v = cls.violations(safe)
        return cls({k: float(torch.quantile(v[k], quantile)) for k in cls.MEASURES})


@RESIDUALS.register("mathis_throughput", summary="soft bound: TCP throughput <= (MSS / RTT) C / sqrt(p)")
class MathisThroughputBound:
    """r = relu(log B - log(MSS C / (RTT sqrt(p)))) on loss-limited TCP rows (module docstring).

    Parameters
    ----------
    direction: "fwd" or "bwd": the direction whose payload throughput is bounded.
    rtt_field: the per-flow round-trip-time field (seconds) the adapter supplies.
    """

    def __init__(self, direction: str, rtt_field: str) -> None:
        self.direction = _direction(direction)
        if not rtt_field:
            raise ConfigMissing("the Mathis bound needs the per-flow round-trip-time field")
        self.rtt_field = rtt_field
        self.name: str = f"mathis_throughput.{direction}"
        self.fields: tuple[str, ...] = (f"flow.payload_bytes_{direction}", "flow.duration", "pkt.ip_len_max",
                                        "pkt.retransmissions", "flow.packets_fwd", "flow.packets_bwd", rtt_field,
                                        "flow.protocol")

    def __call__(self, x: Mapping[str, torch.Tensor]) -> torch.Tensor:
        packets = x["flow.packets_fwd"] + x["flow.packets_bwd"]
        p = x["pkt.retransmissions"] / packets.clamp_min(1.0)
        dur, rtt = x["flow.duration"], x[self.rtt_field]
        active = (x["flow.protocol"] == PROTO_TCP) & (p > 0) & (dur > 0) & (rtt > 0)
        # Safe operands on inactive rows, so neither branch can produce a non-finite value or gradient.
        one = torch.ones_like(p)
        p_s, dur_s, rtt_s = torch.where(active, p, one), torch.where(active, dur, one), torch.where(active, rtt, one)
        payload = torch.where(active, x[f"flow.payload_bytes_{self.direction}"], one)
        mss = (x["pkt.ip_len_max"] - _TCP_HEADERS).clamp_min(1.0)
        log_b = torch.log(payload.clamp_min(1.0)) - torch.log(dur_s)
        log_cap = torch.log(mss) + math.log(MATHIS_C) - torch.log(rtt_s) - 0.5 * torch.log(p_s)
        r = torch.relu(log_b - log_cap)
        return torch.where(active, r, torch.zeros_like(r))

    def bound(self, x: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """The residual is a log-ratio, already free of units: relative scaling leaves it as it is."""
        return torch.zeros_like(x["flow.duration"])


@RESIDUALS.register("ot_process_limits", summary="OT values, rates of change and polling periods within asset limits")
class OTProcessLimits:
    """Asset limits of an OT process (module docstring). The field names are those of the site's OT telemetry.

    Parameters
    ----------
    kind: "range", "rate" or "period".
    value: the value field (range), the change field (rate) or the period field (period).
    low, high: the asset's range fields (range).
    interval, limit: the interval field and the maximum rate-of-change field (rate); for "period", `limit` is
        the minimum period field (the device's scan cycle).
    """

    KINDS = ("range", "rate", "period")

    def __init__(self, kind: str, *, value: str, low: str | None = None, high: str | None = None,
                 interval: str | None = None, limit: str | None = None) -> None:
        if kind not in self.KINDS:
            raise ValueError(f"kind must be one of {self.KINDS}")
        need = {"range": (low, high), "rate": (interval, limit), "period": (limit,)}[kind]
        if not value or any(not f for f in need):
            raise ConfigMissing(f"OT {kind} limits need the field names of the value and of the asset limits")
        self.kind, self.value = kind, value
        self.low, self.high, self.interval, self.limit = low, high, interval, limit
        self.name: str = f"ot_process_limits.{kind}.{value}"
        self.fields: tuple[str, ...] = (value, *(f for f in need if f is not None))

    def __call__(self, x: Mapping[str, torch.Tensor]) -> torch.Tensor:
        v = x[self.value]
        if self.kind == "range":
            assert self.low is not None and self.high is not None
            return torch.relu(x[self.low] - v) + torch.relu(v - x[self.high])
        if self.kind == "rate":
            assert self.interval is not None and self.limit is not None
            return torch.relu(torch.abs(v) - x[self.limit] * x[self.interval])
        assert self.limit is not None
        return torch.relu(x[self.limit] - v)

    def bound(self, x: Mapping[str, torch.Tensor]) -> torch.Tensor:
        if self.kind == "range":
            assert self.low is not None and self.high is not None
            return torch.abs(x[self.high] - x[self.low])
        if self.kind == "rate":
            assert self.interval is not None and self.limit is not None
            return x[self.limit] * x[self.interval]
        assert self.limit is not None
        return x[self.limit]
