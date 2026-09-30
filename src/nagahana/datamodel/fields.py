"""Field catalogue: every feature a state update may carry.

What is decided (D-38, [Q-42])
------------------------------
"at state updates, we take both flow level and packet level features atleast, along with any more
features from the observational region". The flow- and packet-level lists below are taken from the
problem statement in CLAUDE.md ("Input Data: Two Levels of Traffic Feature"), item by item, so the
required coverage is checkable.

What is not decided
-------------------
- The exact **observable-region extras** (DNS, Kerberos, TLS, OT protocols …) are *examples*
  (`example=True`). The full list is an output of pipeline stage 1, "deep data analysis" [Q-02],
  [I-01], together with the data-model standards question (D-06).
- **Source-specific definitions** (which layer "bytes" counts, how a source defines its
  bidirectional ratio) are recorded per adapter in stage 1. Fields state their intended meaning;
  adapters document any difference.

Design notes
------------
- **IDs are stable strings** (`"flow.bytes_fwd"`). The model's input layer keys on them (proposal
  P-22, the owner's "empty/disabled extra neurons" idea [Q-01]), so adding a field never renumbers
  others. See `versioning.py`.
- **Addresses are not features.** IP addresses identify entities and build the hypergraph
  (`Kind.IDENTIFIER`); they are not fed to the model, because raw addresses are volatile,
  high-entropy and say little about the threat (ARCH §3.2 macrostate; [Q-09]).
- **Ports are categorical, never numeric.** `flow.src_port` / `flow.dst_port` are
  `Kind.CATEGORICAL`: embedded as categories (port 445 is not "close to" port 443), and they also
  identify the services entities expose. They must stay features: the problem statement asks which
  "flags, ports, or flow patterns" drive a prediction, and docs/architecture.md lists ports among
  the driving-feature families. (Corrected 2026-09-29: this note used to call ports identifiers,
  which contradicted the catalogue below.)
- **OT is first-class** (D-34, [Q-29]): OT protocol fields sit beside IT fields, not in an
  appendix.
"""

from __future__ import annotations

import enum
from collections.abc import Iterable
from dataclasses import dataclass


class Level(enum.Enum):
    """Where a field comes from."""

    FLOW = "flow"            # NetFlow / IPFIX-style aggregates over one flow
    PACKET = "packet"        # derived from packet captures (PCAP)
    PROTOCOL = "protocol"    # application-protocol metadata (Zeek-tier): DNS, Kerberos, TLS …
    OT = "ot"                # industrial protocols: Modbus, DNP3, IEC 60870-5-104 …
    DERIVED = "derived"      # computed across several records (e.g. port-scan signatures)


class Kind(enum.Enum):
    """How the model should treat a field's value."""

    CONTINUOUS = "continuous"
    COUNT = "count"
    CATEGORICAL = "categorical"
    BITMASK = "bitmask"
    IDENTIFIER = "identifier"      # used for graph construction, never as a numeric feature
    FINGERPRINT = "fingerprint"    # e.g. JA4 strings: categorical with an open vocabulary
    HISTOGRAM = "histogram"


@dataclass(frozen=True)
class FieldSpec:
    """Definition of one field.

    Attributes
    ----------
    id: stable identifier, `"<group>.<name>"`.
    level, kind: see the enums.
    unit: physical unit (SI where possible), or None.
    description: intended meaning, in plain words.
    required_by: where the requirement comes from (problem statement, quote ID, …).
    example: True when the field is an illustrative observable-region extra (not yet decided).
    """

    id: str
    level: Level
    kind: Kind
    unit: str | None
    description: str
    required_by: str
    example: bool = False


_PS = "problem statement (CLAUDE.md), flow-level list"
_PP = "problem statement (CLAUDE.md), packet-level list"
_EX = "D-38 'more features from the observational region' — example, list set in stage 1"

_FIELDS: tuple[FieldSpec, ...] = (
    # ---------------------------------------------------------------- flow level (required)
    FieldSpec("flow.src_ip", Level.FLOW, Kind.IDENTIFIER, None, "Source IP address (entity key).", _PS),
    FieldSpec("flow.dst_ip", Level.FLOW, Kind.IDENTIFIER, None, "Destination IP address (entity key).", _PS),
    FieldSpec("flow.src_port", Level.FLOW, Kind.CATEGORICAL, None, "Source transport port.", _PS),
    FieldSpec("flow.dst_port", Level.FLOW, Kind.CATEGORICAL, None, "Destination transport port.", _PS),
    FieldSpec("flow.protocol", Level.FLOW, Kind.CATEGORICAL, None, "IP protocol number (IANA).", _PS),
    FieldSpec(
        "flow.tcp_flags", Level.FLOW, Kind.BITMASK, None,
        "OR of TCP flags seen in the flow (SYN, ACK, FIN, RST, PSH, URG), NetFlow-style.", _PS,
    ),
    *(
        FieldSpec(
            f"flow.flag_count.{f}", Level.FLOW, Kind.COUNT, "packets",
            f"Number of packets in the flow carrying the {f.upper()} flag (CICFlowMeter-style).", _PS,
        )
        for f in ("syn", "ack", "fin", "rst", "psh", "urg")
    ),
    FieldSpec("flow.bytes_fwd", Level.FLOW, Kind.COUNT, "bytes", "IP-layer bytes, initiator → responder.", _PS),
    FieldSpec("flow.bytes_bwd", Level.FLOW, Kind.COUNT, "bytes", "IP-layer bytes, responder → initiator.", _PS),
    FieldSpec("flow.packets_fwd", Level.FLOW, Kind.COUNT, "packets", "Packets, initiator → responder.", _PS),
    FieldSpec("flow.packets_bwd", Level.FLOW, Kind.COUNT, "packets", "Packets, responder → initiator.", _PS),
    FieldSpec("flow.duration", Level.FLOW, Kind.CONTINUOUS, "s", "Time from first to last packet of the flow.", _PS),
    FieldSpec("flow.iat_mean", Level.FLOW, Kind.CONTINUOUS, "s", "Mean inter-arrival time between packets of the flow.", _PS),
    FieldSpec("flow.iat_var", Level.FLOW, Kind.CONTINUOUS, "s^2", "Variance of inter-arrival times.", _PS),
    FieldSpec("flow.iat_max", Level.FLOW, Kind.CONTINUOUS, "s", "Largest inter-arrival time.", _PS),
    FieldSpec(
        "flow.bidir_ratio", Level.FLOW, Kind.CONTINUOUS, None,
        "Bidirectional ratio as supplied by the source (definition recorded per adapter).", _PS,
    ),
    # -------------------------------------------------------------- packet level (required)
    FieldSpec("pkt.ttl_mean", Level.PACKET, Kind.CONTINUOUS, "hops", "Mean IP TTL across the session.", _PP),
    FieldSpec("pkt.ttl_var", Level.PACKET, Kind.CONTINUOUS, "hops^2", "Variance of IP TTL across the session.", _PP),
    FieldSpec("pkt.tcp_window_init_fwd", Level.PACKET, Kind.COUNT, "bytes", "Initial TCP window, initiator.", _PP),
    FieldSpec("pkt.tcp_window_init_bwd", Level.PACKET, Kind.COUNT, "bytes", "Initial TCP window, responder.", _PP),
    FieldSpec("pkt.ip_df_count", Level.PACKET, Kind.COUNT, "packets", "Packets with the IP Don't-Fragment flag.", _PP),
    FieldSpec("pkt.ip_mf_count", Level.PACKET, Kind.COUNT, "packets", "Packets with the IP More-Fragments flag.", _PP),
    FieldSpec(
        "pkt.payload_size_hist", Level.PACKET, Kind.HISTOGRAM, "bytes",
        "Payload-size distribution (bin edges are set in stage 1).", _PP,
    ),
    FieldSpec("pkt.retransmissions", Level.PACKET, Kind.COUNT, "segments", "TCP retransmissions.", _PP),
    FieldSpec(
        "derived.portscan_sequential", Level.DERIVED, Kind.CONTINUOUS, None,
        "Evidence of sequential port access from one source (definition set in stage 1).", _PP,
    ),
    FieldSpec(
        "derived.portscan_random", Level.DERIVED, Kind.CONTINUOUS, None,
        "Evidence of randomised port access from one source (slow scans evade flow thresholds).", _PP,
    ),
    # ------------------------------------------------ observable-region extras (examples only)
    FieldSpec("proto.dns.qtype", Level.PROTOCOL, Kind.CATEGORICAL, None, "DNS query type.", _EX, example=True),
    FieldSpec("proto.dns.rcode", Level.PROTOCOL, Kind.CATEGORICAL, None, "DNS response code.", _EX, example=True),
    FieldSpec("proto.kerberos.msg_type", Level.PROTOCOL, Kind.CATEGORICAL, None, "Kerberos message type.", _EX, example=True),
    FieldSpec("proto.tls.ja4", Level.PROTOCOL, Kind.FINGERPRINT, None, "JA4 client fingerprint.", _EX, example=True),
    FieldSpec(
        "proto.tls.sni", Level.PROTOCOL, Kind.FINGERPRINT, None,
        "TLS server name; typically NOT_OBSERVABLE under ECH (a depreciating observable, ARCH §13).", _EX, example=True,
    ),
    FieldSpec(
        "app.payload", Level.PROTOCOL, Kind.IDENTIFIER, None,
        "Application payload; NOT_OBSERVABLE when encrypted. Listed so the status is explicit.", _EX, example=True,
    ),
    FieldSpec("ot.modbus.function_code", Level.OT, Kind.CATEGORICAL, None, "Modbus function code.", _EX, example=True),
    FieldSpec("ot.modbus.unit_id", Level.OT, Kind.CATEGORICAL, None, "Modbus unit identifier.", _EX, example=True),
    FieldSpec("ot.modbus.register_start", Level.OT, Kind.COUNT, None, "First register addressed.", _EX, example=True),
    FieldSpec("ot.modbus.register_count", Level.OT, Kind.COUNT, None, "Number of registers addressed.", _EX, example=True),
    FieldSpec("ot.dnp3.function_code", Level.OT, Kind.CATEGORICAL, None, "DNP3 application function code.", _EX, example=True),
    FieldSpec("ot.dnp3.object_group", Level.OT, Kind.CATEGORICAL, None, "DNP3 object group.", _EX, example=True),
    FieldSpec("ot.iec104.type_id", Level.OT, Kind.CATEGORICAL, None, "IEC 60870-5-104 ASDU type identifier.", _EX, example=True),
    FieldSpec("ot.iec104.cot", Level.OT, Kind.CATEGORICAL, None, "IEC 60870-5-104 cause of transmission.", _EX, example=True),
)

CATALOGUE: dict[str, FieldSpec] = {}
for _f in _FIELDS:
    if _f.id in CATALOGUE:
        raise RuntimeError(f"Duplicate field id {_f.id}")
    CATALOGUE[_f.id] = _f


def spec(field_id: str) -> FieldSpec:
    """Look up a field; unknown IDs raise with a hint."""
    try:
        return CATALOGUE[field_id]
    except KeyError:
        raise KeyError(f"Unknown field {field_id!r}; add it to datamodel/fields.py first.") from None


def by_level(level: Level) -> tuple[FieldSpec, ...]:
    """All fields at one level, in catalogue order."""
    return tuple(f for f in _FIELDS if f.level is level)


def required_ids() -> tuple[str, ...]:
    """IDs required by the problem statement (not examples)."""
    return tuple(f.id for f in _FIELDS if not f.example)


def ids(fields: Iterable[FieldSpec]) -> tuple[str, ...]:
    """IDs of the given specs."""
    return tuple(f.id for f in fields)
