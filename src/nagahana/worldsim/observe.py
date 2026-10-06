"""Observation models: hidden sessions become telemetry records (P-14, AS-809, AS-810).

The jump process produces ground-truth sessions (a benign flow, an attacker action expanded into a
sweep or a flood, an OT polling cycle). A sensor turns a session into a record of the NagaHana data
model, or fails to, according to what that kind of sensor can see:

    tap        a packet tap: flow-level and packet-level fields observed; encrypted payload is
               observable-but-unreadable (NOT_OBSERVABLE); flow-state or packet granularity (D-51).
    netflow    a NetFlow/IPFIX exporter: flow-level fields only (packet-level NOT_SUPPLIED), with
               1-in-n sampling and active/inactive timeouts that split a long flow into several.
    zeek       a Zeek-style logger: flow-level plus application metadata (DNS, TLS), no packet-level.
    authlog    host authentication logs: identity and remote-admin sessions only, with the service
               and endpoints but no byte or packet counts.
    ids        an intrusion-detection sensor: no flow record; it raises alerts with a per-technique
               detection probability and a Poisson false-alarm rate (reported as a ground-truth table,
               since an alert is not a field of the data model).

A session is seen only by sensors whose coverage includes one of its endpoints (partial coverage);
of those, the most capable not-dropped sensor supplies the record (a superset, never duplicated,
datamodel.md item 6), and host logs fill identity-session gaps where no flow sensor saw them. Dropped
sessions (packet loss, no coverage) leave no record at all, so a blind spot is a real absence, not a
zero (D-41). Every supplied field carries its observation status (P-03); a field the chosen sensor
cannot supply is NOT_SUPPLIED, one on an untapped path is NOT_OBSERVABLE, and a skewed clock shifts
the record's event time (the ground-truth time is kept separately).

All randomness is the counter-based generator keyed by the world and the session index, so the
records are reproducible and identical no matter which backend produced the trajectory (AS-801).
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field as dc_field
from typing import Any

import numpy as np

from nagahana.datamodel.records import EntityRef, FieldValue, OrderingInfo, Provenance, StateUpdate
from nagahana.datamodel.status import ObservationStatus
from nagahana.worldsim import rng, vocab
from nagahana.worldsim.config import ScenarioConfig, SensorConfig
from nagahana.worldsim.params import STREAM_OBS_BASE, STREAM_SESSION
from nagahana.worldsim.state import KIND_ATTACK, KIND_BENIGN
from nagahana.worldsim.topology import Topology

OBS = ObservationStatus.OBSERVED
NS = ObservationStatus.NOT_SUPPLIED
NO = ObservationStatus.NOT_OBSERVABLE
LR = ObservationStatus.LOW_RELIABILITY

_PROTO_NAME = {6: "tcp", 17: "udp", 132: "sctp"}
_FLAGS = ("syn", "ack", "fin", "rst", "psh", "urg")
#: IP + transport header bytes added per packet so IP-layer byte counts obey the physics bounds.
_HEADER_BYTES = 40
_PAYLOAD_EDGES = (0, 1, 64, 128, 256, 512, 1024, 1460)

#: Sensor capability rank: the most capable covering sensor supplies a session's record.
_RANK = {"tap_packet": 5, "tap_flow-state": 4, "zeek": 3, "netflow": 2, "authlog": 1}

#: Which catalogue fields each sensor kind can supply (others are NOT_SUPPLIED for that sensor).
_FLOW_CORE = (
    "flow.src_port", "flow.dst_port", "flow.protocol", "flow.bytes_fwd", "flow.bytes_bwd",
    "flow.packets_fwd", "flow.packets_bwd", "flow.packets_total", "flow.payload_bytes_fwd",
    "flow.payload_bytes_bwd", "flow.duration", "flow.iat_mean", "flow.iat_var", "flow.iat_max",
    "flow.bidir_ratio", "flow.tcp_flags", "flow.end_reason", "flow.unanswered",
)
_FLAG_COUNTS = tuple(f"flow.flag_count.{f}" for f in _FLAGS)
_PACKET_FIELDS = (
    "pkt.ttl_mean", "pkt.ttl_var", "pkt.tcp_window_init_fwd", "pkt.tcp_window_init_bwd",
    "pkt.ip_df_count", "pkt.ip_mf_count", "pkt.payload_size_hist", "pkt.retransmissions",
)
_PROTO_FIELDS = ("proto.dns.qtype", "proto.dns.rcode", "proto.icmp.type")
_OT_FIELDS = ("ot.modbus.function_code", "ot.modbus.register_start", "ot.modbus.register_count")
_SCAN_FIELDS = ("derived.portscan_sequential", "derived.portscan_random")

#: Catalogue fields the world simulator can populate (the column set of its ColumnarUpdates).
WORLDSIM_FIELDS: tuple[str, ...] = (
    *_FLOW_CORE, *_FLAG_COUNTS, *_PACKET_FIELDS, *_SCAN_FIELDS, *_PROTO_FIELDS, *_OT_FIELDS,
)

#: Nominal per-service flow shape: (mean payload bytes initiator->responder, response/request ratio,
#: typical packet size, is_tcp). Byte sizes are drawn heavy-tailed around the mean (AS-804).
_SERVICE_SHAPE: dict[str, tuple[float, float, int, bool]] = {
    "http": (800.0, 8.0, 1200, True),
    "https": (1500.0, 10.0, 1300, True),
    "dns": (60.0, 2.0, 120, False),
    "smb": (4000.0, 3.0, 1300, True),
    "kerberos": (1400.0, 1.5, 700, True),
    "ldap": (600.0, 2.0, 400, True),
    "rdp": (6000.0, 4.0, 900, True),
    "ssh": (3000.0, 1.2, 600, True),
    "winrm": (1200.0, 3.0, 800, True),
    "smtp": (2000.0, 0.3, 1000, True),
    "modbus": (24.0, 1.0, 64, True),
    "dnp3": (40.0, 1.0, 72, True),
    "iec104": (36.0, 1.0, 70, True),
    "s7comm": (48.0, 1.0, 80, True),
    "ethernet_ip": (60.0, 1.0, 90, True),
    "historian_api": (1800.0, 6.0, 1200, True),
}


@dataclass
class Session:
    """One ground-truth session (the physical event a sensor may observe)."""

    t_us: int
    initiator: int            # entity row
    responder: int            # entity row
    service: int              # service code, or -1 (no application service)
    proto: int                # IP protocol number
    dst_port: int             # responder port, or -1
    kind: int                 # KIND_BENIGN or KIND_ATTACK
    sem: int                  # SEM_* for attacker sessions, else -1
    technique: int            # technique code, or -1
    stage: int                # model stage code
    cause: int                # parent event slot (ground-truth attack DAG)
    actor_role: int           # which endpoint performs the labelled stage (0 initiator, 1 responder, -1)
    session_index: int        # unique index (RNG address and audit)


@dataclass
class ObservedRecord:
    """A session as seen by one sensor: the data-model update plus its aligned ground-truth labels."""

    update: StateUpdate
    malicious: float
    stage: int
    technique: str
    family: str
    actor_role: int
    responder_entity: int     # global entity row of the responder (for the episode target)
    cause: int


@dataclass
class IDSAlert:
    """One intrusion-detection alert (not a data-model field; reported as ground truth)."""

    t_us: int
    sensor: str
    entity: int
    true_alert: bool
    technique: str


@dataclass
class ObservationResult:
    """Everything the sensor fabric produced for one world."""

    records: list[ObservedRecord] = dc_field(default_factory=list)
    sessions: list[Session] = dc_field(default_factory=list)
    alerts: list[IDSAlert] = dc_field(default_factory=list)


def _proto_name(proto: int) -> str:
    return _PROTO_NAME.get(proto, str(proto))


def _payload_hist(total_bytes: int, packets: int) -> tuple[int, ...]:
    # Spread a flow's payload across the size histogram bins by a typical packet size.
    bins = [0] * 8
    if packets <= 0:
        return tuple(bins)
    per = max(total_bytes // max(packets, 1), 0)
    b = 0
    for i, edge in enumerate(_PAYLOAD_EDGES):
        if per >= edge:
            b = i
    bins[b] = packets
    return tuple(bins)


def _sessions_from_events(scenario: ScenarioConfig, topo: Topology, events: np.ndarray, wkey: tuple[Any, Any]) -> list[Session]:
    # Expand the event timeline into ground-truth sessions (fan-out for sweeps and floods).
    sk = rng.stream_key(np, wkey, STREAM_SESSION)
    reach = topo.reach
    sessions: list[Session] = []
    idx = 0
    used = events[events[:, 0] > 0]
    for row in used:
        kind, t_us, ini, resp, svc, tech, stage, sem, fan, cause = (int(x) for x in row)
        if kind not in (KIND_BENIGN, KIND_ATTACK):
            continue                                            # defender actions produce no traffic here
        proto, dst_port = _service_proto_port(svc)
        actor = _actor_role(sem)
        if fan <= 1:
            sessions.append(Session(t_us, ini, resp, svc, proto, dst_port, kind, sem, tech, stage,
                                    cause, actor, idx))
            idx += 1
            continue
        # A sweep or flood: `fan` sessions from the initiator to distinct reachable responders/ports.
        targets = np.nonzero(reach[ini])[0]
        if targets.size == 0:
            targets = np.array([resp], dtype=np.int64)
        for j in range(int(fan)):
            pick = int(rng.randint(np, sk, idx, 0, int(targets.size)))
            dst = int(targets[pick])
            port = int(rng.randint(np, sk, idx, 1, 1024)) + 1 if sem != vocab.SEM_DOS else dst_port
            sessions.append(Session(t_us + j, ini, dst, -1, 6, port, kind, sem, tech, stage, cause, actor, idx))
            idx += 1
    sessions.sort(key=lambda s: (s.t_us, s.session_index))
    return sessions


def _ot_polling_sessions(scenario: ScenarioConfig, topo: Topology, start_index: int) -> list[Session]:
    # Deterministic OT polling: each control host polls each reachable controller every interval (AS-805).
    interval = scenario.benign.ot_poll_interval_s
    horizon = scenario.simulation.horizon_s
    if interval <= 0 or horizon <= 0:
        return []
    control_rows = np.nonzero(topo.control & topo.internal)[0]
    plc_rows = np.nonzero(topo.is_ot_device)[0]
    if control_rows.size == 0 or plc_rows.size == 0:
        return []
    modbus = vocab.SERVICE_CODE["modbus"]
    out: list[Session] = []
    idx = start_index
    n_ticks = int(horizon / interval)
    for src in control_rows:
        for dst in plc_rows:
            if not topo.reach[src, dst]:
                continue
            svc = modbus if topo.exposes[dst, modbus] else int(np.argmax(topo.exposes[dst]))
            proto, port = _service_proto_port(svc)
            for k in range(n_ticks):
                t_us = int(round((k * interval) * 1_000_000))
                out.append(Session(t_us, int(src), int(dst), svc, proto, port, KIND_BENIGN, -1, -1, 0, -1, -1, idx))
                idx += 1
    return out


def _service_proto_port(svc: int) -> tuple[int, int]:
    if svc < 0:
        return 6, -1
    s = vocab.SERVICES[svc]
    return s.protocol, s.port


def _actor_role(sem: int) -> int:
    # The internal actor that performs the labelled stage. For moves that act on a controlled host
    # (the attacker's own foothold) the initiator is the actor; for moves against a remote victim the
    # responder is the entity that becomes compromised.
    if sem < 0:
        return -1
    if sem in (vocab.SEM_SCAN_EXTERNAL, vocab.SEM_SCAN_INTERNAL, vocab.SEM_DOS):
        return 0
    if sem in (vocab.SEM_EXPLOIT_SERVICE, vocab.SEM_VALID_ACCOUNT, vocab.SEM_OT_COMMAND):
        return 1
    return 0


def _family_of(sem: int, scenario: ScenarioConfig) -> str:
    return vocab.campaign(scenario.attack.campaign).family if sem >= 0 else "benign"


def _feature_values(sess: Session, key: tuple[Any, Any]) -> dict[str, float]:
    # Synthesize the flow and packet features of a session from its service shape (heavy-tailed sizes).
    i = sess.session_index
    svc_name = vocab.SERVICES[sess.service].name if sess.service >= 0 else ""
    mean_b, ratio, pkt_size, _tcp = _SERVICE_SHAPE.get(svc_name, (200.0, 1.0, 400, True))
    scan = sess.sem in (vocab.SEM_SCAN_EXTERNAL, vocab.SEM_SCAN_INTERNAL)
    dos = sess.sem == vocab.SEM_DOS
    if scan:
        pf, pb = 1, 0
        bf, bb = 0, 0
        dur = 0.0
    elif dos:
        pf = int(rng.randint(np, key, i, 2, 2000)) + 200
        pb = 0
        bf, bb = pf * 64, 0
        dur = float(rng.unit(np, key, i, 3)) * 2.0
    else:
        bf = int(rng.bounded_pareto(np, key, i, 4, scenario_alpha(), mean_b * 0.2 + 1.0, mean_b * 50.0 + 2.0))
        bb = int(bf * ratio * (0.5 + float(rng.unit(np, key, i, 5))))
        pf = max(1, bf // max(pkt_size, 1) + 1)
        pb = max(1, bb // max(pkt_size, 1) + 1) if bb > 0 else 0
        dur = float(rng.lognormal(np, key, i, 6, mu=0.0, sigma=1.0))
    packets = pf + pb
    # IP-layer bytes add a per-packet header to the transport payload, so the records satisfy the
    # physics boundary (bytes >= packets * min header and <= packets * MTU; AS-815, physics/residuals.py).
    ip_fwd = bf + pf * _HEADER_BYTES
    ip_bwd = bb + pb * _HEADER_BYTES
    iat_mean = dur / max(packets - 1, 1) if packets > 1 else 0.0
    iat_max = min(iat_mean * (1.5 + 2.0 * float(rng.unit(np, key, i, 7))), dur)   # a gap cannot exceed the duration
    iat_var = (iat_max * 0.3) ** 2                                                # <= iat_max^2 / 2
    hops = int(rng.randint(np, key, i, 8, 6))
    ttl = float(64 - hops) if sess.initiator % 2 == 0 else float(128 - hops)
    out: dict[str, float] = {
        "flow.bytes_fwd": float(ip_fwd), "flow.bytes_bwd": float(ip_bwd),
        "flow.payload_bytes_fwd": float(bf), "flow.payload_bytes_bwd": float(bb),
        "flow.packets_fwd": float(pf), "flow.packets_bwd": float(pb), "flow.packets_total": float(packets),
        "flow.duration": dur, "flow.iat_mean": iat_mean, "flow.iat_var": iat_var, "flow.iat_max": iat_max,
        "pkt.ttl_mean": ttl, "pkt.ttl_var": 0.0,
        "pkt.tcp_window_init_fwd": 64240.0, "pkt.tcp_window_init_bwd": 64240.0,
        "pkt.ip_df_count": float(pf), "pkt.ip_mf_count": 0.0, "pkt.retransmissions": 0.0,
    }
    if ip_fwd > 0:
        out["flow.bidir_ratio"] = ip_bwd / max(ip_fwd, 1)
    return out


def scenario_alpha() -> float:
    """Tail index used by the heavy-tailed session-size sampler (kept here for the feature synthesis)."""
    return 1.6


def _tcp_flags(sess: Session) -> tuple[int, dict[str, int]]:
    # The TCP flag bitmask and per-flag counts for a session.
    if sess.proto != 6:
        return 0, {f: 0 for f in _FLAGS}
    scan = sess.sem in (vocab.SEM_SCAN_EXTERNAL, vocab.SEM_SCAN_INTERNAL)
    if scan:
        return 0x02, {"syn": 1, "ack": 0, "fin": 0, "rst": 0, "psh": 0, "urg": 0}
    mask = 0x02 | 0x10 | 0x08 | 0x01            # SYN, ACK, PSH, FIN of a normal connection
    return mask, {"syn": 1, "ack": 2, "fin": 1, "rst": 0, "psh": 1, "urg": 0}


def _end_reason(sess: Session) -> tuple[int, int]:
    # (end_reason code, unanswered flag). Scans are unanswered SYNs (idle end).
    if sess.sem in (vocab.SEM_SCAN_EXTERNAL, vocab.SEM_SCAN_INTERNAL):
        return 3, 1                              # idle, unanswered
    if sess.sem == vocab.SEM_DOS:
        return 2, 1                              # reset, often unanswered
    return 1, 0                                  # clean FIN close


def _supplied_fields(sensor: SensorConfig) -> set[str]:
    rank = _sensor_rank(sensor)
    if rank >= _RANK["tap_flow-state"]:
        return set(WORLDSIM_FIELDS)
    if sensor.kind == "zeek":
        return set(_FLOW_CORE) | set(_FLAG_COUNTS) | set(_PROTO_FIELDS) | set(_OT_FIELDS) | set(_SCAN_FIELDS)
    if sensor.kind == "netflow":
        return set(_FLOW_CORE) | set(_SCAN_FIELDS)
    if sensor.kind == "authlog":
        return {"flow.src_port", "flow.dst_port", "flow.protocol"}
    return set()


def _sensor_rank(sensor: SensorConfig) -> int:
    if sensor.kind == "tap":
        return _RANK["tap_packet"] if sensor.granularity == "packet" else _RANK["tap_flow-state"]
    return _RANK.get(sensor.kind, 0)


def _covers(sensor: SensorConfig, topo: Topology, sess: Session) -> bool:
    dom_i = vocab.ENTERPRISE if topo.domain[sess.initiator] == 0 else vocab.OT
    dom_r = vocab.ENTERPRISE if topo.domain[sess.responder] == 0 else vocab.OT
    return (dom_i in sensor.coverage) or (dom_r in sensor.coverage)


def observe_world(scenario: ScenarioConfig, topo: Topology, events: np.ndarray, wkey: tuple[Any, Any],
                  adapter_version: str) -> ObservationResult:
    """Run the sensor fabric over one world's event timeline and return the records and ground truth."""
    result = ObservationResult()
    sessions = _sessions_from_events(scenario, topo, events, wkey)
    ot = _ot_polling_sessions(scenario, topo, start_index=10_000_000)
    sessions = sorted([*sessions, *ot], key=lambda s: (s.t_us, s.session_index))
    result.sessions = sessions

    sensors = scenario.observation.sensors
    flow_sensors = [(s, _sensor_rank(s)) for s in sensors if s.kind != "ids"]
    ids_sensors = [s for s in sensors if s.kind == "ids"]
    skey = rng.stream_key(np, wkey, STREAM_SESSION)

    for sess in sessions:
        chosen = _choose_sensor(scenario, topo, sess, flow_sensors, skey)
        if chosen is not None:
            result.records.extend(_build_record(scenario, topo, sess, chosen, skey, adapter_version))

    # IDS alerts (ground-truth table, not data-model records).
    for si, sensor in enumerate(ids_sensors):
        _ids_alerts(scenario, topo, sessions, sensor, si, wkey, result)
    result.records.sort(key=lambda r: r.update.ordering.event_time)
    return result


def _choose_sensor(scenario: ScenarioConfig, topo: Topology, sess: Session,
                   flow_sensors: list[tuple[SensorConfig, int]], skey: tuple[Any, Any]) -> SensorConfig | None:
    # The most capable covering sensor that did not drop the session; identity sessions may fall back
    # to a host log when no flow sensor covered them (datamodel.md item 6).
    best: SensorConfig | None = None
    best_rank = -1
    for si, (sensor, rank) in enumerate(flow_sensors):
        if not _covers(sensor, topo, sess):
            continue
        if sess.session_index % sensor.sampling_n != 0:          # 1-in-n sampling (deterministic stride)
            continue
        key = rng.stream_key(np, skey, STREAM_OBS_BASE + si)
        if sensor.packet_loss > 0.0 and bool(rng.bernoulli(np, key, sess.session_index, 0, rng.threshold_u32(sensor.packet_loss))):
            continue
        if sensor.kind == "authlog" and vocab.SERVICES[sess.service].plane not in ("identity", "remote_admin") if sess.service >= 0 else sensor.kind == "authlog":
            continue
        if rank > best_rank:
            best, best_rank = sensor, rank
    return best


#: A NetFlow exporter emits at most this many active-timeout records for one long flow.
_MAX_FLOW_SLICES = 8


def _slice_plan(sess: Session, sensor: SensorConfig, duration: float) -> list[tuple[float, float, int, bool]]:
    # (slice duration, fraction of the flow, slice index, is_last). A NetFlow exporter splits a flow
    # longer than its active timeout into several export records; every other sensor emits one.
    if sensor.kind != "netflow" or duration <= sensor.active_timeout_s or duration <= 0:
        return [(duration, 1.0, 0, True)]
    n = min(int(np.ceil(duration / sensor.active_timeout_s)), _MAX_FLOW_SLICES)
    plan: list[tuple[float, float, int, bool]] = []
    remaining = duration
    for k in range(n):
        is_last = k == n - 1
        d = remaining if is_last else sensor.active_timeout_s
        plan.append((d, d / duration, k, is_last))
        remaining -= d
    return plan


def _build_record(scenario: ScenarioConfig, topo: Topology, sess: Session, sensor: SensorConfig,
                  skey: tuple[Any, Any], adapter_version: str) -> list[ObservedRecord]:
    si = scenario.observation.sensors.index(sensor)
    key = rng.stream_key(np, skey, STREAM_OBS_BASE + si)
    supplied = _supplied_fields(sensor)
    vals = _feature_values(sess, key)
    mask, flag_counts = _tcp_flags(sess)
    end_reason, unanswered = _end_reason(sess)
    status_of = _status_fn(sensor)

    src_ref = EntityRef(str(topo.kind[sess.initiator]), str(topo.address[sess.initiator]))
    dst_ref = EntityRef(str(topo.kind[sess.responder]), str(topo.address[sess.responder]))
    base_entities = [src_ref, dst_ref]
    if sess.dst_port >= 0 and sess.proto in (6, 17, 132):
        base_entities.append(EntityRef("service", f"{dst_ref.id}:{sess.dst_port}/{_proto_name(sess.proto)}"))
    src_port = _ephemeral(sess, key)
    base_time = _observed_time(sess, sensor)
    malicious = 1.0 if sess.kind == KIND_ATTACK else 0.0
    technique = vocab.TECHNIQUES[sess.technique].attack_id if sess.technique >= 0 else ""

    out: list[ObservedRecord] = []
    for d_slice, frac, k, is_last in _slice_plan(sess, sensor, float(vals.get("flow.duration", 0.0))):
        fields: dict[str, FieldValue] = {}

        def put(fid: str, value: Any, applies: bool = True, fields: dict[str, FieldValue] = fields) -> None:
            if not applies:
                return
            st = status_of(fid)
            if st in (NS, NO):
                if st is NO:
                    fields[fid] = FieldValue(fid, None, NO, sensor.kind)
                return
            rel = sensor.reliability if st is LR else None
            fields[fid] = FieldValue(fid, value, st, sensor.kind, reliability=rel)

        fields["flow.src_ip"] = FieldValue("flow.src_ip", src_ref.id, OBS, sensor.kind)
        fields["flow.dst_ip"] = FieldValue("flow.dst_ip", dst_ref.id, OBS, sensor.kind)
        # Per-slice packet and byte counts keep the IP-layer bounds (payload plus a per-packet header).
        pf = max(1, int(round(vals["flow.packets_fwd"] * frac)))
        pb = int(round(vals["flow.packets_bwd"] * frac))
        payf = int(round(vals["flow.payload_bytes_fwd"] * frac))
        payb = int(round(vals["flow.payload_bytes_bwd"] * frac))
        bf, bb = payf + pf * _HEADER_BYTES, payb + pb * _HEADER_BYTES
        packets_total = pf + pb
        if sess.dst_port >= 0:
            put("flow.src_port", src_port)
            put("flow.dst_port", sess.dst_port)
        put("flow.protocol", sess.proto)
        if sess.proto == 6:
            put("flow.tcp_flags", mask)
            for f in _FLAGS:
                # SYN opens the flow (first slice), FIN/RST close it (last slice); a flag count is
                # bounded by the slice's packet count (physics FlagCountBound).
                c = flag_counts[f]
                if f == "syn" and k > 0:
                    c = 0
                if f in ("fin", "rst") and not is_last:
                    c = 0
                put(f"flow.flag_count.{f}", min(c, packets_total))
        put("flow.bytes_fwd", bf)
        put("flow.bytes_bwd", bb)
        put("flow.packets_fwd", pf)
        put("flow.packets_bwd", pb)
        put("flow.packets_total", packets_total)
        put("flow.payload_bytes_fwd", payf)
        put("flow.payload_bytes_bwd", payb)
        put("flow.duration", d_slice)
        iat_mean = d_slice / max(packets_total - 1, 1) if packets_total > 1 else 0.0
        put("flow.iat_mean", iat_mean)
        put("flow.iat_max", min(vals.get("flow.iat_max", 0.0), d_slice))
        put("flow.iat_var", min(vals.get("flow.iat_var", 0.0), 0.5 * min(vals.get("flow.iat_max", 0.0), d_slice) ** 2))
        put("flow.bidir_ratio", bb / max(bf, 1))
        # Only the closing slice carries an end reason; earlier active-timeout exports leave the flow open.
        put("flow.end_reason", end_reason, applies=is_last)
        put("flow.unanswered", unanswered, applies=is_last)
        for fid in _PACKET_FIELDS:
            if fid == "pkt.payload_size_hist":
                if fid in supplied:
                    st = status_of(fid)
                    if st not in (NS, NO):
                        fields[fid] = FieldValue(fid, _payload_hist(payf + payb, packets_total), st,
                                                 sensor.kind, reliability=sensor.reliability if st is LR else None)
                continue
            if fid == "pkt.ip_df_count":
                put(fid, pf)
            elif fid in ("pkt.ttl_mean", "pkt.ttl_var"):
                put(fid, vals[fid])
            elif fid in vals:
                put(fid, int(vals[fid]))
        if sess.sem in (vocab.SEM_SCAN_EXTERNAL, vocab.SEM_SCAN_INTERNAL):
            put("derived.portscan_sequential", 0.2)
            put("derived.portscan_random", 0.8)
        _protocol_fields(put, sess)

        t = base_time + k * sensor.active_timeout_s
        upd = StateUpdate(
            update_id=f"{scenario.network}:{sess.session_index}:{k}",
            ordering=OrderingInfo(
                event_time=t, ingest_time=t, watermark=None,
                reorder_uncertainty_s=abs(sensor.clock_skew_s) or None,
                clock_quality="skewed" if sensor.clock_skew_s or sensor.clock_drift_ppm else "ntp-synced",
            ),
            entities=tuple(base_entities), fields=fields,
            provenance=Provenance(scenario.network, sensor.kind, adapter_version, None),
        )
        out.append(ObservedRecord(
            update=upd, malicious=malicious, stage=sess.stage, technique=technique,
            family=_family_of(sess.sem, scenario), actor_role=sess.actor_role,
            responder_entity=sess.responder, cause=sess.cause,
        ))
    return out


def _ephemeral(sess: Session, key: tuple[Any, Any]) -> int:
    return 1024 + int(rng.randint(np, key, sess.session_index, 20, 64511))


def _observed_time(sess: Session, sensor: SensorConfig) -> float:
    t = sess.t_us / 1_000_000.0
    return t + sensor.clock_skew_s + sensor.clock_drift_ppm * 1e-6 * t


def _status_fn(sensor: SensorConfig) -> Any:
    supplied = _supplied_fields(sensor)
    low = sensor.reliability > 0.0

    def status_of(fid: str) -> ObservationStatus:
        if fid not in supplied:
            # Packet-level fields that a flow exporter structurally lacks are NOT_SUPPLIED (not zero).
            return NS
        return LR if low else OBS

    return status_of


def _protocol_fields(put: Any, sess: Session) -> None:
    if sess.service < 0:
        return
    name = vocab.SERVICES[sess.service].name
    if name == "dns":
        put("proto.dns.qtype", 1)
        put("proto.dns.rcode", 0)
    elif name in ("modbus",):
        put("ot.modbus.function_code", 3 if sess.sem < 0 else 6)   # 3 read, 6 write-single (manipulation)
        put("ot.modbus.register_start", 0)
        put("ot.modbus.register_count", 10)


def _ids_alerts(scenario: ScenarioConfig, topo: Topology, sessions: list[Session], sensor: SensorConfig,
                si: int, wkey: tuple[Any, Any], result: ObservationResult) -> None:
    key = rng.stream_key(np, wkey, STREAM_OBS_BASE + 100 + si)
    thr = rng.threshold_u32(sensor.detection_prob)
    for sess in sessions:
        if sess.kind != KIND_ATTACK or sess.technique < 0:
            continue
        if not _covers(sensor, topo, sess):
            continue
        noise = vocab.TECHNIQUES[sess.technique].noise
        if bool(rng.bernoulli(np, key, sess.session_index, 0, int(thr * noise))):
            result.alerts.append(IDSAlert(sess.t_us, sensor.kind, sess.responder, True,
                                          vocab.TECHNIQUES[sess.technique].attack_id))
    # False alarms: a Poisson number over the horizon, attached to random benign sessions.
    benign = [s for s in sessions if s.kind == KIND_BENIGN and _covers(sensor, topo, s)]
    if benign:
        rate = sensor.false_alarm_per_hour * (scenario.simulation.horizon_s / 3600.0)
        n_false = _poisson(key, 7, rate)
        for j in range(n_false):
            pick = int(rng.randint(np, key, j, 8, len(benign)))
            s = benign[pick]
            result.alerts.append(IDSAlert(s.t_us, sensor.kind, s.responder, False, ""))


def _poisson(key: tuple[Any, Any], lane: int, mean: float) -> int:
    # Knuth's method; `mean` is small (a few per hour), so the loop is short.
    if mean <= 0:
        return 0
    limit = np.exp(-mean)
    k = 0
    p = 1.0
    while True:
        p *= float(rng.unit(np, key, k, lane))
        if p <= limit:
            return k
        k += 1
        if k > 10000:
            return k


__all__ = [
    "IDSAlert", "ObservationResult", "ObservedRecord", "Session", "WORLDSIM_FIELDS", "observe_world",
]
