"""Physically consistent synthetic windows of state updates, for Generator tests and smoke runs.

Why it exists
-------------
`testing/synthetic.py` builds tensor batches whose values "carry no network meaning". The Generator's
transforms and hard limits need the opposite: windows whose fields obey IP/TCP accounting exactly,
so that a test can show a transform *keeps* the boundary. This builder simulates packets and derives
every field from them with the catalogue's definitions:

    duration = t_last − t_first;  IAT over the flow's packets in time order (mean, population variance,
    max); bytes = Σ IP lengths per direction; flag counts = packets carrying the flag; tcp_flags = OR;
    TTL mean / population variance; initial window = first packet of each side; DF / MF / retransmission
    tallies; payload-size histogram over both directions; bidir_ratio = bytes_bwd / bytes_fwd.

Two record granularities, as in the real adapters:
- "flow": one row per flow (CSV / NetFlow form), at the flow's first packet time;
- "packet": one row per packet carrying the flow's running state (the PCAP adapter's form, with a
  `flow`, `direction` and `ip_len` column). Statistics that do not exist yet (a variance before the
  second packet, the responder's window before its first packet) are NOT_SUPPLIED (D-41).

Some packet-level cells are hidden (NOT_SUPPLIED / NOT_OBSERVABLE, NaN), per row at flow granularity
and per flow at packet granularity (so running tallies stay whole). UDP flows have no TCP fields.

It is *not* a traffic model (the ground-truth world simulator is proposal P-14); it makes no claim
about realism beyond physical consistency. Labels: an attacker host scans (reconnaissance) and then
connects to SMB (lateral movement); everything else is benign; a few benign rows are left unknown.

Decisions: D-41 (absence is a status, NaN never a value). Assumptions: none (test fixture only).
Invariants: `ColumnarUpdates.validate()` holds and every hard limit of `limits.py` holds at the given
MTU (`test_every_transform_preserves_labels_absence_and_limits` checks the source first).
Extension point: more protocols (UDP-only services, ICMP) by extending `_packets` / `_aggregate`.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

from nagahana.datamodel.columnar import STATUS_CODE, ColumnarUpdates, columns_for, finalise_tables
from nagahana.datamodel.fields import CATALOGUE, Level
from nagahana.datamodel.status import ObservationStatus
from nagahana.models.generator.variants import UpdateLabels
from nagahana.models.vocab import STAGE_CODE

SYNTH_FIELDS: tuple[str, ...] = (
    "flow.src_port", "flow.dst_port", "flow.protocol", "flow.tcp_flags",
    "flow.flag_count.syn", "flow.flag_count.ack", "flow.flag_count.fin", "flow.flag_count.rst",
    "flow.flag_count.psh", "flow.flag_count.urg",
    "flow.bytes_fwd", "flow.bytes_bwd", "flow.packets_fwd", "flow.packets_bwd",
    "flow.duration", "flow.iat_mean", "flow.iat_var", "flow.iat_max", "flow.bidir_ratio",
    "pkt.ttl_mean", "pkt.ttl_var", "pkt.tcp_window_init_fwd", "pkt.tcp_window_init_bwd",
    "pkt.ip_df_count", "pkt.ip_mf_count", "pkt.payload_size_hist", "pkt.retransmissions",
)
PAYLOAD_BINS: tuple[str, ...] = ("0", "1-255", "256-1023", "1024+")
_BIN_EDGES = (0, 1, 256, 1024)                                    # lower edges of the bins above
_FLAG_BIT = {"fin": 0x01, "syn": 0x02, "rst": 0x04, "psh": 0x08, "ack": 0x10, "urg": 0x20}
CODE_OBS = STATUS_CODE[ObservationStatus.OBSERVED]
CODE_NS = STATUS_CODE[ObservationStatus.NOT_SUPPLIED]
CODE_NO = STATUS_CODE[ObservationStatus.NOT_OBSERVABLE]


@dataclass
class _Packet:
    t: float
    fwd: bool
    ip_len: int
    flags: int
    ttl: int
    window: int
    df: bool
    mf: bool
    retrans: bool


def _packets(rng: np.random.Generator, *, tcp: bool, mtu: int, n: int) -> list[_Packet]:
    # One flow's packets: first forward, then random directions; TCP opens with SYN / SYN+ACK.
    t = float(rng.exponential(0.02))
    out: list[_Packet] = []
    ttl_f, ttl_b = int(rng.choice([64, 128])) - int(rng.integers(0, 8)), int(rng.choice([64, 128, 255])) - int(rng.integers(0, 8))
    for k in range(n):
        fwd = True if k == 0 else (False if (tcp and k == 1) else bool(rng.random() < 0.55))
        hdr = 40 if tcp else 28
        ip_len = int(rng.integers(hdr, mtu + 1)) if rng.random() < 0.6 else hdr
        if tcp:
            flags = _FLAG_BIT["syn"] if k == 0 else (_FLAG_BIT["syn"] | _FLAG_BIT["ack"] if k == 1 else _FLAG_BIT["ack"])
            if k > 1 and ip_len > hdr and rng.random() < 0.5:
                flags |= _FLAG_BIT["psh"]
            if k == n - 1 and n > 3:
                flags |= _FLAG_BIT["fin"]
        else:
            flags = 0
        out.append(_Packet(t, fwd, ip_len, flags, ttl_f if fwd else ttl_b,
                           int(rng.integers(1024, 65536)) if tcp else -1, bool(rng.random() < 0.9),
                           False, bool(tcp and k > 2 and rng.random() < 0.05)))
        t += float(rng.exponential(0.05))
    return out


def _aggregate(pk: list[_Packet], *, tcp: bool, ports: tuple[int, int, int]) -> dict[str, float | list[float] | None]:
    # Every field from the packets seen so far (None = NOT_SUPPLIED: not applicable / not yet existing).
    fwd = [p for p in pk if p.fwd]
    bwd = [p for p in pk if not p.fwd]
    times = np.array([p.t for p in pk])
    gaps = np.diff(times)
    ttl = np.array([p.ttl for p in pk], dtype=float)
    bytes_f, bytes_b = float(sum(p.ip_len for p in fwd)), float(sum(p.ip_len for p in bwd))
    hdr = 40 if tcp else 28
    hist = [0.0] * len(PAYLOAD_BINS)
    for p in pk:
        pay = p.ip_len - hdr
        b = max(i for i, e in enumerate(_BIN_EDGES) if pay >= e)
        hist[b] += 1.0
    agg: dict[str, float | list[float] | None] = {
        "flow.src_port": float(ports[0]), "flow.dst_port": float(ports[1]), "flow.protocol": float(ports[2]),
        "flow.bytes_fwd": bytes_f, "flow.bytes_bwd": bytes_b,
        "flow.packets_fwd": float(len(fwd)), "flow.packets_bwd": float(len(bwd)),
        "flow.duration": float(times[-1] - times[0]),
        "flow.iat_mean": float(gaps.mean()) if len(gaps) else None,
        "flow.iat_var": float(gaps.var()) if len(gaps) else None,
        "flow.iat_max": float(gaps.max()) if len(gaps) else None,
        "flow.bidir_ratio": bytes_b / bytes_f if bytes_f > 0 else None,
        "pkt.ttl_mean": float(ttl.mean()), "pkt.ttl_var": float(ttl.var()) if len(pk) > 1 else None,
        "pkt.ip_df_count": float(sum(p.df for p in pk)), "pkt.ip_mf_count": float(sum(p.mf for p in pk)),
        "pkt.payload_size_hist": hist, "pkt.retransmissions": float(sum(p.retrans for p in pk)),
    }
    if tcp:
        agg["flow.tcp_flags"] = float(np.bitwise_or.reduce([p.flags for p in pk]))
        for name, bit in _FLAG_BIT.items():
            agg[f"flow.flag_count.{name}"] = float(sum(1 for p in pk if p.flags & bit))
        agg["pkt.tcp_window_init_fwd"] = float(fwd[0].window) if fwd else None
        agg["pkt.tcp_window_init_bwd"] = float(bwd[0].window) if bwd else None
    else:
        for f in ("flow.tcp_flags", *(f"flow.flag_count.{n}" for n in _FLAG_BIT), "pkt.tcp_window_init_fwd",
                  "pkt.tcp_window_init_bwd"):
            agg[f] = None
    return agg


def make_flow_window(
    *,
    n_flows: int = 40,
    n_hosts: int = 8,
    seed: int = 0,
    granularity: str = "flow",
    attack: bool = True,
    hidden_rate: float = 0.1,
    mtu: int = 1500,
    max_packets: int = 12,
) -> tuple[ColumnarUpdates, UpdateLabels]:
    """A window of `n_flows` flows among `n_hosts` internal hosts and one external address."""
    if granularity not in ("flow", "packet"):
        raise ValueError("granularity must be 'flow' or 'packet'")
    rng = np.random.default_rng(seed)
    cols = columns_for(SYNTH_FIELDS, histogram_bins={"pkt.payload_size_hist": PAYLOAD_BINS})
    col_of: dict[str, list[int]] = {}
    for j, c in enumerate(cols):
        col_of.setdefault(c.field_id, []).append(j)
    packet_cols = [j for j, c in enumerate(cols) if CATALOGUE[c.field_id].level is Level.PACKET]

    # ---- entities: hosts, one external, services keyed "addr:port/proto" (as the PCAP adapter)
    ent_kind = ["host"] * n_hosts + ["external"]
    ent_key = [f"10.0.0.{i + 1}" for i in range(n_hosts)] + ["203.0.113.7"]
    service_index: dict[str, int] = {}

    def service(resp: int, port: int, proto: str) -> int:
        key = f"{ent_key[resp]}:{port}/{proto}"
        if key not in service_index:
            service_index[key] = len(ent_kind)
            ent_kind.append("service")
            ent_key.append(key)
        return service_index[key]

    # ---- flows: benign background + (optionally) an attacker scanning then moving laterally
    attacker = 1
    start = np.cumsum(rng.exponential(2.0, n_flows))
    rows_v: list[np.ndarray] = []
    rows_s: list[np.ndarray] = []
    meta: list[dict[str, object]] = []
    mal: list[float] = []
    stage: list[int] = []
    ev_times: list[float] = []
    for f in range(n_flows):
        is_attack = attack and f % 4 == 1
        if is_attack:
            i, r = attacker, int(rng.choice([h for h in range(n_hosts) if h != attacker]))
            late = f > n_flows // 2
            dport = 445 if late else int(rng.choice([22, 80, 135, 139, 445, 3389, 8080]))
            st = STAGE_CODE["lateral_movement" if late else "reconnaissance"]
            tcp, n_pk = True, int(rng.integers(1, 4)) if not late else int(rng.integers(4, max_packets + 1))
        else:
            i, r = (int(x) for x in rng.choice(n_hosts + 1, 2, replace=False))
            dport = int(rng.choice([53, 80, 443, 445, 8080, 8008, 139]))
            tcp = dport != 53
            st, n_pk = STAGE_CODE["none"], int(rng.integers(1, max_packets + 1))
        proto = 6 if tcp else 17
        sport = int(rng.integers(49152, 65536))
        ents = (i, r, service(r, dport, "tcp" if tcp else "udp"))
        label = 1.0 if is_attack else (float("nan") if rng.random() < 0.05 else 0.0)
        pk = _packets(rng, tcp=tcp, mtu=mtu, n=n_pk)
        for p in pk:
            p.t += float(start[f])
        flow_hidden = rng.random(len(cols)) < hidden_rate                  # per-flow hiding (packet form)
        upto = [len(pk)] if granularity == "flow" else list(range(1, len(pk) + 1))
        for m in upto:
            agg = _aggregate(pk[:m], tcp=tcp, ports=(sport, dport, proto))
            v = np.full(len(cols), np.nan)
            s = np.full(len(cols), CODE_NS, dtype=np.uint8)
            for fid, val in agg.items():
                if val is None:
                    continue
                js = col_of[fid]
                v[js] = val if isinstance(val, list) else [val]
                s[js] = CODE_OBS
            # hide some packet-level cells (status NOT_SUPPLIED / NOT_OBSERVABLE, value NaN)
            hide = flow_hidden if granularity == "packet" else (rng.random(len(cols)) < hidden_rate)
            for j in packet_cols:
                if hide[j] and s[j] == CODE_OBS:
                    s[j] = CODE_NO if (j % 2) else CODE_NS
                    v[j] = np.nan
            p = pk[m - 1]
            t_ev = float(pk[0].t) if granularity == "flow" else float(p.t)
            meta.append({"event_time": t_ev, "flow": f, "direction": 0 if (granularity == "flow" or p.fwd) else 1,
                         "ip_len": p.ip_len, "entity_0": ents[0], "entity_1": ents[1], "entity_2": ents[2],
                         "reorder_uncertainty_s": float(rng.uniform(0, 0.5)) if rng.random() < 0.3 else float("nan")})
            ev_times.append(t_ev)
            rows_v.append(v)
            rows_s.append(s)
            mal.append(label)
            stage.append(st if label == label else -1)

    # ---- assemble in event-time order
    order = np.argsort(np.array(ev_times, dtype=np.float64), kind="stable")
    frame = pd.DataFrame([meta[k] for k in order])
    n = len(frame)
    frame.insert(0, "seq", np.arange(n, dtype=np.int64))
    frame.insert(1, "record", np.arange(n, dtype=np.int64))
    frame["ingest_time"] = frame["event_time"] + 0.001
    frame["watermark"] = np.nan
    rel_index: dict[tuple[int, int, int], int] = {}
    frame["relation"] = [rel_index.setdefault((int(a), int(b), int(c)), len(rel_index))
                         for a, b, c in frame[["entity_0", "entity_1", "entity_2"]].to_numpy()]
    frame["raw_offset"] = -1
    frame["raw_len"] = -1
    values = np.vstack(rows_v)[order]
    status = np.vstack(rows_s)[order]
    raw = np.frombuffer(b"".join(hashlib.sha256(values[k].tobytes()).digest() for k in range(n)), dtype=np.uint8)
    if granularity == "flow":
        frame = frame.drop(columns=["flow", "ip_len"])
    cu = ColumnarUpdates(
        source_id=f"synthetic-{granularity}-{seed}", adapter="synthetic", adapter_version="0",
        columns=cols, updates=frame, entities=pd.DataFrame({"kind": ent_kind, "key": ent_key}),
        relations=pd.DataFrame(), values=values, status=status, raw_hash=raw.reshape(n, 32).copy(),
    )
    finalise_tables(cu, relation_entities={v: k for k, v in rel_index.items()})
    cu.validate()
    labels = UpdateLabels(np.asarray(mal, np.float32)[order], np.asarray(stage, np.int64)[order],
                          np.full(n, -1, dtype=np.int64), "synthetic-attack" if attack else "benign")
    return cu, labels
