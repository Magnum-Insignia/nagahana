"""Synthetic labelled sources with a planted attack process, for tests and examples of the LR family.

This is a test fixture, not a traffic model: it produces state updates in the data model (through
`datamodel.columnar.to_columnar` with the PCAP adapter's columns), a mapped label table, a stream of
segments and a chronological split manifest, so the whole extraction path of corpus.py (plan_stream,
build_window, survival_targets) runs exactly as on real captures.

Process. Internal hosts 10.0.0.1 ... 10.0.0.n and external hosts exchange benign flows as a Poisson
process (web, DNS, file sharing). Each attack episode, starting at a chosen time, plays three phases:

    reconnaissance       an external scanner probes internal hosts on random ports: bare SYN flows that are
                         never answered (malicious, actor the external initiator: no infiltration)
    command and control  the victim (internal) opens periodic flows to the scanner on port 4444 (malicious,
                         actor the internal initiator: the victim is infiltrated at its first beacon, AS-18)
    lateral movement     the victim connects to other internal hosts on port 445

so the reconnaissance windows precede infiltration: a forecaster that reads the current window can see
an infiltration coming. Benign look-alikes overlap with every phase, as on real networks: unanswered
probes from internet-wide scanners and periodic update checks of every host, all labelled benign. Some fields are left out on purpose (TTL on a share of the flows, flow end on
open flows), so status indicators are exercised.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from nagahana.data.labels import LABEL_COLUMNS
from nagahana.data.sampling import Role, SplitManifest, WindowRecord
from nagahana.data.windows import PreparedSource, SourceData, prepare_source
from nagahana.datamodel.columnar import to_columnar
from nagahana.datamodel.records import EntityRef, FieldValue, OrderingInfo, Provenance, StateUpdate
from nagahana.datamodel.status import ObservationStatus
from nagahana.models.vocab import STAGE_CODE
from nagahana.pipeline.splits import Novelty

OBS = ObservationStatus.OBSERVED
T0 = 1_519_829_100.0          # a multiple of 60 s, so the cadence grid starts on a whole minute


@dataclass(frozen=True)
class _Flow:
    t: float
    src: tuple[str, str]
    dst: tuple[str, str]
    port: int
    proto: int
    flags: int
    pkts_f: int
    pkts_b: int
    bytes_f: int
    bytes_b: int
    duration: float
    unanswered: int
    ttl: float | None
    stage: str
    malicious: float
    actor: int
    family: str


def _update(i: int, f: _Flow) -> StateUpdate:
    fields = {
        "flow.protocol": FieldValue("flow.protocol", f.proto, OBS, "synthetic"),
        "flow.dst_port": FieldValue("flow.dst_port", f.port, OBS, "synthetic"),
        "flow.src_port": FieldValue("flow.src_port", 40000 + (i % 20000), OBS, "synthetic"),
        "flow.packets_fwd": FieldValue("flow.packets_fwd", f.pkts_f, OBS, "synthetic"),
        "flow.packets_bwd": FieldValue("flow.packets_bwd", f.pkts_b, OBS, "synthetic"),
        "flow.bytes_fwd": FieldValue("flow.bytes_fwd", f.bytes_f, OBS, "synthetic"),
        "flow.bytes_bwd": FieldValue("flow.bytes_bwd", f.bytes_b, OBS, "synthetic"),
        "flow.duration": FieldValue("flow.duration", f.duration, OBS, "synthetic"),
        "flow.unanswered": FieldValue("flow.unanswered", f.unanswered, OBS, "synthetic"),
    }
    if f.proto == 6:
        fields["flow.tcp_flags"] = FieldValue("flow.tcp_flags", f.flags, OBS, "synthetic")
        fields["flow.flag_count.syn"] = FieldValue("flow.flag_count.syn", 1 if f.flags & 0x02 else 0, OBS, "synthetic")
    if f.ttl is not None:
        fields["pkt.ttl_mean"] = FieldValue("pkt.ttl_mean", f.ttl, OBS, "synthetic")
    ents = [EntityRef(*f.src), EntityRef(*f.dst)]
    if f.proto in (6, 17):
        ents.append(EntityRef("service", f"{f.dst[1]}:{f.port}/{'tcp' if f.proto == 6 else 'udp'}"))
    return StateUpdate(update_id=f"syn:{i}", ordering=OrderingInfo(event_time=f.t, ingest_time=f.t),
                       entities=tuple(ents), fields=fields, provenance=Provenance("synthetic", "synthetic", "1"))


def synthetic_source(*, seed: int, network: str = "lab", dataset: str = "synthetic", minutes: float = 240.0,
                     benign_per_minute: float = 6.0, scan_noise_per_minute: float = 3.0,
                     episode_starts: tuple[float, ...] | None = None, n_hosts: int = 8, source_id: str | None = None,
                     day: int = 0) -> SourceData:
    """One labelled source (module docstring). Times are T0 + day * 86400 + seconds; episode starts are in minutes."""
    rng = np.random.default_rng(seed)
    hosts = [("host", f"10.0.0.{i + 1}") for i in range(n_hosts)]
    externals = [("external", f"93.184.216.{i + 10}") for i in range(4)]
    scanner = ("external", "13.58.225.34")
    span = minutes * 60.0
    t_base = T0 + 86_400.0 * day
    flows: list[_Flow] = []
    # benign background
    n_benign = rng.poisson(benign_per_minute * minutes)
    for t in np.sort(rng.uniform(0.0, span, size=n_benign)).tolist():
        kind = rng.choice(3, p=[0.7, 0.2, 0.1])
        src = hosts[int(rng.integers(n_hosts))]
        if kind == 0:
            dst, port, proto = externals[int(rng.integers(len(externals)))], int(rng.choice([443, 80])), 6
        elif kind == 1:
            dst, port, proto = externals[0], 53, 17
        else:
            dst, port, proto = hosts[(hosts.index(src) + 1 + int(rng.integers(n_hosts - 1))) % n_hosts], 445, 6
        pf, pb = int(rng.integers(3, 40)), int(rng.integers(2, 60))
        flows.append(_Flow(t_base + t, src, dst, port, proto, 0x1B, pf, pb, pf * int(rng.integers(60, 900)),
                           pb * int(rng.integers(60, 1400)), float(rng.exponential(2.0)), 0,
                           float(rng.choice([64.0, 128.0])) if rng.random() < 0.8 else None, "none", 0.0, -1, "benign"))
    # benign look-alikes, as on any network attached to the internet: unanswered probes from internet-wide
    # scanners (labelled benign background, as the datasets label them) and periodic update checks of hosts
    for t in np.sort(rng.uniform(0.0, span, size=rng.poisson(scan_noise_per_minute * minutes))).tolist():
        dst = hosts[int(rng.integers(n_hosts))]
        noise_src = ("external", f"198.51.100.{int(rng.integers(1, 250))}")
        flows.append(_Flow(t_base + t, noise_src, dst, int(rng.integers(1, 1024)), 6, 0x02, 1, 0, 60, 0, 0.0, 1,
                           float(rng.choice([48.0, 52.0, 112.0])), "none", 0.0, -1, "benign"))
    for h in range(n_hosts):
        period = float(rng.uniform(15.0, 40.0))
        for t in np.arange(float(rng.uniform(0.0, period)), span, period).tolist():
            flows.append(_Flow(t_base + t + float(rng.uniform(-1.0, 1.0)), hosts[h], externals[1], 443, 6, 0x1B,
                               int(rng.integers(4, 9)), int(rng.integers(4, 9)), int(rng.integers(600, 1200)),
                               int(rng.integers(500, 1000)), 1.5, 0, 64.0, "none", 0.0, -1, "benign"))
    # attack episodes
    starts = episode_starts if episode_starts is not None else tuple(np.arange(10.0, minutes - 15.0, 23.0).tolist())
    for e, s0 in enumerate(starts):
        base = t_base + s0 * 60.0
        victim = hosts[e % n_hosts]
        t = base
        while t < base + 180.0:                                               # reconnaissance, 3 min
            dst = hosts[int(rng.integers(n_hosts))]
            flows.append(_Flow(t, scanner, dst, int(rng.integers(1, 1024)), 6, 0x02, 1, 0, 60, 0, 0.0, 1, 52.0,
                               "reconnaissance", 1.0, 0, "recon-c2"))
            t += float(rng.uniform(2.0, 8.0))
        for j in range(18):                                                   # command and control, 6 min
            flows.append(_Flow(base + 240.0 + 20.0 * j + float(rng.uniform(-1.0, 1.0)), victim, scanner, 4444, 6, 0x1B,
                               6, 6, 900, 700, 1.5, 0, 64.0, "command_and_control", 1.0, 0, "recon-c2"))
        for j in range(4):                                                    # lateral movement
            tgt = hosts[(hosts.index(victim) + 1 + j) % n_hosts]
            flows.append(_Flow(base + 420.0 + 15.0 * j, victim, tgt, 445, 6, 0x1B, 12, 10, 4000, 3000, 3.0, 0, 128.0,
                               "lateral_movement", 1.0, 0, "recon-c2"))
    flows.sort(key=lambda f: f.t)
    ups = [_update(i, f) for i, f in enumerate(flows)]
    from nagahana.ingest.pcap import COLUMNS as PCAP_COLUMNS

    cu = to_columnar(ups, PCAP_COLUMNS)
    if source_id is not None:
        cu.source_id = source_id
    n = len(flows)
    labels = pd.DataFrame({
        "seq": np.arange(n), "record": np.arange(n), "label_raw": [f.stage for f in flows],
        "malicious": np.asarray([f.malicious for f in flows], dtype=np.float32),
        "stage": np.asarray([STAGE_CODE[f.stage] for f in flows], dtype=np.int64),
        "technique": ["T1046" if f.stage == "reconnaissance" else ("T1571" if f.stage == "command_and_control" else "")
                      for f in flows],
        "family": [f.family for f in flows], "subfamily": [f.family for f in flows],
        "actor_role": np.asarray([f.actor for f in flows], dtype=np.int8), "mapped": True,
    })[list(LABEL_COLUMNS)]
    return SourceData(cu, labels, network=network, dataset=dataset)


def chronological_manifest(sources: list[PreparedSource], records: list[WindowRecord], *,
                           shares: tuple[float, float, float, float] = (0.5, 0.15, 0.15, 0.2), purge: int = 1,
                           novelty: str = "known") -> SplitManifest:
    """Roles by time inside each source: train, val, test, zero_shot shares, `purge` records excluded at each cut."""
    role: dict[str, Role] = {}
    nov: dict[str, Novelty] = {}
    names = (Role.TRAIN, Role.VAL, Role.TEST, Role.ZERO_SHOT)
    by_source: dict[int, list[WindowRecord]] = {}
    for r in records:
        by_source.setdefault(r.source, []).append(r)
    for recs in by_source.values():
        recs.sort(key=lambda r: (r.t_start, r.id))
        cuts = np.cumsum(np.asarray(shares) / np.sum(shares)) * len(recs)
        ks = [min(int(np.searchsorted(cuts, i, side="right")), 3) for i in range(len(recs))]
        for i, r in enumerate(recs):
            k = ks[i]
            after_cut = any(ks[i - j] != k for j in range(1, purge + 1) if i - j >= 0)
            role[r.id] = Role.EXCLUDED if after_cut else names[k]
            if role[r.id] is Role.ZERO_SHOT:
                nov[r.id] = Novelty(novelty)
    return SplitManifest(records={r.id: r for r in records}, role=role, novelty=nov, mode="full")


def manifest_by_source(records: list[WindowRecord], roles: dict[int, str], *,
                       novelty: dict[int, str] | None = None) -> SplitManifest:
    """Every record of source i gets roles[i] (train, val, test, zero_shot or excluded); zero-shot records
    take novelty[i] ("known" or "novel")."""
    role: dict[str, Role] = {}
    nov: dict[str, Novelty] = {}
    for r in records:
        role[r.id] = Role(roles[r.source])
        if role[r.id] is Role.ZERO_SHOT:
            nov[r.id] = Novelty((novelty or {}).get(r.source, "known"))
    return SplitManifest(records={r.id: r for r in records}, role=role, novelty=nov, mode="full")


def synthetic_corpus_inputs(cfg: object, plan: list[tuple[str, int, str]], *, minutes: float = 60.0,
                            episodes_per_source: int = 1, novelty: str = "known",
                            seed: int = 0) -> tuple[list[PreparedSource], SplitManifest]:
    """Prepared sources and a manifest over their stream segments (AS-333).

    plan: one (network, day, role) per source; a network's sources are its days, so its time line runs
    across them as a multi-day capture does. Each source plays `episodes_per_source` attack episodes at
    seeded times in the middle of its span; zero-shot sources take the given novelty mark.
    """
    from nagahana.data.stream import default_segment_seconds, plan_stream, segment_records

    sources: list[PreparedSource] = []
    records: list[WindowRecord] = []
    roles: dict[int, str] = {}
    for i, (net, day, role) in enumerate(plan):
        rng = np.random.default_rng([seed, i])
        starts = tuple(sorted(rng.uniform(0.3 * minutes, 0.6 * minutes, size=episodes_per_source).tolist()))
        src = prepare_source(synthetic_source(seed=seed * 1000 + i, network=net, minutes=minutes,
                                              source_id=f"synthetic-{net}-day{day}", episode_starts=starts, day=day))
        sources.append(src)
        roles[i] = role
        records += segment_records(i, src, plan_stream(src, cfg), segment_seconds=default_segment_seconds(cfg))  # type: ignore[arg-type]
    return sources, manifest_by_source(records, roles, novelty={i: novelty for i in roles})


__all__ = ["T0", "chronological_manifest", "manifest_by_source", "synthetic_corpus_inputs", "synthetic_source"]
