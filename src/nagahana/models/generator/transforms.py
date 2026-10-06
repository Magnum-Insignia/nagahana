"""Deterministic, label-preserving variants of a real window (Generator families 1, 2 and topology).

Purpose
-------
The owner's Generator makes "variants of the same event with varying signatures, traces, impacts as
observability sliding" so the model gets "clarity of the partial observable concept" [Q-37], [A-17].
This module holds the variants that are **label-preserving by construction**: each one changes how
an event is *observed* or *rendered*, never *what happened*, so every surviving update keeps exactly
its real label (`result.labels == labels.take(result.source_rows)`). Each is deterministic given the
seed of the `numpy.random.Generator` passed in.

Families (build-spec §2.11 items 1–2, §4b.5; assumption AS-27 standing for held D-14)
--------------------------------------------------------------------------------------
Observability sliding — the same traffic seen by a weaker sensor:

- `DropFields` (AS-351): a sensor that cannot see packets. Contributing cells of `Level.PACKET` fields
  become NOT_OBSERVABLE (value NaN). *Why label-preserving:* the traffic is unchanged; only the evidence
  shrinks. The model must learn that absence of packet evidence is a property of the sensor (D-41).
- `FlowOnlyExport` (AS-350): an IPFIX/NetFlow exporter. Only the fields of the IPFIX biflow basic
  profile stay (ports, protocol, TCP flags, bytes and packets per direction, duration: RFC 7012 IEs
  octetDeltaCount, packetDeltaCount, protocolIdentifier, tcpControlBits, source/destinationTransportPort,
  flowStart/EndMilliseconds; reverse direction per RFC 5103); every other field becomes NOT_SUPPLIED.
  At packet granularity an exporter emits one record per flow, so only the last row of each flow (its
  complete running state) is kept. *Why label-preserving:* same flows, fewer fields.
- `PacketSampling` (AS-352): 1-in-n packet sampling (sFlow, RFC 3176; sampled NetFlow). Each packet is
  kept independently with probability 1/n, and counts are rescaled by n, the inverse-probability
  estimator used for sampled flow statistics (Duffield, Lund & Thorup, ACM SIGCOMM 2003):
      flow rows:   k_d ~ Binomial(N_d, 1/n);  tally_f ~ Hypergeometric(N, tally_f, k);
                   bins ~ MultivariateHypergeometric(bins ∪ {rest}, k);   reported = n · sampled
                   bytes_d = n · round(B_d · k_d / N_d)  (mean packet size kept: an approximation)
      packet rows: row survives with probability 1/n; running tallies are rebuilt exactly from the
                   per-packet increments of the surviving rows: c_i = n · Σ_{j ≤ i, j kept} Δc_j;
                   duration_i = t_i − t_first kept (the catalogue definition).
  A flow with no sampled packet disappears. Fields that depend on the full packet sequence and whose
  definition is adapter-specific (IAT statistics, TTL, initial windows, bidirectional ratio, derived
  scan evidence, application fields) become NOT_SUPPLIED rather than be invented. *Why
  label-preserving:* sampling is a property of the sensor; the surviving updates are the same acts.
  Physics: sampled tallies are drawn from the sampled packets, so flag ≤ packets and
  20·packets ≤ bytes ≤ MTU·packets hold before and after the ×n rescaling.
- `SensorHiding` (AS-353): an untapped segment. A random set of internal hosts forms the hidden
  segment; updates between two of its members are not seen at all. *Why label-preserving:* surviving
  updates are unchanged.

Signature variation — the same act with different surface features:

- `PortRemap` (AS-354): a destination port moves inside its service-alias class (one service on its
  alternative registered ports: http 80/8080/8008; SMB 445/139), by a bijection so distinct services
  never merge; ephemeral source ports (RFC 6335 §6: 49152–65535) are re-drawn inside that range. Service
  entity keys ("addr:port/proto") follow the remap. *Why label-preserving:* the protocol and the act are
  the same; only port numbers an attacker or admin can choose change. Plane formation by port (AS-01)
  will see some of these as off-port services, which is the robustness this variant trains.
- `TimingJitter` (AS-355): one scale s per flow, |log s| ≤ log(1+ε): duration, iat_mean, iat_max × s and
  iat_var × s² (variance scales with the square). Every ratio and order between the timing fields is
  kept exactly (iat_mean ≤ iat_max ≤ duration still hold). At packet granularity the packets' event
  times inside the flow are scaled about the flow's first packet so running state and times agree.
  Physics: a shorter duration can break the link-rate bound; the acceptance gate rejects such variants.
- `RateScaling` (AS-356): the attack's tempo changes. Event times of malicious flows are dilated about
  the first malicious time, t' = t_a + ρ (t − t_a), with log ρ uniform in [log ρ_min, log ρ_max]; each
  flow's internal timing is kept. A slow scan stays a scan (the problem statement's "slow
  reconnaissance scan designed to evade flow-based thresholds"). Benign windows dilate all flows.
- `ReorderWithinUncertainty` (AS-357, build-spec §4b.5): each update's true time is taken uniform in
  [t − r, t + r] with r = `reorder_uncertainty_s` (NaN: ordering certain, r = 0); rows are re-sorted.
  At packet granularity the order *inside* a flow is kept (a running state cannot go backwards).

Topology variation (build-spec §4b.5, AS-358) — only on relations whose every update is known benign:

- `HyperedgeDropout`: each eligible relation (hyperedge) is dropped with probability p (all its updates).
- `RewireBenign`: each eligible relation, whose endpoints never take part in a malicious update, gets
  a new initiator of the same entity kind (also never malicious) among the entities of the window's own
  rows (AS-584: an entity table may list entities of other splits; a variant never introduces one). The
  service entity stays with the responder. Why label-preserving: attack updates and the attack's
  entities are untouched; benign background traffic is benign whoever initiated it.

Tool fingerprints (build-spec section 2.11 item 2; AS-585):

- `ToolFingerprintSwap`: a client tool fingerprint (`proto.tls.ja4` by default) is replaced by another
  fingerprint of the same tool class. The tool class of a row is (label class, service, transport): the
  attack family for a malicious row or "benign" for a benign row, the named service class of the
  destination port (the Decoder's IANA table, AS-106) and the IP protocol. The table of classes
  (`FingerprintClasses`) is built by the data preparation from training segments only (AS-367) with
  the empirical frequency of each fingerprint; a row draws a different fingerprint of its class with
  those frequencies. Rows of unknown label keep their fingerprint. Why label-preserving: within one
  class the act (who attacks which service over which transport, or benign use of it) is the same;
  only the client implementation that carried it differs, which an attacker or a user can change at will.

Invariants (tested in tests/test_generator_transforms.py)
- labels of surviving rows are the real labels (or, for FlowOnlyExport at packet granularity, the
  flow's aggregate, which equals the last row's label when labels are constant within a flow);
- no transform turns an excluded cell into a contributing one (statuses only move towards exclusion);
- `ColumnarUpdates.validate()` holds (NaN ⇔ excluded);
- hard limits hold whenever they held on the source (acceptance re-checks every variant).

Decisions: D-40, D-41, D-18. Assumptions: AS-27, AS-350 ... AS-358, AS-584, AS-585.
Extension points: register a new transform in `TRANSFORMS`; extend `SERVICE_ALIAS_CLASSES` or
`IPFIX_BIFLOW_FIELDS` after stage-1 analysis.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Sequence
from typing import Any, Protocol

import numpy as np

from nagahana.core.registry import Registry
from nagahana.datamodel.columnar import STATUS_CODE, ColumnarUpdates
from nagahana.datamodel.fields import CATALOGUE, Level
from nagahana.datamodel.status import CONTRIBUTING, ObservationStatus
from nagahana.models.config.components import GeneratorConfig
from nagahana.models.generator.assumptions import use
from nagahana.models.generator.config import GeneratorPolicy
from nagahana.models.generator.limits import BYTES, FLAGS, HISTOGRAM, PACKETS, TALLIES, ColumnIndex
from nagahana.models.generator.variants import (
    TransformResult,
    UpdateLabels,
    changed_cells,
    derive,
    entity_columns,
    is_packet_granularity,
    sort_by_time,
)

_CONTRIB_CODES = np.array([STATUS_CODE[s] for s in CONTRIBUTING], dtype=np.uint8)
CODE_NOT_SUPPLIED = STATUS_CODE[ObservationStatus.NOT_SUPPLIED]
CODE_NOT_OBSERVABLE = STATUS_CODE[ObservationStatus.NOT_OBSERVABLE]

#: IPFIX biflow basic profile (AS-350): fields an IPFIX/NetFlow v9 exporter with RFC 5103 biflows supplies.
IPFIX_BIFLOW_FIELDS: frozenset[str] = frozenset({
    "flow.src_ip", "flow.dst_ip", "flow.src_port", "flow.dst_port", "flow.protocol", "flow.tcp_flags",
    *BYTES, *PACKETS, "flow.duration",
})
#: Flow-key fields: known from any packet of the flow, so they survive packet sampling.
FLOW_KEY_FIELDS: frozenset[str] = frozenset({"flow.src_ip", "flow.dst_ip", "flow.src_port", "flow.dst_port", "flow.protocol"})
#: TCP control bits in the flags bitmask (RFC 9293 header order): FIN 0x01 … URG 0x20.
FLAG_BITS: dict[str, int] = {"flow.flag_count.fin": 0x01, "flow.flag_count.syn": 0x02, "flow.flag_count.rst": 0x04,
                             "flow.flag_count.psh": 0x08, "flow.flag_count.ack": 0x10, "flow.flag_count.urg": 0x20}
#: Service-alias classes (AS-354): one service on its registered alternative ports.
SERVICE_ALIAS_CLASSES: dict[str, tuple[int, ...]] = {
    "http": (80, 8080, 8008),     # IANA: http (80), http-alt (8080, 8008)
    "smb": (445, 139),            # SMB direct over TCP (445) and over NetBIOS session service (139)
}
EPHEMERAL_PORTS = (49152, 65535)  # RFC 6335 §6, dynamic / private ports


class NotApplicable(Exception):  # noqa: N818 (a reason, not an error of the code)
    """The transform cannot be applied to this window (e.g. no packet counts to sample)."""


class Transform(Protocol):
    """A deterministic (given the RNG) variant producer."""

    name: str

    def apply(self, cu: ColumnarUpdates, labels: UpdateLabels, rng: np.random.Generator) -> TransformResult: ...


TRANSFORMS: Registry[Callable[..., Any]] = Registry("generator transform")


# ============================================================================== helpers
def _contributing(status: np.ndarray) -> np.ndarray:
    return np.isin(status, _CONTRIB_CODES)


def _exclude(values: np.ndarray, status: np.ndarray, mask: np.ndarray, code: int) -> None:
    """In place: contributing cells under `mask` become excluded with `code` (NaN value).

    Cells already excluded keep their own status (we never relabel *why* something was absent).
    """
    m = mask & _contributing(status)
    status[m] = code
    values[m] = np.nan


def _result(src: ColumnarUpdates, labels: UpdateLabels, rows: np.ndarray, name: str, params: dict[str, Any], *,
            values: np.ndarray | None = None, status: np.ndarray | None = None,
            frame_updates: dict[str, np.ndarray] | None = None, new_labels: UpdateLabels | None = None) -> TransformResult:
    # Build the variant table for `rows` and record which cells changed versus the source.
    rows = np.asarray(rows, dtype=np.int64)
    out = derive(src, rows, values=values, status=status, frame_updates=frame_updates)
    # Structural change: rows dropped or re-ordered, or update-table columns (times, entities) edited.
    structural = len(rows) != len(src) or bool((rows != np.arange(len(rows))).any())
    for col, arr in (frame_updates or {}).items():
        before = src.updates[col].to_numpy()[rows] if col in src.updates else None
        structural |= before is None or not np.array_equal(before, np.asarray(arr), equal_nan=before.dtype.kind == "f")
    return TransformResult(
        updates=out, labels=new_labels if new_labels is not None else labels.take(rows), source_rows=rows,
        changed=changed_cells(src, rows, out.values, out.status), producer=name, label_mode="by-construction",
        params=params, structural=structural,
    )


def _exclude_cells(values: np.ndarray, status: np.ndarray, i: int, cols: Sequence[int]) -> None:
    """In place: the contributing cells of row i in `cols` become NOT_SUPPLIED (NaN)."""
    for j in cols:
        if status[i, j] in _CONTRIB_CODES:
            values[i, j], status[i, j] = np.nan, CODE_NOT_SUPPLIED


def _side_status_drop(cu: ColumnarUpdates, keep: frozenset[str], code: int) -> dict[str, np.ndarray]:
    """Status columns of side fields (non-matrix fields, e.g. TLS SNI) not in `keep` → `code` where contributing."""
    out: dict[str, np.ndarray] = {}
    for fid, (_value_col, status_col) in cu.side_fields.items():
        if fid in keep or status_col not in cu.updates:
            continue
        s = cu.updates[status_col].to_numpy().astype(np.uint8).copy()
        s[np.isin(s, _CONTRIB_CODES)] = code
        out[status_col] = s
    return out


def _flow_ids(cu: ColumnarUpdates) -> np.ndarray:
    """Flow id per row: the `flow` column at packet granularity, else one flow per row."""
    if is_packet_granularity(cu):
        return cu.updates["flow"].to_numpy().astype(np.int64)
    return np.arange(len(cu), dtype=np.int64)


def _groups(ids: np.ndarray) -> list[np.ndarray]:
    """Row indices of each distinct id, each in row order."""
    order = np.argsort(ids, kind="stable")
    splits = np.flatnonzero(np.diff(ids[order])) + 1
    return [g for g in np.split(order, splits) if len(g)]


# ============================================================================== observability sliding
@TRANSFORMS.register("drop-packet-level", summary="packet-level fields → NOT_OBSERVABLE (AS-351)")
class DropFields:
    """Contributing cells of fields at `levels` become `status` on a fraction of rows (AS-351)."""

    def __init__(self, *, levels: Sequence[Level], status: ObservationStatus, row_fraction: float) -> None:
        if status in CONTRIBUTING:
            raise ValueError("DropFields must drop to an excluded status")
        if not 0.0 < row_fraction <= 1.0:
            raise ValueError("row_fraction must be in (0, 1]")
        self.levels, self.status, self.row_fraction = tuple(levels), status, row_fraction
        self.name = "drop-packet-level"

    def apply(self, cu: ColumnarUpdates, labels: UpdateLabels, rng: np.random.Generator) -> TransformResult:
        use("AS-351", by=__name__)
        cols = np.array([CATALOGUE[c.field_id].level in self.levels for c in cu.columns], dtype=bool)   # [C]
        rows = rng.random(len(cu)) < self.row_fraction if self.row_fraction < 1.0 else np.ones(len(cu), bool)
        values, status = cu.values.copy(), cu.status.copy()
        _exclude(values, status, rows[:, None] & cols[None, :], STATUS_CODE[self.status])
        return _result(cu, labels, np.arange(len(cu)), self.name,
                       {"levels": [lv.value for lv in self.levels], "row_fraction": self.row_fraction},
                       values=values, status=status)


@TRANSFORMS.register("flow-only-export", summary="IPFIX biflow basic profile; other fields NOT_SUPPLIED (AS-350)")
class FlowOnlyExport:
    """Keep only `keep` fields (default profile `IPFIX_BIFLOW_FIELDS`); one record per flow (AS-350)."""

    def __init__(self, *, keep: frozenset[str]) -> None:
        self.keep = keep
        self.name = "flow-only-export"

    def apply(self, cu: ColumnarUpdates, labels: UpdateLabels, rng: np.random.Generator) -> TransformResult:
        use("AS-350", by=__name__)
        packet = is_packet_granularity(cu)
        new_labels: UpdateLabels | None = None
        if packet:
            # One exported record per flow: the flow's last row carries its complete running state.
            groups = _groups(_flow_ids(cu))
            rows = np.sort(np.array([g[-1] for g in groups], dtype=np.int64))
            by_last = {int(g[-1]): g for g in groups}
            mal, stage, tech = [], [], []
            for r in rows.tolist():
                g = by_last[r]
                m = labels.malicious_rows()[g]
                if m.any():                                   # any malicious packet → malicious flow
                    last_m = g[m][-1]
                    mal.append(1.0)
                    stage.append(int(labels.stage[last_m]))
                    tech.append(int(labels.technique[last_m]))
                elif labels.benign_rows()[g].all():           # all known benign → benign flow
                    mal.append(0.0)
                    stage.append(int(labels.stage[r]))
                    tech.append(int(labels.technique[r]))
                else:                                         # some unknown, none malicious → unknown
                    mal.append(float("nan"))
                    stage.append(-1)
                    tech.append(-1)
            new_labels = UpdateLabels(np.asarray(mal, np.float32), np.asarray(stage, np.int64),
                                      np.asarray(tech, np.int64), labels.family)
        else:
            rows = np.arange(len(cu), dtype=np.int64)
        values, status = cu.values[rows].copy(), cu.status[rows].copy()
        drop = np.array([c.field_id not in self.keep for c in cu.columns], dtype=bool)
        _exclude(values, status, np.broadcast_to(drop, values.shape), CODE_NOT_SUPPLIED)
        side = {k: v[rows] for k, v in _side_status_drop(cu, self.keep, CODE_NOT_SUPPLIED).items()}
        return _result(cu, labels, rows, self.name, {"packet_granularity": packet, "kept": sorted(self.keep)},
                       values=values, status=status, frame_updates=side, new_labels=new_labels)


@TRANSFORMS.register("packet-sampling", summary="1-in-n packet sampling with ×n rescaling (AS-352)")
class PacketSampling:
    """1-in-n packet sampling; n drawn from `rates` per variant (AS-352). See the module docstring."""

    def __init__(self, *, rates: Sequence[int]) -> None:
        if not rates or any(int(n) < 2 for n in rates):
            raise ValueError("sampling rates must be integers ≥ 2")
        self.rates = tuple(int(n) for n in rates)
        self.name = "packet-sampling"

    # Field classes under sampling: tallies are thinned and rescaled; flow keys survive; the rest is dropped.
    @staticmethod
    def _tally_fields() -> tuple[str, ...]:
        return (*PACKETS, *BYTES, *FLAGS, *TALLIES)

    def apply(self, cu: ColumnarUpdates, labels: UpdateLabels, rng: np.random.Generator) -> TransformResult:
        use("AS-352", by=__name__)
        idx = ColumnIndex(cu.columns)
        if not idx.has(*PACKETS):
            raise NotApplicable("packet sampling needs packet counts per direction")
        n = int(rng.choice(self.rates))
        values, status = cu.values.copy(), cu.status.copy()
        if is_packet_granularity(cu):
            keep = self._packet_rows(cu, values, status, idx, n, rng)
        else:
            keep = self._flow_rows(cu, values, status, idx, n, rng)
        # Everything that is neither a rescaled tally, the flags bitmask, a histogram bin nor a flow key
        # cannot be vouched for under sampling → NOT_SUPPLIED (never invented).
        handled = set(self._tally_fields()) | {HISTOGRAM, "flow.tcp_flags"} | FLOW_KEY_FIELDS
        if is_packet_granularity(cu):
            handled.add("flow.duration")
        other = np.array([c.field_id not in handled for c in cu.columns], dtype=bool)
        _exclude(values, status, np.broadcast_to(other, values.shape).copy(), CODE_NOT_SUPPLIED)
        rows = np.flatnonzero(keep)
        side = {k: v[rows] for k, v in _side_status_drop(cu, FLOW_KEY_FIELDS, CODE_NOT_SUPPLIED).items()}
        return _result(cu, labels, rows, self.name, {"n": n, "packet_granularity": is_packet_granularity(cu)},
                       values=values[rows], status=status[rows], frame_updates=side)

    # ---------------------------------------------------------------- flow records (one row per flow)
    def _flow_rows(self, cu: ColumnarUpdates, values: np.ndarray, status: np.ndarray, idx: ColumnIndex,
                   n: int, rng: np.random.Generator) -> np.ndarray:
        contrib = _contributing(status)
        jpf, jpb = idx.one(PACKETS[0]), idx.one(PACKETS[1])
        assert jpf is not None and jpb is not None
        keep = np.ones(len(cu), dtype=bool)
        p = 1.0 / n
        for i in range(len(cu)):
            if not (contrib[i, jpf] and contrib[i, jpb]):
                # Packet counts unknown: sampling cannot be simulated → every non-key numeric field excluded.
                _exclude_cells(values, status, i, [j for j, c in enumerate(cu.columns) if c.field_id not in FLOW_KEY_FIELDS])
                continue
            n_dir = [int(values[i, jpf]), int(values[i, jpb])]
            k_dir = [int(rng.binomial(nd, p)) if nd > 0 else 0 for nd in n_dir]
            total, k = sum(n_dir), sum(k_dir)
            if k == 0:
                keep[i] = False                      # no sampled packet: the flow is never exported
                continue
            # packets and bytes per direction (bytes keep the mean packet size of the direction)
            for d, (fb, fp) in enumerate(zip(BYTES, PACKETS, strict=True)):
                jp, jb = idx.one(fp), idx.one(fb)
                assert jp is not None
                values[i, jp] = n * k_dir[d]
                if jb is not None and contrib[i, jb]:
                    obs = round(values[i, jb] * k_dir[d] / n_dir[d]) if n_dir[d] > 0 else 0
                    values[i, jb] = n * obs
            # per-packet tallies: hypergeometric draw of the tallied packets among the k sampled
            sampled_flags: dict[str, int] = {}
            for f in (*FLAGS, *TALLIES):
                jf = idx.one(f)
                if jf is None or not contrib[i, jf]:
                    continue
                good = int(values[i, jf])
                if not 0 <= good <= total:
                    _exclude_cells(values, status, i, [jf])      # cannot thin an inconsistent tally
                    continue
                obs = int(rng.hypergeometric(good, total - good, k)) if good > 0 else 0
                values[i, jf] = n * obs
                sampled_flags[f] = obs
            bins = idx.all(HISTOGRAM)
            if bins:
                if contrib[i, bins].all() and values[i, bins].sum() <= total:
                    colours = np.append(values[i, bins].astype(np.int64), total - int(values[i, bins].sum()))
                    draw = rng.multivariate_hypergeometric(colours, k)
                    values[i, bins] = n * draw[:-1]
                else:
                    _exclude_cells(values, status, i, bins)
            self._flags_bitmask(values, status, contrib, idx, i, {f: sampled_flags.get(f) for f in FLAGS})
        return keep

    # ---------------------------------------------------------------- packet records (running state)
    def _packet_rows(self, cu: ColumnarUpdates, values: np.ndarray, status: np.ndarray, idx: ColumnIndex,
                     n: int, rng: np.random.Generator) -> np.ndarray:
        contrib = _contributing(status)
        survive = rng.random(len(cu)) < 1.0 / n
        times = cu.updates["event_time"].to_numpy().astype(np.float64)
        tally_cols = [j for f in self._tally_fields() for j in ([idx.one(f)] if idx.one(f) is not None else [])]
        tally_cols += idx.all(HISTOGRAM)
        jflags = {f: idx.one(f) for f in FLAGS}
        jt, jd = idx.one("flow.tcp_flags"), idx.one("flow.duration")
        for g in _groups(_flow_ids(cu)):
            kept = g[survive[g]]
            if len(kept) == 0:
                continue
            per_packet_bits = np.zeros(len(g), dtype=np.int64)
            bits_known = True
            for j in tally_cols:
                if not contrib[g, j].all():
                    # a gap in the running tally: the increments are unknown → the field is not supplied
                    kc = kept[contrib[kept, j]]
                    values[kc, j], status[kc, j] = np.nan, CODE_NOT_SUPPLIED
                    if j in jflags.values():
                        bits_known = False
                    continue
                delta = np.diff(values[g, j], prepend=0.0)          # per-packet increment of the running tally
                if (delta < 0).any():                                # not a running tally → cannot rebuild
                    values[kept, j], status[kept, j] = np.nan, CODE_NOT_SUPPLIED
                    if j in jflags.values():
                        bits_known = False
                    continue
                cum = n * np.cumsum(delta * survive[g])              # c_i = n · Σ_{j ≤ i kept} Δc_j
                values[kept, j] = cum[survive[g]]
                for f, jf in jflags.items():
                    if jf == j:
                        per_packet_bits |= np.where(delta > 0, FLAG_BITS[f], 0)
            if jd is not None:
                ok = contrib[kept, jd]
                t0 = times[kept[0]]
                values[kept[ok], jd] = times[kept[ok]] - t0          # duration = t_i − t_first kept
            if jt is not None:
                ok = contrib[kept, jt]
                seen = values[g, jt][contrib[g, jt]].astype(np.int64)
                extra = bool((seen & ~0x3F).any()) if seen.size else False
                if bits_known and all(j is not None for j in jflags.values()) and not extra:
                    running_or = np.bitwise_or.accumulate(np.where(survive[g], per_packet_bits, 0))
                    values[kept[ok], jt] = running_or[survive[g]][ok]
                else:
                    values[kept[ok], jt], status[kept[ok], jt] = np.nan, CODE_NOT_SUPPLIED
        return survive

    @staticmethod
    def _flags_bitmask(values: np.ndarray, status: np.ndarray, contrib: np.ndarray, idx: ColumnIndex, i: int,
                       sampled: dict[str, int | None]) -> None:
        # OR of the TCP flags over the sampled packets: known only if every flag count was sampled and the
        # bitmask has no bits beyond the six counted flags (ECE/CWR would be unknowable).
        jt = idx.one("flow.tcp_flags")
        if jt is None or not contrib[i, jt]:
            return
        if any(v is None for v in sampled.values()) or (int(values[i, jt]) & ~0x3F):
            values[i, jt], status[i, jt] = np.nan, CODE_NOT_SUPPLIED
            return
        values[i, jt] = float(sum(FLAG_BITS[f] for f, v in sampled.items() if v))


@TRANSFORMS.register("sensor-hiding", summary="untapped segment: intra-segment updates vanish (AS-353)")
class SensorHiding:
    """A random `fraction` of internal hosts forms a hidden segment (AS-353)."""

    def __init__(self, *, fraction: float) -> None:
        if not 0.0 < fraction <= 1.0:
            raise ValueError("fraction must be in (0, 1]")
        self.fraction = fraction
        self.name = "sensor-hiding"

    def apply(self, cu: ColumnarUpdates, labels: UpdateLabels, rng: np.random.Generator) -> TransformResult:
        use("AS-353", by=__name__)
        hosts = np.flatnonzero(cu.entities["kind"].to_numpy() == "host")
        if len(hosts) < 2:
            raise NotApplicable("a hidden segment needs at least two internal hosts")
        k = min(len(hosts), max(2, round(self.fraction * len(hosts))))
        hidden = rng.choice(hosts, size=k, replace=False)
        e0, e1 = cu.updates["entity_0"].to_numpy(), cu.updates["entity_1"].to_numpy()
        seen = ~(np.isin(e0, hidden) & np.isin(e1, hidden))      # intra-segment traffic is not seen
        return _result(cu, labels, np.flatnonzero(seen), self.name, {"hidden_entities": sorted(int(h) for h in hidden)})


# ============================================================================== signature variation
_SERVICE_KEY = re.compile(r"^(?P<addr>.*):(?P<port>\d+)/(?P<proto>[^/]+)$")


@TRANSFORMS.register("port-remap", summary="ports inside service-alias classes; ephemeral source ports (AS-354)")
class PortRemap:
    """Bijective remap of destination ports inside each alias class + ephemeral source ports (AS-354)."""

    def __init__(self, *, classes: dict[str, tuple[int, ...]], ephemeral: tuple[int, int]) -> None:
        self.classes, self.ephemeral = classes, ephemeral
        self.name = "port-remap"

    def apply(self, cu: ColumnarUpdates, labels: UpdateLabels, rng: np.random.Generator) -> TransformResult:
        use("AS-354", by=__name__)
        idx = ColumnIndex(cu.columns)
        values, status = cu.values.copy(), cu.status.copy()
        contrib = _contributing(status)
        dst_map: dict[int, int] = {}
        for ports in self.classes.values():
            perm = rng.permutation(len(ports))                   # a bijection: distinct services never merge
            dst_map.update({ports[i]: ports[int(perm[i])] for i in range(len(ports))})
        jd = idx.one("flow.dst_port")
        if jd is not None:
            m = contrib[:, jd]
            values[m, jd] = [float(dst_map.get(int(v), int(v))) for v in values[m, jd]]
        src_map: dict[int, int] = {}
        js = idx.one("flow.src_port")
        if js is not None:
            m = contrib[:, js]
            lo, hi = self.ephemeral
            eph = sorted({int(v) for v in values[m, js] if lo <= int(v) <= hi})
            if eph:
                new = rng.choice(np.arange(lo, hi + 1), size=len(eph), replace=False)
                src_map = {p: int(q) for p, q in zip(eph, new, strict=True)}
                values[m, js] = [float(src_map.get(int(v), int(v))) for v in values[m, js]]
        res = _result(cu, labels, np.arange(len(cu)), self.name,
                      {"dst_map": {k: v for k, v in dst_map.items() if k != v}, "src_remapped": len(src_map)},
                      values=values, status=status)
        # Service entity keys follow the destination-port remap (identifiers, for audit consistency).
        if dst_map:
            keys = res.updates.entities["key"].astype(str).to_numpy().copy()
            kinds = res.updates.entities["kind"].to_numpy()
            for i in np.flatnonzero(kinds == "service"):
                mt = _SERVICE_KEY.match(keys[i])
                if mt and int(mt["port"]) in dst_map:
                    keys[i] = f"{mt['addr']}:{dst_map[int(mt['port'])]}/{mt['proto']}"
            res.updates.entities["key"] = keys
        return res


@TRANSFORMS.register("timing-jitter", summary="one timing scale per flow on duration/IAT (AS-355)")
class TimingJitter:
    """Scale each flow's timing fields by s, |log s| ≤ log(1 + eps) (AS-355)."""

    def __init__(self, *, eps: float) -> None:
        if not eps > 0:
            raise ValueError("eps must be positive")
        self.eps = eps
        self.name = "timing-jitter"

    def apply(self, cu: ColumnarUpdates, labels: UpdateLabels, rng: np.random.Generator) -> TransformResult:
        use("AS-355", by=__name__)
        idx = ColumnIndex(cu.columns)
        values, status = cu.values.copy(), cu.status.copy()
        contrib = _contributing(status)
        flows = _flow_ids(cu)
        uniq, inv = np.unique(flows, return_inverse=True)
        s_flow = np.exp(rng.uniform(-math.log1p(self.eps), math.log1p(self.eps), size=len(uniq)))
        s = s_flow[inv]                                                       # [U] one scale per flow
        for f, power in (("flow.duration", 1), ("flow.iat_mean", 1), ("flow.iat_max", 1), ("flow.iat_var", 2)):
            j = idx.one(f)
            if j is not None:
                m = contrib[:, j]
                values[m, j] = values[m, j] * s[m] ** power
        frame: dict[str, np.ndarray] = {}
        rows = np.arange(len(cu))
        if is_packet_granularity(cu):
            # packets of a flow move with its timing: t' = t_first + s (t − t_first)
            t = cu.updates["event_time"].to_numpy().astype(np.float64)
            t_new = t.copy()
            for g in _groups(flows):
                t_new[g] = t[g[0]] + s[g] * (t[g] - t[g[0]])
            rows = sort_by_time(t_new)
            frame["event_time"] = t_new[rows]
        return _result(cu, labels, rows, self.name, {"eps": self.eps, "scale_min": float(s.min(initial=1.0)),
                                                     "scale_max": float(s.max(initial=1.0))},
                       values=values[rows], status=status[rows], frame_updates=frame)


@TRANSFORMS.register("rate-scaling", summary="time dilation of the attack's flows (AS-356)")
class RateScaling:
    """Dilate start times of malicious flows (all flows in a benign window) by ρ (AS-356)."""

    def __init__(self, *, factor_range: tuple[float, float]) -> None:
        lo, hi = factor_range
        if not 0 < lo <= hi:
            raise ValueError("need 0 < ρ_min ≤ ρ_max")
        self.factor_range = (float(lo), float(hi))
        self.name = "rate-scaling"

    def apply(self, cu: ColumnarUpdates, labels: UpdateLabels, rng: np.random.Generator) -> TransformResult:
        use("AS-356", by=__name__)
        lo, hi = self.factor_range
        rho = float(math.exp(rng.uniform(math.log(lo), math.log(hi))))
        t = cu.updates["event_time"].to_numpy().astype(np.float64)
        flows = _flow_ids(cu)
        mal = labels.malicious_rows()
        groups = _groups(flows)
        scope = [g for g in groups if mal[g].any()] or groups                 # attack flows, else every flow
        t_a = min(float(t[g[0]]) for g in scope)
        t_new = t.copy()
        for g in scope:
            start = t[g[0]]
            t_new[g] = t_a + rho * (start - t_a) + (t[g] - start)            # internal timing of a flow kept
        rows = sort_by_time(t_new)
        return _result(cu, labels, rows, self.name, {"rho": rho, "scope_flows": len(scope)},
                       frame_updates={"event_time": t_new[rows]})


@TRANSFORMS.register("reorder", summary="re-order within recorded reorder uncertainty (AS-357)")
class ReorderWithinUncertainty:
    """True time uniform in [t − r, t + r]; rows re-sorted; order inside a flow kept (AS-357)."""

    def __init__(self) -> None:
        self.name = "reorder"

    def apply(self, cu: ColumnarUpdates, labels: UpdateLabels, rng: np.random.Generator) -> TransformResult:
        use("AS-357", by=__name__)
        t = cu.updates["event_time"].to_numpy().astype(np.float64)
        r = np.nan_to_num(cu.updates["reorder_uncertainty_s"].to_numpy().astype(np.float64), nan=0.0)
        t_new = t + rng.uniform(-1.0, 1.0, size=len(t)) * np.maximum(r, 0.0)
        if is_packet_granularity(cu):
            # inside a flow the running state fixes the order: give its rows their sorted new times
            for g in _groups(_flow_ids(cu)):
                t_new[g] = np.sort(t_new[g])
        rows = sort_by_time(t_new)
        return _result(cu, labels, rows, self.name, {"moved_rows": int((rows != np.arange(len(rows))).sum())},
                       frame_updates={"event_time": t_new[rows]})


# ============================================================================== topology variation
def _benign_relations(cu: ColumnarUpdates, labels: UpdateLabels, *, avoid_malicious_entities: bool) -> np.ndarray:
    """Relations whose every update is known benign (and, optionally, whose entities are never malicious)."""
    rel = cu.updates["relation"].to_numpy().astype(np.int64)
    benign = labels.benign_rows()
    rels = np.unique(rel)
    ok = np.array([benign[rel == r].all() for r in rels], dtype=bool)
    if avoid_malicious_entities:
        ents = entity_columns(cu.updates)
        mal_ents = set(np.unique(cu.updates.loc[labels.malicious_rows(), ents].to_numpy()).tolist()) - {-1}
        e0, e1 = cu.updates["entity_0"].to_numpy(), cu.updates["entity_1"].to_numpy()
        for i, r in enumerate(rels):
            m = rel == r
            if ok[i] and ({int(e0[m][0]), int(e1[m][0])} & mal_ents):
                ok[i] = False
    return rels[ok]


@TRANSFORMS.register("hyperedge-dropout", summary="drop benign-only relations (AS-358)")
class HyperedgeDropout:
    """Each benign-only relation is dropped with probability `rate` (AS-358)."""

    def __init__(self, *, rate: float) -> None:
        if not 0.0 < rate < 1.0:
            raise ValueError("rate must be in (0, 1)")
        self.rate = rate
        self.name = "hyperedge-dropout"

    def apply(self, cu: ColumnarUpdates, labels: UpdateLabels, rng: np.random.Generator) -> TransformResult:
        use("AS-358", by=__name__)
        eligible = _benign_relations(cu, labels, avoid_malicious_entities=False)
        dropped = eligible[rng.random(len(eligible)) < self.rate]
        rel = cu.updates["relation"].to_numpy()
        rows = np.flatnonzero(~np.isin(rel, dropped))
        return _result(cu, labels, rows, self.name, {"rate": self.rate, "dropped_relations": len(dropped)})


@TRANSFORMS.register("rewire-benign", summary="new same-kind initiator for benign-only relations (AS-358)")
class RewireBenign:
    """Each benign-only relation away from the attack gets a new same-kind initiator w.p. `rate` (AS-358)."""

    def __init__(self, *, rate: float) -> None:
        if not 0.0 < rate < 1.0:
            raise ValueError("rate must be in (0, 1)")
        self.rate = rate
        self.name = "rewire-benign"

    def apply(self, cu: ColumnarUpdates, labels: UpdateLabels, rng: np.random.Generator) -> TransformResult:
        use("AS-358", by=__name__)
        eligible = _benign_relations(cu, labels, avoid_malicious_entities=True)
        chosen = eligible[rng.random(len(eligible)) < self.rate]
        ents = entity_columns(cu.updates)
        mal_ents = set(np.unique(cu.updates.loc[labels.malicious_rows(), ents].to_numpy()).tolist()) - {-1}
        kinds = cu.entities["kind"].to_numpy()
        rel = cu.updates["relation"].to_numpy()
        e0 = cu.updates["entity_0"].to_numpy().astype(np.int64).copy()
        e1 = cu.updates["entity_1"].to_numpy()
        # AS-584: candidates are entities of this window's own rows (never another split's entity).
        use("AS-584", by=__name__)
        own = set(np.unique(cu.updates[ents].to_numpy()).tolist()) - {-1}
        moved = 0
        for r in chosen.tolist():
            m = rel == r
            old, resp = int(e0[m][0]), int(e1[m][0])
            cand = [v for v in np.flatnonzero(kinds == kinds[old]).tolist()
                    if v in own and v not in (old, resp) and v not in mal_ents]
            if not cand:
                continue
            e0[m] = int(rng.choice(cand))
            moved += 1
        return _result(cu, labels, np.arange(len(cu)), self.name, {"rate": self.rate, "rewired_relations": moved},
                       frame_updates={"entity_0": e0.astype(cu.updates["entity_0"].dtype)})


def _service_table() -> dict[int, str]:
    """Destination port -> named service class (the Decoder's IANA table, AS-106)."""
    from nagahana.models.decoder.buckets import PORT_CLASSES

    table: dict[int, str] = {}
    for name, ports in PORT_CLASSES:
        for port in ports:
            table.setdefault(int(port), name)
    return table


_SERVICES: dict[int, str] = {}


def service_name(port: float) -> str:
    """Named service class of a destination port, else its RFC 6335 range class."""
    if not _SERVICES:
        _SERVICES.update(_service_table())
    if not math.isfinite(port):
        return "unknown"
    p = int(port)
    if p in _SERVICES:
        return _SERVICES[p]
    if p == 0:
        return "port-0"
    if p < 1024:
        return "well-known"
    if p < 49152:
        return "registered"
    return "dynamic"


ToolClass = tuple[str, str, int]


def tool_class_keys(cu: ColumnarUpdates, labels: UpdateLabels) -> list[ToolClass | None]:
    """Tool class (label class, service, protocol) of every row; None for rows of unknown label (AS-585)."""
    idx = ColumnIndex(cu.columns)
    contrib = _contributing(cu.status)
    jd, jp = idx.one("flow.dst_port"), idx.one("flow.protocol")
    mal, ben = labels.malicious_rows(), labels.benign_rows()
    out: list[ToolClass | None] = []
    for i in range(len(cu)):
        if mal[i]:
            label = f"attack:{labels.family}"
        elif ben[i]:
            label = "benign"
        else:
            out.append(None)
            continue
        port = float(cu.values[i, jd]) if (jd is not None and contrib[i, jd]) else math.nan
        proto = int(cu.values[i, jp]) if (jp is not None and contrib[i, jp]) else -1
        out.append((label, service_name(port), proto))
    return out


def _fingerprint_columns(cu: ColumnarUpdates, field_id: str) -> tuple[str, str] | None:
    """(value column, status column) of a fingerprint side field held as a vocabulary index, else None."""
    if field_id not in cu.side_fields:
        return None
    value_col, status_col = cu.side_fields[field_id]
    if value_col is None or value_col.startswith("@") or status_col not in cu.updates or value_col not in cu.updates:
        return None
    return value_col, status_col


class FingerprintClasses:
    """Fingerprint values of each tool class with their training frequencies (AS-585).

    `table[field][class key] = (values, probabilities)`; built from real training windows only by the
    data preparation (`build`), stored as JSON (`to_json` / `from_json`).
    """

    def __init__(self, table: dict[str, dict[ToolClass, tuple[tuple[str, ...], tuple[float, ...]]]]) -> None:
        self.table = table

    @classmethod
    def build(cls, windows: Sequence[tuple[ColumnarUpdates, UpdateLabels]], fields: Sequence[str]) -> FingerprintClasses:
        """Count each fingerprint per tool class over the given (training) windows."""
        use("AS-585", by=__name__)
        counts: dict[str, dict[ToolClass, dict[str, int]]] = {f: {} for f in fields}
        for cu, labels in windows:
            keys = tool_class_keys(cu, labels)
            for f in fields:
                cols = _fingerprint_columns(cu, f)
                if cols is None:
                    continue
                st = cu.updates[cols[1]].to_numpy().astype(np.uint8)
                vals = cu.updates[cols[0]].to_numpy()
                vocab = cu.vocab.get(f, [])
                for i, key in enumerate(keys):
                    if key is None or st[i] not in _CONTRIB_CODES:
                        continue
                    v = int(vals[i])
                    if not 0 <= v < len(vocab):
                        continue
                    bucket = counts[f].setdefault(key, {})
                    bucket[vocab[v]] = bucket.get(vocab[v], 0) + 1
        table: dict[str, dict[ToolClass, tuple[tuple[str, ...], tuple[float, ...]]]] = {}
        for f, per in counts.items():
            table[f] = {}
            for key, c in per.items():
                names = tuple(sorted(c))
                tot = float(sum(c.values()))
                table[f][key] = (names, tuple(c[n] / tot for n in names))
        return cls(table)

    def to_json(self) -> dict[str, list[list[Any]]]:
        """Plain JSON data of the table (sorted, so equal tables give equal files)."""
        return {f: [[list(k), list(v[0]), list(v[1])] for k, v in sorted(per.items())] for f, per in sorted(self.table.items())}

    @classmethod
    def from_json(cls, data: dict[str, list[list[Any]]]) -> FingerprintClasses:
        """Inverse of `to_json`."""
        table: dict[str, dict[ToolClass, tuple[tuple[str, ...], tuple[float, ...]]]] = {}
        for f, rows in data.items():
            table[f] = {}
            for key, names, probs in rows:
                table[f][(str(key[0]), str(key[1]), int(key[2]))] = (tuple(str(n) for n in names),
                                                                     tuple(float(p) for p in probs))
        return cls(table)


@TRANSFORMS.register("tool-fingerprint-swap", summary="client tool fingerprints swapped inside their tool class (AS-585)")
class ToolFingerprintSwap:
    """Swap each row's client fingerprint for another of its tool class (module docstring; AS-585)."""

    def __init__(self, *, table: FingerprintClasses) -> None:
        self.table = table
        self.name = "tool-fingerprint"

    def apply(self, cu: ColumnarUpdates, labels: UpdateLabels, rng: np.random.Generator) -> TransformResult:
        use("AS-585", by=__name__)
        keys = tool_class_keys(cu, labels)
        frame: dict[str, np.ndarray] = {}
        new_vocab: dict[str, list[str]] = {}
        swapped = 0
        for f, per in self.table.table.items():
            cols = _fingerprint_columns(cu, f)
            if cols is None:
                continue
            value_col, status_col = cols
            st = cu.updates[status_col].to_numpy().astype(np.uint8)
            vals = cu.updates[value_col].to_numpy().astype(np.int64).copy()
            vocab = list(cu.vocab.get(f, []))
            index = {v: i for i, v in enumerate(vocab)}
            for i, key in enumerate(keys):
                if key is None or st[i] not in _CONTRIB_CODES or key not in per or not 0 <= vals[i] < len(vocab):
                    continue
                names, probs = per[key]
                current = vocab[vals[i]]
                choices = [(n, p) for n, p in zip(names, probs, strict=True) if n != current]
                if not choices:
                    continue
                w = np.asarray([p for _, p in choices], dtype=np.float64)
                pick = choices[int(rng.choice(len(choices), p=w / w.sum()))][0]
                if pick not in index:
                    index[pick] = len(vocab)
                    vocab.append(pick)
                vals[i] = index[pick]
                swapped += 1
            frame[value_col] = vals
            new_vocab[f] = vocab
        if swapped == 0:
            raise NotApplicable("no fingerprint of the window has an alternative in its tool class")
        res = _result(cu, labels, np.arange(len(cu)), self.name, {"swapped": swapped}, frame_updates=frame)
        res.updates.vocab.update(new_vocab)
        return res


# ============================================================================== config → transforms
def observability_transforms(cfg: GeneratorConfig, policy: GeneratorPolicy | None = None) -> list[Transform]:
    """Observability sliding (AS-350 ... AS-353) from the Generator config and the run's tables."""
    levels = tuple(Level(lv) for lv in policy.drop_levels) if policy is not None else (Level.PACKET,)
    keep = frozenset(policy.flow_only_fields) if policy is not None else IPFIX_BIFLOW_FIELDS
    return [
        DropFields(levels=levels, status=ObservationStatus.NOT_OBSERVABLE, row_fraction=1.0),
        FlowOnlyExport(keep=keep),
        PacketSampling(rates=cfg.sampling_rates),
        SensorHiding(fraction=cfg.sensor_hide_fraction),
    ]


def signature_transforms(cfg: GeneratorConfig, policy: GeneratorPolicy | None = None) -> list[Transform]:
    """Signature variation (AS-354 ... AS-357) from the Generator config and the run's tables."""
    classes = dict(policy.service_alias_classes) if policy is not None else SERVICE_ALIAS_CLASSES
    ephemeral = tuple(policy.ephemeral_ports) if policy is not None else EPHEMERAL_PORTS
    return [
        PortRemap(classes=classes, ephemeral=(int(ephemeral[0]), int(ephemeral[1]))),
        TimingJitter(eps=cfg.jitter_rel),
        RateScaling(factor_range=cfg.rate_scale),
        ReorderWithinUncertainty(),
    ]


def topology_transforms(cfg: GeneratorConfig) -> list[Transform]:
    """Topology variation (AS-358) from the Generator config."""
    return [HyperedgeDropout(rate=cfg.edge_dropout), RewireBenign(rate=cfg.rewire_rate)]

