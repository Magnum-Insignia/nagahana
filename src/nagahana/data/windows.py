"""Windows of the event log: entity tables, positions, triggers, field matrices and labels of one window.

Purpose
-------
`ColumnarUpdates` is the durable event log (AS-11). Training reads it in **windows**: runs of
consecutive state updates (in event-time order) of at most `TrainingConfig.window_updates` updates
and at most `TrainingConfig.max_entities` entities. For one window this module builds everything of
`models/batch.py` except the structure that the graph builder owns:

    fields          the value/status matrices in the canonical column layout (`CANONICAL_COLUMNS`),
                    with stable field slots (`COLUMN_SLOTS`, P-22)
    entity table    window-local entities: kind code (`vocab.NODE_KIND_CODE`) and internal flag
    positions       2 per update (initiator, responder; AS-41), sorted by (time, update, role), with
                    `next_index` / `next_dt` (the transition target of the world model)
    triggers        fixed cadence (AS-12), `entity_latest` as of each trigger
    labels          per update (malicious, stage, technique slot), per entity the infiltration time
                    (AS-18) and the malicious share per trigger (build-spec §4b.4)
    structure       from `graph.window.build_window_structure` (engineer A): relations, planes,
                    contact matrices, as-of local subgraphs

`data/collate.py` pads and stacks `WindowItem`s into `WindowBatch` + `LabelBatch`.

Owner sources, decisions, assumptions
-------------------------------------
D-30 (one record = one state update), D-41 (absence never zero), D-49 (time, not index: positions
carry times; no index encodings), D-50 (no clock features; the origin is kept only for them),
AS-12 (trigger cadence), AS-18 (infiltration), AS-41 (2 positions per update), AS-11 (event log),
build-spec §4b.4–4b.5; new AS-317 … AS-324 and AS-331 (`docs/assumptions/data.md`).

Maths and definitions
---------------------
Ordering. Updates are sorted by event time t (stable on the log's order). Optional out-of-order
augmentation (§4b.5, AS-320): t'_i = t_i + U(−r_i, r_i), r_i = `reorder_uncertainty_s` (0 if
unknown), then a stable re-sort; every structure below is built from t', so it stays consistent.

Window plan (AS-317). Greedy over the sorted log: a window takes updates while it has fewer than
`window_updates` updates and adding the next update's entities keeps it within `max_entities`.
Windows do not overlap. The plan depends on the order of updates, never on their content.

Times. origin = t of the window's first update (float64 epoch seconds); every time in the window is
float64 seconds relative to it. A source without times (CIC-IoT-2023 CSV) has all times 0 and
origin 0 (AS-307): it supplies no temporal information, so none is invented.

Positions (AS-41). For update u (in window order) and role ρ ∈ {initiator 0, responder 1} with an
entity e_{u,ρ} ≥ 0, one position (time t_u, update u, role ρ). Positions are listed in (t, u, ρ) order
and padded at the end. For position p of entity e:
    next_index[p] = min{q > p : entity[q] = e}  (−1 if none),   next_dt[p] = time[q] − time[p] ≥ 0.

Triggers (AS-12, AS-318). Fixed cadence c = `ForecasterConfig.window_seconds` on the epoch grid:
    trigger times = { k·c : k ∈ ℤ, t_first ≤ k·c ≤ t_last }   (relative to origin after selection),
so windows of one stream share trigger instants and no volume can move a trigger. A window shorter
than c may hold no trigger (mask all False). The capped priority triggers of AS-12 depend on the
model's marginal energy and are added at inference, not here.
    entity_latest[m, v] = max{ p : entity[p] = v, time[p] ≤ τ_m }  (−1 if none)
uses positions at or before the trigger only (tested).

Labels (never inputs).
- update_malicious / stage / technique: from the mapped label table (`data/labels.py`); technique
  IDs become slots with `labels.technique_slot(·, ForecasterConfig.n_techniques)`.
- entity_infiltrated_at[v] (AS-18, AS-319): the first time t at which v is the *internal actor*
  (`actor_role`) of a malicious update whose stage is in `vocab.INFILTRATION_STAGES`, over the whole
  source, kept if t ≤ H with the label horizon H = t_last + K·c (K = `ForecasterConfig.horizon_k`):
  the forecast targets reach K cadence steps past the window. +inf otherwise. May be negative
  (infiltrated before the window began).
- entity_malicious_share[v, m] (§4b.4, AS-321): over updates u of the window with t_u ≤ τ_m in which
  v is initiator or responder and whose malicious label is known,
      share = Σ malicious_u / #u,   NaN if there is none
  (the trailing window is "from the window start to the trigger", as `testing/synthetic.py`).
- family: the most frequent family among the window's malicious updates; "benign" if it has only
  benign ones; "unknown" otherwise.

Internal flag (AS-322). From the entity table's `internal` column when the adapter wrote one (CSV
adapters do); else `host` ⇒ internal, `external` ⇒ not (the PCAP adapter decides kind by its own
network rule); else the address inside the key (a service key "<addr>:<port>/<proto>") by the RFC
1918 / RFC 4193 / link-local / loopback rule (`ingest.csv_flows.DEFAULT_INTERNAL_NETWORKS`).

Invariants (tests/test_data_windows.py): positions sorted; next_index correct; entity_latest only
uses positions ≤ trigger time; labels never in the field matrices; NaN only in excluded cells.

Extension points: `structure_fn` (any function with the `build_window_structure` signature);
`perturb_rng` for the out-of-order augmentation; new columns are appended to `COLUMN_SLOTS`.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from nagahana.core.errors import InvariantViolation
from nagahana.datamodel.columnar import CODE_NOT_SUPPLIED, STATUS_ORDER, Column, ColumnarUpdates, columns_for
from nagahana.datamodel.fields import CATALOGUE
from nagahana.datamodel.status import CONTRIBUTING
from nagahana.datamodel.versioning import FieldIndex
from nagahana.governance.assumptions import assume
from nagahana.graph.window import build_window_structure
from nagahana.ingest.csv_flows import is_internal_address, parse_networks
from nagahana.ingest.pcap import PAYLOAD_BIN_LABELS
from nagahana.models.config import NagaHanaConfig
from nagahana.models.vocab import COLUMN_KIND_CODE, INFILTRATION_STAGES, NODE_KIND_CODE

#: Stable column → field-slot table (P-22, AS-323). APPEND ONLY: a column's slot is its index here,
#: frozen once a model is trained. Initial order = the catalogue's matrix columns on 2026-10-02.
COLUMN_SLOTS: tuple[str, ...] = (
    "flow.src_port", "flow.dst_port", "flow.protocol", "flow.tcp_flags", "flow.flag_count.syn",
    "flow.flag_count.ack", "flow.flag_count.fin", "flow.flag_count.rst", "flow.flag_count.psh",
    "flow.flag_count.urg", "flow.bytes_fwd", "flow.bytes_bwd", "flow.packets_fwd", "flow.packets_bwd",
    "flow.duration", "flow.iat_mean", "flow.iat_var", "flow.iat_max", "flow.bidir_ratio", "pkt.ttl_mean",
    "pkt.ttl_var", "pkt.tcp_window_init_fwd", "pkt.tcp_window_init_bwd", "pkt.ip_df_count", "pkt.ip_mf_count",
    "pkt.payload_size_hist[0]", "pkt.payload_size_hist[1]", "pkt.payload_size_hist[2]", "pkt.payload_size_hist[3]",
    "pkt.payload_size_hist[4]", "pkt.payload_size_hist[5]", "pkt.payload_size_hist[6]", "pkt.payload_size_hist[7]",
    "pkt.retransmissions", "derived.portscan_sequential", "derived.portscan_random", "proto.dns.qtype",
    "proto.dns.rcode", "proto.kerberos.msg_type", "proto.icmp.type", "proto.arp.opcode",
    "ot.modbus.function_code", "ot.modbus.unit_id", "ot.modbus.register_start", "ot.modbus.register_count",
    "ot.dnp3.function_code", "ot.dnp3.object_group", "ot.iec104.type_id", "ot.iec104.cot",
    # appended 2026-10-02 (lead-approved catalogue fields, AS-303)
    "flow.payload_bytes_fwd", "flow.payload_bytes_bwd", "flow.packets_total",
    "flow.end_reason", "flow.unanswered",                                       # D-53
    # Data-model version 0.2: matrix fields of the superset formats, in catalogue order (fields.py).
    "flow.conn_state", "flow.missed_bytes", "flow.vlan", "flow.ip_tos", "flow.sampling_rate",
    "flow.forwarding_status", "flow.tcp_flags_fwd", "flow.tcp_flags_bwd", "flow.app_proto",
    "pkt.ttl_min", "pkt.ttl_max", "pkt.ip_len_min", "pkt.ip_len_max", "proto.icmp.code",
    "dev.interval", "dev.if_in_octets", "dev.if_out_octets", "dev.if_in_packets", "dev.if_out_packets",
    "dev.if_errors", "dev.if_discards", "dev.if_status", "dev.capture_received", "dev.capture_dropped",
    "proto.dns.qclass", "proto.dns.flags", "proto.dns.rtt", "proto.dns.answer_count", "proto.dns.ttl_min",
    "proto.dns.query_length", "proto.dns.query_entropy",
    "proto.http.method", "proto.http.status_code", "proto.http.request_body_len", "proto.http.response_body_len",
    "proto.http.uri_length",
    "proto.tls.version", "proto.tls.established", "proto.tls.alert", "proto.tls.cert_validity",
    "proto.tls.cert_self_signed",
    "proto.ssh.auth_success", "proto.ssh.auth_attempts", "proto.kerberos.error_code", "proto.kerberos.etype",
    "auth.activity", "auth.result", "auth.logon_type", "auth.protocol", "auth.method", "auth.failure_status",
    "auth.elevated",
    "alert.severity", "alert.signature", "alert.category", "event.action",
    "ot.modbus.exception_code", "ot.dnp3.iin", "proto.mqtt.message_type", "proto.smb.command",
    "proto.smb.file_action", "proto.dhcp.message_type", "file.size", "proto.ntp.mode",
)


def _canonical_columns() -> tuple[Column, ...]:
    """The canonical matrix columns, in `COLUMN_SLOTS` order (histogram bins of the PCAP adapter)."""
    fields: list[str] = []
    for name in COLUMN_SLOTS:
        fid = name.split("[", 1)[0]
        if fid not in fields:
            fields.append(fid)
    cols = columns_for(fields, histogram_bins={"pkt.payload_size_hist": PAYLOAD_BIN_LABELS})
    if tuple(c.name for c in cols) != COLUMN_SLOTS:
        raise InvariantViolation("COLUMN_SLOTS no longer matches the catalogue's columns; append, never reorder.")
    return cols


CANONICAL_COLUMNS: tuple[Column, ...] = _canonical_columns()
CANONICAL_INDEX: dict[str, int] = {c.name: j for j, c in enumerate(CANONICAL_COLUMNS)}
#: Column kind codes [C] (vocab.COLUMN_KIND_CODE).
CANONICAL_KIND_CODES: tuple[int, ...] = tuple(COLUMN_KIND_CODE[c.kind] for c in CANONICAL_COLUMNS)
_CONTRIB = np.array([i for i, s in enumerate(STATUS_ORDER) if s in CONTRIBUTING], dtype=np.int64)


def field_slots(n_slots: int) -> np.ndarray:
    """Slot of every canonical column for a field-slot table of `n_slots` rows (P-22 `FieldIndex`)."""
    if n_slots < len(COLUMN_SLOTS):
        raise InvariantViolation(f"FieldEncoderConfig.n_slots={n_slots} < {len(COLUMN_SLOTS)} canonical columns.")
    index = FieldIndex(COLUMN_SLOTS, reserved=n_slots - len(COLUMN_SLOTS))
    return np.array([index.slot(n) for n in COLUMN_SLOTS], dtype=np.int64)


def uncatalogued_columns() -> list[str]:
    """Catalogue matrix fields that have no slot yet (must be appended to `COLUMN_SLOTS`)."""
    have = {n.split("[", 1)[0] for n in COLUMN_SLOTS}
    from nagahana.datamodel.columnar import MATRIX_KINDS

    return [f for f, s in CATALOGUE.items() if s.kind in MATRIX_KINDS and f not in have]


# ====================================================================================== sources
@dataclass
class SourceData:
    """One source of the training corpus: its event log, its mapped labels and where it comes from.

    labels: mapped label table (`data.labels.LABEL_COLUMNS`), one row per update (by `seq`).
    network: the network the traffic was captured on (for leave-one-network-out, AS-35).
    origin: "real" or "generated"; derived_from: for generated sources, the real source id.
    """

    updates: ColumnarUpdates
    labels: pd.DataFrame
    network: str
    dataset: str = ""
    origin: str = "real"
    derived_from: str | None = None

    @property
    def source_id(self) -> str:
        return self.updates.source_id


@dataclass
class PreparedSource:
    """Per-source arrays in sorted (event-time) order, computed once and shared by all its windows."""

    data: SourceData
    order: np.ndarray            # [n] row of the log at each sorted position
    time: np.ndarray             # [n] float64 epoch seconds (0 when the source has no times)
    timeless: bool
    reorder: np.ndarray          # [n] float64 reorder uncertainty (0 when unknown)
    ents: np.ndarray             # [n, 3] global entity rows (initiator, responder, service)
    src_cols: np.ndarray         # source column index of each canonical column (−1 = the source lacks it)
    entity_kind: np.ndarray      # [V_src] kind codes
    entity_internal: np.ndarray  # [V_src] bool
    malicious: np.ndarray        # [n] float32 (sorted order)
    stage: np.ndarray            # [n] int64
    technique: np.ndarray        # [n] object (ATT&CK ID or "")
    family: np.ndarray           # [n] object
    first_infiltration: np.ndarray  # [V_src] float64 epoch seconds (+inf never), AS-18 / AS-319


def _internal_flags(cu: ColumnarUpdates) -> np.ndarray:
    """AS-322 (module docstring)."""
    ent = cu.entities
    if "internal" in ent:
        return ent["internal"].to_numpy(dtype=bool)
    nets = parse_networks(None)
    out = np.zeros(len(ent), dtype=bool)
    for i, (kind, key) in enumerate(zip(ent["kind"].astype(str), ent["key"].astype(str), strict=True)):
        if kind == "host":
            out[i] = True
        elif kind == "external":
            out[i] = False
        else:
            addr = key.rsplit(":", 1)[0] if kind == "service" else key
            out[i] = bool(is_internal_address(addr, nets))
    return out


def prepare_source(data: SourceData) -> PreparedSource:
    """Sort the log by event time and gather the per-source arrays (see `PreparedSource`)."""
    for a in ("AS-11", "AS-18"):
        assume(a, by=__name__)
    cu = data.updates
    n = len(cu)
    if len(data.labels) != n:
        raise InvariantViolation(f"{cu.source_id}: {len(data.labels)} label rows for {n} updates.")
    labels = data.labels.set_index("seq").sort_index()
    t = cu.updates["event_time"].to_numpy(dtype=np.float64)
    timeless = not np.isfinite(t).any()
    if not timeless and not np.isfinite(t).all():
        raise InvariantViolation(f"{cu.source_id}: some updates have no event time; a window cannot place them.")
    order = np.argsort(t if not timeless else np.zeros(n), kind="stable")
    time = (t if not timeless else np.zeros(n))[order]
    reorder = np.nan_to_num(cu.updates["reorder_uncertainty_s"].to_numpy(dtype=np.float64), nan=0.0)[order]
    ents = np.stack([
        cu.updates[f"entity_{k}"].to_numpy(dtype=np.int64) if f"entity_{k}" in cu.updates else np.full(n, -1, dtype=np.int64)
        for k in range(3)
    ], axis=1)[order]
    src_index = {c.name: j for j, c in enumerate(cu.columns)}
    src_cols = np.array([src_index.get(name, -1) for name in COLUMN_SLOTS], dtype=np.int64)
    kinds = np.array([NODE_KIND_CODE[str(k)] for k in cu.entities["kind"]], dtype=np.int64)
    internal = _internal_flags(cu)
    mal = labels["malicious"].to_numpy(dtype=np.float32)[order]
    stage = labels["stage"].to_numpy(dtype=np.int64)[order]
    tech = labels["technique"].to_numpy(dtype=object)[order]
    fam = labels["family"].to_numpy(dtype=object)[order]
    actor = labels["actor_role"].to_numpy(dtype=np.int64)[order]
    # AS-18 / AS-319: the internal actor of a malicious update in an infiltration stage
    first = np.full(len(cu.entities), np.inf)
    hit = (mal == 1.0) & np.isin(stage, list(INFILTRATION_STAGES)) & (actor >= 0) & (actor <= 1)
    if hit.any():
        rows = np.nonzero(hit)[0]
        who = ents[rows, actor[rows]]
        ok = who >= 0
        rows, who = rows[ok], who[ok]
        inside = internal[who]
        np.minimum.at(first, who[inside], time[rows[inside]])
    return PreparedSource(
        data=data, order=order, time=time, timeless=timeless, reorder=reorder, ents=ents, src_cols=src_cols,
        entity_kind=kinds, entity_internal=internal, malicious=mal, stage=stage, technique=tech, family=fam,
        first_infiltration=first,
    )


def plan_windows(src: PreparedSource, *, window_updates: int, max_entities: int) -> list[tuple[int, int]]:
    """Non-overlapping windows [start, stop) over the sorted log (AS-317)."""
    assume("AS-41", by=__name__)
    n = len(src.order)
    if window_updates < 1 or max_entities < 2:
        raise InvariantViolation("a window needs ≥ 1 update and room for ≥ 2 entities")
    out: list[tuple[int, int]] = []
    start = 0
    seen: set[int] = set()
    for i in range(n):
        new = {int(x) for x in src.ents[i] if x >= 0} - seen
        if i > start and (i - start >= window_updates or len(seen) + len(new) > max_entities):
            out.append((start, i))
            start, seen = i, set()
            new = {int(x) for x in src.ents[i] if x >= 0}
        if len(new) > max_entities:
            raise InvariantViolation(f"update {i} touches more than max_entities={max_entities} entities")
        seen |= new
    if start < n:
        out.append((start, n))
    return out


# ====================================================================================== one window
@dataclass
class WindowItem:
    """One window before padding (numpy arrays; shapes in comments). See the module docstring."""

    source_id: str
    network: str
    start: int
    stop: int
    origin: float
    timeless: bool
    rows: np.ndarray                 # [U] rows of the event log
    update_time: np.ndarray          # [U] float64 relative
    update_entities: np.ndarray      # [U, 3] window-local entity indices, −1 none
    reorder: np.ndarray              # [U] float32
    values: np.ndarray               # [U, C] float32 (NaN where not contributing)
    status: np.ndarray               # [U, C] int64
    entity_rows: np.ndarray          # [V] global entity rows
    entity_kind: np.ndarray          # [V] int64
    entity_internal: np.ndarray      # [V] bool
    pos_entity: np.ndarray           # [P_real]
    pos_time: np.ndarray             # [P_real] float64
    pos_update: np.ndarray           # [P_real]
    pos_role: np.ndarray             # [P_real]
    next_index: np.ndarray           # [P_real]
    next_dt: np.ndarray              # [P_real] float64
    trigger_time: np.ndarray         # [M] float64 relative
    entity_latest: np.ndarray        # [M, V]
    update_malicious: np.ndarray     # [U] float32
    update_stage: np.ndarray         # [U] int64
    update_technique: np.ndarray     # [U] int64
    entity_infiltrated_at: np.ndarray  # [V] float64 relative (+inf never)
    entity_malicious_share: np.ndarray  # [V, M] float32
    family: str
    families: frozenset[str]
    label_horizon: float = math.inf  # relative time up to which infiltration labels are observed (censoring)
    structure: Any = None            # graph.window.WindowStructure
    extra: dict[str, Any] = field(default_factory=dict)


StructureFn = Callable[..., Any]


def default_structure_fn() -> StructureFn:
    """The graph builder's `graph.window.build_window_structure` (planes, hyperedges, local subgraphs)."""
    return build_window_structure


def _positions(ents: np.ndarray, times: np.ndarray) -> tuple[np.ndarray, ...]:
    """Positions in (time, update, role) order and their next-state links (module docstring)."""
    u_n = ents.shape[0]
    upd = np.repeat(np.arange(u_n), 2)
    role = np.tile(np.array([0, 1]), u_n)
    ent = ents[:, :2].reshape(-1)
    keep = ent >= 0
    upd, role, ent = upd[keep], role[keep], ent[keep]
    ptime = times[upd]
    p_n = ent.shape[0]
    nxt = np.full(p_n, -1, dtype=np.int64)
    ndt = np.zeros(p_n, dtype=np.float64)
    last: dict[int, int] = {}
    for p in range(p_n - 1, -1, -1):          # backward sweep: the next position of the same entity
        e = int(ent[p])
        q = last.get(e)
        if q is not None:
            nxt[p] = q
            ndt[p] = ptime[q] - ptime[p]
        last[e] = p
    return ent.astype(np.int64), ptime, upd.astype(np.int64), role.astype(np.int64), nxt, ndt


def _triggers(t_first: float, t_last: float, cadence: float) -> np.ndarray:
    """Epoch-grid trigger instants in [t_first, t_last] (AS-12, AS-318)."""
    k0 = math.ceil(t_first / cadence)
    k1 = math.floor(t_last / cadence)
    return np.arange(k0, k1 + 1, dtype=np.float64) * cadence if k1 >= k0 else np.zeros(0)


def _entity_latest(pos_entity: np.ndarray, pos_time: np.ndarray, trig: np.ndarray, v_n: int) -> np.ndarray:
    """[M, V] latest position of each entity with time ≤ each trigger (−1 none)."""
    out = np.full((trig.shape[0], v_n), -1, dtype=np.int64)
    latest = np.full(v_n, -1, dtype=np.int64)
    p = 0
    for m, tau in enumerate(trig.tolist()):
        while p < pos_entity.shape[0] and pos_time[p] <= tau:
            latest[pos_entity[p]] = p
            p += 1
        out[m] = latest
    return out


def build_window(
    src: PreparedSource,
    start: int,
    stop: int,
    cfg: NagaHanaConfig,
    *,
    structure_fn: StructureFn | None = None,
    perturb_rng: np.random.Generator | None = None,
    trigger_window: tuple[float, float] | None = None,
    label_limit: float = math.inf,
) -> WindowItem:
    """Build one window (module docstring). `structure_fn` defaults to engineer A's builder.

    trigger_window: epoch [lo, hi) whose cadence grid points become this window's triggers (stream
        order, D-51: [first update, next window's first update), so every grid point of a stream
        belongs to exactly one window). Default: [first update, last update].
    label_limit: epoch time the label look-ahead never passes (the next window of another split,
        AS-334); the horizon is min(last update or trigger + K·c, label_limit).
    """
    for a in ("AS-12", "AS-18", "AS-41"):
        assume(a, by=__name__)
    if not 0 <= start < stop <= len(src.order):
        raise InvariantViolation(f"bad window [{start}, {stop}) for {len(src.order)} updates")
    sl = slice(start, stop)
    times = src.time[sl].copy()
    idx = np.arange(start, stop)
    # ---- out-of-order augmentation (§4b.5, AS-320): jitter within the recorded uncertainty, re-sort
    if perturb_rng is not None and not src.timeless:
        r = src.reorder[sl]
        times = times + perturb_rng.uniform(-1.0, 1.0, size=times.shape) * r
        o = np.argsort(times, kind="stable")
        times, idx = times[o], idx[o]
    rows = src.order[idx]
    origin = 0.0 if src.timeless else float(times[0])
    rel = times - origin                                           # float64 relative seconds (D-49 note)

    # ---- window-local entity table, in order of first appearance
    g_ents = src.ents[idx]                                          # [U, 3] global
    flat = g_ents.reshape(-1)
    seen_order = pd.unique(flat[flat >= 0])
    # global → local by an index lookup; −1 (none) is never in the index, so it stays −1
    ents = pd.Index(seen_order).get_indexer(flat).astype(np.int64).reshape(g_ents.shape)
    entity_rows = np.asarray(seen_order, dtype=np.int64)
    v_n = entity_rows.shape[0]
    if v_n > cfg.training.max_entities:
        raise InvariantViolation(f"window has {v_n} entities > max_entities={cfg.training.max_entities}")

    # ---- field matrices in the canonical layout; excluded cells NaN (D-41)
    cu = src.data.updates
    u_n = rows.shape[0]
    c_n = len(COLUMN_SLOTS)
    values = np.full((u_n, c_n), np.nan, dtype=np.float32)
    status = np.full((u_n, c_n), CODE_NOT_SUPPLIED, dtype=np.int64)
    have = src.src_cols >= 0
    status[:, have] = cu.status[rows][:, src.src_cols[have]]
    vals = cu.values[rows][:, src.src_cols[have]]
    contributing = np.isin(status[:, have], _CONTRIB)
    values[:, have] = np.where(contributing, vals, np.nan).astype(np.float32)

    # ---- positions (AS-41)
    pos_entity, pos_time, pos_update, pos_role, nxt, ndt = _positions(ents, rel)

    # ---- triggers (AS-12, AS-318)
    cadence = cfg.forecaster.window_seconds
    if src.timeless:
        trig_abs = np.zeros(1)
    elif trigger_window is not None:
        lo, hi = trigger_window
        trig_abs = _triggers(lo, hi, cadence)
        trig_abs = trig_abs[trig_abs < hi]                          # half-open [lo, hi)
    else:
        trig_abs = _triggers(float(times[0]), float(times[-1]), cadence)
    trig = trig_abs - origin
    latest = _entity_latest(pos_entity, pos_time, trig, v_n)

    # ---- labels (never inputs)
    mal = src.malicious[idx]
    stage = src.stage[idx]
    from nagahana.data.labels import technique_slot

    n_tech = cfg.forecaster.n_techniques
    tech = np.array([technique_slot(str(x), n_tech) for x in src.technique[idx]], dtype=np.int64)
    last_t = max(float(times[-1]), float(trig_abs.max()) if trig_abs.size else -math.inf)
    horizon = min(last_t + cfg.forecaster.horizon_k * cadence, label_limit)          # AS-319, AS-334
    first = src.first_infiltration[entity_rows]
    infil = np.where(first <= horizon, first - origin, np.inf)
    share = np.full((v_n, trig.shape[0]), np.nan, dtype=np.float32)
    known = ~np.isnan(mal)
    mal64 = np.nan_to_num(mal.astype(np.float64))
    for m, tau in enumerate(trig.tolist()):                         # §4b.4, AS-321
        upto = known & (rel <= tau)
        num = np.zeros(v_n)
        den = np.zeros(v_n)
        for role in (0, 1):                                          # the entity as initiator or responder
            e = ents[upto, role]
            ok = e >= 0
            num += np.bincount(e[ok], weights=mal64[upto][ok], minlength=v_n)
            den += np.bincount(e[ok], minlength=v_n)
        share[:, m] = np.where(den > 0, num / np.maximum(den, 1.0), np.nan).astype(np.float32)
    fam = src.family[idx]
    mal_fams = pd.Series(fam[mal == 1.0]).value_counts()
    family = str(mal_fams.index[0]) if len(mal_fams) else ("benign" if (mal == 0.0).any() else "unknown")
    families = frozenset(str(f) for f in fam[known])

    # ---- structure from the graph builder (engineer A)
    fn = structure_fn or default_structure_fn()
    dcol, pcol = CANONICAL_INDEX["flow.dst_port"], CANONICAL_INDEX["flow.protocol"]
    structure = fn(
        update_time=rel, update_entities=ents, dst_port=values[:, dcol].astype(np.float64),
        protocol=values[:, pcol].astype(np.float64), entity_kind=src.entity_kind[entity_rows],
        pos_entity=pos_entity, pos_time=pos_time, cfg=cfg.graph, pos_update=pos_update,  # max_members: the builder's own
    )
    return WindowItem(
        source_id=src.data.source_id, network=src.data.network, start=start, stop=stop, origin=origin,
        timeless=src.timeless, rows=rows, update_time=rel, update_entities=ents,
        reorder=src.reorder[idx].astype(np.float32), values=values, status=status, entity_rows=entity_rows,
        entity_kind=src.entity_kind[entity_rows], entity_internal=src.entity_internal[entity_rows],
        pos_entity=pos_entity, pos_time=pos_time, pos_update=pos_update, pos_role=pos_role, next_index=nxt,
        next_dt=ndt, trigger_time=trig, entity_latest=latest, update_malicious=mal.astype(np.float32),
        update_stage=stage, update_technique=tech, entity_infiltrated_at=infil, entity_malicious_share=share,
        label_horizon=horizon - origin,
        family=family, families=families, structure=structure,
    )


__all__ = [
    "CANONICAL_COLUMNS", "CANONICAL_INDEX", "CANONICAL_KIND_CODES", "COLUMN_SLOTS", "PreparedSource", "SourceData",
    "WindowItem", "build_window", "default_structure_fn", "field_slots", "plan_windows", "prepare_source",
    "uncatalogued_columns",
]
