"""Design columns from the canonical window tables: the features NagaHana sees, for a linear model.

Input. The value/status matrices of a window in the canonical column layout (data/windows.py,
`CANONICAL_COLUMNS`, the flow-level, packet-level and observable-region fields of the data model) plus
the times, entities and flow keys of the updates. Labels never enter a design column.

Per-update columns (the detection unit, one state update; prefix "u|"), for canonical column c with
status m and value x:

    numeric (CONTINUOUS, COUNT, HISTOGRAM bin)   v = clip(sign(x) log(1 + |x|), +-L)      if m contributes
    bitmask                                       bit_k(x) for k < max_bits                if m contributes
    categorical (ports, protocol, codes)          1[x = code] for every vocabulary code,
                                                  1[x not in the vocabulary] ("other")     if m contributes
    status                                        1[m = s] for s in STALE, LOW_RELIABILITY, NOT_SUPPLIED,
                                                  NOT_OBSERVABLE (OBSERVED is the reference level)

L is the FieldEncoder's clip (AS-100) and the transform is its signed log1p (AS-31), so a numeric value
reaches the baseline as it reaches NagaHana. A value whose status does not contribute is NaN in the raw
design and is filled with the training location by the standardiser, while its status indicator carries
the absence: "not supplied" is never encoded as zero (D-41; AS-501, AS-502). Ports and other codes are
categories, never magnitudes (datamodel/fields.py); the vocabulary of a categorical column is the set
of codes seen at least `min_category_count` times in the training rows, most frequent first, capped at
`max_categories` (AS-503).

Window columns (the forecasting unit, one trigger at tau on the cadence grid; prefix "w<l>|" for the
cadence window ((g - l - 1) w, (g - l) w] with g = tau / w, l = 0 ... L). A window aggregates the updates
of the trigger's segment that fall inside it, so a trigger sees exactly the updates at or before its
time that the carried Environment of its segment holds (D-51; AS-506):

    window       unavailable (the window lies before the segment), empty (inside, no update), coverage
                 (share of the window inside the segment), updates (log(1 + count))
    state        flows, packets, bytes_ip, bytes_payload, dst_hosts, dst_ports (log(1 + count) when
                 `log_count_states`), syn_share, rst_share, failed_share, and one "undefined" indicator per
                 state feature (1 exactly where the state value is NaN)
    numeric c    mean and max of v over the contributing cells
    bitmask c    share of contributing cells with bit k set
    categorical  share of updates with each code of the aggregate vocabulary, and "other"
    status       share of updates with each of the four non-reference statuses

Window states (AS-507, AS-508). For the updates of a window:

    flows          distinct flow keys (the PCAP adapter's `flow` column; a source without it has one flow
                   per update, which is what a flow-record CSV row is)
    packets        sum over updates of the increment of the flow's packet total since the flow's previous
                   update (flow-state updates carry running totals; a flow record is its own increment);
                   the total is flow.packets_total, else packets_fwd + packets_bwd; undefined when any
                   update of the window lacks it
    bytes_ip       the same with flow.bytes_fwd + flow.bytes_bwd (IP-layer bytes)
    bytes_payload  the same with flow.payload_bytes_fwd + flow.payload_bytes_bwd (transport payload);
                   the two byte definitions stay separate, as the data model keeps them (AS-303)
    dst_hosts      distinct responder entities
    dst_ports      distinct contributing flow.dst_port codes (undefined for a source that never supplies one)
    syn_share      share of updates with a contributing flow.tcp_flags that has SYN (0x02) set
    rst_share      the same with RST (0x04)
    failed_share   share of updates with a contributing flow.unanswered equal to 1 (the responder never
                   answered: the connection failed)

These are the observable features the next-state forecast is scored on (evaluation chapter: counts of
flows, packets and bytes, distinct destination hosts and ports, shares of SYN, RST and failed
connections).

Every design column has a `FeatureColumn` provenance entry: its data-model field, the canonical column
or window quantity it comes from, the status condition under which it is defined, the operation and
the lag. Column names and their order are deterministic functions of the configuration and of the
fitted vocabularies.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from nagahana.core.errors import InvariantViolation
from nagahana.data.windows import CANONICAL_COLUMNS, CANONICAL_INDEX, COLUMN_SLOTS
from nagahana.datamodel.columnar import STATUS_CODE, STATUS_ORDER
from nagahana.datamodel.fields import Kind
from nagahana.datamodel.status import CONTRIBUTING, ObservationStatus

from .config import FeatureConfig

#: Kind of every canonical column, in COLUMN_SLOTS order.
COLUMN_KINDS: tuple[Kind, ...] = tuple(c.kind for c in CANONICAL_COLUMNS)
NUMERIC_KINDS: frozenset[Kind] = frozenset({Kind.CONTINUOUS, Kind.COUNT, Kind.HISTOGRAM})
CONTRIB_CODES: np.ndarray = np.array(sorted(STATUS_CODE[s] for s in CONTRIBUTING), dtype=np.int64)
#: The statuses that get an indicator column; OBSERVED is the reference level.
INDICATOR_STATUSES: tuple[ObservationStatus, ...] = tuple(s for s in STATUS_ORDER if s is not ObservationStatus.OBSERVED)
INDICATOR_CODES: tuple[int, ...] = tuple(STATUS_CODE[s] for s in INDICATOR_STATUSES)

STATE_FEATURES: tuple[str, ...] = ("flows", "packets", "bytes_ip", "bytes_payload", "dst_hosts", "dst_ports",
                                   "syn_share", "rst_share", "failed_share")
STATE_COUNT_FEATURES: frozenset[str] = frozenset(STATE_FEATURES[:6])
TCP_SYN, TCP_RST = 0x02, 0x04
_MAX_CODE = float(2 ** 53)
_C = CANONICAL_INDEX


def to_codes(x: np.ndarray) -> np.ndarray:
    """Round float codes (443.0) to int64; NaN becomes 0 (callers mask non-contributing cells)."""
    return np.round(np.clip(np.nan_to_num(np.asarray(x, dtype=np.float64)), -_MAX_CODE, _MAX_CODE)).astype(np.int64)


def contributing(status: np.ndarray) -> np.ndarray:
    """True where a status code contributes evidence (OBSERVED, STALE, LOW_RELIABILITY)."""
    return np.isin(status, CONTRIB_CODES)


def signed_log1p_clip(x: np.ndarray, clip: float) -> np.ndarray:
    """clip(sign(x) log(1 + |x|), +-clip) in float64 (the FieldEncoder's numeric transform, AS-31, AS-100)."""
    x = np.asarray(x, dtype=np.float64)
    return np.clip(np.sign(x) * np.log1p(np.abs(x)), -clip, clip)


@dataclass(frozen=True)
class FeatureColumn:
    """Provenance of one design column (module docstring)."""

    name: str
    source: str        # canonical column name, "state.<name>" or "window"
    field: str         # data-model field id, "" for window quantities
    kind: str          # value, bit, category, status, mean, max, share, state, indicator, window
    status: str        # condition under which the column is defined
    op: str            # transformation
    lag: int           # cadence windows back from the unit's window; -1 for a per-update column
    binary: bool       # a 0/1 column (standardised by z-score)

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "source": self.source, "field": self.field, "kind": self.kind,
                "status": self.status, "op": self.op, "lag": self.lag, "binary": self.binary}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> FeatureColumn:
        return cls(str(d["name"]), str(d["source"]), str(d["field"]), str(d["kind"]), str(d["status"]), str(d["op"]),
                   int(d["lag"]), bool(d["binary"]))


def _field_of(j: int) -> str:
    return CANONICAL_COLUMNS[j].field_id


@dataclass
class UpdateEncoder:
    """Fitted encoder of the canonical columns (module docstring).

    vocab: categorical canonical column name -> vocabulary codes, most frequent first.
    """

    cfg: FeatureConfig
    vocab: dict[str, tuple[int, ...]] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)     # training rows seen per categorical column

    @classmethod
    def fit(cls, batches: Iterable[tuple[np.ndarray, np.ndarray]], cfg: FeatureConfig) -> UpdateEncoder:
        """Count the codes of every categorical column over the contributing cells of training updates."""
        tallies: dict[int, dict[int, int]] = {j: {} for j, k in enumerate(COLUMN_KINDS) if k is Kind.CATEGORICAL}
        rows = 0
        for values, status in batches:
            _check_matrices(values, status)
            rows += values.shape[0]
            con = contributing(status)
            for j, tally in tallies.items():
                codes = to_codes(values[con[:, j], j])
                if codes.size:
                    u, n = np.unique(codes, return_counts=True)
                    for code, cnt in zip(u.tolist(), n.tolist(), strict=True):
                        tally[code] = tally.get(code, 0) + cnt
        if rows == 0:
            raise InvariantViolation("the update encoder needs training updates")
        vocab: dict[str, tuple[int, ...]] = {}
        seen: dict[str, int] = {}
        for j, tally in tallies.items():
            items = sorted((code for code, cnt in tally.items() if cnt >= cfg.min_category_count),
                           key=lambda c: (-tally[c], c))
            name = COLUMN_SLOTS[j]
            vocab[name] = tuple(items[:cfg.max_categories])
            seen[name] = int(sum(tally.values()))
        return cls(cfg=cfg, vocab=vocab, counts=seen)

    def agg_vocab(self, name: str) -> tuple[int, ...]:
        """The aggregate vocabulary of a categorical column: a prefix of its per-update vocabulary."""
        return self.vocab.get(name, ())[: self.cfg.max_categories_aggregate]

    def update_columns(self) -> list[FeatureColumn]:
        """Per-update design columns, in the order `encode` writes them."""
        cols: list[FeatureColumn] = []
        clip = self.cfg.numeric_clip
        for j, name in enumerate(COLUMN_SLOTS):
            kind, fid = COLUMN_KINDS[j], _field_of(j)
            if kind in NUMERIC_KINDS:
                cols.append(FeatureColumn(f"u|{name}|value", name, fid, "value", "contributing",
                                          f"signed_log1p clip {clip:g}", -1, False))
            elif kind is Kind.BITMASK:
                cols += [FeatureColumn(f"u|{name}|bit{k}", name, fid, "bit", "contributing", f"bit {k}", -1, True)
                         for k in range(self.cfg.max_bits)]
            elif kind is Kind.CATEGORICAL:
                cols += [FeatureColumn(f"u|{name}|code={c}", name, fid, "category", "contributing", f"code == {c}", -1, True)
                         for c in self.vocab.get(name, ())]
                cols.append(FeatureColumn(f"u|{name}|code=other", name, fid, "category", "contributing",
                                          "code not in vocabulary", -1, True))
            else:
                raise InvariantViolation(f"canonical column {name} has an unsupported kind {kind}")
            cols += [FeatureColumn(f"u|{name}|status={s.value}", name, fid, "status", s.value, "indicator", -1, True)
                     for s in INDICATOR_STATUSES]
        return cols

    def encode(self, values: np.ndarray, status: np.ndarray) -> np.ndarray:
        """Raw per-update design, float32 [n, D_u] with NaN where a value does not exist (module docstring)."""
        _check_matrices(values, status)
        n = values.shape[0]
        con = contributing(status)                                              # [n, C]
        if bool((con & ~np.isfinite(values)).any()):
            raise InvariantViolation("a contributing cell holds a non-finite value (D-41 pairing broken)")
        parts: list[np.ndarray] = []
        clip = self.cfg.numeric_clip
        for j, name in enumerate(COLUMN_SLOTS):
            kind = COLUMN_KINDS[j]
            cj = con[:, j]
            if kind in NUMERIC_KINDS:
                v = signed_log1p_clip(np.where(cj, values[:, j], 0.0), clip)
                parts.append(np.where(cj, v, np.nan)[:, None])
            elif kind is Kind.BITMASK:
                code = np.maximum(to_codes(np.where(cj, values[:, j], 0.0)), 0)
                bits = (code[:, None] >> np.arange(self.cfg.max_bits, dtype=np.int64)[None, :]) & 1
                parts.append(np.where(cj[:, None], bits.astype(np.float64), np.nan))
            else:
                code = to_codes(np.where(cj, values[:, j], 0.0))
                voc = np.asarray(self.vocab.get(name, ()), dtype=np.int64)
                hot = (code[:, None] == voc[None, :]).astype(np.float64)       # [n, |vocab|]
                other = (~np.isin(code, voc)).astype(np.float64)[:, None]
                parts.append(np.where(cj[:, None], np.concatenate([hot, other], axis=1), np.nan))
            st = status[:, j]
            parts.append(np.stack([(st == code_s).astype(np.float64) for code_s in INDICATOR_CODES], axis=1))
        out = np.concatenate(parts, axis=1).astype(np.float32) if parts else np.zeros((n, 0), np.float32)
        return out

    def window_block_columns(self) -> list[FeatureColumn]:
        """The columns of one window block, without the lag prefix (lag = 0 placeholder)."""
        cols = [
            FeatureColumn("window|unavailable", "window", "", "window", "any", "1 if the window lies before the segment", 0, True),
            FeatureColumn("window|empty", "window", "", "window", "any", "1 if inside the segment and without updates", 0, True),
            FeatureColumn("window|coverage", "window", "", "window", "any", "share of the window inside the segment", 0, False),
            FeatureColumn("window|updates", "window", "", "window", "available", "log(1 + update count)", 0, False),
        ]
        for s in STATE_FEATURES:
            op = ("log(1 + count)" if self.cfg.log_count_states else "count") if s in STATE_COUNT_FEATURES else "share"
            cols.append(FeatureColumn(f"state|{s}", f"state.{s}", "", "state", "defined", op, 0, False))
        cols += [FeatureColumn(f"state|{s}|undefined", f"state.{s}", "", "indicator", "any", "1 if the state value is undefined", 0, True)
                 for s in STATE_FEATURES]
        for j, name in enumerate(COLUMN_SLOTS):
            kind, fid = COLUMN_KINDS[j], _field_of(j)
            if kind in NUMERIC_KINDS:
                cols.append(FeatureColumn(f"{name}|mean", name, fid, "mean", "contributing", "mean of v over contributing cells", 0, False))
                cols.append(FeatureColumn(f"{name}|max", name, fid, "max", "contributing", "max of v over contributing cells", 0, False))
            elif kind is Kind.BITMASK:
                cols += [FeatureColumn(f"{name}|bit{k}_share", name, fid, "share", "contributing", f"share with bit {k}", 0, False)
                         for k in range(self.cfg.max_bits)]
            else:
                cols += [FeatureColumn(f"{name}|code={c}_share", name, fid, "share", "contributing", f"share of updates with code {c}", 0, False)
                         for c in self.agg_vocab(name)]
                cols.append(FeatureColumn(f"{name}|code=other_share", name, fid, "share", "contributing",
                                          "share of updates with a code outside the vocabulary", 0, False))
            cols += [FeatureColumn(f"{name}|status={s.value}_share", name, fid, "share", s.value, "share of updates", 0, False)
                     for s in INDICATOR_STATUSES]
        return cols

    def window_columns(self, lag: int) -> list[FeatureColumn]:
        """The columns of the window block at `lag`, with the "w<lag>|" prefix."""
        return [FeatureColumn(f"w{lag}|{c.name}", c.source, c.field, c.kind, c.status, c.op, lag, c.binary)
                for c in self.window_block_columns()]

    @property
    def block_width(self) -> int:
        return len(self.window_block_columns())

    def state_dict(self) -> dict[str, Any]:
        return {"vocab": {k: list(v) for k, v in self.vocab.items()}, "counts": dict(self.counts)}

    @classmethod
    def from_state_dict(cls, cfg: FeatureConfig, d: dict[str, Any]) -> UpdateEncoder:
        return cls(cfg=cfg, vocab={str(k): tuple(int(c) for c in v) for k, v in d["vocab"].items()},
                   counts={str(k): int(v) for k, v in d.get("counts", {}).items()})


def _check_matrices(values: np.ndarray, status: np.ndarray) -> None:
    c = len(COLUMN_SLOTS)
    if values.ndim != 2 or values.shape[1] != c or status.shape != values.shape:
        raise InvariantViolation(f"values and status must be [n, {c}] in the canonical layout; got {values.shape}, {status.shape}")


def flow_totals(values: np.ndarray, status: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per update: packet total, IP-layer byte total, payload byte total (NaN where not defined; module docstring)."""
    con = contributing(status)

    def col(name: str) -> np.ndarray:
        j = _C[name]
        return np.where(con[:, j], values[:, j].astype(np.float64), np.nan)

    pt = col("flow.packets_total")
    pk = col("flow.packets_fwd") + col("flow.packets_bwd")
    packets = np.where(np.isfinite(pt), pt, pk)
    bytes_ip = col("flow.bytes_fwd") + col("flow.bytes_bwd")
    bytes_pl = col("flow.payload_bytes_fwd") + col("flow.payload_bytes_bwd")
    return packets, bytes_ip, bytes_pl


def flow_deltas(total: np.ndarray, flow: np.ndarray) -> np.ndarray:
    """Increment of each update's running total since the previous update of the same flow (module docstring).

    total float64 [n] in stream order (NaN undefined); flow int64 [n] (-1: the update is a flow of its own).
    A decrease of a running total (inconsistent counters) gives NaN.
    """
    total = np.asarray(total, dtype=np.float64)
    flow = np.asarray(flow, dtype=np.int64)
    out = total.copy()
    idx = np.flatnonzero(flow >= 0)
    if idx.size:
        order = idx[np.argsort(flow[idx], kind="mergesort")]                 # rows grouped by flow, in stream order
        same = np.r_[False, flow[order][1:] == flow[order][:-1]]
        cur, prv = order[same], order[np.flatnonzero(same) - 1]
        d = total[cur] - total[prv]
        out[cur] = np.where(d < 0, np.nan, d)
    return out


def _distinct_per_bin(bin_idx: np.ndarray, key: np.ndarray, n_bins: int) -> np.ndarray:
    """Number of distinct keys per bin."""
    if key.size == 0:
        return np.zeros(n_bins, dtype=np.int64)
    order = np.lexsort((key, bin_idx))
    b, k = bin_idx[order], key[order]
    new = np.r_[True, (b[1:] != b[:-1]) | (k[1:] != k[:-1])]
    return np.bincount(b[new], minlength=n_bins).astype(np.int64)


@dataclass
class SourceArrays:
    """The per-update arrays of one source that window states and aggregates need (stream order)."""

    values: np.ndarray          # float32 [n, C]
    status: np.ndarray          # uint8 [n, C]
    flow: np.ndarray            # int64 [n]
    responder: np.ndarray       # int64 [n]
    d_packets: np.ndarray       # float64 [n]
    d_bytes_ip: np.ndarray      # float64 [n]
    d_bytes_payload: np.ndarray  # float64 [n]

    def supplies(self) -> dict[str, bool]:
        """Whether the source supplies each count state at all (decides 0 versus undefined for an empty window)."""
        con = contributing(self.status[:, _C["flow.dst_port"]]) if self.status.shape[0] else np.zeros(0, bool)
        return {"flows": True, "dst_hosts": True, "packets": bool(np.isfinite(self.d_packets).any()),
                "bytes_ip": bool(np.isfinite(self.d_bytes_ip).any()),
                "bytes_payload": bool(np.isfinite(self.d_bytes_payload).any()), "dst_ports": bool(con.any())}


def window_states(arr: SourceArrays, bin_idx: np.ndarray, n_bins: int, supplies: dict[str, bool]) -> np.ndarray:
    """Raw window states [n_bins, 9] (counts not log-transformed; NaN where undefined) of non-empty bins."""
    out = np.full((n_bins, len(STATE_FEATURES)), np.nan, dtype=np.float64)
    if n_bins == 0:
        return out
    n_b = np.bincount(bin_idx, minlength=n_bins).astype(np.float64)
    # flows: distinct flow keys, plus one per update without a flow key
    has = arr.flow >= 0
    out[:, 0] = _distinct_per_bin(bin_idx[has], arr.flow[has], n_bins) + np.bincount(bin_idx[~has], minlength=n_bins)
    for col, d in ((1, arr.d_packets), (2, arr.d_bytes_ip), (3, arr.d_bytes_payload)):
        s = np.bincount(bin_idx, weights=np.nan_to_num(d), minlength=n_bins)
        bad = np.bincount(bin_idx, weights=np.isnan(d).astype(np.float64), minlength=n_bins) > 0
        out[:, col] = np.where(bad, np.nan, s)
    resp = arr.responder >= 0
    out[:, 4] = _distinct_per_bin(bin_idx[resp], arr.responder[resp], n_bins)
    jp = _C["flow.dst_port"]
    cp = contributing(arr.status[:, jp])
    ports = _distinct_per_bin(bin_idx[cp], to_codes(arr.values[cp, jp]), n_bins).astype(np.float64)
    out[:, 5] = ports if supplies["dst_ports"] else np.nan
    jf = _C["flow.tcp_flags"]
    cf = contributing(arr.status[:, jf])
    fl = np.maximum(to_codes(np.where(cf, arr.values[:, jf], 0.0)), 0)
    den = np.bincount(bin_idx, weights=cf.astype(np.float64), minlength=n_bins)
    for col, bit in ((6, TCP_SYN), (7, TCP_RST)):
        num = np.bincount(bin_idx, weights=(cf & ((fl & bit) != 0)).astype(np.float64), minlength=n_bins)
        out[:, col] = np.divide(num, den, out=np.full(n_bins, np.nan), where=den > 0)
    ju = _C["flow.unanswered"]
    cu = contributing(arr.status[:, ju])
    un = to_codes(np.where(cu, arr.values[:, ju], 0.0))
    den_u = np.bincount(bin_idx, weights=cu.astype(np.float64), minlength=n_bins)
    num_u = np.bincount(bin_idx, weights=(cu & (un == 1)).astype(np.float64), minlength=n_bins)
    out[:, 8] = np.divide(num_u, den_u, out=np.full(n_bins, np.nan), where=den_u > 0)
    out[n_b == 0] = np.nan
    return out


def empty_states(supplies: dict[str, bool]) -> np.ndarray:
    """States [9] of a window inside the observed span without updates: zero counts, undefined shares."""
    out = np.full(len(STATE_FEATURES), np.nan)
    for i, s in enumerate(STATE_FEATURES):
        if s in STATE_COUNT_FEATURES and supplies.get(s, False):
            out[i] = 0.0
    return out


def transform_states(states: np.ndarray, cfg: FeatureConfig) -> np.ndarray:
    """log(1 + count) on the count states when `log_count_states` (AS-509); shares unchanged."""
    out = np.array(states, dtype=np.float64, copy=True)
    if cfg.log_count_states:
        for i, s in enumerate(STATE_FEATURES):
            if s in STATE_COUNT_FEATURES:
                out[..., i] = np.log1p(out[..., i])
    return out


def aggregate_blocks(enc: UpdateEncoder, arr: SourceArrays, bin_idx: np.ndarray, n_bins: int, coverage: np.ndarray,
                     supplies: dict[str, bool]) -> np.ndarray:
    """Window blocks float32 [n_bins, A] of non-empty bins, in `window_block_columns` order (module docstring).

    bin_idx int64 [n] maps every update to its bin; bins with no update are not allowed here (empty and
    unavailable windows are built by `missing_blocks`).
    """
    cfg = enc.cfg
    n_b = np.bincount(bin_idx, minlength=n_bins).astype(np.float64)
    if np.any(n_b == 0):
        raise InvariantViolation("aggregate_blocks needs every bin to hold at least one update")
    parts: list[np.ndarray] = [
        np.zeros((n_bins, 1)), np.zeros((n_bins, 1)), np.asarray(coverage, dtype=np.float64)[:, None],
        np.log1p(n_b)[:, None],
    ]
    st = transform_states(window_states(arr, bin_idx, n_bins, supplies), cfg)
    parts += [st, np.isnan(st).astype(np.float64)]
    con = contributing(arr.status)
    starts = np.r_[0, np.flatnonzero(np.diff(bin_idx) != 0) + 1]
    if np.any(np.diff(bin_idx) < 0):
        raise InvariantViolation("updates must be grouped by bin (stream order)")
    for j, name in enumerate(COLUMN_SLOTS):
        kind = COLUMN_KINDS[j]
        cj = con[:, j]
        cnt = np.bincount(bin_idx, weights=cj.astype(np.float64), minlength=n_bins)
        if kind in NUMERIC_KINDS:
            v = signed_log1p_clip(np.where(cj, arr.values[:, j], 0.0), cfg.numeric_clip)
            s = np.bincount(bin_idx, weights=np.where(cj, v, 0.0), minlength=n_bins)
            mean = np.divide(s, cnt, out=np.full(n_bins, np.nan), where=cnt > 0)
            mx = np.maximum.reduceat(np.where(cj, v, -np.inf), starts)
            parts += [mean[:, None], np.where(cnt > 0, mx, np.nan)[:, None]]
        elif kind is Kind.BITMASK:
            code = np.maximum(to_codes(np.where(cj, arr.values[:, j], 0.0)), 0)
            shares = []
            for k in range(cfg.max_bits):
                num = np.bincount(bin_idx, weights=(cj & (((code >> k) & 1) == 1)).astype(np.float64), minlength=n_bins)
                shares.append(np.divide(num, cnt, out=np.full(n_bins, np.nan), where=cnt > 0))
            parts.append(np.stack(shares, axis=1))
        else:
            code = to_codes(np.where(cj, arr.values[:, j], 0.0))
            voc = np.asarray(enc.agg_vocab(name), dtype=np.int64)
            shares = [np.bincount(bin_idx, weights=(cj & (code == c)).astype(np.float64), minlength=n_bins) / n_b
                      for c in voc.tolist()]
            shares.append(np.bincount(bin_idx, weights=(cj & ~np.isin(code, voc)).astype(np.float64), minlength=n_bins) / n_b)
            parts.append(np.stack(shares, axis=1))
        stj = arr.status[:, j]
        parts.append(np.stack([np.bincount(bin_idx, weights=(stj == c).astype(np.float64), minlength=n_bins) / n_b
                               for c in INDICATOR_CODES], axis=1))
    out = np.concatenate(parts, axis=1).astype(np.float32)
    if out.shape[1] != enc.block_width:
        raise InvariantViolation("window block width does not match its column list")
    return out


def missing_blocks(enc: UpdateEncoder, coverage: np.ndarray, supplies: dict[str, bool]) -> np.ndarray:
    """Blocks float32 [q, A] of windows without updates: empty (coverage > 0) or unavailable (coverage = 0)."""
    cov = np.asarray(coverage, dtype=np.float64)
    q = cov.size
    out = np.full((q, enc.block_width), np.nan, dtype=np.float64)
    avail = cov > 0
    out[:, 0] = (~avail).astype(np.float64)
    out[:, 1] = avail.astype(np.float64)
    out[:, 2] = cov
    out[:, 3] = np.where(avail, 0.0, np.nan)
    ns = len(STATE_FEATURES)
    st = transform_states(empty_states(supplies), enc.cfg)
    out[:, 4:4 + ns] = np.where(avail[:, None], st[None, :], np.nan)
    out[:, 4 + ns:4 + 2 * ns] = np.isnan(out[:, 4:4 + ns]).astype(np.float64)
    return out.astype(np.float32)


def bin_of(t: np.ndarray, w: float) -> np.ndarray:
    """Cadence bin g of each time: the window ((g - 1) w, g w] that contains it (exact at grid points)."""
    t = np.asarray(t, dtype=np.float64)
    g = np.ceil(t / w).astype(np.int64)
    g = np.where((g - 1) * w >= t, g - 1, g)
    g = np.where(g * w < t, g + 1, g)
    return g


def coverage_of(g: np.ndarray, w: float, start: np.ndarray) -> np.ndarray:
    """Share of the window ((g - 1) w, g w] at or after `start` (the segment's first update time), in [0, 1]."""
    hi = np.asarray(g, dtype=np.float64) * w
    lo = np.maximum(hi - w, np.asarray(start, dtype=np.float64))
    return np.clip((hi - lo) / w, 0.0, 1.0)


__all__ = [
    "COLUMN_KINDS", "CONTRIB_CODES", "FeatureColumn", "INDICATOR_STATUSES", "STATE_FEATURES", "SourceArrays",
    "UpdateEncoder", "aggregate_blocks", "bin_of", "contributing", "coverage_of", "empty_states", "flow_deltas",
    "flow_totals", "missing_blocks", "signed_log1p_clip", "to_codes", "transform_states", "window_states",
]
