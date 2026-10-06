"""Design matrices, units and metadata of the LR family, from an `LRCorpus` and a fitted `UpdateEncoder`.

Context windows. For every source, the updates of each (record, cadence window) pair are aggregated once
into a window block (features.aggregate_blocks). A unit then reads the blocks of its own record at the
lags it needs; a window of the record without updates is an empty block and a window before the record
is an unavailable block (features.missing_blocks), so context never crosses into another record, and
therefore never into another split (AS-506).

    trigger at tau            blocks of windows g, g - 1, ..., g - L with g = tau / w (window g ends at tau,
                              so it holds the updates at or before the trigger)
    update at t (context)     blocks of windows g_u - 1, ..., g_u - L with g_u the window holding t: complete
                              windows that end before the update's own window starts

No block of a unit ever contains an update later than the unit (tests/test_lr_features.py).

Designs. The trigger design is small (one row per cadence trigger) and is held in memory for the largest
lag count any model reads; a model with fewer lags reads its leading columns, which are ordered by lag.
The update design is wide and has one row per state update, so it is produced chunk by chunk from the
stored value/status matrices (`UpdateDesign.chunk`), which is what lets the streamed solvers fit corpora
that do not fit in memory.

Row sources. `rows_source` combines a design, a set of units, their labels and weights, and a fitted
standardiser (with its kept columns) into the logistic.RowSource every solver reads.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from nagahana.core.errors import InvariantViolation
from nagahana.evaluation.predictions import make_meta

from .corpus import UNIT_ROLES, LRCorpus, SourceTables
from .features import FeatureColumn, UpdateEncoder, aggregate_blocks, bin_of, coverage_of, missing_blocks
from .logistic import Chunk, RowSource
from .standardise import Standardiser


@dataclass
class ContextTable:
    """Window blocks of the (record, window) pairs of one source that hold updates."""

    seg: np.ndarray        # int64 [B]
    g: np.ndarray          # int64 [B]
    block: np.ndarray      # float32 [B, A]
    t_first: np.ndarray    # float64 [R] first update time of every record of the source
    supplies: dict[str, bool]
    window_seconds: float

    def lookup(self, enc: UpdateEncoder, seg: np.ndarray, g: np.ndarray) -> np.ndarray:
        """Blocks [q, A] of the windows g of records seg (present, empty or unavailable)."""
        seg = np.asarray(seg, dtype=np.int64)
        g = np.asarray(g, dtype=np.int64)
        q = seg.size
        out = np.empty((q, enc.block_width), dtype=np.float32)
        if q == 0:
            return out
        if self.g.size:
            g0 = int(min(self.g.min(), g.min()))
            span = int(max(self.g.max(), g.max()) - g0 + 1)
            table_key = self.seg * span + (self.g - g0)
            query_key = seg * span + (g - g0)
            pos = np.searchsorted(table_key, query_key)
            hit = (pos < table_key.size) & (table_key[np.minimum(pos, table_key.size - 1)] == query_key)
        else:
            pos = np.zeros(q, dtype=np.int64)
            hit = np.zeros(q, dtype=bool)
        if hit.any():
            out[hit] = self.block[pos[hit]]
        miss = ~hit
        if miss.any():
            cov = coverage_of(g[miss], self.window_seconds, self.t_first[seg[miss]])
            out[miss] = missing_blocks(enc, cov, self.supplies)
        return out


def context_table(s: SourceTables, enc: UpdateEncoder) -> ContextTable:
    """Aggregate every (record, window) pair of a source that holds updates (module docstring)."""
    w = s.window_seconds
    t = np.asarray(s["time"], dtype=np.float64)
    seg = np.asarray(s["segment"], dtype=np.int64)
    t_first = np.asarray([g.t_first for g in s.segments], dtype=np.float64)
    supplies = s.supplies()
    if t.size == 0:
        return ContextTable(np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros((0, enc.block_width), np.float32),
                            t_first, supplies, w)
    g = bin_of(t, w)
    change = np.r_[True, (seg[1:] != seg[:-1]) | (g[1:] != g[:-1])]
    bidx = np.cumsum(change) - 1
    first = np.flatnonzero(change)
    b_seg, b_g = seg[first], g[first]
    key_ok = np.all((b_seg[1:] > b_seg[:-1]) | ((b_seg[1:] == b_seg[:-1]) & (b_g[1:] > b_g[:-1]))) if first.size > 1 else True
    if not key_ok:
        raise InvariantViolation(f"{s.source_id}: (record, window) keys must increase along the stream")
    cov = coverage_of(b_g, w, t_first[b_seg])
    block = aggregate_blocks(enc, s.source_arrays(), bidx.astype(np.int64), first.size, cov, supplies)
    return ContextTable(b_seg, b_g, block, t_first, supplies, w)


@dataclass
class Contexts:
    """Context tables of every source of a corpus for one encoder (built on first use)."""

    corpus: LRCorpus
    encoder: UpdateEncoder
    tables: dict[int, ContextTable] = field(default_factory=dict)

    def __getitem__(self, i: int) -> ContextTable:
        if i not in self.tables:
            self.tables[i] = context_table(self.corpus.sources[i], self.encoder)
        return self.tables[i]


@dataclass
class TriggerDesign:
    """Raw window blocks of every trigger of the corpus at lags 0 ... lags (module docstring)."""

    x: np.ndarray              # float32 [M, A * (lags + 1)]
    columns: list[FeatureColumn]
    source: np.ndarray         # int64 [M]
    index: np.ndarray          # int64 [M] trigger index inside its source
    lags: int
    block_width: int

    def columns_for(self, lags: int) -> np.ndarray:
        """Column indices of lags 0 ... lags (leading columns)."""
        if lags > self.lags:
            raise InvariantViolation(f"the trigger design holds {self.lags} lags, {lags} requested")
        return np.arange((lags + 1) * self.block_width)


def design_rows(design: TriggerDesign, units: Units) -> np.ndarray:
    """Rows of the trigger design for trigger units (source, trigger index)."""
    key = design.source * (1 << 32) + design.index
    want = units.source * (1 << 32) + units.row
    if want.size == 0:
        return np.zeros(0, dtype=np.int64)
    order = np.argsort(key, kind="mergesort")
    pos = np.searchsorted(key[order], want)
    if np.any(pos >= key.size) or np.any(key[order][np.minimum(pos, key.size - 1)] != want):
        raise InvariantViolation("a trigger unit is missing from the trigger design")
    return order[pos]


def trigger_design(corpus: LRCorpus, contexts: Contexts, lags: int) -> TriggerDesign:
    enc = contexts.encoder
    xs, src, idx = [], [], []
    w = corpus.window_seconds
    for i, s in enumerate(corpus.sources):
        m = s.n_triggers
        if m == 0:
            continue
        tg = bin_of(np.asarray(s["t_time"], dtype=np.float64), w)
        seg = np.asarray(s["t_segment"], dtype=np.int64)
        ctx = contexts[i]
        xs.append(np.concatenate([ctx.lookup(enc, seg, tg - lag) for lag in range(lags + 1)], axis=1))
        src.append(np.full(m, i, dtype=np.int64))
        idx.append(np.arange(m, dtype=np.int64))
    cols = [c for lag in range(lags + 1) for c in enc.window_columns(lag)]
    if not xs:
        return TriggerDesign(np.zeros((0, len(cols)), np.float32), cols, np.zeros(0, np.int64), np.zeros(0, np.int64),
                             lags, enc.block_width)
    return TriggerDesign(np.concatenate(xs), cols, np.concatenate(src), np.concatenate(idx), lags, enc.block_width)


@dataclass
class UpdateDesign:
    """Per-update design, produced in chunks (module docstring)."""

    corpus: LRCorpus
    contexts: Contexts
    context_lags: int

    @property
    def encoder(self) -> UpdateEncoder:
        return self.contexts.encoder

    def columns(self) -> list[FeatureColumn]:
        cols = self.encoder.update_columns()
        for lag in range(1, self.context_lags + 1):
            cols += self.encoder.window_columns(lag)
        return cols

    def chunk(self, source: int, rows: np.ndarray) -> np.ndarray:
        """Raw design float32 [len(rows), D] of the given update rows of one source."""
        s = self.corpus.sources[source]
        rows = np.asarray(rows, dtype=np.int64)
        values = np.asarray(s["values"][rows])
        status = np.asarray(s["status"][rows])
        x = self.encoder.encode(values, status)
        if self.context_lags == 0:
            return x
        g = bin_of(np.asarray(s["time"][rows], dtype=np.float64), s.window_seconds)
        seg = np.asarray(s["segment"][rows], dtype=np.int64)
        ctx = self.contexts[source]
        parts = [x] + [ctx.lookup(self.encoder, seg, g - lag) for lag in range(1, self.context_lags + 1)]
        return np.concatenate(parts, axis=1)


@dataclass
class Units:
    """A set of units: (source, row) pairs in corpus order, with their roles."""

    source: np.ndarray     # int64 [N]
    row: np.ndarray        # int64 [N] (update row, or trigger index)
    role: np.ndarray       # str [N]

    def __len__(self) -> int:
        return int(self.source.size)

    def select(self, mask: np.ndarray) -> Units:
        return Units(self.source[mask], self.row[mask], self.role[mask])

    def take(self, idx: np.ndarray) -> Units:
        return Units(self.source[idx], self.row[idx], self.role[idx])


def update_units(corpus: LRCorpus, roles: Sequence[str] | None = None) -> Units:
    """Every update in a record of the given roles (default: train, val, test, zero_shot)."""
    want = set(roles) if roles is not None else set(UNIT_ROLES)
    src, row, role = [], [], []
    for i, s in enumerate(corpus.sources):
        r = s.update_roles()
        keep = np.flatnonzero(np.isin(r, list(want)))
        src.append(np.full(keep.size, i, dtype=np.int64))
        row.append(keep.astype(np.int64))
        role.append(r[keep])
    return Units(np.concatenate(src), np.concatenate(row), np.concatenate(role).astype(str))


def trigger_units(corpus: LRCorpus, roles: Sequence[str] | None = None, *, usable_only: bool = False) -> Units:
    """Every trigger of a timed source in a record of the given roles (timeless sources give none, AS-520)."""
    want = set(roles) if roles is not None else set(UNIT_ROLES)
    src, row, role = [], [], []
    for i, s in enumerate(corpus.sources):
        if s.timeless or s.n_triggers == 0:
            continue
        r = s.trigger_roles()
        keep = np.isin(r, list(want))
        if usable_only:
            keep &= np.asarray(s["t_usable"], dtype=bool)
        k = np.flatnonzero(keep)
        src.append(np.full(k.size, i, dtype=np.int64))
        row.append(k.astype(np.int64))
        role.append(r[k])
    if not src:
        return Units(np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0, dtype="<U9"))
    return Units(np.concatenate(src), np.concatenate(row), np.concatenate(role).astype(str))


def gather(corpus: LRCorpus, units: Units, key: str) -> np.ndarray:
    """One table column for every unit (update or trigger table, by key)."""
    parts = []
    order = []
    for i in np.unique(units.source):
        sel = np.flatnonzero(units.source == i)
        parts.append(np.asarray(corpus.sources[int(i)][key])[units.row[sel]])
        order.append(sel)
    if not parts:
        return np.zeros(0)
    out = np.concatenate(parts)
    inv = np.empty(out.shape[0], dtype=np.int64)
    inv[np.concatenate(order)] = np.arange(out.shape[0])
    return out[inv]


def unit_meta(corpus: LRCorpus, units: Units, *, triggers: bool, extra: dict[str, np.ndarray] | None = None) -> pd.DataFrame:
    """META_COLUMNS for a set of units (evaluation/predictions.py), plus source and record ids."""
    n = len(units)
    time = gather(corpus, units, "t_time" if triggers else "time").astype(np.float64)
    fam = gather(corpus, units, "t_family" if triggers else "family").astype(str)
    seg = gather(corpus, units, "t_segment" if triggers else "segment").astype(np.int64)
    dataset = np.asarray([corpus.sources[int(i)].dataset for i in units.source], dtype=str)
    network = np.asarray([corpus.sources[int(i)].network for i in units.source], dtype=str)
    source_id = np.asarray([corpus.sources[int(i)].source_id for i in units.source], dtype=str)
    record = np.asarray([corpus.sources[int(i)].segments[int(g)].id for i, g in zip(units.source, seg, strict=True)], dtype=str)
    novelty = np.asarray([corpus.sources[int(i)].segments[int(g)].novelty for i, g in zip(units.source, seg, strict=True)],
                         dtype=str)
    cols: dict[str, np.ndarray] = {"source": source_id, "record": record}
    if extra:
        cols.update(extra)
    if n == 0:
        return make_meta(0, time=np.zeros(0), dataset=np.zeros(0, str), network=np.zeros(0, str),
                         family=np.zeros(0, str), novelty=np.zeros(0, str), split=np.zeros(0, str),
                         entity=np.zeros(0, np.int64), **{k: np.asarray(v)[:0] for k, v in cols.items()})
    return make_meta(n, time=time, dataset=dataset, network=network, family=fam, novelty=novelty,
                     split=units.role.astype(str), entity=np.full(n, -1, dtype=np.int64), **cols)


def update_chunks(design: UpdateDesign, units: Units, chunk_rows: int) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """(positions into `units`, raw design chunk) over the units, source by source."""
    for i in np.unique(units.source):
        sel = np.flatnonzero(units.source == i)
        for a in range(0, sel.size, chunk_rows):
            pos = sel[a:a + chunk_rows]
            yield pos, design.chunk(int(i), units.row[pos])


#: chunk_factory(chunk_rows) -> fresh iterator of (positions into the unit arrays, raw float32 design chunk).
RawChunkFactory = Callable[[int], Iterator[tuple[np.ndarray, np.ndarray]]]


def rows_source(chunk_factory: RawChunkFactory, *, n_rows: int, std: Standardiser, keep: np.ndarray, y: np.ndarray,
                weight: np.ndarray) -> RowSource:
    """Standardised (X, y, weight) chunks for a solver; y and weight are indexed by the chunk positions."""
    y = np.asarray(y, dtype=np.float64)
    weight = np.asarray(weight, dtype=np.float64)
    keep = np.asarray(keep, dtype=np.int64)

    def factory(chunk_rows: int) -> Iterator[Chunk]:
        for pos, raw in chunk_factory(chunk_rows):
            yield std.transform(raw, keep), y[pos], weight[pos]

    return RowSource(n_rows=n_rows, n_cols=int(keep.size), factory=factory)


__all__ = ["ContextTable", "Contexts", "RawChunkFactory", "TriggerDesign", "UpdateDesign", "Units", "context_table",
           "design_rows", "gather", "rows_source", "trigger_design", "trigger_units", "unit_meta", "update_chunks",
           "update_units"]
