"""Stream-ordered training data (D-51): consecutive windows, segments, carried entities, a lane loader.

Purpose
-------
D-51 (owner, 2026-10-02): training runs on **consecutive windows in time order**; each window's TSTCT
reads the previous windows' Environment K/V as read-only memory (Transformer-XL style, Dai et al.,
ACL 2019, arXiv:1901.02860), exactly as inference reads the Environment store. This module gives the
model side (engineer B, `TSTCT.forward(..., carry=CarriedEnvironment)`) what it needs to carry state
from window to window:

    plan_stream(src, cfg)          the consecutive windows of one source, each with its trigger range
    segment_records(...)           runs of consecutive windows ("segments") as `WindowRecord`s, so the
                                   split rules (`sampling.assign_splits`) and the class-balanced
                                   sampler work on segments unchanged
    StreamLoader                   B lanes, each walking its segments window by window, yielding
                                   (WindowBatch, LabelBatch, [StreamContext per lane])

Owner sources, decisions, assumptions: D-51 (training across time), D-49 (times relative to each
window's origin), AS-12 (cadence), AS-318 (epoch grid); new AS-333 (segment length), AS-334 (label
limit), AS-335 (carry bookkeeping) in `docs/assumptions/data.md`.

Definitions
-----------
Windows. Window k of a source covers sorted updates [start_k, stop_k) (`windows.plan_windows`, the
same plan as the window mode). Its trigger range is [t_first(k), t_first(k+1)) on the epoch grid of
cadence c (the last window of a source: [t_first, t_last]). Every grid point of the stream therefore
belongs to exactly one window, the one whose updates precede it; no trigger is lost in the gap
between two windows, and no trigger sees an update after it.

Entity keys. A window's local entity index i stands for the source's entity row `entity_keys[i]`
(an integer, stable across all windows of the source; text keys are `cu.entities.kind/key`).
`key_to_index` inverts it. `carried_keys` are the entity rows seen in earlier windows of the same
segment (sorted); `carried_index[i]` is the position of local entity i in `carried_keys`, −1 if the
entity is new in this window. That is the map TSTCT needs from carried memory slots to current
entity indices.

Times. Each window keeps its own origin (D-49 precision). `origin_shift` = origin(k) − origin(k−1)
(float64 seconds, 0 at a reset): a carried time t relative to the previous origin is t − origin_shift
relative to the current one.

Segments (AS-333). Consecutive windows are grouped until the segment spans at least
`segment_seconds` (default K·c = `horizon_k · window_seconds`). A segment is the unit of splitting
and of class-balanced sampling: carry flows only inside a segment (`reset` = True at its first
window), so no memory crosses from one split into another. With `purge` ≥ 1 segment between splits
(AS-327) and segments of at least K·c, the K-step label look-ahead of a trigger never reaches another
split's traffic; `label_limit` (AS-334) enforces it exactly in every case.

Lanes. Batch element b is lane b; a lane walks one segment at a time in window order, then takes the
next segment from the sampling order. Lanes that run out of segments drop out, so the last batches
can be smaller. `StreamContext.lane` says which carry a window continues.

Invariants (tests/test_data_stream.py): windows of a segment are consecutive and in time order;
every grid point of a source lies in exactly one window's trigger range; carried_index is consistent
with key_to_index; reset exactly at segment starts; no window of a non-train segment is served by a
training loader.

Extension points: a burn-in (run windows before a segment without loss to warm the carry, R2D2,
Kapturowski et al., ICLR 2019) is possible only from windows of the same split; `segment_seconds`.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from nagahana.core.errors import InvariantViolation
from nagahana.models.batch import LabelBatch, WindowBatch
from nagahana.models.config import NagaHanaConfig

from .collate import collate_items
from .sampling import WindowRecord, _family_of
from .windows import PreparedSource, StructureFn, WindowItem, build_window, plan_windows


@dataclass(frozen=True)
class StreamWindow:
    """One window of a source in stream order."""

    index: int                       # position in the source's stream
    start: int
    stop: int
    t_first: float                   # epoch time of the first update
    t_last: float
    trigger_lo: float                # epoch [trigger_lo, trigger_hi): this window's cadence grid points
    trigger_hi: float
    entity_keys: np.ndarray          # [V] source entity rows, in the window's local order


def _local_keys(src: PreparedSource, start: int, stop: int) -> np.ndarray:
    """Source entity rows of a window in first-appearance order (= `build_window`'s local order)."""
    flat = src.ents[start:stop].reshape(-1)
    return np.asarray(pd.unique(flat[flat >= 0]), dtype=np.int64)


def plan_stream(src: PreparedSource, cfg: NagaHanaConfig) -> list[StreamWindow]:
    """The consecutive windows of a source with their trigger ranges and entity keys (module docstring)."""
    plan = plan_windows(src, window_updates=cfg.training.window_updates, max_entities=cfg.training.max_entities)
    out: list[StreamWindow] = []
    for k, (a, b) in enumerate(plan):
        t_first, t_last = float(src.time[a]), float(src.time[b - 1])
        hi = float(src.time[plan[k + 1][0]]) if k + 1 < len(plan) else float(np.nextafter(t_last, math.inf))
        out.append(StreamWindow(index=k, start=a, stop=b, t_first=t_first, t_last=t_last, trigger_lo=t_first,
                                trigger_hi=max(hi, float(np.nextafter(t_last, math.inf))),
                                entity_keys=_local_keys(src, a, b)))
    return out


def segment_records(
    src_index: int,
    src: PreparedSource,
    stream: Sequence[StreamWindow],
    *,
    segment_seconds: float,
    family_key: str = "family",
) -> list[WindowRecord]:
    """Group consecutive windows into segments spanning ≥ `segment_seconds` (AS-333), as `WindowRecord`s.

    A segment's record covers updates [first window start, last window stop); the trailing run of a
    source may be shorter than `segment_seconds`.
    """
    if segment_seconds <= 0:
        raise InvariantViolation("segment_seconds must be > 0")
    out: list[WindowRecord] = []
    group: list[StreamWindow] = []

    def close() -> None:
        a, b = group[0].start, group[-1].stop
        dom, fams = _family_of(src, a, b, family_key)
        out.append(WindowRecord(
            id=f"{src.data.source_id}:seg{len(out)}", source=src_index, start=a, stop=b, network=src.data.network,
            t_start=group[0].t_first, t_end=group[-1].t_last, family=dom, families=fams, origin=src.data.origin,
            derived_from=f"{src.data.derived_from}:seg{len(out)}" if src.data.derived_from else None,
        ))

    for w in stream:
        group.append(w)
        if w.t_last - group[0].t_first >= segment_seconds:
            close()
            group = []
    if group:
        close()
    return out


def default_segment_seconds(cfg: NagaHanaConfig) -> float:
    """K·c: a segment holds at least the forecast horizon of cadence steps (AS-333)."""
    return cfg.forecaster.horizon_k * cfg.forecaster.window_seconds


@dataclass
class StreamContext:
    """What the carry of one lane needs for one window (module docstring)."""

    lane: int
    segment_id: str
    step: int                         # window number inside the segment (0 = reset)
    reset: bool                       # True: clear this lane's carried Environment before this window
    source_id: str
    window_start: int
    window_stop: int
    origin: float
    origin_shift: float               # origin − previous window's origin (0 at a reset)
    entity_keys: np.ndarray           # [V] stable source entity rows of the window's local entities
    key_to_index: dict[int, int]
    carried_keys: np.ndarray          # sorted source entity rows seen earlier in the segment
    carried_index: np.ndarray         # [V] position in carried_keys, −1 if new in this window
    novelty: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class _Lane:
    segment: WindowRecord
    windows: list[StreamWindow]
    pos: int = 0
    seen: set[int] = field(default_factory=set)
    prev_origin: float | None = None


class StreamLoader:
    """B lanes walking segments window by window; yields (WindowBatch, LabelBatch, [StreamContext]).

    sources: prepared sources; segments: the segment records to serve (one split); order: indices into
    `segments` in the order lanes take them (e.g. a `ClassBalancedSampler` draw; default: as given);
    lanes: parallel streams (default `TrainingConfig.batch_windows`); label_limits: segment id →
    epoch label limit (`sampling.label_limits`); novelty: segment id → "known" | "novel";
    perturb_seed: out-of-order augmentation (training only, §4b.5).
    """

    def __init__(
        self,
        sources: Sequence[PreparedSource],
        segments: Sequence[WindowRecord],
        cfg: NagaHanaConfig,
        *,
        order: Sequence[int] | None = None,
        lanes: int | None = None,
        label_limits: dict[str, float] | None = None,
        novelty: dict[str, str] | None = None,
        structure_fn: StructureFn | None = None,
        collate_fn: Any = None,
        perturb_seed: int | None = None,
    ) -> None:
        self.sources = list(sources)
        self.segments = list(segments)
        self.cfg = cfg
        self.order = list(order) if order is not None else list(range(len(self.segments)))
        self.lanes = lanes or cfg.training.batch_windows
        self.label_limits = dict(label_limits or {})
        self.novelty = dict(novelty or {})
        self.structure_fn = structure_fn
        self.collate_fn = collate_fn
        self.perturb_seed = perturb_seed
        self._streams: dict[int, list[StreamWindow]] = {}

    def _stream(self, s: int) -> list[StreamWindow]:
        if s not in self._streams:
            self._streams[s] = plan_stream(self.sources[s], self.cfg)
        return self._streams[s]

    def _windows_of(self, seg: WindowRecord) -> list[StreamWindow]:
        ws = [w for w in self._stream(seg.source) if w.start >= seg.start and w.stop <= seg.stop]
        if not ws or ws[0].start != seg.start or ws[-1].stop != seg.stop:
            raise InvariantViolation(f"segment {seg.id} does not align with the source's stream windows")
        return ws

    def __iter__(self) -> Iterator[tuple[WindowBatch, LabelBatch, list[StreamContext]]]:
        queue = list(self.order)
        lanes: list[_Lane | None] = [None] * self.lanes
        step_no = 0
        while True:
            items: list[WindowItem] = []
            contexts: list[StreamContext] = []
            for b in range(self.lanes):
                lane = lanes[b]
                if lane is None or lane.pos >= len(lane.windows):
                    if not queue:
                        lanes[b] = None
                        continue
                    seg = self.segments[queue.pop(0)]
                    lane = lanes[b] = _Lane(segment=seg, windows=self._windows_of(seg))
                w = lane.windows[lane.pos]
                src = self.sources[lane.segment.source]
                rng = (np.random.default_rng([self.perturb_seed, step_no, b]) if self.perturb_seed is not None else None)
                item = build_window(
                    src, w.start, w.stop, self.cfg, structure_fn=self.structure_fn, perturb_rng=rng,
                    trigger_window=(w.trigger_lo, w.trigger_hi),
                    label_limit=self.label_limits.get(lane.segment.id, math.inf),
                )
                keys = item.entity_rows
                carried = np.array(sorted(lane.seen), dtype=np.int64)
                pos_of = {int(k): i for i, k in enumerate(carried.tolist())}
                ctx = StreamContext(
                    lane=b, segment_id=lane.segment.id, step=lane.pos, reset=lane.pos == 0, source_id=src.data.source_id,
                    window_start=w.start, window_stop=w.stop, origin=item.origin,
                    origin_shift=0.0 if lane.prev_origin is None else item.origin - lane.prev_origin,
                    entity_keys=keys, key_to_index={int(k): i for i, k in enumerate(keys.tolist())},
                    carried_keys=carried, carried_index=np.array([pos_of.get(int(k), -1) for k in keys.tolist()], dtype=np.int64),
                    novelty=self.novelty.get(lane.segment.id, ""),
                )
                item.extra["stream"] = ctx
                item.extra["novelty"] = ctx.novelty
                lane.seen.update(int(k) for k in keys.tolist())
                lane.prev_origin = item.origin
                lane.pos += 1
                items.append(item)
                contexts.append(ctx)
            if not items:
                return
            window, labels = collate_items(items, self.cfg, novelty=[c.novelty for c in contexts], collate_fn=self.collate_fn)
            step_no += 1
            yield window, labels, contexts


__all__ = [
    "StreamContext", "StreamLoader", "StreamWindow", "default_segment_seconds", "plan_stream", "segment_records",
]
