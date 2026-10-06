"""Data of a training run: sources, nested split manifests, exactly resumable stream loaders, step plans.

Purpose
-------
1. `ingest_source`: one raw source (capture or flow CSV) -> labelled `PreparedSource` through the
   project's adapters (ingest/pcap.py in flow-state mode, D-51; ingest/csv_flows.py) and label tables
   (data/labels.py, AS-34). `source_digest` fingerprints the prepared content (run manifests, and the
   check that later stages read exactly the data stage 1 analysed).
2. `plan_splits`: the full split manifest over segments (60 % train, 20 % test, 20 % validation;
   zero-shot = novel families and held-out networks; data/sampling.py, AS-326 ... AS-328, AS-35) and,
   nested inside its training split, the pretraining manifest of stages 1 and 2 (70 % / 30 %, AS-590).
   Test, validation and zero-shot segments are therefore never read by any training stage.
3. `ResumableStreamLoader`: the stream loader of data/stream.py (same windows, same contexts, same
   batches) with an explicit position, `state_dict()` / `load_state_dict()`, so a resumed run continues
   at the batch after the last one consumed (AS-576).
4. `epoch_order`, `shard_order`, `plan_steps`: the class-balanced order of an epoch (AS-325), the
   rank's share of it (AS-577), and the exact number of synchronised optimiser steps a stage will take
   (computed from window plans and cadence grids, without building any window), which the learning-rate
   schedule and the STAGED switch need (AS-570, AS-597).

Decisions: D-23, D-51. Assumptions: AS-34, AS-35, AS-325 ... AS-334, AS-577, AS-580, AS-590.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from nagahana.core.errors import ConfigMissing, InvariantViolation
from nagahana.data.collate import collate_items
from nagahana.data.sampling import (
    ClassBalancedSampler,
    Role,
    SplitManifest,
    SplitPolicy,
    WindowRecord,
    assign_splits,
    label_limits,
)
from nagahana.data.stream import StreamContext, StreamLoader, StreamWindow, default_segment_seconds, plan_stream, segment_records
from nagahana.data.windows import PreparedSource, SourceData, WindowItem, build_window, prepare_source
from nagahana.models.batch import LabelBatch, WindowBatch
from nagahana.models.config import NagaHanaConfig
from nagahana.training.assumptions import use
from nagahana.training.config import DataConfig, SourceSpec

StreamBatch = tuple[WindowBatch, LabelBatch, list[StreamContext]]


def ingest_source(spec: SourceSpec) -> PreparedSource:
    """Read, label and prepare one source (module docstring)."""
    from nagahana.data.labels import check_label_table, map_labels

    if spec.kind == "pcap":
        from nagahana.ingest.pcap import PcapSource
        from nagahana.training.runs import LABELLERS

        if spec.labeller not in LABELLERS:
            raise ConfigMissing(f"unknown labeller {spec.labeller!r}; known: {sorted(LABELLERS)}")
        cu = PcapSource(spec.path, sandboxed=spec.sandboxed, emit="flow-state", internal_networks=spec.internal_networks,
                        max_records=spec.max_records).columnar()
        labels = LABELLERS[spec.labeller](cu)
        dataset = spec.dataset or spec.labeller or "pcap"
    else:
        from nagahana.ingest.csv_flows import CICFlowSource, CICIoT2023Source, CTU13Source

        adapters = {"cic-flows": CICFlowSource, "ctu13": CTU13Source, "ciciot2023": CICIoT2023Source}
        read = adapters[spec.kind](spec.path, internal_networks=spec.internal_networks, utc_offset_hours=spec.utc_offset_hours,
                                   max_rows=spec.max_records).read()
        cu = read.updates
        labels = map_labels(read.labels, spec.dataset)
        dataset = spec.dataset
    check_label_table(labels, len(cu))
    return prepare_source(SourceData(cu, labels, network=spec.network, dataset=dataset))


def source_digest(src: PreparedSource) -> str:
    """SHA-256 over a prepared source's content: values, statuses, times, entities and labels."""
    h = hashlib.sha256()
    cu = src.data.updates
    h.update(cu.source_id.encode("utf-8"))
    h.update(src.data.network.encode("utf-8"))
    for name in [c.name for c in cu.columns]:
        h.update(name.encode("utf-8"))
    for arr in (cu.values, cu.status, src.order, src.time, src.ents, src.malicious, src.stage):
        a = np.ascontiguousarray(arr)
        h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    h.update("\x1f".join(str(x) for x in src.family.tolist()).encode("utf-8"))
    h.update("\x1f".join(str(x) for x in src.technique.tolist()).encode("utf-8"))
    return h.hexdigest()


def source_segments(sources: Sequence[PreparedSource], cfg: NagaHanaConfig, data: DataConfig) -> list[WindowRecord]:
    """Segments of every source (AS-333), in source order."""
    seconds = data.segment_seconds if data.segment_seconds is not None else default_segment_seconds(cfg)
    out: list[WindowRecord] = []
    for i, src in enumerate(sources):
        out += segment_records(i, src, plan_stream(src, cfg), segment_seconds=seconds)
    return out


@dataclass
class SplitPlan:
    """The full manifest and, nested in its training split, the pretraining manifest (AS-590)."""

    full: SplitManifest
    pretrain: SplitManifest

    def records(self, scope: str, role: Role) -> list[WindowRecord]:
        """Records of a role in a scope ("full" or "pretrain"), in time order then id."""
        man = self.full if scope == "full" else self.pretrain
        recs = [man.records[i] for i, r in man.role.items() if r is role]
        return sorted(recs, key=lambda r: (r.t_start, r.id))

    def limits(self, scope: str) -> dict[str, float]:
        """Label limits (AS-334) of a scope; for "pretrain" the tighter of both manifests' limits."""
        outer = label_limits(self.full)
        if scope == "full":
            return outer
        inner = label_limits(self.pretrain)
        return {k: min(v, outer.get(k, math.inf)) for k, v in inner.items()}

    def novelty(self) -> dict[str, str]:
        """Zero-shot novelty marks of the full manifest ("known" | "novel")."""
        return {k: v.value for k, v in self.full.novelty.items()}


def plan_splits(records: Sequence[WindowRecord], cfg: NagaHanaConfig, data: DataConfig) -> SplitPlan:
    """The nested split plan of real segments (module docstring); both manifests are validated."""
    use("AS-590", by=__name__)
    if any(r.origin != "real" for r in records):
        raise InvariantViolation("splits are planned over real segments; variants join the training split later")
    policy = SplitPolicy.from_config(cfg, novel_families=frozenset(data.novel_families),
                                     held_out_networks=frozenset(data.held_out_networks), purge=data.purge)
    full = assign_splits(records, policy)
    full.validate()
    train = [full.records[i] for i, r in full.role.items() if r is Role.TRAIN]
    pre_policy = SplitPolicy.from_config(cfg, mode="pretrain", purge=data.purge)
    pretrain = assign_splits(train, pre_policy)
    pretrain.validate()
    return SplitPlan(full=full, pretrain=pretrain)


def epoch_order(records: Sequence[WindowRecord], *, epoch: int, seed: int, balanced: bool, power: float) -> list[int]:
    """Indices into `records` for one epoch: a class-balanced draw (AS-325) or time order."""
    if not records:
        return []
    if balanced:
        sampler = ClassBalancedSampler(records, num_samples=len(records), seed=seed, power=power)
        sampler.set_epoch(epoch)
        return list(sampler)
    return sorted(range(len(records)), key=lambda i: (records[i].t_start, records[i].id))


def shard_order(order: Sequence[int], *, rank: int, world: int) -> list[int]:
    """The rank's share of an epoch order: positions rank, rank + world, ... (AS-577)."""
    use("AS-577", by=__name__)
    return list(order[rank::world])


@dataclass
class _Lane:
    """One lane's position: which segment, which window, entities seen, previous origin."""

    segment_index: int
    segment: WindowRecord
    windows: list[StreamWindow]
    pos: int = 0
    seen: set[int] = field(default_factory=set)
    prev_origin: float | None = None


class ResumableStreamLoader(StreamLoader):
    """`data.stream.StreamLoader` with an explicit, restorable position (module docstring).

    The iteration is the stream loader's, step for step: lanes take segments from `order` in turn,
    walk their windows in time order, and the out-of-order augmentation draws from
    (perturb_seed, batch number, lane). `state_dict()` describes the position after the last batch
    yielded; a loader built with the same arguments and given that state yields the next batch.
    tests/test_training_data.py checks equality with `StreamLoader` batch for batch.
    """

    def __init__(self, sources: Sequence[PreparedSource], segments: Sequence[WindowRecord], cfg: NagaHanaConfig,
                 **kwargs: Any) -> None:
        super().__init__(sources, segments, cfg, **kwargs)
        self._queue_pos = 0
        self._step_no = 0
        self._lanes: list[_Lane | None] = [None] * self.lanes

    def state_dict(self) -> dict[str, Any]:
        """The position after the last yielded batch (plain data)."""
        lanes: list[dict[str, Any] | None] = []
        for lane in self._lanes:
            if lane is None:
                lanes.append(None)
            else:
                lanes.append({"segment_index": lane.segment_index, "segment_id": lane.segment.id, "pos": lane.pos,
                              "seen": sorted(lane.seen), "prev_origin": lane.prev_origin})
        return {"queue_pos": self._queue_pos, "step_no": self._step_no, "lanes": lanes, "order": list(self.order),
                "n_segments": len(self.segments)}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore a position (the order and segments must be the ones the state was taken with)."""
        if list(state["order"]) != list(self.order) or int(state["n_segments"]) != len(self.segments):
            raise InvariantViolation("stream state belongs to another order or segment list")
        if len(state["lanes"]) != self.lanes:
            raise InvariantViolation("stream state has another number of lanes")
        self._queue_pos = int(state["queue_pos"])
        self._step_no = int(state["step_no"])
        lanes: list[_Lane | None] = []
        for s in state["lanes"]:
            if s is None:
                lanes.append(None)
                continue
            seg = self.segments[int(s["segment_index"])]
            if seg.id != s["segment_id"]:
                raise InvariantViolation(f"stream state names segment {s['segment_id']!r}, found {seg.id!r}")
            lanes.append(_Lane(segment_index=int(s["segment_index"]), segment=seg, windows=self._windows_of(seg),
                               pos=int(s["pos"]), seen={int(k) for k in s["seen"]}, prev_origin=s["prev_origin"]))
        self._lanes = lanes

    @property
    def batches_yielded(self) -> int:
        return self._step_no

    def __iter__(self) -> Iterator[StreamBatch]:
        while True:
            items: list[WindowItem] = []
            contexts: list[StreamContext] = []
            for b in range(self.lanes):
                lane = self._lanes[b]
                if lane is None or lane.pos >= len(lane.windows):
                    if self._queue_pos >= len(self.order):
                        self._lanes[b] = None
                        continue
                    idx = self.order[self._queue_pos]
                    self._queue_pos += 1
                    seg = self.segments[idx]
                    lane = self._lanes[b] = _Lane(segment_index=idx, segment=seg, windows=self._windows_of(seg))
                w = lane.windows[lane.pos]
                src = self.sources[lane.segment.source]
                rng = (np.random.default_rng([self.perturb_seed, self._step_no, b]) if self.perturb_seed is not None else None)
                item = build_window(src, w.start, w.stop, self.cfg, structure_fn=self.structure_fn, perturb_rng=rng,
                                    trigger_window=(w.trigger_lo, w.trigger_hi),
                                    label_limit=self.label_limits.get(lane.segment.id, math.inf))
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
            self._step_no += 1
            yield window, labels, contexts


def window_has_trigger(w: StreamWindow, cadence: float) -> bool:
    """True when the window's trigger range [lo, hi) holds a cadence grid point (AS-318, AS-335)."""
    return math.ceil(w.trigger_lo / cadence) * cadence < w.trigger_hi


def simulate_batches(sources: Sequence[PreparedSource], segments: Sequence[WindowRecord], order: Sequence[int],
                     cfg: NagaHanaConfig, *, lanes: int) -> list[bool]:
    """Per batch of a stream, whether it holds a trigger, without building a window (same lane schedule)."""
    streams: dict[int, list[StreamWindow]] = {}
    cadence = cfg.forecaster.window_seconds

    def windows_of(seg: WindowRecord) -> list[StreamWindow]:
        if seg.source not in streams:
            streams[seg.source] = plan_stream(sources[seg.source], cfg)
        return [w for w in streams[seg.source] if w.start >= seg.start and w.stop <= seg.stop]

    queue = list(order)
    lane_w: list[list[StreamWindow] | None] = [None] * lanes
    pos = [0] * lanes
    out: list[bool] = []
    while True:
        any_item, trig = False, False
        for b in range(lanes):
            ws = lane_w[b]
            if ws is None or pos[b] >= len(ws):
                if not queue:
                    lane_w[b] = None
                    continue
                ws = lane_w[b] = windows_of(segments[queue.pop(0)])
                pos[b] = 0
            trig |= window_has_trigger(ws[pos[b]], cadence)
            pos[b] += 1
            any_item = True
        if not any_item:
            return out
        out.append(trig)


@dataclass(frozen=True)
class StepPlan:
    """Synchronised objective calls and optimiser steps of a stage, per epoch and in total."""

    calls_per_epoch: tuple[int, ...]
    steps_per_epoch: tuple[int, ...]
    total_steps: int


def plan_steps(sources: Sequence[PreparedSource], segments: Sequence[WindowRecord], cfg: NagaHanaConfig, *,
               orders: Sequence[Sequence[int]], world: int, lanes: int, accumulation: int, needs_trigger: bool,
               max_steps: int) -> StepPlan:
    """The exact step plan (module docstring): per epoch the minimum over ranks of objective batches
    (AS-577, AS-580), divided by the accumulation, capped by `max_steps` (0 = no cap)."""
    calls, steps = [], []
    for order in orders:
        per_rank = []
        for r in range(world):
            flags = simulate_batches(sources, segments, list(order)[r::world], cfg, lanes=lanes)
            per_rank.append(sum(flags) if needs_trigger else len(flags))
        n = min(per_rank) if per_rank else 0
        calls.append(n)
        steps.append(n // accumulation)
    total = sum(steps)
    if max_steps > 0:
        total = min(total, max_steps)
    return StepPlan(calls_per_epoch=tuple(calls), steps_per_epoch=tuple(steps), total_steps=total)


__all__ = ["ResumableStreamLoader", "SplitPlan", "StepPlan", "StreamBatch", "epoch_order", "ingest_source", "plan_splits",
           "plan_steps", "shard_order", "simulate_batches", "source_digest", "source_segments", "window_has_trigger"]
