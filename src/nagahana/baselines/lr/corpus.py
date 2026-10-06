"""The LR corpus: per-source update and trigger tables built from the windows NagaHana trains on.

Where the rows come from. Each source is walked in stream order (D-51): `data.stream.plan_stream` gives
its consecutive windows with trigger ranges that partition the cadence grid, and every window is built
by `data.windows.build_window` with that trigger range and the label limit of its record (AS-334), as the
StreamLoader does, without the graph structure the baseline does not read. The same tables can be
filled from collated (WindowBatch, LabelBatch, StreamContext) triples (`CorpusBuilder.add_batch`), so a
NagaHana evaluation loop can extract the baseline's rows from the batches it already holds. Either way
the values, statuses, times, triggers and labels are those of NagaHana's windows.

Update table (one row per state update, stream order): epoch time, the value/status matrices in the
canonical layout, flow key, responder entity, record (segment) index, malicious flag, stage code,
family, and the flow increments of packets and bytes (features.flow_deltas).

Trigger table (one row per cadence trigger): epoch time, record index, the window's family and the
forecast targets of NagaHana's Forecaster, computed by the same function (`forecaster.losses.
survival_targets`, AS-18) on the window's labels (AS-523):

    usable          no internal entity is already infiltrated at tau (P_inf is 1 there, so the trigger
                    is not a forecast unit)
    event_step      first step k with infiltration (t* in (tau + (k - 1) w, tau + k w]), 0 if none observed
    observed_steps  floor((end - tau) / w) clipped to [0, K], at least event_step for an event, where end is
                    the window's label horizon min(last time + K w, label limit)
    event_time      t* - tau for an event, else min(K w, end - tau): the censoring time (seconds)
    stage_step      for k = 1 ... K, the furthest stage among the labelled updates of the source in step k
                    (the rule of `forecaster.losses.step_labels`), -1 when the step has no labelled update
                    or is not fully inside the label horizon (AS-510)
    infiltrated_now the infiltration state holds in the window ending at the trigger: an internal entity
                    with an update in (tau - w, tau] has its first infiltration at or before tau (AS-521)

Source bins. Per cadence window of the whole source (all records): the window states of
features.window_states, the furthest stage and whether a malicious update occurred. They give the
step-stage labels, the observed future states that the ridge forecaster is scored on, and the
persistence references.

Timeless sources (all times 0, as CIC-IoT-2023; AS-307) give detection rows only: they carry no time, so
no trigger of theirs is a forecast unit (AS-520).

Persistence. `LRCorpus.save` writes one shard directory per source (NumPy .npy arrays, loadable as
memory maps, plus a JSON header); `LRCorpus.load` reads them back, so corpora larger than memory are
processed one source at a time.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from nagahana.core.errors import InvariantViolation
from nagahana.data.windows import COLUMN_SLOTS

from .features import SourceArrays, bin_of, empty_states, flow_deltas, flow_totals, window_states

if TYPE_CHECKING:
    from nagahana.data.sampling import SplitManifest, WindowRecord
    from nagahana.data.stream import StreamContext
    from nagahana.data.windows import PreparedSource, WindowItem
    from nagahana.models.batch import LabelBatch, WindowBatch
    from nagahana.models.config import NagaHanaConfig

ROLES: tuple[str, ...] = ("train", "val", "test", "zero_shot", "excluded")
UNIT_ROLES: frozenset[str] = frozenset({"train", "val", "test", "zero_shot"})
CORPUS_FORMAT = "nagahana.lr.corpus"
CORPUS_VERSION = 1


@dataclass
class SegmentInfo:
    """One record (segment or window) of a source: its split role and where its rows are."""

    id: str
    role: str
    novelty: str
    label_limit: float
    t_first: float
    t_last: float
    start: int          # first row of the source's update table
    stop: int           # one past the last row

    def as_dict(self) -> dict[str, Any]:
        return {"id": self.id, "role": self.role, "novelty": self.novelty,
                "label_limit": None if math.isinf(self.label_limit) else self.label_limit,
                "t_first": self.t_first, "t_last": self.t_last, "start": self.start, "stop": self.stop}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> SegmentInfo:
        lim = d["label_limit"]
        return cls(str(d["id"]), str(d["role"]), str(d["novelty"]), math.inf if lim is None else float(lim),
                   float(d["t_first"]), float(d["t_last"]), int(d["start"]), int(d["stop"]))


@dataclass
class SourceTables:
    """Update, trigger and bin tables of one source (module docstring). Arrays are NumPy (or memory maps)."""

    source_id: str
    dataset: str
    network: str
    origin: str
    timeless: bool
    window_seconds: float
    horizon_k: int
    segments: list[SegmentInfo]
    arrays: dict[str, np.ndarray] = field(default_factory=dict)

    UPDATE_KEYS: tuple[str, ...] = ("time", "values", "status", "flow", "responder", "segment", "malicious", "stage",
                                    "family", "d_packets", "d_bytes_ip", "d_bytes_payload")
    TRIGGER_KEYS: tuple[str, ...] = ("t_time", "t_segment", "t_family", "t_usable", "t_event_step", "t_observed_steps",
                                     "t_event_time", "t_event_observed", "t_horizon_end", "t_stage_step",
                                     "t_infiltrated_now")
    BIN_KEYS: tuple[str, ...] = ("b_g", "b_state", "b_stage_max", "b_malicious", "b_infil_min")

    def __getitem__(self, key: str) -> np.ndarray:
        return self.arrays[key]

    @property
    def n_updates(self) -> int:
        return int(self.arrays["time"].shape[0])

    @property
    def n_triggers(self) -> int:
        return int(self.arrays["t_time"].shape[0])

    @property
    def t_first(self) -> float:
        return float(self.arrays["time"][0]) if self.n_updates else math.nan

    @property
    def t_last(self) -> float:
        return float(self.arrays["time"][-1]) if self.n_updates else math.nan

    def update_roles(self) -> np.ndarray:
        """Role of every update (str [n])."""
        roles = np.asarray([s.role for s in self.segments] or ["excluded"], dtype="<U9")
        return roles[np.asarray(self.arrays["segment"], dtype=np.int64)]

    def trigger_roles(self) -> np.ndarray:
        roles = np.asarray([s.role for s in self.segments] or ["excluded"], dtype="<U9")
        return roles[np.asarray(self.arrays["t_segment"], dtype=np.int64)]

    def update_novelty(self) -> np.ndarray:
        nov = np.asarray([s.novelty for s in self.segments] or [""], dtype="<U5")
        return nov[np.asarray(self.arrays["segment"], dtype=np.int64)]

    def trigger_novelty(self) -> np.ndarray:
        nov = np.asarray([s.novelty for s in self.segments] or [""], dtype="<U5")
        return nov[np.asarray(self.arrays["t_segment"], dtype=np.int64)]

    def source_arrays(self, rows: slice | np.ndarray | None = None) -> SourceArrays:
        """The arrays window states and aggregates need, for all rows or a subset."""
        r = slice(None) if rows is None else rows
        a = self.arrays
        return SourceArrays(values=np.asarray(a["values"][r]), status=np.asarray(a["status"][r]),
                            flow=np.asarray(a["flow"][r]), responder=np.asarray(a["responder"][r]),
                            d_packets=np.asarray(a["d_packets"][r]), d_bytes_ip=np.asarray(a["d_bytes_ip"][r]),
                            d_bytes_payload=np.asarray(a["d_bytes_payload"][r]))

    def supplies(self) -> dict[str, bool]:
        return self.source_arrays().supplies()

    def bin_lookup(self, g: np.ndarray) -> np.ndarray:
        """Row of the source-bin table for each window index g, -1 where the window holds no update."""
        bg = np.asarray(self.arrays["b_g"], dtype=np.int64)
        g = np.asarray(g, dtype=np.int64)
        pos = np.searchsorted(bg, g)
        ok = (pos < bg.size) & (bg[np.minimum(pos, max(bg.size - 1, 0))] == g) if bg.size else np.zeros(g.shape, bool)
        return np.where(ok, pos, -1)

    def states_at(self, g: np.ndarray, until: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Raw window states [q, 9] of the windows g of this source and whether each is observed.

        A window is observed when it ends at or before `until` (epoch) and lies inside the source's observed
        span [t_first, t_last]; an observed window without updates has the states of `empty_states`.
        """
        g = np.asarray(g, dtype=np.int64)
        w = self.window_seconds
        row = self.bin_lookup(g)
        lo, hi = (g - 1) * w, g * w
        observed = (hi <= np.asarray(until, dtype=np.float64)) & (lo >= self.t_first) & (hi <= self.t_last)
        st = np.broadcast_to(empty_states(self.supplies()), (g.size, 9)).copy()
        has = row >= 0
        if has.any():
            st[has] = np.asarray(self.arrays["b_state"])[row[has]]
        st[~observed] = np.nan
        return st, observed


def _no_structure(**_: Any) -> None:
    """The baseline reads no graph structure; window building skips it."""
    return None


@dataclass
class _WindowView:
    """The unpadded arrays of one window that the corpus needs (from a WindowItem or a collated batch)."""

    record: str
    origin: float
    update_time: np.ndarray          # [U] relative
    values: np.ndarray               # [U, C] float32
    status: np.ndarray               # [U, C]
    responder: np.ndarray            # [U] global entity rows (-1 none)
    initiator: np.ndarray            # [U] global entity rows (-1 none)
    flow: np.ndarray                 # [U]
    malicious: np.ndarray            # [U] float32
    stage: np.ndarray                # [U]
    family: np.ndarray               # [U] str
    trigger_time: np.ndarray         # [M] relative
    infiltrated_at: np.ndarray       # [V] relative (+inf never)
    internal: np.ndarray             # [V] bool
    entity_rows: np.ndarray          # [V] global entity rows of the window's entities
    label_horizon: float             # relative
    window_family: str


class SourceAccumulator:
    """Collects the windows of one source in stream order and finishes them into `SourceTables`."""

    def __init__(self, source_id: str, *, dataset: str, network: str, origin: str, timeless: bool,
                 window_seconds: float, horizon_k: int) -> None:
        self.source_id, self.dataset, self.network, self.origin = source_id, dataset, network, origin
        self.timeless = timeless
        self.w = float(window_seconds)
        self.k = int(horizon_k)
        self.views: list[_WindowView] = []
        self.records: dict[str, tuple[str, str, float]] = {}       # id -> (role, novelty, label limit)

    def add_record(self, record: str, role: str, novelty: str, label_limit: float) -> None:
        if role not in ROLES:
            raise InvariantViolation(f"unknown role {role!r}")
        if novelty and role != "zero_shot":
            raise InvariantViolation("only zero-shot records carry a novelty mark (D-23)")
        self.records[record] = (role, novelty, float(label_limit))

    def add_view(self, v: _WindowView) -> None:
        if v.record not in self.records:
            raise InvariantViolation(f"window of unregistered record {v.record!r}")
        if self.views:
            last = self.views[-1]
            if v.update_time.size and last.update_time.size and \
                    v.origin + float(v.update_time[0]) < last.origin + float(last.update_time[-1]):
                raise InvariantViolation(f"{self.source_id}: windows must arrive in stream (time) order")
        self.views.append(v)

    def finish(self) -> SourceTables:
        """Concatenate the windows and compute deltas, source bins, targets and step stages."""
        import torch

        from nagahana.models.forecaster.losses import survival_targets

        if not self.views:
            raise InvariantViolation(f"{self.source_id}: no windows were added")
        w, k = self.w, self.k
        rec_ids = list(dict.fromkeys(v.record for v in self.views))
        rec_index = {r: i for i, r in enumerate(rec_ids)}
        times, vals, stats, flows, resp, ini, segs, mal, stage, fam = [], [], [], [], [], [], [], [], [], []
        inf_ent: list[np.ndarray] = []                              # internal entities with a finite first infiltration
        inf_time: list[np.ndarray] = []                             # and that time (epoch), one entry per window
        t_time, t_seg, t_fam, t_usable, t_ev, t_obs, t_etime, t_eobs, t_end = [], [], [], [], [], [], [], [], []
        for v in self.views:
            u = v.update_time.shape[0]
            times.append(v.origin + v.update_time.astype(np.float64))
            vals.append(np.asarray(v.values, dtype=np.float32))
            stats.append(np.asarray(v.status, dtype=np.uint8))
            flows.append(np.asarray(v.flow, dtype=np.int64))
            resp.append(np.asarray(v.responder, dtype=np.int64))
            ini.append(np.asarray(v.initiator, dtype=np.int64))
            segs.append(np.full(u, rec_index[v.record], dtype=np.int32))
            # an entity's first infiltration is the same fact in every window that holds it within its horizon
            t_inf = np.asarray(v.infiltrated_at, dtype=np.float64)
            keep = np.isfinite(t_inf) & np.asarray(v.internal, dtype=bool)
            inf_ent.append(np.asarray(v.entity_rows, dtype=np.int64)[keep])
            inf_time.append(v.origin + t_inf[keep])
            mal.append(np.asarray(v.malicious, dtype=np.float32))
            stage.append(np.asarray(v.stage, dtype=np.int64))
            fam.append(np.asarray(v.family, dtype=str))
            m = v.trigger_time.shape[0]
            if m == 0:
                continue
            tau = v.trigger_time.astype(np.float64)
            infil = np.asarray(v.infiltrated_at, dtype=np.float64)
            internal = np.asarray(v.internal, dtype=bool)
            ev, cens, usable = survival_targets(
                torch.from_numpy(tau)[None], torch.ones(1, m, dtype=torch.bool), torch.from_numpy(infil)[None],
                torch.from_numpy(internal)[None], torch.tensor([float(v.label_horizon)], dtype=torch.float64),
                window_seconds=w, horizon_k=k)
            ev_n, is_event, use = ev[0].numpy().astype(np.int64), ~cens[0].numpy(), usable[0].numpy()
            end = float(v.label_horizon)
            obs = np.floor(np.clip((end - tau) / w, 0.0, float(k))).astype(np.int64)
            event_step = np.where(is_event, ev_n, 0)
            observed = np.where(is_event, np.maximum(obs, ev_n), obs)
            # time of the next infiltration of an internal entity after tau (the rule of survival_targets)
            cand = np.where(internal, infil, np.inf)
            later = np.where(cand[None, :] > tau[:, None], cand[None, :], np.inf)
            t_next = later.min(axis=1) if cand.size else np.full(m, np.inf)
            etime = np.where(is_event, t_next - tau, np.minimum(k * w, np.maximum(end - tau, 0.0)))
            t_time.append(v.origin + tau)
            t_seg.append(np.full(m, rec_index[v.record], dtype=np.int32))
            t_fam.append(np.full(m, v.window_family, dtype=object))
            t_usable.append(use.astype(bool))
            t_ev.append(event_step)
            t_obs.append(observed.astype(np.int64))
            t_etime.append(etime.astype(np.float64))
            t_eobs.append(is_event.astype(bool))
            t_end.append(np.full(m, v.origin + end))
        a: dict[str, np.ndarray] = {
            "time": np.concatenate(times), "values": np.concatenate(vals), "status": np.concatenate(stats),
            "flow": np.concatenate(flows), "responder": np.concatenate(resp), "segment": np.concatenate(segs),
            "malicious": np.concatenate(mal), "stage": np.concatenate(stage),
            "family": np.asarray(np.concatenate(fam), dtype=str),
        }
        if a["values"].shape[1] != len(COLUMN_SLOTS):
            raise InvariantViolation("windows must use the canonical column layout")
        if np.any(np.diff(a["time"]) < 0):
            raise InvariantViolation(f"{self.source_id}: update times must be non-decreasing in stream order")
        packets, bytes_ip, bytes_pl = flow_totals(a["values"], a["status"])
        a["d_packets"] = flow_deltas(packets, a["flow"])
        a["d_bytes_ip"] = flow_deltas(bytes_ip, a["flow"])
        a["d_bytes_payload"] = flow_deltas(bytes_pl, a["flow"])
        # triggers
        if t_time:
            a.update({"t_time": np.concatenate(t_time), "t_segment": np.concatenate(t_seg),
                      "t_family": np.asarray(np.concatenate(t_fam), dtype=str), "t_usable": np.concatenate(t_usable),
                      "t_event_step": np.concatenate(t_ev), "t_observed_steps": np.concatenate(t_obs),
                      "t_event_time": np.concatenate(t_etime), "t_event_observed": np.concatenate(t_eobs),
                      "t_horizon_end": np.concatenate(t_end)})
        else:
            a.update({"t_time": np.zeros(0), "t_segment": np.zeros(0, np.int32), "t_family": np.zeros(0, dtype="<U1"),
                      "t_usable": np.zeros(0, bool), "t_event_step": np.zeros(0, np.int64),
                      "t_observed_steps": np.zeros(0, np.int64), "t_event_time": np.zeros(0),
                      "t_event_observed": np.zeros(0, bool), "t_horizon_end": np.zeros(0)})
        # source bins (all records)
        g = bin_of(a["time"], w)
        bg, bidx = np.unique(g, return_inverse=True)
        arr = SourceArrays(values=a["values"], status=a["status"], flow=a["flow"], responder=a["responder"],
                           d_packets=a["d_packets"], d_bytes_ip=a["d_bytes_ip"], d_bytes_payload=a["d_bytes_payload"])
        a["b_g"] = bg.astype(np.int64)
        a["b_state"] = window_states(arr, bidx.astype(np.int64), bg.size, arr.supplies())
        starts = np.r_[0, np.flatnonzero(np.diff(bidx) != 0) + 1]
        a["b_stage_max"] = np.maximum.reduceat(a["stage"], starts).astype(np.int64)
        a["b_malicious"] = np.maximum.reduceat((a["malicious"] == 1.0).astype(np.int8), starts).astype(bool)
        # earliest first infiltration of an internal endpoint of the bin's updates (+inf none): the infiltration
        # state holds in the window ending at tau when it is <= tau (ForecastPredictions.infiltrated_now)
        ents = np.concatenate(inf_ent) if inf_ent else np.zeros(0, np.int64)
        ent_t = np.concatenate(inf_time) if inf_time else np.zeros(0)
        known, inv = np.unique(ents, return_inverse=True)
        first = np.full(known.size, np.inf)
        np.minimum.at(first, inv, ent_t)                                       # [E] first infiltration per entity
        ends = np.stack([np.concatenate(ini), a["responder"]], axis=1)         # [n, 2] initiator, responder
        pos = np.searchsorted(known, ends)
        hit = (pos < known.size) & (known[np.minimum(pos, max(known.size - 1, 0))] == ends) if known.size else np.zeros(ends.shape, bool)
        end_first = np.where(hit, first[np.minimum(pos, max(known.size - 1, 0))] if known.size else np.inf, np.inf)
        a["b_infil_min"] = np.minimum.reduceat(end_first.min(axis=1), starts)
        # step stages: the furthest labelled stage of source window g_tau + k, if fully inside the label horizon
        tg = bin_of(a["t_time"], w)
        steps = tg[:, None] + np.arange(1, k + 1)[None, :]                            # [m, K]
        pos = np.searchsorted(bg, steps)
        hit = (pos < bg.size) & (bg[np.minimum(pos, bg.size - 1)] == steps)
        inside = steps * w <= a["t_horizon_end"][:, None]
        a["t_stage_step"] = np.where(hit & inside, a["b_stage_max"][np.minimum(pos, bg.size - 1)], -1).astype(np.int64)
        own = np.searchsorted(bg, tg)
        own_hit = (own < bg.size) & (bg[np.minimum(own, bg.size - 1)] == tg)
        a["t_infiltrated_now"] = own_hit & (a["b_infil_min"][np.minimum(own, bg.size - 1)] <= a["t_time"])
        # records: row ranges and times
        segments: list[SegmentInfo] = []
        for i, r in enumerate(rec_ids):
            rows = np.flatnonzero(a["segment"] == i)
            role, nov, lim = self.records[r]
            segments.append(SegmentInfo(id=r, role=role, novelty=nov, label_limit=lim,
                                        t_first=float(a["time"][rows[0]]) if rows.size else math.nan,
                                        t_last=float(a["time"][rows[-1]]) if rows.size else math.nan,
                                        start=int(rows[0]) if rows.size else 0, stop=int(rows[-1]) + 1 if rows.size else 0))
            if rows.size and rows[-1] - rows[0] + 1 != rows.size:
                raise InvariantViolation(f"record {r} is not a contiguous run of the stream")
        return SourceTables(source_id=self.source_id, dataset=self.dataset, network=self.network, origin=self.origin,
                            timeless=self.timeless, window_seconds=w, horizon_k=k, segments=segments, arrays=a)


def _view_from_item(item: WindowItem, src: PreparedSource, record: str, start: int, stop: int) -> _WindowView:
    """A WindowItem built without perturbation, with the per-update flow keys and families of its source."""
    if not np.array_equal(item.rows, src.order[start:stop]):
        raise InvariantViolation("LR windows must be built without the out-of-order augmentation")
    loc = item.update_entities[:, 1]
    responder = np.where(loc >= 0, item.entity_rows[np.maximum(loc, 0)], -1)
    loc0 = item.update_entities[:, 0]
    initiator = np.where(loc0 >= 0, item.entity_rows[np.maximum(loc0, 0)], -1)
    upd = src.data.updates.updates
    flow = upd["flow"].to_numpy(dtype=np.int64)[item.rows] if "flow" in upd else np.full(item.rows.size, -1, np.int64)
    return _WindowView(record=record, origin=float(item.origin), update_time=item.update_time, values=item.values,
                       status=item.status, responder=responder.astype(np.int64), initiator=initiator.astype(np.int64),
                       flow=flow,
                       malicious=item.update_malicious, stage=item.update_stage,
                       family=np.asarray(src.family[start:stop], dtype=str), trigger_time=item.trigger_time,
                       infiltrated_at=item.entity_infiltrated_at, internal=item.entity_internal,
                       entity_rows=np.asarray(item.entity_rows, dtype=np.int64),
                       label_horizon=float(item.label_horizon), window_family=str(item.family))


def extract_source(src: PreparedSource, records: Sequence[WindowRecord], *, roles: Mapping[str, str],
                   novelty: Mapping[str, str], label_limits: Mapping[str, float], cfg: NagaHanaConfig) -> SourceTables:
    """Walk one source's records in stream order and build its tables (module docstring)."""
    from nagahana.data.stream import plan_stream
    from nagahana.data.windows import build_window

    stream = plan_stream(src, cfg)
    acc = SourceAccumulator(src.data.source_id, dataset=src.data.dataset, network=src.data.network,
                            origin=src.data.origin, timeless=src.timeless, window_seconds=cfg.forecaster.window_seconds,
                            horizon_k=cfg.forecaster.horizon_k)
    for rec in sorted(records, key=lambda r: (r.start, r.id)):
        ws = [x for x in stream if x.start >= rec.start and x.stop <= rec.stop]
        if not ws or ws[0].start != rec.start or ws[-1].stop != rec.stop:
            raise InvariantViolation(f"record {rec.id} does not align with the source's stream windows")
        lim = float(label_limits.get(rec.id, math.inf))
        acc.add_record(rec.id, roles[rec.id], novelty.get(rec.id, ""), lim)
        for x in ws:
            item = build_window(src, x.start, x.stop, cfg, structure_fn=_no_structure, perturb_rng=None,
                                trigger_window=(x.trigger_lo, x.trigger_hi), label_limit=lim)
            acc.add_view(_view_from_item(item, src, rec.id, x.start, x.stop))
    return acc.finish()


class CorpusBuilder:
    """Fills an `LRCorpus` from collated stream batches (D-51), source by source.

    sources: source id -> PreparedSource (for flow keys and per-update families, which batches do not
    carry); roles / label_limits: record (segment) id -> role / label limit. Call `add_batch` with every
    (WindowBatch, LabelBatch, [StreamContext]) a StreamLoader yields, then `finish`.
    """

    def __init__(self, sources: Mapping[str, PreparedSource], cfg: NagaHanaConfig, *, roles: Mapping[str, str],
                 label_limits: Mapping[str, float] | None = None) -> None:
        self.sources = dict(sources)
        self.cfg = cfg
        self.roles = dict(roles)
        self.label_limits = dict(label_limits or {})
        self.acc: dict[str, SourceAccumulator] = {}
        self.pending: dict[str, list[tuple[float, _WindowView]]] = {}

    def add_batch(self, window: WindowBatch, labels: LabelBatch, contexts: Sequence[StreamContext]) -> None:
        if len(contexts) != window.update_mask.shape[0]:
            raise InvariantViolation("one StreamContext per batch element is required")
        for b, ctx in enumerate(contexts):
            src = self.sources[ctx.source_id]
            if ctx.source_id not in self.acc:
                self.acc[ctx.source_id] = SourceAccumulator(
                    ctx.source_id, dataset=src.data.dataset, network=src.data.network, origin=src.data.origin,
                    timeless=src.timeless, window_seconds=self.cfg.forecaster.window_seconds,
                    horizon_k=self.cfg.forecaster.horizon_k)
                self.pending[ctx.source_id] = []
            acc = self.acc[ctx.source_id]
            if ctx.segment_id not in acc.records:
                acc.add_record(ctx.segment_id, self.roles[ctx.segment_id], ctx.novelty,
                               float(self.label_limits.get(ctx.segment_id, math.inf)))
            u = int(window.update_mask[b].sum())
            v = int(window.entity_mask[b].sum())
            tm = window.triggers.mask[b]
            loc = window.update_entities[b, :u, 1].numpy()
            keys = np.asarray(ctx.entity_keys, dtype=np.int64)
            responder = np.where(loc >= 0, keys[np.maximum(loc, 0)], -1)
            loc0 = window.update_entities[b, :u, 0].numpy()
            initiator = np.where(loc0 >= 0, keys[np.maximum(loc0, 0)], -1)
            rows = src.order[ctx.window_start:ctx.window_stop]
            if rows.size != u:
                raise InvariantViolation("a batch window does not match its source range")
            upd = src.data.updates.updates
            flow = upd["flow"].to_numpy(dtype=np.int64)[rows] if "flow" in upd else np.full(u, -1, np.int64)
            if labels.label_horizon is None:
                raise InvariantViolation("LabelBatch.label_horizon is required (censoring)")
            view = _WindowView(
                record=ctx.segment_id, origin=float(window.origin[b]), update_time=window.update_time[b, :u].numpy(),
                values=window.fields.values[b, :u].numpy(), status=window.fields.status[b, :u].numpy(),
                responder=responder, initiator=initiator, flow=flow, malicious=labels.update_malicious[b, :u].numpy(),
                stage=labels.update_stage[b, :u].numpy(),
                family=np.asarray(src.family[ctx.window_start:ctx.window_stop], dtype=str),
                trigger_time=window.triggers.time[b][tm].numpy(), infiltrated_at=labels.entity_infiltrated_at[b, :v].numpy(),
                internal=window.entity_internal[b, :v].numpy(), entity_rows=keys[:v],
                label_horizon=float(labels.label_horizon[b]),
                window_family=str(labels.family[b]))
            # lanes interleave segments, so windows are kept and ordered by time per source at finish
            self.pending[ctx.source_id].append((float(window.origin[b]) + (float(view.update_time[0]) if u else 0.0), view))

    def finish(self) -> LRCorpus:
        tables = []
        for sid, acc in self.acc.items():
            for _t, view in sorted(self.pending[sid], key=lambda p: p[0]):
                acc.add_view(view)
            tables.append(acc.finish())
        return LRCorpus(tables)


@dataclass
class LRCorpus:
    """The tables of every source (module docstring)."""

    sources: list[SourceTables]

    def __post_init__(self) -> None:
        if not self.sources:
            raise InvariantViolation("an LR corpus needs at least one source")
        ws = {s.window_seconds for s in self.sources}
        ks = {s.horizon_k for s in self.sources}
        if len(ws) != 1 or len(ks) != 1:
            raise InvariantViolation("every source of a corpus must share the cadence and the horizon")

    @property
    def window_seconds(self) -> float:
        return self.sources[0].window_seconds

    @property
    def horizon_k(self) -> int:
        return self.sources[0].horizon_k

    def save(self, path: str | Path) -> None:
        """One shard directory per source: arrays as .npy, the rest in shard.json (module docstring)."""
        root = Path(path)
        root.mkdir(parents=True, exist_ok=True)
        index = []
        for i, s in enumerate(self.sources):
            d = root / f"source_{i:04d}"
            d.mkdir(exist_ok=True)
            for key, arr in s.arrays.items():
                a = np.asarray(arr)
                np.save(d / f"{key}.npy", a.astype(str) if a.dtype == object else a, allow_pickle=False)
            header = {"source_id": s.source_id, "dataset": s.dataset, "network": s.network, "origin": s.origin,
                      "timeless": s.timeless, "window_seconds": s.window_seconds, "horizon_k": s.horizon_k,
                      "segments": [g.as_dict() for g in s.segments], "arrays": sorted(s.arrays)}
            (d / "shard.json").write_text(json.dumps(header, indent=1), encoding="utf-8")
            index.append(d.name)
        (root / "corpus.json").write_text(json.dumps({"format": CORPUS_FORMAT, "version": CORPUS_VERSION,
                                                      "columns": list(COLUMN_SLOTS), "shards": index}, indent=1),
                                          encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path, *, mmap: bool = True) -> LRCorpus:
        root = Path(path)
        meta = json.loads((root / "corpus.json").read_text(encoding="utf-8"))
        if meta.get("format") != CORPUS_FORMAT or int(meta.get("version", -1)) != CORPUS_VERSION:
            raise InvariantViolation(f"{root} is not an LR corpus of version {CORPUS_VERSION}")
        if tuple(meta["columns"]) != COLUMN_SLOTS:
            raise InvariantViolation("the corpus was written with another canonical column layout")
        out = []
        for name in meta["shards"]:
            d = root / name
            h = json.loads((d / "shard.json").read_text(encoding="utf-8"))
            arrays = {k: np.load(d / f"{k}.npy", mmap_mode="r" if mmap and k in ("values", "status") else None,
                                 allow_pickle=False) for k in h["arrays"]}
            out.append(SourceTables(source_id=h["source_id"], dataset=h["dataset"], network=h["network"],
                                    origin=h["origin"], timeless=bool(h["timeless"]),
                                    window_seconds=float(h["window_seconds"]), horizon_k=int(h["horizon_k"]),
                                    segments=[SegmentInfo.from_dict(g) for g in h["segments"]], arrays=arrays))
        return cls(out)


def build_corpus(sources: Sequence[PreparedSource], manifest: SplitManifest, cfg: NagaHanaConfig, *,
                 label_limits: Mapping[str, float] | None = None,
                 progress: Callable[[str], None] | None = None) -> LRCorpus:
    """Tables of every source from a split manifest over stream records (segments or windows)."""
    from nagahana.data.sampling import label_limits as manifest_limits

    lim = dict(label_limits) if label_limits is not None else manifest_limits(manifest)
    roles = {i: r.value for i, r in manifest.role.items()}
    nov = {i: n.value for i, n in manifest.novelty.items()}
    by_source: dict[int, list[WindowRecord]] = {}
    for rid, rec in manifest.records.items():
        if rid in roles:
            by_source.setdefault(rec.source, []).append(rec)
    tables = []
    for s_i in sorted(by_source):
        if progress is not None:
            progress(f"extracting {sources[s_i].data.source_id}")
        tables.append(extract_source(sources[s_i], by_source[s_i], roles=roles, novelty=nov, label_limits=lim, cfg=cfg))
    return LRCorpus(tables)


__all__ = ["CORPUS_FORMAT", "CorpusBuilder", "LRCorpus", "ROLES", "SegmentInfo", "SourceAccumulator", "SourceTables",
           "UNIT_ROLES", "build_corpus", "extract_source"]
