"""The inference engine: live forecasting and forensic replay with one computation (architecture §1, §3; [Q-17]).

Purpose
-------
`Engine` ingests state updates incrementally (`ColumnarUpdates` chunks from any adapter), keeps the
Environment up to date with TSTCT's cached path, and at every trigger runs TAAFT, the Forecaster and the
Verifier's trust and calibration, returning the role contracts (`ForecastBundle`, `BeliefReadout`). The
Advisor runs on demand. `replay(path)` runs the same engine over a capture and writes a
`ForensicReport`. It runs only in RunMode.INFER_LIVE or RunMode.FORENSIC_REPLAY (`core/modes.py`).

Per state update (build-spec §1)
--------------------------------
    1. event log    the chunk is merged into one `ColumnarUpdates` with stable entity rows (`buffer.EventLog`)
    2. window       the update joins the open window, planned by the training rule (AS-317: ≤ window_updates
                    updates, ≤ max_entities entities); `data.windows.build_window` on the open window gives
                    the local hypergraph of every new position **as of its own time** (AS-418: the cost is
                    one rebuild of the open window per processed chunk — O(window) — because the streaming
                    graph builder is not implemented; old positions' graphs are unchanged by construction)
    3. CVG-AE       posterior mean of the new positions (AS-05)
    4. TSTCT.step   the new states against the `EnvironmentStore` (D-35: written by the Simulator role),
                    with neighbours from a cumulative contact ledger (AS-405) as of each state's time
    5. triggers     the cadence grid k·c (AS-12, AS-318) and at most one priority trigger per cadence
                    interval when the marginal energy jumps (AS-12, AS-416)

At a trigger τ (same view as training, AS-404/AS-405, AS-220 … AS-223)
-------------------------------------------------------------
    window_τ   = the window holding τ (positions ≤ τ), entity table extended with carried neighbours,
                 contacts cumulative; its Environment assembled from the stored step outputs (keys
                 re-based from the store origin to the window origin by one rotation)
    memory     = TAAFT's long-term memory keys read from the state before τ's own write (AS-222)
    past       = this window's tokens' Imagination of the kept earlier triggers, read from the
                 ImaginationStore (`past_from_store`, AS-223) with each trigger's ŷ kept by the engine
    analysis   = TAAFT(window_τ, R, S, memory, past)
    forecast   = Forecaster.imagine(K, N) → P_inf calibrated by the Verifier's temperature in force
                 (identity until a human applies a proposal, D-21)
    explain    = Expected Gradients over field states + energy-lens shares (inference/explain.py)
    record     = ComputeRecord(K, N, S, R_TSTCT, R_TAAFT, wall time) (D-44)
    write      = then long-term memory (once per trigger, D-36), Imagination store (Forecaster role, D-35)

Decisions: D-02 (held → AS-12), D-21, D-33, D-35, D-36, D-44, D-49, D-51. Assumptions: AS-05, AS-12,
AS-220 … AS-223, AS-317, AS-401 … AS-405, AS-416 … AS-419.

Statistical physics (D-56): the engine queues every ingested state update for a `StatPhysTracker`
(statphys/trajectory.py), which keeps the traffic histograms and the activity multiplex of the state
window, and attaches to every TriggerResult a `StatPhysReading`: the Gibbs ensembles of the entity
tokens, the adversary slots and the imagined routes at the Verifier temperature in force (AS-761),
per-entity readings, traffic and graph entropies, growth rates and the streaming early-warning
indicators with their alarm (cadence triggers; calibrated thresholds only from
`StatPhysConfig.calibration_path`). `EngineSettings.statphys` configures it (enabled by default).

Precision (D-54): the model's outputs arrive as float64 tensors (P_inf, band, median, stage, hazard,
route weights, readouts, energies, lens shares); the calibrated P_inf is float64 (`apply_temperature`);
the contracts are filled with Python floats (IEEE double) straight from those tensors — no `.float()`
narrowing on the way (tests/test_precision_outputs.py checks the bundle against the tensors exactly).
The Environment and Imagination K/V caches the engine keeps are stored in fp32 (`CACHE_DTYPE`, D-54).

Invariants (tests/test_integration_e2e.py): every ForecastBundle validates (P_inf non-decreasing,
stage rows sum to 1); no trigger reads an update after its time; a priority trigger fires at most once
per cadence interval; the engine refuses to run outside the inference modes.

Known limits (reported): TSTCT.step's key gathering is per state in Python (its own extension
point), so replay speed is bounded by it; the open window's graph is rebuilt per processed chunk
(AS-418); TAAFT's cause lens reads only the current window's gates (AS-403). Each trigger's ŷ is kept
for the triggers the Imagination store keeps (M_im), so memory is bounded.
"""

from __future__ import annotations

import math
import time as _time
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import torch

from nagahana.core.errors import InvariantViolation
from nagahana.core.modes import RunMode, require_mode
from nagahana.core.roles import Role
from nagahana.data.collate import collate_items
from nagahana.data.windows import WindowItem, build_window
from nagahana.datamodel.columnar import Column, ColumnarUpdates, to_columnar
from nagahana.datamodel.records import StateUpdate
from nagahana.governance.assumptions import assume
from nagahana.graph.planes import declared_update_planes
from nagahana.inference.buffer import EventLog
from nagahana.inference.explain import Explanation, carry_before, explain_trigger, lens_features
from nagahana.memory.access import Region
from nagahana.memory.environment import EnvironmentStore
from nagahana.memory.imagination import ImaginationStore
from nagahana.memory.kvcache import CACHE_DTYPE, KVCacheMeta
from nagahana.models.advisor.model import slice_trigger
from nagahana.models.batch import AnalysisOut, EnvironmentOut, ForecastOut, WindowBatch
from nagahana.models.forecaster.routes import route_cumulative
from nagahana.models.nagahana import NagaHana, model_hash
from nagahana.models.taaft.imagination import PastImagination, past_from_store
from nagahana.models.taaft.objectives import total_energy
from nagahana.models.tstct.model import StepContext
from nagahana.models.verifier.calibration import apply_temperature
from nagahana.models.verifier.gate import apply_calibration
from nagahana.models.verifier.heads import forecast_features, monitor_features
from nagahana.models.verifier.monitor import Monitor
from nagahana.models.verifier.reports import CalibrationProposal, TemperatureState
from nagahana.models.vocab import NODE_KIND_CODE, STAGES
from nagahana.nn.attention import rotate_heads
from nagahana.physics.term import PhysicsTerm
from nagahana.roles.contracts import (
    AdvisoryBundle,
    AttackStage,
    BeliefReadout,
    ComputeRecord,
    ForecastBundle,
    HumanCommand,
)
from nagahana.statphys.config import StatPhysConfig
from nagahana.statphys.entropy import update_arrays_from_columnar
from nagahana.statphys.ews import load_calibrations
from nagahana.statphys.graphs import MULTICAST_CODE
from nagahana.statphys.trajectory import StatPhysReading, StatPhysTracker, TriggerInputs
from nagahana.training.carry import ContactLedger, assemble_extension, extend_row, filter_carry

INFER_SCHEMA = "nagahana-infer-v1"
TYPE_NAMES = ("opportunistic", "targeted", "insider")    # TAAFTConfig.n_types = 3 (P-13, assumed)


# ===================================================================================== settings
@dataclass(frozen=True)
class Budgets:
    """Run-time budgets of one forecast (D-44), recorded in every `ComputeRecord`."""

    tstct_passes: int
    taaft_passes: int
    descent_steps: int
    horizon_k: int
    routes_n: int

    @classmethod
    def defaults(cls, model: NagaHana) -> Budgets:
        """The configured run-time defaults (TSTCT/TAAFT default_passes, TAAFT descent_steps, Forecaster K, N)."""
        c = model.cfg
        return cls(c.tstct.default_passes, c.taaft.default_passes, c.taaft.descent_steps, c.forecaster.horizon_k,
                   c.forecaster.routes_n)


@dataclass(frozen=True)
class EngineSettings:
    """Engine knobs (each value an assumption where the design does not fix it)."""

    budgets: Budgets
    cadence_s: float                 # AS-12: fixed cadence (= ForecasterConfig.window_seconds)
    priority_sigma: float            # AS-416: energy jump threshold, in standard deviations of cadence energies
    priority_warmup: int             # AS-416: cadence triggers observed before a priority trigger may fire
    probe_every: int                 # AS-416: processed updates between marginal-energy probes
    attribution_samples: int         # AS-417: Expected-Gradients samples per trigger
    top_features: int                # driving features reported per forecast
    alert_threshold: float           # AS-419: P_inf(K) / compromise level that counts as an alert in reports
    network: str                     # name of the monitored network (provenance of windows)
    statphys: StatPhysConfig = field(default_factory=StatPhysConfig)   # D-56: statistical-physics readings

    @classmethod
    def assumed(cls, model: NagaHana, *, network: str) -> EngineSettings:
        for a in ("AS-12", "AS-416", "AS-417", "AS-419"):
            assume(a, by=__name__)
        return cls(budgets=Budgets.defaults(model), cadence_s=model.cfg.forecaster.window_seconds, priority_sigma=3.0,
                   priority_warmup=5, probe_every=max(1, model.cfg.training.window_updates // 2), attribution_samples=8,
                   top_features=8, alert_threshold=0.5, network=network)


# ===================================================================================== results
@dataclass
class TriggerResult:
    """Everything the engine produced at one trigger."""

    time: float                                   # epoch seconds
    kind: str                                     # "cadence" | "priority"
    forecast: ForecastBundle
    belief: BeliefReadout
    trust: float                                  # Verifier value head: P(this forecast is right)
    energy: float                                 # E_total at ŷ (mean over the trigger's tokens' sum, D-42)
    marginal_energy: float                        # E(∅, ŷ) (novelty reading, P-09)
    compromise: dict[str, float]                  # entity → compromise belief (internal entities)
    p_inf_raw: tuple[float, ...]                  # P_inf before the Verifier's temperature
    explanation: Explanation
    analysis: AnalysisOut                         # [1, 1, …] (the Advisor reads it on demand)
    entity_names: list[str]
    entity_internal: torch.Tensor                 # bool [V']
    entity_kind: torch.Tensor                     # long [V']
    past_trigger_times: tuple[float, ...] = ()    # epoch times of the carried triggers TAAFT read (AS-223)
    memory_writes_before: int = 0                 # long-term writes in the state TAAFT read (= triggers before τ)
    notes: list[str] = field(default_factory=list)
    statphys: StatPhysReading | None = None       # D-56: thermodynamic, entropy and early-warning reading of the trigger


@dataclass
class _Window:
    """A planned window of the stream and the step outputs of its processed positions."""

    rows: list[int] = field(default_factory=list)
    entities: set[int] = field(default_factory=set)
    n_done: int = 0                                                   # positions already stepped
    memory: list[torch.Tensor] = field(default_factory=list)          # [d] per position
    refined: list[torch.Tensor] = field(default_factory=list)
    z: list[torch.Tensor] = field(default_factory=list)               # [dz]
    kv: list[list[tuple[torch.Tensor, torch.Tensor]]] = field(default_factory=list)   # per position: per block (K, V) [H, d_h]
    gate: list[torch.Tensor] = field(default_factory=list)            # [H_c, T_c]
    cause_entity: list[torch.Tensor] = field(default_factory=list)    # [T_c]
    cause_time: list[torch.Tensor] = field(default_factory=list)      # [T_c] seconds since the store origin


# ===================================================================================== the engine
class Engine:
    """Incremental live / replay inference (module docstring).

    model: a trained NagaHana (put in eval mode; never trained here). physics: Φ_phys inside TAAFT's
    E_total (requires the site MTU), or None (the term is then 0 and every result notes it).
    temperatures: the Verifier's temperatures in force (identity by default; changed only by
    `apply_calibration` with a HumanCommand, D-21). seed: Forecaster route sampling and attributions.
    """

    def __init__(self, model: NagaHana, *, settings: EngineSettings, physics: PhysicsTerm | None, seed: int,
                 temperatures: TemperatureState | None = None) -> None:
        require_mode(RunMode.INFER_LIVE, RunMode.FORENSIC_REPLAY, component="inference engine")
        assume("AS-418", by=__name__)
        self.model, self.cfg, self.settings, self.physics = model, model.cfg, settings, physics
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        self.model_hash = model_hash(model)
        self.log = EventLog()
        self.store: EnvironmentStore | None = None
        tc = self.cfg.taaft
        # Both caches are stored in fp32 (D-54 follow-up): the Environment through `TSTCT.new_store`.
        self.imagination = ImaginationStore(self.cfg.memory, KVCacheMeta(
            Region.IMAGINATION, "taaft", self.model_hash, self.cfg.latent_space, INFER_SCHEMA, tc.blocks, tc.heads,
            tc.dim // tc.heads), dtype=CACHE_DTYPE)
        self.ledger = ContactLedger(len(self.cfg.graph.planes))
        self.last_seen: dict[int, float] = {}                          # entity row → epoch time of its latest state
        self.kind: dict[int, int] = {}
        self.internal: dict[int, bool] = {}
        self.longterm = model.longterm_init(1)
        self.y_by_trigger: dict[float, dict[int, torch.Tensor]] = {}     # ŷ per kept trigger and token id (AS-223)
        self.monitor = Monitor(self.cfg.verifier, region_dims={"environment": self.cfg.latent_dim,
                                                               "imagination": self.cfg.taaft.d_hyp})
        self.temperatures = temperatures if temperatures is not None else TemperatureState()
        self.audit: list[HumanCommand] = []
        self.open = _Window()
        self.last_closed: _Window | None = None
        self.next_grid: float | None = None
        self.results: list[TriggerResult] = []
        self.cadence_energy: list[float] = []
        self.priority_intervals: set[int] = set()
        self.since_probe = 0
        self.last_trigger_time = -math.inf
        self.last_processed_time = -math.inf
        self.gen = torch.Generator().manual_seed(int(seed))
        self.notes: list[str] = []
        if physics is None:
            self.notes.append("physics term off in TAAFT's energy (no site MTU configured)")
        # Statistical-physics readings (D-56): traffic and multiplex of the state window, Gibbs ensembles and
        # early warning. Alarm calibrations come only from a file (`statphys calibrate`); none is defaulted.
        sp = settings.statphys
        self.statphys: StatPhysTracker | None = None
        self._statphys_group = np.zeros(0, dtype=bool)                    # entity row -> multicast group (D-47)
        if sp.enabled:
            cals = load_calibrations(sp.calibration_path) if sp.calibration_path else None
            self.statphys = StatPhysTracker(sp, planes=tuple(self.cfg.graph.planes), calibrations=cals)

    # ------------------------------------------------------------------ ingestion
    def ingest(self, chunk: ColumnarUpdates) -> list[TriggerResult]:
        """Merge a chunk into the event log and process it in time order. Returns the triggers fired."""
        require_mode(RunMode.INFER_LIVE, RunMode.FORENSIC_REPLAY, component="inference engine")
        rows = self.log.append(chunk)
        if not len(rows):
            return []
        if self.statphys is not None:
            self._statphys_observe(rows)          # queued; each reading at tau commits only updates <= tau
        assert self.log.cu is not None
        times = self.log.cu.updates["event_time"].to_numpy(dtype=np.float64)[rows]
        ents = np.stack([self.log.cu.updates[f"entity_{k}"].to_numpy(dtype=np.int64)[rows] if f"entity_{k}" in self.log.cu.updates
                         else np.full(len(rows), -1, dtype=np.int64) for k in range(3)], axis=1)
        out: list[TriggerResult] = []
        c = self.settings.cadence_s
        if self.next_grid is None:
            self.next_grid = math.ceil(float(times[0]) / c) * c                  # first grid point ≥ the first update (AS-318)
        i = 0
        with torch.no_grad():
            while i < len(rows):
                out += self._fire_due(before=float(times[i]))
                # take rows up to the next cadence point, the window's capacity and the probe interval
                j = i
                while j < len(rows) and float(times[j]) <= self.next_grid and self.since_probe + (j - i) < self.settings.probe_every:
                    new = {int(e) for e in ents[j] if e >= 0}
                    if self._overflows(new):
                        break
                    self.open.rows.append(int(rows[j]))
                    self.open.entities |= new
                    j += 1
                if j == i:
                    if self._overflows({int(e) for e in ents[i] if e >= 0}):
                        self._close()                                            # the next row opens a new window
                        continue
                    # the probe interval is reached: probe at the last processed update, then continue
                    out += self._probe(self.last_processed_time)
                    continue
                self._process_open()
                self.since_probe += j - i
                self.last_processed_time = float(times[j - 1])
                i = j
                if self.since_probe >= self.settings.probe_every:
                    out += self._probe(float(times[j - 1]))
        return out

    def ingest_states(self, states: Sequence[StateUpdate], columns: Sequence[Column]) -> list[TriggerResult]:
        """State-update objects (any adapter) → the columnar form (`datamodel.columnar.to_columnar`) → `ingest`."""
        return self.ingest(to_columnar(states, columns))

    def advance_clock(self, now: float) -> list[TriggerResult]:
        """Live use: every update up to `now` has arrived; fire the cadence triggers < now."""
        with torch.no_grad():
            return self._fire_due(before=now)

    def finish(self) -> list[TriggerResult]:
        """End of a capture: fire the cadence triggers up to the last update's time (inclusive, as training does)."""
        if self.next_grid is None:
            return []
        with torch.no_grad():
            return self._fire_due(before=math.nextafter(self.log.last_time, math.inf))

    # ------------------------------------------------------------------ windows
    def _overflows(self, new: set[int]) -> bool:
        """The training window rule (AS-317): would adding an update with these entities overflow the open window?"""
        t = self.cfg.training
        return bool(self.open.rows) and (len(self.open.rows) >= t.window_updates
                                         or len(self.open.entities | new) > t.max_entities)

    def _close(self) -> None:
        """Close the open window (all its rows are processed) and start a new one."""
        if self.open.rows:
            self.last_closed = self.open
        self.open = _Window()

    def _build(self, w: _Window, *, trigger: float | None) -> tuple[WindowItem, WindowBatch]:
        """The window's item (as training builds it) and its one-row batch; `trigger` sets one trigger at τ."""
        src = self.log.window_source(np.asarray(w.rows, dtype=np.int64), network=self.settings.network)
        item = build_window(src, 0, len(w.rows), self.cfg)
        if trigger is not None:
            tau = trigger - item.origin
            latest = np.full((1, len(item.entity_rows)), -1, dtype=np.int64)
            for p in range(len(item.pos_entity)):
                if item.pos_time[p] <= tau:
                    latest[0, item.pos_entity[p]] = p
            item.trigger_time = np.array([tau])
            item.entity_latest = latest
            item.entity_malicious_share = np.full((len(item.entity_rows), 1), np.nan, dtype=np.float32)
        window, _ = collate_items([item], self.cfg)
        return item, window

    def _ensure_store(self, origin: float) -> EnvironmentStore:
        if self.store is None:
            self.store = self.model.tstct.new_store(self.cfg.memory, model_hash=self.model_hash,
                                                    latent_space=self.cfg.latent_space, schema_version=INFER_SCHEMA,
                                                    origin=origin)
        return self.store

    def _process_open(self) -> None:
        """Rebuild the open window, encode its new positions and step them into the Environment store."""
        w = self.open
        item, window = self._build(w, trigger=None)
        store = self._ensure_store(item.origin)
        # contacts of the open window into the ledger (first contact is monotone: merging again is harmless)
        keys = item.entity_rows
        self.ledger.update(keys, window.contact1[0], window.contact_planes[0], item.origin)
        for i, k in enumerate(keys.tolist()):
            self.kind[int(k)] = int(item.entity_kind[i])
            self.internal[int(k)] = bool(item.entity_internal[i])
        p_real = len(item.pos_entity)
        new = list(range(w.n_done, p_real))
        if not new:
            return
        u, _ = self.model.encode_updates(window.fields, window)
        z, _, _, _ = self.model.latents(window, u, sample=False)
        ent = torch.as_tensor(keys[item.pos_entity[new]], dtype=torch.long)
        t_epoch = item.pos_time[new] + item.origin
        t_rel = torch.as_tensor(t_epoch - store.origin, dtype=torch.float64)
        ctx = self._step_context(ent.tolist(), t_epoch.tolist())
        so = self.model.tstct.step(z[0, new], ent, t_rel, store, ctx, passes=self.settings.budgets.tstct_passes)
        for k in range(len(new)):
            w.memory.append(so.memory[k])
            w.refined.append(so.refined[k])
            w.z.append(z[0, new[k]])
            w.kv.append([(kb[k], vb[k]) for kb, vb in so.kv])
            w.gate.append(so.causal_gate[k])
            w.cause_entity.append(so.causal_entity[k])
            w.cause_time.append(so.causal_time[k])
            self.last_seen[int(ent[k])] = float(t_epoch[k])
        w.n_done = p_real

    def _step_context(self, ents: list[int], times: list[float]) -> StepContext:
        """Neighbours within two hops as of each state's time (AS-405), restricted to entities with states (AS-418)."""
        cfg = self.cfg.tstct
        tie, lag = cfg.time_tie_s, cfg.causal_lag_s
        n_pl = len(self.cfg.graph.planes)
        rows: list[list[tuple[int, int, np.ndarray]]] = []
        for k, (e, t) in enumerate(zip(ents, times, strict=True)):
            earlier = {ents[j]: times[j] for j in range(k)}                       # states earlier in this call
            hop1: dict[int, np.ndarray] = {}
            for w_ in self.ledger.nbrs.get(e, ()):
                vec = self.ledger.get(e, w_)
                if vec is not None and vec[0] <= t + tie:
                    hop1[w_] = vec
            hop2: set[int] = set()
            for w_, v1 in hop1.items():
                for v in self.ledger.nbrs.get(w_, ()):
                    if v == e or v in hop1:
                        continue
                    v2 = self.ledger.get(w_, v)
                    if v2 is not None and max(v1[0], v2[0]) <= t + tie:
                        hop2.add(v)

            def recency(v: int, earlier: dict[int, float] = earlier) -> float:
                return earlier.get(v, self.last_seen.get(v, -math.inf))

            with_state = [(v, 1) for v in hop1 if recency(v) > -math.inf] + [(v, 2) for v in hop2 if recency(v) > -math.inf]
            with_state.sort(key=lambda x: -recency(x[0]))
            spatial = with_state[: cfg.spatial_keys]
            causal = [(v, 1) for v in hop1 if recency(v) >= t - lag - tie and (v, 1) not in spatial]
            chosen = spatial + causal
            rows.append([(v, h, (hop1[v][1:] <= t + tie) if h == 1 else np.zeros(n_pl, dtype=bool)) for v, h in chosen])
        width = max([1, *(len(r) for r in rows)])
        ne = torch.full((len(ents), width), -1, dtype=torch.long)
        nh = torch.zeros((len(ents), width), dtype=torch.long)
        npl = torch.zeros((len(ents), width, n_pl), dtype=torch.bool)
        for k, r in enumerate(rows):
            for c, (v, h, pl) in enumerate(r):
                ne[k, c], nh[k, c] = v, h
                npl[k, c] = torch.from_numpy(np.asarray(pl, dtype=bool))
        horizon = torch.full((len(ents),), float(self.settings.cadence_s), dtype=torch.float64)
        return StepContext(neighbour_entity=ne, neighbour_hop=nh, neighbour_planes=npl, horizon_dt=horizon)

    # ------------------------------------------------------------------ the Environment view of a window
    def _environment(self, w: _Window, window: WindowBatch, item: WindowItem) -> EnvironmentOut:
        """EnvironmentOut of the window's positions from the stored step outputs (keys re-based to the window origin)."""
        assert self.store is not None
        p_real = len(item.pos_entity)
        if p_real > w.n_done:
            raise InvariantViolation("a trigger needs every position of its window processed")
        p_n = window.positions.entity.shape[1]
        tst = self.model.tstct
        d, h, dh, hc = tst.dim, tst.heads, tst.head_dim, self.cfg.tstct.causal_heads
        memory = torch.zeros(1, p_n, d)
        refined = torch.zeros(1, p_n, d)
        if p_real:
            memory[0, :p_real] = torch.stack(w.memory[:p_real])
            refined[0, :p_real] = torch.stack(w.refined[:p_real])
        shift = torch.tensor([[self.store.origin - item.origin]], dtype=torch.float64).expand(1, p_n)
        cos, sin = tst.rotary.angles(shift)                                         # re-base store → window origin
        kv = []
        for blk in range(self.cfg.tstct.blocks):
            k = torch.zeros(1, h, p_n, dh)
            v = torch.zeros(1, h, p_n, dh)
            if p_real:
                k[0, :, :p_real] = torch.stack([w.kv[p][blk][0] for p in range(p_real)], dim=1)
                v[0, :, :p_real] = torch.stack([w.kv[p][blk][1] for p in range(p_real)], dim=1)
            kv.append((rotate_heads(k, cos, sin, tst.rot_heads), v))
        # causal gates: candidates that are positions of this window (others lie in earlier windows, AS-403)
        gate = torch.zeros(1, hc, p_n, p_n)
        keys = item.entity_rows
        pos_of: dict[int, list[tuple[float, int]]] = {}
        off = item.origin - self.store.origin
        for p in range(p_real):
            pos_of.setdefault(int(keys[item.pos_entity[p]]), []).append((float(item.pos_time[p]) + off, p))
        for i in range(p_real):
            g, ce, ct = w.gate[i], w.cause_entity[i], w.cause_time[i]
            for c in range(ce.shape[0]):
                e = int(ce[c])
                if e < 0:
                    continue
                for t_j, j in pos_of.get(e, ()):
                    if abs(t_j - float(ct[c])) <= 1e-6 and j < i:
                        gate[0, :, i, j] = g[:, c]
                        break
        prior = tst.transition_prior(memory, window.positions.next_dt)
        return EnvironmentOut(memory=memory, kv=kv, refined=refined, prior=prior, causal_gate=gate,
                              passes=self.settings.budgets.tstct_passes)

    def _trigger_view(self, w: _Window, tau: float) -> tuple[WindowItem, WindowBatch, torch.Tensor, EnvironmentOut]:
        """The extended one-trigger window at τ (as training builds it, AS-404/AS-405) and its Environment."""
        item, window = self._build(w, trigger=tau)
        env = self._environment(w, window, item)
        assert self.store is not None
        carried = set(self.store.entities())
        row = extend_row(window, 0, [int(k) for k in item.entity_rows], carried, self.ledger, self.last_seen,
                         cap=self.cfg.training.max_entities)
        ext, _, keys = assemble_extension(window, None, [row], kinds=[self.kind], internals=[self.internal])
        return item, ext, keys, env

    # ------------------------------------------------------------------ triggers
    def _fire_due(self, *, before: float) -> list[TriggerResult]:
        out: list[TriggerResult] = []
        assert self.next_grid is not None
        while self.next_grid < before:
            res = self._fire(self.next_grid, "cadence")
            if res is not None:
                out.append(res)
            self.next_grid += self.settings.cadence_s
        return out

    def _probe(self, t: float) -> list[TriggerResult]:
        """Marginal-energy probe at t (AS-416): fire a priority trigger on a jump, at most once per cadence interval."""
        self.since_probe = 0
        w = self.open if self.open.rows else self.last_closed
        n = len(self.cadence_energy)
        interval = math.floor(t / self.settings.cadence_s)
        if w is None or n < self.settings.priority_warmup or interval in self.priority_intervals or t <= self.last_trigger_time:
            return []
        item, ext, keys, env = self._trigger_view(w, t)
        b = self.settings.budgets
        past = self._past(t, keys, item.origin)
        an = self.model.analyse(env, ext, passes=b.taaft_passes, descent_steps=b.descent_steps, longterm=self.longterm,
                                past=past, physics=self.physics)
        e = self._marginal(an, ext)
        mean = float(np.mean(self.cadence_energy))
        std = float(np.std(self.cadence_energy))
        if std > 0 and e - mean > self.settings.priority_sigma * std:
            self.priority_intervals.add(interval)
            res = self._fire(t, "priority")
            return [res] if res is not None else []
        return []

    def _marginal(self, an: AnalysisOut, window: WindowBatch) -> float:
        """Mean E(∅, ŷ) over the trigger's active entity tokens (P-09 reading)."""
        v = window.entity_mask.shape[1]
        e, _ = self.model.taaft.marginal_energy(an.y, an.token_mask, n_entities=v)
        return float(e.mean())

    def _fire(self, tau: float, kind: str) -> TriggerResult | None:
        """Run the analysis, forecast, explanation and Verifier readings at τ (module docstring)."""
        w = self.open if self.open.rows else self.last_closed
        if w is None or tau <= self.last_trigger_time:
            return None
        t0 = _time.perf_counter()
        b = self.settings.budgets
        item, ext, keys, env = self._trigger_view(w, tau)
        # read before write (AS-222): the memory TAAFT reads holds only earlier triggers' writes
        lt_before = self.longterm
        past = self._past(tau, keys, item.origin)
        an = self.model.analyse(env, ext, passes=b.taaft_passes, descent_steps=b.descent_steps, longterm=lt_before,
                                past=past, physics=self.physics)
        fo = self.model.forecast(an, horizon_k=b.horizon_k, routes_n=b.routes_n, generator=self.gen)
        p_raw = tuple(float(x) for x in fo.p_inf[0, 0])
        fo = self._calibrated(fo)
        # explanation: EG over field states (dense recomputation with the carry as of the window start)
        assert self.store is not None
        carry = None
        if self.store.entities():
            full = self.model.tstct.export_carry([self.store])
            t_first = torch.tensor([item.origin - self.store.origin], dtype=torch.float64)
            carry = filter_carry(carry_before(full, t_first), keys).align(keys)
        expl = explain_trigger(self.model, ext, carry, forecast=fo, longterm=lt_before, past=past, physics=self.physics,
                               tstct_passes=b.tstct_passes, taaft_passes=b.taaft_passes, descent_steps=b.descent_steps,
                               samples=self.settings.attribution_samples, top=self.settings.top_features,
                               generator=self.gen)
        names = self._names(keys)
        feats = [*expl.features, *lens_features(an.lens_share)]
        wall = _time.perf_counter() - t0
        compute = ComputeRecord(samples_n=b.routes_n, horizon_k=b.horizon_k, refinement_steps=b.descent_steps,
                                wall_time_s=wall, tstct_passes=b.tstct_passes, taaft_passes=b.taaft_passes)
        v_n = ext.entity_mask.shape[1]
        bundle = self.model.forecaster.to_bundle(fo, 0, 0, compute=compute, stage_names=[s for s, _ in STAGES],
                                                 driving_features=feats, entity_names=names[:v_n],
                                                 max_paths=min(b.routes_n, 10))
        belief = self._belief(an, ext, names)
        # Verifier readings (no weight change): trust value head and Monitor statistics
        ff = forecast_features(fo)
        mf = monitor_features(self.monitor.report(), cusum_h=self.cfg.verifier.cusum_h, ph_lambda=self.cfg.verifier.ph_lambda)
        # The trust head's float32 logit → float64 before σ (D-54: a reported probability, no float32 link).
        trust = float(torch.sigmoid(self.model.verifier.trust(ff, mf).double())[0, 0])
        p_real = len(item.pos_entity)
        if p_real:
            self.monitor.observe_latents("environment", torch.stack(w.z[:p_real]))
        self.monitor.observe_latents("imagination", an.y[an.token_mask])
        # writes: long-term memory (once per trigger, D-36) and Imagination (Forecaster role, D-35)
        x, mask = self.model.longterm_inputs(env, ext, 0)
        self.longterm = self.model.longterm_write(self.longterm, x, mask)
        self._write_imagination(tau, an, keys, bundle)
        energy = float(total_energy(an)[0, 0])
        marginal = self._marginal(an, ext)
        reading = self._statphys_reading(tau, kind, an, fo, ext, keys, names, energy, marginal)
        if kind == "cadence":
            self.cadence_energy.append(marginal)
        comp = an.readouts["compromise"][0, 0]
        active = an.token_mask[0, 0, :v_n]
        compromise = {names[v]: float(comp[v]) for v in range(v_n) if bool(active[v]) and bool(ext.entity_internal[0, v])}
        res = TriggerResult(time=tau, kind=kind, forecast=bundle, belief=belief, trust=trust, energy=energy,
                            marginal_energy=marginal, compromise=compromise, p_inf_raw=p_raw, explanation=expl,
                            analysis=slice_trigger(an, 0, 0), entity_names=names[:v_n],
                            entity_internal=ext.entity_internal[0].clone(), entity_kind=ext.entity_kind[0].clone(),
                            past_trigger_times=self._carried_times(past, item.origin),
                            memory_writes_before=lt_before.trigger, notes=list(self.notes), statphys=reading)
        self.results.append(res)
        self.last_trigger_time = tau
        return res

    def _calibrated(self, fo: ForecastOut) -> ForecastOut:
        """P_inf, its band and median under the temperature in force (σ(logit p / T) is monotone: order kept)."""
        t = float(self.temperatures.temperatures.get("p_inf", 1.0))
        if t == 1.0:
            return fo
        p = torch.cummax(apply_temperature(fo.p_inf, t), dim=-1).values
        band = apply_temperature(fo.p_inf_band, t)
        med = apply_temperature(fo.p_inf_median, t)
        return ForecastOut(p_inf=p, p_inf_band=band, p_inf_median=med, stage=fo.stage, hazard=fo.hazard,
                           route_weight=fo.route_weight, route_actions=fo.route_actions, route_distinct=fo.route_distinct,
                           mode_route=fo.mode_route, step_state=fo.step_state, step_value=fo.step_value,
                           step_reward=fo.step_reward, step_latent=fo.step_latent, horizon_k=fo.horizon_k, routes_n=fo.routes_n)

    def _names(self, keys: torch.Tensor) -> list[str]:
        """Readable entity names (the entity key, e.g. an address) for the extended table's stable keys."""
        assert self.log.cu is not None
        ents = self.log.cu.entities
        return [str(ents["key"].iloc[int(k)]) if int(k) >= 0 else "-" for k in keys[0].tolist()]

    def _belief(self, an: AnalysisOut, window: WindowBatch, names: list[str]) -> BeliefReadout:
        """BeliefReadout of trigger (0, 0) (AS-419: std is the Bernoulli spread √(p(1−p)) of the belief)."""
        ro = an.readouts
        v_n = window.entity_mask.shape[1]
        active = an.token_mask[0, 0, :v_n]
        comp = ro["compromise"][0, 0].double()
        stage = ro["stage"][0, 0].double()                                        # [V, S]
        trust = ro["trust"][0, 0].double()
        ent = {names[v]: (float(comp[v]), float(math.sqrt(max(0.0, float(comp[v] * (1 - comp[v]))))))
               for v in range(v_n) if bool(active[v])}
        w = comp * active.double()
        if float(w.sum()) > 0:
            mix = (w[:, None] * stage).sum(0) / w.sum()
        else:
            mix = torch.zeros(stage.shape[-1], dtype=torch.float64)
            mix[0] = 1.0
        mix = mix / mix.sum()
        stage_post = {AttackStage(s): float(mix[i]) for i, (s, _) in enumerate(STAGES)}
        goal = ro["goal"][0, 0].double()
        goal = goal / goal.sum()
        typ = ro["type"][0, 0].double()
        typ = typ / typ.sum()
        tnames = TYPE_NAMES if typ.shape[0] == len(TYPE_NAMES) else tuple(f"type_{i}" for i in range(typ.shape[0]))
        return BeliefReadout(
            entity_compromise=ent, suspicion_floor=float(self.cfg.taaft.suspicion_floor), stage_posterior=stage_post,
            goal_posterior={f"goal_{i}": float(goal[i]) for i in range(goal.shape[0])},
            telemetry_trust={names[v]: float(trust[v]) for v in range(v_n) if bool(active[v])},
            type_posterior={tnames[i]: float(typ[i]) for i in range(typ.shape[0])},
        )

    def _token_ids(self, keys: torch.Tensor) -> torch.Tensor:
        """Store token ids of a call's tokens: entity keys, adversary slots −1 … −G, padding −(G+1) − v (never stored)."""
        g = self.cfg.taaft.adversary_slots
        k = keys[0]
        pad = -(g + 1) - torch.arange(k.shape[0])
        return torch.cat([torch.where(k >= 0, k, pad), -1 - torch.arange(g)])

    def _past(self, tau: float, keys: torch.Tensor, origin: float) -> PastImagination | None:
        """This call's tokens' Imagination of the kept triggers before τ (AS-223), with their ŷ; None if none."""
        times = self.imagination.trigger_times
        if not times:
            return None
        ids = self._token_ids(keys)
        read = self.imagination.read(ids, before_time=tau, role=Role.FORECASTER)
        y = torch.zeros(ids.shape[0], len(times), self.cfg.taaft.d_hyp)
        for j, t in enumerate(times):
            ymap = self.y_by_trigger.get(t, {})
            for i, tok in enumerate(ids.tolist()):
                if tok in ymap:
                    y[i, j] = ymap[tok]
        return past_from_store(read, times, origin=origin, y=y)

    @staticmethod
    def _carried_times(past: PastImagination | None, origin: float) -> tuple[float, ...]:
        """Epoch times of the carried triggers that some token actually read."""
        if past is None:
            return ()
        used = past.mask[0].any(dim=0)                                            # [M_c]
        return tuple(float(t) + origin for t, u in zip(past.time[0].tolist(), used.tolist(), strict=True) if u)

    def _write_imagination(self, tau: float, an: AnalysisOut, keys: torch.Tensor, bundle: ForecastBundle) -> None:
        """Memory-stream K/V of the trigger's active tokens into the Imagination store (role Forecaster, D-35),
        and their ŷ for the temporal lens of later triggers (kept as long as the store keeps the trigger)."""
        mask = an.token_mask[0, 0]                                                # [V' + G]
        ids = self._token_ids(keys)
        sel = torch.nonzero(mask).flatten()
        kv = [(k[0].permute(1, 0, 2)[sel], v[0].permute(1, 0, 2)[sel]) for k, v in an.imagination_kv]
        self.imagination.write(tau, ids[sel], kv, role=Role.FORECASTER, forecast=bundle)
        self.y_by_trigger[float(tau)] = {int(ids[i]): an.y[0, 0, i].detach().clone() for i in sel.tolist()}
        kept = set(self.imagination.trigger_times)
        self.y_by_trigger = {t: v for t, v in self.y_by_trigger.items() if t in kept}

    def _statphys_observe(self, rows: np.ndarray) -> None:
        """Queue state updates for the statistical-physics tracker (D-56): traffic fields, members, planes.

        Members are the log's stable entity rows; a member is a group when its kind is multicast (D-47);
        planes follow the training rule (AS-01, `declared_update_planes`) on the contributing destination
        port and protocol (absence stays unknown, D-41). The tracker commits an update to a reading at tau
        only when its time is <= tau, so queuing a whole chunk cannot leak a later update into a trigger.
        """
        assert self.log.cu is not None and self.statphys is not None
        cu = self.log.cu
        upd = cu.updates
        times = upd["event_time"].to_numpy(dtype=np.float64)[rows]
        members = np.stack([upd[f"entity_{k}"].to_numpy(dtype=np.int64)[rows] if f"entity_{k}" in upd
                            else np.full(len(rows), -1, dtype=np.int64) for k in range(3)], axis=1)
        traffic = update_arrays_from_columnar(cu.values[rows], cu.status[rows], [c.name for c in cu.columns],
                                              times, members)
        kinds = cu.entities["kind"].astype(str).to_numpy()
        if self._statphys_group.shape[0] < kinds.shape[0]:                 # entity rows are append-only
            new = kinds[self._statphys_group.shape[0]:]
            self._statphys_group = np.concatenate(
                [self._statphys_group, np.array([NODE_KIND_CODE[k] == MULTICAST_CODE for k in new], dtype=bool)])
        group = np.where(members >= 0, self._statphys_group[np.maximum(members, 0)], False)
        unknown = np.full(len(rows), np.nan, dtype=np.float64)
        planes = declared_update_planes(tuple(self.cfg.graph.planes),
                                        dst_port=traffic.columns.get("flow.dst_port", unknown),
                                        protocol=traffic.columns.get("flow.protocol", unknown),
                                        has_service=members[:, 2] >= 0)
        self.statphys.observe(traffic, multicast=group, planes=planes)

    def _gibbs_temperature(self) -> float:
        """T of the Gibbs readings (AS-761): the Verifier temperature in force for the configured output
        family (changed only under a HumanCommand, D-21), or the configured fixed temperature."""
        g = self.settings.statphys.gibbs
        if g.verifier_family is None:
            return float(g.temperature)
        return float(self.temperatures.temperatures.get(g.verifier_family, g.temperature))

    def _statphys_reading(self, tau: float, kind: str, an: AnalysisOut, fo: ForecastOut, window: WindowBatch,
                          keys: torch.Tensor, names: list[str], energy: float, marginal: float) -> StatPhysReading | None:
        """The statistical-physics reading of trigger (0, 0) (statphys/trajectory.py; None when disabled).

        Imagined steps: the Forecaster's step heads re-applied to the stored imagined states give each step's
        imagined hypothesis (the computation imagination ran), whose marginal energy E(null, y) is the step
        energy (AS-779); route weights, counts, targets and hazards come from the forecast.
        """
        if self.statphys is None:
            return None
        v_n = window.entity_mask.shape[1]
        hyp = self.model.forecaster.step_heads(fo.step_state[0, 0])["hyp"]                 # [N, K, d_y]
        inputs = TriggerInputs(
            time=float(tau), regular=kind == "cadence", token_energy=an.readouts["token_energy"][0, 0],
            token_mask=an.token_mask[0, 0], n_entities=v_n, entity_keys=[int(k) for k in keys[0, :v_n].tolist()],
            entity_names=names[:v_n], total_energy=energy, marginal_energy=marginal,
            lens_energy={nm: float(e[0, 0]) for nm, e in an.lens_energy.items()},
            stage_logits=self.model.taaft.readouts.stage_logits(an.y[0, 0, :v_n]),
            route_energy=self.model.exposure()(hyp).to(torch.float64), route_weight=fo.route_weight[0, 0],
            route_count=int(fo.routes_n), route_target=fo.route_actions[0, 0, :, :, 1],
            route_p_inf=route_cumulative(fo.hazard[0, 0].double())[:, -1])
        return self.statphys.read(inputs, temperature=self._gibbs_temperature())

    # ------------------------------------------------------------------ on demand
    def advise(self, result: TriggerResult | None = None, *, generator_seed: int = 0) -> tuple[AdvisoryBundle, dict[str, object]]:
        """The Advisor at a trigger (default: the latest); advisory only (D-33)."""
        res = result if result is not None else (self.results[-1] if self.results else None)
        if res is None:
            raise InvariantViolation("no trigger has been analysed yet")
        a = self.cfg.advisor
        with torch.no_grad():
            return self.model.advisor.advise(res.analysis, self.model.forecaster, beam_width=a.beam_width,
                                             max_steps=a.max_steps, rollouts=a.rollouts,
                                             generator=torch.Generator().manual_seed(generator_seed), b=0, m=0,
                                             horizon_k=self.settings.budgets.horizon_k, exposure=self.model.exposure(),
                                             entity_kind=res.entity_kind, entity_internal=res.entity_internal,
                                             entity_names=res.entity_names)

    def apply_calibration(self, proposal: CalibrationProposal, command: HumanCommand | None) -> TemperatureState:
        """Apply a Verifier calibration proposal — only under a HumanCommand("apply-calibration") (D-21)."""
        self.temperatures = apply_calibration(proposal, command, state=self.temperatures, audit=self.audit)
        return self.temperatures


__all__ = ["INFER_SCHEMA", "Budgets", "Engine", "EngineSettings", "TriggerResult"]
