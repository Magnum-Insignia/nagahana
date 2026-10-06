"""The thermodynamic trajectory of the network state: one reading per trigger, online and in batch (D-56).

Purpose
-------
This module assembles the statistical-physics view of NagaHana at every trigger tau from the model's
outputs and the traffic of the state window:

    energy        E_total at the refined hypotheses (D-42) and E(null, y_hat) (novelty, P-09)
    ensembles     Gibbs readouts at temperature T (thermo.py) of
                    entities   the active entity tokens, energies = TAAFT's per-token energies
                    slots      the active adversary-hypothesis slots, same energies
                    routes     the N imagined routes of the Forecaster (AS-779)
    entities      per entity: energy, Boltzmann occupation, energy growth, the stage-head ensemble
                  (energies -logit_s; its free energy -T log sum_s exp(logit_s / T) is the energy-based
                  novelty score of Liu et al., NeurIPS 2020, arXiv:2010.03759), the ensemble of the
                  imagined states that target it, and its traffic entropies
    traffic       network-level Shannon entropies and their divergence from the previous trigger
    graph         von Neumann and spectral entropies of the activity multiplex, relative entropy and
                  reduction of the multiplex, quantum JSD from the previous state (graphs.py)
    growth        growth rates (per second) of every energy and entropy series (AS-763)
    ews           streaming early-warning indicators of the monitored series and the alarm (ews.py)

Route ensemble (AS-779): the Forecaster samples N routes from its MPPI-tilted policy and merges
identical action sequences (route weights w_n = count_n / N, AS-251). The energy of an imagined step
is E(null, y_hat_{n,k}), the marginal energy of the step's imagined hypothesis, the Forecaster's own
exposure reading (AS-17, AS-252); the energy of a route is the sum over its K steps (the action of the
path). Each of the N draws is one microstate, so the base measure of a distinct route is g_n = N w_n
(its multiplicity) and S = log Z + U / T counts the draws (0 <= S <= log N). The observable
O_n = 1 - prod_k (1 - h_{n,k}) (the route's infiltration probability at the horizon) gives the
Gibbs mean <O> (the energy-tilted infiltration probability), its susceptibility Var(O) / T and its
thermal response Cov(O, E) / T^2: a positive response says the infiltrating futures are the
unfamiliar ones. Imagined states of an entity: the steps (n, k) whose target is the entity, with
the multiplicities of their routes.

Temperature (AS-761): the reading's T is the Verifier temperature in force for the configured
output family (changed only by a human command, D-21), or the configured fixed T. TAAFT's own
readouts (models/taaft/readouts.py) are at the native T = 1.

Sampling (AS-772): the early-warning indicators assume a regular sampling grid, so they are updated
by cadence triggers only; a priority trigger gets every other part of the reading.

Precision (D-54): every value is float64 (Python floats are IEEE doubles).
"""

from __future__ import annotations

import math
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import torch

from nagahana.models.batch import AnalysisOut, WindowBatch
from nagahana.statphys.config import StatPhysConfig
from nagahana.statphys.entropy import TrafficEntropyTracker, TrafficReading, UpdateArrays, traffic_entropies
from nagahana.statphys.ews import AlarmCalibration, EWSStep, StreamingEWS
from nagahana.statphys.graphs import (
    GraphReading,
    MultiplexState,
    SlidingMultiplex,
    activity_state,
    contact_state,
    graph_reading,
)
from nagahana.statphys.spectral import Estimate
from nagahana.statphys.thermo import GibbsState, SlopeTracker, gibbs, gibbs_grouped, observable_response, trailing_slope

#: Potentials of a Gibbs ensemble, in the order of the series names "<ensemble>.<potential>".
POTENTIALS: tuple[str, ...] = ("log_partition", "free_energy", "mean_energy", "entropy", "heat_capacity", "size")


def grows(name: str) -> bool:
    """Series whose growth rate is read (AS-763): energies, free energies, mean energies and entropies."""
    if name.startswith("energy."):
        return True
    if name.startswith(("graph.vn.", "graph.spectral_entropy.", "traffic.")):
        return True
    return name.endswith((".free_energy", ".mean_energy", ".entropy"))


@dataclass(frozen=True)
class EnsembleSummary:
    """The potentials of one Gibbs ensemble (NaN when it has no member)."""

    temperature: float
    size: int
    log_partition: float
    free_energy: float
    mean_energy: float
    entropy: float
    heat_capacity: float

    @staticmethod
    def of(state: GibbsState, index: tuple[int, ...] = ()) -> EnsembleSummary:
        """Summary of the ensemble at `index` of a (batched) Gibbs state."""
        def f(x: torch.Tensor) -> float:
            return float(x[index]) if bool(state.valid[index]) else math.nan

        return EnsembleSummary(temperature=float(state.temperature[index]), size=int(state.size[index]),
                               log_partition=f(state.log_partition), free_energy=f(state.free_energy),
                               mean_energy=f(state.mean_energy), entropy=f(state.entropy),
                               heat_capacity=f(state.heat_capacity))

    @staticmethod
    def empty(temperature: float) -> EnsembleSummary:
        return EnsembleSummary(temperature, 0, math.nan, math.nan, math.nan, math.nan, math.nan)

    def values(self) -> dict[str, float]:
        """The potentials keyed as in `POTENTIALS`."""
        return {"log_partition": self.log_partition, "free_energy": self.free_energy, "mean_energy": self.mean_energy,
                "entropy": self.entropy, "heat_capacity": self.heat_capacity, "size": float(self.size)}


@dataclass(frozen=True)
class EntityReading:
    """The statistical-physics reading of one active entity at a trigger (module docstring)."""

    key: int
    name: str
    energy: float
    occupation: float
    energy_growth: float
    stage: EnsembleSummary
    imagined: EnsembleSummary
    traffic: dict[str, float]


@dataclass(frozen=True)
class TriggerInputs:
    """What a reading needs from one trigger of the model (engine-agnostic, validated).

    time: epoch seconds; regular: a cadence trigger (updates the early-warning indicators);
    token_energy float64 [N] and token_mask bool [N]: TAAFT's per-token energies at y_hat (V entity tokens,
    then the adversary slots); n_entities V; entity_keys [V] stable entity ids (-1 padding);
    entity_names [V]; stage_logits [V, S] or None; total_energy and marginal_energy: E_total and
    E(null, y_hat) of the trigger; lens_energy: per-lens energies; route_energy [N_r, K] float64
    E(null, y_hat_{n,k}) of the imagined steps, route_weight [N_r] (sum 1, merged duplicates 0),
    route_count N (draws), route_target long [N_r, K] (entity token index or -1), route_p_inf [N_r]
    (each route's infiltration probability at the horizon); all route fields None without a forecast.
    """

    time: float
    regular: bool
    token_energy: torch.Tensor
    token_mask: torch.Tensor
    n_entities: int
    entity_keys: Sequence[int]
    entity_names: Sequence[str]
    total_energy: float
    marginal_energy: float
    lens_energy: Mapping[str, float] = field(default_factory=dict)
    stage_logits: torch.Tensor | None = None
    route_energy: torch.Tensor | None = None
    route_weight: torch.Tensor | None = None
    route_count: int = 0
    route_target: torch.Tensor | None = None
    route_p_inf: torch.Tensor | None = None

    def __post_init__(self) -> None:
        n = self.token_energy.shape[0]
        v = self.n_entities
        if self.token_energy.dim() != 1 or self.token_mask.shape != (n,) or not 0 <= v <= n:
            raise ValueError("token_energy and token_mask must be [N] with 0 <= n_entities <= N")
        if len(self.entity_keys) != v or len(self.entity_names) != v:
            raise ValueError("entity_keys and entity_names need one entry per entity token")
        if self.stage_logits is not None and (self.stage_logits.dim() != 2 or self.stage_logits.shape[0] != v):
            raise ValueError("stage_logits must be [V, n_stages]")
        routes = (self.route_energy, self.route_weight, self.route_target)
        if any(x is not None for x in routes):
            if any(x is None for x in routes):
                raise ValueError("route_energy, route_weight and route_target come together")
            assert self.route_energy is not None and self.route_weight is not None and self.route_target is not None
            nr, k = self.route_energy.shape
            if self.route_weight.shape != (nr,) or self.route_target.shape != (nr, k):
                raise ValueError("route_weight must be [N_r] and route_target [N_r, K]")
            if self.route_p_inf is not None and self.route_p_inf.shape != (nr,):
                raise ValueError("route_p_inf must be [N_r]")
            if self.route_count < 1:
                raise ValueError("route_count must be the number of drawn routes (>= 1)")


@dataclass(frozen=True)
class StatPhysReading:
    """The statistical-physics reading of one trigger (module docstring).

    series: every scalar of the reading by name (the early-warning inputs and the evaluation's
    component arrays, evaluation.py); growth: "growth.<series>" per second; ews: the streaming
    early-warning step of every monitored series (cadence triggers only); alarm: True / False when at
    least one monitored series is calibrated and defined, else None.
    """

    time: float
    regular: bool
    temperature: float
    ensembles: dict[str, EnsembleSummary]
    route_response: dict[str, float]
    entities: tuple[EntityReading, ...]
    traffic: TrafficReading | None
    graph: GraphReading | None
    series: dict[str, float]
    growth: dict[str, float]
    ews: dict[str, EWSStep]
    alarm: bool | None
    notes: tuple[str, ...]


def _route_ensembles(inputs: TriggerInputs, temperature: float
                     ) -> tuple[EnsembleSummary, dict[str, float], GibbsState | None]:
    # The route ensemble, the response of the route infiltration probability, and the per-entity
    # ensembles of imagined states (grouped by target entity).
    assert inputs.route_energy is not None and inputs.route_weight is not None and inputs.route_target is not None
    e = inputs.route_energy.to(torch.float64)                                   # [N_r, K]
    w = inputs.route_weight.to(torch.float64)
    on = w > 0
    log_g = torch.where(on, torch.log(torch.where(on, w * inputs.route_count, torch.ones_like(w))),
                        torch.full_like(w, -math.inf))
    e_route = e.sum(-1)                                                          # path energy (action)
    st = gibbs(e_route, temperature=temperature, log_measure=log_g, check_finite=True)
    resp = {"p_inf_mean": math.nan, "p_inf_susceptibility": math.nan, "p_inf_thermal_response": math.nan}
    if inputs.route_p_inf is not None and bool(st.valid):
        r = observable_response(st, e_route, inputs.route_p_inf)
        resp = {"p_inf_mean": float(r.mean), "p_inf_susceptibility": float(r.susceptibility),
                "p_inf_thermal_response": float(r.thermal_response)}
    v = inputs.n_entities
    k = e.shape[1]
    tgt = inputs.route_target.to(torch.long)
    step_measure = log_g[:, None].expand(-1, k)
    per_entity = gibbs_grouped(e.reshape(-1), tgt.reshape(-1), v, temperature=temperature,
                               log_measure=step_measure.reshape(-1)) if v else None
    return EnsembleSummary.of(st), resp, per_entity


class StatPhysTracker:
    """Online statistical-physics readings over a stream of triggers (module docstring).

    planes: the relation planes (column order of the update planes); calibrations: per-series alarm
    calibrations (ews.load_calibrations), checked against the early-warning settings.
    """

    def __init__(self, config: StatPhysConfig, *, planes: tuple[str, ...],
                 calibrations: Mapping[str, AlarmCalibration] | None = None) -> None:
        self.config = config
        self.planes = planes
        window = config.state_window_seconds
        self.traffic = TrafficEntropyTracker(config.traffic, window_seconds=window)
        self.multiplex = SlidingMultiplex(planes, window_seconds=window, weight=config.spectral.weight)
        cals = dict(calibrations or {})
        for name, cal in cals.items():
            if name not in config.ews_series:
                raise ValueError(f"calibration for {name!r}, which is not a monitored series")
            cal.check(config.ews)
        self.calibrations = cals
        self._ews = {name: StreamingEWS(config.ews, cals.get(name)) for name in config.ews_series}
        self._growth: dict[str, SlopeTracker] = {}
        self._entity_growth: OrderedDict[int, SlopeTracker] = OrderedDict()
        self._previous: MultiplexState | None = None
        self._previous_entropy: dict[str, Estimate] | None = None
        self._last_time = -math.inf

    def observe(self, traffic: UpdateArrays, *, multicast: np.ndarray, planes: np.ndarray) -> None:
        """Queue a time-sorted block of state updates (members as stable entity ids)."""
        self.traffic.observe(traffic)
        self.multiplex.observe(traffic.time, traffic.members, multicast, planes)

    def _entity_slope(self, key: int, t: float, x: float) -> float:
        tr = self._entity_growth.get(key)
        if tr is None:
            tr = SlopeTracker(self.config.gibbs.growth_window)
            self._entity_growth[key] = tr
        self._entity_growth.move_to_end(key)
        while len(self._entity_growth) > self.config.gibbs.growth_entities:
            self._entity_growth.popitem(last=False)
        return tr.update(t, x)

    def read(self, inputs: TriggerInputs, *, temperature: float) -> StatPhysReading:
        """The reading of one trigger at Gibbs temperature `temperature` (times must increase)."""
        if not (math.isfinite(temperature) and temperature > 0):
            raise ValueError("temperature must be finite and > 0")
        if inputs.time <= self._last_time:
            raise ValueError("readings must come in increasing trigger time")
        self._last_time = inputs.time
        t_now = float(inputs.time)
        notes: list[str] = []
        v = inputs.n_entities
        e_tok = inputs.token_energy.detach().to(torch.float64).cpu()
        m_tok = inputs.token_mask.detach().to(torch.bool).cpu()
        ent = gibbs(e_tok[:v], temperature=temperature, mask=m_tok[:v])
        slots = gibbs(e_tok[v:], temperature=temperature, mask=m_tok[v:])
        ensembles = {"entities": EnsembleSummary.of(ent), "slots": EnsembleSummary.of(slots)}
        response = {"p_inf_mean": math.nan, "p_inf_susceptibility": math.nan, "p_inf_thermal_response": math.nan}
        imagined: GibbsState | None = None
        if inputs.route_energy is not None:
            ensembles["routes"], response, imagined = _route_ensembles(inputs, temperature)
        else:
            ensembles["routes"] = EnsembleSummary.empty(temperature)
            notes.append("no imagined routes at this trigger: the route and imagined-state ensembles are empty")
        stage: GibbsState | None = None
        if inputs.stage_logits is not None:
            stage = gibbs(-inputs.stage_logits.detach().to(torch.float64).cpu(), temperature=temperature)
        traffic = self.traffic.read(t_now)
        state = self.multiplex.state(t_now)
        graph = graph_reading(state, config=self.config.spectral, previous=self._previous,
                              previous_entropy=self._previous_entropy)
        self._previous, self._previous_entropy = state, graph.estimates

        # Flat series of the reading (names documented in evaluation.COMPONENTS).
        series: dict[str, float] = {"energy.total": float(inputs.total_energy),
                                    "energy.marginal": float(inputs.marginal_energy)}
        for name, value in inputs.lens_energy.items():
            series[f"energy.lens.{name}"] = float(value)
        for ens, summary in ensembles.items():
            for pot, value in summary.values().items():
                series[f"{ens}.{pot}"] = value
        for key, value in response.items():
            series[f"routes.{key}"] = value
        for name, value in traffic.network.items():
            series[f"traffic.{name}"] = value
            series[f"traffic_samples.{name}"] = traffic.network_samples[name]
            series[f"traffic_jsd.{name}"] = traffic.network_jsd[name]
        series.update(graph_series(graph))
        growth: dict[str, float] = {}
        for name, value in series.items():
            if grows(name):
                tr = self._growth.setdefault(name, SlopeTracker(self.config.gibbs.growth_window))
                growth[f"growth.{name}"] = tr.update(t_now, value)

        # Per-entity readings (active entity tokens with a stable key).
        entities: list[EntityReading] = []
        for i in range(v):
            ent_key = int(inputs.entity_keys[i])
            if not bool(m_tok[i]) or ent_key < 0:
                continue
            energy = float(e_tok[i])
            entities.append(EntityReading(
                key=ent_key, name=str(inputs.entity_names[i]), energy=energy, occupation=float(ent.occupation[i]),
                energy_growth=self._entity_slope(ent_key, t_now, energy),
                stage=EnsembleSummary.of(stage, (i,)) if stage is not None else EnsembleSummary.empty(temperature),
                imagined=EnsembleSummary.of(imagined, (i,)) if imagined is not None else EnsembleSummary.empty(temperature),
                traffic=dict(traffic.entity.get(ent_key, {}))))

        # Early warning on the regular grid of cadence triggers.
        steps: dict[str, EWSStep] = {}
        alarm: bool | None = None
        if inputs.regular:
            for name, tracker in self._ews.items():
                step = tracker.update(series.get(name, math.nan), t_now)
                steps[name] = step
                if step.alarm is not None:
                    alarm = bool(alarm) or step.alarm
            if not self.calibrations:
                notes.append("no alarm calibration loaded: early-warning indicators are reported, no alarm is raised")
        else:
            notes.append("priority trigger: early-warning indicators update on cadence triggers only")
        return StatPhysReading(time=t_now, regular=inputs.regular, temperature=float(temperature), ensembles=ensembles,
                               route_response=response, entities=tuple(entities), traffic=traffic, graph=graph,
                               series=series, growth=growth, ews=steps, alarm=alarm, notes=tuple(notes))


def graph_series(graph: GraphReading) -> dict[str, float]:
    """The scalars of a graph reading as series (names documented in evaluation.COMPONENTS)."""
    out: dict[str, float] = {"graph.nodes": float(graph.nodes)}
    for name, count in graph.edges.items():
        out[f"graph.edges.{name}"] = float(count)
    for name, value in graph.von_neumann.items():
        out[f"graph.vn.{name}"] = value
    for i, st in enumerate(graph.spectral):
        out[f"graph.spectral_entropy.t{i}"] = st.entropy.value
        out[f"graph.spectral_free_energy.t{i}"] = st.free_energy.value
        out[f"graph.spectral_mean.t{i}"] = st.mean.value
        out[f"graph.spectral_heat_capacity.t{i}"] = st.heat_capacity.value
    out["graph.relative_entropy"] = graph.relative_entropy
    out["graph.best_relative_entropy"] = graph.best_relative_entropy
    out["graph.reduced_layers"] = float(graph.reduced_layers) if graph.best_partition else math.nan
    for name, value in graph.jsd_previous.items():
        out[f"graph.jsd.{name}"] = value
    return out


def window_series(an: AnalysisOut, window: WindowBatch, *, config: StatPhysConfig, planes: tuple[str, ...],
                  temperature: float = 1.0) -> dict[str, np.ndarray]:
    """Per-trigger series of a batch of windows: name -> float64 [B, M] (NaN where undefined).

    The batch path of the evaluation (no Forecaster routes): E_total, the entity and slot ensembles at
    `temperature` from TAAFT's per-token energies, the network traffic entropies of the state window,
    the graph reading of each trigger's multiplex (`SpectralConfig.graph_source`) and the growth rate of
    every growing series within each window (times relative to the window origin).
    """
    tm = an.token_mask.detach().cpu()
    b_n, m_n, _ = tm.shape
    v = window.entity_mask.shape[1]
    tok_e = an.readouts["token_energy"].detach().to(torch.float64).cpu()
    valid = window.triggers.mask.detach().cpu()
    out: dict[str, np.ndarray] = {}

    def put(name: str, b: int, m: int, value: float) -> None:
        out.setdefault(name, np.full((b_n, m_n), np.nan, dtype=np.float64))[b, m] = value

    total = torch.stack([e.detach().to(torch.float64) for e in an.lens_energy.values()]).sum(0).cpu()
    ent = gibbs(tok_e[..., :v], temperature=temperature, mask=tm[..., :v])
    slots = gibbs(tok_e[..., v:], temperature=temperature, mask=tm[..., v:])
    tr = traffic_entropies(window, config=config.traffic, window_seconds=config.state_window_seconds)
    previous: list[MultiplexState | None] = [None] * b_n
    previous_entropy: list[dict[str, Estimate] | None] = [None] * b_n
    for b in range(b_n):
        for m in range(m_n):
            if not bool(valid[b, m]):
                continue
            put("energy.total", b, m, float(total[b, m]))
            for ens, st in (("entities", ent), ("slots", slots)):
                for pot, value in EnsembleSummary.of(st, (b, m)).values().items():
                    put(f"{ens}.{pot}", b, m, value)
            for name, values in tr.network.items():
                put(f"traffic.{name}", b, m, float(values[b, m]))
                put(f"traffic_samples.{name}", b, m, float(tr.network_samples[name][b, m]))
                put(f"traffic_jsd.{name}", b, m, float(tr.network_jsd[name][b, m]))
            tau = float(window.triggers.time[b, m])
            if config.spectral.graph_source == "activity":
                state = activity_state(window, b, tau, window_seconds=config.state_window_seconds,
                                       weight=config.spectral.weight, planes=planes)
            else:
                state = contact_state(window, b, m, planes=planes)
            reading = graph_reading(state, config=config.spectral, previous=previous[b], previous_entropy=previous_entropy[b])
            for name, value in graph_series(reading).items():
                put(name, b, m, value)
            previous[b], previous_entropy[b] = state, reading.estimates
    times = window.triggers.time.detach().to(torch.float64).cpu()
    for name in [n for n in out if grows(n)]:
        x = torch.from_numpy(out[name])
        slope, _ = trailing_slope(x, times, valid & torch.isfinite(x), window=config.gibbs.growth_window)
        out[f"growth.{name}"] = slope.numpy()
    return out


__all__ = ["POTENTIALS", "EnsembleSummary", "EntityReading", "StatPhysReading", "StatPhysTracker", "TriggerInputs",
           "graph_series", "grows", "window_series"]
