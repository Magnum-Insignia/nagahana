"""Forensic replay: the live engine run over an uploaded capture, summarised as a `ForensicReport` ([Q-17]).

Purpose
-------
"Two settings of one engine: live forecasting, and forensic replay of an uploaded capture"
(architecture §1). `replay(model, path, …)` reads a capture with the PCAP adapter (D-51 flow-state
updates), runs `Engine` over it in RunMode.FORENSIC_REPLAY, and writes (architecture §6 "Forensic replay"):

- **timeline**: per trigger, what would have been forecast: P_inf(K) with its band, the dominant
  forecast stage, the most suspected entity, the Verifier's trust, and the compute spent;
- **narrative by stage**: each time the dominant forecast stage (route mixture at step 1, "none"
  excluded) changes to a new stage, with the time it first became dominant;
- **patient zero**: internal entities ranked by the earliest trigger at which their compromise belief
  reached the alert level (AS-419), then by their highest belief; the probability is that highest
  belief (a belief, never a fact);
- **counterfactuals**: at the first alerting trigger (else the highest P_inf), the top-ranked entity
  isolated (no inbound targeting, no outbound reads, `ForecastIntervention`) and re-imagined with the
  same random numbers: P_inf(K) without vs with;
- **observability gaps**: catalogue columns never observed in the capture, silences longer than the
  cadence, and the share of updates without packet-level fields;
- **tamper signs**: out-of-order records (ordering uncertainty > 0) and the clock quality — reported as
  facts about the capture, never as proof of tampering.

Decisions: D-21 (no weight change), D-33 (advisory), D-35, D-41 (absence is reported, never filled).
Assumptions: AS-419.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

from nagahana.core.modes import RunMode, run_mode
from nagahana.datamodel.columnar import ColumnarUpdates
from nagahana.datamodel.fields import CATALOGUE, Level
from nagahana.governance.assumptions import assume
from nagahana.inference.engine import Engine, EngineSettings, TriggerResult
from nagahana.models.forecaster.model import ForecastIntervention
from nagahana.models.forecaster.routes import route_cumulative
from nagahana.models.nagahana import NagaHana
from nagahana.physics.term import PhysicsTerm
from nagahana.roles.contracts import AttackStage, ForensicReport


def _utc(t: float) -> str:
    """Epoch seconds → 'YYYY-MM-DD HH:MM:SS.sss UTC'."""
    import datetime as _dt

    return _dt.datetime.fromtimestamp(t, tz=_dt.UTC).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] + " UTC"


def _dominant_stage(r: TriggerResult) -> tuple[AttackStage, float]:
    """The most probable non-"none" stage of the route mixture at step 1, with its probability."""
    row = r.forecast.stage_probs[0]
    best = max((i for i in range(len(row)) if r.forecast.stages[i] is not AttackStage.NONE), key=lambda i: row[i])
    return r.forecast.stages[best], float(row[best])


def timeline(results: Sequence[TriggerResult], k: int) -> list[tuple[float, str]]:
    out = []
    for r in results:
        p = r.forecast.p_inf[-1]
        lo, hi = r.forecast.p_inf_interval[-1] if r.forecast.p_inf_interval else (math.nan, math.nan)
        stage, ps = _dominant_stage(r)
        top = max(r.compromise.items(), key=lambda kv: kv[1]) if r.compromise else ("-", math.nan)
        c = r.forecast.compute
        out.append((r.time, f"{r.kind} trigger {_utc(r.time)}: P_inf({k}) = {p:.3f} [{lo:.3f}, {hi:.3f}]; "
                            f"forecast stage {stage.value} ({ps:.2f}); most suspected {top[0]} ({top[1]:.2f}); "
                            f"trust {r.trust:.2f}; compute R_TSTCT={c.tstct_passes} R_TAAFT={c.taaft_passes} "
                            f"S={c.refinement_steps} K={c.horizon_k} N={c.samples_n} ({c.wall_time_s:.2f} s)"))
    return out


def narrative(results: Sequence[TriggerResult]) -> list[tuple[AttackStage, str, float]]:
    out: list[tuple[AttackStage, str, float]] = []
    last: AttackStage | None = None
    for r in results:
        stage, ps = _dominant_stage(r)
        if stage is not last:
            top = max(r.compromise.items(), key=lambda kv: kv[1])[0] if r.compromise else "-"
            out.append((stage, f"forecast stage became {stage.value} (p = {ps:.2f}) at {_utc(r.time)}; "
                               f"most suspected entity {top}", r.time))
            last = stage
    return out


def patient_zero(results: Sequence[TriggerResult], alert: float, top: int = 10) -> list[tuple[str, float]]:
    """Internal entities by (earliest alerting trigger, highest belief) — beliefs, not facts (AS-419)."""
    first: dict[str, float] = {}
    best: dict[str, float] = {}
    for r in results:
        for name, p in r.compromise.items():
            best[name] = max(best.get(name, 0.0), p)
            if p >= alert and name not in first:
                first[name] = r.time
    ranked = sorted(best, key=lambda n: (first.get(n, math.inf), -best[n]))
    return [(n, float(best[n])) for n in ranked[:top]]


def counterfactuals(engine: Engine, results: Sequence[TriggerResult], alert: float, ranking: Sequence[tuple[str, float]],
                    *, seed: int) -> list[str]:
    """Isolate the top-ranked entity at the first alerting trigger and re-imagine (common random numbers)."""
    if not results or not ranking:
        return []
    k = engine.settings.budgets.horizon_k
    n = engine.settings.budgets.routes_n
    target = ranking[0][0]
    alerting = [r for r in results if r.forecast.p_inf[-1] >= alert]
    r = alerting[0] if alerting else max(results, key=lambda x: x.forecast.p_inf[-1])
    if target not in r.entity_names:
        return [f"{target} is not in the window of the trigger at {_utc(r.time)}: no counterfactual computed"]
    v = r.entity_names.index(target)
    v_n = len(r.entity_names)
    flag = torch.zeros(1, 1, v_n, dtype=torch.bool)
    flag[0, 0, v] = True
    inter = ForecastIntervention(no_target=flag, no_outbound=flag)
    with torch.no_grad():
        base = engine.model.forecast(r.analysis, horizon_k=k, routes_n=n, generator=torch.Generator().manual_seed(seed))
        cf = engine.model.forecast(r.analysis, horizon_k=k, routes_n=n, generator=torch.Generator().manual_seed(seed),
                                   intervention=inter)
    p0 = float(route_cumulative(base.hazard[0, 0].double())[:, -1].mean())
    p1 = float(route_cumulative(cf.hazard[0, 0].double())[:, -1].mean())
    return [f"if {target} had been isolated at {_utc(r.time)}: P_inf({k}) {p0:.3f} → {p1:.3f} "
            f"(re-imagined over {n} routes with the same random numbers; advisory only)"]


def observability(cu: ColumnarUpdates, cadence: float) -> tuple[list[str], list[str]]:
    """(observability gaps, tamper signs) of a capture (module docstring)."""
    gaps: list[str] = []
    contrib = cu.contributing_cells()
    never = [c.name for j, c in enumerate(cu.columns) if not contrib[:, j].any()]
    if never:
        gaps.append(f"never observed in this capture ({len(never)} columns): {', '.join(never[:12])}"
                    + (" …" if len(never) > 12 else ""))
    pkt_cols = [j for j, c in enumerate(cu.columns) if c.field_id in CATALOGUE and CATALOGUE[c.field_id].level is Level.PACKET]
    if pkt_cols and len(cu):
        no_pkt = float((~contrib[:, pkt_cols].any(axis=1)).mean())
        gaps.append(f"{no_pkt:.1%} of state updates carry no packet-level field")
    t = np.sort(cu.updates["event_time"].to_numpy(dtype=np.float64))
    if t.size > 1:
        d = np.diff(t)
        long = np.nonzero(d > cadence)[0]
        for i in long[:10]:
            gaps.append(f"no state update for {d[i]:.1f} s after {_utc(float(t[i]))} (longer than the {cadence:.0f} s cadence)")
    signs: list[str] = []
    r = cu.updates["reorder_uncertainty_s"].to_numpy(dtype=np.float64) if "reorder_uncertainty_s" in cu.updates else np.zeros(0)
    n_ooo = int(np.nansum(r > 0))
    signs.append(f"{n_ooo} records arrived out of time order (ordering uncertainty > 0); "
                 "a fact about the capture, not proof of tampering")
    signs.append(f"clock quality: {cu.clock_quality or 'unknown'}")
    return gaps, signs


def build_report(engine: Engine, *, seed: int) -> ForensicReport:
    """The ForensicReport of everything the engine has processed (module docstring)."""
    assume("AS-419", by=__name__)
    res = engine.results
    alert = engine.settings.alert_threshold
    ranking = patient_zero(res, alert)
    assert engine.log.cu is not None
    gaps, signs = observability(engine.log.cu, engine.settings.cadence_s)
    return ForensicReport(timeline=timeline(res, engine.settings.budgets.horizon_k), narrative=narrative(res),
                          patient_zero=ranking, counterfactuals=counterfactuals(engine, res, alert, ranking, seed=seed),
                          observability_gaps=gaps + [f"engine: {n}" for n in engine.notes], tamper_signs=signs)


def replay(model: NagaHana, path: str | Path, *, settings: EngineSettings, physics: PhysicsTerm | None, seed: int,
           sandboxed: bool) -> tuple[ForensicReport, Engine]:
    """Forensic replay of a capture (RunMode.FORENSIC_REPLAY). Returns (report, the engine with every result)."""
    from nagahana.ingest.pcap import PcapSource

    with run_mode(RunMode.FORENSIC_REPLAY):
        cu = PcapSource(path, sandboxed=sandboxed).columnar()
        engine = Engine(model, settings=settings, physics=physics, seed=seed)
        engine.ingest(cu)
        engine.finish()
        return build_report(engine, seed=seed), engine


__all__ = ["build_report", "counterfactuals", "narrative", "observability", "patient_zero", "replay", "timeline"]
