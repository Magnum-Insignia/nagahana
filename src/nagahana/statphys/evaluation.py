"""Statistical-physics outputs in the evaluation contract, and their scores (D-56; architecture section 7).

Contract
--------
Readings reach the shared evaluation records through `ModelOutputs.component`
(src/nagahana/evaluation/predictions.py): a free-form name -> array map that `save_outputs` stores
with every run and `load_outputs` reads back. `component_arrays` writes the names below, one row
per trigger in reading order (m triggers); floats are float64 and NaN marks a value that is not
defined (absence is not zero, D-41).

    statphys.time                          [m]    trigger time, epoch seconds
    statphys.regular                       [m]    1.0 cadence trigger, 0.0 priority trigger
    statphys.temperature                   [m]    Gibbs temperature T of the reading
    statphys.alarm                         [m]    1.0 / 0.0, NaN when no calibrated series is defined
    statphys.series.<name>                 [m]    every scalar of `StatPhysReading.series`:
        energy.total, energy.marginal, energy.lens.<lens>
        <ensemble>.<potential>             ensemble in entities, slots, routes; potential in
                                           log_partition, free_energy, mean_energy, entropy,
                                           heat_capacity, size
        routes.p_inf_mean, routes.p_inf_susceptibility, routes.p_inf_thermal_response
        traffic.<dist>, traffic_samples.<dist>, traffic_jsd.<dist>   network traffic distributions
        graph.nodes, graph.edges.<plane | aggregate>, graph.vn.<plane | aggregate>,
        graph.spectral_<entropy | free_energy | mean | heat_capacity>.t<i> (tau = statphys.spectral_taus[i]),
        graph.relative_entropy, graph.best_relative_entropy, graph.reduced_layers,
        graph.jsd.<plane | aggregate>
    statphys.growth.<name>                 [m]    growth rate per second of a growing series
    statphys.ews.<series>.<indicator>      [m]    streaming indicators (NaN on priority triggers)
    statphys.ews.<series>.tau.<indicator>  [m]    trailing Kendall trend of the indicator
    statphys.ews.<series>.level / .trend_score / .alarm   [m]  composite scores and the series alarm
    statphys.spectral_taus                 [n_tau] diffusion times of the spectral series
    statphys.entity.key                    int64 [m, V_max]  stable entity id (-1 padding)
    statphys.entity.name                   str [m, V_max]    entity name ("" padding)
    statphys.entity.<field>                [m, V_max]  energy, occupation, energy_growth,
                                           stage_free_energy, stage_entropy, stage_heat_capacity,
                                           imagined_free_energy, imagined_entropy, imagined_size,
                                           traffic.<dist>
The batch path (`window_component_arrays`) writes statphys.time and statphys.series.* and
statphys.growth.* from `trajectory.window_series`.

Scores (architecture section 7, Energy: "OOD AUROC; early-warning lead time; false-alarm rate of
energy alerts")
- `novelty_auroc`: AUROC of a novelty score (free energy, marginal energy) for unfamiliar against
  familiar units, by the Mann-Whitney statistic with mid-ranks for ties (Hanley and McNeil, "The
  meaning and use of the area under a receiver operating characteristic (ROC) curve", Radiology
  143:29, 1982): AUC = (R_1 - n_1 (n_1 + 1) / 2) / (n_1 n_0).
- `alarm_lead_times`: for an episode (start s, completion c), t_c minus the first alarm in
  [s - lookback, c); NaN without a timely alarm (the convention of evaluation/forecasting.lead_time).
- `false_alarm_rate`: alarms among benign triggers over benign triggers (NaN without benign triggers).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np

from nagahana.evaluation.predictions import EpisodeTable
from nagahana.statphys.trajectory import StatPhysReading

PREFIX = "statphys."
#: Per-entity fields written as [m, V_max] matrices (besides key and name).
ENTITY_FIELDS: tuple[str, ...] = ("energy", "occupation", "energy_growth", "stage_free_energy", "stage_entropy",
                                  "stage_heat_capacity", "imagined_free_energy", "imagined_entropy", "imagined_size")


def _flag(x: bool | None) -> float:
    return math.nan if x is None else float(bool(x))


def component_arrays(readings: Sequence[StatPhysReading], *, spectral_taus: Sequence[float] = ()) -> dict[str, np.ndarray]:
    """The `ModelOutputs.component` arrays of a sequence of readings (module docstring)."""
    m = len(readings)
    out: dict[str, np.ndarray] = {
        PREFIX + "time": np.array([r.time for r in readings], dtype=np.float64),
        PREFIX + "regular": np.array([float(r.regular) for r in readings], dtype=np.float64),
        PREFIX + "temperature": np.array([r.temperature for r in readings], dtype=np.float64),
        PREFIX + "alarm": np.array([_flag(r.alarm) for r in readings], dtype=np.float64),
        PREFIX + "spectral_taus": np.asarray(list(spectral_taus), dtype=np.float64),
    }

    def column(prefix: str, rows: Sequence[Mapping[str, float]]) -> None:
        names = sorted({k for row in rows for k in row})
        for name in names:
            out[prefix + name] = np.array([row.get(name, math.nan) for row in rows], dtype=np.float64)

    column(PREFIX + "series.", [r.series for r in readings])
    column(PREFIX + "growth.", [{k[len("growth."):]: v for k, v in r.growth.items()} for r in readings])
    ews_rows: list[dict[str, float]] = []
    for r in readings:
        row: dict[str, float] = {}
        for series, step in r.ews.items():
            for ind, value in step.indicators.items():
                row[f"{series}.{ind}"] = value
            for ind, value in step.taus.items():
                row[f"{series}.tau.{ind}"] = value
            row[f"{series}.level"] = step.level
            row[f"{series}.trend_score"] = step.trend_score
            row[f"{series}.alarm"] = _flag(step.alarm)
        ews_rows.append(row)
    column(PREFIX + "ews.", ews_rows)
    v_max = max((len(r.entities) for r in readings), default=0)
    keys = np.full((m, v_max), -1, dtype=np.int64)
    names = np.full((m, v_max), "", dtype=object)
    fields = {f: np.full((m, v_max), np.nan, dtype=np.float64) for f in ENTITY_FIELDS}
    dists = sorted({d for r in readings for e in r.entities for d in e.traffic})
    traffic = {d: np.full((m, v_max), np.nan, dtype=np.float64) for d in dists}
    for i, r in enumerate(readings):
        for j, e in enumerate(r.entities):
            keys[i, j], names[i, j] = e.key, e.name
            vals = {"energy": e.energy, "occupation": e.occupation, "energy_growth": e.energy_growth,
                    "stage_free_energy": e.stage.free_energy, "stage_entropy": e.stage.entropy,
                    "stage_heat_capacity": e.stage.heat_capacity, "imagined_free_energy": e.imagined.free_energy,
                    "imagined_entropy": e.imagined.entropy, "imagined_size": float(e.imagined.size)}
            for f in ENTITY_FIELDS:
                fields[f][i, j] = vals[f]
            for d, value in e.traffic.items():
                traffic[d][i, j] = value
    out[PREFIX + "entity.key"] = keys
    out[PREFIX + "entity.name"] = names.astype(str) if v_max else np.zeros((m, 0), dtype=str)
    for f, arr in fields.items():
        out[PREFIX + "entity." + f] = arr
    for d, arr in traffic.items():
        out[PREFIX + "entity.traffic." + d] = arr
    return out


def window_component_arrays(series: Mapping[str, np.ndarray], *, times: np.ndarray, valid: np.ndarray) -> dict[str, np.ndarray]:
    """Component arrays of the batch path: the valid triggers of [B, M] series in (b, m) order.

    series: name -> [B, M] (`trajectory.window_series`; names starting with "growth." go to
    statphys.growth.*); times: [B, M] epoch seconds of the triggers; valid: bool [B, M].
    """
    ok = np.asarray(valid, dtype=bool)
    out = {PREFIX + "time": np.asarray(times, dtype=np.float64)[ok]}
    for name, arr in series.items():
        a = np.asarray(arr, dtype=np.float64)
        if a.shape != ok.shape:
            raise ValueError(f"series {name!r} must be [B, M] like `valid`")
        key = PREFIX + (name if name.startswith("growth.") else "series." + name)
        out[key] = a[ok]
    return out


def series_from_component(component: Mapping[str, np.ndarray]) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
    """(times [m], series name -> [m] (also "growth.<name>"), regular bool [m]) from component arrays."""
    if PREFIX + "time" not in component:
        raise ValueError("no statphys component arrays (statphys.time is missing)")
    times = np.asarray(component[PREFIX + "time"], dtype=np.float64)
    reg = component.get(PREFIX + "regular")
    regular = np.ones(times.shape[0], dtype=bool) if reg is None else np.asarray(reg, dtype=np.float64) > 0.5
    series: dict[str, np.ndarray] = {}
    for key, arr in component.items():
        if key.startswith(PREFIX + "series."):
            series[key[len(PREFIX + "series."):]] = np.asarray(arr, dtype=np.float64)
        elif key.startswith(PREFIX + "growth."):
            series["growth." + key[len(PREFIX + "growth."):]] = np.asarray(arr, dtype=np.float64)
    return times, series, regular


def novelty_auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """AUROC of `scores` for labels 1 (unfamiliar) against 0 (familiar); -1 or non-finite scores ignored."""
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    if s.shape != y.shape:
        raise ValueError("scores and labels must have the same shape")
    ok = np.isfinite(s) & ((y == 0) | (y == 1))
    s, y = s[ok], y[ok]
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return math.nan
    order = np.argsort(s, kind="stable")
    ranks = np.empty(s.shape[0], dtype=np.float64)
    ss = s[order]
    i = 0
    while i < ss.shape[0]:                                                     # mid-ranks of tied scores
        j = i
        while j + 1 < ss.shape[0] and ss[j + 1] == ss[i]:
            j += 1
        ranks[order[i: j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    r1 = float(ranks[y == 1].sum())
    return (r1 - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def alarm_lead_times(times: np.ndarray, alarm: np.ndarray, episodes: EpisodeTable, *, network: np.ndarray | None = None,
                     lookback: float = 0.0) -> np.ndarray:
    """Lead time (seconds) of every episode: completion minus the first alarm in [start - lookback, completion).

    times [m] epoch seconds and alarm [m] (1.0 / 0.0 / NaN) of the triggers; network [m] (optional): the
    network of each trigger, matched against the episode's network. NaN for an episode without a timely
    alarm.
    """
    if lookback < 0:
        raise ValueError("lookback must be >= 0")
    t = np.asarray(times, dtype=np.float64)
    a = np.asarray(alarm, dtype=np.float64) > 0.5
    net = None if network is None else np.asarray(network).astype(str)
    frame = episodes.frame
    out = np.full(len(frame), np.nan, dtype=np.float64)
    for i, (start, done, ep_net) in enumerate(zip(frame["start"].to_numpy(dtype=np.float64),
                                                  frame["completion"].to_numpy(dtype=np.float64),
                                                  frame["network"].astype(str), strict=True)):
        sel = a & (t >= start - lookback) & (t < done)
        if net is not None:
            sel &= net == ep_net
        hits = np.nonzero(sel)[0]
        if hits.size:
            out[i] = done - float(t[hits].min())
    return out


def false_alarm_rate(alarm: np.ndarray, benign: np.ndarray) -> float:
    """Alarms among benign triggers over benign triggers (alarm NaN counts as no alarm)."""
    a = np.asarray(alarm, dtype=np.float64) > 0.5
    b = np.asarray(benign, dtype=bool)
    if a.shape != b.shape:
        raise ValueError("alarm and benign must have the same shape")
    n = int(b.sum())
    return float((a & b).sum()) / n if n else math.nan


def early_warning_report(times: np.ndarray, alarm: np.ndarray, episodes: EpisodeTable, benign: np.ndarray, *,
                         network: np.ndarray | None = None, lookback: float = 0.0) -> dict[str, float]:
    """Episode detection rate, median lead time over detected episodes and false-alarm rate."""
    lead = alarm_lead_times(times, alarm, episodes, network=network, lookback=lookback)
    found = np.isfinite(lead)
    return {"episodes": float(lead.shape[0]), "detected": float(found.sum()),
            "detection_rate": float(found.mean()) if lead.size else math.nan,
            "median_lead_time": float(np.median(lead[found])) if found.any() else math.nan,
            "false_alarm_rate": false_alarm_rate(alarm, benign), "benign_triggers": float(np.asarray(benign).sum())}


__all__ = ["ENTITY_FIELDS", "PREFIX", "alarm_lead_times", "component_arrays", "early_warning_report",
           "false_alarm_rate", "novelty_auroc", "series_from_component", "window_component_arrays"]
