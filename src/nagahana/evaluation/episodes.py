"""Attack episodes: detection rate, time to detect, lead time and its distribution at fixed operating points.

An episode a (EpisodeTable) is one labelled attack on one target: it starts at s_a (first malicious
activity) and completes at T_a, the time it reaches its infiltration state (AS-18). Alerts come from a
time series of scored units (forecast triggers scored by P_inf(K), or detection units) on the same
dataset and network, restricted to the episode's target entity when both sides are entity-level.

Lead time (thesis equation). With the alert time A_a(theta) = inf{t in W_a : score(t) >= theta},

    LT_a(theta) = T_a - A_a(theta),

over the alert window W_a = [T_a - K w, T_a + late]: a forecast issued more than one horizon K w before
completion cannot be about this completion, and alerts up to `late` seconds after completion give the
negative lead times of alerts that followed completion. An episode never alerted in W_a has
LT = -inf, so it ranks below every alerted episode (thesis section on timeliness) and the median lead
time is an order statistic that stays defined while fewer than half the episodes are missed. The
window may instead start at max(s_a, T_a - K w) (`window_start = "episode"`).

Reported per group: the median lead time, the share of episodes alerted before completion (LT > 0), the
lead-time quantiles, the detection rate (an alert in [s_a, T_a + late]) and the time to detect (first
alert at or after s_a, minus s_a). The thesis reports lead time at each method's own operating
threshold and at the threshold for the configured false-positive rate alpha, the conformal threshold
fitted on benign calibration units (ranking.conformal_threshold), so that lead time cannot be bought
with false alarms. The resampling unit of these metrics is the episode (cluster bootstrap).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from nagahana.core.errors import InvariantViolation
from nagahana.evaluation._arrays import safe_ratio, unbatch, weight_matrix, weighted_quantile
from nagahana.evaluation.predictions import EpisodeTable


@dataclass(frozen=True)
class AlertSettings:
    """How alerts are matched to episodes."""

    horizon_seconds: float
    late_seconds: float
    match_entity: bool = True
    window_start: str = "horizon"

    def __post_init__(self) -> None:
        if not self.horizon_seconds > 0 or self.late_seconds < 0:
            raise ValueError("horizon_seconds must be positive and late_seconds non-negative")
        if self.window_start not in ("horizon", "episode"):
            raise ValueError("window_start must be 'horizon' or 'episode'")


def _keys(dataset: np.ndarray, network: np.ndarray) -> np.ndarray:
    return np.char.add(np.char.add(dataset.astype(str), "\x1f"), network.astype(str))


def episode_alerts(episodes: EpisodeTable, meta: pd.DataFrame, score: Any, threshold: Any,
                   settings: AlertSettings) -> pd.DataFrame:
    """Per episode: alert time in the lead-time window, lead time, first alert after the start, detection.

    meta: the scored units (columns time, dataset, network, entity); score [n]; alert when score >= threshold,
    with one threshold for all units or one per unit (for example per-dataset conformal thresholds).
    Returns one row per episode with columns episode, alert_time, lead_time, detect_time, detected,
    time_to_detect, alerted_before_completion.
    """
    s = np.asarray(score, dtype=np.float64)
    if s.shape != (len(meta),):
        raise InvariantViolation("score must have one value per unit of meta")
    fr = episodes.frame
    t_u = meta["time"].to_numpy(dtype=np.float64)
    ent_u = meta["entity"].to_numpy(dtype=np.int64)
    key_u = _keys(meta["dataset"].to_numpy(), meta["network"].to_numpy())
    thr = np.asarray(threshold, dtype=np.float64)
    if thr.ndim not in (0, 1) or (thr.ndim == 1 and thr.shape != s.shape):
        raise InvariantViolation("threshold must be a number or one value per unit")
    alert = s >= thr
    start = fr["start"].to_numpy(dtype=np.float64)
    done = fr["completion"].to_numpy(dtype=np.float64)
    key_e = _keys(fr["dataset"].to_numpy(), fr["network"].to_numpy())
    ent_e = fr["entity"].to_numpy(dtype=np.int64)
    n_ep = len(fr)
    lead_alert = np.full(n_ep, np.nan)
    detect_alert = np.full(n_ep, np.nan)
    # Alert times indexed once per network: all units, network-level units (entity -1) and per entity.
    alert_idx = np.flatnonzero(alert)
    order = alert_idx[np.argsort(t_u[alert_idx], kind="stable")]
    every: dict[str, np.ndarray] = {}
    network_level: dict[str, np.ndarray] = {}
    per_entity: dict[tuple[str, int], np.ndarray] = {}
    for k in (np.unique(key_u[order]) if order.size else []):
        sel = order[key_u[order] == k]
        every[str(k)] = t_u[sel]
        network_level[str(k)] = t_u[sel[ent_u[sel] < 0]]
        for e in np.unique(ent_u[sel]):
            if e >= 0:
                per_entity[(str(k), int(e))] = t_u[sel[ent_u[sel] == e]]
    empty = np.zeros(0)
    for a in range(n_ep):
        k = str(key_e[a])
        if settings.match_entity and ent_e[a] >= 0:
            # The target's own units plus units without entity resolution (network-level forecasts).
            times = np.sort(np.concatenate([per_entity.get((k, int(ent_e[a])), empty), network_level.get(k, empty)]))
        else:
            times = every.get(k, empty)
        if times.size == 0:
            continue
        lo = done[a] - settings.horizon_seconds
        if settings.window_start == "episode":
            lo = max(lo, start[a])
        hi = done[a] + settings.late_seconds
        i = int(np.searchsorted(times, lo, side="left"))
        if i < times.size and times[i] <= hi:
            lead_alert[a] = times[i]
        j = int(np.searchsorted(times, start[a], side="left"))
        if j < times.size and times[j] <= hi:
            detect_alert[a] = times[j]
    lead = np.where(np.isfinite(lead_alert), done - lead_alert, -np.inf)
    return pd.DataFrame({
        "episode": fr["episode"].astype(str).to_numpy(),
        "alert_time": lead_alert,
        "lead_time": lead,
        "detect_time": detect_alert,
        "detected": np.isfinite(detect_alert),
        "time_to_detect": detect_alert - start,
        "alerted_before_completion": lead > 0,
    })


def lead_time_metrics(alerts: pd.DataFrame, weights: Any = None, *, quantiles: tuple[float, ...] = (0.1, 0.25, 0.75, 0.9)
                      ) -> dict[str, Any]:
    """Median lead time, share alerted before completion, detection rate, median time to detect, quantiles.

    weights: over episodes ([E] or [B, E]). Lead times of missed episodes are -inf (ranked last).
    """
    lead = alerts["lead_time"].to_numpy(dtype=np.float64)
    w, batched = weight_matrix(weights, lead.size)
    total = w.sum(axis=1)
    out: dict[str, np.ndarray] = {"n_episodes": total}
    out["median_lead_time"] = weighted_quantile(lead, w, 0.5) if lead.size else np.full(w.shape[0], np.nan)
    for q in quantiles:
        out[f"lead_time_q{round(q * 100):02d}"] = weighted_quantile(lead, w, q) if lead.size else np.full(w.shape[0], np.nan)
    out["alerted_before_completion"] = safe_ratio(w @ (lead > 0).astype(np.float64), total)
    det = alerts["detected"].to_numpy(dtype=bool)
    out["detection_rate"] = safe_ratio(w @ det.astype(np.float64), total)
    ttd = alerts["time_to_detect"].to_numpy(dtype=np.float64)
    ttd_w = w * det[None, :]
    out["median_time_to_detect"] = (weighted_quantile(np.where(det, ttd, np.inf), ttd_w, 0.5)
                                    if det.any() else np.full(w.shape[0], np.nan))
    return {name: unbatch(np.atleast_1d(v), batched) for name, v in out.items()}


def assign_episodes(meta: pd.DataFrame, episodes: EpisodeTable, horizon_seconds: float, *, late_seconds: float = 0.0
                    ) -> np.ndarray:
    """Episode id of each unit inside some episode's window [T_a - K w, T_a + late] on its network ("" otherwise).

    Entity-level units are matched to their target's episodes; a unit inside several windows takes the
    episode with the earliest completion. Used to resample triggers by episode.
    """
    fr = episodes.frame
    out = np.full(len(meta), "", dtype=object)
    if len(fr) == 0 or len(meta) == 0:
        return out.astype(str)
    t_u = meta["time"].to_numpy(dtype=np.float64)
    ent_u = meta["entity"].to_numpy(dtype=np.int64)
    key_u = _keys(meta["dataset"].to_numpy(), meta["network"].to_numpy())
    key_e = _keys(fr["dataset"].to_numpy(), fr["network"].to_numpy())
    ent_e = fr["entity"].to_numpy(dtype=np.int64)
    done = fr["completion"].to_numpy(dtype=np.float64)
    ids = fr["episode"].astype(str).to_numpy()
    best = np.full(len(meta), math.inf)
    for a in np.argsort(done, kind="stable"):
        inside = (key_u == key_e[a]) & (t_u >= done[a] - horizon_seconds) & (t_u <= done[a] + late_seconds)
        if ent_e[a] >= 0:
            inside &= (ent_u == ent_e[a]) | (ent_u < 0)
        take = inside & (done[a] < best)
        out[take] = ids[a]
        best[take] = done[a]
    return out.astype(str)
