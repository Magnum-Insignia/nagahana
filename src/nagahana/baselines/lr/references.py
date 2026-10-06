"""Reference forecasts for skill scores: persistence and climatology (evaluation chapter, reference forecasts).

Infiltration forecast (the units of the hazard model: usable triggers)

    climatology   the base rate of infiltration within k steps estimated on the training split, censoring
                  included: the life-table (actuarial) estimator h_j = d_j / n_j over the training triggers,
                  with d_j the first infiltrations in step j and n_j the triggers at risk in step j, and
                  P(T <= k) = 1 - prod_{j <= k} (1 - h_j) (Kaplan and Meier, JASA 53(282), 1958, in discrete
                  time; it is also the hazard model without features). The same curve is issued for every
                  trigger. A step with no training trigger at risk has no estimate and raises.
    persistence   "the current state continues": P(T <= k) = 1 for every k when the infiltration state holds
                  in the window ending at the trigger (ForecastPredictions.infiltrated_now, corpus.py), else 0
                  (AS-521): the definition the evaluation applies to any forecast record. It is a 0/1
                  forecast, so its log score is unbounded and its expected calibration error equals its Brier
                  score.

Time to event: the climatology survival curve 1 - P(T <= k) for every trigger (risk -RMST, the same for
all), and the persistence curve S = 0 from the first step when the infiltration state holds now, else 1.

Next-state forecast (the units of the ridge forecaster): persistence repeats the current window's
standardised states at every horizon; climatology issues the training mean, which is 0 in standardised
units. A current state that is undefined (a share over an empty window) has no persisted value, and the
climatology value 0 is issued for it; the count of such entries is reported (AS-522).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from nagahana.core.errors import InvariantViolation
from nagahana.evaluation.predictions import ForecastPredictions, StateForecastPredictions, TimeToEventPredictions

from .corpus import LRCorpus
from .design import Units, gather, trigger_units, unit_meta
from .features import STATE_FEATURES
from .hazard import risk_sets, survival_from_hazard
from .ridge import FEATURE_NAMES, StateScaler, state_targets, state_units


@dataclass
class LifeTable:
    """Climatology of the time to infiltration (module docstring)."""

    hazard: np.ndarray       # [K]
    at_risk: np.ndarray      # [K]
    events: np.ndarray       # [K]
    window_seconds: float

    @classmethod
    def fit(cls, corpus: LRCorpus) -> LifeTable:
        units = trigger_units(corpus, ("train",), usable_only=True)
        k = corpus.horizon_k
        ar, y = risk_sets(gather(corpus, units, "t_event_step").astype(np.int64),
                          gather(corpus, units, "t_observed_steps").astype(np.int64), k)
        n = ar.sum(axis=0).astype(np.float64)
        d = (y * ar).sum(axis=0)
        if np.any(n == 0):
            raise InvariantViolation(f"climatology: no training trigger is at risk in steps {(np.flatnonzero(n == 0) + 1).tolist()}")
        return cls(hazard=d / n, at_risk=n, events=d, window_seconds=corpus.window_seconds)

    def forecast(self, corpus: LRCorpus, units: Units) -> tuple[ForecastPredictions, TimeToEventPredictions]:
        m, k = len(units), self.hazard.size
        h = np.broadcast_to(self.hazard, (m, k)).copy()
        surv, p_inf = survival_from_hazard(h)
        meta = unit_meta(corpus, units, triggers=True)
        fc = ForecastPredictions(p_inf=p_inf, window_seconds=self.window_seconds,
                                 event_step=gather(corpus, units, "t_event_step").astype(np.int64),
                                 observed_steps=gather(corpus, units, "t_observed_steps").astype(np.int64), meta=meta, hazard=h,
                                 infiltrated_now=gather(corpus, units, "t_infiltrated_now").astype(bool))
        risk = -self.window_seconds * (1.0 + surv[:, :-1].sum(axis=1)) if m else np.zeros(0)
        tte = TimeToEventPredictions(risk=risk, survival=surv, time_grid=self.window_seconds * np.arange(1, k + 1),
                                     event_time=gather(corpus, units, "t_event_time").astype(np.float64),
                                     event_observed=gather(corpus, units, "t_event_observed").astype(bool), meta=meta.copy())
        return fc, tte

    def state(self) -> dict[str, Any]:
        return {"hazard": self.hazard.tolist(), "at_risk": self.at_risk.tolist(), "events": self.events.tolist(),
                "window_seconds": self.window_seconds}

    @classmethod
    def from_state(cls, d: dict[str, Any]) -> LifeTable:
        return cls(np.asarray(d["hazard"], dtype=np.float64), np.asarray(d["at_risk"], dtype=np.float64),
                   np.asarray(d["events"], dtype=np.float64), float(d["window_seconds"]))


def persistence_forecast(corpus: LRCorpus, units: Units) -> tuple[ForecastPredictions, TimeToEventPredictions]:
    """Persistence of the infiltration state (module docstring)."""
    k, w = corpus.horizon_k, corpus.window_seconds
    now = gather(corpus, units, "t_infiltrated_now").astype(bool)
    p = now.astype(np.float64)
    h = np.zeros((len(units), k))
    h[:, 0] = p
    p_inf = np.repeat(p[:, None], k, axis=1)
    meta = unit_meta(corpus, units, triggers=True)
    fc = ForecastPredictions(p_inf=p_inf, window_seconds=w, event_step=gather(corpus, units, "t_event_step").astype(np.int64),
                             observed_steps=gather(corpus, units, "t_observed_steps").astype(np.int64), meta=meta, hazard=h,
                             infiltrated_now=now)
    tte = TimeToEventPredictions(risk=p.copy(), survival=1.0 - p_inf, time_grid=w * np.arange(1, k + 1),
                                 event_time=gather(corpus, units, "t_event_time").astype(np.float64),
                                 event_observed=gather(corpus, units, "t_event_observed").astype(bool), meta=meta.copy())
    return fc, tte


def state_references(corpus: LRCorpus, roles: tuple[str, ...], scaler: StateScaler,
                     feat_cfg: Any) -> tuple[StateForecastPredictions, StateForecastPredictions, int]:
    """(persistence, climatology, number of persisted entries filled with the climatology value)."""
    units = state_units(corpus, roles)
    k, f = corpus.horizon_k, len(STATE_FEATURES)
    cur, fut, obs = state_targets(corpus, units, feat_cfg)
    cur_z = scaler.transform(cur)
    missing = ~np.isfinite(cur_z)
    pers = np.repeat(np.where(missing, 0.0, cur_z)[:, None, :], k, axis=1)
    observed = np.where(obs, scaler.transform(fut), 0.0)
    meta = unit_meta(corpus, units, triggers=True)
    horizons = np.arange(1, k + 1)
    current = np.where(missing, 0.0, cur_z)
    p = StateForecastPredictions(predicted=pers, observed=observed, mask=obs, horizons=horizons,
                                 feature_names=FEATURE_NAMES, meta=meta, current=current, current_mask=~missing)
    c = StateForecastPredictions(predicted=np.zeros((len(units), k, f)), observed=observed, mask=obs, horizons=horizons,
                                 feature_names=FEATURE_NAMES, meta=meta.copy(), current=current, current_mask=~missing)
    return p, c, int(missing.sum())


__all__ = ["LifeTable", "persistence_forecast", "state_references"]
