"""Prediction records shared by every model and by the evaluation code.

NagaHana, the logistic-regression baselines and the published baselines all report their outputs in
these records, so one evaluation pipeline scores every model in exactly the same way. Arrays are
NumPy: float64 for scores, probabilities and times, int64 for codes and steps, bool for masks. Every
record validates its shapes and value ranges when it is constructed, so a malformed output is caught
where it is produced and not later inside a metric.

Units of evaluation

    detection unit   one scored item: a state update, a time window or a flow (field `unit`)
    forecast unit    one trigger: a time at which a model forecasts the next K windows
    episode          one labelled attack occurrence on one target, with a start and a completion time

Metadata

Every record carries a pandas DataFrame `meta` with one row per unit and at least the columns in
META_COLUMNS. The evaluation code groups by these columns to report results per dataset, per network,
per protocol and for known and novel attack families separately (D-23, AS-35):

    time      float64, epoch seconds of the unit (the trigger time for forecast units)
    dataset   dataset name, for example "cse-cic-ids2018"
    network   network or capture identifier inside the dataset
    family    attack family of the unit, or "benign"
    novelty   "known", "novel" or "" (only zero-shot units carry a novelty mark)
    split     "train", "val", "test" or "zero_shot"
    entity    entity id inside its window table, or -1 when the unit is not entity-level

Forecast targets

For a trigger at time t with window length w, step k (k = 1 ... K) covers (t + (k - 1) w, t + k w].
`event_step` is the first step in which infiltration occurs (AS-18), or 0 when none occurs inside the
observed steps. `observed_steps` is how many of the K steps lie inside the label horizon; a unit with
event_step = 0 and observed_steps < K is right-censored after observed_steps steps. P_inf(k) is the
probability that the first infiltration falls in steps 1 ... k, so it never decreases with k.

Optional fields for the evaluation protocols

Every field added after the first release of these records is optional and defaults to None, so a
record built without it behaves exactly as before. ForecastPredictions may carry the current state
(`infiltrated_now`, used by the persistence reference), an uncertainty band (`p_inf_lower`,
`p_inf_upper`), the safe horizon and the model's own alert threshold on P_inf(K). StateForecastPredictions
may carry a predictive variance (Gaussian CRPS), predictive samples (energy score) and the features of
the window at the forecast origin (the persistence reference). PathPredictions may carry the true value
of each imagined route in simulated worlds (ordinal safety, protocol P6). ForensicPredictions (protocol
P7), OperationsMeasurements (protocol P8), ArenaRun (protocol P-CW, a defender agent trained and evaluated in
a cyber-defence environment such as CyberWheel) and ExplanationCurves (deletion and insertion curves of
the explanations) are records of their own.

Persistence

`save_outputs` and `load_outputs` write and read a ModelOutputs bundle as one .npz file (metadata
columns stored as string or numeric arrays), so predictions of every run can be archived next to the
run's configuration and re-scored later without re-running the model.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pandas as pd

from nagahana.core.errors import InvariantViolation

META_COLUMNS: tuple[str, ...] = ("time", "dataset", "network", "family", "novelty", "split", "entity")
SPLITS: frozenset[str] = frozenset({"train", "val", "test", "zero_shot"})
NOVELTY: frozenset[str] = frozenset({"known", "novel", ""})
_TOL = 1e-9


def _f64(name: str, x: Any, ndim: int) -> np.ndarray:
    # Convert to a float64 array of the expected rank, rejecting NaN and infinity.
    a = np.asarray(x, dtype=np.float64)
    if a.ndim != ndim:
        raise InvariantViolation(f"{name} must have {ndim} dimensions, got shape {a.shape}")
    if not np.all(np.isfinite(a)):
        raise InvariantViolation(f"{name} contains NaN or infinite values")
    return a


def _prob(name: str, a: np.ndarray) -> None:
    # Probabilities must lie in [0, 1].
    if a.size and (a.min() < -_TOL or a.max() > 1.0 + _TOL):
        raise InvariantViolation(f"{name} must lie in [0, 1]; got range [{a.min()}, {a.max()}]")


def make_meta(n: int, **columns: Any) -> pd.DataFrame:
    """Build a metadata frame with n rows; missing standard columns get neutral defaults.

    Defaults: time 0.0, dataset "", network "", family "benign", novelty "", split "test", entity -1.
    """
    defaults: dict[str, Any] = {"time": 0.0, "dataset": "", "network": "", "family": "benign",
                                "novelty": "", "split": "test", "entity": -1}
    data: dict[str, Any] = {}
    for col in META_COLUMNS:
        value = columns.pop(col, defaults[col])
        arr = np.asarray(value)
        data[col] = np.repeat(arr, n) if arr.ndim == 0 else arr
    for col, value in columns.items():                    # extra columns are kept as given
        data[col] = value
    meta = pd.DataFrame(data)
    validate_meta(meta, n)
    return meta


def validate_meta(meta: pd.DataFrame, n: int) -> None:
    """Check that `meta` has n rows, the standard columns and admissible values."""
    if len(meta) != n:
        raise InvariantViolation(f"meta has {len(meta)} rows, expected {n}")
    missing = [c for c in META_COLUMNS if c not in meta.columns]
    if missing:
        raise InvariantViolation(f"meta is missing columns {missing}")
    bad_split = set(meta["split"].astype(str)) - SPLITS
    if bad_split:
        raise InvariantViolation(f"unknown split values {sorted(bad_split)}")
    bad_nov = set(meta["novelty"].astype(str)) - NOVELTY
    if bad_nov:
        raise InvariantViolation(f"unknown novelty values {sorted(bad_nov)}")
    marked = meta["novelty"].astype(str) != ""
    if bool((marked & (meta["split"].astype(str) != "zero_shot")).any()):
        raise InvariantViolation("only zero-shot units may carry a novelty mark (D-23)")
    if not np.all(np.isfinite(meta["time"].to_numpy(dtype=np.float64))):
        raise InvariantViolation("meta.time must be finite")


@dataclass
class DetectionPredictions:
    """Attack probability per detection unit.

    score       float64 [n], probability that the unit is malicious
    label       int64 [n], 1 malicious, 0 benign, -1 unknown (excluded from scoring)
    unit        "state_update", "window" or "flow"
    threshold   operating threshold chosen on validation data, or None when only threshold-free
                metrics are to be computed
    meta        one row per unit (META_COLUMNS)
    """

    score: np.ndarray
    label: np.ndarray
    unit: str
    meta: pd.DataFrame
    threshold: float | None = None

    def __post_init__(self) -> None:
        self.score = _f64("score", self.score, 1)
        _prob("score", self.score)
        self.label = np.asarray(self.label, dtype=np.int64)
        if self.label.shape != self.score.shape:
            raise InvariantViolation("label and score must have the same shape")
        if not np.isin(self.label, (-1, 0, 1)).all():
            raise InvariantViolation("label must be 1, 0 or -1")
        if self.unit not in ("state_update", "window", "flow"):
            raise InvariantViolation(f"unknown detection unit {self.unit!r}")
        if self.threshold is not None and not 0.0 <= float(self.threshold) <= 1.0:
            raise InvariantViolation("threshold must lie in [0, 1]")
        validate_meta(self.meta, len(self.score))

    @property
    def scored(self) -> np.ndarray:
        """Boolean mask of units with a known label."""
        return self.label >= 0


@dataclass
class ForecastPredictions:
    """Infiltration forecast over the next K windows, per trigger.

    p_inf            float64 [m, K], P(first infiltration in steps 1 ... k); non-decreasing in k
    window_seconds   length w of one step in seconds
    event_step       int64 [m], first step with infiltration (1 ... K), 0 if none was observed
    observed_steps   int64 [m], number of the K steps inside the label horizon (0 ... K)
    hazard           optional float64 [m, K], P(infiltration at k | none before k)
    ensemble         optional float64 [m, N, K], per-route cumulative probabilities (for CRPS)
    ensemble_weight  optional float64 [m, N], route weights, rows summing to 1
    meta             one row per trigger (META_COLUMNS, `time` is the trigger time)
    infiltrated_now  optional bool [m], the infiltration state holds in the window ending at the trigger
    p_inf_lower      optional float64 [m, K], lower edge of the forecast's uncertainty band
    p_inf_upper      optional float64 [m, K], upper edge of the band (p_inf_lower <= p_inf_upper)
    safe_horizon     optional int64 [m], steps (0 ... K) over which the model reports its forecast as reliable
    alert_threshold  optional threshold on P_inf(K) chosen on validation data (alert when P_inf(K) >= it)
    """

    p_inf: np.ndarray
    window_seconds: float
    event_step: np.ndarray
    observed_steps: np.ndarray
    meta: pd.DataFrame
    hazard: np.ndarray | None = None
    ensemble: np.ndarray | None = None
    ensemble_weight: np.ndarray | None = None
    infiltrated_now: np.ndarray | None = None
    p_inf_lower: np.ndarray | None = None
    p_inf_upper: np.ndarray | None = None
    safe_horizon: np.ndarray | None = None
    alert_threshold: float | None = None

    def __post_init__(self) -> None:
        self.p_inf = _f64("p_inf", self.p_inf, 2)
        _prob("p_inf", self.p_inf)
        m, k = self.p_inf.shape
        if k < 1:
            raise InvariantViolation("p_inf needs at least one step")
        if np.any(np.diff(self.p_inf, axis=1) < -_TOL):
            raise InvariantViolation("p_inf must be non-decreasing in k (it is a cumulative probability)")
        if not self.window_seconds > 0:
            raise InvariantViolation("window_seconds must be positive")
        self.event_step = np.asarray(self.event_step, dtype=np.int64)
        self.observed_steps = np.asarray(self.observed_steps, dtype=np.int64)
        if self.event_step.shape != (m,) or self.observed_steps.shape != (m,):
            raise InvariantViolation("event_step and observed_steps must have one entry per trigger")
        if np.any((self.event_step < 0) | (self.event_step > k)):
            raise InvariantViolation("event_step must lie in 0 ... K")
        if np.any((self.observed_steps < 0) | (self.observed_steps > k)):
            raise InvariantViolation("observed_steps must lie in 0 ... K")
        if np.any((self.event_step > 0) & (self.event_step > self.observed_steps)):
            raise InvariantViolation("an event cannot fall after the last observed step")
        if self.hazard is not None:
            self.hazard = _f64("hazard", self.hazard, 2)
            _prob("hazard", self.hazard)
            if self.hazard.shape != (m, k):
                raise InvariantViolation("hazard must have the shape of p_inf")
        if self.ensemble is not None:
            self.ensemble = _f64("ensemble", self.ensemble, 3)
            _prob("ensemble", self.ensemble)
            if self.ensemble.shape[0] != m or self.ensemble.shape[2] != k:
                raise InvariantViolation("ensemble must be [m, N, K]")
            if np.any(np.diff(self.ensemble, axis=2) < -_TOL):
                raise InvariantViolation("ensemble members must be non-decreasing in k")
            n_routes = self.ensemble.shape[1]
            w = (np.full((m, n_routes), 1.0 / n_routes) if self.ensemble_weight is None
                 else _f64("ensemble_weight", self.ensemble_weight, 2))
            if w.shape != (m, n_routes) or np.any(w < -_TOL) or np.any(np.abs(w.sum(axis=1) - 1.0) > 1e-6):
                raise InvariantViolation("ensemble_weight must be [m, N], non-negative, rows summing to 1")
            self.ensemble_weight = w
        if self.infiltrated_now is not None:
            now = np.asarray(self.infiltrated_now)
            if now.shape != (m,) or not np.isin(now.astype(np.int64), (0, 1)).all():
                raise InvariantViolation("infiltrated_now must be [m] booleans")
            self.infiltrated_now = now.astype(bool)
        for name in ("p_inf_lower", "p_inf_upper"):
            value = getattr(self, name)
            if value is not None:
                arr = _f64(name, value, 2)
                _prob(name, arr)
                if arr.shape != (m, k):
                    raise InvariantViolation(f"{name} must have the shape of p_inf")
                setattr(self, name, arr)
        if (self.p_inf_lower is None) != (self.p_inf_upper is None):
            raise InvariantViolation("p_inf_lower and p_inf_upper are given together")
        if self.p_inf_lower is not None and self.p_inf_upper is not None and np.any(self.p_inf_lower > self.p_inf_upper + _TOL):
            raise InvariantViolation("p_inf_lower must not exceed p_inf_upper")
        if self.safe_horizon is not None:
            sh = np.asarray(self.safe_horizon, dtype=np.int64)
            if sh.shape != (m,) or np.any((sh < 0) | (sh > k)):
                raise InvariantViolation("safe_horizon must be [m] with values in 0 ... K")
            self.safe_horizon = sh
        if self.alert_threshold is not None and not 0.0 <= float(self.alert_threshold) <= 1.0:
            raise InvariantViolation("alert_threshold must lie in [0, 1]")
        validate_meta(self.meta, m)

    @property
    def horizon(self) -> int:
        """K, the number of forecast steps."""
        return int(self.p_inf.shape[1])

    def outcome(self, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Binary outcome of the event 'infiltration within k steps' and the mask where it is known.

        Known when an event occurred at a step <= k, or when at least k steps were observed. A trigger
        censored before step k without an event has an unknown outcome and is excluded.
        """
        if not 1 <= k <= self.horizon:
            raise ValueError(f"k must lie in 1 ... {self.horizon}")
        happened = (self.event_step > 0) & (self.event_step <= k)
        known = happened | (self.observed_steps >= k)
        return happened.astype(np.int64), known


@dataclass
class StagePredictions:
    """Attack-stage posterior per unit (MITRE ATT&CK tactics plus "none", models/vocab.py).

    probs        float64 [n, S], rows summing to 1
    label        int64 [n], true stage code, -1 unknown
    stage_names  names of the S classes, in code order
    meta         one row per unit
    """

    probs: np.ndarray
    label: np.ndarray
    stage_names: tuple[str, ...]
    meta: pd.DataFrame

    def __post_init__(self) -> None:
        self.probs = _f64("probs", self.probs, 2)
        _prob("probs", self.probs)
        n, s = self.probs.shape
        if len(self.stage_names) != s:
            raise InvariantViolation("stage_names must name every class")
        if n and np.any(np.abs(self.probs.sum(axis=1) - 1.0) > 1e-6):
            raise InvariantViolation("each row of probs must sum to 1")
        self.label = np.asarray(self.label, dtype=np.int64)
        if self.label.shape != (n,) or np.any((self.label < -1) | (self.label >= s)):
            raise InvariantViolation("label must be [n] with values in -1 ... S-1")
        validate_meta(self.meta, n)


@dataclass
class StateForecastPredictions:
    """Forecast of future network-state features (the world model's P(S_t+k | S_t), scored directly).

    predicted      float64 [n, H, D], forecast feature vectors (standardised units)
    observed       float64 [n, H, D], the feature vectors that occurred (any value where masked out)
    mask           bool [n, H, D], True where the observed value exists and is scored
    horizons       int64 [H], the forecast offsets in windows (for example 1, K/2, K)
    feature_names  names of the D features
    meta           one row per forecast origin
    variance       optional float64 [n, H, D], predictive variance of each forecast (>= 0 where scored)
    samples        optional float64 [n, S, H, D], S predictive samples (finite where scored)
    current        optional float64 [n, D], the features of the window at the forecast origin
    current_mask   optional bool [n, D], True where `current` exists (all True when omitted)
    """

    predicted: np.ndarray
    observed: np.ndarray
    mask: np.ndarray
    horizons: np.ndarray
    feature_names: tuple[str, ...]
    meta: pd.DataFrame
    variance: np.ndarray | None = None
    samples: np.ndarray | None = None
    current: np.ndarray | None = None
    current_mask: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.mask = np.asarray(self.mask, dtype=bool)
        self.predicted = _f64("predicted", self.predicted, 3)
        observed = np.asarray(self.observed, dtype=np.float64)
        if observed.shape != self.predicted.shape or self.mask.shape != self.predicted.shape:
            raise InvariantViolation("predicted, observed and mask must share the shape [n, H, D]")
        if not np.all(np.isfinite(observed[self.mask])):
            raise InvariantViolation("observed values must be finite wherever mask is True")
        self.observed = np.where(self.mask, observed, 0.0)
        self.horizons = np.asarray(self.horizons, dtype=np.int64)
        n, h, d = self.predicted.shape
        if self.horizons.shape != (h,) or np.any(self.horizons < 1):
            raise InvariantViolation("horizons must be [H] positive offsets")
        if len(self.feature_names) != d:
            raise InvariantViolation("feature_names must name every feature")
        if self.variance is not None:
            var = np.asarray(self.variance, dtype=np.float64)
            if var.shape != (n, h, d):
                raise InvariantViolation("variance must have the shape [n, H, D]")
            scored = var[self.mask]
            if not np.all(np.isfinite(scored)) or np.any(scored < 0):
                raise InvariantViolation("variance must be finite and non-negative wherever mask is True")
            self.variance = np.where(self.mask, var, 0.0)
        if self.samples is not None:
            smp = np.asarray(self.samples, dtype=np.float64)
            if smp.ndim != 4 or smp.shape[0] != n or smp.shape[2:] != (h, d) or smp.shape[1] < 1:
                raise InvariantViolation("samples must have the shape [n, S, H, D] with S >= 1")
            if not np.all(np.isfinite(smp[np.broadcast_to(self.mask[:, None], smp.shape)])):
                raise InvariantViolation("samples must be finite wherever mask is True")
            self.samples = np.where(self.mask[:, None], smp, 0.0)
        if self.current is not None:
            cur = np.asarray(self.current, dtype=np.float64)
            cmask = (np.ones((n, d), dtype=bool) if self.current_mask is None
                     else np.asarray(self.current_mask, dtype=bool))
            if cur.shape != (n, d) or cmask.shape != (n, d):
                raise InvariantViolation("current and current_mask must have the shape [n, D]")
            if not np.all(np.isfinite(cur[cmask])):
                raise InvariantViolation("current must be finite wherever current_mask is True")
            self.current, self.current_mask = np.where(cmask, cur, 0.0), cmask
        elif self.current_mask is not None:
            raise InvariantViolation("current_mask is given only with current")
        validate_meta(self.meta, n)


@dataclass
class TimeToEventPredictions:
    """Time-to-infiltration forecast per trigger, for survival metrics.

    risk            float64 [m], a score that is higher when the event is expected sooner
    survival        float64 [m, T], S(t) = P(no infiltration by time t after the trigger), non-increasing
    time_grid       float64 [T], increasing times in seconds after the trigger
    event_time      float64 [m], seconds from trigger to the event, or to censoring
    event_observed  bool [m], True if the event was observed, False if censored at event_time
    meta            one row per trigger
    """

    risk: np.ndarray
    survival: np.ndarray
    time_grid: np.ndarray
    event_time: np.ndarray
    event_observed: np.ndarray
    meta: pd.DataFrame

    def __post_init__(self) -> None:
        self.risk = _f64("risk", self.risk, 1)
        self.survival = _f64("survival", self.survival, 2)
        _prob("survival", self.survival)
        self.time_grid = _f64("time_grid", self.time_grid, 1)
        self.event_time = _f64("event_time", self.event_time, 1)
        self.event_observed = np.asarray(self.event_observed, dtype=bool)
        m = self.risk.shape[0]
        if self.survival.shape != (m, self.time_grid.shape[0]):
            raise InvariantViolation("survival must be [m, T] with T = len(time_grid)")
        if np.any(np.diff(self.time_grid) <= 0) or (self.time_grid.size and self.time_grid[0] < 0):
            raise InvariantViolation("time_grid must be increasing and non-negative")
        if np.any(np.diff(self.survival, axis=1) > _TOL):
            raise InvariantViolation("survival curves must be non-increasing")
        if self.event_time.shape != (m,) or self.event_observed.shape != (m,) or np.any(self.event_time < 0):
            raise InvariantViolation("event_time and event_observed must be [m], event_time >= 0")
        validate_meta(self.meta, m)


@dataclass(frozen=True)
class PathStep:
    """One step of an attack path: an ATT&CK stage code and the entity it acts on (-1 if unknown)."""

    stage: int
    entity: int


@dataclass
class PathPredictions:
    """Imagined attack paths per trigger against the path that unfolded.

    predicted   per trigger, a list of (probability, path) pairs, best first; a path is a tuple of PathStep
    realised    per trigger, the path that unfolded (empty tuple when no attack followed)
    meta        one row per trigger
    true_value  optional, per trigger one finite value per predicted route (same order): the route's value
                under the true dynamics of a simulated world, for ordinal safety (protocol P6)
    """

    predicted: list[list[tuple[float, tuple[PathStep, ...]]]]
    realised: list[tuple[PathStep, ...]]
    meta: pd.DataFrame
    true_value: list[list[float]] | None = None

    def __post_init__(self) -> None:
        m = len(self.predicted)
        if len(self.realised) != m:
            raise InvariantViolation("predicted and realised must have one entry per trigger")
        for routes in self.predicted:
            probs = [p for p, _ in routes]
            if any(not 0.0 <= p <= 1.0 + _TOL for p in probs):
                raise InvariantViolation("route probabilities must lie in [0, 1]")
            if sum(probs) > 1.0 + 1e-6:
                raise InvariantViolation("route probabilities of one trigger must sum to at most 1")
            if any(probs[i] < probs[i + 1] - _TOL for i in range(len(probs) - 1)):
                raise InvariantViolation("routes must be ordered best first")
        if self.true_value is not None:
            if len(self.true_value) != m:
                raise InvariantViolation("true_value must have one entry per trigger")
            for vals, routes in zip(self.true_value, self.predicted, strict=True):
                if len(vals) != len(routes) or not all(np.isfinite(float(v)) for v in vals):
                    raise InvariantViolation("true_value must give one finite value per predicted route")
            self.true_value = [[float(v) for v in vals] for vals in self.true_value]
        validate_meta(self.meta, m)


@dataclass
class EpisodeTable:
    """Labelled attack episodes, for lead time and onset metrics.

    Columns of `frame`: episode (str id), dataset, network, family, novelty, entity (target entity id),
    start (epoch seconds, first malicious activity), completion (epoch seconds, the time at which the
    episode reaches its infiltration state, AS-18).
    """

    frame: pd.DataFrame

    REQUIRED: tuple[str, ...] = ("episode", "dataset", "network", "family", "novelty", "entity", "start", "completion")

    def __post_init__(self) -> None:
        missing = [c for c in self.REQUIRED if c not in self.frame.columns]
        if missing:
            raise InvariantViolation(f"episode table is missing columns {missing}")
        start = self.frame["start"].to_numpy(dtype=np.float64)
        completion = self.frame["completion"].to_numpy(dtype=np.float64)
        if np.any(completion < start):
            raise InvariantViolation("an episode cannot complete before it starts")
        if self.frame["episode"].duplicated().any():
            raise InvariantViolation("episode ids must be unique")


@dataclass
class ForensicPredictions:
    """Forensic replay of labelled incidents (protocol P7), one row of `meta` per incident.

    onset_pred         float64 [I, S], predicted onset time (epoch seconds) of each stage, NaN if not reported
    onset_true         float64 [I, S], annotated onset time of each stage, NaN if the incident lacks the stage
    stage_names        names of the S stage columns (codes of models/vocab.py STAGES, in code order)
    patient_zero       int64 [I, R], entity ids ranked best first, -1 as padding
    patient_zero_true  int64 [I], the annotated first compromised entity, -1 when not annotated
    narrative_pred     float64 [P, 4], rows (incident index, stage code, entity, time) of the reconstruction
    narrative_true     float64 [Q, 4], rows (incident index, stage code, entity, time) of the annotation
    meta               one row per incident (META_COLUMNS; `time` is the incident start)
    """

    onset_pred: np.ndarray
    onset_true: np.ndarray
    stage_names: tuple[str, ...]
    patient_zero: np.ndarray
    patient_zero_true: np.ndarray
    narrative_pred: np.ndarray
    narrative_true: np.ndarray
    meta: pd.DataFrame

    def __post_init__(self) -> None:
        self.onset_pred = np.asarray(self.onset_pred, dtype=np.float64)
        self.onset_true = np.asarray(self.onset_true, dtype=np.float64)
        if self.onset_pred.ndim != 2 or self.onset_true.shape != self.onset_pred.shape:
            raise InvariantViolation("onset_pred and onset_true must share the shape [I, S]")
        n_inc, n_st = self.onset_pred.shape
        if len(self.stage_names) != n_st:
            raise InvariantViolation("stage_names must name every onset column")
        if np.any(np.isinf(self.onset_pred)) or np.any(np.isinf(self.onset_true)):
            raise InvariantViolation("onset times must be finite or NaN")
        self.patient_zero = np.asarray(self.patient_zero, dtype=np.int64)
        if self.patient_zero.ndim != 2 or self.patient_zero.shape[0] != n_inc:
            raise InvariantViolation("patient_zero must be [I, R]")
        self.patient_zero_true = np.asarray(self.patient_zero_true, dtype=np.int64)
        if self.patient_zero_true.shape != (n_inc,):
            raise InvariantViolation("patient_zero_true must be [I]")
        for name in ("narrative_pred", "narrative_true"):
            arr = np.asarray(getattr(self, name), dtype=np.float64).reshape(-1, 4)
            if not np.all(np.isfinite(arr)):
                raise InvariantViolation(f"{name} must be finite")
            if arr.size and (np.any(arr[:, :3] != np.round(arr[:, :3])) or np.any((arr[:, 0] < 0) | (arr[:, 0] >= n_inc))):
                raise InvariantViolation(f"{name} rows must be (incident index, stage code, entity, time) with valid integers")
            if arr.size and np.any((arr[:, 1] < 0) | (arr[:, 1] >= n_st)):
                raise InvariantViolation(f"{name} stage codes must lie in 0 ... S-1")
            setattr(self, name, arr)
        validate_meta(self.meta, n_inc)


@dataclass
class OperationsMeasurements:
    """Operational measurements of one sustained run (protocol P8).

    latency_s      float64 [n], time from the arrival of a state update to the update of the Environment, s
    arrival_time   float64 [n], epoch seconds of each arrival, non-decreasing
    memory_time    float64 [T], times of the retained-memory samples, increasing
    memory_bytes   float64 [T], retained memory at each sample, bytes
    trigger_s      float64 [k], duration of each complete Forecaster trigger, s (may be empty)
    telemetry      telemetry level of the run, for example "full" or "flow_only"
    offered_rate   offered state-update rate, updates per second
    passes         thinking passes R of TSTCT and TAAFT used by the run (D-44)
    host           free-form description of the host
    dataset        dataset whose stream was replayed
    stage_latency_s   optional, pipeline stage -> float64 [n_s] measured durations of that stage, s
    flops_per_update  optional floating-point operations per state update (counted or profiled)
    peak_memory_bytes optional peak accelerator memory of the run, bytes
    accelerator       description of the accelerators used
    """

    latency_s: np.ndarray
    arrival_time: np.ndarray
    memory_time: np.ndarray
    memory_bytes: np.ndarray
    trigger_s: np.ndarray
    telemetry: str
    offered_rate: float
    passes: int
    host: str = ""
    dataset: str = ""
    stage_latency_s: dict[str, np.ndarray] | None = None
    flops_per_update: float | None = None
    peak_memory_bytes: float | None = None
    accelerator: str = ""

    def __post_init__(self) -> None:
        self.latency_s = _f64("latency_s", self.latency_s, 1)
        self.arrival_time = _f64("arrival_time", self.arrival_time, 1)
        if self.arrival_time.shape != self.latency_s.shape:
            raise InvariantViolation("latency_s and arrival_time must have one entry per state update")
        if np.any(self.latency_s < 0) or np.any(np.diff(self.arrival_time) < 0):
            raise InvariantViolation("latencies must be >= 0 and arrivals non-decreasing")
        self.memory_time = _f64("memory_time", self.memory_time, 1)
        self.memory_bytes = _f64("memory_bytes", self.memory_bytes, 1)
        if self.memory_time.shape != self.memory_bytes.shape or np.any(np.diff(self.memory_time) <= 0):
            raise InvariantViolation("memory samples must be paired and their times increasing")
        if np.any(self.memory_bytes < 0):
            raise InvariantViolation("memory_bytes must be >= 0")
        self.trigger_s = _f64("trigger_s", self.trigger_s, 1)
        if np.any(self.trigger_s < 0):
            raise InvariantViolation("trigger durations must be >= 0")
        if not self.telemetry:
            raise InvariantViolation("telemetry must name the telemetry level")
        if not float(self.offered_rate) > 0:
            raise InvariantViolation("offered_rate must be positive")
        if int(self.passes) < 1:
            raise InvariantViolation("passes must be >= 1")
        if self.stage_latency_s is not None:
            stages = {}
            for name, value in self.stage_latency_s.items():
                arr = _f64(f"stage_latency_s[{name}]", value, 1)
                if not name or np.any(arr < 0):
                    raise InvariantViolation("stage latencies need a stage name and non-negative durations")
                stages[str(name)] = arr
            self.stage_latency_s = stages
        for name in ("flops_per_update", "peak_memory_bytes"):
            value = getattr(self, name)
            if value is not None and not (np.isfinite(float(value)) and float(value) >= 0):
                raise InvariantViolation(f"{name} must be finite and non-negative")


@dataclass
class ArenaRun:
    """A defender agent trained and evaluated in a cyber-defence environment (protocol P-CW), one seed.

    environment         environment id, for example "cyberwheel"
    steps               int64 [T], environment steps at the evaluation points, increasing, >= 0
    returns             float64 [T, E], returns of E evaluation episodes at each evaluation point
    control_returns     float64 [E0], evaluation returns of the control policy in the same environment
    final_returns       float64 [E1], evaluation returns of the final policy
    conditions          generalisation condition -> float64 [E_c], final-policy returns under that condition
    control_conditions  condition -> float64 [E_c'], control returns under that condition (same keys)
    condition_kinds     condition -> "held_out_strategy", "network_size" or "observation"
    """

    environment: str
    steps: np.ndarray
    returns: np.ndarray
    control_returns: np.ndarray
    final_returns: np.ndarray
    conditions: dict[str, np.ndarray] = field(default_factory=dict)
    control_conditions: dict[str, np.ndarray] = field(default_factory=dict)
    condition_kinds: dict[str, str] = field(default_factory=dict)

    KINDS: ClassVar[tuple[str, ...]] = ("held_out_strategy", "network_size", "observation")

    def __post_init__(self) -> None:
        if not self.environment:
            raise InvariantViolation("environment must name the arena")
        self.steps = np.asarray(self.steps, dtype=np.int64)
        if self.steps.ndim != 1 or self.steps.size == 0 or np.any(self.steps < 0) or np.any(np.diff(self.steps) <= 0):
            raise InvariantViolation("steps must be a non-empty increasing vector of non-negative step counts")
        self.returns = _f64("returns", self.returns, 2)
        if self.returns.shape[0] != self.steps.size or self.returns.shape[1] < 1:
            raise InvariantViolation("returns must be [T, E] with one row per evaluation point and E >= 1")
        self.control_returns = _f64("control_returns", self.control_returns, 1)
        self.final_returns = _f64("final_returns", self.final_returns, 1)
        if self.control_returns.size == 0 or self.final_returns.size == 0:
            raise InvariantViolation("control_returns and final_returns need at least one episode each")
        if set(self.conditions) != set(self.control_conditions) or set(self.conditions) != set(self.condition_kinds):
            raise InvariantViolation("every condition needs control returns and a kind")
        self.conditions = {str(k): _f64(f"conditions[{k}]", v, 1) for k, v in self.conditions.items()}
        self.control_conditions = {str(k): _f64(f"control_conditions[{k}]", v, 1) for k, v in self.control_conditions.items()}
        for k, kind in self.condition_kinds.items():
            if kind not in self.KINDS:
                raise InvariantViolation(f"condition {k!r} has unknown kind {kind!r}; known: {self.KINDS}")
            if self.conditions[k].size == 0 or self.control_conditions[k].size == 0:
                raise InvariantViolation(f"condition {k!r} needs at least one episode for the agent and the control")


@dataclass
class ExplanationCurves:
    """Deletion and insertion curves of one explanation method, one row per explained prediction.

    method            explanation method, for example "expected_gradients", "lime" or "attention"
    fractions         float64 [F], fraction of the input features deleted or inserted, increasing from 0 to 1
    deletion          float64 [n, F], model output after deleting the top fraction of features by attribution
    insertion         float64 [n, F], model output after inserting the top fraction into the baseline input
    random_deletion   optional float64 [n, F], the same with features in random order (mean over orders)
    random_insertion  optional float64 [n, F]
    meta              one row per explained prediction
    """

    method: str
    fractions: np.ndarray
    deletion: np.ndarray
    insertion: np.ndarray
    meta: pd.DataFrame
    random_deletion: np.ndarray | None = None
    random_insertion: np.ndarray | None = None

    def __post_init__(self) -> None:
        if not self.method:
            raise InvariantViolation("method must name the explanation method")
        self.fractions = _f64("fractions", self.fractions, 1)
        fr = self.fractions
        if fr.size < 2 or abs(fr[0]) > _TOL or abs(fr[-1] - 1.0) > _TOL or np.any(np.diff(fr) <= 0):
            raise InvariantViolation("fractions must increase from 0 to 1")
        n = len(self.meta)
        for name in ("deletion", "insertion", "random_deletion", "random_insertion"):
            value = getattr(self, name)
            if value is None:
                continue
            arr = _f64(name, value, 2)
            if arr.shape != (n, fr.size):
                raise InvariantViolation(f"{name} must be [n, F] with n = rows of meta and F = len(fractions)")
            setattr(self, name, arr)
        if (self.random_deletion is None) != (self.random_insertion is None):
            raise InvariantViolation("random_deletion and random_insertion are given together")
        validate_meta(self.meta, n)


@dataclass
class ModelOutputs:
    """Everything one model produced on one evaluation protocol, ready for scoring.

    model      model name, for example "nagahana", "logistic_regression", "flowtransformer"
    protocol   protocol id from the evaluation design (P1 ... P8)
    seed       random seed of the run
    config     free-form run configuration (stored with the outputs for reproducibility)
    component  model-specific diagnostics, name -> array (for example per-lens energies)
    forensics  optional forensic replay record (protocol P7)
    operations optional operational measurements (protocol P8)
    arena      optional arena run of a defender agent (protocol P-CW)
    explanations optional deletion and insertion curves, one record per explanation method
    """

    model: str
    protocol: str
    seed: int = 0
    detection: DetectionPredictions | None = None
    forecast: ForecastPredictions | None = None
    stage: StagePredictions | None = None
    state_forecast: StateForecastPredictions | None = None
    time_to_event: TimeToEventPredictions | None = None
    paths: PathPredictions | None = None
    episodes: EpisodeTable | None = None
    config: dict[str, Any] = field(default_factory=dict)
    component: dict[str, np.ndarray] = field(default_factory=dict)
    forensics: ForensicPredictions | None = None
    operations: OperationsMeasurements | None = None
    arena: ArenaRun | None = None
    explanations: list[ExplanationCurves] | None = None

    def tasks(self) -> tuple[str, ...]:
        """Names of the prediction records present."""
        names = ("detection", "forecast", "stage", "state_forecast", "time_to_event", "paths", "episodes",
                 "forensics", "operations", "arena", "explanations")
        return tuple(n for n in names if getattr(self, n) is not None)


def _pack_meta(prefix: str, meta: pd.DataFrame, out: dict[str, np.ndarray]) -> None:
    # Store each metadata column as an array; strings become fixed-width unicode arrays.
    out[f"{prefix}.meta.__columns__"] = np.asarray(list(meta.columns), dtype=str)
    for col in meta.columns:
        values = meta[col].to_numpy()
        out[f"{prefix}.meta.{col}"] = values.astype(str) if values.dtype == object else values


def _unpack_meta(prefix: str, data: Any) -> pd.DataFrame:
    cols = [str(c) for c in data[f"{prefix}.meta.__columns__"]]
    return pd.DataFrame({c: data[f"{prefix}.meta.{c}"] for c in cols})


def save_outputs(outputs: ModelOutputs, path: str | Path) -> None:
    """Write a ModelOutputs bundle to one .npz file (paths and config as JSON strings)."""
    import json

    out: dict[str, np.ndarray] = {
        "model": np.asarray(outputs.model), "protocol": np.asarray(outputs.protocol),
        "seed": np.asarray(outputs.seed), "config": np.asarray(json.dumps(outputs.config, default=str)),
        "tasks": np.asarray(list(outputs.tasks()), dtype=str),
    }
    if outputs.detection is not None:
        d = outputs.detection
        out.update({"detection.score": d.score, "detection.label": d.label, "detection.unit": np.asarray(d.unit),
                    "detection.threshold": np.asarray(np.nan if d.threshold is None else d.threshold)})
        _pack_meta("detection", d.meta, out)
    if outputs.forecast is not None:
        f = outputs.forecast
        out.update({"forecast.p_inf": f.p_inf, "forecast.window_seconds": np.asarray(f.window_seconds),
                    "forecast.event_step": f.event_step, "forecast.observed_steps": f.observed_steps})
        for name in ("hazard", "ensemble", "ensemble_weight"):
            value = getattr(f, name)
            if value is not None:
                out[f"forecast.{name}"] = value
        # Optional fields added for the evaluation protocols: written only when present.
        for name in ("infiltrated_now", "p_inf_lower", "p_inf_upper", "safe_horizon"):
            value = getattr(f, name)
            if value is not None:
                out[f"forecast.{name}"] = value
        if f.alert_threshold is not None:
            out["forecast.alert_threshold"] = np.asarray(float(f.alert_threshold))
        _pack_meta("forecast", f.meta, out)
    if outputs.stage is not None:
        s = outputs.stage
        out.update({"stage.probs": s.probs, "stage.label": s.label,
                    "stage.stage_names": np.asarray(s.stage_names, dtype=str)})
        _pack_meta("stage", s.meta, out)
    if outputs.state_forecast is not None:
        x = outputs.state_forecast
        out.update({"state_forecast.predicted": x.predicted, "state_forecast.observed": x.observed,
                    "state_forecast.mask": x.mask, "state_forecast.horizons": x.horizons,
                    "state_forecast.feature_names": np.asarray(x.feature_names, dtype=str)})
        for name in ("variance", "samples", "current", "current_mask"):
            value = getattr(x, name)
            if value is not None:
                out[f"state_forecast.{name}"] = value
        _pack_meta("state_forecast", x.meta, out)
    if outputs.time_to_event is not None:
        t = outputs.time_to_event
        out.update({"tte.risk": t.risk, "tte.survival": t.survival, "tte.time_grid": t.time_grid,
                    "tte.event_time": t.event_time, "tte.event_observed": t.event_observed})
        _pack_meta("tte", t.meta, out)
    if outputs.paths is not None:
        p = outputs.paths
        enc = {"predicted": [[[pr, [[s.stage, s.entity] for s in path]] for pr, path in routes] for routes in p.predicted],
               "realised": [[[s.stage, s.entity] for s in path] for path in p.realised]}
        out["paths.json"] = np.asarray(json.dumps(enc))
        if p.true_value is not None:
            out["paths.true_value.json"] = np.asarray(json.dumps(p.true_value))
        _pack_meta("paths", p.meta, out)
    if outputs.episodes is not None:
        _pack_meta("episodes", outputs.episodes.frame, out)
    if outputs.forensics is not None:
        z = outputs.forensics
        out.update({"forensics.onset_pred": z.onset_pred, "forensics.onset_true": z.onset_true,
                    "forensics.stage_names": np.asarray(z.stage_names, dtype=str),
                    "forensics.patient_zero": z.patient_zero, "forensics.patient_zero_true": z.patient_zero_true,
                    "forensics.narrative_pred": z.narrative_pred, "forensics.narrative_true": z.narrative_true})
        _pack_meta("forensics", z.meta, out)
    if outputs.operations is not None:
        o = outputs.operations
        out.update({"operations.latency_s": o.latency_s, "operations.arrival_time": o.arrival_time,
                    "operations.memory_time": o.memory_time, "operations.memory_bytes": o.memory_bytes,
                    "operations.trigger_s": o.trigger_s, "operations.telemetry": np.asarray(o.telemetry),
                    "operations.offered_rate": np.asarray(float(o.offered_rate)),
                    "operations.passes": np.asarray(int(o.passes)), "operations.host": np.asarray(o.host),
                    "operations.dataset": np.asarray(o.dataset), "operations.accelerator": np.asarray(o.accelerator)})
        if o.stage_latency_s is not None:
            out["operations.stage_latency.__names__"] = np.asarray(list(o.stage_latency_s), dtype=str)
            for i, (_, value) in enumerate(o.stage_latency_s.items()):
                out[f"operations.stage_latency.{i}"] = value
        for name in ("flops_per_update", "peak_memory_bytes"):
            if getattr(o, name) is not None:
                out[f"operations.{name}"] = np.asarray(float(getattr(o, name)))
    if outputs.arena is not None:
        a = outputs.arena
        names = list(a.conditions)
        out.update({"arena.environment": np.asarray(a.environment), "arena.steps": a.steps, "arena.returns": a.returns,
                    "arena.control_returns": a.control_returns, "arena.final_returns": a.final_returns,
                    "arena.conditions.__names__": np.asarray(names, dtype=str),
                    "arena.condition_kinds": np.asarray(json.dumps(a.condition_kinds))})
        for i, name in enumerate(names):
            out[f"arena.conditions.{i}"] = a.conditions[name]
            out[f"arena.control_conditions.{i}"] = a.control_conditions[name]
    if outputs.explanations is not None:
        out["explanations.__count__"] = np.asarray(len(outputs.explanations))
        for i, e in enumerate(outputs.explanations):
            out.update({f"explanations.{i}.method": np.asarray(e.method), f"explanations.{i}.fractions": e.fractions,
                        f"explanations.{i}.deletion": e.deletion, f"explanations.{i}.insertion": e.insertion})
            if e.random_deletion is not None and e.random_insertion is not None:
                out[f"explanations.{i}.random_deletion"] = e.random_deletion
                out[f"explanations.{i}.random_insertion"] = e.random_insertion
            _pack_meta(f"explanations.{i}", e.meta, out)
    for name, value in outputs.component.items():
        out[f"component.{name}"] = np.asarray(value)
    # Object arrays never reach the file: metadata strings were converted to unicode arrays above.
    np.savez_compressed(Path(path), allow_pickle=False, **out)


def load_outputs(path: str | Path) -> ModelOutputs:
    """Read a bundle written by `save_outputs`; every record is re-validated on construction."""
    import json

    with np.load(Path(path), allow_pickle=False) as data:
        tasks = {str(t) for t in data["tasks"]}
        res = ModelOutputs(model=str(data["model"]), protocol=str(data["protocol"]), seed=int(data["seed"]),
                           config=json.loads(str(data["config"])))
        if "detection" in tasks:
            thr = float(data["detection.threshold"])
            res.detection = DetectionPredictions(score=data["detection.score"], label=data["detection.label"],
                                                 unit=str(data["detection.unit"]), meta=_unpack_meta("detection", data),
                                                 threshold=None if np.isnan(thr) else thr)
        if "forecast" in tasks:
            res.forecast = ForecastPredictions(
                p_inf=data["forecast.p_inf"], window_seconds=float(data["forecast.window_seconds"]),
                event_step=data["forecast.event_step"], observed_steps=data["forecast.observed_steps"],
                meta=_unpack_meta("forecast", data),
                hazard=data.get("forecast.hazard"), ensemble=data.get("forecast.ensemble"),
                ensemble_weight=data.get("forecast.ensemble_weight"),
                infiltrated_now=data.get("forecast.infiltrated_now"), p_inf_lower=data.get("forecast.p_inf_lower"),
                p_inf_upper=data.get("forecast.p_inf_upper"), safe_horizon=data.get("forecast.safe_horizon"),
                alert_threshold=(float(data["forecast.alert_threshold"]) if "forecast.alert_threshold" in data.files
                                 else None))
        if "stage" in tasks:
            res.stage = StagePredictions(probs=data["stage.probs"], label=data["stage.label"],
                                         stage_names=tuple(str(s) for s in data["stage.stage_names"]),
                                         meta=_unpack_meta("stage", data))
        if "state_forecast" in tasks:
            res.state_forecast = StateForecastPredictions(
                predicted=data["state_forecast.predicted"], observed=data["state_forecast.observed"],
                mask=data["state_forecast.mask"], horizons=data["state_forecast.horizons"],
                feature_names=tuple(str(s) for s in data["state_forecast.feature_names"]),
                meta=_unpack_meta("state_forecast", data),
                variance=data.get("state_forecast.variance"), samples=data.get("state_forecast.samples"),
                current=data.get("state_forecast.current"), current_mask=data.get("state_forecast.current_mask"))
        if "time_to_event" in tasks:
            res.time_to_event = TimeToEventPredictions(
                risk=data["tte.risk"], survival=data["tte.survival"], time_grid=data["tte.time_grid"],
                event_time=data["tte.event_time"], event_observed=data["tte.event_observed"],
                meta=_unpack_meta("tte", data))
        if "paths" in tasks:
            enc = json.loads(str(data["paths.json"]))
            predicted = [[(float(pr), tuple(PathStep(int(a), int(b)) for a, b in path)) for pr, path in routes]
                         for routes in enc["predicted"]]
            realised = [tuple(PathStep(int(a), int(b)) for a, b in path) for path in enc["realised"]]
            true_value = (json.loads(str(data["paths.true_value.json"])) if "paths.true_value.json" in data.files
                          else None)
            res.paths = PathPredictions(predicted=predicted, realised=realised, meta=_unpack_meta("paths", data),
                                        true_value=true_value)
        if "episodes" in tasks:
            res.episodes = EpisodeTable(frame=_unpack_meta("episodes", data))
        if "forensics" in tasks:
            res.forensics = ForensicPredictions(
                onset_pred=data["forensics.onset_pred"], onset_true=data["forensics.onset_true"],
                stage_names=tuple(str(s) for s in data["forensics.stage_names"]),
                patient_zero=data["forensics.patient_zero"], patient_zero_true=data["forensics.patient_zero_true"],
                narrative_pred=data["forensics.narrative_pred"], narrative_true=data["forensics.narrative_true"],
                meta=_unpack_meta("forensics", data))
        if "operations" in tasks:
            res.operations = OperationsMeasurements(
                latency_s=data["operations.latency_s"], arrival_time=data["operations.arrival_time"],
                memory_time=data["operations.memory_time"], memory_bytes=data["operations.memory_bytes"],
                trigger_s=data["operations.trigger_s"], telemetry=str(data["operations.telemetry"]),
                offered_rate=float(data["operations.offered_rate"]), passes=int(data["operations.passes"]),
                host=str(data["operations.host"]), dataset=str(data["operations.dataset"]),
                stage_latency_s=({str(nm): data[f"operations.stage_latency.{i}"]
                                  for i, nm in enumerate(data["operations.stage_latency.__names__"])}
                                 if "operations.stage_latency.__names__" in data.files else None),
                flops_per_update=(float(data["operations.flops_per_update"])
                                  if "operations.flops_per_update" in data.files else None),
                peak_memory_bytes=(float(data["operations.peak_memory_bytes"])
                                   if "operations.peak_memory_bytes" in data.files else None),
                accelerator=str(data["operations.accelerator"]) if "operations.accelerator" in data.files else "")
        if "arena" in tasks:
            names = [str(nm) for nm in data["arena.conditions.__names__"]]
            res.arena = ArenaRun(
                environment=str(data["arena.environment"]), steps=data["arena.steps"], returns=data["arena.returns"],
                control_returns=data["arena.control_returns"], final_returns=data["arena.final_returns"],
                conditions={nm: data[f"arena.conditions.{i}"] for i, nm in enumerate(names)},
                control_conditions={nm: data[f"arena.control_conditions.{i}"] for i, nm in enumerate(names)},
                condition_kinds={str(k): str(v) for k, v in json.loads(str(data["arena.condition_kinds"])).items()})
        if "explanations" in tasks:
            curves = []
            for i in range(int(data["explanations.__count__"])):
                key = f"explanations.{i}"
                curves.append(ExplanationCurves(
                    method=str(data[f"{key}.method"]), fractions=data[f"{key}.fractions"],
                    deletion=data[f"{key}.deletion"], insertion=data[f"{key}.insertion"],
                    meta=_unpack_meta(key, data), random_deletion=data.get(f"{key}.random_deletion"),
                    random_insertion=data.get(f"{key}.random_insertion")))
            res.explanations = curves
        for key in data.files:
            if key.startswith("component."):
                res.component[key[len("component."):]] = data[key]
    return res


def concat_detection(parts: Sequence[DetectionPredictions]) -> DetectionPredictions:
    """Concatenate detection records of the same unit type (for example several windows or seeds)."""
    if not parts:
        raise ValueError("nothing to concatenate")
    units = {p.unit for p in parts}
    if len(units) != 1:
        raise InvariantViolation(f"cannot concatenate different units {sorted(units)}")
    thresholds = {p.threshold for p in parts}
    return DetectionPredictions(score=np.concatenate([p.score for p in parts]),
                                label=np.concatenate([p.label for p in parts]), unit=parts[0].unit,
                                meta=pd.concat([p.meta for p in parts], ignore_index=True),
                                threshold=thresholds.pop() if len(thresholds) == 1 else None)
