"""Operational measurements on sustained streams (protocol P8): latency, throughput, memory growth.

Latency per state update is summarised by its median and 99th percentile (thesis section on operations
and forensics). Successive latencies of a loaded system are dependent (a queue carries delay forward),
so their intervals come from the stationary bootstrap over updates in arrival order with the
Politis-White block length of the latency series (resampling.py), using the weighted quantile of
_arrays.weighted_quantile.

Sustained throughput is the number of completed state updates per second over the run. Its interval
uses the method of batch means for steady-state simulation output (Schmeiser, Operations Research
30:556-568, 1982; Law, Simulation Modeling and Analysis, 5th ed., McGraw-Hill 2015, section 9.5.3):
the run is cut into M equal batches of time, the completion rate of each batch is one observation, and
the interval is the Student-t interval of their mean on M - 1 degrees of freedom.

Memory growth is the least-squares slope of retained memory against time, reported in GiB per day,
with a Newey-West (heteroskedasticity- and autocorrelation-consistent) standard error (Newey and West,
Econometrica 55:703-708, 1987) with Bartlett weights and the bandwidth floor(4 (T / 100)^(2/9)) of
Newey and West (Review of Economic Studies 61:631-653, 1994), because successive memory samples are
autocorrelated. The time of a complete Forecaster trigger is summarised by its median.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy import stats

from nagahana.core.errors import InvariantViolation
from nagahana.evaluation._arrays import weighted_quantile
from nagahana.evaluation.predictions import OperationsMeasurements
from nagahana.evaluation.resampling import BootstrapSettings, estimate, stationary_scheme

GIB = float(1 << 30)


def latency_quantiles(o: OperationsMeasurements, settings: BootstrapSettings, rng: np.random.Generator,
                      *, quantiles: tuple[float, ...] = (0.5, 0.99)) -> dict[str, tuple[float, float, float]]:
    """(value, low, high) of each latency quantile, in milliseconds."""
    lat = o.latency_s * 1000.0
    if lat.size < 2:
        raise InvariantViolation("latency quantiles need at least two state updates")
    scheme = stationary_scheme(np.zeros(lat.size), o.arrival_time, mean_block="auto", loss=lat)
    names = [f"latency_p{round(q * 100):02d}_ms" for q in quantiles]

    def stat(w: np.ndarray) -> np.ndarray:
        return np.column_stack([weighted_quantile(lat, w, q) for q in quantiles])

    est = estimate(stat, names, scheme, settings, rng)
    return {n: est.get(n) for n in names}


def throughput(o: OperationsMeasurements, *, batches: int = 20, confidence: float = 0.95) -> tuple[float, float, float]:
    """(rate, low, high) of completed state updates per second by batch means."""
    if batches < 2:
        raise ValueError("batches must be >= 2")
    done = o.arrival_time + o.latency_s
    t0, t1 = float(o.arrival_time.min()), float(done.max())
    span = t1 - t0
    if not span > 0:
        raise InvariantViolation("the run must span positive time")
    edges = np.linspace(t0, t1, batches + 1)
    counts, _ = np.histogram(done, bins=edges)
    per_batch = counts / np.diff(edges)
    mean = float(per_batch.mean())
    se = float(per_batch.std(ddof=1)) / math.sqrt(batches)
    q = float(stats.t.ppf(0.5 + confidence / 2.0, batches - 1))
    return float(done.size / span), mean - q * se, mean + q * se


def memory_growth(o: OperationsMeasurements, *, confidence: float = 0.95) -> tuple[float, float, float]:
    """(slope, low, high) of retained memory in GiB per day (OLS with a Newey-West standard error)."""
    t = o.memory_time
    y = o.memory_bytes / GIB
    if t.size < 3:
        raise InvariantViolation("memory growth needs at least three samples")
    x = (t - t.mean()) / 86400.0                                       # days, centred
    sxx = float(x @ x)
    slope = float(x @ (y - y.mean())) / sxx
    resid = y - y.mean() - slope * x
    u = x * resid                                                       # score contributions
    lags = int(math.floor(4.0 * (t.size / 100.0) ** (2.0 / 9.0)))
    s = float(u @ u)
    for k in range(1, min(lags, t.size - 1) + 1):
        s += 2.0 * (1.0 - k / (lags + 1.0)) * float(u[k:] @ u[:-k])
    se = math.sqrt(max(s, 0.0)) / sxx
    q = float(stats.t.ppf(0.5 + confidence / 2.0, t.size - 2))
    return slope, slope - q * se, slope + q * se


def quantile_interval(values: np.ndarray, q: float, *, confidence: float = 0.95) -> tuple[float, float, float]:
    """(quantile, low, high): the sample quantile and the distribution-free order-statistic interval.

    For n independent samples, [x_(l), x_(u)] with l = F^-1(alpha / 2) and u = F^-1(1 - alpha / 2) + 1 of
    Binomial(n, q) covers the q-quantile with probability >= confidence (Hahn and Meeker, Statistical
    Intervals, Wiley 1991, section 5.2); an end that the sample cannot reach is reported as -inf or +inf.
    """
    v = np.sort(np.asarray(values, dtype=np.float64))
    n = v.size
    if n == 0:
        return math.nan, math.nan, math.nan
    if not 0.0 < q < 1.0:
        raise ValueError("q must lie in (0, 1)")
    a = 1.0 - confidence
    lo_i = int(stats.binom.ppf(a / 2.0, n, q))
    hi_i = int(stats.binom.ppf(1.0 - a / 2.0, n, q)) + 1
    lo = float(v[lo_i - 1]) if lo_i >= 1 else -math.inf
    hi = float(v[hi_i - 1]) if hi_i <= n else math.inf
    return float(np.quantile(v, q)), lo, hi


def trigger_median(o: OperationsMeasurements, *, confidence: float = 0.95) -> tuple[float, float, float]:
    """(median, low, high) trigger time in seconds with the order-statistic interval of the median."""
    return quantile_interval(o.trigger_s, 0.5, confidence=confidence)


def operations_summary(o: OperationsMeasurements, settings: BootstrapSettings, rng: np.random.Generator,
                       *, batches: int = 20) -> dict[str, tuple[float, float, float]]:
    """Latency p50 and p99, throughput, memory growth and trigger time, each as (value, low, high)."""
    out: dict[str, tuple[float, float, float]] = dict(latency_quantiles(o, settings, rng))
    out["throughput_per_s"] = throughput(o, batches=batches, confidence=settings.confidence)
    out["memory_gib_per_day"] = memory_growth(o, confidence=settings.confidence)
    out["trigger_median_s"] = trigger_median(o, confidence=settings.confidence)
    return out


def relative_change(full: Any, reduced: Any) -> float:
    """(reduced - full) / full, the relative change of a metric between telemetry levels (NaN for full = 0)."""
    f, r = float(full), float(reduced)
    return (r - f) / f if f != 0 else math.nan
