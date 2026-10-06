"""Event-time statistics: inter-arrival times, burstiness, memory, count series.

Inter-arrival times
-------------------
For sorted event times t_1 <= ... <= t_N, tau_i = t_{i+1} - t_i. Simultaneous events (tau = 0) are kept:
they are what timestamps of a given resolution produce (AS-301), and their share is reported.

Burstiness (Goh and Barabasi, EPL 81:48002, 2008)
-------------------------------------------------
B = (sigma - mu) / (sigma + mu) of the n inter-arrival times: -1 periodic, 0 Poisson, -> 1 bursty.
For a finite sequence the coefficient of variation r = sigma / mu (population sigma) lies in
[0, sqrt(n - 1)], so B cannot reach 1. The finite-size form of Kim and Jo (Phys. Rev. E 94:032311, 2016)
    A_n = (sqrt(n + 1) r - sqrt(n - 1)) / ((sqrt(n + 1) - 2) r + sqrt(n - 1))
maps r = 0, r = sqrt(n - 1) to exactly -1 and 1 and tends to B for large n.

Memory coefficient (Goh and Barabasi 2008)
------------------------------------------
M = (1 / (n - 1)) sum_i (tau_i - m1)(tau_{i+1} - m2) / (s1 s2), with m1, s1 the mean and standard
deviation of tau_1 ... tau_{n-1} and m2, s2 those of tau_2 ... tau_n: the Pearson correlation of
consecutive gaps (M > 0: short gaps follow short gaps).

Count series
------------
Events binned on a fixed grid of `bin_seconds` anchored at the first event (or a given origin). A bin
without events is a true zero count (no event happened), unlike an absent field value (D-41).
"""

from __future__ import annotations

import numpy as np


def inter_arrival(times: np.ndarray) -> np.ndarray:
    """Gaps between consecutive finite event times (sorted first)."""
    t = np.asarray(times, dtype=np.float64).reshape(-1)
    t = np.sort(t[np.isfinite(t)])
    return np.diff(t)


def burstiness(iat: np.ndarray) -> dict[str, float]:
    """n, mean, cv, B, A_n (finite-size), memory M and the share of zero gaps (module docstring)."""
    tau = np.asarray(iat, dtype=np.float64).reshape(-1)
    tau = tau[np.isfinite(tau)]
    n = tau.size
    out = {"n": float(n), "mean": float("nan"), "cv": float("nan"), "burstiness": float("nan"),
           "burstiness_finite": float("nan"), "memory": float("nan"), "share_zero_gaps": float("nan")}
    if n == 0:
        return out
    mu, sd = float(tau.mean()), float(tau.std())
    out["mean"] = mu
    out["share_zero_gaps"] = float(np.mean(tau == 0.0))
    if mu > 0:
        r = sd / mu
        out["cv"] = r
        out["burstiness"] = (sd - mu) / (sd + mu)
        if n >= 2:
            a, b = np.sqrt(n + 1.0), np.sqrt(n - 1.0)
            out["burstiness_finite"] = float((a * r - b) / ((a - 2.0) * r + b))
    if n >= 3:
        x, y = tau[:-1], tau[1:]
        if x.std() > 0 and y.std() > 0:
            out["memory"] = float(np.corrcoef(x, y)[0, 1])
    return out


def count_series(times: np.ndarray, *, bin_seconds: float, origin: float | None = None,
                 end: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(bin start times, counts) of events on a fixed grid (module docstring)."""
    if bin_seconds <= 0:
        raise ValueError("bin_seconds must be > 0")
    t = np.asarray(times, dtype=np.float64).reshape(-1)
    t = t[np.isfinite(t)]
    if t.size == 0:
        return np.zeros(0), np.zeros(0, dtype=np.int64)
    o = float(t.min()) if origin is None else float(origin)
    e = float(t.max()) if end is None else float(end)
    nb = int(np.floor((e - o) / bin_seconds)) + 1
    idx = np.floor((t - o) / bin_seconds).astype(np.int64)
    ok = (idx >= 0) & (idx < nb)
    counts = np.bincount(idx[ok], minlength=nb)
    return o + bin_seconds * np.arange(nb), counts


def log_histogram(values: np.ndarray, *, bins_per_decade: int = 10) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(left edges, right edges, density) of the positive values on logarithmic bins."""
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v) & (v > 0)]
    if v.size == 0:
        return np.zeros(0), np.zeros(0), np.zeros(0)
    lo, hi = np.floor(np.log10(v.min())), np.ceil(np.log10(v.max()))
    hi = max(hi, lo + 1.0)
    edges = 10.0 ** np.linspace(lo, hi, int((hi - lo) * bins_per_decade) + 1)
    counts, _ = np.histogram(v, bins=edges)
    dens = counts / (v.size * np.diff(edges))
    return edges[:-1], edges[1:], dens


__all__ = ["burstiness", "count_series", "inter_arrival", "log_histogram"]
