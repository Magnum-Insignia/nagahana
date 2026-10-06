"""Early-warning indicators of an approaching transition: critical slowing down (D-56).

Purpose
-------
Architecture section 6 lists "Energy and entropy growth as an early warning". A network moving
towards an infiltration is a system approaching a change of regime; near such a transition the
dominant eigenvalue of the linearised dynamics approaches zero, so the state recovers ever more
slowly from perturbations ("critical slowing down"): variance rises, lag-1 autocorrelation rises
towards 1, the spectrum reddens and the recovery rate falls (Wissel, "A universal law of the
characteristic return time near thresholds", Oecologia 65:101, 1984; Scheffer et al., "Early-warning
signals for critical transitions", Nature 461:53, 2009). Skewness and kurtosis change when the
potential becomes asymmetric or the system flickers between states (Guttal and Jayaprakash,
"Changing skewness: an early warning signal of regime shifts in ecosystems", Ecology Letters 11:450,
2008). The pipeline follows Dakos et al., "Methods for detecting early warnings of critical
transitions in time series illustrated using simulated ecological data", PLoS ONE 7(7):e41010, 2012.
The series it reads are the thermodynamic and entropy trajectories of trajectory.py (energy, free
energy, entropies of the Gibbs ensembles, graph and traffic entropies).

Pipeline (one series x_0 ... x_{n-1} at times t_i; invalid samples are skipped, so the series is the
sequence of valid observations)
1. Detrending (AS-772): r_i = x_i - xhat_i with the Gaussian-kernel smoother
       xhat_i = sum_j K(i - j) x_j / sum_j K(i - j),   K(d) = exp(-d^2 / (2 h^2)),  |d| <= ceil(truncate h)
   "gaussian" sums over both sides (the retrospective smoother of Dakos et al. 2008, PNAS
   105(38):14308, and 2012) and serves offline analysis of a saved trajectory; "gaussian_causal" sums
   over j <= i only, so a residual never depends on a later sample (online use); "none" keeps x.
2. Rolling indicators on every full window of W residuals r_{i-W+1} ... r_i (AS-773), with
   mu the window mean and m_k = (1 / W) sum (r - mu)^k:
       variance           sum (r - mu)^2 / (W - 1)
       ar1                rho_1 = sum_{j=1}^{W-1} (r_j - mu)(r_{j+1} - mu) / sum_{j=1}^{W} (r_j - mu)^2
       skewness           m_3 / m_2^(3/2)
       kurtosis           m_4 / m_2^2  (Pearson; 3 for a Gaussian)
       return_rate        -log(rho_1) / dt for rho_1 > 0 (undefined otherwise; dt the mean sampling
                          interval of the window): the recovery rate lambda, because an AR(1) sampled
                          from dx = -lambda x dt + noise has coefficient exp(-lambda dt) (Scheffer et al.
                          2009, Box 1; Held and Kleinen, "Detection of climate system bifurcations by
                          degenerate fingerprinting", Geophysical Research Letters 31:L23207, 2004)
       spectral_ratio     mean periodogram power at f <= f_low over mean power at f >= f_high (cycles per
                          sample), Hann-tapered periodogram of the demeaned window; reddening raises it
                          (Kleinen, Held and Petschel-Held, Ocean Dynamics 53:53, 2003; Biggs, Carpenter
                          and Brock, "Turning back from the brink", PNAS 106(3):826, 2009)
       spectral_exponent  beta of I(f) ~ f^(-beta): minus the least-squares slope of log I on log f over
                          the Fourier frequencies k / W, k = 1 ... floor(W / 2)
       dfa                DFA-1 exponent: least-squares slope of log F(s) on log s,
                          F(s)^2 = mean over the boxes of size s inside the window of RSS_box / s, RSS_box
                          the residual sum of squares of a straight line fitted to the box's cumulative
                          sum (Peng et al., "Mosaic organization of DNA nucleotides", Physical Review E
                          49:1685, 1994; as an early warning: Livina and Lenton, Geophysical Research
                          Letters 34:L03712, 2007). White noise gives 0.5; an AR(1) with coefficient near 1
                          approaches 1.5 at scales below its correlation time.
   DFA boxes are aligned to absolute sample indices (box b covers samples b s ... b s + s - 1), not to
   the window start, so a complete box never changes while the window slides; the batch and the
   streaming forms both use this alignment. A box's residual sum of squares does not depend on the
   offset or on a linear term of the profile, so the box-local cumulative sum gives the same value as
   the profile of the whole window.
3. Trend (AS-774): Kendall's tau-b between time and an indicator, over the trailing `trend_window`
   valid values (online) or over the whole indicator series (offline); positive for a rising
   indicator (Kendall, "A new measure of rank correlation", Biometrika 30:81, 1938; tau-b, the tie
   correction: Kendall, Biometrika 33:239, 1945).
4. Significance (AS-775): the null hypothesis is a stationary linear process without a trend.
   Surrogates of the residual series with its autocorrelation: phase-randomised (Fourier amplitudes
   kept, phases uniform, the Nyquist term kept real: Theiler, Eubank, Longtin, Galdrikian and Farmer,
   "Testing for nonlinearity in time series: the method of surrogate data", Physica D 58:77, 1992) and
   fitted AR(1) (Yule-Walker phi = rho_1 of the whole residual series, innovation variance
   gamma_0 (1 - phi^2), started from the stationary law, so no burn-in is needed). Every surrogate
   passes through the same rolling indicators and trend statistic; the one-sided p-value in the
   expected direction is p = (1 + #{surrogate tau at least as extreme}) / (1 + n_surrogates) (North,
   Curtis and Sham, "A note on the calculation of empirical P values from Monte Carlo procedures",
   American Journal of Human Genetics 71:439, 2002). Two-sided indicators compare |tau|.
5. Composite scores and alarm (AS-776, AS-777): level score c(t) = mean_i d_i z_i(t),
   z_i = (x_i(t) - median_i) / scale_i with median and scale = 1.4826 MAD fitted on benign data
   (Rousseeuw and Croux, "Alternatives to the median absolute deviation", JASA 88:1273, 1993, for the
   consistency factor); d_i = +1 for a rising, -1 for a falling indicator, and |z_i| for a two-sided
   one. Trend score: mean_i d_i tau_i (|tau_i| for two-sided). The alarm threshold is the
   split-conformal quantile q = s_(k), k = ceil((n + 1)(1 - alpha)), of benign scores that were not
   used to fit the baselines (Vovk, Gammerman and Shafer, Algorithmic Learning in a Random World,
   Springer 2005; Angelopoulos and Bates, arXiv:2107.07511, 2021): a new benign score exchangeable
   with them exceeds q with probability at most alpha. With k > n the threshold is +inf (never fires).

Expected directions (`DIRECTIONS`): variance +1, ar1 +1, skewness 0 (two-sided), kurtosis +1,
return_rate -1, spectral_ratio +1, spectral_exponent +1, dfa +1.

Streaming (AS-778)
------------------
`StreamingEWS` updates every quantity with work per sample that does not grow with the length of the
stream: the causal kernel has L = ceil(truncate h) + 1 taps; the moments and the lag-1 product come
from running power sums of shifted residuals (O(1)); the spectral indicators from a sliding DFT of
the window's coefficients k = 1 ... floor(W / 2) with the Hann taper applied in frequency,
X_k^w = X_k / 2 - (X_{k-1} + X_{k+1}) / 4 with X_0 = 0 for the demeaned window (O(W / 2)); the DFA
from box statistics computed once per completed box (O(number of scales) amortised); the trend from a
sorted window of the last H indicator values (O(log H) comparisons, O(H) element moves). Running sums
and coefficients are recomputed exactly every `resync_every` samples, so rounding never accumulates.
Every value equals the batch functions on the same series (tested to 1e-9).

Precision (D-54): all indicators, statistics, scores and thresholds are float64.
"""

from __future__ import annotations

import bisect
import json
import math
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from nagahana.statphys.config import INDICATORS, EarlyWarningConfig

#: Expected direction of change of each indicator before a transition (module docstring).
DIRECTIONS: dict[str, int] = {"variance": 1, "ar1": 1, "skewness": 0, "kurtosis": 1, "return_rate": -1,
                              "spectral_ratio": 1, "spectral_exponent": 1, "dfa": 1}
#: Consistency factor of the median absolute deviation for a normal law (Rousseeuw and Croux 1993).
MAD_SCALE = 1.4826
assert set(DIRECTIONS) == set(INDICATORS)


def kernel_weights(bandwidth: float, truncate: float) -> np.ndarray:
    """Gaussian kernel taps w_d = exp(-d^2 / (2 h^2)) for lags d = 0 ... ceil(truncate h)."""
    if not (bandwidth > 0 and truncate > 0):
        raise ValueError("bandwidth and truncate must be > 0")
    lags = np.arange(int(math.ceil(truncate * bandwidth)) + 1, dtype=np.float64)
    return np.exp(-0.5 * (lags / bandwidth) ** 2)


def detrend(x: np.ndarray, *, method: str, bandwidth: float, truncate: float) -> tuple[np.ndarray, np.ndarray]:
    """(trend, residual) of a series by the Gaussian-kernel smoother (module docstring, step 1)."""
    v = np.asarray(x, dtype=np.float64)
    if v.ndim != 1 or not bool(np.all(np.isfinite(v))):
        raise ValueError("detrend needs a finite 1-D series")
    if method == "none":
        return np.zeros_like(v), v.copy()
    w = kernel_weights(bandwidth, truncate)
    n = v.shape[0]
    if method == "gaussian_causal":
        num = np.convolve(v, w)[:n]                                            # sum_{d >= 0} w_d x_{i-d}
        den = np.convolve(np.ones(n), w)[:n]
    elif method == "gaussian":
        sym = np.concatenate([w[:0:-1], w])                                    # lags -L+1 ... L-1
        half = w.shape[0] - 1
        num = np.convolve(v, sym)[half: half + n]
        den = np.convolve(np.ones(n), sym)[half: half + n]
    else:
        raise ValueError(f"unknown detrending {method!r}")
    trend = num / den
    return trend, v - trend


def _box_rss(box: np.ndarray) -> np.ndarray:
    """Residual sum of squares of the straight-line fit to the cumulative sum of each box: [..., s] -> [...]."""
    z = np.cumsum(box, axis=-1)
    s = box.shape[-1]
    j = np.arange(s, dtype=np.float64)
    jc = j - j.mean()
    zc = z - z.mean(axis=-1, keepdims=True)
    slope = (zc * jc).sum(-1) / (jc * jc).sum()
    res = zc - slope[..., None] * jc
    return (res * res).sum(-1)


def _hann(w: int) -> np.ndarray:
    # Periodic Hann taper: w_j = 1/2 - cos(2 pi j / W) / 2, j = 0 ... W-1.
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(w) / w)


def _spectral_bands(window: int, low_max: float, high_min: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Fourier frequencies k / W (k = 1 ... floor(W / 2)) and the masks of the two bands.
    k = np.arange(1, window // 2 + 1)
    f = k / window
    return f, f <= low_max, f >= high_min


def _slope(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    # Least-squares slope of y on x along the last axis (x shared).
    xc = x - x.mean()
    yc = y - y.mean(axis=-1, keepdims=True)
    return (yc * xc).sum(-1) / (xc * xc).sum()


def _spectral(power: np.ndarray, config: EarlyWarningConfig) -> tuple[np.ndarray, np.ndarray]:
    # Spectral ratio and exponent from periodogram ordinates [..., floor(W/2)].
    f, low, high = _spectral_bands(config.window, config.spectral_low_max, config.spectral_high_min)
    ok = np.all(power > 0, axis=-1)
    safe = np.where(power > 0, power, 1.0)
    ratio = safe[..., low].mean(-1) / safe[..., high].mean(-1)
    expo = -_slope(np.log(f), np.log(safe))
    return np.where(ok, ratio, np.nan), np.where(ok, expo, np.nan)


def _dfa_exponent(f2: np.ndarray, scales: np.ndarray) -> np.ndarray:
    # DFA exponent from F(s)^2 [..., n_scales]: slope of log F on log s.
    ok = np.all(f2 > 0, axis=-1)
    log_f = 0.5 * np.log(np.where(f2 > 0, f2, 1.0))
    return np.where(ok, _slope(np.log(scales.astype(np.float64)), log_f), np.nan)


def rolling_indicators(residual: np.ndarray, *, config: EarlyWarningConfig, times: np.ndarray | None = None
                       ) -> dict[str, np.ndarray]:
    """Indicators of every full window (module docstring, step 2): name -> [..., n], NaN before index W - 1.

    residual [..., n] (rows are independent series sharing the times); times [n] (default: the index).
    """
    r = np.asarray(residual, dtype=np.float64)
    single = r.ndim == 1
    rr = r[None] if single else r.reshape(-1, r.shape[-1])
    n = rr.shape[-1]
    w = config.window
    t = np.arange(n, dtype=np.float64) if times is None else np.asarray(times, dtype=np.float64)
    if t.shape != (n,):
        raise ValueError("times must be [n]")
    out = {name: np.full(rr.shape, np.nan, dtype=np.float64) for name in config.indicators}
    if n >= w:
        win = np.lib.stride_tricks.sliding_window_view(rr, w, axis=-1)        # [R, n - W + 1, W]
        mu = win.mean(-1, keepdims=True)
        d = win - mu
        s2 = (d * d).sum(-1)
        m2, m3, m4 = s2 / w, (d**3).mean(-1), (d**4).mean(-1)
        pos = s2 > 0
        safe2 = np.where(pos, s2, 1.0)
        rho = np.where(pos, (d[..., :-1] * d[..., 1:]).sum(-1) / safe2, np.nan)
        span = np.lib.stride_tricks.sliding_window_view(t, w)
        dt = (span[:, -1] - span[:, 0]) / (w - 1)
        vals: dict[str, np.ndarray] = {
            "variance": s2 / (w - 1),
            "ar1": rho,
            "skewness": np.where(pos, m3 / np.where(pos, m2, 1.0) ** 1.5, np.nan),
            "kurtosis": np.where(pos, m4 / np.where(pos, m2, 1.0) ** 2, np.nan),
            "return_rate": np.where((rho > 0) & (dt > 0), -np.log(np.where(rho > 0, rho, 1.0)) / np.where(dt > 0, dt, 1.0),
                                    np.nan),
        }
        if {"spectral_ratio", "spectral_exponent"} & set(config.indicators):
            spec = np.fft.rfft(d * _hann(w), axis=-1)[..., 1: w // 2 + 1]
            power = (spec.real**2 + spec.imag**2) / float((_hann(w) ** 2).sum())
            vals["spectral_ratio"], vals["spectral_exponent"] = _spectral(power, config)
        if "dfa" in config.indicators:
            vals["dfa"] = _dfa_batch(rr, config)[:, w - 1:]
        for name in config.indicators:
            out[name][:, w - 1:] = vals[name]
    return {k: (v[0] if single else v.reshape(r.shape)) for k, v in out.items()}


def _dfa_batch(rr: np.ndarray, config: EarlyWarningConfig) -> np.ndarray:
    # DFA exponent at every index of rows [R, n] with absolute box alignment (NaN before a full window).
    n = rr.shape[-1]
    w = config.window
    scales = np.asarray(config.dfa_scales, dtype=np.int64)
    ends = np.arange(n)
    f2 = np.full((rr.shape[0], n, scales.shape[0]), np.nan, dtype=np.float64)
    for si, s in enumerate(scales.tolist()):
        nb = n // s
        if nb == 0:
            continue
        rss = _box_rss(rr[:, : nb * s].reshape(rr.shape[0], nb, s)) / s       # [R, nb]
        csum = np.concatenate([np.zeros((rr.shape[0], 1)), np.cumsum(rss, axis=-1)], axis=-1)
        b_lo = -((-(ends - w + 1)) // s)                                      # ceil((i - W + 1) / s)
        b_hi = (ends + 1) // s - 1
        ok = (ends >= w - 1) & (b_hi >= b_lo)
        lo, hi = np.clip(b_lo, 0, nb), np.clip(b_hi + 1, 0, nb)
        cnt = np.where(ok, hi - lo, 0)
        f2[:, :, si] = np.where(ok & (cnt > 0), (csum[:, hi] - csum[:, lo]) / np.maximum(cnt, 1), np.nan)
    return _dfa_exponent(f2, scales)


def _dense_ranks(y: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Dense ranks (1-based) of the valid entries of each row, and the tie term n2 = sum t (t - 1) / 2.
    r_n, n = y.shape
    key = np.where(valid, y, np.inf)
    order = np.argsort(key, axis=1, kind="stable")
    sk = np.take_along_axis(key, order, axis=1)
    sv = np.take_along_axis(valid, order, axis=1)
    new = np.ones_like(sv)
    new[:, 1:] = sk[:, 1:] != sk[:, :-1]
    dense = np.cumsum(new, axis=1)
    ranks = np.zeros((r_n, n), dtype=np.int64)
    np.put_along_axis(ranks, order, dense, axis=1)
    group = dense - 1
    flat = (np.arange(r_n)[:, None] * n + group)[sv]
    counts = np.bincount(flat, minlength=r_n * n).reshape(r_n, n).astype(np.float64)
    return ranks, (counts * (counts - 1.0) / 2.0).sum(1)


def _discordant(ranks: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Discordant pairs D = #{i < j : y_i > y_j} per row (both valid), by a bottom-up merge sort.

    ranks long [R, n] (dense ranks of y), valid bool [R, n]. Rows are padded to a power of two with
    invalid entries. At the level of block size b, every merged block of size 2 b is ordered by
    (rank, origin) with origin 0 for its left half and 1 for its right half, so for a right-half
    element y_j the number of left-half elements ordered before it is #{left y_i <= y_j}; the pairs
    (left y_i > y_j) are the left-half total minus that count. Summed over all levels this counts each
    pair i < j exactly once (at the level where i and j first share a merged block). One integer
    argsort per level, all rows at once: O(R n log^2 n) work, vectorised.
    """
    r_n, n = ranks.shape
    size = 1
    while size < n:
        size *= 2
    rk = np.zeros((r_n, size), dtype=np.int64)
    ok = np.zeros((r_n, size), dtype=bool)
    rk[:, :n] = np.where(valid, ranks, 0)
    ok[:, :n] = valid
    pos = np.arange(size)
    rows = np.repeat(np.arange(r_n), size)
    span = 2 * (int(rk.max(initial=0)) + 1)                                    # room for 2 rank + origin
    d = np.zeros(r_n, dtype=np.int64)
    b = 1
    while b < size:
        merged = pos // (2 * b)                                                 # merged block of every position
        origin = (pos // b) % 2                                                 # 0 left half, 1 right half
        n_merged = size // (2 * b)
        block = (np.arange(r_n)[:, None] * n_merged + merged[None, :])          # [R, size] global block id
        key = (block * span + 2 * rk + origin[None, :]).ravel()
        order = np.argsort(key, kind="stable")
        o_block = block.ravel()[order]
        o_left = ((origin[None, :] == 0) & ok).ravel()[order]
        o_right = ((origin[None, :] == 1) & ok).ravel()[order]
        before = np.cumsum(o_left)                                              # left-valid elements up to here
        start = np.searchsorted(o_block, np.arange(r_n * n_merged), side="left")
        offset = np.concatenate([[0], before])[start]                           # count before each block
        total = np.bincount(o_block, weights=o_left, minlength=r_n * n_merged).astype(np.int64)
        contrib = np.where(o_right, total[o_block] - (before - offset[o_block]), 0)
        d += np.bincount(rows[order], weights=contrib, minlength=r_n).astype(np.int64)
        b *= 2
    return d


def trend_tau(y: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    """Kendall tau-b of each row of y [R, n] against its index (time), over the valid entries.

    With m valid entries, n0 = m (m - 1) / 2 pairs, n2 = sum_t t (t - 1) / 2 tied pairs of y (time has no
    ties) and D discordant pairs (`_discordant`), the concordant pairs are C = n0 - n2 - D, so
    S = C - D = n0 - n2 - 2 D and tau_b = S / sqrt(n0 (n0 - n2)). NaN when fewer than two valid entries
    or all of them tie. Equal to the exact O(n^2) `kendall_tau_b` (tested).
    """
    yy = np.atleast_2d(np.asarray(y, dtype=np.float64))
    ok = np.isfinite(yy) if valid is None else (np.atleast_2d(valid).astype(bool) & np.isfinite(yy))
    ranks, n2 = _dense_ranks(yy, ok)
    d = _discordant(ranks, ok).astype(np.float64)
    m_v = ok.sum(1).astype(np.float64)
    n0 = m_v * (m_v - 1.0) / 2.0
    s = n0 - n2 - 2.0 * d
    den = np.sqrt(n0 * (n0 - n2))
    return np.where((m_v >= 2) & (den > 0), s / np.where(den > 0, den, 1.0), np.nan)


def kendall_tau_b(x: np.ndarray, y: np.ndarray) -> float:
    """Kendall tau-b of two samples with ties in either (exact O(n^2); a reference for tests and small n)."""
    a = np.asarray(x, dtype=np.float64)
    b = np.asarray(y, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("x and y must be 1-D of equal length")
    sx = np.sign(a[None, :] - a[:, None])
    sy = np.sign(b[None, :] - b[:, None])
    iu = np.triu_indices(a.shape[0], k=1)
    s = float((sx[iu] * sy[iu]).sum())
    n0 = len(iu[0])
    tx = float((sx[iu] == 0).sum())
    ty = float((sy[iu] == 0).sum())
    den = math.sqrt((n0 - tx) * (n0 - ty))
    return s / den if den > 0 else math.nan


def trailing_tau(values: np.ndarray, horizon: int) -> np.ndarray:
    """Kendall trend over the last `horizon` valid values ending at each valid index: [n].

    NaN until `horizon` valid values exist and wherever the value itself is not valid (no indicator,
    no trend reading), exactly as the streaming `SlidingKendall`.
    """
    v = np.asarray(values, dtype=np.float64)
    ok = np.isfinite(v)
    pos = np.nonzero(ok)[0]
    out = np.full(v.shape[0], np.nan, dtype=np.float64)
    if pos.shape[0] < horizon:
        return out
    mat = np.lib.stride_tricks.sliding_window_view(v[pos], horizon)            # [m - H + 1, H]
    taus = trend_tau(mat)
    # The j-th valid value (0-based) closes the window of valid values j - H + 1 ... j.
    out[pos[horizon - 1:]] = taus
    return out


def phase_surrogates(x: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Phase-randomised surrogates [n, len(x)]: same periodogram, uniform phases, mean kept."""
    v = np.asarray(x, dtype=np.float64)
    m = v.shape[0]
    mu = v.mean()
    spec = np.fft.rfft(v - mu)
    amp = np.abs(spec)
    phases = rng.uniform(0.0, 2.0 * np.pi, size=(n, spec.shape[0]))
    phases[:, 0] = 0.0                                                         # the mean term (zero after demeaning)
    if m % 2 == 0:
        phases[:, -1] = np.where(spec[-1].real >= 0, 0.0, np.pi)               # the Nyquist term stays real
    return np.fft.irfft(amp[None, :] * np.exp(1j * phases), n=m, axis=-1) + mu


def ar1_surrogates(x: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Stationary AR(1) surrogates [n, len(x)] with the Yule-Walker fit of x (mean kept)."""
    v = np.asarray(x, dtype=np.float64)
    mu = v.mean()
    d = v - mu
    g0 = float((d * d).mean())
    phi = float((d[:-1] * d[1:]).sum() / (d * d).sum()) if g0 > 0 else 0.0
    phi = min(max(phi, -1.0 + 1e-12), 1.0 - 1e-12)
    sigma = math.sqrt(max(g0 * (1.0 - phi * phi), 0.0))
    out = np.empty((n, v.shape[0]), dtype=np.float64)
    out[:, 0] = rng.normal(0.0, math.sqrt(g0), size=n)                        # the stationary law
    eps = rng.normal(0.0, sigma, size=(n, v.shape[0]))
    for i in range(1, v.shape[0]):
        out[:, i] = phi * out[:, i - 1] + eps[:, i]
    return out + mu


def _p_value(observed: float, null: np.ndarray, direction: int) -> float:
    if not math.isfinite(observed):
        return math.nan
    null = null[np.isfinite(null)]
    if direction > 0:
        hits = int((null >= observed).sum())
    elif direction < 0:
        hits = int((null <= observed).sum())
    else:
        hits = int((np.abs(null) >= abs(observed)).sum())
    return (1 + hits) / (1 + null.shape[0])


@dataclass(frozen=True)
class SurrogateTest:
    """Trend statistics of the observed indicators and their surrogate p-values (module docstring, step 4).

    tau[indicator]: Kendall tau-b over the whole indicator series; p[kind][indicator]: p-value against
    the surrogate family `kind`; null[kind][indicator]: the surrogate taus [n_surrogates].
    """

    tau: dict[str, float]
    p: dict[str, dict[str, float]]
    null: dict[str, dict[str, np.ndarray]]


def trend_statistics(indicators: Mapping[str, np.ndarray]) -> dict[str, float]:
    """Kendall tau-b of every indicator series against time over all its valid values (offline statistic)."""
    return {k: float(trend_tau(np.asarray(v, dtype=np.float64)[None])[0]) for k, v in indicators.items()}


def surrogate_test(residual: np.ndarray, *, config: EarlyWarningConfig, times: np.ndarray | None = None,
                   rng: np.random.Generator | None = None, batch: int = 32) -> SurrogateTest:
    """Significance of the indicator trends of a residual series against stationary surrogates."""
    r = np.asarray(residual, dtype=np.float64)
    if r.ndim != 1 or r.shape[0] < config.window + 2:
        raise ValueError("the surrogate test needs a 1-D residual series longer than the window plus two")
    gen = rng if rng is not None else np.random.default_rng(config.seed)
    observed = trend_statistics(rolling_indicators(r, config=config, times=times))
    makers = {"phase": phase_surrogates, "ar1": ar1_surrogates}
    p: dict[str, dict[str, float]] = {}
    null: dict[str, dict[str, np.ndarray]] = {}
    for kind in config.surrogate_kinds:
        # Indicators per chunk of surrogates (bounded memory), one trend computation per indicator.
        series: dict[str, list[np.ndarray]] = {k: [] for k in config.indicators}
        done = 0
        while done < config.surrogates:
            k = min(batch, config.surrogates - done)
            ind = rolling_indicators(makers[kind](r, k, gen), config=config, times=times)
            for name in config.indicators:
                series[name].append(ind[name])
            done += k
        null[kind] = {name: trend_tau(np.concatenate(v, axis=0)) for name, v in series.items()}
        p[kind] = {name: _p_value(observed[name], null[kind][name], DIRECTIONS[name]) for name in config.indicators}
    return SurrogateTest(tau=observed, p=p, null=null)


def composite_level(indicators: Mapping[str, np.ndarray], *, names: Sequence[str], location: Mapping[str, float],
                    scale: Mapping[str, float]) -> np.ndarray:
    """Level score mean_i d_i z_i (two-sided: |z_i|), NaN where an indicator is undefined."""
    zs = []
    for name in names:
        z = (np.asarray(indicators[name], dtype=np.float64) - location[name]) / scale[name]
        d = DIRECTIONS[name]
        zs.append(np.abs(z) if d == 0 else d * z)
    return np.mean(np.stack(zs, axis=0), axis=0)


def composite_trend(taus: Mapping[str, np.ndarray], *, names: Sequence[str]) -> np.ndarray:
    """Trend score mean_i d_i tau_i (two-sided: |tau_i|), NaN where a trend is undefined."""
    ts = []
    for name in names:
        t = np.asarray(taus[name], dtype=np.float64)
        d = DIRECTIONS[name]
        ts.append(np.abs(t) if d == 0 else d * t)
    return np.mean(np.stack(ts, axis=0), axis=0)


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Split-conformal threshold s_(k), k = ceil((n + 1)(1 - alpha)) (+inf if k > n), alarm when score > it."""
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    s = np.sort(np.asarray(scores, dtype=np.float64)[np.isfinite(scores)])
    k = math.ceil((s.shape[0] + 1) * (1.0 - alpha))
    return float(s[k - 1]) if 1 <= k <= s.shape[0] else math.inf


def _settings(config: EarlyWarningConfig) -> dict[str, object]:
    # The settings an alarm calibration depends on (a calibration is refused under other settings).
    return {"window": config.window, "detrend": config.detrend, "bandwidth": config.bandwidth,
            "kernel_truncate": config.kernel_truncate, "spectral_low_max": config.spectral_low_max,
            "spectral_high_min": config.spectral_high_min, "dfa_scales": list(config.dfa_scales),
            "trend_window": config.trend_window, "composite_indicators": list(config.composite_indicators),
            "alarm_score": config.alarm_score}


@dataclass(frozen=True)
class AlarmCalibration:
    """Baselines and alarm threshold of one monitored series (module docstring, step 5).

    score: "level" or "trend"; location / scale: per composite indicator (level baselines); threshold:
    alarm when the score exceeds it (+inf: too little benign data for alpha, never fires); alpha: target
    false-alarm rate per trigger of this series; n_fit / n_threshold: benign samples used for the
    baselines / the threshold; settings: the early-warning settings it is valid for.
    """

    series: str
    score: str
    indicators: tuple[str, ...]
    location: dict[str, float]
    scale: dict[str, float]
    threshold: float
    alpha: float
    n_fit: int
    n_threshold: int
    settings: dict[str, object] = field(default_factory=dict)

    def check(self, config: EarlyWarningConfig) -> None:
        """Raise if the calibration was made under other early-warning settings."""
        if self.settings != _settings(config):
            raise ValueError(f"the alarm calibration of {self.series!r} was made with other early-warning settings")

    def to_dict(self) -> dict[str, object]:
        d = asdict(self)
        d["indicators"] = list(self.indicators)
        d["threshold"] = None if math.isinf(self.threshold) else self.threshold
        return d

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> AlarmCalibration:
        """Inverse of `to_dict` (JSON data; a null threshold is +inf)."""
        thr = d["threshold"]
        return AlarmCalibration(
            series=str(d["series"]), score=str(d["score"]), indicators=tuple(str(x) for x in d["indicators"]),
            location={str(k): float(v) for k, v in dict(d["location"]).items()},
            scale={str(k): float(v) for k, v in dict(d["scale"]).items()},
            threshold=math.inf if thr is None else float(thr), alpha=float(d["alpha"]), n_fit=int(d["n_fit"]),
            n_threshold=int(d["n_threshold"]), settings=dict(d.get("settings", {})))


def save_calibrations(path: str | Path, calibrations: Mapping[str, AlarmCalibration]) -> Path:
    """Write per-series calibrations as JSON (keys: series names)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({k: c.to_dict() for k, c in calibrations.items()}, indent=2, sort_keys=True), encoding="utf-8")
    return p


def load_calibrations(path: str | Path) -> dict[str, AlarmCalibration]:
    """Read calibrations written by `save_calibrations`."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {str(k): AlarmCalibration.from_dict(v) for k, v in data.items()}


@dataclass(frozen=True)
class EWSResult:
    """Batch early-warning analysis of one series (all arrays [n], float64; NaN where undefined).

    trend, residual; indicators[name]; taus[name] (trailing Kendall trend); level (NaN without a
    calibration), trend_score; alarm (bool, False where the score is undefined; all False without a
    calibration); calibrated: whether a calibration was applied.
    """

    times: np.ndarray
    values: np.ndarray
    trend: np.ndarray
    residual: np.ndarray
    indicators: dict[str, np.ndarray]
    taus: dict[str, np.ndarray]
    level: np.ndarray
    trend_score: np.ndarray
    alarm: np.ndarray
    calibrated: bool


def early_warning(values: np.ndarray, *, config: EarlyWarningConfig, times: np.ndarray | None = None,
                  calibration: AlarmCalibration | None = None) -> EWSResult:
    """Detrending, rolling indicators, trailing trends, composite scores and alarms of a series.

    Non-finite values are dropped (the series is the sequence of valid observations; `times` follow).
    """
    v = np.asarray(values, dtype=np.float64)
    t = np.arange(v.shape[0], dtype=np.float64) if times is None else np.asarray(times, dtype=np.float64)
    if t.shape != v.shape:
        raise ValueError("times must match values")
    keep = np.isfinite(v)
    v, t = v[keep], t[keep]
    trend, resid = detrend(v, method=config.detrend, bandwidth=config.bandwidth, truncate=config.kernel_truncate)
    ind = rolling_indicators(resid, config=config, times=t)
    taus = {k: trailing_tau(x, config.trend_window) for k, x in ind.items()}
    names = config.composite_indicators
    tscore = composite_trend(taus, names=names)
    if calibration is not None:
        calibration.check(config)
        level = composite_level(ind, names=names, location=calibration.location, scale=calibration.scale)
        score = level if calibration.score == "level" else tscore
        alarm = np.where(np.isfinite(score), score > calibration.threshold, False)
    else:
        level = np.full(v.shape[0], np.nan)
        alarm = np.zeros(v.shape[0], dtype=bool)
    return EWSResult(times=t, values=v, trend=trend, residual=resid, indicators=ind, taus=taus, level=level,
                     trend_score=tscore, alarm=alarm.astype(bool), calibrated=calibration is not None)


def calibrate_alarm(benign: Sequence[tuple[np.ndarray, np.ndarray | None]], *, series: str,
                    config: EarlyWarningConfig, alpha: float | None = None) -> AlarmCalibration:
    """Baselines and conformal threshold from benign (values, times) series (module docstring, step 5).

    With two or more series, the first ceil(fit_fraction * n) series fit the baselines and the others
    set the threshold; with one series its valid indicator points are split in time at fit_fraction.
    alpha defaults to `config.target_false_alarm_rate`. Online alarms need a causal detrending.
    """
    if config.detrend == "gaussian":
        raise ValueError("an online alarm needs a causal detrending (gaussian_causal or none), not two-sided")
    if not benign:
        raise ValueError("calibrate_alarm needs at least one benign series")
    a = config.target_false_alarm_rate if alpha is None else float(alpha)
    names = config.composite_indicators
    runs = [early_warning(v, config=config, times=t) for v, t in benign]
    if len(runs) >= 2:
        n_fit = max(1, min(len(runs) - 1, math.ceil(config.calibration_fit_fraction * len(runs))))
        fit_runs, thr_runs = runs[:n_fit], runs[n_fit:]
        fit = {k: np.concatenate([r.indicators[k] for r in fit_runs]) for k in names}
        thr_ind = {k: np.concatenate([r.indicators[k] for r in thr_runs]) for k in names}
        thr_tau = {k: np.concatenate([r.taus[k] for r in thr_runs]) for k in names}
    else:
        r0 = runs[0]
        valid = np.nonzero(np.all(np.stack([np.isfinite(r0.indicators[k]) for k in names]), axis=0))[0]
        if valid.shape[0] < 4:
            raise ValueError("too few complete indicator windows in the benign series to calibrate")
        cut = valid[int(math.floor(config.calibration_fit_fraction * valid.shape[0]))]
        fit = {k: r0.indicators[k][:cut] for k in names}
        thr_ind = {k: r0.indicators[k][cut:] for k in names}
        thr_tau = {k: r0.taus[k][cut:] for k in names}
    location: dict[str, float] = {}
    scale: dict[str, float] = {}
    for k in names:
        x = fit[k][np.isfinite(fit[k])]
        if x.shape[0] < 2:
            raise ValueError(f"indicator {k!r} has fewer than two benign values to fit its baseline")
        med = float(np.median(x))
        s = MAD_SCALE * float(np.median(np.abs(x - med)))
        if not s > 0:
            s = float(np.std(x, ddof=1))
        if not s > 0:
            raise ValueError(f"indicator {k!r} is constant on the benign data; it cannot be standardised")
        location[k], scale[k] = med, s
    if config.alarm_score == "level":
        scores = composite_level(thr_ind, names=names, location=location, scale=scale)
    else:
        scores = composite_trend(thr_tau, names=names)
    scores = scores[np.isfinite(scores)]
    n_fit_samples = int(sum(np.isfinite(fit[k]).sum() for k in names[:1]))
    return AlarmCalibration(series=series, score=config.alarm_score, indicators=tuple(names), location=location,
                            scale=scale, threshold=conformal_quantile(scores, a), alpha=a, n_fit=n_fit_samples,
                            n_threshold=int(scores.shape[0]), settings=_settings(config))


class SlidingKendall:
    """Kendall tau-b of the last `horizon` values against their arrival order, updated per value."""

    def __init__(self, horizon: int) -> None:
        if horizon < 2:
            raise ValueError("horizon must be >= 2")
        self.horizon = horizon
        self._order: deque[float] = deque()
        self._sorted: list[float] = []
        self._ties: dict[float, int] = {}
        self._s = 0                                                            # sum_{i<j} sign(y_j - y_i)

    def update(self, y: float) -> float:
        """Add y (as the newest value) and return tau_b once `horizon` values are held (else NaN)."""
        if len(self._order) == self.horizon:
            old = self._order.popleft()
            self._sorted.pop(bisect.bisect_left(self._sorted, old))
            greater = len(self._sorted) - bisect.bisect_right(self._sorted, old)
            less = bisect.bisect_left(self._sorted, old)
            self._s -= greater - less                                          # pairs (old, later) removed
            c = self._ties[old] - 1
            if c:
                self._ties[old] = c
            else:
                del self._ties[old]
        less = bisect.bisect_left(self._sorted, y)
        greater = len(self._sorted) - bisect.bisect_right(self._sorted, y)
        self._s += less - greater                                              # pairs (earlier, new) added
        bisect.insort(self._sorted, y)
        self._order.append(y)
        self._ties[y] = self._ties.get(y, 0) + 1
        if len(self._order) < self.horizon:
            return math.nan
        n0 = self.horizon * (self.horizon - 1) / 2.0
        n2 = sum(c * (c - 1) / 2.0 for c in self._ties.values() if c > 1)
        den = math.sqrt(n0 * (n0 - n2))
        return self._s / den if den > 0 else math.nan


@dataclass(frozen=True)
class EWSStep:
    """One streaming early-warning update (NaN where a quantity is not yet defined).

    index: count of valid samples so far minus one; value, trend, residual of this sample;
    indicators[name], taus[name]; level (NaN without a calibration), trend_score; alarm: True / False,
    or None when the alarm score is undefined or no calibration is loaded.
    """

    index: int
    time: float
    value: float
    trend: float
    residual: float
    indicators: dict[str, float]
    taus: dict[str, float]
    level: float
    trend_score: float
    alarm: bool | None


class StreamingEWS:
    """Online early-warning indicators of one series, equal to `early_warning` (module docstring)."""

    def __init__(self, config: EarlyWarningConfig, calibration: AlarmCalibration | None = None) -> None:
        if config.detrend == "gaussian":
            raise ValueError("two-sided detrending is offline only; streaming needs gaussian_causal or none")
        if calibration is not None:
            calibration.check(config)
        self.config = config
        self.calibration = calibration
        self.window = config.window
        self._kernel = kernel_weights(config.bandwidth, config.kernel_truncate) if config.detrend != "none" else None
        self._raw: deque[float] = deque(maxlen=0 if self._kernel is None else self._kernel.shape[0])
        self._res: deque[float] = deque()                                     # residuals of the window
        self._time: deque[float] = deque()
        self._count = 0                                                        # valid samples seen
        self._since_sync = 0
        self._shift = 0.0
        self._sums = np.zeros(5)                                               # S1 ... S4 of r - shift, lag-1 product
        half = self.window // 2
        self._kidx = np.arange(1, half + 1)
        self._twiddle = np.exp(2j * np.pi * self._kidx / self.window)          # e^{2 pi i k / W}
        self._dft = np.zeros(half, dtype=np.complex128)
        self._scales = list(config.dfa_scales)
        self._boxes: dict[int, deque[tuple[int, float]]] = {s: deque() for s in self._scales}
        self._kendall = {name: SlidingKendall(config.trend_window) for name in config.indicators}

    def _resync(self) -> None:
        # Exact recomputation of the running sums and of the DFT from the window buffer.
        r = np.array(self._res, dtype=np.float64)
        self._shift = float(r.mean()) if r.size else 0.0
        y = r - self._shift
        self._sums = np.array([y.sum(), (y**2).sum(), (y**3).sum(), (y**4).sum(),
                               (y[:-1] * y[1:]).sum() if y.size > 1 else 0.0])
        j = np.arange(r.shape[0])
        self._dft = (r[None, :] * np.exp(-2j * np.pi * self._kidx[:, None] * j[None, :] / self.window)).sum(1)
        self._since_sync = 0

    def _detrend(self, x: float) -> float:
        if self._kernel is None:
            return 0.0
        self._raw.appendleft(x)                                                # lag 0 first
        taps = self._kernel[: len(self._raw)]
        return float(np.dot(taps, np.fromiter(self._raw, dtype=np.float64, count=len(self._raw))) / taps.sum())

    def _push(self, r: float, t: float) -> None:
        # Add the residual to the window (evicting the oldest when full) and update sums and DFT.
        y = r - self._shift
        if len(self._res) == self.window:
            old = self._res.popleft()
            self._time.popleft()
            yo = old - self._shift
            second = self._res[0] - self._shift
            self._sums -= np.array([yo, yo**2, yo**3, yo**4, yo * second])
            last = self._res[-1] - self._shift
            self._res.append(r)
            self._time.append(t)
            self._sums += np.array([y, y**2, y**3, y**4, last * y])
            self._dft = self._twiddle * (self._dft - old + r)
        else:
            pos = len(self._res)
            if pos:
                self._sums[4] += (self._res[-1] - self._shift) * y
            self._res.append(r)
            self._time.append(t)
            self._sums[:4] += np.array([y, y**2, y**3, y**4])
            self._dft = self._dft + r * np.exp(-2j * np.pi * self._kidx * pos / self.window)
        self._since_sync += 1
        if self._since_sync >= self.config.resync_every:
            self._resync()

    def _moments(self) -> dict[str, float]:
        w = self.window
        s1, s2, s3, s4, p = self._sums.tolist()
        yb = s1 / w
        c2 = s2 - s1 * yb                                                      # sum (r - mu)^2
        c3 = s3 - 3.0 * yb * s2 + 3.0 * yb * yb * s1 - w * yb**3
        c4 = s4 - 4.0 * yb * s3 + 6.0 * yb * yb * s2 - 4.0 * yb**3 * s1 + w * yb**4
        first, last = self._res[0] - self._shift, self._res[-1] - self._shift
        lag = p - yb * ((s1 - last) + (s1 - first)) + (w - 1) * yb * yb
        out: dict[str, float] = {"variance": c2 / (w - 1)}
        if c2 > 0:
            rho = lag / c2
            m2 = c2 / w
            dt = (self._time[-1] - self._time[0]) / (w - 1)
            out.update(ar1=rho, skewness=(c3 / w) / m2**1.5, kurtosis=(c4 / w) / m2**2,
                       return_rate=(-math.log(rho) / dt) if (rho > 0 and dt > 0) else math.nan)
        else:
            out.update(ar1=math.nan, skewness=math.nan, kurtosis=math.nan, return_rate=math.nan)
        return out

    def _spectral(self) -> tuple[float, float]:
        # Hann taper in frequency on the demeaned window: X_0 = 0, X_{floor(W/2)+1} = conj(X_{W-floor(W/2)-1}).
        half = self.window // 2
        x = np.concatenate([[0.0 + 0.0j], self._dft, [np.conj(self._dft[self.window - half - 2])]])  # k = 0 ... half+1
        xw = 0.5 * x[1: half + 1] - 0.25 * (x[: half] + x[2: half + 2])
        power = (xw.real**2 + xw.imag**2) / float((_hann(self.window) ** 2).sum())
        ratio, expo = _spectral(power[None], self.config)
        return float(ratio[0]), float(expo[0])

    def _dfa(self, idx: int) -> float:
        # Complete the box ending at idx for every scale, drop boxes leaving the window, return the exponent.
        start = idx - self.window + 1
        f2 = np.empty(len(self._scales), dtype=np.float64)
        res = np.array(self._res, dtype=np.float64)
        for si, s in enumerate(self._scales):
            q = self._boxes[s]
            if (idx + 1) % s == 0 and len(self._res) >= s:
                q.append((idx + 1 - s, float(_box_rss(res[-s:]) / s)))
            while q and q[0][0] < start:
                q.popleft()
            f2[si] = math.fsum(v for _, v in q) / len(q) if q else math.nan
        if idx < self.window - 1:
            return math.nan
        return float(_dfa_exponent(f2[None], np.asarray(self._scales))[0])

    def update(self, value: float, time: float | None = None) -> EWSStep:
        """Add one sample (time defaults to the sample index); non-finite values are skipped."""
        names = self.config.indicators
        nan_ind = {k: math.nan for k in names}
        t = float(self._count if time is None else time)
        if not math.isfinite(value):
            return EWSStep(self._count - 1, t, float(value), math.nan, math.nan, dict(nan_ind), dict(nan_ind),
                           math.nan, math.nan, None)
        idx = self._count
        self._count += 1
        trend = self._detrend(float(value))
        r = float(value) - trend
        self._push(r, t)
        dfa = self._dfa(idx) if "dfa" in names else math.nan
        ind = dict(nan_ind)
        if len(self._res) == self.window:
            mom = self._moments()
            for k in ("variance", "ar1", "skewness", "kurtosis", "return_rate"):
                if k in names:
                    ind[k] = mom[k]
            if {"spectral_ratio", "spectral_exponent"} & set(names):
                ratio, expo = self._spectral()
                if "spectral_ratio" in names:
                    ind["spectral_ratio"] = ratio
                if "spectral_exponent" in names:
                    ind["spectral_exponent"] = expo
            if "dfa" in names:
                ind["dfa"] = dfa
        taus = {k: (self._kendall[k].update(ind[k]) if math.isfinite(ind[k]) else math.nan) for k in names}
        comp = self.config.composite_indicators
        tvals = [abs(taus[k]) if DIRECTIONS[k] == 0 else DIRECTIONS[k] * taus[k] for k in comp]
        tscore = float(np.mean(tvals))
        level = math.nan
        alarm: bool | None = None
        if self.calibration is not None:
            cal = self.calibration
            zs = [(ind[k] - cal.location[k]) / cal.scale[k] for k in comp]
            zs = [abs(z) if DIRECTIONS[k] == 0 else DIRECTIONS[k] * z for k, z in zip(comp, zs, strict=True)]
            level = float(np.mean(zs))
            score = level if cal.score == "level" else tscore
            alarm = bool(score > cal.threshold) if math.isfinite(score) else None
        return EWSStep(idx, t, float(value), trend, r, ind, taus, level, tscore, alarm)


__all__ = [
    "DIRECTIONS", "MAD_SCALE", "AlarmCalibration", "EWSResult", "EWSStep", "SlidingKendall", "StreamingEWS",
    "SurrogateTest", "ar1_surrogates", "calibrate_alarm", "composite_level", "composite_trend", "conformal_quantile",
    "detrend", "early_warning", "kendall_tau_b", "kernel_weights", "load_calibrations", "phase_surrogates",
    "rolling_indicators", "save_calibrations", "surrogate_test", "trailing_tau", "trend_statistics", "trend_tau",
]
