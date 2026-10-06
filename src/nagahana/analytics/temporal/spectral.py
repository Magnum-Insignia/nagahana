"""Spectral density and periodicity: Welch's estimate, the periodogram, Fisher's g test, Lomb-Scargle.

Beaconing (command-and-control check-ins at a fixed interval, under jitter) shows as a peak in the
spectrum of the event counts; benign aggregate traffic is broadband and self-similar (AS-36; Leland,
Taqqu, Willinger and Wilson, IEEE/ACM ToN 2(1):1-15, 1994; Hu et al., BAYWATCH, DSN 2016).

Welch (Welch, IEEE Trans. Audio and Electroacoustics 15(2):70-73, 1967)
------------------------------------------------------------------------
Averaged periodograms of Hann-windowed segments with 50 % overlap and constant detrending
(scipy.signal.welch), one-sided density in units^2 / Hz.

Fisher's g test (Fisher, Proc. R. Soc. London A 125:54-59, 1929)
----------------------------------------------------------------
With periodogram ordinates I_1 ... I_q at the Fourier frequencies j / n, j = 1 ... q = floor((n - 1) / 2)
(zero and Nyquist excluded), g = max_j I_j / sum_j I_j. Under Gaussian white noise
    P(G > g) = sum_{j=1..floor(1/g)} (-1)^(j-1) C(q, j) (1 - j g)^(q-1).
The first term q (1 - g)^(q-1) is a Bonferroni bound, exact as P -> 0. The alternating sum cancels
catastrophically in float64 when its terms are large, so it is evaluated in decimal arithmetic whose
precision is set from the magnitude of the largest term (40 significant digits beyond it), with
binomials as exact integers; with more than `max_terms` terms (g < 1 / max_terms) the bound is
reported instead and flagged as such. The peak ratio max_j I_j / mean_j I_j = q g is the periodicity
feature of AS-36.

Lomb-Scargle (Lomb, Astrophysics and Space Science 39:447-462, 1976; Scargle, Astrophysical Journal
263:835-853, 1982)
-------------------------------------------------------------------------------------------------
For irregular times t_j, with tau(omega) from tan(2 omega tau) = sum sin 2 omega t_j / sum cos 2 omega t_j,
    P(omega) = (1 / (2 s^2)) ([sum y'_j cos omega(t_j - tau)]^2 / sum cos^2 omega(t_j - tau)
                             + [sum y'_j sin omega(t_j - tau)]^2 / sum sin^2 omega(t_j - tau)),
y' = y - mean(y), s^2 the sample variance; under Gaussian white noise P(omega) is Exp(1) at each
frequency (Horne and Baliunas, Astrophysical Journal 302:757-763, 1986), so the false-alarm
probability of the maximum over M independent frequencies is 1 - (1 - exp(-P_max))^M.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, localcontext
from math import comb

import numpy as np
from scipy import signal, special


@dataclass(frozen=True)
class Periodogram:
    """Spectrum summary of one regular series (module docstring)."""

    freq: np.ndarray
    welch_freq: np.ndarray
    welch_density: np.ndarray
    ordinates: np.ndarray
    g: float
    p_value: float
    p_exact: bool
    peak_frequency: float
    peak_period: float
    peak_ratio: float


def fisher_g_pvalue(g: float, q: int, *, max_terms: int = 2_000) -> tuple[float, bool]:
    """(P(G > g), exact) for q ordinates (module docstring)."""
    if q < 2 or not np.isfinite(g):
        return float("nan"), False
    if g <= 0:
        return 1.0, True
    if g >= 1:
        return 0.0, True
    terms = min(int(np.floor(1.0 / g)), q)
    if terms > max_terms:
        return float(min(1.0, q * np.exp((q - 1) * np.log1p(-g)))), False
    js = np.arange(1, terms + 1, dtype=np.float64)
    base = 1.0 - js * g
    ok = base > 0
    log_terms = (special.gammaln(q + 1.0) - special.gammaln(js[ok] + 1.0) - special.gammaln(q - js[ok] + 1.0)
                 + (q - 1) * np.log(base[ok]))
    digits = int(40 + max(0.0, float(log_terms.max(initial=0.0)) / np.log(10.0)))
    with localcontext() as ctx:
        ctx.prec = digits
        gd = Decimal(g)                                                  # exact value of the float g
        total = Decimal(0)
        for j in range(1, terms + 1):
            b = 1 - j * gd
            if b <= 0:
                break
            total += (-1) ** (j - 1) * Decimal(comb(q, j)) * b ** (q - 1)
        p = float(min(max(total, Decimal(0)), Decimal(1)))
    return p, True


def periodogram(x: np.ndarray, *, fs: float = 1.0, nperseg: int = 256) -> Periodogram:
    """Welch density, raw periodogram, Fisher's g test and the peak ratio of a regular series."""
    a = np.asarray(x, dtype=np.float64).reshape(-1)
    if not np.isfinite(a).all():
        raise ValueError("the series must be finite")
    n = a.size
    if n < 8:
        raise ValueError("need at least 8 observations")
    wf, wd = signal.welch(a, fs=fs, nperseg=min(nperseg, n), detrend="constant", scaling="density")
    d = a - a.mean()
    q = (n - 1) // 2
    j = np.arange(1, q + 1)
    spec = np.abs(np.fft.rfft(d)[1: q + 1]) ** 2 / n                     # I_j at frequencies j / n
    total = spec.sum()
    g = float(spec.max() / total) if total > 0 else float("nan")
    p, exact = fisher_g_pvalue(g, q)
    k = int(np.argmax(spec)) if total > 0 else 0
    f_peak = float(j[k] / n * fs)
    return Periodogram(freq=j / n * fs, welch_freq=wf, welch_density=wd, ordinates=spec, g=g, p_value=p, p_exact=exact,
                       peak_frequency=f_peak, peak_period=1.0 / f_peak if f_peak > 0 else float("inf"),
                       peak_ratio=float(q * g) if np.isfinite(g) else float("nan"))


def lomb_scargle(t: np.ndarray, y: np.ndarray, freqs: np.ndarray, *, chunk: int = 256) -> np.ndarray:
    """Normalised Lomb-Scargle power at cyclic frequencies `freqs` (module docstring)."""
    tt = np.asarray(t, dtype=np.float64).reshape(-1)
    yy = np.asarray(y, dtype=np.float64).reshape(-1)
    if tt.shape != yy.shape or tt.size < 3:
        raise ValueError("t and y must have the same length (>= 3)")
    yc = yy - yy.mean()
    s2 = yc.var(ddof=1)
    tt = tt - tt.min()
    out = np.empty(len(freqs))
    w_all = 2.0 * np.pi * np.asarray(freqs, dtype=np.float64)
    for f0 in range(0, w_all.size, chunk):
        w = w_all[f0: f0 + chunk, None]                                  # [F, 1]
        tau = np.arctan2(np.sin(2 * w * tt).sum(axis=1), np.cos(2 * w * tt).sum(axis=1))[:, None] / (2 * w)
        arg = w * (tt[None, :] - tau)
        c, s = np.cos(arg), np.sin(arg)
        cc, ss = (c * c).sum(axis=1), (s * s).sum(axis=1)
        pc = (c @ yc) ** 2 / np.where(cc > 0, cc, np.inf)
        ps = (s @ yc) ** 2 / np.where(ss > 0, ss, np.inf)
        out[f0: f0 + chunk] = (pc + ps) / (2.0 * s2) if s2 > 0 else np.nan
    return out


def false_alarm(power_max: float, m: int) -> float:
    """1 - (1 - exp(-P_max))^M (module docstring)."""
    return float(-np.expm1(m * np.log1p(-np.exp(-power_max)))) if np.isfinite(power_max) else float("nan")


__all__ = ["Periodogram", "false_alarm", "fisher_g_pvalue", "lomb_scargle", "periodogram"]
