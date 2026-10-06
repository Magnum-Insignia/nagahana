"""Serial dependence of a regularly sampled series: ACF, PACF and the Ljung-Box test.

ACF
---
r_k = c_k / c_0 with c_k = (1/n) sum_{t=1..n-k} (x_t - xbar)(x_{t+k} - xbar), the biased estimator (it
keeps the sequence positive semi-definite). Computed by FFT with zero padding to at least 2n, so the
circular wrap never mixes the two ends of the series (Wiener-Khinchin). Bands: white noise +- z / sqrt(n);
Bartlett's formula for an MA(k - 1) null, se_k = sqrt((1 + 2 sum_{j<k} r_j^2) / n) (Box, Jenkins and
Reinsel, "Time Series Analysis", 4th ed., Wiley 2008, section 6.2.2).

PACF
----
phi_kk by the Durbin-Levinson recursion (Durbin, Revue de l'Institut International de Statistique 28:
233-244, 1960):
    phi_11 = r_1,
    phi_kk = (r_k - sum_{j<k} phi_{k-1,j} r_{k-j}) / (1 - sum_{j<k} phi_{k-1,j} r_j),
    phi_kj = phi_{k-1,j} - phi_kk phi_{k-1,k-j}.
Bands +- z / sqrt(n) (Quenouille's approximation under an AR(k - 1) null).

Ljung-Box (Ljung and Box, Biometrika 65(2):297-303, 1978)
-------------------------------------------------------
Q(h) = n (n + 2) sum_{k=1..h} r_k^2 / (n - k), compared with chi^2 on h - fitted degrees of freedom.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import special, stats


def acf(x: np.ndarray, nlags: int) -> np.ndarray:
    """r_0 ... r_nlags of a series (module docstring); NaN entries are not allowed."""
    a = np.asarray(x, dtype=np.float64).reshape(-1)
    if not np.isfinite(a).all():
        raise ValueError("the series must be finite (bin event counts first; absent values are not zero, D-41)")
    n = a.size
    if n < 2:
        raise ValueError("need at least two observations")
    nlags = int(min(nlags, n - 1))
    d = a - a.mean()
    size = 1 << int(np.ceil(np.log2(2 * n)))
    f = np.fft.rfft(d, n=size)
    c = np.fft.irfft(f * np.conj(f), n=size)[: nlags + 1] / n
    return c / c[0] if c[0] > 0 else np.full(nlags + 1, np.nan)


def pacf(r: np.ndarray) -> np.ndarray:
    """phi_00 = 1, phi_11 ... phi_KK from autocorrelations r_0 ... r_K (Durbin-Levinson)."""
    rr = np.asarray(r, dtype=np.float64)
    k_max = rr.size - 1
    out = np.ones(k_max + 1)
    if k_max == 0:
        return out
    phi = np.zeros(k_max + 1)
    phi[1] = rr[1]
    out[1] = rr[1]
    for k in range(2, k_max + 1):
        prev = phi[1:k].copy()
        num = rr[k] - float(prev @ rr[k - 1:0:-1])
        den = 1.0 - float(prev @ rr[1:k])
        pk = num / den if den != 0 else np.nan
        phi[1:k] = prev - pk * prev[::-1]
        phi[k] = pk
        out[k] = pk
    return out


@dataclass(frozen=True)
class SerialDependence:
    """ACF, PACF, bands and Ljung-Box statistics of one series (module docstring)."""

    lags: np.ndarray
    acf: np.ndarray
    pacf: np.ndarray
    white_band: float
    bartlett_band: np.ndarray
    ljung_box_q: np.ndarray
    ljung_box_p: np.ndarray
    n: int


def serial_dependence(x: np.ndarray, *, nlags: int, level: float = 0.95, fitted: int = 0) -> SerialDependence:
    """ACF, PACF, their bands and the Ljung-Box statistics for lags 1 ... nlags (module docstring)."""
    a = np.asarray(x, dtype=np.float64).reshape(-1)
    n = a.size
    r = acf(a, nlags)
    k = r.size - 1
    p = pacf(r)
    z = float(special.ndtri(0.5 + level / 2.0))
    lags = np.arange(k + 1)
    bart = z * np.sqrt((1.0 + 2.0 * np.concatenate([[0.0], np.cumsum(r[1:] ** 2)[:-1]])) / n) if k else np.zeros(1)
    bart = np.concatenate([[np.nan], bart])[: k + 1] if k else np.array([np.nan])
    q = n * (n + 2.0) * np.cumsum(r[1:] ** 2 / (n - np.arange(1, k + 1)))
    dof = np.arange(1, k + 1) - fitted
    pv = np.where(dof > 0, stats.chi2.sf(q, np.maximum(dof, 1)), np.nan)
    return SerialDependence(lags=lags, acf=r, pacf=p, white_band=z / np.sqrt(n), bartlett_band=bart,
                            ljung_box_q=np.concatenate([[np.nan], q]), ljung_box_p=np.concatenate([[np.nan], pv]), n=n)


__all__ = ["SerialDependence", "acf", "pacf", "serial_dependence"]
