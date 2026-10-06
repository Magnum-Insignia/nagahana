"""Stationarity tests implemented in-house: augmented Dickey-Fuller and KPSS.

Augmented Dickey-Fuller (Dickey and Fuller, JASA 74:427-431, 1979; Said and Dickey, Biometrika
71:599-607, 1984)
---------------------------------------------------------------------------------------------
    Delta y_t = d_t' b + gamma y_{t-1} + sum_{i=1..p} delta_i Delta y_{t-i} + e_t,
with deterministic terms d_t: none ("n"), constant ("c"), constant and trend ("ct"), constant, trend
and squared trend ("ctt"). H0: gamma = 0 (unit root); the statistic is tau = gamma_hat / se(gamma_hat)
from OLS. Lag order p: every p in 0 ... p_max is fitted on the common sample (the last T - 1 - p_max
differences) and the one minimising AIC = -2 llf + 2k or BIC = -2 llf + k log n is chosen (Gaussian
llf = -n/2 (log(2 pi) + log(SSR / n) + 1)); "t-stat" starts at p_max and drops the last lag while
its |t| < 1.959963984540054 (two-sided 5 %; general to specific, Ng and Perron, JASA 90:268-281,
1995). The chosen model is re-estimated on the largest sample its lags allow. Default
p_max = floor(12 (T / 100)^(1/4)) (Schwert, Journal of Business and Economic Statistics 7(2):
147-159, 1989), reduced when the sample is too short.

Distribution of tau under H0. For "n", "c" and "ct" (one series, N = 1):
    p-value   Phi(sum_k b_k tau^k) with the response-surface coefficients of MacKinnon (Journal of
              Business and Economic Statistics 12(2):167-176, 1994), the small-p set for
              tau <= tau_star and the large-p set above; 0 below tau_min and 1 above tau_max.
    critical  c(T) = b0 + b1 / T + b2 / T^2 + b3 / T^3 at 1 %, 5 % and 10 % with T the regression
              sample size (MacKinnon, "Critical values for cointegration tests", Queen's Economics
              Department Working Paper 1227, 2010).
The b0 terms are the asymptotic critical values (-3.43, -2.86, -2.57 with a constant; -3.96, -3.41,
-3.13 with a trend; -2.57, -1.94, -1.62 with neither, as in Hamilton, "Time Series Analysis", Princeton
1994, Table B.6). The tests simulate the null distribution in-house (`simulate_df`) and check the
coefficients against it. "ctt", and any regression with critical="simulate", uses that simulation:
R random walks of length T, tau of each by the Frisch-Waugh-Lovell projection of the deterministic
terms (vectorised over replicates), p = (1 + #{tau_sim <= tau}) / (R + 1), critical values = empirical
quantiles. The lag-augmented statistic has the same limit distribution as the Dickey-Fuller one (Said
and Dickey 1984), so the simulation without lags is the right reference.

KPSS (Kwiatkowski, Phillips, Schmidt and Shin, Journal of Econometrics 54:159-178, 1992)
------------------------------------------------------------------------------------
H0: (trend-)stationarity. With e_t the OLS residuals of y on a constant ("c") or constant and trend
("ct"), S_t = sum_{i<=t} e_i and the Newey-West long-run variance with Bartlett weights
    s^2(l) = g_0 + 2 sum_{j=1..l} (1 - j / (l + 1)) g_j,   g_j = (1/T) sum_{t>j} e_t e_{t-j},
the statistic is eta = sum_t S_t^2 / (T^2 s^2(l)). Bandwidth l:
    nw1994   automatic (Newey and West, Review of Economic Studies 61(4):631-653, 1994):
             n = floor(4 (T/100)^(2/9)), s0 = g_0 + 2 sum_{j<=n} g_j, s1 = 2 sum_{j<=n} j g_j,
             l = min(T - 1, floor(1.1447 ((s1 / s0)^2 T)^(1/3)))  (Bartlett constants; citation to verify
             for the pre-whitening rule of n)
    schwert  l = floor(12 (T/100)^(1/4))
    integer  a fixed l.
Critical values from KPSS (1992) Table 1: level 0.347, 0.463, 0.574, 0.739 and trend 0.119, 0.146,
0.176, 0.216 at 10, 5, 2.5 and 1 %; the p-value is interpolated linearly in that table and reported
as a bound outside [0.01, 0.10].
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import special

#: MacKinnon (1994) response surfaces, N = 1: tau_star, tau_min, tau_max and the small-p / large-p coefficients.
_TAU_STAR = {"n": -1.04, "c": -1.61, "ct": -2.89}
_TAU_MIN = {"n": -19.04, "c": -18.83, "ct": -16.18}
_TAU_MAX = {"n": np.inf, "c": 2.74, "ct": 0.7}
_SMALLP = {"n": (0.6344, 1.2378, 3.2496e-2), "c": (2.1659, 1.4412, 3.8269e-2), "ct": (3.2512, 1.6047, 4.9588e-2)}
_LARGEP = {"n": (0.4797, 9.3557e-1, -0.6999e-1, 3.3066e-2), "c": (1.7339, 9.3202e-1, -1.2745e-1, -1.0368e-2),
           "ct": (2.5261, 6.1654e-1, -3.7956e-1, -6.0285e-2)}
#: MacKinnon (2010) critical-value surfaces, N = 1: level -> (b0, b1, b2, b3).
_CRIT = {
    "n": {0.01: (-2.56574, -2.2358, -3.627, 0.0), 0.05: (-1.94100, -0.2686, -3.365, 31.223),
          0.10: (-1.61682, 0.2656, -2.714, 25.364)},
    "c": {0.01: (-3.43035, -6.5393, -16.786, -79.433), 0.05: (-2.86154, -2.8903, -4.234, -40.040),
          0.10: (-2.56677, -1.5384, -2.809, 0.0)},
    "ct": {0.01: (-3.95877, -9.0531, -28.428, -134.155), 0.05: (-3.41049, -4.3904, -9.036, -45.374),
           0.10: (-3.12705, -2.5856, -3.925, -22.380)},
}
#: KPSS (1992) Table 1: regression -> ((significance, critical value), ...), increasing critical values.
_KPSS = {"c": ((0.10, 0.347), (0.05, 0.463), (0.025, 0.574), (0.01, 0.739)),
         "ct": ((0.10, 0.119), (0.05, 0.146), (0.025, 0.176), (0.01, 0.216))}
_Z975 = 1.959963984540054


def mackinnon_p(tau: float, regression: str) -> float:
    """MacKinnon (1994) p-value of a Dickey-Fuller tau (N = 1; regression n, c or ct)."""
    if regression not in _TAU_STAR:
        raise ValueError("response surfaces exist for regression n, c and ct; use critical='simulate' otherwise")
    if tau > _TAU_MAX[regression]:
        return 1.0
    if tau < _TAU_MIN[regression]:
        return 0.0
    coef = _SMALLP[regression] if tau <= _TAU_STAR[regression] else _LARGEP[regression]
    return float(special.ndtr(sum(c * tau ** k for k, c in enumerate(coef))))


def mackinnon_critical(regression: str, nobs: int) -> dict[float, float]:
    """MacKinnon (2010) critical values {0.01, 0.05, 0.10} at sample size `nobs` (N = 1)."""
    if regression not in _CRIT:
        raise ValueError("critical-value surfaces exist for regression n, c and ct")
    return {lvl: b[0] + b[1] / nobs + b[2] / nobs**2 + b[3] / nobs**3 for lvl, b in _CRIT[regression].items()}


def _deterministic(n: int, regression: str, start: int = 1) -> np.ndarray:
    """Deterministic regressors [n, k] (constant, t, t^2 as requested); t counts from `start`."""
    t = np.arange(start, start + n, dtype=np.float64)
    cols = {"n": [], "c": [np.ones(n)], "ct": [np.ones(n), t], "ctt": [np.ones(n), t, t * t]}
    if regression not in cols:
        raise ValueError("regression must be one of n, c, ct, ctt")
    return np.stack(cols[regression], axis=1) if cols[regression] else np.zeros((n, 0))


def simulate_df(nobs: int, regression: str, *, replicates: int = 20_000, seed: int = 0, chunk: int = 2_000) -> np.ndarray:
    """Null distribution of the Dickey-Fuller tau for `nobs` regression observations (module docstring)."""
    rng = np.random.default_rng(seed)
    d = _deterministic(nobs, regression)
    k = d.shape[1] + 1
    if d.shape[1]:
        q, _ = np.linalg.qr(d)                                           # orthonormal basis of the deterministic terms
    out = np.empty(replicates)
    for r0 in range(0, replicates, chunk):
        r = min(chunk, replicates - r0)
        e = rng.standard_normal((r, nobs + 1))
        y = np.cumsum(e, axis=1)                                         # random walks y_0 ... y_nobs
        dy, ylag = np.diff(y, axis=1), y[:, :-1]                         # [r, nobs]
        if d.shape[1]:
            dy = dy - (dy @ q) @ q.T                                     # Frisch-Waugh-Lovell projection
            ylag = ylag - (ylag @ q) @ q.T
        sxx = (ylag * ylag).sum(axis=1)
        gamma = (ylag * dy).sum(axis=1) / sxx
        resid = dy - gamma[:, None] * ylag
        s2 = (resid * resid).sum(axis=1) / (nobs - k)
        out[r0: r0 + r] = gamma / np.sqrt(s2 / sxx)
    return out


@dataclass(frozen=True)
class ADFResult:
    """Augmented Dickey-Fuller result (module docstring)."""

    statistic: float
    p_value: float
    lags: int
    nobs: int
    critical: dict[float, float]
    regression: str
    autolag: str
    ic: dict[int, float] = field(default_factory=dict)
    method: str = "mackinnon"

    @property
    def rejects_unit_root_5pct(self) -> bool:
        return bool(self.statistic < self.critical[0.05])


def _ols(y: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """(coefficients, standard errors, SSR) of OLS of y on x."""
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    resid = y - x @ beta
    ssr = float(resid @ resid)
    dof = x.shape[0] - x.shape[1]
    xtx_inv = np.linalg.pinv(x.T @ x)
    se = np.sqrt(np.maximum(np.diag(xtx_inv) * ssr / max(dof, 1), 0.0))
    return beta, se, ssr


def _adf_design(y: np.ndarray, p: int, regression: str, start: int) -> tuple[np.ndarray, np.ndarray]:
    """(response Delta y_t, regressors [y_{t-1}, d_t, Delta y_{t-1..t-p}]) for t from `start` (in differences)."""
    dy = np.diff(y)
    n = dy.size - start
    resp = dy[start:]
    cols = [y[start: start + n]]                                          # y_{t-1} aligned with dy[t]
    det = _deterministic(n, regression, start=start + 1)
    lag_cols = [dy[start - i: start - i + n] for i in range(1, p + 1)]
    x = np.column_stack(cols + [det[:, j] for j in range(det.shape[1])] + lag_cols)
    return resp, x


def adf(
    y: np.ndarray,
    *,
    regression: str = "c",
    autolag: str = "aic",
    max_lag: int | None = None,
    critical: str = "mackinnon",
    replicates: int = 20_000,
    seed: int = 0,
) -> ADFResult:
    """Augmented Dickey-Fuller test (module docstring)."""
    a = np.asarray(y, dtype=np.float64).reshape(-1)
    a = a[np.isfinite(a)]
    t_len = a.size
    if regression not in ("n", "c", "ct", "ctt"):
        raise ValueError("regression must be one of n, c, ct, ctt")
    if autolag not in ("aic", "bic", "t-stat", "none"):
        raise ValueError("autolag must be aic, bic, t-stat or none")
    n_det = {"n": 0, "c": 1, "ct": 2, "ctt": 3}[regression]
    p_max = int(np.floor(12.0 * (t_len / 100.0) ** 0.25)) if max_lag is None else int(max_lag)
    p_max = max(0, min(p_max, (t_len - 1) // 2 - n_det - 2))
    if t_len - 1 - p_max < n_det + p_max + 3:
        raise ValueError(f"series of length {t_len} is too short for the ADF regression")
    ic: dict[int, float] = {}
    if autolag in ("aic", "bic"):
        for p in range(p_max + 1):
            resp, x = _adf_design(a, p, regression, start=p_max)
            _, _, ssr = _ols(resp, x)
            n = resp.size
            llf = -0.5 * n * (np.log(2.0 * np.pi) + np.log(ssr / n) + 1.0)
            k = x.shape[1]
            ic[p] = float(-2.0 * llf + (2.0 * k if autolag == "aic" else k * np.log(n)))
        lags = min(ic, key=lambda q: (ic[q], q))
    elif autolag == "t-stat":
        lags = p_max
        while lags > 0:
            resp, x = _adf_design(a, lags, regression, start=p_max)
            beta, se, _ = _ols(resp, x)
            if abs(beta[-1] / se[-1]) >= _Z975:
                break
            lags -= 1
    else:
        lags = p_max
    resp, x = _adf_design(a, lags, regression, start=lags)
    beta, se, _ = _ols(resp, x)
    tau = float(beta[0] / se[0])
    nobs = int(resp.size)
    if critical == "simulate" or regression == "ctt":
        sims = np.sort(simulate_df(nobs, regression, replicates=replicates, seed=seed))
        p_value = float((1 + np.searchsorted(sims, tau, side="right")) / (sims.size + 1))
        crit = {lvl: float(np.quantile(sims, lvl)) for lvl in (0.01, 0.05, 0.10)}
        method = "simulated"
    elif critical == "mackinnon":
        p_value = mackinnon_p(tau, regression)
        crit = mackinnon_critical(regression, nobs)
        method = "mackinnon"
    else:
        raise ValueError("critical must be 'mackinnon' or 'simulate'")
    return ADFResult(statistic=tau, p_value=p_value, lags=int(lags), nobs=nobs, critical=crit, regression=regression,
                     autolag=autolag, ic=ic, method=method)


@dataclass(frozen=True)
class KPSSResult:
    """KPSS result (module docstring). p_value is clipped to [0.01, 0.10]; `p_bound` says which side, if any."""

    statistic: float
    p_value: float
    p_bound: str
    lags: int
    critical: dict[float, float]
    regression: str

    @property
    def rejects_stationarity_5pct(self) -> bool:
        return bool(self.statistic > self.critical[0.05])


def _autocov(e: np.ndarray, j: int) -> float:
    return float(e[j:] @ e[: e.size - j]) / e.size


def kpss(y: np.ndarray, *, regression: str = "c", lags: str | int = "nw1994") -> KPSSResult:
    """KPSS stationarity test (module docstring)."""
    a = np.asarray(y, dtype=np.float64).reshape(-1)
    a = a[np.isfinite(a)]
    t_len = a.size
    if regression not in _KPSS:
        raise ValueError("regression must be 'c' or 'ct'")
    if t_len < 10:
        raise ValueError("KPSS needs at least 10 observations")
    x = _deterministic(t_len, regression)
    beta, *_ = np.linalg.lstsq(x, a, rcond=None)
    e = a - x @ beta
    if isinstance(lags, int) or (isinstance(lags, str) and lags.isdigit()):
        lag = int(lags)
    elif lags == "schwert":
        lag = int(np.floor(12.0 * (t_len / 100.0) ** 0.25))
    elif lags == "nw1994":
        n_pre = int(np.floor(4.0 * (t_len / 100.0) ** (2.0 / 9.0)))
        g = [_autocov(e, j) for j in range(n_pre + 1)]
        s0 = g[0] + 2.0 * sum(g[1:])
        s1 = 2.0 * sum(j * g[j] for j in range(1, n_pre + 1))
        lag = int(np.floor(1.1447 * ((s1 / s0) ** 2 * t_len) ** (1.0 / 3.0))) if s0 > 0 else 0
    else:
        raise ValueError("lags must be 'nw1994', 'schwert' or an integer")
    lag = int(min(max(lag, 0), t_len - 1))
    lrv = _autocov(e, 0) + 2.0 * sum((1.0 - j / (lag + 1.0)) * _autocov(e, j) for j in range(1, lag + 1))
    s = np.cumsum(e)
    eta = float((s @ s) / (t_len**2 * lrv)) if lrv > 0 else float("inf")
    table = _KPSS[regression]
    crit = {lvl: cv for lvl, cv in table}
    cvs = np.array([cv for _, cv in table])
    ps = np.array([lvl for lvl, _ in table])
    if eta < cvs[0]:
        p, bound = 0.10, "greater"
    elif eta > cvs[-1]:
        p, bound = 0.01, "smaller"
    else:
        p, bound = float(np.interp(eta, cvs, ps)), ""
    return KPSSResult(statistic=eta, p_value=p, p_bound=bound, lags=lag, critical=crit, regression=regression)


__all__ = ["ADFResult", "KPSSResult", "adf", "kpss", "mackinnon_critical", "mackinnon_p", "simulate_df"]
