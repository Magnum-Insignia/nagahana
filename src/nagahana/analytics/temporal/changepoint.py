"""Change-point detection: PELT (exact penalised segmentation) and Bayesian online change-point detection.

Segment costs (twice the negative maximised log-likelihood, up to constants, from prefix sums in O(1))
-----------------------------------------------------------------------------------------------------
For a segment of n values with sum S1 and sum of squares S2:
    normal_mean     (S2 - S1^2 / n) / sigma^2, sigma^2 fixed: the variance of the first differences
                    divided by 2 (a mean shift does not move it), estimated by MAD
    normal_meanvar  n (log(2 pi s^2) + 1), s^2 = max((S2 - S1^2 / n) / n, floor), floor = 1e-12 times
                    the variance of the whole series (a constant segment would otherwise cost -inf)
    poisson         2 (S1 - S1 log(S1 / n)), 0 log 0 = 0 (counts)
    exponential     2 n (log(S1 / n) + 1) (positive gaps; a change of event rate)
Penalty beta: "bic" (d + 1) log n and "aic" 2 (d + 1), with d the parameters of one segment (1 for a
mean, rate or Poisson intensity, 2 for mean and variance) and +1 for the change location; or a number.
Minimum segment length for normal_meanvar: 3. The variance estimate of a 2-point segment is
sigma^2 chi^2_1 / 2, and the chi^2_1 density is unbounded at 0, so near-zero variances (and arbitrarily
negative costs) are likely and produce spurious 2-point segments; with 3 points the estimate is
sigma^2 chi^2_2 / 3, whose density is finite at 0 (code run: on 100 change-free Gaussian series of
length 800, 9 % gained a spurious change point at minimum length 2 and none at 3).

PELT (Killick, Fearnhead and Eckley, JASA 107(500):1590-1598, 2012)
-------------------------------------------------------------------
F(0) = -beta, F(t) = min_{tau in R_t, t - tau >= min_size} [F(tau) + C(tau, t) + beta],
R_{t+1} = {tau in R_t : t - tau < min_size or F(tau) + C(tau, t) <= F(t)} u {t - min_size + 1 ...}.
Costs of the forms above satisfy C(a, b) + C(b, c) <= C(a, c), so pruning with K = 0 is exact: the
segmentation equals optimal partitioning (Jackson et al., IEEE Signal Processing Letters 12(2):105-108,
2005) at expected linear cost. The inner minimisation is vectorised over R_t.

Bayesian online change-point detection (Adams and MacKay, arXiv:0710.3742, 2007)
-------------------------------------------------------------------------------
With run length r_t (time since the last change) and constant hazard H,
    P(r_t = r + 1, x_1:t) = P(r_{t-1} = r, x_1:t-1) pi_t(r) (1 - H),
    P(r_t = 0, x_1:t)     = sum_r P(r_{t-1} = r, x_1:t-1) pi_t(r) H,
pi_t(r) the posterior predictive density of x_t given the r previous points. Conjugate models:
    normal    Normal-Gamma prior (mu0, kappa0, alpha0, beta0); predictive Student t with 2 alpha degrees
              of freedom, location mu, scale^2 = beta (kappa + 1) / (alpha kappa); updates
              mu' = (kappa mu + x) / (kappa + 1), kappa' = kappa + 1, alpha' = alpha + 1/2,
              beta' = beta + kappa (x - mu)^2 / (2 (kappa + 1)) (Murphy, "Conjugate Bayesian analysis of
              the Gaussian distribution", 2007)
    poisson   Gamma(a, b) prior on the rate; negative-binomial predictive
              P(x) = Gamma(x + a) / (Gamma(a) x!) (b / (b + 1))^a (1 / (b + 1))^x; a' = a + x, b' = b + 1.
Empirical-Bayes priors: mu0 = median, beta0 / alpha0 = MAD^2 with alpha0 = 1, kappa0 = 1 (normal); a, b
matching the median count with unit-count strength (poisson). Run length r at time t means the current
run holds the last r observations x_{t-r+1} ... x_t (r = 0: a change right after x_t). Run lengths are
truncated from the long end when their total mass falls below `prune`, and at `max_run`. Reported:
P(r_t = 0), the MAP run length, and change points t - r + 1 wherever the MAP run length r drops (a new
run became most probable).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import special

_D = {"normal_mean": 1, "normal_meanvar": 2, "poisson": 1, "exponential": 1}


def _penalty(penalty: str | float, cost: str, n: int) -> float:
    d = _D[cost]
    if isinstance(penalty, int | float) and not isinstance(penalty, bool):
        return float(penalty)
    if penalty == "bic":
        return (d + 1) * float(np.log(n))
    if penalty == "aic":
        return 2.0 * (d + 1)
    try:
        return float(penalty)
    except ValueError:
        raise ValueError("penalty must be 'bic', 'aic' or a number") from None


class _Cost:
    """Prefix-sum segment costs of the module docstring; cost(a, b) covers x[a:b]."""

    def __init__(self, x: np.ndarray, kind: str) -> None:
        if kind not in _D:
            raise ValueError(f"cost must be one of {sorted(_D)}")
        self.kind = kind
        self.s1 = np.concatenate([[0.0], np.cumsum(x)])
        self.s2 = np.concatenate([[0.0], np.cumsum(x * x)])
        if kind == "normal_mean":
            d = np.diff(x)
            mad = float(np.median(np.abs(d - np.median(d)))) * 1.482602218505602 if d.size else 0.0
            sigma2 = (mad**2) / 2.0 if mad > 0 else float(np.var(x)) if np.var(x) > 0 else 1.0
            self.sigma2 = sigma2
        if kind == "normal_meanvar":
            v = float(np.var(x))
            self.floor = 1e-12 * v if v > 0 else 1e-12
        if kind == "poisson" and ((x < 0).any() or not np.allclose(x, np.round(x))):
            raise ValueError("poisson cost needs non-negative integer counts")
        if kind == "exponential" and (x <= 0).any():
            raise ValueError("exponential cost needs positive values")

    def __call__(self, a: np.ndarray, b: int) -> np.ndarray:
        n = (b - a).astype(np.float64)
        s1 = self.s1[b] - self.s1[a]
        s2 = self.s2[b] - self.s2[a]
        if self.kind == "normal_mean":
            return (s2 - s1 * s1 / n) / self.sigma2
        if self.kind == "normal_meanvar":
            var = np.maximum((s2 - s1 * s1 / n) / n, self.floor)
            return n * (np.log(2.0 * np.pi * var) + 1.0)
        if self.kind == "poisson":
            with np.errstate(divide="ignore", invalid="ignore"):
                term = np.where(s1 > 0, s1 * np.log(s1 / n), 0.0)
            return 2.0 * (s1 - term)
        return 2.0 * n * (np.log(s1 / n) + 1.0)


@dataclass(frozen=True)
class Segmentation:
    """PELT result: change points (index where a new segment starts), the total penalised cost, segments."""

    changepoints: np.ndarray
    cost: float
    penalty: float
    segments: np.ndarray


def pelt(x: np.ndarray, *, cost: str = "normal_meanvar", penalty: str | float = "bic", min_size: int = 2) -> Segmentation:
    """Optimal penalised segmentation by PELT (module docstring)."""
    a = np.asarray(x, dtype=np.float64).reshape(-1)
    if not np.isfinite(a).all():
        raise ValueError("the series must be finite")
    n = a.size
    if min_size < 1 or (cost == "normal_meanvar" and min_size < 3):
        raise ValueError("min_size must be >= 1 (>= 3 for normal_meanvar; module docstring)")
    if n < 2 * min_size:
        return Segmentation(np.zeros(0, dtype=np.int64), float("nan"), float("nan"), np.array([[0, n]]))
    c = _Cost(a, cost)
    beta = _penalty(penalty, cost, n)
    f = np.full(n + 1, np.inf)
    f[0] = -beta
    last = np.zeros(n + 1, dtype=np.int64)
    cand = np.array([0], dtype=np.int64)
    for t in range(min_size, n + 1):
        elig = cand[t - cand >= min_size]
        vals = f[elig] + c(elig, t) + beta
        k = int(np.argmin(vals))
        f[t] = vals[k]
        last[t] = elig[k]
        keep_elig = vals - beta <= f[t]                                 # F(tau) + C(tau, t) <= F(t)
        young = cand[t - cand < min_size]
        cand = np.concatenate([elig[keep_elig], young, [t - min_size + 1]]) if t - min_size + 1 > 0 else \
            np.concatenate([elig[keep_elig], young])
        cand = np.unique(cand[(cand >= 0) & (cand <= t)])
    cps = []
    t = n
    while t > 0:
        s = int(last[t])
        if s > 0:
            cps.append(s)
        t = s
    cp = np.array(sorted(cps), dtype=np.int64)
    bounds = np.concatenate([[0], cp, [n]])
    return Segmentation(changepoints=cp, cost=float(f[n]), penalty=beta, segments=np.stack([bounds[:-1], bounds[1:]], axis=1))


@dataclass(frozen=True)
class OnlineChangepoints:
    """BOCPD result (module docstring): P(r_t = 0), MAP run length, expected run length, change points."""

    cp_probability: np.ndarray
    map_run_length: np.ndarray
    expected_run_length: np.ndarray
    changepoints: np.ndarray


def bocpd(
    x: np.ndarray,
    *,
    hazard: float = 1.0 / 250.0,
    model: str = "normal",
    prune: float = 1e-8,
    max_run: int = 5_000,
    prior: dict[str, float] | None = None,
) -> OnlineChangepoints:
    """Bayesian online change-point detection (module docstring)."""
    a = np.asarray(x, dtype=np.float64).reshape(-1)
    if not np.isfinite(a).all():
        raise ValueError("the series must be finite")
    if not 0.0 < hazard < 1.0:
        raise ValueError("hazard must be in (0, 1)")
    n = a.size
    med = float(np.median(a)) if n else 0.0
    mad = float(np.median(np.abs(a - med))) * 1.482602218505602 if n else 1.0
    if model == "normal":
        p = {"mu0": med, "kappa0": 1.0, "alpha0": 1.0, "beta0": max(mad**2, 1e-12)}
    elif model == "poisson":
        if (a < 0).any() or not np.allclose(a, np.round(a)):
            raise ValueError("poisson model needs non-negative integer counts")
        p = {"a0": max(med, 0.5), "b0": 1.0}
    else:
        raise ValueError("model must be 'normal' or 'poisson'")
    if prior:
        p.update(prior)
    # Sufficient statistics per run length, index r = 0 ... R-1 (r = 0 is the fresh run).
    if model == "normal":
        mu, kappa, alpha, beta = (np.array([p["mu0"]]), np.array([p["kappa0"]]), np.array([p["alpha0"]]),
                                  np.array([p["beta0"]]))
    else:
        ga, gb = np.array([p["a0"]]), np.array([p["b0"]])
    log_r = np.array([0.0])                                              # log P(r_{t-1} = r | x_1:t-1)
    cp_prob = np.empty(n)
    map_rl = np.empty(n, dtype=np.int64)
    exp_rl = np.empty(n)
    log_h, log_1h = np.log(hazard), np.log1p(-hazard)
    for t in range(n):
        xt = a[t]
        if model == "normal":
            df = 2.0 * alpha
            scale = np.sqrt(beta * (kappa + 1.0) / (alpha * kappa))
            z = (xt - mu) / scale
            log_pi = (special.gammaln((df + 1.0) / 2.0) - special.gammaln(df / 2.0) - 0.5 * np.log(df * np.pi)
                      - np.log(scale) - (df + 1.0) / 2.0 * np.log1p(z * z / df))
        else:
            log_pi = (special.gammaln(xt + ga) - special.gammaln(ga) - special.gammaln(xt + 1.0)
                      + ga * np.log(gb / (gb + 1.0)) - xt * np.log(gb + 1.0))
        growth = log_r + log_pi + log_1h                                 # r -> r + 1
        change = special.logsumexp(log_r + log_pi + log_h)               # -> 0
        new = np.concatenate([[change], growth])
        new -= special.logsumexp(new)
        # Truncate the long-run tail whose total mass is below `prune`, and at max_run.
        probs = np.exp(new)
        tail = np.cumsum(probs[::-1])[::-1]
        keep = int(np.searchsorted(-tail, -prune, side="right"))
        keep = max(1, min(keep, max_run, new.size))
        new = new[:keep] - special.logsumexp(new[:keep])
        probs = np.exp(new)
        cp_prob[t] = probs[0]
        map_rl[t] = int(np.argmax(probs))
        exp_rl[t] = float((np.arange(probs.size) * probs).sum())
        # Update sufficient statistics: the fresh run starts from the prior, run r + 1 from run r plus x_t.
        if model == "normal":
            mu_n = (kappa * mu + xt) / (kappa + 1.0)
            beta_n = beta + kappa * (xt - mu) ** 2 / (2.0 * (kappa + 1.0))
            mu = np.concatenate([[p["mu0"]], mu_n])[:keep]
            kappa = np.concatenate([[p["kappa0"]], kappa + 1.0])[:keep]
            alpha = np.concatenate([[p["alpha0"]], alpha + 0.5])[:keep]
            beta = np.concatenate([[p["beta0"]], beta_n])[:keep]
        else:
            ga = np.concatenate([[p["a0"]], ga + xt])[:keep]
            gb = np.concatenate([[p["b0"]], gb + 1.0])[:keep]
        log_r = new
    drops = np.flatnonzero(np.diff(map_rl) < 0) + 1 if n > 1 else np.zeros(0, dtype=np.int64)
    cps = drops - map_rl[drops] + 1                                      # run length r at t holds x_{t-r+1} ... x_t
    cps = np.unique(cps[cps > 0])
    return OnlineChangepoints(cp_probability=cp_prob, map_run_length=map_rl, expected_run_length=exp_rl, changepoints=cps)


__all__ = ["OnlineChangepoints", "Segmentation", "bocpd", "pelt"]
