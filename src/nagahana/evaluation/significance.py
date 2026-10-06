"""Paired significance tests, multiple-comparison control, effect sizes and aggregation over seeds.

Every comparison of two models is paired on the same evaluation units (thesis section on statistical
analysis): McNemar's test for paired binary decisions at a fixed threshold, DeLong's test for
correlated AUROCs, the Diebold-Mariano test for differences in proper scores over time, Wilcoxon's
signed-rank test across datasets or attack instances, and paired permutation tests where no
distributional form is assumed. The false discovery rate over a family of comparisons is controlled
with the Benjamini-Hochberg procedure (Holm's and Benjamini-Yekutieli's procedures are available).

DeLong's test

For two scores on the same units, with the DeLong covariance (ranking.delong_covariance),
z = (A1 - A2) / sqrt(V11 + V22 - 2 V12), two-sided p = 2 Phi(-|z|) (DeLong, DeLong and Clarke-Pearson,
Biometrics 44:837-845, 1988).

Exact McNemar test

With b the number of units model A decides correctly and B wrongly, and c the reverse, under H0 b is
Binomial(b + c, 1/2). The exact two-sided p-value is min(1, 2 P(X <= min(b, c))) and the mid-p value
subtracts the probability of the observed point, 2 P(X <= min(b, c)) - P(X = min(b, c)) (McNemar,
Psychometrika 12:153-157, 1947; Fagerland, Lydersen and Laake, BMC Medical Research Methodology
13:91, 2013, doi:10.1186/1471-2288-13-91, who recommend the mid-p version). The effect size is the
conditional odds ratio b / c with the exact interval obtained from the Clopper-Pearson interval of
b / (b + c).

Paired permutation test

Under H0 the labels "A" and "B" are exchangeable within each pair (or within each cluster of pairs,
for dependent units), so flipping the sign of the per-cluster sum of loss differences leaves the null
distribution unchanged. The p-value is computed exactly by enumerating all 2^C sign patterns when
C <= 16 clusters, and otherwise from R random patterns as (1 + #{|T*| >= |T|}) / (1 + R), which is a
valid p-value (Phipson and Smyth, Statistical Applications in Genetics and Molecular Biology 9:39,
2010, doi:10.2202/1544-6115.1585). `randomization_test` does the same for a metric that is not a mean
of per-unit losses (an F1 difference, for example) by swapping the two models' predictions on the
selected clusters and recomputing the metric difference (Noreen, Computer-Intensive Methods for
Testing Hypotheses, Wiley 1989).

Diebold-Mariano test

With loss differentials d_t = L_A(t) - L_B(t) of h-step forecasts, mean dbar over T periods and
autocovariances g_k = (1/T) sum_{t>k} (d_t - dbar)(d_{t-k} - dbar), the long-run variance is the
Newey-West estimate with Bartlett weights (Newey and West, Econometrica 55:703-708, 1987)

    s^2 = g_0 + 2 sum_{k=1}^{L} (1 - k / (L + 1)) g_k,   L = h - 1 by default,

DM = dbar / sqrt(s^2 / T) (Diebold and Mariano, JBES 13:253-263, 1995), and the Harvey-Leybourne-
Newbold correction (International Journal of Forecasting 13:281-291, 1997)

    DM* = DM sqrt((T + 1 - 2h + h (h - 1) / T) / T),

compared with Student's t on T - 1 degrees of freedom. With several independent series (one per
network), each series is centred on its own mean for its autocovariances, the pooled mean is
sum_s T_s dbar_s / T, and Var(dbar) = sum_s T_s s_s^2 / T^2; the correction and the degrees of freedom
then use the total length T (an approximation recorded as an assumption).

Wilcoxon signed-rank test

scipy.stats.wilcoxon (exact distribution for small samples without ties, normal approximation with
tie correction otherwise); zero differences are dropped (Wilcoxon's convention) unless "pratt" or
"zsplit" is chosen. The effect size is the matched-pairs rank-biserial correlation
r = (W+ - W-) / (W+ + W-) (Kerby, Comprehensive Psychology 3:11.IT.3.1, 2014).

Multiple comparisons

Benjamini-Hochberg (JRSS-B 57:289-300, 1995): adjusted p_(i) = min_{j >= i} min(1, m p_(j) / j).
Benjamini-Yekutieli (Annals of Statistics 29:1165-1188, 2001) multiplies by c(m) = sum_{i<=m} 1/i
and controls the FDR under arbitrary dependence. Holm (Scandinavian Journal of Statistics 6:65-70,
1979): adjusted p_(i) = max_{j <= i} min(1, (m - j + 1) p_(j)), controlling the family-wise error rate.

Seeds

Every configuration is trained with several seeds. `aggregate_seeds` reports the mean over seeds,
their standard deviation and a Student-t interval for the seed mean. A paired test between two models
trained with several seeds is run per matched seed, and the comparison reports the largest of the
per-seed p-values: rejecting only when every seed rejects is a valid level-alpha test of the
hypothesis that the difference holds for every seed (the intersection-union principle, Berger,
Technometrics 24:295-300, 1982).
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import stats

from nagahana.core.errors import InvariantViolation
from nagahana.evaluation._arrays import as_binary, as_float, codes
from nagahana.evaluation.ranking import delong_covariance


@dataclass(frozen=True)
class HypothesisTest:
    """Outcome of a paired test: the statistic, its p-value and the effect it tests."""

    test: str
    statistic: float
    p_value: float
    estimate: float
    ci_low: float = math.nan
    ci_high: float = math.nan
    n: int = 0
    df: float = math.nan
    detail: dict[str, float] = field(default_factory=dict)


def _standardised(estimate: float, se: float) -> float:
    # estimate / se; with a zero standard error, 0 for a zero estimate and an infinity of its sign otherwise.
    if se > 0.0:
        return estimate / se
    return 0.0 if estimate == 0.0 else math.copysign(math.inf, estimate)


def _odds(pi: float) -> float:
    # Odds pi / (1 - pi) of a proportion (infinite at 1).
    return pi / (1.0 - pi) if pi < 1.0 else math.inf


def delong_test(score_a: Any, score_b: Any, label: Any, *, confidence: float = 0.95) -> HypothesisTest:
    """DeLong's test of AUROC_A = AUROC_B for two scores of the same units (Wald interval of the difference)."""
    sa, sb = as_float("score_a", score_a, 1), as_float("score_b", score_b, 1)
    if sa.shape != sb.shape:
        raise InvariantViolation("both scores must cover the same units")
    auc, cov = delong_covariance(np.vstack([sa, sb]), label)
    diff = float(auc[0] - auc[1])
    var = float(cov[0, 0] + cov[1, 1] - 2.0 * cov[0, 1])
    se = math.sqrt(max(var, 0.0))
    z = _standardised(diff, se)
    p = float(2.0 * stats.norm.sf(abs(z)))
    q = float(stats.norm.ppf(0.5 + confidence / 2.0))
    y = as_binary("label", label)
    return HypothesisTest("delong", z, p, diff, diff - q * se, diff + q * se, n=int(y.size),
                      detail={"auroc_a": float(auc[0]), "auroc_b": float(auc[1]), "se": se})


def mcnemar_test(correct_a: Any, correct_b: Any, *, mid_p: bool = True, confidence: float = 0.95) -> HypothesisTest:
    """Exact (or mid-p) McNemar test on paired correctness indicators [n]; estimate = odds ratio b / c."""
    ca = as_binary("correct_a", correct_a).astype(bool)
    cb = as_binary("correct_b", correct_b).astype(bool)
    if ca.shape != cb.shape:
        raise InvariantViolation("both indicator vectors must cover the same units")
    b = int(np.sum(ca & ~cb))
    c = int(np.sum(~ca & cb))
    m = b + c
    if m == 0:
        return HypothesisTest("mcnemar_midp" if mid_p else "mcnemar_exact", 0.0, 1.0, math.nan, n=int(ca.size),
                          detail={"b": 0.0, "c": 0.0})
    k = min(b, c)
    tail = float(stats.binom.cdf(k, m, 0.5))
    p_exact = min(1.0, 2.0 * tail)
    p = min(1.0, 2.0 * tail - float(stats.binom.pmf(k, m, 0.5))) if mid_p else p_exact
    # Odds ratio b / c with the exact interval from the Clopper-Pearson interval of pi = b / (b + c).
    ci = stats.binomtest(b, m, 0.5).proportion_ci(confidence_level=confidence, method="exact")
    odds = b / c if c > 0 else math.inf
    return HypothesisTest("mcnemar_midp" if mid_p else "mcnemar_exact", float(b - c) / math.sqrt(m), p, odds,
                          _odds(ci.low), _odds(ci.high), n=int(ca.size),
                          detail={"b": float(b), "c": float(c), "p_exact": p_exact})


def _cluster_sums(d: np.ndarray, clusters: Any) -> np.ndarray:
    # Sum of the differences per cluster (each unit its own cluster when clusters is None).
    if clusters is None:
        return d
    c, _ = codes(clusters)
    if c.shape != d.shape:
        raise InvariantViolation("clusters must label every unit")
    return np.bincount(c, weights=d)


def _sign_patterns(n_clusters: int, exact_max: int, n_random: int, rng: np.random.Generator) -> tuple[np.ndarray, bool]:
    # All 2^C sign vectors when C is small (exact), otherwise random sign vectors.
    if n_clusters <= exact_max:
        pats = np.array(list(itertools.product((1.0, -1.0), repeat=n_clusters)))
        return pats, True
    return rng.choice((1.0, -1.0), size=(n_random, n_clusters)), False


def paired_permutation_test(loss_a: Any, loss_b: Any, *, clusters: Any = None, n_permutations: int = 10_000,
                            rng: np.random.Generator | None = None, exact_max: int = 16,
                            alternative: str = "two-sided") -> HypothesisTest:
    """Sign-flip test of mean(loss_a - loss_b) = 0, flipping whole clusters when given."""
    a, b = as_float("loss_a", loss_a, 1), as_float("loss_b", loss_b, 1)
    if a.shape != b.shape:
        raise InvariantViolation("both losses must cover the same units")
    d = a - b
    sums = _cluster_sums(d, clusters)
    n = d.size
    observed = float(sums.sum()) / n
    gen = rng if rng is not None else np.random.default_rng(0)
    pats, exact = _sign_patterns(sums.size, exact_max, n_permutations, gen)
    null = pats @ sums / n
    if alternative == "two-sided":
        extreme = np.abs(null) >= abs(observed) - 1e-15
    elif alternative == "greater":
        extreme = null >= observed - 1e-15
    elif alternative == "less":
        extreme = null <= observed + 1e-15
    else:
        raise ValueError("alternative must be 'two-sided', 'greater' or 'less'")
    p = float(extreme.mean()) if exact else float((1 + extreme.sum()) / (1 + null.size))
    return HypothesisTest("paired_permutation", observed, p, observed, n=n,
                      detail={"clusters": float(sums.size), "exact": float(exact)})


def randomization_test(stat: Callable[[np.ndarray], np.ndarray], n_clusters: int, *, n_permutations: int = 2000,
                       rng: np.random.Generator | None = None, exact_max: int = 12) -> HypothesisTest:
    """Approximate randomisation test of a paired metric difference.

    `stat(swap)` takes a boolean matrix [R, C] (True: swap the two models' predictions on cluster c)
    and returns the metric difference A - B for each row [R]. Row 0 of the evaluated matrix is the
    observed arrangement (no swaps). Two-sided.
    """
    gen = rng if rng is not None else np.random.default_rng(0)
    pats, exact = _sign_patterns(n_clusters, exact_max, n_permutations, gen)
    swaps = pats < 0
    observed = float(np.asarray(stat(np.zeros((1, n_clusters), dtype=bool)))[0])
    null = np.asarray(stat(swaps), dtype=np.float64)
    finite = null[np.isfinite(null)]
    if not math.isfinite(observed) or finite.size == 0:
        return HypothesisTest("randomization", observed, math.nan, observed, detail={"clusters": float(n_clusters)})
    extreme = np.abs(finite) >= abs(observed) - 1e-15
    p = float(extreme.mean()) if exact else float((1 + extreme.sum()) / (1 + finite.size))
    return HypothesisTest("randomization", observed, p, observed, detail={"clusters": float(n_clusters), "exact": float(exact)})


def newey_west_variance(d: np.ndarray, lags: int) -> float:
    """Long-run variance g_0 + 2 sum_{k=1}^{L} (1 - k/(L+1)) g_k of a series (Bartlett weights)."""
    t = d.size
    if t == 0:
        return math.nan
    e = d - d.mean()
    s2 = float(e @ e) / t
    for k in range(1, min(lags, t - 1) + 1):
        s2 += 2.0 * (1.0 - k / (lags + 1.0)) * float(e[k:] @ e[:-k]) / t
    return s2


def diebold_mariano(loss_a: Any, loss_b: Any, *, horizon: int = 1, series: Any = None, time: Any = None,
                    lags: int | None = None, hln: bool = True, alternative: str = "two-sided") -> HypothesisTest:
    """Diebold-Mariano test of equal expected loss with Newey-West variance and the HLN correction.

    loss_a, loss_b: per-period losses [T]; series: optional series labels [T] (independent series);
    time: optional times [T] used to order each series (otherwise the given order is used).
    The estimate is mean(loss_a - loss_b): negative means model A has the lower loss.
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    a, b = as_float("loss_a", loss_a, 1), as_float("loss_b", loss_b, 1)
    if a.shape != b.shape:
        raise InvariantViolation("both losses must cover the same periods")
    d = a - b
    lag = horizon - 1 if lags is None else lags
    if series is None:
        groups = [np.arange(d.size)]
    else:
        c, _ = codes(series)
        groups = [np.flatnonzero(c == s) for s in np.unique(c)]
    t_all = np.asarray(time, dtype=np.float64) if time is not None else None
    total, var_sum = 0, 0.0
    for idx in groups:
        if t_all is not None:
            idx = idx[np.argsort(t_all[idx], kind="stable")]
        ds = d[idx]
        if ds.size == 0:
            continue
        s2 = newey_west_variance(ds, lag)
        var_sum += ds.size * s2
        total += ds.size
    if total < 2:
        return HypothesisTest("diebold_mariano", math.nan, math.nan, float(d.mean()) if d.size else math.nan, n=total)
    dbar = float(d.mean())
    var_mean = var_sum / (total * total)
    stat = _standardised(dbar, math.sqrt(max(var_mean, 0.0)))
    df = float(total - 1)
    if hln:
        h = horizon
        factor = (total + 1.0 - 2.0 * h + h * (h - 1.0) / total) / total
        stat = stat * math.sqrt(max(factor, 0.0))
        dist: Any = stats.t(df)
    else:
        dist = stats.norm()
    if alternative == "two-sided":
        p = float(2.0 * dist.sf(abs(stat)))
    elif alternative == "less":
        p = float(dist.cdf(stat))
    elif alternative == "greater":
        p = float(dist.sf(stat))
    else:
        raise ValueError("alternative must be 'two-sided', 'less' or 'greater'")
    se = math.sqrt(max(var_mean, 0.0))
    q = float(dist.ppf(0.975))
    return HypothesisTest("diebold_mariano", stat, p, dbar, dbar - q * se, dbar + q * se, n=total, df=df if hln else math.inf,
                      detail={"series": float(len(groups)), "lags": float(lag)})


def wilcoxon_signed_rank(x: Any, y: Any = None, *, zero_method: str = "wilcox", alternative: str = "two-sided",
                         correction: bool = False) -> HypothesisTest:
    """Wilcoxon signed-rank test of paired samples (or of x alone); estimate = median difference."""
    xa = as_float("x", x, 1)
    d = xa - as_float("y", y, 1) if y is not None else xa
    nz = d[d != 0] if zero_method == "wilcox" else d
    if nz.size == 0:
        return HypothesisTest("wilcoxon", 0.0, 1.0, 0.0, n=int(d.size), detail={"rank_biserial": 0.0})
    res = stats.wilcoxon(d, zero_method=zero_method, alternative=alternative, correction=correction)
    ranks = stats.rankdata(np.abs(nz))
    w_plus, w_minus = float(ranks[nz > 0].sum()), float(ranks[nz < 0].sum())
    rbc = (w_plus - w_minus) / (w_plus + w_minus) if (w_plus + w_minus) > 0 else 0.0
    return HypothesisTest("wilcoxon", float(res.statistic), float(res.pvalue), float(np.median(d)), n=int(d.size),
                      detail={"rank_biserial": rbc, "w_plus": w_plus, "w_minus": w_minus})


def adjust_pvalues(p_values: Sequence[float], method: str = "bh") -> np.ndarray:
    """Adjusted p-values: 'bh' (Benjamini-Hochberg), 'by' (Benjamini-Yekutieli), 'holm', 'bonferroni'.

    NaN p-values (tests that could not be computed) stay NaN and do not count towards m.
    """
    p = np.asarray(p_values, dtype=np.float64)
    out = np.full(p.shape, np.nan)
    ok = np.flatnonzero(np.isfinite(p))
    m = ok.size
    if m == 0:
        return out
    if np.any((p[ok] < 0) | (p[ok] > 1)):
        raise InvariantViolation("p-values must lie in [0, 1]")
    pv = p[ok]
    order = np.argsort(pv, kind="stable")
    ranked = pv[order]
    i = np.arange(1, m + 1)
    if method in ("bh", "by"):
        scale = m / i
        if method == "by":
            scale = scale * float(np.sum(1.0 / i))
        adj = np.minimum.accumulate((ranked * scale)[::-1])[::-1]
    elif method == "holm":
        adj = np.maximum.accumulate(ranked * (m - i + 1))
    elif method == "bonferroni":
        adj = ranked * m
    else:
        raise ValueError("method must be 'bh', 'by', 'holm' or 'bonferroni'")
    res = np.empty(m)
    res[order] = np.minimum(adj, 1.0)
    out[ok] = res
    return out


def cohen_dz(diff: Any) -> float:
    """Standardised mean of paired differences, mean(d) / sd(d) (Cohen 1988); NaN when sd = 0."""
    d = as_float("diff", diff, 1)
    if d.size < 2:
        return math.nan
    sd = float(d.std(ddof=1))
    return float(d.mean()) / sd if sd > 0 else math.nan


def hedges_correction(n: int) -> float:
    """Small-sample correction J(df) = 1 - 3 / (4 df - 1), df = n - 1 (Hedges, J. Educ. Stat. 6:107-128, 1981)."""
    df = n - 1
    return 1.0 - 3.0 / (4.0 * df - 1.0) if df > 0 else math.nan


def cliffs_delta(a: Any, b: Any) -> float:
    """P(A > B) - P(A < B) over all pairs (Cliff, Psychological Bulletin 114:494-509, 1993)."""
    xa, xb = as_float("a", a, 1), as_float("b", b, 1)
    if xa.size == 0 or xb.size == 0:
        return math.nan
    sb = np.sort(xb)
    greater = np.searchsorted(sb, xa, side="left").sum()               # pairs with b < a
    less = (xb.size - np.searchsorted(sb, xa, side="right")).sum()     # pairs with b > a
    return float(greater - less) / (xa.size * xb.size)


@dataclass(frozen=True)
class SeedSummary:
    """Mean over seeds, spread and a Student-t interval of the seed mean."""

    mean: float
    sd: float
    se: float
    low: float
    high: float
    n_seeds: int


def aggregate_seeds(values: Sequence[float], *, confidence: float = 0.95) -> SeedSummary:
    """Summarise one metric over seeds (NaN values are ignored and not counted)."""
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    n = v.size
    if n == 0:
        return SeedSummary(math.nan, math.nan, math.nan, math.nan, math.nan, 0)
    mean = float(v.mean())
    if n == 1:
        return SeedSummary(mean, math.nan, math.nan, math.nan, math.nan, 1)
    sd = float(v.std(ddof=1))
    se = sd / math.sqrt(n)
    q = float(stats.t.ppf(0.5 + confidence / 2.0, n - 1))
    return SeedSummary(mean, sd, se, mean - q * se, mean + q * se, n)


def intersection_union_p(p_values: Sequence[float]) -> float:
    """Largest of the per-seed p-values (NaN when none is finite)."""
    p = np.asarray(p_values, dtype=np.float64)
    p = p[np.isfinite(p)]
    return float(p.max()) if p.size else math.nan
