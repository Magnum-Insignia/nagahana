"""Information audit (proposal P-15): what the observables reveal about a hidden quantity, per observation
regime, and the Bayes ceiling a model is judged against.

Before a model is asked to forecast the hidden stage S of an intrusion from an observation regime
(full packets, flow records only, encrypted payloads, an untapped segment), the audit measures how
much the regime's observables O carry about S:

    I(O; S) = sum_{o, s} p(o, s) log(p(o, s) / (p(o) p(s)))      [nats]

If I(O; S) is small, no model can recover S from O well, and the remedy is better sensing, not a
larger model. On a simulated world with known hidden state (P-14) the audit gives the ceiling that
NagaHana's performance is compared against; every function takes plain arrays (observables, hidden
labels, regime ids), so the simulator stays independent of the audit.

Estimators (`analytics.information`)
------------------------------------
Records with absent fields (NaN, D-41) are handled by the chain rule
I(O; S) = I(D; S) + sum_d P(D = d) I(C; S | D = d), D = (observation pattern, discrete fields): the
plug-in estimator with the Miller-Madow correction for D, the Ross (2014) k-NN estimator for the
continuous fields C inside each pattern. `mutual_information` (all discrete) is the plug-in estimate
with an optional Miller-Madow correction -(|O, S| - |O| - |S| + 1) / (2n) over occupied cells.
k-sensitivity: a small k lowers the bias of the k-NN estimators and raises their variance, a large k
the reverse (Kraskov et al. 2004, section IV). The estimate is computed for every k of `k_values`
(default 3, 5, 10), each is reported, and their median is the point estimate, so that no single
high-variance draw sets the ceiling. Intervals: half-sample subsampling of the same median estimator
(k-NN estimators do not tolerate the duplicated points of a bootstrap;
`analytics.information.subsample_interval`).

Bounds on the Bayes error R* of predicting S (M classes) from O
---------------------------------------------------------------
    Fano (Fano 1961; Cover and Thomas, "Elements of Information Theory", 2nd ed., Wiley 2006, Thm 2.10.1):
        H(S | O) <= h(P_e) + P_e log(M - 1), so R* >= the root P of h(P) + P log(M - 1) = H(S | O) on
        [0, (M - 1) / M] (h the binary entropy; H(S | O) = H(S) - I(O; S)).
    Hellman-Raviv (IEEE Trans. Information Theory 16(4):368-372, 1970): R* <= H(S | O) / 2 with H in bits.
    Trivial: R* <= 1 - max_s p(s) (always predict the most frequent class).
    Cover-Hart (IEEE Trans. Information Theory 13(1):21-27, 1967): asymptotically
        R* <= R_NN <= R* (2 - M R* / (M - 1)),
        so R* >= ((M - 1) / M) (1 - sqrt(1 - M R_NN / (M - 1))) and R* <= R_NN, with R_NN the
        leave-one-out 1-nearest-neighbour error. Neighbours are searched inside each observation pattern
        only (D-41: a distance across different observed fields is undefined), on slog1p values scaled by
        their MAD (AS-31); ties are broken at random by a seeded jitter of 1e-10 standard deviations. In a
        pattern with no continuous field every other record of the pattern is a nearest neighbour, and the
        expected leave-one-out error of the random tie-break, 1 - (c(d, s_i) - 1) / (n_d - 1), is used
        exactly. The bounds hold in the large-sample limit; R_NN is a finite-sample estimate.
    Plug-in (discrete observables): resubstitution 1 - sum_d max_s n(d, s) / n (optimistic) and two-fold
        cross-validated error of the plug-in rule (pessimistic: a cell unseen in the training half gets the
        majority class), averaged over both directions; they bracket R* in expectation.
The ceiling bracket is R* in [max of the lower bounds, min of the upper bounds] and accuracy in
[1 - upper, 1 - lower]. H(S) enters the bounds as its plug-in value (its bias -(M - 1) / (2n) is
negligible for M classes much fewer than n), so that with I(O; S) clipped at 0 the Fano bound never
exceeds the majority-class error. Every bound here is estimated from data; when two tight estimates
cross by sampling noise (lower > upper, typical when R* is pinned down from both sides),
`bracket_crossed` is set and the accuracy ceiling is reported over the interval they span.

Forecast horizon
----------------
`lagged_audit` pairs O at step i with S at step i + k inside each sequence (an entity, a trigger stream),
measuring the predictive information I(O_t; S_{t+k}) and its ceiling for each lag k: what any
forecaster could know k steps ahead.

Judging a model
---------------
`model_gap`: a model's error against the bracket (gap_min = error - upper(R*) is a guaranteed excess when
positive; gap_max = error - lower(R*) the largest possible excess), and its information efficiency
I(S_hat; S) / I(O; S) <= 1 up to estimation error (data-processing inequality; S_hat is a function of O).

Gate: proposal P-15 ("information-audit") must be enabled (`enabled_proposals`) for any of these
functions to run.
"""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import optimize
from scipy.spatial import cKDTree

from nagahana.analytics import information as info
from nagahana.analytics.config import InfoAuditConfig, to_dict
from nagahana.analytics.dependence import slog1p
from nagahana.analytics.report import Report
from nagahana.analytics.robust import MAD_NORMAL
from nagahana.governance import decisions

_GATE = "information-audit"


def mutual_information(
    obs: object,
    hidden: object,
    *,
    enabled_proposals: Collection[str],
    miller_madow: bool,
) -> float:
    """Plug-in estimate of I(O; S) in nats for discrete samples (optionally Miller-Madow corrected). Gated by P-15.

    With the correction the estimate is clipped at 0 (mutual information is never negative).
    """
    decisions.require_proposal(_GATE, enabled_proposals)
    o = np.asarray(obs, dtype=object).reshape(-1)
    s = np.asarray(hidden, dtype=object).reshape(-1)
    if o.size != s.size or o.size == 0:
        raise ValueError("obs and hidden must be non-empty and of equal length")
    mi = info.mutual_information_discrete(o, s, correction="miller_madow" if miller_madow else "none")
    return max(mi, 0.0) if miller_madow else mi


def binary_entropy(p: float) -> float:
    """h(p) = -p log p - (1 - p) log(1 - p) in nats."""
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return float(-p * np.log(p) - (1.0 - p) * np.log1p(-p))


def fano_lower_bound(h_cond: float, n_classes: int) -> float:
    """Smallest error probability allowed by Fano's inequality for H(S | O) = h_cond nats (module docstring)."""
    if n_classes < 2:
        return 0.0
    top = (n_classes - 1) / n_classes
    if h_cond <= 0:
        return 0.0
    if h_cond >= np.log(n_classes):
        return top

    def g(p: float) -> float:
        return binary_entropy(p) + p * np.log(n_classes - 1) - h_cond

    return float(optimize.brentq(g, 0.0, top, xtol=1e-14, rtol=1e-12))


def hellman_raviv_upper_bound(h_cond: float) -> float:
    """R* <= H(S | O) / 2 with H in bits (h_cond given in nats)."""
    return float(max(h_cond, 0.0) / (2.0 * np.log(2.0)))


def cover_hart_bounds(nn_error: float, n_classes: int) -> tuple[float, float]:
    """(lower, upper) bounds on R* from the 1-NN error (module docstring)."""
    if not np.isfinite(nn_error) or n_classes < 2:
        return float("nan"), float("nan")
    c = n_classes / (n_classes - 1.0)
    inside = max(0.0, 1.0 - c * min(nn_error, 1.0 / c))
    return float((1.0 / c) * (1.0 - np.sqrt(inside))), float(nn_error)


def _scaled(values: np.ndarray) -> np.ndarray:
    """slog1p values scaled by their column MAD (or standard deviation), NaN kept."""
    z = slog1p(values)
    out = np.full_like(z, np.nan)
    for j in range(z.shape[1]):
        col = z[:, j]
        ok = np.isfinite(col)
        if not ok.any():
            continue
        med = np.median(col[ok])
        mad = np.median(np.abs(col[ok] - med)) * MAD_NORMAL
        sd = col[ok].std()
        scale = mad if mad > 0 else (sd if sd > 0 else 1.0)
        out[ok, j] = (col[ok] - med) / scale
    return out


def nn_error_loo(values: np.ndarray, hidden: np.ndarray, *, discrete: np.ndarray, jitter: float = 1e-10,
                 seed: int = 0) -> float:
    """Leave-one-out 1-NN error inside observation patterns, expected under random tie-breaking (module docstring)."""
    v = np.asarray(values, dtype=np.float64)
    v = v.reshape(-1, 1) if v.ndim == 1 else v
    disc = np.asarray(discrete, dtype=bool)
    s_codes, _ = info.factorize(np.asarray(hidden, dtype=object))
    n, d = v.shape
    observed = np.isfinite(v)
    parts = [observed[:, j] for j in range(d)] + [np.where(observed[:, j], v[:, j], -np.inf) for j in np.flatnonzero(disc)]
    dcode, _ = info.factorize(*parts) if parts else (np.zeros(n, dtype=np.int64), 1)
    z = _scaled(np.where(disc[None, :], np.nan, v)) if (~disc).any() else np.zeros((n, 0))
    rng = np.random.default_rng(seed)
    errors = np.zeros(n)
    order = np.argsort(dcode, kind="stable")
    for rows in np.split(order, np.flatnonzero(np.diff(dcode[order])) + 1):
        if rows.size == 1:
            errors[rows] = np.nan                                       # no neighbour in its pattern
            continue
        cols = np.flatnonzero(~disc & observed[rows[0]])
        s_r = s_codes[rows]
        if cols.size == 0:
            counts = np.bincount(s_r)
            errors[rows] = 1.0 - (counts[s_r] - 1.0) / (rows.size - 1.0)
            continue
        x = z[np.ix_(rows, cols)] + rng.normal(0.0, jitter, size=(rows.size, cols.size))
        _, idx = cKDTree(x).query(x, k=2, p=np.inf)
        errors[rows] = (s_r[idx[:, 1]] != s_r).astype(np.float64)
    ok = np.isfinite(errors)
    return float(errors[ok].mean()) if ok.any() else float("nan")


def plugin_bayes_error(obs_codes: np.ndarray, hidden: np.ndarray, *, seed: int = 0) -> tuple[float, float]:
    """(resubstitution, two-fold cross-validated) error of the plug-in Bayes rule on discrete observables."""
    o, _ = info.factorize(np.asarray(obs_codes, dtype=object))
    s, k = info.factorize(np.asarray(hidden, dtype=object))
    n = o.size
    table = pd.crosstab(o, s)
    resub = 1.0 - table.max(axis=1).sum() / n
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    halves = (perm[: n // 2], perm[n // 2:])
    errs = []
    for a, b in (halves, halves[::-1]):
        if a.size == 0 or b.size == 0:
            continue
        tab = pd.crosstab(o[a], s[a])
        rule = tab.idxmax(axis=1)                                        # most frequent class per observation
        default = int(np.bincount(s[a], minlength=k).argmax())
        pred = pd.Series(o[b]).map(rule).fillna(default).to_numpy(dtype=np.int64)
        errs.append(float((pred != s[b]).mean()))
    return float(resub), float(np.mean(errs)) if errs else float("nan")


@dataclass(frozen=True)
class Ceiling:
    """Information and Bayes-error bracket of one regime (module docstring)."""

    regime: str
    n: int
    classes: int
    entropy_hidden: float
    mi: float
    mi_by_k: str
    mi_lower: float
    mi_upper: float
    mi_discrete_part: float
    mi_continuous_part: float
    unestimated_mass: float
    conditional_entropy: float
    normalized_mi: float
    fano_lower: float
    hellman_raviv_upper: float
    majority_upper: float
    nn_error: float
    cover_hart_lower: float
    cover_hart_upper: float
    plugin_resubstitution: float
    plugin_cross_validated: float
    bayes_error_lower: float
    bayes_error_upper: float
    bracket_crossed: bool
    accuracy_ceiling_lower: float
    accuracy_ceiling_upper: float


def regime_ceiling(values: np.ndarray, hidden: np.ndarray, *, discrete: np.ndarray, regime: str = "all",
                   cfg: InfoAuditConfig | None = None, enabled_proposals: Collection[str] = ()) -> Ceiling:
    """The information audit of one regime (module docstring)."""
    cfg = cfg or InfoAuditConfig()
    decisions.require_proposal(_GATE, tuple(enabled_proposals) + tuple(cfg.enabled_proposals))
    v = np.asarray(values, dtype=np.float64)
    v = v.reshape(-1, 1) if v.ndim == 1 else v
    h = np.asarray(hidden, dtype=object).reshape(-1)
    disc = np.asarray(discrete, dtype=bool).reshape(-1)
    n = h.size
    s_codes, m_classes = info.factorize(h)
    h_s = info.entropy_from_counts(np.bincount(s_codes))                # plug-in H(S) (module docstring)
    if not cfg.k_values or min(cfg.k_values) < 1:
        raise ValueError("k_values must hold positive neighbour counts")

    def estimates(idx: np.ndarray) -> list[info.ObservablesMI]:
        return [info.mutual_information_observables(v[idx], h[idx], discrete=disc, k=k, min_stratum=cfg.min_stratum,
                                                    correction=cfg.correction, jitter=cfg.jitter, seed=cfg.seed)
                for k in cfg.k_values]

    by_k = estimates(np.arange(n))
    totals = np.array([e.total for e in by_k])
    pick = by_k[int(np.argsort(totals, kind="stable")[(totals.size - 1) // 2])]   # the (lower) median estimate
    est_total = float(np.median(totals))

    def stat(idx: np.ndarray) -> float:
        return float(np.median([e.total for e in estimates(idx)]))

    lo, hi, _ = info.subsample_interval(stat, n, estimate=est_total, n_sub=cfg.n_sub, level=cfg.level, seed=cfg.seed) \
        if n >= 8 and cfg.n_sub >= 2 else (float("nan"), float("nan"), None)
    mi = float(np.clip(est_total, 0.0, h_s))
    h_cond = float(max(h_s - mi, 0.0))
    fano = fano_lower_bound(h_cond, m_classes)
    hr = hellman_raviv_upper_bound(h_cond)
    majority = 1.0 - np.bincount(s_codes).max() / n
    rng = np.random.default_rng(cfg.seed)
    sub = np.sort(rng.choice(n, size=cfg.nn_max_rows, replace=False)) if n > cfg.nn_max_rows else np.arange(n)
    r_nn = nn_error_loo(v[sub], h[sub], discrete=disc, jitter=cfg.jitter, seed=cfg.seed)
    ch_lo, ch_hi = cover_hart_bounds(r_nn, m_classes)
    if disc.all() or (~np.isfinite(v[:, ~disc])).all():
        obs_codes = info.factorize(*[np.where(np.isfinite(v[:, j]), v[:, j], -np.inf) for j in range(v.shape[1])])[0]
        resub, cv = plugin_bayes_error(obs_codes, h, seed=cfg.seed)
    else:
        resub, cv = float("nan"), float("nan")
    lowers = [x for x in (fano, ch_lo, resub) if np.isfinite(x)]
    uppers = [x for x in (hr, majority, ch_hi, cv, (m_classes - 1) / m_classes) if np.isfinite(x)]
    lower = max(lowers) if lowers else 0.0
    upper = min(uppers) if uppers else float("nan")
    crossed = bool(np.isfinite(upper) and lower > upper)
    return Ceiling(
        regime=regime, n=int(n), classes=int(m_classes), entropy_hidden=float(h_s), mi=est_total,
        mi_by_k=", ".join(f"k={k}: {t:.6g}" for k, t in zip(cfg.k_values, totals.tolist(), strict=True)), mi_lower=lo,
        mi_upper=hi, mi_discrete_part=pick.discrete, mi_continuous_part=pick.continuous,
        unestimated_mass=max(e.unestimated_mass for e in by_k), conditional_entropy=h_cond,
        normalized_mi=float(mi / h_s) if h_s > 0 else float("nan"), fano_lower=fano, hellman_raviv_upper=hr,
        majority_upper=float(majority), nn_error=r_nn, cover_hart_lower=ch_lo, cover_hart_upper=ch_hi,
        plugin_resubstitution=resub, plugin_cross_validated=cv, bayes_error_lower=float(lower),
        bayes_error_upper=float(upper), bracket_crossed=crossed,
        accuracy_ceiling_lower=float(1.0 - max(lower, upper)) if np.isfinite(upper) else float("nan"),
        accuracy_ceiling_upper=float(1.0 - min(lower, upper)) if np.isfinite(upper) else float(1.0 - lower),
    )


def audit(values: np.ndarray, hidden: np.ndarray, *, discrete: np.ndarray, regimes: np.ndarray | None = None,
          cfg: InfoAuditConfig | None = None, enabled_proposals: Collection[str] = ()) -> pd.DataFrame:
    """One `Ceiling` row per regime (and "all" when there are several regimes)."""
    cfg = cfg or InfoAuditConfig()
    decisions.require_proposal(_GATE, tuple(enabled_proposals) + tuple(cfg.enabled_proposals))
    v = np.asarray(values, dtype=np.float64)
    h = np.asarray(hidden, dtype=object).reshape(-1)
    reg = np.full(h.size, "all", dtype=object) if regimes is None else np.asarray(regimes, dtype=object).reshape(-1)
    if reg.size != h.size:
        raise ValueError("regimes must have one entry per record")
    rows = []
    names = list(pd.unique(reg))
    for name in names:
        m = reg == name
        rows.append(regime_ceiling(v[m], h[m], discrete=discrete, regime=str(name), cfg=cfg,
                                   enabled_proposals=enabled_proposals).__dict__)
    if len(names) > 1:
        rows.append(regime_ceiling(v, h, discrete=discrete, regime="all", cfg=cfg, enabled_proposals=enabled_proposals).__dict__)
    return pd.DataFrame(rows)


def lagged_pairs(groups: np.ndarray, order: np.ndarray, lag: int) -> tuple[np.ndarray, np.ndarray]:
    """(source rows, target rows): row i and the row `lag` steps later in the same group, by `order`."""
    if lag < 0:
        raise ValueError("lag must be >= 0")
    g = np.asarray(groups, dtype=object).reshape(-1)
    codes, _ = info.factorize(g)
    srt = np.lexsort((np.asarray(order, dtype=np.float64), codes))
    sc = codes[srt]
    if lag == 0:
        return srt, srt
    ok = np.zeros(srt.size, dtype=bool)
    ok[: srt.size - lag] = sc[: srt.size - lag] == sc[lag:]
    src = srt[np.flatnonzero(ok)]
    tgt = srt[np.flatnonzero(ok) + lag]
    return src, tgt


def lagged_audit(values: np.ndarray, hidden: np.ndarray, *, discrete: np.ndarray, groups: np.ndarray, order: np.ndarray,
                 lags: tuple[int, ...] | None = None, regimes: np.ndarray | None = None, cfg: InfoAuditConfig | None = None,
                 enabled_proposals: Collection[str] = ()) -> pd.DataFrame:
    """I(O_t; S_{t+k}) and its ceiling for every lag k (module docstring, "Forecast horizon")."""
    cfg = cfg or InfoAuditConfig()
    decisions.require_proposal(_GATE, tuple(enabled_proposals) + tuple(cfg.enabled_proposals))
    v = np.asarray(values, dtype=np.float64)
    v = v.reshape(-1, 1) if v.ndim == 1 else v
    h = np.asarray(hidden, dtype=object).reshape(-1)
    reg = np.full(h.size, "all", dtype=object) if regimes is None else np.asarray(regimes, dtype=object).reshape(-1)
    rows = []
    for lag in (lags if lags is not None else cfg.lags):
        src, tgt = lagged_pairs(groups, order, int(lag))
        if src.size < 4:
            continue
        tab = audit(v[src], h[tgt], discrete=discrete, regimes=reg[src], cfg=cfg, enabled_proposals=enabled_proposals)
        rows.append(tab.assign(lag=int(lag), pairs=int(src.size)))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


@dataclass(frozen=True)
class ModelGap:
    """A model judged against the ceiling of its regime (module docstring)."""

    error: float
    gap_min: float
    gap_max: float
    mi_model: float
    information_efficiency: float


def model_gap(predictions: np.ndarray, hidden: np.ndarray, ceiling: Ceiling, *, enabled_proposals: Collection[str] = (),
              correction: str = "miller_madow") -> ModelGap:
    """Error and information efficiency of a model's predictions against a `Ceiling` (module docstring)."""
    decisions.require_proposal(_GATE, enabled_proposals)
    p = np.asarray(predictions, dtype=object).reshape(-1)
    h = np.asarray(hidden, dtype=object).reshape(-1)
    if p.size != h.size:
        raise ValueError("predictions and hidden must have the same length")
    err = float(np.mean(p != h))
    mi_model = max(info.mutual_information_discrete(p, h, correction=correction), 0.0)
    eff = mi_model / ceiling.mi if ceiling.mi > 0 else float("nan")
    return ModelGap(error=err, gap_min=err - ceiling.bayes_error_upper, gap_max=err - ceiling.bayes_error_lower,
                    mi_model=mi_model, information_efficiency=float(eff))


def report(values: np.ndarray, hidden: np.ndarray, *, discrete: np.ndarray, regimes: np.ndarray | None = None,
           groups: np.ndarray | None = None, order: np.ndarray | None = None, predictions: np.ndarray | None = None,
           columns: list[str] | None = None, cfg: InfoAuditConfig | None = None,
           enabled_proposals: Collection[str] = ()) -> Report:
    """The audit as a `Report`: ceilings per regime, per lag (with groups and order), and a model's gap."""
    cfg = cfg or InfoAuditConfig()
    enabled = tuple(enabled_proposals) + tuple(cfg.enabled_proposals)
    decisions.require_proposal(_GATE, enabled)
    rep = Report(kind="info_audit", title="Information audit (P-15)", provenance={"config": to_dict(cfg),
                                                                                  "observables": columns or []})
    tab = audit(values, hidden, discrete=discrete, regimes=regimes, cfg=cfg, enabled_proposals=enabled)
    rep.add_table("ceilings", tab, title="Information and Bayes-error ceiling per regime",
                  description="mi in nats; bayes_error_lower/upper bracket R*; accuracy ceilings are 1 - R* bounds.")
    for r in tab.to_dict(orient="records"):
        rep.summary[f"{r['regime']}.mi"] = r["mi"]
        rep.summary[f"{r['regime']}.normalized_mi"] = r["normalized_mi"]
        rep.summary[f"{r['regime']}.accuracy_ceiling"] = [r["accuracy_ceiling_lower"], r["accuracy_ceiling_upper"]]
    if groups is not None and order is not None:
        lag_tab = lagged_audit(values, hidden, discrete=discrete, groups=groups, order=order, regimes=regimes, cfg=cfg,
                               enabled_proposals=enabled)
        rep.add_table("lagged", lag_tab, title="Predictive information I(O_t; S_t+k) per lag")
    if predictions is not None:
        whole = regime_ceiling(values, hidden, discrete=discrete, cfg=cfg, enabled_proposals=enabled)
        gap = model_gap(predictions, hidden, whole, enabled_proposals=enabled, correction=cfg.correction)
        rep.summary.update({f"model.{k}": v for k, v in gap.__dict__.items()})
    if any(r["unestimated_mass"] > 0 for r in tab.to_dict(orient="records")):
        rep.notes.append("Some observation patterns were too small for the k-NN estimator; their share is "
                         "'unestimated_mass' and the MI is a lower bound there.")
    rep.notes.append("Cover-Hart bounds are large-sample statements; the 1-NN error is a finite-sample estimate.")
    return rep


__all__ = [
    "Ceiling", "ModelGap", "audit", "binary_entropy", "cover_hart_bounds", "fano_lower_bound", "hellman_raviv_upper_bound",
    "lagged_audit", "lagged_pairs", "model_gap", "mutual_information", "nn_error_loo", "plugin_bayes_error",
    "regime_ceiling", "report",
]
