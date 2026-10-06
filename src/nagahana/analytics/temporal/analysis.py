"""Temporal reports: a regular series, and the event times of a corpus.

Series report
-------------
For one regularly sampled series: summary, ACF / PACF with bands and Ljung-Box, Welch density and the
periodogram with Fisher's g test, ADF and KPSS (read together: ADF rejecting and KPSS not rejecting
points to stationarity, the reverse to a unit root, both rejecting to fractional integration or
breaks), PELT and BOCPD change points (Poisson costs for count series), and the Hurst exponent by DFA
and by aggregated variance.

Corpus report
-------------
    updates      the series of state updates per `bin_seconds` bin over the whole corpus (zero bins are
                 bins without updates, true zeros), analysed as above, plus the malicious share per bin
                 (labels are read for auditing only)
    gaps         inter-arrival times of all updates: burstiness, memory, log-binned density, tail index
    entities     per initiator entity with at least `entity_min_events` updates (the `max_entities` most
                 active): gaps, burstiness, memory and the periodicity of its count series
    relations    per (initiator, responder) pair with at least `entity_min_events` updates: coefficient
                 of variation of the gaps, Fisher's g and peak ratio of the count series. Regular,
                 periodic relations are the beaconing candidates (AS-36).
Sources without event times (AS-307) carry no temporal information and are left out, with a note.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from nagahana.analytics import robust
from nagahana.analytics.config import TemporalConfig, to_dict
from nagahana.analytics.report import Figure, Report
from nagahana.analytics.temporal import changepoint as cpd
from nagahana.analytics.temporal import correlation as corr
from nagahana.analytics.temporal import events as ev
from nagahana.analytics.temporal import hurst as hu
from nagahana.analytics.temporal import spectral as spec
from nagahana.analytics.temporal import stationarity as stn

if TYPE_CHECKING:                                                        # the data model is needed by corpus inputs only
    from nagahana.analytics.corpus import Corpus


def _is_count(x: np.ndarray) -> bool:
    return bool((x >= 0).all() and np.allclose(x, np.round(x)))


def series_report(x: np.ndarray, cfg: TemporalConfig | None = None, *, name: str = "series",
                  bin_seconds: float | None = None, kind: str = "temporal_series") -> Report:
    """Temporal analysis of one regular series (module docstring)."""
    cfg = cfg or TemporalConfig()
    a = np.asarray(x, dtype=np.float64).reshape(-1)
    if not np.isfinite(a).all():
        raise ValueError("the series must be finite")
    rep = Report(kind=kind, title=f"Temporal analysis of {name}", provenance={"config": to_dict(cfg)})
    n = a.size
    rep.summary.update({"name": name, "length": n, "bin_seconds": bin_seconds if bin_seconds is not None else float("nan"),
                        **{f"value.{k}": v for k, v in robust.robust_summary(a).items() if k in ("mean", "std", "median",
                                                                                                 "mad", "q099", "q100")}})
    if n < 16 or a.std() == 0:
        rep.notes.append("The series is shorter than 16 points or constant; only its summary is reported.")
        return rep
    sd = corr.serial_dependence(a, nlags=min(cfg.acf_lags, n - 2))
    rep.add_table("serial_dependence", pd.DataFrame({"lag": sd.lags, "acf": sd.acf, "pacf": sd.pacf,
                                                     "bartlett_band": sd.bartlett_band, "ljung_box_q": sd.ljung_box_q,
                                                     "ljung_box_p": sd.ljung_box_p}),
                  title="ACF, PACF and Ljung-Box", description=f"White-noise band +-{sd.white_band:.4g}.")
    rep.add_figure(Figure(name="acf", kind="bar", data=rep.table("serial_dependence"), x="lag", y="acf", title="ACF"))
    rep.summary["ljung_box_p_at_max_lag"] = float(sd.ljung_box_p[-1])
    fs = 1.0 / bin_seconds if bin_seconds else 1.0
    pg = spec.periodogram(a, fs=fs, nperseg=cfg.welch_segment)
    rep.add_figure(Figure(name="welch", kind="line", data=pd.DataFrame({"frequency": pg.welch_freq, "density": pg.welch_density}),
                          x="frequency", y="density", title="Welch spectral density", log_y=True,
                          x_label="frequency (1/s)" if bin_seconds else "frequency (1/sample)"))
    rep.summary.update({"fisher_g": pg.g, "fisher_p": pg.p_value, "fisher_p_exact": pg.p_exact,
                        "periodic_at_alpha": bool(pg.p_value < cfg.periodicity_alpha), "peak_period": pg.peak_period,
                        "peak_ratio": pg.peak_ratio})
    try:
        ad = stn.adf(a, regression=cfg.adf_regression, autolag=cfg.adf_autolag, max_lag=cfg.adf_max_lag, seed=cfg.seed)
        rep.summary.update({"adf_statistic": ad.statistic, "adf_p": ad.p_value, "adf_lags": ad.lags,
                            "adf_critical_5pct": ad.critical[0.05], "adf_method": ad.method})
    except ValueError as exc:
        rep.notes.append(f"ADF not computed: {exc}")
    kp_lags: str | int = int(cfg.kpss_lags) if cfg.kpss_lags.isdigit() else cfg.kpss_lags
    kp = stn.kpss(a, regression=cfg.kpss_regression, lags=kp_lags)
    rep.summary.update({"kpss_statistic": kp.statistic, "kpss_p": kp.p_value, "kpss_p_bound": kp.p_bound, "kpss_lags": kp.lags})
    counts = _is_count(a)
    seg = cpd.pelt(a, cost="poisson" if counts else cfg.pelt_cost, penalty=cfg.pelt_penalty, min_size=cfg.pelt_min_size)
    online = cpd.bocpd(a, hazard=cfg.bocpd_hazard, model="poisson" if counts else cfg.bocpd_model, prune=cfg.bocpd_prune,
                       max_run=cfg.bocpd_max_run)
    rep.summary.update({"pelt_changepoints": seg.changepoints.tolist(), "pelt_cost": "poisson" if counts else cfg.pelt_cost,
                        "bocpd_changepoints": online.changepoints.tolist()})
    rep.add_figure(Figure(name="changepoints", kind="line",
                          data=pd.DataFrame({"index": np.arange(n), "value": a, "cp_probability": online.cp_probability,
                                             "map_run_length": online.map_run_length}),
                          x="index", y="value", title="Series with online change-point probability",
                          description="cp_probability = P(r_t = 0); PELT change points are in the summary."))
    fl = hu.dfa(a, order=cfg.dfa_order, min_scale=cfg.dfa_min_scale, max_scale_share=cfg.dfa_max_scale_share,
                scales=cfg.dfa_scales)
    h_av, h_av_se = hu.aggregated_variance(a)
    rep.summary.update({"dfa_alpha": fl.alpha, "dfa_alpha_se": fl.alpha_se, "hurst_aggregated_variance": h_av,
                        "hurst_aggregated_variance_se": h_av_se})
    if fl.scales.size:
        rep.add_figure(Figure(name="dfa", kind="scatter", data=pd.DataFrame({"scale": fl.scales, "fluctuation": fl.fluctuation}),
                              x="scale", y="fluctuation", title="DFA fluctuation function", log_x=True, log_y=True))
    return rep


def _relation_stats(times: np.ndarray, cfg: TemporalConfig) -> dict[str, float]:
    """Gaps, burstiness and count-series periodicity of one event sequence."""
    b = ev.burstiness(ev.inter_arrival(times))
    out = dict(b)
    _, counts = ev.count_series(times, bin_seconds=cfg.bin_seconds)
    if counts.size >= 16 and counts.std() > 0:
        pg = spec.periodogram(counts.astype(np.float64), fs=1.0 / cfg.bin_seconds, nperseg=cfg.welch_segment)
        out.update({"fisher_g": pg.g, "fisher_p": pg.p_value, "peak_period_s": pg.peak_period, "peak_ratio": pg.peak_ratio})
    else:
        out.update({"fisher_g": float("nan"), "fisher_p": float("nan"), "peak_period_s": float("nan"),
                    "peak_ratio": float("nan")})
    return out


def corpus_report(corpus: Corpus, cfg: TemporalConfig | None = None) -> Report:
    """Temporal report of a corpus (module docstring)."""
    cfg = cfg or TemporalConfig()
    rep = Report(kind="temporal", title="Temporal analytics", provenance={"config": to_dict(cfg)})
    timed = np.isfinite(corpus.time)
    if (~timed).any():
        rep.notes.append(f"{int((~timed).sum())} updates come from sources without event times (AS-307) and are left out.")
    if timed.sum() < 2:
        rep.notes.append("Fewer than two timed updates; nothing temporal to analyse.")
        return rep
    t = corpus.time[timed]
    mal = corpus.malicious[timed]
    starts, counts = ev.count_series(t, bin_seconds=cfg.bin_seconds)
    rep.merge(series_report(counts.astype(np.float64), cfg, name="updates per bin", bin_seconds=cfg.bin_seconds),
              prefix="updates")
    idx = np.floor((t - t.min()) / cfg.bin_seconds).astype(np.int64)
    known = np.isin(mal, (0.0, 1.0))
    num = np.bincount(idx[known], weights=mal[known], minlength=counts.size)
    den = np.bincount(idx[known], minlength=counts.size)
    rep.add_figure(Figure(name="malicious_share", kind="line",
                          data=pd.DataFrame({"bin_start": starts, "updates": counts,
                                             "malicious_share": np.where(den > 0, num / np.maximum(den, 1), np.nan)}),
                          x="bin_start", y="malicious_share", title="Malicious share per bin (labels, for auditing)"))
    gaps = ev.inter_arrival(t)
    b = ev.burstiness(gaps)
    rep.summary.update({f"gaps.{k}": v for k, v in b.items()})
    ti = robust.tail_index(gaps)
    rep.summary.update({"gaps.tail_alpha": ti.alpha_star, "gaps.tail_k_star": ti.k_star})
    lo, hi, dens = ev.log_histogram(gaps)
    rep.add_figure(Figure(name="gap_density", kind="histogram", data=pd.DataFrame({"gap_left_s": lo, "gap_right_s": hi,
                                                                                   "density": dens}),
                          x="gap_left_s", y="density", title="Inter-arrival density (log bins)", log_x=True, log_y=True))
    # Per initiator entity and per relation.
    ent = corpus.entities[timed]
    keys = np.array([f"{nw}|{k}|{key}" for nw, k, key in zip(corpus.entity_network, corpus.entity_kind,
                                                             corpus.entity_key, strict=True)], dtype=object)
    frame = pd.DataFrame({"init": ent[:, 0], "resp": ent[:, 1], "t": t, "mal": mal})
    frame = frame[frame["init"] >= 0]
    act = frame.groupby("init").size().sort_values(ascending=False)
    act = act[act >= cfg.entity_min_events].iloc[: cfg.max_entities]
    rows = []
    groups = frame.groupby("init")
    for e_id in act.index.tolist():
        g = groups.get_group(e_id)
        m = g["mal"].to_numpy()
        kn = np.isin(m, (0.0, 1.0))
        rows.append({"entity": keys[e_id], "kind": corpus.entity_kind[e_id], "events": len(g),
                     "malicious_share": float(m[kn].mean()) if kn.any() else float("nan"),
                     **_relation_stats(g["t"].to_numpy(), cfg)})
    rep.add_table("entities", pd.DataFrame(rows), title="Initiator entities: gaps, burstiness, periodicity",
                  description=f"Entities with at least {cfg.entity_min_events} updates, most active first.")
    pairs = frame[frame["resp"] >= 0].groupby(["init", "resp"])
    sizes = pairs.size()
    sizes = sizes[sizes >= cfg.entity_min_events].sort_values(ascending=False).iloc[: cfg.max_entities]
    rel_rows = []
    for (i, j) in sizes.index.tolist():
        g = pairs.get_group((i, j))
        m = g["mal"].to_numpy()
        kn = np.isin(m, (0.0, 1.0))
        rel_rows.append({"initiator": keys[i], "responder": keys[j], "events": len(g),
                         "malicious_share": float(m[kn].mean()) if kn.any() else float("nan"),
                         **_relation_stats(g["t"].to_numpy(), cfg)})
    rel = pd.DataFrame(rel_rows)
    if len(rel):
        rel = rel.sort_values(["fisher_p", "cv"], na_position="last").reset_index(drop=True)
    rep.add_table("relations", rel, title="Relations: regularity and periodicity (beaconing candidates)",
                  description="Sorted by Fisher's g p-value, then by the coefficient of variation of the gaps.")
    rep.summary["periodic_relations"] = int((rel["fisher_p"] < cfg.periodicity_alpha).sum()) if len(rel) else 0
    return rep


__all__ = ["corpus_report", "series_report"]
