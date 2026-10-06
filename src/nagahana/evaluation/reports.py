"""Results on disk: CSV and JSON of every row, figure data, and LaTeX tables in the layouts of the thesis.

Every number written here comes from the rows of a scored protocol run (scorer.ProtocolResult): a table
cell is looked up among the metric and comparison rows, and a cell without a row is printed as "n/a".
No value is copied from anywhere else, so a table can contain only measured values. The LaTeX fragments
reproduce the layouts of the thesis results chapter (column order, units, arrows, two decimals for
percentages, three for proper and skill scores, units on times) and are written to the output
directory of the run, never into the thesis sources.

Significance marks. A dagger marks a paired difference that remains significant after the configured
false-discovery-rate procedure, applied to the family of comparisons the table itself displays (the
thesis captions define the dagger per table). The p-values come from the paired tests of the scorer.

Main tables with intervals. Besides the thesis layouts, which print point estimates, every main table is
also written with each cell as "value [low, high]" (the bootstrap interval of the scorer), and the CSV
files carry every interval, seed count and unit and event count.

JSON. NaN is written as null and infinities as the strings "inf" and "-inf", so the files are standard
JSON.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nagahana.evaluation import multidataset as mds
from nagahana.evaluation.components import COMPONENT_METRICS
from nagahana.evaluation.config import EvaluationConfig
from nagahana.evaluation.protocols import MATRIX_SOURCES, PROTOCOLS
from nagahana.evaluation.scorer import ProtocolResult
from nagahana.evaluation.significance import adjust_pvalues

DAGGER = r"\textsuperscript{$\dagger$}"
NOVELTY_ORDER = ("known", "novel", "known_unseen_network", "novel_unseen_network", "unmarked", "")
NOVELTY_LABEL = {"known": "Known", "novel": "Novel", "known_unseen_network": "Known, unseen network",
                 "novel_unseen_network": "Novel, unseen network", "unmarked": "Unmarked", "": "All"}
UP, DOWN = r"\,$\uparrow$", r"\,$\downarrow$"


def _plain(v: Any) -> Any:
    # JSON-safe value: NaN -> None, infinities -> strings, numpy scalars -> Python.
    if isinstance(v, float | np.floating):
        f = float(v)
        if math.isnan(f):
            return None
        if math.isinf(f):
            return "inf" if f > 0 else "-inf"
        return f
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, dict):
        return {str(k): _plain(x) for k, x in v.items()}
    if isinstance(v, list | tuple):
        return [_plain(x) for x in v]
    return v


def frame_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Rows of a frame as JSON-safe dictionaries."""
    return [{str(k): _plain(v) for k, v in row.items()} for row in frame.to_dict(orient="records")]


def to_json(obj: Any) -> str:
    """Standard JSON text (NaN as null, infinities as strings)."""
    return json.dumps(_plain(obj), indent=1, sort_keys=False, allow_nan=False)


def _fmt(v: Any, decimals: int, *, percent: bool = False, signed: bool = False) -> str:
    # Number in the thesis style: n/a for undefined, $-$ for negatives, optional explicit sign.
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "n/a"
    if math.isnan(f):
        return "n/a"
    if math.isinf(f):
        return r"$\infty$" if f > 0 else r"$-\infty$"
    if percent:
        f *= 100.0
    text = f"{abs(f):.{decimals}f}"
    if float(text) == 0.0:
        return text
    if f < 0:
        return "$-$" + text
    return ("$+$" + text) if signed else text


def _ci(v: Any, lo: Any, hi: Any, decimals: int, *, percent: bool = False) -> str:
    # "value [low, high]" (n/a parts where undefined).
    return f"{_fmt(v, decimals, percent=percent)} [{_fmt(lo, decimals, percent=percent)}, {_fmt(hi, decimals, percent=percent)}]"


def _rows(frame: pd.DataFrame, **filters: Any) -> pd.DataFrame:
    out = frame
    for k, v in filters.items():
        if v is None:
            continue
        out = out[out[k].astype(str) == str(v)]
    return out


def _one(frame: pd.DataFrame, **filters: Any) -> pd.Series | None:
    # The single row matching the filters; with several, the one whose novelty comes first in NOVELTY_ORDER.
    sel = _rows(frame, **filters)
    if sel.empty:
        return None
    if len(sel) > 1 and "novelty" in sel.columns:
        rank = sel["novelty"].map(lambda n: NOVELTY_ORDER.index(n) if n in NOVELTY_ORDER else len(NOVELTY_ORDER))
        sel = sel.assign(_r=rank.to_numpy()).sort_values("_r", kind="stable")
    return sel.iloc[0]


def _value(frame: pd.DataFrame, **filters: Any) -> tuple[float, float, float]:
    row = _one(frame, **filters)
    if row is None:
        return math.nan, math.nan, math.nan
    return float(row["value"]), float(row["ci_low"]), float(row["ci_high"])


def _latex_table(caption: str, label: str, colspec: str, header: Sequence[str], body: Sequence[str], *, wide: bool,
                 source: str, tabcolsep: str = "") -> str:
    # One table float in the style of the thesis (booktabs, tabularx).
    env = "table*" if wide else "table"
    width = r"\textwidth" if wide else r"\columnwidth"
    lines = [f"% Generated by nagahana.evaluation.reports from scored runs ({source}).",
             "% Do not edit by hand: rerun python -m nagahana evaluate run.",
             rf"\begin{{{env}}}[tp]", r"  \centering", rf"  \caption{{{caption}}}", rf"  \label{{{label}}}"]
    if tabcolsep:
        lines.append(rf"  \setlength{{\tabcolsep}}{{{tabcolsep}}}")
    lines.append(rf"  \begin{{tabularx}}{{{width}}}{{{colspec}}}")
    lines.append(r"    \toprule")
    lines += [f"    {h}" for h in header]
    lines.append(r"    \midrule")
    lines += [f"    {b}" for b in body]
    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabularx}")
    lines.append(rf"\end{{{env}}}")
    return "\n".join(lines) + "\n"


def _source(result: ProtocolResult) -> str:
    reg = result.registration
    return f"protocol {result.protocol}, registration {reg.get('id', '?')} {str(reg.get('document_sha256', ''))[:12]}"


def _ordered_models(frame: pd.DataFrame, cfg: EvaluationConfig, *, with_refs: bool = False) -> list[str]:
    # Rows of the main tables: references, the LR family, published models (sorted), NagaHana last.
    present = list(dict.fromkeys(frame["model"].astype(str)))
    refs = [m for m in ("persistence", "climatology") if m in present] if with_refs else []
    lr = [m for m in cfg.models.lr_family if m in present]
    prim = [cfg.models.primary] if cfg.models.primary in present else []
    others = sorted(m for m in present if m not in refs + lr + prim and "[" not in m and m not in ("persistence", "climatology"))
    return refs + lr + others + prim


def _family_daggers(pvals: Sequence[float], cfg: EvaluationConfig) -> list[bool]:
    # Significance within the family of the comparisons a table displays.
    p = np.asarray(pvals, dtype=np.float64)
    if p.size == 0:
        return []
    adj = adjust_pvalues(p, cfg.comparisons.fdr_method)
    return [bool(a <= cfg.comparisons.fdr_level) for a in adj]


def _groups(frame: pd.DataFrame, cfg: EvaluationConfig) -> list[str]:
    present = set(frame["group"].astype(str))
    order = [g.id for g in cfg.groups if g.id in present]
    return order + sorted(present - set(order) - {"all"})


def res_detection(result: ProtocolResult, cfg: EvaluationConfig) -> str:
    """Thesis Table res-detection: detection per dataset group, known and novel apart (protocol P1)."""
    m = _rows(result.metrics, task="detection", variant="")
    m = m[m["role"] != "ablation"]
    models = _ordered_models(m, cfg)
    cols = ["precision", "recall", "f1", "fpr", "fnr", "detection_error_rate", "auroc", "auprc"]
    comps = _rows(result.comparisons, task="detection", metric="f1", family="baselines")
    lr = cfg.models.lr_family[0] if cfg.models.lr_family else ""
    marks: dict[tuple[str, str], bool] = {}
    keys, pvals = [], []
    for g in _groups(m, cfg):
        for nov in ("known", "novel"):
            row = _one(comps, group=g, novelty=nov, model_a=cfg.models.primary, model_b=lr)
            if row is not None and np.isfinite(float(row["p_value"])):
                keys.append((g, nov))
                pvals.append(float(row["p_value"]))
    for key, flag in zip(keys, _family_daggers(pvals, cfg), strict=True):
        marks[key] = flag
    body: list[str] = []
    for g in _groups(m, cfg):
        gm = _rows(m, group=g)
        if gm.empty:
            continue
        rates = []
        for nov in ("known", "novel"):
            v, _, _ = _value(gm, model=cfg.models.primary if cfg.models.primary in models else models[0], novelty=nov,
                             metric="base_rate")
            rates.append(f"{_fmt(v, 1, percent=True)}\\% {nov}")
        if body:
            body.append(r"\midrule")
        body.append(rf"\multicolumn{{10}}{{@{{}}l}}{{\textit{{{cfg.group(g).label}}} (base rate {', '.join(rates)})}} \\")
        for nov in ("known", "novel"):
            for model in models:
                cells = []
                for c in cols:
                    v, _, _ = _value(gm, model=model, novelty=nov, metric=c)
                    text = _fmt(v, 2, percent=True)
                    if c == "f1" and model == cfg.models.primary and marks.get((g, nov)):
                        text += DAGGER
                    cells.append(text)
                if all(c == "n/a" for c in cells):
                    continue
                body.append(f"{NOVELTY_LABEL[nov]} & {cfg.display(model)} & " + " & ".join(cells) + r" \\")
    header = [r"Split & Method & Prec." + UP + r" & Rec." + UP + r" &",
              r"  F1" + UP + r" & FPR" + DOWN + r" & FNR" + DOWN + r" &",
              r"  DE rate" + DOWN + r" & AUROC" + UP + r" & AUPRC" + UP + r" \\"]
    caption = (r"Detection on real zero-shot data (protocol P1), known and novel attack families reported separately. "
               r"Logistic regression (LR) is trained on the same features, normalisation and splits as \NagaHana{}. "
               r"$\dagger$: paired difference from LR significant after false-discovery-rate control at level "
               f"{cfg.comparisons.fdr_level:g} over the comparisons of this table (McNemar's test).")
    return _latex_table(caption, "tab:res-detection", r"@{}l l *{8}{>{\centering\arraybackslash}X}@{}", header, body,
                        wide=True, source=_source(result))


def _horizon_label(h: str, k_max: int) -> str:
    k = int(h)
    if k == 1:
        return "$k=1$"
    if k == k_max:
        return "$k=K$"
    if k == math.ceil(k_max / 2):
        return r"$k=\lceil K/2\rceil$"
    return f"$k={k}$"


def res_forecast(result: ProtocolResult, cfg: EvaluationConfig, novelty: str) -> str:
    """Thesis Table res-forecast for one novelty group: proper scores, ECE, skill and CRPS by horizon."""
    m = _rows(result.metrics, task="forecast", group="all" if not _rows(result.metrics, task="forecast",
                                                                       group="all").empty else None,
              novelty=novelty, variant="")
    m = m[m["role"] != "ablation"]
    if m.empty:
        return ""
    horizons = sorted({str(h) for h in m["horizon"]}, key=int)
    k_max = max(int(h) for h in horizons)
    models = _ordered_models(m, cfg, with_refs=True)
    body: list[str] = []
    for h in horizons:
        if body:
            body.append(r"\midrule")
        for i, model in enumerate(models):
            sel = _rows(m, horizon=h, model=model)
            if sel.empty:
                continue
            vals = {c: _value(sel, metric=c)[0] for c in ("brier", "log_score", "ece", "bss_persistence",
                                                         "bss_climatology", "crps", "crpss_climatology")}
            bssp = "Ref." if model == "persistence" else _fmt(vals["bss_persistence"], 3)
            bssc = "Ref." if model == "climatology" else _fmt(vals["bss_climatology"], 3)
            crpss = "Ref." if model == "climatology" else _fmt(vals["crpss_climatology"], 3)
            name = cfg.display(model) + (" (static forecaster)" if model in cfg.models.lr_family else "")
            label = _horizon_label(h, k_max) if i == 0 or not body or body[-1] == r"\midrule" else ""
            body.append(f"{label} & {name} & {_fmt(vals['brier'], 3)} & {_fmt(vals['log_score'], 3)} & "
                        f"{_fmt(vals['ece'], 3)} & {bssp} & {bssc} & {_fmt(vals['crps'], 3)} & {crpss}" + r" \\")
    header = [r"Horizon & Method & Brier" + DOWN + r" & Log score" + DOWN + r" &",
              r"  ECE" + DOWN + r" & BSS vs pers." + UP + r" & BSS vs clim." + UP + r" &",
              r"  CRPS" + DOWN + r" & CRPSS vs clim." + UP + r" \\"]
    caption = (rf"Forecast quality of $\nhPinf(k)$ on real zero-shot data (protocol P1), {NOVELTY_LABEL[novelty].lower()} "
               r"families, pooled over datasets with multi-stage attacks, at horizons $k=1$, $\lceil K/2\rceil$ and $K$.")
    return _latex_table(caption, f"tab:res-forecast-{novelty or 'all'}", r"@{}l l *{7}{>{\centering\arraybackslash}X}@{}",
                        header, body, wide=True, source=_source(result))


def res_state(result: ProtocolResult, cfg: EvaluationConfig, novelty: str) -> str:
    """Thesis Table res-state: MAE, RMSE and MSE skill against persistence by horizon."""
    base = _rows(result.metrics, task="state_forecast", variant="", novelty=novelty)
    m = _rows(base, group="all") if not _rows(base, group="all").empty else base
    m = m[m["role"] != "ablation"]
    if m.empty:
        return ""
    horizons = sorted({str(h) for h in m["horizon"]}, key=int)
    k_max = max(int(h) for h in horizons)
    models = _ordered_models(m, cfg, with_refs=True)
    body: list[str] = []
    for h in horizons:
        if body:
            body.append(r"\midrule")
        first = True
        for model in models:
            sel = _rows(m, horizon=h, model=model)
            if sel.empty:
                continue
            mae, rmse, ss = (_value(sel, metric=c)[0] for c in ("mae", "rmse", "msess_persistence"))
            name = "Linear (ridge)" if model in cfg.models.lr_family else cfg.display(model)
            label = _horizon_label(h, k_max) if first else ""
            first = False
            body.append(f"{label} & {name} & {_fmt(mae, 3)} & {_fmt(rmse, 3)} & "
                        f"{'Ref.' if model == 'persistence' else _fmt(ss, 3)}" + r" \\")
    header = [r"Horizon & Method & MAE" + DOWN + r" & RMSE" + DOWN + r" & MSESS" + UP + r" \\"]
    caption = (rf"Next-state forecasts on real zero-shot data (protocol P1), {NOVELTY_LABEL[novelty].lower()} families: "
               r"error of the forecast traffic features of each window, standardised on the training split, at horizons "
               r"$k=1$, $\lceil K/2\rceil$ and $K$. MSESS is the mean-squared-error skill score against persistence.")
    return _latex_table(caption, f"tab:res-state-{novelty or 'all'}", r"@{}l l *{3}{>{\centering\arraybackslash}X}@{}",
                        header, body, wide=False, source=_source(result), tabcolsep="3pt")


def _time(v: float, unit: str) -> str:
    # Lead time in seconds printed in the dataset's unit.
    scale = {"s": 1.0, "min": 60.0, "h": 3600.0}[unit]
    if not np.isfinite(v):
        return "n/a" if np.isnan(v) else ("never" if v < 0 else r"$\infty$")
    return _fmt(v / scale, 1) + rf"\,{unit}"


def res_timeliness(results: Mapping[str, ProtocolResult], cfg: EvaluationConfig) -> str:
    """Thesis Table res-timeliness: lead times and time-to-event quality per dataset group (P1, then P7)."""
    body: list[str] = []
    used: list[str] = []
    for pid in ("P1", "P7"):
        if pid not in results:
            continue
        res = results[pid]
        t = _rows(res.metrics, task="timeliness", variant="", novelty="known")
        s = _rows(res.metrics, task="time_to_event", variant="", novelty="known")
        for g in _groups(pd.concat([t, s]) if len(t) or len(s) else t, cfg):
            if g in used or g == "all":
                continue
            unit = next((d.time_unit for d in cfg.datasets if d.group == g), "min")
            rows = []
            for model in [*cfg.models.lr_family, cfg.models.primary]:
                med = _value(t, group=g, model=model, metric="median_lead_time")[0]
                frac = _value(t, group=g, model=model, metric="alerted_before_completion")[0]
                fix = _value(t, group=g, model=model, metric="lead_time_at_fpr")[0]
                c = _value(s, group=g, model=model, metric="c_index")[0]
                auc = _value(s, group=g, model=model, metric="td_auc")[0]
                ibs = _value(s, group=g, model=model, metric="ibs")[0]
                if all(np.isnan(x) for x in (med, frac, fix, c, auc, ibs)):
                    continue
                name = cfg.display(model) + (" (static forecaster)" if model in cfg.models.lr_family else "")
                rows.append(f"{{}} & {name} & {_time(med, unit)} & {_fmt(frac, 1, percent=True)}\\% & {_time(fix, unit)} & "
                            f"{_fmt(c, 3)} & {_fmt(auc, 3)} & {_fmt(ibs, 3)}" + r" \\")
            if not rows:
                continue
            used.append(g)
            if body:
                body.append(r"\midrule")
            rows[0] = rows[0].replace("{}", cfg.group(g).label, 1)
            body += [r.replace("{} & ", " & ", 1) if i else r for i, r in enumerate(rows)]
    if not body:
        return ""
    header = [r"Dataset & Method & Median lead time" + UP + r" & Alerted before completion" + UP + r" &",
              r"  Lead time at FPR $\alpha$" + UP + r" & C-index" + UP + r" &",
              r"  Time-dep.\ AUC" + UP + r" & IBS" + DOWN + r" \\"]
    caption = ("Lead time at a fixed false-positive rate and time-to-event quality, per dataset with annotated "
               "multi-stage attacks (protocols P1 and P7), known families.")
    src = "; ".join(_source(results[p]) for p in ("P1", "P7") if p in results)
    return _latex_table(caption, "tab:res-timeliness", r"@{}l l *{6}{>{\centering\arraybackslash}X}@{}", header, body,
                        wide=True, source=src)


def res_stages(results: Mapping[str, ProtocolResult], cfg: EvaluationConfig) -> str:
    """Thesis Table res-stages: stage prediction and attack-path quality (P1; ordinal safety from P6 when present)."""
    if "P1" not in results:
        return ""
    res = results["P1"]
    st = _rows(res.metrics, task="stage", variant="", group="all")
    if st.empty:
        st = _rows(res.metrics, task="stage", variant="")
    pa = _rows(res.metrics, task="paths", variant="", group="all")
    if pa.empty:
        pa = _rows(res.metrics, task="paths", variant="")
    p6 = _rows(results["P6"].metrics, task="paths") if "P6" in results else pd.DataFrame(columns=res.metrics.columns)
    body: list[str] = []
    for model in [*cfg.models.lr_family, *[m for m in _ordered_models(st, cfg) if m not in cfg.models.lr_family]]:
        lines = []
        for nov in ("known", "novel"):
            top1, top3, mf1 = (_value(st, model=model, novelty=nov, metric=c)[0] for c in ("top1", "top3", "macro_f1"))
            pp, pr, ed, tau = (_value(pa, model=model, novelty=nov, metric=c)[0]
                               for c in ("precision_at_n", "recall_at_n", "median_best_distance", "kendall_tau"))
            os_ = _value(p6, model=model, novelty=nov, metric="ordinal_safety")[0] if len(p6) else math.nan
            if np.isnan(os_):
                os_ = _value(pa, model=model, novelty=nov, metric="ordinal_safety")[0]
            if all(np.isnan(x) for x in (top1, top3, mf1, pp)):
                continue
            name = cfg.display(model) + (" (multinomial)" if model in cfg.models.lr_family else "")
            lines.append(f"{name if not lines else ''} & {NOVELTY_LABEL[nov]} & {_fmt(top1, 2, percent=True)} & "
                         f"{_fmt(top3, 2, percent=True)} & {_fmt(mf1, 2, percent=True)} & {_fmt(pp, 2, percent=True)} & "
                         f"{_fmt(pr, 2, percent=True)} & {_fmt(ed, 2)} & {_fmt(tau, 2)} & {_fmt(os_, 3)}" + r" \\")
        if lines:
            if body:
                body.append(r"\midrule")
            body += lines
    if not body:
        return ""
    header = [r"Method & Split & Top-1" + UP + r" & Top-3" + UP + r" & Macro-F1" + UP + r" &",
              r"  Path prec.@$N$" + UP + r" & Path rec.@$N$" + UP + r" &",
              r"  Median edit dist." + DOWN + r" & Kendall $\tau$" + UP + r" &",
              r"  Ordinal safety" + UP + r" \\"]
    caption = r"ATT\&CK stage prediction and attack-path quality (protocols P1 and P6)."
    return _latex_table(caption, "tab:res-stages", r"@{}l l *{8}{>{\centering\arraybackslash}X}@{}", header, body,
                        wide=True, source=_source(res), tabcolsep="4pt")


def res_transfer(result: ProtocolResult, cfg: EvaluationConfig, metric: str) -> str:
    """Thesis Tables res-transfer (AUPRC) and res-transfer-f1 (F1): train dataset x test dataset (P2)."""
    m = _rows(result.metrics, task="detection", metric=metric)
    m = m[(m["group"] != "all") & (m["role"] != "ablation")]
    if m.empty:
        return ""
    trains = sorted({v.split("=", 1)[1] for v in m["variant"] if v.startswith("train=")},
                    key=lambda d: [g.id for g in cfg.groups].index(cfg.dataset(d).group)
                    if cfg.dataset(d).group in [g.id for g in cfg.groups] else 99)
    tests = _groups(m, cfg)
    models = [cfg.models.primary, *cfg.models.lr_family]
    body: list[str] = []
    for tr in trains:
        if body:
            body.append(r"\midrule")
        for i, model in enumerate(models):
            cells = [_fmt(_value(m, variant=f"train={tr}", model=model, group=te)[0], 2, percent=True) for te in tests]
            label = cfg.group(cfg.dataset(tr).group).short if i == 0 else ""
            body.append(f"{label} & {cfg.display(model)} & " + " & ".join(cells) + r" \\")
    header = [r"Train $\backslash$ Test & Method & " + " & ".join(rf"\mbox{{{cfg.group(t).short}}}" for t in tests) + r" \\"]
    name = "AUPRC" if metric == "auprc" else "F1 score"
    label = "tab:res-transfer" if metric == "auprc" else "tab:res-transfer-f1"
    caption = rf"Cross-dataset {name} ($\uparrow$), \NagaHana{{}} / LR (protocol P2)."
    return _latex_table(caption, label, rf"@{{}}l l *{{{len(tests)}}}{{>{{\centering\arraybackslash}}X}}@{{}}", header, body,
                        wide=False, source=_source(result), tabcolsep="2pt")


def res_lono(result: ProtocolResult, cfg: EvaluationConfig, *, site: bool) -> str:
    """Thesis Tables res-lono (F1 and AUPRC) and res-sitecal (F1 and balanced accuracy), protocol P3."""
    m = _rows(result.metrics, task="detection", group="held_out")
    m = m[m["variant"].str.endswith(";site") == site]
    if m.empty:
        return ""
    second = "balanced_accuracy" if site else "auprc"
    held = sorted({v.split("=", 1)[1].split(";")[0] for v in m["variant"]})
    body = []
    for h in held:
        variant = f"held_out={h}" + (";site" if site else "")
        cells = [_fmt(_value(m, variant=variant, model=model, metric=metric)[0], 2, percent=True)
                 for metric in ("f1", second) for model in (cfg.models.primary, *cfg.models.lr_family[:1])]
        body.append(f"{h} & " + " & ".join(cells) + r" \\")
    sec_name = "Balanced accuracy" if site else "AUPRC"
    header = [rf" & \multicolumn{{2}}{{c}}{{F1}} & \multicolumn{{2}}{{c}}{{{sec_name}}} \\",
              r"\cmidrule(lr){2-3}\cmidrule(l){4-5}",
              r"Held out & \NagaHana{} & LR & \NagaHana{} & LR \\"]
    if site:
        caption = (rf"Leave-one-network-out after site calibration, F1 score and balanced accuracy ($\uparrow$), \NagaHana{{}} / "
                   rf"LR (protocol P3). The held-out network's first {cfg.protocols.p3_site_benign_hours:g}\,h of traffic, without "
                   rf"attack labels, and {cfg.protocols.p3_site_analyst_alerts} analyst-confirmed alerts calibrate both models; "
                   "the rest of its timeline is the test set.")
        label = "tab:res-sitecal"
    else:
        caption = (r"Leave-one-network-out F1 score and AUPRC ($\uparrow$), \NagaHana{} / LR (protocol P3). Each row is the "
                   "network held out of training.")
        label = "tab:res-lono"
    return _latex_table(caption, label, r"@{}l *{4}{>{\centering\arraybackslash}X}@{}", header, body, wide=False,
                        source=_source(result), tabcolsep="3pt")


def res_robustness(results: Mapping[str, ProtocolResult], cfg: EvaluationConfig) -> str:
    """Thesis Table res-robustness: observability regimes (P4) and telemetry corruption (P5)."""
    pr = cfg.protocols
    body: list[str] = []
    for pid, key, names in (("P4", "regime", pr.p4_regimes), ("P5", "corruption", pr.p5_corruptions)):
        if pid not in results:
            continue
        res = results[pid]
        m = _rows(res.metrics, group=cfg.dataset(pr.p4_dataset).group)
        if body:
            body.append(r"\midrule")
        for name in names:
            variant = f"{key}={name}"
            f1n = _value(m, task="detection", variant=variant, model=cfg.models.primary, novelty=pr.p4_novelty, metric="f1")[0]
            f1l = _value(m, task="detection", variant=variant, model=cfg.models.lr_family[0] if cfg.models.lr_family else "",
                         novelty=pr.p4_novelty, metric="f1")[0]
            ap = _value(m, task="detection", variant=variant, model=cfg.models.primary, novelty=pr.p4_novelty, metric="auprc")[0]
            hz = _rows(res.metrics, task="forecast", variant=variant, model=cfg.models.primary)
            k = str(max((int(h) for h in hz["horizon"]), default=0)) if len(hz) else ""
            bs = _value(hz, horizon=k, group=cfg.dataset(pr.p4_dataset).group, novelty=pr.p4_novelty, metric="brier")[0]
            trust = _value(_rows(res.metrics, task="component", variant=variant, model=cfg.models.primary),
                           metric="taaft.trust_auroc")[0]
            sig = _one(_rows(res.metrics, task="degradation", variant=variant), group=cfg.dataset(pr.p4_dataset).group,
                       novelty=pr.p4_novelty, metric="degradation_signalled")
            flag = "n/a" if sig is None or np.isnan(float(sig["value"])) else ("yes" if float(sig["value"]) > 0.5 else "no")
            if name in (pr.p4_reference_regime, "none"):
                flag = "n/a"
            if all(np.isnan(x) for x in (f1n, f1l, ap, bs)):
                continue
            body.append(f"{name.replace('_', ' ').capitalize()} & {_fmt(f1n, 2, percent=True)} & {_fmt(f1l, 2, percent=True)} & "
                        f"{_fmt(ap, 2, percent=True)} & {_fmt(bs, 3)} & {_fmt(trust, 3)} & {flag}" + r" \\")
    if not body:
        return ""
    header = [r"Condition & F1 \NagaHana{}" + UP + r" & F1 LR" + UP + r" & AUPRC" + UP + r" &",
              r"  Brier" + DOWN + r" & Trust AUROC" + UP + r" & Degradation signalled \\"]
    caption = "Robustness to observability regimes and telemetry corruption (protocols P4 and P5)."
    src = "; ".join(_source(results[p]) for p in ("P4", "P5") if p in results)
    return _latex_table(caption, "tab:res-robustness", r"@{}>{\raggedright\arraybackslash}p{0.2\textwidth} "
                        r"*{6}{>{\centering\arraybackslash}X}@{}", header, body, wide=True, source=src)


def res_components(results: Mapping[str, ProtocolResult], cfg: EvaluationConfig) -> str:
    """Thesis Table res-components: component diagnostics (P1, P5, P6)."""
    rows = pd.concat([_rows(results[p].metrics, task="component", model=cfg.models.primary)
                      for p in ("P1", "P5", "P6") if p in results] or [pd.DataFrame()], ignore_index=True)
    if rows.empty:
        return ""
    body, last = [], ""
    for spec in COMPONENT_METRICS:
        sel = rows[rows["metric"].astype(str).str.startswith(spec.name)]
        if sel.empty:
            continue
        vals = []
        for _, r in sel.iterrows():
            vals.append(_fmt(float(r["value"]), 3))
        comp_label = {"CVG-AE": r"\CVGAE{}", "TSTCT": r"\TSTCT{}", "TAAFT": r"\TAAFT{}"}.get(spec.component, spec.component)
        direction = {"higher": r"$\uparrow$", "lower": r"$\downarrow$"}.get(spec.direction, "--")
        required = spec.required or "--"
        measured = " / ".join(dict.fromkeys(vals))
        if spec.required == "pass":
            measured = "pass" if all(float(r["value"]) > 0.5 for _, r in sel.iterrows()) else "fail"
        body.append(f"{comp_label if comp_label != last else ''} & {spec.label} & {direction} & {required} & {measured}" + r" \\")
        last = comp_label
    header = [r"Component & Metric & Direction & Required & Measured \\"]
    caption = ("Component diagnostics (protocols P1, P5 and P6). ``Required'' gives the value that a design invariant "
               "demands; other rows have no required value.")
    src = "; ".join(_source(results[p]) for p in ("P1", "P5", "P6") if p in results)
    return _latex_table(caption, "tab:res-components", r"@{}>{\raggedright\arraybackslash}p{0.13\textwidth} X "
                        r">{\centering\arraybackslash}p{0.09\textwidth} >{\centering\arraybackslash}p{0.09\textwidth} "
                        r">{\centering\arraybackslash}p{0.13\textwidth}@{}", header, body, wide=True, source=src)


def ablation_cells(results: Mapping[str, ProtocolResult], cfg: EvaluationConfig) -> pd.DataFrame:
    """Every (ablation, table column) difference with its interval and paired p-value, adjusted over the table."""
    frames = []
    for pid, res in results.items():
        c = _rows(res.comparisons, family="ablations")
        if len(c):
            frames.append(c.assign(protocol=pid))
    if not frames:
        return pd.DataFrame()
    comps = pd.concat(frames, ignore_index=True)
    rows = []
    for _, r in comps[["model_a"]].drop_duplicates().iterrows():
        key = str(r["model_a"])
        aid = key[key.index("[") + 1: -1] if "[" in key else key
        info = cfg.ablation(aid)
        for col in cfg.ablation_columns:
            variant = f"base={col.base}" + (f";{col.variant}" if col.variant else "")
            sel = _rows(comps, model_a=key, task=col.task, metric=col.metric, group=col.group or None,
                        novelty=col.novelty or None)
            sel = sel[(sel["variant"] == variant) | ((sel["protocol"] == col.base) & (sel["variant"] == col.variant))]
            if col.horizon:
                hs = sel["horizon"].astype(str)
                target = str(max((int(h) for h in hs if h.isdigit()), default=0)) if col.horizon == "K" else col.horizon
                sel = sel[hs == target]
            row = sel.iloc[0] if len(sel) else None
            rows.append({"ablation": aid, "tier": info.tier if info else "", "thesis_id": info.thesis_id if info else "",
                         "column": col.id, "difference": float(row["difference"]) * col.scale if row is not None else math.nan,
                         "ci_low": float(row["ci_low"]) * col.scale if row is not None else math.nan,
                         "ci_high": float(row["ci_high"]) * col.scale if row is not None else math.nan,
                         "test": str(row["test"]) if row is not None else "",
                         "p_value": float(row["p_value"]) if row is not None else math.nan,
                         "ci_method": str(row["ci_method"]) if row is not None else ""})
    out = pd.DataFrame(rows)
    out["p_adjusted"] = adjust_pvalues(out["p_value"].to_numpy(dtype=np.float64), cfg.comparisons.fdr_method)
    out["significant"] = out["p_adjusted"].to_numpy(dtype=np.float64) <= cfg.comparisons.fdr_level
    return out


def res_ablations(results: Mapping[str, ProtocolResult], cfg: EvaluationConfig, *, with_intervals: bool) -> str:
    """Ablation table: paired change (ablation - full model) per column, by tier, with daggers over the table family."""
    cells = ablation_cells(results, cfg)
    if cells.empty:
        return ""
    cols = cfg.ablation_columns
    tier_name = {"inference": "Inference-time (no retraining)", "single_stage": "Single-stage retrains",
                 "full": "Full retrains"}
    order = {a.id: i for i, a in enumerate(cfg.ablations)}
    body: list[str] = []
    for tier in ("inference", "single_stage", "full"):
        ids = sorted(set(cells.loc[cells["tier"] == tier, "ablation"]), key=lambda a: order.get(a, 999))
        if not ids:
            continue
        if body:
            body.append(r"\midrule")
        body.append(rf"\multicolumn{{{len(cols) + 1}}}{{@{{}}l}}{{\textit{{{tier_name[tier]}}}}} \\")
        for aid in ids:
            sub = cells[cells["ablation"] == aid]
            texts = []
            for col in cols:
                r = sub[sub["column"] == col.id]
                if r.empty:
                    texts.append("n/a")
                    continue
                rr = r.iloc[0]
                t = _fmt(rr["difference"], col.decimals, signed=True)
                if with_intervals:
                    t += f" [{_fmt(rr['ci_low'], col.decimals)}, {_fmt(rr['ci_high'], col.decimals)}]"
                if bool(rr["significant"]):
                    t += DAGGER
                texts.append(t)
            info = cfg.ablation(aid)
            label = aid + (f" ({info.thesis_id})" if info and info.thesis_id else "")
            body.append(f"{label} & " + " & ".join(texts) + r" \\")
    spans = [c for c in cols]
    header = [" & " + " & ".join(c.id for c in spans) + r" \\", " & " + " & ".join(c.label for c in spans) + r" \\"]
    caption = (r"Ablations: paired change relative to the full model (ablated minus full) with "
               f"{'BCa intervals and ' if with_intervals else ''}" r"$\dagger$ where significant after "
               f"false-discovery-rate control at level {cfg.comparisons.fdr_level:g} over the cells of this table. "
               "Columns: " + "; ".join(f"{c.id}: {c.task} {c.metric} on {c.group or 'any group'}"
                                     f"{', ' + c.novelty if c.novelty else ''}{', ' + c.variant if c.variant else ''}"
                                     f" (protocol {c.base})" for c in cols) + ".")
    label = "tab:res-ablations-ci" if with_intervals else "tab:res-ablations"
    src = "; ".join(_source(r) for r in results.values())
    return _latex_table(caption, label, r"@{}l *{" + str(len(cols)) + r"}{>{\centering\arraybackslash}X}@{}", header,
                        body, wide=with_intervals, source=src, tabcolsep="2pt")


def res_operations(result: ProtocolResult, cfg: EvaluationConfig) -> str:
    """Thesis Table res-operations: latency, sustained rate and memory growth per telemetry level (P8)."""
    m = _rows(result.metrics, task="operations", model=cfg.models.primary)
    if m.empty:
        return ""
    body = []
    for g in dict.fromkeys(m["group"].astype(str)):
        for h in dict.fromkeys(_rows(m, group=g)["horizon"].astype(str)):
            sel = _rows(m, group=g, horizon=h)
            p50, p99, rate, mem = (_value(sel, metric=c)[0] for c in ("latency_p50_ms", "latency_p99_ms", "throughput_per_s",
                                                                      "memory_gib_per_day"))
            label = g.replace("_", " ").capitalize() + (f" ({h.replace('rate=', '')} s$^{{-1}}$ offered)" if h else "")
            body.append(f"{label} & {_fmt(p50, 1)} / {_fmt(p99, 1)} & {_fmt(rate, 0)} & {_fmt(mem, 1)}" + r" \\")
    header = [r"Telemetry & p50 / p99" + DOWN + r" & Rate" + UP + r" & Memory" + DOWN + r" \\",
              r" & ms & s$^{-1}$ & GiB/day \\"]
    caption = (r"Operational measurements of \NagaHana{} (protocol P8): latency per state update at the median and 99th "
               r"percentile, sustained rate of state updates per second, and memory growth per day of retention.")
    return _latex_table(caption, "tab:res-operations", r"@{}l *{3}{>{\centering\arraybackslash}X}@{}", header, body,
                        wide=False, source=_source(result), tabcolsep="3pt")


def res_profile(result: ProtocolResult, cfg: EvaluationConfig) -> str:
    """Compute and latency profile (P8): per-stage latency quantiles with intervals, operations and peak memory."""
    m = _rows(result.metrics, task="profile", model=cfg.models.primary)
    if m.empty:
        return ""
    body = []
    for g in dict.fromkeys(m["group"].astype(str)):
        sel = _rows(m, group=g)
        for stage in [c for c in dict.fromkeys(sel["condition"].astype(str)) if c]:
            p50 = _value(sel, condition=stage, metric="stage_latency_p50_ms")
            p99 = _value(sel, condition=stage, metric="stage_latency_p99_ms")
            body.append(f"{g.replace('_', ' ')} & {stage} & {_ci(*p50, 2)} & {_ci(*p99, 2)}" + r" \\")
        for metric, label, dec in (("flops_per_update", "operations per update", 0), ("peak_memory_gib", "peak memory (GiB)", 2)):
            v = _value(_rows(sel, condition=""), metric=metric)
            if not np.isnan(v[0]):
                body.append(f"{g.replace('_', ' ')} & {label} & {_ci(*v, dec)} & -- " + r"\\")
    header = [r"Telemetry & Stage & p50 (ms)" + DOWN + r" & p99 (ms)" + DOWN + r" \\"]
    caption = (r"Compute and latency profile of \NagaHana{} (protocol P8): latency of each pipeline stage at the median "
               r"and 99th percentile with order-statistic intervals, operations per state update and peak memory.")
    return _latex_table(caption, "tab:res-profile", r"@{}l l *{2}{>{\centering\arraybackslash}X}@{}", header, body,
                        wide=False, source=_source(result), tabcolsep="3pt")


def res_forensics(result: ProtocolResult, cfg: EvaluationConfig) -> str:
    """Thesis Table res-forensics: onset error, patient zero, narrative precision and recall (P7)."""
    m = _rows(result.metrics, task="forensics", model=cfg.models.primary)
    if m.empty:
        return ""
    k = cfg.forensics.top_k
    body = []
    for g in _groups(m, cfg):
        sel = _rows(m, group=g)
        unit = next((d.time_unit for d in cfg.datasets if d.group == g), "min")
        on = _value(sel, metric="median_onset_error_s")[0]
        p1, pk = _value(sel, metric="patient_zero_top1")[0], _value(sel, metric=f"patient_zero_top{k}")[0]
        npr, nre = _value(sel, metric="narrative_precision")[0], _value(sel, metric="narrative_recall")[0]
        body.append(f"{cfg.group(g).label} & {_time(on, unit)} & {_fmt(p1, 1, percent=True)}\\,/\\,{_fmt(pk, 1, percent=True)} & "
                    f"{_fmt(npr, 1, percent=True)} & {_fmt(nre, 1, percent=True)}" + r" \\")
    header = [r"Data & Onset error" + DOWN + r" & Patient zero" + UP + r" &",
              r"  Narr.\ prec." + UP + r" & Narr.\ rec." + UP + r" \\"]
    caption = (r"Forensic backtesting on labelled incidents (protocol P7): stage-onset timing error, patient-zero accuracy "
               rf"at top-1 and top-{k}, and precision and recall of the steps of the reconstructed narrative.")
    return _latex_table(caption, "tab:res-forensics", r"@{}l *{4}{>{\centering\arraybackslash}X}@{}", header, body,
                        wide=False, source=_source(result), tabcolsep="2pt")


def _scaled_ci(t: tuple[float, float, float], scale: float, decimals: int, percent: bool) -> str:
    # "value [low, high]" after multiplying finite values by scale (unit conversion of times).
    v = [x * scale if np.isfinite(x) else x for x in t]
    return _ci(v[0], v[1], v[2], decimals, percent=percent)


def res_ci(results: Mapping[str, ProtocolResult], cfg: EvaluationConfig) -> str:
    """Thesis Table res-ci: headline comparisons with intervals and the paired test, daggers over the table."""
    lines, pvals = [], []
    lr = cfg.models.lr_family[0] if cfg.models.lr_family else ""
    for h in cfg.headline:
        res = results.get(h.protocol)
        if res is None:
            continue
        mrows = _rows(res.metrics, task=h.task, metric=h.metric, group=h.group, novelty=h.novelty or None,
                      variant=h.variant if h.variant else ("" if h.protocol != "P2" else None))
        crow = _rows(res.comparisons, task=h.task, metric=h.metric, group=h.group, novelty=h.novelty or None,
                     model_a=cfg.models.primary, model_b=lr, variant=h.variant if h.variant else None)
        if h.horizon:
            hs = [int(x) for x in mrows["horizon"].astype(str) if x.isdigit()]
            target = str(max(hs)) if (h.horizon == "K" and hs) else h.horizon
            mrows, crow = _rows(mrows, horizon=target), _rows(crow, horizon=target)
        a = _value(mrows, model=cfg.models.primary)
        b = _value(mrows, model=lr)
        c = _one(crow)
        if all(np.isnan(x) for x in (*a, *b)) and c is None:
            continue
        scale = 1.0
        if h.task == "timeliness":
            unit = cfg.dataset(h.group).time_unit if h.group != "all" else "min"
            scale = 1.0 / {"s": 1.0, "min": 60.0, "h": 3600.0}[unit]
        diff = (float(c["difference"]), float(c["ci_low"]), float(c["ci_high"])) if c is not None else (math.nan,) * 3
        lines.append([h.label, *(_scaled_ci(t, scale, h.decimals, h.percent) for t in (a, b, diff)),
                      str(c["test"]) if c is not None else "n/a"])
        pvals.append(float(c["p_value"]) if c is not None else math.nan)
    if not lines:
        return ""
    marks = _family_daggers([p for p in pvals if np.isfinite(p)], cfg)
    it = iter(marks)
    body = []
    test_names = {"mcnemar_midp": "McNemar", "mcnemar": "McNemar", "delong": "DeLong", "dm": "Diebold--Mariano",
                  "wilcoxon": "Wilcoxon signed-rank", "paired_bootstrap": "paired bootstrap"}
    for line, p in zip(lines, pvals, strict=True):
        flag = next(it) if np.isfinite(p) else False
        body.append(f"{line[0]} & {line[1]} & {line[2]} & {line[3]}{DAGGER if flag else ''} & "
                    f"{test_names.get(line[4], line[4])}" + r" \\")
    header = [r"Comparison & \NagaHana{} & Logistic regression & Difference & Paired test \\"]
    caption = (rf"Headline comparisons with {round(100 * float(next(iter(results.values())).config['resampling']['confidence']))}\% "
               r"bootstrap confidence intervals. Differences are \NagaHana{} minus logistic regression on the same evaluation "
               rf"units. $\dagger$: significant after false-discovery-rate control at level {cfg.comparisons.fdr_level:g} over "
               "the comparisons of this table with the paired test named in the last column.")
    src = "; ".join(_source(r) for r in results.values())
    return _latex_table(caption, "tab:res-ci", r"@{}>{\raggedright\arraybackslash}X l l l l@{}", header, body, wide=True,
                        source=src)


def res_matrix(results: Mapping[str, ProtocolResult], cfg: EvaluationConfig) -> str:
    """Thesis Table res-matrix: which protocol was run on which data source (from the scored runs)."""
    body = []
    for pid, proto in PROTOCOLS.items():
        if pid not in results:
            continue
        groups = set(results[pid].metrics["group"].astype(str))
        marks = [r"$\bullet$" if src in groups else "" for src, _ in MATRIX_SOURCES]
        body.append(f"{pid} {proto.name} & " + " & ".join(marks) + r" \\")
    if not body:
        return ""
    header = ["Protocol & " + " & ".join(label for _, label in MATRIX_SOURCES) + r" \\"]
    caption = (r"Evaluation matrix: protocols (rows) against data sources (columns). A bullet marks a protocol that was "
               r"scored on that source.")
    return _latex_table(caption, "tab:res-matrix", r"@{}>{\raggedright\arraybackslash}X " + "c" * len(MATRIX_SOURCES) + "@{}",
                        header, body, wide=True, source="; ".join(_source(r) for r in results.values()))


def res_arena(result: ProtocolResult, cfg: EvaluationConfig) -> str:
    """Arena table (P-CW): final return, improvement over the control, steps to exceed it, generalisation."""
    m = _rows(result.metrics, task="arena")
    if m.empty:
        return ""
    agents = list(dict.fromkeys(m["model"].astype(str)))
    if cfg.models.primary in agents:
        agents = [a for a in agents if a != cfg.models.primary] + [cfg.models.primary]
    kinds = ("held_out_strategy", "network_size", "observation")
    body = []
    comps = _rows(result.comparisons, family="arena")
    for a in agents:
        sel = _rows(m, model=a, condition="")
        fin = _value(sel, metric="final_iqm")
        imp = _value(sel, metric="improvement_over_control")
        stp = _value(sel, metric="steps_to_exceed")
        gen = []
        for kind in kinds:
            g = _rows(m, model=a, novelty=kind, metric="iqm")
            gen.append(_ci(*_value(g), 1) if len(g) else "n/a")
        poi = _one(comps, model_b=a)
        body.append(f"{cfg.display(a)} & {_ci(*fin, 1)} & {_ci(*imp, 1)} & {_ci(*stp, 0)} & " + " & ".join(gen) +
                    f" & {_fmt(poi['effect'], 2) if poi is not None else '--'}" + r" \\")
    header = [r"Agent & Final IQM" + UP + r" & Gain over control" + UP + r" & Steps to exceed control" + DOWN + r" &",
              r"  Held-out strategy" + UP + r" & Larger network" + UP + r" & Degraded observation" + UP + r" &",
              r"  $P(\NagaHana{} > \text{agent})$ \\"]
    envs = sorted(set(m["group"].astype(str)))
    caption = (rf"Cyber-defence arena (protocol P-CW, {', '.join(envs)}): interquartile mean of the final return over seeds, "
               r"its gain over the control policy, environment steps until the evaluation return first exceeds the control, "
               r"and returns under generalisation conditions, each with percentile-bootstrap intervals over runs; the last "
               r"column is the probability of improvement of \NagaHana{} over the agent.")
    return _latex_table(caption, "tab:res-arena", r"@{}l *{7}{>{\centering\arraybackslash}X}@{}", header, body, wide=True,
                        source=_source(result), tabcolsep="3pt")


def res_faithfulness(results: Mapping[str, ProtocolResult], cfg: EvaluationConfig) -> str:
    """Explanation faithfulness: deletion and insertion areas, AOPC and gains over random orders, with intervals."""
    frames = [_rows(r.metrics, task="explanations", group="all") for r in results.values()]
    frames = [f if len(f) else _rows(r.metrics, task="explanations") for f, r in zip(frames, results.values(), strict=True)]
    m = pd.concat([f for f in frames if len(f)] or [pd.DataFrame()], ignore_index=True)
    if m.empty:
        return ""
    body = []
    for (model, method), sel in m.groupby(["model", "method"], sort=False):
        cells = [_ci(*_value(sel, metric=c), 3) for c in ("deletion_auc", "insertion_auc", "aopc", "deletion_gain",
                                                          "insertion_gain")]
        body.append(f"{cfg.display(str(model))} & {str(method).replace('_', ' ')} & " + " & ".join(cells) + r" \\")
    header = [r"Model & Method & Deletion AUC" + DOWN + r" & Insertion AUC" + UP + r" & AOPC" + UP + r" &",
              r"  Deletion gain" + UP + r" & Insertion gain" + UP + r" \\"]
    caption = ("Faithfulness of the explanations: areas under the deletion and insertion curves, the area over the "
               "perturbation curve and the gains over random feature orders, with bootstrap intervals over explained "
               "predictions.")
    return _latex_table(caption, "tab:res-faithfulness", r"@{}l l *{5}{>{\centering\arraybackslash}X}@{}", header, body,
                        wide=True, source="; ".join(_source(r) for r in results.values()), tabcolsep="3pt")


def critical_difference(results: Mapping[str, ProtocolResult], cfg: EvaluationConfig) -> dict[str, Any]:
    """Friedman, Iman-Davenport and Nemenyi results with the data of a critical-difference diagram per metric."""
    metrics = pd.concat([r.metrics for r in results.values()], ignore_index=True)
    metrics = metrics[~metrics["role"].isin(["ablation", "reference"])]
    out: dict[str, Any] = {}
    for spec in cfg.multidataset:
        sel = metrics
        if spec.horizon == "K":
            ks = sel[(sel["task"] == spec.task)]["horizon"].astype(str)
            kmax = str(max((int(h) for h in ks if h.isdigit()), default=0))
            mat = mds.score_matrix(sel, metric=spec.metric, task=spec.task, horizon=kmax)
        else:
            mat = mds.score_matrix(sel, metric=spec.metric, task=spec.task)
        if mat.empty or mat.shape[1] < 2 or mat.dropna().shape[0] < 2:
            out[spec.id] = {"status": "not enough complete blocks", "blocks": int(mat.shape[0]), "models": list(mat.columns)}
            continue
        res = mds.friedman_nemenyi(mat, higher_is_better=spec.higher_is_better, alpha=cfg.comparisons.fdr_level)
        wil = mds.wilcoxon_holm(mat)
        out[spec.id] = {"status": "ok", "metric": spec.metric, "task": spec.task, "higher_is_better": spec.higher_is_better,
                        **res.diagram(), "dropped_blocks": res.dropped_blocks,
                        "nemenyi_p": {f"{a}|{b}": float(res.nemenyi_p[i, j]) for i, a in enumerate(res.models)
                                      for j, b in enumerate(res.models) if i < j},
                        "wilcoxon_holm": frame_records(wil)}
    return out


def res_cd(cd: Mapping[str, Any], cfg: EvaluationConfig) -> str:
    """Average ranks, critical difference and cliques per metric (the data behind critical-difference diagrams)."""
    body = []
    for key, d in cd.items():
        if d.get("status") != "ok":
            continue
        ranks = ", ".join(f"{cfg.display(m)} {r:.2f}" for m, r in zip(d["models"], d["average_ranks"], strict=True))
        cl = "; ".join("{" + ", ".join(cfg.display(m) for m in c) + "}" for c in d["cliques"]) or "none"
        body.append(f"{key} & {d['blocks']} & {_fmt(d['iman_davenport_p'], 4)} & {_fmt(d['critical_difference'], 3)} & "
                    f"{ranks} & {cl}" + r" \\")
    if not body:
        return ""
    header = [r"Metric & Blocks & $p$ (Iman--Davenport) & CD & Average ranks (1 best) & Cliques \\"]
    caption = (r"Comparison of all models over datasets and protocols (Demsar 2006): Friedman test with the Iman--Davenport "
               r"correction, Nemenyi critical difference (CD) at level " f"{cfg.comparisons.fdr_level:g}" r", average ranks and "
               r"groups of models whose ranks differ by less than CD.")
    return _latex_table(caption, "tab:res-cd", r"@{}l c c c X X@{}", header, body, wide=True, source="all scored protocols")


def res_hypotheses(result: ProtocolResult) -> str:
    """Pre-registered hypotheses: difference, interval, adjusted one-sided p-value and verdict."""
    h = result.hypotheses
    if h is None or h.empty:
        return ""
    body = []
    for _, r in h.iterrows():
        body.append(f"{r['id']} & {r['statement']} & {_ci(r['difference'], r['ci_low'], r['ci_high'], 4)} & "
                    f"{_fmt(r['p_adjusted'], 4)} & {'yes' if bool(r['supported']) else 'no'} & {r['verdict']}" + r" \\")
    header = [r"Id & Hypothesis & Difference & Adjusted $p$ & Supported & Verdict \\"]
    reg = result.registration
    caption = (f"Pre-registered hypotheses of protocol {result.protocol} (registration {reg.get('id')}, SHA-256 "
               f"{str(reg.get('document_sha256', ''))[:16]}, registered {reg.get('created_utc')}).")
    return _latex_table(caption, f"tab:res-hypotheses-{result.protocol.lower()}", r"@{}l X l c c l@{}", header, body,
                        wide=True, source=_source(result))


def main_table_ci(result: ProtocolResult, cfg: EvaluationConfig, task: str, metrics: Sequence[tuple[str, str, int, bool]],
                  label: str, caption: str) -> str:
    """A main table with every cell as value [low, high]: rows (group, novelty, horizon, model), given metric columns."""
    m = _rows(result.metrics, task=task)
    m = m[m["role"] != "ablation"]
    if m.empty:
        return ""
    body = []
    keys = m[["group", "novelty", "horizon", "variant"]].drop_duplicates()
    for _, k in keys.iterrows():
        sel = _rows(m, group=k["group"], novelty=k["novelty"], horizon=k["horizon"], variant=k["variant"])
        for model in _ordered_models(sel, cfg, with_refs=True):
            cells = [_ci(*_value(_rows(sel, model=model), metric=name), dec, percent=pct) for name, _, dec, pct in metrics]
            if all(c.startswith("n/a [n/a") for c in cells):
                continue
            where = cfg.group(str(k["group"])).label if k["group"] != "all" else "All"
            extra = f", $k={k['horizon']}$" if k["horizon"] else ""
            body.append(f"{where}, {NOVELTY_LABEL.get(str(k['novelty']), k['novelty'])}{extra} & {cfg.display(model)} & "
                        + " & ".join(cells) + r" \\")
    if not body:
        return ""
    header = ["Cell & Method & " + " & ".join(title for _, title, _, _ in metrics) + r" \\"]
    return _latex_table(caption, label, r"@{}l l *{" + str(len(metrics)) + r"}{>{\centering\arraybackslash}X}@{}", header, body,
                        wide=True, source=_source(result), tabcolsep="2pt")


def _write(path: Path, text: str, written: list[Path]) -> None:
    if text:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        written.append(path)


def write_result(result: ProtocolResult, out_dir: str | Path, cfg: EvaluationConfig) -> list[Path]:
    """Write one protocol result: CSV and JSON of all rows, figure data, and the protocol's LaTeX tables."""
    out = Path(out_dir) / result.protocol
    written: list[Path] = []
    out.mkdir(parents=True, exist_ok=True)
    for name, frame in (("metrics", result.metrics), ("comparisons", result.comparisons), ("deviations", result.deviations),
                        ("hypotheses", result.hypotheses), ("inputs", result.inputs)):
        path = out / f"{name}.csv"
        frame.to_csv(path, index=False)
        written.append(path)
    for name, frame in result.figures.items():
        path = out / "figures" / f"{name}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
        written.append(path)
        _write(out / "figures" / f"{name}.json", to_json(frame_records(frame)), written)
    doc = {"protocol": result.protocol, "registration": result.registration, "config": result.config,
           "notes": result.notes, "metrics": frame_records(result.metrics), "comparisons": frame_records(result.comparisons),
           "deviations": frame_records(result.deviations), "hypotheses": frame_records(result.hypotheses),
           "inputs": frame_records(result.inputs)}
    _write(out / "result.json", to_json(doc), written)
    lx = out / "latex"
    pid = result.protocol
    if pid == "P1":
        _write(lx / "res-detection.tex", res_detection(result, cfg), written)
        for nov in ("known", "novel"):
            _write(lx / f"res-forecast-{nov}.tex", res_forecast(result, cfg, nov), written)
            _write(lx / f"res-state-{nov}.tex", res_state(result, cfg, nov), written)
        _write(lx / "res-detection-ci.tex", main_table_ci(
            result, cfg, "detection", [("f1", "F1" + UP, 2, True), ("fpr", "FPR" + DOWN, 2, True),
                                       ("auroc", "AUROC" + UP, 2, True), ("auprc", "AUPRC" + UP, 2, True)],
            "tab:res-detection-ci", "Detection (protocol P1) with bootstrap intervals."), written)
        _write(lx / "res-forecast-ci.tex", main_table_ci(
            result, cfg, "forecast", [("brier", "Brier" + DOWN, 3, False), ("bss_persistence", "BSS vs pers." + UP, 3, False),
                                      ("bss_climatology", "BSS vs clim." + UP, 3, False), ("crps", "CRPS" + DOWN, 3, False)],
            "tab:res-forecast-ci", r"Forecast quality of $\nhPinf(k)$ (protocol P1) with bootstrap intervals."), written)
        _write(lx / "res-stages-ci.tex", main_table_ci(
            result, cfg, "stage", [("top1", "Top-1" + UP, 2, True), ("top3", "Top-3" + UP, 2, True),
                                   ("macro_f1", "Macro-F1" + UP, 2, True), ("rps", "RPS" + DOWN, 3, False)],
            "tab:res-stages-ci", "Stage prediction (protocol P1) with bootstrap intervals."), written)
        _write(lx / "res-timeliness-ci.tex", main_table_ci(
            result, cfg, "timeliness", [("median_lead_time", "Median lead time (s)" + UP, 1, False),
                                        ("alerted_before_completion", "Alerted" + UP, 1, True),
                                        ("lead_time_at_fpr", r"Lead time at FPR $\alpha$ (s)" + UP, 1, False)],
            "tab:res-timeliness-ci", "Lead times (protocol P1) with cluster-bootstrap intervals over attack episodes."),
            written)
    if pid == "P2":
        _write(lx / "res-transfer.tex", res_transfer(result, cfg, "auprc") + "\n" + res_transfer(result, cfg, "f1"), written)
    if pid == "P3":
        _write(lx / "res-lono.tex", res_lono(result, cfg, site=False), written)
        _write(lx / "res-sitecal.tex", res_lono(result, cfg, site=True), written)
    if pid == "P7":
        _write(lx / "res-forensics.tex", res_forensics(result, cfg), written)
    if pid == "P8":
        _write(lx / "res-operations.tex", res_operations(result, cfg), written)
        _write(lx / "res-profile.tex", res_profile(result, cfg), written)
    if pid == "P-CW":
        _write(lx / "res-arena.tex", res_arena(result, cfg), written)
    _write(lx / "res-hypotheses.tex", res_hypotheses(result), written)
    return written


def write_report(results: Mapping[str, ProtocolResult], out_dir: str | Path, cfg: EvaluationConfig) -> list[Path]:
    """Write the tables that combine protocols: headline intervals, timeliness, stages, robustness, components,
    ablations, evaluation matrix, explanation faithfulness and the critical-difference comparison."""
    out = Path(out_dir) / "report"
    written: list[Path] = []
    _write(out / "res-ci.tex", res_ci(results, cfg), written)
    _write(out / "res-timeliness.tex", res_timeliness(results, cfg), written)
    _write(out / "res-stages.tex", res_stages(results, cfg), written)
    _write(out / "res-robustness.tex", res_robustness(results, cfg), written)
    _write(out / "res-components.tex", res_components(results, cfg), written)
    _write(out / "res-ablations.tex", res_ablations(results, cfg, with_intervals=False), written)
    _write(out / "res-ablations-ci.tex", res_ablations(results, cfg, with_intervals=True), written)
    _write(out / "res-matrix.tex", res_matrix(results, cfg), written)
    _write(out / "res-faithfulness.tex", res_faithfulness(results, cfg), written)
    cd = critical_difference(results, cfg)
    _write(out / "critical_difference.json", to_json(cd), written)
    _write(out / "res-cd.tex", res_cd(cd, cfg), written)
    cells = ablation_cells(results, cfg)
    if len(cells):
        path = out / "ablations.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        cells.to_csv(path, index=False)
        written.append(path)
    return written

