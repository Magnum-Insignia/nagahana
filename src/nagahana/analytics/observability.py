"""Observability: what each sensor sees, when coverage drops, how reliable the observations are.

Every field of every state update carries an observation status (P-03 taxonomy; D-41: absence is a
fact about the sensor, never a value). This module turns the status matrix into the observability
picture of a corpus.

Coverage over time
------------------
Per source, time bin (`bin_seconds`) and column: the share of updates in each status and the
contributing share (observed, stale or low reliability). Only columns that contribute somewhere in the
source are listed per bin; the others are covered by the capability matrix.

Gaps
----
For each (source, column) with median contributing share c_med > 0 over the bins that hold updates, a
bin is a gap bin when its contributing share is below `gap_drop` * c_med; maximal runs of at least
`gap_min_bins` gap bins are the coverage gaps (a sensor dropout, a feed that stopped, an encrypted
period).

Capability matrix
-----------------
Per adapter (sensor type) and column: "always" (contributing share >= 0.99), "mostly" (>= 0.5),
"sometimes" (> 0), "not observable" (no contributing cell, NOT_OBSERVABLE the most frequent status),
"never supplied" (otherwise). Per adapter and catalogue level (flow, packet, protocol, OT, derived, auth,
alert, device, event), the share of the level's canonical columns that contribute at all; per adapter
the share of the problem statement's required fields (`fields.required_ids`) it supplies.

Reliability and ordering
------------------------
Per source and column, the shares of STALE and LOW_RELIABILITY cells (only non-zero rows). Per source:
the clock quality declared by the adapter, the quantiles of the residual reorder uncertainty, the share
of updates that arrived after the watermark had passed them (event time < watermark: out of order),
the timestamp resolution as the smallest positive gap between distinct event times, and the share of
exactly simultaneous updates.

What the sensors saw of the attacks
-----------------------------------
Per source and attack family, the share of malicious updates in which at least one column of each
level contributes: an attack seen only through flow fields cannot reveal packet-level evidence, which
bounds what any model can forecast from that sensor (the information audit, P-15, quantifies it).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nagahana.analytics.config import ObservabilityConfig, to_dict
from nagahana.analytics.corpus import CONTRIBUTING_CODES, STATUS_NAMES, Corpus
from nagahana.analytics.report import Figure, Report
from nagahana.datamodel.columnar import STATUS_CODE
from nagahana.datamodel.fields import CATALOGUE, required_ids
from nagahana.datamodel.status import ObservationStatus

_CODE_NOT_OBS = STATUS_CODE[ObservationStatus.NOT_OBSERVABLE]
_CODE_STALE = STATUS_CODE[ObservationStatus.STALE]
_CODE_LOWREL = STATUS_CODE[ObservationStatus.LOW_RELIABILITY]


def _level(field_id: str) -> str:
    spec = CATALOGUE.get(field_id)
    return spec.level.value if spec is not None else "other"


def coverage_over_time(corpus: Corpus, cfg: ObservabilityConfig) -> pd.DataFrame:
    """Per source, bin and contributing column: status shares (module docstring)."""
    contributing = corpus.contributing()
    parts = []
    for si, info in enumerate(corpus.sources):
        rows = np.flatnonzero(corpus.source == si)
        if rows.size == 0:
            continue
        t = corpus.time[rows]
        timed = np.isfinite(t)
        if not timed.any():
            bins = np.zeros(rows.size, dtype=np.int64)
            starts = np.array([np.nan])
        else:
            t0 = float(np.nanmin(t))
            bins = np.where(timed, np.floor((t - t0) / cfg.bin_seconds), -1).astype(np.int64)
            starts = t0 + cfg.bin_seconds * np.arange(int(bins.max()) + 1)
        cols = np.flatnonzero(contributing[rows].any(axis=0))
        ok = bins >= 0
        st = corpus.status[rows][ok][:, cols]
        b = bins[ok]
        nb = starts.size
        n_per_bin = np.bincount(b, minlength=nb)
        for k, j in enumerate(cols.tolist()):
            counts = np.zeros((nb, len(STATUS_NAMES)))
            np.add.at(counts, (b, st[:, k].astype(np.int64)), 1.0)
            has = n_per_bin > 0
            frame = pd.DataFrame({"source": info.source_id, "adapter": info.adapter, "column": corpus.columns[j].name,
                                  "bin_start": starts[has], "updates": n_per_bin[has]})
            for c, name in enumerate(STATUS_NAMES):
                frame[f"share_{name}"] = counts[has, c] / n_per_bin[has]
            frame["share_contributing"] = counts[has][:, CONTRIBUTING_CODES.astype(np.int64)].sum(axis=1) / n_per_bin[has]
            parts.append(frame)
    cols_out = ["source", "adapter", "column", "bin_start", "updates", *[f"share_{n}" for n in STATUS_NAMES],
                "share_contributing"]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=cols_out)


def coverage_gaps(cov: pd.DataFrame, cfg: ObservabilityConfig) -> pd.DataFrame:
    """Runs of bins whose contributing share fell below gap_drop times the median (module docstring)."""
    rows = []
    for (src, col), g in cov.groupby(["source", "column"]):
        g = g.sort_values("bin_start")
        share = g["share_contributing"].to_numpy()
        med = float(np.median(share))
        if med <= 0:
            continue
        low = share < cfg.gap_drop * med
        if not low.any():
            continue
        edges = np.diff(np.concatenate([[0], low.astype(np.int8), [0]]))
        starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
        times = g["bin_start"].to_numpy()
        for a, b in zip(starts.tolist(), ends.tolist(), strict=True):
            if b - a >= cfg.gap_min_bins:
                rows.append({"source": src, "column": col, "gap_start": float(times[a]),
                             "gap_end": float(times[b - 1]) + cfg.bin_seconds, "bins": b - a, "median_share": med,
                             "share_in_gap": float(share[a:b].mean())})
    return pd.DataFrame(rows, columns=["source", "column", "gap_start", "gap_end", "bins", "median_share", "share_in_gap"])


def capability(corpus: Corpus) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(capability matrix in long form, level coverage per adapter, required-field coverage per adapter)."""
    contributing = corpus.contributing()
    adapters = corpus.row_attr("adapter")
    required = set(required_ids())
    cap_rows, level_rows, req_rows = [], [], []
    for ad in sorted({str(a) for a in adapters}):
        m = adapters == ad
        sub_c = contributing[m]
        sub_s = corpus.status[m]
        share = sub_c.mean(axis=0)
        for j, col in enumerate(corpus.columns):
            counts = np.bincount(sub_s[:, j], minlength=len(STATUS_NAMES))
            if share[j] >= 0.99:
                cls = "always"
            elif share[j] >= 0.5:
                cls = "mostly"
            elif share[j] > 0:
                cls = "sometimes"
            elif int(np.argmax(counts)) == _CODE_NOT_OBS:
                cls = "not observable"
            else:
                cls = "never supplied"
            cap_rows.append({"adapter": ad, "column": col.name, "field_id": col.field_id, "level": _level(col.field_id),
                             "share_contributing": float(share[j]), "capability": cls})
        cap = pd.DataFrame([r for r in cap_rows if r["adapter"] == ad])
        for lvl, g in cap.groupby("level"):
            level_rows.append({"adapter": ad, "level": lvl, "columns": len(g),
                               "columns_contributing": int((g["share_contributing"] > 0).sum()),
                               "share_columns_contributing": float((g["share_contributing"] > 0).mean())})
        fields_seen = set(cap.loc[cap["share_contributing"] > 0, "field_id"].tolist())
        req_in_matrix = {f for f in required if any(c.field_id == f for c in corpus.columns)}
        req_rows.append({"adapter": ad, "required_matrix_fields": len(req_in_matrix),
                         "supplied": len(req_in_matrix & fields_seen),
                         "share_supplied": len(req_in_matrix & fields_seen) / len(req_in_matrix) if req_in_matrix else float("nan"),
                         "missing": ", ".join(sorted(req_in_matrix - fields_seen))})
    return pd.DataFrame(cap_rows), pd.DataFrame(level_rows), pd.DataFrame(req_rows)


def reliability(corpus: Corpus) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(stale / low-reliability shares per source and column, ordering quality per source)."""
    rel_rows, ord_rows = [], []
    for si, info in enumerate(corpus.sources):
        m = corpus.source == si
        if not m.any():
            continue
        st = corpus.status[m]
        stale = (st == _CODE_STALE).mean(axis=0)
        low = (st == _CODE_LOWREL).mean(axis=0)
        for j in np.flatnonzero((stale > 0) | (low > 0)).tolist():
            rel_rows.append({"source": info.source_id, "column": corpus.columns[j].name, "share_stale": float(stale[j]),
                             "share_low_reliability": float(low[j])})
        t = corpus.time[m]
        wm = corpus.watermark[m]
        ru = corpus.reorder[m]
        has_wm = np.isfinite(wm) & np.isfinite(t)
        tf = np.sort(t[np.isfinite(t)])
        gaps = np.diff(np.unique(tf)) if tf.size > 1 else np.zeros(0)
        q = np.quantile(ru[np.isfinite(ru)], [0.5, 0.95, 1.0]) if np.isfinite(ru).any() else np.full(3, np.nan)
        ord_rows.append({
            "source": info.source_id, "adapter": info.adapter, "clock_quality": info.clock_quality or "",
            "updates": int(m.sum()), "timeless": info.timeless,
            "share_out_of_order": float((t[has_wm] < wm[has_wm]).mean()) if has_wm.any() else float("nan"),
            "reorder_median_s": float(q[0]), "reorder_q95_s": float(q[1]), "reorder_max_s": float(q[2]),
            "timestamp_resolution_s": float(gaps.min()) if gaps.size else float("nan"),
            "share_simultaneous": float(1.0 - np.unique(tf).size / tf.size) if tf.size else float("nan"),
        })
    return (pd.DataFrame(rel_rows, columns=["source", "column", "share_stale", "share_low_reliability"]),
            pd.DataFrame(ord_rows))


def attack_visibility(corpus: Corpus) -> pd.DataFrame:
    """Per source and family: share of malicious updates with a contributing column of each level."""
    mal = corpus.malicious == 1.0
    if not mal.any():
        return pd.DataFrame(columns=["source", "family", "malicious_updates", "level", "share_visible"])
    levels = np.array([_level(c.field_id) for c in corpus.columns], dtype=object)
    contributing = corpus.contributing()
    src = corpus.row_attr("source_id")
    rows = []
    frame = pd.DataFrame({"src": src[mal], "fam": corpus.family[mal].astype(str)})
    idx = np.flatnonzero(mal)
    for (s, f), g in frame.groupby(["src", "fam"]).groups.items():
        r = idx[np.asarray(g)]
        for lvl in sorted(set(levels.tolist())):
            cols = levels == lvl
            rows.append({"source": s, "family": f, "malicious_updates": int(r.size), "level": lvl,
                         "share_visible": float(contributing[np.ix_(r, cols)].any(axis=1).mean())})
    return pd.DataFrame(rows)


def run(corpus: Corpus, cfg: ObservabilityConfig | None = None) -> Report:
    """The observability analysis as one report (module docstring)."""
    cfg = cfg or ObservabilityConfig()
    rep = Report(kind="observability", title="Observability", provenance={"config": to_dict(cfg)})
    cov = coverage_over_time(corpus, cfg)
    rep.add_table("coverage_over_time", cov, title="Status shares per source, time bin and column")
    gaps = coverage_gaps(cov, cfg)
    rep.add_table("coverage_gaps", gaps, title="Coverage gaps")
    cap, lvl, req = capability(corpus)
    rep.add_table("capability", cap, title="Capability matrix (adapter x field)")
    rep.add_table("level_coverage", lvl, title="Coverage of catalogue levels per adapter")
    rep.add_table("required_coverage", req, title="Problem-statement fields supplied per adapter")
    rel, order = reliability(corpus)
    rep.add_table("reliability", rel, title="Stale and low-reliability shares")
    rep.add_table("ordering", order, title="Clock and ordering quality per source")
    vis = attack_visibility(corpus)
    rep.add_table("attack_visibility", vis, title="What the sensors saw of the attacks")
    rep.summary.update({
        "sources": len(corpus.sources), "adapters": sorted({s.adapter for s in corpus.sources}),
        "coverage_gaps": len(gaps),
        "columns_always_observed": int((cap["capability"] == "always").sum()) if len(cap) else 0,
        "required_share_supplied": {r["adapter"]: r["share_supplied"] for r in req.to_dict(orient="records")},
    })
    if len(cov):
        rep.add_figure(Figure(name="coverage_heatmap", kind="heatmap", data=cov[["source", "column", "bin_start",
                                                                                  "share_contributing"]],
                              x="bin_start", y="column", value="share_contributing",
                              title="Contributing share per column over time", description="One panel per source."))
    return rep


__all__ = ["attack_visibility", "capability", "coverage_gaps", "coverage_over_time", "reliability", "run"]
