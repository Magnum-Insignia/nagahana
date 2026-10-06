"""Exploratory data analysis of a corpus: distributions by observation status, missingness, tails,
class balance, dependence and duplicates.

Data preparation, the phase of the training pipeline that precedes Stages 1 to 4, begins with a deep
analysis of the raw open datasets (D-22). This module computes the exploratory part of that analysis on
a `Corpus` (every state update in the canonical column layout) and returns one `Report`.

Distributions by observation status
-----------------------------------
For every column the share of each of the five statuses (P-03 taxonomy) is reported, overall and per
dataset. Values are summarised separately for each contributing status (observed, stale, low
reliability), and optionally per group (label class, dataset, split, family): an absent cell is a
fact about the sensor, never a value (D-41), so it never enters a value statistic. Numeric columns get
the robust summary of `analytics.robust` and a histogram on the slog1p scale (AS-31) with
Freedman-Diaconis bins; categorical columns the most frequent codes; each bitmask column the share of
each bit of its code table in the field catalogue (for `flow.tcp_flags`: FIN 0x01, SYN 0x02, RST 0x04,
PSH 0x08, ACK 0x10, URG 0x20, ECE 0x40, CWR 0x80, RFC 793 and RFC 3168), or of each of its low 16 bits by
position when the catalogue documents none.

Missingness structure
---------------------
With X_ic = 1 when cell (i, c) is excluded (not supplied or not observable):
    co-absence N_cd = sum_i X_ic X_id,  Jaccard J_cd = N_cd / (N_c + N_d - N_cd),  phi_cd = corr(X_c, X_d).
The full status co-occurrence counts M[(c, a), (d, b)] = #{i : status_ic = a, status_id = b} are
listed for c < d. Status patterns (the vector of statuses of a row) are counted, most frequent first:
structural absence (a NetFlow source never supplies TTL) shows as a pattern tied to a source.
Missingness informativeness: per column, the absent share among malicious and benign updates, the
mutual information between the absence indicator and the label (Miller-Madow), and the AUROC of the
indicator. GRU-D (Che et al., Scientific Reports 8:6085, 2018) showed that missingness carries signal;
a field whose presence alone separates the classes is also a shortcut candidate (`analytics.leakage`).

Tails, class balance, dependence, duplicates
--------------------------------------------
Tail indices: `robust.tail_index` per numeric column (positive contributing values). Class balance:
counts and shares per dataset, split, family and ATT&CK stage (`models.vocab.STAGES`, AS-19), with
evenness measures: Shannon H, Pielou J = H / log K, effective number exp(H), Gini-Simpson 1 - sum p^2
and the imbalance ratio max/min. Dependence: `analytics.dependence` on a seeded row sample, pairwise
complete. Duplicates: `analytics.duplicates` (exact by value and status, raw by digest, near by LSH,
label conflicts and the duplicate error floor).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from nagahana.analytics import dependence as dep
from nagahana.analytics import discrimination, duplicates, information, robust
from nagahana.analytics.config import EDAConfig, to_dict
from nagahana.analytics.corpus import EXCLUDED_CODES, STATUS_NAMES, Corpus
from nagahana.analytics.report import Figure, Report
from nagahana.datamodel.columnar import STATUS_CODE
from nagahana.datamodel.fields import CATALOGUE, Kind
from nagahana.datamodel.status import CONTRIBUTING

_CONTRIB_STATUSES = tuple(s for s in STATUS_CODE if s in CONTRIBUTING)


def bit_table(field_id: str) -> tuple[tuple[str, int], ...]:
    """(name, bit) pairs of a bitmask field: its catalogue code table, else bits 0 ... 15 by position."""
    spec = CATALOGUE.get(field_id)
    codes = spec.codes if spec is not None else None
    if codes:
        return tuple((str(name), int(bit)) for bit, name in sorted(codes.items()))
    return tuple((f"bit{k}", 1 << k) for k in range(16))


def _groups(corpus: Corpus, key: str) -> np.ndarray:
    """object [n] group label of every row for a `group_by` key."""
    if key == "label":
        return corpus.label_class()
    if key == "dataset":
        return corpus.row_attr("dataset")
    if key == "network":
        return corpus.row_attr("network")
    if key == "split":
        return np.where(corpus.split == "", "unassigned", corpus.split).astype(object)
    if key == "family":
        return corpus.family.astype(object)
    raise ValueError(f"unknown group_by key {key!r}; known: label, dataset, network, split, family")


def status_tables(corpus: Corpus) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(per-column status counts and shares, per-dataset status shares in long format)."""
    n = len(corpus)
    rows = []
    for j, col in enumerate(corpus.columns):
        counts = np.bincount(corpus.status[:, j], minlength=len(STATUS_NAMES))
        row: dict[str, object] = {"column": col.name, "field_id": col.field_id, "kind": col.kind.value,
                                  "unit": col.unit or "", "n": n}
        for k, name in enumerate(STATUS_NAMES):
            row[f"n_{name}"] = int(counts[k])
            row[f"share_{name}"] = float(counts[k] / n) if n else float("nan")
        row["share_contributing"] = float(np.isin(corpus.status[:, j], EXCLUDED_CODES, invert=True).mean()) if n else float("nan")
        rows.append(row)
    datasets = corpus.row_attr("dataset")
    long = []
    for ds in pd.unique(datasets):
        m = datasets == ds
        sub = corpus.status[m]
        for j, col in enumerate(corpus.columns):
            counts = np.bincount(sub[:, j], minlength=len(STATUS_NAMES))
            for k, name in enumerate(STATUS_NAMES):
                if counts[k]:
                    long.append({"dataset": ds, "column": col.name, "status": name, "count": int(counts[k]),
                                 "share": float(counts[k] / m.sum())})
    return pd.DataFrame(rows), pd.DataFrame(long, columns=["dataset", "column", "status", "count", "share"])


def distributions(corpus: Corpus, cfg: EDAConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(numeric summaries, categorical frequencies, bitmask shares, histogram data), split by status and group."""
    group_cols = {key: _groups(corpus, key) for key in cfg.group_by}
    num_rows: list[dict[str, object]] = []
    cat_rows: list[dict[str, object]] = []
    bit_rows: list[dict[str, object]] = []
    hist_rows: list[pd.DataFrame] = []
    for j, col in enumerate(corpus.columns):
        st = corpus.status[:, j]
        vals = corpus.values[:, j]
        for status in _CONTRIB_STATUSES:
            sel = st == STATUS_CODE[status]
            if not sel.any():
                continue
            # The overall slice plus one slice per value of every group_by key.
            slices: list[tuple[str, str, np.ndarray]] = [("all", "all", sel)]
            for key, g in group_cols.items():
                for value in pd.unique(g[sel]):
                    slices.append((key, str(value), sel & (g == value)))
            for gkey, gval, m in slices:
                x = vals[m]
                base = {"column": col.name, "kind": col.kind.value, "status": status.value, "group_by": gkey, "group": gval}
                if col.kind in (Kind.CONTINUOUS, Kind.COUNT, Kind.HISTOGRAM):
                    num_rows.append({**base, **robust.robust_summary(x, trim=cfg.trim)})
                elif col.kind is Kind.BITMASK:
                    xi = x.astype(np.int64)
                    for name, bit in bit_table(col.field_id):
                        bit_rows.append({**base, "bit": name, "share_set": float(np.mean((xi & bit) != 0)) if x.size else
                                         float("nan"), "n": int(x.size)})
                else:
                    codes, counts = np.unique(x, return_counts=True)
                    order = np.argsort(-counts, kind="stable")
                    top = order[: cfg.top_categories]
                    for t in top.tolist():
                        cat_rows.append({**base, "category": float(codes[t]), "count": int(counts[t]),
                                         "share": float(counts[t] / x.size)})
                    rest = int(counts[order[cfg.top_categories:]].sum())
                    if rest:
                        cat_rows.append({**base, "category": float("nan"), "count": rest, "share": float(rest / x.size),
                                         "other": True})
            # Histogram on the slog1p scale (AS-31), overall slice of this status only.
            if col.kind in (Kind.CONTINUOUS, Kind.COUNT, Kind.HISTOGRAM):
                z = dep.slog1p(vals[sel])
                edges = robust.freedman_diaconis_edges(z, max_bins=cfg.max_bins)
                if edges.size >= 2:
                    counts, _ = np.histogram(z, bins=edges)
                    width = np.diff(edges)
                    hist_rows.append(pd.DataFrame({
                        "field": col.name, "status": status.value, "bin_left": edges[:-1], "bin_right": edges[1:],
                        "bin_center": 0.5 * (edges[:-1] + edges[1:]), "count": counts,
                        "density": counts / (counts.sum() * width) if counts.sum() else counts * 0.0,
                    }))
    hist = pd.concat(hist_rows, ignore_index=True) if hist_rows else pd.DataFrame(
        columns=["field", "status", "bin_left", "bin_right", "bin_center", "count", "density"])
    cat = pd.DataFrame(cat_rows)
    if "other" in cat:
        cat["other"] = cat["other"].fillna(False).astype(bool)
    return pd.DataFrame(num_rows), cat, pd.DataFrame(bit_rows), hist


def missingness(corpus: Corpus, *, patterns_top: int) -> dict[str, pd.DataFrame]:
    """Co-absence, status co-occurrence, status patterns and missingness informativeness (module docstring)."""
    n, c = corpus.status.shape
    names = [col.name for col in corpus.columns]
    x = np.isin(corpus.status, EXCLUDED_CODES).astype(np.float64)    # [n, C] absence indicators
    co = x.T @ x                                                      # [C, C] co-absence counts
    nc = np.diag(co)
    with np.errstate(divide="ignore", invalid="ignore"):
        jac = co / (nc[:, None] + nc[None, :] - co)
        mean = x.mean(axis=0)
        cov = co / n - mean[:, None] * mean[None, :]
        sd = np.sqrt(mean * (1 - mean))
        phi = cov / (sd[:, None] * sd[None, :])
    iu, ju = np.triu_indices(c, 1)
    coabs = pd.DataFrame({"column_i": np.array(names)[iu], "column_j": np.array(names)[ju], "both_absent": co[iu, ju],
                          "jaccard": jac[iu, ju], "phi": phi[iu, ju]})
    # Full status co-occurrence for c < d, accumulated over row chunks with one-hot matmuls (exact counts).
    s_n = len(STATUS_NAMES)
    acc = np.zeros((c * s_n, c * s_n))
    chunk = max(1, 4_000_000 // max(c * s_n, 1))
    for r0 in range(0, n, chunk):
        st = corpus.status[r0: r0 + chunk].astype(np.int64)
        onehot = np.zeros((st.shape[0], c * s_n), dtype=np.float32)
        onehot[np.arange(st.shape[0])[:, None], np.arange(c)[None, :] * s_n + st] = 1.0
        acc += (onehot.T @ onehot).astype(np.float64)
    rows = []
    for i_, j_ in zip(iu.tolist(), ju.tolist(), strict=True):
        block = acc[i_ * s_n:(i_ + 1) * s_n, j_ * s_n:(j_ + 1) * s_n]
        for a, b in zip(*np.nonzero(block), strict=True):
            rows.append((names[i_], names[j_], STATUS_NAMES[a], STATUS_NAMES[b], int(block[a, b])))
    cooc = pd.DataFrame(rows, columns=["column_i", "column_j", "status_i", "status_j", "count"])
    # Status patterns: one letter per column (first letters of the status names are unique: o s l n n?).
    letters = np.array([{"observed": "O", "stale": "S", "low_reliability": "L", "not_supplied": "N",
                         "not_observable": "X"}[s] for s in STATUS_NAMES])
    codes = information.factorize(*[corpus.status[:, j] for j in range(c)])[0] if c else np.zeros(n, dtype=np.int64)
    uniq, first, counts = np.unique(codes, return_index=True, return_counts=True)
    order = np.argsort(-counts, kind="stable")[:patterns_top]
    datasets = corpus.row_attr("dataset")
    pat_rows = []
    for t in order.tolist():
        r = first[t]
        members = codes == uniq[t]
        pat_rows.append({"pattern": "".join(letters[corpus.status[r]]), "count": int(counts[t]),
                         "share": float(counts[t] / n), "n_absent": int(x[r].sum()),
                         "datasets": ",".join(sorted({str(d) for d in pd.unique(datasets[members])}))})
    patterns = pd.DataFrame(pat_rows)
    # Informativeness of absence for the malicious label.
    y = corpus.malicious
    known = np.isin(y, (0.0, 1.0))
    info_rows = []
    for j in range(c):
        xi = x[:, j]
        row = {"column": names[j], "share_absent": float(xi.mean()) if n else float("nan")}
        if known.any() and 0 < xi[known].mean() < 1:
            row["share_absent_malicious"] = float(xi[known & (y == 1)].mean()) if (known & (y == 1)).any() else float("nan")
            row["share_absent_benign"] = float(xi[known & (y == 0)].mean()) if (known & (y == 0)).any() else float("nan")
            row["mi_absent_label"] = information.mutual_information_discrete(xi[known], y[known], correction="miller_madow")
            row["auroc_absent"] = discrimination.auroc(xi, y).auc
        info_rows.append(row)
    return {"coabsence": coabs, "status_cooccurrence": cooc, "status_patterns": patterns,
            "absence_informativeness": pd.DataFrame(info_rows)}


def tails(corpus: Corpus, cfg: EDAConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(tail-index table, stability paths) for numeric columns (contributing positive values)."""
    rows, paths = [], []
    contributing = corpus.contributing()
    for j, col in enumerate(corpus.columns):
        if col.kind not in (Kind.CONTINUOUS, Kind.COUNT, Kind.HISTOGRAM):
            continue
        x = corpus.values[contributing[:, j], j]
        ti = robust.tail_index(x, k_min=cfg.tail_k_min, max_share=cfg.tail_max_share, k_cap=cfg.tail_k_cap,
                               grid=cfg.tail_grid, level=cfg.tail_level)
        rows.append({"column": col.name, "n_positive": ti.n_positive, "k_star": ti.k_star, "alpha": ti.alpha_star,
                     "alpha_lower": ti.alpha_star_lower, "alpha_upper": ti.alpha_star_upper,
                     "gamma_moment": ti.gamma_moment_star, "ks_distance": ti.ks_star, "threshold": ti.threshold,
                     "distinct_tail_values": ti.distinct_tail_values})
        if ti.k.size:
            paths.append(pd.DataFrame({"field": col.name, "k": ti.k, "alpha_hill": ti.alpha_hill,
                                       "alpha_lower": ti.alpha_lower, "alpha_upper": ti.alpha_upper,
                                       "gamma_moment": ti.gamma_moment, "ks_distance": ti.ks_distance}))
    path = pd.concat(paths, ignore_index=True) if paths else pd.DataFrame(
        columns=["field", "k", "alpha_hill", "alpha_lower", "alpha_upper", "gamma_moment", "ks_distance"])
    return pd.DataFrame(rows), path


def _evenness(counts: np.ndarray) -> dict[str, float]:
    """Shannon H, Pielou J, effective number, Gini-Simpson and imbalance ratio of class counts."""
    c = np.asarray(counts, dtype=np.float64)
    c = c[c > 0]
    if c.size == 0:
        return {"classes": 0.0, "shannon": float("nan"), "pielou": float("nan"), "effective_number": float("nan"),
                "gini_simpson": float("nan"), "imbalance_ratio": float("nan")}
    p = c / c.sum()
    h = float(-(p * np.log(p)).sum())
    return {"classes": float(c.size), "shannon": h, "pielou": h / np.log(c.size) if c.size > 1 else float("nan"),
            "effective_number": float(np.exp(h)), "gini_simpson": float(1.0 - (p * p).sum()),
            "imbalance_ratio": float(c.max() / c.min())}


def class_balance(corpus: Corpus) -> dict[str, pd.DataFrame]:
    """Counts, shares and evenness per dataset, split, family, label class and ATT&CK stage."""
    from nagahana.models.vocab import STAGES

    frame = pd.DataFrame({
        "dataset": corpus.row_attr("dataset"), "network": corpus.row_attr("network"),
        "split": np.where(corpus.split == "", "unassigned", corpus.split), "family": corpus.family.astype(str),
        "label": corpus.label_class(), "stage": [STAGES[s][0] if 0 <= s < len(STAGES) else "unknown" for s in corpus.stage],
        "window": corpus.window,
    })
    out: dict[str, pd.DataFrame] = {}
    for keys, name in ((["dataset", "family"], "by_dataset_family"), (["split", "family"], "by_split_family"),
                       (["dataset", "label"], "by_dataset_label"), (["split", "label"], "by_split_label"),
                       (["dataset", "stage"], "by_dataset_stage")):
        g = frame.groupby(keys, dropna=False).size().rename("count").reset_index()
        g["share"] = g["count"] / g.groupby(keys[0])["count"].transform("sum")
        out[name] = g
    ev = []
    for key in ("dataset", "split"):
        for value, sub in frame.groupby(key):
            fam = sub[sub["label"] == "malicious"]["family"].value_counts().to_numpy()
            ev.append({"scope": key, "value": value, "updates": len(sub), "malicious": int((sub["label"] == "malicious").sum()),
                       "benign": int((sub["label"] == "benign").sum()), "unknown": int((sub["label"] == "unknown").sum()),
                       **{f"family_{k}": v for k, v in _evenness(fam).items()}})
    out["evenness"] = pd.DataFrame(ev)
    # Window-level balance: the dominant malicious family of each window (the data.sampling rule).
    w = frame[frame["window"] != ""]
    if len(w):
        mal = w[w["label"] == "malicious"].groupby("window")["family"].agg(lambda s: s.value_counts().index[0])
        has_benign = w[w["label"] == "benign"].groupby("window").size()
        windows = w.groupby("window")["split"].first().to_frame()
        windows["family"] = mal.reindex(windows.index)
        windows.loc[windows["family"].isna() & windows.index.isin(has_benign.index), "family"] = "benign"
        windows["family"] = windows["family"].fillna("unknown")
        g = windows.groupby(["split", "family"]).size().rename("windows").reset_index()
        g["share"] = g["windows"] / g.groupby("split")["windows"].transform("sum")
        out["windows_by_split_family"] = g
    return out


def dependence_table(corpus: Corpus, cfg: EDAConfig) -> pd.DataFrame:
    """Pairwise dependence of the columns on a seeded row sample (module docstring)."""
    rng = np.random.default_rng(cfg.seed)
    n = len(corpus)
    rows = np.sort(rng.choice(n, size=cfg.dependence_max_rows, replace=False)) if n > cfg.dependence_max_rows else np.arange(n)
    vals = corpus.values[rows]
    contributing = corpus.contributing()[rows]
    usable = [j for j in range(vals.shape[1])
              if contributing[:, j].sum() >= cfg.dependence_min_pairs and np.unique(vals[contributing[:, j], j]).size > 1]
    numeric = corpus.numeric_columns()
    methods = set(cfg.dependence_methods)
    unknown = methods - {"pearson", "spearman", "dcor", "mi"}
    if unknown:
        raise ValueError(f"unknown dependence methods {sorted(unknown)}")
    out = []
    for a_i, i in enumerate(usable):
        for j in usable[a_i + 1:]:
            both = contributing[:, i] & contributing[:, j]
            m = int(both.sum())
            row: dict[str, object] = {"column_i": corpus.columns[i].name, "column_j": corpus.columns[j].name, "n": m}
            if m < cfg.dependence_min_pairs:
                row["pair_kind"] = "too_few"
                out.append(row)
                continue
            xi, xj = vals[both, i], vals[both, j]
            if np.unique(xi).size < 2 or np.unique(xj).size < 2:
                row["pair_kind"] = "constant"
                out.append(row)
                continue
            if numeric[i] and numeric[j]:
                row["pair_kind"] = "numeric"
                if "pearson" in methods:
                    row["pearson"] = dep.pearson(xi, xj, transform=cfg.pearson_transform).r
                if "spearman" in methods:
                    row["spearman"] = dep.spearman(xi, xj).r
                if "dcor" in methods:
                    d = dep.distance_correlation(xi, xj, permutations=cfg.dcor_permutations, seed=cfg.seed)
                    row["dcor"], row["dcor_unbiased"], row["dcor_p_value"] = d.dcor, d.dcor_unbiased, d.p_value
                if "mi" in methods:
                    mi = information.mutual_information_ksg(dep.slog1p(xi), dep.slog1p(xj), k=cfg.mi_k, seed=cfg.seed)
                    row["mi"], row["info_coefficient"] = mi, information.information_coefficient(mi)
            elif not numeric[i] and not numeric[j]:
                row["pair_kind"] = "categorical"
                row["cramers_v"] = dep.cramers_v(xi, xj)
                if "mi" in methods:
                    mi = information.mutual_information_discrete(xi, xj, correction="miller_madow")
                    row["mi"], row["info_coefficient"] = mi, information.information_coefficient(mi)
            else:
                row["pair_kind"] = "mixed"
                cont, cat = (xi, xj) if numeric[i] else (xj, xi)
                if "mi" in methods:
                    mi = information.mutual_information_mixed(dep.slog1p(cont), cat, k=cfg.mi_k, seed=cfg.seed)
                    row["mi"], row["info_coefficient"] = mi, information.information_coefficient(mi)
            out.append(row)
    cols = ["column_i", "column_j", "pair_kind", "n", "pearson", "spearman", "dcor", "dcor_unbiased", "dcor_p_value",
            "mi", "info_coefficient", "cramers_v"]
    frame = pd.DataFrame(out)
    for c in cols:
        if c not in frame:
            frame[c] = np.nan
    return frame[cols]


def duplicate_tables(corpus: Corpus, cfg: EDAConfig) -> tuple[dict[str, object], dict[str, pd.DataFrame]]:
    """Exact, raw and near duplicates with label conflicts (module docstring)."""
    ex = duplicates.exact_duplicates(corpus.values, corpus.status)
    raw = duplicates.raw_duplicates(corpus.raw_hash)
    conf_mal = duplicates.label_conflicts(ex.group, pd.Series(corpus.malicious).to_numpy(dtype=object))
    conf_stage = duplicates.label_conflicts(ex.group, pd.Series(np.where(corpus.stage < 0, np.nan, corpus.stage)).to_numpy(dtype=object))
    n = len(corpus)
    rng = np.random.default_rng(cfg.seed)
    rows = np.sort(rng.choice(n, size=cfg.near_max_rows, replace=False)) if n > cfg.near_max_rows else np.arange(n)
    nd = duplicates.near_duplicates(
        corpus.values[rows], corpus.status[rows], corpus.numeric_columns(), tolerance=cfg.near_tolerance,
        threshold=cfg.near_threshold, bands=cfg.near_bands, rows_per_band=cfg.near_rows_per_band,
        max_bucket=cfg.near_max_bucket, pivots=cfg.near_pivots, seed=cfg.seed,
    )
    labels = corpus.label_class()[rows]
    clusters = []
    for k in range(nd.sizes.size):
        members = np.flatnonzero(nd.cluster == k)
        lab = pd.Series(labels[members]).value_counts()
        known = lab.drop("unknown", errors="ignore")
        clusters.append({"cluster": k, "rows": int(members.size), "malicious": int(lab.get("malicious", 0)),
                         "benign": int(lab.get("benign", 0)), "unknown": int(lab.get("unknown", 0)),
                         "label_conflict": bool(known.size >= 2)})
    summary: dict[str, object] = {
        "exact_duplicate_rows": ex.n_duplicate_rows, "exact_redundant_rows": ex.n_redundant_rows,
        "exact_duplicate_groups": int(ex.sizes.size),
        "exact_redundant_share": float(ex.n_redundant_rows / n) if n else float("nan"),
        "raw_duplicate_rows": raw.n_duplicate_rows, "raw_redundant_rows": raw.n_redundant_rows,
        "label_conflict_groups_malicious": conf_mal.conflicting_groups,
        "duplicate_error_floor_malicious": conf_mal.error_floor,
        "label_conflict_groups_stage": conf_stage.conflicting_groups,
        "duplicate_error_floor_stage": conf_stage.error_floor,
        "near_rows_examined": int(rows.size), "near_clusters": int(nd.sizes.size),
        "near_rows_in_clusters": int(nd.sizes.sum()), "near_candidates": nd.n_candidates, "near_verified": nd.n_verified,
    }
    pairs = nd.pairs.copy()
    if len(pairs):
        pairs["row_i"] = rows[pairs["row_i"].to_numpy()]
        pairs["row_j"] = rows[pairs["row_j"].to_numpy()]
    groups = pd.DataFrame({"group": np.arange(ex.sizes.size), "size": ex.sizes, "representative_row": ex.representative})
    return summary, {"exact_groups": groups, "near_clusters": pd.DataFrame(clusters, columns=[
        "cluster", "rows", "malicious", "benign", "unknown", "label_conflict"]), "near_pairs": pairs}


def run(corpus: Corpus, cfg: EDAConfig | None = None, *, sections: Sequence[str] | None = None) -> Report:
    """The full exploratory analysis of `corpus` as one report (module docstring).

    sections: subset of ("status", "distributions", "missingness", "tails", "balance", "dependence",
    "duplicates"); default all.
    """
    cfg = cfg or EDAConfig()
    all_sections = ("status", "distributions", "missingness", "tails", "balance", "dependence", "duplicates")
    chosen = tuple(sections) if sections is not None else all_sections
    bad = set(chosen) - set(all_sections)
    if bad:
        raise ValueError(f"unknown EDA sections {sorted(bad)}")
    rep = Report(kind="eda", title="Exploratory data analysis", provenance={"config": to_dict(cfg), "sections": list(chosen),
                 "sources": [s.__dict__ for s in corpus.sources]})
    n = len(corpus)
    lab = corpus.label_class()
    rep.summary.update({
        "rows": n, "sources": len(corpus.sources), "columns": len(corpus.columns),
        "entities": int(corpus.entity_kind.shape[0]),
        "datasets": sorted({s.dataset for s in corpus.sources}), "networks": sorted({s.network for s in corpus.sources}),
        "share_malicious": float(np.mean(lab == "malicious")) if n else float("nan"),
        "share_benign": float(np.mean(lab == "benign")) if n else float("nan"),
        "share_unknown_label": float(np.mean(lab == "unknown")) if n else float("nan"),
        "time_span_s": float(np.nanmax(corpus.time) - np.nanmin(corpus.time)) if np.isfinite(corpus.time).any() else float("nan"),
    })
    if "status" in chosen:
        st, st_ds = status_tables(corpus)
        rep.add_table("field_status", st, title="Observation status per field",
                      description="Counts and shares of the five observation statuses (P-03) per column.")
        rep.add_table("field_status_by_dataset", st_ds, title="Observation status per field and dataset")
        rep.summary["columns_never_supplied"] = int((st["share_contributing"] == 0).sum())
    if "distributions" in chosen:
        num, cat, bits, hist = distributions(corpus, cfg)
        rep.add_table("distributions_numeric", num, title="Numeric fields by status and group",
                      description="Robust summary of contributing values; absent cells never enter (D-41).")
        rep.add_table("distributions_categorical", cat, title="Categorical fields by status and group")
        rep.add_table("distributions_bitmask", bits, title="Bitmask fields: share of each bit by status and group")
        rep.add_figure(Figure(name="histograms", kind="histogram", data=hist, x="bin_center", y="density", series="status",
                              title="Value histograms (slog1p scale)", x_label="slog1p(value)", y_label="density",
                              description="One histogram per field (column 'field') and contributing status."))
    if "missingness" in chosen:
        miss = missingness(corpus, patterns_top=cfg.patterns_top)
        rep.add_table("coabsence", miss["coabsence"], title="Co-absence of fields",
                      description="Both cells excluded; Jaccard and phi of the absence indicators.")
        rep.add_table("status_cooccurrence", miss["status_cooccurrence"], title="Status co-occurrence (column pairs)")
        rep.add_table("status_patterns", miss["status_patterns"], title="Most frequent status patterns",
                      description="One letter per canonical column: O observed, S stale, L low reliability, "
                                  "N not supplied, X not observable.")
        rep.add_table("absence_informativeness", miss["absence_informativeness"], title="Is absence informative?")
        co = miss["coabsence"]
        rep.add_figure(Figure(name="coabsence_matrix", kind="heatmap", data=co, x="column_i", y="column_j", value="jaccard",
                              title="Jaccard co-absence of fields"))
    if "tails" in chosen:
        tab, paths = tails(corpus, cfg)
        rep.add_table("tail_index", tab, title="Tail index per numeric field",
                      description="Hill estimate at the Kolmogorov-Smirnov-optimal k (Clauset et al. 2009), "
                                  "moment estimator of the extreme-value index.")
        rep.add_figure(Figure(name="hill_stability", kind="line", data=paths, x="k", y="alpha_hill", series="field",
                              title="Hill stability paths", x_label="k (upper order statistics)", y_label="alpha",
                              log_x=True, description="alpha_lower/alpha_upper give the interval; gamma_moment and "
                                                      "ks_distance are on the same rows."))
        finite = tab[np.isfinite(tab["alpha"].astype(float))] if len(tab) else tab
        if len(finite):
            heaviest = finite.sort_values("alpha").iloc[0]
            rep.summary["heaviest_tail_field"] = str(heaviest["column"])
            rep.summary["heaviest_tail_alpha"] = float(heaviest["alpha"])
    if "balance" in chosen:
        for name, frame in class_balance(corpus).items():
            rep.add_table(f"balance_{name}", frame, title=f"Class balance: {name.replace('_', ' ')}")
    if "dependence" in chosen:
        dt = dependence_table(corpus, cfg)
        rep.add_table("dependence", dt, title="Pairwise dependence",
                      description="Pairwise-complete records of a seeded sample; Pearson on slog1p values (AS-31), "
                                  "Spearman, distance correlation, mutual information (KSG / plug-in / Ross), Cramer's V.")
        for measure in ("spearman", "dcor", "mi"):
            sub = dt[np.isfinite(dt[measure].astype(float))][["column_i", "column_j", measure]]
            rep.add_figure(Figure(name=f"dependence_{measure}", kind="heatmap", data=sub.reset_index(drop=True),
                                  x="column_i", y="column_j", value=measure, title=f"Pairwise {measure}"))
    if "duplicates" in chosen:
        summ, tabs = duplicate_tables(corpus, cfg)
        rep.summary.update({f"duplicates.{k}": v for k, v in summ.items()})
        for name, frame in tabs.items():
            rep.add_table(f"duplicates_{name}", frame, title=f"Duplicates: {name.replace('_', ' ')}")
    return rep


__all__ = ["bit_table", "class_balance", "dependence_table", "distributions", "duplicate_tables", "missingness", "run",
           "status_tables", "tails"]
