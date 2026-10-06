"""Label and leakage audits: what a split shares across its boundaries, shortcut features, known labelling
issues of the public datasets, and the integrity of the family and novelty splits (D-23).

Temporal ordering and overlap
-----------------------------
Per network, for an evaluation split E (test, val, zero_shot) against train: the share of training
updates later than the first update of E ("train after eval"; 0 for a clean chronological split, the
temporal snooping of Arp et al., "Dos and Don'ts of Machine Learning in Computer Security", USENIX
Security 2022), and the length of time covered by both splits (the measure of the intersection of the
unions of their window intervals).

Shared units
------------
    windows    a window or segment id with rows in two splits (never allowed)
    flows      a flow (source, flow id) with updates in two splits: a PCAP flow emits several flow-state
               updates (D-51) and can straddle a split boundary
    sessions   the 5-tuple (network, initiator, responder, source port, destination port, protocol) in two
               splits: one connection exported as several flow records lands on both sides
    relations  (network, initiator, responder) pairs, and hosts: shares of an evaluation split's units
               that also occur in train (high within one network by design; 0 for a held-out network)
Duplicates across splits: rows of an evaluation split with an exact duplicate (value and status) in
train, and near-duplicate clusters (`duplicates.near_duplicates`) that mix splits.

Shortcut features
-----------------
For every column, the AUROC of its contributing values for malicious against benign updates (numeric
columns as values; categorical and bitmask columns through the label-stratified out-of-fold malicious
rate of their code, `discrimination.out_of_fold_rate`, with the optimistic in-sample rate beside it),
with the DeLong interval; the AUROC of the column's absence
indicator; the AUROC of the hour of day (UTC) of the event time (absolute clock time is kept out of
training by D-50 for exactly this reason); and the malicious share per source. A field whose
separability max(AUC, 1 - AUC) reaches `auroc_flag` alone is flagged for review: on lab datasets such
fields are often artefacts of the testbed (the attacker machine's TTL or window size, a fixed port)
rather than of the attack. Artefacts are specific to the testbed that produced them, and pooling
datasets dilutes them, so with several datasets the ranking is repeated per dataset.

Known labelling issues (executable checks)
------------------------------------------
    tcp_appendix       TCP flow records without a SYN that carry a FIN or RST and at most 3 packets: the
                       tail of a connection exported as a flow of its own when the flow meter closed the
                       flow at the first FIN (described for CICFlowMeter on CIC-IDS2017 by Engelen, Rimmer
                       and Joosen, "Troubleshooting an Intrusion Detection Dataset: the CICIDS2017 Case
                       Study", IEEE Security and Privacy Workshops 2021). Counted per label class.
    attempted_attack   malicious records the responder never answered (`flow.unanswered` = 1): attempts
                       that carried no exchange with the target; Engelen et al. (2021) relabel such flows
                       as attempted rather than successful attacks (exact criterion of the paper: citation
                       to verify).
    conflicting_duplicates  identical inputs with different labels (`duplicates.label_conflicts`).
    unknown_labels     updates whose label is neither benign nor malicious (D-41 applied to labels, AS-34).
Error prevalence in CIC-IDS2017 and CSE-CIC-IDS2018 more broadly: Liu, Engelen, Lynar, Essam and Joosen,
IEEE CNS 2022 (already cited by `data.labels` for the infiltration labels).

Split integrity (D-23)
----------------------
The manifest rules of `pipeline.splits.validate` (through `SplitManifest.validate`) when a manifest is
given, and row-level checks: zero-shot rows are real; generated rows are in train only; no family marked
novel in zero-shot occurs in train, val or test; and novelty leakage, the share of novel-family rows
with an exact duplicate in train.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nagahana.analytics import discrimination, duplicates
from nagahana.analytics.config import LeakageConfig, to_dict
from nagahana.analytics.corpus import EXCLUDED_CODES, Corpus
from nagahana.analytics.report import Report

EVAL_SPLITS: tuple[str, ...] = ("test", "val", "zero_shot")


def _interval_union(starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """Disjoint sorted intervals [k, 2] covering the union of [starts_i, ends_i]."""
    if starts.size == 0:
        return np.zeros((0, 2))
    o = np.argsort(starts, kind="stable")
    s, e = starts[o], ends[o]
    out = []
    cs, ce = s[0], e[0]
    for a, b in zip(s[1:].tolist(), e[1:].tolist(), strict=True):
        if a <= ce:
            ce = max(ce, b)
        else:
            out.append((cs, ce))
            cs, ce = a, b
    out.append((cs, ce))
    return np.array(out, dtype=np.float64)


def _intersection_length(a: np.ndarray, b: np.ndarray) -> float:
    """Total length of the intersection of two disjoint sorted interval sets."""
    total, i, j = 0.0, 0, 0
    while i < a.shape[0] and j < b.shape[0]:
        lo, hi = max(a[i, 0], b[j, 0]), min(a[i, 1], b[j, 1])
        if hi > lo:
            total += hi - lo
        if a[i, 1] < b[j, 1]:
            i += 1
        else:
            j += 1
    return total


def temporal_overlap(corpus: Corpus) -> pd.DataFrame:
    """Per network and evaluation split: train-after-eval share and overlapping time (module docstring)."""
    nets = corpus.row_attr("network")
    frame = pd.DataFrame({"net": nets, "split": corpus.split, "window": corpus.window, "t": corpus.time})
    frame = frame[(frame["split"] != "") & np.isfinite(frame["t"])]
    rows = []
    for net, sub in frame.groupby("net"):
        train = sub[sub["split"] == "train"]
        if train.empty:
            continue
        win = train.groupby("window")["t"].agg(["min", "max"])
        tr_int = _interval_union(win["min"].to_numpy(), win["max"].to_numpy())
        for split in EVAL_SPLITS:
            ev = sub[sub["split"] == split]
            if ev.empty:
                continue
            w2 = ev.groupby("window")["t"].agg(["min", "max"])
            ev_int = _interval_union(w2["min"].to_numpy(), w2["max"].to_numpy())
            first = float(ev["t"].min())
            rows.append({"network": net, "split": split, "train_rows": len(train), "eval_rows": len(ev),
                         "train_after_eval_share": float((train["t"] > first).mean()),
                         "overlap_seconds": _intersection_length(tr_int, ev_int),
                         "eval_span_seconds": float((ev_int[:, 1] - ev_int[:, 0]).sum()) if ev_int.size else 0.0})
    return pd.DataFrame(rows, columns=["network", "split", "train_rows", "eval_rows", "train_after_eval_share",
                                       "overlap_seconds", "eval_span_seconds"])


def _shared(keys: pd.Series, split: np.ndarray, name: str) -> list[dict[str, object]]:
    """Units (keys) by the splits they occur in: units in several splits, and eval units seen in train."""
    frame = pd.DataFrame({"k": keys.to_numpy(), "s": split})
    frame = frame[(frame["s"] != "") & frame["k"].notna()]
    per = frame.groupby("k")["s"].agg(lambda s: frozenset(s))
    multi = per[per.map(len) > 1]
    rows: list[dict[str, object]] = [{"unit": name, "split": "any", "units": int(per.size), "units_in_several_splits":
                                      int(multi.size), "rows_of_those_units": int(frame["k"].isin(multi.index).sum()),
                                      "share_seen_in_train": float("nan")}]
    train_units = set(frame.loc[frame["s"] == "train", "k"].tolist())
    for ev_split in EVAL_SPLITS:
        units = set(frame.loc[frame["s"] == ev_split, "k"].tolist())
        if not units:
            continue
        rows.append({"unit": name, "split": ev_split, "units": len(units), "units_in_several_splits": float("nan"),
                     "rows_of_those_units": float("nan"), "share_seen_in_train": len(units & train_units) / len(units)})
    return rows


def shared_units(corpus: Corpus) -> pd.DataFrame:
    """Windows, flows, sessions, relations and hosts across splits (module docstring)."""
    nets = corpus.row_attr("network").astype(str)
    src = corpus.source.astype(str)
    e = corpus.entities
    rows: list[dict[str, object]] = []
    rows += _shared(pd.Series(corpus.window).replace("", np.nan), corpus.split, "window")
    rows += _shared(pd.Series(src + ":" + corpus.flow.astype(str)), corpus.split, "flow")
    sport = corpus.values[:, corpus.column_index("flow.src_port")] if corpus.has_column("flow.src_port") else np.full(len(corpus), np.nan)
    dport = corpus.values[:, corpus.column_index("flow.dst_port")] if corpus.has_column("flow.dst_port") else np.full(len(corpus), np.nan)
    proto = corpus.values[:, corpus.column_index("flow.protocol")] if corpus.has_column("flow.protocol") else np.full(len(corpus), np.nan)
    ok5 = (e[:, 0] >= 0) & (e[:, 1] >= 0) & np.isfinite(sport) & np.isfinite(dport) & np.isfinite(proto)
    five = pd.Series(nets + ":" + e[:, 0].astype(str) + ">" + e[:, 1].astype(str) + ":" + np.nan_to_num(sport).astype(np.int64).astype(str)
                     + ":" + np.nan_to_num(dport).astype(np.int64).astype(str) + ":" + np.nan_to_num(proto).astype(np.int64).astype(str))
    rows += _shared(five.where(ok5), corpus.split, "session_5tuple")
    okr = (e[:, 0] >= 0) & (e[:, 1] >= 0)
    rel = pd.Series(e[:, 0].astype(str) + ">" + e[:, 1].astype(str))
    rows += _shared(rel.where(okr), corpus.split, "relation")
    hosts_k = np.concatenate([e[:, 0], e[:, 1]])
    hosts_s = np.concatenate([corpus.split, corpus.split])
    host_keys = pd.Series(hosts_k.astype(str)).where(hosts_k >= 0)      # entity -1 (none) is not a host
    rows += _shared(host_keys, hosts_s, "host")
    return pd.DataFrame(rows)


def cross_split_duplicates(corpus: Corpus, cfg: LeakageConfig) -> tuple[pd.DataFrame, dict[str, object]]:
    """Exact duplicates of evaluation rows in train, and near-duplicate clusters mixing splits."""
    ex = duplicates.exact_duplicates(corpus.values, corpus.status)
    frame = pd.DataFrame({"g": ex.group, "s": corpus.split, "fam": corpus.family.astype(str)})
    in_train = set(frame.loc[(frame["g"] >= 0) & (frame["s"] == "train"), "g"].tolist())
    rows = []
    for split in EVAL_SPLITS:
        m = frame["s"] == split
        if not m.any():
            continue
        dup = m & frame["g"].isin(in_train)
        rows.append({"split": split, "rows": int(m.sum()), "rows_with_exact_duplicate_in_train": int(dup.sum()),
                     "contamination_rate": float(dup.sum() / m.sum())})
    n = len(corpus)
    rng = np.random.default_rng(cfg.seed)
    sel = np.sort(rng.choice(n, size=cfg.near_max_rows, replace=False)) if n > cfg.near_max_rows else np.arange(n)
    nd = duplicates.near_duplicates(corpus.values[sel], corpus.status[sel], corpus.numeric_columns(),
                                    tolerance=cfg.near_tolerance, threshold=cfg.near_threshold, bands=cfg.near_bands,
                                    rows_per_band=cfg.near_rows_per_band, max_bucket=cfg.near_max_bucket,
                                    pivots=cfg.near_pivots, seed=cfg.seed)
    cl = pd.DataFrame({"c": nd.cluster, "s": corpus.split[sel]})
    cl = cl[(cl["c"] >= 0) & (cl["s"] != "")]
    mixed = cl.groupby("c")["s"].nunique()
    summary: dict[str, object] = {"near_clusters": int(nd.sizes.size), "near_clusters_mixing_splits": int((mixed > 1).sum()),
               "near_rows_in_mixed_clusters": int(cl["c"].isin(mixed[mixed > 1].index).sum()),
               "near_rows_examined": int(sel.size)}
    return pd.DataFrame(rows, columns=["split", "rows", "rows_with_exact_duplicate_in_train", "contamination_rate"]), summary


def shortcut_features(corpus: Corpus, cfg: LeakageConfig) -> pd.DataFrame:
    """Single-feature AUROC ranking of values and absence indicators (module docstring)."""
    y = corpus.malicious
    numeric = corpus.numeric_columns()
    contributing = corpus.contributing()
    rows = []
    for j, col in enumerate(corpus.columns):
        c = contributing[:, j]
        absent = np.isin(corpus.status[:, j], EXCLUDED_CODES).astype(np.float64)
        row: dict[str, object] = {"feature": col.name, "kind": col.kind.value, "contributing_rows": int(c.sum())}
        if c.sum() >= 2:
            vals = corpus.values[c, j]
            if numeric[j] or np.unique(vals).size < 2:
                # Values directly; a single-valued field ties every row and gets AUC 0.5 exactly.
                a = discrimination.auroc(np.where(c, corpus.values[:, j], np.nan), y)
            else:
                a = discrimination.out_of_fold_auroc(vals, y[c], folds=cfg.oof_folds, smoothing=cfg.oof_smoothing,
                                                     seed=cfg.seed)
                ins = np.full(len(corpus), np.nan)
                ins[c] = discrimination.in_sample_rate(vals, y[c])
                row["auroc_in_sample"] = discrimination.auroc(ins, y).auc
            row.update({"auroc": a.auc, "auroc_lower": a.lower, "auroc_upper": a.upper, "separability": a.separability,
                        "direction": a.direction, "n_pos": a.n_pos, "n_neg": a.n_neg})
        ab = discrimination.auroc(absent, y) if 0 < absent.mean() < 1 else None
        row["absence_auroc"] = ab.auc if ab is not None else float("nan")
        row["absence_separability"] = ab.separability if ab is not None else float("nan")
        seps = [float(x) for x in (row.get("separability", float("nan")), row["absence_separability"])
                if isinstance(x, int | float) and np.isfinite(x)]
        sep = max(seps, default=0.0)
        row["flagged"] = bool(sep >= cfg.auroc_flag)
        rows.append(row)
    t = corpus.time
    if np.isfinite(t).any():
        timed = np.isfinite(t)
        hour = np.floor((t[timed] % 86400.0) / 3600.0)
        a = discrimination.out_of_fold_auroc(hour, y[timed], folds=cfg.oof_folds, smoothing=cfg.oof_smoothing, seed=cfg.seed)
        rows.append({"feature": "(hour of day, UTC; not a model input, D-50)", "kind": "clock", "contributing_rows":
                     int(np.isfinite(t).sum()), "auroc": a.auc, "auroc_lower": a.lower, "auroc_upper": a.upper,
                     "separability": a.separability, "direction": a.direction, "n_pos": a.n_pos, "n_neg": a.n_neg,
                     "absence_auroc": float("nan"), "absence_separability": float("nan"),
                     "flagged": bool(a.separability >= cfg.auroc_flag) if np.isfinite(a.separability) else False})
    out = pd.DataFrame(rows)
    if "separability" in out:
        out = out.sort_values("separability", ascending=False, na_position="last").reset_index(drop=True)
    return out


def source_label_shares(corpus: Corpus) -> pd.DataFrame:
    """Malicious, benign and unknown shares per source: a source whose labels are all of one kind turns the
    source itself (its day, its sensor) into a shortcut."""
    frame = pd.DataFrame({"source": corpus.row_attr("source_id"), "dataset": corpus.row_attr("dataset"),
                          "label": corpus.label_class()})
    g = frame.groupby(["source", "dataset", "label"]).size().unstack(fill_value=0)
    for col in ("malicious", "benign", "unknown"):
        if col not in g:
            g[col] = 0
    g["rows"] = g[["malicious", "benign", "unknown"]].sum(axis=1)
    g["malicious_share"] = g["malicious"] / g["rows"]
    return g.reset_index()


def known_issues(corpus: Corpus) -> pd.DataFrame:
    """Executable checks of documented labelling issues (module docstring)."""
    label = corpus.label_class()

    def col(name: str) -> np.ndarray:
        return corpus.values[:, corpus.column_index(name)] if corpus.has_column(name) else np.full(len(corpus), np.nan)

    proto, syn, fin, rst = col("flow.protocol"), col("flow.flag_count.syn"), col("flow.flag_count.fin"), col("flow.flag_count.rst")
    pkts = col("flow.packets_total")
    pk_alt = col("flow.packets_fwd") + col("flow.packets_bwd")
    pkts = np.where(np.isfinite(pkts), pkts, pk_alt)
    unanswered = col("flow.unanswered")
    rows = []
    appendix = (proto == 6) & (syn == 0) & ((fin >= 1) | (rst >= 1)) & (pkts <= 3)
    checkable = np.isfinite(proto) & np.isfinite(syn) & np.isfinite(fin) & np.isfinite(pkts)
    attempted = (label == "malicious") & (unanswered == 1)
    ex = duplicates.exact_duplicates(corpus.values, corpus.status)
    conf = duplicates.label_conflicts(ex.group, pd.Series(corpus.malicious).to_numpy(dtype=object))
    for issue, hit, base, source in (
        ("tcp_appendix", appendix, checkable, "Engelen, Rimmer and Joosen, IEEE SPW 2021"),
        ("attempted_attack", attempted, (label == "malicious") & np.isfinite(unanswered),
         "Engelen, Rimmer and Joosen, IEEE SPW 2021 (criterion: citation to verify)"),
        ("unknown_labels", label == "unknown", np.ones(len(corpus), dtype=bool), "AS-34; D-41 applied to labels"),
    ):
        for cls in ("malicious", "benign", "unknown"):
            m = base & (label == cls)
            rows.append({"issue": issue, "label": cls, "checkable_rows": int(m.sum()), "hits": int((hit & m).sum()),
                         "share": float((hit & m).sum() / m.sum()) if m.sum() else float("nan"), "source": source})
    rows.append({"issue": "conflicting_duplicates", "label": "all", "checkable_rows": conf.n_known,
                 "hits": conf.conflicting_rows, "share": conf.error_floor, "source": "duplicate error floor (analytics.duplicates)"})
    return pd.DataFrame(rows)


def split_integrity(corpus: Corpus, manifest: object | None = None) -> tuple[pd.DataFrame, list[str]]:
    """(row-level checks, rule violations) of the D-23 split rules (module docstring)."""
    problems: list[str] = []
    if manifest is not None:
        from nagahana.core.errors import InvariantViolation
        try:
            manifest.validate()                                        # type: ignore[attr-defined]
        except InvariantViolation as exc:
            problems.append(f"manifest rule: {exc}")
    origin = corpus.row_attr("origin").astype(str)
    split = corpus.split.astype(str)
    fam = corpus.family.astype(str)
    checks = []
    zs_generated = int(((split == "zero_shot") & (origin != "real")).sum())
    gen_outside_train = int(((origin == "generated") & ~np.isin(split, ("train", "excluded", ""))).sum())
    novel_fams = set(fam[(split == "zero_shot") & (corpus.novelty == "novel") & (corpus.label_class() == "malicious")].tolist())
    seen = np.isin(split, ("train", "val", "test")) & (corpus.label_class() == "malicious")
    leaked = sorted(novel_fams & set(fam[seen].tolist()))
    ex = duplicates.exact_duplicates(corpus.values, corpus.status)
    train_groups = set(ex.group[(split == "train") & (ex.group >= 0)].tolist())
    novel_rows = (split == "zero_shot") & (corpus.novelty == "novel")
    nov_dup = novel_rows & np.isin(ex.group, list(train_groups)) & (ex.group >= 0)
    checks.append({"check": "zero_shot_rows_not_real", "violations": zs_generated})
    checks.append({"check": "generated_rows_outside_train", "violations": gen_outside_train})
    checks.append({"check": "novel_families_seen_in_train_val_test", "violations": len(leaked)})
    checks.append({"check": "novel_rows_with_exact_duplicate_in_train", "violations": int(nov_dup.sum())})
    if zs_generated:
        problems.append(f"{zs_generated} zero-shot rows are not real (D-23)")
    if gen_outside_train:
        problems.append(f"{gen_outside_train} generated rows outside train (D-23)")
    if leaked:
        problems.append(f"novel families also seen in train/val/test: {leaked} (D-23)")
    if nov_dup.any():
        problems.append(f"{int(nov_dup.sum())} novel-family rows have an exact duplicate in train (novelty leakage)")
    out = pd.DataFrame(checks)
    out["novel_families"] = ", ".join(sorted(novel_fams))
    return out, problems


def run(corpus: Corpus, cfg: LeakageConfig | None = None, *, manifest: object | None = None) -> Report:
    """The label and leakage audit as one report (module docstring)."""
    cfg = cfg or LeakageConfig()
    rep = Report(kind="leakage", title="Label and leakage audit", provenance={"config": to_dict(cfg)})
    has_splits = bool((corpus.split != "").any())
    rep.summary["splits_present"] = sorted({s for s in corpus.split.tolist() if s})
    if has_splits:
        ov = temporal_overlap(corpus)
        rep.add_table("temporal_overlap", ov, title="Temporal ordering and overlap of splits")
        rep.summary["max_train_after_eval_share"] = float(ov["train_after_eval_share"].max()) if len(ov) else float("nan")
        su = shared_units(corpus)
        rep.add_table("shared_units", su, title="Units shared across splits")
        multi = su[su["split"] == "any"].set_index("unit")["units_in_several_splits"]
        rep.summary.update({f"shared.{k}": int(v) for k, v in multi.items()})
        dup, near = cross_split_duplicates(corpus, cfg)
        rep.add_table("cross_split_duplicates", dup, title="Evaluation rows duplicated in train")
        rep.summary.update({f"duplicates.{k}": v for k, v in near.items()})
        integ, problems = split_integrity(corpus, manifest)
        rep.add_table("split_integrity", integ, title="Split integrity (D-23)")
        rep.summary["split_rule_violations"] = len(problems)
        rep.notes.extend(problems)
        if multi.get("window", 0):
            rep.notes.append("Some window ids carry rows of several splits; every window must have one role.")
    else:
        rep.notes.append("No split manifest was given; split-dependent audits were skipped.")
    sc = shortcut_features(corpus, cfg)
    rep.add_table("shortcut_features", sc, title="Single-feature separability (shortcut candidates)",
                  description=f"Flagged when max(AUC, 1 - AUC) >= {cfg.auroc_flag} for the values or for the absence indicator.")
    rep.summary["flagged_features"] = sc.loc[sc["flagged"], "feature"].tolist() if len(sc) else []
    # Testbed artefacts are usually specific to one dataset, and pooling datasets dilutes them: repeat per dataset.
    datasets = corpus.row_attr("dataset").astype(str)
    if len(set(datasets.tolist())) > 1:
        parts = []
        for ds in sorted(set(datasets.tolist())):
            sub = corpus.subset(datasets == ds)
            if np.isin(sub.malicious, (0.0, 1.0)).any() and (sub.malicious == 1.0).any() and (sub.malicious == 0.0).any():
                parts.append(shortcut_features(sub, cfg).assign(dataset=ds))
        by_ds = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        rep.add_table("shortcut_features_by_dataset", by_ds, title="Single-feature separability per dataset")
        if len(by_ds):
            flagged = by_ds[by_ds["flagged"]]
            rep.summary["flagged_features_by_dataset"] = {ds: g["feature"].tolist() for ds, g in flagged.groupby("dataset")}
    rep.add_table("source_label_shares", source_label_shares(corpus), title="Label shares per source")
    ki = known_issues(corpus)
    rep.add_table("known_issues", ki, title="Known labelling issues (executable checks)")
    return rep


__all__ = ["EVAL_SPLITS", "cross_split_duplicates", "known_issues", "run", "shared_units", "shortcut_features",
           "source_label_shares", "split_integrity", "temporal_overlap"]
