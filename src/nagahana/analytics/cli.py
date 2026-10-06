"""Command line of the analytics: `python -m nagahana analyze <analysis> ...`.

    analyze eda            exploratory analysis of a corpus
    analyze tda            topology of the corpus point cloud, the update-count series and the windows'
                           contact filtrations; or of a table's point cloud (--points) or series (--series)
    analyze spatial        traffic graphs per network and window, their measures and structural drift
    analyze temporal       event-time analysis of a corpus, or of a table's series (--series)
    analyze leakage        split overlap, shared units, cross-split duplicates, shortcuts, known issues, D-23
    analyze drift          reference against current (--reference KEY=VALUE --current KEY=VALUE)
    analyze observability  coverage over time, gaps, capabilities, reliability, attack visibility
    analyze info-audit     the information audit and Bayes ceiling (proposal P-15, `nagahana.lab.info_audit`)

Inputs
------
A corpus is read from flow CSVs (`--csv PATH --dataset NAME`, repeatable, paired in order), captures
(`--pcap PATH --labeller NAME`, with `--sandboxed` or `--not-sandboxed`) or a generic table
(`--table PATH` with `--label-column`, `--time-column`, `--family-column`, `--split-column`). Every
CSV and capture needs its `--network` (one value for all inputs, or one per input in order): entities
of one network are the same machines (D-48), so the network is never guessed. Splits come from the
project's split policy (`--splits` with `--preset`, `--split-mode`, `--novel-family`,
`--held-out-network`; `data.sampling.assign_splits`). The info audit can also read a table directly
(`--table` with `--hidden`, `--observables`, `--discrete`, `--regime`, `--group`, `--order`,
`--predictions`).

Outputs
-------
`--out DIR` (required) receives the report as JSON, CSV and Markdown (`--formats`). `--config FILE`
overrides the analysis configuration (the YAML files of conf/analytics/ carry the defaults);
`--set KEY=VALUE` overrides one field (the value is read as YAML).

Wiring: the package's cli.py calls `register_cli(subparsers)`. The info-audit command imports the lab
module only when it runs: the lab stays isolated from the core, and the audit runs only with P-15
enabled (`--enable-proposal P-15` or `enabled_proposals` in the configuration).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

#: Analyses exposed by `analyze`.
ANALYSES: tuple[str, ...] = ("eda", "tda", "spatial", "temporal", "leakage", "drift", "observability", "info-audit")
#: CSV dataset name -> adapter class name in `ingest.csv_flows`.
CSV_ADAPTERS: dict[str, str] = {"cic-ids2018": "CICFlowSource", "cic-ids2017": "CICFlowSource", "ctu13": "CTU13Source",
                                "ciciot2023": "CICIoT2023Source"}
#: PCAP labeller name -> function name in `data.labels`.
PCAP_LABELLERS: dict[str, str] = {"cic2018-infiltration-slice": "label_cic2018_infiltration_slice"}


def _add_inputs(q: argparse.ArgumentParser) -> None:
    g = q.add_argument_group("corpus inputs")
    g.add_argument("--csv", action="append", default=[], help="flow CSV file (repeatable)")
    g.add_argument("--dataset", action="append", default=[], choices=sorted(CSV_ADAPTERS),
                   help="dataset of each --csv, in order")
    g.add_argument("--pcap", action="append", default=[], help="packet capture (repeatable)")
    g.add_argument("--labeller", action="append", default=[], choices=sorted(PCAP_LABELLERS),
                   help="labeller of each --pcap, in order")
    sb = g.add_mutually_exclusive_group()
    sb.add_argument("--sandboxed", dest="sandboxed", action="store_true", default=None, help="the capture parser runs in a sandbox")
    sb.add_argument("--not-sandboxed", dest="sandboxed", action="store_false")
    g.add_argument("--network", action="append", default=[], help="network of the inputs (one, or one per input in order)")
    g.add_argument("--utc-offset-hours", type=float, default=None, help="local time zone of CSV timestamps (AS-301)")
    g.add_argument("--internal-network", action="append", default=None, help="CIDR of the monitored network (repeatable)")
    g.add_argument("--synthetic-internal", choices=("true", "false"), default=None,
                   help="CIC-IoT-2023: whether address-less rows are internal (AS-306)")
    g.add_argument("--max-rows", type=int, default=None, help="read at most this many rows per CSV")
    g.add_argument("--allow-unknown-labels", action="store_true", help="map unknown dataset labels to 'unknown'")
    g.add_argument("--table", default=None, help="generic CSV table")
    g.add_argument("--columns", default=None, help="comma-separated table columns to use")
    g.add_argument("--label-column", default=None)
    g.add_argument("--time-column", default=None)
    g.add_argument("--family-column", default=None)
    g.add_argument("--split-column", default=None)
    s = q.add_argument_group("splits")
    s.add_argument("--splits", action="store_true", help="assign splits with the project's split policy")
    s.add_argument("--preset", choices=("L", "tiny"), default=None, help="model preset whose window sizes plan the windows")
    s.add_argument("--split-mode", choices=("full", "pretrain"), default="full")
    s.add_argument("--novel-family", action="append", default=[], help="family held out as novel (repeatable)")
    s.add_argument("--held-out-network", action="append", default=[], help="network held out as zero-shot (repeatable)")


def _add_common(q: argparse.ArgumentParser) -> None:
    q.add_argument("--out", required=True, help="output directory")
    q.add_argument("--formats", default="json,csv,md", help="comma-separated subset of json, csv, md")
    q.add_argument("--config", default=None, help="YAML configuration of the analysis")
    q.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override one configuration field")


def register_cli(subparsers: Any) -> None:
    """Add `analyze` and its analyses to an argparse subparsers object (module docstring)."""
    p = subparsers.add_parser("analyze", help="data analytics of the corpora (data preparation)")
    sub = p.add_subparsers(dest="analysis", required=True)
    helps = {
        "eda": "exploratory data analysis", "tda": "topological data analysis", "spatial": "graph analytics",
        "temporal": "temporal analytics", "leakage": "label and leakage audits", "drift": "covariate, label and concept drift",
        "observability": "coverage, capabilities and reliability of the sensors",
        "info-audit": "information audit and Bayes ceiling (P-15)",
    }
    for name in ANALYSES:
        q = sub.add_parser(name, help=helps[name])
        _add_common(q)
        _add_inputs(q)
        if name == "tda":
            q.add_argument("--points", default=None, help="table columns (comma-separated) forming a point cloud")
            q.add_argument("--series", default=None, help="table column analysed as a series")
        if name == "temporal":
            q.add_argument("--series", default=None, help="table column analysed as a regular series")
            q.add_argument("--bin-seconds", type=float, default=None, help="sampling interval of --series in seconds")
        if name == "drift":
            q.add_argument("--reference", required=True, metavar="KEY=VALUE",
                           help="reference rows: split, network, dataset, source or family = value")
            q.add_argument("--current", required=True, metavar="KEY=VALUE", help="current rows, same keys")
            q.add_argument("--label", choices=("stage", "malicious"), default="stage", help="label of the shift estimate")
        if name == "info-audit":
            q.add_argument("--enable-proposal", action="append", default=[], help="enable a proposal (P-15 is required)")
            q.add_argument("--hidden", default="stage", help="table column of the hidden quantity, or stage / malicious")
            q.add_argument("--observables", default=None, help="comma-separated observable columns")
            q.add_argument("--discrete", default=None, help="comma-separated discrete observable columns (table input)")
            q.add_argument("--regime", default=None, help="table column of regime ids, or adapter / dataset / source")
            q.add_argument("--group", default=None, help="table column of sequence ids for the lagged audit")
            q.add_argument("--order", default=None, help="table column ordering each sequence")
            q.add_argument("--predictions", default=None, help="table column with a model's predicted labels")
        q.set_defaults(func=_dispatch)


def _overrides(items: list[str]) -> dict[str, Any]:
    import yaml

    out: dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"--set expects KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        out[key.strip()] = yaml.safe_load(value)
    return out


def _networks(args: argparse.Namespace, count: int) -> list[str]:
    nets = list(args.network)
    if count == 0:
        return []
    if len(nets) == 1:
        return nets * count
    if len(nets) != count:
        raise SystemExit(f"give one --network for all {count} inputs or one per input (got {len(nets)}); "
                         "the network names which machines are the same (D-48) and is never guessed")
    return nets


def load_corpus(args: argparse.Namespace) -> tuple[Any, Any]:
    """(Corpus, SplitManifest or None) from the command-line inputs (module docstring)."""
    import pandas as pd

    from nagahana.analytics.corpus import Corpus

    if args.table and not (args.csv or args.pcap):
        frame = pd.read_csv(args.table)
        cols = args.columns.split(",") if args.columns else None
        return Corpus.from_frame(frame, columns=cols, time_column=args.time_column, label_column=args.label_column,
                                 family_column=args.family_column, split_column=args.split_column), None
    from nagahana.data import labels as lab
    from nagahana.data.windows import SourceData, prepare_source
    from nagahana.ingest import csv_flows

    if len(args.csv) != len(args.dataset):
        raise SystemExit("give one --dataset per --csv, in order")
    if len(args.pcap) != len(args.labeller):
        raise SystemExit("give one --labeller per --pcap, in order")
    if not (args.csv or args.pcap):
        raise SystemExit("no input: give --csv/--dataset, --pcap/--labeller or --table")
    nets = _networks(args, len(args.csv) + len(args.pcap))
    sources = []
    for k, (path, dataset) in enumerate(zip(args.csv, args.dataset, strict=True)):
        cls = getattr(csv_flows, CSV_ADAPTERS[dataset])
        kw: dict[str, Any] = {"utc_offset_hours": args.utc_offset_hours, "max_rows": args.max_rows,
                              "internal_networks": args.internal_network}
        if args.synthetic_internal is not None:
            kw["synthetic_internal"] = args.synthetic_internal == "true"
        read = cls(path, **kw).read()
        labels = lab.map_labels(read.labels, dataset, allow_unknown=args.allow_unknown_labels)
        sources.append(SourceData(read.updates, labels, network=nets[k], dataset=dataset))
    if args.pcap:
        if args.sandboxed is None:
            raise SystemExit("captures need --sandboxed or --not-sandboxed")
        from nagahana.ingest.pcap import PcapSource

        for k, (path, labeller) in enumerate(zip(args.pcap, args.labeller, strict=True)):
            cu = PcapSource(path, sandboxed=args.sandboxed, emit="flow-state",
                            internal_networks=args.internal_network).columnar()
            labels = getattr(lab, PCAP_LABELLERS[labeller])(cu)
            sources.append(SourceData(cu, labels, network=nets[len(args.csv) + k], dataset=labeller))
    prepared = [prepare_source(s) for s in sources]
    manifest = None
    if args.splits:
        if args.preset is None:
            raise SystemExit("--splits needs --preset (its window sizes plan the windows)")
        from nagahana.data import sampling
        from nagahana.models.config import preset

        cfg = preset(args.preset)
        records = sampling.index_windows(prepared, cfg)
        policy = sampling.SplitPolicy.from_config(cfg, mode=args.split_mode, novel_families=frozenset(args.novel_family),
                                                  held_out_networks=frozenset(args.held_out_network))
        manifest = sampling.assign_splits(records, policy)
    return Corpus.from_prepared(prepared, manifest), manifest


def _select(corpus: Any, spec: str) -> Any:
    """Rows of the corpus selected by KEY=VALUE (split, network, dataset, source, family)."""
    import numpy as np

    if "=" not in spec:
        raise SystemExit(f"selector {spec!r} must be KEY=VALUE")
    key, value = spec.split("=", 1)
    if key == "split":
        col = corpus.split
    elif key in ("network", "dataset"):
        col = corpus.row_attr(key)
    elif key == "source":
        col = corpus.row_attr("source_id")
    elif key == "family":
        col = corpus.family
    else:
        raise SystemExit("selector keys: split, network, dataset, source, family")
    rows = np.asarray(col).astype(str) == value
    if not rows.any():
        raise SystemExit(f"no rows with {key} = {value!r}")
    return corpus.subset(rows)


def _info_audit(args: argparse.Namespace, cfg: Any) -> Any:
    import numpy as np
    import pandas as pd

    from nagahana.lab import info_audit

    enabled = tuple(args.enable_proposal) + tuple(cfg.enabled_proposals)
    if args.table and not (args.csv or args.pcap):
        frame = pd.read_csv(args.table)
        if not args.observables:
            raise SystemExit("--observables is required with --table")
        obs = args.observables.split(",")
        disc_names = set(args.discrete.split(",")) if args.discrete else set()
        values = frame[obs].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
        return info_audit.report(
            values, frame[args.hidden].to_numpy(dtype=object), discrete=np.array([c in disc_names for c in obs]),
            regimes=frame[args.regime].to_numpy(dtype=object) if args.regime else None,
            groups=frame[args.group].to_numpy(dtype=object) if args.group else None,
            order=frame[args.order].to_numpy(dtype=np.float64) if args.order else None,
            predictions=frame[args.predictions].to_numpy(dtype=object) if args.predictions else None,
            columns=obs, cfg=cfg, enabled_proposals=enabled)
    corpus, _ = load_corpus(args)
    from nagahana.datamodel.fields import Kind

    contributing = corpus.contributing()
    if args.observables:
        cols = [corpus.column_index(c) for c in args.observables.split(",")]
    else:
        cols = [j for j in range(len(corpus.columns)) if contributing[:, j].any()]
    disc = np.array([corpus.columns[j].kind in (Kind.CATEGORICAL, Kind.BITMASK) for j in cols])
    if args.hidden == "stage":
        known = corpus.stage >= 0
        hidden = corpus.stage
    elif args.hidden == "malicious":
        known = np.isin(corpus.malicious, (0.0, 1.0))
        hidden = corpus.malicious
    else:
        raise SystemExit("with corpus inputs --hidden is 'stage' or 'malicious'")
    reg = None
    if args.regime in ("adapter", "dataset"):
        reg = corpus.row_attr(args.regime)[known]
    elif args.regime == "source":
        reg = corpus.row_attr("source_id")[known]
    elif args.regime is not None:
        raise SystemExit("with corpus inputs --regime is adapter, dataset or source")
    values = corpus.values[np.ix_(np.flatnonzero(known), cols)]
    return info_audit.report(values, hidden[known].astype(object), discrete=disc, regimes=reg,
                             groups=corpus.entities[known, 0].astype(object), order=np.nan_to_num(corpus.time[known]),
                             columns=[corpus.columns[j].name for j in cols], cfg=cfg, enabled_proposals=enabled)


def _dispatch(args: argparse.Namespace) -> int:
    import pandas as pd

    from nagahana.analytics.config import load_config

    cfg = load_config(args.analysis, args.config, _overrides(args.set))
    formats = [f.strip() for f in args.formats.split(",") if f.strip()]
    kind = args.analysis
    if kind == "info-audit":
        report = _info_audit(args, cfg)
    elif kind in ("tda", "temporal") and args.table and getattr(args, "series", None):
        series = pd.read_csv(args.table)[args.series].to_numpy(dtype=float)
        if kind == "tda":
            from nagahana.analytics.tda import series_report

            report = series_report(series, cfg)
        else:
            from nagahana.analytics.temporal import series_report as t_series

            report = t_series(series, cfg, name=args.series, bin_seconds=args.bin_seconds)
    elif kind == "tda" and args.table and getattr(args, "points", None):
        from nagahana.analytics.tda import point_cloud_report

        pts = pd.read_csv(args.table)[args.points.split(",")].to_numpy(dtype=float)
        report = point_cloud_report(pts, cfg)
    else:
        corpus, manifest = load_corpus(args)
        if kind == "eda":
            from nagahana.analytics import eda

            report = eda.run(corpus, cfg)
        elif kind == "tda":
            from nagahana.analytics.tda import corpus_report

            report = corpus_report(corpus, cfg)
        elif kind == "spatial":
            from nagahana.analytics.spatial import corpus_report as sp_report

            report = sp_report(corpus, cfg)
        elif kind == "temporal":
            from nagahana.analytics.temporal import corpus_report as tm_report

            report = tm_report(corpus, cfg)
        elif kind == "leakage":
            from nagahana.analytics import leakage

            report = leakage.run(corpus, cfg, manifest=manifest)
        elif kind == "drift":
            from nagahana.analytics import drift

            report = drift.run(_select(corpus, args.reference), _select(corpus, args.current), cfg, label=args.label)
        elif kind == "observability":
            from nagahana.analytics import observability

            report = observability.run(corpus, cfg)
        else:
            raise SystemExit(f"unknown analysis {kind!r}")
    report.provenance["command"] = " ".join(sys.argv)
    written = report.write(Path(args.out), formats)
    for path in written:
        print(path)
    return 0


__all__ = ["ANALYSES", "CSV_ADAPTERS", "PCAP_LABELLERS", "load_corpus", "register_cli"]
