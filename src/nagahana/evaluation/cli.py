"""Command line of the evaluation: `python -m nagahana evaluate <command>`.

    evaluate register --protocol P1 --spec plan.yaml --registry DIR [--config FILE]
        pre-register a protocol: the plan (hypotheses, metrics, models, seeds, splits) and the evaluation
        configuration are hashed, time-stamped and appended to the registry (registry.py)
    evaluate verify --registry DIR
        check the hash chain and the documents of a registry
    evaluate run --protocol P1 --outputs a.npz b.npz ... --registry DIR [--registration ID] --out DIR [--config FILE]
        score saved ModelOutputs bundles under one protocol (or "all": every protocol present in the bundles),
        then write CSV, JSON, figure data and LaTeX tables, and the tables that combine protocols
    evaluate compare --protocol P1 --outputs ... --registry DIR [--registration ID] --out DIR [--config FILE]
        the paired comparisons only: NagaHana against every other model and every ablation against the full
        model, with intervals, tests and adjusted p-values, and the ablation table
    evaluate catalogue [--format text|json|markdown] [--component NAME]
        the evaluation catalogue (catalogue.py)
    evaluate config [--write PATH | --check PATH]
        print the evaluation configuration as YAML, write it, or check a file against the dataclasses

Every scoring command needs a registration of its protocol made before the runs.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path
from typing import Any

from nagahana.core.errors import NagaHanaError


def register_cli(subparsers: Any) -> None:
    """Add the `evaluate` command and its subcommands to an argparse subparsers object."""
    ev = subparsers.add_parser("evaluate", help="score model outputs under the evaluation protocols")
    sub = ev.add_subparsers(dest="evaluate_command", required=True)

    reg = sub.add_parser("register", help="pre-register a protocol (hypotheses, metrics, analysis plan, seeds, splits)")
    reg.add_argument("--protocol", required=True)
    reg.add_argument("--spec", required=True, help="YAML file with the registration plan")
    reg.add_argument("--registry", required=True, help="registry directory")
    reg.add_argument("--config", default=None, help="evaluation configuration YAML (defaults to the dataclass defaults)")
    reg.set_defaults(func=_cmd_register)

    ver = sub.add_parser("verify", help="verify the hash chain and documents of a registry")
    ver.add_argument("--registry", required=True)
    ver.set_defaults(func=_cmd_verify)

    for name, fn, hlp in (("run", _cmd_run, "score bundles and write every table"),
                          ("compare", _cmd_compare, "paired comparisons and the ablation table only")):
        q = sub.add_parser(name, help=hlp)
        q.add_argument("--protocol", required=True, help="P1 ... P8, P-ABL, P-CW, or all")
        q.add_argument("--outputs", required=True, nargs="+", help="ModelOutputs bundles (.npz) written by save_outputs")
        q.add_argument("--registry", required=True, help="registry directory holding the protocol's registration")
        q.add_argument("--registration", default=None, help="registration id (default: the latest of the protocol)")
        q.add_argument("--out", required=True, help="output directory")
        q.add_argument("--config", default=None, help="evaluation configuration YAML")
        q.set_defaults(func=fn)

    cat = sub.add_parser("catalogue", help="print the evaluation catalogue")
    cat.add_argument("--format", default="text", choices=("text", "json", "markdown"))
    cat.add_argument("--component", default=None)
    cat.set_defaults(func=_cmd_catalogue)

    cfg = sub.add_parser("config", help="print, write or check the evaluation configuration")
    group = cfg.add_mutually_exclusive_group()
    group.add_argument("--write", default=None, help="write the generated YAML to this path")
    group.add_argument("--check", default=None, help="validate this YAML against the dataclasses")
    cfg.set_defaults(func=_cmd_config)


def _load_spec(path: str) -> dict[str, Any]:
    from nagahana.core.config import load_yaml

    return load_yaml(path)


def _cmd_register(args: argparse.Namespace) -> int:
    from nagahana.evaluation.config import load_config, to_dict
    from nagahana.evaluation.protocols import get_protocol
    from nagahana.evaluation.registry import Registry, registration_from_plain

    proto = get_protocol(args.protocol)
    spec = _load_spec(args.spec)
    spec = dict(spec, protocol=proto.id, config=to_dict(load_config(args.config)))
    reg = registration_from_plain(spec)
    entry = Registry(args.registry).register(reg)
    print(f"registered {entry.id} for {entry.protocol} at {entry.created_utc}")
    print(f"document sha256 {entry.document_sha256}")
    print(f"entry    sha256 {entry.entry_sha256}")
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    from nagahana.evaluation.registry import Registry

    reg = Registry(args.registry)
    problems = reg.verify()
    for p in problems:
        print(f"problem: {p}")
    print(f"{len(reg.entries())} entries, {len(problems)} problem(s)")
    return 1 if problems else 0


def _score(args: argparse.Namespace) -> tuple[dict[str, Any], Any]:
    # Load bundles, group them by protocol, look up registrations and score.
    from nagahana.evaluation.config import load_config
    from nagahana.evaluation.predictions import load_outputs
    from nagahana.evaluation.protocols import get_protocol
    from nagahana.evaluation.registry import Registry
    from nagahana.evaluation.scorer import Scorer

    cfg = load_config(args.config)
    bundles = [(load_outputs(p), str(p)) for p in args.outputs]
    wanted = None if args.protocol.lower() == "all" else get_protocol(args.protocol).id
    by_protocol: dict[str, list[tuple[Any, str]]] = {}
    for b, src in bundles:
        pid = get_protocol(b.protocol).id
        if wanted is None or pid == wanted:
            by_protocol.setdefault(pid, []).append((b, src))
    if not by_protocol:
        raise NagaHanaError(f"no bundles of protocol {args.protocol}")
    registry = Registry(args.registry)
    scorer = Scorer(cfg)
    results = {}
    for pid, items in by_protocol.items():
        if args.registration and wanted is not None:
            registered = registry.get(args.registration)
        else:
            registered = registry.latest(pid)
        if registered is None:
            raise NagaHanaError(f"protocol {pid} is not pre-registered in {args.registry}")
        results[pid] = scorer.run(pid, [b for b, _ in items], [s for _, s in items], registration=registered)
    return results, cfg


def _summary(results: dict[str, Any]) -> None:
    for pid, res in results.items():
        major = int((res.deviations["severity"] == "major").sum()) if len(res.deviations) else 0
        print(f"{pid}: {len(res.metrics)} metric rows, {len(res.comparisons)} comparisons, "
              f"{len(res.deviations)} deviations ({major} major), {len(res.hypotheses)} hypotheses")
        for note in res.notes:
            print(f"  note: {note}")
        for _, d in res.deviations.iterrows():
            print(f"  deviation ({d['severity']}, {d['kind']}): {d['detail']}")


def _cmd_run(args: argparse.Namespace) -> int:
    from nagahana.evaluation.reports import write_report, write_result

    results, cfg = _score(args)
    written = []
    for res in results.values():
        written += write_result(res, args.out, cfg)
    written += write_report(results, args.out, cfg)
    _summary(results)
    print(f"wrote {len(written)} files to {args.out}")
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    from nagahana.evaluation.reports import ablation_cells, res_ablations

    results, cfg = _score(args)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for pid, res in results.items():
        path = out / f"comparisons-{pid}.csv"
        res.comparisons.to_csv(path, index=False)
        print(f"{pid}: {len(res.comparisons)} comparisons -> {path}")
    cells = ablation_cells(results, cfg)
    if len(cells):
        cells.to_csv(out / "ablations.csv", index=False)
        (out / "res-ablations-ci.tex").write_text(res_ablations(results, cfg, with_intervals=True), encoding="utf-8")
        print(f"ablation table: {len(cells)} cells")
    _summary(results)
    return 0


def _cmd_catalogue(args: argparse.Namespace) -> int:
    from nagahana.evaluation.catalogue import CATALOGUE

    rows = [m for m in CATALOGUE if args.component is None or m.component.lower() == args.component.lower()]
    if args.format == "json":
        print(json.dumps([dataclasses.asdict(m) for m in rows], indent=1))
    elif args.format == "markdown":
        print("| Component | Metric | Method | Status | Where |")
        print("|---|---|---|---|---|")
        for m in rows:
            print(f"| {m.component} | {m.name} | {m.method} | {m.status} | {m.where} |")
    else:
        for m in rows:
            print(f"{m.component:<11} {m.name}  ({m.status}) [{m.where}]\n{'':<11} method: {m.method}")
    return 0


def _cmd_config(args: argparse.Namespace) -> int:
    from nagahana.evaluation.config import load_config, write_yaml, yaml_text

    if args.write:
        print(f"wrote {write_yaml(args.write)}")
    elif args.check:
        load_config(args.check)
        print(f"{args.check}: valid")
    else:
        sys.stdout.write(yaml_text())
    return 0
