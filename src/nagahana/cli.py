"""Command line: `python -m nagahana <command>`.

Commands
--------
- `decisions [--write-docs]`: decided / held / proposed, with the code that depends on each.
- `stages`: the six-stage training pipeline (pipeline/stages.py).
- `metrics`: the evaluation catalogue (evaluation/catalogue.py).
- `access`: the memory access matrix (memory/access.py).
- `check-config FILE`: list every `???` (undecided) key in a YAML config.

This is plain argparse so it works without optional dependencies. Hydra entry points for the training
stages are added when the `[config]` extra is installed. The problem statement's demonstration
interface (Streamlit / Flask / CLI) is held (D-28).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from nagahana.core.config import iter_missing, load_yaml
from nagahana.core.roles import Role
from nagahana.evaluation.catalogue import CATALOGUE
from nagahana.governance import report
from nagahana.memory.access import Region, matrix
from nagahana.pipeline.stages import STAGES


def _cmd_decisions(args: argparse.Namespace) -> int:
    text = report.markdown()
    if args.write_docs:
        out = Path(__file__).resolve().parents[2] / "docs" / "decisions.md"
        out.write_text(text + "\n", encoding="utf-8")
        print(f"wrote {out}")
    else:
        print(text)
    return 0


def _cmd_stages(_args: argparse.Namespace) -> int:
    for s in STAGES:
        print(f"{s.id}. {s.name}")
        print(f"   trains: {', '.join(s.trains) or '—'} | frozen: {', '.join(s.frozen) or '—'} | splits: {', '.join(s.splits)}")
        print(f"   self-supervised: {s.self_supervised} | human feedback: {s.human_feedback}")
        print(f"   waiting on: {', '.join(s.waiting_on) or '—'}")
    return 0


def _cmd_metrics(_args: argparse.Namespace) -> int:
    for m in CATALOGUE:
        where = f" [{m.where}]" if m.where else ""
        print(f"{m.component:<10} {m.name}  ({m.status}){where}\n{'':<10} method: {m.method}")
    return 0


def _cmd_access(_args: argparse.Namespace) -> int:
    mat = matrix()
    print(f"{'role':<12}" + "".join(f"{r.value:<14}" for r in Region))
    for role in Role:
        cells = "".join(f"{(mat[role][r].name or '-'):<14}" for r in Region)
        print(f"{role.value:<12}{cells}")
    return 0


def _cmd_check_config(args: argparse.Namespace) -> int:
    missing = list(iter_missing(load_yaml(args.file)))
    for key in missing:
        print(f"??? {key}")
    print(f"{len(missing)} undecided value(s) in {args.file}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    p = argparse.ArgumentParser(prog="nagahana", description="NagaHana research codebase")
    sub = p.add_subparsers(dest="command", required=True)
    d = sub.add_parser("decisions", help="decided / held / proposed, with code usage")
    d.add_argument("--write-docs", action="store_true", help="write docs/decisions.md")
    d.set_defaults(func=_cmd_decisions)
    sub.add_parser("stages", help="the six-stage training pipeline").set_defaults(func=_cmd_stages)
    sub.add_parser("metrics", help="the evaluation catalogue").set_defaults(func=_cmd_metrics)
    sub.add_parser("access", help="memory access matrix").set_defaults(func=_cmd_access)
    c = sub.add_parser("check-config", help="list undecided (???) keys in a YAML config")
    c.add_argument("file")
    c.set_defaults(func=_cmd_check_config)
    args = p.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
