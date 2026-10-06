"""Command line for the world simulator: `nagahana worldsim generate|list` (P-14).

This module exposes `register_cli(subparsers)`, which the top-level CLI wires in (it never edits
cli.py itself). `worldsim list` prints the named scenario library; `worldsim generate` simulates a
scenario's worlds and writes their data-model records, labels, episodes and ground-truth tables to a
directory, along with the generated scenario YAML.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def register_cli(subparsers: Any) -> None:
    """Add the `worldsim` command group to the top-level CLI subparsers."""
    p = subparsers.add_parser("worldsim", help="ground-truth world simulator (P-14)")
    sub = p.add_subparsers(dest="worldsim_cmd", required=True)

    ls = sub.add_parser("list", help="list the named scenario library")
    ls.set_defaults(func=_cmd_list)

    gen = sub.add_parser("generate", help="simulate a scenario's worlds and write the dataset")
    gen.add_argument("--scenario", required=True, help="library scenario name, or a path to a scenario YAML")
    gen.add_argument("--worlds", type=int, required=True, help="number of worlds to simulate")
    gen.add_argument("--out", required=True, help="output directory")
    gen.add_argument("--seed", type=int, default=0, help="run seed (worlds are indices under this seed)")
    gen.add_argument("--backend", default="numpy", choices=("numpy", "jax"), help="simulation backend")
    gen.set_defaults(func=_cmd_generate)

    wl = sub.add_parser("write-library", help="write the library scenarios as YAML under a directory")
    wl.add_argument("--out", default="conf/worldsim", help="directory for the generated YAML files")
    wl.set_defaults(func=_cmd_write_library)


def _cmd_list(_args: argparse.Namespace) -> int:
    from nagahana.worldsim.scenarios import list_scenarios

    for name, description in list_scenarios():
        print(f"{name:28s} {description}")
    return 0


def _cmd_write_library(args: argparse.Namespace) -> int:
    from nagahana.worldsim.scenarios import write_library

    for path in write_library(args.out):
        print(f"wrote {path}")
    return 0


def _resolve_scenario(name_or_path: str) -> Any:
    from nagahana.worldsim.scenarios import SCENARIOS, load_scenario

    if name_or_path in SCENARIOS:
        return SCENARIOS[name_or_path]
    return load_scenario(name_or_path)


def _cmd_generate(args: argparse.Namespace) -> int:
    import numpy as np

    from nagahana.worldsim.scenarios import to_dict
    from nagahana.worldsim.simulate import simulate_worlds

    scenario = _resolve_scenario(args.scenario)
    out = Path(args.out) / scenario.name
    out.mkdir(parents=True, exist_ok=True)
    _dump_yaml(out / "scenario.yaml", to_dict(scenario))

    outputs = simulate_worlds(scenario, seed=args.seed, n_worlds=args.worlds, backend=args.backend)
    manifest: list[dict[str, Any]] = []
    for w in outputs:
        wd = out / f"w{w.meta['world_index']:04d}"
        wd.mkdir(exist_ok=True)
        cu = w.updates
        np.savez_compressed(wd / "matrix.npz", values=cu.values, status=cu.status, raw_hash=cu.raw_hash)
        cu.updates.to_csv(wd / "updates.csv", index=False)
        cu.entities.to_csv(wd / "entities.csv", index=False)
        _dump_json(wd / "columns.json", [{"name": c.name, "field_id": c.field_id, "kind": c.kind.value}
                                         for c in cu.columns])
        w.labels.to_csv(wd / "labels.csv", index=False)
        w.episodes.to_csv(wd / "episodes.csv", index=False)
        w.entity_truth.to_csv(wd / "entity_truth.csv", index=False)
        w.event_truth.to_csv(wd / "event_truth.csv", index=False)
        w.alerts.to_csv(wd / "alerts.csv", index=False)
        _dump_json(wd / "meta.json", w.meta)
        manifest.append(w.meta)
        print(f"world {w.meta['world_index']}: {w.meta['n_records']} records, "
              f"{w.meta['n_attack_events']} attack events, {len(w.episodes)} episodes"
              + (" (saturated)" if w.meta["saturated"] else ""))
    _dump_json(out / "manifest.json", {"scenario": scenario.name, "seed": args.seed,
                                       "worlds": len(outputs), "backend": args.backend, "meta": manifest})
    print(f"wrote {len(outputs)} worlds to {out}")
    return 0


def _dump_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _dump_yaml(path: Path, payload: Any) -> None:
    import yaml

    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=False), encoding="utf-8")


__all__ = ["register_cli"]
