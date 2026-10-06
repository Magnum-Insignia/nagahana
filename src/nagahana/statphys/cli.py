"""Command line of the statistical-physics module: `python -m nagahana statphys <command>` (D-56).

Commands
--------
- `write-config PATH`: write the YAML rendering of the configuration dataclasses (the single source of
  truth, config.py) to PATH.
- `check-config PATH`: read PATH strictly against the dataclasses and list every value that differs
  from the dataclass default.
- `show FILE`: the series of a saved trajectory (io.py) or of the statphys component arrays of a
  ModelOutputs file, with their valid counts and ranges.
- `ews FILE --series NAME`: offline early-warning analysis of one series: detrending, rolling
  indicators, Kendall trends over the whole indicator series and their surrogate p-values.
- `calibrate FILE [FILE ...] --out JSON`: alarm calibration of the monitored series on benign
  trajectories; the target false-alarm rate is split over the series (Bonferroni: Dunn, "Multiple
  comparisons among means", JASA 56:52, 1961), so the alarm of any series keeps the target rate.
- `alarm FILE --calibration JSON`: replays the streaming indicators and the alarm over a saved
  trajectory and lists the alarm times.

Every command takes `--config PATH` (a statphys YAML; default: the dataclass defaults) and
`--json PATH` (write the result as JSON as well as printing it).
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from nagahana.statphys.config import DETRENDS, StatPhysConfig, load_config, render_yaml, to_mapping, write_yaml

# The analysis modules (ews, io and, through io, the trajectory and data-model code) are imported inside
# the commands, so registering the command costs no more than the configuration module at start-up.


def _config(args: argparse.Namespace) -> StatPhysConfig:
    return load_config(args.config) if getattr(args, "config", None) else StatPhysConfig()


def _emit(args: argparse.Namespace, result: dict[str, Any]) -> None:
    # Write the result as JSON when asked (NaN and infinities as null).
    path = getattr(args, "json", None)
    if not path:
        return

    def clean(x: Any) -> Any:
        if isinstance(x, float):
            return x if math.isfinite(x) else None
        if isinstance(x, dict):
            return {str(k): clean(v) for k, v in x.items()}
        if isinstance(x, list | tuple):
            return [clean(v) for v in x]
        return x

    Path(path).write_text(json.dumps(clean(result), indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {path}")


def _diff(a: Any, b: Any, prefix: str = "") -> list[str]:
    # Dotted keys whose values differ between two plain-data configurations.
    out: list[str] = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in a:
            out += _diff(a[k], b.get(k), f"{prefix}{k}.")
        return out
    if a != b:
        out.append(f"{prefix[:-1]}: {b!r} (default {a!r})")
    return out


def _cmd_write_config(args: argparse.Namespace) -> int:
    path = write_yaml(args.path, _config(args))
    print(f"wrote {path}")
    return 0


def _cmd_check_config(args: argparse.Namespace) -> int:
    cfg = load_config(args.path)
    changes = _diff(to_mapping(StatPhysConfig()), to_mapping(cfg))
    for line in changes:
        print(line)
    print(f"{args.path}: valid; {len(changes)} value(s) differ from the dataclass defaults")
    if args.strict and render_yaml(cfg) != Path(args.path).read_text(encoding="utf-8"):
        print("the file is not the rendering of its own values (regenerate it with write-config)")
        return 1
    return 0


def _cmd_show(args: argparse.Namespace) -> int:
    from nagahana.statphys.io import load_trajectory

    traj = load_trajectory(args.file)
    print(f"{args.file}: {traj.times.shape[0]} triggers ({int(traj.regular.sum())} cadence), meta {traj.meta}")
    result: dict[str, Any] = {}
    for name in sorted(traj.series):
        v = traj.series[name]
        ok = np.isfinite(v)
        rng = (float(v[ok].min()), float(v[ok].max())) if ok.any() else (math.nan, math.nan)
        result[name] = {"valid": int(ok.sum()), "min": rng[0], "max": rng[1]}
        print(f"{name:<48} valid {int(ok.sum()):>6}  range [{rng[0]:.6g}, {rng[1]:.6g}]")
    _emit(args, result)
    return 0


def _cmd_ews(args: argparse.Namespace) -> int:
    from nagahana.statphys.ews import early_warning, surrogate_test, trend_statistics
    from nagahana.statphys.io import load_trajectory

    cfg = _config(args)
    ews_cfg = cfg.ews
    if args.detrend is not None:
        ews_cfg = replace(ews_cfg, detrend=args.detrend)
    if args.surrogates is not None:
        ews_cfg = replace(ews_cfg, surrogates=args.surrogates)
    if args.seed is not None:
        ews_cfg = replace(ews_cfg, seed=args.seed)
    traj = load_trajectory(args.file)
    values, times = traj.values(args.series, regular_only=not args.all_triggers)
    res = early_warning(values, config=ews_cfg, times=times)
    taus = trend_statistics(res.indicators)
    print(f"{args.series}: {res.values.shape[0]} valid samples, window {ews_cfg.window}, detrend {ews_cfg.detrend}")
    result: dict[str, Any] = {"series": args.series, "samples": int(res.values.shape[0]), "tau": taus}
    p_values: dict[str, dict[str, float]] = {}
    if res.values.shape[0] >= ews_cfg.window + 2 and not args.no_surrogates:
        test = surrogate_test(res.residual, config=ews_cfg, times=res.times)
        p_values = test.p
    for name in ews_cfg.indicators:
        last = res.indicators[name][np.isfinite(res.indicators[name])]
        line = f"  {name:<18} tau {taus[name]:+.4f}  last {last[-1] if last.size else math.nan:.6g}"
        for kind, p in p_values.items():
            line += f"  p[{kind}] {p[name]:.4f}"
        print(line)
    result["p"] = p_values
    _emit(args, result)
    return 0


def _cmd_calibrate(args: argparse.Namespace) -> int:
    from nagahana.statphys.ews import calibrate_alarm, save_calibrations
    from nagahana.statphys.io import load_trajectory

    cfg = _config(args)
    series = list(args.series) if args.series else list(cfg.ews_series)
    alpha = cfg.ews.target_false_alarm_rate if args.alpha is None else float(args.alpha)
    trajs = [load_trajectory(f) for f in args.files]
    per = alpha / len(series)
    cals = {}
    for name in series:
        benign = [t.values(name) for t in trajs if name in t.series]
        if not benign:
            raise SystemExit(f"no benign trajectory holds the series {name!r}")
        cals[name] = calibrate_alarm(benign, series=name, config=cfg.ews, alpha=per)
        c = cals[name]
        print(f"{name:<32} threshold {c.threshold:.6g} ({c.score}, alpha {per:.3g}, fit {c.n_fit}, "
              f"threshold samples {c.n_threshold})")
    path = save_calibrations(args.out, cals)
    print(f"wrote {path} (alarm of any of {len(series)} series: false-alarm rate <= {alpha:g} per trigger)")
    _emit(args, {k: v.to_dict() for k, v in cals.items()})
    return 0


def _cmd_alarm(args: argparse.Namespace) -> int:
    from nagahana.statphys.ews import StreamingEWS, load_calibrations
    from nagahana.statphys.io import load_trajectory

    cfg = _config(args)
    cals = load_calibrations(args.calibration)
    traj = load_trajectory(args.file)
    result: dict[str, Any] = {}
    for name, cal in cals.items():
        if name not in traj.series:
            print(f"{name}: not in {args.file}")
            continue
        stream = StreamingEWS(cfg.ews, cal)
        values, times = traj.values(name)
        steps = [stream.update(float(x), float(t)) for x, t in zip(values, times, strict=True)]
        fired = [s.time for s in steps if s.alarm]
        defined = sum(1 for s in steps if s.alarm is not None)
        result[name] = {"alarms": fired, "defined": defined}
        first = f", first at {fired[0]:.3f}" if fired else ""
        print(f"{name:<32} {len(fired)} alarm(s) over {defined} defined trigger(s){first}")
    _emit(args, result)
    return 0


def register_cli(subparsers: Any) -> argparse.ArgumentParser:
    """Add the `statphys` command and its subcommands to an argparse subparsers object."""
    p = subparsers.add_parser("statphys", help="statistical-physics readouts and early warning (D-56)")
    sub = p.add_subparsers(dest="statphys_command", required=True)

    def common(q: argparse.ArgumentParser) -> None:
        q.add_argument("--config", default=None, help="statphys YAML (default: the dataclass defaults)")
        q.add_argument("--json", default=None, help="also write the result as JSON to this path")

    q = sub.add_parser("write-config", help="write the YAML rendering of the configuration dataclasses")
    q.add_argument("path")
    common(q)
    q.set_defaults(func=_cmd_write_config)
    q = sub.add_parser("check-config", help="validate a statphys YAML against the dataclasses")
    q.add_argument("path")
    q.add_argument("--strict", action="store_true", help="also require the file to be its own rendering")
    q.set_defaults(func=_cmd_check_config)
    q = sub.add_parser("show", help="list the series of a saved trajectory or ModelOutputs file")
    q.add_argument("file")
    common(q)
    q.set_defaults(func=_cmd_show)
    q = sub.add_parser("ews", help="offline early-warning analysis of one series")
    q.add_argument("file")
    q.add_argument("--series", required=True)
    q.add_argument("--detrend", default=None, choices=DETRENDS, help="override the detrending (gaussian: retrospective)")
    q.add_argument("--surrogates", type=int, default=None, help="surrogates per family (default: config)")
    q.add_argument("--seed", type=int, default=None)
    q.add_argument("--no-surrogates", action="store_true", help="skip the significance test")
    q.add_argument("--all-triggers", action="store_true", help="include priority triggers (irregular sampling)")
    common(q)
    q.set_defaults(func=_cmd_ews)
    q = sub.add_parser("calibrate", help="alarm calibration on benign trajectories")
    q.add_argument("files", nargs="+")
    q.add_argument("--out", required=True, help="calibration JSON to write")
    q.add_argument("--series", nargs="*", default=None, help="series to calibrate (default: config ews_series)")
    q.add_argument("--alpha", type=float, default=None, help="target false-alarm rate over all series")
    common(q)
    q.set_defaults(func=_cmd_calibrate)
    q = sub.add_parser("alarm", help="replay the streaming alarm over a saved trajectory")
    q.add_argument("file")
    q.add_argument("--calibration", required=True)
    common(q)
    q.set_defaults(func=_cmd_alarm)
    return p


__all__ = ["register_cli"]
