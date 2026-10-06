"""Command line: `python -m nagahana <command>`.

Commands
--------
- `decisions [--write-docs]`: decided / held / proposed, with the code that depends on each.
- `stages`: the six-stage training pipeline (pipeline/stages.py).
- `metrics`: the evaluation catalogue (evaluation/catalogue.py).
- `access`: the memory access matrix (memory/access.py).
- `check-config FILE`: list every `???` (undecided) key in a YAML config.
- `params [--preset L]`: the model's parameter count per component (meta-device build, no memory).
- `train --stage {3,4,5} --pcap FILE --labeller NAME …`: one training stage on a labelled capture, in
  stream order with the TSTCT carry (D-51); weights in and out as checkpoints (model hash, P-18).
- `replay --pcap FILE --checkpoint CKPT …`: forensic replay of a capture (RunMode.FORENSIC_REPLAY),
  printing the `ForensicReport`. An untrained model is refused unless `--untrained` says so.

The site MTU (`--mtu`) is required wherever the physics term is used: it is never defaulted.

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


def _utf8_stdout() -> None:
    # Reports contain arrows and subscripts; a legacy console code page must not crash the run.
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure is not None:
        reconfigure(errors="replace")


def _model_and_cfg(args: argparse.Namespace) -> tuple[object, object]:
    # Lazy imports: the governance commands above must run without torch-heavy modules loaded.
    import dataclasses

    from nagahana.models.config import preset
    from nagahana.models.nagahana import NagaHana

    cfg = preset(args.preset)
    if getattr(args, "mtu", None) is not None:
        cfg = dataclasses.replace(cfg, generator=dataclasses.replace(cfg.generator, mtu=float(args.mtu)))
    return NagaHana(cfg), cfg


def _cmd_params(args: argparse.Namespace) -> int:
    from nagahana.models.config import preset
    from nagahana.models.nagahana import count_parameters

    counts = count_parameters(preset(args.preset))
    for k, v in counts.items():
        print(f"{k:<12}{v:>16,d}")
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    import time

    from nagahana.models.nagahana import NagaHana, physics_term
    from nagahana.roles.contracts import HumanCommand
    from nagahana.training.common import load_checkpoint, save_checkpoint
    from nagahana.training.runs import load_capture, run_stage, stream_segments

    _utf8_stdout()
    model, cfg = _model_and_cfg(args)
    assert isinstance(model, NagaHana)
    if args.checkpoint_in:
        load_checkpoint(args.checkpoint_in, model)
    src = load_capture(args.pcap, labeller=args.labeller, network=args.network, sandboxed=args.sandboxed)
    segs = stream_segments(src, model.cfg, segment_seconds=args.segment_seconds)
    physics = physics_term(model.cfg) if model.cfg.generator.mtu is not None else None
    command = (HumanCommand(args.approver, "update-weights", args.reason, time.time())
               if args.approver and args.reason else None)
    run = run_stage(args.stage, model, [src], segs, steps=args.steps, seed=args.seed, precision=args.precision,
                    lanes=args.lanes, physics=physics, verifier_command=command, joint_after=args.joint_after,
                    perturb_seed=args.perturb_seed)
    for i, h in enumerate(run.history):
        keys = sorted(k for k in h if k.endswith("total"))
        print(i, {k: round(h[k], 4) for k in keys})
    for n in run.notes:
        print("note:", n)
    if args.checkpoint_out:
        header = save_checkpoint(args.checkpoint_out, model, stage=f"stage{args.stage}", step=len(run.history))
        print("saved", args.checkpoint_out, header["model_hash"][:16])
    del cfg
    return 0


def _cmd_replay(args: argparse.Namespace) -> int:
    from nagahana.inference.engine import EngineSettings
    from nagahana.inference.forensic import replay
    from nagahana.models.nagahana import NagaHana, physics_term
    from nagahana.training.common import load_checkpoint

    _utf8_stdout()
    model, _cfg = _model_and_cfg(args)
    assert isinstance(model, NagaHana)
    if args.checkpoint:
        load_checkpoint(args.checkpoint, model)
    elif not args.untrained:
        print("refusing to replay with untrained weights: pass --checkpoint, or --untrained to say so explicitly")
        return 2
    physics = physics_term(model.cfg) if model.cfg.generator.mtu is not None else None
    settings = EngineSettings.assumed(model, network=args.network)
    report, engine = replay(model, args.pcap, settings=settings, physics=physics, seed=args.seed, sandboxed=args.sandboxed)
    for _t, line in report.timeline:
        print(line)
    for stage, text, _t in report.narrative:
        print(f"[{stage.value}] {text}")
    print("patient zero (belief):", ", ".join(f"{n} {p:.2f}" for n, p in report.patient_zero))
    for line in (*report.counterfactuals, *report.observability_gaps, *report.tamper_signs):
        print("-", line)
    print(f"{len(engine.results)} triggers, {len(engine.log)} state updates")
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
    pp = sub.add_parser("params", help="parameter count per component (meta device)")
    pp.add_argument("--preset", default="L", choices=("L", "tiny"))
    pp.set_defaults(func=_cmd_params)
    for name, fn, hlp in (("train", _cmd_train, "one training stage on a labelled capture"),
                          ("replay", _cmd_replay, "forensic replay of a capture")):
        q = sub.add_parser(name, help=hlp)
        q.add_argument("--pcap", required=True)
        q.add_argument("--preset", required=True, choices=("L", "tiny"))
        q.add_argument("--mtu", type=float, default=None, help="site MTU in bytes (required for the physics term)")
        q.add_argument("--network", required=True, help="name of the monitored network")
        q.add_argument("--seed", type=int, required=True)
        sb = q.add_mutually_exclusive_group(required=True)
        sb.add_argument("--sandboxed", dest="sandboxed", action="store_true", help="the parser runs in a sandbox")
        sb.add_argument("--not-sandboxed", dest="sandboxed", action="store_false")
        q.set_defaults(func=fn)
        if name == "train":
            q.add_argument("--stage", type=int, required=True, choices=(3, 4, 5))
            q.add_argument("--labeller", required=True)
            q.add_argument("--steps", type=int, required=True)
            q.add_argument("--lanes", type=int, required=True)
            q.add_argument("--segment-seconds", type=float, required=True)
            q.add_argument("--precision", required=True, choices=("bf16", "fp32"))
            q.add_argument("--joint-after", type=int, default=None, help="stage 5: optimiser steps before the joint phase")
            q.add_argument("--perturb-seed", type=int, default=None,
                           help="stage 3: out-of-order augmentation inside the recorded uncertainty (AS-320)")
            q.add_argument("--checkpoint-in", default=None)
            q.add_argument("--checkpoint-out", default=None)
            q.add_argument("--approver", default=None, help="stage 5: who authorises the Verifier update (D-21)")
            q.add_argument("--reason", default=None)
        else:
            q.add_argument("--checkpoint", default=None)
            q.add_argument("--untrained", action="store_true", help="replay with untrained weights (smoke runs only)")
    args = p.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
