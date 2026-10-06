"""Command line of the Verifier's feedback learning: `python -m nagahana verifier <command>` (D-21, D-65).

Commands

    write-config        write the configuration YAML generated from the dataclasses (config.py)
    ingest              admit analyst feedback (JSON Lines of feedback records) under an "ingest-feedback" command
    fit                 fit an RLHF, RLVR or RLCD candidate under a "fit-feedback-update" command
    evaluate            held-out evaluation and gates of a candidate (read-only; recorded)
    promote             apply a candidate's deltas to a checkpoint under an "update-weights" command
    apply-calibration   apply a candidate's temperatures under an "apply-calibration" command
    rollback            restore the weights a promotion replaced, under a "rollback-update" command
    verify              re-verify the ledger's hash chain (optionally against an anchored head)
    status              candidates, evaluations, live promotions and refusals of a store

Every state-changing command needs `--approver` and `--reason`: they form the HumanCommand (approver,
action, reason, time) that the gate checks and the ledger records; nothing has a default approver.
Model weights are read from and written to the project's checkpoints (training.common.load_checkpoint /
save_checkpoint); `promote` and `rollback` write a new checkpoint (`--out`) and never overwrite the input.
Situations are tensor files written by `situations.save_situations`.

Errors of the project's rules (a missing command, failed gates, a stale candidate, a tampered ledger)
print their message and return exit code 2.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from nagahana.core.errors import NagaHanaError


def _command(args: argparse.Namespace, action: str) -> Any:
    from nagahana.roles.contracts import HumanCommand

    return HumanCommand(approver=str(args.approver), action=action, reason=str(args.reason), time=time.time())


def _config(args: argparse.Namespace) -> Any:
    from nagahana.core.config import load_yaml
    from nagahana.models.verifier.config import FeedbackLearningConfig, feedback_config_from_mapping

    if getattr(args, "config", None):
        return feedback_config_from_mapping(load_yaml(args.config))
    return FeedbackLearningConfig()


def _model(args: argparse.Namespace) -> Any:
    from nagahana.models.config import preset
    from nagahana.models.nagahana import NagaHana
    from nagahana.training.common import load_checkpoint

    model = NagaHana(preset(args.preset))
    load_checkpoint(args.checkpoint, model)
    model.eval()
    return model


def _service(args: argparse.Namespace, model: Any | None) -> Any:
    from nagahana.models.config import preset
    from nagahana.models.verifier.service import FeedbackService

    cfg = _config(args)
    vcfg = model.cfg.verifier if model is not None else preset(getattr(args, "preset", None) or "L").verifier
    kwargs: dict[str, Any] = {}
    if model is not None:
        kwargs = {"forecaster": model.forecaster, "advisor": model.advisor, "verifier": model.verifier,
                  "exposure": model.exposure()}
    return FeedbackService(args.store, cfg, verifier_cfg=vcfg, **kwargs)


def _situations(path: str) -> Any:
    from nagahana.models.verifier.situations import load_situations

    return load_situations(path)


def _print(obj: Any) -> None:
    print(json.dumps(obj, indent=2, sort_keys=True, default=str))


def _cmd_write_config(args: argparse.Namespace) -> int:
    from nagahana.models.verifier.config import feedback_config_yaml

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(feedback_config_yaml(_config(args)), encoding="utf-8", newline="\n")
    print(f"wrote {out}")
    return 0


def _cmd_ingest(args: argparse.Namespace) -> int:
    from nagahana.models.verifier.feedback import event_from_record
    from nagahana.models.verifier.gate import INGEST_FEEDBACK

    events = []
    for n, line in enumerate(Path(args.events).read_text(encoding="utf-8").splitlines(), start=1):
        if line.strip():
            try:
                events.append(event_from_record(json.loads(line)))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                raise SystemExit(f"{args.events}:{n}: not a feedback record ({exc})") from exc
    svc = _service(args, None)
    idx = svc.ingest(events, _command(args, INGEST_FEEDBACK))
    _print({"admitted": len(idx), "indices": idx, "head": svc.ledger.head})
    return 0


def _cmd_fit(args: argparse.Namespace) -> int:
    from nagahana.models.verifier.gate import FIT_FEEDBACK

    model = _model(args)
    svc = _service(args, model)
    cand = svc.fit(args.method, _command(args, FIT_FEEDBACK), situations=_situations(args.situations), target=args.target)
    _print({"candidate_id": cand.candidate_id, "method": cand.method, "target": cand.target, "used": len(cand.report.used),
            "excluded": len(cand.report.excluded), "stats": dict(cand.report.stats),
            "temperatures": None if cand.temperatures is None else dict(cand.temperatures.temperatures)})
    return 0


def _temperature_state(path: str | None) -> Any:
    from nagahana.models.verifier.reports import TemperatureState

    if not path:
        return TemperatureState()
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return TemperatureState(temperatures={str(k): float(v) for k, v in data["temperatures"].items()})


def _cmd_evaluate(args: argparse.Namespace) -> int:
    model = _model(args)
    svc = _service(args, model)
    rep = svc.evaluate(args.candidate, situations=_situations(args.situations), temperatures=_temperature_state(args.temperatures))
    _print(rep.to_record() | {"evaluation_id": rep.evaluation_id})
    return 0


def _cmd_promote(args: argparse.Namespace) -> int:
    from nagahana.models.verifier.gate import UPDATE_WEIGHTS
    from nagahana.training.common import save_checkpoint

    model = _model(args)
    svc = _service(args, model)
    pid = svc.promote(args.candidate, _command(args, UPDATE_WEIGHTS), accept_failed_gates=bool(args.accept_failed_gates))
    header = save_checkpoint(args.out, model, stage="verifier-promotion", step=0)
    _print({"promotion_id": pid, "checkpoint": str(args.out), "model_hash": header["model_hash"]})
    return 0


def _cmd_apply(args: argparse.Namespace) -> int:
    from nagahana.models.verifier.canonical import canonical_json
    from nagahana.models.verifier.gate import APPLY_CALIBRATION

    svc = _service(args, None)
    state = _temperature_state(args.temperatures_in)
    new = svc.apply_temperatures(args.candidate, _command(args, APPLY_CALIBRATION), state=state,
                                 accept_failed_gates=bool(args.accept_failed_gates))
    out = Path(args.temperatures_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(canonical_json({"temperatures": dict(new.temperatures)}) + "\n", encoding="utf-8", newline="\n")
    _print({"temperatures": dict(new.temperatures), "written": str(out)})
    return 0


def _cmd_rollback(args: argparse.Namespace) -> int:
    from nagahana.models.verifier.gate import ROLLBACK_UPDATE
    from nagahana.training.common import save_checkpoint

    model = _model(args)
    svc = _service(args, model)
    restored = svc.rollback(args.promotion, _command(args, ROLLBACK_UPDATE))
    header = save_checkpoint(args.out, model, stage="verifier-rollback", step=0)
    _print({"restored_target_hash": restored, "checkpoint": str(args.out), "model_hash": header["model_hash"]})
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    from nagahana.models.verifier.ledger import FeedbackLedger

    ledger = FeedbackLedger(Path(args.store) / "ledger.jsonl", fsync=False, expected_head=args.head)
    ledger.verify()
    _print({"records": len(ledger), "head": ledger.head, "verified": True})
    return 0


def _cmd_status(args: argparse.Namespace) -> int:
    _print(_service(args, None).status())
    return 0


def _guard(fn: Any) -> Any:
    # Project-rule errors become a printed message and exit code 2 (the main CLI prints the same way).
    def run(args: argparse.Namespace) -> int:
        try:
            return int(fn(args))
        except NagaHanaError as exc:
            print(f"refused: {exc}")
            return 2

    return run


def _approval(p: argparse.ArgumentParser) -> None:
    p.add_argument("--approver", required=True, help="who authorises this change (recorded in the ledger)")
    p.add_argument("--reason", required=True, help="why (recorded in the ledger)")


def _model_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--preset", required=True, choices=("L", "tiny"))
    p.add_argument("--checkpoint", required=True, help="the deployed model checkpoint")


def register_cli(subparsers: Any) -> argparse.ArgumentParser:
    """Add the `verifier` command and its subcommands to an argparse subparsers object (no heavy imports here)."""
    top = subparsers.add_parser("verifier", help="the Verifier's feedback learning: RLHF, RLVR, RLCD under the human gate")
    sub = top.add_subparsers(dest="verifier_command", required=True)
    w = sub.add_parser("write-config", help="write the feedback-learning configuration YAML from the dataclasses")
    w.add_argument("--out", required=True)
    w.add_argument("--config", default=None, help="an override file to render instead of the defaults")
    w.set_defaults(func=_guard(_cmd_write_config))
    i = sub.add_parser("ingest", help="admit analyst feedback into the ledger")
    i.add_argument("--store", required=True)
    i.add_argument("--events", required=True, help="JSON Lines of feedback records (feedback.event_to_record)")
    _approval(i)
    i.set_defaults(func=_guard(_cmd_ingest))
    f = sub.add_parser("fit", help="fit a candidate update")
    f.add_argument("--store", required=True)
    f.add_argument("--method", required=True, choices=("rlhf", "rlvr", "rlcd"))
    f.add_argument("--target", default=None, choices=("forecaster", "advisor", "verifier"), help="RLHF: the policy to train")
    f.add_argument("--situations", required=True)
    f.add_argument("--config", default=None)
    _model_args(f)
    _approval(f)
    f.set_defaults(func=_guard(_cmd_fit))
    e = sub.add_parser("evaluate", help="held-out evaluation and gates of a candidate")
    e.add_argument("--store", required=True)
    e.add_argument("--candidate", required=True)
    e.add_argument("--situations", required=True)
    e.add_argument("--temperatures", default=None, help="JSON with the temperatures in force (default: all 1)")
    e.add_argument("--config", default=None)
    _model_args(e)
    e.set_defaults(func=_guard(_cmd_evaluate))
    p = sub.add_parser("promote", help="apply a candidate's deltas and write a new checkpoint")
    p.add_argument("--store", required=True)
    p.add_argument("--candidate", required=True)
    p.add_argument("--out", required=True, help="the promoted checkpoint (never the input)")
    p.add_argument("--accept-failed-gates", action="store_true", help="promote although gates failed (recorded)")
    p.add_argument("--config", default=None)
    _model_args(p)
    _approval(p)
    p.set_defaults(func=_guard(_cmd_promote))
    a = sub.add_parser("apply-calibration", help="apply a candidate's temperatures")
    a.add_argument("--store", required=True)
    a.add_argument("--candidate", required=True)
    a.add_argument("--temperatures-in", default=None, help="JSON with the temperatures in force (default: all 1)")
    a.add_argument("--temperatures-out", required=True)
    a.add_argument("--accept-failed-gates", action="store_true")
    a.add_argument("--preset", default="L", choices=("L", "tiny"), help="the configuration whose temperature interval applies")
    a.add_argument("--config", default=None)
    _approval(a)
    a.set_defaults(func=_guard(_cmd_apply))
    r = sub.add_parser("rollback", help="restore the weights a promotion replaced and write a new checkpoint")
    r.add_argument("--store", required=True)
    r.add_argument("--promotion", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--config", default=None)
    _model_args(r)
    _approval(r)
    r.set_defaults(func=_guard(_cmd_rollback))
    v = sub.add_parser("verify", help="re-verify the ledger's hash chain")
    v.add_argument("--store", required=True)
    v.add_argument("--head", default=None, help="an anchored head the ledger must end at")
    v.set_defaults(func=_guard(_cmd_verify))
    s = sub.add_parser("status", help="candidates, evaluations, promotions and refusals of a store")
    s.add_argument("--store", required=True)
    s.add_argument("--config", default=None)
    s.set_defaults(func=_guard(_cmd_status))
    return top


def main(argv: Sequence[str] | None = None) -> int:
    """Run the `verifier` commands on their own (the same parser `register_cli` adds to the main CLI)."""
    parser = argparse.ArgumentParser(prog="nagahana-verifier")
    sub = parser.add_subparsers(dest="command", required=True)
    register_cli(sub)
    args = parser.parse_args(list(argv) if argv is not None else None)
    return int(args.func(args))


__all__ = ["main", "register_cli"]
