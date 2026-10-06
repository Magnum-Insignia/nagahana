"""The Verifier's feedback workflow under the human gate: ingest, fit, evaluate, promote, apply, roll back (D-21, D-65).

    ingest     HumanCommand "ingest-feedback"      stamp events with the command id, append them to the ledger
    fit        HumanCommand "fit-feedback-update"  RLHF, RLVR or RLCD on the ledger feedback minus the
                                                   held-out share; the candidate is stored beside the ledger
    evaluate   no command (read-only)              held-out evaluation and gates (evaluation.py)
    promote    HumanCommand "update-weights"       theta <- theta + Delta for an evaluated candidate whose
                                                   reference hash equals the live weights; snapshot stored
    apply      HumanCommand "apply-calibration"    the candidate's temperatures (gate.apply_calibration)
    rollback   HumanCommand "rollback-update"      restore the snapshot of the latest promotion, bit for bit

Every step is recorded in the ledger (ledger.py) with the command that authorised it; an attempt the gate
refuses (no command, a command for another action, a command already used, a stale candidate, failed
gates not accepted, a rollback out of order) is recorded as "audit.refused" and raises. Nothing changes
before every check has passed, so a refused call leaves weights, temperatures and files exactly as they
were (tested).

Promotion (AS-844). The candidate applies only to the weights it was fitted against: the module's
state hash must equal the candidate's reference hash. The touched tensors are cloned exactly before the
float32 addition, stored (a tensor file whose SHA-256 the promote record holds) and the state hashes
before and after are recorded. A failed evaluation gate blocks the promotion unless the call states
`accept_failed_gates=True`, which the record lists with the failed gates (AS-845).

Rollback (AS-844). Only the latest promotion of a module that is not yet rolled back can be undone
(last in, first out), and only while the module's state hash equals that promotion's post-hash (nothing
else changed the weights since). The stored tensors are copied back and the state hash must then equal
the recorded pre-hash: a bit-identical restore, checked, not assumed.

Files under the store root: ledger.jsonl, candidates/<candidate id>.pt, snapshots/<promotion id>.pt.
Without a root everything stays in memory (in-loop use and tests); the ledger rules are the same.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import nn

from nagahana.core.errors import HumanCommandRequired, InvariantViolation
from nagahana.models.advisor.model import Advisor
from nagahana.models.config.components import VerifierConfig
from nagahana.models.forecaster.model import Forecaster
from nagahana.models.verifier.candidates import CandidateUpdate, load_candidate, save_candidate
from nagahana.models.verifier.canonical import digest_of
from nagahana.models.verifier.config import FeedbackLearningConfig
from nagahana.models.verifier.evaluation import EvaluationReport, evaluate_candidate
from nagahana.models.verifier.feedback import FeedbackEvent, event_id, stamp
from nagahana.models.verifier.gate import (
    APPLY_CALIBRATION,
    FIT_FEEDBACK,
    INGEST_FEEDBACK,
    ROLLBACK_UPDATE,
    UPDATE_WEIGHTS,
    apply_calibration,
    require_command,
)
from nagahana.models.verifier.learning import FeedbackBatch, split_batch
from nagahana.models.verifier.ledger import FeedbackLedger, LedgerRecord
from nagahana.models.verifier.model import VerifierNet
from nagahana.models.verifier.params import apply_deltas_, load_tensors, restore_, save_tensors, state_hash
from nagahana.models.verifier.reports import TemperatureState
from nagahana.models.verifier.rlcd import RLCDLearner
from nagahana.models.verifier.rlhf import RLHFLearner
from nagahana.models.verifier.rlvr import RLVRLearner
from nagahana.models.verifier.situations import Situation
from nagahana.roles.contracts import HumanCommand

MarginalEnergy = Callable[[torch.Tensor], torch.Tensor]


class FeedbackService:
    """The feedback workflow of one deployment (module docstring).

    Parameters
    ----------
    root:
        Store directory (ledger, candidates, snapshots), or None to keep everything in memory.
    cfg:
        Feedback-learning configuration.
    verifier_cfg:
        The Verifier's configuration (temperature interval, Monitor thresholds).
    forecaster, advisor, verifier:
        The deployed modules a candidate may target (each optional; a step that needs a missing one raises).
    exposure:
        TAAFT's marginal-energy callable for held-out imagination, as the deployed Forecaster uses it.
    """

    def __init__(self, root: str | Path | None, cfg: FeedbackLearningConfig, *, verifier_cfg: VerifierConfig,
                 forecaster: Forecaster | None = None, advisor: Advisor | None = None, verifier: VerifierNet | None = None,
                 exposure: MarginalEnergy | None = None, expected_head: str | None = None) -> None:
        self.root = None if root is None else Path(root)
        self.cfg, self.verifier_cfg = cfg, verifier_cfg
        self.forecaster, self.advisor, self.verifier, self.exposure = forecaster, advisor, verifier, exposure
        ledger_path = None if self.root is None else self.root / "ledger.jsonl"
        self.ledger = FeedbackLedger(ledger_path, fsync=cfg.ledger.fsync, expected_head=expected_head)
        self._candidates: dict[str, CandidateUpdate] = {}
        self._snapshots: dict[str, dict[str, torch.Tensor]] = {}

    # ---- the gate and refusals
    def _refuse(self, step: str, action: str, reason: str, command: object | None) -> None:
        body: dict[str, Any] = {"step": step, "action": action, "reason": reason}
        if isinstance(command, HumanCommand):
            body |= {"approver": command.approver, "command_action": command.action}
        self.ledger.append("audit.refused", body)

    def _authorise(self, command: object, action: str, step: str) -> tuple[HumanCommand, str]:
        try:
            cmd = require_command(command, action)
        except HumanCommandRequired as exc:
            self._refuse(step, action, str(exc), command)
            raise
        cid = self.ledger.register_command(cmd)
        try:
            self.ledger.assert_unused(cid)
        except HumanCommandRequired as exc:
            self._refuse(step, action, str(exc), cmd)
            raise
        return cmd, cid

    def _module(self, target: str) -> nn.Module:
        m = {"forecaster": self.forecaster, "advisor": self.advisor, "verifier": self.verifier}.get(target)
        if m is None:
            raise InvariantViolation(f"no deployed {target} was given to the feedback service")
        return m

    # ---- ingest
    def ingest(self, events: Sequence[FeedbackEvent], command: HumanCommand | None) -> list[int]:
        """Admit feedback under an "ingest-feedback" command; returns the events' ledger indices."""
        if not events:
            raise InvariantViolation("nothing to ingest")
        cmd, cid = self._authorise(command, INGEST_FEEDBACK, "ingest")
        stamped = [stamp(e, cid) for e in events]
        ids = [event_id(e) for e in stamped]
        dup = sorted({i for i in ids if ids.count(i) > 1} | {i for i in ids if self.ledger.has_event(i)})
        if dup:
            self._refuse("ingest", INGEST_FEEDBACK, f"{len(dup)} event(s) already admitted or repeated in the batch", cmd)
            raise InvariantViolation(f"feedback events already in the ledger or repeated: {[d[:12] for d in dup]}")
        indices = self.ledger.admit(stamped)
        self.ledger.append("audit.ingest", {"command_id": cid, "indices": indices, "event_ids": ids})
        return indices

    # ---- batches
    def batch(self, situations: Mapping[str, Situation], *, kinds: Iterable[str] | None = None,
              indices: Iterable[int] | None = None) -> FeedbackBatch:
        """Ledger feedback (optionally of some kinds or some record indices) with the given situations."""
        events = self.ledger.feedback(kinds)
        if indices is not None:
            wanted = set(int(i) for i in indices)
            events = [(i, e) for i, e in events if i in wanted]
        return FeedbackBatch(tuple(events), dict(situations), self.ledger.head)

    # ---- fit
    def learner(self, method: str, target: str | None = None) -> RLHFLearner | RLVRLearner | RLCDLearner:
        """The learner of a method (and, for RLHF, its policy target)."""
        if method == "rlhf":
            if target not in ("forecaster", "advisor"):
                raise InvariantViolation("RLHF needs target 'forecaster' or 'advisor'")
            return RLHFLearner(self.cfg, target=target, module=self._module(target))
        if method == "rlvr":
            if target not in (None, "forecaster"):
                raise InvariantViolation("RLVR trains the Forecaster")
            fc = self._module("forecaster")
            assert isinstance(fc, Forecaster)
            return RLVRLearner(self.cfg, forecaster=fc)
        if method == "rlcd":
            if target not in (None, "verifier"):
                raise InvariantViolation("RLCD trains the Verifier's calibration")
            v = self._module("verifier")
            assert isinstance(v, VerifierNet)
            return RLCDLearner(self.cfg, verifier=v, verifier_cfg=self.verifier_cfg)
        raise InvariantViolation(f"unknown method {method!r}; methods: rlhf, rlvr, rlcd")

    def fit(self, method: str, command: HumanCommand | None, *, situations: Mapping[str, Situation],
            target: str | None = None) -> CandidateUpdate:
        """Fit a candidate on the ledger feedback minus its held-out share, under a "fit-feedback-update" command."""
        cmd, cid = self._authorise(command, FIT_FEEDBACK, "fit")
        learner = self.learner(method, target)
        full = self.batch(situations)
        fit_part, held = split_batch(full, self.cfg.evaluation.held_out_fraction)
        try:
            cand = learner.fit(fit_part)
        except InvariantViolation as exc:
            self._refuse("fit", FIT_FEEDBACK, f"{method}: {exc}", cmd)
            raise
        file, sha = self._store_candidate(cand)
        self.ledger.append("audit.fit", {
            "command_id": cid, "candidate_id": cand.candidate_id, "method": cand.method, "target": cand.target,
            "reference_hash": cand.reference_hash, "file": file, "sha256": sha, "used": list(cand.report.used),
            "held_out": [i for i, _ in held.events], "fit_events": [i for i, _ in fit_part.events],
            "excluded": len(cand.report.excluded), "ledger_head": full.ledger_head, "approver": cmd.approver,
        })
        return cand

    def _store_candidate(self, cand: CandidateUpdate) -> tuple[str, str]:
        self._candidates[cand.candidate_id] = cand
        if self.root is None:
            return "", ""
        rel = f"candidates/{cand.candidate_id}.pt"
        return rel, save_candidate(self.root / rel, cand)

    def fit_record(self, candidate_id: str) -> LedgerRecord:
        """The audit.fit record of a candidate."""
        for rec in self.ledger.records("audit.fit"):
            if rec.body["candidate_id"] == candidate_id:
                return rec
        raise KeyError(f"no candidate {candidate_id[:12]}... in the ledger")

    def candidate(self, candidate_id: str) -> CandidateUpdate:
        """A fitted candidate, from memory or from its file (digest checked against the ledger)."""
        if candidate_id in self._candidates:
            return self._candidates[candidate_id]
        rec = self.fit_record(candidate_id)
        if self.root is None or not rec.body["file"]:
            raise InvariantViolation(f"candidate {candidate_id[:12]}... was not stored on disk")
        cand = load_candidate(self.root / str(rec.body["file"]), expected_sha256=str(rec.body["sha256"]))
        if cand.candidate_id != candidate_id:
            raise InvariantViolation("the candidate file holds another candidate")
        self._candidates[candidate_id] = cand
        return cand

    # ---- evaluate
    def evaluate(self, candidate_id: str, *, situations: Mapping[str, Situation],
                 temperatures: TemperatureState | None = None) -> EvaluationReport:
        """Held-out evaluation of a candidate on the feedback its fit held out (recorded; no command needed)."""
        cand = self.candidate(candidate_id)
        rec = self.fit_record(candidate_id)
        held = self.batch(situations, indices=[int(i) for i in rec.body["held_out"]])
        module = self._module(cand.target)
        if state_hash(module) != cand.reference_hash:
            raise InvariantViolation("the deployed weights differ from the candidate's reference; it cannot be evaluated")
        report = evaluate_candidate(cand, held, cfg=self.cfg, verifier_cfg=self.verifier_cfg, forecaster=self.forecaster,
                                    advisor=self.advisor, verifier=self.verifier, temperatures=temperatures, exposure=self.exposure)
        self.ledger.append("audit.evaluate", {"candidate_id": candidate_id, "evaluation_id": report.evaluation_id,
                                              "passed": report.passed, "failed": list(report.failed()), "report": report.to_record()})
        return report

    def latest_evaluation(self, candidate_id: str) -> LedgerRecord | None:
        """The latest audit.evaluate record of a candidate."""
        recs = [r for r in self.ledger.records("audit.evaluate") if r.body["candidate_id"] == candidate_id]
        return recs[-1] if recs else None

    def _evaluated(self, candidate_id: str, step: str, action: str, cmd: HumanCommand, accept_failed_gates: bool
                   ) -> tuple[str, list[str]]:
        # The candidate's latest evaluation must exist and pass, or the failure must be accepted explicitly.
        ev = self.latest_evaluation(candidate_id)
        if ev is None:
            self._refuse(step, action, "the candidate has not been evaluated on held-out feedback", cmd)
            raise InvariantViolation("evaluate the candidate before it is promoted or applied")
        failed = [str(x) for x in ev.body["failed"]]
        if failed and not accept_failed_gates:
            self._refuse(step, action, f"failed gates not accepted: {failed}", cmd)
            raise InvariantViolation(f"the candidate failed the gates {failed}; pass accept_failed_gates=True to accept them")
        return str(ev.body["evaluation_id"]), failed

    # ---- promote and apply
    def promote(self, candidate_id: str, command: HumanCommand | None, *, accept_failed_gates: bool = False) -> str:
        """Apply a candidate's deltas to its target under an "update-weights" command; returns the promotion id."""
        cmd, cid = self._authorise(command, UPDATE_WEIGHTS, "promote")
        cand = self.candidate(candidate_id)
        if not cand.deltas:
            self._refuse("promote", UPDATE_WEIGHTS, "the candidate has no weight deltas", cmd)
            raise InvariantViolation("the candidate proposes temperatures only: use apply_temperatures")
        module = self._module(cand.target)
        evaluation_id, failed = self._evaluated(candidate_id, "promote", UPDATE_WEIGHTS, cmd, accept_failed_gates)
        pre = state_hash(module)
        if pre != cand.reference_hash:
            self._refuse("promote", UPDATE_WEIGHTS, "stale candidate: the deployed weights differ from its reference", cmd)
            raise InvariantViolation("the deployed weights differ from the candidate's reference weights; refit it")
        snapshot = apply_deltas_(module, cand.deltas)
        post = state_hash(module)
        promotion_id = digest_of({"candidate_id": candidate_id, "command_id": cid, "pre": pre, "post": post})
        file, sha = "", ""
        if self.root is not None:
            file = f"snapshots/{promotion_id}.pt"
            sha = save_tensors(self.root / file, snapshot, {"kind": "verifier-snapshot", "promotion_id": promotion_id,
                                                             "target": cand.target, "pre_hash": pre})
        self._snapshots[promotion_id] = snapshot
        self.ledger.append("audit.promote", {
            "command_id": cid, "promotion_id": promotion_id, "candidate_id": candidate_id, "evaluation_id": evaluation_id,
            "target": cand.target, "pre_hash": pre, "post_hash": post, "snapshot_file": file, "snapshot_sha256": sha,
            "accepted_failed_gates": failed, "approver": cmd.approver, "reason": cmd.reason,
        })
        return promotion_id

    def apply_temperatures(self, candidate_id: str, command: HumanCommand | None, *, state: TemperatureState,
                           accept_failed_gates: bool = False) -> TemperatureState:
        """The candidate's temperatures under an "apply-calibration" command (gate.apply_calibration); `state` is unchanged."""
        cmd, cid = self._authorise(command, APPLY_CALIBRATION, "apply-calibration")
        cand = self.candidate(candidate_id)
        if cand.temperatures is None:
            self._refuse("apply-calibration", APPLY_CALIBRATION, "the candidate proposes no temperatures", cmd)
            raise InvariantViolation("the candidate proposes no temperatures")
        evaluation_id, failed = self._evaluated(candidate_id, "apply-calibration", APPLY_CALIBRATION, cmd, accept_failed_gates)
        new = apply_calibration(cand.temperatures, cmd, state=state)
        self.ledger.append("audit.apply-calibration", {
            "command_id": cid, "candidate_id": candidate_id, "evaluation_id": evaluation_id,
            "before": {k: float(v) for k, v in sorted(state.temperatures.items())},
            "after": {k: float(v) for k, v in sorted(new.temperatures.items())}, "accepted_failed_gates": failed,
            "approver": cmd.approver, "reason": cmd.reason,
        })
        return new

    # ---- rollback
    def promotions(self, target: str | None = None) -> list[LedgerRecord]:
        """Promotions not yet rolled back, oldest first (optionally of one target)."""
        rolled = {str(r.body["promotion_id"]) for r in self.ledger.records("audit.rollback")}
        return [r for r in self.ledger.records("audit.promote")
                if str(r.body["promotion_id"]) not in rolled and (target is None or r.body["target"] == target)]

    def rollback(self, promotion_id: str, command: HumanCommand | None) -> str:
        """Restore the weights a promotion replaced, bit for bit, under a "rollback-update" command. Returns the restored hash."""
        cmd, cid = self._authorise(command, ROLLBACK_UPDATE, "rollback")
        recs = [r for r in self.ledger.records("audit.promote") if r.body["promotion_id"] == promotion_id]
        if not recs:
            self._refuse("rollback", ROLLBACK_UPDATE, "unknown promotion", cmd)
            raise InvariantViolation(f"no promotion {promotion_id[:12]}... in the ledger")
        rec = recs[0]
        target = str(rec.body["target"])
        live = self.promotions(target)
        if not live or live[-1].body["promotion_id"] != promotion_id:
            self._refuse("rollback", ROLLBACK_UPDATE, "not the latest live promotion of its target (rollbacks are last in, first out)", cmd)
            raise InvariantViolation("only the latest promotion of a module that is not rolled back can be undone")
        module = self._module(target)
        if state_hash(module) != rec.body["post_hash"]:
            self._refuse("rollback", ROLLBACK_UPDATE, "the weights changed since the promotion", cmd)
            raise InvariantViolation("the deployed weights are not the promoted ones; nothing is restored")
        snapshot = self._snapshot(promotion_id, rec)
        restore_(module, snapshot)
        restored = state_hash(module)
        if restored != rec.body["pre_hash"]:
            raise InvariantViolation("the restored weights do not reproduce the pre-promotion state hash")
        self.ledger.append("audit.rollback", {"command_id": cid, "promotion_id": promotion_id, "target": target,
                                              "restored_hash": restored, "approver": cmd.approver, "reason": cmd.reason})
        return restored

    def _snapshot(self, promotion_id: str, rec: LedgerRecord) -> dict[str, torch.Tensor]:
        if promotion_id in self._snapshots:
            return self._snapshots[promotion_id]
        if self.root is None or not rec.body["snapshot_file"]:
            raise InvariantViolation("the promotion's snapshot is not available")
        tensors, meta = load_tensors(self.root / str(rec.body["snapshot_file"]), expected_sha256=str(rec.body["snapshot_sha256"]))
        if meta.get("promotion_id") != promotion_id:
            raise InvariantViolation("the snapshot file belongs to another promotion")
        return tensors

    # ---- overview
    def status(self) -> dict[str, Any]:
        """Counts and the live state of the workflow (for the command line and reports)."""
        evals = {str(r.body["candidate_id"]): (bool(r.body["passed"]), list(r.body["failed"])) for r in self.ledger.records("audit.evaluate")}
        cands = [{"candidate_id": str(r.body["candidate_id"]), "method": str(r.body["method"]), "target": str(r.body["target"]),
                  "evaluated": str(r.body["candidate_id"]) in evals,
                  "passed": evals.get(str(r.body["candidate_id"]), (math.nan, []))[0]} for r in self.ledger.records("audit.fit")]
        return {"records": len(self.ledger), "head": self.ledger.head, "feedback": len(self.ledger.records("feedback")),
                "candidates": cands, "live_promotions": [str(r.body["promotion_id"]) for r in self.promotions()],
                "refusals": len(self.ledger.records("audit.refused"))}


__all__ = ["FeedbackService"]
