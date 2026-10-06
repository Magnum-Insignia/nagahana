"""Candidate updates: the one output of RLHF, RLVR and RLCD, kept apart from the deployed state (D-21, D-65; AS-835).

A `CandidateUpdate` holds what a fit proposes and everything needed to audit it:

    method          "rlhf", "rlvr" or "rlcd"
    target          "forecaster", "advisor" or "verifier": the module whose weights the deltas change
    reference_hash  state hash (params.state_hash) of that module when the fit started: the candidate
                    applies to exactly these weights and is refused against any other
    deltas          name -> Delta tensor in the weights' dtype (possibly empty for a temperature-only RLCD)
    temperatures    a CalibrationProposal (RLCD) or None
    reward_model    the Bradley-Terry reward model fitted beside DPO (RLHF) or None
    report          FitReport: the ledger records used, every exclusion with its reason, the loss curve,
                    summary statistics, the configuration digest, the ledger head, the seed

`candidate_id` is the SHA-256 of the canonical JSON of (method, target, reference hash, the deltas'
tensor digest, temperatures, reward model, report), so a candidate is addressed by its content and the
ledger's audit records name it unambiguously. `save_candidate` writes one tensor file
(params.save_tensors) whose SHA-256 the ledger records; `load_candidate` checks that digest before
decoding and re-derives the candidate id.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from nagahana.core.errors import InvariantViolation
from nagahana.models.verifier.bradley_terry import BradleyTerryModel
from nagahana.models.verifier.canonical import digest_of
from nagahana.models.verifier.learning import Exclusion
from nagahana.models.verifier.params import load_tensors, save_tensors, tensor_digest
from nagahana.models.verifier.reports import CalibrationProposal

METHODS: tuple[str, ...] = ("rlhf", "rlvr", "rlcd")
TARGETS: tuple[str, ...] = ("forecaster", "advisor", "verifier")


@dataclass(frozen=True)
class FitReport:
    """What a fit read, what it excluded and why, and how the optimisation went."""

    method: str
    target: str
    used: tuple[int, ...]
    excluded: tuple[Exclusion, ...]
    losses: tuple[float, ...]
    stats: Mapping[str, float]
    config_digest: str
    ledger_head: str
    seed: int

    def to_record(self) -> dict[str, Any]:
        return {"method": self.method, "target": self.target, "used": list(self.used),
                "excluded": [{"index": e.index, "event_id": e.event_id, "reason": e.reason} for e in self.excluded],
                "losses": [float(x) for x in self.losses], "stats": {k: float(v) for k, v in sorted(self.stats.items())},
                "config_digest": self.config_digest, "ledger_head": self.ledger_head, "seed": int(self.seed)}

    @classmethod
    def from_record(cls, r: Mapping[str, Any]) -> FitReport:
        return cls(method=str(r["method"]), target=str(r["target"]), used=tuple(int(i) for i in r["used"]),
                   excluded=tuple(Exclusion(int(e["index"]), str(e["event_id"]), str(e["reason"])) for e in r["excluded"]),
                   losses=tuple(float(x) for x in r["losses"]), stats={str(k): float(v) for k, v in r["stats"].items()},
                   config_digest=str(r["config_digest"]), ledger_head=str(r["ledger_head"]), seed=int(r["seed"]))


def _proposal_record(p: CalibrationProposal | None) -> dict[str, Any] | None:
    if p is None:
        return None
    return {"temperatures": {k: float(v) for k, v in sorted(p.temperatures.items())}, "source": p.source,
            "n_pairs": {k: int(v) for k, v in sorted(p.n_pairs.items())}, "t_min": float(p.t_min), "t_max": float(p.t_max)}


def _proposal_from(r: Mapping[str, Any] | None) -> CalibrationProposal | None:
    if r is None:
        return None
    return CalibrationProposal(temperatures={str(k): float(v) for k, v in r["temperatures"].items()}, source=str(r["source"]),
                               n_pairs={str(k): int(v) for k, v in r["n_pairs"].items()}, t_min=float(r["t_min"]),
                               t_max=float(r["t_max"]))


@dataclass(frozen=True)
class CandidateUpdate:
    """A proposed update, kept apart from the deployed weights and temperatures (module docstring)."""

    method: str
    target: str
    reference_hash: str
    deltas: Mapping[str, torch.Tensor] = field(repr=False)
    temperatures: CalibrationProposal | None
    reward_model: BradleyTerryModel | None
    report: FitReport

    def __post_init__(self) -> None:
        if self.method not in METHODS:
            raise InvariantViolation(f"unknown method {self.method!r}; methods: {list(METHODS)}")
        if self.target not in TARGETS:
            raise InvariantViolation(f"unknown target {self.target!r}; targets: {list(TARGETS)}")
        if len(self.reference_hash) != 64:
            raise InvariantViolation("reference_hash must be a SHA-256 hex digest (params.state_hash)")
        if not self.deltas and self.temperatures is None:
            raise InvariantViolation("a candidate proposes deltas, temperatures, or both")
        for name, d in self.deltas.items():
            if not bool(torch.isfinite(d).all()):
                raise InvariantViolation(f"delta {name!r} holds non-finite values")

    def metadata(self) -> dict[str, Any]:
        """Everything but the tensors, plus the tensors' digest (the content the id commits to)."""
        return {"method": self.method, "target": self.target, "reference_hash": self.reference_hash,
                "deltas_digest": tensor_digest(self.deltas), "delta_names": sorted(self.deltas),
                "temperatures": _proposal_record(self.temperatures),
                "reward_model": None if self.reward_model is None else self.reward_model.to_record(),
                "report": self.report.to_record()}

    @property
    def candidate_id(self) -> str:
        """SHA-256 of `metadata()` (content address)."""
        return digest_of(self.metadata())

    def delta_norm(self) -> float:
        """sqrt(sum_n ||Delta_n||_F^2) in float64."""
        return float(sum(float((d.double() ** 2).sum()) for d in self.deltas.values()) ** 0.5)


def save_candidate(path: str | Path, candidate: CandidateUpdate) -> str:
    """Write the candidate (deltas + metadata) atomically; returns the file's SHA-256 for the ledger."""
    meta = candidate.metadata() | {"kind": "verifier-candidate", "candidate_id": candidate.candidate_id}
    return save_tensors(path, dict(candidate.deltas), meta)


def load_candidate(path: str | Path, *, expected_sha256: str | None) -> CandidateUpdate:
    """Read a candidate file, checking the file digest and the candidate id it was saved under."""
    tensors, meta = load_tensors(path, expected_sha256=expected_sha256)
    if meta.get("kind") != "verifier-candidate":
        raise InvariantViolation(f"{path}: not a candidate file")
    c = CandidateUpdate(method=str(meta["method"]), target=str(meta["target"]), reference_hash=str(meta["reference_hash"]),
                        deltas=tensors, temperatures=_proposal_from(meta["temperatures"]),
                        reward_model=None if meta["reward_model"] is None else BradleyTerryModel.from_record(meta["reward_model"]),
                        report=FitReport.from_record(meta["report"]))
    if c.candidate_id != meta["candidate_id"]:
        raise InvariantViolation(f"{path}: content does not match the candidate id it was saved under")
    return c


__all__ = ["METHODS", "TARGETS", "CandidateUpdate", "FitReport", "load_candidate", "save_candidate"]
