"""The Verifier role: regulator, calibrator and memory-drift watcher (D-07, D-17, D-21, D-45, D-65).

What the role is
----------------
- It holds human feedback (step labels on imagined steps, outcome labels, "responded-to" tags) as supplied
  truth: human feedback is trusted and outside the threat model, telemetry is inside it (D-17).
- It keeps the outcome-forecast ledger, the raw material of calibration and of the drift analysis of the Monitor
  region, which makes poisoning visible.
- It changes the model only on a human's supervised command (D-21); an automatic feedback path would let
  poisoned telemetry steer the model.
- An outcome that did not happen because a team acted on the forecast is tagged responded-to and is excluded
  from scoring, not penalised: a forecast that changed the future is not wrong.

What this module provides, and what it delegates
------------------------------------------------
`Verifier` is the role's ledger and gate. The mathematics lives in the working Verifier (`models/verifier`): the
Brier reward over ledger pairs, responded-to pairs excluded (`rlcd.rewards_from_pairs`, P-10 under AS-25), and the
command check (`gate.require_command`, also used by `apply_calibration` and `train_on_feedback`); the process-
reward model, the trust value head and the calibration policy head (D-45), the Monitor statistics and the
split-conformal thresholds are there too. `calibration_summary` and `apply_update` are thin uses of them.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

from nagahana.models.verifier.gate import require_command
from nagahana.models.verifier.rlcd import rewards_from_pairs
from nagahana.roles.contracts import FeedbackKind, HumanCommand, HumanFeedback, OutcomeForecastPair

T = TypeVar("T")


@dataclass(frozen=True)
class CalibrationSummary:
    """Calibration over the scored (not responded-to) outcome-forecast pairs."""

    n_scored: int
    n_responded_to: int
    brier: float | None     # mean (p - y)^2, None if nothing to score


class Verifier:
    """Human-gated regulator. See the module docstring."""

    def __init__(self) -> None:
        self._feedback: list[HumanFeedback] = []
        self._ledger: list[OutcomeForecastPair] = []
        self._commands: list[HumanCommand] = []

    def record_feedback(self, feedback: HumanFeedback) -> None:
        """Store one piece of analyst feedback (supplied truth)."""
        self._feedback.append(feedback)

    def feedback(self, kind: FeedbackKind | None = None) -> tuple[HumanFeedback, ...]:
        """Stored feedback, optionally filtered by kind."""
        return tuple(f for f in self._feedback if kind is None or f.kind is kind)

    def record_outcome(self, pair: OutcomeForecastPair) -> None:
        """Add one outcome-forecast pair to the ledger (the Monitor's raw material)."""
        self._ledger.append(pair)

    def calibration_summary(self) -> CalibrationSummary:
        """Brier score over the pairs nobody acted on; responded-to pairs are excluded, not penalised."""
        n = len(self._ledger)
        if n == 0:
            return CalibrationSummary(0, 0, None)
        reward, scored = rewards_from_pairs(self._ledger)                       # r = -(p - y)^2 on scored pairs
        n_scored = int(scored.sum())
        brier = float(-(reward[scored]).mean()) if n_scored else None
        return CalibrationSummary(n_scored, n - n_scored, brier)

    def apply_update(self, update: Callable[[], T], command: HumanCommand | None, *, action: str | None = None) -> T:
        """Run a model-changing operation only under an explicit human command (D-21).

        `update` is the operation (fit a calibration map, update a site adapter, ...); `action` is the action it
        performs, by default the one the command names. It never runs without a valid command for that action
        (`gate.require_command` raises `HumanCommandRequired` first), and every run is recorded with the command
        that authorised it.
        """
        cmd = require_command(command, action if action is not None else getattr(command, "action", ""))
        self._commands.append(cmd)
        return update()

    def command_log(self) -> tuple[HumanCommand, ...]:
        """Every command that authorised a change, in order (audit trail)."""
        return tuple(self._commands)
