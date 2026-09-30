"""The Verifier: regulator, calibrator, and memory-drift watcher (D-07, D-17, D-21; [A-16], [A-21]).

What the owner decided
----------------------
- "the verifier as always will work as a regulator, to store the knowledge of the human feedback for
  calibration of the model's working across all of its parts; which also computes the memory drift
  analysis (outcome forecast pair) which will allow us to analyze the poisoning scenarios" [A-16].
- "i just eliminated the idea of verifier giving feedback live/automatically to the forecaster cuz it
  would conduct the poisoning again, it will only do so under the supervision by a human" [A-16].
- "verifier will simply as always work as our supplied truth … it will only adjust model's weights
  based on human's supervised command to do so, otherwise it wont" [A-21].
- Human feedback is trusted and outside the threat model; telemetry is inside it (D-17, [A-18]).
- If the SOC acted on a forecast and the attack therefore did not complete, the outcome is tagged
  "responded-to": "not wrong but was responded to so no need to mark or penalize anything" [Q-38].

What this module implements (real code)
---------------------------------------
- A feedback store (step labels, outcomes, responded-to tags).
- An outcome–forecast ledger and a calibration summary that **excludes responded-to pairs**.
- `apply_update(fn, command)`: the ONLY path for model-changing operations. Without a `HumanCommand`
  it raises `HumanCommandRequired`. With one, it records who, what and why before running. There is
  deliberately no automatic path.

What stays open
---------------
- The RLCD reward mechanism: a Brier-score reward is proposal P-10. The Brier score itself is
  standard and is computed here as a *report*, not as a training reward.
- Site calibration adapters: output-only recalibration vs trainable adapters is held (D-13).
- The drift divergence D(P_t ‖ P_ref) per memory region (KL, MMD, …) is not specified yet.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

from nagahana.core.errors import HumanCommandRequired
from nagahana.roles.contracts import FeedbackKind, HumanCommand, HumanFeedback, OutcomeForecastPair

T = TypeVar("T")


@dataclass(frozen=True)
class CalibrationSummary:
    """Calibration over scored (not responded-to) outcome–forecast pairs."""

    n_scored: int
    n_responded_to: int
    brier: float | None     # mean (p − y)², None if nothing to score


class Verifier:
    """Human-gated regulator. See the module docstring."""

    def __init__(self) -> None:
        self._feedback: list[HumanFeedback] = []
        self._ledger: list[OutcomeForecastPair] = []
        self._commands: list[HumanCommand] = []

    # ------------------------------------------------------------------ feedback (supplied truth)
    def record_feedback(self, feedback: HumanFeedback) -> None:
        """Store one piece of analyst feedback."""
        self._feedback.append(feedback)

    def feedback(self, kind: FeedbackKind | None = None) -> tuple[HumanFeedback, ...]:
        """Stored feedback, optionally filtered by kind."""
        return tuple(f for f in self._feedback if kind is None or f.kind is kind)

    # ------------------------------------------------------------- outcome–forecast ledger (Monitor)
    def record_outcome(self, pair: OutcomeForecastPair) -> None:
        """Add one outcome–forecast pair to the ledger (the Monitor's raw material)."""
        self._ledger.append(pair)

    def calibration_summary(self) -> CalibrationSummary:
        """Brier score over pairs nobody acted on. Responded-to pairs are excluded, not penalised [Q-38]."""
        scored = [p for p in self._ledger if not p.responded_to]
        responded = len(self._ledger) - len(scored)
        if not scored:
            return CalibrationSummary(0, responded, None)
        brier = sum((p.predicted - float(p.occurred)) ** 2 for p in scored) / len(scored)
        return CalibrationSummary(len(scored), responded, brier)

    # ------------------------------------------------------------ the only model-changing path
    def apply_update(self, update: Callable[[], T], command: HumanCommand | None) -> T:
        """Run a model-changing operation only under an explicit human command (D-21).

        `update` is the operation (e.g. fit a calibration map, update a site adapter). It is never
        run without `command`, and every run is recorded with the command that authorised it.
        """
        if command is None:
            raise HumanCommandRequired(
                "The Verifier changes the model only on a human's supervised command [A-21]; "
                "pass a HumanCommand (approver, action, reason)."
            )
        self._commands.append(command)
        return update()

    def command_log(self) -> tuple[HumanCommand, ...]:
        """Every command that authorised a change, in order (audit trail)."""
        return tuple(self._commands)
