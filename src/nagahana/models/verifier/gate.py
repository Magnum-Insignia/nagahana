"""The human gate: temperatures and weights change only under a valid HumanCommand (D-21, D-65; build-spec section 2.10).

"it will only adjust model's weights based on human's supervised command to do so, otherwise it wont"
[A-21]; automatic feedback was removed because "it would conduct the poisoning again" [A-16].

Rules enforced here (tested)

- `require_command(command, action)` returns the command only when it is a `roles.contracts.HumanCommand`
  whose action is exactly `action`; otherwise it raises `core.errors.HumanCommandRequired` before
  anything else runs. A command authorises the action it names and no other: a command for
  "update-site-adapter" cannot apply a calibration.
- `apply_calibration(proposal, command, state=...)` returns a new `TemperatureState` only under an
  "apply-calibration" command; the given state is never modified.
- `train_on_feedback(model, optimizer, loss_fn, command)` runs one optimisation step only under an
  "update-weights" command; without it, it raises before computing anything, so no parameter, gradient
  or optimiser state is touched.
- Every authorised change is appended to the caller's audit list (who, what, why, when).

Actions of the feedback-learning workflow (D-65; `service.FeedbackService`)

    ingest-feedback       admit analyst feedback into the hash-chained ledger
    fit-feedback-update   compute a candidate update from ledger feedback (no weight changes)
    update-weights        promote a candidate's weight deltas into the deployed weights
    apply-calibration     apply a candidate's temperature proposal
    rollback-update       restore the weights a promotion replaced, bit for bit

Command identity (AS-831): `command_id(c)` is the SHA-256 of the canonical JSON of (approver, action,
reason, time). The ledger stores every command it was shown under this id, and a command authorises at
most one state-changing operation of the feedback workflow (single use, enforced by the ledger).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torch import nn

from nagahana.core.errors import HumanCommandRequired
from nagahana.models.verifier.canonical import digest_of
from nagahana.models.verifier.reports import CalibrationProposal, TemperatureState
from nagahana.roles.contracts import HumanCommand

APPLY_CALIBRATION = "apply-calibration"
UPDATE_WEIGHTS = "update-weights"
INGEST_FEEDBACK = "ingest-feedback"
FIT_FEEDBACK = "fit-feedback-update"
ROLLBACK_UPDATE = "rollback-update"

#: Actions of the feedback-learning workflow, in workflow order.
FEEDBACK_ACTIONS: tuple[str, ...] = (INGEST_FEEDBACK, FIT_FEEDBACK, UPDATE_WEIGHTS, APPLY_CALIBRATION, ROLLBACK_UPDATE)


def require_command(command: object, action: str) -> HumanCommand:
    """Return `command` if it is a HumanCommand for `action`; raise HumanCommandRequired otherwise."""
    if not isinstance(command, HumanCommand):
        raise HumanCommandRequired(
            f"'{action}' changes the model and runs only on a human's supervised command [A-21]; "
            "pass a roles.contracts.HumanCommand (approver, action, reason)."
        )
    if command.action != action:
        raise HumanCommandRequired(f"the HumanCommand authorises {command.action!r}, not {action!r}")
    return command


def command_record(command: HumanCommand) -> dict[str, Any]:
    """The JSON record of a command: exactly the fields its identity is computed from."""
    return {"approver": command.approver, "action": command.action, "reason": command.reason, "time": float(command.time)}


def command_id(command: HumanCommand) -> str:
    """SHA-256 of the canonical JSON of the command's fields (AS-831): equal commands share one id."""
    return digest_of(command_record(command))


def apply_calibration(proposal: CalibrationProposal, command: HumanCommand | None, *, state: TemperatureState,
                      audit: list[HumanCommand] | None = None) -> TemperatureState:
    """New temperatures = the current ones updated with the proposal's families, under a valid command."""
    cmd = require_command(command, APPLY_CALIBRATION)
    temps = dict(state.temperatures)
    temps.update(proposal.temperatures)
    if audit is not None:
        audit.append(cmd)
    return TemperatureState(temperatures=temps, applied_by=(*state.applied_by, cmd))


def train_on_feedback(model: nn.Module, optimizer: torch.optim.Optimizer, loss_fn: Callable[[], torch.Tensor],
                      command: HumanCommand | None, *, audit: list[HumanCommand] | None = None) -> float:
    """One optimisation step of `loss_fn()` on `model`, under a valid "update-weights" command. Returns the loss."""
    cmd = require_command(command, UPDATE_WEIGHTS)
    optimizer.zero_grad(set_to_none=True)
    loss = loss_fn()
    loss.backward()
    optimizer.step()
    if audit is not None:
        audit.append(cmd)
    return float(loss.detach())


__all__ = ["APPLY_CALIBRATION", "FEEDBACK_ACTIONS", "FIT_FEEDBACK", "INGEST_FEEDBACK", "ROLLBACK_UPDATE", "UPDATE_WEIGHTS",
           "apply_calibration", "command_id", "command_record", "require_command", "train_on_feedback"]
