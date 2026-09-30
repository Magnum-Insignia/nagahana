"""Exceptions that carry the project's rules.

What this is
------------
Many of NagaHana's constraints are *governance* rules, not maths:
- nothing undecided may be silently defaulted [Q-39];
- the Verifier changes weights only on a human's command [A-16, A-21];
- the Generator exists only in training [A-02, Q-37];
- memory regions have owners [Q-31, Q-32].

Each rule is enforced by raising one of the exceptions below. Every message names what the code
is waiting on, so a failure reads as an instruction ("decide D-12", "enable proposal P-18"),
never as a mystery.

Design notes
------------
- `NotBuiltYet` subclasses `NotImplementedError` so linters and IDEs treat template bodies as
  intentionally unimplemented. It additionally records *why*: the decision IDs, proposal IDs or
  pipeline stage it waits on. This keeps the template honest (CLAUDE.md: "no manipulations, mocks
  or faking of work").
- All exceptions derive from `NagaHanaError`, so callers can catch "a project rule fired" in one
  place (for example the CLI, which prints them as to-do items).
"""

from __future__ import annotations

from collections.abc import Iterable


class NagaHanaError(Exception):
    """Base class for every error this codebase raises on purpose."""


class DecisionHeld(NagaHanaError):
    """A code path needs a design decision the owner has not made (or is holding).

    Raised by `nagahana.governance.decisions.require`. The fix is always to decide and then record
    the decision (registry + DESIGN_LOG.md), never to pick a default in place [Q-39].
    """


class ProposalNotEnabled(NagaHanaError):
    """A component implements a proposal (P-xx) that has not been enabled.

    Proposals are engineering suggestions awaiting the owner's approval (DESIGN_LOG.md §3). Their
    code may exist, but it runs only when the proposal ID is listed in `enabled_proposals`.
    """


class AccessDenied(NagaHanaError):
    """A role touched a memory region its access rules do not allow (see `memory/access.py`)."""


class HumanCommandRequired(NagaHanaError):
    """A model-changing operation was attempted without an explicit human command.

    The Verifier "will only adjust model's weights based on human's supervised command to do so,
    otherwise it wont" [A-21]. Automatic feedback into the Forecaster was removed because "it
    would conduct the poisoning again" [A-16].
    """


class ModeViolation(NagaHanaError):
    """A component was used in a run mode it does not belong to (e.g. Generator at inference)."""


class ConfigMissing(NagaHanaError):
    """A configuration value is required but absent or still `???` (Hydra's mandatory marker)."""


class InvariantViolation(NagaHanaError):
    """A data or model object broke a documented invariant (e.g. an excluded field carrying a value)."""


class NotBuiltYet(NotImplementedError):
    """A template body that is intentionally not implemented yet.

    Parameters
    ----------
    what:
        What is not built, in plain words ("CVG-AE hypergraph encoder forward pass").
    waiting_on:
        Decision IDs (D-xx), proposal IDs (P-xx) or pipeline stages ("stage-1 analysis") that
        must be settled first. May be empty when the only blocker is engineering time.
    """

    def __init__(self, what: str, waiting_on: Iterable[str] = ()) -> None:
        self.what = what
        self.waiting_on = tuple(waiting_on)
        suffix = f" (waiting on: {', '.join(self.waiting_on)})" if self.waiting_on else ""
        super().__init__(f"Not built yet: {what}{suffix}")
