"""Exceptions that carry the project's rules.

Several of NagaHana's constraints are governance rules rather than mathematics:

- an undecided design question is never silently defaulted: a held decision resolves only to its
  recorded working option or to an option configured for the run (governance/decisions.py);
- the Verifier changes weights only under an explicit human command (D-21);
- the Generator runs only in training (D-40);
- memory regions have owners (D-35).

Each rule is enforced by raising one of the exceptions below. Every message names what the code
needs (a decision ID, a proposal ID, a configured option), so a failure reads as an instruction and
never as a mystery.

Design notes
------------
- All exceptions derive from `NagaHanaError`, so callers can catch "a project rule fired" in one
  place (for example the CLI, which prints them as to-do items).
- `NotBuiltYet` subclasses `NotImplementedError` and records what a body waits on. It is kept so that
  modules importing it keep working; no normal path of the model raises it.
"""

from __future__ import annotations

from collections.abc import Iterable


class NagaHanaError(Exception):
    """Base class for every error this codebase raises on purpose."""


class DecisionHeld(NagaHanaError):
    """A code path needs a design decision that cannot be resolved here.

    Raised by `nagahana.governance.decisions.require` for a proposal (proposals are gated with
    `require_proposal`), for a superseded entry, and for a held entry that has neither a recorded
    working option nor an option configured for the run. Raised by
    `nagahana.governance.assumptions.assume` in strict (audit) mode, which runs the code as if every
    working assumption were still held.
    """


class InvalidOption(NagaHanaError):
    """A configured option is not admissible.

    Raised when a run configures an option that a decision does not list, or when a component is
    built under an option it does not implement (for example a lens that exists only when its
    placement option says so).
    """


class ProposalNotEnabled(NagaHanaError):
    """A component implements a proposal (P-xx) that has not been enabled for this run.

    Proposal code exists in the tree but runs only when the proposal ID is listed in the run's
    `enabled_proposals`.
    """


class AccessDenied(NagaHanaError):
    """A role touched a memory region its access rules do not allow (see `memory/access.py`)."""


class HumanCommandRequired(NagaHanaError):
    """A model-changing operation was attempted without an explicit human command.

    The Verifier adjusts weights or calibrations only under a supervised human command (D-21); an
    automatic feedback path would let poisoned telemetry steer the model.
    """


class ModeViolation(NagaHanaError):
    """A component was used in a run mode it does not belong to (e.g. the Generator at inference)."""


class ConfigMissing(NagaHanaError):
    """A configuration value is required but absent or still `???` (Hydra's mandatory marker)."""


class InvariantViolation(NagaHanaError):
    """A data or model object broke a documented invariant (e.g. an excluded field carrying a value)."""


class NotBuiltYet(NotImplementedError):
    """A body that is not available, with the items it waits on.

    Parameters
    ----------
    what:
        What is not available, in plain words.
    waiting_on:
        Decision IDs (D-xx), proposal IDs (P-xx) or pipeline stages that must be settled first. May be
        empty.
    """

    def __init__(self, what: str, waiting_on: Iterable[str] = ()) -> None:
        self.what = what
        self.waiting_on = tuple(waiting_on)
        suffix = f" (waiting on: {', '.join(self.waiting_on)})" if self.waiting_on else ""
        super().__init__(f"Not built yet: {what}{suffix}")
