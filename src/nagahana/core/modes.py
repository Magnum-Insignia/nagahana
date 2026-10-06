"""Run modes, and guards that keep each component in the modes it belongs to.

Modes
-----
- `TRAIN`: the six training stages (pipeline/stages.py).
- `EVALUATE`: benchmarking and zero-shot validation, with no weight updates.
- `INFER_LIVE`: deployment on a live telemetry stream. Passive and advisory only (D-32, D-33).
- `FORENSIC_REPLAY`: the same engine on an uploaded capture (architecture section 1: two settings of
  one engine).
- `LAB`: experiments (ground-truth worlds, information audits, P-14, P-15).

Why modes are explicit
----------------------
Two decided rules depend on the mode:
- the Generator is training-only (D-40); using it at inference would mix imagined events into
  evidence;
- weights are frozen outside training unless a human commands otherwise (D-21).

The current mode has no default. Code that needs it raises until a caller sets it, which prevents the
dangerous case of forgetting to leave TRAIN before deployment.
"""

from __future__ import annotations

import contextvars
import enum
from collections.abc import Iterator
from contextlib import contextmanager

from nagahana.core.errors import ModeViolation


class RunMode(enum.Enum):
    """Where the code is running. See the module docstring."""

    TRAIN = "train"
    EVALUATE = "evaluate"
    INFER_LIVE = "infer_live"
    FORENSIC_REPLAY = "forensic_replay"
    LAB = "lab"


_CURRENT: contextvars.ContextVar[RunMode | None] = contextvars.ContextVar(
    "nagahana_run_mode", default=None
)


@contextmanager
def run_mode(mode: RunMode) -> Iterator[RunMode]:
    """Set the run mode for the enclosed block (thread- and task-local via contextvars)."""
    previous = _CURRENT.set(mode)
    try:
        yield mode
    finally:
        _CURRENT.reset(previous)


def current_mode() -> RunMode:
    """The active run mode. Raises `ModeViolation` if none was set; there is deliberately no default."""
    mode = _CURRENT.get()
    if mode is None:
        raise ModeViolation(
            "No run mode set. Wrap the call in `with run_mode(RunMode.X):` "
            "(there is deliberately no default mode)."
        )
    return mode


def require_mode(*allowed: RunMode, component: str) -> RunMode:
    """Raise `ModeViolation` unless the current mode is one of `allowed`."""
    mode = current_mode()
    if mode not in allowed:
        names = ", ".join(m.value for m in allowed)
        raise ModeViolation(f"{component} may run only in [{names}], not in {mode.value!r}.")
    return mode
