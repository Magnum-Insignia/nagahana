"""Run modes, and guards that keep each component in the modes it belongs to.

Modes
-----
- `TRAIN`: the six training stages (pipeline/stages.py).
- `EVALUATE`: benchmarking and zero-shot validation, with no weight updates.
- `INFER_LIVE`: deployment on a live telemetry stream. Passive, advisory only [Q-13].
- `FORENSIC_REPLAY`: the same engine on an uploaded capture ("a forecaster and a forensics tool",
  [Q-17]; ARCH §7). Forecasting and forensics are one engine at two settings.
- `LAB`: experiments (JAX/Equinox ground-truth worlds, information audits) [Q-40], [Q-44].

Why modes are explicit
----------------------
Two decided rules depend on the mode:
- the Generator is training-only (D-40, [A-17]); using it at inference would mix imagined events
  into evidence;
- weights are frozen outside training unless a human commands otherwise (D-21, [Q-14]).

The current mode has no default. Code that needs it raises until a caller sets it. This follows the
owner's rule against silent defaults [Q-39], and it prevents the dangerous case of forgetting to
leave TRAIN before deployment.
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
    token = _CURRENT.set(mode)
    try:
        yield mode
    finally:
        _CURRENT.reset(token)


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
