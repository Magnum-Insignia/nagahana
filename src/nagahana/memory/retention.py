"""Retention per Forecaster trigger (D-36) and the rule that fires triggers (D-02, held; option in force).

Retention (D-36)
----------------
Retention must not be controllable by attack-generated volume; it is regular and consistent for every
Forecaster trigger. In the gated long-term memory update (Titans form, Behrouz et al., arXiv:2501.00663,
`memory/longterm.py`)

    M_k = (1 - alpha_k) M_(k-1) + S_k,      S_k = eta S_(k-1) - theta grad l(M_(k-1); k_k, v_k)

the retention factor alpha_k depends only on the trigger index k: not on how many state updates arrived since
the last trigger, their content, or their rate. A flood of attacker-generated traffic therefore cannot
accelerate forgetting. The schedule's signature enforces this: `alpha(k)` receives nothing but k.

The trigger rule (D-02)
-----------------------
If triggers were purely event-driven, attack volume could still drive the number of triggers, and so the
forgetting per unit time. D-02 is held; its admissible options are both rules below, and
`trigger_policy()` resolves the option in force through `governance.decisions` ("forecaster-trigger-policy"):
- "fixed cadence": triggers at the epoch grid k c (c = `ForecasterConfig.window_seconds`), and nothing else;
- "fixed cadence + capped priority triggers" (working option, AS-12): the cadence grid plus at most one
  priority trigger per cadence interval [k c, (k + 1) c), fired when TAAFT's marginal energy jumps (AS-416).
Under either rule the number of triggers in an interval of length L is at most ceil(L / c) + (cap) ceil(L / c),
whatever the input volume: the cap bounds what an attacker can force.

`TriggerRule` is the rule as data (option, cadence, cap). It is applied by the components that fire triggers
(`data/windows.py` places the cadence grid for training; `inference/engine.py` adds the priority triggers) and
audited by `TriggerRule.check`, which verifies a log of fired triggers against the rule (the invariant the
retention guarantee rests on).
"""

from __future__ import annotations

import enum
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from nagahana.core.errors import InvariantViolation
from nagahana.governance import decisions


class RetentionSchedule(Protocol):
    """alpha_k as a function of the trigger index alone (D-36)."""

    def alpha(self, k: int) -> float:
        """Forgetting fraction at trigger k, in [0, 1)."""
        ...


@dataclass(frozen=True)
class ConstantRetention:
    """alpha_k = alpha for every trigger (`MemoryConfig.retention_alpha`, AS-11)."""

    value: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.value < 1.0:
            raise InvariantViolation("retention alpha must be in [0, 1)")

    def alpha(self, k: int) -> float:
        if k < 0:
            raise ValueError("trigger index must be >= 0")
        return self.value


class TriggerOption(enum.Enum):
    """The admissible options of D-02 (values equal the option strings of the decision registry)."""

    FIXED_CADENCE = "fixed cadence"
    CAPPED_PRIORITY = "fixed cadence + capped priority triggers"


def trigger_policy(configured: str | None = None) -> TriggerOption:
    """The D-02 option in force: `configured`, else the run's configured option, else AS-12's working option."""
    d = decisions.require("forecaster-trigger-policy", configured, by=__name__)
    assert d.value is not None
    return TriggerOption(d.value)


@dataclass(frozen=True)
class TriggerRule:
    """The trigger rule in force (module docstring): option, cadence c (s) and priority triggers per interval."""

    option: TriggerOption
    cadence_s: float
    priority_per_interval: int

    def __post_init__(self) -> None:
        if not (self.cadence_s > 0 and math.isfinite(self.cadence_s)):
            raise InvariantViolation("the cadence must be a positive number of seconds")
        expected = 0 if self.option is TriggerOption.FIXED_CADENCE else 1
        if self.priority_per_interval != expected:
            raise InvariantViolation(f"the option {self.option.value!r} allows {expected} priority trigger(s) per interval")

    @classmethod
    def in_force(cls, cadence_s: float, configured: str | None = None) -> TriggerRule:
        """The rule of the D-02 option in force, at cadence `cadence_s` (`ForecasterConfig.window_seconds`)."""
        option = trigger_policy(configured)
        return cls(option, float(cadence_s), 0 if option is TriggerOption.FIXED_CADENCE else 1)

    @property
    def priority_enabled(self) -> bool:
        return self.priority_per_interval > 0

    def interval(self, t: float) -> int:
        """Index k of the cadence interval [k c, (k + 1) c) holding the epoch time t."""
        return math.floor(t / self.cadence_s)

    def max_triggers(self, span_s: float) -> int:
        """Most triggers the rule allows in any interval of length `span_s`: independent of the input volume."""
        if span_s < 0:
            raise ValueError("span_s must be >= 0")
        grid = math.floor(span_s / self.cadence_s) + 1
        return grid * (1 + self.priority_per_interval)

    def check(self, times: Sequence[float], kinds: Sequence[str]) -> None:
        """Raise `InvariantViolation` unless a trigger log obeys the rule.

        times: epoch seconds of the fired triggers, strictly increasing; kinds: "cadence" or "priority" for each.
        Cadence triggers must sit on the grid k c (within 1e-9 relative); priority triggers are allowed only under
        the capped-priority option, at most `priority_per_interval` per cadence interval.
        """
        if len(times) != len(kinds):
            raise InvariantViolation("one kind per trigger time")
        if any(b <= a for a, b in zip(times, times[1:], strict=False)):
            raise InvariantViolation("trigger times must strictly increase")
        per_interval: dict[int, int] = {}
        for t, kind in zip(times, kinds, strict=True):
            if kind == "cadence":
                k = round(t / self.cadence_s)
                if abs(t - k * self.cadence_s) > 1e-9 * max(1.0, abs(t)):
                    raise InvariantViolation(f"cadence trigger at {t} is off the grid of {self.cadence_s} s")
            elif kind == "priority":
                if not self.priority_enabled:
                    raise InvariantViolation(f"a priority trigger at {t}, but the option is {self.option.value!r}")
                k = self.interval(t)
                per_interval[k] = per_interval.get(k, 0) + 1
                if per_interval[k] > self.priority_per_interval:
                    raise InvariantViolation(f"more than {self.priority_per_interval} priority trigger(s) in interval {k}")
            else:
                raise InvariantViolation(f"unknown trigger kind {kind!r}")
