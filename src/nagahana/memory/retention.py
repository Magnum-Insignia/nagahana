"""Retention: how much memory is kept at each Forecaster trigger (D-36, [Q-33]); what fires a trigger (D-02, held).

The decided rule (D-36)
-----------------------
"the retention not be controllable by attack generated volume, it should be regular & consistent for
every renderer trigger of computation" [Q-33] (the renderer is now the Forecaster, D-19).

In the gated memory update drawn in diagram 08 (Titans-style, refs.md#L2423):

    M_k = (1 − α_k) · M_{k−1} + S_k,      S_k = η_k · S_{k−1} − θ_k · ∇ℓ(M_{k−1}; k_k, v_k)

the retention factor α_k may depend **only on the trigger index k**. It must not depend on how many
state updates arrived since the last trigger, their content, or their rate. Then a flood of
attacker-generated traffic cannot accelerate forgetting (the "detection-blinding via overload"
threat, ARCH §10.1). The schedule's signature enforces this: `alpha(k)` receives nothing but k.

Held: the trigger policy (D-02)
-------------------------------
If triggers were purely event-driven, attack volume could still drive the *number* of triggers, and
so the forgetting per unit time. Options on record: fixed cadence; fixed cadence plus capped
priority triggers; other. `trigger_policy()` raises until the owner decides.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from nagahana.core.errors import InvariantViolation
from nagahana.governance import decisions


class RetentionSchedule(Protocol):
    """α_k as a function of the trigger index alone (D-36)."""

    def alpha(self, k: int) -> float:
        """Forgetting fraction at trigger k, in [0, 1)."""
        ...


@dataclass(frozen=True)
class ConstantRetention:
    """α_k = α for every trigger. The value is required (a sizing decision; no default)."""

    value: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.value < 1.0:
            raise InvariantViolation("retention α must be in [0, 1)")

    def alpha(self, k: int) -> float:
        if k < 0:
            raise ValueError("trigger index must be >= 0")
        return self.value


class TriggerPolicy(Protocol):
    """Decides when the Forecaster recomputes (held: D-02)."""

    def should_trigger(self, now: float, last_trigger: float) -> bool:
        """True when a new computation should start. Must not be driven by input volume alone."""
        ...


def trigger_policy() -> str:
    """The decided trigger policy; raises `DecisionHeld` while D-02 is held."""
    d = decisions.require("forecaster-trigger-policy")
    assert d.value is not None
    return d.value
