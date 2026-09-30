"""The Advisor (formerly Planner): counter-measure sequences, advisory only (D-01, D-19, D-33).

Role
----
"The advisor is another agent/policy-value pair that directly works on both env & imagination to
produce counters" [A-15]. For v1, its actions are "defend based counters, better expressed as a
sequence of actions … but not actional, only advisory" [Q-30], expressed in MITRE D3FEND.

The loop (diagram 07): propose a sequence → re-imagine with it applied (Forecaster's model) → let
the modelled adversary respond → score → repeat. It is GAN-like in function only: the Advisor looks
for counters to the forecast and never changes the Forecaster [Q-38]. It keeps no memory of its own
(D-35).

Scoring each sequence needs three held decisions:
- D-03a: what counts as the attacker succeeding (the P_inf target);
- D-03b: how disruption is priced per asset (OT availability first?);
- D-03c: expected vs worst case over adversary policies.
Its step reward (DESIGN_LOG 2026-09-28) combines cost, potential-based energy shaping and information
value (objectives/rewards.py).
"""

from __future__ import annotations

from nagahana.core.errors import NotBuiltYet


class Advisor:
    """Role orchestrator (template). Output type: roles.contracts.AdvisoryBundle (always advisory)."""

    def advise(self, *args: object, **kwargs: object) -> object:
        """Search and rank counter-measure sequences for the current forecast."""
        raise NotBuiltYet("Advisor search (propose → re-imagine → respond → score)", waiting_on=("D-03a", "D-03b", "D-03c", "D-12"))
