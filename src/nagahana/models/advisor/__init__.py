"""Advisor: D3FEND counter-measure sequences, re-imagined against the Forecaster, advisory only (build-spec §2.9).

- `model.Advisor`: π_D / V_D over the D3FEND table; `advise` (beam search → AdvisoryBundle, info).
- `d3fend`: the action table (slot → D3FEND technique, tactic, level, effect, disruption).
- `effects`: structural effect model, disruption cost, information value, rule checks.
- `losses`: model-based policy improvement on −ΔP_inf − κ·cost.
"""

from nagahana.models.advisor.model import Advisor, Evaluation, PhysicsCheck, slice_trigger

__all__ = ["Advisor", "Evaluation", "PhysicsCheck", "slice_trigger"]
