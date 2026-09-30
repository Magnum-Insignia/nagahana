"""The shared physics term Φ_phys: one term, used by every component that produces states (D-37).

Definition (legend 00)
----------------------
    Φ_phys(x) = Σ_c  w_c · ‖ m_c ⊙ r_c(x) ‖²,        x ∈ 𝒞

- r_c: residuals (`residuals.py`), zero inside the physical boundary;
- m_c: 1 on rows where *every* field residual c reads contributes as evidence, else 0 (D-41). A
  residual never fires on data that is absent;
- w_c > 0: per-residual weights, required from config (no defaults);
- x ∈ 𝒞: hard limits already built into the outputs (`constraints.py`).

The norm is a sum over rows, exactly as in the legend. Batch-size scaling is absorbed by λ_phys in
each component's loss (`objectives/template.py`).

Where it is used (D-37, [Q-41])
-------------------------------
"physics informed aspect is at global level rather than specific areas … integrate it as a common
term across all the loss functions" [Q-41]:
- Simulator: on reconstructions;
- Forecaster: on every imagined step;
- Advisor: on the predicted effects of counters;
- Generator: on variants, via the physics-informed generator [A-17];
- Verifier: as a hallucination monitor.
The Decoder has none of its own. It decodes, and the losses above check what it outputs.

Target guard (D-18 vs D-25)
---------------------------
The decided use is on **model outputs** (a boundary against hallucination [A-08]). Scoring **incoming
observations** would turn physics into a telemetry-trust signal. That is held (D-25), so
`target=Target.OBSERVATION` raises until the owner decides.

Numerical note
--------------
Masked rows are zeroed *before* the residual is computed (not only after). Otherwise a NaN in an
absent field would poison the gradient through the masked-out branch. `torch.where` passes a zero
upstream gradient, but 0 × NaN is NaN.
"""

from __future__ import annotations

import enum
from collections.abc import Mapping, Sequence

import torch

from nagahana.core.errors import ConfigMissing, InvariantViolation
from nagahana.governance import decisions
from nagahana.physics.residuals import Residual


class Target(enum.Enum):
    """What the physics term is applied to."""

    MODEL_OUTPUT = "model_output"   # decided use (D-18)
    OBSERVATION = "observation"     # held (D-25)


class PhysicsTerm:
    """Φ_phys over a set of residuals. See the module docstring.

    Parameters
    ----------
    residuals:
        The residual instances to include (catalogue selection comes from config).
    weights:
        `{residual.name: w_c}` with every w_c > 0. A residual without a weight raises
        `ConfigMissing`.
    target:
        `Target.MODEL_OUTPUT` (decided) or `Target.OBSERVATION` (held, D-25).
    """

    def __init__(
        self,
        residuals: Sequence[Residual],
        weights: Mapping[str, float],
        *,
        target: Target,
    ) -> None:
        if target is Target.OBSERVATION:
            decisions.require("physics-on-telemetry")
        names = [r.name for r in residuals]
        if len(set(names)) != len(names):
            raise InvariantViolation(f"Duplicate residual names: {names}")
        for n in names:
            if n not in weights:
                raise ConfigMissing(f"No weight for physics residual {n!r} (w_c must be set explicitly).")
            if float(weights[n]) <= 0:
                raise InvariantViolation(f"Weight for {n!r} must be > 0.")
        self.residuals = tuple(residuals)
        self.weights = {n: float(weights[n]) for n in names}
        self.target = target

    def __call__(
        self,
        values: Mapping[str, torch.Tensor],
        contributing: Mapping[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return (Φ_phys, per-residual weighted terms).

        Parameters
        ----------
        values: field ID → tensor [N] of values (any content where not contributing).
        contributing: field ID → bool tensor [N], True where that field contributes. A field
            missing from this mapping counts as not contributing on every row.
        """
        breakdown: dict[str, torch.Tensor] = {}
        total: torch.Tensor | None = None
        n_rows = next(iter(values.values())).shape[0] if values else 0
        for r in self.residuals:
            mask = torch.ones(n_rows, dtype=torch.bool)
            for f in r.fields:
                m_f = contributing.get(f)
                mask = mask & (m_f if m_f is not None else torch.zeros(n_rows, dtype=torch.bool))
            safe = {
                f: torch.where(mask, values[f], torch.zeros_like(values[f]))
                for f in r.fields
                if f in values
            }
            if len(safe) != len(r.fields):
                # Fields absent from `values` entirely: nothing to check, the residual is masked out.
                term = torch.zeros(())
            else:
                res = torch.where(mask, r(safe), torch.zeros(n_rows))
                term = self.weights[r.name] * (res**2).sum()
            breakdown[r.name] = term
            total = term if total is None else total + term
        return (total if total is not None else torch.zeros(())), breakdown
