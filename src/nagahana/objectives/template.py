"""The one loss template shared by every component (D-37).

    L_c = L_c^task + lambda_phys Phi_phys + lambda_E L_c^E + lambda_S S

- L_c^task: the component's own objective (masked reconstruction for the Simulator; dynamics, fidelity and
  adversary return for the Forecaster; the counter-sequence objective for the Advisor; calibration and human
  feedback for the Verifier; the conditional generative loss for the Generator).
- Phi_phys: the shared physics term (physics/term.py). One term, reused everywhere.
- L_c^E: the shared energy term (models/taaft/energy.py). Which energy jobs are in v1 is the held D-11b,
  resolved to its working option (AS-16).
- S: a proper scoring loss (objectives/scoring.py).

A component includes a shared term only where the matrix marks it. `USAGE` is that matrix, as data. A
component may use only the shared terms its row allows, and every term it uses needs an explicit lambda (no
defaults). The Decoder's row is empty: it has no physics of its own (D-37).
"""

from __future__ import annotations

from collections.abc import Mapping

import torch

from nagahana.core.errors import ConfigMissing, InvariantViolation
from nagahana.core.roles import Role

#: Shared terms each role's loss may include.
USAGE: dict[Role, frozenset[str]] = {
    Role.SIMULATOR: frozenset({"physics"}),
    Role.FORECASTER: frozenset({"physics", "energy", "scoring"}),
    Role.ADVISOR: frozenset({"physics", "energy"}),
    Role.VERIFIER: frozenset({"physics", "scoring"}),
    Role.GENERATOR: frozenset({"physics"}),
    Role.DECODER: frozenset(),
}


class ComponentLoss:
    """L_c for one role. `weights` maps each used shared term to its lambda (> 0).

    Parameters
    ----------
    role: whose loss this is.
    weights: {"physics": lambda_phys, "energy": lambda_E, "scoring": lambda_S}. Only terms in USAGE[role] are
        allowed, and all of them must be given (change the matrix first if a term should be dropped).
    """

    def __init__(self, role: Role, weights: Mapping[str, float]) -> None:
        allowed = USAGE[role]
        extra = set(weights) - allowed
        if extra:
            raise InvariantViolation(f"{role.value} may not use shared terms {sorted(extra)} (usage matrix)")
        missing = allowed - set(weights)
        if missing:
            raise ConfigMissing(f"{role.value} loss needs lambda for {sorted(missing)} (no defaults)")
        for k, w in weights.items():
            if float(w) <= 0:
                raise InvariantViolation(f"lambda for {k!r} must be > 0")
        self.role = role
        self.weights = {k: float(v) for k, v in weights.items()}

    def __call__(self, task: torch.Tensor, shared: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return (L_c, breakdown). `shared` must hold a scalar for every used term."""
        missing = set(self.weights) - set(shared)
        if missing:
            raise InvariantViolation(f"shared terms not computed: {sorted(missing)}")
        parts = {"task": task}
        total = task
        for k, w in self.weights.items():
            parts[k] = w * shared[k]
            total = total + parts[k]
        return total, parts
