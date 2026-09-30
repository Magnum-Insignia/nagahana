"""The one loss template shared by every component (legend 00; D-37, [Q-41]).

    𝓛_c = 𝓛_c^task + λ_phys · Φ_phys + λ_E · 𝓛_c^E + λ_S · S

- 𝓛_c^task: the component's own objective (masked reconstruction for the Simulator; dynamics,
  fidelity and adversary return for the Forecaster; counter-sequence objective for the Advisor;
  calibration and human feedback for the Verifier; conditional generative loss for the Generator).
- Φ_phys: the shared physics term (physics/term.py). One term, reused everywhere.
- 𝓛_c^E: the shared energy term (models/taaft/energy.py). Which uses are in v1 is held (D-11b).
- S: a proper scoring loss (objectives/scoring.py).

"include a shared term only where the matrix marks it" (legend 00). `USAGE` is that matrix, as
data. A component may use only the shared terms its row allows, and every term it uses needs an
explicit λ (no defaults). The Decoder's row is empty: it has no physics of its own (D-37).

The matrix below is legend 00 as drawn on 2026-09-29, with the new role names. Revisit it as TAAFT's
design settles; it is one place to edit.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch

from nagahana.core.errors import ConfigMissing, InvariantViolation
from nagahana.core.roles import Role

#: Shared terms each role's loss may include (legend 00).
USAGE: dict[Role, frozenset[str]] = {
    Role.SIMULATOR: frozenset({"physics"}),
    Role.FORECASTER: frozenset({"physics", "energy", "scoring"}),
    Role.ADVISOR: frozenset({"physics", "energy"}),
    Role.VERIFIER: frozenset({"physics", "scoring"}),
    Role.GENERATOR: frozenset({"physics"}),
    Role.DECODER: frozenset(),
}


class ComponentLoss:
    """𝓛_c for one role. `weights` maps each used shared term to its λ (> 0).

    Parameters
    ----------
    role: whose loss this is.
    weights: {"physics": λ_phys, "energy": λ_E, "scoring": λ_S}. Only terms in USAGE[role] are allowed,
        and all of them must be given (write the matrix change first if a term should be dropped).
    """

    def __init__(self, role: Role, weights: Mapping[str, float]) -> None:
        allowed = USAGE[role]
        extra = set(weights) - allowed
        if extra:
            raise InvariantViolation(f"{role.value} may not use shared terms {sorted(extra)} (legend 00 matrix)")
        missing = allowed - set(weights)
        if missing:
            raise ConfigMissing(f"{role.value} loss needs λ for {sorted(missing)} (no defaults)")
        for k, w in weights.items():
            if float(w) <= 0:
                raise InvariantViolation(f"λ for {k!r} must be > 0")
        self.role = role
        self.weights = {k: float(v) for k, v in weights.items()}

    def __call__(self, task: torch.Tensor, shared: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return (𝓛_c, breakdown). `shared` must hold a scalar for every used term."""
        missing = set(self.weights) - set(shared)
        if missing:
            raise InvariantViolation(f"shared terms not computed: {sorted(missing)}")
        parts = {"task": task}
        total = task
        for k, w in self.weights.items():
            parts[k] = w * shared[k]
            total = total + parts[k]
        return total, parts
