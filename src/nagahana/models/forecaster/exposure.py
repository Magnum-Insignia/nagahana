"""Adapter: TAAFT's set-level marginal energy → the per-hypothesis exposure callable of the Forecaster (AS-17, AS-252).

TAAFT (engineer C) exposes the null-context energy of a *set* of hypotheses:

    TAAFT.marginal_energy(y [B, M, N, d_y], token_mask [B, M, N], *, n_entities) → (E_total [B, M], terms)

The Forecaster's exposure term needs E(∅, ŷ) of *one* imagined hypothesis (the target's). The adapter
evaluates each hypothesis as a one-entity set (N = 1, no adversary slot, no pairs), so the pairwise
lens terms (topology, cause) contribute nothing and the unary terms score the hypothesis alone:

    exposure_callable(ŷ [..., d_y]) = E_total(ŷ as a 1-entity set)  [...]

This is a reading of "the marginal-energy novelty of the step" (AS-17) recorded as AS-252; whether
TAAFT's set energy should instead be evaluated on the whole imagined set at each step is open (it
needs every entity's imagined hypothesis, which the Forecaster does not produce).
"""

from __future__ import annotations

from typing import Protocol

import torch

from nagahana.models.forecaster.model import MarginalEnergy


class SetMarginalEnergy(Protocol):
    """Anything with TAAFT's `marginal_energy` signature."""

    def marginal_energy(self, y: torch.Tensor, token_mask: torch.Tensor, *, n_entities: int
                        ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]: ...


def exposure_from_taaft(taaft: SetMarginalEnergy) -> MarginalEnergy:
    """Wrap a TAAFT-like model into the Forecaster's per-hypothesis marginal-energy callable."""

    def energy(y: torch.Tensor) -> torch.Tensor:
        flat = y.reshape(-1, 1, 1, y.shape[-1])                                # [n, 1, 1, d_y]: n one-entity sets
        mask = torch.ones(flat.shape[:3], dtype=torch.bool, device=y.device)
        total, _ = taaft.marginal_energy(flat, mask, n_entities=1)             # [n, 1]
        return total.reshape(y.shape[:-1])

    return energy
