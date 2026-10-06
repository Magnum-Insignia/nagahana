"""Residual normalisation: violations measured relative to the limit they break (AS-15, AS-114).

Why
---
Raw residuals have the units and the scale of their fields: one missing packet in a 3-packet flow and one in
a million-packet flow give the same r = 1, while 1 kB over the MTU budget is r = 1000. Summed into one
Phi_phys with fixed weights, the large-scale fields would dominate. AS-15 sets lambda_phys = 0.1 "on
residuals normalised by their field scale"; this module is that normalisation.

Definition
----------
For a residual r with limit b(x) (its `bound` method; every built-in residual has one):

    r_hat(x) = r(x) / (s0 + |b(x)|)

- r_hat = 0 exactly where r = 0, so the boundary itself is unchanged (zero inside, positive outside);
- r_hat is the violation as a fraction of the limit; s0 > 0 (one unit of the field, default 1) keeps it
  finite at b = 0 (e.g. a zero-duration flow with a positive gap) and makes tiny limits count in absolute
  units.
The scale is treated as a constant of the row (`detach`): the gradient pushes the violating value back
inside the limit, never the limit outward to shrink the ratio. A residual that is already a dimensionless
log-ratio reports a zero bound and is divided by s0 = 1 only.

Use: `PhysicsTerm([RelativeResidual(MTUBound("fwd", 1500)), ...], weights, target=...)`; the name gets the
suffix `.rel`, so weights are explicit for the normalised form.

Invariants (tests/test_perception_decoder.py): zero inside, positive outside, scale-invariant (the same
relative violation at two field scales gives the same r_hat up to s0).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

import torch

from nagahana.physics.residuals import Residual


class RelativeResidual:
    """r / (s0 + |bound|) for a residual `inner` (see the module docstring).

    Parameters
    ----------
    inner: the residual; it must expose `bound(x)` unless `bound` is given.
    bound: optional callable x -> limit [N], overriding `inner.bound`.
    floor: s0 > 0, in the field's units.
    """

    def __init__(
        self,
        inner: Residual,
        *,
        bound: Callable[[Mapping[str, torch.Tensor]], torch.Tensor] | None = None,
        floor: float = 1.0,
    ) -> None:
        fn = bound if bound is not None else getattr(inner, "bound", None)
        if fn is None:
            raise ValueError(f"residual {inner.name!r} has no bound(); pass bound=...")
        if floor <= 0:
            raise ValueError("floor must be positive")
        self.inner, self._bound, self.floor = inner, fn, float(floor)
        self.name: str = f"{inner.name}.rel"
        self.fields: tuple[str, ...] = tuple(inner.fields)

    def __call__(self, x: Mapping[str, torch.Tensor]) -> torch.Tensor:
        scale = (self.floor + torch.abs(self._bound(x))).detach()
        return self.inner(x) / scale
