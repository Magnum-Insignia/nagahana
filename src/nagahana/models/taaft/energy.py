"""Energy refinement: the energy-based-transformer core of TAAFT ([A-19] item 2; refs.md#L1031).

Idea
----
An energy function E(context, y) scores how compatible a candidate y (e.g. an imagined next state)
is with its context. Low energy means compatible. Instead of emitting y in one shot, the model
*thinks* by descending the energy from a rough candidate:

    ŷ^{(i+1)} = ŷ^{(i)} − α · ∇_y E_tot(ctx, ŷ^{(i)}) + ε_i,       i = 0 … S−1
    E_tot = E_θ(ctx, y) + λ_phys · Φ_phys(decode(y))

- More steps S when the stakes are higher: this is inference-time scaling. The owner asked for
  "inference time scaling (similar to the idea of reasoning …)" (DESIGN_LOG 2026-09-28) and for
  looking "deeper and longer before flagging" [Q-21].
- ε_i is small annealed noise, so refinement can explore several basins (hypotheses) before
  committing.
- The physics term is evaluated on the *decoded* candidate, because residuals are defined on
  observables (flows, bytes), not on latents. Refinement therefore cannot drift into physically
  impossible states (the boundary role, D-18).

This module implements the generic descent. Which energy networks exist and what they are used for
in v1 is held (D-11b); the proposal to read one network two ways (P-09) lives in the lens layer.
"""

from __future__ import annotations

from collections.abc import Callable

import torch


def energy_descent(
    energy: Callable[[torch.Tensor], torch.Tensor],
    y0: torch.Tensor,
    *,
    steps: int,
    step_size: float,
    noise: Callable[[int], float],
    generator: torch.Generator | None = None,
    create_graph: bool = False,
) -> tuple[torch.Tensor, list[float]]:
    """Refine `y0` by (noisy) gradient descent on `energy`.

    Parameters
    ----------
    energy:
        Maps a candidate batch [..., D] to energies [...] (lower is better). Include any physics
        term inside it.
    y0:
        Initial candidate(s).
    steps, step_size:
        S and α. Required, no defaults: they are inference-time-scaling knobs set per deployment.
    noise:
        i ↦ σ_i, the standard deviation of ε_i at step i. Pass `lambda i: 0.0` for plain descent.
    create_graph:
        Keep the refinement differentiable, for training *through* the thinking steps.

    Returns
    -------
    (ŷ_S, [E(ŷ_0), …, E(ŷ_{S−1})] summed over the batch, for monitoring convergence)
    """
    if steps < 0 or step_size <= 0:
        raise ValueError("steps must be >= 0 and step_size > 0")
    y = y0 if create_graph else y0.detach().clone()
    if not y.requires_grad:
        y.requires_grad_(True)
    trace: list[float] = []
    for i in range(steps):
        e = energy(y).sum()
        trace.append(float(e.detach()))
        (grad,) = torch.autograd.grad(e, y, create_graph=create_graph)
        sigma = float(noise(i))
        step = y - step_size * grad
        if sigma > 0:
            step = step + sigma * torch.randn(y.shape, generator=generator, dtype=y.dtype, device=y.device)
        y = step if create_graph else step.detach().requires_grad_(True)
    return (y if create_graph else y.detach()), trace
