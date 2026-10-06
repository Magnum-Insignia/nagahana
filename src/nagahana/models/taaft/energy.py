"""Energy refinement: the energy-based-transformer core of TAAFT ([A-19] item 2; refs.md#L1031).

Idea
----
An energy function E(context, y) scores how compatible a candidate y (TAAFT's hypotheses about the
entities and the adversary) is with its context. Low energy means compatible. Instead of emitting y
in one shot, the model *thinks* by descending the energy from a rough candidate:

    ŷ^{(i+1)} = ŷ^{(i)} − α · ∇_y E_total(c, ŷ^{(i)}) + ε_i,       i = 0 … S−1
    E_total = Σ_ℓ E_ℓ(c, y) + λ_phys · Φ_phys(decode(y))                       (D-42, ADR-0007)

- More steps S when the stakes are higher: inference-time scaling. The owner asked for "inference
  time scaling (similar to the idea of reasoning …)" (DESIGN_LOG 2026-09-28) and for looking "deeper
  and longer before flagging" [Q-21]. S is a run-time budget recorded with each forecast (D-44).
- α is learned (TAAFT parameterises it as softplus(ρ) > 0) and trained *through* the unrolled
  descent (create_graph), the energy-based-transformer recipe (Gladstone et al. 2025,
  arXiv:2507.02092; AS-16).
- ε_i ~ 𝒩(0, σ_i²) is annealed noise, σ_i = σ_0 (1 − i/S) (`annealed_noise`, AS-212), so early steps
  can leave a poor basin and the last step is deterministic.
- The physics term is evaluated on the *decoded* candidate, because residuals are defined on
  observables (flows, bytes), not on latents (the boundary role, D-18).

Decomposition (explanations, D-42)
----------------------------------
The deterministic part of a step is d = −α Σ_ℓ ∇E_ℓ. `lens_step_shares` reports for each lens the
share of that displacement it is responsible for, as the projection of its own step on d:

    share_ℓ = ⟨−α ∇E_ℓ, d⟩ / ‖d‖²,          Σ_ℓ share_ℓ = ⟨d, d⟩ / ‖d‖² = 1 exactly.

A share can be negative: that lens pulled against the step that was taken (it was outvoted). When
d = 0 (a fixed point) every share is reported as 0, and the shares then do not sum to 1; this is
the only exception and it is documented in AS-211. Noise is excluded from d: it explains nothing.

Precision (D-54, AS-451)
------------------------
The energy E(y) is float64 (the lenses reduce and sum in float64), but the descent variable y stays
float32: autograd returns ∇_y E in y's dtype (the backward of the float64 cast), and the update
y − α∇E is a float32 step. This is the choice D-54 leaves open ("the descent may stay fp32"): the
step is a learned optimisation move whose float32 rounding (relative 6·10⁻⁸ per coordinate) is far
below the step's own scale (and, in training, below the annealed noise σ_i), and ŷ then feeds float32
readout heads, whose logits are cast to float64 only before their link functions (readouts.py). What is
*reported* about the descent is float64: the trace E(ŷ_i) (Python floats from float64 sums) and the
lens shares (`lens_step_shares` projects the float32 per-lens gradients in float64). No lens of this
build needs more precision in its gradient: every lens is a smooth function of float32 projections.

Contracts
---------
- `descend` evaluates the energy under `torch.enable_grad()`, so refinement also works when the
  caller runs inference under `torch.no_grad()` (the energy gradient is part of the forward
  computation of TAAFT, not of training).
- With create_graph=False the result is detached from y0 and from α; with create_graph=True the
  whole unrolled chain stays differentiable, including α.

Assumptions: AS-14, AS-16, AS-211, AS-212. Decisions: D-42, D-44, D-11b (held beyond TAAFT's own energy).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import torch

StepSize = float | torch.Tensor


@dataclass
class DescentResult:
    """Outcome of `descend`.

    y: ŷ_S (dtype of y0: float32 in TAAFT). trace: E summed over the batch at ŷ_0 … ŷ_S (S + 1 Python
    floats; float64 when the energy is, D-54). y_last_input: ŷ_{S−1}
    (ŷ_0 when S = 0), detached: the point at which the last step's lens decomposition is read.
    """

    y: torch.Tensor
    trace: list[float]
    y_last_input: torch.Tensor


def annealed_noise(sigma0: float, steps: int) -> Callable[[int], float]:
    """σ_i = σ_0 · (1 − i/S): linear annealing to zero over S steps (AS-212)."""
    if sigma0 < 0:
        raise ValueError("sigma0 must be >= 0")
    s = max(1, steps)
    return lambda i: sigma0 * max(0.0, 1.0 - i / s)


def _check_step(step_size: StepSize) -> None:
    if isinstance(step_size, torch.Tensor):
        if bool((step_size.detach() <= 0).any()):
            raise ValueError("step_size must be > 0")
    elif step_size <= 0:
        raise ValueError("step_size must be > 0")


def descend(
    energy: Callable[[torch.Tensor], torch.Tensor],
    y0: torch.Tensor,
    *,
    steps: int,
    step_size: StepSize,
    noise: Callable[[int], float],
    generator: torch.Generator | None = None,
    create_graph: bool = False,
) -> DescentResult:
    """Refine `y0` by (noisy) gradient descent on `energy`; see the module docstring.

    energy: maps candidates [..., D] to energies [...] (lower is better), physics included.
    steps, step_size: S and α (a float or a positive scalar tensor, e.g. a learned softplus value).
    noise: i ↦ σ_i. generator: for reproducible noise. create_graph: differentiate through the steps.
    """
    if steps < 0:
        raise ValueError("steps must be >= 0")
    _check_step(step_size)
    with torch.enable_grad():
        y = y0 if create_graph else y0.detach().clone()
        if not y.requires_grad:
            y.requires_grad_(True)
        trace: list[float] = []
        last_input = y.detach()
        for i in range(steps):
            e = energy(y).sum()
            trace.append(float(e.detach()))
            (grad,) = torch.autograd.grad(e, y, create_graph=create_graph)
            last_input = y.detach()
            step = y - step_size * grad
            sigma = float(noise(i))
            if sigma > 0:
                step = step + sigma * torch.randn(y.shape, generator=generator, dtype=y.dtype, device=y.device)
            y = step if create_graph else step.detach().requires_grad_(True)
        trace.append(float(energy(y).sum().detach()))          # E at ŷ_S
    return DescentResult(y=(y if create_graph else y.detach()), trace=trace, y_last_input=last_input)


def energy_descent(
    energy: Callable[[torch.Tensor], torch.Tensor],
    y0: torch.Tensor,
    *,
    steps: int,
    step_size: StepSize,
    noise: Callable[[int], float],
    generator: torch.Generator | None = None,
    create_graph: bool = False,
) -> tuple[torch.Tensor, list[float]]:
    """Refine `y0` by (noisy) gradient descent on `energy` (the original API, kept for its callers).

    Returns (ŷ_S, [E(ŷ_0), …, E(ŷ_{S−1})] summed over the batch, for monitoring convergence).
    `descend` returns the same plus E(ŷ_S) and the last step's input point.
    """
    res = descend(energy, y0, steps=steps, step_size=step_size, noise=noise, generator=generator,
                  create_graph=create_graph)
    return res.y, res.trace[:-1]


def lens_step_shares(
    terms: Callable[[torch.Tensor], Mapping[str, torch.Tensor]],
    y: torch.Tensor,
    step_size: StepSize,
) -> dict[str, torch.Tensor]:
    """Each lens's share of the deterministic descent step taken at `y` (D-42; AS-211).

    terms: y ↦ {lens: energy per batch row [B]} (E_total is their sum). y: [B, ...].
    Returns {lens: share [B]} (detached, float64, D-54), Σ_ℓ share_ℓ = 1 wherever the displacement is
    non-zero. The per-lens gradients are float32 (y's dtype); they are cast to float64 (exactly)
    before the total displacement, its norm and the projections are formed, so the shares sum to 1
    to float64 rounding.
    """
    alpha = step_size.detach() if isinstance(step_size, torch.Tensor) else float(step_size)
    with torch.enable_grad():
        yy = y.detach().requires_grad_(True)
        energies = terms(yy)
        names = list(energies)
        steps: dict[str, torch.Tensor] = {}
        for k, n in enumerate(names):
            e = energies[n].sum()
            if e.requires_grad:
                (g,) = torch.autograd.grad(e, yy, retain_graph=k < len(names) - 1, allow_unused=True)
                g = torch.zeros_like(yy) if g is None else g
            else:
                g = torch.zeros_like(yy)
            steps[n] = -alpha * g.detach()                       # −α ∇E_ℓ  [B, ...]
    flat = {n: s.reshape(s.shape[0], -1).double() for n, s in steps.items()}   # float64 [B, D] each
    d = torch.stack(list(flat.values()), dim=0).sum(dim=0)     # [B, D] total displacement (float64)
    dd = (d * d).sum(-1)                                        # ‖d‖²  [B]
    out: dict[str, torch.Tensor] = {}
    for n, s in flat.items():
        num = (s * d).sum(-1)
        out[n] = torch.where(dd > 0, num / dd.clamp_min(torch.finfo(dd.dtype).tiny), torch.zeros_like(num))
    return out
