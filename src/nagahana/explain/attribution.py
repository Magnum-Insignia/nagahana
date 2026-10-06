"""Field attributions: Expected Gradients (SHAP) and LIME, written for NagaHana's field states.

Why on field states, not raw values
-----------------------------------
A raw value can be absent (D-41) or categorical (a port is not a magnitude), so "x − baseline" over
raw values is often undefined. The FieldEncoder maps every cell, absent or present, to a field state
f_{i,c} ∈ ℝ^{d_field}. Attributions are computed over those states and summed over the embedding
axis, giving one number per (update, column): a contribution of a *named field* to the output.

1. Integrated and Expected Gradients
------------------------------------
Integrated Gradients (Sundararajan, Taly & Yan, ICML 2017, arXiv:1703.01365) for input x, baseline
x′ and a scalar output F:

    IG_j(x; x′) = (x_j − x′_j) · ∫₀¹ ∂F(x′ + α(x − x′)) / ∂x_j dα

satisfies *completeness*: Σ_j IG_j = F(x) − F(x′). IG is the Aumann–Shapley value of F, i.e. a
Shapley value for a continuous game.

Expected Gradients (Erion et al., "Improving performance of deep learning models with axiomatic
attribution priors and expected gradients", Nature Machine Intelligence 2021, arXiv:1906.10670)
averages IG over baselines drawn from a background distribution D:

    EG_j(x) = E_{x′∼D, α∼U(0,1)} [ (x_j − x′_j) · ∂F(x′ + α(x − x′)) / ∂x_j ]

which is the estimator behind SHAP's GradientExplainer (Lundberg & Lee, NeurIPS 2017). Completeness
holds in expectation: Σ_j EG_j = F(x) − E_{x′}[F(x′)]. `expected_gradients` returns the attributions
and this completeness gap, so every explanation carries its own check.

2. LIME over field groups
-------------------------
LIME (Ribeiro, Singh & Guestrin, KDD 2016, arXiv:1602.04938) fits a local weighted linear surrogate.
Here the "interpretable features" are field groups (e.g. all TCP-flag columns): a group is either
kept or replaced by its baseline state. With z ∈ {0,1}^G sampled uniformly, weights
π(z) = exp(−d(z)²/σ²) with d the fraction of groups switched off, the surrogate solves

    min_w Σ_s π(z_s) (F(x_{z_s}) − w₀ − wᵀ z_s)² + λ‖w‖²

in closed form (weighted ridge). LIME is reported beside EG as a cross-check, never alone: it is
known to be unstable across seeds (Alvarez-Melis & Jaakkola, 2018, arXiv:1806.08049), so the seed
and sample count are part of the output.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Attribution:
    """Per-column attributions and their own sanity check.

    values: [..., C] contributions (summed over the field-state width).
    completeness_gap: F(x) − E[F(baseline)] − Σ values (≈ 0 up to Monte-Carlo error for EG).
    method: "expected_gradients" | "lime".
    detail: method parameters (samples, seed …) for reproducibility.
    """

    values: torch.Tensor
    completeness_gap: torch.Tensor
    method: str
    detail: dict[str, float]


def expected_gradients(
    f: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    background: torch.Tensor,
    *,
    samples: int,
    generator: torch.Generator | None = None,
) -> Attribution:
    """Expected Gradients of a scalar-per-example function over field states.

    Parameters
    ----------
    f: maps field states [E, ..., C, d] to one scalar per example [E].
    x: the explained field states [E, ..., C, d].
    background: baseline field states [K, ..., C, d] (e.g. benign windows of the same site).
    samples: Monte-Carlo pairs (baseline, α) per example.

    Returns attributions [E, ..., C] (summed over d) and the completeness gap per example [E].
    """
    if samples < 1:
        raise ValueError("samples must be ≥ 1")
    e = x.shape[0]
    total = torch.zeros_like(x)
    for _ in range(samples):
        # One baseline and one α per example: an unbiased draw of the EG integrand.
        idx = torch.randint(0, background.shape[0], (e,), generator=generator)
        base = background[idx]
        alpha = torch.rand((e,) + (1,) * (x.dim() - 1), generator=generator, dtype=x.dtype)
        point = (base + alpha * (x - base)).detach().requires_grad_(True)
        (grad,) = torch.autograd.grad(f(point).sum(), point)
        total = total + (x - base) * grad
    attr = (total / samples).sum(dim=-1)                       # [E, ..., C]
    with torch.no_grad():
        fx = f(x)
        f_base = torch.stack([f(background[k : k + 1].expand_as(x[:1]).expand_as(x)) for k in range(background.shape[0])]).mean(0)
        gap = fx - f_base - attr.flatten(1).sum(dim=1)
    return Attribution(attr.detach(), gap, "expected_gradients", {"samples": float(samples)})


def lime_groups(
    f: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    baseline: torch.Tensor,
    groups: Sequence[Sequence[int]],
    *,
    samples: int,
    kernel_width: float = 0.25,
    ridge: float = 1e-3,
    generator: torch.Generator | None = None,
) -> Attribution:
    """LIME weights per column group for ONE example.

    x, baseline: field states [..., C, d] of one example; groups: column indices per group (a column
    may belong to one group only). Returns per-group weights as `values` [G] and the surrogate's
    residual at the full input as `completeness_gap`.
    """
    g = len(groups)
    z = (torch.rand(samples, g, generator=generator) > 0.5).float()
    z[0] = 1.0                                                  # include the unperturbed input
    col_dim = x.dim() - 2                                       # axis of columns in x
    outs = []
    with torch.no_grad():
        for s in range(samples):
            xs = x.clone()
            for gi, cols in enumerate(groups):
                if z[s, gi] == 0:
                    idx = torch.tensor(list(cols), dtype=torch.long)
                    xs.index_copy_(col_dim, idx, baseline.index_select(col_dim, idx))
            outs.append(f(xs.unsqueeze(0)).reshape(()))
    y = torch.stack(outs)                                       # [S]
    dist = 1.0 - z.mean(dim=1)                                  # fraction of groups switched off
    w = torch.exp(-(dist**2) / kernel_width**2)                 # LIME kernel π(z)
    design = torch.cat([torch.ones(samples, 1), z], dim=1)      # [S, 1 + G]
    wd = design * w[:, None]
    coef = torch.linalg.solve(design.t() @ wd + ridge * torch.eye(g + 1), wd.t() @ y)
    gap = y[0] - (coef[0] + coef[1:].sum())
    return Attribution(coef[1:], gap, "lime", {"samples": float(samples), "kernel_width": kernel_width})
