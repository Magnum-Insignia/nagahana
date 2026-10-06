"""A stand-in `AnalysisOut` with the documented shapes, for tests and smoke runs of the Forecaster,
Advisor and Verifier while TAAFT is built in parallel (engineer C).

It is *not* TAAFT: values are random and carry no meaning. It only honours the tensor contract of
`models/batch.py::AnalysisOut`:

    context [B, M, V+G, d_context], token_mask [B, M, V+G], y [B, M, V+G, d_y],
    readouts: compromise [B, M, V] ∈ (φ, 1), stage [B, M, V, 15] (probabilities), trust [B, M, V].

Inactive entities are masked per trigger (a share `inactive` of them), adversary slots are always
active. `marginal_energy_stub` is a fixed quadratic E(∅, y) = ½‖y‖²/d_y — a stand-in with the
callable signature the Forecaster expects, not TAAFT's energy.
"""

from __future__ import annotations

import torch

from nagahana.models.batch import AnalysisOut
from nagahana.models.vocab import N_STAGES


def fake_analysis(
    *,
    batch: int = 2,
    triggers: int = 3,
    entities: int = 8,
    adversary_slots: int = 4,
    d_context: int = 64,
    d_hyp: int = 16,
    n_goals: int = 4,
    inactive: float = 0.25,
    seed: int = 0,
    requires_grad: bool = False,
) -> AnalysisOut:
    """Random tensors in the documented shapes (see module docstring)."""
    g = torch.Generator().manual_seed(seed)
    b, m, v, ga = batch, triggers, entities, adversary_slots
    ctx = torch.randn(b, m, v + ga, d_context, generator=g)
    y = torch.randn(b, m, v + ga, d_hyp, generator=g)
    if requires_grad:
        ctx.requires_grad_(True)
        y.requires_grad_(True)
    active = torch.rand(b, m, v, generator=g) >= inactive
    active[..., 0] = True                                       # at least one active entity per trigger
    token_mask = torch.cat([active, torch.ones(b, m, ga, dtype=torch.bool)], dim=-1)
    floor = 0.01
    comp = floor + (1 - floor) * torch.sigmoid(torch.randn(b, m, v, generator=g))
    stage = torch.softmax(torch.randn(b, m, v, N_STAGES, generator=g), dim=-1)
    trust = torch.sigmoid(torch.randn(b, m, v, generator=g))
    readouts = {
        "compromise": comp, "stage": stage, "trust": trust,
        "malignity": torch.rand(b, m, v + ga, generator=g),
        "goal": torch.softmax(torch.randn(b, m, n_goals, generator=g), -1),
    }
    return AnalysisOut(
        context=ctx, token_mask=token_mask, imagination_kv=[], y0=y.detach().clone(), y=y,
        energy_trace=torch.zeros(1), lens_energy={}, lens_share={}, readouts=readouts, passes=1, descent_steps=0,
    )


def marginal_energy_stub(y: torch.Tensor) -> torch.Tensor:
    """E(∅, y) = ½‖y‖²/d_y: a fixed stand-in with TAAFT's marginal-energy callable signature [..., d_y] → [...]."""
    return 0.5 * (y.float() ** 2).mean(dim=-1)
