"""The Forecaster (formerly Renderer): TAAFT + policy/value heads → Imagination and forecasts.

Role (D-19, [A-14], [Q-28])
---------------------------
On each trigger (policy held, D-02), the Forecaster reads the Environment, runs TAAFT's lenses, and
writes the analysis cache (Imagination: belief/suspicion, energy, game state). Then the policy/value
heads imagine K future states over N samples, planning as the adversary would (MPC-guided
model-based RL). It returns a `ForecastBundle` (roles/contracts.py).

Implemented here: the infiltration-probability estimator
--------------------------------------------------------
Given N imagined paths, let τ_n be the first step at which path n reaches an infiltration state
(∞ if it never does within K). The Monte-Carlo estimate of the cumulative infiltration probability
is

    P̂_inf(k) = (1/N) Σ_n 𝟙[τ_n ≤ k],      k = 1 … K

It is non-decreasing in k by construction. "What counts as an infiltration state" is held
(D-03a), so this function takes the first-hit steps as input and does not decide them.

Why not average the paths: two different attack paths do not average into a path; averaging would
invent a trajectory nobody imagined (DESIGN_LOG setup plan, documentation example). The owner's
"average, median & mode of these forecast pathways" [A-06] applies to *summary statistics* (e.g. the
distribution of τ or of stage-at-k), which is what the bundle reports.
"""

from __future__ import annotations

import torch

from nagahana.core.errors import NotBuiltYet


def p_inf_from_first_hits(first_hit: torch.Tensor, horizon_k: int) -> torch.Tensor:
    """P̂_inf(k) for k = 1…K from first-hit steps τ_n (use a value > K for 'never').

    Parameters
    ----------
    first_hit: LongTensor [N], 1-based step of first infiltration per imagined path.
    horizon_k: K.

    Returns
    -------
    Tensor [K] of non-decreasing probabilities.
    """
    if first_hit.dim() != 1 or first_hit.numel() == 0:
        raise ValueError("first_hit must be a non-empty 1-D tensor")
    if horizon_k < 1:
        raise ValueError("horizon_k must be >= 1")
    ks = torch.arange(1, horizon_k + 1, device=first_hit.device)
    return (first_hit[None, :] <= ks[:, None]).float().mean(dim=1)


class Forecaster:
    """Role orchestrator (template)."""

    def trigger(self, k: int) -> object:
        """Run one Forecaster computation at trigger index k."""
        raise NotBuiltYet("Forecaster trigger (TAAFT → heads → ForecastBundle)", waiting_on=("D-02", "D-12", "D-03a"))
