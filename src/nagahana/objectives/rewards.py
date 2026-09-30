"""Process rewards: a reward for every imagined step, not only the end ([Q-24]; ARCH #24).

"both models need to have a process reward model idea but modified for each k states to tune them
for prediction every training iteration; but the rewards/signals (reward & return) for both of them
are very different; one visualizes what is the future of network & attack possibilities, while the
other very distinctly finds ways" [Q-24].

Forecaster (proposal logged 2026-09-28; attacker objective held, D-11c)
------------------------------------------------------------------------
    r_k = proper score of step k against what happened (training)  +  physics  +  energy
The modelled attacker's own objective r^A_k (goal progress, stealth, …) drives the adversary
policy. What it contains is held (D-11c).

Advisor (proposal logged 2026-09-28; pricing and aggregation held, D-03b, D-03c)
--------------------------------------------------------------------------------
    r^D_k = −c(a^D_k)  +  [Φ(ŝ_k) − γ·Φ(ŝ_{k+1})]  +  η·ΔH_k
- c: disruption cost (D-03b);
- Φ: a potential over states (energy-based; P-09 proposes the marginal reading E_θ(∅, s));
- ΔH_k: uncertainty removed by the step (information value).

Potential-based shaping (Ng, Harada & Russell, ICML 1999) adds F = γΦ(s') − Φ(s) and leaves the
optimal policy unchanged in infinite-horizon (or properly terminated) problems. Over a *finite*
horizon H, the shaped return differs by γ^H Φ(s_H) − Φ(s_0), which still depends on where the plan
ends. So with MPC the shaping also steers toward low-energy end states, unless the terminal potential
is set to zero. `potential_shaping` makes that choice explicit.
"""

from __future__ import annotations

from typing import Protocol

import torch

from nagahana.core.errors import NotBuiltYet


class ProcessReward(Protocol):
    """Reward for one imagined step."""

    def __call__(self, step: int, state: object, action: object, next_state: object) -> torch.Tensor:
        """r_k for one step."""
        ...


def potential_shaping(
    phi_s: torch.Tensor,
    phi_next: torch.Tensor,
    *,
    gamma: float,
    next_is_terminal: torch.Tensor,
    zero_terminal_potential: bool,
) -> torch.Tensor:
    """F = γ·Φ(s') − Φ(s), with an explicit choice about the terminal potential (see module docstring).

    If `zero_terminal_potential`, Φ(s') is treated as 0 where `next_is_terminal`, which keeps the
    optimum unchanged. If False, the terminal potential is kept and the shaping also steers toward
    low-potential end states.
    """
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be in [0, 1]")
    phi_n = torch.where(next_is_terminal, torch.zeros_like(phi_next), phi_next) if zero_terminal_potential else phi_next
    return gamma * phi_n - phi_s


def forecaster_step_reward(*_a: object, **_k: object) -> torch.Tensor:
    """Forecaster process reward (template)."""
    raise NotBuiltYet("Forecaster process reward", waiting_on=("D-11c", "D-03a"))


def advisor_step_reward(*_a: object, **_k: object) -> torch.Tensor:
    """Advisor process reward (template)."""
    raise NotBuiltYet("Advisor process reward", waiting_on=("D-03b", "D-03c", "D-11b"))
