"""Rewards of imagined steps: a reward for every step, not only the end ([Q-24]).

Both agents get a process reward per imagined step, and the two differ in kind: the Forecaster's measures how
faithfully it imagines what happens, the Advisor's how well a counter-measure step serves the defence.

The modelled adversary's reward (AS-17, working option "realistic" of the held D-11c)
------------------------------------------------------------------------------------
    r^A_k = (progress_k - progress_(k-1)) + beta [infiltration at k] - kappa exposure_k
drives the adversary's policy pi_A in the Forecaster (`adversary_reward`; trained in
`models/forecaster/losses.py`). The infiltration state is the working option of D-03a (AS-18).

The Forecaster's process reward (architecture, Forecaster process rewards)
-------------------------------------------------------------------------
Where the realised future is known (training, the attack timelines of the datasets), imagined step k scores
    r^F_k = -S(p_hat_(t+k), y_(t+k)) - lambda_phys Phi_phys(x_hat_(t+k)) - lambda_E E(z_<=t, z_hat_(t+k)),
with S a strictly proper scoring loss of the step's forecast against what happened (`step_score`,
`objectives/scoring.py`), Phi_phys the shared physics term on the decoded imagined step (`physics/term.py`) and E
TAAFT's conditional energy of the imagined hypothesis (`models/taaft`): fidelity, physical validity and
consistency with the history. `forecaster_step_reward` composes the three per-step quantities that the working
components compute; lambda_phys is the one weight of the shared physics term (`TAAFTConfig.lambda_phys`).

The Advisor's step reward (architecture, the Advisor)
-----------------------------------------------------
    r^D_k = -c(a^D_k) + [E(empty, s_k) - gamma E(empty, s_(k+1))] + eta Delta H_k
the cost of the step (disruption pricing, AS-23 / working option of D-03b; `models/advisor/effects.py`), a
potential-based shaping towards normal states built on TAAFT's marginal energy (the potential is
Phi(s) = -E(empty, s); P-09 reading, Advisor shaping under the working option of D-11b, AS-16), and the
information value of the step (`models/advisor/effects.information_value`). `advisor_step_reward` composes
them; `advisor_objective` is the plan-level objective J = -Delta P_inf - kappa cost used for ranking.

Potential-based shaping (Ng, Harada and Russell, ICML 1999) adds F = gamma Phi(s') - Phi(s) and leaves the
optimal policy unchanged in infinite-horizon (or properly terminated) problems. Over a finite horizon H the
shaped return differs by gamma^H Phi(s_H) - Phi(s_0), which still depends on where the plan ends, so with MPC
the shaping also steers toward low-energy end states unless the terminal potential is set to zero.
`potential_shaping` makes that choice explicit.
"""

from __future__ import annotations

import torch

from nagahana.objectives.scoring import brier, brier_categorical, log_score, log_score_categorical

#: Proper scoring rules accepted by `step_score`.
SCORING_RULES: tuple[str, ...] = ("log", "brier")


def potential_shaping(
    phi_s: torch.Tensor,
    phi_next: torch.Tensor,
    *,
    gamma: float,
    next_is_terminal: torch.Tensor,
    zero_terminal_potential: bool,
) -> torch.Tensor:
    """F = gamma Phi(s') - Phi(s), with an explicit choice about the terminal potential (module docstring).

    If `zero_terminal_potential`, Phi(s') is treated as 0 where `next_is_terminal`, which keeps the optimum
    unchanged. If False, the terminal potential is kept and the shaping also steers toward low-potential end
    states.
    """
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be in [0, 1]")
    phi_n = torch.where(next_is_terminal, torch.zeros_like(phi_next), phi_next) if zero_terminal_potential else phi_next
    return gamma * phi_n - phi_s


def adversary_reward(
    progress_prev: torch.Tensor,
    progress_next: torch.Tensor,
    infiltrated: torch.Tensor,
    exposure: torch.Tensor,
    *,
    beta: float,
    kappa: float,
) -> torch.Tensor:
    """The modelled adversary's step reward (AS-17, working option of the held D-11c):

        r^A_k = (progress_k - progress_(k-1)) + beta [infiltration at k] - kappa exposure_k

    progress: `vocab.STAGE_PROGRESS` of the stage before and after the step (kill-chain position in [0, 1]);
    infiltrated: {0, 1} (AS-18 infiltration state reached at k); exposure >= 0: marginal-energy novelty of the
    step (AS-252; 0 when TAAFT's marginal energy is not available). beta and kappa are
    `ForecasterConfig.infiltration_bonus` and `.exposure_cost`.
    """
    from nagahana.governance.assumptions import assume  # local: keep this module import-light

    assume("AS-17", by=__name__)
    return (progress_next - progress_prev) + beta * infiltrated.to(progress_next.dtype) - kappa * exposure.to(progress_next.dtype)


def advisor_objective(delta_p_inf: torch.Tensor, cost: torch.Tensor, *, kappa: float) -> torch.Tensor:
    """The Advisor's objective J = -Delta P_inf - kappa cost (AS-23 pricing, AS-24 risk measure).

    Delta P_inf = P_inf(K | counter) - P_inf(K | none) (negative = risk reduced), or its CVaR over routes;
    cost = disruption cost (criticality x disruption weight). Higher J is better. kappa is
    `AdvisorConfig.cost_weight`.
    """
    from nagahana.governance.assumptions import assume

    assume("AS-23", by=__name__)
    return -delta_p_inf - kappa * cost


def step_score(forecast: torch.Tensor, outcome: torch.Tensor, *, rule: str, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    """S of each imagined step's forecast against what happened, and the mask of known outcomes.

    forecast: either probabilities [..., K] of a categorical outcome (e.g. the stage posterior per step) with
    outcome long [...] (class index, -1 unknown), or event probabilities [...] (e.g. the hazard of the step) with
    outcome float [...] in {0, 1} (NaN unknown). rule: "log" or "brier" (both strictly proper). Returns
    (score [...] with 0 where the outcome is unknown, known bool [...]); unknown outcomes never score (D-41).
    """
    if rule not in SCORING_RULES:
        raise ValueError(f"rule must be one of {SCORING_RULES}")
    if forecast.dim() == outcome.dim() + 1:
        known = outcome >= 0
        safe = torch.where(known, outcome, torch.zeros_like(outcome)).long()
        s = log_score_categorical(forecast, safe, eps=eps) if rule == "log" else brier_categorical(forecast, safe)
    elif forecast.shape == outcome.shape:
        known = ~torch.isnan(outcome)
        safe_y = torch.where(known, outcome, torch.zeros_like(outcome))
        s = log_score(forecast, safe_y, eps=eps) if rule == "log" else brier(forecast, safe_y)
    else:
        raise ValueError(f"forecast {tuple(forecast.shape)} does not match outcome {tuple(outcome.shape)}")
    return torch.where(known, s, torch.zeros_like(s)), known


def forecaster_step_reward(
    score: torch.Tensor,
    physics: torch.Tensor,
    energy: torch.Tensor,
    *,
    lambda_phys: float,
    lambda_energy: float,
) -> torch.Tensor:
    """r^F_k = -S_k - lambda_phys Phi_k - lambda_E E_k (module docstring), elementwise over imagined steps.

    score [...]: the proper score of each step (`step_score`); physics [...]: Phi_phys of the decoded imagined
    step (>= 0, `physics.term.PhysicsTerm.per_row`); energy [...]: TAAFT's conditional energy of the step's
    hypothesis. lambda_phys, lambda_energy >= 0. The result has the promoted dtype of the inputs (float64 when
    any input is a D-54 output).
    """
    if lambda_phys < 0 or lambda_energy < 0:
        raise ValueError("the weights of the physics and energy terms must be >= 0")
    if bool((physics < 0).any()):
        raise ValueError("Phi_phys is >= 0 by definition")
    return -score - lambda_phys * physics - lambda_energy * energy


def advisor_step_reward(
    cost: torch.Tensor,
    energy_now: torch.Tensor,
    energy_next: torch.Tensor,
    information_gain: torch.Tensor,
    *,
    gamma: float,
    eta: float,
    next_is_terminal: torch.Tensor,
    zero_terminal_potential: bool,
) -> torch.Tensor:
    """r^D_k = -c_k + [E(empty, s_k) - gamma E(empty, s_(k+1))] + eta Delta H_k (module docstring).

    cost >= 0: disruption cost of the step; energy_now, energy_next: TAAFT's marginal energy of the imagined
    state before and after the step; information_gain: Delta H_k, the entropy the step removes; eta >= 0. The
    shaping is `potential_shaping` with Phi = -E(empty, .), so its terminal choice is explicit here too.
    """
    if eta < 0:
        raise ValueError("eta must be >= 0")
    if bool((cost < 0).any()):
        raise ValueError("a disruption cost is >= 0")
    shaping = potential_shaping(-energy_now, -energy_next, gamma=gamma, next_is_terminal=next_is_terminal,
                                zero_terminal_potential=zero_terminal_potential)
    return -cost + shaping + eta * information_gain
