"""The Forecaster role: TAAFT and the policy/value heads at each trigger, into Imagination (D-19, [A-14], [Q-28]).

What the role does
------------------
At every trigger (the D-02 option in force: fixed cadence plus capped priority triggers under the working
option, AS-12; `memory/retention.TriggerRule`) the Forecaster reads the Environment, runs TAAFT's lenses, writes
the analysis cache (Imagination: belief, energy, game state) and imagines K future states along at most N routes
(D-46), planning as the adversary would (MPC-guided model-based RL). The computation is the inference engine's
(`inference/engine.Engine`: `_fire` runs TAAFT, `models.forecaster.Forecaster.imagine`, the Verifier's trust
and calibration, and writes Imagination as this role, D-35). `Forecaster.trigger(k)` is the role's handle on
it: the result of the cadence trigger k, computed if it is due.

The first-hit estimator
-----------------------
Given N imagined paths with first infiltration steps tau_n (the working option of D-03a, AS-18; "never within
K" written as a value > K),

    P_inf_hat(k) = (1 / N) sum_n [tau_n <= k],      k = 1 ... K,

non-decreasing in k. It is the special case of the Forecaster's hazard mixture
P_inf(k) = sum_n w_n (1 - prod_(j<=k) (1 - h_(n,j))) with hazards h_(n,j) = [j = tau_n] and weights w_n = 1 / N, so
`p_inf_from_first_hits` evaluates exactly that function (`models.forecaster.routes.p_inf_from_hazards`).
Different attack paths are not averaged into a path; summaries (mean, median, mode) are taken over route
statistics such as tau, never over trajectories.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from nagahana.core.errors import InvariantViolation
from nagahana.memory.retention import TriggerRule
from nagahana.models.forecaster.routes import p_inf_from_hazards

if TYPE_CHECKING:
    from nagahana.inference.engine import Engine, TriggerResult


def p_inf_from_first_hits(first_hit: torch.Tensor, horizon_k: int) -> torch.Tensor:
    """P_inf_hat(k) for k = 1 ... K from first-hit steps tau_n (a value > K for "never"); float64 [K].

    Parameters
    ----------
    first_hit: LongTensor [N], 1-based step of first infiltration per imagined path.
    horizon_k: K.
    """
    if first_hit.dim() != 1 or first_hit.numel() == 0:
        raise ValueError("first_hit must be a non-empty 1-D tensor")
    if horizon_k < 1:
        raise ValueError("horizon_k must be >= 1")
    ks = torch.arange(1, horizon_k + 1, device=first_hit.device)
    hazard = (first_hit[:, None] == ks[None, :]).to(torch.float64)                 # [N, K]: 1 at the first hit
    weight = torch.full((first_hit.numel(),), 1.0 / first_hit.numel(), dtype=torch.float64, device=first_hit.device)
    return p_inf_from_hazards(hazard, weight)


class Forecaster:
    """The Forecaster role over the inference engine (module docstring).

    Parameters
    ----------
    engine: the inference engine of the site (live or forensic replay).
    """

    def __init__(self, engine: Engine) -> None:
        self.engine = engine

    @property
    def rule(self) -> TriggerRule:
        """The trigger rule in force at the engine's cadence (D-02 option, `memory/retention.py`)."""
        return TriggerRule.in_force(self.engine.settings.cadence_s)

    def _cadence_result(self, tau: float) -> TriggerResult | None:
        for r in reversed(self.engine.results):
            if r.kind == "cadence" and r.time == tau:
                return r
        return None

    def trigger(self, k: int) -> TriggerResult:
        """The Forecaster computation at the cadence trigger k (tau = k c): returned if done, run if due.

        Running it fires every cadence trigger up to and including tau that is still due, in time order, as the
        engine does when the clock advances (no trigger is skipped). Raises `InvariantViolation` when no state
        update has been ingested yet, or when no window holds tau.
        """
        tau = k * self.engine.settings.cadence_s
        done = self._cadence_result(tau)
        if done is not None:
            return done
        if self.engine.next_grid is None:
            raise InvariantViolation("no state update has been ingested yet: there is nothing to forecast from")
        self.engine.advance_clock(math.nextafter(tau, math.inf))
        result = self._cadence_result(tau)
        if result is None:
            raise InvariantViolation(f"no trigger fired at {tau}: no processed window holds that time")
        return result

    def latest(self) -> TriggerResult | None:
        """The latest trigger result (cadence or priority), or None before the first trigger."""
        return self.engine.results[-1] if self.engine.results else None
