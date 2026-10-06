"""Forecaster: the adversary's policy/value world model over imagined routes (build-spec §2.8).

- `model.Forecaster`: context → route transformer → π_A, hazard, stage, value, process reward, ẑ;
  `imagine` (MPPI-guided routes → ForecastOut) and `to_bundle` (→ roles.contracts.ForecastBundle).
- `losses`: discrete-time survival NLL, teacher-forced dynamics, behaviour cloning, TD(λ), reward model.
- `routes`: the route mathematics (P_inf, merging, weights, MPPI, quantiles, CVaR).
"""

from nagahana.models.forecaster.model import EncodedContext, Forecaster, ForecastIntervention, MarginalEnergy

__all__ = ["EncodedContext", "ForecastIntervention", "Forecaster", "MarginalEnergy"]
