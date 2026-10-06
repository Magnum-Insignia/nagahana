"""The Advisor role: counter-measure sequences, advisory only (D-01, D-19, D-33).

The Advisor is a second policy/value agent that works on both the Environment and Imagination to produce counters
([A-15], [A-20]); its actions are D3FEND-based counter-measure sequences at graph or sensor level, and its output
is only advisory (D-33). It keeps no memory of its own (D-35) and relates to the Forecaster like a GAN in function
only: it searches for counters to the forecast adversary and never changes the Forecaster.

The search is the working Advisor's (`models/advisor/model.Advisor.advise`): propose sequences by pi_D,
re-imagine each against the Forecaster's adversary with common random numbers, rank by the risk measure of the
held D-03c (working option CVaR_0.2, AS-24), the disruption pricing of the held D-03b (AS-23) and physical
feasibility, at the infiltration state of the held D-03a (AS-18); see AS-254 ... AS-261. The step reward of a
sequence is `objectives.rewards.advisor_step_reward`. On a site the inference engine runs it on demand
(`inference/engine.Engine.advise`) at a trigger's analysis; `Advisor` is the role's handle on that.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nagahana.inference.engine import Engine, TriggerResult
    from nagahana.roles.contracts import AdvisoryBundle


class Advisor:
    """The Advisor role over the inference engine (module docstring). Output: `AdvisoryBundle` (always advisory).

    Parameters
    ----------
    engine: the inference engine of the site.
    """

    def __init__(self, engine: Engine) -> None:
        self.engine = engine

    def advise(self, result: TriggerResult | None = None, *, generator_seed: int = 0) -> tuple[AdvisoryBundle, dict[str, object]]:
        """Ranked counter-measure sequences at a trigger (default: the latest) and the search's diagnostics."""
        return self.engine.advise(result, generator_seed=generator_seed)
