"""Policy/value heads for the Forecaster and the Advisor (templates), and the coupling question (D-12).

Two agents, one latent space
----------------------------
- **Forecaster heads**: the adversary's policy π_A and value V_φ. They plan as the adversary would,
  by MPC-guided model-based RL [Q-28], over K steps × N samples [A-06], [A-14]. The Forecaster's
  receding-horizon objective (diagram 06):
      a^A_{t:t+H−1} = argmax E[ Σ_{k<H} γ^k r^A_{t+k} + γ^H V_φ(z_{t+H}) ],   H ≤ K
  What the modelled attacker optimises (r^A) is held (D-11c). P_inf(k) needs the definition of an
  infiltration state (D-03a).
- **Advisor heads**: the defender's policy π_D over D3FEND counter-measures, and its value. "another
  agent/policy-value pair that directly works on both env & imagination to produce counters"
  [A-15], [A-20]. Its objective aggregates over adversary policies (D-03c) and prices disruption
  (D-03b). Both are held.

The coupling question (D-12, the owner's open analysis [A-19] item 4)
---------------------------------------------------------------------
"if we keep the transformer and policy/value separate then we can penalize its intuition about
adversarial analysis & threat hunting, and forecasting separately otherwise we can conjoin"
- **JOINT**: one model trained end to end. Shared representation (as in MuZero, refs.md#L437). The
  risk is gradient conflict between analysis and forecasting objectives; mitigations such as
  gradient surgery exist.
- **SEPARATE**: heads read TAAFT features through a stop-gradient. Analysis and forecasting are
  penalised separately (the owner's stated wish). The risk is that the representation lacks what
  the heads need.
- **STAGED**: separate first (stage 5 trains heads on a frozen TAAFT), then a light joint
  fine-tune. The 6-stage pipeline already supports this [I-01].

`head_input` implements what each option *means*. Choosing one stays with the owner.
"""

from __future__ import annotations

import enum

import torch
from torch import nn

from nagahana.core.errors import NotBuiltYet
from nagahana.governance import decisions


class Coupling(enum.Enum):
    """Options for D-12."""

    JOINT = "joint"
    SEPARATE = "separate"
    STAGED = "staged"


def decided_coupling() -> Coupling:
    """The owner's decision for D-12; raises `DecisionHeld` until then."""
    d = decisions.require("taaft-policy-coupling")
    assert d.value is not None
    return Coupling(d.value)


def head_input(features: torch.Tensor, coupling: Coupling, *, joint_phase: bool = False) -> torch.Tensor:
    """TAAFT features as the heads should see them under a given coupling (explicit, no default).

    - JOINT: gradients from the heads flow into TAAFT.
    - SEPARATE: stop-gradient, so head losses cannot reshape TAAFT.
    - STAGED: stop-gradient unless `joint_phase` is True (the fine-tune at the end).
    """
    if coupling is Coupling.JOINT:
        return features
    if coupling is Coupling.SEPARATE:
        return features.detach()
    return features if joint_phase else features.detach()


class PolicyValueHead(nn.Module):
    """Template for both agents. `agent` is "forecaster" or "advisor"."""

    def __init__(self, *, agent: str, **config: object) -> None:
        super().__init__()
        if agent not in ("forecaster", "advisor"):
            raise ValueError("agent must be 'forecaster' or 'advisor'")
        self.agent = agent
        self.config = dict(config)

    def forward(self, *args: object, **kwargs: object) -> object:
        waits = ("D-12", "D-11c", "D-03a") if self.agent == "forecaster" else ("D-12", "D-03a", "D-03b", "D-03c")
        raise NotBuiltYet(f"{self.agent} policy/value heads (MPC-guided model-based RL)", waiting_on=waits)
