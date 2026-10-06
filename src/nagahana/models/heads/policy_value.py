"""How the policy/value heads of the Forecaster and the Advisor read TAAFT: the coupling of D-12.

Two agents, one latent space
----------------------------
- Forecaster heads (`models/forecaster/model.py`): the adversary's policy pi_A and value V_phi. They plan
  as the adversary would, by MPC-guided model-based RL ([Q-28]), over K steps along at most N routes
  (D-30, D-46). The receding-horizon objective:
      a^A_(t:t+H-1) = argmax E[ sum over k < H of gamma^k r^A_(t+k) + gamma^H V_phi(z_(t+H)) ],   H <= K
- Advisor heads (`models/advisor/model.py`): the defender's policy pi_D over D3FEND counter-measures and
  its value, re-imagined against the Forecaster's adversary ([A-15], [A-20]).
Both agents own their heads; this module holds what they share: how their inputs read TAAFT.

The coupling (D-12, held; option in force from `governance.decisions`)
----------------------------------------------------------------------
- JOINT: one model trained end to end; head losses reshape TAAFT (a shared representation, as in
  MuZero, Schrittwieser et al., Nature 2020). The risk is gradient conflict between analysis and
  forecasting objectives.
- SEPARATE: heads read TAAFT through a stop-gradient, so analysis and forecasting are penalised
  separately [A-19]; the risk is that the representation lacks what the heads need.
- STAGED (working option, AS-22): separate first (the first part of stage 5 trains the heads on a
  stop-gradient TAAFT), then a joint fine-tune.

`head_input` implements what each option means. `coupling_in_force` resolves the option through
`decisions.require("taaft-policy-coupling")`: the run's configured option, else the working option of
AS-22. `PolicyValueHead` is the handle on one agent's working heads that applies the coupling and switches
the STAGED phase.
"""

from __future__ import annotations

import enum

import torch
from torch import nn

from nagahana.core.errors import InvariantViolation
from nagahana.governance import decisions


class Coupling(enum.Enum):
    """The admissible options of D-12 (values equal the option strings of the decision registry)."""

    JOINT = "joint"
    SEPARATE = "separate"
    STAGED = "staged"


def coupling_in_force(configured: str | None = None) -> Coupling:
    """The D-12 option in force: `configured`, else the run's configured option, else STAGED (AS-22)."""
    d = decisions.require("taaft-policy-coupling", configured, by=__name__)
    assert d.value is not None
    return Coupling(d.value)


def decided_coupling() -> Coupling:
    """The coupling in force (kept name): the decided value of D-12 once decided, its configured option until then."""
    return coupling_in_force()


def assumed_coupling() -> Coupling:
    """The coupling the agents read TAAFT under (kept name of `coupling_in_force`, used by both agents)."""
    return coupling_in_force()


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


class PolicyValueHead:
    """The handle on one agent's working policy/value heads and the coupling they read TAAFT under.

    The heads themselves live in the agent's model (`Forecaster`, `Advisor`); both read TAAFT through
    `head_input(..., coupling_in_force(), joint_phase=module.joint_phase)`. This handle gives training
    one place to read the option in force and to switch the STAGED phase of either agent.

    Parameters
    ----------
    agent: "forecaster" or "advisor".
    module: the agent's model; it must expose the boolean attribute `joint_phase`.
    """

    AGENTS = ("forecaster", "advisor")

    def __init__(self, *, agent: str, module: nn.Module) -> None:
        if agent not in self.AGENTS:
            raise ValueError(f"agent must be one of {self.AGENTS}")
        if not isinstance(getattr(module, "joint_phase", None), bool):
            raise InvariantViolation(f"the {agent} module must expose a boolean `joint_phase` (the STAGED switch)")
        self.agent = agent
        self.module = module

    @property
    def coupling(self) -> Coupling:
        """The D-12 option in force."""
        return coupling_in_force()

    @property
    def joint_phase(self) -> bool:
        """Whether the STAGED coupling is in its joint fine-tune."""
        return bool(getattr(self.module, "joint_phase"))

    def set_joint_phase(self, joint: bool) -> None:
        """Switch the STAGED phase: False reads TAAFT through a stop-gradient, True fine-tunes jointly.

        Meaningful only under STAGED; under JOINT or SEPARATE the option alone decides the gradient path,
        so a switch request there raises rather than doing nothing silently.
        """
        if self.coupling is not Coupling.STAGED:
            raise InvariantViolation(f"the joint phase applies to the STAGED coupling; the option in force is {self.coupling.value!r}")
        setattr(self.module, "joint_phase", bool(joint))

    def read(self, features: torch.Tensor) -> torch.Tensor:
        """TAAFT features as this agent's heads see them under the coupling in force."""
        return head_input(features, self.coupling, joint_phase=self.joint_phase)
