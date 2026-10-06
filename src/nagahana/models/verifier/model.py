"""The Verifier's learned parts in one module: PRM, trust value head, calibration policy head (build-spec section 2.10).

Holding them together gives one parameter count (build-spec section 4: "4 blocks x 1024 PRM + heads") and one
object whose weights change only under a HumanCommand (D-21): `gate.train_on_feedback` in training, or a
promoted candidate update of the feedback workflow (service.py, D-65). The non-learned parts (Monitor
statistics, calibration maths, rewards) are plain functions and classes beside it.
"""

from __future__ import annotations

from torch import nn

from nagahana.models.config.components import VerifierConfig
from nagahana.models.verifier.heads import CalibrationPolicyHead, TrustValueHead
from nagahana.models.verifier.prm import ProcessRewardModel


class VerifierNet(nn.Module):
    """PRM + trust value head + calibration policy head."""

    def __init__(self, cfg: VerifierConfig, *, d_context: int, d_hyp: int, d_state: int, n_techniques: int,
                 n_stages: int, window_seconds: float) -> None:
        super().__init__()
        self.prm = ProcessRewardModel(cfg, d_context=d_context, d_hyp=d_hyp, d_state=d_state, n_techniques=n_techniques,
                                      n_stages=n_stages, window_seconds=window_seconds)
        self.trust = TrustValueHead(cfg)
        self.calibration = CalibrationPolicyHead(cfg)
