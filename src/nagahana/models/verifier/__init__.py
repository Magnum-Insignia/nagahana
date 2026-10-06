"""Verifier: regulator and calibrator; weights and temperatures change only on a HumanCommand (D-21, D-45, D-65).

Learned parts and statistics (build-spec section 2.10)

- `prm.ProcessRewardModel`: per-step plausibility of imagined routes.
- `heads`: trust value head P(correct | forecast, Monitor) and calibration policy head (log T per family).
- `monitor.Monitor`: Welford and Page-Hinkley per memory region, residual CUSUM, systematic gap, alerts.
- `calibration`: temperature, exact ML temperature (bisection), split-conformal thresholds, reports.
- `model.VerifierNet`: the learned parts together.
- `gate`: `require_command`, `apply_calibration`, `train_on_feedback`, the command actions and ids.

Feedback learning (D-65): RLHF, RLVR and RLCD behind one interface, fit(batch) -> candidate update

- `feedback`: typed, validated, content-addressed feedback events with provenance.
- `ledger.FeedbackLedger`: the hash-chained, persisted record of feedback and every audit step.
- `situations`: the model records (analysis, deployed forecast, Monitor statistics) feedback refers to.
- `params`: candidate deltas apart from the reference weights; exact promotion and rollback.
- `policies`: per-step log-probabilities, exact ancestral sampling and KL of the Forecaster and Advisor.
- `rewards`: verifiable rewards (proper scores, censoring, responded-to weighting).
- `bradley_terry`, `rlhf`: the preference reward model and DPO.
- `rlvr`: GRPO over imagined routes with verifiable rewards.
- `rlcd`: the calibrated-decision objective (Brier reward, RLCD temperatures, trust and calibration heads).
- `candidates`, `evaluation`: candidate updates, their held-out evaluation and promotion gates.
- `service.FeedbackService`: ingest, fit, evaluate, promote, apply, roll back, all under the gate.
- `online.OnlineFeedback`: the same objectives on the live weights for full training.
- `config`: the dataclass configuration; `cli.register_cli`: the command line.
"""

from nagahana.models.verifier.candidates import CandidateUpdate, FitReport
from nagahana.models.verifier.config import FeedbackLearningConfig
from nagahana.models.verifier.evaluation import EvaluationReport, evaluate_candidate
from nagahana.models.verifier.feedback import (
    AlertFeedback,
    OutcomeConfirmation,
    PlanSpec,
    PreferenceFeedback,
    PreferenceItem,
    Provenance,
    RouteSpec,
    StageCorrection,
)
from nagahana.models.verifier.gate import apply_calibration, command_id, require_command, train_on_feedback
from nagahana.models.verifier.learning import FeedbackBatch, split_batch
from nagahana.models.verifier.ledger import FeedbackLedger
from nagahana.models.verifier.model import VerifierNet
from nagahana.models.verifier.monitor import Monitor
from nagahana.models.verifier.online import OnlineFeedback
from nagahana.models.verifier.rlcd import RLCDLearner
from nagahana.models.verifier.rlhf import RLHFLearner
from nagahana.models.verifier.rlvr import RLVRLearner
from nagahana.models.verifier.service import FeedbackService
from nagahana.models.verifier.situations import Situation, trigger_situation

__all__ = [
    "AlertFeedback", "CandidateUpdate", "EvaluationReport", "FeedbackBatch", "FeedbackLearningConfig", "FeedbackLedger",
    "FeedbackService", "FitReport", "Monitor", "OnlineFeedback", "OutcomeConfirmation", "PlanSpec", "PreferenceFeedback",
    "PreferenceItem", "Provenance", "RLCDLearner", "RLHFLearner", "RLVRLearner", "RouteSpec", "Situation", "StageCorrection",
    "VerifierNet", "apply_calibration", "command_id", "evaluate_candidate", "require_command", "split_batch",
    "train_on_feedback", "trigger_situation",
]
