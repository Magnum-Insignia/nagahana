"""Configuration of the Verifier's feedback learning: RLHF, RLVR and RLCD under the human gate (D-21, D-65).

Typed frozen dataclasses are the single source of configuration (D-57). Every field is declared with
`core.config.setting(default, source=..., doc=...)`; its source names the decision (D-xx) or the
assumption (AS-830 ... AS-849, docs/assumptions/verifier-feedback.md) behind the default. The YAML view
is generated from these classes (`feedback_config_yaml`, rendered by `core.config.render_yaml`) and an
override file is validated against them (`feedback_config_from_mapping`, `core.config.from_mapping`).

Sections

    LedgerConfig            the hash-chained feedback and audit ledger (AS-830)
    AdapterConfig           which parameters a candidate update may change, dense or low rank (AS-835)
    OptimiserConfig         AdamW on the candidate delta with decoupled decay toward the reference
    BradleyTerryConfig      reward model of analyst preferences (AS-833)
    DPOConfig               direct preference optimisation of the Forecaster and Advisor policies (AS-834)
    OutcomeRewardConfig     verifiable rewards from confirmed outcomes (AS-839, AS-840)
    GRPOConfig              group-relative policy optimisation over imagined routes (AS-836 ... AS-838)
    RLCDConfig              the calibrated-decision objective (AS-841)
    EvaluationConfig        held-out evaluation and promotion gates (AS-842, AS-843, AS-845)
    FeedbackLearningConfig  all of the above plus the seed (AS-848)

Precision (D-54, AS-849): every loss, reward, advantage, KL value and score is computed in float64; the
candidate deltas are stored in the dtype of the weights they change (float32).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from nagahana.core.config import from_mapping, render_yaml, setting
from nagahana.core.errors import InvariantViolation
from nagahana.models.verifier.reports import OUTPUT_FAMILIES

#: Policies a preference can train (item kind "forecast" and "route_set" train the Forecaster, "advisory" the Advisor).
POLICY_TARGETS: tuple[str, ...] = ("forecaster", "advisor")
#: Proper scoring rules of the infiltration forecast (AS-839).
P_INF_SCORES: tuple[str, ...] = ("log", "brier")
#: Stage rewards at confirmed steps: correctness of the route's stage, or a proper score of its posterior (AS-839).
STAGE_SCORES: tuple[str, ...] = ("correct", "log", "brier")
#: Handling of right-censored outcomes by the Brier reward (AS-839).
BRIER_CENSORING: tuple[str, ...] = ("ipcw", "exclude")
#: Handling of responded-to outcomes (AS-840).
RESPONDED_TO: tuple[str, ...] = ("exclude", "inverse-propensity")
#: KL penalty to the reference policy (AS-837).
KL_ESTIMATORS: tuple[str, ...] = ("analytic", "k3")
#: RLCD reward (AS-841): the Brier reward of P-10, or the log score.
RLCD_SCORES: tuple[str, ...] = ("brier", "log")


def _check_choice(name: str, value: str, allowed: tuple[str, ...]) -> None:
    # A string option outside its admissible set is a configuration error, never a silent fallback.
    if value not in allowed:
        raise InvariantViolation(f"{name} must be one of {list(allowed)}, got {value!r}")


def _check_positive(name: str, value: float) -> None:
    if not (math.isfinite(value) and value > 0):
        raise InvariantViolation(f"{name} must be a positive finite number, got {value!r}")


def _check_probability_open(name: str, value: float) -> None:
    if not 0.0 < value < 1.0:
        raise InvariantViolation(f"{name} must lie in (0, 1), got {value!r}")


@dataclass(frozen=True)
class LedgerConfig:
    """The hash-chained, append-only ledger of feedback and audit records (AS-830)."""

    fsync: bool = setting(True, source="AS-830", doc="force every appended record to stable storage before the call returns")


@dataclass(frozen=True)
class AdapterConfig:
    """Which parameters a candidate update may change, and how the change is parameterised (AS-835).

    A pattern is an fnmatch glob over dotted parameter names of the target module (for example
    "tech_head.*" matches tech_head.weight and tech_head.bias). With rank 0 every selected parameter gets a
    dense delta; with rank r > 0 every selected two-dimensional weight gets the low-rank delta
    (alpha / r) B A of Hu et al. (ICLR 2022, arXiv:2106.09685) and the others stay dense.
    """

    forecaster_targets: tuple[str, ...] = setting(("tech_head.*", "ptr_q.*", "ptr_k.*", "null_target_logit"), source="AS-835",
                                                 doc="Forecaster parameters a candidate may change: the adversary policy heads (technique head, target pointer)")
    advisor_targets: tuple[str, ...] = setting(("action_head.*", "ptr_q.*", "ptr_k.*"), source="AS-835",
                                              doc="Advisor parameters a candidate may change: the defender policy heads (action head, target pointer)")
    verifier_targets: tuple[str, ...] = setting(("trust.*", "calibration.*"), source="AS-835, D-45",
                                               doc="Verifier parameters an RLCD candidate may change: the trust value head and the calibration policy head")
    rank: int = setting(0, source="AS-835", doc="0 = dense deltas; r > 0 = low-rank deltas (alpha / r) B A on two-dimensional weights")
    alpha: float = setting(32.0, source="AS-835, AS-26", doc="scale alpha of low-rank deltas (the site-adapter value of AS-26)")

    def __post_init__(self) -> None:
        if self.rank < 0:
            raise InvariantViolation("AdapterConfig.rank must be >= 0")
        _check_positive("AdapterConfig.alpha", self.alpha)
        for name in ("forecaster_targets", "advisor_targets", "verifier_targets"):
            pats = getattr(self, name)
            if not pats or any(not p for p in pats):
                raise InvariantViolation(f"AdapterConfig.{name} must hold at least one non-empty pattern")


@dataclass(frozen=True)
class OptimiserConfig:
    """AdamW on the candidate delta (Loshchilov and Hutter, ICLR 2019, arXiv:1711.05101).

    The decoupled weight decay acts on the delta, so it pulls the candidate toward the reference
    weights (the L2-SP regulariser of Li, Grandvalet and Davoine, ICML 2018, arXiv:1802.01483).
    """

    lr: float = setting(5e-5, source="AS-834", doc="AdamW learning rate of the delta")
    weight_decay: float = setting(0.01, source="AS-834", doc="decoupled decay of the delta toward zero, i.e. of the candidate toward the reference")
    beta1: float = setting(0.9, source="AS-834", doc="AdamW first-moment decay")
    beta2: float = setting(0.999, source="AS-834", doc="AdamW second-moment decay")
    eps: float = setting(1e-8, source="AS-834", doc="AdamW epsilon")
    grad_clip: float = setting(1.0, source="AS-834", doc="global gradient-norm clip of the delta")
    steps: int = setting(200, source="AS-834", doc="optimiser steps of one fit")
    batch_size: int = setting(16, source="AS-834", doc="examples (preferences, triggers, decisions) per optimiser step")

    def __post_init__(self) -> None:
        _check_positive("OptimiserConfig.lr", self.lr)
        _check_positive("OptimiserConfig.eps", self.eps)
        _check_positive("OptimiserConfig.grad_clip", self.grad_clip)
        if not (math.isfinite(self.weight_decay) and self.weight_decay >= 0):
            raise InvariantViolation("OptimiserConfig.weight_decay must be finite and >= 0")
        for name in ("beta1", "beta2"):
            if not 0.0 <= getattr(self, name) < 1.0:
                raise InvariantViolation(f"OptimiserConfig.{name} must lie in [0, 1)")
        if self.steps < 1 or self.batch_size < 1:
            raise InvariantViolation("OptimiserConfig.steps and batch_size must be >= 1")


@dataclass(frozen=True)
class BradleyTerryConfig:
    """Bradley-Terry reward model of analyst preferences, fitted by MAP / maximum likelihood (AS-833)."""

    l2: float = setting(1e-2, source="AS-833", doc="precision lambda of the Gaussian prior on the reward parameters (0 = maximum likelihood)")
    max_iter: int = setting(100, source="AS-833", doc="Newton iterations at most")
    tol: float = setting(1e-10, source="AS-833", doc="stop when the gradient's infinity norm is below this")
    armijo: float = setting(1e-4, source="AS-833", doc="sufficient-decrease constant of the backtracking line search")
    backtrack: float = setting(0.5, source="AS-833", doc="step shrink factor of the backtracking line search")

    def __post_init__(self) -> None:
        if not (math.isfinite(self.l2) and self.l2 >= 0):
            raise InvariantViolation("BradleyTerryConfig.l2 must be finite and >= 0")
        if self.max_iter < 1:
            raise InvariantViolation("BradleyTerryConfig.max_iter must be >= 1")
        _check_positive("BradleyTerryConfig.tol", self.tol)
        _check_probability_open("BradleyTerryConfig.armijo", self.armijo)
        _check_probability_open("BradleyTerryConfig.backtrack", self.backtrack)


@dataclass(frozen=True)
class DPOConfig:
    """Direct preference optimisation (Rafailov et al., NeurIPS 2023, arXiv:2305.18290; AS-834)."""

    beta: float = setting(0.1, source="AS-834", doc="beta of the implicit reward beta log pi / pi_ref")
    optimiser: OptimiserConfig = setting(OptimiserConfig(), source="AS-834", doc="optimiser of the DPO delta")

    def __post_init__(self) -> None:
        _check_positive("DPOConfig.beta", self.beta)


@dataclass(frozen=True)
class OutcomeRewardConfig:
    """Verifiable rewards from confirmed outcomes, with censoring and responded-to handling (AS-839, AS-840)."""

    p_inf_score: str = setting("log", source="AS-839", doc="log: right-censored survival log-likelihood; brier: negative Brier score of P_inf(K)")
    stage_score: str = setting("correct", source="AS-839", doc="reward of the route's stage at confirmed steps: correct (0/1), log or brier")
    stage_weight: float = setting(1.0, source="AS-839", doc="weight of the mean stage reward beside the infiltration score")
    brier_censoring: str = setting("ipcw", source="AS-839", doc="ipcw: inverse-probability-of-censoring weights (Kaplan-Meier); exclude: drop unknown outcomes only")
    responded_to: str = setting("exclude", source="AS-840, Q-38", doc="exclude responded-to outcomes, or also weight scored ones by 1 / (1 - response propensity)")
    max_ipw_weight: float = setting(20.0, source="AS-840", doc="truncation of inverse-propensity and inverse-censoring weights")
    log_eps: float = setting(1e-6, source="AS-839", doc="probabilities are clipped to [eps, 1 - eps] inside logarithms (the hazard NLL guard)")

    def __post_init__(self) -> None:
        _check_choice("OutcomeRewardConfig.p_inf_score", self.p_inf_score, P_INF_SCORES)
        _check_choice("OutcomeRewardConfig.stage_score", self.stage_score, STAGE_SCORES)
        _check_choice("OutcomeRewardConfig.brier_censoring", self.brier_censoring, BRIER_CENSORING)
        _check_choice("OutcomeRewardConfig.responded_to", self.responded_to, RESPONDED_TO)
        if not (math.isfinite(self.stage_weight) and self.stage_weight >= 0):
            raise InvariantViolation("OutcomeRewardConfig.stage_weight must be finite and >= 0")
        if not (math.isfinite(self.max_ipw_weight) and self.max_ipw_weight >= 1.0):
            raise InvariantViolation("OutcomeRewardConfig.max_ipw_weight must be finite and >= 1")
        if not 0.0 < self.log_eps < 0.5:
            raise InvariantViolation("OutcomeRewardConfig.log_eps must lie in (0, 0.5)")


@dataclass(frozen=True)
class GRPOConfig:
    """Group-relative policy optimisation over imagined routes (Shao et al., arXiv:2402.03300; AS-836 ... AS-838)."""

    group_size: int = setting(16, source="AS-836", doc="routes G drawn from pi_old per trigger (one group)")
    clip_eps: float = setting(0.2, source="AS-836", doc="epsilon of the clipped ratio (Schulman et al., arXiv:1707.06347)")
    kl_beta: float = setting(0.04, source="AS-836", doc="weight of the KL penalty to the reference (the value of Shao et al.)")
    kl_estimator: str = setting("analytic", source="AS-837", doc="analytic: exact per-step KL on visited prefixes; k3: the sampled estimator of Shao et al.")
    iterations: int = setting(20, source="AS-836", doc="outer iterations: fresh groups from the current candidate, which becomes pi_old")
    inner_epochs: int = setting(4, source="AS-836", doc="optimiser passes over each iteration's groups (mu)")
    triggers_per_iteration: int = setting(16, source="AS-836", doc="triggers whose groups are drawn per iteration (all if fewer)")
    advantage_eps: float = setting(1e-6, source="AS-838", doc="a group whose reward spread is below this has zero advantages")
    optimiser: OptimiserConfig = setting(OptimiserConfig(lr=1e-5, steps=1, batch_size=4), source="AS-836",
                                         doc="optimiser of the GRPO delta (steps is unused: iterations x inner epochs x batches)")

    def __post_init__(self) -> None:
        if self.group_size < 2:
            raise InvariantViolation("GRPOConfig.group_size must be >= 2 (advantages are relative to the group)")
        _check_probability_open("GRPOConfig.clip_eps", self.clip_eps)
        if not (math.isfinite(self.kl_beta) and self.kl_beta >= 0):
            raise InvariantViolation("GRPOConfig.kl_beta must be finite and >= 0")
        _check_choice("GRPOConfig.kl_estimator", self.kl_estimator, KL_ESTIMATORS)
        if self.iterations < 1 or self.inner_epochs < 1 or self.triggers_per_iteration < 1:
            raise InvariantViolation("GRPOConfig.iterations, inner_epochs and triggers_per_iteration must be >= 1")
        _check_positive("GRPOConfig.advantage_eps", self.advantage_eps)


@dataclass(frozen=True)
class RLCDConfig:
    """Reinforcement learning for calibrated decisions (P-10, AS-25; AS-841)."""

    score: str = setting("brier", source="AS-841, P-10", doc="reward of a stated probability: brier (P-10) or log")
    families: tuple[str, ...] = setting(OUTPUT_FAMILIES, source="D-45, AS-841", doc="output families whose temperature RLCD may propose")
    min_pairs: int = setting(50, source="AS-841, AS-26", doc="scored pairs a family needs before its temperature is proposed")
    min_decisions: int = setting(50, source="AS-841", doc="resolved decisions needed before the trust head is updated")
    grid: int = setting(401, source="AS-841", doc="grid points in log T for the global search of the Brier-optimal temperature")
    tol: float = setting(1e-10, source="AS-841", doc="tolerance in log T of the golden-section refinement")
    decision_threshold: float = setting(0.5, source="AS-841, AS-419", doc="alert decision P_inf(K) >= threshold whose correctness the trust head predicts")
    trust_optimiser: OptimiserConfig = setting(OptimiserConfig(lr=1e-4, steps=200, batch_size=64), source="AS-841",
                                               doc="optimiser of the trust-head delta (Brier on decision correctness)")
    calibration_optimiser: OptimiserConfig = setting(OptimiserConfig(lr=1e-3, steps=200, batch_size=4), source="AS-841",
                                                     doc="optimiser of the calibration-head delta (toward the RLCD temperature)")

    def __post_init__(self) -> None:
        _check_choice("RLCDConfig.score", self.score, RLCD_SCORES)
        if not self.families or any(f not in OUTPUT_FAMILIES for f in self.families):
            raise InvariantViolation(f"RLCDConfig.families must be a non-empty subset of {list(OUTPUT_FAMILIES)}")
        if self.min_pairs < 2 or self.min_decisions < 2:
            raise InvariantViolation("RLCDConfig.min_pairs and min_decisions must be >= 2")
        if self.grid < 3:
            raise InvariantViolation("RLCDConfig.grid must be >= 3")
        _check_positive("RLCDConfig.tol", self.tol)
        _check_probability_open("RLCDConfig.decision_threshold", self.decision_threshold)


@dataclass(frozen=True)
class EvaluationConfig:
    """Held-out evaluation of a candidate and the gates a promotion checks (AS-842, AS-843, AS-845)."""

    held_out_fraction: float = setting(0.2, source="AS-842", doc="latest share of the feedback (by event time) held out from the fit")
    min_outcomes: int = setting(20, source="AS-843", doc="held-out confirmed outcomes a forecast-changing candidate needs")
    min_preferences: int = setting(10, source="AS-843", doc="held-out preferences an RLHF candidate needs")
    min_pairs: int = setting(20, source="AS-843", doc="held-out scored pairs an RLCD family needs")
    margin_log: float = setting(0.02, source="AS-843", doc="non-inferiority margin of the log score, nats per trigger")
    margin_brier: float = setting(0.005, source="AS-843", doc="non-inferiority margin of Brier scores")
    margin_stage: float = setting(0.02, source="AS-843", doc="non-inferiority margin of the stage log score, nats per confirmed step")
    max_kl: float = setting(0.1, source="AS-843", doc="largest mean per-step KL(candidate || reference) in nats")
    min_preference_accuracy: float = setting(0.5, source="AS-843", doc="held-out share of preferences the implicit reward orders correctly")
    require_no_new_drift_alerts: bool = setting(True, source="AS-843, D-21", doc="the Monitor may raise no alert kind on the candidate's held-out pairs that it does not raise on the reference's")
    bootstrap_resamples: int = setting(2000, source="AS-842", doc="paired bootstrap resamples of the before/after differences")
    confidence: float = setting(0.95, source="AS-842", doc="two-sided confidence of the bootstrap intervals (BCa)")
    routes_n: int = setting(0, source="AS-842", doc="routes imagined per held-out trigger (0 = the Forecaster's run-time default); each forecast is judged at its outcome's horizon")
    kl_routes: int = setting(16, source="AS-842", doc="routes drawn from the reference per held-out situation to measure KL(candidate || reference)")

    def __post_init__(self) -> None:
        _check_probability_open("EvaluationConfig.held_out_fraction", self.held_out_fraction)
        if min(self.min_outcomes, self.min_preferences, self.min_pairs) < 1:
            raise InvariantViolation("EvaluationConfig minimum counts must be >= 1")
        for name in ("margin_log", "margin_brier", "margin_stage"):
            v = getattr(self, name)
            if not (math.isfinite(v) and v >= 0):
                raise InvariantViolation(f"EvaluationConfig.{name} must be finite and >= 0")
        _check_positive("EvaluationConfig.max_kl", self.max_kl)
        if not 0.0 <= self.min_preference_accuracy <= 1.0:
            raise InvariantViolation("EvaluationConfig.min_preference_accuracy must lie in [0, 1]")
        if self.bootstrap_resamples < 100:
            raise InvariantViolation("EvaluationConfig.bootstrap_resamples must be >= 100")
        _check_probability_open("EvaluationConfig.confidence", self.confidence)
        if self.routes_n < 0 or self.kl_routes < 1:
            raise InvariantViolation("EvaluationConfig.routes_n must be >= 0 and kl_routes >= 1")


@dataclass(frozen=True)
class FeedbackLearningConfig:
    """Everything the Verifier's feedback learning reads (D-65)."""

    seed: int = setting(20_261_006, source="AS-848", doc="root seed; every draw derives its own seed from it and the situation id")
    ledger: LedgerConfig = setting(LedgerConfig(), source="AS-830", doc="feedback and audit ledger")
    adapter: AdapterConfig = setting(AdapterConfig(), source="AS-835", doc="parameters a candidate may change")
    bradley_terry: BradleyTerryConfig = setting(BradleyTerryConfig(), source="AS-833", doc="preference reward model")
    dpo: DPOConfig = setting(DPOConfig(), source="AS-834", doc="direct preference optimisation")
    outcome_reward: OutcomeRewardConfig = setting(OutcomeRewardConfig(), source="AS-839, AS-840", doc="verifiable rewards")
    grpo: GRPOConfig = setting(GRPOConfig(), source="AS-836 ... AS-838", doc="group-relative policy optimisation")
    rlcd: RLCDConfig = setting(RLCDConfig(), source="AS-841", doc="calibrated-decision objective")
    evaluation: EvaluationConfig = setting(EvaluationConfig(), source="AS-842, AS-843", doc="held-out evaluation and gates")

    def __post_init__(self) -> None:
        if not 0 <= self.seed < 2**63:
            raise InvariantViolation("FeedbackLearningConfig.seed must lie in [0, 2**63)")


def feedback_config_yaml(cfg: FeedbackLearningConfig | None = None) -> str:
    """The YAML text of a configuration (defaults when None), with the provenance of every field as a comment."""
    header = ("Feedback learning of the Verifier: RLHF, RLVR, RLCD (models/verifier/config.py, FeedbackLearningConfig).",
              "Generated from the typed dataclasses (D-57). Do not edit by hand: change the dataclass and regenerate.")
    return render_yaml(cfg if cfg is not None else FeedbackLearningConfig(), header=header, key="verifier_feedback")


def feedback_config_from_mapping(data: Mapping[str, Any]) -> FeedbackLearningConfig:
    """A validated configuration from plain data (top level, or under the key `verifier_feedback`)."""
    section = data.get("verifier_feedback", data) if isinstance(data.get("verifier_feedback"), Mapping) else data
    return from_mapping(FeedbackLearningConfig, section)


__all__ = [
    "BRIER_CENSORING", "KL_ESTIMATORS", "POLICY_TARGETS", "P_INF_SCORES", "RESPONDED_TO", "RLCD_SCORES", "STAGE_SCORES",
    "AdapterConfig", "BradleyTerryConfig", "DPOConfig", "EvaluationConfig", "FeedbackLearningConfig", "GRPOConfig",
    "LedgerConfig", "OptimiserConfig", "OutcomeRewardConfig", "RLCDConfig", "feedback_config_from_mapping",
    "feedback_config_yaml",
]
