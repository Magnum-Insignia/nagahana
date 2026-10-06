"""What the roles hand to each other and to people: the output contracts (docs/architecture.md section 6).

The problem statement requires three outputs per forecast: an infiltration probability over the
next K windows, a predicted MITRE ATT&CK stage, and the driving features (attention or attribution).
"Black-box outputs without interpretability are not acceptable." The design adds belief and
trust readouts, energy readouts, advisory counter-measure sequences, Verifier reports, a decoded view
and forensic reports ([Q-17], ARCH #26).

Every contract validates its own invariants on construction, so an ill-formed output cannot leave
a role. Examples: P_inf(k) is a cumulative probability, so it can never decrease with k. Stage
probabilities sum to 1. Advisory output is always advisory.
"""

from __future__ import annotations

import enum
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from nagahana.core.errors import InvariantViolation


class AttackStage(enum.Enum):
    """ATT&CK phases named in the problem statement. Extend to full Enterprise + ICS tactics as the
    data model grows (D-34: OT in scope)."""

    RECONNAISSANCE = "reconnaissance"
    INITIAL_ACCESS = "initial_access"
    LATERAL_MOVEMENT = "lateral_movement"
    COMMAND_AND_CONTROL = "command_and_control"
    EXFILTRATION = "exfiltration"
    # The rest of `models.vocab.STAGES` (AS-19: "none" + the 14
    # ATT&CK Enterprise tactics), so the model's 15-class stage head converts one-to-one. Values equal the
    # vocab names; members above keep their order and values.
    NONE = "none"
    RESOURCE_DEVELOPMENT = "resource_development"
    EXECUTION = "execution"
    PERSISTENCE = "persistence"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    DEFENSE_EVASION = "defense_evasion"
    CREDENTIAL_ACCESS = "credential_access"
    DISCOVERY = "discovery"
    COLLECTION = "collection"
    IMPACT = "impact"


class D3FENDTactic(enum.Enum):
    """MITRE D3FEND tactics: the vocabulary of the Advisor's counter-measures (D-33)."""

    MODEL = "model"
    HARDEN = "harden"
    DETECT = "detect"
    ISOLATE = "isolate"
    DECEIVE = "deceive"
    EVICT = "evict"
    RESTORE = "restore"


class ProvenanceTag(enum.Enum):
    """How a rendered element is known (Decoder view, P-01). Beliefs never render as facts."""

    OBSERVED = "observed"     # from the Environment
    BELIEVED = "believed"     # from Imagination (belief)
    FORECAST = "forecast"     # from Imagination (imagined futures)


def _prob(x: float, what: str) -> None:
    if not (0.0 <= x <= 1.0) or math.isnan(x):
        raise InvariantViolation(f"{what} must be a probability in [0, 1]; got {x}")


@dataclass(frozen=True)
class DrivingFeature:
    """One reason behind a prediction: a feature and its signed contribution."""

    feature: str          # field ID or derived pattern, e.g. "flow.flag_count.syn"
    contribution: float
    method: str           # "attention" | "attribution" | "counterfactual"


@dataclass(frozen=True)
class ImaginedPath:
    """One imagined attack path (one of the top-N)."""

    probability: float
    stages: tuple[AttackStage, ...]
    entities: tuple[str, ...]

    def __post_init__(self) -> None:
        _prob(self.probability, "path probability")


@dataclass(frozen=True)
class ComputeRecord:
    """What a forecast cost: inference-time scaling is reported, never hidden (D-44)."""

    samples_n: int
    horizon_k: int
    refinement_steps: int
    wall_time_s: float
    # D-44: the loop passes are a run-time budget recorded with each forecast. One field per looped
    # model (AS-08).
    tstct_passes: int
    taaft_passes: int


@dataclass(frozen=True)
class ForecastBundle:
    """The Forecaster's output for one trigger.

    Invariants: len(p_inf) = K; P_inf(k) in [0, 1] and non-decreasing in k (it is P(tau <= k)); each row
    of stage_probs sums to 1.
    """

    p_inf: tuple[float, ...]
    stage_probs: tuple[tuple[float, ...], ...]
    stages: tuple[AttackStage, ...]
    top_paths: tuple[ImaginedPath, ...]
    driving_features: tuple[DrivingFeature, ...]
    compute: ComputeRecord
    hazard: tuple[float, ...] | None = None
    p_inf_interval: tuple[tuple[float, float], ...] | None = None
    safe_horizon: int | None = None
    observability_gaps: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        k = len(self.p_inf)
        if k != self.compute.horizon_k:
            raise InvariantViolation(f"p_inf has {k} steps but horizon K = {self.compute.horizon_k}")
        for i, p in enumerate(self.p_inf):
            _prob(p, f"P_inf({i + 1})")
            if i and p + 1e-12 < self.p_inf[i - 1]:
                raise InvariantViolation("P_inf(k) must be non-decreasing: it is P(first infiltration <= k)")
        if len(self.stage_probs) != k:
            raise InvariantViolation("stage_probs needs one row per step")
        for row in self.stage_probs:
            if len(row) != len(self.stages) or abs(sum(row) - 1.0) > 1e-6:
                raise InvariantViolation("each stage_probs row must cover every stage and sum to 1")
        if self.hazard is not None:
            if len(self.hazard) != k:
                raise InvariantViolation("hazard needs one value per step")
            for h in self.hazard:
                _prob(h, "hazard")


@dataclass(frozen=True)
class BeliefReadout:
    """Belief and trust from TAAFT's belief-trust lens (Imagination only, D-35)."""

    entity_compromise: Mapping[str, tuple[float, float]]   # entity -> (mean, std)
    suspicion_floor: float                                  # assume-breach floor: > 0 by design (ARCH section 4.7)
    stage_posterior: Mapping[AttackStage, float]
    goal_posterior: Mapping[str, float]
    telemetry_trust: Mapping[str, float]                    # source or field -> trust in [0, 1]
    type_posterior: Mapping[str, float] = field(default_factory=dict)  # proposal P-13

    def __post_init__(self) -> None:
        if not 0.0 < self.suspicion_floor < 1.0:
            raise InvariantViolation("suspicion floor must be in (0, 1): never zero (assume breach)")
        for name, dist in (("stage", self.stage_posterior), ("goal", self.goal_posterior)):
            if dist and abs(sum(dist.values()) - 1.0) > 1e-6:
                raise InvariantViolation(f"{name} posterior must sum to 1")
        for src, t in self.telemetry_trust.items():
            _prob(t, f"trust of {src}")


@dataclass(frozen=True)
class CounterStep:
    """One step of a counter-measure sequence."""

    tactic: D3FENDTactic
    action: str
    target: str
    level: str   # "graph" | "sensor" (D-33)

    def __post_init__(self) -> None:
        if self.level not in ("graph", "sensor"):
            raise InvariantViolation("level must be 'graph' or 'sensor'")


@dataclass(frozen=True)
class CounterSequence:
    """A ranked counter-measure sequence with its evidence."""

    steps: tuple[CounterStep, ...]
    delta_p_inf: float           # change in P_inf(K) after re-imagination (negative = risk reduced)
    disruption_cost: float       # priced per asset (pricing held: D-03b)
    feasible: bool               # its predicted effects respect the physics boundary
    information_value: float     # uncertainty it removes (information value)
    # AS-24: the risk view over imagined routes, reported beside the expected delta P_inf (`delta_p_inf`).
    # None when the sequence was not evaluated over routes.
    delta_p_inf_cvar: float | None = None    # CVaR_alpha of delta P_inf over routes (the worst alpha share)
    delta_p_inf_worst: float | None = None   # the worst route's delta P_inf
    cvar_alpha: float | None = None          # alpha used
    infeasible_reasons: tuple[str, ...] = () # rule or physics checks that failed (empty when feasible)

    def __post_init__(self) -> None:
        if self.cvar_alpha is not None and not 0.0 < self.cvar_alpha <= 1.0:
            raise InvariantViolation("cvar_alpha must be in (0, 1]")
        if self.feasible and self.infeasible_reasons:
            raise InvariantViolation("a feasible sequence cannot carry infeasibility reasons")


@dataclass(frozen=True)
class AdvisoryBundle:
    """The Advisor's output. Always advisory: people decide and act (D-33, [Q-13])."""

    sequences: tuple[CounterSequence, ...]
    advisory_only: bool = True

    def __post_init__(self) -> None:
        if not self.advisory_only:
            raise InvariantViolation("NagaHana never acts on the network: advisory_only must stay True")


class FeedbackKind(enum.Enum):
    """Kinds of human feedback the Verifier stores as supplied truth [A-21]."""

    STEP_LABEL = "step_label"        # is this imagined step plausible? (trains process rewards)
    OUTCOME = "outcome"              # what actually happened
    RESPONDED_TO = "responded_to"    # the SOC acted on the forecast, so it is not scored as wrong [Q-38]


@dataclass(frozen=True)
class HumanFeedback:
    """One piece of analyst feedback. Trusted: human feedback is outside the threat model (D-17)."""

    kind: FeedbackKind
    ref: str          # forecast id or imagined-step id
    label: str
    analyst: str
    time: float


@dataclass(frozen=True)
class HumanCommand:
    """An explicit, attributable human instruction to change the model (D-21).

    The only way model-changing operations run. "it will only adjust model's weights based on
    human's supervised command to do so, otherwise it wont" [A-21].
    """

    approver: str
    action: str       # e.g. "apply-calibration", "update-site-adapter", "update-weights"
    reason: str
    time: float

    def __post_init__(self) -> None:
        if not (self.approver and self.action and self.reason):
            raise InvariantViolation("a HumanCommand needs an approver, an action and a reason")


@dataclass(frozen=True)
class OutcomeForecastPair:
    """A forecast probability and what happened. The unit of calibration and memory-drift analysis [A-16]."""

    forecast_id: str
    predicted: float
    occurred: bool
    responded_to: bool = False

    def __post_init__(self) -> None:
        _prob(self.predicted, "predicted probability")


@dataclass(frozen=True)
class ForensicReport:
    """Forensic replay of an uploaded capture ([Q-17]; ARCH section 7)."""

    timeline: Sequence[tuple[float, str]]                 # (time, what would have been forecast)
    narrative: Sequence[tuple[AttackStage, str, float]]   # (stage, description, time)
    patient_zero: Sequence[tuple[str, float]]             # (entity, probability)
    counterfactuals: Sequence[str]
    observability_gaps: Sequence[str]
    tamper_signs: Sequence[str] = ()
