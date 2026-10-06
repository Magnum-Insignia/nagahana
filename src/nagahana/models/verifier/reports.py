"""Verifier report types (frozen): calibration, drift, poisoning alerts, calibration proposals and state.

`roles/contracts.py` has the feedback and ledger contracts (HumanFeedback, OutcomeForecastPair,
HumanCommand) but no calibration or drift report types, so they live here. Each validates its own
invariants, like the contracts do.

Decisions: D-21 (changes only on a HumanCommand), D-45 (value head scores trust; policy head proposes
the correction), D-65 (RLCD proposals, source "rlcd"). Assumptions: AS-25.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field

from nagahana.core.errors import InvariantViolation
from nagahana.evaluation.calibration import ReliabilityBin
from nagahana.roles.contracts import HumanCommand

#: Output families whose probabilities the Verifier calibrates (VerifierConfig.n_output_families = 4).
OUTPUT_FAMILIES: tuple[str, ...] = ("p_inf", "stage", "compromise", "advice")


@dataclass(frozen=True)
class PoisoningAlert:
    """A drift statistic crossed its threshold: for human review, never an automatic action (D-21, [A-16])."""

    kind: str          # "page_hinkley" | "cusum_up" | "cusum_down" | "systematic_gap"
    region: str        # "environment" | "imagination" | "ledger"
    statistic: float
    threshold: float
    index: int         # observation count at which it fired
    message: str


@dataclass(frozen=True)
class RegionDrift:
    """Welford + Page-Hinkley summary of one memory region's latents."""

    region: str
    n: int
    mean_norm: float          # ||running mean||_2
    mean_variance: float      # mean of the running per-dimension variances
    ph_statistic: float
    ph_alarm: bool


@dataclass(frozen=True)
class DriftReport:
    """The Monitor's drift report (architecture section 6 "Drift report per memory region", "Poisoning alerts")."""

    regions: tuple[RegionDrift, ...]
    cusum_up: float
    cusum_down: float
    cusum_alarm: bool
    systematic_gap: float | None     # |sum (f - y)| / n over the last drift_window scored resolutions
    gap_window_n: int
    n_scored: int
    n_responded_to: int
    alerts: tuple[PoisoningAlert, ...] = ()


@dataclass(frozen=True)
class CalibrationReport:
    """Reliability, ECE and Brier of one output family over scored (not responded-to) pairs."""

    family: str
    n: int
    ece: float
    brier: float
    bins: tuple[ReliabilityBin, ...]
    ml_temperature: float | None = None

    def __post_init__(self) -> None:
        if self.n < 0 or (self.n and not (0.0 <= self.brier <= 1.0)):
            raise InvariantViolation("a calibration report needs n >= 0 and a Brier score in [0, 1]")


def _check_temperatures(temps: Mapping[str, float], lo: float, hi: float) -> None:
    for fam, t in temps.items():
        if fam not in OUTPUT_FAMILIES:
            raise InvariantViolation(f"unknown output family {fam!r}")
        if not (math.isfinite(t) and lo <= t <= hi):
            raise InvariantViolation(f"temperature of {fam} must be in [{lo}, {hi}]; got {t}")


@dataclass(frozen=True)
class CalibrationProposal:
    """Proposed temperatures per output family (D-45 policy head or the ML fit). Applied only via `gate`."""

    temperatures: Mapping[str, float]
    source: str                                  # "ml-fit" | "policy-head" | "rlcd"
    n_pairs: Mapping[str, int] = field(default_factory=dict)
    t_min: float = 0.25
    t_max: float = 4.0

    def __post_init__(self) -> None:
        _check_temperatures(self.temperatures, self.t_min, self.t_max)


@dataclass(frozen=True)
class TemperatureState:
    """Temperatures in force. Identity (T = 1) until a human applies a proposal."""

    temperatures: Mapping[str, float] = field(default_factory=lambda: {f: 1.0 for f in OUTPUT_FAMILIES})
    applied_by: tuple[HumanCommand, ...] = ()

    def __post_init__(self) -> None:
        _check_temperatures(self.temperatures, 0.0, math.inf)
        if any(t <= 0 for t in self.temperatures.values()):
            raise InvariantViolation("temperatures must be > 0")
