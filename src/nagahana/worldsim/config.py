"""Configuration of a simulated world (P-14): topology, benign traffic, attacker, defender, sensors.

Every setting a scenario can change is a field of a frozen dataclass here, so a scenario is a pure
data object (`conf/worldsim/*.yaml` loads into `ScenarioConfig`) and nothing about a world is hidden
in code. Collections are tuples, so configs are immutable and hashable and can be static arguments of
a jitted simulation.

Units: rates are per hour, times are in seconds, probabilities in [0, 1]. Diurnal and weekly profiles
are integer weights per hour of day (24) and per day of week (7, Monday first), applied as a
piecewise-constant modulation of the base rate; integer weights keep the benign arrival intensity
bit-identical across backends (AS-804, AS-819).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from nagahana.core.errors import InvariantViolation
from nagahana.worldsim import vocab

#: A flat business-day profile: low overnight, ramp in the morning, plateau through the day.
BUSINESS_HOURS_24: tuple[int, ...] = (
    1, 1, 1, 1, 1, 2, 4, 8, 16, 20, 22, 22, 18, 22, 22, 20, 16, 10, 6, 4, 3, 2, 2, 1,
)
#: Monday to Sunday: full on weekdays, light at the weekend.
BUSINESS_WEEK_7: tuple[int, ...] = (10, 10, 10, 10, 10, 3, 2)
#: A flat profile for sites with no diurnal rhythm (always-on OT polling backbones).
FLAT_24: tuple[int, ...] = (10,) * 24
FLAT_7: tuple[int, ...] = (10,) * 7


@dataclass(frozen=True)
class TopologyConfig:
    """How many of each archetype and how the segments connect.

    `counts` maps an archetype name (vocab.ARCHETYPES) to how many to place. `enterprise_segments`
    and the OT Purdue levels are derived from the archetypes; `allow_cross_segment` lets enterprise
    workstations reach servers in other segments (east-west), and `ot_from_enterprise` whether the
    enterprise can reach the OT DMZ (a conduit, IEC 62443). `internet` and `multicast` are added
    automatically when referenced.
    """

    counts: tuple[tuple[str, int], ...]
    enterprise_segments: int = 2
    allow_cross_segment: bool = True
    ot_from_enterprise: bool = True
    ipv4_base: str = "10.0.0.0"

    def __post_init__(self) -> None:
        for name, n in self.counts:
            if name not in vocab.ARCHETYPE_CODE:
                raise InvariantViolation(f"unknown archetype {name!r}")
            if n < 0:
                raise InvariantViolation(f"count for {name!r} must be >= 0")
        if self.enterprise_segments < 1:
            raise InvariantViolation("enterprise_segments must be >= 1")

    def count(self, name: str) -> int:
        """Configured count of an archetype (0 if absent)."""
        return dict(self.counts).get(name, 0)


@dataclass(frozen=True)
class BenignConfig:
    """Benign traffic intensity and shape.

    `session_rate_per_host_hour` is the base rate at which an active client opens a session at the
    diurnal peak; `diurnal` and `weekly` modulate it. `ot_poll_interval_s` is the period of OT polling
    (a deterministic periodic channel, AS-805) and `ot_poll_jitter_s` its bounded jitter.
    """

    session_rate_per_host_hour: float = 30.0
    diurnal: tuple[int, ...] = BUSINESS_HOURS_24
    weekly: tuple[int, ...] = BUSINESS_WEEK_7
    ot_poll_interval_s: float = 2.0
    ot_poll_jitter_s: float = 0.2
    heavy_tail_alpha: float = 1.6          # bounded-Pareto tail index for flow sizes (AS-804)

    def __post_init__(self) -> None:
        if len(self.diurnal) != 24 or len(self.weekly) != 7:
            raise InvariantViolation("diurnal must have 24 entries and weekly 7")
        if min(self.diurnal) < 0 or min(self.weekly) < 0:
            raise InvariantViolation("profile weights must be non-negative")
        if self.session_rate_per_host_hour < 0 or self.ot_poll_interval_s <= 0:
            raise InvariantViolation("rates and the poll interval must be positive")
        if not 1.0 < self.heavy_tail_alpha < 3.0:
            raise InvariantViolation("heavy_tail_alpha must lie in (1, 3)")


@dataclass(frozen=True)
class AttackConfig:
    """The attacker: which campaign, when it starts, how exposed the network is to it.

    `start_min_s` and `start_max_s` bound a per-world uniform start offset (randomised so that a
    campaign is not tied to a fixed clock time, which would be a shortcut, D-50). `vuln_density` is
    the probability a host carries each applicable service vulnerability; `local_vuln_density` the
    probability it carries the local privilege-escalation weakness; `credential_reuse` the probability
    a credential valid on one host is also valid on another (lateral-movement surface).
    """

    campaign: str = "apt_full"
    enabled: bool = True
    start_min_s: float = 300.0
    start_max_s: float = 1800.0
    vuln_density: float = 0.25
    local_vuln_density: float = 0.5
    credential_reuse: float = 0.3
    max_events: int = 256

    def __post_init__(self) -> None:
        if self.campaign not in vocab.CAMPAIGN_CODE:
            raise InvariantViolation(f"unknown campaign {self.campaign!r}")
        for p in (self.vuln_density, self.local_vuln_density, self.credential_reuse):
            if not 0.0 <= p <= 1.0:
                raise InvariantViolation("densities must lie in [0, 1]")
        if self.start_max_s < self.start_min_s < 0:
            raise InvariantViolation("start offsets must satisfy 0 <= start_min_s <= start_max_s")
        if self.max_events < 1:
            raise InvariantViolation("max_events must be >= 1")


@dataclass(frozen=True)
class DefenderConfig:
    """Optional defender. Reacts to accumulated suspicion by isolating, blocking or patching.

    `response_rate_per_hour` is the firing rate when at least one entity is over `suspicion_threshold`;
    the action is chosen by the integer weights `isolate`, `block`, `patch`.
    """

    enabled: bool = False
    response_rate_per_hour: float = 4.0
    suspicion_threshold: float = 3.0
    isolate: int = 2
    block: int = 1
    patch: int = 1

    def __post_init__(self) -> None:
        if self.response_rate_per_hour < 0 or self.suspicion_threshold <= 0:
            raise InvariantViolation("response rate must be >= 0 and the threshold > 0")
        if min(self.isolate, self.block, self.patch) < 0 or (self.isolate + self.block + self.patch) == 0:
            raise InvariantViolation("action weights must be non-negative and not all zero")


@dataclass(frozen=True)
class SensorConfig:
    """One observation sensor (P-03 statuses; D-41 absence is never zero).

    kind: "tap", "netflow", "zeek", "ids" or "authlog".
    coverage: archetype domains this sensor sees ("enterprise", "ot", or both). A session neither of
        whose endpoints is covered is NOT_OBSERVABLE to this sensor.
    granularity: "flow-state" (D-51, default for a tap), "packet" or "flow".
    sampling_n: keep 1 in n sessions (NetFlow sampling); 1 keeps all.
    active_timeout_s / inactive_timeout_s: NetFlow/IPFIX export timers.
    packet_loss: probability a session is missed entirely (partial coverage, drops).
    clock_skew_s / clock_drift_ppm: fixed offset and linear drift of this sensor's clock.
    detection_prob: per noisy attacker action, probability the IDS raises a true alert.
    false_alarm_per_hour: Poisson rate of benign-looking false alerts (IDS).
    reliability: evidence reliability in (0, 1] attached to LOW_RELIABILITY fields (0 means OBSERVED).
    """

    kind: str
    coverage: tuple[str, ...] = (vocab.ENTERPRISE, vocab.OT)
    granularity: str = "flow-state"
    sampling_n: int = 1
    active_timeout_s: float = 60.0
    inactive_timeout_s: float = 15.0
    packet_loss: float = 0.0
    clock_skew_s: float = 0.0
    clock_drift_ppm: float = 0.0
    detection_prob: float = 0.6
    false_alarm_per_hour: float = 1.0
    reliability: float = 0.0

    _KINDS = ("tap", "netflow", "zeek", "ids", "authlog")
    _GRAINS = ("flow-state", "packet", "flow")

    def __post_init__(self) -> None:
        if self.kind not in self._KINDS:
            raise InvariantViolation(f"sensor kind must be one of {self._KINDS}, got {self.kind!r}")
        if self.granularity not in self._GRAINS:
            raise InvariantViolation(f"granularity must be one of {self._GRAINS}")
        for name in self.coverage:
            if name not in (vocab.ENTERPRISE, vocab.OT):
                raise InvariantViolation(f"coverage domain {name!r} is not enterprise or ot")
        if self.sampling_n < 1:
            raise InvariantViolation("sampling_n must be >= 1")
        for p in (self.packet_loss, self.detection_prob):
            if not 0.0 <= p <= 1.0:
                raise InvariantViolation("probabilities must lie in [0, 1]")
        if not 0.0 <= self.reliability <= 1.0:
            raise InvariantViolation("reliability must lie in [0, 1]")
        if self.active_timeout_s <= 0 or self.inactive_timeout_s <= 0:
            raise InvariantViolation("export timers must be positive")


@dataclass(frozen=True)
class ObservationConfig:
    """The sensor fabric. `sensors` lists the sensors; the emitted records concatenate their output."""

    sensors: tuple[SensorConfig, ...] = (SensorConfig("tap"),)

    def __post_init__(self) -> None:
        if not self.sensors:
            raise InvariantViolation("at least one sensor is required")


@dataclass(frozen=True)
class SimulationConfig:
    """Time horizon and the fixed budgets that keep the simulation fixed-shape (AS-803, AS-817).

    horizon_s: simulated duration.
    event_slots: the number of jump-process candidate slots (thinned candidates included). A world
        that fills them before the horizon is marked saturated; the count is reported, never hidden.
    start_epoch_s: wall-clock epoch of t = 0 (sets the diurnal phase; randomised per world around it).
    start_jitter_s: per-world uniform jitter of the wall-clock origin (randomises the diurnal phase so
        attacks are not pinned to a fixed clock time, D-50).
    """

    horizon_s: float = 3600.0
    event_slots: int = 4096
    start_epoch_s: float = 1_700_000_000.0
    start_jitter_s: float = 604800.0

    def __post_init__(self) -> None:
        if self.horizon_s <= 0 or self.event_slots < 1:
            raise InvariantViolation("horizon_s must be positive and event_slots >= 1")
        if self.start_jitter_s < 0:
            raise InvariantViolation("start_jitter_s must be >= 0")


@dataclass(frozen=True)
class ScenarioConfig:
    """A complete, named world recipe."""

    name: str
    description: str
    network: str
    topology: TopologyConfig
    benign: BenignConfig = field(default_factory=BenignConfig)
    attack: AttackConfig = field(default_factory=AttackConfig)
    defender: DefenderConfig = field(default_factory=DefenderConfig)
    observation: ObservationConfig = field(default_factory=ObservationConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    novelty: str = ""                 # "", "known" or "novel": carried into evaluation metadata (D-23)

    def __post_init__(self) -> None:
        if not self.name or any(c in self.name for c in " /."):
            raise InvariantViolation("scenario name must be a token without spaces, dots or slashes")
        if self.novelty not in ("", "known", "novel"):
            raise InvariantViolation("novelty must be '', 'known' or 'novel'")

    def with_overrides(self, **changes: Any) -> ScenarioConfig:
        """A copy with top-level fields replaced (used by the CLI and tests)."""
        return replace(self, **changes)


__all__ = [
    "BUSINESS_HOURS_24", "BUSINESS_WEEK_7", "FLAT_24", "FLAT_7", "AttackConfig", "BenignConfig",
    "DefenderConfig", "ObservationConfig", "ScenarioConfig", "SensorConfig", "SimulationConfig",
    "TopologyConfig",
]
