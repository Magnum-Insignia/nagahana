"""Run-level choices of the Generator that its component config does not hold (D-57 convention).

Purpose
-------
`models/config/components.GeneratorConfig` holds the Generator's network widths and the numeric
settings of each family. The fields here are the run-level choices around them, each with its default
and provenance (`core.config.setting`):

- `energy_acceptance`: "off" | "from-stage-3" | "always" (AS-583; AS-361 gives the quantile range).
  TAAFT's marginal energy exists only after TAAFT pretraining (stage 2), so the default screens the
  variant pool of full training (stage 3) and accepts stage-1 and stage-2 variants by the hard limits
  and the physics gate.
- `family_weights`: sampling weight per producer family (empty = uniform).
- `flow_only_fields`, `drop_levels` (AS-350, AS-351), `service_alias_classes`, `ephemeral_ports`
  (AS-354), `fingerprint_fields` (AS-585): the tables of the deterministic transforms.
- `energy_hidden`, `energy_blocks`, `langevin`: the energy-SSL family (AS-586).

Which families run is the held decision D-14, resolved through `governance.decisions.require("D-14")`
(working option "all families", AS-27; a run may configure another admissible option); the mapping
from an option to producer families is `families_for_option`. The hard physics gate is proposal P-11,
enabled per run (`enabled_proposals`; the training run enables it by default, AS-28).

Decisions: D-14 (held), D-40, D-57. Proposals: P-11. Assumptions: AS-27, AS-28, AS-350, AS-351, AS-354,
AS-361, AS-583, AS-585, AS-586.
"""

from __future__ import annotations

from dataclasses import dataclass

from nagahana.core.config import setting
from nagahana.core.errors import InvariantViolation

#: The deterministic producer families: the owner's observability sliding, signature and trace variation
#: ([Q-37], [A-17]); they preserve labels by construction and run under every D-14 option.
DETERMINISTIC_FAMILIES: tuple[str, ...] = ("observability-sliding", "signature-variation", "topology-variation", "tool-fingerprint")
#: The learned producer families of [A-17]; which of them run is D-14.
LEARNED_FAMILIES: tuple[str, ...] = ("masked-generative", "autoregressive", "diffusion", "energy-ssl")
PRODUCER_FAMILIES: tuple[str, ...] = (*DETERMINISTIC_FAMILIES, *LEARNED_FAMILIES)
ENERGY_ACCEPTANCE_MODES: tuple[str, ...] = ("off", "from-stage-3", "always")

#: AS-350: the IPFIX biflow basic profile kept by the flow-only export (RFC 7012 IEs, RFC 5103 biflows).
DEFAULT_FLOW_ONLY_FIELDS: tuple[str, ...] = (
    "flow.src_ip", "flow.dst_ip", "flow.src_port", "flow.dst_port", "flow.protocol", "flow.tcp_flags",
    "flow.bytes_fwd", "flow.bytes_bwd", "flow.packets_fwd", "flow.packets_bwd", "flow.duration",
)
#: AS-354: one service on its registered alternative ports (IANA: http 80, http-alt 8080 and 8008;
#: SMB direct over TCP 445 and over the NetBIOS session service 139).
DEFAULT_SERVICE_ALIAS_CLASSES: tuple[tuple[str, tuple[int, ...]], ...] = (("http", (80, 8080, 8008)), ("smb", (445, 139)))


def families_for_option(option: str) -> tuple[tuple[str, ...], bool]:
    """(producer families, JEM energy acceptance on) for a D-14 option.

    "all families": every producer and the JEM-style acceptance; "jem": the deterministic producers and
    the acceptance; "masked-generative", "autoregressive", "diffusion": the deterministic producers and
    that learned family, without the acceptance.
    """
    if option == "all families":
        return PRODUCER_FAMILIES, True
    if option == "jem":
        return DETERMINISTIC_FAMILIES, True
    if option in ("masked-generative", "autoregressive", "diffusion"):
        return (*DETERMINISTIC_FAMILIES, option), False
    raise InvariantViolation(f"D-14 option {option!r} has no producer mapping")


@dataclass(frozen=True)
class LangevinConfig:
    """Annealed Langevin sampling of the energy-SSL family (AS-586; Song and Ermon, NeurIPS 2019).

    The noise ladder sigma_1 > ... > sigma_L is geometric in standardised units; level i takes
    `steps_per_level` steps of size alpha_i = step_scale * sigma_i^2 / sigma_L^2 (Song and Ermon,
    Algorithm 1). With `metropolis` the steps of the last level are Metropolis-adjusted (MALA, Roberts
    and Tweedie, Bernoulli 1996), so the last level samples its energy exactly.
    """

    sigma_max: float = setting(1.0, source="AS-586", doc="largest noise level (standardised units)")
    sigma_min: float = setting(0.01, source="AS-586", doc="smallest noise level")
    levels: int = setting(10, source="AS-586", doc="noise levels of the ladder")
    steps_per_level: int = setting(20, source="AS-586", doc="Langevin steps per level")
    step_scale: float = setting(2e-5, source="AS-586", doc="eps of alpha_i = eps * sigma_i^2 / sigma_L^2")
    metropolis: bool = setting(True, source="AS-586", doc="Metropolis-adjust the last level (MALA)")

    def __post_init__(self) -> None:
        if not 0.0 < self.sigma_min < self.sigma_max:
            raise InvariantViolation("langevin: need 0 < sigma_min < sigma_max")
        if self.levels < 1 or self.steps_per_level < 1:
            raise InvariantViolation("langevin: levels and steps_per_level must be >= 1")
        if not self.step_scale > 0.0:
            raise InvariantViolation("langevin: step_scale must be > 0")


@dataclass(frozen=True)
class GeneratorPolicy:
    """The Generator's run-level choices (module docstring)."""

    energy_acceptance: str = setting("from-stage-3", source="AS-583, AS-361", doc="off | from-stage-3 | always")
    family_weights: tuple[tuple[str, float], ...] = setting((), source="AS-27", doc="(family, sampling weight) pairs; empty = uniform")
    flow_only_fields: tuple[str, ...] = setting(DEFAULT_FLOW_ONLY_FIELDS, source="AS-350", doc="fields kept by the flow-only export")
    drop_levels: tuple[str, ...] = setting(("packet",), source="AS-351", doc="field levels made NOT_OBSERVABLE by the drop transform")
    service_alias_classes: tuple[tuple[str, tuple[int, ...]], ...] = setting(DEFAULT_SERVICE_ALIAS_CLASSES, source="AS-354",
                                                                            doc="ports of one service, remapped among each other")
    ephemeral_ports: tuple[int, int] = setting((49152, 65535), source="AS-354, RFC 6335 section 6", doc="dynamic port range")
    fingerprint_fields: tuple[str, ...] = setting(("proto.tls.ja4",), source="AS-585", doc="client tool fingerprints swapped inside a tool class")
    energy_hidden: int = setting(1024, source="AS-586", doc="width of the energy-SSL network")
    energy_blocks: int = setting(4, source="AS-586", doc="residual blocks of the energy-SSL network")
    langevin: LangevinConfig = setting(LangevinConfig(), source="AS-586", doc="annealed Langevin sampler")

    def __post_init__(self) -> None:
        if self.energy_acceptance not in ENERGY_ACCEPTANCE_MODES:
            raise InvariantViolation(f"generator: energy_acceptance must be one of {ENERGY_ACCEPTANCE_MODES}")
        names = [f for f, _ in self.family_weights]
        unknown = [f for f in names if f not in PRODUCER_FAMILIES]
        if unknown or len(set(names)) != len(names):
            raise InvariantViolation(f"generator: family_weights must name distinct families of {PRODUCER_FAMILIES}")
        if any(w < 0 for _, w in self.family_weights) or (self.family_weights and sum(w for _, w in self.family_weights) <= 0):
            raise InvariantViolation("generator: family_weights must be non-negative and not all zero")
        lo, hi = self.ephemeral_ports
        if not 0 <= lo < hi <= 65535:
            raise InvariantViolation("generator: ephemeral_ports must be 0 <= lo < hi <= 65535")
        seen: set[int] = set()
        for name, ports in self.service_alias_classes:
            if len(set(ports)) != len(ports) or any(not 0 <= p <= 65535 for p in ports) or seen & set(ports):
                raise InvariantViolation(f"generator: service alias class {name!r} must list distinct ports, each in one class")
            seen |= set(ports)
        if self.energy_hidden < 1 or self.energy_blocks < 1:
            raise InvariantViolation("generator: energy_hidden and energy_blocks must be >= 1")

    def weight_of(self, family: str) -> float:
        """Sampling weight of one family (1 when no weights are given)."""
        if not self.family_weights:
            return 1.0
        return dict(self.family_weights).get(family, 0.0)


__all__ = ["DEFAULT_FLOW_ONLY_FIELDS", "DEFAULT_SERVICE_ALIAS_CLASSES", "DETERMINISTIC_FAMILIES", "ENERGY_ACCEPTANCE_MODES",
           "GeneratorPolicy", "LEARNED_FAMILIES", "LangevinConfig", "PRODUCER_FAMILIES", "families_for_option"]
