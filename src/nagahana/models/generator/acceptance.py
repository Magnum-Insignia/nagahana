"""Acceptance of Generator variants: hard limits, the physics gate Phi_phys <= tau, and the JEM-style energy range.

Purpose
-------
Every candidate variant passes three checks before it may join a training batch (build-spec section 2.11
items 5 and "Rules"; D-40, D-18, [A-17]):

1. Hard limits (decided principle: the Generator is physics-bounded [A-17]): the exact limits of
   `limits.py` must hold on every row (zero violating rows). Always on.
2. Physics gate (proposal P-11): Phi_phys(variant) <= tau, with the shared term of `physics/term.py` over
   the catalogue residuals that apply to one flow record (flag counts <= packets, bytes <= packets * MTU
   per direction, iat_max <= duration):

       Phi_phys(x) = sum_c w_c || m_c (.) r_c(x) ||^2,      accept iff Phi_phys <= tau   (tau = `physics_tau`)

   with w_c = `physics_weight` (AS-360) on raw-unit residuals. Phi_phys is computed and reported for
   every variant (the decided "Phi reported per variant"); it rejects when P-11 is enabled for the run
   (`require_proposal`; the training run enables P-11 by default, AS-28). Phi is computed with
   `Target.MODEL_OUTPUT`: a variant is a model output. Real telemetry is never scored (D-25 held).
3. Energy acceptance (JEM-style; Grathwohl et al., "Your Classifier is Secretly an Energy Based Model
   and You Should Treat it Like One", ICLR 2020, arXiv:1912.03263): TAAFT's marginal energy E(empty, .)
   (P-09 two readings, AS-16) scores how typical a window is. Calibrated on real training windows, a
   variant is kept only if

       Q_{q_lo}(E_real) <= E(variant) <= Q_{q_hi}(E_real)        (q_lo, q_hi) = `energy_quantiles` (AS-361)

   too high = less typical than almost any real window (likely implausible); too low = more typical
   than almost any real window (likely a degenerate sample). The energy callable is optional: without
   one the check is not performed and the report says so; TAAFT's marginal energy exists after TAAFT
   pretraining, so the training pipeline screens the stage-3 variant pool with it (AS-583).

Plus two pipeline-level reasons recorded by the pipeline: an empty variant, and a variant identical to
its source.

Owner sources: [A-08], [A-17], [Q-06]. Decisions: D-18, D-25 (held: no telemetry scoring), D-37, D-40.
Proposals: P-09, P-11. Assumptions: AS-16, AS-28, AS-360, AS-361, AS-583.
Invariant (tested): a planted violation is rejected; a clean variant is accepted; energy outside the
calibrated quantile range is rejected; without P-11 Phi is reported and does not reject.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass, field

import numpy as np
import torch

from nagahana.core.errors import ConfigMissing, InvariantViolation, ProposalNotEnabled
from nagahana.datamodel.columnar import ColumnarUpdates
from nagahana.governance import decisions
from nagahana.models.config.components import GeneratorConfig
from nagahana.models.generator.assumptions import use
from nagahana.models.generator.limits import PhysicalSetting, check_hard_limits
from nagahana.physics.residuals import FlagCountBound, IATMaxBound, MTUBound, Residual
from nagahana.physics.term import PhysicsTerm, Target

#: A marginal-energy callable: one variant window → E(∅, window) (a float). Supplied by TAAFT's owner.
MarginalEnergy = Callable[[ColumnarUpdates], float]


@dataclass
class AcceptanceReport:
    """Why a variant was accepted or rejected (every number that decided it)."""

    accepted: bool
    reasons: tuple[str, ...]
    hard_violations: dict[str, int]
    phi_phys: float
    phi_breakdown: dict[str, float]
    gate_active: bool
    energy: float | None = None
    energy_range: tuple[float, float] | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)


def physical_setting(cfg: GeneratorConfig) -> PhysicalSetting:
    """The site physics from config. MTU is required (no default: a wrong MTU teaches a false boundary)."""
    if cfg.mtu is None:
        raise ConfigMissing("GeneratorConfig.mtu is required (site-specific; physics/residuals.MTUBound).")
    return PhysicalSetting(mtu=float(cfg.mtu), link_bps=cfg.link_bps)


def gate_physics_term(setting: PhysicalSetting, weight: float) -> PhysicsTerm:
    """Φ_phys over the catalogue residuals that apply to single flow records (weights: AS-360 when used)."""
    residuals: list[Residual] = [FlagCountBound(f) for f in ("syn", "ack", "fin", "rst", "psh", "urg")]
    residuals += [MTUBound("fwd", setting.mtu), MTUBound("bwd", setting.mtu), IATMaxBound()]
    return PhysicsTerm(residuals, {r.name: weight for r in residuals}, target=Target.MODEL_OUTPUT)


def phi_phys(term: PhysicsTerm, cu: ColumnarUpdates) -> tuple[float, dict[str, float]]:
    """Φ_phys of one variant window: field values / contributing masks per row, float64."""
    contrib = cu.contributing_cells()
    values: dict[str, torch.Tensor] = {}
    mask: dict[str, torch.Tensor] = {}
    for j, c in enumerate(cu.columns):
        if c.component is not None:
            continue                                                         # histogram bins: no residual
        values[c.field_id] = torch.from_numpy(np.where(contrib[:, j], cu.values[:, j], 0.0))
        mask[c.field_id] = torch.from_numpy(contrib[:, j].copy())
    if not values or len(cu) == 0:
        return 0.0, {}
    total, parts = term(values, mask)
    return float(total), {k: float(v) for k, v in parts.items()}


class EnergyAcceptance:
    """JEM-style acceptance by the real-data range of TAAFT's marginal energy (AS-361).

    energy_fn: window → E(∅, window). quantiles: (q_lo, q_hi) of the real-energy distribution.
    `calibrate` must be called with energies of *real training* windows before `accepts` is used.
    """

    def __init__(self, energy_fn: MarginalEnergy, *, quantiles: tuple[float, float]) -> None:
        lo, hi = quantiles
        if not 0.0 <= lo < hi <= 1.0:
            raise InvariantViolation("energy quantiles must satisfy 0 ≤ q_lo < q_hi ≤ 1")
        self.energy_fn = energy_fn
        self.quantiles = (float(lo), float(hi))
        self.range: tuple[float, float] | None = None

    def calibrate(self, real_energies: Sequence[float] | np.ndarray) -> tuple[float, float]:
        """Set the accepted range from real training-window energies; returns it."""
        use("AS-361", by=__name__)
        e = np.asarray(real_energies, dtype=np.float64)
        if e.size < 2 or not np.isfinite(e).all():
            raise InvariantViolation("energy calibration needs ≥ 2 finite real-window energies")
        lo, hi = np.quantile(e, self.quantiles)
        self.range = (float(lo), float(hi))
        return self.range

    def calibrate_on(self, windows: Sequence[ColumnarUpdates]) -> tuple[float, float]:
        """Calibrate by scoring real training windows with the energy callable."""
        return self.calibrate([float(self.energy_fn(w)) for w in windows])

    def accepts(self, energy: float) -> bool:
        if self.range is None:
            raise InvariantViolation("EnergyAcceptance used before calibrate()")
        return bool(np.isfinite(energy) and self.range[0] <= energy <= self.range[1])


def physics_gate_active(enabled_proposals: Collection[str]) -> bool:
    """True when the hard Phi_phys gate applies: proposal P-11 is enabled for this run (or decided)."""
    try:
        decisions.require_proposal("generator-physics-gate", enabled_proposals)
    except ProposalNotEnabled:
        return False
    return True


class AcceptanceGate:
    """Hard limits + physics gate + optional energy range. See the module docstring."""

    def __init__(self, setting: PhysicalSetting, *, tau: float, physics_weight: float,
                 energy: EnergyAcceptance | None = None, enabled_proposals: Collection[str] = ()) -> None:
        if not tau >= 0:
            raise InvariantViolation("τ must be ≥ 0")
        self.setting = setting
        self.tau = float(tau)
        self.term = gate_physics_term(setting, physics_weight)
        self.energy = energy
        self.enabled_proposals = frozenset(enabled_proposals)

    @classmethod
    def from_config(cls, cfg: GeneratorConfig, *, energy: EnergyAcceptance | None = None,
                    enabled_proposals: Collection[str] = ()) -> AcceptanceGate:
        return cls(physical_setting(cfg), tau=cfg.physics_tau, physics_weight=cfg.physics_weight, energy=energy,
                   enabled_proposals=enabled_proposals)

    def evaluate(self, cu: ColumnarUpdates) -> AcceptanceReport:
        """Check one candidate variant window."""
        reasons: list[str] = []
        notes: list[str] = []
        if len(cu) == 0:
            reasons.append("empty")
        # 1. exact hard limits (always)
        viol = check_hard_limits(cu.values, cu.contributing_cells(), cu.columns, self.setting)
        counts = {k: int(v.sum()) for k, v in viol.items()}
        reasons += [f"hard:{k}" for k, v in counts.items() if v]
        if self.setting.link_bps is None:
            notes.append("link_rate not checked (no site link rate configured)")
        # 2. Phi_phys: always computed and reported (weights AS-360); it rejects when P-11 is enabled.
        active = physics_gate_active(self.enabled_proposals)
        use("AS-360", by=__name__)
        phi, parts = phi_phys(self.term, cu)
        if active and phi > self.tau:
            reasons.append("physics:phi>tau")
        if not active:
            notes.append("physics gate not enabled for this run (P-11): Phi_phys reported only")
        # 3. JEM-style energy range (optional)
        e_val: float | None = None
        e_rng: tuple[float, float] | None = None
        if self.energy is not None and len(cu):
            e_val = float(self.energy.energy_fn(cu))
            e_rng = self.energy.range
            if not self.energy.accepts(e_val):
                reasons.append("energy:outside-real-range")
        elif self.energy is None:
            notes.append("energy acceptance not performed (no marginal-energy callable)")
        return AcceptanceReport(accepted=not reasons, reasons=tuple(reasons), hard_violations=counts, phi_phys=phi,
                                phi_breakdown=parts, gate_active=active, energy=e_val, energy_range=e_rng,
                                notes=tuple(notes))
