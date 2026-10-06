"""Driving features of a forecast: Expected Gradients over field states, plus the energy-lens shares.

Purpose
-------
The problem statement requires, per prediction, "which specific flags, ports, or flow patterns are
contributing most to the infiltration prediction (via attention weights or SHAP values)"; black-box
outputs are not acceptable. This module attributes the forecast to the **field states** of the
trigger's window (`FieldEncoder.forward(states=…)`, `explain/attribution.py`) and adds the share of
each energy lens in the last refinement step (D-42).

The explained function (AS-417)
-------------------------------
P_inf(K) of `Forecaster.imagine` is a Monte-Carlo mixture over sampled routes (not differentiable). The
attribution explains its differentiable reading along the **mode route** (the heaviest route of the
forecast), with the route's actions held fixed:

    F(f) = 1 − Π_{k=1..K} (1 − h_k(f)),     h_k = hazard of step k under teacher forcing of the mode route

computed by the full chain f → FieldEncoder → CVG-AE (posterior mean) → TSTCT (dense, carry as of the
window start) → TAAFT (R, S, unrolled descent, the same long-term memory keys and carried Imagination
as the forecast; both are constants of the attribution) → Forecaster. During the attribution the Forecaster reads
TAAFT without the STAGED stop-gradient (`joint_phase` set for the call and restored): that flag only
controls where gradients flow, never a value.

Expected Gradients (Erion et al., Nature Machine Intelligence 2021, arXiv:1906.10670; SHAP's
GradientExplainer form, Lundberg & Lee, NeurIPS 2017):

    EG_c = E_{f′∼D, α∼U(0,1)} [ (f_c − f′_c) · ∂F(f′ + α(f − f′)) / ∂f_c ]   summed over the state width and updates

with the background D = the "nothing supplied" baseline: every cell of the window with status
NOT_SUPPLIED (D-41: absence is a fact, so the baseline is "this sensor reported nothing", not zero).
Completeness holds in expectation: Σ_c EG_c = F(f) − F(f′); the gap is returned with every
explanation (`Attribution.completeness_gap`).

Invariants: contributions are per catalogue column; the completeness gap and F(f) are reported, so
an explanation carries its own check.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import torch

from nagahana.datamodel.columnar import CODE_NOT_SUPPLIED
from nagahana.datamodel.fields import Kind
from nagahana.explain.attribution import Attribution, expected_gradients
from nagahana.governance.assumptions import assume
from nagahana.memory.longterm import NeuralMemoryState
from nagahana.models.batch import FieldBatch, ForecastOut, WindowBatch
from nagahana.models.forecaster.losses import NO_TARGET
from nagahana.models.nagahana import NagaHana
from nagahana.models.taaft.imagination import PastImagination
from nagahana.models.tstct.model import CarriedEnvironment
from nagahana.models.vocab import COLUMN_KIND_CODE
from nagahana.physics.term import PhysicsTerm
from nagahana.roles.contracts import DrivingFeature

#: Column kinds whose most contributing *value* is named in the feature label (e.g. "flow.dst_port=31337").
_NAMED_VALUE_KINDS = (COLUMN_KIND_CODE[Kind.CATEGORICAL], COLUMN_KIND_CODE[Kind.BITMASK])


@dataclass
class Explanation:
    """Driving features of one trigger and the evidence behind them."""

    features: list[DrivingFeature]
    column_attribution: dict[str, float]      # column → Σ EG over the window's updates
    f_value: float                            # F(f): P_inf(K) along the mode route (the explained reading)
    completeness_gap: float                   # F(f) − F(baseline) − Σ EG (Monte-Carlo error)
    samples: int


def absent_baseline(fields: FieldBatch) -> FieldBatch:
    """Every cell NOT_SUPPLIED (value NaN): the "nothing reported" background of the attribution."""
    return FieldBatch(values=torch.full_like(fields.values, float("nan")),
                      status=torch.full_like(fields.status, CODE_NOT_SUPPLIED),
                      column_kind=fields.column_kind, column_slot=fields.column_slot, column_names=fields.column_names)


def explain_trigger(model: NagaHana, window: WindowBatch, carry: CarriedEnvironment | None, *, forecast: ForecastOut,
                    longterm: NeuralMemoryState | None, past: PastImagination | None, physics: PhysicsTerm | None, tstct_passes: int, taaft_passes: int,
                    descent_steps: int, samples: int, top: int, generator: torch.Generator | None) -> Explanation:
    """EG attribution of the mode-route P_inf(K) of trigger 0 of a one-window batch (module docstring)."""
    assume("AS-417", by=__name__)
    b, m = 0, 0
    mode = int(forecast.mode_route[b, m])
    acts = forecast.route_actions[b, m, mode]                                     # [K, 2] (technique, entity)
    tech = acts[:, 0].view(1, 1, -1)
    tgt = torch.where(acts[:, 1] >= 0, acts[:, 1], torch.full_like(acts[:, 1], NO_TARGET)).view(1, 1, -1)
    fields = window.fields

    def f(states: torch.Tensor) -> torch.Tensor:
        out = []
        for e in range(states.shape[0]):
            u, _ = model.encode_updates(fields, window, states=states[e:e + 1])
            z, _, _, _ = model.latents(window, u, sample=False)
            env = model.tstct(z, window, passes=tstct_passes, carry=carry)
            an = model.analyse(env, window, passes=taaft_passes, descent_steps=descent_steps, longterm=longterm,
                               past=past, physics=physics, create_graph=True)
            tf = model.forecaster.teacher_forced(an, tech, tgt)
            # Hazards are float64 (D-54); the explained P_inf(K) = 1 − Π(1 − h) stays float64 (no .float()).
            h = tf["heads"]["hazard"][0].double()                                 # [K] float64
            out.append(1.0 - torch.prod(1.0 - h))
        return torch.stack(out)

    joint = model.forecaster.joint_phase
    model.forecaster.joint_phase = True
    try:
        with torch.enable_grad():
            x = model.inputs.field_states(fields).detach()                        # [1, U, C, d]
            base = model.inputs.field_states(absent_baseline(fields)).detach()
            attr: Attribution = expected_gradients(f, x, base, samples=samples, generator=generator)
            with torch.no_grad():
                fx = float(f(x)[0])
    finally:
        model.forecaster.joint_phase = joint
    real = window.update_mask[0]                                                  # [U]
    per_col = (attr.values[0] * real[:, None].to(attr.values.dtype)).sum(0)       # [C]
    names = fields.column_names
    columns = {names[c]: float(per_col[c]) for c in range(len(names))}
    order = sorted(range(len(names)), key=lambda c: -abs(float(per_col[c])))[:top]
    feats: list[DrivingFeature] = []
    for c in order:
        if float(per_col[c]) == 0.0:
            continue
        label = names[c]
        # for categorical / bitmask columns name the value of the update that contributed most (a port, a flag set)
        upd_attr = attr.values[0, :, c] * real.to(attr.values.dtype)
        u_top = int(torch.argmax(upd_attr.abs()))
        v = float(fields.values[0, u_top, c])
        if int(fields.column_kind[c]) in _NAMED_VALUE_KINDS and v == v:            # a port, a protocol, a flag set
            label = f"{names[c]}={int(v)}"
        feats.append(DrivingFeature(feature=label, contribution=float(per_col[c]), method="attribution"))
    return Explanation(features=feats, column_attribution=columns, f_value=fx,
                       completeness_gap=float(attr.completeness_gap[0]), samples=samples)


def lens_features(lens_share: dict[str, torch.Tensor]) -> list[DrivingFeature]:
    """The share of each energy lens in the last descent step at trigger (0, 0) (D-42 decomposition)."""
    return [DrivingFeature(feature=f"energy-lens:{name}", contribution=float(v[0, 0]), method="attribution")
            for name, v in lens_share.items()]


def carry_before(carry: CarriedEnvironment, t_rel: torch.Tensor) -> CarriedEnvironment:
    """Mask carried slots at or after t_rel [B] (seconds relative to the carry origin): the carry as of a window start."""
    keep = carry.mask & (carry.time < t_rel[:, None])
    return dataclasses.replace(carry, mask=keep, bucket=carry.bucket & keep, latest=carry.latest & keep)


__all__ = ["Explanation", "absent_baseline", "carry_before", "explain_trigger", "lens_features"]
