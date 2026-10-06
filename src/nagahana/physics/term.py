"""The shared physics term Phi_phys: one term, used by every component that produces states (D-37).

Definition
----------
    Phi_phys(x) = sum_c  w_c * || m_c * r_c(x) ||^2,        x in C

- r_c: residuals (`residuals.py`), zero inside the physical boundary;
- m_c: 1 on rows where every field the residual reads contributes as evidence, else 0 (D-41): a residual
  never fires on data that is absent;
- w_c > 0: per-residual weights, required (no defaults);
- x in C: hard limits already built into the outputs (`constraints.py`).

The norm is a sum over rows. Batch-size scaling is absorbed by lambda_phys in each component's loss
(`objectives/template.py`). `per_row` returns the same quantity before the sum over rows,
Phi_row(x_i) = sum_c w_c (m_c r_c(x_i))^2, so a consumer can normalise per row or per entity.

Where it is used (D-37)
-----------------------
One common term across all loss functions:
- Simulator: on reconstructions;
- Forecaster: on every imagined step;
- Advisor: on the predicted effects of counters;
- Generator: on variants;
- Verifier: as a hallucination monitor.
The Decoder has none of its own: it decodes, and the losses above score what it outputs.

Target guard (D-18 and the held D-25)
-------------------------------------
The decided use is on model outputs (`Target.MODEL_OUTPUT`): a boundary against hallucination (D-18).
Scoring incoming observations (`Target.OBSERVATION`) turns physics into a telemetry-reliability input of
trust management. That is the held decision D-25, resolved to its option in force:
- "boundary only" (working option, AS-40): a term with `Target.OBSERVATION` is refused (`InvalidOption`);
- "boundary + telemetry reliability input": the term may score observations, and
  `telemetry_violation` turns its per-row values into a per-entity reliability input for TAAFT's
  belief-and-trust lens (`models/taaft/lenses.py`, `LensInputs.telemetry_violation`).

Numerical note
--------------
Masked rows are zeroed before the residual is computed (not only after). Otherwise a NaN in an absent
field would poison the gradient through the masked-out branch: `torch.where` passes a zero upstream
gradient, but 0 x NaN is NaN.
"""

from __future__ import annotations

import enum
from collections.abc import Mapping, Sequence

import torch

from nagahana.core.errors import ConfigMissing, InvalidOption, InvariantViolation
from nagahana.governance import decisions
from nagahana.physics.residuals import Residual

#: The D-25 option under which physics may score incoming telemetry.
TELEMETRY_OPTION = "boundary + telemetry reliability input"


class Target(enum.Enum):
    """What the physics term is applied to."""

    MODEL_OUTPUT = "model_output"   # the decided use (D-18)
    OBSERVATION = "observation"     # a telemetry-reliability input, only under the D-25 option above


def telemetry_scoring_enabled(configured: str | None = None) -> bool:
    """True when the D-25 option in force lets physics score incoming telemetry (working option: no, AS-40)."""
    d = decisions.require("physics-on-telemetry", configured, by=__name__)
    return d.value == TELEMETRY_OPTION


class PhysicsTerm:
    """Phi_phys over a set of residuals. See the module docstring.

    Parameters
    ----------
    residuals:
        The residual instances to include (the selection comes from configuration).
    weights:
        `{residual.name: w_c}` with every w_c > 0. A residual without a weight raises `ConfigMissing`.
    target:
        `Target.MODEL_OUTPUT` (decided) or `Target.OBSERVATION` (only under the D-25 telemetry option).
    """

    def __init__(
        self,
        residuals: Sequence[Residual],
        weights: Mapping[str, float],
        *,
        target: Target,
    ) -> None:
        if target is Target.OBSERVATION and not telemetry_scoring_enabled():
            raise InvalidOption(
                "physics scores incoming telemetry only under the D-25 option "
                f"{TELEMETRY_OPTION!r}; the option in force keeps physics a boundary on model outputs (AS-40, D-18)"
            )
        names = [r.name for r in residuals]
        if len(set(names)) != len(names):
            raise InvariantViolation(f"Duplicate residual names: {names}")
        for n in names:
            if n not in weights:
                raise ConfigMissing(f"No weight for physics residual {n!r} (w_c must be set explicitly).")
            if float(weights[n]) <= 0:
                raise InvariantViolation(f"Weight for {n!r} must be > 0.")
        self.residuals = tuple(residuals)
        self.weights = {n: float(weights[n]) for n in names}
        self.target = target

    def per_residual(
        self,
        values: Mapping[str, torch.Tensor],
        contributing: Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """{residual name: w_c (m_c r_c(x_i))^2 per row} as tensors [N] (zeros where the residual is masked).

        values: field ID -> tensor [N] (any content where not contributing). contributing: field ID -> bool
        tensor [N], True where that field contributes; a field missing from this mapping counts as not
        contributing on every row.
        """
        n_rows, ref = 0, None
        if values:
            ref = next(iter(values.values()))
            n_rows = int(ref.shape[0])
        device = ref.device if ref is not None else None
        out: dict[str, torch.Tensor] = {}
        for r in self.residuals:
            mask = torch.ones(n_rows, dtype=torch.bool, device=device)
            for f in r.fields:
                m_f = contributing.get(f)
                mask = mask & (m_f.to(device=device, dtype=torch.bool) if m_f is not None
                               else torch.zeros(n_rows, dtype=torch.bool, device=device))
            if any(f not in values for f in r.fields):
                # A field absent from `values` entirely: nothing to check, the residual is masked out.
                out[r.name] = torch.zeros(n_rows, device=device)
                continue
            safe = {f: torch.where(mask, values[f], torch.zeros_like(values[f])) for f in r.fields}
            res = torch.where(mask, r(safe), torch.zeros((), device=device))
            out[r.name] = self.weights[r.name] * res**2
        return out

    def per_row(self, values: Mapping[str, torch.Tensor], contributing: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Phi_row(x_i) = sum_c w_c (m_c r_c(x_i))^2: tensor [N] (the term before its sum over rows)."""
        parts = self.per_residual(values, contributing)
        n_rows = int(next(iter(values.values())).shape[0]) if values else 0
        if not parts:
            return torch.zeros(n_rows)
        return torch.stack(list(parts.values()), dim=0).sum(0)

    def __call__(
        self,
        values: Mapping[str, torch.Tensor],
        contributing: Mapping[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return (Phi_phys, per-residual weighted terms), each summed over rows (see `per_residual`)."""
        breakdown = {name: rows.sum() for name, rows in self.per_residual(values, contributing).items()}
        if not breakdown:
            return torch.zeros(()), breakdown
        total = torch.stack(list(breakdown.values())).sum()
        return total, breakdown


def telemetry_violation(
    term: PhysicsTerm,
    values: Mapping[str, torch.Tensor],
    contributing: Mapping[str, torch.Tensor],
    entity: torch.Tensor,
    n_entities: int,
) -> torch.Tensor:
    """Per-entity physics violation of observed records, the D-25 reliability input: float32 [V].

        v_e = log(1 + (1 / n_e) sum over rows i of entity e of Phi_row(x_i)),   n_e = max(1, rows of e)

    `term` must score observations (`Target.OBSERVATION`, available only under the D-25 telemetry option).
    `entity` long [N] gives the entity of each record row (-1: no entity). The log keeps the zero set and the
    ordering while bounding the scale of raw-unit violations (as the physics lens does with its
    log(1 + Phi / n)); an entity without records gets 0 (absence of evidence is not a violation, D-41).
    """
    if term.target is not Target.OBSERVATION:
        raise InvalidOption("telemetry_violation needs a term that scores observations (Target.OBSERVATION)")
    rows = term.per_row(values, contributing).detach().double()                       # [N]
    if entity.shape != rows.shape:
        raise InvariantViolation(f"entity has shape {tuple(entity.shape)}, the records {tuple(rows.shape)}")
    if n_entities < 0 or (entity.numel() and int(entity.max()) >= n_entities):
        raise InvariantViolation("record entity index outside [0, n_entities)")
    ok = entity >= 0
    e = entity[ok].long()
    total = torch.zeros(n_entities, dtype=torch.float64).index_add(0, e, rows[ok])
    count = torch.zeros(n_entities, dtype=torch.float64).index_add(0, e, torch.ones_like(rows[ok]))
    return torch.log1p(total / count.clamp_min(1.0)).float()
