"""The Advisor's effect model, price, information value and rule checks (build-spec §2.9). No parameters.

Purpose
-------
Turn a counter sequence (slot, target entity) … into
- a `ForecastIntervention` (the structural edit the Forecaster re-imagines under);
- its disruption cost (AS-23);
- its information value (expected entropy reduction of the stage posterior, AS-255);
- rule checks (feasibility before physics).

Decisions: D-32 (passive only → host-agent actions infeasible), D-33 (advisory only), D-34 (OT
availability first: highest criticality). Assumptions: AS-23, AS-254 (criticality table), AS-255
(information-value model), AS-256 (counters take effect at the trigger, all at once).

Maths
-----
- Cost:  c(seq) = Σ_j crit(kind(v_j)) · disruption(a_j).
- Information value (AS-255). A sensor-level action with strength ρ_a on entity v, whose current
  telemetry trust is t_v and stage posterior π_v, reveals v's true stage with probability
  ρ_a·(1 − t_v) and changes nothing otherwise. The expected posterior entropy after the observation is
  (1 − ρ_a(1 − t_v))·H(π_v), so the expected entropy reduction is

      IV(a, v) = ρ_a · (1 − t_v) · H(π_v)        (nats),  H(π) = −Σ_s π_s log π_s

  A sequence's IV sums over its distinct (sensor, entity) pairs, capped per entity at H(π_v) (an
  entity's stage cannot be revealed more than fully).
- Rule checks: assigned slot; valid, active target; no host-agent action (D-32); a sensor action
  needs a sensor point on our side (an internal target) when internal flags are known; a plane block
  needs the technique → plane table; no repeated (action, target) pair.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from nagahana.models.advisor.d3fend import D3FENDAction, EffectType
from nagahana.models.forecaster.model import ForecastIntervention
from nagahana.models.vocab import PLANE_CODE, PLANES

#: One counter step: (slot, target entity index).
Step = tuple[int, int]


def build_intervention(plan: Sequence[Step], table: Sequence[D3FENDAction | None], n_entities: int,
                       technique_plane: torch.Tensor | None, device: torch.device | None = None) -> ForecastIntervention:
    """Structural edits of one trigger ([1, 1, V] tensors) for all steps of `plan` (applied at once, AS-256)."""
    no_target = torch.zeros(1, 1, n_entities, dtype=torch.bool, device=device)
    no_out = torch.zeros_like(no_target)
    blocked = torch.zeros(1, 1, n_entities, len(PLANES), dtype=torch.bool, device=device)
    for slot, v in plan:
        act = table[slot]
        if act is None:
            continue
        if act.effect in (EffectType.ISOLATE, EffectType.BLOCK_INBOUND):
            no_target[0, 0, v] = True
        if act.effect in (EffectType.ISOLATE, EffectType.BLOCK_OUTBOUND):
            no_out[0, 0, v] = True
        if act.effect is EffectType.BLOCK_PLANE and act.plane is not None:
            blocked[0, 0, v, PLANE_CODE[act.plane]] = True
    return ForecastIntervention(no_target=no_target, no_outbound=no_out,
                                blocked_plane=blocked if technique_plane is not None else None,
                                technique_plane=technique_plane)


def disruption_cost(plan: Sequence[Step], table: Sequence[D3FENDAction | None], entity_kind: torch.Tensor | None,
                    criticality: Sequence[float]) -> tuple[float, bool]:
    """c(seq) = Σ crit(kind(v)) · disruption(a). Returns (cost, kinds_known).

    With `entity_kind` None the entity kinds are unknown and every target is priced at the most
    critical kind (a conservative, reported choice; never a silent cheaper default).
    """
    worst = max(criticality)
    cost = 0.0
    for slot, v in plan:
        act = table[slot]
        if act is None:
            continue
        crit = worst if entity_kind is None else float(criticality[int(entity_kind[v])])
        cost += crit * act.disruption
    return cost, entity_kind is not None


def stage_entropy(stage_probs: torch.Tensor) -> torch.Tensor:
    """H(π) = −Σ π log π along the last axis (0·log 0 = 0)."""
    p = stage_probs.double().clamp_min(0.0)
    return -(torch.where(p > 0, p * torch.log(p), torch.zeros_like(p))).sum(-1)


def information_value(plan: Sequence[Step], table: Sequence[D3FENDAction | None], stage_probs: torch.Tensor,
                      trust: torch.Tensor | None) -> float:
    """Expected entropy reduction (nats) of the stage posterior (module docstring, AS-255).

    stage_probs: [V, S] of the trigger; trust: [V] in [0, 1] or None (treated as 0 trust: nothing seen yet).
    """
    h = stage_entropy(stage_probs)                                               # [V]
    gain: dict[int, float] = {}
    seen: set[Step] = set()
    for slot, v in plan:
        act = table[slot]
        if act is None or act.effect is not EffectType.OBSERVE or (slot, v) in seen:
            continue
        seen.add((slot, v))
        t_v = 0.0 if trust is None else float(trust[v])
        gain[v] = gain.get(v, 0.0) + act.observe_strength * (1.0 - t_v) * float(h[v])
    return float(sum(min(g, float(h[v])) for v, g in gain.items()))


def rule_check(plan: Sequence[Step], table: Sequence[D3FENDAction | None], active: torch.Tensor,
               entity_internal: torch.Tensor | None, technique_plane: torch.Tensor | None) -> tuple[str, ...]:
    """Reasons the plan is infeasible by rule (empty = passes). active: bool [V]."""
    reasons: list[str] = []
    seen: set[Step] = set()
    for slot, v in plan:
        act = table[slot] if 0 <= slot < len(table) else None
        if act is None:
            reasons.append(f"slot {slot} has no D3FEND action")
            continue
        if not (0 <= v < active.shape[0]) or not bool(active[v]):
            reasons.append(f"{act.d3fend_id}: target {v} is not an active entity")
        if act.needs_host_agent:
            reasons.append(f"{act.d3fend_id}: needs a host agent (D-32: passive posture)")
        if act.effect is EffectType.OBSERVE and entity_internal is not None and 0 <= v < entity_internal.shape[0] \
                and not bool(entity_internal[v]):
            reasons.append(f"{act.d3fend_id}: no sensor point on an external entity")
        if act.effect is EffectType.BLOCK_PLANE and technique_plane is None:
            reasons.append(f"{act.d3fend_id}: effect not modelled (no technique → plane table)")
        if (slot, v) in seen:
            reasons.append(f"{act.d3fend_id}: repeated on the same target")
        seen.add((slot, v))
    return tuple(reasons)
