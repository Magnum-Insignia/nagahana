"""The Advisor's action table: D3FEND counter-measures as policy slots, with level, effect and price (build-spec §2.9).

Purpose
-------
π_D chooses among `n_actions` slots (256 at L). A slot means something only through this table: the
MITRE D3FEND technique, its D3FEND tactic (`roles.contracts.D3FENDTactic`), the level at which it acts
(graph: edits contact; sensor: edits observation, D-33), a disruption weight (AS-23) and the
*structural* effect the Forecaster re-imagines under (`ForecastIntervention`). Slots without an
entry are unassigned and masked out of π_D.

Owner sources: [Q-30] ("defend based counters … not actional, only advisory"), [A-15]. Decisions: D-33
(advisory only), D-32 (passive only: no host agents), D-34 (OT availability first). Assumptions: AS-23
(disruption pricing; values here are starting values for the owner to review), AS-254 (criticality
table), AS-255 (effect types and their strengths).

Honesty notes
-------------
- **IDs are not verified in this build.** The technique names and tactics are the well-known D3FEND
  matrix entries as recalled by the engineer; the "D3-…" identifiers must be checked against
  https://d3fend.mitre.org before release (this build has no network access). Every entry carries
  `id_verified=False` until then.
- **Effects are structural and coarse.** isolate = no targeting and no outbound reads; inbound
  filtering = no targeting; outbound filtering = no outbound reads; plane blocking = techniques of a
  plane cannot target the entity (needs a technique → plane table, which AS-20's slots do not yet
  have; without it the action is reported as "effect not modelled" and never ranked as feasible);
  observe = more observability (changes no attacker capability: ΔP_inf comes out ≈ 0 and the value
  shows as information value). Effects are never learned, so the Advisor cannot learn an effect that
  flatters its own objective.
- Host-agent techniques (e.g. Process Spawn Analysis) are listed but infeasible under D-32 (passive
  posture): they show the rule check working and are never proposed as feasible.

Maths: disruption cost of a sequence = Σ_steps criticality(kind(target)) × disruption(action) (AS-23).

Extension points: append entries (never reorder: slot codes are baked into π_D's output layer);
set `id_verified=True` once checked; add an effect type together with its `ForecastIntervention` field.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass

from nagahana.models.vocab import PLANES
from nagahana.roles.contracts import D3FENDTactic

LEVELS: tuple[str, ...] = ("graph", "sensor")


class EffectType(enum.Enum):
    """Structural effect of a counter on the imagined state (module docstring)."""

    ISOLATE = "isolate"                 # no targeting + no outbound reads
    BLOCK_INBOUND = "block_inbound"     # no targeting
    BLOCK_OUTBOUND = "block_outbound"   # no outbound reads
    BLOCK_PLANE = "block_plane"         # techniques of `plane` cannot target the entity
    OBSERVE = "observe"                 # observability up (information value only)


@dataclass(frozen=True)
class D3FENDAction:
    """One action slot. `disruption` ∈ [0, 1] (AS-23); `observe_strength` ∈ [0, 1] is the probability that
    the added observable reveals the target's true stage (AS-255; OBSERVE only)."""

    d3fend_id: str
    name: str
    tactic: D3FENDTactic
    level: str
    effect: EffectType
    disruption: float
    plane: str | None = None
    observe_strength: float = 0.0
    needs_host_agent: bool = False
    id_verified: bool = False
    note: str = ""

    def __post_init__(self) -> None:
        if self.level not in LEVELS:
            raise ValueError("level must be 'graph' or 'sensor'")
        if (self.effect is EffectType.OBSERVE) != (self.level == "sensor"):
            raise ValueError("observe effects are sensor-level and only they are")
        if self.effect is EffectType.BLOCK_PLANE and self.plane not in PLANES:
            raise ValueError("a plane block needs a plane from vocab.PLANES")
        if not 0.0 <= self.disruption <= 1.0 or not 0.0 <= self.observe_strength <= 1.0:
            raise ValueError("disruption and observe_strength must be in [0, 1]")


_T, _E = D3FENDTactic, EffectType
#: The starting table (AS-23, AS-255). Order = slot code. IDs to verify (module docstring).
D3FEND_TABLE: tuple[D3FENDAction, ...] = (
    D3FENDAction("D3-NI", "Network Isolation", _T.ISOLATE, "graph", _E.ISOLATE, 1.0,
                 note="parent technique; full isolation of the target entity"),
    D3FENDAction("D3-BDI", "Broadcast Domain Isolation", _T.ISOLATE, "graph", _E.ISOLATE, 0.8),
    D3FENDAction("D3-ITF", "Inbound Traffic Filtering", _T.ISOLATE, "graph", _E.BLOCK_INBOUND, 0.3),
    D3FENDAction("D3-OTF", "Outbound Traffic Filtering", _T.ISOLATE, "graph", _E.BLOCK_OUTBOUND, 0.3),
    D3FENDAction("D3-DNSDL", "DNS Denylisting", _T.ISOLATE, "graph", _E.BLOCK_PLANE, 0.1, plane="name_resolution"),
    D3FENDAction("D3-AL", "Account Locking", _T.EVICT, "graph", _E.ISOLATE, 0.5,
                 note="for account entities; on other kinds it acts as isolating the entity's sessions"),
    D3FENDAction("D3-CR", "Credential Revoking", _T.EVICT, "graph", _E.BLOCK_PLANE, 0.4, plane="identity"),
    D3FENDAction("D3-NTA", "Network Traffic Analysis", _T.DETECT, "sensor", _E.OBSERVE, 0.05, observe_strength=0.5),
    D3FENDAction("D3-DNSTA", "DNS Traffic Analysis", _T.DETECT, "sensor", _E.OBSERVE, 0.02, observe_strength=0.3),
    D3FENDAction("D3-RTSD", "Remote Terminal Session Detection", _T.DETECT, "sensor", _E.OBSERVE, 0.02, observe_strength=0.4),
    D3FENDAction("D3-PHDURA", "Per Host Download-Upload Ratio Analysis", _T.DETECT, "sensor", _E.OBSERVE, 0.02,
                 observe_strength=0.3),
    D3FENDAction("D3-PMAD", "Protocol Metadata Anomaly Detection", _T.DETECT, "sensor", _E.OBSERVE, 0.02, observe_strength=0.4),
    D3FENDAction("D3-PSA", "Process Spawn Analysis", _T.DETECT, "sensor", _E.OBSERVE, 0.05, observe_strength=0.6,
                 needs_host_agent=True, note="host agent: infeasible under D-32 (passive posture)"),
    D3FENDAction("D3-NM", "Network Mapping", _T.MODEL, "sensor", _E.OBSERVE, 0.0, observe_strength=0.2),
    D3FENDAction("D3-AI", "Asset Inventory", _T.MODEL, "sensor", _E.OBSERVE, 0.0, observe_strength=0.1),
)


def action_table(n_actions: int) -> tuple[D3FENDAction | None, ...]:
    """Slot → action (None = unassigned) for a policy with `n_actions` slots."""
    if n_actions < len(D3FEND_TABLE):
        raise ValueError(f"n_actions = {n_actions} is smaller than the D3FEND table ({len(D3FEND_TABLE)} entries)")
    return tuple(D3FEND_TABLE) + (None,) * (n_actions - len(D3FEND_TABLE))
