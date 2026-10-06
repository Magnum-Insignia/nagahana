"""Hidden state and event records of the jump process (P-14).

`DynState` is the carry of the next-event loop: the attacker's progress (control level, privilege,
persistence, knowledge, stolen credentials, staged and exfiltrated data, impact per entity), the
defender's state (isolation, patches, port blocks) and the per-entity suspicion the defender reacts
to. None of it is observable; it is the ground truth the observation models hide (D-41) and the
evaluation scores belief and forecast against.

`EVENT_FIELDS` names the columns of one emitted event. The loop produces one row per candidate slot;
thinned or past-horizon slots have kind 0 (`KIND_NONE`) and are dropped when the trajectory is read.
"""

from __future__ import annotations

from typing import Any, NamedTuple

#: Event kinds.
KIND_NONE = 0
KIND_BENIGN = 1
KIND_ATTACK = 2
KIND_DEFENSE = 3

#: Attacker control levels of an entity.
CONTROL_NONE = 0
CONTROL_USER = 1
CONTROL_ADMIN = 2

#: Defender action codes (DEFENSE events carry one in the `sem` column).
DEF_ISOLATE = 0
DEF_BLOCK = 1
DEF_PATCH = 2

#: Columns of one event row (used by the scan output and by `emit.py`). `cause` is the slot index of
#: the earlier event that established the precondition this event used (the parent in the attack DAG),
#: or -1 for a root event (a benign event, a defender action, or the attacker's first external move).
EVENT_FIELDS: tuple[str, ...] = (
    "kind", "t_us", "initiator", "responder", "service", "technique", "stage", "sem", "fan", "cause",
)


class DynState(NamedTuple):
    """The next-event loop carry. Arrays are over entities (V); scalars are 0-d arrays."""

    t_s: Any              # current time in seconds (float64 scalar)
    foothold: Any         # [V] int, CONTROL_*
    persist: Any          # [V] bool
    knowledge: Any        # [V] bool, discovered by the attacker
    cred_known: Any       # [V] bool, attacker holds a credential valid on this entity
    collected: Any        # [V] bool, data staged from this entity
    c2: Any               # [V] bool, command-and-control established from this entity
    exfiltrated: Any      # [V] bool, exfiltration completed through this egress
    dosed: Any            # [V] bool, denial of service delivered to this entity
    manipulated: Any      # [V] bool, OT control manipulated on this device
    ransomed: Any         # [V] bool, impact (encryption) delivered to this entity
    exposure: Any         # [V] float, suspicion accumulated at this entity
    isolated: Any         # [V] bool, removed from reachability by the defender
    responded: Any        # [V] bool, the defender has already acted on this entity
    patched: Any          # [V, N_VULNS] bool
    blocked: Any          # [V, N_SERVICES] bool
    foothold_event: Any   # [V] int, slot index where the attacker first controlled this entity (-1 none)
    n_attacker: Any       # int scalar, attacker events so far
    saturated: Any        # bool scalar, the event budget filled before the horizon


class EventRow(NamedTuple):
    """One event, as scalar arrays (the per-slot output of the loop)."""

    kind: Any
    t_us: Any
    initiator: Any
    responder: Any
    service: Any
    technique: Any
    stage: Any
    sem: Any
    fan: Any
    cause: Any


__all__ = [
    "CONTROL_ADMIN", "CONTROL_NONE", "CONTROL_USER", "DEF_BLOCK", "DEF_ISOLATE", "DEF_PATCH",
    "EVENT_FIELDS", "KIND_ATTACK", "KIND_BENIGN", "KIND_DEFENSE", "KIND_NONE", "DynState", "EventRow",
]
