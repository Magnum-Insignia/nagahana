"""Fixed vocabularies shared by every component: codes are part of the model's interface.

A code (the integer behind a stage, a plane, a role …) is baked into embedding tables and heads, so
it must have one definition. Changing an order here changes the meaning of trained weights; append,
never reorder.

Sources
-------
- ATT&CK Enterprise tactics and their IDs: https://attack.mitre.org/tactics/enterprise/ (14 tactics).
  Stage class 0 is "none" (no adversarial stage), so the stage head has 15 classes (AS-19).
- Entity kinds: `datamodel/records.py` ENTITY_KINDS (D-47 added `multicast`).
- Planes and hyperedge kinds: assumptions AS-01 and AS-02 (standing in for held D-04).
- Observation statuses: `datamodel/status.py` (P-03 taxonomy, D-41 principle) + a MASK code used only
  by self-supervised masking (never produced by an adapter).
"""

from __future__ import annotations

from nagahana.datamodel.columnar import STATUS_ORDER
from nagahana.datamodel.fields import Kind

# --------------------------------------------------------------------------- ATT&CK stages (AS-19)
#: (stage name, ATT&CK tactic ID); index = class code. Order follows the ATT&CK matrix columns.
STAGES: tuple[tuple[str, str], ...] = (
    ("none", ""),
    ("reconnaissance", "TA0043"),
    ("resource_development", "TA0042"),
    ("initial_access", "TA0001"),
    ("execution", "TA0002"),
    ("persistence", "TA0003"),
    ("privilege_escalation", "TA0004"),
    ("defense_evasion", "TA0005"),
    ("credential_access", "TA0006"),
    ("discovery", "TA0007"),
    ("lateral_movement", "TA0008"),
    ("collection", "TA0009"),
    ("command_and_control", "TA0011"),
    ("exfiltration", "TA0010"),
    ("impact", "TA0040"),
)
STAGE_CODE: dict[str, int] = {name: i for i, (name, _) in enumerate(STAGES)}
N_STAGES = len(STAGES)

#: Kill-chain progress used by the adversary reward (AS-17): position along the matrix, 0 for none.
STAGE_PROGRESS: tuple[float, ...] = tuple(i / (N_STAGES - 1) for i in range(N_STAGES))

#: Infiltration states (AS-18, standing in for held D-03a): an *internal* entity in one of these.
INFILTRATION_STAGES: frozenset[int] = frozenset(
    STAGE_CODE[s]
    for s in ("execution", "persistence", "privilege_escalation", "defense_evasion", "credential_access",
              "discovery", "lateral_movement", "collection", "command_and_control", "exfiltration", "impact")
)

# --------------------------------------------------------------------------- entities and roles
NODE_KINDS: tuple[str, ...] = ("host", "service", "account", "ot_device", "external", "subnet", "application", "multicast")
NODE_KIND_CODE: dict[str, int] = {k: i for i, k in enumerate(NODE_KINDS)}

#: Role of an entity in an update (flow-adapter convention: 0 initiator, 1 responder, 2 service).
ROLES: tuple[str, ...] = ("initiator", "responder", "service", "none")
ROLE_CODE: dict[str, int] = {r: i for i, r in enumerate(ROLES)}

# --------------------------------------------------------------------------- planes (AS-01, AS-02)
PLANES: tuple[str, ...] = ("connectivity", "services", "identity", "remote_admin", "name_resolution", "ot_control")
PLANE_CODE: dict[str, int] = {p: i for i, p in enumerate(PLANES)}
HYPEREDGE_KINDS: tuple[str, ...] = ("session", "exchange", "group", "fan")
HYPEREDGE_KIND_CODE: dict[str, int] = {k: i for i, k in enumerate(HYPEREDGE_KINDS)}

# --------------------------------------------------------------------------- statuses and kinds
#: Status codes follow `STATUS_ORDER` (columnar form); MASK is one past the last real status.
N_REAL_STATUSES = len(STATUS_ORDER)
MASK_STATUS = N_REAL_STATUSES
N_STATUS_CODES = N_REAL_STATUSES + 1

#: Column kinds as integer codes (matrix kinds only; identifiers and fingerprints are not columns).
COLUMN_KINDS: tuple[Kind, ...] = (Kind.CONTINUOUS, Kind.COUNT, Kind.CATEGORICAL, Kind.BITMASK, Kind.HISTOGRAM)
COLUMN_KIND_CODE: dict[Kind, int] = {k: i for i, k in enumerate(COLUMN_KINDS)}

# --------------------------------------------------------------------------- provenance tags
PROVENANCE: tuple[str, ...] = ("observed", "believed", "forecast")
