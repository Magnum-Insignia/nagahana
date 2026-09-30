"""Relation planes: the "vertical layers" of the model [A-01], [A-03].

The owner's framing
-------------------
"Each veritical layer is a HGNN/MGNN basically complex gnn, and their modeling/alignment is based on
mathematical topological structure, properties & computation" [A-03]. "Vertical parallelism is
the parallel execution of units" [A-01]. Each plane gets its own parallel branch in CVG-AE. The
proposed standard term is "relation-specific parallel branches" (P-08).

Held: how planes are formed (D-04)
----------------------------------
Options on record: planes **declared** from physics/protocols (auditable, like RouteNet's
structure-from-known-facts, refs.md#L3180); **learned** (flexible, data-hungry); or **declared +
learned**, with learned planes flagged as inferred. Code that forms planes calls
`formation_policy()`, which raises until the owner decides.

The planes listed below are the *examples* drawn in diagram 01. They are not a decision.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass

from nagahana.governance import decisions


class Formation(enum.Enum):
    """How a plane comes into existence (options for D-04)."""

    DECLARED = "declared"    # built deterministically from the data model (physics / protocol)
    LEARNED = "learned"      # inferred by the model; must be flagged as inferred in every view


@dataclass(frozen=True)
class PlaneSpec:
    """A relation plane.

    Attributes
    ----------
    name: plane identifier (used as the branch key in CVG-AE).
    formation: DECLARED or LEARNED.
    description: what relation the plane carries.
    """

    name: str
    formation: Formation
    description: str


#: Examples from diagram 01 ("example planes"), NOT a decision (D-04 held).
EXAMPLE_PLANES: tuple[PlaneSpec, ...] = (
    PlaneSpec("connectivity", Formation.DECLARED, "L3/L4 reachability and flows between hosts"),
    PlaneSpec("services", Formation.DECLARED, "L7 client–service–server sessions"),
    PlaneSpec("identity", Formation.DECLARED, "accounts, authentication and authorisation (Kerberos, LDAP …)"),
    PlaneSpec("ot_control", Formation.DECLARED, "HMI/PLC control and polling relations (Modbus, DNP3, IEC-104)"),
)


def formation_policy() -> str:
    """Return the decided plane-formation policy; raises `DecisionHeld` while D-04 is held."""
    d = decisions.require("relation-plane-formation")
    assert d.value is not None
    return d.value
