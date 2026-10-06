"""Relation planes: the parallel branches of the encoder ([A-01], [A-03]) and how they are formed (D-04).

Each relation plane gets its own parallel branch of hypergraph layers in CVG-AE ("relation-specific
parallel branches", P-08). How planes are formed is the held decision D-04, with three admissible
options, each implemented:

- "declared" (working option, AS-01): the six planes of build-spec section 2.2, formed
  deterministically from protocol and port evidence in the data model (`declared_update_planes`);
- "learned" (AS-720): the encoder learns K planes as soft memberships of the relations of the
  connectivity plane (every flow, so every relation), with no declared plane branch;
- "declared + learned" (AS-720): the declared planes plus K learned planes.

Learned planes are flagged as inferred wherever they are shown (`PlaneSpec.formation`), as D-04 requires.
`formation_policy()` resolves the option in force through `governance.decisions.require`, and
`validate_formation` checks that a graph configuration can realise it.

Declared rules (AS-01, AS-104)
------------------------------
A flow belongs to a port-ruled plane when its responder (destination) port is in the plane's set and its
transport carries ports (TCP 6, UDP 17, SCTP 132) or is unknown. Port numbers follow the IANA Service Name
and Transport Protocol Port Number Registry (https://www.iana.org/assignments/service-names-port-numbers):
Kerberos 88, LDAP 389, LDAPS 636, microsoft-ds (SMB) 445, SSH 22, Telnet 23, ms-wbt-server (RDP) 3389,
WinRM 5985/5986, VNC (rfb) 5900, domain (DNS) 53, LLMNR 5355, netbios-ns 137, mDNS 5353, Modbus (mbap)
502, DNP3 20000, IEC-104 2404, ISO-TSAP (S7) 102, EtherNet/IP 44818, BACnet 47808, OPC UA 4840.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass

import numpy as np

from nagahana.core.errors import InvalidOption, InvariantViolation
from nagahana.governance import decisions


class Formation(enum.Enum):
    """How one plane comes into existence."""

    DECLARED = "declared"    # built deterministically from the data model (protocol and port evidence)
    LEARNED = "learned"      # inferred by the model; flagged as inferred in every view


class PlaneFormation(enum.Enum):
    """The admissible options of D-04 (values equal the option strings of the decision registry)."""

    DECLARED = "declared"
    LEARNED = "learned"
    DECLARED_AND_LEARNED = "declared + learned"


@dataclass(frozen=True)
class PlaneSpec:
    """A relation plane.

    Attributes
    ----------
    name: plane identifier (the branch key in CVG-AE).
    formation: DECLARED or LEARNED.
    description: what relation the plane carries.
    """

    name: str
    formation: Formation
    description: str


#: The six declared planes of AS-01 (build-spec section 2.2), in `vocab.PLANES` order.
DECLARED_PLANES: tuple[PlaneSpec, ...] = (
    PlaneSpec("connectivity", Formation.DECLARED, "every flow: L3/L4 reachability between entities"),
    PlaneSpec("services", Formation.DECLARED, "L7 sessions through a service entity"),
    PlaneSpec("identity", Formation.DECLARED, "authentication and directory: Kerberos, LDAP, LDAPS, SMB"),
    PlaneSpec("remote_admin", Formation.DECLARED, "remote administration: SSH, RDP, WinRM, VNC, Telnet"),
    PlaneSpec("name_resolution", Formation.DECLARED, "name resolution: DNS, LLMNR, NBT-NS, mDNS"),
    PlaneSpec("ot_control", Formation.DECLARED, "OT control and polling: Modbus, DNP3, IEC-104, S7, EtherNet/IP, BACnet, OPC UA"),
)

#: The plane whose relations learned planes are formed from: it holds every flow (AS-01).
SUBSTRATE_PLANE = "connectivity"

#: Port sets of the port-ruled declared planes (AS-01).
DECLARED_PLANE_PORTS: dict[str, frozenset[int]] = {
    "identity": frozenset({88, 389, 636, 445}),
    "remote_admin": frozenset({22, 3389, 5985, 5986, 5900, 23}),
    "name_resolution": frozenset({53, 5355, 137, 5353}),
    "ot_control": frozenset({502, 20000, 2404, 102, 44818, 47808, 4840}),
}
#: IP protocol numbers whose flows carry transport ports (IANA protocol numbers registry).
PORT_TRANSPORTS: frozenset[int] = frozenset({6, 17, 132})


def formation_policy(configured: str | None = None) -> PlaneFormation:
    """The D-04 option in force: `configured`, else the run's configured option, else "declared" (AS-01)."""
    d = decisions.require("relation-plane-formation", configured, by=__name__)
    assert d.value is not None
    return PlaneFormation(d.value)


def validate_formation(planes: tuple[str, ...], learned_planes: int, formation: PlaneFormation) -> None:
    """Raise `InvalidOption` unless a graph configuration can realise the D-04 option `formation`.

    - "declared": no learned planes;
    - "learned": the connectivity plane alone (the substrate of every relation) and >= 1 learned plane;
    - "declared + learned": the connectivity plane among the declared planes and >= 1 learned plane.
    """
    if formation is PlaneFormation.DECLARED and learned_planes != 0:
        raise InvalidOption(f"D-04 option 'declared' takes no learned planes; GraphConfig.learned_planes = {learned_planes}")
    if formation is PlaneFormation.LEARNED and (tuple(planes) != (SUBSTRATE_PLANE,) or learned_planes < 1):
        raise InvalidOption("D-04 option 'learned' needs GraphConfig.planes = ('connectivity',) and learned_planes >= 1")
    if formation is PlaneFormation.DECLARED_AND_LEARNED and (SUBSTRATE_PLANE not in planes or learned_planes < 1):
        raise InvalidOption("D-04 option 'declared + learned' needs the connectivity plane and learned_planes >= 1")


def learned_plane_name(k: int) -> str:
    """Name of learned plane k (flagged as inferred by its `PlaneSpec`)."""
    return f"learned/{k}"


def plane_specs(planes: tuple[str, ...], learned_planes: int, formation: PlaneFormation) -> tuple[PlaneSpec, ...]:
    """The planes of the encoder's branches under `formation`: declared branches, then learned ones.

    Under "learned" the connectivity plane is the substrate only (no branch of its own). A declared plane
    name without a rule in `DECLARED_PLANES` raises `InvariantViolation` (no silent default).
    """
    validate_formation(planes, learned_planes, formation)
    by_name = {p.name: p for p in DECLARED_PLANES}
    out: list[PlaneSpec] = []
    if formation is not PlaneFormation.LEARNED:
        for name in planes:
            if name not in by_name:
                raise InvariantViolation(f"plane {name!r} has no declared rule (AS-01)")
            out.append(by_name[name])
    out += [PlaneSpec(learned_plane_name(k), Formation.LEARNED,
                      f"learned plane {k}: soft membership of connectivity relations (inferred, D-04)")
            for k in range(learned_planes)]
    return tuple(out)


def declared_update_planes(
    planes: tuple[str, ...],
    *,
    dst_port: np.ndarray,
    protocol: np.ndarray,
    has_service: np.ndarray,
) -> np.ndarray:
    """Planes of each state update by the AS-01 rules: bool [U, len(planes)].

    - `connectivity`: every flow;
    - `services`: the update goes through a service entity (`has_service`, an L7 session);
    - port-ruled planes (`DECLARED_PLANE_PORTS`): the destination port is in the plane's set and the
      protocol is a port-carrying transport or unknown (NaN). Unknown ports (NaN) match no port rule.

    dst_port, protocol: float [U] with NaN for unknown (absence is not zero, D-41).
    A plane name without a declared rule raises `InvariantViolation`: no silent default.
    """
    from nagahana.governance.assumptions import assume

    assume("AS-01", by=__name__)
    port = np.asarray(dst_port, dtype=np.float64)
    proto = np.asarray(protocol, dtype=np.float64)
    known_port = np.isfinite(port)
    port_int = np.where(known_port, port, -1.0).astype(np.int64)
    transport_ok = ~np.isfinite(proto) | np.isin(np.where(np.isfinite(proto), proto, -1).astype(np.int64),
                                                 list(PORT_TRANSPORTS))
    out = np.zeros((port.shape[0], len(planes)), dtype=bool)
    for i, name in enumerate(planes):
        if name == "connectivity":
            out[:, i] = True
        elif name == "services":
            out[:, i] = np.asarray(has_service, dtype=bool)
        elif name in DECLARED_PLANE_PORTS:
            out[:, i] = known_port & transport_ok & np.isin(port_int, sorted(DECLARED_PLANE_PORTS[name]))
        else:
            raise InvariantViolation(f"plane {name!r} has no declared rule (AS-01); add one before using it")
    return out
