"""Network topology: entities, services and reachability of a scenario (P-14).

The structure of a scenario (how many machines of each archetype, which segments, which Purdue
levels, who can reach whom) is fixed and the same for every world of that scenario; only per-world
attributes (which host is vulnerable, where credentials sit, where the attacker starts, benign rates
and timing) vary (AS-800, AS-818). Fixing the structure is what makes a batch of worlds a stack of
equal-shaped arrays that `jax.vmap` can run in one call.

Enterprise layout: workstations and admin stations spread over `enterprise_segments` user segments;
file, application and database servers in a server segment; domain controllers in the identity
segment; public web and mail hosts in a DMZ; a gateway; a cloud-egress and an internet peer; one
multicast group for name resolution. OT layout follows the Purdue reference model: engineering
workstations, historians and an OT DMZ at level 3; HMIs and SCADA servers at level 2; PLCs and RTUs
at level 1; field devices at level 0.

Reachability (who may open a connection to whom) encodes the segmentation and the IEC 62443
zone-and-conduit policy: user segments reach the server and identity segments and the egress; the
internet reaches only the DMZ; the enterprise reaches OT only through the OT DMZ when
`ot_from_enterprise`; inside OT a level reaches its own and the adjacent levels; engineering
workstations reach the controllers they program. The matrix is the ground truth that benign sessions
and attacker movement both obey (AS-806, AS-813).
"""

from __future__ import annotations

import ipaddress
from dataclasses import dataclass

import numpy as np

from nagahana.worldsim import vocab
from nagahana.worldsim.config import TopologyConfig


@dataclass(frozen=True)
class Topology:
    """Static structure of a scenario. All arrays are indexed by entity row 0 ... V-1.

    archetype      [V] int, vocab.ARCHETYPE_CODE
    kind           [V] object, datamodel entity kind
    domain         [V] int, 0 enterprise / 1 ot
    purdue         [V] int, Purdue level or -1
    internal       [V] bool
    segment        [V] int, segment id (OT levels use 100 + level)
    holds_data     [V] bool
    control        [V] bool, can issue OT control commands
    is_ot_device   [V] bool, archetype kind is ot_device
    address        [V] object, IPv4 string (the entity key)
    exposes        [V, S] bool, services each entity offers (server side)
    reach          [V, V] bool, reach[i, j] = i may initiate a connection to j
    """

    archetype: np.ndarray
    kind: np.ndarray
    domain: np.ndarray
    purdue: np.ndarray
    internal: np.ndarray
    segment: np.ndarray
    holds_data: np.ndarray
    control: np.ndarray
    is_ot_device: np.ndarray
    address: np.ndarray
    exposes: np.ndarray
    reach: np.ndarray
    name: str

    @property
    def n_entities(self) -> int:
        return int(self.archetype.shape[0])

    def rows_of(self, archetype_name: str) -> np.ndarray:
        """Entity rows of one archetype."""
        return np.nonzero(self.archetype == vocab.ARCHETYPE_CODE[archetype_name])[0]


#: Archetypes added automatically so benign name resolution and egress always have a target.
_ALWAYS = ("internet", "cloud_egress", "multicast_group")

#: Domain codes stored in the entity table (0 enterprise, 1 ot), used by the reachability policy.
_DOM_ENTERPRISE = 0
_DOM_OT = 1

#: Enterprise segment ids.
_SEG_DMZ = 90
_SEG_IDENTITY = 80
_SEG_SERVER = 70
_SEG_EGRESS = 60


def _segment_of(name: str, user_segment: int, arche: vocab.Archetype) -> int:
    # Which segment a machine of this archetype sits in.
    if arche.domain == vocab.OT:
        return 100 + max(arche.purdue, 0)
    if name in ("dmz_web", "dmz_mail"):
        return _SEG_DMZ
    if name == "domain_controller":
        return _SEG_IDENTITY
    if name in ("file_server", "app_server", "database"):
        return _SEG_SERVER
    if name in ("cloud_egress", "internet", "gateway", "multicast_group"):
        return _SEG_EGRESS
    return user_segment


def _address(name: str, segment: int, host_number: int, ordinal: int) -> str:
    # A deterministic, globally unique IPv4 address. Internet peers get a routable-looking address;
    # everything else sits in 10.0.0.0/8 by segment, numbered within the segment so no two machines
    # share an address. The multicast group gets an IPv4 multicast address (D-47).
    if name == "multicast_group":
        return str(ipaddress.ip_address(0xEF000001 + ordinal))      # 239.0.0.x
    if name == "internet":
        return str(ipaddress.ip_address(0xCB007100 + ordinal))      # 203.0.113.x (TEST-NET-3)
    if name == "cloud_egress":
        return str(ipaddress.ip_address(0xC6336400 + ordinal))      # 198.51.100.x (TEST-NET-2)
    third = segment if segment < 100 else 20 + (segment - 100)       # OT levels into 10.20..10.25
    return f"10.{third}.0.{10 + host_number}"


@dataclass(frozen=True)
class _Row:
    """One placed entity while the topology is being built (typed, before it becomes arrays)."""

    row: int
    name: str
    code: int
    kind: str
    domain: int
    purdue: int
    internal: bool
    seg: int
    holds_data: bool
    control: bool
    ot_device: bool
    address: str


def _reachable(src: _Row, dst: _Row, cfg: TopologyConfig) -> bool:
    # Reachability policy (module docstring).
    if src.row == dst.row:
        return False
    s_name, d_name = src.name, dst.name
    s_dom, d_dom = src.domain, dst.domain
    s_pur, d_pur = src.purdue, dst.purdue
    s_seg, d_seg = src.seg, dst.seg
    s_int = src.internal
    # Internet reaches only the DMZ (public-facing services).
    if s_name == "internet":
        return d_name in ("dmz_web", "dmz_mail")
    # Internal to internet / cloud egress is allowed (web browsing, cloud sync).
    if d_name in ("internet", "cloud_egress"):
        return bool(s_int)
    # Name resolution to the multicast group from any internal host.
    if d_name == "multicast_group":
        return bool(s_int)
    if s_name == "multicast_group":
        return False
    # Enterprise east-west (domain codes: 0 enterprise, 1 ot).
    if s_dom == _DOM_ENTERPRISE and d_dom == _DOM_ENTERPRISE:
        if d_name in ("dmz_web", "dmz_mail"):
            return True                                             # internal may use the DMZ services
        if d_name in ("file_server", "app_server", "database", "domain_controller", "gateway"):
            return cfg.allow_cross_segment or s_seg == d_seg or s_seg < 70
        # workstation to workstation only within a segment
        return s_seg == d_seg
    # Enterprise into OT only through the OT DMZ (a conduit).
    if s_dom == _DOM_ENTERPRISE and d_dom == _DOM_OT:
        return bool(cfg.ot_from_enterprise) and d_name == "ot_dmz"
    if s_dom == _DOM_OT and d_dom == _DOM_ENTERPRISE:
        return s_name in ("ot_dmz", "historian") and d_name in ("file_server", "app_server", "database")
    # Inside OT: adjacent Purdue levels, plus engineering workstations reaching the controllers.
    if s_dom == _DOM_OT and d_dom == _DOM_OT:
        if s_name == "engineering_workstation" and d_name in ("plc", "rtu", "field_device", "scada_server", "hmi"):
            return True
        if s_name in ("hmi", "scada_server") and d_name in ("plc", "rtu", "field_device"):
            return True
        if s_name in ("plc", "rtu") and d_name == "field_device":
            return True
        return abs(s_pur - d_pur) <= 1
    return False


def build_topology(cfg: TopologyConfig) -> Topology:
    """Build the static `Topology` of a scenario from its `TopologyConfig`."""
    counts = dict(cfg.counts)
    for extra in _ALWAYS:
        counts.setdefault(extra, 1)
    seg_host_count: dict[int, int] = {}
    rows: list[_Row] = []
    for name in (a.name for a in vocab.ARCHETYPES):            # placement in archetype order (stable)
        n = counts.get(name, 0)
        arche = vocab.archetype(name)
        for ordinal in range(n):
            if arche.domain == vocab.ENTERPRISE and name in ("workstation", "admin_workstation"):
                seg = ordinal % cfg.enterprise_segments
            else:
                seg = _segment_of(name, 0, arche)
            host_number = seg_host_count.get(seg, 0)
            seg_host_count[seg] = host_number + 1
            rows.append(_Row(
                row=len(rows), name=name, code=vocab.ARCHETYPE_CODE[name], kind=arche.kind,
                domain=_DOM_ENTERPRISE if arche.domain == vocab.ENTERPRISE else _DOM_OT,
                purdue=arche.purdue, internal=arche.internal, seg=seg,
                holds_data=arche.holds_data, control=arche.control_capable,
                ot_device=arche.kind == "ot_device",
                address=_address(name, seg, host_number, ordinal),
            ))
    if not rows:
        raise ValueError("topology has no entities; set some counts")
    v = len(rows)
    archetype = np.array([r.code for r in rows], dtype=np.int64)
    kind = np.array([r.kind for r in rows], dtype=object)
    domain = np.array([r.domain for r in rows], dtype=np.int64)
    purdue = np.array([r.purdue for r in rows], dtype=np.int64)
    internal = np.array([r.internal for r in rows], dtype=bool)
    segment = np.array([r.seg for r in rows], dtype=np.int64)
    holds_data = np.array([r.holds_data for r in rows], dtype=bool)
    control = np.array([r.control for r in rows], dtype=bool)
    is_ot_device = np.array([r.ot_device for r in rows], dtype=bool)
    address = np.array([r.address for r in rows], dtype=object)

    exposes = np.zeros((v, vocab.N_SERVICES), dtype=bool)
    for r in rows:
        for svc in vocab.DEFAULT_SERVICES.get(r.name, ()):
            exposes[r.row, vocab.SERVICE_CODE[svc]] = True

    reach = np.zeros((v, v), dtype=bool)
    for i in range(v):
        for j in range(v):
            reach[i, j] = _reachable(rows[i], rows[j], cfg)
    return Topology(
        archetype=archetype, kind=kind, domain=domain, purdue=purdue, internal=internal,
        segment=segment, holds_data=holds_data, control=control, is_ot_device=is_ot_device,
        address=address, exposes=exposes, reach=reach, name="topology",
    )


__all__ = ["Topology", "build_topology"]
