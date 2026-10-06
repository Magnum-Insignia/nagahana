"""Per-world attributes sampled from a world key (P-14).

A scenario's structure is shared (`topology.py`); what differs between worlds of a scenario is drawn
here from that world's key: which host carries which vulnerability, where credentials are reusable,
where the attacker starts, how busy each host is, when the campaign begins and the wall-clock phase.
Every draw is addressed by a fixed structural coordinate under a per-purpose stream, so the result is
bit-identical across backends and `jax.vmap` over worlds equals a loop over worlds (AS-801, AS-818).

The topology numeric arrays reach the backends as `TopoArrays` (no object columns), so this module
and `dynamics.py` run under both numpy and jax.numpy unchanged.
"""

from __future__ import annotations

from typing import Any, NamedTuple

from nagahana.worldsim import rng, vocab
from nagahana.worldsim.config import ScenarioConfig
from nagahana.worldsim.topology import Topology

#: Stream identifiers (domain separation; see rng.stream_key). Append-only.
STREAM_VULN = 1
STREAM_CRED = 2
STREAM_ACTIVITY = 3
STREAM_START = 4
STREAM_EPOCH = 5
STREAM_DYNAMICS = 10
STREAM_SESSION = 30
STREAM_OBS_BASE = 40          # sensor s uses STREAM_OBS_BASE + s


class TopoArrays(NamedTuple):
    """Numeric topology arrays passed to the backends (object columns stay in `Topology`)."""

    archetype: Any            # [V] int
    domain: Any               # [V] int (0 enterprise, 1 ot)
    purdue: Any               # [V] int
    internal: Any             # [V] bool
    holds_data: Any           # [V] bool
    control: Any              # [V] bool
    is_ot_device: Any         # [V] bool
    exposes: Any              # [V, S] bool
    reach: Any                # [V, V] bool


def topo_arrays(xp: Any, topo: Topology) -> TopoArrays:
    """Convert a `Topology`'s numeric columns into `xp` arrays."""
    return TopoArrays(
        archetype=xp.asarray(topo.archetype, dtype=xp.int64),
        domain=xp.asarray(topo.domain, dtype=xp.int64),
        purdue=xp.asarray(topo.purdue, dtype=xp.int64),
        internal=xp.asarray(topo.internal, dtype=bool),
        holds_data=xp.asarray(topo.holds_data, dtype=bool),
        control=xp.asarray(topo.control, dtype=bool),
        is_ot_device=xp.asarray(topo.is_ot_device, dtype=bool),
        exposes=xp.asarray(topo.exposes, dtype=bool),
        reach=xp.asarray(topo.reach, dtype=bool),
    )


class WorldParams(NamedTuple):
    """Attributes of one world (arrays over entities, or scalars). Timing and intensity are integers
    so that time gates and benign intensity are bit-identical across backends (AS-802, AS-804)."""

    has_vuln: Any             # [V, N_VULNS] bool
    cred_store: Any           # [V, V] bool, i stores a credential valid on j
    foothold0: Any            # [V] int, initial attacker control level (0/1/2)
    knowledge0: Any           # [V] bool, initially discovered entities
    host_weight: Any          # [V] int, benign source intensity in milli-units (0 for non-senders)
    benign_base_micro: Any    # int scalar, peak benign micro-rate of the whole world
    attack_start_us: Any      # int scalar, campaign start time in microseconds
    epoch_offset_s: Any       # int scalar, per-world wall-clock phase offset in seconds


def _vuln_eligibility(xp: Any, topo: TopoArrays) -> Any:
    # [V, N_VULNS] bool: a host can carry vuln k if it exposes the vuln's service (or k is local).
    v = topo.exposes.shape[0]
    cols = []
    for k in range(vocab.N_VULNS):
        sv = vocab.VULNERABILITIES[k].service
        # A local weakness (service -1) can sit on any internal host; otherwise the host must expose
        # the vulnerable service.
        col = topo.internal if sv < 0 else topo.exposes[:, sv]
        cols.append(xp.asarray(col, dtype=bool).reshape(v, 1))
    return xp.concatenate(cols, axis=1)


def sample_world_params(xp: Any, topo: TopoArrays, scenario: ScenarioConfig, wkey: tuple[Any, Any]) -> WorldParams:
    """Draw one world's attributes from its key."""
    v = topo.archetype.shape[0]
    atk = scenario.attack

    # Vulnerabilities: Bernoulli per eligible (host, vuln) at the configured densities.
    kv = rng.stream_key(xp, wkey, STREAM_VULN)
    iidx = xp.reshape(xp.arange(v * vocab.N_VULNS, dtype=xp.int64), (v, vocab.N_VULNS))
    local = xp.asarray([vocab.VULNERABILITIES[k].service < 0 for k in range(vocab.N_VULNS)], dtype=bool)
    t_service = rng.threshold_u32(atk.vuln_density)
    t_local = rng.threshold_u32(atk.local_vuln_density)
    w0 = rng.bits(xp, kv, iidx, 0)[0]
    draw_lt = rng.threshold_lt(xp, w0, t_service)
    draw_lt_local = rng.threshold_lt(xp, w0, t_local)
    present = xp.where(local[None, :], draw_lt_local, draw_lt)
    has_vuln = present & _vuln_eligibility(xp, topo)
    # Seed a reliable entry: every internal host the internet can reach that exposes a web service
    # carries the matching web exploit, so initial access through the perimeter is always possible
    # (the rest of the chain stays stochastic). This is the scenario's intended way in (AS-806).
    internet = topo.archetype == vocab.ARCHETYPE_CODE["internet"]
    reach_internal = xp.any(internet[:, None] & topo.reach, axis=0) & topo.internal       # [V]
    cols = xp.arange(vocab.N_VULNS, dtype=xp.int64)
    for svc_name, vuln_name in (("http", "web_app_rce"), ("https", "web_app_rce_tls")):
        host = reach_internal & topo.exposes[:, vocab.SERVICE_CODE[svc_name]]              # [V]
        col = cols == vocab.VULN_CODE[vuln_name]                                            # [N_VULNS]
        has_vuln = has_vuln | (host[:, None] & col[None, :])

    # Credential reuse: holder i (admin station, server, DC, OT control host) stores a credential
    # valid on internal reachable host j with probability credential_reuse.
    kc = rng.stream_key(xp, wkey, STREAM_CRED)
    holder_codes = [vocab.ARCHETYPE_CODE[n] for n in (
        "admin_workstation", "file_server", "app_server", "database", "domain_controller",
        "engineering_workstation", "scada_server", "ot_dmz", "historian")]
    is_holder = xp.zeros((v,), dtype=bool)
    for c in holder_codes:
        is_holder = is_holder | (topo.archetype == c)
    pair_idx = xp.reshape(xp.arange(v * v, dtype=xp.int64), (v, v))
    t_reuse = rng.threshold_u32(atk.credential_reuse)
    pw0 = rng.bits(xp, kc, pair_idx, 0)[0]
    pair_draw = rng.threshold_lt(xp, pw0, t_reuse)
    eye = xp.reshape(xp.arange(v, dtype=xp.int64), (v, 1)) == xp.reshape(xp.arange(v, dtype=xp.int64), (1, v))
    # Credential validity is independent of current reachability (the attacker's reach to j is
    # checked at use time by the VALID_ACCOUNT technique, attack.py).
    cred_store = (pair_draw & is_holder[:, None] & topo.internal[None, :] & (~eye))
    # A domain controller stores credentials valid on every internal host (domain trust).
    dc = topo.archetype == vocab.ARCHETYPE_CODE["domain_controller"]
    cred_store = cred_store | (dc[:, None] & topo.internal[None, :] & (~eye))

    # Attacker origin: external campaigns start from the internet peer (foothold level 2); the hosts
    # reachable from it (the DMZ) are the initial knowledge.
    foothold0 = xp.where(internet, xp.asarray(2, dtype=xp.int64), xp.asarray(0, dtype=xp.int64))
    # The attacker initially knows only its own external origin; it must scan to discover the DMZ
    # (so the kill chain opens with reconnaissance), then exploit, then move internally.
    knowledge0 = internet

    # Benign source intensity: internal hosts get a lognormal weight quantised to integer milli-units
    # (so the per-world rate is an exact integer sum, bit-identical across backends).
    ka = rng.stream_key(xp, wkey, STREAM_ACTIVITY)
    weight = rng.lognormal(xp, ka, xp.arange(v, dtype=xp.int64), 0, mu=0.0, sigma=0.5)
    weight_milli = xp.floor(weight * 1000.0 + 0.5).astype(xp.int64)
    host_weight = xp.where(topo.internal, weight_milli, xp.asarray(0, dtype=xp.int64))
    base_const_micro = int(round(scenario.benign.session_rate_per_host_hour / 3600.0 * 1_000_000))
    benign_base_micro = (xp.asarray(base_const_micro, dtype=xp.int64) * xp.sum(host_weight)) // 1000

    # Campaign start and wall-clock phase, both quantised to integers (randomised so a campaign is
    # not pinned to a fixed clock time, D-50).
    ks = rng.stream_key(xp, wkey, STREAM_START)
    us = rng.unit(xp, ks, 0, 0)
    start_s = xp.asarray(atk.start_min_s, dtype=xp.float64) + us * (atk.start_max_s - atk.start_min_s)
    attack_start_us = xp.floor(start_s * 1_000_000.0 + 0.5).astype(xp.int64)
    ke = rng.stream_key(xp, wkey, STREAM_EPOCH)
    ue = rng.unit(xp, ke, 0, 0)
    epoch_offset_s = xp.floor(ue * xp.asarray(scenario.simulation.start_jitter_s, dtype=xp.float64)).astype(xp.int64)

    return WorldParams(
        has_vuln=has_vuln, cred_store=cred_store, foothold0=foothold0, knowledge0=knowledge0,
        host_weight=host_weight, benign_base_micro=benign_base_micro, attack_start_us=attack_start_us,
        epoch_offset_s=epoch_offset_s,
    )


__all__ = [
    "STREAM_DYNAMICS", "STREAM_OBS_BASE", "STREAM_SESSION", "TopoArrays", "WorldParams",
    "sample_world_params", "topo_arrays",
]
