"""Benign traffic: the legitimate sessions the attacker hides among (P-14, AS-804).

Benign sessions arrive as an inhomogeneous Poisson marked process whose intensity follows a
piecewise-constant diurnal and weekly profile (integer weights per hour of day and per day of week),
so the arrival intensity is an exact integer at every instant and bit-identical across backends.
When a benign session arrives, its source is chosen in proportion to a host's activity weight and its
destination and service uniformly among the services that host may legitimately reach (the topology
reachability and the service exposure). Name resolution to the multicast group (D-47) is an eligible
destination for the DNS/LLMNR service even though a group exposes no service of its own.

Periodic OT polling (a historian or SCADA server polling its controllers at a fixed interval) is not
Poisson; it is generated deterministically in `emit.py` as scheduled sessions (AS-805), so this
module covers only the stochastic IT sessions.

Flow shapes (duration, byte and packet asymmetry, flags) per service live in `observe.py`; this module
decides only when a session starts and between whom.
"""

from __future__ import annotations

from typing import Any

from nagahana.worldsim import vocab
from nagahana.worldsim.config import BenignConfig
from nagahana.worldsim.params import TopoArrays, WorldParams
from nagahana.worldsim.state import EventRow

_SECONDS_PER_HOUR = 3600
_SECONDS_PER_DAY = 86400
_DNS = vocab.SERVICE_CODE["dns"]


def wall_clock(xp: Any, t_us: Any, epoch_offset_s: Any, start_epoch_s: float) -> tuple[Any, Any]:
    """Hour of day (0..23) and day of week (0..6) at simulated time t_us (integer arithmetic)."""
    secs = (t_us // 1_000_000) + epoch_offset_s + xp.asarray(int(start_epoch_s), dtype=xp.int64)
    hour = (secs // _SECONDS_PER_HOUR) % 24
    day = (secs // _SECONDS_PER_DAY) % 7
    return hour, day


def benign_rate(xp: Any, params: WorldParams, benign: BenignConfig, t_us: Any, epoch_offset_s: Any,
                start_epoch_s: float) -> Any:
    """Benign micro-rate now (int scalar) = base * diurnal(hour) * weekly(day) / peak (integer)."""
    hour, day = wall_clock(xp, t_us, epoch_offset_s, start_epoch_s)
    d_tab = xp.asarray(benign.diurnal, dtype=xp.int64)
    w_tab = xp.asarray(benign.weekly, dtype=xp.int64)
    d = d_tab[hour]
    w = w_tab[day]
    peak = int(max(benign.diurnal)) * int(max(benign.weekly))
    return (params.benign_base_micro * d * w) // xp.asarray(peak, dtype=xp.int64)


def _eligible(xp: Any, topo: TopoArrays, params: WorldParams, state: Any, onehot_src: Any) -> Any:
    # [V, S] bool: services the one-hot source may legitimately reach now (plus DNS to a group).
    reach_eff = topo.reach & (~state.isolated)[:, None] & (~state.isolated)[None, :]
    reach_row = xp.any(reach_eff & onehot_src[:, None], axis=0)         # [V]: src can reach j
    exposes_eff = topo.exposes & (~state.blocked)
    elig = reach_row[:, None] & exposes_eff                             # [V, S]
    is_multicast = topo.archetype == vocab.ARCHETYPE_CODE["multicast_group"]
    dns_onehot = xp.arange(vocab.N_SERVICES, dtype=xp.int64) == _DNS
    group_dns = (reach_row & is_multicast)[:, None] & dns_onehot[None, :]
    return elig | group_dns


def benign_candidate(
    xp: Any, topo: TopoArrays, params: WorldParams, benign: BenignConfig, state: Any,
    t_us: Any, d_src: Any, d_pair: Any,
) -> tuple[EventRow, Any]:
    """The candidate benign session when the benign channel fires (initiator, responder, service).

    Returns the event row and a boolean `moved` (False when the chosen source has no reachable
    service, in which case the caller treats the slot as a no-op).
    """
    v = topo.archetype.shape[0]
    rows = xp.arange(v, dtype=xp.int64)
    weight = xp.where(state.isolated, xp.asarray(0, dtype=xp.int64), params.host_weight)   # [V] int
    total_w = xp.sum(weight)
    denom = xp.maximum(total_w, xp.asarray(1, dtype=xp.int64))
    sel = d_src % denom
    cum = xp.cumsum(weight)
    src = xp.argmax((cum > sel).astype(xp.int64)).astype(xp.int64)
    src = xp.where(total_w > 0, src, xp.asarray(-1, dtype=xp.int64))
    src_safe = xp.maximum(src, xp.asarray(0, dtype=xp.int64))
    onehot_src = rows == src_safe

    elig = _eligible(xp, topo, params, state, onehot_src)               # [V, S]
    flat = elig.reshape(-1)
    count = xp.sum(flat.astype(xp.int64))
    denom2 = xp.maximum(count, xp.asarray(1, dtype=xp.int64))
    rank = (d_pair % denom2) + 1
    pos = xp.cumsum(flat.astype(xp.int64))
    pick = flat & (pos == rank)
    flat_idx = xp.argmax(pick.astype(xp.int64)).astype(xp.int64)
    dst = flat_idx // vocab.N_SERVICES
    svc = flat_idx % vocab.N_SERVICES

    moved = (src >= 0) & (count > 0)
    kind = xp.where(moved, xp.asarray(1, dtype=xp.int64), xp.asarray(0, dtype=xp.int64))
    event = EventRow(
        kind=kind, t_us=t_us, initiator=src_safe, responder=dst, service=svc,
        technique=xp.asarray(-1, dtype=xp.int64), stage=xp.asarray(0, dtype=xp.int64),
        sem=xp.asarray(-1, dtype=xp.int64), fan=xp.asarray(1, dtype=xp.int64),
        cause=xp.asarray(-1, dtype=xp.int64),
    )
    return event, moved


__all__ = ["benign_candidate", "benign_rate", "wall_clock"]
