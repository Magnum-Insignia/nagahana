"""The attacker as a stochastic policy over a logical attack graph (P-14, AS-806).

The attacker is a marked point process whose enabled moves are the techniques whose preconditions
hold in the current hidden state, in the style of a MulVAL logical attack graph (Ou, Govindavajhala,
Appel, USENIX Security 2005): each technique reads facts (reachability, an exposed vulnerable
service, a stolen credential, a foothold, a privilege) and, when it fires, writes facts that may
enable further techniques. Forward simulation of this graph is the ground-truth kill chain.

Preconditions and effects are evaluated as array masks, with no data-dependent branching, so the same
code runs in a Python loop (numpy) and inside jax.lax.scan / jax.vmap (jax.numpy). Reachability,
service exposure and vulnerability are the effective ones after the defender's isolation, port blocks
and patches (AS-813). Rates are integers (micro-Hz), so which technique and which target fire is a
bit-identical integer decision (AS-802).

Campaign pacing (vocab.CAMPAIGNS): a technique's base rate is divided by the campaign dwell (a slow
campaign waits between actions) and its exposure increment is scaled by the campaign stealth (a quiet
campaign is slower to raise the defender's suspicion).
"""

from __future__ import annotations

from typing import Any, NamedTuple

from nagahana.worldsim import vocab
from nagahana.worldsim.config import ScenarioConfig
from nagahana.worldsim.params import TopoArrays, WorldParams
from nagahana.worldsim.state import CONTROL_ADMIN, CONTROL_USER, DynState, EventRow

#: Integer resolution of a per-second rate (micro-Hz): rate r per hour is round(r/3600 * RATE_SCALE).
RATE_SCALE = 1_000_000
#: Exposure units: an attacker action adds round(noise * stealth * EXPOSURE_SCALE) at its target.
EXPOSURE_SCALE = 100
#: Fan-out (sessions per event) of a scan sweep and of a denial-of-service flood (observe expands it).
SCAN_FAN = 16
DOS_FAN = 64

_LATERAL_SERVICES = ("smb", "rdp", "ssh", "winrm", "kerberos")
_DATA_SERVICES = ("smb", "historian_api", "https")
_OT_SERVICES = ("modbus", "s7comm", "dnp3", "iec104", "ethernet_ip")


class AttackStatic(NamedTuple):
    """Scenario-level integer constants of the attacker (same for every world of a scenario)."""

    rate_micro: Any           # [T] int, per-second micro-rate of each technique
    exp_inc: Any              # [T] int, exposure a firing adds at its target
    stage_code: Any           # [T] int, model stage of each technique
    sem_of: Any               # [T] int, SEM_* category of each technique
    a_max: int                # sum of rate_micro (the attacker thinning bound)
    max_events: int
    enabled: bool


def build_attack_static(xp: Any, scenario: ScenarioConfig) -> AttackStatic:
    """Integer constants of the attacker for a scenario (host-side, then moved to `xp`)."""
    camp = vocab.campaign(scenario.attack.campaign)
    rate_micro: list[int] = []
    exp_inc: list[int] = []
    for t in vocab.TECHNIQUES:
        per_hour = camp.rates.get(t.sem, 0.0) / max(camp.dwell, 1e-9)
        rate_micro.append(int(round(per_hour / 3600.0 * RATE_SCALE)))
        exp_inc.append(int(round(t.noise * camp.stealth * EXPOSURE_SCALE)))
    return AttackStatic(
        rate_micro=xp.asarray(rate_micro, dtype=xp.int64),
        exp_inc=xp.asarray(exp_inc, dtype=xp.int64),
        stage_code=xp.asarray(vocab.TECHNIQUE_STAGE_CODE, dtype=xp.int64),
        sem_of=xp.asarray([t.sem for t in vocab.TECHNIQUES], dtype=xp.int64),
        a_max=int(sum(rate_micro)),
        max_events=int(scenario.attack.max_events),
        enabled=bool(scenario.attack.enabled),
    )


def _first_true(xp: Any, mask: Any) -> Any:
    # Index of the first True entry of a [V] boolean mask, or -1 if none (branchless).
    any_true = xp.any(mask)
    idx = xp.argmax(mask.astype(xp.int64))
    return xp.where(any_true, idx.astype(xp.int64), xp.asarray(-1, dtype=xp.int64))


def _pick_true(xp: Any, mask: Any, r: Any) -> Any:
    # The (r mod count)-th True entry of a [V] boolean mask, or -1 if none (branchless).
    count = xp.sum(mask.astype(xp.int64))
    denom = xp.maximum(count, xp.asarray(1, dtype=xp.int64))
    rank = (r % denom) + 1
    pos = xp.cumsum(mask.astype(xp.int64))
    sel = mask & (pos == rank)
    idx = xp.argmax(sel.astype(xp.int64))
    return xp.where(count > 0, idx.astype(xp.int64), xp.asarray(-1, dtype=xp.int64))


class _Ctx(NamedTuple):
    masks: Any            # [N_SEM, V] bool
    reach_eff: Any        # [V, V] bool
    controlled: Any       # [V] bool
    control_hosts: Any    # [V] bool


def _context(xp: Any, topo: TopoArrays, params: WorldParams, state: DynState) -> _Ctx:
    # All per-step preconditions, computed once (reused by the rate and the candidate).
    reach_eff = topo.reach & (~state.isolated)[:, None] & (~state.isolated)[None, :]
    exposes_eff = topo.exposes & (~state.blocked)
    vuln_eff = params.has_vuln & (~state.patched)
    controlled = state.foothold >= CONTROL_USER
    priv = state.foothold >= CONTROL_ADMIN
    frontier_reach = xp.any(controlled[:, None] & reach_eff, axis=0)
    internal_controlled = xp.any(controlled & topo.internal)
    control_hosts = controlled & topo.control
    control_reach = xp.any(control_hosts[:, None] & reach_eff, axis=0)
    egress = _is_egress(xp, topo)
    egress_reach = xp.any(reach_eff & egress[None, :], axis=1)

    unknown = ~state.knowledge
    none_ctrl = state.foothold == 0
    exploit_present = _service_vuln_present(xp, topo, vuln_eff, exposes_eff, vocab.SEM_EXPLOIT_SERVICE)
    ot_present = _service_vuln_present(xp, topo, vuln_eff, exposes_eff, vocab.SEM_OT_COMMAND)
    local_present = _local_vuln_present(xp, vuln_eff)
    stores_unknown = xp.any(params.cred_store & (~state.cred_known)[None, :], axis=1)
    any_collected = xp.any(state.collected)

    # One [V] mask per SEM_* category, in category-code order, stacked into [N_SEM, V].
    rows_by_sem = [
        frontier_reach & unknown & (~internal_controlled),                                   # SEM_SCAN_EXTERNAL
        frontier_reach & unknown & internal_controlled,                                       # SEM_SCAN_INTERNAL
        state.knowledge & none_ctrl & frontier_reach & topo.internal & exploit_present,       # SEM_EXPLOIT_SERVICE
        state.knowledge & none_ctrl & frontier_reach & topo.internal & state.cred_known,      # SEM_VALID_ACCOUNT
        controlled & topo.internal & (~priv) & local_present,                                 # SEM_PRIV_ESC
        priv & topo.internal & stores_unknown,                                                # SEM_CRED_ACCESS
        controlled & topo.internal & (~state.persist),                                        # SEM_PERSIST
        controlled & topo.internal & (~state.c2) & egress_reach,                              # SEM_C2
        topo.holds_data & topo.internal & (~state.collected) & frontier_reach,                # SEM_COLLECT
        egress & frontier_reach & (~state.exfiltrated) & any_collected,                       # SEM_EXFIL
        frontier_reach & (~state.dosed),                                                      # SEM_DOS
        topo.is_ot_device & (~state.manipulated) & control_reach & ot_present,                # SEM_OT_COMMAND
        controlled & topo.internal & (~state.ransomed),                                       # SEM_RANSOM
    ]
    masks = xp.stack(rows_by_sem, axis=0)
    return _Ctx(masks=masks, reach_eff=reach_eff, controlled=controlled, control_hosts=control_hosts)


def enabled_masks(xp: Any, topo: TopoArrays, params: WorldParams, state: DynState) -> Any:
    """Stack of per-category target masks [N_SEM, V] (targets whose preconditions hold now)."""
    return _context(xp, topo, params, state).masks


def _service_vuln_present(xp: Any, topo: TopoArrays, vuln_eff: Any, exposes_eff: Any, sem: int) -> Any:
    v = topo.archetype.shape[0]
    out = xp.zeros((v,), dtype=bool)
    for k, vul in enumerate(vocab.VULNERABILITIES):
        if vul.enables != sem or vul.service < 0:
            continue
        out = out | (vuln_eff[:, k] & exposes_eff[:, vul.service])
    return out


def _local_vuln_present(xp: Any, vuln_eff: Any) -> Any:
    v = vuln_eff.shape[0]
    out = xp.zeros((v,), dtype=bool)
    for k, vul in enumerate(vocab.VULNERABILITIES):
        if vul.enables == vocab.SEM_PRIV_ESC and vul.service < 0:
            out = out | vuln_eff[:, k]
    return out


def _is_egress(xp: Any, topo: TopoArrays) -> Any:
    internet = topo.archetype == vocab.ARCHETYPE_CODE["internet"]
    cloud = topo.archetype == vocab.ARCHETYPE_CODE["cloud_egress"]
    return internet | cloud


def _exposed_row(xp: Any, exposes_eff: Any, onehot: Any) -> Any:
    return xp.any(exposes_eff & onehot[:, None], axis=0)


def _first_service(xp: Any, exposed_row: Any, priority: tuple[str, ...]) -> Any:
    choice = xp.asarray(-1, dtype=xp.int64)
    for name in reversed(priority):
        code = vocab.SERVICE_CODE[name]
        choice = xp.where(exposed_row[code], xp.asarray(code, dtype=xp.int64), choice)
    return choice


def _exploit_service(xp: Any, topo: TopoArrays, params: WorldParams, state: DynState, onehot: Any) -> Any:
    vuln_eff = params.has_vuln & (~state.patched)
    exposes_eff = topo.exposes & (~state.blocked)
    choice = xp.asarray(-1, dtype=xp.int64)
    for k in range(vocab.N_VULNS - 1, -1, -1):
        vul = vocab.VULNERABILITIES[k]
        if vul.enables != vocab.SEM_EXPLOIT_SERVICE or vul.service < 0:
            continue
        has = xp.any(onehot & vuln_eff[:, k] & exposes_eff[:, vul.service])
        choice = xp.where(has, xp.asarray(vul.service, dtype=xp.int64), choice)
    return choice


def attacker_step(
    xp: Any, topo: TopoArrays, params: WorldParams, static: AttackStatic, state: DynState,
    t_us: Any, slot: Any, d_tech: Any, d_target: Any,
) -> tuple[Any, DynState, EventRow, Any]:
    """Compute the attacker's total rate and its candidate move for one slot (masks computed once).

    Returns the total attacker micro-rate (int scalar), the candidate next state, the candidate event
    and a boolean `moved` (False when no technique is enabled; the caller then treats the slot as a
    no-op). The state and event are consistent whether or not the move is accepted: all effects are
    gated by `moved`, so applying the candidate when the attacker did not actually move is a no-op.
    """
    ctx = _context(xp, topo, params, state)
    masks, reach_eff, controlled, control_hosts = ctx
    counts = xp.sum(masks.astype(xp.int64), axis=1)
    sem_has = (counts > 0).astype(xp.int64)[static.sem_of]
    gate = xp.asarray(1 if static.enabled else 0, dtype=xp.int64)
    time_gate = (t_us >= params.attack_start_us).astype(xp.int64)
    budget_gate = (state.n_attacker < static.max_events).astype(xp.int64)
    prop = static.rate_micro * sem_has * gate * time_gate * budget_gate
    total = xp.sum(prop)

    denom = xp.maximum(total, xp.asarray(1, dtype=xp.int64))
    tsel = d_tech % denom
    cum = xp.cumsum(prop)
    tech = xp.argmax((cum > tsel).astype(xp.int64)).astype(xp.int64)
    tech = xp.where(total > 0, tech, xp.asarray(-1, dtype=xp.int64))
    tech_safe = xp.maximum(tech, xp.asarray(0, dtype=xp.int64))
    sem = static.sem_of[tech_safe]

    v = topo.archetype.shape[0]
    rows = xp.arange(v, dtype=xp.int64)
    target = _pick_true(xp, masks[sem], d_target)
    moved = (tech >= 0) & (target >= 0)
    target_safe = xp.maximum(target, xp.asarray(0, dtype=xp.int64))
    onehot = rows == target_safe

    reach_to_target = xp.any(reach_eff & onehot[None, :], axis=1)
    internet = topo.archetype == vocab.ARCHETYPE_CODE["internet"]
    src_general = _first_true(xp, controlled & reach_to_target)
    src_control = _first_true(xp, control_hosts & reach_to_target)
    src_internet = _first_true(xp, internet)
    src_exfil = _first_true(xp, controlled & state.collected & reach_to_target)
    source = src_general
    source = xp.where(sem == vocab.SEM_OT_COMMAND, src_control, source)
    source = xp.where(sem == vocab.SEM_SCAN_EXTERNAL, src_internet, source)
    source = xp.where((sem == vocab.SEM_EXFIL) & (src_exfil >= 0), src_exfil, source)
    source = xp.where(source >= 0, source, target_safe)
    src_safe = xp.maximum(source, xp.asarray(0, dtype=xp.int64))

    exposed = _exposed_row(xp, topo.exposes & (~state.blocked), onehot)
    svc_lateral = _first_service(xp, exposed, _LATERAL_SERVICES)
    svc_data = _first_service(xp, exposed, _DATA_SERVICES)
    svc_ot = _first_service(xp, exposed, _OT_SERVICES)
    svc_exploit = _exploit_service(xp, topo, params, state, onehot)
    https = xp.asarray(vocab.SERVICE_CODE["https"], dtype=xp.int64)
    service = xp.asarray(-1, dtype=xp.int64)
    service = xp.where(sem == vocab.SEM_EXPLOIT_SERVICE, svc_exploit, service)
    service = xp.where(sem == vocab.SEM_VALID_ACCOUNT, svc_lateral, service)
    service = xp.where(sem == vocab.SEM_CRED_ACCESS, svc_lateral, service)
    service = xp.where(sem == vocab.SEM_COLLECT, svc_data, service)
    service = xp.where(sem == vocab.SEM_OT_COMMAND, svc_ot, service)
    service = xp.where((sem == vocab.SEM_C2) | (sem == vocab.SEM_EXFIL), https, service)

    is_scan = (sem == vocab.SEM_SCAN_EXTERNAL) | (sem == vocab.SEM_SCAN_INTERNAL)
    reach_count = xp.sum(reach_eff[src_safe].astype(xp.int64))
    fan = xp.asarray(1, dtype=xp.int64)
    fan = xp.where(is_scan, xp.minimum(reach_count, xp.asarray(SCAN_FAN, dtype=xp.int64)), fan)
    fan = xp.where(sem == vocab.SEM_DOS, xp.asarray(DOS_FAN, dtype=xp.int64), fan)
    fan = xp.maximum(fan, xp.asarray(1, dtype=xp.int64))

    self_acting = ((sem == vocab.SEM_PRIV_ESC) | (sem == vocab.SEM_PERSIST) | (sem == vocab.SEM_C2)
                   | (sem == vocab.SEM_COLLECT) | (sem == vocab.SEM_RANSOM))
    cause = xp.where(self_acting, state.foothold_event[target_safe], state.foothold_event[src_safe])
    cause = xp.where(moved, cause, xp.asarray(-1, dtype=xp.int64))

    new_state = _apply_effect(xp, params, static, state, sem, onehot, tech_safe, moved, slot)
    stage = xp.where(moved, static.stage_code[tech_safe], xp.asarray(0, dtype=xp.int64))
    event = EventRow(
        kind=xp.where(moved, xp.asarray(2, dtype=xp.int64), xp.asarray(0, dtype=xp.int64)),
        t_us=t_us, initiator=src_safe, responder=target_safe, service=service,
        technique=xp.where(moved, tech, xp.asarray(-1, dtype=xp.int64)), stage=stage,
        sem=xp.where(moved, sem, xp.asarray(-1, dtype=xp.int64)), fan=fan, cause=cause,
    )
    return total, new_state, event, moved


def _apply_effect(
    xp: Any, params: WorldParams, static: AttackStatic, state: DynState,
    sem: Any, onehot: Any, tech_safe: Any, moved: Any, slot: Any,
) -> DynState:
    g = onehot & moved
    is_scan = (sem == vocab.SEM_SCAN_EXTERNAL) | (sem == vocab.SEM_SCAN_INTERNAL)
    is_foothold1 = (sem == vocab.SEM_EXPLOIT_SERVICE) | (sem == vocab.SEM_VALID_ACCOUNT)
    is_priv = sem == vocab.SEM_PRIV_ESC

    knowledge = state.knowledge | (g & is_scan) | (g & is_foothold1)
    lvl1 = xp.where(g & is_foothold1, xp.asarray(CONTROL_USER, dtype=xp.int64), xp.asarray(0, dtype=xp.int64))
    lvl2 = xp.where(g & is_priv, xp.asarray(CONTROL_ADMIN, dtype=xp.int64), xp.asarray(0, dtype=xp.int64))
    foothold = xp.maximum(xp.maximum(state.foothold, lvl1), lvl2)
    gained = g & is_foothold1 & (state.foothold_event < 0)
    foothold_event = xp.where(gained, slot.astype(xp.int64), state.foothold_event)
    persist = state.persist | (g & (sem == vocab.SEM_PERSIST))
    c2 = state.c2 | (g & (sem == vocab.SEM_C2))
    collected = state.collected | (g & (sem == vocab.SEM_COLLECT))
    exfiltrated = state.exfiltrated | (g & (sem == vocab.SEM_EXFIL))
    dosed = state.dosed | (g & (sem == vocab.SEM_DOS))
    manipulated = state.manipulated | (g & (sem == vocab.SEM_OT_COMMAND))
    ransomed = state.ransomed | (g & (sem == vocab.SEM_RANSOM))
    cred_gain = xp.any(params.cred_store & g[:, None], axis=0)
    cred_known = state.cred_known | xp.where(sem == vocab.SEM_CRED_ACCESS, cred_gain, xp.zeros_like(cred_gain))
    exposure = state.exposure + g.astype(xp.int64) * static.exp_inc[tech_safe]
    n_attacker = state.n_attacker + moved.astype(xp.int64)
    return state._replace(
        foothold=foothold, persist=persist, knowledge=knowledge, cred_known=cred_known,
        collected=collected, c2=c2, exfiltrated=exfiltrated, dosed=dosed, manipulated=manipulated,
        ransomed=ransomed, exposure=exposure, foothold_event=foothold_event, n_attacker=n_attacker,
    )


__all__ = [
    "AttackStatic", "DOS_FAN", "EXPOSURE_SCALE", "RATE_SCALE", "SCAN_FAN", "attacker_step",
    "build_attack_static", "enabled_masks",
]
