"""The next-event simulation loop and its two backends (P-14, AS-803).

The hidden process is a marked point process: benign sessions, attacker techniques and defender
actions all arrive in continuous time. It is advanced by Ogata's modified thinning (Ogata, IEEE
Trans. Information Theory 27(1), 1981; Lewis and Shedler, Naval Research Logistics 26(3), 1979): a
constant bound a_max on the total intensity gives candidate times at rate a_max (an exponential gap),
and each candidate is accepted into a channel in proportion to that channel's instantaneous
intensity, the remaining mass being thinned away. Because the per-channel intensities are integers
(micro-Hz), the choice of channel, technique, target and session is a bit-identical integer decision
(AS-802); only the exponential gap uses a floating transform, and it is quantised to integer
microseconds, so the event timeline is integer and identical across backends.

The loop is a fixed number of candidate slots (`SimulationConfig.event_slots`), so it has a fixed
shape: a Python loop for the NumPy backend and `jax.lax.scan` for the JAX backend, both calling the
same `step`. Worlds are run together with `equinox.filter_vmap` over the world index, which only
changes the counter-based key, so the batch equals a loop over worlds (AS-801, AS-818). A world that
fills its slots before the horizon is marked saturated; the count is reported, never hidden.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple

import numpy as np

from nagahana.worldsim import attack, benign, rng, vocab
from nagahana.worldsim.attack import RATE_SCALE, AttackStatic, build_attack_static
from nagahana.worldsim.config import ScenarioConfig
from nagahana.worldsim.params import STREAM_DYNAMICS, TopoArrays, WorldParams, sample_world_params, topo_arrays
from nagahana.worldsim.state import (
    DEF_BLOCK,
    DEF_ISOLATE,
    DEF_PATCH,
    EVENT_FIELDS,
    DynState,
    EventRow,
)
from nagahana.worldsim.topology import Topology, build_topology

#: Draw lanes of the dynamics stream (one draw index per slot; distinct lanes per decision).
_LANE_DT = 0
_LANE_CHANNEL = 1
_LANE_TECH = 2
_LANE_TARGET = 3
_LANE_BSRC = 4
_LANE_BPAIR = 5
_LANE_DEF = 6


class DefenderStatic(NamedTuple):
    """Scenario-level integer constants of the defender."""

    rate_micro: int
    threshold_units: int
    weights: tuple[int, int, int]        # (isolate, block, patch)
    enabled: bool


def build_defender_static(scenario: ScenarioConfig) -> DefenderStatic:
    """Integer constants of the defender for a scenario."""
    d = scenario.defender
    return DefenderStatic(
        rate_micro=int(round(d.response_rate_per_hour / 3600.0 * RATE_SCALE)),
        threshold_units=int(round(d.suspicion_threshold * attack.EXPOSURE_SCALE)),
        weights=(int(d.isolate), int(d.block), int(d.patch)),
        enabled=bool(d.enabled),
    )


class Runtime(NamedTuple):
    """Everything a backend needs to simulate one scenario, in one array namespace."""

    scenario: ScenarioConfig
    topo: TopoArrays
    attack_static: AttackStatic
    defender_static: DefenderStatic
    horizon_us: int
    slots: int
    start_epoch_s: float
    seed: int


class Trajectory(NamedTuple):
    """The output of one world: the event timeline and the final hidden state."""

    events: Any           # [slots, len(EVENT_FIELDS)] int64
    final_state: DynState
    saturated: Any        # bool scalar


def build_runtime(xp: Any, scenario: ScenarioConfig, topo: Topology, seed: int) -> Runtime:
    """Assemble the per-scenario runtime in namespace `xp`."""
    return Runtime(
        scenario=scenario,
        topo=topo_arrays(xp, topo),
        attack_static=build_attack_static(xp, scenario),
        defender_static=build_defender_static(scenario),
        horizon_us=int(round(scenario.simulation.horizon_s * 1_000_000)),
        slots=int(scenario.simulation.event_slots),
        start_epoch_s=float(scenario.simulation.start_epoch_s),
        seed=int(seed),
    )


def initial_state(xp: Any, topo: TopoArrays, params: WorldParams) -> DynState:
    """The hidden state at t = 0 (the attacker holds its external origin; nothing else has happened)."""
    v = topo.archetype.shape[0]
    z_bool = xp.zeros((v,), dtype=bool)
    return DynState(
        t_s=xp.asarray(0.0, dtype=xp.float64),
        foothold=params.foothold0,
        persist=z_bool, knowledge=params.knowledge0, cred_known=z_bool,
        collected=z_bool, c2=z_bool, exfiltrated=z_bool, dosed=z_bool, manipulated=z_bool, ransomed=z_bool,
        exposure=xp.zeros((v,), dtype=xp.int64),
        isolated=z_bool, responded=z_bool,
        patched=xp.zeros((v, vocab.N_VULNS), dtype=bool),
        blocked=xp.zeros((v, vocab.N_SERVICES), dtype=bool),
        foothold_event=xp.full((v,), -1, dtype=xp.int64),
        n_attacker=xp.asarray(0, dtype=xp.int64),
        saturated=xp.asarray(False),
    )


def defender_rate(xp: Any, static: DefenderStatic, state: DynState) -> Any:
    """Defender micro-rate now (int scalar): its base rate while some entity is over the threshold."""
    over = (state.exposure >= static.threshold_units) & (~state.responded) & (~state.isolated)
    active = xp.any(over) & bool(static.enabled)
    return xp.where(active, xp.asarray(static.rate_micro, dtype=xp.int64), xp.asarray(0, dtype=xp.int64))


def defender_candidate(
    xp: Any, topo: TopoArrays, static: DefenderStatic, state: DynState, t_us: Any, d_def: Any
) -> tuple[DynState, EventRow, Any]:
    """The candidate defender action: respond to the most suspicious un-handled entity."""
    v = topo.archetype.shape[0]
    rows = xp.arange(v, dtype=xp.int64)
    over = (state.exposure >= static.threshold_units) & (~state.responded) & (~state.isolated)
    score = xp.where(over, state.exposure, xp.asarray(-1, dtype=xp.int64))
    target = xp.argmax(score).astype(xp.int64)
    moved = xp.any(over) & bool(static.enabled)
    onehot = (rows == target) & moved

    wi, wb, wp = static.weights
    wsum = max(wi + wb + wp, 1)
    pick = (d_def % xp.asarray(wsum, dtype=xp.int64))
    is_isolate = pick < wi
    is_block = (pick >= wi) & (pick < wi + wb)
    is_patch = pick >= (wi + wb)

    isolated = state.isolated | (onehot & is_isolate)
    blocked = state.blocked | (onehot[:, None] & topo.exposes & is_block)
    patched = state.patched | (onehot[:, None] & is_patch)
    responded = state.responded | onehot
    new_state = state._replace(isolated=isolated, blocked=blocked, patched=patched, responded=responded)

    action = xp.where(is_isolate, xp.asarray(DEF_ISOLATE, dtype=xp.int64),
                      xp.where(is_block, xp.asarray(DEF_BLOCK, dtype=xp.int64),
                               xp.asarray(DEF_PATCH, dtype=xp.int64)))
    event = EventRow(
        kind=xp.where(moved, xp.asarray(3, dtype=xp.int64), xp.asarray(0, dtype=xp.int64)),
        t_us=t_us, initiator=target, responder=target, service=xp.asarray(-1, dtype=xp.int64),
        technique=xp.asarray(-1, dtype=xp.int64), stage=xp.asarray(0, dtype=xp.int64),
        sem=xp.where(moved, action, xp.asarray(-1, dtype=xp.int64)), fan=xp.asarray(1, dtype=xp.int64),
        cause=xp.asarray(-1, dtype=xp.int64),
    )
    return new_state, event, moved


def _select_state(xp: Any, cond: Any, a: DynState, b: DynState) -> DynState:
    # Field-wise where(cond, a, b) for a scalar boolean cond.
    return DynState(*[xp.where(cond, xa, xb) for xa, xb in zip(a, b, strict=True)])


def _event_vec(xp: Any, ev: EventRow) -> Any:
    # One event as an int64 row in EVENT_FIELDS order.
    return xp.stack([xp.asarray(x, dtype=xp.int64) for x in ev])


def _none_event(xp: Any, t_us: Any) -> EventRow:
    z = xp.asarray(0, dtype=xp.int64)
    m = xp.asarray(-1, dtype=xp.int64)
    return EventRow(kind=z, t_us=t_us, initiator=m, responder=m, service=m, technique=m, stage=z,
                    sem=m, fan=z, cause=m)


def _select_event(xp: Any, cond: Any, a: EventRow, b: EventRow) -> EventRow:
    return EventRow(*[xp.where(cond, xa, xb) for xa, xb in zip(a, b, strict=True)])


class SlotDraws(NamedTuple):
    """The random draws of one slot, precomputed vectorised over slots (outside the scan body)."""

    slot: Any          # int64
    dt: Any            # float64, the thinning inter-event gap in seconds
    r: Any             # int64, the channel-selection draw in [0, a_max)
    d_tech: Any        # int64
    d_target: Any      # int64
    d_bsrc: Any        # int64
    d_bpair: Any       # int64
    d_def: Any         # int64


def precompute_draws(xp: Any, dyn_key: tuple[Any, Any], slots: Any, a_max_eff: Any) -> SlotDraws:
    """All per-slot draws as arrays over the slot index (one vectorised Threefry pass per lane).

    Keeping the generator out of the loop body is what makes the JAX scan cheap to compile and keeps
    the NumPy loop light; the draws are still addressed by the slot index, so the result is unchanged.
    """
    u_dt = rng.unit(xp, dyn_key, slots, _LANE_DT)
    a_max_rate = a_max_eff.astype(xp.float64) / float(RATE_SCALE)
    dt = -xp.log1p(-u_dt) / a_max_rate
    ch0, ch1 = rng.bits(xp, dyn_key, slots, _LANE_CHANNEL)
    r = rng.below(xp, ch0, ch1, a_max_eff)
    big = xp.asarray(1 << 62, dtype=xp.int64)
    return SlotDraws(
        slot=slots, dt=dt, r=r,
        d_tech=rng.randint(xp, dyn_key, slots, _LANE_TECH, big),
        d_target=rng.randint(xp, dyn_key, slots, _LANE_TARGET, big),
        d_bsrc=rng.randint(xp, dyn_key, slots, _LANE_BSRC, big),
        d_bpair=rng.randint(xp, dyn_key, slots, _LANE_BPAIR, big),
        d_def=rng.randint(xp, dyn_key, slots, _LANE_DEF, big),
    )


def make_step(xp: Any, rt: Runtime) -> Callable[[WorldParams, DynState, SlotDraws], tuple[DynState, Any]]:
    """Build the per-slot transition. `params` is passed per call so the step can be vmapped over
    worlds (a batch shares the scenario runtime and differs only in `params`)."""
    horizon_us = xp.asarray(rt.horizon_us, dtype=xp.int64)

    def step(params: WorldParams, state: DynState, d: SlotDraws) -> tuple[DynState, Any]:
        slot = d.slot
        new_t = state.t_s + d.dt
        t_us = xp.floor(new_t * 1_000_000.0 + 0.5).astype(xp.int64)
        past = t_us > horizon_us

        r_benign = benign.benign_rate(xp, params, rt.scenario.benign, t_us, params.epoch_offset_s, rt.start_epoch_s)
        r_attack, a_state, a_event, _a_moved = attack.attacker_step(
            xp, rt.topo, params, rt.attack_static, state, t_us, slot, d.d_tech, d.d_target)
        r_defend = defender_rate(xp, rt.defender_static, state)
        c_benign = r_benign
        c_attack = r_benign + r_attack
        c_total = c_benign + r_attack + r_defend

        active = ~past
        fired_benign = active & (d.r < c_benign)
        fired_attack = active & (d.r >= c_benign) & (d.r < c_attack)
        fired_defend = active & (d.r >= c_attack) & (d.r < c_total)

        b_event, _b_moved = benign.benign_candidate(xp, rt.topo, params, rt.scenario.benign, state, t_us, d.d_bsrc, d.d_bpair)
        de_state, de_event, _d_moved = defender_candidate(xp, rt.topo, rt.defender_static, state, t_us, d.d_def)

        next_state = _select_state(xp, fired_attack, a_state, state)
        next_state = _select_state(xp, fired_defend, de_state, next_state)
        next_state = next_state._replace(t_s=new_t)

        ev = _none_event(xp, t_us)
        ev = _select_event(xp, fired_benign, b_event, ev)
        ev = _select_event(xp, fired_attack, a_event, ev)
        ev = _select_event(xp, fired_defend, de_event, ev)
        return next_state, _event_vec(xp, ev)

    return step


def _a_max(xp: Any, rt: Runtime, params: WorldParams) -> Any:
    total = params.benign_base_micro + rt.attack_static.a_max + rt.defender_static.rate_micro
    return xp.maximum(total, xp.asarray(1, dtype=xp.int64))


def run_numpy(rt: Runtime, world_index: int) -> Trajectory:
    """Simulate one world with the NumPy backend (a Python loop over the event slots)."""
    xp = np
    wkey = rng.world_key(xp, rt.seed, world_index)
    params = sample_world_params(xp, rt.topo, rt.scenario, wkey)
    dyn_key = rng.stream_key(xp, wkey, STREAM_DYNAMICS)
    slots = np.arange(rt.slots, dtype=np.int64)
    draws = precompute_draws(xp, dyn_key, slots, _a_max(xp, rt, params))
    step = make_step(xp, rt)
    state = initial_state(xp, rt.topo, params)
    n_fields = len(EVENT_FIELDS)
    events = np.empty((rt.slots, n_fields), dtype=np.int64)
    for i in range(rt.slots):
        slot_draw = SlotDraws(*(x[i] for x in draws))
        state, row = step(params, state, slot_draw)
        events[i] = row
    saturated = bool(state.t_s < rt.horizon_us / 1_000_000.0)
    state = state._replace(saturated=np.asarray(saturated))
    return Trajectory(events=events, final_state=state, saturated=np.asarray(saturated))


def _one_world_fn(rt_j: Runtime, topo_j: TopoArrays, slots: Any) -> Callable[[Any], Trajectory]:
    # A function world_index -> Trajectory for one world (the per-world scan over time), to be mapped
    # over the world axis. It is built once and shared by the lax.map and the vmap batch runners.
    import jax

    xp = _jnp()
    step = make_step(xp, rt_j)

    def one(world_index: Any) -> Trajectory:
        wkey = rng.world_key(xp, rt_j.seed, world_index)
        params = sample_world_params(xp, topo_j, rt_j.scenario, wkey)
        dyn_key = rng.stream_key(xp, wkey, STREAM_DYNAMICS)
        draws = precompute_draws(xp, dyn_key, slots, _a_max(xp, rt_j, params))
        state0 = initial_state(xp, topo_j, params)
        final, events = jax.lax.scan(lambda s, d: step(params, s, d), state0, draws)
        saturated = final.t_s < (rt_j.horizon_us / 1_000_000.0)
        return Trajectory(events=events, final_state=final._replace(saturated=saturated), saturated=saturated)

    return one


def _jnp() -> Any:
    import jax.numpy as jnp

    return jnp


def run_jax_batch(rt: Runtime, world_indices: Any) -> Trajectory:
    """Simulate several worlds in one compiled JAX call: a scan over time per world, mapped over the
    world axis with `jax.lax.map`, compiled with `equinox.filter_jit`.

    The result is a `Trajectory` whose leaves have a leading world axis. It runs under jax.enable_x64
    so int64 and float64 match the NumPy backend bit for bit (AS-819). `lax.map` is the default world
    mapping because the XLA CPU backend compiles a `vmap` of a `scan` poorly for more than two worlds;
    `run_jax_vmap` is the explicit `jax.vmap` path for accelerators (AS-820).
    """
    import equinox as eqx
    import jax
    import jax.numpy as jnp

    with jax.enable_x64(True):
        topo_j = topo_arrays(jnp, _as_topology(rt))
        attack_j = build_attack_static(jnp, rt.scenario)
        rt_j = rt._replace(topo=topo_j, attack_static=attack_j)
        slots = jnp.arange(rt.slots, dtype=jnp.int64)
        one = _one_world_fn(rt_j, topo_j, slots)
        run = eqx.filter_jit(lambda idx: jax.lax.map(one, idx))
        return run(jnp.asarray(world_indices, dtype=jnp.int64))


def run_jax_vmap(rt: Runtime, world_indices: Any) -> Trajectory:
    """Simulate several worlds with an explicit `jax.vmap` over the world axis (scan over time inside).

    This is the genuine batched-parallel path for accelerators. On the XLA CPU backend the compile of
    `vmap` over a `scan` is pathological beyond two worlds (AS-820), so `run_jax_batch` (lax.map) is
    the default; this entry is kept for the vmap-equals-sequential invariant and for GPU/TPU hosts.
    """
    import equinox as eqx
    import jax
    import jax.numpy as jnp

    with jax.enable_x64(True):
        topo_j = topo_arrays(jnp, _as_topology(rt))
        attack_j = build_attack_static(jnp, rt.scenario)
        rt_j = rt._replace(topo=topo_j, attack_static=attack_j)
        slots = jnp.arange(rt.slots, dtype=jnp.int64)
        one = _one_world_fn(rt_j, topo_j, slots)
        run = eqx.filter_jit(jax.vmap(one))
        return run(jnp.asarray(world_indices, dtype=jnp.int64))


def _as_topology(rt: Runtime) -> Topology:
    # Rebuild the full Topology (object columns included) from the scenario for the JAX conversion.
    return build_topology(rt.scenario.topology)


__all__ = [
    "DefenderStatic", "Runtime", "SlotDraws", "Trajectory", "build_defender_static", "build_runtime",
    "defender_candidate", "defender_rate", "initial_state", "make_step", "precompute_draws",
    "run_jax_batch", "run_jax_vmap", "run_numpy",
]
