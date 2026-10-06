"""Top-level orchestration: simulate whole worlds end to end (P-14).

`simulate_world` runs one world (hidden dynamics, then the observation fabric, then the data-model
assembly) and returns a `WorldOutput`. `simulate_worlds` runs a batch: with the NumPy backend it loops
(the reference path), with the JAX backend it runs them together through `jax.lax.scan` and
`equinox.filter_vmap`. Both backends produce the same bit-for-bit trajectories for a seed, so the
emitted records, labels, episodes and ground-truth tables are identical regardless of backend; the
observation and assembly stages run in NumPy on those trajectories.
"""

from __future__ import annotations

import numpy as np

from nagahana.worldsim import rng
from nagahana.worldsim.config import ScenarioConfig
from nagahana.worldsim.dynamics import Trajectory, build_runtime, run_jax_batch, run_numpy
from nagahana.worldsim.emit import ADAPTER_VERSION, WorldOutput, build_world_output
from nagahana.worldsim.observe import observe_world
from nagahana.worldsim.scenarios import get_scenario
from nagahana.worldsim.state import DynState
from nagahana.worldsim.topology import Topology, build_topology

Backend = str


def _resolve(scenario: ScenarioConfig | str) -> ScenarioConfig:
    return get_scenario(scenario) if isinstance(scenario, str) else scenario


def _assemble(scenario: ScenarioConfig, topo: Topology, traj: Trajectory, seed: int, world_index: int) -> WorldOutput:
    # Observe the trajectory and build the data-model artifacts (NumPy host side).
    events = np.asarray(traj.events)
    final = DynState(*[np.asarray(x) for x in traj.final_state])
    saturated = bool(np.asarray(traj.saturated))
    wkey = rng.world_key(np, seed, world_index)
    result = observe_world(scenario, topo, events, wkey, ADAPTER_VERSION)
    return build_world_output(scenario, topo, result, events, saturated, seed, world_index, final)


def simulate_world(scenario: ScenarioConfig | str, seed: int, world_index: int = 0,
                   backend: Backend = "numpy") -> WorldOutput:
    """Simulate one world and assemble its data-model artifacts and ground truth."""
    sc = _resolve(scenario)
    topo = build_topology(sc.topology)
    if backend == "numpy":
        rt = build_runtime(np, sc, topo, seed)
        traj = run_numpy(rt, world_index)
    elif backend == "jax":
        rt = build_runtime(np, sc, topo, seed)
        batch = run_jax_batch(rt, [world_index])
        traj = _unbatch(batch, 0)
    else:
        raise ValueError(f"backend must be 'numpy' or 'jax', got {backend!r}")
    return _assemble(sc, topo, traj, seed, world_index)


def simulate_worlds(scenario: ScenarioConfig | str, seed: int, n_worlds: int,
                    backend: Backend = "numpy") -> list[WorldOutput]:
    """Simulate `n_worlds` worlds (world indices 0 ... n_worlds-1)."""
    if n_worlds < 1:
        raise ValueError("n_worlds must be >= 1")
    sc = _resolve(scenario)
    topo = build_topology(sc.topology)
    rt = build_runtime(np, sc, topo, seed)
    outputs: list[WorldOutput] = []
    if backend == "numpy":
        for w in range(n_worlds):
            outputs.append(_assemble(sc, topo, run_numpy(rt, w), seed, w))
    elif backend == "jax":
        batch = run_jax_batch(rt, list(range(n_worlds)))
        for w in range(n_worlds):
            outputs.append(_assemble(sc, topo, _unbatch(batch, w), seed, w))
    else:
        raise ValueError(f"backend must be 'numpy' or 'jax', got {backend!r}")
    return outputs


def _unbatch(batch: Trajectory, w: int) -> Trajectory:
    # Slice the leading world axis out of a batched trajectory.
    events = np.asarray(batch.events)[w]
    final = DynState(*[np.asarray(x)[w] for x in batch.final_state])
    saturated = np.asarray(batch.saturated)[w]
    return Trajectory(events=events, final_state=final, saturated=saturated)


def trajectory(scenario: ScenarioConfig | str, seed: int, world_index: int = 0,
               backend: Backend = "numpy") -> Trajectory:
    """The hidden-state trajectory of one world (without the observation and assembly stages)."""
    sc = _resolve(scenario)
    topo = build_topology(sc.topology)
    rt = build_runtime(np, sc, topo, seed)
    if backend == "numpy":
        return run_numpy(rt, world_index)
    if backend == "jax":
        return _unbatch(run_jax_batch(rt, [world_index]), 0)
    raise ValueError(f"backend must be 'numpy' or 'jax', got {backend!r}")


__all__ = ["simulate_world", "simulate_worlds", "trajectory"]
