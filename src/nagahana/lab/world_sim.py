"""Ground-truth world simulator entry point (decision P-14).

The lab exposes the world simulator through this module; the implementation lives in the
`nagahana.worldsim` package. A ground-truth world is a simulated IT or OT network whose hidden state
(attacker position, stage, intent, the defender's actions) is known and whose observation is modelled
explicitly (packet taps, NetFlow sampling, Zeek logs, IDS alerts, encryption, partial coverage, clock
skew). Public datasets have label errors (Engelen, Rimmer and Joosen, IEEE SPW 2021) and no
hidden-state truth, so belief accuracy, trust estimation and forecast skill, and the Bayes-optimal
information ceiling of the audit (P-15), can be measured only against such worlds.

The lab stays isolated from the production core: it imports JAX lazily (through `worldsim`), and
nothing in the core depends on it (lab/README.md).
"""

from __future__ import annotations

from typing import Any

from nagahana.worldsim import (
    SCENARIOS,
    WorldOutput,
    build_topology,
    get_scenario,
    list_scenarios,
    simulate_worlds,
    trajectory,
)
from nagahana.worldsim.simulate import simulate_world


def simulate(scenario: Any, seed: int, world_index: int = 0, backend: str = "numpy") -> WorldOutput:
    """Simulate one world with known hidden state and return its data-model and ground-truth output.

    Parameters
    ----------
    scenario:
        A library scenario name (`list_scenarios`) or a `worldsim.config.ScenarioConfig`.
    seed:
        Run seed; the world is a bit-identical function of (seed, world_index) on either backend.
    world_index:
        Index of the world within the run.
    backend:
        "numpy" (reference) or "jax" (vmap over worlds, scan over time, jit).
    """
    return simulate_world(scenario, seed=seed, world_index=world_index, backend=backend)


__all__ = [
    "SCENARIOS", "WorldOutput", "build_topology", "get_scenario", "list_scenarios", "simulate",
    "simulate_world", "simulate_worlds", "trajectory",
]
