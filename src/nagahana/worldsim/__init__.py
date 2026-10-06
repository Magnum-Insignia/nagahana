"""Ground-truth world simulator (decision P-14).

Simulated IT and OT networks with known hidden state and explicit observation models, emitting records
in the NagaHana data model so that the pipeline, the information audit (P-15) and the evaluation can be
run against worlds whose ground truth is known. JAX + Equinox is the primary backend (vmap over worlds,
scan over time, jit), with a NumPy fallback; a counter-based generator makes a world reproducible bit
for bit across backends for a seed.

Entry points:
    simulate_world / simulate_worlds   run one or many worlds end to end -> WorldOutput
    get_scenario / list_scenarios      the named scenario library
    build_topology                      the static structure of a scenario
    register_cli                        wire `nagahana worldsim ...` into the top-level CLI
"""

from __future__ import annotations

from nagahana.worldsim.cli import register_cli
from nagahana.worldsim.config import ScenarioConfig
from nagahana.worldsim.emit import WorldOutput
from nagahana.worldsim.scenarios import SCENARIOS, get_scenario, list_scenarios, load_scenario, write_library
from nagahana.worldsim.simulate import simulate_world, simulate_worlds, trajectory
from nagahana.worldsim.topology import build_topology

__all__ = [
    "SCENARIOS", "ScenarioConfig", "WorldOutput", "build_topology", "get_scenario", "list_scenarios",
    "load_scenario", "register_cli", "simulate_world", "simulate_worlds", "trajectory", "write_library",
]
