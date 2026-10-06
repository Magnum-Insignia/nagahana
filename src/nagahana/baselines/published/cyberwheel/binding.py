"""CyberWheel behind the defense-environment protocol (env.py), imported lazily from third_party/.

CyberWheel is fetched by tools/third_party/fetch.py into third_party/cyberwheel (never part of the
package). `import_cyberwheel` puts that checkout on the import path and imports it, or raises
MissingDependency with the fetch command when it is absent.

What the binding assumes about CyberWheel (AS-558; to verify against third_party/cyberwheel once it is
fetched; every item is a setting of `CyberWheelSettings`, so a mismatch is corrected in configuration,
not in code):
    env_entry           "module:Class" of a gymnasium-style environment whose reset() returns
                        (observation, info) and step(action) returns (observation, reward, terminated,
                        truncated, info)
    env_kwargs          keyword arguments of its constructor; `network_argument` and `strategy_argument`
                        name the arguments that receive the network and red-strategy configuration files
    network_configs     configuration file of each network size (15, 25, 50, 100 hosts)
    strategy_configs    red-agent configuration file of each scripted strategy
    observation layout  the observation vector holds, for each of `max_slots` host slots, the fields named
                        in `observation_fields` (a subset of HOST_FIELDS, in that order), followed by the
                        detector telemetry; fields the vector does not carry are tracked by the binding
                        from its own actions (`is_decoy` and `present` of decoy slots)
    topology            read from `topology_attribute` of the environment (an adjacency matrix or a
                        networkx graph); the full graph of slots when the attribute is absent and
                        `require_topology` is False
    actions             the environment's discrete action i is the protocol action i (deploy and remove
                        per target, no-op 0) unless `action_map` lists the environment action of every
                        protocol action
A setting that does not match the fetched code raises BindingMismatch naming the setting.
"""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

from nagahana.baselines.published.base import MissingDependency
from nagahana.baselines.published.cyberwheel.env import (
    HOST_FIELDS,
    NETWORK_SIZES,
    STRATEGIES,
    ActionSpace,
    DefenseObservation,
    StepResult,
)
from nagahana.core.errors import InvariantViolation

#: Repository root (src/nagahana/baselines/published/cyberwheel -> 5 levels up).
REPO_ROOT = Path(__file__).resolve().parents[5]


class BindingMismatch(InvariantViolation):
    """The fetched CyberWheel does not match a binding setting (the message names the setting)."""


@dataclass
class CyberWheelSettings:
    """Entry points into CyberWheel (module docstring; defaults to verify against the fetched repository)."""

    root: str = "third_party/cyberwheel"
    package: str = "cyberwheel"
    env_entry: str = "cyberwheel.cyberwheel_envs.cyberwheel_dynamic:DynamicCyberwheel"
    env_kwargs: dict[str, Any] = field(default_factory=dict)
    network_argument: str = "network_config"
    strategy_argument: str = "red_agent"
    network_configs: dict[str, str] = field(default_factory=lambda: {
        "15": "cyberwheel/data/configs/network/15-host-network.yaml",
        "25": "cyberwheel/data/configs/network/25-host-network.yaml",
        "50": "cyberwheel/data/configs/network/50-host-network.yaml",
        "100": "cyberwheel/data/configs/network/100-host-network.yaml",
    })
    strategy_configs: dict[str, str] = field(default_factory=lambda: {s: s for s in STRATEGIES})
    max_slots: int = 0
    decoy_targets: int = 0
    observation_fields: tuple[str, ...] = ("alert_now", "alert_ever")
    telemetry_dim: int = 0
    topology_attribute: str = "network.graph"
    require_topology: bool = False
    action_map: tuple[int, ...] = ()

    def validate(self) -> None:
        bad = [f for f in self.observation_fields if f not in HOST_FIELDS]
        if bad:
            raise ValueError(f"observation_fields {bad} are not host fields {HOST_FIELDS}")
        unknown = [s for s in self.strategy_configs if s not in STRATEGIES]
        if unknown:
            raise ValueError(f"strategy_configs names unknown strategies {unknown}; known: {STRATEGIES}")
        sizes = [int(k) for k in self.network_configs]
        if any(s not in NETWORK_SIZES for s in sizes):
            raise ValueError(f"network_configs keys must be among {NETWORK_SIZES}")
        if self.max_slots < 0 or self.decoy_targets < 0 or self.telemetry_dim < 0:
            raise ValueError("max_slots, decoy_targets and telemetry_dim must be non-negative")


def cyberwheel_root(settings: CyberWheelSettings) -> Path:
    root = Path(settings.root)
    return root if root.is_absolute() else REPO_ROOT / root


def import_cyberwheel(settings: CyberWheelSettings) -> ModuleType:
    """Import CyberWheel from its third_party checkout (or raise MissingDependency with instructions)."""
    root = cyberwheel_root(settings)
    if not (root / settings.package).exists():
        raise MissingDependency(
            f"CyberWheel is not fetched: {root / settings.package} does not exist. Fetch it with "
            "'python tools/third_party/fetch.py --confirm cyberwheel' (it is never installed as a dependency).")
    path = str(root)
    if path not in sys.path:
        sys.path.insert(0, path)
    try:
        return importlib.import_module(settings.package)
    except ImportError as exc:
        raise MissingDependency(f"CyberWheel at {root} failed to import ({exc}); install its own requirements "
                                f"from {root}/requirements.txt or pyproject.toml") from exc


def _resolve(entry: str, setting: str) -> Any:
    module_name, _, attr = entry.partition(":")
    try:
        obj: Any = importlib.import_module(module_name)
    except ImportError as exc:
        raise BindingMismatch(f"setting {setting}: cannot import {module_name!r} ({exc})") from exc
    for part in attr.split("."):
        if not hasattr(obj, part):
            raise BindingMismatch(f"setting {setting}: {module_name} has no attribute {attr!r}")
        obj = getattr(obj, part)
    return obj


def _attribute_path(obj: Any, path: str) -> Any:
    for part in path.split("."):
        if not hasattr(obj, part):
            return None
        obj = getattr(obj, part)
    return obj


def _adjacency(raw: Any, n: int) -> np.ndarray:
    # A dense matrix, or a networkx-style graph with nodes() and edges(); nodes in insertion order.
    if raw is None:
        return np.ones((n, n), dtype=bool) & ~np.eye(n, dtype=bool)
    if hasattr(raw, "nodes") and hasattr(raw, "edges"):
        nodes = list(raw.nodes())
        index = {v: i for i, v in enumerate(nodes[:n])}
        adj = np.zeros((n, n), dtype=bool)
        for u, v in raw.edges():
            if u in index and v in index:
                adj[index[u], index[v]] = adj[index[v], index[u]] = True
        return adj
    arr = np.asarray(raw, dtype=bool)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise BindingMismatch("setting topology_attribute: the attribute is neither a graph nor a square matrix")
    out = np.zeros((n, n), dtype=bool)
    k = min(n, arr.shape[0])
    out[:k, :k] = arr[:k, :k]
    return out


class CyberWheelEnv:
    """A CyberWheel environment of one network size and red strategy, behind the DefenseEnv protocol."""

    def __init__(self, network_size: int, strategy: str, settings: CyberWheelSettings | None = None, *, seed: int = 0) -> None:
        self.settings = settings or CyberWheelSettings()
        self.settings.validate()
        if network_size not in NETWORK_SIZES:
            raise ValueError(f"network size must be one of {NETWORK_SIZES}")
        if strategy not in STRATEGIES:
            raise ValueError(f"strategy must be one of {STRATEGIES}")
        import_cyberwheel(self.settings)
        cls = _resolve(self.settings.env_entry, "env_entry")
        root = cyberwheel_root(self.settings)
        kwargs = dict(self.settings.env_kwargs)
        net_cfg = self.settings.network_configs.get(str(network_size))
        if net_cfg is None:
            raise BindingMismatch(f"setting network_configs: no configuration for {network_size} hosts")
        kwargs[self.settings.network_argument] = str(root / net_cfg) if (root / net_cfg).exists() else net_cfg
        kwargs[self.settings.strategy_argument] = self.settings.strategy_configs.get(strategy, strategy)
        try:
            self.env = cls(**kwargs)
        except TypeError as exc:
            raise BindingMismatch(f"settings env_kwargs / network_argument / strategy_argument: {exc}") from exc
        n_actions = int(getattr(getattr(self.env, "action_space", None), "n", 0))
        if n_actions < 1:
            raise BindingMismatch("setting env_entry: the environment has no discrete action_space.n")
        targets = self.settings.decoy_targets or (n_actions - 1) // 2
        self._space = ActionSpace(targets)
        if self.settings.action_map and len(self.settings.action_map) != self._space.n:
            raise BindingMismatch(f"setting action_map: needs {self._space.n} entries")
        if not self.settings.action_map and n_actions != self._space.n:
            raise BindingMismatch(f"setting action_map: the environment has {n_actions} actions, the protocol {self._space.n}")
        obs_dim = int(np.prod(getattr(getattr(self.env, "observation_space", None), "shape", (0,)) or (0,)))
        per_host = len(self.settings.observation_fields)
        self._slots = self.settings.max_slots or (obs_dim - self.settings.telemetry_dim) // max(per_host, 1)
        if self._slots < 1:
            raise BindingMismatch("settings max_slots / observation_fields / telemetry_dim do not fit the observation size")
        self.network_size, self.strategy, self.seed = network_size, strategy, seed
        self._decoy = np.zeros(self._slots, dtype=bool)
        self._present = np.zeros(self._slots, dtype=bool)
        self._present[: min(network_size, self._slots)] = True

    @property
    def action_space(self) -> ActionSpace:
        return self._space

    @property
    def n_slots(self) -> int:
        return self._slots

    @property
    def telemetry_dim(self) -> int:
        return self.settings.telemetry_dim

    def _observation(self, raw: Any) -> DefenseObservation:
        vec = np.asarray(raw, dtype=np.float32).ravel()
        per_host = len(self.settings.observation_fields)
        need = self._slots * per_host + self.settings.telemetry_dim
        if vec.size < need:
            raise BindingMismatch(f"settings observation_fields / max_slots / telemetry_dim: observation has {vec.size} "
                                  f"values, the layout needs {need}")
        hosts = np.zeros((self._slots, len(HOST_FIELDS)), dtype=np.float32)
        block = vec[: self._slots * per_host].reshape(self._slots, per_host)
        for j, name in enumerate(self.settings.observation_fields):
            hosts[:, HOST_FIELDS.index(name)] = block[:, j]
        if "present" not in self.settings.observation_fields:
            hosts[:, HOST_FIELDS.index("present")] = (self._present | self._decoy).astype(np.float32)
        if "is_decoy" not in self.settings.observation_fields:
            hosts[:, HOST_FIELDS.index("is_decoy")] = self._decoy.astype(np.float32)
        topo = _attribute_path(self.env, self.settings.topology_attribute)
        if topo is None and self.settings.require_topology:
            raise BindingMismatch(f"setting topology_attribute: environment has no {self.settings.topology_attribute!r}")
        telemetry = vec[self._slots * per_host: need]
        return DefenseObservation(hosts, _adjacency(topo, self._slots), telemetry)

    def reset(self, *, seed: int | None = None) -> DefenseObservation:
        out = self.env.reset(seed=self.seed if seed is None else seed)
        raw = out[0] if isinstance(out, tuple) else out
        self._decoy[:] = False
        return self._observation(raw)

    def step(self, action: int) -> StepResult:
        kind, target = self._space.decode(action)
        env_action = self.settings.action_map[action] if self.settings.action_map else int(action)
        out = self.env.step(env_action)
        if not (isinstance(out, tuple) and len(out) == 5):
            raise BindingMismatch("setting env_entry: step() must return (observation, reward, terminated, truncated, info)")
        raw, reward, terminated, truncated, info = out
        if kind == "deploy" and 0 <= target < self._slots:
            self._decoy[target] = True
        elif kind == "remove" and 0 <= target < self._slots:
            self._decoy[target] = False
        return StepResult(self._observation(raw), float(reward), bool(terminated), bool(truncated), dict(info or {}))


def cyberwheel_factory(network_size: int, strategy: str, settings: CyberWheelSettings | None = None) -> Any:
    """Seed -> CyberWheelEnv factory, for VectorEnv and the agents."""
    def make(seed: int) -> CyberWheelEnv:
        return CyberWheelEnv(network_size, strategy, settings, seed=seed)
    return make
