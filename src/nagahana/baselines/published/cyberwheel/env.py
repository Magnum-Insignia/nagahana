"""The defense-environment protocol shared by every CyberWheel agent (AS-558).

Observation (per step)
    hosts       [V, 5] float: for each host slot (real hosts and decoy slots) the fields
                present, is_decoy, isolated, alert_now, alert_ever
                (alert_now: the detectors raised an alert on the host in this step; alert_ever: at any
                step of the episode so far)
    adjacency   [V, V] bool: which slots can communicate (the network topology, decoys included)
    telemetry   [D] float: detector telemetry (for example alert counts per detector)
    The vector form of an observation is [hosts.ravel() ; telemetry] (the input of the vector variants
    and of PPO); the graph form keeps hosts as node features over the adjacency (GNN-PPO, the graph
    variant of CyberWorld).

Actions: a discrete code a in 0 ... 2 T over T decoy targets:
    a = 0 no-op,  a = 1 + t deploy a decoy on target t,  a = 1 + T + t remove the decoy of target t.
A binding maps targets to what the environment deploys decoys on (host slots or subnets).

Red agents follow one of four scripted strategies of CyberWheel: ServerDowntime and BFSServerDowntime
(reach and disrupt servers), Exfiltration and BFSExfiltration (reach servers and exfiltrate data); the
BFS variants explore the network breadth-first. Networks of 15, 25, 50 and 100 hosts are evaluated.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from nagahana.core.errors import InvariantViolation

HOST_FIELDS: tuple[str, ...] = ("present", "is_decoy", "isolated", "alert_now", "alert_ever")
STRATEGIES: tuple[str, ...] = ("BFSServerDowntime", "ServerDowntime", "Exfiltration", "BFSExfiltration")
NETWORK_SIZES: tuple[int, ...] = (15, 25, 50, 100)


@dataclass(frozen=True)
class DefenseObservation:
    """One observation (module docstring)."""

    hosts: np.ndarray
    adjacency: np.ndarray
    telemetry: np.ndarray

    def __post_init__(self) -> None:
        hosts = np.asarray(self.hosts, dtype=np.float32)
        adj = np.asarray(self.adjacency, dtype=bool)
        tel = np.asarray(self.telemetry, dtype=np.float32)
        if hosts.ndim != 2 or hosts.shape[1] != len(HOST_FIELDS):
            raise InvariantViolation(f"hosts must be [V, {len(HOST_FIELDS)}] ({', '.join(HOST_FIELDS)})")
        if adj.shape != (hosts.shape[0], hosts.shape[0]):
            raise InvariantViolation("adjacency must be [V, V]")
        if tel.ndim != 1:
            raise InvariantViolation("telemetry must be a vector")
        object.__setattr__(self, "hosts", hosts)
        object.__setattr__(self, "adjacency", adj)
        object.__setattr__(self, "telemetry", tel)

    @property
    def n_slots(self) -> int:
        return int(self.hosts.shape[0])

    def vector(self) -> np.ndarray:
        """[V * 5 + D] float32: the flat observation."""
        return np.concatenate([self.hosts.ravel(), self.telemetry]).astype(np.float32)


@dataclass(frozen=True)
class ActionSpace:
    """Decoy actions over `targets` targets (module docstring)."""

    targets: int

    @property
    def n(self) -> int:
        return 1 + 2 * self.targets

    def decode(self, action: int) -> tuple[str, int]:
        """("noop", -1), ("deploy", t) or ("remove", t)."""
        a = int(action)
        if not 0 <= a < self.n:
            raise InvariantViolation(f"action {a} outside 0 ... {self.n - 1}")
        if a == 0:
            return "noop", -1
        if a <= self.targets:
            return "deploy", a - 1
        return "remove", a - 1 - self.targets

    def encode(self, kind: str, target: int = -1) -> int:
        if kind == "noop":
            return 0
        if not 0 <= target < self.targets:
            raise InvariantViolation(f"target {target} outside 0 ... {self.targets - 1}")
        if kind == "deploy":
            return 1 + target
        if kind == "remove":
            return 1 + self.targets + target
        raise InvariantViolation(f"unknown action kind {kind!r}")


@dataclass
class StepResult:
    """Outcome of one environment step (gymnasium convention)."""

    observation: DefenseObservation
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any] = field(default_factory=dict)


class DefenseEnv(Protocol):
    """What a defense agent needs from an environment."""

    @property
    def action_space(self) -> ActionSpace: ...

    @property
    def n_slots(self) -> int: ...

    @property
    def telemetry_dim(self) -> int: ...

    def reset(self, *, seed: int | None = None) -> DefenseObservation: ...

    def step(self, action: int) -> StepResult: ...


EnvFactory = Callable[[int], DefenseEnv]


@dataclass
class EpisodeStats:
    """Return and length of finished episodes."""

    returns: list[float] = field(default_factory=list)
    lengths: list[int] = field(default_factory=list)

    def summary(self) -> dict[str, float]:
        r = np.asarray(self.returns, dtype=np.float64)
        n = np.asarray(self.lengths, dtype=np.float64)
        if r.size == 0:
            return {"episodes": 0.0, "return_mean": float("nan"), "return_std": float("nan"), "length_mean": float("nan")}
        return {"episodes": float(r.size), "return_mean": float(r.mean()), "return_std": float(r.std(ddof=1)) if r.size > 1 else 0.0,
                "length_mean": float(n.mean())}


class VectorEnv:
    """N environments stepped in lock-step with automatic reset (the episode's last observation is returned
    in `info["final_observation"]` and the next episode's first observation in its place)."""

    def __init__(self, factory: EnvFactory, n: int, seed: int) -> None:
        if n < 1:
            raise ValueError("a vector environment needs at least one environment")
        self.envs = [factory(seed + i) for i in range(n)]
        self.seed = seed
        self._episode = [0] * n
        self._ret = np.zeros(n)
        self._len = np.zeros(n, dtype=np.int64)
        self.stats = EpisodeStats()
        spaces = {e.action_space.n for e in self.envs}
        if len(spaces) != 1:
            raise InvariantViolation("all environments of a vector environment need the same action space")

    @property
    def n(self) -> int:
        return len(self.envs)

    @property
    def action_space(self) -> ActionSpace:
        return self.envs[0].action_space

    def reset(self) -> list[DefenseObservation]:
        out = []
        for i, e in enumerate(self.envs):
            out.append(e.reset(seed=self.seed + 7919 * (i + 1)))
            self._episode[i] = 0
        self._ret[:] = 0.0
        self._len[:] = 0
        return out

    def step(self, actions: Sequence[int]) -> tuple[list[DefenseObservation], np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
        obs: list[DefenseObservation] = []
        rewards = np.zeros(self.n, dtype=np.float64)
        terminated = np.zeros(self.n, dtype=bool)
        truncated = np.zeros(self.n, dtype=bool)
        infos: list[dict[str, Any]] = []
        for i, (e, a) in enumerate(zip(self.envs, actions, strict=True)):
            res = e.step(int(a))
            rewards[i], terminated[i], truncated[i] = res.reward, res.terminated, res.truncated
            self._ret[i] += res.reward
            self._len[i] += 1
            info = dict(res.info)
            if res.terminated or res.truncated:
                self.stats.returns.append(float(self._ret[i]))
                self.stats.lengths.append(int(self._len[i]))
                self._ret[i], self._len[i] = 0.0, 0
                self._episode[i] += 1
                info["final_observation"] = res.observation
                obs.append(e.reset(seed=self.seed + 7919 * (i + 1) + 104729 * self._episode[i]))
            else:
                obs.append(res.observation)
            infos.append(info)
        return obs, rewards, terminated, truncated, infos


def batch_vectors(observations: Sequence[DefenseObservation]) -> np.ndarray:
    """[N, V * 5 + D] float32."""
    return np.stack([o.vector() for o in observations])


def batch_graphs(observations: Sequence[DefenseObservation]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(hosts [N, V, 5], adjacency [N, V, V], telemetry [N, D]) of observations of equal size."""
    return (np.stack([o.hosts for o in observations]), np.stack([o.adjacency for o in observations]),
            np.stack([o.telemetry for o in observations]))


def evaluate_policy(factory: EnvFactory, act: Callable[[DefenseObservation], int], *, episodes: int, seed: int,
                    max_steps: int = 10_000) -> EpisodeStats:
    """Run `episodes` episodes with the given policy and collect their returns and lengths."""
    stats = EpisodeStats()
    for ep in range(episodes):
        env = factory(seed + ep)
        obs = env.reset(seed=seed + ep)
        total, steps = 0.0, 0
        for _ in range(max_steps):
            res = env.step(act(obs))
            total += res.reward
            steps += 1
            obs = res.observation
            if res.terminated or res.truncated:
                break
        stats.returns.append(total)
        stats.lengths.append(steps)
    return stats
