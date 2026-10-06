"""Uniform interface of the defense agents trained in CyberWheel (PPO, GNN-PPO, CyberWorld).

A defense agent has a config dataclass, `train(factory)` on an environment factory (seed -> DefenseEnv),
`act(observation)`, `evaluate(factory, episodes)` and save / load through the same verified manifest as
the published-baseline reproductions (base.py). Training runs inside `seeded(config.seed)`.
"""

from __future__ import annotations

import abc
import importlib
import json
import typing
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, TypeVar

import numpy as np

from nagahana import __version__ as _PACKAGE_VERSION
from nagahana.baselines.published.base import (
    MANIFEST_FORMAT,
    MANIFEST_NAME,
    MANIFEST_VERSION,
    BaselineConfig,
    BaselineStateError,
    Reference,
    config_from_mapping,
    config_to_mapping,
    jsonable,
    read_manifest,
    seeded,
    sha256_file,
)
from nagahana.baselines.published.cyberwheel.env import DefenseObservation, EnvFactory, EpisodeStats
from nagahana.core.errors import InvariantViolation

A = TypeVar("A", bound="DefenseAgent")


@dataclass(frozen=True)
class AgentSpec:
    """Static description of a defense agent."""

    name: str
    title: str
    reference: Reference
    environment: str
    budget_steps: int
    third_party: str
    assumptions: tuple[str, ...] = ()


class DefenseAgent(abc.ABC):
    """Base class (module docstring). Subclasses implement _train, _act, _export_state and _import_state."""

    spec: ClassVar[AgentSpec]
    config_type: ClassVar[type[BaselineConfig]]

    def __init__(self, config: BaselineConfig | None = None) -> None:
        cfg = config if config is not None else self.config_type()
        if not isinstance(cfg, self.config_type):
            raise TypeError(f"{type(self).__name__} needs a {self.config_type.__name__}")
        cfg.validate()
        self.config = cfg
        self._trained = False
        self.train_report: dict[str, Any] = {}

    @property
    def is_trained(self) -> bool:
        return self._trained

    def train(self: A, factory: EnvFactory) -> A:
        """Train on environments built by `factory` (seed -> environment)."""
        self.train_report = {}
        with seeded(self.config.seed, deterministic=self.config.deterministic) as rng:
            self._train(factory, rng)
        self._trained = True
        return self

    def act(self, observation: DefenseObservation, *, greedy: bool = True) -> int:
        """Action for one observation (the mode of the policy when greedy)."""
        if not self._trained:
            raise BaselineStateError(f"{self.spec.name}: train (or load) before acting")
        return self._act(observation, greedy)

    def reset_state(self) -> None:
        """Forget any per-episode state (recurrent agents override it); called before every evaluation episode."""

    def evaluate(self, factory: EnvFactory, *, episodes: int, seed: int, greedy: bool = True, max_steps: int = 100_000) -> EpisodeStats:
        """Returns and lengths of `episodes` evaluation episodes (environment seeds seed, seed + 1, ...)."""
        stats = EpisodeStats()
        for ep in range(episodes):
            env = factory(seed + ep)
            obs = env.reset(seed=seed + ep)
            self.reset_state()
            total, steps = 0.0, 0
            while steps < max_steps:
                res = env.step(self.act(obs, greedy=greedy))
                total += res.reward
                steps += 1
                obs = res.observation
                if res.terminated or res.truncated:
                    break
            stats.returns.append(total)
            stats.lengths.append(steps)
        return stats

    def save(self, path: str | Path, *, overwrite: bool = False) -> Path:
        if not self._trained:
            raise BaselineStateError(f"{self.spec.name}: nothing to save before training")
        directory = Path(path)
        manifest_path = directory / MANIFEST_NAME
        if manifest_path.exists() and not overwrite:
            raise FileExistsError(f"{manifest_path} exists; pass overwrite=True to replace it")
        directory.mkdir(parents=True, exist_ok=True)
        state = self._export_state(directory)
        files = sorted(p for p in directory.rglob("*") if p.is_file() and p.name != MANIFEST_NAME)
        manifest = {"format": MANIFEST_FORMAT, "format_version": MANIFEST_VERSION, "baseline": self.spec.name,
                    "class": f"{type(self).__module__}:{type(self).__qualname__}", "package_version": _PACKAGE_VERSION,
                    "reference": self.spec.reference.key, "config": config_to_mapping(self.config), "state": jsonable(state),
                    "fit_report": jsonable(self.train_report),
                    "files": {p.relative_to(directory).as_posix(): sha256_file(p) for p in files}}
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        return directory

    @classmethod
    def load(cls: type[A], path: str | Path) -> A:
        directory = Path(path)
        manifest = read_manifest(directory)
        target: type[Any] = cls
        if cls is DefenseAgent:
            module_name, _, qualname = str(manifest["class"]).partition(":")
            target = getattr(importlib.import_module(module_name), qualname)
            if not (isinstance(target, type) and issubclass(target, DefenseAgent)):
                raise InvariantViolation(f"{manifest['class']} is not a DefenseAgent")
        elif manifest["baseline"] != cls.spec.name:
            raise InvariantViolation(f"{directory} holds {manifest['baseline']!r}, not {cls.spec.name!r}")
        obj = target(config_from_mapping(target.config_type, manifest["config"]))
        obj._import_state(directory, manifest["state"])
        obj.train_report = dict(manifest.get("fit_report", {}))
        obj._trained = True
        return typing.cast(A, obj)

    @abc.abstractmethod
    def _train(self, factory: EnvFactory, rng: np.random.Generator) -> None: ...

    @abc.abstractmethod
    def _act(self, observation: DefenseObservation, greedy: bool) -> int: ...

    @abc.abstractmethod
    def _export_state(self, directory: Path) -> dict[str, Any]: ...

    @abc.abstractmethod
    def _import_state(self, directory: Path, state: Mapping[str, Any]) -> None: ...
