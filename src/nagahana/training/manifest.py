"""Run manifests: what a training run was, so every result traces back to code, configuration and data.

A manifest (`run.json` in the run directory, updated after every stage) records:

- identity: run name, preset, seed, the configuration hash (training/config.py `config_hash`) and the
  full configuration;
- code: `git describe --always --dirty --tags` and the commit when the source tree is a git checkout
  (read-only commands), else the package version and a SHA-256 over the package's source files;
- data: per source, the file's SHA-256, its size, and the digest of the prepared content
  (training/data.py `source_digest`); the split plan's digest; the augmentation plan's digest;
- governance: the status of every decision and proposal, the held-decision options in force, the
  enabled proposals, the assumptions used by this process;
- ablation: the variant and its switches (training/ablation.py);
- environment: Python, PyTorch, NumPy, pandas versions, the platform, the devices, the world size;
- stages: per stage its status, checkpoint lineage (directory, step, digests), metrics and notes.

Writes are atomic (training/serialization.py).
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from nagahana.core.config import to_mapping
from nagahana.training.config import TrainingRun, config_hash
from nagahana.training.serialization import atomic_write_text, sha256_file

MANIFEST = "run.json"


def code_version() -> dict[str, str]:
    """The source tree's version: git when available (read-only), else a digest of the package sources."""
    root = Path(__file__).resolve().parents[3]
    out: dict[str, str] = {}
    try:
        desc = subprocess.run(["git", "describe", "--always", "--dirty", "--tags"], cwd=root, capture_output=True,
                              text=True, timeout=10, check=False)
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, timeout=10, check=False)
        if desc.returncode == 0 and head.returncode == 0:
            out["git_describe"] = desc.stdout.strip()
            out["git_commit"] = head.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    pkg = Path(__file__).resolve().parents[1]
    h = hashlib.sha256()
    for f in sorted(pkg.rglob("*.py")):
        h.update(f.relative_to(pkg).as_posix().encode("utf-8"))
        h.update(f.read_bytes())
    out["source_sha256"] = h.hexdigest()
    try:
        from importlib.metadata import version

        out["package_version"] = version("nagahana")
    except Exception as exc:  # noqa: BLE001 - metadata is absent when the package runs from a source tree
        out["package_version"] = f"unavailable ({type(exc).__name__})"
    return out


def environment(info: Any = None) -> dict[str, Any]:
    """Library versions, platform and devices."""
    import numpy
    import pandas
    import torch

    env: dict[str, Any] = {"python": sys.version.split()[0], "torch": torch.__version__, "numpy": numpy.__version__,
                           "pandas": pandas.__version__, "platform": platform.platform(),
                           "cuda_available": torch.cuda.is_available()}
    if torch.cuda.is_available():
        env["cuda"] = torch.version.cuda
        env["devices"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    if info is not None:
        env |= {"world_size": info.world, "strategy": info.strategy, "backend": info.backend}
    return env


def governance() -> dict[str, Any]:
    """Decision statuses, held options in force, assumptions used (this process)."""
    from nagahana.governance import assumptions as central
    from nagahana.governance import decisions
    from nagahana.training.assumptions import local_uses

    held = {d.id: decisions.option_in_force(d.id) for d in decisions.by_status(decisions.Status.HELD)}
    return {"status": {d.id: d.status.value for d in decisions.all_entries()}, "held_options": held,
            "configured": decisions.configured_options(),
            "assumptions_used": sorted(central.uses()), "training_assumptions_used": sorted(local_uses())}


def file_digest(path: str | Path) -> dict[str, Any]:
    """SHA-256 and size of a file."""
    p = Path(path)
    return {"path": str(p), "sha256": sha256_file(p), "bytes": p.stat().st_size}


class RunManifest:
    """The run manifest of one run directory (module docstring). Only rank 0 writes it."""

    def __init__(self, run_dir: str | Path, cfg: TrainingRun, *, info: Any = None) -> None:
        self.path = Path(run_dir) / MANIFEST
        self.cfg = cfg
        if self.path.is_file():
            self.data: dict[str, Any] = json.loads(self.path.read_text(encoding="utf-8"))
            if self.data.get("config_hash") != config_hash(cfg):
                self.data.setdefault("config_history", []).append(
                    {"config_hash": self.data.get("config_hash"), "replaced": time.time()})
        else:
            self.data = {"created": time.time(), "stages": {}}
        self.data |= {"name": cfg.name, "preset": cfg.preset, "seed": cfg.seed, "config_hash": config_hash(cfg),
                      "config": to_mapping(cfg), "code": code_version(), "environment": environment(info),
                      "ablation": to_mapping(cfg.ablation), "enabled_proposals": list(cfg.enabled_proposals)}

    def set(self, key: str, value: Any) -> None:
        self.data[key] = value

    def stage(self, key: str, record: Mapping[str, Any]) -> None:
        """Record a stage's status, lineage and metrics."""
        stages = self.data.setdefault("stages", {})
        stages[key] = dict(stages.get(key, {})) | dict(record) | {"updated": time.time()}

    def write(self) -> Path:
        self.data["governance"] = governance()
        self.data["updated"] = time.time()
        atomic_write_text(self.path, json.dumps(self.data, indent=1, sort_keys=True, default=str))
        return self.path


__all__ = ["MANIFEST", "RunManifest", "code_version", "environment", "file_digest", "governance"]
