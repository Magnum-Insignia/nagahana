"""Shared training helpers: precision contexts, parameter selection, model checkpoints, logging scalars.

- `autocast(precision, device)`: bf16 autocast for "bf16" (AS-39), fp16 autocast for "fp16" (AS-587),
  nothing for "fp32" (CPU tests); weights stay float32 master weights (D-54).
- `trainable(modules)`: the parameters that require gradients, deduplicated, in module order.
- `warmup_factor(step, warmup_steps)`: the warm-up part of every schedule of training/optim.py.
- `save_checkpoint` / `load_checkpoint`: a model checkpoint (weights, the model hash of P-18, the latent
  space of P-19, the stage and step, optionally the optimiser state) in the safe format of
  training/serialization.py: tensors only in a weights-only file, JSON for the rest, SHA-256 verified
  before anything is deserialised (AS-576). Loading refuses another latent space and a file whose
  weights do not match their recorded model hash. Stage runs use the resumable checkpoints of
  training/checkpoint.py; these are the files exchanged between commands (`--checkpoint`).
- `scalars`, `config_params`: values for the run logger.

Invariants (tests): a checkpoint round-trips bit for bit and its hash matches; an altered file is refused
before it is opened; frozen parameters are never handed to an optimiser (training/optim.py).
"""

from __future__ import annotations

import contextlib
import math
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn

from nagahana.core.errors import ConfigMissing, InvariantViolation
from nagahana.governance.assumptions import assume
from nagahana.models.config import NagaHanaConfig
from nagahana.models.nagahana import NagaHana, latent_space_hash, latent_space_id, model_hash
from nagahana.training.config import PRECISIONS
from nagahana.training.serialization import load_state, save_state

MODEL_CHECKPOINT = "nagahana-model"


def trainable(modules: Iterable[nn.Module]) -> list[nn.Parameter]:
    """Parameters of `modules` that require gradients (deduplicated, in module order)."""
    seen: set[int] = set()
    out: list[nn.Parameter] = []
    for m in modules:
        for p in m.parameters():
            if p.requires_grad and id(p) not in seen:
                seen.add(id(p))
                out.append(p)
    return out


def warmup_factor(step: int, warmup_steps: int) -> float:
    """lr multiplier at `step` (0-based) during warm-up: (step + 1) / W, then 1 (AS-406, AS-570)."""
    if warmup_steps <= 0:
        return 1.0
    return min(1.0, (step + 1) / warmup_steps)


def check_precision(precision: str) -> str:
    """Validate a precision name ("bf16", "fp32" or "fp16")."""
    if precision not in PRECISIONS:
        raise ConfigMissing(f"precision must be one of {PRECISIONS}, got {precision!r}")
    return precision


@contextlib.contextmanager
def autocast(precision: str, device: torch.device | str = "cpu") -> Iterator[None]:
    """bf16 or fp16 autocast (AS-39, AS-587), nothing for "fp32"."""
    assume("AS-39", by=__name__)
    p = check_precision(precision)
    if p == "fp32":
        yield
        return
    dev = torch.device(device).type
    with torch.autocast(device_type=dev, dtype=torch.bfloat16 if p == "bf16" else torch.float16):
        yield


def save_checkpoint(path: str | Path, model: NagaHana, *, stage: str, step: int, optimiser: Any = None,
                    extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Save weights + identity (model hash P-18, latent space P-19) [+ optimiser]. Returns the header."""
    header = {
        "config": model.cfg.name, "stage": stage, "step": int(step), "model_hash": model_hash(model),
        "latent_space_id": latent_space_id(model.cfg), "latent_space_hash": latent_space_hash(model.cfg),
    }
    blob: dict[str, Any] = {"state_dict": {k: v.detach().to("cpu") for k, v in model.state_dict().items()},
                            "extra": dict(extra or {})}
    if optimiser is not None:
        from nagahana.training.checkpoint import optimizer_state_by_name

        blob["optimizers"] = {name: optimizer_state_by_name(opt) for name, opt in optimiser.optimizers()}
        blob["counters"] = optimiser.counters()
    save_state(path, blob, kind=MODEL_CHECKPOINT, meta=header)
    return header


def load_checkpoint(path: str | Path, model: NagaHana, *, optimiser: Any = None) -> dict[str, Any]:
    """Load weights into `model` (same latent space required); verify the stored model hash. Returns the header.

    The file's SHA-256 is checked against its sidecar before it is opened, and only tensors are
    deserialised (weights-only); a latent space other than the model's is refused (P-19), and so are
    weights whose hash differs from the recorded one (P-18).
    """
    blob, header = load_state(path, kind=MODEL_CHECKPOINT)
    if header["latent_space_hash"] != latent_space_hash(model.cfg):
        raise InvariantViolation(f"checkpoint latent space {header['latent_space_id']} differs from the model's "
                                 f"{latent_space_id(model.cfg)} (P-19)")
    model.load_state_dict(blob["state_dict"])
    if model_hash(model) != header["model_hash"]:
        raise InvariantViolation("checkpoint weights do not match their recorded hash (corrupt or edited file)")
    if optimiser is not None:
        if "optimizers" not in blob:
            raise InvariantViolation("the checkpoint holds no optimiser state")
        from nagahana.training.checkpoint import load_optimizer_state_by_name

        for name, opt in optimiser.optimizers():
            load_optimizer_state_by_name(opt, blob["optimizers"][name])
        optimiser.load_counters(blob["counters"])
    return dict(header)


def scalars(parts: Mapping[str, torch.Tensor | float], prefix: str) -> dict[str, float]:
    """Loss parts as floats for logging (NaN for anything that is not a scalar)."""
    out: dict[str, float] = {}
    for k, v in parts.items():
        if isinstance(v, torch.Tensor) and v.numel() == 1:
            x = float(v.detach())
        elif isinstance(v, float | int):
            x = float(v)
        else:
            x = math.nan
        out[f"{prefix}/{k}"] = x
    return out


def config_params(cfg: NagaHanaConfig) -> dict[str, str]:
    """Flat model config for the run logger (component.field -> repr)."""
    from dataclasses import asdict, fields, is_dataclass

    out: dict[str, str] = {"name": cfg.name, "latent_space": cfg.latent_space}
    for f in fields(cfg):
        val = getattr(cfg, f.name)
        if is_dataclass(val) and not isinstance(val, type):
            for k, v in asdict(val).items():
                out[f"{f.name}.{k}"] = repr(v)
    return out


__all__ = ["MODEL_CHECKPOINT", "PRECISIONS", "autocast", "check_precision", "config_params", "load_checkpoint",
           "save_checkpoint", "scalars", "trainable", "warmup_factor"]
