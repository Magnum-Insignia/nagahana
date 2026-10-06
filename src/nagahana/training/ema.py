"""Exponential moving average of trained weights (AS-573: built for every stage, off by default).

Maths
-----
After optimiser step t (t = 1, 2, ...), for every trained parameter theta:

    d_t       = min(decay, (1 + t) / (10 + t))      with warm-up, else d_t = decay
    ema_t     = d_t * ema_{t-1} + (1 - d_t) * theta_t,      ema_0 = theta_0

The warm-up term is TensorFlow's `ExponentialMovingAverage(num_updates=t)` rule: early averages are
not dominated by the random initial weights. Polyak averaging (Polyak and Juditsky, SIAM J. Control
Optim. 30(4), 1992) is the limit of a decay close to 1.

The averages are kept in float32 beside the float32 master weights (D-54), with the parameters'
placement: a sharded parameter (FSDP) keeps a sharded average, so the memory cost per rank is one copy
of the rank's shard. `state_dict()` gathers full tensors (a collective under FSDP: every rank calls
it); `load_state_dict()` takes full tensors and re-shards them locally. `swapped()` exchanges
averages and weights for an evaluation and restores the weights afterwards.

Invariant (tests/test_training_engine.py): with decay d and no warm-up, after n steps the average of a
known parameter sequence equals the closed form.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator, Sequence
from typing import Any

import torch
from torch import nn

from nagahana.core.errors import InvariantViolation
from nagahana.training.assumptions import use


def full_tensor(t: torch.Tensor) -> torch.Tensor:
    """The full value of a tensor that may be a distributed (sharded) tensor."""
    fn = getattr(t, "full_tensor", None)
    return fn() if callable(fn) else t


def shard_like(full: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    """`full` with the placement of `like`: re-sharded locally for a distributed tensor (no communication)."""
    mesh = getattr(like, "device_mesh", None)
    if mesh is None:
        return full.to(device=like.device, dtype=like.dtype)
    from torch.distributed.tensor import distribute_tensor

    local_device = like.to_local().device                                    # type: ignore[attr-defined]
    return distribute_tensor(full.to(device=local_device, dtype=like.dtype), mesh, like.placements,  # type: ignore[attr-defined]
                             src_data_rank=None)


class WeightEMA:
    """EMA of named parameters (module docstring)."""

    def __init__(self, named: Sequence[tuple[str, nn.Parameter]], *, decay: float, warmup: bool) -> None:
        use("AS-573", by=__name__)
        if not 0.0 < decay < 1.0:
            raise ValueError("decay must lie in (0, 1)")
        self.names = [n for n, _ in named]
        self.params = [p for _, p in named]
        self.decay = float(decay)
        self.warmup = bool(warmup)
        self.updates = 0
        with torch.no_grad():
            self.shadow = [p.detach().clone().float() for p in self.params]

    def current_decay(self) -> float:
        """d_t for the next update."""
        t = self.updates + 1
        return min(self.decay, (1.0 + t) / (10.0 + t)) if self.warmup else self.decay

    @torch.no_grad()
    def update(self) -> None:
        """One EMA update after an optimiser step: ema <- ema + (1 - d) (theta - ema)."""
        d = self.current_decay()
        for s, p in zip(self.shadow, self.params, strict=True):
            s.lerp_(p.detach().float(), 1.0 - d)
        self.updates += 1

    @contextlib.contextmanager
    def swapped(self) -> Iterator[None]:
        """Use the averages as the weights inside the block; the trained weights come back afterwards."""
        with torch.no_grad():
            saved = [p.detach().clone() for p in self.params]
            for p, s in zip(self.params, self.shadow, strict=True):
                p.copy_(s.to(p.dtype))
        try:
            yield
        finally:
            with torch.no_grad():
                for p, s in zip(self.params, saved, strict=True):
                    p.copy_(s)

    def state_dict(self) -> dict[str, Any]:
        """Full averages by parameter name and the update count (a collective for sharded parameters)."""
        return {"names": list(self.names), "updates": self.updates, "decay": self.decay, "warmup": self.warmup,
                "shadow": {n: full_tensor(s).detach().to("cpu") for n, s in zip(self.names, self.shadow, strict=True)}}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore `state_dict()`; names must match the trained parameters exactly."""
        if list(state["names"]) != self.names:
            raise InvariantViolation("EMA state names differ from the trained parameters")
        if float(state["decay"]) != self.decay or bool(state["warmup"]) != self.warmup:
            raise InvariantViolation("EMA decay or warm-up differ from the checkpoint's (a resumed run keeps them)")
        self.updates = int(state["updates"])
        with torch.no_grad():
            self.shadow = [shard_like(state["shadow"][n].float(), s) for s, n in zip(self.shadow, self.names, strict=True)]


__all__ = ["WeightEMA", "full_tensor", "shard_like"]
