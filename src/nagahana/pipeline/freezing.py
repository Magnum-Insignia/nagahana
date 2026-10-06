"""Freezing modules between stages, with proof that frozen weights did not move (ADR-0006; AS-593).

Stage 2 pretrains TAAFT "with frozen pretrained AE, Decoder & TSTCT" [I-01], and stage 3 keeps the
perceptors frozen (AS-413). A freeze that silently fails (an optimiser built before the freeze, a
module left in a parameter group, an in-place update by mistake) would retrain the perceptors and
invalidate the Environment caches built with them (memory/kvcache.py). Freezing therefore comes with
proof, at three costs (AS-593):

1. structure, once per stage: every frozen parameter has `requires_grad = False` and belongs to no
   optimiser parameter group (exact);
2. a checksum, every `verify_every` optimiser steps, computed on the parameters' own device: per
   tensor the float64 sums S0 = sum_i x_i, S1 = sum_i w_i x_i and S2 = sum_i x_i^2 with w_i = 1 + i / n
   (i the flat index); an accidental update that leaves all three sums of every tensor bit-identical
   would have to be constructed on purpose;
3. the SHA-256 fingerprint of names, shapes and bytes, at the start and the end of the stage and at
   every checkpoint (exact; `fingerprint`, also the natural model hash of a frozen module).

`freeze` and `unfreeze` set the training mode and the gradient flags; `assert_unchanged` compares a
fingerprint.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping, Sequence

import torch
from torch import nn

from nagahana.core.errors import InvariantViolation


def freeze(modules: Iterable[nn.Module]) -> None:
    """Stop gradients for every parameter of `modules` and put them in eval mode."""
    for m in modules:
        m.eval()
        for p in m.parameters():
            p.requires_grad_(False)


def unfreeze(modules: Iterable[nn.Module]) -> None:
    """Re-enable gradients and training mode (the inverse of `freeze`)."""
    for m in modules:
        m.train()
        for p in m.parameters():
            p.requires_grad_(True)


def fingerprint(module: nn.Module) -> str:
    """SHA-256 over parameter and buffer names, shapes and bytes, in a stable order."""
    h = hashlib.sha256()
    for name, p in sorted(module.state_dict().items()):
        full = getattr(p, "full_tensor", None)
        t = full() if callable(full) else p
        h.update(name.encode())
        h.update(str(tuple(t.shape)).encode())
        h.update(t.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes() if t.numel() else b"")
    return h.hexdigest()


def assert_unchanged(module: nn.Module, before: str, *, what: str) -> None:
    """Raise if a frozen module's fingerprint changed during a stage."""
    after = fingerprint(module)
    if after != before:
        raise InvariantViolation(f"{what} changed while frozen ({before[:12]}... -> {after[:12]}...)")


def checksum(module: nn.Module) -> torch.Tensor:
    """float64 [n_tensors, 3] of (S0, S1, S2) per parameter (module docstring), on the CPU."""
    rows = []
    for _, p in sorted(module.named_parameters()):
        x = p.detach().reshape(-1).double()
        n = max(1, x.numel())
        w = 1.0 + torch.arange(x.numel(), device=x.device, dtype=torch.float64) / n
        rows.append(torch.stack([x.sum(), (w * x).sum(), (x * x).sum()]).cpu())
    return torch.stack(rows) if rows else torch.zeros(0, 3, dtype=torch.float64)


class FreezeGuard:
    """Proof that frozen modules stay unchanged during a stage (module docstring).

    modules: name -> frozen module. optimizers: the stage's optimisers (none may hold a frozen
    parameter). verify_every: optimiser steps between checksums.
    """

    def __init__(self, modules: Mapping[str, nn.Module], *, optimizers: Sequence[torch.optim.Optimizer] = (),
                 verify_every: int = 1) -> None:
        if verify_every < 1:
            raise ValueError("verify_every must be >= 1")
        self.modules = dict(modules)
        self.optimizers = list(optimizers)
        self.verify_every = verify_every
        self._fingerprints: dict[str, str] = {}
        self._checksums: dict[str, torch.Tensor] = {}

    def check_structure(self) -> None:
        """Every frozen parameter has requires_grad False and is in no optimiser group."""
        held = {id(p) for opt in self.optimizers for g in opt.param_groups for p in g["params"]}
        for name, m in self.modules.items():
            for pname, p in m.named_parameters():
                if p.requires_grad:
                    raise InvariantViolation(f"frozen {name}.{pname} requires a gradient")
                if id(p) in held:
                    raise InvariantViolation(f"frozen {name}.{pname} is held by an optimiser")

    def start(self) -> None:
        """Record fingerprints and checksums at the start of the stage."""
        self.check_structure()
        self._fingerprints = {n: fingerprint(m) for n, m in self.modules.items()}
        self._checksums = {n: checksum(m) for n, m in self.modules.items()}

    def check(self, step: int) -> None:
        """Compare checksums every `verify_every` steps."""
        if not self._checksums or step % self.verify_every != 0:
            return
        for n, m in self.modules.items():
            if not torch.equal(checksum(m), self._checksums[n]):
                raise InvariantViolation(f"frozen {n} changed by optimiser step {step} (checksum)")

    def verify_full(self) -> None:
        """Compare SHA-256 fingerprints (stage end, checkpoints)."""
        if not self._fingerprints:
            return
        self.check_structure()
        for n, m in self.modules.items():
            assert_unchanged(m, self._fingerprints[n], what=f"frozen {n}")


__all__ = ["FreezeGuard", "assert_unchanged", "checksum", "fingerprint", "freeze", "unfreeze"]
