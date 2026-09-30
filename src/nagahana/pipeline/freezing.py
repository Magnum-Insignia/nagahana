"""Freezing modules between stages, with proof that frozen weights did not move.

Stage 4 pretrains TAAFT "with frozen pretrained AE, Decoder & TSTCT" [I-01]. A freeze that silently
fails (an optimiser built before the freeze, a module left in the parameter groups) would retrain the
perceptors and invalidate the Environment caches built with them (memory/kvcache.py). So freezing
comes with a fingerprint: a hash of the frozen parameters taken before the stage and checked after.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable

from torch import nn

from nagahana.core.errors import InvariantViolation


def freeze(modules: Iterable[nn.Module]) -> None:
    """Stop gradients for every parameter of `modules` and put them in eval mode."""
    for m in modules:
        m.eval()
        for p in m.parameters():
            p.requires_grad_(False)


def fingerprint(module: nn.Module) -> str:
    """SHA-256 over parameter names, shapes and bytes, in a stable order.

    Also the natural `model_hash` for KV-cache compatibility (memory/kvcache.py).
    """
    h = hashlib.sha256()
    for name, p in sorted(module.state_dict().items()):
        h.update(name.encode())
        h.update(str(tuple(p.shape)).encode())
        h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def assert_unchanged(module: nn.Module, before: str, *, what: str) -> None:
    """Raise if a frozen module's fingerprint changed during a stage."""
    after = fingerprint(module)
    if after != before:
        raise InvariantViolation(f"{what} changed while frozen ({before[:12]}… → {after[:12]}…)")
