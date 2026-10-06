"""Introspection hooks: record what named submodules produce during a forward pass.

The module is named `introspection` so that it never shadows the standard library's `inspect`.

What this is
------------
`ActivationRecorder` attaches forward hooks to named submodules and keeps what each one produced
during a forward pass. It is the lowest layer of the inspection stack:
- mechanistic analysis (P-16): probes on latents and checks that interventions have the expected
  effect;
- the explainability the problem statement requires: attention weights and feature attribution per
  prediction (black-box outputs are not acceptable);
- step-by-step inspection: run one input and see every intermediate result.

Design notes
------------
- Opt-in and bounded. Nothing is recorded unless a recorder is active; `max_items` caps memory use,
  because inputs at CII scale are large.
- Detached copies. Recorded tensors are detached (and optionally moved to CPU), so inspection never
  changes gradients or training.
- Names, not indices. Modules are selected by their qualified names from `model.named_modules()`, so
  recordings stay readable as architectures change.

Example
-------
>>> import torch
>>> net = torch.nn.Sequential(torch.nn.Linear(2, 3), torch.nn.ReLU())
>>> with ActivationRecorder(net, ["0", "1"]) as rec:
...     _ = net(torch.ones(1, 2))
>>> sorted(rec.records)
['0', '1']
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from torch import nn


class ActivationRecorder:
    """Record outputs of selected submodules during forward passes.

    Parameters
    ----------
    model:
        The module to inspect.
    names:
        Qualified submodule names (as in `model.named_modules()`). Unknown names raise at entry, so a
        typo cannot silently record nothing.
    to_cpu:
        Move recorded tensors to CPU (keeps accelerator memory free during long inspections).
    max_items:
        Maximum number of recorded outputs per module (oldest dropped first).
    """

    def __init__(
        self,
        model: nn.Module,
        names: Iterable[str],
        *,
        to_cpu: bool = True,
        max_items: int = 16,
    ) -> None:
        if max_items < 1:
            raise ValueError("max_items must be >= 1")
        self.model = model
        self.names = tuple(names)
        self.to_cpu = to_cpu
        self.max_items = max_items
        self.records: dict[str, list[Any]] = {}
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    def _capture(self, value: Any) -> Any:
        # Detach tensors (and move them to CPU when asked); recurse through tuples, lists and dicts.
        if isinstance(value, torch.Tensor):
            out = value.detach()
            return out.cpu() if self.to_cpu else out
        if isinstance(value, (tuple, list)):
            return type(value)(self._capture(v) for v in value)
        if isinstance(value, dict):
            return {k: self._capture(v) for k, v in value.items()}
        return value

    def __enter__(self) -> ActivationRecorder:
        modules = dict(self.model.named_modules())
        missing = [n for n in self.names if n not in modules]
        if missing:
            raise KeyError(f"Unknown submodule names: {missing}. Known (first 20): {sorted(modules)[:20]}")
        for name in self.names:
            self.records[name] = []

            def hook(_m: nn.Module, _inp: Any, out: Any, _name: str = name) -> None:
                bucket = self.records[_name]
                bucket.append(self._capture(out))
                if len(bucket) > self.max_items:
                    bucket.pop(0)

            self._handles.append(modules[name].register_forward_hook(hook))
        return self

    def __exit__(self, *exc: object) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()
