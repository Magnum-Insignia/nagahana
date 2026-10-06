"""Training utilities shared by the neural reproductions.

Mini-batches are drawn from a NumPy generator seeded by the baseline, so the order of examples is part
of the reproducible state. Weights are stored with `torch.save` of the state dict and read back with
`weights_only=True`, which refuses arbitrary pickled objects.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeVar

import numpy as np
import torch
from torch import nn

M = TypeVar("M", bound=nn.Module)


def resolve_device(name: str) -> torch.device:
    """A torch.device; a CUDA device that is not available raises instead of silently using the CPU."""
    dev = torch.device(name)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"device {name!r} requested but CUDA is not available")
    return dev


def minibatches(n: int, batch_size: int, rng: np.random.Generator, *, shuffle: bool = True) -> Iterator[np.ndarray]:
    """Index arrays covering 0 ... n - 1 in batches (a fresh permutation per call when shuffling)."""
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    order = rng.permutation(n) if shuffle else np.arange(n)
    for start in range(0, n, batch_size):
        yield order[start:start + batch_size]


def balanced_minibatches(labels: np.ndarray, batch_size: int, steps: int, rng: np.random.Generator) -> Iterator[np.ndarray]:
    """`steps` batches, each with half of its rows drawn from each binary class (with replacement).

    Used where a paper trains on class-balanced batches; a class absent from the data falls back to
    uniform sampling over all rows.
    """
    pos = np.nonzero(labels == 1)[0]
    neg = np.nonzero(labels == 0)[0]
    for _ in range(steps):
        if pos.size == 0 or neg.size == 0:
            yield rng.integers(0, labels.size, size=batch_size)
            continue
        k = batch_size // 2
        idx = np.concatenate([rng.choice(pos, size=k, replace=True), rng.choice(neg, size=batch_size - k, replace=True)])
        yield idx[rng.permutation(idx.size)]


@dataclass
class EarlyStopping:
    """Stop when the monitored loss has not improved by more than `min_delta` for `patience` checks.

    Keeps a copy of the best state dict so the model can be restored to its best epoch.
    """

    patience: int
    min_delta: float = 0.0
    best: float = math.inf
    bad: int = 0
    best_epoch: int = -1
    best_state: dict[str, torch.Tensor] = field(default_factory=dict)

    def step(self, loss: float, model: nn.Module, epoch: int) -> bool:
        """Record `loss` of `epoch`; return True when training should stop."""
        if loss < self.best - self.min_delta:
            self.best, self.bad, self.best_epoch = loss, 0, epoch
            self.best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            return False
        self.bad += 1
        return self.bad >= self.patience

    def restore(self, model: nn.Module) -> None:
        """Load the best state dict recorded so far (no-op when none was recorded)."""
        if self.best_state:
            model.load_state_dict(self.best_state)


def save_module(module: nn.Module, path: Path) -> str:
    """Write the module's state dict; return the file name."""
    torch.save({k: v.detach().cpu() for k, v in module.state_dict().items()}, path)
    return path.name


def load_module(module: M, path: Path, device: torch.device) -> M:
    """Load a state dict written by `save_module` (weights only) into `module` and move it to `device`."""
    state = torch.load(path, map_location="cpu", weights_only=True)
    module.load_state_dict(state)
    module.to(device)
    return module


@torch.no_grad()
def predict_batched(fn: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor, batch_size: int) -> torch.Tensor:
    """Apply `fn` to x in batches along the first axis and concatenate the results."""
    outs = [fn(x[i:i + batch_size]) for i in range(0, x.shape[0], batch_size)]
    return torch.cat(outs, dim=0) if outs else torch.zeros((0,))


#: Adam epsilon of Keras (1e-7), used by the reproductions of Keras-based papers in place of PyTorch's 1e-8.
KERAS_ADAM_EPS = 1e-7


def keras_dense_init(module: nn.Module) -> None:
    """Keras Dense defaults on every nn.Linear of `module`: Glorot-uniform weights, zero bias."""
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


def keras_lstm_init(lstm: nn.LSTM) -> None:
    """Keras LSTM defaults: Glorot-uniform input kernel, orthogonal recurrent kernel, zero bias with the
    forget-gate bias set to 1 (unit_forget_bias). PyTorch has two bias vectors per layer; the 1 is put in
    one of them so the sum is 1, as in Keras's single bias.
    """
    h = lstm.hidden_size
    for name, p in lstm.named_parameters():
        if name.startswith("weight_ih"):
            # PyTorch stacks the gates (i, f, g, o) along the first axis; Glorot-uniform per gate block,
            # with fan-in and fan-out of the full kernel as in Keras.
            nn.init.xavier_uniform_(p)
        elif name.startswith("weight_hh"):
            # Keras makes its [h, 4h] recurrent kernel orthogonal as a whole (orthonormal rows); PyTorch
            # stores the transpose [4h, h], so one orthogonal init (orthonormal columns) is the same law.
            nn.init.orthogonal_(p)
        elif name.startswith("bias_ih"):
            nn.init.zeros_(p)
            p.data[h:2 * h] = 1.0
        elif name.startswith("bias_hh"):
            nn.init.zeros_(p)


def keras_batchnorm(width: int) -> nn.BatchNorm1d:
    """BatchNorm1d with Keras defaults: epsilon 1e-3 and moving-average momentum 0.99 (PyTorch 0.01)."""
    return nn.BatchNorm1d(width, eps=1e-3, momentum=0.01)


def causal_windows(n: int, length: int, groups: np.ndarray | None = None) -> np.ndarray:
    """Index matrix [n, length] of each row's window: the `length - 1` preceding rows and the row itself.

    Row i's window is [i - length + 1, ..., i] in the given (time) order; positions before the first row
    (or before the first row of the same group, when `groups` is given) are -1, to be padded by the
    caller. With groups, the preceding rows are the preceding rows of the same group (for example the
    same source host), still in the given order.
    """
    if length < 1:
        raise ValueError("window length must be >= 1")
    offsets = np.arange(length - 1, -1, -1)[None, :]                                  # [1, length]
    if groups is None:
        base = np.arange(n)[:, None] - offsets                                         # [n, length]
        return np.where(base >= 0, base, -1).astype(np.int64)
    out = np.full((n, length), -1, dtype=np.int64)
    group_codes = np.unique(np.asarray(groups), return_inverse=True)[1]
    for g in np.unique(group_codes):
        rows = np.nonzero(group_codes == g)[0]                                         # this group, in order
        base = np.arange(rows.size)[:, None] - offsets                                 # positions within the group
        out[rows] = np.where(base >= 0, rows[np.clip(base, 0, None)], -1)
    return out
