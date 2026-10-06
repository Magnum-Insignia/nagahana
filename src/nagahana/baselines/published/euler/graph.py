"""Snapshot graphs of an event stream (EULER's input).

Events (time, source, destination) are cut into snapshots of length delta (30 minutes in the paper):
snapshot index s = floor((t - t0) / delta), with t0 the start of the training stream, so training and
scoring use the same snapshot boundaries. A snapshot's graph holds each distinct directed pair once,
with the number of events between them as its multiplicity.

Node index: index 0 is reserved for nodes unseen in training (their feature row is never trained); the
training nodes take 1 ... V in order of first appearance.

GCN propagation (Kipf and Welling, ICLR 2017, arXiv:1609.02907) with the normalisation of PyTorch
Geometric's GCNConv for a directed edge list (messages flow source -> destination; degrees are counted
on destinations, self-loops added):

    deg_i = 1 + sum_{(j -> i)} w_ji,      norm_ji = w_ji / sqrt(deg_j deg_i),      norm_ii = 1 / deg_i
    x'_i = sum_{(j -> i)} norm_ji x_j + norm_ii x_i

Edge weights w are 1 per distinct pair ("binary"), the event count ("count") or log(1 + count)
("log_count"); AS-552.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch

from nagahana.core.errors import InvariantViolation


@dataclass
class NodeIndex:
    """Node name -> index (0 = unseen)."""

    names: list[str]

    @classmethod
    def build(cls, *columns: pd.Series) -> NodeIndex:
        seen: dict[str, None] = {}
        for col in columns:
            for v in col.astype(str).tolist():
                seen.setdefault(v, None)
        return cls(list(seen))

    def __post_init__(self) -> None:
        self._index = {n: i + 1 for i, n in enumerate(self.names)}

    @property
    def size(self) -> int:
        """Number of rows of the node table (training nodes + the unseen row)."""
        return len(self.names) + 1

    def encode(self, values: pd.Series) -> np.ndarray:
        return np.asarray([self._index.get(v, 0) for v in values.astype(str).tolist()], dtype=np.int64)

    def state(self) -> dict[str, Any]:
        return {"names": self.names}

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> NodeIndex:
        return cls(list(state["names"]))


@dataclass
class Snapshot:
    """Distinct directed edges of one snapshot with their event counts."""

    index: int
    src: np.ndarray
    dst: np.ndarray
    count: np.ndarray

    @property
    def n_edges(self) -> int:
        return int(self.src.size)


def snapshot_ids(times: np.ndarray, origin: float, delta: float) -> np.ndarray:
    """Snapshot index of each event time."""
    return np.floor((np.asarray(times, dtype=np.float64) - origin) / delta).astype(np.int64)


def build_snapshots(snap: np.ndarray, src: np.ndarray, dst: np.ndarray, *, first: int, last: int,
                    drop_self_loops: bool) -> list[Snapshot]:
    """One Snapshot per index first ... last (empty where no event falls)."""
    keep = (snap >= first) & (snap <= last)
    if drop_self_loops:
        keep &= src != dst
    s, u, v = snap[keep], src[keep], dst[keep]
    frame = pd.DataFrame({"s": s, "u": u, "v": v})
    counts = frame.groupby(["s", "u", "v"], sort=True).size().reset_index(name="n")
    out: list[Snapshot] = []
    by_s = {int(k): g for k, g in counts.groupby("s", sort=True)}
    for i in range(first, last + 1):
        g = by_s.get(i)
        if g is None:
            empty = np.zeros(0, dtype=np.int64)
            out.append(Snapshot(i, empty, empty, empty))
        else:
            out.append(Snapshot(i, g["u"].to_numpy(np.int64).copy(), g["v"].to_numpy(np.int64).copy(),
                                g["n"].to_numpy(np.int64).copy()))
    return out


def edge_weights(count: np.ndarray, kind: str) -> np.ndarray:
    """Propagation weights of distinct edges (module docstring)."""
    c = np.asarray(count, dtype=np.float64)
    if kind == "binary":
        return np.ones_like(c)
    if kind == "count":
        return c
    if kind == "log_count":
        return np.log1p(c)
    raise InvariantViolation(f"unknown edge weighting {kind!r}")


def gcn_operator(src: np.ndarray, dst: np.ndarray, weight: np.ndarray, n_nodes: int,
                 device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Normalised propagation (row = destination, col = source, value) plus the self-loop weights [V]."""
    w = torch.as_tensor(weight, dtype=torch.float32, device=device)
    d = torch.as_tensor(dst, dtype=torch.long, device=device)
    s = torch.as_tensor(src, dtype=torch.long, device=device)
    deg = torch.ones(n_nodes, dtype=torch.float32, device=device).index_add_(0, d, w)
    inv_sqrt = deg.rsqrt()
    values = w * inv_sqrt[s] * inv_sqrt[d]
    return d, s, values, 1.0 / deg


def propagate(x: torch.Tensor, rows: torch.Tensor, cols: torch.Tensor, values: torch.Tensor, self_weight: torch.Tensor) -> torch.Tensor:
    """x' = A_norm x + diag(self_weight) x, with A_norm given as (rows, cols, values); x [V, d]."""
    out = x * self_weight[:, None]
    if rows.numel():
        out = out.index_add(0, rows, x[cols] * values[:, None])
    return out


def mean_neighbours(x: torch.Tensor, src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """Mean of the source features over each destination's in-neighbours (0 for nodes without any)."""
    out = torch.zeros_like(x)
    if src.numel() == 0:
        return out
    out = out.index_add(0, dst, x[src])
    deg = torch.zeros(x.shape[0], dtype=x.dtype, device=x.device).index_add_(0, dst, torch.ones_like(dst, dtype=x.dtype))
    return out / deg.clamp_min(1.0)[:, None]
