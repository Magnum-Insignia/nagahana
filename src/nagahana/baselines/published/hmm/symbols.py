"""Observation symbols for discrete HMMs from feature vectors (AS-547).

The reproduced HMM forecasters observe intrusion-detection alerts (Snort alert types, an APT detector's
alert types). The NagaHana datasets carry flows and windows, not alerts, and no alert generator is part of
the approved stack, so a stream without alert codes is turned into symbols by a k-means codebook learned
on the training rows: each row's feature vector x is mapped to

    z = (slog(x) - mu) / sigma,   slog(x) = sign(x) log(1 + |x|)       (heavy-tailed counts compressed)
    symbol(x) = argmin_c ||z - m_c||^2

with the centres m_c from k-means (Lloyd's algorithm, Lloyd, IEEE Trans. Inf. Theory 28(2), 1982) seeded
by k-means++ (Arthur and Vassilvitskii, SODA 2007): the first centre is a uniformly drawn row, each next
centre a row drawn with probability proportional to its squared distance to the nearest chosen centre.
Of `n_init` seeded runs the one with the lowest inertia sum_i min_c ||z_i - m_c||^2 is kept. An empty
cluster is re-seeded with the row farthest from its centre. Non-finite inputs are repaired with training
statistics first (preprocessing.FiniteGuard).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from nagahana.baselines.published.preprocessing import FiniteGuard
from nagahana.core.errors import InvariantViolation

_CHUNK = 65536


def _slog(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.log1p(np.abs(x))


def _sq_dist_argmin(z: np.ndarray, centres: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Nearest centre and squared distance per row, computed in chunks of rows.
    labels = np.empty(z.shape[0], dtype=np.int64)
    dist = np.empty(z.shape[0], dtype=np.float64)
    c2 = np.einsum("kd,kd->k", centres, centres)
    for s in range(0, z.shape[0], _CHUNK):
        zz = z[s:s + _CHUNK]
        d = np.einsum("nd,nd->n", zz, zz)[:, None] - 2.0 * zz @ centres.T + c2[None, :]
        labels[s:s + _CHUNK] = np.argmin(d, axis=1)
        dist[s:s + _CHUNK] = np.maximum(d[np.arange(zz.shape[0]), labels[s:s + _CHUNK]], 0.0)
    return labels, dist


def kmeans_plus_plus(z: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """k initial centres by k-means++ seeding (duplicates of chosen rows are never re-drawn)."""
    n = z.shape[0]
    centres = np.empty((k, z.shape[1]), dtype=np.float64)
    centres[0] = z[rng.integers(n)]
    closest = np.einsum("nd,nd->n", z - centres[0], z - centres[0])
    for c in range(1, k):
        total = closest.sum()
        idx = int(rng.integers(n)) if total <= 0 else int(rng.choice(n, p=closest / total))
        centres[c] = z[idx]
        closest = np.minimum(closest, np.einsum("nd,nd->n", z - centres[c], z - centres[c]))
    return centres


def lloyd(z: np.ndarray, centres: np.ndarray, *, max_iter: int, tol: float) -> tuple[np.ndarray, float]:
    """Lloyd iterations from `centres`; returns the final centres and inertia."""
    c = centres.copy()
    inertia = np.inf
    for _ in range(max_iter):
        labels, dist = _sq_dist_argmin(z, c)
        new = np.zeros_like(c)
        counts = np.bincount(labels, minlength=c.shape[0]).astype(np.float64)
        np.add.at(new, labels, z)
        empty = counts == 0
        if empty.any():
            # Re-seed empty clusters with the rows farthest from their centres.
            far = np.argsort(-dist, kind="stable")[: int(empty.sum())]
            new[empty] = z[far]
            counts[empty] = 1.0
        new /= counts[:, None]
        shift = float(np.max(np.einsum("kd,kd->k", new - c, new - c)))
        c = new
        inertia_new = float(dist.sum())
        if shift <= tol or abs(inertia - inertia_new) <= tol * max(1.0, inertia_new):
            inertia = inertia_new
            break
        inertia = inertia_new
    _, dist = _sq_dist_argmin(z, c)
    return c, float(dist.sum())


@dataclass
class KMeansCodebook:
    """Feature vectors -> symbols 0 ... n_symbols - 1 (see the module docstring)."""

    n_symbols: int
    n_init: int = 4
    max_iter: int = 100
    tol: float = 1e-6
    guard: FiniteGuard = field(default_factory=FiniteGuard)
    mean: np.ndarray = field(default_factory=lambda: np.zeros(0))
    scale: np.ndarray = field(default_factory=lambda: np.zeros(0))
    centres: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    inertia: float = float("nan")

    def _standardise(self, x: np.ndarray) -> np.ndarray:
        return (_slog(self.guard.transform(x)) - self.mean) / self.scale

    def fit(self, x: np.ndarray, rng: np.random.Generator) -> KMeansCodebook:
        if self.n_symbols < 1 or self.n_init < 1:
            raise InvariantViolation("n_symbols and n_init must be >= 1")
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2 or x.shape[0] < self.n_symbols:
            raise InvariantViolation(f"need at least {self.n_symbols} rows of a 2-D feature matrix")
        self.guard = FiniteGuard().fit(x)
        s = _slog(self.guard.transform(x))
        self.mean, self.scale = s.mean(axis=0), s.std(axis=0)
        self.scale[self.scale == 0.0] = 1.0
        z = (s - self.mean) / self.scale
        best: tuple[float, np.ndarray] | None = None
        for _ in range(self.n_init):
            c, inertia = lloyd(z, kmeans_plus_plus(z, self.n_symbols, rng), max_iter=self.max_iter, tol=self.tol)
            if best is None or inertia < best[0]:
                best = (inertia, c)
        assert best is not None
        self.inertia, self.centres = best
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Symbol of each row (int64)."""
        if self.centres.size == 0:
            raise InvariantViolation("KMeansCodebook.transform called before fit")
        labels, _ = _sq_dist_argmin(self._standardise(np.asarray(x, dtype=np.float64)), self.centres)
        return labels

    def state(self) -> dict[str, Any]:
        return {"n_symbols": self.n_symbols, "n_init": self.n_init, "max_iter": self.max_iter, "tol": self.tol,
                "guard": self.guard.state(), "mean": self.mean.tolist(), "scale": self.scale.tolist(),
                "centres": self.centres.tolist(), "inertia": self.inertia}

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> KMeansCodebook:
        return cls(int(state["n_symbols"]), int(state["n_init"]), int(state["max_iter"]), float(state["tol"]),
                   FiniteGuard.from_state(state["guard"]), np.asarray(state["mean"], dtype=np.float64),
                   np.asarray(state["scale"], dtype=np.float64), np.asarray(state["centres"], dtype=np.float64),
                   float(state["inertia"]))
