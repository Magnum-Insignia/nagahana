"""Shared machinery of the LR models: raw-row access, standardisation per training set, CV groups.

Raw rows. A model reads the raw design of a set of units either from a matrix held in memory (the trigger
design, or the update design materialised once for the in-memory L-BFGS solver) or chunk by chunk from the
corpus (the streamed solvers). `RawRows.factory(positions)` serves the rows of any subset of the units in
both cases, so a fold of the blocked cross-validation, the final training set and the validation set are
all read the same way.

Standardisation per training set. Every fit (each fold of the cross-validation and the final fit)
standardises with statistics of its own training rows only, then drops the columns that are constant on
them (FeatureConfig.drop_constant); validation and test rows are transformed with those statistics and
never enter them (Hastie, Tibshirani and Friedman, The Elements of Statistical Learning, 2nd ed., 2009,
Section 7.10.2, on preprocessing inside the cross-validation loop; AS-504).
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass

import numpy as np

from nagahana.core.errors import InvariantViolation

from .config import FeatureConfig, SolverConfig, StandardiserConfig
from .corpus import LRCorpus
from .design import RawChunkFactory, Units, UpdateDesign
from .standardise import Standardiser


@dataclass
class RawRows:
    """Raw design rows of a unit set (module docstring). Positions index the unit set."""

    n_cols: int
    matrix: np.ndarray | None = None                 # float32 [N, D] when held in memory
    design: UpdateDesign | None = None               # otherwise streamed from the corpus
    units: Units | None = None
    cached: np.ndarray | None = None                 # positions held in `matrix` (sorted) when partially materialised

    @classmethod
    def from_matrix(cls, x: np.ndarray) -> RawRows:
        x = np.asarray(x, dtype=np.float32)
        return cls(n_cols=int(x.shape[1]), matrix=x)

    @classmethod
    def from_design(cls, design: UpdateDesign, units: Units, *, materialise: np.ndarray | None, chunk_rows: int) -> RawRows:
        """Stream the update design; with `materialise`, hold the raw rows of those positions in memory."""
        d = len(design.columns())
        out = cls(n_cols=d, design=design, units=units)
        if materialise is not None:
            pos = np.unique(np.asarray(materialise, dtype=np.int64))
            x = np.empty((pos.size, d), dtype=np.float32)
            sub = units.take(pos)
            for i in np.unique(sub.source):
                sel = np.flatnonzero(sub.source == i)
                for a in range(0, sel.size, chunk_rows):
                    p = sel[a:a + chunk_rows]
                    x[p] = design.chunk(int(i), sub.row[p])
            out.matrix, out.cached = x, pos
        return out

    def factory(self, positions: np.ndarray) -> RawChunkFactory:
        """chunk_rows -> iterator of (positions, raw chunk) over the given positions, in order."""
        positions = np.asarray(positions, dtype=np.int64)

        def gen(chunk_rows: int) -> Iterator[tuple[np.ndarray, np.ndarray]]:
            if self.matrix is not None and self.cached is None:
                for a in range(0, positions.size, chunk_rows):
                    p = positions[a:a + chunk_rows]
                    yield p, self.matrix[p]
                return
            if self.matrix is not None and self.cached is not None:
                loc = np.searchsorted(self.cached, positions)
                ok = (loc < self.cached.size) & (self.cached[np.minimum(loc, self.cached.size - 1)] == positions)
                if ok.all():
                    for a in range(0, positions.size, chunk_rows):
                        yield positions[a:a + chunk_rows], self.matrix[loc[a:a + chunk_rows]]
                    return
            if self.design is None or self.units is None:
                raise InvariantViolation("raw rows outside the materialised set need the design")
            sub = self.units.take(positions)
            for i in np.unique(sub.source):
                sel = np.flatnonzero(sub.source == i)
                for a in range(0, sel.size, chunk_rows):
                    p = sel[a:a + chunk_rows]
                    yield positions[p], self.design.chunk(int(i), sub.row[p])

        return gen

    def chunks(self, positions: np.ndarray, chunk_rows: int) -> Callable[[], Iterator[np.ndarray]]:
        """A re-iterable factory of raw chunks only (for the standardiser)."""
        f = self.factory(positions)
        return lambda: (x for _p, x in f(chunk_rows))


def fit_standardiser(raw: RawRows, positions: np.ndarray, binary: np.ndarray, std_cfg: StandardiserConfig,
                     feat_cfg: FeatureConfig, chunk_rows: int) -> tuple[Standardiser, np.ndarray]:
    """Standardiser on the given (training) positions and the indices of the columns kept."""
    if np.asarray(positions).size == 0:
        raise InvariantViolation("a standardiser needs at least one training row")
    std = Standardiser.fit(raw.chunks(positions, chunk_rows), binary, std_cfg)
    keep = np.flatnonzero(~std.constant) if feat_cfg.drop_constant else np.arange(std.n_cols)
    return std, keep.astype(np.int64)


def _locator(positions: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    """Map positions of a chunk back to their index in `positions` (which must be distinct)."""
    order = np.argsort(positions, kind="mergesort")
    sorted_pos = positions[order]
    if sorted_pos.size > 1 and np.any(sorted_pos[1:] == sorted_pos[:-1]):
        raise InvariantViolation("positions must be distinct")
    return lambda pos: order[np.searchsorted(sorted_pos, pos)]


def standardised(raw: RawRows, positions: np.ndarray, std: Standardiser, keep: np.ndarray, chunk_rows: int) -> np.ndarray:
    """The standardised rows [len(positions), len(keep)] in position order (in memory)."""
    positions = np.asarray(positions, dtype=np.int64)
    out = np.empty((positions.size, np.asarray(keep).size), dtype=np.float64)
    locate = _locator(positions)
    for pos, x in raw.factory(positions)(chunk_rows):
        out[locate(pos)] = std.transform(x, keep)
    return out


def apply_linear(raw: RawRows, positions: np.ndarray, std: Standardiser, keep: np.ndarray, coef: np.ndarray,
                 chunk_rows: int) -> np.ndarray:
    """Z @ coef for the given positions, streamed; coef [D'] or [D', K] -> [n] or [n, K] in position order."""
    positions = np.asarray(positions, dtype=np.int64)
    coef = np.asarray(coef, dtype=np.float64)
    out = np.empty((positions.size, *coef.shape[1:]), dtype=np.float64)
    locate = _locator(positions)
    for pos, x in raw.factory(positions)(chunk_rows):
        out[locate(pos)] = std.transform(x, keep) @ coef
    return out


#: () -> fresh iterator of (positions, standardised float64 rows).
ZRows = Callable[[], Iterator[tuple[np.ndarray, np.ndarray]]]


def solver_rows(raw: RawRows, positions: np.ndarray, std: Standardiser, keep: np.ndarray, solver: SolverConfig) -> ZRows:
    """Standardised rows of the given positions for an exact solver.

    "lbfgs" standardises the rows once and serves them as one block; "streamed_lbfgs" standardises chunk by
    chunk on every pass (memory bounded by `chunk_rows`). The averaged mini-batch solver serves the binary
    detector, whose per-update design is the one that can outgrow memory; the trigger-level models hold one
    row per cadence window and are fitted exactly.
    """
    positions = np.asarray(positions, dtype=np.int64)
    if solver.method == "lbfgs":
        z = standardised(raw, positions, std, keep, solver.chunk_rows)
        return lambda: iter([(positions, z)])
    if solver.method == "streamed_lbfgs":
        f = raw.factory(positions)
        return lambda: ((p, std.transform(x, keep)) for p, x in f(solver.chunk_rows))
    raise InvariantViolation(f"solver {solver.method!r} is for the binary detector; use 'lbfgs' or 'streamed_lbfgs' here")


def groups_of(corpus: LRCorpus, units: Units, group_by: str) -> np.ndarray:
    """Group key of every unit for blocked CV: its network or its source id."""
    if group_by == "network":
        names = np.asarray([s.network for s in corpus.sources], dtype=str)
    elif group_by == "source":
        names = np.asarray([s.source_id for s in corpus.sources], dtype=str)
    else:
        raise InvariantViolation(f"unknown group_by {group_by!r}")
    return names[units.source]


__all__ = ["RawRows", "ZRows", "apply_linear", "fit_standardiser", "groups_of", "solver_rows", "standardised"]
