"""Traffic graphs for spatial analytics: nodes are entities, edges are who talked to whom.

A `TrafficGraph` holds directed, weighted edges, optionally labelled with a relation plane (the
multiplex planes of the CVG-AE, AS-01 rules of `graph.planes`) and node labels (stable entity keys,
so that graphs of different windows can be aligned). Parallel edges are aggregated on construction
(weights summed per (source, target, plane)); self-loops are dropped and counted.

Sources of graphs:
    from_edges      plain arrays (source, target, weight, plane)
    from_contacts   the per-plane first-contact matrices of a window (`graph.window.WindowStructure`):
                    an undirected edge per plane wherever the contact time is finite (D-52 star semantics
                    are already inside the contact matrices)
    corpus graphs   built in `analytics.spatial.analysis` from the corpus rows (initiator -> responder)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import sparse


@dataclass(frozen=True)
class TrafficGraph:
    """Directed weighted (multiplex) graph on nodes 0 ... n - 1 (module docstring).

    src, dst: int64 [m]; weight: float64 [m] > 0; plane: object [m] plane label of each edge, or None
    for a single-plane graph; node_label: object [n] stable node keys, or None; node_kind: object [n]
    entity kinds, or None. directed: whether edge direction is meaningful. self_loops_dropped: count.
    """

    n: int
    src: np.ndarray
    dst: np.ndarray
    weight: np.ndarray
    plane: np.ndarray | None = None
    node_label: np.ndarray | None = None
    node_kind: np.ndarray | None = None
    directed: bool = True
    self_loops_dropped: int = field(default=0, compare=False)

    @classmethod
    def from_edges(
        cls,
        src: np.ndarray,
        dst: np.ndarray,
        *,
        weight: np.ndarray | None = None,
        n: int | None = None,
        plane: np.ndarray | None = None,
        node_label: np.ndarray | None = None,
        node_kind: np.ndarray | None = None,
        directed: bool = True,
    ) -> TrafficGraph:
        """Graph from edge arrays; parallel edges are summed and self-loops dropped (module docstring)."""
        s = np.asarray(src, dtype=np.int64).reshape(-1)
        t = np.asarray(dst, dtype=np.int64).reshape(-1)
        w = np.ones(s.size) if weight is None else np.asarray(weight, dtype=np.float64).reshape(-1)
        if not (s.size == t.size == w.size):
            raise ValueError("src, dst and weight must have the same length")
        if (w <= 0).any() or not np.isfinite(w).all():
            raise ValueError("edge weights must be finite and > 0")
        nn = int(max(s.max(initial=-1), t.max(initial=-1)) + 1) if n is None else int(n)
        if s.size and (min(s.min(), t.min()) < 0 or max(s.max(), t.max()) >= nn):
            raise ValueError("node index outside [0, n)")
        loops = s == t
        s, t, w = s[~loops], t[~loops], w[~loops]
        pl = None if plane is None else np.asarray(plane, dtype=object).reshape(-1)[~loops]
        if not directed:                                                 # canonical orientation for undirected graphs
            s, t = np.minimum(s, t), np.maximum(s, t)
        key = pd.DataFrame({"s": s, "t": t}) if pl is None else pd.DataFrame({"s": s, "t": t, "p": pl.astype(str)})
        key["w"] = w
        g = key.groupby(list(key.columns[:-1]), sort=True, as_index=False)["w"].sum()
        return cls(
            n=nn, src=g["s"].to_numpy(dtype=np.int64), dst=g["t"].to_numpy(dtype=np.int64),
            weight=g["w"].to_numpy(dtype=np.float64), plane=None if pl is None else g["p"].to_numpy(dtype=object),
            node_label=None if node_label is None else np.asarray(node_label, dtype=object),
            node_kind=None if node_kind is None else np.asarray(node_kind, dtype=object),
            directed=directed, self_loops_dropped=int(loops.sum()),
        )

    @classmethod
    def from_contacts(
        cls,
        contact_planes: np.ndarray,
        planes: tuple[str, ...],
        *,
        node_label: np.ndarray | None = None,
        node_kind: np.ndarray | None = None,
    ) -> TrafficGraph:
        """Undirected multiplex graph from per-plane first-contact matrices [V, V, P] (+inf: never)."""
        c = np.asarray(contact_planes, dtype=np.float64)
        if c.ndim != 3 or c.shape[0] != c.shape[1] or c.shape[2] != len(planes):
            raise ValueError("contact_planes must be [V, V, n_planes] with one plane name per slice")
        v = c.shape[0]
        iu, ju = np.triu_indices(v, 1)
        srcs, dsts, pls = [], [], []
        for p, name in enumerate(planes):
            ok = np.isfinite(c[iu, ju, p]) | np.isfinite(c[ju, iu, p])
            srcs.append(iu[ok])
            dsts.append(ju[ok])
            pls.append(np.full(int(ok.sum()), name, dtype=object))
        return cls.from_edges(np.concatenate(srcs), np.concatenate(dsts), n=v, plane=np.concatenate(pls),
                              node_label=node_label, node_kind=node_kind, directed=False)

    @property
    def m(self) -> int:
        """Number of (aggregated) edges."""
        return int(self.src.size)

    def planes(self) -> list[str]:
        """Plane labels present, sorted."""
        return [] if self.plane is None else sorted({str(p) for p in self.plane})

    def layer(self, plane: str) -> TrafficGraph:
        """The single-plane graph of `plane` (same node set)."""
        if self.plane is None:
            raise ValueError("graph has no planes")
        sel = self.plane == plane
        return TrafficGraph(self.n, self.src[sel], self.dst[sel], self.weight[sel], None, self.node_label,
                            self.node_kind, self.directed)

    def adjacency(self, *, weighted: bool = True, symmetric: bool | None = None) -> sparse.csr_matrix:
        """CSR adjacency, planes merged (weights summed). symmetric: default not `directed`; an
        undirected view has A[u, v] = A[v, u] = total weight between u and v."""
        sym = (not self.directed) if symmetric is None else symmetric
        w = self.weight if weighted else np.ones(self.m)
        a = sparse.coo_matrix((w, (self.src, self.dst)), shape=(self.n, self.n)).tocsr()
        a.sum_duplicates()
        if sym:
            a = (a + a.T).tocsr()
            if not weighted:
                a.data[:] = 1.0
        elif not weighted:
            a.data[:] = 1.0
        a.setdiag(0)
        a.eliminate_zeros()
        return a

    def simple_undirected(self) -> sparse.csr_matrix:
        """Binary symmetric adjacency of the underlying simple undirected graph (planes merged)."""
        return self.adjacency(weighted=False, symmetric=True)


__all__ = ["TrafficGraph"]
