"""Filtered flag complexes: Vietoris-Rips of point clouds, weighted-clique and contact filtrations of graphs.

Flag filtration
---------------
Given a graph with vertex values f(v) and edge values f(uv) >= max(f(u), f(v)), the flag (clique)
complex contains every clique; a clique sigma enters at f(sigma) = max of the values of its vertices
and edges. f is monotone (a face never enters after a coface), so sorting by (value, dimension) is a
valid filtration order. Cliques are enumerated from the sparse graph only: a clique (v_0 < ... < v_d)
is extended by every forward neighbour x > v_d of its last vertex that is adjacent to all of v_0 ...
v_d (vectorised per dimension; edge membership by binary search on sorted edge keys). Simplices of
dimension up to `max_dim + 1` are built, which is exactly what homology up to `max_dim` needs.

Vietoris-Rips
-------------
VR_r(X) = { sigma : diam(sigma) <= r }. Edges are the pairs within the maximum radius, found by a k-d
tree range search (scipy.spatial.cKDTree.query_pairs; p = 2, inf or 1 for the euclidean, chebyshev and
cityblock metrics), never by a full distance matrix. With no radius given the enclosing radius
r_enc = min_i max_j d(x_i, x_j) is used: beyond it the complex is a cone with apex argmin_i, so every
homology class has died (Bauer, Journal of Applied and Computational Topology 5:391-423, 2021), and
truncating there loses nothing.

Landmarks
---------
Maxmin (farthest-point) landmarks L subset X (de Silva and Carlsson, SPBG 2004): start from the point
farthest from the centroid, then repeatedly add the point farthest from the chosen set. The covering
radius eps = max_x min_l d(x, l) is the Hausdorff distance d_H(X, L); since the Gromov-Hausdorff
distance satisfies d_GH(X, L) <= d_H(X, L) and Rips diagrams are 2 d_GH-stable (Chazal, Cohen-Steiner,
Guibas, Memoli and Oudot, Computer Graphics Forum 28(5):1393-1403, 2009), the landmark diagrams are
within bottleneck distance 2 eps of the full ones. The bound is reported with every landmark run.

Graph filtrations
-----------------
Weighted-clique filtration (Petri, Scolamiero, Donato and Vaccarino, PLoS ONE 8(6):e66506, 2013):
strong ties enter first. A tie strength w > 0 becomes an entry value by one of
    rank           r(w) / m, the descending rank of w among the m edges (the weight-rank filtration)
    inverse        1 / w
    negative_log   -log(w / max w)
    identity       w itself (for values that already are times or lengths)
Contact filtration of a window (graph/window.py): the edge uv enters at the first contact time
C1[u, v] and each vertex at its first contact, so the complex at time t is the clique complex of
everything that had communicated by t (D-52 star semantics are already inside C1).
"""

from __future__ import annotations

from dataclasses import dataclass
from math import comb

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

from nagahana.core.errors import InvariantViolation

#: Metric name -> Minkowski p for cKDTree and the norm of difference vectors.
METRIC_P: dict[str, float] = {"euclidean": 2.0, "chebyshev": float("inf"), "cityblock": 1.0}


@dataclass(frozen=True)
class FilteredComplex:
    """A filtered simplicial complex, simplices grouped by dimension.

    simplices[d]: int64 [m_d, d + 1] vertex tuples, each row strictly increasing.
    values[d]: float64 [m_d] filtration values (monotone: a face never enters after a coface).
    max_value: the truncation of the filtration (+inf when none).
    exact_through: homology of the intended filtration is exact through this dimension. A flag
    complex enumerated up to dimension m + 1 is exact through m (its (m + 1)-cycles may be killed by
    cliques that were not enumerated); None means the complex is taken as given, exact through its
    top dimension.
    """

    n_vertices: int
    simplices: tuple[np.ndarray, ...]
    values: tuple[np.ndarray, ...]
    max_value: float
    exact_through: int | None = None

    @property
    def dimension(self) -> int:
        """Largest dimension present (-1 for the empty complex)."""
        return len(self.simplices) - 1

    def size(self) -> int:
        """Total number of simplices."""
        return int(sum(s.shape[0] for s in self.simplices))

    def counts(self) -> list[int]:
        """Simplices per dimension."""
        return [int(s.shape[0]) for s in self.simplices]


def _edge_keys(edges: np.ndarray, n: int) -> np.ndarray:
    return edges[:, 0].astype(np.int64) * n + edges[:, 1].astype(np.int64)


def flag_complex(
    n_vertices: int,
    edges: np.ndarray,
    edge_values: np.ndarray,
    *,
    max_dim: int,
    vertex_values: np.ndarray | None = None,
    max_value: float = float("inf"),
    max_simplices: int = 5_000_000,
    chunk: int = 2_000_000,
) -> FilteredComplex:
    """The flag filtration of a graph up to dimension max_dim + 1 (module docstring).

    edges: int [m, 2] vertex pairs (any order; self-loops rejected, duplicates keep their smallest
    value); edge_values: [m]; vertex_values: [n] (default 0). Edges with value > max_value are dropped.
    An edge value below the value of one of its vertices is raised to it, so the filtration is monotone.
    """
    if max_dim < 0:
        raise ValueError("max_dim must be >= 0")
    n = int(n_vertices)
    e = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    w = np.asarray(edge_values, dtype=np.float64).reshape(-1)
    if e.shape[0] != w.shape[0]:
        raise ValueError("edges and edge_values must have the same length")
    if e.size and (e.min() < 0 or e.max() >= n):
        raise ValueError("edge endpoint outside [0, n_vertices)")
    if (e[:, 0] == e[:, 1]).any():
        raise ValueError("self-loops are not simplices")
    if not np.isfinite(w).all():
        raise ValueError("edge values must be finite (absent edges are simply not listed)")
    fv = np.zeros(n) if vertex_values is None else np.asarray(vertex_values, dtype=np.float64).reshape(-1)
    if fv.shape != (n,) or not np.isfinite(fv).all():
        raise ValueError("vertex_values must be finite and of length n_vertices")
    keep_v = fv <= max_value
    lo, hi = np.minimum(e[:, 0], e[:, 1]), np.maximum(e[:, 0], e[:, 1])
    w = np.maximum(w, np.maximum(fv[lo], fv[hi]))                          # monotone filtration
    ok = (w <= max_value) & keep_v[lo] & keep_v[hi]
    lo, hi, w = lo[ok], hi[ok], w[ok]
    # Deduplicate: keep the smallest value of a repeated edge.
    key = lo * n + hi
    order = np.lexsort((w, key))
    key, w = key[order], w[order]
    first = np.r_[True, key[1:] != key[:-1]] if key.size else np.zeros(0, dtype=bool)
    key, w = key[first], w[first]
    e1 = np.stack([key // n, key % n], axis=1)
    v_idx = np.flatnonzero(keep_v)
    simplices: list[np.ndarray] = [v_idx.reshape(-1, 1).astype(np.int64), e1]
    values: list[np.ndarray] = [fv[v_idx], w]
    total = v_idx.size + e1.shape[0]
    if total > max_simplices:
        raise InvariantViolation(f"{total} simplices exceed max_simplices={max_simplices}; lower the radius or use landmarks")
    # Forward adjacency in CSR form: neighbours x > u, sorted (edges are sorted by key = lo * n + hi).
    ptr = np.searchsorted(e1[:, 0], np.arange(n + 1)) if e1.size else np.zeros(n + 1, dtype=np.int64)
    nbr = e1[:, 1]
    for _d in range(2, max_dim + 2):
        prev, prev_w = simplices[-1], values[-1]
        if prev.shape[0] == 0:
            simplices.append(np.zeros((0, prev.shape[1] + 1), dtype=np.int64))
            values.append(np.zeros(0))
            continue
        last = prev[:, -1]
        deg = ptr[last + 1] - ptr[last]
        new_s: list[np.ndarray] = []
        new_w: list[np.ndarray] = []
        # Process simplices in chunks so the candidate arrays stay bounded.
        bounds = np.searchsorted(np.cumsum(deg), np.arange(chunk, int(deg.sum()) + chunk, chunk))
        start = 0
        for stop in np.unique(np.r_[bounds + 1, prev.shape[0]]).tolist():
            stop = min(stop, prev.shape[0])
            if stop <= start:
                continue
            sl = slice(start, stop)
            dg = deg[sl]
            owner = np.repeat(np.arange(start, stop), dg)                 # simplex of each candidate
            offs = np.cumsum(dg) - dg
            cand = nbr[np.repeat(ptr[last[sl]], dg) + np.arange(int(dg.sum())) - np.repeat(offs, dg)]
            ok = np.ones(cand.shape[0], dtype=bool)
            val = prev_w[owner].copy()
            for col in range(prev.shape[1] - 1):                          # the last vertex is adjacent by construction
                k = prev[owner, col] * n + cand
                pos = np.searchsorted(key, k)
                pos_c = np.minimum(pos, key.size - 1)
                hit = (pos < key.size) & (key[pos_c] == k)
                ok &= hit
                val = np.maximum(val, np.where(hit, w[pos_c], np.inf))
            k_last = last[owner] * n + cand
            pos = np.minimum(np.searchsorted(key, k_last), key.size - 1)
            val = np.maximum(val, w[pos])
            owner, cand, val = owner[ok], cand[ok], val[ok]
            new_s.append(np.concatenate([prev[owner], cand[:, None]], axis=1))
            new_w.append(val)
            total += int(cand.shape[0])
            if total > max_simplices:
                raise InvariantViolation(
                    f"more than max_simplices={max_simplices} simplices; lower the radius or use landmarks")
            start = stop
        simplices.append(np.concatenate(new_s) if new_s else np.zeros((0, prev.shape[1] + 1), dtype=np.int64))
        values.append(np.concatenate(new_w) if new_w else np.zeros(0))
    return FilteredComplex(n_vertices=n, simplices=tuple(simplices), values=tuple(values), max_value=float(max_value),
                           exact_through=max_dim)


def _pair_distances(points: np.ndarray, pairs: np.ndarray, p: float) -> np.ndarray:
    diff = points[pairs[:, 0]] - points[pairs[:, 1]]
    if p == 2.0:
        return np.sqrt((diff * diff).sum(axis=1))
    if np.isinf(p):
        return np.abs(diff).max(axis=1)
    return np.abs(diff).sum(axis=1)


def enclosing_radius(points: np.ndarray, *, metric: str = "euclidean", block: int = 2048) -> float:
    """min_i max_j d(x_i, x_j), computed in row blocks (module docstring)."""
    x = np.asarray(points, dtype=np.float64)
    name = {"euclidean": "euclidean", "chebyshev": "chebyshev", "cityblock": "cityblock"}[metric]
    best = np.inf
    for r0 in range(0, x.shape[0], block):
        best = min(best, float(cdist(x[r0: r0 + block], x, metric=name).max(axis=1).min()))
    return best


def rips_complex(
    points: np.ndarray,
    *,
    max_dim: int,
    max_radius: float | None = None,
    metric: str = "euclidean",
    max_simplices: int = 5_000_000,
) -> FilteredComplex:
    """Vietoris-Rips filtration of a point cloud [n, d] up to dimension max_dim + 1 (module docstring)."""
    x = np.asarray(points, dtype=np.float64)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("points must be a finite [n, d] array")
    if metric not in METRIC_P:
        raise ValueError(f"metric must be one of {sorted(METRIC_P)}")
    p = METRIC_P[metric]
    r = enclosing_radius(x, metric=metric) if max_radius is None else float(max_radius)
    if r < 0:
        raise ValueError("max_radius must be >= 0")
    pairs = cKDTree(x).query_pairs(r=r, p=p, output_type="ndarray").astype(np.int64)
    d = _pair_distances(x, pairs, p) if pairs.size else np.zeros(0)
    ok = d <= r                                                           # exact closed-ball filter
    return flag_complex(x.shape[0], pairs[ok], d[ok], max_dim=max_dim, max_value=r, max_simplices=max_simplices)


def rips_from_distances(
    distances: np.ndarray,
    *,
    max_dim: int,
    max_radius: float | None = None,
    max_simplices: int = 5_000_000,
) -> FilteredComplex:
    """Vietoris-Rips filtration of a symmetric distance matrix [n, n] (zero diagonal)."""
    dm = np.asarray(distances, dtype=np.float64)
    if dm.ndim != 2 or dm.shape[0] != dm.shape[1]:
        raise ValueError("distances must be a square matrix")
    if not np.allclose(dm, dm.T, rtol=0, atol=1e-12 * max(1.0, float(np.abs(dm).max(initial=0)))):
        raise ValueError("distances must be symmetric")
    n = dm.shape[0]
    r = float(dm.max(axis=1).min()) if max_radius is None else float(max_radius)
    iu, ju = np.triu_indices(n, 1)
    d = dm[iu, ju]
    ok = np.isfinite(d) & (d <= r)
    return flag_complex(n, np.stack([iu[ok], ju[ok]], axis=1), d[ok], max_dim=max_dim, max_value=r,
                        max_simplices=max_simplices)


def maxmin_landmarks(points: np.ndarray, m: int, *, metric: str = "euclidean") -> tuple[np.ndarray, float]:
    """(landmark indices [m], covering radius) by maxmin selection (module docstring); deterministic."""
    x = np.asarray(points, dtype=np.float64)
    n = x.shape[0]
    if not 1 <= m <= n:
        raise ValueError(f"m must be in [1, n]; got m={m}, n={n}")
    name = {"euclidean": "euclidean", "chebyshev": "chebyshev", "cityblock": "cityblock"}[metric]
    first = int(np.argmax(cdist(x.mean(axis=0, keepdims=True), x, metric=name)[0]))
    chosen = np.empty(m, dtype=np.int64)
    chosen[0] = first
    mind = cdist(x[first: first + 1], x, metric=name)[0]
    for t in range(1, m):
        nxt = int(np.argmax(mind))
        chosen[t] = nxt
        mind = np.minimum(mind, cdist(x[nxt: nxt + 1], x, metric=name)[0])
    return chosen, float(mind.max())


def weight_values(weights: np.ndarray, transform: str) -> np.ndarray:
    """Entry values of edges from tie strengths w > 0 (module docstring, "Graph filtrations")."""
    w = np.asarray(weights, dtype=np.float64)
    if transform == "identity":
        return w.copy()
    if (w <= 0).any() or not np.isfinite(w).all():
        raise ValueError("tie strengths must be finite and > 0")
    if transform == "rank":
        order = np.argsort(-w, kind="stable")
        ranks = np.empty(w.size)
        ranks[order] = np.arange(1, w.size + 1)
        # Equal weights share the rank of their first occurrence (they enter together).
        uniq, inv = np.unique(-w, return_inverse=True)
        first_rank = np.full(uniq.size, np.inf)
        np.minimum.at(first_rank, inv.reshape(-1), ranks)
        return first_rank[inv.reshape(-1)] / w.size
    if transform == "inverse":
        return 1.0 / w
    if transform == "negative_log":
        return -np.log(w / w.max())
    raise ValueError(f"unknown weight transform {transform!r}")


def weighted_clique_complex(
    n_vertices: int,
    edges: np.ndarray,
    weights: np.ndarray,
    *,
    max_dim: int,
    transform: str = "rank",
    max_value: float = float("inf"),
    max_simplices: int = 5_000_000,
) -> FilteredComplex:
    """Weighted-clique filtration: strong ties enter first (module docstring)."""
    e = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    # Repeated pairs (several flows between two entities) add their strengths before the transform.
    lo, hi = np.minimum(e[:, 0], e[:, 1]), np.maximum(e[:, 0], e[:, 1])
    keep = lo != hi
    key = lo[keep] * int(n_vertices) + hi[keep]
    uniq, inv = np.unique(key, return_inverse=True)
    total = np.bincount(inv.reshape(-1), weights=w[keep], minlength=uniq.size)
    pairs = np.stack([uniq // int(n_vertices), uniq % int(n_vertices)], axis=1)
    return flag_complex(int(n_vertices), pairs, weight_values(total, transform), max_dim=max_dim, max_value=max_value,
                        max_simplices=max_simplices)


def contact_complex(contact1: np.ndarray, *, max_dim: int, max_simplices: int = 5_000_000) -> FilteredComplex:
    """Contact filtration of a window from its first-contact matrix C1 [V, V] (+inf: never)."""
    c1 = np.asarray(contact1, dtype=np.float64)
    if c1.ndim != 2 or c1.shape[0] != c1.shape[1]:
        raise ValueError("contact1 must be a square matrix")
    v = c1.shape[0]
    iu, ju = np.triu_indices(v, 1)
    t = np.minimum(c1[iu, ju], c1[ju, iu])
    ok = np.isfinite(t)
    off = c1.copy()
    np.fill_diagonal(off, np.inf)
    first = off.min(axis=1)
    vertex = np.where(np.isfinite(first), first, 0.0)
    return flag_complex(v, np.stack([iu[ok], ju[ok]], axis=1), t[ok], max_dim=max_dim, vertex_values=vertex,
                        max_simplices=max_simplices)


def simplex_keys(simplices: np.ndarray, n_vertices: int) -> np.ndarray:
    """Exact integer key of each simplex row (combinatorial number system, colex order) or byte keys.

    For rows v_0 < ... < v_d the key sum_i C(v_i, i + 1) is a bijection onto [0, C(n, d + 1)). When
    C(n, d + 1) does not fit in int64, big-endian byte strings of the rows are returned instead
    (also exact; comparable with each other and searchable with numpy.searchsorted).
    """
    s = np.asarray(simplices, dtype=np.int64)
    d1 = s.shape[1]
    n = int(n_vertices)
    if n >= 1 and comb(n, d1) < 2**62 and comb(n, min(d1, n // 2)) < 2**62:
        # table[v, k] = C(v, k) by the hockey-stick identity C(v, k) = sum_{u<v} C(u, k-1): sums only, so no
        # intermediate product can overflow; every entry is at most max_k C(n, k) < 2^62 (checked above).
        table = np.zeros((n + 1, d1 + 1), dtype=np.int64)
        table[:, 0] = 1
        for k in range(1, d1 + 1):
            table[1:, k] = np.cumsum(table[:-1, k - 1])
        key = np.zeros(s.shape[0], dtype=np.int64)
        for i in range(d1):
            key += table[s[:, i], i + 1]
        return key
    big = np.ascontiguousarray(s.astype(">i8"))
    return big.view(np.dtype((np.void, 8 * d1))).reshape(-1)


__all__ = [
    "METRIC_P", "FilteredComplex", "contact_complex", "enclosing_radius", "flag_complex", "maxmin_landmarks",
    "rips_complex", "rips_from_distances", "simplex_keys", "weight_values", "weighted_clique_complex",
]
