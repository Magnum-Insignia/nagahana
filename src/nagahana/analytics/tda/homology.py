"""Persistent homology over Z/2 of a filtered complex: union-find for H_0, matrix reduction for H_k.

Filtration order
----------------
Simplices are totally ordered by (value, dimension, vertex tuple in lexicographic order). Faces
precede cofaces because values are monotone and dimension breaks ties, so this is a valid filtration
order; the lexicographic tie-break makes every output deterministic. The diagram does not depend on
how ties are broken (the persistence module is a function of the filtration only); the pairing of
simplices does, and every step below uses this one order.

H_0 by union-find
-----------------
Process edges in filtration order; an edge joining two components kills the younger one (larger
(value, rank) of its oldest vertex: the elder rule) at the edge's value. The merging edges are the
spanning forest that Kruskal's algorithm builds in this order; it is obtained in compiled code
(scipy.sparse.csgraph.minimum_spanning_tree with the filtration rank + 1 as the weight, so the forest
is unique and equals Kruskal's), and the elder-rule pairing runs over its |V| - 1 edges only. These
edges are exactly the negative edges, which matters for the clearing below.

H_k, k >= 1, by boundary-matrix reduction (Edelsbrunner, Letscher and Zomorodian, Discrete Comput.
Geom. 28:511-533, 2002; Zomorodian and Carlsson, Discrete Comput. Geom. 33:249-274, 2005)
-------------------------------------------------------------------------------------------------
Column j of the boundary matrix D lists the facets of simplex j (global ranks); low(j) is its largest
entry. Columns are reduced left to right by adding earlier columns with the same low (XOR over Z/2)
until the low is unique or the column is zero; a pair (low(j), j) is the bar [f(low(j)), f(j)).
Twist / clearing (Chen and Kerber, EuroCG 2011): dimensions are reduced from the highest down; when
column j of dimension d gets low i, simplex i is positive, so column i (dimension d - 1) would reduce
to zero and is skipped. Dimension 1 columns are never reduced: the union-find already says which edges
are negative.

Cohomology option (de Silva, Morozov and Vejdemo-Johansson, Inverse Problems 27:124003, 2011)
------------------------------------------------------------------------------------------
Reducing the anti-transposed matrix gives the same pairs: dimension by dimension upwards, the
coboundary column of every k-simplex, in decreasing filtration order, is reduced with pivot = its
earliest coface. A pivot pair (sigma, tau) is the bar [f(sigma), f(tau)); tau's own coboundary column
would reduce to zero, so it is skipped in dimension k + 1 (clearing in cohomology). The negative
edges of the union-find are cleared in dimension 1. For Vietoris-Rips filtrations this direction
usually reduces far fewer columns (Bauer 2021).

Apparent pairs (Bauer, Journal of Applied and Computational Topology 5:391-423, 2021)
---------------------------------------------------------------------------------
(sigma, tau) is apparent when sigma is the youngest facet of tau and tau the oldest coface of sigma.
Then column tau of D has final low sigma (no earlier column can contain sigma) and the coboundary
column of sigma has final pivot tau, so the pair is recorded without reduction and its column is kept
unreduced for later additions. Most zero-persistence pairs of a Rips filtration are apparent.

Essential classes and truncation
--------------------------------
A k-simplex that is positive and never paired starts an essential class (death +inf). In a truncated
filtration (`max_value` < +inf) "essential" means alive at the truncation value.

Representatives
---------------
With algorithm="homology", the reduced column R_j of a finite pair (i, j) is a cycle born at i and
killed at j; its simplices are returned as the representative of the bar (H_1: a closed edge path).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

from nagahana.analytics.tda.complex import FilteredComplex, simplex_keys
from nagahana.analytics.tda.diagrams import Diagram
from nagahana.core.errors import InvariantViolation


@dataclass(frozen=True)
class Persistence:
    """Result of a persistence computation.

    diagrams: one `Diagram` per dimension 0 ... max_dim. pairs: per dimension int64 [m, 2] global ranks
    (birth simplex, death simplex or -1), including zero-length pairs. stats: counts of reduced
    columns, apparent pairs, cleared columns and zero-length pairs.
    """

    diagrams: tuple[Diagram, ...]
    pairs: tuple[np.ndarray, ...]
    stats: dict[str, int]
    algorithm: str


class _Indexed:
    """Global filtration order of a complex, with boundary and coboundary lookups in global ranks."""

    def __init__(self, cx: FilteredComplex, top: int) -> None:
        self.simplices: list[np.ndarray] = []
        self.values: list[np.ndarray] = []
        for d in range(top + 1):
            s, v = cx.simplices[d], cx.values[d]
            # Within a dimension: by value, then lexicographic vertex tuple.
            order = np.lexsort(tuple(s[:, c] for c in range(s.shape[1] - 1, -1, -1)) + (v,))
            self.simplices.append(s[order])
            self.values.append(v[order])
        sizes = [s.shape[0] for s in self.simplices]
        val_all = np.concatenate(self.values)
        dim_all = np.concatenate([np.full(m, d, dtype=np.int64) for d, m in enumerate(sizes)])
        loc_all = np.concatenate([np.arange(m, dtype=np.int64) for m in sizes])
        order = np.lexsort((loc_all, dim_all, val_all))                 # global filtration order
        self.n = int(val_all.size)
        rank = np.empty(self.n, dtype=np.int64)
        rank[order] = np.arange(self.n)
        offs = np.cumsum([0] + sizes)
        self.grank = [rank[offs[d]: offs[d + 1]] for d in range(top + 1)]   # global rank of each local simplex
        self.g_dim = dim_all[order]
        self.g_loc = loc_all[order]
        self.g_val = val_all[order]
        self.n_vertices = cx.n_vertices
        self.facets: dict[int, np.ndarray] = {}
        for d in range(1, top + 1):
            self.facets[d] = self._facets(d)
        self.cof_ptr: dict[int, np.ndarray] = {}
        self.cof_idx: dict[int, np.ndarray] = {}
        for d in range(0, top):
            f = self.facets[d + 1]                                     # [m_{d+1}, d+2] global ranks
            m_d = self.simplices[d].shape[0]
            if f.size == 0:
                self.cof_ptr[d] = np.zeros(m_d + 1, dtype=np.int64)
                self.cof_idx[d] = np.zeros(0, dtype=np.int64)
                continue
            face_loc = self.g_loc[f.ravel()]                             # local index of each facet
            coface = np.repeat(self.grank[d + 1], f.shape[1])
            o = np.lexsort((coface, face_loc))
            self.cof_idx[d] = coface[o]
            self.cof_ptr[d] = np.searchsorted(face_loc[o], np.arange(m_d + 1))

    def _facets(self, d: int) -> np.ndarray:
        """Global ranks of the facets of every d-simplex, each row ascending: int64 [m_d, d + 1]."""
        s = self.simplices[d]
        lower = self.simplices[d - 1]
        if s.shape[0] == 0:
            return np.zeros((0, d + 1), dtype=np.int64)
        keys = simplex_keys(lower, self.n_vertices)
        order = np.argsort(keys, kind="stable")
        sk = keys[order]
        out = np.empty((s.shape[0], d + 1), dtype=np.int64)
        for drop in range(d + 1):
            face = np.delete(s, drop, axis=1)
            fk = simplex_keys(face, self.n_vertices)
            pos = np.searchsorted(sk, fk)
            pos_c = np.minimum(pos, sk.size - 1)
            if (pos >= sk.size).any() or not np.all(sk[pos_c] == fk):
                raise InvariantViolation("a facet is missing: the complex is not closed under faces")
            out[:, drop] = self.grank[d - 1][order[pos_c]]
        out.sort(axis=1)
        return out

    def coboundary(self, d: int, loc: int) -> np.ndarray:
        """Global ranks of the cofaces of local d-simplex `loc`, ascending."""
        p = self.cof_ptr[d]
        return self.cof_idx[d][p[loc]: p[loc + 1]]


def _h0(ix: _Indexed) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(finite pairs [m, 2], essential vertex ranks, negative-edge mask over local edges) by union-find."""
    verts = ix.simplices[0][:, 0]
    m0 = verts.size
    vmap = np.full(ix.n_vertices, -1, dtype=np.int64)
    vmap[verts] = np.arange(m0)
    edges = ix.simplices[1] if len(ix.simplices) > 1 else np.zeros((0, 2), dtype=np.int64)
    m1 = edges.shape[0]
    negative = np.zeros(m1, dtype=bool)
    if m1 == 0:
        return np.zeros((0, 2), dtype=np.int64), ix.grank[0].copy(), negative
    u, v = vmap[edges[:, 0]], vmap[edges[:, 1]]
    erank = ix.grank[1]
    # Unique weights (rank + 1) make the minimum spanning forest equal Kruskal's in filtration order.
    g = coo_matrix((erank.astype(np.float64) + 1.0, (u, v)), shape=(m0, m0)).tocsr()
    mst = minimum_spanning_tree(g).tocoo()
    tree_rank = np.rint(mst.data).astype(np.int64) - 1
    loc_of_rank = np.full(ix.n, -1, dtype=np.int64)
    loc_of_rank[erank] = np.arange(m1)
    tree_loc = loc_of_rank[tree_rank]
    negative[tree_loc] = True
    order = np.argsort(tree_rank, kind="stable")
    parent = list(range(m0))
    vrank = ix.grank[0].tolist()

    def find(a: int) -> int:
        root = a
        while parent[root] != root:
            root = parent[root]
        while parent[a] != root:                                        # path compression
            parent[a], a = root, parent[a]
        return root

    pairs: list[tuple[int, int]] = []
    for k in order.tolist():
        e = int(tree_loc[k])
        ra, rb = find(int(u[e])), find(int(v[e]))
        if ra == rb:
            raise InvariantViolation("spanning-forest edge inside one component")
        elder, younger = (ra, rb) if vrank[ra] < vrank[rb] else (rb, ra)
        pairs.append((vrank[younger], int(erank[e])))
        parent[younger] = elder
    roots = np.array(sorted({find(a) for a in range(m0)}), dtype=np.int64)
    return np.array(pairs, dtype=np.int64).reshape(-1, 2), ix.grank[0][roots], negative


class _Columns:
    """Reduced columns as Python sets (symmetric difference in C), with lazily materialised apparent columns.

    Columns of a reduction are short (a few to a few hundred entries), where hash-set XOR beats array
    merging; `get` materialises an apparent column (kept unreduced, see the module docstring) on first use.
    """

    def __init__(self) -> None:
        self.cols: dict[int, set[int]] = {}
        self.lazy: dict[int, np.ndarray] = {}

    def put(self, key: int, col: set[int]) -> None:
        self.cols[key] = col

    def put_lazy(self, key: int, col: np.ndarray) -> None:
        self.lazy[key] = col

    def get(self, key: int) -> set[int]:
        c = self.cols.get(key)
        if c is None:
            c = set(self.lazy.pop(key).tolist())
            self.cols[key] = c
        return c


def persistence(
    cx: FilteredComplex,
    *,
    max_dim: int,
    algorithm: str = "cohomology",
    clearing: bool = True,
    apparent_pairs: bool = True,
    representatives: bool = False,
) -> Persistence:
    """Persistence diagrams of dimensions 0 ... max_dim (module docstring)."""
    if algorithm not in ("cohomology", "homology"):
        raise ValueError("algorithm must be 'cohomology' or 'homology'")
    if representatives and algorithm != "homology":
        raise ValueError("representatives are produced by algorithm='homology'")
    exact = cx.dimension if cx.exact_through is None else cx.exact_through
    if max_dim > exact:
        raise ValueError(f"the complex is exact through dimension {exact} only; build it with max_dim >= {max_dim}")
    top = min(max_dim + 1, cx.dimension)
    ix = _Indexed(cx, top)
    stats = {"columns_reduced": 0, "apparent_pairs": 0, "cleared": 0, "zero_length_pairs": 0}
    pairs_by_dim: list[list[tuple[int, int]]] = [[] for _ in range(max_dim + 1)]
    reps: dict[int, list[np.ndarray]] = {k: [] for k in range(max_dim + 1)}
    h0_pairs, h0_ess, negative_edges = _h0(ix)
    pairs_by_dim[0] = [(int(a), int(b)) for a, b in h0_pairs] + [(int(a), -1) for a in h0_ess]
    if max_dim >= 1 and top >= 1:
        if algorithm == "homology":
            _reduce_homology(ix, max_dim, top, negative_edges, clearing, apparent_pairs, representatives,
                             pairs_by_dim, reps, stats)
        else:
            _reduce_cohomology(ix, max_dim, top, negative_edges, clearing, apparent_pairs, pairs_by_dim, stats)
    diagrams = []
    pair_arrays = []
    for k in range(max_dim + 1):
        arr = np.array(pairs_by_dim[k], dtype=np.int64).reshape(-1, 2)
        pair_arrays.append(arr)
        b = ix.g_val[arr[:, 0]] if arr.size else np.zeros(0)
        d = np.where(arr[:, 1] >= 0, ix.g_val[np.maximum(arr[:, 1], 0)], np.inf) if arr.size else np.zeros(0)
        keep = d > b
        stats["zero_length_pairs"] += int((~keep).sum())
        rep_k = None
        if representatives and k >= 1:
            rep_list = reps[k]
            rep_k = tuple(r for r, kk in zip(rep_list, keep, strict=True) if kk)
        diagrams.append(Diagram(k, b[keep], d[keep], rep_k))
    return Persistence(tuple(diagrams), tuple(pair_arrays), stats, algorithm)


def _apparent(ix: _Indexed, d: int) -> tuple[np.ndarray, np.ndarray]:
    """Apparent pairs between dimension d - 1 and d: (sigma ranks, tau ranks)."""
    f = ix.facets[d]
    if f.size == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    youngest = f[:, -1]                                                  # rows ascending: last = youngest facet
    ptr, idx = ix.cof_ptr[d - 1], ix.cof_idx[d - 1]
    has = ptr[1:] > ptr[:-1]
    oldest = np.full(ptr.size - 1, -1, dtype=np.int64)
    oldest[has] = idx[ptr[:-1][has]]                                     # cofaces ascending: first = oldest
    tau = ix.grank[d]
    sig_loc = ix.g_loc[youngest]
    hit = oldest[sig_loc] == tau
    return youngest[hit], tau[hit]


def _reduce_homology(ix: _Indexed, max_dim: int, top: int, negative_edges: np.ndarray, clearing: bool,
                     apparent_pairs: bool, representatives: bool, pairs_by_dim: list[list[tuple[int, int]]],
                     reps: dict[int, list[np.ndarray]], stats: dict[str, int]) -> None:
    """Boundary reduction with the twist, dimensions top ... 2 (module docstring)."""
    pivot = [-1] * ix.n                                                  # low -> column
    reduced = _Columns()
    zero_cols: dict[int, set[int]] = {d: set() for d in range(top + 1)}  # dimension -> zero-reduced columns
    cleared = bytearray(ix.n)
    lows_in: dict[int, set[int]] = {d: set() for d in range(top + 1)}    # dimension -> simplices that are lows
    for d in range(top, 1, -1):
        app_sig, app_tau = _apparent(ix, d) if apparent_pairs else (np.zeros(0, dtype=np.int64),) * 2
        apparent = set(app_tau.tolist())
        for s_, t_ in zip(app_sig.tolist(), app_tau.tolist(), strict=True):
            pivot[s_] = t_
            if clearing:
                cleared[s_] = 1
            lows_in[d - 1].add(s_)
        stats["apparent_pairs"] += len(apparent)
        facets = ix.facets[d]
        for loc, tau in enumerate(ix.grank[d].tolist()):
            if tau in apparent:
                reduced.put_lazy(tau, facets[loc])
                if d - 1 <= max_dim:
                    pairs_by_dim[d - 1].append((int(facets[loc, -1]), tau))
                    if representatives:
                        reps[d - 1].append(_simplices_of(ix, facets[loc]))
                continue
            if clearing and cleared[tau]:
                stats["cleared"] += 1
                continue
            col = set(facets[loc].tolist())
            stats["columns_reduced"] += 1
            while col:
                low = max(col)
                i = pivot[low]
                if i < 0:
                    break
                col ^= reduced.get(i)
            if col:
                low = max(col)
                pivot[low] = tau
                reduced.put(tau, col)
                lows_in[d - 1].add(low)
                if clearing:
                    cleared[low] = 1
                if d - 1 <= max_dim:
                    pairs_by_dim[d - 1].append((low, tau))
                    if representatives:
                        reps[d - 1].append(_simplices_of(ix, np.array(sorted(col), dtype=np.int64)))
            else:
                zero_cols[d].add(tau)
    # Essential classes of dimension k (1 <= k <= max_dim): positive k-simplices that are never a low.
    for k in range(1, max_dim + 1):
        if k > top:
            break
        if k == 1:
            positive = set(ix.grank[1][~negative_edges].tolist())
        else:
            positive = zero_cols[k] | {int(t) for t in ix.grank[k].tolist() if cleared[t]}
            if k == top:                                                 # columns of the top dimension were reduced
                positive = zero_cols[k]
        for s in sorted(positive - lows_in[k]):
            pairs_by_dim[k].append((s, -1))
            if representatives:
                reps[k].append(np.zeros((0, k + 1), dtype=np.int64))


def _reduce_cohomology(ix: _Indexed, max_dim: int, top: int, negative_edges: np.ndarray, clearing: bool,
                       apparent_pairs: bool, pairs_by_dim: list[list[tuple[int, int]]], stats: dict[str, int]) -> None:
    """Coboundary reduction with clearing, dimensions 1 ... max_dim (module docstring)."""
    pivot = [-1] * ix.n                                                  # coface pivot -> column (simplex)
    reduced = _Columns()
    death = bytearray(ix.n)                                              # simplices that are deaths of a lower pair
    for e in ix.grank[1][negative_edges].tolist():                       # negative edges kill H_0 classes
        death[e] = 1
    for k in range(1, max_dim + 1):
        if k >= top:                                                     # no cofaces: every unpaired k-simplex is essential
            for loc in range(ix.simplices[k].shape[0] if k < len(ix.simplices) else 0):
                s = int(ix.grank[k][loc])
                if not death[s]:
                    pairs_by_dim[k].append((s, -1))
            continue
        apparent: dict[int, int] = {}
        if apparent_pairs:
            app_sig, app_tau = _apparent(ix, k + 1)
            apparent = dict(zip(app_sig.tolist(), app_tau.tolist(), strict=True))
            stats["apparent_pairs"] += len(apparent)
            for s_, t_ in apparent.items():
                pivot[t_] = s_
                death[t_] = 1
        ranks = ix.grank[k]
        ranks_l = ranks.tolist()
        ptr = ix.cof_ptr[k].tolist()
        idx = ix.cof_idx[k]
        for loc in np.argsort(-ranks, kind="stable").tolist():           # decreasing filtration order
            sigma = ranks_l[loc]
            if death[sigma] and clearing:
                stats["cleared"] += 1
                continue
            if sigma in apparent:
                reduced.put_lazy(sigma, idx[ptr[loc]: ptr[loc + 1]])
                pairs_by_dim[k].append((sigma, apparent[sigma]))
                continue
            col = set(idx[ptr[loc]: ptr[loc + 1]].tolist())
            stats["columns_reduced"] += 1
            while col:
                piv = min(col)
                i = pivot[piv]
                if i < 0:
                    break
                col ^= reduced.get(i)
            if col:
                piv = min(col)
                pivot[piv] = sigma
                reduced.put(sigma, col)
                death[piv] = 1
                pairs_by_dim[k].append((sigma, piv))
            elif not death[sigma]:
                pairs_by_dim[k].append((sigma, -1))


def _simplices_of(ix: _Indexed, col: np.ndarray) -> np.ndarray:
    """Vertex tuples [s, k + 1] of the simplices with global ranks `col`."""
    if col.size == 0:
        return np.zeros((0, 1), dtype=np.int64)
    d = int(ix.g_dim[col[0]])
    return ix.simplices[d][ix.g_loc[col]]


__all__ = ["Persistence", "persistence"]
