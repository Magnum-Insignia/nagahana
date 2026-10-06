"""Mixing and small-subgraph structure: assortativity, triangles, clustering, 3-node motifs.

Assortativity (Newman, Phys. Rev. Lett. 89:208701, 2002; Phys. Rev. E 67:026126, 2003)
-----------------------------------------------------------------------------------
Degree assortativity r is the Pearson correlation of the degrees at the two ends of an edge (both
orientations for undirected graphs; out-degree of the source against in-degree of the target, and
the other three combinations, for directed graphs). Categorical assortativity uses the mixing matrix
e (fraction of edge ends joining type i to type j), a = e 1, b = e^T 1:
    r = (tr e - sum_i a_i b_i) / (1 - sum_i a_i b_i).

Triangles and clustering
------------------------
Triangles of the simple undirected graph are enumerated with the degree ordering of the
compact-forward algorithm (Latapy, Theoretical Computer Science 407:458-473, 2008): edges point from
the lower to the higher (degree, id); a triangle is a pair (v, w) of forward neighbours of u that are
themselves adjacent, so a hub never expands its whole neighbourhood and the work is O(m^1.5) (forward
degrees are at most sqrt(2m)). Never forming A^2 matters for traffic graphs, where one scanner can
touch thousands of hosts. Global transitivity = 3T / W with W = sum_v C(k_v, 2) connected triples;
local clustering c_v = t_v / C(k_v, 2) (0 when k_v < 2).

Directed 3-node motifs: the triad census (Holland and Leinhardt, Sociological Methodology 7:1-45,
1976; motifs: Milo, Shen-Orr, Itzkovitz, Kashtan, Chklovskii and Alon, Science 298:824-827, 2002)
------------------------------------------------------------------------------------------------
The 16 isomorphism classes of directed graphs on three nodes are named by their numbers of mutual,
asymmetric and null dyads plus an orientation letter; the 64 arc configurations of an ordered triple
are mapped to them by canonicalisation under the six permutations, starting from one representative
per class (A, B, C = nodes 0, 1, 2):
    003 none; 012 A->B; 102 A<->B; 021D A<-B->C; 021U A->B<-C; 021C A->B->C; 111D A<->B<-C;
    111U A<->B->C; 030T A->B<-C, A->C; 030C A<-B<-C, A->C; 201 A<->B<->C; 120D A<-B->C, A<->C;
    120U A->B<-C, A<->C; 120C A->B->C, A<->C; 210 A->B<->C, A<->C; 300 all six arcs.
Counts without enumerating triples. With m_c, o_c, i_c the numbers of mutual, asymmetric-out and
asymmetric-in dyads at node c, the wedges (two non-null dyads at a centre c) of each kind are
    021D: C(o_c, 2)   021U: C(i_c, 2)   021C: o_c i_c   111D: m_c i_c   111U: m_c o_c   201: C(m_c, 2).
A wedge whose ends are adjacent lies in a triangle of the undirected skeleton; enumerating the
triangles gives the closed triads (030T ... 300) and the closed wedges, which are subtracted. A triad
with one non-null dyad (u, v) has a third node adjacent to neither: n - k_u - k_v + c_uv of them, with
k the skeleton degree and c_uv the triangles on (u, v); asymmetric dyads give 012, mutual ones 102.
003 = C(n, 3) minus the rest. The tests compare with brute force over all triples.

Significance
------------
Motif z-scores z = (N_obs - mean N_rand) / sd N_rand against degree-preserving randomisations by
directed edge switches (Maslov and Sneppen, Science 296:910-913, 2002): pairs of arcs (a -> b, c -> d)
become (a -> d, c -> b) unless that creates a self-loop or an existing arc; every node keeps its in-
and out-degree. Switches are proposed in rounds of disjoint pairs; a proposal that collides with
another one of the same round is rejected.
"""

from __future__ import annotations

from itertools import permutations

import numpy as np
from scipy import sparse
from scipy.special import comb

TRIAD_NAMES: tuple[str, ...] = ("003", "012", "102", "021D", "021U", "021C", "111D", "111U", "030T", "030C", "201",
                                "120D", "120U", "120C", "210", "300")
_REPRESENTATIVES: dict[str, tuple[tuple[int, int], ...]] = {
    "003": (), "012": ((0, 1),), "102": ((0, 1), (1, 0)), "021D": ((1, 0), (1, 2)), "021U": ((0, 1), (2, 1)),
    "021C": ((0, 1), (1, 2)), "111D": ((0, 1), (1, 0), (2, 1)), "111U": ((0, 1), (1, 0), (1, 2)),
    "030T": ((0, 1), (2, 1), (0, 2)), "030C": ((1, 0), (2, 1), (0, 2)), "201": ((0, 1), (1, 0), (1, 2), (2, 1)),
    "120D": ((1, 0), (1, 2), (0, 2), (2, 0)), "120U": ((0, 1), (2, 1), (0, 2), (2, 0)),
    "120C": ((0, 1), (1, 2), (0, 2), (2, 0)), "210": ((0, 1), (1, 2), (2, 1), (0, 2), (2, 0)),
    "300": ((0, 1), (1, 0), (0, 2), (2, 0), (1, 2), (2, 1)),
}
#: Bit of each arc in a 6-bit triad code.
_ARC_BITS: dict[tuple[int, int], int] = {(0, 1): 1, (1, 0): 2, (0, 2): 4, (2, 0): 8, (1, 2): 16, (2, 1): 32}


def _canonical(arcs: frozenset[tuple[int, int]]) -> int:
    """Smallest 6-bit code of an arc set over the six relabellings of its nodes."""
    best = 64
    for perm in permutations(range(3)):
        code = sum(_ARC_BITS[(perm[a], perm[b])] for a, b in arcs)
        best = min(best, code)
    return best


def _triad_table() -> np.ndarray:
    """int64 [64]: triad class index of every arc configuration (module docstring)."""
    by_canon = {_canonical(frozenset(arcs)): TRIAD_NAMES.index(name) for name, arcs in _REPRESENTATIVES.items()}
    if len(by_canon) != 16:
        raise RuntimeError("triad representatives are not pairwise non-isomorphic")
    table = np.empty(64, dtype=np.int64)
    for code in range(64):
        arcs = frozenset(arc for arc, bit in _ARC_BITS.items() if code & bit)
        table[code] = by_canon[_canonical(arcs)]
    return table


TRIAD_TABLE: np.ndarray = _triad_table()


def _skeleton(adj: sparse.spmatrix) -> sparse.csr_matrix:
    a = sparse.csr_matrix(adj, dtype=np.float64)
    s = ((a + a.T) > 0).astype(np.float64).tocsr()
    s.setdiag(0)
    s.eliminate_zeros()
    return s


def degree_assortativity(adj: sparse.spmatrix, *, directed: bool = False, mode: str = "out-in") -> float:
    """Degree assortativity (module docstring); mode for directed graphs: out-in, in-out, out-out, in-in."""
    a = sparse.csr_matrix(adj, dtype=np.float64)
    a.setdiag(0)
    a.eliminate_zeros()
    if not directed:
        s = _skeleton(a)
        coo = s.tocoo()
        deg = np.asarray(s.sum(axis=1)).ravel()
        x, y = deg[coo.row], deg[coo.col]
    else:
        b = (a > 0).astype(np.float64).tocsr()
        coo = b.tocoo()
        out_d = np.asarray(b.sum(axis=1)).ravel()
        in_d = np.asarray(b.sum(axis=0)).ravel()
        first, second = mode.split("-")
        pick = {"out": out_d, "in": in_d}
        x, y = pick[first][coo.row], pick[second][coo.col]
    if x.size < 2 or x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def attribute_assortativity(adj: sparse.spmatrix, labels: np.ndarray, *, directed: bool = False) -> float:
    """Categorical assortativity coefficient (module docstring)."""
    from nagahana.analytics.information import factorize

    a = sparse.csr_matrix(adj, dtype=np.float64)
    a.setdiag(0)
    a.eliminate_zeros()
    m = a if directed else _skeleton(a)
    coo = m.tocoo()
    c, k = factorize(np.asarray(labels))
    e = np.zeros((k, k))
    np.add.at(e, (c[coo.row], c[coo.col]), 1.0)
    total = e.sum()
    if total == 0:
        return float("nan")
    e /= total
    ab = float((e.sum(axis=1) * e.sum(axis=0)).sum())
    if ab >= 1.0:
        return float("nan")
    return float((np.trace(e) - ab) / (1.0 - ab))


def triangles(adj: sparse.spmatrix) -> np.ndarray:
    """Triangles of the undirected skeleton as int64 [T, 3] rows (each once), compact-forward order."""
    s = _skeleton(adj)
    n = s.shape[0]
    deg = np.asarray(s.sum(axis=1)).ravel()
    rank = np.empty(n, dtype=np.int64)
    rank[np.lexsort((np.arange(n), deg))] = np.arange(n)                 # order by (degree, id)
    coo = s.tocoo()
    fwd = rank[coo.row] < rank[coo.col]
    u, v = coo.row[fwd].astype(np.int64), coo.col[fwd].astype(np.int64)
    order = np.lexsort((rank[v], u))
    u, v = u[order], v[order]
    ptr = np.searchsorted(u, np.arange(n + 1))
    keys = np.sort(np.minimum(coo.row, coo.col).astype(np.int64) * n + np.maximum(coo.row, coo.col))
    dplus = np.diff(ptr)
    out = []
    # Pairs of forward neighbours (v_i, v_j), i < j, of each u; vectorised per forward-degree value.
    for d in np.unique(dplus[dplus >= 2]).tolist():
        us = np.flatnonzero(dplus == d)
        nb = v[ptr[us][:, None] + np.arange(d)[None, :]]                 # [U, d]
        iu, ju = np.triu_indices(d, 1)
        a_, b_ = nb[:, iu].reshape(-1), nb[:, ju].reshape(-1)
        owner = np.repeat(us, iu.size)
        k = np.minimum(a_, b_) * n + np.maximum(a_, b_)
        pos = np.minimum(np.searchsorted(keys, k), keys.size - 1)
        hit = keys[pos] == k
        out.append(np.stack([owner[hit], a_[hit], b_[hit]], axis=1))
    tri = np.concatenate(out) if out else np.zeros((0, 3), dtype=np.int64)
    return np.sort(tri, axis=1)


def clustering(adj: sparse.spmatrix) -> tuple[np.ndarray, np.ndarray, float, float]:
    """(triangles per node, local clustering, average clustering, transitivity) of the skeleton."""
    s = _skeleton(adj)
    n = s.shape[0]
    tri = triangles(s)
    t = np.bincount(tri.ravel(), minlength=n).astype(np.float64)
    k = np.asarray(s.sum(axis=1)).ravel()
    w = comb(k, 2)
    local = np.where(w > 0, t / np.where(w > 0, w, 1.0), 0.0)
    trans = float(3.0 * tri.shape[0] / w.sum()) if w.sum() > 0 else float("nan")
    return t, local, float(local.mean()) if n else float("nan"), trans


def triad_census(adj: sparse.spmatrix) -> dict[str, int]:
    """Counts of the 16 triad classes of a directed graph (module docstring)."""
    a = sparse.csr_matrix(adj, dtype=np.float64)
    a.setdiag(0)
    a.eliminate_zeros()
    b = (a > 0).astype(np.int8).tocsr()
    n = b.shape[0]
    bt = b.T.tocsr()
    mutual = b.multiply(bt).tocsr()
    m_c = np.asarray(mutual.sum(axis=1)).ravel().astype(np.int64)
    o_c = np.asarray(b.sum(axis=1)).ravel().astype(np.int64) - m_c
    i_c = np.asarray(b.sum(axis=0)).ravel().astype(np.int64) - m_c
    counts = dict.fromkeys(TRIAD_NAMES, 0)
    wedges = {"021D": comb(o_c, 2, exact=False).sum(), "021U": comb(i_c, 2, exact=False).sum(),
              "021C": float((o_c * i_c).sum()), "111D": float((m_c * i_c).sum()), "111U": float((m_c * o_c).sum()),
              "201": comb(m_c, 2, exact=False).sum()}
    s = _skeleton(b)
    tri = triangles(s)
    arc = sparse.csr_matrix(b, dtype=np.int8)
    keyset = arc.tocoo()
    arc_keys = np.sort(keyset.row.astype(np.int64) * n + keyset.col)

    def has(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        k = x * n + y
        pos = np.minimum(np.searchsorted(arc_keys, k), max(arc_keys.size - 1, 0))
        return (arc_keys[pos] == k) if arc_keys.size else np.zeros(k.shape, dtype=bool)

    closed_wedges = dict.fromkeys(wedges, 0)
    if tri.size:
        x, y, z = tri[:, 0], tri[:, 1], tri[:, 2]
        code = (has(x, y) * 1 + has(y, x) * 2 + has(x, z) * 4 + has(z, x) * 8 + has(y, z) * 16 + has(z, y) * 32)
        kinds = TRIAD_TABLE[code]
        for t_idx, cnt in zip(*np.unique(kinds, return_counts=True), strict=True):
            counts[TRIAD_NAMES[int(t_idx)]] += int(cnt)
        # Closed wedges: at each triangle corner, the kinds of its two dyads.
        for c_, p_, q_ in ((x, y, z), (y, x, z), (z, x, y)):
            def dyad(c: np.ndarray, o: np.ndarray) -> np.ndarray:        # 0 out, 1 in, 2 mutual (seen from c)
                fwd, back = has(c, o), has(o, c)
                return np.where(fwd & back, 2, np.where(fwd, 0, 1))
            d1, d2 = dyad(c_, p_), dyad(c_, q_)
            lo, hi = np.minimum(d1, d2), np.maximum(d1, d2)
            for (l_, h_), name in (((0, 0), "021D"), ((1, 1), "021U"), ((0, 1), "021C"), ((1, 2), "111D"),
                                   ((0, 2), "111U"), ((2, 2), "201")):
                closed_wedges[name] += int(np.sum((lo == l_) & (hi == h_)))
    for name in wedges:
        counts[name] += int(round(wedges[name])) - closed_wedges[name]
    # One non-null dyad: third nodes adjacent to neither end.
    k = np.asarray(s.sum(axis=1)).ravel().astype(np.int64)
    sk = sparse.triu(s, 1).tocoo()
    pu, pv = sk.row.astype(np.int64), sk.col.astype(np.int64)
    common = np.zeros(pu.size, dtype=np.int64)
    if tri.size:
        tri_edges = np.concatenate([tri[:, [0, 1]], tri[:, [0, 2]], tri[:, [1, 2]]])
        tk, tc = np.unique(tri_edges[:, 0] * n + tri_edges[:, 1], return_counts=True)
        pk = pu * n + pv
        pos = np.minimum(np.searchsorted(tk, pk), tk.size - 1)
        common = np.where(tk[pos] == pk, tc[pos], 0)
    third = n - k[pu] - k[pv] + common
    is_mut = has(pu, pv) & has(pv, pu)
    counts["102"] += int(third[is_mut].sum())
    counts["012"] += int(third[~is_mut].sum())
    counts["003"] = int(comb(n, 3, exact=True)) - sum(v for key, v in counts.items() if key != "003")
    return counts


def undirected_motifs(adj: sparse.spmatrix) -> dict[str, int]:
    """Connected 3-node subgraphs of the skeleton: open wedges and triangles."""
    s = _skeleton(adj)
    k = np.asarray(s.sum(axis=1)).ravel()
    t = triangles(s).shape[0]
    return {"wedge": int(round(comb(k, 2).sum())) - 3 * t, "triangle": int(t)}


def degree_preserving_switch(src: np.ndarray, dst: np.ndarray, n: int, *, swaps_per_edge: float = 10.0,
                             seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Directed edge switches keeping every in- and out-degree (module docstring)."""
    s = np.asarray(src, dtype=np.int64).copy()
    t = np.asarray(dst, dtype=np.int64).copy()
    m = s.size
    if m < 2:
        return s, t
    rng = np.random.default_rng(seed)
    target = swaps_per_edge * m
    done = 0
    for _ in range(int(np.ceil(4 * target / max(m // 2, 1))) + 10):
        if done >= target:
            break
        perm = rng.permutation(m)
        half = m // 2
        e1, e2 = perm[:half], perm[half: 2 * half]
        a, b, c, d = s[e1], t[e1], s[e2], t[e2]
        current = np.sort(s * n + t)
        k1, k2 = a * n + d, c * n + b
        ok = (a != d) & (c != b) & (a != c) & (b != d)

        def exists(k: np.ndarray, keys: np.ndarray = current) -> np.ndarray:
            pos = np.minimum(np.searchsorted(keys, k), m - 1)
            return keys[pos] == k

        ok &= ~exists(k1) & ~exists(k2)
        allk = np.concatenate([k1[ok], k2[ok]])
        uniq, cnt = np.unique(allk, return_counts=True)
        dup = uniq[cnt > 1]
        ok_idx = np.flatnonzero(ok)
        clash = np.isin(k1[ok_idx], dup) | np.isin(k2[ok_idx], dup)
        ok[ok_idx[clash]] = False
        t[e1[ok]] = d[ok]
        t[e2[ok]] = b[ok]
        done += int(ok.sum())
    return s, t


def motif_zscores(adj: sparse.spmatrix, *, samples: int, seed: int = 0) -> dict[str, tuple[int, float, float, float]]:
    """name -> (observed, random mean, random sd, z) of the triad census against edge-switch randomisations."""
    a = sparse.csr_matrix(adj, dtype=np.float64)
    a.setdiag(0)
    a.eliminate_zeros()
    coo = (a > 0).tocoo()
    n = a.shape[0]
    obs = triad_census(a)
    reps = []
    for r in range(samples):
        s2, t2 = degree_preserving_switch(coo.row, coo.col, n, seed=seed + r)
        g = sparse.csr_matrix((np.ones(s2.size), (s2, t2)), shape=(n, n))
        reps.append(triad_census(g))
    out = {}
    for name in TRIAD_NAMES:
        vals = np.array([rc[name] for rc in reps], dtype=np.float64)
        mu, sd = float(vals.mean()), float(vals.std(ddof=1)) if vals.size > 1 else float("nan")
        z = (obs[name] - mu) / sd if sd and np.isfinite(sd) and sd > 0 else float("nan")
        out[name] = (obs[name], mu, sd, float(z))
    return out


__all__ = [
    "TRIAD_NAMES", "TRIAD_TABLE", "attribute_assortativity", "clustering", "degree_assortativity",
    "degree_preserving_switch", "motif_zscores", "triad_census", "triangles", "undirected_motifs",
]
