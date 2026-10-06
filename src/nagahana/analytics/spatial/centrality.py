"""Centralities of traffic graphs: degree distributions, betweenness, PageRank, eigenvector, k-core.

Betweenness (Brandes, Journal of Mathematical Sociology 25(2):163-177, 2001)
------------------------------------------------------------------------
C_B(v) = sum_{s != v != t} sigma_st(v) / sigma_st, sigma_st the number of shortest s-t paths. For each
source s Brandes accumulates dependencies delta_s(v) = sum_{w : v in P_s(w)} (sigma_sv / sigma_sw)(1 + delta_s(w))
over the shortest-path DAG. Here the DAG of a batch of sources is handled algebraically: distances come
from scipy.sparse.csgraph.shortest_path (Dijkstra, or breadth-first search when unweighted), an edge
(u, v) of length w is tight for s when d(s, u) + w = d(s, v) (relative tolerance 1e-12 for weighted
graphs), and with T_s the tight-edge matrix
    sigma_s = e_s + T_s^T sigma_s                      (path counts, fixed point reached after L + 1 sweeps)
    delta_s(u) = sum_{(u, v) tight} (sigma_s(u) / sigma_s(v)) (1 + delta_s(v))   (same number of sweeps)
where L is the largest number of edges on a shortest path. Every sweep is one dense-batch times sparse
product, so a batch of B sources costs O(L B m). Undirected graphs count each pair twice, so their
values are halved. Normalisation (as Brandes): 1 / ((n - 1)(n - 2)) directed, 2 / ((n - 1)(n - 2))
undirected. With `sources` = k < n, k seeded random sources are used and the sums scaled by n / k, an
unbiased estimate (Brandes and Pich, Int. J. Bifurcation and Chaos 17(7):2303-2318, 2007). Edge
weights are lengths (a larger weight is a longer step); pass `weighted=False` for hop counts.

PageRank (Brin and Page, Computer Networks 30:107-117, 1998)
----------------------------------------------------------
x = alpha (x P + (sum_{dangling} x_d) v) + (1 - alpha) v with P = D_out^-1 W and v uniform, iterated
until ||x_{k+1} - x_k||_1 < tol. The tests compare it with the exact linear-system solution.

Eigenvector centrality (Bonacich, Journal of Mathematical Sociology 2(1):113-120, 1972)
------------------------------------------------------------------------------------
The Perron vector of A (in-links of a directed graph: x = A^T x), by power iteration on I + A, which
has the same eigenvectors and no oscillation on bipartite graphs; normalised to unit Euclidean norm.
On a disconnected graph the vector concentrates on the component with the largest eigenvalue.

k-core (Seidman, Social Networks 5(3):269-287, 1983; Batagelj and Zaversnik, arXiv:cs/0310049, 2003)
------------------------------------------------------------------------------------------------
core(v) = the largest k such that v belongs to a subgraph of minimum degree k. Peeling: for k = 0, 1,
... remove every node of current degree <= k, repeatedly, and give it core number k; degrees are
updated by one sparse product per round. Equal to the sequential bucket algorithm (tested).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import shortest_path

from nagahana.analytics import robust


@dataclass(frozen=True)
class DegreeStats:
    """Degree distribution of one degree sequence: summary, tail index and CCDF (k, P(K >= k))."""

    summary: dict[str, float]
    ccdf_k: np.ndarray
    ccdf_p: np.ndarray
    tail_alpha: float
    tail_k_star: int


def degree_stats(degrees: np.ndarray, *, tail_k_min: int = 10) -> DegreeStats:
    """Summary, complementary CDF and Hill tail index of a degree sequence."""
    d = np.asarray(degrees, dtype=np.float64)
    summ = robust.robust_summary(d)
    vals, counts = np.unique(d, return_counts=True)
    ccdf = np.cumsum(counts[::-1])[::-1] / max(d.size, 1)
    ti = robust.tail_index(d, k_min=tail_k_min)
    return DegreeStats(summary=summ, ccdf_k=vals, ccdf_p=ccdf, tail_alpha=ti.alpha_star, tail_k_star=ti.k_star)


def _edge_arrays(adj: sparse.csr_matrix) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coo = adj.tocoo()
    return coo.row.astype(np.int64), coo.col.astype(np.int64), coo.data.astype(np.float64)


def betweenness(
    adj: sparse.spmatrix,
    *,
    weighted: bool = False,
    undirected: bool = False,
    sources: int | None = None,
    batch: int = 64,
    normalized: bool = True,
    seed: int = 0,
) -> np.ndarray:
    """Betweenness centrality of every node (module docstring).

    adj: [n, n] adjacency (for an undirected graph, the symmetric matrix and `undirected=True`).
    """
    a = sparse.csr_matrix(adj, dtype=np.float64)
    a.setdiag(0)
    a.eliminate_zeros()
    n = a.shape[0]
    if n < 3:
        return np.zeros(n)
    u, v, w = _edge_arrays(a)
    if not weighted:
        w = np.ones_like(w)
    elif (w <= 0).any():
        raise ValueError("weighted betweenness needs positive edge lengths")
    m = u.size
    to_v = sparse.csr_matrix((np.ones(m), (np.arange(m), v)), shape=(m, n))   # edge -> head
    to_u = sparse.csr_matrix((np.ones(m), (np.arange(m), u)), shape=(m, n))   # edge -> tail
    if sources is not None and sources < n:
        rng = np.random.default_rng(seed)
        src_all = np.sort(rng.choice(n, size=int(sources), replace=False))
        scale = n / float(sources)
    else:
        src_all = np.arange(n)
        scale = 1.0
    bc = np.zeros(n)
    for b0 in range(0, src_all.size, batch):
        srcs = src_all[b0: b0 + batch]
        bsz = srcs.size
        dist = shortest_path(a, method="D", directed=True, unweighted=not weighted, indices=srcs)   # [B, n]
        du, dv = dist[:, u], dist[:, v]                                  # [B, m]
        finite = np.isfinite(du)
        if weighted:
            with np.errstate(invalid="ignore"):                          # inf - inf where the tail is unreachable
                tol = 1e-12 * np.maximum(1.0, np.abs(dv))
                tight = finite & (np.abs(du + w[None, :] - dv) <= tol)
        else:
            tight = finite & (du + 1.0 == dv)
        tight_f = tight.astype(np.float64)
        eye = np.zeros((bsz, n))
        eye[np.arange(bsz), srcs] = 1.0
        sigma = eye.copy()
        for _ in range(n):                                               # fixed point after L + 1 sweeps
            nxt = eye + (to_v.T @ (tight_f * sigma[:, u]).T).T
            if np.array_equal(nxt, sigma):
                break
            sigma = nxt
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(tight, sigma[:, u] / sigma[:, v], 0.0)      # sigma_s(u) / sigma_s(v) on tight edges
        delta = np.zeros((bsz, n))
        for _ in range(n):
            nxt = (to_u.T @ (ratio * (1.0 + delta[:, v])).T).T
            if np.array_equal(nxt, delta):
                break
            delta = nxt
        delta[np.arange(bsz), srcs] = 0.0                                # a source is not between itself and others
        bc += delta.sum(axis=0)
    bc *= scale
    if undirected:
        bc /= 2.0
    if normalized:
        bc *= (2.0 if undirected else 1.0) / ((n - 1) * (n - 2))
    return bc


def pagerank(
    adj: sparse.spmatrix,
    *,
    alpha: float = 0.85,
    tol: float = 1e-12,
    max_iter: int = 10_000,
    weighted: bool = True,
) -> tuple[np.ndarray, int]:
    """(PageRank vector, iterations) by power iteration (module docstring)."""
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    a = sparse.csr_matrix(adj, dtype=np.float64)
    n = a.shape[0]
    if n == 0:
        return np.zeros(0), 0
    if not weighted:
        a.data[:] = 1.0
    out = np.asarray(a.sum(axis=1)).ravel()
    dangling = out == 0
    inv = np.where(dangling, 0.0, 1.0 / np.where(dangling, 1.0, out))
    p = sparse.diags(inv) @ a                                            # row-stochastic on non-dangling rows
    v = np.full(n, 1.0 / n)
    x = v.copy()
    for it in range(1, max_iter + 1):
        nxt = alpha * (p.T @ x + x[dangling].sum() * v) + (1.0 - alpha) * v
        if np.abs(nxt - x).sum() < tol:
            return nxt / nxt.sum(), it
        x = nxt
    return x / x.sum(), max_iter


def eigenvector_centrality(adj: sparse.spmatrix, *, tol: float = 1e-12, max_iter: int = 10_000,
                           directed: bool = False) -> tuple[np.ndarray, int]:
    """(eigenvector centrality, iterations) by power iteration on I + A (module docstring)."""
    a = sparse.csr_matrix(adj, dtype=np.float64)
    n = a.shape[0]
    if n == 0:
        return np.zeros(0), 0
    op = a.T.tocsr() if directed else a
    x = np.full(n, 1.0 / np.sqrt(n))
    for it in range(1, max_iter + 1):
        nxt = x + op @ x
        norm = np.linalg.norm(nxt)
        if norm == 0:
            return np.zeros(n), it
        nxt /= norm
        if np.abs(nxt - x).sum() < n * tol:
            return nxt, it
        x = nxt
    return x, max_iter


def core_numbers(adj: sparse.spmatrix) -> np.ndarray:
    """k-core number of every node of the simple undirected graph of `adj` (module docstring)."""
    a = sparse.csr_matrix(adj, dtype=np.float64)
    a = ((a + a.T) > 0).astype(np.float64).tocsr()
    a.setdiag(0)
    a.eliminate_zeros()
    n = a.shape[0]
    deg = np.asarray(a.sum(axis=1)).ravel()
    alive = np.ones(n, dtype=bool)
    core = np.zeros(n, dtype=np.int64)
    k = 0
    while alive.any():
        while True:
            rem = alive & (deg <= k)
            if not rem.any():
                break
            core[rem] = k
            alive[rem] = False
            deg = deg - a @ rem.astype(np.float64)
        k += 1
    return core


__all__ = ["DegreeStats", "betweenness", "core_numbers", "degree_stats", "eigenvector_centrality", "pagerank"]
