"""Multiplex measures over the relation planes: edge overlap, participation, interlayer degree
correlation, von Neumann entropy and structural reducibility.

The planes are the CVG-AE relation planes (AS-01 declared rules: connectivity, services, identity,
remote_admin, name_resolution, ot_control). Every layer is taken as an undirected simple graph on the
common node set.

Edge overlap and participation (Battiston, Nicosia and Latora, Phys. Rev. E 89:032804, 2014)
----------------------------------------------------------------------------------------
With o_ij = sum_a a_ij^[a] the number of layers joining i and j, the global overlap is
    O = sum_{i<j} o_ij / (M sum_{i<j} [o_ij > 0])   in [1/M, 1].
Pairwise, the Jaccard index |E_a n E_b| / |E_a u E_b| of the layers' edge sets. With k_i^[a] the degree
of i in layer a and o_i = sum_a k_i^[a], the participation coefficient is
    P_i = M / (M - 1) [1 - sum_a (k_i^[a] / o_i)^2]   (0: one layer only, 1: evenly spread).
Interlayer degree correlation: Spearman's rho of the node degrees of two layers (Nicosia and Latora,
Phys. Rev. E 92:032805, 2015).

Structural reducibility (De Domenico, Nicosia, Arenas and Latora, Nature Communications 6:6864, 2015)
-------------------------------------------------------------------------------------------------
Each layer is a density matrix rho = L / tr(L), L = D - A (Braunstein, Ghosh and Severini, Annals of
Combinatorics 10:291-317, 2006), with von Neumann entropy h = -sum_i lambda_i log2 lambda_i over the
eigenvalues of rho (0 log 0 = 0). The quantum Jensen-Shannon divergence of two layers is
D_JS = h((rho_a + rho_b) / 2) - (h(rho_a) + h(rho_b)) / 2; its square root is a metric (Virosztek,
Advances in Mathematics 380:107595, 2021). Layers are clustered hierarchically on these distances
(scipy.cluster.hierarchy.linkage; "average" linkage by default, which is valid for any metric);
merging two clusters sums their adjacency matrices. After each merge the relative entropy
    q = 1 - H_bar / h_A,   H_bar = (1/X) sum of the entropies of the X current layers,
h_A the entropy of the aggregate of all layers, measures how distinguishable the reduced multiplex is
from its aggregate; the reduction with the largest q is reported. The linkage used in the original
study is not restated here (citation to verify); the method is configurable.
Entropies of large layers (more than `max_dense` nodes) use the stochastic Lanczos quadrature of
`spectral.lanczos_quadrature`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr

from nagahana.analytics.spatial.graph import TrafficGraph
from nagahana.analytics.spatial.spectral import lanczos_quadrature


def layer_adjacencies(graph: TrafficGraph) -> dict[str, sparse.csr_matrix]:
    """plane -> binary symmetric adjacency on the common node set."""
    return {p: graph.layer(p).simple_undirected() for p in graph.planes()}


def edge_overlap(layers: dict[str, sparse.csr_matrix]) -> tuple[float, pd.DataFrame]:
    """(global overlap O, pairwise layer table with shared edges and Jaccard index)."""
    names = list(layers)
    m = len(names)
    if m == 0:
        return float("nan"), pd.DataFrame(columns=["layer_a", "layer_b", "edges_a", "edges_b", "shared", "jaccard"])
    total = sum(sparse.triu(layers[p], 1) for p in names)
    total = sparse.csr_matrix(total)
    o_sum = float(total.sum())
    pairs = total.nnz
    overlap = o_sum / (m * pairs) if pairs else float("nan")
    rows = []
    ups = {p: sparse.triu(layers[p], 1).tocsr() for p in names}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            shared = int(ups[a].multiply(ups[b]).nnz)
            ea, eb = int(ups[a].nnz), int(ups[b].nnz)
            union = ea + eb - shared
            rows.append({"layer_a": a, "layer_b": b, "edges_a": ea, "edges_b": eb, "shared": shared,
                         "jaccard": shared / union if union else float("nan")})
    return overlap, pd.DataFrame(rows, columns=["layer_a", "layer_b", "edges_a", "edges_b", "shared", "jaccard"])


def participation(layers: dict[str, sparse.csr_matrix]) -> tuple[np.ndarray, np.ndarray]:
    """(participation coefficient [n], overlapping degree [n]); nodes in no layer get NaN."""
    names = list(layers)
    m = len(names)
    deg = np.stack([np.asarray(layers[p].sum(axis=1)).ravel() for p in names], axis=1) if m else np.zeros((0, 0))
    o = deg.sum(axis=1) if m else np.zeros(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        share = deg / o[:, None]
        p = (m / (m - 1)) * (1.0 - (share ** 2).sum(axis=1)) if m > 1 else np.zeros(o.size)
    p = np.where(o > 0, p, np.nan)
    return p, o


def interlayer_degree_correlation(layers: dict[str, sparse.csr_matrix]) -> pd.DataFrame:
    """Spearman's rho of node degrees for every pair of layers."""
    names = list(layers)
    deg = {p: np.asarray(layers[p].sum(axis=1)).ravel() for p in names}
    rows = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            active = (deg[a] > 0) | (deg[b] > 0)
            if active.sum() < 3 or deg[a][active].std() == 0 or deg[b][active].std() == 0:
                rho = float("nan")
            else:
                rho = float(spearmanr(deg[a][active], deg[b][active]).statistic)
            rows.append({"layer_a": a, "layer_b": b, "active_nodes": int(active.sum()), "spearman": rho})
    return pd.DataFrame(rows, columns=["layer_a", "layer_b", "active_nodes", "spearman"])


def von_neumann_entropy(adj: sparse.spmatrix, *, max_dense: int = 4_000, vectors: int = 32, steps: int = 48,
                        seed: int = 0) -> float:
    """h(rho) in bits with rho = L / tr(L), L = D - A (module docstring)."""
    a = sparse.csr_matrix(adj, dtype=np.float64)
    a = ((a + a.T) * 0.5).tocsr()
    a.setdiag(0)
    a.eliminate_zeros()
    d = np.asarray(a.sum(axis=1)).ravel()
    tr = float(d.sum())
    if tr <= 0:
        return 0.0
    lap = sparse.diags(d) - a
    n = lap.shape[0]
    if n <= max_dense:
        lam = np.clip(np.linalg.eigvalsh(lap.toarray()) / tr, 0.0, None)
        lam = lam[lam > 0]
        return float(-(lam * np.log2(lam)).sum())
    nodes, weights = lanczos_quadrature(lap / tr, vectors=vectors, steps=steps, seed=seed)
    nodes = np.clip(nodes, 0.0, None)
    f = np.where(nodes > 0, -nodes * np.log2(np.where(nodes > 0, nodes, 1.0)), 0.0)
    return float(n * (weights * f).sum())


@dataclass(frozen=True)
class Reducibility:
    """Structural reducibility result (module docstring).

    distances: [M, M] square-root Jensen-Shannon distances. steps: one row per configuration (number of
    layers, q, the layer groups). best_layers: number of layers maximising q.
    """

    layers: tuple[str, ...]
    distances: np.ndarray
    steps: pd.DataFrame
    best_layers: int


def reducibility(layers: dict[str, sparse.csr_matrix], *, method: str = "average", max_dense: int = 4_000,
                 vectors: int = 32, steps: int = 48, seed: int = 0) -> Reducibility:
    """Structural reducibility of a multiplex (module docstring)."""
    names = tuple(p for p in layers if layers[p].nnz > 0)
    m = len(names)
    if m < 2:
        return Reducibility(names, np.zeros((m, m)), pd.DataFrame(columns=["layers", "q", "groups"]), m)

    def h(adj: sparse.spmatrix) -> float:
        return von_neumann_entropy(adj, max_dense=max_dense, vectors=vectors, steps=steps, seed=seed)

    def density(adj: sparse.spmatrix) -> sparse.csr_matrix:
        a = sparse.csr_matrix(adj, dtype=np.float64)
        d = np.asarray(a.sum(axis=1)).ravel()
        lap = sparse.diags(d) - a
        return sparse.csr_matrix(lap / max(float(d.sum()), 1e-300))

    def entropy_of_density(rho: sparse.csr_matrix) -> float:
        n = rho.shape[0]
        if n <= max_dense:
            lam = np.clip(np.linalg.eigvalsh(rho.toarray()), 0.0, None)
            lam = lam[lam > 0]
            return float(-(lam * np.log2(lam)).sum())
        nodes, weights = lanczos_quadrature(rho, vectors=vectors, steps=steps, seed=seed)
        nodes = np.clip(nodes, 0.0, None)
        f = np.where(nodes > 0, -nodes * np.log2(np.where(nodes > 0, nodes, 1.0)), 0.0)
        return float(n * (weights * f).sum())

    rho = {p: density(layers[p]) for p in names}
    ent = {p: entropy_of_density(rho[p]) for p in names}
    dist = np.zeros((m, m))
    for i, a in enumerate(names):
        for j in range(i + 1, m):
            b = names[j]
            js = entropy_of_density((rho[a] + rho[b]) * 0.5) - 0.5 * (ent[a] + ent[b])
            dist[i, j] = dist[j, i] = float(np.sqrt(max(js, 0.0)))
    z = linkage(squareform(dist, checks=False), method=method)
    aggregate = sum(layers[p] for p in names)
    h_a = h(aggregate)
    groups: dict[int, tuple[str, ...]] = {i: (names[i],) for i in range(m)}
    adjs: dict[int, sparse.spmatrix] = {i: layers[names[i]] for i in range(m)}
    rows = [{"layers": m, "q": 1.0 - np.mean([ent[p] for p in names]) / h_a if h_a > 0 else float("nan"),
             "groups": [list(g) for g in groups.values()]}]
    for k, (ia, ib, _d, _cnt) in enumerate(z):
        ia, ib = int(ia), int(ib)
        new = m + k
        groups[new] = groups.pop(ia) + groups.pop(ib)
        adjs[new] = adjs.pop(ia) + adjs.pop(ib)
        hbar = float(np.mean([h(x) for x in adjs.values()]))
        rows.append({"layers": len(groups), "q": 1.0 - hbar / h_a if h_a > 0 else float("nan"),
                     "groups": [list(g) for g in groups.values()]})
    frame = pd.DataFrame(rows)
    best = int(frame.loc[frame["q"].astype(float).idxmax(), "layers"]) if frame["q"].notna().any() else m
    return Reducibility(names, dist, frame, best)


__all__ = ["Reducibility", "edge_overlap", "interlayer_degree_correlation", "layer_adjacencies", "participation",
           "reducibility", "von_neumann_entropy"]
