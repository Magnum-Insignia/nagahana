"""Communities by Louvain modularity optimisation, with a connectivity repair; partition agreement.

Modularity (Newman and Girvan, Phys. Rev. E 69:026113, 2004; resolution gamma of Reichardt and
Bornholdt, Phys. Rev. E 74:016110, 2006)
----------------------------------------------------------------------------------------------
For a symmetric weighted adjacency A (self-loops on the diagonal allowed), 2m = sum_ij A_ij and
k_i = sum_j A_ij:
    Q = (1 / 2m) sum_ij [A_ij - gamma k_i k_j / 2m] [c_i = c_j]
      = (1 / 2m) sum_c [Sigma_in(c) - gamma Sigma_tot(c)^2 / 2m],
Sigma_in(c) = sum_{i, j in c} A_ij and Sigma_tot(c) = sum_{i in c} k_i.

Louvain (Blondel, Guillaume, Lambiotte and Lefebvre, J. Stat. Mech. P10008, 2008)
-----------------------------------------------------------------------------
Phase 1 (local moving): nodes are visited in a seeded random order; node i is taken out of its
community and put into the neighbouring community D with the largest gain
    Delta Q(i -> D) = (1 / m) [k_{i,D} - gamma Sigma_tot(D) k_i / 2m],
k_{i,D} the weight from i to D (self-loop excluded), staying when no gain beats its own community by
more than `tol`. Sweeps repeat until no node moves. Phase 2 (aggregation): communities become nodes,
A' = S^T A S with the membership matrix S, which keeps the self-loop convention above, so Q is
unchanged by aggregation. Levels repeat until a level moves nothing.

Connectivity repair
-------------------
Louvain can return communities that are internally disconnected (Traag, Waltman and van Eck,
Scientific Reports 9:5233, 2019). Each community is split into its connected components in the input
graph. With no edge between two parts A and B of a community, splitting changes Q by
+gamma 2 Sigma_tot(A) Sigma_tot(B) / (2m)^2 >= 0, so the repair never lowers modularity.

Partition agreement
-------------------
Adjusted Rand index (Hubert and Arabie, Journal of Classification 2:193-218, 1985) and normalised
mutual information with the arithmetic-mean normalisation, NMI = 2 I(X; Y) / (H(X) + H(Y)).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.special import comb

from nagahana.analytics.information import entropy_discrete, factorize, mutual_information_discrete


def modularity(adj: sparse.spmatrix, membership: np.ndarray, *, resolution: float = 1.0) -> float:
    """Q of a partition of a symmetric weighted graph (module docstring)."""
    a = sparse.csr_matrix(adj, dtype=np.float64)
    two_m = float(a.sum())
    if two_m <= 0:
        return float("nan")
    c, k = factorize(np.asarray(membership))
    s = sparse.csr_matrix((np.ones(c.size), (np.arange(c.size), c)), shape=(c.size, k))
    inner = (s.T @ a @ s).diagonal()                                     # Sigma_in per community
    tot = np.asarray(s.T @ np.asarray(a.sum(axis=1)).ravel()).ravel()    # Sigma_tot per community
    return float((inner - resolution * tot**2 / two_m).sum() / two_m)


@dataclass(frozen=True)
class Communities:
    """Louvain result: membership [n] (0 ... k - 1), modularity, number of communities, Q after each level."""

    membership: np.ndarray
    modularity: float
    n_communities: int
    level_modularity: tuple[float, ...] = field(default=())
    repaired_splits: int = 0


def _local_moving(a: sparse.csr_matrix, resolution: float, tol: float, max_sweeps: int,
                  rng: np.random.Generator) -> tuple[np.ndarray, bool]:
    """Phase 1 on graph `a`; returns (community of each node, whether any node moved)."""
    n = a.shape[0]
    k = np.asarray(a.sum(axis=1)).ravel()
    two_m = float(k.sum())
    comm = np.arange(n)
    tot = k.copy()
    indptr, indices, data = a.indptr, a.indices, a.data
    moved_any = False
    for _ in range(max_sweeps):
        moved = 0
        for i in rng.permutation(n).tolist():
            lo, hi = indptr[i], indptr[i + 1]
            nb, wt = indices[lo:hi], data[lo:hi]
            not_self = nb != i
            nb, wt = nb[not_self], wt[not_self]
            own = comm[i]
            tot[own] -= k[i]                                             # take i out of its community
            cands, inv = np.unique(comm[nb], return_inverse=True)
            kin = np.bincount(inv.reshape(-1), weights=wt, minlength=cands.size)
            gains = kin - resolution * tot[cands] * k[i] / two_m
            pos = np.searchsorted(cands, own)
            own_gain = (gains[pos] if pos < cands.size and cands[pos] == own
                        else -resolution * tot[own] * k[i] / two_m)
            best = own
            if gains.size:
                j = int(np.argmax(gains))
                if gains[j] > own_gain + tol:
                    best = int(cands[j])
            comm[i] = best
            tot[best] += k[i]
            if best != own:
                moved += 1
        if moved == 0:
            break
        moved_any = True
    _, comm = np.unique(comm, return_inverse=True)
    return comm.reshape(-1), moved_any


def louvain(
    adj: sparse.spmatrix,
    *,
    resolution: float = 1.0,
    tol: float = 1e-10,
    max_levels: int = 50,
    max_sweeps: int = 1_000,
    seed: int = 0,
    repair: bool = True,
) -> Communities:
    """Louvain communities of a symmetric weighted graph (module docstring)."""
    a0 = sparse.csr_matrix(adj, dtype=np.float64)
    if (abs(a0 - a0.T) > 1e-12 * max(1.0, float(abs(a0).max()) if a0.nnz else 1.0)).nnz:
        raise ValueError("louvain needs a symmetric adjacency (an undirected graph)")
    n = a0.shape[0]
    rng = np.random.default_rng(seed)
    membership = np.arange(n)
    a = a0.copy()
    levels: list[float] = []
    for _ in range(max_levels):
        comm, moved = _local_moving(a, resolution, tol, max_sweeps, rng)
        if not moved:
            break
        membership = comm[membership]
        s = sparse.csr_matrix((np.ones(comm.size), (np.arange(comm.size), comm)), shape=(comm.size, int(comm.max()) + 1))
        a = (s.T @ a @ s).tocsr()                                        # aggregation keeps Q (module docstring)
        levels.append(modularity(a0, membership, resolution=resolution))
    splits = 0
    if repair and n:
        out = np.empty(n, dtype=np.int64)
        nxt = 0
        for c in np.unique(membership).tolist():
            nodes = np.flatnonzero(membership == c)
            ncomp, lab = connected_components(a0[nodes][:, nodes], directed=False)
            out[nodes] = nxt + lab
            nxt += ncomp
            splits += ncomp - 1
        membership = out
    _, membership = np.unique(membership, return_inverse=True)
    membership = membership.reshape(-1)
    return Communities(membership=membership, modularity=modularity(a0, membership, resolution=resolution),
                       n_communities=int(membership.max()) + 1 if n else 0, level_modularity=tuple(levels),
                       repaired_splits=splits)


def adjusted_rand_index(x: np.ndarray, y: np.ndarray) -> float:
    """ARI of two partitions (module docstring)."""
    cx, kx = factorize(np.asarray(x))
    cy, ky = factorize(np.asarray(y))
    n = cx.size
    table = np.zeros((kx, ky))
    np.add.at(table, (cx, cy), 1.0)
    sum_comb = comb(table, 2).sum()
    a = comb(table.sum(axis=1), 2).sum()
    b = comb(table.sum(axis=0), 2).sum()
    expected = a * b / comb(n, 2) if n > 1 else 0.0
    maximum = 0.5 * (a + b)
    if maximum == expected:
        return 1.0
    return float((sum_comb - expected) / (maximum - expected))


def normalized_mutual_information(x: np.ndarray, y: np.ndarray) -> float:
    """NMI = 2 I / (H(X) + H(Y)) of two partitions (1 when both are trivial and equal)."""
    hx, hy = entropy_discrete(np.asarray(x)), entropy_discrete(np.asarray(y))
    if hx + hy == 0:
        return 1.0
    return float(2.0 * mutual_information_discrete(np.asarray(x), np.asarray(y)) / (hx + hy))


__all__ = ["Communities", "adjusted_rand_index", "louvain", "modularity", "normalized_mutual_information"]
