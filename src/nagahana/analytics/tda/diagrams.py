"""Persistence diagrams, their exact bottleneck and Wasserstein distances, and summary statistics.

Diagram
-------
A diagram of dimension k is the multiset of intervals [b, d) of H_k classes; d = +inf for a class
still alive at the end of the (possibly truncated) filtration. Points on the diagonal carry no
information and are never stored.

Ground metric
-------------
With the L-infinity ground norm (the standard one), ||p - q|| = max(|b - b'|, |d - d'|) and the
distance of p to the diagonal is (d - b) / 2; with the L2 norm, sqrt((b - b')^2 + (d - d')^2) and
(d - b) / sqrt(2). A matching may send any point to its diagonal projection.

Essential classes
-----------------
Points with d = +inf can only be matched to each other: the distance is +inf when two diagrams have
different numbers of them. Otherwise, in one dimension the sorted matching of their births is optimal
for every convex cost and for the maximum, so the essential part contributes max |b_i - b'_i|
(bottleneck) or sum |b_i - b'_i|^p (Wasserstein) over sorted births.

Bottleneck distance (exact)
---------------------------
d_B = min over matchings of the largest matched distance. The optimum is one of the candidate values
(a point-to-point distance or a point-to-diagonal distance), so binary search over the sorted
candidates with a feasibility test is exact (Efrat, Itai and Katz, Algorithmica 31:1-28, 2001;
Kerber, Morozov and Nigmetov, ACM J. Experimental Algorithmics 22, 2017). Feasibility at delta: the
points with diagonal distance > delta must be matched to points within delta. By the
Mendelsohn-Dulmage theorem (Canadian J. Math. 10:517-534, 1958), a matching covering both must-sets
exists if and only if one covers the must-set of the first diagram and another covers the must-set of
the second; each is a maximum bipartite matching (Hopcroft-Karp, scipy.sparse.csgraph.
maximum_bipartite_matching). Unmatched points go to the diagonal at cost <= delta.

p-Wasserstein distance (exact)
------------------------------
W_p = (min over matchings sum ||p - gamma(p)||^p)^(1/p). The assignment problem on the augmented
square matrix of size n1 + n2 is solved exactly (scipy.optimize.linear_sum_assignment):
    [ C(a_i, b_j)^p                   (row i: dist(a_i, diagonal)^p everywhere) ]
    [ (column j: dist(b_j, diagonal)^p everywhere)   0                         ]
Matching a point to any diagonal copy costs its own diagonal distance, because diagonal copies are
interchangeable at zero cost (Kerber, Morozov and Nigmetov 2017 use the same construction).

Summaries
---------
count, essential count, total persistence sum (d - b), maximum persistence, and the persistent
entropy E = -sum (l_i / L) log(l_i / L) of the finite bar lengths l_i, L = sum l_i (Chintakunta,
Gentimis, Gonzalez-Diaz, Jimenez and Krim, Pattern Recognition 48(2):391-401, 2015).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching


@dataclass(frozen=True)
class Diagram:
    """Persistence diagram of one homology dimension (module docstring).

    birth, death: float64 [m] (death +inf for essential classes). representatives: optional list of
    int64 arrays [s, k + 1] (the simplices of a representative cycle of each point), aligned with birth.
    """

    dim: int
    birth: np.ndarray
    death: np.ndarray
    representatives: tuple[np.ndarray, ...] | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        b = np.asarray(self.birth, dtype=np.float64).reshape(-1)
        d = np.asarray(self.death, dtype=np.float64).reshape(-1)
        if b.shape != d.shape:
            raise ValueError("birth and death must have the same length")
        if not np.isfinite(b).all() or np.isnan(d).any() or (d < b).any():
            raise ValueError("diagram points need finite births and deaths >= births")
        object.__setattr__(self, "birth", b)
        object.__setattr__(self, "death", d)

    @property
    def persistence(self) -> np.ndarray:
        """d - b per point (+inf for essential classes)."""
        return self.death - self.birth

    def __len__(self) -> int:
        return int(self.birth.size)

    def finite(self) -> tuple[np.ndarray, np.ndarray]:
        """(births, deaths) of the finite points."""
        m = np.isfinite(self.death)
        return self.birth[m], self.death[m]

    def essential(self) -> np.ndarray:
        """Births of the essential points."""
        return self.birth[~np.isfinite(self.death)]

    def filtered(self, min_persistence: float) -> Diagram:
        """Points with persistence > min_persistence (essential points are always kept)."""
        keep = self.persistence > min_persistence
        reps = None if self.representatives is None else tuple(r for r, k in zip(self.representatives, keep, strict=True) if k)
        return Diagram(self.dim, self.birth[keep], self.death[keep], reps)

    def capped(self, cap: float) -> Diagram:
        """Essential deaths replaced by `cap` (>= every birth); for vectorisations that need finite bars."""
        if (self.birth > cap).any():
            raise ValueError("cap must be at least every birth")
        return Diagram(self.dim, self.birth, np.where(np.isfinite(self.death), self.death, cap))


def _diag_distance(b: np.ndarray, d: np.ndarray, norm: str) -> np.ndarray:
    if norm == "inf":
        return (d - b) / 2.0
    if norm == "2":
        return (d - b) / np.sqrt(2.0)
    raise ValueError("ground norm must be 'inf' or '2'")


def _cross(b1: np.ndarray, d1: np.ndarray, b2: np.ndarray, d2: np.ndarray, norm: str) -> np.ndarray:
    db = np.abs(b1[:, None] - b2[None, :])
    dd = np.abs(d1[:, None] - d2[None, :])
    return np.maximum(db, dd) if norm == "inf" else np.sqrt(db * db + dd * dd)


def _essential_parts(x: Diagram, y: Diagram) -> tuple[np.ndarray, np.ndarray] | None:
    """Sorted essential births of both diagrams, or None when their counts differ (distance +inf)."""
    ex, ey = np.sort(x.essential()), np.sort(y.essential())
    if ex.size != ey.size:
        return None
    return ex, ey


def _covers(adj: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> bool:
    """True when the bipartite graph adj[rows][:, cols] has a matching covering every row."""
    if rows.size == 0:
        return True
    if cols.size == 0:
        return False
    sub = csr_matrix(adj[np.ix_(rows, cols)])
    match = maximum_bipartite_matching(sub, perm_type="column")
    return int((match >= 0).sum()) == rows.size


def bottleneck_distance(x: Diagram, y: Diagram, *, norm: str = "inf", max_pairs: int = 25_000_000) -> float:
    """Exact bottleneck distance (module docstring)."""
    ess = _essential_parts(x, y)
    if ess is None:
        return float("inf")
    d_ess = float(np.abs(ess[0] - ess[1]).max()) if ess[0].size else 0.0
    b1, d1 = x.finite()
    b2, d2 = y.finite()
    n1, n2 = b1.size, b2.size
    if n1 * n2 > max_pairs:
        raise ValueError(f"{n1} x {n2} points exceed max_pairs={max_pairs}; filter by min_persistence first")
    g1, g2 = _diag_distance(b1, d1, norm), _diag_distance(b2, d2, norm)
    if n1 == 0 and n2 == 0:
        return d_ess
    if n1 == 0 or n2 == 0:
        return max(d_ess, float(np.max(g1 if n1 else g2)))
    cross = _cross(b1, d1, b2, d2, norm)                                # [n1, n2]
    cand = np.unique(np.concatenate([cross.ravel(), g1, g2, [0.0]]))
    lo, hi = 0, cand.size - 1                                           # cand[hi] is always feasible
    while lo < hi:
        mid = (lo + hi) // 2
        delta = cand[mid]
        adj = cross <= delta
        must1, must2 = np.flatnonzero(g1 > delta), np.flatnonzero(g2 > delta)
        ok = _covers(adj, must1, np.arange(n2)) and _covers(adj.T, must2, np.arange(n1))
        if ok:
            hi = mid
        else:
            lo = mid + 1
    return max(d_ess, float(cand[lo]))


def wasserstein_distance(x: Diagram, y: Diagram, *, p: float = 2.0, norm: str = "inf") -> float:
    """Exact p-Wasserstein distance (module docstring)."""
    if p < 1:
        raise ValueError("p must be >= 1")
    ess = _essential_parts(x, y)
    if ess is None:
        return float("inf")
    total = float((np.abs(ess[0] - ess[1]) ** p).sum())
    b1, d1 = x.finite()
    b2, d2 = y.finite()
    n1, n2 = b1.size, b2.size
    g1, g2 = _diag_distance(b1, d1, norm) ** p, _diag_distance(b2, d2, norm) ** p
    if n1 and n2:
        cost = np.zeros((n1 + n2, n1 + n2))
        cost[:n1, :n2] = _cross(b1, d1, b2, d2, norm) ** p
        cost[:n1, n2:] = g1[:, None]
        cost[n1:, :n2] = g2[None, :]
        r, c = linear_sum_assignment(cost)
        total += float(cost[r, c].sum())
    else:
        total += float(g1.sum() + g2.sum())
    return total ** (1.0 / p)


def summary(diagram: Diagram) -> dict[str, float]:
    """Count, essential count, total and maximum persistence, persistent entropy (module docstring)."""
    b, d = diagram.finite()
    lengths = d - b
    total = float(lengths.sum())
    if lengths.size and total > 0:
        q = lengths[lengths > 0] / total
        entropy = float(-(q * np.log(q)).sum())
    else:
        entropy = float("nan")
    return {
        "dim": float(diagram.dim), "count": float(len(diagram)), "essential": float(diagram.essential().size),
        "total_persistence": total, "max_persistence": float(lengths.max()) if lengths.size else 0.0,
        "mean_persistence": float(lengths.mean()) if lengths.size else float("nan"),
        "persistent_entropy": entropy,
    }


def distance_matrix(diagrams: list[Diagram], *, metric: str = "bottleneck", p: float = 2.0, norm: str = "inf") -> np.ndarray:
    """Symmetric matrix of pairwise diagram distances (bottleneck or wasserstein)."""
    k = len(diagrams)
    out = np.zeros((k, k))
    for i in range(k):
        for j in range(i + 1, k):
            v = (bottleneck_distance(diagrams[i], diagrams[j], norm=norm) if metric == "bottleneck"
                 else wasserstein_distance(diagrams[i], diagrams[j], p=p, norm=norm))
            out[i, j] = out[j, i] = v
    return out


__all__ = ["Diagram", "bottleneck_distance", "distance_matrix", "summary", "wasserstein_distance"]
