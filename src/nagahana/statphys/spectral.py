"""Von Neumann and spectral entropies of the network graph and its multiplex, exact or stochastic (D-56).

Purpose
-------
The network state at a trigger is a multiplex graph: one layer per relation plane of CVG-AE (AS-01),
the same entities in every layer (graphs.py). This module reads the information content of that
structure: how evenly the graph spreads over its Laplacian modes (von Neumann entropy), how that
spread changes with the diffusion scale (spectral entropy, a canonical ensemble over the modes), how
far two states or two layers are apart (quantum Jensen-Shannon divergence) and how many layers the
multiplex really needs (structural reducibility). Their trajectories over triggers are early-warning
series (ews.py).

Mathematics
-----------
Graph G on n nodes with symmetric weights w_uv > 0 (u != v), degrees d_u = sum_v w_uv, combinatorial
Laplacian L = D - W: positive semidefinite, eigenvalues 0 = lambda_1 <= ... <= lambda_n with one zero
per connected component, trace tr L = sum_u d_u = 2 sum_{u<v} w_uv.

1. Von Neumann entropy (Braunstein, Ghosh and Severini, "The Laplacian of a graph as a density matrix:
   a basic combinatorial approach to separability of mixed states", Annals of Combinatorics 10:291,
   2006; Passerini and Severini, "Quantifying complexity in networks: the von Neumann entropy",
   International Journal of Agent Technologies and Systems 1(4):58, 2009):
       rho = L / tr L,    S_VN = -tr(rho log rho) = -sum_i (lambda_i / tr L) log(lambda_i / tr L)
   defined when the graph has an edge (tr L > 0). The complete graph K_n has spectrum {0, n (n - 1
   times)}, so S_VN(K_n) = log(n - 1). A disjoint union G_1 + G_2 with traces t_1, t_2 and
   w_k = t_k / (t_1 + t_2) has S_VN = w_1 S_VN(G_1) + w_2 S_VN(G_2) + H(w_1, w_2), because the spectrum
   of its rho is {w_k nu : nu in spec(rho_k)} (the grouping property). Isolated nodes add zero
   eigenvalues and change nothing.
2. Spectral entropy with diffusion time tau (De Domenico and Biamonte, "Spectral entropies as
   information-theoretic tools for complex network comparison", Physical Review X 6:041062, 2016,
   arXiv:1609.01214):
       rho_tau = exp(-tau L) / Z,  Z = tr exp(-tau L) = sum_i exp(-tau lambda_i)
       S_tau = -tr(rho_tau log rho_tau) = log Z + tau U,  U = tr(L rho_tau) = <lambda>
   a canonical ensemble over the Laplacian modes at inverse temperature tau, so also
       F_tau = -log(Z) / tau,   C_tau = tau^2 Var(lambda) = -dS_tau / d log tau.
   Limits: tau -> 0 gives the uniform distribution over the n modes (S_tau -> log n); tau -> inf keeps
   the zero modes only (S_tau -> log c, c connected components, isolated nodes included).
3. Quantum Jensen-Shannon divergence (Lamberti, Majtey, Borras, Casas and Plastino, "Metric character
   of the quantum Jensen-Shannon divergence", Physical Review A 77:052311, 2008):
       D_QJS(rho, sigma) = S((rho + sigma) / 2) - (S(rho) + S(sigma)) / 2,  0 <= D_QJS <= log 2
   whose square root is a metric on density matrices (Virosztek, "The metric property of the quantum
   Jensen-Shannon divergence", Advances in Mathematics 380:107595, 2021, arXiv:1910.10447). For the
   laplacian kind, (L_a / t_a + L_b / t_b) / 2 is the Laplacian of the graph with weights
   w_a / (2 t_a) + w_b / (2 t_b) (trace 1), so the divergence takes three von Neumann entropies at any
   size. For the diffusion kind the mixture is formed explicitly on each connected component of the
   union graph (exp(-tau L) is block diagonal over components), which needs every union component to
   be diagonalisable exactly (AS-770).
4. Multiplex reducibility (De Domenico, Nicosia, Arenas and Latora, "Structural reducibility of
   multilayer networks", Nature Communications 6:6864, 2015): layers G_1 ... G_P on one node set,
   aggregate A = sum_p G_p, h_A = S_VN(A). For a partition C of the layers into groups, each group
   aggregated into one layer, the relative entropy
       q(C) = 1 - (1 / |C|) sum_{c in C} S_VN(sum_{p in c} G_p) / h_A
   measures how distinguishable the multilayer description is from the aggregate. The layers are
   clustered hierarchically on the distance sqrt(D_QJS) (Lance and Williams, "A general theory of
   classificatory sorting strategies 1. Hierarchical systems", The Computer Journal 9(4):373, 1967;
   linkage `SpectralConfig.linkage`); q is reported at every level of the dendrogram together with the
   partition that maximises it. Layers without an edge have no density matrix and are left out (AS-771).

Exact path and stochastic Lanczos quadrature (AS-769)
-----------------------------------------------------
Every quantity is a sum over Laplacian eigenvalues and L is block diagonal over connected
components, so each graph is split:
- isolated nodes: eigenvalue 0, analytically;
- components with at most `exact_max_nodes` nodes: exact eigenvalues (LAPACK, float64; batched per
  component size), clamped at 0 (rounding can give -1e-16 for the zero mode);
- the union of the larger components: stochastic Lanczos quadrature (Ubaru, Chen and Saad, "Fast
  estimation of tr(f(A)) via stochastic Lanczos quadrature", SIAM Journal on Matrix Analysis and
  Applications 38(4):1075, 2017). With Rademacher probes v_l (Hutchinson, Communications in
  Statistics - Simulation and Computation 18(3):1059, 1989) and the m-step Lanczos tridiagonal T_m of
  each probe,
      v_l^T f(L) v_l ~ ||v_l||^2 sum_k (e_1^T y_k)^2 f(theta_k),   (theta_k, y_k) eigenpairs of T_m
  (Gauss quadrature of the spectral measure of v_l; exact for polynomials of degree <= 2m - 1).
  Lanczos runs batched over probes with full reorthogonalisation applied twice (classical
  Gram-Schmidt twice: Giraud, Langou, Rozloznik and van den Eshof, Numerische Mathematik 101:87,
  2005); a column whose beta falls below 1e-10 * ||L|| (bounded by 2 max degree) has found an
  invariant subspace and its quadrature is exact.
- Control variate: tr f(L) = tr p(L) + tr (f - p)(L) for a polynomial p of degree d <= 4
  (`slq_control_degree`) whose traces are exact: tr L^0 = n, tr L = sum_u d_u,
  tr L^2 = sum_u d_u^2 + 2 sum_e w_e^2, tr L^3 = sum_uv (L^2)_uv L_uv, tr L^4 = ||L^2||_F^2. The probes
  estimate only tr (f - p)(L). The Rademacher variance of v^T A v is 2 (||A||_F^2 - sum_u A_uu^2)
  (Avron and Toledo, "Randomized algorithms for estimating the trace of an implicit symmetric positive
  semi-definite matrix", Journal of the ACM 58(2):8, 2011), at most 2 sum_i (f - p)(lambda_i)^2 for
  A = f(L) - p(L); p minimises the quadrature estimate of that sum (weighted least squares of f on the
  Ritz nodes of a pilot batch, the Gauss weights as weights, in the variable x / scale), and the pilot
  probes are not reused in the estimate, which therefore stays unbiased.
- Zero-mode deflation: the null space of L is known exactly (one normalised indicator vector per
  connected component), so the probes are projected onto its orthogonal complement P v and the zero
  eigenvalues contribute c f(0) analytically: tr f(L) = c f(0) + E[(P v)^T f(L) (P v)] (P commutes with
  L). This removes the dominant rank-c part of exp(-tau L) at large tau, where plain probes have a
  relative variance of order one. The Lanczos recursions run on P0 L P0 (projection after every
  product), so rounding cannot re-excite the zero modes.
- Low-mode deflation: for any orthonormal U orthogonal to the zero modes and P = I - Z Z^T - U U^T
  (Z the zero modes), tr f(L) = c f(0) + sum_i u_i^T f(L) u_i + E[(P v)^T f(L) (P v)], because
  tr(U U^T f P) = 0. U holds approximate lowest non-zero modes (Ritz vectors of a projected Lanczos
  run, `slq_deflation` of them); their part is evaluated deterministically by Gauss quadrature from
  each u_i, so the probes only see the flatter rest of the spectrum. This is the low-rank-plus-
  residual split of Hutch++ (Meyer, Musco, Musco and Woodruff, "Hutch++: optimal stochastic trace
  estimation", SOSA 2021, arXiv:2010.09649) with a Krylov basis chosen for the decreasing functions
  exp(-tau x); the polynomial control variate applies to the residual (u^T p(L) u is integrated exactly).
- Error control: probes are added in batches until z * SE + depth error <= max(abs_tol, rel_tol *
  |value|) for every reported quantity (SE: the standard error over probes, propagated to ratios and
  logarithms by the delta method with the exact Jacobian; z = `slq_confidence`); the depth error is
  |Q(T_m) - Q(T_{m-1})|, and the depth is doubled (up to `slq_max_steps`, all probes recomputed) while
  it exceeds half the tolerance. The probe sequence depends only on `seed`, so successive triggers
  use common random numbers and estimation noise does not appear as white noise in a trajectory.

Products with L use a SciPy CSR matrix when SciPy is installed (imported lazily) and a NumPy
bincount scatter otherwise; both are exact. Degrees 3 and 4 of the control variate need SciPy (the
sparse square of L).

Precision (D-54): all spectra, entropies and estimates are float64.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from nagahana.statphys.config import JSD_KINDS, SpectralConfig

TraceFn = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class Graph:
    """Undirected weighted graph on nodes 0 ... n-1: one entry per edge with row < col and weight > 0."""

    n: int
    row: np.ndarray
    col: np.ndarray
    weight: np.ndarray

    def __post_init__(self) -> None:
        if self.n < 0:
            raise ValueError("a graph needs n >= 0 nodes")
        e = self.row.shape[0]
        if self.row.shape != (e,) or self.col.shape != (e,) or self.weight.shape != (e,):
            raise ValueError("row, col and weight must be [E]")
        if e and (int(self.row.min()) < 0 or int(self.col.max()) >= self.n or bool((self.row >= self.col).any())):
            raise ValueError("edges must satisfy 0 <= row < col < n (use Graph.from_edges)")
        if e and not bool(np.all(np.isfinite(self.weight)) and np.all(self.weight > 0)):
            raise ValueError("edge weights must be finite and > 0")

    @staticmethod
    def from_edges(n: int, u: np.ndarray, v: np.ndarray, w: np.ndarray | None = None) -> Graph:
        """Canonical graph from edge lists: self-loops dropped, (u, v) and (v, u) merged, weights summed."""
        uu = np.asarray(u, dtype=np.int64).ravel()
        vv = np.asarray(v, dtype=np.int64).ravel()
        ww = np.ones(uu.shape[0], dtype=np.float64) if w is None else np.asarray(w, dtype=np.float64).ravel()
        if uu.shape != vv.shape or uu.shape != ww.shape:
            raise ValueError("u, v and w must have the same length")
        if uu.size and (min(int(uu.min()), int(vv.min())) < 0 or max(int(uu.max()), int(vv.max())) >= n):
            raise ValueError("edge endpoints must lie in [0, n)")
        keep = (uu != vv) & (ww > 0)
        uu, vv, ww = uu[keep], vv[keep], ww[keep]
        lo, hi = np.minimum(uu, vv), np.maximum(uu, vv)
        base = max(n, 1)
        uniq, inv = np.unique(lo * base + hi, return_inverse=True)
        wsum = np.bincount(inv, weights=ww, minlength=uniq.shape[0]).astype(np.float64)
        return Graph(n=n, row=(uniq // base).astype(np.int64), col=(uniq % base).astype(np.int64), weight=wsum)

    @staticmethod
    def from_dense(adjacency: np.ndarray | torch.Tensor, *, atol: float = 1e-12) -> Graph:
        """Graph of a symmetric non-negative adjacency matrix (the diagonal is ignored)."""
        a = adjacency.detach().cpu().numpy() if isinstance(adjacency, torch.Tensor) else np.asarray(adjacency)
        a = a.astype(np.float64)
        if a.ndim != 2 or a.shape[0] != a.shape[1]:
            raise ValueError("adjacency must be a square matrix")
        if not np.allclose(a, a.T, atol=atol, rtol=0.0):
            raise ValueError("adjacency must be symmetric")
        if bool((a < -atol).any()):
            raise ValueError("adjacency must be non-negative")
        u, v = np.triu_indices(a.shape[0], k=1)
        w = a[u, v]
        sel = w > 0
        return Graph(n=a.shape[0], row=u[sel].astype(np.int64), col=v[sel].astype(np.int64), weight=w[sel])

    @property
    def num_edges(self) -> int:
        return int(self.row.shape[0])

    def degree(self) -> np.ndarray:
        """Weighted degree d_u [n] (float64)."""
        return (np.bincount(self.row, weights=self.weight, minlength=self.n)
                + np.bincount(self.col, weights=self.weight, minlength=self.n)).astype(np.float64)

    def trace(self) -> float:
        """tr L = sum_u d_u = 2 sum of edge weights."""
        return 2.0 * math.fsum(self.weight.tolist())

    def laplacian(self) -> np.ndarray:
        """Dense combinatorial Laplacian L = D - W [n, n] (float64)."""
        lap = np.zeros((self.n, self.n), dtype=np.float64)
        lap[self.row, self.col] = -self.weight
        lap[self.col, self.row] = -self.weight
        lap[np.arange(self.n), np.arange(self.n)] = self.degree()
        return lap

    def components(self) -> tuple[np.ndarray, int]:
        """Connected components: (label [n] in 0 ... c-1, c). Isolated nodes are components of their own.

        Hook-and-jump union-find: roots are hooked under the smaller root of every edge whose endpoints
        have different roots, then pointers jump to their roots; parent[x] <= x always holds, so no cycle
        can form, and the loop ends when every edge joins equal roots.
        """
        parent = np.arange(self.n, dtype=np.int64)
        if self.num_edges:
            while True:
                while True:
                    grand = parent[parent]
                    if np.array_equal(grand, parent):
                        break
                    parent = grand
                pr, pc = parent[self.row], parent[self.col]
                diff = pr != pc
                if not bool(diff.any()):
                    break
                np.minimum.at(parent, np.maximum(pr[diff], pc[diff]), np.minimum(pr[diff], pc[diff]))
        if self.n == 0:
            return parent, 0
        _, label = np.unique(parent, return_inverse=True)
        return label.astype(np.int64), int(label.max()) + 1

    def subgraph(self, nodes: np.ndarray) -> Graph:
        """The induced subgraph on `nodes` (sorted unique node ids), relabelled 0 ... len(nodes)-1."""
        nodes = np.asarray(nodes, dtype=np.int64)
        index = np.full(self.n, -1, dtype=np.int64)
        index[nodes] = np.arange(nodes.shape[0])
        r, c = index[self.row], index[self.col]
        sel = (r >= 0) & (c >= 0)
        lo, hi = np.minimum(r[sel], c[sel]), np.maximum(r[sel], c[sel])
        return Graph(n=int(nodes.shape[0]), row=lo, col=hi, weight=self.weight[sel])


def combine(graphs: Sequence[Graph], coefficients: Sequence[float] | None = None) -> Graph:
    """sum_k c_k G_k on a common node set (weights added edge by edge)."""
    if not graphs:
        raise ValueError("combine needs at least one graph")
    n = graphs[0].n
    if any(g.n != n for g in graphs):
        raise ValueError("combine needs graphs on the same node set")
    coef = [1.0] * len(graphs) if coefficients is None else [float(c) for c in coefficients]
    if len(coef) != len(graphs) or any(not (math.isfinite(c) and c > 0) for c in coef):
        raise ValueError("coefficients must be one finite positive number per graph")
    return Graph.from_edges(n, np.concatenate([g.row for g in graphs]), np.concatenate([g.col for g in graphs]),
                            np.concatenate([g.weight * c for g, c in zip(graphs, coef, strict=True)]))


@dataclass(frozen=True)
class Estimate:
    """A reported quantity: value, standard error (0 when exact), and how it was obtained."""

    value: float
    stderr: float
    exact: bool
    probes: int
    steps: int


_NAN_ESTIMATE = Estimate(math.nan, math.nan, True, 0, 0)


def _exact(value: float) -> Estimate:
    return Estimate(value=float(value), stderr=0.0, exact=True, probes=0, steps=0)


@dataclass(frozen=True)
class _Split:
    # A graph split for spectral sums: isolated nodes, exact eigenvalues of the small components, and the
    # subgraph of all large components (None when there is none).
    isolated: int
    eigenvalues: np.ndarray
    large: Graph | None
    components: int


def _small_spectra(g: Graph, label: np.ndarray, sizes: np.ndarray, comps: np.ndarray) -> np.ndarray:
    # Exact Laplacian eigenvalues of the components `comps` (all of size >= 2), batched per size.
    out: list[np.ndarray] = []
    deg = g.degree()
    n_comp = int(sizes.shape[0])
    for s in np.unique(sizes[comps]).tolist():
        cs = comps[sizes[comps] == s]
        slot = np.full(n_comp, -1, dtype=np.int64)
        slot[cs] = np.arange(cs.shape[0])
        nodes = np.nonzero(slot[label] >= 0)[0]
        nodes = nodes[np.lexsort((nodes, label[nodes]))]                       # grouped by component
        local = np.empty(g.n, dtype=np.int64)
        local[nodes] = np.arange(nodes.shape[0]) % s                           # rank inside the component
        lap = np.zeros((cs.shape[0], s, s), dtype=np.float64)
        lap[slot[label[nodes]], local[nodes], local[nodes]] = deg[nodes]
        sel = slot[label[g.row]] >= 0
        er, ec, ew = g.row[sel], g.col[sel], g.weight[sel]
        k = slot[label[er]]
        lap[k, local[er], local[ec]] = -ew
        lap[k, local[ec], local[er]] = -ew
        out.append(np.clip(np.linalg.eigvalsh(lap).reshape(-1), 0.0, None))
    return np.concatenate(out) if out else np.zeros(0, dtype=np.float64)


def _split(g: Graph, exact_max: int) -> _Split:
    if g.n == 0:
        return _Split(0, np.zeros(0, dtype=np.float64), None, 0)
    label, c = g.components()
    sizes = np.bincount(label, minlength=c)
    comps = np.arange(c)
    small = comps[(sizes >= 2) & (sizes <= exact_max)]
    large = comps[sizes > exact_max]
    eig = _small_spectra(g, label, sizes, small) if small.size else np.zeros(0, dtype=np.float64)
    big = g.subgraph(np.nonzero(np.isin(label, large))[0]) if large.size else None
    return _Split(isolated=int((sizes == 1).sum()), eigenvalues=eig, large=big, components=c)


def laplacian_spectrum(graph: Graph) -> np.ndarray:
    """All n Laplacian eigenvalues, ascending (exact, float64; clamped at 0)."""
    if graph.n == 0:
        return np.zeros(0, dtype=np.float64)
    return np.clip(np.linalg.eigvalsh(graph.laplacian()), 0.0, None)


class _LaplacianOperator:
    """Products with L (SciPy CSR if available, NumPy scatter otherwise) and exact traces of its powers."""

    def __init__(self, g: Graph) -> None:
        self.g = g
        self.n = g.n
        self.deg = g.degree()
        self.norm_bound = 2.0 * float(self.deg.max()) if self.n else 0.0      # Gershgorin bound of ||L||
        # Zero modes: one per connected component, spanned by the component's indicator vector.
        self.label, self.zero_modes = g.components()
        self._size = np.bincount(self.label, minlength=self.zero_modes).astype(np.float64)
        self._order = np.argsort(self.label, kind="stable")                    # nodes grouped by component
        self._starts = np.searchsorted(self.label[self._order], np.arange(self.zero_modes))
        self._csr: Any = None
        try:
            import scipy.sparse as sp
        except ImportError:
            sp = None
        if sp is not None:
            idx = np.arange(self.n)
            self._csr = sp.csr_matrix(
                (np.concatenate([-g.weight, -g.weight, self.deg]),
                 (np.concatenate([g.row, g.col, idx]), np.concatenate([g.col, g.row, idx]))), shape=(self.n, self.n))

    def project(self, x: np.ndarray) -> np.ndarray:
        """X minus its projection on the zero modes (component means removed per column), X [n, k]."""
        if self.zero_modes == 1:
            return x - x.mean(axis=0, keepdims=True)
        sums = np.add.reduceat(x[self._order], self._starts, axis=0)          # [c, k] per-component sums
        return x - (sums / self._size[:, None])[self.label]

    def matmul(self, x: np.ndarray) -> np.ndarray:
        """L X for X [n, k]."""
        if self._csr is not None:
            return np.asarray(self._csr @ x)
        g = self.g
        y = self.deg[:, None] * x
        for j in range(x.shape[1]):
            y[:, j] -= np.bincount(g.row, weights=g.weight * x[g.col, j], minlength=self.n)
            y[:, j] -= np.bincount(g.col, weights=g.weight * x[g.row, j], minlength=self.n)
        return y

    def power_traces(self, degree: int) -> np.ndarray:
        """tr L^i for i = 0 ... degree (exact; degree <= 4)."""
        w = self.g.weight
        out = [float(self.n), math.fsum(self.deg.tolist()),
               math.fsum((self.deg**2).tolist()) + 2.0 * math.fsum((w**2).tolist())]
        if degree >= 3:
            if self._csr is None:
                raise ImportError("control-variate degrees 3 and 4 need SciPy (the sparse square of L); "
                                  "install scipy or set spectral.slq_control_degree <= 2")
            sq = self._csr @ self._csr
            out.append(float(sq.multiply(self._csr).sum()))                    # tr L^3 = sum_uv (L^2)_uv L_uv
            out.append(float(sq.multiply(sq).sum()))                           # tr L^4 = ||L^2||_F^2
        return np.array(out[: degree + 1], dtype=np.float64)


@dataclass(frozen=True)
class _LanczosRun:
    # Lanczos coefficients of k probe columns: alpha, beta [k, m]; effective steps per column; ||v||^2;
    # the Lanczos vectors [k, m+1, n] when kept.
    alpha: np.ndarray
    beta: np.ndarray
    length: np.ndarray
    sq_norm: np.ndarray
    basis: np.ndarray | None = None


def _lanczos(op: _LaplacianOperator, v0: np.ndarray, steps: int, *, keep_basis: bool = False,
             project: bool = True) -> _LanczosRun:
    """Batched Lanczos on the columns of v0 [n, k], full reorthogonalisation twice per step.

    project: apply the zero-mode projection after every product, i.e. run on P0 L P0. For start vectors
    in the complement of the zero modes this changes nothing in exact arithmetic (L maps the complement
    into itself); in floating point it stops rounding errors from re-exciting the zero modes, which
    long runs would otherwise find as spurious Ritz values near 0.
    """
    n, k = v0.shape
    m = min(steps, n)
    basis = np.zeros((k, m + 1, n), dtype=np.float64)                         # [k, m+1, n] Lanczos vectors
    sq_norm = (v0 * v0).sum(0)
    basis[:, 0, :] = (v0 / np.sqrt(sq_norm)).T
    alpha = np.zeros((k, m), dtype=np.float64)
    beta = np.zeros((k, m), dtype=np.float64)
    length = np.zeros(k, dtype=np.int64)
    active = np.ones(k, dtype=bool)
    tol = 1e-10 * max(op.norm_bound, 1e-300)
    for j in range(m):
        q = basis[:, j, :]                                                     # [k, n]
        w = op.matmul(q.T).T
        if project:
            w = op.project(w.T).T
        if j > 0:
            w = w - beta[:, j - 1, None] * basis[:, j - 1, :]
        a = (q * w).sum(1)
        w = w - a[:, None] * q
        span = basis[:, : j + 1, :]                                            # [k, j+1, n]
        for _ in range(2):
            coef = np.matmul(span, w[:, :, None])                              # [k, j+1, 1]
            w = w - np.matmul(span.transpose(0, 2, 1), coef)[:, :, 0]
        b = np.sqrt((w * w).sum(1))
        alpha[:, j] = np.where(active, a, 0.0)
        length += active
        cont = active & (b > tol)
        beta[:, j] = np.where(cont, b, 0.0)
        active = cont
        basis[:, j + 1, :] = np.where(cont[:, None], w / np.where(cont, b, 1.0)[:, None], 0.0)
        if not active.any():
            break
    return _LanczosRun(alpha=alpha, beta=beta, length=length, sq_norm=sq_norm, basis=basis if keep_basis else None)


def _gauss_rules(run: _LanczosRun, depth: int) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Gauss rules of every column with min(depth, length) nodes: (columns, nodes [g, s], weights [g, s]).

    The weights of one column sum to 1 (squared first components of the eigenvectors of T).
    """
    use = np.minimum(run.length, depth)
    rules = []
    for s in np.unique(use).tolist():
        if s < 1:
            continue
        cols = np.nonzero(use == s)[0]
        t = np.zeros((cols.shape[0], s, s), dtype=np.float64)
        idx = np.arange(s)
        t[:, idx, idx] = run.alpha[cols, :s]
        if s > 1:
            t[:, idx[:-1], idx[1:]] = run.beta[cols, : s - 1]
            t[:, idx[1:], idx[:-1]] = run.beta[cols, : s - 1]
        theta, vec = np.linalg.eigh(t)
        rules.append((cols, np.clip(theta, 0.0, None), vec[:, 0, :] ** 2))      # L is PSD: Ritz values >= 0
    return rules


def _samples(run: _LanczosRun, depth: int, fns: Sequence[TraceFn]) -> np.ndarray:
    """Per-column quadrature estimates of v^T f(L) v: [k, F]."""
    out = np.zeros((run.alpha.shape[0], len(fns)), dtype=np.float64)
    for cols, theta, wts in _gauss_rules(run, depth):
        w = wts * run.sq_norm[cols, None]
        for i, f in enumerate(fns):
            out[cols, i] = (w * f(theta)).sum(1)
    return out


def _rademacher(rng: np.random.Generator, n: int, k: int) -> np.ndarray:
    return rng.integers(0, 2, size=(n, k)).astype(np.float64) * 2.0 - 1.0


def _deflation_basis(op: _LaplacianOperator, r: int, steps: int, rng: np.random.Generator) -> np.ndarray:
    """Orthonormal U [n, r'] of approximate lowest non-zero Laplacian modes, orthogonal to the zero modes.

    Ritz vectors of the r smallest Ritz values of a projected Lanczos run with max(3 r, steps) steps.
    Any orthonormal U keeps the deflated estimator exact; closer eigenvectors only lower its variance.
    """
    room = op.n - op.zero_modes                                                # dimension of the complement
    r = min(r, room)
    if r <= 0:
        return np.zeros((op.n, 0), dtype=np.float64)
    v = op.project(_rademacher(rng, op.n, 1))
    run = _lanczos(op, v, min(max(3 * r, steps), room), keep_basis=True)
    m = int(run.length[0])
    idx = np.arange(m)
    t = np.zeros((m, m), dtype=np.float64)
    t[idx, idx] = run.alpha[0, :m]
    t[idx[:-1], idx[1:]] = run.beta[0, : m - 1]
    t[idx[1:], idx[:-1]] = run.beta[0, : m - 1]
    theta, vec = np.linalg.eigh(t)
    assert run.basis is not None
    keep = np.nonzero(theta > 1e-8 * max(op.norm_bound, 1e-300))[0][:r]        # numerical zero modes dropped
    u = op.project(run.basis[0, :m, :].T @ vec[:, keep])                       # [n, r'] Ritz vectors
    for _ in range(2):                                                         # orthonormalise twice
        u, _ = np.linalg.qr(op.project(u))
    return u


def _deflate(u: np.ndarray, x: np.ndarray) -> np.ndarray:
    # x minus its projection on the columns of the orthonormal u.
    return x - u @ (u.T @ x) if u.shape[1] else x


def _control_variates(op: _LaplacianOperator, fns: Sequence[TraceFn], degree: int, depth: int, pilot: np.ndarray
                      ) -> tuple[list[TraceFn], np.ndarray]:
    """Fitted polynomials p_j (module docstring): residual functions f_j - p_j and the exact traces of p_j
    over the complement of the zero modes, tr p_j(L) - c p_j(0). `pilot` holds projected probes."""
    if degree == 0:
        return list(fns), np.zeros(len(fns), dtype=np.float64)
    rules = _gauss_rules(_lanczos(op, pilot, depth), depth)
    nodes = np.concatenate([th.reshape(-1) for _, th, _ in rules])
    wts = np.concatenate([w.reshape(-1) for _, _, w in rules])
    scale = max(float(nodes.max()), op.norm_bound * 1e-12, 1e-300)
    powers = np.arange(degree + 1)
    design = (nodes[:, None] / scale) ** powers[None, :] * np.sqrt(wts)[:, None]
    coefs = np.stack([np.linalg.lstsq(design, f(nodes) * np.sqrt(wts), rcond=None)[0] for f in fns])   # [F, d+1]
    traces = coefs @ (op.power_traces(degree) / scale**powers) - op.zero_modes * coefs[:, 0]   # p(0) = c_0

    def residual(j: int) -> TraceFn:
        c = coefs[j]

        def f(x: np.ndarray) -> np.ndarray:
            return fns[j](x) - np.polynomial.polynomial.polyval(x / scale, c)

        return f

    return [residual(j) for j in range(len(fns))], traces


def _derived(derive: Callable[[torch.Tensor], torch.Tensor], traces: np.ndarray, cov: np.ndarray
             ) -> tuple[np.ndarray, np.ndarray]:
    # Quantities derive(traces) and their standard errors by the delta method (exact Jacobian).
    t = torch.from_numpy(traces)
    values = derive(t).detach().numpy()
    if not np.any(cov):
        return values, np.zeros_like(values)
    jac = torch.autograd.functional.jacobian(derive, t).numpy()               # [Q, F]
    var = np.einsum("qf,fg,qg->q", jac, cov, jac)
    return values, np.sqrt(np.clip(var, 0.0, None))


def _estimate(split: _Split, fns: Sequence[TraceFn], exact: np.ndarray, derive: Callable[[torch.Tensor], torch.Tensor],
              config: SpectralConfig, *, deflate: bool) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Values and standard errors of derive(traces), traces = exact part + stochastic part (module docstring).

    deflate: deflate the approximate lowest modes (worth it for exp(-tau x), whose mass sits on them; the
    function x log x of the von Neumann entropy puts almost none there). Returns (values [Q], stderr [Q],
    probes of the estimate, Lanczos depth). Exact without a large component.
    """
    if split.large is None:
        values = derive(torch.from_numpy(exact)).detach().numpy()
        return values, np.zeros_like(values), 0, 0
    op = _LaplacianOperator(split.large)
    rng = np.random.default_rng(int(config.seed))
    depth = min(config.slq_steps, op.n)
    # Exact deflation of the zero modes (c f(0) analytically, c = connected components of the large
    # part) and of approximate lowest modes U (their part evaluated deterministically by quadrature).
    zero = np.array([float(f(np.zeros(1))[0]) for f in fns], dtype=np.float64) * op.zero_modes
    low = _deflation_basis(op, config.slq_deflation if deflate else 0, config.slq_steps, rng)

    def probe(k: int) -> np.ndarray:
        return _deflate(low, op.project(_rademacher(rng, op.n, k)))

    resid, cv_traces = _control_variates(op, fns, config.slq_control_degree, depth, probe(config.slq_probe_batch))
    base = exact + zero + cv_traces
    probes: list[np.ndarray] = []
    cur: list[np.ndarray] = []                                                 # [k, F] per batch at `depth`
    prev: list[np.ndarray] = []                                                # ... with depth - 1 nodes

    def run(v: np.ndarray, d: int) -> tuple[np.ndarray, np.ndarray]:
        lz = _lanczos(op, v, d)
        return _samples(lz, d, resid), _samples(lz, max(d - 1, 1), resid)

    def deflated(d: int) -> tuple[np.ndarray, np.ndarray]:
        # sum_i u_i^T (f - p)(L) u_i by Gauss quadrature from each u_i (deterministic), at d and d - 1 nodes.
        if not low.shape[1]:
            return np.zeros(len(fns)), np.zeros(len(fns))
        a, b = run(low, d)
        return a.sum(0), b.sum(0)

    part, part_prev = deflated(depth)
    while True:
        k = min(config.slq_probe_batch, config.slq_max_probes - sum(p.shape[1] for p in probes))
        v = probe(k)
        probes.append(v)
        a, b = run(v, depth)
        cur.append(a)
        prev.append(b)
        while True:
            s = np.concatenate(cur, axis=0)                                    # [p, F]
            p = s.shape[0]
            cov = np.atleast_2d(np.cov(s, rowvar=False, ddof=1)) / p if p > 1 else np.zeros((len(fns), len(fns)))
            values, se = _derived(derive, base + part + s.mean(0), cov)
            values_prev = derive(torch.from_numpy(base + part_prev + np.concatenate(prev, axis=0).mean(0))
                                 ).detach().numpy()
            depth_err = np.abs(values - values_prev)
            tol = np.maximum(config.slq_abs_tol, config.slq_rel_tol * np.abs(values))
            if not (bool(np.any(depth_err > 0.5 * tol)) and depth < min(config.slq_max_steps, op.n)):
                break
            # The quadrature has not converged in depth: rerun every probe with twice the steps.
            depth = min(2 * depth, config.slq_max_steps, op.n)
            pairs = [run(pv, depth) for pv in probes]
            cur = [x for x, _ in pairs]
            prev = [y for _, y in pairs]
            part, part_prev = deflated(depth)
        done = bool(np.all(config.slq_confidence * se + depth_err <= tol))
        if (p >= config.slq_min_probes and done) or p >= config.slq_max_probes:
            return values, se, p, depth


def _vn_fn(trace: float) -> TraceFn:
    # f(x) = -(x / t) log(x / t) with f(0) = 0.
    def f(x: np.ndarray) -> np.ndarray:
        y = np.asarray(x, dtype=np.float64) / trace
        pos = y > 0
        return np.where(pos, -y * np.log(np.where(pos, y, 1.0)), 0.0)

    return f


def _spectral_fns(tau: float) -> list[TraceFn]:
    # exp(-tau x), x exp(-tau x), x^2 exp(-tau x): Z, M1 and M2 of the diffusion ensemble.
    def z(x: np.ndarray) -> np.ndarray:
        return np.exp(-tau * x)

    def m1(x: np.ndarray) -> np.ndarray:
        return x * np.exp(-tau * x)

    def m2(x: np.ndarray) -> np.ndarray:
        return x * x * np.exp(-tau * x)

    return [z, m1, m2]


@dataclass(frozen=True)
class SpectralThermo:
    """Canonical ensemble over the Laplacian modes at diffusion time tau (module docstring, item 2)."""

    tau: float
    entropy: Estimate
    free_energy: Estimate
    mean: Estimate
    heat_capacity: Estimate


@dataclass(frozen=True)
class GraphEntropies:
    """Entropies of one graph: von Neumann (NaN without an edge) and the spectral ensemble per tau."""

    nodes: int
    edges: int
    components: int
    von_neumann: Estimate
    spectral: tuple[SpectralThermo, ...]


def graph_entropies(graph: Graph, *, config: SpectralConfig, taus: Sequence[float] | None = None) -> GraphEntropies:
    """Von Neumann entropy and spectral thermodynamics of `graph` (exact or stochastic, module docstring)."""
    tau_list = [float(t) for t in (config.taus if taus is None else taus)]
    for t in tau_list:
        if not (math.isfinite(t) and t > 0):
            raise ValueError("every tau must be finite and > 0")
    if graph.n == 0:
        return GraphEntropies(0, 0, 0, _NAN_ESTIMATE,
                              tuple(SpectralThermo(t, _NAN_ESTIMATE, _NAN_ESTIMATE, _NAN_ESTIMATE, _NAN_ESTIMATE)
                                    for t in tau_list))
    split = _split(graph, config.exact_max_nodes)
    tr = graph.trace()
    has_vn = tr > 0
    fns: list[TraceFn] = [_vn_fn(tr)] if has_vn else []
    for t in tau_list:
        fns.extend(_spectral_fns(t))
    if not fns:
        return GraphEntropies(graph.n, graph.num_edges, split.components, _NAN_ESTIMATE, ())
    lam = split.eigenvalues
    exact = np.array([float(f(lam).sum()) for f in fns], dtype=np.float64)
    offset = 1 if has_vn else 0
    for i in range(len(tau_list)):
        exact[offset + 3 * i] += float(split.isolated)                         # isolated nodes: exp(0) = 1 in Z

    def derive(traces: torch.Tensor) -> torch.Tensor:
        out = [traces[0]] if has_vn else []
        for i, t in enumerate(tau_list):
            z, m1, m2 = traces[offset + 3 * i], traces[offset + 3 * i + 1], traces[offset + 3 * i + 2]
            u = m1 / z
            out += [torch.log(z) + t * u, -torch.log(z) / t, u, t * t * (m2 / z - u * u)]
        return torch.stack(out)

    values, se, probes, steps = _estimate(split, fns, exact, derive, config, deflate=bool(tau_list))
    is_exact = split.large is None

    def est(i: int) -> Estimate:
        return Estimate(float(values[i]), float(se[i]), is_exact, probes, steps)

    vn = est(0) if has_vn else _NAN_ESTIMATE
    spec = tuple(SpectralThermo(t, est(offset + 4 * i), est(offset + 4 * i + 1), est(offset + 4 * i + 2),
                                est(offset + 4 * i + 3)) for i, t in enumerate(tau_list))
    return GraphEntropies(graph.n, graph.num_edges, split.components, vn, spec)


def von_neumann_entropy(graph: Graph, *, config: SpectralConfig) -> Estimate:
    """S_VN of rho = L / tr L (NaN for a graph without an edge)."""
    return graph_entropies(graph, config=config, taus=()).von_neumann


def spectral_entropy(graph: Graph, tau: float, *, config: SpectralConfig) -> SpectralThermo:
    """Spectral entropy and thermodynamics of rho_tau = exp(-tau L) / Z."""
    return graph_entropies(graph, config=config, taus=(tau,)).spectral[0]


def _diffusion_mixture_entropy(a: Graph, b: Graph, tau: float, za: float, zb: float, exact_max: int) -> float:
    # S((rho_a + rho_b) / 2) for diffusion-kind matrices, per connected component of the union graph.
    union = combine([a, b]) if a.num_edges + b.num_edges else a
    label, c = union.components()
    sizes = np.bincount(label, minlength=c) if union.n else np.zeros(0, dtype=np.int64)
    if sizes.size and bool((sizes > exact_max).any()):
        raise ValueError(f"diffusion-kind divergence needs every connected component of the union graph to have at "
                         f"most exact_max_nodes = {exact_max} nodes (largest: {int(sizes.max())}); use kind 'laplacian'")
    total = 0.0
    iso = int((sizes == 1).sum())
    if iso:
        mu = 0.5 * (1.0 / za + 1.0 / zb)                                        # exp(0) / Z of an isolated node
        total += -iso * mu * math.log(mu)
    for comp in np.nonzero(sizes >= 2)[0].tolist():
        nodes = np.nonzero(label == comp)[0]
        mats = []
        for g, z in ((a, za), (b, zb)):
            lam, vec = np.linalg.eigh(g.subgraph(nodes).laplacian())
            mats.append((vec * np.exp(-tau * np.clip(lam, 0.0, None))) @ vec.T / z)
        mix = 0.5 * (mats[0] + mats[1])
        mu_c = np.clip(np.linalg.eigvalsh(0.5 * (mix + mix.T)), 0.0, None)
        pos = mu_c[mu_c > 0]
        total += float(-(pos * np.log(pos)).sum())
    return total


def laplacian_jsd(a: Graph, b: Graph, ha: Estimate, hb: Estimate, config: SpectralConfig) -> Estimate:
    """D_QJS of laplacian-kind density matrices when S(rho_a) and S(rho_b) are known already.

    Only the mixture graph's entropy is computed; ha and hb must be the von Neumann entropies of a and b
    (isolated nodes do not change them, so entropies of unaligned graphs may be passed).
    """
    ta, tb = a.trace(), b.trace()
    if ta <= 0 or tb <= 0:
        return _NAN_ESTIMATE
    hm = von_neumann_entropy(combine([a, b], [0.5 / ta, 0.5 / tb]), config=config)
    value = hm.value - 0.5 * (ha.value + hb.value)
    exact = hm.exact and ha.exact and hb.exact
    # The independent-error bound on the standard error; shared probes make the true error smaller.
    se = math.sqrt(hm.stderr**2 + 0.25 * (ha.stderr**2 + hb.stderr**2))
    return Estimate(value=max(value, 0.0) if exact else value, stderr=se, exact=exact,
                    probes=max(hm.probes, ha.probes, hb.probes), steps=max(hm.steps, ha.steps, hb.steps))


def quantum_jsd(a: Graph, b: Graph, *, config: SpectralConfig, kind: str | None = None,
                tau: float | None = None) -> Estimate:
    """D_QJS between the density matrices of two graphs on the same node set (module docstring, item 3).

    kind: "laplacian" (rho = L / tr L; NaN if either graph has no edge) or "diffusion" (rho_tau; needs
    every connected component of the union graph to be diagonalisable exactly); default
    `config.jsd_kind`, tau default `config.jsd_tau`.
    """
    k = config.jsd_kind if kind is None else kind
    if k not in JSD_KINDS:
        raise ValueError(f"kind must be one of {JSD_KINDS}")
    if a.n != b.n:
        raise ValueError("quantum_jsd needs graphs on the same node set (align them first)")
    if k == "laplacian":
        if a.trace() <= 0 or b.trace() <= 0:
            return _NAN_ESTIMATE
        return laplacian_jsd(a, b, von_neumann_entropy(a, config=config), von_neumann_entropy(b, config=config), config)
    t = config.jsd_tau if tau is None else float(tau)
    if not (math.isfinite(t) and t > 0):
        raise ValueError("tau must be finite and > 0")
    if a.n == 0:
        return _NAN_ESTIMATE
    sa = graph_entropies(a, config=config, taus=(t,)).spectral[0]
    sb = graph_entropies(b, config=config, taus=(t,)).spectral[0]
    if not (sa.entropy.exact and sb.entropy.exact):
        raise ValueError("diffusion-kind divergence needs exact spectra (components of at most exact_max_nodes "
                         "nodes); use kind 'laplacian'")
    za = math.exp(-t * sa.free_energy.value)                                   # Z = exp(-tau F)
    zb = math.exp(-t * sb.free_energy.value)
    hm = _diffusion_mixture_entropy(a, b, t, za, zb, config.exact_max_nodes)
    return _exact(max(hm - 0.5 * (sa.entropy.value + sb.entropy.value), 0.0))


@dataclass(frozen=True)
class MultiplexEntropy:
    """Multiplex reading (module docstring, item 4); layer indices refer to `names`.

    layer_entropy [P] (NaN: no edge) and layer_stderr [P]; aggregate_entropy; jsd [P, P] (NaN where a
    layer has no edge); relative_entropy: q of the unreduced multiplex (every layer with an edge kept
    apart); levels: (partition, q) from the unreduced multiplex down to one aggregate;
    best_partition / best_relative_entropy: the level that maximises q (ties: the finer level).
    """

    names: tuple[str, ...]
    layer_entropy: np.ndarray
    layer_stderr: np.ndarray
    aggregate_entropy: float
    jsd: np.ndarray
    relative_entropy: float
    levels: tuple[tuple[tuple[tuple[int, ...], ...], float], ...]
    best_partition: tuple[tuple[int, ...], ...]
    best_relative_entropy: float


def merge_order(dist: np.ndarray, linkage: str) -> list[tuple[int, int]]:
    """Agglomerative clustering with Lance-Williams updates of a distance matrix [P, P].

    Returns the merges in order as (kept slot, retired slot); the merged cluster takes the kept slot.
    Ties: the lexicographically first pair. Ward's rule updates squared distances (Ward, Journal of the
    American Statistical Association 58:236, 1963).
    """
    p = dist.shape[0]
    d = dist.astype(np.float64).copy()
    if linkage == "ward":
        d = d * d
    size = np.ones(p, dtype=np.float64)
    alive = list(range(p))
    merges: list[tuple[int, int]] = []
    while len(alive) > 1:
        best = (math.inf, -1, -1)
        for x in range(len(alive)):
            for y in range(x + 1, len(alive)):
                i, j = alive[x], alive[y]
                if d[i, j] < best[0]:
                    best = (float(d[i, j]), i, j)
        _, i, j = best
        for k in alive:
            if k in (i, j):
                continue
            if linkage == "single":
                new = min(d[k, i], d[k, j])
            elif linkage == "complete":
                new = max(d[k, i], d[k, j])
            elif linkage == "average":
                new = (size[i] * d[k, i] + size[j] * d[k, j]) / (size[i] + size[j])
            elif linkage == "ward":
                tot = size[i] + size[j] + size[k]
                new = ((size[i] + size[k]) * d[k, i] + (size[j] + size[k]) * d[k, j] - size[k] * d[i, j]) / tot
            else:
                raise ValueError(f"unknown linkage {linkage!r}")
            d[k, i] = d[i, k] = new
        size[i] += size[j]
        alive.remove(j)
        merges.append((i, j))
    return merges


def multiplex_entropy(layers: Sequence[Graph], names: Sequence[str], *, config: SpectralConfig,
                      aggregate: Estimate | None = None) -> MultiplexEntropy:
    """Per-layer and aggregate entropies, inter-layer divergences, relative entropy and reduction.

    aggregate: the von Neumann entropy of the aggregate graph when the caller has it already (it is the
    same estimate, so passing it only saves its recomputation).
    """
    if len(layers) != len(names) or not layers:
        raise ValueError("one name per layer, at least one layer")
    n = layers[0].n
    if any(g.n != n for g in layers):
        raise ValueError("multiplex layers must share their node set")
    p_n = len(layers)
    h_est = [von_neumann_entropy(g, config=config) for g in layers]
    h = np.array([e.value for e in h_est], dtype=np.float64)
    agg = combine(list(layers)) if any(g.num_edges for g in layers) else layers[0]
    if aggregate is not None:
        h_a = aggregate.value
    else:
        h_a = von_neumann_entropy(agg, config=config).value if agg.num_edges else math.nan
    jsd = np.full((p_n, p_n), math.nan, dtype=np.float64)
    ok = [i for i in range(p_n) if layers[i].num_edges > 0]
    for i in ok:
        jsd[i, i] = 0.0
    for x, i in enumerate(ok):
        for j in ok[x + 1:]:
            jsd[i, j] = jsd[j, i] = laplacian_jsd(layers[i], layers[j], h_est[i], h_est[j], config).value
    levels: list[tuple[tuple[tuple[int, ...], ...], float]] = []
    if ok and math.isfinite(h_a) and h_a > 0:
        clusters: dict[int, tuple[int, ...]] = {s: (layer,) for s, layer in enumerate(ok)}
        cache: dict[tuple[int, ...], float] = {(layer,): float(h[layer]) for layer in ok}

        def group_entropy(group: tuple[int, ...]) -> float:
            if group not in cache:
                cache[group] = von_neumann_entropy(combine([layers[i] for i in group]), config=config).value
            return cache[group]

        def q_of(parts: list[tuple[int, ...]]) -> float:
            return 1.0 - float(np.mean([group_entropy(g) for g in parts])) / h_a

        parts = sorted(clusters.values())
        levels.append((tuple(parts), q_of(parts)))
        dist = np.sqrt(np.clip(jsd[np.ix_(ok, ok)], 0.0, None))
        for i, j in merge_order(dist, config.linkage):
            clusters[i] = tuple(sorted(clusters[i] + clusters.pop(j)))
            parts = sorted(clusters.values())
            levels.append((tuple(parts), q_of(parts)))
    if levels:
        best = max(range(len(levels)), key=lambda k: (levels[k][1], -k))       # ties: the earlier (finer) level
        best_partition, best_q, q0 = levels[best][0], levels[best][1], levels[0][1]
    else:
        best_partition, best_q, q0 = (), math.nan, math.nan
    return MultiplexEntropy(names=tuple(names), layer_entropy=h,
                            layer_stderr=np.array([e.stderr for e in h_est], dtype=np.float64), aggregate_entropy=h_a,
                            jsd=jsd, relative_entropy=q0, levels=tuple(levels), best_partition=best_partition,
                            best_relative_entropy=best_q)


__all__ = ["Estimate", "Graph", "GraphEntropies", "MultiplexEntropy", "SpectralThermo", "combine", "graph_entropies",
           "laplacian_jsd", "laplacian_spectrum", "merge_order", "multiplex_entropy", "quantum_jsd", "spectral_entropy",
           "von_neumann_entropy"]
