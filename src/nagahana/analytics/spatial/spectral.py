"""Graph spectra, spectral distances between graphs and structural drift between windows.

Normalised Laplacian
--------------------
L = I - D^-1/2 A D^-1/2 of the symmetric (weighted) adjacency; an isolated node contributes a zero
row, as for the combinatorial Laplacian. Its eigenvalues lie in [0, 2] whatever the size of the graph,
which makes spectra of windows of different sizes comparable.

Spectral measure
----------------
mu_G = (1/n) sum_i delta(lambda_i). Up to `max_dense` nodes the eigenvalues are exact
(numpy.linalg.eigvalsh). Above, stochastic Lanczos quadrature (Ubaru, Chen and Saad, SIAM J. Matrix
Anal. Appl. 38(4):1075-1099, 2017) approximates it: for Rademacher probes z_l / sqrt(n) and s Lanczos
steps with full reorthogonalisation, the Ritz values theta_k with weights tau_k^2 / n_v (tau_k the first
components of the eigenvectors of the tridiagonal matrix) form a quadrature of mu_G, so
Tr f(L) ~ (n / n_v) sum_l sum_k tau_lk^2 f(theta_lk).

Distances between graphs G and H
--------------------------------
    spectral_l2   || lambda_G[:k] - lambda_H[:k] ||_2 over the k smallest eigenvalues, the smaller
                  graph padded with zeros (isolated nodes) (Wilson and Zhu, Pattern Recognition 41(9):
                  2833-2841, 2008; dense spectra only)
    spectral_w1   Wasserstein-1 distance between mu_G and mu_H (scipy.stats.wasserstein_distance)
    heat          || h_G - h_H ||_2 over log-spaced times, h(t) = Tr exp(-t L) / n: the NetLSD heat-trace
                  signature with the empty-graph normalisation (Tsitsulin, Mottin, Karras, Bronstein and
                  Mueller, KDD 2018)
    edge_jaccard  1 - |E_G n E_H| / |E_G u E_H| over node labels (stable entity keys), the
                  identity-aware change of the edge set
Structural drift is the series of these distances between consecutive windows.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import wasserstein_distance

from nagahana.analytics.spatial.graph import TrafficGraph


def normalized_laplacian(adj: sparse.spmatrix) -> sparse.csr_matrix:
    """L = I - D^-1/2 A D^-1/2 of a symmetric adjacency (isolated nodes: zero rows)."""
    a = sparse.csr_matrix(adj, dtype=np.float64)
    a = ((a + a.T) * 0.5).tocsr()
    a.setdiag(0)
    a.eliminate_zeros()
    d = np.asarray(a.sum(axis=1)).ravel()
    inv = np.where(d > 0, 1.0 / np.sqrt(np.where(d > 0, d, 1.0)), 0.0)
    lap = sparse.diags((d > 0).astype(np.float64)) - sparse.diags(inv) @ a @ sparse.diags(inv)
    return sparse.csr_matrix(lap)


def lanczos_quadrature(op: sparse.spmatrix, *, vectors: int, steps: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """(nodes, weights) of the stochastic Lanczos quadrature of the spectral measure of `op` (weights sum to 1)."""
    n = op.shape[0]
    rng = np.random.default_rng(seed)
    nodes, weights = [], []
    for _ in range(vectors):
        z = rng.choice([-1.0, 1.0], size=n)
        q = z / np.linalg.norm(z)
        basis = np.zeros((min(steps, n), n))
        alpha, beta = [], []
        basis[0] = q
        w = op @ q
        a = float(w @ q)
        w = w - a * q
        alpha.append(a)
        for j in range(1, basis.shape[0]):
            b = float(np.linalg.norm(w))
            if b < 1e-12:
                break
            v = w / b
            v -= basis[:j].T @ (basis[:j] @ v)                           # full reorthogonalisation
            v /= np.linalg.norm(v)
            basis[j] = v
            w = op @ v - b * basis[j - 1]
            a = float(w @ v)
            w = w - a * v
            alpha.append(a)
            beta.append(b)
        t = np.diag(alpha) + np.diag(beta, 1) + np.diag(beta, -1)
        theta, u = np.linalg.eigh(t)
        nodes.append(theta)
        weights.append(u[0] ** 2)
    nd, wt = np.concatenate(nodes), np.concatenate(weights)
    return nd, wt / wt.sum()


@dataclass(frozen=True)
class Spectrum:
    """Spectral measure of one graph: exact eigenvalues (dense) or quadrature nodes with weights."""

    n: int
    nodes: np.ndarray
    weights: np.ndarray
    exact: bool


def spectrum(adj: sparse.spmatrix, *, max_dense: int = 4_000, vectors: int = 32, steps: int = 48, seed: int = 0) -> Spectrum:
    """Spectral measure of the normalised Laplacian (module docstring)."""
    lap = normalized_laplacian(adj)
    n = lap.shape[0]
    if n == 0:
        return Spectrum(0, np.zeros(0), np.zeros(0), True)
    if n <= max_dense:
        ev = np.clip(np.linalg.eigvalsh(lap.toarray()), 0.0, 2.0)
        return Spectrum(n, ev, np.full(n, 1.0 / n), True)
    nd, wt = lanczos_quadrature(lap, vectors=vectors, steps=steps, seed=seed)
    return Spectrum(n, np.clip(nd, 0.0, 2.0), wt, False)


def heat_trace(sp: Spectrum, times: np.ndarray) -> np.ndarray:
    """h(t) = Tr exp(-t L) / n on `times`."""
    t = np.asarray(times, dtype=np.float64)
    if sp.n == 0:
        return np.ones(t.size)
    return (sp.weights[None, :] * np.exp(-t[:, None] * sp.nodes[None, :])).sum(axis=1)


def spectral_l2(a: Spectrum, b: Spectrum, *, k: int) -> float:
    """L2 distance of the k smallest eigenvalues, zero-padded (exact spectra only)."""
    if not (a.exact and b.exact):
        return float("nan")
    size = max(a.n, b.n)
    ea = np.sort(np.concatenate([a.nodes, np.zeros(size - a.n)]))[:k]
    eb = np.sort(np.concatenate([b.nodes, np.zeros(size - b.n)]))[:k]
    return float(np.linalg.norm(ea - eb))


def spectral_w1(a: Spectrum, b: Spectrum) -> float:
    """Wasserstein-1 distance of the two spectral measures."""
    if a.n == 0 or b.n == 0:
        return float("nan")
    return float(wasserstein_distance(a.nodes, b.nodes, u_weights=a.weights, v_weights=b.weights))


def edge_jaccard(g: TrafficGraph, h: TrafficGraph) -> float:
    """1 - Jaccard similarity of the undirected edge sets over node labels (module docstring)."""
    if g.node_label is None or h.node_label is None:
        return float("nan")

    def edges(x: TrafficGraph) -> set[tuple[str, str]]:
        lab = x.node_label
        assert lab is not None
        a, b = lab[x.src].astype(str), lab[x.dst].astype(str)
        return {(min(p, q), max(p, q)) for p, q in zip(a.tolist(), b.tolist(), strict=True)}

    eg, eh = edges(g), edges(h)
    union = eg | eh
    return float(1.0 - len(eg & eh) / len(union)) if union else 0.0


def structural_drift(graphs: list[TrafficGraph], names: list[str], *, k: int, times: np.ndarray, max_dense: int,
                     vectors: int, steps: int, seed: int = 0) -> pd.DataFrame:
    """Distances between consecutive graphs (module docstring)."""
    sp = [spectrum(g.adjacency(weighted=False, symmetric=True), max_dense=max_dense, vectors=vectors, steps=steps,
                   seed=seed) for g in graphs]
    heats = [heat_trace(s, times) for s in sp]
    rows = []
    for i in range(1, len(graphs)):
        rows.append({
            "window": names[i], "previous": names[i - 1], "nodes": graphs[i].n, "edges": graphs[i].m,
            "spectral_l2": spectral_l2(sp[i - 1], sp[i], k=k), "spectral_w1": spectral_w1(sp[i - 1], sp[i]),
            "heat": float(np.linalg.norm(heats[i] - heats[i - 1])), "edge_jaccard_distance": edge_jaccard(graphs[i - 1], graphs[i]),
            "exact_spectra": bool(sp[i - 1].exact and sp[i].exact),
        })
    return pd.DataFrame(rows, columns=["window", "previous", "nodes", "edges", "spectral_l2", "spectral_w1", "heat",
                                       "edge_jaccard_distance", "exact_spectra"])


__all__ = ["Spectrum", "edge_jaccard", "heat_trace", "lanczos_quadrature", "normalized_laplacian", "spectral_l2",
           "spectral_w1", "spectrum", "structural_drift"]
