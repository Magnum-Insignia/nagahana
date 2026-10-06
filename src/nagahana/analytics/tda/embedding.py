"""Sliding-window (Takens) embeddings of time series and the periodicity score of their H_1.

Sliding window (Perea and Harer, Foundations of Computational Mathematics 15:799-838, 2015)
---------------------------------------------------------------------------------------
SW_{d, tau} x(t) = (x(t), x(t + tau), ..., x(t + (d - 1) tau)) in R^d, for t = 0, s, 2s, ... (stride s).
For a periodic signal the window cloud lies on a closed curve, so its Vietoris-Rips H_1 has one long
bar; Perea and Harer show that the bar is longest when the window length (d - 1) tau is close to
the period times (d - 1) / d.

Periodicity score (SW1PerS; Perea, Deckard, Haase and Harer, BMC Bioinformatics 16:257, 2015)
------------------------------------------------------------------------------------------
Every window vector is centred (its mean removed) and scaled to unit norm, which removes trends in
level and amplitude and puts the cloud on the unit sphere. With mp the largest H_1 persistence,
score = 1 - mp / sqrt(3) in [0, 1]: sqrt(3) is the side of the equilateral triangle inscribed in a
unit circle, the largest H_1 persistence a Rips complex of points on a great circle reaches. Lower
scores mean stronger periodicity.

Delay by average mutual information (Fraser and Swinney, Phys. Rev. A 33:1134-1140, 1986)
-------------------------------------------------------------------------------------
AMI(tau) = I(x_t; x_{t+tau}) with an equal-width histogram estimator (sqrt(n) bins, at most 64);
the delay is the first local minimum, or the first tau where AMI falls below AMI(0) / e when there
is no local minimum within `max_delay`.

Dimension by false nearest neighbours (Kennel, Brown and Abarbanel, Phys. Rev. A 45:3403-3411, 1992)
-------------------------------------------------------------------------------------------------
For each embedded point its nearest neighbour in dimension d is false when adding coordinate d + 1
separates them: |x_{i+d tau} - x_{j+d tau}| / R_d(i, j) > r_tol, or R_{d+1}(i, j) / sigma_x > a_tol
(r_tol = 15, a_tol = 2 as in the paper). The dimension is the smallest d whose false share is below
`threshold`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from nagahana.analytics.information import mutual_information_discrete
from nagahana.analytics.tda.complex import maxmin_landmarks, rips_complex
from nagahana.analytics.tda.diagrams import Diagram
from nagahana.analytics.tda.homology import persistence


def sliding_window(x: np.ndarray, *, dimension: int, delay: int, stride: int = 1) -> np.ndarray:
    """Window cloud [m, dimension] of a 1-D series (module docstring)."""
    a = np.asarray(x, dtype=np.float64).reshape(-1)
    if dimension < 1 or delay < 1 or stride < 1:
        raise ValueError("dimension, delay and stride must be >= 1")
    span = (dimension - 1) * delay
    if a.size <= span:
        raise ValueError(f"series of length {a.size} is too short for dimension {dimension} and delay {delay}")
    starts = np.arange(0, a.size - span, stride)
    return a[starts[:, None] + delay * np.arange(dimension)[None, :]]


def average_mutual_information(x: np.ndarray, *, max_delay: int) -> np.ndarray:
    """AMI(tau) for tau = 0 ... max_delay (nats), equal-width histogram estimator."""
    a = np.asarray(x, dtype=np.float64).reshape(-1)
    a = a[np.isfinite(a)]
    if a.size < 4:
        raise ValueError("series too short for mutual information")
    bins = int(min(64, max(2, np.sqrt(a.size))))
    edges = np.linspace(a.min(), a.max() + 1e-12 * (abs(a.max()) + 1.0), bins + 1)
    codes = np.clip(np.searchsorted(edges, a, side="right") - 1, 0, bins - 1)
    out = np.zeros(max_delay + 1)
    for tau in range(max_delay + 1):
        if a.size - tau < 2:
            out[tau:] = np.nan
            break
        out[tau] = mutual_information_discrete(codes[: a.size - tau], codes[tau:])
    return out


def delay_by_ami(x: np.ndarray, *, max_delay: int) -> tuple[int, np.ndarray]:
    """(delay, AMI curve) by the first local minimum rule (module docstring)."""
    ami = average_mutual_information(x, max_delay=max_delay)
    for tau in range(1, max_delay):
        if np.isfinite(ami[tau + 1]) and ami[tau] < ami[tau - 1] and ami[tau] <= ami[tau + 1]:
            return tau, ami
    below = np.flatnonzero(ami[1:] < ami[0] / np.e)
    return (int(below[0]) + 1 if below.size else max(1, max_delay)), ami


def false_nearest_neighbours(x: np.ndarray, *, delay: int, max_dimension: int, r_tol: float = 15.0,
                             a_tol: float = 2.0) -> np.ndarray:
    """Share of false nearest neighbours for d = 1 ... max_dimension (module docstring)."""
    a = np.asarray(x, dtype=np.float64).reshape(-1)
    sd = float(a.std())
    out = np.full(max_dimension, np.nan)
    for d in range(1, max_dimension + 1):
        m = a.size - d * delay                                           # points that have a (d + 1)-th coordinate
        if m < 3:
            break
        emb = a[np.arange(m)[:, None] + delay * np.arange(d)[None, :]]
        nxt = a[np.arange(m) + d * delay]
        dist, idx = cKDTree(emb).query(emb, k=2)
        r_d = dist[:, 1]
        j = idx[:, 1]
        extra = np.abs(nxt - nxt[j])
        with np.errstate(divide="ignore", invalid="ignore"):
            crit1 = np.where(r_d > 0, extra / r_d, np.inf) > r_tol
            crit2 = (np.sqrt(r_d ** 2 + extra ** 2) / sd > a_tol) if sd > 0 else np.zeros(m, dtype=bool)
        out[d - 1] = float(np.mean(crit1 | crit2))
    return out


def dimension_by_fnn(x: np.ndarray, *, delay: int, max_dimension: int, threshold: float = 0.01) -> tuple[int, np.ndarray]:
    """(dimension, FNN curve): smallest d whose false share is below `threshold` (else the minimiser)."""
    fnn = false_nearest_neighbours(x, delay=delay, max_dimension=max_dimension)
    ok = np.flatnonzero(fnn < threshold)
    if ok.size:
        return int(ok[0]) + 1, fnn
    return int(np.nanargmin(fnn)) + 1 if np.isfinite(fnn).any() else 2, fnn


@dataclass(frozen=True)
class Periodicity:
    """SW1PerS result: score in [0, 1] (lower = more periodic), the largest H_1 persistence, the H_1 diagram.

    points: window vectors used (maxmin landmarks when the cloud was larger); covering_radius: their
    covering radius on the normalised cloud, so the H_1 diagram is within 2 * covering_radius of the full one.
    """

    score: float
    max_persistence: float
    diagram: Diagram
    dimension: int
    delay: int
    points: int
    covering_radius: float


def periodicity_score(x: np.ndarray, *, dimension: int, delay: int, stride: int = 1, max_points: int = 150) -> Periodicity:
    """SW1PerS periodicity score of a series (module docstring).

    The normalised window cloud lies on the unit sphere, where the Rips complex up to sqrt(3) is close
    to complete; clouds larger than `max_points` are reduced to maxmin landmarks first.
    """
    cloud = sliding_window(x, dimension=dimension, delay=delay, stride=stride)
    cloud = cloud - cloud.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(cloud, axis=1, keepdims=True)
    cloud = np.where(norms > 0, cloud / np.where(norms > 0, norms, 1.0), 0.0)
    eps = 0.0
    if cloud.shape[0] > max_points:
        idx, eps = maxmin_landmarks(cloud, max_points)
        cloud = cloud[idx]
    cx = rips_complex(cloud, max_dim=1, max_radius=float(np.sqrt(3.0)) + 1e-9)
    h1 = persistence(cx, max_dim=1).diagrams[1]
    pers = np.where(np.isfinite(h1.death), h1.death, np.sqrt(3.0)) - h1.birth
    mp = float(pers.max()) if pers.size else 0.0
    return Periodicity(score=float(1.0 - mp / np.sqrt(3.0)), max_persistence=mp, diagram=h1, dimension=dimension,
                       delay=delay, points=int(cloud.shape[0]), covering_radius=float(eps))


__all__ = [
    "Periodicity", "average_mutual_information", "delay_by_ami", "dimension_by_fnn", "false_nearest_neighbours",
    "periodicity_score", "sliding_window",
]
