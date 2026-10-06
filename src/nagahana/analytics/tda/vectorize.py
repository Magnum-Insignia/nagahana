"""Vectorisations of persistence diagrams: Betti and Euler curves, landscapes, persistence images.

Betti curve
-----------
beta_k(t) = #{i : b_i <= t < d_i}, evaluated on a grid by binary search over sorted births and deaths.

Euler characteristic curve
--------------------------
chi(t) = sum_d (-1)^d #{d-simplices with value <= t}. By the Euler-Poincare formula it equals
sum_k (-1)^k beta_k(t) of the complex at t, which the tests use to check the homology engine.

Persistence landscapes (Bubenik, JMLR 16:77-102, 2015)
----------------------------------------------------
lambda_j(t) = j-th largest of max(0, min(t - b_i, d_i - t)) over the bars; j = 1 ... k, sampled on a
grid. Essential bars are capped at a given value first (a landscape needs finite bars).

Persistence images (Adams, Emerson, Kirby, Neville, Peterson, Shipman, Chepushtanova, Hanson, Motta and
Ziegelmeier, JMLR 18(8):1-35, 2017)
------------------------------------------------------------------------------------------------
Points (b, d) become (b, p = d - b). The surface rho(x, y) = sum_i w(p_i) N((x, y); (b_i, p_i), sigma^2 I)
with the piecewise-linear weight w(p) = 0 for p <= 0, p / p_max for 0 < p < p_max, 1 for p >= p_max
is integrated exactly over each pixel; the Gaussian is separable, so a pixel [x0, x1] x [y0, y1] gets
w (Phi((x1 - b) / sigma) - Phi((x0 - b) / sigma)) (Phi((y1 - p) / sigma) - Phi((y0 - p) / sigma)).
"""

from __future__ import annotations

import numpy as np
from scipy import special

from nagahana.analytics.tda.complex import FilteredComplex
from nagahana.analytics.tda.diagrams import Diagram


def grid_for(diagrams: list[Diagram], resolution: int, *, cap: float | None = None) -> np.ndarray:
    """An evenly spaced grid covering every finite birth and death (and `cap`, when given)."""
    if resolution < 2:
        raise ValueError("resolution must be >= 2")
    pts = [np.concatenate([dg.birth, dg.death[np.isfinite(dg.death)]]) for dg in diagrams]
    allv = np.concatenate(pts) if pts else np.zeros(0)
    if cap is not None:
        allv = np.append(allv, cap)
    if allv.size == 0:
        return np.linspace(0.0, 1.0, resolution)
    lo, hi = float(allv.min()), float(allv.max())
    if hi <= lo:
        hi = lo + 1.0
    return np.linspace(lo, hi, resolution)


def betti_curve(diagram: Diagram, grid: np.ndarray) -> np.ndarray:
    """beta_k(t) on `grid` (int64)."""
    t = np.asarray(grid, dtype=np.float64)
    b = np.sort(diagram.birth)
    d = np.sort(diagram.death)
    return (np.searchsorted(b, t, side="right") - np.searchsorted(d, t, side="right")).astype(np.int64)


def euler_curve(cx: FilteredComplex, grid: np.ndarray) -> np.ndarray:
    """chi(t) on `grid` from the simplex counts of the complex (int64)."""
    t = np.asarray(grid, dtype=np.float64)
    out = np.zeros(t.shape, dtype=np.int64)
    for dim, vals in enumerate(cx.values):
        out += (-1) ** dim * np.searchsorted(np.sort(vals), t, side="right")
    return out


def landscape(diagram: Diagram, grid: np.ndarray, *, k: int, cap: float | None = None) -> np.ndarray:
    """Landscape levels lambda_1 ... lambda_k on `grid`: float64 [k, len(grid)]."""
    if k < 1:
        raise ValueError("k must be >= 1")
    t = np.asarray(grid, dtype=np.float64)
    dg = diagram if cap is None else diagram.capped(cap)
    b, d = dg.finite()
    out = np.zeros((k, t.size))
    if b.size == 0:
        return out
    tent = np.maximum(0.0, np.minimum(t[None, :] - b[:, None], d[:, None] - t[None, :]))   # [m, T]
    top = min(k, b.size)
    part = -np.sort(-tent, axis=0)[:top]                                 # largest first, [top, T]
    out[:top] = part
    return out


def landscape_norm(levels: np.ndarray, grid: np.ndarray, *, p: float = 2.0) -> float:
    """L^p norm of a landscape (sum over levels of the integral of |lambda_j|^p, then ^(1/p)), trapezoid rule."""
    t = np.asarray(grid, dtype=np.float64)
    return float(np.trapezoid(np.abs(levels) ** p, t, axis=1).sum() ** (1.0 / p))


def persistence_image(
    diagram: Diagram,
    *,
    resolution: int,
    sigma: float,
    birth_range: tuple[float, float],
    pers_range: tuple[float, float],
    p_max: float | None = None,
    cap: float | None = None,
) -> np.ndarray:
    """Persistence image [resolution (persistence, low to high), resolution (birth)] (module docstring)."""
    if resolution < 1 or sigma <= 0:
        raise ValueError("resolution must be >= 1 and sigma > 0")
    dg = diagram if cap is None else diagram.capped(cap)
    b, d = dg.finite()
    pers = d - b
    pm = float(p_max) if p_max is not None else (float(pers.max()) if pers.size else 1.0)
    w = np.where(pers <= 0, 0.0, np.where(pers < pm, pers / pm, 1.0)) if pm > 0 else np.ones_like(pers)
    xe = np.linspace(birth_range[0], birth_range[1], resolution + 1)
    ye = np.linspace(pers_range[0], pers_range[1], resolution + 1)
    if b.size == 0:
        return np.zeros((resolution, resolution))
    cx = special.ndtr((xe[None, :] - b[:, None]) / sigma)                # [m, R + 1]
    cy = special.ndtr((ye[None, :] - pers[:, None]) / sigma)
    px = np.diff(cx, axis=1)                                             # [m, R]
    py = np.diff(cy, axis=1)
    return np.einsum("m,my,mx->yx", w, py, px)


__all__ = ["betti_curve", "euler_curve", "grid_for", "landscape", "landscape_norm", "persistence_image"]
