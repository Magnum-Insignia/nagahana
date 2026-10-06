"""Probability calibration of a score, fitted on validation units only.

Every calibrator maps a real-valued score f (the logit of a logistic regression) to a probability.

    platt        p = sigmoid(a f + b), fitted by minimising the cross-entropy against Platt's smoothed
                 targets t+ = (N+ + 1) / (N+ + 2) for positives and t- = 1 / (N- + 2) for negatives
                 (Platt, "Probabilistic outputs for support vector machines and comparisons to regularized
                 likelihood methods", in Advances in Large Margin Classifiers, MIT Press, 1999), by the
                 Newton method with backtracking of Lin, Lin and Weng ("A note on Platt's probabilistic
                 outputs for support vector machines", Machine Learning 68(3), 2007), started from a = 0,
                 b = log((N+ + 1) / (N- + 1)).
    temperature  p = sigmoid(f / T), T > 0 fitted by minimising the cross-entropy on the hard labels
                 (Guo, Pleiss, Sun and Weinberger, "On calibration of modern neural networks", ICML 2017,
                 arXiv:1706.04599): a single-parameter, order-preserving rescaling.
    isotonic     a non-decreasing function m minimising sum_i w_i (y_i - m(f_i))^2, by pool-adjacent-
                 violators over the distinct scores (Ayer et al., Ann. Math. Statist. 26(4), 1955; Barlow et
                 al., Statistical Inference under Order Restrictions, Wiley 1972), used for calibration by
                 Zadrozny and Elkan ("Transforming classifier scores into accurate multiclass probability
                 estimates", KDD 2002). The fitted step function is kept as the first and last score of every
                 block and evaluated by linear interpolation between those points, clipped outside the fitted
                 range (the representation of scikit-learn's IsotonicRegression, so the two can be compared).
    none         p = sigmoid(f).

Platt and temperature are strictly increasing, so they keep the ranking of the scores; isotonic is
non-decreasing and may tie scores that it pools.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from nagahana.core.errors import InvariantViolation

from .logistic import expit

_MAX_NEWTON = 200
_MIN_STEP = 1e-10
_SIGMA = 1e-12


def _xent(z: np.ndarray, t: np.ndarray, w: np.ndarray) -> float:
    # sum_i w_i [log(1 + exp(z_i)) - t_i z_i], stable for every z
    return float(np.sum(w * (np.maximum(z, 0.0) + np.log1p(np.exp(-np.abs(z))) - t * z)))


def _newton_2d(f: np.ndarray, t: np.ndarray, w: np.ndarray, a0: float, b0: float, *, fit_b: bool) -> tuple[float, float, int]:
    """Minimise sum_i w_i xent(sigmoid(a f_i + b), t_i) over (a, b) (or a alone) by damped Newton steps."""
    a, b = a0, b0
    fval = _xent(a * f + b, t, w)
    it = 0
    while it < _MAX_NEWTON:
        it += 1
        p = expit(a * f + b)
        r = w * (p - t)
        q = w * p * (1.0 - p)
        g = np.array([np.sum(f * r), np.sum(r)])
        h = np.array([[np.sum(f * f * q) + _SIGMA, np.sum(f * q)], [np.sum(f * q), np.sum(q) + _SIGMA]])
        if not fit_b:
            g, h = g[:1], h[:1, :1]
        if np.max(np.abs(g)) < 1e-11 * max(1.0, float(np.sum(w))):
            break
        step = -np.linalg.solve(h, g)
        gd = float(g @ step)
        s = 1.0
        while s >= _MIN_STEP:
            na = a + s * step[0]
            nb = b + s * step[1] if fit_b else b
            nval = _xent(na * f + nb, t, w)
            if nval < fval + 1e-4 * s * gd:
                a, b, fval = na, nb, nval
                break
            s /= 2.0
        else:
            break
    return a, b, it


def _pav(y: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pool-adjacent-violators on values y with weights w (already in increasing score order).

    Returns (block values, block start index, block end index) with end exclusive.
    """
    vals: list[float] = []
    wts: list[float] = []
    starts: list[int] = []
    for i, (yi, wi) in enumerate(zip(y.tolist(), w.tolist(), strict=True)):
        vals.append(yi)
        wts.append(wi)
        starts.append(i)
        # merge while the last two blocks violate monotonicity
        while len(vals) > 1 and vals[-2] > vals[-1]:
            wv = wts[-2] + wts[-1]
            vals[-2] = (vals[-2] * wts[-2] + vals[-1] * wts[-1]) / wv
            wts[-2] = wv
            vals.pop()
            wts.pop()
            starts.pop()
    s = np.asarray(starts, dtype=np.int64)
    e = np.r_[s[1:], y.size].astype(np.int64)
    return np.asarray(vals, dtype=np.float64), s, e


@dataclass
class Calibrator:
    """A fitted calibration map (module docstring). `params` holds the arrays that define it."""

    method: str
    params: dict[str, np.ndarray] = field(default_factory=dict)
    fitted_on: int = 0

    def __post_init__(self) -> None:
        if self.method not in ("platt", "isotonic", "temperature", "none"):
            raise InvariantViolation(f"unknown calibration method {self.method!r}")

    @classmethod
    def fit(cls, score: np.ndarray, y: np.ndarray, method: str, *, weight: np.ndarray | None = None) -> Calibrator:
        """Fit on validation scores (logits) and labels in {0, 1}."""
        f = np.asarray(score, dtype=np.float64).ravel()
        yy = np.asarray(y, dtype=np.float64).ravel()
        w = np.ones_like(f) if weight is None else np.asarray(weight, dtype=np.float64).ravel()
        if f.shape != yy.shape or w.shape != f.shape:
            raise InvariantViolation("calibration inputs must have equal lengths")
        if not np.all(np.isfinite(f)):
            raise InvariantViolation("calibration scores must be finite")
        if np.any((yy != 0) & (yy != 1)):
            raise InvariantViolation("calibration labels must be 0 or 1")
        if method == "none":
            return cls("none", {}, int(f.size))
        n_pos = float(np.sum(w[yy == 1]))
        n_neg = float(np.sum(w[yy == 0]))
        if n_pos <= 0 or n_neg <= 0:
            raise InvariantViolation(f"{method} calibration needs both classes among the validation units")
        if method == "platt":
            t = np.where(yy == 1, (n_pos + 1.0) / (n_pos + 2.0), 1.0 / (n_neg + 2.0))
            a, b, it = _newton_2d(f, t, w, 0.0, float(np.log((n_pos + 1.0) / (n_neg + 1.0))), fit_b=True)
            return cls("platt", {"a": np.asarray(a), "b": np.asarray(b), "iterations": np.asarray(it)}, int(f.size))
        if method == "temperature":
            a, _b, it = _newton_2d(f, yy, w, 1.0, 0.0, fit_b=False)
            if a <= 0:
                raise InvariantViolation("temperature scaling found a non-positive inverse temperature; the scores "
                                         "rank the validation units in the wrong order")
            return cls("temperature", {"a": np.asarray(a), "iterations": np.asarray(it)}, int(f.size))
        # isotonic: pool equal scores, then PAV over the distinct scores
        order = np.argsort(f, kind="mergesort")
        fs, ys, ws = f[order], yy[order], w[order]
        uniq, first = np.unique(fs, return_index=True)
        wsum = np.add.reduceat(ws, first)
        ybar = np.add.reduceat(ws * ys, first) / wsum
        vals, s, e = _pav(ybar, wsum)
        # keep the first and last distinct score of each block
        xs: list[float] = []
        vs: list[float] = []
        for v, a0, b0 in zip(vals.tolist(), s.tolist(), e.tolist(), strict=True):
            xs.append(float(uniq[a0]))
            vs.append(v)
            if b0 - 1 > a0:
                xs.append(float(uniq[b0 - 1]))
                vs.append(v)
        return cls("isotonic", {"x": np.asarray(xs), "y": np.asarray(vs)}, int(f.size))

    def transform(self, score: np.ndarray) -> np.ndarray:
        """Calibrated probabilities for scores (logits); float64, in [0, 1]."""
        f = np.asarray(score, dtype=np.float64)
        if self.method == "none":
            return expit(f)
        if self.method == "platt":
            return expit(float(self.params["a"]) * f + float(self.params["b"]))
        if self.method == "temperature":
            return expit(float(self.params["a"]) * f)
        x, y = self.params["x"], self.params["y"]
        return np.clip(np.interp(f, x, y), 0.0, 1.0)

    def state(self) -> dict[str, np.ndarray]:
        out = {f"p.{k}": np.asarray(v) for k, v in self.params.items()}
        out["method"] = np.asarray(self.method)
        out["fitted_on"] = np.asarray(self.fitted_on, dtype=np.int64)
        return out

    @classmethod
    def from_state(cls, s: dict[str, np.ndarray]) -> Calibrator:
        params = {k[2:]: np.asarray(v) for k, v in s.items() if k.startswith("p.")}
        return cls(str(np.asarray(s["method"])), params, int(np.asarray(s["fitted_on"])))


__all__ = ["Calibrator"]
