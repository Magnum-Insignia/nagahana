"""Unsupervised anomaly detectors used on Anomal-E's edge embeddings (AS-554).

Caville et al. (Knowledge-Based Systems 258, 2022) apply four detectors of the PyOD library to the edge
embeddings: PCA, isolation forest, CBLOF and HBOS. PyOD is not an approved dependency; the detectors are
written here from their original definitions (isolation forest from scikit-learn):

    PCA      principal-component classifier (Shyu, Chen, Sarinnapakorn and Chang, ICDM Foundations and New
             Directions of Data Mining workshop, 2003): with standardised data z and the eigenpairs
             (lambda_j, v_j) of its covariance, score(z) = sum_j (v_j^T z)^2 / lambda_j (the Mahalanobis
             distance in principal coordinates, major and minor components together)
    IF       isolation forest (Liu, Ting and Zhou, ICDM 2008); score = -score_samples (higher = more anomalous)
    CBLOF    cluster-based local outlier factor (He, Xu and Deng, Pattern Recognition Letters 24, 2003):
             k-means into k clusters sorted by size; the large clusters are the first b clusters, b the
             first boundary where the clusters before it hold at least alpha of the data or the size ratio
             |C_b| / |C_{b+1}| >= beta; a point in a large cluster scores its distance to its centre, a point
             in a small cluster its distance to the nearest large-cluster centre (PyOD defaults: k = 8,
             alpha = 0.9, beta = 5, unweighted)
    HBOS     histogram-based outlier score (Goldstein and Dengel, KI 2012): per feature a histogram of
             `bins` equal-width bins with densities d_i(x); score = sum_i -log(d_i(x) + a), a = 0.1; a value
             outside the training range by at most tol bin widths takes the edge bin, farther values the
             smallest density

Every detector is fitted on training data only. `Detector.percentile` maps a raw score to the share of
training scores at or below it (an empirical CDF), so a detection record gets scores in [0, 1] and the
contamination nu becomes the threshold 1 - nu (the decision rule of PyOD's `contamination`).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from nagahana.baselines.published.hmm.symbols import kmeans_plus_plus, lloyd
from nagahana.core.errors import InvariantViolation


def distances(x: np.ndarray, centres: np.ndarray, chunk: int = 65536) -> np.ndarray:
    """Euclidean distances [n, k] of rows to centres, computed in chunks of rows."""
    out = np.empty((x.shape[0], centres.shape[0]), dtype=np.float64)
    c2 = np.einsum("kd,kd->k", centres, centres)
    for s in range(0, x.shape[0], chunk):
        xx = x[s:s + chunk]
        sq = np.einsum("nd,nd->n", xx, xx)[:, None] - 2.0 * xx @ centres.T + c2[None, :]
        out[s:s + chunk] = np.sqrt(np.maximum(sq, 0.0))
    return out


@dataclass
class Detector:
    """One fitted detector: kind, parameters and fitted arrays, plus the sorted training scores."""

    kind: str
    params: dict[str, Any] = field(default_factory=dict)
    arrays: dict[str, np.ndarray] = field(default_factory=dict)
    train_scores: np.ndarray = field(default_factory=lambda: np.zeros(0))
    model: Any = None

    def score(self, x: np.ndarray) -> np.ndarray:
        """Raw anomaly scores (higher = more anomalous)."""
        x = np.asarray(x, dtype=np.float64)
        a = self.arrays
        if self.kind == "pca":
            z = (x - a["mean"]) / a["scale"]
            proj = z @ a["vectors"]                                              # [n, r]
            return np.sum(proj ** 2 / a["values"], axis=1)
        if self.kind == "iforest":
            if self.model is None:
                raise InvariantViolation("isolation forest state was not restored")
            return -np.asarray(self.model.score_samples(x), dtype=np.float64)
        if self.kind == "cblof":
            d = distances(x, a["centres"])                                       # [n, k]
            own = np.argmin(d, axis=1)
            large = a["large"].astype(bool)
            to_large = d[:, large].min(axis=1)
            return np.where(large[own], d[np.arange(x.shape[0]), own], to_large)
        if self.kind == "hbos":
            edges, dens, widths = a["edges"], a["density"], a["widths"]
            bins = int(self.params["bins"])
            alpha, tol = float(self.params["alpha"]), float(self.params["tol"])
            total = np.zeros(x.shape[0])
            for i in range(x.shape[1]):
                lo, hi, w = edges[i, 0], edges[i, -1], widths[i]
                idx = np.clip(np.floor((x[:, i] - lo) / w).astype(np.int64) if w > 0 else np.zeros(x.shape[0], np.int64), 0, bins - 1)
                d = dens[i, idx]
                outside = (x[:, i] < lo - tol * w) | (x[:, i] > hi + tol * w) if w > 0 else (x[:, i] != lo)
                d = np.where(outside, dens[i].min(), d)
                total += -np.log(d + alpha)
            return total
        raise InvariantViolation(f"unknown detector {self.kind!r}")

    def percentile(self, raw: np.ndarray) -> np.ndarray:
        """Share of training scores <= each raw score (in [0, 1])."""
        if self.train_scores.size == 0:
            raise InvariantViolation("detector has no training scores")
        return np.searchsorted(self.train_scores, raw, side="right") / self.train_scores.size

    def state(self) -> dict[str, Any]:
        return {"kind": self.kind, "params": self.params, "arrays": {k: v.tolist() for k, v in self.arrays.items()},
                "train_scores": self.train_scores.tolist()}

    @classmethod
    def from_state(cls, state: Mapping[str, Any], model: Any = None) -> Detector:
        return cls(str(state["kind"]), dict(state["params"]), {k: np.asarray(v, dtype=np.float64) for k, v in state["arrays"].items()},
                   np.asarray(state["train_scores"], dtype=np.float64), model)


def fit_detector(kind: str, x: np.ndarray, rng: np.random.Generator, *, params: Mapping[str, Any] | None = None,
                 seed: int = 0) -> Detector:
    """Fit a detector of the module docstring on training rows x [n, d]."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 2:
        raise InvariantViolation("detectors need a 2-D training matrix with at least two rows")
    p = dict(params or {})
    det = Detector(kind, p)
    if kind == "pca":
        mean, scale = x.mean(axis=0), x.std(axis=0)
        scale[scale == 0] = 1.0
        z = (x - mean) / scale
        cov = np.cov(z, rowvar=False).reshape(x.shape[1], x.shape[1])
        values, vectors = np.linalg.eigh(cov)
        keep = values > 1e-12 * max(values.max(), 1e-300)                       # drop null directions
        det.arrays = {"mean": mean, "scale": scale, "values": values[keep], "vectors": vectors[:, keep]}
    elif kind == "iforest":
        from nagahana.baselines.published.backends import require

        ens = require("sklearn.ensemble", needed_by="anomal-e isolation forest")
        det.params = {"n_estimators": 100, **p}
        det.model = ens.IsolationForest(n_estimators=int(det.params["n_estimators"]), random_state=seed).fit(x)
    elif kind == "cblof":
        k = int(p.get("n_clusters", 8))
        alpha, beta = float(p.get("alpha", 0.9)), float(p.get("beta", 5.0))
        det.params = {"n_clusters": k, "alpha": alpha, "beta": beta}
        k = min(k, x.shape[0])
        centres, _ = lloyd(x, kmeans_plus_plus(x, k, rng), max_iter=300, tol=1e-6)
        labels = np.argmin(distances(x, centres), axis=1)
        sizes = np.bincount(labels, minlength=k)
        order = np.argsort(-sizes, kind="stable")
        centres, sizes = centres[order], sizes[order]
        cum = np.cumsum(sizes)
        b = k
        for i in range(k - 1):
            if cum[i] >= alpha * x.shape[0] or (sizes[i + 1] > 0 and sizes[i] / sizes[i + 1] >= beta):
                b = i + 1
                break
        large = np.zeros(k)
        large[:b] = 1.0
        det.arrays = {"centres": centres, "large": large}
    elif kind == "hbos":
        bins, alpha, tol = int(p.get("bins", 10)), float(p.get("alpha", 0.1)), float(p.get("tol", 0.5))
        det.params = {"bins": bins, "alpha": alpha, "tol": tol}
        edges = np.zeros((x.shape[1], bins + 1))
        dens = np.zeros((x.shape[1], bins))
        widths = np.zeros(x.shape[1])
        for i in range(x.shape[1]):
            lo, hi = x[:, i].min(), x[:, i].max()
            if hi == lo:
                edges[i] = lo
                dens[i] = 1.0
                continue
            hist, e = np.histogram(x[:, i], bins=bins, range=(lo, hi), density=True)
            edges[i], dens[i], widths[i] = e, hist, (hi - lo) / bins
        det.arrays = {"edges": edges, "density": dens, "widths": widths}
    else:
        raise InvariantViolation(f"unknown detector {kind!r} (pca, iforest, cblof, hbos)")
    det.train_scores = np.sort(det.score(x))
    return det
