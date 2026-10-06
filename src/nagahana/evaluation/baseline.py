"""Logistic-regression baseline on the same features: the evaluation's entry point to the LR family.

The required comparison (F1, precision, recall and false-positive rate against a logistic regression
trained on the same features, plus the forecast, stage and next-state comparisons of the evaluation
chapter) is produced by the LR family of `nagahana.baselines.lr`:

    corpus = build_corpus(sources, manifest, cfg)            # rows from NagaHana's own windows
    run = run_protocol(corpus, load_config(path), protocol="P1")
    run.outputs                                               # ModelOutputs of every split (meta["split"])
    run.references["persistence"]                             # reference forecasts for skill scores

This module re-exports that entry point (loaded on first access) and keeps `fit_logistic`, the direct
fit of the detection objective on a feature tensor, for callers that already hold a feature matrix:

    p(y = 1 | x) = sigmoid(w.x + b) on x standardised with the training mean and the unbiased standard
    deviation (fitted on the training rows only and stored with the model);
    loss = mean binary cross-entropy + (l2 / 2) ||w||^2, intercept unpenalised;
    solver: the L-BFGS driver of baselines/lr/logistic.py (strong Wolfe line search, float64), at most
    max_iter iterations, deterministic for a given input.

Everything here runs on PyTorch and NumPy. scikit-learn belongs to the optional [baselines] extra (D-55)
and is used only by the LR family's cross-check, which refits the same objective with
sklearn.linear_model.LogisticRegression and asserts agreement.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import TYPE_CHECKING, Any

import torch

from nagahana.baselines.lr.config import SolverConfig
from nagahana.baselines.lr.logistic import RowSource, lbfgs_minimise, logistic_value_grad

if TYPE_CHECKING:
    from nagahana.baselines.lr.config import LRBaselineConfig, load_config
    from nagahana.baselines.lr.corpus import LRCorpus, build_corpus
    from nagahana.baselines.lr.family import LRFamily, ProtocolRun, run_protocol

_LAZY: dict[str, str] = {
    "LRBaselineConfig": "nagahana.baselines.lr.config", "load_config": "nagahana.baselines.lr.config",
    "LRCorpus": "nagahana.baselines.lr.corpus", "build_corpus": "nagahana.baselines.lr.corpus",
    "LRFamily": "nagahana.baselines.lr.family", "ProtocolRun": "nagahana.baselines.lr.family",
    "run_protocol": "nagahana.baselines.lr.family",
}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        return getattr(import_module(_LAZY[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


@dataclass
class LogisticBaseline:
    """Fitted parameters of `fit_logistic`: weights, intercept and the training standardisation."""

    w: torch.Tensor
    b: torch.Tensor
    mean: torch.Tensor
    std: torch.Tensor

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """P(y = 1 | x)."""
        z = (x.double() - self.mean) / self.std
        return torch.sigmoid(z @ self.w + self.b)

    def predict(self, x: torch.Tensor, *, threshold: float) -> torch.Tensor:
        """0/1 decisions at an explicit threshold (the operating point is always reported)."""
        return (self.predict_proba(x) >= threshold).long()


def fit_logistic(x: torch.Tensor, y: torch.Tensor, *, l2: float, max_iter: int) -> LogisticBaseline:
    """Fit on training data (module docstring). x: [N, D] float; y: [N] in {0, 1}."""
    if x.dim() != 2 or y.shape != (x.shape[0],):
        raise ValueError("x must be [N, D] and y must be [N]")
    if l2 < 0 or max_iter < 1:
        raise ValueError("l2 must be >= 0 and max_iter >= 1")
    x64 = x.double()
    mean = x64.mean(dim=0)
    std = x64.std(dim=0).clamp_min(1e-12)
    z = ((x64 - mean) / std).numpy()
    yf = y.double().numpy()
    src = RowSource.from_arrays(z, yf)
    rows = max(1, z.shape[0])
    cfg = SolverConfig(method="lbfgs", max_iter=max_iter, check_every=min(10, max_iter))
    theta, _info = lbfgs_minimise(lambda th: logistic_value_grad(th, src, l2, rows, float(rows)),
                                  torch.zeros(z.shape[1] + 1, dtype=torch.float64), cfg)
    return LogisticBaseline(theta[:-1].clone(), theta[-1].clone(), mean, std)


__all__ = ["LRBaselineConfig", "LRCorpus", "LRFamily", "LogisticBaseline", "ProtocolRun", "build_corpus", "fit_logistic",
           "load_config", "run_protocol"]
