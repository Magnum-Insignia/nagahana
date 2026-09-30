"""Logistic-regression baseline on the same features (required by the problem statement).

"Benchmark results comparing model performance (F1 score, precision, recall, false positive rate)
against a logistic regression baseline trained on the same features" (CLAUDE.md).

Model:   p(y = 1 | x) = σ(wᵀx + b)
Loss:    mean binary cross-entropy + (λ/2)·‖w‖²       (λ required: a reported choice)
Solver:  L-BFGS (full batch; deterministic for a given input)

Why implement it here rather than import scikit-learn: it runs on the same tensors, device and data
pipeline as NagaHana, so "the same features" holds by construction. (scikit-learn is not in the
approved stack, D-09.)

Feature standardisation is part of the baseline's protocol: it is fitted on training data only and
stored with the model, so no statistics leak from evaluation data.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class LogisticBaseline:
    """Fitted parameters of the baseline."""

    w: torch.Tensor
    b: torch.Tensor
    mean: torch.Tensor
    std: torch.Tensor

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """P(y = 1 | x)."""
        z = (x - self.mean) / self.std
        return torch.sigmoid(z @ self.w + self.b)

    def predict(self, x: torch.Tensor, *, threshold: float) -> torch.Tensor:
        """0/1 decisions at an explicit threshold (the operating point is always reported)."""
        return (self.predict_proba(x) >= threshold).long()


def fit_logistic(x: torch.Tensor, y: torch.Tensor, *, l2: float, max_iter: int) -> LogisticBaseline:
    """Fit on training data. x: [N, D] float; y: [N] in {0, 1}."""
    if x.dim() != 2 or y.shape != (x.shape[0],):
        raise ValueError("x must be [N, D] and y must be [N]")
    if l2 < 0 or max_iter < 1:
        raise ValueError("l2 must be >= 0 and max_iter >= 1")
    x = x.double()
    yf = y.double()
    mean = x.mean(dim=0)
    std = x.std(dim=0).clamp_min(1e-12)
    z = (x - mean) / std
    w = torch.zeros(x.shape[1], dtype=torch.float64, requires_grad=True)
    b = torch.zeros((), dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure() -> torch.Tensor:
        opt.zero_grad()
        logits = z @ w + b
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, yf) + 0.5 * l2 * (w @ w)
        loss.backward()
        return loss

    opt.step(closure)
    return LogisticBaseline(w.detach(), b.detach(), mean, std)
