"""Preconditioning of the L-BFGS solves by fixed quadratic upper bounds of the Hessian.

Every objective of the family has the form F(theta) = (1 / S) sum_r c_r l(A_r theta) + (lambda / 2) ||P theta||^2
with a convex loss l whose curvature is bounded: l'' <= 1/4 for the logistic loss, and
d^2 l / d eta^2 = diag(p) - p p^T <= (1/2)(I - 1 1^T / P) for the softmax cross-entropy over P classes
(Boehning and Lindsay, "Monotonicity of quadratic-approximation algorithms", Annals of the Institute of
Statistical Mathematics 40(4), 1988; Boehning, "Multinomial logistic regression algorithm", same journal
44(1), 1992). Hence the Hessian of F is bounded above by a fixed matrix B built from the weighted Gram
matrix of the design and the penalty. A standardised window design is strongly collinear, so F is badly
conditioned for small lambda and plain L-BFGS needs thousands of iterations; with the change of variables

    theta = L^{-T} phi,    L L^T = B,    grad_phi F = L^{-1} grad_theta F,

the Hessian in phi is at most the identity and close to a multiple of it, and L-BFGS converges in tens of
iterations. The change of variables is exact: the minimiser is the same point (Nocedal and Wright,
Numerical Optimization, 2nd ed., Springer 2006, Section 5.1 on preconditioning; the L-BFGS initial matrix
plays the role of B^{-1}). The bound only has to be symmetric positive definite, so it may be computed on a
seeded subsample of the rows (`gram_sample`); its quality affects speed, never the solution.

Transforms

    CholeskyPreconditioner   a dense B (binary LR, hazard model with shared coefficients)
    DiagonalPreconditioner   a diagonal B (softmax intercepts)
    SoftmaxPreconditioner    the class-coupled block of a symmetric multinomial model: for coefficients
                             W [D, P], B_W = (1/2)(I - 1 1^T / P) (x) G + lambda I. In the rotated class basis
                             Q (last column 1 / sqrt(P)) it is block-diagonal: (1/2) G + lambda I on the first
                             P - 1 columns and lambda I on the last, the direction in which the softmax is
                             invariant and only the penalty curves F
    BlockPreconditioner      a block-diagonal composition of the above over slices of theta
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import torch

from nagahana.core.errors import InvariantViolation


class Preconditioner(Protocol):
    """A linear change of variables theta = M phi with its adjoint for gradients."""

    def to_theta(self, phi: torch.Tensor) -> torch.Tensor: ...

    def to_phi(self, theta: torch.Tensor) -> torch.Tensor: ...

    def grad_to_phi(self, g: torch.Tensor) -> torch.Tensor: ...


def stable_cholesky(b: np.ndarray) -> torch.Tensor:
    """Lower Cholesky factor of a symmetric positive semi-definite matrix, with the smallest diagonal jitter
    eps * mean(diag) (eps = 1e-12, 1e-10, ...) that makes the factorisation succeed."""
    b = 0.5 * (np.asarray(b, dtype=np.float64) + np.asarray(b, dtype=np.float64).T)
    scale = float(np.mean(np.diag(b))) if b.size else 1.0
    scale = scale if scale > 0 else 1.0
    eye = np.eye(b.shape[0])
    for eps in (0.0, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4):
        chol, info = torch.linalg.cholesky_ex(torch.from_numpy(b + eps * scale * eye))
        if int(info) == 0:
            return chol
    raise InvariantViolation("the preconditioning bound is not positive definite")


@dataclass
class CholeskyPreconditioner:
    """theta = L^{-T} phi for B = L L^T."""

    chol: torch.Tensor

    @classmethod
    def from_matrix(cls, b: np.ndarray) -> CholeskyPreconditioner:
        return cls(stable_cholesky(b))

    def to_theta(self, phi: torch.Tensor) -> torch.Tensor:
        return torch.linalg.solve_triangular(self.chol.T, phi[:, None], upper=True)[:, 0]

    def to_phi(self, theta: torch.Tensor) -> torch.Tensor:
        return self.chol.T @ theta

    def grad_to_phi(self, g: torch.Tensor) -> torch.Tensor:
        return torch.linalg.solve_triangular(self.chol, g[:, None], upper=False)[:, 0]


@dataclass
class DiagonalPreconditioner:
    """theta = phi / sqrt(b) for a positive diagonal B = diag(b)."""

    root: torch.Tensor

    @classmethod
    def from_diagonal(cls, b: np.ndarray) -> DiagonalPreconditioner:
        b = np.asarray(b, dtype=np.float64)
        return cls(torch.from_numpy(np.sqrt(np.where(b > 0, b, 1.0))))

    def to_theta(self, phi: torch.Tensor) -> torch.Tensor:
        return phi / self.root

    def to_phi(self, theta: torch.Tensor) -> torch.Tensor:
        return theta * self.root

    def grad_to_phi(self, g: torch.Tensor) -> torch.Tensor:
        return g / self.root


def class_rotation(p: int) -> np.ndarray:
    """Orthogonal Q [P, P] whose last column is 1 / sqrt(P) (QR of [1 / sqrt(P), e_1 ... e_{P-1}])."""
    if p < 1:
        raise InvariantViolation("a class rotation needs at least one class")
    m = np.concatenate([np.full((p, 1), 1.0 / np.sqrt(p)), np.eye(p)[:, : p - 1]], axis=1)
    q, _r = np.linalg.qr(m)
    q = np.concatenate([q[:, 1:], q[:, :1]], axis=1)
    if q[0, -1] < 0:
        q[:, -1] = -q[:, -1]
    return q


@dataclass
class SoftmaxPreconditioner:
    """The class-coupled block W [D, P] (row-major flattening) of a symmetric multinomial model (module docstring)."""

    chol: torch.Tensor           # L1 with L1 L1^T = (1/2) G + lambda I, [D, D]
    q: torch.Tensor              # [P, P]
    sqrt_lam: float
    d: int
    p: int

    @classmethod
    def build(cls, gram: np.ndarray, lam: float, p: int) -> SoftmaxPreconditioner:
        if lam <= 0:
            raise InvariantViolation("the softmax preconditioner needs lambda > 0 (the invariant direction is curved by the penalty only)")
        d = gram.shape[0]
        return cls(stable_cholesky(0.5 * gram + lam * np.eye(d)), torch.from_numpy(class_rotation(p)), float(np.sqrt(lam)), d, p)

    def to_theta(self, phi: torch.Tensor) -> torch.Tensor:
        f = phi.view(self.d, self.p)
        u = torch.empty_like(f)
        u[:, :-1] = torch.linalg.solve_triangular(self.chol.T, f[:, :-1], upper=True)
        u[:, -1] = f[:, -1] / self.sqrt_lam
        return (u @ self.q.T).reshape(-1)

    def to_phi(self, theta: torch.Tensor) -> torch.Tensor:
        u = theta.view(self.d, self.p) @ self.q
        f = torch.empty_like(u)
        f[:, :-1] = self.chol.T @ u[:, :-1]
        f[:, -1] = u[:, -1] * self.sqrt_lam
        return f.reshape(-1)

    def grad_to_phi(self, g: torch.Tensor) -> torch.Tensor:
        gu = g.view(self.d, self.p) @ self.q
        out = torch.empty_like(gu)
        out[:, :-1] = torch.linalg.solve_triangular(self.chol, gu[:, :-1], upper=False)
        out[:, -1] = gu[:, -1] / self.sqrt_lam
        return out.reshape(-1)


@dataclass
class BlockPreconditioner:
    """Block-diagonal composition over consecutive slices of theta."""

    blocks: Sequence[tuple[slice, Preconditioner]]

    def _apply(self, x: torch.Tensor, name: str) -> torch.Tensor:
        out = torch.empty_like(x)
        for sl, pc in self.blocks:
            out[sl] = getattr(pc, name)(x[sl])
        return out

    def to_theta(self, phi: torch.Tensor) -> torch.Tensor:
        return self._apply(phi, "to_theta")

    def to_phi(self, theta: torch.Tensor) -> torch.Tensor:
        return self._apply(theta, "to_phi")

    def grad_to_phi(self, g: torch.Tensor) -> torch.Tensor:
        return self._apply(g, "grad_to_phi")


def sample_mask(n: int, rows: int, seed: int) -> np.ndarray:
    """A seeded Bernoulli sample of about `rows` of n rows (all rows when n <= rows)."""
    if n <= rows:
        return np.ones(n, dtype=bool)
    return np.random.default_rng(seed).random(n) < rows / n


def gram_sample(chunks: Iterator[tuple[np.ndarray, np.ndarray]], *, n_rows: int, rows: int, seed: int,
                intercept: bool) -> tuple[np.ndarray, float]:
    """Weighted Gram (1 / sum c) sum_r c_r a_r a_r^T over a seeded row sample, a = [z, 1] with `intercept`.

    chunks yields (Z float64 [c, D], weights [c]); returns (G [D (+1), D (+1)], sampled weight sum).
    """
    rng = np.random.default_rng(seed)
    rate = 1.0 if n_rows <= rows else rows / n_rows
    g: np.ndarray | None = None
    total = 0.0
    for z, c in chunks:
        keep = np.asarray(rng.random(z.shape[0]) < rate, dtype=bool) if rate < 1.0 else np.ones(z.shape[0], dtype=bool)
        if not keep.any():
            continue
        a = z[keep]
        if intercept:
            a = np.concatenate([a, np.ones((a.shape[0], 1))], axis=1)
        w = np.asarray(c, dtype=np.float64)[keep]
        part = (a * w[:, None]).T @ a
        g = part if g is None else g + part
        total += float(w.sum())
    if g is None or total <= 0:
        raise InvariantViolation("the preconditioning sample holds no weighted row")
    return g / total, total


__all__ = ["BlockPreconditioner", "CholeskyPreconditioner", "DiagonalPreconditioner", "Preconditioner",
           "SoftmaxPreconditioner", "class_rotation", "gram_sample", "sample_mask", "stable_cholesky"]
