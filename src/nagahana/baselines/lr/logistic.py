"""Binary logistic regression: objective, class weights, solvers, convergence diagnostics, cross-check.

Model and objective

    p(y = 1 | x) = sigmoid(w.x + b)

    F(w, b) = (1 / S) * sum_i c_i * l(y_i, w.x_i + b) + (lambda / 2) * ||w||^2,    S = sum_i c_i
    l(y, z) = log(1 + exp(z)) - y * z                                              (binary cross-entropy)

with the intercept b unpenalised and class weights c_i = (n / (2 n_{y_i})) ** p: p = 1 is the balanced
weighting n / (2 n_c) (King and Zeng, "Logistic regression in rare events data", Political Analysis
9(2), 2001, discuss weighting for rare events), p = 0 is no weighting, and the strength p is chosen
with lambda (selection in temporal_cv.py). Normalising by S makes F independent of the scale of the
weights, so lambda means the same thing for every p. F is strictly convex in w for lambda > 0, so the
minimiser is unique and every solver below converges to the same point.

Gradient (used directly; tests/test_lr_logistic.py checks it against autograd):

    dF/dw = (1 / S) * X^T (c * (sigmoid(z) - y)) + lambda * w,      dF/db = (1 / S) * sum_i c_i (sigmoid(z_i) - y_i)

log(1 + exp(z)) is evaluated as max(z, 0) + log1p(exp(-|z|)), exact in float64 for every z.

Solvers

    lbfgs           full-batch L-BFGS with the strong-Wolfe line search of torch.optim.LBFGS, float64
                    (Liu and Nocedal, "On the limited memory BFGS method for large scale optimization",
                    Mathematical Programming 45, 1989; Nocedal and Wright, Numerical Optimization, 2nd ed.,
                    Springer 2006, Algorithms 7.4 and 3.5-3.6). The data are materialised once.
    streamed_lbfgs  the same iterations, with every objective and gradient accumulated over chunks of a
                    re-iterable row source: exact full-batch L-BFGS for data that do not fit in memory (AS-528).
    minibatch       averaged stochastic gradient descent over a shuffled stream of mini-batches with the
                    gain eta_t = eta_0 / (1 + eta_0 * lambda * t) ** 0.75 and Polyak-Ruppert averaging of
                    the iterates (Polyak and Juditsky, SIAM J. Control Optim. 30(4), 1992; Bottou,
                    "Stochastic gradient descent tricks", LNCS 7700, 2012; Xu, arXiv:1107.2490), followed
                    by `polish_iter` streamed L-BFGS iterations from the averaged point, which bring it to
                    the exact optimum.

Every exact solve is preconditioned by Boehning's fixed bound of the Hessian (precondition.py): the
iterations run in rotated and rescaled variables, the minimiser is unchanged and the iteration count no
longer grows with the collinearity of the design.

Convergence diagnostics (SolveInfo): iterations, objective evaluations, final objective, max-abs and
Euclidean norms of the gradient, the relative objective change between the last two checks, the
objective at every check, the reason for stopping and whether it is a convergence.

The scikit-learn cross-check (lazy import of the [baselines] extra, D-55) fits
sklearn.linear_model.LogisticRegression on the same standardised rows with C = 1 / (lambda * S) and the
same per-row weights, which is the same objective multiplied by 1 / (C * S), and checks that the
coefficients, the intercept and the probabilities agree to tolerance.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field

import numpy as np
import torch

from nagahana.core.errors import InvariantViolation

from .config import SolverConfig
from .precondition import CholeskyPreconditioner, Preconditioner, gram_sample

#: (X float64 [c, D], y float64 [c], c float64 [c]) chunks of a row source.
Chunk = tuple[np.ndarray, np.ndarray, np.ndarray]


@dataclass
class RowSource:
    """A re-iterable source of standardised training rows, read in chunks.

    factory(chunk_rows) returns a fresh iterator of (X float64 [c, D], y float64 [c], weight float64 [c]).
    An in-memory source (`from_arrays`) and an out-of-core one (features.TriggerDesign / UpdateDesign
    views) expose the same interface, so every solver accepts both.
    """

    n_rows: int
    n_cols: int
    factory: Callable[[int], Iterator[Chunk]]

    def chunks(self, chunk_rows: int) -> Iterator[Chunk]:
        return self.factory(chunk_rows)

    @classmethod
    def from_arrays(cls, x: np.ndarray, y: np.ndarray, weight: np.ndarray | None = None) -> RowSource:
        x = np.ascontiguousarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        w = np.ones_like(y) if weight is None else np.asarray(weight, dtype=np.float64)
        if x.ndim != 2 or y.shape != (x.shape[0],) or w.shape != y.shape:
            raise InvariantViolation("rows must be X [n, D], y [n], weight [n]")

        def factory(chunk_rows: int) -> Iterator[Chunk]:
            for a in range(0, x.shape[0], chunk_rows):
                yield x[a:a + chunk_rows], y[a:a + chunk_rows], w[a:a + chunk_rows]

        return cls(n_rows=int(x.shape[0]), n_cols=int(x.shape[1]), factory=factory)

    def materialise(self, chunk_rows: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """All rows as arrays (in-memory solvers)."""
        parts = list(self.chunks(chunk_rows))
        if not parts:
            return np.zeros((0, self.n_cols)), np.zeros(0), np.zeros(0)
        return (np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts]),
                np.concatenate([p[2] for p in parts]))


@dataclass
class SolveInfo:
    """Convergence diagnostics of one solve (module docstring)."""

    method: str
    iterations: int
    evaluations: int
    objective: float
    grad_inf: float
    grad_norm: float
    rel_change: float
    converged: bool
    reason: str
    seconds: float
    history: list[float] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {"method": self.method, "iterations": self.iterations, "evaluations": self.evaluations,
                "objective": self.objective, "grad_inf": self.grad_inf, "grad_norm": self.grad_norm,
                "rel_change": self.rel_change, "converged": self.converged, "reason": self.reason,
                "seconds": self.seconds, "history": list(self.history)}


def log1pexp(z: torch.Tensor) -> torch.Tensor:
    """log(1 + exp(z)), exact for every z: max(z, 0) + log1p(exp(-|z|))."""
    return torch.clamp(z, min=0.0) + torch.log1p(torch.exp(-z.abs()))


def class_weights(y: np.ndarray, power: float) -> tuple[float, float]:
    """(c_0, c_1) with c_y = (n / (2 n_y)) ** power over the given labels (module docstring)."""
    y = np.asarray(y, dtype=np.float64)
    n = float(y.size)
    n1 = float(y.sum())
    n0 = n - n1
    if n1 <= 0 or n0 <= 0:
        raise InvariantViolation(f"class weights need both classes in the training rows (n0={n0:.0f}, n1={n1:.0f})")
    if power < 0:
        raise InvariantViolation("the class-weight power must be >= 0")
    return (n / (2.0 * n0)) ** power, (n / (2.0 * n1)) ** power


def sample_weights(y: np.ndarray, power: float) -> np.ndarray:
    """c_i for every row (class_weights applied to its label)."""
    c0, c1 = class_weights(y, power)
    return np.where(np.asarray(y) > 0.5, c1, c0).astype(np.float64)


def _to_t(a: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(a, dtype=np.float64))


def logistic_value_grad(theta: torch.Tensor, source: RowSource, lam: float, chunk_rows: int,
                        total_weight: float) -> tuple[float, torch.Tensor]:
    """F and dF/dtheta at theta = [w (D), b] over every chunk of `source` (module docstring)."""
    d = source.n_cols
    w, b = theta[:d], theta[d]
    f = 0.0
    g = torch.zeros_like(theta)
    for x, y, c in source.chunks(chunk_rows):
        xt, yt, ct = _to_t(x), _to_t(y), _to_t(c)
        z = xt @ w + b                                                         # [c]
        f += float((ct * (log1pexp(z) - yt * z)).sum()) / total_weight
        r = ct * (torch.sigmoid(z) - yt) / total_weight                        # [c]
        g[:d] += xt.T @ r
        g[d] += r.sum()
    f += 0.5 * lam * float(w @ w)
    g[:d] += lam * w
    return f, g


ValueGrad = Callable[[torch.Tensor], tuple[float, torch.Tensor]]


def lbfgs_minimise(value_grad: ValueGrad, theta0: torch.Tensor, cfg: SolverConfig, *, method: str = "lbfgs",
                   precond: Preconditioner | None = None) -> tuple[torch.Tensor, SolveInfo]:
    """Minimise a smooth convex function with torch's L-BFGS (strong Wolfe) and our stopping rules.

    value_grad(theta) returns (value, gradient) at a detached float64 theta. With `precond`, the iterations
    run in the variables phi of the change of variables theta = M phi (precondition.py), which leaves the
    minimiser unchanged. The optimiser runs in blocks of `check_every` iterations; after each block the
    gradient in theta (the true gradient, whatever the preconditioning) and the relative objective change
    are checked (module docstring). Evaluations are cached, so a check costs no extra evaluation.
    """
    t0 = time.perf_counter()
    theta_init = theta0.detach().clone().to(torch.float64)
    phi = (precond.to_phi(theta_init) if precond is not None else theta_init).clone().requires_grad_(True)
    opt = torch.optim.LBFGS([phi], lr=1.0, max_iter=cfg.check_every, max_eval=None, tolerance_grad=0.0,
                            tolerance_change=0.0, history_size=cfg.history_size, line_search_fn="strong_wolfe")
    cache: dict[str, object] = {}
    evals = 0

    def evaluate(th: torch.Tensor) -> tuple[float, torch.Tensor]:
        nonlocal evals
        key = cache.get("theta")
        if isinstance(key, torch.Tensor) and torch.equal(key, th):
            f_c, g_c = cache["f"], cache["g"]
            assert isinstance(f_c, float) and isinstance(g_c, torch.Tensor)
            return f_c, g_c
        f, g = value_grad(th.detach())
        if not math.isfinite(f) or not bool(torch.isfinite(g).all()):
            raise InvariantViolation("the objective or its gradient is not finite (check the design for NaN or overflow)")
        evals += 1
        cache.update(theta=th.detach().clone(), f=f, g=g.detach().clone())
        return f, g

    def theta_of(ph: torch.Tensor) -> torch.Tensor:
        return precond.to_theta(ph) if precond is not None else ph

    def closure() -> torch.Tensor:
        opt.zero_grad()
        f, g = evaluate(theta_of(phi.detach()))
        phi.grad = (precond.grad_to_phi(g) if precond is not None else g).clone()
        return torch.tensor(f, dtype=torch.float64)

    f_prev, g_now = evaluate(theta_of(phi.detach()))
    history = [f_prev]
    reason, converged, rel = "max_iter", False, math.inf
    blocks = max(1, math.ceil(cfg.max_iter / cfg.check_every))
    iters_before = 0
    for _ in range(blocks):
        if float(g_now.abs().max()) <= cfg.tol_grad:
            reason, converged = "gradient", True
            break
        opt.step(closure)
        iters = int(opt.state[phi].get("n_iter", 0))
        f_now, g_now = evaluate(theta_of(phi.detach()))
        history.append(f_now)
        rel = abs(f_prev - f_now) / max(1.0, abs(f_now))
        if float(g_now.abs().max()) <= cfg.tol_grad:
            reason, converged = "gradient", True
            break
        if rel <= cfg.tol_rel_obj:
            reason, converged = "objective", True
            break
        if iters == iters_before:
            reason = "stalled"
            break
        iters_before = iters
        f_prev = f_now
    theta = theta_of(phi.detach()).clone()
    f_fin, g_fin = evaluate(theta)
    info = SolveInfo(method=method, iterations=int(opt.state[phi].get("n_iter", 0)), evaluations=evals, objective=f_fin,
                     grad_inf=float(g_fin.abs().max()), grad_norm=float(torch.linalg.vector_norm(g_fin)),
                     rel_change=float(rel), converged=converged, reason=reason, seconds=time.perf_counter() - t0,
                     history=history)
    return theta, info


def binary_preconditioner(source: RowSource, lam: float, cfg: SolverConfig, seed: int) -> CholeskyPreconditioner:
    """B = (1/4) G + lambda diag(1, ..., 1, 0) with G the weighted Gram of [z, 1] on a row sample (precondition.py)."""
    g, _total = gram_sample(((x, c) for x, _y, c in source.chunks(cfg.chunk_rows)), n_rows=source.n_rows,
                            rows=cfg.precond_rows, seed=seed, intercept=True)
    pen = np.r_[np.ones(source.n_cols), 0.0]
    return CholeskyPreconditioner.from_matrix(0.25 * g + lam * np.diag(pen))


def _shuffled_batches(source: RowSource, cfg: SolverConfig, rng: np.random.Generator) -> Iterator[Chunk]:
    """Mini-batches of a sequential stream, shuffled inside buffers of `shuffle_buffer` rows."""
    buf: list[Chunk] = []
    held = 0

    def drain() -> Iterator[Chunk]:
        x = np.concatenate([b[0] for b in buf])
        y = np.concatenate([b[1] for b in buf])
        c = np.concatenate([b[2] for b in buf])
        perm = rng.permutation(x.shape[0])
        for a in range(0, perm.size, cfg.batch_size):
            idx = perm[a:a + cfg.batch_size]
            yield x[idx], y[idx], c[idx]

    for chunk in source.chunks(cfg.chunk_rows):
        buf.append(chunk)
        held += chunk[0].shape[0]
        if held >= cfg.shuffle_buffer:
            yield from drain()
            buf, held = [], 0
    if buf:
        yield from drain()


def minibatch_minimise(source: RowSource, lam: float, total_weight: float, cfg: SolverConfig, *, seed: int,
                       theta0: torch.Tensor | None = None,
                       precond: Preconditioner | None = None) -> tuple[torch.Tensor, SolveInfo]:
    """Averaged mini-batch SGD on F, then streamed L-BFGS polishing (module docstring)."""
    t0 = time.perf_counter()
    d = source.n_cols
    rng = np.random.default_rng(seed)
    theta = torch.zeros(d + 1, dtype=torch.float64) if theta0 is None else theta0.detach().clone().to(torch.float64)
    n = float(source.n_rows)
    # Curvature bound of the data term on the first chunk: L <= (n / S) * mean_i c_i ||x_i||^2 / 4 (+ lambda).
    first = next(iter(source.chunks(cfg.chunk_rows)), None)
    if first is None:
        raise InvariantViolation("minibatch solver: the row source is empty")
    xf, _, cf = first
    l_hat = (n / total_weight) * float(np.mean(cf * (np.einsum("ij,ij->i", xf, xf) + 1.0))) / 4.0 + lam
    eta0 = cfg.step0 / max(l_hat, 1e-12)
    avg = theta.clone()
    t = 0
    t_avg0 = 0 if cfg.epochs == 1 else None
    history: list[float] = []
    for epoch in range(cfg.epochs):
        if epoch == 1 and t_avg0 is None:
            t_avg0 = t
            avg = theta.clone()
        for x, y, c in _shuffled_batches(source, cfg, rng):
            xt, yt, ct = _to_t(x), _to_t(y), _to_t(c)
            z = xt @ theta[:d] + theta[d]
            # unbiased estimate of grad F: (n / S) * mean_{i in B} c_i grad l_i + lambda * w
            r = (n / total_weight) * ct * (torch.sigmoid(z) - yt) / float(x.shape[0])
            g = torch.empty_like(theta)
            g[:d] = xt.T @ r + lam * theta[:d]
            g[d] = r.sum()
            eta = eta0 / (1.0 + eta0 * lam * t) ** cfg.decay if lam > 0 else eta0 / (1.0 + t) ** cfg.decay
            theta = theta - eta * g
            t += 1
            if t_avg0 is not None:
                k = t - t_avg0
                avg = avg + (theta - avg) / float(k)
        f_ep, _ = logistic_value_grad(avg if t_avg0 is not None else theta, source, lam, cfg.chunk_rows, total_weight)
        history.append(f_ep)
    start = avg if t_avg0 is not None else theta
    if cfg.polish_iter > 0:
        polish_cfg = SolverConfig(method="streamed_lbfgs", max_iter=cfg.polish_iter, tol_grad=cfg.tol_grad,
                                  tol_rel_obj=cfg.tol_rel_obj, history_size=cfg.history_size,
                                  check_every=min(cfg.check_every, cfg.polish_iter), chunk_rows=cfg.chunk_rows,
                                  precondition=cfg.precondition, precond_rows=cfg.precond_rows)
        theta_p, info = lbfgs_minimise(lambda th: logistic_value_grad(th, source, lam, cfg.chunk_rows, total_weight),
                                       start, polish_cfg, method="minibatch", precond=precond)
        info.history = history + info.history
        info.seconds = time.perf_counter() - t0
        info.iterations += t
        return theta_p, info
    f_fin, g_fin = logistic_value_grad(start, source, lam, cfg.chunk_rows, total_weight)
    rel = abs(history[-2] - history[-1]) / max(1.0, abs(history[-1])) if len(history) > 1 else math.inf
    info = SolveInfo(method="minibatch", iterations=t, evaluations=len(history) + 1, objective=f_fin,
                     grad_inf=float(g_fin.abs().max()), grad_norm=float(torch.linalg.vector_norm(g_fin)), rel_change=rel,
                     converged=float(g_fin.abs().max()) <= cfg.tol_grad, reason="epochs", seconds=time.perf_counter() - t0,
                     history=history)
    return start, info


@dataclass
class BinaryFit:
    """A fitted binary logistic regression on standardised features."""

    coef: np.ndarray            # float64 [D]
    intercept: float
    lam: float
    power: float
    weights: tuple[float, float]  # (c_0, c_1) used in training
    info: SolveInfo

    def decision(self, x: np.ndarray) -> np.ndarray:
        """Logits w.x + b for standardised rows x [n, D]."""
        return np.asarray(x, dtype=np.float64) @ self.coef + self.intercept

    def proba(self, x: np.ndarray) -> np.ndarray:
        """p(y = 1 | x)."""
        return expit(self.decision(x))

    def theta(self) -> torch.Tensor:
        return torch.from_numpy(np.concatenate([self.coef, [self.intercept]]))


def expit(z: np.ndarray) -> np.ndarray:
    """Numerically stable logistic sigmoid in float64."""
    z = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def fit_binary(source: RowSource, *, lam: float, power: float, weights: tuple[float, float], cfg: SolverConfig,
               seed: int = 0, theta0: torch.Tensor | None = None) -> BinaryFit:
    """Fit F on the rows of `source` (labels in {0, 1}; per-row weights already in the source).

    `weights` records the (c_0, c_1) the caller used to build the source's weight column.
    """
    if lam < 0:
        raise InvariantViolation("lambda must be >= 0")
    total = 0.0
    pos = 0.0
    for _x, y, c in source.chunks(cfg.chunk_rows):
        if np.any((y != 0.0) & (y != 1.0)):
            raise InvariantViolation("binary labels must be 0 or 1")
        total += float(c.sum())
        pos += float(y.sum())
    if total <= 0:
        raise InvariantViolation("the training rows carry no weight")
    if pos <= 0 or pos >= source.n_rows:
        raise InvariantViolation("a logistic regression needs both classes in its training rows")
    d = source.n_cols
    if theta0 is None:
        theta0 = torch.zeros(d + 1, dtype=torch.float64)
    src = source
    rows = cfg.chunk_rows
    if cfg.method == "lbfgs":
        x, y, c = source.materialise(cfg.chunk_rows)
        src = RowSource.from_arrays(x, y, c)
        rows = max(1, x.shape[0])
    precond = binary_preconditioner(src, lam, cfg, seed) if cfg.precondition and lam > 0 else None
    if cfg.method == "minibatch":
        theta, info = minibatch_minimise(source, lam, total, cfg, seed=seed, theta0=theta0, precond=precond)
    else:
        theta, info = lbfgs_minimise(lambda th: logistic_value_grad(th, src, lam, rows, total), theta0, cfg,
                                     method=cfg.method, precond=precond)
    th = theta.numpy()
    return BinaryFit(coef=th[:d].copy(), intercept=float(th[d]), lam=float(lam), power=float(power), weights=weights,
                     info=info)


def sklearn_cross_check(x: np.ndarray, y: np.ndarray, weight: np.ndarray, fit: BinaryFit, *, coef_rtol: float = 1e-4,
                        coef_atol: float = 1e-6, prob_atol: float = 1e-6, max_iter: int = 100_000) -> dict[str, float]:
    """Check the fit against scikit-learn on the same rows and objective (module docstring).

    The fitted solution is first polished on our objective to machine precision (L-BFGS from the fitted point,
    stopped only when the objective no longer changes), so the comparison tests the objective and the solver,
    not the stopping tolerance of the deployed fit. The polished solution must agree with scikit-learn's
    (tol 1e-12) to the given tolerances, or InvariantViolation is raised. Returned: the largest differences
    between scikit-learn and the polished solution ("coef", "intercept", "proba"), and between the deployed fit
    and the polished one ("deployed_coef", "deployed_proba"), its optimisation error. Needs the [baselines] extra.
    """
    try:
        import sklearn
        from sklearn.linear_model import LogisticRegression
    except ImportError as exc:                                   # pragma: no cover - depends on the extra
        raise ImportError("the scikit-learn cross-check needs the optional [baselines] extra: pip install 'nagahana[baselines]'") from exc
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    weight = np.asarray(weight, dtype=np.float64)
    total = float(weight.sum())
    polish = SolverConfig(method="lbfgs", max_iter=20_000, tol_grad=1e-12, tol_rel_obj=0.0)
    ref = fit_binary(RowSource.from_arrays(x, y, weight), lam=fit.lam, power=fit.power, weights=fit.weights, cfg=polish,
                     theta0=fit.theta())
    c = math.inf if fit.lam == 0 else 1.0 / (fit.lam * total)
    major, minor = (int(v) for v in sklearn.__version__.split(".")[:2])
    kwargs: dict[str, object] = {"l1_ratio": 0.0} if (major, minor) >= (1, 8) else {"penalty": "l2"}
    model = LogisticRegression(C=c, fit_intercept=True, solver="lbfgs", tol=1e-12, max_iter=max_iter, **kwargs)
    model.fit(x, y.astype(np.int64), sample_weight=weight)
    sk_coef = model.coef_.ravel().astype(np.float64)
    sk_b = float(model.intercept_[0])
    d_coef = float(np.max(np.abs(sk_coef - ref.coef))) if sk_coef.size else 0.0
    d_b = abs(sk_b - ref.intercept)
    d_prob = float(np.max(np.abs(model.predict_proba(x)[:, 1] - ref.proba(x)))) if x.shape[0] else 0.0
    scale = float(np.max(np.abs(ref.coef))) if ref.coef.size else 0.0
    if d_coef > coef_atol + coef_rtol * scale or d_b > coef_atol + coef_rtol * max(1.0, abs(ref.intercept)) or d_prob > prob_atol:
        raise InvariantViolation(f"scikit-learn disagrees: max |d coef| {d_coef:.3e}, |d intercept| {d_b:.3e}, "
                                 f"max |d p| {d_prob:.3e}")
    dep_coef = float(np.max(np.abs(fit.coef - ref.coef))) if ref.coef.size else 0.0
    dep_prob = float(np.max(np.abs(fit.proba(x) - ref.proba(x)))) if x.shape[0] else 0.0
    return {"coef": d_coef, "intercept": d_b, "proba": d_prob, "deployed_coef": dep_coef, "deployed_proba": dep_prob}


__all__ = [
    "BinaryFit", "RowSource", "SolveInfo", "binary_preconditioner", "class_weights", "expit", "fit_binary",
    "lbfgs_minimise", "log1pexp",
    "logistic_value_grad", "minibatch_minimise", "sample_weights", "sklearn_cross_check",
]
