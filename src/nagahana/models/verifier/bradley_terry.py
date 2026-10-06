"""Bradley-Terry reward model of analyst preferences, fitted by maximum a posteriori / maximum likelihood (AS-833).

Model (Bradley and Terry, Biometrika 39, 1952; Zermelo, Math. Zeitschrift 29, 1929; the reward-model
form of Christiano et al., NeurIPS 2017, arXiv:1706.03741, and Ouyang et al., NeurIPS 2022,
arXiv:2203.02155):

    P(a preferred to b) = sigma(r(a) - r(b))

with two parameterisations:

    linear   r(x) = w . phi(x), phi a fixed bounded feature map of the item (`item_features`)
    items    r_j, one reward per item j (the classical model; rewards are identified up to a constant)

Soft labels s_i in [0, 1] (feedback.PreferenceFeedback.label) and case weights omega_i give the negative
log posterior with a Gaussian prior of precision lambda (lambda = 0: maximum likelihood):

    L(w) = sum_i omega_i [ softplus(z_i) - s_i z_i ] + (lambda / 2) ||w||^2,      z_i = w . d_i
    g(w) = sum_i omega_i (sigma(z_i) - s_i) d_i + lambda w
    H(w) = sum_i omega_i sigma(z_i) (1 - sigma(z_i)) d_i d_i^T + lambda I

with d_i = phi(a_i) - phi(b_i) (linear) or d_i = e_(a_i) - e_(b_i) (items). H is positive semi-definite,
so L is convex; with lambda > 0 it is strictly convex and the minimiser is unique. Item rewards with
lambda = 0 are unique only up to a constant; the gauge sum_j r_j = 0 fixes it (with lambda > 0 the
optimum satisfies it automatically, because every d_i sums to zero). The MLE exists for the item model
when the comparison graph is strongly connected (Ford, American Mathematical Monthly 64, 1957).

Solver: damped Newton. The direction solves (H + c 1 1^T / n) p = -g for the item model with lambda = 0
(c = 1 adds the gauge direction so the system is non-singular on a connected comparison graph and p sums
to zero), else H p = -g by Cholesky; the step length halves until the Armijo condition
L(w + t p) <= L(w) + armijo t g . p holds (Boyd and Vandenberghe, Convex Optimization, 2004, sections
9.2 and 9.5). It stops when ||g||_inf <= tol; convergence is quadratic near the optimum. All arithmetic is
float64.

Uncertainty: the Laplace approximation of the posterior covariance is H^-1 at the optimum (for the gauge
fixed item model the pseudo-inverse on the sum-zero subspace), reported as standard errors of the
reward parameters.

Item features (bounded; the reward model compares items of one family)

    forecast / route set (10): curve_known, P(K), mean_k P(k), weighted spread of the routes' F_n(K),
        route concentration sum_n w_n^2, log(1 + routes)/10, stage_known, weighted furthest stage progress,
        weighted share of steps with a target, weighted technique diversity (distinct / K)
    advisory (7): Delta P_inf (expected), CVaR, worst case, log(1 + disruption cost), information
        value, feasible, log(1 + steps)

Unknown values (a route set shown without curves or stages) are 0 with their flag 0, never imputed.

Invariants (tested): the fit recovers planted rewards (both parameterisations); its gradient is zero at
the returned optimum; it agrees with a general-purpose optimiser on the same objective.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

from nagahana.core.errors import InvariantViolation
from nagahana.models.verifier.config import BradleyTerryConfig
from nagahana.models.verifier.feedback import PreferenceFeedback, PreferenceItem
from nagahana.models.vocab import STAGE_PROGRESS

FORECAST_FEATURES: tuple[str, ...] = (
    "curve_known", "p_inf_K", "p_inf_mean", "route_spread", "route_concentration", "log_routes", "stage_known",
    "stage_progress", "target_share", "technique_diversity",
)
ADVISORY_FEATURES: tuple[str, ...] = (
    "delta_p_inf", "delta_p_inf_cvar", "delta_p_inf_worst", "log_cost", "information_value", "feasible", "log_steps",
)


def feature_family(item_kind: str) -> str:
    """The reward-model family of an item kind: forecasts and route sets share one, advisories have their own."""
    if item_kind in ("forecast", "route_set"):
        return "forecast"
    if item_kind == "advisory":
        return "advisory"
    raise InvariantViolation(f"unknown item kind {item_kind!r}")


def item_features(item_kind: str, item: PreferenceItem) -> torch.Tensor:
    """phi(item) [d] float64 for the family of `item_kind` (module docstring)."""
    if feature_family(item_kind) == "advisory":
        plan = item.plan
        if plan is None:
            raise InvariantViolation("an advisory item needs its plan")
        return torch.tensor([plan.delta_p_inf, plan.delta_p_inf_cvar, plan.delta_p_inf_worst, math.log1p(plan.disruption_cost),
                             plan.information_value, 1.0 if plan.feasible else 0.0, math.log1p(len(plan.steps))], dtype=torch.float64)
    routes = item.routes
    if not routes:
        raise InvariantViolation("a forecast or route-set item needs its routes")
    counts = torch.tensor([float(r.count) for r in routes], dtype=torch.float64)
    w = counts / counts.sum()                                                    # route weights [N]
    k = routes[0].horizon
    curves_known = all(len(r.cumulative) == k for r in routes)
    if item.p_inf:
        curve = torch.tensor(item.p_inf, dtype=torch.float64)
        curve_known, p_k, p_mean = 1.0, float(curve[-1]), float(curve.mean())
    elif curves_known:
        curve = (w[:, None] * torch.tensor([list(r.cumulative) for r in routes], dtype=torch.float64)).sum(0)
        curve_known, p_k, p_mean = 1.0, float(curve[-1]), float(curve.mean())
    else:
        curve_known, p_k, p_mean = 0.0, 0.0, 0.0
    spread = 0.0
    if curves_known and len(routes) > 1:
        f_k = torch.tensor([r.cumulative[-1] for r in routes], dtype=torch.float64)
        mu = float((w * f_k).sum())
        spread = math.sqrt(max(float((w * (f_k - mu) ** 2).sum()), 0.0))
    stage_known = all(len(r.stage_codes) == k for r in routes)
    progress = 0.0
    if stage_known:
        progress = float(sum(float(wi) * max(STAGE_PROGRESS[int(s)] for s in r.stage_codes) for wi, r in zip(w, routes, strict=True)))
    target_share = float(sum(float(wi) * sum(1 for v in r.targets if v >= 0) / k for wi, r in zip(w, routes, strict=True)))
    diversity = float(sum(float(wi) * len(set(r.techniques)) / k for wi, r in zip(w, routes, strict=True)))
    return torch.tensor([curve_known, p_k, p_mean, spread, float((w ** 2).sum()), math.log1p(len(routes)) / 10.0,
                         1.0 if stage_known else 0.0, progress, target_share, diversity], dtype=torch.float64)


@dataclass(frozen=True)
class BTSolution:
    """Result of one Bradley-Terry fit: parameters, Laplace covariance and convergence facts."""

    params: torch.Tensor
    covariance: torch.Tensor
    nll: float
    grad_norm: float
    iterations: int
    converged: bool


def bt_objective(w: torch.Tensor, d: torch.Tensor, s: torch.Tensor, omega: torch.Tensor, l2: float
                 ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(L, g, H) of the module docstring at w [p] for differences d [n, p], labels s [n], weights omega [n]."""
    z = d @ w                                                                    # [n]
    sig = torch.sigmoid(z)
    value = (omega * (torch.nn.functional.softplus(z) - s * z)).sum() + 0.5 * l2 * (w @ w)
    grad = d.T @ (omega * (sig - s)) + l2 * w
    curv = omega * sig * (1.0 - sig)                                             # [n]
    hess = (d.T * curv) @ d + l2 * torch.eye(w.numel(), dtype=w.dtype)
    return value, grad, hess


def _check_inputs(d: torch.Tensor, s: torch.Tensor, omega: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dd = d.double()
    ss = s.double().reshape(-1)
    if dd.dim() != 2 or dd.shape[0] != ss.shape[0] or dd.shape[0] == 0:
        raise InvariantViolation("Bradley-Terry needs differences [n, p] and labels [n] with n >= 1")
    if not bool(torch.isfinite(dd).all()) or bool(((ss < 0) | (ss > 1)).any()):
        raise InvariantViolation("differences must be finite and labels must lie in [0, 1]")
    om = torch.ones_like(ss) if omega is None else omega.double().reshape(-1)
    if om.shape != ss.shape or bool((om < 0).any()) or not bool(torch.isfinite(om).all()):
        raise InvariantViolation("case weights must be finite, non-negative, one per comparison")
    return dd, ss, om


def _newton(d: torch.Tensor, s: torch.Tensor, omega: torch.Tensor, cfg: BradleyTerryConfig, *, gauge: bool) -> BTSolution:
    # Damped Newton with Armijo backtracking (module docstring); `gauge` adds the sum-zero direction.
    p = d.shape[1]
    w = torch.zeros(p, dtype=torch.float64)
    ones = torch.ones(p, dtype=torch.float64)
    value, grad, hess = bt_objective(w, d, s, omega, cfg.l2)
    it, converged = 0, False
    for it in range(1, cfg.max_iter + 1):
        if float(grad.abs().max()) <= cfg.tol:
            converged = True
            it -= 1
            break
        system = hess + torch.outer(ones, ones) / p if gauge else hess
        try:
            step = -torch.linalg.solve(system, grad)
        except RuntimeError as exc:
            raise InvariantViolation("the Bradley-Terry Hessian is singular: the comparisons do not identify the rewards "
                                     "(a disconnected comparison graph or collinear features); set l2 > 0") from exc
        if gauge:
            step = step - step.mean()
        slope = float(grad @ step)
        if slope >= 0:                                                           # not a descent direction: use the gradient
            step, slope = -grad, -float(grad @ grad)
        t = 1.0
        while True:
            cand = w + t * step
            v_new, g_new, h_new = bt_objective(cand, d, s, omega, cfg.l2)
            if float(v_new) <= float(value) + cfg.armijo * t * slope or t < 1e-12:
                break
            t *= cfg.backtrack
        w, value, grad, hess = cand, v_new, g_new, h_new
    else:
        converged = float(grad.abs().max()) <= cfg.tol
    if gauge:
        # Laplace covariance on the sum-zero subspace: pseudo-inverse of H restricted to it.
        proj = torch.eye(p, dtype=torch.float64) - torch.outer(ones, ones) / p
        cov = proj @ torch.linalg.pinv(proj @ hess @ proj, hermitian=True) @ proj
    else:
        cov = torch.linalg.inv(hess)
    n_eff = float(omega.sum()) if float(omega.sum()) > 0 else 1.0
    return BTSolution(params=w, covariance=cov, nll=float(value) / n_eff, grad_norm=float(grad.abs().max()),
                      iterations=it, converged=converged)


def fit_linear(differences: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor | None, cfg: BradleyTerryConfig) -> BTSolution:
    """MAP / ML fit of r(x) = w . phi(x) from feature differences [n, p] (module docstring)."""
    d, s, om = _check_inputs(differences, labels, weights)
    if cfg.l2 == 0.0 and int(torch.linalg.matrix_rank(d)) < d.shape[1]:
        raise InvariantViolation("with l2 = 0 the feature differences must have full column rank")
    return _newton(d, s, om, cfg, gauge=False)


def fit_items(first: torch.Tensor, second: torch.Tensor, labels: torch.Tensor, n_items: int,
              weights: torch.Tensor | None, cfg: BradleyTerryConfig) -> BTSolution:
    """MAP / ML fit of one reward per item from comparisons (first[i], second[i]) (module docstring)."""
    a, b = first.long().reshape(-1), second.long().reshape(-1)
    if a.shape != b.shape or bool(((a < 0) | (a >= n_items) | (b < 0) | (b >= n_items) | (a == b)).any()):
        raise InvariantViolation("item indices must lie in 0 ... n_items - 1 and differ within a comparison")
    d = torch.zeros(a.numel(), n_items, dtype=torch.float64)
    rows = torch.arange(a.numel())
    d[rows, a] = 1.0
    d[rows, b] = -1.0
    dd, s, om = _check_inputs(d, labels, weights)
    return _newton(dd, s, om, cfg, gauge=cfg.l2 == 0.0)


@dataclass(frozen=True)
class BradleyTerryModel:
    """A fitted linear reward model of one item family (module docstring), as stored with a candidate."""

    family: str
    feature_names: tuple[str, ...]
    weights: tuple[float, ...]
    std_errors: tuple[float, ...]
    l2: float
    n_pairs: int
    nll: float
    converged: bool

    def __post_init__(self) -> None:
        if self.family not in ("forecast", "advisory"):
            raise InvariantViolation(f"unknown reward-model family {self.family!r}")
        if not (len(self.feature_names) == len(self.weights) == len(self.std_errors)):
            raise InvariantViolation("feature names, weights and standard errors must align")

    def reward(self, item_kind: str, item: PreferenceItem) -> float:
        """r(item) = w . phi(item)."""
        if feature_family(item_kind) != self.family:
            raise InvariantViolation(f"a {self.family} reward model cannot score a {item_kind} item")
        return float(item_features(item_kind, item) @ torch.tensor(self.weights, dtype=torch.float64))

    def prob_first(self, event: PreferenceFeedback) -> float:
        """P(first preferred) = sigma(r(first) - r(second))."""
        z = self.reward(event.item_kind, event.first) - self.reward(event.item_kind, event.second)
        return 1.0 / (1.0 + math.exp(-z)) if z >= 0 else math.exp(z) / (1.0 + math.exp(z))

    def to_record(self) -> dict[str, Any]:
        return {"family": self.family, "feature_names": list(self.feature_names), "weights": list(self.weights),
                "std_errors": list(self.std_errors), "l2": self.l2, "n_pairs": self.n_pairs, "nll": self.nll,
                "converged": self.converged}

    @classmethod
    def from_record(cls, r: dict[str, Any]) -> BradleyTerryModel:
        return cls(family=str(r["family"]), feature_names=tuple(str(x) for x in r["feature_names"]),
                   weights=tuple(float(x) for x in r["weights"]), std_errors=tuple(float(x) for x in r["std_errors"]),
                   l2=float(r["l2"]), n_pairs=int(r["n_pairs"]), nll=float(r["nll"]), converged=bool(r["converged"]))


def fit_reward_model(preferences: Sequence[PreferenceFeedback], cfg: BradleyTerryConfig,
                     weights: Sequence[float] | None = None) -> BradleyTerryModel:
    """Fit the linear reward model of one family on preferences of that family."""
    if not preferences:
        raise InvariantViolation("a reward model needs at least one preference")
    families = {feature_family(p.item_kind) for p in preferences}
    if len(families) != 1:
        raise InvariantViolation(f"preferences of one family only, got {sorted(families)}")
    family = families.pop()
    names = ADVISORY_FEATURES if family == "advisory" else FORECAST_FEATURES
    d = torch.stack([item_features(p.item_kind, p.first) - item_features(p.item_kind, p.second) for p in preferences])
    s = torch.tensor([p.label for p in preferences], dtype=torch.float64)
    om = None if weights is None else torch.tensor(list(weights), dtype=torch.float64)
    sol = fit_linear(d, s, om, cfg)
    se = torch.sqrt(torch.diagonal(sol.covariance).clamp_min(0.0))
    return BradleyTerryModel(family=family, feature_names=names, weights=tuple(float(x) for x in sol.params),
                             std_errors=tuple(float(x) for x in se), l2=cfg.l2, n_pairs=len(preferences), nll=sol.nll,
                             converged=sol.converged)


__all__ = ["ADVISORY_FEATURES", "FORECAST_FEATURES", "BTSolution", "BradleyTerryModel", "bt_objective", "feature_family",
           "fit_items", "fit_linear", "fit_reward_model", "item_features"]
