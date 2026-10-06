"""Calibration maths of the Verifier: temperature, maximum-likelihood temperature, split-conformal thresholds.

Sources: [Q-34] (calibrated decisions), [A-16], [A-21]. Decisions: D-21 (temperatures change only through
`gate.apply_calibration` with a HumanCommand), D-45. Assumptions: AS-25 (temperature policy, conformal
thresholds), AS-26 (the Verifier temperature is part of the site adapter).

1. Temperature scaling of a probability (Guo et al., "On Calibration of Modern Neural Networks", ICML 2017,
   arXiv:1706.04599, for the softmax form; the binary form here):

       p_T = sigma(logit(p) / T),   logit(p) = log p - log(1 - p)

   T > 1 softens, T < 1 sharpens; T = 1 is the identity; the order (ranking, AUROC) is unchanged.

2. Maximum-likelihood temperature on resolved pairs (z_i = logit(p_i), y_i in {0, 1}). With beta = 1 / T the
   negative log-likelihood is

       NLL(beta)   = sum_i [ log(1 + exp(beta z_i)) - y_i beta z_i ]
       NLL'(beta)  = sum_i z_i (sigma(beta z_i) - y_i),     NLL''(beta) = sum_i z_i^2 sigma(beta z_i)(1 - sigma(beta z_i)) >= 0

   so NLL is convex in beta and NLL' is non-decreasing. The minimiser over beta in [1/T_max, 1/T_min] is the
   root of NLL' found by bisection (clipped to an end when NLL' has one sign there); T = 1 / beta.
   (Convexity holds in beta, not T; the arg-min is the same point.)

3. Split-conformal alert threshold at target false-positive rate alpha (Angelopoulos and Bates, "A Gentle
   Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification",
   arXiv:2107.07511, sections 1 and 3; Vovk, Gammerman and Shafer 2005). With n calibration scores
   s_1 ... s_n of benign items, let k = ceil((n + 1)(1 - alpha)) and q = s_(k) (the k-th smallest). Alerting
   when s > q gives, for a new exchangeable benign item,

       P(s_new > q) <= alpha     (and >= alpha - 1/(n + 1) when scores have no ties)

   If k > n the guarantee needs more data and the threshold is +inf (never alert by this rule; reported).

Precision (D-54): every quantity here is computed in float64: logits, tempered probabilities
(`apply_temperature` returns float64 whatever its input dtype), the bisection on NLL'(beta), the sorted
conformal scores, the reliability bins, ECE and Brier (`evaluation.calibration` works in float64). Python
floats returned (temperature, threshold, ECE, Brier) are IEEE double. Callers that hold Python floats must
build tensors with dtype=torch.float64 (torch.tensor(list) defaults to float32).

Invariants (tested): `ml_temperature` recovers a planted temperature; the conformal threshold achieves its
target FPR on fresh synthetic scores within sampling tolerance; outputs are float64.
"""

from __future__ import annotations

import math

import torch

from nagahana.evaluation.calibration import ece, reliability
from nagahana.governance.assumptions import assume
from nagahana.models.verifier.reports import CalibrationReport


def logit(p: torch.Tensor, *, eps: float = 1e-7) -> torch.Tensor:
    """log p - log(1 - p), with p clipped to [eps, 1 - eps] (eps only guards saturated inputs)."""
    q = p.double().clamp(eps, 1.0 - eps)
    return torch.log(q) - torch.log1p(-q)


def apply_temperature(p: torch.Tensor, temperature: float) -> torch.Tensor:
    """p_T = sigma(logit(p) / T) (module docstring, part 1). Returns float64 (D-54), whatever p's dtype.

    The result is a calibrated output; casting it back to a float32 input's dtype would discard the
    precision D-54 asks for, so it is never narrowed here.
    """
    if not temperature > 0:
        raise ValueError("temperature must be > 0")
    assume("AS-450", by=__name__)                                   # float64 output whatever p's dtype (D-54)
    return torch.sigmoid(logit(p) / temperature)                    # logit(p) is float64


def ml_temperature(z: torch.Tensor, y: torch.Tensor, *, t_min: float = 0.25, t_max: float = 4.0,
                   tol: float = 1e-10, max_iter: int = 200) -> float:
    """Exact ML temperature by bisection on NLL'(beta) (module docstring, part 2).

    z: logits [n] (use `logit(p)` for probabilities); y: {0, 1} [n]. Returns T in [t_min, t_max].
    """
    if not 0 < t_min < t_max:
        raise ValueError("need 0 < t_min < t_max")
    # float64 throughout (D-54): the bisection tolerance 1e-10 is below float32 resolution.
    zz, yy = z.double().flatten(), y.double().flatten()
    if zz.numel() == 0:
        raise ValueError("no resolved pairs")

    def grad(beta: float) -> float:
        return float((zz * (torch.sigmoid(beta * zz) - yy)).sum())

    lo, hi = 1.0 / t_max, 1.0 / t_min
    if grad(lo) >= 0.0:            # NLL increasing over the whole interval: smallest beta (largest T)
        return t_max
    if grad(hi) <= 0.0:            # NLL decreasing over the whole interval: largest beta (smallest T)
        return t_min
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if grad(mid) > 0.0:
            hi = mid
        else:
            lo = mid
        if hi - lo < tol:
            break
    return 1.0 / (0.5 * (lo + hi))


def conformal_threshold(benign_scores: torch.Tensor, alpha: float) -> float:
    """Split-conformal threshold q = s_(ceil((n + 1)(1 - alpha))) over benign calibration scores (module docstring, part 3).

    Alert when a score is strictly greater than q. Returns +inf when n is too small for alpha.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    s = torch.sort(benign_scores.double().flatten()).values
    n = s.numel()
    k = math.ceil((n + 1) * (1.0 - alpha))
    if k > n:
        return math.inf
    return float(s[k - 1])


def calibration_report(family: str, p: torch.Tensor, y: torch.Tensor, *, bins: int,
                       responded_to: torch.Tensor | None = None, t_min: float = 0.25, t_max: float = 4.0) -> CalibrationReport:
    """Reliability bins, ECE (`evaluation.calibration`), Brier and the ML temperature of one family.

    Responded-to pairs are excluded, not penalised [Q-38]. The ML temperature is computed only when
    both outcomes occur (otherwise it runs to an end of the interval and is not informative: None).
    """
    keep = torch.ones_like(p, dtype=torch.bool) if responded_to is None else ~responded_to.bool()
    pk, yk = p[keep].double(), y[keep].double()
    n = int(pk.numel())
    if n == 0:
        return CalibrationReport(family, 0, float("nan"), float("nan"), (), None)
    both = bool((yk > 0.5).any()) and bool((yk < 0.5).any())
    t = ml_temperature(logit(pk), yk, t_min=t_min, t_max=t_max) if both else None
    return CalibrationReport(family=family, n=n, ece=ece(pk, yk, bins=bins), brier=float(((pk - yk) ** 2).mean()),
                             bins=tuple(reliability(pk, yk, bins=bins)), ml_temperature=t)
