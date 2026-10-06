"""Route mathematics of the Forecaster: infiltration curve, merging, weights, MPPI, quantiles (build-spec §2.8).

Purpose
-------
Pure tensor functions (no parameters) used by `Forecaster.imagine`, the Advisor and the tests. Keeping
them separate makes every equation testable against its closed form.

Owner sources: [A-06] ("average, median & mode of these forecast pathways"), [Q-28] (MPC-guided
model-based RL), [A-14]. Decisions: D-46 (N is a maximum; ≤ N distinct routes). Assumptions: AS-18
(infiltration state), AS-21 (MPPI re-weighting), AS-251 (route estimator reading).

Maths
-----
1. **Infiltration curve** (monotone by construction). For route n with per-step hazards
   h_{n,j} = P(first infiltration at j | none before j) ∈ [0, 1]:

       S_n(k) = Π_{j≤k} (1 − h_{n,j})            (survival: no infiltration up to k)
       F_n(k) = 1 − S_n(k)                       (route-wise cumulative infiltration probability)
       P_inf(k) = Σ_n w_n F_n(k),  Σ_n w_n = 1, w_n ≥ 0

   Each factor (1 − h) ∈ [0, 1], so S_n is non-increasing and F_n non-decreasing; a convex
   combination of non-decreasing functions is non-decreasing. In floating point, a product with a
   factor ≤ 1 never rounds above the previous product (round-to-nearest is monotone), so the
   property survives rounding; a final `cummax` removes any summation-order wobble (a no-op in exact
   arithmetic, kept as a guard so the ForecastBundle invariant can never fire on rounding).

2. **Route weights** (AS-251). Routes are drawn from the MPPI-improved policy π̃. Two estimators:
   - "probability": w_r ∝ q(r) = Π_k π̃(a_k | s_k) over *distinct* routes r (build-spec §2.8). A
     duplicate has the same q(r); merged copies get weight 0 (summing the copies would weight a
     route by count·q ≈ N·q², which is not a consistent estimator).
   - "monte_carlo": w_r = count_r / N, the unbiased Monte-Carlo estimator of E_π̃[F].
   Both sum to 1 over the distinct routes, and both leave P_inf monotone.

3. **MPPI re-weighting** (AS-21; Williams et al., "Information Theoretic MPC for Model-Based
   Reinforcement Learning", ICRA 2017; TD-MPC2, Hansen et al. 2024, arXiv:2310.16828). Over a
   candidate set 𝒜_B (the top-B actions of π):

       π̃(a | s) = π(a | s) · exp(Q(s, a)/η) / Σ_{a'∈𝒜_B} π(a' | s) · exp(Q(s, a')/η)

   with Q = r̂ + γ V(s′). As η → ∞ this is π restricted to 𝒜_B and renormalised; with 𝒜_B the whole
   action set it is exactly π (tested).

4. **Weighted quantile** of route curves: for values x_n with weights w_n, the q-quantile is the
   smallest x_(i) (sorted) whose cumulative weight reaches q. Applied per step to F_n(k); because
   each F_n is non-decreasing in k, so is every quantile (quantiles are monotone in the values).

5. **Gumbel-max sampling**: argmax_i (log p_i + G_i), G_i = −log(−log U_i), U ~ U(0,1), draws i ~ p.
   It consumes a fixed number of uniforms per draw, which gives the Advisor common random numbers
   between counterfactual re-imaginations (the same U drive both).

Precision (D-54: fp32 weights and compute, outputs in fp64)
----------------------------------------------------------
Every output of this module is float64: `route_cumulative` keeps the dtype of its (float64) hazards,
`p_inf_from_hazards`, `route_weights` and `mppi_log_weights` compute in and return float64 whatever
their input dtype. Rationale: the survival product Π(1 − h) and the softmax of route log-probabilities
are exactly the quantities whose float32 rounding (unit roundoff 2⁻²⁴ ≈ 6·10⁻⁸; Higham, *Accuracy and
Stability of Numerical Algorithms*, 2nd ed., SIAM 2002, §2.1) reaches the reported probabilities; in
float64 (2⁻⁵³ ≈ 1.1·10⁻¹⁶) it does not. Casting a float32 value to float64 is exact, so nothing is lost
on the way in.

Invariants (tested): P_inf ∈ [0, 1] and non-decreasing; merged routes have weight 0 and the number of
distinct routes is ≤ N; weights sum to 1 over each trigger; every output is float64.

Extension points: other risk summaries of the route distribution can be added beside `weighted_quantile`.
"""

from __future__ import annotations

import math

import torch


# --------------------------------------------------------------------------- infiltration curve
def route_cumulative(hazard: torch.Tensor) -> torch.Tensor:
    """F_n(k) = 1 − Π_{j≤k}(1 − h_{n,j}). hazard [..., K] in [0, 1] → [..., K] (same dtype: pass float64, D-54)."""
    # Survival as a cumulative product of factors in [0, 1]: monotone non-increasing even after rounding.
    survival = torch.cumprod((1.0 - hazard).clamp(0.0, 1.0), dim=-1)    # [..., K]
    return 1.0 - survival


def p_inf_from_hazards(hazard: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """P_inf(k) = Σ_n w_n (1 − Π_{j≤k}(1 − h_{n,j})).

    hazard: [..., N, K] in [0, 1]; weight: [..., N] ≥ 0, summing to 1 over N (zero for merged copies).
    Returns float64 [..., K] (D-54), in [0, 1] and non-decreasing in k. The model path
    (`Forecaster.imagine`) calls exactly this function on its float64 hazards and weights, so P_inf
    recomputed from `ForecastOut.hazard` / `route_weight` equals `ForecastOut.p_inf` bit for bit (tested).
    """
    f = route_cumulative(hazard.double())                                # [..., N, K] float64
    p = (weight.double().unsqueeze(-1) * f).sum(dim=-2)                  # [..., K] float64
    # Guard against summation-order wobble (no-op in exact arithmetic), then clip into [0, 1].
    # The result stays float64: casting back to a narrower input dtype would discard the precision
    # D-54 asks for (the old `.to(hazard.dtype)` did exactly that for float32 hazards).
    return torch.cummax(p, dim=-1).values.clamp(0.0, 1.0)


def mixture_hazard(p_inf: torch.Tensor) -> torch.Tensor:
    """Hazard of the route mixture: h(k) = (P(k) − P(k−1)) / (1 − P(k−1)), P(0) = 0. [..., K] → [..., K].

    Where no probability mass is left at risk (P(k−1) = 1) the conditional hazard is undefined; it is
    reported as 0 there (nothing remains to infiltrate) so the output stays a probability.
    """
    prev = torch.cat([torch.zeros_like(p_inf[..., :1]), p_inf[..., :-1]], dim=-1)   # P(k−1)
    at_risk = 1.0 - prev
    inc = (p_inf - prev).clamp_min(0.0)
    safe = torch.where(at_risk > 0, at_risk, torch.ones_like(at_risk))
    return torch.where(at_risk > 0, (inc / safe).clamp(0.0, 1.0), torch.zeros_like(inc))


# --------------------------------------------------------------------------- merging and weights
def merge_routes(actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Find identical action sequences per trigger (D-46).

    actions: long [..., N, K, A] (A action components, e.g. technique and target).
    Returns (first, count):
    - first: bool [..., N], True for the first occurrence of each distinct sequence;
    - count: long [..., N], for a first occurrence the number of copies of its sequence, else 0.
    """
    flat = actions.flatten(-2)                                            # [..., N, K·A]
    same = (flat.unsqueeze(-2) == flat.unsqueeze(-3)).all(dim=-1)         # [..., N, N]: route i == route j
    n = same.shape[-1]
    earlier = torch.ones(n, n, dtype=torch.bool, device=actions.device).tril(diagonal=-1)  # j < i
    first = ~(same & earlier).any(dim=-1)                                 # no identical earlier route
    count = torch.where(first, same.sum(dim=-1), torch.zeros_like(first, dtype=torch.long))
    return first, count


def route_weights(log_prob: torch.Tensor, first: torch.Tensor, count: torch.Tensor, *, estimator: str) -> torch.Tensor:
    """Normalised route weights over distinct routes (AS-251; see module docstring, part 2).

    log_prob: [..., N] log q(r) = Σ_k log π̃(a_k | s_k); first, count: from `merge_routes`.
    Returns float64 [..., N] weights ≥ 0 (D-54), zero for merged copies, summing to 1 along N.
    """
    if estimator == "probability":
        # Softmax of log q over distinct routes only: w_r = q(r) / Σ_{distinct} q.
        masked = log_prob.double().masked_fill(~first, float("-inf"))
        w = torch.softmax(masked, dim=-1)
    elif estimator == "monte_carlo":
        c = count.double()
        w = c / c.sum(dim=-1, keepdim=True)
    else:
        raise ValueError(f"unknown route estimator {estimator!r}; use 'probability' or 'monte_carlo'")
    # Kept in float64 (D-54): the weights are a reported output and the mixture weights of P_inf.
    return w


# --------------------------------------------------------------------------- MPPI
def mppi_log_weights(log_pi: torch.Tensor, q: torch.Tensor, temperature: float) -> torch.Tensor:
    """log π̃ over a candidate set: log_softmax(log π + Q/η) along the last axis (AS-21).

    `temperature` = η > 0; `math.inf` gives the renormalised policy (no re-weighting).
    log_pi, q: [..., B] → float64 [..., B] (D-54: these log-weights are summed into the route
    log-probability log q(r), whose softmax is the reported route weight).
    """
    if not temperature > 0:
        raise ValueError("MPPI temperature must be > 0")
    # Promote to float64 before the add and the log-softmax (exact for float32 inputs).
    lp = log_pi.double()
    logits = lp if math.isinf(temperature) else lp + q.double() / temperature
    return torch.log_softmax(logits, dim=-1)


def gumbel_argmax(logits: torch.Tensor, uniform: torch.Tensor) -> torch.Tensor:
    """Draw an index ~ softmax(logits) along the last axis using given uniforms U ∈ (0, 1) of the same shape."""
    u = uniform.clamp(1e-12, 1.0 - 1e-7)
    g = -torch.log(-torch.log(u))
    # −inf logits stay −inf after adding a finite Gumbel, so masked options are never drawn.
    return torch.argmax(logits + g, dim=-1)


# --------------------------------------------------------------------------- quantiles and risk
def weighted_quantile(values: torch.Tensor, weight: torch.Tensor, q: float) -> torch.Tensor:
    """q-quantile along axis −2 of values [..., N, K] with weights [..., N] (zero weights ignored) → [..., K]."""
    if not 0.0 <= q <= 1.0:
        raise ValueError("q must be in [0, 1]")
    v = values.transpose(-1, -2)                                          # [..., K, N]
    w = weight.unsqueeze(-2).expand_as(v)                                 # [..., K, N]
    order = torch.argsort(v, dim=-1)
    v_sorted = torch.gather(v, -1, order)
    cw = torch.cumsum(torch.gather(w, -1, order).double(), dim=-1)        # cumulative weight
    total = cw[..., -1:].clamp_min(1e-300)
    # First sorted index whose cumulative share reaches q. Tolerance 1e-6: the cumulative sum of weights
    # carries rounding error (≈1e-7 if a caller passes float32 weights, ≈1e-16 for the float64 weights
    # of D-54), so a share exactly at q (a boundary tie) resolves to the lower value either way. The
    # returned quantile is one of the input values (gathered, not interpolated): float64 in, float64 out.
    reach = (cw / total) >= (q - 1e-6)
    idx = torch.argmax(reach.int(), dim=-1, keepdim=True)                 # first True
    return torch.gather(v_sorted, -1, idx).squeeze(-1)


def cvar_upper(values: torch.Tensor, weight: torch.Tensor, alpha: float) -> torch.Tensor:
    """CVaR_α of the *upper* tail (the worst α share when larger is worse), along the last axis.

    Rockafellar & Uryasev, "Optimization of conditional value-at-risk", Journal of Risk 2(3), 2000:
    CVaR_α(X) = mean of X over the worst α probability mass, with the boundary atom split fractionally:

        CVaR_α = (1/α) Σ_i w_(i) · x_(i) over x sorted in decreasing order, where the weights are
        truncated so that exactly mass α is taken.

    values, weight: [..., N] (weights ≥ 0, normalised internally). Returns [...].
    """
    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must be in (0, 1]")
    w = weight.double() / weight.double().sum(dim=-1, keepdim=True).clamp_min(1e-300)
    order = torch.argsort(values, dim=-1, descending=True)
    x = torch.gather(values.double(), -1, order)
    ws = torch.gather(w, -1, order)
    before = torch.cumsum(ws, dim=-1) - ws                                # mass strictly above each item
    take = (alpha - before).clamp(min=0.0).minimum(ws)                    # fractional share of the boundary atom
    return ((take * x).sum(dim=-1) / alpha).to(values.dtype)
