"""KL terms of the hybrid latent: DreamerV3 balancing with free bits, and the KL to the fixed prior.

Purpose
-------
Stage 3 (build-spec §3) trains CVG-AE (posterior q) and TSTCT's transition prior p jointly:

    L₃ ⊃ β_dyn · max(fb, KL(sg q_{t+1} ‖ p_{t+1})) + β_rep · max(fb, KL(q_{t+1} ‖ sg p_{t+1}))
         + β_0 · KL(q_first ‖ 𝒩(0, I) × Unif)

This module computes those terms for the hybrid latent z = (z_c ∈ ℝ^{Dc}, z_d ∈ (Δ^C)^G).

Owner sources, decisions and assumptions
----------------------------------------
- D-20 (variational hybrid latent), AS-05 (DreamerV3 settings: β_dyn 0.5, β_rep 0.1, free bits
  1 nat, 1 % unimix), new AS-109 (free bits on the total KL per position; unimix applied where each
  categorical distribution is formed from raw logits).
- Hafner et al., "Mastering Diverse Domains through World Models" (DreamerV3), arXiv:2301.04104,
  §3 (KL balancing, free bits) and the 1 % uniform mix of categorical distributions.

Maths
-----
Diagonal Gaussians q = 𝒩(μ_q, diag σ_q²), p = 𝒩(μ_p, diag σ_p²):

    KL(q ‖ p) = ½ Σ_d [ log σ_p² − log σ_q² + (σ_q² + (μ_q − μ_p)²) / σ_p² − 1 ]

Categoricals per group g, with unimix u: π = (1 − u)·softmax(ℓ) + u/C (`unimix_logits` returns log π):

    KL(q_g ‖ p_g) = Σ_c π^q_c (log π^q_c − log π^p_c)

Total per position: KL = KL_gauss + Σ_g KL_cat,g. DreamerV3 balancing with stop-gradients:

    dyn = max(fb, KL(sg q ‖ p))     trains the prior towards the posterior,
    rep = max(fb, KL(q ‖ sg p))     trains the posterior towards the prior (weakly),
    total = β_dyn · dyn + β_rep · rep.

Free bits clip the *summed* KL of a position (as DreamerV3 clips its summed KL), so a position whose
KL is already below fb gets no gradient from that term: the latent is not pushed to carry less than
fb nats.

Invariants (tests/test_perception_latent.py)
--------------------------------------------
- `kl_gauss` and `kl_cat` equal `torch.distributions.kl_divergence` of the same distributions.
- dyn/rep values are equal (same KL) and their gradients go only to the prior / posterior.
- Below free bits, gradients are zero.

Extension points
----------------
- A learned categorical prior for the first state (instead of uniform) is a change to `kl_standard`.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from nagahana.governance.assumptions import assume


def unimix_logits(logits: torch.Tensor, unimix: float) -> torch.Tensor:
    """log((1 − u)·softmax(ℓ) + u/C) over the last dim (u = 0 returns log-softmax)."""
    if not 0.0 <= unimix < 1.0:
        raise ValueError("unimix must lie in [0, 1)")
    logp = F.log_softmax(logits.float(), dim=-1)
    if unimix == 0.0:
        return logp
    c = logits.shape[-1]
    # log((1−u)·p + u/C) computed as a logsumexp of the two terms (stable for p → 0).
    return torch.logaddexp(logp + math.log1p(-unimix), torch.full_like(logp, math.log(unimix / c)))


def kl_gaussian(mean_q: torch.Tensor, logvar_q: torch.Tensor, mean_p: torch.Tensor, logvar_p: torch.Tensor) -> torch.Tensor:
    """KL(𝒩(μ_q, σ_q²) ‖ 𝒩(μ_p, σ_p²)) for diagonal Gaussians, summed over the last dim (float32)."""
    mq, lq, mp, lp = mean_q.float(), logvar_q.float(), mean_p.float(), logvar_p.float()
    return 0.5 * (lp - lq + (torch.exp(lq) + (mq - mp) ** 2) * torch.exp(-lp) - 1.0).sum(dim=-1)


def kl_categorical_logprobs(logp_q: torch.Tensor, logp_p: torch.Tensor) -> torch.Tensor:
    """Σ_c π^q_c (log π^q_c − log π^p_c) over the last dim, from log-probabilities."""
    return (logp_q.exp() * (logp_q - logp_p)).sum(dim=-1)


def kl_balanced(
    post_mean: torch.Tensor,
    post_logvar: torch.Tensor,
    post_logits: torch.Tensor,
    prior_mean: torch.Tensor,
    prior_logvar: torch.Tensor,
    prior_logits: torch.Tensor,
    *,
    free_bits: float,
    beta_dyn: float,
    beta_rep: float,
    unimix: float,
) -> dict[str, torch.Tensor]:
    """DreamerV3-balanced KL per position. See the module docstring.

    Shapes: means/logvars [..., Dc], logits [..., G, C] (raw, unmixed; `unimix` is applied to both the
    posterior and the prior here, the one place each distribution is formed, AS-109).
    Returns per position [...]: 'kl' (KL(q ‖ p), no stop-gradient), 'kl_gauss', 'kl_cat',
    'dyn' = max(fb, KL(sg q ‖ p)), 'rep' = max(fb, KL(q ‖ sg p)), 'total' = β_dyn·dyn + β_rep·rep.
    """
    assume("AS-05", by=__name__)
    lq = unimix_logits(post_logits, unimix)                                  # [..., G, C]
    lp = unimix_logits(prior_logits, unimix)

    def kl(mq: torch.Tensor, vq: torch.Tensor, lgq: torch.Tensor,
           mp: torch.Tensor, vp: torch.Tensor, lgp: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return kl_gaussian(mq, vq, mp, vp), kl_categorical_logprobs(lgq, lgp).sum(dim=-1)

    # Raw KL (for logging and evaluation).
    kg, kc = kl(post_mean, post_logvar, lq, prior_mean, prior_logvar, lp)
    # Dynamics term: the posterior is a fixed target; gradients reach the prior only.
    dg, dc = kl(post_mean.detach(), post_logvar.detach(), lq.detach(), prior_mean, prior_logvar, lp)
    # Representation term: the prior is a fixed target; gradients reach the posterior only.
    rg, rc = kl(post_mean, post_logvar, lq, prior_mean.detach(), prior_logvar.detach(), lp.detach())
    fb = torch.tensor(float(free_bits))
    dyn = torch.maximum(dg + dc, fb)
    rep = torch.maximum(rg + rc, fb)
    return {"kl": kg + kc, "kl_gauss": kg, "kl_cat": kc, "dyn": dyn, "rep": rep, "total": beta_dyn * dyn + beta_rep * rep}


def kl_standard(mean: torch.Tensor, logvar: torch.Tensor, logits: torch.Tensor, *, unimix: float) -> torch.Tensor:
    """KL(q ‖ 𝒩(0, I) × Unif(C)^G) per position [...]: the first state of an entity (β_0 term).

    logits are raw; the posterior is formed with `unimix` (the uniform prior is unchanged by mixing).
    KL(π ‖ Unif) = Σ_c π_c log π_c + log C.
    """
    lq = unimix_logits(logits, unimix)                                       # [..., G, C]
    kl_cat = ((lq.exp() * lq).sum(dim=-1) + math.log(logits.shape[-1])).sum(dim=-1)
    m, lv = mean.float(), logvar.float()
    kl_gauss = 0.5 * (torch.exp(lv) + m**2 - 1.0 - lv).sum(dim=-1)
    return kl_gauss + kl_cat
