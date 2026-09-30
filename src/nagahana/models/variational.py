"""Variational building blocks for CVG-AE (D-20): sampling and KL terms, with the maths.

Continuous factors: diagonal Gaussian posterior
-----------------------------------------------
    q(z_cont | 𝒢) = 𝒩(μ, diag σ²),   σ² = exp(logvar)
    sample:  z = μ + σ ⊙ ε,  ε ~ 𝒩(0, I)                        (reparameterisation; Kingma & Welling 2014)
    KL(q ‖ 𝒩(0, I)) = ½ Σ_d ( σ_d² + μ_d² − 1 − log σ_d² )

Discrete factors: categorical posterior with straight-through gradients
-----------------------------------------------------------------------
    q(z_disc,g | 𝒢) = Cat(softmax(ℓ_g)),  g = 1…G
    sample:  one_hot(k) + p − stop_grad(p),  k ~ Cat(p)          (DreamerV3-style, refs.md#L221)
The forward value is an exact one-hot sample. The gradient flows as if the output were p.
    KL(q ‖ prior) = Σ_c q_c ( log q_c − log p_c )

Choices deliberately left to config and experiments, not defaulted here: the categorical prior
(uniform? learned?), KL balancing or free bits (as in DreamerV3), and any uniform mixing of
probabilities. `kl_categorical` therefore requires the prior explicitly.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def reparameterize(mean: torch.Tensor, logvar: torch.Tensor, *, generator: torch.Generator | None = None) -> torch.Tensor:
    """z = μ + exp(½·logvar) ⊙ ε with ε ~ 𝒩(0, I)."""
    eps = torch.randn(mean.shape, generator=generator, dtype=mean.dtype, device=mean.device)
    return mean + torch.exp(0.5 * logvar) * eps


def kl_diag_gaussian(mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """KL(𝒩(μ, diag σ²) ‖ 𝒩(0, I)), summed over the last dimension."""
    return 0.5 * (torch.exp(logvar) + mean**2 - 1.0 - logvar).sum(dim=-1)


def straight_through_categorical(logits: torch.Tensor, *, generator: torch.Generator | None = None) -> torch.Tensor:
    """One-hot categorical samples over the last dim with straight-through gradients."""
    probs = F.softmax(logits, dim=-1)
    c = probs.shape[-1]
    idx = torch.multinomial(probs.reshape(-1, c), 1, generator=generator).reshape(probs.shape[:-1])
    one_hot = F.one_hot(idx, c).to(probs.dtype)
    return one_hot + probs - probs.detach()


def kl_categorical(logits: torch.Tensor, prior_logits: torch.Tensor) -> torch.Tensor:
    """KL(Cat(softmax(logits)) ‖ Cat(softmax(prior_logits))) over the last dim.

    The prior is required: choosing it (uniform, learned, balanced) is a modelling decision.
    """
    log_q = F.log_softmax(logits, dim=-1)
    log_p = F.log_softmax(prior_logits, dim=-1)
    return (log_q.exp() * (log_q - log_p)).sum(dim=-1)
