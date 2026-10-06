"""Where a state sits: time, graph structure and clock (owner's decisions D-49 and D-50, 2026-10-02).

Nothing here encodes an *index*. An attacker can inflate an index by sending traffic; it cannot
inflate time (D-49). Four encodings:

1. Continuous-time rotary encoding (`TimeRotary`)
-------------------------------------------------
RoPE (Su et al. 2021, arXiv:2104.09864) rotates each pair of query/key dimensions by an angle
proportional to position, so that ⟨R(θ_i) q, R(θ_j) k⟩ depends on θ_i − θ_j only. Here the
position is event time t in seconds:

    θ_{i,m} = ω_m · (t_i − t_0),   ω_m = 2π / p_m,   p_m log-spaced in [p_min, p_max]

so an attention score depends only on the real time between two states. Defaults p_min = 1 ms,
p_max = 1 week cover packet timing up to weekly rhythms. Properties used elsewhere:
- **Cache-friendly**: a key is rotated once, by its own time, when written; it stays valid forever.
- **Shared past/future axis**: the Forecaster's imagined step k uses t = k · window_seconds (D-49).
- **Precision**: angles are computed in float64 relative to an origin t_0 and reduced mod 2π before
  the float32 cast. Epoch seconds (~1.7e9) in float32 have a resolution of 128 s; float64 keeps
  sub-microsecond resolution.

2. Log-Δt bucket bias (`LogDeltaBias`)
--------------------------------------
A learned per-head scalar for each bucket b(Δt) = ⌊log₂(1 + Δt/δ₀)⌋ (clipped). Rotary encodes
*phase*; the bias lets a head learn *recency* (ALiBi-like decay, Press et al. ICLR 2022,
arXiv:2108.12409, but learned per bucket as in T5's relative buckets, Raffel et al. JMLR 2020).

3. Clock features (`ClockFeatures`, D-50)
-----------------------------------------
sin/cos of time of day and day of week (UTC). Built as a switch that callers keep **off** in training
on lab datasets (CIC-IDS2018 attacks run at fixed clock times: a shortcut, Arp et al., USENIX
Security 2022) and turn on for site calibration.

4. Random-walk structural encodings (`random_walk_se`)
------------------------------------------------------
RWSE (Dwivedi et al., ICLR 2022, arXiv:2110.07875): for M = D⁻¹A, the vector
(M_vv, (M²)_vv, …, (M^k)_vv) per node: the probability of returning to v after 1…k steps. It is
permutation-equivariant (relabelling nodes permutes the output rows) and needs no eigenvectors, so
it has none of the sign/basis ambiguity of Laplacian encodings (Lim et al., "Sign and Basis
Invariant Networks", ICLR 2023, arXiv:2202.13013).
"""

from __future__ import annotations

import math

import torch
from torch import nn

SECONDS_PER_DAY = 86_400.0
SECONDS_PER_WEEK = 7 * SECONDS_PER_DAY


# ----------------------------------------------------------------------------------------- rotary
class TimeRotary(nn.Module):
    """Continuous-time rotary encoding for one head width (see the module docstring, part 1).

    Parameters
    ----------
    head_dim: d_h (even). d_h/2 frequencies.
    p_min, p_max: shortest and longest period, in seconds.
    """

    def __init__(self, head_dim: int, *, p_min: float = 1e-3, p_max: float = SECONDS_PER_WEEK) -> None:
        super().__init__()
        if head_dim % 2:
            raise ValueError("head_dim must be even for rotary encoding")
        if not 0 < p_min < p_max:
            raise ValueError("need 0 < p_min < p_max")
        n = head_dim // 2
        periods = torch.logspace(math.log10(p_min), math.log10(p_max), n, dtype=torch.float64)
        # Fixed, not learned: a learned frequency could drift to alias two different gaps.
        self.omega: torch.Tensor
        self.register_buffer("omega", (2 * math.pi) / periods, persistent=False)

    def angles(self, t: torch.Tensor, origin: torch.Tensor | float = 0.0) -> tuple[torch.Tensor, torch.Tensor]:
        """cos and sin for times `t` [...] → two tensors [..., head_dim] (float32).

        `t` and `origin` should be float64 seconds; any float dtype is promoted to float64 first.
        """
        rel = t.to(torch.float64) - (origin if isinstance(origin, float) else origin.to(torch.float64))
        ang = torch.remainder(rel.unsqueeze(-1) * self.omega, 2 * math.pi)  # reduce before the cast
        ang = torch.cat([ang, ang], dim=-1).to(torch.float32)               # GPT-NeoX half layout
        return torch.cos(ang), torch.sin(ang)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """(x1, x2) → (−x2, x1) on the two halves of the last dimension."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate `x` [..., d_h] by precomputed cos/sin [..., d_h] (broadcastable)."""
    return (x.float() * cos + rotate_half(x.float()) * sin).to(x.dtype)


# ------------------------------------------------------------------------------- recency bias
class LogDeltaBias(nn.Module):
    """Learned per-head bias by log-time bucket of Δt ≥ 0 (see the module docstring, part 2).

    Parameters
    ----------
    n_heads: heads that receive the bias.
    n_buckets: number of buckets; bucket b covers Δt ∈ [δ₀(2^b − 1), δ₀(2^{b+1} − 1)).
    delta0: δ₀ in seconds (1 ms default; with 32 buckets the last starts at ≈ 49.7 days).
    """

    def __init__(self, n_heads: int, *, n_buckets: int = 32, delta0: float = 1e-3) -> None:
        super().__init__()
        self.n_buckets, self.delta0 = n_buckets, delta0
        self.table = nn.Parameter(torch.zeros(n_buckets, n_heads))  # starts neutral

    def bucket(self, dt: torch.Tensor) -> torch.Tensor:
        """Bucket index of Δt (negative Δt, which masks exclude, maps to bucket 0)."""
        b = torch.floor(torch.log2(1.0 + dt.to(torch.float64).clamp_min(0.0) / self.delta0))
        return b.clamp(0, self.n_buckets - 1).long()

    def forward(self, dt: torch.Tensor) -> torch.Tensor:
        """Δt [...] → bias [..., n_heads]."""
        return self.table[self.bucket(dt)]


# ------------------------------------------------------------------------------------- clock
class ClockFeatures(nn.Module):
    """sin/cos of time of day and day of week → a learned projection (D-50). Off unless enabled."""

    def __init__(self, out_dim: int, *, enabled: bool) -> None:
        super().__init__()
        self.enabled = enabled
        self.proj = nn.Linear(4, out_dim, bias=False)

    def forward(self, epoch_seconds: torch.Tensor) -> torch.Tensor:
        """Absolute epoch seconds [...] (float64) → [..., out_dim]; zeros when disabled."""
        if not self.enabled:
            return epoch_seconds.new_zeros(*epoch_seconds.shape, self.proj.out_features, dtype=self.proj.weight.dtype)
        t = epoch_seconds.to(torch.float64)
        day = torch.remainder(t, SECONDS_PER_DAY) / SECONDS_PER_DAY * 2 * math.pi
        # 1970-01-01 was a Thursday; the phase offset does not matter to a learned projection.
        week = torch.remainder(t, SECONDS_PER_WEEK) / SECONDS_PER_WEEK * 2 * math.pi
        feats = torch.stack([torch.sin(day), torch.cos(day), torch.sin(week), torch.cos(week)], dim=-1)
        return self.proj(feats.to(self.proj.weight.dtype))


# ---------------------------------------------------------------------------------- RWSE
def random_walk_se(adjacency: torch.Tensor, steps: int) -> torch.Tensor:
    """RWSE of a small dense graph: [V, V] (weights ≥ 0) → [V, steps], row v = ((Mᵏ)_vv)_{k=1..steps}.

    Isolated nodes (degree 0) get a self-loop for the walk, so their encoding is all ones (they
    always "return"), which is distinguishable from any connected node.
    """
    if adjacency.dim() != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("adjacency must be square")
    a = adjacency.to(torch.float64).clone()
    a.fill_diagonal_(0.0)
    deg = a.sum(dim=1)
    isolated = deg == 0
    a[isolated, isolated] = 1.0                       # self-loop only where needed
    m = a / a.sum(dim=1, keepdim=True)                # D⁻¹A, rows sum to 1
    out = torch.empty(a.shape[0], steps, dtype=torch.float64)
    p = m
    for k in range(steps):
        out[:, k] = torch.diagonal(p)
        p = p @ m
    return out.to(torch.float32)
