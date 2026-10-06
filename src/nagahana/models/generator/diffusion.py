"""TabDDPM-style Gaussian diffusion for the numeric fields of a state update, conditioned on the rest.

Purpose
-------
The diffusion member of the owner's Generator families [A-17] (build-spec §2.11 item 4; AS-27, AS-365).
It regenerates a subset of a record's *numeric* cells (counts, bytes, timings, packet statistics) given
everything else observed about that record (its other numeric cells, its categorical / bitmask cells,
its statuses, its stage label). TabDDPM (Kotelnikov et al., "TabDDPM: Modelling Tabular Data with
Diffusion Models", ICML 2023, arXiv:2209.15421) models numerical features with Gaussian diffusion and an
MLP denoiser; this module follows that for the numeric part and conditions on the categorical part
instead of diffusing it (categorical cells are kept from the real record).

Maths (Ho, Jain & Abbeel, "Denoising Diffusion Probabilistic Models", NeurIPS 2020, arXiv:2006.11239)
--------------------------------------------------------------------------------------------------------
Space: z = (signed_log1p(x) − μ_c)/σ_c per column (codec.py). Forward process, closed form:

    q(z_t | z_0) = 𝒩( √ᾱ_t z_0, (1 − ᾱ_t) I ),     ᾱ_t = Π_{s ≤ t} (1 − β_s)

Noise schedule: cosine (Nichol & Dhariwal, "Improved Denoising Diffusion Probabilistic Models", ICML 2021,
arXiv:2102.09672): f(t) = cos²(((t/T + s)/(1 + s))·π/2), ᾱ_t = f(t)/f(0), β_t = min(1 − ᾱ_t/ᾱ_{t−1},
0.999); ᾱ is then recomputed from the clipped β so the closed form above is exact for the β used.

Training (ε-prediction, Ho et al. Eq. 14, "L_simple"): for target cells T of a row,

    L = E_{t, ε} [ Σ_{c ∈ T} ( ε_c − ε̂_θ(z_t, t, cond)_c )² ] / |T|,    z_t = √ᾱ_t z_0 + √(1 − ᾱ_t) ε

Ancestral sampling (Ho et al. Algorithm 2), with the posterior variance β̃_t = (1 − ᾱ_{t−1})/(1 − ᾱ_t)·β_t:

    z_{t−1} = (1/√α_t) ( z_t − (β_t/√(1 − ᾱ_t)) ε̂_θ(z_t, t, cond) ) + √β̃_t · ξ,   ξ ~ 𝒩(0, I)  (ξ = 0 at t = 1)

Only target cells are noised and sampled; condition cells are fed clean every step (conditioning by
input, as in TabDDPM's class conditioning extended to observed fields). Final samples are clipped to the
column's training range in z-space (decoded values never leave the range seen in training), then decoded
and passed through the hard-limit projection and the acceptance gate.

Denoiser: input [z_in ; 𝟙_target ; 𝟙_condition] per numeric column → linear → `denoiser_blocks`
residual blocks x ← x + SwiGLU(RMSNorm(x)) (shared primitives, AS-32), plus embeddings of the time
step (sinusoidal, Vaswani et al. 2017 / Ho et al. §4), of each discrete cell's class, and of the stage.

Invariants (tests/test_generator_learned.py): forward-process moments match the closed form; the DDPM
loss decreases on a toy (fixed seed); excluded cells are never targets and never decoded (D-41).
Decisions: D-40, D-41. Assumptions: AS-27, AS-365, AS-366.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import torch
from torch import nn

from nagahana.models.config.components import GeneratorConfig
from nagahana.models.generator.codec import FieldCodec
from nagahana.models.vocab import N_STAGES
from nagahana.nn.mlp import SwiGLU
from nagahana.nn.norms import RMSNorm


def _rand(shape: Sequence[int], generator: torch.Generator | None, device: torch.device) -> torch.Tensor:
    """Uniform draws on the generator's device, moved to `device` (AS-588: identical draws on every accelerator)."""
    gdev = generator.device if generator is not None else device
    return torch.rand(tuple(shape), generator=generator, device=gdev).to(device)


def _randn(shape: Sequence[int], generator: torch.Generator | None, like: torch.Tensor) -> torch.Tensor:
    """Standard normal draws on the generator's device, moved to `like`'s device and dtype (AS-588)."""
    gdev = generator.device if generator is not None else like.device
    return torch.randn(tuple(shape), generator=generator, device=gdev, dtype=like.dtype).to(like.device)


class GaussianDiffusion:
    """Cosine-schedule DDPM (no learned parameters). All schedule tensors are float64 [T]."""

    def __init__(self, steps: int, *, s: float) -> None:
        if steps < 1:
            raise ValueError("diffusion needs at least one step")
        self.steps = steps
        t = torch.arange(steps + 1, dtype=torch.float64)
        f = torch.cos(((t / steps + s) / (1 + s)) * math.pi / 2) ** 2
        ab = f / f[0]
        betas = (1 - ab[1:] / ab[:-1]).clamp(max=0.999)                      # β_1 … β_T
        self.betas = betas
        self.alphas = 1 - betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)                   # ᾱ_t recomputed from clipped β
        self.alpha_bar_prev = torch.cat([torch.ones(1, dtype=torch.float64), self.alpha_bar[:-1]])
        self.posterior_var = betas * (1 - self.alpha_bar_prev) / (1 - self.alpha_bar)   # β̃_t

    # index convention: t_idx ∈ {0, …, T−1} stands for step t = t_idx + 1
    def q_moments(self, z0: torch.Tensor, t_idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Mean and variance of q(z_t | z_0) per row: (√ᾱ_t z_0, (1 − ᾱ_t))."""
        ab = self.alpha_bar[t_idx].to(z0.dtype).unsqueeze(-1)
        return ab.sqrt() * z0, (1 - ab).expand_as(z0)

    def q_sample(self, z0: torch.Tensor, t_idx: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """z_t = √ᾱ_t z_0 + √(1 − ᾱ_t) ε."""
        ab = self.alpha_bar[t_idx].to(z0.dtype).unsqueeze(-1)
        return ab.sqrt() * z0 + (1 - ab).sqrt() * noise

    def p_step(self, eps_hat: torch.Tensor, z_t: torch.Tensor, t_idx: int, noise: torch.Tensor) -> torch.Tensor:
        """One ancestral step z_t → z_{t−1} (no noise at the last step)."""
        beta, alpha, ab = (float(x[t_idx]) for x in (self.betas, self.alphas, self.alpha_bar))
        mean = (z_t - beta / math.sqrt(1 - ab) * eps_hat) / math.sqrt(alpha)
        if t_idx == 0:
            return mean
        return mean + math.sqrt(float(self.posterior_var[t_idx])) * noise


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal embedding of integer steps [N] → [N, dim] (transformer-style, as in Ho et al. §4)."""
    half = dim // 2
    freqs = torch.exp(-math.log(10_000.0) * torch.arange(half, dtype=torch.float32, device=t.device) / max(half, 1))
    ang = t.float().unsqueeze(-1) * freqs
    emb = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)
    return emb if dim % 2 == 0 else torch.cat([emb, torch.zeros(len(t), 1, device=t.device)], dim=-1)


class TabularDenoiser(nn.Module):
    """ε̂_θ(z_in, t, cond): residual SwiGLU MLP over one record. See the module docstring."""

    def __init__(self, cfg: GeneratorConfig, n_numeric: int, discrete_classes: Sequence[int]) -> None:
        super().__init__()
        h = cfg.denoiser_hidden
        self.n_numeric = n_numeric
        self.inp = nn.Linear(3 * n_numeric, h)
        # one embedding table for every discrete column's classes plus one "absent" row per column
        sizes = [int(k) + 1 for k in discrete_classes]
        offs = np.concatenate([[0], np.cumsum(sizes)[:-1]]).astype(np.int64) if sizes else np.zeros(0, np.int64)
        self.disc_offsets: torch.Tensor
        self.register_buffer("disc_offsets", torch.from_numpy(offs), persistent=False)
        self.disc_sizes = tuple(sizes)
        self.disc = nn.Embedding(max(1, int(sum(sizes))), h)
        self.stage = nn.Embedding(N_STAGES + 1, h)
        self.time = nn.Sequential(nn.Linear(h, h), nn.SiLU(), nn.Linear(h, h))
        self.norms = nn.ModuleList(RMSNorm(h) for _ in range(cfg.denoiser_blocks))
        self.mlps = nn.ModuleList(SwiGLU(h) for _ in range(cfg.denoiser_blocks))
        self.out_norm = RMSNorm(h)
        self.out = nn.Linear(h, n_numeric)
        nn.init.zeros_(self.out.weight)                                    # start as "predict zero noise"
        nn.init.zeros_(self.out.bias)

    def forward(self, z_in: torch.Tensor, target: torch.Tensor, cond: torch.Tensor, disc: torch.Tensor,
                stage: torch.Tensor, t_idx: torch.Tensor) -> torch.Tensor:
        """z_in, target, cond: [N, Cn]; disc: long [N, Cd] classes (−1 absent); stage: [N]; t_idx: [N] → ε̂ [N, Cn]."""
        x = self.inp(torch.cat([z_in, target.float(), cond.float()], dim=-1))
        if disc.shape[1]:
            absent = torch.tensor([s - 1 for s in self.disc_sizes], dtype=torch.long, device=disc.device).view(1, -1)
            idx = torch.where(disc >= 0, disc, absent) + self.disc_offsets.view(1, -1)
            x = x + self.disc(idx).sum(dim=1)
        x = x + self.stage(torch.where(stage >= 0, stage, torch.full_like(stage, N_STAGES)))
        x = x + self.time(timestep_embedding(t_idx, x.shape[-1]))
        for norm, mlp in zip(self.norms, self.mlps, strict=True):
            x = x + mlp(norm(x))
        out: torch.Tensor = self.out(self.out_norm(x))
        return out


class TabularDiffusion(nn.Module):
    """Denoiser + schedule + the codec's column layout. Training: `loss`; generation: `sample`."""

    def __init__(self, cfg: GeneratorConfig, codec: FieldCodec) -> None:
        super().__init__()
        self.codec = codec
        self.schedule = GaussianDiffusion(cfg.diffusion_steps, s=cfg.cosine_s)
        disc_classes = [codec.n_classes(j) for j in codec.discrete_columns]
        self.denoiser = TabularDenoiser(cfg, len(codec.numeric_columns), disc_classes)
        lo = [(codec.numeric[j].lo - codec.numeric[j].mean) / codec.numeric[j].std for j in codec.numeric_columns]
        hi = [(codec.numeric[j].hi - codec.numeric[j].mean) / codec.numeric[j].std for j in codec.numeric_columns]
        self.z_lo: torch.Tensor
        self.z_hi: torch.Tensor
        self.register_buffer("z_lo", torch.tensor(lo, dtype=torch.float32), persistent=False)
        self.register_buffer("z_hi", torch.tensor(hi, dtype=torch.float32), persistent=False)
        self.modelled: torch.Tensor
        self.register_buffer("modelled", torch.tensor([codec.modelled(j) for j in codec.numeric_columns]), persistent=False)

    def random_targets(self, contrib: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
        """Training targets: per row a uniform share of the contributing modelled cells (>= 1 when any)."""
        cand = contrib & self.modelled.view(1, -1)
        share = _rand((cand.shape[0], 1), generator, cand.device)
        tgt = cand & (_rand(cand.shape, generator, cand.device) < share)
        # rows with candidates but no target get one random candidate
        need = cand.any(1) & ~tgt.any(1)
        if bool(need.any()):
            scores = _rand(cand.shape, generator, cand.device).masked_fill(~cand, -1.0)
            pick = scores.argmax(dim=1)
            tgt[need, pick[need]] = True
        return tgt

    def loss(self, z0: torch.Tensor, contrib: torch.Tensor, disc: torch.Tensor, stage: torch.Tensor,
             generator: torch.Generator, target: torch.Tensor | None = None,
             t_idx: torch.Tensor | None = None, noise: torch.Tensor | None = None) -> torch.Tensor:
        """L_simple over target cells (targets / t / ε drawn here unless given, e.g. for a fixed evaluation)."""
        tgt = self.random_targets(contrib, generator) if target is None else target
        cond = contrib & ~tgt                                                  # observed, kept cells
        n = z0.shape[0]
        gdev = generator.device if generator is not None else z0.device
        t = (torch.randint(0, self.schedule.steps, (n,), generator=generator, device=gdev).to(z0.device)
             if t_idx is None else t_idx)
        eps = _randn(z0.shape, generator, z0) if noise is None else noise
        z_t = self.schedule.q_sample(z0, t, eps).float()
        z_in = torch.where(tgt, z_t, torch.where(cond, z0, torch.zeros_like(z0)))
        eps_hat = self.denoiser(z_in, tgt, cond, disc, stage, t)
        se = ((eps_hat - eps) ** 2) * tgt
        return se.sum() / tgt.sum().clamp_min(1)

    @torch.no_grad()
    def sample(self, z0: torch.Tensor, contrib: torch.Tensor, target: torch.Tensor, disc: torch.Tensor,
               stage: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
        """Ancestral sampling of the `target` cells given the rest; returns z [N, Cn] (non-targets = z0)."""
        cond = contrib & ~target
        z = _randn(z0.shape, generator, z0) * target
        for t_idx in range(self.schedule.steps - 1, -1, -1):
            z_in = torch.where(target, z, torch.where(cond, z0, torch.zeros_like(z0)))
            t = torch.full((z0.shape[0],), t_idx, dtype=torch.long, device=z0.device)
            eps_hat = self.denoiser(z_in, target, cond, disc, stage, t)
            z = self.schedule.p_step(eps_hat, z, t_idx, _randn(z0.shape, generator, z0)) * target
        z = torch.maximum(torch.minimum(z, self.z_hi.view(1, -1)), self.z_lo.view(1, -1))   # training range
        return torch.where(target, z, z0)
