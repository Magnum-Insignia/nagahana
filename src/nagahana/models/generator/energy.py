"""Energy-based self-supervised Generator family: a denoising-score-matched record energy, sampled by SGLD.

Purpose
-------
[A-17] names "energy based modeling via ssl" among the Generator's methods. This family learns an
energy over the numeric cells of a state update, conditioned on everything else observed about it,
by self-supervision (denoising), and makes variants by Langevin dynamics on that energy: a real record's
cells are partially re-noised and descended back onto the learned data manifold, so a variant keeps the
record's act (its other cells, its statuses, its stage label) and changes its realisation. It is
training-only augmentation (D-40) under AS-27 (D-14 held; option in force "all families") and AS-586.

Maths
-----
Space: standardised signed-log1p values z (codec.py, as the diffusion family; AS-365). For one record,
target cells T move, condition cells C (contributing, not targets) are fixed, excluded cells do not
exist (D-41).

Energy (noise-conditional, Song and Ermon, "Generative Modeling by Estimating Gradients of the Data
Distribution", NeurIPS 2019, arXiv:1907.05600; energy parameterisation as in Salimans and Ho, "Should
EBMs model the energy or the score?", 2021, arXiv:2101.03288):

    E_phi(x, sigma | cond) = e_phi(x_T, log sigma, z_C, discrete cells, stage) / sigma,     s = -grad_x E

Training, denoising score matching (Vincent, Neural Computation 23(7), 2011) with the weighting
lambda(sigma) = sigma^2 of Song and Ermon, which with the 1/sigma parameterisation is sigma-free:

    x~_T = x_T + sigma eps,  eps ~ N(0, I),  sigma ~ Uniform over the ladder
    L = E [ || sigma s_phi(x~, sigma)_T + eps ||^2 ] / |T| = E [ || -grad_{x_T} e_phi(x~, ...) + eps ||^2 ] / |T|

whose minimiser is the score of the sigma-smoothed data distribution.

Sampling, annealed Langevin dynamics (Song and Ermon, Algorithm 1) on the target cells, started from
the real record re-noised at the largest level (a guided start, as in SDEdit, Meng et al., ICLR 2022,
arXiv:2108.01073):

    sigma_1 > ... > sigma_L geometric;   alpha_i = eps_step sigma_i^2 / sigma_L^2
    x <- x - (alpha_i / 2) grad_x E(x, sigma_i) + sqrt(alpha_i) xi,   xi ~ N(0, I),   T steps per level

This is stochastic gradient Langevin dynamics (Welling and Teh, ICML 2011) with exact gradients, i.e.
the unadjusted Langevin algorithm (ULA): its stationary law approaches p(x) ~ exp(-E(x)) as alpha -> 0,
with an O(alpha) bias (for E = (x - mu)^2 / (2 s^2): variance s^2 / (1 - alpha / (4 s^2))). With
`metropolis`, the steps of the last level are Metropolis-adjusted (MALA, Roberts and Tweedie,
Bernoulli 2(4), 1996): a proposal x' is accepted with probability

    min(1, exp(-E(x') + E(x) - log q(x | x') + log q(x' | x))),   log q(b | a) = -|| b - a + (alpha/2) grad E(a) ||^2 / (2 alpha)

so the last level samples its energy exactly. `langevin` is the generic sampler; tests check it on a
Gaussian energy: mean and variance match the target (MALA) and ULA matches its closed-form bias.

A variant: targets are drawn as in the other learned families (each contributing modelled numeric cell
with probability `regen_fraction`, at least one per row; AS-366), sampled, clipped to the training range,
decoded, moved into the hard limits (`limits.project_hard_limits`, AS-359) and passed to the acceptance
gate; labels are copied from the source rows (label mode "copied").

Randomness on the device (AS-588): draws are made on the generator's device and moved to the tensors'.

Decisions: D-40, D-41, D-14 (held). Assumptions: AS-27, AS-359, AS-365, AS-366, AS-586.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from nagahana.core.errors import InvariantViolation
from nagahana.core.modes import RunMode, require_mode
from nagahana.datamodel.columnar import ColumnarUpdates
from nagahana.models.config.components import GeneratorConfig
from nagahana.models.generator.assumptions import use
from nagahana.models.generator.codec import FieldCodec
from nagahana.models.generator.config import GeneratorPolicy, LangevinConfig
from nagahana.models.generator.limits import PhysicalSetting, project_hard_limits
from nagahana.models.generator.variants import TransformResult, UpdateLabels, changed_cells, derive
from nagahana.models.vocab import N_STAGES
from nagahana.nn.mlp import SwiGLU
from nagahana.nn.norms import RMSNorm
from nagahana.pipeline.splits import Sample

EnergyFn = Callable[[torch.Tensor], torch.Tensor]


def _randn(shape: Sequence[int], generator: torch.Generator | None, like: torch.Tensor) -> torch.Tensor:
    """Standard normal draws on the generator's device, moved to `like`'s device and dtype (AS-588)."""
    dev = generator.device if generator is not None else like.device
    return torch.randn(tuple(shape), generator=generator, device=dev, dtype=like.dtype).to(like.device)


def _rand(shape: Sequence[int], generator: torch.Generator | None, like: torch.Tensor) -> torch.Tensor:
    dev = generator.device if generator is not None else like.device
    return torch.rand(tuple(shape), generator=generator, device=dev, dtype=like.dtype).to(like.device)


def energy_and_grad(energy_fn: EnergyFn, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(E(x) per row [N], grad_x sum E [N, D]) without building a graph for the caller."""
    with torch.enable_grad():
        xg = x.detach().requires_grad_(True)
        e = energy_fn(xg)
        (g,) = torch.autograd.grad(e.sum(), xg)
    return e.detach(), g.detach()


@dataclass
class LangevinStats:
    """What a chain did: steps taken and the Metropolis acceptance rate (NaN when not adjusted)."""

    steps: int
    acceptance: float


def langevin(energy_fn: EnergyFn, x0: torch.Tensor, *, step_size: float, steps: int, mask: torch.Tensor | None = None,
             generator: torch.Generator | None = None, metropolis: bool = False) -> tuple[torch.Tensor, LangevinStats]:
    """Langevin chains on E: ULA (SGLD with exact gradients) or MALA (module docstring).

    x0: [N, D] one chain per row; energy_fn: x [N, D] -> E [N]; mask: bool [N, D] cells that move (others
    fixed). Returns (x after `steps` steps, statistics).
    """
    if steps < 0 or not step_size > 0:
        raise ValueError("need steps >= 0 and step_size > 0")
    move = torch.ones_like(x0, dtype=torch.bool) if mask is None else mask.to(torch.bool)
    m = move.to(x0.dtype)
    x = x0.detach().clone()
    e, g = energy_and_grad(energy_fn, x)
    accepted, proposed = 0.0, 0.0
    a = float(step_size)
    for _ in range(steps):
        noise = _randn(x.shape, generator, x)
        prop = x + m * (-0.5 * a * g + math.sqrt(a) * noise)
        if not metropolis:
            x = prop
            e, g = energy_and_grad(energy_fn, x)
            continue
        e_p, g_p = energy_and_grad(energy_fn, prop)
        # log q(prop | x) and log q(x | prop) over the moving cells only.
        fwd = -(((prop - x + 0.5 * a * g) * m) ** 2).sum(-1) / (2 * a)
        bwd = -(((x - prop + 0.5 * a * g_p) * m) ** 2).sum(-1) / (2 * a)
        log_ratio = (-e_p + e + bwd - fwd).to(torch.float64)
        u = _rand((x.shape[0],), generator, log_ratio).clamp_min(1e-300)
        ok = (torch.log(u) < log_ratio).to(x.dtype)[:, None]
        x = ok * prop + (1 - ok) * x
        e = torch.where(ok[:, 0] > 0, e_p, e)
        g = ok * g_p + (1 - ok) * g
        accepted += float(ok.sum())
        proposed += float(x.shape[0])
    return x, LangevinStats(steps=steps, acceptance=(accepted / proposed) if proposed else float("nan"))


def noise_ladder(cfg: LangevinConfig) -> torch.Tensor:
    """Geometric sigma_1 > ... > sigma_L (float64)."""
    if cfg.levels == 1:
        return torch.tensor([cfg.sigma_min], dtype=torch.float64)
    return torch.exp(torch.linspace(math.log(cfg.sigma_max), math.log(cfg.sigma_min), cfg.levels, dtype=torch.float64))


def annealed_langevin(energy_at: Callable[[torch.Tensor, float], torch.Tensor], x0: torch.Tensor, *, mask: torch.Tensor,
                      cfg: LangevinConfig, generator: torch.Generator | None) -> tuple[torch.Tensor, LangevinStats]:
    """Song and Ermon's Algorithm 1 from a guided start; the last level Metropolis-adjusted when configured."""
    sig = noise_ladder(cfg)
    m = mask.to(x0.dtype)
    x = x0 + m * float(sig[0]) * _randn(x0.shape, generator, x0)               # re-noise at sigma_1 (guided start)
    last = float(sig[-1])
    stats = LangevinStats(steps=0, acceptance=float("nan"))
    for i, s in enumerate(sig.tolist()):
        alpha = cfg.step_scale * (s / last) ** 2
        adjusted = cfg.metropolis and i == len(sig) - 1
        x, st = langevin(lambda z, s=s: energy_at(z, s), x, step_size=alpha, steps=cfg.steps_per_level, mask=mask,
                         generator=generator, metropolis=adjusted)
        stats = LangevinStats(steps=stats.steps + st.steps, acceptance=st.acceptance if adjusted else stats.acceptance)
    return x, stats


class RecordEnergy(nn.Module):
    """e_phi(x_T, log sigma, z_C, discrete cells, stage): a residual SwiGLU network to one energy per row."""

    def __init__(self, n_numeric: int, discrete_classes: Sequence[int], *, hidden: int, blocks: int) -> None:
        super().__init__()
        self.n_numeric = n_numeric
        self.inp = nn.Linear(4 * n_numeric, hidden)                          # [x_T ; 1_T ; z_C ; 1_C]
        sizes = [int(k) + 1 for k in discrete_classes]                        # one "absent" row per column
        offs = np.concatenate([[0], np.cumsum(sizes)[:-1]]).astype(np.int64) if sizes else np.zeros(0, np.int64)
        self.disc_offsets: torch.Tensor
        self.register_buffer("disc_offsets", torch.from_numpy(offs), persistent=False)
        self.disc_sizes = tuple(sizes)
        self.disc = nn.Embedding(max(1, int(sum(sizes))), hidden)
        self.stage = nn.Embedding(N_STAGES + 1, hidden)
        self.sigma = nn.Sequential(nn.Linear(2, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.norms = nn.ModuleList(RMSNorm(hidden) for _ in range(blocks))
        self.mlps = nn.ModuleList(SwiGLU(hidden) for _ in range(blocks))
        self.out_norm = RMSNorm(hidden)
        self.out = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor, target: torch.Tensor, z_cond: torch.Tensor, cond: torch.Tensor, disc: torch.Tensor,
                stage: torch.Tensor, log_sigma: torch.Tensor) -> torch.Tensor:
        """x, target, z_cond, cond: [N, Cn]; disc: long [N, Cd] (-1 absent); stage [N]; log_sigma [N] -> e [N]."""
        t, c = target.to(x.dtype), cond.to(x.dtype)
        h = self.inp(torch.cat([x * t, t, z_cond * c, c], dim=-1))
        if disc.shape[1]:
            absent = torch.tensor([s - 1 for s in self.disc_sizes], dtype=torch.long, device=disc.device).view(1, -1)
            idx = torch.where(disc >= 0, disc, absent) + self.disc_offsets.view(1, -1)
            h = h + self.disc(idx).sum(dim=1)
        h = h + self.stage(torch.where(stage >= 0, stage, torch.full_like(stage, N_STAGES)))
        ls = log_sigma.to(x.dtype).unsqueeze(-1)
        h = h + self.sigma(torch.cat([ls, torch.exp(-ls)], dim=-1))
        for norm, mlp in zip(self.norms, self.mlps, strict=True):
            h = h + mlp(norm(h))
        out: torch.Tensor = self.out(self.out_norm(h)).squeeze(-1)
        return out


class EnergySSL(nn.Module):
    """The family's model: codec layout, record energy, ladder. Training: `loss`; generation: `sample`."""

    def __init__(self, cfg: GeneratorConfig, codec: FieldCodec, policy: GeneratorPolicy) -> None:
        super().__init__()
        use("AS-586", by=__name__)
        self.codec = codec
        self.policy = policy
        disc_classes = [codec.n_classes(j) for j in codec.discrete_columns]
        self.energy = RecordEnergy(len(codec.numeric_columns), disc_classes, hidden=policy.energy_hidden,
                                   blocks=policy.energy_blocks)
        self.register_buffer("sigmas", noise_ladder(policy.langevin), persistent=False)
        self.sigmas: torch.Tensor
        lo = [(codec.numeric[j].lo - codec.numeric[j].mean) / codec.numeric[j].std for j in codec.numeric_columns]
        hi = [(codec.numeric[j].hi - codec.numeric[j].mean) / codec.numeric[j].std for j in codec.numeric_columns]
        self.register_buffer("z_lo", torch.tensor(lo, dtype=torch.float32), persistent=False)
        self.register_buffer("z_hi", torch.tensor(hi, dtype=torch.float32), persistent=False)
        self.register_buffer("modelled", torch.tensor([codec.modelled(j) for j in codec.numeric_columns], dtype=torch.bool),
                             persistent=False)
        self.z_lo: torch.Tensor
        self.z_hi: torch.Tensor
        self.modelled: torch.Tensor

    def e_of(self, x: torch.Tensor, sigma: torch.Tensor | float, *, target: torch.Tensor, z0: torch.Tensor,
             cond: torch.Tensor, disc: torch.Tensor, stage: torch.Tensor) -> torch.Tensor:
        """E(x, sigma | cond) = e_phi(...) / sigma per row (module docstring)."""
        s = torch.as_tensor(sigma, dtype=x.dtype, device=x.device).expand(x.shape[0])
        e = self.energy(x, target, z0, cond, disc, stage, torch.log(s))
        return e / s

    def random_targets(self, contrib: torch.Tensor, generator: torch.Generator | None) -> torch.Tensor:
        """Training targets: per row a uniform share of the contributing modelled cells (at least one when any)."""
        cand = contrib & self.modelled.view(1, -1)
        share = _rand((cand.shape[0], 1), generator, torch.zeros((), device=cand.device))
        tgt = cand & (_rand(cand.shape, generator, torch.zeros((), device=cand.device)) < share)
        need = cand.any(1) & ~tgt.any(1)
        if bool(need.any()):
            scores = _rand(cand.shape, generator, torch.zeros((), device=cand.device)).masked_fill(~cand, -1.0)
            pick = scores.argmax(dim=1)
            tgt[need, pick[need]] = True
        return tgt

    def loss(self, z0: torch.Tensor, contrib: torch.Tensor, disc: torch.Tensor, stage: torch.Tensor,
             generator: torch.Generator | None, target: torch.Tensor | None = None) -> torch.Tensor:
        """Denoising score matching over target cells (module docstring)."""
        tgt = self.random_targets(contrib, generator) if target is None else target
        cond = contrib & ~tgt
        n = z0.shape[0]
        lev = torch.randint(0, self.sigmas.numel(), (n,), generator=generator,
                            device=generator.device if generator is not None else z0.device).to(z0.device)
        sigma = self.sigmas.to(z0.dtype)[lev]                                      # [N]
        eps = _randn(z0.shape, generator, z0)
        x_t = torch.where(tgt, z0 + sigma[:, None] * eps, torch.zeros_like(z0)).requires_grad_(True)
        e = self.energy(x_t, tgt, z0, cond, disc, stage, torch.log(sigma))
        (grad,) = torch.autograd.grad(e.sum(), x_t, create_graph=True)
        # sigma * score = -grad_x e_phi (the 1/sigma parameterisation): residual against -eps.
        res = (-grad + eps) * tgt
        return (res ** 2).sum() / tgt.sum().clamp_min(1)

    def sample(self, z0: torch.Tensor, contrib: torch.Tensor, target: torch.Tensor, disc: torch.Tensor, stage: torch.Tensor,
               generator: torch.Generator | None) -> tuple[torch.Tensor, LangevinStats]:
        """Annealed Langevin on the target cells from the re-noised record; returns (z [N, Cn], statistics)."""
        cond = contrib & ~target
        start = torch.where(target, z0, torch.zeros_like(z0))

        def energy_at(x: torch.Tensor, s: float) -> torch.Tensor:
            return self.e_of(x, s, target=target, z0=z0, cond=cond, disc=disc, stage=stage)

        x, stats = annealed_langevin(energy_at, start, mask=target, cfg=self.policy.langevin, generator=generator)
        x = torch.maximum(torch.minimum(x, self.z_hi.view(1, -1)), self.z_lo.view(1, -1))   # training range
        return torch.where(target, x, z0).detach(), stats


def _target_cells(cand: np.ndarray, fraction: float, rng: np.random.Generator) -> np.ndarray:
    # Each candidate cell with probability `fraction`; rows with candidates get at least one target (AS-366).
    tgt = cand & (rng.random(cand.shape) < fraction)
    need = cand.any(1) & ~tgt.any(1)
    for i in np.flatnonzero(need):
        tgt[i, rng.choice(np.flatnonzero(cand[i]))] = True
    return tgt


class EnergyProducer:
    """Energy-SSL variant producer (module docstring)."""

    def __init__(self, model: EnergySSL, cfg: GeneratorConfig, setting: PhysicalSetting) -> None:
        self.model, self.cfg, self.setting = model, cfg, setting
        self.name = "energy-ssl"

    def apply(self, cu: ColumnarUpdates, labels: UpdateLabels, rng: np.random.Generator) -> TransformResult:
        use("AS-586", by=__name__)
        codec = self.model.codec
        codec.check_columns(cu.columns)
        z0, contrib_n = codec.to_std(cu.values, cu.status)                       # [U, Cn]
        modelled = self.model.modelled.cpu().numpy()
        target_n = _target_cells(contrib_n & modelled[None, :], self.cfg.regen_fraction, rng)
        disc = codec.encode(cu.values, cu.status)[:, codec.discrete_columns]       # [U, Cd]
        gen = torch.Generator().manual_seed(int(rng.integers(0, 2**62)))
        dev = next(self.model.parameters()).device
        self.model.eval()
        z, stats = self.model.sample(torch.from_numpy(z0).to(dev), torch.from_numpy(contrib_n).to(dev),
                                     torch.from_numpy(target_n).to(dev), torch.from_numpy(disc).to(dev),
                                     torch.from_numpy(labels.stage).to(dev), gen)
        new_numeric = codec.from_std(z.cpu().numpy())
        values = cu.values.copy()
        target = np.zeros(cu.values.shape, dtype=bool)
        for k, j in enumerate(codec.numeric_columns):
            values[target_n[:, k], j] = new_numeric[target_n[:, k], k]
            target[:, j] = target_n[:, k]
        use("AS-359", by=__name__)
        values = project_hard_limits(values, cu.contributing_cells(), cu.columns, self.setting, target)
        rows = np.arange(len(cu), dtype=np.int64)
        out = derive(cu, rows, values=values, status=cu.status)
        return TransformResult(updates=out, labels=labels.take(rows), source_rows=rows,
                               changed=changed_cells(cu, rows, out.values, out.status), producer=self.name,
                               label_mode="copied", params={"regenerated": int(target.sum()), "acceptance": stats.acceptance},
                               free=target)


def fit_energy_ssl(model: EnergySSL, windows: Sequence[tuple[ColumnarUpdates, UpdateLabels, Sample]], *, steps: int,
                   lr: float, seed: int) -> list[float]:
    """Train the record energy on real training windows (AdamW; training split only, AS-367). Returns the losses."""
    from nagahana.models.generator.learned import check_source

    require_mode(RunMode.TRAIN, component="Generator")
    for _, _, s in windows:
        check_source(s)
    if not windows:
        raise InvariantViolation("the energy-SSL family needs at least one training window")
    codec = model.codec
    gen = torch.Generator().manual_seed(seed)
    dev = next(model.parameters()).device
    rows = []
    for cu, labels, _ in windows:
        codec.check_columns(cu.columns)
        z0, contrib = codec.to_std(cu.values, cu.status)
        disc = codec.encode(cu.values, cu.status)[:, codec.discrete_columns]
        rows.append((torch.from_numpy(z0).to(dev), torch.from_numpy(contrib).to(dev), torch.from_numpy(disc).to(dev),
                     torch.from_numpy(labels.stage).to(dev)))
    opt = torch.optim.AdamW(model.energy.parameters(), lr=lr)
    model.train()
    losses: list[float] = []
    for step in range(steps):
        z_t, c_t, d_t, s_t = rows[step % len(rows)]
        loss = model.loss(z_t, c_t, d_t, s_t, gen)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    return losses


__all__ = ["EnergyProducer", "EnergySSL", "LangevinStats", "RecordEnergy", "annealed_langevin", "energy_and_grad",
           "fit_energy_ssl", "langevin", "noise_ladder"]
