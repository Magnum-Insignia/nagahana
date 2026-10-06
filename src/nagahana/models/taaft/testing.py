"""Test support for TAAFT: a contract-shaped fake Environment and a minimal decoder for the physics term.

Why this exists
---------------
TAAFT consumes `EnvironmentOut` (TSTCT) and a Decoder, both built in parallel by other engineers.
To test TAAFT against the *contracts* (batch.py) without their implementations, these fakes
produce tensors of exactly the contracted shapes and invariants:

- `fake_environment`: memory/refined [B, P, d_t] random; kv per TSTCT block (K, V) [B, H, P, d_h]
  with K QK-normed (unit RMS per head, as `MultiHeadAttention.project_kv` produces) and the
  temporal + causal heads rotated by position time with `TimeRotary(d_h, p_min, p_max)` from
  `TSTCTConfig` (the contract stated by TSTCT); causal_gate [B, H_c, P, P] in (0, 1) on Granger
  candidates (t_j < t_i, other entity) and zero elsewhere.
- `FakeDecoder`: decode_fields(z, role, planes) maps the first latent coordinates to flow fields
  (packets and SYN counts, softplus so counts are ≥ 0, as the Decoder's hard limits require) and
  physics_inputs(decoded) returns (values, contributing) for `physics.term.PhysicsTerm`. It has no
  parameters: it exists to test that the physics term reaches TAAFT's beliefs, not to decode.

Nothing here is used by the model itself.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any

import torch
import torch.nn.functional as F

from nagahana.models.batch import EnvironmentOut, LatentPrior, WindowBatch
from nagahana.models.config import NagaHanaConfig
from nagahana.models.taaft.model import TAAFT
from nagahana.models.vocab import N_STAGES
from nagahana.nn.attention import rotate_heads
from nagahana.nn.positional import TimeRotary


def fake_environment(cfg: NagaHanaConfig, window: WindowBatch, *, seed: int = 0) -> EnvironmentOut:
    """A contract-shaped `EnvironmentOut` for `window` (see the module docstring)."""
    t = cfg.tstct
    g = torch.Generator().manual_seed(seed)
    pos = window.positions
    b, p = pos.entity.shape
    hd = t.dim // t.heads
    rot = TimeRotary(hd, p_min=t.rotary_p_min, p_max=t.rotary_p_max)
    cos, sin = rot.angles(pos.time)                                             # [B, P, d_h]
    rotated = torch.arange(t.heads) >= t.spatial_heads                          # temporal + causal heads
    kv = []
    for _ in range(t.blocks):
        k = torch.randn(b, t.heads, p, hd, generator=g)
        k = k * torch.rsqrt(k.pow(2).mean(-1, keepdim=True) + 1e-6)             # QK-norm (unit RMS)
        k = rotate_heads(k, cos, sin, rotated)
        kv.append((k, torch.randn(b, t.heads, p, hd, generator=g)))
    real = pos.mask & (pos.entity >= 0)
    cand = (pos.time[:, None, :] < pos.time[:, :, None]) & (pos.entity[:, None, :] != pos.entity[:, :, None])
    cand = cand & real[:, :, None] & real[:, None, :]
    gate = torch.rand(b, t.causal_heads, p, p, generator=g) * cand[:, None].float()
    dz = cfg.latent_dim
    dc, gz, cz = cfg.cvgae.cont_dim, cfg.cvgae.disc_groups, cfg.cvgae.disc_classes
    assert dc + gz * cz == dz
    prior = LatentPrior(mean=torch.randn(b, p, dc, generator=g), logvar=torch.zeros(b, p, dc),
                        logits=torch.randn(b, p, gz, cz, generator=g))
    return EnvironmentOut(memory=torch.randn(b, p, t.dim, generator=g), kv=kv,
                          refined=torch.randn(b, p, t.dim, generator=g), prior=prior, causal_gate=gate, passes=1)


def build_taaft(cfg: NagaHanaConfig, **taaft_overrides: Any) -> TAAFT:
    """TAAFT for `cfg` (any preset), with `TAAFTConfig` fields overridden (e.g. blocks=1)."""
    tcfg = dataclasses.replace(cfg.taaft, **taaft_overrides)
    return TAAFT(tcfg, tstct=cfg.tstct, latent_dim=cfg.latent_dim, n_planes=len(cfg.graph.planes), n_stages=N_STAGES,
                 latent_split=(cfg.cvgae.cont_dim, cfg.cvgae.disc_groups, cfg.cvgae.disc_classes),
                 memory_input_dim=cfg.tstct.dim, memory_dim=cfg.memory.longterm_dim)


class FakeDecoder:
    """Parameter-free decoder of a few flow fields from a latent (see the module docstring)."""

    fields = ("flow.flag_count.syn", "flow.packets_fwd", "flow.packets_bwd")

    def decode_fields(self, z: torch.Tensor, role: torch.Tensor, planes: torch.Tensor) -> dict[str, torch.Tensor]:
        # Counts ≥ 0 by construction (softplus), like the Decoder's hard limits; scaled so that the
        # SYN count can exceed the packet count (a physically impossible state) for some latents.
        return {
            "flow.flag_count.syn": 10.0 * F.softplus(z[:, 0]),
            "flow.packets_fwd": F.softplus(z[:, 1]),
            "flow.packets_bwd": F.softplus(z[:, 2]),
        }

    def physics_inputs(self, decoded: Any) -> tuple[Mapping[str, torch.Tensor], Mapping[str, torch.Tensor]]:
        values: dict[str, torch.Tensor] = dict(decoded)
        contributing = {k: torch.ones_like(v0, dtype=torch.bool) for k, v0 in values.items()}
        return values, contributing
