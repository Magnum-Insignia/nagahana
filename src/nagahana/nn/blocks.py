"""Pre-norm transformer blocks shared by TSTCT, TAAFT, the Forecaster, the Advisor and the Verifier.

    SelfBlock:     x ← x + Attn(N₁x | K/V);            x ← x + SwiGLU(N₂x)
    CrossBlock:    x ← x + Attn(N₁x | K/V_self);  x ← x + XAttn(N₂x | K/V_mem);  x ← x + SwiGLU(N₃x)

Pre-norm (normalise the input of each sub-layer, keep the residual stream un-normalised) trains
stably at depth without warm-up tricks (Xiong et al., "On Layer Normalization in the Transformer
Architecture", ICML 2020, arXiv:2002.04745).

Keys and values are an *argument*, not computed inside `forward`
-------------------------------------------------------------------
This is what makes the two-stream loop (`nn.loop`, AS-06) possible: in the memory stream a block
attends to K/V projected from its own input (`kv(x)`); in the thinking stream the same block attends
to the K/V the memory stream produced. A block therefore exposes

    kv(x, ctx)      → (k, v)  from N₁x (keys rotated by ctx.k_hook if given)
    forward(x, kv, ctx, index) → (x', aux)

`AttnContext` carries everything that varies per call but not per pass: masks, biases, rotary hooks,
the key layout, cross-attention memories, and whether to return attention weights (explanations).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn

from nagahana.nn.attention import MultiHeadAttention
from nagahana.nn.mlp import SwiGLU
from nagahana.nn.norms import RMSNorm

KV = tuple[torch.Tensor, torch.Tensor]
Hook = Callable[[torch.Tensor], torch.Tensor]


@dataclass
class AttnContext:
    """Per-call attention settings, identical across loop passes.

    Attributes
    ----------
    q_hook, k_hook: applied to projected queries / keys (e.g. time-rotary on some heads).
    allowed, bias: self-attention pattern and bias (broadcast to [B, H, S, T] or the gathered shape).
    gathered: K/V are per-query key sets [B, H, S, T_k, d].
    cross_kv: per-block cross-attention memories (index = block index), already projected (e.g.
        TSTCT's cached K/V read by TAAFT). None entries mean "project `cross_source` with this block's
        own cross projections".
    cross_source: raw memory vectors [B, T, d_mem] for blocks that project their own cross K/V.
    cross_q_hook, cross_allowed, cross_bias, cross_gathered: as above, for the cross-attention.
    need_weights: return attention weights in `aux` (explanations; slower path).
    """

    q_hook: Hook | None = None
    k_hook: Hook | None = None
    allowed: torch.Tensor | None = None
    bias: torch.Tensor | None = None
    gathered: bool = False
    cross_kv: Sequence[KV | None] | None = None
    cross_source: torch.Tensor | None = None
    cross_q_hook: Hook | None = None
    cross_allowed: torch.Tensor | None = None
    cross_bias: torch.Tensor | None = None
    cross_gathered: bool = False
    need_weights: bool = False
    extras: dict[str, Any] = field(default_factory=dict)


def _attend(attn: MultiHeadAttention, q: torch.Tensor, kv: KV, *, allowed: torch.Tensor | None,
            bias: torch.Tensor | None, gathered: bool, need_weights: bool,
            null_q: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
    # `null_q`: the un-rotated query, so the null key's logit carries no absolute time (D-49).
    fn = attn.attend_gathered if gathered else attn.attend
    return fn(q, kv[0], kv[1], allowed=allowed, bias=bias, need_weights=need_weights, null_q=null_q)


class SelfBlock(nn.Module):
    """Pre-norm self-attention block (encoder-style). See the module docstring."""

    def __init__(self, dim: int, n_heads: int, *, mlp_hidden: int | None = None, qk_norm: bool = True) -> None:
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = MultiHeadAttention(dim, n_heads, qk_norm=qk_norm)
        self.norm2 = RMSNorm(dim)
        self.mlp = SwiGLU(dim, mlp_hidden)

    def kv(self, x: torch.Tensor, ctx: AttnContext) -> KV:
        """This block's keys/values for input `x` (the memory stream's cache entry)."""
        k, v = self.attn.project_kv(self.norm1(x))
        return (ctx.k_hook(k) if ctx.k_hook else k), v

    def forward(self, x: torch.Tensor, kv: KV, ctx: AttnContext, index: int = 0) -> tuple[torch.Tensor, dict[str, Any]]:
        q0 = self.attn.project_q(self.norm1(x))
        q = ctx.q_hook(q0) if ctx.q_hook else q0
        a, w = _attend(self.attn, q, kv, allowed=ctx.allowed, bias=ctx.bias, gathered=ctx.gathered,
                       need_weights=ctx.need_weights, null_q=q0 if ctx.q_hook else None)
        x = x + a
        x = x + self.mlp(self.norm2(x))
        return x, ({"self_weights": w} if w is not None else {})


class CrossBlock(nn.Module):
    """Pre-norm decoder-style block: self-attention, cross-attention to a memory, SwiGLU.

    `mem_dim`: width of raw memory vectors when this block projects its own cross K/V; when the
    cross K/V are supplied already projected (TAAFT reading TSTCT's cache) only the cross query and
    output projections of this block are used.
    """

    def __init__(self, dim: int, n_heads: int, *, mem_dim: int | None = None, mlp_hidden: int | None = None,
                 qk_norm: bool = True, project_cross_kv: bool = True) -> None:
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = MultiHeadAttention(dim, n_heads, qk_norm=qk_norm)
        self.norm2 = RMSNorm(dim)
        self.xattn = MultiHeadAttention(dim, n_heads, kv_dim=mem_dim or dim, qk_norm=qk_norm)
        if not project_cross_kv:
            # Cross K/V arrive already projected (TAAFT reading TSTCT's cache, AS-13): no own k/v weights.
            del self.xattn.k_proj, self.xattn.v_proj, self.xattn.k_norm
        self.norm3 = RMSNorm(dim)
        self.mlp = SwiGLU(dim, mlp_hidden)

    def kv(self, x: torch.Tensor, ctx: AttnContext) -> KV:
        """Self-attention keys/values for input `x` (memory-stream cache entry)."""
        k, v = self.attn.project_kv(self.norm1(x))
        return (ctx.k_hook(k) if ctx.k_hook else k), v

    def cross_kv_from(self, memory: torch.Tensor) -> KV:
        """Project raw memory vectors [B, T, d_mem] with this block's cross projections."""
        return self.xattn.project_kv(memory)

    def forward(self, x: torch.Tensor, kv: KV, ctx: AttnContext, index: int = 0) -> tuple[torch.Tensor, dict[str, Any]]:
        aux: dict[str, Any] = {}
        q0 = self.attn.project_q(self.norm1(x))
        q = ctx.q_hook(q0) if ctx.q_hook else q0
        a, w = _attend(self.attn, q, kv, allowed=ctx.allowed, bias=ctx.bias, gathered=ctx.gathered,
                       need_weights=ctx.need_weights, null_q=q0 if ctx.q_hook else None)
        x = x + a
        if w is not None:
            aux["self_weights"] = w
        # Cross-attention: supplied K/V for this block index, else project the raw memory.
        ckv = ctx.cross_kv[index] if ctx.cross_kv is not None else None
        if ckv is None and ctx.cross_source is not None:
            ckv = self.cross_kv_from(ctx.cross_source)
        if ckv is not None:
            cq0 = self.xattn.project_q(self.norm2(x))
            cq = ctx.cross_q_hook(cq0) if ctx.cross_q_hook else cq0
            c, cw = _attend(self.xattn, cq, ckv, allowed=ctx.cross_allowed, bias=ctx.cross_bias,
                            gathered=ctx.cross_gathered, need_weights=ctx.need_weights,
                            null_q=cq0 if ctx.cross_q_hook else None)
            x = x + c
            if cw is not None:
                aux["cross_weights"] = cw
        x = x + self.mlp(self.norm3(x))
        return x, aux
