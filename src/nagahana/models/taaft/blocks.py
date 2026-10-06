"""TAAFT's decoder-style block: belief-recursion self-attention + cross-attention into TSTCT's cache.

Purpose
-------
`nn.blocks.CrossBlock` is the shared decoder-style block (self-attention, cross-attention, SwiGLU).
TAAFT needs one thing the shared attention does not offer in a single call: at trigger τ_m each
token attends **jointly** (one softmax) to

1. the current tokens of trigger m (a key set shared by all queries: the *dense* layout), and
2. its **own** Imagination keys/values from the last M_im triggers (a different small key set per
   query: the *gathered* layout) — the belief recursion b_τ ← b_{τ−1} (build-spec §2.7).

`attend_mixed` computes that joint softmax with the block's own `MultiHeadAttention` parameters
(projections, QK-norm, learned null key/value, output projection), so nothing is re-parameterised:
it is `attend` and `attend_gathered` with the two score sets concatenated before the softmax.
`tests/test_taaft_model.py::test_attend_mixed_matches_gathered_attention` checks it equals
`attend_gathered` on the explicitly concatenated per-query key sets. A requested change in the final
report proposes moving it into `nn/attention.py`.

    a_i = softmax( [ q_i K_curᵀ/√d_h + B_cur(i, ·) ;  q_i·K_past(i, s)/√d_h + B_past(i, s) ;  q_i·k_∅/√d_h ] )
    out_i = a_i,cur V_cur + Σ_s a_i,s V_past(i, s) + a_i,∅ v_∅

`TAAFTBlock` subclasses `CrossBlock` and keeps the `LoopBlock` protocol (`kv`, `__call__`), so
`nn.loop.TwoStreamStack` runs it as memory stream + thinking stream unchanged (AS-06). Its cross
attention reads TSTCT's cached K/V directly with TAAFT's own query and output projections (AS-13).
The cross key/value projections and key norm that `CrossBlock` creates are deleted: TSTCT's keys
are already projected, QK-normed and time-rotated, and parameters that never receive a gradient
would only inflate the count (a requested change adds a constructor switch to `CrossBlock`).

`LazyGatheredKV` hands each TAAFT block its TSTCT block's K/V gathered to the per-query key sets
(block map b ↦ ⌊b·L_TSTCT / L_TAAFT⌋; 34 → 16 at L, every TSTCT block read by 2–3 TAAFT blocks),
gathering on access so only one block's gathered keys are alive at a time at inference.

Long-term memory keys (AS-220, AS-221)
--------------------------------------
When `AttnContext.extras[MEMORY_KV]` holds the long-term memory's K/V [B, H, X, d_h] (read by TAAFT's
X probe queries, `TAAFT.read_longterm`), the cross-attention becomes `attend_with_memory`: one
softmax over the Environment keys (queries rotated by τ on TSTCT's temporal/causal heads), the X
memory keys and the null key — the last two scored with the *un-rotated* query, because neither
carries a time (D-49: no absolute time may enter a score). A learned per-head bias by token type
(entity / adversary slot) is added to the memory logits. Without memory keys the block computes
exactly what it computed before (the shared `attend` / `attend_gathered`).

Decisions / assumptions: D-35 (TAAFT only reads the Environment), D-43 (loop), D-49 (time rotary),
AS-06, AS-13, AS-200, AS-201, AS-220, AS-221.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, overload

import torch
from torch import nn

from nagahana.nn.attention import MultiHeadAttention
from nagahana.nn.blocks import KV, AttnContext, CrossBlock

#: Keys of `AttnContext.extras` read by `TAAFTBlock`.
PAST_KV = "past_kv"            # list per block of (K, V) [B, H, N, M_im, d_h] (or None)
PAST_ALLOWED = "past_allowed"  # bool [B, 1, N, M_im]
PAST_BIAS = "past_bias"        # float [B, H, 1 or N, M_im]
MEMORY_KV = "memory_kv"        # (K, V) [B, H, X, d_h]: long-term memory keys, shared by all blocks (AS-220)
MEMORY_BIAS = "memory_bias"    # float [B or 1, H, N, 1]: learned per-head bias by token type (AS-220, AS-221)


def attend_with_memory(
    attn: MultiHeadAttention,
    q: torch.Tensor,
    q0: torch.Tensor,
    kv: KV,
    memory: KV,
    *,
    gathered: bool,
    allowed: torch.Tensor | None,
    bias: torch.Tensor | None,
    memory_bias: torch.Tensor | None,
    need_weights: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Cross-attention over the Environment keys *and* the long-term memory keys, in one softmax.

    q: queries rotated by the trigger time on TSTCT's temporal/causal heads [B, H, S, d];
    q0: the same queries un-rotated (memory keys and the null key carry no time, D-49);
    kv: Environment (K, V), dense [B, H, T, d] or gathered [B, H, S, T, d]; memory: (K, V) [B, H, X, d].

        a = softmax([ q·K_env/√d + B_env ; q0·K_mem/√d + b_mem ; q0·k_∅/√d ])

    Returns (out [B, S, dim], weights [B, H, S, T + X (+1)] or None).
    """
    k, v = kv
    km, vm = memory
    scale = 1.0 / math.sqrt(attn.head_dim)
    if gathered:
        s_env = torch.einsum("bhsd,bhstd->bhst", q.float(), k.float()) * scale       # [B, H, S, T]
    else:
        s_env = (q.float() @ k.float().transpose(-1, -2)) * scale
    if bias is not None:
        s_env = s_env + bias.float()
    if allowed is not None:
        s_env = s_env.masked_fill(~allowed.expand_as(s_env), float("-inf"))
    s_mem = (q0.float() @ km.float().transpose(-1, -2)) * scale                       # [B, H, S, X]
    if memory_bias is not None:
        s_mem = s_mem + memory_bias.float()
    parts = [s_env, s_mem]
    if attn.null_kv:
        parts.append(torch.einsum("bhsd,hd->bhs", q0.float(), attn.k_null.float())[..., None] * scale)
    w = torch.softmax(torch.cat(parts, dim=-1), dim=-1)
    t, x = s_env.shape[-1], s_mem.shape[-1]
    w_env = w[..., :t]
    out = torch.einsum("bhst,bhstd->bhsd", w_env, v.float()) if gathered else w_env @ v.float()
    out = out + w[..., t : t + x] @ vm.float()
    if attn.null_kv:
        out = out + w[..., -1:] * attn.v_null.float()[None, :, None, :]
    return attn.merge(out.to(q.dtype)), (w.to(q.dtype) if need_weights else None)


def attend_mixed(
    attn: MultiHeadAttention,
    q: torch.Tensor,
    kv: KV,
    past: KV | None,
    *,
    allowed: torch.Tensor | None,
    bias: torch.Tensor | None,
    past_allowed: torch.Tensor | None,
    past_bias: torch.Tensor | None,
    need_weights: bool,
    null_q: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Joint softmax over a dense key set and a per-query key set (see the module docstring).

    q [B, H, S, d]; kv: (K, V) [B, H, T, d]; past: (K, V) [B, H, S, M, d] or None; allowed / bias
    broadcast to [B, H, S, T]; past_allowed / past_bias broadcast to [B, H, S, M].
    Returns (out [B, S, dim], weights [B, H, S, T + M (+1 null)] or None).
    `null_q`: the un-rotated query for the null key's logit, so it carries no absolute time (D-49;
    the same fix as `nn.attention.MultiHeadAttention.attend`).
    """
    k, v = kv
    scale = 1.0 / math.sqrt(attn.head_dim)
    neg = float("-inf")
    # Scores in float32 (softmax precision, build-agents numerics rule).
    s_cur = (q.float() @ k.float().transpose(-1, -2)) * scale                     # [B, H, S, T]
    if bias is not None:
        s_cur = s_cur + bias.float()
    if allowed is not None:
        s_cur = s_cur.masked_fill(~allowed.expand_as(s_cur), neg)
    parts = [s_cur]
    if past is not None:
        kp, vp = past
        s_past = torch.einsum("bhsd,bhsmd->bhsm", q.float(), kp.float()) * scale  # [B, H, S, M]
        if past_bias is not None:
            s_past = s_past + past_bias.float()
        if past_allowed is not None:
            s_past = s_past.masked_fill(~past_allowed.expand_as(s_past), neg)
        parts.append(s_past)
    if attn.null_kv:
        nq = q if null_q is None else null_q
        parts.append(torch.einsum("bhsd,hd->bhs", nq.float(), attn.k_null.float())[..., None] * scale)
    w = torch.softmax(torch.cat(parts, dim=-1), dim=-1)
    t = k.shape[2]
    out = w[..., :t] @ v.float()                                                  # [B, H, S, d]
    if past is not None:
        m = past[0].shape[3]
        out = out + torch.einsum("bhsm,bhsmd->bhsd", w[..., t : t + m], past[1].float())
    if attn.null_kv:
        out = out + w[..., -1:] * attn.v_null.float()[None, :, None, :]
    return attn.merge(out.to(q.dtype)), (w.to(q.dtype) if need_weights else None)


class TAAFTBlock(CrossBlock):
    """Decoder-style block with belief-recursion self-attention (see the module docstring)."""

    def __init__(self, dim: int, n_heads: int, *, mlp_hidden: int | None = None) -> None:
        # TSTCT's cached keys/values are already projected and QK-normed (AS-13): no cross k/v
        # projections, so every parameter of the block is trained.
        super().__init__(dim, n_heads, mlp_hidden=mlp_hidden, project_cross_kv=False)

    def forward(self, x: torch.Tensor, kv: KV, ctx: AttnContext, index: int = 0) -> tuple[torch.Tensor, dict[str, Any]]:
        aux: dict[str, Any] = {}
        # ---- self-attention: current tokens (dense) + own past Imagination (per query).
        q0 = self.attn.project_q(self.norm1(x))                                   # [B, H, N, d_h]
        q = ctx.q_hook(q0) if ctx.q_hook else q0
        past_list = ctx.extras.get(PAST_KV)
        past = past_list[index] if past_list is not None else None
        a, w = attend_mixed(self.attn, q, kv, past, allowed=ctx.allowed, bias=ctx.bias,
                            past_allowed=ctx.extras.get(PAST_ALLOWED), past_bias=ctx.extras.get(PAST_BIAS),
                            need_weights=ctx.need_weights, null_q=q0 if ctx.q_hook else None)
        x = x + a
        if w is not None:
            aux["self_weights"] = w
        # ---- cross-attention: TSTCT's cached K/V (block-mapped), TAAFT's own q / o projections.
        ckv = ctx.cross_kv[index] if ctx.cross_kv is not None else None
        mem = ctx.extras.get(MEMORY_KV)
        if ckv is not None:
            cq0 = self.xattn.project_q(self.norm2(x))
            cq = ctx.cross_q_hook(cq0) if ctx.cross_q_hook else cq0
            if mem is not None:
                # Environment keys + long-term memory keys in one softmax (AS-220).
                c, cw = attend_with_memory(self.xattn, cq, cq0, ckv, mem, gathered=ctx.cross_gathered,
                                           allowed=ctx.cross_allowed, bias=ctx.cross_bias,
                                           memory_bias=ctx.extras.get(MEMORY_BIAS), need_weights=ctx.need_weights)
            else:
                fn = self.xattn.attend_gathered if ctx.cross_gathered else self.xattn.attend
                c, cw = fn(cq, ckv[0], ckv[1], allowed=ctx.cross_allowed, bias=ctx.cross_bias,
                           need_weights=ctx.need_weights, null_q=cq0 if ctx.cross_q_hook else None)
            x = x + c
            if cw is not None:
                aux["cross_weights"] = cw
        x = x + self.mlp(self.norm3(x))
        return x, aux


def block_map(taaft_blocks: int, tstct_blocks: int) -> list[int]:
    """TAAFT block b reads TSTCT block ⌊b · L_TSTCT / L_TAAFT⌋ (AS-13; 34 → 16 at L)."""
    return [(b * tstct_blocks) // taaft_blocks for b in range(taaft_blocks)]


def gather_kv(kv: KV, index: torch.Tensor) -> KV:
    """Gather per-query key sets: K/V [B, H, P, d], index long [B, N, T_k] (≥ 0) → [B, H, N, T_k, d]."""
    k, v = kv
    b, h, _, d = k.shape
    n, tk = index.shape[1], index.shape[2]
    idx = index.reshape(b, 1, n * tk, 1).expand(b, h, n * tk, d)
    return k.gather(2, idx).view(b, h, n, tk, d), v.gather(2, idx).view(b, h, n, tk, d)


class LazyGatheredKV(Sequence[KV | None]):
    """Per-TAAFT-block cross K/V gathered from the mapped TSTCT block on access (see module docstring)."""

    def __init__(self, kv: Sequence[KV], mapping: Sequence[int], index: torch.Tensor) -> None:
        self.kv, self.mapping, self.key_index = kv, list(mapping), index.clamp_min(0)

    def __len__(self) -> int:
        return len(self.mapping)

    @overload
    def __getitem__(self, i: int) -> KV | None: ...

    @overload
    def __getitem__(self, i: slice) -> Sequence[KV | None]: ...

    def __getitem__(self, i: int | slice) -> KV | None | Sequence[KV | None]:
        if isinstance(i, slice):
            return [self[j] for j in range(*i.indices(len(self)))]
        return gather_kv(self.kv[self.mapping[i]], self.key_index)


class DenseMappedKV(Sequence[KV | None]):
    """Per-TAAFT-block cross K/V in the dense layout: the mapped TSTCT block's [B, H, P, d] as is."""

    def __init__(self, kv: Sequence[KV], mapping: Sequence[int]) -> None:
        self.kv, self.mapping = kv, list(mapping)

    def __len__(self) -> int:
        return len(self.mapping)

    @overload
    def __getitem__(self, i: int) -> KV | None: ...

    @overload
    def __getitem__(self, i: slice) -> Sequence[KV | None]: ...

    def __getitem__(self, i: int | slice) -> KV | None | Sequence[KV | None]:
        if isinstance(i, slice):
            return [self[j] for j in range(*i.indices(len(self)))]
        return self.kv[self.mapping[i]]


def stack_blocks(dim: int, n_heads: int, n_blocks: int, mlp_hidden: int) -> list[nn.Module]:
    """The TAAFT block stack (weight-tied across loop passes by `TwoStreamStack`)."""
    return [TAAFTBlock(dim, n_heads, mlp_hidden=mlp_hidden) for _ in range(n_blocks)]
