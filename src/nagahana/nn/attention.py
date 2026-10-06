"""Multi-head attention with typed masks, bias, null keys and two key layouts.

Maths
-----
For head h, queries q_i, keys k_j, values v_j (each of width d_h):

    a_ij = ⟨q̂_i, k̂_j⟩ / √d_h  +  B_h(i, j)  +  M_h(i, j)
    out_i = Σ_j softmax_j(a_ij) v_j

- **QK-norm** (q̂, k̂): per-head RMSNorm of queries and keys before any rotation (AS-32). It bounds
  logit growth, which otherwise destabilises large transformers (Dehghani et al., "Scaling Vision
  Transformers to 22 Billion Parameters", ICML 2023, arXiv:2302.05442, §2.1; also Wortsman et al.
  2023, arXiv:2309.14322).
- **Typed masks** M_h ∈ {0, −∞}: given as a boolean `allowed` tensor broadcastable to [B, H, S, T].
  Different head groups can carry different patterns (TSTCT's spatial / temporal / causal heads).
- **Bias** B_h: any float tensor broadcastable to [B, H, S, T] (topology, recency, log n for merged
  memory slots, log g for causal gates).
- **Null key / value** (k_∅, v_∅ per head, learned): appended to every key set and always allowed
  with bias 0. A query whose pattern allows nothing then attends only to the null value instead of
  producing NaN, and every query can choose to "read nothing". This is the "attention sink"
  behaviour that transformers otherwise invent on their own (Xiao et al., "Efficient Streaming
  Language Models with Attention Sinks", ICLR 2024, arXiv:2309.17453).

Two layouts
-----------
- **dense**: keys [B, H, T, d_h] shared by all queries; used in training with full masks.
- **gathered**: keys [B, H, S, T_k, d_h], a separate small key set per query; used at inference,
  where each new state reads its own 608 keys from the Environment cache. Both layouts compute the
  same function; `tests/test_nn.py` checks it.

Rotary encodings are applied by the caller between `project_q`/`project_kv` and `attend` (through the
hooks of `nn.blocks.AttnContext`), because only some head groups rotate.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from nagahana.nn.norms import RMSNorm


class MultiHeadAttention(nn.Module):
    """Attention module. See the module docstring.

    Parameters
    ----------
    dim: model width of the queries (and of the output).
    n_heads: H.
    kv_dim: width of the key/value source (defaults to `dim`).
    head_dim: d_h (defaults to dim // n_heads).
    null_kv: append the learned null key/value (default True).
    qk_norm: per-head RMSNorm on queries and keys (default True).
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        *,
        kv_dim: int | None = None,
        head_dim: int | None = None,
        null_kv: bool = True,
        qk_norm: bool = True,
    ) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim or dim // n_heads
        inner = self.n_heads * self.head_dim
        self.q_proj = nn.Linear(dim, inner, bias=False)
        self.k_proj = nn.Linear(kv_dim or dim, inner, bias=False)
        self.v_proj = nn.Linear(kv_dim or dim, inner, bias=False)
        self.o_proj = nn.Linear(inner, dim, bias=False)
        self.q_norm = RMSNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(self.head_dim) if qk_norm else nn.Identity()
        self.null_kv = null_kv
        if null_kv:
            self.k_null = nn.Parameter(torch.randn(n_heads, self.head_dim) * 0.02)
            self.v_null = nn.Parameter(torch.zeros(n_heads, self.head_dim))

    # ------------------------------------------------------------------ projections
    def _split(self, x: torch.Tensor) -> torch.Tensor:
        # [B, T, H·d_h] → [B, H, T, d_h]
        b, t, _ = x.shape
        return x.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)

    def project_q(self, x: torch.Tensor) -> torch.Tensor:
        """Queries [B, S, dim] → [B, H, S, d_h] (QK-normed, not rotated)."""
        return self.q_norm(self._split(self.q_proj(x)))

    def project_kv(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Key/value source [B, T, kv_dim] → (keys, values), each [B, H, T, d_h] (keys QK-normed)."""
        return self.k_norm(self._split(self.k_proj(x))), self._split(self.v_proj(x))

    def merge(self, out: torch.Tensor) -> torch.Tensor:
        """Per-head outputs [B, H, S, d_h] → [B, S, dim] through the output projection."""
        b, h, s, d = out.shape
        return self.o_proj(out.transpose(1, 2).reshape(b, s, h * d))

    # ------------------------------------------------------------------ attention
    def attend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        allowed: torch.Tensor | None = None,
        bias: torch.Tensor | None = None,
        need_weights: bool = False,
        null_q: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Dense attention. q [B,H,S,d], k/v [B,H,T,d] → (out [B,S,dim], weights [B,H,S,T(+1)] or None).

        `allowed` and `bias` broadcast to [B, H, S, T]. With the null key, the returned weights have
        one extra last column: the weight on "nothing".

        `null_q` [B,H,S,d]: the query used for the null key's logit. Pass the *un-rotated* query when
        `q` carries a time rotation: ⟨R(t)q, k_∅⟩ would depend on absolute time since an arbitrary
        origin (a hidden clock, against D-49/D-50), while ⟨q, k_∅⟩ does not. Implemented as an additive
        correction (⟨null_q, k_∅⟩ − ⟨q, k_∅⟩)/√d on the null column, so the fused kernel is kept.
        Found by the TSTCT engineer's origin-shift test (2026-10-02).
        """
        b, h, s, _ = q.shape
        t = k.shape[2]
        mask = self._additive(allowed, bias, (b, h, s, t), q.dtype, q.device)
        if self.null_kv:
            k_null = self.k_null.to(k.dtype)
            k = torch.cat([k, k_null[None, :, None, :].expand(b, h, 1, -1)], dim=2)
            v = torch.cat([v, self.v_null.to(v.dtype)[None, :, None, :].expand(b, h, 1, -1)], dim=2)
            if null_q is not None:
                # [B, H, S, 1]: move the null logit from ⟨q, k_∅⟩ to ⟨null_q, k_∅⟩.
                null_col = torch.einsum("bhsd,hd->bhs", null_q - q, k_null).unsqueeze(-1) / math.sqrt(self.head_dim)
                base = mask if mask is not None else q.new_zeros(b, h, s, t)
                mask = torch.cat([base, null_col.to(base.dtype)], dim=-1)
            elif mask is not None:
                mask = torch.cat([mask, mask.new_zeros(*mask.shape[:-1], 1)], dim=-1)
        if need_weights:
            scores = (q @ k.transpose(-1, -2)) / math.sqrt(self.head_dim)
            if mask is not None:
                scores = scores + mask
            w = torch.softmax(scores.float(), dim=-1).to(q.dtype)
            return self.merge(w @ v), w
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.merge(out), None

    def attend_gathered(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        allowed: torch.Tensor | None = None,
        bias: torch.Tensor | None = None,
        need_weights: bool = False,
        null_q: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Per-query key sets. q [B,H,S,d], k/v [B,H,S,Tk,d], allowed/bias → [B,H,S,Tk].

        `null_q`: as in `attend` (the un-rotated query for the null key's logit)."""
        b, h, s, tk, _ = k.shape
        scores = torch.einsum("bhsd,bhstd->bhst", q, k) / math.sqrt(self.head_dim)
        mask = self._additive(allowed, bias, (b, h, s, tk), q.dtype, q.device)
        if mask is not None:
            scores = scores + mask
        if self.null_kv:
            nq = q if null_q is None else null_q
            null_score = torch.einsum("bhsd,hd->bhs", nq, self.k_null.to(q.dtype)) / math.sqrt(self.head_dim)
            scores = torch.cat([scores, null_score.unsqueeze(-1)], dim=-1)
        w = torch.softmax(scores.float(), dim=-1).to(q.dtype)
        out = torch.einsum("bhst,bhstd->bhsd", w[..., :tk], v)
        if self.null_kv:
            out = out + w[..., tk:] * self.v_null.to(v.dtype)[None, :, None, :]
        return self.merge(out), (w if need_weights else None)

    @staticmethod
    def _additive(
        allowed: torch.Tensor | None,
        bias: torch.Tensor | None,
        shape: tuple[int, int, int, int],
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor | None:
        # Combine the boolean pattern and the float bias into one additive mask of the full shape.
        if allowed is None and bias is None:
            return None
        m = torch.zeros(shape, dtype=dtype, device=device)
        if bias is not None:
            m = m + bias.to(dtype)
        if allowed is not None:
            m = m.masked_fill(~allowed.expand(shape), float("-inf"))
        return m


def rotate_heads(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    head_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Rotary-rotate the heads selected by `head_mask` [H] (all heads if None).

    x: [B, H, T, d] (or gathered [B, H, S, T, d]); cos/sin broadcastable to x without the head axis
    inserted, i.e. [B, T, d] → used as [B, 1, T, d] (gathered: [B, S, T, d] → [B, 1, S, T, d]).
    """
    from nagahana.nn.positional import apply_rotary  # local import: positional has no attention deps

    c, s_ = cos.unsqueeze(1), sin.unsqueeze(1)
    rot = apply_rotary(x, c, s_)
    if head_mask is None:
        return rot
    shape = [1, -1] + [1] * (x.dim() - 2)
    return torch.where(head_mask.view(shape), rot, x)
