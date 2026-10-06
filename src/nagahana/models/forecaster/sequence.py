"""The route transformer: pre-norm self-attention blocks over [context ; route steps] (build-spec §2.8, §2.10).

Purpose
-------
One stack, two layouts, the same function:

- **dense** (training, teacher forcing; the Verifier's PRM): the whole sequence
  [context (T positions) ; step 0 … step K] at once, with a mask that is causal over steps;
- **prefix + step** (imagination): run the prefix once, keep each block's keys/values, then append
  one imagined step at a time, each new step reading the cached keys (a KV cache). Because a key is
  computed from its own position's block input and rotated by its own time when written
  (`nn.positional`, "cache-friendly"), the two layouts compute the same outputs (tested to 1e-5).

Decisions: D-49 (positions are time: an imagined step k sits at t = k · window_seconds and is
time-rotated; context positions sit at the trigger time t = 0; no index or pass embeddings).
Assumptions: AS-32 (block design; shared `nn.blocks.SelfBlock`).

Maths
-----
For block b, queries Q, keys K (QK-normed, then rotated by time on every head), values V:

    a_ij = ⟨R(t_i) q̂_i, R(t_j) k̂_j⟩ / √d_h + M_ij,    M_ij ∈ {0, −∞}

    allowed(i, j) = context i: j is a valid context position (and passes intervention masks)
                    step i:    j a valid context position, or a step j ≤ i  (causal over steps)

The rotary product depends only on t_i − t_j (Su et al. 2021, arXiv:2104.09864), so a step reads the
context "k windows ago" and earlier steps by their real time gaps.

Invariants: a step never reads a later step (tested: changing step k+1's input leaves steps ≤ k
unchanged); context positions never read steps (so the context is route-independent and cached once).

Extension points: `need_weights=True` returns attention weights per block (driving-feature
explanations over context positions).
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from nagahana.nn.attention import rotate_heads
from nagahana.nn.blocks import KV, AttnContext, SelfBlock
from nagahana.nn.loop import TwoStreamStack
from nagahana.nn.positional import TimeRotary


class RouteStack(nn.Module):
    """`blocks` SelfBlocks of width `dim` with continuous-time rotary on all heads."""

    def __init__(self, dim: int, heads: int, blocks: int, mlp_hidden: int, *, p_min: float, p_max: float) -> None:
        super().__init__()
        if dim % heads:
            raise ValueError("dim must be divisible by heads")
        self.heads = heads
        self.head_dim = dim // heads
        # TwoStreamStack.memory is exactly the one-pass "each block attends to its own K/V" stack.
        self.stack = TwoStreamStack([SelfBlock(dim, heads, mlp_hidden=mlp_hidden) for _ in range(blocks)])
        self.rotary = TimeRotary(self.head_dim, p_min=p_min, p_max=p_max)

    # ------------------------------------------------------------------ helpers
    def _ctx(self, times: torch.Tensor, allowed: torch.Tensor, need_weights: bool = False) -> AttnContext:
        # cos/sin of the positions being processed: [R, L, d_h]; queries and new keys rotate by them.
        cos, sin = self.rotary.angles(times)
        return AttnContext(
            q_hook=lambda q: rotate_heads(q, cos, sin),
            k_hook=lambda k: rotate_heads(k, cos, sin),
            allowed=allowed,
            need_weights=need_weights,
        )

    # ------------------------------------------------------------------ dense layout
    def dense(self, x: torch.Tensor, times: torch.Tensor, allowed: torch.Tensor, *,
              need_weights: bool = False) -> tuple[torch.Tensor, list[KV], list[dict[str, Any]]]:
        """x [R, L, dim], times float64 [R, L], allowed bool [R, 1, L, L] → (out [R, L, dim], per-block K/V, aux)."""
        return self.stack.memory(x, self._ctx(times, allowed, need_weights))

    # ------------------------------------------------------------------ incremental layout
    def step(self, x: torch.Tensor, times: torch.Tensor, prefix: list[KV], prefix_allowed: torch.Tensor
             ) -> tuple[torch.Tensor, list[KV]]:
        """Append S new positions to a cached prefix.

        x [R, S, dim]; times float64 [R, S]; prefix: per block (K, V) each [R, H, L, d_h] (already
        rotated); prefix_allowed bool [R, L] (which cached keys the new positions may read).
        Returns (out [R, S, dim], per-block new (K, V) [R, H, S, d_h]) — append these to the cache.
        """
        r, s, _ = x.shape
        causal = torch.ones(s, s, dtype=torch.bool, device=x.device).tril()           # new ↔ new, causal
        allowed = torch.cat([prefix_allowed[:, None, None, :].expand(r, 1, s, -1),
                             causal[None, None].expand(r, 1, s, s)], dim=-1)          # [R, 1, S, L+S]
        ctx = self._ctx(times, allowed)
        new_kv: list[KV] = []
        for i, block in enumerate(self.stack.blocks):
            assert isinstance(block, SelfBlock)
            k_new, v_new = block.kv(x, ctx)                                             # [R, H, S, d_h]
            new_kv.append((k_new, v_new))
            kv_full = (torch.cat([prefix[i][0], k_new], dim=2), torch.cat([prefix[i][1], v_new], dim=2))
            x, _ = block(x, kv_full, ctx, i)
        return x, new_kv
