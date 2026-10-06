"""The two-stream weight-tied loop of TSTCT and TAAFT (D-43, D-44; assumption AS-06).

The problem it solves
---------------------
D-43 makes TSTCT and TAAFT loop on themselves: one block stack applied for R passes with shared
weights (Universal Transformers, Dehghani et al. ICLR 2019, arXiv:1807.03819; recurrent depth,
Geiping et al. 2025, arXiv:2502.05171). Both also *cache* keys and values: TSTCT's cache is the
Environment, TAAFT's is Imagination. ADR-0008 left open which pass's keys/values the cache holds, and
noted that if R changes between runs, cached entries would depend on R.

A naive loop has a worse problem. In training, all positions loop in parallel, so at pass r a query
reads keys from pass r of every other position. At inference a new state reads the *cache*, which
holds whatever pass was stored. Training and inference then compute different functions, and only
an expensive sequential training (Feedback Transformer, Fan et al. 2020, arXiv:2002.09402) would
make them agree.

The two-stream construction
---------------------------
    memory stream (1 pass):    m = Stack(e₀),  caching (K_b, V_b) of every block b    → the cache
    thinking stream (R passes): h⁰ = 0;  h^{r+1} = Stack_read(h^r + e₀)                → the output

In `Stack_read`, block b computes queries from the thinking stream and reads the memory stream's
(K_b, V_b), the same keys for every pass. Consequences, each tested in `tests/test_nn.py`:

1. **Exact train/inference agreement.** Memory K/V of a position depend only on earlier memory K/V
   (an ordinary masked transformer), so they are computed in parallel in training and appended one
   by one at inference with identical results. The thinking stream reads fixed keys either way.
2. **The cache does not depend on R.** Raising R at run time changes only the thinking stream.
3. **R = 1 reproduces the memory stream** (h⁰ + e₀ = e₀ and every block reads the K/V it would have
   computed itself), so looping strictly extends the one-pass model.
4. **No pass embedding** (D-49). The input is re-injected every pass instead (Geiping et al. §3),
   which is what lets R grow beyond the values seen in training.

Training (AS-07): R is sampled per batch; passes before the last `grad_passes` run without gradient
(truncated backpropagation through depth, as in Geiping et al. §3.3), bounding training memory
independently of R.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, cast

import torch
from torch import nn

from nagahana.nn.blocks import KV, AttnContext


class LoopBlock(Protocol):
    """What a block must provide to be looped (SelfBlock and CrossBlock do)."""

    def kv(self, x: torch.Tensor, ctx: AttnContext) -> KV: ...

    def __call__(self, x: torch.Tensor, kv: KV, ctx: AttnContext, index: int = 0) -> tuple[torch.Tensor, dict[str, Any]]: ...


class TwoStreamStack(nn.Module):
    """A block stack run as a memory stream and a looped thinking stream. See the module docstring."""

    def __init__(self, blocks: Sequence[nn.Module]) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(blocks)

    def memory(self, x: torch.Tensor, ctx: AttnContext) -> tuple[torch.Tensor, list[KV], list[dict[str, Any]]]:
        """One pass where each block attends to its own K/V. Returns (output, per-block K/V, aux)."""
        kvs: list[KV] = []
        auxes: list[dict[str, Any]] = []
        for i, module in enumerate(self.blocks):
            block = cast(LoopBlock, module)
            kv = block.kv(x, ctx)
            kvs.append(kv)
            x, aux = block(x, kv, ctx, i)
            auxes.append(aux)
        return x, kvs, auxes

    def read_pass(self, x: torch.Tensor, kvs: Sequence[KV], ctx: AttnContext) -> tuple[torch.Tensor, list[dict[str, Any]]]:
        """One pass of the thinking stream: block b reads `kvs[b]`."""
        auxes: list[dict[str, Any]] = []
        for i, module in enumerate(self.blocks):
            x, aux = cast(LoopBlock, module)(x, kvs[i], ctx, i)
            auxes.append(aux)
        return x, auxes

    def think(
        self,
        e0: torch.Tensor,
        kvs: Sequence[KV],
        ctx: AttnContext,
        *,
        passes: int,
        grad_passes: int | None = None,
    ) -> tuple[torch.Tensor, list[dict[str, Any]]]:
        """R thinking passes with input re-injection. Returns (h^R, aux of the last pass).

        `grad_passes`: number of final passes that keep gradients (None = all). Earlier passes run
        under `torch.no_grad()` and their output is detached.
        """
        if passes < 1:
            raise ValueError("the thinking stream needs at least one pass (R ≥ 1)")
        keep = passes if grad_passes is None else max(1, min(grad_passes, passes))
        h = torch.zeros_like(e0)
        aux: list[dict[str, Any]] = []
        for r in range(passes):
            if r < passes - keep:
                with torch.no_grad():
                    h, aux = self.read_pass(h + e0, kvs, ctx)
                h = h.detach()
            else:
                h, aux = self.read_pass(h + e0, kvs, ctx)
        return h, aux
