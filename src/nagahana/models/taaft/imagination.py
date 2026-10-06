"""Carrying Imagination across TAAFT calls: belief recursion that spans windows and live triggers.

Purpose
-------
Within one `TAAFT.forward` call, trigger m reads each token's own memory-stream K/V of the last
M_im triggers (the belief recursion, AS-200). Live inference calls TAAFT once per trigger, and
training cuts a stream into consecutive windows; without a carry the recursion would restart at
every call. This module builds the `past_*` arguments of `TAAFT.forward` from

- a previous `AnalysisOut` (training: consecutive windows of one lane), `past_from_analysis`;
- the `ImaginationStore` (inference: the Forecaster's Imagination region, D-35), `past_from_store`.

Layout (AS-223): one slot per carried *trigger* (not per token entry), oldest first, right-aligned
per window (padding slots at the front with mask False), on the receiving call's token axis
(V entity tokens of its entity table, then G adversary slots). With this layout the receiving call
counts "the last M_im triggers" exactly as one long call would, so

    forward(triggers m₁ … m₂ in one call)  ≡  forward(m₁ …) then forward(… m₂, past = carry)

(tests/test_taaft_imagination.py checks it to 1e-5). Store retention is also counted in triggers
(AS-157), so the store and the in-call history agree on which triggers are "the last M_im".

Token alignment
---------------
The caller aligns tokens: `past_from_analysis(token_index=…)` maps each receiving token to its row
in the previous call (−1 = not present); `past_from_store(read)` expects the store to have been read
with the receiving call's token ids (entity keys, then the slot ids −1 … −G the engine writes), so
the read is already aligned. That is why `TAAFT.forward` takes no token-id argument.

Time: carried trigger times are re-expressed relative to the receiving window's origin (they are
negative for earlier windows). The receiving call re-checks time < τ (AS-223's as-of guard).

Decisions: D-35 (Imagination R/W by the Forecaster), D-36 (retention per trigger).
Assumptions: AS-13, AS-157, AS-223, AS-224.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

from nagahana.memory.imagination import ImaginationRead
from nagahana.models.batch import AnalysisOut
from nagahana.nn.blocks import KV


@dataclass
class PastImagination:
    """The `past_*` arguments of `TAAFT.forward` (shapes there). `kwargs()` unpacks them."""

    kv: list[KV]                  # per block (K, V) [B, H, N, M_c, d_h]
    time: torch.Tensor            # float64 [B, M_c], relative to the receiving window's origin
    mask: torch.Tensor            # bool [B, N, M_c]
    y: torch.Tensor | None = None  # [B, N, M_c, d_y]

    def kwargs(self) -> dict[str, Any]:
        return {"past_imagination": self.kv, "past_time": self.time, "past_mask": self.mask, "past_y": self.y}


def past_from_analysis(
    out: AnalysisOut,
    trigger_time: torch.Tensor,
    trigger_mask: torch.Tensor,
    *,
    keep: int,
    origin_shift: torch.Tensor | float = 0.0,
    token_index: torch.Tensor | None = None,
    previous: PastImagination | None = None,
    detach: bool = True,
) -> PastImagination:
    """Carry the last `keep` triggers of a previous call into the next one.

    out: the previous call's output; trigger_time float64 [B, M] / trigger_mask bool [B, M]: its
    triggers (relative to *its* origin). origin_shift: previous origin − receiving origin (seconds,
    [B] or scalar), added to the times. token_index long [B, N_new]: row of each receiving token in the
    previous call (−1 = absent); None = same token axis. previous: the carry the previous call itself
    received (so a short window does not drop older triggers still within M_im). detach: cut the
    gradient into the previous call (truncated recurrence; False keeps it).
    """
    b, m_tr, n_old = out.token_mask.shape
    kvs = []
    for k, v in out.imagination_kv:
        h, d = k.shape[1], k.shape[3]
        kvs.append((k.view(b, h, m_tr, n_old, d).permute(0, 1, 3, 2, 4),           # [B, H, N_old, M, d]
                    v.view(b, h, m_tr, n_old, d).permute(0, 1, 3, 2, 4)))
    shift = origin_shift if isinstance(origin_shift, torch.Tensor) else torch.full((b,), float(origin_shift))
    time = trigger_time.to(torch.float64) + shift.to(torch.float64)[:, None]
    mask = out.token_mask.permute(0, 2, 1) & trigger_mask[:, None, :]                  # [B, N_old, M]
    y = out.y.permute(0, 2, 1, 3)                                                       # [B, N_old, M, d_y]
    if previous is not None:
        # Older carried triggers first (they were relative to the previous origin: shift them too).
        kvs = [(torch.cat([pk, k], dim=3), torch.cat([pv, v], dim=3)) for (pk, pv), (k, v) in zip(previous.kv, kvs, strict=True)]
        time = torch.cat([previous.time + shift.to(torch.float64)[:, None], time], dim=1)
        mask = torch.cat([previous.mask, mask], dim=2)
        py = previous.y if previous.y is not None else torch.zeros(*previous.mask.shape, y.shape[-1], dtype=y.dtype)
        y = torch.cat([py, y], dim=2)
    # The last `keep` triggers (right-aligned; a shorter history is padded at the front).
    total = time.shape[1]
    if total >= keep:
        sl = slice(total - keep, total)
        kvs = [(k[:, :, :, sl], v[:, :, :, sl]) for k, v in kvs]
        time, mask, y = time[:, sl], mask[:, :, sl], y[:, :, sl]
    else:
        pad = keep - total
        kvs = [(torch.cat([k.new_zeros(*k.shape[:3], pad, k.shape[4]), k], dim=3),
                torch.cat([v.new_zeros(*v.shape[:3], pad, v.shape[4]), v], dim=3)) for k, v in kvs]
        time = torch.cat([time.new_full((b, pad), float("-inf")), time], dim=1)
        mask = torch.cat([mask.new_zeros(b, mask.shape[1], pad), mask], dim=2)
        y = torch.cat([y.new_zeros(b, y.shape[1], pad, y.shape[3]), y], dim=2)
    # Align to the receiving call's tokens.
    if token_index is not None:
        idx = token_index.clamp_min(0)
        present = token_index >= 0                                                       # [B, N_new]
        kvs = [(_take(k, idx, 2) * present[:, None, :, None, None], _take(v, idx, 2) * present[:, None, :, None, None])
               for k, v in kvs]
        mask = _take(mask, idx, 1) & present[:, :, None]
        y = _take(y, idx, 1)
    if detach:
        kvs = [(k.detach(), v.detach()) for k, v in kvs]
        y = y.detach()
    return PastImagination(kv=kvs, time=time.nan_to_num(neginf=-1e18), mask=mask, y=y)


def _take(x: torch.Tensor, index: torch.Tensor, dim: int) -> torch.Tensor:
    """x[b, …, index[b, i], …] along `dim` (batch first): index long [B, N] → same rank as x."""
    shape = [1] * x.dim()
    shape[0], shape[dim] = index.shape[0], index.shape[1]
    idx = index.view(shape).expand(*[x.shape[i] if i != dim else index.shape[1] for i in range(x.dim())])
    return x.gather(dim, idx)


def past_from_store(
    read: ImaginationRead,
    trigger_times: Sequence[float],
    *,
    origin: float,
    y: torch.Tensor | None = None,
) -> PastImagination:
    """Inference carry (B = 1) from `ImaginationStore.read(token_ids, before_time=τ, role=FORECASTER)`.

    read: entries of the receiving call's N tokens (compact per token, oldest first, with times);
    trigger_times: `store.trigger_times` (the kept triggers, ascending; epoch seconds); origin: epoch
    seconds of the receiving window's origin; y: optional [N, M_c, d_y] ŷ per kept trigger slot.
    Each entry goes to the slot of its trigger time, so the layout is per trigger (AS-223).
    """
    times = [float(t) for t in trigger_times]
    m_c = len(times)
    slot_of = {t: j for j, t in enumerate(times)}
    n = int(read.mask.shape[0])
    blocks = len(read.k)
    h, d = (read.k[0].shape[2], read.k[0].shape[3]) if blocks else (0, 0)
    ks = [torch.zeros(1, h, n, m_c, d, dtype=read.k[0].dtype) for _ in range(blocks)]
    vs = [torch.zeros(1, h, n, m_c, d, dtype=read.k[0].dtype) for _ in range(blocks)]
    mask = torch.zeros(1, n, m_c, dtype=torch.bool)
    for i in range(n):
        for e in range(read.mask.shape[1]):
            if not bool(read.mask[i, e]):
                continue
            j = slot_of.get(float(read.time[i, e]))
            if j is None:                                    # entry of a trigger no longer kept: skip
                continue
            for blk in range(blocks):
                ks[blk][0, :, i, j] = read.k[blk][i, e]
                vs[blk][0, :, i, j] = read.v[blk][i, e]
            mask[0, i, j] = True
    time = torch.tensor(times, dtype=torch.float64)[None] - float(origin)
    return PastImagination(kv=list(zip(ks, vs, strict=True)), time=time, mask=mask,
                           y=None if y is None else y[None].detach())
