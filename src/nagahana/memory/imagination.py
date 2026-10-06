"""The Imagination store: TAAFT's memory-stream K/V of past triggers (build-spec §2.6, §2.7).

Purpose
-------
TAAFT's memory stream at a trigger τ produces, for every token (an active entity or an adversary
hypothesis slot), one (K, V) pair per block: "the kv cache of the analysis" [A-14]. At the next
trigger each token's self-attention reads its own entries of earlier triggers (time-rotated by
trigger time), which is how belief recursion b_τ ← b_{τ−1} is carried (build-spec §2.7). This
module holds those entries for the last M_im triggers.

Owner sources: [A-14], [Q-31] (beliefs isolated from facts), [Q-33] (retention per trigger).
Decisions: D-35 (Imagination: Forecaster R W; Advisor, Verifier, Decoder R; Simulator none),
D-36 (retention regular per trigger). Assumptions: AS-13 (Imagination = TAAFT memory-stream K/V of
the last 8 triggers), AS-157 (retention counted in global triggers, below).

Retention (AS-157)
------------------
Entries are kept for the last M_im = `imagination_triggers` *triggers of the store*, not the last
M_im writes of each token: when trigger τ is written, every entry from a trigger older than the
M_im-th most recent one is dropped, for all tokens at once. Forgetting is then a function of the
trigger count only (D-36): no token, however active, changes how long another token's beliefs
last, and an inactive token's old beliefs expire on the same schedule as everyone's.

Access (D-35)
-------------
Every `write` checks Region.IMAGINATION WRITE for the caller's role (only the Forecaster holds it);
every `read` checks READ (Forecaster, Advisor, Verifier, Decoder). The Simulator, which writes the
Environment, can neither read nor write beliefs here: a belief can never become an observation.

Precision (D-54)
----------------
Entries are stored in fp32 (`memory.kvcache.CACHE_DTYPE`, the default `dtype`): the owner's
follow-up to D-54 ("Caches fp32 too"). `write` casts the K/V it receives to the store's dtype and
`read` returns that dtype, padding included, so the stored beliefs never depend on the writer's
compute precision. Trigger times are float64.

Invariants
----------
- Trigger times strictly increase across writes (one write per trigger).
- `read(..., before_time)` returns only entries with trigger time < before_time (no leakage from
  the trigger being computed or later).
- Entries are detached data; `meta` identifies the producing weights (`assert_compatible`, P-18).

Extension points
----------------
- The trigger's forecasts (build-spec §2.6 "plus the trigger's forecasts") can be kept beside the
  K/V through `write(..., forecast=...)` payloads; this store keeps an opaque payload per trigger.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

from nagahana.core.errors import InvariantViolation
from nagahana.core.roles import Role
from nagahana.governance.assumptions import assume
from nagahana.memory.access import Op, Region, check
from nagahana.memory.kvcache import CACHE_DTYPE, KVCacheMeta, assert_compatible
from nagahana.models.config.components import MemoryConfig
from nagahana.nn.blocks import KV


@dataclass
class ImaginationRead:
    """Entries of n tokens, oldest trigger first, padded at the end.

    k, v: per block [n, M, H, d_h]; time: float64 [n, M] trigger times (0 at padding); mask: bool [n, M].
    """

    k: list[torch.Tensor]
    v: list[torch.Tensor]
    time: torch.Tensor
    mask: torch.Tensor


class ImaginationStore:
    """K/V of TAAFT's memory stream per token id for the last M_im triggers (see the module docstring).

    Parameters
    ----------
    cfg: MemoryConfig (imagination_triggers = M_im).
    meta: identity of the producing weights; region must be IMAGINATION.
    dtype: stored K/V dtype, fp32 by default (`CACHE_DTYPE`, D-54); every write is cast to it.
    """

    def __init__(self, cfg: MemoryConfig, meta: KVCacheMeta, *, dtype: torch.dtype = CACHE_DTYPE) -> None:
        assume("AS-13", by=__name__)
        if meta.region is not Region.IMAGINATION:
            raise InvariantViolation("ImaginationStore needs a KVCacheMeta with region IMAGINATION")
        if cfg.imagination_triggers < 1:
            raise InvariantViolation("imagination_triggers must be ≥ 1")
        self.cfg, self.meta = cfg, meta
        self.keep = cfg.imagination_triggers
        self.blocks, self.heads, self.head_dim = meta.layers, meta.heads, meta.head_dim
        self.dtype = dtype
        self._triggers: list[float] = []                                        # kept trigger times, ascending
        # token id → list of (trigger time, K [blocks, H, d_h], V [blocks, H, d_h]), ascending time
        self._entries: dict[int, list[tuple[float, torch.Tensor, torch.Tensor]]] = {}
        self._payload: dict[float, Any] = {}

    def assert_compatible(self, *, model_hash: str, latent_space: str) -> None:
        """Refuse other weights or another latent space (P-18, P-19)."""
        assert_compatible(self.meta, model_hash=model_hash, latent_space=latent_space)

    @property
    def trigger_times(self) -> tuple[float, ...]:
        """Times of the triggers currently kept (ascending)."""
        return tuple(self._triggers)

    def write(self, trigger_time: float, token_ids: torch.Tensor, kv_per_block: Sequence[KV], *, role: Role,
              forecast: Any = None) -> None:
        """Store the memory-stream K/V of one trigger: token_ids long [n]; per block (K, V) [n, H, d_h]."""
        check(role, Region.IMAGINATION, Op.WRITE)
        if self._triggers and trigger_time <= self._triggers[-1]:
            raise InvariantViolation("trigger times must strictly increase (one write per trigger)")
        if len(kv_per_block) != self.blocks:
            raise InvariantViolation(f"expected K/V for {self.blocks} blocks, got {len(kv_per_block)}")
        n = int(token_ids.shape[0])
        for b, (k, v) in enumerate(kv_per_block):
            if k.shape != (n, self.heads, self.head_dim) or v.shape != k.shape:
                raise InvariantViolation(f"block {b}: K/V must be [n, H, d_h] = {(n, self.heads, self.head_dim)}")
        # [n, blocks, H, d_h], cast to the stored dtype (fp32, D-54) whatever the writer computed in
        ks = torch.stack([k.detach().to(self.dtype) for k, _ in kv_per_block], dim=1)
        vs = torch.stack([v.detach().to(self.dtype) for _, v in kv_per_block], dim=1)
        ids = token_ids.tolist()
        if len(set(ids)) != len(ids):
            raise InvariantViolation("token ids of one trigger must be distinct")
        for i, tok in enumerate(ids):
            self._entries.setdefault(int(tok), []).append((float(trigger_time), ks[i].clone(), vs[i].clone()))
        self._triggers.append(float(trigger_time))
        if forecast is not None:
            self._payload[float(trigger_time)] = forecast
        # Retention by trigger count only (AS-157, D-36): drop triggers older than the last M_im.
        if len(self._triggers) > self.keep:
            cutoff = self._triggers[-self.keep]
            self._triggers = self._triggers[-self.keep:]
            for tok in list(self._entries):
                kept = [e for e in self._entries[tok] if e[0] >= cutoff]
                if kept:
                    self._entries[tok] = kept
                else:
                    del self._entries[tok]
            self._payload = {t: p for t, p in self._payload.items() if t >= cutoff}

    def read(self, token_ids: torch.Tensor, before_time: float, *, role: Role) -> ImaginationRead:
        """Entries of tokens [n] from triggers strictly before `before_time`, oldest first."""
        check(role, Region.IMAGINATION, Op.READ)
        n, m = int(token_ids.shape[0]), self.keep
        k = [torch.zeros(n, m, self.heads, self.head_dim, dtype=self.dtype) for _ in range(self.blocks)]
        v = [torch.zeros(n, m, self.heads, self.head_dim, dtype=self.dtype) for _ in range(self.blocks)]
        time = torch.zeros(n, m, dtype=torch.float64)
        mask = torch.zeros(n, m, dtype=torch.bool)
        for i, tok in enumerate(token_ids.tolist()):
            rows = [e for e in self._entries.get(int(tok), []) if e[0] < before_time][-m:]
            for j, (t, ke, ve) in enumerate(rows):
                for b in range(self.blocks):
                    k[b][i, j], v[b][i, j] = ke[b], ve[b]
                time[i, j], mask[i, j] = t, True
        if self.blocks and self._entries:
            ref = next(iter(self._entries.values()))[0][1]
            k = [x.to(ref.device) for x in k]
            v = [x.to(ref.device) for x in v]
        return ImaginationRead(k=k, v=v, time=time, mask=mask)

    def forecast(self, trigger_time: float, *, role: Role) -> Any:
        """The payload stored with a kept trigger (KeyError if none)."""
        check(role, Region.IMAGINATION, Op.READ)
        return self._payload[float(trigger_time)]
