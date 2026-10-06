"""KV caches: the Environment (TSTCT) and Imagination (TAAFT) working memories [A-12], [A-14].

What the design says
--------------------
- "TSTCT will run by taking this state as input, to generate the spatio-temporal-causal kv cache;
  this basically stores the state model & updates of it … which makes up the environment" [A-12].
- "the TAAFT will work on this this environment's data (TSTCT's kv cache) to generate the kv cache
  of the analysis" [A-14].

Three facts about KV caches that shape the contract
---------------------------------------------------
1. **Size grows linearly with history.** One cache holds keys and values for every cached position:

       bytes ≈ 2 · L · H · d_h · T · (bytes per element)

   with L layers, H heads, head width d_h and T cached positions (state updates). For months of
   updates [Q-04], [Q-19], T is very large. This is the core tension with "absolute best memory over
   months", and why D-15 asks what the *durable* Environment is.
2. **A cache is only meaningful to the weights that produced it.** Retraining or fine-tuning
   changes what keys and values mean, so old caches become invalid. Every cache therefore records
   the hash of the producing weights (`model_hash`), and `assert_compatible` refuses mismatches.
   (Proposal P-18: rebuild caches from the event log after retraining.)
3. **Caches are internal representations, not the shared latent space.** The Decoder renders
   latents z (CVG-AE space), not raw keys and values. Forecasts must be projected back into z to be
   decoded (proposal P-19).

Element precision (D-54)
------------------------
Both caches store keys and values in single precision: `CACHE_DTYPE` = torch.float32, 4 bytes per
element (`CACHE_ELEMENT_BYTES`), so one cached position takes 2 · L · H · d_h · 4 bytes. The owner's
follow-up to D-54 (2026-10-02, "Caches fp32 too") extends "fp32 overall" from the weights to the
Environment and Imagination caches: what is written is what the fp32 weights computed, with no
16-bit rounding on the way into memory. Training's bf16 autocast (AS-39) may produce 16-bit K/V
inside a forward pass; the stores cast them to fp32 when they are written.

Prior art for bounded or extended caches (refs.md): Transformer-XL segment caching (#L2388);
Memorizing Transformers, kNN retrieval over a large external KV memory (#L2402); Infini-attention
(#L2409); Titans neural long-term memory (#L2423). Which of these bounds our caches is open (D-15).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch

from nagahana.core.errors import InvariantViolation
from nagahana.memory.access import Region

#: Stored dtype of the Environment and Imagination K/V caches (D-54 follow-up: fp32, never 16-bit).
CACHE_DTYPE: torch.dtype = torch.float32
#: Bytes per stored cache element (fp32).
CACHE_ELEMENT_BYTES: int = 4

@dataclass(frozen=True)
class KVCacheMeta:
    """Identity and shape of one cache.

    Attributes
    ----------
    region: ENVIRONMENT (TSTCT) or IMAGINATION (TAAFT).
    producer: component that wrote it ("tstct", "taaft").
    model_hash: hash of the producing weights. A cache is valid only for these weights.
    latent_space: version of the shared latent space it was computed in (P-19).
    schema_version: state-model version of the inputs (datamodel/versioning.py).
    layers, heads, head_dim: cache geometry.
    """

    region: Region
    producer: str
    model_hash: str
    latent_space: str
    schema_version: str
    layers: int
    heads: int
    head_dim: int

    def __post_init__(self) -> None:
        if self.region is Region.MONITOR:
            raise InvariantViolation("The Monitor region is not a KV cache (it holds drift statistics).")
        if min(self.layers, self.heads, self.head_dim) <= 0:
            raise InvariantViolation("layers, heads and head_dim must be positive")

    def bytes_per_position(self, element_bytes: int = CACHE_ELEMENT_BYTES) -> int:
        """Memory for one cached position: 2·L·H·d_h·element_bytes (keys and values; fp32 = 4 B, D-54)."""
        return 2 * self.layers * self.heads * self.head_dim * element_bytes


def assert_compatible(meta: KVCacheMeta, *, model_hash: str, latent_space: str) -> None:
    """Refuse to use a cache with weights or a latent space other than those that produced it."""
    if meta.model_hash != model_hash:
        raise InvariantViolation(
            f"{meta.region.value} cache was produced by weights {meta.model_hash[:12]}…, not "
            f"{model_hash[:12]}…; rebuild it (P-18) instead of reusing it."
        )
    if meta.latent_space != latent_space:
        raise InvariantViolation(
            f"cache latent space {meta.latent_space!r} ≠ {latent_space!r} (P-19)."
        )


class KVCacheStore(Protocol):
    """Storage contract for one cache. Implementations: in-memory (tests), paged GPU, disk-backed."""

    meta: KVCacheMeta

    def append(self, layer: int, keys: torch.Tensor, values: torch.Tensor) -> None:
        """Append keys/values [H, T_new, d_h] for one layer."""
        ...

    def read(self, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
        """All cached keys and values for one layer, [H, T, d_h] each."""
        ...

    def length(self) -> int:
        """Number of cached positions T."""
        ...

    def truncate(self, keep_last: int) -> None:
        """Keep only the most recent positions (retention applies here; schedule: memory/retention.py)."""
        ...
