"""The Environment store: TSTCT's bounded, log-time bucketed K/V memory per entity (build-spec §2.6).

Purpose
-------
TSTCT's memory stream writes one (K, V) pair per block for every entity state (`EnvironmentOut.kv`,
`TSTCT.step`). This store keeps those pairs per entity so that a later state can gather its keys:
its own past (temporal heads), its neighbours' latest states (spatial heads), and recent states of
contacted entities (causal heads). TAAFT, the Advisor, the Verifier and the Decoder read it (D-35).

Owner sources: [A-12] ("the spatio-temporal-causal kv cache … makes up the environment"), [Q-19]
(months of memory), [Q-33] (retention not controllable by attack volume).
Decisions: D-35 (access matrix, `memory/access.py`), D-36 (volume must not drive forgetting), D-15
held (persistence; the store is a rebuildable view keyed to the model hash, P-18 assumed in AS-11).
Assumptions: AS-11 (bounded Environment, log-time buckets, merge with log n), AS-150 (dyadic
absolute-time cells with a per-cell quota, this module), AS-154 (latest-state register).

Why the buckets are anchored to absolute time
---------------------------------------------
The build-spec states the buckets by *age*: bucket b holds ages [2^b, 2^{b+1})·δ₀. Ages change as
time passes, so a bucket defined by age keeps changing members, and a slot written during a flood
later shares a coarse bucket with history written long before the flood. If merging is decided per
age bucket, the flood then consumes the quota of that shared bucket and forces merges among the
*older* slots: volume would drive forgetting after all, only later (D-36 violated in slow motion).

This store therefore uses **dyadic cells of absolute time** (relative to a fixed store origin), whose
*level* is chosen by age:

    unit index      u(t) = ⌊t / δ₀⌋                     (t: float64 seconds since the store origin)
    cell (ℓ, n)     = units [n·2^ℓ, (n+1)·2^ℓ)           (level ℓ = 0 … L, L = n_buckets − 1)
    "aged" at T     ⇔ ℓ = 0  or  end(ℓ, n) + 2^ℓ ≤ u_T   (the cell ended at least one own width ago)

At store clock T, each slot lives in its **maximal aged cell**; level-L cells that ended more than
2^{n_buckets}·δ₀ ago are absorbed into one **archive** cell. Every cell holds at most m slots
(`MemoryConfig.bucket_slots`); when it holds more, its two most similar slots are merged.

Properties (each tested in `tests/test_memory_environment.py`)
--------------------------------------------------------------
P1. *Partition that only coarsens.* "Aged" is downward closed in ℓ (a cell inside an aged cell ends
    no later and is half as wide) and monotone in T, so the maximal aged cells partition the past and
    can only coalesce as T grows. Two siblings coalesce into their parent exactly when the parent ages.
P2. *Aged cells are complete.* For ℓ ≥ 1 an aged cell ended before u_T − 2^ℓ < u_T; with appends in
    time order (enforced) no later state can enter it.
P3. *Locality (volume invariance).* A cell's content is fixed once, when it ages:
        content(parent) = Compact_m( content(left child) ∪ content(right child) ),
    and a level-0 cell's content is the sequential compaction of the states written in its δ₀. By
    induction, the content of a cell is a function of the states whose times lie in that cell, of the
    clock T, and of the model; it is independent of how often or when compaction was evaluated (the
    recursion is canonical). Hence a flood in [s, T] can change only cells that intersect [s, T]:
    every cell that ended by s is bit-for-bit unchanged. The one cell that may straddle s has width
    ≤ T − s (it is aged, so width ≤ u_T − end < u_T − s), so **no slot older than s − (T − s) is
    ever touched by the flood**: a flood in the last minute cannot touch anything older than two
    minutes. A flood compresses inside its own cells; it never evicts older cells.
P4. *Capacity independent of volume and history length.* At most 4 cells at level 0, 2 per level
    for 1 ≤ ℓ < L, 1 at level L, plus the archive: |cells| ≤ 2·n_buckets + 2 (`max_cells`). With the
    latest-state register (below), slots per entity ≤ m·(2·n_buckets + 2) + 1. The store refuses a
    configuration where this exceeds `slots_per_entity` (L: 7·66 + 1 = 463 ≤ 512).
P5. *Resolution proportional to age.* A slot at age a (below the archive) sits in a cell of width
    w with a/4 < w ≤ a (aged: w ≤ T − end ≤ a; parent not aged: a < 4w). Every octave of age keeps
    about 2 cells, i.e. ≈ 2m reserved slots: mid ages always keep their share (build-spec §4b.1,
    "lost in the middle", Liu et al., TACL 2024, arXiv:2307.03172).

Merging and the log-count bias (AS-11)
--------------------------------------
Two slots (K₁, V₁, t₁, n₁), (K₂, V₂, t₂, n₂) of one cell merge into
    K = (n₁K₁ + n₂K₂)/(n₁+n₂),   V likewise,   n = n₁ + n₂,   t = (n₁t₁ + n₂t₂)/(n₁+n₂),
for every block at once (the slot is one entity state across blocks). Readers add log n to the
merged key's attention logit. For n identical keys this is exact:
    Σ_{c=1..n} e^{s} = e^{s + log n},
so softmax over n copies equals softmax over one copy with bias log n (tested). For different keys
it is an approximation: by Jensen, e^{q·K̄} ≤ mean_c e^{q·K_c}, so a merged slot slightly
under-weights its members; merging the *most similar* pair (cosine of the keys concatenated over
blocks and heads) keeps that gap small. The merged time is the count-weighted mean, which stays
inside the cell, so the global time order of slots is preserved. Keys are stored already
time-rotated (TSTCT rotates a key by its own time at write); the mean of rotated keys is the rotated
key of nothing in particular, which is one more reason to merge only within short cells relative to
the age (P5).

Latest-state register (AS-154)
------------------------------
Spatial heads and TAAFT read each entity's *latest* state exactly. Protecting the newest slot from
merging would break P3 (which slot is newest depends on later volume), so each entity instead keeps
one extra copy of its newest state outside the cells (the "register"), overwritten at every append.

Time
----
All times are float64 seconds relative to `origin` (float64 epoch seconds), never float32 epoch
(D-49 precision note). The store clock T is the latest appended (or `compact`-ed) time. Appends must
be in non-decreasing time order: ordering is resolved before the model ([Q-20]); the store raises
otherwise rather than silently re-ordering.

Precision (D-54)
----------------
K, V and the auxiliary rows are stored in fp32 (`memory.kvcache.CACHE_DTYPE`, the default `dtype`):
the owner's follow-up to D-54 ("Caches fp32 too") keeps the cache at the precision of the fp32
weights that produced it. `append` casts whatever it receives (bf16 under training autocast, AS-39)
to the store's dtype, so the stored rows never depend on the caller's compute precision. Merged
slots are count-weighted means of fp32 rows; times and counts are float64 (numpy).

Invariants
----------
- A slot's K/V rows are data (detached); the store holds no autograd graph (training uses the dense
  path, `TSTCT.forward`).
- Every read or write checks `memory/access.py` (Region.ENVIRONMENT) for the caller's role.
- `meta.model_hash` / `meta.latent_space` identify the producing weights (P-18/P-19);
  `assert_compatible` refuses other weights.

Extension points
----------------
- Storage backends (paged GPU, disk) replace the flat tensors behind `keys`/`values`/`gather`.
- Another similarity for merging (e.g. value-aware) replaces `_most_similar_pair`.
- The aged rule's ratio (one own width) is AS-150; a different ratio changes `max_cells` accordingly.
"""

from __future__ import annotations

import bisect
import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import torch

from nagahana.core.errors import InvariantViolation
from nagahana.core.roles import Role
from nagahana.governance.assumptions import assume
from nagahana.memory.access import Op, Region, check
from nagahana.memory.kvcache import CACHE_DTYPE, KVCacheMeta, assert_compatible
from nagahana.models.config.components import MemoryConfig
from nagahana.nn.blocks import KV

#: Key of the archive cell (all level-L cells older than the horizon). Sorts before every real cell.
ARCHIVE: tuple[int, int] = (-1, 0)

CellKey = tuple[int, int]


def max_cells(n_buckets: int) -> int:
    """Upper bound on the number of cells per entity (property P4): 2·n_buckets + 2.

    Level 0: cells whose parent is not aged have start in (u_T − 4, u_T] → ≤ 4. Level 1 ≤ ℓ < L: aged
    (end ≤ u_T − w) and parent not aged (end > u_T − 3w) → ≤ 2. Level L: end ∈ (u_T − 2w, u_T − w]
    → ≤ 1. Archive: 1. Total 4 + 2(L − 1) + 1 + 1 = 2L + 4 = 2·n_buckets + 2.
    """
    if n_buckets < 2:
        raise InvariantViolation("n_buckets must be ≥ 2")
    return 2 * n_buckets + 2


@dataclass
class Gathered:
    """K/V of gathered slots, per block, with their metadata. Padding slots have mask False.

    k, v: per block [n, T, H, d_h]; time: float64 [n, T] (seconds since the store origin);
    count: float32 [n, T] (states merged into the slot; 1 for unmerged); mask: bool [n, T];
    slot: long [n, T] slot ids (−1 for padding).
    """

    k: list[torch.Tensor]
    v: list[torch.Tensor]
    time: torch.Tensor
    count: torch.Tensor
    mask: torch.Tensor
    slot: torch.Tensor

    def log_count_bias(self) -> torch.Tensor:
        """log n per slot [n, T] (0 for padding), the additive logit of merged slots (AS-11)."""
        return torch.where(self.mask, self.count.clamp_min(1.0).log(), torch.zeros_like(self.count))


@dataclass
class _EntityMemory:
    """Per-entity bookkeeping: cells (key → slot ids sorted by (time, seq)) and the register."""

    cells: dict[CellKey, list[int]] = field(default_factory=dict)
    register: int = -1
    sorted_cache: list[int] | None = None


class EnvironmentStore:
    """Per-entity, log-time bucketed K/V memory of TSTCT (see the module docstring).

    Parameters
    ----------
    cfg: MemoryConfig (slots_per_entity, bucket_delta0 = δ₀, n_buckets, bucket_slots = m).
    meta: identity of the producing weights; region must be ENVIRONMENT.
    origin: float64 epoch seconds of time zero. All times passed to the store are relative to it.
    aux_dim: width of an optional per-slot auxiliary vector merged like V (TSTCT stores its causal
        gate keys here, so causal gates can be computed for stored states without re-encoding them).
    device, dtype: of the K/V buffers; dtype defaults to fp32 (`CACHE_DTYPE`, D-54) and every append is
        cast to it.
    """

    def __init__(
        self,
        cfg: MemoryConfig,
        meta: KVCacheMeta,
        *,
        origin: float,
        aux_dim: int = 0,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = CACHE_DTYPE,
    ) -> None:
        assume("AS-11", by=__name__)
        if meta.region is not Region.ENVIRONMENT:
            raise InvariantViolation("EnvironmentStore needs a KVCacheMeta with region ENVIRONMENT")
        if cfg.bucket_slots < 1:
            raise InvariantViolation("bucket_slots (m) must be ≥ 1")
        self.cfg, self.meta = cfg, meta
        self.origin = float(origin)
        self.blocks, self.heads, self.head_dim = meta.layers, meta.heads, meta.head_dim
        self.aux_dim = aux_dim
        self.quota = cfg.bucket_slots
        self.top_level = cfg.n_buckets - 1                     # L
        self.horizon_units = 2 ** cfg.n_buckets                # archive horizon in units of δ₀
        # P4: the bound must fit the per-entity capacity, else the guarantee is void → refuse.
        self.capacity_required = self.quota * max_cells(cfg.n_buckets) + 1
        if self.capacity_required > cfg.slots_per_entity:
            raise InvariantViolation(
                f"bucket_slots={self.quota} with n_buckets={cfg.n_buckets} needs up to {self.capacity_required} slots "
                f"per entity (m·(2·n_buckets+2)+1), more than slots_per_entity={cfg.slots_per_entity}. "
                "Lower bucket_slots or n_buckets, or raise slots_per_entity (AS-150)."
            )
        self.device, self.dtype = torch.device(device), dtype
        # Flat slot storage, grown by doubling. Rows of freed slots are garbage until reused.
        cap = 256
        self._k = [torch.zeros(cap, self.heads, self.head_dim, device=self.device, dtype=dtype) for _ in range(self.blocks)]
        self._v = [torch.zeros(cap, self.heads, self.head_dim, device=self.device, dtype=dtype) for _ in range(self.blocks)]
        self._aux = torch.zeros(cap, aux_dim, device=self.device, dtype=dtype)
        self._entity = np.full(cap, -1, dtype=np.int64)
        self._time = np.zeros(cap, dtype=np.float64)
        self._count = np.zeros(cap, dtype=np.float64)
        self._seq = np.zeros(cap, dtype=np.int64)
        self._alive = np.zeros(cap, dtype=bool)
        self._free: list[int] = []
        self._next = 0                                         # first never-used row
        self._next_seq = 0                                     # global write order (ties in time)
        self._entities: dict[int, _EntityMemory] = {}
        self.clock = -math.inf                                 # store clock T (seconds since origin)
        self.merges = 0                                        # number of merges performed (diagnostics)

    # ================================================================== identity and access
    def assert_compatible(self, *, model_hash: str, latent_space: str) -> None:
        """Refuse to use this store with other weights or another latent space (P-18, P-19)."""
        assert_compatible(self.meta, model_hash=model_hash, latent_space=latent_space)

    @staticmethod
    def _check(role: Role, op: Op) -> None:
        check(role, Region.ENVIRONMENT, op)

    # ================================================================== slot storage
    @property
    def buffer_rows(self) -> int:
        """Number of rows of the flat buffers (slot ids are in [0, buffer_rows))."""
        return int(self._k[0].shape[0]) if self.blocks else 0

    @property
    def next_seq(self) -> int:
        """Sequence number the next appended state will get (global write order)."""
        return self._next_seq

    def _grow(self, need: int) -> None:
        # Double the flat buffers until `need` rows exist.
        cap = self.buffer_rows
        if need <= cap:
            return
        new = cap
        while new < need:
            new *= 2
        extra = new - cap

        def pad(x: torch.Tensor) -> torch.Tensor:
            return torch.cat([x, x.new_zeros(extra, *x.shape[1:])], dim=0)

        self._k = [pad(x) for x in self._k]
        self._v = [pad(x) for x in self._v]
        self._aux = pad(self._aux)
        self._entity = np.concatenate([self._entity, np.full(extra, -1, dtype=np.int64)])
        self._time = np.concatenate([self._time, np.zeros(extra)])
        self._count = np.concatenate([self._count, np.zeros(extra)])
        self._seq = np.concatenate([self._seq, np.zeros(extra, dtype=np.int64)])
        self._alive = np.concatenate([self._alive, np.zeros(extra, dtype=bool)])

    def _alloc(self) -> int:
        # Reuse a freed row if any, else take the next fresh row.
        if self._free:
            return self._free.pop()
        self._grow(self._next + 1)
        self._next += 1
        return self._next - 1

    def _release(self, slot: int) -> None:
        self._alive[slot] = False
        self._entity[slot] = -1
        self._free.append(slot)

    # ================================================================== writing
    def append(
        self,
        entity: torch.Tensor,
        time: torch.Tensor,
        kv_per_block: Sequence[KV],
        *,
        role: Role,
        aux: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Write n new entity states. Returns their slot ids (long [n]; slots may merge later).

        entity: long [n]; time: float64 [n], non-decreasing and ≥ the store clock;
        kv_per_block: per block (K, V), each [n, H, d_h]; aux: [n, aux_dim] or None (zeros).
        """
        self._check(role, Op.WRITE)
        n = int(entity.shape[0])
        if len(kv_per_block) != self.blocks:
            raise InvariantViolation(f"expected K/V for {self.blocks} blocks, got {len(kv_per_block)}")
        times = time.to(torch.float64).tolist()
        if n and (times[0] < self.clock or any(b < a for a, b in zip(times, times[1:], strict=False))):
            raise InvariantViolation("Environment appends must be in non-decreasing time order (ordering is "
                                     "resolved before the model, [Q-20])")
        ents = entity.tolist()
        ids = [self._alloc() for _ in range(n)]
        idx = torch.tensor(ids, dtype=torch.long, device=self.device)
        # Rows for all blocks at once: [n, H, d_h] each (detached: the store holds data, not graphs).
        for b, (k, v) in enumerate(kv_per_block):
            if k.shape != (n, self.heads, self.head_dim) or v.shape != k.shape:
                raise InvariantViolation(f"block {b}: K/V must be [n, H, d_h] = {(n, self.heads, self.head_dim)}")
            self._k[b][idx] = k.detach().to(self.device, self.dtype)
            self._v[b][idx] = v.detach().to(self.device, self.dtype)
        if self.aux_dim:
            if aux is None or aux.shape != (n, self.aux_dim):
                raise InvariantViolation(f"aux must be [n, {self.aux_dim}]")
            self._aux[idx] = aux.detach().to(self.device, self.dtype)
        # Per state, in time order: metadata, level-0 cell insert + compaction, register, coarsening.
        for slot, e, t in zip(ids, ents, times, strict=True):
            e = int(e)
            self._entity[slot], self._time[slot], self._count[slot] = e, t, 1.0
            self._seq[slot], self._alive[slot] = self._next_seq, True
            self._next_seq += 1
            mem = self._entities.setdefault(e, _EntityMemory())
            cell = (0, math.floor(t / self.cfg.bucket_delta0))
            members = mem.cells.setdefault(cell, [])
            members.append(slot)                                   # newest: already last in (time, seq)
            self._compact(members)
            self._set_register(mem, e, slot)
            self.clock = t
            self._coarsen(mem)
            mem.sorted_cache = None
        return idx

    def _set_register(self, mem: _EntityMemory, entity: int, slot: int) -> None:
        # AS-154: exact copy of the newest state, outside the cells (never merged).
        if mem.register < 0:
            mem.register = self._alloc()
        r = mem.register
        for b in range(self.blocks):
            self._k[b][r] = self._k[b][slot]
            self._v[b][r] = self._v[b][slot]
        if self.aux_dim:
            self._aux[r] = self._aux[slot]
        self._entity[r], self._time[r], self._count[r] = entity, self._time[slot], 1.0
        self._seq[r], self._alive[r] = self._seq[slot], True

    def compact(self, now: float) -> None:
        """Advance the clock to `now` (≥ clock) and coarsen every entity's cells (P1). Path-independent (P3)."""
        if now < self.clock:
            raise InvariantViolation("the store clock cannot go backwards")
        self.clock = float(now)
        for mem in self._entities.values():
            self._coarsen(mem)
            mem.sorted_cache = None

    # ================================================================== cells
    def _aged(self, level: int, index: int) -> bool:
        # A cell is aged at clock T iff ℓ = 0 or it ended at least one own width ago (AS-150).
        if level == 0:
            return True
        width = 1 << level
        end = (index + 1) * width
        return end + width <= self.clock / self.cfg.bucket_delta0

    def _coarsen(self, mem: _EntityMemory) -> None:
        # Bottom-up sweep: a cell whose parent has aged moves into the parent; the parent's union of the
        # (already compacted) children is compacted once. This is the canonical recursion of P3.
        for level in range(self.top_level):
            children = [key for key in mem.cells if key[0] == level]
            groups: dict[CellKey, list[CellKey]] = {}
            for key in children:
                parent = (level + 1, key[1] >> 1)
                if self._aged(*parent):
                    groups.setdefault(parent, []).append(key)
            for parent, keys in sorted(groups.items()):
                merged = sorted((s for key in keys for s in mem.cells.pop(key)), key=self._order)
                merged.extend(mem.cells.pop(parent, []))           # (cannot exist: aged parents are new)
                merged.sort(key=self._order)
                mem.cells[parent] = merged
                self._compact(merged)
        # Archive: level-L cells that ended more than 2^{n_buckets}·δ₀ before the clock.
        limit = self.clock / self.cfg.bucket_delta0 - self.horizon_units
        old = sorted(key for key in mem.cells if key[0] == self.top_level and (key[1] + 1) * (1 << key[0]) <= limit)
        for key in old:                                             # oldest first: canonical order
            archive = mem.cells.setdefault(ARCHIVE, [])
            archive.extend(mem.cells.pop(key))
            archive.sort(key=self._order)
            self._compact(archive)

    def _order(self, slot: int) -> tuple[float, int]:
        return (float(self._time[slot]), int(self._seq[slot]))

    def _compact(self, members: list[int]) -> None:
        # Merge the most similar pair until the cell holds at most m slots (in place, sorted).
        while len(members) > self.quota:
            a, c = self._most_similar_pair(members)
            self._merge(members[a], members[c])
            del members[c]
            members.sort(key=self._order)

    def _most_similar_pair(self, members: list[int]) -> tuple[int, int]:
        # Cosine similarity of keys concatenated over blocks and heads; ties → the earliest pair.
        idx = torch.tensor(members, dtype=torch.long, device=self.device)
        feats = torch.cat([self._k[b][idx].reshape(len(members), -1) for b in range(self.blocks)], dim=1).float()
        feats = feats / feats.norm(dim=1, keepdim=True).clamp_min(1e-12)
        sim = feats @ feats.t()                                     # [m+1, m+1]
        upper = torch.triu(torch.ones_like(sim, dtype=torch.bool), diagonal=1)
        sim = sim.masked_fill(~upper, -math.inf)
        flat = int(torch.argmax(sim).item())
        a, c = divmod(flat, len(members))
        return a, c

    def _merge(self, keep: int, drop: int) -> None:
        # Count-weighted means of K, V, aux and time; counts add; seq = the later one.
        na, nb = float(self._count[keep]), float(self._count[drop])
        wa, wb = na / (na + nb), nb / (na + nb)
        for b in range(self.blocks):
            self._k[b][keep] = wa * self._k[b][keep] + wb * self._k[b][drop]
            self._v[b][keep] = wa * self._v[b][keep] + wb * self._v[b][drop]
        if self.aux_dim:
            self._aux[keep] = wa * self._aux[keep] + wb * self._aux[drop]
        self._time[keep] = wa * self._time[keep] + wb * self._time[drop]
        self._count[keep] = na + nb
        self._seq[keep] = max(self._seq[keep], self._seq[drop])
        self._release(drop)
        self.merges += 1

    # ================================================================== reading: slot ids
    def _sorted_slots(self, entity: int) -> list[int]:
        # Cells are disjoint time intervals: archive first, then by cell start; inside, (time, seq).
        mem = self._entities.get(entity)
        if mem is None:
            return []
        if mem.sorted_cache is None:
            keys = sorted(mem.cells, key=lambda k: -math.inf if k == ARCHIVE else k[1] * (1 << k[0]))
            mem.sorted_cache = [s for key in keys for s in mem.cells[key]]
        return mem.sorted_cache

    def entities(self) -> list[int]:
        """Entity ids with at least one stored state."""
        return sorted(self._entities)

    def num_slots(self, entity: int) -> int:
        """Slots held for `entity`, including its register (the quantity bounded by P4)."""
        mem = self._entities.get(entity)
        if mem is None:
            return 0
        return sum(len(s) for s in mem.cells.values()) + (1 if mem.register >= 0 else 0)

    def cells(self, entity: int) -> dict[CellKey, list[int]]:
        """Copy of the entity's cells (diagnostics and tests)."""
        mem = self._entities.get(entity)
        return {} if mem is None else {k: list(v) for k, v in mem.cells.items()}

    def temporal_slots(self, entity: int, t: float, k: int, *, role: Role) -> list[int]:
        """The entity's last `k` slots with time ≤ t, oldest first (temporal heads)."""
        self._check(role, Op.READ)
        slots = self._sorted_slots(entity)
        times = [float(self._time[s]) for s in slots]
        end = bisect.bisect_right(times, t)
        return slots[max(0, end - k):end]

    def latest_slot(self, entity: int, t: float, *, role: Role) -> int:
        """Slot of the entity's latest state as of t (register if its time ≤ t), −1 if none (spatial heads)."""
        self._check(role, Op.READ)
        mem = self._entities.get(entity)
        if mem is None:
            return -1
        if mem.register >= 0 and float(self._time[mem.register]) <= t:
            return mem.register
        slots = self.temporal_slots(entity, t, 1, role=role)
        return slots[-1] if slots else -1

    def lagged_slots(self, entity: int, t: float, lag: float, *, role: Role, tie: float) -> list[int]:
        """Slots of the entity with tie < t − time ≤ lag, oldest first (causal-head candidates; tie: AS-160)."""
        self._check(role, Op.READ)
        out = []
        for s in self._sorted_slots(entity):
            dt = t - float(self._time[s])
            if tie < dt <= lag:
                out.append(s)
        return out

    def export_slots(self, *, role: Role) -> tuple[list[int], list[bool]]:
        """Every cell slot and every latest-state register of every entity, sorted by (time, write order).

        Returns (slot ids, is_register flags). This is the carried Environment of D-51 (training across
        windows): bounded by P4 per entity, with the same volume invariance as the store (P3).
        """
        self._check(role, Op.READ)
        rows: list[tuple[float, int, int, int, bool]] = []
        for mem in self._entities.values():
            for members in mem.cells.values():
                rows += [(float(self._time[s]), int(self._seq[s]), 0, s, False) for s in members]
            if mem.register >= 0:
                r = mem.register
                rows.append((float(self._time[r]), int(self._seq[r]), 1, r, True))
        rows.sort()
        return [r[3] for r in rows], [r[4] for r in rows]

    def slot_entity(self, slots: Sequence[int]) -> list[int]:
        return [int(self._entity[s]) for s in slots]

    # ================================================================== reading: tensors
    def keys(self, block: int) -> torch.Tensor:
        """Flat key buffer of one block [buffer_rows, H, d_h] (index with slot ids)."""
        return self._k[block]

    def values(self, block: int) -> torch.Tensor:
        """Flat value buffer of one block [buffer_rows, H, d_h]."""
        return self._v[block]

    def slot_aux(self, slots: torch.Tensor) -> torch.Tensor:
        """Auxiliary vectors of slots [..., aux_dim] (−1 ids give zeros)."""
        safe = slots.clamp_min(0)
        out = self._aux[safe]
        return torch.where((slots >= 0).unsqueeze(-1), out, torch.zeros_like(out))

    def slot_time(self, slots: Sequence[int]) -> list[float]:
        return [float(self._time[s]) for s in slots]

    def slot_count(self, slots: Sequence[int]) -> list[float]:
        return [float(self._count[s]) for s in slots]

    def slot_seq(self, slots: Sequence[int]) -> list[int]:
        return [int(self._seq[s]) for s in slots]

    def gather(self, slots: torch.Tensor, *, role: Role) -> Gathered:
        """K/V and metadata of slot ids [n, T] (−1 = padding) → `Gathered`."""
        self._check(role, Op.READ)
        mask = slots >= 0
        safe = slots.clamp_min(0)
        flat = safe.reshape(-1).cpu().numpy()
        time = torch.from_numpy(self._time[flat].reshape(tuple(slots.shape))).to(slots.device)
        count = torch.from_numpy(self._count[flat].reshape(tuple(slots.shape))).to(slots.device, torch.float32)
        zero = torch.zeros((), dtype=self.dtype, device=self.device)
        m4 = mask.to(self.device)[..., None, None]
        k = [torch.where(m4, self._k[b][safe.to(self.device)], zero) for b in range(self.blocks)]
        v = [torch.where(m4, self._v[b][safe.to(self.device)], zero) for b in range(self.blocks)]
        return Gathered(k=k, v=v, time=torch.where(mask, time, torch.zeros_like(time)),
                        count=torch.where(mask, count, torch.zeros_like(count)), mask=mask, slot=torch.where(mask, slots, -1))

    def gather_temporal(self, entity: torch.Tensor, t: torch.Tensor, k: int, *, role: Role) -> Gathered:
        """For each query (entity [n], time [n] float64): its last k slots with time ≤ t, padded at the end."""
        rows = [self.temporal_slots(int(e), float(tt), k, role=role) for e, tt in zip(entity.tolist(), t.tolist(), strict=True)]
        return self.gather(_pad(rows), role=role)

    def snapshot(self, entity: int) -> list[tuple[CellKey, float, float, torch.Tensor]]:
        """(cell, time, count, keys over blocks) per slot of the entity, oldest first (tests, audits)."""
        mem = self._entities.get(entity)
        if mem is None:
            return []
        cell_of = {s: key for key, members in mem.cells.items() for s in members}
        return [(cell_of[s], float(self._time[s]), float(self._count[s]),
                 torch.stack([self._k[b][s] for b in range(self.blocks)]).clone()) for s in self._sorted_slots(entity)]


def _pad(rows: Sequence[Sequence[int]]) -> torch.Tensor:
    """Ragged slot-id lists → long [n, max(1, T)] padded with −1."""
    width = max([1, *(len(r) for r in rows)])
    out = torch.full((len(rows), width), -1, dtype=torch.long)
    for i, r in enumerate(rows):
        if r:
            out[i, : len(r)] = torch.tensor(list(r), dtype=torch.long)
    return out
