"""The stream bridge: what training needs to carry the Environment across windows (D-51, AS-161).

Purpose
-------
`data.stream.StreamLoader` yields consecutive windows per lane with stable entity keys
(`StreamContext.entity_keys`, `carried_keys`, `reset`, `origin_shift`). TSTCT's carry contract
(AS-161, `models/tstct/model.CarriedEnvironment`) needs four things; the loader provides two of them
(stable keys, time order). This module closes the other two gaps without changing the data pipeline:

    gap 2  carried entities must be listed in the window's entity table to be readable
           → `prepare` appends carried entities near the window's entities (AS-404);
    gap 3  contact matrices must be cumulative over the stream
           → a per-lane `ContactLedger` keeps first contacts by stable key; `prepare` merges it with
             the window's own contacts and recomputes C² (AS-405).

It also owns the per-lane state that lives across windows: one `EnvironmentStore` per lane (the
bounded carry, P3/P4 of `memory/environment.py`), the long-term memory (AS-220, AS-222, AS-401) and
the Imagination carried between TAAFT calls (AS-223).

Lanes and batch rows
--------------------
Batch row b continues the stream of lane `contexts[b].lane`; lanes that run out of segments drop
out, so row b and lane b can differ. Every per-lane object is keyed by the lane, never by the row.

Maths
-----
Cumulative contacts, relative to the window origin o (epoch seconds), for entities u, v of the
extended table T and the intermediate set E = T ∪ N₁ (N₁: ledger neighbours of the window entities):

    C¹[u, v]   = min( C¹_window[u, v],  first_ledger(u, v) − o ),      C¹[u, u] = 0
    C¹_p[u, v] = min( C¹_p,window[u, v], first_ledger,p(u, v) − o )
    C²[u, v]   = min_{x ∈ E} max( C¹[u, x], C¹[x, v] )                  (tropical min–max product)

For a window entity u every 2-hop path u–x–v has x ∈ N₁ ∪ window ⊆ E, so its rows of C² are exact;
rows of carried entities (never queries) may miss paths through entities outside E.

Extension of the entity table (AS-404): carried keys within two ledger hops of a window entity, hop-1
first, then by recency of their latest state, at most `TrainingConfig.max_entities` per row; they are
appended after the window's entities (existing indices unchanged), with no positions (inactive at
every trigger), kind and internal flag from the lane's registry, labels +inf / NaN (unknown, never 0).

Long-term memory (AS-220, AS-222): per lane a detached base state and the previous window's trigger
writes (detached inputs). `longterm_state` recomputes base → writes with gradient on every call, so
W_K, W_V and M₀ train through one window of recurrence and repeated forward passes never share a
graph. `longterm_states` adds this window's own writes trigger by trigger: entry m holds exactly the
writes of triggers strictly before τ_m (the exact training form of AS-222).

Imagination across windows (AS-223): after a TAAFT call, `record_analysis` keeps per lane the call's
memory-stream K/V, ŷ, trigger times and the carry that call itself received (detached); `past(prep)`
turns them into the next call's `past_*` arguments with `models.taaft.imagination.past_from_analysis`,
re-aligned to the new entity table by stable key (adversary slots map to themselves) and re-based to
the new origin. A segment start resets it with everything else.

Precision (D-54): both carries are stored in fp32 (`CACHE_DTYPE`). The Environment stores cast on
append (`TSTCT.new_store`); the Imagination carry is cast in `record_analysis`, because the bf16
autocast of training (AS-39) produces 16-bit K/V inside the forward pass.

Invariants (tests/test_integration_carry.py)
--------------------------------------------
- Indices of the window's own entities are unchanged by the extension.
- Cumulative contacts equal a brute-force evaluation over the lane's windows.
- A reset clears the store, the ledger and the long-term memory of that lane only.

Extension points
----------------
- A streaming graph builder that emits cumulative contacts directly replaces the ledger.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import torch

from nagahana.core.errors import InvariantViolation
from nagahana.data.stream import StreamContext
from nagahana.governance.assumptions import assume
from nagahana.memory.environment import EnvironmentStore
from nagahana.memory.kvcache import CACHE_DTYPE
from nagahana.memory.longterm import NeuralMemoryState
from nagahana.models.batch import AnalysisOut, EnvironmentOut, LabelBatch, TriggerBatch, WindowBatch
from nagahana.models.nagahana import NagaHana, latent_space_hash
from nagahana.models.taaft.imagination import PastImagination, past_from_analysis
from nagahana.models.tstct.model import CarriedEnvironment, CarryPolicy

INF = math.inf
#: Schema tag of Environment stores built by training (P-18 keys a cache to its producer).
TRAIN_SCHEMA = "nagahana-train-stream-v1"


# ===================================================================================== contacts
class ContactLedger:
    """First contacts of one stream by stable entity key, in epoch seconds (any plane and per plane).

    `first[(a, b)]` with a < b holds float64 [1 + n_planes]: index 0 = any plane, 1 + p = plane p.
    """

    def __init__(self, n_planes: int) -> None:
        self.n_planes = n_planes
        self.first: dict[tuple[int, int], np.ndarray] = {}
        self.nbrs: dict[int, set[int]] = {}

    def update(self, keys: np.ndarray, contact1: torch.Tensor, contact_planes: torch.Tensor, origin: float) -> None:
        """Merge one window's contacts: keys [V] (stable), contact1 [V, V], contact_planes [V, V, n_planes] relative."""
        v = len(keys)
        if v < 2:
            return
        c1 = contact1[:v, :v].to(torch.float64)
        cp = contact_planes[:v, :v].to(torch.float64)
        iu = torch.triu_indices(v, v, offset=1)
        finite = torch.isfinite(c1[iu[0], iu[1]])
        rows, cols = iu[0][finite].tolist(), iu[1][finite].tolist()
        for i, j in zip(rows, cols, strict=True):
            a, b = int(keys[i]), int(keys[j])
            if a == b:
                continue
            key = (a, b) if a < b else (b, a)
            vec = np.concatenate([[float(c1[i, j])], cp[i, j].numpy()]) + origin       # epoch seconds (inf stays inf)
            old = self.first.get(key)
            self.first[key] = vec if old is None else np.minimum(old, vec)
            self.nbrs.setdefault(a, set()).add(b)
            self.nbrs.setdefault(b, set()).add(a)

    def neighbours(self, keys: Sequence[int]) -> set[int]:
        """Union of the ledger neighbours of `keys`."""
        out: set[int] = set()
        for k in keys:
            out |= self.nbrs.get(int(k), set())
        return out

    def get(self, a: int, b: int) -> np.ndarray | None:
        """First contact vector of the pair (epoch), or None if they never met."""
        return self.first.get((a, b) if a < b else (b, a))


def min_max_product(c1_rows: torch.Tensor, c1_cols: torch.Tensor, *, chunk: int = 64) -> torch.Tensor:
    """C²[u, v] = min_x max(A[u, x], B[x, v]) for A [U, X], B [X, W] (chunked over u; float64)."""
    out = []
    for s in range(0, c1_rows.shape[0], chunk):
        a = c1_rows[s:s + chunk]                                                     # [u, X]
        out.append(torch.maximum(a[:, :, None], c1_cols[None, :, :]).amin(dim=1))     # [u, W]
    return torch.cat(out, dim=0) if out else c1_rows.new_zeros(0, c1_cols.shape[1])


# ===================================================================================== table extension
@dataclass
class RowExtension:
    """One row's extended entity table and cumulative contacts (relative to the row's window origin)."""

    n_window: int                 # entities of the window itself (indices 0 … n_window − 1, unchanged)
    keys: list[int]               # stable keys of the extended table: window keys, then appended carried keys
    extra: list[int]              # the appended carried keys
    c1: torch.Tensor              # float64 [T, T]
    c2: torch.Tensor              # float64 [T, T]
    cp: torch.Tensor              # float64 [T, T, n_planes]

    @property
    def n_total(self) -> int:
        return len(self.keys)


def extend_row(window: WindowBatch, b: int, window_keys: Sequence[int], carried: set[int], ledger: ContactLedger,
               recency: Mapping[int, float], *, cap: int) -> RowExtension:
    """Extra carried keys and cumulative C¹, C², C¹_p for row b (module docstring; AS-404, AS-405).

    window_keys: stable key of each window entity (in the window's index order); carried: keys with
    states in the stream's Environment; ledger: first contacts of earlier windows; recency: key → epoch
    time of its latest state (ranks the appended entities).
    """
    v_w = int(window.entity_mask[b].sum())
    wkeys = [int(k) for k in window_keys]
    if len(wkeys) != v_w:
        raise InvariantViolation(f"row {b}: {len(wkeys)} entity keys for {v_w} window entities")
    origin = float(window.origin[b])
    n_planes = window.contact_planes.shape[-1]
    wset = set(wkeys)
    n1 = ledger.neighbours(wkeys)
    n2 = ledger.neighbours(list(n1))
    hop1 = [k for k in n1 if k in carried and k not in wset]
    hop2 = [k for k in n2 if k in carried and k not in wset and k not in n1]
    # AS-404: hop-1 first, then hop-2, each by recency of the entity's latest state; capped.
    hop1.sort(key=lambda k: (-recency.get(k, -INF), k))
    hop2.sort(key=lambda k: (-recency.get(k, -INF), k))
    extra = (hop1 + hop2)[:cap]
    table = wkeys + extra
    inter = table + sorted(n1 - set(table))                                          # E = T ∪ N₁
    e_n, t_n = len(inter), len(table)
    pos = {k: i for i, k in enumerate(inter)}
    c1 = torch.full((e_n, e_n), INF, dtype=torch.float64)
    cp = torch.full((e_n, e_n, n_planes), INF, dtype=torch.float64)
    c1[:v_w, :v_w] = window.contact1[b, :v_w, :v_w].to(torch.float64)
    cp[:v_w, :v_w] = window.contact_planes[b, :v_w, :v_w].to(torch.float64)
    # ledger pairs among E (epoch → relative to this window's origin)
    for a in inter:
        i = pos[a]
        for nb in ledger.nbrs.get(a, ()):
            j = pos.get(nb)
            if j is None or j <= i:
                continue
            vec = ledger.get(a, nb)
            assert vec is not None
            rel = torch.from_numpy(vec - origin)
            c1[i, j] = c1[j, i] = torch.minimum(c1[i, j], rel[0])
            cp[i, j] = cp[j, i] = torch.minimum(cp[i, j], rel[1:])
    idx = torch.arange(e_n)
    c1[idx, idx] = 0.0
    cp[idx, idx] = 0.0
    c2 = min_max_product(c1[:t_n], c1[:, :t_n])                                       # [T, T]
    return RowExtension(n_window=v_w, keys=table, extra=extra, c1=c1[:t_n, :t_n].clone(), c2=c2,
                        cp=cp[:t_n, :t_n].clone())


def assemble_extension(window: WindowBatch, labels: LabelBatch | None, rows: Sequence[RowExtension], *,
                       kinds: Sequence[Mapping[int, int]], internals: Sequence[Mapping[int, bool]]
                       ) -> tuple[WindowBatch, LabelBatch | None, torch.Tensor]:
    """Pad every per-entity tensor of the window (and labels) to V' and write the cumulative contacts.

    kinds / internals: per row, key → entity kind code / internal flag of the appended entities.
    Returns (extended window, extended labels or None, stable keys long [B, V'] with −1 padding).
    """
    b_n = window.entity_mask.shape[0]
    v_new = max([1, *(r.n_total for r in rows)])
    n_pl = window.contact_planes.shape[-1]

    def pad(x: torch.Tensor, dim: int, value: float | bool | int) -> torch.Tensor:
        # pad the entity axis `dim` from V to V' (or crop padding beyond V')
        if x.shape[dim] >= v_new:
            return x.narrow(dim, 0, v_new).clone()
        shape = list(x.shape)
        shape[dim] = v_new - x.shape[dim]
        return torch.cat([x, torch.full(shape, value, dtype=x.dtype)], dim=dim)

    kind = pad(window.entity_kind, 1, 0)
    internal = pad(window.entity_internal, 1, False)
    emask = pad(window.entity_mask, 1, False)
    c1 = torch.full((b_n, v_new, v_new), INF, dtype=torch.float64)
    c2 = torch.full((b_n, v_new, v_new), INF, dtype=torch.float64)
    cp = torch.full((b_n, v_new, v_new, n_pl), INF, dtype=torch.float64)
    keys = torch.full((b_n, v_new), -1, dtype=torch.long)
    for b, r in enumerate(rows):
        t_n, v_w = r.n_total, r.n_window
        for j, k in enumerate(r.extra):
            kind[b, v_w + j] = kinds[b].get(k, 0)
            internal[b, v_w + j] = internals[b].get(k, False)
            emask[b, v_w + j] = True
        c1[b, :t_n, :t_n] = r.c1
        c2[b, :t_n, :t_n] = r.c2
        cp[b, :t_n, :t_n] = r.cp
        keys[b, :t_n] = torch.tensor(r.keys, dtype=torch.long)
    trig = window.triggers
    triggers = TriggerBatch(time=trig.time, mask=trig.mask, entity_latest=pad(trig.entity_latest, 2, -1))
    ext_w = dataclasses.replace(window, entity_kind=kind, entity_internal=internal, entity_mask=emask, contact1=c1,
                                contact2=c2, contact_planes=cp, triggers=triggers)
    ext_l = None
    if labels is not None:
        ext_l = dataclasses.replace(labels, entity_infiltrated_at=pad(labels.entity_infiltrated_at, 1, INF),
                                    entity_malicious_share=pad(labels.entity_malicious_share, 1, float("nan")))
    return ext_w, ext_l, keys


# ===================================================================================== per-lane state
@dataclass
class LaneState:
    """What one lane keeps across the windows of its current segment."""

    segment_id: str
    store: EnvironmentStore
    ledger: ContactLedger
    kind: dict[int, int] = field(default_factory=dict)          # key → entity kind code
    internal: dict[int, bool] = field(default_factory=dict)     # key → internal flag
    last_seen: dict[int, float] = field(default_factory=dict)   # key → epoch time of its latest state
    lt_base: NeuralMemoryState | None = None                    # detached; None = M₀ (with gradient)
    lt_pending: list[tuple[torch.Tensor, torch.Tensor]] = field(default_factory=list)   # (x [1, V, d], mask [1, V])
    past: _LaneImagination | None = None                        # the lane's last TAAFT call (AS-223)


@dataclass
class _LaneImagination:
    """One row of the lane's last TAAFT call, detached (what `past_from_analysis` needs)."""

    out: AnalysisOut                          # token_mask [1, M, N], imagination_kv per block [1, H, M·N, d_h], y
    trigger_time: torch.Tensor                # float64 [1, M] relative to `origin`
    trigger_mask: torch.Tensor                # bool [1, M]
    received: PastImagination | None          # the carry that call received (aligned to its tokens)
    keys: list[int]                           # stable key of each entity token of that call (−1 padding)
    origin: float                             # epoch seconds of that call's window origin


@dataclass
class PreparedBatch:
    """One batch after the bridge: the extended window, its labels, the aligned carry and bookkeeping."""

    window: WindowBatch
    labels: LabelBatch
    carry: CarriedEnvironment | None
    entity_keys: torch.Tensor                 # long [B, V'] stable keys (−1 padding)
    lanes: list[int]                          # lane of each row
    contexts: list[StreamContext]
    n_window_entities: list[int]              # real window entities per row (before the extension)
    original: WindowBatch                     # the window as the loader produced it


class StreamBridge:
    """Per-lane carry state and the contract adapter (module docstring).

    model: the NagaHana model (its TSTCT shapes the stores). carry_enabled: False runs every window
    without carry (ablation / evaluation of the windowed model); the long-term memory is still carried.
    extra_cap: largest number of carried entities appended per row (AS-404; default max_entities).
    """

    def __init__(self, model: NagaHana, *, carry_enabled: bool = True, extra_cap: int | None = None) -> None:
        assume("AS-404", by=__name__)
        assume("AS-405", by=__name__)
        self.model = model
        self.cfg = model.cfg
        self.carry_enabled = carry_enabled
        self.extra_cap = self.cfg.training.max_entities if extra_cap is None else int(extra_cap)
        self.lanes: dict[int, LaneState] = {}
        self.n_planes = len(self.cfg.graph.planes)

    # ------------------------------------------------------------------ lanes
    def _lane(self, ctx: StreamContext) -> LaneState:
        st = self.lanes.get(ctx.lane)
        if ctx.reset or st is None or st.segment_id != ctx.segment_id:
            if not ctx.reset and st is not None:
                raise InvariantViolation(f"lane {ctx.lane}: segment changed without a reset ({st.segment_id} → {ctx.segment_id})")
            store = self.model.tstct.new_store(self.cfg.memory, model_hash=f"train:{latent_space_hash(self.cfg)}",
                                               latent_space=self.cfg.latent_space, schema_version=TRAIN_SCHEMA,
                                               origin=float(ctx.origin))
            st = LaneState(segment_id=ctx.segment_id, store=store, ledger=ContactLedger(self.n_planes))
            self.lanes[ctx.lane] = st
        return st

    def reset(self) -> None:
        """Forget every lane (a new pass over the data)."""
        self.lanes.clear()

    # ------------------------------------------------------------------ before the forward pass
    def prepare(self, window: WindowBatch, labels: LabelBatch, contexts: Sequence[StreamContext]) -> PreparedBatch:
        """Extend entity tables, make contacts cumulative, export and align the carry (module docstring)."""
        b_n = window.entity_mask.shape[0]
        if len(contexts) != b_n:
            raise InvariantViolation("one StreamContext per batch row is required")
        lanes = [self._lane(c) for c in contexts]
        rows: list[RowExtension] = []
        for b, (ctx, st) in enumerate(zip(contexts, lanes, strict=True)):
            carried = {int(k) for k in ctx.carried_keys.tolist()} if self.carry_enabled else set()
            rows.append(extend_row(window, b, [int(k) for k in ctx.entity_keys.tolist()], carried, st.ledger,
                                   st.last_seen, cap=self.extra_cap))
        ext_w, ext_l, keys = assemble_extension(window, labels, rows, kinds=[st.kind for st in lanes],
                                                internals=[st.internal for st in lanes])
        assert ext_l is not None
        carry = None
        if self.carry_enabled:
            carry = self._carry(lanes, keys)
        return PreparedBatch(window=ext_w, labels=ext_l, carry=carry, entity_keys=keys, lanes=[c.lane for c in contexts],
                             contexts=list(contexts), n_window_entities=[r.n_window for r in rows], original=window)

    def _carry(self, lanes: list[LaneState], keys: torch.Tensor) -> CarriedEnvironment:
        """Export each lane's store, keep only slots of entities in the row's table, align to it."""
        carry = self.model.tstct.export_carry([st.store for st in lanes])
        return filter_carry(carry, keys).align(keys)

    # ------------------------------------------------------------------ after the forward pass
    def commit(self, prep: PreparedBatch, env: EnvironmentOut, *,
               longterm_bases: Sequence[NeuralMemoryState] | None = None) -> None:
        """Write the window into each lane's store, ledger, registry and long-term memory (no gradient).

        longterm_bases: each row's long-term memory as of the window start with the previous window's
        writes folded in, already computed by the caller (the stage objective computes it inside its
        forward pass, where a sharded long-term memory's parameters are available); None = computed here.
        """
        if longterm_bases is not None and len(longterm_bases) != len(prep.lanes):
            raise InvariantViolation("one long-term base per batch row is required")
        lanes = [self.lanes[ln] for ln in prep.lanes]
        w, o = prep.window, prep.original
        with torch.no_grad():
            if self.carry_enabled:
                # The out-of-order augmentation (AS-320) jitters times inside their recorded uncertainty;
                # a jittered first update may fall before the previous window's last one. The store needs
                # time order, so written times are clamped to its clock (a shift ≤ the uncertainty, AS-405).
                floor = torch.tensor([st.store.clock - (float(w.origin[b]) - st.store.origin)
                                      for b, st in enumerate(lanes)], dtype=torch.float64)
                times = torch.maximum(w.positions.time.to(torch.float64), floor[:, None])
                w_store = dataclasses.replace(w, positions=dataclasses.replace(w.positions, time=times))
                self.model.tstct.carry_out(env, w_store, keep=CarryPolicy(stores=[st.store for st in lanes],
                                                                           entity_keys=prep.entity_keys))
            for b, (st, ctx) in enumerate(zip(lanes, prep.contexts, strict=True)):
                v_w = prep.n_window_entities[b]
                keys = ctx.entity_keys
                origin = float(o.origin[b])
                st.ledger.update(keys, o.contact1[b], o.contact_planes[b], origin)
                for i, k in enumerate(keys.tolist()):
                    st.kind[int(k)] = int(o.entity_kind[b, i])
                    st.internal[int(k)] = bool(o.entity_internal[b, i])
                # latest state time of every window entity
                pos = o.positions
                real = pos.mask[b] & (pos.entity[b] >= 0)
                for p in torch.nonzero(real).flatten().tolist():
                    st.last_seen[int(keys[int(pos.entity[b, p])])] = float(pos.time[b, p]) + origin
                del v_w
            # long-term memory: fold the previous window's writes into the base, queue this window's writes
            for b, st in enumerate(lanes):
                state = self._lane_state(st) if longterm_bases is None else longterm_bases[b]
                st.lt_base = _detach_state(state)
                st.lt_pending = []
                m_n = w.triggers.time.shape[1]
                for m in range(m_n):
                    if not bool(w.triggers.mask[b, m]):
                        continue
                    x, mask = self.model.longterm_inputs(env, w, m)
                    st.lt_pending.append((x[b:b + 1].detach(), mask[b:b + 1].detach()))

    # ------------------------------------------------------------------ long-term memory
    def _lane_state(self, st: LaneState) -> NeuralMemoryState:
        state = self.model.longterm_init(1) if st.lt_base is None else st.lt_base
        for x, mask in st.lt_pending:
            state = self.model.longterm_write(state, x, mask)
        return state

    def longterm_states(self, prep: PreparedBatch, env: EnvironmentOut) -> list[NeuralMemoryState]:
        """Per-trigger long-term states of the batch (AS-222): [start, start + write(τ_0), …]."""
        return self.model.longterm_per_trigger(self.longterm_state(prep), env, prep.window)

    # ------------------------------------------------------------------ Imagination across windows
    def past(self, prep: PreparedBatch) -> PastImagination | None:
        """The carried Imagination of each row's lane, aligned to this batch's tokens (None if no lane has any)."""
        lanes = [self.lanes[ln] for ln in prep.lanes]
        if all(st.past is None for st in lanes):
            return None
        tc = self.cfg.taaft
        keep = tc.imagination_triggers
        v_new = prep.entity_keys.shape[1]
        n_new = v_new + tc.adversary_slots
        rows: list[PastImagination] = []
        for b, st in enumerate(lanes):
            lp = st.past
            if lp is None:
                rows.append(_empty_past(n_new, keep, blocks=tc.blocks, heads=tc.heads, head_dim=tc.dim // tc.heads,
                                        d_hyp=tc.d_hyp))
                continue
            pos = {k: i for i, k in enumerate(lp.keys) if k >= 0}
            v_old = len(lp.keys)
            idx = [pos.get(int(k), -1) if int(k) >= 0 else -1 for k in prep.entity_keys[b].tolist()]
            idx += [v_old + g for g in range(tc.adversary_slots)]                      # slots map to themselves
            shift = torch.tensor([lp.origin - float(prep.window.origin[b])], dtype=torch.float64)
            rows.append(past_from_analysis(lp.out, lp.trigger_time, lp.trigger_mask, keep=keep, origin_shift=shift,
                                           token_index=torch.tensor([idx], dtype=torch.long), previous=lp.received,
                                           detach=True))
        return PastImagination(
            kv=[(torch.cat([r.kv[i][0] for r in rows]), torch.cat([r.kv[i][1] for r in rows])) for i in range(tc.blocks)],
            time=torch.cat([r.time for r in rows]), mask=torch.cat([r.mask for r in rows]),
            y=torch.cat([r.y if r.y is not None else torch.zeros(*r.mask.shape, tc.d_hyp) for r in rows]))

    def record_analysis(self, prep: PreparedBatch, out: AnalysisOut, received: PastImagination | None) -> None:
        """Keep each row's TAAFT call (detached) for the next window of its lane; rows without a trigger keep theirs."""
        trig = prep.window.triggers
        n = out.token_mask.shape[2]
        m_tr = out.token_mask.shape[1]
        for b, ln in enumerate(prep.lanes):
            if not bool(trig.mask[b].any()):
                continue
            row = AnalysisOut(
                context=out.y[b:b + 1].detach(), token_mask=out.token_mask[b:b + 1],
                imagination_kv=[(k[b:b + 1].detach().to(CACHE_DTYPE), v[b:b + 1].detach().to(CACHE_DTYPE))
                                for k, v in out.imagination_kv],                          # stored fp32 (D-54)
                y0=out.y[b:b + 1].detach(), y=out.y[b:b + 1].detach(), energy_trace=out.energy_trace.detach(),
                lens_energy={}, lens_share={}, readouts={}, passes=out.passes, descent_steps=out.descent_steps)
            rec = None
            if received is not None:
                rec = PastImagination(kv=[(k[b:b + 1].detach().to(CACHE_DTYPE), v[b:b + 1].detach().to(CACHE_DTYPE))
                                          for k, v in received.kv],
                                      time=received.time[b:b + 1], mask=received.mask[b:b + 1],
                                      y=None if received.y is None else received.y[b:b + 1].detach())
            assert n == prep.entity_keys.shape[1] + self.cfg.taaft.adversary_slots and m_tr == trig.time.shape[1]
            self.lanes[ln].past = _LaneImagination(out=row, trigger_time=trig.time[b:b + 1].to(torch.float64),
                                                   trigger_mask=trig.mask[b:b + 1], received=rec,
                                                   keys=[int(k) for k in prep.entity_keys[b].tolist()],
                                                   origin=float(prep.window.origin[b]))

    def lane_bases(self, prep: PreparedBatch) -> list[NeuralMemoryState]:
        """Each row's long-term memory as of the window start, detached (what `commit` folds in)."""
        with torch.no_grad():
            return [_detach_state(self._lane_state(self.lanes[ln])) for ln in prep.lanes]

    def state_dict(self) -> dict[str, object]:
        """The per-lane carry (stores, ledgers, registries, long-term memory, Imagination) for checkpoints."""
        return {"lanes": dict(self.lanes), "carry_enabled": self.carry_enabled, "extra_cap": self.extra_cap}

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        """Restore `state_dict()` (the bridge's settings must match)."""
        if bool(state["carry_enabled"]) != self.carry_enabled or int(state["extra_cap"]) != self.extra_cap:  # type: ignore[call-overload]
            raise InvariantViolation("bridge state was taken with other carry settings")
        lanes = state["lanes"]
        if not isinstance(lanes, dict):
            raise InvariantViolation("malformed bridge state")
        self.lanes = {int(k): v for k, v in lanes.items()}

    def longterm_state(self, prep: PreparedBatch) -> NeuralMemoryState:
        """The batch's long-term memory as of the window start (with gradient through one window of writes)."""
        states = [self._lane_state(self.lanes[ln]) for ln in prep.lanes]
        return NeuralMemoryState(w1=torch.cat([s.w1 for s in states]), w2=torch.cat([s.w2 for s in states]),
                                 s1=torch.cat([s.s1 for s in states]), s2=torch.cat([s.s2 for s in states]),
                                 trigger=max(s.trigger for s in states))


def _empty_past(n: int, keep: int, *, blocks: int, heads: int, head_dim: int, d_hyp: int) -> PastImagination:
    """A carry with no triggers (mask False everywhere; time at the padding value of `past_from_analysis`)."""
    z = torch.zeros(1, heads, n, keep, head_dim, dtype=CACHE_DTYPE)
    return PastImagination(kv=[(z.clone(), z.clone()) for _ in range(blocks)], time=torch.full((1, keep), -1e18, dtype=torch.float64),
                           mask=torch.zeros(1, n, keep, dtype=torch.bool), y=torch.zeros(1, n, keep, d_hyp))


def _detach_state(s: NeuralMemoryState) -> NeuralMemoryState:
    return NeuralMemoryState(w1=s.w1.detach(), w2=s.w2.detach(), s1=s.s1.detach(), s2=s.s2.detach(), trigger=s.trigger)


def filter_carry(carry: CarriedEnvironment, keys: torch.Tensor) -> CarriedEnvironment:
    """Keep only carried slots whose key is in the row's entity table (keys [B, V'], −1 padding); re-pad C.

    Slots of other entities would be masked anyway (`align` maps them to −1); dropping them first
    bounds the key axis by |table| × slots_per_entity instead of the whole store.
    """
    b_n = carry.key.shape[0]
    keep = torch.zeros_like(carry.mask)
    for b in range(b_n):
        table = keys[b][keys[b] >= 0]
        keep[b] = carry.mask[b] & torch.isin(carry.key[b], table)
    c_new = max(1, int(keep.sum(1).max()))
    order = torch.argsort((~keep).to(torch.int8), dim=1, stable=True)[:, :c_new]      # kept slots first, in order
    ok = torch.gather(keep, 1, order)

    def take(x: torch.Tensor, fill: float | int | bool) -> torch.Tensor:
        g = torch.gather(x, 1, order)
        return torch.where(ok, g, torch.full_like(g, fill))

    def take_kv(x: torch.Tensor) -> torch.Tensor:
        # x [B, H, C, d_h] → [B, H, C', d_h]
        idx = order[:, None, :, None].expand(-1, x.shape[1], -1, x.shape[3])
        return torch.gather(x, 2, idx) * ok[:, None, :, None].to(x.dtype)

    return CarriedEnvironment(
        k=[take_kv(k) for k in carry.k], v=[take_kv(v) for v in carry.v], origin=carry.origin,
        time=take(carry.time, 0.0), key=take(carry.key, -1), count=take(carry.count, 1.0),
        bucket=take(carry.bucket, False) & ok, latest=take(carry.latest, False) & ok, mask=ok,
    )


__all__ = ["ContactLedger", "LaneState", "PreparedBatch", "RowExtension", "StreamBridge", "assemble_extension", "extend_row",
           "filter_carry", "min_max_product"]
