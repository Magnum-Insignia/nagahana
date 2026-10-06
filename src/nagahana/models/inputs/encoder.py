"""FieldEncoder: the input layer of NagaHana (build-spec §2.1; engineer A, Perception).

Purpose
-------
Turn the value/status matrices of a window (`models.batch.FieldBatch`, built from `ColumnarUpdates`)
into one vector per state update, u_i ∈ ℝ^{d_update}, plus the attention-pooling weights that say
which fields each update was "about" (the first explanation layer, build-spec §2.1 and §0.6).

Owner sources and decisions
---------------------------
- D-41 (absence is not zero): a cell that does not contribute never enters arithmetic; it is
  represented by its status and a learned "no value" vector.
- D-49 (positional encodings): fields are identified by a learned slot embedding per column, never
  by an index of anything an attacker can inflate.
- D-50 (clock features): built behind `FieldEncoderConfig.clock_features`, off in training on lab
  datasets.
- P-22 (slot keyed by stable field ID), assumed in AS-31; categorical hashing AS-33; status codes of
  `datamodel.columnar.STATUS_ORDER` plus `vocab.MASK_STATUS`.
- New assumptions (docs/assumptions/perception.md): AS-100 (numeric clip, the fixed hash), AS-107
  (pooling attends to every cell, absent ones included; MASK has its own vectors).

Maths
-----
For update i and column c with status m = status(i, c) and kind κ = kind(c):

    f_{i,c} = s_{slot(c)} + σ_m + 𝟙[m ∈ 𝒪] · v_κ(x_{i,c}) + 𝟙[m ∉ 𝒪] · a_m

with 𝒪 = {OBSERVED, STALE, LOW_RELIABILITY} the contributing statuses (`datamodel.status`), and

- CONTINUOUS, COUNT, HISTOGRAM bin (AS-31; Gorishniy et al., NeurIPS 2022, arXiv:2203.05556):
      x̃ = clip(sign(x)·log(1 + |x|), ±L),   v(x) = W_c · [sin(2π w_c x̃) ; cos(2π w_c x̃)] + b_c
  with learned per-slot frequencies w_c ∈ ℝ^k (the `nn.numeric.PeriodicEmbedding` parameters indexed
  by slot) and L = `max_log_magnitude` (AS-100; e^50 ≈ 5·10^21 is beyond any byte or packet count).
- CATEGORICAL (ports, protocol, codes; AS-33):  v(x) = T[h(slot(c), round(x)) mod R_hash], with h the
  fixed Carter–Wegman integer hash below.
- BITMASK:  v(x) = Σ_{k < max_bits} bit_k(x) · E_{slot(c), k}  (one learned vector per slot and bit;
  bits at or above `max_bits` are ignored, documented in AS-100).

The update vector is attention pooling with a learned query q and H_pool heads (shared
`nn.attention.MultiHeadAttention`, no null key: every update has C ≥ 1 field states, so no row is
empty):

    α_{i,h,c} = softmax_c( ⟨q̂_h, k̂_{h}(N(f_{i,c}))⟩ / √d_h ),   u_i = N_out( W_o [Σ_c α_{i,h,c} v_h(N(f_{i,c}))]_h + clock(t_i) )

with N an RMSNorm (pre-norm on field states) and q̂, k̂ QK-normed (AS-32). α is returned as
`field_weights` [B, U, H_pool, C].

The fixed categorical hash (AS-100)
-----------------------------------
Carter & Wegman, "Universal classes of hash functions", JCSS 18(2), 1979: for a prime p,
h_{a,b}(x) = ((a·x + b) mod p) mod m is a universal family. We fix one member, over pairs:

    p  = 2³¹ − 1 (Mersenne prime)
    h₁ = (A·(slot mod p) + B·(code mod p) + C) mod p        each product reduced mod p first
    h  = ((A₂·h₁ + C₂) mod p) mod R_hash

All intermediate values stay below 2⁶², so int64 arithmetic never overflows: the result is exact and
identical on every platform and in every run (tested against a pure-Python implementation). Codes
are unique modulo p, i.e. for |code| < 2³¹ − 1 (ports, protocol numbers, vocabulary indices).

Invariants (tested in tests/test_perception_inputs.py)
------------------------------------------------------
- NaN never propagates: values of non-contributing cells are replaced by 0 *before* any arithmetic
  and their encodings are discarded; the output is identical whatever those cells hold.
- A contributing cell holding a non-finite value raises `InvariantViolation` (a data-model breach,
  `ColumnarUpdates.validate` should have caught it).
- MASK status (self-supervised masking) is a non-contributing status with its own σ and a vectors.
- `categorical_row` is deterministic across runs and platforms.

Extension points
----------------
- New column kinds: add a branch in `field_states` (the kind codes are `vocab.COLUMN_KINDS`).
- Explanations pass perturbed or baseline field states through `forward(..., states=...)`.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from nagahana.core.errors import InvariantViolation
from nagahana.datamodel.columnar import STATUS_CODE
from nagahana.datamodel.fields import Kind
from nagahana.datamodel.status import CONTRIBUTING
from nagahana.governance.assumptions import assume
from nagahana.models.batch import FieldBatch
from nagahana.models.config.components import FieldEncoderConfig
from nagahana.models.vocab import COLUMN_KIND_CODE, N_STATUS_CODES
from nagahana.nn.attention import MultiHeadAttention
from nagahana.nn.norms import RMSNorm
from nagahana.nn.numeric import PeriodicEmbedding, signed_log1p
from nagahana.nn.positional import ClockFeatures

# ----------------------------------------------------------------------------- status and kind codes
#: Bool lookup over status codes (incl. MASK): True where the status contributes evidence (D-41).
CONTRIBUTING_STATUS: torch.Tensor = torch.zeros(N_STATUS_CODES, dtype=torch.bool)
for _s in CONTRIBUTING:
    CONTRIBUTING_STATUS[STATUS_CODE[_s]] = True

_KIND_CONT = COLUMN_KIND_CODE[Kind.CONTINUOUS]
_KIND_COUNT = COLUMN_KIND_CODE[Kind.COUNT]
_KIND_CAT = COLUMN_KIND_CODE[Kind.CATEGORICAL]
_KIND_BIT = COLUMN_KIND_CODE[Kind.BITMASK]
_KIND_HIST = COLUMN_KIND_CODE[Kind.HISTOGRAM]
NUMERIC_KINDS: tuple[int, ...] = (_KIND_CONT, _KIND_COUNT, _KIND_HIST)

# ----------------------------------------------------------------------------- the fixed hash (AS-100)
_P = 2_147_483_647          # 2^31 − 1, a Mersenne prime
_A, _B, _C = 1_597_334_677, 1_181_783_497, 1_013_904_223
_A2, _C2 = 914_237_431, 362_437
_MAX_EXACT = float(2**53)   # floats beyond this are not integers we can trust; clamp before the cast


def categorical_row(slot: torch.Tensor, code: torch.Tensor, rows: int) -> torch.Tensor:
    """h(slot, code) mod `rows` with the fixed Carter–Wegman hash of the module docstring.

    slot, code: int64 tensors (broadcastable). Returns int64 in [0, rows). Exact int64 arithmetic,
    so the result is the same on every platform and run.
    """
    if rows <= 0:
        raise ValueError("rows must be positive")
    s = torch.remainder(slot.to(torch.int64), _P)              # [..] in [0, p)
    c = torch.remainder(code.to(torch.int64), _P)              # negative codes wrap into [0, p)
    # Each product < p·2^31 < 2^62; reduce before adding so the sum stays < 2^63.
    h1 = torch.remainder(torch.remainder(_A * s, _P) + torch.remainder(_B * c, _P) + _C, _P)
    h2 = torch.remainder(_A2 * h1 + _C2, _P)                   # second mixing round
    return torch.remainder(h2, rows)


def float_to_code(x: torch.Tensor) -> torch.Tensor:
    """Round a float code (port 443.0) to int64, clamped to ±2^53 so the cast is defined."""
    return torch.round(x.to(torch.float64)).clamp(-_MAX_EXACT, _MAX_EXACT).to(torch.int64)


class FieldEncoder(nn.Module):
    """Field states and attention-pooled update vectors. See the module docstring.

    Parameters
    ----------
    cfg: `FieldEncoderConfig` (L preset: 128 slots, d_field 128, 65,536 hash rows, 4 pooling heads).
    """

    def __init__(self, cfg: FieldEncoderConfig) -> None:
        super().__init__()
        assume("AS-31", by=__name__)   # signed log1p + periodic numeric encoding, slot per field ID
        assume("AS-33", by=__name__)   # categorical fields via hashed rows
        if cfg.d_update % cfg.pool_heads:
            raise ValueError("d_update must be divisible by pool_heads")
        self.cfg = cfg
        d = cfg.d_field
        # s_c: one learned vector per field slot (D-49, P-22).
        self.slot = nn.Embedding(cfg.n_slots, d)
        # σ_m and a_m: status embeddings and "no value" vectors over the 5 statuses + MASK.
        self.status = nn.Embedding(N_STATUS_CODES, d)
        self.absent = nn.Embedding(N_STATUS_CODES, d)
        # v_numeric: per-slot learned frequencies and linear map (indexed by slot below).
        self.numeric = PeriodicEmbedding(cfg.n_slots, cfg.n_frequencies, d, sigma=cfg.freq_sigma)
        # v_categorical: hashed table T (AS-33).
        self.categorical = nn.Embedding(cfg.hash_rows, d)
        # v_bitmask: one vector per (slot, bit).
        self.bits = nn.Parameter(torch.randn(cfg.n_slots, cfg.max_bits, d) * (1.0 / math.sqrt(cfg.max_bits)))
        # Attention pooling: learned query of width d_update, keys/values from field states.
        self.state_norm = RMSNorm(d)
        self.pool = MultiHeadAttention(cfg.d_update, cfg.pool_heads, kv_dim=d, null_kv=False, qk_norm=True)
        self.query = nn.Parameter(torch.randn(cfg.d_update) * 0.02)
        # Clock features (D-50): zeros unless enabled; never on in lab-dataset training.
        self.clock = ClockFeatures(cfg.d_update, enabled=cfg.clock_features)
        self.out_norm = RMSNorm(cfg.d_update)
        self.register_buffer("_contrib", CONTRIBUTING_STATUS.clone(), persistent=False)
        self._contrib: torch.Tensor

    # ------------------------------------------------------------------------------ field states
    def field_states(self, fields: FieldBatch) -> torch.Tensor:
        """f_{i,c} for every cell: [B, U, C] → [B, U, C, d_field]. See the module docstring."""
        values, status = fields.values, fields.status
        if not values.is_floating_point():
            values = values.float()                                          # contract: float32 [B, U, C]
        kinds, slots = fields.column_kind.long(), fields.column_slot.long()
        b, u, c = values.shape
        # ---- validate the contract before touching any value
        if status.shape != values.shape or kinds.shape != (c,) or slots.shape != (c,):
            raise InvariantViolation("FieldBatch shapes disagree (values/status [B,U,C], kind/slot [C])")
        if status.numel() and (int(status.min()) < 0 or int(status.max()) >= N_STATUS_CODES):
            raise InvariantViolation(f"status codes must lie in [0, {N_STATUS_CODES})")
        if c and (int(slots.min()) < 0 or int(slots.max()) >= self.cfg.n_slots):
            raise InvariantViolation(f"column_slot must lie in [0, n_slots={self.cfg.n_slots})")
        contrib = self._contrib[status]                                       # [B, U, C] bool
        if bool((contrib & ~torch.isfinite(values)).any()):
            raise InvariantViolation("a contributing cell holds a non-finite value (D-41 pairing broken)")
        # ---- D-41: non-contributing values are replaced by 0 BEFORE any arithmetic.
        x = torch.where(contrib, values, torch.zeros((), dtype=values.dtype))   # [B, U, C] finite
        v = x.new_zeros(b, u, c, self.cfg.d_field)                              # value encodings
        # ---- numeric columns: signed log1p → periodic features → per-slot linear map
        num_idx = torch.nonzero(torch.isin(kinds, torch.tensor(NUMERIC_KINDS))).flatten()
        if num_idx.numel():
            sl = slots[num_idx]                                                  # [Cn]
            xt = signed_log1p(x[..., num_idx].float()).clamp(-self.cfg.max_log_magnitude, self.cfg.max_log_magnitude)
            ang = 2 * math.pi * xt.unsqueeze(-1) * self.numeric.freq[sl]         # [B, U, Cn, k]
            per = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)            # [B, U, Cn, 2k]
            v_num = torch.einsum("...ck,cko->...co", per, self.numeric.weight[sl]) + self.numeric.bias[sl]
            v = v.index_copy(2, num_idx, v_num.to(v.dtype))
        # ---- categorical columns: hashed table row h(slot, code)
        cat_idx = torch.nonzero(kinds == _KIND_CAT).flatten()
        if cat_idx.numel():
            code = float_to_code(x[..., cat_idx])                                # [B, U, Cc] int64
            rows = categorical_row(slots[cat_idx].expand_as(code), code, self.cfg.hash_rows)
            v = v.index_copy(2, cat_idx, self.categorical(rows).to(v.dtype))
        # ---- bitmask columns: sum of per-(slot, bit) embeddings over the set bits
        bit_idx = torch.nonzero(kinds == _KIND_BIT).flatten()
        if bit_idx.numel():
            code = float_to_code(x[..., bit_idx]).clamp_min(0)                   # [B, U, Cb]
            shifts = torch.arange(self.cfg.max_bits, dtype=torch.int64)
            bits = torch.bitwise_and(torch.bitwise_right_shift(code.unsqueeze(-1), shifts), 1).to(v.dtype)
            v_bit = torch.einsum("...ck,cko->...co", bits, self.bits[slots[bit_idx]])
            v = v.index_copy(2, bit_idx, v_bit)
        known = torch.isin(kinds, torch.tensor((*NUMERIC_KINDS, _KIND_CAT, _KIND_BIT)))
        if not bool(known.all()):
            raise InvariantViolation(f"unknown column kind codes {kinds[~known].tolist()}")
        # ---- f = s + σ + (v if contributing else a_status)
        s = self.slot(slots)                                                     # [C, d]
        sig = self.status(status)                                                # [B, U, C, d]
        val = torch.where(contrib.unsqueeze(-1), v, self.absent(status))         # [B, U, C, d]
        return s + sig + val

    # ------------------------------------------------------------------------------ pooling
    def forward(
        self,
        fields: FieldBatch,
        *,
        origin: torch.Tensor | None = None,
        update_time: torch.Tensor | None = None,
        states: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """(u [B, U, d_update], field_weights [B, U, pool_heads, C]).

        origin: float64 [B] epoch seconds; update_time: float64 [B, U] relative seconds. Both are
            needed only when `clock_features` is on (D-50).
        states: precomputed or perturbed field states [B, U, C, d_field] (explanations); when given,
            `fields` supplies only shapes.
        """
        f = self.field_states(fields) if states is None else states             # [B, U, C, d]
        b, u, c, d = f.shape
        kv_src = self.state_norm(f).reshape(b * u, c, d)                         # [B·U, C, d]
        q = self.pool.project_q(self.query.view(1, 1, -1).expand(b * u, 1, -1))  # [B·U, H, 1, d_h]
        k, v = self.pool.project_kv(kv_src)                                      # [B·U, H, C, d_h]
        pooled, w = self.pool.attend(q, k, v, need_weights=True)                 # [B·U, 1, d_update], [B·U, H, 1, C]
        assert w is not None
        out = pooled.view(b, u, -1)
        # Clock features (D-50): added only when enabled; needs absolute epoch seconds in float64.
        if self.clock.enabled:
            if origin is None or update_time is None:
                raise InvariantViolation("clock_features is on: origin and update_time are required (D-50)")
            epoch = origin.to(torch.float64).view(-1, 1) + update_time.to(torch.float64)   # [B, U]
            out = out + self.clock(epoch).to(out.dtype)
        weights = w.view(b, u, self.cfg.pool_heads, c)                           # [B, U, H, C]
        return self.out_norm(out), weights
