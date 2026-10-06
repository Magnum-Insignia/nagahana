"""Masked-generative field-state transformer (MaskGIT-style) and its autoregressive reading.

Purpose
-------
The owner's Generator uses "generative training (where we make the model generate the missing parts
etc to make it our expert network data generator), autoregressive & diffusion methods" [A-17]. This is
the "generate the missing parts" family: a transformer over the *cells* of a short sequence of state
updates (records) that predicts hidden cells from the visible ones, and generates by iterative
unmasking (MaskGIT, Chang et al., "MaskGIT: Masked Generative Image Transformer", CVPR 2022,
arXiv:2202.04200). Run left-to-right over records it is the autoregressive family (build-spec §2.11
item 3; AS-364). It is training-only data augmentation (D-40), outside the ≈ 1 B count.

Representation (AS-362, AS-363)
-------------------------------
A sequence of L records × C columns gives L·C positions. The state of cell (ℓ, c) enters as

    f_{ℓ,c} = s_c + σ_{status(ℓ,c)} + 𝟙[contributes ∧ visible] · v_c(k_{ℓ,c}) + 𝟙[excluded] · a + g_{stage(ℓ)}

- s_c: column-slot embedding (field identity, keyed by column as in the input layer, D-49);
- σ: status embedding over the 5 statuses + MASK (vocab.MASK_STATUS) — hidden cells carry MASK;
- v_c(k): embedding of the cell's class k (codec.py: signed-log1p bin or vocabulary class);
- a: learned "no value" vector for excluded cells (absence is a status, never a zero value, D-41);
- g: stage-label embedding of the record (15 stages + unknown). Generation is *conditioned on the
  label*, which is why a learned variant may copy its source's label (AS-366).
Time enters only through continuous-time rotary encoding of queries and keys by the record's event
time (nn.positional.TimeRotary, D-49): no index positions.

Blocks are the shared pre-norm `nn.blocks.SelfBlock` (RMSNorm, QK-norm, SwiGLU, null key; AS-32).
A linear head per column gives logits over that column's classes: p_θ(k_{ℓ,c} | visible cells).

Training (MaskGIT §3.2; AS-363, AS-364)
---------------------------------------
Per sequence draw r ~ U(0, 1) and hide ⌈γ(r)·N⌉ of the N contributing modelled cells, with the cosine
schedule γ(r) = cos(π r / 2) (MaskGIT found cosine best among the schedules it compared). Loss:

    L = − (1 / |M|) Σ_{(ℓ,c) ∈ M} log p_θ(k_{ℓ,c} | cells ∉ M)

With probability `ar_fraction` a sequence is instead *truncated* at a random record ℓ (later records
are removed from attention) and only cells of record ℓ are hidden: this trains p(record ℓ | records < ℓ).

Generation
----------
- MaskGIT iterative decoding over T steps: sample every hidden cell; confidence = log p of the sampled
  class + Gumbel noise × τ·(1 − t/T) (annealed); keep the ⌊γ(t/T)·N⌋ least confident cells hidden.
  Disallowed classes (unused vocabulary slots, an empty OOV pool) get −∞ logits.
- Autoregressive: records ℓ = 0 … L−1 in time order; for record ℓ only records ≤ ℓ are attended to,
  and its hidden cells are filled by the MaskGIT loop. Earlier generated records are context.

Invariants (tests/test_generator_learned.py)
- statuses are never generated: only contributing cells are hidden and filled (D-41);
- truncated (AR) records are invisible to earlier records (attention mask), so the AR reading has no
  look-ahead;
- the masked-cell loss decreases on synthetic data (fixed seed).

Decisions: D-40, D-41, D-49. Assumptions: AS-27, AS-31, AS-32, AS-362, AS-363, AS-364, AS-366.
Extension points: per-kind heads (e.g. a Gaussian head for continuous cells) would replace `heads`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from nagahana.datamodel.columnar import STATUS_CODE
from nagahana.datamodel.status import CONTRIBUTING
from nagahana.models.config.components import GeneratorConfig
from nagahana.models.generator.codec import FieldCodec
from nagahana.models.vocab import MASK_STATUS, N_STAGES, N_STATUS_CODES
from nagahana.nn.attention import rotate_heads
from nagahana.nn.blocks import AttnContext, SelfBlock
from nagahana.nn.norms import RMSNorm
from nagahana.nn.positional import TimeRotary

_CONTRIB = torch.tensor(sorted(STATUS_CODE[s] for s in CONTRIBUTING), dtype=torch.long)


def contributing(status: torch.Tensor) -> torch.Tensor:
    """bool mask of contributing status codes (any shape)."""
    return torch.isin(status, _CONTRIB.to(status.device))


@dataclass
class RecordBatch:
    """B sequences of up to L records (rows of `ColumnarUpdates`) for the learned families.

    codes: long [B, L, C] codec classes, −1 where excluded / not modelled. status: long [B, L, C].
    times: float64 [B, L] seconds relative to each sequence's first record. record_mask: bool [B, L].
    stage: long [B, L] stage code or −1. rows: long [B, L] source row of each record (−1 = padding).
    """

    codes: torch.Tensor
    status: torch.Tensor
    times: torch.Tensor
    record_mask: torch.Tensor
    stage: torch.Tensor
    rows: torch.Tensor


def records_from_matrices(codes: np.ndarray, status: np.ndarray, times: np.ndarray, stage: np.ndarray,
                          max_records: int) -> RecordBatch:
    """Cut a window's rows (already in time order) into consecutive sequences of ≤ `max_records`."""
    n, c = codes.shape
    b = max(1, math.ceil(n / max_records))
    out_c = np.full((b, max_records, c), -1, dtype=np.int64)
    out_s = np.zeros((b, max_records, c), dtype=np.int64)
    out_t = np.zeros((b, max_records), dtype=np.float64)
    out_m = np.zeros((b, max_records), dtype=bool)
    out_g = np.full((b, max_records), -1, dtype=np.int64)
    out_r = np.full((b, max_records), -1, dtype=np.int64)
    for i in range(b):
        rows = np.arange(i * max_records, min(n, (i + 1) * max_records))
        k = len(rows)
        out_c[i, :k], out_s[i, :k] = codes[rows], status[rows]
        out_t[i, :k] = times[rows] - times[rows[0]] if k else 0.0         # time relative to the sequence start
        out_m[i, :k], out_g[i, :k], out_r[i, :k] = True, stage[rows], rows
    return RecordBatch(torch.from_numpy(out_c), torch.from_numpy(out_s), torch.from_numpy(out_t),
                       torch.from_numpy(out_m), torch.from_numpy(out_g), torch.from_numpy(out_r))


class MaskedFieldTransformer(nn.Module):
    """Field-state transformer over the cells of a record sequence. See the module docstring.

    n_classes: classes per column (codec.n_classes(j)); class_masks: bool [K_c] per column (allowed).
    """

    def __init__(self, cfg: GeneratorConfig, n_classes: Sequence[int], class_masks: Sequence[np.ndarray] | None = None) -> None:
        super().__init__()
        self.n_columns = len(n_classes)
        self.n_classes = tuple(int(k) for k in n_classes)
        dim, heads = cfg.dim, cfg.heads
        self.slot = nn.Embedding(self.n_columns, dim)                       # s_c
        self.status_emb = nn.Embedding(N_STATUS_CODES, dim)                 # σ (5 statuses + MASK)
        offsets = np.concatenate([[0], np.cumsum(self.n_classes)[:-1]]).astype(np.int64)
        self.offsets: torch.Tensor
        self.register_buffer("offsets", torch.from_numpy(offsets), persistent=False)
        self.value_emb = nn.Embedding(int(sum(self.n_classes)), dim)        # v_c(k), one table with offsets
        self.no_value = nn.Parameter(torch.zeros(dim))                      # a (excluded cells)
        self.stage_emb = nn.Embedding(N_STAGES + 1, dim)                    # g (last row = unknown)
        self.blocks = nn.ModuleList(SelfBlock(dim, heads, mlp_hidden=cfg.mlp_hidden) for _ in range(cfg.blocks))
        self.rotary = TimeRotary(dim // heads)
        self.norm = RMSNorm(dim)
        self.heads = nn.ModuleList(nn.Linear(dim, k, bias=False) for k in self.n_classes)
        masks = class_masks if class_masks is not None else [np.ones(k, dtype=bool) for k in self.n_classes]
        for j, m in enumerate(masks):
            self.register_buffer(f"allowed_{j}", torch.from_numpy(np.asarray(m, dtype=bool)), persistent=False)
        nn.init.normal_(self.value_emb.weight, std=0.02)
        nn.init.normal_(self.slot.weight, std=0.02)

    def allowed(self, j: int) -> torch.Tensor:
        """bool [K_j]: decodable classes of column j."""
        out: torch.Tensor = getattr(self, f"allowed_{j}")
        return out

    def embed(self, codes: torch.Tensor, status: torch.Tensor, hidden: torch.Tensor, stage: torch.Tensor) -> torch.Tensor:
        """Cell states f [B, L·C, d] (see the module docstring)."""
        b, ell, c = codes.shape
        contrib = contributing(status)                                       # [B, L, C]
        visible = contrib & ~hidden & (codes >= 0)
        st = torch.where(hidden, torch.full_like(status, MASK_STATUS), status)
        idx = (codes.clamp_min(0) + self.offsets.view(1, 1, c)).clamp_max(self.value_emb.num_embeddings - 1)
        val = self.value_emb(idx) * visible.unsqueeze(-1)                    # value only where visible
        val = val + (~contrib).unsqueeze(-1) * self.no_value                 # "no value" where excluded
        g = self.stage_emb(torch.where(stage >= 0, stage, torch.full_like(stage, N_STAGES)))   # [B, L, d]
        f = self.slot.weight.view(1, 1, c, -1) + self.status_emb(st) + val + g.unsqueeze(2)
        return f.reshape(b, ell * c, -1)

    def forward(self, codes: torch.Tensor, status: torch.Tensor, hidden: torch.Tensor, times: torch.Tensor,
                record_mask: torch.Tensor, stage: torch.Tensor) -> list[torch.Tensor]:
        """Logits per column, each [B, L, K_c]. Hidden cells (`hidden`, bool [B, L, C]) carry MASK."""
        b, ell, c = codes.shape
        x = self.embed(codes, status, hidden, stage)                         # [B, T = L·C, d]
        cell_t = times.unsqueeze(-1).expand(b, ell, c).reshape(b, ell * c)   # [B, T] float64 seconds
        cos, sin = self.rotary.angles(cell_t)                                # [B, T, d_h]
        keys_ok = record_mask.unsqueeze(-1).expand(b, ell, c).reshape(b, 1, 1, ell * c)
        ctx = AttnContext(q_hook=lambda q: rotate_heads(q, cos, sin), k_hook=lambda k: rotate_heads(k, cos, sin),
                          allowed=keys_ok)
        for blk in self.blocks:
            assert isinstance(blk, SelfBlock)
            x, _ = blk(x, blk.kv(x, ctx), ctx)
        h = self.norm(x).view(b, ell, c, -1)                                 # [B, L, C, d]
        return [head(h[:, :, j]) for j, head in enumerate(self.heads)]


# ============================================================================== training
def cosine_gamma(r: torch.Tensor) -> torch.Tensor:
    """MaskGIT mask schedule γ(r) = cos(π r / 2): share of cells still hidden at progress r ∈ [0, 1]."""
    return torch.cos(math.pi / 2 * r)


def training_mask(batch: RecordBatch, modelled: torch.Tensor, *, ar_fraction: float,
                  generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """(hidden bool [B, L, C], effective record_mask bool [B, L]) for one training step (AS-363, AS-364).

    modelled: bool [C], columns the codec models. Only contributing, modelled, real cells are hidden.
    """
    b, ell, c = batch.codes.shape
    candidates = (contributing(batch.status) & (batch.codes >= 0) & modelled.view(1, 1, c)
                  & batch.record_mask.unsqueeze(-1))                         # [B, L, C]
    hidden = torch.zeros_like(candidates)
    rec = batch.record_mask.clone()
    for i in range(b):
        n_rec = int(batch.record_mask[i].sum())
        cand = candidates[i].clone()
        if n_rec > 0 and float(torch.rand((), generator=generator)) < ar_fraction:
            # autoregressive example: truncate after a random record ℓ and hide cells of ℓ only
            last = int(torch.randint(0, n_rec, (), generator=generator))
            rec[i, last + 1:] = False
            keep_rec = torch.zeros(ell, dtype=torch.bool)
            keep_rec[last] = True
            cand &= keep_rec.unsqueeze(-1)
        flat = cand.flatten().nonzero().squeeze(-1)
        if flat.numel() == 0:
            continue
        r = torch.rand((), generator=generator)
        n_hide = max(1, math.ceil(float(cosine_gamma(r)) * flat.numel()))
        pick = flat[torch.randperm(flat.numel(), generator=generator)[:n_hide]]
        hidden[i].view(-1)[pick] = True
    return hidden, rec


def masked_loss(model: MaskedFieldTransformer, batch: RecordBatch, hidden: torch.Tensor,
                record_mask: torch.Tensor) -> torch.Tensor:
    """Mean cross-entropy over hidden cells (the masked-cell loss)."""
    logits = model(batch.codes, batch.status, hidden, batch.times, record_mask, batch.stage)
    total = torch.zeros((), dtype=torch.float32)
    count = 0
    for j, lg in enumerate(logits):
        sel = hidden[:, :, j]                                                # [B, L]
        if not bool(sel.any()):
            continue
        total = total + nn.functional.cross_entropy(lg[sel].float(), batch.codes[:, :, j][sel], reduction="sum")
        count += int(sel.sum())
    return total / max(count, 1)


# ============================================================================== generation
@torch.no_grad()
def maskgit_fill(model: MaskedFieldTransformer, batch: RecordBatch, target: torch.Tensor, *, steps: int,
                 choice_temperature: float, generator: torch.Generator,
                 record_mask: torch.Tensor | None = None) -> torch.Tensor:
    """Fill `target` cells (bool [B, L, C], contributing cells only) by iterative unmasking. Returns codes."""
    codes = batch.codes.clone()
    hidden = target.clone()
    rec = batch.record_mask if record_mask is None else record_mask
    b, ell, c = codes.shape
    n_total = target.flatten(1).sum(1)                                       # [B] cells to generate
    for t in range(steps):
        logits = model(codes, batch.status, hidden, batch.times, rec, batch.stage)
        sampled = torch.zeros_like(codes)
        conf = torch.full((b, ell, c), float("inf"))
        for j, lg in enumerate(logits):
            lg = lg.float().masked_fill(~model.allowed(j).view(1, 1, -1), float("-inf"))
            logp = torch.log_softmax(lg, dim=-1)                             # [B, L, K]
            draw = torch.multinomial(logp.exp().view(-1, lg.shape[-1]), 1, generator=generator).view(b, ell)
            sampled[:, :, j] = draw
            conf[:, :, j] = torch.where(hidden[:, :, j], logp.gather(-1, draw.unsqueeze(-1)).squeeze(-1),
                                        torch.full((b, ell), float("inf")))
        # annealed Gumbel noise on confidences (more exploration early, none at the last step)
        temp = choice_temperature * (1.0 - (t + 1) / steps)
        u = torch.rand(conf.shape, generator=generator).clamp(1e-20, 1.0)
        conf = torch.where(hidden, conf + temp * (-torch.log(-torch.log(u))), conf)
        # number of cells that stay hidden after this step: ⌊γ((t+1)/T)·N⌋, and always some progress
        n_cur = hidden.flatten(1).sum(1)
        n_keep = torch.floor(cosine_gamma(torch.tensor((t + 1) / steps)) * n_total).long()
        n_keep = torch.minimum(n_keep, (n_cur - 1).clamp_min(0))
        flat = conf.flatten(1)                                               # [B, L·C]
        rank = flat.argsort(dim=1).argsort(dim=1)                            # 0 = least confident
        still = (rank < n_keep.unsqueeze(1)).view(b, ell, c) & hidden
        newly = hidden & ~still
        codes = torch.where(newly, sampled, codes)
        hidden = still
        if not bool(hidden.any()):
            break
    return codes


@torch.no_grad()
def autoregressive_fill(model: MaskedFieldTransformer, batch: RecordBatch, target: torch.Tensor, *, steps: int,
                        choice_temperature: float, generator: torch.Generator) -> torch.Tensor:
    """Left-to-right over records: record ℓ is filled seeing only records ≤ ℓ (AS-364)."""
    codes = batch.codes.clone()
    ell = codes.shape[1]
    for i in range(ell):
        tgt = torch.zeros_like(target)
        tgt[:, i] = target[:, i]
        if not bool(tgt.any()):
            continue
        rec = batch.record_mask & (torch.arange(ell) <= i).view(1, -1)      # no look-ahead
        step_batch = RecordBatch(codes, batch.status, batch.times, batch.record_mask, batch.stage, batch.rows)
        codes = maskgit_fill(model, step_batch, tgt, steps=steps, choice_temperature=choice_temperature,
                             generator=generator, record_mask=rec)
    return codes


def build_masked_model(cfg: GeneratorConfig, codec: FieldCodec) -> MaskedFieldTransformer:
    """The family's model for a fitted codec (all columns; unmodelled ones are never hidden)."""
    n_classes = [codec.n_classes(j) if (j in codec.numeric or j in codec.discrete) else 1 for j in range(len(codec.columns))]
    masks = [codec.class_mask(j) if (j in codec.numeric or j in codec.discrete) else np.ones(1, bool)
             for j in range(len(codec.columns))]
    return MaskedFieldTransformer(cfg, n_classes, masks)


def modelled_columns(codec: FieldCodec) -> torch.Tensor:
    """bool [C]: columns the codec has training data for."""
    return torch.tensor([codec.modelled(j) for j in range(len(codec.columns))], dtype=torch.bool)
