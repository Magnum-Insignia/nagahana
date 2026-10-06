"""Learned Generator families as variant producers, and their training loops (training split only).

Purpose
-------
Wraps the masked-generative / autoregressive model (`masked.py`) and the diffusion model
(`diffusion.py`) so that, like a deterministic transform, each turns one real window + labels into a
candidate variant (`TransformResult`). The pipeline then applies the same acceptance gate to all.

How a learned variant is made (AS-366)
--------------------------------------
1. Encode the window with the codec (fitted on the training split, AS-367).
2. Choose target cells: in every row, each contributing *modelled* cell is a target with probability
   `regen_fraction` (at least one per row that has candidates). Diffusion targets numeric cells only.
   Excluded cells are never targets, so no status changes and no value appears in an excluded cell (D-41).
3. Generate the targets conditioned on everything else in the record (and, for the masked model, the
   other records of the sequence) *and on the record's stage label*.
4. Decode, then `project_hard_limits` moves only the generated cells into the physical boundary
   (limits.py, AS-359). Kept real cells are never edited.
5. Labels are **copied** from the source rows (`label_mode = "copied"`). Why that is defensible: the
   generation is conditioned on the label, regenerates only a share of the cells (the rest of the act
   stays real), and the variant must still pass the physics gate and — when TAAFT's marginal energy is
   supplied — the real-data energy range. It is *not* label-preserving by construction; the report says
   which variants were learned so evaluation (label preservation, TSTR; ARCH §7) can audit them.

Training (`fit_masked`, `fit_diffusion`): require RunMode.TRAIN and refuse any sample that is not a
REAL training-split sample (P-23 via AS-367). The fitted model records the sample IDs it saw
(`trained_on`), for `pipeline/splits.validate(generator_training_ids=...)`.

Decisions: D-40, D-41, D-23. Assumptions: AS-27, AS-366, AS-367.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch

from nagahana.core.errors import InvariantViolation
from nagahana.core.modes import RunMode, require_mode
from nagahana.datamodel.columnar import ColumnarUpdates
from nagahana.models.config.components import GeneratorConfig
from nagahana.models.generator.assumptions import use
from nagahana.models.generator.codec import FieldCodec
from nagahana.models.generator.diffusion import TabularDiffusion
from nagahana.models.generator.limits import PhysicalSetting, project_hard_limits
from nagahana.models.generator.masked import (
    MaskedFieldTransformer,
    autoregressive_fill,
    masked_loss,
    maskgit_fill,
    modelled_columns,
    records_from_matrices,
    training_mask,
)
from nagahana.models.generator.variants import TransformResult, UpdateLabels, changed_cells, derive
from nagahana.pipeline.splits import Origin, Sample, Split


# ============================================================================== source checks
def check_source(sample: Sample) -> None:
    """A Generator input must be a REAL training-split sample (AS-367; zero-shot always refused, D-23)."""
    if sample.split is Split.ZERO_SHOT:
        raise InvariantViolation(f"{sample.id}: zero-shot samples never feed the Generator (D-23)")
    use("AS-367", by=__name__)
    if sample.origin is not Origin.REAL:
        raise InvariantViolation(f"{sample.id}: variants are made from real samples only (no variants of variants)")
    if sample.split is not Split.TRAIN:
        raise InvariantViolation(f"{sample.id}: the Generator uses the training split only (P-23, assumed in AS-367)")


def _target_cells(cand: np.ndarray, fraction: float, rng: np.random.Generator) -> np.ndarray:
    # Each candidate cell with probability `fraction`; rows with candidates get at least one target.
    tgt = cand & (rng.random(cand.shape) < fraction)
    need = cand.any(1) & ~tgt.any(1)
    for i in np.flatnonzero(need):
        tgt[i, rng.choice(np.flatnonzero(cand[i]))] = True
    return tgt


def _torch_gen(rng: np.random.Generator) -> torch.Generator:
    return torch.Generator().manual_seed(int(rng.integers(0, 2**62)))


# ============================================================================== producers
class MaskedProducer:
    """Masked-generative (MaskGIT order) or autoregressive (left-to-right) variant producer."""

    def __init__(self, model: MaskedFieldTransformer, codec: FieldCodec, cfg: GeneratorConfig, setting: PhysicalSetting,
                 *, order: str) -> None:
        if order not in ("maskgit", "left-to-right"):
            raise ValueError("order must be 'maskgit' or 'left-to-right'")
        self.model, self.codec, self.cfg, self.setting, self.order = model, codec, cfg, setting, order
        self.name = "masked-generative" if order == "maskgit" else "autoregressive"

    def apply(self, cu: ColumnarUpdates, labels: UpdateLabels, rng: np.random.Generator) -> TransformResult:
        use("AS-366", by=__name__)
        self.codec.check_columns(cu.columns)
        codes = self.codec.encode(cu.values, cu.status)                          # [U, C]
        contrib = cu.contributing_cells()
        modelled = modelled_columns(self.codec).numpy()
        target = _target_cells((codes >= 0) & contrib & modelled[None, :], self.cfg.regen_fraction, rng)
        times = cu.updates["event_time"].to_numpy().astype(np.float64)
        batch = records_from_matrices(codes, cu.status.astype(np.int64), times, labels.stage, self.cfg.max_records)
        ell = self.cfg.max_records
        # target cells in the batch layout: row r ↦ (r // L, r % L)
        tgt = torch.zeros(batch.codes.shape, dtype=torch.bool)
        rows = np.arange(len(cu))
        tgt[torch.from_numpy(rows // ell), torch.from_numpy(rows % ell)] = torch.from_numpy(target)
        gen = _torch_gen(rng)
        self.model.eval()
        fill = maskgit_fill if self.order == "maskgit" else autoregressive_fill
        filled = fill(self.model, batch, tgt, steps=self.cfg.unmask_steps,
                      choice_temperature=self.cfg.choice_temperature, generator=gen)
        new_codes = filled[torch.from_numpy(rows // ell), torch.from_numpy(rows % ell)].numpy()   # [U, C]
        decoded = self.codec.decode(np.where(target, new_codes, -1), rng)
        values = np.where(target, decoded, cu.values)
        use("AS-359", by=__name__)
        values = project_hard_limits(values, contrib, cu.columns, self.setting, target)
        return _learned_result(cu, labels, values, target, self.name, {"regenerated": int(target.sum()), "order": self.order})


class DiffusionProducer:
    """TabDDPM-style producer: regenerates numeric cells given the rest of the record."""

    def __init__(self, model: TabularDiffusion, cfg: GeneratorConfig, setting: PhysicalSetting) -> None:
        self.model, self.cfg, self.setting = model, cfg, setting
        self.name = "diffusion"

    def apply(self, cu: ColumnarUpdates, labels: UpdateLabels, rng: np.random.Generator) -> TransformResult:
        use("AS-366", by=__name__)
        codec = self.model.codec
        codec.check_columns(cu.columns)
        z0, contrib_n = codec.to_std(cu.values, cu.status)                       # [U, Cn]
        modelled = self.model.modelled.numpy()
        target_n = _target_cells(contrib_n & modelled[None, :], self.cfg.regen_fraction, rng)
        disc = codec.encode(cu.values, cu.status)[:, codec.discrete_columns]       # [U, Cd]
        self.model.eval()
        z = self.model.sample(torch.from_numpy(z0), torch.from_numpy(contrib_n), torch.from_numpy(target_n),
                              torch.from_numpy(disc), torch.from_numpy(labels.stage), _torch_gen(rng)).numpy()
        new_numeric = codec.from_std(z)                                          # [U, Cn]
        values = cu.values.copy()
        target = np.zeros(cu.values.shape, dtype=bool)
        for k, j in enumerate(codec.numeric_columns):
            values[target_n[:, k], j] = new_numeric[target_n[:, k], k]
            target[:, j] = target_n[:, k]
        use("AS-359", by=__name__)
        values = project_hard_limits(values, cu.contributing_cells(), cu.columns, self.setting, target)
        return _learned_result(cu, labels, values, target, self.name, {"regenerated": int(target.sum())})


def _learned_result(cu: ColumnarUpdates, labels: UpdateLabels, values: np.ndarray, target: np.ndarray, name: str,
                    params: dict[str, object]) -> TransformResult:
    # Same rows, same statuses; generated values only in target cells; labels copied (AS-366).
    rows = np.arange(len(cu), dtype=np.int64)
    out = derive(cu, rows, values=values, status=cu.status)
    return TransformResult(updates=out, labels=labels.take(rows), source_rows=rows,
                           changed=changed_cells(cu, rows, out.values, out.status), producer=name, label_mode="copied",
                           params=dict(params), free=target)


# ============================================================================== training
@dataclass
class FitReport:
    """What a fit saw and how its loss moved (first / last logged losses)."""

    trained_on: tuple[str, ...]
    losses: list[float]


def _check_training_set(samples: Sequence[Sample]) -> tuple[str, ...]:
    for s in samples:
        check_source(s)
    return tuple(s.id for s in samples)


def fit_masked(model: MaskedFieldTransformer, codec: FieldCodec, windows: Sequence[tuple[ColumnarUpdates, UpdateLabels, Sample]],
               cfg: GeneratorConfig, *, steps: int, lr: float, seed: int) -> FitReport:
    """Train the masked-generative / autoregressive model on real training windows (AdamW)."""
    require_mode(RunMode.TRAIN, component="Generator")
    ids = _check_training_set([s for _, _, s in windows])
    use("AS-363", by=__name__)
    use("AS-364", by=__name__)
    gen = torch.Generator().manual_seed(seed)
    batches = []
    for cu, labels, _ in windows:
        codec.check_columns(cu.columns)
        codes = codec.encode(cu.values, cu.status)
        times = cu.updates["event_time"].to_numpy().astype(np.float64)
        batches.append(records_from_matrices(codes, cu.status.astype(np.int64), times, labels.stage, cfg.max_records))
    modelled = modelled_columns(codec)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    losses: list[float] = []
    for step in range(steps):
        batch = batches[step % len(batches)]
        hidden, rec = training_mask(batch, modelled, ar_fraction=cfg.ar_fraction, generator=gen)
        loss = masked_loss(model, batch, hidden, rec)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    return FitReport(ids, losses)


def fit_diffusion(model: TabularDiffusion, windows: Sequence[tuple[ColumnarUpdates, UpdateLabels, Sample]], *,
                  steps: int, lr: float, seed: int) -> FitReport:
    """Train the diffusion denoiser on real training windows (AdamW, L_simple)."""
    require_mode(RunMode.TRAIN, component="Generator")
    ids = _check_training_set([s for _, _, s in windows])
    use("AS-365", by=__name__)
    codec = model.codec
    gen = torch.Generator().manual_seed(seed)
    rows = []
    for cu, labels, _ in windows:
        codec.check_columns(cu.columns)
        z0, contrib = codec.to_std(cu.values, cu.status)
        disc = codec.encode(cu.values, cu.status)[:, codec.discrete_columns]
        rows.append((torch.from_numpy(z0), torch.from_numpy(contrib), torch.from_numpy(disc), torch.from_numpy(labels.stage)))
    opt = torch.optim.AdamW(model.denoiser.parameters(), lr=lr)
    model.train()
    losses: list[float] = []
    for step in range(steps):
        z_t, c_t, d_t, s_t = rows[step % len(rows)]
        loss = model.loss(z_t, c_t, d_t, s_t, gen)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    return FitReport(ids, losses)
