"""Torch dataset and batch iterator: (WindowBatch, LabelBatch) from event logs + labels.

Purpose
-------
Glue between the split manifest (`data/sampling.py`), the window builder (`data/windows.py`) and the
collation (`data/collate.py`):

    sources ──prepare_source──► PreparedSource ──index_windows──► WindowRecord ──assign_splits──► manifest
    WindowDataset(records of one split)[i] ──build_window──► WindowItem
    DataLoader(dataset, sampler=ClassBalancedSampler, collate_fn=WindowCollator) ──► (WindowBatch, LabelBatch)

Owner sources, decisions, assumptions: build-spec §3 (what training needs), §4b.5 (out-of-order
augmentation, train only), §4b.6 (class balance), §4b.7 (splits); AS-320 (jitter), AS-325 (balance).

Invariants
----------
- Only windows of the requested roles are served (a test or zero-shot window can never reach a
  training loader built from train records).
- The out-of-order augmentation is applied only when `perturb_seed` is given, and it is seeded per
  (seed, epoch, index) so that a run is reproducible. It must be off for evaluation splits.
- Novelty marks of zero-shot windows travel to `LabelBatch.novelty`, so known and novel families are
  scored separately (D-23).

Extension points: any `structure_fn` / `collate_fn` with the graph builder's signatures (tests pass a
stub); workers > 0 work because items are plain NumPy plus the structure's tensors.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from typing import Any

import numpy as np
from torch.utils.data import DataLoader, Dataset, Sampler

from nagahana.core.errors import InvariantViolation
from nagahana.models.batch import LabelBatch, WindowBatch
from nagahana.models.config import NagaHanaConfig

from .collate import collate_items
from .sampling import ClassBalancedSampler, Role, SplitManifest, WindowRecord, label_limits, records_for
from .windows import PreparedSource, StructureFn, WindowItem, build_window


class WindowDataset(Dataset[WindowItem]):
    """Windows of the given records, built on demand.

    sources: the prepared sources the records index into. records: the windows served (one split).
    novelty: window id → "known" | "novel" (zero-shot only). perturb_seed: enables the §4b.5
    out-of-order augmentation (training only).
    """

    def __init__(
        self,
        sources: Sequence[PreparedSource],
        records: Sequence[WindowRecord],
        cfg: NagaHanaConfig,
        *,
        novelty: dict[str, str] | None = None,
        structure_fn: StructureFn | None = None,
        perturb_seed: int | None = None,
        label_limits: dict[str, float] | None = None,
    ) -> None:
        self.sources = list(sources)
        self.records = list(records)
        self.cfg = cfg
        self.novelty = dict(novelty or {})
        self.structure_fn = structure_fn
        self.perturb_seed = perturb_seed
        self.label_limits = dict(label_limits or {})
        self.epoch = 0

    @classmethod
    def from_manifest(
        cls, sources: Sequence[PreparedSource], manifest: SplitManifest, roles: Sequence[Role], cfg: NagaHanaConfig,
        **kw: Any,
    ) -> WindowDataset:
        """The windows of `roles` in `manifest` (validated first)."""
        manifest.validate()
        if kw.get("perturb_seed") is not None and set(roles) - {Role.TRAIN}:
            raise InvariantViolation("the out-of-order augmentation is for training windows only (§4b.5)")
        novelty = {i: n.value for i, n in manifest.novelty.items()}
        return cls(sources, records_for(manifest, roles), cfg, novelty=novelty, label_limits=label_limits(manifest), **kw)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, i: int) -> WindowItem:
        r = self.records[i]
        rng = None
        if self.perturb_seed is not None:
            rng = np.random.default_rng([self.perturb_seed, self.epoch, i])
        item = build_window(self.sources[r.source], r.start, r.stop, self.cfg, structure_fn=self.structure_fn,
                            perturb_rng=rng, label_limit=self.label_limits.get(r.id, math.inf))
        item.extra["window_id"] = r.id
        item.extra["novelty"] = self.novelty.get(r.id, "")
        return item


class WindowCollator:
    """`collate_fn` for a DataLoader: list of `WindowItem` → (WindowBatch, LabelBatch)."""

    def __init__(self, cfg: NagaHanaConfig, *, collate_fn: Any = None) -> None:
        self.cfg = cfg
        self.collate_fn = collate_fn

    def __call__(self, items: list[WindowItem]) -> tuple[WindowBatch, LabelBatch]:
        return collate_items(items, self.cfg, novelty=[str(it.extra.get("novelty", "")) for it in items],
                             collate_fn=self.collate_fn)


def make_loader(
    dataset: WindowDataset,
    *,
    sampler: Sampler[int] | None = None,
    balanced_samples: int | None = None,
    seed: int = 0,
    num_workers: int = 0,
    collate_fn: Any = None,
) -> DataLoader[WindowItem]:
    """A DataLoader of (WindowBatch, LabelBatch) with `TrainingConfig.batch_windows` windows per batch.

    `balanced_samples` builds a `ClassBalancedSampler` over the dataset's records (§4b.6); otherwise
    `sampler` (or sequential order) is used.
    """
    if sampler is None and balanced_samples is not None:
        sampler = ClassBalancedSampler(dataset.records, num_samples=balanced_samples, seed=seed)
    return DataLoader(dataset, batch_size=dataset.cfg.training.batch_windows, sampler=sampler, shuffle=False,
                      num_workers=num_workers, collate_fn=WindowCollator(dataset.cfg, collate_fn=collate_fn))


def iterate_batches(loader: DataLoader[WindowItem]) -> Iterator[tuple[WindowBatch, LabelBatch]]:
    """Typed iteration over a loader made by `make_loader`."""
    yield from loader


__all__ = ["WindowCollator", "WindowDataset", "iterate_batches", "make_loader"]
