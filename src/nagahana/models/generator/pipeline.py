"""VariantPipeline: real training windows → accepted, provenance-tagged variants (training only).

Purpose
-------
The Generator's entry point for pipeline stage 2 ("Preparation (with the Generator)", I-01): for each
real training window it draws candidate variants from the configured producers (deterministic
transforms and learned families), rejects what is physically impossible, unchanged, or
energy-atypical, and returns the accepted variants with labels and a manifest row each.

Rules enforced here
-------------------
- **Training only** (D-40): `generate` requires RunMode.TRAIN (core/modes.py), like every Generator path.
- **Sources** (AS-367 standing for P-23; D-23): only REAL, TRAIN-split samples; zero-shot refused always.
- **Provenance** (D-23, P-23): each variant is a `Sample(origin=GENERATED, split=TRAIN, derived_from=<real
  id>)`, so `pipeline/splits.validate` can prove that zero-shot never contains a variant and (with P-23
  enabled) that every variant derives from a real training sample. Variant tables carry
  `origin`/`derived_from_seq` columns and zero raw hashes (AS-369).
- **Labels:** deterministic transforms keep labels by construction; learned families copy them (AS-366).
  A variant of an attack window that loses every malicious update is rejected (AS-368): its window
  family would no longer be true.
- **Class imbalance** (build-spec §4b.6): `generate_many` gives attack windows `attack_share` of the
  variant budget (AS-370), so most variants come from attack events.
- **Acceptance** (acceptance.py): hard limits always; Φ_phys ≤ τ when P-11 is enabled or AS-28 assumed;
  JEM-style energy range when an energy callable is configured. Plus "duplicate" (no change at all).

Budget maths (AS-370)
---------------------
With A attack windows, B benign windows and total budget n:
    n_A = round(attack_share · n) if A > 0 and B > 0;  n_A = n if B = 0;  n_A = 0 if A = 0;  n_B = n − n_A
and each class's budget is split over its windows by the largest-remainder method (Hamilton), so the
per-window counts differ by at most one and sum exactly to the class budget.

Determinism: the RNG of a window is seeded by (pipeline seed, SHA-256 of the source sample id), so the
same configuration reproduces the same variants regardless of the order windows are processed in.

Decisions: D-23, D-40, D-41. Proposals: P-11, P-23. Assumptions: AS-27, AS-28, AS-366 … AS-370.
Extension point: any object with `name` and `apply(cu, labels, rng) -> TransformResult` is a producer.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from nagahana.core.errors import InvariantViolation
from nagahana.core.modes import RunMode, require_mode
from nagahana.datamodel.columnar import ColumnarUpdates
from nagahana.models.config.components import GeneratorConfig
from nagahana.models.generator.acceptance import AcceptanceGate, AcceptanceReport
from nagahana.models.generator.assumptions import use
from nagahana.models.generator.learned import check_source
from nagahana.models.generator.transforms import NotApplicable
from nagahana.models.generator.variants import GENERATOR_ADAPTER, TransformResult, UpdateLabels
from nagahana.pipeline.splits import Origin, Sample, Split


class Producer(Protocol):
    """Anything that makes one candidate variant of a window (transforms and learned families)."""

    name: str

    def apply(self, cu: ColumnarUpdates, labels: UpdateLabels, rng: np.random.Generator) -> TransformResult: ...


@dataclass
class Variant:
    """One accepted variant."""

    id: str
    updates: ColumnarUpdates
    labels: UpdateLabels
    sample: Sample
    producer: str
    label_mode: str
    source_rows: np.ndarray
    params: dict[str, Any]
    report: AcceptanceReport


@dataclass
class Rejection:
    """One rejected candidate and why."""

    source: str
    producer: str
    reasons: tuple[str, ...]
    report: AcceptanceReport | None = None


@dataclass
class VariantBatch:
    """Result of a generation run: accepted variants, rejections, and the requested count."""

    accepted: list[Variant] = field(default_factory=list)
    rejected: list[Rejection] = field(default_factory=list)
    requested: int = 0
    notes: list[str] = field(default_factory=list)

    def manifest(self) -> list[Sample]:
        """Manifest rows of the accepted variants (for pipeline/splits.validate)."""
        return [v.sample for v in self.accepted]

    def rejection_counts(self) -> dict[str, int]:
        """Rejection reasons → counts (an honest acceptance profile)."""
        return dict(Counter(r for rej in self.rejected for r in rej.reasons))

    def extend(self, other: VariantBatch) -> None:
        self.accepted += other.accepted
        self.rejected += other.rejected
        self.requested += other.requested
        self.notes += other.notes


def allocate_budget(is_attack: Sequence[bool], total: int, attack_share: float) -> tuple[np.ndarray, list[str]]:
    """Variants per window (AS-370). Returns (counts int64 [W], notes)."""
    use("AS-370", by=__name__)
    if total < 0 or not 0.0 <= attack_share <= 1.0:
        raise InvariantViolation("need total ≥ 0 and attack_share ∈ [0, 1]")
    att = np.asarray(is_attack, dtype=bool)
    notes: list[str] = []
    n_a_win, n_b_win = int(att.sum()), int((~att).sum())
    if n_a_win and n_b_win:
        n_att = int(round(attack_share * total))
    elif n_a_win:
        n_att = total
        notes.append("no benign windows: the whole budget went to attack windows")
    else:
        n_att = 0
        if total:
            notes.append("no attack windows: attack_share could not be honoured; budget went to benign windows")
    out = np.zeros(len(att), dtype=np.int64)
    for mask, budget in ((att, n_att), (~att, total - n_att)):
        idx = np.flatnonzero(mask)
        if len(idx) == 0:
            continue
        quota = np.full(len(idx), budget / len(idx))              # largest remainder (Hamilton)
        base = np.floor(quota).astype(np.int64)
        rest = budget - int(base.sum())
        order = np.argsort(-(quota - base), kind="stable")
        base[order[:rest]] += 1
        out[idx] = base
    return out, notes


def _window_seed(seed: int, source_id: str) -> np.random.Generator:
    digest = int.from_bytes(hashlib.sha256(source_id.encode("utf-8")).digest()[:8], "big")
    return np.random.default_rng([seed, digest])


class VariantPipeline:
    """Draw, check and tag variants. See the module docstring.

    producers: the families to draw from; weights: their sampling probabilities (uniform if None).
    gate: the acceptance gate. seed: base seed for every window's RNG.
    """

    def __init__(self, cfg: GeneratorConfig, producers: Sequence[Producer], gate: AcceptanceGate, *, seed: int,
                 weights: Sequence[float] | None = None) -> None:
        if not producers:
            raise InvariantViolation("the pipeline needs at least one producer")
        w = np.ones(len(producers)) if weights is None else np.asarray(weights, dtype=np.float64)
        if w.shape != (len(producers),) or (w < 0).any() or w.sum() <= 0:
            raise InvariantViolation("weights must be non-negative, one per producer, not all zero")
        self.cfg, self.producers, self.gate, self.seed = cfg, list(producers), gate, int(seed)
        self.weights = w / w.sum()

    def generate(self, window: ColumnarUpdates, labels: UpdateLabels, source: Sample, *, n: int) -> VariantBatch:
        """Up to `n` accepted variants of one real training window (at most n · max_attempts tries)."""
        require_mode(RunMode.TRAIN, component="Generator")
        check_source(source)
        if len(labels) != len(window):
            raise InvariantViolation("labels must have one entry per update of the window")
        window.validate()
        rng = _window_seed(self.seed, source.id)
        batch = VariantBatch(requested=n)
        attempts = 0
        while len(batch.accepted) < n and attempts < n * self.cfg.max_attempts:
            attempts += 1
            producer = self.producers[int(rng.choice(len(self.producers), p=self.weights))]
            child = np.random.default_rng(rng.integers(0, 2**63))
            try:
                res = producer.apply(window, labels, child)
            except NotApplicable as e:
                batch.rejected.append(Rejection(source.id, producer.name, (f"not-applicable:{e}",)))
                continue
            # duplicates and erased attacks are rejected before the (more expensive) gate
            if res.is_duplicate():
                batch.rejected.append(Rejection(source.id, producer.name, ("duplicate",)))
                continue
            if labels.has_attack() and not res.labels.has_attack():
                use("AS-368", by=__name__)
                batch.rejected.append(Rejection(source.id, producer.name, ("attack-erased",)))
                continue
            report = self.gate.evaluate(res.updates)
            if not report.accepted:
                batch.rejected.append(Rejection(source.id, producer.name, report.reasons, report))
                continue
            batch.accepted.append(self._tag(window, source, res, report, len(batch.accepted)))
        if len(batch.accepted) < n:
            batch.notes.append(f"{source.id}: {len(batch.accepted)} of {n} variants accepted after {attempts} attempts")
        return batch

    def generate_many(self, items: Sequence[tuple[ColumnarUpdates, UpdateLabels, Sample]], *, total: int) -> VariantBatch:
        """Variants for many windows with the attack-share budget (build-spec §4b.6, AS-370)."""
        require_mode(RunMode.TRAIN, component="Generator")
        counts, notes = allocate_budget([lab.has_attack() for _, lab, _ in items], total, self.cfg.attack_share)
        out = VariantBatch(notes=list(notes))
        for (cu, lab, src), k in zip(items, counts.tolist(), strict=True):
            if k:
                out.extend(self.generate(cu, lab, src, n=k))
        return out

    @staticmethod
    def _tag(window: ColumnarUpdates, source: Sample, res: TransformResult, report: AcceptanceReport, k: int) -> Variant:
        # Provenance (AS-369): ids derive from the real sample; tables name the generator and producer.
        use("AS-369", by=__name__)
        vid = f"{source.id}~g{k:04d}"
        res.updates.source_id = f"{window.source_id}~{vid}"
        res.updates.adapter = f"{GENERATOR_ADAPTER}:{res.producer}"
        sample = Sample(id=vid, origin=Origin.GENERATED, split=Split.TRAIN, family=source.family,
                        network=source.network, derived_from=source.id)
        return Variant(vid, res.updates, res.labels, sample, res.producer, res.label_mode, res.source_rows, res.params, report)
