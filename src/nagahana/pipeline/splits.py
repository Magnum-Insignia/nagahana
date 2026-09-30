"""Data splits and their invariants (D-23 decided; D-16 and P-23 open).

Decided by the owner [I-01]
---------------------------
- Training and validation may use **real and generated** data ("learn from this").
- Zero-shot uses **real data only**.
- Novel and known attacks are evaluated **separately**. By definition, a novel family never appears in
  training or validation.

Open or proposed
----------------
- D-16: does "novel" also include unseen *networks* (leave-one-network-out, ARCH §12)?
- P-23: the Generator trains only on the real training split, so no zero-shot family leaks into
  generated data. Leakage of this kind is the data-snooping pitfall of Arp et al. (USENIX Security 2022).

`validate` checks the decided invariants always, and the proposal invariant only when P-23 is enabled.
A manifest that breaks a rule is rejected before any training reads it.
"""

from __future__ import annotations

import enum
from collections.abc import Collection, Iterable
from dataclasses import dataclass

from nagahana.core.errors import InvariantViolation
from nagahana.governance import decisions


class Origin(enum.Enum):
    """Where a sample came from."""

    REAL = "real"
    GENERATED = "generated"


class Split(enum.Enum):
    """Which split a sample belongs to."""

    TRAIN = "train"
    VAL = "val"
    ZERO_SHOT = "zero_shot"


class Novelty(enum.Enum):
    """Zero-shot samples only: is the attack family known from training?"""

    KNOWN = "known"
    NOVEL = "novel"


@dataclass(frozen=True)
class Sample:
    """One row of a split manifest.

    `family` is the attack family ("benign" for benign traffic). `derived_from` is set for generated
    samples: the real sample they vary.
    """

    id: str
    origin: Origin
    split: Split
    family: str
    network: str
    novelty: Novelty | None = None
    derived_from: str | None = None


def validate(
    manifest: Iterable[Sample],
    *,
    enabled_proposals: Collection[str] = (),
    generator_training_ids: Collection[str] = (),
) -> None:
    """Raise `InvariantViolation` on the first broken rule (see module docstring)."""
    rows = list(manifest)
    by_id = {r.id: r for r in rows}
    if len(by_id) != len(rows):
        raise InvariantViolation("duplicate sample ids in manifest")
    seen_families = {r.family for r in rows if r.split in (Split.TRAIN, Split.VAL)}
    for r in rows:
        if r.split is Split.ZERO_SHOT:
            if r.origin is not Origin.REAL:
                raise InvariantViolation(f"{r.id}: zero-shot data must be real (D-23)")
            if r.novelty is None:
                raise InvariantViolation(f"{r.id}: zero-shot samples must be marked known or novel")
            if r.novelty is Novelty.NOVEL and r.family in seen_families:
                raise InvariantViolation(f"{r.id}: family {r.family!r} is marked novel but appears in train/val")
            if r.novelty is Novelty.KNOWN and r.family not in seen_families:
                raise InvariantViolation(f"{r.id}: family {r.family!r} is marked known but never seen in train/val")
        elif r.novelty is not None:
            raise InvariantViolation(f"{r.id}: only zero-shot samples carry a novelty mark")
        if r.origin is Origin.GENERATED and r.derived_from is None:
            raise InvariantViolation(f"{r.id}: generated samples must record the real sample they derive from")

    try:
        decisions.require_proposal("generator-no-leakage", enabled_proposals)
    except Exception:
        return
    # P-23: generated samples derive from real TRAIN samples only; the Generator saw only TRAIN.
    for r in rows:
        if r.origin is Origin.GENERATED:
            src = by_id.get(r.derived_from or "")
            if src is None or src.split is not Split.TRAIN or src.origin is not Origin.REAL:
                raise InvariantViolation(f"{r.id}: generated from {r.derived_from!r}, which is not a real training sample (P-23)")
    for gid in generator_training_ids:
        src = by_id.get(gid)
        if src is None or src.split is not Split.TRAIN:
            raise InvariantViolation(f"Generator trained on {gid!r}, which is not in the training split (P-23)")
