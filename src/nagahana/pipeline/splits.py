"""Data splits and their invariants: one split vocabulary for the whole code base (D-23; AS-367, AS-575).

Decided [I-01] (D-23)
---------------------
- Training uses real and generated data.
- Zero-shot uses real data only.
- Novel and known attacks are evaluated separately. By definition a novel family never appears in
  training, test or validation.

D-23 also allows generated data in validation. In this build validation is real (build-spec section
4b.7: "20 % validation and zero-shot (real + unseen datasets)"): under the no-leakage rule below a
variant can only derive from a training sample, so a generated validation sample would carry
training data into model selection (AS-575).

The split vocabulary (`Split`)
------------------------------
    TRAIN       training (real segments and Generator variants of real training segments)
    TEST        real held-out evaluation of known families (ai-mod-arch section 7d: 20 %)
    VAL         real validation, used for early stopping and model selection
    ZERO_SHOT   real windows of novel families or held-out networks (AS-35), marked known or novel
    EXCLUDED    purged at a cut between two splits (AS-327); never read by any stage

`data/sampling.Role` is this enum (the data pipeline's manifest roles and the pipeline's sample splits
are one vocabulary). EXCLUDED rows are not samples: `validate` skips them.

The Generator's no-leakage rule (always checked)
------------------------------------------------
P-23 (proposal) is assumed by this build in AS-367: the Generator reads real training-split samples
only. Its check is a safety property, so `validate` runs it unconditionally; nothing can switch it
off. Every generated sample must derive from a real TRAIN sample, every sample the Generator was fitted
on must be a TRAIN sample, and generated samples never sit outside TRAIN (validation and test read
real windows only, AS-575). This prevents the data-snooping pitfall of Arp et al., "Dos and Don'ts of
Machine Learning in Computer Security", USENIX Security 2022.

A manifest that breaks a rule is rejected (`InvariantViolation`) before any training reads it.
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
    """Which split a sample (or a planned window or segment) belongs to (module docstring)."""

    TRAIN = "train"
    TEST = "test"
    VAL = "val"
    ZERO_SHOT = "zero_shot"
    EXCLUDED = "excluded"


#: The splits whose families count as "seen" when zero-shot novelty is checked.
SEEN_SPLITS: frozenset[Split] = frozenset({Split.TRAIN, Split.VAL, Split.TEST})


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
    """Raise `InvariantViolation` on the first broken rule (module docstring).

    enabled_proposals: the run's enabled proposals. They do not switch any check on or off (the leakage
    check always runs); unknown IDs raise `KeyError`, so a misspelt proposal is caught here.
    generator_training_ids: the samples the Generator's learned families were fitted on.
    """
    for p in enabled_proposals:
        decisions.get(p)
    rows = [r for r in manifest if r.split is not Split.EXCLUDED]
    by_id = {r.id: r for r in rows}
    if len(by_id) != len(rows):
        raise InvariantViolation("duplicate sample ids in manifest")
    seen_families = {r.family for r in rows if r.split in SEEN_SPLITS}
    for r in rows:
        if r.split is Split.TEST and r.origin is not Origin.REAL:
            raise InvariantViolation(f"{r.id}: test data must be real (ai-mod-arch section 7d)")
        if r.split is Split.VAL and r.origin is not Origin.REAL:
            raise InvariantViolation(f"{r.id}: validation data must be real (build-spec section 4b.7; AS-575)")
        if r.split is Split.ZERO_SHOT:
            if r.origin is not Origin.REAL:
                raise InvariantViolation(f"{r.id}: zero-shot data must be real (D-23)")
            if r.novelty is None:
                raise InvariantViolation(f"{r.id}: zero-shot samples must be marked known or novel")
            if r.novelty is Novelty.NOVEL and r.family in seen_families:
                raise InvariantViolation(f"{r.id}: family {r.family!r} is marked novel but appears in train/val/test")
            if r.novelty is Novelty.KNOWN and r.family not in seen_families:
                raise InvariantViolation(f"{r.id}: family {r.family!r} is marked known but never seen in train/val/test")
        elif r.novelty is not None:
            raise InvariantViolation(f"{r.id}: only zero-shot samples carry a novelty mark")
        if r.origin is Origin.GENERATED and r.derived_from is None:
            raise InvariantViolation(f"{r.id}: generated samples must record the real sample they derive from")
    # The Generator's no-leakage rule (P-23 via AS-367): unconditional.
    for r in rows:
        if r.origin is Origin.GENERATED:
            src = by_id.get(r.derived_from or "")
            if src is None or src.split is not Split.TRAIN or src.origin is not Origin.REAL:
                raise InvariantViolation(f"{r.id}: generated from {r.derived_from!r}, which is not a real training sample (P-23)")
            if r.split is not Split.TRAIN:
                raise InvariantViolation(f"{r.id}: generated samples belong to the training split only (P-23, AS-575)")
    for gid in generator_training_ids:
        src = by_id.get(gid)
        if src is None or src.split is not Split.TRAIN or src.origin is not Origin.REAL:
            raise InvariantViolation(f"Generator trained on {gid!r}, which is not a real training sample (P-23)")


__all__ = ["Novelty", "Origin", "SEEN_SPLITS", "Sample", "Split", "validate"]
