"""The training pipeline's own engineering assumptions AS-570 ... AS-599, and how code declares their use.

Why this module exists
----------------------
`governance/assumptions.py` holds the centrally registered assumptions. The training pipeline needs
finer-grained ones (schedules, accumulation, checkpoint format, distributed data, augmentation
targets). Until they are registered centrally, `use()` behaves exactly like the Generator's
`models/generator/assumptions.use`:

1. if the ID is registered centrally, `assume(id)` is called directly (the use is recorded; strict
   mode raises through the central registry);
2. otherwise the umbrella assumption it refines is assumed (strict mode therefore still blocks it,
   and the central use report shows the pipeline relying on the umbrella), and the fine-grained use
   is recorded locally (`local_uses()`).

The human-readable entries (what is assumed, what it stands for, reasoning, evidence) are in
`docs/assumptions/training-pipeline.md`; the one-line summaries below must match that document.
Generator-side assumptions of the same ID range (AS-584 ... AS-586) live in
`models/generator/assumptions.py`, because the code that relies on them is the Generator's.
"""

from __future__ import annotations

from collections import defaultdict

from nagahana.governance import assumptions as _central

#: ID -> (registered umbrella assumption it refines, one-line summary). Full text: docs/assumptions/training-pipeline.md.
TRAINING_ASSUMPTIONS: dict[str, tuple[str, str]] = {
    "AS-570": ("AS-406", "learning rate: warmup-stable-decay (1 - sqrt decay over the last 20 %), decayed branches; cosine an option"),
    "AS-571": ("AS-406", "AdamW member: betas (0.9, 0.95), eps 1e-8, weight decay on its matrices only"),
    "AS-572": ("AS-406", "accumulation: each micro-batch loss / n; one global-norm clip per step; non-finite steps skipped in sync"),
    "AS-573": ("AS-406", "weight EMA is built for every stage and off by default (no design document specifies it)"),
    "AS-574": ("AS-406", "early stopping on the stage objective over validation segments, eval mode, fixed draws, run-time budgets"),
    "AS-575": ("AS-367", "validation and test read real windows only; the Generator never sees them (P-23 via AS-367)"),
    "AS-576": ("AS-406", "checkpoints: tensor-only file + JSON manifest, SHA-256 verified before loading, full state on rank 0"),
    "AS-577": ("AS-333", "ranks: disjoint segments, a shared generator for R and S, epoch ends with the first rank, same world size"),
    "AS-578": ("AS-406", "activation recomputation of TSTCT/TAAFT blocks (non-reentrant checkpoint), TAAFT on by default at L"),
    "AS-579": ("AS-406", "ablations: switch realisations (head masks, config lenses and planes, data-side planes) and three tiers"),
    "AS-580": ("AS-412", "stages 2 and 3 accumulate trigger-bearing micro-batches; other batches only write the carry"),
    "AS-581": ("AS-406", "QK-Clip on the QK-norm gains and null keys with tau = 100, from Cauchy-Schwarz bounds of tracked norms"),
    "AS-582": ("AS-325", "augmentation target: class balance 1/2-1/2, families uniform within attacks; water-filling budget"),
    "AS-583": ("AS-361", "energy acceptance runs from stage 3 (TAAFT pretrained); stage-1/2 variants pass limits and the gate only"),
    "AS-587": ("AS-39", "precision: bf16 autocast (default), fp32, or fp16 autocast with dynamic loss scaling; weights fp32"),
    "AS-588": ("AS-406", "seeds: per-purpose streams derived from (seed, keys); CPU generators; draws moved to the device"),
    "AS-589": ("AS-34", "prep analytics and cleaning: coverage, distributions, labels, windows, hashes; duplicates dropped; no physics on telemetry"),
    "AS-590": ("AS-326", "pretraining (stages 1, 2) splits 70/30 inside the full training split; test, val, zero-shot unseen"),
    "AS-591": ("AS-18", "evaluation outputs: usable triggers, event step from the infiltration timeline, observed steps"),
    "AS-592": ("AS-27", "stages 1, 2 and 3 mix accepted variants of training segments (D-23); stage 4 reads real data only"),
    "AS-593": ("AS-413", "frozen modules: optimizer membership checked once, checksum every verify_every steps, SHA-256 at ends"),
    "AS-594": ("AS-27", "learned Generator families are fitted in prep on training segments only, before any variant is drawn"),
    "AS-595": ("AS-25", "feedback learners step through the human gate with the stage schedule and clipping; grads averaged over ranks"),
    "AS-596": ("AS-27", "the augmentation plan (source, producer, seed, digest) is stored; later stages replay and verify it"),
    "AS-597": ("AS-22", "the STAGED stop-gradient phase covers the first half of the stage-3 optimiser steps"),
    "AS-598": ("AS-26", "site adapters: lr 1e-4, 100 warm-up steps, 2,000 steps, the stage-1/2 objectives plus alerts"),
    "AS-599": ("AS-406", "Muon group rule: hidden 2-D matrices (and CVG-AE's stacked maps) to Muon; the rest to AdamW"),
}

_LOCAL_USES: dict[str, set[str]] = defaultdict(set)


def use(key: str, *, by: str) -> str:
    """Declare that code relies on training assumption `key`; returns the ID.

    Raises `KeyError` for unknown IDs. Strict mode raises through the central registry (the umbrella
    or the registered ID itself).
    """
    if key not in TRAINING_ASSUMPTIONS:
        raise KeyError(f"unknown training assumption {key!r}; known: {', '.join(TRAINING_ASSUMPTIONS)}")
    try:
        _central.get(key)
        registered = True
    except KeyError:
        registered = False
    # Registered centrally: use it directly; otherwise use the umbrella it refines.
    _central.assume(key if registered else TRAINING_ASSUMPTIONS[key][0], by=by)
    _LOCAL_USES[key].add(by)
    return key


def local_uses() -> dict[str, frozenset[str]]:
    """Which modules used which training assumptions in this process."""
    return {k: frozenset(v) for k, v in _LOCAL_USES.items()}


__all__ = ["TRAINING_ASSUMPTIONS", "local_uses", "use"]
