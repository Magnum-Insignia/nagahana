"""Projection of published models' classes and stages onto the ATT&CK stage vocabulary (AS-563).

NagaHana reports stages in the vocabulary of models/vocab.py (14 ATT&CK Enterprise tactics plus "none").
A published model speaks its own vocabulary: attack classes ("DoS attacks-Hulk", "Recon-PortScan") or
stage names of its own model ("break_in", "data_exfiltration"). Two projections make its outputs
comparable with NagaHana's stage metrics:

    by name    an explicit map from the model's stage names to tactic names (the HMM forecasters);
               P(tactic = g | x) = sum over names n with map(n) = g of P(n | x)
    by labels  for a classifier over attack classes, P(tactic | class) is estimated on the training rows
               from the dataset's label table (data/labels.py, AS-34): for class c with training labels
               l, P(g | c) = #{rows of c whose label maps to g} / #{rows of c}; then
               P(tactic = g | x) = sum_c P(c | x) P(g | c)
    A class whose labels all map to one tactic projects exactly; a grouped class (for example the
    8-class "Recon" of CIC-IoT-2023) spreads its mass over its members' tactics in proportion to their
    training frequencies, which is the marginal of the label table under the training distribution.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from nagahana.core.errors import InvariantViolation


def attack_stages() -> tuple[str, ...]:
    """Stage names of models/vocab.py in code order ("none" first)."""
    from nagahana.models.vocab import STAGES

    return tuple(name for name, _ in STAGES)


def infiltration_stage_names() -> frozenset[str]:
    """Tactics that constitute an infiltration state (models/vocab.py INFILTRATION_STAGES, AS-18)."""
    from nagahana.models.vocab import INFILTRATION_STAGES, STAGES

    return frozenset(STAGES[i][0] for i in INFILTRATION_STAGES)


def name_projection(names: Sequence[str], mapping: Mapping[str, str]) -> np.ndarray:
    """[len(names), 15] 0/1 matrix sending each name to its tactic; every name must be mapped."""
    vocab = attack_stages()
    index = {s: i for i, s in enumerate(vocab)}
    out = np.zeros((len(names), len(vocab)), dtype=np.float64)
    for i, n in enumerate(names):
        if n not in mapping:
            raise InvariantViolation(f"stage {n!r} has no tactic in the map")
        target = mapping[n]
        if target not in index:
            raise InvariantViolation(f"{target!r} is not a stage of models/vocab.py: {vocab}")
        out[i, index[target]] = 1.0
    return out


def label_stage_codes(raw_labels: Sequence[object], dataset: str) -> np.ndarray:
    """Tactic code of each raw label through the dataset's label table; -1 for unknown or unmapped labels."""
    from nagahana.data.labels import MAPPERS
    from nagahana.models.vocab import STAGE_CODE

    if dataset not in MAPPERS:
        raise KeyError(f"no label table for dataset {dataset!r}; known: {sorted(MAPPERS)}")
    mapper = MAPPERS[dataset]
    cache: dict[str, int] = {}
    out = np.empty(len(raw_labels), dtype=np.int64)
    for i, raw in enumerate(raw_labels):
        key = str(raw)
        if key not in cache:
            spec = mapper(key)
            cache[key] = -1 if spec is None else STAGE_CODE.get(spec.stage, -1)
        out[i] = cache[key]
    return out


def class_stage_matrix(class_codes: np.ndarray, stage_codes: np.ndarray, n_classes: int) -> np.ndarray:
    """P(tactic | class) [n_classes, 15] from training rows (rows with an unknown class or stage are skipped).

    A class with training rows none of whose labels maps to a tactic raises: its mass could not be placed
    (the label table must be completed, AS-34). A class with no training row at all receives probability 0
    from the classifier, so its row is set uniform only to keep the matrix row-stochastic.
    """
    cls = np.asarray(class_codes, dtype=np.int64)
    stg = np.asarray(stage_codes, dtype=np.int64)
    n_stages = len(attack_stages())
    counts = np.zeros((n_classes, n_stages), dtype=np.float64)
    ok = (cls >= 0) & (stg >= 0)
    np.add.at(counts, (cls[ok], stg[ok]), 1.0)
    present = np.bincount(cls[cls >= 0], minlength=n_classes) > 0
    totals = counts.sum(axis=1)
    unplaced = present & (totals == 0)
    if unplaced.any():
        raise InvariantViolation(f"classes {np.nonzero(unplaced)[0].tolist()} have training rows but no label mapped to a tactic")
    out = np.full((n_classes, n_stages), 1.0 / n_stages)
    has = totals > 0
    out[has] = counts[has] / totals[has, None]
    return out
