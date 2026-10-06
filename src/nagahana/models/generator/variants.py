"""Variant data structures: per-update labels, transform results, and provenance-safe table surgery.

Purpose
-------
Every Generator family (deterministic transforms, masked-generative, autoregressive, diffusion) turns
one real window of state updates (`ColumnarUpdates`) and its labels into a variant window and its
labels. This module holds what they share:

- `UpdateLabels`: supervision per update row (malicious, stage, technique) plus the window's family.
  Labels are training data, never model inputs (models/batch.py, "Labels never enter the forward
  pass"). Entity-level labels (`LabelBatch.entity_infiltrated_at`, `entity_malicious_share`) are *not*
  stored here: the data pipeline derives them from update labels and times when it builds windows, so
  they stay consistent with whatever rows a variant keeps.
- `TransformResult`: the variant table, its labels, and `source_rows` — for each variant row, the
  row of the real window it derives from. This is row-level provenance; it is how every test checks
  label preservation: `result.labels == source_labels.take(result.source_rows)` for every transform
  that keeps labels by construction.
- `derive(...)`: builds a variant `ColumnarUpdates` from a selection of source rows, with new
  values/status/update columns, rebuilt relations and recomputed entity summaries.

Provenance of a variant table (AS-369)
-------------------------------------
- `adapter = "nagahana-generator"`; `source_id` is set by the pipeline to "<real source>~<variant id>".
- `raw_hash` rows are **zero**: a generated row has no raw record, and copying the real record's hash
  would claim a chain of custody (ARCH §7, §10) for something that was never on the wire.
- `record = -1` (no source record); `derived_from_seq` = the real row's `seq`; `origin = "generated"`.
- Fact tables (`aliases`, `roles`, `names`) are carried unchanged (they are facts about entities).

Owner sources: [A-17], [Q-37]. Decisions: D-40 (training only), D-41 (absence is not zero), D-23.
Assumptions: AS-27, AS-369.

Invariants (checked by `derive` through `ColumnarUpdates.validate()`)
- excluded cells hold NaN and contributing cells hold a value;
- entity indices stay valid (the entity table is never re-indexed);
- `len(labels) == len(updates) == len(source_rows)`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from nagahana.core.errors import InvariantViolation
from nagahana.datamodel.columnar import ColumnarUpdates, finalise_tables

GENERATOR_ADAPTER = "nagahana-generator"
GENERATOR_VERSION = "1.0.0"


# ============================================================================== labels
@dataclass(frozen=True)
class UpdateLabels:
    """Per-update supervision of one window (never a model input).

    malicious: float32 [U], 1 / 0 or NaN (unknown). stage: int64 [U], `vocab.STAGES` code or −1.
    technique: int64 [U], technique slot or −1. family: the window's attack family ("benign" if none).
    """

    malicious: np.ndarray
    stage: np.ndarray
    technique: np.ndarray
    family: str

    def __post_init__(self) -> None:
        n = len(self.malicious)
        if len(self.stage) != n or len(self.technique) != n:
            raise InvariantViolation("UpdateLabels arrays must have one entry per update")
        ok = np.isnan(self.malicious) | (self.malicious == 0) | (self.malicious == 1)
        if not bool(np.all(ok)):
            raise InvariantViolation("malicious labels must be 0, 1 or NaN")

    def __len__(self) -> int:
        return len(self.malicious)

    def take(self, rows: np.ndarray) -> UpdateLabels:
        """Labels of the given source rows, in that order (labels move with their rows)."""
        r = np.asarray(rows, dtype=np.int64)
        return UpdateLabels(self.malicious[r].copy(), self.stage[r].copy(), self.technique[r].copy(), self.family)

    def malicious_rows(self) -> np.ndarray:
        """bool [U]: rows known to be malicious (NaN counts as not known)."""
        return np.nan_to_num(self.malicious, nan=0.0) == 1.0

    def benign_rows(self) -> np.ndarray:
        """bool [U]: rows known to be benign (NaN counts as not known)."""
        return np.nan_to_num(self.malicious, nan=-1.0) == 0.0

    def has_attack(self) -> bool:
        return bool(self.malicious_rows().any())

    def same_as(self, other: UpdateLabels) -> bool:
        """Exact equality (NaN equal to NaN)."""
        return (
            self.family == other.family
            and np.array_equal(self.malicious, other.malicious, equal_nan=True)
            and np.array_equal(self.stage, other.stage)
            and np.array_equal(self.technique, other.technique)
        )


# ============================================================================== results
@dataclass
class TransformResult:
    """One candidate variant before acceptance.

    updates: the variant table. labels: its labels. source_rows: int64 [U'], source row of each variant
    row. changed: bool [U', C], cells whose value or status differs from the source row's cell.
    free: bool [U', C] or None, cells a learned family generated (projection may move only these).
    params: what the producer drew (rates, factors, seeds), recorded for audit.
    label_mode: "by-construction" (deterministic, label-preserving transform) or "copied" (learned).
    structural: True when rows were dropped, re-ordered, re-timed or re-wired (changes outside the
        value/status matrices). A variant with no changed cell and no structural change is a duplicate.
    """

    updates: ColumnarUpdates
    labels: UpdateLabels
    source_rows: np.ndarray
    changed: np.ndarray
    producer: str
    label_mode: str
    params: dict[str, Any] = field(default_factory=dict)
    free: np.ndarray | None = None
    structural: bool = False

    def is_duplicate(self) -> bool:
        """True when the variant equals its source window (nothing to learn from it)."""
        return not self.structural and not bool(self.changed.any())


# ============================================================================== table surgery
def entity_columns(frame: pd.DataFrame) -> list[str]:
    """`entity_0 … entity_k` columns, in order."""
    return sorted((c for c in frame.columns if c.startswith("entity_")), key=lambda c: int(c.split("_")[1]))


def is_packet_granularity(cu: ColumnarUpdates) -> bool:
    """True when rows are packets carrying their flow's running state (the PCAP adapter's form).

    Recognised by a `flow` column in which some flow has more than one row; a source with one row per
    flow (CSV/NetFlow form) is flow granularity.
    """
    if "flow" not in cu.updates.columns or len(cu) == 0:
        return False
    f = cu.updates["flow"].to_numpy()
    return bool(len(np.unique(f)) < len(f))


def changed_cells(src: ColumnarUpdates, rows: np.ndarray, values: np.ndarray, status: np.ndarray) -> np.ndarray:
    """bool [U', C]: cells differing from the source rows (value or status; NaN equals NaN)."""
    sv, ss = src.values[rows], src.status[rows]
    same_v = (sv == values) | (np.isnan(sv) & np.isnan(values))
    return ~same_v | (ss != status)


def derive(
    src: ColumnarUpdates,
    rows: np.ndarray,
    *,
    values: np.ndarray | None = None,
    status: np.ndarray | None = None,
    frame_updates: Mapping[str, np.ndarray] | None = None,
) -> ColumnarUpdates:
    """A variant table made of source `rows` (in that order), with optional new matrices / columns.

    values, status: [U', C] replacements (default: the source rows' cells). frame_updates: column →
    new array [U'] for the `updates` table (e.g. new event times or entity indices). Relations are
    rebuilt from the entity columns (relation = the tuple of entities touched, as in `to_columnar`),
    entity summaries recomputed, and provenance columns stamped (AS-369). Validates before returning.
    """
    r = np.asarray(rows, dtype=np.int64)
    vals = src.values[r].copy() if values is None else np.asarray(values, dtype=np.float64).copy()
    stat = src.status[r].copy() if status is None else np.asarray(status, dtype=np.uint8).copy()
    frame = src.updates.iloc[r].reset_index(drop=True).copy()
    for col, arr in (frame_updates or {}).items():
        frame[col] = np.asarray(arr)
    # Provenance columns (AS-369): the real row's seq, no source record, generated origin.
    prev = frame["derived_from_seq"].to_numpy() if "derived_from_seq" in frame else frame["seq"].to_numpy()
    frame["derived_from_seq"] = prev.astype(np.int64)
    frame["seq"] = np.arange(len(frame), dtype=np.int64)
    frame["record"] = np.full(len(frame), -1, dtype=np.int64)
    frame["origin"] = "generated"
    # Relations: dense renumbering of the distinct entity tuples, in order of first appearance.
    ents = entity_columns(frame)
    keys = [tuple(int(v) for v in row) for row in frame[ents].to_numpy()] if ents else [() for _ in range(len(frame))]
    rel_index: dict[tuple[int, ...], int] = {}
    rel = np.array([rel_index.setdefault(k, len(rel_index)) for k in keys], dtype=np.int64)
    frame["relation"] = rel
    out = ColumnarUpdates(
        source_id=src.source_id, adapter=GENERATOR_ADAPTER, adapter_version=GENERATOR_VERSION,
        columns=src.columns, updates=frame, entities=src.entities[["kind", "key"]].copy(),
        relations=pd.DataFrame(), values=vals, status=stat,
        raw_hash=np.zeros((len(frame), 32), dtype=np.uint8),                 # no raw record (AS-369)
        clock_quality=src.clock_quality, vocab={k: list(v) for k, v in src.vocab.items()},
        side_fields=dict(src.side_fields), explicit_fields=src.explicit_fields,
        aliases=src.aliases.copy(), roles=src.roles.copy(), names=src.names.copy(),
    )
    # keep adapter-specific entity columns (they describe entities, not rows)
    for c in src.entities.columns:
        if c not in ("kind", "key", "first_seen", "last_seen", "updates"):
            out.entities[c] = src.entities[c].to_numpy()
    if "direction" not in out.updates:
        out.updates["direction"] = np.full(len(frame), -1, dtype=np.int8)
    finalise_tables(out, relation_entities={v: k for k, v in rel_index.items()})
    out.validate()
    return out


def sort_by_time(times: np.ndarray) -> np.ndarray:
    """Stable order of rows by (possibly new) event times."""
    return np.argsort(np.asarray(times, dtype=np.float64), kind="stable")
