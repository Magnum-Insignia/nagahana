"""The engine's event log: incremental `ColumnarUpdates` with one entity table, and window slices of it.

Purpose
-------
Live use delivers state updates in chunks (Kafka batches, pcap reads), each a `ColumnarUpdates` with
its own entity table. The engine needs one durable log (AS-11: the event log is `ColumnarUpdates`)
whose entity rows are stable for the whole run, because they are the Environment store's entity ids
(TSTCT's carried keys, AS-161 item 1). `EventLog.append` merges a chunk into the log:

    entity rows: matched by (kind, key); a new (kind, key) gets the next row (append only)
    update rows: appended; entity columns re-mapped to the log's rows; values/status/raw hashes stacked

and `EventLog.window_source(rows)` builds a `PreparedSource` over a run of log rows (the open window),
so `data.windows.build_window` produces exactly what training builds from the same rows.

Labels never enter the model (models/batch.py): the window source carries an "unknown" label table
(malicious NaN, stage −1), which `prepare_source` requires and the engine never reads.

Invariants: entity row ids never change once assigned; a chunk older than the log's last event time
is refused ([Q-20]: ordering is resolved before the model); columns must match the log's.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd

from nagahana.core.errors import InvariantViolation
from nagahana.data.labels import LABEL_COLUMNS
from nagahana.data.windows import PreparedSource, SourceData, prepare_source
from nagahana.datamodel.columnar import ColumnarUpdates

_ENTITY_COLS = ("entity_0", "entity_1", "entity_2")


def unknown_labels(cu: ColumnarUpdates) -> pd.DataFrame:
    """A label table that says nothing (malicious NaN, stage −1): labels are never model inputs."""
    n = len(cu)
    return pd.DataFrame({
        "seq": cu.updates["seq"].to_numpy(dtype=np.int64), "record": cu.updates["record"].to_numpy(dtype=np.int64),
        "label_raw": np.full(n, "", dtype=object), "malicious": np.full(n, np.nan, dtype=np.float32),
        "stage": np.full(n, -1, dtype=np.int64), "technique": np.full(n, "", dtype=object),
        "family": np.full(n, "unknown", dtype=object), "subfamily": np.full(n, "unknown", dtype=object),
        "actor_role": np.full(n, -1, dtype=np.int8), "mapped": np.zeros(n, dtype=bool),
    })[list(LABEL_COLUMNS)]


def take_rows(cu: ColumnarUpdates, rows: np.ndarray) -> ColumnarUpdates:
    """The log restricted to `rows` (in that order), entity table unchanged (row ids keep their meaning).

    Unlike `generator.variants.derive`, nothing is re-stamped: the rows stay real observations with their
    own provenance. `seq` is renumbered 0…n−1 so the label table of the slice lines up.
    """
    r = np.asarray(rows, dtype=np.int64)
    upd = cu.updates.iloc[r].reset_index(drop=True).copy()
    upd["seq"] = np.arange(len(upd), dtype=np.int64)
    return dataclasses.replace(cu, updates=upd, values=cu.values[r].copy(), status=cu.status[r].copy(),
                               raw_hash=cu.raw_hash[r].copy())


class EventLog:
    """One growing `ColumnarUpdates` with stable entity rows (module docstring)."""

    def __init__(self) -> None:
        self.cu: ColumnarUpdates | None = None
        self._index: dict[tuple[str, str], int] = {}

    def __len__(self) -> int:
        return 0 if self.cu is None else len(self.cu)

    @property
    def last_time(self) -> float:
        """Latest event time in the log (−inf when empty)."""
        if self.cu is None or not len(self.cu):
            return -np.inf
        return float(np.nanmax(self.cu.updates["event_time"].to_numpy(dtype=np.float64)))

    def append(self, chunk: ColumnarUpdates) -> np.ndarray:
        """Merge a chunk (sorted by event time inside). Returns the log rows of the chunk's updates."""
        if not len(chunk):
            return np.zeros(0, dtype=np.int64)
        t = chunk.updates["event_time"].to_numpy(dtype=np.float64)
        if not np.isfinite(t).all():
            raise InvariantViolation("every live update needs an event time (the engine cannot place it otherwise)")
        order = np.argsort(t, kind="stable")
        chunk = take_rows(chunk, order)
        if float(chunk.updates["event_time"].iloc[0]) < self.last_time:
            raise InvariantViolation("a chunk older than the log's last event time: ordering must be resolved before "
                                     "the model ([Q-20])")
        # ---- entities: match by (kind, key), append new ones
        ent = chunk.entities
        mapping = np.empty(len(ent), dtype=np.int64)
        new_rows: list[int] = []
        for i, (kind, key) in enumerate(zip(ent["kind"].astype(str), ent["key"].astype(str), strict=True)):
            k = (kind, key)
            if k not in self._index:
                self._index[k] = len(self._index)
                new_rows.append(i)
            mapping[i] = self._index[k]
        upd = chunk.updates.copy()
        for c in _ENTITY_COLS:
            if c in upd:
                e = upd[c].to_numpy(dtype=np.int64)
                upd[c] = np.where(e >= 0, mapping[np.maximum(e, 0)], -1)
        if self.cu is None:
            ents = ent.iloc[new_rows].reset_index(drop=True).copy()
            upd["seq"] = np.arange(len(upd), dtype=np.int64)
            self.cu = dataclasses.replace(chunk, updates=upd, entities=ents)
            return np.arange(len(upd), dtype=np.int64)
        if tuple(c.name for c in chunk.columns) != tuple(c.name for c in self.cu.columns):
            raise InvariantViolation("a chunk with another column layout than the log (one adapter per engine)")
        n0 = len(self.cu)
        upd["seq"] = np.arange(n0, n0 + len(upd), dtype=np.int64)
        extra = ent.iloc[new_rows].reset_index(drop=True)
        ents = pd.concat([self.cu.entities, extra[[c for c in extra.columns if c in self.cu.entities.columns]]],
                         ignore_index=True)
        self.cu = dataclasses.replace(
            self.cu, updates=pd.concat([self.cu.updates, upd], ignore_index=True), entities=ents,
            values=np.concatenate([self.cu.values, chunk.values]), status=np.concatenate([self.cu.status, chunk.status]),
            raw_hash=np.concatenate([self.cu.raw_hash, chunk.raw_hash]),
        )
        return np.arange(n0, n0 + len(upd), dtype=np.int64)

    def window_source(self, rows: np.ndarray, *, network: str) -> PreparedSource:
        """A PreparedSource over log rows (time-sorted), with unknown labels and the log's entity rows."""
        if self.cu is None:
            raise InvariantViolation("the event log is empty")
        part = take_rows(self.cu, rows)
        return prepare_source(SourceData(part, unknown_labels(part), network=network))


__all__ = ["EventLog", "take_rows", "unknown_labels"]
