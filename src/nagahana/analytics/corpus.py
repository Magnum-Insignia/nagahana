"""The analysis view of a corpus: every state update of every source in the canonical column layout.

A `Corpus` gathers what the analyses read, row-aligned, one row per state update (D-30):

    values, status      float64 / uint8 [n, C] in the canonical column layout of `data.windows`
                        (`CANONICAL_COLUMNS`, stable slots, P-22). A column a source does not have is
                        NOT_SUPPLIED with NaN for all its rows (D-41: absence is never zero).
    time                float64 [n] event time in epoch seconds; NaN for sources without times
                        (CIC-IoT-2023 CSVs, AS-307)
    source              int32 [n] index into `sources` (dataset, network, adapter, origin, clock quality)
    entities            int64 [n, 3] corpus entity ids of initiator, responder and service (-1 none).
                        An entity is identified by (network, kind, key): the same address in two files of
                        one network is one machine (D-48), the same private address in two networks is not.
    labels              malicious (1 / 0 / NaN unknown), stage code (-1 unknown), family, subfamily,
                        technique and the raw label, from `data.labels` (AS-34). Labels are carried
                        beside the observations for auditing only.
    split, novelty,     the split role of the window or segment record covering the row, its novelty
    window              mark (zero-shot only) and the record id, from a `data.sampling.SplitManifest`
                        ("" where no manifest is given)
    flow                int64 [n] flow id inside its source: the PCAP adapter's `flow` column (one flow,
                        many flow-state updates, D-51); for flow-record sources every record is one flow
                        and gets its record number
    raw_hash            uint8 [n, 32] SHA-256 of the raw record (provenance)
    reorder, watermark  float64 [n] ordering quality of each update (NaN when unknown)

Rows of one source follow that source's event-time order (`PreparedSource.order`), so the
[start, stop) ranges of window and segment records index rows directly.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from nagahana.core.errors import InvariantViolation
from nagahana.datamodel.columnar import CODE_NOT_SUPPLIED, CODE_OBSERVED, STATUS_ORDER, Column
from nagahana.datamodel.fields import Kind
from nagahana.datamodel.status import CONTRIBUTING, EXCLUDED

#: Status codes that carry a value (O_t of datamodel/status.py) and those that never do.
CONTRIBUTING_CODES: np.ndarray = np.array([i for i, s in enumerate(STATUS_ORDER) if s in CONTRIBUTING], dtype=np.uint8)
EXCLUDED_CODES: np.ndarray = np.array([i for i, s in enumerate(STATUS_ORDER) if s in EXCLUDED], dtype=np.uint8)
#: Status names in code order (columns of the status tables).
STATUS_NAMES: tuple[str, ...] = tuple(s.value for s in STATUS_ORDER)
#: Kinds whose values are magnitudes (compared on a numeric scale); the rest are codes.
NUMERIC_KINDS: frozenset[Kind] = frozenset({Kind.CONTINUOUS, Kind.COUNT, Kind.HISTOGRAM})


@dataclass(frozen=True)
class SourceInfo:
    """Where the rows of one source come from."""

    source_id: str
    dataset: str
    network: str
    adapter: str
    adapter_version: str
    origin: str
    timeless: bool
    clock_quality: str | None
    rows: int
    explicit_fields: tuple[str, ...] = ()


@dataclass
class Corpus:
    """Row-aligned analysis view (module docstring)."""

    columns: tuple[Column, ...]
    values: np.ndarray
    status: np.ndarray
    time: np.ndarray
    source: np.ndarray
    sources: tuple[SourceInfo, ...]
    entities: np.ndarray
    entity_kind: np.ndarray
    entity_key: np.ndarray
    entity_network: np.ndarray
    entity_internal: np.ndarray
    malicious: np.ndarray
    stage: np.ndarray
    family: np.ndarray
    subfamily: np.ndarray
    technique: np.ndarray
    label_raw: np.ndarray
    split: np.ndarray
    novelty: np.ndarray
    window: np.ndarray
    flow: np.ndarray
    raw_hash: np.ndarray
    reorder: np.ndarray
    watermark: np.ndarray

    def __post_init__(self) -> None:
        self.validate()

    def __len__(self) -> int:
        return int(self.values.shape[0])

    def validate(self) -> None:
        """Shape and D-41 pairing checks (an excluded cell is NaN, a contributing cell is not)."""
        n, c = self.values.shape
        if self.status.shape != (n, c) or len(self.columns) != c:
            raise InvariantViolation("values, status and columns disagree in shape")
        for name in ("time", "source", "malicious", "stage", "family", "subfamily", "technique", "label_raw", "split",
                     "novelty", "window", "flow", "reorder", "watermark"):
            if getattr(self, name).shape[0] != n:
                raise InvariantViolation(f"Corpus.{name} must have one entry per row")
        if self.entities.shape != (n, 3) or self.raw_hash.shape != (n, 32):
            raise InvariantViolation("entities must be [n, 3] and raw_hash [n, 32]")
        excluded = np.isin(self.status, EXCLUDED_CODES)
        nan = np.isnan(self.values)
        if (excluded & ~nan).any():
            raise InvariantViolation("an excluded cell carries a value (absence is never zero, D-41)")
        if (~excluded & nan).any():
            raise InvariantViolation("a contributing cell carries no value")
        e = self.entity_kind.shape[0]
        if self.entities.size and int(self.entities.max(initial=-1)) >= e:
            raise InvariantViolation("entity id out of range")

    def column_index(self, name: str) -> int:
        """Position of a column by name (KeyError if absent)."""
        for j, col in enumerate(self.columns):
            if col.name == name:
                return j
        raise KeyError(f"no column {name!r}; columns: {[c.name for c in self.columns]}")

    def has_column(self, name: str) -> bool:
        """True when a column of that name exists."""
        return any(col.name == name for col in self.columns)

    def contributing(self) -> np.ndarray:
        """bool [n, C]: cells that count as evidence."""
        return np.isin(self.status, CONTRIBUTING_CODES)

    def numeric_columns(self) -> np.ndarray:
        """bool [C]: columns holding magnitudes (continuous, count, histogram bins)."""
        return np.array([col.kind in NUMERIC_KINDS for col in self.columns], dtype=bool)

    def label_class(self) -> np.ndarray:
        """object [n]: "malicious", "benign" or "unknown" from the malicious flag."""
        out = np.full(len(self), "unknown", dtype=object)
        out[self.malicious == 1.0] = "malicious"
        out[self.malicious == 0.0] = "benign"
        return out

    def row_attr(self, attr: str) -> np.ndarray:
        """object [n]: a `SourceInfo` attribute (dataset, network, adapter, origin, source_id) per row."""
        table = np.array([getattr(s, attr) for s in self.sources], dtype=object)
        return table[self.source] if len(self) else np.zeros(0, dtype=object)

    def subset(self, rows: np.ndarray) -> Corpus:
        """The corpus restricted to `rows` (bool mask or integer indices); entity tables are kept whole."""
        r = np.asarray(rows)
        idx = np.flatnonzero(r) if r.dtype == bool else r.astype(np.int64)
        return replace(
            self, values=self.values[idx], status=self.status[idx], time=self.time[idx], source=self.source[idx],
            entities=self.entities[idx], malicious=self.malicious[idx], stage=self.stage[idx],
            family=self.family[idx], subfamily=self.subfamily[idx], technique=self.technique[idx],
            label_raw=self.label_raw[idx], split=self.split[idx], novelty=self.novelty[idx], window=self.window[idx],
            flow=self.flow[idx], raw_hash=self.raw_hash[idx], reorder=self.reorder[idx], watermark=self.watermark[idx],
        )

    # Constructors
    @classmethod
    def from_prepared(cls, sources: Sequence[object], manifest: object | None = None) -> Corpus:
        """Corpus of prepared sources (`data.windows.PreparedSource`), with splits from a manifest.

        manifest: a `data.sampling.SplitManifest` whose records index these sources (record.source is
        the position in `sources`); every row covered by a record takes the record's role, novelty and id.
        """
        from nagahana.data.windows import CANONICAL_COLUMNS, PreparedSource

        parts: list[dict[str, np.ndarray]] = []
        infos: list[SourceInfo] = []
        ent_frames: list[pd.DataFrame] = []
        c_n = len(CANONICAL_COLUMNS)
        for si, src in enumerate(sources):
            if not isinstance(src, PreparedSource):
                raise TypeError("from_prepared expects data.windows.PreparedSource objects")
            cu = src.data.updates
            order = src.order
            n = order.shape[0]
            # Canonical value/status matrices (as data.windows.build_window does for one window).
            values = np.full((n, c_n), np.nan)
            status = np.full((n, c_n), CODE_NOT_SUPPLIED, dtype=np.uint8)
            have = src.src_cols >= 0
            st = cu.status[order][:, src.src_cols[have]]
            status[:, have] = st
            values[:, have] = np.where(np.isin(st, CONTRIBUTING_CODES), cu.values[order][:, src.src_cols[have]], np.nan)
            labels = src.data.labels.set_index("seq").sort_index()
            upd = cu.updates
            flow = (upd["flow"].to_numpy(dtype=np.int64)[order] if "flow" in upd
                    else upd["record"].to_numpy(dtype=np.int64)[order] if "record" in upd else order.astype(np.int64))
            parts.append({
                "values": values, "status": status,
                "time": np.full(n, np.nan) if src.timeless else src.time.astype(np.float64),
                "source": np.full(n, si, dtype=np.int32),
                "ents": src.ents.astype(np.int64),
                "malicious": src.malicious.astype(np.float64), "stage": src.stage.astype(np.int64),
                "family": src.family.astype(object), "technique": src.technique.astype(object),
                "subfamily": labels["subfamily"].to_numpy(dtype=object)[order] if "subfamily" in labels
                else src.family.astype(object),
                "label_raw": labels["label_raw"].to_numpy(dtype=object)[order] if "label_raw" in labels
                else np.full(n, "", dtype=object),
                "flow": flow, "raw_hash": cu.raw_hash[order],
                "reorder": upd["reorder_uncertainty_s"].to_numpy(dtype=np.float64)[order],
                "watermark": upd["watermark"].to_numpy(dtype=np.float64)[order],
            })
            ents = cu.entities
            ent_frames.append(pd.DataFrame({
                "source": si, "row": np.arange(len(ents)), "network": src.data.network,
                "kind": ents["kind"].astype(str).to_numpy(), "key": ents["key"].astype(str).to_numpy(),
                "internal": src.entity_internal.astype(bool),
            }))
            infos.append(SourceInfo(
                source_id=src.data.source_id, dataset=src.data.dataset, network=src.data.network, adapter=cu.adapter,
                adapter_version=cu.adapter_version, origin=src.data.origin, timeless=bool(src.timeless),
                clock_quality=cu.clock_quality, rows=int(n), explicit_fields=tuple(cu.explicit_fields),
            ))
        if not parts:
            raise InvariantViolation("a corpus needs at least one source")
        # Corpus entity ids by (network, kind, key) (module docstring).
        ent = pd.concat(ent_frames, ignore_index=True)
        ident = ent["network"].astype(str) + "\x1f" + ent["kind"] + "\x1f" + ent["key"]
        eid, uniq = pd.factorize(ident)
        first = pd.Series(np.arange(len(ent))).groupby(eid).min().to_numpy()
        offsets = np.cumsum([0] + [len(f) for f in ent_frames])
        entity_rows = []
        for si, p in enumerate(parts):
            e = p["ents"]
            m = np.where(e >= 0, eid[np.clip(e, 0, None) + offsets[si]], -1)
            entity_rows.append(m.astype(np.int64))
        # Internal flag of an entity: from its first occurrence (adapters agree within one network).
        cat = {k: np.concatenate([p[k] for p in parts]) for k in parts[0] if k != "ents"}
        n_all = cat["values"].shape[0]
        corpus = cls(
            columns=tuple(CANONICAL_COLUMNS), values=cat["values"], status=cat["status"], time=cat["time"],
            source=cat["source"], sources=tuple(infos), entities=np.concatenate(entity_rows),
            entity_kind=ent["kind"].to_numpy(dtype=object)[first], entity_key=ent["key"].to_numpy(dtype=object)[first],
            entity_network=ent["network"].to_numpy(dtype=object)[first],
            entity_internal=ent["internal"].to_numpy(dtype=bool)[first],
            malicious=cat["malicious"], stage=cat["stage"], family=cat["family"], subfamily=cat["subfamily"],
            technique=cat["technique"], label_raw=cat["label_raw"], split=np.full(n_all, "", dtype=object),
            novelty=np.full(n_all, "", dtype=object), window=np.full(n_all, "", dtype=object), flow=cat["flow"],
            raw_hash=cat["raw_hash"], reorder=cat["reorder"], watermark=cat["watermark"],
        )
        if len(uniq) != corpus.entity_kind.shape[0]:
            raise InvariantViolation("entity table construction failed")
        if manifest is not None:
            corpus.assign_manifest(manifest)
        return corpus

    @classmethod
    def from_sources(cls, sources: Sequence[object], manifest: object | None = None) -> Corpus:
        """Corpus of `data.windows.SourceData` objects (prepared here, then `from_prepared`)."""
        from nagahana.data.windows import SourceData, prepare_source

        prepared = []
        for s in sources:
            if not isinstance(s, SourceData):
                raise TypeError("from_sources expects data.windows.SourceData objects")
            prepared.append(prepare_source(s))
        return cls.from_prepared(prepared, manifest)

    def assign_manifest(self, manifest: object) -> None:
        """Set split, novelty and window of the rows covered by the manifest's records."""
        from nagahana.data.sampling import SplitManifest

        if not isinstance(manifest, SplitManifest):
            raise TypeError("assign_manifest expects a data.sampling.SplitManifest")
        offsets = np.cumsum([0] + [s.rows for s in self.sources])
        for rid, rec in manifest.records.items():
            if not 0 <= rec.source < len(self.sources):
                raise InvariantViolation(f"record {rid} refers to source {rec.source}, outside the corpus")
            lo, hi = offsets[rec.source] + rec.start, offsets[rec.source] + rec.stop
            if not offsets[rec.source] <= lo < hi <= offsets[rec.source + 1]:
                raise InvariantViolation(f"record {rid} [{rec.start}, {rec.stop}) lies outside its source")
            role = manifest.role.get(rid)
            self.split[lo:hi] = role.value if role is not None else ""
            nov = manifest.novelty.get(rid)
            self.novelty[lo:hi] = nov.value if nov is not None else ""
            self.window[lo:hi] = rid

    @classmethod
    def from_arrays(
        cls,
        values: np.ndarray,
        *,
        status: np.ndarray | None = None,
        names: Sequence[str] | None = None,
        kinds: Sequence[Kind] | None = None,
        time: np.ndarray | None = None,
        malicious: np.ndarray | None = None,
        stage: np.ndarray | None = None,
        family: np.ndarray | None = None,
        split: np.ndarray | None = None,
        novelty: np.ndarray | None = None,
        window: np.ndarray | None = None,
        entities: np.ndarray | None = None,
        source: np.ndarray | None = None,
        sources: Sequence[SourceInfo] | None = None,
        flow: np.ndarray | None = None,
    ) -> Corpus:
        """Corpus from plain arrays (generic tables, simulated worlds, tests).

        values: [n, C], NaN where a cell carries no value. status: uint8 [n, C] codes of
        `STATUS_ORDER`; default OBSERVED where a value exists and NOT_SUPPLIED where it does not.
        names, kinds: column names (default "c0", "c1", ...) and kinds (default continuous).
        entities: int64 [n, 2] or [n, 3] entity ids (default none). Missing label arrays are unknown.
        """
        v = np.asarray(values, dtype=np.float64)
        v = v.reshape(-1, 1) if v.ndim == 1 else v
        n, c = v.shape
        st = (np.where(np.isnan(v), CODE_NOT_SUPPLIED, CODE_OBSERVED).astype(np.uint8) if status is None
              else np.asarray(status, dtype=np.uint8))
        col_names = list(names) if names is not None else [f"c{j}" for j in range(c)]
        col_kinds = list(kinds) if kinds is not None else [Kind.CONTINUOUS] * c
        if len(col_names) != c or len(col_kinds) != c:
            raise ValueError("names and kinds must have one entry per column")
        cols = tuple(Column(name=nm, field_id=nm, component=None, kind=kd, unit=None)
                     for nm, kd in zip(col_names, col_kinds, strict=True))
        ents = np.full((n, 3), -1, dtype=np.int64)
        if entities is not None:
            e = np.asarray(entities, dtype=np.int64)
            ents[:, : e.shape[1]] = e
        n_ent = int(ents.max(initial=-1)) + 1
        src = np.zeros(n, dtype=np.int32) if source is None else np.asarray(source, dtype=np.int32)
        infos = tuple(sources) if sources is not None else tuple(
            SourceInfo(source_id=f"source{k}", dataset="", network="", adapter="arrays", adapter_version="",
                       origin="real", timeless=time is None, clock_quality=None, rows=int((src == k).sum()))
            for k in range(int(src.max(initial=0)) + 1))

        def obj(a: np.ndarray | None, fill: str) -> np.ndarray:
            return np.full(n, fill, dtype=object) if a is None else np.asarray(a, dtype=object)

        return cls(
            columns=cols, values=v, status=st, time=np.full(n, np.nan) if time is None else np.asarray(time, np.float64),
            source=src, sources=infos, entities=ents, entity_kind=np.full(n_ent, "host", dtype=object),
            entity_key=np.array([str(i) for i in range(n_ent)], dtype=object),
            entity_network=np.full(n_ent, "", dtype=object), entity_internal=np.ones(n_ent, dtype=bool),
            malicious=np.full(n, np.nan) if malicious is None else np.asarray(malicious, dtype=np.float64),
            stage=np.full(n, -1, dtype=np.int64) if stage is None else np.asarray(stage, dtype=np.int64),
            family=obj(family, "unknown"), subfamily=obj(family, "unknown"), technique=obj(None, ""),
            label_raw=obj(None, ""), split=obj(split, ""), novelty=obj(novelty, ""), window=obj(window, ""),
            flow=np.arange(n, dtype=np.int64) if flow is None else np.asarray(flow, dtype=np.int64),
            raw_hash=np.zeros((n, 32), dtype=np.uint8), reorder=np.full(n, np.nan), watermark=np.full(n, np.nan),
        )

    @classmethod
    def from_frame(
        cls,
        frame: pd.DataFrame,
        *,
        columns: Sequence[str] | None = None,
        kinds: Mapping[str, Kind] | None = None,
        time_column: str | None = None,
        label_column: str | None = None,
        family_column: str | None = None,
        split_column: str | None = None,
    ) -> Corpus:
        """Corpus from a generic table: numeric columns become fields, empty cells NOT_SUPPLIED (D-41).

        label_column: 1 / 0 (or true / false) malicious flag; other values are unknown.
        """
        reserved = {c for c in (time_column, label_column, family_column, split_column) if c}
        use = list(columns) if columns is not None else [
            c for c in frame.columns if c not in reserved and pd.api.types.is_numeric_dtype(frame[c])]
        if not use:
            raise ValueError("no numeric columns to analyse")
        values = np.array(frame[use].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64), copy=True)
        values[~np.isfinite(values)] = np.nan                             # inf is not a measurement either (AS-302)
        mal = None
        if label_column:
            raw = frame[label_column]
            num = pd.to_numeric(raw.map({True: 1, False: 0, "true": 1, "false": 0}).fillna(raw), errors="coerce")
            mal = np.where(num.isin([0, 1]), num, np.nan).astype(np.float64)
        return cls.from_arrays(
            values, names=use, kinds=[(kinds or {}).get(c, Kind.CONTINUOUS) for c in use],
            time=frame[time_column].to_numpy(dtype=np.float64) if time_column else None, malicious=mal,
            family=frame[family_column].astype(str).to_numpy(dtype=object) if family_column else None,
            split=frame[split_column].astype(str).to_numpy(dtype=object) if split_column else None,
        )


__all__ = ["CONTRIBUTING_CODES", "EXCLUDED_CODES", "NUMERIC_KINDS", "STATUS_NAMES", "Corpus", "SourceInfo"]
