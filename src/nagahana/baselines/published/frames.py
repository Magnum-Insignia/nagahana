"""Frame utilities shared by the reproductions: metadata, labels, event time and ordering.

Labels
------
Datasets spell their classes differently ("BENIGN", "Benign", "Normal", "BenignTraffic", 0/1,
"flow=From-Botnet-V42-TCP-CC6...", "Background"). `binary_labels` maps a label column to the coding of
evaluation/predictions.py: 1 malicious, 0 benign, -1 unknown. A label that names neither benign nor
malicious traffic (CTU-13 "Background", empty cells, NaN) is unknown, never benign: unknown units are
excluded from scoring, the same rule data/labels.py applies to NagaHana's own labels.

Metadata
--------
`build_meta` assembles the per-unit metadata frame of the prediction records from the meta columns a
frame carries (time, dataset, network, family, novelty, split, entity) and neutral defaults for the
rest, then validates it with `evaluation.predictions.make_meta`.
"""

from __future__ import annotations

import re
from collections.abc import Collection, Mapping
from typing import Any

import numpy as np
import pandas as pd

from nagahana.core.errors import InvariantViolation
from nagahana.evaluation.predictions import META_COLUMNS, make_meta

#: Normalised class names meaning benign traffic, over the datasets this package reads.
BENIGN_TOKENS: frozenset[str] = frozenset({"benign", "normal", "benigntraffic", "normaltraffic", "legitimate"})
#: Normalised class names meaning "not annotated".
UNKNOWN_TOKENS: frozenset[str] = frozenset({"", "nan", "none", "unknown", "background", "unlabeled", "unlabelled"})


def normalise_token(value: Any) -> str:
    """Lower case, letters and digits only ("DoS attacks-Hulk" -> "dosattackshulk")."""
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def ctu13_label_class(text: str) -> int:
    """CTU-13 binetflow label -> 1 botnet, 0 normal, -1 background or unknown.

    The CTU-13 labels read "flow=From-Botnet-...", "flow=To-Botnet-...", "flow=From-Normal-...",
    "flow=To-Normal-..." and "flow=Background-..." (Garcia et al., Computers and Security 45, 2014).
    """
    t = str(text).lower()
    if "botnet" in t:
        return 1
    if "normal" in t:
        return 0
    return -1


def binary_labels(
    values: pd.Series | np.ndarray | Collection[Any],
    *,
    benign: Collection[str] = BENIGN_TOKENS,
    unknown: Collection[str] = UNKNOWN_TOKENS,
    style: str = "auto",
) -> np.ndarray:
    """Map a label column to int64 codes: 1 malicious, 0 benign, -1 unknown.

    Parameters
    ----------
    values:
        The label column.
    benign, unknown:
        Normalised class names (see `normalise_token`) meaning benign and unknown.
    style:
        "auto" (numeric 0/1 columns pass through, text columns use the token sets), "numeric",
        "text" or "ctu13" (binetflow labels, `ctu13_label_class`).
    """
    s = values.reset_index(drop=True) if isinstance(values, pd.Series) else pd.Series(np.asarray(list(values), dtype=object))
    if style == "ctu13":
        return np.asarray([ctu13_label_class(v) for v in s], dtype=np.int64)
    raw_null = s.isna()
    numeric = pd.to_numeric(s, errors="coerce")
    # A column is numeric when every non-missing cell parses as a number (missing cells become unknown).
    convertible = numeric.notna() | raw_null
    is_numeric = style == "numeric" or (style == "auto" and len(s) > 0 and bool(convertible.all()) and bool((~raw_null).any()))
    if is_numeric:
        arr = numeric.to_numpy(dtype=np.float64)
        out = np.full(arr.shape, -1, dtype=np.int64)
        ok = np.isfinite(arr)
        bad = ok & ~np.isin(arr, (0.0, 1.0))
        if bad.any():
            raise InvariantViolation(f"numeric binary labels must be 0 or 1; found {sorted(set(arr[bad].tolist()))[:5]}")
        out[ok] = arr[ok].astype(np.int64)
        return out
    if style not in ("auto", "text"):
        raise ValueError(f"unknown label style {style!r}")
    benign_set, unknown_set = set(benign), set(unknown)
    out = np.empty(len(s), dtype=np.int64)
    for i, v in enumerate(s.tolist()):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            out[i] = -1
            continue
        tok = normalise_token(v)
        out[i] = 0 if tok in benign_set else (-1 if tok in unknown_set else 1)
    return out


def class_codes(values: pd.Series, classes: tuple[str, ...]) -> np.ndarray:
    """Map class names to their index in `classes` (by normalised name); names not listed map to -1."""
    index = {normalise_token(c): i for i, c in enumerate(classes)}
    return np.asarray([index.get(normalise_token(v), -1) for v in values.tolist()], dtype=np.int64)


def epoch_seconds(values: pd.Series) -> np.ndarray:
    """Event time in float64 epoch seconds from numeric seconds or datetime-like values.

    Datetime values without a zone are read as UTC (the same convention as ingest/csv_flows.py when no
    UTC offset is given). Unparseable values raise.
    """
    if pd.api.types.is_numeric_dtype(values) and not pd.api.types.is_bool_dtype(values):
        out = values.to_numpy(dtype=np.float64)
    else:
        parsed = pd.to_datetime(values, utc=True, errors="coerce")
        if bool(parsed.isna().any()):
            bad = values[parsed.isna()].head(3).tolist()
            raise InvariantViolation(f"time values cannot be parsed as datetimes, e.g. {bad}")
        out = (parsed - pd.Timestamp(0, tz="UTC")).dt.total_seconds().to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(out)):
        raise InvariantViolation("time column contains NaN or infinite values")
    return out


def time_order(frame: pd.DataFrame, time_column: str | None) -> np.ndarray:
    """Stable permutation that sorts rows by event time (file order breaks ties); identity without time."""
    if time_column is None or time_column not in frame.columns:
        return np.arange(len(frame), dtype=np.int64)
    t = epoch_seconds(frame[time_column])
    return np.argsort(t, kind="stable").astype(np.int64)


def build_meta(
    frame: pd.DataFrame,
    *,
    time: np.ndarray | None = None,
    family: np.ndarray | pd.Series | None = None,
    entity: np.ndarray | None = None,
    extra: Mapping[str, Any] | None = None,
    time_column: str | None = None,
) -> pd.DataFrame:
    """Metadata frame (META_COLUMNS plus `extra`) with one row per row of `frame`.

    Values come from, in order of precedence: the explicit arguments, the frame's own meta columns, the
    neutral defaults of `make_meta`. `time_column` names the frame's event-time column when it is not
    called "time".
    """
    n = len(frame)
    cols: dict[str, Any] = {}
    for col in META_COLUMNS:
        if col in frame.columns:
            cols[col] = frame[col].to_numpy()
    if time is not None:
        cols["time"] = np.asarray(time, dtype=np.float64)
    elif time_column is not None and time_column in frame.columns:
        cols["time"] = epoch_seconds(frame[time_column])
    elif "time" in cols:
        cols["time"] = epoch_seconds(frame["time"])
    if family is not None:
        cols["family"] = np.asarray(family, dtype=object).astype(str)
    if entity is not None:
        cols["entity"] = np.asarray(entity, dtype=np.int64)
    for key in ("dataset", "network", "family", "novelty", "split"):
        if key in cols:
            cols[key] = np.asarray(["" if (v is None or (isinstance(v, float) and np.isnan(v))) else str(v) for v in cols[key]],
                                   dtype=object)
    if "entity" in cols:
        cols["entity"] = np.asarray(cols["entity"], dtype=np.int64)
    return make_meta(n, **cols, **dict(extra or {}))


def take_rows(frame: pd.DataFrame, index: np.ndarray) -> pd.DataFrame:
    """Rows of `frame` at integer positions `index`, with a fresh RangeIndex."""
    return frame.iloc[np.asarray(index, dtype=np.int64)].reset_index(drop=True)
