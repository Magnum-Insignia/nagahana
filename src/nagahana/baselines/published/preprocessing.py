"""Feature preprocessing of the tabular reproductions, fitted on training data only.

Every statistic (minimum, maximum, mean, standard deviation, category list, fill value) is estimated
on the training frame and stored with the fitted baseline, so no information from evaluation data
reaches the features of the model that scores it.

Scalers follow the definitions of the scikit-learn classes the papers name, so a reproduction that
says "min-max" or "StandardScaler" computes the same numbers without needing the library:

    minmax     x' = (x - min) / (max - min); a constant column has range 1 (sklearn MinMaxScaler)
    standard   x' = (x - mean) / std with the population std (ddof 0); std 0 -> 1 (sklearn StandardScaler)
    l2         x' = x / ||x||_2 per row; an all-zero row stays zero (sklearn Normalizer(norm="l2"))

Non-finite values (AS-532)
--------------------------
Flow exports contain values that are not measurements: CICFlowMeter writes "Infinity" and "NaN" for
rates of zero-duration flows. Training rows with such values are dropped or repaired according to the
baseline's `nonfinite` setting (papers that state a policy set it; otherwise "drop", the common practice
of the cited studies). At prediction time every row must still receive a score, so non-finite cells are
repaired with training statistics: +inf by the column's finite training maximum, -inf by its minimum,
NaN by its median (`FiniteGuard`).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

from nagahana.core.errors import InvariantViolation

ScalerName = Literal["none", "minmax", "standard", "l2"]
OTHER_CATEGORY = "__other__"


def numeric_matrix(frame: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    """float64 matrix [n, len(columns)] of the named columns; text that is not a number becomes NaN.

    "Infinity" / "-Infinity" strings parse as +/-inf, as CICFlowMeter writes them.
    """
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise InvariantViolation(f"frame is missing feature columns {missing}")
    if not columns:
        return np.zeros((len(frame), 0), dtype=np.float64)
    out = np.empty((len(frame), len(columns)), dtype=np.float64)
    for j, c in enumerate(columns):
        col = frame[c]
        if pd.api.types.is_bool_dtype(col):
            out[:, j] = col.to_numpy(dtype=np.float64)
        elif pd.api.types.is_numeric_dtype(col):
            out[:, j] = col.to_numpy(dtype=np.float64, na_value=np.nan)
        else:
            out[:, j] = pd.to_numeric(col.astype(str).str.strip(), errors="coerce").to_numpy(dtype=np.float64)
    return out


@dataclass
class FiniteGuard:
    """Repair non-finite cells with per-column training statistics (see the module docstring)."""

    low: np.ndarray = field(default_factory=lambda: np.zeros(0))
    high: np.ndarray = field(default_factory=lambda: np.zeros(0))
    fill: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def fit(self, x: np.ndarray) -> FiniteGuard:
        """Estimate finite min, max and median per column; a column without finite values gets 0."""
        d = x.shape[1]
        self.low, self.high, self.fill = np.zeros(d), np.zeros(d), np.zeros(d)
        for j in range(d):
            col = x[:, j]
            fin = col[np.isfinite(col)]
            if fin.size:
                self.low[j], self.high[j], self.fill[j] = fin.min(), fin.max(), float(np.median(fin))
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Copy of x with +inf -> high, -inf -> low, NaN -> fill, column by column."""
        if x.shape[1] != self.low.shape[0]:
            raise InvariantViolation(f"guard fitted on {self.low.shape[0]} columns, got {x.shape[1]}")
        out = x.copy()
        pos, neg, nan = np.isposinf(out), np.isneginf(out), np.isnan(out)
        if pos.any() or neg.any() or nan.any():
            cols = np.broadcast_to(np.arange(out.shape[1]), out.shape)
            out[pos] = self.high[cols[pos]]
            out[neg] = self.low[cols[neg]]
            out[nan] = self.fill[cols[nan]]
        return out

    def state(self) -> dict[str, Any]:
        return {"low": self.low.tolist(), "high": self.high.tolist(), "fill": self.fill.tolist()}

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> FiniteGuard:
        return cls(np.asarray(state["low"], dtype=np.float64), np.asarray(state["high"], dtype=np.float64),
                   np.asarray(state["fill"], dtype=np.float64))


@dataclass
class Scaler:
    """Column scaling with scikit-learn semantics ("minmax", "standard") or row-wise "l2" normalisation."""

    kind: ScalerName = "none"
    offset: np.ndarray = field(default_factory=lambda: np.zeros(0))
    scale: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def fit(self, x: np.ndarray) -> Scaler:
        d = x.shape[1]
        if self.kind == "minmax":
            lo, hi = (x.min(axis=0), x.max(axis=0)) if len(x) else (np.zeros(d), np.ones(d))
            rng = hi - lo
            rng[rng == 0.0] = 1.0
            self.offset, self.scale = lo, 1.0 / rng
        elif self.kind == "standard":
            mu, sd = (x.mean(axis=0), x.std(axis=0)) if len(x) else (np.zeros(d), np.ones(d))
            sd[sd == 0.0] = 1.0
            self.offset, self.scale = mu, 1.0 / sd
        elif self.kind in ("none", "l2"):
            self.offset, self.scale = np.zeros(d), np.ones(d)
        else:
            raise ValueError(f"unknown scaler {self.kind!r}")
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if x.shape[1] != self.offset.shape[0]:
            raise InvariantViolation(f"scaler fitted on {self.offset.shape[0]} columns, got {x.shape[1]}")
        if self.kind == "l2":
            norm = np.sqrt(np.einsum("ij,ij->i", x, x))
            norm[norm == 0.0] = 1.0
            return x / norm[:, None]
        return (x - self.offset) * self.scale

    def state(self) -> dict[str, Any]:
        return {"kind": self.kind, "offset": self.offset.tolist(), "scale": self.scale.tolist()}

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> Scaler:
        return cls(state["kind"], np.asarray(state["offset"], dtype=np.float64), np.asarray(state["scale"], dtype=np.float64))


def category_text(value: Any) -> str:
    """Canonical text of a categorical value: integral floats print as integers (80.0 -> "80")."""
    if value is None:
        return ""
    if isinstance(value, float | np.floating):
        if np.isnan(value):
            return ""
        if float(value).is_integer():
            return str(int(value))
    return str(value).strip()


@dataclass
class OneHot:
    """One-hot encoding of categorical columns, categories learned on training data.

    With `max_categories`, only the most frequent categories of a column get their own indicator (ties
    broken by the category text) and every other value, including values unseen in training, sets the
    column's OTHER indicator. Without it, unseen values set no indicator (scikit-learn's
    handle_unknown="ignore").
    """

    columns: tuple[str, ...] = ()
    max_categories: int | None = None
    categories: dict[str, list[str]] = field(default_factory=dict)

    def fit(self, frame: pd.DataFrame) -> OneHot:
        self.categories = {}
        for c in self.columns:
            counts = pd.Series([category_text(v) for v in frame[c].tolist()]).value_counts()
            ranked = sorted(counts.items(), key=lambda kv: (-int(kv[1]), kv[0]))
            names = [k for k, _ in ranked]
            if self.max_categories is not None and len(names) > self.max_categories:
                names = sorted(names[: self.max_categories]) + [OTHER_CATEGORY]
            else:
                names = sorted(names)
            self.categories[c] = names
        return self

    @property
    def width(self) -> int:
        return sum(len(v) for v in self.categories.values())

    def feature_names(self) -> list[str]:
        return [f"{c}={v}" for c in self.columns for v in self.categories[c]]

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        out = np.zeros((len(frame), self.width), dtype=np.float64)
        start = 0
        for c in self.columns:
            names = self.categories[c]
            index = {v: i for i, v in enumerate(names)}
            other = index.get(OTHER_CATEGORY)
            codes = np.asarray([index.get(category_text(v), -1 if other is None else other) for v in frame[c].tolist()],
                               dtype=np.int64)
            rows = np.nonzero(codes >= 0)[0]
            out[rows, start + codes[rows]] = 1.0
            start += len(names)
        return out

    def state(self) -> dict[str, Any]:
        return {"columns": list(self.columns), "max_categories": self.max_categories, "categories": self.categories}

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> OneHot:
        return cls(tuple(state["columns"]), state["max_categories"], {k: list(v) for k, v in state["categories"].items()})


@dataclass
class ColumnPipeline:
    """Numeric and categorical columns -> one float64 design matrix, fitted on training rows.

    Steps: numeric coercion, finite repair (FiniteGuard), one-hot encoding of categorical columns, and
    scaling. "minmax" and "standard" scale the numeric block only (indicators stay 0/1, as in a
    scikit-learn ColumnTransformer); "l2" normalises the whole row, the definition of a row norm.
    """

    numeric: tuple[str, ...]
    categorical: tuple[str, ...] = ()
    scaler: ScalerName = "none"
    max_categories: int | None = None
    guard: FiniteGuard = field(default_factory=FiniteGuard)
    numeric_scaler: Scaler = field(default_factory=Scaler)
    onehot: OneHot = field(default_factory=OneHot)
    fitted: bool = False

    def nonfinite_rows(self, frame: pd.DataFrame) -> np.ndarray:
        """Rows with at least one non-finite numeric feature (before repair)."""
        x = numeric_matrix(frame, self.numeric)
        return ~np.all(np.isfinite(x), axis=1) if x.shape[1] else np.zeros(len(frame), dtype=bool)

    def fit(self, frame: pd.DataFrame) -> ColumnPipeline:
        x = numeric_matrix(frame, self.numeric)
        self.guard = FiniteGuard().fit(x)
        x = self.guard.transform(x)
        self.numeric_scaler = Scaler("minmax" if self.scaler == "minmax" else "standard" if self.scaler == "standard" else "none").fit(x)
        self.onehot = OneHot(self.categorical, self.max_categories).fit(frame) if self.categorical else OneHot()
        self.fitted = True
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if not self.fitted:
            raise InvariantViolation("ColumnPipeline.transform called before fit")
        x = self.numeric_scaler.transform(self.guard.transform(numeric_matrix(frame, self.numeric)))
        if self.categorical:
            x = np.concatenate([x, self.onehot.transform(frame)], axis=1)
        if self.scaler == "l2":
            x = Scaler("l2").fit(x).transform(x)
        return x

    def feature_names(self) -> list[str]:
        return [*self.numeric, *(self.onehot.feature_names() if self.categorical else [])]

    def state(self) -> dict[str, Any]:
        return {"numeric": list(self.numeric), "categorical": list(self.categorical), "scaler": self.scaler,
                "max_categories": self.max_categories, "guard": self.guard.state(),
                "numeric_scaler": self.numeric_scaler.state(), "onehot": self.onehot.state(), "fitted": self.fitted}

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> ColumnPipeline:
        return cls(tuple(state["numeric"]), tuple(state["categorical"]), state["scaler"], state["max_categories"],
                   FiniteGuard.from_state(state["guard"]), Scaler.from_state(state["numeric_scaler"]),
                   OneHot.from_state(state["onehot"]), bool(state["fitted"]))
