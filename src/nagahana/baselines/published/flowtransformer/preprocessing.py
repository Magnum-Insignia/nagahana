"""FlowTransformer's standard pre-processing, fitted on the training rows only (the paper fits
pre-processing on the training data, baselines-notes.md, B4).

Numeric field c with training minimum m_c and range r_c = max_c - m_c:

    x' = log(max(x - m_c, 0) + 1) / log(r_c + 1)        (x' = 0 when r_c = 0)

so the training values map onto [0, 1] on a logarithmic scale (heavy-tailed byte and packet counts are
compressed); values below the training minimum are clamped at the minimum before the logarithm (the
logarithm of a negative shift is undefined), values above the maximum exceed 1 unless `clip` is set.

Categorical field c: the `n_levels` most frequent training values (ties by their text) get codes 1 ...
n_levels; every other value, including values unseen in training, gets code 0.

Both rules follow the framework's StandardPreProcessing as read for this reproduction (AS-542; to verify
against the official repository).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from nagahana.baselines.published.preprocessing import FiniteGuard, category_text, numeric_matrix
from nagahana.core.errors import InvariantViolation


@dataclass
class FlowTransformerPreprocessing:
    """Numeric log-min-max scaling and top-n categorical level coding."""

    numeric: tuple[str, ...]
    categorical: tuple[str, ...]
    n_levels: int = 32
    clip: bool = False
    guard: FiniteGuard = field(default_factory=FiniteGuard)
    minimum: np.ndarray = field(default_factory=lambda: np.zeros(0))
    span: np.ndarray = field(default_factory=lambda: np.zeros(0))
    levels: dict[str, list[str]] = field(default_factory=dict)
    fitted: bool = False

    def fit(self, frame: pd.DataFrame) -> FlowTransformerPreprocessing:
        if self.n_levels < 1:
            raise InvariantViolation("n_levels must be >= 1")
        x = numeric_matrix(frame, self.numeric)
        self.guard = FiniteGuard().fit(x)
        x = self.guard.transform(x)
        self.minimum = x.min(axis=0) if len(x) else np.zeros(len(self.numeric))
        self.span = (x.max(axis=0) - self.minimum) if len(x) else np.zeros(len(self.numeric))
        self.levels = {}
        for c in self.categorical:
            counts = pd.Series([category_text(v) for v in frame[c].tolist()]).value_counts()
            ranked = sorted(counts.items(), key=lambda kv: (-int(kv[1]), kv[0]))
            self.levels[c] = [k for k, _ in ranked[: self.n_levels]]
        self.fitted = True
        return self

    @property
    def level_counts(self) -> list[int]:
        """Number of codes per categorical field (levels + the code 0 for other / unseen)."""
        return [len(self.levels[c]) + 1 for c in self.categorical]

    def transform_numeric(self, frame: pd.DataFrame) -> np.ndarray:
        """float32 [n, len(numeric)] scaled numeric fields."""
        if not self.fitted:
            raise InvariantViolation("FlowTransformerPreprocessing used before fit")
        x = self.guard.transform(numeric_matrix(frame, self.numeric))
        shifted = np.maximum(x - self.minimum, 0.0)
        denom = np.log(self.span + 1.0)
        out = np.divide(np.log(shifted + 1.0), denom, out=np.zeros_like(shifted), where=denom > 0)
        if self.clip:
            out = np.clip(out, 0.0, 1.0)
        return out.astype(np.float32)

    def transform_categorical(self, frame: pd.DataFrame) -> np.ndarray:
        """int64 [n, len(categorical)] level codes (0 = other / unseen)."""
        if not self.fitted:
            raise InvariantViolation("FlowTransformerPreprocessing used before fit")
        out = np.zeros((len(frame), len(self.categorical)), dtype=np.int64)
        for j, c in enumerate(self.categorical):
            index = {v: i + 1 for i, v in enumerate(self.levels[c])}
            out[:, j] = [index.get(category_text(v), 0) for v in frame[c].tolist()]
        return out

    def state(self) -> dict[str, Any]:
        return {"numeric": list(self.numeric), "categorical": list(self.categorical), "n_levels": self.n_levels,
                "clip": self.clip, "guard": self.guard.state(), "minimum": self.minimum.tolist(), "span": self.span.tolist(),
                "levels": self.levels, "fitted": self.fitted}

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> FlowTransformerPreprocessing:
        return cls(tuple(state["numeric"]), tuple(state["categorical"]), int(state["n_levels"]), bool(state["clip"]),
                   FiniteGuard.from_state(state["guard"]), np.asarray(state["minimum"], dtype=np.float64),
                   np.asarray(state["span"], dtype=np.float64), {k: list(v) for k, v in state["levels"].items()},
                   bool(state["fitted"]))
