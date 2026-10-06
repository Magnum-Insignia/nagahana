"""FieldCodec: how the learned Generator families see a cell's value (fitted on the training split only).

Purpose
-------
The masked-generative model predicts a *class* per cell (MaskGIT works on discrete codes, Chang et al.
CVPR 2022, arXiv:2202.04200); the diffusion model denoises *standardised reals* (TabDDPM's Gaussian
part, Kotelnikov et al. ICML 2023, arXiv:2209.15421). The codec maps the value/status matrices of
`ColumnarUpdates` into both forms and back. It is fitted on real training-split windows only (P-23 via
AS-367); `fitted_on` records which samples, so `pipeline/splits.validate(generator_training_ids=…)`
can check it.

Maths (AS-362, AS-365)
----------------------
Numeric columns (CONTINUOUS, COUNT, HISTOGRAM bins), in signed-log1p space x̃ = sign(x)·log(1 + |x|)
(the input layer's and Decoder's space, AS-31):

    code(x)  = clip(⌊(x̃ − lo_c) / (hi_c − lo_c) · K⌋, 0, K − 1)               K = value_bins
    value(k) = expm1_s(lo_c + (k + u)(hi_c − lo_c)/K),  u ~ U(0, 1)            (uniform dequantisation)
    std(x)   = (x̃ − μ_c) / σ_c,     value(z) = expm1_s(μ_c + σ_c z)            (diffusion space)

with lo_c, hi_c the min/max and μ_c, σ_c the mean/std of x̃ over contributing training cells. COUNT and
histogram values are rounded to integers ≥ 0 on decoding (a count is an integer: hard limit "integral").

Discrete columns (CATEGORICAL, BITMASK): the `cat_vocab − 1` most frequent training codes get a class
each; class `cat_vocab − 1` is "other" (OOV). Decoding "other" draws a code from the empirical
distribution of the rare training codes, so a decoded value is always one that was seen in training.
If no rare codes exist, the OOV class is disallowed (`class_mask`).

Status is never encoded as a value: excluded cells get code −1 and are never predicted or decoded
(D-41). The learned families only regenerate *contributing* cells and keep every status.

Invariants: `decode(encode(x))` returns a value in the same bin as x (numeric) or x itself (discrete,
in-vocabulary); excluded cells stay NaN.
Extension point: a learned (rather than uniform) bin layout, e.g. quantile bins, would only change
`_edges`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from nagahana.core.errors import InvariantViolation
from nagahana.datamodel.columnar import STATUS_CODE, Column, ColumnarUpdates
from nagahana.datamodel.fields import Kind
from nagahana.datamodel.status import CONTRIBUTING

_CONTRIB_CODES = np.array([STATUS_CODE[s] for s in CONTRIBUTING], dtype=np.uint8)
NUMERIC_KINDS = frozenset({Kind.CONTINUOUS, Kind.COUNT, Kind.HISTOGRAM})
DISCRETE_KINDS = frozenset({Kind.CATEGORICAL, Kind.BITMASK})


def slog(x: np.ndarray) -> np.ndarray:
    """sign(x)·log(1 + |x|) (NumPy form of nn.numeric.signed_log1p)."""
    return np.sign(x) * np.log1p(np.abs(x))


def sexp(y: np.ndarray) -> np.ndarray:
    """Inverse of `slog`."""
    return np.sign(y) * np.expm1(np.abs(y))


@dataclass(frozen=True)
class NumericSpec:
    """Signed-log1p range and moments of one numeric column over training cells. seen = had data."""

    lo: float
    hi: float
    mean: float
    std: float
    seen: bool


@dataclass(frozen=True)
class DiscreteSpec:
    """Vocabulary of one discrete column: kept codes (by frequency) and the OOV pool."""

    codes: np.ndarray        # int64 [V_kept]
    oov_codes: np.ndarray    # int64 [n_rare]
    oov_probs: np.ndarray    # float64 [n_rare], sums to 1 (empty if no rare codes)
    seen: bool


class FieldCodec:
    """Value ⇄ class / standardised-real maps for every column. See the module docstring."""

    def __init__(self, columns: Sequence[Column], numeric: dict[int, NumericSpec], discrete: dict[int, DiscreteSpec],
                 *, value_bins: int, cat_vocab: int, fitted_on: Sequence[str]) -> None:
        self.columns = tuple(columns)
        self.numeric = dict(numeric)
        self.discrete = dict(discrete)
        self.value_bins, self.cat_vocab = int(value_bins), int(cat_vocab)
        self.fitted_on = tuple(fitted_on)
        self.numeric_columns = sorted(self.numeric)              # column indices of numeric columns
        self.discrete_columns = sorted(self.discrete)

    # ------------------------------------------------------------------ fitting
    @classmethod
    def fit(cls, tables: Sequence[ColumnarUpdates], *, value_bins: int, cat_vocab: int,
            fitted_on: Sequence[str]) -> FieldCodec:
        """Fit on real training-split windows (all with the same columns)."""
        if not tables:
            raise InvariantViolation("the codec needs at least one training window")
        if value_bins < 2 or cat_vocab < 2:
            raise InvariantViolation("value_bins and cat_vocab must be ≥ 2")
        cols = tables[0].columns
        if any(t.columns != cols for t in tables):
            raise InvariantViolation("all training windows must share one column layout")
        values = np.vstack([t.values for t in tables])
        contrib = np.vstack([np.isin(t.status, _CONTRIB_CODES) for t in tables])
        numeric: dict[int, NumericSpec] = {}
        discrete: dict[int, DiscreteSpec] = {}
        for j, c in enumerate(cols):
            v = values[contrib[:, j], j]                      # contributing training cells only (D-41)
            if c.kind in NUMERIC_KINDS:
                if v.size == 0:
                    numeric[j] = NumericSpec(0.0, 1.0, 0.0, 1.0, seen=False)
                    continue
                y = slog(v)
                lo, hi = float(y.min()), float(y.max())
                numeric[j] = NumericSpec(lo, max(hi, lo + 1e-6), float(y.mean()), float(max(y.std(), 1e-3)), seen=True)
            elif c.kind in DISCRETE_KINDS:
                if v.size == 0:
                    discrete[j] = DiscreteSpec(np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0), seen=False)
                    continue
                codes, counts = np.unique(np.round(v).astype(np.int64), return_counts=True)
                order = np.argsort(-counts, kind="stable")
                kept, rare = order[: cat_vocab - 1], order[cat_vocab - 1:]
                probs = counts[rare] / counts[rare].sum() if rare.size else np.zeros(0)
                discrete[j] = DiscreteSpec(codes[kept], codes[rare], probs.astype(np.float64), seen=True)
        return cls(cols, numeric, discrete, value_bins=value_bins, cat_vocab=cat_vocab, fitted_on=fitted_on)

    # ------------------------------------------------------------------ class view (masked model)
    def n_classes(self, j: int) -> int:
        """Number of classes of column j (numeric: value_bins; discrete: kept codes + OOV)."""
        if j in self.numeric:
            return self.value_bins
        if j in self.discrete:
            return self.cat_vocab
        raise KeyError(f"column {j} is not modelled")

    def class_mask(self, j: int) -> np.ndarray:
        """bool [n_classes(j)]: classes that can be decoded (unused vocabulary slots and empty OOV are not)."""
        k = self.n_classes(j)
        if j in self.numeric:
            return np.ones(k, dtype=bool)
        d = self.discrete[j]
        m = np.zeros(k, dtype=bool)
        m[: len(d.codes)] = True
        m[k - 1] = d.oov_codes.size > 0
        return m

    def modelled(self, j: int) -> bool:
        """True if column j had training data (only such columns are regenerated)."""
        spec = self.numeric.get(j) or self.discrete.get(j)
        return bool(spec is not None and spec.seen)

    def check_columns(self, columns: Sequence[Column]) -> None:
        if tuple(columns) != self.columns:
            raise InvariantViolation("window columns differ from the codec's (refit the codec on this layout)")

    def encode(self, values: np.ndarray, status: np.ndarray) -> np.ndarray:
        """Codes int64 [N, C]; −1 for excluded cells and for columns that are not modelled."""
        contrib = np.isin(status, _CONTRIB_CODES)
        out = np.full(values.shape, -1, dtype=np.int64)
        for j, s in self.numeric.items():
            m = contrib[:, j]
            y = slog(np.where(m, values[:, j], 0.0))
            k = np.floor((y - s.lo) / (s.hi - s.lo) * self.value_bins)
            out[m, j] = np.clip(k, 0, self.value_bins - 1).astype(np.int64)[m]
        for j, d in self.discrete.items():
            m = contrib[:, j]
            v = np.round(np.where(m, values[:, j], 0.0)).astype(np.int64)
            lookup = {int(c): i for i, c in enumerate(d.codes.tolist())}
            cls = np.array([lookup.get(int(x), self.cat_vocab - 1) for x in v], dtype=np.int64)
            out[m, j] = cls[m]
        return out

    def decode(self, codes: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Values float64 [N, C] from codes (NaN where code < 0). Dequantised; counts rounded."""
        out = np.full(codes.shape, np.nan, dtype=np.float64)
        for j, s in self.numeric.items():
            m = codes[:, j] >= 0
            u = rng.random(int(m.sum()))
            y = s.lo + (codes[m, j] + u) * (s.hi - s.lo) / self.value_bins
            x = sexp(y)
            if self.columns[j].kind in (Kind.COUNT, Kind.HISTOGRAM):
                x = np.round(np.maximum(x, 0.0))
            out[m, j] = x
        for j, d in self.discrete.items():
            m = codes[:, j] >= 0
            k = codes[m, j]
            vals = np.empty(k.shape, dtype=np.float64)
            inv = k < len(d.codes)
            vals[inv] = d.codes[k[inv]]
            n_oov = int((~inv).sum())
            if n_oov:
                if d.oov_codes.size == 0:
                    raise InvariantViolation(f"column {self.columns[j].name}: OOV class decoded but no rare codes exist")
                vals[~inv] = rng.choice(d.oov_codes, size=n_oov, p=d.oov_probs)
            out[m, j] = vals
        return out

    # ------------------------------------------------------------------ standardised view (diffusion)
    def to_std(self, values: np.ndarray, status: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(z float32 [N, Cn], contributing bool [N, Cn]) over `numeric_columns`; z = 0 where excluded."""
        contrib = np.isin(status, _CONTRIB_CODES)[:, self.numeric_columns]
        z = np.zeros((values.shape[0], len(self.numeric_columns)), dtype=np.float32)
        for k, j in enumerate(self.numeric_columns):
            s = self.numeric[j]
            y = slog(np.where(contrib[:, k], values[:, j], 0.0))
            z[:, k] = np.where(contrib[:, k], (y - s.mean) / s.std, 0.0)
        return z, contrib

    def from_std(self, z: np.ndarray) -> np.ndarray:
        """Values float64 [N, Cn] (numeric columns order) from standardised reals; counts rounded."""
        out = np.empty(z.shape, dtype=np.float64)
        for k, j in enumerate(self.numeric_columns):
            s = self.numeric[j]
            x = sexp(s.mean + s.std * z[:, k].astype(np.float64))
            if self.columns[j].kind in (Kind.COUNT, Kind.HISTOGRAM):
                x = np.round(np.maximum(x, 0.0))
            out[:, k] = x
        return out
