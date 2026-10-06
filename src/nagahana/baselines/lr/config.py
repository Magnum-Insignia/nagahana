"""Configuration of the logistic-regression baseline family (dataclasses and YAML I/O).

Every model of the family reads one frozen dataclass. The YAML files in conf/baselines/lr/ hold the
same tree; `load_config` builds the dataclasses from a file, rejecting unknown keys and values of the
wrong type, and `config_to_dict` writes them back, so the configuration of every fitted model can be
stored next to its weights and reproduced exactly.

Defaults stand for the protocol of the evaluation chapter (logistic regression with l2 regularisation
and class weighting, strengths chosen on held-out data in time, threshold and calibration on the
validation split) and for the assumptions AS-500 ... AS-529 (docs/assumptions/lr-baseline.md). The
windowing quantities (cadence w, horizon K, entity caps) are not repeated here: they come from the
NagaHana configuration that built the windows, so both models see the same triggers and targets.

Grids are log-spaced: `GridSpec(log10_min, log10_max, num)` gives `num` values from 10^log10_max down
to 10^log10_min (descending, so a regularisation path can be warm-started from the most regularised
fit).
"""

from __future__ import annotations

import dataclasses
import types
import typing
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

import numpy as np

from nagahana.core.config import MISSING
from nagahana.core.errors import ConfigMissing, InvariantViolation

T = TypeVar("T")

#: The members of the family, by the prediction record each produces.
TASK_NAMES: tuple[str, ...] = ("detection", "forecast", "stage", "state_forecast")

#: Values a selection criterion may take, per task.
DETECTION_CRITERIA: frozenset[str] = frozenset({"average_precision", "log_loss", "balanced_log_loss", "brier", "auroc"})
HAZARD_CRITERIA: frozenset[str] = frozenset({"survival_nll", "brier_k"})
STAGE_CRITERIA: frozenset[str] = frozenset({"balanced_log_loss", "log_loss", "macro_f1", "top1"})
RIDGE_CRITERIA: frozenset[str] = frozenset({"mse"})


@dataclass(frozen=True)
class GridSpec:
    """A log-spaced grid of positive values, returned in descending order."""

    log10_min: float
    log10_max: float
    num: int

    def __post_init__(self) -> None:
        if self.num < 1:
            raise InvariantViolation("a grid needs at least one value")
        if self.log10_min > self.log10_max:
            raise InvariantViolation("log10_min must not exceed log10_max")

    def values(self) -> tuple[float, ...]:
        """The grid values, largest first."""
        if self.num == 1:
            return (float(10.0 ** self.log10_max),)
        return tuple(float(v) for v in np.logspace(self.log10_max, self.log10_min, self.num))


@dataclass(frozen=True)
class FeatureConfig:
    """How the canonical columns of a window become design columns (module docstring of features.py)."""

    numeric_clip: float = 50.0               # |signed_log1p(x)| clip; the FieldEncoder's max_log_magnitude (AS-100)
    max_bits: int = 16                       # bits of a bitmask column; the FieldEncoder's max_bits
    max_categories: int = 32                 # vocabulary cap per categorical column, per-update design (AS-503)
    max_categories_aggregate: int = 16       # vocabulary cap per categorical column, window aggregates (AS-503)
    min_category_count: int = 5              # training occurrences a code needs for its own column (AS-503)
    drop_constant: bool = True               # drop columns constant on the training rows (AS-505)
    log_count_states: bool = True            # signed log1p on the count-type window states (AS-509)

    def __post_init__(self) -> None:
        if self.numeric_clip <= 0 or self.max_bits < 1 or self.max_bits > 62:
            raise InvariantViolation("numeric_clip must be > 0 and max_bits in 1 ... 62")
        if self.max_categories < 1 or self.max_categories_aggregate < 1 or self.min_category_count < 1:
            raise InvariantViolation("category caps and min_category_count must be >= 1")


@dataclass(frozen=True)
class StandardiserConfig:
    """Location and scale of every design column, fitted on training rows only (AS-504)."""

    numeric: str = "robust"                  # "robust" (median, IQR / 1.349) or "zscore" (mean, standard deviation)
    robust_scale: str = "iqr"                # "iqr" or "mad" (1.4826 * median absolute deviation)

    def __post_init__(self) -> None:
        if self.numeric not in ("robust", "zscore"):
            raise InvariantViolation(f"unknown standardiser {self.numeric!r}")
        if self.robust_scale not in ("iqr", "mad"):
            raise InvariantViolation(f"unknown robust scale {self.robust_scale!r}")


@dataclass(frozen=True)
class SolverConfig:
    """Optimiser of a convex objective (logistic.py)."""

    method: str = "lbfgs"                    # "lbfgs" (in memory), "streamed_lbfgs" (exact, out of core), "minibatch"
    max_iter: int = 1000                     # L-BFGS iterations
    tol_grad: float = 1e-6                   # stop when max |grad F| <= tol_grad (F is a mean loss; AS-529)
    tol_rel_obj: float = 1e-10               # stop when |f_prev - f| <= tol_rel_obj * max(1, |f|) between checks
    history_size: int = 20                   # L-BFGS memory
    check_every: int = 10                    # L-BFGS iterations between convergence checks
    chunk_rows: int = 65_536                 # rows per chunk of a streamed pass
    batch_size: int = 4096                   # minibatch: rows per stochastic step
    epochs: int = 10                         # minibatch: passes over the training rows
    step0: float = 1.0                       # minibatch: initial step size eta_0
    decay: float = 0.75                      # minibatch: eta_t = eta_0 / (1 + eta_0 * lambda_eff * t) ** decay
    shuffle_buffer: int = 262_144            # minibatch: rows held for shuffling a sequential stream
    polish_iter: int = 200                   # minibatch: streamed L-BFGS iterations after the stochastic phase
    precondition: bool = True                # L-BFGS in the variables of Boehning's Hessian bound (precondition.py)
    precond_rows: int = 100_000              # rows sampled (seeded) for the Gram matrix of the bound

    def __post_init__(self) -> None:
        if self.method not in ("lbfgs", "streamed_lbfgs", "minibatch"):
            raise InvariantViolation(f"unknown solver {self.method!r}")
        if min(self.max_iter, self.history_size, self.check_every, self.chunk_rows, self.batch_size,
               self.epochs, self.shuffle_buffer, self.precond_rows) < 1 or self.polish_iter < 0:
            raise InvariantViolation("solver sizes must be >= 1 (polish_iter >= 0)")
        if not (0.5 < self.decay <= 1.0) or self.step0 <= 0:
            raise InvariantViolation("decay must lie in (0.5, 1] (Robbins-Monro) and step0 must be > 0")
        if self.tol_grad < 0 or self.tol_rel_obj < 0:
            raise InvariantViolation("tolerances must be >= 0")


@dataclass(frozen=True)
class SelectionConfig:
    """How regularisation and class-weight strengths are chosen (temporal_cv.py; AS-511, AS-512)."""

    method: str = "temporal_cv"              # "temporal_cv", "validation" or "fixed"
    l2: GridSpec = field(default_factory=lambda: GridSpec(-7.0, 0.0, 15))
    class_weight_power: tuple[float, ...] = (0.0, 0.5, 1.0)   # c_y = (n / (2 n_y)) ** power; 1 = balanced
    fixed_l2: float | None = None            # method "fixed"
    fixed_power: float | None = None         # method "fixed"
    n_blocks: int = 5                        # time blocks per group for forward chaining
    min_train_blocks: int = 2                # blocks before the first validation block
    max_train_blocks: int | None = None      # None: expanding window; else sliding window of this many blocks
    embargo_windows: float = 1.0             # gap before (and after) a validation block, in cadence windows
    rule: str = "best"                       # "best" or "one_se" (one-standard-error rule)
    group_by: str = "network"                # blocks are formed per "network" or per "source"

    def __post_init__(self) -> None:
        if self.method not in ("temporal_cv", "validation", "fixed"):
            raise InvariantViolation(f"unknown selection method {self.method!r}")
        if self.method == "fixed" and (self.fixed_l2 is None or self.fixed_power is None):
            raise ConfigMissing("selection method 'fixed' needs fixed_l2 and fixed_power")
        if self.fixed_l2 is not None and self.fixed_l2 < 0:
            raise InvariantViolation("fixed_l2 must be >= 0")
        if not self.class_weight_power or any(p < 0 for p in self.class_weight_power):
            raise InvariantViolation("class_weight_power needs at least one value, all >= 0")
        if self.n_blocks < 2 or not 1 <= self.min_train_blocks < self.n_blocks:
            raise InvariantViolation("need n_blocks >= 2 and 1 <= min_train_blocks < n_blocks")
        if self.max_train_blocks is not None and self.max_train_blocks < 1:
            raise InvariantViolation("max_train_blocks must be >= 1")
        if self.embargo_windows < 0:
            raise InvariantViolation("embargo_windows must be >= 0")
        if self.rule not in ("best", "one_se"):
            raise InvariantViolation(f"unknown selection rule {self.rule!r}")
        if self.group_by not in ("network", "source"):
            raise InvariantViolation(f"unknown group_by {self.group_by!r}")

    def grid(self) -> list[tuple[float, float]]:
        """(l2, class-weight power) pairs to evaluate; l2 descending inside each power."""
        if self.method == "fixed":
            assert self.fixed_l2 is not None and self.fixed_power is not None
            return [(float(self.fixed_l2), float(self.fixed_power))]
        return [(lam, float(p)) for p in self.class_weight_power for lam in self.l2.values()]


@dataclass(frozen=True)
class ThresholdConfig:
    """Operating thresholds chosen on validation units (thresholds.py; AS-514)."""

    rule: str = "max_f1"                     # the model's own threshold: "max_f1", "fixed_fpr" or "youden"
    alpha: float = 0.001                     # false-positive rate of the conformal threshold (0.10 %)

    def __post_init__(self) -> None:
        if self.rule not in ("max_f1", "fixed_fpr", "youden"):
            raise InvariantViolation(f"unknown threshold rule {self.rule!r}")
        if not 0.0 < self.alpha < 1.0:
            raise InvariantViolation("alpha must lie in (0, 1)")


@dataclass(frozen=True)
class CalibrationConfig:
    """Probability calibration fitted on validation units (calibration.py; AS-515)."""

    method: str = "platt"                    # "platt", "isotonic", "temperature" or "none"

    def __post_init__(self) -> None:
        if self.method not in ("platt", "isotonic", "temperature", "none"):
            raise InvariantViolation(f"unknown calibration method {self.method!r}")


@dataclass(frozen=True)
class DetectorConfig:
    """Detection LR on state updates (detector.py)."""

    context_lags: int = 0                    # previous cadence windows aggregated beside each update (0 = static, AS-525)
    criterion: str = "average_precision"     # AS-513
    cross_check: bool = False                # refit with scikit-learn ([baselines] extra) and assert agreement
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)
    threshold: ThresholdConfig = field(default_factory=ThresholdConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)

    def __post_init__(self) -> None:
        if self.context_lags < 0:
            raise InvariantViolation("context_lags must be >= 0")
        if self.criterion not in DETECTION_CRITERIA:
            raise InvariantViolation(f"unknown detection criterion {self.criterion!r}")


@dataclass(frozen=True)
class HazardConfig:
    """Discrete-time hazard LR, the static infiltration forecaster (hazard.py)."""

    lags: int = 0                            # cadence windows before the trigger window (0 = static forecaster, AS-525)
    coefficients: str = "shared"             # "shared": logit h_j = alpha_j + w.x; "per_horizon": alpha_j + w_j.x
    criterion: str = "survival_nll"
    calibration: str = "platt"               # "platt" (on the hazard logit, survival likelihood) or "none"
    risk: str = "neg_rmst"                   # time-to-event risk score: "neg_rmst" or "p_inf_k"
    threshold: ThresholdConfig = field(default_factory=ThresholdConfig)   # own alert threshold on P_inf(K)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)

    def __post_init__(self) -> None:
        if self.lags < 0:
            raise InvariantViolation("lags must be >= 0")
        if self.coefficients not in ("shared", "per_horizon"):
            raise InvariantViolation(f"unknown hazard coefficients {self.coefficients!r}")
        if self.criterion not in HAZARD_CRITERIA:
            raise InvariantViolation(f"unknown hazard criterion {self.criterion!r}")
        if self.calibration not in ("platt", "none"):
            raise InvariantViolation(f"unknown hazard calibration {self.calibration!r}")
        if self.risk not in ("neg_rmst", "p_inf_k"):
            raise InvariantViolation(f"unknown risk score {self.risk!r}")


@dataclass(frozen=True)
class StageConfig:
    """Multinomial (softmax) stage LR over the ATT&CK stage vocabulary (stage.py)."""

    target: str = "step"                     # "step": stage of each future step k; "update": stage of the update
    lags: int = 0                            # step target: cadence windows before the trigger window
    coefficients: str = "shared"             # step target: "shared" slopes with per-step intercepts, or "per_horizon"
    criterion: str = "balanced_log_loss"
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)

    def __post_init__(self) -> None:
        if self.target not in ("step", "update"):
            raise InvariantViolation(f"unknown stage target {self.target!r}")
        if self.lags < 0:
            raise InvariantViolation("lags must be >= 0")
        if self.coefficients not in ("shared", "per_horizon"):
            raise InvariantViolation(f"unknown stage coefficients {self.coefficients!r}")
        if self.criterion not in STAGE_CRITERIA:
            raise InvariantViolation(f"unknown stage criterion {self.criterion!r}")


@dataclass(frozen=True)
class RidgeConfig:
    """Ridge next-state forecaster on the same features and their lags (ridge.py)."""

    lags: tuple[int, ...] = (0, 1, 2, 3)     # lag counts compared by the selection
    l2: GridSpec = field(default_factory=lambda: GridSpec(-4.0, 4.0, 17))
    criterion: str = "mse"
    selection: SelectionConfig = field(default_factory=lambda: SelectionConfig(class_weight_power=(0.0,)))

    def __post_init__(self) -> None:
        if not self.lags or any(lag < 0 for lag in self.lags) or len(set(self.lags)) != len(self.lags):
            raise InvariantViolation("ridge lags must be distinct non-negative integers")
        if self.criterion not in RIDGE_CRITERIA:
            raise InvariantViolation(f"unknown ridge criterion {self.criterion!r}")


@dataclass(frozen=True)
class LRBaselineConfig:
    """The whole family."""

    name: str = "logistic_regression"
    seed: int = 0
    features: FeatureConfig = field(default_factory=FeatureConfig)
    standardiser: StandardiserConfig = field(default_factory=StandardiserConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    hazard: HazardConfig = field(default_factory=HazardConfig)
    stage: StageConfig = field(default_factory=StageConfig)
    ridge: RidgeConfig = field(default_factory=RidgeConfig)

    def max_trigger_lags(self) -> int:
        """Largest number of lag windows any trigger-level model reads."""
        return max(self.hazard.lags, self.stage.lags if self.stage.target == "step" else 0, max(self.ridge.lags))


def config_to_dict(cfg: Any) -> dict[str, Any]:
    """A dataclass tree as plain Python containers (tuples become lists), for YAML and JSON."""

    def conv(v: Any) -> Any:
        if dataclasses.is_dataclass(v) and not isinstance(v, type):
            return {f.name: conv(getattr(v, f.name)) for f in dataclasses.fields(v)}
        if isinstance(v, tuple | list):
            return [conv(x) for x in v]
        return v

    out = conv(cfg)
    assert isinstance(out, dict)
    return out


def _coerce(tp: Any, value: Any, where: str) -> Any:
    """Convert a YAML value to the annotated type `tp`, raising on a mismatch."""
    if isinstance(value, str) and value == MISSING:
        raise ConfigMissing(f"{where} is '???' (undecided); give a value or remove the key to use the default")
    origin = typing.get_origin(tp)
    args = typing.get_args(tp)
    if origin in (typing.Union, types.UnionType):
        if value is None and type(None) in args:
            return None
        errors = []
        for a in args:
            if a is type(None):
                continue
            try:
                return _coerce(a, value, where)
            except (TypeError, InvariantViolation, ValueError) as exc:
                errors.append(str(exc))
        raise TypeError(f"{where}: {value!r} matches none of {args} ({'; '.join(errors)})")
    if origin is tuple:
        if not isinstance(value, list | tuple):
            raise TypeError(f"{where}: expected a list, got {type(value).__name__}")
        inner = args[0] if args else Any
        return tuple(_coerce(inner, v, f"{where}[{i}]") for i, v in enumerate(value))
    if isinstance(tp, type) and dataclasses.is_dataclass(tp):
        if not isinstance(value, Mapping):
            raise TypeError(f"{where}: expected a mapping, got {type(value).__name__}")
        return from_dict(tp, value, where=where)
    if tp is bool:
        if not isinstance(value, bool):
            raise TypeError(f"{where}: expected true/false, got {value!r}")
        return value
    if tp is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{where}: expected an integer, got {value!r}")
        return int(value)
    if tp is float:
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise TypeError(f"{where}: expected a number, got {value!r}")
        return float(value)
    if tp is str:
        if not isinstance(value, str):
            raise TypeError(f"{where}: expected a string, got {value!r}")
        return value
    if tp is Any:
        return value
    raise TypeError(f"{where}: unsupported annotation {tp!r}")


def from_dict(cls: type[T], data: Mapping[str, Any], *, where: str = "lr") -> T:
    """Build dataclass `cls` from a mapping; missing keys keep their defaults, unknown keys raise."""
    if not dataclasses.is_dataclass(cls):
        raise TypeError(f"{cls!r} is not a dataclass")
    hints = typing.get_type_hints(cls)
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = sorted(set(data) - names)
    if unknown:
        raise InvariantViolation(f"{where}: unknown keys {unknown}; known: {sorted(names)}")
    kwargs = {k: _coerce(hints[k], v, f"{where}.{k}") for k, v in data.items()}
    return cls(**kwargs)


def load_config(path: str | Path) -> LRBaselineConfig:
    """Read a YAML file (conf/baselines/lr/*.yaml) into an `LRBaselineConfig`."""
    from nagahana.core.config import load_yaml

    data = load_yaml(path)
    root = data.get("lr", data)
    if not isinstance(root, Mapping):
        raise InvariantViolation(f"{path}: the 'lr' section must be a mapping")
    return from_dict(LRBaselineConfig, root)


__all__ = [
    "TASK_NAMES", "CalibrationConfig", "DetectorConfig", "FeatureConfig", "GridSpec", "HazardConfig", "LRBaselineConfig",
    "RidgeConfig", "SelectionConfig", "SolverConfig", "StageConfig", "StandardiserConfig", "ThresholdConfig",
    "config_to_dict", "from_dict", "load_config",
]
