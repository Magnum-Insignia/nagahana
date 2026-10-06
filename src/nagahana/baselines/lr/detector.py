"""Detection LR: labels the current state update as malicious or benign (the required baseline).

Unit and label. One state update; the label is its `malicious` flag (1, 0, or unknown, which is never
trained on and is reported as -1). Features: the per-update design of features.py, optionally with the
window blocks of the `context_lags` previous cadence windows of its record (0 by default: the static
detector of the evaluation chapter).

Protocol (evaluation chapter; AS-511 to AS-516)

1. Standardiser on the training updates only; constant columns dropped.
2. Regularisation lambda and class-weight strength p over the grid of SelectionConfig, chosen by blocked
   temporal cross-validation inside the training split (forward chaining, purging, embargo;
   temporal_cv.py), or on the validation split (method "validation"), or fixed.
3. Final fit on all training updates with the chosen (lambda, p).
4. Calibration on the validation updates (calibration.py), Platt scaling by default.
5. Thresholds on the calibrated validation probabilities: the model's own operating threshold
   (ThresholdConfig.rule, max-F1 by default) and the split-conformal threshold at the false-positive
   rate alpha (thresholds.py), so results can be reported at both.

Driving features. The logit is additive in the standardised features, so the exact Shapley values of a
prediction under independent features are phi_j = w_j (z_j - E[z_j]) (Lundberg and Lee, "A unified
approach to interpreting model predictions", NeurIPS 2017, linear-model case); `contributions` returns
them with the provenance of every column.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from nagahana.core.errors import InvariantViolation
from nagahana.evaluation.predictions import DetectionPredictions

from .calibration import Calibrator
from .common import RawRows, apply_linear, fit_standardiser, groups_of, standardised
from .config import DetectorConfig, FeatureConfig, StandardiserConfig, config_to_dict, from_dict
from .corpus import LRCorpus
from .design import Contexts, Units, UpdateDesign, gather, rows_source, unit_meta, update_units
from .features import FeatureColumn
from .logistic import BinaryFit, SolveInfo, class_weights, expit, fit_binary, sklearn_cross_check
from .scoring import LOWER_IS_BETTER, binary_score
from .standardise import Standardiser
from .temporal_cv import SelectionResult, select_hyperparameters, selection_from_dict
from .thresholds import ThresholdChoice, conformal_fpr_threshold, select_threshold


def detection_labels(corpus: LRCorpus, units: Units) -> np.ndarray:
    """1 malicious, 0 benign, -1 unknown, per update unit."""
    mal = gather(corpus, units, "malicious").astype(np.float64)
    return np.where(np.isnan(mal), -1, np.where(mal > 0.5, 1, 0)).astype(np.int64)


@dataclass
class LogisticDetector:
    """A fitted detection LR (module docstring)."""

    cfg: DetectorConfig
    columns: list[FeatureColumn]
    keep: np.ndarray
    standardiser: Standardiser
    fit: BinaryFit
    calibrator: Calibrator
    threshold: ThresholdChoice
    threshold_conformal: ThresholdChoice
    selection: SelectionResult | None
    window_seconds: float
    cross_check: dict[str, float] = field(default_factory=dict)

    @property
    def kept_columns(self) -> list[FeatureColumn]:
        return [self.columns[i] for i in self.keep.tolist()]

    @classmethod
    def train(cls, corpus: LRCorpus, contexts: Contexts, cfg: DetectorConfig, std_cfg: StandardiserConfig,
              feat_cfg: FeatureConfig, *, seed: int) -> LogisticDetector:
        """Run the protocol of the module docstring on a corpus whose records carry split roles."""
        design = UpdateDesign(corpus, contexts, cfg.context_lags)
        columns = design.columns()
        units = update_units(corpus)
        y = detection_labels(corpus, units)
        tr = np.flatnonzero((units.role == "train") & (y >= 0))
        va = np.flatnonzero((units.role == "val") & (y >= 0))
        if tr.size == 0:
            raise InvariantViolation("the detector needs labelled training updates")
        if va.size == 0:
            raise InvariantViolation("the detector needs labelled validation updates (calibration and thresholds)")
        chunk = cfg.solver.chunk_rows
        materialise = np.r_[tr, va] if cfg.solver.method == "lbfgs" else None
        raw = RawRows.from_design(design, units, materialise=materialise, chunk_rows=chunk)
        binary = np.asarray([c.binary for c in columns], dtype=bool)
        yf = y.astype(np.float64)

        def fit_on(pos: np.ndarray, std: Standardiser, keep: np.ndarray, lam: float, power: float,
                   theta0: Any = None) -> BinaryFit:
            c0, c1 = class_weights(yf[pos], power)
            wts = np.where(yf > 0.5, c1, c0)
            src = rows_source(raw.factory(pos), n_rows=int(pos.size), std=std, keep=keep, y=yf, weight=wts)
            return fit_binary(src, lam=lam, power=power, weights=(c0, c1), cfg=cfg.solver, seed=seed, theta0=theta0)

        def fit_and_score(pos_tr: np.ndarray, pos_va: np.ndarray, grid: list[tuple[float, float]]) -> np.ndarray:
            std, keep = fit_standardiser(raw, pos_tr, binary, std_cfg, feat_cfg, chunk)
            out = np.full(len(grid), np.nan)
            theta: dict[float, Any] = {}
            for g_i, (lam, power) in enumerate(grid):
                f = fit_on(pos_tr, std, keep, lam, power, theta.get(power))
                theta[power] = f.theta()                       # warm start along the lambda path of this power
                p = expit(apply_linear(raw, pos_va, std, keep, f.coef, chunk) + f.intercept)
                out[g_i] = binary_score(cfg.criterion, yf[pos_va], p)
            return out

        times = gather(corpus, units, "time").astype(np.float64)
        selection = select_hyperparameters(cfg.selection, tr=tr, va=va, times=times, label_end=times,
                                           groups=groups_of(corpus, units, cfg.selection.group_by),
                                           fit_and_score=fit_and_score, criterion=cfg.criterion,
                                           lower_is_better=LOWER_IS_BETTER[cfg.criterion],
                                           embargo_seconds=cfg.selection.embargo_windows * corpus.window_seconds)
        lam, power = selection.chosen if selection is not None else (float(cfg.selection.fixed_l2 or 0.0),
                                                                      float(cfg.selection.fixed_power or 0.0))
        std, keep = fit_standardiser(raw, tr, binary, std_cfg, feat_cfg, chunk)
        final = fit_on(tr, std, keep, lam, power)
        z_va = apply_linear(raw, va, std, keep, final.coef, chunk) + final.intercept
        cal = Calibrator.fit(z_va, y[va], cfg.calibration.method)
        p_va = cal.transform(z_va)
        own = select_threshold(p_va, y[va], cfg.threshold.rule, alpha=cfg.threshold.alpha)
        conf = conformal_fpr_threshold(p_va, y[va], cfg.threshold.alpha)
        check: dict[str, float] = {}
        if cfg.cross_check:
            x_tr = standardised(raw, tr, std, keep, chunk)
            c0, c1 = final.weights
            check = sklearn_cross_check(x_tr, yf[tr], np.where(yf[tr] > 0.5, c1, c0), final)
        return cls(cfg=cfg, columns=columns, keep=keep, standardiser=std, fit=final, calibrator=cal, threshold=own,
                   threshold_conformal=conf, selection=selection, window_seconds=corpus.window_seconds, cross_check=check)

    def recalibrate(self, corpus: LRCorpus, contexts: Contexts, roles: tuple[str, ...] = ("val",), *,
                    method: str | None = None, rule: str | None = None, alpha: float | None = None) -> LogisticDetector:
        """Refit the calibrator and both thresholds on the labelled updates of `roles` (for example a site's
        own calibration data), keeping the weights. Returns a new detector."""
        units = update_units(corpus, roles)
        y = detection_labels(corpus, units)
        pos = np.flatnonzero(y >= 0)
        if pos.size == 0:
            raise InvariantViolation("recalibration needs labelled updates")
        z = self.logits(corpus, contexts, units)[pos]
        cal = Calibrator.fit(z, y[pos], method or self.cfg.calibration.method)
        p = cal.transform(z)
        a = self.cfg.threshold.alpha if alpha is None else float(alpha)
        own = select_threshold(p, y[pos], rule or self.cfg.threshold.rule, alpha=a)
        conf = conformal_fpr_threshold(p, y[pos], a)
        return LogisticDetector(cfg=self.cfg, columns=self.columns, keep=self.keep, standardiser=self.standardiser,
                                fit=self.fit, calibrator=cal, threshold=own, threshold_conformal=conf,
                                selection=self.selection, window_seconds=self.window_seconds, cross_check=self.cross_check)

    def _raw(self, corpus: LRCorpus, contexts: Contexts, units: Units) -> RawRows:
        design = UpdateDesign(corpus, contexts, self.cfg.context_lags)
        if [c.name for c in design.columns()] != [c.name for c in self.columns]:
            raise InvariantViolation("the corpus encoder does not produce this detector's columns")
        return RawRows.from_design(design, units, materialise=None, chunk_rows=self.cfg.solver.chunk_rows)

    def logits(self, corpus: LRCorpus, contexts: Contexts, units: Units) -> np.ndarray:
        """w.z + b for every unit."""
        raw = self._raw(corpus, contexts, units)
        pos = np.arange(len(units))
        return apply_linear(raw, pos, self.standardiser, self.keep, self.fit.coef, self.cfg.solver.chunk_rows) + self.fit.intercept

    def predict(self, corpus: LRCorpus, contexts: Contexts, roles: tuple[str, ...]) -> DetectionPredictions:
        """Calibrated probabilities of every update of the given roles, with the own threshold."""
        units = update_units(corpus, roles)
        p = self.calibrator.transform(self.logits(corpus, contexts, units)) if len(units) else np.zeros(0)
        return DetectionPredictions(score=np.clip(p, 0.0, 1.0), label=detection_labels(corpus, units), unit="state_update",
                                    meta=unit_meta(corpus, units, triggers=False), threshold=self.threshold.value)

    def contributions(self, corpus: LRCorpus, contexts: Contexts, units: Units) -> np.ndarray:
        """Exact additive attributions phi [n, D'] of the logit to the kept columns (module docstring)."""
        raw = self._raw(corpus, contexts, units)
        z = standardised(raw, np.arange(len(units)), self.standardiser, self.keep, self.cfg.solver.chunk_rows)
        return (z - self.standardiser.standardised_mean()[self.keep]) * self.fit.coef

    def to_bundle(self, prefix: str) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
        arrays = {f"{prefix}.keep": self.keep, f"{prefix}.coef": self.fit.coef,
                  f"{prefix}.intercept": np.asarray(self.fit.intercept)}
        arrays.update({f"{prefix}.std.{k}": v for k, v in self.standardiser.state().items()})
        arrays.update({f"{prefix}.cal.{k}": v for k, v in self.calibrator.state().items()})
        header = {"config": config_to_dict(self.cfg), "columns": [c.as_dict() for c in self.columns],
                  "lam": self.fit.lam, "power": self.fit.power, "weights": list(self.fit.weights),
                  "info": self.fit.info.as_dict(), "threshold": self.threshold.as_dict(),
                  "threshold_conformal": self.threshold_conformal.as_dict(),
                  "selection": self.selection.as_dict() if self.selection is not None else None,
                  "window_seconds": self.window_seconds, "cross_check": self.cross_check}
        return header, arrays

    @classmethod
    def from_bundle(cls, prefix: str, header: dict[str, Any], arrays: dict[str, np.ndarray]) -> LogisticDetector:
        std = Standardiser.from_state({k[len(prefix) + 5:]: v for k, v in arrays.items() if k.startswith(f"{prefix}.std.")})
        cal = Calibrator.from_state({k[len(prefix) + 5:]: v for k, v in arrays.items() if k.startswith(f"{prefix}.cal.")})
        info = SolveInfo(**header["info"])
        fit = BinaryFit(coef=np.asarray(arrays[f"{prefix}.coef"], dtype=np.float64),
                        intercept=float(arrays[f"{prefix}.intercept"]), lam=float(header["lam"]),
                        power=float(header["power"]), weights=(float(header["weights"][0]), float(header["weights"][1])),
                        info=info)
        return cls(cfg=from_dict(DetectorConfig, header["config"], where="detector"),
                   columns=[FeatureColumn.from_dict(c) for c in header["columns"]],
                   keep=np.asarray(arrays[f"{prefix}.keep"], dtype=np.int64), standardiser=std, fit=fit, calibrator=cal,
                   threshold=ThresholdChoice(**header["threshold"]),
                   threshold_conformal=ThresholdChoice(**header["threshold_conformal"]),
                   selection=selection_from_dict(header["selection"]), window_seconds=float(header["window_seconds"]),
                   cross_check=dict(header.get("cross_check", {})))


__all__ = ["LogisticDetector", "detection_labels"]
