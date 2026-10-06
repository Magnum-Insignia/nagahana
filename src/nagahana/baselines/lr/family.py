"""The LR baseline family: fit every member on one corpus, predict any split, store and reload.

Members (each optional through `tasks`)

    detection        LogisticDetector   DetectionPredictions per state update (own and conformal thresholds)
    forecast         HazardLR           ForecastPredictions (P_inf(k), hazards) and TimeToEventPredictions
    stage            StageLR            StagePredictions per (trigger, step) or per update
    state_forecast   RidgeForecaster    StateForecastPredictions per trigger, horizons 1 ... K

plus the reference forecasts of references.py (persistence and climatology) on the same units, for the
skill scores. All members share one UpdateEncoder fitted on the training updates, so every model reads
the same columns with the same vocabularies, and every record is built through the prediction contract
of evaluation/predictions.py.

`run_protocol` is the entry point of an evaluation protocol: it fits the family on the corpus's training
and validation records and returns one ModelOutputs holding every requested split (train, val, test and
zero_shot by default, told apart by meta["split"]), with the reference outputs beside it. The records are
the ones evaluation/scorer.py reads: model name "logistic_regression" (the evaluation's LR family) and the
references "persistence" and "climatology".
"""

from __future__ import annotations

import platform
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from nagahana.core.errors import InvariantViolation
from nagahana.evaluation.predictions import ModelOutputs

from .config import TASK_NAMES, LRBaselineConfig, config_to_dict, from_dict
from .corpus import UNIT_ROLES, LRCorpus
from .design import Contexts, TriggerDesign, trigger_design, trigger_units
from .detector import LogisticDetector
from .features import UpdateEncoder
from .hazard import HazardLR
from .references import LifeTable, persistence_forecast, state_references
from .ridge import RidgeForecaster
from .serialize import load_bundle, save_bundle
from .stage import StageLR

TASKS: tuple[str, ...] = TASK_NAMES
#: Every split a unit can belong to, in the order of the data pipeline.
SPLIT_ORDER: tuple[str, ...] = ("train", "val", "test", "zero_shot")
FAMILY_FORMAT_NAME = "lr-family"


def training_batches(corpus: LRCorpus, chunk_rows: int) -> Any:
    """(values, status) chunks of every training update of the corpus (for the encoder)."""
    for s in corpus.sources:
        rows = np.flatnonzero(s.update_roles() == "train")
        for a in range(0, rows.size, chunk_rows):
            r = rows[a:a + chunk_rows]
            yield np.asarray(s["values"][r]), np.asarray(s["status"][r])


def corpus_summary(corpus: LRCorpus) -> dict[str, Any]:
    """Counts that identify the data a family was fitted on (stored with the model)."""
    out: dict[str, Any] = {"sources": [], "window_seconds": corpus.window_seconds, "horizon_k": corpus.horizon_k}
    for s in corpus.sources:
        roles = s.update_roles()
        out["sources"].append({"source_id": s.source_id, "dataset": s.dataset, "network": s.network, "origin": s.origin,
                               "updates": {r: int((roles == r).sum()) for r in sorted(set(roles.tolist()))},
                               "triggers": int(s.n_triggers), "timeless": s.timeless})
    return out


@dataclass
class LRFamily:
    """The fitted family (module docstring)."""

    config: LRBaselineConfig
    encoder: UpdateEncoder
    window_seconds: float
    horizon_k: int
    detector: LogisticDetector | None = None
    hazard: HazardLR | None = None
    stage: StageLR | None = None
    ridge: RidgeForecaster | None = None
    climatology: LifeTable | None = None
    provenance: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def fit(cls, corpus: LRCorpus, config: LRBaselineConfig, *, tasks: Sequence[str] = TASKS,
            progress: Callable[[str], None] | None = None) -> LRFamily:
        """Fit the requested members on the training records, calibrating on the validation records."""
        unknown = set(tasks) - set(TASKS)
        if unknown:
            raise InvariantViolation(f"unknown tasks {sorted(unknown)}; known: {list(TASKS)}")
        t0 = time.perf_counter()

        def say(msg: str) -> None:
            if progress is not None:
                progress(msg)

        say("fitting the update encoder")
        encoder = UpdateEncoder.fit(training_batches(corpus, config.detector.solver.chunk_rows), config.features)
        fam = cls(config=config, encoder=encoder, window_seconds=corpus.window_seconds, horizon_k=corpus.horizon_k)
        contexts = Contexts(corpus, encoder)
        design = fam._trigger_design(corpus, contexts, tasks)
        if "detection" in tasks:
            say("fitting the detection LR")
            fam.detector = LogisticDetector.train(corpus, contexts, config.detector, config.standardiser, config.features,
                                                  seed=config.seed)
        if "forecast" in tasks:
            assert design is not None
            say("fitting the discrete-time hazard LR")
            fam.hazard = HazardLR.train(corpus, design, config.hazard, config.standardiser, config.features)
            fam.climatology = LifeTable.fit(corpus)
        if "stage" in tasks:
            say("fitting the multinomial stage LR")
            fam.stage = StageLR.train(corpus, contexts, design, config.stage, config.standardiser, config.features)
        if "state_forecast" in tasks:
            assert design is not None
            say("fitting the ridge next-state forecaster")
            fam.ridge = RidgeForecaster.train(corpus, design, config.ridge, config.standardiser, config.features)
        fam.provenance = {"corpus": corpus_summary(corpus), "tasks": list(tasks), "seconds": time.perf_counter() - t0,
                          "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                          "versions": _versions()}
        return fam

    def _trigger_design(self, corpus: LRCorpus, contexts: Contexts, tasks: Sequence[str]) -> TriggerDesign | None:
        lags = []
        if "forecast" in tasks or (self.hazard is not None):
            lags.append(self.config.hazard.lags)
        if ("stage" in tasks or self.stage is not None) and self.config.stage.target == "step":
            lags.append(self.config.stage.lags)
        if "state_forecast" in tasks or self.ridge is not None:
            lags.append(self.ridge.lags if self.ridge is not None else max(self.config.ridge.lags))
        return trigger_design(corpus, contexts, max(lags)) if lags else None

    @property
    def tasks(self) -> tuple[str, ...]:
        have = {"detection": self.detector, "forecast": self.hazard, "stage": self.stage, "state_forecast": self.ridge}
        return tuple(t for t in TASKS if have[t] is not None)

    def _check(self, corpus: LRCorpus) -> None:
        if corpus.window_seconds != self.window_seconds or corpus.horizon_k != self.horizon_k:
            raise InvariantViolation(f"the corpus has cadence {corpus.window_seconds} s and horizon {corpus.horizon_k}; "
                                     f"the family was fitted with {self.window_seconds} s and {self.horizon_k}")

    def summary(self) -> dict[str, Any]:
        """Chosen hyperparameters, thresholds and convergence of every member (stored in ModelOutputs.config)."""
        out: dict[str, Any] = {"name": self.config.name, "tasks": list(self.tasks)}
        if self.detector is not None:
            d = self.detector
            out["detection"] = {"l2": d.fit.lam, "class_weight_power": d.fit.power, "threshold": d.threshold.as_dict(),
                                "threshold_conformal": d.threshold_conformal.as_dict(), "calibration": d.calibrator.method,
                                "columns": int(d.keep.size), "converged": d.fit.info.converged, "cross_check": d.cross_check}
        if self.hazard is not None:
            h = self.hazard
            out["forecast"] = {"l2": h.lam, "class_weight_power": h.power, "calibration": list(h.calib),
                               "columns": int(h.keep.size), "converged": h.info.converged}
        if self.stage is not None:
            s = self.stage
            out["stage"] = {"l2": s.lam, "class_weight_power": s.power, "classes": s.classes.tolist(),
                            "columns": int(s.keep.size), "converged": s.info.converged, "target": s.cfg.target}
        if self.ridge is not None:
            r = self.ridge
            out["state_forecast"] = {"l2": r.lam, "lags": r.lags, "columns": int(r.keep.size)}
        return out

    def predict(self, corpus: LRCorpus, splits: str | Sequence[str] = SPLIT_ORDER, *, protocol: str) -> ModelOutputs:
        """Every fitted member's record for the units of the given splits (all four by default), as ModelOutputs.

        One record holds every requested split, distinguished by meta["split"]: the evaluation fits its
        conformal thresholds on the calibration split and its climatology on the training triggers of the
        record it scores (evaluation/scorer.py), so a protocol's record carries train, val, test and zero_shot.
        """
        roles = _roles(splits)
        self._check(corpus)
        contexts = Contexts(corpus, self.encoder)
        design = self._trigger_design(corpus, contexts, ())
        out = ModelOutputs(model=self.config.name, protocol=protocol, seed=self.config.seed,
                           config={"lr": config_to_dict(self.config), "fitted": self.summary()})
        if self.detector is not None:
            out.detection = self.detector.predict(corpus, contexts, roles)
            out.component["detection.threshold_conformal"] = np.asarray(self.detector.threshold_conformal.value)
            out.component["detection.coef"] = self.detector.fit.coef.copy()
        if self.hazard is not None:
            assert design is not None
            out.forecast, out.time_to_event = self.hazard.predict(corpus, design, roles)
        if self.stage is not None:
            out.stage = self.stage.predict(corpus, contexts, design, roles)
        if self.ridge is not None:
            assert design is not None
            out.state_forecast = self.ridge.predict(corpus, design, roles)
        return out

    def references(self, corpus: LRCorpus, splits: str | Sequence[str] = SPLIT_ORDER, *,
                   protocol: str) -> dict[str, ModelOutputs]:
        """Persistence and climatology records on the units of the forecast and next-state members."""
        roles = _roles(splits)
        self._check(corpus)
        pers = ModelOutputs(model="persistence", protocol=protocol, seed=self.config.seed)
        clim = ModelOutputs(model="climatology", protocol=protocol, seed=self.config.seed)
        if self.hazard is not None:
            units = trigger_units(corpus, roles, usable_only=True)
            pers.forecast, pers.time_to_event = persistence_forecast(corpus, units)
            if self.climatology is None:
                raise InvariantViolation("the climatology was not fitted")
            clim.forecast, clim.time_to_event = self.climatology.forecast(corpus, units)
        if self.ridge is not None:
            p, c, filled = state_references(corpus, roles, self.ridge.scaler, self.config.features)
            pers.state_forecast, clim.state_forecast = p, c
            pers.component["state_forecast.climatology_filled"] = np.asarray(filled)
        return {"persistence": pers, "climatology": clim}

    def save(self, path: str | Path) -> str:
        """Write the family as one checksummed bundle (serialize.py); returns its digest."""
        header: dict[str, Any] = {"kind": FAMILY_FORMAT_NAME, "config": config_to_dict(self.config),
                                  "encoder": self.encoder.state_dict(), "window_seconds": self.window_seconds,
                                  "horizon_k": self.horizon_k, "provenance": self.provenance, "members": {}}
        arrays: dict[str, np.ndarray] = {}
        for name, member in (("detector", self.detector), ("hazard", self.hazard), ("stage", self.stage),
                             ("ridge", self.ridge)):
            if member is None:
                continue
            h, a = member.to_bundle(name)
            header["members"][name] = h
            arrays.update(a)
        if self.climatology is not None:
            header["climatology"] = self.climatology.state()
        return save_bundle(path, header, arrays)

    @classmethod
    def load(cls, path: str | Path) -> LRFamily:
        header, arrays = load_bundle(path)
        if header.get("kind") != FAMILY_FORMAT_NAME:
            raise InvariantViolation(f"{path} does not hold an LR family")
        config = from_dict(LRBaselineConfig, header["config"])
        fam = cls(config=config, encoder=UpdateEncoder.from_state_dict(config.features, header["encoder"]),
                  window_seconds=float(header["window_seconds"]), horizon_k=int(header["horizon_k"]),
                  provenance=dict(header.get("provenance", {})))
        m = header["members"]
        if "detector" in m:
            fam.detector = LogisticDetector.from_bundle("detector", m["detector"], arrays)
        if "hazard" in m:
            fam.hazard = HazardLR.from_bundle("hazard", m["hazard"], arrays)
        if "stage" in m:
            fam.stage = StageLR.from_bundle("stage", m["stage"], arrays)
        if "ridge" in m:
            fam.ridge = RidgeForecaster.from_bundle("ridge", m["ridge"], arrays)
        if "climatology" in header:
            fam.climatology = LifeTable.from_state(header["climatology"])
        return fam


def _versions() -> dict[str, str]:
    import torch

    return {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__}


@dataclass
class ProtocolRun:
    """The outputs of one evaluation protocol for the LR family: the family's record and the references'."""

    family: LRFamily
    outputs: ModelOutputs
    references: dict[str, ModelOutputs]


def run_protocol(corpus: LRCorpus, config: LRBaselineConfig, *, protocol: str, splits: Sequence[str] = SPLIT_ORDER,
                 tasks: Sequence[str] = TASKS, progress: Callable[[str], None] | None = None) -> ProtocolRun:
    """Fit on the corpus's training and validation records and predict the requested splits (module docstring)."""
    fam = LRFamily.fit(corpus, config, tasks=tasks, progress=progress)
    return ProtocolRun(family=fam, outputs=fam.predict(corpus, splits, protocol=protocol),
                       references=fam.references(corpus, splits, protocol=protocol))


def _roles(splits: str | Sequence[str]) -> tuple[str, ...]:
    """Validated tuple of split names."""
    roles = (splits,) if isinstance(splits, str) else tuple(splits)
    bad = [r for r in roles if r not in UNIT_ROLES]
    if bad or not roles:
        raise InvariantViolation(f"unknown or missing splits {bad}; known: {list(SPLIT_ORDER)}")
    return roles


__all__ = ["SPLIT_ORDER", "TASKS", "LRFamily", "ProtocolRun", "corpus_summary", "run_protocol", "training_batches"]
