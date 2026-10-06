"""Logistic-regression baseline family on exactly the features NagaHana sees.

The problem statement requires benchmark results (F1, precision, recall, false-positive rate) against a
logistic regression trained on the same features; the evaluation chapter also compares forecasts, stages
and next-state forecasts against linear comparators and naive references. This package builds all of
them from the windows NagaHana consumes and reports through the shared prediction contract
(evaluation/predictions.py):

    corpus.py        per-source update and trigger tables, extracted from NagaHana's windows (or batches)
    features.py      the design columns: canonical fields with status indicators, window states, aggregates
    design.py        context windows, trigger and update designs, units and metadata
    standardise.py   location and scale fitted on training rows only (exact, out-of-core capable)
    logistic.py      binary LR: weighted cross-entropy + l2, L-BFGS (in memory and streamed), averaged SGD
    calibration.py   Platt, temperature and isotonic calibration on validation units
    thresholds.py    max-F1, Youden and split-conformal fixed-FPR thresholds on validation units
    temporal_cv.py   blocked forward-chaining cross-validation with purging and embargo; grid selection
    detector.py      detection LR per state update
    hazard.py        discrete-time hazard LR: P_inf(k), hazards, survival curves
    stage.py         multinomial stage LR over the ATT&CK stage vocabulary
    ridge.py         ridge next-state forecaster on the same features and their lags
    references.py    persistence and climatology reference forecasts
    family.py        fit, predict, store and reload the whole family; run_protocol
    serialize.py     versioned, checksummed model bundles
    synthetic.py     a labelled synthetic source with a planted attack process (test fixture)
    config.py        configuration dataclasses (conf/baselines/lr/*.yaml are generated from them)
    cli.py           `nagahana lr ...` commands (extract, fit, cross-validate, calibrate, predict, write-config)

Decisions and assumptions: D-23 (splits), D-41 (absence is never zero), D-51 (stream order), D-54
(float64 outputs), D-55 (scikit-learn only for the optional cross-check); AS-500 ... AS-529
(docs/assumptions/lr-baseline.md). Protocol and usage: docs/baselines-lr.md.

The configuration classes are imported eagerly; the corpus and family classes, which pull in the data
pipeline, are loaded on first access.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

from .config import (
    CalibrationConfig,
    DetectorConfig,
    FeatureConfig,
    GridSpec,
    HazardConfig,
    LRBaselineConfig,
    RidgeConfig,
    SelectionConfig,
    SolverConfig,
    StageConfig,
    StandardiserConfig,
    ThresholdConfig,
    config_to_dict,
    load_config,
)

if TYPE_CHECKING:
    from .corpus import CorpusBuilder, LRCorpus, build_corpus, extract_source
    from .family import TASKS, LRFamily, ProtocolRun, run_protocol

_LAZY: dict[str, str] = {
    "CorpusBuilder": ".corpus", "LRCorpus": ".corpus", "build_corpus": ".corpus", "extract_source": ".corpus",
    "TASKS": ".family", "LRFamily": ".family", "ProtocolRun": ".family", "run_protocol": ".family",
}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        return getattr(import_module(_LAZY[name], __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "TASKS", "CalibrationConfig", "CorpusBuilder", "DetectorConfig", "FeatureConfig", "GridSpec", "HazardConfig",
    "LRBaselineConfig", "LRCorpus", "LRFamily", "ProtocolRun", "RidgeConfig", "SelectionConfig", "SolverConfig",
    "StageConfig", "StandardiserConfig", "ThresholdConfig", "build_corpus", "config_to_dict", "extract_source",
    "load_config", "run_protocol",
]
