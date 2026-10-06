"""LightGBM and the other learners of Leevy, Hancock, Zuech and Khoshgoftaar, Journal of Big Data 8:38,
2021, "Detecting cybersecurity attacks across different network features and learners",
DOI 10.1186/s40537-021-00426-w.

What the paper states (baselines-notes.md, A3)
    data          CSE-CIC-IDS2018, all ten CSV days combined (about 16 million instances, about 17 %
                  attacks)
    task          binary, the attack (minority) class positive
    features      Timestamp, Protocol, the Flow ID / IP / port columns and fields with invalid values
                  are removed; the strongest results use feature group "1A" (14 features)
    learners      CatBoost, decision tree, LightGBM, logistic regression, naive Bayes, random forest,
                  XGBoost; no hyperparameters set for logistic regression (library defaults); no
                  feature scaling reported
    protocol      stratified 5-fold cross-validation repeated 10 times; the mean of the 50 measurements
    metrics       AUC and F1

What the paper leaves open (AS-538)
    "fields with invalid values" are read as features constant on the training rows (dropped) and rows
    with non-finite values (dropped from training); every learner, CatBoost included, runs with its
    library defaults. The 14 columns of feature group 1A are not transcribed in baselines-notes.md, so
    the default feature list is the full set after the paper's removals; a run of group 1A sets
    `features` to the paper's list.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar, Literal

import pandas as pd

from nagahana.baselines.published.base import BaselineConfig, BaselineSpec, InputSchema, Reference, ReportedResult
from nagahana.baselines.published.features import cicflowmeter
from nagahana.baselines.published.protocols import SplitProtocol, StratifiedKFold
from nagahana.baselines.published.tabular import TabularBaseline, TabularConfig

#: Columns the paper removes before training (baselines-notes.md, A3).
LEEVY_REMOVED: tuple[str, ...] = ("Timestamp", "Protocol", "Flow ID", "Src IP", "Src Port", "Dst IP", "Dst Port")
LEEVY_LEARNERS: tuple[str, ...] = (
    "lightgbm", "logistic_regression", "random_forest", "xgboost", "catboost", "decision_tree", "gaussian_nb",
)


@dataclass
class LeevyConfig(TabularConfig):
    """One learner of the study with library-default hyperparameters on unscaled features."""

    learner: str = "lightgbm"
    scaler: Literal["none", "minmax", "standard", "l2"] = "none"
    label_column: str = "Label"
    drop: tuple[str, ...] = LEEVY_REMOVED
    drop_constant: bool = True

    def validate(self) -> None:
        super().validate()
        if self.learner not in LEEVY_LEARNERS:
            raise ValueError(f"the study's reproducible learners are {LEEVY_LEARNERS}")


REFERENCE = Reference(
    key="bl-leevy2021features",
    authors="Leevy, Hancock, Zuech, Khoshgoftaar",
    title="Detecting cybersecurity attacks across different network features and learners",
    venue="Journal of Big Data 8:38",
    year=2021,
    doi="10.1186/s40537-021-00426-w",
)

_PROTOCOL = "10 x stratified 5-fold cross-validation, mean of 50; binary; feature group 1A (14 features)"
_NOTE = "feature group 1A: set `features` to the paper's 14 columns to reproduce this row (AS-538)"


def _row(model: str, auc: str, f1: str) -> ReportedResult:
    return ReportedResult(dataset="cse-cic-ids2018", protocol=_PROTOCOL, task="binary", model=model,
                          values={"auc": auc, "f1": f1}, location="Tab. 25, p. 17", variant={"learner": model}, note=_NOTE)


SPEC = BaselineSpec(
    name="leevy-cse-cic-ids2018",
    title="LightGBM and the learners of Leevy et al. on CSE-CIC-IDS2018",
    reference=REFERENCE,
    family="flow-classifier",
    input_schema=InputSchema(
        description="One row per CICFlowMeter flow record of CSE-CIC-IDS2018 (columns harmonised to the TrafficForML "
                    "names), with the text `Label`.",
        label="Label",
        optional=("time", "dataset", "network", "split"),
    ),
    outputs=("detection",),
    datasets=("cse-cic-ids2018",),
    reported=(
        _row("logistic_regression", "0.66772", "0.49471"),
        _row("random_forest", "0.95132", "0.92949"),
        _row("lightgbm", "0.96147", "0.94690"),
        _row("xgboost", "0.95385", "0.93647"),
        _row("decision_tree", "0.90891", "0.88583"),
        _row("gaussian_nb", "0.56711", "0.24319"),
        _row("catboost", "0.94200", "0.91650"),
    ),
    third_party="leevy-cse-cic-ids2018",
    assumptions=("AS-530", "AS-532", "AS-533", "AS-538"),
)


class LeevyLearners(TabularBaseline):
    """Leevy et al. 2021: library-default learners on unscaled CSE-CIC-IDS2018 features."""

    spec: ClassVar[BaselineSpec] = SPEC
    config_type: ClassVar[type[BaselineConfig]] = LeevyConfig
    config: LeevyConfig

    def prepare(self, frame: pd.DataFrame) -> pd.DataFrame:
        return cicflowmeter.clean(cicflowmeter.harmonise(frame))

    def default_features(self, columns: Sequence[str]) -> tuple[str, ...]:
        return cicflowmeter.feature_columns(list(columns), drop=())

    @classmethod
    def paper_protocol(cls) -> SplitProtocol:
        """10 x stratified 5-fold cross-validation on the label (baselines-notes.md, A3)."""
        return StratifiedKFold(n_splits=5, repeats=10, stratify="Label")


def leevy_reported_learners() -> tuple[str, ...]:
    """Learners of the reported rows, in table order (every row is reproducible)."""
    return tuple(r.model for r in SPEC.reported if r.model in LEEVY_LEARNERS)

