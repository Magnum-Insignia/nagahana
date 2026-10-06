"""Cross-dataset generalisation study (Cantone, Marrocco and Bria, IEEE Access 12:144489-144508, 2024,
"Machine Learning in Network Intrusion Detection: A Cross-Dataset Generalization Study",
DOI 10.1109/ACCESS.2024.3472907).

What the paper states (baselines-notes.md, B1)
    task          binary: every attack class grouped as malicious
    learners      linear discriminant analysis, decision tree, random forest, XGBoost
    scaling       min-max normalisation
    tuning        grid search on a 20 % subset of the training data
    within        random 80:20 train/test split of one dataset
    cross         train on one entire dataset, test on another (CIC-IDS2017 <-> CSE-CIC-IDS2018;
                  LycoS-IDS2017 <-> LycoS-Unicas-IDS2018)
    metrics       MCC, F1 and AUROC (accuracy deliberately not reported)

What the paper leaves open (AS-537)
    the grid of each learner (below), the tuning criterion (MCC, the paper's primary metric) and its
    cross-validation (3 stratified folds on the 20 % subset); the feature space is the harmonised set of
    CICFlowMeter features common to both datasets, without the flow identifiers and the destination port
    (features.cicflowmeter); rows with non-finite values are dropped from training.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal

import numpy as np
import pandas as pd

from nagahana.baselines.published.base import BaselineConfig, BaselineSpec, InputSchema, Reference, ReportedResult, derive_seed
from nagahana.baselines.published.features import cicflowmeter
from nagahana.baselines.published.protocols import CrossDataset, RandomHoldout, SplitProtocol, stratified_choice
from nagahana.baselines.published.tabular import TabularBaseline, TabularConfig, grid_search


@dataclass
class CantoneConfig(TabularConfig):
    """One learner of the study, with its grid.

    Attributes
    ----------
    tune:
        Run the grid search before the final fit.
    tune_fraction:
        Share of the training rows (stratified) the grid search runs on.
    tune_folds:
        Stratified folds of the grid search.
    grid_lda, grid_decision_tree, grid_random_forest, grid_xgboost:
        Grid per learner; only the selected learner's grid is used.
    feature_set:
        "cicflowmeter" (harmonised CIC-IDS2017 / CSE-CIC-IDS2018 columns) or "custom" (`features` must
        list the columns, for example for the LycoS datasets).
    """

    learner: str = "random_forest"
    scaler: Literal["none", "minmax", "standard", "l2"] = "minmax"
    label_column: str = "Label"
    drop: tuple[str, ...] = ("Dst Port",)
    tune: bool = True
    tune_fraction: float = 0.2
    tune_folds: int = 3
    grid_lda: dict[str, tuple[Any, ...]] = field(default_factory=lambda: {"solver": ("svd", "lsqr")})
    grid_decision_tree: dict[str, tuple[Any, ...]] = field(
        default_factory=lambda: {"criterion": ("gini", "entropy"), "max_depth": (None, 10, 20, 30)})
    grid_random_forest: dict[str, tuple[Any, ...]] = field(
        default_factory=lambda: {"n_estimators": (50, 100, 200), "max_depth": (None, 20)})
    grid_xgboost: dict[str, tuple[Any, ...]] = field(
        default_factory=lambda: {"n_estimators": (100, 200), "max_depth": (3, 6, 9), "learning_rate": (0.1, 0.3)})
    feature_set: Literal["cicflowmeter", "custom"] = "cicflowmeter"

    def validate(self) -> None:
        super().validate()
        if self.learner not in ("lda", "decision_tree", "random_forest", "xgboost"):
            raise ValueError("the study's learners are lda, decision_tree, random_forest and xgboost")
        if not 0.0 < self.tune_fraction <= 1.0:
            raise ValueError("tune_fraction must lie in (0, 1]")
        if self.tune_folds < 2:
            raise ValueError("tune_folds must be >= 2")
        if self.feature_set == "custom" and not self.features:
            raise ValueError("feature_set 'custom' needs an explicit features list")

    def grid(self) -> dict[str, tuple[Any, ...]]:
        """The grid of the selected learner."""
        return {"lda": self.grid_lda, "decision_tree": self.grid_decision_tree, "random_forest": self.grid_random_forest,
                "xgboost": self.grid_xgboost}[self.learner]


REFERENCE = Reference(
    key="cantone2024cross",
    authors="Cantone, Marrocco, Bria",
    title="Machine Learning in Network Intrusion Detection: A Cross-Dataset Generalization Study",
    venue="IEEE Access 12:144489-144508",
    year=2024,
    doi="10.1109/ACCESS.2024.3472907",
    arxiv="2402.10974",
    note="values from the published version (Tab. 4, p. 144495); the arXiv preprint prints the same values",
)

_LEARNER = {"LDA": "lda", "DT": "decision_tree", "RF": "random_forest", "XGB": "xgboost"}


def _rows() -> tuple[ReportedResult, ...]:
    table = {
        ("cic-ids2017", "cic-ids2017"): {"LDA": ("64.86", "69.02", "96.45"), "DT": ("99.72", "99.78", "99.97"),
                                         "RF": ("99.74", "99.79", "99.98"), "XGB": ("99.74", "99.79", "100.00")},
        ("cic-ids2017", "cse-cic-ids2018"): {"LDA": ("32.62", "39.12", "81.70"), "DT": ("30.82", "31.20", "59.14"),
                                             "RF": ("25.96", "25.76", "66.93"), "XGB": ("35.68", "26.43", "80.44")},
        ("cse-cic-ids2018", "cse-cic-ids2018"): {"LDA": ("76.52", "79.82", "94.13"), "DT": ("96.33", "96.89", "98.85"),
                                                 "RF": ("96.45", "96.99", "98.87"), "XGB": ("96.48", "97.02", "99.10")},
        ("cse-cic-ids2018", "cic-ids2017"): {"LDA": ("39.29", "48.41", "85.06"), "DT": ("35.50", "33.86", "57.80"),
                                             "RF": ("44.44", "51.21", "75.77"), "XGB": ("41.75", "44.58", "79.15")},
        ("lycos-ids2017", "lycos-unicas-ids2018"): {"RF": ("9.29", "13.97", "65.46"), "XGB": ("10.17", "14.42", "61.54"),
                                                    "LDA": ("-15.32", "10.24", "57.24")},
        ("lycos-unicas-ids2018", "lycos-ids2017"): {"RF": ("38.33", "37.29", "77.49"), "XGB": ("40.29", "35.54", "81.37"),
                                                    "LDA": ("60.41", "70.46", "91.66")},
    }
    out = []
    for (train, test), models in table.items():
        protocol = "random 80:20 split; binary" if train == test else "train on the whole of one dataset, test on the other; binary"
        for short, (mcc, f1, auroc) in models.items():
            variant: dict[str, Any] = {"learner": _LEARNER[short]}
            if train.startswith("lycos"):
                variant["feature_set"] = "custom"
            out.append(ReportedResult(dataset=f"{train}->{test}", protocol=protocol, task="binary", model=_LEARNER[short],
                                      values={"mcc": mcc, "f1": f1, "auroc": auroc}, location="Tab. 4, p. 144495",
                                      variant=variant, percent=("mcc", "f1", "auroc")))
    return tuple(out)


SPEC = BaselineSpec(
    name="cantone-cross-dataset",
    title="Cross-dataset generalisation study (LDA, decision tree, random forest, XGBoost)",
    reference=REFERENCE,
    family="flow-classifier",
    input_schema=InputSchema(
        description="One row per CICFlowMeter flow record of CIC-IDS2017 or CSE-CIC-IDS2018 (any spelling; columns are "
                    "harmonised to the CSE-CIC-IDS2018 names), with the text `Label` and a `dataset` column naming "
                    "the source for cross-dataset runs.",
        label="Label",
        optional=("dataset", "time", "network", "split"),
    ),
    outputs=("detection",),
    datasets=("cic-ids2017", "cse-cic-ids2018", "lycos-ids2017", "lycos-unicas-ids2018"),
    reported=_rows(),
    third_party="cantone-cross-dataset",
    assumptions=("AS-530", "AS-532", "AS-533", "AS-537"),
)


class CantoneCrossDataset(TabularBaseline):
    """Cantone et al. 2024: one of LDA, DT, RF, XGBoost with min-max scaling and a subset grid search."""

    spec: ClassVar[BaselineSpec] = SPEC
    config_type: ClassVar[type[BaselineConfig]] = CantoneConfig
    config: CantoneConfig

    def prepare(self, frame: pd.DataFrame) -> pd.DataFrame:
        # Harmonise column spellings before the schema check, so either dataset's files are accepted.
        if self.config.feature_set == "cicflowmeter":
            return cicflowmeter.clean(cicflowmeter.harmonise(frame))
        return frame

    def default_features(self, columns: Sequence[str]) -> tuple[str, ...]:
        return cicflowmeter.feature_columns(list(columns), drop=cicflowmeter.CIC_IDENTIFIERS)

    def _choose_params(self, train: pd.DataFrame, codes: np.ndarray, numeric: tuple[str, ...],
                       categorical: tuple[str, ...]) -> dict[str, Any]:
        cfg = self.config
        if not cfg.tune:
            return dict(cfg.params)
        # Grid search on a stratified subset of the training rows (the paper's 20 % subset).
        rng = np.random.default_rng(derive_seed(cfg.seed, "cantone-tune"))
        n_sub = max(cfg.tune_folds * 2, int(round(cfg.tune_fraction * len(train))))
        subset = stratified_choice(codes, min(n_sub, len(train)), rng)
        best, table = grid_search(train.iloc[subset].reset_index(drop=True), codes[subset], numeric=numeric,
                                  categorical=categorical, scaler=cfg.scaler, max_categories=cfg.max_categories,
                                  learner=cfg.learner, base_params=cfg.params, grid=cfg.grid(), n_classes=2,
                                  folds=cfg.tune_folds, seed=cfg.seed, needed_by=self.spec.name)
        self.fit_report["grid_search"] = {"subset_rows": int(subset.size), "results": table}
        return best

    @classmethod
    def paper_protocol(cls, train_dataset: str | None = None, test_dataset: str | None = None) -> SplitProtocol:
        """Random 80:20 within one dataset, or train-on-one / test-on-another across datasets."""
        if train_dataset is not None and test_dataset is not None and train_dataset != test_dataset:
            return CrossDataset(train_dataset, test_dataset)
        return RandomHoldout(test_fraction=0.2)
