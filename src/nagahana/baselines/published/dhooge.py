"""Per-class cross-dataset models of D'hooge, Verkerken, Wauters, De Turck and Volckaert, Sensors 23(4):1846,
2023, "Investigating Generalized Performance of Data-Constrained Supervised Machine Learning Models on
Novel, Related Samples in Intrusion Detection", DOI 10.3390/s23041846.

What the paper states (baselines-notes.md, A4)
    task          binary, one model per attack class, trained on CSE-CIC-IDS2018 and tested on the
                  corresponding class of CIC-IDS2017 (also CIC-DoS2017 and CIC-DDoS2019)
    learners      12 algorithms, among them logistic regression ("binlr", trained on min-max scaled
                  features), linear and RBF SVMs, kNN and tree ensembles
    variations    training volume, and removal of the top features
    sampling      stratified
    results       stated in the text: for layer-7 DoS the logistic regression reaches F1 "up to 78%", with
                  precision "~65%"; for brute force the best tree ensembles preserve "up to 89%" F1 (recall
                  99.9+ %, precision 81.5 %)

What the paper leaves open (AS-561)
    the class correspondence between the two datasets (ATTACK_GROUPS); the feature space (harmonised
    CICFlowMeter features without identifiers); "top features" ranked by random-forest impurity
    importance on the training rows; benign rows enter every class model; attacks of other classes are
    excluded from a class model's evaluation (label -1).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar, Literal

import numpy as np
import pandas as pd

from nagahana.baselines.published.base import BaselineConfig, BaselineSpec, InputSchema, Reference, ReportedResult, derive_seed
from nagahana.baselines.published.estimators import TabularModel
from nagahana.baselines.published.features import cicflowmeter
from nagahana.baselines.published.frames import BENIGN_TOKENS, normalise_token
from nagahana.baselines.published.preprocessing import ColumnPipeline
from nagahana.baselines.published.protocols import CrossDataset, SplitProtocol, stratified_choice
from nagahana.baselines.published.tabular import TabularBaseline, TabularConfig
from nagahana.core.errors import InvariantViolation

#: Attack group -> labels of either dataset (normalised), CSE-CIC-IDS2018 and CIC-IDS2017 spellings.
ATTACK_GROUPS: dict[str, frozenset[str]] = {
    "bruteforce": frozenset({"ftpbruteforce", "sshbruteforce", "ftppatator", "sshpatator"}),
    "l7dos": frozenset({"dosattackshulk", "dosattacksgoldeneye", "dosattacksslowloris", "dosattacksslowhttptest",
                        "doshulk", "dosgoldeneye", "dosslowloris", "dosslowhttptest"}),
    "web": frozenset({"bruteforceweb", "bruteforcexss", "sqlinjection", "webattackbruteforce", "webattackxss",
                      "webattacksqlinjection"}),
    "botnet": frozenset({"bot"}),
    "infiltration": frozenset({"infilteration", "infiltration"}),
    "ddos": frozenset({"ddosattacksloichttp", "ddosattackhoic", "ddosattackloicudp", "ddos"}),
}
AttackGroup = Literal["bruteforce", "l7dos", "web", "botnet", "infiltration", "ddos"]


@dataclass
class DhoogeConfig(TabularConfig):
    """One per-class model of the study.

    Attributes
    ----------
    attack_group:
        The attack class the model is trained and tested on (ATTACK_GROUPS).
    train_dataset, dataset_column:
        Training rows are those whose `dataset_column` equals `train_dataset` (all rows when the column
        is absent).
    train_samples:
        Training volume: a stratified sample of this many rows (0: every row of the class subset).
    remove_top_features:
        Number of most important features removed before training (0: none).
    """

    learner: str = "logistic_regression"
    scaler: Literal["none", "minmax", "standard", "l2"] = "minmax"
    label_column: str = "Label"
    attack_group: AttackGroup = "l7dos"
    train_dataset: str = "cse-cic-ids2018"
    dataset_column: str = "dataset"
    train_samples: int = 0
    remove_top_features: int = 0

    def validate(self) -> None:
        super().validate()
        if self.train_samples < 0 or self.remove_top_features < 0:
            raise ValueError("train_samples and remove_top_features must be non-negative")


REFERENCE = Reference(
    key="bl-dhooge2023generalized",
    authors="D'hooge, Verkerken, Wauters, De Turck, Volckaert",
    title="Investigating Generalized Performance of Data-Constrained Supervised Machine Learning Models on Novel, "
          "Related Samples in Intrusion Detection",
    venue="Sensors 23(4):1846",
    year=2023,
    doi="10.3390/s23041846",
)

SPEC = BaselineSpec(
    name="dhooge-generalisation",
    title="Per-class models trained on CSE-CIC-IDS2018, tested on CIC-IDS2017",
    reference=REFERENCE,
    family="flow-classifier",
    input_schema=InputSchema(
        description="CICFlowMeter flow records of both datasets (any spelling, harmonised) with the text `Label` and a "
                    "`dataset` column naming each row's dataset.",
        label="Label",
        optional=("dataset", "time", "network", "split"),
    ),
    outputs=("detection",),
    datasets=("cse-cic-ids2018", "cic-ids2017"),
    reported=(
        ReportedResult(dataset="cse-cic-ids2018->cic-ids2017", protocol="cross-dataset, one model per attack class; binary",
                       task="binary", model="logistic_regression", values={"f1": "<=78%", "precision": "~65%"},
                       location="Sec. 3.3.2, pp. 17-18", variant={"learner": "logistic_regression", "attack_group": "l7dos"},
                       note="the authors' wording (\"up to 78%\", \"~65%\"); bounds, not point values"),
        ReportedResult(dataset="cse-cic-ids2018->cic-ids2017", protocol="cross-dataset, one model per attack class; binary",
                       task="binary", model="best tree ensemble", values={"f1": "<=89%", "recall": ">=99.9%", "precision": "81.5%"},
                       location="Sec. 3.3.1, p. 15", variant={"attack_group": "bruteforce"},
                       note="the ensemble is not named; compare a tree-ensemble learner"),
    ),
    third_party="dhooge-generalisation",
    assumptions=("AS-530", "AS-532", "AS-533", "AS-561"),
)


class DhoogeGeneralisation(TabularBaseline):
    """D'hooge et al. 2023: a per-class binary model trained on one dataset, tested on another."""

    spec: ClassVar[BaselineSpec] = SPEC
    config_type: ClassVar[type[BaselineConfig]] = DhoogeConfig
    config: DhoogeConfig

    def __init__(self, config: BaselineConfig | None = None) -> None:
        super().__init__(config)
        self.removed_: tuple[str, ...] = ()

    def prepare(self, frame: pd.DataFrame) -> pd.DataFrame:
        return cicflowmeter.clean(cicflowmeter.harmonise(frame))

    def default_features(self, columns: Sequence[str]) -> tuple[str, ...]:
        return cicflowmeter.feature_columns(list(columns), drop=cicflowmeter.CIC_IDENTIFIERS)

    def _feature_list(self, frame: pd.DataFrame) -> tuple[str, ...]:
        return tuple(c for c in super()._feature_list(frame) if c not in set(self.removed_))

    def _codes(self, frame: pd.DataFrame) -> np.ndarray:
        # 1: the model's attack group; 0: benign; -1: other attacks and unlabelled rows.
        group = ATTACK_GROUPS[self.config.attack_group]
        tokens = [normalise_token(v) for v in frame[self.config.label_column].tolist()] if self.config.label_column in frame.columns else []
        if not tokens:
            return np.full(len(frame), -1, dtype=np.int64)
        return np.asarray([1 if t in group else (0 if t in BENIGN_TOKENS else -1) for t in tokens], dtype=np.int64)

    def _fit(self, data: pd.DataFrame, validation: pd.DataFrame | None, rng: np.random.Generator) -> None:
        cfg = self.config
        rows = np.ones(len(data), dtype=bool)
        if cfg.dataset_column in data.columns:
            rows &= data[cfg.dataset_column].astype(str).to_numpy() == cfg.train_dataset
        codes = self._codes(data)
        rows &= codes >= 0
        subset = np.nonzero(rows)[0]
        if subset.size == 0:
            raise InvariantViolation(f"{self.spec.name}: no training rows of {cfg.train_dataset} for group {cfg.attack_group}")
        if 0 < cfg.train_samples < subset.size:
            # Stratified training volume: keep the class balance of the subset.
            sample_rng = np.random.default_rng(derive_seed(cfg.seed, "dhooge-volume"))
            subset = subset[stratified_choice(codes[subset], cfg.train_samples, sample_rng)]
        train = data.iloc[subset].reset_index(drop=True)
        self.removed_ = ()
        if cfg.remove_top_features > 0:
            self.removed_ = self._top_features(train)
        super()._fit(train, validation, rng)
        self.fit_report.update({"attack_group": cfg.attack_group, "removed_top_features": list(self.removed_)})

    def _top_features(self, train: pd.DataFrame) -> tuple[str, ...]:
        # Random-forest impurity importance on the (class-subset) training rows.
        cfg = self.config
        feats = super()._feature_list(train)
        codes = self._codes(train)
        pipe = ColumnPipeline(feats, (), "none")
        keep = ~pipe.nonfinite_rows(train)
        pipe.fit(train.loc[keep])
        rf = TabularModel("random_forest", {}, cfg.seed, 2, needed_by=self.spec.name)
        rf.fit(pipe.transform(train.loc[keep]), codes[keep])
        imp = rf.feature_importances()
        assert imp is not None
        order = np.argsort(-imp, kind="stable")[: cfg.remove_top_features]
        return tuple(feats[i] for i in order)

    @classmethod
    def paper_protocol(cls) -> SplitProtocol:
        """Train on CSE-CIC-IDS2018, test on CIC-IDS2017 (rows selected by the `dataset` column)."""
        return CrossDataset("cse-cic-ids2018", "cic-ids2017")
