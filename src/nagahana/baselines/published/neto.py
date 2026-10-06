"""Random forest and logistic regression on CIC-IoT-2023 (Neto, Dadkhah, Ferreira, Zohourian, Lu and
Ghorbani, Sensors 23(13):5941, 2023, "CICIoT2023: A Real-Time Dataset and Benchmark for Large-Scale
Attacks in IoT Environment", DOI 10.3390/s23135941).

What the paper states (baselines-notes.md, D3)
    data          the CIC-IoT-2023 CSV files: 46 features per window of packets
    tasks         2 classes (benign against attack), 8 classes (7 attack categories and benign), 34 classes
    scaling       StandardScaler
    protocol      random 80/20 train/test split
    metrics       accuracy, recall, precision, F1; for the binary task the printed recall, precision
                  and F1 appear to be averaged over the two classes (the notes' reading)

What the paper leaves open (AS-540)
    every learner runs with scikit-learn defaults; the multi-class F1 is read as the macro average (the
    averaging of the binary rows); the 34 -> 8 grouping of features.ciciot2023.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar, Literal

import numpy as np
import pandas as pd

from nagahana.baselines.published.base import BaselineConfig, BaselineSpec, InputSchema, Reference, ReportedResult
from nagahana.baselines.published.features import ciciot2023
from nagahana.baselines.published.protocols import RandomHoldout, SplitProtocol
from nagahana.baselines.published.tabular import TabularBaseline, TabularConfig

NETO_LEARNERS: tuple[str, ...] = ("random_forest", "logistic_regression", "adaboost", "mlp")


@dataclass
class NetoConfig(TabularConfig):
    """One learner of the benchmark on one of the three class groupings.

    Attributes
    ----------
    grouping:
        "binary", "multiclass-8" or "multiclass-34" (features.ciciot2023).
    """

    learner: str = "random_forest"
    scaler: Literal["none", "minmax", "standard", "l2"] = "standard"
    label_column: str = "label"
    unit: Literal["flow", "window", "state_update"] = "window"
    stage_dataset: str = "ciciot2023"
    grouping: Literal["binary", "multiclass-8", "multiclass-34"] = "binary"

    def validate(self) -> None:
        super().validate()
        if self.learner not in NETO_LEARNERS:
            raise ValueError(f"the benchmark's learners reproduced here are {NETO_LEARNERS}")
        if (self.grouping == "binary") != (self.task == "binary"):
            raise ValueError("task must be 'binary' exactly when grouping is 'binary'")


REFERENCE = Reference(
    key="bl-neto2023ciciot",
    authors="Neto, Dadkhah, Ferreira, Zohourian, Lu, Ghorbani",
    title="CICIoT2023: A Real-Time Dataset and Benchmark for Large-Scale Attacks in IoT Environment",
    venue="Sensors 23(13):5941",
    year=2023,
    doi="10.3390/s23135941",
)

_PROTOCOL = "random 80/20 split; StandardScaler"
_LOC = "Tab. 6 (PMC full text, PMC10346235)"

SPEC = BaselineSpec(
    name="neto-ciciot2023",
    title="Random forest and logistic regression on CIC-IoT-2023",
    reference=REFERENCE,
    family="window-classifier",
    input_schema=InputSchema(
        description="One row per CIC-IoT-2023 CSV record (46 features computed over a window of packets) with the class "
                    "name in `label` (34 classes).",
        required=ciciot2023.CICIOT2023_FEATURES,
        label="label",
        optional=("time", "dataset", "network", "split"),
    ),
    outputs=("detection",),
    datasets=("cic-iot-2023",),
    reported=(
        ReportedResult(dataset="cic-iot-2023", protocol=_PROTOCOL, task="binary", model="random_forest",
                       values={"accuracy": "0.99680798", "macro_recall": "0.965163906", "macro_precision": "0.965395244",
                               "macro_f1": "0.965279544"}, location=_LOC, variant={"learner": "random_forest"},
                       note="recall, precision and F1 appear averaged over the two classes"),
        ReportedResult(dataset="cic-iot-2023", protocol=_PROTOCOL, task="binary", model="logistic_regression",
                       values={"accuracy": "0.989023188", "macro_recall": "0.890400624", "macro_precision": "0.863157959",
                               "macro_f1": "0.876258983"}, location=_LOC, variant={"learner": "logistic_regression"},
                       note="recall, precision and F1 appear averaged over the two classes"),
        ReportedResult(dataset="cic-iot-2023", protocol=_PROTOCOL, task="multiclass-8", model="random_forest",
                       values={"macro_f1": "0.71928904"}, location=_LOC,
                       variant={"learner": "random_forest", "grouping": "multiclass-8", "task": "multiclass"},
                       note="averaging not stated; read as macro (AS-540)"),
        ReportedResult(dataset="cic-iot-2023", protocol=_PROTOCOL, task="multiclass-34", model="random_forest",
                       values={"macro_f1": "0.714021981"}, location=_LOC,
                       variant={"learner": "random_forest", "grouping": "multiclass-34", "task": "multiclass"},
                       note="averaging not stated; read as macro (AS-540)"),
    ),
    third_party="neto-ciciot2023",
    assumptions=("AS-530", "AS-532", "AS-533", "AS-540"),
)


class NetoCICIoT2023(TabularBaseline):
    """Neto et al. 2023: StandardScaler and a scikit-learn learner on the 46 CIC-IoT-2023 features."""

    spec: ClassVar[BaselineSpec] = SPEC
    config_type: ClassVar[type[BaselineConfig]] = NetoConfig
    config: NetoConfig

    def default_features(self, columns: Sequence[str]) -> tuple[str, ...]:
        return ciciot2023.CICIOT2023_FEATURES

    def _codes(self, frame: pd.DataFrame) -> np.ndarray:
        cfg = self.config
        if cfg.label_column not in frame.columns:
            return np.full(len(frame), -1, dtype=np.int64)
        return ciciot2023.class_index(frame[cfg.label_column], cfg.grouping)

    def _class_names(self, labels: pd.Series) -> tuple[str, ...]:
        # The class list is fixed by the grouping, not learned from the training labels.
        return ciciot2023.task_classes(self.config.grouping)

    @classmethod
    def paper_protocol(cls) -> SplitProtocol:
        """Random 80/20 split (baselines-notes.md, D3)."""
        return RandomHoldout(test_fraction=0.2)
