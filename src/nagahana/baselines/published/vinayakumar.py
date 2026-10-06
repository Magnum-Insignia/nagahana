"""Deep neural network and classical learners of Vinayakumar, Alazab, Soman, Poornachandran, Al-Nemrat and
Venkatraman, IEEE Access 7:41525-41550, 2019, "Deep Learning Approach for Intelligent Intrusion Detection
System", DOI 10.1109/ACCESS.2019.2895334.

What the paper states (baselines-notes.md, A1)
    data          CICIDS2017: a subset of 93,500 training and 28,481 test records, Normal 60,000 / 20,000,
                  attack classes SSH-Patator, FTP-Patator, DoS, Web, Bot, DDoS and PortScan (Tab. 4); the
                  sampling is not described. UNSW-NB15: the official partition (175,341 / 82,332).
    preprocessing the train and test data are L2-normalised
    tasks         binary (normal against attack) and multi-class
    models        DNNs with 1 to 5 hidden layers (best binary F1 with 1 layer on CICIDS2017, best
                  multi-class F1 with 3 layers); classical learners with scikit-learn (LR, NB, KNN, DT, AB,
                  RF, SVM-rbf)
    metrics       accuracy, precision, recall (TPR), F1; the multi-class averaging is not stated (accuracy
                  equals recall in every multi-class row, consistent with support weighting)

DNN as reproduced (AS-541; to verify against the official repository, tools/third_party)
    hidden layers 1024, 768, 512, 256, 128 (the first n_layers of them), each Dense -> ReLU -> batch
    normalisation -> dropout 0.01; output Dense with sigmoid (binary) or softmax (multi-class); Keras
    defaults for initialisation (Glorot uniform) and batch normalisation (epsilon 1e-3, momentum 0.99);
    Adam with learning rate 1e-3 and epsilon 1e-7; batch size 64; 100 epochs; no early stopping.

Classical learners: scikit-learn defaults on the same L2-normalised features (AS-541). UNSW-NB15's proto,
service and state are one-hot encoded before normalisation.

Subset of CICIDS2017 (AS-541): `cicids2017_subset` draws Normal 60,000 / 20,000 and the attack totals
33,500 / 8,481, split over the seven attack classes in proportion to their sizes; Infiltration and
Heartbleed are outside the paper's classes and are not drawn.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Literal

import numpy as np
import pandas as pd
import torch
from torch import nn

from nagahana.baselines.published.base import (
    BaselineConfig,
    BaselineSpec,
    InputSchema,
    PredictionParts,
    PublishedBaseline,
    Reference,
    ReportedResult,
    derive_seed,
)
from nagahana.baselines.published.features import cicflowmeter, unsw_nb15
from nagahana.baselines.published.frames import build_meta, normalise_token
from nagahana.baselines.published.neural import (
    KERAS_ADAM_EPS,
    keras_batchnorm,
    keras_dense_init,
    load_module,
    minibatches,
    resolve_device,
    save_module,
)
from nagahana.baselines.published.preprocessing import ColumnPipeline
from nagahana.baselines.published.protocols import GivenSplit, SplitProtocol
from nagahana.baselines.published.tabular import TabularBaseline, TabularConfig
from nagahana.core.errors import InvariantViolation
from nagahana.baselines.published.stages import attack_stages, class_stage_matrix, label_stage_codes
from nagahana.evaluation.predictions import DetectionPredictions, StagePredictions

Dataset = Literal["cicids2017", "unsw-nb15"]

#: The paper's CICIDS2017 classes (Tab. 4), Normal first.
CICIDS2017_CLASSES: tuple[str, ...] = ("Normal", "SSH-Patator", "FTP-Patator", "DoS", "Web", "Bot", "DDoS", "PortScan")
_CICIDS2017_MAP: dict[str, str] = {
    "benign": "Normal", "sshpatator": "SSH-Patator", "ftppatator": "FTP-Patator", "doshulk": "DoS",
    "dosgoldeneye": "DoS", "dosslowloris": "DoS", "dosslowhttptest": "DoS", "webattackbruteforce": "Web",
    "webattackxss": "Web", "webattacksqlinjection": "Web", "bot": "Bot", "ddos": "DDoS", "portscan": "PortScan",
}
_TRAIN_COUNTS = {"Normal": 60_000, "attacks": 33_500}
_TEST_COUNTS = {"Normal": 20_000, "attacks": 8_481}


def cicids2017_class(raw: Any) -> str | None:
    """Paper class of a CIC-IDS2017 label, or None for labels outside the paper's classes."""
    return _CICIDS2017_MAP.get(normalise_token(raw))


def dataset_classes(dataset: str, task: str) -> tuple[str, ...]:
    """Class names of a dataset and task."""
    if task == "binary":
        return ("benign", "attack")
    return CICIDS2017_CLASSES if dataset == "cicids2017" else unsw_nb15.UNSW_CLASSES


def default_label_column(dataset: str, task: str) -> str:
    """Label column of the released files: CICIDS2017 `Label`; UNSW-NB15 `label` (binary) or `attack_cat`."""
    if dataset == "cicids2017":
        return "Label"
    return "label" if task == "binary" else "attack_cat"


def default_stage_dataset(dataset: str) -> str:
    """Label table for the ATT&CK projection: CIC-IDS2017 has one (data/labels.py), UNSW-NB15 none."""
    return "cic-ids2017" if dataset == "cicids2017" else ""


def dataset_codes(frame: pd.DataFrame, dataset: str, task: str, label_column: str) -> np.ndarray:
    """Codes of the task: binary 1/0/-1 or class indices (-1 outside the paper's classes or unlabelled)."""
    if label_column not in frame.columns:
        return np.full(len(frame), -1, dtype=np.int64)
    raw = frame[label_column].tolist()
    classes = dataset_classes(dataset, task)
    if dataset == "cicids2017":
        mapped = [cicids2017_class(v) for v in raw]
        if task == "binary":
            return np.asarray([-1 if m is None else (0 if m == "Normal" else 1) for m in mapped], dtype=np.int64)
        return np.asarray([-1 if m is None else classes.index(m) for m in mapped], dtype=np.int64)
    if task == "binary":
        vals = pd.to_numeric(pd.Series(raw), errors="coerce").to_numpy(dtype=np.float64)
        return np.where(np.isin(vals, (0.0, 1.0)), vals, -1).astype(np.int64)
    index = {normalise_token(c): i for i, c in enumerate(classes)}
    index["backdoors"] = classes.index("Backdoor")
    return np.asarray([index.get(normalise_token(v), -1) for v in raw], dtype=np.int64)


def dataset_features(dataset: str, columns: Sequence[str]) -> tuple[str, ...]:
    """Default features: CICIDS2017 MachineLearningCSV features (harmonised), or the 42 UNSW-NB15 features."""
    if dataset == "cicids2017":
        return cicflowmeter.feature_columns(list(columns), drop=("Flow ID", "Src IP", "Src Port", "Dst IP", "Timestamp"))
    return unsw_nb15.partitioned_features()


def cicids2017_subset(frame: pd.DataFrame, *, seed: int, label_column: str = "Label") -> pd.DataFrame:
    """The paper's subset sizes drawn from a CIC-IDS2017 frame, with a `split` column (train / test).

    Normal: 60,000 train and 20,000 test rows. Attacks: 33,500 train and 8,481 test rows, allocated to
    the seven classes in proportion to their sizes (largest remainders), drawn without replacement and
    disjoint between train and test.
    """
    classes = np.asarray([cicids2017_class(v) or "" for v in frame[label_column].tolist()], dtype=object)
    rng = np.random.default_rng(derive_seed(seed, "vinayakumar-subset"))
    attack_names = [c for c in CICIDS2017_CLASSES if c != "Normal"]
    sizes = np.asarray([(classes == c).sum() for c in attack_names], dtype=np.float64)
    if sizes.sum() == 0:
        raise InvariantViolation("frame contains none of the paper's attack classes")

    def allocate(total: int) -> np.ndarray:
        quota = sizes / sizes.sum() * total
        base = np.floor(quota).astype(np.int64)
        base[np.argsort(-(quota - base), kind="stable")[: total - int(base.sum())]] += 1
        return base

    train_alloc, test_alloc = allocate(_TRAIN_COUNTS["attacks"]), allocate(_TEST_COUNTS["attacks"])
    picks_train: list[np.ndarray] = []
    picks_test: list[np.ndarray] = []
    for name, n_tr, n_te in [("Normal", _TRAIN_COUNTS["Normal"], _TEST_COUNTS["Normal"]),
                             *zip(attack_names, train_alloc.tolist(), test_alloc.tolist(), strict=True)]:
        idx = np.nonzero(classes == name)[0]
        if idx.size < n_tr + n_te:
            raise InvariantViolation(f"class {name} has {idx.size} rows; the subset needs {n_tr + n_te}")
        idx = idx[rng.permutation(idx.size)]
        picks_train.append(idx[:n_tr])
        picks_test.append(idx[n_tr:n_tr + n_te])
    train, test = np.sort(np.concatenate(picks_train)), np.sort(np.concatenate(picks_test))
    out = frame.iloc[np.concatenate([train, test])].reset_index(drop=True)
    out["split"] = ["train"] * train.size + ["test"] * test.size
    return out


REFERENCE = Reference(
    key="bl-vinayakumar2019dnn",
    authors="Vinayakumar, Alazab, Soman, Poornachandran, Al-Nemrat, Venkatraman",
    title="Deep Learning Approach for Intelligent Intrusion Detection System",
    venue="IEEE Access 7:41525-41550",
    year=2019,
    doi="10.1109/ACCESS.2019.2895334",
)

_P_CIC = "CICIDS2017 subset 93,500 / 28,481, single hold-out; L2-normalised"
_P_UNSW = "UNSW-NB15 official partition 175,341 / 82,332; L2-normalised"
_POS_NOTE = "positive class ambiguous in the source (Sec. III-D); compared as attack-class metrics"
_MC_NOTE = "averaging not stated; accuracy equals recall, read as support-weighted"


def _r(dataset: str, protocol: str, task: str, model: str, vals: tuple[str, str, str, str], loc: str,
       variant: dict[str, Any]) -> ReportedResult:
    keys = ("accuracy", "precision", "recall", "f1") if task == "binary" else (
        "accuracy", "weighted_precision", "weighted_recall", "weighted_f1")
    return ReportedResult(dataset=dataset, protocol=protocol, task=task, model=model, values=dict(zip(keys, vals, strict=True)),
                          location=loc, variant=variant, note=_POS_NOTE if task == "binary" else _MC_NOTE)


DNN_SPEC = BaselineSpec(
    name="vinayakumar-dnn",
    title="Deep neural network of Vinayakumar et al. (1 to 5 hidden layers)",
    reference=REFERENCE,
    family="flow-classifier",
    input_schema=InputSchema(
        description="One row per flow record: the CIC-IDS2017 MachineLearningCSV columns (any spelling) with `Label`, or "
                    "the UNSW-NB15 partitioned columns with `label` and `attack_cat`.",
        optional=("split", "time", "dataset", "network"),
    ),
    outputs=("detection",),
    datasets=("cic-ids2017", "unsw-nb15"),
    reported=(
        _r("cic-ids2017", _P_CIC, "binary", "dnn-1", ("0.963", "0.908", "0.973", "0.939"), "Tab. 9, p. 41542",
           {"dataset": "cicids2017", "task": "binary", "n_layers": 1}),
        _r("unsw-nb15", _P_UNSW, "binary", "dnn-1", ("0.784", "0.944", "0.725", "0.820"), "Tab. 9, p. 41542",
           {"dataset": "unsw-nb15", "task": "binary", "n_layers": 1}),
        _r("cic-ids2017", _P_CIC, "multiclass", "dnn-3", ("0.962", "0.972", "0.962", "0.965"), "Tab. 10, p. 41542",
           {"dataset": "cicids2017", "task": "multiclass", "n_layers": 3}),
        _r("unsw-nb15", _P_UNSW, "multiclass", "dnn-2", ("0.660", "0.623", "0.660", "0.596"), "Tab. 10, p. 41542",
           {"dataset": "unsw-nb15", "task": "multiclass", "n_layers": 2}),
    ),
    third_party="vinayakumar-dnn",
    assumptions=("AS-530", "AS-532", "AS-533", "AS-541"),
)

CLASSICAL_SPEC = BaselineSpec(
    name="vinayakumar-classical",
    title="Classical learners of Vinayakumar et al. (logistic regression, random forest, ...)",
    reference=REFERENCE,
    family="flow-classifier",
    input_schema=DNN_SPEC.input_schema,
    outputs=("detection",),
    datasets=("cic-ids2017", "unsw-nb15"),
    reported=(
        _r("cic-ids2017", _P_CIC, "binary", "logistic_regression", ("0.839", "0.685", "0.850", "0.758"), "Tab. 11, p. 41543",
           {"dataset": "cicids2017", "task": "binary", "learner": "logistic_regression"}),
        _r("cic-ids2017", _P_CIC, "binary", "random_forest", ("0.940", "0.849", "0.969", "0.905"), "Tab. 11, p. 41543",
           {"dataset": "cicids2017", "task": "binary", "learner": "random_forest"}),
        _r("unsw-nb15", _P_UNSW, "binary", "logistic_regression", ("0.743", "0.955", "0.653", "0.775"), "Tab. 11, p. 41543",
           {"dataset": "unsw-nb15", "task": "binary", "learner": "logistic_regression"}),
        _r("unsw-nb15", _P_UNSW, "binary", "random_forest", ("0.903", "0.988", "0.867", "0.924"), "Tab. 11, p. 41543",
           {"dataset": "unsw-nb15", "task": "binary", "learner": "random_forest"}),
        _r("cic-ids2017", _P_CIC, "multiclass", "logistic_regression", ("0.870", "0.889", "0.870", "0.868"), "Tab. 12, p. 41543",
           {"dataset": "cicids2017", "task": "multiclass", "learner": "logistic_regression"}),
        _r("cic-ids2017", _P_CIC, "multiclass", "random_forest", ("0.944", "0.970", "0.944", "0.953"), "Tab. 12, p. 41543",
           {"dataset": "cicids2017", "task": "multiclass", "learner": "random_forest"}),
        _r("unsw-nb15", _P_UNSW, "multiclass", "logistic_regression", ("0.538", "0.414", "0.538", "0.397"), "Tab. 12, p. 41543",
           {"dataset": "unsw-nb15", "task": "multiclass", "learner": "logistic_regression"}),
        _r("unsw-nb15", _P_UNSW, "multiclass", "random_forest", ("0.755", "0.755", "0.755", "0.724"), "Tab. 12, p. 41543",
           {"dataset": "unsw-nb15", "task": "multiclass", "learner": "random_forest"}),
        ReportedResult(dataset="cic-ids2017", protocol=_P_CIC, task="binary", model="logistic_regression",
                       values={"auc": "0.8504"}, location="Fig. 7(c), p. 41542 (legend)",
                       variant={"dataset": "cicids2017", "task": "binary", "learner": "logistic_regression"}),
        ReportedResult(dataset="cic-ids2017", protocol=_P_CIC, task="binary", model="random_forest",
                       values={"auc": "0.9947"}, location="Fig. 7(c), p. 41542 (legend)",
                       variant={"dataset": "cicids2017", "task": "binary", "learner": "random_forest"}),
    ),
    third_party="vinayakumar-dnn",
    assumptions=("AS-530", "AS-532", "AS-533", "AS-541"),
)


@dataclass
class VinayakumarClassicalConfig(TabularConfig):
    """A scikit-learn learner of the study on L2-normalised features.

    Attributes
    ----------
    dataset:
        "cicids2017" or "unsw-nb15" (selects features, classes and label column).
    """

    learner: str = "logistic_regression"
    scaler: Literal["none", "minmax", "standard", "l2"] = "l2"
    categorical: tuple[str, ...] = unsw_nb15.UNSW_CATEGORICAL
    label_column: str = ""
    stage_dataset: str = "auto"
    dataset: Dataset = "cicids2017"

    def validate(self) -> None:
        super().validate()
        if self.learner not in ("logistic_regression", "gaussian_nb", "knn", "decision_tree", "adaboost", "random_forest", "svm_rbf"):
            raise ValueError("the study's classical learners are LR, NB, KNN, DT, AB, RF and SVM-rbf")


class VinayakumarClassical(TabularBaseline):
    """Vinayakumar et al. 2019, classical learners (Tables 11 and 12)."""

    spec: ClassVar[BaselineSpec] = CLASSICAL_SPEC
    config_type: ClassVar[type[BaselineConfig]] = VinayakumarClassicalConfig
    config: VinayakumarClassicalConfig

    def __init__(self, config: BaselineConfig | None = None) -> None:
        super().__init__(config)
        # The label column defaults to the dataset's own; the tabular pipeline reads config.label_column,
        # so the resolved value goes into a copy of the config (the caller's object is left unchanged).
        if not self.config.label_column:
            self.config = dataclasses.replace(self.config, label_column=default_label_column(self.config.dataset, self.config.task))
        if self.config.stage_dataset == "auto":
            self.config = dataclasses.replace(self.config, stage_dataset=default_stage_dataset(self.config.dataset))

    def prepare(self, frame: pd.DataFrame) -> pd.DataFrame:
        return cicflowmeter.clean(cicflowmeter.harmonise(frame)) if self.config.dataset == "cicids2017" else frame

    def default_features(self, columns: Sequence[str]) -> tuple[str, ...]:
        return dataset_features(self.config.dataset, columns)

    def _class_names(self, labels: pd.Series) -> tuple[str, ...]:
        return dataset_classes(self.config.dataset, "multiclass")

    def _codes(self, frame: pd.DataFrame) -> np.ndarray:
        cfg = self.config
        return dataset_codes(frame, cfg.dataset, cfg.task, cfg.label_column)

    @classmethod
    def paper_protocol(cls) -> SplitProtocol:
        """The paper's fixed train / test sets, carried in the frame's `split` column."""
        return GivenSplit(train=("train",), validation=(), test=("test",))


class VinayakumarNet(nn.Module):
    """Dense -> ReLU -> BatchNorm -> Dropout blocks, then a linear output layer."""

    def __init__(self, d_in: int, hidden: tuple[int, ...], dropout: float, batch_norm: bool, n_out: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = d_in
        for width in hidden:
            layers += [nn.Linear(prev, width), nn.ReLU()]
            if batch_norm:
                layers.append(keras_batchnorm(width))
            layers.append(nn.Dropout(dropout))
            prev = width
        layers.append(nn.Linear(prev, n_out))
        self.net = nn.Sequential(*layers)
        keras_dense_init(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Logits [B, n_out] for inputs [B, d_in]."""
        out: torch.Tensor = self.net(x)
        return out


@dataclass
class VinayakumarDNNConfig(BaselineConfig):
    """The DNN of the study (see the module docstring for the source of each value).

    Attributes
    ----------
    dataset, task:
        Dataset ("cicids2017" or "unsw-nb15") and task ("binary" or "multiclass").
    n_layers:
        Number of hidden layers, 1 ... 5 (the first n of `hidden_units`).
    hidden_units, dropout, batch_norm:
        Architecture.
    learning_rate, batch_size, epochs:
        Adam training settings.
    label_column:
        Label column; empty selects the dataset's own (`Label`, `label` or `attack_cat`).
    features:
        Explicit feature columns; empty selects the dataset's default list.
    categorical:
        Columns one-hot encoded (UNSW-NB15's proto, service, state).
    threshold:
        Operating threshold on P(attack).
    """

    dataset: Dataset = "cicids2017"
    task: Literal["binary", "multiclass"] = "binary"
    n_layers: int = 1
    hidden_units: tuple[int, ...] = (1024, 768, 512, 256, 128)
    dropout: float = 0.01
    batch_norm: bool = True
    learning_rate: float = 0.001
    batch_size: int = 64
    epochs: int = 100
    label_column: str = ""
    features: tuple[str, ...] = ()
    categorical: tuple[str, ...] = unsw_nb15.UNSW_CATEGORICAL
    threshold: float = 0.5
    stage_dataset: str = "auto"

    def validate(self) -> None:
        super().validate()
        if not 1 <= self.n_layers <= len(self.hidden_units):
            raise ValueError(f"n_layers must lie in 1 ... {len(self.hidden_units)}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must lie in [0, 1)")
        if self.learning_rate <= 0 or self.batch_size < 1 or self.epochs < 1:
            raise ValueError("learning_rate, batch_size and epochs must be positive")


class VinayakumarDNN(PublishedBaseline):
    """Vinayakumar et al. 2019, DNN with 1 ... 5 hidden layers (Tables 9 and 10)."""

    spec: ClassVar[BaselineSpec] = DNN_SPEC
    config_type: ClassVar[type[BaselineConfig]] = VinayakumarDNNConfig
    config: VinayakumarDNNConfig

    def __init__(self, config: BaselineConfig | None = None) -> None:
        super().__init__(config)
        self.pipeline: ColumnPipeline | None = None
        self.net: VinayakumarNet | None = None
        self.classes_: tuple[str, ...] = dataset_classes(self.config.dataset, self.config.task)
        self.class_stage_: np.ndarray | None = None
        if self.config.stage_dataset == "auto":
            self.config = dataclasses.replace(self.config, stage_dataset=default_stage_dataset(self.config.dataset))

    @property
    def label_column(self) -> str:
        return self.config.label_column or default_label_column(self.config.dataset, self.config.task)

    def prepare(self, frame: pd.DataFrame) -> pd.DataFrame:
        return cicflowmeter.clean(cicflowmeter.harmonise(frame)) if self.config.dataset == "cicids2017" else frame

    def required_columns(self) -> tuple[str, ...]:
        return tuple(self.config.features)

    def _features(self, frame: pd.DataFrame) -> tuple[str, ...]:
        feats = self.config.features or dataset_features(self.config.dataset, list(frame.columns))
        missing = [c for c in feats if c not in frame.columns]
        if missing:
            raise InvariantViolation(f"{self.spec.name}: frame lacks feature columns {missing}")
        return tuple(feats)

    def _fit(self, data: pd.DataFrame, validation: pd.DataFrame | None, rng: np.random.Generator) -> None:
        cfg = self.config
        if self.label_column not in data.columns:
            raise InvariantViolation(f"{self.spec.name}: training frame lacks label column {self.label_column!r}")
        feats = self._features(data)
        codes = dataset_codes(data, cfg.dataset, cfg.task, self.label_column)
        categorical = tuple(c for c in cfg.categorical if c in feats)
        numeric = tuple(c for c in feats if c not in categorical)
        pipeline = ColumnPipeline(numeric, categorical, "l2")
        keep = (codes >= 0) & ~pipeline.nonfinite_rows(data)
        train = data.loc[keep].reset_index(drop=True)
        y = codes[keep]
        if train.empty or np.unique(y).size < 2:
            raise InvariantViolation(f"{self.spec.name}: training data need rows of at least two classes")
        pipeline.fit(train)
        device = resolve_device(cfg.device)
        x = torch.as_tensor(pipeline.transform(train), dtype=torch.float32, device=device)
        n_out = 1 if cfg.task == "binary" else len(self.classes_)
        net = VinayakumarNet(x.shape[1], cfg.hidden_units[: cfg.n_layers], cfg.dropout, cfg.batch_norm, n_out).to(device)
        opt = torch.optim.Adam(net.parameters(), lr=cfg.learning_rate, eps=KERAS_ADAM_EPS)
        target = torch.as_tensor(y, device=device)
        losses: list[float] = []
        for _epoch in range(cfg.epochs):
            net.train()
            total, count = 0.0, 0
            for idx in minibatches(x.shape[0], cfg.batch_size, rng):
                if idx.size < 2 and cfg.batch_norm:
                    continue                      # batch normalisation needs two rows per batch in training
                ib = torch.as_tensor(idx, device=device)
                logits = net(x[ib])
                if cfg.task == "binary":
                    loss = nn.functional.binary_cross_entropy_with_logits(logits[:, 0], target[ib].float())
                else:
                    loss = nn.functional.cross_entropy(logits, target[ib])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                total += float(loss.detach()) * idx.size
                count += idx.size
            losses.append(total / max(count, 1))
        self.pipeline, self.net = pipeline, net
        self.class_stage_ = None
        if cfg.task == "multiclass" and cfg.stage_dataset:
            # P(tactic | class) from the training rows' raw labels (stages.py, AS-563).
            stage = label_stage_codes(train[self.label_column].tolist(), cfg.stage_dataset)
            self.class_stage_ = class_stage_matrix(y, stage, len(self.classes_))
        self.fit_report.update({"train_rows": int(train.shape[0]), "features": list(feats), "epochs": cfg.epochs,
                                "final_loss": losses[-1] if losses else float("nan"), "loss_curve": losses})

    @torch.no_grad()
    def _probs(self, frame: pd.DataFrame) -> np.ndarray:
        assert self.pipeline is not None and self.net is not None
        device = resolve_device(self.config.device)
        self.net.eval()
        x = torch.as_tensor(self.pipeline.transform(frame), dtype=torch.float32, device=device)
        logits = torch.cat([self.net(x[i:i + 8192]) for i in range(0, x.shape[0], 8192)], dim=0).double()
        if self.config.task == "binary":
            p1 = torch.sigmoid(logits[:, 0])
            return torch.stack([1.0 - p1, p1], dim=1).cpu().numpy()
        return torch.softmax(logits, dim=1).cpu().numpy()

    def _predict(self, data: pd.DataFrame) -> PredictionParts:
        cfg = self.config
        probs = self._probs(data)
        codes = dataset_codes(data, cfg.dataset, cfg.task, self.label_column)
        component: dict[str, np.ndarray] = {}
        if cfg.task == "binary":
            score, label = probs[:, 1], codes
        else:
            score = 1.0 - probs[:, 0]                      # class 0 is Normal in both datasets
            label = np.where(codes < 0, -1, (codes != 0).astype(np.int64))
            component["class_probs"] = probs
            component["class_label"] = codes
        family = np.where(label == 1, "attack", np.where(label == 0, "benign", "unknown"))
        meta = build_meta(data, family=family)
        det = DetectionPredictions(score=score, label=label, unit="flow", meta=meta, threshold=cfg.threshold)
        stage = None
        if self.class_stage_ is not None:
            known = self.label_column in data.columns
            stage_label = label_stage_codes(data[self.label_column].tolist(), cfg.stage_dataset) if known else np.full(len(data), -1)
            stage = StagePredictions(probs=probs @ self.class_stage_, label=stage_label, stage_names=attack_stages(), meta=meta)
        return PredictionParts(detection=det, stage=stage, component=component)

    def _export_state(self, directory: Path) -> dict[str, Any]:
        assert self.pipeline is not None and self.net is not None
        return {"pipeline": self.pipeline.state(), "weights": save_module(self.net, directory / "dnn.pt"),
                "d_in": int(self.net.net[0].in_features),
                "class_stage": None if self.class_stage_ is None else self.class_stage_.tolist()}

    def _import_state(self, directory: Path, state: Mapping[str, Any]) -> None:
        cfg = self.config
        self.pipeline = ColumnPipeline.from_state(state["pipeline"])
        cs = state.get("class_stage")
        self.class_stage_ = None if cs is None else np.asarray(cs, dtype=np.float64)
        n_out = 1 if cfg.task == "binary" else len(self.classes_)
        net = VinayakumarNet(int(state["d_in"]), cfg.hidden_units[: cfg.n_layers], cfg.dropout, cfg.batch_norm, n_out)
        self.net = load_module(net, directory / str(state["weights"]), resolve_device(cfg.device))

    @classmethod
    def paper_protocol(cls) -> SplitProtocol:
        """The paper's fixed train / test sets, carried in the frame's `split` column."""
        return GivenSplit(train=("train",), validation=(), test=("test",))
