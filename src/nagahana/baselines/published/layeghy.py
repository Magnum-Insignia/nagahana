"""Generalisability study of Layeghy and Portmann, arXiv:2205.04112, 2022, "On Generalisability of Machine
Learning-based Network Intrusion Detection Systems" (preprint, not peer reviewed).

What the paper states (baselines-notes.md, B6)
    data          NetFlow v2 datasets (NFv2-CIC-2018 = NF-CSE-CIC-IDS2018-v2, NFv2-UNSW-NB15, ...); a
                  stratified sample of 1,000,000 flows per dataset; binary benign / attack labels
    models        extra trees, random forest, a feed-forward network of 5 hidden layers with 10 nodes, and
                  an LSTM with the same number of layers and nodes
    settings      library default hyperparameters, "in order to avoid over-fitting to the training
                  datasets"
    protocol      within one dataset, and across datasets (train on one, test on another)
    metric        F1

What the paper leaves open (AS-556)
    the train/test ratio of the single-dataset runs (70/30, stratified, the convention of the authors'
    group); preprocessing (identifiers removed, min-max scaling, as in Sarhan et al. 2022); how the LSTM's
    input sequences are formed (each flow is a sequence of length 1, the common reshaping of tabular
    rows for a recurrent layer; longer windows of consecutive flows are a config option); the LSTM's
    training settings (Keras defaults: Adam 1e-3, batch 32, sigmoid output, binary cross-entropy;
    10 epochs, since Keras's default of one epoch would leave the network nearly untrained). The
    feed-forward network is scikit-learn's MLPClassifier with hidden_layer_sizes = (10,) * 5 and every
    other parameter at its default.
"""

from __future__ import annotations

from collections.abc import Mapping
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
)
from nagahana.baselines.published.estimators import TabularModel
from nagahana.baselines.published.features import netflow
from nagahana.baselines.published.frames import binary_labels, build_meta, normalise_token, time_order
from nagahana.baselines.published.neural import (
    KERAS_ADAM_EPS,
    causal_windows,
    keras_dense_init,
    keras_lstm_init,
    load_module,
    minibatches,
    resolve_device,
    save_module,
)
from nagahana.baselines.published.preprocessing import ColumnPipeline
from nagahana.baselines.published.protocols import CrossDataset, RandomHoldout, SplitProtocol
from nagahana.core.errors import InvariantViolation
from nagahana.evaluation.predictions import DetectionPredictions

Model = Literal["extra_trees", "random_forest", "feed_forward", "lstm"]


@dataclass
class LayeghyConfig(BaselineConfig):
    """One model of the study.

    Attributes
    ----------
    model:
        "extra_trees", "random_forest", "feed_forward" or "lstm".
    hidden_layers, hidden_units:
        Depth and width of the feed-forward network and of the LSTM stack (5 x 10 in the paper).
    features:
        Explicit feature columns; empty selects the NetFlow v2 features without identifiers.
    label_column, family_column:
        Binary label (0 / 1) and class-name columns of the NF-v2 files.
    sequence_length:
        Flows per LSTM input sequence (the flow itself and its predecessors in time order).
    learning_rate, batch_size, epochs:
        LSTM training settings.
    threshold:
        Operating threshold on P(attack).
    """

    model: Model = "lstm"
    hidden_layers: int = 5
    hidden_units: int = 10
    features: tuple[str, ...] = ()
    label_column: str = "Label"
    family_column: str = "Attack"
    sequence_length: int = 1
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 10
    threshold: float = 0.5

    def validate(self) -> None:
        super().validate()
        if self.hidden_layers < 1 or self.hidden_units < 1 or self.sequence_length < 1:
            raise ValueError("hidden_layers, hidden_units and sequence_length must be >= 1")
        if self.learning_rate <= 0 or self.batch_size < 1 or self.epochs < 1:
            raise ValueError("learning_rate, batch_size and epochs must be positive")


class LayeghyLSTM(nn.Module):
    """A stack of LSTM layers over a flow sequence, then a sigmoid unit on the last step's output."""

    def __init__(self, d_in: int, hidden: int, layers: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(d_in, hidden, num_layers=layers, batch_first=True)
        self.head = nn.Linear(hidden, 1)
        keras_lstm_init(self.lstm)
        keras_dense_init(self.head)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Logits [B] for sequences [B, L, d_in]."""
        out, _ = self.lstm(x)                                   # [B, L, hidden]
        logit: torch.Tensor = self.head(out[:, -1, :])[:, 0]    # [B]
        return logit


REFERENCE = Reference(
    key="bl-layeghy2022generalisability",
    authors="Layeghy, Portmann",
    title="On Generalisability of Machine Learning-based Network Intrusion Detection Systems",
    venue="arXiv preprint",
    year=2022,
    arxiv="2205.04112",
    note="preprint, not peer reviewed",
)

_MODELS = ("extra_trees", "random_forest", "feed_forward", "lstm")


def _rows() -> tuple[ReportedResult, ...]:
    same = {"nf-cse-cic-ids2018-v2": ("84.62", "95.44", "46.27", "90.17"), "nf-unsw-nb15-v2": ("91.73", "92.17", "90.63", "92.82")}
    cross = {"nf-unsw-nb15-v2->nf-cse-cic-ids2018-v2": ("17.47", "7.70", "34.89", "14.20"),
             "nf-cse-cic-ids2018-v2->nf-unsw-nb15-v2": ("0.57", "0.84", "0.05", "9.63")}
    out = []
    for ds, vals in same.items():
        for m, v in zip(_MODELS, vals, strict=True):
            out.append(ReportedResult(dataset=ds, protocol="1M-flow stratified sample; binary; default settings", task="binary",
                                      model=m, values={"f1": v}, location="Tab. 2, p. 3", variant={"model": m}, percent=("f1",)))
    for ds, vals in cross.items():
        for m, v in zip(_MODELS, vals, strict=True):
            out.append(ReportedResult(dataset=ds, protocol="train on the source sample, test on the target sample; binary",
                                      task="binary", model=m, values={"f1": v}, location="Tab. 4, p. 5", variant={"model": m},
                                      percent=("f1",)))
    return tuple(out)


SPEC = BaselineSpec(
    name="layeghy-generalisation",
    title="Extra trees, random forest, feed-forward network and LSTM of Layeghy and Portmann",
    reference=REFERENCE,
    family="flow-classifier",
    input_schema=InputSchema(
        description="One row per NetFlow v2 flow record (NF-*-v2 columns) with the binary `Label`; rows in time order "
                    "(or a `time` column) when the LSTM reads windows of consecutive flows.",
        label="Label",
        optional=("Attack", "time", "dataset", "network", "split"),
    ),
    outputs=("detection",),
    datasets=("nf-cse-cic-ids2018-v2", "nf-unsw-nb15-v2"),
    reported=_rows(),
    third_party="layeghy-generalisation",
    assumptions=("AS-530", "AS-532", "AS-533", "AS-556"),
)


class LayeghyGeneralisation(PublishedBaseline):
    """Layeghy and Portmann 2022: one of ET, RF, FF (sklearn defaults) or a Keras-default LSTM."""

    spec: ClassVar[BaselineSpec] = SPEC
    config_type: ClassVar[type[BaselineConfig]] = LayeghyConfig
    config: LayeghyConfig

    def __init__(self, config: BaselineConfig | None = None) -> None:
        super().__init__(config)
        self.pipeline: ColumnPipeline | None = None
        self.tabular: TabularModel | None = None
        self.lstm: LayeghyLSTM | None = None
        self.features_: tuple[str, ...] = ()

    def required_columns(self) -> tuple[str, ...]:
        return tuple(self.config.features)

    def _features(self, frame: pd.DataFrame) -> tuple[str, ...]:
        feats = self.config.features or netflow.standard_features(2, drop_identifiers=True)
        missing = [c for c in feats if c not in frame.columns]
        if missing:
            raise InvariantViolation(f"{self.spec.name}: frame lacks feature columns {missing}")
        return tuple(feats)

    def _sequences(self, x: np.ndarray, frame: pd.DataFrame) -> np.ndarray:
        # [n, L, d] windows of consecutive flows in time order; positions before the first flow are zeros.
        order = time_order(frame, "time" if "time" in frame.columns else None)
        xo = x[order]
        win = causal_windows(xo.shape[0], self.config.sequence_length)
        seq = np.where((win >= 0)[:, :, None], xo[np.clip(win, 0, None)], 0.0)
        out = np.empty_like(seq)
        out[order] = seq
        return out

    def _fit(self, data: pd.DataFrame, validation: pd.DataFrame | None, rng: np.random.Generator) -> None:
        cfg = self.config
        self.features_ = self._features(data)
        y = binary_labels(data[cfg.label_column])
        pipeline = ColumnPipeline(self.features_, (), "minmax")
        keep = (y >= 0) & ~pipeline.nonfinite_rows(data)
        if np.unique(y[keep]).size < 2:
            raise InvariantViolation(f"{self.spec.name}: training data need both classes")
        if cfg.model == "lstm" and cfg.sequence_length > 1:
            # Windows need the full ordered stream: fit the scaler on the kept rows, window over all rows.
            pipeline.fit(data.loc[keep].reset_index(drop=True))
            x_all = self._sequences(pipeline.transform(data), data)
            x_train, y_train = x_all[keep], y[keep]
        else:
            train = data.loc[keep].reset_index(drop=True)
            pipeline.fit(train)
            x2 = pipeline.transform(train)
            x_train, y_train = (x2[:, None, :] if cfg.model == "lstm" else x2), y[keep]
        self.pipeline = pipeline
        if cfg.model == "lstm":
            self._fit_lstm(x_train, y_train, rng)
        else:
            learner = {"extra_trees": "extra_trees", "random_forest": "random_forest", "feed_forward": "mlp"}[cfg.model]
            params: dict[str, Any] = {"hidden_layer_sizes": (cfg.hidden_units,) * cfg.hidden_layers} if cfg.model == "feed_forward" else {}
            self.tabular = TabularModel(learner, params, cfg.seed, 2, needed_by=self.spec.name).fit(x_train, y_train)
        self.fit_report.update({"train_rows": int(keep.sum()), "features": list(self.features_)})

    def _fit_lstm(self, x: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> None:
        cfg = self.config
        device = resolve_device(cfg.device)
        net = LayeghyLSTM(x.shape[2], cfg.hidden_units, cfg.hidden_layers).to(device)
        opt = torch.optim.Adam(net.parameters(), lr=cfg.learning_rate, eps=KERAS_ADAM_EPS)
        xt = torch.as_tensor(x, dtype=torch.float32, device=device)
        yt = torch.as_tensor(y, dtype=torch.float32, device=device)
        curve: list[float] = []
        for _epoch in range(cfg.epochs):
            net.train()
            total = 0.0
            for idx in minibatches(xt.shape[0], cfg.batch_size, rng):
                ib = torch.as_tensor(idx, device=device)
                loss = nn.functional.binary_cross_entropy_with_logits(net(xt[ib]), yt[ib])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                total += float(loss.detach()) * idx.size
            curve.append(total / xt.shape[0])
        self.lstm = net
        self.fit_report["loss_curve"] = curve

    @torch.no_grad()
    def _score(self, frame: pd.DataFrame) -> np.ndarray:
        assert self.pipeline is not None
        cfg = self.config
        x2 = self.pipeline.transform(frame)
        if cfg.model != "lstm":
            assert self.tabular is not None
            return self.tabular.predict_proba(x2)[:, 1]
        assert self.lstm is not None
        x = self._sequences(x2, frame) if cfg.sequence_length > 1 else x2[:, None, :]
        device = resolve_device(cfg.device)
        self.lstm.eval()
        xt = torch.as_tensor(x, dtype=torch.float32, device=device)
        logits = torch.cat([self.lstm(xt[i:i + 8192]) for i in range(0, xt.shape[0], 8192)]).double()
        return torch.sigmoid(logits).cpu().numpy()

    def _predict(self, data: pd.DataFrame) -> PredictionParts:
        cfg = self.config
        score = self._score(data)
        label = binary_labels(data[cfg.label_column]) if cfg.label_column in data.columns else np.full(len(data), -1, dtype=np.int64)
        if cfg.family_column in data.columns:
            family = np.asarray(["benign" if normalise_token(v) == "benign" else str(v) for v in data[cfg.family_column]], dtype=object)
        else:
            family = np.where(label == 1, "attack", np.where(label == 0, "benign", "unknown"))
        det = DetectionPredictions(score=score, label=label, unit="flow", meta=build_meta(data, family=family), threshold=cfg.threshold)
        return PredictionParts(detection=det)

    def _export_state(self, directory: Path) -> dict[str, Any]:
        assert self.pipeline is not None
        state: dict[str, Any] = {"pipeline": self.pipeline.state(), "features": list(self.features_)}
        if self.config.model == "lstm":
            assert self.lstm is not None
            state["weights"] = save_module(self.lstm, directory / "lstm.pt")
            state["d_in"] = int(self.lstm.lstm.input_size)
        else:
            assert self.tabular is not None
            state["model"] = self.tabular.save(directory, "model")
        return state

    def _import_state(self, directory: Path, state: Mapping[str, Any]) -> None:
        cfg = self.config
        self.pipeline = ColumnPipeline.from_state(state["pipeline"])
        self.features_ = tuple(state["features"])
        if cfg.model == "lstm":
            net = LayeghyLSTM(int(state["d_in"]), cfg.hidden_units, cfg.hidden_layers)
            self.lstm = load_module(net, directory / str(state["weights"]), resolve_device(cfg.device))
        else:
            self.tabular = TabularModel.load(directory, state["model"], needed_by=self.spec.name)

    @classmethod
    def paper_protocol(cls, train_dataset: str | None = None, test_dataset: str | None = None) -> SplitProtocol:
        """70/30 stratified within one dataset (AS-556); train on one, test on another across datasets."""
        if train_dataset is not None and test_dataset is not None and train_dataset != test_dataset:
            return CrossDataset(train_dataset, test_dataset)
        return RandomHoldout(test_fraction=0.3, stratify="Label")
