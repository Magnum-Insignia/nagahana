"""FlowTransformer reproduction (Manocchio et al., Expert Systems with Applications 241:122564, 2024,
DOI 10.1016/j.eswa.2023.122564; values from arXiv:2304.14746).

What the paper states (baselines-notes.md, B4)
    data          NetFlow versions of the datasets (Sarhan et al.'s flow format)
    protocol      90 % / 10 % train / evaluation split, pre-processing fitted on the training data; a grid
                  over input encodings, transformers and heads; at least 3 repeats per configuration, the
                  best repeat reported; early stopping with patience 5, at most 20 epochs
    model         basic transformer of 2 layers, 2 heads, feed-forward size 128 (Tables II and V)
    task          binary; metrics F1, detection rate, false-alarm rate
    results       CSE-CIC-IDS2018: best F1 0.9705 (featurewise-embedding head, record-level dense
                  encoding), DR 95.96 %, FAR 0.10 %; global-average-pooling and CLS heads 0.3199 - 0.3615.
                  UNSW-NB15: best F1 0.9045 (last-token head, record-level dense encoding), DR 99.89 %,
                  FAR 1.03 %

What the paper leaves open (AS-542 ... AS-545)
    the field selection and pre-processing (features.netflow, preprocessing.py), the window of T = 8
    flows, the block form and the absence of positional encoding (model.py), the context of a flow
    (the T - 1 flows preceding it in time order, globally; per source address as an option), the label
    of a window (the label of its last flow), class-balanced batches of 128, Adam with Keras defaults,
    an epoch of ceil(n / 128) batches, early stopping on the binary cross-entropy of a validation split
    (the last 10 % of the training rows in time order), and the 90 / 10 split read as the last 10 % of
    the rows in time order (the framework's "last rows" evaluation sampling).
"""

from __future__ import annotations

import math
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
from nagahana.baselines.published.features import netflow
from nagahana.baselines.published.flowtransformer.model import ENCODINGS, HEADS, FlowTransformerNet
from nagahana.baselines.published.flowtransformer.preprocessing import FlowTransformerPreprocessing
from nagahana.baselines.published.frames import binary_labels, build_meta, normalise_token, time_order
from nagahana.baselines.published.neural import (
    KERAS_ADAM_EPS,
    EarlyStopping,
    balanced_minibatches,
    causal_windows,
    load_module,
    minibatches,
    resolve_device,
    save_module,
)
from nagahana.baselines.published.protocols import LastRows, SplitProtocol
from nagahana.core.errors import InvariantViolation
from nagahana.evaluation.predictions import DetectionPredictions

Encoding = Literal["none", "record_dense", "record_projection", "categorical_dense", "categorical_lookup", "categorical_projection"]
Head = Literal["last_token", "flatten", "global_average_pooling", "cls_token", "featurewise"]


@dataclass
class FlowTransformerConfig(BaselineConfig):
    """FlowTransformer settings (see the module docstring for the source of each default).

    Attributes
    ----------
    fields, categorical:
        Input fields and the subset treated as categorical.
    label_column, family_column, time_column:
        Binary label, class name and event-time columns (rows keep file order without a time column).
    context, source_column:
        "global": a flow's context is the preceding flows of the stream; "source": the preceding flows of
        the same `source_column` value.
    n_categorical_levels, clip_numeric:
        Pre-processing (preprocessing.py).
    window:
        Flows per window (the classified flow and its T - 1 predecessors).
    encoding, embed_dim, categorical_embed_dim:
        Input encoding (model.py) and its widths.
    transformer:
        "encoder" or "decoder" (causal attention).
    n_layers, ff_dim, n_heads, head_dim, dropout:
        Transformer blocks; head_dim 0 means d_model // n_heads.
    head, featurewise_dim:
        Classification head and the per-position width of the featurewise head.
    mlp_sizes, mlp_dropout:
        MLP after the head.
    learning_rate, batch_size, max_epochs, patience, steps_per_epoch, balanced_batches:
        Training; steps_per_epoch 0 means ceil(n_train / batch_size).
    validation_fraction:
        Share of the training rows (the latest) held out for early stopping when no validation frame is
        given.
    threshold:
        Operating threshold on P(attack).
    """

    fields: tuple[str, ...] = netflow.FLOWTRANSFORMER_FIELDS
    categorical: tuple[str, ...] = netflow.FLOWTRANSFORMER_CATEGORICAL
    label_column: str = "Label"
    family_column: str = "Attack"
    time_column: str = "time"
    context: Literal["global", "source"] = "global"
    source_column: str = "IPV4_SRC_ADDR"
    n_categorical_levels: int = 32
    clip_numeric: bool = False
    window: int = 8
    encoding: Encoding = "record_dense"
    embed_dim: int = 64
    categorical_embed_dim: int = 16
    transformer: Literal["encoder", "decoder"] = "encoder"
    n_layers: int = 2
    ff_dim: int = 128
    n_heads: int = 2
    head_dim: int = 0
    dropout: float = 0.1
    head: Head = "featurewise"
    featurewise_dim: int = 1
    mlp_sizes: tuple[int, ...] = (128,)
    mlp_dropout: float = 0.1
    learning_rate: float = 0.001
    batch_size: int = 128
    max_epochs: int = 20
    patience: int = 5
    steps_per_epoch: int = 0
    balanced_batches: bool = True
    validation_fraction: float = 0.1
    threshold: float = 0.5

    def validate(self) -> None:
        super().validate()
        if self.encoding not in ENCODINGS or self.head not in HEADS:
            raise ValueError("unknown encoding or head")
        unknown_cat = [c for c in self.categorical if c not in self.fields]
        if unknown_cat:
            raise ValueError(f"categorical fields {unknown_cat} are not among the input fields")
        if self.window < 1 or self.n_layers < 0 or self.n_heads < 1 or self.ff_dim < 1 or self.featurewise_dim < 1:
            raise ValueError("window, n_heads, ff_dim, featurewise_dim must be >= 1 and n_layers >= 0")
        if not 0.0 <= self.dropout < 1.0 or not 0.0 <= self.mlp_dropout < 1.0:
            raise ValueError("dropout rates must lie in [0, 1)")
        if self.learning_rate <= 0 or self.batch_size < 1 or self.max_epochs < 1 or self.patience < 1:
            raise ValueError("learning_rate, batch_size, max_epochs and patience must be positive")
        if not 0.0 <= self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must lie in [0, 1)")


REFERENCE = Reference(
    key="bl-manocchio2024flowtransformer",
    authors="Manocchio, Layeghy, Lo, Kulatilleke, Sarhan, Portmann",
    title="FlowTransformer: A transformer framework for flow-based network intrusion detection systems",
    venue="Expert Systems with Applications 241:122564",
    year=2024,
    doi="10.1016/j.eswa.2023.122564",
    arxiv="2304.14746",
    note="values taken from arXiv:2304.14746",
)

_PROTOCOL = "90/10 split; binary; best of >= 3 repeats; basic transformer 2 layers, 2 heads, ff 128"

SPEC = BaselineSpec(
    name="flowtransformer",
    title="FlowTransformer (transformer over sequences of flow records)",
    reference=REFERENCE,
    family="sequence-classifier",
    input_schema=InputSchema(
        description="One row per NetFlow v2 flow record (NF-*-v2 columns) with the binary `Label`; rows in time order or "
                    "with a `time` column, since a flow's context is the flows that precede it.",
        label="Label",
        optional=("Attack", "time", "IPV4_SRC_ADDR", "dataset", "network", "split"),
    ),
    outputs=("detection",),
    datasets=("nf-cse-cic-ids2018", "nf-unsw-nb15"),
    reported=(
        ReportedResult(dataset="nf-cse-cic-ids2018", protocol=_PROTOCOL, task="binary", model="featurewise+record_dense",
                       values={"f1": "0.9705", "dr": "95.96%", "far": "0.10%"}, location="Tab. II, p. 10 (arXiv PDF)",
                       variant={"head": "featurewise", "encoding": "record_dense"}),
        ReportedResult(dataset="nf-unsw-nb15", protocol=_PROTOCOL, task="binary", model="last_token+record_dense",
                       values={"f1": "0.9045", "dr": "99.89%", "far": "1.03%"}, location="Tab. V, p. 15 (arXiv PDF)",
                       variant={"head": "last_token", "encoding": "record_dense"}),
        ReportedResult(dataset="nf-cse-cic-ids2018", protocol=_PROTOCOL, task="binary", model="global_average_pooling heads",
                       values={"f1": "0.3311"}, location="Tab. II, p. 10 (arXiv PDF)", variant={"head": "global_average_pooling"},
                       note="the six GAP rows print F1 0.3311, 0.3466, 0.3379, 0.3371, 0.3395 and 0.351 (FAR 17.19 - 33.91 %); "
                            "the first is listed"),
        ReportedResult(dataset="nf-cse-cic-ids2018", protocol=_PROTOCOL, task="binary", model="cls_token heads",
                       values={"f1": "0.3199"}, location="Tab. II, p. 10 (arXiv PDF)", variant={"head": "cls_token"},
                       note="the CLS rows print F1 between 0.3199 and 0.3615 (FAR 19.07 - 42.22 %); the lowest is listed"),
    ),
    third_party="flowtransformer",
    assumptions=("AS-530", "AS-532", "AS-533", "AS-542", "AS-543", "AS-544", "AS-545"),
)


class FlowTransformer(PublishedBaseline):
    """FlowTransformer: each flow is classified from a window of itself and its predecessors."""

    spec: ClassVar[BaselineSpec] = SPEC
    config_type: ClassVar[type[BaselineConfig]] = FlowTransformerConfig
    config: FlowTransformerConfig

    def __init__(self, config: BaselineConfig | None = None) -> None:
        super().__init__(config)
        self.prep: FlowTransformerPreprocessing | None = None
        self.net: FlowTransformerNet | None = None

    def required_columns(self) -> tuple[str, ...]:
        cfg = self.config
        return tuple(cfg.fields) + ((cfg.source_column,) if cfg.context == "source" else ())

    def _build_net(self, n_numeric: int, level_counts: list[int]) -> FlowTransformerNet:
        cfg = self.config
        return FlowTransformerNet(
            n_numeric=n_numeric, level_counts=level_counts, window=cfg.window, encoding=cfg.encoding, embed_dim=cfg.embed_dim,
            categorical_dim=cfg.categorical_embed_dim, n_layers=cfg.n_layers, ff_dim=cfg.ff_dim, n_heads=cfg.n_heads,
            head_dim=cfg.head_dim, dropout=cfg.dropout, causal=cfg.transformer == "decoder", head=cfg.head,
            featurewise_dim=cfg.featurewise_dim, mlp_sizes=cfg.mlp_sizes, mlp_dropout=cfg.mlp_dropout)

    def _stream(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Ordered encodings of a frame: (order, numeric [n, D_n], codes [n, C], windows [n, T])."""
        assert self.prep is not None
        cfg = self.config
        order = time_order(frame, cfg.time_column if cfg.time_column in frame.columns else None)
        ordered = frame.iloc[order]
        groups = ordered[cfg.source_column].astype(str).to_numpy() if cfg.context == "source" else None
        num = self.prep.transform_numeric(ordered)
        cat = self.prep.transform_categorical(ordered)
        win = causal_windows(len(ordered), cfg.window, groups)
        return order, num, cat, win

    @staticmethod
    def _gather(num: torch.Tensor, cat: torch.Tensor, win: np.ndarray, idx: np.ndarray,
                device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Batch tensors for target rows idx: numeric [B, T, D_n], codes [B, T, C], pad [B, T].
        w = torch.as_tensor(win[idx], device=device)                              # [B, T]
        pad = w < 0
        safe = w.clamp_min(0)
        x_num = num[safe] * (~pad)[..., None].to(num.dtype)
        x_cat = cat[safe] * (~pad)[..., None].to(cat.dtype)
        return x_num, x_cat, pad

    def _fit(self, data: pd.DataFrame, validation: pd.DataFrame | None, rng: np.random.Generator) -> None:
        cfg = self.config
        numeric = tuple(f for f in cfg.fields if f not in cfg.categorical)
        self.prep = FlowTransformerPreprocessing(numeric, tuple(cfg.categorical), cfg.n_categorical_levels, cfg.clip_numeric)
        order = time_order(data, cfg.time_column if cfg.time_column in data.columns else None)
        y_all = binary_labels(data[cfg.label_column])[order]
        n = len(order)
        # Validation targets: the given frame, or the latest rows of the training stream.
        n_val = 0 if validation is not None else int(round(cfg.validation_fraction * n))
        fit_rows = order[: n - n_val]
        self.prep.fit(data.iloc[fit_rows[y_all[: n - n_val] >= 0]] if (y_all[: n - n_val] >= 0).any() else data.iloc[fit_rows])
        _, num_np, cat_np, win = self._stream(data)
        device = resolve_device(cfg.device)
        num = torch.as_tensor(num_np, device=device)
        cat = torch.as_tensor(cat_np, device=device)
        y = torch.as_tensor(y_all, device=device)
        train_idx = np.nonzero(y_all[: n - n_val] >= 0)[0]
        val_idx = (n - n_val) + np.nonzero(y_all[n - n_val:] >= 0)[0]
        if np.unique(y_all[train_idx]).size < 2:
            raise InvariantViolation(f"{self.spec.name}: training flows need both classes")
        if validation is not None:
            v_order, v_num, v_cat, v_win = self._stream(validation)
            v_y = binary_labels(validation[cfg.label_column])[v_order]
            val_tensors = (torch.as_tensor(v_num, device=device), torch.as_tensor(v_cat, device=device), v_win,
                           np.nonzero(v_y >= 0)[0], torch.as_tensor(v_y, device=device))
        else:
            val_tensors = (num, cat, win, val_idx, y)
        net = self._build_net(num_np.shape[1], self.prep.level_counts).to(device)
        opt = torch.optim.Adam(net.parameters(), lr=cfg.learning_rate, eps=KERAS_ADAM_EPS)
        stopper = EarlyStopping(cfg.patience)
        steps = cfg.steps_per_epoch or math.ceil(train_idx.size / cfg.batch_size)
        history: list[dict[str, float]] = []
        for epoch in range(cfg.max_epochs):
            net.train()
            batches = (balanced_minibatches(y_all[train_idx], cfg.batch_size, steps, rng) if cfg.balanced_batches
                       else minibatches(train_idx.size, cfg.batch_size, rng))
            total, count = 0.0, 0
            for local in batches:
                idx = train_idx[local]
                x_num, x_cat, pad = self._gather(num, cat, win, idx, device)
                loss = nn.functional.binary_cross_entropy_with_logits(net(x_num, x_cat, pad), y[torch.as_tensor(idx, device=device)].float())
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                total += float(loss.detach()) * idx.size
                count += idx.size
            val_loss = self._loss(net, *val_tensors, device=device)
            history.append({"epoch": float(epoch), "train_loss": total / max(count, 1), "val_loss": val_loss})
            if math.isnan(val_loss):
                stopper.best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
                continue
            if stopper.step(val_loss, net, epoch):
                break
        stopper.restore(net)
        self.net = net
        self.fit_report.update({"train_flows": int(train_idx.size), "validation_flows": int(val_tensors[3].size),
                                "epochs_run": len(history), "best_epoch": stopper.best_epoch, "history": history})

    @torch.no_grad()
    def _loss(self, net: FlowTransformerNet, num: torch.Tensor, cat: torch.Tensor, win: np.ndarray, idx: np.ndarray,
              y: torch.Tensor, *, device: torch.device) -> float:
        # Mean binary cross-entropy over the labelled validation targets (NaN without any).
        if idx.size == 0:
            return math.nan
        net.eval()
        total = 0.0
        for start in range(0, idx.size, 4096):
            part = idx[start:start + 4096]
            x_num, x_cat, pad = self._gather(num, cat, win, part, device)
            logits = net(x_num, x_cat, pad)
            total += float(nn.functional.binary_cross_entropy_with_logits(
                logits, y[torch.as_tensor(part, device=device)].float(), reduction="sum"))
        return total / idx.size

    @torch.no_grad()
    def _scores(self, frame: pd.DataFrame) -> np.ndarray:
        assert self.net is not None
        device = resolve_device(self.config.device)
        order, num_np, cat_np, win = self._stream(frame)
        num, cat = torch.as_tensor(num_np, device=device), torch.as_tensor(cat_np, device=device)
        self.net.eval()
        out = np.empty(len(frame), dtype=np.float64)
        all_idx = np.arange(len(frame))
        for start in range(0, all_idx.size, 4096):
            part = all_idx[start:start + 4096]
            x_num, x_cat, pad = self._gather(num, cat, win, part, device)
            # Probabilities from float64 logits (outputs in fp64, D-54).
            out[order[part]] = torch.sigmoid(self.net(x_num, x_cat, pad).double()).cpu().numpy()
        return out

    def _predict(self, data: pd.DataFrame) -> PredictionParts:
        cfg = self.config
        score = self._scores(data)
        label = binary_labels(data[cfg.label_column]) if cfg.label_column in data.columns else np.full(len(data), -1, dtype=np.int64)
        if cfg.family_column in data.columns:
            family = np.asarray(["benign" if normalise_token(v) == "benign" else str(v) for v in data[cfg.family_column]], dtype=object)
        else:
            family = np.where(label == 1, "attack", np.where(label == 0, "benign", "unknown"))
        meta = build_meta(data, family=family, time_column=cfg.time_column if cfg.time_column in data.columns else None)
        det = DetectionPredictions(score=score, label=label, unit="flow", meta=meta, threshold=cfg.threshold)
        return PredictionParts(detection=det, component={"window": np.asarray([cfg.window])})

    def _export_state(self, directory: Path) -> dict[str, Any]:
        assert self.prep is not None and self.net is not None
        return {"preprocessing": self.prep.state(), "weights": save_module(self.net, directory / "flowtransformer.pt")}

    def _import_state(self, directory: Path, state: Mapping[str, Any]) -> None:
        self.prep = FlowTransformerPreprocessing.from_state(state["preprocessing"])
        net = self._build_net(len(self.prep.numeric), self.prep.level_counts)
        self.net = load_module(net, directory / str(state["weights"]), resolve_device(self.config.device))

    @classmethod
    def paper_protocol(cls) -> SplitProtocol:
        """The last 10 % of the rows in time order evaluate, the first 90 % train (AS-544)."""
        return LastRows(eval_fraction=0.1, time_column="time")
