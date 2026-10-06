"""Message-passing GNN of Pujol-Perich, Suarez-Varela, Cabellos-Aparicio and Barlet-Ros, ACM SIGMETRICS
Performance Evaluation Review 49(4):111-117, 2022, "Unveiling the potential of Graph Neural Networks for
robust Intrusion Detection", DOI 10.1145/3543146.3543171 (values from arXiv:2107.14756v1).

What the paper states (baselines-notes.md, B5)
    graph         a host-connection graph: hosts and flows as nodes; each flow node is linked to its
                  source and destination hosts
    model         message passing between host and flow nodes with GRU updates; per-flow classification
    task          12 classes of CIC-IDS2017 (the classes with more than 100 flows)
    protocol      random 80/20 split of graph samples, 5 cross-validation folds; 90 % of the benign-only
                  graphs dropped to rebalance the classes
    result        weighted F1 0.99 over all flows; lowest class F1 0.73 (Web brute force)

Message passing as reproduced (T iterations, hidden width d; AS-555, citation to verify):
    initial states   flow f: [x_f ; 0] padded to d (x_f its scaled CICFlowMeter features), host v: ones
    host update      m_v = sum over flows f with v in {src(f), dst(f)} of MLP_fh(h_f);  h_v = GRU_h(m_v, h_v)
    flow update      m_f = MLP_hf([h_src(f) ; h_dst(f)]);                                h_f = GRU_f(m_f, h_f)
    readout          class logits = MLP_out(h_f)
Graph samples are windows of `flows_per_graph` consecutive flows in time order (AS-555). T = 8, d = 128,
Adam 1e-3, cross-entropy, 50 epochs of shuffled graph batches.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

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
from nagahana.baselines.published.features import cicflowmeter
from nagahana.baselines.published.frames import BENIGN_TOKENS, build_meta, normalise_token, time_order
from nagahana.baselines.published.neural import load_module, resolve_device, save_module
from nagahana.baselines.published.preprocessing import ColumnPipeline
from nagahana.baselines.published.protocols import RandomHoldout, SplitProtocol
from nagahana.baselines.published.stages import attack_stages, class_stage_matrix, label_stage_codes
from nagahana.core.errors import InvariantViolation
from nagahana.evaluation.predictions import DetectionPredictions, StagePredictions


class HostFlowGNN(nn.Module):
    """The message passing of the module docstring."""

    def __init__(self, d_feat: int, hidden: int, iterations: int, n_classes: int) -> None:
        super().__init__()
        if d_feat > hidden:
            raise ValueError(f"the hidden width {hidden} must hold the {d_feat} flow features")
        self.hidden, self.iterations, self.d_feat = hidden, iterations, d_feat
        self.msg_fh = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.msg_hf = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.gru_h = nn.GRUCell(hidden, hidden)
        self.gru_f = nn.GRUCell(hidden, hidden)
        self.out = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden // 2), nn.ReLU(),
                                 nn.Linear(hidden // 2, n_classes))

    def forward(self, flow_x: torch.Tensor, src: torch.Tensor, dst: torch.Tensor, n_hosts: int) -> torch.Tensor:
        """flow_x [F, d_feat], src / dst [F] host indices -> class logits [F, C]."""
        f = flow_x.shape[0]
        h_f = torch.cat([flow_x, torch.zeros(f, self.hidden - self.d_feat, device=flow_x.device)], dim=1)
        h_v = torch.ones(n_hosts, self.hidden, device=flow_x.device)
        ends = torch.cat([src, dst])
        for _ in range(self.iterations):
            msg = self.msg_fh(h_f)
            m_v = torch.zeros(n_hosts, self.hidden, device=flow_x.device).index_add_(0, ends, torch.cat([msg, msg]))
            h_v = self.gru_h(m_v, h_v)
            h_f = self.gru_f(self.msg_hf(torch.cat([h_v[src], h_v[dst]], dim=1)), h_f)
        logits: torch.Tensor = self.out(h_f)
        return logits


@dataclass
class PujolPerichConfig(BaselineConfig):
    """Settings of the reproduction (module docstring).

    Attributes
    ----------
    classes:
        Class names in code order (benign first); empty learns the training classes with more than
        `min_class_flows` flows.
    min_class_flows:
        Classes with fewer training flows are excluded (the paper keeps classes with more than 100 flows).
    flows_per_graph, iterations, hidden, learning_rate, epochs, batch_graphs:
        Graph samples and training.
    drop_benign_graphs:
        Share of benign-only training graphs dropped.
    features, label_column, src_column, dst_column, time_column, stage_dataset:
        Columns (harmonised CICFlowMeter names) and the label table of the ATT&CK projection ("" to skip).
    threshold:
        Operating threshold on P(not benign).
    """

    classes: tuple[str, ...] = ()
    min_class_flows: int = 100
    flows_per_graph: int = 200
    iterations: int = 8
    hidden: int = 128
    learning_rate: float = 0.001
    epochs: int = 50
    batch_graphs: int = 8
    drop_benign_graphs: float = 0.9
    features: tuple[str, ...] = ()
    label_column: str = "Label"
    src_column: str = "Src IP"
    dst_column: str = "Dst IP"
    time_column: str = "Timestamp"
    stage_dataset: str = "cic-ids2017"
    threshold: float = 0.5

    def validate(self) -> None:
        super().validate()
        if min(self.flows_per_graph, self.iterations, self.hidden, self.epochs, self.batch_graphs) < 1 or self.learning_rate <= 0:
            raise ValueError("graph size, iterations, hidden, epochs, batch_graphs and learning_rate must be positive")
        if not 0.0 <= self.drop_benign_graphs < 1.0 or self.min_class_flows < 0:
            raise ValueError("drop_benign_graphs must lie in [0, 1) and min_class_flows >= 0")


REFERENCE = Reference(
    key="bl-pujolperich2022gnn",
    authors="Pujol-Perich, Suarez-Varela, Cabellos-Aparicio, Barlet-Ros",
    title="Unveiling the potential of Graph Neural Networks for robust Intrusion Detection",
    venue="ACM SIGMETRICS Performance Evaluation Review 49(4):111-117",
    year=2022,
    doi="10.1145/3543146.3543171",
    arxiv="2107.14756",
    note="values from arXiv:2107.14756v1",
)

SPEC = BaselineSpec(
    name="pujol-perich-gnn",
    title="Host-connection graph neural network for flow classification",
    reference=REFERENCE,
    family="flow-classifier",
    input_schema=InputSchema(
        description="CIC-IDS2017 TrafficLabelling flow records (any spelling, harmonised) with source and destination "
                    "addresses, a timestamp and the text `Label`.",
        required=("Src IP", "Dst IP"),
        label="Label",
        optional=("Timestamp", "dataset", "network", "split"),
    ),
    outputs=("detection", "stage"),
    datasets=("cic-ids2017",),
    reported=(
        ReportedResult(dataset="cic-ids2017", protocol="random 80/20 split of graph samples, 5 folds; 90 % of benign-only graphs dropped",
                       task="multiclass-12", model="gnn", values={"weighted_f1": "0.99"}, location="Sec. 5.2 and Tab. 1, p. 4 (arXiv v1)",
                       note="lowest class F1 0.73 (Web brute force); per-class F1 in Tab. 1"),
    ),
    third_party="pujol-perich-gnn",
    assumptions=("AS-530", "AS-532", "AS-533", "AS-555"),
)


class PujolPerichGNN(PublishedBaseline):
    """Pujol-Perich et al. 2022 (module docstring)."""

    spec: ClassVar[BaselineSpec] = SPEC
    config_type: ClassVar[type[BaselineConfig]] = PujolPerichConfig
    config: PujolPerichConfig

    def __init__(self, config: BaselineConfig | None = None) -> None:
        super().__init__(config)
        self.pipeline: ColumnPipeline | None = None
        self.net: HostFlowGNN | None = None
        self.classes_: tuple[str, ...] = ()
        self.class_stage_: np.ndarray | None = None

    def prepare(self, frame: pd.DataFrame) -> pd.DataFrame:
        return cicflowmeter.clean(cicflowmeter.harmonise(frame))

    def _features(self, columns: Sequence[str]) -> tuple[str, ...]:
        return self.config.features or cicflowmeter.feature_columns(list(columns), drop=cicflowmeter.CIC_IDENTIFIERS)

    def _codes(self, frame: pd.DataFrame) -> np.ndarray:
        index = {normalise_token(c): i for i, c in enumerate(self.classes_)}
        col = self.config.label_column
        if col not in frame.columns:
            return np.full(len(frame), -1, dtype=np.int64)
        return np.asarray([index.get(normalise_token(v), -1) for v in frame[col].tolist()], dtype=np.int64)

    def _graphs(self, frame: pd.DataFrame) -> list[np.ndarray]:
        """Row positions of each graph sample: consecutive flows in time order."""
        cfg = self.config
        order = time_order(frame, cfg.time_column if cfg.time_column in frame.columns else None)
        return [order[i:i + cfg.flows_per_graph] for i in range(0, len(order), cfg.flows_per_graph)]

    def _tensors(self, frame: pd.DataFrame, rows: np.ndarray, x: np.ndarray, device: torch.device
                 ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        cfg = self.config
        s = frame[cfg.src_column].astype(str).to_numpy()[rows]
        d = frame[cfg.dst_column].astype(str).to_numpy()[rows]
        codes, uniques = pd.factorize(pd.Series(np.concatenate([s, d])))
        n = rows.size
        return (torch.as_tensor(x[rows], dtype=torch.float32, device=device), torch.as_tensor(codes[:n], device=device),
                torch.as_tensor(codes[n:], device=device), int(len(uniques)))

    def _fit(self, data: pd.DataFrame, validation: pd.DataFrame | None, rng: np.random.Generator) -> None:
        cfg = self.config
        device = resolve_device(cfg.device)
        if cfg.classes:
            self.classes_ = tuple(cfg.classes)
        else:
            counts = data[cfg.label_column].astype(str).str.strip().value_counts()
            kept = sorted(c for c, k in counts.items() if k > cfg.min_class_flows or normalise_token(c) in BENIGN_TOKENS)
            benign = [c for c in kept if normalise_token(c) in BENIGN_TOKENS]
            self.classes_ = tuple(benign + [c for c in kept if c not in benign])
        if len(self.classes_) < 2:
            raise InvariantViolation(f"{self.spec.name}: fewer than two classes to train on")
        codes = self._codes(data)
        feats = self._features(list(data.columns))
        pipeline = ColumnPipeline(feats, (), "minmax")
        pipeline.fit(data.loc[codes >= 0])
        x = pipeline.transform(data)
        graphs = self._graphs(data)
        benign = next(i for i, c in enumerate(self.classes_) if normalise_token(c) in BENIGN_TOKENS)
        drop_rng = np.random.default_rng(derive_seed(cfg.seed, "drop-benign-graphs"))
        kept_graphs = [g for g in graphs if not (np.all(codes[g][codes[g] >= 0] == benign) and drop_rng.random() < cfg.drop_benign_graphs)]
        net = HostFlowGNN(len(feats), cfg.hidden, cfg.iterations, len(self.classes_)).to(device)
        opt = torch.optim.Adam(net.parameters(), lr=cfg.learning_rate)
        curve: list[float] = []
        for _epoch in range(cfg.epochs):
            net.train()
            order = rng.permutation(len(kept_graphs))
            total = 0.0
            for start in range(0, order.size, cfg.batch_graphs):
                losses = []
                n_labelled = 0
                for gi in order[start:start + cfg.batch_graphs]:
                    rows = kept_graphs[gi]
                    y = torch.as_tensor(codes[rows], device=device)
                    known = y >= 0
                    if not bool(known.any()):
                        continue
                    flow_x, src, dst, n_hosts = self._tensors(data, rows, x, device)
                    logits = net(flow_x, src, dst, n_hosts)
                    losses.append(nn.functional.cross_entropy(logits[known], y[known], reduction="sum"))
                    n_labelled += int(known.sum())
                if losses:
                    # Mean cross-entropy over the labelled flows of the batch of graphs.
                    loss = torch.stack(losses).sum() / n_labelled
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()
                    total += float(loss.detach())
            curve.append(total / max(1, math.ceil(order.size / cfg.batch_graphs)))
        self.pipeline, self.net = pipeline, net
        if cfg.stage_dataset:
            stage = label_stage_codes(data[cfg.label_column].tolist(), cfg.stage_dataset)
            self.class_stage_ = class_stage_matrix(codes, stage, len(self.classes_))
        self.fit_report.update({"classes": list(self.classes_), "graphs": len(graphs), "graphs_used": len(kept_graphs),
                                "loss_curve": curve})

    @torch.no_grad()
    def _predict(self, data: pd.DataFrame) -> PredictionParts:
        assert self.pipeline is not None and self.net is not None
        cfg = self.config
        device = resolve_device(cfg.device)
        self.net.eval()
        x = self.pipeline.transform(data)
        probs = np.zeros((len(data), len(self.classes_)))
        for rows in self._graphs(data):
            flow_x, src, dst, n_hosts = self._tensors(data, rows, x, device)
            probs[rows] = torch.softmax(self.net(flow_x, src, dst, n_hosts).double(), dim=1).cpu().numpy()
        codes = self._codes(data)
        benign = next(i for i, c in enumerate(self.classes_) if normalise_token(c) in BENIGN_TOKENS)
        label = np.where(codes < 0, -1, (codes != benign).astype(np.int64))
        family = np.where(label == 1, "attack", np.where(label == 0, "benign", "unknown"))
        if cfg.label_column in data.columns:
            family = np.asarray(["benign" if normalise_token(v) in BENIGN_TOKENS else str(v).strip() for v in data[cfg.label_column]], dtype=object)
        meta = build_meta(data, family=family)
        stage = None
        if self.class_stage_ is not None:
            stage_label = label_stage_codes(data[cfg.label_column].tolist(), cfg.stage_dataset) if cfg.label_column in data.columns else np.full(len(data), -1)
            stage = StagePredictions(probs=probs @ self.class_stage_, label=stage_label, stage_names=attack_stages(), meta=meta)
        det = DetectionPredictions(score=np.clip(1.0 - probs[:, benign], 0, 1), label=label, unit="flow", meta=meta, threshold=cfg.threshold)
        return PredictionParts(detection=det, stage=stage, component={"class_probs": probs, "class_label": codes})

    def _export_state(self, directory: Path) -> dict[str, Any]:
        assert self.pipeline is not None and self.net is not None
        return {"pipeline": self.pipeline.state(), "weights": save_module(self.net, directory / "gnn.pt"), "classes": list(self.classes_),
                "class_stage": None if self.class_stage_ is None else self.class_stage_.tolist()}

    def _import_state(self, directory: Path, state: Mapping[str, Any]) -> None:
        cfg = self.config
        self.pipeline = ColumnPipeline.from_state(state["pipeline"])
        self.classes_ = tuple(state["classes"])
        net = HostFlowGNN(len(self.pipeline.numeric), cfg.hidden, cfg.iterations, len(self.classes_))
        self.net = load_module(net, directory / str(state["weights"]), resolve_device(cfg.device))
        cs = state.get("class_stage")
        self.class_stage_ = None if cs is None else np.asarray(cs, dtype=np.float64)

    @classmethod
    def paper_protocol(cls) -> SplitProtocol:
        """Random 80/20 split (the paper splits graph samples; rows are split here, AS-555)."""
        return RandomHoldout(test_fraction=0.2)

