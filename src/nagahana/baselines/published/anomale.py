"""Anomal-E (Caville, Lo, Layeghy and Portmann, Knowledge-Based Systems 258:110030, 2022, "Anomal-E: A
self-supervised network intrusion detection system based on graph neural networks",
DOI 10.1016/j.knosys.2022.110030; values from arXiv:2207.06819v5).

What the paper states (baselines-notes.md, B3)
    model         an E-GraphSAGE edge encoder trained self-supervised with Deep-Graph-Infomax-style
                  corruption (no labels); its edge embeddings feed four unsupervised detectors (PCA,
                  isolation forest, CBLOF, HBOS)
    data          NetFlow v2 datasets, randomly downsampled to 10 %, 70/30 train/test; in the "0 %
                  contamination" setting attack flows are removed from training
    metrics       accuracy, macro F1 (over benign and attack), detection rate
    results       NF-UNSW-NB15-v2: embeddings + HBOS macro F1 88.45 % (DR 80.36 %); NF-CSE-CIC-IDS2018-v2:
                  embeddings + IF macro F1 95.39 % (DR 85.38 %)

The encoder (E-GraphSAGE, Lo, Layeghy, Sarhan, Gallagher and Portmann, NOMS 2022, arXiv:2103.16329)
    graph       one node per endpoint (address and port), one edge per flow, the flow features (min-max
                scaled, fitted on training flows) as edge features e_uv, node features all ones
    layer k     a_v = mean of e_uv over the flows incident to v;  h_v^k = ReLU(W_k [h_v^{k-1} ; a_v])
    embedding   z_uv = [h_u^K ; h_v^K] for flow (u, v)
Deep Graph Infomax (Velickovic et al., ICLR 2019, arXiv:1809.10341) on edges
    summary s = sigmoid(mean_uv z_uv);  corruption: the edge features are permuted across edges;
    D(z, s) = sigmoid(z^T W s);  L = -(1 / 2E) [sum log D(z_uv, s) + sum log(1 - D(z~_uv, s))]

What the paper leaves open (AS-554): K = 2 layers of 128 units (embedding 256), Adam 1e-3, 100 epochs with
patience 10 on the DGI loss, incident edges in both directions, the detectors' PyOD defaults (anomaly.py)
and the contamination nu = 0.01 that sets the decision threshold (1 - nu on the training-score percentile);
the paper's per-dataset choice of detector is a selection on test results and is reproduced by running
each detector.
"""

from __future__ import annotations

import pickle
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Literal

import numpy as np
import pandas as pd
import torch
from torch import nn

from nagahana.baselines.published.anomaly import Detector, fit_detector
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
from nagahana.baselines.published.frames import binary_labels, build_meta, normalise_token
from nagahana.baselines.published.neural import EarlyStopping, load_module, resolve_device, save_module
from nagahana.baselines.published.preprocessing import ColumnPipeline
from nagahana.baselines.published.protocols import RandomHoldout, SplitProtocol
from nagahana.core.errors import InvariantViolation
from nagahana.evaluation.predictions import DetectionPredictions


class EGraphSAGE(nn.Module):
    """E-GraphSAGE layers over a flow graph (module docstring)."""

    def __init__(self, edge_dim: int, hidden: int, layers: int) -> None:
        super().__init__()
        dims = [edge_dim] + [hidden] * layers
        self.layers = nn.ModuleList(nn.Linear(dims[i] + edge_dim, dims[i + 1]) for i in range(layers))
        self.edge_dim = edge_dim

    def forward(self, edge_feat: torch.Tensor, src: torch.Tensor, dst: torch.Tensor, n_nodes: int) -> torch.Tensor:
        """Edge embeddings [E, 2 hidden] from edge features [E, d_e] and endpoints src, dst [E]."""
        ends = torch.cat([src, dst])
        feats = torch.cat([edge_feat, edge_feat])
        deg = torch.zeros(n_nodes, device=edge_feat.device).index_add_(0, ends, torch.ones_like(ends, dtype=edge_feat.dtype))
        agg = torch.zeros(n_nodes, self.edge_dim, device=edge_feat.device).index_add_(0, ends, feats) / deg.clamp_min(1.0)[:, None]
        h = torch.ones(n_nodes, self.edge_dim, device=edge_feat.device)
        for layer in self.layers:
            h = torch.relu(layer(torch.cat([h, agg], dim=-1)))
        return torch.cat([h[src], h[dst]], dim=-1)


class DGIDiscriminator(nn.Module):
    """Bilinear discriminator D(z, s) = sigmoid(z^T W s) (returns logits)."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.bilinear = nn.Bilinear(dim, dim, 1)

    def forward(self, z: torch.Tensor, summary: torch.Tensor) -> torch.Tensor:
        logits: torch.Tensor = self.bilinear(z, summary.expand_as(z))[:, 0]
        return logits


@dataclass
class AnomalEConfig(BaselineConfig):
    """Anomal-E settings (module docstring).

    Attributes
    ----------
    detector:
        "pca", "iforest", "cblof" or "hbos".
    representation:
        "embedding" (Anomal-E) or "raw" (the detector on the scaled flow features, the paper's comparison).
    node_key:
        "address_port" (an endpoint per address and port) or "address".
    features, label_column, family_column, src/dst address and port columns:
        Flow columns (NetFlow v2 by default).
    hidden, layers, learning_rate, epochs, patience:
        Encoder and DGI training.
    contamination:
        nu: the decision threshold is 1 - nu on the training-score percentile.
    benign_only:
        Remove attack flows from training (the 0 % contamination setting).
    """

    detector: Literal["pca", "iforest", "cblof", "hbos"] = "hbos"
    representation: Literal["embedding", "raw"] = "embedding"
    node_key: Literal["address_port", "address"] = "address_port"
    features: tuple[str, ...] = ()
    label_column: str = "Label"
    family_column: str = "Attack"
    src_column: str = "IPV4_SRC_ADDR"
    dst_column: str = "IPV4_DST_ADDR"
    sport_column: str = "L4_SRC_PORT"
    dport_column: str = "L4_DST_PORT"
    hidden: int = 128
    layers: int = 2
    learning_rate: float = 0.001
    epochs: int = 100
    patience: int = 10
    contamination: float = 0.01
    benign_only: bool = True

    def validate(self) -> None:
        super().validate()
        if min(self.hidden, self.layers, self.epochs, self.patience) < 1 or self.learning_rate <= 0:
            raise ValueError("hidden, layers, epochs, patience and learning_rate must be positive")
        if not 0.0 < self.contamination < 0.5:
            raise ValueError("contamination must lie in (0, 0.5)")


REFERENCE = Reference(
    key="caville2022anomale",
    authors="Caville, Lo, Layeghy, Portmann",
    title="Anomal-E: A self-supervised network intrusion detection system based on graph neural networks",
    venue="Knowledge-Based Systems 258:110030",
    year=2022,
    doi="10.1016/j.knosys.2022.110030",
    arxiv="2207.06819",
    note="values from arXiv:2207.06819v5",
)

_PROTO = "random 10 % sample, 70/30; no attacks in training (0 % contamination)"


def _rows() -> tuple[ReportedResult, ...]:
    out = []
    for ds, rep, det, acc, mf1, dr, loc in (
            ("nf-unsw-nb15-v2", "embedding", "hbos", "98.18%", "88.45%", "80.36%", "Tab. 3, p. 9"),
            ("nf-unsw-nb15-v2", "raw", "hbos", "94.47%", "71.34%", "58.24%", "Tab. 3, p. 9"),
            ("nf-cse-cic-ids2018-v2", "embedding", "iforest", "98.18%", "95.39%", "85.38%", "Tab. 5, p. 9"),
            ("nf-cse-cic-ids2018-v2", "raw", "iforest", "92.06%", "81.77%", "70.82%", "Tab. 5, p. 9")):
        out.append(ReportedResult(dataset=ds, protocol=_PROTO, task="binary", model=f"{rep}+{det}",
                                  values={"accuracy": acc, "macro_f1": mf1, "dr": dr}, location=f"{loc} (arXiv v5)",
                                  variant={"representation": rep, "detector": det}))
    for ds, det, mf1 in (("nf-unsw-nb15-v2", "pca", "83.59%"), ("nf-unsw-nb15-v2", "iforest", "85.62%"),
                         ("nf-unsw-nb15-v2", "cblof", "84.17%"), ("nf-cse-cic-ids2018-v2", "pca", "94.43%"),
                         ("nf-cse-cic-ids2018-v2", "cblof", "94.44%"), ("nf-cse-cic-ids2018-v2", "hbos", "94.51%")):
        out.append(ReportedResult(dataset=ds, protocol=_PROTO, task="binary", model=f"embedding+{det}", values={"macro_f1": mf1},
                                  location="Tabs. 3, 5, p. 9 (arXiv v5)", variant={"representation": "embedding", "detector": det}))
    return tuple(out)


SPEC = BaselineSpec(
    name="anomal-e",
    title="Anomal-E: self-supervised E-GraphSAGE edge embeddings with anomaly detectors",
    reference=REFERENCE,
    family="edge-anomaly",
    input_schema=InputSchema(
        description="One row per NetFlow v2 flow with endpoint addresses and ports, the flow features and the binary "
                    "`Label` (used only to drop attacks from training and to score).",
        required=("IPV4_SRC_ADDR", "IPV4_DST_ADDR"),
        label="Label",
        optional=("Attack", "L4_SRC_PORT", "L4_DST_PORT", "time", "dataset", "network", "split"),
    ),
    outputs=("detection",),
    datasets=("nf-unsw-nb15-v2", "nf-cse-cic-ids2018-v2"),
    reported=_rows(),
    third_party="anomal-e",
    assumptions=("AS-530", "AS-533", "AS-554"),
)


class AnomalE(PublishedBaseline):
    """Anomal-E (module docstring)."""

    spec: ClassVar[BaselineSpec] = SPEC
    config_type: ClassVar[type[BaselineConfig]] = AnomalEConfig
    config: AnomalEConfig

    def __init__(self, config: BaselineConfig | None = None) -> None:
        super().__init__(config)
        self.pipeline: ColumnPipeline | None = None
        self.encoder: EGraphSAGE | None = None
        self.detector: Detector | None = None

    def required_columns(self) -> tuple[str, ...]:
        cfg = self.config
        cols: tuple[str, ...] = (cfg.src_column, cfg.dst_column) + tuple(cfg.features)
        if cfg.node_key == "address_port":
            cols += (cfg.sport_column, cfg.dport_column)
        return cols

    def _features(self) -> tuple[str, ...]:
        return self.config.features or netflow.standard_features(2, drop_identifiers=True)

    def _graph(self, frame: pd.DataFrame, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, int]:
        cfg = self.config
        if cfg.node_key == "address_port":
            s = frame[cfg.src_column].astype(str) + ":" + frame[cfg.sport_column].astype(str)
            d = frame[cfg.dst_column].astype(str) + ":" + frame[cfg.dport_column].astype(str)
        else:
            s, d = frame[cfg.src_column].astype(str), frame[cfg.dst_column].astype(str)
        codes, uniques = pd.factorize(pd.concat([s, d], ignore_index=True))
        n = len(frame)
        return (torch.as_tensor(codes[:n], device=device), torch.as_tensor(codes[n:], device=device), int(len(uniques)))

    def _embed(self, frame: pd.DataFrame) -> np.ndarray:
        assert self.pipeline is not None
        x = self.pipeline.transform(frame).astype(np.float32)
        if self.config.representation == "raw":
            return x.astype(np.float64)
        assert self.encoder is not None
        device = resolve_device(self.config.device)
        src, dst, n_nodes = self._graph(frame, device)
        self.encoder.eval()
        with torch.no_grad():
            z = self.encoder(torch.as_tensor(x, device=device), src, dst, n_nodes)
        return z.double().cpu().numpy()

    def _fit(self, data: pd.DataFrame, validation: pd.DataFrame | None, rng: np.random.Generator) -> None:
        cfg = self.config
        label = binary_labels(data[cfg.label_column])
        keep = label == 0 if cfg.benign_only else label >= 0
        train = data.loc[keep].reset_index(drop=True)
        if len(train) < 2:
            raise InvariantViolation(f"{self.spec.name}: fewer than two training flows after removing attacks")
        self.pipeline = ColumnPipeline(self._features(), (), "minmax").fit(train)
        report: dict[str, Any] = {"train_flows": len(train)}
        if cfg.representation == "embedding":
            device = resolve_device(cfg.device)
            x = torch.as_tensor(self.pipeline.transform(train).astype(np.float32), device=device)
            src, dst, n_nodes = self._graph(train, device)
            enc = EGraphSAGE(x.shape[1], cfg.hidden, cfg.layers).to(device)
            disc = DGIDiscriminator(2 * cfg.hidden).to(device)
            opt = torch.optim.Adam(list(enc.parameters()) + list(disc.parameters()), lr=cfg.learning_rate)
            stopper = EarlyStopping(cfg.patience)
            curve: list[float] = []
            for epoch in range(cfg.epochs):
                enc.train()
                z = enc(x, src, dst, n_nodes)
                perm = torch.as_tensor(rng.permutation(x.shape[0]), device=device)
                z_corrupt = enc(x[perm], src, dst, n_nodes)
                summary = torch.sigmoid(z.mean(dim=0, keepdim=True))
                pos, neg = disc(z, summary), disc(z_corrupt, summary)
                loss = 0.5 * (nn.functional.binary_cross_entropy_with_logits(pos, torch.ones_like(pos))
                              + nn.functional.binary_cross_entropy_with_logits(neg, torch.zeros_like(neg)))
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                curve.append(float(loss.detach()))
                if stopper.step(curve[-1], enc, epoch):
                    break
            stopper.restore(enc)
            self.encoder = enc
            report.update({"dgi_loss": curve, "best_epoch": stopper.best_epoch})
        emb = self._embed(train)
        self.detector = fit_detector(cfg.detector, emb, rng, seed=cfg.seed)
        self.fit_report.update(report)

    def _predict(self, data: pd.DataFrame) -> PredictionParts:
        assert self.detector is not None
        cfg = self.config
        raw = self.detector.score(self._embed(data))
        score = self.detector.percentile(raw)
        label = binary_labels(data[cfg.label_column]) if cfg.label_column in data.columns else np.full(len(data), -1, dtype=np.int64)
        if cfg.family_column in data.columns:
            family = np.asarray(["benign" if normalise_token(v) == "benign" else str(v) for v in data[cfg.family_column]], dtype=object)
        else:
            family = np.where(label == 1, "attack", np.where(label == 0, "benign", "unknown"))
        det = DetectionPredictions(score=np.clip(score, 0.0, 1.0), label=label, unit="flow", meta=build_meta(data, family=family),
                                   threshold=1.0 - cfg.contamination)
        return PredictionParts(detection=det, component={"raw_anomaly_score": raw})

    def _export_state(self, directory: Path) -> dict[str, Any]:
        assert self.pipeline is not None and self.detector is not None
        state: dict[str, Any] = {"pipeline": self.pipeline.state(), "detector": self.detector.state()}
        if self.encoder is not None:
            state["encoder"] = save_module(self.encoder, directory / "egraphsage.pt")
            state["edge_dim"] = int(self.encoder.edge_dim)
        if self.detector.model is not None:
            (directory / "detector.sklearn.pkl").write_bytes(pickle.dumps(self.detector.model, protocol=pickle.HIGHEST_PROTOCOL))
            state["detector_model"] = "detector.sklearn.pkl"
        return state

    def _import_state(self, directory: Path, state: Mapping[str, Any]) -> None:
        cfg = self.config
        self.pipeline = ColumnPipeline.from_state(state["pipeline"])
        model = pickle.loads((directory / str(state["detector_model"])).read_bytes()) if "detector_model" in state else None
        self.detector = Detector.from_state(state["detector"], model)
        if "encoder" in state:
            enc = EGraphSAGE(int(state["edge_dim"]), cfg.hidden, cfg.layers)
            self.encoder = load_module(enc, directory / str(state["encoder"]), resolve_device(cfg.device))

    @classmethod
    def paper_protocol(cls) -> SplitProtocol:
        """Random 70/30 split (after the 10 % sample, protocols.stratified_sample)."""
        return RandomHoldout(test_fraction=0.3)
