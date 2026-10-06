"""EULER reproduction (King and Huang, "EULER: Detecting Network Lateral Movement via Scalable Temporal
Link Prediction", NDSS 2022, DOI 10.14722/ndss.2022.24107).

What the paper states (baselines-notes.md, C5)
    data          LANL 2015 authentication events (17,685 nodes, 518 anomalous red-team edges, 58 days)
    snapshots     30 minutes (1800 s)
    task          link prediction (n = 1): the edges of snapshot t + 1 are scored from the embeddings of
                  snapshot t; an unlikely edge is flagged; link detection (n = 0) scores snapshot t
    models        GCN + GRU (best), SAGE + LSTM; inner-product decoder
    protocol      train on the snapshots before the first anomalous edge, test on the rest; the cutoff is
                  chosen on held-out normal snapshots (Eq. 6); results are averages of 5 runs
    results       prediction, GCN+GRU: AUC 0.9906, AP 0.0155, TPR 85.49 %, FPR 0.6088 %, P 0.0050

What the paper leaves open (AS-552, AS-553; citation to verify)
    layer sizes (two GCN layers, 32 hidden, 32 out; GRU 32; embedding 16), dropout 0.25, Adam 0.005, at
    most 100 epochs with patience 5 on the validation loss, one negative pair drawn uniformly per
    positive edge, unweighted edges, self-loops dropped; the held-out snapshots are the last 5 % of the
    training snapshots; Eq. 6 is read as the weighted equal-error cutoff
        tau = argmin_tau | (1 - w) (1 - TPR(tau)) - w FPR(tau) |,  w = 0.5,
    over validation edges (positives) and as many uniform random pairs (negatives); an edge with
    P(edge) < tau is anomalous.

Scoring: the anomaly score of an edge is 1 - P(edge) and the operating threshold is 1 - tau, so the
detection record's decision rule (score >= threshold) is EULER's (P(edge) <= tau). An edge in a
snapshot is a pair inside a time window: the record's unit is "window", with the endpoints and the
snapshot in the metadata; `unit = "state_update"` scores every event with its edge's score instead.

Continuity: fit keeps the recurrent state and the embedding at the end of training; predict continues
from it through every later snapshot (empty ones included), so a test frame must start after the
training period.
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
from nagahana.baselines.published.euler.graph import NodeIndex, Snapshot, build_snapshots, edge_weights, gcn_operator, snapshot_ids
from nagahana.baselines.published.euler.model import EulerNet, edge_logits
from nagahana.baselines.published.frames import epoch_seconds
from nagahana.baselines.published.neural import EarlyStopping, load_module, resolve_device, save_module
from nagahana.baselines.published.protocols import SplitProtocol, TimeCutoff
from nagahana.core.errors import InvariantViolation
from nagahana.evaluation.predictions import DetectionPredictions, make_meta


@dataclass
class EulerConfig(BaselineConfig):
    """EULER settings (see the module docstring for the source of each default).

    Attributes
    ----------
    snapshot_seconds:
        Snapshot length.
    mode:
        "prediction" (score snapshot t + 1 from t) or "detection" (score snapshot t from t).
    encoder, rnn, gnn_layers, hidden, gnn_out, rnn_hidden, embed, dropout:
        Architecture (model.py).
    learning_rate, epochs, patience, negative_ratio:
        Training: Adam, epochs, early-stopping patience, negatives per positive edge.
    edge_weighting, drop_self_loops:
        Snapshot graph construction (graph.py).
    validation_fraction:
        Share of the training snapshots (the latest) held out for early stopping and the cutoff.
    cutoff_fpr_weight:
        w of Eq. 6.
    bptt:
        Snapshots per truncated back-propagation chunk (0: the whole training sequence per step).
    src_column, dst_column, time_column, label_column:
        Event columns; the label (1 anomalous, 0 normal) is optional.
    unit:
        "edge" or "state_update" (module docstring).
    """

    snapshot_seconds: float = 1800.0
    mode: Literal["prediction", "detection"] = "prediction"
    encoder: Literal["gcn", "sage"] = "gcn"
    rnn: Literal["gru", "lstm"] = "gru"
    gnn_layers: int = 2
    hidden: int = 32
    gnn_out: int = 32
    rnn_hidden: int = 32
    embed: int = 16
    dropout: float = 0.25
    learning_rate: float = 0.005
    epochs: int = 100
    patience: int = 5
    negative_ratio: float = 1.0
    edge_weighting: Literal["binary", "count", "log_count"] = "binary"
    drop_self_loops: bool = True
    validation_fraction: float = 0.05
    cutoff_fpr_weight: float = 0.5
    bptt: int = 0
    src_column: str = "src"
    dst_column: str = "dst"
    time_column: str = "time"
    label_column: str = "label"
    unit: Literal["edge", "state_update"] = "edge"

    def validate(self) -> None:
        super().validate()
        if self.snapshot_seconds <= 0 or self.learning_rate <= 0 or self.negative_ratio <= 0:
            raise ValueError("snapshot_seconds, learning_rate and negative_ratio must be positive")
        if min(self.gnn_layers, self.hidden, self.gnn_out, self.rnn_hidden, self.embed, self.epochs, self.patience) < 1:
            raise ValueError("layer counts, widths, epochs and patience must be >= 1")
        if not 0.0 <= self.dropout < 1.0 or not 0.0 < self.validation_fraction < 1.0:
            raise ValueError("dropout must lie in [0, 1) and validation_fraction in (0, 1)")
        if not 0.0 <= self.cutoff_fpr_weight <= 1.0 or self.bptt < 0:
            raise ValueError("cutoff_fpr_weight must lie in [0, 1] and bptt must be >= 0")


def weighted_equal_error_cutoff(pos: np.ndarray, neg: np.ndarray, fpr_weight: float) -> float:
    """tau minimising |(1 - w)(1 - TPR) - w FPR| over the candidate cutoffs (the observed scores).

    TPR(tau) = share of positive scores >= tau, FPR(tau) = share of negative scores >= tau.
    """
    if pos.size == 0 or neg.size == 0:
        raise InvariantViolation("the cutoff needs positive and negative validation scores")
    cand = np.unique(np.concatenate([pos, neg]))
    pos_s, neg_s = np.sort(pos), np.sort(neg)
    tpr = 1.0 - np.searchsorted(pos_s, cand, side="left") / pos_s.size
    fpr = 1.0 - np.searchsorted(neg_s, cand, side="left") / neg_s.size
    gap = np.abs((1.0 - fpr_weight) * (1.0 - tpr) - fpr_weight * fpr)
    return float(cand[int(np.argmin(gap))])


REFERENCE = Reference(
    key="bl-king2022euler",
    authors="King, Huang",
    title="EULER: Detecting Network Lateral Movement via Scalable Temporal Link Prediction",
    venue="Proceedings of the Network and Distributed System Security Symposium (NDSS 2022)",
    year=2022,
    doi="10.14722/ndss.2022.24107",
)

_PROTO = "LANL 2015, 30-min snapshots; train before the first anomalous edge; cutoff on held-out normal snapshots"

SPEC = BaselineSpec(
    name="euler",
    title="EULER temporal link prediction (GCN or SAGE encoder, GRU or LSTM)",
    reference=REFERENCE,
    family="link-predictor",
    input_schema=InputSchema(
        description="One row per event between two entities (LANL authentication events, flows): event `time` in "
                    "seconds, `src` and `dst` entity names, and optionally the `label` (1 for red-team events).",
        required=("time", "src", "dst"),
        time="time",
        optional=("label", "dataset", "network", "split"),
    ),
    outputs=("detection",),
    datasets=("lanl-2015",),
    reported=(
        ReportedResult(dataset="lanl-2015", protocol=_PROTO, task="link-prediction", model="gcn+gru",
                       values={"auc": "0.9906", "ap": "0.0155", "tpr": "85.49", "fpr": "0.6088", "precision": "0.0050"},
                       location="Tab. VI, p. 11", variant={"encoder": "gcn", "rnn": "gru", "mode": "prediction"},
                       percent=("tpr", "fpr"), note="precision printed without unit; read as a fraction"),
        ReportedResult(dataset="lanl-2015", protocol=_PROTO, task="link-prediction", model="sage+lstm",
                       values={"auc": "0.9865", "ap": "0.0228", "tpr": "85.29", "fpr": "0.8037", "precision": "0.0038"},
                       location="Tab. VI, p. 11", variant={"encoder": "sage", "rnn": "lstm", "mode": "prediction"},
                       percent=("tpr", "fpr")),
        ReportedResult(dataset="lanl-2015", protocol=_PROTO, task="link-detection", model="gcn+gru",
                       values={"auc": "0.9912", "ap": "0.0523", "tpr": "86.10", "fpr": "0.5698", "precision": "0.0054"},
                       location="Tab. VI, p. 11", variant={"encoder": "gcn", "rnn": "gru", "mode": "detection"},
                       percent=("tpr", "fpr")),
    ),
    third_party="euler",
    assumptions=("AS-530", "AS-533", "AS-552", "AS-553"),
)


class Euler(PublishedBaseline):
    """EULER: per-snapshot GNN + RNN embeddings, inner-product link scores, Eq. 6 cutoff."""

    spec: ClassVar[BaselineSpec] = SPEC
    config_type: ClassVar[type[BaselineConfig]] = EulerConfig
    config: EulerConfig

    def __init__(self, config: BaselineConfig | None = None) -> None:
        super().__init__(config)
        self.nodes: NodeIndex | None = None
        self.net: EulerNet | None = None
        self.origin = 0.0
        self.last_snapshot = -1
        self.cutoff = 0.5
        self.state_: tuple[torch.Tensor, torch.Tensor] | None = None
        self.z_last: torch.Tensor | None = None

    def required_columns(self) -> tuple[str, ...]:
        cfg = self.config
        return (cfg.time_column, cfg.src_column, cfg.dst_column)

    def _graph(self, snap: Snapshot, device: torch.device) -> tuple[torch.Tensor, ...]:
        assert self.nodes is not None
        w = edge_weights(snap.count, self.config.edge_weighting)
        rows, cols, values, self_w = gcn_operator(snap.src, snap.dst, w, self.nodes.size, device)
        return rows, cols, values, self_w, torch.as_tensor(snap.src, device=device), torch.as_tensor(snap.dst, device=device)

    def _negatives(self, count: int, rng: np.random.Generator, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        # Uniform random ordered pairs over the training nodes (indices 1 ... V).
        assert self.nodes is not None
        n_nodes = self.nodes.size - 1
        u = rng.integers(1, n_nodes + 1, size=count)
        v = rng.integers(1, n_nodes + 1, size=count)
        return torch.as_tensor(u, device=device), torch.as_tensor(v, device=device)

    def _pair_loss(self, z: torch.Tensor, snap: Snapshot, rng: np.random.Generator, device: torch.device) -> torch.Tensor | None:
        if snap.n_edges == 0:
            return None
        src, dst = torch.as_tensor(snap.src, device=device), torch.as_tensor(snap.dst, device=device)
        k = max(1, int(math.ceil(self.config.negative_ratio * snap.n_edges)))
        nu, nv = self._negatives(k, rng, device)
        pos = edge_logits(z, src, dst)
        neg = edge_logits(z, nu, nv)
        return (nn.functional.binary_cross_entropy_with_logits(pos, torch.ones_like(pos))
                + nn.functional.binary_cross_entropy_with_logits(neg, torch.zeros_like(neg)))

    def _fit(self, data: pd.DataFrame, validation: pd.DataFrame | None, rng: np.random.Generator) -> None:
        cfg = self.config
        device = resolve_device(cfg.device)
        t = epoch_seconds(data[cfg.time_column])
        self.origin = float(t.min())
        self.nodes = NodeIndex.build(data[cfg.src_column], data[cfg.dst_column])
        snap_id = snapshot_ids(t, self.origin, cfg.snapshot_seconds)
        src, dst = self.nodes.encode(data[cfg.src_column]), self.nodes.encode(data[cfg.dst_column])
        last = int(snap_id.max())
        snaps = build_snapshots(snap_id, src, dst, first=0, last=last, drop_self_loops=cfg.drop_self_loops)
        n_snap = len(snaps)
        if n_snap < 3:
            raise InvariantViolation(f"{self.spec.name}: training needs at least 3 snapshots, got {n_snap}")
        n_val = max(1, int(round(cfg.validation_fraction * n_snap)))
        n_train = n_snap - n_val
        graphs = [self._graph(s, device) for s in snaps]
        offset = 1 if cfg.mode == "prediction" else 0
        net = EulerNet(encoder=cfg.encoder, rnn=cfg.rnn, n_nodes=self.nodes.size, hidden=cfg.hidden, gnn_out=cfg.gnn_out,
                       rnn_hidden=cfg.rnn_hidden, embed=cfg.embed, layers=cfg.gnn_layers, dropout=cfg.dropout).to(device)
        opt = torch.optim.Adam(net.parameters(), lr=cfg.learning_rate)
        stopper = EarlyStopping(cfg.patience)
        chunk = cfg.bptt or n_train
        # Fixed negatives for the validation loss, so its changes reflect the model, not the draw.
        val_rng_seed = int(rng.integers(0, 2**31 - 1))
        history: list[dict[str, float]] = []
        for epoch in range(cfg.epochs):
            net.train()
            state = net.initial_state(self.nodes.size, device)
            train_losses: list[float] = []
            for start in range(0, n_train, chunk):
                terms: list[torch.Tensor] = []
                for i in range(start, min(start + chunk, n_train)):
                    z, state = net.step(graphs[i], state)
                    target = i + offset
                    if target < n_train:
                        loss = self._pair_loss(z, snaps[target], rng, device)
                        if loss is not None:
                            terms.append(loss)
                if terms:
                    total = torch.stack(terms).mean()
                    opt.zero_grad(set_to_none=True)
                    total.backward()
                    opt.step()
                    train_losses.append(float(total.detach()))
                state = (state[0].detach(), state[1].detach())
            val_loss = self._validation_loss(net, graphs, snaps, n_train, offset, np.random.default_rng(val_rng_seed), device)
            history.append({"epoch": float(epoch), "train_loss": float(np.mean(train_losses)) if train_losses else math.nan,
                            "val_loss": val_loss})
            if stopper.step(val_loss, net, epoch):
                break
        stopper.restore(net)
        net.eval()
        # Cutoff (Eq. 6) on the validation snapshots, then the state at the end of the training period.
        pos_scores, neg_scores, state, z = self._validation_scores(net, graphs, snaps, n_train, offset,
                                                                   np.random.default_rng(val_rng_seed), device)
        self.cutoff = weighted_equal_error_cutoff(pos_scores, neg_scores, cfg.cutoff_fpr_weight)
        self.net, self.state_, self.z_last, self.last_snapshot = net, state, z, last
        self.fit_report.update({"nodes": self.nodes.size - 1, "snapshots": n_snap, "train_snapshots": n_train,
                                "validation_snapshots": n_val, "epochs_run": len(history), "best_epoch": stopper.best_epoch,
                                "cutoff": self.cutoff, "history": history})

    @torch.no_grad()
    def _validation_loss(self, net: EulerNet, graphs: list[tuple[torch.Tensor, ...]], snaps: list[Snapshot], n_train: int,
                         offset: int, rng: np.random.Generator, device: torch.device) -> float:
        net.eval()
        state = net.initial_state(self.nodes.size if self.nodes else 1, device)
        losses: list[float] = []
        for i in range(len(snaps)):
            z, state = net.step(graphs[i], state)
            target = i + offset
            if n_train <= target < len(snaps):
                loss = self._pair_loss(z, snaps[target], rng, device)
                if loss is not None:
                    losses.append(float(loss))
        net.train()
        return float(np.mean(losses)) if losses else math.inf

    @torch.no_grad()
    def _validation_scores(self, net: EulerNet, graphs: list[tuple[torch.Tensor, ...]], snaps: list[Snapshot], n_train: int,
                           offset: int, rng: np.random.Generator, device: torch.device
                           ) -> tuple[np.ndarray, np.ndarray, tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        assert self.nodes is not None
        state = net.initial_state(self.nodes.size, device)
        pos: list[np.ndarray] = []
        neg: list[np.ndarray] = []
        z = torch.zeros(self.nodes.size, self.config.embed, device=device)
        for i in range(len(snaps)):
            z, state = net.step(graphs[i], state)
            target = i + offset
            if n_train <= target < len(snaps) and snaps[target].n_edges:
                s = snaps[target]
                src, dst = torch.as_tensor(s.src, device=device), torch.as_tensor(s.dst, device=device)
                pos.append(torch.sigmoid(edge_logits(z, src, dst).double()).cpu().numpy())
                nu, nv = self._negatives(s.n_edges, rng, device)
                neg.append(torch.sigmoid(edge_logits(z, nu, nv).double()).cpu().numpy())
        if not pos:
            raise InvariantViolation(f"{self.spec.name}: the validation snapshots hold no edge to set the cutoff on")
        return np.concatenate(pos), np.concatenate(neg), state, z

    @torch.no_grad()
    def _edge_probabilities(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
        """Distinct (snapshot, src, dst) edges of the frame with P(edge), continuing from the training state."""
        assert self.net is not None and self.nodes is not None and self.state_ is not None and self.z_last is not None
        cfg = self.config
        device = resolve_device(cfg.device)
        t = epoch_seconds(frame[cfg.time_column])
        snap_id = snapshot_ids(t, self.origin, cfg.snapshot_seconds)
        first = int(snap_id.min())
        if first <= self.last_snapshot:
            raise InvariantViolation(f"{self.spec.name}: events of snapshot {first} are not after the training period "
                                     f"(last training snapshot {self.last_snapshot}); EULER scores later snapshots only")
        src, dst = self.nodes.encode(frame[cfg.src_column]), self.nodes.encode(frame[cfg.dst_column])
        last = int(snap_id.max())
        snaps = build_snapshots(snap_id, src, dst, first=self.last_snapshot + 1, last=last, drop_self_loops=cfg.drop_self_loops)
        self.net.eval()
        state, z_prev = self.state_, self.z_last
        rows: list[pd.DataFrame] = []
        probs: list[np.ndarray] = []
        for s in snaps:
            if cfg.mode == "prediction":
                z_score = z_prev                                      # embeddings of the previous snapshot
                z_prev, state = self.net.step(self._graph(s, device), state)
            else:
                z_prev, state = self.net.step(self._graph(s, device), state)
                z_score = z_prev
            if s.n_edges:
                e_src, e_dst = torch.as_tensor(s.src, device=device), torch.as_tensor(s.dst, device=device)
                probs.append(torch.sigmoid(edge_logits(z_score, e_src, e_dst).double()).cpu().numpy())
                rows.append(pd.DataFrame({"snapshot": s.index, "src_i": s.src, "dst_i": s.dst}))
        edges = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame({"snapshot": [], "src_i": [], "dst_i": []})
        return edges, (np.concatenate(probs) if probs else np.zeros(0))

    def _predict(self, data: pd.DataFrame) -> PredictionParts:
        cfg = self.config
        edges, p_edge = self._edge_probabilities(data)
        t = epoch_seconds(data[cfg.time_column])
        snap_id = snapshot_ids(t, self.origin, cfg.snapshot_seconds)
        assert self.nodes is not None
        ev = pd.DataFrame({"snapshot": snap_id, "src_i": self.nodes.encode(data[cfg.src_column]),
                           "dst_i": self.nodes.encode(data[cfg.dst_column]),
                           "src": data[cfg.src_column].astype(str).to_numpy(), "dst": data[cfg.dst_column].astype(str).to_numpy(),
                           "label": (pd.to_numeric(data[cfg.label_column], errors="coerce").fillna(-1).astype(np.int64)
                                     if cfg.label_column in data.columns else -1)})
        if cfg.drop_self_loops:
            ev = ev.loc[ev["src_i"] != ev["dst_i"]] if cfg.unit == "edge" else ev
        edges = edges.assign(p=p_edge)
        anomaly_threshold = float(np.clip(1.0 - self.cutoff, 0.0, 1.0))
        component = {"cutoff": np.asarray([self.cutoff])}
        if cfg.unit == "edge":
            # One unit per distinct (snapshot, src, dst): its label is the maximum event label (1 if any
            # red-team event), its names are those of its events.
            agg = ev.groupby(["snapshot", "src_i", "dst_i"], sort=False).agg(src=("src", "first"), dst=("dst", "first"),
                                                                               label=("label", "max")).reset_index()
            merged = agg.merge(edges, on=["snapshot", "src_i", "dst_i"], how="inner")
            score = 1.0 - merged["p"].to_numpy(dtype=np.float64)
            label = merged["label"].to_numpy(dtype=np.int64)
            meta = make_meta(len(merged), time=self.origin + merged["snapshot"].to_numpy(dtype=np.float64) * cfg.snapshot_seconds,
                             entity=merged["src_i"].to_numpy(dtype=np.int64),
                             family=np.where(label == 1, "lateral_movement", np.where(label == 0, "benign", "unknown")),
                             src=merged["src"].to_numpy(), dst=merged["dst"].to_numpy(),
                             snapshot=merged["snapshot"].to_numpy(dtype=np.int64))
            component["edge_probability"] = merged["p"].to_numpy(dtype=np.float64)
            det = DetectionPredictions(score=score, label=label, unit="window", meta=meta, threshold=anomaly_threshold)
        else:
            merged = ev.reset_index(drop=True).merge(edges, on=["snapshot", "src_i", "dst_i"], how="left")
            # Self-loop events (dropped from the graphs) and nothing else lack an edge probability.
            p = merged["p"].to_numpy(dtype=np.float64)
            p = np.where(np.isnan(p), 1.0, p)
            label = merged["label"].to_numpy(dtype=np.int64)
            meta = make_meta(len(merged), time=t, family=np.where(label == 1, "lateral_movement", np.where(label == 0, "benign", "unknown")),
                             src=merged["src"].to_numpy(), dst=merged["dst"].to_numpy(),
                             snapshot=merged["snapshot"].to_numpy(dtype=np.int64))
            component["edge_probability"] = p
            det = DetectionPredictions(score=1.0 - p, label=label, unit="state_update", meta=meta, threshold=anomaly_threshold)
        return PredictionParts(detection=det, component=component)

    def _export_state(self, directory: Path) -> dict[str, Any]:
        assert self.net is not None and self.nodes is not None and self.state_ is not None and self.z_last is not None
        torch.save({"h": self.state_[0].cpu(), "c": self.state_[1].cpu(), "z": self.z_last.cpu()}, directory / "state.pt")
        return {"weights": save_module(self.net, directory / "euler.pt"), "state_file": "state.pt", "nodes": self.nodes.state(),
                "origin": self.origin, "last_snapshot": self.last_snapshot, "cutoff": self.cutoff}

    def _import_state(self, directory: Path, state: Mapping[str, Any]) -> None:
        cfg = self.config
        device = resolve_device(cfg.device)
        self.nodes = NodeIndex.from_state(state["nodes"])
        net = EulerNet(encoder=cfg.encoder, rnn=cfg.rnn, n_nodes=self.nodes.size, hidden=cfg.hidden, gnn_out=cfg.gnn_out,
                       rnn_hidden=cfg.rnn_hidden, embed=cfg.embed, layers=cfg.gnn_layers, dropout=cfg.dropout)
        self.net = load_module(net, directory / str(state["weights"]), device)
        self.net.eval()
        tensors = torch.load(directory / str(state["state_file"]), map_location="cpu", weights_only=True)
        self.state_ = (tensors["h"].to(device), tensors["c"].to(device))
        self.z_last = tensors["z"].to(device)
        self.origin = float(state["origin"])
        self.last_snapshot = int(state["last_snapshot"])
        self.cutoff = float(state["cutoff"])

    @classmethod
    def paper_protocol(cls) -> SplitProtocol:
        """Train on the events before the first anomalous one, test on the rest (baselines-notes.md, C5)."""
        return TimeCutoff(time_column="time", before_first_positive="label", align_seconds=1800.0)
