"""CyberWorld as a forecasting baseline on NagaHana's data (AS-560).

The world model of world.py, without actions, learns the dynamics of a network's windows; heads on its
latent state give the current window's infiltration probability and ATT&CK stage. From the filtered state
at each trigger (the end of a window), N latent rollouts of K prior steps forecast:

    P_inf(k) = (1/N) sum_n [1 - prod_{j <= k} (1 - c(s^n_j))]       c: the infiltration head
    the stage of the current window (stage head on the filtered state)
    the next states: the decoded window features at the horizons 1, ceil(K/2), K, averaged over rollouts
Each rollout's curve is non-decreasing, so P_inf is too; the rollouts are kept as an ensemble for the CRPS.

Windows (the units of NagaHana's forecast evaluation): per network (sequence column), windows of w
seconds from the network's first event; trigger time = window end. Window features are the next-state
features of the evaluation design: flows, packets, bytes, distinct destination hosts, distinct destination
ports, and the shares of SYN, RST and failed (unanswered) flows. A flow field that is not supplied (NaN)
contributes nothing to a sum, and a share is taken over the flows where its field is supplied.

Graph variant: per window, the hosts (at most `max_nodes`, the most active) as nodes with features (flows
out, flows in, bytes out, bytes in, distinct peers, SYN share of outgoing flows) and an undirected
adjacency from the window's flows.

Labels: a window is infiltrated when one of its flows carries a stage of the infiltration set (AS-18), or,
without a stage column, a malicious label; the window's stage is the latest stage along the ATT&CK order
among its flows ("none" for benign windows, unknown when no flow is labelled).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
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
)
from nagahana.baselines.published.cyberwheel.ppo import CYBERWORLD_REFERENCE
from nagahana.baselines.published.cyberworld.networks import symexp
from nagahana.baselines.published.cyberworld.rssm import RSSMState
from nagahana.baselines.published.cyberworld.world import Modality, Variant, WorldModel, WorldModelSettings, variant_modalities
from nagahana.baselines.published.frames import epoch_seconds, normalise_token
from nagahana.baselines.published.neural import resolve_device
from nagahana.baselines.published.protocols import GivenSplit, SplitProtocol
from nagahana.baselines.published.stages import attack_stages, infiltration_stage_names
from nagahana.core.errors import InvariantViolation
from nagahana.evaluation.predictions import (
    DetectionPredictions,
    ForecastPredictions,
    StagePredictions,
    StateForecastPredictions,
    make_meta,
)

WINDOW_FEATURES: tuple[str, ...] = ("flows", "packets", "bytes", "distinct_dst_hosts", "distinct_dst_ports", "syn_share",
                                    "rst_share", "failed_share")
NODE_FEATURES: tuple[str, ...] = ("flows_out", "flows_in", "bytes_out", "bytes_in", "distinct_peers", "syn_share_out")


@dataclass
class CyberWorldForecasterConfig(BaselineConfig):
    """The forecasting form of CyberWorld (module docstring).

    Attributes
    ----------
    variant, world:
        "vector" or "graph" ("text" and "multimodal" need a frozen text encoder and are not available for
        flow data); world-model sizes.
    window_seconds, horizon, rollouts, state_horizons, max_nodes:
        Window length w, forecast steps K, rollouts N, next-state horizons (empty: 1, ceil(K/2), K) and the
        node cap of the graph variant.
    sequence_column, time_column, src_column, dst_column, dport_column, packet_columns, byte_columns,
    syn_column, rst_column, flags_column, failed_column, label_column, stage_column, family_column:
        Flow columns (defaults: NagaHana field identifiers; absent columns are not supplied).
    train_steps, batch_size, batch_length, learning_rate, adam_eps, grad_clip:
        World-model training (Adam, global gradient-norm clipping).
    threshold:
        Operating threshold on the current window's infiltration probability.
    """

    variant: Variant = "vector"
    world: WorldModelSettings = field(default_factory=WorldModelSettings)
    window_seconds: float = 60.0
    horizon: int = 12
    rollouts: int = 16
    state_horizons: tuple[int, ...] = ()
    max_nodes: int = 64
    sequence_column: str = "network"
    time_column: str = "time"
    src_column: str = "src"
    dst_column: str = "dst"
    dport_column: str = "flow.dst_port"
    packet_columns: tuple[str, ...] = ("flow.packets_fwd", "flow.packets_bwd")
    byte_columns: tuple[str, ...] = ("flow.bytes_fwd", "flow.bytes_bwd")
    syn_column: str = "flow.flag_count.syn"
    rst_column: str = "flow.flag_count.rst"
    flags_column: str = "flow.tcp_flags"
    failed_column: str = "flow.unanswered"
    label_column: str = "malicious"
    stage_column: str = "stage"
    family_column: str = "family"
    train_steps: int = 10_000
    batch_size: int = 16
    batch_length: int = 64
    learning_rate: float = 1e-4
    adam_eps: float = 1e-8
    grad_clip: float = 1000.0
    threshold: float = 0.5

    def validate(self) -> None:
        super().validate()
        self.world.validate()
        if self.variant not in ("vector", "graph"):
            raise ValueError("flow data support the 'vector' and 'graph' variants (text variants need text and a frozen encoder)")
        if self.window_seconds <= 0 or min(self.horizon, self.rollouts, self.max_nodes, self.train_steps, self.batch_size,
                                           self.batch_length) < 1:
            raise ValueError("window_seconds > 0 and horizon, rollouts, max_nodes, train_steps, batch sizes >= 1")
        if any(h < 1 or h > self.horizon for h in self.state_horizons):
            raise ValueError("state_horizons must lie in 1 ... horizon")

    def horizons(self) -> tuple[int, ...]:
        return self.state_horizons or tuple(sorted({1, math.ceil(self.horizon / 2), self.horizon}))


@dataclass
class WindowSequence:
    """The windows of one network: features [W, 8], graphs, labels and times."""

    name: str
    start: float
    features: np.ndarray
    infiltrated: np.ndarray
    stage: np.ndarray
    family: list[str]
    nodes: list[np.ndarray] = field(default_factory=list)
    edges: list[np.ndarray] = field(default_factory=list)

    @property
    def length(self) -> int:
        return int(self.features.shape[0])


def _num(frame: pd.DataFrame, column: str) -> np.ndarray:
    if column not in frame.columns:
        return np.full(len(frame), np.nan)
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float64)


def _sum_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> np.ndarray:
    # NaN only where every listed column is missing for the flow.
    parts = np.stack([_num(frame, c) for c in columns]) if columns else np.full((1, len(frame)), np.nan)
    present = ~np.isnan(parts)
    return np.where(present.any(axis=0), np.nansum(parts, axis=0), np.nan)


def _flag(frame: pd.DataFrame, count_column: str, flags_column: str, bit: int) -> np.ndarray:
    # 1 when the flag was seen (count > 0, or the bit of the flag bitmask), NaN when neither field is supplied.
    count = _num(frame, count_column)
    bits = _num(frame, flags_column)
    from_bits = np.where(np.isnan(bits), np.nan, (np.nan_to_num(bits).astype(np.int64) & bit) > 0)
    return np.where(~np.isnan(count), (count > 0).astype(np.float64), from_bits.astype(np.float64))


FORECASTER_SPEC = BaselineSpec(
    name="cyberworld-forecaster",
    title="CyberWorld world model as a forecaster of infiltration, stages and next states",
    reference=CYBERWORLD_REFERENCE,
    family="stage-forecaster",
    input_schema=InputSchema(
        description="One row per flow (NagaHana's state updates): event `time`, `src` and `dst` entities, the flow fields "
                    "named in the config (NagaHana field identifiers by default), a `network` id, and the labels "
                    "(`stage` as ATT&CK tactic names, or `malicious`).",
        required=("time", "src", "dst"),
        time="time",
        optional=("network", "stage", "malicious", "family", "dataset", "split"),
    ),
    outputs=("detection", "forecast", "stage", "state_forecast"),
    datasets=("cse-cic-ids2018", "cic-ids2017", "ctu13", "unsw-nb15", "cic-iot-2023"),
    reported=(),
    third_party="cyberworld",
    assumptions=("AS-557", "AS-560"),
)

class CyberWorldForecaster(PublishedBaseline):
    """CyberWorld's RSSM as a forecaster of P_inf, stages and next window states (module docstring)."""

    spec: ClassVar[BaselineSpec] = FORECASTER_SPEC
    config_type: ClassVar[type[BaselineConfig]] = CyberWorldForecasterConfig
    config: CyberWorldForecasterConfig

    def __init__(self, config: BaselineConfig | None = None) -> None:
        super().__init__(config)
        self.world: WorldModel | None = None
        self.mean_: np.ndarray = np.zeros(len(WINDOW_FEATURES))
        self.std_: np.ndarray = np.ones(len(WINDOW_FEATURES))

    def required_columns(self) -> tuple[str, ...]:
        cfg = self.config
        return (cfg.time_column, cfg.src_column, cfg.dst_column)

    def windows(self, frame: pd.DataFrame) -> list[WindowSequence]:
        """Window sequences of a flow frame (module docstring)."""
        cfg = self.config
        t = epoch_seconds(frame[cfg.time_column])
        seq = frame[cfg.sequence_column].astype(str).to_numpy() if cfg.sequence_column in frame.columns else np.full(len(frame), "network")
        infil_names = {normalise_token(s) for s in infiltration_stage_names()}
        stage_names = attack_stages()
        stage_index = {normalise_token(s): i for i, s in enumerate(stage_names)}
        if cfg.stage_column in frame.columns:
            raw = frame[cfg.stage_column].tolist()
            stage_code = np.asarray([stage_index.get(normalise_token(v), int(v) if isinstance(v, int | np.integer) and 0 <= v < len(stage_names) else -1)
                                     for v in raw], dtype=np.int64)
            infil = np.where(stage_code >= 0, np.isin([normalise_token(stage_names[c]) if c >= 0 else "" for c in stage_code],
                                                      list(infil_names)).astype(np.int64), -1)
        else:
            lab = _num(frame, cfg.label_column)
            stage_code = np.full(len(frame), -1, dtype=np.int64)
            infil = np.where(np.isnan(lab), -1, (lab > 0).astype(np.int64))
        packets, nbytes = _sum_columns(frame, cfg.packet_columns), _sum_columns(frame, cfg.byte_columns)
        syn, rst = _flag(frame, cfg.syn_column, cfg.flags_column, 0x02), _flag(frame, cfg.rst_column, cfg.flags_column, 0x04)
        failed = _num(frame, cfg.failed_column)
        dport = _num(frame, cfg.dport_column)
        src = frame[cfg.src_column].astype(str).to_numpy()
        dst = frame[cfg.dst_column].astype(str).to_numpy()
        fam = frame[cfg.family_column].astype(str).to_numpy() if cfg.family_column in frame.columns else np.full(len(frame), "")
        out: list[WindowSequence] = []
        for name in pd.unique(seq):
            rows = np.nonzero(seq == name)[0]
            t0 = float(t[rows].min())
            w = np.floor((t[rows] - t0) / cfg.window_seconds).astype(np.int64)
            n_w = int(w.max()) + 1
            feats = np.zeros((n_w, len(WINDOW_FEATURES)))
            win_infil = np.full(n_w, -1, dtype=np.int64)
            win_stage = np.full(n_w, -1, dtype=np.int64)
            win_family = ["benign"] * n_w
            nodes: list[np.ndarray] = []
            edges: list[np.ndarray] = []
            order = np.argsort(w, kind="stable")
            bounds = np.searchsorted(w[order], np.arange(n_w + 1))
            for k in range(n_w):
                r = rows[order[bounds[k]:bounds[k + 1]]]
                if r.size:
                    def share(x: np.ndarray) -> float:
                        ok = ~np.isnan(x)
                        return float(x[ok].mean()) if ok.any() else 0.0
                    feats[k] = [r.size, np.nansum(packets[r]), np.nansum(nbytes[r]), np.unique(dst[r]).size,
                                np.unique(dport[r][~np.isnan(dport[r])]).size, share(syn[r]), share(rst[r]), share(failed[r])]
                    known = infil[r] >= 0
                    if known.any():
                        win_infil[k] = int(infil[r][known].max())
                    st = stage_code[r]
                    if (st >= 0).any():
                        win_stage[k] = int(st[st >= 0].max())               # the latest stage along the matrix order
                    elif cfg.stage_column not in frame.columns and known.any():
                        win_stage[k] = 0 if win_infil[k] == 0 else -1
                    malicious = (infil[r] > 0) | (st > 0)
                    if malicious.any():
                        names = [f for f in fam[r][malicious].tolist() if f and normalise_token(f) != "benign"]
                        win_family[k] = str(pd.Series(names).mode().iloc[0]) if names else "attack"
                    else:
                        win_family[k] = "benign" if (known.any() or (st >= 0).any()) else "unknown"
                if cfg.variant == "graph":
                    nf, ed = self._window_graph(src[r], dst[r], nbytes[r], syn[r])
                    nodes.append(nf)
                    edges.append(ed)
            out.append(WindowSequence(str(name), t0, feats, win_infil, win_stage, win_family, nodes, edges))
        return out

    def _window_graph(self, src: np.ndarray, dst: np.ndarray, nbytes: np.ndarray, syn: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Node features [v, 6] of the most active hosts and their undirected edges [e, 2] (local indices)."""
        if src.size == 0:
            return np.zeros((0, len(NODE_FEATURES))), np.zeros((0, 2), dtype=np.int64)
        hosts, inv = np.unique(np.concatenate([src, dst]), return_inverse=True)
        s_i, d_i = inv[: src.size], inv[src.size:]
        flows_out = np.bincount(s_i, minlength=hosts.size).astype(np.float64)
        flows_in = np.bincount(d_i, minlength=hosts.size).astype(np.float64)
        b = np.nan_to_num(nbytes)
        bytes_out = np.bincount(s_i, weights=b, minlength=hosts.size)
        bytes_in = np.bincount(d_i, weights=b, minlength=hosts.size)
        pairs = np.unique(np.stack([np.minimum(s_i, d_i), np.maximum(s_i, d_i)], axis=1), axis=0)
        pairs = pairs[pairs[:, 0] != pairs[:, 1]]
        peers = np.bincount(pairs.ravel(), minlength=hosts.size).astype(np.float64)
        syn_ok = ~np.isnan(syn)
        syn_out = np.divide(np.bincount(s_i[syn_ok], weights=syn[syn_ok], minlength=hosts.size),
                            np.bincount(s_i[syn_ok], minlength=hosts.size), out=np.zeros(hosts.size),
                            where=np.bincount(s_i[syn_ok], minlength=hosts.size) > 0)
        feats = np.stack([flows_out, flows_in, bytes_out, bytes_in, peers, syn_out], axis=1)
        keep = np.argsort(-(flows_out + flows_in), kind="stable")[: self.config.max_nodes]
        remap = np.full(hosts.size, -1)
        remap[keep] = np.arange(keep.size)
        e = remap[pairs]
        e = e[(e >= 0).all(axis=1)]
        return feats[keep], e

    def _modalities(self) -> list[Modality]:
        vector = [Modality("window", "vector", len(WINDOW_FEATURES))]
        graph = Modality("hosts", "graph", len(NODE_FEATURES), slots=0, decode=False)
        return variant_modalities(self.config.variant, vector=vector, graph=graph, text=None)

    def _stream(self, seqs: list[WindowSequence]) -> dict[str, Any]:
        """All sequences back to back, with `first` set at every sequence start (DreamerV3's replay layout)."""
        feats = np.concatenate([s.features for s in seqs]).astype(np.float32)
        first = np.concatenate([np.r_[True, np.zeros(s.length - 1, dtype=bool)] for s in seqs])
        out: dict[str, Any] = {"window": feats, "first": first,
                               "infiltration": np.concatenate([s.infiltrated for s in seqs]),
                               "stage": np.concatenate([s.stage for s in seqs])}
        if self.config.variant == "graph":
            out["nodes"] = [n for s in seqs for n in s.nodes]
            out["edges"] = [e for s in seqs for e in s.edges]
        return out

    def _batch_obs(self, stream: Mapping[str, Any], starts: np.ndarray, length: int, device: torch.device) -> dict[str, torch.Tensor]:
        """Observation tensors [B, L, ...] of stream positions start ... start + L - 1; each sequence starts fresh."""
        idx = starts[:, None] + np.arange(length)[None, :]                            # [B, L]
        first = stream["first"][idx].copy()
        first[:, 0] = True
        out: dict[str, torch.Tensor] = {
            "window": torch.as_tensor(stream["window"][idx], device=device),
            "first": torch.as_tensor(first, device=device),
            "label.infiltration": torch.as_tensor(stream["infiltration"][idx], device=device),
            "label.stage": torch.as_tensor(stream["stage"][idx], device=device),
        }
        if self.config.variant == "graph":
            v_max = max(1, max(stream["nodes"][i].shape[0] for i in idx.ravel()))
            b = idx.shape[0]
            nodes = np.zeros((b, length, v_max, len(NODE_FEATURES)), dtype=np.float32)
            adj = np.zeros((b, length, v_max, v_max), dtype=bool)
            mask = np.zeros((b, length, v_max), dtype=bool)
            for bi in range(b):
                for j in range(length):
                    nf, ed = stream["nodes"][idx[bi, j]], stream["edges"][idx[bi, j]]
                    nodes[bi, j, : nf.shape[0]] = nf
                    mask[bi, j, : nf.shape[0]] = True
                    if ed.size:
                        adj[bi, j, ed[:, 0], ed[:, 1]] = True
                        adj[bi, j, ed[:, 1], ed[:, 0]] = True
            out["hosts.nodes"] = torch.as_tensor(nodes, device=device)
            out["hosts.adjacency"] = torch.as_tensor(adj, device=device)
            out["hosts.mask"] = torch.as_tensor(mask, device=device)
        return out

    def _build(self) -> WorldModel:
        return WorldModel(self._modalities(), self.config.world, action_dim=0, reward_head=False, continue_head=False,
                          label_heads={"infiltration": 1, "stage": len(attack_stages())})

    def _fit(self, data: pd.DataFrame, validation: pd.DataFrame | None, rng: np.random.Generator) -> None:
        cfg = self.config
        device = resolve_device(cfg.device)
        seqs = self.windows(data)
        allf = np.concatenate([s.features for s in seqs])
        self.mean_, self.std_ = allf.mean(axis=0), allf.std(axis=0)
        self.std_[self.std_ == 0] = 1.0
        world = self._build().to(device)
        opt = torch.optim.Adam(world.parameters(), lr=cfg.learning_rate, eps=cfg.adam_eps)
        stream = self._stream(seqs)
        total = int(stream["first"].size)
        length = min(cfg.batch_length, total)
        lengths = np.asarray([s.length for s in seqs])
        history: list[dict[str, float]] = []
        for step in range(cfg.train_steps):
            starts = rng.integers(0, total - length + 1, size=cfg.batch_size)
            batch = self._batch_obs(stream, starts, length, device)
            loss, metrics, _ = world.loss(batch)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(world.parameters(), cfg.grad_clip)
            opt.step()
            if step % max(1, cfg.train_steps // 50) == 0 or step == cfg.train_steps - 1:
                history.append({"step": float(step), **metrics})
        self.world = world
        self.fit_report.update({"sequences": len(seqs), "windows": int(lengths.sum()), "history": history,
                                "infiltrated_windows": int(sum((s.infiltrated > 0).sum() for s in seqs))})

    @torch.no_grad()
    def _filter(self, seq: WindowSequence, device: torch.device) -> RSSMState:
        assert self.world is not None
        batch = self._batch_obs(self._stream([seq]), np.zeros(1, dtype=np.int64), seq.length, device)
        obs = {k: v for k, v in batch.items() if k.split(".")[0] in self.world.encoders}
        e = self.world.embed(obs, (1, seq.length))
        post, _ = self.world.rssm.observe(e, None, batch["first"].bool())
        return RSSMState(post.h[0], post.logits[0], post.z[0])                     # [W, ...]

    @torch.no_grad()
    def _rollouts(self, start: RSSMState) -> tuple[np.ndarray, np.ndarray]:
        """Infiltration probabilities [m, N, K] and decoded raw features [m, N, K, 8] of N rollouts per start."""
        assert self.world is not None
        cfg = self.config
        m = start.h.shape[0]
        rep = lambda x: x.repeat_interleave(cfg.rollouts, dim=0)                    # noqa: E731
        state = RSSMState(rep(start.h), rep(start.logits), rep(start.z))
        probs, feats = [], []
        for _ in range(cfg.horizon):
            state = self.world.rssm.img_step(state, None)
            f = state.features()
            probs.append(self.world.label_probs(f, "infiltration")[..., 0])
            feats.append(symexp(self.world.decoders["window"](f).double()))
        p = torch.stack(probs, dim=1).view(m, cfg.rollouts, cfg.horizon).cpu().numpy()
        x = torch.stack(feats, dim=1).view(m, cfg.rollouts, cfg.horizon, len(WINDOW_FEATURES)).cpu().numpy()
        return p, x

    @torch.no_grad()
    def _predict(self, data: pd.DataFrame) -> PredictionParts:
        assert self.world is not None
        cfg = self.config
        device = resolve_device(cfg.device)
        self.world.eval()
        seqs = self.windows(data)
        k_max, hz = cfg.horizon, self.config.horizons()
        rows: dict[str, list[Any]] = {k: [] for k in ("time", "network", "family", "infil", "stage", "p_cur", "stage_probs",
                                                     "ensemble", "event", "observed", "pred", "obs", "mask")}
        for seq in seqs:
            post = self._filter(seq, device)
            feat = post.features()
            p_cur = self.world.label_probs(feat, "infiltration")[..., 0].cpu().numpy()
            stage_p = self.world.label_probs(feat, "stage").cpu().numpy()
            c, x = self._rollouts(post)
            ens = 1.0 - np.cumprod(1.0 - c, axis=2)                                 # [W, N, K] per-rollout P_inf
            pred = (x.mean(axis=1)[:, [h - 1 for h in hz]] - self.mean_) / self.std_  # [W, H, 8] standardised
            for t in range(seq.length):
                ev, obs_steps = 0, 0
                for k in range(1, k_max + 1):
                    if t + k >= seq.length or seq.infiltrated[t + k] < 0:
                        break
                    obs_steps = k
                    if seq.infiltrated[t + k] > 0:
                        ev = k
                        break
                o = np.zeros((len(hz), len(WINDOW_FEATURES)))
                msk = np.zeros((len(hz), len(WINDOW_FEATURES)), dtype=bool)
                for j, h in enumerate(hz):
                    if t + h < seq.length:
                        o[j] = (seq.features[t + h] - self.mean_) / self.std_
                        msk[j] = True
                rows["time"].append(seq.start + (t + 1) * cfg.window_seconds)
                rows["network"].append(seq.name)
                rows["family"].append(seq.family[t])
                rows["infil"].append(int(seq.infiltrated[t]))
                rows["stage"].append(int(seq.stage[t]))
                rows["p_cur"].append(float(p_cur[t]))
                rows["stage_probs"].append(stage_p[t])
                rows["ensemble"].append(ens[t])
                rows["event"].append(ev)
                rows["observed"].append(obs_steps)
                rows["pred"].append(pred[t])
                rows["obs"].append(o)
                rows["mask"].append(msk)
        n = len(rows["time"])
        if n == 0:
            raise InvariantViolation(f"{self.spec.name}: the frame produced no window")
        meta = make_meta(n, time=np.asarray(rows["time"]), network=np.asarray(rows["network"], dtype=object),
                         family=np.asarray(rows["family"], dtype=object), dataset=str(data["dataset"].iloc[0]) if "dataset" in data.columns else "")
        ensemble = np.stack(rows["ensemble"])
        p_inf = ensemble.mean(axis=1)
        prev = np.concatenate([np.zeros((n, 1)), p_inf[:, :-1]], axis=1)
        hazard = np.clip(np.divide(p_inf - prev, 1.0 - prev, out=np.ones_like(p_inf), where=(1.0 - prev) > 1e-12), 0.0, 1.0)
        stage_probs = np.stack(rows["stage_probs"])
        stage_probs = stage_probs / stage_probs.sum(axis=1, keepdims=True)
        return PredictionParts(
            detection=DetectionPredictions(score=np.clip(np.asarray(rows["p_cur"]), 0, 1), label=np.asarray(rows["infil"]),
                                           unit="window", meta=meta, threshold=cfg.threshold),
            forecast=ForecastPredictions(p_inf=np.maximum.accumulate(p_inf, axis=1), window_seconds=cfg.window_seconds,
                                         event_step=np.asarray(rows["event"]), observed_steps=np.asarray(rows["observed"]),
                                         meta=meta, hazard=hazard, ensemble=np.maximum.accumulate(ensemble, axis=2)),
            stage=StagePredictions(probs=stage_probs, label=np.asarray(rows["stage"]), stage_names=attack_stages(), meta=meta),
            state_forecast=StateForecastPredictions(predicted=np.stack(rows["pred"]), observed=np.stack(rows["obs"]),
                                                    mask=np.stack(rows["mask"]), horizons=np.asarray(hz), feature_names=WINDOW_FEATURES,
                                                    meta=meta),
        )

    def _export_state(self, directory: Path) -> dict[str, Any]:
        assert self.world is not None
        torch.save(self.world.state_dict(), directory / "world.pt")
        return {"weights": "world.pt", "mean": self.mean_.tolist(), "std": self.std_.tolist()}

    def _import_state(self, directory: Path, state: Mapping[str, Any]) -> None:
        self.mean_ = np.asarray(state["mean"], dtype=np.float64)
        self.std_ = np.asarray(state["std"], dtype=np.float64)
        world = self._build()
        world.load_state_dict(torch.load(directory / str(state["weights"]), map_location="cpu", weights_only=True))
        self.world = world.to(resolve_device(self.config.device))
        self.world.eval()

    @classmethod
    def paper_protocol(cls) -> SplitProtocol:
        """NagaHana's own splits (the frame's split column), as the comparison requires."""
        return GivenSplit()
