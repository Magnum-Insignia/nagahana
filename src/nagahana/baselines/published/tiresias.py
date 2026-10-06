"""Tiresias (Shen, Mariconti, Vervier and Stringhini, "Tiresias: Predicting Security Events Through Deep
Learning", ACM CCS 2018, DOI 10.1145/3243734.3243811) and the comparators of the same study.

What the paper states (baselines-notes.md, C1)
    task          predict the exact next security event of a machine (one of 4,495 event types) from its
                  earlier events, with recurrent neural networks
    data          3.4 billion events of a commercial intrusion-prevention system (27 days)
    protocol      80/10/10 train / validation / test with disjoint machines; temporal ageing tests
    metric        precision, micro-averaged over event types (equal to recall and F1, i.e. top-1 accuracy)
    results       0.81 - 0.83 on five same-day tests; up to 0.935 on later days
    comparators   3-gram 0.54 - 0.67, Markov chain 0.52 - 0.62, spectral learning 0.013 - 0.05
    monitoring    a mechanism that detects drops in precision and triggers retraining

Methods (config `method`)
    lstm      Tiresias: event embedding -> stacked LSTM -> softmax over the event vocabulary, trained on
              windows of `context` consecutive events of a machine by next-event cross-entropy
    ngram     order-3 n-gram: the next event's distribution given the two previous events; an unseen
              context backs off to the previous event, then to the unigram distribution
    markov    first-order Markov chain (backs off to the unigram distribution)
    spectral  spectral learning of an HMM (Hsu, Kakade and Zhang, Journal of Computer and System Sciences
              78(5), 2012, arXiv:0811.4413): with P1 = P(o1), P21 = P(o2, o1), P3x1[x] = P(o3, o2 = x, o1)
              over the `spectral_vocab` most frequent events (others merged into one symbol) and U the top
              `spectral_rank` left singular vectors of P21:
                  b1 = U^T P1,  b_inf = (P21^T U)^+ P1,  B_x = U^T P3x1[x] (U^T P21)^+
                  b_{t+1} = B_{o_t} b_t / (b_inf^T B_{o_t} b_t),  P(o_{t+1} = x | o_1:t) ~ b_inf^T B_x b_{t+1}
              (negative estimates clipped to 0 and renormalised: spectral estimates are not constrained to
              be probabilities)
Precision monitor: the top-1 accuracy over the last `monitor_window` scored events, compared with the
validation precision; an alarm marks a drop below (1 - monitor_drop) times it. The settings the paper does
not state are recorded in AS-562 (citation to verify).
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Iterator, Mapping
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
    Reference,
    ReportedResult,
)
from nagahana.baselines.published.frames import epoch_seconds, time_order
from nagahana.baselines.published.neural import EarlyStopping, load_module, minibatches, resolve_device, save_module
from nagahana.baselines.published.protocols import DisjointGroups, SplitProtocol
from nagahana.baselines.published.stages import attack_stages
from nagahana.core.errors import InvariantViolation
from nagahana.evaluation.predictions import StagePredictions, make_meta

Method = Literal["lstm", "ngram", "markov", "spectral"]


class TiresiasLSTM(nn.Module):
    """Embedding -> stacked LSTM -> logits over the vocabulary."""

    def __init__(self, vocab: int, embed: int, hidden: int, layers: int, dropout: float) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab, embed)
        self.lstm = nn.LSTM(embed, hidden, num_layers=layers, batch_first=True, dropout=dropout if layers > 1 else 0.0)
        self.out = nn.Linear(hidden, vocab)

    def forward(self, codes: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor] | None = None
                ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """codes [B, L] -> logits [B, L, V] (logit at position t predicts the event at t + 1)."""
        h, state = self.lstm(self.embed(codes), state)
        return self.out(h), state


@dataclass
class TiresiasConfig(BaselineConfig):
    """Tiresias and its comparators (module docstring).

    Attributes
    ----------
    method:
        "lstm", "ngram", "markov" or "spectral".
    entity_column, time_column, event_column:
        Machine, event time and event type columns.
    min_count:
        Events seen fewer times in training share the out-of-vocabulary code 0.
    embed, hidden, layers, dropout, context, learning_rate, batch_size, epochs, patience, validation_fraction:
        The LSTM and its training (validation: the latest share of each machine's training events).
    top_k:
        Ranked predictions kept per event (for top-k precision).
    spectral_vocab, spectral_rank:
        Spectral learning.
    monitor_window, monitor_drop:
        Precision monitor.
    event_stages:
        Optional event type -> ATT&CK tactic name, for stage posteriors of the next event.
    """

    method: Method = "lstm"
    entity_column: str = "entity"
    time_column: str = "time"
    event_column: str = "event"
    min_count: int = 1
    embed: int = 128
    hidden: int = 256
    layers: int = 2
    dropout: float = 0.2
    context: int = 50
    learning_rate: float = 0.001
    batch_size: int = 128
    epochs: int = 20
    patience: int = 3
    validation_fraction: float = 0.1
    top_k: int = 10
    spectral_vocab: int = 200
    spectral_rank: int = 20
    monitor_window: int = 1000
    monitor_drop: float = 0.1
    event_stages: dict[str, str] = field(default_factory=dict)

    def validate(self) -> None:
        super().validate()
        if min(self.embed, self.hidden, self.layers, self.context, self.batch_size, self.epochs, self.patience, self.top_k,
               self.spectral_vocab, self.spectral_rank, self.monitor_window) < 1 or self.min_count < 1:
            raise ValueError("sizes, epochs, patience, top_k, monitor_window and min_count must be >= 1")
        if not 0.0 <= self.dropout < 1.0 or not 0.0 <= self.validation_fraction < 1.0 or not 0.0 < self.monitor_drop < 1.0:
            raise ValueError("dropout and validation_fraction must lie in [0, 1), monitor_drop in (0, 1)")


REFERENCE = Reference(
    key="shen2018tiresias",
    authors="Shen, Mariconti, Vervier, Stringhini",
    title="Tiresias: Predicting Security Events Through Deep Learning",
    venue="Proceedings of the ACM Conference on Computer and Communications Security (CCS 2018), 592-605",
    year=2018,
    doi="10.1145/3243734.3243811",
    arxiv="1905.10328",
)

_P = "80/10/10 with disjoint machines; same-day test 01 Nov 2017"
SPEC = BaselineSpec(
    name="tiresias",
    title="Tiresias next-event prediction (LSTM) with its n-gram, Markov and spectral comparators",
    reference=REFERENCE,
    family="event-forecaster",
    input_schema=InputSchema(
        description="One row per security event: the machine (`entity`), event `time` and the event type (`event`).",
        required=("entity", "time", "event"),
        time="time",
        optional=("dataset", "network", "split"),
    ),
    outputs=("stage",),
    datasets=("symantec-ips-telemetry",),
    reported=(
        ReportedResult(dataset="symantec-ips-telemetry", protocol=_P, task="next-event", model="lstm",
                       values={"micro_precision": "0.83"}, location="Tab. 1, p. 6", variant={"method": "lstm"},
                       note="five same-day tests print 0.83, 0.82, 0.83, 0.82, 0.81"),
        ReportedResult(dataset="symantec-ips-telemetry", protocol=_P, task="next-event", model="ngram",
                       values={"micro_precision": "0.67"}, location="Tab. 1, p. 6", variant={"method": "ngram"}),
        ReportedResult(dataset="symantec-ips-telemetry", protocol=_P, task="next-event", model="markov",
                       values={"micro_precision": "0.62"}, location="Tab. 1, p. 6", variant={"method": "markov"}),
        ReportedResult(dataset="symantec-ips-telemetry", protocol=_P, task="next-event", model="spectral",
                       values={"micro_precision": "0.05"}, location="Tab. 1, p. 6", variant={"method": "spectral"}),
    ),
    third_party="tiresias",
    assumptions=("AS-533", "AS-562"),
)


class Tiresias(PublishedBaseline):
    """Tiresias and its comparators (module docstring)."""

    spec: ClassVar[BaselineSpec] = SPEC
    config_type: ClassVar[type[BaselineConfig]] = TiresiasConfig
    config: TiresiasConfig

    def __init__(self, config: BaselineConfig | None = None) -> None:
        super().__init__(config)
        self.vocab_: list[str] = []                       # code i + 1 -> event type; code 0 = out of vocabulary
        self.net: TiresiasLSTM | None = None
        self.counts: dict[str, Any] = {}
        self.spectral: dict[str, np.ndarray] = {}
        self.reference_precision = math.nan

    def required_columns(self) -> tuple[str, ...]:
        cfg = self.config
        return (cfg.entity_column, cfg.time_column, cfg.event_column)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab_) + 1

    def _encode(self, frame: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray]]:
        """Per machine (in time order): (row positions, event codes)."""
        cfg = self.config
        index = {v: i + 1 for i, v in enumerate(self.vocab_)}
        order = time_order(frame, cfg.time_column)
        ent = frame[cfg.entity_column].astype(str).to_numpy()[order]
        ev = frame[cfg.event_column].astype(str).to_numpy()[order]
        codes = np.asarray([index.get(e, 0) for e in ev], dtype=np.int64)
        out = []
        for name in pd.unique(ent):
            sel = ent == name
            out.append((order[sel], codes[sel]))
        return out

    def _distribution(self, history: np.ndarray) -> np.ndarray:
        """Next-event distribution [V] of the count-based and spectral methods for one history."""
        cfg = self.config
        uni = self.counts["unigram"]
        if cfg.method in ("ngram", "markov"):
            if cfg.method == "ngram" and history.size >= 2:
                c3 = self.counts["trigram"].get((int(history[-2]), int(history[-1])))
                if c3 is not None:
                    return c3
            if history.size >= 1:
                c2 = self.counts["bigram"].get(int(history[-1]))
                if c2 is not None:
                    return c2
            return uni
        raise InvariantViolation(f"no count distribution for method {cfg.method!r}")

    def _fit_counts(self, seqs: list[np.ndarray]) -> None:
        v = self.vocab_size
        uni = np.zeros(v)
        bi: dict[int, np.ndarray] = defaultdict(lambda: np.zeros(v))
        tri: dict[tuple[int, int], np.ndarray] = defaultdict(lambda: np.zeros(v))
        for s in seqs:
            np.add.at(uni, s, 1.0)
            for a, b in zip(s[:-1], s[1:], strict=True):
                bi[int(a)][b] += 1.0
            for a, b, c in zip(s[:-2], s[1:-1], s[2:], strict=True):
                tri[(int(a), int(b))][c] += 1.0
        norm = lambda x: x / x.sum()                                              # noqa: E731
        self.counts = {"unigram": norm(uni + 1e-12), "bigram": {k: norm(c) for k, c in bi.items()},
                       "trigram": {k: norm(c) for k, c in tri.items()}}

    def _fit_spectral(self, seqs: list[np.ndarray]) -> None:
        cfg = self.config
        uni = np.bincount(np.concatenate(seqs), minlength=self.vocab_size)
        top = np.argsort(-uni, kind="stable")[: cfg.spectral_vocab]
        sym = np.full(self.vocab_size, cfg.spectral_vocab, dtype=np.int64)       # other events -> one symbol
        sym[top] = np.arange(top.size)
        n = top.size + 1
        p1, p21, p3x1 = np.zeros(n), np.zeros((n, n)), np.zeros((n, n, n))       # p3x1[x, i, j] = P(o3=i, o2=x, o1=j)
        for s in seqs:
            o = sym[s]
            if o.size < 3:
                continue
            np.add.at(p1, o[:-2], 1.0)
            np.add.at(p21, (o[1:-1], o[:-2]), 1.0)
            np.add.at(p3x1, (o[1:-1], o[2:], o[:-2]), 1.0)
        total = p1.sum()
        if total == 0:
            raise InvariantViolation(f"{self.spec.name}: spectral learning needs sequences of at least three events")
        p1, p21, p3x1 = p1 / total, p21 / total, p3x1 / total
        m = min(cfg.spectral_rank, n)
        u = np.linalg.svd(p21)[0][:, :m]
        pinv = np.linalg.pinv(u.T @ p21)
        self.spectral = {"map": sym, "top": top, "b1": u.T @ p1, "binf": np.linalg.pinv(p21.T @ u) @ p1,
                         "B": np.stack([u.T @ p3x1[x] @ pinv for x in range(n)])}

    def _spectral_chunks(self, codes: np.ndarray, chunk: int) -> Iterator[tuple[int, np.ndarray]]:
        """P(next | history) for every prefix of one machine's codes (spectral method), `chunk` rows at a time."""
        sp = self.spectral
        o = sp["map"][codes]
        b = sp["b1"].copy()
        top = sp["top"]
        for s in range(0, codes.size, chunk):
            out = np.zeros((min(chunk, codes.size - s), self.vocab_size))
            for r in range(out.shape[0]):
                bx = sp["B"][o[s + r]] @ b
                denom = float(sp["binf"] @ bx)
                b = bx / denom if abs(denom) > 1e-300 else sp["b1"].copy()
                scores = np.clip(np.einsum("j,xjk,k->x", sp["binf"], sp["B"], b), 0.0, None)      # [n_symbols]
                p = np.zeros(self.vocab_size)
                p[top] = scores[: top.size]                                       # the "other" symbol gets no event
                total = p.sum()
                out[r] = p / total if total > 0 else self.counts["unigram"]
            yield s, out

    def _fit(self, data: pd.DataFrame, validation: pd.DataFrame | None, rng: np.random.Generator) -> None:
        cfg = self.config
        counts = Counter(data[cfg.event_column].astype(str).tolist())
        self.vocab_ = sorted(e for e, c in counts.items() if c >= cfg.min_count)
        seqs_all = [c for _, c in self._encode(data)]
        cut = [max(1, int(round((1.0 - cfg.validation_fraction) * s.size))) for s in seqs_all]
        train = [s[:k] for s, k in zip(seqs_all, cut, strict=True)]
        val = [s[max(0, k - 1):] for s, k in zip(seqs_all, cut, strict=True) if s.size - k >= 1]
        self._fit_counts(train)
        if cfg.method == "spectral":
            self._fit_spectral(train)
        elif cfg.method == "lstm":
            self._fit_lstm(train, val, rng)
        # Reference precision for the monitor: top-1 accuracy on the validation events.
        hits = [self._ranked(s)[:-1, 0] == s[1:] for s in val if s.size >= 2]
        self.reference_precision = float(np.concatenate(hits).mean()) if hits else math.nan
        self.fit_report.update({"vocabulary": len(self.vocab_), "machines": len(seqs_all), "validation_precision": self.reference_precision})

    def _windows(self, seqs: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        # Inputs [N, L] and next-event targets [N, L] (-100 past the end) from consecutive windows.
        length = self.config.context
        xs, ys = [], []
        for s in seqs:
            for start in range(0, max(1, s.size - 1), length):
                x = s[start:start + length]
                y = s[start + 1:start + 1 + length]
                if y.size == 0:
                    continue
                xpad = np.zeros(length, dtype=np.int64)
                ypad = np.full(length, -100, dtype=np.int64)
                xpad[: y.size] = x[: y.size]
                ypad[: y.size] = y
                xs.append(xpad)
                ys.append(ypad)
        if not xs:
            raise InvariantViolation(f"{self.spec.name}: no machine has two or more events")
        return np.stack(xs), np.stack(ys)

    def _fit_lstm(self, train: list[np.ndarray], val: list[np.ndarray], rng: np.random.Generator) -> None:
        cfg = self.config
        device = resolve_device(cfg.device)
        x, y = self._windows(train)
        vx, vy = self._windows(val) if any(s.size >= 2 for s in val) else (x[:0], y[:0])
        net = TiresiasLSTM(self.vocab_size, cfg.embed, cfg.hidden, cfg.layers, cfg.dropout).to(device)
        opt = torch.optim.Adam(net.parameters(), lr=cfg.learning_rate)
        stopper = EarlyStopping(cfg.patience)
        xt, yt = torch.as_tensor(x, device=device), torch.as_tensor(y, device=device)
        history: list[dict[str, float]] = []
        for epoch in range(cfg.epochs):
            net.train()
            for idx in minibatches(x.shape[0], cfg.batch_size, rng):
                ib = torch.as_tensor(idx, device=device)
                logits, _ = net(xt[ib])
                loss = nn.functional.cross_entropy(logits.flatten(0, 1), yt[ib].flatten(), ignore_index=-100)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 5.0)
                opt.step()
            net.eval()
            with torch.no_grad():
                if vx.shape[0]:
                    logits, _ = net(torch.as_tensor(vx, device=device))
                    val_loss = float(nn.functional.cross_entropy(logits.flatten(0, 1), torch.as_tensor(vy, device=device).flatten(),
                                                                 ignore_index=-100))
                else:
                    val_loss = float(loss.detach())
            history.append({"epoch": float(epoch), "train_loss": float(loss.detach()), "val_loss": val_loss})
            if stopper.step(val_loss, net, epoch):
                break
        stopper.restore(net)
        self.net = net
        self.fit_report["history"] = history

    @torch.no_grad()
    def _chunks(self, codes: np.ndarray, chunk: int = 2048) -> Iterator[tuple[int, np.ndarray]]:
        """(start, P(next event | events up to t) [rows, V]) over one machine's positions, in chunks.

        The LSTM state (or the spectral state) is carried from chunk to chunk, so every position conditions
        on the machine's whole history.
        """
        cfg = self.config
        if cfg.method == "lstm":
            assert self.net is not None
            device = resolve_device(cfg.device)
            self.net.eval()
            state: tuple[torch.Tensor, torch.Tensor] | None = None
            for s in range(0, codes.size, chunk):
                logits, state = self.net(torch.as_tensor(codes[None, s:s + chunk], device=device), state)
                yield s, torch.softmax(logits[0].double(), dim=-1).cpu().numpy()
        elif cfg.method == "spectral":
            yield from self._spectral_chunks(codes, chunk)
        else:
            for s in range(0, codes.size, chunk):
                yield s, np.stack([self._distribution(codes[max(0, t - 1): t + 1]) for t in range(s, min(s + chunk, codes.size))])

    def _top(self, d: np.ndarray) -> np.ndarray:
        """Top-k codes [rows, k] of distributions d (stable: ties go to the lower code); ranks beyond the
        vocabulary are -1."""
        k = self.config.top_k
        out = np.full((d.shape[0], k), -1, dtype=np.int64)
        ranked = np.argsort(-d, axis=1, kind="mergesort")[:, :k]
        out[:, : ranked.shape[1]] = ranked
        return out

    def _ranked(self, codes: np.ndarray) -> np.ndarray:
        """Top-k next-event codes [T, k] for every position of one machine."""
        out = np.zeros((codes.size, self.config.top_k), dtype=np.int64)
        for s, d in self._chunks(codes):
            out[s:s + d.shape[0]] = self._top(d)
        return out

    def _predict(self, data: pd.DataFrame) -> PredictionParts:
        cfg = self.config
        n = len(data)
        topk = np.zeros((n, cfg.top_k), dtype=np.int64)
        label = np.full(n, -1, dtype=np.int64)
        stage_probs: np.ndarray | None = None
        names = attack_stages() if cfg.event_stages else ()
        if cfg.event_stages:
            index = {s: i for i, s in enumerate(names)}
            ev_stage = np.full(self.vocab_size, -1, dtype=np.int64)
            for i, e in enumerate(self.vocab_):
                if e in cfg.event_stages:
                    ev_stage[i + 1] = index[cfg.event_stages[e]]
            stage_probs = np.zeros((n, len(names)))
            stage_label = np.full(n, -1, dtype=np.int64)
        for rows, codes in self._encode(data):
            label[rows[:-1]] = codes[1:]
            if stage_probs is not None:
                stage_label[rows[:-1]] = ev_stage[codes[1:]]
            for s, d in self._chunks(codes):                                       # d [chunk, V]
                r = rows[s:s + d.shape[0]]
                topk[r] = self._top(d)
                if stage_probs is not None:
                    mapped = ev_stage >= 0
                    sp = np.zeros((d.shape[0], len(names)))
                    np.add.at(sp.T, ev_stage[mapped], d[:, mapped].T)
                    tot = sp.sum(axis=1, keepdims=True)
                    stage_probs[r] = np.where(tot > 0, sp / np.where(tot > 0, tot, 1.0), 1.0 / len(names))
        # Precision monitor over the scored events in time order.
        t = epoch_seconds(data[cfg.time_column])
        order = np.argsort(t, kind="stable")
        scored = order[label[order] >= 0]
        hit = (topk[scored, 0] == label[scored]).astype(np.float64)
        rolling = np.convolve(hit, np.ones(cfg.monitor_window), "full")[: hit.size] / np.minimum(np.arange(1, hit.size + 1), cfg.monitor_window)
        alarm = np.zeros(n, dtype=np.int64)
        if not math.isnan(self.reference_precision):
            alarm[scored] = (rolling < (1.0 - cfg.monitor_drop) * self.reference_precision).astype(np.int64)
        meta = make_meta(n, time=t, entity=pd.factorize(data[cfg.entity_column].astype(str))[0].astype(np.int64))
        component = {"next_event_topk": topk, "next_event_label": label, "precision_alarm": alarm}
        stage = None
        if stage_probs is not None:
            stage = StagePredictions(probs=stage_probs, label=stage_label, stage_names=names, meta=meta)
        return PredictionParts(stage=stage, component=component)

    def _export_state(self, directory: Path) -> dict[str, Any]:
        state: dict[str, Any] = {"vocab": self.vocab_, "reference_precision": self.reference_precision,
                                 "unigram": self.counts["unigram"].tolist(),
                                 "bigram": {str(k): v.tolist() for k, v in self.counts["bigram"].items()},
                                 "trigram": {f"{a},{b}": v.tolist() for (a, b), v in self.counts["trigram"].items()}}
        if self.net is not None:
            state["weights"] = save_module(self.net, directory / "tiresias.pt")
        if self.spectral:
            np.savez(directory / "spectral.npz", **self.spectral)
            state["spectral"] = "spectral.npz"
        return state

    def _import_state(self, directory: Path, state: Mapping[str, Any]) -> None:
        cfg = self.config
        self.vocab_ = list(state["vocab"])
        self.reference_precision = float(state["reference_precision"])
        self.counts = {"unigram": np.asarray(state["unigram"]),
                       "bigram": {int(k): np.asarray(v) for k, v in state["bigram"].items()},
                       "trigram": {tuple(int(x) for x in k.split(",")): np.asarray(v) for k, v in state["trigram"].items()}}
        if "weights" in state:
            net = TiresiasLSTM(self.vocab_size, cfg.embed, cfg.hidden, cfg.layers, cfg.dropout)
            self.net = load_module(net, directory / str(state["weights"]), resolve_device(cfg.device))
        if "spectral" in state:
            with np.load(directory / str(state["spectral"])) as z:
                self.spectral = {k: z[k] for k in z.files}

    @classmethod
    def paper_protocol(cls) -> SplitProtocol:
        """80/10/10 with disjoint machines (baselines-notes.md, C1)."""
        return DisjointGroups("entity", (0.8, 0.1, 0.1))
