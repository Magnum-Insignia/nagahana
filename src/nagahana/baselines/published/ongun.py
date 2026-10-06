"""Host-window classifiers on CTU-13 (Ongun, Sakharaov, Boboila, Oprea and Eliassi-Rad, arXiv:1907.04846,
2019, "On Designing Machine Learning Models for Malicious Network Traffic Classification"; preprint).

What the paper states (baselines-notes.md, A2)
    data          CTU-13 connection logs (Zeek); Neris scenarios 1, 2 and 9 (and three Rbot scenarios)
    protocol      train on two scenarios, test on the held-out third (a random split would correlate
                  training and test windows)
    representation aggregated traffic statistics and temporal (inter-arrival) features per internal host,
                  per port, over 30-second windows
    labels        a window is malicious if it contains at least one attack event; Neris uses coarse
                  labels (all traffic of the botnet addresses)
    learners      logistic regression with L1 (Lasso) regularisation, random forest, gradient boosting
    metrics       precision, recall, F1 and AUC of the malicious (minority) class

The window features as reproduced (AS-539)
    Per (scenario, internal host, 30-second window) with at least one connection of the host:
      totals of the host's outgoing connections: count, bytes sent, bytes received, packets, distinct
      destination addresses, distinct destination ports, mean / std / max duration, share of
      connections without a single response byte; count of incoming connections
      per destination port p (the `top_ports` most frequent ports of the training connections, plus one
      "other" port class): count, bytes sent, bytes received, distinct destination addresses
      per transport protocol (tcp, udp, icmp, other): count
      inter-arrival times of the host's outgoing connection starts in the window: mean, std, min, max
    A window with fewer than two outgoing connections has no inter-arrival time: those cells are NaN and
    repaired with the training medians (preprocessing.FiniteGuard); count features are never NaN.
    Coarse labels: every window of an infected host is malicious, every window of another internal host
    benign (the paper's imbalance, 1:134 at 30 s for Neris, implies that the other hosts' windows count as
    benign). Fine labels: a window is malicious if one of its connections carries a botnet label.
    Learners: scikit-learn defaults except the L1 penalty (liblinear solver) of logistic regression,
    which also gets standardised features.
"""

from __future__ import annotations

import ipaddress
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Literal

import numpy as np
import pandas as pd

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
from nagahana.baselines.published.features import ctu13
from nagahana.baselines.published.frames import ctu13_label_class
from nagahana.baselines.published.preprocessing import ColumnPipeline
from nagahana.baselines.published.protocols import GroupHoldout, SplitProtocol
from nagahana.core.errors import InvariantViolation
from nagahana.evaluation.predictions import DetectionPredictions, make_meta

PROTOCOLS: tuple[str, ...] = ("tcp", "udp", "icmp")


@dataclass
class OngunConfig(BaselineConfig):
    """One learner of the study on 30-second host windows.

    Attributes
    ----------
    learner:
        "logistic_regression" (L1), "random_forest" or "gradient_boosting".
    params:
        Learner hyperparameters (logistic regression's L1 penalty is the paper's).
    window_seconds:
        Aggregation window (30 s in the reported table).
    internal_network:
        CIDR of the monitored network; only its hosts get windows.
    top_ports:
        Destination ports with their own per-port statistics (most frequent in training).
    label_mode:
        "coarse" (all windows of infected hosts malicious) or "fine" (connection labels).
    scenario_column:
        Column naming each connection's CTU-13 scenario (windows never span scenarios).
    infected:
        Infected addresses; empty derives them from the scenario numbers (features.ctu13.INFECTED_HOSTS).
    threshold:
        Operating threshold on P(malicious).
    """

    learner: Literal["logistic_regression", "random_forest", "gradient_boosting"] = "random_forest"
    params: dict[str, Any] = field(default_factory=dict)
    window_seconds: float = 30.0
    internal_network: str = ctu13.CTU_INTERNAL_NETWORK
    top_ports: int = 20
    label_mode: Literal["coarse", "fine"] = "coarse"
    scenario_column: str = "scenario"
    infected: tuple[str, ...] = ()
    threshold: float = 0.5

    def validate(self) -> None:
        super().validate()
        if self.window_seconds <= 0:
            raise ValueError("window_seconds must be positive")
        if self.top_ports < 0:
            raise ValueError("top_ports must be non-negative")
        ipaddress.ip_network(self.internal_network)

    def learner_params(self) -> dict[str, Any]:
        """Hyperparameters of the selected learner (L1 penalty for logistic regression)."""
        if self.learner == "logistic_regression":
            return {"penalty": "l1", "solver": "liblinear", **self.params}
        return dict(self.params)


def aggregate_windows(conn: pd.DataFrame, *, ports: tuple[int, ...], window_seconds: float, internal_network: str,
                      scenario_column: str) -> pd.DataFrame:
    """Window feature table: one row per (scenario, host, window) with the features of the module docstring.

    Returns the features plus the key columns `scenario`, `host`, `window`, `window_start` and the
    label helper columns `n_botnet` (connections with a botnet label) and `n_conn` (all connections).
    A sum or a distinct count over an empty set of connections is 0; a sum over connections whose values
    are all unknown is NaN (repaired at training time), so absence of traffic and absence of a
    measurement stay distinct.
    """
    net = ipaddress.ip_network(internal_network)
    c = conn.copy()
    c["scenario"] = c[scenario_column].astype(str) if scenario_column in c.columns else "0"
    t0 = c.groupby("scenario")["time"].transform("min")
    c["window"] = np.floor((c["time"] - t0) / window_seconds).astype(np.int64)
    c["is_bot"] = np.asarray([ctu13_label_class(v) == 1 for v in c["label"].tolist()], dtype=np.int64)
    addresses = pd.unique(pd.concat([c["src"], c["dst"]], ignore_index=True))
    inside = {a: (ipaddress.ip_address(a) in net) if _is_ip(a) else False for a in addresses}
    c["src_in"] = c["src"].map(inside).astype(bool)
    c["dst_in"] = c["dst"].map(inside).astype(bool)
    c["unanswered"] = (c["resp_bytes"].fillna(0.0) <= 0).astype(np.int64)
    port = c["dport"].to_numpy(dtype=np.float64)
    port_class = np.full(len(c), "other", dtype=object)
    for p in ports:
        port_class[port == p] = str(p)
    c["port_class"] = port_class
    proto = c["proto"].astype(str).str.lower()
    c["proto_class"] = np.where(proto.isin(PROTOCOLS), proto, "other")

    keys = ["scenario", "host", "window"]
    out_conn = c.loc[c["src_in"]].rename(columns={"src": "host"}).sort_values([*keys, "time"], kind="stable")
    g = out_conn.groupby(keys, sort=True)
    feats = pd.DataFrame({
        "out_count": g.size(),
        "out_bytes_sent": g["orig_bytes"].sum(min_count=1),
        "out_bytes_recv": g["resp_bytes"].sum(min_count=1),
        "out_pkts": g["tot_pkts"].sum(min_count=1),
        "out_distinct_dst": g["dst"].nunique(),
        "out_distinct_dport": g["dport"].nunique(),
        "out_duration_mean": g["duration"].mean(),
        "out_duration_std": g["duration"].std(ddof=0),
        "out_duration_max": g["duration"].max(),
        "out_unanswered_share": g["unanswered"].mean(),
        "time_min": g["time"].min(),
        "n_botnet_out": g["is_bot"].sum(),
    })
    # Inter-arrival times within each window (the first connection of a window has none).
    gaps = g["time"].diff()
    gk = gaps.groupby([out_conn[k] for k in keys])
    iat = pd.DataFrame({"iat_mean": gk.mean(), "iat_std": gk.std(ddof=0), "iat_min": gk.min(), "iat_max": gk.max()})
    feats = feats.join(iat)
    for pc in [*map(str, ports), "other"]:
        sub = out_conn.loc[out_conn["port_class"] == pc].groupby(keys)
        present = feats.index.isin(sub.size().index)
        block = pd.DataFrame({
            f"port_{pc}_count": sub.size(),
            f"port_{pc}_bytes_sent": sub["orig_bytes"].sum(min_count=1),
            f"port_{pc}_bytes_recv": sub["resp_bytes"].sum(min_count=1),
            f"port_{pc}_distinct_dst": sub["dst"].nunique(),
        }).reindex(feats.index)
        block.loc[~present, :] = 0.0                       # no connection on this port class: empty sums
        feats = feats.join(block)
    for pr in (*PROTOCOLS, "other"):
        feats[f"proto_{pr}_count"] = out_conn.loc[out_conn["proto_class"] == pr].groupby(keys).size().reindex(feats.index).fillna(0.0)

    in_conn = c.loc[c["dst_in"]].rename(columns={"dst": "host"})
    gi = in_conn.groupby(keys, sort=True)
    incoming = pd.DataFrame({"in_count": gi.size(), "time_min_in": gi["time"].min(), "n_botnet_in": gi["is_bot"].sum()})
    table = feats.join(incoming, how="outer")
    no_out = table["out_count"].isna()
    # Windows with only incoming connections: every outgoing sum and count is over an empty set.
    empty_sum_cols = [col for col in table.columns if col.endswith(("_count", "_bytes_sent", "_bytes_recv", "_distinct_dst"))
                      or col in ("out_pkts", "out_distinct_dport")]
    table.loc[no_out, empty_sum_cols] = 0.0
    table[["in_count", "n_botnet_in", "n_botnet_out"]] = table[["in_count", "n_botnet_in", "n_botnet_out"]].fillna(0.0)
    table["window_start"] = table[["time_min", "time_min_in"]].min(axis=1)
    table["n_botnet"] = table["n_botnet_out"] + table["n_botnet_in"]
    table["n_conn"] = table["out_count"] + table["in_count"]
    table = table.drop(columns=["time_min", "time_min_in", "n_botnet_out", "n_botnet_in"]).reset_index()
    return table


def _is_ip(text: object) -> bool:
    try:
        ipaddress.ip_address(str(text))
    except ValueError:
        return False
    return True


def window_feature_names(ports: tuple[int, ...]) -> tuple[str, ...]:
    """Feature columns of `aggregate_windows` for a port set, in a fixed order."""
    names = ["out_count", "out_bytes_sent", "out_bytes_recv", "out_pkts", "out_distinct_dst", "out_distinct_dport",
             "out_duration_mean", "out_duration_std", "out_duration_max", "out_unanswered_share", "in_count",
             "iat_mean", "iat_std", "iat_min", "iat_max"]
    for pc in [*map(str, ports), "other"]:
        names += [f"port_{pc}_count", f"port_{pc}_bytes_sent", f"port_{pc}_bytes_recv", f"port_{pc}_distinct_dst"]
    names += [f"proto_{pr}_count" for pr in (*PROTOCOLS, "other")]
    return tuple(names)


REFERENCE = Reference(
    key="bl-ongun2019botnet",
    authors="Ongun, Sakharaov, Boboila, Oprea, Eliassi-Rad",
    title="On Designing Machine Learning Models for Malicious Network Traffic Classification",
    venue="arXiv preprint",
    year=2019,
    arxiv="1907.04846",
    note="preprint, not peer reviewed; author spelling as printed",
)


def _rows() -> tuple[ReportedResult, ...]:
    table = {
        "logistic_regression": {"2,9->1": ("0.90", "0.90", "0.90", "0.94"), "1,9->2": ("0.98", "0.95", "0.97", "0.99"),
                                "1,2->9": ("0.97", "0.87", "0.92", "0.96")},
        "random_forest": {"2,9->1": ("0.99", "0.97", "0.98", "0.99"), "1,9->2": ("0.95", "0.96", "0.95", "0.98"),
                          "1,2->9": ("1", "0.90", "0.94", "0.96")},
        "gradient_boosting": {"2,9->1": ("1", "0.97", "0.98", "0.99"), "1,9->2": ("1", "0.92", "0.96", "0.99"),
                              "1,2->9": ("1", "0.87", "0.93", "0.95")},
    }
    out = []
    for learner, rows in table.items():
        for split, (p, r, f1, auc) in rows.items():
            out.append(ReportedResult(dataset=f"ctu13-neris {split}", protocol="train on two scenarios, test on the third; 30 s host windows",
                                      task="binary", model=learner, values={"precision": p, "recall": r, "f1": f1, "auc": auc},
                                      location="Tab. 7, p. 6 (PDF page)", variant={"learner": learner}))
    return tuple(out)


SPEC = BaselineSpec(
    name="ongun-ctu13",
    title="Host-window classifiers of Ongun et al. on CTU-13",
    reference=REFERENCE,
    family="window-classifier",
    input_schema=InputSchema(
        description="One row per CTU-13 connection: Argus binetflow columns, Zeek conn.log columns or the normalised "
                    "connection schema (features.ctu13), with a `scenario` column naming the capture.",
        optional=("scenario", "Label", "label"),
    ),
    outputs=("detection",),
    datasets=("ctu13",),
    reported=_rows(),
    third_party="ongun-ctu13",
    assumptions=("AS-530", "AS-532", "AS-533", "AS-539"),
)


class OngunCTU13(PublishedBaseline):
    """Ongun et al. 2019: 30-second host windows, LR (L1), RF or GB, leave-one-scenario-out."""

    spec: ClassVar[BaselineSpec] = SPEC
    config_type: ClassVar[type[BaselineConfig]] = OngunConfig
    config: OngunConfig

    def __init__(self, config: BaselineConfig | None = None) -> None:
        super().__init__(config)
        self.ports_: tuple[int, ...] = ()
        self.pipeline: ColumnPipeline | None = None
        self.model: TabularModel | None = None

    def prepare(self, frame: pd.DataFrame) -> pd.DataFrame:
        return ctu13.connections(frame)

    def _infected(self, windows: pd.DataFrame) -> set[str]:
        if self.config.infected:
            return set(self.config.infected)
        hosts: set[str] = set()
        for s in pd.unique(windows["scenario"]):
            try:
                hosts |= set(ctu13.infected(int(s)))
            except (KeyError, ValueError) as exc:
                raise InvariantViolation(f"{self.spec.name}: scenario {s!r} has no infected-host list; set config.infected") from exc
        return hosts

    def _labels(self, windows: pd.DataFrame) -> np.ndarray:
        if self.config.label_mode == "fine":
            return (windows["n_botnet"].to_numpy() > 0).astype(np.int64)
        infected = self._infected(windows)
        return windows["host"].isin(infected).to_numpy().astype(np.int64)

    def _windows(self, conn: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config
        return aggregate_windows(conn, ports=self.ports_, window_seconds=cfg.window_seconds,
                                 internal_network=cfg.internal_network, scenario_column=cfg.scenario_column)

    def _fit(self, data: pd.DataFrame, validation: pd.DataFrame | None, rng: np.random.Generator) -> None:
        cfg = self.config
        # Port set: the most frequent destination ports of the training connections (ties by port number).
        counts = data["dport"].dropna().astype(np.int64).value_counts()
        ranked = sorted(counts.items(), key=lambda kv: (-int(kv[1]), int(kv[0])))
        self.ports_ = tuple(int(p) for p, _ in ranked[: cfg.top_ports])
        windows = self._windows(data)
        y = self._labels(windows)
        if np.unique(y).size < 2:
            raise InvariantViolation(f"{self.spec.name}: training windows contain a single class")
        feats = window_feature_names(self.ports_)
        scaler: Literal["none", "standard"] = "standard" if cfg.learner == "logistic_regression" else "none"
        pipeline = ColumnPipeline(feats, (), scaler).fit(windows)
        model = TabularModel(cfg.learner, cfg.learner_params(), cfg.seed, 2, needed_by=self.spec.name)
        model.fit(pipeline.transform(windows), y)
        self.pipeline, self.model = pipeline, model
        self.fit_report.update({"connections": len(data), "windows": len(windows), "malicious_windows": int(y.sum()),
                                "ports": list(self.ports_)})

    def _predict(self, data: pd.DataFrame) -> PredictionParts:
        assert self.pipeline is not None and self.model is not None
        windows = self._windows(data)
        score = self.model.predict_proba(self.pipeline.transform(windows))[:, 1]
        try:
            label = self._labels(windows)
        except InvariantViolation:
            label = np.full(len(windows), -1, dtype=np.int64)
        meta = make_meta(len(windows), time=windows["window_start"].to_numpy(dtype=np.float64), dataset="ctu13",
                         network=windows["scenario"].astype(str).to_numpy(),
                         family=np.where(label == 1, "botnet", np.where(label == 0, "benign", "unknown")),
                         host=windows["host"].astype(str).to_numpy(), window=windows["window"].to_numpy(dtype=np.int64))
        det = DetectionPredictions(score=score, label=label, unit="window", meta=meta, threshold=self.config.threshold)
        return PredictionParts(detection=det)

    def _export_state(self, directory: Path) -> dict[str, Any]:
        assert self.pipeline is not None and self.model is not None
        return {"ports": list(self.ports_), "pipeline": self.pipeline.state(), "model": self.model.save(directory, "model")}

    def _import_state(self, directory: Path, state: Mapping[str, Any]) -> None:
        self.ports_ = tuple(int(p) for p in state["ports"])
        self.pipeline = ColumnPipeline.from_state(state["pipeline"])
        self.model = TabularModel.load(directory, state["model"], needed_by=self.spec.name)

    @classmethod
    def paper_protocol(cls) -> SplitProtocol:
        """Leave one scenario out (train on the other two Neris scenarios)."""
        return GroupHoldout("scenario")
