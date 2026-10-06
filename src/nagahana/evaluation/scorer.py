"""One scoring path for every model: protocol runs with intervals, seeds, unit counts, paired tests and FDR.

The scorer receives ModelOutputs bundles (load_outputs) of NagaHana, the logistic-regression family and
the reproduced published baselines, groups them into runs (one model on one protocol variant, over all
its seeds) and scores every run with the same code. Persistence and climatology forecasts are built by
the evaluation itself; ablated variants of NagaHana (ModelOutputs.config["ablation"]) are scored like any
other run and compared with the full model.

Cells. Within a protocol variant, the evaluation units are split into cells: report group (dataset, or
the OT testbeds together) by novelty group (known and novel families apart, never pooled, D-23) by split,
plus the pooled group "all" of each novelty group, and, for forecasts, by horizon. The cells are defined
on the reference record of the variant (the primary model's, else the first run's), and every other run
is scored on exactly those units, paired through their metadata (dataset, network, entity, time, split,
family, novelty). A run whose records do not contain the reference units is scored on its own units
without paired comparisons, and the notes say so.

Every metric row carries

    value        the mean over seeds of the metric on the cell's units
    ci_low/high  the bootstrap interval of that seed mean (resampling.py; the same resamples of the units
                 are used for every seed, so the interval reflects the sampling of the evaluation units)
    ci_method    the interval method and the resampling scheme (stationary block bootstrap per dataset and
                 network series with the Politis-White block length of the metric's per-unit loss series,
                 a cluster bootstrap over attack episodes, or an iid bootstrap, per task configuration)
    n_seeds      the number of seeds, and seed_sd the standard deviation of the metric over seeds
    n_units      the number of evaluated units of the cell, and n_events its number of events (positives,
                 infiltrations within the horizon, observed survival events, scored state cells, episodes,
                 annotated narrative steps)

Comparisons. In every cell, NagaHana is compared with each other model of the same variant (difference
NagaHana - model) and every ablation with the full model (difference ablation - full). The difference
of the seed means carries its paired bootstrap interval, and the paired test of the thesis is run per
matched seed: McNemar on paired decisions (overall for F1, accuracy and the detection-error rate, on the
positives for recall and FNR, on the negatives for FPR and specificity), DeLong for AUROC,
Diebold-Mariano for per-unit proper scores and state errors over time, Wilcoxon's signed-rank test for
paired lead times of attack episodes. The comparison's p-value is the largest per-seed p-value
(significance.intersection_union_p). Metrics without a named test use the paired percentile-bootstrap
test, p = min(1, 2 (1 + #{replicates on the far side of 0}) / (B + 1)) (Davison and Hinkley, Bootstrap
Methods and their Application, Cambridge 1997, section 4.4). All p-values of a protocol run form one
family, adjusted with the configured procedure (Benjamini-Hochberg at level 0.05 by default).

Families. The comparisons of a protocol run fall into families that are adjusted separately: NagaHana
against the other models ("baselines"), ablations against the full model ("ablations"), regimes and
corruptions against the full-telemetry run ("degradation"), arena agents ("arena") and explanation
methods ("explanations"). Pre-registered hypotheses form their own confirmatory family (registry.py).

Degradation under reduced observability (P4) and corrupted telemetry (P5) is signalled when, against the
full-telemetry (or clean) run of the same model on the same triggers, the uncertainty band at the full
horizon is wider and the safe horizon shorter, each by a one-sided Wilcoxon signed-rank test that stays
significant after the adjustment of its family.

Pre-registration. `Scorer.run` refuses a protocol without a registration of that protocol, reports the
deviations of the run from its registration and tests the registered hypotheses (registry.py).

Further protocols: tiered ablations (P-ABL) are scored on the units of their base protocol; arena runs
(P-CW) are summarised by arena.py; explanation curves by faithfulness.py; the compute and latency profile
comes from the stage timings, operation counts and peak memory of the operations records.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from nagahana.core.errors import InvariantViolation, ProposalNotEnabled
from nagahana.evaluation import arena as arn
from nagahana.evaluation import calibration as cal
from nagahana.evaluation import components as comp
from nagahana.evaluation import episodes as epi
from nagahana.evaluation import faithfulness as fth
from nagahana.evaluation import forecasting as fc
from nagahana.evaluation import forensics as fz
from nagahana.evaluation import operations as ops
from nagahana.evaluation import paths as pth
from nagahana.evaluation import ranking as rk
from nagahana.evaluation import registry as regm
from nagahana.evaluation import significance as sig
from nagahana.evaluation import stages as stg
from nagahana.evaluation import state as sta
from nagahana.evaluation import survival as srv
from nagahana.evaluation._arrays import safe_ratio
from nagahana.evaluation.config import EvaluationConfig, to_dict
from nagahana.evaluation.generalisation import (
    check_not_pooled,
    novelty_masks,
    report_groups,
    training_networks,
    variant_of,
    variant_value,
)
from nagahana.evaluation.metrics import Counts, binary_report, rates
from nagahana.evaluation.predictions import (
    ArenaRun,
    DetectionPredictions,
    EpisodeTable,
    ExplanationCurves,
    ForecastPredictions,
    ForensicPredictions,
    ModelOutputs,
    PathPredictions,
    StagePredictions,
    StateForecastPredictions,
    TimeToEventPredictions,
)
from nagahana.evaluation.protocols import get_protocol
from nagahana.evaluation.registry import Deviation, Registered
from nagahana.evaluation.resampling import (
    BootstrapSettings,
    Scheme,
    cluster_scheme,
    estimate,
    iid_scheme,
    stationary_scheme,
    stratified_scheme,
)

#: A statistic of the evaluation: weights [b, n] -> values [b, q].
Statistic = Callable[[np.ndarray], np.ndarray]

METRIC_COLUMNS: tuple[str, ...] = (
    "protocol", "table", "task", "model", "role", "variant", "group", "novelty", "split", "horizon", "method",
    "condition", "metric", "value", "ci_low", "ci_high", "ci_method", "confidence", "n_resamples", "n_seeds", "seed_sd",
    "n_units", "n_events", "unit", "note")
COMPARISON_COLUMNS: tuple[str, ...] = (
    "protocol", "family", "task", "model_a", "model_b", "variant", "group", "novelty", "split", "horizon", "metric",
    "value_a", "value_b", "difference", "ci_low", "ci_high", "ci_method", "test", "statistic", "p_value", "p_adjusted",
    "significant", "p_values_by_seed", "effect", "n_units", "n_seeds_a", "n_seeds_b")
HYPOTHESIS_COLUMNS: tuple[str, ...] = (
    "id", "statement", "task", "metric", "model_a", "model_b", "direction", "margin", "difference", "ci_low", "ci_high",
    "test", "p_two_sided", "p_directional", "p_adjusted", "supported", "verdict", "note")
KEY_COLUMNS: tuple[str, ...] = ("dataset", "network", "entity", "time", "split", "family", "novelty")
ALL = "all"


@dataclass
class Run:
    """One model on one protocol variant, over all its seeds."""

    key: str
    model: str
    role: str
    variant: str
    seeds: list[ModelOutputs]
    sources: list[str] = field(default_factory=list)

    @property
    def n_seeds(self) -> int:
        """Number of seeds of the run."""
        return len(self.seeds)

    def records(self, task: str) -> list[Any]:
        """The records of one task over the seeds (empty unless every seed carries it)."""
        recs = [getattr(b, task) for b in self.seeds]
        return recs if recs and all(r is not None for r in recs) else []


@dataclass
class ProtocolResult:
    """Everything one protocol run produced."""

    protocol: str
    metrics: pd.DataFrame
    comparisons: pd.DataFrame
    figures: dict[str, pd.DataFrame]
    notes: list[str]
    inputs: pd.DataFrame
    config: dict[str, Any]
    registration: dict[str, Any] = field(default_factory=dict)
    deviations: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=["severity", "kind", "detail"]))
    hypotheses: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=list(HYPOTHESIS_COLUMNS)))

    def select(self, **filters: Any) -> pd.DataFrame:
        """Metric rows matching every column=value filter."""
        out = self.metrics
        for k, v in filters.items():
            out = out[out[k] == v]
        return out


@dataclass
class Cell:
    """Evaluation units of one reference record: report group, novelty group, split and horizon."""

    group: str
    novelty: str
    split: str
    horizon: str
    index: np.ndarray
    meta: pd.DataFrame


@dataclass
class CellStats:
    """What one run contributes to one cell: statistics per seed, loss series, counts and test inputs."""

    names: list[str]
    stats: list[Statistic]
    loss: np.ndarray
    counts: dict[str, float]
    tests: dict[str, tuple[str, list[Any]]] = field(default_factory=dict)
    unit: str = "unit"
    truth: np.ndarray | None = None
    clusters: np.ndarray | None = None
    note: str = ""


def _key_frame(meta: pd.DataFrame) -> pd.DataFrame:
    # Normalised unit keys: strings as str, time as float, entity as int.
    data = {}
    for c in KEY_COLUMNS:
        if c == "time":
            data[c] = meta[c].to_numpy(dtype=np.float64)
        elif c == "entity":
            data[c] = meta[c].to_numpy(dtype=np.int64)
        else:
            data[c] = meta[c].astype(str).to_numpy()
    return pd.DataFrame(data)


def align_units(ref: pd.DataFrame, other: pd.DataFrame) -> np.ndarray:
    """Index into `other` of every unit of `ref` (raises when the evaluation units differ)."""
    kr, ko = _key_frame(ref), _key_frame(other)
    if len(kr) == len(ko) and all(np.array_equal(kr[c].to_numpy(), ko[c].to_numpy()) for c in KEY_COLUMNS):
        return np.arange(len(kr))
    if ko.duplicated().any():
        raise InvariantViolation("units cannot be paired: the metadata keys of a record are not unique")
    merged = kr.reset_index().merge(ko.reset_index(), on=list(KEY_COLUMNS), how="left", suffixes=("_r", "_o"))
    if merged["index_o"].isna().any():
        raise InvariantViolation(f"units cannot be paired: {int(merged['index_o'].isna().sum())} units of one record "
                                 "are missing from the other")
    return merged.sort_values("index_r")["index_o"].to_numpy(dtype=np.int64)


def _seed_mean(stack: np.ndarray) -> np.ndarray:
    # Mean over the seed axis 0 of [S, b, q], ignoring undefined (NaN) values. Infinities are values (a
    # median lead time of -inf when most episodes were missed) and propagate; an all-NaN entry stays NaN.
    defined = ~np.isnan(stack)
    with np.errstate(invalid="ignore"):
        total = np.where(defined, stack, 0.0).sum(axis=0)
        return safe_ratio(total, defined.sum(axis=0))


def _bootstrap_p(reps: np.ndarray) -> float:
    # Paired percentile-bootstrap test of a difference: two-sided, with the (1 + count) / (B + 1) correction.
    r = reps[~np.isnan(reps)]
    if r.size == 0:
        return math.nan
    far = min(int(np.sum(r <= 0)), int(np.sum(r >= 0)))
    return float(min(1.0, 2.0 * (1 + far) / (r.size + 1)))


def _columns(values: Sequence[Any], b: int) -> np.ndarray:
    # Stack metric values (each a float or an array [b]) into a matrix [b, q].
    return np.column_stack([np.broadcast_to(np.asarray(v, dtype=np.float64).reshape(-1), (b,)) for v in values])


class _Session:
    """State of one protocol run: configuration, random numbers, rows, comparisons, figures and notes."""

    def __init__(self, cfg: EvaluationConfig, protocol: str) -> None:
        self.cfg = cfg
        self.protocol = protocol
        r = cfg.resampling
        self.boot = BootstrapSettings(n_resamples=r.n_resamples, confidence=r.confidence, interval=r.interval,
                                      jackknife_groups=r.jackknife_groups, chunk_elements=r.chunk_elements)
        self.rng = np.random.default_rng(cfg.seed)
        self.rows: list[dict[str, Any]] = []
        self.comparisons: list[dict[str, Any]] = []
        self.figures: dict[str, list[pd.DataFrame]] = {}
        self.notes: list[str] = []
        self.splits: tuple[str, ...] = ()
        self.train_nets: set[str] = set()
        self.episodes: dict[str, EpisodeTable] = {}
        self.horizon_seconds: float | None = None
        self.pending_flags: list[dict[str, Any]] = []
        self.cache: dict[tuple[int, str], Any] = {}

    def note(self, text: str) -> None:
        """Record a note once (missing references, fallbacks, skipped runs)."""
        if text not in self.notes:
            self.notes.append(text)

    def figure(self, name: str, frame: pd.DataFrame) -> None:
        """Append rows to the data of one figure."""
        self.figures.setdefault(name, []).append(frame)

    def scheme(self, kind: str, meta: pd.DataFrame, loss: np.ndarray, *, strata: np.ndarray | None = None,
               clusters: np.ndarray | None = None, what: str = "") -> Scheme:
        """The resampling scheme of a cell (falls back to the stationary scheme, with a note, when the cluster
        labels a cluster scheme needs are missing)."""
        n = len(meta)
        if kind == "iid":
            return iid_scheme(n)
        if kind == "stratified":
            return stratified_scheme(strata if strata is not None else np.zeros(n))
        if kind == "cluster":
            if clusters is not None and len(np.unique(clusters)) >= 2:
                return cluster_scheme(clusters)
            self.note(f"{what}: fewer than two clusters (attack episodes) for a cluster bootstrap; the stationary "
                      "block bootstrap was used")
            kind = "stationary"
        if kind == "stationary":
            series = meta["dataset"].astype(str).to_numpy() + "\x1f" + meta["network"].astype(str).to_numpy()
            block = self.cfg.resampling.mean_block_length
            return stationary_scheme(series, meta["time"].to_numpy(dtype=np.float64),
                                     mean_block="auto" if block is None else block, loss=np.nan_to_num(loss))
        raise InvariantViolation(f"unknown resampling scheme {kind!r}")

    def estimate_rows(self, names: Sequence[str], seed_stats: Sequence[Statistic], scheme: Scheme, base: dict[str, Any],
                      counts_: dict[str, float]) -> dict[str, tuple[float, float, float]]:
        """Seed-mean estimates with intervals of the named statistics; appends one row per statistic."""
        def stat(w: np.ndarray) -> np.ndarray:
            return _seed_mean(np.stack([np.asarray(f(w), dtype=np.float64).reshape(w.shape[0], -1) for f in seed_stats]))

        est = estimate(stat, names, scheme, self.boot, self.rng)
        per_seed = np.stack([np.asarray(f(np.ones((1, scheme.n))), dtype=np.float64).reshape(1, -1)[0] for f in seed_stats])
        out: dict[str, tuple[float, float, float]] = {}
        for i, name in enumerate(names):
            vals = per_seed[:, i]
            fin = vals[np.isfinite(vals)]
            self.rows.append({**base, "metric": name, "value": float(est.value[i]), "ci_low": float(est.low[i]),
                              "ci_high": float(est.high[i]), "ci_method": f"{est.method[i]} ({scheme.kind})",
                              "confidence": self.boot.confidence, "n_resamples": self.boot.n_resamples,
                              "n_seeds": len(seed_stats), "seed_sd": float(fin.std(ddof=1)) if fin.size > 1 else math.nan,
                              **counts_})
            out[name] = (float(est.value[i]), float(est.low[i]), float(est.high[i]))
        return out

    def plain_row(self, base: dict[str, Any], metric: str, value: float, low: float, high: float, method: str,
                  n_seeds: int, counts_: dict[str, float], seed_sd: float = math.nan, n_resamples: int = 0) -> None:
        """A metric row whose interval was computed by a method of its own (operations, invariants, audits)."""
        self.rows.append({**base, "metric": metric, "value": value, "ci_low": low, "ci_high": high, "ci_method": method,
                          "confidence": self.boot.confidence, "n_resamples": n_resamples, "n_seeds": n_seeds,
                          "seed_sd": seed_sd, **counts_})

    def compare(self, names: Sequence[str], stats_a: Sequence[Statistic], stats_b: Sequence[Statistic], scheme: Scheme,
                base: dict[str, Any], tests: dict[str, tuple[str, list[float], float, float]], n_units: int) -> None:
        """Paired differences A - B of the seed means with intervals; one comparison row per statistic.

        tests: metric -> (test name, per-seed p-values, statistic, effect) for metrics with a named test;
        the other metrics get the paired bootstrap test.
        """
        q = len(names)

        def diff(w: np.ndarray) -> np.ndarray:
            a = _seed_mean(np.stack([np.asarray(f(w), dtype=np.float64).reshape(w.shape[0], -1) for f in stats_a]))
            b = _seed_mean(np.stack([np.asarray(f(w), dtype=np.float64).reshape(w.shape[0], -1) for f in stats_b]))
            with np.errstate(invalid="ignore"):
                d = a - b                                               # inf - inf is undefined (NaN)
            return np.hstack([d, a, b])

        labels = [*(f"d:{n}" for n in names), *(f"a:{n}" for n in names), *(f"b:{n}" for n in names)]
        est = estimate(diff, labels, scheme, self.boot, self.rng)
        for i, name in enumerate(names):
            if name in tests:
                test, pvals, statistic, effect = tests[name]
                p = sig.intersection_union_p(pvals)
            else:
                test, pvals, statistic, effect = "paired_bootstrap", [], math.nan, math.nan
                p = _bootstrap_p(est.replicates[:, i])
            self.comparisons.append({**base, "metric": name, "value_a": float(est.value[q + i]),
                                     "value_b": float(est.value[2 * q + i]), "difference": float(est.value[i]),
                                     "ci_low": float(est.low[i]), "ci_high": float(est.high[i]),
                                     "ci_method": f"{est.method[i]} ({scheme.kind})", "test": test,
                                     "statistic": statistic, "p_value": p,
                                     "p_values_by_seed": ";".join(f"{x:.6g}" for x in pvals), "effect": effect,
                                     "n_units": n_units, "p_adjusted": math.nan, "significant": False})


class Scorer:
    """Scores ModelOutputs bundles under protocols P1 to P8 (module docstring)."""

    def __init__(self, config: EvaluationConfig | None = None) -> None:
        self.cfg = config if config is not None else EvaluationConfig()

    def runs(self, bundles: Sequence[ModelOutputs], protocol: str, sources: Sequence[str] | None = None) -> list[Run]:
        """Group bundles into runs (model and variant): primary first, then the LR family, others, ablations."""
        cfg = self.cfg
        groups: dict[tuple[str, str], Run] = {}
        for i, b in enumerate(bundles):
            if b.protocol.upper() != protocol:
                raise InvariantViolation(f"bundle {b.model!r} (seed {b.seed}) belongs to protocol {b.protocol!r}, "
                                         f"not {protocol}")
            abl = str(b.config.get("ablation", "") or "")
            key = f"{b.model}[{abl}]" if abl else b.model
            variant = variant_of(b, protocol)
            if abl:
                role = "ablation"
            elif b.model == cfg.models.primary:
                role = "primary"
            elif b.model in cfg.models.lr_family:
                role = "lr"
            else:
                role = "published"
            run = groups.setdefault((key, variant), Run(key, b.model, role, variant, []))
            if any(s.seed == b.seed for s in run.seeds):
                raise InvariantViolation(f"two bundles of {key!r} ({variant or 'no variant'}) share seed {b.seed}")
            run.seeds.append(b)
            run.sources.append(str(sources[i]) if sources is not None else "")
        order = {"primary": 0, "lr": 1, "published": 2, "ablation": 3}
        out = sorted(groups.values(), key=lambda r: (order[r.role], r.variant, r.key))
        for r in out:
            paired = sorted(zip(r.seeds, r.sources, strict=True), key=lambda t: t[0].seed)
            r.seeds = [s for s, _ in paired]
            r.sources = [src for _, src in paired]
        return out

    def run(self, protocol: str, bundles: Sequence[ModelOutputs], sources: Sequence[str] | None = None, *,
            registration: Registered | None) -> ProtocolResult:
        """Score every run of one pre-registered protocol: tables, comparisons, figures, deviations, hypotheses."""
        p = get_protocol(protocol)
        if registration is None:
            raise InvariantViolation(f"protocol {p.id} is not pre-registered: register it (python -m nagahana evaluate "
                                     "register) and pass the registration")
        if registration.registration.protocol.upper() != p.id:
            raise InvariantViolation(f"registration {registration.registration.id!r} is for protocol "
                                     f"{registration.registration.protocol}, not {p.id}")
        if regm.sha256(regm.to_plain(registration.registration)) != registration.entry.document_sha256:
            raise InvariantViolation(f"registration {registration.registration.id!r} does not match its registered hash")
        sess = _Session(self.cfg, p.id)
        runs = self.runs(bundles, p.id, sources)
        if not runs:
            raise InvariantViolation(f"no bundles for protocol {p.id}")
        deviations = _registration_deviations(sess, registration, runs)
        _score_protocol(sess, runs)
        comparisons = _adjust(sess)
        _degradation_flags(sess, comparisons)
        metrics = pd.DataFrame(sess.rows, columns=list(METRIC_COLUMNS))
        deviations += _primary_deviations(registration, metrics)
        hypotheses = _hypotheses(sess, registration, comparisons, deviations)
        figures = {k: pd.concat(v, ignore_index=True) for k, v in sess.figures.items()}
        inputs = pd.DataFrame([{"model": r.key, "role": r.role, "variant": r.variant, "seed": s.seed,
                                "tasks": ",".join(s.tasks()), "source": src,
                                "started_utc": str(s.config.get("started_utc", ""))}
                               for r in runs for s, src in zip(r.seeds, r.sources, strict=True)])
        e = registration.entry
        info = {"id": e.id, "protocol": e.protocol, "created_utc": e.created_utc, "document_sha256": e.document_sha256,
                "entry_sha256": e.entry_sha256, "title": registration.registration.title}
        dev = pd.DataFrame([{"severity": d.severity, "kind": d.kind, "detail": d.detail} for d in deviations],
                           columns=["severity", "kind", "detail"])
        return ProtocolResult(p.id, metrics, comparisons, figures, list(sess.notes), inputs, to_dict(self.cfg), info,
                              dev, hypotheses)


def score_zero_shot(outputs: ModelOutputs, baselines: Sequence[ModelOutputs] = (), config: EvaluationConfig | None = None,
                    *, registration: Registered | None) -> ProtocolResult:
    """Protocol P1 for one NagaHana bundle and its baselines through the same path as every other run."""
    return Scorer(config).run("P1", [outputs, *baselines], registration=registration)


def binary_group_report(p: Sequence[float], y: Sequence[float], *, threshold: float, bins: int = 10) -> dict[str, float]:
    """Decision metrics at threshold (alert when p >= threshold), Brier and equal-width ECE of one group.

    Keys: precision, recall, f1, fpr, fnr, detection_error (the rate (FP + FN) / N), base_rate, n, brier, ece;
    an empty group gives {"n": 0.0}.
    """
    if len(p) == 0:
        return {"n": 0.0}
    pp = np.asarray(p, dtype=np.float64)
    yy = (np.asarray(y, dtype=np.float64) > 0.5).astype(np.int64)
    r = binary_report(pp, yy, threshold)
    return {"precision": float(r["precision"]), "recall": float(r["recall"]), "f1": float(r["f1"]),
            "fpr": float(r["fpr"]), "fnr": float(r["fnr"]), "detection_error": float(r["detection_error_rate"]),
            "base_rate": float(r["base_rate"]), "n": float(pp.size), "brier": float(cal.brier_score(pp, yy)),
            "ece": float(cal.calibration_error(pp, yy, bins=bins))}


def _adjust(sess: _Session) -> pd.DataFrame:
    # The configured procedure (Benjamini-Hochberg by default) within each family of comparisons.
    frame = pd.DataFrame(sess.comparisons, columns=list(COMPARISON_COLUMNS))
    if len(frame):
        c = sess.cfg.comparisons
        adjusted = np.full(len(frame), np.nan)
        for fam in frame["family"].unique():
            m = (frame["family"] == fam).to_numpy()
            adjusted[m] = sig.adjust_pvalues(frame.loc[m, "p_value"].to_numpy(dtype=np.float64), c.fdr_method)
        frame["p_adjusted"] = adjusted
        frame["significant"] = adjusted <= c.fdr_level
    return frame


def _table(protocol: str, task: str, variant: str) -> str:
    # The thesis table a metric row feeds ("" for rows reported in the CSV and JSON only).
    if protocol == "P-ABL":
        return ""
    if task in ("arena", "explanations", "profile"):
        return {"arena": "res-arena", "explanations": "faithfulness", "profile": "res-profile"}[task]
    if task == "detection":
        if protocol == "P3":
            return "res-sitecal" if variant.endswith(";site") else "res-lono"
        return {"P1": "res-detection", "P2": "res-transfer", "P4": "res-robustness", "P5": "res-robustness"}.get(protocol, "")
    if task == "forecast" and protocol in ("P4", "P5"):
        return "res-robustness"
    return {"forecast": "res-forecast", "state_forecast": "res-state", "stage": "res-stages", "paths": "res-stages",
            "time_to_event": "res-timeliness", "timeliness": "res-timeliness", "component": "res-components",
            "forensics": "res-forensics", "operations": "res-operations", "audit": "information_audit"}.get(task, "")


def _base(sess: _Session, run: Run, task: str, cell: Cell, unit: str, table: str | None = None) -> dict[str, Any]:
    return {"protocol": sess.protocol, "table": _table(sess.protocol, task, run.variant) if table is None else table,
            "task": task, "model": run.key, "role": run.role, "variant": run.variant, "group": cell.group,
            "novelty": cell.novelty, "split": cell.split, "horizon": cell.horizon, "method": "", "condition": "",
            "unit": unit, "note": ""}


def _variants(runs: Sequence[Run]) -> list[str]:
    seen: list[str] = []
    for r in runs:
        if r.variant not in seen:
            seen.append(r.variant)
    return seen


def _cell(group: str, novelty: str, split: str, horizon: str, meta: pd.DataFrame, mask: np.ndarray) -> Cell:
    idx = np.flatnonzero(mask)
    return Cell(group, novelty, split, horizon, idx, meta.iloc[idx].reset_index(drop=True))


def _effective_protocol(sess: _Session, variant: str) -> str:
    """The protocol whose units a variant is scored on (the base protocol of a P-ABL variant)."""
    return variant_value(variant, "base") if sess.protocol == "P-ABL" else sess.protocol


def _splits_for(sess: _Session, variant: str) -> tuple[str, ...]:
    """Scored splits of a variant: P1 reads the configured zero-shot split, other protocols their own splits."""
    proto = _effective_protocol(sess, variant)
    return (sess.cfg.protocols.p1_split,) if proto == "P1" else get_protocol(proto).splits


def _unit_cells(sess: _Session, meta: pd.DataFrame, variant: str, base_mask: np.ndarray | None = None) -> list[Cell]:
    """Cells of report group x novelty group x split (and the pooled group per novelty group)."""
    groups = report_groups(meta, sess.cfg)
    held = variant_value(variant, "held_out") if _effective_protocol(sess, variant) == "P3" else ""
    net = meta["network"].astype(str).to_numpy()
    out: list[Cell] = []
    for split in _splits_for(sess, variant):
        for nov, nmask in novelty_masks(meta, split=split, novelty_rule=sess.cfg.protocols.p1_novelty,
                                        train_networks=sess.train_nets).items():
            check_not_pooled(meta, nmask)
            m = nmask if base_mask is None else nmask & base_mask
            if held:
                hm = m & (net == held)
                if hm.any():
                    out.append(_cell("held_out", nov, split, "", meta, hm))
                continue
            present = sorted(set(groups[m]))
            for g in present:
                out.append(_cell(g, nov, split, "", meta, m & (groups == g)))
            if len(present) > 1:
                out.append(_cell(ALL, nov, split, "", meta, m))
    return out


def _with_horizons(cells: list[Cell], horizons: Sequence[int]) -> list[Cell]:
    return [Cell(c.group, c.novelty, c.split, str(int(h)), c.index, c.meta) for c in cells for h in horizons]


def _clusters(sess: _Session, cell: Cell, variant: str) -> np.ndarray | None:
    """Episode of each unit (meta column "episode", else assigned from the episode table); others are singletons."""
    if "episode" in cell.meta.columns:
        lab = cell.meta["episode"].astype(str).to_numpy()
    else:
        eps = sess.episodes.get(variant)
        if eps is None or sess.horizon_seconds is None:
            return None
        lab = epi.assign_episodes(cell.meta, eps, sess.horizon_seconds)
    out = lab.astype(object)
    empty = np.flatnonzero(lab == "")
    out[empty] = [f"\x1funit{i}" for i in empty]
    return out.astype(str)


def _seed_pairs(sess: _Session, a: Run, b: Run) -> list[tuple[int, int]]:
    """Seed pairs for paired tests: matched seed ids, else seeds paired in order (noted)."""
    ids_a, ids_b = [s.seed for s in a.seeds], [s.seed for s in b.seeds]
    common = [s for s in ids_a if s in ids_b]
    if common:
        return [(ids_a.index(s), ids_b.index(s)) for s in common]
    sess.note(f"{sess.protocol}: {a.key} and {b.key} share no seed id; their seeds were paired in order")
    return [(i, i) for i in range(min(len(ids_a), len(ids_b)))]


def _run_test(sess: _Session, kind: str, pa: Any, pb: Any, cell: Cell) -> tuple[float, float, float]:
    """(p-value, statistic, effect) of one named paired test on one pair of seeds."""
    c = sess.cfg.comparisons
    if kind == "mcnemar":
        (ca, sub), (cb, _) = pa, pb
        if not np.any(sub):
            return math.nan, math.nan, math.nan
        r = sig.mcnemar_test(ca[sub].astype(np.int64), cb[sub].astype(np.int64), mid_p=c.mcnemar_mid_p,
                             confidence=sess.boot.confidence)
        return r.p_value, r.statistic, r.estimate
    if kind == "delong":
        (sa, y), (sb, _) = pa, pb
        if y.size == 0 or y.min() == y.max():
            return math.nan, math.nan, math.nan
        r = sig.delong_test(sa, sb, y, confidence=sess.boot.confidence)
        return r.p_value, r.statistic, r.estimate
    if kind == "dm":
        (la, h), (lb, _) = pa, pb
        ok = np.isfinite(la) & np.isfinite(lb)
        if ok.sum() < 3:
            return math.nan, math.nan, math.nan
        series = cell.meta["dataset"].astype(str).to_numpy() + "\x1f" + cell.meta["network"].astype(str).to_numpy()
        r = sig.diebold_mariano(la[ok], lb[ok], horizon=int(h), series=series[ok],
                                time=cell.meta["time"].to_numpy(dtype=np.float64)[ok], lags=c.dm_lags)
        return r.p_value, r.statistic, r.estimate
    if kind in ("wilcoxon", "wilcoxon_greater", "wilcoxon_less"):
        va, vb = np.asarray(pa, dtype=np.float64), np.asarray(pb, dtype=np.float64)
        both = np.r_[va, vb]
        fin = both[np.isfinite(both)]
        span = float(fin.max() - fin.min()) if fin.size else 0.0
        low = (float(fin.min()) - span - 1.0) if fin.size else -1.0
        high = (float(fin.max()) + span + 1.0) if fin.size else 1.0
        # An episode never alerted ranks below every alerted one: -inf becomes a value below all finite ones.
        xa = np.where(np.isneginf(va), low, np.where(np.isposinf(va), high, va))
        xb = np.where(np.isneginf(vb), low, np.where(np.isposinf(vb), high, vb))
        ok = ~(np.isnan(xa) | np.isnan(xb))
        if ok.sum() < 1:
            return math.nan, math.nan, math.nan
        alt = {"wilcoxon": "two-sided", "wilcoxon_greater": "greater", "wilcoxon_less": "less"}[kind]
        r = sig.wilcoxon_signed_rank(xa[ok], xb[ok], zero_method=c.wilcoxon_zero_method, alternative=alt)
        return r.p_value, r.statistic, r.detail.get("rank_biserial", math.nan)
    raise InvariantViolation(f"unknown paired test {kind!r}")


def _select(f: Statistic, cols: list[int]) -> Statistic:
    def g(w: np.ndarray) -> np.ndarray:
        return np.asarray(f(w), dtype=np.float64).reshape(w.shape[0], -1)[:, cols]
    return g


def _compare_pair(sess: _Session, task: str, cell: Cell, scheme_kind: str, a: tuple[Run, CellStats],
                  b: tuple[Run, CellStats]) -> None:
    """Paired comparison of run A against run B on one cell (difference A - B)."""
    (ra, ca), (rb, cb) = a, b
    if ca.truth is not None and cb.truth is not None and not np.array_equal(ca.truth, cb.truth):
        raise InvariantViolation(f"{sess.protocol} {task}: {ra.key} and {rb.key} disagree on the ground truth of the "
                                 "same units")
    names = [n for n in ca.names if n in cb.names]
    if not names:
        return
    ia, ib = [ca.names.index(n) for n in names], [cb.names.index(n) for n in names]
    stats_a = [_select(f, ia) for f in ca.stats]
    stats_b = [_select(f, ib) for f in cb.stats]
    loss = np.nan_to_num(ca.loss) - np.nan_to_num(cb.loss)
    what = f"{sess.protocol} {task} {ra.key} vs {rb.key} {cell.group}"
    scheme = sess.scheme(scheme_kind, cell.meta, loss, strata=ca.truth, clusters=ca.clusters, what=what)
    tests: dict[str, tuple[str, list[float], float, float]] = {}
    pairs = _seed_pairs(sess, ra, rb)
    for n in names:
        if n in ca.tests and n in cb.tests and ca.tests[n][0] == cb.tests[n][0]:
            kind = ca.tests[n][0]
            res = [_run_test(sess, kind, ca.tests[n][1][i], cb.tests[n][1][j], cell) for i, j in pairs]
            pvals = [r[0] for r in res]
            tests[n] = (kind, pvals, res[0][1] if res else math.nan, res[0][2] if res else math.nan)
    family = "ablations" if "ablation" in (ra.role, rb.role) else "baselines"
    base = {"protocol": sess.protocol, "family": family, "task": task, "model_a": ra.key, "model_b": rb.key,
            "variant": ra.variant, "group": cell.group, "novelty": cell.novelty, "split": cell.split,
            "horizon": cell.horizon, "n_seeds_a": ra.n_seeds, "n_seeds_b": rb.n_seeds}
    sess.compare(names, stats_a, stats_b, scheme, base, tests, int(cell.index.size))


Builder = Callable[[_Session, Run, list[Any], list[np.ndarray], Cell, Any], "CellStats | None"]
CellsFn = Callable[[_Session, Any, str], list[Cell]]


def _score_unit_task(sess: _Session, runs: list[Run], task: str, records: Callable[[Run], list[Any]], cells_fn: CellsFn,
                     builder: Builder, scheme_kind: str,
                     context: Callable[[_Session, list[Run], Any], Any] | None = None,
                     after: Callable[[_Session, list[Run], Any, list[Cell], Any, str], None] | None = None) -> None:
    """Score one task for every variant: rows per run and cell, then paired comparisons per cell."""
    for variant in _variants(runs):
        vruns = [r for r in runs if r.variant == variant and records(r)]
        if vruns:
            _score_unit_group(sess, vruns, task, records, cells_fn, builder, scheme_kind, context, after)


def _score_unit_group(sess: _Session, vruns: list[Run], task: str, records: Callable[[Run], list[Any]], cells_fn: CellsFn,
                      builder: Builder, scheme_kind: str, context: Any, after: Any) -> None:
    ref_rec = records(vruns[0])[0]
    variant = vruns[0].variant
    cells = cells_fn(sess, ref_rec, variant)
    if not cells:
        return
    ctx = context(sess, vruns, ref_rec) if context is not None else None
    # Pair every run's seeds with the units of all cells at once; runs that cannot be paired are scored alone.
    union = np.unique(np.concatenate([c.index for c in cells]))
    where = np.full(len(ref_rec.meta), -1, dtype=np.int64)
    where[union] = np.arange(union.size)
    union_meta = ref_rec.meta.iloc[union].reset_index(drop=True)
    maps: dict[str, list[np.ndarray]] = {}
    alone: list[Run] = []
    for run in vruns:
        try:
            maps[run.key] = [align_units(union_meta, rec.meta) for rec in records(run)]
        except InvariantViolation as exc:
            sess.note(f"{sess.protocol} {task}: {run.key} ({variant or 'no variant'}) is scored on its own units "
                      f"without paired comparisons ({exc})")
            alone.append(run)
    for cell in cells:
        built: dict[str, tuple[Run, CellStats]] = {}
        for run in vruns:
            if run.key not in maps:
                continue
            cmaps = [m[where[cell.index]] for m in maps[run.key]]
            try:
                cs = builder(sess, run, records(run), cmaps, cell, ctx)
            except InvariantViolation as exc:
                sess.note(f"{sess.protocol} {task}: {run.key} {cell.group}/{cell.novelty or cell.split}"
                          f"{'/' + cell.horizon if cell.horizon else ''} not scored ({exc})")
                continue
            if cs is None:
                continue
            what = f"{sess.protocol} {task} {run.key} {cell.group}"
            scheme = sess.scheme(scheme_kind, cell.meta, cs.loss, strata=cs.truth, clusters=cs.clusters, what=what)
            base = _base(sess, run, task, cell, cs.unit)
            base["note"] = cs.note
            sess.estimate_rows(cs.names, cs.stats, scheme, base, cs.counts)
            built[run.key] = (run, cs)
        _compare_cell(sess, task, cell, scheme_kind, built, vruns)
    if after is not None:
        after(sess, vruns, ref_rec, cells, ctx, variant)
    for run in alone:
        _score_unit_group(sess, [run], task, records, cells_fn, builder, scheme_kind, context, None)


def _compare_cell(sess: _Session, task: str, cell: Cell, scheme_kind: str, built: dict[str, tuple[Run, CellStats]],
                  vruns: list[Run]) -> None:
    primary = next((r for r in vruns if r.role == "primary"), None)
    if primary is None or primary.key not in built:
        return
    full = built[primary.key]
    for run in vruns:
        if run is primary or run.key not in built:
            continue
        if run.role == "ablation":
            _compare_pair(sess, task, cell, scheme_kind, built[run.key], full)     # ablated - full model
        else:
            _compare_pair(sess, task, cell, scheme_kind, full, built[run.key])     # NagaHana - other model


def _records(task: str) -> Callable[[Run], list[Any]]:
    def get(run: Run) -> list[Any]:
        return run.records(task)
    return get


def _conformal_thresholds(sess: _Session, meta: pd.DataFrame, score: np.ndarray, benign: np.ndarray, idx: np.ndarray,
                          what: str) -> np.ndarray:
    """Per-unit conformal thresholds at alpha from benign calibration units (per dataset or pooled)."""
    op = sess.cfg.operating_point
    cal_mask = (meta["split"].astype(str).to_numpy() == op.calibration_split) & benign
    if not cal_mask.any():
        raise InvariantViolation(f"{what}: no benign units in the calibration split {op.calibration_split!r} for the "
                                 "conformal threshold")
    ds = meta["dataset"].astype(str).to_numpy()
    pooled = rk.conformal_threshold(score[cal_mask], op.alpha)
    out = np.full(idx.size, pooled)
    if op.conformal_scope == "dataset":
        for d in np.unique(ds[idx]):
            m = cal_mask & (ds == d)
            if m.any():
                out[ds[idx] == d] = rk.conformal_threshold(score[m], op.alpha)
            else:
                sess.note(f"{what}: dataset {d} has no calibration units; the pooled conformal threshold was used")
    if not np.all(np.isfinite(out)):
        sess.note(f"{what}: too few benign calibration units for alpha = {op.alpha}; the affected units never alert")
    return out


def _detection_thresholds(sess: _Session, rec: DetectionPredictions, idx: np.ndarray, what: str) -> np.ndarray:
    """Per-unit decision thresholds of a detection record for the evaluated units idx."""
    op = sess.cfg.operating_point
    policy = op.detection
    if policy == "own":
        if rec.threshold is not None:
            return np.full(idx.size, float(rec.threshold))
        if op.missing_own == "error":
            raise InvariantViolation(f"{what}: the record has no own threshold")
        sess.note(f"{what}: no own threshold in the record; the conformal threshold at alpha was used")
        policy = "conformal"
    if policy == "conformal":
        return _conformal_thresholds(sess, rec.meta, rec.score, rec.label == 0, idx, what)
    point = rk.operating_point_at_fpr(rec.score[idx], np.clip(rec.label[idx], 0, 1), op.alpha)
    return np.full(idx.size, float(point.threshold))


def _detection_names(cfg: EvaluationConfig) -> list[str]:
    names = ["precision", "recall", "f1", "fpr", "fnr", "specificity", "balanced_accuracy", "mcc", "accuracy",
             "detection_error", "detection_error_rate", "base_rate", "alert_rate", "auroc", "auprc", "pauroc"]
    for a in cfg.detection.fixed_fprs:
        names += [f"recall_at_fpr_{a:g}", f"precision_at_fpr_{a:g}"]
    names += ["alerts_per_day", "false_alerts_per_day"]
    names += [f"precision_at_base_rate_{pi:g}" for pi in cfg.detection.deployment_base_rates]
    return names


def _detection_stat(score: np.ndarray, label: np.ndarray, thr: np.ndarray, days: float, cfg: EvaluationConfig) -> Statistic:
    # Confusion indicators and the descending-score groups are computed once; each resample is one weight row.
    dec = (score >= thr).astype(np.float64)
    yf = label.astype(np.float64)
    ind = np.column_stack([yf * dec, (1.0 - yf) * dec, (1.0 - yf) * (1.0 - dec), yf * (1.0 - dec)])   # [n, 4]
    groups = rk.grouped(score, label)
    d = cfg.detection

    def fn(w: np.ndarray) -> np.ndarray:
        cnt = w @ ind                                                   # [B, 4] = TP, FP, TN, FN
        c = Counts(cnt[:, 0], cnt[:, 1], cnt[:, 2], cnt[:, 3])
        r = rates(c)
        pos, neg = groups.class_weights(w)
        cols: list[Any] = [r[k] for k in ("precision", "recall", "f1", "fpr", "fnr", "specificity", "balanced_accuracy",
                                         "mcc", "accuracy", "detection_error", "detection_error_rate", "base_rate",
                                         "alert_rate")]
        cols += [rk.auroc_from_groups(pos, neg), rk.average_precision_from_groups(pos, neg),
                 rk.partial_auroc_from_groups(pos, neg, d.partial_auroc_max_fpr)]
        for a in d.fixed_fprs:
            _, tpr, _, prec = rk.operating_point_from_groups(pos, neg, groups.values, a)
            cols += [tpr, prec]
        cols += [rk.alerts_per_day(c.tp + c.fp, days), rk.alerts_per_day(c.fp, days)]
        cols += [rk.precision_at_base_rate(r["recall"], r["fpr"], pi) for pi in d.deployment_base_rates]
        return _columns(cols, w.shape[0])
    return fn


def _detection_cells(sess: _Session, rec: DetectionPredictions, variant: str) -> list[Cell]:
    return _unit_cells(sess, rec.meta, variant, base_mask=rec.scored)


def _build_detection(sess: _Session, run: Run, recs: list[DetectionPredictions], maps: list[np.ndarray], cell: Cell,
                     _ctx: Any) -> CellStats | None:
    what = f"{sess.protocol} {run.key} {run.variant} {cell.group}/{cell.novelty or cell.split}"
    label = recs[0].label[maps[0]]
    if np.any(label < 0):
        raise InvariantViolation(f"{what}: unscored units reached a detection cell")
    days = rk.observed_days(cell.meta)
    stats: list[Statistic] = []
    mc_all, mc_pos, mc_neg, dl = [], [], [], []
    loss = np.zeros(label.size)
    for s, (rec, idx) in enumerate(zip(recs, maps, strict=True)):
        lab = rec.label[idx]
        if not np.array_equal(lab, label):
            raise InvariantViolation(f"{what}: the seeds disagree on labels")
        thr = _detection_thresholds(sess, rec, idx, what)
        score = rec.score[idx]
        stats.append(_detection_stat(score, lab, thr, days, sess.cfg))
        correct = (score >= thr).astype(np.int64) == lab
        if s == 0:
            loss = (~correct).astype(np.float64)
        mc_all.append((correct, np.ones(lab.size, dtype=bool)))
        mc_pos.append((correct, lab == 1))
        mc_neg.append((correct, lab == 0))
        dl.append((score, lab))
    tests: dict[str, tuple[str, list[Any]]] = {
        "f1": ("mcnemar", mc_all), "accuracy": ("mcnemar", mc_all), "detection_error_rate": ("mcnemar", mc_all),
        "recall": ("mcnemar", mc_pos), "fnr": ("mcnemar", mc_pos), "fpr": ("mcnemar", mc_neg),
        "specificity": ("mcnemar", mc_neg), "auroc": ("delong", dl)}
    _detection_figures(sess, run, recs[0].score[maps[0]], label, cell)
    return CellStats(_detection_names(sess.cfg), stats, loss, {"n_units": float(label.size), "n_events": float(label.sum())},
                     tests, unit=recs[0].unit, truth=label, clusters=_clusters(sess, cell, run.variant))


def _detection_figures(sess: _Session, run: Run, score: np.ndarray, label: np.ndarray, cell: Cell) -> None:
    # ROC and PR curves of the first seed.
    if label.min() == label.max():
        return
    key = {"model": run.key, "variant": run.variant, "group": cell.group, "novelty": cell.novelty, "split": cell.split}
    fpr, tpr, thr = rk.roc_curve(score, label)
    sess.figure("roc", pd.DataFrame({**key, "fpr": fpr, "tpr": tpr, "threshold": thr}))
    prec, rec_, thr2 = rk.pr_curve(score, label)
    sess.figure("pr", pd.DataFrame({**key, "precision": prec, "recall": rec_, "threshold": thr2}))


class _References:
    """Persistence and climatology forecasts of a variant, built from labels (forecasting.py)."""

    def __init__(self, sess: _Session, records: list[ForecastPredictions]) -> None:
        self.horizon = records[0].horizon
        self.pers_src = next((r for r in records if r.infiltrated_now is not None and r.horizon == self.horizon), None)
        self.clim_src = next((r for r in records if r.horizon == self.horizon
                              and bool((r.meta["split"].astype(str) == "train").any())), None)
        self.pers = fc.persistence(self.pers_src) if self.pers_src is not None else None
        self.clim: np.ndarray | None = None
        if self.pers_src is None:
            sess.note(f"{sess.protocol}: persistence reference unavailable: no forecast record carries "
                      "infiltrated_now (the current state at the trigger)")
        if self.clim_src is None:
            sess.note(f"{sess.protocol}: climatology reference unavailable: no forecast record carries training "
                      "triggers (split 'train')")
        else:
            train = self.clim_src.meta["split"].astype(str).to_numpy() == "train"
            grp = (self.clim_src.meta["dataset"].astype(str).to_numpy()
                   if sess.cfg.forecast.climatology_scope == "dataset" else None)
            clim = fc.climatology(self.clim_src, train=train, group=grp)
            self.clim = clim.p
            for name, how in clim.source.items():
                sess.note(f"{sess.protocol}: climatology of {name} estimated from {how}")

    def for_cell(self, sess: _Session, cell: Cell) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Reference forecasts [n_c, K] on the units of a cell (None where unavailable)."""
        out: list[np.ndarray | None] = []
        for src, mat in ((self.pers_src, self.pers), (self.clim_src, self.clim)):
            if src is None or mat is None:
                out.append(None)
                continue
            key = (id(cell), "pers" if src is self.pers_src and mat is self.pers else "clim")
            if key not in sess.cache:
                try:
                    sess.cache[key] = mat[align_units(cell.meta, src.meta)]
                except InvariantViolation as exc:
                    sess.note(f"{sess.protocol}: a reference forecast could not be paired with {cell.group} ({exc})")
                    sess.cache[key] = None
            out.append(sess.cache[key])
        return out[0], out[1]


def _forecast_context(sess: _Session, vruns: list[Run], _ref: Any) -> _References:
    return _References(sess, [r.records("forecast")[0] for r in vruns])


def _forecast_names(with_ens: bool) -> list[str]:
    return ["brier", "log_score", "ece", "crps"] + (["crps_ensemble"] if with_ens else []) + [
        "bss_persistence", "bss_climatology", "crpss_persistence", "crpss_climatology"]


def _forecast_stat(cfg: EvaluationConfig, pm: np.ndarray, event_step: np.ndarray, y: np.ndarray, known: np.ndarray, k: int,
                   pers: np.ndarray | None, clim: np.ndarray | None, with_ens: bool, ens_u: np.ndarray | None) -> Statistic:
    """Proper scores, ECE and skill at horizon k of the forecast matrix pm [n, K] on units with outcomes y."""
    fcfg = cfg.forecast
    p = pm[:, k - 1]
    obs = ((event_step[:, None] > 0) & (event_step[:, None] <= np.arange(1, k + 1)[None, :])).astype(np.float64)
    yf = y.astype(np.float64)
    crps_u = ((pm[:, :k] - obs) ** 2).sum(axis=1)
    refs: list[tuple[np.ndarray, np.ndarray] | None] = []
    for ref in (pers, clim):
        refs.append(None if ref is None else ((ref[:, k - 1] - yf) ** 2, ((ref[:, :k] - obs) ** 2).sum(axis=1)))
    yi = y.astype(np.int64)

    def fn(w: np.ndarray) -> np.ndarray:
        wk = w * known[None, :]
        tot = wk.sum(axis=1)
        brier = safe_ratio(wk @ (p - yf) ** 2, tot)
        crps = safe_ratio(wk @ crps_u, tot)
        cols: list[Any] = [brier, cal.log_loss(p, yi, eps=fcfg.log_score_eps, weights=wk),
                           cal.calibration_error(p, yi, bins=fcfg.ece_bins, strategy=fcfg.ece_strategy, weights=wk), crps]
        if with_ens:
            if ens_u is None:
                cols.append(np.full(w.shape[0], np.nan))
            else:
                ok = np.isfinite(ens_u)
                cols.append(safe_ratio((wk * ok) @ np.where(ok, ens_u, 0.0), (wk * ok).sum(axis=1)))
        bss, crpss = [], []
        for r in refs:
            if r is None:
                bss.append(np.full(w.shape[0], np.nan))
                crpss.append(np.full(w.shape[0], np.nan))
            else:
                bss.append(1.0 - safe_ratio(brier, safe_ratio(wk @ r[0], tot)))
                crpss.append(1.0 - safe_ratio(crps, safe_ratio(wk @ r[1], tot)))
        cols += [bss[0], bss[1], crpss[0], crpss[1]]
        return _columns(cols, w.shape[0])
    return fn


def _forecast_cells(sess: _Session, rec: ForecastPredictions, variant: str) -> list[Cell]:
    return _with_horizons(_unit_cells(sess, rec.meta, variant), fc.reporting_horizons(sess.cfg.forecast.horizons, rec.horizon))


def _outcomes(rec: ForecastPredictions, idx: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    y, known = rec.outcome(k)
    return y[idx], known[idx]


def _build_forecast(sess: _Session, run: Run, recs: list[ForecastPredictions], maps: list[np.ndarray], cell: Cell,
                    refs: _References) -> CellStats | None:
    k = int(cell.horizon)
    if any(r.horizon != recs[0].horizon or r.window_seconds != recs[0].window_seconds for r in recs):
        raise InvariantViolation("the seeds forecast different horizons or window lengths")
    y, known = _outcomes(recs[0], maps[0], k)
    if not known.any():
        return None
    pers, clim = refs.for_cell(sess, cell) if refs is not None else (None, None)
    if refs is not None and refs.horizon != recs[0].horizon:
        pers, clim = None, None
    with_ens = all(r.ensemble is not None for r in recs)
    stats: list[Statistic] = []
    dm_b, dm_l, dm_c = [], [], []
    for rec, idx in zip(recs, maps, strict=True):
        yy, kk = _outcomes(rec, idx, k)
        if not (np.array_equal(yy, y) and np.array_equal(kk, known)):
            raise InvariantViolation("the seeds disagree on forecast outcomes")
        pm = rec.p_inf[idx]
        ens_u = fc.crps_ensemble(rec, k, fair=sess.cfg.forecast.fair_crps)[idx] if with_ens else None
        stats.append(_forecast_stat(sess.cfg, pm, rec.event_step[idx], y, known, k, pers, clim, with_ens, ens_u))
        p = pm[:, k - 1]
        dm_b.append((np.where(known, (p - y) ** 2, np.nan), k))
        with np.errstate(divide="ignore"):
            ll = -np.where(y > 0, np.log(p), np.log1p(-p))
        dm_l.append((np.where(known, ll, np.nan), k))
        dm_c.append((fc.crps_cumulative(rec, k)[idx], k))
    tests: dict[str, tuple[str, list[Any]]] = {"brier": ("dm", dm_b), "log_score": ("dm", dm_l), "crps": ("dm", dm_c)}
    loss = np.where(known, (recs[0].p_inf[maps[0], k - 1] - y) ** 2, 0.0)
    _forecast_figures(sess, run, recs, maps, cell, refs, k, y, known)
    return CellStats(_forecast_names(with_ens), stats, loss, {"n_units": float(known.sum()), "n_events": float(y[known].sum())},
                     tests, unit="trigger", truth=np.where(known, y, -1), clusters=_clusters(sess, cell, run.variant))


def _forecast_figures(sess: _Session, run: Run, recs: list[ForecastPredictions], maps: list[np.ndarray], cell: Cell,
                      refs: _References | None, k: int, y: np.ndarray, known: np.ndarray) -> None:
    # Reliability data at this horizon (first seed); skill by horizon over all steps once per cell (seed mean);
    # temperature effects of the primary model at the full horizon.
    rec0, idx0 = recs[0], maps[0]
    key = {"model": run.key, "variant": run.variant, "group": cell.group, "novelty": cell.novelty, "split": cell.split}
    rel = cal.reliability_table(rec0.p_inf[idx0, k - 1][known], y[known], bins=sess.cfg.forecast.reliability_bins,
                                confidence=sess.boot.confidence)
    sess.figure("reliability", rel.assign(**key, horizon=k))
    if k == rec0.horizon and run.role == "primary" and known.sum() > 1 and 0 < y[known].sum() < known.sum():
        temp = cal.temperature_effects(rec0.p_inf[idx0, k - 1][known], y[known], sess.cfg.forecast.temperatures,
                                       bins=sess.cfg.forecast.ece_bins)
        sess.figure("temperature_effects", temp.assign(**key, horizon=k))
    flag = (id(cell.meta), run.key, "skill")
    if flag in sess.cache:
        return
    sess.cache[flag] = True
    pers, clim = refs.for_cell(sess, cell) if refs is not None else (None, None)
    rows = []
    for kk in range(1, rec0.horizon + 1):
        yk, kn = _outcomes(rec0, idx0, kk)
        if not kn.any():
            continue
        vals = [_forecast_stat(sess.cfg, rec.p_inf[idx], rec.event_step[idx], yk, kn, kk, pers, clim, False, None)(
            np.ones((1, idx.size)))[0] for rec, idx in zip(recs, maps, strict=True)]
        v = _seed_mean(np.stack(vals)[:, None, :])[0]
        rows.append({**key, "horizon": kk, "brier": v[0], "crps": v[3], "bss_persistence": v[4], "bss_climatology": v[5],
                     "crpss_persistence": v[6], "crpss_climatology": v[7]})
    if rows:
        sess.figure("skill_by_horizon", pd.DataFrame(rows))


def _forecast_reference_rows(sess: _Session, vruns: list[Run], ref_rec: ForecastPredictions, cells: list[Cell],
                             refs: _References, variant: str) -> None:
    """Rows of the persistence and climatology forecasts on the cells of the variant's reference record."""
    if refs is None:
        return
    for cell in cells:
        k = int(cell.horizon)
        y, known = _outcomes(ref_rec, cell.index, k)
        if not known.any():
            continue
        pers, clim = refs.for_cell(sess, cell)
        for name, pm in (("persistence", pers), ("climatology", clim)):
            if pm is None:
                continue
            run = Run(name, name, "reference", variant, [])
            stat = _forecast_stat(sess.cfg, pm, ref_rec.event_step[cell.index], y, known, k, pers, clim, False, None)
            loss = np.where(known, (pm[:, k - 1] - y) ** 2, 0.0)
            scheme = sess.scheme(sess.cfg.schemes.forecast, cell.meta, loss, clusters=_clusters(sess, cell, variant),
                                 what=f"{sess.protocol} {name} {cell.group}")
            base = _base(sess, run, "forecast", cell, "trigger")
            base["note"] = "reference forecast built by the evaluation from labels"
            sess.estimate_rows(_forecast_names(False), [stat], scheme, base,
                               {"n_units": float(known.sum()), "n_events": float(y[known].sum())})


class _StatePersistence:
    """The window at the forecast origin, from the first record of the variant that carries it."""

    def __init__(self, sess: _Session, records: list[StateForecastPredictions]) -> None:
        self.src = next((r for r in records if r.current is not None), None)
        if self.src is None:
            sess.note(f"{sess.protocol}: state persistence unavailable: no state-forecast record carries `current`")

    def for_cell(self, sess: _Session, cell: Cell) -> tuple[np.ndarray | None, np.ndarray | None]:
        if self.src is None or self.src.current is None or self.src.current_mask is None:
            return None, None
        key = (id(cell), "state-pers")
        if key not in sess.cache:
            try:
                idx = align_units(cell.meta, self.src.meta)
                sess.cache[key] = (self.src.current[idx], self.src.current_mask[idx])
            except InvariantViolation as exc:
                sess.note(f"{sess.protocol}: state persistence could not be paired with {cell.group} ({exc})")
                sess.cache[key] = (None, None)
        out: tuple[np.ndarray | None, np.ndarray | None] = sess.cache[key]
        return out


def _state_context(sess: _Session, vruns: list[Run], _ref: Any) -> _StatePersistence:
    return _StatePersistence(sess, [r.records("state_forecast")[0] for r in vruns])


def _state_cells(sess: _Session, rec: StateForecastPredictions, variant: str) -> list[Cell]:
    return _with_horizons(_unit_cells(sess, rec.meta, variant), [int(h) for h in rec.horizons])


def _state_stat(pred: np.ndarray, obs: np.ndarray, mask: np.ndarray, pers: np.ndarray | None, pmask: np.ndarray | None,
                var: np.ndarray | None, es_u: np.ndarray | None, with_es: bool, with_crps: bool) -> Statistic:
    """MAE, RMSE, MSESS against persistence, and the energy score and Gaussian CRPS when available."""
    err = np.where(mask, pred - obs, 0.0)
    cells = mask.sum(axis=1).astype(np.float64)
    if pers is not None and pmask is not None:
        common = mask & pmask
        e_mod = np.where(common, pred - obs, 0.0)
        e_per = np.where(common, pers - obs, 0.0)
        ccells = common.sum(axis=1).astype(np.float64)
    crps_u = np.where(mask, sta.gaussian_crps(pred, var, obs), 0.0).sum(axis=1) if (with_crps and var is not None) else None

    def fn(w: np.ndarray) -> np.ndarray:
        tot = w @ cells
        mse = safe_ratio(w @ (err ** 2).sum(axis=1), tot)
        cols: list[Any] = [safe_ratio(w @ np.abs(err).sum(axis=1), tot), np.sqrt(mse)]
        if pers is not None and pmask is not None:
            ct = w @ ccells
            cols.append(1.0 - safe_ratio(safe_ratio(w @ (e_mod ** 2).sum(axis=1), ct),
                                         safe_ratio(w @ (e_per ** 2).sum(axis=1), ct)))
        else:
            cols.append(np.full(w.shape[0], np.nan))
        if with_es:
            if es_u is None:
                cols.append(np.full(w.shape[0], np.nan))
            else:
                ok = np.isfinite(es_u)
                cols.append(safe_ratio((w * ok) @ np.where(ok, es_u, 0.0), (w * ok).sum(axis=1)))
        if with_crps:
            cols.append(safe_ratio(w @ crps_u, tot) if crps_u is not None else np.full(w.shape[0], np.nan))
        return _columns(cols, w.shape[0])
    return fn


def _energy(sess: _Session, rec: StateForecastPredictions) -> np.ndarray:
    key = (id(rec), "energy")
    if key not in sess.cache:
        sess.cache[key] = sta.energy_score(rec)
    out: np.ndarray = sess.cache[key]
    return out


def _build_state(sess: _Session, run: Run, recs: list[StateForecastPredictions], maps: list[np.ndarray], cell: Cell,
                 pctx: _StatePersistence) -> CellStats | None:
    h = int(cell.horizon)
    his = [int(np.flatnonzero(r.horizons == h)[0]) if np.any(r.horizons == h) else -1 for r in recs]
    if min(his) < 0:
        raise InvariantViolation(f"a seed has no forecast at horizon {h}")
    ref, idx0, hi0 = recs[0], maps[0], his[0]
    mask0 = ref.mask[idx0, hi0]
    if not mask0.any():
        return None
    obs0 = ref.observed[idx0, hi0]
    pers, pmask = pctx.for_cell(sess, cell) if pctx is not None else (None, None)
    with_es = all(r.samples is not None for r in recs)
    with_crps = all(r.variance is not None for r in recs)
    stats: list[Statistic] = []
    dm_sq, dm_abs = [], []
    for rec, idx, hi in zip(recs, maps, his, strict=True):
        pred, obs, mask = rec.predicted[idx, hi], rec.observed[idx, hi], rec.mask[idx, hi]
        if not (np.array_equal(mask, mask0) and np.allclose(obs[mask], obs0[mask0])):
            raise InvariantViolation("the seeds disagree on the observed state")
        var = rec.variance[idx, hi] if (with_crps and rec.variance is not None) else None
        es_u = _energy(sess, rec)[idx, hi] if with_es else None
        stats.append(_state_stat(pred, obs, mask, pers, pmask, var, es_u, with_es, with_crps))
        n_cells = mask.sum(axis=1)
        err = np.where(mask, pred - obs, 0.0)
        dm_sq.append((np.where(n_cells > 0, (err ** 2).sum(axis=1) / np.maximum(n_cells, 1), np.nan), h))
        dm_abs.append((np.where(n_cells > 0, np.abs(err).sum(axis=1) / np.maximum(n_cells, 1), np.nan), h))
    names = ["mae", "rmse", "msess_persistence"] + (["energy_score"] if with_es else []) + (["gaussian_crps"] if with_crps else [])
    n_cells0 = mask0.sum(axis=1)
    loss = np.where(n_cells0 > 0, (np.where(mask0, ref.predicted[idx0, hi0] - obs0, 0.0) ** 2).sum(axis=1)
                    / np.maximum(n_cells0, 1), 0.0)
    return CellStats(names, stats, loss, {"n_units": float((n_cells0 > 0).sum()), "n_events": float(mask0.sum())},
                     {"rmse": ("dm", dm_sq), "mae": ("dm", dm_abs)}, unit="origin",
                     clusters=_clusters(sess, cell, run.variant))


def _state_reference_rows(sess: _Session, vruns: list[Run], ref_rec: StateForecastPredictions, cells: list[Cell],
                          pctx: _StatePersistence, variant: str) -> None:
    """Rows of the persistence forecast (the origin's window repeated) on every state cell."""
    if pctx is None:
        return
    for cell in cells:
        pers, pmask = pctx.for_cell(sess, cell)
        if pers is None or pmask is None:
            continue
        hi = int(np.flatnonzero(ref_rec.horizons == int(cell.horizon))[0])
        mask = ref_rec.mask[cell.index, hi] & pmask
        if not mask.any():
            continue
        obs = ref_rec.observed[cell.index, hi]
        stat = _state_stat(pers, obs, mask, pers, pmask, None, None, False, False)
        n_cells = mask.sum(axis=1)
        loss = np.where(n_cells > 0, (np.where(mask, pers - obs, 0.0) ** 2).sum(axis=1) / np.maximum(n_cells, 1), 0.0)
        run = Run("persistence", "persistence", "reference", variant, [])
        scheme = sess.scheme(sess.cfg.schemes.state_forecast, cell.meta, loss, what=f"{sess.protocol} state persistence")
        base = _base(sess, run, "state_forecast", cell, "origin")
        base["note"] = "reference forecast built by the evaluation"
        sess.estimate_rows(["mae", "rmse", "msess_persistence"], [stat], scheme, base,
                           {"n_units": float((n_cells > 0).sum()), "n_events": float(mask.sum())})


def _stage_cells(sess: _Session, rec: StagePredictions, variant: str) -> list[Cell]:
    return _unit_cells(sess, rec.meta, variant, base_mask=rec.label >= 0)


def _stage_stat(probs: np.ndarray, label: np.ndarray, names: tuple[str, ...], class_set: np.ndarray, cfg: EvaluationConfig
                ) -> Statistic:
    st = cfg.stages

    def fn(w: np.ndarray) -> np.ndarray:
        m = stg.stage_metrics_arrays(probs, label, names, w, top_k=st.top_k, classes=st.classes, class_set=class_set)
        cols: list[Any] = [m[f"top{k}"] for k in st.top_k]
        cols += [m[k] for k in ("macro_f1", "weighted_f1", "accuracy", "ordinal_mae", "ahead_share", "behind_share", "rps",
                                "brier")]
        cols.append(cal.top_label_ece(probs, label, bins=st.ece_bins, weights=w))
        cols.append(cal.adaptive_calibration_error(probs, label, ranges=st.ace_ranges, weights=w))
        return _columns(cols, w.shape[0])
    return fn


def _build_stage(sess: _Session, run: Run, recs: list[StagePredictions], maps: list[np.ndarray], cell: Cell, _ctx: Any
                 ) -> CellStats | None:
    st = sess.cfg.stages
    label = recs[0].label[maps[0]]
    stats: list[Statistic] = []
    mc = []
    for rec, idx in zip(recs, maps, strict=True):
        lab = rec.label[idx]
        if not np.array_equal(lab, label):
            raise InvariantViolation("the seeds disagree on stage labels")
        probs = rec.probs[idx]
        yhat = stg.predicted_class(probs)
        class_set = stg.evaluated_classes(lab, yhat, probs.shape[1], st.classes)
        stats.append(_stage_stat(probs, lab, rec.stage_names, class_set, sess.cfg))
        mc.append((yhat == lab, np.ones(lab.size, dtype=bool)))
    names = [f"top{k}" for k in st.top_k] + ["macro_f1", "weighted_f1", "accuracy", "ordinal_mae", "ahead_share",
                                             "behind_share", "rps", "brier", "top_label_ece", "ace"]
    probs0 = recs[0].probs[maps[0]]
    yhat0 = stg.predicted_class(probs0)
    cm = stg.confusion(label, yhat0, probs0.shape[1])[0]
    tr, pr = np.nonzero(cm)
    sess.figure("stage_confusion", pd.DataFrame({
        "model": run.key, "variant": run.variant, "group": cell.group, "novelty": cell.novelty, "split": cell.split,
        "true": [recs[0].stage_names[i] for i in tr], "predicted": [recs[0].stage_names[j] for j in pr], "weight": cm[tr, pr]}))
    return CellStats(names, stats, (yhat0 != label).astype(np.float64),
                     {"n_units": float(label.size), "n_events": float(np.unique(label).size)},
                     {"top1": ("mcnemar", mc), "accuracy": ("mcnemar", mc)}, unit="unit", truth=label,
                     clusters=_clusters(sess, cell, run.variant))


def _plain_cells(sess: _Session, rec: Any, variant: str) -> list[Cell]:
    return _unit_cells(sess, rec.meta, variant)


PATH_NAMES: tuple[str, ...] = ("precision_at_n", "recall_at_n", "hit_at_1", "exact_at_1", "prefix_at_1",
                               "mean_best_distance", "median_best_distance", "kendall_tau", "ndcg_at_n", "ordinal_safety")


def _build_paths(sess: _Session, run: Run, recs: list[PathPredictions], maps: list[np.ndarray], cell: Cell, _ctx: Any
                 ) -> CellStats | None:
    pc = sess.cfg.paths
    subs = [pth.subset(rec, idx) for rec, idx in zip(recs, maps, strict=True)]
    realised = subs[0].realised
    if any(s.realised != realised for s in subs[1:]):
        raise InvariantViolation("the seeds disagree on realised paths")
    has = np.array([len(r) > 0 for r in realised])
    if not has.any():
        return None

    def make(sub: PathPredictions) -> Statistic:
        def fn(w: np.ndarray) -> np.ndarray:
            m = pth.path_metrics(sub, w, top_n=pc.top_n, tolerance=pc.tolerance, rule=pc.entity_rule,
                                 include_empty=pc.include_empty)
            return _columns([m[n] for n in PATH_NAMES], w.shape[0])
        return fn

    best = np.zeros(len(realised))
    dist = pth.best_distances(subs[0], top_n=pc.top_n, rule=pc.entity_rule)
    best[has] = dist
    sess.figure("edit_distance", pd.DataFrame({"model": run.key, "variant": run.variant, "group": cell.group,
                                               "novelty": cell.novelty, "best_distance": dist}))
    routes = float(sum(min(len(r), pc.top_n) for r, h in zip(subs[0].predicted, has, strict=True) if h))
    return CellStats(list(PATH_NAMES), [make(s) for s in subs], best,
                     {"n_units": float(has.sum()), "n_events": routes}, unit="trigger",
                     clusters=_clusters(sess, cell, run.variant))


def _tte_records(run: Run) -> list[TimeToEventPredictions]:
    """Time-to-event records of a run: given ones, else derived from its forecasts (censoring as `outcome`)."""
    given = run.records("time_to_event")
    if given:
        return given
    fcs = run.records("forecast")
    return [srv.time_to_event_from_forecast(f) for f in fcs] if fcs else []


class _TteCache:
    """Derived time-to-event records are built once per run."""

    def __init__(self) -> None:
        self.store: dict[str, list[TimeToEventPredictions]] = {}

    def __call__(self, run: Run) -> list[TimeToEventPredictions]:
        key = f"{run.key}|{run.variant}"
        if key not in self.store:
            self.store[key] = _tte_records(run)
        return self.store[key]


SURVIVAL_NAMES: tuple[str, ...] = ("c_index", "uno_c", "td_auc", "ibs", "dcal_statistic")


def _build_survival(sess: _Session, run: Run, recs: list[TimeToEventPredictions], maps: list[np.ndarray], cell: Cell,
                    _ctx: Any) -> CellStats | None:
    sc = sess.cfg.survival
    subs = [srv.subset(rec, idx) for rec, idx in zip(recs, maps, strict=True)]
    s0 = subs[0]
    for s in subs[1:]:
        if not (np.array_equal(s.event_time, s0.event_time) and np.array_equal(s.event_observed, s0.event_observed)):
            raise InvariantViolation("the seeds disagree on event times")
    if not s0.event_observed.any():
        return None
    times = s0.time_grid[s0.time_grid > 0]

    def make(sub: TimeToEventPredictions) -> Statistic:
        def fn(w: np.ndarray) -> np.ndarray:
            c = srv.harrell_c(sub.risk, sub.event_time, sub.event_observed, w, tied_times=sc.tied_times)
            u = srv.uno_c(sub.risk, sub.event_time, sub.event_observed, w)
            _, auc = srv.cumulative_dynamic_auc(sub, times, w, marker=sc.marker, interpolation=sc.interpolation)
            ibs = srv.integrated_brier(sub, None, w, interpolation=sc.interpolation)
            dcal = srv.d_calibration_statistic(sub, bins=sc.dcal_bins, weights=w)
            return _columns([c, u, auc, ibs, dcal], w.shape[0])
        return fn

    s_own = srv.survival_at_own_time(s0.survival, s0.time_grid, s0.event_time, interpolation="step")
    loss = (s0.event_observed.astype(np.float64) - (1.0 - s_own)) ** 2
    dc = srv.d_calibration(s0, bins=sc.dcal_bins)
    aucs, _ = srv.cumulative_dynamic_auc(s0, times, None, marker=sc.marker, interpolation=sc.interpolation)
    key = {"model": run.key, "variant": run.variant, "group": cell.group, "novelty": cell.novelty, "split": cell.split}
    sess.figure("td_auc", pd.DataFrame({**key, "time": times, "auc": aucs}))
    sess.figure("dcal", pd.DataFrame({**key, "bin": np.arange(dc.observed.size), "observed": dc.observed,
                                      "expected": dc.expected}))
    return CellStats(list(SURVIVAL_NAMES), [make(s) for s in subs], loss,
                     {"n_units": float(s0.event_time.size), "n_events": float(s0.event_observed.sum())}, unit="trigger",
                     clusters=_clusters(sess, cell, run.variant),
                     note=f"D-calibration chi-square p = {dc.p_value:.4g} (first seed, all units of the cell)")


def _build_forensics(sess: _Session, run: Run, recs: list[ForensicPredictions], maps: list[np.ndarray], cell: Cell,
                     _ctx: Any) -> CellStats | None:
    zc = sess.cfg.forensics
    subs = [fz.subset(rec, idx) for rec, idx in zip(recs, maps, strict=True)]
    windows = {sess.cfg.dataset(d).window_seconds for d in cell.meta["dataset"].astype(str)}
    tol = zc.tolerance_windows * max(windows)
    if len(windows) > 1:
        sess.note(f"{sess.protocol} forensics {cell.group}: datasets with different windows; the narrative tolerance "
                  f"uses the longest window ({tol:g} s)")
    names = ["median_onset_error_s", "onset_coverage", "patient_zero_top1", f"patient_zero_top{zc.top_k}",
             "narrative_precision", "narrative_recall"]

    def make(sub: ForensicPredictions) -> Statistic:
        def fn(w: np.ndarray) -> np.ndarray:
            m = fz.forensic_metrics(sub, w, top_k=zc.top_k, tolerance_seconds=tol, rule=zc.entity_rule)
            return _columns([m[n] for n in names], w.shape[0])
        return fn

    n_true = float(subs[0].narrative_true.shape[0])
    loss = np.zeros(subs[0].onset_pred.shape[0])
    return CellStats(names, [make(s) for s in subs], loss, {"n_units": float(loss.size), "n_events": n_true},
                     unit="incident", clusters=np.arange(loss.size).astype(str))


def _score_timeliness(sess: _Session, runs: list[Run]) -> None:
    """Lead time and episode metrics at each model's own threshold and at the conformal threshold for alpha."""
    for variant in _variants(runs):
        vruns = [r for r in runs if r.variant == variant and r.records("forecast") and _episodes(r) is not None]
        if not vruns:
            continue
        eps = _episodes(vruns[0])
        assert eps is not None
        frame = eps.frame.reset_index(drop=True)
        keep = []
        for r in vruns:
            other = _episodes(r)
            if other is not None and other.frame.reset_index(drop=True).equals(frame):
                keep.append(r)
            else:
                sess.note(f"{sess.protocol} timeliness: {r.key} carries a different episode table and is not compared")
        cells = _episode_cells(sess, frame, _splits_for(sess, variant))
        alerts = {r.key: _alert_tables(sess, r, eps) for r in keep}
        for cell in cells:
            built: dict[str, tuple[Run, CellStats]] = {}
            for r in keep:
                cs = _build_timeliness(sess, r, alerts[r.key], cell)
                scheme = sess.scheme(sess.cfg.schemes.timeliness, cell.meta, cs.loss, clusters=cs.clusters,
                                     what=f"{sess.protocol} timeliness {r.key}")
                base = _base(sess, r, "timeliness", cell, "episode")
                base["note"] = cs.note
                sess.estimate_rows(cs.names, cs.stats, scheme, base, cs.counts)
                built[r.key] = (r, cs)
            _compare_cell(sess, "timeliness", cell, sess.cfg.schemes.timeliness, built, keep)


def _episodes(run: Run) -> EpisodeTable | None:
    """The episode table of a run (every seed must carry the same table)."""
    eps = [s.episodes for s in run.seeds]
    if not eps or any(e is None for e in eps):
        return None
    first = eps[0]
    assert first is not None
    for e in eps[1:]:
        assert e is not None
        if not e.frame.reset_index(drop=True).equals(first.frame.reset_index(drop=True)):
            raise InvariantViolation(f"seeds of {run.key!r} carry different episode tables")
    return first


def _episode_cells(sess: _Session, frame: pd.DataFrame, splits: tuple[str, ...]) -> list[Cell]:
    # Episodes by report group and novelty (never pooled), plus the pooled group per novelty value.
    meta = pd.DataFrame({"time": frame["completion"].to_numpy(dtype=np.float64),
                         "dataset": frame["dataset"].astype(str).to_numpy(),
                         "network": frame["network"].astype(str).to_numpy(),
                         "family": frame["family"].astype(str).to_numpy(),
                         "novelty": frame["novelty"].astype(str).to_numpy(),
                         "entity": frame["entity"].to_numpy(dtype=np.int64),
                         "episode": frame["episode"].astype(str).to_numpy()})
    groups = report_groups(meta, sess.cfg)
    split = splits[0] if splits else ""
    out: list[Cell] = []
    for nov in sorted(set(meta["novelty"])):
        m = meta["novelty"].to_numpy() == nov
        present = sorted(set(groups[m]))
        for g in present:
            out.append(_cell(g, nov, split, "", meta, m & (groups == g)))
        if len(present) > 1:
            out.append(_cell(ALL, nov, split, "", meta, m))
    return out


def _alert_tables(sess: _Session, run: Run, eps: EpisodeTable) -> list[tuple[pd.DataFrame, pd.DataFrame, str]]:
    """Per seed: alerts at the model's own threshold, alerts at the conformal threshold, and a note."""
    out = []
    op = sess.cfg.operating_point
    for f in run.records("forecast"):
        what = f"{sess.protocol} timeliness {run.key} seed"
        horizon = f.horizon * f.window_seconds
        settings = epi.AlertSettings(horizon_seconds=horizon, late_seconds=sess.cfg.timeliness.late_horizons * horizon,
                                     match_entity=sess.cfg.timeliness.match_entity,
                                     window_start=sess.cfg.timeliness.window_start)
        stream = np.isin(f.meta["split"].astype(str).to_numpy(), list(_splits_for(sess, run.variant)))
        idx = np.flatnonzero(stream)
        score = f.p_inf[:, -1]
        y, known = f.outcome(f.horizon)
        benign = known & (y == 0)
        note = ""
        conformal = _conformal_thresholds(sess, f.meta, score, benign, idx, what)
        if op.timeliness_own == "own" and f.alert_threshold is not None:
            own = np.full(idx.size, float(f.alert_threshold))
        elif op.timeliness_own == "own" and op.missing_own == "error":
            raise InvariantViolation(f"{what}: the forecast record has no alert threshold")
        else:
            own = conformal
            note = "own threshold: the conformal threshold at alpha (no alert_threshold in the record)"
            sess.note(f"{what}: {note}")
        meta = f.meta.iloc[idx].reset_index(drop=True)
        a_own = epi.episode_alerts(eps, meta, score[idx], own, settings)
        a_fix = epi.episode_alerts(eps, meta, score[idx], conformal, settings)
        out.append((a_own, a_fix, note))
    return out


def _build_timeliness(sess: _Session, run: Run, tables: list[tuple[pd.DataFrame, pd.DataFrame, str]], cell: Cell) -> CellStats:
    q = sess.cfg.timeliness.quantiles
    names = ["median_lead_time", "alerted_before_completion", "detection_rate", "median_time_to_detect"]
    names += [f"lead_time_q{round(x * 100):02d}" for x in q]
    names += ["lead_time_at_fpr", "alerted_before_completion_at_fpr"]
    idx = cell.index

    def make(a_own: pd.DataFrame, a_fix: pd.DataFrame) -> Statistic:
        own, fix = a_own.iloc[idx].reset_index(drop=True), a_fix.iloc[idx].reset_index(drop=True)

        def fn(w: np.ndarray) -> np.ndarray:
            m = epi.lead_time_metrics(own, w, quantiles=q)
            mf = epi.lead_time_metrics(fix, w, quantiles=q)
            cols = [m["median_lead_time"], m["alerted_before_completion"], m["detection_rate"], m["median_time_to_detect"]]
            cols += [m[f"lead_time_q{round(x * 100):02d}"] for x in q]
            cols += [mf["median_lead_time"], mf["alerted_before_completion"]]
            return _columns(cols, w.shape[0])
        return fn

    stats = [make(a, b) for a, b, _ in tables]
    lead_own = [a["lead_time"].to_numpy(dtype=np.float64)[idx] for a, _, _ in tables]
    lead_fix = [b["lead_time"].to_numpy(dtype=np.float64)[idx] for _, b, _ in tables]
    alerted = [(a["alerted_before_completion"].to_numpy(dtype=bool)[idx], np.ones(idx.size, dtype=bool)) for a, _, _ in tables]
    for s, (a, b, _) in enumerate(tables):
        sess.figure("lead_time_distribution", pd.DataFrame({
            "model": run.key, "variant": run.variant, "group": cell.group, "novelty": cell.novelty, "seed": run.seeds[s].seed,
            "episode": cell.meta["episode"].to_numpy(), "lead_time_own": a["lead_time"].to_numpy()[idx],
            "lead_time_at_fpr": b["lead_time"].to_numpy()[idx]}))
    loss = (~np.isfinite(lead_own[0])).astype(np.float64)
    tests: dict[str, tuple[str, list[Any]]] = {
        "median_lead_time": ("wilcoxon", lead_own), "lead_time_at_fpr": ("wilcoxon", lead_fix),
        "alerted_before_completion": ("mcnemar", alerted)}
    return CellStats(names, stats, loss, {"n_units": float(idx.size), "n_events": float(idx.size)}, tests,
                     unit="episode", clusters=cell.meta["episode"].astype(str).to_numpy(), note=tables[0][2])


def _score_components(sess: _Session, runs: list[Run]) -> None:
    """Component diagnostics of every run whose bundles carry them (components.py), per seed mean."""
    for run in runs:
        comps = [s.component for s in run.seeds]
        if not comps or not all(comps):
            continue
        cell = Cell("component", "", sess.splits[0] if sess.splits else "", "", np.zeros(0, dtype=np.int64), pd.DataFrame())
        avail = [m for m in comp.available_metrics(comps[0]) if all(all(k in c for k in m.keys) for c in comps)]
        for m in avail:
            base = _base(sess, run, "component", Cell(m.component, "", cell.split, "", cell.index, cell.meta),
                         "component unit")
            sizes = [comp.unit_count(m, c) for c in comps]
            if not m.resample:
                vals = [float(np.asarray(m.compute(c, None), dtype=np.float64).reshape(-1)[0]) for c in comps]
                base["note"] = f"required: {m.required}" if m.required else ""
                sess.plain_row(base, m.name, float(np.mean(vals)), math.nan, math.nan,
                               "invariant (exact, not resampled)", len(vals), {"n_units": float(sizes[0]), "n_events": 0.0},
                               seed_sd=float(np.std(vals, ddof=1)) if len(vals) > 1 else math.nan)
                continue
            use = comps if len(set(sizes)) == 1 else comps[:1]
            if len(use) < len(comps):
                sess.note(f"{sess.protocol} {run.key}: {m.name} has different unit counts across seeds; the first seed is used")
            first = np.asarray(m.compute(use[0], np.ones((1, sizes[0]))), dtype=np.float64).reshape(1, -1)
            names = [m.name] if first.shape[1] == 1 else [f"{m.name}_{s}" for s in ("min", "max")][: first.shape[1]]
            stats = [_component_stat(m, c) for c in use]
            meta = pd.DataFrame({"time": np.arange(sizes[0], dtype=np.float64), "dataset": "", "network": ""})
            scheme = sess.scheme(sess.cfg.schemes.components, meta, np.zeros(sizes[0]), what=f"{run.key} {m.name}")
            base["note"] = f"required: {m.required}" if m.required else ""
            sess.estimate_rows(names, stats, scheme, base, {"n_units": float(sizes[0]), "n_events": 0.0})
        _generator_rows(sess, run, comps)
        _ordinal_horizon_rows(sess, run, comps)


def _component_stat(m: comp.ComponentMetric, c: dict[str, np.ndarray]) -> Statistic:
    def fn(w: np.ndarray) -> np.ndarray:
        return np.asarray(m.compute(c, w), dtype=np.float64).reshape(w.shape[0], -1)
    return fn


def _generator_rows(sess: _Session, run: Run, comps: list[dict[str, np.ndarray]]) -> None:
    # MMD and Wasserstein distances: seed mean of the point values, interval from the seed-mean replicates of
    # independent two-sample bootstraps (the replicate b of every seed is averaged).
    if not all("generator.real" in c and "generator.synthetic" in c for c in comps):
        return
    reps: list[dict[str, tuple[float, float, float]]] = []
    b = sess.cfg.resampling.heavy_resamples
    for c in comps:
        reps.append(comp.generator_distance_intervals(c, n_resamples=b, confidence=sess.boot.confidence, rng=sess.rng))
    cell = Cell("Generator", "", sess.splits[0] if sess.splits else "", "", np.zeros(0, dtype=np.int64), pd.DataFrame())
    base = _base(sess, run, "component", cell, "sample")
    for name in reps[0]:
        vals = np.array([r[name][0] for r in reps])
        lows = np.array([r[name][1] for r in reps])
        highs = np.array([r[name][2] for r in reps])
        sess.plain_row(base, name, float(vals.mean()), float(lows.mean()), float(highs.mean()),
                       "percentile (two-sample bootstrap of real and generated samples; seed mean of the bounds)", len(vals),
                       {"n_units": float(np.asarray(comps[0]["generator.synthetic"]).shape[0]),
                        "n_events": float(np.asarray(comps[0]["generator.real"]).shape[0])},
                       seed_sd=float(vals.std(ddof=1)) if vals.size > 1 else math.nan, n_resamples=b)


def _ordinal_horizon_rows(sess: _Session, run: Run, comps: list[dict[str, np.ndarray]]) -> None:
    # Ordinal safety per horizon from ordinal.model_value / ordinal.true_value [m, C, H] (protocol P6).
    keys = ("ordinal.model_value", "ordinal.true_value")
    if not all(all(k in c for k in keys) for c in comps):
        return
    arrs = [(np.asarray(c["ordinal.model_value"], dtype=np.float64), np.asarray(c["ordinal.true_value"], dtype=np.float64),
             np.asarray(c.get("ordinal.valid", np.ones(np.shape(c["ordinal.model_value"]), dtype=bool)), dtype=bool))
            for c in comps]
    m, _, h = arrs[0][0].shape
    if any(a[0].shape != arrs[0][0].shape for a in arrs):
        sess.note(f"{sess.protocol} {run.key}: ordinal values differ in shape across seeds; the first seed is used")
        arrs = arrs[:1]
    names = [f"ordinal_safety_h{j + 1}" for j in range(h)]

    def make(mv: np.ndarray, tv: np.ndarray, ok: np.ndarray) -> Statistic:
        def fn(w: np.ndarray) -> np.ndarray:
            return np.asarray(pth.ordinal_safety(mv, tv, ok, w), dtype=np.float64).reshape(w.shape[0], -1)
        return fn

    meta = pd.DataFrame({"time": np.arange(m, dtype=np.float64), "dataset": "", "network": ""})
    scheme = sess.scheme("iid", meta, np.zeros(m))
    cell = Cell(ALL, "", sess.splits[0] if sess.splits else "", "", np.zeros(0, dtype=np.int64), meta)
    base = _base(sess, run, "paths", cell, "trigger", table="res-stages")
    est = sess.estimate_rows(names, [make(*a) for a in arrs], scheme, base, {"n_units": float(m), "n_events": 0.0})
    sess.figure("ordinal_safety_horizon", pd.DataFrame({"model": run.key, "variant": run.variant,
                                                        "horizon": np.arange(1, h + 1),
                                                        "ordinal_safety": [est[n][0] for n in names],
                                                        "low": [est[n][1] for n in names],
                                                        "high": [est[n][2] for n in names]}))


def _score_operations(sess: _Session, runs: list[Run]) -> None:
    """Latency, throughput, memory growth and trigger time per telemetry level and offered rate (P8)."""
    for run in runs:
        recs = run.records("operations")
        if not recs:
            continue
        keyed: dict[tuple[str, float], list[int]] = {}
        for i, o in enumerate(recs):
            keyed.setdefault((o.telemetry, float(o.offered_rate)), []).append(i)
        for (tele, rate), members in keyed.items():
            cell = Cell(tele, "", sess.splits[0] if sess.splits else "", f"rate={rate:g}", np.zeros(0, dtype=np.int64),
                        pd.DataFrame())
            base = _base(sess, run, "operations", cell, "state update")
            o0 = recs[members[0]]
            base["note"] = f"passes R = {o0.passes}; host: {o0.host or 'n/a'}; dataset: {o0.dataset or 'n/a'}"
            counts_ = {"n_units": float(sum(recs[i].latency_s.size for i in members)),
                       "n_events": float(sum(recs[i].trigger_s.size for i in members))}
            summaries = [ops.operations_summary(recs[i], sess.boot, sess.rng, batches=sess.cfg.operations.batches)
                         for i in members]
            methods = {"latency_p50_ms": "stationary block bootstrap over updates", "latency_p99_ms":
                       "stationary block bootstrap over updates", "throughput_per_s": "batch means, Student t",
                       "memory_gib_per_day": "OLS slope, Newey-West standard error, Student t",
                       "trigger_median_s": "distribution-free order-statistic interval"}
            for name in summaries[0]:
                vals = np.array([s[name][0] for s in summaries])
                if len(summaries) == 1:
                    sess.plain_row(base, name, float(vals[0]), summaries[0][name][1], summaries[0][name][2],
                                   methods.get(name, ""), 1, counts_, n_resamples=sess.boot.n_resamples)
                else:
                    agg = sig.aggregate_seeds(vals, confidence=sess.boot.confidence)
                    sess.plain_row(base, name, agg.mean, agg.low, agg.high, "Student t over repeated runs", agg.n_seeds,
                                   counts_, seed_sd=agg.sd)
            _profile_rows(sess, run, [recs[i] for i in members], cell)


def _degradation(sess: _Session, runs: list[Run]) -> None:
    """P4 and P5: wider uncertainty and shorter safe horizon against the reference run, per cell (one-sided)."""
    key = "regime" if sess.protocol == "P4" else "corruption"
    ref_value = sess.cfg.protocols.p4_reference_regime if sess.protocol == "P4" else "none"
    prim = [r for r in runs if r.role == "primary" and r.records("forecast")]
    ref_run = next((r for r in prim if variant_value(r.variant, key) == ref_value), None)
    if ref_run is None:
        sess.note(f"{sess.protocol}: no reference run ({key} = {ref_value}) of the primary model; degradation is not assessed")
        return
    ref_recs = ref_run.records("forecast")
    for run in prim:
        if run is ref_run:
            continue
        recs = run.records("forecast")
        missing = [r for r in recs + ref_recs if r.p_inf_lower is None or r.p_inf_upper is None or r.safe_horizon is None]
        cells = _unit_cells(sess, ref_recs[0].meta, ref_run.variant)
        for cell in cells:
            flag = {"variant": run.variant, "model": run.key, "group": cell.group, "novelty": cell.novelty,
                    "split": cell.split, "reference": ref_run.variant, "available": not missing}
            sess.pending_flags.append(flag)
            if missing:
                continue
            try:
                maps_a = [align_units(cell.meta, r.meta) for r in recs]
                maps_b = [align_units(cell.meta, r.meta) for r in ref_recs]
            except InvariantViolation as exc:
                sess.note(f"{sess.protocol}: {run.variant} cannot be paired with {ref_run.variant} in {cell.group} ({exc})")
                flag["available"] = False
                continue
            width_a = [r.p_inf_upper[i, -1] - r.p_inf_lower[i, -1] for r, i in zip(recs, maps_a, strict=True)]   # type: ignore[index]
            width_b = [r.p_inf_upper[i, -1] - r.p_inf_lower[i, -1] for r, i in zip(ref_recs, maps_b, strict=True)]   # type: ignore[index]
            safe_a = [r.safe_horizon[i].astype(np.float64) for r, i in zip(recs, maps_a, strict=True)]   # type: ignore[index]
            safe_b = [r.safe_horizon[i].astype(np.float64) for r, i in zip(ref_recs, maps_b, strict=True)]   # type: ignore[index]

            def mean_stat(v: np.ndarray) -> Statistic:
                def fn(w: np.ndarray) -> np.ndarray:
                    return safe_ratio(w @ v, w.sum(axis=1))[:, None]
                return fn

            pairs = _seed_pairs(sess, run, ref_run)
            for metric, va, vb, kind in (("uncertainty_width_K", width_a, width_b, "wilcoxon_greater"),
                                         ("safe_horizon", safe_a, safe_b, "wilcoxon_less")):
                res = [_run_test(sess, kind, va[i], vb[j], cell) for i, j in pairs]
                loss = np.nan_to_num(va[0] - vb[0])
                scheme = sess.scheme(sess.cfg.schemes.forecast, cell.meta, loss, what=f"{sess.protocol} degradation")
                base = {"protocol": sess.protocol, "family": "degradation", "task": "degradation", "model_a": run.key,
                        "model_b": f"{ref_run.key} ({ref_run.variant})", "variant": run.variant, "group": cell.group,
                        "novelty": cell.novelty, "split": cell.split, "horizon": "", "n_seeds_a": run.n_seeds,
                        "n_seeds_b": ref_run.n_seeds}
                sess.compare([metric], [mean_stat(v) for v in va], [mean_stat(v) for v in vb], scheme, base,
                             {metric: (kind, [r[0] for r in res], res[0][1] if res else math.nan,
                                       res[0][2] if res else math.nan)}, int(cell.index.size))


def _degradation_flags(sess: _Session, comparisons: pd.DataFrame) -> None:
    """Rows "degradation_signalled": 1 when both one-sided tests stay significant after adjustment, else 0."""
    for f in sess.pending_flags:
        cell = Cell(f["group"], f["novelty"], f["split"], "", np.zeros(0, dtype=np.int64), pd.DataFrame())
        run = Run(f["model"], f["model"], "primary", f["variant"], [])
        base = _base(sess, run, "degradation", cell, "trigger", table="res-robustness")
        if not f["available"]:
            base["note"] = "the forecast records carry no uncertainty band or safe horizon, or cannot be paired"
            sess.plain_row(base, "degradation_signalled", math.nan, math.nan, math.nan, "decision (not available)", 0,
                           {"n_units": 0.0, "n_events": 0.0})
            continue
        sel = comparisons[(comparisons["task"] == "degradation") & (comparisons["variant"] == f["variant"])
                          & (comparisons["group"] == f["group"]) & (comparisons["novelty"] == f["novelty"])
                          & (comparisons["split"] == f["split"])]
        width = sel[sel["metric"] == "uncertainty_width_K"]
        safe = sel[sel["metric"] == "safe_horizon"]
        ok = (len(width) == 1 and len(safe) == 1 and bool(width["significant"].iloc[0]) and bool(safe["significant"].iloc[0])
              and float(width["difference"].iloc[0]) > 0 and float(safe["difference"].iloc[0]) < 0)
        base["note"] = (f"against {f['reference']}: width change {float(width['difference'].iloc[0]) if len(width) else math.nan:.4g}"
                        f" (adjusted p {float(width['p_adjusted'].iloc[0]) if len(width) else math.nan:.3g}), safe-horizon "
                        f"change {float(safe['difference'].iloc[0]) if len(safe) else math.nan:.4g} (adjusted p "
                        f"{float(safe['p_adjusted'].iloc[0]) if len(safe) else math.nan:.3g})")
        n = float(width["n_units"].iloc[0]) if len(width) else 0.0
        sess.plain_row(base, "degradation_signalled", 1.0 if ok else 0.0, math.nan, math.nan,
                       "decision from the comparisons uncertainty_width_K and safe_horizon (adjusted one-sided Wilcoxon)",
                       int(width["n_seeds_a"].iloc[0]) if len(width) else 0, {"n_units": n, "n_events": 0.0})


def _information_audit(sess: _Session, runs: list[Run]) -> None:
    """P6: mutual information, Fano ceiling, model accuracy and gap per regime, with bootstrap intervals."""
    keys = ("audit.observation", "audit.hidden", "audit.regime", "audit.correct")
    pc = sess.cfg.protocols
    for run in runs:
        comps = [s.component for s in run.seeds]
        if not comps or not all(all(k in c for k in keys) for c in comps):
            continue
        c0 = comps[0]
        obs, hid = np.asarray(c0["audit.observation"]).reshape(-1), np.asarray(c0["audit.hidden"]).reshape(-1)
        reg = np.asarray(c0["audit.regime"]).astype(str).reshape(-1)
        correct = [np.asarray(c["audit.correct"], dtype=np.float64).reshape(-1) for c in comps]
        try:
            point = [comp.information_audit(obs, hid, reg, cr, enabled_proposals=pc.p6_enabled_proposals,
                                            miller_madow=pc.p6_miller_madow) for cr in correct]
        except ProposalNotEnabled as exc:
            sess.note(f"{sess.protocol}: the information audit needs proposal P-15 in protocols.p6_enabled_proposals ({exc})")
            return
        metrics = ("mutual_information", "entropy", "accuracy_ceiling", "model_accuracy", "gap")
        b = sess.cfg.resampling.heavy_resamples
        for i, row in enumerate(point[0]):
            regime = str(row["regime"])
            members = np.flatnonzero(reg == regime)
            reps = np.full((b, len(metrics)), np.nan)
            for rb in range(b):
                pick = members[sess.rng.integers(0, members.size, members.size)]
                vals = []
                for cr in correct:
                    out = comp.information_audit(obs[pick], hid[pick], reg[pick], cr[pick],
                                                 enabled_proposals=pc.p6_enabled_proposals, miller_madow=pc.p6_miller_madow)
                    vals.append([float(out[0][m]) for m in metrics])
                reps[rb] = np.mean(np.asarray(vals), axis=0)
            cell = Cell(regime, "", sess.splits[0] if sess.splits else "", "", members, pd.DataFrame())
            base = _base(sess, run, "audit", cell, "unit")
            for j, mname in enumerate(metrics):
                vals_seed = np.array([float(p[i][mname]) for p in point])
                fin = reps[:, j][np.isfinite(reps[:, j])]
                lo, hi = ((float(np.quantile(fin, (1 - sess.boot.confidence) / 2)), float(np.quantile(fin, (1 + sess.boot.confidence) / 2)))
                          if fin.size else (math.nan, math.nan))
                sess.plain_row(base, mname, float(vals_seed.mean()), lo, hi, "percentile (iid bootstrap within the regime)",
                               len(vals_seed), {"n_units": float(members.size), "n_events": float(np.unique(hid[members]).size)},
                               seed_sd=float(vals_seed.std(ddof=1)) if vals_seed.size > 1 else math.nan, n_resamples=b)


def _protocol_context(sess: _Session, runs: list[Run]) -> None:
    # Splits, training networks, episode tables per variant, the forecast horizon in seconds, ablation ids.
    if sess.protocol == "P-ABL":
        sess.splits = tuple(dict.fromkeys(sp for r in runs for sp in _splits_for(sess, r.variant)))
    elif sess.protocol == "P-CW":
        sess.splits = ()
    else:
        sess.splits = _splits_for(sess, "")
    for r in runs:
        if r.role == "ablation":
            aid = r.key[r.key.index("[") + 1: -1]
            if sess.cfg.ablation(aid) is None:
                sess.note(f"{sess.protocol}: ablation {aid} of {r.model} is not in the configured ablation registry")
    sess.train_nets = training_networks([b for r in runs for b in r.seeds])
    for r in runs:
        eps = _episodes(r)
        if eps is not None and r.variant not in sess.episodes:
            sess.episodes[r.variant] = eps
        fcs = r.records("forecast")
        if fcs and sess.horizon_seconds is None:
            sess.horizon_seconds = float(fcs[0].horizon * fcs[0].window_seconds)


def _score_protocol(sess: _Session, runs: list[Run]) -> None:
    """Score every task present, then the protocol-specific views (degradation, information audit)."""
    _protocol_context(sess, runs)
    s = sess.cfg.schemes
    _score_unit_task(sess, runs, "detection", _records("detection"), _detection_cells, _build_detection, s.detection)
    _score_unit_task(sess, runs, "forecast", _records("forecast"), _forecast_cells, _build_forecast, s.forecast,
                     context=_forecast_context, after=_forecast_reference_rows)
    _score_unit_task(sess, runs, "state_forecast", _records("state_forecast"), _state_cells, _build_state, s.state_forecast,
                     context=_state_context, after=_state_reference_rows)
    _score_unit_task(sess, runs, "stage", _records("stage"), _stage_cells, _build_stage, s.stage)
    _score_unit_task(sess, runs, "paths", _records("paths"), _plain_cells, _build_paths, s.paths)
    _score_unit_task(sess, runs, "time_to_event", _TteCache(), _plain_cells, _build_survival, s.time_to_event)
    _score_unit_task(sess, runs, "forensics", _records("forensics"), _plain_cells, _build_forensics, s.forensics)
    _score_timeliness(sess, runs)
    _score_components(sess, runs)
    _score_operations(sess, runs)
    _score_explanations(sess, runs)
    if sess.protocol == "P-CW":
        _score_arena(sess, runs)
    if sess.protocol in ("P4", "P5"):
        _degradation(sess, runs)
    if sess.protocol == "P6":
        _information_audit(sess, runs)


def _registration_deviations(sess: _Session, registered: Registered, runs: list[Run]) -> list[Deviation]:
    """Deviations of the bundles from their registration (registry.run_deviations)."""
    models = {r.key: [s.seed for s in r.seeds] for r in runs}
    splits: set[str] = set()
    datasets: set[str] = set()
    for r in runs:
        for b in r.seeds:
            metas = [getattr(b, t).meta for t in ("detection", "forecast", "stage", "state_forecast", "time_to_event",
                                                   "paths", "forensics") if getattr(b, t) is not None]
            metas += [e.meta for e in (b.explanations or [])]
            for m in metas:
                splits |= set(m["split"].astype(str))
                datasets |= set(m["dataset"].astype(str))
            if b.operations is not None and b.operations.dataset:
                datasets.add(b.operations.dataset)
    variants = {r.variant for r in runs if r.variant}
    started = [(r.key, b.seed, str(b.config["started_utc"]) if b.config.get("started_utc") else None)
               for r in runs for b in r.seeds]
    return regm.run_deviations(registered, config=to_dict(sess.cfg), models=models, splits=splits, datasets=datasets,
                               variants=variants, started=started)


def _matching(frame: pd.DataFrame, spec: Any, fields: Sequence[str]) -> pd.DataFrame:
    # Rows equal to every non-empty field of spec.
    out = frame
    for f in fields:
        v = getattr(spec, f)
        if v != "":
            out = out[out[f].astype(str) == str(v)]
    return out


def _primary_deviations(registered: Registered, metrics: pd.DataFrame) -> list[Deviation]:
    """A major deviation for every registered primary metric that no metric row reports."""
    out: list[Deviation] = []
    for pm in registered.registration.primary:
        rows = _matching(metrics, pm, ("task", "metric", "group", "novelty", "variant", "horizon"))
        if rows.empty or not np.isfinite(rows["value"].to_numpy(dtype=np.float64)).any():
            out.append(Deviation("major", "primary_metric", f"primary metric {pm.task}/{pm.metric} (group {pm.group or 'any'}, "
                                 f"novelty {pm.novelty or 'any'}, variant {pm.variant or 'any'}, horizon "
                                 f"{pm.horizon or 'any'}) was not computed"))
    return out


def _hypotheses(sess: _Session, registered: Registered, comparisons: pd.DataFrame, deviations: list[Deviation]
                ) -> pd.DataFrame:
    """Registered hypotheses tested as one confirmatory family (registry.py)."""
    hyps = registered.registration.hypotheses
    if not hyps:
        return pd.DataFrame(columns=list(HYPOTHESIS_COLUMNS))
    major = any(d.severity == "major" for d in deviations)
    rows = []
    for h in hyps:
        sel = comparisons[(comparisons["task"] == h.task) & (comparisons["metric"] == h.metric)] if len(comparisons) else comparisons
        sel = _matching(sel, h, ("group", "novelty", "variant", "horizon")) if len(sel) else sel
        fwd = sel[(sel["model_a"] == h.model_a) & (sel["model_b"] == h.model_b)] if len(sel) else sel
        rev = sel[(sel["model_a"] == h.model_b) & (sel["model_b"] == h.model_a)] if len(sel) else sel
        row: dict[str, Any] = {"id": h.id, "statement": h.statement, "task": h.task, "metric": h.metric,
                               "model_a": h.model_a, "model_b": h.model_b, "direction": h.direction, "margin": h.margin,
                               "difference": math.nan, "ci_low": math.nan, "ci_high": math.nan, "test": "",
                               "p_two_sided": math.nan, "p_directional": math.nan, "note": ""}
        found = len(fwd) + len(rev)
        if found != 1:
            row["note"] = f"not testable: {found} comparisons match the registered cell"
        else:
            r = (fwd if len(fwd) else rev).iloc[0]
            sign = 1.0 if len(fwd) else -1.0
            row["difference"] = sign * float(r["difference"])
            lo, hi = float(r["ci_low"]), float(r["ci_high"])
            row["ci_low"], row["ci_high"] = (lo, hi) if sign > 0 else (-hi, -lo)
            row["test"] = str(r["test"])
            row["p_two_sided"] = float(r["p_value"])
            row["p_directional"] = regm.directional_p(row["p_two_sided"], row["difference"], h.direction)
        rows.append(row)
    frame = pd.DataFrame(rows)
    c = sess.cfg.comparisons
    frame["p_adjusted"] = sig.adjust_pvalues(frame["p_directional"].to_numpy(dtype=np.float64), c.fdr_method)
    d = frame["difference"].to_numpy(dtype=np.float64)
    m = frame["margin"].to_numpy(dtype=np.float64)
    side = np.where(frame["direction"] == "greater", d > m, np.where(frame["direction"] == "less", d < -m, np.abs(d) > m))
    frame["supported"] = (frame["p_adjusted"].to_numpy(dtype=np.float64) <= c.fdr_level) & side
    frame["verdict"] = np.where(frame["note"] != "", "not testable", "exploratory" if major else "confirmatory")
    return frame[list(HYPOTHESIS_COLUMNS)]


def _faith_stat(rec: ExplanationCurves, idx: np.ndarray) -> Statistic:
    sub = fth.subset(rec, idx)

    def fn(w: np.ndarray) -> np.ndarray:
        m = fth.faithfulness_metrics(sub, w)
        return _columns([m[n] for n in fth.FAITHFULNESS_NAMES], w.shape[0])
    return fn


def _score_explanations(sess: _Session, runs: list[Run]) -> None:
    """Deletion and insertion faithfulness per explanation method, and paired comparisons between methods."""
    names = list(fth.FAITHFULNESS_NAMES)
    for run in runs:
        per_seed = [{e.method: e for e in (b.explanations or [])} for b in run.seeds]
        if not per_seed or not all(per_seed):
            continue
        methods = [m for m in per_seed[0] if all(m in d for d in per_seed)]
        for method in methods:
            recs = [d[method] for d in per_seed]
            for cell in _unit_cells(sess, recs[0].meta, run.variant):
                try:
                    maps = [align_units(cell.meta, r.meta) for r in recs]
                except InvariantViolation as exc:
                    sess.note(f"{sess.protocol} explanations {run.key} {method}: seeds cannot be paired ({exc})")
                    continue
                stats = [_faith_stat(r, idx) for r, idx in zip(recs, maps, strict=True)]
                scores0 = fth.unit_scores(fth.subset(recs[0], maps[0]))
                scheme = sess.scheme(sess.cfg.schemes.explanations, cell.meta, scores0["deletion_auc"],
                                     what=f"{sess.protocol} explanations {run.key}")
                base = _base(sess, run, "explanations", cell, "explained prediction")
                base["method"] = method
                sess.estimate_rows(names, stats, scheme, base, {"n_units": float(cell.index.size),
                                                                "n_events": float(recs[0].fractions.size)})
                _faith_curves(sess, run, method, fth.subset(recs[0], maps[0]), cell)
        _compare_methods(sess, run, per_seed, methods)


def _faith_curves(sess: _Session, run: Run, method: str, sub: ExplanationCurves, cell: Cell) -> None:
    # Mean deletion and insertion curves with pointwise percentile intervals of an iid bootstrap over units.
    n = len(sub.meta)
    b = sess.cfg.resampling.heavy_resamples
    w = sess.rng.multinomial(n, np.full(n, 1.0 / n), size=b).astype(np.float64)
    q = [(1 - sess.boot.confidence) / 2, (1 + sess.boot.confidence) / 2]
    data: dict[str, Any] = {"model": run.key, "variant": run.variant, "method": method, "group": cell.group,
                            "novelty": cell.novelty, "fraction": sub.fractions}
    curves = {"deletion": sub.deletion, "insertion": sub.insertion}
    if sub.random_deletion is not None and sub.random_insertion is not None:
        curves |= {"random_deletion": sub.random_deletion, "random_insertion": sub.random_insertion}
    for name, mat in curves.items():
        reps = (w @ mat) / w.sum(axis=1, keepdims=True)                 # [B, F]
        data[name] = mat.mean(axis=0)
        data[f"{name}_low"], data[f"{name}_high"] = np.quantile(reps, q, axis=0)
    sess.figure("faithfulness_curves", pd.DataFrame(data))


def _compare_methods(sess: _Session, run: Run, per_seed: list[dict[str, ExplanationCurves]], methods: list[str]) -> None:
    # Paired comparisons of every pair of explanation methods on the units of the first method's cells.
    names = list(fth.FAITHFULNESS_NAMES)
    for a, b in itertools.combinations(methods, 2):
        recs_a = [d[a] for d in per_seed]
        recs_b = [d[b] for d in per_seed]
        for cell in _unit_cells(sess, recs_a[0].meta, run.variant):
            try:
                maps_a = [align_units(cell.meta, r.meta) for r in recs_a]
                maps_b = [align_units(cell.meta, r.meta) for r in recs_b]
            except InvariantViolation as exc:
                sess.note(f"{sess.protocol} explanations {run.key}: {a} and {b} cannot be paired ({exc})")
                continue
            sa = [_faith_stat(r, i) for r, i in zip(recs_a, maps_a, strict=True)]
            sb = [_faith_stat(r, i) for r, i in zip(recs_b, maps_b, strict=True)]
            ua = [fth.unit_scores(fth.subset(r, i)) for r, i in zip(recs_a, maps_a, strict=True)]
            ub = [fth.unit_scores(fth.subset(r, i)) for r, i in zip(recs_b, maps_b, strict=True)]
            tests: dict[str, tuple[str, list[float], float, float]] = {}
            for metric in ("deletion_auc", "insertion_auc", "aopc"):
                res = [_run_test(sess, "wilcoxon", x[metric], y[metric], cell) for x, y in zip(ua, ub, strict=True)]
                tests[metric] = ("wilcoxon", [r[0] for r in res], res[0][1], res[0][2])
            loss = np.nan_to_num(ua[0]["deletion_auc"] - ub[0]["deletion_auc"])
            scheme = sess.scheme(sess.cfg.schemes.explanations, cell.meta, loss, what=f"{sess.protocol} {a} vs {b}")
            base = {"protocol": sess.protocol, "family": "explanations", "task": "explanations",
                    "model_a": f"{run.key}:{a}", "model_b": f"{run.key}:{b}", "variant": run.variant,
                    "group": cell.group, "novelty": cell.novelty, "split": cell.split, "horizon": "",
                    "n_seeds_a": run.n_seeds, "n_seeds_b": run.n_seeds}
            sess.compare(names, sa, sb, scheme, base, tests, int(cell.index.size))


def _score_arena(sess: _Session, runs: list[Run]) -> None:
    """P-CW: final return, improvement over the control, steps to exceed it, generalisation, learning curves."""
    cfg = sess.cfg
    method = "percentile bootstrap over runs"
    nb, conf = cfg.resampling.n_resamples, sess.boot.confidence
    by_variant: dict[str, list[tuple[Run, list[ArenaRun]]]] = {}
    for run in runs:
        recs: list[ArenaRun] = run.records("arena")
        if not recs:
            continue
        if len(recs) < cfg.arena.min_seeds:
            sess.note(f"P-CW: {run.key} has {len(recs)} seeds; the arena needs at least {cfg.arena.min_seeds}; not scored")
            continue
        try:
            summary = arn.summarise(run.key, recs, n_resamples=nb, confidence=conf, rng=sess.rng)
        except InvariantViolation as exc:
            sess.note(f"P-CW: {run.key} not scored ({exc})")
            continue
        env = recs[0].environment
        cell = Cell(env, "", "", "", np.zeros(0, dtype=np.int64), pd.DataFrame())
        base = _base(sess, run, "arena", cell, "run")
        exceed = np.array([arn.steps_to_exceed(r, sustained=False) for r in recs])
        counts_ = {"n_units": float(sum(r.final_returns.size for r in recs)), "n_events": float(np.isfinite(exceed).sum())}
        finals = arn.run_scores([r.final_returns for r in recs])
        sd = float(finals.std(ddof=1)) if finals.size > 1 else math.nan
        for name, triple in (("final_iqm", summary.final_iqm), ("final_mean", summary.final_mean),
                             ("improvement_over_control", summary.improvement_over_control),
                             ("steps_to_exceed", summary.steps_to_exceed),
                             ("steps_to_exceed_sustained", summary.steps_to_exceed_sustained),
                             ("exceeded_share", arn.bootstrap_runs(np.isfinite(exceed).astype(np.float64), np.mean,
                                                                   n_resamples=nb, confidence=conf, rng=sess.rng))):
            sess.plain_row(base, name, *triple, method, len(recs), counts_, seed_sd=sd if name.startswith("final") else math.nan,
                           n_resamples=nb)
        for cname, metrics in summary.conditions.items():
            cbase = dict(base, condition=cname, novelty=recs[0].condition_kinds[cname])
            for mname, triple in metrics.items():
                sess.plain_row(cbase, mname, *triple, method, len(recs),
                               {"n_units": float(sum(r.conditions[cname].size for r in recs)), "n_events": 0.0},
                               n_resamples=nb)
        curve = arn.learning_curve(run.key, recs, n_resamples=nb, confidence=conf, rng=sess.rng)
        sess.figure("learning_curves", pd.DataFrame(curve).assign(variant=run.variant))
        sess.figure("steps_to_exceed", pd.DataFrame({"agent": run.key, "variant": run.variant,
                                                     "seed": [s.seed for s in run.seeds], "steps_to_exceed": exceed,
                                                     "steps_to_exceed_sustained": [arn.steps_to_exceed(r, sustained=True)
                                                                                   for r in recs]}))
        by_variant.setdefault(run.variant, []).append((run, recs))
    for variant, agents in by_variant.items():
        finals = {r.key: arn.run_scores([x.final_returns for x in recs]) for r, recs in agents}
        pooled = np.concatenate(list(finals.values()))
        taus = np.linspace(float(pooled.min()), float(pooled.max()), cfg.arena.profile_points)
        for key, sc in finals.items():
            sess.figure("performance_profiles", pd.DataFrame({"agent": key, "variant": variant, "tau": taus,
                                                              "fraction_above": arn.performance_profile(sc, taus)}))
        primary = next(((r, recs) for r, recs in agents if r.role == "primary"), None)
        if primary is None:
            continue
        for r, recs in agents:
            if r is primary[0]:
                continue
            res = arn.compare_agents(primary[1], recs, n_resamples=nb, confidence=conf, rng=sess.rng)
            sess.comparisons.append({
                "protocol": sess.protocol, "family": "arena", "task": "arena", "model_a": primary[0].key, "model_b": r.key,
                "variant": variant, "group": primary[1][0].environment, "novelty": "", "split": "", "horizon": "",
                "metric": "final_iqm", "value_a": float(arn.iqm(arn.run_scores([x.final_returns for x in primary[1]]))),
                "value_b": float(arn.iqm(arn.run_scores([x.final_returns for x in recs]))),
                "difference": res["iqm_difference"], "ci_low": res["iqm_difference_low"],
                "ci_high": res["iqm_difference_high"], "ci_method": "percentile bootstrap over runs (independent agents)",
                "test": "permutation (two-sample, runs)", "statistic": math.nan, "p_value": res["p_value"],
                "p_adjusted": math.nan, "significant": False, "p_values_by_seed": "",
                "effect": res["probability_of_improvement"], "n_units": int(res["runs_x"] + res["runs_y"]),
                "n_seeds_a": primary[0].n_seeds, "n_seeds_b": r.n_seeds})


def _profile_rows(sess: _Session, run: Run, recs: list[Any], cell: Cell) -> None:
    """Compute and latency profile: per-stage latency quantiles, operations per update and peak memory."""
    conf = sess.boot.confidence
    stages: dict[str, list[np.ndarray]] = {}
    for o in recs:
        for name, values in (o.stage_latency_s or {}).items():
            stages.setdefault(name, []).append(values)
    for name, parts in stages.items():
        values = np.concatenate(parts) * 1000.0
        base = _base(sess, run, "profile", Cell(cell.group, "", cell.split, cell.horizon, cell.index, cell.meta), "stage call")
        base["condition"] = name
        for q, label in ((0.5, "stage_latency_p50_ms"), (0.99, "stage_latency_p99_ms")):
            v, lo, hi = ops.quantile_interval(values, q, confidence=conf)
            sess.plain_row(base, label, v, lo, hi, "distribution-free order-statistic interval", len(parts),
                           {"n_units": float(values.size), "n_events": 0.0})
    for attr, label, scale in (("flops_per_update", "flops_per_update", 1.0), ("peak_memory_bytes", "peak_memory_gib",
                                                                              1.0 / ops.GIB)):
        vals = np.array([float(getattr(o, attr)) * scale for o in recs if getattr(o, attr) is not None])
        if vals.size == 0:
            continue
        base = _base(sess, run, "profile", cell, "run")
        if vals.size == 1:
            sess.plain_row(base, label, float(vals[0]), math.nan, math.nan, "single measurement (no interval)", 1,
                           {"n_units": 1.0, "n_events": 0.0})
        else:
            agg = sig.aggregate_seeds(vals, confidence=conf)
            sess.plain_row(base, label, agg.mean, agg.low, agg.high, "Student t over repeated runs", agg.n_seeds,
                           {"n_units": float(vals.size), "n_events": 0.0}, seed_sd=agg.sd)
