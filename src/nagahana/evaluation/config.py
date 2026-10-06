"""Evaluation settings as typed dataclasses: the single source of truth for conf/evaluation/evaluation.yaml.

The YAML file is generated from these defaults (`write_yaml`, `python -m nagahana evaluate config
--write`) and every file read with `load_config` is validated against the dataclasses: unknown keys,
missing sections and values of the wrong type are rejected, so the file and the code cannot drift
apart (a test compares the checked-in file with the generated text).

The defaults are the reporting configuration of the thesis results chapter where it states one: the
false-alarm rate alpha = 0.10 % of the conformal threshold, the horizons k = 1, ceil(K / 2) and K, the
N = 5 most probable paths and the path tolerance epsilon = 0.25, 95 % confidence intervals and
false-discovery-rate control at level 0.05 with the Benjamini-Hochberg procedure. The other values are
engineering assumptions recorded in docs/assumptions/evaluation.md (AS-630 onward). Held design
decisions appear as protocol settings: whether zero-shot "novel" includes unseen networks (D-16) is
`protocols.p1_novelty`, set to the reading of AS-35.
"""

from __future__ import annotations

import dataclasses
import types
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from nagahana.core.errors import InvariantViolation

SCHEMES = ("iid", "stratified", "stationary", "cluster")
TIERS = ("inference", "single_stage", "full")


@dataclass(frozen=True)
class ResamplingConfig:
    """Bootstrap intervals (resampling.py)."""

    n_resamples: int = 2000
    confidence: float = 0.95
    interval: str = "bca"
    jackknife_groups: int = 100
    chunk_elements: int = 20_000_000
    mean_block_length: float | None = None
    heavy_resamples: int = 200


@dataclass(frozen=True)
class SchemeConfig:
    """Resampling scheme per task: iid, stratified, stationary (per dataset and network series) or cluster."""

    detection: str = "stationary"
    forecast: str = "stationary"
    state_forecast: str = "stationary"
    stage: str = "stationary"
    time_to_event: str = "stationary"
    paths: str = "cluster"
    timeliness: str = "cluster"
    forensics: str = "cluster"
    components: str = "iid"
    explanations: str = "stationary"


@dataclass(frozen=True)
class OperatingPointConfig:
    """Decision thresholds: the model's own, the conformal threshold at alpha, or a matched FPR."""

    alpha: float = 0.001
    detection: str = "own"
    timeliness_own: str = "own"
    missing_own: str = "conformal"
    calibration_split: str = "val"
    conformal_scope: str = "dataset"


@dataclass(frozen=True)
class DetectionConfig:
    """Threshold-free and operational detection metrics."""

    partial_auroc_max_fpr: float = 0.01
    fixed_fprs: tuple[float, ...] = (0.001, 0.01)
    deployment_base_rates: tuple[float, ...] = (0.001, 0.01)


@dataclass(frozen=True)
class ForecastConfig:
    """Probabilistic forecast verification."""

    horizons: tuple[str, ...] = ("1", "mid", "K")
    ece_bins: int = 10
    ece_strategy: str = "width"
    log_score_eps: float | None = None
    climatology_scope: str = "dataset"
    fair_crps: bool = True
    reliability_bins: int = 10
    temperatures: tuple[float, ...] = (0.5, 0.75, 1.0, 1.5, 2.0)


@dataclass(frozen=True)
class TimelinessConfig:
    """Lead time and episode metrics (episodes.py)."""

    late_horizons: float = 1.0
    match_entity: bool = True
    window_start: str = "horizon"
    quantiles: tuple[float, ...] = (0.1, 0.25, 0.75, 0.9)


@dataclass(frozen=True)
class SurvivalConfig:
    """Time-to-event metrics (survival.py)."""

    tied_times: str = "exclude"
    interpolation: str = "step"
    marker: str = "survival"
    dcal_bins: int = 10


@dataclass(frozen=True)
class StageConfig:
    """ATT&CK stage metrics (stages.py)."""

    top_k: tuple[int, ...] = (1, 3)
    classes: str = "union"
    ece_bins: int = 10
    ace_ranges: int = 15


@dataclass(frozen=True)
class PathConfig:
    """Attack-path metrics (paths.py)."""

    top_n: int = 5
    tolerance: float = 0.25
    entity_rule: str = "exact"
    include_empty: bool = False


@dataclass(frozen=True)
class ComparisonConfig:
    """Paired tests and multiple-comparison control (significance.py)."""

    fdr_method: str = "bh"
    fdr_level: float = 0.05
    mcnemar_mid_p: bool = True
    dm_lags: int | None = None
    wilcoxon_zero_method: str = "wilcox"
    permutations: int = 10_000


@dataclass(frozen=True)
class ModelRoles:
    """Which model names play which role; display names for the tables."""

    primary: str = "nagahana"
    lr_family: tuple[str, ...] = ("logistic_regression",)
    display: dict[str, str] = field(default_factory=lambda: {
        "nagahana": r"\NagaHana{}", "logistic_regression": "LR", "persistence": "Persistence",
        "climatology": "Climatology", "cyberworld": "CyberWorld (RSSM)", "flowtransformer": "FlowTransformer"})


@dataclass(frozen=True)
class ForensicsConfig:
    """Forensic backtesting (forensics.py)."""

    top_k: int = 3
    tolerance_windows: float = 1.0
    entity_rule: str = "exact"


@dataclass(frozen=True)
class OperationsConfig:
    """Operational load (operations.py)."""

    batches: int = 20


@dataclass(frozen=True)
class DatasetInfo:
    """A dataset: identifier in ModelOutputs meta, labels, report group, lead-time unit, window length."""

    id: str
    label: str
    short: str
    group: str
    time_unit: str
    window_seconds: float


def _default_datasets() -> tuple[DatasetInfo, ...]:
    # Order and labels of the thesis tables; the OT testbeds share the report group "ot".
    return (
        DatasetInfo("cse-cic-ids2018", "CSE-CIC-IDS2018", "CIC-18", "cse-cic-ids2018", "min", 60.0),
        DatasetInfo("ctu-13", "CTU-13", "CTU-13", "ctu-13", "min", 60.0),
        DatasetInfo("cic-ids2017", "CIC-IDS2017", "CIC-17", "cic-ids2017", "min", 60.0),
        DatasetInfo("unsw-nb15", "UNSW-NB15", "UNSW", "unsw-nb15", "min", 60.0),
        DatasetInfo("cic-iot-2023", "CIC-IoT-2023", "CIC-IoT", "cic-iot-2023", "min", 60.0),
        DatasetInfo("lanl", "LANL", "LANL", "lanl", "h", 1800.0),
        DatasetInfo("swat", "SWaT", "SWaT", "ot", "min", 30.0),
        DatasetInfo("wadi", "WADI", "WADI", "ot", "min", 30.0),
        DatasetInfo("hai", "HAI", "HAI", "ot", "min", 30.0),
        DatasetInfo("morris-gao-ics", "Morris-Gao ICS", "Morris-Gao", "ot", "min", 30.0),
        DatasetInfo("ics-flow", "ICS-Flow", "ICS-Flow", "ot", "min", 30.0),
        DatasetInfo("simulated", "Simulated worlds", "Simulated", "simulated", "min", 60.0),
    )


@dataclass(frozen=True)
class GroupInfo:
    """A report group (one block or column of the tables)."""

    id: str
    label: str
    short: str


def _default_groups() -> tuple[GroupInfo, ...]:
    return (
        GroupInfo("cse-cic-ids2018", "CSE-CIC-IDS2018", "CIC-18"),
        GroupInfo("ctu-13", "CTU-13", "CTU-13"),
        GroupInfo("cic-ids2017", "CIC-IDS2017", "CIC-17"),
        GroupInfo("unsw-nb15", "UNSW-NB15", "UNSW"),
        GroupInfo("cic-iot-2023", "CIC-IoT-2023", "CIC-IoT"),
        GroupInfo("lanl", "LANL", "LANL"),
        GroupInfo("ot", "OT datasets", "OT"),
        GroupInfo("simulated", "Simulated worlds", "Simulated"),
    )


@dataclass(frozen=True)
class ProtocolSettings:
    """Settings of protocols P1 to P8 (protocols.py)."""

    p1_split: str = "zero_shot"
    p1_novelty: str = "families_and_networks"
    p2_test_splits: tuple[str, ...] = ("test", "zero_shot")
    p3_site_benign_hours: float = 24.0
    p3_site_analyst_alerts: int = 50
    p4_reference_regime: str = "full"
    p4_regimes: tuple[str, ...] = ("full", "flow_packet", "flow_only", "sampled", "encrypted", "untapped")
    p4_dataset: str = "cse-cic-ids2018"
    p4_novelty: str = "known"
    p5_corruptions: tuple[str, ...] = ("skew", "reorder", "drop", "dup", "spoof", "flood")
    p5_share: float = 0.10
    p6_enabled_proposals: tuple[str, ...] = ()
    p6_miller_madow: bool = True
    p8_telemetry: tuple[str, ...] = ("full", "flow_only")


@dataclass(frozen=True)
class HeadlineSpec:
    """One headline comparison of the interval table (res-ci)."""

    id: str
    label: str
    protocol: str
    task: str
    metric: str
    group: str
    novelty: str
    variant: str
    horizon: str
    test: str
    percent: bool
    decimals: int


def _default_headline() -> tuple[HeadlineSpec, ...]:
    # The comparisons of the thesis interval table, in its order.
    return (
        HeadlineSpec("f1kn", r"F1 score (\%), CSE-CIC-IDS2018, known families", "P1", "detection", "f1",
                     "cse-cic-ids2018", "known", "", "", "mcnemar", True, 2),
        HeadlineSpec("f1nov", r"F1 score (\%), CSE-CIC-IDS2018, novel families", "P1", "detection", "f1",
                     "cse-cic-ids2018", "novel", "", "", "mcnemar", True, 2),
        HeadlineSpec("aurocnov", r"AUROC (\%), CSE-CIC-IDS2018, novel families", "P1", "detection", "auroc",
                     "cse-cic-ids2018", "novel", "", "", "delong", True, 2),
        HeadlineSpec("x1718", r"F1 score (\%), CIC-IDS2017 $\rightarrow$ CSE-CIC-IDS2018", "P2", "detection", "f1",
                     "cse-cic-ids2018", "", "train=cic-ids2017", "", "mcnemar", True, 2),
        HeadlineSpec("x1817", r"F1 score (\%), CSE-CIC-IDS2018 $\rightarrow$ CIC-IDS2017", "P2", "detection", "f1",
                     "cic-ids2017", "", "train=cse-cic-ids2018", "", "mcnemar", True, 2),
        HeadlineSpec("bsK", r"Brier score at the full horizon $K$", "P1", "forecast", "brier", "all", "known", "",
                     "K", "diebold_mariano", False, 3),
        HeadlineSpec("lt", r"Median lead time (min), CSE-CIC-IDS2018", "P7", "timeliness", "median_lead_time",
                     "cse-cic-ids2018", "known", "", "", "wilcoxon", False, 1),
    )


@dataclass(frozen=True)
class AblationSpec:
    """One column of the ablation table: a metric on a cell of the base protocol, its scale and its unit."""

    id: str
    label: str
    base: str
    task: str
    metric: str
    group: str
    novelty: str
    variant: str
    horizon: str
    scale: float
    decimals: int


def _default_ablation_columns() -> tuple[AblationSpec, ...]:
    return (
        AblationSpec("kn", "known", "P1", "detection", "f1", "cse-cic-ids2018", "known", "", "", 100.0, 2),
        AblationSpec("nov", "novel", "P1", "detection", "f1", "cse-cic-ids2018", "novel", "", "", 100.0, 2),
        AblationSpec("cr", "cross", "P2", "detection", "f1", "cse-cic-ids2018", "", "train=cic-ids2017", "", 100.0, 2),
        AblationSpec("ap", "novel", "P1", "detection", "auprc", "cse-cic-ids2018", "novel", "", "", 100.0, 2),
        AblationSpec("bs", r"$\times10^{-3}$", "P1", "forecast", "brier", "all", "known", "", "K", 1000.0, 1),
        AblationSpec("lt", "min", "P1", "timeliness", "median_lead_time", "cse-cic-ids2018", "known", "", "", 1.0 / 60.0, 1),
    )


@dataclass(frozen=True)
class AblationInfo:
    """One registered ablation: an L model with one component removed or one setting changed.

    tier       "inference" (no retraining), "single_stage" (one training stage repeated) or "full" (all
               training stages repeated)
    thesis_id  the identifier of the thesis ablation table when the ablation appears there, else ""
    """

    id: str
    tier: str
    component: str
    removal: str
    thesis_id: str


def _default_ablations() -> tuple[AblationInfo, ...]:
    # Inference-time ablations change a run-time budget or switch a module off without retraining;
    # single-stage retrains repeat one training stage; full retrains repeat every stage (architecture changes).
    lens = ("belief_trust", "game", "information", "topology", "time", "cause")
    return (
        AblationInfo("INF-R1", "inference", "TSTCT and TAAFT loops", "one thinking pass (R = 1)", ""),
        AblationInfo("INF-D0", "inference", "TAAFT energy refinement", "no descent steps", "A8"),
        AblationInfo("INF-ENV", "inference", "Environment memory", "TSTCT reads only the current window", ""),
        AblationInfo("INF-LTM", "inference", "Long-term memory", "long-term memory not read", ""),
        AblationInfo("INF-PHYS", "inference", "Physics boundary", "Phi_phys term and hard limits off at inference", ""),
        AblationInfo("INF-N1", "inference", "Forecaster routes", "one imagined route (N = 1)", ""),
        *(AblationInfo(f"SS-LENS-{name.upper()}", "single_stage", "TAAFT lens", f"{name} term removed from E_total", "")
          for name in lens),
        AblationInfo("SS-CAUSAL", "single_stage", "TSTCT causal heads", "causal heads off", ""),
        AblationInfo("SS-GEN", "single_stage", "Generator", "no Generator variants in training", "A10"),
        AblationInfo("SS-NOPRE", "single_stage", "TAAFT pretraining", "Stage 2 TAAFT pretraining skipped", ""),
        AblationInfo("SS-REWARD", "single_stage", "Process rewards", "outcome-only rewards", "A11"),
        AblationInfo("SS-COUPLING", "single_stage", "TAAFT and heads coupling", "joint or separate instead of staged", "A12"),
        AblationInfo("FR-PLANES1", "full", "CVG-AE planes", "one plane instead of six", ""),
        AblationInfo("FR-NOTEMP", "full", "TSTCT temporal heads", "no temporal heads", ""),
        AblationInfo("FR-PAIRWISE", "full", "Hypergraph", "pairwise graph instead of hyperedges", "A1"),
        AblationInfo("FR-COUPLING0", "full", "Cross-plane coupling", "coupling omega_pq = 0", "A2"),
        AblationInfo("FR-DETERMINISTIC", "full", "Variational latent", "deterministic encoder", "A3"),
        AblationInfo("FR-CONTINUOUS", "full", "Categorical latent factors", "continuous-only latent", "A4"),
        AblationInfo("FR-NOPHYSICS", "full", "Physics boundary", "no physics term and no hard limits in training", "A5"),
        AblationInfo("FR-ZEROFILL", "full", "Observation status", "zero-filling instead of status", "A6"),
        AblationInfo("FR-NOCAUSALTOPO", "full", "Causal heads and topology bias", "both removed", "A7"),
        AblationInfo("FR-FULLOBS", "full", "Partial observability", "full observability assumed", "A9"),
        AblationInfo("FR-STATIC", "full", "Dynamics", "static head on the same encoder", "A13"),
    )


@dataclass(frozen=True)
class MultiDatasetSpec:
    """A metric compared over datasets with the Friedman test and the Nemenyi post-hoc."""

    id: str
    task: str
    metric: str
    higher_is_better: bool
    horizon: str


def _default_multidataset() -> tuple[MultiDatasetSpec, ...]:
    return (
        MultiDatasetSpec("f1", "detection", "f1", True, ""),
        MultiDatasetSpec("auprc", "detection", "auprc", True, ""),
        MultiDatasetSpec("auroc", "detection", "auroc", True, ""),
        MultiDatasetSpec("brier_K", "forecast", "brier", False, "K"),
        MultiDatasetSpec("crps_K", "forecast", "crps", False, "K"),
        MultiDatasetSpec("c_index", "time_to_event", "c_index", True, ""),
        MultiDatasetSpec("macro_f1", "stage", "macro_f1", True, ""),
    )


@dataclass(frozen=True)
class ArenaConfig:
    """Arena (protocol P-CW) settings (arena.py)."""

    min_seeds: int = 3
    profile_points: int = 50
    control: str = "control"


@dataclass(frozen=True)
class EvaluationConfig:
    """Every setting of the evaluation (one section per concern)."""

    seed: int = 0
    resampling: ResamplingConfig = field(default_factory=ResamplingConfig)
    schemes: SchemeConfig = field(default_factory=SchemeConfig)
    operating_point: OperatingPointConfig = field(default_factory=OperatingPointConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    forecast: ForecastConfig = field(default_factory=ForecastConfig)
    timeliness: TimelinessConfig = field(default_factory=TimelinessConfig)
    survival: SurvivalConfig = field(default_factory=SurvivalConfig)
    stages: StageConfig = field(default_factory=StageConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    comparisons: ComparisonConfig = field(default_factory=ComparisonConfig)
    models: ModelRoles = field(default_factory=ModelRoles)
    forensics: ForensicsConfig = field(default_factory=ForensicsConfig)
    operations: OperationsConfig = field(default_factory=OperationsConfig)
    arena: ArenaConfig = field(default_factory=ArenaConfig)
    protocols: ProtocolSettings = field(default_factory=ProtocolSettings)
    datasets: tuple[DatasetInfo, ...] = field(default_factory=_default_datasets)
    groups: tuple[GroupInfo, ...] = field(default_factory=_default_groups)
    headline: tuple[HeadlineSpec, ...] = field(default_factory=_default_headline)
    ablation_columns: tuple[AblationSpec, ...] = field(default_factory=_default_ablation_columns)
    ablations: tuple[AblationInfo, ...] = field(default_factory=_default_ablations)
    multidataset: tuple[MultiDatasetSpec, ...] = field(default_factory=_default_multidataset)

    def __post_init__(self) -> None:
        validate(self)

    def dataset(self, name: str) -> DatasetInfo:
        """The DatasetInfo of a dataset id; an unknown id is its own group with a minute unit and 60 s windows."""
        for d in self.datasets:
            if d.id == name:
                return d
        return DatasetInfo(name, name, name, name, "min", 60.0)

    def group(self, gid: str) -> GroupInfo:
        """The GroupInfo of a report group id (an unknown id labels itself)."""
        for g in self.groups:
            if g.id == gid:
                return g
        return GroupInfo(gid, gid, gid)

    def ablation(self, aid: str) -> AblationInfo | None:
        """The registered ablation with this id, or None."""
        for a in self.ablations:
            if a.id == aid:
                return a
        return None

    def display(self, model: str) -> str:
        """Display name of a model in the tables."""
        base, _, tag = model.partition("[")
        name = self.models.display.get(base, base)
        return f"{name} [{tag}" if tag else name


def validate(cfg: EvaluationConfig) -> None:
    """Check the admissible values of every setting (raises InvariantViolation)."""
    r = cfg.resampling
    _check(r.n_resamples >= 1 and r.heavy_resamples >= 1, "resampling counts must be >= 1")
    _check(0.0 < r.confidence < 1.0, "confidence must lie in (0, 1)")
    _check(r.interval in ("percentile", "bca"), "interval must be 'percentile' or 'bca'")
    _check(r.jackknife_groups >= 2 and r.chunk_elements >= 1, "jackknife_groups >= 2 and chunk_elements >= 1")
    _check(r.mean_block_length is None or r.mean_block_length >= 1.0, "mean_block_length must be >= 1 or null")
    for name in (f.name for f in dataclasses.fields(SchemeConfig)):
        _check(getattr(cfg.schemes, name) in SCHEMES, f"scheme {name} must be one of {SCHEMES}")
    o = cfg.operating_point
    _check(0.0 < o.alpha < 1.0, "operating_point.alpha must lie in (0, 1)")
    _check(o.detection in ("own", "conformal", "matched_fpr"), "operating_point.detection: own, conformal or matched_fpr")
    _check(o.timeliness_own in ("own", "conformal"), "operating_point.timeliness_own: own or conformal")
    _check(o.missing_own in ("conformal", "error"), "operating_point.missing_own: conformal or error")
    _check(o.conformal_scope in ("dataset", "pooled"), "operating_point.conformal_scope: dataset or pooled")
    d = cfg.detection
    _check(0.0 < d.partial_auroc_max_fpr <= 1.0, "partial_auroc_max_fpr must lie in (0, 1]")
    _check(all(0.0 < a < 1.0 for a in (*d.fixed_fprs, *d.deployment_base_rates)), "rates must lie in (0, 1)")
    f = cfg.forecast
    _check(f.ece_bins >= 1 and f.reliability_bins >= 1, "bins must be >= 1")
    _check(f.ece_strategy in ("width", "mass"), "ece_strategy: width or mass")
    _check(f.log_score_eps is None or 0.0 < f.log_score_eps < 0.5, "log_score_eps must lie in (0, 0.5) or be null")
    _check(f.climatology_scope in ("dataset", "pooled"), "climatology_scope: dataset or pooled")
    _check(all(h in ("mid", "K") or h.isdigit() for h in f.horizons), "horizons: positive integers, 'mid' or 'K'")
    _check(len(f.temperatures) >= 1 and all(t > 0 for t in f.temperatures), "temperatures must be positive")
    t = cfg.timeliness
    _check(t.late_horizons >= 0.0 and t.window_start in ("horizon", "episode"), "timeliness settings out of range")
    _check(all(0.0 < q < 1.0 for q in t.quantiles), "timeliness quantiles must lie in (0, 1)")
    s = cfg.survival
    _check(s.tied_times in ("exclude", "censored_later") and s.interpolation in ("step", "linear")
           and s.marker in ("survival", "risk") and s.dcal_bins >= 2, "survival settings out of range")
    st = cfg.stages
    _check(all(k >= 1 for k in st.top_k) and st.classes in ("union", "present", "all") and st.ece_bins >= 1
           and st.ace_ranges >= 1, "stage settings out of range")
    p = cfg.paths
    _check(p.top_n >= 1 and 0.0 <= p.tolerance <= 1.0 and p.entity_rule in ("exact", "stage"), "path settings out of range")
    c = cfg.comparisons
    _check(c.fdr_method in ("bh", "by", "holm", "bonferroni") and 0.0 < c.fdr_level < 1.0, "comparison settings out of range")
    _check(c.wilcoxon_zero_method in ("wilcox", "pratt", "zsplit") and c.permutations >= 1, "comparison settings out of range")
    _check(c.dm_lags is None or c.dm_lags >= 0, "dm_lags must be >= 0 or null")
    _check(cfg.forensics.top_k >= 1 and cfg.forensics.tolerance_windows >= 0 and cfg.forensics.entity_rule in ("exact", "stage"),
           "forensics settings out of range")
    _check(cfg.operations.batches >= 2, "operations.batches must be >= 2")
    pr = cfg.protocols
    _check(pr.p1_novelty in ("families", "families_and_networks"), "p1_novelty: families or families_and_networks")
    _check(0.0 < pr.p5_share <= 1.0, "p5_share must lie in (0, 1]")
    _check(len({d.id for d in cfg.datasets}) == len(cfg.datasets), "dataset ids must be unique")
    _check(all(d.time_unit in ("s", "min", "h") and d.window_seconds > 0 for d in cfg.datasets), "dataset units out of range")
    _check(len({h.id for h in cfg.headline}) == len(cfg.headline), "headline ids must be unique")
    _check(len({a.id for a in cfg.ablations}) == len(cfg.ablations), "ablation ids must be unique")
    _check(all(a.tier in TIERS for a in cfg.ablations), f"ablation tiers must be one of {TIERS}")
    _check(all(c.base in ("P1", "P2", "P3", "P4", "P5", "P6", "P7") for c in cfg.ablation_columns),
           "ablation columns read a base protocol P1 ... P7")
    _check(cfg.arena.min_seeds >= 3, "the arena needs at least three seeds per agent")
    _check(cfg.arena.profile_points >= 2, "arena.profile_points must be >= 2")
    _check(len({m.id for m in cfg.multidataset}) == len(cfg.multidataset), "multidataset ids must be unique")


def _check(ok: bool, message: str) -> None:
    if not ok:
        raise InvariantViolation(f"evaluation config: {message}")


def to_dict(obj: Any) -> Any:
    """Plain-data form of a config (tuples as lists) for YAML and JSON."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: to_dict(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, tuple | list):
        return [to_dict(v) for v in obj]
    if isinstance(obj, dict):
        return {str(k): to_dict(v) for k, v in obj.items()}
    return obj


def _convert(tp: Any, value: Any, path: str) -> Any:
    # Strict conversion of plain data to the annotated type (unions with None, tuples, dicts, dataclasses).
    origin = typing.get_origin(tp)
    if origin in (typing.Union, types.UnionType):
        args = typing.get_args(tp)
        if value is None and type(None) in args:
            return None
        for a in args:
            if a is type(None):
                continue
            try:
                return _convert(a, value, path)
            except InvariantViolation:
                continue
        raise InvariantViolation(f"evaluation config: {path} has the wrong type ({value!r})")
    if origin is tuple:
        args = typing.get_args(tp)
        if not isinstance(value, list | tuple):
            raise InvariantViolation(f"evaluation config: {path} must be a list")
        return tuple(_convert(args[0], v, f"{path}[{i}]") for i, v in enumerate(value))
    if origin is dict:
        _, vt = typing.get_args(tp)
        if not isinstance(value, dict):
            raise InvariantViolation(f"evaluation config: {path} must be a mapping")
        return {str(k): _convert(vt, v, f"{path}.{k}") for k, v in value.items()}
    if dataclasses.is_dataclass(tp):
        return from_dict(tp, value, path)
    if tp is bool:
        if not isinstance(value, bool):
            raise InvariantViolation(f"evaluation config: {path} must be true or false")
        return value
    if tp is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise InvariantViolation(f"evaluation config: {path} must be an integer")
        return value
    if tp is float:
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise InvariantViolation(f"evaluation config: {path} must be a number")
        return float(value)
    if tp is str:
        if not isinstance(value, str):
            raise InvariantViolation(f"evaluation config: {path} must be a string")
        if value == "???":
            raise InvariantViolation(f"evaluation config: {path} is '???'; every setting has a value")
        return value
    raise InvariantViolation(f"evaluation config: unsupported type at {path}")


def from_dict(cls: Any, data: Any, path: str = "evaluation") -> Any:
    """Build a config dataclass from plain data; every field must be present and of the annotated type."""
    if not isinstance(data, dict):
        raise InvariantViolation(f"evaluation config: {path} must be a mapping")
    hints = typing.get_type_hints(cls)
    names = [f.name for f in dataclasses.fields(cls)]
    unknown = sorted(set(data) - set(names))
    missing = sorted(set(names) - set(data))
    if unknown:
        raise InvariantViolation(f"evaluation config: unknown keys at {path}: {unknown}")
    if missing:
        raise InvariantViolation(f"evaluation config: missing keys at {path}: {missing}")
    return cls(**{n: _convert(hints[n], data[n], f"{path}.{n}") for n in names})


HEADER = (
    "# Evaluation settings (src/nagahana/evaluation/config.py, docs/evaluation.md).\n"
    "# Generated from the EvaluationConfig dataclasses: python -m nagahana evaluate config --write\n"
    "# conf/evaluation/evaluation.yaml. Edit the dataclasses, then regenerate; a test checks that this file\n"
    "# equals the generated text. Every value is set; assumptions are documented in\n"
    "# docs/assumptions/evaluation.md.\n"
)


def yaml_text(cfg: EvaluationConfig | None = None) -> str:
    """The YAML text of a configuration (the defaults when cfg is None), with the header comment."""
    import yaml

    body = yaml.safe_dump(to_dict(cfg if cfg is not None else EvaluationConfig()), sort_keys=False,
                          default_flow_style=False, allow_unicode=False, width=110)
    return HEADER + body


def write_yaml(path: str | Path, cfg: EvaluationConfig | None = None) -> Path:
    """Write the YAML of a configuration to path (creating the directory)."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml_text(cfg), encoding="utf-8")
    return out


def load_config(path: str | Path | None = None) -> EvaluationConfig:
    """Read and validate a YAML configuration; None gives the defaults."""
    if path is None:
        return EvaluationConfig()
    from nagahana.core.config import load_yaml

    cfg = from_dict(EvaluationConfig, load_yaml(path))
    assert isinstance(cfg, EvaluationConfig)
    return cfg


def default_path() -> Path:
    """conf/evaluation/evaluation.yaml of the repository."""
    return Path(__file__).resolve().parents[3] / "conf" / "evaluation" / "evaluation.yaml"
