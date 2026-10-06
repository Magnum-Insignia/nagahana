"""Configuration of every analysis: typed frozen dataclasses, the single source of truth.

Each analysis reads one dataclass. The files `conf/analytics/<kind>.yaml` are generated from these
dataclasses by `render_yaml` (`write_yaml_files` writes all of them) and carry exactly their defaults,
with each field's documentation as a comment; `check_yaml_file` verifies that a file still matches its
dataclass (same keys, same values, no `???`). A YAML file given at run time overrides any subset of
the fields; an unknown key, a value of the wrong type or the undecided marker `???` raises instead of
being ignored (core/config.py convention). Every default is an estimator setting documented in the
module that uses it, never a held design decision.
"""

from __future__ import annotations

import dataclasses
import types
import typing
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

from nagahana.core.config import MISSING, load_yaml
from nagahana.core.errors import ConfigMissing, InvariantViolation


def _f(default: Any, doc: str) -> Any:
    """A dataclass field with a default and its documentation (rendered as the YAML comment)."""
    return field(default=default, metadata={"doc": doc})


@dataclass(frozen=True)
class EDAConfig:
    """Exploratory analysis (`analytics.eda`)."""

    group_by: tuple[str, ...] = _f(("label",), "extra split of value distributions: label, dataset, network, split, family")
    max_bins: int = _f(64, "histogram bins at most (Freedman-Diaconis width, clipped)")
    top_categories: int = _f(20, "categories listed per categorical field")
    trim: float = _f(0.1, "trimmed-mean proportion cut from each tail")
    tail_k_min: int = _f(10, "smallest number of upper order statistics on the Hill path")
    tail_max_share: float = _f(0.5, "largest share of the positive sample on the Hill path")
    tail_k_cap: int = _f(100_000, "hard bound on k for the Kolmogorov-Smirnov scan")
    tail_grid: int = _f(128, "values of k on the (geometric) Hill path")
    tail_level: float = _f(0.95, "confidence level of the Hill intervals")
    pearson_transform: str = _f("slog1p", "transform before Pearson's r: slog1p (AS-31) or none")
    dependence_methods: tuple[str, ...] = _f(("pearson", "spearman", "dcor", "mi"), "pairwise measures computed")
    dependence_max_rows: int = _f(20_000, "seeded row sample for the pairwise measures")
    dependence_min_pairs: int = _f(50, "pairs with fewer pairwise-complete records are not estimated")
    mi_k: int = _f(3, "neighbours of the k-NN mutual-information estimators")
    dcor_permutations: int = _f(0, "permutations for distance-correlation p-values (0: none)")
    patterns_top: int = _f(25, "most frequent status patterns listed")
    near_tolerance: float = _f(0.05, "near duplicates: tolerance on the slog1p scale")
    near_threshold: float = _f(0.95, "near duplicates: share of matching columns")
    near_bands: int = _f(24, "near duplicates: LSH bands")
    near_rows_per_band: int = _f(8, "near duplicates: columns sampled per band")
    near_max_bucket: int = _f(64, "near duplicates: buckets above this size are verified against pivots")
    near_pivots: int = _f(8, "near duplicates: pivots per oversized bucket")
    near_max_rows: int = _f(500_000, "near duplicates: seeded row sample above this size")
    seed: int = _f(0, "seed of every random choice of the analysis")


@dataclass(frozen=True)
class TDAConfig:
    """Topological data analysis (`analytics.tda`)."""

    max_dimension: int = _f(2, "homology dimensions H_0 ... H_max_dimension")
    max_radius: float | None = _f(None, "Rips radius; null: radius_quantile, else the degree budget, else the enclosing radius")
    radius_quantile: float | None = _f(None, "quantile of pairwise distances used as the radius (null: unused)")
    max_mean_degree: float = _f(32.0, "without a radius: the radius where the mean vertex degree reaches this (capped by the enclosing radius)")
    metric: str = _f("euclidean", "point metric: euclidean, chebyshev or cityblock")
    n_landmarks: int | None = _f(None, "maxmin landmarks (null: every point up to cloud_max_points)")
    cloud_max_points: int = _f(2_000, "larger clouds are reduced to this many maxmin landmarks")
    cloud_sample_rows: int = _f(50_000, "seeded uniform row sample before landmark selection")
    max_simplices: int = _f(5_000_000, "refuse complexes larger than this")
    algorithm: str = _f("cohomology", "reduction: cohomology or homology")
    clearing: bool = _f(True, "clearing (twist) optimisation")
    apparent_pairs: bool = _f(True, "apparent-pair shortcut")
    min_persistence: float = _f(0.0, "bars with smaller persistence are dropped from the outputs")
    landscape_k: int = _f(5, "persistence landscape levels")
    landscape_resolution: int = _f(256, "landscape grid points")
    image_resolution: int = _f(32, "persistence image pixels per side")
    image_sigma: float | None = _f(None, "persistence image kernel width (null: 1/30 of the largest persistence)")
    betti_resolution: int = _f(256, "Betti curve grid points")
    wasserstein_p: float = _f(2.0, "order p of the Wasserstein distance")
    ground_norm: str = _f("inf", "ground norm of diagram distances: inf or 2")
    sw_dimension: int | None = _f(None, "sliding-window dimension (null: false nearest neighbours)")
    sw_delay: int | None = _f(None, "sliding-window delay in samples (null: first minimum of the average mutual information)")
    sw_stride: int = _f(1, "sliding-window stride in samples")
    sw_max_points: int = _f(150, "sliding-window points at most (maxmin landmarks of the normalised window cloud)")
    sw_max_dimension: int = _f(12, "largest dimension tried by false nearest neighbours")
    sw_max_delay: int = _f(64, "largest delay tried by the average mutual information")
    sw_bin_seconds: float = _f(1.0, "bin of the update-count series embedded by the sliding window")
    sw_max_series: int = _f(20_000, "longer count series are aggregated into this many bins")
    features: tuple[str, ...] = _f((), "corpus columns of the point cloud (empty: numeric columns contributing in at least half of the rows)")
    window_seconds: float = _f(60.0, "contact-topology windows when the corpus has no window records")
    max_windows: int = _f(500, "windows analysed at most (evenly spread over the corpus)")
    seed: int = _f(0, "seed of every random choice of the analysis")


@dataclass(frozen=True)
class SpatialConfig:
    """Graph analytics (`analytics.spatial`)."""

    weighted: bool = _f(False, "use edge weights as lengths for shortest paths")
    betweenness_sources: int | None = _f(None, "null: exact betweenness; else sampled sources (Brandes and Pich 2007)")
    betweenness_batch: int = _f(64, "sources processed together")
    pagerank_alpha: float = _f(0.85, "PageRank damping")
    pagerank_tol: float = _f(1e-12, "PageRank L1 tolerance")
    pagerank_max_iter: int = _f(10_000, "PageRank iterations at most")
    eigen_tol: float = _f(1e-12, "eigenvector centrality tolerance")
    eigen_max_iter: int = _f(10_000, "eigenvector centrality iterations at most")
    louvain_resolution: float = _f(1.0, "modularity resolution gamma")
    louvain_tol: float = _f(1e-10, "smallest modularity gain that counts as an improvement")
    louvain_max_levels: int = _f(50, "aggregation levels at most")
    louvain_max_sweeps: int = _f(1_000, "local-moving sweeps per level at most")
    motif_null_samples: int = _f(0, "configuration-model samples for motif z-scores (0: none)")
    spectral_k: int = _f(50, "Laplacian eigenvalues compared by the spectral distance")
    heat_times: tuple[float, ...] = _f((0.01, 0.0316, 0.1, 0.316, 1.0, 3.16, 10.0, 31.6, 100.0), "heat-trace times")
    max_dense_nodes: int = _f(4_000, "dense eigendecomposition up to this many nodes, Lanczos quadrature above")
    slq_vectors: int = _f(32, "stochastic Lanczos quadrature probe vectors")
    slq_steps: int = _f(48, "Lanczos steps per probe")
    window_seconds: float = _f(60.0, "per-window graphs of a corpus (the AS-12 cadence)")
    seed: int = _f(0, "seed of every random choice of the analysis")


@dataclass(frozen=True)
class TemporalConfig:
    """Temporal analytics (`analytics.temporal`)."""

    bin_seconds: float = _f(60.0, "count series resolution in seconds (the AS-12 cadence)")
    acf_lags: int = _f(60, "lags of the ACF and PACF")
    welch_segment: int = _f(256, "Welch segment length in bins")
    periodicity_alpha: float = _f(0.01, "level of Fisher's g test")
    adf_regression: str = _f("c", "ADF deterministic terms: n, c, ct or ctt")
    adf_autolag: str = _f("aic", "ADF lag selection: aic, bic, t-stat or none")
    adf_max_lag: int | None = _f(None, "ADF largest lag (null: floor(12 (T/100)^(1/4)), Schwert 1989)")
    kpss_regression: str = _f("c", "KPSS deterministic terms: c or ct")
    kpss_lags: str = _f("nw1994", "KPSS bandwidth: nw1994, schwert or an integer")
    pelt_cost: str = _f("normal_meanvar", "PELT cost: normal_mean, normal_meanvar, poisson or exponential")
    pelt_penalty: str = _f("bic", "PELT penalty: bic, aic or a number")
    pelt_min_size: int = _f(3, "PELT minimum segment length (at least 3 for normal_meanvar)")
    bocpd_hazard: float = _f(1.0 / 250.0, "BOCPD constant hazard (1 / expected run length)")
    bocpd_model: str = _f("normal", "BOCPD observation model: normal or poisson")
    bocpd_prune: float = _f(1e-8, "run lengths with smaller posterior mass are pruned")
    bocpd_max_run: int = _f(5_000, "run lengths kept at most")
    dfa_order: int = _f(1, "DFA detrending order")
    dfa_min_scale: int = _f(8, "DFA smallest box")
    dfa_max_scale_share: float = _f(0.25, "DFA largest box as a share of the series")
    dfa_scales: int = _f(24, "DFA box sizes (geometric)")
    entity_min_events: int = _f(10, "entities with fewer events get no burstiness")
    max_entities: int = _f(500, "entities listed in the burstiness table (most active first)")
    seed: int = _f(0, "seed of every random choice of the analysis")


@dataclass(frozen=True)
class LeakageConfig:
    """Label and leakage audits (`analytics.leakage`)."""

    auroc_flag: float = _f(0.95, "single-feature separability at or above this is flagged")
    oof_folds: int = _f(5, "folds of the out-of-fold category rates")
    oof_smoothing: float = _f(1.0, "additive smoothing of the out-of-fold category rates")
    near_tolerance: float = _f(0.05, "near duplicates: tolerance on the slog1p scale")
    near_threshold: float = _f(0.95, "near duplicates: share of matching columns")
    near_bands: int = _f(24, "near duplicates: LSH bands")
    near_rows_per_band: int = _f(8, "near duplicates: columns sampled per band")
    near_max_bucket: int = _f(64, "near duplicates: buckets above this size are verified against pivots")
    near_pivots: int = _f(8, "near duplicates: pivots per oversized bucket")
    near_max_rows: int = _f(500_000, "near duplicates: seeded row sample above this size")
    seed: int = _f(0, "seed of every random choice of the analysis")


@dataclass(frozen=True)
class DriftConfig:
    """Covariate, label and concept drift (`analytics.drift`)."""

    mmd_max_rows: int = _f(2_000, "rows per sample for the MMD test (seeded)")
    mmd_permutations: int = _f(500, "permutations of the MMD test")
    ks_alpha: float = _f(0.05, "false discovery rate of the per-feature tests")
    fdr: str = _f("bh", "multiple-testing procedure: bh (Benjamini-Hochberg) or by (Benjamini-Yekutieli)")
    psi_bins: int = _f(10, "population stability index bins (reference quantiles)")
    psi_epsilon: float = _f(1e-4, "floor of empty PSI bins")
    adwin_delta: float = _f(0.002, "ADWIN confidence")
    adwin_max_buckets: int = _f(5, "ADWIN buckets per size")
    ph_delta: float = _f(0.005, "Page-Hinkley tolerated change")
    ph_threshold: float = _f(50.0, "Page-Hinkley alarm threshold")
    ddm_min_samples: int = _f(30, "DDM samples before testing")
    ddm_warning: float = _f(2.0, "DDM warning level (standard deviations)")
    ddm_drift: float = _f(3.0, "DDM drift level (standard deviations)")
    stream_bin_seconds: float = _f(60.0, "bin of the streamed statistics in seconds")
    bbse_l2: float = _f(1e-3, "L2 penalty of the black-box classifier fitted for BBSE")
    seed: int = _f(0, "seed of every random choice of the analysis")


@dataclass(frozen=True)
class ObservabilityConfig:
    """Observability analysis (`analytics.observability`)."""

    bin_seconds: float = _f(3_600.0, "coverage bins in seconds")
    gap_drop: float = _f(0.5, "a bin is a gap when coverage falls below this share of the field's median")
    gap_min_bins: int = _f(1, "shortest reported gap in bins")
    seed: int = _f(0, "seed of every random choice of the analysis")


@dataclass(frozen=True)
class InfoAuditConfig:
    """Information audit (`lab.info_audit`, P-15)."""

    enabled_proposals: tuple[str, ...] = _f((), "proposals enabled for this run; P-15 must be listed for the audit to run")
    k_values: tuple[int, ...] = _f((3, 5, 10), "neighbours of the k-NN estimators; the point estimate is the median over them")
    min_stratum: int = _f(20, "smallest observation-pattern stratum estimated by k-NN")
    correction: str = _f("miller_madow", "discrete entropy correction: miller_madow or none")
    jitter: float = _f(1e-10, "tie-breaking noise in standard deviations")
    n_sub: int = _f(20, "half-sample replicates of the subsampling interval")
    level: float = _f(0.95, "interval level")
    nn_max_rows: int = _f(50_000, "seeded row sample for the nearest-neighbour error")
    lags: tuple[int, ...] = _f((0,), "forecast lags k of I(O_t; S_t+k)")
    seed: int = _f(0, "seed of every random choice of the analysis")


#: Analysis kind -> configuration class.
CONFIGS: dict[str, type] = {
    "eda": EDAConfig, "tda": TDAConfig, "spatial": SpatialConfig, "temporal": TemporalConfig,
    "leakage": LeakageConfig, "drift": DriftConfig, "observability": ObservabilityConfig,
    "info-audit": InfoAuditConfig,
}

T = TypeVar("T")


def _coerce(name: str, value: Any, hint: Any) -> Any:
    """Check and convert one YAML value against the field's type hint."""
    if isinstance(value, str) and value == MISSING:
        raise ConfigMissing(f"config key {name!r} is '???' (undecided); set it explicitly")
    origin = typing.get_origin(hint)
    args = typing.get_args(hint)
    if origin in (typing.Union, types.UnionType):
        if value is None and type(None) in args:
            return None
        errors = []
        for a in args:
            if a is type(None):
                continue
            try:
                return _coerce(name, value, a)
            except (TypeError, ValueError) as exc:
                errors.append(str(exc))
        raise TypeError(f"config key {name!r}: {value!r} matches none of {hint} ({'; '.join(errors)})")
    if origin is tuple:
        if not isinstance(value, list | tuple):
            raise TypeError(f"config key {name!r} must be a list, got {type(value).__name__}")
        item = args[0] if args else Any
        return tuple(_coerce(name, v, item) for v in value)
    if hint is bool:
        if not isinstance(value, bool):
            raise TypeError(f"config key {name!r} must be true or false")
        return value
    if hint is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"config key {name!r} must be an integer")
        return value
    if hint is float:
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise TypeError(f"config key {name!r} must be a number")
        return float(value)
    if hint is str:
        if not isinstance(value, str):
            raise TypeError(f"config key {name!r} must be a string")
        return value
    return value


def from_mapping(cls: type[T], data: Mapping[str, Any]) -> T:
    """Build a config dataclass from a mapping; unknown keys, wrong types and `???` raise."""
    if not dataclasses.is_dataclass(cls):
        raise TypeError(f"{cls} is not a dataclass")
    hints = typing.get_type_hints(cls)
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = sorted(set(data) - known)
    if unknown:
        raise ValueError(f"unknown keys for {cls.__name__}: {unknown}; known: {sorted(known)}")
    kwargs = {k: _coerce(k, v, hints[k]) for k, v in data.items()}
    return cls(**kwargs)


def _file_mapping(kind: str, raw: Mapping[str, Any]) -> dict[str, Any]:
    """The analysis section of a loaded YAML file: a top-level key named after the kind, or the whole file."""
    key = kind.replace("-", "_")
    section = raw.get(key, raw)
    if not isinstance(section, Mapping):
        raise ValueError(f"the {key!r} section of the config must be a mapping")
    return dict(section)


def load_config(kind: str, path: str | Path | None = None, overrides: Mapping[str, Any] | None = None) -> Any:
    """The configuration of an analysis kind: dataclass defaults, then the YAML file, then `overrides`."""
    if kind not in CONFIGS:
        raise KeyError(f"unknown analysis kind {kind!r}; known: {sorted(CONFIGS)}")
    data: dict[str, Any] = {}
    if path is not None:
        data.update(_file_mapping(kind, load_yaml(path)))
    if overrides:
        data.update(overrides)
    return from_mapping(CONFIGS[kind], data)


def to_dict(cfg: Any) -> dict[str, Any]:
    """A config as a plain dict (for report provenance)."""
    return dataclasses.asdict(cfg)


def _yaml_value(value: Any) -> Any:
    """Plain YAML-serialisable form of a default (tuples become lists)."""
    if isinstance(value, tuple):
        return [_yaml_value(v) for v in value]
    return value


def render_yaml(kind: str) -> str:
    """The YAML text of an analysis kind's defaults: one key per field, its documentation as a comment."""
    import yaml

    if kind not in CONFIGS:
        raise KeyError(f"unknown analysis kind {kind!r}; known: {sorted(CONFIGS)}")
    cls = CONFIGS[kind]
    defaults = cls()
    key = kind.replace("-", "_")
    lines = [
        f"# {cls.__doc__.strip() if cls.__doc__ else kind}",
        f"# Generated from nagahana.analytics.config.{cls.__name__} by render_yaml; the dataclass is the single",
        "# source of truth. Every value equals its dataclass default (check_yaml_file verifies it).",
        f"{key}:",
    ]
    for f in dataclasses.fields(cls):
        doc = str(f.metadata.get("doc", "")).strip()
        if doc:
            lines.append(f"  # {doc}")
        dumped = yaml.safe_dump({f.name: _yaml_value(getattr(defaults, f.name))}, default_flow_style=True,
                                sort_keys=False, width=1_000).strip()
        if dumped.startswith("{") and dumped.endswith("}"):
            dumped = dumped[1:-1]
        lines.append(f"  {dumped}")
    return "\n".join(lines) + "\n"


def write_yaml_files(directory: str | Path) -> list[Path]:
    """Write `render_yaml(kind)` for every kind to `<directory>/<kind>.yaml`; returns the paths."""
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    out = []
    for kind in CONFIGS:
        p = root / f"{kind.replace('-', '_')}.yaml"
        p.write_text(render_yaml(kind), encoding="utf-8")
        out.append(p)
    return out


def check_yaml_file(kind: str, path: str | Path) -> None:
    """Raise unless the file holds exactly the dataclass's keys with its default values and no `???`."""
    cls = CONFIGS[kind]
    data = _file_mapping(kind, load_yaml(path))
    known = [f.name for f in dataclasses.fields(cls)]
    if sorted(data) != sorted(known):
        raise InvariantViolation(f"{path}: keys {sorted(data)} differ from {cls.__name__} fields {sorted(known)}")
    loaded: Any = from_mapping(cls, data)                               # raises on '???' and on wrong types
    if loaded != cls():
        diff = [k for k in known if getattr(loaded, k) != getattr(cls(), k)]
        raise InvariantViolation(f"{path}: values of {diff} differ from the {cls.__name__} defaults")


__all__ = [
    "CONFIGS", "DriftConfig", "EDAConfig", "InfoAuditConfig", "LeakageConfig", "ObservabilityConfig",
    "SpatialConfig", "TDAConfig", "TemporalConfig", "check_yaml_file", "from_mapping", "load_config", "render_yaml",
    "to_dict", "write_yaml_files",
]
