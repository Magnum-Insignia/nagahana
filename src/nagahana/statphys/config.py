"""Configuration of the statistical-physics readouts and the early-warning module (D-56).

The dataclasses below are the single source of truth for every setting of `nagahana.statphys`.
`conf/statphys/statphys.yaml` is generated from them (`render_yaml`, command `statphys write-config`)
and read back by `load_config`, which accepts only the keys declared here, checks every value
against its declared type, rejects the "???" marker and applies the dataclass default for a key the
file leaves out. A test keeps the file identical to the rendered defaults, so file and code cannot
drift apart.

Every default is an engineering assumption recorded in docs/assumptions/statphys.md (AS-760 to
AS-779); the documentation of each field names its assumption.

Sections
    GibbsConfig           temperature of the Gibbs readouts, growth-rate window (AS-761, AS-763)
    TrafficConfig         Shannon entropies of traffic distributions (AS-765, AS-766, AS-778)
    SpectralConfig        von Neumann and spectral entropies of the multiplex graph (AS-767 to AS-771)
    EarlyWarningConfig    critical-slowing-down indicators, trend tests and the alarm (AS-772 to AS-778)
    StatPhysConfig        the module: the network-state window, the sections, the monitored series
"""

from __future__ import annotations

import json
import math
import types
import typing
from collections.abc import Mapping
from dataclasses import MISSING as DC_MISSING
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar

from nagahana.core.config import MISSING, load_yaml
from nagahana.core.errors import ConfigMissing

#: Entropy estimators of `entropy.entropy_from_counts` (AS-766).
ENTROPY_ESTIMATORS: tuple[str, ...] = ("plugin", "miller_madow", "chao_shen")
#: Traffic distributions read per entity (AS-765).
ENTITY_DISTRIBUTIONS: tuple[str, ...] = ("dst_port", "src_port", "protocol", "flags", "flag_mass", "peer")
#: Traffic distributions read over the whole network (AS-765).
NETWORK_DISTRIBUTIONS: tuple[str, ...] = ("dst_port", "src_port", "protocol", "flags", "flag_mass", "src_entity",
                                          "dst_entity")
#: Which participations of an entity in a state update count for its distributions (AS-765).
ROLES: tuple[str, ...] = ("any", "initiator", "responder")
#: Edge weights of the activity multiplex (AS-767).
EDGE_WEIGHTS: tuple[str, ...] = ("binary", "count", "log1p_count")
#: Graph state read by the batch path (AS-767).
GRAPH_SOURCES: tuple[str, ...] = ("activity", "contacts")
#: Linkage rules of the multiplex reduction (AS-771).
LINKAGES: tuple[str, ...] = ("average", "single", "complete", "ward")
#: Density matrices compared by the quantum Jensen-Shannon divergence (AS-770).
JSD_KINDS: tuple[str, ...] = ("laplacian", "diffusion")
#: Detrending of early-warning series (AS-772).
DETRENDS: tuple[str, ...] = ("none", "gaussian", "gaussian_causal")
#: Early-warning indicators (AS-773).
INDICATORS: tuple[str, ...] = ("variance", "ar1", "skewness", "kurtosis", "return_rate", "spectral_ratio",
                               "spectral_exponent", "dfa")
#: Surrogate families of the trend significance test (AS-775).
SURROGATE_KINDS: tuple[str, ...] = ("phase", "ar1")
#: Composite scores that may raise the alarm (AS-776, AS-777).
ALARM_SCORES: tuple[str, ...] = ("level", "trend")


def _doc(text: str) -> dict[str, str]:
    # Field metadata: the documentation line rendered as a YAML comment above the key.
    return {"doc": text}


def _check_choice(name: str, value: str, choices: tuple[str, ...]) -> None:
    if value not in choices:
        raise ValueError(f"{name} must be one of {choices}; got {value!r}")


def _check_subset(name: str, values: tuple[str, ...], choices: tuple[str, ...], *, allow_empty: bool = False) -> None:
    if not values and not allow_empty:
        raise ValueError(f"{name} must not be empty")
    unknown = [v for v in values if v not in choices]
    if unknown:
        raise ValueError(f"{name} has unknown entries {unknown}; known: {choices}")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} has duplicate entries")


def _check_positive(name: str, value: float) -> None:
    if not (math.isfinite(value) and value > 0):
        raise ValueError(f"{name} must be finite and > 0; got {value}")


@dataclass(frozen=True)
class GibbsConfig:
    """Temperature of the Gibbs readouts and the window of the growth rates."""

    temperature: float = field(default=1.0, metadata=_doc(
        "Gibbs temperature T when verifier_family is null; 1.0 is the native temperature of TAAFT's energy in "
        "nats (AS-761)."))
    verifier_family: str | None = field(default="p_inf", metadata=_doc(
        "Verifier output family whose temperature in force sets T (p_inf, stage, compromise, advice); null uses "
        "`temperature` (AS-761)."))
    growth_window: int = field(default=5, metadata=_doc(
        "Valid points of the trailing least-squares slope of every growth rate; 2 is the backward difference "
        "(AS-763)."))
    growth_entities: int = field(default=65536, metadata=_doc(
        "Most entities whose energy growth is tracked online; the least recently updated is dropped beyond it "
        "(AS-763)."))

    def __post_init__(self) -> None:
        _check_positive("gibbs.temperature", self.temperature)
        if self.growth_entities < 1:
            raise ValueError("gibbs.growth_entities must be >= 1")
        if self.verifier_family is not None:
            # Imported here, not at module level: the Verifier package imports model code, and
            # model code (TAAFT readouts) imports this package.
            from nagahana.models.verifier.reports import OUTPUT_FAMILIES

            _check_choice("gibbs.verifier_family", self.verifier_family, OUTPUT_FAMILIES)
        if self.growth_window < 2:
            raise ValueError("gibbs.growth_window must be >= 2")


@dataclass(frozen=True)
class TrafficConfig:
    """Shannon entropies of the traffic distributions of the network state window."""

    estimator: str = field(default="miller_madow", metadata=_doc(
        "Entropy estimator: plugin, miller_madow or chao_shen (AS-766)."))
    entity_distributions: tuple[str, ...] = field(default=ENTITY_DISTRIBUTIONS, metadata=_doc(
        "Distributions read per entity (AS-765)."))
    network_distributions: tuple[str, ...] = field(default=NETWORK_DISTRIBUTIONS, metadata=_doc(
        "Distributions read over the whole network (AS-765)."))
    roles: str = field(default="any", metadata=_doc(
        "Participations that count for an entity: any, initiator or responder (AS-765)."))
    resync_every: int = field(default=4096, metadata=_doc(
        "Histogram changes between exact recomputations of sum n log n in the streaming counters (AS-778)."))

    def __post_init__(self) -> None:
        _check_choice("traffic.estimator", self.estimator, ENTROPY_ESTIMATORS)
        _check_subset("traffic.entity_distributions", self.entity_distributions, ENTITY_DISTRIBUTIONS, allow_empty=True)
        _check_subset("traffic.network_distributions", self.network_distributions, NETWORK_DISTRIBUTIONS,
                      allow_empty=True)
        _check_choice("traffic.roles", self.roles, ROLES)
        if self.resync_every < 1:
            raise ValueError("traffic.resync_every must be >= 1")


@dataclass(frozen=True)
class SpectralConfig:
    """Von Neumann and spectral entropies of the multiplex graph, exact or by stochastic Lanczos quadrature."""

    graph_source: str = field(default="activity", metadata=_doc(
        "Graph state of the batch path: activity (state updates of the state window) or contacts (first "
        "contacts as of the trigger) (AS-767)."))
    weight: str = field(default="binary", metadata=_doc(
        "Edge weight of the activity multiplex: binary, count or log1p_count (AS-767)."))
    taus: tuple[float, ...] = field(default=(0.1, 1.0, 10.0), metadata=_doc(
        "Diffusion times tau of the spectral entropy, in units of 1 / edge weight (AS-768)."))
    exact_max_nodes: int = field(default=512, metadata=_doc(
        "Connected components up to this size are diagonalised exactly; larger ones use stochastic Lanczos "
        "quadrature (AS-769)."))
    slq_steps: int = field(default=32, metadata=_doc(
        "Initial Lanczos steps per probe vector (AS-769)."))
    slq_max_steps: int = field(default=128, metadata=_doc(
        "Largest Lanczos depth the adaptive depth control may reach (AS-769)."))
    slq_min_probes: int = field(default=16, metadata=_doc(
        "Fewest Rademacher probe vectors before the stopping rule may stop (AS-769)."))
    slq_max_probes: int = field(default=256, metadata=_doc(
        "Most Rademacher probe vectors (AS-769)."))
    slq_probe_batch: int = field(default=16, metadata=_doc(
        "Probe vectors run together in one batched Lanczos pass (AS-769)."))
    slq_rel_tol: float = field(default=1e-3, metadata=_doc(
        "Relative error target of every stochastic Lanczos estimate (AS-769)."))
    slq_abs_tol: float = field(default=1e-6, metadata=_doc(
        "Absolute error target of every stochastic Lanczos estimate (AS-769)."))
    slq_confidence: float = field(default=2.576, metadata=_doc(
        "Normal quantile multiplying the standard error in the stopping rule (2.576: two-sided 99 %) (AS-769)."))
    slq_deflation: int = field(default=64, metadata=_doc(
        "Approximate lowest non-zero Laplacian modes deflated exactly from the stochastic traces (0: none) "
        "(AS-769)."))
    slq_control_degree: int = field(default=2, metadata=_doc(
        "Degree of the polynomial control variate of the stochastic traces, 0 to 4 (0: none; up to 2 needs only "
        "degrees and edge weights, 3 and 4 the sparse square of L) (AS-769)."))
    seed: int = field(default=0, metadata=_doc(
        "Seed of the probe vectors (estimates are reproducible) (AS-769)."))
    multiplex: bool = field(default=True, metadata=_doc(
        "Read the multiplex: per-plane entropies, inter-plane divergences, relative entropy and reduction "
        "(AS-771)."))
    linkage: str = field(default="average", metadata=_doc(
        "Linkage of the hierarchical reduction of the planes: average, single, complete or ward (AS-771)."))
    jsd_kind: str = field(default="laplacian", metadata=_doc(
        "Density matrices of the divergence between successive states: laplacian (L / tr L) or diffusion "
        "(exp(-tau L) / Z) (AS-770)."))
    jsd_tau: float = field(default=1.0, metadata=_doc(
        "Diffusion time of the diffusion-kind divergence (AS-770)."))

    def __post_init__(self) -> None:
        _check_choice("spectral.graph_source", self.graph_source, GRAPH_SOURCES)
        _check_choice("spectral.weight", self.weight, EDGE_WEIGHTS)
        if not self.taus:
            raise ValueError("spectral.taus must not be empty")
        for t in self.taus:
            _check_positive("spectral.taus entry", t)
        if self.exact_max_nodes < 2:
            raise ValueError("spectral.exact_max_nodes must be >= 2")
        if not 2 <= self.slq_steps <= self.slq_max_steps:
            raise ValueError("spectral needs 2 <= slq_steps <= slq_max_steps")
        if self.slq_probe_batch < 1:
            raise ValueError("spectral.slq_probe_batch must be >= 1")
        if not 2 <= self.slq_min_probes <= self.slq_max_probes:
            raise ValueError("spectral needs 2 <= slq_min_probes <= slq_max_probes (a standard error needs two probes)")
        _check_positive("spectral.slq_rel_tol", self.slq_rel_tol)
        _check_positive("spectral.slq_abs_tol", self.slq_abs_tol)
        _check_positive("spectral.slq_confidence", self.slq_confidence)
        if self.slq_deflation < 0:
            raise ValueError("spectral.slq_deflation must be >= 0")
        if not 0 <= self.slq_control_degree <= 4:
            raise ValueError("spectral.slq_control_degree must lie in 0 ... 4")
        _check_choice("spectral.linkage", self.linkage, LINKAGES)
        _check_choice("spectral.jsd_kind", self.jsd_kind, JSD_KINDS)
        _check_positive("spectral.jsd_tau", self.jsd_tau)


@dataclass(frozen=True)
class EarlyWarningConfig:
    """Critical-slowing-down indicators, Kendall trend tests, surrogates, composite scores and the alarm."""

    window: int = field(default=60, metadata=_doc(
        "Rolling window of the indicators, in samples (one hour of 60 s triggers) (AS-772)."))
    detrend: str = field(default="gaussian_causal", metadata=_doc(
        "Detrending: none, gaussian (two-sided, offline analysis) or gaussian_causal (one-sided, online) (AS-772)."))
    bandwidth: float = field(default=10.0, metadata=_doc(
        "Standard deviation of the Gaussian detrending kernel, in samples (AS-772)."))
    kernel_truncate: float = field(default=4.0, metadata=_doc(
        "Kernel support in standard deviations: the kernel has ceil(truncate * bandwidth) + 1 taps per side "
        "(AS-772)."))
    indicators: tuple[str, ...] = field(default=INDICATORS, metadata=_doc(
        "Indicators computed on every full window (AS-773)."))
    spectral_low_max: float = field(default=0.05, metadata=_doc(
        "Upper edge of the low-frequency band of the spectral ratio, cycles per sample (AS-773)."))
    spectral_high_min: float = field(default=0.25, metadata=_doc(
        "Lower edge of the high-frequency band of the spectral ratio, cycles per sample (AS-773)."))
    dfa_scales: tuple[int, ...] = field(default=(4, 6, 8, 12), metadata=_doc(
        "Box sizes of the detrended fluctuation analysis, in samples (AS-773)."))
    dfa_min_boxes: int = field(default=4, metadata=_doc(
        "Fewest complete boxes per scale inside a window (AS-773)."))
    trend_window: int = field(default=60, metadata=_doc(
        "Indicator values in the trailing Kendall trend statistic (AS-774)."))
    surrogates: int = field(default=199, metadata=_doc(
        "Surrogate series per family in the significance test (AS-775)."))
    surrogate_kinds: tuple[str, ...] = field(default=SURROGATE_KINDS, metadata=_doc(
        "Surrogate families: phase (phase-randomised) and ar1 (fitted AR(1)) (AS-775)."))
    composite_indicators: tuple[str, ...] = field(default=("variance", "ar1"), metadata=_doc(
        "Indicators combined into the composite scores (AS-776)."))
    alarm_score: str = field(default="level", metadata=_doc(
        "Composite that raises the alarm: level (robust z of indicator levels) or trend (Kendall tau) (AS-777)."))
    target_false_alarm_rate: float = field(default=0.01, metadata=_doc(
        "Target false-alarm rate per trigger over all monitored series, calibrated on benign data (AS-777)."))
    calibration_fit_fraction: float = field(default=0.5, metadata=_doc(
        "Share of the benign calibration data that fits the baselines; the rest sets the threshold (AS-777)."))
    resync_every: int = field(default=60, metadata=_doc(
        "Streaming updates between exact recomputations of the running sums (AS-778)."))
    seed: int = field(default=0, metadata=_doc(
        "Seed of the surrogate generator (AS-775)."))

    def __post_init__(self) -> None:
        if self.window < 8:
            raise ValueError("ews.window must be >= 8 samples")
        _check_choice("ews.detrend", self.detrend, DETRENDS)
        _check_positive("ews.bandwidth", self.bandwidth)
        _check_positive("ews.kernel_truncate", self.kernel_truncate)
        _check_subset("ews.indicators", self.indicators, INDICATORS)
        if not 0.0 < self.spectral_low_max < self.spectral_high_min <= 0.5:
            raise ValueError("ews needs 0 < spectral_low_max < spectral_high_min <= 0.5 (cycles per sample)")
        if {"spectral_ratio", "spectral_exponent"} & set(self.indicators):
            half = self.window // 2
            freqs = [k / self.window for k in range(1, half + 1)]
            if not any(f <= self.spectral_low_max for f in freqs):
                raise ValueError(f"ews: no Fourier frequency of a {self.window}-sample window lies in the low band")
            if not any(f >= self.spectral_high_min for f in freqs):
                raise ValueError(f"ews: no Fourier frequency of a {self.window}-sample window lies in the high band")
        if "dfa" in self.indicators:
            if len(set(self.dfa_scales)) < 2:
                raise ValueError("ews.dfa_scales needs at least two distinct box sizes")
            if min(self.dfa_scales) < 3:
                raise ValueError("ews.dfa_scales: a box needs >= 3 samples (a line fit with a residual)")
            if self.dfa_min_boxes < 1 or (self.dfa_min_boxes + 1) * max(self.dfa_scales) > self.window:
                raise ValueError("ews needs (dfa_min_boxes + 1) * max(dfa_scales) <= window, so every window holds "
                                 "dfa_min_boxes complete boxes of every scale")
            if list(self.dfa_scales) != sorted(set(self.dfa_scales)):
                raise ValueError("ews.dfa_scales must be strictly increasing")
        if self.trend_window < 3:
            raise ValueError("ews.trend_window must be >= 3")
        if self.surrogates < 1:
            raise ValueError("ews.surrogates must be >= 1")
        _check_subset("ews.surrogate_kinds", self.surrogate_kinds, SURROGATE_KINDS)
        _check_subset("ews.composite_indicators", self.composite_indicators, self.indicators)
        _check_choice("ews.alarm_score", self.alarm_score, ALARM_SCORES)
        if not 0.0 < self.target_false_alarm_rate < 1.0:
            raise ValueError("ews.target_false_alarm_rate must lie in (0, 1)")
        if not 0.0 < self.calibration_fit_fraction < 1.0:
            raise ValueError("ews.calibration_fit_fraction must lie in (0, 1)")
        if self.resync_every < 1:
            raise ValueError("ews.resync_every must be >= 1")

    @property
    def kernel_taps(self) -> int:
        """Taps of the one-sided Gaussian kernel: ceil(truncate * bandwidth) + 1 (lags 0 ... ceil(truncate * bandwidth))."""
        return int(math.ceil(self.kernel_truncate * self.bandwidth)) + 1


#: Series monitored online by default (names of `trajectory.StatPhysReading.series`) (AS-777).
DEFAULT_EWS_SERIES: tuple[str, ...] = ("energy.total", "energy.marginal", "entities.free_energy", "entities.entropy",
                                       "routes.free_energy", "routes.entropy", "graph.vn.aggregate",
                                       "traffic.dst_port")


@dataclass(frozen=True)
class StatPhysConfig:
    """The statistical-physics module (D-56): network-state window, sections and monitored series."""

    enabled: bool = field(default=True, metadata=_doc(
        "Emit a statistical-physics reading with every trigger of the inference engine."))
    state_window_seconds: float = field(default=300.0, metadata=_doc(
        "Network state at a trigger tau: the state updates of (tau - window, tau], for the traffic entropies and "
        "the activity multiplex (AS-764)."))
    gibbs: GibbsConfig = field(default_factory=GibbsConfig, metadata=_doc(
        "Gibbs readouts (AS-760 to AS-763)."))
    traffic: TrafficConfig = field(default_factory=TrafficConfig, metadata=_doc(
        "Shannon entropies of traffic distributions (AS-765, AS-766)."))
    spectral: SpectralConfig = field(default_factory=SpectralConfig, metadata=_doc(
        "Von Neumann and spectral entropies of the multiplex graph (AS-767 to AS-771)."))
    ews: EarlyWarningConfig = field(default_factory=EarlyWarningConfig, metadata=_doc(
        "Early-warning indicators and alarm (AS-772 to AS-778)."))
    ews_series: tuple[str, ...] = field(default=DEFAULT_EWS_SERIES, metadata=_doc(
        "Reading series monitored by the streaming early-warning indicators (AS-777)."))
    calibration_path: str | None = field(default=None, metadata=_doc(
        "JSON file of per-series alarm calibrations (written by `statphys calibrate`); null raises no alarm, "
        "because a threshold is never defaulted (AS-777)."))

    def __post_init__(self) -> None:
        _check_positive("state_window_seconds", self.state_window_seconds)
        if len(set(self.ews_series)) != len(self.ews_series):
            raise ValueError("ews_series has duplicate entries")


T = TypeVar("T")


def to_mapping(cfg: Any) -> dict[str, Any]:
    """A config dataclass as plain data (nested dicts, lists for tuples), in field order."""
    if not is_dataclass(cfg) or isinstance(cfg, type):
        raise TypeError("to_mapping needs a config dataclass instance")
    out: dict[str, Any] = {}
    for f in fields(cfg):
        v = getattr(cfg, f.name)
        if is_dataclass(v) and not isinstance(v, type):
            out[f.name] = to_mapping(v)
        elif isinstance(v, tuple):
            out[f.name] = list(v)
        else:
            out[f.name] = v
    return out


def _yaml_scalar(v: Any) -> str:
    # YAML 1.1 scalars that PyYAML's safe loader reads back to the same Python value and type.
    if isinstance(v, bool):
        return "true" if v else "false"
    if v is None:
        return "null"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        if math.isnan(v):
            return ".nan"
        if math.isinf(v):
            return ".inf" if v > 0 else "-.inf"
        s = repr(v)
        if "e" in s or "E" in s:
            mant, exp = s.lower().split("e")
            if "." not in mant:
                mant += ".0"                    # YAML 1.1 floats need a dot in the mantissa
            if exp[0] not in "+-":
                exp = "+" + exp                 # ... and a signed exponent
            s = f"{mant}e{exp}"
        elif "." not in s:
            s += ".0"
        return s
    if isinstance(v, str):
        return json.dumps(v)                    # a JSON string is a YAML double-quoted scalar
    raise TypeError(f"no YAML scalar form for {type(v).__name__}")


def _render(cfg: Any, indent: int, lines: list[str]) -> None:
    pad = "  " * indent
    for f in fields(cfg):
        v = getattr(cfg, f.name)
        doc = f.metadata.get("doc")
        if is_dataclass(v) and not isinstance(v, type):
            lines.append("")
            if doc:
                lines.append(f"{pad}# {doc}")
            lines.append(f"{pad}{f.name}:")
            _render(v, indent + 1, lines)
            continue
        if doc:
            lines.append(f"{pad}# {doc}")
        body = "[" + ", ".join(_yaml_scalar(x) for x in v) + "]" if isinstance(v, tuple) else _yaml_scalar(v)
        lines.append(f"{pad}{f.name}: {body}")


def render_yaml(cfg: StatPhysConfig | None = None) -> str:
    """The YAML form of a configuration (default: the dataclass defaults), with field documentation as comments."""
    c = cfg if cfg is not None else StatPhysConfig()
    lines = [
        "# Statistical-physics readouts and early warning (src/nagahana/statphys, D-56).",
        "# Generated from the dataclasses in src/nagahana/statphys/config.py, the single source of truth:",
        "#     python -m nagahana statphys write-config conf/statphys/statphys.yaml",
        "# Do not edit by hand; change the dataclass defaults and regenerate (a test compares the two).",
    ]
    _render(c, 0, lines)
    return "\n".join(lines) + "\n"


def write_yaml(path: str | Path, cfg: StatPhysConfig | None = None) -> Path:
    """Write `render_yaml(cfg)` to `path` (UTF-8, LF line ends). Returns the path."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(render_yaml(cfg), encoding="utf-8", newline="\n")
    return p


def _coerce(value: Any, tp: Any, where: str) -> Any:
    # Check (and lightly convert: int -> float, list -> tuple) one value against a declared field type.
    if isinstance(value, str) and value == MISSING:
        raise ConfigMissing(f"{where} is '???': statphys settings are never left undecided; give a value")
    if isinstance(tp, type) and is_dataclass(tp):
        if not isinstance(value, Mapping):
            raise ValueError(f"{where} must be a mapping")
        return from_mapping(tp, value, path=where + ".")
    origin = typing.get_origin(tp)
    if origin in (typing.Union, types.UnionType):
        args = typing.get_args(tp)
        if value is None and type(None) in args:
            return None
        others = [a for a in args if a is not type(None)]
        if len(others) != 1:
            raise TypeError(f"{where}: unsupported union type {tp}")
        return _coerce(value, others[0], where)
    if origin is tuple:
        args = typing.get_args(tp)
        if not isinstance(value, list | tuple):
            raise ValueError(f"{where} must be a list")
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_coerce(x, args[0], f"{where}[{i}]") for i, x in enumerate(value))
        raise TypeError(f"{where}: unsupported tuple type {tp}")
    if tp is bool:
        if not isinstance(value, bool):
            raise ValueError(f"{where} must be true or false; got {value!r}")
        return value
    if tp is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{where} must be an integer; got {value!r}")
        return value
    if tp is float:
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError(f"{where} must be a number; got {value!r}")
        return float(value)
    if tp is str:
        if not isinstance(value, str):
            raise ValueError(f"{where} must be a string; got {value!r}")
        return value
    raise TypeError(f"{where}: unsupported field type {tp}")


def from_mapping(cls: type[T], data: Mapping[str, Any], *, path: str = "") -> T:
    """Build config dataclass `cls` from plain data, strictly (unknown keys and wrong types raise).

    A key the mapping leaves out takes the dataclass default, so the dataclass stays the single source
    of truth; "???" raises `ConfigMissing`; ranges are checked by the dataclass's own validation.
    """
    if not (isinstance(cls, type) and is_dataclass(cls)):
        raise TypeError("from_mapping needs a config dataclass type")
    hints = typing.get_type_hints(cls)
    names = {f.name for f in fields(cls)}
    unknown = sorted(set(data) - names)
    if unknown:
        raise ValueError(f"unknown statphys config keys {[path + k for k in unknown]}")
    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        if f.name not in data:
            if f.default is DC_MISSING and f.default_factory is DC_MISSING:
                raise ValueError(f"{path}{f.name} is required")
            continue
        kwargs[f.name] = _coerce(data[f.name], hints[f.name], path + f.name)
    return cls(**kwargs)


def load_config(path: str | Path) -> StatPhysConfig:
    """Read a statphys YAML file (top-level keys = `StatPhysConfig` fields) and validate it."""
    return from_mapping(StatPhysConfig, load_yaml(path))


__all__ = [
    "ALARM_SCORES", "DEFAULT_EWS_SERIES", "DETRENDS", "EDGE_WEIGHTS", "ENTITY_DISTRIBUTIONS", "ENTROPY_ESTIMATORS",
    "GRAPH_SOURCES", "INDICATORS", "JSD_KINDS", "LINKAGES", "NETWORK_DISTRIBUTIONS", "ROLES", "SURROGATE_KINDS",
    "EarlyWarningConfig", "GibbsConfig", "SpectralConfig", "StatPhysConfig", "TrafficConfig", "from_mapping",
    "load_config", "render_yaml", "to_mapping", "write_yaml",
]
