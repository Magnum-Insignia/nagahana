"""Uniform interface of the published-baseline reproductions.

Every reproduction in this package is a subclass of `PublishedBaseline` with

    a config dataclass      the paper's stated settings as defaults; anything the paper leaves open is
                            a default recorded as an assumption (AS-530 ... AS-569)
    fit(frame)              trains on a pandas DataFrame in the baseline's input schema
    predict(frame)          returns `evaluation.predictions.ModelOutputs`, the record every model of
                            the evaluation reports through, so one scoring pipeline treats NagaHana,
                            the logistic-regression baselines and these reproductions identically
    save(path) / load(path) a directory with a JSON manifest (config, fitted state, SHA-256 of every
                            payload file) and the payload files
    spec                    a `BaselineSpec`: the reference, the input schema, the outputs, the results
                            the paper prints (`ReportedResult`) and the official-code entry of
                            tools/third_party/manifest.toml

Reproducibility
---------------
`fit` and `predict` run inside `seeded(config.seed)`: the PyTorch generator state is forked, seeded and
restored afterwards, so a baseline never perturbs the random state of the caller, and every source of
randomness in the package (NumPy generators, scikit-learn `random_state`, XGBoost and LightGBM seeds) is
derived from the same seed (AS-533).

Configs from YAML
-----------------
`config_from_mapping` converts a plain mapping (for example a file of conf/baselines/published) into the
baseline's config dataclass and rejects unknown keys, wrong types and Hydra's mandatory marker `???`, so
a typo in a config fails loudly instead of silently falling back to a default.
"""

from __future__ import annotations

import abc
import dataclasses
import hashlib
import importlib
import json
import math
import types
import typing
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Literal, TypeVar, get_args, get_origin, get_type_hints

import numpy as np
import pandas as pd
import torch

from nagahana import __version__ as _PACKAGE_VERSION
from nagahana.core.config import MISSING, load_yaml
from nagahana.core.errors import ConfigMissing, InvariantViolation, NagaHanaError
from nagahana.evaluation.predictions import (
    DetectionPredictions,
    EpisodeTable,
    ForecastPredictions,
    ModelOutputs,
    PathPredictions,
    StagePredictions,
    StateForecastPredictions,
    TimeToEventPredictions,
)

#: Format tag written into every saved manifest; `load` refuses other formats.
MANIFEST_FORMAT = "nagahana-published-baseline"
#: Version of the manifest layout (bumped when the layout changes incompatibly).
MANIFEST_VERSION = 1
MANIFEST_NAME = "manifest.json"


class MissingDependency(NagaHanaError, ImportError):
    """An optional library a reproduction needs is not installed (the [baselines] extra, D-55)."""


class BaselineStateError(NagaHanaError):
    """A baseline was used in a state that does not allow the call (for example predict before fit)."""


@dataclass(frozen=True)
class Reference:
    """Bibliographic record of the paper a baseline reproduces.

    Attributes
    ----------
    key:
        Citation key of latexdocs/full-docs/references-baselines.bib (prefix "bl-") or of the main
        bibliography where the thesis cites the paper there.
    authors, title, venue, year:
        As printed on the paper.
    doi, arxiv:
        Persistent identifiers ("" when the paper has none).
    note:
        Where the transcribed values come from when it is not the version of record (for example an
        arXiv version).
    """

    key: str
    authors: str
    title: str
    venue: str
    year: int
    doi: str = ""
    arxiv: str = ""
    note: str = ""

    def cite(self) -> str:
        """One-line citation: authors, title, venue, year and identifiers."""
        ids = ", ".join(x for x in (f"DOI {self.doi}" if self.doi else "", f"arXiv:{self.arxiv}" if self.arxiv else "") if x)
        tail = f" ({ids})" if ids else ""
        return f"{self.authors}, \"{self.title}\", {self.venue}, {self.year}{tail}"


@dataclass(frozen=True)
class ReportedResult:
    """One result as printed in the source, with the variant of the baseline that produced it.

    `values` maps a canonical metric name (see `metrics.METRIC_ALIASES`) to the value exactly as
    printed, for example "0.9705", "95.96%" or "85.49". A value printed as a bare number that the source
    means as a percentage is declared in `percent` (EULER's Table VI prints TPR 85.49 meaning 85.49 %).
    `variant` holds the config overrides that select the model the row describes, so a comparison can
    build the same variant. Every value was transcribed in latexdocs/full-docs/baselines-notes.md, which
    records the table and page it was read from (AS-531).
    """

    dataset: str
    protocol: str
    task: str
    model: str
    values: Mapping[str, str]
    location: str
    variant: Mapping[str, Any] = field(default_factory=dict)
    percent: tuple[str, ...] = ()
    note: str = ""

    def value(self, metric: str) -> float:
        """The printed value of `metric` as a fraction in [0, 1] (or the raw number for counts and MCC).

        A trailing "%" or membership in `percent` divides by 100. Values with qualifiers ("~65%",
        "<=78%") are not numbers and raise ValueError.
        """
        if metric not in self.values:
            raise KeyError(f"{self.model} on {self.dataset} reports no {metric!r}; reported: {sorted(self.values)}")
        text = self.values[metric].strip()
        is_percent = text.endswith("%") or metric in self.percent
        number = text[:-1] if text.endswith("%") else text
        try:
            v = float(number)
        except ValueError as exc:
            raise ValueError(f"reported {metric} {text!r} is not a plain number") from exc
        return v / 100.0 if is_percent else v

    def numeric_metrics(self) -> tuple[str, ...]:
        """Metrics whose printed value parses as a plain number."""
        out = []
        for m in self.values:
            try:
                self.value(m)
            except ValueError:
                continue
            out.append(m)
        return tuple(out)


@dataclass(frozen=True)
class InputSchema:
    """The DataFrame a baseline consumes.

    Attributes
    ----------
    description:
        One paragraph: what one row is and where such rows come from.
    required:
        Columns every frame must have. Feature columns that depend on the config are checked by the
        baseline itself (`PublishedBaseline.required_columns`).
    label:
        Label column needed by `fit` (None for self-supervised or label-free training).
    time:
        Column with event time in epoch seconds (None when the baseline is order-free).
    optional:
        Columns used when present (meta columns of evaluation/predictions.py are always optional).
    """

    description: str
    required: tuple[str, ...] = ()
    label: str | None = None
    time: str | None = None
    optional: tuple[str, ...] = ()

    def check(self, frame: pd.DataFrame, columns: tuple[str, ...], *, need_label: bool) -> None:
        """Raise InvariantViolation when `frame` lacks a required column (or the label when needed)."""
        if not isinstance(frame, pd.DataFrame):
            raise InvariantViolation(f"expected a pandas DataFrame, got {type(frame).__name__}")
        wanted = list(dict.fromkeys((*self.required, *columns)))
        if need_label and self.label is not None:
            wanted.append(self.label)
        missing = [c for c in wanted if c not in frame.columns]
        if missing:
            raise InvariantViolation(f"input frame is missing columns {missing}")
        if len(frame) == 0:
            raise InvariantViolation("input frame has no rows")


@dataclass(frozen=True)
class BaselineSpec:
    """Static description of one reproduction.

    Attributes
    ----------
    name:
        Registry key (kebab case), also the `model` field of the produced ModelOutputs.
    title:
        Human-readable name.
    reference:
        The paper.
    family:
        What the model does: "flow-classifier", "window-classifier", "sequence-classifier",
        "stage-forecaster", "link-predictor", "edge-anomaly", "event-forecaster", "path-detector" or
        "risk-forecaster".
    input_schema:
        The frame `fit` and `predict` consume.
    outputs:
        Fields of ModelOutputs the baseline fills.
    datasets:
        Datasets the paper evaluates on, in the names of the evaluation design.
    reported:
        Results as printed in the paper (the reported operating points).
    third_party:
        Name of the entry in tools/third_party/manifest.toml (official code, when it exists).
    assumptions:
        AS-IDs of docs/assumptions/published-baselines.md that the reproduction relies on.
    """

    name: str
    title: str
    reference: Reference
    family: str
    input_schema: InputSchema
    outputs: tuple[str, ...]
    datasets: tuple[str, ...]
    reported: tuple[ReportedResult, ...]
    third_party: str
    assumptions: tuple[str, ...] = ()

    def reported_for(self, *, dataset: str | None = None, model: str | None = None) -> tuple[ReportedResult, ...]:
        """Reported results filtered by dataset and model variant."""
        return tuple(r for r in self.reported if (dataset is None or r.dataset == dataset) and (model is None or r.model == model))


@dataclass
class BaselineConfig:
    """Settings shared by every reproduction. Subclasses add the paper's own settings.

    Attributes
    ----------
    seed:
        Seed of every random choice of fit and predict (AS-533).
    device:
        PyTorch device of the neural reproductions ("cpu", "cuda", "cuda:1", ...). Tree and linear models
        ignore it.
    deterministic:
        Ask PyTorch for deterministic kernels inside fit and predict (warn-only, restored afterwards).
    """

    seed: int = 0
    device: str = "cpu"
    deterministic: bool = True

    def validate(self) -> None:
        """Check value ranges; subclasses extend it and call super().validate()."""
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if not self.device:
            raise ValueError("device must name a PyTorch device")


B = TypeVar("B", bound="PublishedBaseline")


def _type_error(where: str, expected: str, value: Any) -> TypeError:
    return TypeError(f"config field {where}: expected {expected}, got {value!r} ({type(value).__name__})")


def _convert(tp: Any, value: Any, where: str) -> Any:
    # Convert one YAML/JSON value to the annotated field type, strictly (no silent coercion).
    if isinstance(value, str) and value == MISSING:
        raise ConfigMissing(f"config field {where} is still '???' (undecided); give a value")
    if tp is Any:
        return value
    origin = get_origin(tp)
    if origin is typing.Union or origin is types.UnionType:
        args = get_args(tp)
        if value is None:
            if type(None) in args:
                return None
            raise _type_error(where, str(tp), value)
        problems: list[str] = []
        for arg in args:
            if arg is type(None):
                continue
            try:
                return _convert(arg, value, where)
            except (TypeError, ValueError) as exc:
                problems.append(str(exc))
        raise TypeError(f"config field {where}: {value!r} matches none of {tp}: {'; '.join(problems)}")
    if origin is Literal:
        if value not in get_args(tp):
            raise ValueError(f"config field {where}: {value!r} is not one of {get_args(tp)}")
        return value
    if origin is tuple:
        args = get_args(tp)
        if not isinstance(value, list | tuple):
            raise _type_error(where, "a list", value)
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_convert(args[0], v, f"{where}[{i}]") for i, v in enumerate(value))
        if len(args) != len(value):
            raise ValueError(f"config field {where}: expected {len(args)} items, got {len(value)}")
        return tuple(_convert(a, v, f"{where}[{i}]") for i, (a, v) in enumerate(zip(args, value, strict=True)))
    if origin in (dict, Mapping):
        key_t, val_t = get_args(tp) or (Any, Any)
        if not isinstance(value, Mapping):
            raise _type_error(where, "a mapping", value)
        return {_convert(key_t, k, f"{where}.key"): _convert(val_t, v, f"{where}[{k!r}]") for k, v in value.items()}
    if isinstance(tp, type) and dataclasses.is_dataclass(tp):
        if isinstance(value, tp):
            return value
        if not isinstance(value, Mapping):
            raise _type_error(where, f"a mapping for {tp.__name__}", value)
        return config_from_mapping(tp, value, prefix=f"{where}.")
    if tp is bool:
        if not isinstance(value, bool | np.bool_):
            raise _type_error(where, "a boolean", value)
        return bool(value)
    if tp is int:
        if isinstance(value, bool | np.bool_) or not isinstance(value, int | np.integer):
            raise _type_error(where, "an integer", value)
        return int(value)
    if tp is float:
        if isinstance(value, bool | np.bool_) or not isinstance(value, int | float | np.integer | np.floating):
            raise _type_error(where, "a number", value)
        return float(value)
    if tp is str:
        if not isinstance(value, str):
            raise _type_error(where, "a string", value)
        return value
    raise TypeError(f"config field {where}: unsupported annotation {tp!r}")


def config_from_mapping(cls: type[Any], data: Mapping[str, Any], *, prefix: str = "") -> Any:
    """Build the config dataclass `cls` from a mapping, strictly.

    Unknown keys raise ValueError (a misspelt key never falls back to a default); "???" raises
    ConfigMissing; every value is converted to its annotated type or rejected. Keys absent from `data`
    keep the dataclass default (the paper's setting or the recorded assumption).
    """
    if not (isinstance(cls, type) and dataclasses.is_dataclass(cls)):
        raise TypeError(f"{cls!r} is not a dataclass type")
    hints = get_type_hints(cls)
    names = {f.name for f in dataclasses.fields(cls) if f.init}
    unknown = sorted(set(data) - names)
    if unknown:
        raise ValueError(f"unknown config keys for {cls.__name__}: {[prefix + k for k in unknown]}; known: {sorted(names)}")
    kwargs = {k: _convert(hints[k], v, prefix + k) for k, v in data.items()}
    obj = cls(**kwargs)
    validate = getattr(obj, "validate", None)
    if callable(validate):
        validate()
    return obj


def jsonable(value: Any) -> Any:
    """Convert dataclasses, tuples and NumPy scalars/arrays to JSON-compatible Python objects."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: jsonable(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, Mapping):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        f = float(value)
        return f if math.isfinite(f) else str(f)
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if value is None or isinstance(value, bool | int | float | str):
        return value
    raise TypeError(f"cannot serialise {type(value).__name__} to JSON")


def config_to_mapping(config: BaselineConfig) -> dict[str, Any]:
    """The config as a JSON-compatible mapping (inverse of `config_from_mapping`)."""
    out = jsonable(config)
    assert isinstance(out, dict)
    return out


def load_config(cls: type[Any], path: str | Path) -> Any:
    """Read a YAML file of conf/baselines/published into the config dataclass `cls`.

    The file holds a top-level mapping whose `config` key is the mapping of config fields; the other
    top-level keys (`baseline`, `reference`, `notes`) document the file and are checked: `baseline`
    must be present.
    """
    data = load_yaml(path)
    if "baseline" not in data or "config" not in data:
        raise ValueError(f"{path}: a baseline config file needs the keys 'baseline' and 'config'")
    cfg = data["config"]
    if not isinstance(cfg, Mapping):
        raise ValueError(f"{path}: 'config' must be a mapping")
    return config_from_mapping(cls, cfg)


@contextmanager
def seeded(seed: int, *, deterministic: bool = True) -> Iterator[np.random.Generator]:
    """Run a block with PyTorch seeded by `seed`, restoring the caller's random state afterwards.

    Yields a NumPy Generator seeded with the same seed for the block's own random choices. With
    `deterministic`, PyTorch's deterministic-algorithms switch is turned on in warn-only mode for the
    block and restored on exit.
    """
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    prev_mode = torch.are_deterministic_algorithms_enabled()
    prev_warn = torch.is_deterministic_algorithms_warn_only_enabled()
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        if deterministic:
            torch.use_deterministic_algorithms(True, warn_only=True)
        try:
            yield np.random.default_rng(seed)
        finally:
            torch.use_deterministic_algorithms(prev_mode, warn_only=prev_warn)


def derive_seed(seed: int, *salt: int | str) -> int:
    """A 31-bit seed derived from `seed` and a salt (for example a repeat index), stable across processes."""
    text = ":".join([str(seed), *map(str, salt)]).encode()
    return int.from_bytes(hashlib.sha256(text).digest()[:4], "big") & 0x7FFFFFFF


@dataclass
class PredictionParts:
    """The records a baseline's `_predict` produces; the base class wraps them into ModelOutputs."""

    detection: DetectionPredictions | None = None
    forecast: ForecastPredictions | None = None
    stage: StagePredictions | None = None
    state_forecast: StateForecastPredictions | None = None
    time_to_event: TimeToEventPredictions | None = None
    paths: PathPredictions | None = None
    episodes: EpisodeTable | None = None
    component: dict[str, np.ndarray] = field(default_factory=dict)


def sha256_file(path: Path) -> str:
    """Hex SHA-256 of a file's bytes, read in 1 MiB blocks."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class PublishedBaseline(abc.ABC):
    """Base class of every reproduction. See the module docstring for the contract.

    Subclasses define the class attributes `spec` and `config_type` and implement

        required_columns()      feature columns the config needs in addition to spec.input_schema
        _fit(data, validation, rng)
        _predict(data) -> PredictionParts
        _export_state(directory) -> dict    write payload files, return JSON state
        _import_state(directory, state)     restore from the payload files and the JSON state
    """

    spec: ClassVar[BaselineSpec]
    config_type: ClassVar[type[BaselineConfig]]

    def __init__(self, config: BaselineConfig | None = None) -> None:
        cfg = config if config is not None else self.config_type()
        if not isinstance(cfg, self.config_type):
            raise TypeError(f"{type(self).__name__} needs a {self.config_type.__name__}, got {type(cfg).__name__}")
        cfg.validate()
        self.config = cfg
        self._fitted = False
        #: Free-form facts recorded during fit (training sizes, epochs run, selected hyperparameters).
        self.fit_report: dict[str, Any] = {}

    @property
    def is_fitted(self) -> bool:
        """True once fit (or load) has completed."""
        return self._fitted

    def required_columns(self) -> tuple[str, ...]:
        """Columns the current config needs beyond `spec.input_schema.required` (default: none)."""
        return ()

    def prepare(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Normalise an input frame before the schema check (for example harmonise column spellings).

        Called by fit (on the training and validation frames) and by predict. Default: unchanged.
        """
        return frame

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise BaselineStateError(f"{self.spec.name}: call fit (or load) before predict or save")

    def fit(self: B, data: pd.DataFrame, *, validation: pd.DataFrame | None = None) -> B:
        """Train on `data` (labels required unless the schema has none); `validation` is optional.

        Returns self, so `Baseline(cfg).fit(train).predict(test)` reads as one line.
        """
        data = self.prepare(data)
        self.spec.input_schema.check(data, self.required_columns(), need_label=True)
        if validation is not None:
            validation = self.prepare(validation)
            self.spec.input_schema.check(validation, self.required_columns(), need_label=True)
        self.fit_report = {}
        with seeded(self.config.seed, deterministic=self.config.deterministic) as rng:
            self._fit(data, validation, rng)
        self._fitted = True
        return self

    def predict(self, data: pd.DataFrame, *, protocol: str = "paper") -> ModelOutputs:
        """Score `data` and return ModelOutputs (labels in `data` are used for the records' `label` only)."""
        self._require_fitted()
        data = self.prepare(data)
        self.spec.input_schema.check(data, self.required_columns(), need_label=False)
        with seeded(self.config.seed, deterministic=self.config.deterministic):
            parts = self._predict(data)
        return ModelOutputs(
            model=self.spec.name,
            protocol=protocol,
            seed=self.config.seed,
            detection=parts.detection,
            forecast=parts.forecast,
            stage=parts.stage,
            state_forecast=parts.state_forecast,
            time_to_event=parts.time_to_event,
            paths=parts.paths,
            episodes=parts.episodes,
            config={"baseline": self.spec.name, "reference": self.spec.reference.key, "config": config_to_mapping(self.config),
                    "fit_report": jsonable(self.fit_report)},
            component=parts.component,
        )

    def save(self, path: str | Path, *, overwrite: bool = False) -> Path:
        """Write the fitted baseline to directory `path` (created). Refuses to overwrite unless asked."""
        self._require_fitted()
        directory = Path(path)
        manifest_path = directory / MANIFEST_NAME
        if manifest_path.exists() and not overwrite:
            raise FileExistsError(f"{manifest_path} exists; pass overwrite=True to replace it")
        directory.mkdir(parents=True, exist_ok=True)
        state = self._export_state(directory)
        files = sorted(p for p in directory.rglob("*") if p.is_file() and p.name != MANIFEST_NAME)
        manifest = {
            "format": MANIFEST_FORMAT,
            "format_version": MANIFEST_VERSION,
            "baseline": self.spec.name,
            "class": f"{type(self).__module__}:{type(self).__qualname__}",
            "package_version": _PACKAGE_VERSION,
            "reference": self.spec.reference.key,
            "config": config_to_mapping(self.config),
            "state": jsonable(state),
            "fit_report": jsonable(self.fit_report),
            "files": {p.relative_to(directory).as_posix(): sha256_file(p) for p in files},
        }
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        return directory

    @classmethod
    def load(cls: type[B], path: str | Path) -> B:
        """Read a directory written by `save`, verifying the format and every file's SHA-256 first.

        Called on a subclass, the manifest must name that subclass; called on PublishedBaseline itself,
        the class named in the manifest is imported and used.
        """
        directory = Path(path)
        manifest = read_manifest(directory)
        target: type[Any] = cls
        if cls is PublishedBaseline:
            module_name, _, qualname = str(manifest["class"]).partition(":")
            target = getattr(importlib.import_module(module_name), qualname)
            if not (isinstance(target, type) and issubclass(target, PublishedBaseline)):
                raise InvariantViolation(f"{manifest['class']} is not a PublishedBaseline")
        elif manifest["baseline"] != cls.spec.name:
            raise InvariantViolation(f"{directory} holds {manifest['baseline']!r}, not {cls.spec.name!r}")
        config = config_from_mapping(target.config_type, manifest["config"])
        obj = target(config)
        obj._import_state(directory, manifest["state"])
        obj.fit_report = dict(manifest.get("fit_report", {}))
        obj._fitted = True
        return typing.cast(B, obj)

    @abc.abstractmethod
    def _fit(self, data: pd.DataFrame, validation: pd.DataFrame | None, rng: np.random.Generator) -> None:
        """Train. Called inside `seeded`; `rng` is the block's NumPy generator."""

    @abc.abstractmethod
    def _predict(self, data: pd.DataFrame) -> PredictionParts:
        """Score a frame that passed the schema check."""

    @abc.abstractmethod
    def _export_state(self, directory: Path) -> dict[str, Any]:
        """Write payload files into `directory` and return the JSON-compatible fitted state."""

    @abc.abstractmethod
    def _import_state(self, directory: Path, state: Mapping[str, Any]) -> None:
        """Restore the fitted state from `directory` and the JSON state."""


def read_manifest(directory: Path) -> dict[str, Any]:
    """Read and verify a saved baseline's manifest: format, version and the SHA-256 of every listed file."""
    manifest_path = directory / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"{manifest_path} not found")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != MANIFEST_FORMAT:
        raise InvariantViolation(f"{manifest_path} is not a {MANIFEST_FORMAT} manifest")
    if int(manifest.get("format_version", -1)) != MANIFEST_VERSION:
        raise InvariantViolation(f"{manifest_path}: unsupported format version {manifest.get('format_version')}")
    for rel, digest in manifest.get("files", {}).items():
        p = directory / rel
        if not p.is_file():
            raise InvariantViolation(f"{directory}: payload file {rel} is missing")
        if sha256_file(p) != digest:
            raise InvariantViolation(f"{directory}: payload file {rel} does not match its recorded SHA-256")
    return dict(manifest)


def load_baseline(path: str | Path) -> PublishedBaseline:
    """Load any saved reproduction; the manifest names its class."""
    return PublishedBaseline.load(path)
