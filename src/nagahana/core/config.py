"""Configuration: typed dataclasses are the single source of truth; YAML files are generated views (D-57).

Convention
----------
Every setting is a field of a frozen dataclass whose default is the build's value and whose metadata
carries its provenance (`setting(default, source=..., doc=...)`): the decision (D-xx) or assumption
(AS-xx) behind it, and one line of documentation. The YAML files under `conf/` are rendered from those
dataclasses (`render_yaml`, `generate_conf`), with identical values and the provenance as comments, so
the files can never drift from the code unnoticed (`check_conf` compares them; a test runs it).

A YAML file given at run time is an override: `from_mapping` and `apply_overrides` validate it against
the dataclass. Unknown keys, values of the wrong type, and the marker `???` on a field that has a default
raise instead of being ignored. `???` (Hydra/OmegaConf's mandatory missing value) is legitimate only for a
field without a default (a required run setting such as the run mode) or for a held decision without a
working option; reading it raises `ConfigMissing`.

Held decisions are configured in one place, `conf/decisions/decisions.yaml`, rendered from
`governance/decisions.py`: each held decision with a working option shows that option (its recorded
assumption); `decision_options_from_mapping` validates an override file and returns the options for
`governance.decisions.configure`.

`iter_missing` walks mappings and lists, so a `???` inside a list (for example inside the argument
mapping of a list entry) is found too.
"""

from __future__ import annotations

import math
import types
import typing
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import MISSING as DC_MISSING
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar

from nagahana.core.errors import ConfigMissing, InvalidOption, InvariantViolation

#: Hydra/OmegaConf's marker for "mandatory, not yet set".
MISSING = "???"

T = TypeVar("T")


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Read one YAML file into a dict (no Hydra composition)."""
    import yaml  # PyYAML; imported lazily so the core does not depend on it at import time

    with Path(path).open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top level must be a mapping, got {type(data).__name__}")
    return data


def get_required(cfg: Mapping[str, Any], key: str) -> Any:
    """Return `cfg[key]` for a dotted key; raise `ConfigMissing` if absent or `???`.

    >>> get_required({"a": {"b": 3}}, "a.b")
    3
    """
    node: Any = cfg
    for part in key.split("."):
        if not isinstance(node, Mapping) or part not in node:
            raise ConfigMissing(f"Config key {key!r} is absent.")
        node = node[part]
    if node == MISSING:
        raise ConfigMissing(f"Config key {key!r} is still '???' (undecided); no default is applied.")
    return node


def iter_missing(cfg: Any, prefix: str = "") -> Iterator[str]:
    """Yield every key whose value is `???`, depth first, through mappings and lists.

    Mapping keys join with "."; list items are written "[i]", e.g. "residuals[1].args.mtu".
    """
    if isinstance(cfg, Mapping):
        for k, v in cfg.items():
            dotted = f"{prefix}.{k}" if prefix else str(k)
            if isinstance(v, (Mapping, list, tuple)):
                yield from iter_missing(v, dotted)
            elif v == MISSING:
                yield dotted
    elif isinstance(cfg, (list, tuple)):
        for i, v in enumerate(cfg):
            here = f"{prefix}[{i}]"
            if isinstance(v, (Mapping, list, tuple)):
                yield from iter_missing(v, here)
            elif v == MISSING:
                yield here


def setting(default: Any, *, source: str, doc: str) -> Any:
    """A dataclass field with a default and its provenance (`source`: D-xx / AS-xx / reference; `doc`)."""
    if not source or not doc:
        raise ValueError("a setting needs both its source and its documentation")
    return field(default=default, metadata={"source": source, "doc": doc})


def required(*, source: str, doc: str) -> Any:
    """A dataclass field without a default (a required run setting), with its provenance."""
    if not source or not doc:
        raise ValueError("a required setting needs both its source and its documentation")
    return field(metadata={"source": source, "doc": doc})


def provenance(cls: type) -> dict[str, tuple[str, str]]:
    """Field name -> (source, doc) of a config dataclass ("" where the metadata is absent)."""
    if not (isinstance(cls, type) and is_dataclass(cls)):
        raise TypeError("provenance needs a config dataclass type")
    return {f.name: (str(f.metadata.get("source", "")), str(f.metadata.get("doc", ""))) for f in fields(cls)}


def to_mapping(obj: Any) -> Any:
    """Plain YAML-ready data of a config: dataclasses become mappings, tuples become lists."""
    if is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: to_mapping(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, (tuple, list)):
        return [to_mapping(v) for v in obj]
    if isinstance(obj, Mapping):
        return {str(k): to_mapping(v) for k, v in obj.items()}
    return obj


def _coerce(value: Any, tp: Any, where: str) -> Any:
    """Check (and losslessly convert: int -> float, list -> tuple) one value against a declared field type."""
    if isinstance(tp, type) and is_dataclass(tp):
        if not isinstance(value, Mapping):
            raise InvariantViolation(f"{where} must be a mapping")
        return from_mapping(tp, value, path=where + ".")
    origin = typing.get_origin(tp)
    if origin in (typing.Union, types.UnionType):
        args = typing.get_args(tp)
        if value is None and type(None) in args:
            return None
        others = [a for a in args if a is not type(None)]
        errors: list[str] = []
        for a in others:
            try:
                return _coerce(value, a, where)
            except InvariantViolation as exc:
                errors.append(str(exc))
        raise InvariantViolation(f"{where}: {value!r} matches none of {tp}: {'; '.join(errors)}")
    if origin is tuple:
        args = typing.get_args(tp)
        if not isinstance(value, (list, tuple)):
            raise InvariantViolation(f"{where} must be a list, got {type(value).__name__}")
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_coerce(x, args[0], f"{where}[{i}]") for i, x in enumerate(value))
        if len(args) != len(value):
            raise InvariantViolation(f"{where} must have {len(args)} items, got {len(value)}")
        return tuple(_coerce(x, a, f"{where}[{i}]") for i, (x, a) in enumerate(zip(value, args, strict=True)))
    if tp is bool:
        if not isinstance(value, bool):
            raise InvariantViolation(f"{where} must be true or false, got {value!r}")
        return value
    if tp is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise InvariantViolation(f"{where} must be an integer, got {value!r}")
        return value
    if tp is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise InvariantViolation(f"{where} must be a number, got {value!r}")
        return float(value)
    if tp is str:
        if not isinstance(value, str):
            raise InvariantViolation(f"{where} must be a string, got {value!r}")
        return value
    raise TypeError(f"{where}: unsupported field type {tp!r}")


def from_mapping(cls: type[T], data: Mapping[str, Any], *, path: str = "") -> T:
    """Build config dataclass `cls` from plain data, strictly.

    - an unknown key raises `InvariantViolation`;
    - a value of the wrong type raises `InvariantViolation` (ints are accepted for floats, lists for tuples);
    - `???` on a field that has a default contradicts the default and raises `InvariantViolation`;
    - `???` on a field without a default, or a missing required key, raises `ConfigMissing`;
    - a key the mapping leaves out takes the dataclass default, so the dataclass stays the single source;
    - value ranges are checked by the dataclass's own `__post_init__`.
    """
    if not (isinstance(cls, type) and is_dataclass(cls)):
        raise TypeError("from_mapping needs a config dataclass type")
    if not isinstance(data, Mapping):
        raise InvariantViolation(f"{path or cls.__name__}: expected a mapping, got {type(data).__name__}")
    hints = typing.get_type_hints(cls)
    names = {f.name for f in fields(cls)}
    unknown = sorted(set(map(str, data)) - names)
    if unknown:
        raise InvariantViolation(f"unknown keys for {cls.__name__}: {[path + k for k in unknown]}; known: {sorted(names)}")
    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        has_default = f.default is not DC_MISSING or f.default_factory is not DC_MISSING
        where = path + f.name
        if f.name not in data:
            if not has_default:
                raise ConfigMissing(f"{where} is required and has no default")
            continue
        value = data[f.name]
        if isinstance(value, str) and value == MISSING:
            if has_default:
                default = f.default if f.default is not DC_MISSING else f.default_factory()  # type: ignore[misc]
                raise InvariantViolation(f"{where}: '???' contradicts the dataclass default {default!r}; "
                                         "give a value or leave the key out")
            raise ConfigMissing(f"{where} is required and still '???'")
        kwargs[f.name] = _coerce(value, hints[f.name], where)
    return cls(**kwargs)


def apply_overrides(obj: T, overrides: Mapping[str, Any]) -> T:
    """`obj` with the fields named in `overrides` replaced, each validated as in `from_mapping`."""
    if not (is_dataclass(obj) and not isinstance(obj, type)):
        raise TypeError("apply_overrides needs a config dataclass instance")
    cls = type(obj)
    merged = {**to_mapping(obj), **dict(overrides)}
    # Re-validate the whole mapping so cross-field checks of __post_init__ see the new combination.
    return from_mapping(cls, merged)


def _yaml_scalar(value: Any) -> str:
    """One value in YAML flow syntax (exact for floats: PyYAML writes repr(float))."""
    import yaml

    if isinstance(value, float) and not math.isfinite(value):
        return ".inf" if value > 0 else ("-.inf" if value < 0 else ".nan")
    text = yaml.safe_dump(to_mapping(value), default_flow_style=True, width=10**9, allow_unicode=False)
    text = text.strip()
    if text.endswith("..."):
        text = text[: -3].strip()
    return text


def _render_fields(obj: Any, indent: int, lines: list[str]) -> None:
    """Append `key: value` lines of a dataclass instance, each preceded by its provenance comment."""
    pad = "  " * indent
    for f in fields(obj):
        source = str(f.metadata.get("source", "")).strip()
        doc = str(f.metadata.get("doc", "")).strip()
        comment = f"[{source}] {doc}".strip() if source else doc
        value = getattr(obj, f.name)
        if comment:
            lines.append(f"{pad}# {comment}")
        if is_dataclass(value) and not isinstance(value, type):
            lines.append(f"{pad}{f.name}:")
            _render_fields(value, indent + 1, lines)
        else:
            lines.append(f"{pad}{f.name}: {_yaml_scalar(value)}")


def render_yaml(obj: Any, *, header: Sequence[str] = (), key: str | None = None) -> str:
    """YAML text of a config instance: one key per field, its provenance as a comment above it.

    `key`, when given, nests the fields under one top-level key. Values are rendered exactly
    (`check_yaml` reads them back and compares).
    """
    if not (is_dataclass(obj) and not isinstance(obj, type)):
        raise TypeError("render_yaml needs a config dataclass instance")
    lines = [f"# {h}" if h else "#" for h in header]
    if key is not None:
        lines.append(f"{key}:")
        _render_fields(obj, 1, lines)
    else:
        _render_fields(obj, 0, lines)
    return "\n".join(lines) + "\n"


def check_yaml(cls: type, data: Mapping[str, Any], *, key: str | None = None) -> None:
    """Raise `InvariantViolation` unless `data` holds exactly the fields of `cls` with its default values."""
    section = data.get(key, None) if key is not None else data
    if not isinstance(section, Mapping):
        raise InvariantViolation(f"expected a mapping under {key!r}")
    known = sorted(f.name for f in fields(cls))
    if sorted(map(str, section)) != known:
        raise InvariantViolation(f"keys {sorted(map(str, section))} differ from the {cls.__name__} fields {known}")
    loaded = from_mapping(cls, section)
    default = cls()
    if loaded != default:
        diff = [n for n in known if getattr(loaded, n) != getattr(default, n)]
        raise InvariantViolation(f"values of {diff} differ from the {cls.__name__} defaults")


@dataclass(frozen=True)
class RunConfig:
    """Settings of one run that have no default (the run mode is deliberately never defaulted)."""

    mode: str = required(source="core/modes.py", doc="train | evaluate | infer_live | forensic_replay | lab (no default mode)")
    seed: int = required(source="run setting", doc="seed of every random draw of the run (route sampling, masking, splits)")
    site: str = required(source="run setting", doc="deployment site id (topics, calibration adapters)")
    enabled_proposals: tuple[str, ...] = setting((), source="governance/decisions.py",
                                                 doc="proposals (P-xx) whose code may run in this run; empty by default")

    def __post_init__(self) -> None:
        from nagahana.core.modes import RunMode

        if self.mode not in {m.value for m in RunMode}:
            raise InvariantViolation(f"run mode {self.mode!r} is not one of {[m.value for m in RunMode]}")
        from nagahana.governance import decisions

        for pid in self.enabled_proposals:
            if decisions.get(pid).status is not decisions.Status.PROPOSED:
                raise InvariantViolation(f"{pid} is not a proposal; only proposals are enabled per run")


def _components() -> dict[str, type]:
    """Component name -> config dataclass (the model configuration, `models/config/components.py`)."""
    from nagahana.models.config import components as c  # lazy: core does not import models at module load

    return {
        "inputs": c.FieldEncoderConfig, "graph": c.GraphConfig, "cvgae": c.CVGAEConfig,
        "decoder": c.DecoderConfig, "tstct": c.TSTCTConfig, "memory": c.MemoryConfig, "taaft": c.TAAFTConfig,
        "forecaster": c.ForecasterConfig, "advisor": c.AdvisorConfig, "verifier": c.VerifierConfig,
        "generator": c.GeneratorConfig, "training": c.TrainingConfig,
    }


def _site_class() -> type:
    from nagahana.models.config.components import SiteConfig  # lazy, as `_components`

    return SiteConfig


def site_config(overrides: str | Path | Mapping[str, Any] | None = None) -> Any:
    """The site configuration (D-63): `SiteConfig` defaults, then a validated override file or mapping."""
    cls = _site_class()
    if overrides is None:
        return cls()
    data = load_yaml(overrides) if isinstance(overrides, (str, Path)) else dict(overrides)
    section = data.get("site", data) if isinstance(data.get("site"), Mapping) else data
    return from_mapping(cls, section)


def component_config(name: str, overrides: str | Path | Mapping[str, Any] | None = None) -> Any:
    """The config of one model component: its dataclass defaults, then a validated override file or mapping.

    An override file may hold the fields at top level or under the component's name (the layout of the
    generated files under conf/model/).
    """
    comps = _components()
    if name not in comps:
        raise KeyError(f"unknown component {name!r}; known: {sorted(comps)}")
    cls = comps[name]
    if overrides is None:
        return cls()
    data = load_yaml(overrides) if isinstance(overrides, (str, Path)) else dict(overrides)
    section = data.get(name, data) if isinstance(data.get(name), Mapping) else data
    return from_mapping(cls, section)


def decision_options_from_mapping(data: Mapping[str, Any]) -> dict[str, str]:
    """Validate an override of held-decision options; returns ID -> option for `decisions.configure`.

    Keys are decision IDs or slugs (a top-level `decisions:` section is accepted). A held decision with a
    working option may not be set to `???` (that would contradict its recorded option); a held decision
    without one may stay `???` and is then simply not configured. Every option must be admissible.
    """
    from nagahana.governance import decisions

    section = data.get("decisions", data) if isinstance(data.get("decisions"), Mapping) else data
    out: dict[str, str] = {}
    for key, value in section.items():
        d = decisions.get(str(key))
        if d.status is not decisions.Status.HELD:
            raise InvalidOption(f"{d.id} ({d.slug}) is {d.status.value}; only held decisions take an option")
        if isinstance(value, str) and value == MISSING:
            if d.working is not None:
                raise InvariantViolation(f"{d.id}: '???' contradicts its working option {d.working!r} ({d.assumption})")
            continue
        if not isinstance(value, str):
            raise InvariantViolation(f"{d.id}: an option must be a string, got {value!r}")
        if value not in d.admissible:
            raise InvalidOption(f"{d.id}: {value!r} is not admissible; admissible: {list(d.admissible)}")
        out[d.id] = value
    return out


_GENERATED = "Generated from the typed dataclasses by nagahana.core.config.generate_conf (D-57). Do not edit by hand:"
_REGENERATE = "change the dataclass default and regenerate; tests compare this file with the code."


def _render_decisions() -> str:
    """conf/decisions/decisions.yaml: every held decision with its option in the build."""
    from nagahana.governance import decisions

    lines = [
        "# Options of held decisions (governance/decisions.py).",
        f"# {_GENERATED}",
        f"# {_REGENERATE}",
        "# A held decision shows its working option, recorded by the assumption named in its comment. A held",
        "# decision without a working option shows ??? and is resolved only when a run configures an option.",
        "decisions:",
    ]
    for d in decisions.by_status(decisions.Status.HELD):
        admissible = "; ".join(d.admissible) if d.admissible else "none recorded"
        title = d.title if d.title.endswith(("?", ".", "!")) else f"{d.title}."
        if d.working is not None:
            lines.append(f"  # [{d.id}, {d.assumption}] {title} Admissible: {admissible}.")
            lines.append(f"  {d.slug}: {_yaml_scalar(d.working)}")
        else:
            lines.append(f"  # [{d.id}] {title} No working option. Admissible: {admissible}.")
            lines.append(f"  {d.slug}: {_yaml_scalar(MISSING)}")
    return "\n".join(lines) + "\n"


def _render_root() -> str:
    """conf/config.yaml: the composition root (Hydra defaults list) and the run settings."""
    lines = [
        "# NagaHana root configuration (Hydra composition format; readable with PyYAML too).",
        f"# {_GENERATED}",
        f"# {_REGENERATE}",
        "# The run settings below have no default: ??? marks them as required (reading one raises).",
        "defaults:",
        "  - decisions: decisions",
        "  - site: site",
        *[f"  - model/{name}: {name}" for name in _components()],
        "  - _self_",
        "run:",
    ]
    for f in fields(RunConfig):
        source = str(f.metadata.get("source", ""))
        doc = str(f.metadata.get("doc", ""))
        lines.append(f"  # [{source}] {doc}")
        value = MISSING if (f.default is DC_MISSING and f.default_factory is DC_MISSING) else (
            f.default if f.default is not DC_MISSING else f.default_factory())  # type: ignore[misc]
        lines.append(f"  {f.name}: {_yaml_scalar(value)}")
    return "\n".join(lines) + "\n"


#: Packages whose own dataclasses render files under conf/ through this module: "module:function", the function
#: returning relative path -> text. They are imported only when the files are generated or checked.
CONF_PROVIDERS: tuple[str, ...] = ("nagahana.training.config:conf_files",)


def _provided_files() -> dict[str, str]:
    """The files of every `CONF_PROVIDERS` entry (an import or a duplicate path fails loudly)."""
    import importlib

    out: dict[str, str] = {}
    for spec in CONF_PROVIDERS:
        module, _, func = spec.partition(":")
        files = getattr(importlib.import_module(module), func)()
        for rel, text in files.items():
            if rel in out:
                raise InvariantViolation(f"conf/{rel} is generated by two providers")
            out[rel] = text
    return out


def conf_files() -> dict[str, str]:
    """Relative path under conf/ -> generated text: the root, the held-decision options, the site, every model
    component, and the files of `CONF_PROVIDERS`."""
    out: dict[str, str] = {"config.yaml": _render_root(), "decisions/decisions.yaml": _render_decisions()}
    site_cls = _site_class()
    out["site/site.yaml"] = render_yaml(site_cls(), header=(
        "The monitored site: it sizes the working memory of the one L model (D-63; lab/sizing.py).",
        _GENERATED, _REGENERATE), key="site")
    for name, cls in _components().items():
        doc = (cls.__doc__ or name).strip().splitlines()[0]
        header = (f"{doc} (models/config/components.py, {cls.__name__}).", _GENERATED, _REGENERATE)
        out[f"model/{name}/{name}.yaml"] = render_yaml(cls(), header=header, key=name)
    for rel, text in _provided_files().items():
        if rel in out:
            raise InvariantViolation(f"conf/{rel} is generated twice")
        out[rel] = text
    return out


def generate_conf(root: str | Path) -> list[Path]:
    """Write every generated file under `root` (UTF-8, LF line ends); returns the paths written."""
    base = Path(root)
    written = []
    for rel, text in conf_files().items():
        p = base / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8", newline="\n")
        written.append(p)
    return written


def check_conf(root: str | Path) -> list[str]:
    """Differences between `root` and the generated files: missing, changed and stray YAML files.

    Stray files are looked for at the top level of `root` and in the directories the generated files occupy
    (decisions/, site/, model/ and the providers' directories); other directories under conf/ belong to the
    packages that generate and check them (for example conf/analytics/). Returns an empty list when the files
    match the dataclasses exactly.
    """
    base = Path(root)
    expected = conf_files()
    problems: list[str] = []
    for rel, text in expected.items():
        p = base / rel
        if not p.exists():
            problems.append(f"missing: {rel}")
        elif p.read_text(encoding="utf-8").replace("\r\n", "\n") != text:
            problems.append(f"differs from the dataclasses: {rel}")
    owned = {rel.split("/", 1)[0] for rel in expected if "/" in rel}
    for p in sorted(base.rglob("*.yaml")):
        rel = p.relative_to(base).as_posix()
        top = rel.split("/", 1)[0] if "/" in rel else None
        if rel not in expected and (top is None or top in owned):
            problems.append(f"not generated from a dataclass: {rel}")
    return problems


def describe_yaml(path: str | Path) -> list[tuple[str, Any, str]]:
    """(dotted key, value, provenance) of every leaf of a YAML file, for `check-config`.

    Provenance comes from the dataclass field the key belongs to (a generated component file, the run
    settings, or the held-decision options); keys outside them are marked "no dataclass field".
    """
    from nagahana.governance import decisions

    data = load_yaml(path)
    comps = _components()
    rows: list[tuple[str, Any, str]] = []

    def walk(node: Any, prefix: str, prov: Mapping[str, tuple[str, str]] | None) -> None:
        if isinstance(node, Mapping):
            for k, v in node.items():
                key = f"{prefix}.{k}" if prefix else str(k)
                if prov is not None and str(k) in prov and not isinstance(v, (Mapping, list)):
                    src, _doc = prov[str(k)]
                    rows.append((key, v, src or "no provenance recorded"))
                elif isinstance(v, (Mapping, list)):
                    walk(v, key, prov)
                else:
                    rows.append((key, v, "no dataclass field"))
        elif isinstance(node, list):
            for i, v in enumerate(node):
                walk(v, f"{prefix}[{i}]", None)
        else:
            rows.append((prefix, node, "no dataclass field"))

    for top, value in data.items():
        if top in comps and isinstance(value, Mapping):
            walk(value, top, provenance(comps[top]))
        elif top == "run" and isinstance(value, Mapping):
            walk(value, top, provenance(RunConfig))
        elif top == "site" and isinstance(value, Mapping):
            walk(value, top, provenance(_site_class()))
        elif top == "decisions" and isinstance(value, Mapping):
            for slug, option in value.items():
                d = decisions.get(str(slug))
                src = f"{d.id} held; working option {d.assumption}" if d.working is not None else f"{d.id} held; no working option"
                rows.append((f"decisions.{slug}", option, src))
        elif isinstance(value, (Mapping, list)):
            walk(value, str(top), None)
        else:
            rows.append((str(top), value, "no dataclass field"))
    return rows


__all__ = [
    "MISSING", "RunConfig", "apply_overrides", "check_conf", "check_yaml", "component_config", "conf_files",
    "decision_options_from_mapping", "describe_yaml", "from_mapping", "generate_conf", "get_required",
    "iter_missing", "load_yaml", "provenance", "render_yaml", "required", "setting", "site_config", "to_mapping",
]
