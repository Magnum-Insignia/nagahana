"""Configuration helpers that refuse to invent values.

Convention
----------
Configs live in `conf/` as YAML in Hydra's composition format. A value that nobody has decided yet
is written `???`. That is Hydra/OmegaConf's *mandatory missing value*, so the same files work
unchanged once Hydra is installed (tech stack D-09). Until then this module reads them with PyYAML.

Two kinds of `???`
------------------
1. A held design decision (e.g. `taaft.policy_coupling: ???`, D-12). It becomes a value only when the
   owner decides [Q-39].
2. A hyperparameter nobody has chosen yet (e.g. a learning rate). It becomes a value when an
   experiment chooses one and records why (MLflow run + note).

In both cases reading the key raises `ConfigMissing` naming the key. Nothing silently falls back.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

from nagahana.core.errors import ConfigMissing

#: Hydra/OmegaConf's marker for "mandatory, not yet set".
MISSING = "???"


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Read one YAML file into a dict (no Hydra composition; see `conf/README.md`)."""
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
        raise ConfigMissing(
            f"Config key {key!r} is still '???' (undecided). Decide it and record why; "
            "no default is applied."
        )
    return node


def iter_missing(cfg: Mapping[str, Any], prefix: str = "") -> Iterator[str]:
    """Yield every dotted key whose value is `???`, depth first.

    Used by `python -m nagahana check-config` to list what is still undecided in a config.
    """
    for k, v in cfg.items():
        dotted = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, Mapping):
            yield from iter_missing(v, dotted)
        elif v == MISSING:
            yield dotted
