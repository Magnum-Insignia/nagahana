"""Lazy access to the optional libraries of the [baselines] extra (D-55).

scikit-learn, XGBoost, LightGBM and CatBoost are imported only when a reproduction that needs them is built, so
the core package and every PyTorch-only reproduction import without them. A missing library raises
`MissingDependency` naming the library, the baseline that needs it and the install command.
"""

from __future__ import annotations

import importlib
from types import ModuleType

from nagahana.baselines.published.base import MissingDependency

_INSTALL = "install the optional extra: pip install 'nagahana[baselines]'"
_DISTRIBUTION = {"sklearn": "scikit-learn", "xgboost": "xgboost", "lightgbm": "lightgbm", "catboost": "catboost"}


def require(module: str, *, needed_by: str) -> ModuleType:
    """Import `module` (for example "sklearn.ensemble"), or raise MissingDependency with instructions."""
    root = module.split(".", 1)[0]
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        dist = _DISTRIBUTION.get(root, root)
        raise MissingDependency(f"{needed_by} needs {dist} ({module}); {_INSTALL}") from exc


def version_of(module: str) -> str:
    """Installed version string of a library root ("sklearn", "xgboost", "lightgbm", "catboost"), or "" if absent."""
    try:
        mod = importlib.import_module(module.split(".", 1)[0])
    except ImportError:
        return ""
    return str(getattr(mod, "__version__", ""))


def available(module: str) -> bool:
    """True when `module` imports."""
    try:
        importlib.import_module(module)
    except ImportError:
        return False
    return True
