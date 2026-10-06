"""Run loggers of the training pipeline, behind the `tracking.logger.RunLogger` interface (D-09).

- "jsonl": one JSON object per call in `<run>/logs/metrics.jsonl` (and params in `params.jsonl`),
  flushed per line, readable without any tracking server: the default of an offline run.
- "mlflow": `tracking.logger.MLflowLogger`, imported lazily (the `track` extra), with a tracking URI.
- "null": `tracking.logger.NullLogger` (tests).

Only rank 0 logs; other ranks get a null logger.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from nagahana.core.errors import ConfigMissing
from nagahana.tracking.logger import NullLogger, RunLogger, governance_snapshot


def _clean(v: Any) -> Any:
    # JSON has no NaN or infinity: they are written as strings so a reader sees them.
    if isinstance(v, float) and not math.isfinite(v):
        return "nan" if math.isnan(v) else ("inf" if v > 0 else "-inf")
    return v


class JsonlLogger:
    """Append-only JSON-lines logger (module docstring)."""

    def __init__(self, directory: str | Path) -> None:
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._metrics = (self.dir / "metrics.jsonl").open("a", encoding="utf-8")
        self._params = (self.dir / "params.jsonl").open("a", encoding="utf-8")
        self.log_params({f"governance.{k}": v for k, v in governance_snapshot().items()})

    def log_params(self, params: Mapping[str, Any]) -> None:
        self._params.write(json.dumps({"time": time.time(), **{k: _clean(v) for k, v in params.items()}}, default=str) + "\n")
        self._params.flush()

    def log_metrics(self, metrics: Mapping[str, float], step: int | None = None) -> None:
        row = {"time": time.time(), "step": step, **{k: _clean(float(v)) for k, v in metrics.items()}}
        self._metrics.write(json.dumps(row) + "\n")
        self._metrics.flush()

    def log_artifact(self, path: str) -> None:
        self.log_params({"artifact": path})

    def close(self) -> None:
        self._metrics.close()
        self._params.close()


def make_logger(kind: str, *, run_dir: str | Path, is_main: bool, experiment: str = "nagahana",
                tracking_uri: str | None = None) -> RunLogger:
    """The run's logger (module docstring)."""
    if not is_main or kind == "null":
        return NullLogger()
    if kind == "jsonl":
        return JsonlLogger(Path(run_dir) / "logs")
    if kind == "mlflow":
        if tracking_uri is None:
            raise ConfigMissing("an MLflow tracking URI is required (no default location)")
        from nagahana.tracking.logger import MLflowLogger  # lazy: optional dependency

        return MLflowLogger(experiment, tracking_uri=tracking_uri)
    raise ConfigMissing(f"unknown logger {kind!r}; known: 'jsonl', 'null', 'mlflow'")


__all__ = ["JsonlLogger", "make_logger"]
