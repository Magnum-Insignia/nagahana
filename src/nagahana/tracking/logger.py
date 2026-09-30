"""Experiment tracking behind one small interface (MLflow in the approved stack, D-09).

Every run records its configuration, the decision registry state and the enabled proposals. That way
a result can always be traced to what was decided at the time (Q-39: nothing silently defaulted,
nothing silently changed).

- `MLflowLogger`: imports MLflow lazily, so the core never requires it.
- `NullLogger`: records nothing. For unit tests and dry runs; it is not a substitute for tracking
  real experiments.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

from nagahana.governance import decisions


class RunLogger(Protocol):
    """What training and evaluation code may call."""

    def log_params(self, params: Mapping[str, Any]) -> None: ...
    def log_metrics(self, metrics: Mapping[str, float], step: int | None = None) -> None: ...
    def log_artifact(self, path: str) -> None: ...


def governance_snapshot() -> dict[str, str]:
    """Status of every decision and proposal, e.g. {'D-12': 'held', …}, for logging with each run."""
    return {d.id: d.status.value for d in decisions.all_entries()}


class NullLogger:
    """Records nothing (tests, dry runs)."""

    def log_params(self, params: Mapping[str, Any]) -> None:
        return None

    def log_metrics(self, metrics: Mapping[str, float], step: int | None = None) -> None:
        return None

    def log_artifact(self, path: str) -> None:
        return None


class MLflowLogger:
    """MLflow-backed logger; logs the governance snapshot at start."""

    def __init__(self, experiment: str, *, tracking_uri: str) -> None:
        import mlflow  # lazy: optional dependency (extra "track")

        self._mlflow = mlflow
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment)
        self._run = mlflow.start_run()
        mlflow.log_params({f"governance.{k}": v for k, v in governance_snapshot().items()})

    def log_params(self, params: Mapping[str, Any]) -> None:
        self._mlflow.log_params(dict(params))

    def log_metrics(self, metrics: Mapping[str, float], step: int | None = None) -> None:
        self._mlflow.log_metrics(dict(metrics), step=step)

    def log_artifact(self, path: str) -> None:
        self._mlflow.log_artifact(path)
