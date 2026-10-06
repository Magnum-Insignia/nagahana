"""Saved statistical-physics trajectories for offline analysis (D-56).

A trajectory is the sequence of trigger times with named float64 series (the `series` and `growth`
of the readings). It is stored in one .npz file (no pickled objects):

    statphys.format            "nagahana-statphys-trajectory-v1"
    times                      float64 [m] epoch seconds
    regular                    bool [m] cadence triggers
    series.<name>              float64 [m]
    meta.<key>                 str scalars (network, dataset, model hash ...)

`load_trajectory` also reads a `ModelOutputs` file written by `evaluation.predictions.save_outputs`
whose component arrays hold statphys.* entries (evaluation.py), so a run's archived predictions can
be analysed without re-running the model.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from nagahana.statphys.evaluation import PREFIX, series_from_component
from nagahana.statphys.trajectory import StatPhysReading

FORMAT = "nagahana-statphys-trajectory-v1"


@dataclass
class Trajectory:
    """Trigger times, regular-grid flags and named series of one stream (module docstring)."""

    times: np.ndarray
    series: dict[str, np.ndarray]
    regular: np.ndarray
    meta: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.times = np.asarray(self.times, dtype=np.float64)
        m = self.times.shape[0]
        if self.times.ndim != 1 or not bool(np.all(np.isfinite(self.times))):
            raise ValueError("times must be a finite 1-D array")
        if m > 1 and not bool(np.all(np.diff(self.times) > 0)):
            raise ValueError("times must increase strictly")
        self.regular = np.asarray(self.regular, dtype=bool)
        if self.regular.shape != (m,):
            raise ValueError("regular must be [m]")
        self.series = {str(k): np.asarray(v, dtype=np.float64) for k, v in self.series.items()}
        for k, v in self.series.items():
            if v.shape != (m,):
                raise ValueError(f"series {k!r} must be [m] with m = {m}")

    @staticmethod
    def from_readings(readings: Sequence[StatPhysReading], meta: Mapping[str, str] | None = None) -> Trajectory:
        """The trajectory of a sequence of readings (series and growth rates; missing values NaN)."""
        names = sorted({k for r in readings for k in (*r.series, *r.growth)})
        series = {n: np.array([r.series.get(n, r.growth.get(n, np.nan)) for r in readings], dtype=np.float64)
                  for n in names}
        return Trajectory(times=np.array([r.time for r in readings], dtype=np.float64), series=series,
                          regular=np.array([r.regular for r in readings], dtype=bool), meta=dict(meta or {}))

    def values(self, name: str, *, regular_only: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """(values, times) of one series, on the regular grid only by default (early-warning input)."""
        if name not in self.series:
            raise KeyError(f"no series {name!r}; known: {sorted(self.series)}")
        keep = self.regular if regular_only else np.ones_like(self.regular)
        return self.series[name][keep], self.times[keep]


def save_trajectory(traj: Trajectory, path: str | Path) -> Path:
    """Write a trajectory as .npz (module docstring)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    out: dict[str, np.ndarray] = {"statphys.format": np.asarray(FORMAT), "times": traj.times, "regular": traj.regular}
    for k, v in traj.series.items():
        out[f"series.{k}"] = v
    for k, text in traj.meta.items():
        out[f"meta.{k}"] = np.asarray(str(text))
    np.savez_compressed(p, allow_pickle=False, **out)
    return p


def load_trajectory(path: str | Path) -> Trajectory:
    """Read a trajectory file, or the statphys component arrays of a ModelOutputs file."""
    with np.load(Path(path), allow_pickle=False) as data:
        files = set(data.files)
        if "statphys.format" in files:
            if str(data["statphys.format"]) != FORMAT:
                raise ValueError(f"unknown trajectory format {str(data['statphys.format'])!r}")
            series = {k[len("series."):]: data[k] for k in files if k.startswith("series.")}
            meta = {k[len("meta."):]: str(data[k]) for k in files if k.startswith("meta.")}
            return Trajectory(times=data["times"], series=series, regular=data["regular"], meta=meta)
        comp = {k[len("component."):]: data[k] for k in files if k.startswith("component." + PREFIX)}
        if not comp:
            raise ValueError(f"{path}: neither a statphys trajectory nor a ModelOutputs file with statphys components")
        times, series, regular = series_from_component(comp)
        meta = {"model": str(data["model"])} if "model" in files else {}
        return Trajectory(times=times, series=series, regular=regular, meta=meta)


__all__ = ["FORMAT", "Trajectory", "load_trajectory", "save_trajectory"]
