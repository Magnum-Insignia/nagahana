"""Situations: the model records that feedback refers to, resolved at fit time (AS-832).

A situation is one Forecaster trigger as the deployed model saw it: TAAFT's analysis of that trigger
(the conditioning input x of every policy), the forecast the deployed Forecaster produced (raw, before any
temperature), the Monitor statistics at that time (the trust head's input), and the trigger's entity
table. Feedback names a situation by id (`Provenance.refers_to`); the tensors are not copied into the
ledger, which stays small and text-only, but a feedback event may carry `situation_digest`, the
SHA-256 of the situation's tensors, and a situation whose digest differs from the one the analyst saw is
excluded from learning with a recorded reason.

Shapes (one trigger): analysis tensors [1, 1, ...] as `models.advisor.model.slice_trigger` returns them;
forecast tensors [1, 1, ...] (`slice_forecast`); monitor_features [6]; entity_internal, entity_kind [V].

`repeat_trigger(an1, n)` expands a one-trigger analysis to [1, n, ...] (views, no copy) so that n routes
or n items of the same situation are evaluated in one teacher-forced pass of the Forecaster.

Files: `save_situations` / `load_situations` store many situations in one tensor file
(`params.save_tensors`, loaded with weights_only=True) with a canonical-JSON index of the non-tensor
fields.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import torch

from nagahana.core.errors import InvariantViolation
from nagahana.models.advisor.model import slice_trigger
from nagahana.models.batch import AnalysisOut, ForecastOut
from nagahana.models.verifier.params import load_tensors, save_tensors, tensor_digest

_FORECAST_TENSORS: tuple[str, ...] = tuple(f.name for f in fields(ForecastOut) if f.name not in ("horizon_k", "routes_n"))


def slice_forecast(fo: ForecastOut, b: int, m: int) -> ForecastOut:
    """One trigger of a `ForecastOut` as [1, 1, ...] tensors."""
    kw: dict[str, Any] = {name: getattr(fo, name)[b:b + 1, m:m + 1] for name in _FORECAST_TENSORS}
    return ForecastOut(**kw, horizon_k=fo.horizon_k, routes_n=fo.routes_n)


def _check_one(name: str, t: torch.Tensor) -> None:
    if t.dim() < 2 or tuple(t.shape[:2]) != (1, 1):
        raise InvariantViolation(f"situation tensor {name} must be one trigger [1, 1, ...], got {tuple(t.shape)}")


@dataclass(frozen=True)
class Situation:
    """One trigger as the deployed model saw it (module docstring)."""

    situation_id: str
    analysis: AnalysisOut
    forecast: ForecastOut | None = None
    monitor_features: torch.Tensor | None = None
    entity_internal: torch.Tensor | None = None
    entity_kind: torch.Tensor | None = None
    model_hash: str = ""
    time: float = 0.0

    def __post_init__(self) -> None:
        if not self.situation_id.strip():
            raise InvariantViolation("a situation needs its id")
        an = self.analysis
        for name in ("context", "token_mask", "y0", "y"):
            _check_one(f"analysis.{name}", getattr(an, name))
        for name, t in an.readouts.items():
            _check_one(f"analysis.readouts[{name}]", t)
        for key in ("compromise", "stage"):
            if key not in an.readouts:
                raise InvariantViolation(f"a situation's analysis needs the {key!r} readout")
        v = an.readouts["compromise"].shape[-1]
        if self.forecast is not None:
            for name in _FORECAST_TENSORS:
                _check_one(f"forecast.{name}", getattr(self.forecast, name))
        if self.monitor_features is not None and tuple(self.monitor_features.shape) != (6,):
            raise InvariantViolation("monitor_features must be the 6 Monitor statistics (heads.monitor_features)")
        for name in ("entity_internal", "entity_kind"):
            t = getattr(self, name)
            if t is not None and tuple(t.shape) != (v,):
                raise InvariantViolation(f"{name} must have one entry per entity ({v})")

    @property
    def n_entities(self) -> int:
        """V, the entities of the trigger's window."""
        return int(self.analysis.readouts["compromise"].shape[-1])

    def tensors(self) -> dict[str, torch.Tensor]:
        """Every tensor of the situation under a stable flat name (digest and persistence)."""
        an = self.analysis
        out: dict[str, torch.Tensor] = {"analysis/context": an.context, "analysis/token_mask": an.token_mask,
                                        "analysis/y0": an.y0, "analysis/y": an.y, "analysis/energy_trace": an.energy_trace}
        out |= {f"analysis/readouts/{k}": t for k, t in an.readouts.items()}
        out |= {f"analysis/lens_energy/{k}": t for k, t in an.lens_energy.items()}
        out |= {f"analysis/lens_share/{k}": t for k, t in an.lens_share.items()}
        if self.forecast is not None:
            out |= {f"forecast/{name}": getattr(self.forecast, name) for name in _FORECAST_TENSORS}
        for name in ("monitor_features", "entity_internal", "entity_kind"):
            t = getattr(self, name)
            if t is not None:
                out[name] = t
        return out

    @property
    def digest(self) -> str:
        """SHA-256 of the situation's tensors (params.tensor_digest), the value feedback may pin."""
        return tensor_digest(self.tensors())


def trigger_situation(situation_id: str, analysis: AnalysisOut, b: int, m: int, *, forecast: ForecastOut | None = None,
                      monitor_features: torch.Tensor | None = None, entity_internal: torch.Tensor | None = None,
                      entity_kind: torch.Tensor | None = None, model_hash: str = "", time: float = 0.0) -> Situation:
    """The situation of trigger (b, m) of a batch: analysis and forecast sliced, everything detached and on CPU-agnostic views."""
    an1 = slice_trigger(analysis, b, m)
    an1 = AnalysisOut(context=an1.context.detach(), token_mask=an1.token_mask, imagination_kv=[], y0=an1.y0.detach(),
                      y=an1.y.detach(), energy_trace=an1.energy_trace.detach(),
                      lens_energy={k: v.detach() for k, v in an1.lens_energy.items()},
                      lens_share={k: v.detach() for k, v in an1.lens_share.items()},
                      readouts={k: v.detach() for k, v in an1.readouts.items()}, passes=an1.passes,
                      descent_steps=an1.descent_steps)
    fo = None if forecast is None else slice_forecast(forecast, b, m)
    return Situation(situation_id=situation_id, analysis=an1, forecast=fo,
                     monitor_features=None if monitor_features is None else monitor_features.detach().reshape(6),
                     entity_internal=entity_internal, entity_kind=entity_kind, model_hash=model_hash, time=float(time))


def repeat_trigger(an1: AnalysisOut, n: int) -> AnalysisOut:
    """A one-trigger analysis [1, 1, ...] expanded to [1, n, ...] (expand views; no copy)."""
    if n < 1:
        raise ValueError("n must be >= 1")

    def rep(t: torch.Tensor) -> torch.Tensor:
        return t.expand(1, n, *t.shape[2:])

    return AnalysisOut(context=rep(an1.context), token_mask=rep(an1.token_mask), imagination_kv=[], y0=rep(an1.y0),
                       y=rep(an1.y), energy_trace=an1.energy_trace,
                       lens_energy={k: rep(v) for k, v in an1.lens_energy.items()},
                       lens_share={k: rep(v) for k, v in an1.lens_share.items()},
                       readouts={k: rep(v) for k, v in an1.readouts.items()}, passes=an1.passes,
                       descent_steps=an1.descent_steps)


def save_situations(path: str | Path, situations: Mapping[str, Situation]) -> str:
    """Write situations to one tensor file (module docstring); returns the file's SHA-256."""
    tensors: dict[str, torch.Tensor] = {}
    index: list[dict[str, Any]] = []
    for i, (sid, s) in enumerate(sorted(situations.items())):
        if sid != s.situation_id:
            raise InvariantViolation(f"situation stored under {sid!r} has id {s.situation_id!r}")
        for name, t in s.tensors().items():
            tensors[f"{i}:{name}"] = t
        an = s.analysis
        index.append({
            "id": sid, "model_hash": s.model_hash, "time": float(s.time), "passes": int(an.passes),
            "descent_steps": int(an.descent_steps), "readouts": sorted(an.readouts), "lens_energy": sorted(an.lens_energy),
            "lens_share": sorted(an.lens_share),
            "forecast": None if s.forecast is None else {"horizon_k": int(s.forecast.horizon_k), "routes_n": int(s.forecast.routes_n)},
            "optional": sorted(n for n in ("monitor_features", "entity_internal", "entity_kind") if getattr(s, n) is not None),
            "digest": s.digest,
        })
    return save_tensors(path, tensors, {"kind": "verifier-situations", "situations": index})


def load_situations(path: str | Path, *, expected_sha256: str | None = None) -> dict[str, Situation]:
    """Read a file of `save_situations`; every situation's digest is re-checked."""
    tensors, meta = load_tensors(path, expected_sha256=expected_sha256)
    if meta.get("kind") != "verifier-situations":
        raise InvariantViolation(f"{path}: not a situations file")
    out: dict[str, Situation] = {}
    for i, rec in enumerate(meta["situations"]):
        def t(name: str, i: int = i) -> torch.Tensor:
            key = f"{i}:{name}"
            if key not in tensors:
                raise InvariantViolation(f"{path}: situation {i} lacks tensor {name!r}")
            return tensors[key]

        an = AnalysisOut(context=t("analysis/context"), token_mask=t("analysis/token_mask"), imagination_kv=[],
                         y0=t("analysis/y0"), y=t("analysis/y"), energy_trace=t("analysis/energy_trace"),
                         lens_energy={k: t(f"analysis/lens_energy/{k}") for k in rec["lens_energy"]},
                         lens_share={k: t(f"analysis/lens_share/{k}") for k in rec["lens_share"]},
                         readouts={k: t(f"analysis/readouts/{k}") for k in rec["readouts"]},
                         passes=int(rec["passes"]), descent_steps=int(rec["descent_steps"]))
        fo = None
        if rec["forecast"] is not None:
            fo = ForecastOut(**{name: t(f"forecast/{name}") for name in _FORECAST_TENSORS},
                             horizon_k=int(rec["forecast"]["horizon_k"]), routes_n=int(rec["forecast"]["routes_n"]))
        opt = set(rec["optional"])
        s = Situation(situation_id=str(rec["id"]), analysis=an, forecast=fo,
                      monitor_features=t("monitor_features") if "monitor_features" in opt else None,
                      entity_internal=t("entity_internal") if "entity_internal" in opt else None,
                      entity_kind=t("entity_kind") if "entity_kind" in opt else None,
                      model_hash=str(rec["model_hash"]), time=float(rec["time"]))
        if s.digest != rec["digest"]:
            raise InvariantViolation(f"{path}: situation {s.situation_id!r} does not match its recorded digest")
        out[s.situation_id] = s
    return out


__all__ = ["Situation", "load_situations", "repeat_trigger", "save_situations", "slice_forecast", "trigger_situation"]
