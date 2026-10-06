"""Stage 4: zero-shot validation with human-feedback confidence calibration (D-22 step 6; real data only).

Purpose (build-spec section 3; architecture section 5; D-23; D-13 held -> AS-26)
---------------------------------------------------------------------------------
1. Evaluation (`evaluate_split`, `zero_shot_evaluate`; RunMode.EVALUATE): the trained model forecasts on
   real segments of one split, test (known families) or zero-shot (novel families and held-out networks,
   AS-35, D-16 held), in stream order with the carry, and every output goes through the shared prediction
   contract `evaluation/predictions.ModelOutputs` (AS-591):
       forecast       one unit per usable trigger: P_inf(k) for k = 1..K, the event step from the
                      infiltration timeline (AS-18), the observed steps (the label horizon, raised to the
                      event step for an observed event), the route-mixture hazard, the per-route
                      cumulative curves and their weights (for CRPS)
       stage          one unit per active entity with a known stage label at a trigger: TAAFT's stage
                      posterior (float64, D-54) against the label of the entity's latest update (AS-414)
       time_to_event  one unit per usable trigger: risk P_inf(K), survival S(k w) = 1 - P_inf(k) on the
                      step grid, the event time or the censoring time
   meta: trigger epoch time, dataset, network, window family, novelty ("known" / "novel" on zero-shot
   units only, D-23), split. Zero-shot units are reported separately for known and novel families;
   segments that are not real, or not of the evaluated split, are refused before anything runs.
   Detection metrics of the problem statement (precision, recall, F1, FPR, FNR, detection error) use
   the decision P_inf(K) >= theta (AS-419), with the Brier score and ECE beside them.
2. Confidence calibration with human feedback (`calibration_pairs`, `propose_calibration`): resolved
   outcome-forecast pairs per output family (P_inf, compromise, stage confidence) feed the configured
   calibrators (training/feedback.py; built in: the maximum-likelihood temperature). A proposal is
   applied only through `models.verifier.gate.apply_calibration` with a HumanCommand("apply-calibration")
   (`apply_site_temperature`), so nothing changes without a human (D-21).
3. Site adapters (`SiteCalibrator`, AS-26, AS-420, AS-598): LoRA adapters (rank and alpha from
   `TrainingConfig`) on the attention projections (q, k, v, o) of TSTCT and TAAFT, trained on the site's
   unlabelled traffic (stage-1 objective plus the stage-2 objective without the malignity term) and the
   analyst-confirmed alerts (BCE of the compromise readout of the alerted entity at the first trigger at
   or after the alert); attaching and training them requires a HumanCommand("update-site-adapter"). The
   data requirement (24 h of traffic, 50 confirmed alerts) is checked and reported; a shorter run must
   say so explicitly (`allow_short`).

Ablation switches of the inference tier (training/ablation.py, AS-579) apply here through `switches`.

Decisions: D-13 (held), D-16 (held), D-21, D-23, D-50 (clock features on only for site calibration: a
config switch the operator sets). Assumptions: AS-18, AS-26, AS-35, AS-414, AS-419, AS-420, AS-579,
AS-591, AS-598.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from nagahana.core.errors import InvariantViolation
from nagahana.core.modes import RunMode, require_mode
from nagahana.data.sampling import Role, SplitManifest, WindowRecord, label_limits
from nagahana.data.stream import StreamContext, StreamLoader
from nagahana.data.windows import PreparedSource
from nagahana.evaluation.calibration import ece
from nagahana.evaluation.metrics import Confusion, report
from nagahana.evaluation.predictions import (
    ForecastPredictions,
    ModelOutputs,
    StagePredictions,
    TimeToEventPredictions,
    make_meta,
)
from nagahana.governance.assumptions import assume
from nagahana.models.batch import AnalysisOut, ForecastOut, LabelBatch, WindowBatch
from nagahana.models.config import NagaHanaConfig
from nagahana.models.forecaster.losses import survival_targets
from nagahana.models.nagahana import NagaHana
from nagahana.models.verifier.gate import APPLY_CALIBRATION, apply_calibration, require_command
from nagahana.models.verifier.reports import CalibrationProposal, TemperatureState
from nagahana.models.vocab import STAGES
from nagahana.nn.lora import attach_lora, lora_parameters
from nagahana.physics.term import PhysicsTerm
from nagahana.pipeline.freezing import freeze
from nagahana.roles.contracts import HumanCommand
from nagahana.training.assumptions import use
from nagahana.training.carry import PreparedBatch, StreamBridge
from nagahana.training.config import OptimConfig, Stage1Options, Stage2Options
from nagahana.training.engine import Draws
from nagahana.training.feedback import CalibrationPairs, build_calibrators
from nagahana.training.optim import HybridStepper, named_trainable
from nagahana.training.randomness import derive_seed, make_generator
from nagahana.training.stage1 import stage1_loss
from nagahana.training.stage2 import frozen_perception, sample_budgets, stage2_step_loss
from nagahana.training.stage3 import readout_targets, trigger_outcomes

UPDATE_SITE_ADAPTER = "update-site-adapter"
LORA_NAMES: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")
STAGE_NAMES: tuple[str, ...] = tuple(name for name, _ in STAGES)


@dataclass(frozen=True)
class EvalSettings:
    """Run-time budgets and the decision threshold of an evaluation (all explicit)."""

    tstct_passes: int
    taaft_passes: int
    descent_steps: int
    horizon_k: int
    routes_n: int
    threshold: float
    seed: int

    @classmethod
    def for_model(cls, model: NagaHana, *, threshold: float, seed: int, tstct_passes: int | None = None,
                  taaft_passes: int | None = None, descent_steps: int | None = None, horizon_k: int | None = None,
                  routes_n: int | None = None) -> EvalSettings:
        """Budgets left None take the model's run-time defaults (D-44)."""
        cfg = model.cfg
        return cls(tstct_passes=tstct_passes or cfg.tstct.default_passes, taaft_passes=taaft_passes or cfg.taaft.default_passes,
                   descent_steps=descent_steps or cfg.taaft.descent_steps, horizon_k=horizon_k or cfg.forecaster.horizon_k,
                   routes_n=routes_n or cfg.forecaster.routes_n, threshold=threshold, seed=seed)


@dataclass
class ZeroShotReport:
    """Metrics per group ("known", "novel"; or "all" for the test split), the outputs, the pairs."""

    groups: dict[str, dict[str, float]]
    pairs: dict[str, tuple[list[float], list[float]]]
    triggers_seen: int
    outputs: ModelOutputs | None = None
    calibration: CalibrationPairs | None = None
    notes: list[str] = field(default_factory=list)


def check_zero_shot(manifest: SplitManifest, segments: Sequence[WindowRecord]) -> None:
    """Refuse anything but REAL zero-shot segments (D-23: zero-shot never sees variants)."""
    check_split(manifest, segments, Role.ZERO_SHOT)


def check_split(manifest: SplitManifest, segments: Sequence[WindowRecord], role: Role) -> None:
    """Refuse segments that are not real or not of `role` (evaluation reads real data only, D-23, AS-575)."""
    for r in segments:
        if r.origin != "real":
            raise InvariantViolation(f"{r.id}: evaluation reads real data only (D-23); got origin {r.origin!r}")
        if manifest.role.get(r.id) is not role:
            raise InvariantViolation(f"{r.id}: not a {role.value} segment of the manifest")


def group_metrics(p: Sequence[float], y: Sequence[float], *, threshold: float) -> dict[str, float]:
    """Problem-statement metrics + Brier + ECE of one group (NaN where undefined)."""
    if not p:
        return {"n": 0.0}
    pt, yt = torch.tensor(p, dtype=torch.float64), torch.tensor(y, dtype=torch.float64)
    c = Confusion.from_predictions(yt > 0.5, pt >= threshold)
    out = report(c)
    out["n"] = float(len(p))
    out["brier"] = float(((pt - yt) ** 2).mean())
    out["ece"] = float(ece(pt, yt, bins=10))
    return out


def zero_shot_loader(sources: Sequence[PreparedSource], segments: Sequence[WindowRecord], manifest: SplitManifest,
                     cfg: NagaHanaConfig, *, lanes: int) -> StreamLoader:
    """The stream loader of the zero-shot segments (checked first), with novelty marks and label limits."""
    check_zero_shot(manifest, segments)
    novelty = {r.id: manifest.novelty[r.id].value for r in segments if r.id in manifest.novelty}
    return StreamLoader(sources, segments, cfg, lanes=lanes, novelty=novelty, label_limits=label_limits(manifest))


def split_loader(sources: Sequence[PreparedSource], segments: Sequence[WindowRecord], manifest: SplitManifest,
                 cfg: NagaHanaConfig, *, role: Role, lanes: int) -> StreamLoader:
    """The stream loader of the segments of one evaluated split (checked first), in time order."""
    check_split(manifest, segments, role)
    novelty = {r.id: manifest.novelty[r.id].value for r in segments if r.id in manifest.novelty} if role is Role.ZERO_SHOT else {}
    ordered = sorted(segments, key=lambda r: (r.t_start, r.id))
    return StreamLoader(sources, ordered, cfg, lanes=lanes, novelty=novelty, label_limits=label_limits(manifest))


@dataclass
class _Units:
    """Prediction units accumulated over batches (module docstring)."""

    p_inf: list[np.ndarray] = field(default_factory=list)
    hazard: list[np.ndarray] = field(default_factory=list)
    ensemble: list[np.ndarray] = field(default_factory=list)
    ensemble_w: list[np.ndarray] = field(default_factory=list)
    event_step: list[int] = field(default_factory=list)
    observed: list[int] = field(default_factory=list)
    meta: list[dict[str, Any]] = field(default_factory=list)
    stage_probs: list[np.ndarray] = field(default_factory=list)
    stage_label: list[int] = field(default_factory=list)
    stage_meta: list[dict[str, Any]] = field(default_factory=list)
    comp_p: list[float] = field(default_factory=list)
    comp_y: list[float] = field(default_factory=list)
    stage_conf: list[float] = field(default_factory=list)
    stage_correct: list[float] = field(default_factory=list)


def mixture_hazard(p_inf: np.ndarray) -> np.ndarray:
    """h_k = (P(k) - P(k-1)) / (1 - P(k-1)), P(0) = 0, of a non-decreasing cumulative curve (float64)."""
    prev = np.concatenate([np.zeros((p_inf.shape[0], 1)), p_inf[:, :-1]], axis=1)
    denom = np.clip(1.0 - prev, 1e-15, None)
    return np.clip((p_inf - prev) / denom, 0.0, 1.0)


def _collect(units: _Units, *, an: AnalysisOut, fo: ForecastOut, window: WindowBatch, labels: LabelBatch,
             ctxs: Sequence[StreamContext], meta_of: dict[str, dict[str, Any]], split: str, window_seconds: float,
             horizon_k: int) -> int:
    """Append the units of one batch (AS-591). Returns the number of real triggers seen."""
    assume("AS-18", by=__name__)
    use("AS-591", by=__name__)
    trig = window.triggers
    ev, cens, usable = survival_targets(trig.time, trig.mask, labels.entity_infiltrated_at, window.entity_internal,
                                        labels.label_horizon, window_seconds=window_seconds, horizon_k=horizon_k)
    tau = trig.time.to(torch.float64)
    end = labels.label_horizon.to(torch.float64)[:, None] if labels.label_horizon is not None else tau
    obs = torch.floor(((end - tau) / window_seconds).clamp(min=0.0, max=float(horizon_k))).long()
    comp_t, stage_t = readout_targets(window, labels)
    p_all = fo.p_inf.detach().to("cpu", torch.float64).numpy()                   # [B, M, K]
    haz_routes = fo.hazard.detach().to("cpu", torch.float64).numpy()             # [B, M, N, K]
    w_routes = fo.route_weight.detach().to("cpu", torch.float64).numpy()         # [B, M, N]
    comp_r = an.readouts["compromise"].detach().to("cpu", torch.float64)
    stage_r = an.readouts["stage"].detach().to("cpu", torch.float64)
    seen = 0
    b_n, m_n = trig.mask.shape
    v_n = comp_t.shape[-1]
    for b in range(b_n):
        base = dict(meta_of.get(ctxs[b].segment_id, {}))
        family = labels.family[b] if b < len(labels.family) else "unknown"
        novelty = labels.novelty[b] if (split == "zero_shot" and b < len(labels.novelty)) else ""
        for m in range(m_n):
            if not bool(trig.mask[b, m]):
                continue
            seen += 1
            t_abs = float(window.origin[b]) + float(tau[b, m])
            row = base | {"time": t_abs, "family": family, "novelty": novelty, "split": split, "entity": -1}
            # Stage units: every active entity with a known stage label (float64 posterior, D-54).
            for v in range(v_n):
                s_lab = int(stage_t[b, m, v])
                if s_lab >= 0:
                    probs = stage_r[b, m, v].numpy()
                    units.stage_probs.append(probs / probs.sum())
                    units.stage_label.append(s_lab)
                    units.stage_meta.append(row | {"entity": v})
                    units.stage_conf.append(float(probs.max() / probs.sum()))
                    units.stage_correct.append(float(int(np.argmax(probs)) == s_lab))
                c = float(comp_t[b, m, v])
                if math.isfinite(c):
                    units.comp_p.append(float(comp_r[b, m, v]))
                    units.comp_y.append(c)
            if not bool(usable[b, m]):
                continue                                                         # already infiltrated: P_inf is 1
            curve = np.maximum.accumulate(np.clip(p_all[b, m, :horizon_k], 0.0, 1.0))
            units.p_inf.append(curve)
            units.hazard.append(mixture_hazard(curve[None])[0])
            surv = np.cumprod(1.0 - np.clip(haz_routes[b, m, :, :horizon_k], 0.0, 1.0), axis=-1)
            units.ensemble.append(np.maximum.accumulate(1.0 - surv, axis=-1))
            wts = np.clip(w_routes[b, m], 0.0, None)
            units.ensemble_w.append(wts / wts.sum() if wts.sum() > 0 else np.full_like(wts, 1.0 / len(wts)))
            event = not bool(cens[b, m])
            e_step = int(ev[b, m]) if event else 0
            o_steps = int(obs[b, m])
            units.event_step.append(e_step)
            units.observed.append(max(o_steps, e_step) if event else min(o_steps, horizon_k))
            units.meta.append(row)
    return seen


def _outputs(units: _Units, *, split: str, window_seconds: float, seed: int, config: dict[str, Any]) -> ModelOutputs:
    """The ModelOutputs bundle of the accumulated units (evaluation/predictions.py)."""
    out = ModelOutputs(model="nagahana", protocol=f"stage4-{split}", seed=seed, config=config)
    if units.p_inf:
        meta = pd.DataFrame(units.meta)
        p = np.stack(units.p_inf)
        n_routes = max(e.shape[0] for e in units.ensemble)
        ens = np.zeros((len(units.ensemble), n_routes, p.shape[1]))
        wts = np.zeros((len(units.ensemble), n_routes))
        for i, (e, w) in enumerate(zip(units.ensemble, units.ensemble_w, strict=True)):
            ens[i, : e.shape[0]] = e
            wts[i, : w.shape[0]] = w
        out.forecast = ForecastPredictions(p_inf=p, window_seconds=window_seconds, event_step=np.asarray(units.event_step),
                                           observed_steps=np.asarray(units.observed), meta=make_meta(len(p), **_cols(meta)),
                                           hazard=np.stack(units.hazard), ensemble=ens, ensemble_weight=wts)
        k = p.shape[1]
        grid = window_seconds * np.arange(1, k + 1, dtype=np.float64)
        e_step = np.asarray(units.event_step)
        obs = np.asarray(units.observed)
        event_time = np.where(e_step > 0, e_step * window_seconds, np.maximum(obs, 0) * window_seconds)
        out.time_to_event = TimeToEventPredictions(risk=p[:, -1], survival=np.minimum.accumulate(1.0 - p, axis=1),
                                                   time_grid=grid, event_time=event_time, event_observed=e_step > 0,
                                                   meta=make_meta(len(p), **_cols(meta)))
    if units.stage_probs:
        smeta = pd.DataFrame(units.stage_meta)
        out.stage = StagePredictions(probs=np.stack(units.stage_probs), label=np.asarray(units.stage_label),
                                     stage_names=STAGE_NAMES, meta=make_meta(len(units.stage_probs), **_cols(smeta)))
    return out


def _cols(df: pd.DataFrame) -> dict[str, Any]:
    return {c: df[c].to_numpy() for c in df.columns}


def evaluate_split(model: NagaHana, loader: Iterable[tuple[WindowBatch, LabelBatch, list[StreamContext]]],
                   bridge: StreamBridge, *, settings: EvalSettings, split: str, meta_of: dict[str, dict[str, Any]],
                   switches: Any = None, physics: PhysicsTerm | None = None, device: torch.device | None = None
                   ) -> ZeroShotReport:
    """Forecast on one split's windows in stream order; ModelOutputs + metrics per group (module docstring).

    switches: inference-tier ablation switches (`training.ablation.RuntimeSwitches`); None = the design.
    meta_of: segment id -> {"dataset": ..., "network": ...} for the units' metadata.
    """
    require_mode(RunMode.EVALUATE, component="stage-4 evaluation")
    if split not in ("test", "zero_shot"):
        raise InvariantViolation(f"stage 4 evaluates the test or zero-shot split, not {split!r}")
    model.eval()
    gen = make_generator(derive_seed(settings.seed, "stage4", split))
    cfg = model.cfg
    units = _Units()
    n_trig = 0
    notes: list[str] = []
    r_tstct = settings.tstct_passes if switches is None or switches.loop_passes is None else switches.loop_passes
    r_taaft = settings.taaft_passes if switches is None or switches.loop_passes is None else switches.loop_passes
    use_longterm = True if switches is None else switches.longterm_memory
    phys = physics if (switches is None or switches.physics) else None
    with torch.no_grad():
        for w, lab, ctxs in loader:
            if switches is not None:
                w = switches.apply_window(w)
            prep = bridge.prepare(w, lab, ctxs)
            if device is not None and device.type != "cpu":
                from nagahana.training.engine import to_device

                prep = to_device(prep, device)
            _, env = model.perceive(prep.window, sample=False, passes=r_tstct, carry=prep.carry)
            if bool(prep.window.triggers.mask.any()):
                past = bridge.past(prep)
                lt = bridge.longterm_states(prep, env) if use_longterm else None
                an = model.analyse(env, prep.window, passes=r_taaft, descent_steps=settings.descent_steps, longterm=lt,
                                   past=past, physics=phys)
                bridge.record_analysis(prep, an, past)
                fo = model.forecast(an, horizon_k=settings.horizon_k, routes_n=settings.routes_n, generator=gen)
                n_trig += _collect(units, an=an, fo=fo, window=prep.window, labels=prep.labels, ctxs=ctxs, meta_of=meta_of,
                                   split=split, window_seconds=cfg.forecaster.window_seconds, horizon_k=settings.horizon_k)
            bridge.commit(prep, env)
    outputs = _outputs(units, split=split, window_seconds=cfg.forecaster.window_seconds, seed=settings.seed,
                       config={"threshold": settings.threshold, "budgets": settings.__dict__})
    pairs: dict[str, tuple[list[float], list[float]]] = {}
    groups: dict[str, dict[str, float]] = {}
    if outputs.forecast is not None:
        f = outputs.forecast
        y_known, known = f.outcome(f.horizon)
        p_k = f.p_inf[:, -1]
        nov = f.meta["novelty"].astype(str).to_numpy()
        keys = ("known", "novel") if split == "zero_shot" else ("all",)
        for g in keys:
            sel = known & ((nov == g) if g != "all" else np.ones_like(known))
            pairs[g] = (p_k[sel].tolist(), y_known[sel].astype(np.float64).tolist())
            groups[g] = group_metrics(pairs[g][0], pairs[g][1], threshold=settings.threshold)
    else:
        for g in (("known", "novel") if split == "zero_shot" else ("all",)):
            pairs[g], groups[g] = ([], []), {"n": 0.0}
        notes.append("no usable trigger in the split")
    calib = calibration_pairs(outputs, units)
    return ZeroShotReport(groups=groups, pairs=pairs, triggers_seen=n_trig, outputs=outputs, calibration=calib, notes=notes)


def zero_shot_evaluate(model: NagaHana, loader: Iterable[tuple[WindowBatch, LabelBatch, list[StreamContext]]],
                       bridge: StreamBridge, *, settings: EvalSettings, meta_of: dict[str, dict[str, Any]] | None = None,
                       switches: Any = None, physics: PhysicsTerm | None = None) -> ZeroShotReport:
    """`evaluate_split` on the zero-shot split; `loader` comes from `zero_shot_loader` (known and novel separately)."""
    return evaluate_split(model, loader, bridge, settings=settings, split="zero_shot", meta_of=meta_of or {},
                          switches=switches, physics=physics)


def calibration_pairs(outputs: ModelOutputs, units: _Units) -> CalibrationPairs:
    """Resolved pairs per output family: P_inf(K), compromise readouts, stage confidence (module docstring)."""
    pairs: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    if outputs.forecast is not None:
        f = outputs.forecast
        y, known = f.outcome(f.horizon)
        pairs["p_inf"] = (torch.from_numpy(f.p_inf[known, -1].copy()), torch.from_numpy(y[known].astype(np.float64)))
    if units.comp_p:
        pairs["compromise"] = (torch.tensor(units.comp_p, dtype=torch.float64), torch.tensor(units.comp_y, dtype=torch.float64))
    if units.stage_conf:
        pairs["stage"] = (torch.tensor(units.stage_conf, dtype=torch.float64),
                          torch.tensor(units.stage_correct, dtype=torch.float64))
    return CalibrationPairs(pairs=pairs)


def propose_calibration(model: NagaHana, pairs: CalibrationPairs, names: Sequence[str]) -> list[CalibrationProposal]:
    """Proposals of the configured calibrators (training/feedback.py); never applied here (D-21)."""
    out: list[CalibrationProposal] = []
    for cal in build_calibrators(names, model):
        prop = cal.fit(pairs)
        if prop is not None:
            out.append(prop)
    return out


def apply_site_temperature(proposal: CalibrationProposal, command: HumanCommand | None, *, state: TemperatureState,
                           audit: list[HumanCommand] | None = None) -> TemperatureState:
    """Apply a temperature proposal: only under a HumanCommand("apply-calibration") (D-21)."""
    require_command(command, APPLY_CALIBRATION)
    return apply_calibration(proposal, command, state=state, audit=audit)


@dataclass(frozen=True)
class AnalystAlert:
    """An analyst-confirmed alert: the entity (stable key of the site stream), when, and the verdict."""

    entity_key: int
    time: float            # epoch seconds
    malicious: bool


@dataclass(frozen=True)
class SiteCalibrationSettings:
    """Site-adapter training knobs (AS-26, AS-420, AS-598)."""

    precision: str
    min_span_s: float
    min_alerts: int
    allow_short: bool
    alert_weight: float
    stage1: Stage1Options
    stage2: Stage2Options
    optim: OptimConfig

    @classmethod
    def assumed(cls, model: NagaHana, *, precision: str, allow_short: bool, optim: OptimConfig | None = None) -> SiteCalibrationSettings:
        """The values of this build's assumptions (AS-26, AS-420, AS-598)."""
        assume("AS-26", by=__name__)
        assume("AS-420", by=__name__)
        use("AS-598", by=__name__)
        return cls(precision=precision, min_span_s=86_400.0, min_alerts=50, allow_short=allow_short, alert_weight=1.0,
                   stage1=Stage1Options(), stage2=Stage2Options(),
                   optim=optim if optim is not None else OptimConfig(lr=1e-4, warmup_steps=100))


@dataclass
class SiteCalibrationReport:
    """What a site-calibration run did, with its coverage against the AS-26 requirement."""

    steps: int
    span_s: float
    alerts_used: int
    adapter_paths: list[str]
    proposal: CalibrationProposal | None
    losses: list[dict[str, float]]
    notes: list[str] = field(default_factory=list)


class SiteCalibrator:
    """LoRA site adapters + the Verifier temperature, under human commands only (module docstring)."""

    def __init__(self, model: NagaHana, *, settings: SiteCalibrationSettings, physics: PhysicsTerm | None, seed: int,
                 command: HumanCommand | None, total_steps: int | None = None) -> None:
        self.command = require_command(command, UPDATE_SITE_ADAPTER)            # D-21: nothing before this check
        self.model, self.settings, self.physics = model, settings, physics
        t = model.cfg.training
        freeze([model])
        paths = attach_lora(model.tstct.stack, names=LORA_NAMES, rank=t.lora_rank, alpha=t.lora_alpha)
        paths += attach_lora(model.taaft.stack, names=LORA_NAMES, rank=t.lora_rank, alpha=t.lora_alpha)
        self.adapter_paths = paths
        params = lora_parameters(model)
        for p in params:
            p.requires_grad_(True)
        model.tstct.train()
        model.taaft.train()
        named = named_trainable([("tstct", model.tstct), ("taaft", model.taaft)])
        self.stepper = HybridStepper(model, named, settings.optim, total_steps=total_steps or 2000, precision="fp32")
        self.draws = Draws(local=make_generator(derive_seed(seed, "site-local")), shared=make_generator(derive_seed(seed, "site-shared")))
        self.alert_pairs: tuple[list[float], list[float]] = ([], [])

    def _alert_loss(self, prep: PreparedBatch, an_compromise: torch.Tensor, alerts: Sequence[AnalystAlert]) -> tuple[torch.Tensor, int]:
        """BCE of the compromise readout of alerted entities at the first trigger at or after each alert."""
        w = prep.window
        losses: list[torch.Tensor] = []
        for b in range(w.entity_mask.shape[0]):
            keys = prep.entity_keys[b].tolist()
            origin = float(w.origin[b])
            for a in alerts:
                if a.entity_key not in keys:
                    continue
                v = keys.index(a.entity_key)
                for m in range(w.triggers.time.shape[1]):
                    tau = float(w.triggers.time[b, m]) + origin
                    if bool(w.triggers.mask[b, m]) and tau >= a.time and int(w.triggers.entity_latest[b, m, v]) >= 0:
                        p = an_compromise[b, m, v].clamp(1e-6, 1 - 1e-6)      # float64 readout (D-54)
                        y = torch.tensor(1.0 if a.malicious else 0.0, dtype=p.dtype, device=p.device)
                        losses.append(F.binary_cross_entropy(p, y))
                        self.alert_pairs[0].append(float(p.detach()))
                        self.alert_pairs[1].append(float(y))
                        break
        if not losses:
            return an_compromise.sum() * 0.0, 0
        return torch.stack(losses).mean(), len(losses)

    def fit(self, loader: Iterable[tuple[WindowBatch, LabelBatch, list[StreamContext]]], bridge: StreamBridge,
            alerts: Sequence[AnalystAlert], *, max_steps: int) -> SiteCalibrationReport:
        """Train the adapters over the site stream; check coverage (AS-26); propose the temperature."""
        require_mode(RunMode.TRAIN, component="site calibration")
        s = self.settings
        losses: list[dict[str, float]] = []
        t_min, t_max = math.inf, -math.inf
        used = 0
        for w, lab, ctxs in loader:
            if self.stepper.step_count >= max_steps:
                break
            prep = bridge.prepare(w, lab, ctxs)
            real = w.update_mask
            if bool(real.any()):
                t_abs = w.update_time + w.origin[:, None]
                t_min = min(t_min, float(t_abs[real].min()))
                t_max = max(t_max, float(t_abs[real].max()))
            passes = self.model.tstct.sample_passes(self.draws.shared)
            l1, p1, _lat1, env1 = stage1_loss(self.model, prep, options=s.stage1, physics=self.physics, gen=self.draws.local,
                                              passes=passes)
            total = l1
            parts = {"stage1": float(l1.detach())}
            if bool(prep.window.triggers.mask.any()):
                lat, env = frozen_perception(self.model, prep, passes=self.model.cfg.tstct.default_passes)
                negative = "swap" if self.stepper.step_count % 2 == 0 else "shuffle"
                past = bridge.past(prep)
                r, sd = sample_budgets(self.model, self.draws, train=True, fixed_passes=None)
                l2, _p2, an = stage2_step_loss(self.model, prep, lat, env, options=s.stage2, physics=self.physics,
                                               gen=self.draws.local, negative=negative, passes=r, descent_steps=sd,
                                               longterm=bridge.longterm_states(prep, env), past=past, train=True,
                                               malignity=False)
                bridge.record_analysis(prep, an, past)
                la, n_a = self._alert_loss(prep, an.readouts["compromise"], alerts)
                used += n_a
                total = total + l2 + s.alert_weight * la
                parts |= {"stage2": float(l2.detach()), "alerts": float(la.detach())}
            self.stepper.backward(total, accumulation=1)
            self.stepper.step()
            bridge.commit(prep, env1)
            losses.append(parts | {k: float(v.detach()) for k, v in p1.items() if v.numel() == 1 and k.startswith("kl")})
        span = max(0.0, t_max - t_min) if math.isfinite(t_min) else 0.0
        notes: list[str] = []
        short = span < s.min_span_s or used < s.min_alerts
        if short:
            msg = (f"site data below the AS-26 requirement: {span:.0f} s of traffic (need {s.min_span_s:.0f} s), "
                   f"{used} alerts used (need {s.min_alerts})")
            if not s.allow_short:
                raise InvariantViolation(msg + "; pass allow_short=True to acknowledge a pilot run")
            notes.append(msg)
        return SiteCalibrationReport(steps=self.stepper.step_count, span_s=span, alerts_used=used,
                                     adapter_paths=list(self.adapter_paths), proposal=self.proposal(), losses=losses,
                                     notes=notes)

    def proposal(self) -> CalibrationProposal | None:
        """ML temperature of the "compromise" family on the alert pairs (None without both outcomes)."""
        p, y = self.alert_pairs
        pairs = CalibrationPairs(pairs={"compromise": (torch.tensor(p, dtype=torch.float64), torch.tensor(y, dtype=torch.float64))})
        props = propose_calibration(self.model, pairs, ("ml-temperature",)) if p else []
        return props[0] if props else None


__all__ = ["AnalystAlert", "EvalSettings", "LORA_NAMES", "SiteCalibrationReport", "SiteCalibrationSettings", "SiteCalibrator",
           "UPDATE_SITE_ADAPTER", "ZeroShotReport", "apply_site_temperature", "calibration_pairs", "check_split",
           "check_zero_shot", "evaluate_split", "group_metrics", "mixture_hazard", "propose_calibration", "split_loader",
           "zero_shot_evaluate", "zero_shot_loader"]
