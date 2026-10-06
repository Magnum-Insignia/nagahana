"""Stage 6: zero-shot validation (real data only; novel and known separately) and human-gated site calibration.

Purpose (build-spec §3 stage 6; architecture §5.6; D-23, D-13 held → AS-26)
---------------------------------------------------------------------------
1. **Zero-shot evaluation** (`zero_shot_evaluate`, RunMode.EVALUATE): the model forecasts on the
   zero-shot segments of a split manifest — refused unless every segment is REAL and in the zero-shot
   role (D-23: zero-shot never sees Generator variants) — and the outcome of every scorable trigger
   (`stage5.trigger_outcomes`: an internal entity first infiltrated within K steps, censoring respected)
   is scored separately for *known* and *novel* families:

       decision = 𝟙[P_inf(K) ≥ θ]   →  precision, recall, F1, FPR, FNR, detection error, base rate
       Brier = mean (P_inf(K) − y)²,  ECE (10 bins)

   θ is a required setting (AS-419 value ½; a site replaces it with the Verifier's conformal threshold).

2. **Site calibration** (`SiteCalibrator`, AS-26): LoRA adapters (rank, α from `TrainingConfig`) on the
   attention projections (q, k, v, o) of TSTCT and TAAFT, trained on the site's unlabelled traffic
   (stage-3 objective + stage-4 objective without the malignity term) plus analyst-confirmed alerts
   (BCE of the compromise readout of the alerted entity at the first trigger at or after the alert),
   then the Verifier's temperature of the "compromise" family fitted on the alert pairs (maximum
   likelihood, `verifier/calibration.ml_temperature`). Nothing changes without a human:
   - attaching and training adapters requires a HumanCommand("update-site-adapter");
   - applying the temperature requires a HumanCommand("apply-calibration") (`verifier/gate`).
   The data requirement (24 h of traffic, 50 confirmed alerts) is checked and reported; a shorter run
   must say so explicitly (`allow_short`).

Decisions: D-13 (held), D-16 (held), D-21, D-23, D-50 (clock features on only for site calibration: a
config switch the operator sets, not changed here). Assumptions: AS-26, AS-419, AS-420.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from nagahana.core.errors import InvariantViolation
from nagahana.core.modes import RunMode, require_mode
from nagahana.data.sampling import Role, SplitManifest, WindowRecord, label_limits
from nagahana.data.stream import StreamContext, StreamLoader
from nagahana.data.windows import PreparedSource
from nagahana.evaluation.calibration import ece
from nagahana.evaluation.metrics import Confusion, report
from nagahana.governance.assumptions import assume
from nagahana.models.batch import LabelBatch, WindowBatch
from nagahana.models.config import NagaHanaConfig
from nagahana.models.nagahana import NagaHana
from nagahana.models.taaft.objectives import Stage4Weights
from nagahana.models.verifier.calibration import logit, ml_temperature
from nagahana.models.verifier.gate import APPLY_CALIBRATION, apply_calibration, require_command
from nagahana.models.verifier.reports import CalibrationProposal, TemperatureState
from nagahana.nn.lora import attach_lora, lora_parameters
from nagahana.physics.term import PhysicsTerm
from nagahana.pipeline.freezing import freeze
from nagahana.roles.contracts import HumanCommand
from nagahana.training.carry import PreparedBatch, StreamBridge
from nagahana.training.common import Optimiser, generator
from nagahana.training.stage3 import Stage3Settings, stage3_loss
from nagahana.training.stage4 import Stage4Settings, frozen_perception, stage4_step_loss
from nagahana.training.stage5 import trigger_outcomes

UPDATE_SITE_ADAPTER = "update-site-adapter"
LORA_NAMES = ("q_proj", "k_proj", "v_proj", "o_proj")


# ===================================================================================== zero-shot evaluation
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


@dataclass
class ZeroShotReport:
    """Metrics per novelty group ("known", "novel") plus the pooled pairs for later calibration."""

    groups: dict[str, dict[str, float]]
    pairs: dict[str, tuple[list[float], list[float]]]
    triggers_seen: int
    notes: list[str] = field(default_factory=list)


def check_zero_shot(manifest: SplitManifest, segments: Sequence[WindowRecord]) -> None:
    """Refuse anything but REAL zero-shot segments (D-23: zero-shot never sees variants)."""
    for r in segments:
        if r.origin != "real":
            raise InvariantViolation(f"{r.id}: zero-shot evaluation reads real data only (D-23); got origin {r.origin!r}")
        if manifest.role.get(r.id) is not Role.ZERO_SHOT:
            raise InvariantViolation(f"{r.id}: not a zero-shot segment of the manifest")


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


def zero_shot_evaluate(model: NagaHana, loader: Iterable[tuple[WindowBatch, LabelBatch, list[StreamContext]]],
                       bridge: StreamBridge, *, settings: EvalSettings) -> ZeroShotReport:
    """Forecast on zero-shot windows in stream order and score known and novel families separately.

    `loader` must come from `zero_shot_loader` (real zero-shot segments only, novelty marks set).
    """
    require_mode(RunMode.EVALUATE, component="zero-shot evaluation")
    model.eval()
    gen = generator(settings.seed)
    pairs: dict[str, tuple[list[float], list[float]]] = {"known": ([], []), "novel": ([], [])}
    n_trig = 0
    notes: list[str] = []
    cfg = model.cfg
    with torch.no_grad():
        for w, lab, ctxs in loader:
            prep = bridge.prepare(w, lab, ctxs)
            _, env = model.perceive(prep.window, sample=False, passes=settings.tstct_passes, carry=prep.carry)
            if bool(prep.window.triggers.mask.any()):
                past = bridge.past(prep)
                an = model.analyse(env, prep.window, passes=settings.taaft_passes, descent_steps=settings.descent_steps,
                                   longterm=bridge.longterm_states(prep, env), past=past)
                bridge.record_analysis(prep, an, past)
                fo = model.forecast(an, horizon_k=settings.horizon_k, routes_n=settings.routes_n, generator=gen)
                y = trigger_outcomes(prep.window, prep.labels, window_seconds=cfg.forecaster.window_seconds,
                                     horizon_k=settings.horizon_k)
                for b in range(y.shape[0]):
                    group = prep.labels.novelty[b] if b < len(prep.labels.novelty) else ""
                    if group not in pairs:
                        notes.append(f"window without a novelty mark skipped ({ctxs[b].segment_id})")
                        continue
                    for m in range(y.shape[1]):
                        if bool(prep.window.triggers.mask[b, m]):
                            n_trig += 1
                        if math.isfinite(float(y[b, m])):
                            pairs[group][0].append(float(fo.p_inf[b, m, -1]))
                            pairs[group][1].append(float(y[b, m]))
            bridge.commit(prep, env)
    groups = {g: group_metrics(p, yy, threshold=settings.threshold) for g, (p, yy) in pairs.items()}
    return ZeroShotReport(groups=groups, pairs=pairs, triggers_seen=n_trig, notes=notes)


# ===================================================================================== site calibration
@dataclass(frozen=True)
class AnalystAlert:
    """An analyst-confirmed alert: the entity (stable key of the site stream), when, and the verdict."""

    entity_key: int
    time: float            # epoch seconds
    malicious: bool


@dataclass(frozen=True)
class SiteCalibrationSettings:
    """Site-adapter training knobs (AS-26, AS-420)."""

    precision: str
    min_span_s: float              # AS-26: 24 h of unlabelled site traffic
    min_alerts: int                # AS-26: 50 analyst-confirmed alerts
    allow_short: bool              # explicit acknowledgement that a run uses less than the above (tests, pilots)
    alert_weight: float            # AS-420: weight of the alert BCE beside the self-supervised terms
    stage3: Stage3Settings
    stage4: Stage4Settings

    @classmethod
    def assumed(cls, model: NagaHana, *, precision: str, allow_short: bool) -> SiteCalibrationSettings:
        assume("AS-26", by=__name__)
        assume("AS-420", by=__name__)
        s4 = Stage4Settings.assumed(precision=precision, perception_passes=model.cfg.tstct.default_passes)
        w = s4.weights
        s4 = Stage4Settings(precision=precision, perception_passes=s4.perception_passes, hidden_ratio=s4.hidden_ratio,
                            weights=Stage4Weights(masked_entity=w.masked_entity, future_latent=w.future_latent,
                                                  contrastive=w.contrastive, malignity=0.0, energy_reg=w.energy_reg))
        return cls(precision=precision, min_span_s=86_400.0, min_alerts=50, allow_short=allow_short, alert_weight=1.0,
                   stage3=Stage3Settings.assumed(precision=precision), stage4=s4)


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
    """LoRA site adapters + Verifier temperature, under human commands only (module docstring)."""

    def __init__(self, model: NagaHana, *, settings: SiteCalibrationSettings, physics: PhysicsTerm | None, seed: int,
                 command: HumanCommand | None) -> None:
        self.command = require_command(command, UPDATE_SITE_ADAPTER)            # D-21: nothing before this check
        self.model, self.settings, self.physics = model, settings, physics
        t = model.cfg.training
        freeze([model])
        paths = attach_lora(model.tstct.stack, names=LORA_NAMES, rank=t.lora_rank, alpha=t.lora_alpha)
        paths += attach_lora(model.taaft.stack, names=LORA_NAMES, rank=t.lora_rank, alpha=t.lora_alpha)
        self.adapter_paths = paths
        self.params = lora_parameters(model)
        for p in self.params:
            p.requires_grad_(True)
        model.tstct.train()
        model.taaft.train()
        self.optim = Optimiser(self.params, t)
        self.gen = generator(seed)
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
                        y = torch.tensor(1.0 if a.malicious else 0.0, dtype=p.dtype)   # target in p's dtype
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
            if self.optim.step_count >= max_steps:
                break
            prep = bridge.prepare(w, lab, ctxs)
            real = w.update_mask
            if bool(real.any()):
                t_abs = w.update_time + w.origin[:, None]
                t_min = min(t_min, float(t_abs[real].min()))
                t_max = max(t_max, float(t_abs[real].max()))
            l3, p3, _lat3, env3 = stage3_loss(self.model, prep, settings=s.stage3, physics=self.physics, gen=self.gen)
            total = l3
            parts = {"stage3": float(l3.detach())}
            if bool(prep.window.triggers.mask.any()):
                lat, env = frozen_perception(self.model, prep, passes=s.stage4.perception_passes, precision=s.precision)
                negative = "swap" if self.optim.step_count % 2 == 0 else "shuffle"
                past = bridge.past(prep)
                l4, _p4, an = stage4_step_loss(self.model, prep, lat, env, settings=s.stage4, physics=self.physics,
                                               gen=self.gen, negative=negative, longterm=bridge.longterm_states(prep, env),
                                               past=past)
                bridge.record_analysis(prep, an, past)
                la, n_a = self._alert_loss(prep, an.readouts["compromise"], alerts)
                used += n_a
                total = total + l4 + s.alert_weight * la
                parts |= {"stage4": float(l4.detach()), "alerts": float(la.detach())}
            self.optim.step(total)
            bridge.commit(prep, env3)
            losses.append(parts | {k: float(v.detach()) for k, v in p3.items() if v.numel() == 1 and k.startswith("kl")})
        span = max(0.0, t_max - t_min) if math.isfinite(t_min) else 0.0
        notes: list[str] = []
        short = span < s.min_span_s or used < s.min_alerts
        if short:
            msg = (f"site data below the AS-26 requirement: {span:.0f} s of traffic (need {s.min_span_s:.0f} s), "
                   f"{used} alerts used (need {s.min_alerts})")
            if not s.allow_short:
                raise InvariantViolation(msg + "; pass allow_short=True to acknowledge a pilot run")
            notes.append(msg)
        return SiteCalibrationReport(steps=self.optim.step_count, span_s=span, alerts_used=used,
                                     adapter_paths=list(self.adapter_paths), proposal=self.proposal(), losses=losses,
                                     notes=notes)

    def proposal(self) -> CalibrationProposal | None:
        """ML temperature of the "compromise" family on the alert pairs (None without both outcomes)."""
        p, y = self.alert_pairs
        if not p or not (0 < sum(y) < len(y)):
            return None
        vc = self.model.cfg.verifier
        # float64 tensors (D-54): torch.tensor(list) would round the stored probabilities to float32.
        t = ml_temperature(logit(torch.tensor(p, dtype=torch.float64)), torch.tensor(y, dtype=torch.float64),
                           t_min=vc.temperature_min, t_max=vc.temperature_max)
        return CalibrationProposal(temperatures={"compromise": t}, source="ml-fit", n_pairs={"compromise": len(p)},
                                   t_min=vc.temperature_min, t_max=vc.temperature_max)


def apply_site_temperature(proposal: CalibrationProposal, command: HumanCommand | None, *, state: TemperatureState,
                           audit: list[HumanCommand] | None = None) -> TemperatureState:
    """Apply the site's temperature proposal: only under a HumanCommand("apply-calibration") (D-21)."""
    require_command(command, APPLY_CALIBRATION)
    return apply_calibration(proposal, command, state=state, audit=audit)


__all__ = ["AnalystAlert", "EvalSettings", "SiteCalibrationReport", "SiteCalibrationSettings", "SiteCalibrator",
           "UPDATE_SITE_ADAPTER", "ZeroShotReport", "apply_site_temperature", "check_zero_shot", "group_metrics",
           "zero_shot_evaluate", "zero_shot_loader"]
