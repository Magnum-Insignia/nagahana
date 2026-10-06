"""Stage 5: full training — TAAFT's supervised readouts, Forecaster, Advisor; Verifier on human-gated feedback.

Purpose (build-spec §3 stage 5; architecture §5.5)
--------------------------------------------------
"Full training of the whole architecture, including the Forecaster and Advisor policy/value agents.
The Verifier is trained on human feedback." Per trigger-bearing batch, in stream order with the carry:

    Environment = frozen perception (posterior mean; perceptors stay frozen, AS-413)
    analysis    = TAAFT(view with long-term memory, R, S, create_graph)                  (AS-412)
    L₅ = L_readouts(analysis)                                                            (TAAFT, AS-414)
         + L_F(teacher-forced Forecaster on the real future)                             (forecaster/losses.py)
         + L_A(model-based policy improvement on −CVaR ΔP_inf − κ·cost)                  (advisor/losses.py)
    Verifier: PRM on step labels + trust head on resolved pairs (+ calibration head when ≥ min_pairs),
              one separate optimiser step **only under a HumanCommand("update-weights")** (D-21).

Readout supervision (AS-414), at trigger τ_m for entity v:
    compromise target  c_v = 𝟙[t_infil(v) ≤ τ_m] for internal entities (AS-18), masked for external ones
    stage target       the stage label of the update behind v's latest position ≤ τ_m (−1 = masked)
    L_readouts = BCE(p_v, c_v) + CE(stage_v, s_v), each a mean over its defined entries.
The infiltration hazard is trained by the Forecaster's survival NLL with censoring
(`forecaster.losses.hazard_nll`, horizon from `LabelBatch.label_horizon`).

Coupling (AS-22, STAGED; D-12 held): the Forecaster and Advisor read TAAFT through a stop-gradient for
the first `joint_after` optimiser steps, then jointly (`joint_phase = True`).

Verifier step labels (AS-415): imagined step k of route n is plausible (1) when its target equals the
target of the realised step k (`step_labels`: responder of the first malicious update in the step, or
"no target"), implausible (0) when the realised target is known and differs, unknown (−1) otherwise.
Trust-head label: the alert decision P_inf(K) ≥ ½ matched the outcome (infiltration within K steps,
`trigger_outcomes`). Both are dataset annotations, which count as human-supplied truth (AS-25, D-17).

Decisions: D-12 (held), D-17, D-21, D-22, D-33, D-51. Assumptions: AS-17, AS-18, AS-22, AS-25,
AS-260, AS-412 … AS-415.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from nagahana.core.errors import InvariantViolation
from nagahana.data.stream import StreamContext
from nagahana.governance.assumptions import assume
from nagahana.memory.longterm import NeuralMemoryState
from nagahana.models.advisor.losses import policy_improvement_loss
from nagahana.models.batch import AnalysisOut, EnvironmentOut, ForecastOut, LabelBatch, LatentOut, WindowBatch
from nagahana.models.forecaster.losses import (
    NO_TARGET,
    ForecasterLossWeights,
    ForecasterTargets,
    latent_targets,
    step_labels,
    survival_targets,
    teacher_forced,
)
from nagahana.models.nagahana import NagaHana
from nagahana.models.taaft.imagination import PastImagination
from nagahana.models.taaft.objectives import sample_descent_steps
from nagahana.models.taaft.structure import gather_rows
from nagahana.models.verifier.calibration import logit, ml_temperature
from nagahana.models.verifier.gate import train_on_feedback
from nagahana.models.verifier.heads import forecast_features, monitor_features, reliability_features
from nagahana.models.verifier.monitor import Monitor
from nagahana.models.verifier.reports import OUTPUT_FAMILIES
from nagahana.physics.term import PhysicsTerm
from nagahana.pipeline.freezing import freeze
from nagahana.roles.contracts import HumanCommand, OutcomeForecastPair
from nagahana.tracking.logger import NullLogger, RunLogger
from nagahana.training.carry import PreparedBatch, StreamBridge
from nagahana.training.common import Optimiser, autocast, generator, scalars, trainable
from nagahana.training.stage3 import unfreeze
from nagahana.training.stage4 import frozen_perception


@dataclass(frozen=True)
class Stage5Settings:
    """Stage-5 knobs (AS-412 … AS-415)."""

    precision: str
    perception_passes: int                 # AS-412
    joint_after: int                       # AS-22 STAGED: optimiser steps in the stop-gradient phase (AS-413)
    readout_weight: float                  # AS-414: weight of L_readouts
    advisor_weight: float                  # AS-413: weight of L_A
    forecaster_weights: ForecasterLossWeights
    verifier_routes: int                   # AS-415: routes imagined per trigger for the PRM step labels

    @classmethod
    def assumed(cls, *, precision: str, perception_passes: int, joint_after: int) -> Stage5Settings:
        for a in ("AS-412", "AS-413", "AS-414", "AS-415"):
            assume(a, by=__name__)
        return cls(precision=precision, perception_passes=int(perception_passes), joint_after=int(joint_after),
                   readout_weight=1.0, advisor_weight=1.0, forecaster_weights=ForecasterLossWeights(), verifier_routes=4)


# ===================================================================================== targets
def trigger_outcomes(window: WindowBatch, labels: LabelBatch, *, window_seconds: float, horizon_k: int) -> torch.Tensor:
    """Outcome of each trigger: 1 = an internal entity is first infiltrated within K steps, 0 = all K steps
    observed without one, NaN = not scorable (already infiltrated, or censored early). [B, M] float32."""
    if labels.label_horizon is None:
        raise InvariantViolation("LabelBatch.label_horizon is required to score outcomes (censoring)")
    ev, cens, usable = survival_targets(window.triggers.time, window.triggers.mask, labels.entity_infiltrated_at,
                                        window.entity_internal, labels.label_horizon, window_seconds=window_seconds,
                                        horizon_k=horizon_k)
    y = torch.full(ev.shape, float("nan"))
    y = torch.where(usable & ~cens, torch.ones_like(y), y)
    y = torch.where(usable & cens & (ev >= horizon_k), torch.zeros_like(y), y)
    return y


def readout_targets(window: WindowBatch, labels: LabelBatch) -> tuple[torch.Tensor, torch.Tensor]:
    """(compromise [B, M, V] float with NaN = masked, stage [B, M, V] long with −1 = masked) (AS-414)."""
    assume("AS-18", by=__name__)
    trig = window.triggers
    tau = trig.time.to(torch.float64)                                          # [B, M]
    latest = trig.entity_latest                                                # [B, M, V]
    active = (latest >= 0) & window.entity_mask[:, None, :] & trig.mask[..., None]
    infil = labels.entity_infiltrated_at.to(torch.float64)[:, None, :]          # [B, 1, V]
    comp = (infil <= tau[..., None]).float()
    comp = torch.where(active & window.entity_internal[:, None, :], comp, torch.full_like(comp, float("nan")))
    upd = gather_rows(window.positions.update, latest)                          # [B, M, V]
    st = gather_rows(labels.update_stage, upd)
    stage = torch.where(active & (upd >= 0), st, torch.full_like(st, -1))
    return comp, stage


def readout_loss(an: AnalysisOut, comp: torch.Tensor, stage: torch.Tensor) -> dict[str, torch.Tensor]:
    """BCE on the floored compromise readout + CE on the stage readout (AS-414).

    Precision (D-54, AS-452): the readouts are float64, so both terms are float64 (the targets are
    cast to the readout's dtype); backward reaches the float32 heads through the exact `.double()` cast.
    """
    assume("AS-452", by=__name__)
    v = comp.shape[-1]
    p = an.readouts["compromise"][..., :v].clamp(1e-6, 1 - 1e-6)               # float64 [B, M, V]
    comp = comp.to(p.dtype)                                                    # target at the readout's precision
    ok = torch.isfinite(comp) & an.token_mask[..., :v]
    t = torch.where(ok, comp, torch.zeros_like(comp))
    bce = -(t * torch.log(p) + (1 - t) * torch.log1p(-p))
    l_c = (bce * ok).sum() / ok.sum().clamp_min(1)
    probs = an.readouts["stage"][..., :v, :].clamp_min(1e-8)
    oks = (stage >= 0) & an.token_mask[..., :v]
    ce = -torch.log(torch.gather(probs, -1, stage.clamp_min(0).unsqueeze(-1)).squeeze(-1))
    l_s = (ce * oks).sum() / oks.sum().clamp_min(1)
    return {"compromise": l_c, "stage": l_s, "n_compromise": ok.sum().float(), "n_stage": oks.sum().float()}


def prm_step_labels(fo: ForecastOut, target_entity: torch.Tensor) -> torch.Tensor:
    """[B, M, N, K] step labels (AS-415) from imagined targets and realised targets [B, M, K]."""
    imagined = fo.route_actions[..., 1]                                          # [B, M, N, K] (−1 = no target)
    real = target_entity[:, :, None, :].expand_as(imagined)
    known = (real >= 0) | (real == NO_TARGET)
    match = torch.where(real == NO_TARGET, imagined < 0, imagined == real)
    return torch.where(known, match.long(), torch.full_like(imagined, -1))


# ===================================================================================== trainer
@dataclass
class Stage5State:
    """Running state of a stage-5 run: the Verifier's Monitor and resolved pairs per output family."""

    monitor: Monitor
    pairs: dict[str, tuple[list[float], list[float]]] = field(default_factory=lambda: {f: ([], []) for f in OUTPUT_FAMILIES})
    notes: list[str] = field(default_factory=list)


class Stage5Trainer:
    """Stage-5 optimisation (module docstring).

    verifier_command: the HumanCommand("update-weights") that authorises the Verifier's training
    (D-21). None = the Verifier is not trained in this run (recorded in `state.notes`).
    """

    def __init__(self, model: NagaHana, *, settings: Stage5Settings, physics: PhysicsTerm | None, seed: int,
                 verifier_command: HumanCommand | None, logger: RunLogger | None = None) -> None:
        self.model, self.settings, self.physics = model, settings, physics
        self.cfg = model.cfg
        freeze(model.perceptors())                                                   # AS-413
        unfreeze([*model.analyser(), model.forecaster, model.advisor])
        self.params = trainable([*model.analyser(), model.forecaster, model.advisor])
        self.optim = Optimiser(self.params, model.cfg.training)
        self.verifier_command = verifier_command
        unfreeze([model.verifier])
        self.v_params = trainable([model.verifier])
        self.v_optim = Optimiser(self.v_params, model.cfg.training)
        self.gen = generator(seed)
        self.logger: RunLogger = logger if logger is not None else NullLogger()
        self.state = Stage5State(monitor=Monitor(self.cfg.verifier, region_dims={
            "environment": self.cfg.latent_dim, "imagination": self.cfg.taaft.d_hyp}))
        if verifier_command is None:
            self.state.notes.append("Verifier not trained: no HumanCommand('update-weights') was given (D-21)")
        self._set_phase()

    def _set_phase(self) -> None:
        joint = self.optim.step_count >= self.settings.joint_after
        self.model.forecaster.joint_phase = joint
        self.model.advisor.joint_phase = joint

    # ------------------------------------------------------------------ losses
    def main_loss(self, prep: PreparedBatch, lat: LatentOut, env: EnvironmentOut,
                  longterm: NeuralMemoryState | Sequence[NeuralMemoryState] | None, past: PastImagination | None = None
                  ) -> tuple[torch.Tensor, dict[str, torch.Tensor], AnalysisOut, ForecasterTargets]:
        """L₅ (TAAFT readouts + Forecaster + Advisor) on one prepared, trigger-bearing batch."""
        cfg, s = self.cfg, self.settings
        w, lab = prep.window, prep.labels
        r = self.model.tstct.sample_passes(self.gen)
        steps = sample_descent_steps(self.gen)
        with autocast(s.precision):
            an = self.model.analyse(env, w, passes=r, descent_steps=steps, longterm=longterm, past=past,
                                    physics=self.physics, create_graph=True, generator=self.gen)
        parts: dict[str, torch.Tensor] = {}
        comp, stage = readout_targets(w, lab)
        ro = readout_loss(an, comp, stage)
        parts["readout_compromise"], parts["readout_stage"] = ro["compromise"], ro["stage"]
        # ---- Forecaster on the real future (teacher forcing; survival NLL with censoring)
        fc = cfg.forecaster
        tech, tgt, st = step_labels(w, lab, window_seconds=fc.window_seconds, horizon_k=fc.horizon_k)
        assert lab.label_horizon is not None
        ev, cens, usable = survival_targets(w.triggers.time, w.triggers.mask, lab.entity_infiltrated_at, w.entity_internal,
                                            lab.label_horizon, window_seconds=fc.window_seconds, horizon_k=fc.horizon_k)
        lat_t, lat_m = latent_targets(w, lat.z.detach(), tgt, window_seconds=fc.window_seconds)
        targets = ForecasterTargets(technique=tech, target_entity=tgt, stage=st, event_step=ev, censored=cens, usable=usable,
                                    latent=lat_t, latent_mask=lat_m)
        l_f, fparts = teacher_forced(self.model.forecaster, an, targets, exposure=self.model.exposure(),
                                     weights=s.forecaster_weights)
        parts.update({f"forecaster_{k}": v for k, v in fparts.items()})
        # ---- Advisor: policy improvement at the first valid trigger of the batch
        valid = torch.nonzero(w.triggers.mask)
        b0, m0 = int(valid[0, 0]), int(valid[0, 1])
        l_a, aparts = policy_improvement_loss(self.model.advisor, an, self.model.forecaster, b=b0, m=m0,
                                              rollouts=cfg.advisor.rollouts, generator=self.gen, horizon_k=fc.horizon_k,
                                              exposure=self.model.exposure(), entity_kind=w.entity_kind[b0],
                                              entity_internal=w.entity_internal[b0])
        parts["advisor_policy"], parts["advisor_value"] = aparts["policy"], aparts["value"]
        total = s.readout_weight * (ro["compromise"] + ro["stage"]) + l_f + s.advisor_weight * l_a
        parts["total"] = total
        if not torch.isfinite(total):
            raise InvariantViolation(f"stage-5 loss is not finite: { {k: float(v) for k, v in parts.items() if v.numel() == 1} }")
        return total, parts, an, targets

    def verifier_loss(self, an: AnalysisOut, prep: PreparedBatch, targets: ForecasterTargets, lat: LatentOut
                      ) -> tuple[torch.Tensor, dict[str, float]]:
        """PRM on step labels + trust head on resolved triggers (+ calibration head if enough pairs)."""
        cfg = self.cfg
        an_d = _detached(an)
        with torch.no_grad():
            fo = self.model.forecast(an_d, horizon_k=cfg.forecaster.horizon_k, routes_n=self.settings.verifier_routes,
                                     generator=self.gen)
        v = self.model.verifier
        logits = v.prm(an_d, fo)
        labels = prm_step_labels(fo, targets.target_entity)
        l_prm = v.prm.loss(logits, labels)
        # trust head on resolved triggers
        y = trigger_outcomes(prep.window, prep.labels, window_seconds=cfg.forecaster.window_seconds,
                             horizon_k=cfg.forecaster.horizon_k)
        ff = forecast_features(fo, torch.sigmoid(logits.detach()))
        mf = monitor_features(self.state.monitor.report(), cusum_h=cfg.verifier.cusum_h, ph_lambda=cfg.verifier.ph_lambda)
        trust_logit = v.trust(ff, mf)                                                   # [B, M]
        ok = torch.isfinite(y)
        p_k = fo.p_inf[..., -1]
        correct = ((p_k >= 0.5).float() == y).float()
        l_trust = (F.binary_cross_entropy_with_logits(trust_logit[ok], correct[ok]) if bool(ok.any())
                   else trust_logit.sum() * 0.0)
        # ledger: resolved pairs (P_inf family) and Monitor statistics (latents of both regions)
        for pv, yv in zip(p_k[ok].tolist(), y[ok].tolist(), strict=True):
            self.state.pairs["p_inf"][0].append(float(min(max(pv, 0.0), 1.0)))
            self.state.pairs["p_inf"][1].append(float(yv))
            self.state.monitor.record_resolution(OutcomeForecastPair("train", float(min(max(pv, 0.0), 1.0)), bool(yv > 0.5)))
        real = prep.window.positions.mask
        self.state.monitor.observe_latents("environment", lat.z.detach()[real])
        self.state.monitor.observe_latents("imagination", an_d.y[an_d.token_mask])
        # calibration head: only with enough resolved pairs (VerifierConfig.min_pairs)
        l_cal = l_prm * 0.0
        pv_all, yv_all = self.state.pairs["p_inf"]
        n_pairs = len(pv_all)
        if n_pairs >= cfg.verifier.min_pairs and 0 < sum(yv_all) < n_pairs:
            # float64 (D-54): torch.tensor(list) would round the Python-double probabilities to float32.
            p_t, y_t = torch.tensor(pv_all, dtype=torch.float64), torch.tensor(yv_all, dtype=torch.float64)
            t_ml = ml_temperature(logit(p_t), y_t, t_min=cfg.verifier.temperature_min, t_max=cfg.verifier.temperature_max)
            feats = reliability_features(p_t, y_t, bins=cfg.verifier.reliability_bins)[None]
            log_t = v.calibration(feats, torch.tensor([OUTPUT_FAMILIES.index("p_inf")]))
            l_cal = v.calibration.loss(log_t, torch.tensor([t_ml]))
        total = l_prm + l_trust + l_cal
        return total, {"prm": float(l_prm.detach()), "trust": float(l_trust.detach()), "calibration": float(l_cal.detach()),
                       "resolved_pairs": float(n_pairs), "prm_labelled": float((labels >= 0).sum())}

    # ------------------------------------------------------------------ steps
    def step(self, prep: PreparedBatch, bridge: StreamBridge) -> tuple[dict[str, float] | None, EnvironmentOut]:
        """One step (None when the batch has no trigger). Returns (scalars or None, the Environment for the carry)."""
        lat, env = frozen_perception(self.model, prep, passes=self.settings.perception_passes,
                                     precision=self.settings.precision)
        if not bool(prep.window.triggers.mask.any()):
            return None, env
        self._set_phase()
        past = bridge.past(prep)
        loss, parts, an, targets = self.main_loss(prep, lat, env, bridge.longterm_states(prep, env), past)
        info = self.optim.step(loss)
        bridge.record_analysis(prep, an, past)                                    # Imagination for the next window
        out = scalars({k: v for k, v in parts.items() if v.numel() == 1}, "stage5") | {f"stage5/{k}": v for k, v in info.items()}
        out["stage5/joint_phase"] = float(self.model.forecaster.joint_phase)
        if self.verifier_command is not None:
            holder: dict[str, float] = {}

            def loss_fn() -> torch.Tensor:
                l_v, vparts = self.verifier_loss(an, prep, targets, lat)
                holder.update(vparts)
                return l_v

            train_on_feedback(self.model.verifier, self.v_optim.opt, loss_fn, self.verifier_command)
            out |= {f"stage5/verifier_{k}": v for k, v in holder.items()}
        self.logger.log_metrics(out, step=self.optim.step_count)
        return out, env

    def run(self, loader: Iterable[tuple[WindowBatch, LabelBatch, list[StreamContext]]], bridge: StreamBridge, *,
            max_steps: int) -> list[dict[str, float]]:
        """Stream-order training: every batch keeps the carry; trigger-bearing batches take an update (≤ max_steps)."""
        history: list[dict[str, float]] = []
        for w, lab, ctxs in loader:
            if self.optim.step_count >= max_steps:
                break
            prep = bridge.prepare(w, lab, ctxs)
            out, env = self.step(prep, bridge)
            bridge.commit(prep, env)
            if out is not None:
                history.append(out)
        return history


def _detached(an: AnalysisOut) -> AnalysisOut:
    """A copy of an analysis with every tensor detached (the Verifier never trains the models it checks)."""
    return AnalysisOut(
        context=an.context.detach(), token_mask=an.token_mask, imagination_kv=[], y0=an.y0.detach(), y=an.y.detach(),
        energy_trace=an.energy_trace.detach(), lens_energy={k: v.detach() for k, v in an.lens_energy.items()},
        lens_share={k: v.detach() for k, v in an.lens_share.items()},
        readouts={k: v.detach() for k, v in an.readouts.items()}, passes=an.passes, descent_steps=an.descent_steps,
    )


__all__ = ["Stage5Settings", "Stage5State", "Stage5Trainer", "prm_step_labels", "readout_loss", "readout_targets",
           "trigger_outcomes"]
