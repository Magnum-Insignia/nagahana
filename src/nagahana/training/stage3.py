"""Stage 3: full training of the complete architecture with training-phase human feedback (D-22 step 5).

Purpose (build-spec section 3; architecture section 5)
-------------------------------------------------------
"Full training of the whole architecture, including the Forecaster and Advisor policy/value agents. The
Verifier is trained on human feedback." Per trigger-bearing batch (AS-580), in stream order with the
carry:

    Environment = frozen perception (posterior mean; the perceptors stay frozen, AS-413)
    analysis    = TAAFT(view with long-term memory, R, S, create_graph)                (AS-412, AS-577)
    L_3 = w_r L_readouts(analysis)                                                      (TAAFT, AS-414)
          + L_F(teacher-forced Forecaster on the real future)                          (forecaster/losses.py)
          + w_a L_A(model-based policy improvement on -CVaR dP_inf - kappa cost)        (advisor/losses.py)
          + sum of the "policy" feedback learners' losses                              (training/feedback.py)
    Verifier: the "verifier" feedback learners (built in: PRM on step labels, trust head on resolved
              triggers, calibration head), in a step of their own, only under a HumanCommand (D-21).

Readout supervision (AS-414), at trigger tau_m for entity v:
    compromise target  c_v = 1[t_infil(v) <= tau_m] for internal entities (AS-18), masked for external ones
    stage target       the stage label of the update behind v's latest position <= tau_m (-1 = masked)
    L_readouts = BCE(p_v, c_v) + CE(stage_v, s_v), each a mean over its defined entries, in float64 (D-54).
The infiltration hazard is trained by the Forecaster's survival NLL with censoring.

Coupling (AS-22, STAGED; D-12 held, option in force through governance.decisions): the Forecaster and
Advisor read TAAFT through a stop-gradient for the first `joint_after` optimiser steps (AS-597: half of
the stage's planned steps unless set), then jointly.

Human feedback (training/feedback.py; D-07, D-17, D-21, AS-25): every feedback learner runs only when
the run holds a HumanCommand("update-weights"); without it the learners are skipped and the run records
it, and the Verifier stays frozen. Dataset annotations count as human-supplied truth (AS-25).

Decisions: D-12 (held), D-17, D-21, D-22, D-33, D-51. Assumptions: AS-17, AS-18, AS-22, AS-25, AS-260,
AS-412 ... AS-415, AS-452, AS-580, AS-595, AS-597.
"""

from __future__ import annotations

import contextlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

from nagahana.core.errors import InvariantViolation
from nagahana.governance import decisions
from nagahana.governance.assumptions import assume
from nagahana.models.advisor.losses import policy_improvement_loss
from nagahana.models.batch import AnalysisOut, LabelBatch, WindowBatch
from nagahana.models.forecaster.losses import (
    ForecasterLossWeights,
    ForecasterTargets,
    latent_targets,
    step_labels,
    survival_targets,
    teacher_forced,
)
from nagahana.models.nagahana import NagaHana
from nagahana.models.taaft.structure import gather_rows
from nagahana.physics.term import PhysicsTerm
from nagahana.roles.contracts import HumanCommand
from nagahana.training.carry import PreparedBatch, StreamBridge
from nagahana.training.checkpoint import load_optimizer_state_by_name, optimizer_state_by_name
from nagahana.training.config import OptimConfig, Stage3Options
from nagahana.training.distributed import DistInfo, allreduce_mean_grads
from nagahana.training.engine import Draws, ObjectiveOut, ProgramTrainer, StageProgram, detach_tree
from nagahana.training.feedback import FeedbackBatch, FeedbackLearner, authorise, build_learners
from nagahana.training.optim import HybridStepper, named_trainable
from nagahana.training.qkclip import QKClip
from nagahana.training.stage1 import scalar_parts
from nagahana.training.stage2 import Stage2Pre, frozen_perception, observe_taaft_keys, sample_budgets

TRAINED: tuple[str, ...] = ("taaft", "forecaster", "advisor")
PERCEPTORS: tuple[str, ...] = ("inputs", "cvgae", "decoder", "tstct")


def trigger_outcomes(window: WindowBatch, labels: LabelBatch, *, window_seconds: float, horizon_k: int) -> torch.Tensor:
    """Outcome of each trigger: 1 = an internal entity is first infiltrated within K steps, 0 = all K steps
    observed without one, NaN = not scorable (already infiltrated, or censored early). [B, M] float32."""
    if labels.label_horizon is None:
        raise InvariantViolation("LabelBatch.label_horizon is required to score outcomes (censoring)")
    ev, cens, usable = survival_targets(window.triggers.time, window.triggers.mask, labels.entity_infiltrated_at,
                                        window.entity_internal, labels.label_horizon, window_seconds=window_seconds,
                                        horizon_k=horizon_k)
    y = torch.full(ev.shape, float("nan"), device=ev.device)
    y = torch.where(usable & ~cens, torch.ones_like(y), y)
    y = torch.where(usable & cens & (ev >= horizon_k), torch.zeros_like(y), y)
    return y


def readout_targets(window: WindowBatch, labels: LabelBatch) -> tuple[torch.Tensor, torch.Tensor]:
    """(compromise [B, M, V] float with NaN = masked, stage [B, M, V] long with -1 = masked) (AS-414)."""
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

    Precision (D-54, AS-452): the readouts are float64, so both terms are float64 (the targets are cast
    to the readout's dtype); backward reaches the float32 heads through the exact `.double()` cast.
    """
    assume("AS-452", by=__name__)
    v = comp.shape[-1]
    p = an.readouts["compromise"][..., :v].clamp(1e-6, 1 - 1e-6)               # float64 [B, M, V]
    comp = comp.to(p.dtype)
    ok = torch.isfinite(comp) & an.token_mask[..., :v]
    t = torch.where(ok, comp, torch.zeros_like(comp))
    bce = -(t * torch.log(p) + (1 - t) * torch.log1p(-p))
    l_c = (bce * ok).sum() / ok.sum().clamp_min(1)
    probs = an.readouts["stage"][..., :v, :].clamp_min(1e-8)
    oks = (stage >= 0) & an.token_mask[..., :v]
    ce = -torch.log(torch.gather(probs, -1, stage.clamp_min(0).unsqueeze(-1)).squeeze(-1))
    l_s = (ce * oks).sum() / oks.sum().clamp_min(1)
    return {"compromise": l_c, "stage": l_s, "n_compromise": ok.sum().float(), "n_stage": oks.sum().float()}


def forecaster_weights(options: Stage3Options) -> ForecasterLossWeights:
    """The Forecaster loss weights of the stage options."""
    return ForecasterLossWeights(hazard=options.forecaster_hazard, bc_technique=options.forecaster_bc_technique,
                                 bc_target=options.forecaster_bc_target, stage=options.forecaster_stage,
                                 consistency=options.forecaster_consistency, hypothesis=options.forecaster_hypothesis,
                                 latent=options.forecaster_latent, reward=options.forecaster_reward,
                                 value=options.forecaster_value)


def forecaster_targets(model: NagaHana, window: WindowBatch, labels: LabelBatch, z: torch.Tensor) -> ForecasterTargets:
    """The realised steps, events and censoring of every trigger (forecaster/losses.py; AS-260)."""
    fc = model.cfg.forecaster
    tech, tgt, st = step_labels(window, labels, window_seconds=fc.window_seconds, horizon_k=fc.horizon_k)
    if labels.label_horizon is None:
        raise InvariantViolation("LabelBatch.label_horizon is required (censoring)")
    ev, cens, usable = survival_targets(window.triggers.time, window.triggers.mask, labels.entity_infiltrated_at,
                                        window.entity_internal, labels.label_horizon, window_seconds=fc.window_seconds,
                                        horizon_k=fc.horizon_k)
    lat_t, lat_m = latent_targets(window, z.detach(), tgt, window_seconds=fc.window_seconds)
    return ForecasterTargets(technique=tech, target_entity=tgt, stage=st, event_step=ev, censored=cens, usable=usable,
                             latent=lat_t, latent_mask=lat_m)


@dataclass
class _VerifierStep:
    """The Verifier's own optimiser and QK-Clip (trained only under a HumanCommand)."""

    stepper: HybridStepper
    clip: QKClip | None


class Stage3Program(StageProgram):
    """Stage 3 for the engine (module docstring)."""

    stage = 3
    name = "full-training"
    needs_trigger = True

    def __init__(self, model: NagaHana, *, options: Stage3Options, physics: PhysicsTerm | None, optim: OptimConfig,
                 command: HumanCommand | None, info: DistInfo | None = None, fixed_passes: int | None = None,
                 use_longterm: bool = True) -> None:
        super().__init__(model)
        self.options = options
        self.physics = physics
        self.optim_cfg = optim
        self.info = info
        self.fixed_passes = fixed_passes
        self.use_longterm = use_longterm
        self.command = authorise(command) if command is not None else None
        self.notes: list[str] = []
        learners = build_learners(options.feedback, model, options) if self.command is not None else []
        if options.feedback and self.command is None:
            self.notes.append("feedback learners not run: no HumanCommand('update-weights') was given (D-21)")
        self.policy_learners: list[FeedbackLearner] = [x for x in learners if x.target == "policy"]
        self.verifier_learners: list[FeedbackLearner] = [x for x in learners if x.target == "verifier"]
        self.joint_after = options.joint_after
        self.verifier: _VerifierStep | None = None
        self.coupling = decisions.require("D-12", by=__name__).value          # the option in force (AS-22: staged)

    # ---- roles of the components
    def trained(self) -> list[str]:
        return list(TRAINED)

    def replicated(self) -> list[str]:
        return ["longterm"] if self.use_longterm else []

    def frozen(self) -> list[str]:
        out = list(PERCEPTORS)
        if not self.use_longterm:
            out.append("longterm")
        if not self.verifier_learners:
            out.append("verifier")
        return out

    def on_plan(self, total_steps: int) -> None:
        """The STAGED switch from the planned steps (AS-597) and the Verifier's optimiser."""
        if self.joint_after is None:
            assume("AS-22", by=__name__)
            self.joint_after = int(round(self.options.joint_after_fraction * total_steps))
        if self.verifier_learners and self.verifier is None:
            named = named_trainable([(m_name, getattr(self.model, m_name)) for m_name in ("verifier",)])
            clip = QKClip(self.model.verifier, tau=self.optim_cfg.qk_clip_tau, prefix="verifier") if self.optim_cfg.qk_clip else None
            self.verifier = _VerifierStep(stepper=HybridStepper(self.model, named, self.optim_cfg, total_steps=total_steps,
                                                                precision="fp32", qk_clip=clip, info=self.info), clip=clip)

    def before_step(self, step: int) -> None:
        # D-12 option in force: "staged" switches at joint_after; "joint" reads TAAFT jointly from the start;
        # "separate" keeps the stop-gradient for the whole stage.
        if self.coupling == "joint":
            joint = True
        elif self.coupling == "separate":
            joint = False
        else:
            joint = step >= (self.joint_after or 0)
        self.model.forecaster.joint_phase = joint
        self.model.advisor.joint_phase = joint

    # ---- per batch
    def pre(self, prep: PreparedBatch, draws: Draws, bridge: StreamBridge) -> Stage2Pre:
        p = self.options.perception_passes
        lat, env = frozen_perception(self.model, prep, passes=p if p is not None else self.model.cfg.tstct.default_passes)
        past = bridge.past(prep) if bool(prep.window.triggers.mask.any()) else None
        return Stage2Pre(lat=lat, env=env, past=past, bridge=bridge)

    def compute(self, model: NagaHana, prep: PreparedBatch, pre: Stage2Pre, draws: Draws, step: int, train: bool) -> ObjectiveOut:
        cfg, o = model.cfg, self.options
        w, lab = prep.window, prep.labels
        r, s = sample_budgets(model, draws, train=train, fixed_passes=self.fixed_passes)
        longterm = pre.bridge.longterm_states(prep, pre.env) if self.use_longterm else None
        observe_taaft_keys(self.qk_clip, prep, pre.env, pre.past)
        an = model.analyse(pre.env, w, passes=r, descent_steps=s, longterm=longterm, past=pre.past, physics=self.physics,
                           create_graph=train, generator=draws.local)
        parts: dict[str, torch.Tensor] = {}
        comp, stage = readout_targets(w, lab)
        ro = readout_loss(an, comp, stage)
        parts["readout_compromise"], parts["readout_stage"] = ro["compromise"], ro["stage"]
        # The Forecaster on the real future (teacher forcing; survival NLL with censoring).
        targets = forecaster_targets(model, w, lab, pre.lat.z)
        l_f, fparts = teacher_forced(model.forecaster, an, targets, exposure=model.exposure(), weights=forecaster_weights(o))
        parts.update({f"forecaster_{k}": v for k, v in fparts.items()})
        # The Advisor: policy improvement at the first valid trigger of the batch.
        valid = torch.nonzero(w.triggers.mask)
        b0, m0 = int(valid[0, 0]), int(valid[0, 1])
        l_a, aparts = policy_improvement_loss(model.advisor, an, model.forecaster, b=b0, m=m0, rollouts=cfg.advisor.rollouts,
                                              generator=draws.local, horizon_k=cfg.forecaster.horizon_k,
                                              exposure=model.exposure(), entity_kind=w.entity_kind[b0],
                                              entity_internal=w.entity_internal[b0])
        parts["advisor_policy"], parts["advisor_value"] = aparts["policy"], aparts["value"]
        total = o.readout_weight * (ro["compromise"] + ro["stage"]) + l_f + o.advisor_weight * l_a
        aux: dict[str, Any] = {}
        need_feedback = bool(self.policy_learners or self.verifier_learners) and self.command is not None and train
        if need_feedback:
            with torch.no_grad():
                an_d = detach_tree(an)
                fo = model.forecast(an_d, horizon_k=cfg.forecaster.horizon_k, routes_n=o.verifier_routes,
                                    generator=draws.local)
            y = trigger_outcomes(w, lab, window_seconds=cfg.forecaster.window_seconds, horizon_k=cfg.forecaster.horizon_k)
            assert self.command is not None
            for learner in self.policy_learners:
                fb = FeedbackBatch(analysis=an, forecast=fo, targets=targets, outcomes=y, latents=pre.lat, window=w,
                                   labels=lab, step=step, command=self.command)
                l_fb, logs = learner.loss(fb)
                total = total + l_fb
                parts.update({f"feedback_{learner.name}_{k}": torch.tensor(v) for k, v in logs.items()})
            aux.update({"forecast": fo, "outcomes": y, "targets": detach_tree(targets), "analysis_full": an_d,
                        "latents": detach_tree(pre.lat)})
        parts["total"] = total
        parts["descent_steps"] = torch.tensor(float(s))
        parts["passes"] = torch.tensor(float(r))
        if not torch.isfinite(total):
            raise InvariantViolation(f"stage-3 loss is not finite: {scalar_parts(parts)}")
        aux["analysis"] = detach_tree(an)
        return ObjectiveOut(loss=total, parts=scalar_parts(parts), aux=aux)

    def post(self, bridge: StreamBridge, prep: PreparedBatch, pre: Stage2Pre, out: ObjectiveOut | None) -> None:
        if out is not None:
            bridge.record_analysis(prep, out.aux["analysis"], pre.past)          # Imagination for the next window
            out.aux["window"], out.aux["labels"] = prep.window, prep.labels
        bridge.commit(prep, pre.env)

    def extra_step(self, outs: Sequence[ObjectiveOut], step: int) -> dict[str, float]:
        """The Verifier's step on the micro-batches of this optimiser step, under the run's HumanCommand."""
        if not self.verifier_learners or self.command is None or self.verifier is None:
            return {}
        authorise(self.command)                                                   # D-21: before anything is computed
        logs: dict[str, float] = {}
        n = len(outs)
        for out in outs:
            if "forecast" not in out.aux:
                continue
            batch = FeedbackBatch(analysis=out.aux["analysis_full"], forecast=out.aux["forecast"], targets=out.aux["targets"],
                                  outcomes=out.aux["outcomes"], latents=out.aux["latents"], window=out.aux["window"],
                                  labels=out.aux["labels"], step=step, command=self.command)
            loss: torch.Tensor | None = None
            track = self.verifier.clip.tracking() if self.verifier.clip is not None else contextlib.nullcontext()
            with track:
                for learner in self.verifier_learners:
                    l_v, lg = learner.loss(batch)
                    loss = l_v if loss is None else loss + l_v
                    for k, v in lg.items():
                        logs[f"verifier_{learner.name}_{k}"] = logs.get(f"verifier_{learner.name}_{k}", 0.0) + v / n
            if loss is not None:
                self.verifier.stepper.backward(loss, accumulation=n)
        if self.info is not None:
            allreduce_mean_grads(self.info, list(self.verifier.stepper.params))
        rep = self.verifier.stepper.step()
        return logs | {"verifier_lr": rep.lr, "verifier_grad_norm": rep.grad_norm}

    # ---- state
    def state_dict(self) -> dict[str, Any]:
        return {"joint_after": self.joint_after, "learners": {x.name: x.state_dict() for x in (*self.policy_learners,
                                                                                                 *self.verifier_learners)},
                "notes": list(self.notes)}

    def load_state_dict(self, state: Any) -> None:
        self.joint_after = state.get("joint_after", self.joint_after)
        saved = dict(state.get("learners", {}))
        for x in (*self.policy_learners, *self.verifier_learners):
            if x.name in saved:
                x.load_state_dict(saved[x.name])
        self.notes = list(state.get("notes", self.notes))

    def extra_optimizer_state(self) -> dict[str, Any]:
        if self.verifier is None:
            return {}
        return {"verifier": {"optimizers": {nm: optimizer_state_by_name(opt) for nm, opt in self.verifier.stepper.optimizers()},
                             "counters": self.verifier.stepper.counters()}}

    def load_extra_optimizer_state(self, state: Any) -> None:
        if self.verifier is None or "verifier" not in state:
            return
        for nm, opt in self.verifier.stepper.optimizers():
            load_optimizer_state_by_name(opt, state["verifier"]["optimizers"][nm])
        self.verifier.stepper.load_counters(state["verifier"]["counters"])


class Stage3Trainer(ProgramTrainer):
    """Single-process stage-3 trainer on prepared batches (training/engine.ProgramTrainer)."""

    def __init__(self, model: NagaHana, *, options: Stage3Options, physics: PhysicsTerm | None, optim: OptimConfig,
                 total_steps: int, seed: int, command: HumanCommand | None, fixed_passes: int | None = None,
                 use_longterm: bool = True) -> None:
        program = Stage3Program(model, options=options, physics=physics, optim=optim, command=command,
                                fixed_passes=fixed_passes, use_longterm=use_longterm)
        super().__init__(program, optim=optim, total_steps=total_steps, seed=seed)
        program.on_plan(total_steps)


__all__ = ["PERCEPTORS", "Stage3Program", "Stage3Trainer", "TRAINED", "forecaster_targets", "forecaster_weights",
           "readout_loss", "readout_targets", "trigger_outcomes"]
