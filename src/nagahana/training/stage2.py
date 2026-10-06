"""Stage 2: TAAFT pretraining with the stage-1 modules frozen (self-supervised; D-22 step 4).

Purpose
-------
"Self-supervised pretraining of TAAFT with those frozen: the intuition of adversaries, conflict,
cooperation, coordination, strategies, intentions, malignity and benignity" (architecture section 5).
The FieldEncoder, CVG-AE, Decoder and TSTCT are frozen and verified unchanged (pipeline/freezing.py,
AS-593); TAAFT and the long-term memory it reads (AS-220) train.

Per batch (stream order, D-51; the carry is written from the frozen Environment)
---------------------------------------------------------------------------------
    Environment  = TSTCT(CVG-AE posterior mean)            (no gradient; the inference reading, AS-05)
    hidden       ~ Bernoulli(rho) per seen entity           (masked-entity objective, AS-216, AS-411)
    S ~ U{2..8}, R = 1 + Poisson(3) clipped                 (AS-16, AS-07 applied to TAAFT, AS-412; shared draws, AS-577)
    out      = TAAFT(view, S, R, hidden, create_graph)     (unrolled descent: the loss reaches alpha and the lenses)
    out_neg  = TAAFT(corrupted view, S, R)                 (entity_swap / time_shuffle alternately, AS-215, AS-411)
    L_2 = w_me L_masked-entity + w_fl L_future-latent + w_c [softplus(E+ - E-) + lambda_reg (E+^2 + E-^2)] + w_m L_malignity

`models/taaft/objectives.stage4_loss` computes the terms (its name follows D-22's step numbering); this
module supplies the views, targets and weights (AS-411). Batches without a trigger carry no TAAFT
signal: they run the frozen perception only, to keep the carry (D-51: triggers fire wherever they fall;
AS-580).

TAAFT reads the long-term memory with one state per trigger (each holding only earlier triggers'
writes, AS-222) and the lane's Imagination of the previous window (AS-223), recorded after each step.
QK-Clip (AS-581) observes the keys TAAFT reads without projecting them: the Environment keys of the
window and of the carry, the long-term memory keys, and the carried Imagination keys.

Ablation switches (training/ablation.py, AS-579): `fixed_passes` fixes R; `use_longterm` False reads no
long-term memory; `physics` None drops the physics term of E_total.

Decisions: D-22, D-42, D-51. Assumptions: AS-05, AS-07, AS-16, AS-215, AS-216, AS-220, AS-222, AS-223,
AS-411, AS-412, AS-577, AS-580, AS-581.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from nagahana.core.errors import InvariantViolation
from nagahana.governance.assumptions import assume
from nagahana.memory.longterm import NeuralMemoryState
from nagahana.models.batch import AnalysisOut, EnvironmentOut, LatentOut
from nagahana.models.nagahana import NagaHana
from nagahana.models.taaft.imagination import PastImagination
from nagahana.models.taaft.objectives import (
    Stage4Targets,
    Stage4Weights,
    entity_swap,
    sample_descent_steps,
    sample_hidden_entities,
    stage4_loss,
    time_shuffle,
)
from nagahana.physics.term import PhysicsTerm
from nagahana.training.carry import PreparedBatch, StreamBridge
from nagahana.training.config import Stage2Options
from nagahana.training.engine import Draws, ObjectiveOut, ProgramTrainer, StageProgram, detach_tree
from nagahana.training.stage1 import scalar_parts

TRAINED: tuple[str, ...] = ("taaft",)
REPLICATED: tuple[str, ...] = ("longterm",)
FROZEN: tuple[str, ...] = ("inputs", "cvgae", "decoder", "tstct", "forecaster", "advisor", "verifier")
#: QK-Clip site patterns of TAAFT's cross-attention and self-attention (training/qkclip.py).
TAAFT_CROSS = "taaft.stack.blocks.*.xattn"
TAAFT_SELF = "taaft.stack.blocks.*.attn"


def weights_of(options: Stage2Options, *, malignity: bool = True) -> Stage4Weights:
    """The stage-2 weights (AS-411); `malignity` False drops the labelled malignity term (site calibration, AS-420)."""
    assume("AS-411", by=__name__)
    return Stage4Weights(masked_entity=options.masked_entity, future_latent=options.future_latent,
                         contrastive=options.contrastive, malignity=options.malignity if malignity else 0.0,
                         energy_reg=options.energy_reg)


def frozen_perception(model: NagaHana, prep: PreparedBatch, *, passes: int) -> tuple[LatentOut, EnvironmentOut]:
    """The frozen perceptors' reading of a prepared batch (posterior mean, no gradient)."""
    with torch.no_grad():
        return model.perceive(prep.window, sample=False, passes=passes, carry=prep.carry)


def sample_budgets(model: NagaHana, draws: Draws, *, train: bool, fixed_passes: int | None) -> tuple[int, int]:
    """(R, S) of TAAFT: shared draws in training (AS-412, AS-577), the run-time defaults otherwise."""
    assume("AS-412", by=__name__)
    if train:
        r = model.tstct.sample_passes(draws.shared)
        s = sample_descent_steps(draws.shared)                                # S ~ U{2, ..., 8} (AS-16)
    else:
        r, s = model.cfg.taaft.default_passes, model.cfg.taaft.descent_steps
    return (fixed_passes if fixed_passes is not None else r), s


def observe_taaft_keys(clip: Any, prep: PreparedBatch, env: EnvironmentOut, past: PastImagination | None) -> None:
    """Report to QK-Clip the keys TAAFT reads but does not project (module docstring)."""
    if clip is None:
        return
    for k, _v in env.kv:                                                       # [B, H, P, d_h]
        clip.observe_keys(TAAFT_CROSS, k, head_axis=1)
    if prep.carry is not None:
        for k in prep.carry.k:                                                 # [B, H, C, d_h]
            clip.observe_keys(TAAFT_CROSS, k, head_axis=1)
    if past is not None:
        for k, _v in past.kv:                                                  # [B, H, N, keep, d_h]
            clip.observe_keys(TAAFT_SELF, k, head_axis=1)


@dataclass
class Stage2Pre:
    """The frozen reading of a batch and the carried state TAAFT reads."""

    lat: LatentOut
    env: EnvironmentOut
    past: PastImagination | None
    bridge: StreamBridge


def stage2_step_loss(model: NagaHana, prep: PreparedBatch, lat: LatentOut, env: EnvironmentOut, *, options: Stage2Options,
                     physics: PhysicsTerm | None, gen: torch.Generator, negative: str, passes: int, descent_steps: int,
                     longterm: NeuralMemoryState | list[NeuralMemoryState] | None, past: PastImagination | None,
                     train: bool, malignity: bool = True) -> tuple[torch.Tensor, dict[str, torch.Tensor], AnalysisOut]:
    """L_2 on one prepared batch whose frozen Environment is (lat, env). negative: "swap" | "shuffle"."""
    w, lab = prep.window, prep.labels
    hidden = sample_hidden_entities(w, options.hidden_ratio, gen)
    out = model.analyse(env, w, passes=passes, descent_steps=descent_steps, longterm=longterm, past=past, physics=physics,
                        create_graph=train, generator=gen, hidden_entities=hidden)
    if negative == "swap":
        neg_w = entity_swap(w, gen)
    elif negative == "shuffle":
        neg_w = time_shuffle(w, gen)
    else:
        raise InvariantViolation(f"unknown negative view {negative!r}")
    out_neg = model.analyse(env, neg_w, passes=passes, descent_steps=descent_steps, longterm=longterm, past=past,
                            physics=physics, create_graph=train, generator=gen)
    targets = Stage4Targets(latent_mean=lat.mean.detach(), latent_logits=lat.logits.detach(), hidden=hidden,
                            malicious_share=lab.entity_malicious_share)
    parts = stage4_loss(out, w, targets, weights_of(options, malignity=malignity), out_neg=out_neg)
    parts["descent_steps"] = torch.tensor(float(descent_steps))
    parts["passes"] = torch.tensor(float(passes))
    total = parts["total"]
    if not torch.isfinite(total):
        raise InvariantViolation(f"stage-2 loss is not finite: { {k: float(v) for k, v in parts.items() if v.numel() == 1} }")
    return total, parts, out


class Stage2Program(StageProgram):
    """Stage 2 for the engine: TAAFT and the long-term memory train on trigger-bearing batches."""

    stage = 2
    name = "taaft-pretraining"
    needs_trigger = True

    def __init__(self, model: NagaHana, *, options: Stage2Options, physics: PhysicsTerm | None,
                 fixed_passes: int | None = None, use_longterm: bool = True) -> None:
        super().__init__(model)
        self.options = options
        self.physics = physics
        self.fixed_passes = fixed_passes
        self.use_longterm = use_longterm

    def trained(self) -> list[str]:
        return list(TRAINED)

    def replicated(self) -> list[str]:
        return list(REPLICATED) if self.use_longterm else []

    def frozen(self) -> list[str]:
        return list(FROZEN) + ([] if self.use_longterm else list(REPLICATED))

    def attach_qk_clip(self, clip: Any) -> None:
        super().attach_qk_clip(clip)
        if clip is not None and getattr(self.model.taaft, "has_memory", False):
            clip.add_key_source(self.model.taaft.memory_k_norm, TAAFT_CROSS, head_axis=1)

    def perception_passes(self) -> int:
        p = self.options.perception_passes
        return p if p is not None else self.model.cfg.tstct.default_passes

    def pre(self, prep: PreparedBatch, draws: Draws, bridge: StreamBridge) -> Stage2Pre:
        lat, env = frozen_perception(self.model, prep, passes=self.perception_passes())
        past = bridge.past(prep) if bool(prep.window.triggers.mask.any()) else None
        return Stage2Pre(lat=lat, env=env, past=past, bridge=bridge)

    def compute(self, model: NagaHana, prep: PreparedBatch, pre: Stage2Pre, draws: Draws, step: int, train: bool) -> ObjectiveOut:
        r, s = sample_budgets(model, draws, train=train, fixed_passes=self.fixed_passes)
        # AS-411: the negative view alternates by optimiser step; validation always uses entity_swap, so
        # every validation pass scores the same objective.
        negative = ("swap" if step % 2 == 0 else "shuffle") if train else "swap"
        longterm = pre.bridge.longterm_states(prep, pre.env) if self.use_longterm else None
        observe_taaft_keys(self.qk_clip, prep, pre.env, pre.past)
        total, parts, an = stage2_step_loss(model, prep, pre.lat, pre.env, options=self.options, physics=self.physics,
                                            gen=draws.local, negative=negative, passes=r, descent_steps=s, longterm=longterm,
                                            past=pre.past, train=train)
        out = scalar_parts(parts)
        return ObjectiveOut(loss=total, parts=out, aux={"analysis": detach_tree(an)})

    def post(self, bridge: StreamBridge, prep: PreparedBatch, pre: Stage2Pre, out: ObjectiveOut | None) -> None:
        if out is not None:
            bridge.record_analysis(prep, out.aux["analysis"], pre.past)          # Imagination for the next window
        bridge.commit(prep, pre.env)



class Stage2Trainer(ProgramTrainer):
    """Single-process stage-2 trainer on prepared batches (training/engine.ProgramTrainer)."""

    def __init__(self, model: NagaHana, *, options: Stage2Options, physics: PhysicsTerm | None, optim: Any, total_steps: int,
                 seed: int, fixed_passes: int | None = None, use_longterm: bool = True) -> None:
        super().__init__(Stage2Program(model, options=options, physics=physics, fixed_passes=fixed_passes,
                                       use_longterm=use_longterm), optim=optim, total_steps=total_steps, seed=seed)


__all__ = ["FROZEN", "REPLICATED", "Stage2Pre", "Stage2Program", "Stage2Trainer", "TRAINED", "frozen_perception", "observe_taaft_keys",
           "sample_budgets", "stage2_step_loss", "weights_of"]
