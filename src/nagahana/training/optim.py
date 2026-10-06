"""The hybrid optimiser of a training stage: Muon + AdamW, warmup-stable-decay, clipping, QK-Clip (D-61).

Recipe
------
Parameters are split by the explicit rule of `training/muon.py` (AS-599): Muon (torch.optim.Muon's
update; Nesterov, momentum 0.95, 5 Newton-Schulz steps, update scale 0.2 sqrt(max(A, B)); Liu et al.,
"Muon is Scalable for LLM Training", arXiv:2502.16982) on the 2-D hidden weight matrices, AdamW
(Loshchilov and Hutter, ICLR 2019, arXiv:1711.05101; betas (0.9, 0.95), AS-571; weight decay on matrices
only, AS-406) on everything else. With Moonlight's update scale both share one learning rate and one
decoupled weight decay.

Learning rate (AS-570): lr(i) = lr * m(i) for the 0-based optimiser step i, with W warm-up steps, T
planned steps, decay start D (WSD: D = T - round(decay_fraction * T), at least W), floor r = min_lr_ratio:

    warm-up      m(i) = (i + 1) / W                                        i < W
    wsd          m(i) = 1                                                  W <= i < D
                 m(i) = r + (1 - r) * f(p),  p = (i - D) / (T - D)         D <= i  (p clipped to [0, 1])
                 f = 1 - sqrt(p) ("1-sqrt", Haegele et al., arXiv:2405.18392), 1 - p, or (1 + cos(pi p)) / 2
    cosine       m(i) = r + (1 - r) (1 + cos(pi p)) / 2,  p = (i - W) / (T - W)   (Loshchilov and Hutter, arXiv:1608.03983)
    linear       m(i) = r + (1 - r) (1 - p)
    constant     m(i) = 1

Boundary values (tested): m(W - 1) = 1, m(D - 1) = 1, m(D) = 1, m(T) = r. A decayed branch (`decay_start`)
starts the decay at a chosen step of a run in its stable phase while the main run continues
(Haegele et al. section 3: one stable run, many cooldowns).

Each optimiser step (AS-572): every micro-batch backpropagates loss / n; then (fp16 only) the loss scale
is removed; the global gradient norm over all trained parameters (after the distributed reduction) is
clipped to `grad_clip`; a non-finite norm skips the update on every rank (the norm is the same
everywhere) and more than `max_nonfinite_skips` consecutive skips raise; otherwise both optimisers step
with the scheduled learning rate, gradients are cleared and QK-Clip (training/qkclip.py) bounds the
attention logits. fp16 uses a dynamic loss scaler (Micikevicius et al., ICLR 2018, arXiv:1710.03740);
bf16 and fp32 need none (AS-587).

Expected effect: matrix optimisers were the fastest family in the most careful comparison so far (Wen,
Hall, Ma and Liang, arXiv:2509.02046), about 1.1x over a well-tuned AdamW at 1.2 B parameters;
Moonlight reports about 2x in compute-optimal fits. NagaHana has not measured its own speed-up.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from nagahana.core.errors import InvariantViolation
from nagahana.training.assumptions import use
from nagahana.training.config import OptimConfig
from nagahana.training.muon import Assignment, MatrixMuon, assign_groups
from nagahana.training.qkclip import QKClip


def decay_start(total_steps: int, *, warmup_steps: int, decay_fraction: float) -> int:
    """D of the WSD schedule: T - round(decay_fraction * T), never before the end of warm-up."""
    return max(warmup_steps, total_steps - int(round(decay_fraction * total_steps)))


def lr_multiplier(step: int, *, schedule: str, warmup_steps: int, total_steps: int, min_lr_ratio: float,
                  decay_fraction: float = 0.2, decay_shape: str = "1-sqrt", decay_from: int | None = None) -> float:
    """m(i) of the module docstring. `decay_from` overrides the WSD decay start (a decayed branch)."""
    if step < 0:
        raise ValueError("step must be >= 0")
    if warmup_steps > 0 and step < warmup_steps:
        return (step + 1) / warmup_steps
    r = min_lr_ratio
    if schedule == "constant":
        return 1.0
    if schedule == "wsd":
        d0 = decay_from if decay_from is not None else decay_start(total_steps, warmup_steps=warmup_steps,
                                                                   decay_fraction=decay_fraction)
        if step < d0:
            return 1.0
        span = max(1, total_steps - d0)
        p = min(1.0, max(0.0, (step - d0) / span))
        if decay_shape == "1-sqrt":
            f = 1.0 - math.sqrt(p)
        elif decay_shape == "linear":
            f = 1.0 - p
        elif decay_shape == "cosine":
            f = 0.5 * (1.0 + math.cos(math.pi * p))
        else:
            raise ValueError(f"unknown decay shape {decay_shape!r}")
        return r + (1.0 - r) * f
    span = max(1, total_steps - warmup_steps)
    p = min(1.0, max(0.0, (step - warmup_steps) / span))
    if schedule == "cosine":
        return r + (1.0 - r) * 0.5 * (1.0 + math.cos(math.pi * p))
    if schedule == "linear":
        return r + (1.0 - r) * (1.0 - p)
    raise ValueError(f"unknown schedule {schedule!r}")


def decay_groups(named: Sequence[tuple[str, nn.Parameter]], weight_decay: float) -> list[dict[str, Any]]:
    """AdamW groups: weight decay on matrices only (AS-406). Each group keeps its parameter names."""
    decay = [(n, p) for n, p in named if p.dim() >= 2]
    no_decay = [(n, p) for n, p in named if p.dim() < 2]
    groups: list[dict[str, Any]] = []
    if decay:
        groups.append({"params": [p for _, p in decay], "names": [n for n, _ in decay], "weight_decay": weight_decay})
    if no_decay:
        groups.append({"params": [p for _, p in no_decay], "names": [n for n, _ in no_decay], "weight_decay": 0.0})
    return groups


def build_adamw(named: Sequence[tuple[str, nn.Parameter]], cfg: OptimConfig) -> torch.optim.AdamW:
    """The AdamW member over its parameters (AS-406, AS-571)."""
    use("AS-571", by=__name__)
    kwargs: dict[str, Any] = {"lr": cfg.lr, "betas": tuple(cfg.betas), "eps": cfg.eps}
    if cfg.adamw_implementation == "foreach":
        kwargs["foreach"] = True
    elif cfg.adamw_implementation == "fused":
        kwargs["fused"] = True
    elif cfg.adamw_implementation == "loop":
        kwargs["foreach"] = False
    return torch.optim.AdamW(decay_groups(named, cfg.weight_decay), **kwargs)


def build_muon(named: Sequence[tuple[str, nn.Parameter]], cfg: OptimConfig) -> MatrixMuon:
    """The Muon member over the hidden matrices (D-61)."""
    ns = torch.bfloat16 if cfg.muon_ns_dtype == "bf16" else torch.float32
    return MatrixMuon([{"params": [p for _, p in named], "names": [n for n, _ in named]}], lr=cfg.lr,
                      weight_decay=cfg.weight_decay, momentum=cfg.muon_momentum, nesterov=cfg.muon_nesterov,
                      ns_steps=cfg.muon_ns_steps, adjust_lr_fn=cfg.muon_adjust, ns_dtype=ns)


def named_trainable(modules: Sequence[tuple[str, nn.Module]]) -> list[tuple[str, nn.Parameter]]:
    """(prefix.name, parameter) of every trainable parameter of the given named modules, deduplicated."""
    seen: set[int] = set()
    out: list[tuple[str, nn.Parameter]] = []
    for prefix, m in modules:
        for n, p in m.named_parameters():
            if p.requires_grad and id(p) not in seen:
                seen.add(id(p))
                out.append((f"{prefix}.{n}" if prefix else n, p))
    return out


@dataclass
class StepReport:
    """What one optimiser step did."""

    lr: float
    grad_norm: float
    skipped: bool
    loss_scale: float | None
    qk_clip: dict[str, float]


def _scalar(x: torch.Tensor) -> float:
    """A Python float of a (possibly distributed) scalar tensor."""
    full = getattr(x, "full_tensor", None)
    if callable(full):
        x = full()
    return float(x.detach().to("cpu"))


class HybridStepper:
    """Muon + AdamW with the stage schedule, accumulation, clipping, synchronised skips and QK-Clip.

    root: the module that owns the trained parameters (the stage objective or the model);
    named: (name, parameter) of the trained parameters, names as the model names them (prefix the
    names of `root` with `prefix` when `root` is a sub-module). total_steps: T. decay_from: WSD decay
    start of a decayed branch (None = from the schedule). qk_clip: the stage's QK-Clip (None = off).
    info: the distributed context (QK-Clip reduces its maxima over ranks).
    """

    def __init__(self, root: nn.Module, named: Sequence[tuple[str, nn.Parameter]], cfg: OptimConfig, *, total_steps: int,
                 precision: str, device_type: str = "cpu", prefix: str = "", qk_clip: QKClip | None = None,
                 info: Any = None, decay_from: int | None = None) -> None:
        use("AS-570", by=__name__)
        use("AS-572", by=__name__)
        if not named:
            raise InvariantViolation("no trainable parameters for this stage")
        if any(not p.requires_grad for _, p in named):
            raise InvariantViolation("a frozen parameter was handed to the optimiser")
        names = [n for n, _ in named]
        if len(set(names)) != len(names) or len({id(p) for _, p in named}) != len(named):
            raise InvariantViolation("duplicate parameters handed to the optimiser")
        self.cfg = cfg
        self.named = list(named)
        self.params = [p for _, p in self.named]
        self.assignment: Assignment = assign_groups(root, self.named, prefix=prefix)
        muon_named, adamw_named = self.assignment.muon(), self.assignment.adamw()
        self.muon = build_muon(muon_named, cfg) if muon_named else None
        self.adamw = build_adamw(adamw_named, cfg) if adamw_named else None
        self.total_steps = int(max(1, total_steps))
        self.decay_from = decay_from
        self.precision = precision
        self.scaler = torch.amp.GradScaler(device_type, enabled=precision == "fp16")
        self.qk_clip = qk_clip if cfg.qk_clip else None
        self.info = info
        self.step_count = 0
        self.skipped_total = 0
        self.consecutive_skips = 0

    def optimizers(self) -> list[tuple[str, torch.optim.Optimizer]]:
        """The member optimisers by name ("muon", "adamw")."""
        out: list[tuple[str, torch.optim.Optimizer]] = []
        if self.muon is not None:
            out.append(("muon", self.muon))
        if self.adamw is not None:
            out.append(("adamw", self.adamw))
        return out

    def lr_at(self, step: int) -> float:
        """The learning rate of optimiser step `step`."""
        c = self.cfg
        return c.lr * lr_multiplier(step, schedule=c.schedule, warmup_steps=c.warmup_steps, total_steps=self.total_steps,
                                    min_lr_ratio=c.min_lr_ratio, decay_fraction=c.decay_fraction, decay_shape=c.decay_shape,
                                    decay_from=self.decay_from)

    def backward(self, loss: torch.Tensor, *, accumulation: int) -> None:
        """Backpropagate one micro-batch's share loss / accumulation (scaled under fp16)."""
        if accumulation < 1:
            raise ValueError("accumulation must be >= 1")
        share = loss / accumulation
        if self.scaler.is_enabled():
            self.scaler.scale(share).backward()
        else:
            share.backward()

    def zero_grad(self) -> None:
        for _, opt in self.optimizers():
            opt.zero_grad(set_to_none=True)

    def step(self) -> StepReport:
        """Unscale, clip, check, update, clear, QK-Clip (module docstring)."""
        if self.scaler.is_enabled():
            for _, opt in self.optimizers():
                self.scaler.unscale_(opt)
        from nagahana.training.distributed import clip_global_norm

        norm = clip_global_norm(self.info, self.params, self.cfg.grad_clip)
        scale = float(self.scaler.get_scale()) if self.scaler.is_enabled() else None
        if not math.isfinite(norm):
            self.zero_grad()
            if self.scaler.is_enabled():
                self.scaler.update()                       # lowers the scale after an overflow
            if self.qk_clip is not None:
                for s in self.qk_clip.sites.values():
                    s.reset()                              # nothing changed: start the next step afresh
            self.skipped_total += 1
            self.consecutive_skips += 1
            if self.consecutive_skips > self.cfg.max_nonfinite_skips:
                raise InvariantViolation(f"non-finite gradient norm ({norm}) at optimiser step {self.step_count}: "
                                         f"refusing the update (max_nonfinite_skips={self.cfg.max_nonfinite_skips})")
            return StepReport(lr=self.lr_at(self.step_count), grad_norm=norm, skipped=True, loss_scale=scale, qk_clip={})
        lr = self.lr_at(self.step_count)
        for _, opt in self.optimizers():
            for g in opt.param_groups:
                g["lr"] = lr
            if self.scaler.is_enabled():
                self.scaler.step(opt)
            else:
                opt.step()
        if self.scaler.is_enabled():
            self.scaler.update()
        self.zero_grad()
        clip_stats = self.qk_clip.clip(self.info) if self.qk_clip is not None else {}
        self.step_count += 1
        self.consecutive_skips = 0
        return StepReport(lr=lr, grad_norm=norm, skipped=False, loss_scale=scale, qk_clip=clip_stats)

    def counters(self) -> dict[str, Any]:
        """The stepper's own state (the optimisers' moments are saved by the checkpoint code)."""
        return {"step_count": self.step_count, "skipped_total": self.skipped_total,
                "consecutive_skips": self.consecutive_skips, "total_steps": self.total_steps, "decay_from": self.decay_from,
                "scaler": self.scaler.state_dict() if self.scaler.is_enabled() else None,
                "qk_clip": self.qk_clip.state_dict() if self.qk_clip is not None else None}

    def load_counters(self, state: dict[str, Any]) -> None:
        """Restore `counters()`."""
        self.step_count = int(state["step_count"])
        self.skipped_total = int(state["skipped_total"])
        self.consecutive_skips = int(state["consecutive_skips"])
        self.total_steps = int(state["total_steps"])
        if state.get("decay_from") != self.decay_from:
            raise InvariantViolation(f"checkpoint decay start {state.get('decay_from')} differs from this run's "
                                     f"{self.decay_from} (a decayed branch and its main run are kept apart)")
        if state.get("scaler") is not None:
            if not self.scaler.is_enabled():
                raise InvariantViolation("the checkpoint holds a loss scaler but this run is not fp16")
            self.scaler.load_state_dict(state["scaler"])
        if state.get("qk_clip") is not None and self.qk_clip is not None:
            self.qk_clip.load_state_dict(state["qk_clip"])


__all__ = ["HybridStepper", "StepReport", "build_adamw", "build_muon", "decay_groups", "decay_start", "lr_multiplier",
           "named_trainable"]
