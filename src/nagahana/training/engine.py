"""The training engine of stages 1 to 3 and of site calibration: one loop, every strategy, exact resumption.

Structure
---------
A stage supplies a `StageProgram`: which modules train (registered in a `StageObjective`, the
`nn.Module` whose forward computes the stage loss, so DDP and FSDP see the whole step), which train
but stay replicated (long-term memory, the Verifier: their gradients are averaged by the engine), how
a batch is prepared, which batches carry objective work (stages 2 and 3: trigger-bearing batches,
AS-580), what is committed to the stream carry, and the validation objective. The engine owns
everything else:

- epochs over the stage's training segments in a class-balanced order (AS-325), each rank reading
  its share (AS-577), through a `ResumableStreamLoader`;
- accumulation of `accumulation` objective calls per optimiser step; the calls are synchronised
  across ranks (`all_true` before each call: an epoch ends for every rank when the first rank's
  stream ends, AS-577), gradient synchronisation only on the last call (DDP `no_sync`, FSDP
  `set_requires_gradient_sync`);
- loop passes R and descent steps S from a generator shared by every rank (AS-577), every other draw
  from the rank's own generator (AS-588);
- the hybrid optimiser with the stage's schedule, clipping and QK-Clip (training/optim.py), the
  optional weight EMA (AS-573), buffer broadcast after each step;
- validation every `eval_every` steps with fixed draws and the run-time budgets, metrics averaged over
  ranks; early stopping and the best weights (AS-574);
- resumable checkpoints every `checkpoint.every_steps` steps and at the end (training/checkpoint.py),
  with every random state, the stream position and the carry (AS-576);
- metrics to the run logger (rank 0), frozen-module verification (pipeline/freezing.py, AS-593).

Determinism: given the configuration, the data and the seed, a run produces the same parameters on the
same hardware and software; a resumed run produces the parameters of the uninterrupted run
(tests/test_training_checkpoint.py checks bit equality on CPU).
"""

from __future__ import annotations

import contextlib
import dataclasses
import math
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn

from nagahana.core.errors import InvariantViolation
from nagahana.data.sampling import WindowRecord
from nagahana.data.windows import PreparedSource
from nagahana.models.nagahana import NagaHana, model_hash
from nagahana.pipeline.freezing import FreezeGuard
from nagahana.tracking.logger import RunLogger
from nagahana.training.carry import PreparedBatch, StreamBridge
from nagahana.training.checkpoint import (
    CheckpointManager,
    gather_model_state,
    gather_optimizer_state,
    load_model_state,
    load_optimizer_state,
)
from nagahana.training.config import LoopConfig
from nagahana.training.data import ResumableStreamLoader, StepPlan, epoch_order, plan_steps, shard_order
from nagahana.training.distributed import (
    DistInfo,
    all_true,
    allreduce_mean_grads,
    apply_recomputation,
    broadcast_buffers,
    stack_blocks,
    sum_values,
    unwrap,
    wrap_objective,
)
from nagahana.training.ema import WeightEMA
from nagahana.training.optim import HybridStepper, named_trainable
from nagahana.training.qkclip import QKClip
from nagahana.training.randomness import (
    capture_rng_state,
    derive_seed,
    generator_state,
    make_generator,
    restore_rng_state,
    set_generator_state,
)


def to_device(obj: Any, device: torch.device) -> Any:
    """`obj` with every tensor inside (dataclasses, dicts, lists, tuples) moved to `device`, dtypes kept."""
    if isinstance(obj, torch.Tensor):
        return obj.to(device) if obj.device != device else obj
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        changes = {f.name: to_device(getattr(obj, f.name), device) for f in dataclasses.fields(obj)}
        return dataclasses.replace(obj, **changes)
    if isinstance(obj, dict):
        return {k: to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_device(v, device) for v in obj]
    if isinstance(obj, tuple):
        return tuple(to_device(v, device) for v in obj)
    return obj


def detach_tree(obj: Any) -> Any:
    """`obj` with every tensor detached (dataclasses, dicts, lists, tuples)."""
    if isinstance(obj, torch.Tensor):
        return obj.detach()
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.replace(obj, **{f.name: detach_tree(getattr(obj, f.name)) for f in dataclasses.fields(obj)})
    if isinstance(obj, dict):
        return {k: detach_tree(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [detach_tree(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(detach_tree(v) for v in obj)
    return obj


@dataclass
class Draws:
    """The random streams of one rank: `local` (augmentation, masking, negatives) and `shared` (R, S)."""

    local: torch.Generator
    shared: torch.Generator


@dataclass
class ObjectiveOut:
    """What a stage objective returns: the loss (the only tensor with gradient), scalar parts, auxiliaries."""

    loss: torch.Tensor
    parts: dict[str, float]
    aux: dict[str, Any] = field(default_factory=dict)


class StageObjective(nn.Module):
    """The modules a stage trains, registered under their model names, and the forward computing the loss.

    The model itself is held by reference (`self.model`), not registered, so DDP and FSDP manage
    exactly the registered (trained) modules; `replicated` modules train without being registered and
    their gradients are averaged by the engine.
    """

    def __init__(self, model: NagaHana, trained: Sequence[str], replicated: Sequence[str],
                 compute: Callable[..., ObjectiveOut]) -> None:
        super().__init__()
        object.__setattr__(self, "model", model)
        for name in trained:
            self.add_module(name, getattr(model, name))
        object.__setattr__(self, "replicated_names", tuple(replicated))
        object.__setattr__(self, "_compute", compute)

    def forward(self, *args: Any, **kwargs: Any) -> tuple[torch.Tensor, ObjectiveOut]:
        out = self._compute(self.model, *args, **kwargs)  # type: ignore[operator]
        return out.loss, out

    def replicated_modules(self) -> list[nn.Module]:
        return [getattr(self.model, n) for n in self.replicated_names]  # type: ignore[attr-defined]


class StageProgram:
    """What a stage plugs into the engine (module docstring). Subclasses override the hooks."""

    stage: int = 0
    name: str = ""
    #: True: only batches with a trigger carry objective work (stages 2 and 3, AS-580).
    needs_trigger: bool = False

    def __init__(self, model: NagaHana) -> None:
        self.model = model
        self.qk_clip: QKClip | None = None

    def attach_qk_clip(self, clip: QKClip | None) -> None:
        """Receive the stage's QK-Clip; programs whose attention reads keys produced elsewhere register
        those key sources here and observe them in `compute` (training/qkclip.py)."""
        self.qk_clip = clip

    def trained(self) -> list[str]:
        """Model components trained and registered (sharded under FSDP)."""
        raise NotImplementedError

    def replicated(self) -> list[str]:
        """Model components trained but kept replicated (gradients averaged by the engine)."""
        return []

    def frozen(self) -> list[str]:
        """Model components that must not change in this stage."""
        raise NotImplementedError

    def compute(self, model: NagaHana, prep: PreparedBatch, pre: Any, draws: Draws, step: int, train: bool) -> ObjectiveOut:
        """The stage objective on one prepared batch (called inside the objective's forward)."""
        raise NotImplementedError

    def pre(self, prep: PreparedBatch, draws: Draws, bridge: StreamBridge) -> Any:
        """Work done before the objective without gradient (frozen perception, carried Imagination); None = nothing."""
        return None

    def has_work(self, prep: PreparedBatch, pre: Any) -> bool:
        """True when the batch carries objective work."""
        return (not self.needs_trigger) or bool(prep.window.triggers.mask.any())

    def post(self, bridge: StreamBridge, prep: PreparedBatch, pre: Any, out: ObjectiveOut | None) -> None:
        """Commit the window to the stream carry after the objective (or without one)."""
        raise NotImplementedError

    def on_plan(self, total_steps: int) -> None:
        """Called once the stage's optimiser steps are planned (schedules of stage switches)."""
        return None

    def before_step(self, step: int) -> None:
        """Called before each optimiser step's first objective call (stage switches)."""
        return None

    def extra_step(self, outs: Sequence[ObjectiveOut], step: int) -> dict[str, float]:
        """Optional second optimisation after the main step (the Verifier); returns logged scalars."""
        return {}

    def state_dict(self) -> dict[str, Any]:
        return {}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        return None

    def extra_optimizer_state(self) -> dict[str, Any]:
        return {}

    def load_extra_optimizer_state(self, state: Mapping[str, Any]) -> None:
        return None


@dataclass
class StageData:
    """The stage's data: sources and segment records of the training and validation scopes."""

    sources: list[PreparedSource]
    train: list[WindowRecord]
    val: list[WindowRecord]
    label_limits: dict[str, float]
    novelty: dict[str, str] = field(default_factory=dict)


@dataclass
class StageResult:
    """The outcome of a stage run."""

    stage: int
    steps: int
    epochs_completed: int
    stopped_early: bool
    best_metric: float | None
    best_step: int | None
    history: list[dict[str, float]]
    validations: list[dict[str, float]]
    seconds: float
    notes: list[str]
    model_hash: str | None
    interrupted: bool = False


class Engine:
    """Runs one stage program (module docstring)."""

    def __init__(self, program: StageProgram, *, data: StageData, loop: LoopConfig, info: DistInfo, seed: int,
                 run_dir: str, logger: RunLogger, recompute: Sequence[str] = (), shard_units: Sequence[str] = (),
                 dist_cfg: Any = None, bridge_factory: Callable[[], StreamBridge] | None = None,
                 decay_from: int | None = None, total_steps: int | None = None, perturb: bool = False,
                 checkpoint_subdir: str | None = None) -> None:
        self.program = program
        self.model = program.model
        self.data = data
        self.loop = loop
        self.info = info
        self.seed = int(seed)
        self.logger = logger
        self.notes: list[str] = []
        self.perturb = perturb
        cfg = self.model.cfg
        # Freezing: every component that is not trained here is frozen and verified (AS-593).
        trained = set(program.trained()) | set(program.replicated())
        for name in program.frozen():
            if name in trained:
                raise InvariantViolation(f"{name} is both trained and frozen in stage {program.stage}")
        from nagahana.pipeline.freezing import freeze, unfreeze

        freeze([getattr(self.model, n) for n in program.frozen()])
        unfreeze([getattr(self.model, n) for n in sorted(trained)])
        # Recomputation of block activations (AS-578), installed before any wrapping.
        self._undo_recompute: list[Callable[[], None]] = []
        for comp in recompute:
            if comp in trained:
                self._undo_recompute.append(apply_recomputation(stack_blocks(getattr(self.model, comp))))
        objective = StageObjective(self.model, program.trained(), program.replicated(), program.compute)
        shard_blocks = [blk for comp in shard_units if comp in program.trained()
                        for blk in stack_blocks(getattr(self.model, comp))]
        if dist_cfg is None:
            from nagahana.training.config import DistributedConfig

            dist_cfg = DistributedConfig()
        self.objective = objective
        self.wrapped = wrap_objective(objective, info, dist_cfg, shard_blocks=shard_blocks)
        # The optimiser over trained parameters, names as the model names them.
        named = named_trainable([(n, getattr(self.model, n)) for n in program.trained()])
        self.replicated_params = [p for _, p in named_trainable([(n, getattr(self.model, n)) for n in program.replicated()])]
        named += named_trainable([(n, getattr(self.model, n)) for n in program.replicated()])
        self.named = named
        self.loader_lanes = loop.lanes
        orders = [epoch_order(data.train, epoch=e, seed=derive_seed(self.seed, "order", program.stage), balanced=loop.balanced_sampling,
                              power=1.0) for e in range(loop.epochs)]
        self.orders = orders
        if total_steps is None:
            plan: StepPlan = plan_steps(data.sources, data.train, cfg, orders=orders, world=info.world, lanes=loop.lanes,
                                        accumulation=loop.accumulation, needs_trigger=program.needs_trigger,
                                        max_steps=loop.max_steps)
            total_steps = plan.total_steps
            self.step_plan: StepPlan | None = plan
        else:
            self.step_plan = None
        if total_steps < 1:
            raise InvariantViolation(f"stage {program.stage} has no optimiser step to take on its training data "
                                     f"(lanes={loop.lanes}, accumulation={loop.accumulation}, world={info.world})")
        self.total_steps = int(total_steps)
        self.qk_clip = QKClip(objective, tau=loop.optim.qk_clip_tau) if loop.optim.qk_clip else None
        program.attach_qk_clip(self.qk_clip)
        self.stepper = HybridStepper(self.model, named, loop.optim, total_steps=self.total_steps, precision=loop.precision,
                                     device_type=info.device.type, qk_clip=self.qk_clip, info=info, decay_from=decay_from)
        program.on_plan(self.total_steps)
        self.ema = (WeightEMA(named, decay=loop.ema.decay, warmup=loop.ema.warmup) if loop.ema.enabled else None)
        self.ckpt = CheckpointManager(f"{run_dir}/checkpoints" if checkpoint_subdir is None else checkpoint_subdir,
                                      stage=program.stage, info=info, keep_last=loop.checkpoint.keep_last)
        self.guard = FreezeGuard({n: getattr(self.model, n) for n in program.frozen()},
                                 optimizers=[o for _, o in self.stepper.optimizers()])
        self.draws = Draws(local=make_generator(derive_seed(self.seed, "local", program.stage, info.rank)),
                           shared=make_generator(derive_seed(self.seed, "shared", program.stage)))
        self.bridge_factory = bridge_factory or (lambda: StreamBridge(self.model))
        self.bridge = self.bridge_factory()
        self.epoch = 0
        self.loader_state: dict[str, Any] | None = None
        self.best: float | None = None
        self.best_step: int | None = None
        self.bad_evals = 0
        self.history: list[dict[str, float]] = []
        self.validations: list[dict[str, float]] = []
        self.stopped_early = False

    # ------------------------------------------------------------------ data
    def _loader(self, records: Sequence[WindowRecord], order: Sequence[int], *, train: bool,
                epoch: int) -> ResumableStreamLoader:
        perturb_seed = derive_seed(self.seed, "perturb", self.program.stage, epoch) if (train and self.perturb) else None
        return ResumableStreamLoader(self.data.sources, list(records), self.model.cfg, order=list(order),
                                     lanes=self.loader_lanes, label_limits=self.data.label_limits,
                                     novelty=self.data.novelty, perturb_seed=perturb_seed)

    def _rank_order(self, epoch: int) -> list[int]:
        return shard_order(self.orders[epoch], rank=self.info.rank, world=self.info.world)

    # ------------------------------------------------------------------ one objective call
    @contextlib.contextmanager
    def _sync(self, last: bool) -> Iterator[None]:
        # Gradient synchronisation only on the last micro-batch of an optimiser step.
        w = self.wrapped
        if isinstance(w, nn.parallel.DistributedDataParallel) and not last:
            with w.no_sync():
                yield
            return
        setter = getattr(w, "set_requires_gradient_sync", None)
        if self.info.strategy == "fsdp" and callable(setter):
            setter(last)
            try:
                yield
            finally:
                setter(True)
            return
        yield

    def _autocast(self) -> contextlib.AbstractContextManager[Any]:
        p = self.loop.precision
        if p == "fp32":
            return contextlib.nullcontext()
        dtype = torch.bfloat16 if p == "bf16" else torch.float16
        return torch.autocast(device_type=self.info.device.type, dtype=dtype)

    def _call(self, prep: PreparedBatch, pre: Any, *, last: bool) -> ObjectiveOut:
        step = self.stepper.step_count
        with self._sync(last):
            track = self.qk_clip.tracking() if self.qk_clip is not None else contextlib.nullcontext()
            with track, self._autocast():
                loss, out = self.wrapped(prep, pre, self.draws, step, True)
            if not torch.isfinite(loss.detach()):
                self.notes.append(f"step {step}: non-finite loss {float(loss.detach())}")
            self.stepper.backward(loss.float(), accumulation=self.loop.accumulation)
        return out

    # ------------------------------------------------------------------ the loop
    def run(self, *, resume: bool, stop_after: int | None = None) -> StageResult:
        """Train the stage (resuming from its latest checkpoint when asked and when one exists).

        stop_after: interrupt after this many optimiser steps with a resumable checkpoint (preemption;
        the plan, and hence the schedule, is unchanged). The result then reports `interrupted`.
        """
        t0 = time.perf_counter()
        self.stop_after = stop_after
        self.interrupted = False
        steps_at_start = 0
        if resume:
            ptr = self.ckpt.latest()
            if ptr is not None:
                self._load(ptr)
                steps_at_start = self.stepper.step_count
                self.notes.append(f"resumed from step {ptr.step}")
        self.guard.start()
        loop = self.loop
        finished_before = (self.stepper.step_count >= self.total_steps or self.epoch >= loop.epochs or self.stopped_early)
        while (self.epoch < loop.epochs and self.stepper.step_count < self.total_steps and not self.stopped_early
               and not self.interrupted):
            loader = self._loader(self.data.train, self._rank_order(self.epoch), train=True, epoch=self.epoch)
            if self.loader_state is not None:
                loader.load_state_dict(self.loader_state)
                self.loader_state = None
            else:
                self.bridge = self.bridge_factory()
            ended = self._epoch(loader)
            if ended:
                self.epoch += 1
                self.bridge = self.bridge_factory()
        if self.interrupted:
            for undo in self._undo_recompute:
                undo()
            return StageResult(stage=self.program.stage, steps=self.stepper.step_count, epochs_completed=self.epoch,
                               stopped_early=False, best_metric=self.best, best_step=self.best_step, history=self.history,
                               validations=self.validations, seconds=time.perf_counter() - t0,
                               notes=[*self.notes, f"interrupted after step {self.stepper.step_count}"], model_hash=None,
                               interrupted=True)
        if not finished_before and self.stepper.step_count > steps_at_start:
            self._validate_and_track(final=True)
            self._save(loader_state=None)
        if self.loop.early_stop.enabled and self.loop.early_stop.restore_best:
            self._restore_best()
        self.guard.verify_full()
        for undo in self._undo_recompute:
            undo()
        return StageResult(stage=self.program.stage, steps=self.stepper.step_count, epochs_completed=self.epoch,
                           stopped_early=self.stopped_early, best_metric=self.best, best_step=self.best_step,
                           history=self.history, validations=self.validations, seconds=time.perf_counter() - t0,
                           notes=self.notes, model_hash=model_hash(self.model) if not self.info.strategy == "fsdp" else None)

    def _next_work(self, it: Iterator[Any]) -> tuple[PreparedBatch, Any] | None:
        """Advance the rank's stream to the next batch with objective work, committing the others."""
        for w, lab, ctxs in it:
            prep = self._prepare(w, lab, ctxs)
            pre = self.program.pre(prep, self.draws, self.bridge)
            if self.program.has_work(prep, pre):
                return prep, pre
            self.program.post(self.bridge, prep, pre, None)
        return None

    def _prepare(self, w: Any, lab: Any, ctxs: Any) -> PreparedBatch:
        prep = self.bridge.prepare(w, lab, ctxs)
        if self.info.device.type != "cpu":
            prep = dataclasses.replace(prep, window=to_device(prep.window, self.info.device),
                                       labels=to_device(prep.labels, self.info.device),
                                       carry=to_device(prep.carry, self.info.device),
                                       entity_keys=to_device(prep.entity_keys, self.info.device))
        return prep

    def _epoch(self, loader: ResumableStreamLoader) -> bool:
        """One pass (or the rest of one); True when the epoch's stream ended."""
        it = iter(loader)
        loop = self.loop
        while self.stepper.step_count < self.total_steps and not self.stopped_early:
            self.program.before_step(self.stepper.step_count)
            outs: list[ObjectiveOut] = []
            for k in range(loop.accumulation):
                found = self._next_work(it)
                if not all_true(self.info, found is not None):
                    self.stepper.zero_grad()                     # a partial accumulation is discarded
                    for o in outs:
                        del o
                    return True
                assert found is not None
                prep, pre = found
                out = self._call(prep, pre, last=k == loop.accumulation - 1)
                self.program.post(self.bridge, prep, pre, out)
                outs.append(out)
            allreduce_mean_grads(self.info, self.replicated_params)
            report = self.stepper.step()
            extra = self.program.extra_step(outs, self.stepper.step_count)
            broadcast_buffers(self.info, [self.objective, *self.objective.replicated_modules()])
            if self.ema is not None and not report.skipped:
                self.ema.update()
            self.guard.check(self.stepper.step_count)
            self._log(outs, report, extra)
            step = self.stepper.step_count
            if loop.eval_every and step % loop.eval_every == 0 and step < self.total_steps:
                self._validate_and_track(final=False)
            if step % loop.checkpoint.every_steps == 0 and step < self.total_steps:
                self._save(loader_state=loader.state_dict())
            if self.stop_after is not None and step >= self.stop_after and step < self.total_steps:
                if step % loop.checkpoint.every_steps != 0:
                    self._save(loader_state=loader.state_dict())
                self.interrupted = True
                return False
        return False

    def _log(self, outs: Sequence[ObjectiveOut], report: Any, extra: Mapping[str, float]) -> None:
        keys = sorted(outs[0].parts) if outs else []
        sums = [sum(o.parts.get(k, 0.0) for o in outs) for k in keys]
        tot = sum_values(self.info, [*sums, float(len(outs))])
        n = tot[-1] if tot else 1.0
        row = {f"stage{self.program.stage}/{k}": tot[i] / max(n, 1.0) for i, k in enumerate(keys)}
        row |= {f"stage{self.program.stage}/lr": report.lr, f"stage{self.program.stage}/grad_norm": report.grad_norm,
                f"stage{self.program.stage}/skipped": float(report.skipped), "step": float(self.stepper.step_count)}
        row |= {f"stage{self.program.stage}/{k}": float(v) for k, v in report.qk_clip.items()}
        row |= {f"stage{self.program.stage}/{k}": float(v) for k, v in extra.items()}
        self.history.append(row)
        if self.info.is_main and self.stepper.step_count % self.loop.log_every == 0:
            self.logger.log_metrics(row, step=self.stepper.step_count)

    # ------------------------------------------------------------------ validation and early stopping
    def evaluate(self) -> dict[str, float]:
        """The stage objective on validation segments (eval mode, fixed draws, run-time budgets), averaged over ranks."""
        records = self.data.val
        if not records:
            return {}
        order = shard_order(sorted(range(len(records)), key=lambda i: (records[i].t_start, records[i].id)),
                            rank=self.info.rank, world=self.info.world)
        loader = self._loader(records, order, train=False, epoch=0)
        bridge = self.bridge_factory()
        draws = Draws(local=make_generator(derive_seed(self.seed, "val-local", self.program.stage, self.info.rank)),
                      shared=make_generator(derive_seed(self.seed, "val-shared", self.program.stage)))
        saved_bridge, saved_draws = self.bridge, self.draws
        self.bridge, self.draws = bridge, draws
        sums: dict[str, float] = {}
        calls = 0
        modules = [self.objective, *self.objective.replicated_modules()]
        was_training = [m.training for m in modules]
        for m in modules:
            m.eval()
        ema_ctx = (self.ema.swapped() if (self.ema is not None and self.loop.ema.evaluate_with_ema)
                   else contextlib.nullcontext())
        try:
            with ema_ctx, torch.no_grad():
                it = iter(loader)
                for _ in range(self.loop.eval_batches):
                    found = self._next_work(it)
                    if not all_true(self.info, found is not None):
                        break
                    assert found is not None
                    prep, pre = found
                    with self._autocast():
                        _, out = self.wrapped(prep, pre, self.draws, self.stepper.step_count, False)
                    self.program.post(self.bridge, prep, pre, out)
                    for k, v in out.parts.items():
                        sums[k] = sums.get(k, 0.0) + float(v)
                    calls += 1
        finally:
            for m, t in zip(modules, was_training, strict=True):
                m.train(t)
            self.bridge, self.draws = saved_bridge, saved_draws
        keys = sorted(sums)
        tot = sum_values(self.info, [*[sums[k] for k in keys], float(calls)])
        if not tot or tot[-1] <= 0:
            return {}
        return {f"val/{k}": tot[i] / tot[-1] for i, k in enumerate(keys)} | {"val/batches": tot[-1]}

    def _validate_and_track(self, *, final: bool) -> None:
        metrics = self.evaluate()
        if not metrics:
            if not self.data.val:
                self.notes.append("no validation segments: early stopping and best weights are not available")
            return
        metrics["step"] = float(self.stepper.step_count)
        self.validations.append(metrics)
        if self.info.is_main:
            self.logger.log_metrics({f"stage{self.program.stage}/{k}": v for k, v in metrics.items()}, step=self.stepper.step_count)
        es = self.loop.early_stop
        if not es.enabled:
            return
        if es.monitor not in metrics:
            raise InvariantViolation(f"early_stop.monitor {es.monitor!r} is not a validation metric; have {sorted(metrics)}")
        value = metrics[es.monitor]
        better = (self.best is None or (value < self.best - es.min_delta if es.mode == "min" else value > self.best + es.min_delta))
        if better and math.isfinite(value):
            self.best, self.best_step, self.bad_evals = value, self.stepper.step_count, 0
            self.ckpt.save_best(lambda: self._best_state(), meta={"step": self.stepper.step_count, "metric": value})
        elif not final:
            self.bad_evals += 1
            if self.bad_evals >= es.patience:
                self.stopped_early = True
                self.notes.append(f"early stop at step {self.stepper.step_count}: {es.monitor} did not improve for "
                                  f"{es.patience} evaluations (best {self.best} at step {self.best_step})")

    def _best_state(self) -> dict[str, Any] | None:
        ctx = self.ema.swapped() if (self.ema is not None and self.loop.ema.evaluate_with_ema) else contextlib.nullcontext()
        with ctx:
            return {"model": gather_model_state(self.model, self.objective, self.info)}

    def _restore_best(self) -> None:
        best = self.ckpt.load_best()
        if best is None:
            return
        state, meta = best
        load_model_state(self.model, self.objective, self.info, state["model"])
        self.notes.append(f"restored the best weights of step {meta.get('step')} ({self.loop.early_stop.monitor} = {meta.get('metric')})")

    # ------------------------------------------------------------------ checkpoints
    def _global_state(self) -> dict[str, Any] | None:
        model = gather_model_state(self.model, self.objective, self.info)
        optim = {name: gather_optimizer_state(opt, self.objective, self.info) for name, opt in self.stepper.optimizers()}
        ema = self.ema.state_dict() if self.ema is not None else None
        if not self.info.is_main:
            return None
        return {"model": model, "optimizers": optim, "stepper": self.stepper.counters(), "ema": ema,
                "engine": {"epoch": self.epoch, "best": self.best, "best_step": self.best_step, "bad_evals": self.bad_evals,
                           "stopped_early": self.stopped_early, "total_steps": self.total_steps,
                           "history": self.history, "validations": self.validations, "notes": self.notes},
                "program": self.program.state_dict(), "extra_optimizers": self.program.extra_optimizer_state(),
                "shared_generator": generator_state(self.draws.shared)}

    def _rank_state(self, loader_state: dict[str, Any] | None) -> dict[str, Any]:
        return {"rng": capture_rng_state(), "local_generator": generator_state(self.draws.local), "loader": loader_state,
                "bridge": self.bridge.state_dict(), "epoch": self.epoch}

    def _save(self, *, loader_state: dict[str, Any] | None) -> None:
        self.guard.verify_full()
        meta = {"stage": self.program.stage, "step": self.stepper.step_count, "epoch": self.epoch}
        self.ckpt.save(step=self.stepper.step_count, global_state=self._global_state, rank_state=self._rank_state(loader_state),
                       meta=meta)

    def _load(self, ptr: Any) -> None:
        g, r = self.ckpt.load(ptr)
        load_model_state(self.model, self.objective, self.info, g["model"])
        for name, opt in self.stepper.optimizers():
            load_optimizer_state(opt, self.objective, self.info, g["optimizers"][name])
        self.stepper.load_counters(g["stepper"])
        if self.ema is not None:
            if g.get("ema") is None:
                raise InvariantViolation("this run keeps an EMA but the checkpoint has none")
            self.ema.load_state_dict(g["ema"])
        e = g["engine"]
        if int(e["total_steps"]) != self.total_steps:
            raise InvariantViolation(f"the checkpoint planned {e['total_steps']} steps, this run plans {self.total_steps} "
                                     "(data, lanes, accumulation or world size differ)")
        self.epoch = int(e["epoch"])
        self.best, self.best_step, self.bad_evals = e["best"], e["best_step"], int(e["bad_evals"])
        self.stopped_early = bool(e["stopped_early"])
        self.history, self.validations, self.notes = list(e["history"]), list(e["validations"]), list(e["notes"])
        self.program.load_state_dict(g["program"])
        self.program.load_extra_optimizer_state(g["extra_optimizers"])
        set_generator_state(self.draws.shared, g["shared_generator"])
        restore_rng_state(r["rng"])
        set_generator_state(self.draws.local, r["local_generator"])
        self.bridge = self.bridge_factory()
        self.bridge.load_state_dict(r["bridge"])
        if int(r["epoch"]) != self.epoch:
            raise InvariantViolation("rank state and global state belong to different epochs")
        self.loader_state = r["loader"]


class ProgramTrainer:
    """One process, prepared batches, the program's own objective and optimiser (tests, smoke runs).

    The same objective, optimiser and freezing as the engine, without streams, ranks or checkpoints:
    `loss(prep, pre, seed)` evaluates the objective with fixed draws (a same-batch check), `step(prep, pre)`
    takes one optimiser step with the trainer's own draws.
    """

    def __init__(self, program: StageProgram, *, optim: Any, total_steps: int, seed: int) -> None:
        from nagahana.pipeline.freezing import freeze, unfreeze

        model = program.model
        self.program = program
        freeze([getattr(model, n) for n in program.frozen()])
        unfreeze([getattr(model, n) for n in (*program.trained(), *program.replicated())])
        named = named_trainable([(n, getattr(model, n)) for n in (*program.trained(), *program.replicated())])
        self.stepper = HybridStepper(model, named, optim, total_steps=total_steps, precision="fp32")
        self.draws = Draws(local=make_generator(derive_seed(seed, "local", program.stage, 0)),
                           shared=make_generator(derive_seed(seed, "shared", program.stage)))

    def loss(self, prep: PreparedBatch, pre: Any, seed: int) -> torch.Tensor:
        """The objective on one batch with draws fixed by `seed`."""
        d = Draws(local=make_generator(derive_seed(seed, "fixed-local")), shared=make_generator(derive_seed(seed, "fixed-shared")))
        return self.program.compute(self.program.model, prep, pre, d, self.stepper.step_count, True).loss

    def step(self, prep: PreparedBatch, pre: Any) -> dict[str, float]:
        """One optimiser step on one batch."""
        self.program.before_step(self.stepper.step_count)
        out = self.program.compute(self.program.model, prep, pre, self.draws, self.stepper.step_count, True)
        self.stepper.backward(out.loss, accumulation=1)
        rep = self.stepper.step()
        extra = self.program.extra_step([out], self.stepper.step_count)
        return out.parts | {"lr": rep.lr, "grad_norm": rep.grad_norm} | extra


__all__ = ["Draws", "Engine", "ObjectiveOut", "ProgramTrainer", "StageData", "StageObjective", "StageProgram", "StageResult",
           "detach_tree", "to_device"]
