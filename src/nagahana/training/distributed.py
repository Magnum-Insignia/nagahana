"""Multi-process training: process groups, DDP and FSDP wrapping, block recomputation, collectives.

Purpose
-------
NagaHana L (1,134,268,667 parameters) trains on several accelerators (docs/sizing.md: eight 80 GB
accelerators; a mixed-precision Adam state of 16.9 GiB; TAAFT activations of 74 GiB per micro-batch
without recomputation). This module provides:

- `init_distributed`: the process group from the torchrun environment (RANK, WORLD_SIZE, LOCAL_RANK,
  MASTER_ADDR, MASTER_PORT); NCCL on CUDA hosts, Gloo on CPU hosts; one process when WORLD_SIZE is 1.
- `wrap_objective`: a stage objective (an `nn.Module` whose forward computes the stage loss) wrapped
  for the strategy:
    "single"  as it is;
    "ddp"     `DistributedDataParallel` (Li et al., VLDB 2020, arXiv:2006.15704), unused parameters found
              per step because stage graphs vary with R and the data;
    "fsdp"    fully sharded data parallelism with the per-parameter sharding of `fully_shard` (Zhao et al.,
              VLDB 2023, arXiv:2304.11277; ZeRO stage 3, Rajbhandari et al., SC 2020, arXiv:1910.02054):
              every block of the configured components (TSTCT, TAAFT) is one unit, the objective is the
              root unit holding everything else. A block's `kv` method is registered as a forward method,
              because the two-stream loop calls it outside `forward` (nn/loop.py). Parameters stay float32
              (no parameter casting policy): bf16 compute comes from autocast exactly as without sharding,
              so norms, softmax, energies and time angles keep their float32/float64 precision (AS-39, D-54).
- `apply_recomputation`: activation recomputation of whole blocks with non-reentrant
  `torch.utils.checkpoint` (Chen et al. 2016, arXiv:1604.06174; Korthikanti et al. 2022, arXiv:2205.05198),
  installed on block instances so module names, state dicts and hashes are unchanged (AS-578).
- Collectives that are no-ops in one process: flags, sums, means, broadcasts, barriers.

Rules that keep ranks in step (AS-577, AS-580)
---------------------------------------------
- Ranks read disjoint segments; every collective is reached by every rank the same number of times.
- Under FSDP each block's forward gathers its parameters, so all ranks must run the same number of
  block calls: loop passes R and descent steps S come from a generator shared by all ranks (AS-577).
- Buffers updated in the forward pass (TAAFT's lens reference levels, AS-210) are broadcast from rank 0
  after each optimiser step (DDP's own buffer semantics, applied to every strategy).

Invariant (tests/test_training_distributed.py): two CPU ranks with Gloo produce the same parameters
after a step as one process that averages the two ranks' gradients.
"""

from __future__ import annotations

import datetime
import functools
import math
import os
import tempfile
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.distributed as dist
from torch import nn

from nagahana.core.errors import ConfigMissing, InvariantViolation
from nagahana.nn.loop import TwoStreamStack
from nagahana.training.assumptions import use
from nagahana.training.config import DistributedConfig


@dataclass(frozen=True)
class DistInfo:
    """Where this process stands in the run."""

    rank: int
    world: int
    local_rank: int
    device: torch.device
    backend: str | None
    strategy: str
    initialized_here: bool
    mesh: Any = field(default=None, compare=False)

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    @property
    def distributed(self) -> bool:
        return self.world > 1 or self.strategy in ("ddp", "fsdp")


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        raise ConfigMissing(f"environment variable {name}={raw!r} is not an integer") from None


def init_distributed(cfg: DistributedConfig, *, device: str | None = None) -> DistInfo:
    """Initialise (or adopt) the process group for this run (module docstring).

    device: "cpu", "cuda" or None (CUDA when available). A strategy of "ddp" or "fsdp" in a single
    process initialises a one-process group through a file store, so the same code path is exercised.
    """
    world = _env_int("WORLD_SIZE", 1)
    rank = _env_int("RANK", 0)
    local_rank = _env_int("LOCAL_RANK", 0)
    if not 0 <= rank < world:
        raise ConfigMissing(f"RANK={rank} outside WORLD_SIZE={world}")
    use_cuda = (device == "cuda") or (device is None and torch.cuda.is_available())
    if use_cuda and not torch.cuda.is_available():
        raise ConfigMissing("device 'cuda' requested but CUDA is not available")
    dev = torch.device(f"cuda:{local_rank}") if use_cuda else torch.device("cpu")
    strategy = cfg.strategy
    if strategy == "auto":
        strategy = "single" if world == 1 else "fsdp"
    if strategy == "single" and world > 1:
        raise ConfigMissing("strategy 'single' with WORLD_SIZE > 1: choose 'ddp' or 'fsdp'")
    backend: str | None = None
    initialized_here = False
    mesh = None
    if strategy in ("ddp", "fsdp"):
        backend = cfg.backend if cfg.backend != "auto" else ("nccl" if use_cuda else "gloo")
        if backend == "nccl" and not use_cuda:
            raise ConfigMissing("the NCCL backend needs CUDA devices")
        if use_cuda:
            torch.cuda.set_device(dev)
        if not dist.is_initialized():
            timeout = datetime.timedelta(seconds=cfg.timeout_s)
            if world == 1 and "MASTER_ADDR" not in os.environ:
                fd, path = tempfile.mkstemp(prefix="nagahana-pg-")
                os.close(fd)
                os.remove(path)
                dist.init_process_group(backend, init_method="file:///" + path.replace("\\", "/"), rank=0,
                                        world_size=1, timeout=timeout)
            else:
                dist.init_process_group(backend, timeout=timeout)
            initialized_here = True
        if dist.get_world_size() != world:
            raise InvariantViolation(f"process group has {dist.get_world_size()} ranks, WORLD_SIZE says {world}")
        if strategy == "fsdp":
            from torch.distributed.device_mesh import init_device_mesh

            mesh = init_device_mesh(dev.type, (world,))
    elif use_cuda:
        torch.cuda.set_device(dev)
    return DistInfo(rank=rank, world=world, local_rank=local_rank, device=dev, backend=backend, strategy=strategy,
                    initialized_here=initialized_here, mesh=mesh)


def shutdown(info: DistInfo) -> None:
    """Destroy the process group if this run created it."""
    if info.initialized_here and dist.is_initialized():
        dist.destroy_process_group()


def _active(info: DistInfo) -> bool:
    return info.distributed and dist.is_initialized()


def barrier(info: DistInfo) -> None:
    """Wait for every rank."""
    if _active(info):
        dist.barrier()


def _flag_tensor(info: DistInfo, value: float) -> torch.Tensor:
    dev = info.device if info.backend == "nccl" else torch.device("cpu")
    return torch.tensor([value], dtype=torch.float64, device=dev)


def all_true(info: DistInfo, flag: bool) -> bool:
    """True when `flag` holds on every rank."""
    if not _active(info):
        return bool(flag)
    t = _flag_tensor(info, 1.0 if flag else 0.0)
    dist.all_reduce(t, op=dist.ReduceOp.MIN)
    return bool(t.item() > 0.5)


def any_true(info: DistInfo, flag: bool) -> bool:
    """True when `flag` holds on at least one rank."""
    if not _active(info):
        return bool(flag)
    t = _flag_tensor(info, 1.0 if flag else 0.0)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return bool(t.item() > 0.5)


def sum_values(info: DistInfo, values: Sequence[float]) -> list[float]:
    """Element-wise sum of a list of floats over ranks (float64)."""
    if not _active(info):
        return [float(v) for v in values]
    dev = info.device if info.backend == "nccl" else torch.device("cpu")
    t = torch.tensor(list(values), dtype=torch.float64, device=dev)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return [float(x) for x in t.tolist()]


def weighted_means(info: DistInfo, sums: dict[str, float], counts: dict[str, float]) -> dict[str, float]:
    """Global means sum_r sums[k] / sum_r counts[k] for every key (keys must match on every rank)."""
    keys = sorted(sums)
    if sorted(counts) != keys:
        raise InvariantViolation("weighted_means needs the same keys for sums and counts")
    tot = sum_values(info, [sums[k] for k in keys] + [counts[k] for k in keys])
    n = len(keys)
    return {k: (tot[i] / tot[n + i] if tot[n + i] > 0 else float("nan")) for i, k in enumerate(keys)}


def broadcast_object(info: DistInfo, obj: Any, *, src: int = 0) -> Any:
    """`obj` of rank `src` on every rank (between the run's own processes)."""
    if not _active(info):
        return obj
    box = [obj if info.rank == src else None]
    dist.broadcast_object_list(box, src=src)
    return box[0]


def gather_objects(info: DistInfo, obj: Any) -> list[Any]:
    """Every rank's `obj`, in rank order, on every rank."""
    if not _active(info):
        return [obj]
    out: list[Any] = [None] * info.world
    dist.all_gather_object(out, obj)
    return out


def broadcast_buffers(info: DistInfo, modules: Iterable[nn.Module]) -> None:
    """Rank 0's buffers on every rank (after an optimiser step; module docstring)."""
    if not _active(info):
        return
    seen: set[int] = set()
    for m in modules:
        for b in m.buffers():
            if id(b) in seen or b.numel() == 0:
                continue
            seen.add(id(b))
            if info.backend == "nccl" and b.device.type != "cuda":
                tmp = b.to(info.device)
                dist.broadcast(tmp, src=0)
                b.copy_(tmp.to(b.device))
            else:
                dist.broadcast(b, src=0)


def allreduce_mean_grads(info: DistInfo, params: Sequence[nn.Parameter]) -> None:
    """Average the gradients of replicated parameters over ranks (parameters no wrapper synchronises).

    Every rank reduces every parameter (a missing gradient counts as zero), in one flattened buffer, so
    the collective is reached identically on every rank.
    """
    if not _active(info) or not params:
        return
    from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

    grads = []
    for p in params:
        if p.grad is None:
            p.grad = torch.zeros_like(p)
        grads.append(p.grad)
    flat = _flatten_dense_tensors(grads)
    dev = info.device if info.backend == "nccl" else torch.device("cpu")
    buf = flat.to(dev)
    dist.all_reduce(buf, op=dist.ReduceOp.SUM)
    buf = (buf / info.world).to(flat.device)
    for g, new in zip(grads, _unflatten_dense_tensors(buf, grads), strict=True):
        g.copy_(new)


def global_grad_norm(info: DistInfo | None, params: Sequence[nn.Parameter]) -> torch.Tensor:
    """sqrt(sum of squared gradients) over all parameters, sharded (DTensor) or replicated, float64 scalar.

    Sharded gradients contribute their local squares, summed over ranks; replicated gradients are the
    same on every rank and are counted once.
    """
    sharded_sq = torch.zeros((), dtype=torch.float64)
    replicated_sq = torch.zeros((), dtype=torch.float64)
    for p in params:
        g = p.grad
        if g is None:
            continue
        local = getattr(g, "to_local", None)
        if callable(local):
            sharded_sq = sharded_sq + local().detach().double().pow(2).sum().to("cpu")
        else:
            replicated_sq = replicated_sq + g.detach().double().pow(2).sum().to("cpu")
    if info is not None and _active(info) and sharded(info):
        dev = info.device if info.backend == "nccl" else torch.device("cpu")
        t = sharded_sq.to(dev)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        sharded_sq = t.to("cpu")
    return (sharded_sq + replicated_sq).sqrt()


def clip_global_norm(info: DistInfo | None, params: Sequence[nn.Parameter], max_norm: float) -> float:
    """Scale every gradient by min(1, max_norm / (norm + 1e-6)); returns the norm before clipping.

    A non-finite norm leaves the gradients unchanged (the caller skips the step).
    """
    norm = float(global_grad_norm(info, params))
    if not math.isfinite(norm):
        return norm
    coef = min(1.0, max_norm / (norm + 1e-6))
    if coef < 1.0:
        with torch.no_grad():
            for p in params:
                if p.grad is not None:
                    p.grad.mul_(coef)
    return norm


def stack_blocks(module: nn.Module) -> list[nn.Module]:
    """The blocks of every two-stream stack inside `module` (TSTCT, TAAFT, Forecaster, Advisor, Verifier)."""
    out: list[nn.Module] = []
    for m in module.modules():
        if isinstance(m, TwoStreamStack):
            out.extend(list(m.blocks))
    return out


def apply_recomputation(blocks: Sequence[nn.Module]) -> Callable[[], None]:
    """Recompute each block's activations in backward (module docstring). Returns a function that undoes it."""
    use("AS-578", by=__name__)
    from torch.utils.checkpoint import checkpoint

    undo: list[tuple[nn.Module, Any]] = []
    for blk in blocks:
        if getattr(blk, "_nagahana_recompute", False):
            continue
        original = blk.forward

        def recomputed(*args: Any, _f: Any = original, **kwargs: Any) -> Any:
            # Without gradient (truncated thinking passes, evaluation) nothing is stored: run directly.
            if not torch.is_grad_enabled():
                return _f(*args, **kwargs)
            return checkpoint(_f, *args, use_reentrant=False, **kwargs)

        functools.update_wrapper(recomputed, original)
        blk.forward = recomputed  # type: ignore[method-assign]
        blk._nagahana_recompute = True  # type: ignore[assignment]
        undo.append((blk, original))

    def restore() -> None:
        for b, f in undo:
            b.forward = f  # type: ignore[method-assign]
            b._nagahana_recompute = False  # type: ignore[assignment]

    return restore


def wrap_objective(objective: nn.Module, info: DistInfo, cfg: DistributedConfig, *,
                   shard_blocks: Sequence[nn.Module] = ()) -> nn.Module:
    """The objective wrapped for the strategy (module docstring)."""
    if info.strategy == "single":
        return objective
    if not dist.is_initialized():
        raise InvariantViolation(f"strategy {info.strategy!r} needs an initialised process group")
    if info.strategy == "ddp":
        use("AS-577", by=__name__)
        ids = [info.device.index] if info.device.type == "cuda" else None
        return nn.parallel.DistributedDataParallel(objective, device_ids=ids, find_unused_parameters=cfg.find_unused_parameters,
                                                   broadcast_buffers=True)
    if info.strategy == "fsdp":
        use("AS-577", by=__name__)
        from torch.distributed.fsdp import fully_shard, register_fsdp_forward_method

        for blk in shard_blocks:
            if not any(p.requires_grad for p in blk.parameters()):
                continue                                                   # frozen blocks stay replicated
            fully_shard(blk, mesh=info.mesh, reshard_after_forward=cfg.reshard_after_forward)
            register_fsdp_forward_method(blk, "kv")
        fully_shard(objective, mesh=info.mesh, reshard_after_forward=False)
        return objective
    raise ConfigMissing(f"unknown strategy {info.strategy!r}")


def unwrap(module: nn.Module) -> nn.Module:
    """The objective inside a DDP wrapper (FSDP shards in place, so it is the module itself)."""
    return module.module if isinstance(module, nn.parallel.DistributedDataParallel) else module


def sharded(info: DistInfo) -> bool:
    """True when parameters are sharded (FSDP)."""
    return info.strategy == "fsdp"


__all__ = ["DistInfo", "all_true", "allreduce_mean_grads", "any_true", "apply_recomputation", "clip_global_norm",
           "global_grad_norm", "barrier", "broadcast_buffers", "broadcast_object",
           "gather_objects", "init_distributed", "sharded", "shutdown", "stack_blocks", "sum_values", "unwrap",
           "weighted_means", "wrap_objective"]
