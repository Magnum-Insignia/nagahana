"""Atomic, verified, resumable checkpoints of a training stage (AS-576).

What a checkpoint holds
-----------------------
One directory per saved optimiser step, `<run>/checkpoints/stage<N>/step-<step>/`:

    global.ckpt     (written by rank 0) the full model state dict (P-18 model hash, P-19 latent space in
                    the metadata), the optimiser state keyed by parameter name, the stepper's counters
                    and loss scaler, the EMA averages, the trainer's loop counters and early-stopping
                    state, the stage's global state, the shared (budget) generator
    rank-<r>.ckpt   (written by rank r) Python, NumPy, torch CPU and CUDA RNG states, the rank's own
                    generator, the stream position, the stream carry of every lane (Environment stores,
                    contact ledgers, long-term memory, carried Imagination), the stage's per-rank state

and a pointer `stage<N>/LATEST.json` written by rank 0 after every rank has finished (a barrier), so a
directory without a pointer is an incomplete save and is never resumed from. Every file is a safe
state file (`serialization.py`): tensors only in the tensor table, JSON for everything else, SHA-256
verified before it is opened.

Full, not sharded
-----------------
Under FSDP the sharded parameters and optimiser moments are gathered to full tensors on rank 0
(`torch.distributed.checkpoint.state_dict`, full_state_dict with CPU offload) and every rank reads the
full file when it resumes (memory-mapped) and re-shards locally. The checkpoint format is therefore the
same for every strategy, and a run saved under FSDP can be resumed under DDP or in one process with
the same weights and optimiser moments. The per-rank stream state requires the same number of ranks
(the carry belongs to each rank's lanes; AS-577); a different world size is refused.

Bit-exact resumption (tests/test_training_checkpoint.py)
--------------------------------------------------------
On CPU, a run interrupted after k steps and resumed from its checkpoint reproduces the parameters and
losses of the uninterrupted run exactly: every input of the next step (weights, moments, counters,
generators, global RNG states, stream position, carry, stage state) is restored.
"""

from __future__ import annotations

import json
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from nagahana.core.errors import InvariantViolation
from nagahana.training.assumptions import use
from nagahana.training.distributed import DistInfo, barrier, sharded
from nagahana.training.serialization import atomic_write_text, load_state, save_state, verify_state

POINTER = "LATEST.json"
BEST = "best"


def optimizer_state_by_name(optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    """The optimiser's state keyed by parameter name (groups carry their "names", optim.decay_groups)."""
    sd = optimizer.state_dict()
    idx_to_name: dict[int, str] = {}
    groups = []
    for g in sd["param_groups"]:
        names = g.get("names")
        if names is None or len(names) != len(g["params"]):
            raise InvariantViolation("optimiser groups must carry parameter names (training.optim.decay_groups)")
        idx_to_name.update(zip(g["params"], names, strict=True))
        groups.append({k: v for k, v in g.items() if k not in ("params",)} | {"params": list(names)})
    return {"state": {idx_to_name[i]: s for i, s in sd["state"].items()}, "param_groups": groups}


def load_optimizer_state_by_name(optimizer: torch.optim.Optimizer, state: dict[str, Any]) -> None:
    """Inverse of `optimizer_state_by_name` (the group structure must match exactly)."""
    cur = optimizer.state_dict()
    name_to_idx: dict[str, int] = {}
    for g in cur["param_groups"]:
        name_to_idx.update(zip(g["names"], g["params"], strict=True))
    if len(cur["param_groups"]) != len(state["param_groups"]):
        raise InvariantViolation("checkpoint optimiser groups differ from the stage's")
    groups = []
    for g_cur, g_new in zip(cur["param_groups"], state["param_groups"], strict=True):
        if list(g_new["params"]) != list(g_cur["names"]):
            raise InvariantViolation("checkpoint optimiser parameters differ from the stage's (names or order)")
        groups.append({k: v for k, v in g_new.items() if k != "params"} | {"params": list(g_cur["params"])})
    missing = set(state["state"]) - set(name_to_idx)
    if missing:
        raise InvariantViolation(f"checkpoint optimiser state for unknown parameters {sorted(missing)[:5]}")
    optimizer.load_state_dict({"state": {name_to_idx[n]: s for n, s in state["state"].items()}, "param_groups": groups})


def gather_model_state(model: nn.Module, objective: nn.Module, info: DistInfo) -> dict[str, torch.Tensor] | None:
    """The full model state dict on rank 0 (None on other ranks); a collective under FSDP."""
    if not sharded(info):
        return {k: v.detach().to("cpu") for k, v in model.state_dict().items()} if info.is_main else None
    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict

    trained = get_model_state_dict(objective, options=StateDictOptions(full_state_dict=True, cpu_offload=True))
    if not info.is_main:
        return None
    out: dict[str, torch.Tensor] = {}
    for k, v in model.state_dict().items():
        src = trained.get(k)
        out[k] = (src if src is not None else v).detach().to("cpu")
    return out


def load_model_state(model: nn.Module, objective: nn.Module, info: DistInfo, state: dict[str, torch.Tensor]) -> None:
    """Load a full state dict (every rank has it) into a possibly sharded model."""
    if not sharded(info):
        model.load_state_dict(state)
        return
    from torch.distributed.checkpoint.state_dict import StateDictOptions, set_model_state_dict

    own = set(objective.state_dict().keys())
    trained = {k: v for k, v in state.items() if k in own}
    rest = {k: v for k, v in state.items() if k not in own}
    set_model_state_dict(objective, trained, options=StateDictOptions(full_state_dict=True))
    missing, unexpected = model.load_state_dict(rest, strict=False)
    still = [k for k in missing if k not in own]
    if still or unexpected:
        raise InvariantViolation(f"model state mismatch: missing {still[:5]}, unexpected {list(unexpected)[:5]}")


def gather_optimizer_state(optimizer: torch.optim.Optimizer, objective: nn.Module, info: DistInfo) -> dict[str, Any] | None:
    """The full optimiser state keyed by parameter name on rank 0 (None elsewhere); a collective under FSDP."""
    if not sharded(info):
        return optimizer_state_by_name(optimizer) if info.is_main else None
    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_optimizer_state_dict

    osd = get_optimizer_state_dict(objective, optimizer, options=StateDictOptions(full_state_dict=True, cpu_offload=True))
    return dict(osd) if info.is_main else None


def load_optimizer_state(optimizer: torch.optim.Optimizer, objective: nn.Module, info: DistInfo, state: dict[str, Any]) -> None:
    """Load a full optimiser state keyed by name (every rank has it).

    The loaded moments are copied into memory the process owns: a memory-mapped checkpoint file must
    not stay referenced by the optimiser (the file can then be pruned, also on Windows).
    """
    if not sharded(info):
        load_optimizer_state_by_name(optimizer, state)
    else:
        from torch.distributed.checkpoint.state_dict import StateDictOptions, set_optimizer_state_dict

        set_optimizer_state_dict(objective, optimizer, state, options=StateDictOptions(full_state_dict=True))
    for st in optimizer.state.values():
        for k, v in list(st.items()):
            if isinstance(v, torch.Tensor):
                st[k] = v.clone()


@dataclass(frozen=True)
class Pointer:
    """What LATEST.json says: the newest complete checkpoint of a stage."""

    stage: int
    step: int
    directory: str
    world: int
    digests: dict[str, str]
    created: float
    meta: dict[str, Any]


class CheckpointManager:
    """Saves and finds the resumable checkpoints of one stage (module docstring)."""

    def __init__(self, root: str | Path, *, stage: int, info: DistInfo, keep_last: int) -> None:
        use("AS-576", by=__name__)
        if keep_last < 1:
            raise ValueError("keep_last must be >= 1")
        self.dir = Path(root) / f"stage{stage}"
        self.stage = stage
        self.info = info
        self.keep_last = keep_last

    def _step_dir(self, step: int) -> Path:
        return self.dir / f"step-{step:09d}"

    def save(self, *, step: int, global_state: Callable[[], dict[str, Any] | None], rank_state: dict[str, Any],
             meta: dict[str, Any]) -> Path:
        """Write one checkpoint. `global_state` is called on every rank (it may gather) and returns the
        state on rank 0 (None elsewhere)."""
        d = self._step_dir(step)
        g = global_state()                                       # collective under FSDP
        if self.info.is_main:
            if g is None:
                raise InvariantViolation("rank 0 has no global state to save")
            d.mkdir(parents=True, exist_ok=True)
            save_state(d / "global.ckpt", g, kind="nagahana-train-global",
                       meta=dict(meta) | {"stage": self.stage, "step": step, "world": self.info.world})
        barrier(self.info)                                       # the directory exists before ranks write into it
        d.mkdir(parents=True, exist_ok=True)
        save_state(d / f"rank-{self.info.rank}.ckpt", rank_state, kind="nagahana-train-rank",
                   meta={"stage": self.stage, "step": step, "rank": self.info.rank, "world": self.info.world})
        barrier(self.info)                                       # every rank's file is complete
        if self.info.is_main:
            digests = {"global.ckpt": verify_state(d / "global.ckpt")}
            for r in range(self.info.world):
                digests[f"rank-{r}.ckpt"] = verify_state(d / f"rank-{r}.ckpt")
            ptr = {"stage": self.stage, "step": step, "directory": d.name, "world": self.info.world, "digests": digests,
                   "created": time.time(), "meta": dict(meta)}
            atomic_write_text(self.dir / POINTER, json.dumps(ptr, indent=1, sort_keys=True, default=str))
            self._prune(keep=d.name)
        barrier(self.info)
        return d

    def _prune(self, *, keep: str) -> None:
        # Keep the newest `keep_last` complete step directories and the best one; remove the rest.
        steps = sorted((p for p in self.dir.glob("step-*") if p.is_dir()), key=lambda p: p.name)
        complete = [p for p in steps if (p / "global.ckpt").is_file()]
        protected = {keep, *(p.name for p in complete[-self.keep_last:])}
        for p in steps:
            if p.name not in protected:
                shutil.rmtree(p, ignore_errors=True)

    def latest(self) -> Pointer | None:
        """The newest complete checkpoint, verified file by file (None when the stage has none)."""
        path = self.dir / POINTER
        if not path.is_file():
            return None
        raw = json.loads(path.read_text(encoding="utf-8"))
        ptr = Pointer(stage=int(raw["stage"]), step=int(raw["step"]), directory=str(raw["directory"]), world=int(raw["world"]),
                      digests=dict(raw["digests"]), created=float(raw["created"]), meta=dict(raw.get("meta", {})))
        if ptr.stage != self.stage:
            raise InvariantViolation(f"{path}: pointer of stage {ptr.stage} in the stage-{self.stage} directory")
        d = self.dir / ptr.directory
        for name, digest in ptr.digests.items():
            if verify_state(d / name) != digest:
                raise InvariantViolation(f"{d / name}: digest differs from the pointer's (file replaced after the save)")
        return ptr

    def load(self, ptr: Pointer) -> tuple[dict[str, Any], dict[str, Any]]:
        """(global state, this rank's state) of a checkpoint; the world size must match."""
        if ptr.world != self.info.world:
            raise InvariantViolation(f"checkpoint was written by {ptr.world} ranks; this run has {self.info.world} "
                                     "(the per-rank stream carry cannot be re-partitioned, AS-577)")
        d = self.dir / ptr.directory
        g, _ = load_state(d / "global.ckpt", kind="nagahana-train-global", mmap=True)
        r, rmeta = load_state(d / f"rank-{self.info.rank}.ckpt", kind="nagahana-train-rank")
        if int(rmeta["rank"]) != self.info.rank:
            raise InvariantViolation("rank file belongs to another rank")
        return g, r

    def save_best(self, state: Callable[[], dict[str, Any] | None], *, meta: dict[str, Any]) -> Path | None:
        """Write the best weights so far (`best/model.ckpt`, rank 0). Collective under FSDP."""
        s = state()
        out = self.dir / BEST / "model.ckpt"
        if self.info.is_main:
            if s is None:
                raise InvariantViolation("rank 0 has no state to save as best")
            save_state(out, s, kind="nagahana-train-best", meta=dict(meta) | {"stage": self.stage})
        barrier(self.info)
        return out if self.info.is_main else None

    def load_best(self) -> tuple[dict[str, Any], dict[str, Any]] | None:
        """(state, meta) of the best weights, or None when none were saved."""
        p = self.dir / BEST / "model.ckpt"
        if not p.is_file():
            return None
        return load_state(p, kind="nagahana-train-best", mmap=True)


__all__ = ["CheckpointManager", "Pointer", "gather_model_state", "gather_optimizer_state", "load_model_state",
           "load_optimizer_state", "load_optimizer_state_by_name", "optimizer_state_by_name"]
