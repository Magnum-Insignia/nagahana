"""Seeds, generators and RNG state for reproducible and exactly resumable training (AS-588).

Purpose
-------
A training run must give the same numbers when it is run again with the same seed, and a resumed run
must continue exactly where the interrupted one stopped. Both need every source of randomness to be
seeded from the run seed and every random state to be saved in checkpoints:

- Python's `random`, NumPy's global generator, torch's CPU generator and every CUDA device's generator
  (global states: libraries and model code may draw from them);
- the trainers' explicit `torch.Generator` objects (all random choices of a stage go through them).

Seeds (AS-588)
--------------
`derive_seed(seed, *keys)` maps a base seed and a path of keys (strings or integers, for example
("stage3", "rank", 2)) to a 63-bit seed with NumPy's SeedSequence (O'Neill's PCG seeding; NumPy
documentation "Parallel Random Number Generation"): streams derived from different keys are
statistically independent, and a key path always gives the same seed. Each rank draws its data
augmentation from its own stream; the loop budgets R and S come from a stream shared by all ranks
(AS-579), so distributed collectives line up.

Where draws are made
--------------------
Generators are created on the CPU and draws are moved to the device of the tensor they serve. A CPU
stream gives bit-identical draws on every accelerator, so a run is reproducible across hardware, and
CPU draws are cheap next to the model's compute. Model code that samples directly on a tensor's device
must draw on the generator's device and move the draw (see the requested changes of the build report).

Determinism of kernels: `configure_determinism(True)` selects deterministic algorithms
(`torch.use_deterministic_algorithms`) and disables cuDNN autotuning; on CUDA the cuBLAS workspace
variable must be set before the first CUDA call (`CUBLAS_WORKSPACE_CONFIG=:4096:8`, PyTorch
"Reproducibility" notes), which this function sets when CUDA has not been initialised yet.
"""

from __future__ import annotations

import hashlib
import os
import random
from typing import Any

import numpy as np
import torch

from nagahana.training.assumptions import use

_MASK63 = (1 << 63) - 1


def _key_words(key: int | str) -> list[int]:
    """32-bit words of one key: integers as they are, strings by SHA-256 (stable across processes)."""
    if isinstance(key, bool):
        raise TypeError("seed keys are integers or strings, not booleans")
    if isinstance(key, int):
        if key < 0:
            raise ValueError("integer seed keys must be >= 0")
        words = []
        while True:
            words.append(key & 0xFFFFFFFF)
            key >>= 32
            if key == 0:
                return words
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return [int.from_bytes(digest[i:i + 4], "little") for i in range(0, 32, 4)]


def derive_seed(seed: int, *keys: int | str) -> int:
    """A 63-bit seed for the stream named by `keys` under the base `seed` (module docstring)."""
    if seed < 0:
        raise ValueError("the base seed must be >= 0")
    entropy = _key_words(int(seed))
    spawn = [w for k in keys for w in (len(_key_words(k)), *_key_words(k))]   # length-prefixed: no ambiguity
    ss = np.random.SeedSequence(entropy=entropy, spawn_key=tuple(spawn))
    state = ss.generate_state(2, dtype=np.uint32)
    return int((int(state[0]) | (int(state[1]) << 32)) & _MASK63)


def make_generator(seed: int, device: torch.device | str = "cpu") -> torch.Generator:
    """A torch.Generator on `device`, seeded explicitly."""
    g = torch.Generator(device=torch.device(device))
    g.manual_seed(int(seed) & _MASK63)
    return g


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and torch (CPU and every CUDA device) from one seed (AS-588)."""
    use("AS-588", by=__name__)
    random.seed(derive_seed(seed, "python"))
    np.random.seed(derive_seed(seed, "numpy") % (2**32))
    torch.manual_seed(derive_seed(seed, "torch"))          # also seeds every CUDA device


def configure_determinism(enabled: bool, *, warn_only: bool = False) -> dict[str, Any]:
    """Select deterministic kernels (module docstring). Returns what was set, for the run manifest."""
    out: dict[str, Any] = {"deterministic": bool(enabled)}
    if enabled:
        cuda_ready = torch.cuda.is_available() and torch.cuda.is_initialized()
        if not cuda_ready and "CUBLAS_WORKSPACE_CONFIG" not in os.environ:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        out["cublas_workspace_config"] = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        out["cublas_set_before_cuda_init"] = not cuda_ready
    torch.use_deterministic_algorithms(bool(enabled), warn_only=warn_only)
    torch.backends.cudnn.deterministic = bool(enabled)
    torch.backends.cudnn.benchmark = not enabled
    return out


def capture_rng_state() -> dict[str, Any]:
    """Every global random state of this process (Python, NumPy, torch CPU, each CUDA device)."""
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(legacy=True),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": [],
    }
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        state["torch_cuda"] = [torch.cuda.get_rng_state(i) for i in range(torch.cuda.device_count())]
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    """Restore what `capture_rng_state` returned (the CUDA part only when this process has the devices)."""
    py = state["python"]
    # The codec returns tuples for tuples, so the Python state is restored exactly.
    random.setstate((py[0], tuple(py[1]), py[2]))
    np_state = state["numpy"]
    np.random.set_state((str(np_state[0]), np.asarray(np_state[1], dtype=np.uint32), int(np_state[2]),
                         int(np_state[3]), float(np_state[4])))
    torch.set_rng_state(state["torch_cpu"])
    cuda = state.get("torch_cuda") or []
    if cuda:
        if not torch.cuda.is_available() or torch.cuda.device_count() < len(cuda):
            raise RuntimeError(f"the checkpoint holds {len(cuda)} CUDA generator states; this process has "
                               f"{torch.cuda.device_count() if torch.cuda.is_available() else 0} devices")
        for i, s in enumerate(cuda):
            torch.cuda.set_rng_state(s, i)


def generator_state(g: torch.Generator) -> torch.Tensor:
    """The state of one generator (a uint8 tensor)."""
    return g.get_state()


def set_generator_state(g: torch.Generator, state: torch.Tensor) -> None:
    """Restore one generator's state."""
    g.set_state(state.to(torch.uint8))


__all__ = ["capture_rng_state", "configure_determinism", "derive_seed", "generator_state", "make_generator",
           "restore_rng_state", "seed_everything", "set_generator_state"]
