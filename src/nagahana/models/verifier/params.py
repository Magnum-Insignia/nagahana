"""Candidate deltas kept apart from the reference weights; functional evaluation; exact apply and restore (AS-835, AS-844).

A candidate update never writes into the deployed weights while it is fitted or evaluated. It is a set
of tensors Delta over named parameters of the target module, and the candidate policy is the module
evaluated at theta_ref + Delta through `torch.func.functional_call`, which substitutes the named
parameters for one call and restores the module's own afterwards. The reference policy pi_ref is the
module itself, so DPO's reference log-probabilities and the KL anchor need no copy of the model, and the
fit can prove that the reference is untouched (its state hash is unchanged).

Parameterisations of Delta (AdapterConfig)

    dense      Delta_n = D_n, a free tensor of the parameter's shape, initialised to 0
    low rank   Delta_n = (alpha / r) B_n A_n for two-dimensional weights W_n in R^(out x in), with
               A_n in R^(r x in) Kaiming-uniform and B_n = 0 in R^(out x r) (Hu et al., "LoRA",
               ICLR 2022, arXiv:2106.09685); other selected parameters stay dense

Both start at Delta = 0, so the candidate equals the reference before the first step (the DPO loss starts
at log 2 for hard labels and every GRPO ratio starts at 1).

Promotion and rollback (AS-844). `apply_deltas_` replaces each touched parameter by fl(theta + Delta)
(float32 addition) under `torch.no_grad`, after storing an exact clone of the old tensor; `restore_`
copies the stored clones back. Floating-point addition is not invertible (theta + Delta - Delta need not
equal theta), so rollback never subtracts: it restores the stored bits, and `state_hash` before the
promotion equals `state_hash` after the rollback (tested).

Digests. `tensor_digest` is SHA-256 over (name, dtype, shape, raw bytes) of every tensor in name order,
exactly the algorithm of `models.nagahana.model_hash` (P-18), so `state_hash(module) == model_hash(module)`
(tested); it identifies a state bit for bit.

Files. `save_tensors` writes a mapping of tensors plus a JSON metadata record with `torch.save` to a
temporary file, fsyncs it and renames it into place (atomic on POSIX and Windows); `load_tensors` reads
with `weights_only=True` (no pickled code is executed) and checks the SHA-256 of the file against the
digest the ledger recorded before anything is decoded.
"""

from __future__ import annotations

import fnmatch
import hashlib
import math
import os
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, TypeVar

import torch
from torch import nn

from nagahana.core.errors import InvariantViolation
from nagahana.models.verifier.canonical import canonical_json, from_canonical, sha256_hex

T = TypeVar("T")
_META_KEY = "__meta__"


def tensor_digest(tensors: Mapping[str, torch.Tensor]) -> str:
    """SHA-256 over (name, dtype, shape, bytes) of every tensor in name order (module docstring)."""
    h = hashlib.sha256()
    for name, t in sorted(tensors.items()):
        x = t.detach().cpu().contiguous()
        h.update(name.encode())
        h.update(str(x.dtype).encode())
        h.update(str(tuple(x.shape)).encode())
        if x.numel():
            h.update(x.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def state_hash(module: nn.Module) -> str:
    """The module's state digest: identical to `models.nagahana.model_hash` (P-18)."""
    return tensor_digest(module.state_dict())


def select_parameters(module: nn.Module, patterns: Sequence[str]) -> list[tuple[str, nn.Parameter]]:
    """Parameters whose dotted name matches any fnmatch pattern, in registration order (none matched raises)."""
    out = [(n, p) for n, p in module.named_parameters() if any(fnmatch.fnmatchcase(n, pat) for pat in patterns)]
    if not out:
        raise InvariantViolation(f"no parameter of {type(module).__name__} matches {list(patterns)}")
    return out


class ParameterDelta(nn.Module):
    """A trainable delta over selected parameters of a reference module (module docstring).

    The reference module is not registered as a submodule: the delta's parameters are only its own,
    so an optimiser built on `self.parameters()` can never reach the reference weights.
    """

    def __init__(self, module: nn.Module, patterns: Sequence[str], *, rank: int = 0, alpha: float = 1.0) -> None:
        super().__init__()
        if rank < 0:
            raise ValueError("rank must be >= 0")
        self.rank, self.alpha = int(rank), float(alpha)
        self.reference_hash = state_hash(module)
        self.names: tuple[str, ...] = ()
        self._shapes: dict[str, tuple[int, ...]] = {}
        self._kinds: dict[str, str] = {}
        self.dense = nn.ParameterDict()
        self.factor_a = nn.ParameterDict()
        self.factor_b = nn.ParameterDict()
        names = []
        for name, p in select_parameters(module, patterns):
            key = _key(name)
            self._shapes[name] = tuple(p.shape)
            if self.rank > 0 and p.ndim == 2:
                # B A with B = 0: the product starts at exactly zero; A carries the random directions.
                a = torch.empty(self.rank, p.shape[1], dtype=p.dtype, device=p.device)
                nn.init.kaiming_uniform_(a, a=math.sqrt(5))
                self.factor_a[key] = nn.Parameter(a)
                self.factor_b[key] = nn.Parameter(torch.zeros(p.shape[0], self.rank, dtype=p.dtype, device=p.device))
                self._kinds[name] = "low_rank"
            else:
                self.dense[key] = nn.Parameter(torch.zeros_like(p, memory_format=torch.contiguous_format))
                self._kinds[name] = "dense"
            names.append(name)
        self.names = tuple(names)

    @property
    def scale(self) -> float:
        """alpha / r of the low-rank deltas (1 when every delta is dense)."""
        return self.alpha / self.rank if self.rank > 0 else 1.0

    def deltas(self) -> dict[str, torch.Tensor]:
        """Delta per parameter name, differentiable in the delta's own parameters."""
        out: dict[str, torch.Tensor] = {}
        for name in self.names:
            key = _key(name)
            if self._kinds[name] == "low_rank":
                out[name] = self.scale * (self.factor_b[key] @ self.factor_a[key])     # [out, in]
            else:
                out[name] = self.dense[key]
        return out

    def overrides(self, module: nn.Module) -> dict[str, torch.Tensor]:
        """theta_ref + Delta for the touched names; theta_ref is read through a stop-gradient."""
        params = dict(module.named_parameters())
        return {n: params[n].detach() + d for n, d in self.deltas().items()}

    def frozen(self) -> dict[str, torch.Tensor]:
        """The materialised deltas, detached, contiguous and in the weights' dtype (the candidate payload)."""
        with torch.no_grad():
            return {n: d.detach().clone().contiguous() for n, d in self.deltas().items()}

    @torch.no_grad()
    def sq_norm(self) -> torch.Tensor:
        """sum_n ||Delta_n||_F^2 (float64, without autograd history: a reported quantity)."""
        total = torch.zeros((), dtype=torch.float64)
        for d in self.deltas().values():
            total = total + (d.double() ** 2).sum().to(total.device)
        return total

    def parameters_of(self, names: Sequence[str]) -> list[nn.Parameter]:
        """The delta's own parameters that parameterise the given target names (for per-part optimisers)."""
        out: list[nn.Parameter] = []
        for name in names:
            if name not in self._kinds:
                raise KeyError(f"{name!r} is not a parameter of this delta")
            key = _key(name)
            if self._kinds[name] == "low_rank":
                out += [self.factor_a[key], self.factor_b[key]]
            else:
                out.append(self.dense[key])
        return out


def _key(name: str) -> str:
    # ParameterDict keys may not contain "."; the mapping is injective because names never contain "/".
    return name.replace(".", "/")


def overrides_from(module: nn.Module, deltas: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """theta + Delta for frozen deltas (evaluation of a stored candidate), shapes and dtypes checked."""
    params = dict(module.named_parameters())
    out: dict[str, torch.Tensor] = {}
    for name, d in deltas.items():
        if name not in params:
            raise InvariantViolation(f"candidate delta names {name!r}, which {type(module).__name__} does not have")
        p = params[name]
        if tuple(d.shape) != tuple(p.shape) or d.dtype != p.dtype:
            raise InvariantViolation(f"delta {name!r} has shape {tuple(d.shape)} / {d.dtype}, the weight {tuple(p.shape)} / {p.dtype}")
        out[name] = p.detach() + d.to(p.device)
    return out


class _MethodCall(nn.Module):
    """Wraps a module so that `functional_call` can substitute its parameters for any method, not only forward."""

    def __init__(self, inner: nn.Module, fn: Callable[..., Any]) -> None:
        super().__init__()
        self.inner = inner
        self._fn = fn

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self._fn(self.inner, *args, **kwargs)


def call_with(module: nn.Module, overrides: Mapping[str, torch.Tensor] | None, fn: Callable[..., T], /,
              *args: Any, **kwargs: Any) -> T:
    """fn(module, *args, **kwargs) with the named parameters replaced by `overrides` for this call only.

    None runs `fn` on the module's own parameters. The module's parameters are restored when the call
    returns or raises (torch.func.functional_call semantics), so the reference weights never change.
    """
    if not overrides:
        return fn(module, *args, **kwargs)
    wrapper = _MethodCall(module, fn)
    named = {f"inner.{k}": v for k, v in overrides.items()}
    out: T = torch.func.functional_call(wrapper, named, args, kwargs, strict=False, tie_weights=True)
    return out


def apply_deltas_(module: nn.Module, deltas: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """theta <- fl(theta + Delta) in place for each touched name; returns exact clones of the replaced tensors."""
    params = dict(module.named_parameters())
    for name, d in deltas.items():
        if name not in params:
            raise InvariantViolation(f"candidate delta names {name!r}, which {type(module).__name__} does not have")
        if tuple(d.shape) != tuple(params[name].shape) or d.dtype != params[name].dtype:
            raise InvariantViolation(f"delta {name!r} does not match the weight's shape and dtype")
    snapshot = {name: params[name].detach().clone() for name in deltas}
    with torch.no_grad():
        for name, d in deltas.items():
            params[name].add_(d.to(params[name].device))
    return snapshot


def restore_(module: nn.Module, snapshot: Mapping[str, torch.Tensor]) -> None:
    """Copy stored tensors back bit for bit (the inverse of `apply_deltas_`)."""
    params = dict(module.named_parameters())
    for name, old in snapshot.items():
        if name not in params or tuple(old.shape) != tuple(params[name].shape) or old.dtype != params[name].dtype:
            raise InvariantViolation(f"snapshot tensor {name!r} does not fit the module")
    with torch.no_grad():
        for name, old in snapshot.items():
            params[name].copy_(old.to(params[name].device))


def _fsync_dir(directory: Path) -> None:
    # A rename is durable once its directory entry is (POSIX; Windows commits it with the file).
    if os.name != "posix":
        return
    fd = os.open(str(directory), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def save_tensors(path: str | Path, tensors: Mapping[str, torch.Tensor], meta: Mapping[str, Any]) -> str:
    """Write tensors + canonical-JSON metadata atomically (module docstring). Returns the file's SHA-256."""
    p = Path(path)
    if _META_KEY in tensors:
        raise InvariantViolation(f"{_META_KEY!r} is reserved for the metadata")
    payload: dict[str, torch.Tensor] = {k: v.detach().cpu().contiguous() for k, v in tensors.items()}
    text = canonical_json(dict(meta)).encode("ascii")
    payload[_META_KEY] = torch.frombuffer(bytearray(text), dtype=torch.uint8).clone()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    with tmp.open("wb") as fh:
        torch.save(payload, fh)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, p)
    _fsync_dir(p.parent)
    return sha256_hex(p.read_bytes())


def load_tensors(path: str | Path, *, expected_sha256: str | None) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Read a file written by `save_tensors`; the digest is checked before decoding (None skips the check)."""
    p = Path(path)
    if not p.is_file():
        raise InvariantViolation(f"{p}: no such file")
    raw = p.read_bytes()
    if expected_sha256 is not None and sha256_hex(raw) != expected_sha256:
        raise InvariantViolation(f"{p}: SHA-256 differs from the ledger's record (corrupt or altered file); not loaded")
    payload = torch.load(str(p), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or _META_KEY not in payload:
        raise InvariantViolation(f"{p}: not a Verifier tensor file")
    meta_raw = payload.pop(_META_KEY)
    meta = from_canonical(bytes(meta_raw.numpy().tobytes()).decode("ascii"))
    if not isinstance(meta, dict):
        raise InvariantViolation(f"{p}: metadata must be a mapping")
    return {str(k): v for k, v in payload.items()}, meta


def parameters_of(modules: Iterable[nn.Module]) -> list[nn.Parameter]:
    """Distinct parameters of several modules (an optimiser over shared parameters must see each once)."""
    seen: set[int] = set()
    out: list[nn.Parameter] = []
    for m in modules:
        for p in m.parameters():
            if id(p) not in seen:
                seen.add(id(p))
                out.append(p)
    return out


__all__ = ["ParameterDelta", "apply_deltas_", "call_with", "load_tensors", "overrides_from", "parameters_of", "restore_",
           "save_tensors", "select_parameters", "state_hash", "tensor_digest"]
