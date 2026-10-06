"""Muon for NagaHana's hidden matrices, exact under sharding, and the rule that assigns every parameter.

Update (torch.optim.Muon's, unchanged; AS-599)
----------------------------------------------
For a hidden weight matrix W (A x B) with gradient G, momentum buffer M, learning rate lr and decoupled
weight decay wd (Jordan et al. 2024, "Muon: an optimizer for hidden layers in neural networks";
Liu et al., "Muon is Scalable for LLM Training", arXiv:2502.16982):

    M   <- M + (1 - mu) (G - M)                      momentum (mu = 0.95), stored like torch: lerp
    U   =  G + mu (M - G)        if Nesterov, else M
    O   =  NS_5(U)                                   quintic Newton-Schulz, coefficients (3.4445, -4.7750, 2.0315),
                                                     on U / max(||U||_F, eps) in bfloat16, iterated on the wide
                                                     orientation (U^T when A > B)
    W   <- W (1 - lr wd)                             decoupled weight decay (shared with AdamW)
    W   <- W - lr * 0.2 sqrt(max(A, B)) * O          Moonlight's scale ("match_rms_adamw"): the update has the
                                                     RMS of an AdamW update, so Muon and AdamW share lr and wd

`MatrixMuon.step` computes torch.optim.Muon's update with the same formulas in the same order (torch's
two-branch lerp included) and extends it in three ways:

1. Layout-independent element-wise arithmetic. torch's CPU kernels for `lerp` and `add(alpha=...)`
   fuse a multiply and an add in their vector lanes but not in the scalar tail, so the rounding of an
   element depends on where it sits in the tensor (measured: 94 of 200 random row splits changed some
   element of a lerp; separate `sub`, `mul` and `add` calls changed none). The element-wise steps are
   therefore written as separate operations; the result equals torch.optim.Muon's to the last bits
   (tests/test_training_optim.py: relative difference below 1e-6) and does not depend on the layout.
2. Sharded matrices (FSDP, DTensor parameters). Newton-Schulz must see the full matrix. Following the
   Distributed Muon scheme of Liu et al. (section 2.3): the momentum update is element-wise, so it runs
   on the local shard; the Nesterov update U is all-gathered to the full matrix (`full_tensor`), every
   rank computes NS_5 on the full matrix, and each rank keeps the rows of its own shard. Memory: one
   full matrix at a time; communication: one all-gather of U per matrix per step. With (1), the gathered
   U equals the unsharded U bit for bit and the same NS operations follow, so sharded and unsharded
   updates are bit-identical (tests/test_training_distributed.py, two Gloo ranks).
3. Stacked matrices. A parameter of shape [K, A, B] that stores K independent matrices (CVG-AE's typed
   key and value maps, one per hyperedge kind, AS-03) is orthogonalised slice by slice, each slice with
   its own scale 0.2 sqrt(max(A, B)). torch.optim.Muon accepts only 2-D tensors.

The assignment rule (AS-599)
----------------------------
Every trainable parameter gets exactly one group (`assign_groups`, `GROUPS`):

    muon/attention   weights of the Q, K, V, O projections of every attention module
    muon/mlp         weights of the hidden MLP layers (SwiGLU gate, up, and down when it maps back to the
                     model width; the typed MLPs of CVG-AE; hidden layers of small MLPs)
    muon/hidden      every other linear map between hidden representations (stream inputs from latents,
                     context projections, pointer queries and keys, gate projections, memory maps)
    muon/stacked     stacked [K, A, B] hidden matrices (CVG-AE w_k, w_v)
    adamw/embedding  embedding tables (nn.Embedding) and raw embedding-like tables (field bits, periodic maps)
    adamw/norm       normalisation gains (RMSNorm.weight)
    adamw/bias       biases of linear layers
    adamw/scalar     scalar parameters (learned temperatures, step sizes, energy scales, gate offsets)
    adamw/vector     other one-dimensional parameters (learned queries, null contexts, per-head offsets)
    adamw/table      raw multi-dimensional parameters that are not linear maps (bias tables, null keys and
                     values, slot embeddings, per-head attention vectors, initial fast weights, couplings)
    adamw/input      linear maps from raw features (scalars, time encodings, probability readouts, clock
                     and monitor features) into a model width: embedding-like
    adamw/output     linear maps that produce model outputs: readouts and heads (policy, value, hazard,
                     stage, reward, technique, latent), posterior and prior parameters, decoder outputs,
                     energy readouts and lens couplings
    adamw/thin       linear maps with a side of at most `thin_max` (vector-like: Newton-Schulz of a
                     1 x n matrix only normalises it)

Muon receives only parameters of rank 2 (or listed stacked rank-3 matrices); `assign_groups` raises if
a rank-1 parameter would reach it.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn

from nagahana.core.errors import InvariantViolation
from nagahana.nn.attention import MultiHeadAttention
from nagahana.nn.mlp import SwiGLU
from nagahana.nn.norms import RMSNorm
from nagahana.training.assumptions import use

NS_COEFFICIENTS: tuple[float, float, float] = (3.4445, -4.7750, 2.0315)
NS_EPS = 1e-7

GROUPS: tuple[str, ...] = (
    "muon/attention", "muon/mlp", "muon/hidden", "muon/stacked",
    "adamw/embedding", "adamw/norm", "adamw/bias", "adamw/scalar", "adamw/vector", "adamw/table",
    "adamw/input", "adamw/output", "adamw/thin",
)


def newton_schulz(g: torch.Tensor, *, coefficients: tuple[float, float, float] = NS_COEFFICIENTS,
                  steps: int = 5, eps: float = NS_EPS, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    """Quintic Newton-Schulz orthogonalisation of a 2-D matrix, as torch.optim.Muon computes it.

    The operations and their order are torch's (`torch/optim/_muon.py`, `_zeropower_via_newtonschulz`):
    cast, transpose to the wide orientation, Frobenius normalisation clamped at eps, `steps` iterations of
    X <- a X + (b A + c A A) X with A = X X^T, transpose back.
    """
    if g.dim() != 2:
        raise InvariantViolation(f"Newton-Schulz needs a 2-D matrix, got shape {tuple(g.shape)}")
    if not 0 < steps < 100:
        raise ValueError("Newton-Schulz steps must lie in [1, 99]")
    a, b, c = coefficients
    x = g.to(dtype)
    tall = g.size(0) > g.size(1)
    if tall:
        x = x.T
    x.div_(x.norm().clamp(min=eps))
    for _ in range(steps):
        gram = x @ x.T
        update = torch.addmm(gram, gram, gram, beta=b, alpha=c)
        x = torch.addmm(x, update, x, beta=a)
    if tall:
        x = x.T
    return x


def adjusted_lr(lr: float, shape: Sequence[int], mode: str) -> float:
    """The learning rate of one matrix of shape (A, B): Moonlight's 0.2 sqrt(max(A, B)) or Jordan's sqrt(max(1, A/B))."""
    a, b = int(shape[0]), int(shape[1])
    if mode == "match_rms_adamw":
        return lr * 0.2 * math.sqrt(max(a, b))
    if mode == "original":
        return lr * math.sqrt(max(1.0, a / b))
    raise ValueError(f"unknown Muon lr adjustment {mode!r}")


def _lerp(a: torch.Tensor, b: torch.Tensor, w: float) -> torch.Tensor:
    """torch.lerp(a, b, w) with torch's two-branch formula, as separate (layout-independent) operations.

    w < 0.5: a + (b - a) * w;   w >= 0.5: b - (b - a) * (1 - w)   (aten's lerp, chosen for stability).
    """
    diff = b - a
    if w < 0.5:
        return a + diff * w
    return b - diff * (1.0 - w)


def _is_dtensor(t: torch.Tensor) -> bool:
    return hasattr(t, "device_mesh") and hasattr(t, "to_local")


def _local_rows_of(full: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    """The local shard of `full` with the placement of the distributed tensor `like` (no communication)."""
    from torch.distributed.tensor import distribute_tensor

    return distribute_tensor(full, like.device_mesh, like.placements, src_data_rank=None).to_local()  # type: ignore[attr-defined]


class MatrixMuon(torch.optim.Optimizer):
    """torch.optim.Muon's update for 2-D and stacked hidden matrices, exact under sharding (module docstring).

    Hyperparameters and defaults are torch.optim.Muon's. `ns_dtype`: the Newton-Schulz precision
    (bfloat16 as torch; float32 for audits). Parameters must be rank 2, or rank 3 when they are
    stacked matrices (orthogonalised per slice).
    """

    def __init__(self, params: Any, *, lr: float = 1e-3, weight_decay: float = 0.1, momentum: float = 0.95,
                 nesterov: bool = True, ns_coefficients: tuple[float, float, float] = NS_COEFFICIENTS,
                 eps: float = NS_EPS, ns_steps: int = 5, adjust_lr_fn: str = "match_rms_adamw",
                 ns_dtype: torch.dtype = torch.bfloat16) -> None:
        if lr < 0 or momentum < 0 or weight_decay < 0:
            raise ValueError("lr, momentum and weight decay must be >= 0")
        if adjust_lr_fn not in ("original", "match_rms_adamw"):
            raise ValueError(f"unsupported Muon lr adjustment {adjust_lr_fn!r}")
        defaults = {"lr": lr, "weight_decay": weight_decay, "momentum": momentum, "nesterov": nesterov,
                    "ns_coefficients": tuple(ns_coefficients), "eps": eps, "ns_steps": ns_steps,
                    "adjust_lr_fn": adjust_lr_fn}
        super().__init__(params, defaults)
        self.ns_dtype = ns_dtype
        for group in self.param_groups:
            for p in group["params"]:
                if p.dim() not in (2, 3):
                    raise InvariantViolation(f"Muon optimises matrices only; got a parameter of shape {tuple(p.shape)}")
                if torch.is_complex(p):
                    raise InvariantViolation("Muon does not support complex parameters")

    def _orthogonalise(self, u: torch.Tensor, group: dict[str, Any]) -> torch.Tensor:
        # 2-D: one Newton-Schulz; 3-D: one per stacked slice (each slice an independent matrix).
        kw = {"coefficients": group["ns_coefficients"], "steps": group["ns_steps"], "eps": group["eps"],
              "dtype": self.ns_dtype}
        if u.dim() == 2:
            return newton_schulz(u, **kw)
        return torch.stack([newton_schulz(u[k], **kw) for k in range(u.shape[0])])

    @torch.no_grad()
    def step(self, closure: Any = None) -> Any:
        """One Muon step over every parameter with a gradient."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr, wd, mu = float(group["lr"]), float(group["weight_decay"]), float(group["momentum"])
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise InvariantViolation("Muon does not support sparse gradients")
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(grad, memory_format=torch.preserve_format)
                buf = state["momentum_buffer"]
                buf.copy_(_lerp(buf, grad, 1 - mu))                        # M <- lerp(M, G, 1 - mu), on the shard
                update = _lerp(grad, buf, mu) if group["nesterov"] else buf
                if _is_dtensor(update):
                    # Distributed Muon: gather the full matrix, orthogonalise it, keep the local rows.
                    full = update.full_tensor()                            # type: ignore[attr-defined]
                    ortho_full = self._orthogonalise(full, group)
                    ortho = _local_rows_of(ortho_full, update)
                    local = p.to_local()                                   # type: ignore[attr-defined]
                else:
                    ortho = self._orthogonalise(update, group)
                    local = p
                shape = p.shape[-2:]
                local.mul_(1 - lr * wd)                                    # decoupled weight decay
                local.sub_(ortho.to(local.dtype) * adjusted_lr(lr, shape, group["adjust_lr_fn"]))
        return loss


@dataclass(frozen=True)
class GroupRule:
    """Patterns of the assignment rule (module docstring); names are full parameter names of the model."""

    output_patterns: tuple[str, ...] = (
        r"(^|\.)(\w+_head|head)\.(weight|bias)$",              # policy, value, hazard, stage, reward, decoder heads
        r"(^|\.)head_\w+\.(weight|bias)$",                     # variational head (mean, logvar, logits)
        r"^taaft\.readouts\.",                                 # every TAAFT readout
        r"^taaft\.w_y\.",                                      # amortised hypothesis y0 = W_y c
        r"\.summary_pred\.",                                   # Forecaster's summary prediction (latent consistency)
        r"^verifier\.\w+\.out\.",                              # Verifier head outputs
        r"^taaft\.\w+\.\w+\.(w_o|prior|trust_prior|cost|log_prec|mlp_out)\.",   # lens energy readouts and outputs
        r"^taaft\.\w+\.\w+\.(a|b|w_a|w_e|w_i)\.weight$",       # lens couplings (pairwise and game subspaces)
        r"^tstct\.prior_mlp\.down\.",                          # transition-prior parameters (an output)
    )
    input_patterns: tuple[str, ...] = (
        r"\.read_proj\.",                                      # probability readouts -> model width
        r"^verifier\.\w+\.inp\.",                              # forecast-summary and Monitor features
        r"\.clock\.",                                          # clock features (D-50)
    )
    stacked_patterns: tuple[str, ...] = (r"^cvgae\..*\.w_[kv]$",)
    thin_max: int = 4


@dataclass
class Assignment:
    """The group of every trainable parameter, with the reason (for audit and for the run manifest)."""

    groups: dict[str, list[tuple[str, nn.Parameter]]] = field(default_factory=lambda: {g: [] for g in GROUPS})
    reasons: dict[str, str] = field(default_factory=dict)

    def muon(self) -> list[tuple[str, nn.Parameter]]:
        return [x for g in GROUPS if g.startswith("muon/") for x in self.groups[g]]

    def adamw(self) -> list[tuple[str, nn.Parameter]]:
        return [x for g in GROUPS if g.startswith("adamw/") for x in self.groups[g]]

    def counts(self) -> dict[str, tuple[int, int]]:
        """group -> (tensors, elements)."""
        return {g: (len(v), sum(p.numel() for _, p in v)) for g, v in self.groups.items()}


def _owners(root: nn.Module) -> dict[int, tuple[str, str, nn.Module]]:
    """id(parameter) -> (module path, attribute name, owning module), first owner wins (tied weights)."""
    out: dict[int, tuple[str, str, nn.Module]] = {}
    for mpath, mod in root.named_modules():
        for attr, p in mod.named_parameters(recurse=False):
            out.setdefault(id(p), (mpath, attr, mod))
    return out


def assign_groups(root: nn.Module, named: Sequence[tuple[str, nn.Parameter]], *, rule: GroupRule | None = None,
                  prefix: str = "") -> Assignment:
    """Assign every (name, parameter) to exactly one group (module docstring). `root` owns the parameters;
    `prefix` is prepended to `root`'s parameter paths so the patterns see full model names."""
    use("AS-599", by=__name__)
    r = rule or GroupRule()
    owners = _owners(root)
    out = Assignment()
    out_re = [re.compile(p) for p in r.output_patterns]
    in_re = [re.compile(p) for p in r.input_patterns]
    st_re = [re.compile(p) for p in r.stacked_patterns]
    for name, p in named:
        own = owners.get(id(p))
        if own is None:
            raise InvariantViolation(f"{name}: parameter not owned by the given module")
        mpath, attr, mod = own
        full = ".".join(x for x in (prefix, mpath, attr) if x)
        group, reason = _classify(full, attr, mod, p, out_re, in_re, st_re, r, root, mpath)
        if group.startswith("muon/") and p.dim() not in (2, 3):
            raise InvariantViolation(f"{full}: rank-{p.dim()} parameter assigned to Muon")
        if group.startswith("muon/") and p.dim() == 3 and group != "muon/stacked":
            raise InvariantViolation(f"{full}: rank-3 parameter outside the stacked-matrix group")
        out.groups[group].append((name, p))
        out.reasons[name] = f"{group}: {reason}"
    total = sum(len(v) for v in out.groups.values())
    if total != len(named) or len({id(p) for _, p in named}) != len(named):
        raise InvariantViolation("parameter assignment is not one group per parameter")
    return out


def _classify(full: str, attr: str, mod: nn.Module, p: nn.Parameter, out_re: list[re.Pattern[str]],
              in_re: list[re.Pattern[str]], st_re: list[re.Pattern[str]], r: GroupRule, root: nn.Module,
              mpath: str) -> tuple[str, str]:
    """(group, reason) of one parameter (module docstring)."""
    if isinstance(mod, nn.Embedding):
        return "adamw/embedding", "embedding table"
    if isinstance(mod, RMSNorm) or isinstance(mod, nn.LayerNorm):
        return "adamw/norm", "normalisation gain"
    if isinstance(mod, nn.Linear):
        if attr == "bias":
            return "adamw/bias", "bias of a linear map"
        if any(x.search(full) for x in out_re):
            return "adamw/output", "produces a model output"
        if any(x.search(full) for x in in_re):
            return "adamw/input", "reads raw features"
        if min(mod.in_features, mod.out_features) <= r.thin_max:
            return "adamw/thin", f"side <= {r.thin_max}"
        parent = _parent(root, mpath)
        if isinstance(parent, MultiHeadAttention) and mpath.rsplit(".", 1)[-1] in ("q_proj", "k_proj", "v_proj", "o_proj"):
            return "muon/attention", "attention projection"
        if isinstance(parent, SwiGLU):
            leaf = mpath.rsplit(".", 1)[-1]
            if leaf == "down" and parent.down.out_features != parent.gate.in_features:
                return "adamw/output", "SwiGLU output projection to another width"
            return "muon/mlp", f"SwiGLU {leaf}"
        if isinstance(parent, nn.Sequential):
            return "muon/mlp", "hidden layer of an MLP"
        return "muon/hidden", "linear map between hidden representations"
    # Raw parameters (not owned by a standard layer).
    if any(x.search(full) for x in st_re) and p.dim() == 3:
        return "muon/stacked", "stacked hidden matrices (one per kind)"
    if p.dim() == 0 or p.numel() == 1:
        return "adamw/scalar", "scalar"
    if p.dim() == 1:
        return "adamw/vector", "one-dimensional parameter"
    if attr == "bits" or (attr in ("weight", "bias", "freq") and type(mod).__name__ == "PeriodicEmbedding"):
        return "adamw/embedding", "embedding-like table of an input encoding"
    return "adamw/table", "raw table (not a linear map)"


def _parent(root: nn.Module, mpath: str) -> nn.Module | None:
    if "." not in mpath:
        return root if mpath else None
    return root.get_submodule(mpath.rsplit(".", 1)[0])


__all__ = ["Assignment", "GROUPS", "GroupRule", "MatrixMuon", "NS_COEFFICIENTS", "adjusted_lr", "assign_groups",
           "newton_schulz"]
