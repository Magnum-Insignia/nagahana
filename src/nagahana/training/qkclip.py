"""QK-Clip for QK-normed attention: bounding pre-softmax logits after every optimiser step (D-61; AS-581).

Background
----------
MuonClip (Kimi K2, arXiv:2507.20534) keeps attention logits bounded under Muon: after each optimiser
step, for every head h whose largest pre-softmax logit S_h = max_ij <q_hi, k_hj> / sqrt(d_h) in the step
exceeded tau, it rescales that head's W_q and W_k by sqrt(tau / S_h), so the logits computed with the
same inputs fall to tau.

Every attention of NagaHana normalises queries and keys per head after the projection (QK-norm, AS-32):

    q_hat = g_q (.) q / rms(q),     k_hat = g_k (.) k / rms(k),     rms(u) = sqrt(mean(u^2) + eps)

Scaling W_q or W_k leaves q_hat and k_hat unchanged, so clipping the projections would do nothing. The
logit scale lives in the QK-norm gains g_q, g_k (one gain vector of width d_h per attention module,
shared by its heads; nn/attention.py) and in the unnormalised learned null key k_null (per head). QK-Clip
therefore acts on them; this is the same operation as MuonClip's, applied to the parameters that
actually set the logit scale.

Tracking (cheap, during the forward pass)
-----------------------------------------
Forward hooks on the query and key norms record, per head, M_q(h) = max_i ||q_hat_hi|| and
M_k(h) = max_j ||k_hat_hj|| over the step's micro-batches and loop passes (rotary encodings preserve
norms). By Cauchy-Schwarz,

    S_h <= S_bar_h = M_q(h) M_k(h) / sqrt(d_h),        S_null_h <= M_q(h) ||k_null_h|| / sqrt(d_h)

so the bound costs one norm per projected vector and never under-estimates a logit. Keys a module does
not project itself (TAAFT's cross-attention reads TSTCT's cached keys and the long-term memory's keys,
AS-13, AS-220; its self-attention also reads carried Imagination keys, AS-223) are observed explicitly
by the training code (`observe_keys`), so every key a query meets is covered.

Clipping (after the optimiser step; the same on every rank)
-----------------------------------------------------------
The per-head maxima are all-reduced with MAX across ranks. For a module with S = max_h S_bar_h > tau:

    own keys        g_q <- g_q * sqrt(tau / S),   g_k <- g_k * sqrt(tau / S)     (MuonClip's split)
    supplied keys   g_q <- g_q * (tau / S)                                      (the key side is not this module's)

and then, with M_q scaled accordingly, every head whose null-key bound exceeds tau gets
k_null_h <- k_null_h * tau / S_null_h. Because q_hat and k_hat are linear in their gains, the bounds of
the same inputs become <= tau, hence every logit <= tau (tests/test_training_optim.py checks the
logits themselves). With shared gains the rule acts per module: every head of a module whose largest
head exceeds tau is scaled. Additive biases (topology, recency, log-count, causal gates) are not
logits of the QK product and are not clipped. CVG-AE's hyperedge-to-node attention (query norm and
message norm) is a site of its own kind; its node-to-hyperedge scores <W_k h, a> have no QK-norm and
are left to Muon's spectral control of W_k.

Only modules being trained in the stage are clipped; frozen gains never change.
"""

from __future__ import annotations

import contextlib
import fnmatch
import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn

from nagahana.nn.attention import MultiHeadAttention
from nagahana.nn.norms import RMSNorm
from nagahana.training.assumptions import use


@dataclass
class ClipSite:
    """One attention module whose logits are bounded."""

    name: str
    heads: int
    head_dim: int
    q_gain: nn.Parameter
    k_gain: nn.Parameter | None
    null_key: nn.Parameter | None
    q_max: torch.Tensor = field(default_factory=lambda: torch.zeros(0))
    k_max: torch.Tensor = field(default_factory=lambda: torch.zeros(0))

    def reset(self) -> None:
        self.q_max = torch.zeros(self.heads, dtype=torch.float64)
        self.k_max = torch.zeros(self.heads, dtype=torch.float64)


def _local(t: torch.Tensor) -> torch.Tensor:
    fn = getattr(t, "to_local", None)
    return fn() if callable(fn) else t


def _full(t: torch.Tensor) -> torch.Tensor:
    fn = getattr(t, "full_tensor", None)
    return fn() if callable(fn) else t


def head_max_norms(x: torch.Tensor, heads: int, head_axis: int) -> torch.Tensor:
    """Per-head maximum of the L2 norm over the last axis, float64 [H] (head axis given)."""
    with torch.no_grad():
        n = x.detach().float().norm(dim=-1)                              # drop d_h
        ax = head_axis if head_axis >= 0 else head_axis + x.dim()
        if ax >= n.dim():
            raise ValueError("the head axis cannot be the last (feature) axis")
        if n.shape[ax] != heads:
            raise ValueError(f"expected {heads} heads on axis {head_axis}, got shape {tuple(x.shape)}")
        n = n.movedim(ax, 0).reshape(heads, -1)
        if n.shape[1] == 0:
            return torch.zeros(heads, dtype=torch.float64)
        return n.amax(dim=1).to(device="cpu", dtype=torch.float64)


class QKClip:
    """Tracks per-head QK-norm maxima of the trained attention modules and clips after each step.

    root: the module whose attention modules are considered; trainable(p): which gains may change in
    this stage (default: `requires_grad`). tau: the logit bound. The hooks track only inside
    `tracking()`.
    """

    def __init__(self, root: nn.Module, *, tau: float, prefix: str = "") -> None:
        use("AS-581", by=__name__)
        if not tau > 0:
            raise ValueError("tau must be > 0")
        self.tau = float(tau)
        self.sites: dict[str, ClipSite] = {}
        self._handles: list[Any] = []
        self._active = False
        for path, mod in root.named_modules():
            name = ".".join(x for x in (prefix, path) if x)
            site = self._site_of(name, mod)
            if site is None or not site.q_gain.requires_grad:
                continue
            site.reset()
            self.sites[name] = site
            q_norm = mod.q_norm
            self._handles.append(q_norm.register_forward_hook(self._hook(name, "q")))
            k_norm = getattr(mod, "k_norm", None) if isinstance(mod, MultiHeadAttention) else getattr(mod, "m_norm", None)
            if site.k_gain is not None and isinstance(k_norm, RMSNorm):
                self._handles.append(k_norm.register_forward_hook(self._hook(name, "k")))

    @staticmethod
    def _site_of(name: str, mod: nn.Module) -> ClipSite | None:
        # MultiHeadAttention with QK-norm; CVG-AE plane layers (query norm and message norm).
        if isinstance(mod, MultiHeadAttention):
            if not isinstance(mod.q_norm, RMSNorm):
                return None
            k_norm = getattr(mod, "k_norm", None)
            return ClipSite(name=name, heads=mod.n_heads, head_dim=mod.head_dim, q_gain=mod.q_norm.weight,
                            k_gain=k_norm.weight if isinstance(k_norm, RMSNorm) else None,
                            null_key=mod.k_null if mod.null_kv else None)
        q_norm, m_norm = getattr(mod, "q_norm", None), getattr(mod, "m_norm", None)
        h, dh = getattr(mod, "h", None), getattr(mod, "dh", None)
        if isinstance(q_norm, RMSNorm) and isinstance(m_norm, RMSNorm) and isinstance(h, int) and isinstance(dh, int):
            return ClipSite(name=name, heads=h, head_dim=dh, q_gain=q_norm.weight, k_gain=m_norm.weight, null_key=None)
        return None

    def _hook(self, name: str, side: str) -> Any:
        def hook(_module: nn.Module, _inputs: Any, output: torch.Tensor) -> None:
            if not self._active:
                return
            site = self.sites[name]
            # Projected queries and keys are [B, H, T, d_h] (attention) or [R, H, d_h] (CVG-AE): heads on axis 1.
            m = head_max_norms(output, site.heads, 1)
            if side == "q":
                site.q_max = torch.maximum(site.q_max, m)
            else:
                site.k_max = torch.maximum(site.k_max, m)
        return hook

    def add_key_source(self, module: nn.Module, pattern: str, *, head_axis: int = 1) -> None:
        """Observe the outputs of `module` (a key normalisation feeding other modules' attention, such as
        TAAFT's long-term memory keys, AS-220) as keys of the sites matching `pattern`."""
        def hook(_module: nn.Module, _inputs: Any, output: torch.Tensor) -> None:
            self.observe_keys(pattern, output, head_axis=head_axis)
        self._handles.append(module.register_forward_hook(hook))

    def remove(self) -> None:
        """Remove every hook."""
        for h in self._handles:
            h.remove()
        self._handles.clear()

    @contextlib.contextmanager
    def tracking(self) -> Iterator[None]:
        """Record maxima inside the block (training forward passes)."""
        prev = self._active
        self._active = True
        try:
            yield
        finally:
            self._active = prev

    def observe_keys(self, pattern: str, keys: torch.Tensor, *, head_axis: int) -> int:
        """Record keys that the matching sites read but do not project (module docstring). Returns the
        number of sites updated. `pattern` is an fnmatch pattern over site names."""
        if not self._active:
            return 0
        n = 0
        for name, site in self.sites.items():
            if fnmatch.fnmatchcase(name, pattern):
                site.k_max = torch.maximum(site.k_max, head_max_norms(keys, site.heads, head_axis))
                n += 1
        return n

    def bounds(self) -> dict[str, tuple[float, float]]:
        """Per site: (max over heads of the QK logit bound, of the null-key logit bound), as tracked."""
        out: dict[str, tuple[float, float]] = {}
        for name, s in self.sites.items():
            scale = 1.0 / math.sqrt(s.head_dim)
            qk = float((s.q_max * s.k_max).max()) * scale if s.heads else 0.0
            nul = 0.0
            if s.null_key is not None:
                nk = _full(s.null_key.detach()).float().norm(dim=-1).to("cpu", torch.float64)
                nul = float((s.q_max * nk).max()) * scale
            out[name] = (qk, nul)
        return out

    def _reduce(self, info: Any) -> None:
        # Identical decisions on every rank: MAX over ranks of every tracked maximum.
        import torch.distributed as dist

        if info is None or not getattr(info, "distributed", False) or not dist.is_initialized():
            return
        names = sorted(self.sites)
        if not names:
            return
        flat = torch.cat([torch.cat([self.sites[n].q_max, self.sites[n].k_max]) for n in names])
        dev = info.device if getattr(info, "backend", None) == "nccl" else torch.device("cpu")
        t = flat.to(dev)
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        t = t.to("cpu")
        off = 0
        for n in names:
            h = self.sites[n].heads
            self.sites[n].q_max = t[off:off + h].clone()
            self.sites[n].k_max = t[off + h:off + 2 * h].clone()
            off += 2 * h

    @torch.no_grad()
    def clip(self, info: Any = None) -> dict[str, float]:
        """Apply the clipping rule of the module docstring and reset the maxima. Returns statistics."""
        self._reduce(info)
        stats = {"qk_clip/max_bound": 0.0, "qk_clip/sites_clipped": 0.0, "qk_clip/null_heads_clipped": 0.0}
        for s in self.sites.values():
            scale = 1.0 / math.sqrt(s.head_dim)
            q_max = s.q_max.clone()
            seen_keys = bool((s.k_max > 0).any())
            if seen_keys:
                bound = float((s.q_max * s.k_max).max()) * scale
                stats["qk_clip/max_bound"] = max(stats["qk_clip/max_bound"], bound)
                if bound > self.tau:
                    gamma = self.tau / bound
                    if s.k_gain is not None:
                        f = math.sqrt(gamma)
                        _local(s.q_gain).mul_(f)
                        _local(s.k_gain).mul_(f)
                        q_max = q_max * f
                    else:
                        _local(s.q_gain).mul_(gamma)
                        q_max = q_max * gamma
                    stats["qk_clip/sites_clipped"] += 1.0
            if s.null_key is not None and bool((q_max > 0).any()):
                nk = _full(s.null_key.detach()).float().norm(dim=-1).to("cpu", torch.float64)    # [H]
                nb = q_max * nk * scale
                over = nb > self.tau
                if bool(over.any()):
                    factor = torch.where(over, self.tau / nb.clamp_min(1e-30), torch.ones_like(nb))
                    local = _local(s.null_key)
                    rows = _row_slice(s.null_key, local)
                    local.mul_(factor[rows].to(device=local.device, dtype=local.dtype).unsqueeze(-1))
                    stats["qk_clip/null_heads_clipped"] += float(over.sum())
            s.reset()
        return stats

    def state_dict(self) -> dict[str, Any]:
        """The tracked maxima (empty right after a clip, which is when checkpoints are taken)."""
        return {n: {"q_max": s.q_max.clone(), "k_max": s.k_max.clone()} for n, s in self.sites.items()}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        for n, s in self.sites.items():
            if n in state:
                s.q_max = state[n]["q_max"].to(torch.float64).clone()
                s.k_max = state[n]["k_max"].to(torch.float64).clone()


def _row_slice(full_like: torch.Tensor, local: torch.Tensor) -> slice:
    """The rows of a (possibly dim-0-sharded) [H, d_h] parameter held by this rank."""
    if local is full_like or not hasattr(full_like, "device_mesh"):
        return slice(0, local.shape[0])
    from torch.distributed.tensor import Shard

    placements = full_like.placements  # type: ignore[attr-defined]
    mesh = full_like.device_mesh  # type: ignore[attr-defined]
    if not placements or not isinstance(placements[0], Shard) or placements[0].dim != 0:
        return slice(0, local.shape[0])
    world = mesh.size()
    rank = mesh.get_local_rank()
    h = full_like.shape[0]
    chunk = math.ceil(h / world)
    start = min(rank * chunk, h)
    return slice(start, start + local.shape[0])


def logit_bound_sites(clip: QKClip) -> Sequence[str]:
    """Names of the sites under QK-Clip."""
    return sorted(clip.sites)


__all__ = ["ClipSite", "QKClip", "head_max_norms", "logit_bound_sites"]
