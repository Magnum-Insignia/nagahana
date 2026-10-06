"""Long-term memory: a Titans-style neural memory updated once per Forecaster trigger (build-spec §2.6).

Purpose
-------
The Environment store (`memory/environment.py`) keeps weeks of per-entity states at log-time
resolution. Behind it sits a *neural* memory: a small MLP whose weights are the memory, written by
gradient steps on an associative loss and read by a forward pass. TAAFT reads it as extra memory
keys/values (build-spec §2.6, §4b.2 "the Titans memory sits behind them").

Sources: Behrouz, Zhong & Mirrokni, "Titans: Learning to Memorize at Test Time", 2024
(arXiv:2501.00663). Owner: [Q-19] (months of memory), [Q-33] (retention not controllable by attack
volume). Decisions: D-36 (retention per trigger). Assumptions: AS-11 (form and α value), AS-155
(η, θ fixed scalars; mean-over-pairs gradient; ℓ2-normalised keys/queries; SiLU MLP).

Maths
-----
Memory M = (W₁, W₂), M(x) = W₂ SiLU(W₁ x). At trigger k the writer supplies N pairs from its inputs
x_i: keys k_i = normalise(W_K x_i), values v_i = W_V x_i. The associative loss is Titans' squared
error ‖M(k) − v‖² per pair, averaged over the N pairs (summed over the D dimensions):

    ℓ(M; {k_i, v_i}) = (1/N) Σ_i ‖M(k_i) − v_i‖²,

and the update is Titans' momentum-of-surprise with forgetting:

    S_k = η S_{k−1} − θ ∇_M ℓ(M_{k−1}; {k_i, v_i})
    M_k = (1 − α_k) M_{k−1} + S_k,          α_k = RetentionSchedule.alpha(k)   (D-36)

Reading: y = M_k(normalise(W_Q q)).

Why this form here (and how it differs from Titans)
---------------------------------------------------
- **Once per trigger, never per state update (D-36).** Titans updates at every token, with
  data-dependent α_t, η_t, θ_t. Here the update runs once per Forecaster trigger, α_k comes from a
  schedule whose signature takes only k, and η, θ are fixed scalars (AS-155). An attacker can choose
  *what* is written but not how fast the memory forgets.
- **Mean over pairs.** ∇ℓ is the mean over the trigger's N pairs, so N (which traffic volume can
  inflate) does not scale the step. Volume can change the content of a write, never its magnitude.
- **Closed-form gradient.** For the two-layer SiLU MLP the gradient is written out (below) instead of
  calling autograd inside the forward, so the write is an ordinary differentiable tensor program:
  the slow weights (W_K, W_V, W_Q, M₀) can be meta-learned through it with plain backprop.
- **ℓ2-normalised keys and queries** bound the scale of the memory's inputs (Titans normalises
  queries and keys with the ℓ2 norm; section to verify).

Gradient (per batch element; r_i = M(k_i) − v_i, h_i = W₁k_i, a_i = SiLU(h_i), c = N):
    ∂ℓ/∂W₂ = (2/c) Σ_i r_i a_iᵀ
    ∂ℓ/∂W₁ = (2/c) Σ_i [ (W₂ᵀ r_i) ⊙ SiLU′(h_i) ] k_iᵀ,     SiLU′(h) = σ(h)(1 + h(1 − σ(h)))
(tested against autograd in `tests/test_memory_longterm.py`).

Slow and fast weights
---------------------
The module's parameters (projections W_K, W_V, W_Q and the initial memory M₀) are slow weights,
trained by the stage objectives. The memory itself (W₁, W₂, S₁, S₂, k) is *data*: a
`NeuralMemoryState` passed in and returned, never an `nn.Parameter`. It is part of the Environment's
working memory: rebuildable from the event log, keyed to the model hash (P-18 assumed in AS-11).

Invariants
----------
- Retention depends only on the trigger index (`alpha(k)` receives k alone; tested).
- One `write` = one trigger: the trigger index advances by exactly one per call.

Extension points
----------------
- Deeper memories (L_M > 2 layers) replace `_forward`/`_grads` (or use torch.func for the gradient).
- A learned-but-input-independent θ schedule may replace the fixed θ (still volume-free).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from nagahana.core.errors import InvariantViolation
from nagahana.governance.assumptions import assume
from nagahana.memory.retention import RetentionSchedule
from nagahana.models.config.components import MemoryConfig


@dataclass(frozen=True)
class NeuralMemoryState:
    """The fast memory (data, not parameters).

    w1: [B, hidden, D]; w2: [B, D, hidden]; s1, s2: momentum of the same shapes;
    trigger: index k of the next trigger (number of writes so far).
    """

    w1: torch.Tensor
    w2: torch.Tensor
    s1: torch.Tensor
    s2: torch.Tensor
    trigger: int


def _normalise(x: torch.Tensor) -> torch.Tensor:
    # ℓ2 normalisation over the last dimension (safe at 0).
    return F.normalize(x, dim=-1, eps=1e-6)


class LongTermMemory(nn.Module):
    """Titans-style neural memory with per-trigger retention (see the module docstring).

    Parameters
    ----------
    cfg: MemoryConfig (longterm_dim D, longterm_hidden, longterm_momentum η, longterm_step θ).
    input_dim: width of the vectors written and queried (TAAFT's width).
    """

    def __init__(self, cfg: MemoryConfig, *, input_dim: int) -> None:
        super().__init__()
        assume("AS-11", by=__name__)
        d, hid = cfg.longterm_dim, cfg.longterm_hidden
        self.dim, self.hidden = d, hid
        self.eta, self.theta = cfg.longterm_momentum, cfg.longterm_step
        if not 0.0 <= self.eta < 1.0 or self.theta < 0.0:
            raise InvariantViolation("need 0 ≤ η < 1 and θ ≥ 0")
        # Slow weights: projections and the initial memory M₀.
        self.w_k = nn.Linear(input_dim, d, bias=False)
        self.w_v = nn.Linear(input_dim, d, bias=False)
        self.w_q = nn.Linear(input_dim, d, bias=False)
        self.init_w1 = nn.Parameter(torch.randn(hid, d) / d**0.5)
        self.init_w2 = nn.Parameter(torch.randn(d, hid) / hid**0.5 * 0.1)

    # ------------------------------------------------------------------ state
    def init_state(self, batch: int) -> NeuralMemoryState:
        """M₀ for `batch` independent memories (e.g. one per site or window), zero momentum, k = 0."""
        w1 = self.init_w1[None].expand(batch, -1, -1)
        w2 = self.init_w2[None].expand(batch, -1, -1)
        return NeuralMemoryState(w1=w1, w2=w2, s1=torch.zeros_like(w1), s2=torch.zeros_like(w2), trigger=0)

    # ------------------------------------------------------------------ the memory function
    @staticmethod
    def _forward(w1: torch.Tensor, w2: torch.Tensor, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """M(x) for x [B, N, D] → (y [B, N, D], h [B, N, hid], a [B, N, hid])."""
        h = torch.einsum("bnd,bhd->bnh", x, w1)
        a = F.silu(h)
        return torch.einsum("bnh,bdh->bnd", a, w2), h, a

    def associative_loss(self, state: NeuralMemoryState, keys: torch.Tensor, values: torch.Tensor,
                         mask: torch.Tensor | None = None) -> torch.Tensor:
        """ℓ per batch element [B] on memory-space pairs keys/values [B, N, D] (mask [B, N])."""
        y, _, _ = self._forward(state.w1, state.w2, keys)
        m = torch.ones(keys.shape[:2], dtype=keys.dtype, device=keys.device) if mask is None else mask.to(keys.dtype)
        c = m.sum(1).clamp_min(1.0)                                         # N valid pairs
        return (((y - values) ** 2).sum(-1) * m).sum(1) / c

    def _grads(self, state: NeuralMemoryState, keys: torch.Tensor, values: torch.Tensor,
               mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Closed-form ∂ℓ/∂W₁ [B, hid, D] and ∂ℓ/∂W₂ [B, D, hid] (see the module docstring)."""
        y, h, a = self._forward(state.w1, state.w2, keys)
        c = mask.sum(1).clamp_min(1.0)[:, None, None]                       # N valid pairs
        dy = 2.0 * (y - values) * mask[..., None] / c                         # [B, N, D]
        g2 = torch.einsum("bnd,bnh->bdh", dy, a)
        sig = torch.sigmoid(h)
        dh = torch.einsum("bnd,bdh->bnh", dy, state.w2) * sig * (1.0 + h * (1.0 - sig))
        g1 = torch.einsum("bnh,bnd->bhd", dh, keys)
        return g1, g2

    # ------------------------------------------------------------------ write and read
    def project_pairs(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Inputs [B, N, input_dim] → (normalised keys, values), each [B, N, D]."""
        return _normalise(self.w_k(x)), self.w_v(x)

    def write(self, state: NeuralMemoryState, x: torch.Tensor, schedule: RetentionSchedule, *,
              mask: torch.Tensor | None = None) -> NeuralMemoryState:
        """One trigger's write: inputs x [B, N, input_dim] (mask [B, N]) → the next state (k + 1)."""
        keys, values = self.project_pairs(x)
        m = torch.ones(x.shape[:2], dtype=keys.dtype, device=x.device) if mask is None else mask.to(keys.dtype)
        alpha = float(schedule.alpha(state.trigger))                         # D-36: a function of k only
        if not 0.0 <= alpha < 1.0:
            raise InvariantViolation("retention α_k must be in [0, 1)")
        g1, g2 = self._grads(state, keys, values, m)
        s1 = self.eta * state.s1 - self.theta * g1                           # momentum of surprise
        s2 = self.eta * state.s2 - self.theta * g2
        w1 = (1.0 - alpha) * state.w1 + s1                                   # forgetting, then surprise
        w2 = (1.0 - alpha) * state.w2 + s2
        return NeuralMemoryState(w1=w1, w2=w2, s1=s1, s2=s2, trigger=state.trigger + 1)

    def read(self, state: NeuralMemoryState, queries: torch.Tensor) -> torch.Tensor:
        """Retrieve: queries [B, Q, input_dim] → values [B, Q, D] (TAAFT uses them as extra memory keys/values)."""
        y, _, _ = self._forward(state.w1, state.w2, _normalise(self.w_q(queries)))
        return y
