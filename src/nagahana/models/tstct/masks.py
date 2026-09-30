"""Attention patterns of TSTCT's three head types (diagram 03; corrected 2026-09-29).

Each head type h attends only where its allowed pattern M_h is 0 (−∞ elsewhere):

    Attn_h(Q, K, V) = softmax( QKᵀ/√d + B_h + M_h ) V

- **spatial**: entities side by side, each at its *latest state as of t* (event-driven updates rarely
  share a timestamp, so "same timestamp" would be impractical). i attends to j iff i and j are
  linked in 𝒢_t (or i = j). B_h carries the topology: hypergraph distance and kinds, the
  "Topological" in TSTCT [A-10].
- **temporal**: one entity across its own past. i attends to j iff they are the same entity and
  t_j ≤ t_i. **No future leakage** is an invariant tested in `tests/`: attending to the future
  would be data snooping (Arp et al., USENIX Security 2022).
- **causal**: directed influence from an earlier cause to a later effect across entities. It is
  learned and sparse. How the sparse structure is learned is not specified yet, so this pattern
  is a template.

All builders return a boolean "allowed" matrix. `to_additive` converts it to the additive mask M_h
and refuses rows with nothing allowed (a softmax over all −∞ is NaN).
"""

from __future__ import annotations

import torch

from nagahana.core.errors import InvariantViolation, NotBuiltYet


def temporal_allowed(entity: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    """allowed[i, j] = (entity_i == entity_j) ∧ (t_j ≤ t_i). Shapes: [T] → [T, T]."""
    if entity.shape != time.shape or entity.dim() != 1:
        raise InvariantViolation("entity and time must be 1-D and of equal length")
    same = entity[:, None] == entity[None, :]
    past_or_now = time[None, :] <= time[:, None]
    return same & past_or_now


def spatial_allowed(adjacency: torch.Tensor) -> torch.Tensor:
    """allowed = adjacency ∨ I, for entity states as of one time t. Shape [V, V] (bool)."""
    if adjacency.dim() != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise InvariantViolation("adjacency must be square")
    eye = torch.eye(adjacency.shape[0], dtype=torch.bool, device=adjacency.device)
    return adjacency.bool() | eye


def causal_allowed(*_args: object, **_kwargs: object) -> torch.Tensor:
    """Learned, sparse cause → effect pattern (template; design not yet specified)."""
    raise NotBuiltYet("TSTCT causal-head structure learning", waiting_on=("TSTCT causal-head design",))


def to_additive(allowed: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """0 where allowed, −∞ elsewhere. Every row must allow at least one position."""
    if not bool(allowed.any(dim=-1).all()):
        raise InvariantViolation("a query row allows no key; softmax would be NaN")
    out = torch.zeros(allowed.shape, dtype=dtype, device=allowed.device)
    return out.masked_fill(~allowed, float("-inf"))
