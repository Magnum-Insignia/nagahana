"""Shared machinery of the three feedback learners: batches, held-out splits, seeds, exclusions, the delta optimiser.

One interface for RLHF, RLVR and RLCD (D-65):

    learner.fit(batch: FeedbackBatch) -> CandidateUpdate

A `FeedbackBatch` is a selection of ledger feedback (record index, event) plus the situations the events
refer to. A learner reads the kinds it learns from, resolves each event's situation, records every event
it cannot use with the reason (`Exclusions`), and returns a candidate update that leaves the reference
weights untouched (`params.ParameterDelta`).

Held-out split (AS-842). `split_batch` orders the events by event time (ties by ledger index) and holds
out the latest share: a candidate is judged on feedback that arrived after everything it was fitted on,
so no later information can leak into the fit (the temporal discipline of the data splits, D-23).

Seeds (AS-848). Every random draw has its own generator seeded by SHA-256 of (root seed, purpose, ...),
so a fit is reproducible from the configuration and the ledger alone and independent of call order.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol

import torch

from nagahana.core.errors import InvariantViolation
from nagahana.models.verifier.canonical import canonical_json, sha256_hex
from nagahana.models.verifier.config import OptimiserConfig
from nagahana.models.verifier.feedback import FeedbackEvent, event_id
from nagahana.models.verifier.situations import Situation


@dataclass(frozen=True)
class Exclusion:
    """An event a learner could not use, and why."""

    index: int
    event_id: str
    reason: str


@dataclass
class Exclusions:
    """Collector of exclusions during one fit or evaluation."""

    items: list[Exclusion] = field(default_factory=list)

    def add(self, index: int, event: FeedbackEvent, reason: str) -> None:
        self.items.append(Exclusion(int(index), event_id(event), reason))

    def as_tuple(self) -> tuple[Exclusion, ...]:
        return tuple(self.items)


@dataclass(frozen=True)
class FeedbackBatch:
    """Feedback events (ledger index, event) with the situations they refer to (module docstring)."""

    events: tuple[tuple[int, FeedbackEvent], ...]
    situations: Mapping[str, Situation] = field(default_factory=dict)
    ledger_head: str = ""

    def __len__(self) -> int:
        return len(self.events)

    def of_kind(self, *kinds: str) -> list[tuple[int, FeedbackEvent]]:
        """The events of the given kinds, in batch order."""
        return [(i, e) for i, e in self.events if e.kind in kinds]

    def with_events(self, events: Iterable[tuple[int, FeedbackEvent]]) -> FeedbackBatch:
        """The same situations and head with another event selection."""
        return FeedbackBatch(tuple(events), self.situations, self.ledger_head)


def split_batch(batch: FeedbackBatch, held_out_fraction: float) -> tuple[FeedbackBatch, FeedbackBatch]:
    """(fit part, held-out part): the latest ceil(fraction * n) events by time are held out (module docstring)."""
    if not 0.0 < held_out_fraction < 1.0:
        raise InvariantViolation("held_out_fraction must lie in (0, 1)")
    order = sorted(batch.events, key=lambda ie: (float(ie[1].provenance.time), int(ie[0])))
    n = len(order)
    n_held = min(n - 1, math.ceil(held_out_fraction * n)) if n >= 2 else 0
    fit, held = order[: n - n_held], order[n - n_held:]
    return batch.with_events(fit), batch.with_events(held)


def resolve_situation(batch: FeedbackBatch, event: FeedbackEvent) -> tuple[Situation | None, str]:
    """(situation, "") for the record the event refers to, or (None, reason).

    When the event pins a situation digest, the stored situation must carry exactly that digest.
    """
    sid = event.provenance.refers_to
    s = batch.situations.get(sid)
    if s is None:
        return None, f"situation {sid!r} not available"
    pinned = event.provenance.situation_digest
    if pinned and s.digest != pinned:
        return None, f"situation {sid!r} differs from the one the analyst saw (digest mismatch)"
    return s, ""


def derive_seed(root: int, *parts: object) -> int:
    """A 63-bit seed from the root seed and a purpose path (module docstring)."""
    text = canonical_json([int(root), *[str(p) for p in parts]])
    return int(sha256_hex(text)[:16], 16) & (2**63 - 1)


def generator_for(root: int, *parts: object) -> torch.Generator:
    """A CPU generator seeded by `derive_seed(root, *parts)`."""
    return torch.Generator().manual_seed(derive_seed(root, *parts))


def make_optimiser(params: Sequence[torch.nn.Parameter], cfg: OptimiserConfig) -> torch.optim.AdamW:
    """AdamW over the delta's parameters with decoupled decay toward zero (config.OptimiserConfig)."""
    if not params:
        raise InvariantViolation("an optimiser needs at least one parameter")
    return torch.optim.AdamW(list(params), lr=cfg.lr, betas=(cfg.beta1, cfg.beta2), eps=cfg.eps, weight_decay=cfg.weight_decay)


def optimiser_step(opt: torch.optim.Optimizer, params: Sequence[torch.nn.Parameter], loss: torch.Tensor,
                   cfg: OptimiserConfig) -> float:
    """zero_grad, backward, global-norm clip, step. Returns the gradient norm before clipping."""
    if not bool(torch.isfinite(loss)):
        raise InvariantViolation(f"feedback loss is not finite ({float(loss.detach())})")
    opt.zero_grad(set_to_none=True)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(list(params), cfg.grad_clip)
    opt.step()
    return float(norm)


def minibatches(n: int, size: int, generator: torch.Generator) -> list[list[int]]:
    """A random partition of range(n) into batches of at most `size` (one epoch)."""
    perm = torch.randperm(n, generator=generator).tolist()
    return [perm[i:i + size] for i in range(0, n, size)]


class FeedbackLearner(Protocol):
    """The one interface of RLHF, RLVR and RLCD (module docstring)."""

    method: str

    def fit(self, batch: FeedbackBatch) -> object:
        """A candidate update (candidates.CandidateUpdate) from the batch; the reference weights stay unchanged."""
        ...


__all__ = ["Exclusion", "Exclusions", "FeedbackBatch", "FeedbackLearner", "derive_seed", "generator_for", "make_optimiser",
           "minibatches", "optimiser_step", "resolve_situation", "split_batch"]
