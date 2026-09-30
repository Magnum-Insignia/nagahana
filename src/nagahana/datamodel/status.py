"""Observation status: what a field's value is worth as evidence.

The decided principle (D-41, [Q-26])
------------------------------------
"absence of info means that the info hasn't been supplied and this comes into the concept of
modeling of the POMDP, POSG etc partially observable modeling". So a missing value is never
written as zero, never silently imputed, and never treated as "nothing happened". Lack of evidence
is not lack of event (ARCH §2.2 observability floor).

The proposed taxonomy (P-03, awaiting approval)
-----------------------------------------------
Five statuses, attached to every field of every state update:

    OBSERVED          seen, and trusted as measured
    STALE             seen earlier; carries its age
    LOW_RELIABILITY   seen, but doubtful (e.g. clock skew, spoofable source); carries a reliability
    NOT_SUPPLIED      the source cannot provide it (e.g. NetFlow has no TTL)
    NOT_OBSERVABLE    cannot be seen passively at all (e.g. TLS-encrypted payload, untapped segment)

GRU-D (Che et al., Sci. Rep. 2018) showed that missingness patterns are themselves informative, so
the status is also an *input*, not only a mask.

How status enters the maths (diagram 04; tempered likelihood)
-------------------------------------------------------------
    p(o_t | s, m_t) = Π_{i ∈ O_t} p(x_i | s)^{w(m_i)},   O_t = {i : m_i ∈ CONTRIBUTING}

    w(OBSERVED) = 1
    0 < w(STALE), w(LOW_RELIABILITY) < 1       values held: D-29
    NOT_SUPPLIED, NOT_OBSERVABLE               no factor at all (i ∉ O_t)

The last line is the crux. An excluded field is *removed* from the product; it is not multiplied
by w = 0 inside it. The two differ in code: a zero-weighted factor still passes its (fabricated)
value through whatever computes p(x_i | s), and a NaN or an imputation there leaks into gradients.
Removal leaks nothing.
"""

from __future__ import annotations

import enum
from collections.abc import Mapping, Sequence

import torch

from nagahana.core.errors import ConfigMissing, InvariantViolation


class ObservationStatus(enum.Enum):
    """Evidence status of one field value. Taxonomy: proposal P-03; principle: decided D-41."""

    OBSERVED = "observed"
    STALE = "stale"
    LOW_RELIABILITY = "low_reliability"
    NOT_SUPPLIED = "not_supplied"
    NOT_OBSERVABLE = "not_observable"


#: Statuses whose values enter the evidence (the set O_t above).
CONTRIBUTING: frozenset[ObservationStatus] = frozenset(
    {ObservationStatus.OBSERVED, ObservationStatus.STALE, ObservationStatus.LOW_RELIABILITY}
)

#: Statuses that carry no value and no factor.
EXCLUDED: frozenset[ObservationStatus] = frozenset(
    {ObservationStatus.NOT_SUPPLIED, ObservationStatus.NOT_OBSERVABLE}
)

#: Statuses whose weight is held (D-29) and must come from config.
DOWNWEIGHTED: frozenset[ObservationStatus] = frozenset(
    {ObservationStatus.STALE, ObservationStatus.LOW_RELIABILITY}
)


def contributes(status: ObservationStatus) -> bool:
    """True if a value with this status counts as evidence (is in O_t)."""
    return status in CONTRIBUTING


def validate_weights(weights: Mapping[ObservationStatus, float]) -> None:
    """Check configured weights for the down-weighted statuses.

    Every down-weighted status needs a weight strictly inside (0, 1). Weights for OBSERVED or for
    excluded statuses are rejected, because those are fixed by definition (1, and "no factor").
    """
    for status in DOWNWEIGHTED:
        if status not in weights:
            raise ConfigMissing(
                f"No evidence weight configured for {status.value!r} (held decision D-29)."
            )
        w = float(weights[status])
        if not 0.0 < w < 1.0:
            raise InvariantViolation(f"Weight for {status.value!r} must be in (0, 1); got {w}.")
    extra = set(weights) - DOWNWEIGHTED
    if extra:
        raise InvariantViolation(
            f"Weights may be set only for {sorted(s.value for s in DOWNWEIGHTED)}; "
            f"got {sorted(s.value for s in extra)} (observed=1 and exclusion are definitions)."
        )


def evidence_weight(status: ObservationStatus, weights: Mapping[ObservationStatus, float]) -> float:
    """The exponent w(m) of a contributing field. Raises for excluded fields.

    Raising (instead of returning 0) forces callers to *remove* excluded fields rather than zero
    them; see the module docstring for why that difference matters.
    """
    if status in EXCLUDED:
        raise InvariantViolation(
            f"{status.value!r} fields contribute no factor; remove them from O_t instead of "
            "weighting them."
        )
    if status is ObservationStatus.OBSERVED:
        return 1.0
    validate_weights(weights)
    return float(weights[status])


def contributing_mask(statuses: Sequence[ObservationStatus]) -> torch.Tensor:
    """Boolean mask m (True where a field contributes), in the order of `statuses`.

    This is the m_c used by the shared physics term Φ_phys (physics/term.py) and by the masked
    likelihood (P-20).
    """
    return torch.tensor([s in CONTRIBUTING for s in statuses], dtype=torch.bool)
