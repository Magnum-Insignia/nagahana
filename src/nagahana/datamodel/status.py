"""Observation status: what a field's value is worth as evidence.

Principle (D-41): absence of information means the information has not been supplied. A missing value
is never written as zero, never silently imputed, and never read as "nothing happened". Lack of
evidence is not lack of event, which is the partially observable setting (POMDP, POSG) the model works
in.

Taxonomy (P-03). Five statuses, attached to every field of every state update:

    OBSERVED          seen, and trusted as measured
    STALE             seen earlier; carries its age in seconds
    LOW_RELIABILITY   seen, but doubtful (an estimate scaled from samples, a spoofable source, a
                      skewed clock); carries a reliability in (0, 1]
    NOT_SUPPLIED      the source does not provide it (NetFlow has no TTL; a Zeek field written "-")
    NOT_OBSERVABLE    present on the wire but not readable passively (an encrypted payload)

Missingness patterns are informative in themselves (Che et al., "Recurrent Neural Networks for
Multivariate Time Series with Missing Values", Scientific Reports 8:6085, 2018), so the status is a
model input and not only a mask.

How the status enters the likelihood (tempered product over contributing fields):

    p(o_t | s, m_t) = prod_{i in O_t} p(x_i | s)^w(m_i),    O_t = {i : m_i in CONTRIBUTING}

    w(OBSERVED) = 1
    0 < w(STALE), w(LOW_RELIABILITY) < 1        weights held by D-29
    NOT_SUPPLIED, NOT_OBSERVABLE                no factor at all (i not in O_t)

The last line matters for code: an excluded field is removed from the product, not multiplied by a
weight of zero inside it. A zero-weighted factor still passes its fabricated value through whatever
computes p(x_i | s), so a NaN or an imputation there leaks into gradients; removal leaks nothing.

Each catalogue field also declares the statuses it admits (`fields.FieldSpec.statuses`); a status
outside that set is rejected where state updates are built (`records.StateUpdate`).
"""

from __future__ import annotations

import enum
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from nagahana.core.errors import ConfigMissing, InvariantViolation

if TYPE_CHECKING:
    import torch


class ObservationStatus(enum.Enum):
    """Evidence status of one field value. Taxonomy P-03; principle D-41."""

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

#: Every status.
ALL_STATUSES: frozenset[ObservationStatus] = frozenset(ObservationStatus)

#: Admissible statuses of a value measured on the record that carries it (a flow counter, a packet
#: statistic, a protocol field). STALE is not admissible: a record's own measurement is never an
#: earlier observation carried forward.
MEASUREMENT_STATUSES: frozenset[ObservationStatus] = frozenset(
    {ObservationStatus.OBSERVED, ObservationStatus.LOW_RELIABILITY, ObservationStatus.NOT_SUPPLIED,
     ObservationStatus.NOT_OBSERVABLE}
)

#: Admissible statuses of a fact about persisting state (a certificate attribute, a software version,
#: an interface's speed): it may be reported from an earlier observation, so all five apply.
STATE_FACT_STATUSES: frozenset[ObservationStatus] = ALL_STATUSES

#: Admissible statuses of an identity value (an address, a name, a hash): read from the record or
#: absent; it may be doubtful (a spoofable source), and it is never an aged observation.
IDENTITY_STATUSES: frozenset[ObservationStatus] = frozenset(
    {ObservationStatus.OBSERVED, ObservationStatus.LOW_RELIABILITY, ObservationStatus.NOT_SUPPLIED,
     ObservationStatus.NOT_OBSERVABLE}
)


def contributes(status: ObservationStatus) -> bool:
    """True if a value with this status counts as evidence (is in O_t)."""
    return status in CONTRIBUTING


def validate_weights(weights: Mapping[ObservationStatus, float]) -> None:
    """Check configured weights for the down-weighted statuses.

    Every down-weighted status needs a weight strictly inside (0, 1). Weights for OBSERVED or for the
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

    Raising (instead of returning 0) forces callers to remove excluded fields rather than zero them;
    the module docstring says why that difference matters.
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

    This is the mask used by the shared physics term Phi_phys (physics/term.py) and by the masked
    likelihood (P-20). PyTorch is imported here, not at module import, so that ingest processes (which
    never build tensors) do not load it.
    """
    import torch

    return torch.tensor([s in CONTRIBUTING for s in statuses], dtype=torch.bool)


def parse_status(text: str) -> ObservationStatus:
    """The status named by `text` (its enum value, e.g. "not_supplied"); raises on anything else."""
    try:
        return ObservationStatus(text)
    except ValueError:
        raise InvariantViolation(
            f"Unknown observation status {text!r}; known: {[s.value for s in ObservationStatus]}"
        ) from None
