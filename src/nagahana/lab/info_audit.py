"""Information audit: what can the observables reveal about the hidden state? (proposal P-15)

The owner's aim: "see what the model must see" [Q-44]. Before asking a model to forecast from an
observation regime (full packets, NetFlow-only, encrypted, untapped segment), measure how much
information that regime carries about the hidden quantity (e.g. the attack stage):

    I(O; S) = Σ_{o,s} p(o, s) · log( p(o, s) / (p(o) p(s)) )      [nats]

If I(O; S) is small, no model can forecast S well from O, and the honest response is better sensing,
not a bigger model. On a simulated world with known S (P-14), this gives the ceiling that NagaHana's
performance is compared against.

The plug-in estimator below (empirical frequencies) is biased upward for small samples. The
Miller–Madow correction adds −(|O|·|S| − |O| − |S| + 1)/(2n) for occupied bins. Use large samples,
or report a bootstrap interval. This implementation is for discrete (or pre-binned) variables; the
binning of continuous observables is itself part of the audit design.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Collection, Hashable, Sequence

from nagahana.governance import decisions


def mutual_information(
    obs: Sequence[Hashable],
    hidden: Sequence[Hashable],
    *,
    enabled_proposals: Collection[str],
    miller_madow: bool,
) -> float:
    """Plug-in estimate of I(O; S) in nats (optionally Miller–Madow corrected). Gated by P-15."""
    decisions.require_proposal("information-audit", enabled_proposals)
    if len(obs) != len(hidden) or not obs:
        raise ValueError("obs and hidden must be non-empty and of equal length")
    n = len(obs)
    joint = Counter(zip(obs, hidden, strict=True))
    po, ps = Counter(obs), Counter(hidden)
    mi = 0.0
    for (o, s), c in joint.items():
        mi += (c / n) * math.log(c * n / (po[o] * ps[s]))
    if miller_madow:
        mi -= (len(joint) - len(po) - len(ps) + 1) / (2 * n)
    return max(mi, 0.0) if miller_madow else mi
