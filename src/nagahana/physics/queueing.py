"""Queueing law: Little's law L = lambda W with its exact finite-window allowance (Little 1961).

The law
-------
For any queueing system (a server's open connections, a device's packet buffer, an OT polling queue) over
an observation window [t0, t1] of length T, with customers i arriving at a_i and departing at d_i:

    L    = (1 / T) integral over [t0, t1] of N(t) dt   = (1 / T) sum_i |[a_i, d_i] intersect [t0, t1]|
    lam  = (number of arrivals in [t0, t1)) / T
    W    = mean over those arrivals of (d_i - a_i)

Little (Operations Research 9(3), 1961) proved L = lam W for long-run averages. Over a finite window the
identity is exact up to the customers that straddle the window's ends:

    L - lam W = (1 / T) [ sum over customers present at t0 of their time inside the window
                          - sum over arrivals in the window of their time after t1 ],

so |L - lam W| <= A with the allowance
    A = (1 / T) [ sum_carried-in |[a_i, d_i] intersect [t0, t1]| + sum_arrivals max(0, d_i - t1) ]
(Kim and Whitt, "Statistical analysis with Little's law", Operations Research 61(4), 2013, discuss these
edge effects; citation to verify). For a window that is empty at both ends A = 0 and the law is exact.

`queue_statistics` computes (L, lam, W, A) exactly from intervals; `LittlesLaw` is the residual
r = relu(|L - lam W| - A) over per-system aggregates (the occupancy, arrival-rate, sojourn and allowance
fields of the site's queue telemetry or of model outputs), which is zero exactly when the three quantities
are consistent.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch

from nagahana.core.errors import InvariantViolation
from nagahana.physics.residuals import RESIDUALS

#: Aggregate field names of a queueing system over a window (site telemetry or model outputs).
QUEUE_OCCUPANCY, QUEUE_ARRIVAL_RATE, QUEUE_SOJOURN, QUEUE_ALLOWANCE = (
    "queue.occupancy_mean", "queue.arrival_rate", "queue.sojourn_mean", "queue.edge_allowance")


@dataclass(frozen=True)
class QueueStatistics:
    """L, lam, W and the edge allowance A of one window (module docstring), as float64 tensors."""

    occupancy: torch.Tensor
    arrival_rate: torch.Tensor
    sojourn: torch.Tensor
    allowance: torch.Tensor


def queue_statistics(arrival: torch.Tensor, departure: torch.Tensor, t0: float, t1: float) -> QueueStatistics:
    """Exact L, lam, W and A of the customers [a_i, d_i] over the window [t0, t1] (module docstring).

    arrival, departure: float [n] with departure >= arrival. Customers that never overlap the window are
    ignored. W is the mean over arrivals in [t0, t1) (0 when there are none).
    """
    if not t1 > t0:
        raise InvariantViolation("the window needs t1 > t0")
    a, d = arrival.double(), departure.double()
    if a.shape != d.shape or bool((d < a).any()):
        raise InvariantViolation("arrival and departure must share a shape and satisfy departure >= arrival")
    span = t1 - t0
    inside = (torch.minimum(d, torch.full_like(d, t1)) - torch.maximum(a, torch.full_like(a, t0))).clamp_min(0.0)
    occupancy = inside.sum() / span
    arrived = (a >= t0) & (a < t1)
    n_arr = arrived.sum().to(torch.float64)
    sojourn = torch.where(n_arr > 0, ((d - a) * arrived).sum() / n_arr.clamp_min(1.0), torch.zeros((), dtype=torch.float64))
    carried_in = (a < t0) & (d > t0)
    allowance = ((inside * carried_in).sum() + ((d - t1).clamp_min(0.0) * arrived).sum()) / span
    return QueueStatistics(occupancy=occupancy, arrival_rate=n_arr / span, sojourn=sojourn, allowance=allowance)


@RESIDUALS.register("littles_law", summary="occupancy = arrival rate x sojourn, within the window's edge allowance")
class LittlesLaw:
    """r = relu(|L - lam W| - A) over per-system aggregates (module docstring).

    Parameters
    ----------
    occupancy, arrival_rate, sojourn, allowance: field names of the aggregates (defaults: the `QUEUE_*` names).
    """

    def __init__(self, occupancy: str = QUEUE_OCCUPANCY, arrival_rate: str = QUEUE_ARRIVAL_RATE,
                 sojourn: str = QUEUE_SOJOURN, allowance: str = QUEUE_ALLOWANCE) -> None:
        self.names = (occupancy, arrival_rate, sojourn, allowance)
        self.name: str = "littles_law"
        self.fields: tuple[str, ...] = self.names

    def __call__(self, x: Mapping[str, torch.Tensor]) -> torch.Tensor:
        occ, lam, w, a = (x[n] for n in self.names)
        return torch.relu(torch.abs(occ - lam * w) - a)

    def bound(self, x: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return x[self.names[0]]
