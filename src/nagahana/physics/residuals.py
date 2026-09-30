"""Physics residuals r_c: the "known physics of the communication networks & traffic flow" [A-05].

What a residual is for (D-18, decided 2026-09-29)
-------------------------------------------------
"physics informed acts a boundary/guide to not hallucinate, it helps learn the boundaries that can't
be crossed in reality and so the model won't drift into impossibility hallucination … the same way
we can govern the model's understanding to be limited within the possibility boundary" [A-08].

So a residual measures how far a *model output* lies outside what is physically possible. Model
outputs are reconstructions (CVG-AE/Decoder), imagined states (Forecaster), generated variants
(Generator) and the predicted effects of counters (Advisor). A residual is zero inside the
boundary and grows with the violation. Residuals are not attack detectors: attackers obey physics
too. Whether residuals on *incoming telemetry* may inform trust is a separate, held question (D-25).

Conventions
-----------
- A residual reads named fields (catalogue IDs) from a mapping of tensors of shape [N]
  (N records) and returns r ≥ 0 of shape [N]. Zero means "inside the boundary".
- It declares the fields it needs. The shared term (`term.py`) counts it only on rows where *all*
  of those fields contribute (the mask m_c), so absent data never creates a fake violation (D-41).
- Parameters that depend on the site (e.g. link MTU) are required arguments. No defaults, because
  a wrong default would teach the model a false boundary.

The residuals implemented here are unconditional facts of IP/TCP flow accounting:
- `FlagCountBound`: no TCP flag can be set on more packets than the flow has.
- `MTUBound`: a direction cannot carry more IP bytes than packets × MTU.
- `IATMaxBound`: no gap between a flow's packets can exceed the flow's duration.
Others from the catalogue in diagram 09 (flow conservation at forwarding nodes, RTT floors, Little's
law, OT process limits, protocol state machines) need graph- or site-level context. They are
registered as templates that raise `NotBuiltYet`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Protocol

import torch

from nagahana.core.errors import NotBuiltYet
from nagahana.core.registry import Registry


class Residual(Protocol):
    """A physics residual r_c. See the module docstring for conventions."""

    name: str
    fields: tuple[str, ...]

    def __call__(self, x: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Return r ≥ 0 of shape [N]; zero inside the physical boundary."""
        ...


RESIDUALS: Registry[Callable[..., Residual]] = Registry("physics residual")

_FLAGS = ("syn", "ack", "fin", "rst", "psh", "urg")


@RESIDUALS.register("flag_count_bound", summary="packets carrying a flag ≤ packets in the flow")
class FlagCountBound:
    """r = relu(count_f − (packets_fwd + packets_bwd)) for one TCP flag f.

    A packet either carries a flag or not, so the number of packets carrying flag f cannot exceed
    the number of packets. Example hallucination this blocks: a generated SYN flood with more SYN
    packets than packets.
    """

    def __init__(self, flag: str) -> None:
        if flag not in _FLAGS:
            raise ValueError(f"flag must be one of {_FLAGS}")
        self.flag = flag
        self.name: str = f"flag_count_bound.{flag}"
        self.fields: tuple[str, ...] = (f"flow.flag_count.{flag}", "flow.packets_fwd", "flow.packets_bwd")

    def __call__(self, x: Mapping[str, torch.Tensor]) -> torch.Tensor:
        total = x["flow.packets_fwd"] + x["flow.packets_bwd"]
        return torch.relu(x[f"flow.flag_count.{self.flag}"] - total)


@RESIDUALS.register("mtu_bound", summary="IP bytes in one direction ≤ packets × MTU")
class MTUBound:
    """r = relu(bytes_d − packets_d · MTU) for direction d ∈ {fwd, bwd}.

    `flow.bytes_*` is defined as IP-layer bytes (fields.py). Adapters whose source counts another
    layer must convert, or the boundary is wrong. `mtu` is the largest IP packet size on the path.
    It is site-specific (1500 on standard Ethernet, up to 9000 with jumbo frames), so it is
    required, not defaulted.
    """

    def __init__(self, direction: str, mtu: float) -> None:
        if direction not in ("fwd", "bwd"):
            raise ValueError("direction must be 'fwd' or 'bwd'")
        if mtu <= 0:
            raise ValueError("mtu must be positive")
        self.direction, self.mtu = direction, float(mtu)
        self.name: str = f"mtu_bound.{direction}"
        self.fields: tuple[str, ...] = (f"flow.bytes_{direction}", f"flow.packets_{direction}")

    def __call__(self, x: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return torch.relu(x[f"flow.bytes_{self.direction}"] - x[f"flow.packets_{self.direction}"] * self.mtu)


@RESIDUALS.register("iat_max_bound", summary="largest inter-arrival gap ≤ flow duration")
class IATMaxBound:
    """r = relu(iat_max − duration).

    Every inter-arrival gap lies between the first and the last packet, so none can exceed the
    duration.
    """

    name: str = "iat_max_bound"
    fields: tuple[str, ...] = ("flow.iat_max", "flow.duration")

    def __call__(self, x: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return torch.relu(x["flow.iat_max"] - x["flow.duration"])


def _template(name: str, what: str, waiting_on: tuple[str, ...]) -> None:
    """Register a residual that needs context not modelled yet; building it raises NotBuiltYet."""

    def factory(*_a: object, **_k: object) -> Residual:
        raise NotBuiltYet(what, waiting_on=waiting_on)

    RESIDUALS.register(name, summary=f"template: {what}")(factory)


_template("flow_conservation", "flow conservation at a forwarding node (Σin − Σout − drops)", ("D-04", "stage-1 analysis"))
_template("rtt_floor", "RTT ≥ propagation floor for the path", ("site topology data",))
_template("littles_law", "Little's law L = λW on long-run queue averages", ("site telemetry of queues",))
_template("tcp_state_machine", "RFC-legal TCP state transitions within tolerance", ("stage-1 analysis",))
_template("ot_process_limits", "OT setpoint / rate-of-change limits and polling periods", ("D-34 OT asset data",))
