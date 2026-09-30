"""Ingest ports: the passive, source-agnostic input interface (D-32; [Q-03], [Q-15], [Q-22]).

- "we'll never take host agents into account … ultimately have an source-agnostic output interface
  which gives the model input" [Q-03].
- "providing different integration support doesn't account that we'll take them all, its just
  universality in support" [Q-22].
- The data model is the narrow waist: adding a sensor means adding one adapter, and the model is
  unchanged (ARCH §9.2).

Every adapter implements `Source`: it yields `StateUpdate`s, in event-time order with residual
ordering uncertainty tagged (ordering is resolved *before* the model [Q-20]). The protocol is
read-only on purpose. There is no method that writes back to a sensor or the network. Passivity is a
property of the interface, not a convention.

Adapters (templates until stage-1 analysis produces their column maps):
- `csv_flows.CICFlowSource`: CIC-IDS2017/2018 CSV flow records (problem statement).
- `csv_flows.CTU13Source`: CTU-13 NetFlow records (problem statement).
- `pcap.PcapSource`: PCAP files (Scapy/PyShark, approved with D-09); also the forensic-replay upload
  path, which must parse in a sandbox (ARCH §7).
- `kafka.KafkaSource`: live streams (client library held, D-27).
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol

from nagahana.datamodel.records import StateUpdate


class Source(Protocol):
    """A passive producer of state updates."""

    name: str

    def updates(self) -> Iterator[StateUpdate]:
        """Yield state updates in event-time order."""
        ...
