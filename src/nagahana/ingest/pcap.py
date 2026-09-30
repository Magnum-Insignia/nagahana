"""PCAP adapter (template): packet-level features and the forensic-replay upload path.

Packet-level features (problem statement): TTL and its variance across a session, TCP window size, IP
fragment flags, payload-size distribution, port-scan signatures, retransmission counts
(datamodel/fields.py, level PACKET).

Two uses:
1. **Training/evaluation** on dataset captures (CIC-IDS PCAPs).
2. **Forensic replay** of an uploaded capture ([Q-17]; ARCH §7). The upload is attacker-influenced
   input, and packet parsers have a long vulnerability history. Parse in a sandboxed process with
   resource limits, never in the model's process (ARCH §7 honest limit).

Libraries: Scapy or PyShark (approved with D-09; named in the problem statement). Imported lazily.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from nagahana.core.errors import NotBuiltYet
from nagahana.datamodel.records import StateUpdate


class PcapSource:
    """PCAP → state updates (template)."""

    name = "pcap"

    def __init__(self, path: str | Path, *, sandboxed: bool) -> None:
        self.path = Path(path)
        self.sandboxed = sandboxed

    def updates(self) -> Iterator[StateUpdate]:
        raise NotBuiltYet("PCAP session reconstruction → StateUpdate", waiting_on=("stage-1 analysis", "sandbox design"))
