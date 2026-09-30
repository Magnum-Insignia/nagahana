"""CSV flow adapters for CIC-IDS2017/2018 and CTU-13 (templates).

The problem statement expects "a feature extraction pipeline that ingests CIC-IDS-2018 or CTU-13 CSV
flow records and/or raw PCAP files … and outputs a timestamped, normalised feature matrix covering
both flow-level and packet-level attributes".

Why these are templates: the column maps (source column → catalogue field, units, direction
conventions, label mapping to attack families) are outputs of **stage 1, deep data analysis** [Q-02].
Known issues of these datasets must be handled there, not guessed here. CICIDS2017 has documented
labelling and flow-construction errors (Engelen, Rimmer & Joosen, IEEE SPW 2021). CSE-CIC-IDS2018
and CTU-13 differ in flow definitions (CICFlowMeter bidirectional flows vs Argus/NetFlow
unidirectional flows).

Each adapter will:
1. map columns to catalogue fields (datamodel/fields.py), marking every field the source lacks as
   NOT_SUPPLIED (never 0);
2. attach provenance (source, adapter version, row hash) and ordering info;
3. map entity identifiers (IPs, ports) to typed `EntityRef`s;
4. keep labels outside the StateUpdate. Labels are evaluation data, not observations.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from nagahana.core.errors import NotBuiltYet
from nagahana.datamodel.records import StateUpdate


class CICFlowSource:
    """CIC-IDS2017 / CSE-CIC-IDS2018 CSV flows (template)."""

    name = "cic-flows"

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def updates(self) -> Iterator[StateUpdate]:
        raise NotBuiltYet("CIC CSV column map → StateUpdate", waiting_on=("stage-1 analysis",))


class CTU13Source:
    """CTU-13 NetFlow records (template)."""

    name = "ctu13-netflow"

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def updates(self) -> Iterator[StateUpdate]:
        raise NotBuiltYet("CTU-13 NetFlow map → StateUpdate", waiting_on=("stage-1 analysis",))
