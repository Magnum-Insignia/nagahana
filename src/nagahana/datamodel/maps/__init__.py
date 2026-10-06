"""Field-by-field mapping tables of every supported telemetry format (see `datamodel/native.py`).

Each module transcribes one family of formats; `ALL_MAPS` is the registry the catalogue is generated
from. Adding a format means adding its `RecordMap`s here; the catalogue, the mapping engine
(ingest/mapping.py) and the documentation (docs/ingest.md) follow from the table.
"""

from __future__ import annotations

from nagahana.datamodel.maps.devtelemetry import DEVTELEMETRY_MAPS
from nagahana.datamodel.maps.eventlogs import EVENTLOG_MAPS
from nagahana.datamodel.maps.flowexport import FLOWEXPORT_MAPS
from nagahana.datamodel.maps.ocsf import OCSF_MAPS
from nagahana.datamodel.maps.snort import SNORT_MAPS
from nagahana.datamodel.maps.suricata import SURICATA_MAPS
from nagahana.datamodel.maps.wireshark import WIRESHARK_MAPS
from nagahana.datamodel.maps.zeek import ZEEK_MAPS
from nagahana.datamodel.native import RecordMap

#: Every record map, grouped by source family.
ALL_MAPS: tuple[RecordMap, ...] = (
    *ZEEK_MAPS, *SURICATA_MAPS, *SNORT_MAPS, *FLOWEXPORT_MAPS, *WIRESHARK_MAPS, *EVENTLOG_MAPS, *DEVTELEMETRY_MAPS,
    *OCSF_MAPS,
)

__all__ = ["ALL_MAPS"]
