"""Registry of source adapters: one entry per kind of passive telemetry (D-32, ARCH §9.2).

The data model is the narrow waist: every source maps its records into the same `StateUpdate`, so
adding a sensor is adding one adapter here and nothing else changes.

What is built
-------------
The PCAP adapter (`pcap.PcapSource`) and the CSV adapters (`csv_flows`: CICFlowMeter CSVs of
CIC-IDS2017/2018, CTU-13 binetflow, CIC-IoT-2023) are implemented. Every other entry is a stub: it names the
source and what one record of it is, and its `updates()` raises `NotBuiltYet`. A stub is registered
so the superset is visible and checkable, not to suggest it works. `adapters()` reports, for each
entry, whether it is implemented; callers must show that flag and never present a stub as working.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nagahana.core.errors import NotBuiltYet
from nagahana.core.registry import Registry
from nagahana.datamodel.records import StateUpdate
from nagahana.ingest.csv_flows import CICFlowSource, CICIoT2023Source, CTU13Source
from nagahana.ingest.kafka import KafkaSource
from nagahana.ingest.pcap import PcapSource

SOURCES: Registry[Any] = Registry("telemetry source")


class _Stub:
    """A source whose adapter is not built. Subclasses set `name`, `record` and `waiting_on`."""

    name = ""
    record = ""
    implemented = False
    waiting_on: tuple[str, ...] = ("stage-1 analysis",)

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else None

    def updates(self) -> Iterator[StateUpdate]:
        raise NotBuiltYet(f"{self.name} adapter ({self.record} → StateUpdate)", waiting_on=self.waiting_on)


class ZeekSource(_Stub):
    name = "zeek"
    record = "one log line (conn, dns, ssl, kerberos, … logs)"


class SuricataSource(_Stub):
    name = "suricata"
    record = "one EVE JSON event (flow, alert, protocol metadata)"


class SnortSource(_Stub):
    name = "snort"
    record = "one alert or log entry"


class FlowExportSource(_Stub):
    name = "netflow-ipfix"
    record = "one NetFlow v5/v9 or IPFIX flow record"


class SFlowSource(_Stub):
    name = "sflow"
    record = "one sFlow sample (sampled packet header or counter)"


class DeviceTelemetrySource(_Stub):
    name = "device-telemetry"
    record = "one router, switch or TAP telemetry record (interface counters, mirrored-port metadata)"


SOURCES.register("pcap", summary="Packet capture (pcap, pcapng): one packet, carrying the running state of its flow")(PcapSource)
SOURCES.register("zeek", summary=ZeekSource.record)(ZeekSource)
SOURCES.register("suricata", summary=SuricataSource.record)(SuricataSource)
SOURCES.register("snort", summary=SnortSource.record)(SnortSource)
SOURCES.register("netflow-ipfix", summary=FlowExportSource.record)(FlowExportSource)
SOURCES.register("sflow", summary=SFlowSource.record)(SFlowSource)
SOURCES.register("csv-cic", summary="CICFlowMeter CSV: one bidirectional flow row")(CICFlowSource)
SOURCES.register("csv-ctu13", summary="CTU-13 binetflow CSV: one flow row")(CTU13Source)
SOURCES.register("csv-ciciot2023", summary="CIC-IoT-2023 CSV: one packet-window row (no addresses, no times)")(CICIoT2023Source)
SOURCES.register("device-telemetry", summary=DeviceTelemetrySource.record)(DeviceTelemetrySource)
SOURCES.register("kafka", summary="Live stream of records of any source above", requires=("D-27",))(KafkaSource)


@dataclass(frozen=True)
class AdapterInfo:
    """One registered adapter, as shown to people."""

    name: str
    summary: str
    implemented: bool


def adapters() -> tuple[AdapterInfo, ...]:
    """Every registered adapter with its build state, implemented ones first."""
    out = [AdapterInfo(e.name, e.summary, bool(getattr(e.factory, "implemented", False))) for e in SOURCES.entries()]
    return tuple(sorted(out, key=lambda a: (not a.implemented, a.name)))
