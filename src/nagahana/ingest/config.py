"""Ingest configuration: typed dataclasses, the single source of truth for every adapter setting.

Each adapter takes one frozen config dataclass. `CommonConfig` holds what every adapter shares (identity
of the source, the monitored network, clock handling, reordering, quarantine, bounded caches); each
adapter config embeds it as `common` and adds its own settings. The YAML files under conf/ingest/ are
generated from these classes (`write_yaml`), and `load` reads a YAML file back into the class,
rejecting unknown keys and wrong types, so a file and its class cannot drift apart.

Every default is either a definition (a protocol constant) or a recorded assumption (AS-670 to
AS-719, docs/assumptions/ingest.md); the assumption is named next to the field.
"""

from __future__ import annotations

import dataclasses
import types
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

import yaml

from nagahana.core.errors import ConfigMissing, InvariantViolation

#: RFC 1918 private IPv4, RFC 4193 unique-local IPv6, link-local (RFC 3927, RFC 4291) and loopback: the
#: monitored network when none is configured (as ingest/csv_flows.DEFAULT_INTERNAL_NETWORKS).
DEFAULT_INTERNAL: tuple[str, ...] = (
    "10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16", "fc00::/7", "169.254.0.0/16", "fe80::/10", "127.0.0.0/8", "::1/128",
)


@dataclass(frozen=True)
class ClockConfig:
    """How source timestamps become UTC.

    utc_offset_hours: offset east of UTC of timestamps written without a zone (RFC 3164 syslog, Snort,
        CEF without zone); None reads them as UTC and marks clock_quality "timezone-unverified" (AS-698).
    assumed_year: year of timestamps written without one (RFC 3164, Snort without -y); None refuses
        them with a counted reason rather than guessing (AS-698).
    """

    utc_offset_hours: float | None = None
    assumed_year: int | None = None


@dataclass(frozen=True)
class ReorderConfig:
    """Bounded event-time reordering of a stream (AS-671).

    window_s: records are held until the newest event time seen is this much later than theirs, then
        emitted in event-time order. A record older than what was already emitted is passed through with
        its lateness as reorder uncertainty (tagged, not hidden). 0 keeps the source order.
    max_records: hard bound on the records held; beyond it the oldest is emitted early.
    """

    window_s: float = 300.0
    max_records: int = 200_000


@dataclass(frozen=True)
class QuarantineConfig:
    """Where malformed records go (never silently dropped; AS-672).

    max_samples: quarantined records kept in memory for inspection (counts are kept for all).
    max_raw_bytes: bytes of each quarantined raw record kept.
    path: JSON Lines file receiving every quarantined record (reason, location, base64 bytes); None keeps
        only the counts and the in-memory samples.
    """

    max_samples: int = 1000
    max_raw_bytes: int = 65_536
    path: str | None = None


@dataclass(frozen=True)
class CommonConfig:
    """Settings every adapter shares.

    source_id: name recorded in provenance; None uses the file name or stream name.
    sensor_id: the sensor that observed the traffic or produced the log; None leaves it unrecorded.
    internal_networks: CIDR blocks of the monitored network (entity kinds host / external).
    max_record_bytes: a raw record (a line, a JSON element, a binary record) longer than this is
        quarantined without being parsed (bounded memory; AS-673).
    max_entity_cache: addresses whose entity kind is cached (LRU, evictions counted; AS-674).
    keep_unmapped: retain source fields that no mapping table lists as attributes (True: every field is
        mapped or retained).
    """

    source_id: str | None = None
    sensor_id: str | None = None
    internal_networks: tuple[str, ...] = DEFAULT_INTERNAL
    clock: ClockConfig = field(default_factory=ClockConfig)
    reorder: ReorderConfig = field(default_factory=ReorderConfig)
    quarantine: QuarantineConfig = field(default_factory=QuarantineConfig)
    max_record_bytes: int = 4 * 1024 * 1024
    max_entity_cache: int = 1_000_000
    keep_unmapped: bool = True


@dataclass(frozen=True)
class ZeekConfig:
    """Zeek TSV and JSON logs. join_cache: x509 certificates remembered for the ssl join (AS-691)."""

    common: CommonConfig = field(default_factory=CommonConfig)
    join_cache: int = 100_000


@dataclass(frozen=True)
class SuricataConfig:
    """Suricata EVE JSON. stats_sources: sensors whose cumulative counters are tracked for deltas."""

    common: CommonConfig = field(default_factory=CommonConfig)
    stats_sources: int = 10_000


@dataclass(frozen=True)
class SnortConfig:
    """Snort outputs. pending_packets: unified2 packet records kept per pending event (AS-705)."""

    common: CommonConfig = field(default_factory=CommonConfig)
    pending_packets: int = 64


@dataclass(frozen=True)
class FlowExportConfig:
    """NetFlow v5, v9 and IPFIX collection (RFC 3954, RFC 7011).

    template_lifetime_s: a template not refreshed for this long expires (RFC 7011 section 8.4; AS-675).
    max_templates: templates and options templates cached over all exporters (LRU; evictions counted).
    pending_sets: data sets kept while their template has not arrived (RFC 3954 section 9; AS-675).
    pending_max_age_s: export-time age after which a pending data set is quarantined.
    max_flows: flows whose delta counters are accumulated into running totals (AS-696).
    max_sampler_entries: sampler options records remembered per exporter and domain.
    udp_ports: UDP destination ports read as flow export when the input is a packet capture.
    """

    common: CommonConfig = field(default_factory=CommonConfig)
    template_lifetime_s: float = 1800.0
    max_templates: int = 65_536
    pending_sets: int = 10_000
    pending_max_age_s: float = 300.0
    max_flows: int = 1_000_000
    max_sampler_entries: int = 4096
    udp_ports: tuple[int, ...] = (2055, 2056, 4739, 9995, 9996)


@dataclass(frozen=True)
class SFlowConfig:
    """sFlow v5 collection. max_sources: agent sources whose counters are tracked for deltas."""

    common: CommonConfig = field(default_factory=CommonConfig)
    max_sources: int = 100_000
    udp_ports: tuple[int, ...] = (6343,)


@dataclass(frozen=True)
class PcapLimits:
    """Bounds of the PCAP adapter's per-address and per-flow state (spoofed-source floods; AS-676).

    max_flows: open flows tracked; the least recently active is ended (end reason lack_of_resources).
    max_scan_pairs: (source, destination) pairs with port-access memory.
    max_addresses: addresses in each identity table (sending, groups, link-local, ARP bindings,
        external peers per link address, DNS clients).
    max_facts: alias, role and name facts kept for the fact tables.
    max_input_bytes: largest capture accepted (sandboxed or not).
    """

    max_flows: int = 1_000_000
    max_scan_pairs: int = 1_000_000
    max_addresses: int = 1_000_000
    max_facts: int = 1_000_000
    max_input_bytes: int = 64 * 1024 ** 3


@dataclass(frozen=True)
class SandboxLimits:
    """Limits of a sandboxed parse (ARCH section 7: uploaded captures are attacker-influenced; AS-677).

    max_input_bytes: input files larger than this are refused before parsing.
    max_output_bytes: the parse's serialised result may not exceed this.
    cpu_seconds: CPU time of the parsing process (POSIX RLIMIT_CPU).
    wall_seconds: wall-clock budget; the process is killed when it is exceeded (all platforms).
    memory_bytes: address-space limit of the parsing process (POSIX RLIMIT_AS).
    """

    max_input_bytes: int = 8 * 1024 ** 3
    max_output_bytes: int = 32 * 1024 ** 3
    cpu_seconds: int = 3600
    wall_seconds: float = 7200.0
    memory_bytes: int = 32 * 1024 ** 3


@dataclass(frozen=True)
class WiresharkConfig:
    """tshark JSON / EK and PyShark dissections.

    tshark_path: tshark executable for PyShark (None: PyShark's own search).
    display_filter: optional Wireshark display filter applied by PyShark.
    """

    common: CommonConfig = field(default_factory=CommonConfig)
    tshark_path: str | None = None
    display_filter: str | None = None


@dataclass(frozen=True)
class EventLogConfig:
    """CEF, LEEF, syslog, Windows Security events, Linux auth.log and journald."""

    common: CommonConfig = field(default_factory=CommonConfig)
    case_insensitive_accounts: bool = True


@dataclass(frozen=True)
class GnmiConfig:
    """gNMI interface telemetry. max_interfaces: (target, interface) pairs tracked for deltas and status."""

    common: CommonConfig = field(default_factory=CommonConfig)
    max_interfaces: int = 100_000


@dataclass(frozen=True)
class KafkaConfig:
    """Kafka consumption and production of the data model (datamodel.md item 4).

    client: "confluent-kafka" or "kafka-python" (D-27 is held; the default is AS-678).
    bootstrap_servers: broker list "host:port,host:port"; required for a real client.
    group_id: consumer group.
    topics: topics consumed.
    auto_offset_reset: where a new group starts ("earliest" or "latest").
    max_in_flight: records handed out and not yet acknowledged before partitions are paused
        (backpressure, AS-679); resume_below: resume when the backlog falls below this.
    poll_timeout_s: one poll's wait.
    commit_interval_s: acknowledged offsets are committed at least this often (and on close).
    produce_queue: records the producer may hold before it waits for deliveries.
    flush_timeout_s: how long close() waits for outstanding deliveries.
    client_options: extra client settings passed through unchanged (security, compression ...).
    """

    client: str = "confluent-kafka"
    bootstrap_servers: str | None = None
    group_id: str = "nagahana-ingest"
    topics: tuple[str, ...] = ()
    auto_offset_reset: str = "earliest"
    max_in_flight: int = 10_000
    resume_below: int = 5_000
    poll_timeout_s: float = 1.0
    commit_interval_s: float = 5.0
    produce_queue: int = 100_000
    flush_timeout_s: float = 30.0
    client_options: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class OcsfConfig:
    """OCSF export and import (AS-703)."""

    common: CommonConfig = field(default_factory=CommonConfig)
    schema_version: str = "1.3.0"
    product_name: str = "NagaHana"
    vendor_name: str = "NagaHana"
    include_raw: bool = False


@dataclass(frozen=True)
class CstsConfig:
    """CSTS export and import (AS-706)."""

    common: CommonConfig = field(default_factory=CommonConfig)
    substrate_version: str = "nagahana-csts-1"


#: Adapter name -> its config class (the YAML file name is the key).
CONFIGS: dict[str, type] = {
    "common": CommonConfig, "zeek": ZeekConfig, "suricata": SuricataConfig, "snort": SnortConfig,
    "netflow-ipfix": FlowExportConfig, "sflow": SFlowConfig, "pcap": PcapLimits, "sandbox": SandboxLimits,
    "wireshark": WiresharkConfig, "eventlog": EventLogConfig, "gnmi": GnmiConfig, "kafka": KafkaConfig,
    "ocsf": OcsfConfig, "csts": CstsConfig,
}

C = TypeVar("C")


def to_dict(cfg: Any) -> dict[str, Any]:
    """A config as plain YAML-ready data (tuples as lists, nested configs as mappings)."""
    def plain(v: Any) -> Any:
        if dataclasses.is_dataclass(v) and not isinstance(v, type):
            return {f.name: plain(getattr(v, f.name)) for f in dataclasses.fields(v)}
        if isinstance(v, tuple):
            return [plain(x) for x in v]
        return v
    return plain(cfg)


def _coerce(tp: Any, value: Any, where: str) -> Any:
    """`value` as type `tp` (a dataclass, tuple[...], X | None, or a scalar type)."""
    origin = typing.get_origin(tp)
    if origin in (types.UnionType, typing.Union):
        args = typing.get_args(tp)
        if value is None and type(None) in args:
            return None
        for a in args:
            if a is type(None):
                continue
            try:
                return _coerce(a, value, where)
            except InvariantViolation:
                continue
        raise InvariantViolation(f"{where}: {value!r} does not fit {tp}")
    if dataclasses.is_dataclass(tp):
        if not isinstance(value, dict):
            raise InvariantViolation(f"{where}: expected a mapping")
        return from_dict(tp, value, where=where)
    if origin is tuple:
        if not isinstance(value, list | tuple):
            raise InvariantViolation(f"{where}: expected a list")
        args = typing.get_args(tp)
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_coerce(args[0], x, f"{where}[{i}]") for i, x in enumerate(value))
        if len(args) != len(value):
            raise InvariantViolation(f"{where}: expected {len(args)} items")
        return tuple(_coerce(a, x, f"{where}[{i}]") for i, (a, x) in enumerate(zip(args, value, strict=True)))
    if tp is float:
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise InvariantViolation(f"{where}: expected a number, got {value!r}")
        return float(value)
    if tp is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise InvariantViolation(f"{where}: expected an integer, got {value!r}")
        return value
    if tp is bool:
        if not isinstance(value, bool):
            raise InvariantViolation(f"{where}: expected true or false, got {value!r}")
        return value
    if tp is str:
        if not isinstance(value, str):
            raise InvariantViolation(f"{where}: expected text, got {value!r}")
        if value.strip() == "???":
            raise ConfigMissing(f"{where}: undecided value '???'")
        return value
    raise InvariantViolation(f"{where}: unsupported config type {tp}")


def from_dict(cls: type[C], data: dict[str, Any], *, where: str = "") -> C:
    """Build config `cls` from a mapping; unknown keys and wrong types raise, missing keys keep defaults."""
    hints = typing.get_type_hints(cls)
    names = {f.name for f in dataclasses.fields(cls)}          # type: ignore[arg-type]
    unknown = set(data) - names
    if unknown:
        raise InvariantViolation(f"{where or cls.__name__}: unknown keys {sorted(unknown)}")
    kwargs = {k: _coerce(hints[k], v, f"{where or cls.__name__}.{k}") for k, v in data.items()}
    return cls(**kwargs)


def to_yaml(cfg: Any) -> str:
    """YAML text of a config (keys in field order)."""
    header = f"# Generated from nagahana.ingest.config.{type(cfg).__name__}; edit the dataclass, not this file.\n"
    return header + yaml.safe_dump(to_dict(cfg), sort_keys=False, default_flow_style=False)


def write_yaml(directory: str | Path) -> list[Path]:
    """Write the default YAML of every config class into `directory` (one file per adapter)."""
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    out = []
    for name, cls in CONFIGS.items():
        p = d / f"{name}.yaml"
        p.write_text(to_yaml(cls()), encoding="utf-8", newline="\n")
        out.append(p)
    return out


def load(name: str, path: str | Path) -> Any:
    """Read conf/ingest/<name>.yaml into its config class (validated)."""
    cls = CONFIGS[name]
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise InvariantViolation(f"{path}: expected a mapping at the top level")
    return from_dict(cls, data)


def replace_common(cfg: C, **changes: Any) -> C:
    """A copy of an adapter config with fields of its `common` part replaced."""
    common = dataclasses.replace(cfg.common, **changes)        # type: ignore[attr-defined]
    return dataclasses.replace(cfg, common=common)              # type: ignore[type-var]


__all__ = [
    "CONFIGS", "DEFAULT_INTERNAL", "ClockConfig", "CommonConfig", "CstsConfig", "EventLogConfig", "FlowExportConfig",
    "GnmiConfig", "KafkaConfig", "OcsfConfig", "PcapLimits", "QuarantineConfig", "ReorderConfig", "SFlowConfig",
    "SandboxLimits", "SnortConfig", "SuricataConfig", "WiresharkConfig", "ZeekConfig", "from_dict", "load",
    "replace_common", "to_dict", "to_yaml", "write_yaml",
]
