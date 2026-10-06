"""Zeek adapter: TSV and JSON logs -> state updates, through the tables of datamodel/maps/zeek.py.

Formats
-------
TSV (Zeek's ASCII writer): a header of "#" lines, then one record per line.

    #separator \\x09            the field separator, written as a hex escape
    #set_separator ,           separator of set and vector elements
    #empty_field (empty)       an empty string, set or vector
    #unset_field -             a field without a value
    #path conn                 the log stream
    #fields ts uid id.orig_h ...
    #types time string addr ...

Values escape the separators and non-printable bytes as \\xHH. "-" (unset) and "(empty)" are distinct
and keep distinct statuses: unset is NOT_SUPPLIED, empty is OBSERVED with an empty value. A file may
hold several header blocks (concatenated rotations); each new block replaces the context.

JSON (LogAscii::use_json): one object per line with dotted keys ("id.orig_h"); unset fields are absent,
empty sets are []. Numbers are read with their exact digits (timestamps keep nanoseconds). The log
stream comes from the configured `log`, else the record's "_path", else the file name ("conn.log").

Several files can be read together: their records are merged in event-time order (a k-way merge,
one pending record per file), and x509 logs are read first into a bounded cache so that ssl records
carry the facts of their server certificate (AS-691).

Event time per log (AS-300, AS-690): conn ts + duration, dns ts + rtt when the response was seen,
otherwise ts.
"""

from __future__ import annotations

import heapq
import re
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import IO, Any

from nagahana.datamodel.native import record_map
from nagahana.datamodel.status import ObservationStatus
from nagahana.ingest import derive as DV
from nagahana.ingest.config import ZeekConfig
from nagahana.ingest.convert import ConvContext
from nagahana.ingest.core import (
    BoundedLRU,
    IngestStats,
    MalformedRecord,
    RawRecord,
    StreamAdapter,
    UpdateDraft,
    iter_lines,
    json_loads_exact,
    open_source,
)
from nagahana.ingest.mapping import EMPTY, UNSET, Mapper, mapper, set_field
from nagahana.ingest.timeparse import Instant

ADAPTER_VERSION = "1.0.0"
#: Log streams with a mapping table.
LOGS: tuple[str, ...] = (
    "conn", "dns", "http", "ssl", "x509", "files", "notice", "weird", "dhcp", "ssh", "smtp", "rdp", "smb_files",
    "smb_mapping", "kerberos", "ntlm", "dce_rpc", "modbus", "dnp3", "software", "known_hosts", "known_services",
)
#: Logs whose analysers run over TCP only (flow.protocol = 6 when the log has no proto column).
_TCP_LOGS = frozenset({"http", "ssh", "smtp", "rdp", "smb_files", "smb_mapping", "modbus", "dce_rpc", "ntlm"})
_HEX = re.compile(rb"\\x([0-9a-fA-F]{2})")


def _parse_json(line: bytes) -> dict[str, Any]:
    try:
        obj = json_loads_exact(line)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise MalformedRecord("json-decode", str(exc)) from None
    if not isinstance(obj, dict):
        raise MalformedRecord("json-not-object", type(obj).__name__)
    return obj


def unescape(raw: bytes) -> str:
    """Zeek's escaping undone: \\xHH -> the byte; then decoded as UTF-8 (invalid bytes kept as \\xHH)."""
    if b"\\" in raw:
        raw = _HEX.sub(lambda m: bytes.fromhex(m.group(1).decode("ascii")), raw.replace(b"\\\\", b"\\x5c"))
    return raw.decode("utf-8", "backslashreplace")


class _Header:
    """The context of a TSV header block."""

    def __init__(self) -> None:
        self.separator = b"\t"
        self.set_separator = b","
        self.empty = b"(empty)"
        self.unset = b"-"
        self.path: str | None = None
        self.fields: list[str] = []
        self.types: list[str] = []

    def directive(self, line: bytes) -> None:
        if line.startswith(b"#separator"):
            value = line[len(b"#separator"):].strip()
            self.separator = _HEX.sub(lambda m: bytes.fromhex(m.group(1).decode("ascii")), value) or b"\t"
            return
        key, _, value = line[1:].partition(self.separator)
        k = key.decode("ascii", "replace")
        if k == "set_separator":
            self.set_separator = value
        elif k == "empty_field":
            self.empty = value
        elif k == "unset_field":
            self.unset = value
        elif k == "path":
            self.path = value.decode("utf-8", "replace")
        elif k == "fields":
            self.fields = [unescape(x) for x in value.split(self.separator)]
        elif k == "types":
            self.types = [x.decode("ascii", "replace") for x in value.split(self.separator)]

    def parse(self, line: bytes) -> dict[str, Any]:
        """One data line -> {field: text, list of texts, UNSET or EMPTY}."""
        if not self.fields:
            raise MalformedRecord("no-header", "data line before a #fields header")
        parts = line.split(self.separator)
        if len(parts) != len(self.fields):
            raise MalformedRecord("field-count", f"{len(parts)} values for {len(self.fields)} fields")
        out: dict[str, Any] = {}
        for i, (name, value) in enumerate(zip(self.fields, parts, strict=True)):
            typ = self.types[i] if i < len(self.types) else "string"
            if value == self.unset:
                out[name] = UNSET
            elif value == self.empty:
                out[name] = EMPTY
            elif typ.startswith(("set[", "vector[")):
                out[name] = [unescape(x) for x in value.split(self.set_separator)]
            else:
                out[name] = unescape(value)
        return out


class ZeekSource(StreamAdapter):
    """Zeek logs (TSV or JSON) -> state updates. See the module docstring.

    Parameters
    ----------
    source: a log file, a sequence of log files, bytes, or a binary stream.
    log: the log stream when the files do not say ("conn", "dns", ...); None reads it from the data.
    config: `ZeekConfig`.
    """

    name = "zeek"
    source_type = "zeek"
    version = ADAPTER_VERSION

    def __init__(self, source: str | Path | bytes | IO[bytes] | Sequence[str | Path], *, log: str | None = None,
                 config: ZeekConfig | None = None, **kw: Any) -> None:
        self.config = config or ZeekConfig()
        self.paths: list[str | Path] | None = None
        first: Any = source
        if isinstance(source, list | tuple):
            self.paths = list(source)
            if not self.paths:
                raise MalformedRecord("no-input", "empty list of Zeek logs")
            first = self.paths[0]
        super().__init__(first, self.config.common, **kw)
        if self.paths is not None and self.config.common.source_id is None:
            self.source_id = Path(self.paths[0]).parent.name or Path(self.paths[0]).name
        self.log = log
        self.certs: BoundedLRU[str, dict[str, Any]] = BoundedLRU(self.config.join_cache, "x509_join", self.stats)
        self.cert_hosts: BoundedLRU[str, tuple[str, str, int | None]] = BoundedLRU(self.config.join_cache,
                                                                                   "x509_hosts", self.stats)
        self._generic: dict[str, Mapper] = {}
        self._ctx = ConvContext(year=self.config.common.clock.assumed_year,
                                utc_offset_s=None if self.config.common.clock.utc_offset_hours is None
                                else self.config.common.clock.utc_offset_hours * 3600.0)

    # Framing: one file at a time, header context per file.
    def _file_records(self, src: Any) -> Iterator[RawRecord | tuple[str, RawRecord]]:
        stream, location, close = open_source(src)
        header = _Header()
        try:
            for item in iter_lines(stream, location, max_bytes=self.config.common.max_record_bytes):
                if isinstance(item, tuple):
                    yield item
                    continue
                if not item.data.strip():
                    continue
                if item.data.startswith(b"#"):
                    if item.data.startswith(b"#separator"):
                        header = _Header()
                    header.directive(item.data)
                    continue
                item.meta = {"header": header, "file": location}
                yield item
        finally:
            if close:
                stream.close()

    def raw_records(self) -> Iterator[RawRecord | tuple[str, RawRecord]]:
        sources = self.paths if self.paths is not None else [self.source]
        for src in sources:
            yield from self._file_records(src)

    def _log_of(self, raw: RawRecord, values: dict[str, Any]) -> str:
        if self.log:
            return self.log
        header = (raw.meta or {}).get("header")
        if header is not None and header.path:
            return str(header.path)
        path = values.pop("_path", None)
        if isinstance(path, str) and path:
            return path
        stem = Path(str((raw.meta or {}).get("file") or raw.location)).name.split(".")[0]
        return stem

    def values_of(self, raw: RawRecord) -> dict[str, Any]:
        """The record's source values: TSV via its header, JSON as parsed (nested objects flattened)."""
        header = (raw.meta or {}).get("header")
        if raw.data.lstrip().startswith(b"{"):
            obj = _parse_json(raw.data)
            return _flatten(obj)
        if header is None:
            raise MalformedRecord("no-header", "TSV record without a header")
        return header.parse(raw.data)

    def decode(self, raw: RawRecord) -> Iterable[UpdateDraft]:
        values = self.values_of(raw)
        log = self._log_of(raw, values)
        return [self.map_record(log, values, raw)]

    def map_record(self, log: str, values: dict[str, Any], raw: RawRecord) -> UpdateDraft:
        """Map one record of log stream `log`."""
        record_type = f"zeek.{log}"
        d = UpdateDraft(record_type, raw)
        src = f"zeek.{log}"
        if log not in LOGS:
            # No table for this stream: endpoints and time are mapped, every other field is retained.
            m = self._generic.get(log)
            if m is None:
                m = Mapper(record_map("zeek.other"))
                m.extra_prefix = record_type
                self._generic[log] = m
            self._ctx.values = values
            m.apply(values, d, self._ctx, self.stats, source=src, keep_unmapped=True)
            DV.flow_entities(d, self.resolver, service=False)
            self.stats.counters[f"unknown_log:{log}"] += 1
            return d
        ctx = self._ctx
        ctx.values = values
        orig = values.get("id.orig_h") or values.get("client_addr") or values.get("host")
        ctx.ipv6 = isinstance(orig, str) and ":" in orig
        if log == "conn" and _is_icmp(values):
            # Zeek writes the ICMP type in id.orig_p and the code in id.resp_p; ports stay NOT_SUPPLIED.
            values = dict(values)
            for key, fid in (("id.orig_p", "proto.icmp.type"), ("id.resp_p", "proto.icmp.code")):
                v = _int_or_none(values.pop(key, None))
                if v is not None:
                    set_field(d, fid, v, src, self.stats)
        mapper(record_type).apply(values, d, ctx, self.stats, source=src,
                                  keep_unmapped=self.config.common.keep_unmapped)
        getattr(self, f"_after_{log}", self._after_default)(d, values, src)
        return d

    def _after_default(self, d: UpdateDraft, values: dict[str, Any], src: str) -> None:
        log = d.record_type.split(".", 1)[1]
        if log in _TCP_LOGS:
            set_field(d, "flow.protocol", 6, src, self.stats)
        DV.flow_entities(d, self.resolver)

    def _after_conn(self, d: UpdateDraft, values: dict[str, Any], src: str) -> None:
        DV.flow_totals(d, self.stats, src)
        cs = values.get("conn_state")
        if isinstance(cs, str):
            DV.end_reason_from_conn_state(d, cs, self.stats, src)
        hist = values.get("history")
        if isinstance(hist, str) and d.value("flow.protocol") == 6:
            DV.zeek_history_flags(d, hist, self.stats, src)
        if d.time is not None:
            set_field(d, "flow.start_time", d.time.seconds, src, self.stats)
            dur = d.value("flow.duration")
            if dur is not None:
                set_field(d, "flow.end_time", d.time.seconds + float(dur), src, self.stats)
                d.time = _shift(d.time, float(dur))
        DV.flow_entities(d, self.resolver)

    def _after_dns(self, d: UpdateDraft, values: dict[str, Any], src: str) -> None:
        DV.dns_answer_stats(d, self.stats, src)
        DV.dns_name_stats(d, self.stats, src)
        rtt = d.value("proto.dns.rtt")
        if rtt is not None and d.time is not None:
            d.time = _shift(d.time, float(rtt))
        DV.flow_entities(d, self.resolver)

    def _after_http(self, d: UpdateDraft, values: dict[str, Any], src: str) -> None:
        DV.http_uri_length(d, self.stats, src)
        self._after_default(d, values, src)

    def _after_ssl(self, d: UpdateDraft, values: dict[str, Any], src: str) -> None:
        version = values.get("version")
        dtls = isinstance(version, str) and version.upper().startswith("DTLS")
        set_field(d, "flow.protocol", 17 if dtls else 6, src, self.stats)
        fps = values.get("cert_chain_fps")
        if isinstance(fps, list) and fps:
            cert = self.certs.get(fps[0])
            if cert is not None:
                for fid in ("proto.tls.cert_not_before", "proto.tls.cert_not_after", "proto.tls.cert_key_length",
                            "proto.tls.cert_serial", "proto.tls.cert_fingerprint"):
                    if cert.get(fid) is not None:
                        set_field(d, fid, cert[fid], src, self.stats)
                for fid in ("proto.tls.cert_subject", "proto.tls.cert_issuer"):
                    if cert.get(fid) is not None:
                        set_field(d, fid, cert[fid], src, self.stats)
                self.stats.counters["ssl_joined_x509"] += 1
            else:
                self.stats.counters["ssl_without_x509"] += 1
            resp, orig = d.value("flow.dst_ip"), d.value("flow.src_ip")
            if resp is not None and orig is not None:
                port = d.value("flow.dst_port")
                self.cert_hosts[fps[0]] = (str(orig), str(resp), None if port is None else int(port))
        DV.cert_facts(d, self.stats, src)
        DV.flow_entities(d, self.resolver)

    def _after_x509(self, d: UpdateDraft, values: dict[str, Any], src: str) -> None:
        DV.cert_facts(d, self.stats, src)
        fp = d.value("proto.tls.cert_fingerprint")
        if fp is not None:
            self.certs[str(fp)] = {fid: d.value(fid) for fid in (
                "proto.tls.cert_not_before", "proto.tls.cert_not_after", "proto.tls.cert_key_length",
                "proto.tls.cert_serial", "proto.tls.cert_fingerprint", "proto.tls.cert_subject", "proto.tls.cert_issuer")}
            hosts = self.cert_hosts.get(str(fp))
            if hosts is not None:
                server = self.resolver.address(hosts[1])
                d.add_entity(server, "subject")
                if hosts[2] is not None:
                    d.add_entity(self.resolver.service(server, hosts[2], 6), "service")

    def _after_files(self, d: UpdateDraft, values: dict[str, Any], src: str) -> None:
        if d.value("flow.src_ip") is None:
            tx, rx = values.get("tx_hosts"), values.get("rx_hosts")
            if isinstance(tx, list) and tx:
                d.add_entity(self.resolver.address(tx[0]), "initiator")
            if isinstance(rx, list) and rx:
                d.add_entity(self.resolver.address(rx[0]), "responder")
            return
        DV.flow_entities(d, self.resolver, service=False)

    def _after_notice(self, d: UpdateDraft, values: dict[str, Any], src: str) -> None:
        note = d.value("alert.rule")
        if note is not None:
            set_field(d, "alert.signature", DV.name_signature("zeek-notice", str(note)), src, self.stats)
        if d.value("flow.src_ip") is not None:
            DV.flow_entities(d, self.resolver)
        else:
            for key, role in (("src", "initiator"), ("dst", "responder")):
                v = values.get(key)
                if isinstance(v, str):
                    d.add_entity(self.resolver.address(v), role)

    def _after_weird(self, d: UpdateDraft, values: dict[str, Any], src: str) -> None:
        name = d.value("alert.rule")
        if name is not None:
            set_field(d, "alert.signature", DV.name_signature("zeek-weird", str(name)), src, self.stats)
        DV.flow_entities(d, self.resolver, service=False)

    def _after_dhcp(self, d: UpdateDraft, values: dict[str, Any], src: str) -> None:
        set_field(d, "flow.protocol", 17, src, self.stats)
        client = d.value("flow.src_ip") or d.value("proto.dhcp.assigned_addr")
        d.add_entity(self.resolver.address(client), "initiator")
        server = self.resolver.address(d.value("flow.dst_ip"))
        d.add_entity(server, "responder")
        port = d.value("flow.dst_port")
        d.add_entity(self.resolver.service(server, None if port is None else int(port), 17), "service")

    def _after_kerberos(self, d: UpdateDraft, values: dict[str, Any], src: str) -> None:
        mt = d.value("proto.kerberos.msg_type")
        if mt is not None:
            set_field(d, "auth.activity", 3 if int(mt) == 10 else 4, src, self.stats)
        set_field(d, "auth.protocol", 2, src, self.stats)
        ok = d.value("proto.kerberos.success")
        if ok is not None:
            set_field(d, "auth.result", 1 if ok else 2, src, self.stats)
            if ok and d.status("proto.kerberos.error_code") is not ObservationStatus.OBSERVED:
                set_field(d, "proto.kerberos.error_code", 0, src, self.stats)
        DV.flow_entities(d, self.resolver, service=False)
        client = d.value("proto.kerberos.client")
        if client is not None:
            name, _, realm = str(client).rpartition("/") if "/" in str(client) else (str(client), "", "")
            d.add_entity(self.resolver.account(name or str(client), realm or None), "account")
        service = d.value("proto.kerberos.service")
        if service is not None:
            d.add_entity(self.resolver.account(str(service)), "target_account")
        port = d.value("flow.dst_port")
        proto = d.value("flow.protocol")
        b = self.resolver.address(d.value("flow.dst_ip"))
        d.add_entity(self.resolver.service(b, None if port is None else int(port), None if proto is None else int(proto)),
                     "service")

    def _after_ntlm(self, d: UpdateDraft, values: dict[str, Any], src: str) -> None:
        set_field(d, "flow.protocol", 6, src, self.stats)
        set_field(d, "auth.protocol", 1, src, self.stats)
        set_field(d, "auth.activity", 1, src, self.stats)
        DV.flow_entities(d, self.resolver, service=False)
        d.add_entity(self.resolver.account(d.value("auth.user"), d.value("auth.domain")), "account")

    def _after_software(self, d: UpdateDraft, values: dict[str, Any], src: str) -> None:
        d.add_entity(self.resolver.address(d.value("flow.src_ip")), "subject")

    def _after_known_hosts(self, d: UpdateDraft, values: dict[str, Any], src: str) -> None:
        d.add_entity(self.resolver.address(d.value("flow.src_ip")), "subject")

    def _after_known_services(self, d: UpdateDraft, values: dict[str, Any], src: str) -> None:
        host = self.resolver.address(d.value("flow.dst_ip"))
        d.add_entity(host, "subject")
        port, proto = d.value("flow.dst_port"), d.value("flow.protocol")
        d.add_entity(self.resolver.service(host, None if port is None else int(port), None if proto is None else int(proto)),
                     "service")

    def drafts(self) -> Iterator[UpdateDraft]:
        """Drafts of every file, merged in event-time order; x509 logs are read first for the ssl join."""
        if self.paths is None or len(self.paths) == 1:
            yield from super().drafts()
            return
        x509 = [p for p in self.paths if Path(p).name.startswith("x509")]
        stats, self.stats = self.stats, IngestStats()          # pass 1 counts nothing: pass 2 reads these again
        try:
            for p in x509:                                      # pass 1: certificate facts only
                for item in self._file_records(p):
                    if isinstance(item, tuple):
                        continue
                    try:
                        self.map_record("x509", self.values_of(item), item)
                    except MalformedRecord:
                        continue
        finally:
            self.stats = stats
        self.stats.counters["x509_prescanned"] = len(self.certs)
        streams = [self._file_drafts(p) for p in self.paths]
        yield from heapq.merge(*streams, key=lambda d: d.time.seconds if d.time is not None else float("-inf"))

    def _file_drafts(self, path: str | Path) -> Iterator[UpdateDraft]:
        for item in self._file_records(path):
            self.stats.records += 1
            if isinstance(item, tuple):
                self.quarantine.add(item[1], item[0], "framing")
                continue
            try:
                yield from self.decode(item)
            except MalformedRecord as exc:
                self.quarantine.add(item, exc.reason, exc.detail)


def _flatten(obj: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Nested JSON objects to dotted keys (Zeek JSON is flat unless a writer nests it)."""
    out: dict[str, Any] = {}
    for k, v in obj.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "."))
        else:
            out[key] = v
    return out


def _is_icmp(values: dict[str, Any]) -> bool:
    proto = values.get("proto")
    return isinstance(proto, str) and proto.lower() in ("icmp", "icmp6", "icmpv6")


def _int_or_none(v: Any) -> int | None:
    try:
        return int(str(v))
    except (TypeError, ValueError):
        return None


def _shift(t: Instant, seconds: float) -> Instant:
    """An Instant moved by `seconds` (exact when both are given in nanoseconds)."""
    if t.ns is not None:
        ns = t.ns + int(round(seconds * 1e9))
        return Instant(ns / 1e9, ns, t.resolution, t.zone_known)
    return Instant(t.seconds + seconds, None, t.resolution, t.zone_known)


def record_types() -> tuple[str, ...]:
    return tuple(record_map(f"zeek.{log}").record_type for log in LOGS)


__all__ = ["ADAPTER_VERSION", "LOGS", "ZeekSource", "record_types", "unescape"]
