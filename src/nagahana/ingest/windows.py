"""Windows Security events (XML or JSON exports) -> state updates (tables in datamodel/maps/eventlogs.py).

Mapped event IDs: 4624 logon, 4625 failed logon, 4634 logoff, 4648 explicit-credential logon, 4672
special privileges, 4768 Kerberos TGT request, 4769 service ticket request, 4771 pre-authentication
failure, 4776 credential validation. Other event IDs keep their System fields mapped and their
EventData retained.

XML: the event schema of http://schemas.microsoft.com/win/2004/08/events/event, one <Event> per record.
A file may hold an <Events> wrapper or concatenated <Event> elements (wevtutil qe /f:xml); each
<Event> element is cut out of the stream and parsed alone, so memory is bounded by one event. A
document type declaration is refused before parsing (no entity expansion of attacker-supplied input).

JSON (one object per line, or a JSON array), in the layouts of common exporters:
    {"Event": {"System": {...}, "EventData": {"Data": [{"@Name": ..., "#text": ...}]}}}   python-evtx
    {"Event": {"System": {...}, "EventData": {"Name": "value", ...}}}                    evtx_dump
    {"winlog": {"event_id": ..., "event_data": {...}, "computer_name": ...}, "@timestamp": ...}  Winlogbeat
    {"EventID": ..., "Hostname": ..., "EventTime": ..., "<EventData name>": ...}         NXLog
Windows writes "-" for an EventData field without a value; it is NOT_SUPPLIED.
"""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import IO, Any

from nagahana.ingest.config import EventLogConfig
from nagahana.ingest.core import MalformedRecord, RawRecord, UpdateDraft, iter_json_array, iter_lines, open_source
from nagahana.ingest.mapping import UNSET, mapper, set_field
from nagahana.ingest.syslog import LineAdapter
from nagahana.ingest.timeparse import TimeParseError, rfc3339

ADAPTER_VERSION = "1.0.0"
NS = "{http://schemas.microsoft.com/win/2004/08/events/event}"
EVENT_IDS: tuple[int, ...] = (4624, 4625, 4634, 4648, 4672, 4768, 4769, 4771, 4776)
_EVENT_START = re.compile(rb"<Event[\s>]")
_EVENT_END = b"</Event>"
_DTD = re.compile(rb"<!DOCTYPE|<!ENTITY", re.IGNORECASE)
_LOOPBACK = frozenset({"127.0.0.1", "::1", "0.0.0.0", "::"})
#: Status codes of a successful Kerberos request.
_KRB_OK = 0


def _strip(tag: str) -> str:
    return tag.split("}", 1)[1] if "}" in tag else tag


def flatten_xml(elem: ET.Element) -> dict[str, Any]:
    """System and EventData of an <Event> element as {"System.X[.attr]": value, "EventData.Name": value}."""
    out: dict[str, Any] = {}
    system = elem.find(f"{NS}System")
    if system is None:
        system = elem.find("System")
    if system is not None:
        for child in system:
            tag = _strip(child.tag)
            text = (child.text or "").strip()
            if text:
                out[f"System.{tag}"] = text
            for attr, value in child.attrib.items():
                out[f"System.{tag}.{_strip(attr)}"] = value
    data = elem.find(f"{NS}EventData")
    if data is None:
        data = elem.find("EventData")
    if data is not None:
        for k, d in enumerate(list(data)):
            name = d.attrib.get("Name") or f"Data_{k}"
            out[f"EventData.{name}"] = (d.text or "")
    user = elem.find(f"{NS}UserData")
    if user is not None:
        out["UserData"] = ET.tostring(user, encoding="unicode")
    rendering = elem.find(f"{NS}RenderingInfo")
    if rendering is not None:
        for child in rendering:
            if (child.text or "").strip():
                out[f"RenderingInfo.{_strip(child.tag)}"] = child.text.strip()       # type: ignore[union-attr]
    return out


def flatten_json(obj: dict[str, Any]) -> dict[str, Any]:
    """The JSON layouts of the module docstring, to the same keys as `flatten_xml`."""
    out: dict[str, Any] = {}
    if isinstance(obj.get("Event"), dict):
        obj = obj["Event"]
    if isinstance(obj.get("System"), dict):
        for k, v in obj["System"].items():
            key = k.lstrip("@#")
            if isinstance(v, dict):
                for a, av in v.items():
                    if a in ("#text", "value"):
                        out[f"System.{key}"] = av
                    else:
                        out[f"System.{key}.{a.lstrip('@#')}"] = av
            else:
                out[f"System.{key}"] = v
        ed = obj.get("EventData")
        if isinstance(ed, dict):
            data = ed.get("Data")
            if isinstance(data, list):
                for k, item in enumerate(data):
                    if isinstance(item, dict):
                        out[f"EventData.{item.get('@Name') or item.get('Name') or f'Data_{k}'}"] = item.get("#text", item.get("value", ""))
                    else:
                        out[f"EventData.Data_{k}"] = item
            else:
                for k, v in ed.items():
                    out[f"EventData.{k}"] = v
        return out
    win = obj.get("winlog")
    if isinstance(win, dict):
        out["System.EventID"] = win.get("event_id")
        out["System.Computer"] = win.get("computer_name")
        out["System.Channel"] = win.get("channel")
        out["System.Provider.Name"] = win.get("provider_name")
        out["System.Provider.Guid"] = win.get("provider_guid")
        out["System.EventRecordID"] = win.get("record_id")
        out["System.Task"] = win.get("task")
        out["System.Opcode"] = win.get("opcode")
        out["System.Version"] = win.get("version")
        proc = win.get("process")
        if isinstance(proc, dict):
            out["System.Execution.ProcessID"] = proc.get("pid")
            th = proc.get("thread")
            if isinstance(th, dict):
                out["System.Execution.ThreadID"] = th.get("id")
        if obj.get("@timestamp") is not None:
            out["System.TimeCreated.SystemTime"] = obj["@timestamp"]
        for k, v in (win.get("event_data") or {}).items():
            out[f"EventData.{k}"] = v
        return {k: v for k, v in out.items() if v is not None}
    if "EventID" in obj:
        rename = {"EventID": "System.EventID", "Hostname": "System.Computer", "Channel": "System.Channel",
                  "SourceName": "System.Provider.Name", "ProviderGuid": "System.Provider.Guid",
                  "RecordNumber": "System.EventRecordID", "EventTime": "System.TimeCreated.SystemTime",
                  "ProcessID": "System.Execution.ProcessID", "ThreadID": "System.Execution.ThreadID",
                  "Keywords": "System.Keywords", "Task": "System.Task", "Version": "System.Version",
                  "Opcode": "System.Opcode"}
        for k, v in obj.items():
            out[rename.get(k, f"EventData.{k}")] = v
        return out
    raise MalformedRecord("unknown-windows-json", "no Event, System, winlog or EventID key")


class WindowsEventSource(LineAdapter):
    """Windows Security events (XML or JSON) -> state updates. See the module docstring.

    `utc_offset_hours` of the clock config applies to times written without a zone (NXLog EventTime).
    """

    name = "windows-security"
    source_type = "windows-events"
    version = ADAPTER_VERSION

    def __init__(self, source: str | Path | bytes | IO[bytes], *, config: EventLogConfig | None = None, **kw: Any) -> None:
        super().__init__(source, config=config, **kw)
        self.format: str | None = None

    def raw_records(self) -> Iterator[RawRecord | tuple[str, RawRecord]]:
        stream, location, close = open_source(self.source)
        try:
            head = stream.read(256)
            stream.seek(0)
            t = head.lstrip(b"\xef\xbb\xbf \t\r\n")
            if t.startswith(b"<"):
                self.format = "xml"
                yield from self._xml(stream, location)
            elif t.startswith(b"["):
                self.format = "json-array"
                yield from iter_json_array(stream, location, max_bytes=self.config.common.max_record_bytes)
            else:
                self.format = "json-lines"
                for item in iter_lines(stream, location, max_bytes=self.config.common.max_record_bytes):
                    if isinstance(item, tuple) or item.data.strip():
                        yield item
        finally:
            if close:
                stream.close()

    def _xml(self, stream: IO[bytes], location: str) -> Iterator[RawRecord | tuple[str, RawRecord]]:
        """Each <Event>...</Event> of the stream, with its byte offset (module docstring)."""
        limit = self.config.common.max_record_bytes
        buf = b""
        pos = 0
        index = 0
        while True:
            block = stream.read(1 << 20)
            if block:
                buf += block
            while True:
                m = _EVENT_START.search(buf)
                if m is None:
                    break
                end = buf.find(_EVENT_END, m.start())
                if end < 0:
                    if len(buf) - m.start() > limit:
                        yield ("oversize", RawRecord(buf[m.start():m.start() + limit], location, index,
                                                     offset=pos + m.start()))
                        index += 1
                        pos += len(buf)
                        buf = b""
                    break
                stop = end + len(_EVENT_END)
                yield RawRecord(buf[m.start():stop], location, index, offset=pos + m.start())
                index += 1
                pos += stop
                buf = buf[stop:]
            if not block:
                break
            # keep only what may begin an event
            m = _EVENT_START.search(buf)
            cut = m.start() if m is not None else max(len(buf) - 16, 0)
            pos += cut
            buf = buf[cut:]

    def decode(self, raw: RawRecord) -> Iterable[UpdateDraft]:
        if self.format == "xml":
            if _DTD.search(raw.data):
                raise MalformedRecord("xml-dtd-forbidden", "document type declarations are refused")
            try:
                elem = ET.fromstring(raw.data)
            except ET.ParseError as exc:
                raise MalformedRecord("xml-syntax", str(exc)) from None
            values = flatten_xml(elem)
        else:
            try:
                obj = json.loads(raw.data)
            except (ValueError, UnicodeDecodeError) as exc:
                raise MalformedRecord("json-decode", str(exc)) from None
            if not isinstance(obj, dict):
                raise MalformedRecord("json-not-object", type(obj).__name__)
            values = flatten_json(obj)
        return [self.map_event(values, raw)]

    def map_event(self, values: dict[str, Any], raw: RawRecord) -> UpdateDraft:
        """Map one flattened event (module docstring)."""
        try:
            eid = int(str(values.get("System.EventID", "")).strip())
        except ValueError:
            raise MalformedRecord("no-event-id", str(values.get("System.EventID"))) from None
        record = str(eid) if eid in EVENT_IDS else "other"
        record_type = f"windows.{record}"
        vals: dict[str, Any] = {}
        for k, v in values.items():
            if isinstance(v, str) and k.startswith("EventData.") and v.strip() in ("-", ""):
                vals[k] = UNSET                                   # Windows writes "-" for "no value"
            else:
                vals[k] = v
        ts = vals.get("System.TimeCreated.SystemTime")
        if isinstance(ts, str) and " " in ts.strip() and "T" not in ts:
            vals["System.TimeCreated.SystemTime"] = ts.strip().replace(" ", "T", 1)     # NXLog "YYYY-MM-DD hh:mm:ss"
        d = UpdateDraft(record_type, raw)
        src = record_type
        self._ctx.values = vals
        mapper(record_type).apply(vals, d, self._ctx, self.stats, source=src,
                                  keep_unmapped=self.config.common.keep_unmapped)
        if d.time is None and isinstance(ts, str):
            try:
                d.time = rfc3339(ts, default_offset_s=self._ctx.utc_offset_s)
            except TimeParseError as exc:
                self.stats.refuse("@time", str(exc))
        getattr(self, f"_after_{record}", self._after_other)(d, vals, src)
        return d

    def _computer(self, d: UpdateDraft) -> Any:
        return self.resolver.host_name(d.value("event.hostname"))

    def _account(self, user: Any, domain: Any) -> Any:
        return self.resolver.account(None if user is None else str(user), None if domain is None else str(domain),
                                     case_insensitive=self.config.case_insensitive_accounts)

    def _source_host(self, d: UpdateDraft) -> None:
        ip = d.value("flow.src_ip")
        if ip is not None and str(ip) not in _LOOPBACK:
            d.add_entity(self.resolver.address(str(ip)), "initiator")

    def _package(self, d: UpdateDraft, src: str) -> None:
        pkg = str(d.value("auth.package") or "").strip().lower()
        code = {"ntlm": 1, "kerberos": 2}.get(pkg)
        if code is not None:
            set_field(d, "auth.protocol", code, src, self.stats)

    def _after_4624(self, d: UpdateDraft, v: dict[str, Any], src: str) -> None:
        set_field(d, "auth.activity", 1, src, self.stats)
        set_field(d, "auth.result", 1, src, self.stats)
        self._package(d, src)
        self._source_host(d)
        d.add_entity(self._computer(d), "responder")
        d.add_entity(self._account(d.value("auth.user"), d.value("auth.domain")), "account")

    def _after_4625(self, d: UpdateDraft, v: dict[str, Any], src: str) -> None:
        set_field(d, "auth.activity", 1, src, self.stats)
        set_field(d, "auth.result", 2, src, self.stats)
        self._package(d, src)
        sub, st = v.get("EventData.SubStatus"), v.get("EventData.Status")
        for code in (sub, st):
            try:
                value = int(str(code), 16) if isinstance(code, str) and code.lower().startswith("0x") else int(str(code))
            except (TypeError, ValueError):
                continue
            if value != 0:
                set_field(d, "auth.failure_status", value, src, self.stats)
                break
        self._source_host(d)
        d.add_entity(self._computer(d), "responder")
        d.add_entity(self._account(d.value("auth.user"), d.value("auth.domain")), "account")

    def _after_4634(self, d: UpdateDraft, v: dict[str, Any], src: str) -> None:
        set_field(d, "auth.activity", 2, src, self.stats)
        set_field(d, "auth.result", 1, src, self.stats)
        d.add_entity(self._computer(d), "responder")
        d.add_entity(self._account(d.value("auth.user"), d.value("auth.domain")), "account")

    def _after_4648(self, d: UpdateDraft, v: dict[str, Any], src: str) -> None:
        set_field(d, "auth.activity", 1, src, self.stats)
        d.add_entity(self._computer(d), "initiator")
        d.add_entity(self.resolver.host_name(d.value("auth.service")), "responder")
        d.add_entity(self._account(d.value("auth.subject_user"), d.value("auth.subject_domain")), "account")
        d.add_entity(self._account(d.value("auth.target_user"), d.value("auth.target_domain")), "target_account")

    def _after_4672(self, d: UpdateDraft, v: dict[str, Any], src: str) -> None:
        set_field(d, "auth.activity", 99, src, self.stats)
        set_field(d, "auth.elevated", 1, src, self.stats)
        d.add_entity(self._computer(d), "responder")
        d.add_entity(self._account(d.value("auth.user"), d.value("auth.domain")), "account")

    def _kerberos(self, d: UpdateDraft, activity: int, src: str, *, result: int | None = None) -> None:
        set_field(d, "auth.activity", activity, src, self.stats)
        set_field(d, "auth.protocol", 2, src, self.stats)
        status = d.value("proto.kerberos.error_code")
        if result is not None:
            set_field(d, "auth.result", result, src, self.stats)
        elif status is not None:
            set_field(d, "auth.result", 1 if int(status) == _KRB_OK else 2, src, self.stats)
        if status is not None and int(status) != _KRB_OK:
            set_field(d, "auth.failure_status", int(status), src, self.stats)
        self._source_host(d)
        d.add_entity(self._computer(d), "responder")
        user, domain = d.value("auth.user"), d.value("auth.domain")
        d.add_entity(self._account(user, domain), "account")
        service = d.value("auth.service")
        if service is not None:
            d.add_entity(self._account(service, domain), "target_account")

    def _after_4768(self, d: UpdateDraft, v: dict[str, Any], src: str) -> None:
        self._kerberos(d, 3, src)

    def _after_4769(self, d: UpdateDraft, v: dict[str, Any], src: str) -> None:
        self._kerberos(d, 4, src)

    def _after_4771(self, d: UpdateDraft, v: dict[str, Any], src: str) -> None:
        self._kerberos(d, 6, src, result=2)

    def _after_4776(self, d: UpdateDraft, v: dict[str, Any], src: str) -> None:
        set_field(d, "auth.activity", 1, src, self.stats)
        set_field(d, "auth.protocol", 1, src, self.stats)
        st = v.get("EventData.Status")
        try:
            status = int(str(st), 16) if isinstance(st, str) and st.lower().startswith("0x") else int(str(st))
        except (TypeError, ValueError):
            status = None
        if status is not None:
            set_field(d, "auth.result", 1 if status == 0 else 2, src, self.stats)
            if status != 0:
                set_field(d, "auth.failure_status", status, src, self.stats)
        d.add_entity(self.resolver.host_name(d.value("auth.workstation")), "initiator")
        d.add_entity(self._computer(d), "responder")
        d.add_entity(self._account(d.value("auth.user"), None), "account")

    def _after_other(self, d: UpdateDraft, v: dict[str, Any], src: str) -> None:
        d.add_entity(self._computer(d), "subject")


__all__ = ["ADAPTER_VERSION", "EVENT_IDS", "WindowsEventSource", "flatten_json", "flatten_xml"]
