"""Syslog envelopes (RFC 5424, RFC 3164) and the syslog adapter.

RFC 5424 (section 6):  <PRI>VERSION SP TIMESTAMP SP HOSTNAME SP APP-NAME SP PROCID SP MSGID SP SD [SP MSG]
    NILVALUE is "-"; STRUCTURED-DATA is "-" or [SD-ID *(SP PARAM-NAME="PARAM-VALUE")]..., where a value
    escapes '"', '\\' and ']' with a backslash (section 6.3.3); MSG may start with a UTF-8 BOM.
RFC 3164 (section 4.1): [<PRI>]Mmm dd hh:mm:ss HOSTNAME TAG[PID]: MSG
    The timestamp has no year and no zone (configured, AS-698). Files written by syslog daemons usually
    omit PRI.
rsyslog's RSYSLOG_FileFormat writes an RFC 3339 timestamp in place of the RFC 3164 one; it is read the
same way with the timestamp's own zone.

`parse_envelope` returns the envelope fields and the message; adapters for CEF, LEEF and Linux
authentication logs call it before parsing their payload.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import IO, Any

from nagahana.core.errors import ConfigMissing
from nagahana.ingest.config import EventLogConfig
from nagahana.ingest.convert import ConvContext
from nagahana.ingest.core import MalformedRecord, RawRecord, StreamAdapter, UpdateDraft, iter_lines, open_source
from nagahana.ingest.mapping import mapper
from nagahana.ingest.timeparse import Instant, TimeParseError, rfc3164, rfc3339

ADAPTER_VERSION = "1.0.0"
_PRI = re.compile(r"^<(\d{1,3})>")
_RFC5424_HEAD = re.compile(r"^(\d{1,2}) (\S+) (\S+) (\S+) (\S+) (\S+) ")
_BSD_TIME = re.compile(r"^([A-Z][a-z]{2} [ \d]\d \d{2}:\d{2}:\d{2}(?:\.\d{1,9})?) ")
_ISO_TIME = re.compile(r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,9})?(?:Z|[+-]\d{2}:?\d{2})?) ")
_TAG = re.compile(r"^([^\s:\[]{1,128})(?:\[([^\]]*)\])?: ?")


def parse_structured_data(text: str) -> tuple[dict[str, dict[str, Any]], str]:
    """RFC 5424 STRUCTURED-DATA at the start of `text` -> ({sd-id: {param: value or [values]}}, rest)."""
    if text.startswith("-"):
        return {}, text[1:]
    out: dict[str, dict[str, Any]] = {}
    i = 0
    while i < len(text) and text[i] == "[":
        j = i + 1
        while j < len(text) and text[j] not in " ]":
            j += 1
        sd_id = text[i + 1:j]
        params: dict[str, Any] = {}
        while j < len(text) and text[j] == " ":
            j += 1
            k = text.find("=", j)
            if k < 0 or k + 1 >= len(text) or text[k + 1] != '"':
                raise MalformedRecord("bad-structured-data", text[i:i + 80])
            name = text[j:k]
            j = k + 2
            value: list[str] = []
            while j < len(text):
                ch = text[j]
                if ch == "\\" and j + 1 < len(text) and text[j + 1] in '"\\]':
                    value.append(text[j + 1])
                    j += 2
                    continue
                if ch == '"':
                    break
                value.append(ch)
                j += 1
            else:
                raise MalformedRecord("bad-structured-data", "unterminated parameter value")
            j += 1                                       # the closing quote
            v = "".join(value)
            prev = params.get(name)
            params[name] = v if prev is None else ([*prev, v] if isinstance(prev, list) else [prev, v])
        if j >= len(text) or text[j] != "]":
            raise MalformedRecord("bad-structured-data", text[i:i + 80])
        out[sd_id] = params
        i = j + 1
    return out, text[i:]


def parse_envelope(line: str, ctx: ConvContext) -> tuple[str, dict[str, Any]]:
    """("rfc5424" or "rfc3164", envelope values) of one syslog line; the message is values["message"].

    The values use the syslog table's source names; "timestamp" is an Instant when it parsed, else the
    text (the converter then counts the refusal).
    """
    v: dict[str, Any] = {}
    rest = line
    m = _PRI.match(rest)
    if m is not None:
        pri = int(m.group(1))
        if pri > 191:
            raise MalformedRecord("bad-pri", m.group(1))
        v.update(pri=pri, facility=pri >> 3, severity=pri & 7)
        rest = rest[m.end():]
        h = _RFC5424_HEAD.match(rest)
        if h is not None and h.group(1).isdigit() and (h.group(2) == "-" or h.group(2)[:4].isdigit()):
            v["version"] = int(h.group(1))
            ts, host, app, proc, msgid = h.group(2), h.group(3), h.group(4), h.group(5), h.group(6)
            if ts != "-":
                try:
                    v["timestamp"] = rfc3339(ts, default_offset_s=ctx.utc_offset_s)
                except TimeParseError:
                    v["timestamp"] = ts
            for key, val in (("hostname", host), ("app_name", app), ("procid", proc), ("msgid", msgid)):
                if val != "-":
                    v[key] = val
            sd, msg = parse_structured_data(rest[h.end():])
            if sd:
                v["structured_data"] = sd
            msg = msg[1:] if msg.startswith(" ") else msg
            if msg.startswith("\ufeff"):
                msg = msg[1:]
            if msg:
                v["message"] = msg
            return "rfc5424", v
    t = _ISO_TIME.match(rest)
    if t is not None:
        try:
            v["timestamp"] = rfc3339(t.group(1), default_offset_s=ctx.utc_offset_s)
        except TimeParseError:
            v["timestamp"] = t.group(1)
        rest = rest[t.end():]
    else:
        b = _BSD_TIME.match(rest)
        if b is None:
            raise MalformedRecord("syslog-syntax", "no timestamp after PRI" if m else "no syslog timestamp")
        if ctx.year is None:
            raise ConfigMissing("RFC 3164 timestamps carry no year: set clock.assumed_year in the adapter config "
                                "(AS-698)")
        try:
            v["timestamp"] = rfc3164(b.group(1), year=ctx.year, offset_s=ctx.utc_offset_s)
        except TimeParseError:
            v["timestamp"] = b.group(1)
        rest = rest[b.end():]
    host, _, rest = rest.partition(" ")
    if host:
        v["hostname"] = host
    tag = _TAG.match(rest)
    if tag is not None:
        v["app_name"] = tag.group(1)
        if tag.group(2):
            v["procid"] = tag.group(2)
        rest = rest[tag.end():]
    if rest:
        v["message"] = rest
    return "rfc3164", v


def clock_ctx(cfg: EventLogConfig) -> ConvContext:
    clock = cfg.common.clock
    return ConvContext(year=clock.assumed_year,
                       utc_offset_s=None if clock.utc_offset_hours is None else clock.utc_offset_hours * 3600.0)


class LineAdapter(StreamAdapter):
    """Base of line-oriented event-log adapters (one record per non-blank line)."""

    def __init__(self, source: str | Path | bytes | IO[bytes], *, config: EventLogConfig | None = None,
                 **kw: Any) -> None:
        self.config = config or EventLogConfig()
        super().__init__(source, self.config.common, **kw)
        self._ctx = clock_ctx(self.config)

    def raw_records(self) -> Iterator[RawRecord | tuple[str, RawRecord]]:
        stream, location, close = open_source(self.source)
        try:
            for item in iter_lines(stream, location, max_bytes=self.config.common.max_record_bytes):
                if isinstance(item, tuple) or item.data.strip():
                    yield item
        finally:
            if close:
                stream.close()

    @staticmethod
    def text(raw: RawRecord) -> str:
        return raw.data.decode("utf-8", "backslashreplace")


class SyslogSource(LineAdapter):
    """Syslog lines (RFC 5424 or RFC 3164) -> state updates; the logging host is the subject."""

    name = "syslog"
    source_type = "syslog"
    version = ADAPTER_VERSION

    def decode(self, raw: RawRecord) -> Iterable[UpdateDraft]:
        kind, values = parse_envelope(self.text(raw), self._ctx)
        record_type = f"syslog.{kind}"
        d = UpdateDraft(record_type, raw)
        self._ctx.values = values
        mapper(record_type).apply(values, d, self._ctx, self.stats, source=record_type,
                                  keep_unmapped=self.config.common.keep_unmapped)
        if isinstance(values.get("timestamp"), Instant):
            d.time = values["timestamp"]
        d.add_entity(self.resolver.host_name(values.get("hostname")), "subject")
        return [d]


__all__ = ["ADAPTER_VERSION", "LineAdapter", "SyslogSource", "clock_ctx", "parse_envelope", "parse_structured_data"]
