"""CEF and LEEF events -> state updates (tables in datamodel/maps/eventlogs.py).

CEF (ArcSight "Common Event Format" implementation standard):

    [syslog header] CEF:Version|Device Vendor|Device Product|Device Version|Device Event Class ID|Name|Severity|Extension

Header fields escape "|" and "\\" with a backslash. The extension is key=value pairs separated by
spaces; a value runs until the next " key=", so values may hold spaces; "=", "\\", newline and carriage
return are escaped as "\\=", "\\\\", "\\n", "\\r". Keys are the dictionary's short names; full names
("sourceAddress") are read as their short names.

LEEF (IBM QRadar "Log Event Extended Format"):

    LEEF:1.0|Vendor|Product|Version|EventID|key=value<TAB>key=value...
    LEEF:2.0|Vendor|Product|Version|EventID|DelimiterCharacter|key=value<delim>key=value...

LEEF 2.0 names its attribute delimiter in the sixth header field: one character, or its code written
as "xHH" or "0xHH" (an empty field means tab). A backslash escapes the next character.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from nagahana.core.errors import ConfigMissing
from nagahana.ingest import derive as DV
from nagahana.ingest.core import MalformedRecord, RawRecord, UpdateDraft
from nagahana.ingest.mapping import mapper, set_field
from nagahana.ingest.syslog import LineAdapter, parse_envelope
from nagahana.ingest.timeparse import Instant

ADAPTER_VERSION = "1.0.0"
_KEY = re.compile(r"^[A-Za-z0-9_.\-\[\]]+$")
#: CEF full key names -> short names (ArcSight CEF standard, extension dictionary).
CEF_FULL_NAMES: dict[str, str] = {
    "deviceAction": "act", "applicationProtocol": "app", "deviceEventCategory": "cat", "baseEventCount": "cnt",
    "destinationHostName": "dhost", "destinationMacAddress": "dmac", "destinationNtDomain": "dntdom",
    "destinationProcessId": "dpid", "destinationUserPrivileges": "dpriv", "destinationProcessName": "dproc",
    "destinationPort": "dpt", "destinationAddress": "dst", "deviceTimeZone": "dtz", "destinationUserId": "duid",
    "destinationUserName": "duser", "deviceAddress": "dvc", "deviceHostName": "dvchost", "deviceMacAddress": "dvcmac",
    "deviceProcessId": "dvcpid", "endTime": "end", "fileName": "fname", "fileSize": "fsize", "bytesIn": "in",
    "message": "msg", "bytesOut": "out", "eventOutcome": "outcome", "transportProtocol": "proto",
    "requestUrl": "request", "deviceReceiptTime": "rt", "sourceHostName": "shost", "sourceMacAddress": "smac",
    "sourceNtDomain": "sntdom", "sourceProcessId": "spid", "sourceUserPrivileges": "spriv",
    "sourceProcessName": "sproc", "sourcePort": "spt", "sourceAddress": "src", "startTime": "start",
    "sourceUserId": "suid", "sourceUserName": "suser", "eventId": "eventId", "agentAddress": "agt",
    "agentHostName": "ahost", "agentId": "aid", "agentMacAddress": "amac", "agentReceiptTime": "art",
    "agentType": "at", "agentTimeZone": "atz", "agentVersion": "av", "deviceCustomString1": "cs1",
    "deviceCustomString2": "cs2", "deviceCustomString3": "cs3", "deviceCustomString4": "cs4",
    "deviceCustomString5": "cs5", "deviceCustomString6": "cs6", "deviceCustomString1Label": "cs1Label",
    "deviceCustomString2Label": "cs2Label", "deviceCustomString3Label": "cs3Label", "deviceCustomString4Label": "cs4Label",
    "deviceCustomString5Label": "cs5Label", "deviceCustomString6Label": "cs6Label", "deviceCustomNumber1": "cn1",
    "deviceCustomNumber2": "cn2", "deviceCustomNumber3": "cn3", "deviceCustomNumber1Label": "cn1Label",
    "deviceCustomNumber2Label": "cn2Label", "deviceCustomNumber3Label": "cn3Label",
    "deviceCustomFloatingPoint1": "cfp1", "deviceCustomFloatingPoint2": "cfp2", "deviceCustomFloatingPoint3": "cfp3",
    "deviceCustomFloatingPoint4": "cfp4", "deviceCustomIPv6Address1": "c6a1", "deviceCustomIPv6Address2": "c6a2",
    "deviceCustomIPv6Address3": "c6a3", "deviceCustomIPv6Address4": "c6a4", "sourceLatitude": "slat",
    "sourceLongitude": "slong", "destinationLatitude": "dlat", "destinationLongitude": "dlong",
}
_CEF_VALUE_ESCAPES = {"\\": "\\", "=": "=", "n": "\n", "r": "\r"}


def split_header(text: str, count: int) -> tuple[list[str], str]:
    """The first `count` "|"-separated fields of `text` (\\| and \\\\ unescaped) and the rest after them."""
    fields: list[str] = []
    cur: list[str] = []
    i = 0
    while i < len(text) and len(fields) < count:
        ch = text[i]
        if ch == "\\" and i + 1 < len(text) and text[i + 1] in "|\\":
            cur.append(text[i + 1])
            i += 2
            continue
        if ch == "|":
            fields.append("".join(cur))
            cur = []
            i += 1
            continue
        cur.append(ch)
        i += 1
    if len(fields) < count:
        raise MalformedRecord("header-fields", f"{len(fields)} of {count} header fields")
    return fields, text[i:]


def parse_cef_extension(text: str) -> dict[str, str]:
    """CEF extension key=value pairs (module docstring); a later repeat of a key overwrites (counted by caller)."""
    eqs: list[int] = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch == "\\":
            i += 2
            continue
        if ch == "=":
            eqs.append(i)
        i += 1
    keys: list[tuple[int, int, str]] = []                     # (key start, '=' index, key)
    for eq in eqs:
        start = text.rfind(" ", 0, eq) + 1
        key = text[start:eq]
        if _KEY.match(key) and (not keys or start > keys[-1][1]):
            keys.append((start, eq, key))
    out: dict[str, str] = {}
    for n, (_start, eq, key) in enumerate(keys):
        end = keys[n + 1][0] - 1 if n + 1 < len(keys) else len(text)
        raw = text[eq + 1:end]
        if n + 1 == len(keys):
            raw = raw.rstrip()
        out[key] = _unescape_cef(raw)
    return out


def _unescape_cef(v: str) -> str:
    if "\\" not in v:
        return v
    out: list[str] = []
    i = 0
    while i < len(v):
        if v[i] == "\\" and i + 1 < len(v) and v[i + 1] in _CEF_VALUE_ESCAPES:
            out.append(_CEF_VALUE_ESCAPES[v[i + 1]])
            i += 2
        else:
            out.append(v[i])
            i += 1
    return "".join(out)


def leef_delimiter(field: str) -> str:
    """The LEEF 2.0 delimiter named in the sixth header field."""
    f = field.strip()
    if not f:
        return "\t"
    if re.fullmatch(r"(?:0?[xX])[0-9A-Fa-f]{1,4}", f):
        return chr(int(f.lstrip("0").lstrip("xX"), 16))
    if len(f) == 1:
        return f
    raise MalformedRecord("leef-delimiter", f"{field!r}")


def parse_leef_attributes(text: str, delim: str) -> dict[str, str]:
    out: dict[str, str] = {}
    parts: list[str] = []
    cur: list[str] = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch == "\\" and i + 1 < len(text):
            cur.append(text[i + 1])
            i += 2
            continue
        if ch == delim:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
        i += 1
    parts.append("".join(cur))
    for p in parts:
        if not p.strip():
            continue
        key, sep, value = p.partition("=")
        if not sep or not key.strip():
            raise MalformedRecord("leef-attribute", p[:80])
        out[key.strip()] = value
    return out


class _EnvelopeMixin(LineAdapter):
    def _envelope(self, line: str, marker: str) -> tuple[dict[str, Any], str]:
        """(syslog envelope values, payload starting at `marker`)."""
        idx = line.find(marker)
        if idx < 0:
            raise MalformedRecord(f"no-{marker.rstrip(':').lower()}", line[:80])
        env: dict[str, Any] = {}
        prefix = line[:idx].rstrip()
        if prefix:
            try:
                _kind, env = parse_envelope(prefix + " ", self._ctx)
            except (MalformedRecord, ConfigMissing) as exc:
                # The event's own time (rt, devTime) may still place it; without one it is quarantined.
                env = {}
                self.stats.counters[f"unparsed_syslog_prefix: {type(exc).__name__}"] += 1
        return env, line[idx:]


class CEFSource(_EnvelopeMixin):
    """CEF events, one per line, optionally behind a syslog header. See the module docstring."""

    name = "cef"
    source_type = "cef"
    version = ADAPTER_VERSION

    def decode(self, raw: RawRecord) -> Iterable[UpdateDraft]:
        env, payload = self._envelope(self.text(raw), "CEF:")
        header, ext = split_header(payload[4:], 7)
        version, vendor, product, dversion, sig, name, severity = header
        values: dict[str, Any] = {
            "cef_version": version, "device_vendor": vendor, "device_product": product, "device_version": dversion,
            "signature_id": sig, "name": name, "severity": severity,
        }
        for key, value in parse_cef_extension(ext).items():
            short = CEF_FULL_NAMES.get(key, key)
            if short in values:
                self.stats.counters["cef_duplicate_keys"] += 1
            values[short] = value
        d = UpdateDraft("cef.event", raw)
        src = "cef.event"
        self._ctx.values = values
        self._ctx.ipv6 = ":" in values.get("src", "")
        mapper("cef.event").apply(values, d, self._ctx, self.stats, source=src,
                                  keep_unmapped=self.config.common.keep_unmapped)
        if d.time is None and isinstance(env.get("timestamp"), Instant):
            d.time = env["timestamp"]
        if env.get("hostname"):
            set_field(d, "event.hostname", str(env["hostname"]), src, self.stats)
        if sig:
            set_field(d, "alert.signature", DV.name_signature(f"cef:{vendor}:{product}", sig), src, self.stats)
        DV.flow_entities(d, self.resolver)
        if d.value("flow.src_ip") is None and d.value("flow.dst_ip") is None:
            d.add_entity(self.resolver.address(values.get("dvc")) or self.resolver.host_name(d.value("event.hostname")),
                         "subject")
        d.add_entity(self.resolver.account(values.get("suser"), values.get("sntdom"),
                                           case_insensitive=self.config.case_insensitive_accounts), "account")
        d.add_entity(self.resolver.account(values.get("duser"), values.get("dntdom"),
                                           case_insensitive=self.config.case_insensitive_accounts), "target_account")
        return [d]


class LEEFSource(_EnvelopeMixin):
    """LEEF 1.0 and 2.0 events, one per line, optionally behind a syslog header. See the module docstring."""

    name = "leef"
    source_type = "leef"
    version = ADAPTER_VERSION

    def decode(self, raw: RawRecord) -> Iterable[UpdateDraft]:
        env, payload = self._envelope(self.text(raw), "LEEF:")
        version = payload[5:payload.find("|")] if "|" in payload else ""
        if version.startswith("2"):
            header, attrs = split_header(payload[5:], 6)
            delim = leef_delimiter(header[5])
        elif version.startswith("1"):
            header, attrs = split_header(payload[5:], 5)
            delim = "\t"
        else:
            raise MalformedRecord("leef-version", version)
        values: dict[str, Any] = {"leef_version": header[0], "vendor": header[1], "product": header[2],
                                  "version": header[3], "event_id": header[4]}
        if len(header) > 5:
            values["delimiter"] = header[5]
        values.update(parse_leef_attributes(attrs, delim))
        d = UpdateDraft("leef.event", raw)
        src = "leef.event"
        self._ctx.values = values
        self._ctx.ipv6 = ":" in values.get("src", "")
        mapper("leef.event").apply(values, d, self._ctx, self.stats, source=src,
                                   keep_unmapped=self.config.common.keep_unmapped)
        if d.time is None and isinstance(env.get("timestamp"), Instant):
            d.time = env["timestamp"]
        if env.get("hostname"):
            set_field(d, "event.hostname", str(env["hostname"]), src, self.stats)
        set_field(d, "alert.signature", DV.name_signature(f"leef:{header[1]}:{header[2]}", header[4]), src, self.stats)
        for key, code in (("isLoginEvent", 1), ("isLogoutEvent", 2)):
            if str(values.get(key, "")).strip().lower() == "true":
                set_field(d, "auth.activity", code, src, self.stats)
        DV.flow_entities(d, self.resolver)
        if d.value("flow.src_ip") is None and d.value("flow.dst_ip") is None:
            d.add_entity(self.resolver.host_name(d.value("event.hostname")), "subject")
        d.add_entity(self.resolver.account(values.get("usrName"), values.get("domain"),
                                           case_insensitive=self.config.case_insensitive_accounts), "account")
        return [d]


__all__ = ["ADAPTER_VERSION", "CEFSource", "CEF_FULL_NAMES", "LEEFSource", "leef_delimiter", "parse_cef_extension",
           "parse_leef_attributes", "split_header"]
