"""Converters: source values to data-model values, named as the mapping tables name them.

A converter takes the raw source value and a `ConvContext` (the record's other raw values and the time
configuration) and returns the data-model value, or raises `Refusal` with a short reason when the value
cannot be a measurement of the field (a negative count, an unknown code, an unparseable time). A
refusal is never silent: the mapping engine records the field as NOT_SUPPLIED and counts the reason
per field (D-41). Time converters return a `timeparse.Instant`; the engine turns it into seconds for
time-typed fields and keeps the nanoseconds for the event time.
"""

from __future__ import annotations

import base64
import ipaddress
import math
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from nagahana.ingest import codes as C
from nagahana.ingest import timeparse as T


class JsonNumber(str):
    """A JSON number kept as its text, so converters read its exact digits (json.loads parse_float)."""


class Refusal(ValueError):
    """A source value that cannot be a value of its field; the message is the reason."""


@dataclass
class ConvContext:
    """What a converter may consult besides the value itself.

    Attributes
    ----------
    values: the record's raw source values (by source field name).
    year: assumed year for timestamps without one (configuration; None refuses such timestamps).
    utc_offset_s: offset of local timestamps east of UTC (configuration; None reads them as UTC and
        reports the zone as unknown).
    ipv6: True when the record's addresses are IPv6 (Zeek writes "icmp" for ICMPv6).
    """

    values: Mapping[str, Any] = field(default_factory=dict)
    year: int | None = None
    utc_offset_s: float | None = None
    ipv6: bool = False


Converter = Callable[[Any, ConvContext], Any]
_REGISTRY: dict[str, Converter] = {}


def converter(name: str) -> Callable[[Converter], Converter]:
    """Register a converter under `name`."""
    def deco(fn: Converter) -> Converter:
        if name in _REGISTRY:
            raise ValueError(f"converter {name!r} registered twice")
        _REGISTRY[name] = fn
        return fn
    return deco


def get(name: str) -> Converter:
    """The converter named `name` ("code:<table>" and "bit:<mask>" are parametric)."""
    fn = _REGISTRY.get(name)
    if fn is not None:
        return fn
    if name.startswith("code:"):
        table = name[5:]
        if table not in C.TABLES:
            raise KeyError(f"unknown code table {table!r}")
        return lambda raw, ctx: code_lookup(table, raw)
    if name.startswith("bit:"):
        mask = int(name[4:], 16)
        return lambda raw, ctx: mask if as_bool(raw) else 0
    raise KeyError(f"unknown converter {name!r}")


def names() -> tuple[str, ...]:
    """Registered converter names (parametric families as "code:*" and "bit:*")."""
    return (*sorted(_REGISTRY), "code:*", "bit:*")


def _scalar(raw: Any) -> Any:
    """The single value of `raw` (a one-element list counts as its element)."""
    if isinstance(raw, list | tuple):
        if len(raw) != 1:
            raise Refusal(f"expected one value, got {len(raw)}")
        return raw[0]
    return raw


def _text(raw: Any) -> str:
    raw = _scalar(raw)
    if isinstance(raw, bytes):
        return raw.decode("utf-8", "backslashreplace")
    if isinstance(raw, dict):
        raise Refusal("expected a scalar, got an object")
    return str(raw)


_INT = re.compile(r"^\s*[-+]?\d+\s*$")


def as_int(raw: Any) -> int:
    """An integer from an int, an integral float or decimal text (no rounding of fractions)."""
    raw = _scalar(raw)
    if isinstance(raw, bool):
        return int(raw)
    if isinstance(raw, int):
        return raw
    if isinstance(raw, float):
        if not math.isfinite(raw) or raw != int(raw):
            raise Refusal(f"not an integer: {raw!r}")
        return int(raw)
    text = _text(raw)
    if _INT.match(text):
        return int(text)
    try:
        f = float(text)
    except ValueError:
        raise Refusal(f"not an integer: {text!r}") from None
    if not math.isfinite(f) or f != int(f):
        raise Refusal(f"not an integer: {text!r}")
    return int(f)


def as_float(raw: Any) -> float:
    raw = _scalar(raw)
    if isinstance(raw, bool):
        raise Refusal("a boolean is not a number")
    try:
        f = float(raw)
    except (TypeError, ValueError):
        raise Refusal(f"not a number: {raw!r}") from None
    if not math.isfinite(f):
        raise Refusal(f"not a finite number: {raw!r}")
    return f


_TRUE = frozenset({"t", "true", "1", "yes", "y", "on"})
_FALSE = frozenset({"f", "false", "0", "no", "n", "off"})


def as_bool(raw: Any) -> bool:
    raw = _scalar(raw)
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, int | float):
        if raw in (0, 1):
            return bool(raw)
        raise Refusal(f"not a boolean: {raw!r}")
    t = _text(raw).strip().lower()
    if t in _TRUE:
        return True
    if t in _FALSE:
        return False
    raise Refusal(f"not a boolean: {raw!r}")


def as_hex(raw: Any) -> int:
    """An integer written in hex ("0x1A", "1a" when it has a hex letter) or in decimal."""
    raw = _scalar(raw)
    if isinstance(raw, int) and not isinstance(raw, bool):
        return raw
    t = _text(raw).strip()
    try:
        if t.lower().startswith("0x"):
            return int(t, 16)
        return int(t, 10)
    except ValueError:
        try:
            return int(t, 16)
        except ValueError:
            raise Refusal(f"not a hexadecimal number: {t!r}") from None


def as_address(raw: Any) -> str:
    """Canonical text of an IPv4 or IPv6 address (brackets stripped; IPv4-mapped IPv6 read as IPv4)."""
    t = _text(raw).strip().strip("[]")
    try:
        ip = ipaddress.ip_address(t)
    except ValueError:
        raise Refusal(f"not an IP address: {t!r}") from None
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        return str(ip.ipv4_mapped)
    return str(ip)


_MAC_HEX = re.compile(r"[0-9a-fA-F]")


def as_mac(raw: Any) -> str:
    """A link-layer address as lower-case colon-separated hex ("aa:bb:cc:dd:ee:ff")."""
    raw = _scalar(raw)
    if isinstance(raw, bytes):
        if len(raw) != 6:
            raise Refusal(f"a MAC address has 6 bytes, got {len(raw)}")
        return raw.hex(":")
    digits = "".join(_MAC_HEX.findall(_text(raw)))
    rest = re.sub(r"[0-9a-fA-F:.\-\s]", "", _text(raw))
    if len(digits) != 12 or rest:
        raise Refusal(f"not a MAC address: {raw!r}")
    d = digits.lower()
    return ":".join(d[i:i + 2] for i in range(0, 12, 2))


def code_lookup(table: str, raw: Any) -> int:
    """Code of a source spelling in a code table (see ingest/codes.py)."""
    raw = _scalar(raw)
    if isinstance(raw, int) and not isinstance(raw, bool):
        return raw
    text = _text(raw).strip()
    if re.fullmatch(r"(?:0[xX][0-9a-fA-F]+|\d+)", text):
        return as_hex(text)
    mapping, has_other = C.TABLES[table]
    key = C.norm(text)
    for prefix in C.STRIP_PREFIXES.get(table, ()):
        if key.startswith(prefix) and key[len(prefix):] in mapping:
            key = key[len(prefix):]
    if table == "modbus_function" and key.endswith("exception") and key[:-9] in mapping:
        key = key[:-9]
    if table == "dns_qtype" and key.startswith("type") and key[4:].isdigit():
        return int(key[4:])
    if table == "dns_qclass" and key.startswith("class") and key[5:].isdigit():
        return int(key[5:])
    code = mapping.get(key)
    if code is not None:
        return code
    if has_other:
        return 0
    raise Refusal(f"unknown {table} {text!r}")


def _instant(raw: Any, ctx: ConvContext) -> T.Instant:
    raw = _scalar(raw)
    if isinstance(raw, T.Instant):
        return raw
    if isinstance(raw, bool):
        raise Refusal("a boolean is not a time")
    if isinstance(raw, int):
        return T.epoch_integer(raw, "s")
    if isinstance(raw, float):
        if not math.isfinite(raw):
            raise Refusal("not a finite time")
        return T.epoch_number(raw)
    text = _text(raw).strip()
    try:
        if T._DECIMAL.match(text):
            t = T.decimal_epoch(text)
            if isinstance(raw, JsonNumber) and t.resolution > 1e-6:
                # A JSON number drops trailing zeros, so its digits do not state the clock's precision;
                # the sources that write epoch numbers (Zeek, Suricata) keep microseconds (AS-707).
                t = T.Instant(t.seconds, t.ns, 1e-6, t.zone_known)
            return t
        return T.rfc3339(text, default_offset_s=ctx.utc_offset_s)
    except T.TimeParseError as exc:
        raise Refusal(str(exc)) from None


@converter("string")
def _string(raw: Any, ctx: ConvContext) -> str:
    return _text(raw)


@converter("trim")
def _trim(raw: Any, ctx: ConvContext) -> str:
    return _text(raw).strip()


@converter("addr")
def _addr(raw: Any, ctx: ConvContext) -> str:
    return as_address(raw)


@converter("addr_nonzero")
def _addr_nonzero(raw: Any, ctx: ConvContext) -> str:
    a = as_address(raw)
    if ipaddress.ip_address(a).is_unspecified:
        raise Refusal("unspecified address (0.0.0.0 or ::)")
    return a


@converter("mac")
def _mac(raw: Any, ctx: ConvContext) -> str:
    return as_mac(raw)


@converter("port")
def _port(raw: Any, ctx: ConvContext) -> int:
    p = as_int(raw)
    if not 0 <= p <= 65535:
        raise Refusal(f"port out of range: {p}")
    return p


@converter("count")
def _count(raw: Any, ctx: ConvContext) -> int:
    n = as_int(raw)
    if n < 0:
        raise Refusal(f"negative count: {n}")
    return n


@converter("int")
def _int(raw: Any, ctx: ConvContext) -> int:
    return as_int(raw)


@converter("int_text")
def _int_text(raw: Any, ctx: ConvContext) -> int:
    return as_int(raw)


@converter("int_or_none")
def _int_or_none(raw: Any, ctx: ConvContext) -> int:
    if isinstance(_scalar(raw), str) and not _text(raw).strip():
        raise Refusal("empty")
    return as_int(raw)


@converter("flow_id")
def _flow_id(raw: Any, ctx: ConvContext) -> str:
    return str(as_int(raw))


@converter("double")
def _double(raw: Any, ctx: ConvContext) -> float:
    return as_float(raw)


@converter("interval")
def _interval(raw: Any, ctx: ConvContext) -> float:
    x = as_float(raw)
    if x < 0:
        raise Refusal(f"negative interval: {x}")
    return x


@converter("ms")
def _ms(raw: Any, ctx: ConvContext) -> float:
    x = as_float(raw)
    if x < 0:
        raise Refusal(f"negative duration: {x}")
    return x / 1000.0


@converter("time")
def _time(raw: Any, ctx: ConvContext) -> T.Instant:
    return _instant(raw, ctx)


@converter("rfc3339")
def _rfc3339(raw: Any, ctx: ConvContext) -> T.Instant:
    try:
        return T.rfc3339(_text(raw), default_offset_s=ctx.utc_offset_s)
    except T.TimeParseError as exc:
        raise Refusal(str(exc)) from None


@converter("epoch_seconds")
def _epoch_seconds(raw: Any, ctx: ConvContext) -> T.Instant:
    return T.epoch_integer(as_int(raw), "s")


@converter("epoch_ns")
def _epoch_ns(raw: Any, ctx: ConvContext) -> T.Instant:
    return T.epoch_integer(as_int(raw), "ns")


@converter("ms_time")
def _ms_time(raw: Any, ctx: ConvContext) -> T.Instant:
    return T.epoch_integer(as_int(raw), "ms")


@converter("journald_us")
def _journald_us(raw: Any, ctx: ConvContext) -> T.Instant:
    return T.epoch_integer(as_int(raw), "us")


@converter("epoch_decimal")
def _epoch_decimal(raw: Any, ctx: ConvContext) -> T.Instant:
    try:
        return T.decimal_epoch(_text(raw))
    except T.TimeParseError as exc:
        raise Refusal(str(exc)) from None


@converter("syslog_time")
def _syslog_time(raw: Any, ctx: ConvContext) -> T.Instant:
    raw = _scalar(raw)
    if isinstance(raw, T.Instant):
        return raw
    text = _text(raw).strip()
    try:
        if text[:4].isdigit():
            return T.rfc3339(text, default_offset_s=ctx.utc_offset_s)
        if ctx.year is None:
            raise Refusal("RFC 3164 timestamp without a year and no assumed_year configured")
        return T.rfc3164(text, year=ctx.year, offset_s=ctx.utc_offset_s)
    except T.TimeParseError as exc:
        raise Refusal(str(exc)) from None


@converter("snort_time")
def _snort_time(raw: Any, ctx: ConvContext) -> T.Instant:
    try:
        return T.snort(_text(raw), year=ctx.year, offset_s=ctx.utc_offset_s)
    except T.TimeParseError as exc:
        raise Refusal(str(exc)) from None


@converter("cef_time")
def _cef_time(raw: Any, ctx: ConvContext) -> T.Instant:
    try:
        return T.cef(_text(raw), year=ctx.year, offset_s=ctx.utc_offset_s)
    except T.TimeParseError as exc:
        raise Refusal(str(exc)) from None


@converter("leef_time")
def _leef_time(raw: Any, ctx: ConvContext) -> T.Instant:
    text = _text(raw).strip()
    if text.isdigit():
        return T.epoch_integer(int(text), "ms")
    pattern = ctx.values.get("devTimeFormat")
    try:
        if pattern:
            return T.java_pattern(text, str(pattern), year=ctx.year, offset_s=ctx.utc_offset_s)
        return T.cef(text, year=ctx.year, offset_s=ctx.utc_offset_s)
    except T.TimeParseError as exc:
        raise Refusal(str(exc)) from None


@converter("bool")
def _bool(raw: Any, ctx: ConvContext) -> bool:
    return as_bool(raw)


@converter("truth_value")
def _truth_value(raw: Any, ctx: ConvContext) -> bool:
    # SNMP TruthValue (RFC 2579): true(1), false(2).
    v = as_int(raw)
    if v not in (1, 2):
        raise Refusal(f"not a TruthValue: {v}")
    return v == 1


@converter("flag01")
def _flag01(raw: Any, ctx: ConvContext) -> int:
    # A one-bit flag written as 0 / 1, true / false or True / False (tshark JSON, EK, PyShark).
    return 1 if as_bool(raw) else 0


@converter("bool01")
def _bool01(raw: Any, ctx: ConvContext) -> int:
    return 1 if as_bool(raw) else 0


@converter("auth_result")
def _auth_result(raw: Any, ctx: ConvContext) -> int:
    return 1 if as_bool(raw) else 2


@converter("win_yesno")
def _win_yesno(raw: Any, ctx: ConvContext) -> int:
    t = _text(raw).strip().lower()
    # Windows message-table references: %%1842 = Yes, %%1843 = No (Security auditing documentation).
    if t in ("%%1842", "yes", "true", "1"):
        return 1
    if t in ("%%1843", "no", "false", "0"):
        return 0
    raise Refusal(f"not a Windows yes/no value: {t!r}")


_PROTO_NAMES: dict[str, int] = {
    "icmp": 1, "igmp": 2, "ipv4": 4, "ipip": 4, "tcp": 6, "egp": 8, "udp": 17, "ipv6": 41, "rsvp": 46, "gre": 47,
    "esp": 50, "ah": 51, "icmp6": 58, "icmpv6": 58, "ipv6icmp": 58, "ospf": 89, "pim": 103, "vrrp": 112,
    "l2tp": 115, "sctp": 132, "udplite": 136,
}


@converter("proto_name")
def _proto_name(raw: Any, ctx: ConvContext) -> int:
    raw = _scalar(raw)
    if isinstance(raw, int) and not isinstance(raw, bool):
        if not 0 <= raw <= 255:
            raise Refusal(f"IP protocol out of range: {raw}")
        return raw
    t = C.norm(_text(raw))
    if t.isdigit():
        return _proto_name(int(t), ctx)
    if t == "icmp" and ctx.ipv6:
        return 58                                     # Zeek writes "icmp" for ICMPv6 as well
    code = _PROTO_NAMES.get(t)
    if code is None:
        raise Refusal(f"unknown protocol name {t!r}")
    return code


@converter("app_proto")
def _app_proto(raw: Any, ctx: ConvContext) -> int:
    raw = _scalar(raw) if not isinstance(raw, list | tuple) else (raw[0] if raw else "")
    text = _text(raw)
    first = text.split(",")[0].strip()
    if C.norm(first) in C.APP_PROTO_NOT_A_LABEL:
        raise Refusal(f"not a protocol label: {first!r}")
    return code_lookup("app_proto", first)


@converter("app_proto_list")
def _app_proto_list(raw: Any, ctx: ConvContext) -> int:
    items = list(raw) if isinstance(raw, list | tuple) else [raw]
    if not items:
        raise Refusal("empty service list")
    return _app_proto(items[0], ctx)


@converter("action_word")
def _action_word(raw: Any, ctx: ConvContext) -> int:
    return code_lookup("event_action", raw)


@converter("hex")
def _hex(raw: Any, ctx: ConvContext) -> int:
    return as_hex(raw)


@converter("hex_nonzero")
def _hex_nonzero(raw: Any, ctx: ConvContext) -> int:
    v = as_hex(raw)
    if v == 0:
        raise Refusal("zero status (no failure)")
    return v


@converter("hex_first")
def _hex_first(raw: Any, ctx: ConvContext) -> int:
    if isinstance(raw, list | tuple):
        if not raw:
            raise Refusal("empty list")
        raw = raw[0]
    return as_hex(raw)


@converter("hex_tcp_flags")
def _hex_tcp_flags(raw: Any, ctx: ConvContext) -> int:
    # Suricata writes the flags as two hex digits without a prefix ("1b").
    t = _text(raw).strip()
    try:
        return int(t, 16) & 0x3F
    except ValueError:
        raise Refusal(f"not hex TCP flags: {t!r}") from None


@converter("hex_tcp_bits8")
def _hex_tcp_bits8(raw: Any, ctx: ConvContext) -> int:
    return as_hex(raw) & 0xFF


@converter("tcp_bits8")
def _tcp_bits8(raw: Any, ctx: ConvContext) -> int:
    v = as_int(raw)
    if v < 0:
        raise Refusal("negative TCP flags")
    return v & 0xFF


def _tcp_flags_of(ctx: ConvContext) -> int | None:
    raw = ctx.values.get("tcp.flags")
    if raw is None:
        return None
    try:
        return as_hex(raw)
    except Refusal:
        return None


@converter("count_if_syn")
def _count_if_syn(raw: Any, ctx: ConvContext) -> int:
    flags = _tcp_flags_of(ctx)
    if flags is None or (flags & 0x12) != 0x02:
        raise Refusal("not an initial window (the packet is not a SYN without ACK)")
    return _count(raw, ctx)


@converter("present_one")
def _present_one(raw: Any, ctx: ConvContext) -> int:
    return 1


@converter("present_bool")
def _present_bool(raw: Any, ctx: ConvContext) -> bool:
    return True


@converter("first_int")
def _first_int(raw: Any, ctx: ConvContext) -> int:
    if isinstance(raw, list | tuple):
        if not raw:
            raise Refusal("empty list")
        raw = raw[0]
    return as_int(raw)


@converter("first_text")
def _first_text(raw: Any, ctx: ConvContext) -> str:
    if isinstance(raw, list | tuple):
        if not raw:
            raise Refusal("empty list")
        raw = raw[0]
    return _text(raw)


@converter("first_addr")
def _first_addr(raw: Any, ctx: ConvContext) -> str:
    if isinstance(raw, list | tuple):
        if not raw:
            raise Refusal("empty list")
        raw = raw[0]
    return as_address(raw)


def _items(raw: Any) -> list[Any]:
    if raw is None:
        return []
    if isinstance(raw, list | tuple):
        return list(raw)
    return [raw]


@converter("list")
def _list(raw: Any, ctx: ConvContext) -> tuple[str, ...]:
    return tuple(_text(x) for x in _items(raw))


@converter("list_int")
def _list_int(raw: Any, ctx: ConvContext) -> tuple[int, ...]:
    return tuple(as_int(x) for x in _items(raw))


@converter("list_float")
def _list_float(raw: Any, ctx: ConvContext) -> tuple[float, ...]:
    return tuple(as_float(x) for x in _items(raw))


@converter("list_interval")
def _list_interval(raw: Any, ctx: ConvContext) -> tuple[float, ...]:
    out = tuple(as_float(x) for x in _items(raw))
    if any(x < 0 for x in out):
        raise Refusal("negative interval in list")
    return out


@converter("list_addr")
def _list_addr(raw: Any, ctx: ConvContext) -> tuple[str, ...]:
    return tuple(as_address(x) for x in _items(raw))


@converter("join_slash")
def _join_slash(raw: Any, ctx: ConvContext) -> str:
    parts = [_text(x) for x in _items(raw)]
    if not parts:
        raise Refusal("empty principal name")
    return "/".join(parts)


@converter("map")
def _map(raw: Any, ctx: ConvContext) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise Refusal(f"expected an object, got {type(raw).__name__}")
    return dict(raw)


@converter("zbits")
def _zbits(raw: Any, ctx: ConvContext) -> int:
    # The three bits after RA in the DNS header: Z, AD, CD (RFC 1035 4.1.1; RFC 4035 3.2), as Zeek's
    # three-bit Z value (4 = Z, 2 = AD, 1 = CD; AS-689) -> proto.dns.flags bits 0x10, 0x20, 0x40.
    z = as_int(raw)
    if not 0 <= z <= 7:
        raise Refusal(f"Z field out of range: {z}")
    return (0x10 if z & 4 else 0) | (0x20 if z & 2 else 0) | (0x40 if z & 1 else 0)


@converter("dns_header_flags")
def _dns_header_flags(raw: Any, ctx: ConvContext) -> int:
    # Suricata's 16-bit DNS header flags word, hex ("8180") -> proto.dns.flags bits.
    t = _text(raw).strip()
    try:
        w = int(t, 16)
    except ValueError:
        raise Refusal(f"not hex DNS flags: {t!r}") from None
    return _word_to_dns_flags(w)


@converter("dns_flags_word")
def _dns_flags_word(raw: Any, ctx: ConvContext) -> int:
    # The DNS header flags word written as 0x-prefixed hex or decimal (tshark JSON and EK).
    return _word_to_dns_flags(as_hex(raw))


def _word_to_dns_flags(w: int) -> int:
    out = 0
    for wire, bit in ((0x0400, 0x01), (0x0200, 0x02), (0x0100, 0x04), (0x0080, 0x08), (0x0040, 0x10),
                      (0x0020, 0x20), (0x0010, 0x40)):
        if w & wire:
            out |= bit
    return out


@converter("last_dhcp_type")
def _last_dhcp_type(raw: Any, ctx: ConvContext) -> int:
    items = _items(raw)
    if not items:
        raise Refusal("no DHCP message types")
    return code_lookup("dhcp_message", items[-1])


@converter("krb_request")
def _krb_request(raw: Any, ctx: ConvContext) -> int:
    t = C.norm(_text(raw))
    if t in ("as", "asreq"):
        return 10
    if t in ("tgs", "tgsreq"):
        return 12
    raise Refusal(f"unknown Kerberos request type {t!r}")


@converter("ipfix_end_reason")
def _ipfix_end_reason(raw: Any, ctx: ConvContext) -> int:
    # RFC 5102 flowEndReason: 1 idle timeout, 2 active timeout, 3 end of flow detected, 4 forced end,
    # 5 lack of resources -> flow.end_reason codes (AS-681).
    v = as_int(raw)
    mapped = {1: 3, 2: 5, 3: 8, 4: 6, 5: 7}.get(v)
    if mapped is None:
        raise Refusal(f"unknown flowEndReason {v}")
    return mapped


@converter("firewall_event")
def _firewall_event(raw: Any, ctx: ConvContext) -> int:
    # firewallEvent (IANA IE 233): 1 created, 2 deleted, 5 updated -> allowed; 3 denied; 4 alert -> observed.
    v = as_int(raw)
    mapped = {1: 1, 2: 1, 5: 1, 3: 2, 4: 3}.get(v)
    if mapped is None:
        raise Refusal(f"firewallEvent {v} is not an action")
    return mapped


@converter("u2_blocked")
def _u2_blocked(raw: Any, ctx: ConvContext) -> int:
    # unified2 `blocked`: 0 not blocked, 1 blocked, 2 would have been blocked (AS-685).
    v = as_int(raw)
    mapped = {0: 1, 1: 2, 2: 3}.get(v)
    if mapped is None:
        raise Refusal(f"unknown unified2 blocked value {v}")
    return mapped


def _severity_bucket(n: int) -> int:
    # 0-10 device severity -> OCSF severity_id: 0-3 Low, 4-6 Medium, 7-8 High, 9-10 Critical (AS-699).
    if not 0 <= n <= 10:
        raise Refusal(f"severity out of range 0-10: {n}")
    return 2 if n <= 3 else 3 if n <= 6 else 4 if n <= 8 else 5


@converter("cef_severity")
def _cef_severity(raw: Any, ctx: ConvContext) -> int:
    t = _text(raw).strip()
    words = {"unknown": 0, "low": 2, "medium": 3, "high": 4, "veryhigh": 5}
    if C.norm(t) in words:
        return words[C.norm(t)]
    return _severity_bucket(as_int(t))


@converter("leef_severity")
def _leef_severity(raw: Any, ctx: ConvContext) -> int:
    n = as_int(raw)
    if not 1 <= n <= 10:
        raise Refusal(f"LEEF sev out of range 1-10: {n}")
    return _severity_bucket(n)


@converter("ssh_major")
def _ssh_major(raw: Any, ctx: ConvContext) -> int:
    t = _text(raw).strip()
    m = re.match(r"^(?:SSH-)?(\d+)\.(\d+)", t)
    if m is None:
        raise Refusal(f"not an SSH protocol version: {t!r}")
    major, minor = int(m.group(1)), int(m.group(2))
    # "1.99" announces compatibility with both versions (RFC 4253 section 5.1); the session is SSH 2.
    return 2 if (major, minor) == (1, 99) else major


@converter("smb1_cmd")
def _smb1_cmd(raw: Any, ctx: ConvContext) -> int:
    v = as_int(raw)
    if not 0 <= v <= 255:
        raise Refusal(f"SMB1 command out of range: {v}")
    return 0x100 + v


@converter("modbus_fc")
def _modbus_fc(raw: Any, ctx: ConvContext) -> int:
    v = as_int(raw)
    if not 0 <= v <= 255:
        raise Refusal(f"Modbus function code out of range: {v}")
    return v & 0x7F                                   # the high bit marks an exception response


@converter("dnp3_group")
def _dnp3_group(raw: Any, ctx: ConvContext) -> int:
    items = _items(raw)
    if not items:
        raise Refusal("no DNP3 object")
    v = as_hex(items[0])
    return (v >> 8) & 0xFF                            # object header: group in the high byte


@converter("oc_port_speed")
def _oc_port_speed(raw: Any, ctx: ConvContext) -> float:
    t = C.norm(_text(raw).split(":")[-1])
    bps = C.PORT_SPEED_BPS.get(t)
    if bps is None:
        raise Refusal(f"unknown or unknown-speed port identity {raw!r}")
    return bps


@converter("sflow_if")
def _sflow_if(raw: Any, ctx: ConvContext) -> int:
    # sFlow v5 interface encoding: 2-bit format, 30-bit value. Format 0 is an ifIndex; 0x3FFFFFFF is
    # "internal"; format 1 is a discard reason, format 2 a count of interfaces.
    v = as_int(raw)
    fmt, value = (v >> 30) & 0x3, v & 0x3FFFFFFF
    if fmt != 0:
        raise Refusal("not a single interface (discarded packet or multiple interfaces)")
    if value == 0x3FFFFFFF:
        raise Refusal("internal interface")
    return value


@converter("v5_sampling")
def _v5_sampling(raw: Any, ctx: ConvContext) -> int:
    v = as_int(raw)
    mode, interval = (v >> 14) & 0x3, v & 0x3FFF
    if mode == 0 or interval == 0:
        raise Refusal("sampling not reported in the header")
    return interval


@converter("win_ip")
def _win_ip(raw: Any, ctx: ConvContext) -> str:
    t = _text(raw).strip()
    if t in ("-", ""):
        raise Refusal("no address (-)")
    return as_address(t)


@converter("win_port")
def _win_port(raw: Any, ctx: ConvContext) -> int:
    t = _text(raw).strip()
    if t in ("-", "", "0"):
        raise Refusal("no port")
    return _port(t, ctx)


@converter("win_name")
def _win_name(raw: Any, ctx: ConvContext) -> str:
    t = _text(raw).strip().lstrip("\\")
    if t in ("-", ""):
        raise Refusal("no name (-)")
    return t


@converter("win_list")
def _win_list(raw: Any, ctx: ConvContext) -> tuple[str, ...]:
    if isinstance(raw, list | tuple):
        return tuple(_text(x) for x in raw)
    return tuple(_text(raw).split())


@converter("win_etype")
def _win_etype(raw: Any, ctx: ConvContext) -> int:
    v = as_hex(raw)
    if v in (0xFFFFFFFF, -1):
        raise Refusal("no ticket encryption type (failure)")
    return v


@converter("ocsf_answers_rdata")
def _ocsf_answers_rdata(raw: Any, ctx: ConvContext) -> tuple[str, ...]:
    out = []
    for a in _items(raw):
        if not isinstance(a, Mapping) or "rdata" not in a:
            raise Refusal("a DNS answer without rdata")
        out.append(_text(a["rdata"]))
    return tuple(out)


@converter("ocsf_ip_version")
def _ocsf_ip_version(raw: Any, ctx: ConvContext) -> int:
    v = as_int(raw)
    if v not in (4, 6):
        raise Refusal(f"protocol_ver_id {v} is not 4 or 6")
    return v


@converter("ocsf_rcode")
def _ocsf_rcode(raw: Any, ctx: ConvContext) -> int:
    v = as_int(raw)
    if v == 99 or v < 0:
        raise Refusal("rcode 'Other' has no DNS code")
    return v


#: OCSF SSH auth_type_id -> auth.method code.
_OCSF_SSH_AUTH = {1: 7, 2: 5, 3: 4, 4: 3, 5: 1, 6: 2, 99: 99}


@converter("ocsf_ssh_auth")
def _ocsf_ssh_auth(raw: Any, ctx: ConvContext) -> int:
    v = as_int(raw)
    if v not in _OCSF_SSH_AUTH:
        raise Refusal(f"unknown SSH auth_type_id {v}")
    return _OCSF_SSH_AUTH[v]


@converter("ocsf_status_result")
def _ocsf_status_result(raw: Any, ctx: ConvContext) -> int:
    v = as_int(raw)
    if v not in (1, 2):
        raise Refusal(f"status_id {v} is not success or failure")
    return v


_SNORT_FLAG_POS = (0x80, 0x40, 0x20, 0x10, 0x08, 0x04, 0x02, 0x01)       # "12UAPRSF"
_SNORT_FLAG_LETTERS = {"C": 0x80, "E": 0x40, "1": 0x80, "2": 0x40, "U": 0x20, "A": 0x10, "P": 0x08, "R": 0x04,
                       "S": 0x02, "F": 0x01}


@converter("snort_flags")
def _snort_flags(raw: Any, ctx: ConvContext) -> int:
    # Snort prints TCP flags as eight positions "12UAPRSF" with "*" for a clear bit ("***A**S*").
    t = _text(raw).strip()
    if len(t) == 8 and all(c == "*" or c in _SNORT_FLAG_LETTERS for c in t.upper()):
        return sum(bit for c, bit in zip(t, _SNORT_FLAG_POS, strict=True) if c != "*")
    if t and all(c in _SNORT_FLAG_LETTERS for c in t.upper()):
        return sum({_SNORT_FLAG_LETTERS[c] for c in t.upper()})
    raise Refusal(f"not a Snort TCP flags string: {t!r}")


@converter("b64")
def _b64(raw: Any, ctx: ConvContext) -> bytes:
    try:
        return base64.b64decode(_text(raw), validate=True)
    except (ValueError, TypeError):
        raise Refusal("not base64") from None


__all__ = ["ConvContext", "Converter", "Refusal", "as_address", "as_bool", "as_float", "as_hex", "as_int", "as_mac",
           "code_lookup", "converter", "get", "names"]
