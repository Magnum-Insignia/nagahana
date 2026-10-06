"""Timestamp parsing: every source time to UTC epoch seconds, keeping exact nanoseconds where they exist.

An `Instant` holds the float64 epoch seconds the data model orders by, the exact integer nanoseconds
when the source's precision exceeds what a float64 keeps (about 0.24 microseconds at present epoch
values), the resolution of the source's clock text, and whether the source stated its time zone.

Formats
-------
- decimal epoch seconds ("1700000000.123456789"; Zeek, tshark): digits are read exactly, never through
  a float.
- integer epoch in seconds, milliseconds, microseconds or nanoseconds.
- RFC 3339 / ISO 8601 date-times with 1 to 9 fractional digits and a "Z", "+hh:mm" or "+hhmm" offset
  (Suricata "+0000", Windows SystemTime with seven digits). Without an offset the configured UTC offset
  applies; without one either the text is read as UTC and the zone is reported as unknown.
- RFC 3164 syslog "Mmm dd hh:mm:ss" (no year, no zone): year and offset come from configuration.
- Snort "mm/dd-hh:mm:ss.ffffff" (no year) and "yy/mm/dd-hh:mm:ss.ffffff".
- CEF "MMM dd yyyy HH:mm:ss[.SSS] [zone]" and epoch milliseconds; LEEF devTime with a Java
  SimpleDateFormat pattern (devTimeFormat) or epoch milliseconds.
- NTP 64-bit timestamps (RFC 5905 section 6; IPFIX dateTimeMicroseconds and dateTimeNanoseconds,
  RFC 7011 section 6.1.9-10): seconds since 1900-01-01 and a 32-bit binary fraction.
"""

from __future__ import annotations

import calendar
import re
from dataclasses import dataclass

NS = 1_000_000_000
#: Seconds from the NTP era 0 origin (1900-01-01) to the Unix epoch (RFC 868, RFC 5905).
NTP_EPOCH_OFFSET = 2_208_988_800

MONTHS: dict[str, int] = {m: i for i, m in enumerate(
    ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"), start=1)}


class TimeParseError(ValueError):
    """A timestamp text that is not in the expected format (the reason is the message)."""


@dataclass(frozen=True, slots=True)
class Instant:
    """A point in time: float epoch seconds, exact nanoseconds when known, resolution, zone knowledge."""

    seconds: float
    ns: int | None
    resolution: float
    zone_known: bool = True

    @classmethod
    def from_ns(cls, ns: int, resolution: float = 1e-9, zone_known: bool = True) -> Instant:
        return cls(ns / NS, int(ns), resolution, zone_known)


_DECIMAL = re.compile(r"^\s*(-?)(\d+)(?:\.(\d+))?\s*$")


def _frac_ns(frac: str) -> int:
    """Nanoseconds of a fractional-second digit string (beyond nine digits: truncated)."""
    return int((frac + "000000000")[:9]) if frac else 0


def decimal_epoch(text: str) -> Instant:
    """Decimal epoch seconds, read exactly ("1700000000.123456789")."""
    m = _DECIMAL.match(text)
    if m is None:
        raise TimeParseError(f"not a decimal epoch: {text!r}")
    sign = -1 if m.group(1) else 1
    whole, frac = int(m.group(2)), m.group(3) or ""
    ns = sign * (whole * NS + _frac_ns(frac))
    return Instant(ns / NS, ns, 10.0 ** -min(len(frac), 9) if frac else 1.0)


def epoch_integer(value: int, unit: str) -> Instant:
    """Integer epoch time in "s", "ms", "us" or "ns"."""
    scale = {"s": NS, "ms": 1_000_000, "us": 1_000, "ns": 1}[unit]
    ns = int(value) * scale
    return Instant(ns / NS, ns, {"s": 1.0, "ms": 1e-3, "us": 1e-6, "ns": 1e-9}[unit])


def epoch_number(value: float | int) -> Instant:
    """A numeric epoch-seconds value from a typed source (JSON); exact ns only for integers."""
    if isinstance(value, int):
        return epoch_integer(value, "s")
    return Instant(float(value), None, 1e-6)


_RFC3339 = re.compile(
    r"^\s*(\d{4})-(\d{2})-(\d{2})[Tt ](\d{2}):(\d{2}):(\d{2})(?:[.,](\d{1,12}))?\s*"
    r"(Z|z|[+-]\d{2}:?\d{2}|UTC|GMT)?\s*$"
)


def _offset_seconds(zone: str) -> int:
    if zone in ("Z", "z", "UTC", "GMT"):
        return 0
    sign = -1 if zone[0] == "-" else 1
    digits = zone[1:].replace(":", "")
    return sign * (int(digits[:2]) * 3600 + int(digits[2:4]) * 60)


def _validate_civil(y: int, mo: int, d: int, h: int, mi: int, s: int) -> None:
    if not (1 <= mo <= 12 and 1 <= d <= calendar.monthrange(y, mo)[1] and 0 <= h <= 23 and 0 <= mi <= 59 and 0 <= s <= 60):
        raise TimeParseError(f"invalid civil time {y:04d}-{mo:02d}-{d:02d} {h:02d}:{mi:02d}:{s:02d}")


def civil_to_instant(y: int, mo: int, d: int, h: int, mi: int, s: int, frac: str, offset_s: float | None,
                     zone_known: bool) -> Instant:
    """Calendar fields (UTC unless `offset_s`, the local offset east of UTC) to an Instant."""
    _validate_civil(y, mo, d, h, mi, s)
    leap = s == 60                           # a leap second is read as the last second of the minute
    whole = calendar.timegm((y, mo, d, h, mi, 59 if leap else s, 0, 0, 0))
    ns = whole * NS + _frac_ns(frac)
    if offset_s is not None:
        ns -= int(round(offset_s * NS))
    return Instant(ns / NS, ns, 10.0 ** -min(len(frac), 9) if frac else 1.0, zone_known)


def rfc3339(text: str, *, default_offset_s: float | None = None) -> Instant:
    """An RFC 3339 / ISO 8601 date-time (see the module docstring for the zone rule)."""
    m = _RFC3339.match(text)
    if m is None:
        raise TimeParseError(f"not an RFC 3339 date-time: {text!r}")
    y, mo, d, h, mi, s = (int(m.group(i)) for i in range(1, 7))
    frac, zone = m.group(7) or "", m.group(8)
    if zone is not None:
        return civil_to_instant(y, mo, d, h, mi, s, frac, _offset_seconds(zone), True)
    return civil_to_instant(y, mo, d, h, mi, s, frac, default_offset_s, default_offset_s is not None)


_RFC3164 = re.compile(r"^\s*([A-Za-z]{3})\s+(\d{1,2})\s+(\d{2}):(\d{2}):(\d{2})(?:\.(\d{1,9}))?\s*$")


def rfc3164(text: str, *, year: int, offset_s: float | None) -> Instant:
    """An RFC 3164 timestamp "Mmm dd hh:mm:ss"; `year` and `offset_s` come from configuration."""
    m = _RFC3164.match(text)
    if m is None or m.group(1).lower() not in MONTHS:
        raise TimeParseError(f"not an RFC 3164 timestamp: {text!r}")
    return civil_to_instant(year, MONTHS[m.group(1).lower()], int(m.group(2)), int(m.group(3)), int(m.group(4)),
                            int(m.group(5)), m.group(6) or "", offset_s, offset_s is not None)


_SNORT = re.compile(r"^\s*(?:(\d{2})/)?(\d{2})/(\d{2})-(\d{2}):(\d{2}):(\d{2})(?:\.(\d{1,9}))?\s*$")


def snort(text: str, *, year: int | None, offset_s: float | None) -> Instant:
    """Snort alert time "mm/dd-hh:mm:ss.ffffff", or "yy/mm/dd-..." when Snort logs the year."""
    m = _SNORT.match(text)
    if m is None:
        raise TimeParseError(f"not a Snort timestamp: {text!r}")
    if m.group(1) is not None:
        y = 2000 + int(m.group(1))
    elif year is not None:
        y = year
    else:
        raise TimeParseError("Snort timestamp without a year and no assumed_year configured")
    return civil_to_instant(y, int(m.group(2)), int(m.group(3)), int(m.group(4)), int(m.group(5)), int(m.group(6)),
                            m.group(7) or "", offset_s, offset_s is not None)


_CEF_TIME = re.compile(
    r"^\s*([A-Za-z]{3})\s+(\d{1,2})(?:\s+(\d{4}))?\s+(\d{2}):(\d{2}):(\d{2})(?:\.(\d{1,9}))?"
    r"(?:\s+(Z|UTC|GMT|[+-]\d{2}:?\d{2}))?\s*$"
)


def cef(text: str, *, year: int | None, offset_s: float | None) -> Instant:
    """CEF time: epoch milliseconds, or "MMM dd [yyyy] HH:mm:ss[.SSS] [zone]" (ArcSight CEF standard)."""
    t = text.strip()
    if t.isdigit():
        return epoch_integer(int(t), "ms")
    m = _CEF_TIME.match(t)
    if m is None or m.group(1).lower() not in MONTHS:
        raise TimeParseError(f"not a CEF time: {text!r}")
    if m.group(3) is not None:
        y = int(m.group(3))
    elif year is not None:
        y = year
    else:
        raise TimeParseError("CEF time without a year and no assumed_year configured")
    zone = m.group(8)
    off = _offset_seconds(zone) if zone else offset_s
    return civil_to_instant(y, MONTHS[m.group(1).lower()], int(m.group(2)), int(m.group(4)), int(m.group(5)),
                            int(m.group(6)), m.group(7) or "", off, zone is not None or offset_s is not None)


#: Java SimpleDateFormat letters -> (regex, group name).
_JAVA_TOKENS: dict[str, tuple[str, str]] = {
    "yyyy": (r"(?P<y>\d{4})", "y"), "yy": (r"(?P<yy>\d{2})", "yy"), "MMM": (r"(?P<mon>[A-Za-z]{3})", "mon"),
    "MM": (r"(?P<mo>\d{2})", "mo"), "M": (r"(?P<mo>\d{1,2})", "mo"), "dd": (r"(?P<d>\d{2})", "d"),
    "d": (r"(?P<d>\d{1,2})", "d"), "HH": (r"(?P<h>\d{2})", "h"), "H": (r"(?P<h>\d{1,2})", "h"),
    "hh": (r"(?P<h12>\d{2})", "h12"), "h": (r"(?P<h12>\d{1,2})", "h12"), "mm": (r"(?P<mi>\d{2})", "mi"),
    "ss": (r"(?P<s>\d{2})", "s"), "SSS": (r"(?P<f>\d{3})", "f"), "a": (r"(?P<ampm>[AaPp][Mm])", "ampm"),
    "zzz": (r"(?P<z>[A-Za-z]{1,5}|[+-]\d{2}:?\d{2})", "z"), "z": (r"(?P<z>[A-Za-z]{1,5}|[+-]\d{2}:?\d{2})", "z"),
    "Z": (r"(?P<z>[+-]\d{4}|Z)", "z"), "XXX": (r"(?P<z>Z|[+-]\d{2}:\d{2})", "z"), "X": (r"(?P<z>Z|[+-]\d{2})", "z"),
}
_JAVA_ORDER = sorted(_JAVA_TOKENS, key=len, reverse=True)
_JAVA_CACHE: dict[str, re.Pattern[str]] = {}


def _java_regex(pattern: str) -> re.Pattern[str]:
    cached = _JAVA_CACHE.get(pattern)
    if cached is not None:
        return cached
    out, i = [], 0
    while i < len(pattern):
        ch = pattern[i]
        if ch == "'":                                   # quoted literal
            j = pattern.find("'", i + 1)
            if j < 0:
                raise TimeParseError(f"unterminated quote in date pattern {pattern!r}")
            out.append(re.escape(pattern[i + 1:j]) if j > i + 1 else "'")
            i = j + 1
            continue
        for tok in _JAVA_ORDER:
            if pattern.startswith(tok, i):
                out.append(_JAVA_TOKENS[tok][0])
                i += len(tok)
                break
        else:
            if ch.isalpha():
                raise TimeParseError(f"unsupported date pattern letter {ch!r} in {pattern!r}")
            out.append(re.escape(ch))
            i += 1
    rx = re.compile(r"^\s*" + "".join(out) + r"\s*$")
    if len(_JAVA_CACHE) < 256:
        _JAVA_CACHE[pattern] = rx
    return rx


def java_pattern(text: str, pattern: str, *, year: int | None, offset_s: float | None) -> Instant:
    """A time written with a Java SimpleDateFormat pattern (LEEF devTimeFormat)."""
    m = _java_regex(pattern).match(text)
    if m is None:
        raise TimeParseError(f"{text!r} does not match the date pattern {pattern!r}")
    g = m.groupdict()
    if g.get("y"):
        y = int(g["y"])
    elif g.get("yy"):
        y = 2000 + int(g["yy"])
    elif year is not None:
        y = year
    else:
        raise TimeParseError("date pattern without a year and no assumed_year configured")
    if g.get("mon"):
        if g["mon"].lower() not in MONTHS:
            raise TimeParseError(f"unknown month {g['mon']!r}")
        mo = MONTHS[g["mon"].lower()]
    else:
        mo = int(g.get("mo") or 1)
    if g.get("h12") is not None:
        h = int(g["h12"]) % 12 + (12 if (g.get("ampm") or "am").lower() == "pm" else 0)
    else:
        h = int(g.get("h") or 0)
    zone = g.get("z")
    if zone and zone not in ("Z", "UTC", "GMT") and not zone[0] in "+-":
        raise TimeParseError(f"unsupported time-zone name {zone!r}")
    off = _offset_seconds(zone) if zone else offset_s
    return civil_to_instant(y, mo, int(g.get("d") or 1), h, int(g.get("mi") or 0), int(g.get("s") or 0),
                            g.get("f") or "", off, bool(zone) or offset_s is not None)


def ntp64(seconds: int, fraction: int) -> Instant:
    """An NTP 64-bit timestamp (era 0) as an Instant with nanosecond precision."""
    ns = (int(seconds) - NTP_EPOCH_OFFSET) * NS + ((int(fraction) * NS) >> 32)
    return Instant(ns / NS, ns, 2.0 ** -32)


__all__ = ["MONTHS", "NS", "NTP_EPOCH_OFFSET", "Instant", "TimeParseError", "cef", "civil_to_instant", "decimal_epoch",
           "epoch_integer", "epoch_number", "java_pattern", "ntp64", "rfc3164", "rfc3339", "snort"]
