"""Wireshark dissections -> state updates: tshark -T json, tshark -T ek, and PyShark (live tshark).

Every reader flattens one packet into {display-filter field: value or list of values} and maps it with
the one table of datamodel/maps/wireshark.py, so a packet means the same whichever way it was exported.

tshark -T json   one JSON array of packets ({"_source": {"layers": {...}}}); protocol trees nest
                 fields ("ip.flags_tree": {"ip.flags.df": "1"}) under labels. tshark repeats a key
                 when a field occurs twice ("ip.addr"); the reader keeps every occurrence (a list),
                 where a plain JSON parser would keep only the last.
tshark -T ek     newline-delimited pairs of an index line and a packet line ({"timestamp": ms,
                 "layers": {"ip": {"ip_ip_src": ...}}}); a key is "<layer>_<field with dots as
                 underscores>". Keys of the table's fields are translated back; others are retained.
PyShark          pyshark.FileCapture (tshark PDML underneath): layers expose their fields by full
                 name. Optional dependency, imported when used.

One state update per packet (packet granularity, AS-709); see the table's notes for derived fields.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import IO, Any

from nagahana.datamodel.maps.wireshark import PACKET
from nagahana.ingest import derive as DV
from nagahana.ingest.config import WiresharkConfig
from nagahana.ingest.convert import ConvContext
from nagahana.ingest.core import (
    MalformedRecord,
    RawRecord,
    StreamAdapter,
    UpdateDraft,
    iter_json_array,
    iter_lines,
    open_source,
)
from nagahana.ingest.mapping import mapper, set_field
from nagahana.ingest.timeparse import epoch_integer

ADAPTER_VERSION = "1.0.0"
RECORD_TYPE = "wireshark.packet"
_FIELDS = frozenset(PACKET.sources())
#: EK key -> display-filter field, for the fields the table maps.
EK_KEYS: dict[str, str] = {}
for _f in PACKET.sources():
    _layer = _f.split(".", 1)[0]
    EK_KEYS[f"{_layer}_{_f.replace('.', '_')}"] = _f
    EK_KEYS.setdefault(_f.replace(".", "_"), _f)


class _Duplicates(list):  # type: ignore[type-arg]
    """Values of a key that tshark wrote more than once in one object."""


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in pairs:
        if k in out:
            prev = out[k]
            if isinstance(prev, _Duplicates):
                prev.append(v)
            else:
                out[k] = _Duplicates([prev, v])
        else:
            out[k] = v
    return out


def flatten_layers(layers: dict[str, Any]) -> dict[str, Any]:
    """{field: value or list} from tshark JSON layers (trees and labels walked, duplicates kept)."""
    out: dict[str, Any] = {}

    def add(key: str, value: Any) -> None:
        if key in out:
            prev = out[key]
            out[key] = (prev if isinstance(prev, list) else [prev]) + (value if isinstance(value, list) else [value])
        else:
            out[key] = value

    def walk(obj: Any) -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, dict) or (isinstance(v, list) and v and all(isinstance(x, dict) for x in v)):
                    for x in (v if isinstance(v, list) else [v]):
                        walk(x)
                elif "." in k and not k.endswith("_tree"):
                    vals = list(v) if isinstance(v, _Duplicates) else v
                    add(k, vals)
        elif isinstance(obj, list):
            for x in obj:
                walk(x)

    for layer in layers.values():
        walk(layer)
    return out


def from_ek(layers: dict[str, Any]) -> dict[str, Any]:
    """{field: value} from tshark EK layers; unknown keys are kept as "ek.<key>"."""
    out: dict[str, Any] = {}
    for layer in layers.values():
        if not isinstance(layer, dict):
            continue
        for k, v in layer.items():
            if k == "text" or isinstance(v, dict):
                continue
            out[EK_KEYS.get(k, f"ek.{k}")] = v
    return out


class _WiresharkBase(StreamAdapter):
    name = "tshark"
    source_type = "wireshark"
    version = ADAPTER_VERSION

    def __init__(self, source: str | Path | bytes | IO[bytes], *, config: WiresharkConfig | None = None,
                 **kw: Any) -> None:
        self.config = config or WiresharkConfig()
        super().__init__(source, self.config.common, **kw)
        clock = self.config.common.clock
        self._ctx = ConvContext(year=clock.assumed_year,
                                utc_offset_s=None if clock.utc_offset_hours is None else clock.utc_offset_hours * 3600.0)

    def map_packet(self, fields: dict[str, Any], raw: RawRecord) -> UpdateDraft:
        """Map one flattened packet (module docstring)."""
        d = UpdateDraft(RECORD_TYPE, raw)
        src = RECORD_TYPE
        values = {k: v for k, v in fields.items() if k}
        self._ctx.values = values
        self._ctx.ipv6 = "ipv6.src" in values
        mapper(RECORD_TYPE).apply(values, d, self._ctx, self.stats, source=src,
                                  keep_unmapped=self.config.common.keep_unmapped)
        self._derive(d, values, src)
        return d

    def _derive(self, d: UpdateDraft, v: dict[str, Any], src: str) -> None:
        set_field(d, "flow.packets_fwd", 1, src, self.stats)
        ip_len = d.value("pkt.ip_len_min")
        if ip_len is None and v.get("ipv6.plen") is not None:
            try:
                ip_len = int(_first(v["ipv6.plen"])) + 40
                set_field(d, "pkt.ip_len_min", ip_len, src, self.stats)
            except (TypeError, ValueError):
                self.stats.refuse("pkt.ip_len_min", "bad ipv6.plen")
        if ip_len is not None:
            set_field(d, "pkt.ip_len_max", int(ip_len), src, self.stats)
            set_field(d, "flow.bytes_fwd", int(ip_len), src, self.stats)
        ttl = d.value("pkt.ttl_mean")
        if ttl is not None:
            set_field(d, "pkt.ttl_min", int(ttl), src, self.stats)
            set_field(d, "pkt.ttl_max", int(ttl), src, self.stats)
        flags = d.value("flow.tcp_flags_fwd")
        if flags is not None:
            set_field(d, "flow.tcp_flags", int(flags) & 0x3F, src, self.stats)
            win = v.get("tcp.window_size_value")
            if (int(flags) & 0x12) == 0x02 and win is not None:          # a SYN without ACK: its window is initial
                try:
                    set_field(d, "pkt.tcp_window_init_fwd", int(_first(win)), src, self.stats)
                except (TypeError, ValueError):
                    self.stats.refuse("pkt.tcp_window_init_fwd", "bad tcp.window_size_value")
        if "tcp.srcport" in v:
            set_field(d, "flow.protocol", 6, src, self.stats)
        elif "udp.srcport" in v:
            set_field(d, "flow.protocol", 17, src, self.stats)
        clen = v.get("http.content_length")
        if clen is not None:
            fid = "proto.http.request_body_len" if "http.request.method" in v else "proto.http.response_body_len"
            try:
                set_field(d, fid, int(_first(clen)), src, self.stats)
            except (TypeError, ValueError):
                self.stats.refuse(fid, "bad http.content_length")
        answers = [*_list(v.get("dns.a")), *_list(v.get("dns.aaaa")), *_list(v.get("dns.cname"))]
        if answers:
            set_field(d, "proto.dns.answers", tuple(str(a) for a in answers), src, self.stats)
        DV.dns_answer_stats(d, self.stats, src)
        DV.dns_name_stats(d, self.stats, src)
        DV.http_uri_length(d, self.stats, src)
        types = v.get("tls.handshake.type")
        if d.value("proto.tls.cipher") is not None and "2" not in [str(x) for x in _list(types)]:
            d.fields.pop("proto.tls.cipher", None)              # a ClientHello lists ciphers; none is chosen
        banner = v.get("ssh.protocol")
        if banner is not None:
            server = d.value("flow.src_port") == 22
            set_field(d, "proto.ssh.server" if server else "proto.ssh.client", str(_first(banner)), src, self.stats)
        DV.flow_entities(d, self.resolver)


def _first(v: Any) -> Any:
    return v[0] if isinstance(v, list) and v else v


def _list(v: Any) -> list[Any]:
    if v is None:
        return []
    return list(v) if isinstance(v, list) else [v]


class TsharkJsonSource(_WiresharkBase):
    """tshark -T json output -> state updates (module docstring)."""

    name = "tshark"
    source_type = "tshark-json"

    def raw_records(self) -> Iterator[RawRecord | tuple[str, RawRecord]]:
        stream, location, close = open_source(self.source)
        try:
            yield from iter_json_array(stream, location, max_bytes=self.config.common.max_record_bytes)
        finally:
            if close:
                stream.close()

    def decode(self, raw: RawRecord) -> Iterable[UpdateDraft]:
        try:
            obj = json.loads(raw.data, object_pairs_hook=_pairs)
        except (ValueError, UnicodeDecodeError) as exc:
            raise MalformedRecord("json-decode", str(exc)) from None
        layers = obj.get("_source", {}).get("layers") if isinstance(obj, dict) else None
        if not isinstance(layers, dict):
            raise MalformedRecord("no-layers", "packet without _source.layers")
        return [self.map_packet(flatten_layers(layers), raw)]


class TsharkEkSource(_WiresharkBase):
    """tshark -T ek output -> state updates (module docstring)."""

    name = "tshark"
    source_type = "tshark-ek"

    def raw_records(self) -> Iterator[RawRecord | tuple[str, RawRecord]]:
        stream, location, close = open_source(self.source)
        try:
            for item in iter_lines(stream, location, max_bytes=self.config.common.max_record_bytes):
                if isinstance(item, tuple):
                    yield item
                elif item.data.strip() and not item.data.lstrip().startswith(b'{"index"'):
                    yield item
        finally:
            if close:
                stream.close()

    def decode(self, raw: RawRecord) -> Iterable[UpdateDraft]:
        try:
            obj = json.loads(raw.data)
        except (ValueError, UnicodeDecodeError) as exc:
            raise MalformedRecord("json-decode", str(exc)) from None
        if not isinstance(obj, dict) or not isinstance(obj.get("layers"), dict):
            raise MalformedRecord("no-layers", "EK packet without layers")
        fields = from_ek(obj["layers"])
        d = self.map_packet(fields, raw)
        if d.time is None and obj.get("timestamp") is not None:
            try:
                d.time = epoch_integer(int(str(obj["timestamp"])), "ms")
            except ValueError:
                self.stats.refuse("@time", "bad EK timestamp")
        return [d]


class PySharkSource(_WiresharkBase):
    """A capture dissected by tshark through PyShark -> state updates (module docstring).

    The capture is read by tshark in a subprocess; packets are not kept (bounded memory). The source
    must be a file path. Each packet's raw record is its PDML rendering of the mapped fields (there is
    no byte offset: PyShark does not expose one), hashed for provenance.
    """

    name = "pyshark"
    source_type = "pyshark"

    def __init__(self, source: str | Path, *, config: WiresharkConfig | None = None, **kw: Any) -> None:
        super().__init__(source, config=config, **kw)
        if not isinstance(source, str | Path):
            raise MalformedRecord("pyshark-needs-path", "PyShark reads capture files by path")

    def raw_records(self) -> Iterator[RawRecord | tuple[str, RawRecord]]:
        try:
            import pyshark
        except ImportError as exc:                              # pragma: no cover - depends on the environment
            raise ImportError("PyShark is not installed; install the 'pcap' extra (pip install nagahana[pcap])") from exc
        kwargs: dict[str, Any] = {"keep_packets": False}
        if self.config.tshark_path:
            kwargs["tshark_path"] = self.config.tshark_path
        if self.config.display_filter:
            kwargs["display_filter"] = self.config.display_filter
        cap = pyshark.FileCapture(str(self.source), **kwargs)
        try:
            for i, pkt in enumerate(cap):
                fields: dict[str, Any] = {"frame.time_epoch": str(pkt.sniff_timestamp)}
                for layer in pkt.layers:
                    for name, val in layer._all_fields.items():
                        if not name or "." not in name:
                            continue
                        occurrences = getattr(val, "all_fields", None)
                        if occurrences is not None and len(occurrences) > 1:
                            fields[name] = [f.get_default_value() for f in occurrences]
                        else:
                            fields[name] = val.get_default_value() if hasattr(val, "get_default_value") else str(val)
                yield RawRecord(json.dumps(fields, sort_keys=True, default=str).encode("utf-8"), str(self.source), i,
                                meta={"fields": fields})
        finally:
            cap.close()

    def decode(self, raw: RawRecord) -> Iterable[UpdateDraft]:
        fields = (raw.meta or {}).get("fields")
        if not isinstance(fields, dict):
            raise MalformedRecord("no-fields", "PyShark packet without fields")
        return [self.map_packet(fields, raw)]


def detect(head: bytes) -> str:
    """"json" or "ek" from the first bytes of a tshark output file."""
    t = head.lstrip()
    if t.startswith(b"["):
        return "json"
    if t.startswith(b"{"):
        return "ek"
    raise MalformedRecord("unknown-tshark-format", head[:32].hex())


def open_tshark(source: str | Path, *, config: WiresharkConfig | None = None, **kw: Any) -> _WiresharkBase:
    """The reader for a tshark output file (JSON or EK, detected)."""
    with open(source, "rb") as fh:
        fmt = detect(fh.read(64))
    return (TsharkJsonSource if fmt == "json" else TsharkEkSource)(source, config=config, **kw)



__all__ = ["ADAPTER_VERSION", "EK_KEYS", "PySharkSource", "TsharkEkSource", "TsharkJsonSource", "detect",
           "flatten_layers", "from_ek", "open_tshark"]
