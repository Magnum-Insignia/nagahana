"""Suricata adapter: EVE JSON events -> state updates, through datamodel/maps/suricata.py.

One EVE event per line (Suricata user guide, "Eve JSON Output"). Nested objects are flattened into
dotted paths ("flow.pkts_toserver"); lists stay whole, and so do the objects a table lists as maps
("alert.metadata", "dns.answers"). An alert's app-layer objects (http, tls, dns, smb ...) are mapped
with the rows of their own event type as well. Numbers keep their exact digits.

Event time (AS-300): flow and netflow events take the flow's end, every other event the time of the
packet that triggered it ("timestamp").
"""

from __future__ import annotations

import base64
import binascii
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import IO, Any

from nagahana.datamodel.native import record_map
from nagahana.datamodel.spec import Dtype
from nagahana.datamodel.status import ObservationStatus
from nagahana.ingest import derive as DV
from nagahana.ingest.config import SuricataConfig
from nagahana.ingest.convert import ConvContext, Refusal, code_lookup
from nagahana.ingest.core import (
    BoundedLRU,
    MalformedRecord,
    RawRecord,
    StreamAdapter,
    UpdateDraft,
    iter_lines,
    json_loads_exact,
    json_safe,
    open_source,
)
from nagahana.ingest.mapping import mapper, set_field
from nagahana.ingest.packets import apply_packet, decode_packet
from nagahana.ingest.timeparse import Instant

ADAPTER_VERSION = "1.0.0"
EVENT_TYPES: tuple[str, ...] = (
    "flow", "netflow", "alert", "anomaly", "dns", "http", "tls", "fileinfo", "smb", "ssh", "dhcp", "krb5", "mqtt",
    "modbus", "dnp3", "stats",
)
#: App-layer objects an alert (or another event) may carry, mapped with their own table's rows.
_APP_OBJECTS: tuple[str, ...] = ("http", "tls", "dns", "smb", "ssh", "dhcp", "krb5", "mqtt", "modbus", "dnp3", "fileinfo")
#: Ethernet header bytes per packet, used to estimate IP-layer bytes from Suricata's frame bytes (AS-693).
ETHERNET_HEADER = 14
VLAN_TAG = 4
BYTES_ESTIMATE_RELIABILITY = 0.9
_MQTT_TYPES = {"connect": 1, "connack": 2, "publish": 3, "puback": 4, "pubrec": 5, "pubrel": 6, "pubcomp": 7,
               "subscribe": 8, "suback": 9, "unsubscribe": 10, "unsuback": 11, "pingreq": 12, "pingresp": 13,
               "disconnect": 14, "auth": 15}
_DNP3_IIN = {"broadcast": 0x0100, "class_1_events": 0x0200, "class_2_events": 0x0400, "class_3_events": 0x0800,
             "need_time": 0x1000, "local_control": 0x2000, "device_trouble": 0x4000, "device_restart": 0x8000,
             "no_func_code_support": 0x0001, "object_unknown": 0x0002, "parameter_error": 0x0004,
             "event_buffer_overflow": 0x0008, "already_executing": 0x0010, "config_corrupt": 0x0020,
             "reserved_2": 0x0040, "reserved_1": 0x0080}


def _map_paths() -> dict[str, frozenset[str]]:
    """Paths each table keeps whole (rows of dtype MAP)."""
    out = {}
    for ev in EVENT_TYPES:
        rm = record_map(f"suricata.{ev}")
        out[ev] = frozenset(r.source for r in rm.rows if r.dtype is Dtype.MAP)
    return out


_WHOLE: dict[str, frozenset[str]] | None = None


def flatten(obj: dict[str, Any], keep_whole: frozenset[str], prefix: str = "") -> dict[str, Any]:
    """Dotted paths of nested objects; lists and the paths in `keep_whole` stay whole."""
    out: dict[str, Any] = {}
    for k, v in obj.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict) and key not in keep_whole:
            out.update(flatten(v, keep_whole, key + "."))
        else:
            out[key] = v
    return out


class SuricataSource(StreamAdapter):
    """Suricata EVE JSON (one event per line) -> state updates. See the module docstring."""

    name = "suricata"
    source_type = "suricata-eve"
    version = ADAPTER_VERSION

    def __init__(self, source: str | Path | bytes | IO[bytes], *, config: SuricataConfig | None = None, **kw: Any) -> None:
        self.config = config or SuricataConfig()
        super().__init__(source, self.config.common, **kw)
        clock = self.config.common.clock
        self._ctx = ConvContext(year=clock.assumed_year,
                                utc_offset_s=None if clock.utc_offset_hours is None else clock.utc_offset_hours * 3600.0)
        self._stats_prev: BoundedLRU[str, tuple[float, int | None, int | None]] = BoundedLRU(
            self.config.stats_sources, "suricata_stats_sources", self.stats)

    def raw_records(self) -> Iterator[RawRecord | tuple[str, RawRecord]]:
        stream, location, close = open_source(self.source)
        try:
            for item in iter_lines(stream, location, max_bytes=self.config.common.max_record_bytes):
                if isinstance(item, tuple) or item.data.strip():
                    yield item
        finally:
            if close:
                stream.close()

    def decode(self, raw: RawRecord) -> Iterable[UpdateDraft]:
        try:
            obj = json_loads_exact(raw.data)
        except (ValueError, UnicodeDecodeError) as exc:
            raise MalformedRecord("json-decode", str(exc)) from None
        if not isinstance(obj, dict):
            raise MalformedRecord("json-not-object", type(obj).__name__)
        return [self.map_event(obj, raw)]

    def map_event(self, obj: dict[str, Any], raw: RawRecord) -> UpdateDraft:
        """Map one EVE event object."""
        global _WHOLE
        if _WHOLE is None:
            _WHOLE = _map_paths()
        ev = obj.get("event_type")
        if not isinstance(ev, str):
            raise MalformedRecord("no-event-type", "EVE event without event_type")
        known = ev in EVENT_TYPES
        record_type = f"suricata.{ev}"
        whole = _WHOLE.get(ev, frozenset()) if known else frozenset()
        apps = [a for a in _APP_OBJECTS if a != ev and isinstance(obj.get(a), dict)]
        for a in apps:
            whole = whole | _WHOLE.get(a, frozenset())
        values = flatten(obj, whole)
        d = UpdateDraft(record_type, raw)
        src = record_type
        ctx = self._ctx
        ctx.values = values
        ctx.ipv6 = ":" in str(values.get("src_ip", ""))
        if not known:
            self.stats.counters[f"unknown_event_type:{ev}"] += 1
            self._retain(values, d, set(_ENVELOPE), record_type)
            mapper("suricata.flow").apply({k: v for k, v in values.items() if k in _ENVELOPE}, d, ctx, self.stats,
                                          source=src, keep_unmapped=False)
            DV.flow_entities(d, self.resolver)
            return d
        primary = mapper(record_type)
        primary.apply(values, d, ctx, self.stats, source=src, keep_unmapped=False)
        used = set(primary.sources)
        for a in apps:
            sub = mapper(f"suricata.{a}")
            rows = {k: v for k, v in values.items() if k.startswith(a + ".")}
            sub.apply(rows, d, ctx, self.stats, source=f"suricata.{a}", keep_unmapped=False)
            used |= sub.sources
            getattr(self, f"_after_{a}", _noop)(d, values, f"suricata.{a}")
        if self.config.common.keep_unmapped:
            self._retain(values, d, used, record_type)
        getattr(self, f"_after_{ev}", _noop)(d, values, src)
        if ev != "stats":
            DV.flow_entities(d, self.resolver)
        return d

    def _retain(self, values: dict[str, Any], d: UpdateDraft, used: set[str], prefix: str) -> None:
        from nagahana.datamodel.fields import CATALOGUE
        from nagahana.datamodel.records import FieldValue

        for key, v in values.items():
            if key in used:
                continue
            akey = f"{prefix}.{key}"
            if akey in CATALOGUE:
                akey = f"x.{akey}"
            self.stats.retain(akey)
            d.attributes[akey] = (FieldValue(akey, None, ObservationStatus.NOT_SUPPLIED, prefix) if v is None
                                  else FieldValue(akey, json_safe(v), ObservationStatus.OBSERVED, prefix))

    def _estimate_ip_bytes(self, d: UpdateDraft, values: dict[str, Any], src: str) -> None:
        """flow.bytes_* from frame bytes minus Ethernet and VLAN headers per packet (AS-693)."""
        vlans = values.get("vlan")
        per_packet = ETHERNET_HEADER + VLAN_TAG * (len(vlans) if isinstance(vlans, list) else 0)
        for frame, pkts, target in (("flow.frame_bytes_fwd", "flow.packets_fwd", "flow.bytes_fwd"),
                                    ("flow.frame_bytes_bwd", "flow.packets_bwd", "flow.bytes_bwd")):
            fb, pk = d.value(frame), d.value(pkts)
            if fb is None or pk is None:
                continue
            est = int(fb) - per_packet * int(pk)
            if est < 0:
                self.stats.refuse(target, "frame bytes smaller than the link headers of its packets")
                continue
            set_field(d, target, est, src, self.stats, status=ObservationStatus.LOW_RELIABILITY,
                      reliability=BYTES_ESTIMATE_RELIABILITY)

    def _after_flow(self, d: UpdateDraft, values: dict[str, Any], src: str) -> None:
        self._estimate_ip_bytes(d, values, src)
        DV.flow_totals(d, self.stats, src)
        DV.times_and_duration(d, self.stats, src)
        end = d.value("flow.end_time")
        if end is not None:
            d.time = _instant_of(values.get("flow.end"), float(end))
        state, reason = values.get("flow.state"), values.get("flow.reason")
        if values.get("tcp.rst") is True:
            set_field(d, "flow.end_reason", 2, src, self.stats)
        elif values.get("tcp.fin") is True and state == "closed":
            set_field(d, "flow.end_reason", 1, src, self.stats)
        elif reason in ("timeout", "forced", "shutdown"):
            set_field(d, "flow.end_reason", {"timeout": 3, "forced": 6, "shutdown": 4}[str(reason)], src, self.stats)

    def _after_netflow(self, d: UpdateDraft, values: dict[str, Any], src: str) -> None:
        vlans = values.get("vlan")
        per_packet = ETHERNET_HEADER + VLAN_TAG * (len(vlans) if isinstance(vlans, list) else 0)
        fb, pk = d.value("flow.frame_bytes_fwd"), d.value("flow.packets_fwd")
        if fb is not None and pk is not None and int(fb) >= per_packet * int(pk):
            set_field(d, "flow.bytes_fwd", int(fb) - per_packet * int(pk), src, self.stats,
                      status=ObservationStatus.LOW_RELIABILITY, reliability=BYTES_ESTIMATE_RELIABILITY)
        DV.times_and_duration(d, self.stats, src)
        end = d.value("flow.end_time")
        if end is not None:
            d.time = _instant_of(values.get("netflow.end"), float(end))

    def _after_alert(self, d: UpdateDraft, values: dict[str, Any], src: str) -> None:
        DV.alert_identity(d, self.stats, src)
        packet = values.get("packet")
        if isinstance(packet, str):
            try:
                data = base64.b64decode(packet, validate=True)
            except (binascii.Error, ValueError):
                self.stats.refuse("suricata.alert.packet", "not base64")
                return
            link = values.get("packet_info.linktype", 1)
            facts = decode_packet(data, int(link) if isinstance(link, int) else 1)
            apply_packet(d, facts, self.stats, src)

    def _after_anomaly(self, d: UpdateDraft, values: dict[str, Any], src: str) -> None:
        name = values.get("anomaly.event")
        if isinstance(name, str):
            set_field(d, "alert.signature", DV.name_signature("suricata-anomaly", name), src, self.stats)

    def _after_dns(self, d: UpdateDraft, values: dict[str, Any], src: str) -> None:
        queries = values.get("dns.queries")
        if isinstance(queries, list) and queries and isinstance(queries[0], dict):
            q = queries[0]
            if "rrname" in q:
                set_field(d, "proto.dns.query", str(q["rrname"]), src, self.stats)
            if "rrtype" in q:
                try:
                    set_field(d, "proto.dns.qtype", code_lookup("dns_qtype", q["rrtype"]), src, self.stats)
                except Refusal as exc:
                    self.stats.refuse("proto.dns.qtype", str(exc))
        dtype = values.get("dns.type")
        response = dtype in ("answer", "response") or values.get("dns.qr") is True
        answers = values.get("dns.answers")
        if isinstance(answers, list):
            rdata: list[str] = []
            ttls: list[float] = []
            for a in answers:
                if not isinstance(a, dict):
                    continue
                rdata.append(str(a.get("rdata", "")))
                if "ttl" in a:
                    try:
                        ttls.append(float(a["ttl"]))
                    except (TypeError, ValueError):
                        self.stats.refuse("proto.dns.ttls", "non-numeric answer TTL")
            set_field(d, "proto.dns.answers", tuple(rdata), src, self.stats)
            if ttls:
                set_field(d, "proto.dns.ttls", tuple(ttls), src, self.stats)
        elif response:
            set_field(d, "proto.dns.answer_count", 0, src, self.stats)
        DV.dns_answer_stats(d, self.stats, src)
        DV.dns_name_stats(d, self.stats, src)

    def _after_http(self, d: UpdateDraft, values: dict[str, Any], src: str) -> None:
        DV.http_uri_length(d, self.stats, src)

    def _after_tls(self, d: UpdateDraft, values: dict[str, Any], src: str) -> None:
        DV.cert_facts(d, self.stats, src)
        alpns = values.get("tls.server_alpns")
        if isinstance(alpns, list) and alpns:
            set_field(d, "proto.tls.alpn", str(alpns[0]), src, self.stats)

    def _after_smb(self, d: UpdateDraft, values: dict[str, Any], src: str) -> None:
        ntlm = values.get("smb.ntlmssp")
        if isinstance(ntlm, dict):
            for key, fid in (("user", "auth.user"), ("domain", "auth.domain"), ("host", "auth.workstation")):
                if ntlm.get(key):
                    set_field(d, fid, str(ntlm[key]), src, self.stats)
            d.add_entity(self.resolver.account(d.value("auth.user"), d.value("auth.domain")), "account")
        dce = values.get("smb.dcerpc")
        if isinstance(dce, dict):
            ifs = dce.get("interfaces")
            if isinstance(ifs, list) and ifs and isinstance(ifs[0], dict):
                name = ifs[0].get("name") or ifs[0].get("uuid")
                if name:
                    set_field(d, "proto.dcerpc.endpoint", str(name), src, self.stats)

    def _after_krb5(self, d: UpdateDraft, values: dict[str, Any], src: str) -> None:
        set_field(d, "auth.protocol", 2, src, self.stats)
        mt = d.value("proto.kerberos.msg_type")
        failed = values.get("krb5.failed_request")
        req = None
        if mt in (10, 11) or failed in ("KRB_AS_REQ", "AS-REQ"):
            req = 3
        elif mt in (12, 13) or failed in ("KRB_TGS_REQ", "TGS-REQ"):
            req = 4
        if req is not None:
            set_field(d, "auth.activity", req, src, self.stats)
        err = d.value("proto.kerberos.error_code")
        if mt == 30 or (err is not None and int(err) != 0):
            set_field(d, "auth.result", 2, src, self.stats)
        elif mt in (11, 13):
            set_field(d, "auth.result", 1, src, self.stats)
        realm = d.value("proto.kerberos.realm")
        d.add_entity(self.resolver.account(d.value("proto.kerberos.client"), realm), "account")
        d.add_entity(self.resolver.account(d.value("proto.kerberos.service")), "target_account")

    def _after_mqtt(self, d: UpdateDraft, values: dict[str, Any], src: str) -> None:
        for key, code in _MQTT_TYPES.items():
            obj = values.get(f"mqtt.{key}")
            if not isinstance(obj, dict):
                continue
            set_field(d, "proto.mqtt.message_type", code, src, self.stats)
            if isinstance(obj.get("qos"), int):
                set_field(d, "proto.mqtt.qos", obj["qos"], src, self.stats)
            topic = obj.get("topic")
            if topic is None and isinstance(obj.get("topics"), list) and obj["topics"]:
                first = obj["topics"][0]
                topic = first.get("topic") if isinstance(first, dict) else first
            if topic is not None:
                set_field(d, "proto.mqtt.topic", str(topic), src, self.stats)
            if obj.get("client_id") is not None:
                set_field(d, "proto.mqtt.client_id", str(obj["client_id"]), src, self.stats)
            rc = obj.get("return_code", obj.get("reason_code"))
            if isinstance(rc, int):
                set_field(d, "proto.mqtt.return_code", rc, src, self.stats)
            break

    def _after_modbus(self, d: UpdateDraft, values: dict[str, Any], src: str) -> None:
        req, rsp = values.get("modbus.request"), values.get("modbus.response")
        for msg in (req, rsp):
            if not isinstance(msg, dict):
                continue
            for key, fid in (("transaction_id", "ot.modbus.transaction_id"), ("unit_id", "ot.modbus.unit_id"),
                             ("function_raw", "ot.modbus.function_code")):
                if isinstance(msg.get(key), int):
                    set_field(d, fid, msg[key] & 0x7F if key == "function_raw" else msg[key], src, self.stats)
            for part in ("read", "write"):
                sub = msg.get(part)
                if isinstance(sub, dict):
                    if isinstance(sub.get("address"), int):
                        set_field(d, "ot.modbus.register_start", sub["address"], src, self.stats)
                    if isinstance(sub.get("quantity"), int):
                        set_field(d, "ot.modbus.register_count", sub["quantity"], src, self.stats)
        if isinstance(rsp, dict):
            exc = rsp.get("exception")
            code = exc.get("raw", exc.get("code")) if isinstance(exc, dict) else exc
            if code is not None:
                try:
                    set_field(d, "ot.modbus.exception_code", code_lookup("modbus_exception", code), src, self.stats)
                except Refusal as e:
                    self.stats.refuse("ot.modbus.exception_code", str(e))

    def _after_dnp3(self, d: UpdateDraft, values: dict[str, Any], src: str) -> None:
        app = values.get("dnp3.application")
        kind = values.get("dnp3.type")
        if isinstance(app, dict) and isinstance(app.get("function_code"), int):
            fid = "ot.dnp3.function_code" if kind == "request" else "ot.dnp3.function_code_reply"
            set_field(d, fid, app["function_code"], src, self.stats)
            objs = app.get("objects")
            if isinstance(objs, list) and objs and isinstance(objs[0], dict) and isinstance(objs[0].get("group"), int):
                set_field(d, "ot.dnp3.object_group", objs[0]["group"], src, self.stats)
        iin = values.get("dnp3.iin")
        if isinstance(iin, dict) and isinstance(iin.get("indicators"), list):
            bits = 0
            unknown = [x for x in iin["indicators"] if x not in _DNP3_IIN]
            for x in iin["indicators"]:
                bits |= _DNP3_IIN.get(x, 0)
            if unknown:
                self.stats.refuse("ot.dnp3.iin", f"unknown indicator names {unknown}")
            else:
                set_field(d, "ot.dnp3.iin", bits, src, self.stats)

    def _after_stats(self, d: UpdateDraft, values: dict[str, Any], src: str) -> None:
        host = values.get("host")
        sensor = self.resolver.host_name(str(host)) if isinstance(host, str) else self.resolver.sensor()
        d.add_entity(sensor or self.resolver.sensor(), "subject")
        key = str(host) if host else self.resolver.sensor_key
        t = d.time.seconds if d.time is not None else None
        pk, dr = values.get("stats.capture.kernel_packets"), values.get("stats.capture.kernel_drops")
        pk = int(pk) if isinstance(pk, int) else None
        dr = int(dr) if isinstance(dr, int) else None
        prev = self._stats_prev.get(key)
        if t is not None:
            self._stats_prev[key] = (t, pk, dr)
        if prev is None or t is None:
            return
        t0, pk0, dr0 = prev
        if t <= t0:
            self.stats.refuse("dev.interval", "stats records out of order")
            return
        set_field(d, "dev.interval", t - t0, src, self.stats)
        for now, before, fid in ((pk, pk0, "dev.capture_received"), (dr, dr0, "dev.capture_dropped")):
            if now is None or before is None:
                continue
            if now < before:                                     # the engine restarted: counters reset
                self.stats.refuse(fid, "counter reset")
                continue
            set_field(d, fid, now - before, src, self.stats)


_ENVELOPE = frozenset(r.source for r in record_map("suricata.flow").rows if not r.source.startswith(("flow.", "tcp.")))


def _noop(d: UpdateDraft, values: dict[str, Any], src: str) -> None:
    return None


def _instant_of(raw: Any, seconds: float) -> Instant:
    """The Instant of an EVE time text, for an event time moved to it (exact digits kept)."""
    from nagahana.ingest.convert import ConvContext as _C
    from nagahana.ingest.convert import get

    try:
        t = get("time")(raw, _C())
        if isinstance(t, Instant):
            return t
    except Refusal:
        pass
    return Instant(seconds, None, 1e-6)


__all__ = ["ADAPTER_VERSION", "EVENT_TYPES", "SuricataSource", "flatten"]
