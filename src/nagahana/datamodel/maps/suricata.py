"""Suricata EVE JSON mapping tables: flow, netflow, alert, anomaly, dns (versions 2 and 3), http, tls,
fileinfo, smb, ssh, dhcp, krb5, mqtt, modbus, dnp3, stats.

Paths are the EVE JSON keys joined with dots (the adapter flattens nested objects; lists of objects
stay whole). Field lists follow the Suricata user guide, "Eve JSON Format" and "Eve JSON Output"
(Suricata 6.0, 7.0 and 8.0; DNS logging version 2 and version 3). Every EVE event shares the envelope
rows of `_envelope`; their native fields keep one ID across event types ("suricata.<path>").
"""

from __future__ import annotations

from nagahana.datamodel.layers import Layer
from nagahana.datamodel.native import R, RecordMap, Row
from nagahana.datamodel.spec import Level

REF = "Suricata user guide, Eve JSON Format (Suricata 6.0 to 8.0)"


def _n(path: str) -> str:
    return f"suricata.{path}"


def _envelope() -> tuple[Row, ...]:
    return (
        R("timestamp", "t", "@time", "time", note="Time of the packet that triggered the event."),
        R("flow_id", "i", "flow.uid", "flow_id"),
        R("parent_id", "i", conv="count", native=_n("parent_id")),
        R("in_iface", "s", native=_n("in_iface")),
        R("event_type", "s", native=_n("event_type")),
        R("vlan", "I", "flow.vlan", "first_int", also="native", native=_n("vlan")),
        R("src_ip", "a", "flow.src_ip", "addr"),
        R("src_port", "i", "flow.src_port", "port"),
        R("dest_ip", "a", "flow.dst_ip", "addr"),
        R("dest_port", "i", "flow.dst_port", "port"),
        R("proto", "s", "flow.protocol", "proto_name"),
        R("ip_v", "i", "flow.ip_version", "count"),
        R("icmp_type", "i", "proto.icmp.type", "count"),
        R("icmp_code", "i", "proto.icmp.code", "count"),
        R("pkt_src", "s", native=_n("pkt_src")),
        R("community_id", "s", "flow.community_id", "string"),
        R("tx_id", "i", conv="count", native=_n("tx_id")),
        R("tx_guessed", "b", conv="bool", native=_n("tx_guessed")),
        R("app_proto", "s", "flow.app_proto", "app_proto", also="flow.app_proto_text"),
        R("app_proto_ts", "s", native=_n("app_proto_ts")),
        R("app_proto_tc", "s", native=_n("app_proto_tc")),
        R("app_proto_orig", "s", native=_n("app_proto_orig")),
        R("app_proto_expected", "s", native=_n("app_proto_expected")),
        R("direction", "s", native=_n("direction")),
        R("ether.src_mac", "m", "flow.src_mac", "mac"),
        R("ether.dest_mac", "m", "flow.dst_mac", "mac"),
        R("host", "s", "event.hostname", "string", note="Sensor name configured in Suricata."),
        R("pcap_cnt", "i", conv="count", native=_n("pcap_cnt")),
        R("pcap_filename", "s", native=_n("pcap_filename"), kind="id"),
        R("capture", "M", native=_n("capture")),
        R("metadata", "M", native=_n("metadata")),
        R("tunnel", "M", native=_n("tunnel")),
    )


def _flow_counters(prefix: str) -> tuple[Row, ...]:
    return (
        R(f"{prefix}.pkts_toserver", "i", "flow.packets_fwd", "count", unit="packets"),
        R(f"{prefix}.pkts_toclient", "i", "flow.packets_bwd", "count", unit="packets"),
        R(f"{prefix}.bytes_toserver", "i", "flow.frame_bytes_fwd", "count", unit="bytes",
          note="Bytes as captured, link-layer header included (AS-693)."),
        R(f"{prefix}.bytes_toclient", "i", "flow.frame_bytes_bwd", "count", unit="bytes"),
        R(f"{prefix}.start", "t", "flow.start_time", "time", unit="s"),
    )


def _tcp(prefix: str = "tcp") -> tuple[Row, ...]:
    return (
        R(f"{prefix}.tcp_flags", "s", "flow.tcp_flags", "hex_tcp_flags"),
        R(f"{prefix}.tcp_flags_ts", "s", "flow.tcp_flags_fwd", "hex"),
        R(f"{prefix}.tcp_flags_tc", "s", "flow.tcp_flags_bwd", "hex"),
        *(R(f"{prefix}.{f}", "b", conv="bool") for f in ("syn", "fin", "rst", "psh", "ack", "urg", "ecn", "cwr")),
        R(f"{prefix}.state", "s"),
        R(f"{prefix}.gap", "b", conv="bool"),
    )


FLOW = RecordMap(
    "suricata", "flow", "Suricata EVE flow", Level.FLOW, (
        *_envelope(),
        *_flow_counters("flow"),
        R("flow.end", "t", "flow.end_time", "time", unit="s"),
        R("flow.age", "i", conv="count", unit="s"),
        R("flow.state", "s"),
        R("flow.reason", "s"),
        R("flow.alerted", "b", conv="bool"),
        R("flow.bypass", "s"),
        R("flow.bypassed", "M"),
        R("flow.wrong_thread", "b", conv="bool"),
        R("flow.emergency", "b", conv="bool"),
        R("flow.exception_policy", "M"),
        R("flow.tx_cnt", "i", conv="count"),
        *_tcp(),
    ), REF, ocsf_class=4001,
    notes=(
        "Event time: flow.end (the record summarises the flow up to its end, AS-300). flow.duration = end - start.",
        "flow.bytes_fwd / _bwd are estimated from the link-layer byte counts (minus 14 bytes of Ethernet header "
        "and 4 bytes per VLAN tag per packet), LOW_RELIABILITY (AS-693).",
        "Derived: flow.packets_total, flow.unanswered (pkts_toclient = 0), flow.bidir_ratio, flow.end_reason from "
        "tcp.rst / tcp.fin with flow.state closed, else from flow.reason (timeout idle, forced forced_end, "
        "shutdown capture_end).",
    ),
)

NETFLOW = RecordMap(
    "suricata", "netflow", "Suricata EVE netflow", Level.FLOW, (
        *_envelope(),
        R("netflow.pkts", "i", "flow.packets_fwd", "count", unit="packets"),
        R("netflow.bytes", "i", "flow.frame_bytes_fwd", "count", unit="bytes"),
        R("netflow.start", "t", "flow.start_time", "time", unit="s"),
        R("netflow.end", "t", "flow.end_time", "time", unit="s"),
        R("netflow.age", "i", conv="count", unit="s"),
        R("netflow.min_ttl", "i", "pkt.ttl_min", "count", unit="hops"),
        R("netflow.max_ttl", "i", "pkt.ttl_max", "count", unit="hops"),
        R("netflow.tx_cnt", "i", conv="count"),
        *_tcp(),
    ), REF, ocsf_class=4001,
    notes=("One direction of a flow per record: the backward fields stay NOT_SUPPLIED (never zero).",),
)

ALERT = RecordMap(
    "suricata", "alert", "Suricata EVE alert", Level.ALERT, (
        *_envelope(),
        R("alert.action", "s", "event.action", "code:event_action", also="native"),
        R("alert.gid", "i", "alert.generator_id", "count"),
        R("alert.signature_id", "i", "alert.signature_id", "count"),
        R("alert.rev", "i", "alert.revision", "count"),
        R("alert.signature", "s", "alert.message", "string"),
        R("alert.category", "s", "alert.category", "code:alert_category", also="alert.category_text"),
        R("alert.severity", "i", "alert.priority", "count", note="Suricata priority: 1 is the most severe."),
        R("alert.metadata", "M"),
        R("alert.rule", "s", "alert.rule", "string"),
        R("alert.source", "M"),
        R("alert.target", "M"),
        R("alert.xff", "s"),
        R("alert.references", "S", conv="list"),
        *_flow_counters("flow"),
        R("payload", "s", note="Payload, base64."),
        R("payload_printable", "s"),
        R("packet", "s", note="Triggering packet, base64; decoded for packet-level fields."),
        R("packet_info.linktype", "i", conv="count"),
        R("stream", "i", conv="count"),
        R("verdict", "M"),
        R("files", "M"),
    ), REF, ocsf_class=2004,
    notes=(
        "Derived: alert.signature = gid * 2^32 + signature_id; alert.severity on the OCSF scale from the priority "
        "(1 High, 2 Medium, 3 Low, 4 and above Informational, AS-694).",
        "When packet is present it is decoded (dpkt) for pkt.ttl_mean/min/max, pkt.ip_len_min/max, flow.ip_tos "
        "and the TCP flags of that packet.",
        "App-layer objects of the alert (http, tls, dns, smb, ...) are mapped with the rows of their own event type.",
    ),
)

ANOMALY = RecordMap(
    "suricata", "anomaly", "Suricata EVE anomaly", Level.ALERT, (
        *_envelope(),
        R("anomaly.type", "s"),
        R("anomaly.event", "s", "alert.rule", "string"),
        R("anomaly.layer", "s"),
        R("anomaly.code", "i", conv="count"),
    ), REF, ocsf_class=2004,
    notes=("Derived: alert.signature from the anomaly event name in the Suricata-anomaly namespace (AS-687).",),
)

DNS = RecordMap(
    "suricata", "dns", "Suricata EVE dns (versions 2 and 3)", Level.PROTOCOL, (
        *_envelope(),
        R("dns.version", "i", conv="count"),
        R("dns.type", "s", note="query / answer (version 2), request / response (version 3)."),
        R("dns.id", "i", "proto.dns.trans_id", "count"),
        R("dns.flags", "s", "proto.dns.flags", "dns_header_flags", also="native"),
        R("dns.qr", "b", conv="bool"),
        R("dns.aa", "b", conv="bool"),
        R("dns.tc", "b", conv="bool"),
        R("dns.rd", "b", conv="bool"),
        R("dns.ra", "b", conv="bool"),
        R("dns.z", "b", conv="bool"),
        R("dns.ad", "b", conv="bool"),
        R("dns.cd", "b", conv="bool"),
        R("dns.opcode", "i", "proto.dns.opcode", "count"),
        R("dns.rrname", "s", "proto.dns.query", "string"),
        R("dns.rrtype", "s", "proto.dns.qtype", "code:dns_qtype"),
        R("dns.rcode", "s", "proto.dns.rcode", "code:dns_rcode"),
        R("dns.tx_id", "i", conv="count"),
        R("dns.queries", "M"),
        R("dns.answers", "M"),
        R("dns.grouped", "M"),
        R("dns.authorities", "M"),
        R("dns.additionals", "M"),
    ), REF, ocsf_class=4003,
    notes=(
        "Version 3 (and version 2 answers with queries): proto.dns.query and proto.dns.qtype from the first "
        "element of dns.queries when dns.rrname is absent.",
        "Derived from dns.answers: proto.dns.answers (rdata), proto.dns.ttls, proto.dns.answer_count (0 for a "
        "response without answers), proto.dns.ttl_min; proto.dns.query_length and query_entropy from the name.",
    ),
)

HTTP = RecordMap(
    "suricata", "http", "Suricata EVE http", Level.PROTOCOL, (
        *_envelope(),
        R("http.hostname", "s", "proto.http.host", "string"),
        R("http.http_port", "i", conv="port"),
        R("http.url", "s", "proto.http.uri", "string"),
        R("http.http_user_agent", "s", "proto.http.user_agent", "string"),
        R("http.http_content_type", "s", "proto.http.content_type", "string"),
        R("http.http_refer", "s", "proto.http.referrer", "string"),
        R("http.http_method", "s", "proto.http.method", "code:http_method", also="proto.http.method_text"),
        R("http.protocol", "s", "proto.http.version", "string"),
        R("http.status", "i", "proto.http.status_code", "count"),
        R("http.length", "i", "proto.http.response_body_len", "count", unit="bytes"),
        R("http.redirect", "s", kind="id"),
        R("http.xff", "s"),
        R("http.request_headers", "M"),
        R("http.response_headers", "M"),
        R("http.http_request_body", "s"),
        R("http.http_request_body_printable", "s"),
        R("http.http_response_body", "s"),
        R("http.http_response_body_printable", "s"),
    ), REF, ocsf_class=4002,
    notes=("Derived: proto.http.uri_length from http.url.",),
)

TLS = RecordMap(
    "suricata", "tls", "Suricata EVE tls", Level.PROTOCOL, (
        *_envelope(),
        R("tls.subject", "s", "proto.tls.cert_subject", "string"),
        R("tls.issuerdn", "s", "proto.tls.cert_issuer", "string"),
        R("tls.serial", "s", "proto.tls.cert_serial", "string"),
        R("tls.fingerprint", "s", "proto.tls.cert_fingerprint", "string"),
        R("tls.sni", "s", "proto.tls.sni", "string"),
        R("tls.version", "s", "proto.tls.version", "code:tls_version", also="proto.tls.version_text"),
        R("tls.notbefore", "t", "proto.tls.cert_not_before", "time", unit="s"),
        R("tls.notafter", "t", "proto.tls.cert_not_after", "time", unit="s"),
        R("tls.session_resumed", "b", "proto.tls.resumed", "bool"),
        R("tls.ja3.hash", "s", "proto.tls.ja3", "string"),
        R("tls.ja3.string", "s", kind="fp"),
        R("tls.ja3s.hash", "s", "proto.tls.ja3s", "string"),
        R("tls.ja3s.string", "s", kind="fp"),
        R("tls.ja4", "s", "proto.tls.ja4", "string"),
        R("tls.certificate", "s", note="Certificate, base64 DER."),
        R("tls.chain", "S", conv="list"),
        R("tls.client", "M"),
        R("tls.client_alpns", "S", conv="list"),
        R("tls.server_alpns", "S", conv="list"),
    ), REF, ocsf_class=4001,
    notes=(
        "Derived: proto.tls.cert_validity = notafter - notbefore; proto.tls.cert_self_signed = 1 if subject "
        "equals issuerdn; proto.tls.alpn = first server ALPN.",
    ),
)

FILEINFO = RecordMap(
    "suricata", "fileinfo", "Suricata EVE fileinfo", Level.PROTOCOL, (
        *_envelope(),
        R("fileinfo.filename", "s", "file.name", "string"),
        R("fileinfo.sid", "I", conv="list_int"),
        R("fileinfo.gaps", "b", conv="bool"),
        R("fileinfo.state", "s"),
        R("fileinfo.md5", "s", "file.md5", "string"),
        R("fileinfo.sha1", "s", "file.sha1", "string"),
        R("fileinfo.sha256", "s", "file.sha256", "string"),
        R("fileinfo.stored", "b", conv="bool"),
        R("fileinfo.file_id", "i", "file.uid", "int_text"),
        R("fileinfo.size", "i", "file.size", "count", unit="bytes"),
        R("fileinfo.tx_id", "i", conv="count"),
        R("fileinfo.start", "i", conv="count", unit="bytes"),
        R("fileinfo.end", "i", conv="count", unit="bytes"),
        R("fileinfo.magic", "s", kind="fp"),
        R("fileinfo.mimetype", "s", "file.mime_type", "string"),
        R("fileinfo.storing", "b", conv="bool"),
    ), REF, ocsf_class=0,
)

SMB = RecordMap(
    "suricata", "smb", "Suricata EVE smb", Level.PROTOCOL, (
        *_envelope(),
        R("smb.id", "i", conv="count"),
        R("smb.dialect", "s", "proto.smb.dialect", "string"),
        R("smb.command", "s", "proto.smb.command", "code:smb_command", also="native"),
        R("smb.status", "s"),
        R("smb.status_code", "s", "proto.smb.status", "hex"),
        R("smb.session_id", "i", conv="count"),
        R("smb.tree_id", "i", conv="count"),
        R("smb.filename", "s", "file.name", "string"),
        R("smb.disposition", "s"),
        R("smb.access", "s"),
        R("smb.created", "t", conv="time", unit="s"),
        R("smb.accessed", "t", conv="time", unit="s"),
        R("smb.modified", "t", conv="time", unit="s"),
        R("smb.changed", "t", conv="time", unit="s"),
        R("smb.size", "i", "file.size", "count", unit="bytes"),
        R("smb.fuid", "s", kind="id"),
        R("smb.share", "s", "proto.smb.path", "string"),
        R("smb.share_type", "s", "proto.smb.share_type", "string"),
        R("smb.request", "M"),
        R("smb.response", "M"),
        R("smb.ntlmssp", "M"),
        R("smb.kerberos", "M"),
        R("smb.dcerpc", "M"),
        R("smb.client_dialects", "S", conv="list"),
        R("smb.server_guid", "s", kind="id"),
        R("smb.client_guid", "s", kind="id"),
        R("smb.max_read_size", "i", conv="count"),
        R("smb.max_write_size", "i", conv="count"),
        R("smb.named_pipe", "s", kind="id"),
        R("smb.function", "s"),
    ), REF, ocsf_class=4006,
    notes=("Derived: auth.user / auth.domain / auth.workstation from smb.ntlmssp when present; proto.dcerpc.endpoint "
           "from the first interface of smb.dcerpc.",),
)

SSH = RecordMap(
    "suricata", "ssh", "Suricata EVE ssh", Level.PROTOCOL, (
        *_envelope(),
        R("ssh.client.proto_version", "s", "proto.ssh.version", "ssh_major"),
        R("ssh.client.software_version", "s", "proto.ssh.client", "string"),
        R("ssh.client.hassh.hash", "s", "proto.ssh.hassh", "string"),
        R("ssh.client.hassh.string", "s", kind="fp"),
        R("ssh.server.proto_version", "s"),
        R("ssh.server.software_version", "s", "proto.ssh.server", "string"),
        R("ssh.server.hassh.hash", "s", "proto.ssh.hassh_server", "string"),
        R("ssh.server.hassh.string", "s", kind="fp"),
    ), REF, ocsf_class=4007,
)

DHCP = RecordMap(
    "suricata", "dhcp", "Suricata EVE dhcp", Level.PROTOCOL, (
        *_envelope(),
        R("dhcp.type", "s", note="request or reply."),
        R("dhcp.id", "i", conv="count"),
        R("dhcp.client_mac", "m", "proto.dhcp.client_mac", "mac"),
        R("dhcp.assigned_ip", "a", "proto.dhcp.assigned_addr", "addr_nonzero"),
        R("dhcp.client_ip", "a", conv="addr_nonzero", kind="id"),
        R("dhcp.relay_ip", "a", conv="addr_nonzero", kind="id"),
        R("dhcp.next_server_ip", "a", conv="addr_nonzero", kind="id"),
        R("dhcp.dhcp_type", "s", "proto.dhcp.message_type", "code:dhcp_message"),
        R("dhcp.client_id", "s", kind="id"),
        R("dhcp.hostname", "s", "proto.dhcp.hostname", "string"),
        R("dhcp.params", "S", conv="list"),
        R("dhcp.requested_ip", "a", "proto.dhcp.requested_addr", "addr_nonzero"),
        R("dhcp.lease_time", "i", "proto.dhcp.lease_time", "interval", unit="s"),
        R("dhcp.renewal_time", "i", conv="count", unit="s"),
        R("dhcp.rebinding_time", "i", conv="count", unit="s"),
        R("dhcp.subnet_mask", "a", conv="addr", kind="id"),
        R("dhcp.routers", "A", conv="list_addr"),
        R("dhcp.dns_servers", "A", conv="list_addr"),
        R("dhcp.vendor_class_identifier", "s", kind="fp"),
    ), REF, ocsf_class=4004,
)

KRB5 = RecordMap(
    "suricata", "krb5", "Suricata EVE krb5", Level.AUTH, (
        *_envelope(),
        R("krb5.msg_type", "s", "proto.kerberos.msg_type", "code:krb_msg_type", also="native"),
        R("krb5.failed_request", "s"),
        R("krb5.error_code", "s", "proto.kerberos.error_code", "code:krb_error", also="native"),
        R("krb5.cname", "s", "proto.kerberos.client", "string"),
        R("krb5.realm", "s", "proto.kerberos.realm", "string"),
        R("krb5.sname", "s", "proto.kerberos.service", "string"),
        R("krb5.encryption", "s", "proto.kerberos.etype", "code:krb_etype", also="native"),
        R("krb5.weak_encryption", "b", conv="bool"),
        R("krb5.ticket_encryption", "s", kind="fp"),
        R("krb5.ticket_weak_encryption", "b", conv="bool"),
    ), REF, ocsf_class=3002,
    notes=("Entities: account = realm\\cname, target account = sname (AS-692).",),
)

MQTT = RecordMap(
    "suricata", "mqtt", "Suricata EVE mqtt", Level.PROTOCOL, (
        *_envelope(),
        *(R(f"mqtt.{t}", "M") for t in ("connect", "connack", "publish", "puback", "pubrec", "pubrel", "pubcomp",
                                         "subscribe", "suback", "unsubscribe", "unsuback", "pingreq", "pingresp",
                                         "disconnect", "auth")),
    ), REF, ocsf_class=4001,
    notes=(
        "Derived: proto.mqtt.message_type from the control-packet object present; proto.mqtt.qos, topic, "
        "client_id and return_code from its fields (qos, topic, client_id, return_code or reason_code).",
    ),
)

MODBUS = RecordMap(
    "suricata", "modbus", "Suricata EVE modbus", Level.OT, (
        *_envelope(),
        R("modbus.id", "i", conv="count"),
        R("modbus.request", "M"),
        R("modbus.response", "M"),
    ), REF, layer=Layer.OT_CII, ocsf_class=4001,
    notes=(
        "Derived: ot.modbus.transaction_id, unit_id and function_code from the request (else the response): "
        "transaction_id, unit_id, function_raw (function_code); ot.modbus.exception_code from the response's "
        "exception; register start and count from read or write address and quantity.",
    ),
)

DNP3 = RecordMap(
    "suricata", "dnp3", "Suricata EVE dnp3", Level.OT, (
        *_envelope(),
        R("dnp3.type", "s"),
        R("dnp3.control", "M"),
        R("dnp3.src", "i", conv="count"),
        R("dnp3.dst", "i", conv="count"),
        R("dnp3.application", "M"),
        R("dnp3.iin", "M"),
    ), REF, layer=Layer.OT_CII, ocsf_class=4001,
    notes=(
        "Derived: ot.dnp3.function_code from application.function_code of a request, "
        "ot.dnp3.function_code_reply of a response; ot.dnp3.iin from iin.indicators (names to bits); "
        "ot.dnp3.object_group from the first object's group.",
    ),
)

STATS = RecordMap(
    "suricata", "stats", "Suricata EVE stats", Level.DEVICE, (
        R("timestamp", "t", "@time", "time"),
        R("event_type", "s", native=_n("event_type")),
        R("host", "s", "event.hostname", "string"),
        R("stats.uptime", "i", "dev.uptime", "interval", unit="s"),
        R("stats.capture.kernel_packets", "i", conv="count", unit="packets", note="Cumulative since start."),
        R("stats.capture.kernel_drops", "i", conv="count", unit="packets", note="Cumulative since start."),
        R("stats.capture.kernel_ifdrops", "i", conv="count", unit="packets"),
        R("stats.decoder.pkts", "i", conv="count", unit="packets"),
        R("stats.decoder.bytes", "i", conv="count", unit="bytes"),
        R("stats.decoder.invalid", "i", conv="count", unit="packets"),
        R("stats.detect.alert", "i", conv="count"),
        R("stats.flow.memuse", "i", conv="count", unit="bytes"),
    ), REF, ocsf_class=0,
    notes=(
        "Derived: dev.capture_received and dev.capture_dropped as deltas of the cumulative kernel counters between "
        "consecutive stats records of the same sensor (NOT_SUPPLIED on the first record and after a counter reset), "
        "dev.interval the time between them. Every other stats counter is retained as an attribute.",
        "Entities: the sensor (host) as subject.",
    ),
)

SURICATA_MAPS: tuple[RecordMap, ...] = (
    FLOW, NETFLOW, ALERT, ANOMALY, DNS, HTTP, TLS, FILEINFO, SMB, SSH, DHCP, KRB5, MQTT, MODBUS, DNP3, STATS,
)
