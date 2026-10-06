"""OCSF mapping tables: OCSF 1.x events of the classes Network Activity 4001, HTTP Activity 4002, DNS
Activity 4003, DHCP Activity 4004, RDP Activity 4005, SMB Activity 4006, SSH Activity 4007, FTP
Activity 4008, Email Activity 4009, NTP Activity 4013, Authentication 3002 and Detection Finding 2004,
plus the Base Event (class 0) for record types without a class of their own.

Row sources are OCSF attribute paths joined with dots (objects flattened; arrays kept whole). The
targeted schema version is OCSF 1.3.0 (AS-703). The same rows drive both directions: import maps a
path onto its catalogue field, export (ingest/ocsf.py) writes the catalogue field at the path with the
inverse converter. Catalogue fields without an OCSF attribute, and every observation status, travel in
`unmapped.nagahana` so that export followed by import is exact; a consumer that ignores `unmapped`
still reads a valid OCSF event.
"""

from __future__ import annotations

from nagahana.datamodel.native import R, RecordMap, Row
from nagahana.datamodel.spec import Level

REF = "OCSF schema 1.3.0 (schema.ocsf.io): base event, classes and objects"


def _base() -> tuple[Row, ...]:
    n = "ocsf."
    return (
        R("class_uid", "i", conv="count", native=n + "class_uid"),
        R("class_name", "s", native=n + "class_name"),
        R("category_uid", "i", conv="count", native=n + "category_uid"),
        R("category_name", "s", native=n + "category_name"),
        R("activity_name", "s", native=n + "activity_name"),
        R("type_uid", "i", conv="count", native=n + "type_uid"),
        R("type_name", "s", native=n + "type_name"),
        R("time", "i", "@time", "ms_time", note="Milliseconds since the epoch."),
        R("time_dt", "s", native=n + "time_dt"),
        R("start_time", "i", "flow.start_time", "ms_time", unit="s"),
        R("end_time", "i", "flow.end_time", "ms_time", unit="s"),
        R("duration", "i", "flow.duration", "ms", unit="s"),
        R("count", "i", conv="count", native=n + "count"),
        R("severity_id", "i", "event.severity", "count"),
        R("severity", "s", native=n + "severity"),
        R("status", "s", native=n + "status"),
        R("status_detail", "s", native=n + "status_detail"),
        R("message", "s", "event.message", "string"),
        R("timezone_offset", "i", conv="int", native=n + "timezone_offset"),
        R("raw_data", "s", native=n + "raw_data"),
        R("metadata.version", "s", native=n + "metadata.version"),
        R("metadata.uid", "s", native=n + "metadata.uid", kind="id"),
        R("metadata.product.name", "s", "event.product", "string"),
        R("metadata.product.vendor_name", "s", "event.vendor", "string"),
        R("metadata.product.version", "s", "event.product_version", "string"),
        R("metadata.product.uid", "s", native=n + "metadata.product.uid", kind="id"),
        R("metadata.original_time", "s", native=n + "metadata.original_time"),
        R("metadata.logged_time", "i", conv="ms_time", native=n + "metadata.logged_time", unit="s"),
        R("metadata.processed_time", "i", conv="ms_time", native=n + "metadata.processed_time", unit="s"),
        R("metadata.log_name", "s", native=n + "metadata.log_name"),
        R("metadata.log_provider", "s", native=n + "metadata.log_provider"),
        R("metadata.correlation_uid", "s", native=n + "metadata.correlation_uid", kind="id"),
        R("metadata.event_code", "s", "event.class_id", "string"),
        R("metadata.profiles", "S", conv="list", native=n + "metadata.profiles"),
        R("metadata.labels", "S", conv="list", native=n + "metadata.labels"),
        R("metadata.sequence", "i", conv="count", native=n + "metadata.sequence"),
        R("observables", "M", native=n + "observables"),
        R("enrichments", "M", native=n + "enrichments"),
        R("unmapped", "M", native=n + "unmapped", note="unmapped.nagahana carries statuses and fields without an "
          "OCSF attribute; it is read by the adapter, not by this row."),
    )


def _endpoint(side: str, *, primary: bool) -> tuple[Row, ...]:
    n = f"ocsf.{side}."
    fwd = side == "src_endpoint"
    rows = [
        R(f"{side}.ip", "a", "flow.src_ip" if fwd else "flow.dst_ip", "addr"),
        R(f"{side}.port", "i", "flow.src_port" if fwd else "flow.dst_port", "port"),
        R(f"{side}.hostname", "s", "flow.src_hostname" if fwd else "flow.dst_hostname", "string"),
        R(f"{side}.mac", "m", "flow.src_mac" if fwd else "flow.dst_mac", "mac"),
        R(f"{side}.uid", "s", native=n + "uid", kind="id"),
        R(f"{side}.name", "s", native=n + "name", kind="id"),
        R(f"{side}.domain", "s", native=n + "domain", kind="id"),
        R(f"{side}.interface_name", "s", native=n + "interface_name"),
        R(f"{side}.interface_uid", "s", native=n + "interface_uid"),
        R(f"{side}.autonomous_system.number", "i", "flow.src_as" if fwd else "flow.dst_as", "count"),
        R(f"{side}.autonomous_system.name", "s", native=n + "autonomous_system.name"),
        R(f"{side}.subnet_uid", "s", native=n + "subnet_uid"),
        R(f"{side}.svc_name", "s", native=n + "svc_name"),
        R(f"{side}.zone", "s", native=n + "zone"),
        R(f"{side}.location.country", "s", native=n + "location.country"),
        R(f"{side}.intermediate_ips", "A", conv="list_addr", native=n + "intermediate_ips"),
    ]
    rows.insert(4, R(f"{side}.vlan_uid", "s", "flow.vlan" if primary else None, "int_text" if primary else "",
                     native=None if primary else n + "vlan_uid"))
    return tuple(rows)


def _network() -> tuple[Row, ...]:
    n = "ocsf.connection_info."
    return (
        *_endpoint("src_endpoint", primary=True),
        *_endpoint("dst_endpoint", primary=False),
        R("connection_info.protocol_num", "i", "flow.protocol", "count"),
        R("connection_info.protocol_name", "s", native=n + "protocol_name"),
        R("connection_info.protocol_ver_id", "i", "flow.ip_version", "ocsf_ip_version"),
        R("connection_info.protocol_ver", "s", native=n + "protocol_ver"),
        R("connection_info.direction_id", "i", conv="count", native=n + "direction_id"),
        R("connection_info.direction", "s", native=n + "direction"),
        R("connection_info.boundary_id", "i", conv="count", native=n + "boundary_id"),
        R("connection_info.boundary", "s", native=n + "boundary"),
        R("connection_info.tcp_flags", "i", "flow.tcp_flags", "count"),
        R("connection_info.uid", "s", "flow.uid", "string"),
        R("connection_info.community_uid", "s", "flow.community_id", "string"),
        R("traffic.bytes_out", "i", "flow.bytes_fwd", "count", unit="bytes", note="Source to destination (AS-704)."),
        R("traffic.bytes_in", "i", "flow.bytes_bwd", "count", unit="bytes", note="Destination to source."),
        R("traffic.packets_out", "i", "flow.packets_fwd", "count", unit="packets"),
        R("traffic.packets_in", "i", "flow.packets_bwd", "count", unit="packets"),
        R("traffic.packets", "i", "flow.packets_total", "count", unit="packets"),
        R("traffic.bytes", "i", conv="count", native="ocsf.traffic.bytes", unit="bytes"),
        R("app_name", "s", "flow.app_proto", "app_proto", also="flow.app_proto_text"),
    )


def _tls() -> tuple[Row, ...]:
    n = "ocsf.tls."
    return (
        R("tls.version", "s", "proto.tls.version", "code:tls_version", also="proto.tls.version_text"),
        R("tls.cipher", "s", "proto.tls.cipher", "string"),
        R("tls.sni", "s", "proto.tls.sni", "string"),
        R("tls.alert", "i", "proto.tls.alert", "count"),
        R("tls.ja3_hash.value", "s", "proto.tls.ja3", "string"),
        R("tls.ja3s_hash.value", "s", "proto.tls.ja3s", "string"),
        R("tls.certificate.subject", "s", "proto.tls.cert_subject", "string"),
        R("tls.certificate.issuer", "s", "proto.tls.cert_issuer", "string"),
        R("tls.certificate.serial_number", "s", "proto.tls.cert_serial", "string"),
        R("tls.certificate.created_time", "i", "proto.tls.cert_not_before", "ms_time", unit="s"),
        R("tls.certificate.expiration_time", "i", "proto.tls.cert_not_after", "ms_time", unit="s"),
        R("tls.certificate.fingerprints", "M", native=n + "certificate.fingerprints"),
        R("tls.handshake_dur", "i", conv="count", native=n + "handshake_dur", unit="ms"),
        R("tls.extension_list", "M", native=n + "extension_list"),
    )


def _cls(uid: int, name: str, record: str, level: Level, rows: tuple[Row, ...], notes: tuple[str, ...] = ()) -> RecordMap:
    return RecordMap("ocsf", record, f"OCSF {name} ({uid})", level, (*_base(), *rows), REF,
                     native_prefix=f"ocsf.{record}", ocsf_class=uid, notes=notes)


NETWORK = _cls(4001, "Network Activity", "network_activity", Level.FLOW, (
    R("activity_id", "i", conv="count"), *_network(), *_tls(),
    R("ja4_fingerprint_list", "M"),
), ("activity_id on export: 2 Close when flow.end_reason is fin, 3 Reset when rst, 5 Refuse when flow.conn_state is "
    "REJ, else 6 Traffic.",))

HTTP = _cls(4002, "HTTP Activity", "http_activity", Level.PROTOCOL, (
    R("activity_id", "i", conv="count", note="1 Connect, 2 Delete, 3 Get, 4 Head, 5 Options, 6 Post, 7 Put, 8 Trace."),
    *_network(), *_tls(),
    R("http_request.http_method", "s", "proto.http.method", "code:http_method", also="proto.http.method_text"),
    R("http_request.url.url_string", "s", kind="id"),
    R("http_request.url.hostname", "s", "proto.http.host", "string"),
    R("http_request.url.path", "s", "proto.http.uri", "string"),
    R("http_request.url.query_string", "s"),
    R("http_request.url.scheme", "s"),
    R("http_request.url.port", "i", conv="port"),
    R("http_request.user_agent", "s", "proto.http.user_agent", "string"),
    R("http_request.referrer", "s", "proto.http.referrer", "string"),
    R("http_request.version", "s", "proto.http.version", "string"),
    R("http_request.length", "i", conv="count", unit="bytes", note="Whole request, not the body."),
    R("http_request.uid", "s", kind="id"),
    R("http_request.x_forwarded_for", "A", conv="list_addr"),
    R("http_request.http_headers", "M"),
    R("http_response.code", "i", "proto.http.status_code", "count"),
    R("http_response.message", "s", "proto.http.status_msg", "string"),
    R("http_response.status", "s"),
    R("http_response.content_type", "s", "proto.http.content_type", "string"),
    R("http_response.length", "i", conv="count", unit="bytes", note="Whole response, not the body."),
    R("http_response.http_headers", "M"),
    R("http_cookies", "M"),
), ("proto.http.uri = url.path, followed by '?' and url.query_string when present (export splits it).",))

DNS = _cls(4003, "DNS Activity", "dns_activity", Level.PROTOCOL, (
    R("activity_id", "i", conv="count", note="1 Query, 2 Response, 6 Traffic."),
    *_network(),
    R("query.hostname", "s", "proto.dns.query", "string"),
    R("query.type", "s", "proto.dns.qtype", "code:dns_qtype"),
    R("query.class", "s", "proto.dns.qclass", "code:dns_qclass"),
    R("query.opcode_id", "i", "proto.dns.opcode", "count"),
    R("query.opcode", "s"),
    R("query.packet_uid", "i", "proto.dns.trans_id", "count"),
    R("answers", "M", "proto.dns.answers", "ocsf_answers_rdata", also="native"),
    R("rcode_id", "i", "proto.dns.rcode", "ocsf_rcode"),
    R("rcode", "s"),
    R("query_time", "i", conv="ms_time", unit="s"),
    R("response_time", "i", conv="ms_time", unit="s"),
), ("Derived on import: proto.dns.ttls, answer_count and ttl_min from answers; proto.dns.flags from the first "
    "answer's flag_ids (1 AA, 2 TC, 3 RD, 4 RA, 5 AD, 6 CD); proto.dns.rtt = response_time - query_time.",))

DHCP = _cls(4004, "DHCP Activity", "dhcp_activity", Level.PROTOCOL, (
    R("activity_id", "i", "proto.dhcp.message_type", "count",
      note="1 Discover, 2 Offer, 3 Request, 4 Decline, 5 Ack, 6 Nak, 7 Release, 8 Inform (the DHCP message types)."),
    *_network(),
    R("lease_dur", "i", "proto.dhcp.lease_time", "interval", unit="s"),
    R("transaction_uid", "s", kind="id"),
    R("is_renewal", "b", conv="bool"),
    R("relay", "M"),
), ("proto.dhcp.client_mac and assigned_addr follow src_endpoint.mac and src_endpoint.ip.",))

RDP = _cls(4005, "RDP Activity", "rdp_activity", Level.PROTOCOL, (
    R("activity_id", "i", conv="count"), *_network(), *_tls(),
    R("identifier_cookie", "s", "proto.rdp.cookie", "string"),
    R("protocol_ver", "s"),
    R("keyboard_info.keyboard_layout", "s", "proto.rdp.keyboard_layout", "string"),
    R("remote_display.physical_width", "i", "proto.rdp.desktop_width", "count", unit="pixels"),
    R("remote_display.physical_height", "i", "proto.rdp.desktop_height", "count", unit="pixels"),
    R("remote_display.color_depth", "i", conv="count"),
    R("certificate_chain", "S", conv="list"),
    R("request", "M"),
    R("response", "M"),
    R("device", "M"),
))

SMB = _cls(4006, "SMB Activity", "smb_activity", Level.PROTOCOL, (
    R("activity_id", "i", conv="count"), *_network(),
    R("command", "s", "proto.smb.command", "code:smb_command", also="native"),
    R("dialect", "s", "proto.smb.dialect", "string"),
    R("file.name", "s", "file.name", "string"),
    R("file.size", "i", "file.size", "count", unit="bytes"),
    R("file.path", "s", kind="id"),
    R("share", "s", "proto.smb.path", "string"),
    R("share_type", "s", "proto.smb.share_type", "string"),
    R("share_type_id", "i", conv="count"),
    R("tree_uid", "s", kind="id"),
    R("open_type", "s"),
    R("response.code", "i", "proto.smb.status", "count"),
    R("response.message", "s"),
    R("client_dialects", "S", conv="list"),
    R("dce_rpc", "M"),
))

SSH = _cls(4007, "SSH Activity", "ssh_activity", Level.PROTOCOL, (
    R("activity_id", "i", conv="count", note="1 Open, 2 Close, 3 Reset, 4 Fail, 5 Refuse, 6 Traffic, 7 Listen."),
    *_network(),
    R("auth_type_id", "i", "auth.method", "ocsf_ssh_auth"),
    R("auth_type", "s"),
    R("client_hassh.fingerprint.value", "s", "proto.ssh.hassh", "string"),
    R("server_hassh.fingerprint.value", "s", "proto.ssh.hassh_server", "string"),
    R("protocol_ver", "s", "proto.ssh.version", "ssh_major"),
    R("file", "M"),
))

FTP = _cls(4008, "FTP Activity", "ftp_activity", Level.PROTOCOL, (
    R("activity_id", "i", conv="count", note="1 Put, 2 Get, 3 Poll, 4 Delete, 5 Rename, 6 List."),
    *_network(),
    R("command", "s", "proto.ftp.command", "string"),
    R("codes", "I", "proto.ftp.reply_code", "first_int", also="native"),
    R("command_responses", "S", conv="list"),
    R("name", "s", "proto.ftp.argument", "string"),
    R("port", "i", conv="port"),
    R("type", "S", conv="list"),
    R("file", "M"),
))

EMAIL = _cls(4009, "Email Activity", "email_activity", Level.PROTOCOL, (
    R("activity_id", "i", conv="count", note="1 Send, 2 Receive, 3 Scan."),
    *_network(),
    R("direction_id", "i", conv="count"),
    R("email.from", "s", "proto.smtp.from", "string"),
    R("email.to", "S", "proto.smtp.to", "list"),
    R("email.cc", "S", "proto.smtp.cc", "list"),
    R("email.subject", "s", "proto.smtp.subject", "string"),
    R("email.message_uid", "s", "proto.smtp.msg_id", "string"),
    R("email.reply_to", "s", "proto.smtp.reply_to", "string"),
    R("email.smtp_from", "s", "proto.smtp.mailfrom", "string"),
    R("email.smtp_to", "S", "proto.smtp.rcptto", "list"),
    R("email.x_originating_ip", "A", "proto.smtp.x_originating_ip", "first_addr", also="native"),
    R("email.size", "i", conv="count", unit="bytes"),
    R("email.uid", "s", kind="id"),
    R("smtp_hello", "s", "proto.smtp.helo", "string"),
    R("protocol_name", "s"),
    R("banner", "s"),
    R("attempt", "i", conv="count"),
))

NTP = _cls(4013, "NTP Activity", "ntp_activity", Level.PROTOCOL, (
    R("activity_id", "i", "proto.ntp.mode", "count", note="Activities 1 to 7 follow the NTP association modes."),
    *_network(),
    R("version", "s", "proto.ntp.version", "int_text"),
    R("stratum_id", "i", "proto.ntp.stratum", "count"),
    R("stratum", "s"),
    R("precision", "i", conv="int"),
    R("delay", "i", conv="int"),
    R("dispersion", "i", conv="int"),
))

AUTH = _cls(3002, "Authentication", "authentication", Level.AUTH, (
    R("activity_id", "i", "auth.activity", "count"),
    *_network(),
    R("status_id", "i", "auth.result", "ocsf_status_result"),
    R("status_code", "s", "auth.failure_status", "hex_nonzero", also="native"),
    R("user.name", "s", "auth.user", "string"),
    R("user.domain", "s", "auth.domain", "string"),
    R("user.uid", "s", "auth.user_sid", "string"),
    R("user.type_id", "i", conv="count"),
    R("logon_type_id", "i", "auth.logon_type", "count"),
    R("logon_type", "s"),
    R("auth_protocol_id", "i", "auth.protocol", "count"),
    R("auth_protocol", "s"),
    R("is_remote", "b", conv="bool"),
    R("is_mfa", "b", conv="bool"),
    R("is_new_logon", "b", conv="bool"),
    R("is_cleartext", "b", conv="bool"),
    R("logon_process.name", "s", "auth.logon_process", "string"),
    R("logon_process.pid", "i", conv="count"),
    R("session.uid", "s", "auth.logon_id", "string"),
    R("service.name", "s", "auth.service", "string"),
    R("service.uid", "s", kind="id"),
    R("actor.user.name", "s", "auth.subject_user", "string"),
    R("actor.user.domain", "s", "auth.subject_domain", "string"),
    R("actor.process.name", "s", "auth.process_name", "string"),
    R("certificate", "M"),
), ("Entities: src_endpoint host as initiator, dst_endpoint host as responder, user as account, service as target "
    "account.",))

FINDING = _cls(2004, "Detection Finding", "detection_finding", Level.ALERT, (
    R("activity_id", "i", conv="count", note="1 Create, 2 Update, 3 Close."),
    R("finding_info.uid", "s", kind="id"),
    R("finding_info.title", "s", "alert.message", "string"),
    R("finding_info.desc", "s"),
    R("finding_info.types", "S", "alert.category_text", "first_text", also="native"),
    R("finding_info.analytic.uid", "s", "alert.signature_id", "int_text"),
    R("finding_info.analytic.name", "s", "alert.rule", "string"),
    R("finding_info.analytic.type_id", "i", conv="count"),
    R("finding_info.analytic.category", "s"),
    R("finding_info.attacks", "M"),
    R("finding_info.src_url", "s", kind="id"),
    R("finding_info.first_seen_time", "i", conv="ms_time", unit="s"),
    R("finding_info.last_seen_time", "i", conv="ms_time", unit="s"),
    R("finding_info.created_time", "i", conv="ms_time", unit="s"),
    R("confidence_id", "i", conv="count"),
    R("risk_level_id", "i", conv="count"),
    R("risk_score", "i", conv="count"),
    R("impact_id", "i", conv="count"),
    R("action_id", "i", "event.action", "count"),
    R("disposition_id", "i", conv="count"),
    R("evidences", "M"),
    R("resources", "M"),
), ("Derived on import: alert.category from alert.category_text; alert.severity from severity_id; endpoints and "
    "connection info from evidences[0] (src_endpoint, dst_endpoint, connection_info).",))

BASE = _cls(0, "Base Event", "base_event", Level.EVENT, (
    R("activity_id", "i", conv="count"), *_network(),
), ("Record types without an OCSF class of their own (sensor statistics, device counters, inventories, syslog "
    "lines) export as the base event with every field in unmapped.nagahana.",))

OCSF_MAPS: tuple[RecordMap, ...] = (NETWORK, HTTP, DNS, DHCP, RDP, SMB, SSH, FTP, EMAIL, NTP, AUTH, FINDING, BASE)
OCSF_CLASS_MAPS: dict[int, RecordMap] = {m.ocsf_class: m for m in OCSF_MAPS}
