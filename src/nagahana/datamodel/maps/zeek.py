"""Zeek log mapping tables: conn, dns, http, ssl, x509, files, notice, weird, dhcp, ssh, smtp, rdp,
smb_files, smb_mapping, kerberos, ntlm, dce_rpc, modbus, dnp3, software, known_hosts, known_services.

Field lists follow the Zeek documentation of the log streams (the `Info` records of
base/protocols/*/main.zeek and base/frameworks/*/main.zeek, Zeek 6.0 LTS and 7.0) plus the fields that
standard policy scripts and widely deployed packages add (VLAN and MAC logging, community-id, JA3,
JA4, HASSH, software geolocation); a field a log does not carry is simply absent from its records.
Field names are those of the `#fields` header (TSV) and of the JSON keys. Converters are named in
ingest/convert.py. Conversions that are not one-to-one keep the source text verbatim (`also`).
"""

from __future__ import annotations

from nagahana.datamodel.layers import Layer
from nagahana.datamodel.native import R, RecordMap, Row
from nagahana.datamodel.spec import Level
from nagahana.datamodel.status import STATE_FACT_STATUSES

REF = "Zeek documentation, log stream Info records (Zeek 6.0 and 7.0)"


def _ts() -> Row:
    return R("ts", "t", "@time", "time", unit="s", note="Record timestamp; see the record notes for the event time.")


def _id(uid: bool = True) -> tuple[Row, ...]:
    rows = [_ts()]
    if uid:
        rows.append(R("uid", "s", "flow.uid", "string"))
    rows += [
        R("id.orig_h", "a", "flow.src_ip", "addr"),
        R("id.orig_p", "i", "flow.src_port", "port"),
        R("id.resp_h", "a", "flow.dst_ip", "addr"),
        R("id.resp_p", "i", "flow.dst_port", "port"),
    ]
    return tuple(rows)


def _geo(prefix: str) -> tuple[Row, ...]:
    return (
        R(f"{prefix}.country_code", "s"), R(f"{prefix}.region", "s"), R(f"{prefix}.city", "s"),
        R(f"{prefix}.latitude", "f", unit="deg"), R(f"{prefix}.longitude", "f", unit="deg"),
    )


CONN = RecordMap(
    "zeek", "conn", "Zeek conn.log", Level.FLOW, (
        *_id(),
        R("ip_proto", "i", "flow.protocol", "count", note="IP protocol number (newer Zeek); takes precedence over proto."),
        R("proto", "s", "flow.protocol", "proto_name", note="tcp, udp or icmp; icmp is 58 over IPv6."),
        R("service", "s", "flow.app_proto", "app_proto", also="flow.app_proto_text"),
        R("duration", "f", "flow.duration", "interval", unit="s"),
        R("orig_bytes", "i", "flow.payload_bytes_fwd", "count", unit="bytes",
          note="Payload bytes (from TCP sequence numbers)."),
        R("resp_bytes", "i", "flow.payload_bytes_bwd", "count", unit="bytes"),
        R("conn_state", "s", "flow.conn_state", "code:conn_state"),
        R("local_orig", "b", conv="bool"),
        R("local_resp", "b", conv="bool"),
        R("missed_bytes", "i", "flow.missed_bytes", "count", unit="bytes"),
        R("history", "s", kind="fp", note="State history: uppercase originator, lowercase responder."),
        R("orig_pkts", "i", "flow.packets_fwd", "count", unit="packets"),
        R("orig_ip_bytes", "i", "flow.bytes_fwd", "count", unit="bytes", note="IP total-length bytes."),
        R("resp_pkts", "i", "flow.packets_bwd", "count", unit="packets"),
        R("resp_ip_bytes", "i", "flow.bytes_bwd", "count", unit="bytes"),
        R("tunnel_parents", "S", conv="list"),
        R("orig_l2_addr", "m", "flow.src_mac", "mac"),
        R("resp_l2_addr", "m", "flow.dst_mac", "mac"),
        R("vlan", "i", "flow.vlan", "int"),
        R("inner_vlan", "i", "flow.vlan_inner", "int"),
        R("community_id", "s", "flow.community_id", "string"),
        R("speculative_service", "s", kind="fp"),
    ), REF, ocsf_class=4001,
    notes=(
        "Event time: ts + duration (the record summarises the flow up to its last packet, AS-300).",
        "ICMP: Zeek writes the ICMP type in id.orig_p and the code in id.resp_p; they map to proto.icmp.type "
        "and proto.icmp.code and the ports stay NOT_SUPPLIED.",
        "Derived: flow.packets_total = orig_pkts + resp_pkts; flow.unanswered = 1 if resp_pkts = 0; "
        "flow.bidir_ratio = resp_ip_bytes / orig_ip_bytes; flow.end_reason from conn_state (SF fin; REJ, RSTO, "
        "RSTR, RSTOS0, RSTRH rst; other states NOT_SUPPLIED); flow.start_time = ts; flow.end_time = ts + duration.",
        "Derived: flow.tcp_flags_fwd / _bwd / flow.tcp_flags from history letters S, H, A, F, R (PSH and URG are "
        "not encoded), LOW_RELIABILITY with reliability 4/6 (AS-688).",
        "Entities: initiator id.orig_h, responder id.resp_h, service id.resp_h:id.resp_p/proto.",
    ),
)

DNS = RecordMap(
    "zeek", "dns", "Zeek dns.log", Level.PROTOCOL, (
        *_id(),
        R("proto", "s", "flow.protocol", "proto_name"),
        R("trans_id", "i", "proto.dns.trans_id", "count"),
        R("rtt", "f", "proto.dns.rtt", "interval", unit="s"),
        R("query", "s", "proto.dns.query", "string"),
        R("qclass", "i", "proto.dns.qclass", "count"),
        R("qclass_name", "s"),
        R("qtype", "i", "proto.dns.qtype", "count"),
        R("qtype_name", "s"),
        R("rcode", "i", "proto.dns.rcode", "count"),
        R("rcode_name", "s"),
        R("AA", "b", "proto.dns.flags", "bit:0x01"),
        R("TC", "b", "proto.dns.flags", "bit:0x02"),
        R("RD", "b", "proto.dns.flags", "bit:0x04"),
        R("RA", "b", "proto.dns.flags", "bit:0x08"),
        R("Z", "i", "proto.dns.flags", "zbits", note="Three-bit field: Z, AD, CD (AS-689)."),
        R("answers", "S", "proto.dns.answers", "list"),
        R("TTLs", "F", "proto.dns.ttls", "list_interval", unit="s"),
        R("rejected", "b", "proto.dns.rejected", "bool"),
        R("auth", "S", conv="list"),
        R("addl", "S", conv="list"),
        R("opcode", "i", "proto.dns.opcode", "count"),
        R("opcode_name", "s"),
        R("original_query", "s", kind="id"),
    ), REF, ocsf_class=4003,
    notes=(
        "Event time: ts + rtt when the response was seen (the record holds the response), else ts.",
        "Derived: proto.dns.answer_count = number of answers when a response was seen (0 if the answers field is "
        "unset); proto.dns.ttl_min = min(TTLs); proto.dns.query_length and proto.dns.query_entropy from query.",
        "proto.dns.flags is OBSERVED only when AA, TC, RD, RA and Z are all set in the record.",
    ),
)

HTTP = RecordMap(
    "zeek", "http", "Zeek http.log", Level.PROTOCOL, (
        *_id(),
        R("trans_depth", "i", conv="count"),
        R("method", "s", "proto.http.method", "code:http_method", also="proto.http.method_text"),
        R("host", "s", "proto.http.host", "string"),
        R("uri", "s", "proto.http.uri", "string"),
        R("referrer", "s", "proto.http.referrer", "string"),
        R("version", "s", "proto.http.version", "string"),
        R("user_agent", "s", "proto.http.user_agent", "string"),
        R("origin", "s", kind="id"),
        R("request_body_len", "i", "proto.http.request_body_len", "count", unit="bytes"),
        R("response_body_len", "i", "proto.http.response_body_len", "count", unit="bytes"),
        R("status_code", "i", "proto.http.status_code", "count"),
        R("status_msg", "s", "proto.http.status_msg", "string"),
        R("info_code", "i", conv="count"),
        R("info_msg", "s"),
        R("tags", "S", conv="list"),
        R("username", "s", "proto.http.username", "string"),
        R("password", "s", kind="id", note="Logged only when Zeek is configured to capture passwords."),
        R("proxied", "S", conv="list"),
        R("orig_fuids", "S", conv="list"),
        R("orig_filenames", "S", conv="list"),
        R("orig_mime_types", "S", conv="list"),
        R("resp_fuids", "S", conv="list"),
        R("resp_filenames", "S", conv="list"),
        R("resp_mime_types", "S", conv="list"),
    ), REF, ocsf_class=4002,
    notes=(
        "Event time: ts (time of the request; Zeek logs no end time for the transaction, AS-690).",
        "flow.protocol = 6 (Zeek's HTTP analyser runs over TCP only). Derived: proto.http.uri_length from uri.",
    ),
)

SSL = RecordMap(
    "zeek", "ssl", "Zeek ssl.log", Level.PROTOCOL, (
        *_id(),
        R("version", "s", "proto.tls.version", "code:tls_version", also="proto.tls.version_text"),
        R("cipher", "s", "proto.tls.cipher", "string"),
        R("curve", "s", "proto.tls.curve", "string"),
        R("server_name", "s", "proto.tls.sni", "string"),
        R("resumed", "b", "proto.tls.resumed", "bool"),
        R("last_alert", "s", "proto.tls.alert", "code:tls_alert", also="native"),
        R("next_protocol", "s", "proto.tls.alpn", "string"),
        R("established", "b", "proto.tls.established", "bool01"),
        R("ssl_history", "s", kind="fp"),
        R("cert_chain_fps", "S", conv="list"),
        R("client_cert_chain_fps", "S", conv="list"),
        R("cert_chain_fuids", "S", conv="list"),
        R("client_cert_chain_fuids", "S", conv="list"),
        R("subject", "s", "proto.tls.cert_subject", "string"),
        R("issuer", "s", "proto.tls.cert_issuer", "string"),
        R("client_subject", "s", kind="id"),
        R("client_issuer", "s", kind="id"),
        R("sni_matches_cert", "b", conv="bool"),
        R("validation_status", "s"),
        R("ja3", "s", "proto.tls.ja3", "string"),
        R("ja3s", "s", "proto.tls.ja3s", "string"),
        R("ja4", "s", "proto.tls.ja4", "string"),
        R("ja4s", "s", "proto.tls.ja4s", "string"),
    ), REF, ocsf_class=4001,
    notes=(
        "Event time: ts. When an x509 record with the first fingerprint of cert_chain_fps was read earlier in "
        "the same session, proto.tls.cert_validity, cert_self_signed, cert_key_length, cert_not_before and "
        "cert_not_after are filled from it (bounded join, AS-691).",
    ),
)

X509 = RecordMap(
    "zeek", "x509", "Zeek x509.log", Level.PROTOCOL, (
        _ts(),
        R("fingerprint", "s", "proto.tls.cert_fingerprint", "string"),
        R("id", "s", kind="id", note="File identifier (Zeek before 5.0)."),
        R("certificate.version", "i", conv="count"),
        R("certificate.serial", "s", "proto.tls.cert_serial", "string"),
        R("certificate.subject", "s", "proto.tls.cert_subject", "string"),
        R("certificate.issuer", "s", "proto.tls.cert_issuer", "string"),
        R("certificate.not_valid_before", "t", "proto.tls.cert_not_before", "time", unit="s"),
        R("certificate.not_valid_after", "t", "proto.tls.cert_not_after", "time", unit="s"),
        R("certificate.key_alg", "s", kind="fp"),
        R("certificate.sig_alg", "s", kind="fp"),
        R("certificate.key_type", "s", kind="fp"),
        R("certificate.key_length", "i", "proto.tls.cert_key_length", "count", unit="bits"),
        R("certificate.exponent", "s"),
        R("certificate.curve", "s", kind="fp"),
        R("san.dns", "S", conv="list"),
        R("san.uri", "S", conv="list"),
        R("san.email", "S", conv="list"),
        R("san.ip", "A", conv="list_addr"),
        R("basic_constraints.ca", "b", conv="bool"),
        R("basic_constraints.path_len", "i", conv="count"),
        R("host_cert", "b", conv="bool"),
        R("client_cert", "b", conv="bool"),
    ), REF, native_statuses=STATE_FACT_STATUSES, ocsf_class=0,
    notes=(
        "Derived: proto.tls.cert_validity = not_valid_after - not_valid_before; proto.tls.cert_self_signed = "
        "1 if subject equals issuer.",
        "Entities: the endpoints of the ssl record that presented the certificate when it was read earlier in "
        "the session (AS-691); otherwise the sensor as subject.",
    ),
)

FILES = RecordMap(
    "zeek", "files", "Zeek files.log", Level.PROTOCOL, (
        _ts(),
        R("fuid", "s", "file.uid", "string"),
        R("uid", "s", "flow.uid", "string"),
        R("id.orig_h", "a", "flow.src_ip", "addr"),
        R("id.orig_p", "i", "flow.src_port", "port"),
        R("id.resp_h", "a", "flow.dst_ip", "addr"),
        R("id.resp_p", "i", "flow.dst_port", "port"),
        R("tx_hosts", "A", conv="list_addr", note="Sending hosts (Zeek before 5.1)."),
        R("rx_hosts", "A", conv="list_addr", note="Receiving hosts (Zeek before 5.1)."),
        R("conn_uids", "S", conv="list"),
        R("source", "s", "file.source", "string"),
        R("depth", "i", conv="count"),
        R("analyzers", "S", conv="list"),
        R("mime_type", "s", "file.mime_type", "string"),
        R("filename", "s", "file.name", "string"),
        R("duration", "f", conv="interval", unit="s", note="Time over which the file was seen."),
        R("local_orig", "b", conv="bool"),
        R("is_orig", "b", conv="bool"),
        R("seen_bytes", "i", conv="count", unit="bytes"),
        R("total_bytes", "i", "file.size", "count", unit="bytes"),
        R("missing_bytes", "i", conv="count", unit="bytes"),
        R("overflow_bytes", "i", conv="count", unit="bytes"),
        R("timedout", "b", conv="bool"),
        R("parent_fuid", "s", kind="id"),
        R("md5", "s", "file.md5", "string"),
        R("sha1", "s", "file.sha1", "string"),
        R("sha256", "s", "file.sha256", "string"),
        R("extracted", "s", kind="id"),
        R("extracted_cutoff", "b", conv="bool"),
        R("extracted_size", "i", conv="count", unit="bytes"),
        R("entropy", "f", conv="double", unit="bits"),
    ), REF, ocsf_class=0,
    notes=("Entities: id.orig_h / id.resp_h when present, else the first tx_hosts / rx_hosts element.",),
)

NOTICE = RecordMap(
    "zeek", "notice", "Zeek notice.log", Level.ALERT, (
        *_id(),
        R("fuid", "s", "file.uid", "string"),
        R("file_mime_type", "s", "file.mime_type", "string"),
        R("file_desc", "s"),
        R("proto", "s", "flow.protocol", "proto_name"),
        R("note", "s", "alert.rule", "string"),
        R("msg", "s", "alert.message", "string"),
        R("sub", "s"),
        R("src", "a", conv="addr", kind="id"),
        R("dst", "a", conv="addr", kind="id"),
        R("p", "i", conv="port"),
        R("n", "i", conv="count"),
        R("peer_descr", "s"),
        R("peer_name", "s"),
        R("actions", "S", conv="list"),
        R("email_dest", "S", conv="list"),
        R("suppress_for", "f", conv="interval", unit="s"),
        R("dropped", "b", conv="bool"),
        *_geo("remote_location"),
    ), REF, ocsf_class=2004,
    notes=(
        "Derived: alert.signature from the note name in the Zeek-notice namespace (AS-687).",
        "Entities: id.orig_h / id.resp_h when present, else src / dst.",
    ),
)

WEIRD = RecordMap(
    "zeek", "weird", "Zeek weird.log", Level.ALERT, (
        *_id(),
        R("name", "s", "alert.rule", "string"),
        R("addl", "s"),
        R("notice", "b", conv="bool"),
        R("peer", "s"),
        R("source", "s"),
    ), REF, ocsf_class=2004,
    notes=("Derived: alert.signature from the weird name in the Zeek-weird namespace (AS-687).",),
)

DHCP = RecordMap(
    "zeek", "dhcp", "Zeek dhcp.log", Level.PROTOCOL, (
        _ts(),
        R("uids", "S", conv="list"),
        R("client_addr", "a", "flow.src_ip", "addr"),
        R("server_addr", "a", "flow.dst_ip", "addr"),
        R("client_port", "i", "flow.src_port", "port"),
        R("server_port", "i", "flow.dst_port", "port"),
        R("mac", "m", "proto.dhcp.client_mac", "mac"),
        R("host_name", "s", "proto.dhcp.hostname", "string"),
        R("client_fqdn", "s", kind="id"),
        R("domain", "s", "proto.dhcp.domain", "string"),
        R("requested_addr", "a", "proto.dhcp.requested_addr", "addr"),
        R("assigned_addr", "a", "proto.dhcp.assigned_addr", "addr"),
        R("lease_time", "f", "proto.dhcp.lease_time", "interval", unit="s"),
        R("client_message", "s"),
        R("server_message", "s"),
        R("msg_types", "S", "proto.dhcp.message_type", "last_dhcp_type", also="native"),
        R("duration", "f", conv="interval", unit="s", note="Duration of the DHCP exchange."),
        R("client_software", "s", kind="fp"),
        R("server_software", "s", kind="fp"),
        R("circuit_id", "s"),
        R("agent_remote_id", "s"),
        R("subscriber_id", "s"),
        R("msg_orig", "A", conv="list_addr"),
    ), REF, ocsf_class=4004,
    notes=(
        "flow.protocol = 17 (DHCP runs over UDP). proto.dhcp.message_type is the last message type of the "
        "exchange.",
        "Entities: initiator client_addr (else assigned_addr), responder server_addr.",
    ),
)

SSH = RecordMap(
    "zeek", "ssh", "Zeek ssh.log", Level.PROTOCOL, (
        *_id(),
        R("version", "i", "proto.ssh.version", "count"),
        R("auth_success", "b", "proto.ssh.auth_success", "bool01"),
        R("auth_attempts", "i", "proto.ssh.auth_attempts", "count"),
        R("direction", "s"),
        R("client", "s", "proto.ssh.client", "string"),
        R("server", "s", "proto.ssh.server", "string"),
        R("cipher_alg", "s", kind="fp"),
        R("mac_alg", "s", kind="fp"),
        R("compression_alg", "s", kind="fp"),
        R("kex_alg", "s", kind="fp"),
        R("host_key_alg", "s", kind="fp"),
        R("host_key", "s", "proto.ssh.host_key", "string"),
        R("hasshVersion", "s"),
        R("hassh", "s", "proto.ssh.hassh", "string"),
        R("hasshServer", "s", "proto.ssh.hassh_server", "string"),
        R("cshka", "s", kind="fp"),
        R("hasshAlgorithms", "s", kind="fp"),
        R("sshka", "s", kind="fp"),
        R("hasshServerAlgorithms", "s", kind="fp"),
        *_geo("remote_location"),
    ), REF, ocsf_class=4007,
    notes=("flow.protocol = 6 (Zeek's SSH analyser runs over TCP only).",),
)

SMTP = RecordMap(
    "zeek", "smtp", "Zeek smtp.log", Level.PROTOCOL, (
        *_id(),
        R("trans_depth", "i", conv="count"),
        R("helo", "s", "proto.smtp.helo", "string"),
        R("mailfrom", "s", "proto.smtp.mailfrom", "string"),
        R("rcptto", "S", "proto.smtp.rcptto", "list"),
        R("date", "s"),
        R("from", "s", "proto.smtp.from", "string"),
        R("to", "S", "proto.smtp.to", "list"),
        R("cc", "S", "proto.smtp.cc", "list"),
        R("reply_to", "s", "proto.smtp.reply_to", "string"),
        R("msg_id", "s", "proto.smtp.msg_id", "string"),
        R("in_reply_to", "s", kind="id"),
        R("subject", "s", "proto.smtp.subject", "string"),
        R("x_originating_ip", "a", "proto.smtp.x_originating_ip", "addr"),
        R("first_received", "s"),
        R("second_received", "s"),
        R("last_reply", "s"),
        R("path", "A", conv="list_addr"),
        R("user_agent", "s", kind="fp"),
        R("tls", "b", conv="bool"),
        R("fuids", "S", conv="list"),
        R("is_webmail", "b", conv="bool"),
    ), REF, ocsf_class=4009,
    notes=("flow.protocol = 6 (SMTP over TCP).",),
)

RDP = RecordMap(
    "zeek", "rdp", "Zeek rdp.log", Level.PROTOCOL, (
        *_id(),
        R("cookie", "s", "proto.rdp.cookie", "string"),
        R("result", "s", "proto.rdp.result", "string"),
        R("security_protocol", "s", "proto.rdp.security_protocol", "string"),
        R("client_channels", "S", conv="list"),
        R("keyboard_layout", "s", "proto.rdp.keyboard_layout", "string"),
        R("client_build", "s", "proto.rdp.client_build", "string"),
        R("client_name", "s", "proto.rdp.client_name", "string"),
        R("client_dig_product_id", "s", kind="id"),
        R("desktop_width", "i", "proto.rdp.desktop_width", "count", unit="pixels"),
        R("desktop_height", "i", "proto.rdp.desktop_height", "count", unit="pixels"),
        R("requested_color_depth", "s"),
        R("cert_type", "s"),
        R("cert_count", "i", conv="count"),
        R("cert_permanent", "b", conv="bool"),
        R("encryption_level", "s"),
        R("encryption_method", "s"),
        R("ssl", "b", conv="bool"),
    ), REF, ocsf_class=4005,
)

SMB_FILES = RecordMap(
    "zeek", "smb_files", "Zeek smb_files.log", Level.PROTOCOL, (
        *_id(),
        R("fuid", "s", "file.uid", "string"),
        R("action", "s", "proto.smb.file_action", "code:smb_action"),
        R("path", "s", "proto.smb.path", "string"),
        R("name", "s", "file.name", "string"),
        R("size", "i", "file.size", "count", unit="bytes"),
        R("prev_name", "s", kind="id"),
        R("times.modified", "t", conv="time", unit="s"),
        R("times.accessed", "t", conv="time", unit="s"),
        R("times.created", "t", conv="time", unit="s"),
        R("times.changed", "t", conv="time", unit="s"),
        R("data_offset_req", "i", conv="count", unit="bytes"),
        R("data_len_req", "i", conv="count", unit="bytes"),
        R("data_len_rsp", "i", conv="count", unit="bytes"),
    ), REF, ocsf_class=4006,
    notes=("flow.protocol = 6 (SMB over TCP).",),
)

SMB_MAPPING = RecordMap(
    "zeek", "smb_mapping", "Zeek smb_mapping.log", Level.PROTOCOL, (
        *_id(),
        R("path", "s", "proto.smb.path", "string"),
        R("service", "s", kind="fp"),
        R("native_file_system", "s", kind="fp"),
        R("share_type", "s", "proto.smb.share_type", "string"),
    ), REF, ocsf_class=4006,
    notes=("flow.protocol = 6 (SMB over TCP).",),
)

KERBEROS = RecordMap(
    "zeek", "kerberos", "Zeek kerberos.log", Level.AUTH, (
        *_id(),
        R("request_type", "s", "proto.kerberos.msg_type", "krb_request"),
        R("client", "s", "proto.kerberos.client", "string"),
        R("service", "s", "proto.kerberos.service", "string"),
        R("success", "b", "proto.kerberos.success", "bool"),
        R("error_msg", "s", "proto.kerberos.error_code", "code:krb_error", also="native"),
        R("from", "t", conv="time", unit="s"),
        R("till", "t", conv="time", unit="s"),
        R("cipher", "s", "proto.kerberos.etype", "code:krb_etype", also="native"),
        R("forwardable", "b", conv="bool"),
        R("renewable", "b", conv="bool"),
        R("client_cert_subject", "s", kind="id"),
        R("client_cert_fuid", "s", kind="id"),
        R("server_cert_subject", "s", kind="id"),
        R("server_cert_fuid", "s", kind="id"),
    ), REF, ocsf_class=3002,
    notes=(
        "Derived: proto.kerberos.error_code = 0 when success is true and no error is logged; auth.activity = 3 "
        "(AS request) or 4 (TGS request); auth.protocol = 2 (Kerberos); auth.result from success.",
        "Entities: initiator id.orig_h, responder id.resp_h (the KDC), account = client principal, target "
        "account = service principal (AS-692).",
    ),
)

NTLM = RecordMap(
    "zeek", "ntlm", "Zeek ntlm.log", Level.AUTH, (
        *_id(),
        R("username", "s", "auth.user", "string"),
        R("hostname", "s", "auth.workstation", "string"),
        R("domainname", "s", "auth.domain", "string"),
        R("server_nb_computer_name", "s", kind="id"),
        R("server_dns_computer_name", "s", kind="id"),
        R("server_tree_name", "s", kind="id"),
        R("success", "b", "auth.result", "auth_result"),
        R("status", "s", note="Status text (older Zeek)."),
    ), REF, ocsf_class=3002,
    notes=(
        "Derived: auth.protocol = 1 (NTLM); auth.activity = 1 (Logon).",
        "Entities: initiator id.orig_h, responder id.resp_h, account = domainname\\username.",
    ),
)

DCE_RPC = RecordMap(
    "zeek", "dce_rpc", "Zeek dce_rpc.log", Level.PROTOCOL, (
        *_id(),
        R("rtt", "f", "proto.dcerpc.rtt", "interval", unit="s"),
        R("named_pipe", "s", "proto.dcerpc.named_pipe", "string"),
        R("endpoint", "s", "proto.dcerpc.endpoint", "string"),
        R("operation", "s", "proto.dcerpc.operation", "string"),
    ), REF, ocsf_class=4001,
)

MODBUS = RecordMap(
    "zeek", "modbus", "Zeek modbus.log", Level.OT, (
        *_id(),
        R("tid", "i", "ot.modbus.transaction_id", "count"),
        R("unit", "i", "ot.modbus.unit_id", "count"),
        R("func", "s", "ot.modbus.function_code", "code:modbus_function", also="native"),
        R("pdu_type", "s"),
        R("exception", "s", "ot.modbus.exception_code", "code:modbus_exception", also="native"),
    ), REF, layer=Layer.OT_CII, ocsf_class=4001,
    notes=("flow.protocol = 6 (Modbus/TCP).",),
)

DNP3 = RecordMap(
    "zeek", "dnp3", "Zeek dnp3.log", Level.OT, (
        *_id(),
        R("fc_request", "s", "ot.dnp3.function_code", "code:dnp3_function", also="native"),
        R("fc_reply", "s", "ot.dnp3.function_code_reply", "code:dnp3_function", also="native"),
        R("iin", "i", "ot.dnp3.iin", "count"),
    ), REF, layer=Layer.OT_CII, ocsf_class=4001,
)

SOFTWARE = RecordMap(
    "zeek", "software", "Zeek software.log", Level.PROTOCOL, (
        _ts(),
        R("host", "a", "flow.src_ip", "addr", note="Host running the software (the subject entity)."),
        R("host_p", "i", conv="port"),
        R("software_type", "s", kind="fp"),
        R("name", "s", kind="fp"),
        R("version.major", "i", conv="count"),
        R("version.minor", "i", conv="count"),
        R("version.minor2", "i", conv="count"),
        R("version.minor3", "i", conv="count"),
        R("version.addl", "s"),
        R("unparsed_version", "s", kind="fp"),
        R("url", "s", kind="id"),
    ), REF, native_statuses=STATE_FACT_STATUSES, ocsf_class=0,
    notes=("Entities: the host as subject.",),
)

KNOWN_HOSTS = RecordMap(
    "zeek", "known_hosts", "Zeek known_hosts.log", Level.PROTOCOL, (
        _ts(),
        R("host", "a", "flow.src_ip", "addr", note="Host seen completing a connection (the subject entity)."),
    ), REF, native_statuses=STATE_FACT_STATUSES, ocsf_class=0,
)

KNOWN_SERVICES = RecordMap(
    "zeek", "known_services", "Zeek known_services.log", Level.PROTOCOL, (
        _ts(),
        R("host", "a", "flow.dst_ip", "addr", note="Host offering the service."),
        R("port_num", "i", "flow.dst_port", "port"),
        R("port_proto", "s", "flow.protocol", "proto_name"),
        R("service", "S", "flow.app_proto", "app_proto_list", also="native"),
    ), REF, native_statuses=STATE_FACT_STATUSES, ocsf_class=0,
    notes=("Entities: the host as subject and its service host:port/proto.",),
)

OTHER = RecordMap(
    "zeek", "other", "Zeek log without a table of its own", Level.FLOW, (
        _ts(),
        R("uid", "s", "flow.uid", "string"),
        R("id.orig_h", "a", "flow.src_ip", "addr"),
        R("id.orig_p", "i", "flow.src_port", "port"),
        R("id.resp_h", "a", "flow.dst_ip", "addr"),
        R("id.resp_p", "i", "flow.dst_port", "port"),
    ), REF, ocsf_class=0,
    notes=("Every other field is retained as an attribute 'zeek.<log>.<field>'; the record type is 'zeek.<log>'.",),
)

ZEEK_MAPS: tuple[RecordMap, ...] = (
    CONN, DNS, HTTP, SSL, X509, FILES, NOTICE, WEIRD, DHCP, SSH, SMTP, RDP, SMB_FILES, SMB_MAPPING, KERBEROS, NTLM,
    DCE_RPC, MODBUS, DNP3, SOFTWARE, KNOWN_HOSTS, KNOWN_SERVICES, OTHER,
)
