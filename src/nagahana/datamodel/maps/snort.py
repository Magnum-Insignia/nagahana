"""Snort mapping tables: Snort 3 alert_json, Snort 2 unified2 (IDS event v1 and v2, IPv4 and IPv6;
packet; extra data), and the alert_fast and alert_full text formats.

References: Snort 3 reference manual, module alert_json (field list); Snort 2.9 source,
src/sfutil/Unified2_common.h (unified2 record layouts; record types 2, 7, 72, 104, 105, 110);
Snort 2 users manual section 2.6 (output modules alert_fast, alert_full).
"""

from __future__ import annotations

from nagahana.datamodel.native import R, RecordMap, Row
from nagahana.datamodel.spec import Level

REF_JSON = "Snort 3 reference manual, alert_json"
REF_U2 = "Snort 2.9 src/sfutil/Unified2_common.h"
REF_TXT = "Snort 2 users manual, output modules alert_fast and alert_full"

ALERT_JSON = RecordMap(
    "snort", "alert_json", "Snort 3 alert_json", Level.ALERT, (
        R("seconds", "i", "@time", "epoch_seconds", note="Integer seconds; used when timestamp has no year."),
        R("timestamp", "s", native="snort.alert_json.timestamp", note="Text time, year only with show_year."),
        R("pkt_num", "i", conv="count"),
        R("proto", "s", "flow.protocol", "proto_name"),
        R("pkt_gen", "s"),
        R("pkt_len", "i", conv="count", unit="bytes"),
        R("dir", "s"),
        R("src_addr", "a", "flow.src_ip", "addr"),
        R("src_port", "i", "flow.src_port", "port"),
        R("dst_addr", "a", "flow.dst_ip", "addr"),
        R("dst_port", "i", "flow.dst_port", "port"),
        R("src_ap", "s", note="address:port"),
        R("dst_ap", "s", note="address:port"),
        R("rule", "s", "alert.rule", "string", note="gid:sid:rev"),
        R("action", "s", "event.action", "code:event_action", also="native"),
        R("msg", "s", "alert.message", "string"),
        R("class", "s", "alert.category", "code:alert_category", also="alert.category_text"),
        R("priority", "i", "alert.priority", "count"),
        R("gid", "i", "alert.generator_id", "count"),
        R("sid", "i", "alert.signature_id", "count"),
        R("rev", "i", "alert.revision", "count"),
        R("service", "s", "flow.app_proto", "app_proto", also="flow.app_proto_text"),
        R("eth_src", "m", "flow.src_mac", "mac"),
        R("eth_dst", "m", "flow.dst_mac", "mac"),
        R("eth_type", "s"),
        R("eth_len", "i", conv="count", unit="bytes"),
        R("ip_id", "i", conv="count"),
        R("ip_len", "i", conv="count", unit="bytes", note="IP total length of the packet."),
        R("tos", "i", "flow.ip_tos", "count"),
        R("ttl", "i", "pkt.ttl_mean", "count", unit="hops"),
        R("tcp_ack", "i", conv="count"),
        R("tcp_flags", "s", "flow.tcp_flags_fwd", "snort_flags"),
        R("tcp_len", "i", conv="count", unit="bytes"),
        R("tcp_seq", "i", conv="count"),
        R("tcp_win", "i", conv="count", unit="bytes"),
        R("udp_len", "i", conv="count", unit="bytes"),
        R("icmp_code", "i", "proto.icmp.code", "count"),
        R("icmp_id", "i", conv="count"),
        R("icmp_seq", "i", conv="count"),
        R("icmp_type", "i", "proto.icmp.type", "count"),
        R("b64_data", "s", note="Packet payload, base64."),
        R("mpls", "i", conv="count"),
        R("vlan", "i", "flow.vlan", "int"),
        R("iface", "s"),
        R("flowstart_time", "i", "flow.start_time", "epoch_seconds", unit="s"),
        R("client_bytes", "i", "flow.frame_bytes_fwd", "count", unit="bytes"),
        R("client_pkts", "i", "flow.packets_fwd", "count", unit="packets"),
        R("server_bytes", "i", "flow.frame_bytes_bwd", "count", unit="bytes"),
        R("server_pkts", "i", "flow.packets_bwd", "count", unit="packets"),
        R("target", "s"),
        R("geneve_vni", "i", conv="count"),
        R("sgt", "i", conv="count"),
    ), REF_JSON, ocsf_class=2004,
    notes=(
        "Event time: seconds plus the microseconds of timestamp when timestamp parses; otherwise seconds.",
        "Derived: alert.signature = gid * 2^32 + sid; alert.severity from priority (AS-694); for the one packet "
        "of the alert: pkt.ttl_min = pkt.ttl_max = ttl, pkt.ip_len_min = pkt.ip_len_max = ip_len, flow.bytes_fwd "
        "= ip_len, flow.tcp_flags = tcp_flags & 0x3F.",
        "Entities: initiator src_addr (the packet's sender), responder dst_addr.",
    ),
)

_U2_EVENT_ROWS: tuple[Row, ...] = (
    R("sensor_id", "i", conv="count"),
    R("event_id", "i", conv="count"),
    R("event_second", "i", "@time", "epoch_seconds"),
    R("event_microsecond", "i", conv="count", unit="us"),
    R("signature_id", "i", "alert.signature_id", "count"),
    R("generator_id", "i", "alert.generator_id", "count"),
    R("signature_revision", "i", "alert.revision", "count"),
    R("classification_id", "i", conv="count", note="Index into the sensor's classification.config."),
    R("priority_id", "i", "alert.priority", "count"),
    R("ip_source", "a", "flow.src_ip", "addr"),
    R("ip_destination", "a", "flow.dst_ip", "addr"),
    R("sport_itype", "i", "flow.src_port", "port", note="Source port, or ICMP type for ICMP."),
    R("dport_icode", "i", "flow.dst_port", "port", note="Destination port, or ICMP code for ICMP."),
    R("protocol", "i", "flow.protocol", "count"),
    R("impact_flag", "i", conv="count"),
    R("impact", "i", conv="count"),
    R("blocked", "i", "event.action", "u2_blocked", also="native", note="0 not blocked, 1 blocked, 2 would be blocked."),
    R("mpls_label", "i", conv="count", note="Event v2 only."),
    R("vlanId", "i", "flow.vlan", "int", note="Event v2 only."),
    R("pad2", "i", conv="count", note="Event v2 only (padding)."),
)

U2_EVENT = RecordMap(
    "snort", "unified2_event", "Snort unified2 IDS event (types 7, 72, 104, 105)", Level.ALERT, _U2_EVENT_ROWS,
    REF_U2, ocsf_class=2004,
    notes=(
        "Event time: event_second + event_microsecond / 1e6.",
        "Packet records (type 2) and extra-data records (type 110) that follow an event with the same sensor_id, "
        "event_id and event_second are folded into its state update: the first packet is decoded (dpkt) for "
        "packet-level fields; extra data fill the fields of `snort.unified2_extra`.",
        "ICMP: sport_itype and dport_icode map to proto.icmp.type and proto.icmp.code.",
    ),
)

U2_PACKET = RecordMap(
    "snort", "unified2_packet", "Snort unified2 packet (type 2)", Level.PACKET, (
        R("sensor_id", "i", conv="count"),
        R("event_id", "i", conv="count"),
        R("event_second", "i", conv="count", unit="s"),
        R("packet_second", "i", "@time", "epoch_seconds"),
        R("packet_microsecond", "i", conv="count", unit="us"),
        R("linktype", "i", conv="count", note="DLT of packet_data."),
        R("packet_length", "i", conv="count", unit="bytes"),
        R("packet_data", "y", note="The packet bytes; decoded for packet-level fields."),
    ), REF_U2, ocsf_class=2004,
    notes=("A packet record without a preceding event becomes its own state update with the decoded fields.",),
)

U2_EXTRA = RecordMap(
    "snort", "unified2_extra", "Snort unified2 extra data (type 110)", Level.ALERT, (
        R("event_type", "i", conv="count"),
        R("event_length", "i", conv="count", unit="bytes"),
        R("sensor_id", "i", conv="count"),
        R("event_id", "i", conv="count"),
        R("event_second", "i", conv="count", unit="s"),
        R("type", "i", conv="count", note="1 XFF IPv4, 2 XFF IPv6, 3 reviewed by, 4 gzip data, 5 SMTP filename, "
          "6 SMTP mail from, 7 SMTP rcpt to, 8 SMTP headers, 9 HTTP URI, 10 HTTP host name, 11 IPv6 source, "
          "12 IPv6 destination, 13 normalised JavaScript."),
        R("data_type", "i", conv="count"),
        R("blob_length", "i", conv="count", unit="bytes"),
        R("xff_ipv4", "a", conv="addr", kind="id"),
        R("xff_ipv6", "a", conv="addr", kind="id"),
        R("reviewed_by", "s"),
        R("gzip_data", "y"),
        R("smtp_filename", "s", "file.name", "string"),
        R("smtp_mailfrom", "s", "proto.smtp.mailfrom", "string"),
        R("smtp_rcptto", "s", kind="id"),
        R("smtp_headers", "s"),
        R("http_uri", "s", "proto.http.uri", "string"),
        R("http_hostname", "s", "proto.http.host", "string"),
        R("ipv6_source", "a", conv="addr", kind="id"),
        R("ipv6_destination", "a", conv="addr", kind="id"),
        R("jsnorm_data", "s"),
    ), REF_U2, ocsf_class=2004,
    notes=("The data of each extra-data type lands in the row named for it (data length = blob_length - 8).",),
)

ALERT_FAST = RecordMap(
    "snort", "alert_fast", "Snort alert_fast", Level.ALERT, (
        R("timestamp", "s", "@time", "snort_time"),
        R("action", "s", "event.action", "code:event_action", also="native"),
        R("gid", "i", "alert.generator_id", "count"),
        R("sid", "i", "alert.signature_id", "count"),
        R("rev", "i", "alert.revision", "count"),
        R("msg", "s", "alert.message", "string"),
        R("classification", "s", "alert.category", "code:alert_category", also="alert.category_text"),
        R("priority", "i", "alert.priority", "count"),
        R("proto", "s", "flow.protocol", "proto_name"),
        R("src_addr", "a", "flow.src_ip", "addr"),
        R("src_port", "i", "flow.src_port", "port"),
        R("dst_addr", "a", "flow.dst_ip", "addr"),
        R("dst_port", "i", "flow.dst_port", "port"),
    ), REF_TXT, ocsf_class=2004,
    notes=("Line layout: TIME [**] [gid:sid:rev] MSG [**] [Classification: C] [Priority: P] {PROTO} SRC -> DST.",),
)

ALERT_FULL = RecordMap(
    "snort", "alert_full", "Snort alert_full", Level.ALERT, (
        *ALERT_FAST.rows,
        R("ttl", "i", "pkt.ttl_mean", "count", unit="hops"),
        R("tos", "i", "flow.ip_tos", "hex"),
        R("ip_id", "i", conv="count"),
        R("ip_len", "i", conv="count", unit="bytes", note="IpLen: IP header length."),
        R("dgm_len", "i", conv="count", unit="bytes", note="DgmLen: IP total length."),
        R("ip_flags", "s", note="DF / MF markers."),
        R("tcp_flags", "s", "flow.tcp_flags_fwd", "snort_flags"),
        R("tcp_seq", "i", conv="hex"),
        R("tcp_ack", "i", conv="hex"),
        R("tcp_win", "i", conv="hex", unit="bytes"),
        R("tcp_len", "i", conv="count", unit="bytes"),
        R("tcp_options", "s"),
        R("udp_len", "i", conv="count", unit="bytes"),
        R("icmp_type", "i", "proto.icmp.type", "count"),
        R("icmp_code", "i", "proto.icmp.code", "count"),
        R("icmp_id", "i", conv="count"),
        R("icmp_seq", "i", conv="count"),
        R("xref", "S", conv="list"),
    ), REF_TXT, ocsf_class=2004,
    notes=(
        "Multi-line record ended by a blank line; the packet lines give the IP, TCP, UDP or ICMP header fields.",
        "Derived as for alert_json: pkt.ttl_min / max, pkt.ip_len_min / max and flow.bytes_fwd from DgmLen; "
        "pkt.ip_df_count / ip_mf_count from the DF / MF markers; flow.tcp_flags = tcp_flags & 0x3F.",
    ),
)

SNORT_MAPS: tuple[RecordMap, ...] = (ALERT_JSON, U2_EVENT, U2_PACKET, U2_EXTRA, ALERT_FAST, ALERT_FULL)
