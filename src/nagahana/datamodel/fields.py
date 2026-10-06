"""Field catalogue: every field a state update may carry, from every source.

Coverage (D-38): a state update carries flow-level and packet-level features at least, plus further
features from the observable region. The flow-level and packet-level lists below follow the problem
statement's two feature levels item by item, so the required coverage is checkable
(`required_ids`). Since data-model version 0.2 the catalogue is the superset of every supported
telemetry format (datamodel.md items 2 and 9): each field of each source maps to exactly one catalogue
field, either a shared field defined here (when several sources carry the same quantity) or a
source-native field generated from the mapping tables of `datamodel/maps` (`native.py`).

Design rules
------------
- IDs are stable strings ("flow.bytes_fwd"). The model's input layer keys on them (P-22), so adding a
  field never renumbers others; the column order of the matrix kinds is `data.windows.COLUMN_SLOTS`.
- Addresses are not features. IP addresses identify entities and build the hypergraph
  (`Kind.IDENTIFIER`); raw addresses are volatile, high-entropy and say little about the threat.
- Ports are categorical, never numeric: port 445 is not "close to" port 443. They stay features, since
  the problem statement asks which "flags, ports, or flow patterns" drive a prediction.
- OT fields are first-class (D-34) and sit in layer L4.
- Absolute clock times are attributes, never model inputs (D-50).
- A field's `statuses` lists the observation statuses its values may carry (`status.py`); a status
  outside that set is rejected when a state update is built.
- `codes` documents categorical and bitmask code spaces. A code space may grow (a minor version
  bump); a code never changes meaning (that would be a major bump, `versioning.py`).

Source-specific definitions (which layer "bytes" counts, how a source defines a ratio) are recorded
per adapter; a quantity measured under a different definition gets its own field (AS-303).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from types import MappingProxyType

from nagahana.core.errors import InvariantViolation
from nagahana.datamodel.layers import Layer
from nagahana.datamodel.spec import MATRIX_KINDS, Dtype, FieldSpec, Kind, Level, check_value
from nagahana.datamodel.status import IDENTITY_STATUSES, MEASUREMENT_STATUSES, STATE_FACT_STATUSES

_PS = "problem statement, flow-level feature list"
_PP = "problem statement, packet-level feature list"
_EX = "D-38 observable-region example; inclusion set by stage-1 analysis"
#: Sources of the fields the problem statement and the decided design require (`required_ids`).
REQUIRED_SOURCES: frozenset[str] = frozenset({_PS, _PP, "D-53"})

V02 = "0.2"

# Code tables of categorical and bitmask fields. Each documents a code space; adapters map source
# values into it (ingest/codes.py holds the source spellings).

TCP_FLAG_BITS: dict[int, str] = {
    0x01: "FIN", 0x02: "SYN", 0x04: "RST", 0x08: "PSH", 0x10: "ACK", 0x20: "URG", 0x40: "ECE", 0x80: "CWR",
}
#: Zeek connection states (Zeek documentation, base/protocols/conn/main.zeek, Conn::Info$conn_state).
CONN_STATE_CODES: dict[int, str] = {
    0: "other", 1: "S0", 2: "S1", 3: "SF", 4: "REJ", 5: "S2", 6: "S3", 7: "RSTO", 8: "RSTR", 9: "RSTOS0",
    10: "RSTRH", 11: "SH", 12: "SHR", 13: "OTH",
}
#: How a flow ended (D-53; AS-337 for codes 1 to 4, AS-681 for 5 to 8).
END_REASON_CODES: dict[int, str] = {
    1: "fin", 2: "rst", 3: "idle", 4: "capture_end", 5: "active_timeout", 6: "forced_end",
    7: "lack_of_resources", 8: "end_detected",
}
#: Application protocol identified by the source's analysers (AS-682). 0: a label outside this table
#: (the source's text is kept in `flow.app_proto_text`).
APP_PROTO_CODES: dict[int, str] = {
    0: "other", 1: "dns", 2: "http", 3: "tls", 4: "ssh", 5: "smtp", 6: "ftp", 7: "ftp-data", 8: "smb",
    9: "dce-rpc", 10: "kerberos", 11: "ntlm", 12: "rdp", 13: "dhcp", 14: "ntp", 15: "snmp", 16: "sip",
    17: "irc", 18: "mysql", 19: "postgresql", 20: "imap", 21: "pop3", 22: "ldap", 23: "radius", 24: "socks",
    25: "rfb", 26: "telnet", 27: "syslog", 28: "tftp", 29: "nfs", 30: "modbus", 31: "dnp3", 32: "iec104",
    33: "enip", 34: "s7comm", 35: "bacnet", 36: "mqtt", 37: "http2", 38: "quic", 39: "ike", 40: "openvpn",
    41: "wireguard", 42: "gssapi", 43: "netbios", 44: "mdns", 45: "llmnr", 46: "bittorrent", 47: "websocket",
    48: "xmpp", 49: "sunrpc", 50: "dhcpv6", 51: "ssdp", 52: "rtsp", 53: "rtp", 54: "stun", 55: "dtls",
}
#: Forwarding status byte (RFC 7270 section 4.12): status in the top two bits (1 forwarded, 2 dropped,
#: 3 consumed), reason code in the low six bits.
FORWARDING_STATUS_CODES: dict[int, str] = {
    0: "unknown", 64: "forwarded: unknown", 65: "forwarded: fragmented", 66: "forwarded: not fragmented",
    128: "dropped: unknown", 129: "dropped: ACL deny", 130: "dropped: ACL drop", 131: "dropped: unroutable",
    132: "dropped: adjacency", 133: "dropped: fragmentation and DF set", 134: "dropped: bad header checksum",
    135: "dropped: bad total length", 136: "dropped: bad header length", 137: "dropped: bad TTL",
    138: "dropped: policer", 139: "dropped: WRED", 140: "dropped: RPF", 141: "dropped: for us",
    142: "dropped: bad output interface", 143: "dropped: hardware", 192: "consumed: unknown",
    193: "consumed: punt adjacency", 194: "consumed: incomplete adjacency", 195: "consumed: for us",
}
#: DNS classes (IANA DNS CLASSes registry; RFC 1035, RFC 2136).
DNS_QCLASS_CODES: dict[int, str] = {1: "IN", 3: "CH", 4: "HS", 254: "NONE", 255: "ANY"}
#: DNS header flag bits as carried in `proto.dns.flags` (RFC 1035 section 4.1.1; AD and CD RFC 4035).
DNS_FLAG_BITS: dict[int, str] = {0x01: "AA", 0x02: "TC", 0x04: "RD", 0x08: "RA", 0x10: "Z", 0x20: "AD", 0x40: "CD"}
#: DNS response codes (IANA DNS RCODEs registry).
DNS_RCODE_CODES: dict[int, str] = {
    0: "NOERROR", 1: "FORMERR", 2: "SERVFAIL", 3: "NXDOMAIN", 4: "NOTIMP", 5: "REFUSED", 6: "YXDOMAIN",
    7: "YXRRSET", 8: "NXRRSET", 9: "NOTAUTH", 10: "NOTZONE", 11: "DSOTYPENI", 16: "BADVERS", 17: "BADKEY",
    18: "BADTIME", 19: "BADMODE", 20: "BADNAME", 21: "BADALG", 22: "BADTRUNC", 23: "BADCOOKIE",
}
#: DNS RR types (IANA DNS Resource Record TYPEs registry).
DNS_QTYPE_CODES: dict[int, str] = {
    1: "A", 2: "NS", 3: "MD", 4: "MF", 5: "CNAME", 6: "SOA", 7: "MB", 8: "MG", 9: "MR", 10: "NULL", 11: "WKS",
    12: "PTR", 13: "HINFO", 14: "MINFO", 15: "MX", 16: "TXT", 17: "RP", 18: "AFSDB", 19: "X25", 20: "ISDN",
    21: "RT", 22: "NSAP", 23: "NSAP-PTR", 24: "SIG", 25: "KEY", 26: "PX", 27: "GPOS", 28: "AAAA", 29: "LOC",
    30: "NXT", 31: "EID", 32: "NIMLOC", 33: "SRV", 34: "ATMA", 35: "NAPTR", 36: "KX", 37: "CERT", 38: "A6",
    39: "DNAME", 40: "SINK", 41: "OPT", 42: "APL", 43: "DS", 44: "SSHFP", 45: "IPSECKEY", 46: "RRSIG",
    47: "NSEC", 48: "DNSKEY", 49: "DHCID", 50: "NSEC3", 51: "NSEC3PARAM", 52: "TLSA", 53: "SMIMEA", 55: "HIP",
    59: "CDS", 60: "CDNSKEY", 61: "OPENPGPKEY", 62: "CSYNC", 63: "ZONEMD", 64: "SVCB", 65: "HTTPS", 99: "SPF",
    108: "EUI48", 109: "EUI64", 249: "TKEY", 250: "TSIG", 251: "IXFR", 252: "AXFR", 253: "MAILB", 254: "MAILA",
    255: "ANY", 256: "URI", 257: "CAA", 32768: "TA", 32769: "DLV",
}
#: HTTP request methods (RFC 9110 section 9; WebDAV RFC 4918; PATCH RFC 5789; SEARCH RFC 5323).
HTTP_METHOD_CODES: dict[int, str] = {
    0: "other", 1: "GET", 2: "POST", 3: "HEAD", 4: "PUT", 5: "DELETE", 6: "OPTIONS", 7: "CONNECT", 8: "TRACE",
    9: "PATCH", 10: "PROPFIND", 11: "PROPPATCH", 12: "MKCOL", 13: "COPY", 14: "MOVE", 15: "LOCK", 16: "UNLOCK",
    17: "SEARCH",
}
#: TLS and DTLS protocol versions by wire code (RFC 8446 appendix B.1, RFC 9147).
TLS_VERSION_CODES: dict[int, str] = {
    0x0002: "SSLv2", 0x0300: "SSLv3", 0x0301: "TLSv1.0", 0x0302: "TLSv1.1", 0x0303: "TLSv1.2", 0x0304: "TLSv1.3",
    0xFEFF: "DTLSv1.0", 0xFEFD: "DTLSv1.2", 0xFEFC: "DTLSv1.3",
}
#: TLS alert descriptions (RFC 8446 section 6; RFC 5246 section 7.2; IANA TLS Alerts registry).
TLS_ALERT_CODES: dict[int, str] = {
    0: "close_notify", 10: "unexpected_message", 20: "bad_record_mac", 21: "decryption_failed",
    22: "record_overflow", 30: "decompression_failure", 40: "handshake_failure", 41: "no_certificate",
    42: "bad_certificate", 43: "unsupported_certificate", 44: "certificate_revoked", 45: "certificate_expired",
    46: "certificate_unknown", 47: "illegal_parameter", 48: "unknown_ca", 49: "access_denied", 50: "decode_error",
    51: "decrypt_error", 60: "export_restriction", 70: "protocol_version", 71: "insufficient_security",
    80: "internal_error", 86: "inappropriate_fallback", 90: "user_canceled", 100: "no_renegotiation",
    109: "missing_extension", 110: "unsupported_extension", 111: "certificate_unobtainable",
    112: "unrecognized_name", 113: "bad_certificate_status_response", 114: "bad_certificate_hash_value",
    115: "unknown_psk_identity", 116: "certificate_required", 120: "no_application_protocol",
}
#: Kerberos message types (RFC 4120 section 5.10).
KRB_MSG_TYPE_CODES: dict[int, str] = {
    10: "AS-REQ", 11: "AS-REP", 12: "TGS-REQ", 13: "TGS-REP", 14: "AP-REQ", 15: "AP-REP", 20: "KRB-SAFE",
    21: "KRB-PRIV", 22: "KRB-CRED", 30: "KRB-ERROR",
}
#: Kerberos error codes (RFC 4120 section 7.5.9; PKINIT codes RFC 4556 section 3.1.3).
KRB_ERROR_CODES: dict[int, str] = {
    0: "KDC_ERR_NONE", 1: "KDC_ERR_NAME_EXP", 2: "KDC_ERR_SERVICE_EXP", 3: "KDC_ERR_BAD_PVNO",
    4: "KDC_ERR_C_OLD_MAST_KVNO", 5: "KDC_ERR_S_OLD_MAST_KVNO", 6: "KDC_ERR_C_PRINCIPAL_UNKNOWN",
    7: "KDC_ERR_S_PRINCIPAL_UNKNOWN", 8: "KDC_ERR_PRINCIPAL_NOT_UNIQUE", 9: "KDC_ERR_NULL_KEY",
    10: "KDC_ERR_CANNOT_POSTDATE", 11: "KDC_ERR_NEVER_VALID", 12: "KDC_ERR_POLICY", 13: "KDC_ERR_BADOPTION",
    14: "KDC_ERR_ETYPE_NOSUPP", 15: "KDC_ERR_SUMTYPE_NOSUPP", 16: "KDC_ERR_PADATA_TYPE_NOSUPP",
    17: "KDC_ERR_TRTYPE_NOSUPP", 18: "KDC_ERR_CLIENT_REVOKED", 19: "KDC_ERR_SERVICE_REVOKED",
    20: "KDC_ERR_TGT_REVOKED", 21: "KDC_ERR_CLIENT_NOTYET", 22: "KDC_ERR_SERVICE_NOTYET", 23: "KDC_ERR_KEY_EXPIRED",
    24: "KDC_ERR_PREAUTH_FAILED", 25: "KDC_ERR_PREAUTH_REQUIRED", 26: "KDC_ERR_SERVER_NOMATCH",
    27: "KDC_ERR_MUST_USE_USER2USER", 28: "KDC_ERR_PATH_NOT_ACCEPTED", 29: "KDC_ERR_SVC_UNAVAILABLE",
    31: "KRB_AP_ERR_BAD_INTEGRITY", 32: "KRB_AP_ERR_TKT_EXPIRED", 33: "KRB_AP_ERR_TKT_NYV", 34: "KRB_AP_ERR_REPEAT",
    35: "KRB_AP_ERR_NOT_US", 36: "KRB_AP_ERR_BADMATCH", 37: "KRB_AP_ERR_SKEW", 38: "KRB_AP_ERR_BADADDR",
    39: "KRB_AP_ERR_BADVERSION", 40: "KRB_AP_ERR_MSG_TYPE", 41: "KRB_AP_ERR_MODIFIED", 42: "KRB_AP_ERR_BADORDER",
    44: "KRB_AP_ERR_BADKEYVER", 45: "KRB_AP_ERR_NOKEY", 46: "KRB_AP_ERR_MUT_FAIL", 47: "KRB_AP_ERR_BADDIRECTION",
    48: "KRB_AP_ERR_METHOD", 49: "KRB_AP_ERR_BADSEQ", 50: "KRB_AP_ERR_INAPP_CKSUM", 51: "KRB_AP_PATH_NOT_ACCEPTED",
    52: "KRB_ERR_RESPONSE_TOO_BIG", 60: "KRB_ERR_GENERIC", 61: "KRB_ERR_FIELD_TOOLONG",
    62: "KDC_ERROR_CLIENT_NOT_TRUSTED", 63: "KDC_ERROR_KDC_NOT_TRUSTED", 64: "KDC_ERROR_INVALID_SIG",
    65: "KDC_ERR_KEY_TOO_WEAK", 66: "KDC_ERR_CERTIFICATE_MISMATCH", 67: "KRB_AP_ERR_NO_TGT",
    68: "KDC_ERR_WRONG_REALM", 69: "KRB_AP_ERR_USER_TO_USER_REQUIRED", 70: "KDC_ERR_CANT_VERIFY_CERTIFICATE",
    71: "KDC_ERR_INVALID_CERTIFICATE", 72: "KDC_ERR_REVOKED_CERTIFICATE", 73: "KDC_ERR_REVOCATION_STATUS_UNKNOWN",
    74: "KDC_ERR_REVOCATION_STATUS_UNAVAILABLE", 75: "KDC_ERR_CLIENT_NAME_MISMATCH", 76: "KDC_ERR_KDC_NAME_MISMATCH",
}
#: Kerberos encryption types (IANA Kerberos Encryption Type Numbers; RFC 3961, 3962, 4757, 8009, 6803).
KRB_ETYPE_CODES: dict[int, str] = {
    1: "des-cbc-crc", 2: "des-cbc-md4", 3: "des-cbc-md5", 5: "des3-cbc-md5", 7: "des3-cbc-sha1",
    16: "des3-cbc-sha1-kd", 17: "aes128-cts-hmac-sha1-96", 18: "aes256-cts-hmac-sha1-96",
    19: "aes128-cts-hmac-sha256-128", 20: "aes256-cts-hmac-sha384-192", 23: "rc4-hmac", 24: "rc4-hmac-exp",
    25: "camellia128-cts-cmac", 26: "camellia256-cts-cmac",
}
#: Authentication activity (OCSF Authentication class 3002, activity_id).
AUTH_ACTIVITY_CODES: dict[int, str] = {
    1: "Logon", 2: "Logoff", 3: "Authentication Ticket", 4: "Service Ticket Request", 5: "Service Ticket Renew",
    6: "Preauth", 99: "Other",
}
#: Authentication outcome (OCSF status_id).
AUTH_RESULT_CODES: dict[int, str] = {1: "Success", 2: "Failure"}
#: Logon types (Windows Security auditing; OCSF logon_type_id uses the same numbers).
LOGON_TYPE_CODES: dict[int, str] = {
    0: "System", 2: "Interactive", 3: "Network", 4: "Batch", 5: "Service", 7: "Unlock", 8: "NetworkCleartext",
    9: "NewCredentials", 10: "RemoteInteractive", 11: "CachedInteractive", 12: "CachedRemoteInteractive",
    13: "CachedUnlock",
}
#: Authentication protocol (OCSF auth_protocol_id).
AUTH_PROTOCOL_CODES: dict[int, str] = {
    0: "Unknown", 1: "NTLM", 2: "Kerberos", 3: "Digest", 4: "OpenID", 5: "SAML", 6: "OAUTH 2.0", 7: "PAP", 8: "CHAP",
    9: "EAP", 10: "RADIUS", 99: "Other",
}
#: Credential type presented (SSH authentication method names, RFC 4252 and RFC 4256; AS-683).
AUTH_METHOD_CODES: dict[int, str] = {
    1: "password", 2: "publickey", 3: "keyboard-interactive", 4: "hostbased", 5: "gssapi-with-mic", 6: "none",
    7: "certificate", 99: "other",
}
YES_NO_CODES: dict[int, str] = {0: "no", 1: "yes"}
#: Severity (OCSF severity_id).
SEVERITY_CODES: dict[int, str] = {
    0: "Unknown", 1: "Informational", 2: "Low", 3: "Medium", 4: "High", 5: "Critical", 6: "Fatal", 99: "Other",
}
#: Detector rule classification (Snort classification.config classtypes; Emerging Threats additions;
#: AS-684). 0: a classification outside this table (its text is kept in `alert.category_text`).
ALERT_CATEGORY_CODES: dict[int, str] = {
    0: "other", 1: "not-suspicious", 2: "unknown", 3: "bad-unknown", 4: "attempted-recon",
    5: "successful-recon-limited", 6: "successful-recon-largescale", 7: "attempted-dos", 8: "successful-dos",
    9: "attempted-user", 10: "unsuccessful-user", 11: "successful-user", 12: "attempted-admin",
    13: "successful-admin", 14: "rpc-portmap-decode", 15: "shellcode-detect", 16: "string-detect",
    17: "suspicious-filename-detect", 18: "suspicious-login", 19: "system-call-detect", 20: "tcp-connection",
    21: "trojan-activity", 22: "unusual-client-port-connection", 23: "network-scan", 24: "denial-of-service",
    25: "non-standard-protocol", 26: "protocol-command-decode", 27: "web-application-activity",
    28: "web-application-attack", 29: "misc-activity", 30: "misc-attack", 31: "icmp-event",
    32: "inappropriate-content", 33: "policy-violation", 34: "default-login-attempt", 35: "sdf", 36: "file-format",
    37: "malware-cnc", 38: "client-side-exploit", 39: "credential-theft", 40: "social-engineering",
    41: "exploit-kit", 42: "domain-c2", 43: "external-ip-check", 44: "targeted-activity", 45: "pup-activity",
    46: "coin-mining", 47: "command-and-control",
}
#: Action taken on the activity (OCSF action_id; AS-685).
EVENT_ACTION_CODES: dict[int, str] = {0: "Unknown", 1: "Allowed", 2: "Denied", 3: "Observed", 4: "Modified", 99: "Other"}
#: Modbus exception codes (Modbus Application Protocol Specification V1.1b3, section 7).
MODBUS_EXCEPTION_CODES: dict[int, str] = {
    1: "ILLEGAL FUNCTION", 2: "ILLEGAL DATA ADDRESS", 3: "ILLEGAL DATA VALUE", 4: "SERVER DEVICE FAILURE",
    5: "ACKNOWLEDGE", 6: "SERVER DEVICE BUSY", 8: "MEMORY PARITY ERROR", 10: "GATEWAY PATH UNAVAILABLE",
    11: "GATEWAY TARGET DEVICE FAILED TO RESPOND",
}
#: Modbus public function codes (same specification, section 5.1).
MODBUS_FUNCTION_CODES: dict[int, str] = {
    1: "READ_COILS", 2: "READ_DISCRETE_INPUTS", 3: "READ_HOLDING_REGISTERS", 4: "READ_INPUT_REGISTERS",
    5: "WRITE_SINGLE_COIL", 6: "WRITE_SINGLE_REGISTER", 7: "READ_EXCEPTION_STATUS", 8: "DIAGNOSTICS",
    11: "GET_COMM_EVENT_COUNTER", 12: "GET_COMM_EVENT_LOG", 15: "WRITE_MULTIPLE_COILS",
    16: "WRITE_MULTIPLE_REGISTERS", 17: "REPORT_SERVER_ID", 20: "READ_FILE_RECORD", 21: "WRITE_FILE_RECORD",
    22: "MASK_WRITE_REGISTER", 23: "READ_WRITE_MULTIPLE_REGISTERS", 24: "READ_FIFO_QUEUE",
    43: "ENCAPSULATED_INTERFACE_TRANSPORT",
}
#: DNP3 application function codes (IEEE 1815-2012, table 4-1 of the application layer).
DNP3_FUNCTION_CODES: dict[int, str] = {
    0: "CONFIRM", 1: "READ", 2: "WRITE", 3: "SELECT", 4: "OPERATE", 5: "DIRECT_OPERATE", 6: "DIRECT_OPERATE_NR",
    7: "IMMED_FREEZE", 8: "IMMED_FREEZE_NR", 9: "FREEZE_CLEAR", 10: "FREEZE_CLEAR_NR", 11: "FREEZE_AT_TIME",
    12: "FREEZE_AT_TIME_NR", 13: "COLD_RESTART", 14: "WARM_RESTART", 15: "INITIALIZE_DATA", 16: "INITIALIZE_APPL",
    17: "START_APPL", 18: "STOP_APPL", 19: "SAVE_CONFIG", 20: "ENABLE_UNSOLICITED", 21: "DISABLE_UNSOLICITED",
    22: "ASSIGN_CLASS", 23: "DELAY_MEASURE", 24: "RECORD_CURRENT_TIME", 25: "OPEN_FILE", 26: "CLOSE_FILE",
    27: "DELETE_FILE", 28: "GET_FILE_INFO", 29: "AUTHENTICATE_FILE", 30: "ABORT_FILE", 31: "ACTIVATE_CONFIG",
    32: "AUTHENTICATE_REQ", 33: "AUTH_REQ_NO_ACK", 129: "RESPONSE", 130: "UNSOLICITED_RESPONSE",
    131: "AUTHENTICATE_RESP",
}
#: DNP3 internal indications as a 16-bit value (IIN1 << 8) | IIN2 (IEEE 1815-2012 section 4.2.2.7.2).
DNP3_IIN_BITS: dict[int, str] = {
    0x0100: "IIN1.0 BROADCAST", 0x0200: "IIN1.1 CLASS_1_EVENTS", 0x0400: "IIN1.2 CLASS_2_EVENTS",
    0x0800: "IIN1.3 CLASS_3_EVENTS", 0x1000: "IIN1.4 NEED_TIME", 0x2000: "IIN1.5 LOCAL_CONTROL",
    0x4000: "IIN1.6 DEVICE_TROUBLE", 0x8000: "IIN1.7 DEVICE_RESTART", 0x0001: "IIN2.0 NO_FUNC_CODE_SUPPORT",
    0x0002: "IIN2.1 OBJECT_UNKNOWN", 0x0004: "IIN2.2 PARAMETER_ERROR", 0x0008: "IIN2.3 EVENT_BUFFER_OVERFLOW",
    0x0010: "IIN2.4 ALREADY_EXECUTING", 0x0020: "IIN2.5 CONFIG_CORRUPT", 0x0040: "IIN2.6 RESERVED_2",
    0x0080: "IIN2.7 RESERVED_1",
}
#: MQTT control packet types (MQTT Version 5.0, OASIS Standard, section 2.1.2).
MQTT_TYPE_CODES: dict[int, str] = {
    1: "CONNECT", 2: "CONNACK", 3: "PUBLISH", 4: "PUBACK", 5: "PUBREC", 6: "PUBREL", 7: "PUBCOMP", 8: "SUBSCRIBE",
    9: "SUBACK", 10: "UNSUBSCRIBE", 11: "UNSUBACK", 12: "PINGREQ", 13: "PINGRESP", 14: "DISCONNECT", 15: "AUTH",
}
#: SMB commands: SMB2 command codes (MS-SMB2 section 2.2.1.2), SMB1 commands as 0x100 + code (MS-CIFS
#: section 2.2.2.1).
SMB_COMMAND_CODES: dict[int, str] = {
    0x00: "SMB2_NEGOTIATE", 0x01: "SMB2_SESSION_SETUP", 0x02: "SMB2_LOGOFF", 0x03: "SMB2_TREE_CONNECT",
    0x04: "SMB2_TREE_DISCONNECT", 0x05: "SMB2_CREATE", 0x06: "SMB2_CLOSE", 0x07: "SMB2_FLUSH", 0x08: "SMB2_READ",
    0x09: "SMB2_WRITE", 0x0A: "SMB2_LOCK", 0x0B: "SMB2_IOCTL", 0x0C: "SMB2_CANCEL", 0x0D: "SMB2_ECHO",
    0x0E: "SMB2_QUERY_DIRECTORY", 0x0F: "SMB2_CHANGE_NOTIFY", 0x10: "SMB2_QUERY_INFO", 0x11: "SMB2_SET_INFO",
    0x12: "SMB2_OPLOCK_BREAK",
    0x104: "SMB1_CLOSE", 0x106: "SMB1_DELETE", 0x107: "SMB1_RENAME", 0x124: "SMB1_LOCKING_ANDX",
    0x125: "SMB1_TRANSACTION", 0x12B: "SMB1_ECHO", 0x12D: "SMB1_OPEN_ANDX", 0x12E: "SMB1_READ_ANDX",
    0x12F: "SMB1_WRITE_ANDX", 0x132: "SMB1_TRANSACTION2", 0x134: "SMB1_FIND_CLOSE2", 0x171: "SMB1_TREE_DISCONNECT",
    0x172: "SMB1_NEGOTIATE", 0x173: "SMB1_SESSION_SETUP_ANDX", 0x174: "SMB1_LOGOFF_ANDX",
    0x175: "SMB1_TREE_CONNECT_ANDX", 0x1A0: "SMB1_NT_TRANSACT", 0x1A2: "SMB1_NT_CREATE_ANDX",
}
#: File actions seen over SMB (the Zeek SMB::Action enumeration, in its declaration order; AS-686).
SMB_FILE_ACTION_CODES: dict[int, str] = {
    1: "FILE_READ", 2: "FILE_WRITE", 3: "FILE_OPEN", 4: "FILE_CLOSE", 5: "FILE_DELETE", 6: "FILE_RENAME",
    7: "FILE_SET_ATTRIBUTE", 8: "PIPE_READ", 9: "PIPE_WRITE", 10: "PIPE_OPEN", 11: "PIPE_CLOSE", 12: "PRINT_READ",
    13: "PRINT_WRITE", 14: "PRINT_OPEN", 15: "PRINT_CLOSE",
}
#: DHCP message types (RFC 2132 section 9.6; RFC 3203; RFC 4388).
DHCP_MESSAGE_CODES: dict[int, str] = {
    1: "DISCOVER", 2: "OFFER", 3: "REQUEST", 4: "DECLINE", 5: "ACK", 6: "NAK", 7: "RELEASE", 8: "INFORM",
    9: "FORCERENEW", 10: "LEASEQUERY", 11: "LEASEUNASSIGNED", 12: "LEASEUNKNOWN", 13: "LEASEACTIVE",
}
#: NTP association modes (RFC 5905 section 7.3).
NTP_MODE_CODES: dict[int, str] = {
    0: "reserved", 1: "symmetric active", 2: "symmetric passive", 3: "client", 4: "server", 5: "broadcast",
    6: "NTP control message", 7: "private use",
}
#: Interface status bits (sFlow v5 generic interface counters, ifStatus).
IF_STATUS_BITS: dict[int, str] = {0x01: "admin up", 0x02: "operational up"}
#: Port-access evidence is derived over records (pcap.py docstring, AS-305).

_SH = "shared field"


def _s(fid: str, level: Level, kind: Kind, unit: str | None, desc: str, req: str, *, layer: Layer = Layer.STATE,
       dtype: Dtype | None = None, statuses: frozenset | None = None, codes: Mapping[int, str] | None = None,
       since: str = V02) -> FieldSpec:
    """A shared field added in data-model version 0.2 (keyword arguments as in `FieldSpec`)."""
    return FieldSpec(fid, level, kind, unit, desc, req, False, layer, dtype, statuses, codes, since)


_FLOW_SRC = "D-38: flow sources (Zeek conn, Suricata flow, NetFlow, IPFIX, sFlow)"
_DNS_SRC = "D-38: Zeek dns, Suricata dns, tshark, OCSF DNS Activity"
_HTTP_SRC = "D-38: Zeek http, Suricata http, tshark, OCSF HTTP Activity"
_TLS_SRC = "D-38: Zeek ssl and x509, Suricata tls, tshark, OCSF tls object"
_AUTH_SRC = "D-38: Windows Security events, Zeek kerberos and ntlm, Suricata krb5, OCSF Authentication"
_ALERT_SRC = "D-38: Snort, Suricata alert, Zeek notice and weird, CEF and LEEF, OCSF Detection Finding"
_DEV_SRC = "D-38: sFlow counter samples, gNMI interface telemetry, Suricata stats, pcapng statistics"
_EVT_SRC = "D-38: event envelopes of syslog, CEF, LEEF, Windows events and OCSF"

_FIELDS: tuple[FieldSpec, ...] = (
    # Flow level (problem statement). Descriptions state the meaning; notes on definitions are comments.
    FieldSpec("flow.src_ip", Level.FLOW, Kind.IDENTIFIER, None, "Source IP address.", _PS,
              dtype=Dtype.ADDRESS, statuses=IDENTITY_STATUSES),                          # entity key
    FieldSpec("flow.dst_ip", Level.FLOW, Kind.IDENTIFIER, None, "Destination IP address.", _PS,
              dtype=Dtype.ADDRESS, statuses=IDENTITY_STATUSES),                          # entity key
    FieldSpec("flow.src_port", Level.FLOW, Kind.CATEGORICAL, None, "Source transport port.", _PS),
    FieldSpec("flow.dst_port", Level.FLOW, Kind.CATEGORICAL, None, "Destination transport port.", _PS),
    FieldSpec("flow.protocol", Level.FLOW, Kind.CATEGORICAL, None, "IP protocol number.", _PS),   # IANA numbers
    FieldSpec(                                                                   # OR over the flow, as NetFlow
        "flow.tcp_flags", Level.FLOW, Kind.BITMASK, None,
        "TCP flags seen in the flow (SYN, ACK, FIN, RST, PSH, URG).", _PS, codes=TCP_FLAG_BITS,
    ),
    *(
        FieldSpec(                                                               # per-flag counts, as CICFlowMeter
            f"flow.flag_count.{f}", Level.FLOW, Kind.COUNT, "packets",
            f"Packets in the flow with the {f.upper()} flag.", _PS,
        )
        for f in ("syn", "ack", "fin", "rst", "psh", "urg")
    ),
    FieldSpec("flow.bytes_fwd", Level.FLOW, Kind.COUNT, "bytes", "IP-layer bytes, initiator to responder.", _PS),
    FieldSpec("flow.bytes_bwd", Level.FLOW, Kind.COUNT, "bytes", "IP-layer bytes, responder to initiator.", _PS),
    FieldSpec("flow.packets_fwd", Level.FLOW, Kind.COUNT, "packets", "Packets, initiator to responder.", _PS),
    FieldSpec("flow.packets_bwd", Level.FLOW, Kind.COUNT, "packets", "Packets, responder to initiator.", _PS),
    FieldSpec("flow.duration", Level.FLOW, Kind.CONTINUOUS, "s", "Time from first to last packet of the flow.", _PS),
    FieldSpec("flow.iat_mean", Level.FLOW, Kind.CONTINUOUS, "s", "Mean inter-arrival time between packets of the flow.", _PS),
    FieldSpec("flow.iat_var", Level.FLOW, Kind.CONTINUOUS, "s^2", "Variance of inter-arrival times.", _PS),
    FieldSpec("flow.iat_max", Level.FLOW, Kind.CONTINUOUS, "s", "Largest inter-arrival time.", _PS),
    FieldSpec(                                                                   # definition recorded per adapter
        "flow.bidir_ratio", Level.FLOW, Kind.CONTINUOUS, None,
        "Bidirectional ratio: responder-to-initiator over initiator-to-responder IP-layer bytes.", _PS,
    ),
    # Sources define "bytes" differently (CICFlowMeter sums payload bytes, the PCAP adapter IP-layer
    # bytes), and Argus reports only total packets; separate fields keep each definition exact (AS-303).
    FieldSpec("flow.payload_bytes_fwd", Level.FLOW, Kind.COUNT, "bytes", "Transport payload bytes, initiator to responder.", _PS),
    FieldSpec("flow.payload_bytes_bwd", Level.FLOW, Kind.COUNT, "bytes", "Transport payload bytes, responder to initiator.", _PS),
    FieldSpec("flow.packets_total", Level.FLOW, Kind.COUNT, "packets", "Packets in both directions.", _PS),
    # D-53: how a flow ended is evidence (an unanswered SYN is a scan signal; silence after C2 traffic
    # matters). Codes 1 to 4: AS-337; codes 5 to 8 (version 0.2): AS-681.
    FieldSpec("flow.end_reason", Level.FLOW, Kind.CATEGORICAL, None, "Why the flow ended (see the code table).", "D-53",
              codes=END_REASON_CODES),
    FieldSpec("flow.unanswered", Level.FLOW, Kind.CATEGORICAL, None,
              "1 if the flow ended without the responder ever sending a packet, else 0.", "D-53", codes=YES_NO_CODES),
    # Packet level (problem statement).
    FieldSpec("pkt.ttl_mean", Level.PACKET, Kind.CONTINUOUS, "hops", "Mean IP TTL across the session.", _PP),
    FieldSpec("pkt.ttl_var", Level.PACKET, Kind.CONTINUOUS, "hops^2", "Variance of IP TTL across the session.", _PP),
    FieldSpec("pkt.tcp_window_init_fwd", Level.PACKET, Kind.COUNT, "bytes", "Initial TCP window, initiator.", _PP),
    FieldSpec("pkt.tcp_window_init_bwd", Level.PACKET, Kind.COUNT, "bytes", "Initial TCP window, responder.", _PP),
    FieldSpec("pkt.ip_df_count", Level.PACKET, Kind.COUNT, "packets", "Packets with the IP Don't-Fragment flag.", _PP),
    FieldSpec("pkt.ip_mf_count", Level.PACKET, Kind.COUNT, "packets", "Packets with the IP More-Fragments flag.", _PP),
    FieldSpec("pkt.payload_size_hist", Level.PACKET, Kind.HISTOGRAM, "bytes", "Payload-size distribution.", _PP),
    FieldSpec("pkt.retransmissions", Level.PACKET, Kind.COUNT, "segments", "TCP retransmissions.", _PP),
    FieldSpec("derived.portscan_sequential", Level.DERIVED, Kind.CONTINUOUS, None,
              "Evidence of sequential port access from one source.", _PP, layer=Layer.MACROSTATE),
    FieldSpec("derived.portscan_random", Level.DERIVED, Kind.CONTINUOUS, None,      # slow scans evade flow thresholds
              "Evidence of randomised port access from one source.", _PP, layer=Layer.MACROSTATE),
    # Observable-region examples (D-38).
    FieldSpec("proto.dns.qtype", Level.PROTOCOL, Kind.CATEGORICAL, None, "DNS query type.", _EX, example=True,
              codes=DNS_QTYPE_CODES),
    FieldSpec("proto.dns.rcode", Level.PROTOCOL, Kind.CATEGORICAL, None, "DNS response code.", _EX, example=True,
              codes=DNS_RCODE_CODES),
    FieldSpec("proto.kerberos.msg_type", Level.PROTOCOL, Kind.CATEGORICAL, None, "Kerberos message type.", _EX,
              example=True, codes=KRB_MSG_TYPE_CODES),
    FieldSpec("proto.tls.ja4", Level.PROTOCOL, Kind.FINGERPRINT, None, "JA4 client fingerprint.", _EX, example=True),
    # typically NOT_OBSERVABLE under Encrypted Client Hello: a depreciating observable
    FieldSpec("proto.tls.sni", Level.PROTOCOL, Kind.FINGERPRINT, None, "TLS server name.", _EX, example=True),
    # NOT_OBSERVABLE when encrypted; listed so that the status is explicit
    FieldSpec("app.payload", Level.PROTOCOL, Kind.IDENTIFIER, None, "Application payload (digest of the readable payload).",
              _EX, example=True, layer=Layer.EVIDENCE),
    FieldSpec("proto.icmp.type", Level.PROTOCOL, Kind.CATEGORICAL, None, "ICMP message type (8 = echo request).", _EX, example=True),
    FieldSpec("proto.arp.opcode", Level.PROTOCOL, Kind.CATEGORICAL, None, "ARP operation (1 = request, 2 = reply).", _EX, example=True),
    FieldSpec("ot.modbus.function_code", Level.OT, Kind.CATEGORICAL, None, "Modbus function code.", _EX, example=True,
              layer=Layer.OT_CII, codes=MODBUS_FUNCTION_CODES),
    FieldSpec("ot.modbus.unit_id", Level.OT, Kind.CATEGORICAL, None, "Modbus unit identifier.", _EX, example=True,
              layer=Layer.OT_CII),
    FieldSpec("ot.modbus.register_start", Level.OT, Kind.COUNT, None, "First register addressed.", _EX, example=True,
              layer=Layer.OT_CII),
    FieldSpec("ot.modbus.register_count", Level.OT, Kind.COUNT, None, "Number of registers addressed.", _EX, example=True,
              layer=Layer.OT_CII),
    FieldSpec("ot.dnp3.function_code", Level.OT, Kind.CATEGORICAL, None, "DNP3 application function code.", _EX,
              example=True, layer=Layer.OT_CII, codes=DNP3_FUNCTION_CODES),
    FieldSpec("ot.dnp3.object_group", Level.OT, Kind.CATEGORICAL, None, "DNP3 object group.", _EX, example=True,
              layer=Layer.OT_CII),
    FieldSpec("ot.iec104.type_id", Level.OT, Kind.CATEGORICAL, None, "IEC 60870-5-104 ASDU type identifier.", _EX,
              example=True, layer=Layer.OT_CII),
    FieldSpec("ot.iec104.cot", Level.OT, Kind.CATEGORICAL, None, "IEC 60870-5-104 cause of transmission.", _EX,
              example=True, layer=Layer.OT_CII),

    # Version 0.2: shared fields of the superset formats. Matrix kinds first (each takes an input
    # column, appended to `data.windows.COLUMN_SLOTS` in this order), then identifiers, fingerprints and
    # attributes.
    _s("flow.conn_state", Level.FLOW, Kind.CATEGORICAL, None, "Connection state as summarised by Zeek.", _FLOW_SRC,
       codes=CONN_STATE_CODES),
    _s("flow.missed_bytes", Level.FLOW, Kind.COUNT, "bytes", "Bytes the sensor missed in content gaps of the flow.",
       _FLOW_SRC, layer=Layer.EVIDENCE),
    _s("flow.vlan", Level.FLOW, Kind.CATEGORICAL, None, "Outer 802.1Q VLAN identifier of the flow.", _FLOW_SRC),
    _s("flow.ip_tos", Level.FLOW, Kind.CATEGORICAL, None, "IP type-of-service byte (DSCP and ECN), initiator to responder.",
       _FLOW_SRC),
    _s("flow.sampling_rate", Level.FLOW, Kind.COUNT, "packets",
       "Packet sampling interval of the exporter: one packet in N was metered (1 = unsampled).", _FLOW_SRC,
       layer=Layer.EVIDENCE),
    _s("flow.forwarding_status", Level.FLOW, Kind.CATEGORICAL, None,
       "Forwarding status of the flow's packets at the exporter (RFC 7270 byte).", _FLOW_SRC, codes=FORWARDING_STATUS_CODES),
    _s("flow.tcp_flags_fwd", Level.FLOW, Kind.BITMASK, None, "TCP flags sent by the initiator (all eight bits).",
       _FLOW_SRC, codes=TCP_FLAG_BITS),
    _s("flow.tcp_flags_bwd", Level.FLOW, Kind.BITMASK, None, "TCP flags sent by the responder (all eight bits).",
       _FLOW_SRC, codes=TCP_FLAG_BITS),
    _s("flow.app_proto", Level.FLOW, Kind.CATEGORICAL, None, "Application protocol identified by the source's analysers.",
       _FLOW_SRC, codes=APP_PROTO_CODES),
    _s("pkt.ttl_min", Level.PACKET, Kind.COUNT, "hops", "Smallest IP TTL or hop limit seen in the flow.", _FLOW_SRC),
    _s("pkt.ttl_max", Level.PACKET, Kind.COUNT, "hops", "Largest IP TTL or hop limit seen in the flow.", _FLOW_SRC),
    _s("pkt.ip_len_min", Level.PACKET, Kind.COUNT, "bytes", "Smallest IP total length seen in the flow.", _FLOW_SRC),
    _s("pkt.ip_len_max", Level.PACKET, Kind.COUNT, "bytes", "Largest IP total length seen in the flow.", _FLOW_SRC),
    _s("proto.icmp.code", Level.PROTOCOL, Kind.CATEGORICAL, None, "ICMP message code.", _FLOW_SRC),
    _s("dev.interval", Level.DEVICE, Kind.CONTINUOUS, "s", "Length of the interval the counter deltas of this update cover.",
       _DEV_SRC),
    _s("dev.if_in_octets", Level.DEVICE, Kind.COUNT, "bytes", "Octets received on the interface during the interval.", _DEV_SRC),
    _s("dev.if_out_octets", Level.DEVICE, Kind.COUNT, "bytes", "Octets sent on the interface during the interval.", _DEV_SRC),
    _s("dev.if_in_packets", Level.DEVICE, Kind.COUNT, "packets",
       "Packets received on the interface during the interval (unicast, multicast and broadcast).", _DEV_SRC),
    _s("dev.if_out_packets", Level.DEVICE, Kind.COUNT, "packets",
       "Packets sent on the interface during the interval (unicast, multicast and broadcast).", _DEV_SRC),
    _s("dev.if_errors", Level.DEVICE, Kind.COUNT, "packets", "Packets in error on the interface during the interval (in + out).",
       _DEV_SRC),
    _s("dev.if_discards", Level.DEVICE, Kind.COUNT, "packets", "Packets discarded by the interface during the interval (in + out).",
       _DEV_SRC),
    _s("dev.if_status", Level.DEVICE, Kind.BITMASK, None, "Interface status (bit 0 admin up, bit 1 operational up).",
       _DEV_SRC, statuses=STATE_FACT_STATUSES, codes=IF_STATUS_BITS),
    _s("dev.capture_received", Level.DEVICE, Kind.COUNT, "packets", "Packets the sensor received during the interval.",
       _DEV_SRC, layer=Layer.EVIDENCE),
    _s("dev.capture_dropped", Level.DEVICE, Kind.COUNT, "packets",
       "Packets the sensor dropped during the interval (lost to the observation).", _DEV_SRC, layer=Layer.EVIDENCE),
    _s("proto.dns.qclass", Level.PROTOCOL, Kind.CATEGORICAL, None, "DNS query class.", _DNS_SRC, codes=DNS_QCLASS_CODES),
    _s("proto.dns.flags", Level.PROTOCOL, Kind.BITMASK, None, "DNS header flags (AA, TC, RD, RA, Z, AD, CD).", _DNS_SRC,
       codes=DNS_FLAG_BITS),
    _s("proto.dns.rtt", Level.PROTOCOL, Kind.CONTINUOUS, "s", "Time from the DNS query to its response.", _DNS_SRC),
    _s("proto.dns.answer_count", Level.PROTOCOL, Kind.COUNT, "records", "Answer records in the DNS response.", _DNS_SRC),
    _s("proto.dns.ttl_min", Level.PROTOCOL, Kind.CONTINUOUS, "s", "Smallest TTL among the DNS answers.", _DNS_SRC),
    _s("proto.dns.query_length", Level.PROTOCOL, Kind.COUNT, "characters", "Length of the queried name.", _DNS_SRC),
    _s("proto.dns.query_entropy", Level.PROTOCOL, Kind.CONTINUOUS, "bits",
       "Shannon entropy per character of the queried name.", _DNS_SRC),
    _s("proto.http.method", Level.PROTOCOL, Kind.CATEGORICAL, None, "HTTP request method.", _HTTP_SRC, codes=HTTP_METHOD_CODES),
    _s("proto.http.status_code", Level.PROTOCOL, Kind.CATEGORICAL, None, "HTTP response status code.", _HTTP_SRC),
    _s("proto.http.request_body_len", Level.PROTOCOL, Kind.COUNT, "bytes", "Uncompressed HTTP request body length.", _HTTP_SRC),
    _s("proto.http.response_body_len", Level.PROTOCOL, Kind.COUNT, "bytes", "Uncompressed HTTP response body length.", _HTTP_SRC),
    _s("proto.http.uri_length", Level.PROTOCOL, Kind.COUNT, "characters", "Length of the request URI.", _HTTP_SRC),
    _s("proto.tls.version", Level.PROTOCOL, Kind.CATEGORICAL, None, "Negotiated TLS or DTLS version (wire code).", _TLS_SRC,
       codes=TLS_VERSION_CODES),
    _s("proto.tls.established", Level.PROTOCOL, Kind.CATEGORICAL, None, "1 if the TLS handshake completed, else 0.", _TLS_SRC,
       codes=YES_NO_CODES),
    _s("proto.tls.alert", Level.PROTOCOL, Kind.CATEGORICAL, None, "Last TLS alert description seen.", _TLS_SRC,
       codes=TLS_ALERT_CODES),
    _s("proto.tls.cert_validity", Level.PROTOCOL, Kind.CONTINUOUS, "s",
       "Validity period of the server certificate (not after minus not before).", _TLS_SRC, statuses=STATE_FACT_STATUSES),
    _s("proto.tls.cert_self_signed", Level.PROTOCOL, Kind.CATEGORICAL, None,
       "1 if the server certificate's subject equals its issuer, else 0.", _TLS_SRC, statuses=STATE_FACT_STATUSES,
       codes=YES_NO_CODES),
    _s("proto.ssh.auth_success", Level.PROTOCOL, Kind.CATEGORICAL, None, "1 if SSH authentication succeeded, 0 if it failed.",
       "D-38: Zeek ssh, OCSF SSH Activity", codes=YES_NO_CODES),
    _s("proto.ssh.auth_attempts", Level.PROTOCOL, Kind.COUNT, "attempts", "SSH authentication attempts seen.",
       "D-38: Zeek ssh, OCSF SSH Activity"),
    _s("proto.kerberos.error_code", Level.PROTOCOL, Kind.CATEGORICAL, None, "Kerberos error code (0 = no error).",
       "D-38: Zeek kerberos, Suricata krb5, Windows events 4768, 4769, 4771", codes=KRB_ERROR_CODES),
    _s("proto.kerberos.etype", Level.PROTOCOL, Kind.CATEGORICAL, None, "Kerberos encryption type of the ticket.",
       "D-38: Zeek kerberos, Suricata krb5, Windows events 4768, 4769", codes=KRB_ETYPE_CODES),
    _s("auth.activity", Level.AUTH, Kind.CATEGORICAL, None, "Authentication activity.", _AUTH_SRC, codes=AUTH_ACTIVITY_CODES),
    _s("auth.result", Level.AUTH, Kind.CATEGORICAL, None, "Authentication outcome.", _AUTH_SRC, codes=AUTH_RESULT_CODES),
    _s("auth.logon_type", Level.AUTH, Kind.CATEGORICAL, None, "Logon type.", _AUTH_SRC, codes=LOGON_TYPE_CODES),
    _s("auth.protocol", Level.AUTH, Kind.CATEGORICAL, None, "Authentication protocol.", _AUTH_SRC, codes=AUTH_PROTOCOL_CODES),
    _s("auth.method", Level.AUTH, Kind.CATEGORICAL, None, "Credential type presented.", _AUTH_SRC, codes=AUTH_METHOD_CODES),
    _s("auth.failure_status", Level.AUTH, Kind.CATEGORICAL, None,
       "Most specific failure status code (Windows NTSTATUS, unsigned 32-bit).", _AUTH_SRC),
    _s("auth.elevated", Level.AUTH, Kind.CATEGORICAL, None, "1 if the session holds elevated privileges, else 0.", _AUTH_SRC,
       codes=YES_NO_CODES),
    _s("alert.severity", Level.ALERT, Kind.CATEGORICAL, None, "Alert severity on the OCSF severity scale.", _ALERT_SRC,
       codes=SEVERITY_CODES),
    _s("alert.signature", Level.ALERT, Kind.CATEGORICAL, None,
       "Identity of the detection rule: generator id * 2^32 + signature id (AS-687).", _ALERT_SRC),
    _s("alert.category", Level.ALERT, Kind.CATEGORICAL, None, "Rule classification.", _ALERT_SRC, codes=ALERT_CATEGORY_CODES),
    _s("event.action", Level.EVENT, Kind.CATEGORICAL, None, "Action the reporting device took on the activity.",
       _EVT_SRC, codes=EVENT_ACTION_CODES),
    _s("ot.modbus.exception_code", Level.OT, Kind.CATEGORICAL, None, "Modbus exception code of the response.",
       "D-34, D-38: Zeek modbus, Suricata modbus", layer=Layer.OT_CII, codes=MODBUS_EXCEPTION_CODES),
    _s("ot.dnp3.iin", Level.OT, Kind.BITMASK, None, "DNP3 internal indications of the response, (IIN1 << 8) | IIN2.",
       "D-34, D-38: Zeek dnp3, Suricata dnp3", layer=Layer.OT_CII, codes=DNP3_IIN_BITS),
    _s("proto.mqtt.message_type", Level.PROTOCOL, Kind.CATEGORICAL, None, "MQTT control packet type.",
       "D-38: Suricata mqtt", codes=MQTT_TYPE_CODES),
    _s("proto.smb.command", Level.PROTOCOL, Kind.CATEGORICAL, None, "SMB command.", "D-38: Suricata smb, tshark",
       codes=SMB_COMMAND_CODES),
    _s("proto.smb.file_action", Level.PROTOCOL, Kind.CATEGORICAL, None, "File action performed over SMB.",
       "D-38: Zeek smb_files", codes=SMB_FILE_ACTION_CODES),
    _s("proto.dhcp.message_type", Level.PROTOCOL, Kind.CATEGORICAL, None, "DHCP message type (the last of the exchange).",
       "D-38: Zeek dhcp, Suricata dhcp, OCSF DHCP Activity", codes=DHCP_MESSAGE_CODES),
    _s("file.size", Level.PROTOCOL, Kind.COUNT, "bytes", "Size of a transferred file.",
       "D-38: Zeek files and smb_files, Suricata fileinfo, CEF fsize, OCSF file"),
    _s("proto.ntp.mode", Level.PROTOCOL, Kind.CATEGORICAL, None, "NTP association mode.", "D-38: OCSF NTP Activity, tshark",
       codes=NTP_MODE_CODES),

    # Version 0.2 shared identifiers, fingerprints and attributes (not input columns).
    _s("flow.uid", Level.FLOW, Kind.IDENTIFIER, None, "The source's identifier of the flow (Zeek uid, Suricata flow_id).",
       _FLOW_SRC, layer=Layer.EVENT),
    _s("flow.community_id", Level.FLOW, Kind.IDENTIFIER, None, "Community ID flow hash (version 1).", _FLOW_SRC,
       layer=Layer.EVENT),
    _s("flow.src_mac", Level.FLOW, Kind.IDENTIFIER, None, "Link-layer source address.", _FLOW_SRC, dtype=Dtype.MAC),
    _s("flow.dst_mac", Level.FLOW, Kind.IDENTIFIER, None, "Link-layer destination address.", _FLOW_SRC, dtype=Dtype.MAC),
    _s("flow.src_hostname", Level.FLOW, Kind.IDENTIFIER, None, "Host name of the initiator as reported by the source.",
       _EVT_SRC),
    _s("flow.dst_hostname", Level.FLOW, Kind.IDENTIFIER, None, "Host name of the responder as reported by the source.",
       _EVT_SRC),
    _s("flow.start_time", Level.FLOW, Kind.ATTRIBUTE, "s", "Time of the flow's first packet (epoch seconds, UTC).",
       _FLOW_SRC, layer=Layer.EVENT, dtype=Dtype.TIME),
    _s("flow.end_time", Level.FLOW, Kind.ATTRIBUTE, "s", "Time of the flow's last packet (epoch seconds, UTC).",
       _FLOW_SRC, layer=Layer.EVENT, dtype=Dtype.TIME),
    _s("flow.vlan_inner", Level.FLOW, Kind.ATTRIBUTE, None, "Inner 802.1Q VLAN identifier (QinQ).", _FLOW_SRC,
       dtype=Dtype.INT),
    _s("flow.ip_tos_bwd", Level.FLOW, Kind.ATTRIBUTE, None, "IP type-of-service byte, responder to initiator.",
       _FLOW_SRC, dtype=Dtype.INT),
    _s("flow.ip_version", Level.FLOW, Kind.ATTRIBUTE, None, "IP version of the flow (4 or 6).", _FLOW_SRC, dtype=Dtype.INT),
    _s("flow.ingress_if", Level.FLOW, Kind.ATTRIBUTE, None, "Exporter interface index the flow entered on.", _FLOW_SRC,
       dtype=Dtype.INT),
    _s("flow.egress_if", Level.FLOW, Kind.ATTRIBUTE, None, "Exporter interface index the flow left on.", _FLOW_SRC,
       dtype=Dtype.INT),
    _s("flow.next_hop", Level.FLOW, Kind.IDENTIFIER, None, "Next-hop router address of the flow.", _FLOW_SRC,
       dtype=Dtype.ADDRESS),
    _s("flow.src_as", Level.FLOW, Kind.ATTRIBUTE, None, "BGP source autonomous system number.", _FLOW_SRC, dtype=Dtype.INT),
    _s("flow.dst_as", Level.FLOW, Kind.ATTRIBUTE, None, "BGP destination autonomous system number.", _FLOW_SRC,
       dtype=Dtype.INT),
    _s("flow.frame_bytes_fwd", Level.FLOW, Kind.ATTRIBUTE, "bytes",
       "Bytes as captured including the link-layer header, initiator to responder.", _FLOW_SRC, dtype=Dtype.INT),
    _s("flow.frame_bytes_bwd", Level.FLOW, Kind.ATTRIBUTE, "bytes",
       "Bytes as captured including the link-layer header, responder to initiator.", _FLOW_SRC, dtype=Dtype.INT),
    _s("flow.app_proto_text", Level.FLOW, Kind.FINGERPRINT, None, "Application-protocol label as written by the source.",
       _FLOW_SRC, layer=Layer.EVENT),
    _s("proto.dns.query", Level.PROTOCOL, Kind.IDENTIFIER, None, "Queried domain name.", _DNS_SRC),
    _s("proto.dns.answers", Level.PROTOCOL, Kind.ATTRIBUTE, None, "Resource data of the DNS answers, in order.", _DNS_SRC,
       layer=Layer.EVENT, dtype=Dtype.STR_LIST),
    _s("proto.dns.ttls", Level.PROTOCOL, Kind.ATTRIBUTE, "s", "TTLs of the DNS answers, in order.", _DNS_SRC,
       layer=Layer.EVENT, dtype=Dtype.FLOAT_LIST),
    _s("proto.dns.trans_id", Level.PROTOCOL, Kind.ATTRIBUTE, None, "DNS transaction identifier.", _DNS_SRC,
       layer=Layer.EVENT, dtype=Dtype.INT),
    _s("proto.dns.opcode", Level.PROTOCOL, Kind.ATTRIBUTE, None, "DNS opcode.", _DNS_SRC, layer=Layer.EVENT, dtype=Dtype.INT),
    _s("proto.dns.rejected", Level.PROTOCOL, Kind.ATTRIBUTE, None, "True if the server rejected the query.", _DNS_SRC,
       layer=Layer.EVENT, dtype=Dtype.BOOL),
    _s("proto.http.method_text", Level.PROTOCOL, Kind.FINGERPRINT, None, "HTTP method as written on the wire.", _HTTP_SRC,
       layer=Layer.EVENT),
    _s("proto.http.host", Level.PROTOCOL, Kind.IDENTIFIER, None, "HTTP Host header.", _HTTP_SRC),
    _s("proto.http.uri", Level.PROTOCOL, Kind.IDENTIFIER, None, "HTTP request URI (path and query).", _HTTP_SRC,
       layer=Layer.EVENT),
    _s("proto.http.referrer", Level.PROTOCOL, Kind.IDENTIFIER, None, "HTTP Referer header.", _HTTP_SRC, layer=Layer.EVENT),
    _s("proto.http.user_agent", Level.PROTOCOL, Kind.FINGERPRINT, None, "HTTP User-Agent header.", _HTTP_SRC),
    _s("proto.http.version", Level.PROTOCOL, Kind.ATTRIBUTE, None, "HTTP version string.", _HTTP_SRC, layer=Layer.EVENT,
       dtype=Dtype.STR),
    _s("proto.http.status_msg", Level.PROTOCOL, Kind.ATTRIBUTE, None, "HTTP reason phrase.", _HTTP_SRC, layer=Layer.EVENT,
       dtype=Dtype.STR),
    _s("proto.http.content_type", Level.PROTOCOL, Kind.FINGERPRINT, None, "Content type of the HTTP response.", _HTTP_SRC,
       layer=Layer.EVENT),
    _s("proto.http.username", Level.PROTOCOL, Kind.IDENTIFIER, None, "User name in HTTP basic authentication.", _HTTP_SRC,
       layer=Layer.EVENT),
    _s("proto.tls.version_text", Level.PROTOCOL, Kind.ATTRIBUTE, None, "TLS version as written by the source.", _TLS_SRC,
       layer=Layer.EVENT, dtype=Dtype.STR),
    _s("proto.tls.cipher", Level.PROTOCOL, Kind.FINGERPRINT, None, "Negotiated TLS cipher suite.", _TLS_SRC),
    _s("proto.tls.curve", Level.PROTOCOL, Kind.FINGERPRINT, None, "Key-exchange group of the TLS session.", _TLS_SRC),
    _s("proto.tls.resumed", Level.PROTOCOL, Kind.ATTRIBUTE, None, "True if the TLS session was resumed.", _TLS_SRC,
       layer=Layer.EVENT, dtype=Dtype.BOOL),
    _s("proto.tls.alpn", Level.PROTOCOL, Kind.FINGERPRINT, None, "Negotiated application protocol (ALPN or NPN).", _TLS_SRC),
    _s("proto.tls.ja3", Level.PROTOCOL, Kind.FINGERPRINT, None, "JA3 client fingerprint (MD5 hex).", _TLS_SRC),
    _s("proto.tls.ja3s", Level.PROTOCOL, Kind.FINGERPRINT, None, "JA3S server fingerprint (MD5 hex).", _TLS_SRC),
    _s("proto.tls.ja4s", Level.PROTOCOL, Kind.FINGERPRINT, None, "JA4S server fingerprint.", _TLS_SRC),
    _s("proto.tls.cert_subject", Level.PROTOCOL, Kind.IDENTIFIER, None, "Subject of the server certificate.", _TLS_SRC,
       layer=Layer.EVENT),
    _s("proto.tls.cert_issuer", Level.PROTOCOL, Kind.IDENTIFIER, None, "Issuer of the server certificate.", _TLS_SRC,
       layer=Layer.EVENT),
    _s("proto.tls.cert_serial", Level.PROTOCOL, Kind.IDENTIFIER, None, "Serial number of the server certificate.", _TLS_SRC,
       layer=Layer.EVENT),
    _s("proto.tls.cert_fingerprint", Level.PROTOCOL, Kind.IDENTIFIER, None, "Fingerprint of the server certificate.",
       _TLS_SRC, layer=Layer.EVENT),
    _s("proto.tls.cert_not_before", Level.PROTOCOL, Kind.ATTRIBUTE, "s", "Start of the certificate's validity (epoch s).",
       _TLS_SRC, layer=Layer.EVENT, dtype=Dtype.TIME, statuses=STATE_FACT_STATUSES),
    _s("proto.tls.cert_not_after", Level.PROTOCOL, Kind.ATTRIBUTE, "s", "End of the certificate's validity (epoch s).",
       _TLS_SRC, layer=Layer.EVENT, dtype=Dtype.TIME, statuses=STATE_FACT_STATUSES),
    _s("proto.tls.cert_key_length", Level.PROTOCOL, Kind.ATTRIBUTE, "bits", "Key length of the server certificate.",
       _TLS_SRC, layer=Layer.EVENT, dtype=Dtype.INT, statuses=STATE_FACT_STATUSES),
    _s("proto.ssh.version", Level.PROTOCOL, Kind.ATTRIBUTE, None, "SSH protocol major version.",
       "D-38: Zeek ssh, Suricata ssh, OCSF SSH Activity", layer=Layer.EVENT, dtype=Dtype.INT),
    _s("proto.ssh.client", Level.PROTOCOL, Kind.FINGERPRINT, None, "SSH client software banner.",
       "D-38: Zeek ssh, Suricata ssh"),
    _s("proto.ssh.server", Level.PROTOCOL, Kind.FINGERPRINT, None, "SSH server software banner.",
       "D-38: Zeek ssh, Suricata ssh"),
    _s("proto.ssh.hassh", Level.PROTOCOL, Kind.FINGERPRINT, None, "HASSH client fingerprint.", "D-38: Zeek ssh, Suricata ssh"),
    _s("proto.ssh.hassh_server", Level.PROTOCOL, Kind.FINGERPRINT, None, "HASSH server fingerprint.",
       "D-38: Zeek ssh, Suricata ssh"),
    _s("proto.ssh.host_key", Level.PROTOCOL, Kind.IDENTIFIER, None, "Fingerprint of the SSH server host key.",
       "D-38: Zeek ssh", layer=Layer.EVENT),
    _s("proto.kerberos.client", Level.PROTOCOL, Kind.IDENTIFIER, None, "Kerberos client principal.",
       "D-38: Zeek kerberos, Suricata krb5"),
    _s("proto.kerberos.service", Level.PROTOCOL, Kind.IDENTIFIER, None, "Kerberos service principal.",
       "D-38: Zeek kerberos, Suricata krb5, Windows events 4768, 4769"),
    _s("proto.kerberos.realm", Level.PROTOCOL, Kind.IDENTIFIER, None, "Kerberos realm.", "D-38: Suricata krb5"),
    _s("proto.kerberos.success", Level.PROTOCOL, Kind.ATTRIBUTE, None, "True if the Kerberos request succeeded.",
       "D-38: Zeek kerberos", layer=Layer.EVENT, dtype=Dtype.BOOL),
    _s("proto.kerberos.ticket_options", Level.PROTOCOL, Kind.ATTRIBUTE, None, "Kerberos KDC options bit field.",
       "D-38: Windows events 4768, 4769, 4771", layer=Layer.EVENT, dtype=Dtype.INT),
    _s("proto.smb.path", Level.PROTOCOL, Kind.IDENTIFIER, None, "SMB tree path (share).", "D-38: Zeek smb, Suricata smb"),
    _s("proto.smb.share_type", Level.PROTOCOL, Kind.ATTRIBUTE, None, "SMB share type (disk, pipe, print).",
       "D-38: Zeek smb_mapping, Suricata smb", layer=Layer.EVENT, dtype=Dtype.STR),
    _s("proto.smb.status", Level.PROTOCOL, Kind.ATTRIBUTE, None, "NTSTATUS of the SMB response (unsigned 32-bit).",
       "D-38: Suricata smb", layer=Layer.EVENT, dtype=Dtype.INT),
    _s("proto.smb.dialect", Level.PROTOCOL, Kind.ATTRIBUTE, None, "Negotiated SMB dialect.", "D-38: Suricata smb",
       layer=Layer.EVENT, dtype=Dtype.STR),
    _s("proto.dhcp.lease_time", Level.PROTOCOL, Kind.ATTRIBUTE, "s", "Lease duration granted.",
       "D-38: Zeek dhcp, Suricata dhcp, OCSF DHCP Activity", layer=Layer.EVENT, dtype=Dtype.FLOAT),
    _s("proto.dhcp.client_mac", Level.PROTOCOL, Kind.IDENTIFIER, None, "Client hardware address.",
       "D-38: Zeek dhcp, Suricata dhcp", dtype=Dtype.MAC),
    _s("proto.dhcp.assigned_addr", Level.PROTOCOL, Kind.IDENTIFIER, None, "Address assigned to the client.",
       "D-38: Zeek dhcp, Suricata dhcp", dtype=Dtype.ADDRESS),
    _s("proto.dhcp.requested_addr", Level.PROTOCOL, Kind.IDENTIFIER, None, "Address requested by the client.",
       "D-38: Zeek dhcp, Suricata dhcp", dtype=Dtype.ADDRESS),
    _s("proto.dhcp.hostname", Level.PROTOCOL, Kind.IDENTIFIER, None, "Host name the client sent.",
       "D-38: Zeek dhcp, Suricata dhcp"),
    _s("proto.dhcp.domain", Level.PROTOCOL, Kind.IDENTIFIER, None, "Domain name given to the client.", "D-38: Zeek dhcp"),
    _s("proto.rdp.cookie", Level.PROTOCOL, Kind.IDENTIFIER, None, "RDP cookie (often the user name).", "D-38: Zeek rdp"),
    _s("proto.rdp.security_protocol", Level.PROTOCOL, Kind.ATTRIBUTE, None, "RDP security protocol selected.",
       "D-38: Zeek rdp", layer=Layer.EVENT, dtype=Dtype.STR),
    _s("proto.rdp.result", Level.PROTOCOL, Kind.ATTRIBUTE, None, "Result of the RDP connection.", "D-38: Zeek rdp",
       layer=Layer.EVENT, dtype=Dtype.STR),
    _s("proto.rdp.client_name", Level.PROTOCOL, Kind.IDENTIFIER, None, "Name the RDP client reported.", "D-38: Zeek rdp"),
    _s("proto.rdp.client_build", Level.PROTOCOL, Kind.ATTRIBUTE, None, "RDP client build.", "D-38: Zeek rdp",
       layer=Layer.EVENT, dtype=Dtype.STR),
    _s("proto.rdp.keyboard_layout", Level.PROTOCOL, Kind.ATTRIBUTE, None, "Keyboard layout of the RDP client.",
       "D-38: Zeek rdp", layer=Layer.EVENT, dtype=Dtype.STR),
    _s("proto.rdp.desktop_width", Level.PROTOCOL, Kind.ATTRIBUTE, "pixels", "Desktop width requested by the RDP client.",
       "D-38: Zeek rdp", layer=Layer.EVENT, dtype=Dtype.INT),
    _s("proto.rdp.desktop_height", Level.PROTOCOL, Kind.ATTRIBUTE, "pixels", "Desktop height requested by the RDP client.",
       "D-38: Zeek rdp", layer=Layer.EVENT, dtype=Dtype.INT),
    _s("proto.smtp.helo", Level.PROTOCOL, Kind.IDENTIFIER, None, "SMTP HELO or EHLO argument.", "D-38: Zeek smtp, OCSF Email"),
    _s("proto.smtp.mailfrom", Level.PROTOCOL, Kind.IDENTIFIER, None, "SMTP envelope sender (MAIL FROM).",
       "D-38: Zeek smtp, OCSF Email"),
    _s("proto.smtp.rcptto", Level.PROTOCOL, Kind.ATTRIBUTE, None, "SMTP envelope recipients (RCPT TO).",
       "D-38: Zeek smtp, OCSF Email", layer=Layer.EVENT, dtype=Dtype.STR_LIST),
    _s("proto.smtp.from", Level.PROTOCOL, Kind.IDENTIFIER, None, "From header of the message.", "D-38: Zeek smtp, OCSF Email",
       layer=Layer.EVENT),
    _s("proto.smtp.to", Level.PROTOCOL, Kind.ATTRIBUTE, None, "To header recipients.", "D-38: Zeek smtp, OCSF Email",
       layer=Layer.EVENT, dtype=Dtype.STR_LIST),
    _s("proto.smtp.cc", Level.PROTOCOL, Kind.ATTRIBUTE, None, "Cc header recipients.", "D-38: Zeek smtp, OCSF Email",
       layer=Layer.EVENT, dtype=Dtype.STR_LIST),
    _s("proto.smtp.reply_to", Level.PROTOCOL, Kind.IDENTIFIER, None, "Reply-To header.", "D-38: Zeek smtp, OCSF Email",
       layer=Layer.EVENT),
    _s("proto.smtp.msg_id", Level.PROTOCOL, Kind.IDENTIFIER, None, "Message-ID header.", "D-38: Zeek smtp, OCSF Email",
       layer=Layer.EVENT),
    _s("proto.smtp.subject", Level.PROTOCOL, Kind.ATTRIBUTE, None, "Subject header.", "D-38: Zeek smtp, OCSF Email",
       layer=Layer.EVENT, dtype=Dtype.STR),
    _s("proto.smtp.x_originating_ip", Level.PROTOCOL, Kind.IDENTIFIER, None, "X-Originating-IP header.",
       "D-38: Zeek smtp, OCSF Email", layer=Layer.EVENT, dtype=Dtype.ADDRESS),
    _s("proto.ftp.command", Level.PROTOCOL, Kind.FINGERPRINT, None, "FTP command.", "D-38: OCSF FTP Activity, tshark"),
    _s("proto.ftp.reply_code", Level.PROTOCOL, Kind.ATTRIBUTE, None, "FTP reply code.", "D-38: OCSF FTP Activity, tshark",
       layer=Layer.EVENT, dtype=Dtype.INT),
    _s("proto.ftp.argument", Level.PROTOCOL, Kind.IDENTIFIER, None, "Argument of the FTP command.",
       "D-38: OCSF FTP Activity, tshark", layer=Layer.EVENT),
    _s("proto.ntp.stratum", Level.PROTOCOL, Kind.ATTRIBUTE, None, "NTP stratum.", "D-38: OCSF NTP Activity, tshark",
       layer=Layer.EVENT, dtype=Dtype.INT),
    _s("proto.ntp.version", Level.PROTOCOL, Kind.ATTRIBUTE, None, "NTP version.", "D-38: OCSF NTP Activity, tshark",
       layer=Layer.EVENT, dtype=Dtype.INT),
    _s("proto.mqtt.topic", Level.PROTOCOL, Kind.IDENTIFIER, None, "MQTT topic.", "D-38: Suricata mqtt"),
    _s("proto.mqtt.qos", Level.PROTOCOL, Kind.ATTRIBUTE, None, "MQTT quality of service.", "D-38: Suricata mqtt",
       layer=Layer.EVENT, dtype=Dtype.INT),
    _s("proto.mqtt.client_id", Level.PROTOCOL, Kind.IDENTIFIER, None, "MQTT client identifier.", "D-38: Suricata mqtt"),
    _s("proto.mqtt.return_code", Level.PROTOCOL, Kind.ATTRIBUTE, None, "MQTT CONNACK return or reason code.",
       "D-38: Suricata mqtt", layer=Layer.EVENT, dtype=Dtype.INT),
    _s("proto.dcerpc.endpoint", Level.PROTOCOL, Kind.FINGERPRINT, None, "DCE/RPC endpoint (interface) name.",
       "D-38: Zeek dce_rpc, Suricata smb dcerpc"),
    _s("proto.dcerpc.operation", Level.PROTOCOL, Kind.FINGERPRINT, None, "DCE/RPC operation name.", "D-38: Zeek dce_rpc"),
    _s("proto.dcerpc.named_pipe", Level.PROTOCOL, Kind.IDENTIFIER, None, "Named pipe carrying the DCE/RPC call.",
       "D-38: Zeek dce_rpc"),
    _s("proto.dcerpc.rtt", Level.PROTOCOL, Kind.ATTRIBUTE, "s", "Round-trip time of the DCE/RPC call.", "D-38: Zeek dce_rpc",
       layer=Layer.EVENT, dtype=Dtype.FLOAT),
    _s("file.uid", Level.PROTOCOL, Kind.IDENTIFIER, None, "The source's identifier of the file.",
       "D-38: Zeek files, Suricata fileinfo", layer=Layer.EVENT),
    _s("file.name", Level.PROTOCOL, Kind.IDENTIFIER, None, "File name.", "D-38: Zeek files and smb_files, Suricata fileinfo, CEF"),
    _s("file.mime_type", Level.PROTOCOL, Kind.FINGERPRINT, None, "File MIME type.", "D-38: Zeek files, Suricata fileinfo"),
    _s("file.md5", Level.PROTOCOL, Kind.IDENTIFIER, None, "MD5 of the file (hex).", "D-38: Zeek files, Suricata fileinfo",
       layer=Layer.EVENT),
    _s("file.sha1", Level.PROTOCOL, Kind.IDENTIFIER, None, "SHA-1 of the file (hex).", "D-38: Zeek files, Suricata fileinfo",
       layer=Layer.EVENT),
    _s("file.sha256", Level.PROTOCOL, Kind.IDENTIFIER, None, "SHA-256 of the file (hex).",
       "D-38: Zeek files, Suricata fileinfo, CEF fileHash", layer=Layer.EVENT),
    _s("file.source", Level.PROTOCOL, Kind.ATTRIBUTE, None, "Protocol the file was carried by.", "D-38: Zeek files",
       layer=Layer.EVENT, dtype=Dtype.STR),
    _s("auth.user", Level.AUTH, Kind.IDENTIFIER, None, "Account that authenticated or was targeted.", _AUTH_SRC),
    _s("auth.domain", Level.AUTH, Kind.IDENTIFIER, None, "Domain, realm or host scope of `auth.user`.", _AUTH_SRC),
    _s("auth.user_sid", Level.AUTH, Kind.IDENTIFIER, None, "Security identifier of `auth.user`.", _AUTH_SRC,
       layer=Layer.EVENT),
    _s("auth.logon_id", Level.AUTH, Kind.IDENTIFIER, None, "Session (logon) identifier.", _AUTH_SRC, layer=Layer.EVENT),
    _s("auth.subject_user", Level.AUTH, Kind.IDENTIFIER, None, "Account that requested the operation.", _AUTH_SRC),
    _s("auth.subject_domain", Level.AUTH, Kind.IDENTIFIER, None, "Domain of `auth.subject_user`.", _AUTH_SRC),
    _s("auth.target_user", Level.AUTH, Kind.IDENTIFIER, None, "Account whose credentials or identity were used.", _AUTH_SRC),
    _s("auth.target_domain", Level.AUTH, Kind.IDENTIFIER, None, "Domain of `auth.target_user`.", _AUTH_SRC),
    _s("auth.workstation", Level.AUTH, Kind.IDENTIFIER, None, "Workstation name the authentication came from.", _AUTH_SRC),
    _s("auth.process_name", Level.AUTH, Kind.IDENTIFIER, None, "Process that initiated the logon.", _AUTH_SRC,
       layer=Layer.EVENT),
    _s("auth.service", Level.AUTH, Kind.IDENTIFIER, None, "Service or server the authentication was for.", _AUTH_SRC),
    _s("auth.logon_process", Level.AUTH, Kind.FINGERPRINT, None, "Trusted logon process.", _AUTH_SRC),
    _s("auth.package", Level.AUTH, Kind.FINGERPRINT, None, "Authentication package (Windows).", _AUTH_SRC),
    _s("auth.failure_reason", Level.AUTH, Kind.ATTRIBUTE, None, "Failure reason as written by the source.", _AUTH_SRC,
       layer=Layer.EVENT, dtype=Dtype.STR),
    _s("auth.privileges", Level.AUTH, Kind.ATTRIBUTE, None, "Privileges assigned to the session.", _AUTH_SRC,
       layer=Layer.EVENT, dtype=Dtype.STR_LIST),
    _s("auth.key_length", Level.AUTH, Kind.ATTRIBUTE, "bits", "Session key length (Windows logon).", _AUTH_SRC,
       layer=Layer.EVENT, dtype=Dtype.INT),
    _s("alert.signature_id", Level.ALERT, Kind.ATTRIBUTE, None, "Signature (rule) identifier.", _ALERT_SRC,
       layer=Layer.EVENT, dtype=Dtype.INT),
    _s("alert.generator_id", Level.ALERT, Kind.ATTRIBUTE, None, "Generator (detection engine component) identifier.",
       _ALERT_SRC, layer=Layer.EVENT, dtype=Dtype.INT),
    _s("alert.revision", Level.ALERT, Kind.ATTRIBUTE, None, "Rule revision.", _ALERT_SRC, layer=Layer.EVENT, dtype=Dtype.INT),
    _s("alert.message", Level.ALERT, Kind.ATTRIBUTE, None, "Alert message.", _ALERT_SRC, layer=Layer.EVENT, dtype=Dtype.STR),
    _s("alert.category_text", Level.ALERT, Kind.ATTRIBUTE, None, "Rule classification as written by the source.",
       _ALERT_SRC, layer=Layer.EVENT, dtype=Dtype.STR),
    _s("alert.priority", Level.ALERT, Kind.ATTRIBUTE, None, "Priority or severity on the source's own scale.",
       _ALERT_SRC, layer=Layer.EVENT, dtype=Dtype.INT),
    _s("alert.rule", Level.ALERT, Kind.ATTRIBUTE, None, "Rule or notice name, or rule text, as written by the source.",
       _ALERT_SRC, layer=Layer.EVENT, dtype=Dtype.STR),
    _s("event.message", Level.EVENT, Kind.ATTRIBUTE, None, "Free-text message of the event.", _EVT_SRC, layer=Layer.EVENT,
       dtype=Dtype.STR),
    _s("event.severity", Level.EVENT, Kind.ATTRIBUTE, None, "Event severity on the OCSF severity scale.", _EVT_SRC,
       layer=Layer.EVENT, dtype=Dtype.INT, codes=SEVERITY_CODES),
    _s("event.outcome", Level.EVENT, Kind.ATTRIBUTE, None, "Outcome as written by the source.", _EVT_SRC,
       layer=Layer.EVENT, dtype=Dtype.STR),
    _s("event.vendor", Level.EVENT, Kind.ATTRIBUTE, None, "Vendor of the reporting device.", _EVT_SRC, layer=Layer.EVENT,
       dtype=Dtype.STR),
    _s("event.product", Level.EVENT, Kind.ATTRIBUTE, None, "Product of the reporting device.", _EVT_SRC,
       layer=Layer.EVENT, dtype=Dtype.STR),
    _s("event.product_version", Level.EVENT, Kind.ATTRIBUTE, None, "Version of the reporting product.", _EVT_SRC,
       layer=Layer.EVENT, dtype=Dtype.STR),
    _s("event.class_id", Level.EVENT, Kind.ATTRIBUTE, None, "Event class identifier of the reporting product.", _EVT_SRC,
       layer=Layer.EVENT, dtype=Dtype.STR),
    _s("event.name", Level.EVENT, Kind.ATTRIBUTE, None, "Event name.", _EVT_SRC, layer=Layer.EVENT, dtype=Dtype.STR),
    _s("event.category", Level.EVENT, Kind.ATTRIBUTE, None, "Event category as written by the source.", _EVT_SRC,
       layer=Layer.EVENT, dtype=Dtype.STR),
    _s("event.hostname", Level.EVENT, Kind.IDENTIFIER, None, "Host that logged or reported the event.", _EVT_SRC),
    _s("event.app_name", Level.EVENT, Kind.FINGERPRINT, None, "Program or application that logged the event.", _EVT_SRC),
    _s("event.process_id", Level.EVENT, Kind.ATTRIBUTE, None, "Process identifier of the logging program.", _EVT_SRC,
       layer=Layer.EVENT, dtype=Dtype.INT),
    _s("event.facility", Level.EVENT, Kind.ATTRIBUTE, None, "Syslog facility (RFC 5424 section 6.2.1).", _EVT_SRC,
       layer=Layer.EVENT, dtype=Dtype.INT),
    _s("event.syslog_severity", Level.EVENT, Kind.ATTRIBUTE, None, "Syslog severity, 0 emergency to 7 debug.", _EVT_SRC,
       layer=Layer.EVENT, dtype=Dtype.INT),
    _s("event.msgid", Level.EVENT, Kind.ATTRIBUTE, None, "Syslog MSGID (RFC 5424).", _EVT_SRC, layer=Layer.EVENT,
       dtype=Dtype.STR),
    _s("event.structured_data", Level.EVENT, Kind.ATTRIBUTE, None, "Syslog structured data, element id to parameters.",
       _EVT_SRC, layer=Layer.EVENT, dtype=Dtype.MAP),
    _s("dev.if_index", Level.DEVICE, Kind.ATTRIBUTE, None, "Interface index (ifIndex).", _DEV_SRC, dtype=Dtype.INT,
       statuses=STATE_FACT_STATUSES),
    _s("dev.if_name", Level.DEVICE, Kind.IDENTIFIER, None, "Interface name.", _DEV_SRC),
    _s("dev.if_speed", Level.DEVICE, Kind.ATTRIBUTE, "bit/s", "Interface speed.", _DEV_SRC, dtype=Dtype.FLOAT,
       statuses=STATE_FACT_STATUSES),
    _s("dev.if_type", Level.DEVICE, Kind.ATTRIBUTE, None, "Interface type (IANAifType).", _DEV_SRC, dtype=Dtype.INT,
       statuses=STATE_FACT_STATUSES),
    *(
        _s(f"dev.if_{d}_{c}", Level.DEVICE, Kind.ATTRIBUTE, unit, f"{text} {'received' if d == 'in' else 'sent'} during the interval.",
           _DEV_SRC, dtype=Dtype.INT)
        for d in ("in", "out")
        for c, unit, text in (("errors", "packets", "Packets in error"), ("discards", "packets", "Packets discarded"),
                              ("unicast", "packets", "Unicast packets"), ("multicast", "packets", "Multicast packets"),
                              ("broadcast", "packets", "Broadcast packets"))
    ),
    _s("dev.if_in_unknown_protos", Level.DEVICE, Kind.ATTRIBUTE, "packets",
       "Packets received for an unknown protocol during the interval.", _DEV_SRC, dtype=Dtype.INT),
    _s("dev.if_promiscuous", Level.DEVICE, Kind.ATTRIBUTE, None, "True if the interface is in promiscuous mode.",
       _DEV_SRC, dtype=Dtype.BOOL, statuses=STATE_FACT_STATUSES),
    _s("dev.uptime", Level.DEVICE, Kind.ATTRIBUTE, "s", "Time since the device or sensor started.", _DEV_SRC,
       dtype=Dtype.FLOAT, statuses=STATE_FACT_STATUSES),
    _s("ot.modbus.transaction_id", Level.OT, Kind.ATTRIBUTE, None, "Modbus/TCP transaction identifier.",
       "D-34, D-38: Zeek modbus, Suricata modbus", layer=Layer.OT_CII, dtype=Dtype.INT),
    _s("ot.dnp3.function_code_reply", Level.OT, Kind.ATTRIBUTE, None, "DNP3 function code of the response.",
       "D-34, D-38: Zeek dnp3, Suricata dnp3", layer=Layer.OT_CII, dtype=Dtype.INT, codes=DNP3_FUNCTION_CODES),
)


def _build_catalogue() -> tuple[tuple[FieldSpec, ...], dict[str, FieldSpec]]:
    """Shared fields plus the source-native fields of every mapping table, checked for duplicates."""
    from nagahana.datamodel.native import native_specs

    ordered = (*_FIELDS, *native_specs())
    out: dict[str, FieldSpec] = {}
    for f in ordered:
        if f.id in out:
            raise InvariantViolation(f"Duplicate field id {f.id}")
        out[f.id] = f
    return ordered, out


SHARED_FIELDS: tuple[FieldSpec, ...] = _FIELDS
ALL_FIELDS: tuple[FieldSpec, ...]
_CATALOGUE: dict[str, FieldSpec]
ALL_FIELDS, _CATALOGUE = _build_catalogue()
#: Every field of the data model, by ID (read-only).
CATALOGUE: Mapping[str, FieldSpec] = MappingProxyType(_CATALOGUE)


def spec(field_id: str) -> FieldSpec:
    """Look up a field; unknown IDs raise with a hint."""
    try:
        return CATALOGUE[field_id]
    except KeyError:
        raise KeyError(f"Unknown field {field_id!r}; add it to datamodel/fields.py or a mapping table first.") from None


def by_level(level: Level) -> tuple[FieldSpec, ...]:
    """All shared fields at one level, in catalogue order."""
    return tuple(f for f in SHARED_FIELDS if f.level is level)


def by_layer(layer: Layer, *, include_native: bool = True) -> tuple[FieldSpec, ...]:
    """All fields of one data-model layer, in catalogue order."""
    pool = ALL_FIELDS if include_native else SHARED_FIELDS
    return tuple(f for f in pool if f.layer is layer)


def required_ids() -> tuple[str, ...]:
    """IDs required by the problem statement's feature lists and by decided design (D-53)."""
    return tuple(f.id for f in SHARED_FIELDS if f.required_by in REQUIRED_SOURCES)


def matrix_ids() -> tuple[str, ...]:
    """IDs of every matrix-kind field (each has an input column), in catalogue order."""
    return tuple(f.id for f in ALL_FIELDS if f.kind in MATRIX_KINDS)


def ids(fields: Iterable[FieldSpec]) -> tuple[str, ...]:
    """IDs of the given specs."""
    return tuple(f.id for f in fields)


def code_name(field_id: str, code: int) -> str | None:
    """Meaning of `code` in a categorical field's code table (None when undocumented)."""
    codes = CATALOGUE[field_id].codes
    return None if codes is None else codes.get(int(code))


def bit_names(field_id: str, value: int) -> tuple[str, ...]:
    """Names of the bits set in a bitmask field's value, lowest bit first."""
    codes = CATALOGUE[field_id].codes or {}
    return tuple(name for bit, name in sorted(codes.items()) if int(value) & bit)


__all__ = [
    "ALL_FIELDS", "CATALOGUE", "MATRIX_KINDS", "REQUIRED_SOURCES", "SHARED_FIELDS", "Dtype", "FieldSpec", "Kind",
    "Level", "bit_names", "by_layer", "by_level", "check_value", "code_name", "ids", "matrix_ids", "required_ids", "spec",
]
