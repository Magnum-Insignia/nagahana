"""Source spellings of coded values, and their codes in the data model's code tables (datamodel/fields.py).

Each table maps a normalised spelling (`norm`: lower case, letters and digits only) to a code. A
converter "code:<table>" (ingest/convert.py) looks the source text up here; a numeric spelling ("23",
"0x17") is read as the code itself, since code spaces may grow (versioning.py). Tables with an "other"
code (application protocol, HTTP method, alert category) map an unknown spelling to it and keep the
text in a companion field; for every other table an unknown spelling is refused and counted.
"""

from __future__ import annotations

import re
from collections.abc import Mapping

from nagahana.datamodel import fields as F

_NON_ALNUM = re.compile(r"[^a-z0-9]")


def norm(text: str) -> str:
    """Normalised spelling: lower case, letters and digits only ("TLS 1.2" -> "tls12")."""
    return _NON_ALNUM.sub("", text.lower())


def _from_names(names: Mapping[int, str], *extra: tuple[str, int]) -> dict[str, int]:
    out = {norm(name): code for code, name in names.items()}
    for spelling, code in extra:
        out[norm(spelling)] = code
    return out


CONN_STATE: dict[str, int] = _from_names(F.CONN_STATE_CODES)

APP_PROTO: dict[str, int] = _from_names(
    F.APP_PROTO_CODES,
    ("ssl", 3), ("https", 3), ("tls13", 3), ("krb", 10), ("krbtcp", 10), ("krb5", 10), ("kerberos5", 10),
    ("dcerpc", 9), ("dcerpcudp", 9), ("msrpc", 9), ("smb2", 8), ("smb3", 8), ("cifs", 8), ("pgsql", 19),
    ("postgres", 19), ("ikev2", 39), ("isakmp", 39), ("bittorrentdht", 46), ("http1", 2), ("http11", 2),
    ("nbns", 43), ("netbiosns", 43), ("netbiosssn", 43), ("netbiosdgm", 43), ("nbss", 43), ("rdpeudp", 12),
    ("vnc", 25), ("ldaptcp", 22), ("ldapudp", 22), ("cldap", 22), ("tftpdata", 28), ("smtps", 5), ("imaps", 20),
    ("pop3s", 21), ("iec60870104", 32), ("iec104", 32), ("ethernetip", 33), ("cip", 33), ("s7", 34),
    ("s7commplus", 34), ("bacnetip", 35), ("dhcp6", 50), ("upnp", 51), ("quicv1", 38), ("gquic", 38),
    ("wg", 41), ("mysqlx", 18),
)
#: Labels that are not protocols (an analyser's failure) and are refused.
APP_PROTO_NOT_A_LABEL: frozenset[str] = frozenset({"failed", "unknown", "undetermined", ""})

HTTP_METHOD: dict[str, int] = _from_names(F.HTTP_METHOD_CODES)
TLS_VERSION: dict[str, int] = {
    "sslv2": 0x0002, "ssl2": 0x0002, "sslv3": 0x0300, "ssl3": 0x0300, "tlsv10": 0x0301, "tls10": 0x0301, "tlsv1": 0x0301,
    "tls1": 0x0301, "tlsv11": 0x0302, "tls11": 0x0302, "tlsv12": 0x0303, "tls12": 0x0303, "tlsv13": 0x0304,
    "tls13": 0x0304, "dtlsv10": 0xFEFF, "dtls10": 0xFEFF, "dtlsv12": 0xFEFD, "dtls12": 0xFEFD, "dtlsv13": 0xFEFC,
    "dtls13": 0xFEFC,
}
TLS_ALERT: dict[str, int] = _from_names(F.TLS_ALERT_CODES)
KRB_ERROR: dict[str, int] = {**_from_names(F.KRB_ERROR_CODES),
                             **{norm("KRB5" + name): code for code, name in F.KRB_ERROR_CODES.items()}}
KRB_ETYPE: dict[str, int] = _from_names(
    F.KRB_ETYPE_CODES, ("arcfour-hmac", 23), ("arcfour-hmac-md5", 23), ("rc4-hmac-md5", 23), ("aes128-cts", 17),
    ("aes256-cts", 18), ("aes128-sha1", 17), ("aes256-sha1", 18), ("aes128-sha2", 19), ("aes256-sha2", 20),
    ("des3-hmac-sha1", 16), ("des3-cbc-sha1-kd", 16),
)
KRB_MSG_TYPE: dict[str, int] = {
    **_from_names(F.KRB_MSG_TYPE_CODES),
    **{norm("KRB_" + name.replace("-", "_")): code for code, name in F.KRB_MSG_TYPE_CODES.items()},
    "krberror": 30,
}
DNS_QTYPE: dict[str, int] = _from_names(F.DNS_QTYPE_CODES)
DNS_RCODE: dict[str, int] = _from_names(F.DNS_RCODE_CODES, ("NoError", 0), ("FormError", 1), ("ServError", 2),
                                        ("NotImp", 4))
DNS_QCLASS: dict[str, int] = _from_names(
    F.DNS_QCLASS_CODES, ("C_INTERNET", 1), ("C_CHAOS", 3), ("C_HESIOD", 4), ("C_NONE", 254), ("C_ANY", 255),
    ("Internet", 1), ("Chaos", 3), ("Hesiod", 4),
)
DHCP_MESSAGE: dict[str, int] = _from_names(F.DHCP_MESSAGE_CODES, ("DHCPDISCOVER", 1), ("DHCPOFFER", 2),
                                           ("DHCPREQUEST", 3), ("DHCPDECLINE", 4), ("DHCPACK", 5), ("DHCPNAK", 6),
                                           ("DHCPRELEASE", 7), ("DHCPINFORM", 8))
MODBUS_FUNCTION: dict[str, int] = _from_names(
    F.MODBUS_FUNCTION_CODES, ("REPORT_SLAVE_ID", 17), ("ENCAP_INTERFACE_TRANSPORT", 43),
    ("READ_DEVICE_IDENTIFICATION", 43), ("MEI", 43), ("DIAGNOSTIC", 8),
)
MODBUS_EXCEPTION: dict[str, int] = _from_names(
    F.MODBUS_EXCEPTION_CODES, ("SLAVE_DEVICE_FAILURE", 4), ("SLAVE_DEVICE_BUSY", 6), ("NEGATIVE_ACKNOWLEDGE", 7),
    ("GATEWAY_TARGET_DEVICE_FAILED", 11),
)
DNP3_FUNCTION: dict[str, int] = _from_names(F.DNP3_FUNCTION_CODES, ("UNSOLICITED", 130), ("AUTH_RESP", 131))
SMB_ACTION: dict[str, int] = _from_names(F.SMB_FILE_ACTION_CODES)
SMB_COMMAND: dict[str, int] = {}
for _code, _name in F.SMB_COMMAND_CODES.items():
    _short = _name.split("_", 1)[1]                     # "SMB2_CREATE" -> "CREATE"
    SMB_COMMAND[norm(_name)] = _code
    SMB_COMMAND[norm(_name.split("_", 1)[0] + "_COMMAND_" + _short)] = _code
    if _code < 0x100:                                    # a bare SMB2 name means the SMB2 command
        SMB_COMMAND.setdefault(norm(_short), _code)
ALERT_CATEGORY: dict[str, int] = {
    **_from_names(F.ALERT_CATEGORY_CODES),
    **{norm(desc): code for desc, code in (
        ("Not Suspicious Traffic", 1), ("Unknown Traffic", 2), ("Potentially Bad Traffic", 3),
        ("Attempted Information Leak", 4), ("Information Leak", 5), ("Large Scale Information Leak", 6),
        ("Attempted Denial of Service", 7), ("Denial of Service", 8), ("Attempted User Privilege Gain", 9),
        ("Unsuccessful User Privilege Gain", 10), ("Successful User Privilege Gain", 11),
        ("Attempted Administrator Privilege Gain", 12), ("Successful Administrator Privilege Gain", 13),
        ("Decode of an RPC Query", 14), ("Executable code was detected", 15), ("A suspicious string was detected", 16),
        ("A suspicious filename was detected", 17),
        ("An attempted login using a suspicious username was detected", 18), ("A system call was detected", 19),
        ("A TCP connection was detected", 20), ("A Network Trojan was detected", 21),
        ("A client was using an unusual port", 22), ("Detection of a Network Scan", 23),
        ("Detection of a Denial of Service Attack", 24), ("Detection of a non-standard protocol or event", 25),
        ("Generic Protocol Command Decode", 26), ("access to a potentially vulnerable web application", 27),
        ("Web Application Attack", 28), ("Misc activity", 29), ("Misc Attack", 30), ("Generic ICMP event", 31),
        ("Inappropriate Content was Detected", 32), ("Potential Corporate Privacy Violation", 33),
        ("Attempt to login by a default username and password", 34),
        ("Sensitive Data was Transmitted Across the Network", 35), ("Known malicious file or file based exploit", 36),
        ("Known malware command and control traffic", 37), ("Known client side exploit attempt", 38),
        ("Successful Credential Theft Detected", 39), ("Possible Social Engineering Attempted", 40),
        ("Exploit Kit Activity Detected", 41), ("Domain Observed Used for C2 Detected", 42),
        ("Device Retrieving External IP Address Detected", 43), ("Targeted Malicious Activity was Detected", 44),
        ("Possibly Unwanted Program Detected", 45), ("Crypto Currency Mining Activity Detected", 46),
        ("Malware Command and Control Activity Detected", 47),
    )},
}
#: Words for the action a device took (AS-685): 1 allowed, 2 denied, 3 observed without enforcement.
EVENT_ACTION: dict[str, int] = {
    **{w: 1 for w in ("allow", "allowed", "permit", "permitted", "accept", "accepted", "pass", "passed", "success",
                      "succeeded", "forward", "forwarded")},
    **{w: 2 for w in ("deny", "denied", "drop", "dropped", "block", "blocked", "reject", "rejected", "reset",
                      "refuse", "refused", "quarantine", "quarantined", "prevent", "prevented", "failure", "failed")},
    **{w: 3 for w in ("alert", "alerted", "detect", "detected", "log", "logged", "monitor", "monitored", "observe",
                      "observed", "wouldblock", "woulddrop", "wdrop", "wouldreject", "audit")},
    **{w: 4 for w in ("rewrite", "rewritten", "modify", "modified", "replace", "replaced")},
}
SSH_METHOD: dict[str, int] = {
    "password": 1, "publickey": 2, "keyboardinteractive": 3, "keyboardinteractivepam": 3, "hostbased": 4,
    "gssapiwithmic": 5, "gssapi": 5, "gssapikeyex": 5, "none": 6, "certificate": 7,
}

#: Tables by the name used in "code:<table>", and whether an unknown spelling maps to code 0 ("other").
TABLES: dict[str, tuple[dict[str, int], bool]] = {
    "conn_state": (CONN_STATE, False),
    "app_proto": (APP_PROTO, True),
    "http_method": (HTTP_METHOD, True),
    "tls_version": (TLS_VERSION, False),
    "tls_alert": (TLS_ALERT, False),
    "krb_error": (KRB_ERROR, False),
    "krb_etype": (KRB_ETYPE, False),
    "krb_msg_type": (KRB_MSG_TYPE, False),
    "dns_qtype": (DNS_QTYPE, False),
    "dns_rcode": (DNS_RCODE, False),
    "dns_qclass": (DNS_QCLASS, False),
    "dhcp_message": (DHCP_MESSAGE, False),
    "modbus_function": (MODBUS_FUNCTION, False),
    "modbus_exception": (MODBUS_EXCEPTION, False),
    "dnp3_function": (DNP3_FUNCTION, False),
    "smb_action": (SMB_ACTION, False),
    "smb_command": (SMB_COMMAND, False),
    "alert_category": (ALERT_CATEGORY, True),
    "event_action": (EVENT_ACTION, False),
    "ssh_method": (SSH_METHOD, True),
}

#: Prefixes stripped before a lookup ("SMB::FILE_OPEN" -> "FILE_OPEN", "Modbus::..." etc.).
STRIP_PREFIXES: dict[str, tuple[str, ...]] = {
    "smb_action": ("smb",),
    "modbus_function": ("modbus",),
    "modbus_exception": ("modbus",),
    "dnp3_function": ("dnp3",),
}

#: OpenConfig port-speed identities -> bits per second (openconfig-if-ethernet ETHERNET_SPEED).
PORT_SPEED_BPS: dict[str, float] = {
    "speed10mb": 1e7, "speed100mb": 1e8, "speed1gb": 1e9, "speed2500mb": 2.5e9, "speed5gb": 5e9, "speed10gb": 1e10,
    "speed25gb": 2.5e10, "speed40gb": 4e10, "speed50gb": 5e10, "speed100gb": 1e11, "speed200gb": 2e11,
    "speed400gb": 4e11, "speed600gb": 6e11, "speed800gb": 8e11,
}
