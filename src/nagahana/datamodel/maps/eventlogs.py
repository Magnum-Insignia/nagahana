"""Event-log mapping tables: syslog (RFC 5424 and RFC 3164 envelopes), CEF, LEEF 1.0 and 2.0, Windows
Security events (4624, 4625, 4634, 4648, 4672, 4768, 4769, 4771, 4776; XML and JSON exports), Linux log
lines and journald JSON entries.

References: RFC 5424 sections 6 and 6.3 (header and structured data), RFC 3164 section 4.1; ArcSight
"Common Event Format" implementation standard (header, escaping, extension dictionary); IBM QRadar
"Log Event Extended Format (LEEF)" versions 1.0 and 2.0 (header, predefined attributes, custom
delimiter); Microsoft "Audit ..." event documentation for each event ID (EventData fields);
systemd.journal-fields(7) and the journal export JSON format.
"""

from __future__ import annotations

from nagahana.datamodel.native import R, RecordMap, Row
from nagahana.datamodel.spec import Level

REF_5424 = "RFC 5424 (The Syslog Protocol)"
REF_3164 = "RFC 3164 (The BSD syslog Protocol)"
REF_CEF = "ArcSight Common Event Format (CEF) implementation standard"
REF_LEEF = "IBM QRadar Log Event Extended Format (LEEF) 1.0 and 2.0"
REF_WIN = "Microsoft Windows security auditing event documentation (event IDs 4624 to 4776)"
REF_LINUX = "RFC 3164 / RFC 5424 envelopes as written by syslog daemons; systemd.journal-fields(7)"


def _syslog_rows(prefix: str) -> tuple[Row, ...]:
    """Envelope rows; native IDs shared by every syslog-carried record ("syslog.<field>")."""
    return (
        R("pri", "i", conv="count", native="syslog.pri", note="PRI = facility * 8 + severity."),
        R("facility", "i", "event.facility", "count"),
        R("severity", "i", "event.syslog_severity", "count"),
        R("version", "i", conv="count", native="syslog.version"),
        R("timestamp", "s", "@time", "syslog_time"),
        R("hostname", "s", "event.hostname", "string"),
        R("app_name", "s", "event.app_name", "string"),
        R("procid", "s", "event.process_id", "int_or_none", also="native", native=f"{prefix}.procid"),
        R("msgid", "s", "event.msgid", "string"),
        R("structured_data", "M", "event.structured_data", "map"),
        R("message", "s", "event.message", "string"),
    )


SYSLOG_5424 = RecordMap("syslog", "rfc5424", "Syslog RFC 5424 message", Level.EVENT, _syslog_rows("syslog.rfc5424"),
                        REF_5424, ocsf_class=0,
                        notes=("Entities: the HOSTNAME (an address, else a host name) as subject.",))
SYSLOG_3164 = RecordMap("syslog", "rfc3164", "Syslog RFC 3164 message", Level.EVENT, _syslog_rows("syslog.rfc3164"),
                        REF_3164, ocsf_class=0,
                        notes=("The timestamp has no year and no zone: the year and UTC offset come from configuration "
                               "(AS-698); the TAG gives app_name and procid.",))

_CEF_KEYS: tuple[Row, ...] = (
    R("act", "s", "event.action", "action_word", also="native"),
    R("app", "s", "flow.app_proto", "app_proto", also="flow.app_proto_text"),
    R("cat", "s", "event.category", "string"),
    R("cnt", "i", conv="count"),
    R("dhost", "s", "flow.dst_hostname", "string"),
    R("dmac", "m", "flow.dst_mac", "mac"),
    R("dntdom", "s", "auth.domain", "string"),
    R("dpid", "i", conv="count"),
    R("dpriv", "s"),
    R("dproc", "s", kind="id"),
    R("dpt", "i", "flow.dst_port", "port"),
    R("dst", "a", "flow.dst_ip", "addr"),
    R("dtz", "s"),
    R("duid", "s", kind="id"),
    R("duser", "s", "auth.user", "string"),
    R("dvc", "a", conv="addr", kind="id"),
    R("dvchost", "s", "event.hostname", "string"),
    R("dvcmac", "m", conv="mac", kind="id"),
    R("dvcpid", "i", conv="count"),
    R("end", "t", "flow.end_time", "cef_time", unit="s"),
    R("externalId", "s", kind="id"),
    R("fileCreateTime", "t", conv="cef_time", unit="s"),
    R("fileHash", "s", kind="id"),
    R("fileId", "s", kind="id"),
    R("fileModificationTime", "t", conv="cef_time", unit="s"),
    R("filePath", "s", kind="id"),
    R("filePermission", "s"),
    R("fileType", "s"),
    R("flexDate1", "t", conv="cef_time", unit="s"),
    R("flexDate1Label", "s"),
    R("flexString1", "s"),
    R("flexString1Label", "s"),
    R("flexString2", "s"),
    R("flexString2Label", "s"),
    R("fname", "s", "file.name", "string"),
    R("fsize", "i", "file.size", "count", unit="bytes"),
    R("in", "i", conv="count", unit="bytes", note="Bytes from destination to source; the byte layer is vendor-defined."),
    R("msg", "s", "event.message", "string"),
    R("oldFileCreateTime", "t", conv="cef_time", unit="s"),
    R("oldFileHash", "s", kind="id"),
    R("oldFileId", "s", kind="id"),
    R("oldFileModificationTime", "t", conv="cef_time", unit="s"),
    R("oldFileName", "s", kind="id"),
    R("oldFilePath", "s", kind="id"),
    R("oldFilePermission", "s"),
    R("oldFileSize", "i", conv="count", unit="bytes"),
    R("oldFileType", "s"),
    R("out", "i", conv="count", unit="bytes", note="Bytes from source to destination; the byte layer is vendor-defined."),
    R("outcome", "s", "event.outcome", "string"),
    R("proto", "s", "flow.protocol", "proto_name"),
    R("reason", "s"),
    R("request", "s", kind="id", note="URL of the request."),
    R("requestClientApplication", "s", "proto.http.user_agent", "string"),
    R("requestContext", "s", "proto.http.referrer", "string"),
    R("requestCookies", "s"),
    R("requestMethod", "s", "proto.http.method", "code:http_method", also="proto.http.method_text"),
    R("rt", "t", "@time", "cef_time"),
    R("shost", "s", "flow.src_hostname", "string"),
    R("smac", "m", "flow.src_mac", "mac"),
    R("sntdom", "s", "auth.subject_domain", "string"),
    R("spid", "i", conv="count"),
    R("spriv", "s"),
    R("sproc", "s", kind="id"),
    R("spt", "i", "flow.src_port", "port"),
    R("src", "a", "flow.src_ip", "addr"),
    R("start", "t", "flow.start_time", "cef_time", unit="s"),
    R("suid", "s", kind="id"),
    R("suser", "s", "auth.subject_user", "string"),
    R("type", "i", conv="count", note="0 base, 1 aggregated, 2 correlation, 3 action event."),
    R("deviceDirection", "i", conv="count", note="0 inbound, 1 outbound."),
    R("deviceDnsDomain", "s", kind="id"),
    R("deviceExternalId", "s", kind="id"),
    R("deviceFacility", "s"),
    R("deviceInboundInterface", "s", kind="id"),
    R("deviceNtDomain", "s", kind="id"),
    R("deviceOutboundInterface", "s", kind="id"),
    R("devicePayloadId", "s", kind="id"),
    R("deviceProcessName", "s", kind="id"),
    R("deviceTranslatedAddress", "a", conv="addr", kind="id"),
    R("deviceCustomDate1", "t", conv="cef_time", unit="s"),
    R("deviceCustomDate1Label", "s"),
    R("deviceCustomDate2", "t", conv="cef_time", unit="s"),
    R("deviceCustomDate2Label", "s"),
    R("destinationDnsDomain", "s", kind="id"),
    R("destinationServiceName", "s", kind="id"),
    R("destinationTranslatedAddress", "a", conv="addr", kind="id"),
    R("destinationTranslatedPort", "i", conv="port"),
    R("sourceDnsDomain", "s", kind="id"),
    R("sourceServiceName", "s", kind="id"),
    R("sourceTranslatedAddress", "a", conv="addr", kind="id"),
    R("sourceTranslatedPort", "i", conv="port"),
    *(R(f"cs{i}", "s") for i in range(1, 7)),
    *(R(f"cs{i}Label", "s") for i in range(1, 7)),
    *(R(f"cn{i}", "i", conv="int") for i in range(1, 4)),
    *(R(f"cn{i}Label", "s") for i in range(1, 4)),
    *(R(f"cfp{i}", "f", conv="double") for i in range(1, 5)),
    *(R(f"cfp{i}Label", "s") for i in range(1, 5)),
    *(R(f"c6a{i}", "a", conv="addr", kind="id") for i in range(1, 5)),
    *(R(f"c6a{i}Label", "s") for i in range(1, 5)),
    R("agt", "a", conv="addr", kind="id"),
    R("ahost", "s", kind="id"),
    R("aid", "s", kind="id"),
    R("amac", "m", conv="mac", kind="id"),
    R("art", "t", conv="cef_time", unit="s"),
    R("at", "s"),
    R("atz", "s"),
    R("av", "s"),
    R("agentDnsDomain", "s", kind="id"),
    R("agentNtDomain", "s", kind="id"),
    R("agentTranslatedAddress", "a", conv="addr", kind="id"),
    R("agentTranslatedZoneExternalID", "s"),
    R("agentTranslatedZoneURI", "s"),
    R("agentZoneExternalID", "s"),
    R("agentZoneURI", "s"),
    R("customerExternalID", "s"),
    R("customerURI", "s"),
    R("destinationTranslatedZoneExternalID", "s"),
    R("destinationTranslatedZoneURI", "s"),
    R("destinationZoneExternalID", "s"),
    R("destinationZoneURI", "s"),
    R("deviceTranslatedZoneExternalID", "s"),
    R("deviceTranslatedZoneURI", "s"),
    R("deviceZoneExternalID", "s"),
    R("deviceZoneURI", "s"),
    R("sourceTranslatedZoneExternalID", "s"),
    R("sourceTranslatedZoneURI", "s"),
    R("sourceZoneExternalID", "s"),
    R("sourceZoneURI", "s"),
    R("slat", "f", conv="double", unit="deg"),
    R("slong", "f", conv="double", unit="deg"),
    R("dlat", "f", conv="double", unit="deg"),
    R("dlong", "f", conv="double", unit="deg"),
    R("eventId", "i", conv="count"),
    R("rawEvent", "s"),
)

CEF = RecordMap(
    "cef", "event", "CEF event", Level.EVENT, (
        R("cef_version", "i", conv="count"),
        R("device_vendor", "s", "event.vendor", "string"),
        R("device_product", "s", "event.product", "string"),
        R("device_version", "s", "event.product_version", "string"),
        R("signature_id", "s", "event.class_id", "string"),
        R("name", "s", "event.name", "string"),
        R("severity", "s", "alert.severity", "cef_severity", also="native"),
        *_CEF_KEYS,
    ), REF_CEF, ocsf_class=2004,
    notes=(
        "Header fields unescape \\| and \\\\; extension values unescape \\=, \\\\, \\n and \\r. Full key names "
        "(sourceAddress, deviceCustomString1, ...) are read as their short names.",
        "Severity 0-3 Low, 4-6 Medium, 7-8 High, 9-10 Critical on the OCSF scale; Low, Medium, High, Very-High "
        "likewise (AS-699). Derived: alert.signature from Device Event Class ID in the CEF namespace (AS-687).",
        "Event time: rt when present, else the syslog envelope's timestamp; a record with neither has no event time "
        "and is quarantined. Entities: initiator src, responder dst, else the reporting device as subject; "
        "account suser, target account duser.",
        "Keys outside the dictionary (vendor extensions) are retained as attributes 'cef.<key>'.",
    ),
)

_LEEF_KEYS: tuple[Row, ...] = (
    R("cat", "s", "event.category", "string"),
    R("devTime", "s", "@time", "leef_time"),
    R("devTimeFormat", "s"),
    R("proto", "s", "flow.protocol", "proto_name"),
    R("sev", "s", "alert.severity", "leef_severity", also="native"),
    R("src", "a", "flow.src_ip", "addr"),
    R("dst", "a", "flow.dst_ip", "addr"),
    R("srcPort", "i", "flow.src_port", "port"),
    R("dstPort", "i", "flow.dst_port", "port"),
    R("srcPreNAT", "a", conv="addr", kind="id"),
    R("dstPreNAT", "a", conv="addr", kind="id"),
    R("srcPostNAT", "a", conv="addr", kind="id"),
    R("dstPostNAT", "a", conv="addr", kind="id"),
    R("srcPreNATPort", "i", conv="port"),
    R("dstPreNATPort", "i", conv="port"),
    R("srcPostNATPort", "i", conv="port"),
    R("dstPostNATPort", "i", conv="port"),
    R("usrName", "s", "auth.user", "string"),
    R("srcMAC", "m", "flow.src_mac", "mac"),
    R("dstMAC", "m", "flow.dst_mac", "mac"),
    R("srcBytes", "i", conv="count", unit="bytes", note="Bytes sent by the source; the byte layer is vendor-defined."),
    R("dstBytes", "i", conv="count", unit="bytes"),
    R("srcPackets", "i", "flow.packets_fwd", "count", unit="packets"),
    R("dstPackets", "i", "flow.packets_bwd", "count", unit="packets"),
    R("totalPackets", "i", "flow.packets_total", "count", unit="packets"),
    R("role", "s"),
    R("realm", "s", kind="id"),
    R("policy", "s"),
    R("resource", "s", kind="id"),
    R("url", "s", kind="id"),
    R("groupID", "s", kind="id"),
    R("domain", "s", "auth.domain", "string"),
    R("isLoginEvent", "b", conv="bool"),
    R("isLogoutEvent", "b", conv="bool"),
    R("identSrc", "a", conv="addr", kind="id"),
    R("identHostName", "s", kind="id"),
    R("identNetBios", "s", kind="id"),
    R("identGrpName", "s", kind="id"),
    R("identMAC", "m", conv="mac", kind="id"),
    R("vSrc", "a", conv="addr", kind="id"),
    R("vSrcName", "s", kind="id"),
    R("accountName", "s", kind="id"),
    R("action", "s", "event.action", "action_word", also="native"),
    R("msg", "s", "event.message", "string"),
)

LEEF = RecordMap(
    "leef", "event", "LEEF event", Level.EVENT, (
        R("leef_version", "s"),
        R("vendor", "s", "event.vendor", "string"),
        R("product", "s", "event.product", "string"),
        R("version", "s", "event.product_version", "string"),
        R("event_id", "s", "event.class_id", "string"),
        R("delimiter", "s", note="LEEF 2.0 attribute delimiter (a character, or hex as x09 / 0x09)."),
        *_LEEF_KEYS,
    ), REF_LEEF, ocsf_class=2004,
    notes=(
        "LEEF 1.0 attributes are tab-separated; LEEF 2.0 names its delimiter in the sixth header field. A backslash "
        "escapes the next character in header fields and values.",
        "sev 1-10 maps to the OCSF scale as CEF severity (AS-699); devTime is read with devTimeFormat (Java "
        "SimpleDateFormat patterns) or as epoch milliseconds.",
        "Derived: auth.activity = 1 when isLoginEvent is true, 2 when isLogoutEvent is true; alert.signature from "
        "EventID in the LEEF namespace (AS-687). Keys outside the predefined set are retained as 'leef.<key>'.",
    ),
)

_WIN_SYSTEM: tuple[Row, ...] = (
    R("System.Provider.Name", "s", native="windows.system.provider_name"),
    R("System.Provider.Guid", "s", native="windows.system.provider_guid", kind="id"),
    R("System.EventID", "i", conv="count", native="windows.system.event_id"),
    R("System.Version", "i", conv="count", native="windows.system.version"),
    R("System.Level", "i", conv="count", native="windows.system.level"),
    R("System.Task", "i", conv="count", native="windows.system.task"),
    R("System.Opcode", "i", conv="count", native="windows.system.opcode"),
    R("System.Keywords", "s", native="windows.system.keywords", note="0x8020000000000000 audit success, 0x8010... failure."),
    R("System.TimeCreated.SystemTime", "s", "@time", "rfc3339"),
    R("System.EventRecordID", "i", conv="count", native="windows.system.event_record_id"),
    R("System.Correlation.ActivityID", "s", native="windows.system.activity_id", kind="id"),
    R("System.Correlation.RelatedActivityID", "s", native="windows.system.related_activity_id", kind="id"),
    R("System.Execution.ProcessID", "i", conv="count", native="windows.system.process_id"),
    R("System.Execution.ThreadID", "i", conv="count", native="windows.system.thread_id"),
    R("System.Channel", "s", native="windows.system.channel"),
    R("System.Computer", "s", "event.hostname", "string"),
    R("System.Security.UserID", "s", native="windows.system.user_id", kind="id"),
)


def _w(name: str, target: str | None = None, conv: str = "", dtype: str = "s", kind: str = "attr", **kw: str) -> Row:
    """An EventData row; native IDs are shared by every event ID ("windows.<Name>")."""
    return R(f"EventData.{name}", dtype, target, conv or ("string" if target else ""), kind=kind, native=f"windows.{name}",
             **kw)


_SUBJECT = (
    _w("SubjectUserSid", kind="id"), _w("SubjectUserName", "auth.subject_user"),
    _w("SubjectDomainName", "auth.subject_domain"), _w("SubjectLogonId", kind="id"),
)
_SOURCE = (_w("IpAddress", "flow.src_ip", "win_ip", dtype="a"), _w("IpPort", "flow.src_port", "win_port", dtype="i"))
_PROCESS = (_w("ProcessId", conv="hex", dtype="i"), _w("ProcessName", "auth.process_name"))
_LOGON_COMMON = (
    _w("LogonType", "auth.logon_type", "count", dtype="i"), _w("LogonProcessName", "auth.logon_process", "trim"),
    _w("AuthenticationPackageName", "auth.package", "trim"), _w("WorkstationName", "auth.workstation", "win_name"),
    _w("TransmittedServices"), _w("LmPackageName"), _w("KeyLength", "auth.key_length", "count", dtype="i"),
)
_KERB_CERT = (_w("CertIssuerName", kind="id"), _w("CertSerialNumber", kind="id"), _w("CertThumbprint", kind="id"))


def _win(event_id: int, title: str, rows: tuple[Row, ...], notes: tuple[str, ...]) -> RecordMap:
    return RecordMap("windows", str(event_id), f"Windows Security event {event_id} ({title})", Level.AUTH,
                     (*_WIN_SYSTEM, *rows), REF_WIN, native_prefix="windows", ocsf_class=3002, notes=notes)


_WIN_ENTITIES = "Entities: initiator = IpAddress host (when an address), responder = Computer, account = TargetDomainName\\TargetUserName."

WIN_4624 = _win(4624, "an account was successfully logged on", (
    *_SUBJECT, _w("TargetUserSid", "auth.user_sid"), _w("TargetUserName", "auth.user"),
    _w("TargetDomainName", "auth.domain"), _w("TargetLogonId", "auth.logon_id"), *_LOGON_COMMON, _w("LogonGuid", kind="id"),
    *_PROCESS, *_SOURCE, _w("ImpersonationLevel"), _w("RestrictedAdminMode"), _w("TargetOutboundUserName", kind="id"),
    _w("TargetOutboundDomainName", kind="id"), _w("VirtualAccount"), _w("TargetLinkedLogonId", kind="id"),
    _w("ElevatedToken", "auth.elevated", "win_yesno"),
), ("auth.activity = 1 (Logon), auth.result = 1; auth.protocol from AuthenticationPackageName (NTLM 1, Kerberos 2; "
    "Negotiate is not resolved, NOT_SUPPLIED).", _WIN_ENTITIES))

WIN_4625 = _win(4625, "an account failed to log on", (
    *_SUBJECT, _w("TargetUserSid", "auth.user_sid"), _w("TargetUserName", "auth.user"),
    _w("TargetDomainName", "auth.domain"), _w("Status", conv="hex", dtype="i"), _w("FailureReason", "auth.failure_reason"),
    _w("SubStatus", conv="hex", dtype="i"), *_LOGON_COMMON, *_PROCESS, *_SOURCE,
), ("auth.activity = 1 (Logon), auth.result = 2; auth.failure_status = SubStatus when it is non-zero, else Status "
    "(the most specific code, AS-700).", _WIN_ENTITIES))

WIN_4634 = _win(4634, "an account was logged off", (
    _w("TargetUserSid", "auth.user_sid"), _w("TargetUserName", "auth.user"), _w("TargetDomainName", "auth.domain"),
    _w("TargetLogonId", "auth.logon_id"), _w("LogonType", "auth.logon_type", "count", dtype="i"),
), ("auth.activity = 2 (Logoff), auth.result = 1. Entities: responder = Computer, account.",))

WIN_4648 = _win(4648, "a logon was attempted using explicit credentials", (
    *_SUBJECT, _w("LogonGuid", kind="id"), _w("TargetUserName", "auth.target_user"),
    _w("TargetDomainName", "auth.target_domain"), _w("TargetLogonGuid", kind="id"),
    _w("TargetServerName", "auth.service", "win_name"), _w("TargetInfo"), *_PROCESS, *_SOURCE,
), ("auth.activity = 1 (Logon). Entities: initiator = Computer (where the credentials were used), responder = "
    "TargetServerName host, account = subject, target account = TargetDomainName\\TargetUserName.",))

WIN_4672 = _win(4672, "special privileges assigned to new logon", (
    _w("SubjectUserSid", "auth.user_sid"), _w("SubjectUserName", "auth.user"), _w("SubjectDomainName", "auth.domain"),
    _w("SubjectLogonId", "auth.logon_id"), _w("PrivilegeList", "auth.privileges", "win_list", dtype="S"),
), ("auth.activity = 99 (Other), auth.elevated = 1. Entities: responder = Computer, account = subject.",))

WIN_4768 = _win(4768, "a Kerberos authentication ticket (TGT) was requested", (
    _w("TargetUserName", "auth.user"), _w("TargetDomainName", "auth.domain"), _w("TargetSid", "auth.user_sid"),
    _w("ServiceName", "auth.service"), _w("ServiceSid", kind="id"),
    _w("TicketOptions", "proto.kerberos.ticket_options", "hex", dtype="i"),
    _w("Status", "proto.kerberos.error_code", "hex", dtype="i", also="native"),
    _w("TicketEncryptionType", "proto.kerberos.etype", "win_etype", dtype="i", also="native"),
    _w("PreAuthType", conv="int", dtype="i"), *_SOURCE, *_KERB_CERT,
), ("auth.activity = 3 (Authentication Ticket), auth.protocol = 2, auth.result from Status (0x0 success).",
    _WIN_ENTITIES + " Target account = ServiceName."))

WIN_4769 = _win(4769, "a Kerberos service ticket was requested", (
    _w("TargetUserName", "auth.user"), _w("TargetDomainName", "auth.domain"), _w("ServiceName", "auth.service"),
    _w("ServiceSid", kind="id"), _w("TicketOptions", "proto.kerberos.ticket_options", "hex", dtype="i"),
    _w("TicketEncryptionType", "proto.kerberos.etype", "win_etype", dtype="i", also="native"), *_SOURCE,
    _w("Status", "proto.kerberos.error_code", "hex", dtype="i", also="native"), _w("LogonGuid", kind="id"),
    _w("TransmittedServices"),
), ("auth.activity = 4 (Service Ticket Request), auth.protocol = 2, auth.result from Status. TargetUserName may be "
    "user@REALM: the realm part is the domain when TargetDomainName is empty.",
    _WIN_ENTITIES + " Target account = ServiceName (Kerberoasting shows as RC4, etype 23, AS-692)."))

WIN_4771 = _win(4771, "Kerberos pre-authentication failed", (
    _w("TargetUserName", "auth.user"), _w("TargetSid", "auth.user_sid"), _w("ServiceName", "auth.service"),
    _w("TicketOptions", "proto.kerberos.ticket_options", "hex", dtype="i"),
    _w("Status", "proto.kerberos.error_code", "hex", dtype="i", also="native"), _w("PreAuthType", conv="int", dtype="i"),
    *_SOURCE, *_KERB_CERT,
), ("auth.activity = 6 (Preauth), auth.result = 2, auth.protocol = 2.", _WIN_ENTITIES))

WIN_4776 = _win(4776, "the computer attempted to validate the credentials for an account", (
    _w("PackageName", "auth.package", "trim"), _w("TargetUserName", "auth.user"),
    _w("Workstation", "auth.workstation", "win_name"), _w("Status", conv="hex", dtype="i", also=None),
), ("auth.activity = 1 (Logon), auth.protocol = 1 (NTLM), auth.result from Status, auth.failure_status = Status "
    "when non-zero. Entities: initiator = Workstation host, responder = Computer, account = TargetUserName.",))

WIN_OTHER = RecordMap("windows", "other", "Windows event of another ID", Level.EVENT, _WIN_SYSTEM, REF_WIN,
                      native_prefix="windows", ocsf_class=0,
                      notes=("System fields are mapped; EventData fields are retained as 'windows.other.EventData.<Name>'. "
                             "Entities: the Computer as subject.",))

WINDOWS_MAPS: tuple[RecordMap, ...] = (WIN_4624, WIN_4625, WIN_4634, WIN_4648, WIN_4672, WIN_4768, WIN_4769, WIN_4771,
                                       WIN_4776, WIN_OTHER)

LINUX_OTHER = RecordMap(
    "linux", "syslog", "Linux log line (auth.log, syslog)", Level.EVENT, _syslog_rows("linux.syslog"),
    REF_LINUX, ocsf_class=0,
    notes=("The syslog envelope is mapped and the message kept whole in event.message. Entities: the logging host "
           "as subject.",),
)

JOURNALD = RecordMap(
    "linux", "journald", "systemd journal entry (journalctl -o json)", Level.EVENT, (
        R("__REALTIME_TIMESTAMP", "s", "@time", "journald_us"),
        R("__MONOTONIC_TIMESTAMP", "s"),
        R("__CURSOR", "s", kind="id"),
        R("_BOOT_ID", "s", kind="id"),
        R("_MACHINE_ID", "s", kind="id"),
        R("_HOSTNAME", "s", "event.hostname", "string"),
        R("SYSLOG_IDENTIFIER", "s", "event.app_name", "string"),
        R("SYSLOG_PID", "s", "event.process_id", "int_or_none"),
        R("_PID", "i", conv="int_or_none"),
        R("MESSAGE", "s", "event.message", "string"),
        R("PRIORITY", "s", "event.syslog_severity", "int_or_none"),
        R("SYSLOG_FACILITY", "s", "event.facility", "int_or_none"),
        R("SYSLOG_TIMESTAMP", "s"),
        R("MESSAGE_ID", "s", kind="id"),
        R("_COMM", "s", kind="fp"),
        R("_EXE", "s", kind="id"),
        R("_CMDLINE", "s", kind="id"),
        R("_SYSTEMD_UNIT", "s", kind="fp"),
        R("_SYSTEMD_CGROUP", "s"),
        R("_SYSTEMD_SLICE", "s"),
        R("_SYSTEMD_INVOCATION_ID", "s", kind="id"),
        R("INVOCATION_ID", "s", kind="id"),
        R("_UID", "i", conv="int_or_none"),
        R("_GID", "i", conv="int_or_none"),
        R("_TRANSPORT", "s"),
        R("_SOURCE_REALTIME_TIMESTAMP", "s"),
        R("_CAP_EFFECTIVE", "s"),
        R("_SELINUX_CONTEXT", "s"),
        R("_AUDIT_SESSION", "s"),
        R("_AUDIT_LOGINUID", "s"),
        R("_STREAM_ID", "s", kind="id"),
        R("_RUNTIME_SCOPE", "s"),
        R("CODE_FILE", "s"),
        R("CODE_LINE", "s"),
        R("CODE_FUNC", "s"),
    ), REF_LINUX, ocsf_class=0,
    notes=(
        "MESSAGE is kept whole in event.message. Values sent as byte arrays (non-UTF-8) are decoded with replacement; "
        "the array itself is kept in the record's attributes. Entities: the host (_HOSTNAME) as subject.",
    ),
)

EVENTLOG_MAPS: tuple[RecordMap, ...] = (SYSLOG_5424, SYSLOG_3164, CEF, LEEF, *WINDOWS_MAPS, LINUX_OTHER, JOURNALD)
