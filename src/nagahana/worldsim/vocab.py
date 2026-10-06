"""Fixed vocabularies of the world simulator: archetypes, services, techniques, campaigns (P-14).

Everything a scenario refers to by name is defined here once, with integer codes the backends use.
Codes are an interface (they index arrays in `topology.py`, `attack.py` and `observe.py`), so this
table is append-only: adding a row keeps existing codes.

Alignment with the model. Ground-truth attack stages use `nagahana.models.vocab.STAGES` (the 15
classes: "none" plus the 14 ATT&CK Enterprise tactics) so a simulated world's labels are exactly the
head's classes. Techniques keep their native ATT&CK identifier and tactic, Enterprise
(https://attack.mitre.org/techniques/enterprise/) or ICS (https://attack.mitre.org/techniques/ics/).
ICS-only tactics (Inhibit Response Function, Impair Process Control) have no Enterprise-tactic class
in the model, so their stage maps to the nearest model class, "impact", while the native ICS tactic
and technique identifier are preserved in the ground-truth metadata (AS-807).

Purdue levels (ISA-95 / IEC 62264 and the Purdue Enterprise Reference Architecture): level 0 field
devices (sensors, actuators), level 1 controllers (PLC, RTU), level 2 supervisory (HMI, SCADA),
level 3 operations (historian, engineering workstation), level 3.5 the industrial DMZ, levels 4 and
5 enterprise IT. OT protocol ports follow the IANA registry and the protocol standards: Modbus/TCP
502, DNP3 20000, IEC 60870-5-104 2404, S7comm (ISO-TSAP) 102, EtherNet/IP 44818.
"""

from __future__ import annotations

from dataclasses import dataclass

from nagahana.models.vocab import STAGE_CODE

#: Entity archetypes. `kind` is a `datamodel.records.ENTITY_KINDS` value. `purdue` is the Purdue
#: level (-1 for pure IT) and `internal` whether it sits inside the monitored perimeter.
ENTERPRISE = "enterprise"
OT = "ot"


@dataclass(frozen=True)
class Archetype:
    """One kind of machine a topology can place."""

    name: str
    kind: str                 # datamodel entity kind
    domain: str               # ENTERPRISE or OT
    purdue: int               # Purdue level, or -1 for IT
    internal: bool
    holds_data: bool          # a store worth collecting from (file server, historian, database)
    control_capable: bool     # can issue OT control commands (HMI, engineering workstation)


ARCHETYPES: tuple[Archetype, ...] = (
    Archetype("workstation", "host", ENTERPRISE, -1, True, False, False),
    Archetype("admin_workstation", "host", ENTERPRISE, -1, True, False, False),
    Archetype("file_server", "host", ENTERPRISE, -1, True, True, False),
    Archetype("app_server", "host", ENTERPRISE, -1, True, True, False),
    Archetype("database", "host", ENTERPRISE, -1, True, True, False),
    Archetype("domain_controller", "host", ENTERPRISE, -1, True, True, False),
    Archetype("dmz_web", "host", ENTERPRISE, -1, True, True, False),
    Archetype("dmz_mail", "host", ENTERPRISE, -1, True, True, False),
    Archetype("gateway", "host", ENTERPRISE, -1, True, False, False),
    Archetype("cloud_egress", "external", ENTERPRISE, -1, False, False, False),
    Archetype("internet", "external", ENTERPRISE, -1, False, False, False),
    Archetype("multicast_group", "multicast", ENTERPRISE, -1, False, False, False),
    Archetype("engineering_workstation", "host", OT, 3, True, True, True),
    Archetype("historian", "host", OT, 3, True, True, False),
    Archetype("ot_dmz", "host", OT, 3, True, True, False),
    Archetype("hmi", "host", OT, 2, True, False, True),
    Archetype("scada_server", "host", OT, 2, True, True, True),
    Archetype("plc", "ot_device", OT, 1, True, False, False),
    Archetype("rtu", "ot_device", OT, 1, True, False, False),
    Archetype("field_device", "ot_device", OT, 0, True, False, False),
)
ARCHETYPE_CODE: dict[str, int] = {a.name: i for i, a in enumerate(ARCHETYPES)}
N_ARCHETYPES = len(ARCHETYPES)


@dataclass(frozen=True)
class Service:
    """A network service an entity can expose. `plane` matches `graph.planes` where one applies."""

    name: str
    port: int
    protocol: int             # IANA protocol number (6 TCP, 17 UDP)
    plane: str
    encrypted: bool
    ot: bool


SERVICES: tuple[Service, ...] = (
    Service("http", 80, 6, "services", False, False),
    Service("https", 443, 6, "services", True, False),
    Service("dns", 53, 17, "name_resolution", False, False),
    Service("smb", 445, 6, "identity", False, False),
    Service("kerberos", 88, 6, "identity", False, False),
    Service("ldap", 389, 6, "identity", False, False),
    Service("rdp", 3389, 6, "remote_admin", True, False),
    Service("ssh", 22, 6, "remote_admin", True, False),
    Service("winrm", 5985, 6, "remote_admin", False, False),
    Service("smtp", 25, 6, "services", False, False),
    Service("modbus", 502, 6, "ot_control", False, True),
    Service("dnp3", 20000, 6, "ot_control", False, True),
    Service("iec104", 2404, 6, "ot_control", False, True),
    Service("s7comm", 102, 6, "ot_control", False, True),
    Service("ethernet_ip", 44818, 6, "ot_control", False, True),
    Service("historian_api", 8443, 6, "services", True, True),
)
SERVICE_CODE: dict[str, int] = {s.name: i for i, s in enumerate(SERVICES)}
N_SERVICES = len(SERVICES)
OT_SERVICE_CODES: tuple[int, ...] = tuple(i for i, s in enumerate(SERVICES) if s.ot)

#: Default services exposed by each archetype (server side). Clients expose none.
DEFAULT_SERVICES: dict[str, tuple[str, ...]] = {
    "file_server": ("smb",),
    "app_server": ("http", "https"),
    "database": ("https",),
    "domain_controller": ("kerberos", "ldap", "smb", "dns"),
    "dmz_web": ("http", "https"),
    "dmz_mail": ("smtp", "https"),
    "gateway": ("dns",),
    "engineering_workstation": ("winrm",),
    "historian": ("historian_api", "https"),
    "ot_dmz": ("https", "historian_api"),
    "hmi": ("rdp",),
    "scada_server": ("modbus", "iec104", "historian_api"),
    "plc": ("modbus", "s7comm"),
    "rtu": ("dnp3",),
    "field_device": (),
    "workstation": (),
    "admin_workstation": ("winrm",),
    "cloud_egress": ("https",),
    "internet": ("https", "http", "dns"),
    "multicast_group": (),
}

#: Vulnerability catalogue. `enables` is the semantic category of the technique it unlocks (SEM_*);
#: `service` is the exposed service it is reachable through (-1 for a local/host vulnerability).
SEM_SCAN_EXTERNAL = 0
SEM_SCAN_INTERNAL = 1
SEM_EXPLOIT_SERVICE = 2
SEM_VALID_ACCOUNT = 3
SEM_PRIV_ESC = 4
SEM_CRED_ACCESS = 5
SEM_PERSIST = 6
SEM_C2 = 7
SEM_COLLECT = 8
SEM_EXFIL = 9
SEM_DOS = 10
SEM_OT_COMMAND = 11
SEM_RANSOM = 12
N_SEM = 13


@dataclass(frozen=True)
class Vulnerability:
    """A weakness that unlocks a technique on a host."""

    name: str
    enables: int              # SEM_* category
    service: int              # service code the weakness is reachable through, or -1 (local)
    severity: float           # CVSS-like base score in [0, 10]


VULNERABILITIES: tuple[Vulnerability, ...] = (
    Vulnerability("web_app_rce", SEM_EXPLOIT_SERVICE, SERVICE_CODE["http"], 9.8),
    Vulnerability("web_app_rce_tls", SEM_EXPLOIT_SERVICE, SERVICE_CODE["https"], 9.8),
    Vulnerability("smb_remote_exec", SEM_EXPLOIT_SERVICE, SERVICE_CODE["smb"], 9.3),
    Vulnerability("rdp_preauth", SEM_EXPLOIT_SERVICE, SERVICE_CODE["rdp"], 9.8),
    Vulnerability("ssh_weak_auth", SEM_EXPLOIT_SERVICE, SERVICE_CODE["ssh"], 7.5),
    Vulnerability("local_priv_esc", SEM_PRIV_ESC, -1, 7.8),
    Vulnerability("plc_unauth_write", SEM_OT_COMMAND, SERVICE_CODE["modbus"], 10.0),
    Vulnerability("s7_unauth", SEM_OT_COMMAND, SERVICE_CODE["s7comm"], 9.1),
)
VULN_CODE: dict[str, int] = {v.name: i for i, v in enumerate(VULNERABILITIES)}
N_VULNS = len(VULNERABILITIES)


@dataclass(frozen=True)
class Technique:
    """One attacker action. `sem` drives the engine; the rest is ground-truth metadata (AS-807)."""

    name: str
    attack_id: str            # native ATT&CK technique id (Enterprise T1xxx or ICS T0xxx)
    tactic_id: str            # native ATT&CK tactic id
    stage: str                # model stage name (nagahana.models.vocab.STAGES)
    sem: int
    domain: str               # ENTERPRISE or OT
    noise: float              # exposure a firing adds to defender suspicion (and IDS loudness)


TECHNIQUES: tuple[Technique, ...] = (
    Technique("active_scanning", "T1595", "TA0043", "reconnaissance", SEM_SCAN_EXTERNAL, ENTERPRISE, 1.0),
    Technique("network_service_discovery", "T1046", "TA0007", "discovery", SEM_SCAN_INTERNAL, ENTERPRISE, 0.8),
    Technique("exploit_public_app", "T1190", "TA0001", "initial_access", SEM_EXPLOIT_SERVICE, ENTERPRISE, 0.6),
    Technique("external_remote_services", "T1133", "TA0001", "initial_access", SEM_VALID_ACCOUNT, ENTERPRISE, 0.3),
    Technique("remote_services", "T1021", "TA0008", "lateral_movement", SEM_VALID_ACCOUNT, ENTERPRISE, 0.3),
    Technique("brute_force", "T1110", "TA0006", "credential_access", SEM_CRED_ACCESS, ENTERPRISE, 0.9),
    Technique("os_credential_dumping", "T1003", "TA0006", "credential_access", SEM_CRED_ACCESS, ENTERPRISE, 0.4),
    Technique("exploitation_priv_esc", "T1068", "TA0004", "privilege_escalation", SEM_PRIV_ESC, ENTERPRISE, 0.3),
    Technique("web_shell", "T1505.003", "TA0003", "persistence", SEM_PERSIST, ENTERPRISE, 0.3),
    Technique("app_layer_c2", "T1071.001", "TA0011", "command_and_control", SEM_C2, ENTERPRISE, 0.2),
    Technique("data_from_local_system", "T1005", "TA0009", "collection", SEM_COLLECT, ENTERPRISE, 0.2),
    Technique("exfil_over_c2", "T1041", "TA0010", "exfiltration", SEM_EXFIL, ENTERPRISE, 0.4),
    Technique("network_dos", "T1498", "TA0040", "impact", SEM_DOS, ENTERPRISE, 1.0),
    Technique("data_encrypted_impact", "T1486", "TA0040", "impact", SEM_RANSOM, ENTERPRISE, 0.9),
    Technique("ics_remote_discovery", "T0846", "TA0102", "discovery", SEM_SCAN_INTERNAL, OT, 0.7),
    Technique("ics_remote_services", "T0886", "TA0109", "lateral_movement", SEM_VALID_ACCOUNT, OT, 0.3),
    Technique("ics_default_credentials", "T0812", "TA0108", "lateral_movement", SEM_VALID_ACCOUNT, OT, 0.4),
    Technique("ics_unauthorized_command", "T0855", "TA0106", "impact", SEM_OT_COMMAND, OT, 0.5),
    Technique("ics_modify_parameter", "T0836", "TA0106", "impact", SEM_OT_COMMAND, OT, 0.5),
    Technique("ics_denial_of_service", "T0814", "TA0107", "impact", SEM_DOS, OT, 0.9),
)
TECHNIQUE_CODE: dict[str, int] = {t.name: i for i, t in enumerate(TECHNIQUES)}
N_TECHNIQUES = len(TECHNIQUES)
#: Model-stage code of each technique (index = technique code).
TECHNIQUE_STAGE_CODE: tuple[int, ...] = tuple(STAGE_CODE[t.stage] for t in TECHNIQUES)


@dataclass(frozen=True)
class Campaign:
    """A named attacker style: which technique categories it uses and how fast and loud.

    `rates` maps a SEM_* category to a base firing rate per hour (0 for a category the campaign never
    uses). `dwell` scales all rates (a slow campaign waits between actions). `stealth` in [0, 1]
    scales the exposure each action adds (a quiet campaign is harder for the defender to notice).
    """

    name: str
    family: str
    rates: dict[int, float]
    dwell: float
    stealth: float


def _rates(**by_name: float) -> dict[int, float]:
    name_to_sem = {
        "scan_ext": SEM_SCAN_EXTERNAL, "scan_int": SEM_SCAN_INTERNAL, "exploit": SEM_EXPLOIT_SERVICE,
        "account": SEM_VALID_ACCOUNT, "priv": SEM_PRIV_ESC, "cred": SEM_CRED_ACCESS, "persist": SEM_PERSIST,
        "c2": SEM_C2, "collect": SEM_COLLECT, "exfil": SEM_EXFIL, "dos": SEM_DOS, "ot": SEM_OT_COMMAND,
        "ransom": SEM_RANSOM,
    }
    return {name_to_sem[k]: v for k, v in by_name.items()}


CAMPAIGNS: tuple[Campaign, ...] = (
    Campaign("slow_recon", "recon", _rates(scan_ext=60.0, scan_int=40.0, exploit=6.0), dwell=3.0, stealth=0.2),
    Campaign("fast_ransomware", "ransomware",
             _rates(scan_ext=120.0, exploit=90.0, account=120.0, priv=90.0, cred=90.0, persist=60.0,
                    collect=60.0, ransom=90.0), dwell=0.5, stealth=0.9),
    Campaign("data_exfiltration", "exfiltration",
             _rates(scan_ext=60.0, exploit=40.0, account=60.0, cred=40.0, persist=30.0, c2=40.0,
                    collect=60.0, exfil=60.0), dwell=1.5, stealth=0.4),
    Campaign("denial_of_service", "dos", _rates(scan_ext=60.0, dos=240.0), dwell=0.5, stealth=1.0),
    Campaign("ot_manipulation", "ot-manipulation",
             _rates(scan_ext=60.0, exploit=40.0, account=60.0, scan_int=40.0, cred=30.0, persist=30.0,
                    ot=60.0), dwell=1.5, stealth=0.5),
    Campaign("apt_full", "apt",
             _rates(scan_ext=60.0, scan_int=40.0, exploit=40.0, account=50.0, priv=40.0, cred=40.0,
                    persist=30.0, c2=30.0, collect=40.0, exfil=40.0, ot=30.0), dwell=1.5, stealth=0.3),
)
CAMPAIGN_CODE: dict[str, int] = {c.name: i for i, c in enumerate(CAMPAIGNS)}
N_CAMPAIGNS = len(CAMPAIGNS)


def archetype(name: str) -> Archetype:
    """Look up an archetype by name."""
    return ARCHETYPES[ARCHETYPE_CODE[name]]


def service(name: str) -> Service:
    """Look up a service by name."""
    return SERVICES[SERVICE_CODE[name]]


def campaign(name: str) -> Campaign:
    """Look up a campaign by name."""
    return CAMPAIGNS[CAMPAIGN_CODE[name]]


__all__ = [
    "ARCHETYPES", "ARCHETYPE_CODE", "CAMPAIGNS", "CAMPAIGN_CODE", "DEFAULT_SERVICES", "ENTERPRISE",
    "N_ARCHETYPES", "N_CAMPAIGNS", "N_SEM", "N_SERVICES", "N_TECHNIQUES", "N_VULNS", "OT",
    "OT_SERVICE_CODES", "SERVICES", "SERVICE_CODE", "TECHNIQUES", "TECHNIQUE_CODE",
    "TECHNIQUE_STAGE_CODE", "VULNERABILITIES", "VULN_CODE", "Archetype", "Campaign", "Service",
    "Technique", "Vulnerability", "archetype", "campaign", "service",
    "SEM_SCAN_EXTERNAL", "SEM_SCAN_INTERNAL", "SEM_EXPLOIT_SERVICE", "SEM_VALID_ACCOUNT",
    "SEM_PRIV_ESC", "SEM_CRED_ACCESS", "SEM_PERSIST", "SEM_C2", "SEM_COLLECT", "SEM_EXFIL",
    "SEM_DOS", "SEM_OT_COMMAND", "SEM_RANSOM",
]
