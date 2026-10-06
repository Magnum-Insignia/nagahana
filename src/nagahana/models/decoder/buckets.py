"""Service-class buckets: the Decoder's target space for categorical columns (AS-33, AS-106).

Purpose
-------
The FieldEncoder sees a categorical code exactly (hashed row, AS-33); the Decoder does not try to
reconstruct the exact code ("exact port reconstruction is not needed for the world model", AS-33).
It predicts a *service class*: the kind of service the code names. This module defines that
bucketing, deterministically, for any number of classes n (L: 64, tiny: 8).

Definition (AS-106)
-------------------
- Port columns (name ends with `_port`): an ordered table of named classes (`PORT_CLASSES`, IANA
  registry numbers; first the planes' ports of AS-01, then common IT services), followed by four
  range classes at the end of the bucket range:
      n−4: port 0,   n−3: other well-known 1–1023,   n−2: registered 1024–49151,   n−1: dynamic 49152–65535
  (ranges per RFC 6335 §6). Named class i has bucket i when i < n − 4; otherwise it falls back to its
  range class. Codes outside [0, 65535] go to n−1.
- `flow.protocol`: an ordered table of IP protocol numbers (`PROTOCOL_CLASSES`, IANA protocol numbers);
  class i has bucket i when i < n − 1, every other protocol (and any overflow) goes to n − 1.
- Any other categorical column (DNS qtype, Modbus function code …): small codes are their own class,
  bucket = code for 0 ≤ code ≤ n − 2; every other code goes to n − 1 ("other").

The tables are part of the model's interface (append only: reordering changes trained heads).

Invariants
----------
- Every bucket lies in [0, n); the map is a pure function of (column name, code, n).
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import torch

from nagahana.governance.assumptions import assume

#: Named port classes (IANA Service Name and Transport Protocol Port Number Registry). Order = priority.
PORT_CLASSES: tuple[tuple[str, tuple[int, ...]], ...] = (
    ("http", (80, 8080, 8000, 8008, 8888)),
    ("https", (443, 8443)),
    ("dns", (53,)),
    ("ssh", (22,)),
    ("smb", (445, 139)),
    ("rdp", (3389,)),
    ("kerberos", (88, 464)),
    ("ldap", (389, 636, 3268, 3269)),
    ("netbios", (137, 138)),
    ("llmnr_mdns", (5355, 5353)),
    ("winrm", (5985, 5986)),
    ("telnet", (23,)),
    ("vnc", (5900, 5901)),
    ("modbus", (502,)),
    ("dnp3", (20000,)),
    ("iec104", (2404,)),
    ("s7", (102,)),
    ("enip", (44818, 2222)),
    ("bacnet", (47808,)),
    ("opcua", (4840,)),
    ("ftp", (20, 21)),
    ("smtp", (25, 465, 587)),
    ("pop3", (110, 995)),
    ("imap", (143, 993)),
    ("ntp", (123,)),
    ("snmp", (161, 162)),
    ("dhcp", (67, 68)),
    ("tftp", (69,)),
    ("msrpc", (135,)),
    ("syslog", (514,)),
    ("mssql", (1433, 1434)),
    ("mysql", (3306,)),
    ("postgres", (5432,)),
    ("oracle", (1521,)),
    ("redis", (6379,)),
    ("mongodb", (27017,)),
    ("ssdp", (1900,)),
    ("ipsec_ike", (500, 4500)),
    ("sip", (5060, 5061)),
    ("mqtt", (1883, 8883)),
    ("irc", (6667,)),
    ("socks_proxy", (1080, 3128)),
    ("elasticsearch", (9200,)),
    ("memcached", (11211,)),
    ("kubernetes_api", (6443, 10250)),
    ("docker_api", (2375, 2376)),
    ("vxlan_geneve", (4789, 6081)),
    ("radius", (1812, 1813)),
    ("nfs_rpcbind", (2049, 111)),
    ("x11", (6000,)),
)
#: IP protocol classes (IANA Assigned Internet Protocol Numbers). Order = priority.
PROTOCOL_CLASSES: tuple[tuple[str, int], ...] = (
    ("tcp", 6), ("udp", 17), ("icmp", 1), ("ipv6_icmp", 58), ("igmp", 2), ("gre", 47), ("esp", 50),
    ("ah", 51), ("sctp", 132), ("ospf", 89), ("ipv6_encap", 41), ("ipip", 4), ("pim", 103), ("vrrp", 112),
)
N_PORT_RANGE_CLASSES = 4


@lru_cache(maxsize=16)
def _port_table(n: int) -> torch.Tensor:
    """Bucket of every port 0…65535 for n classes (cached)."""
    if n < N_PORT_RANGE_CLASSES + 1:
        raise ValueError(f"port bucketing needs at least {N_PORT_RANGE_CLASSES + 1} classes")
    ports = np.arange(65536)
    # Range classes (RFC 6335 §6): 0 | 1–1023 | 1024–49151 | 49152–65535.
    table = np.where(ports == 0, n - 4, np.where(ports < 1024, n - 3, np.where(ports < 49152, n - 2, n - 1)))
    named = n - N_PORT_RANGE_CLASSES
    for i, (_, members) in enumerate(PORT_CLASSES):
        if i < named:
            table[list(members)] = i
    return torch.from_numpy(table.astype(np.int64))


@lru_cache(maxsize=16)
def _protocol_table(n: int) -> torch.Tensor:
    """Bucket of every IP protocol number 0…255 for n classes (cached)."""
    if n < 2:
        raise ValueError("protocol bucketing needs at least 2 classes")
    table = np.full(256, n - 1, dtype=np.int64)
    for i, (_, num) in enumerate(PROTOCOL_CLASSES):
        if i < n - 1:
            table[num] = i
    return torch.from_numpy(table)


def column_family(column_name: str) -> str:
    """'port' | 'protocol' | 'generic' for a categorical column name (bins and IDs alike)."""
    fid = column_name.split("[", 1)[0]
    if fid.endswith("_port"):
        return "port"
    if fid == "flow.protocol":
        return "protocol"
    return "generic"


def service_class(column_name: str, code: torch.Tensor, n_classes: int) -> torch.Tensor:
    """Bucket in [0, n_classes) of integer codes (any shape) of one categorical column (AS-106)."""
    assume("AS-33", by=__name__)
    code = code.to(torch.int64)
    family = column_family(column_name)
    if family == "port":
        inside = (code >= 0) & (code <= 65535)
        return torch.where(inside, _port_table(n_classes)[code.clamp(0, 65535)], torch.full_like(code, n_classes - 1))
    if family == "protocol":
        inside = (code >= 0) & (code <= 255)
        return torch.where(inside, _protocol_table(n_classes)[code.clamp(0, 255)], torch.full_like(code, n_classes - 1))
    small = (code >= 0) & (code <= n_classes - 2)
    return torch.where(small, code, torch.full_like(code, n_classes - 1))
