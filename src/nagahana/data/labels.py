"""Dataset labels → ATT&CK stage, technique slot and malicious flag (AS-34), plus the sample-slice labeller.

Purpose
-------
Public datasets label *attack families* per flow ("DoS attacks-Hulk", "flow=From-Botnet-V42-TCP-CC6-…"),
not ATT&CK steps. The model's heads need, per state update (models/batch.py `LabelBatch`):

    update_malicious   1 / 0 / NaN (unknown)
    update_stage       code of `vocab.STAGES` (0 = "none"), −1 = unknown
    update_technique   technique slot (AS-20) or −1

This module holds the explicit, reviewable mapping tables (AS-34) and turns a raw label table
(`ingest/csv_flows.CSVRead.labels`: seq, record, label_raw) into the full label table used by
`data/windows.py`:

    seq, record, label_raw, malicious (float32), stage (int64), technique (str ATT&CK ID or ""),
    family (coarse, dataset-independent), subfamily (dataset-level), actor_role (int8), mapped (bool)

`actor_role` says which entity of the update *performs* the labelled stage: 0 initiator, 1 responder,
−1 none (benign or unknown). It is what AS-18 needs: an entity is in an infiltration state when it is
the *internal actor* of an update whose stage is in `vocab.INFILTRATION_STAGES` (`data/windows.py`).
So a DoS from an external attacker (impact, actor external) never marks the victim as infiltrated,
while a bot's C2 beacon (actor = the internal bot) does.

Owner sources: problem statement (stage mapping to ATT&CK), D-17 / AS-25 (dataset annotations are
human-supplied truth). Assumptions: AS-34 (mapping exists and is explicit), AS-18 (infiltration),
AS-20 (technique slots), AS-310 … AS-316 (`docs/assumptions/data.md`).

Mapping principles (AS-310)
---------------------------
1. A label maps to the tactic of the *dominant activity* the dataset authors describe for it. Where
   the label is coarser than ATT&CK (a whole scenario under one label), the choice is written in
   the entry's `note` and flagged "approximate".
2. Techniques are given only where the activity clearly is one technique; otherwise "" (−1 slot).
3. Benign → stage 0 ("none"), malicious 0. A label that does not say benign or malicious (CTU-13
   "Background") → malicious NaN, stage −1: unknown is not benign (D-41 spirit applied to labels).
4. An unknown label string raises (`allow_unknown=False`): an incomplete table must be completed,
   not silently defaulted.

ATT&CK IDs used (Enterprise matrix, https://attack.mitre.org/techniques/enterprise/; names as of
ATT&CK v15–v16, to re-verify when the slot table is frozen):
T1110 Brute Force (.001 Password Guessing) · T1498 Network Denial of Service (.001 Direct Network
Flood) · T1499 Endpoint Denial of Service (.002 Service Exhaustion Flood) · T1190 Exploit
Public-Facing Application · T1046 Network Service Discovery · T1018 Remote System Discovery ·
T1595 Active Scanning (.001 Scanning IP Blocks, .002 Vulnerability Scanning) · T1071 Application
Layer Protocol (.001 Web Protocols, .004 DNS) · T1571 Non-Standard Port · T1496 Resource Hijacking ·
T1557 Adversary-in-the-Middle (.002 ARP Cache Poisoning) · T1185 Browser Session Hijacking ·
T1505.003 Web Shell · T1212 Exploitation for Credential Access · T1566.001 Spearphishing Attachment ·
T1203 Exploitation for Client Execution.

Technique slots (AS-20, AS-311)
-------------------------------
`KNOWN_TECHNIQUES` is an append-only table: technique i has slot i. Any other ID hashes into
[len(KNOWN_TECHNIQUES), n_slots) by SHA-256 (stable across processes, unlike Python's `hash`). The
table is small enough to fit the tiny preset (32 slots). It must be reconciled with the Forecaster
owner's technique table when that exists (requested change in the build report).

Sample PCAP slice (FACTS.md of nagahana-app/sample-data)
--------------------------------------------------------
`label_cic2018_infiltration_slice` labels the PCAP adapter's updates of the CSE-CIC-IDS2018
infiltration slice with rules taken from the measured facts (all UTC, 28 Feb 2018):
- victim 172.31.69.24 ↔ attacker 13.58.225.34 from 14:45:40 → command_and_control, T1571 (the
  backdoor session to port 31337); actor = the victim;
- TCP flows initiated by the victim to 172.31.69.1 … .23, from that target's first SYN in the
  FACTS table → discovery, T1046 (the Nmap scan); actor = the victim (initiator);
- the victim's traffic with 131.202.242.193 → benign (FACTS: "present all day; treat as benign
  background");
- any other update involving the victim from 14:45:40 → unknown (NaN): the victim is compromised
  and its other traffic is not annotated;
- everything else → benign (FACTS: 14:20:00–14:45:39 is benign only; other machines' traffic is
  treated as benign background, as the publisher's table lists no other attack that day).

Extension points: add a dataset = add a table (or a rule function) and register it in `MAPPERS`.
"""

from __future__ import annotations

import calendar
import hashlib
import math
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from nagahana.core.errors import InvariantViolation
from nagahana.datamodel.columnar import ColumnarUpdates
from nagahana.governance.assumptions import assume
from nagahana.models.vocab import STAGE_CODE

NAN = math.nan

#: Append-only ATT&CK technique table: index = technique slot (AS-20, AS-311). Never reorder.
KNOWN_TECHNIQUES: tuple[str, ...] = (
    "T1110", "T1110.001", "T1498", "T1498.001", "T1499", "T1499.002", "T1190", "T1046", "T1595",
    "T1595.001", "T1595.002", "T1071", "T1071.001", "T1071.004", "T1571", "T1496", "T1557", "T1557.002",
    "T1185", "T1505.003", "T1212", "T1018", "T1566.001", "T1203",
)
_TECH_INDEX = {t: i for i, t in enumerate(KNOWN_TECHNIQUES)}

#: Columns of a mapped label table, in order.
LABEL_COLUMNS: tuple[str, ...] = (
    "seq", "record", "label_raw", "malicious", "stage", "technique", "family", "subfamily", "actor_role", "mapped",
)


def technique_slot(technique: str, n_slots: int) -> int:
    """Slot of an ATT&CK technique ID: table index if known, else a stable hash into the spare slots; "" → −1."""
    assume("AS-20", by=__name__)
    if not technique:
        return -1
    k = len(KNOWN_TECHNIQUES)
    if n_slots <= k:
        raise InvariantViolation(f"{n_slots} technique slots cannot hold the {k} known techniques (AS-311).")
    if technique in _TECH_INDEX:
        return _TECH_INDEX[technique]
    h = int.from_bytes(hashlib.sha256(technique.encode()).digest()[:8], "big")
    return k + h % (n_slots - k)


def normalise_label(text: str) -> str:
    """Lower case, letters and digits only. Robust to the CIC files' spacing and encodings
    ("Web Attack \x96 Brute Force" in CIC-IDS2017 → "webattackbruteforce")."""
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


@dataclass(frozen=True)
class LabelSpec:
    """What one dataset label means for supervision.

    stage: a `vocab.STAGES` name, or "unknown" (stage −1). technique: ATT&CK ID or "".
    malicious: 1.0, 0.0 or NaN. family: coarse, dataset-independent family (used for class
    balance and known/novel splits). subfamily: dataset-level family. actor_role: 0 initiator,
    1 responder, −1 none. note: why, especially when "approximate".
    """

    stage: str
    technique: str
    malicious: float
    family: str
    subfamily: str
    actor_role: int
    note: str = ""


def _benign(sub: str = "benign") -> LabelSpec:
    return LabelSpec("none", "", 0.0, "benign", sub, -1)


def _m(stage: str, tech: str, family: str, sub: str, note: str = "", actor: int = 0) -> LabelSpec:
    return LabelSpec(stage, tech, 1.0, family, sub, actor, note)


_HTTP_FLOOD = "HTTP-layer flood exhausting the web service"
_NET_FLOOD = "volumetric flood of the network path"

#: CSE-CIC-IDS2018 labels (https://www.unb.ca/cic/datasets/ids-2018.html; spelling as in the CSVs,
#: including the dataset's own "Infilteration"; to verify per file in stage 1).
CIC_IDS2018: dict[str, LabelSpec] = {
    "benign": _benign(),
    "ftpbruteforce": _m("credential_access", "T1110.001", "bruteforce", "bruteforce-ftp", "Patator against FTP"),
    "sshbruteforce": _m("credential_access", "T1110.001", "bruteforce", "bruteforce-ssh", "Patator against SSH"),
    "dosattacksgoldeneye": _m("impact", "T1499.002", "dos", "dos-goldeneye", _HTTP_FLOOD),
    "dosattacksslowloris": _m("impact", "T1499.002", "dos", "dos-slowloris", "slow HTTP connection exhaustion"),
    "dosattacksslowhttptest": _m("impact", "T1499.002", "dos", "dos-slowhttptest", "slow HTTP connection exhaustion"),
    "dosattackshulk": _m("impact", "T1499.002", "dos", "dos-hulk", _HTTP_FLOOD),
    "ddosattacksloichttp": _m("impact", "T1499.002", "ddos", "ddos-loic-http", _HTTP_FLOOD),
    "ddosattackloicudp": _m("impact", "T1498.001", "ddos", "ddos-loic-udp", _NET_FLOOD),
    "ddosattackhoic": _m("impact", "T1499.002", "ddos", "ddos-hoic", _HTTP_FLOOD),
    "bruteforceweb": _m("credential_access", "T1110", "web-attack", "web-bruteforce", "password brute force on a web login"),
    "bruteforcexss": _m("initial_access", "T1190", "web-attack", "web-xss", "approximate: XSS against a public web app"),
    "sqlinjection": _m("initial_access", "T1190", "web-attack", "web-sqli", "SQL injection against a public web app"),
    "infilteration": _m("discovery", "T1046", "infiltration", "infiltration",
                        "approximate: the scenario is phishing → backdoor → Nmap sweep from the victim; the "
                        "flow-level label covers the victim's flows in the attack window, dominated by the "
                        "internal scan; label noise documented by Liu et al., IEEE CNS 2022"),
    "bot": _m("command_and_control", "T1071.001", "botnet", "botnet-zeus-ares", "Zeus / Ares bots polling HTTP C2"),
}
CIC_IDS2018["infiltration"] = CIC_IDS2018["infilteration"]

#: CIC-IDS2017 labels (https://www.unb.ca/cic/datasets/ids-2017.html; to verify per file).
CIC_IDS2017: dict[str, LabelSpec] = {
    "benign": _benign(),
    "ftppatator": _m("credential_access", "T1110.001", "bruteforce", "bruteforce-ftp", "Patator against FTP"),
    "sshpatator": _m("credential_access", "T1110.001", "bruteforce", "bruteforce-ssh", "Patator against SSH"),
    "dosslowloris": _m("impact", "T1499.002", "dos", "dos-slowloris", "slow HTTP connection exhaustion"),
    "dosslowhttptest": _m("impact", "T1499.002", "dos", "dos-slowhttptest", "slow HTTP connection exhaustion"),
    "doshulk": _m("impact", "T1499.002", "dos", "dos-hulk", _HTTP_FLOOD),
    "dosgoldeneye": _m("impact", "T1499.002", "dos", "dos-goldeneye", _HTTP_FLOOD),
    "heartbleed": _m("credential_access", "T1212", "heartbleed", "heartbleed",
                     "approximate: reads server memory (keys, credentials) through a TLS bug"),
    "webattackbruteforce": _m("credential_access", "T1110", "web-attack", "web-bruteforce", "password brute force on a web login"),
    "webattackxss": _m("initial_access", "T1190", "web-attack", "web-xss", "approximate: XSS against a public web app"),
    "webattacksqlinjection": _m("initial_access", "T1190", "web-attack", "web-sqli", "SQL injection against a public web app"),
    "infiltration": _m("discovery", "T1046", "infiltration", "infiltration",
                       "approximate: malicious download → internal port scan from the victim"),
    "bot": _m("command_and_control", "T1071.001", "botnet", "botnet-ares", "Ares bot polling HTTP C2"),
    "portscan": _m("reconnaissance", "T1595", "portscan", "portscan", "scan of the victim network from outside"),
    "ddos": _m("impact", "T1498", "ddos", "ddos-loic", "approximate: LOIC flood; layer not resolved by the label"),
}

_RECON_NOTE = ("approximate: the attacking devices sit in the testbed; 'reconnaissance' does not assert "
               "that the scanning device was compromised (AS-312)")
#: CIC-IoT-2023 classes (Neto et al., Sensors 2023; 33 attacks + benign; spelling to verify).
CICIOT2023: dict[str, LabelSpec] = {
    "benigntraffic": _benign(),
    **{f"ddos{k}": _m("impact", "T1498.001", "ddos", f"ddos-{k}", _NET_FLOOD) for k in (
        "rstfinflood", "pshackflood", "synflood", "udpflood", "tcpflood", "icmpflood", "synonymousipflood",
        "ackfragmentation", "udpfragmentation", "icmpfragmentation")},
    "ddosslowloris": _m("impact", "T1499.002", "ddos", "ddos-slowloris", "slow HTTP connection exhaustion"),
    "ddoshttpflood": _m("impact", "T1499.002", "ddos", "ddos-httpflood", _HTTP_FLOOD),
    **{f"dos{k}": _m("impact", "T1498.001", "dos", f"dos-{k}", _NET_FLOOD) for k in ("udpflood", "synflood", "tcpflood")},
    "doshttpflood": _m("impact", "T1499.002", "dos", "dos-httpflood", _HTTP_FLOOD),
    **{f"mirai{k}": _m("impact", "T1498.001", "mirai", f"mirai-{k}", "Mirai flood") for k in ("greethflood", "greipflood", "udpplain")},
    "reconpingsweep": _m("reconnaissance", "T1595.001", "recon", "recon-pingsweep", _RECON_NOTE),
    "reconhostdiscovery": _m("reconnaissance", "T1595.001", "recon", "recon-hostdiscovery", _RECON_NOTE),
    "reconosscan": _m("reconnaissance", "T1595", "recon", "recon-osscan", _RECON_NOTE),
    "reconportscan": _m("reconnaissance", "T1595", "recon", "recon-portscan", _RECON_NOTE),
    "vulnerabilityscan": _m("reconnaissance", "T1595.002", "recon", "recon-vulnscan", _RECON_NOTE),
    "sqlinjection": _m("initial_access", "T1190", "web-attack", "web-sqli", "SQL injection against a web app"),
    "commandinjection": _m("initial_access", "T1190", "web-attack", "web-cmdi", "command injection against a web app"),
    "xss": _m("initial_access", "T1190", "web-attack", "web-xss", "approximate: XSS against a web app"),
    "uploadingattack": _m("initial_access", "T1190", "web-attack", "web-upload", "malicious upload to a web app"),
    "browserhijacking": _m("collection", "T1185", "web-attack", "web-browserhijack", "approximate"),
    "backdoormalware": _m("persistence", "T1505.003", "web-attack", "web-backdoor", "approximate: backdoor planted through the web app"),
    "dictionarybruteforce": _m("credential_access", "T1110.001", "bruteforce", "bruteforce-dictionary", "dictionary attack"),
    "mitmarpspoofing": _m("credential_access", "T1557.002", "spoofing", "spoofing-arp", "ARP cache poisoning (AitM)"),
    "dnsspoofing": _m("credential_access", "T1557", "spoofing", "spoofing-dns", "approximate: DNS spoofing as AitM"),
}


def ctu13_spec(label: str) -> LabelSpec | None:
    """CTU-13 binetflow label → spec, by the label's own tokens (Garcia et al. 2014 labelling scheme).

    "flow=From-Botnet-V42-TCP-CC6-HTTP-Not-Encrypted" → command_and_control, T1071.001, actor initiator.
    Rules, first match wins (AS-313, approximate where noted):
        to-botnet …        actor = responder (the bot receives), else initiator
        cc token           command_and_control; T1071.001 if "http" else T1071
        spam               impact, T1496 (approximate: the bot's resources are hijacked to send spam)
        ddos               impact, T1498
        scan / portscan    discovery, T1046 (the internal bot scans)
        click / clickfraud impact, T1496 (approximate)
        p2p                command_and_control, T1071 (approximate)
        dns                command_and_control, T1071.004 (approximate: the bot's name resolution)
        (other botnet)     command_and_control, no technique
        normal             benign
        background         unknown (malicious NaN, stage unknown): the authors did not label it
    """
    s = label.strip()
    if s.lower().startswith("flow="):
        s = s[5:]
    tokens = [t for t in re.split(r"[-_ ]+", s.lower()) if t]
    if "botnet" in tokens:
        actor = 1 if tokens[:2] == ["to", "botnet"] else 0
        sub = "botnet"
        if any(re.fullmatch(r"cc\d*", t) for t in tokens):
            return _m("command_and_control", "T1071.001" if "http" in tokens else "T1071", "botnet", sub, "C2 channel", actor)
        if "spam" in tokens:
            return _m("impact", "T1496", "botnet", sub, "approximate: spam sending", actor)
        if "ddos" in tokens:
            return _m("impact", "T1498", "botnet", sub, "DDoS from the bot", actor)
        if "scan" in tokens or "portscan" in tokens:
            return _m("discovery", "T1046", "botnet", sub, "scan from the bot", actor)
        if "click" in tokens or "clickfraud" in tokens:
            return _m("impact", "T1496", "botnet", sub, "approximate: click fraud", actor)
        if "p2p" in tokens:
            return _m("command_and_control", "T1071", "botnet", sub, "approximate: P2P C2", actor)
        if "dns" in tokens:
            return _m("command_and_control", "T1071.004", "botnet", sub, "approximate: bot DNS", actor)
        return _m("command_and_control", "", "botnet", sub, "approximate: botnet flow, activity not in the label", actor)
    if "normal" in tokens:
        return _benign("normal")
    if "background" in tokens:
        return LabelSpec("unknown", "", NAN, "background", "background", -1, "not labelled by the authors")
    return None


def _table_mapper(table: Mapping[str, LabelSpec]) -> Callable[[str], LabelSpec | None]:
    return lambda label: table.get(normalise_label(label))


#: Dataset name → label mapper. Names match `ingest/csv_flows` datasets plus the 2017/2018 split.
MAPPERS: dict[str, Callable[[str], LabelSpec | None]] = {
    "cic-ids2018": _table_mapper(CIC_IDS2018),
    "cic-ids2017": _table_mapper(CIC_IDS2017),
    "ciciot2023": _table_mapper(CICIOT2023),
    "ctu13": ctu13_spec,
}


def map_labels(raw: pd.DataFrame, dataset: str, *, allow_unknown: bool = False) -> pd.DataFrame:
    """Raw label table (seq, record, label_raw) → the full label table (`LABEL_COLUMNS`). See the module docstring."""
    assume("AS-34", by=__name__)
    if dataset not in MAPPERS:
        raise KeyError(f"no label mapping for dataset {dataset!r}; known: {sorted(MAPPERS)}")
    mapper = MAPPERS[dataset]
    distinct = pd.unique(raw["label_raw"].astype(str))
    specs: dict[str, LabelSpec | None] = {lab: mapper(lab) for lab in distinct}
    unknown = sorted(lab for lab, s in specs.items() if s is None)
    if unknown and not allow_unknown:
        raise InvariantViolation(f"{dataset}: labels with no mapping (complete the table, AS-34): {unknown[:20]}")
    unk = LabelSpec("unknown", "", NAN, "unknown", "unknown", -1)
    labs = raw["label_raw"].astype(str)
    rows = [specs[lab] or unk for lab in labs]
    out = pd.DataFrame({
        "seq": raw["seq"].to_numpy(dtype=np.int64),
        "record": raw["record"].to_numpy(dtype=np.int64) if "record" in raw else raw["seq"].to_numpy(dtype=np.int64),
        "label_raw": labs.to_numpy(dtype=object),
        "malicious": np.asarray([s.malicious for s in rows], dtype=np.float32),
        "stage": np.asarray([STAGE_CODE.get(s.stage, -1) for s in rows], dtype=np.int64),
        "technique": np.asarray([s.technique for s in rows], dtype=object),
        "family": np.asarray([s.family for s in rows], dtype=object),
        "subfamily": np.asarray([s.subfamily for s in rows], dtype=object),
        "actor_role": np.asarray([s.actor_role for s in rows], dtype=np.int8),
        "mapped": np.asarray([specs[lab] is not None for lab in labs], dtype=bool),
    })
    return out


def check_label_table(labels: pd.DataFrame, n_updates: int) -> None:
    """Invariants of a mapped label table: one row per update, codes in range, benign ⇒ stage 0."""
    missing = [c for c in LABEL_COLUMNS if c not in labels]
    if missing:
        raise InvariantViolation(f"label table lacks columns {missing}")
    seq = labels["seq"].to_numpy()
    if len(labels) != n_updates or not np.array_equal(np.sort(seq), np.arange(n_updates)):
        raise InvariantViolation("label table must have exactly one row per update (seq 0 … n−1)")
    mal = labels["malicious"].to_numpy(dtype=np.float64)
    if not np.all(np.isnan(mal) | (mal == 0) | (mal == 1)):
        raise InvariantViolation("malicious must be 0, 1 or NaN")
    stage = labels["stage"].to_numpy()
    if ((stage < -1) | (stage >= len(STAGE_CODE))).any():
        raise InvariantViolation("stage code out of range")
    if ((mal == 0) & (stage != 0)).any():
        raise InvariantViolation("a benign update must have stage 0 ('none')")


# ====================================================================================== sample PCAP slice
def _utc(h: int, m: int, s: int) -> float:
    """Epoch seconds of 28 Feb 2018 h:m:s UTC (FACTS.md times are UTC)."""
    return float(calendar.timegm((2018, 2, 28, h, m, s)))


SLICE_VICTIM = "172.31.69.24"
SLICE_ATTACKER = "13.58.225.34"
SLICE_BACKDOOR_START = _utc(14, 45, 40)
#: FACTS.md: "the busiest external peer of the victim … present all day; treat as benign background".
SLICE_BENIGN_PEERS: frozenset[str] = frozenset({"131.202.242.193"})
#: First SYN of the scan per target (FACTS.md table, second resolution).
SLICE_SCAN_FIRST_SYN: dict[str, float] = {
    f"172.31.69.{k}": _utc(*t) for k, t in {
        1: (14, 46, 22), 4: (14, 46, 50), 5: (14, 47, 53), 6: (14, 48, 56), 7: (14, 50, 4), 8: (14, 50, 20),
        9: (14, 51, 21), 10: (14, 52, 29), 11: (14, 53, 42), 12: (14, 54, 44), 13: (14, 55, 56), 14: (14, 57, 7),
        15: (15, 0, 40), 16: (15, 0, 54), 17: (15, 1, 58), 18: (15, 3, 8), 19: (15, 3, 24), 20: (15, 4, 31),
        21: (15, 5, 41), 22: (15, 5, 45), 23: (15, 6, 0),
    }.items()
}


def label_cic2018_infiltration_slice(cu: ColumnarUpdates) -> pd.DataFrame:
    """Rule-based labels for the PCAP adapter's updates of the CSE-CIC-IDS2018 infiltration slice.

    Rules in the module docstring. Returns a table with `LABEL_COLUMNS`; `label_raw` names the rule
    ("backdoor-c2", "internal-scan", "victim-unannotated", "benign").
    """
    assume("AS-34", by=__name__)
    keys = cu.entities["key"].astype(str).to_numpy()
    e0 = cu.updates["entity_0"].to_numpy()
    e1 = cu.updates["entity_1"].to_numpy()
    ini = np.where(e0 >= 0, keys[np.maximum(e0, 0)], "")
    rsp = np.where(e1 >= 0, keys[np.maximum(e1, 0)], "")
    t = cu.updates["event_time"].to_numpy(dtype=np.float64)
    pcol = [j for j, c in enumerate(cu.columns) if c.name == "flow.protocol"]
    proto = cu.values[:, pcol[0]] if pcol else np.full(len(cu), np.nan)
    after = t >= SLICE_BACKDOOR_START
    c2 = after & (((ini == SLICE_VICTIM) & (rsp == SLICE_ATTACKER)) | ((ini == SLICE_ATTACKER) & (rsp == SLICE_VICTIM)))
    first_syn = np.asarray([SLICE_SCAN_FIRST_SYN.get(r, np.inf) for r in rsp])
    scan = (ini == SLICE_VICTIM) & (proto == 6) & (t >= np.floor(first_syn)) & ~c2
    victim = (ini == SLICE_VICTIM) | (rsp == SLICE_VICTIM)
    benign_peer = np.isin(ini, list(SLICE_BENIGN_PEERS)) | np.isin(rsp, list(SLICE_BENIGN_PEERS))
    unannotated = after & victim & ~c2 & ~scan & ~benign_peer
    n = len(cu)
    rule = np.full(n, "benign", dtype=object)
    rule[c2], rule[scan], rule[unannotated] = "backdoor-c2", "internal-scan", "victim-unannotated"
    mal = np.zeros(n, dtype=np.float32)
    mal[c2 | scan] = 1.0
    mal[unannotated] = np.nan
    stage = np.zeros(n, dtype=np.int64)
    stage[c2] = STAGE_CODE["command_and_control"]
    stage[scan] = STAGE_CODE["discovery"]
    stage[unannotated] = -1
    tech = np.full(n, "", dtype=object)
    tech[c2], tech[scan] = "T1571", "T1046"
    family = np.where(c2 | scan, "infiltration", np.where(unannotated, "unknown", "benign")).astype(object)
    actor = np.full(n, -1, dtype=np.int8)
    actor[c2] = np.where(ini[c2] == SLICE_VICTIM, 0, 1)
    actor[scan] = 0
    return pd.DataFrame({
        "seq": cu.updates["seq"].to_numpy(dtype=np.int64), "record": cu.updates["record"].to_numpy(dtype=np.int64),
        "label_raw": rule, "malicious": mal, "stage": stage, "technique": tech, "family": family,
        "subfamily": family.copy(), "actor_role": actor, "mapped": np.ones(n, dtype=bool),
    })


__all__ = [
    "CICIOT2023", "CIC_IDS2017", "CIC_IDS2018", "KNOWN_TECHNIQUES", "LABEL_COLUMNS", "LabelSpec", "MAPPERS",
    "check_label_table", "ctu13_spec", "label_cic2018_infiltration_slice", "map_labels", "normalise_label",
    "technique_slot",
]
