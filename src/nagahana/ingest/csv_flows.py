"""CSV flow adapters: CIC-IDS2017 / CSE-CIC-IDS2018 (CICFlowMeter), CTU-13 (Argus binetflow), CIC-IoT-2023.

Purpose
-------
The problem statement expects "a feature extraction pipeline that ingests CIC-IDS-2018 or CTU-13 CSV
flow records and/or raw PCAP files … and outputs a timestamped, normalised feature matrix covering
both flow-level and packet-level attributes". This module is the CSV half (the PCAP half is
`ingest/pcap.py`). Each adapter reads one CSV file and returns

    CSVRead.updates   `ColumnarUpdates` (datamodel/columnar.py): one row = one state update (D-30);
    CSVRead.labels    a separate table (seq, record, label_raw): the dataset authors' annotation of
                      each row. Labels never enter a `StateUpdate` (rule 4 below; models/batch.py).
                      Mapping a raw label to an ATT&CK stage is `nagahana.data.labels` (AS-34).
    CSVRead.stats     counts of everything the adapter refused to read as a value (and why).

Owner sources: problem statement (CLAUDE.md), D-30 (one record = one state update), D-41 (absence is
never zero), D-47 (group addresses), D-48 (entity identity), D-50 (no clock features in training).
Assumptions: AS-300 … AS-309 (`docs/assumptions/data.md`).

Rules every adapter follows
---------------------------
1. Map source columns to catalogue fields (`datamodel/fields.py`). Every field the source lacks is
   NOT_SUPPLIED (never 0). A value the source writes but that cannot be a measurement (NaN, ±inf,
   a negative duration, CICFlowMeter's −1 "no window" marker, an IAT statistic of a one-packet flow)
   is NOT_SUPPLIED too, and counted in `stats` (AS-302).
2. Attach provenance: source id, adapter name and version, and the SHA-256 of the raw CSV line (the
   bytes of the line without its terminator; `raw_offset` / `raw_len` locate it in the file).
3. Map identifiers (addresses, ports) to typed entities: initiator (`entity_0`), responder
   (`entity_1`) and, for TCP/UDP, the responder's service (`entity_2`, key `"<address>:<port>/<proto>"`,
   the PCAP adapter's keying). Kind: `multicast` for group destinations (D-47: IPv4 224.0.0.0/4,
   255.255.255.255, IPv6 ff00::/8), `host` inside the monitored network, `external` otherwise. The
   entity table carries an explicit `internal` column and a `synthetic` column.
4. Keep labels outside the state update.

Event time of a flow record (AS-300)
------------------------------------
A flow record summarises the whole flow (total bytes, duration, IAT statistics), and an exporter can
only emit it when the flow ends. Placing it at the flow's *start* would show the model, at time
t_start, facts that only exist at t_start + duration: future leakage. So

    event_time = flow_start + duration        (when the duration is a valid measurement)
    event_time = flow_start                   (otherwise; `time_basis` = 0 records which)

`flow_start` is kept as an audit column. Records are sorted by event time (stable on file order);
`seq` is the sorted position and `record` the data-row number in the file. The `watermark` is the
latest event time among the rows *before* this one in file order (what a live consumer would have
seen), and `reorder_uncertainty_s = max(timestamp resolution, watermark − event_time)`.

Ordering resolution per source: CIC-IDS2018 timestamps have 1 s resolution; some CIC-IDS2017 files
have minute resolution (60 s); CTU-13 has microseconds (AS-301).

Time zone (AS-301): CSV timestamps carry no zone. `utc_offset_hours` converts local time to UTC
(utc = local − offset). When it is None the text is read as UTC and `clock_quality` says
"timezone-unverified". Absolute clock time is not a model input in training (D-50), so the zone
matters only for aligning a CSV with captures or publisher tables.

CICFlowMeter column map (CIC-IDS2017 / CSE-CIC-IDS2018)
-------------------------------------------------------
Sources: the CICFlowMeter feature list published with the datasets
(https://www.unb.ca/cic/datasets/ids-2018.html and …/ids-2017.html), Sharafaldin, Habibi Lashkari &
Ghorbani, ICISSP 2018 ("Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic
Characterization"), and the CICFlowMeter source (`BasicFlow.java`). Items marked (unverified) were
written from memory of those sources and must be checked against the files in stage 1.

    CSV column (2018 / 2017 spelling)                 → field                       conversion
    Src IP / Source IP, Dst IP / Destination IP       → entity keys (flow.src_ip/dst_ip)
    Src Port / Source Port                            → flow.src_port               TCP/UDP only
    Dst Port / Destination Port                       → flow.dst_port               TCP/UDP only
    Protocol                                          → flow.protocol               IANA number; 0 → NOT_SUPPLIED (unverified: CICFlowMeter writes 0 when unknown)
    Timestamp                                         → flow_start (audit)          flow start (unverified: CICFlowMeter's Timestamp is the first packet's time)
    Flow Duration                                     → flow.duration               µs → s (÷ 1e6)
    Tot Fwd Pkts / Total Fwd Packets                  → flow.packets_fwd
    Tot Bwd Pkts / Total Backward Packets             → flow.packets_bwd
    TotLen Fwd Pkts / Total Length of Fwd Packets     → flow.payload_bytes_fwd      AS-303: CICFlowMeter sums payload bytes (unverified)
    TotLen Bwd Pkts / Total Length of Bwd Packets     → flow.payload_bytes_bwd      AS-303; flow.bytes_* (IP layer) stay NOT_SUPPLIED
    (derived) Tot Fwd Pkts + Tot Bwd Pkts             → flow.packets_total
    (derived) Tot Bwd Pkts = 0                        → flow.unanswered = 1, else 0   (AS-338: within CICFlowMeter's flow delimitation)
    flow.end_reason                                   → NOT_SUPPLIED: the files do not say how a flow ended (flag counts
                                                        show flags seen, not which packet ended the flow)
    Flow IAT Mean, Flow IAT Max                       → flow.iat_mean, iat_max      µs → s; needs ≥ 2 packets
    Flow IAT Std                                      → flow.iat_var                (std/1e6)² · (n−1)/n, n = packets − 1 IATs; needs ≥ 3 packets (AS-304)
    FIN/SYN/RST/PSH/ACK/URG Flag Cnt (… Count)        → flow.flag_count.*
    (the six counts + ECE Flag Cnt + CWE Flag Count)  → flow.tcp_flags              derived: bit set where count > 0; TCP only; only when all 8 counts are readable
    Init Fwd Win Byts / Init_Win_bytes_forward        → pkt.tcp_window_init_fwd     TCP only; −1 → NOT_SUPPLIED
    Init Bwd Win Byts / Init_Win_bytes_backward       → pkt.tcp_window_init_bwd     TCP only; −1 → NOT_SUPPLIED
    flow.bidir_ratio                                  → NOT_SUPPLIED: its definition (PCAP adapter) is a ratio of IP-layer bytes,
                                                        which these files do not carry (AS-303); not CICFlowMeter's "Down/Up Ratio"
    (derived, needs addresses)                        → derived.portscan_*          the PCAP adapter's definition, over records in arrival order (AS-305)
    Label                                             → labels table only

Initiator: CICFlowMeter flows are bidirectional and the forward direction is the direction of the
flow's first packet, so the source is the initiator (unverified wording; CICFlowMeter README).
All other CICFlowMeter columns (packet-length statistics, rates, bulk, subflow, active/idle) have no
catalogue field yet; they are listed in `stats["unmapped_columns"]`, not dropped silently.

Addresses (AS-306): the processed CSE-CIC-IDS2018 "TrafficForML" files carry no addresses except the
20-02-2018 file (to verify per file), and the CIC-IDS2017 "MachineLearningCSV" files carry none
(the "TrafficLabelling" files do). Without addresses each row gets its own two *synthetic* entities
(see "Synthetic entities" below).

CTU-13 binetflow column map (Garcia, Grill, Stiborek & Zunino, Computers & Security 45, 2014,
"An empirical comparison of botnet detection methods"; Argus `ra` field names)
--------------------------------------------------------------------------------------------------
    StartTime  "YYYY/MM/DD hh:mm:ss.ffffff"          → flow_start (audit)
    Dur        seconds                               → flow.duration
    Proto      name (tcp, udp, icmp, …)              → flow.protocol               IANA number via `PROTOCOL_NUMBERS`; unknown names → NOT_SUPPLIED
    SrcAddr, DstAddr                                 → entity keys                 the reported source is taken as initiator (Argus `Dir` kept as audit)
    Sport, Dport  decimal or 0x-hex                  → flow.src_port, dst_port     TCP/UDP only (ICMP type/code in these columns are not mapped; unverified layout)
    State      Argus state, e.g. "FSPA_FSPA"          → flow.tcp_flags              TCP only: letters F S R P A U E C on either side of "_" → bits
    TotBytes, SrcBytes                               → flow.bytes_fwd = SrcBytes, flow.bytes_bwd = TotBytes − SrcBytes (Argus byte definition unverified, AS-303)
    TotPkts                                          → flow.packets_total          (no per-direction packet counts)
    TotBytes = SrcBytes                              → flow.unanswered = 1, else 0 (no byte from the destination ⇒ no packet)
    flow.end_reason                                  → NOT_SUPPLIED (Argus State lists flags seen per side; which event
                                                       ended the flow, or whether Argus split it by its own timer, is unverified)
    sTos, dTos                                       → (no field)
    Label      "flow=From-Botnet-…" etc.             → labels table only
Internal network: CTU-13 hosts are in the CTU university network, public addresses 147.32.0.0/16
(to verify: the paper names infected hosts in 147.32.84.0/24); see `DATASET_INTERNAL_NETWORKS`.

CIC-IoT-2023 (Neto et al., Sensors 23(13):5941, 2023, "CICIoT2023: A Real-Time Dataset and
Benchmark for Large-Scale Attacks in IoT Environment")
-----------------------------------------------------------------------------------------
The public CSVs (to verify) carry 46 features computed over *windows of packets* (values averaged
over the window), a `label` column, and **no addresses, no ports and no timestamps**. What can be
mapped honestly (AS-307):

    Protocol Type     → flow.protocol         only when the value is an integer (an average of protocols is not a protocol)
    Duration          → pkt.ttl_mean          the paper's feature table defines "Duration" as the time-to-live (unverified)
    syn/ack/fin/urg/rst_count → flow.flag_count.*   only when integral (a window average is not a count)
    ts (if present)   → event time            otherwise event_time is NaN and `clock_quality` = "not-supplied"
Everything else is listed as unmapped. What a model can and cannot learn from these rows is in
`docs/assumptions/data.md` (AS-307): per-update field statistics, reconstruction of the mapped
fields and per-update malignity / stage readouts, yes; transitions, topology, timing, entity history
and forecasting, no. The dataset's PCAPs (to verify that they are published) through
`ingest/pcap.py` are the path for temporal learning on CIC-IoT-2023.

Synthetic entities (AS-306)
---------------------------
When a source has no addresses, `flow.src_ip` / `flow.dst_ip` are NOT_SUPPLIED and each row gets two
entities of kind `host` with keys `"synthetic:<source>:<record>:initiator"` / `":responder"` and
`synthetic = True`. They let the update stand in the 2-positions-per-update layout (AS-41), but they
carry no identity: they never recur, so they give no entity history, no transition target
(`next_index` = −1) and no shared hyperedges. `internal` for them is the adapter argument
`synthetic_internal` (no default: the caller states it from the dataset's documentation).

Extension points
----------------
- A new CICFlowMeter spelling: add the normalised name to `CIC_ALIASES`.
- A new dataset: subclass `_CSVAdapter`, fill `_extract`.
- New catalogue fields (e.g. payload bytes, total packets) make some unmapped columns mappable.
"""

from __future__ import annotations

import hashlib
import io
import ipaddress
import math
import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nagahana.core.errors import InvariantViolation
from nagahana.datamodel.columnar import (
    CODE_NOT_SUPPLIED,
    CODE_OBSERVED,
    ColumnarUpdates,
    finalise_tables,
)
from nagahana.datamodel.records import StateUpdate
from nagahana.ingest.pcap import COLUMNS as PCAP_COLUMNS
from nagahana.ingest.pcap import SCAN_MIN_PORTS, SCAN_WINDOW

ADAPTER_VERSION = "1.0.0"

#: Matrix columns of every CSV adapter: the PCAP adapter's layout, so the two kinds of source share
#: one column set (fields a CSV cannot carry stay NOT_SUPPLIED).
COLUMNS = PCAP_COLUMNS
_COL: dict[str, int] = {c.name: j for j, c in enumerate(COLUMNS)}
_N = len(COLUMNS)

#: Non-matrix fields: the addresses are entity keys (as in the PCAP adapter).
SIDE_FIELDS: dict[str, tuple[str | None, str]] = {
    "flow.src_ip": ("@entity_0", "src_ip_status"),
    "flow.dst_ip": ("@entity_1", "dst_ip_status"),
}

#: Monitored networks per dataset (AS-308). Each is from the dataset's documentation, to verify in
#: stage 1. CIC-IDS2017: the victim network 192.168.10.0/24 (the firewall's 172.16.0.1, which NATs
#: the external attacker, is deliberately *not* internal). CSE-CIC-IDS2018: the AWS VPC 172.31.0.0/16
#: (FACTS.md of the sample slice). CTU-13: the CTU university network.
DATASET_INTERNAL_NETWORKS: dict[str, tuple[str, ...]] = {
    "cic-ids2017": ("192.168.10.0/24",),
    "cic-ids2018": ("172.31.0.0/16",),
    "ctu13": ("147.32.0.0/16",),
}

#: The default internal rule when no networks are given: RFC 1918 private IPv4, RFC 4193 unique-local
#: IPv6, and the on-link ranges (link-local, loopback).
DEFAULT_INTERNAL_NETWORKS: tuple[str, ...] = (
    "10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16",         # RFC 1918
    "fc00::/7",                                              # RFC 4193 (ULA)
    "169.254.0.0/16", "fe80::/10",                           # link-local (RFC 3927, RFC 4291)
    "127.0.0.0/8", "::1/128",                                # loopback
)

#: Argus protocol names → IANA protocol numbers (IANA "Assigned Internet Protocol Numbers").
PROTOCOL_NUMBERS: dict[str, int] = {
    "icmp": 1, "igmp": 2, "ipv4": 4, "tcp": 6, "egp": 8, "udp": 17, "ipv6": 41, "rsvp": 46, "gre": 47,
    "esp": 50, "ah": 51, "ipv6-icmp": 58, "icmp6": 58, "pim": 103, "sctp": 132,
    # Argus labels RTP/RTCP flows by their payload; they travel over UDP (RFC 3550).
    "rtp": 17, "rtcp": 17, "udt": 17,
}
_PROTO_NAMES = {6: "tcp", 17: "udp", 132: "sctp"}

# TCP flag bits (RFC 9293 header layout; ECE/CWR RFC 3168)
_FIN, _SYN, _RST, _PSH, _ACK, _URG, _ECE, _CWR = 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80
_ARGUS_FLAG_BITS = {"F": _FIN, "S": _SYN, "R": _RST, "P": _PSH, "A": _ACK, "U": _URG, "E": _ECE, "C": _CWR}


def _norm(name: str) -> str:
    """Normalised column name: lower case, letters and digits only (" Flow IAT Mean" → "flowiatmean")."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


#: CICFlowMeter concept → normalised spellings (2018 "TrafficForML", 2017 "TrafficLabelling" /
#: "MachineLearningCSV").
CIC_ALIASES: dict[str, tuple[str, ...]] = {
    "src_ip": ("srcip", "sourceip"),
    "dst_ip": ("dstip", "destinationip"),
    "src_port": ("srcport", "sourceport"),
    "dst_port": ("dstport", "destinationport"),
    "protocol": ("protocol",),
    "timestamp": ("timestamp",),
    "duration_us": ("flowduration",),
    "pkts_fwd": ("totfwdpkts", "totalfwdpackets"),
    "pkts_bwd": ("totbwdpkts", "totalbackwardpackets"),
    "bytes_fwd": ("totlenfwdpkts", "totallengthoffwdpackets"),
    "bytes_bwd": ("totlenbwdpkts", "totallengthofbwdpackets"),
    "iat_mean_us": ("flowiatmean",),
    "iat_std_us": ("flowiatstd",),
    "iat_max_us": ("flowiatmax",),
    "fin": ("finflagcnt", "finflagcount"),
    "syn": ("synflagcnt", "synflagcount"),
    "rst": ("rstflagcnt", "rstflagcount"),
    "psh": ("pshflagcnt", "pshflagcount"),
    "ack": ("ackflagcnt", "ackflagcount"),
    "urg": ("urgflagcnt", "urgflagcount"),
    "ece": ("eceflagcnt", "eceflagcount"),
    "cwr": ("cweflagcount", "cweflagcnt", "cwrflagcount", "cwrflagcnt"),
    "win_fwd": ("initfwdwinbyts", "initwinbytesforward"),
    "win_bwd": ("initbwdwinbyts", "initwinbytesbackward"),
    "label": ("label",),
    "flow_id": ("flowid",),
}

CTU13_ALIASES: dict[str, tuple[str, ...]] = {
    "start": ("starttime",), "dur": ("dur",), "proto": ("proto",), "src": ("srcaddr",), "sport": ("sport",),
    "dir": ("dir",), "dst": ("dstaddr",), "dport": ("dport",), "state": ("state",), "stos": ("stos",),
    "dtos": ("dtos",), "totpkts": ("totpkts",), "totbytes": ("totbytes",), "srcbytes": ("srcbytes",),
    "label": ("label",),
}

CICIOT_ALIASES: dict[str, tuple[str, ...]] = {
    "protocol": ("protocoltype",), "ttl": ("duration",), "ts": ("ts", "timestamp"),
    "syn": ("syncount",), "ack": ("ackcount",), "fin": ("fincount",), "urg": ("urgcount",), "rst": ("rstcount",),
    "label": ("label",),
}

#: Timestamp formats tried, with the resolution (s) they imply (AS-301).
_TIME_FORMATS: tuple[tuple[str, float], ...] = (
    ("%d/%m/%Y %H:%M:%S", 1.0),          # CSE-CIC-IDS2018 "TrafficForML"
    ("%d/%m/%Y %I:%M:%S %p", 1.0),       # CIC-IDS2017 (some files, to verify)
    ("%d/%m/%Y %H:%M", 60.0),            # CIC-IDS2017 "TrafficLabelling" (minute resolution, to verify)
    ("%Y/%m/%d %H:%M:%S.%f", 1e-6),      # CTU-13 binetflow
    ("%Y-%m-%d %H:%M:%S.%f", 1e-6),
    ("%Y-%m-%d %H:%M:%S", 1.0),
)


# ====================================================================================== helpers
def is_internal_address(text: str, networks: Sequence[ipaddress.IPv4Network | ipaddress.IPv6Network]) -> bool | None:
    """True if `text` is an IP address inside `networks`; None if it is not an IP address."""
    try:
        ip = ipaddress.ip_address(text)
    except ValueError:
        return None
    return any(ip.version == n.version and ip in n for n in networks)


def parse_networks(cidrs: Sequence[str] | None) -> list[ipaddress.IPv4Network | ipaddress.IPv6Network]:
    """CIDR strings → networks (`DEFAULT_INTERNAL_NETWORKS` when None)."""
    return [ipaddress.ip_network(c) for c in (cidrs if cidrs is not None else DEFAULT_INTERNAL_NETWORKS)]


def _is_group(text: str) -> bool:
    """D-47 by value: IPv4 224.0.0.0/4 and 255.255.255.255, IPv6 ff00::/8. (Directed broadcasts need
    the link layer, which a flow CSV does not carry.)"""
    try:
        ip = ipaddress.ip_address(text)
    except ValueError:
        return False
    return bool(ip.is_multicast) or text == "255.255.255.255"


@dataclass
class CSVRead:
    """What an adapter returns: the state updates, the raw label table and the read statistics."""

    updates: ColumnarUpdates
    labels: pd.DataFrame
    stats: dict[str, Any] = field(default_factory=dict)


@dataclass
class _Raw:
    """The file's data lines, parsed as text, with their provenance."""

    frame: pd.DataFrame          # dtype str, one row per data line
    offsets: np.ndarray          # int64 byte offset of each line
    lengths: np.ndarray          # int32 byte length (terminator excluded)
    hashes: np.ndarray           # uint8 [n, 32] SHA-256 of each line
    header: list[str]


def _read_lines(path: Path, max_rows: int | None, stats: dict[str, Any]) -> _Raw:
    """Split the file into lines (keeping offsets), drop blank lines and repeated header rows, parse.

    Parsing goes through pandas on exactly the kept lines, so row i of the frame is line i of the
    hash table: provenance and values cannot drift apart (checked: same row count).
    """
    raw = path.read_bytes()
    lines: list[bytes] = []
    offsets: list[int] = []
    pos = 0
    header: bytes | None = None
    repeated = blank = 0
    for line in raw.split(b"\n"):
        start = pos
        pos += len(line) + 1
        body = line[:-1] if line.endswith(b"\r") else line
        if not body.strip():
            blank += 1
            continue
        if header is None:
            header = body
            continue
        if body == header:                       # CIC files repeat the header inside the data (stage-1 finding)
            repeated += 1
            continue
        lines.append(body)
        offsets.append(start)
        if max_rows is not None and len(lines) >= max_rows:
            stats["truncated"] = True
            break
    if header is None:
        raise InvariantViolation(f"{path.name}: empty CSV file.")
    stats.update(blank_lines=blank, repeated_header_rows=repeated)
    stats.setdefault("truncated", False)
    frame = pd.read_csv(
        io.BytesIO(b"\n".join([header, *lines])), dtype=str, keep_default_na=False, na_filter=False,
        encoding="utf-8", encoding_errors="replace", skipinitialspace=True,
    )
    if len(frame) != len(lines):
        raise InvariantViolation(f"{path.name}: parsed {len(frame)} rows from {len(lines)} lines (quoted newlines?).")
    sha = hashlib.sha256
    hashes = np.frombuffer(b"".join(sha(b).digest() for b in lines), dtype=np.uint8).reshape(len(lines), 32).copy()
    return _Raw(
        frame=frame, offsets=np.asarray(offsets, dtype=np.int64),
        lengths=np.asarray([len(b) for b in lines], dtype=np.int32), hashes=hashes,
        header=[str(c) for c in frame.columns],
    )


def _resolve(columns: Sequence[str], aliases: Mapping[str, tuple[str, ...]]) -> dict[str, str | None]:
    """Concept → actual column name (None when the file has no such column)."""
    by_norm: dict[str, str] = {}
    for c in columns:
        by_norm.setdefault(_norm(c), c)
    return {k: next((by_norm[a] for a in names if a in by_norm), None) for k, names in aliases.items()}


def _num(frame: pd.DataFrame, col: str | None) -> np.ndarray:
    """A column as float64 (unparseable → NaN; 'Infinity' → inf). An absent column is all NaN."""
    if col is None:
        return np.full(len(frame), np.nan)
    return pd.to_numeric(frame[col].str.strip(), errors="coerce").to_numpy(dtype=np.float64)


def _parse_time(text: pd.Series, stats: dict[str, Any]) -> tuple[np.ndarray, float]:
    """Text timestamps → epoch seconds as written (zone handled by the caller) and the resolution."""
    s = text.str.strip()
    best: tuple[int, np.ndarray, float, str] | None = None
    for fmt, res in _TIME_FORMATS:
        dt = pd.to_datetime(s, format=fmt, errors="coerce")
        ok = int(dt.notna().sum())
        if best is None or ok > best[0]:
            secs = ((dt - pd.Timestamp("1970-01-01")) / pd.Timedelta(seconds=1)).to_numpy(dtype=np.float64)
            best = (ok, secs, res, fmt)
        if ok == len(s):
            break
    assert best is not None
    stats["timestamp_format"] = best[3]
    stats["unparsed_timestamps"] = int(len(s) - best[0])
    return best[1], best[2]


class _Matrix:
    """Value/status matrices in the PCAP column layout, filled column by column (NaN / NOT_SUPPLIED by default)."""

    def __init__(self, n: int, stats: dict[str, Any]) -> None:
        self.values = np.full((n, _N), np.nan, dtype=np.float64)
        self.status = np.full((n, _N), CODE_NOT_SUPPLIED, dtype=np.uint8)
        self.stats = stats

    def set(self, field_id: str, values: np.ndarray, ok: np.ndarray | None = None) -> None:
        """Write `values` where `ok` (and the value is finite); everything else stays NOT_SUPPLIED.

        Cells that the source supplied but that are not a finite number are counted per field in
        `stats["refused_values"]` (D-41: they are absence, not zero).
        """
        j = _COL[field_id]
        keep = np.isfinite(values) if ok is None else (ok & np.isfinite(values))
        refused = int(((ok if ok is not None else np.ones(len(values), bool)) & ~np.isfinite(values)).sum())
        if refused:
            self.stats.setdefault("refused_values", {})[field_id] = self.stats.get("refused_values", {}).get(field_id, 0) + refused
        self.values[keep, j] = values[keep]
        self.status[keep, j] = CODE_OBSERVED

    def refuse(self, field_id: str, mask: np.ndarray, reason: str) -> None:
        """Count cells refused for a stated reason (they are already NOT_SUPPLIED)."""
        n = int(mask.sum())
        if n:
            key = f"{field_id}: {reason}"
            self.stats.setdefault("refused_values", {})[key] = self.stats.get("refused_values", {}).get(key, 0) + n


def _port_evidence(src: np.ndarray, dst: np.ndarray, dport: np.ndarray, ok: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Port-access evidence in the PCAP adapter's definition (pcap.py docstring), over records in order.

    Per (source, destination) entity pair, the last `SCAN_WINDOW` distinct destination ports in order
    of first appearance among the records seen so far (rows are already in event-time order, so a
    record only uses records that arrived no later: AS-305). With at least `SCAN_MIN_PORTS` ports,
        sequential = share of consecutive pairs that differ by exactly one port,
        random     = min(1, ports / SCAN_WINDOW) × (1 − sequential).
    Returns NaN where the evidence does not exist yet.
    """
    seq = np.full(len(src), np.nan)
    rnd = np.full(len(src), np.nan)
    memory: dict[tuple[int, int], tuple[list[int], set[int]]] = {}
    for i in np.nonzero(ok)[0].tolist():
        pair = (int(src[i]), int(dst[i]))
        order, known = memory.setdefault(pair, ([], set()))
        p = int(dport[i])
        if p not in known:
            known.add(p)
            order.append(p)
            if len(order) > SCAN_WINDOW:
                known.discard(order.pop(0))
        if len(order) >= SCAN_MIN_PORTS:
            steps = sum(1 for a, b in zip(order, order[1:], strict=False) if abs(a - b) == 1)
            share = steps / (len(order) - 1)
            seq[i] = share
            rnd[i] = min(1.0, len(order) / SCAN_WINDOW) * (1.0 - share)
    return seq, rnd


# ====================================================================================== base adapter
class _CSVAdapter:
    """Shared machinery: read lines, extract per-dataset columns, order, build entities, assemble.

    Subclasses implement `_extract(raw, stats)`, returning an `_Extract`.
    """

    name = "csv"
    dataset = ""
    implemented = True

    def __init__(
        self,
        path: str | Path,
        *,
        source_id: str | None = None,
        internal_networks: Sequence[str] | None = None,
        utc_offset_hours: float | None = None,
        max_rows: int | None = None,
        synthetic_internal: bool | None = None,
    ) -> None:
        self.path = Path(path)
        self.source_id = source_id or self.path.name
        self.internal_networks = tuple(internal_networks) if internal_networks is not None else None
        self._nets = parse_networks(self.internal_networks)
        self.utc_offset_hours = utc_offset_hours
        self.max_rows = max_rows
        self.synthetic_internal = synthetic_internal
        self.stats: dict[str, Any] = {}

    # ------------------------------------------------------------------ to implement
    def _extract(self, raw: _Raw, stats: dict[str, Any]) -> _Extract:
        raise NotImplementedError

    # ------------------------------------------------------------------ public API
    def updates(self) -> Iterator[StateUpdate]:
        """Yield the state updates as objects (`Source` protocol), in event-time order."""
        cu = self.read().updates
        for i in range(len(cu)):
            yield cu.update(i)

    def read(self) -> CSVRead:
        """Read the file: state updates, raw labels, statistics."""
        stats: dict[str, Any] = {"dataset": self.dataset, "internal_networks": self.internal_networks or DEFAULT_INTERNAL_NETWORKS}
        raw = _read_lines(self.path, self.max_rows, stats)
        ex = self._extract(raw, stats)
        n = len(raw.frame)
        if n == 0:
            raise InvariantViolation(f"{self.path.name}: no data rows.")

        # --- event time (AS-300) and ordering. Local → UTC: utc = local − offset.
        start = ex.start
        if self.utc_offset_hours is not None:
            start = start - self.utc_offset_hours * 3600.0
            clock = f"timezone=UTC{self.utc_offset_hours:+g} (converted to UTC)"
        else:
            clock = "timezone-unverified (read as UTC)" if np.isfinite(start).any() else "not-supplied"
        dur_ok = ex.matrix.status[:, _COL["flow.duration"]] == CODE_OBSERVED
        event = np.where(dur_ok & np.isfinite(start), start + np.nan_to_num(ex.matrix.values[:, _COL["flow.duration"]]), start)
        basis = np.where(~np.isfinite(start), -1, np.where(dur_ok, 1, 0)).astype(np.int8)
        # watermark: latest event time among earlier rows in file order (arrival order of a live feed)
        prior = np.concatenate([[np.nan], np.fmax.accumulate(event)[:-1]]) if n else np.zeros(0)
        reorder = np.where(np.isfinite(event), np.fmax(ex.resolution, np.fmax(prior - event, 0.0)), np.nan)
        # stable sort by event time; rows without a time keep file order at the end
        order = np.argsort(np.where(np.isfinite(event), event, np.inf), kind="stable")

        # --- entities (initiator, responder, service), in sorted order
        entity_index: dict[tuple[str, str], int] = {}
        internal: list[bool] = []
        synthetic: list[bool] = []
        e = np.full((n, 3), -1, dtype=np.int32)
        proto = ex.matrix.values[:, _COL["flow.protocol"]]
        dport = ex.matrix.values[:, _COL["flow.dst_port"]]
        port_ok = ex.matrix.status[:, _COL["flow.dst_port"]] == CODE_OBSERVED
        # rows without both addresses get synthetic entities (AS-306); their `internal` flag must be stated
        has_addr = (
            np.asarray([bool(a) and bool(b) for a, b in zip(ex.src, ex.dst, strict=True)])
            if ex.src is not None and ex.dst is not None else np.zeros(n, dtype=bool)
        )
        stats["rows_without_addresses"] = int((~has_addr).sum())
        if (~has_addr).any() and self.synthetic_internal is None:
            raise InvariantViolation(
                f"{self.path.name}: {int((~has_addr).sum())} rows have no addresses, so they get synthetic entities; "
                "pass synthetic_internal=True/False from the dataset's documentation (AS-306)."
            )

        def entity(kind: str, key: str, inside: bool, synth: bool) -> int:
            idx = entity_index.get((kind, key))
            if idx is None:
                idx = entity_index[(kind, key)] = len(entity_index)
                internal.append(inside)
                synthetic.append(synth)
            return idx

        kind_cache: dict[str, tuple[str, bool]] = {}

        def address(text: str) -> tuple[str, bool]:
            got = kind_cache.get(text)
            if got is None:
                inside = is_internal_address(text, self._nets)
                if _is_group(text):
                    got = ("multicast", False)
                elif inside is None:                     # not an IP address (e.g. a MAC in Argus ARP rows): on-link
                    got = ("host", True)
                else:
                    got = ("host" if inside else "external", inside)
                kind_cache[text] = got
            return got

        for out_i, i in enumerate(order.tolist()):
            if has_addr[i]:
                assert ex.src is not None and ex.dst is not None
                ka, ia = address(ex.src[i])
                kb, ib = address(ex.dst[i])
                e[out_i, 0] = entity(ka, ex.src[i], ia, False)
                e[out_i, 1] = entity(kb, ex.dst[i], ib, False)
                p = int(proto[i]) if np.isfinite(proto[i]) else -1
                if port_ok[i] and p in _PROTO_NAMES:
                    e[out_i, 2] = entity("service", f"{ex.dst[i]}:{int(dport[i])}/{_PROTO_NAMES[p]}", ib, False)
            else:
                rec = int(i)
                inside = bool(self.synthetic_internal)
                e[out_i, 0] = entity("host", f"synthetic:{self.source_id}:{rec}:initiator", inside, True)
                e[out_i, 1] = entity("host", f"synthetic:{self.source_id}:{rec}:responder", inside, True)
        stats["synthetic_entities"] = int(sum(synthetic))

        values = ex.matrix.values[order]
        status = ex.matrix.status[order]
        # --- port-access evidence needs real entities (AS-305)
        if ex.src is not None:
            ok = (status[:, _COL["flow.dst_port"]] == CODE_OBSERVED) & (e[:, 0] >= 0) & ~np.asarray(synthetic, bool)[np.maximum(e[:, 0], 0)]
            seq_e, rnd_e = _port_evidence(e[:, 0], e[:, 1], values[:, _COL["flow.dst_port"]], ok)
            has = np.isfinite(seq_e)
            for fid, arr in (("derived.portscan_sequential", seq_e), ("derived.portscan_random", rnd_e)):
                j = _COL[fid]
                values[has, j] = arr[has]
                status[has, j] = CODE_OBSERVED

        relation_index: dict[tuple[int, int, int], int] = {}
        rel = np.empty(n, dtype=np.int32)
        for k in range(n):
            key = (int(e[k, 0]), int(e[k, 1]), int(e[k, 2]))
            rel[k] = relation_index.setdefault(key, len(relation_index))

        ev_sorted = event[order]
        ip_status = np.where(has_addr[order], CODE_OBSERVED, CODE_NOT_SUPPLIED).astype(np.uint8)
        frame = pd.DataFrame({
            "seq": np.arange(n, dtype=np.int64),
            "record": order.astype(np.int64),
            "event_time": ev_sorted,
            "ingest_time": ev_sorted,                     # replayed on its own clock (as the PCAP adapter)
            "watermark": prior[order],
            "reorder_uncertainty_s": reorder[order],
            "entity_0": e[:, 0], "entity_1": e[:, 1], "entity_2": e[:, 2],
            "relation": rel,
            "direction": np.full(n, -1, dtype=np.int8),   # a flow record carries both directions
            "raw_offset": raw.offsets[order],
            "raw_len": raw.lengths[order],
            "flow_start": start[order],
            "time_basis": basis[order],                   # 1 flow end, 0 flow start, -1 no time (AS-300)
            "src_ip_status": ip_status,
            "dst_ip_status": ip_status,
            **{k: np.asarray(v)[order] for k, v in ex.audit.items()},
        })
        entities = pd.DataFrame({
            "kind": [k for k, _ in entity_index], "key": [key for _, key in entity_index],
            "internal": np.asarray(internal, dtype=bool), "synthetic": np.asarray(synthetic, dtype=bool),
        })
        explicit = tuple(sorted({*ex.explicit, *(("flow.src_ip", "flow.dst_ip") if ex.src is not None else ())}))
        cu = ColumnarUpdates(
            source_id=self.source_id, adapter=self.name, adapter_version=ADAPTER_VERSION, columns=COLUMNS,
            updates=frame, entities=entities, relations=pd.DataFrame(),
            values=np.ascontiguousarray(values), status=np.ascontiguousarray(status),
            raw_hash=raw.hashes[order], clock_quality=clock, side_fields=dict(SIDE_FIELDS), explicit_fields=explicit,
        )
        with np.errstate(invalid="ignore"):              # a source without times has NaN first/last seen
            finalise_tables(cu, relation_entities={v: k for k, v in relation_index.items()})
        # protocol and destination port of each relation, from its first update (as the PCAP adapter)
        first_row = pd.Series(np.arange(n)).groupby(rel).min().reindex(range(len(relation_index))).to_numpy()
        cu.relations["protocol"] = np.where(np.isfinite(values[first_row, _COL["flow.protocol"]]),
                                            values[first_row, _COL["flow.protocol"]], -1).astype(np.int32)
        cu.relations["dst_port"] = np.where(np.isfinite(values[first_row, _COL["flow.dst_port"]]),
                                            values[first_row, _COL["flow.dst_port"]], -1).astype(np.int32)
        cu.validate()

        labels = pd.DataFrame({
            "seq": np.arange(n, dtype=np.int64), "record": order.astype(np.int64),
            "label_raw": np.asarray(ex.label, dtype=object)[order] if ex.label is not None else np.full(n, "", dtype=object),
        })
        stats["rows"] = n
        stats["unmapped_columns"] = ex.unmapped
        stats["timestamp_resolution_s"] = ex.resolution
        self.stats = stats
        return CSVRead(updates=cu, labels=labels, stats=stats)


@dataclass
class _Extract:
    """Per-dataset extraction result (file order)."""

    matrix: _Matrix
    start: np.ndarray                  # float64 epoch seconds as written (NaN = none)
    resolution: float                  # timestamp resolution (s)
    src: list[str] | None              # initiator address per row ("" = missing), None = the file has none
    dst: list[str] | None
    label: list[str] | None
    explicit: tuple[str, ...]          # fields this adapter maps (present in every update, maybe NOT_SUPPLIED)
    unmapped: list[str]
    audit: dict[str, np.ndarray] = field(default_factory=dict)


# ====================================================================================== CICFlowMeter
class CICFlowSource(_CSVAdapter):
    """CIC-IDS2017 / CSE-CIC-IDS2018 CICFlowMeter CSV (one bidirectional flow per row). See the module docstring."""

    name = "cic-flows"
    dataset = "cic-ids"

    def _extract(self, raw: _Raw, stats: dict[str, Any]) -> _Extract:
        f = raw.frame
        col = _resolve(raw.header, CIC_ALIASES)
        stats["columns_found"] = {k: v for k, v in col.items() if v is not None}
        m = _Matrix(len(f), stats)
        proto = _num(f, col["protocol"])
        m.refuse("flow.protocol", proto == 0, "protocol 0 (CICFlowMeter unknown, AS-302)")
        m.set("flow.protocol", proto, proto > 0)
        tcp_udp = (proto == 6) | (proto == 17)
        tcp = proto == 6
        for fid, key in (("flow.src_port", "src_port"), ("flow.dst_port", "dst_port")):
            if col[key] is not None:
                m.set(fid, _num(f, col[key]), tcp_udp)

        # durations and IATs: µs → s; negative values are not measurements (AS-302)
        dur = _num(f, col["duration_us"]) / 1e6
        m.refuse("flow.duration", dur < 0, "negative duration")
        m.set("flow.duration", dur, dur >= 0)
        n_f, n_b = _num(f, col["pkts_fwd"]), _num(f, col["pkts_bwd"])
        m.set("flow.packets_fwd", n_f, n_f >= 0)
        m.set("flow.packets_bwd", n_b, n_b >= 0)
        pk = n_f + n_b                                   # NaN if either is missing
        b_f, b_b = _num(f, col["bytes_fwd"]), _num(f, col["bytes_bwd"])
        m.set("flow.payload_bytes_fwd", b_f, b_f >= 0)   # AS-303: CICFlowMeter's lengths are payload bytes (unverified)
        m.set("flow.payload_bytes_bwd", b_b, b_b >= 0)
        m.set("flow.packets_total", pk, pk >= 0)
        m.set("flow.unanswered", np.where(n_b == 0, 1.0, 0.0), n_b >= 0)   # AS-338
        im, ix, isd = _num(f, col["iat_mean_us"]) / 1e6, _num(f, col["iat_max_us"]) / 1e6, _num(f, col["iat_std_us"]) / 1e6
        two = pk >= 2
        m.refuse("flow.iat_mean", ~two & np.isfinite(im), "fewer than 2 packets (IAT undefined, AS-304)")
        m.set("flow.iat_mean", im, two & (im >= 0))
        m.set("flow.iat_max", ix, two & (ix >= 0))
        n_iat = pk - 1
        var = np.where(n_iat >= 2, isd * isd * (n_iat - 1) / np.maximum(n_iat, 1), np.nan)   # sample → population (AS-304)
        m.set("flow.iat_var", var, (pk >= 3) & (isd >= 0))
        # flag counts (TCP only) and the derived bitmask
        counts: dict[str, np.ndarray] = {}
        for flag in ("syn", "ack", "fin", "rst", "psh", "urg", "ece", "cwr"):
            counts[flag] = _num(f, col[flag])
            if flag in ("syn", "ack", "fin", "rst", "psh", "urg"):
                m.set(f"flow.flag_count.{flag}", counts[flag], tcp & (counts[flag] >= 0))
        bits = {"fin": _FIN, "syn": _SYN, "rst": _RST, "psh": _PSH, "ack": _ACK, "urg": _URG, "ece": _ECE, "cwr": _CWR}
        all_ok = tcp.copy()
        mask = np.zeros(len(f))
        for flag, bit in bits.items():
            c = counts[flag]
            all_ok &= np.isfinite(c) & (c >= 0)
            mask = mask + np.where(np.nan_to_num(c) > 0, bit, 0)
        m.set("flow.tcp_flags", mask, all_ok)
        for fid, key in (("pkt.tcp_window_init_fwd", "win_fwd"), ("pkt.tcp_window_init_bwd", "win_bwd")):
            w = _num(f, col[key])
            m.refuse(fid, tcp & (w < 0), "-1 marker (no window seen)")
            m.set(fid, w, tcp & (w >= 0))

        # time
        if col["timestamp"] is not None:
            start, res = _parse_time(f[col["timestamp"]], stats)
        else:
            start, res = np.full(len(f), np.nan), math.nan
        has_ip = col["src_ip"] is not None and col["dst_ip"] is not None
        src = f[col["src_ip"]].str.strip().tolist() if has_ip and col["src_ip"] else None
        dst = f[col["dst_ip"]].str.strip().tolist() if has_ip and col["dst_ip"] else None
        mapped = {v for v in col.values() if v is not None}
        explicit = (
            "flow.protocol", "flow.src_port", "flow.dst_port", "flow.duration", "flow.packets_fwd", "flow.packets_bwd",
            "flow.payload_bytes_fwd", "flow.payload_bytes_bwd", "flow.packets_total", "flow.iat_mean", "flow.iat_var", "flow.iat_max",
            "flow.tcp_flags", *(f"flow.flag_count.{x}" for x in ("syn", "ack", "fin", "rst", "psh", "urg")),
            "pkt.tcp_window_init_fwd", "pkt.tcp_window_init_bwd", "flow.unanswered",
        )
        return _Extract(
            matrix=m, start=start, resolution=res, src=src, dst=dst,
            label=f[col["label"]].str.strip().tolist() if col["label"] else None,
            explicit=explicit, unmapped=[c for c in raw.header if c not in mapped],
        )


# ====================================================================================== CTU-13
def _port(text: str) -> float:
    """Argus port text: decimal or 0x-hex; empty → NaN."""
    t = text.strip()
    if not t:
        return math.nan
    try:
        return float(int(t, 16)) if t.lower().startswith("0x") else float(int(t))
    except ValueError:
        return math.nan


def _argus_flags(state: str) -> float:
    """TCP flags seen on either side of an Argus state ("FSPA_FSPA" → FIN|SYN|PSH|ACK); NaN if not that form."""
    if "_" not in state:
        return math.nan
    bits = 0
    for side in state.split("_", 1):
        for ch in side:
            b = _ARGUS_FLAG_BITS.get(ch)
            if b is None:
                return math.nan
            bits |= b
    return float(bits)


class CTU13Source(_CSVAdapter):
    """CTU-13 bidirectional NetFlow (Argus binetflow) CSV. See the module docstring."""

    name = "ctu13-netflow"
    dataset = "ctu13"

    def _extract(self, raw: _Raw, stats: dict[str, Any]) -> _Extract:
        f = raw.frame
        col = _resolve(raw.header, CTU13_ALIASES)
        missing = [k for k in ("start", "dur", "proto", "src", "dst", "label") if col[k] is None]
        if missing:
            raise InvariantViolation(f"{self.path.name}: not a binetflow file (missing {missing}).")
        n = len(f)
        m = _Matrix(n, stats)
        names = f[col["proto"]].str.strip().str.lower() 
        proto = names.map(PROTOCOL_NUMBERS).to_numpy(dtype=np.float64, na_value=np.nan)
        unknown = names[np.isnan(proto)].value_counts().to_dict()
        stats["unknown_protocols"] = {str(k): int(v) for k, v in unknown.items()}
        m.set("flow.protocol", proto)
        tcp_udp = (proto == 6) | (proto == 17)
        if col["sport"] is not None:
            m.set("flow.src_port", f[col["sport"]].map(_port).to_numpy(dtype=np.float64), tcp_udp)
        if col["dport"] is not None:
            m.set("flow.dst_port", f[col["dport"]].map(_port).to_numpy(dtype=np.float64), tcp_udp)
        dur = _num(f, col["dur"])
        m.refuse("flow.duration", dur < 0, "negative duration")
        m.set("flow.duration", dur, dur >= 0)
        tot, srcb = _num(f, col["totbytes"]), _num(f, col["srcbytes"])
        bwd = tot - srcb
        m.refuse("flow.bytes_bwd", bwd < 0, "SrcBytes > TotBytes")
        m.set("flow.bytes_fwd", srcb, srcb >= 0)
        m.set("flow.bytes_bwd", bwd, (bwd >= 0) & (srcb >= 0))
        ratio = np.divide(bwd, srcb, out=np.full(n, np.nan), where=(srcb > 0) & (bwd >= 0))
        m.set("flow.bidir_ratio", ratio, (srcb > 0) & (bwd >= 0))
        if col["state"] is not None:
            m.set("flow.tcp_flags", f[col["state"]].map(_argus_flags).to_numpy(dtype=np.float64), proto == 6)
        tp = _num(f, col["totpkts"])
        m.set("flow.packets_total", tp, tp >= 0)
        m.set("flow.unanswered", np.where(bwd == 0, 1.0, 0.0), (bwd >= 0) & (srcb >= 0))   # AS-338
        start, res = _parse_time(f[col["start"]], stats)
        audit = {
            "argus_dir": f[col["dir"]].str.strip().to_numpy(dtype=object) if col["dir"] else np.full(n, "", dtype=object),
            "argus_state": f[col["state"]].str.strip().to_numpy(dtype=object) if col["state"] else np.full(n, "", dtype=object),
            "argus_proto": names.to_numpy(dtype=object),
        }
        mapped = {v for k, v in col.items() if v is not None and k not in ("stos", "dtos")}
        return _Extract(
            matrix=m, start=start, resolution=res,
            src=f[col["src"]].str.strip().tolist(), dst=f[col["dst"]].str.strip().tolist(),
            label=f[col["label"]].str.strip().tolist(),
            explicit=("flow.protocol", "flow.src_port", "flow.dst_port", "flow.duration", "flow.bytes_fwd",
                      "flow.bytes_bwd", "flow.bidir_ratio", "flow.tcp_flags", "flow.packets_total", "flow.unanswered"),
            unmapped=[c for c in raw.header if c not in mapped], audit=audit,
        )


# ====================================================================================== CIC-IoT-2023
class CICIoT2023Source(_CSVAdapter):
    """CIC-IoT-2023 CSV (packet-window features, no addresses or times). See the module docstring, AS-307."""

    name = "ciciot2023-csv"
    dataset = "ciciot2023"

    def _extract(self, raw: _Raw, stats: dict[str, Any]) -> _Extract:
        f = raw.frame
        col = _resolve(raw.header, CICIOT_ALIASES)
        if col["label"] is None:
            raise InvariantViolation(f"{self.path.name}: no label column; not a CIC-IoT-2023 CSV.")
        m = _Matrix(len(f), stats)

        def integral(x: np.ndarray) -> np.ndarray:
            return np.isfinite(x) & (np.abs(x - np.round(x)) < 1e-9)

        proto = _num(f, col["protocol"])
        m.refuse("flow.protocol", np.isfinite(proto) & ~integral(proto), "window average, not a protocol (AS-307)")
        m.set("flow.protocol", proto, integral(proto) & (proto >= 0))
        ttl = _num(f, col["ttl"])
        m.set("pkt.ttl_mean", ttl, ttl >= 0)              # "Duration" = TTL per the paper (unverified)
        tcp = proto == 6
        for flag in ("syn", "ack", "fin", "urg", "rst"):
            c = _num(f, col[flag])
            m.refuse(f"flow.flag_count.{flag}", np.isfinite(c) & ~integral(c), "window average, not a count (AS-307)")
            m.set(f"flow.flag_count.{flag}", c, tcp & integral(c) & (c >= 0))
        if col["ts"] is not None:
            start = _num(f, col["ts"])                    # epoch seconds if present (unverified unit)
            res = 1e-6
        else:
            start, res = np.full(len(f), np.nan), math.nan
        mapped = {v for v in col.values() if v is not None}
        return _Extract(
            matrix=m, start=start, resolution=res, src=None, dst=None,
            label=f[col["label"]].str.strip().tolist(),
            explicit=("flow.protocol", "pkt.ttl_mean", *(f"flow.flag_count.{x}" for x in ("syn", "ack", "fin", "urg", "rst"))),
            unmapped=[c for c in raw.header if c not in mapped],
        )


__all__ = [
    "CICFlowSource", "CICIoT2023Source", "COLUMNS", "CSVRead", "CTU13Source", "DATASET_INTERNAL_NETWORKS",
    "DEFAULT_INTERNAL_NETWORKS", "PROTOCOL_NUMBERS", "is_internal_address", "parse_networks",
]
