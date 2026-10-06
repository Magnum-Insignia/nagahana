"""CICFlowMeter feature columns of CIC-IDS2017 and CSE-CIC-IDS2018, and their harmonisation.

The two datasets were exported by two versions of CICFlowMeter with different column spellings for
the same quantities: CIC-IDS2017 writes "Total Fwd Packets", "Init_Win_bytes_forward" (with leading
spaces in most headers), CSE-CIC-IDS2018 writes "Tot Fwd Pkts", "Init Fwd Win Byts". A cross-dataset
study (Cantone et al. 2024) or any model trained on one and tested on the other needs one column space;
`harmonise` renames every column to the CSE-CIC-IDS2018 spelling (the canonical names here), matching
names after normalisation (lower case, letters and digits only).

Sources: the column headers of the published CSV files (CIC-IDS2017 MachineLearningCSV and
TrafficLabelling; CSE-CIC-IDS2018 TrafficForML) and the CICFlowMeter feature list on the datasets' pages
(https://www.unb.ca/cic/datasets/ids-2017.html, .../ids-2018.html). The 2017 files repeat the column
"Fwd Header Length" (pandas reads the second copy as "Fwd Header Length.1"); the copy is dropped.

Some CSE-CIC-IDS2018 day files repeat the header line inside the data; `clean` removes such rows.
"""

from __future__ import annotations

import pandas as pd

from nagahana.baselines.published.frames import normalise_token

#: CSE-CIC-IDS2018 TrafficForML columns (canonical names), in file order; `Label` last.
CIC2018_COLUMNS: tuple[str, ...] = (
    "Dst Port", "Protocol", "Timestamp", "Flow Duration", "Tot Fwd Pkts", "Tot Bwd Pkts", "TotLen Fwd Pkts",
    "TotLen Bwd Pkts", "Fwd Pkt Len Max", "Fwd Pkt Len Min", "Fwd Pkt Len Mean", "Fwd Pkt Len Std", "Bwd Pkt Len Max",
    "Bwd Pkt Len Min", "Bwd Pkt Len Mean", "Bwd Pkt Len Std", "Flow Byts/s", "Flow Pkts/s", "Flow IAT Mean",
    "Flow IAT Std", "Flow IAT Max", "Flow IAT Min", "Fwd IAT Tot", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max",
    "Fwd IAT Min", "Bwd IAT Tot", "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min", "Fwd PSH Flags",
    "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags", "Fwd Header Len", "Bwd Header Len", "Fwd Pkts/s",
    "Bwd Pkts/s", "Pkt Len Min", "Pkt Len Max", "Pkt Len Mean", "Pkt Len Std", "Pkt Len Var", "FIN Flag Cnt",
    "SYN Flag Cnt", "RST Flag Cnt", "PSH Flag Cnt", "ACK Flag Cnt", "URG Flag Cnt", "CWE Flag Count", "ECE Flag Cnt",
    "Down/Up Ratio", "Pkt Size Avg", "Fwd Seg Size Avg", "Bwd Seg Size Avg", "Fwd Byts/b Avg", "Fwd Pkts/b Avg",
    "Fwd Blk Rate Avg", "Bwd Byts/b Avg", "Bwd Pkts/b Avg", "Bwd Blk Rate Avg", "Subflow Fwd Pkts",
    "Subflow Fwd Byts", "Subflow Bwd Pkts", "Subflow Bwd Byts", "Init Fwd Win Byts", "Init Bwd Win Byts",
    "Fwd Act Data Pkts", "Fwd Seg Size Min", "Active Mean", "Active Std", "Active Max", "Active Min", "Idle Mean",
    "Idle Std", "Idle Max", "Idle Min", "Label",
)

#: Identifier columns of the files that carry them (the 20-02-2018 file and the 2017 TrafficLabelling files).
CIC_IDENTIFIERS: tuple[str, ...] = ("Flow ID", "Src IP", "Src Port", "Dst IP", "Dst Port", "Timestamp")
CIC_LABEL = "Label"

#: CIC-IDS2017 spelling (normalised) -> canonical CSE-CIC-IDS2018 name.
_CIC2017_TO_2018: dict[str, str] = {
    "flowid": "Flow ID", "sourceip": "Src IP", "sourceport": "Src Port", "destinationip": "Dst IP",
    "destinationport": "Dst Port", "protocol": "Protocol", "timestamp": "Timestamp",
    "flowduration": "Flow Duration", "totalfwdpackets": "Tot Fwd Pkts", "totalbackwardpackets": "Tot Bwd Pkts",
    "totallengthoffwdpackets": "TotLen Fwd Pkts", "totallengthofbwdpackets": "TotLen Bwd Pkts",
    "fwdpacketlengthmax": "Fwd Pkt Len Max", "fwdpacketlengthmin": "Fwd Pkt Len Min",
    "fwdpacketlengthmean": "Fwd Pkt Len Mean", "fwdpacketlengthstd": "Fwd Pkt Len Std",
    "bwdpacketlengthmax": "Bwd Pkt Len Max", "bwdpacketlengthmin": "Bwd Pkt Len Min",
    "bwdpacketlengthmean": "Bwd Pkt Len Mean", "bwdpacketlengthstd": "Bwd Pkt Len Std",
    "flowbytess": "Flow Byts/s", "flowpacketss": "Flow Pkts/s", "flowiatmean": "Flow IAT Mean",
    "flowiatstd": "Flow IAT Std", "flowiatmax": "Flow IAT Max", "flowiatmin": "Flow IAT Min",
    "fwdiattotal": "Fwd IAT Tot", "fwdiatmean": "Fwd IAT Mean", "fwdiatstd": "Fwd IAT Std", "fwdiatmax": "Fwd IAT Max",
    "fwdiatmin": "Fwd IAT Min", "bwdiattotal": "Bwd IAT Tot", "bwdiatmean": "Bwd IAT Mean", "bwdiatstd": "Bwd IAT Std",
    "bwdiatmax": "Bwd IAT Max", "bwdiatmin": "Bwd IAT Min", "fwdpshflags": "Fwd PSH Flags", "bwdpshflags": "Bwd PSH Flags",
    "fwdurgflags": "Fwd URG Flags", "bwdurgflags": "Bwd URG Flags", "fwdheaderlength": "Fwd Header Len",
    "bwdheaderlength": "Bwd Header Len", "fwdpacketss": "Fwd Pkts/s", "bwdpacketss": "Bwd Pkts/s",
    "minpacketlength": "Pkt Len Min", "maxpacketlength": "Pkt Len Max", "packetlengthmean": "Pkt Len Mean",
    "packetlengthstd": "Pkt Len Std", "packetlengthvariance": "Pkt Len Var", "finflagcount": "FIN Flag Cnt",
    "synflagcount": "SYN Flag Cnt", "rstflagcount": "RST Flag Cnt", "pshflagcount": "PSH Flag Cnt",
    "ackflagcount": "ACK Flag Cnt", "urgflagcount": "URG Flag Cnt", "cweflagcount": "CWE Flag Count",
    "eceflagcount": "ECE Flag Cnt", "downupratio": "Down/Up Ratio", "averagepacketsize": "Pkt Size Avg",
    "avgfwdsegmentsize": "Fwd Seg Size Avg", "avgbwdsegmentsize": "Bwd Seg Size Avg",
    "fwdavgbytesbulk": "Fwd Byts/b Avg", "fwdavgpacketsbulk": "Fwd Pkts/b Avg", "fwdavgbulkrate": "Fwd Blk Rate Avg",
    "bwdavgbytesbulk": "Bwd Byts/b Avg", "bwdavgpacketsbulk": "Bwd Pkts/b Avg", "bwdavgbulkrate": "Bwd Blk Rate Avg",
    "subflowfwdpackets": "Subflow Fwd Pkts", "subflowfwdbytes": "Subflow Fwd Byts",
    "subflowbwdpackets": "Subflow Bwd Pkts", "subflowbwdbytes": "Subflow Bwd Byts",
    "initwinbytesforward": "Init Fwd Win Byts", "initwinbytesbackward": "Init Bwd Win Byts",
    "actdatapktfwd": "Fwd Act Data Pkts", "minsegsizeforward": "Fwd Seg Size Min",
    "activemean": "Active Mean", "activestd": "Active Std", "activemax": "Active Max", "activemin": "Active Min",
    "idlemean": "Idle Mean", "idlestd": "Idle Std", "idlemax": "Idle Max", "idlemin": "Idle Min", "label": "Label",
}

#: Every known spelling (normalised) -> canonical name, both datasets.
CANONICAL: dict[str, str] = {**{normalise_token(c): c for c in (*CIC2018_COLUMNS, *CIC_IDENTIFIERS)}, **_CIC2017_TO_2018}
#: Normalised names of duplicated columns that carry no information of their own.
_DUPLICATES: frozenset[str] = frozenset({"fwdheaderlength1"})

#: The CICFlowMeter features shared by both datasets, without identifiers and label (canonical names).
CIC_COMMON_FEATURES: tuple[str, ...] = tuple(c for c in CIC2018_COLUMNS if c not in (*CIC_IDENTIFIERS, CIC_LABEL))


def harmonise(frame: pd.DataFrame) -> pd.DataFrame:
    """Rename CIC-IDS2017 / CSE-CIC-IDS2018 columns to the canonical names; drop duplicated columns.

    Columns with no known spelling keep their name (stripped of surrounding spaces).
    """
    rename: dict[str, str] = {}
    drop: list[str] = []
    for col in frame.columns:
        key = normalise_token(col)
        if key in _DUPLICATES:
            drop.append(col)
            continue
        rename[col] = CANONICAL.get(key, str(col).strip())
    out = frame.drop(columns=drop).rename(columns=rename)
    if out.columns.duplicated().any():
        dup = out.columns[out.columns.duplicated()].tolist()
        raise ValueError(f"harmonised columns collide: {dup}")
    return out


def clean(frame: pd.DataFrame, *, label: str = CIC_LABEL) -> pd.DataFrame:
    """Drop rows that repeat the header line inside the data (label cell equal to the label's name)."""
    if label not in frame.columns:
        return frame
    repeated = frame[label].astype(str).str.strip().str.lower() == label.lower()
    return frame.loc[~repeated].reset_index(drop=True)


def feature_columns(frame_columns: list[str] | tuple[str, ...], *, drop: tuple[str, ...] = CIC_IDENTIFIERS) -> tuple[str, ...]:
    """Canonical feature columns present in a harmonised frame, without `drop` and the label."""
    present = set(frame_columns)
    return tuple(c for c in CIC2018_COLUMNS if c in present and c not in drop and c != CIC_LABEL)
