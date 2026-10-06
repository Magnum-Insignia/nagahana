"""CTU-13 connection records (Garcia, Grill, Stiborek and Zunino, Computers and Security 45, 2014, "An
empirical comparison of botnet detection methods").

Two record formats carry the dataset's traffic:

    binetflow   Argus bidirectional flows with the authors' labels: StartTime, Dur, Proto, SrcAddr, Sport,
                Dir, DstAddr, Dport, State, sTos, dTos, TotPkts, TotBytes, SrcBytes, Label
    conn.log    Zeek connection logs (Ongun et al. 2019 work from Zeek logs of the captures): ts, uid,
                id.orig_h, id.orig_p, id.resp_h, id.resp_p, proto, service, duration, orig_bytes,
                resp_bytes, conn_state, local_orig, local_resp, missed_bytes, history, orig_pkts,
                orig_ip_bytes, resp_pkts, resp_ip_bytes, ...

`connections` maps either to one schema (CONNECTION_COLUMNS). A quantity a format does not carry is NaN
(binetflow has no per-direction packet counts; Zeek writes "-" for unset values), never zero.

Infected hosts per scenario (INFECTED_HOSTS) follow the CTU-13 scenario descriptions published with the
dataset (to verify against the scenario pages, AS-539). They give the coarse labels Ongun et al. use for
Neris: all traffic of a botnet address is malicious.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nagahana.baselines.published.frames import epoch_seconds
from nagahana.core.errors import InvariantViolation

#: The normalised connection schema.
CONNECTION_COLUMNS: tuple[str, ...] = (
    "time", "src", "sport", "dst", "dport", "proto", "duration", "orig_bytes", "resp_bytes", "orig_pkts",
    "resp_pkts", "tot_pkts", "state", "label",
)

#: Internal network of the CTU-13 captures (the CTU university network; ingest/csv_flows.py, AS-308).
CTU_INTERNAL_NETWORK = "147.32.0.0/16"

_TEN = ("147.32.84.165", "147.32.84.191", "147.32.84.192", "147.32.84.193", "147.32.84.204", "147.32.84.205",
        "147.32.84.206", "147.32.84.207", "147.32.84.208", "147.32.84.209")
_THREE = ("147.32.84.165", "147.32.84.191", "147.32.84.192")
_ONE = ("147.32.84.165",)

#: Scenario -> (botnet family, infected addresses).
INFECTED_HOSTS: dict[int, tuple[str, tuple[str, ...]]] = {
    1: ("Neris", _ONE), 2: ("Neris", _ONE), 3: ("Rbot", _ONE), 4: ("Rbot", _ONE), 5: ("Virut", _ONE),
    6: ("Menti", _ONE), 7: ("Sogou", _ONE), 8: ("Murlo", _ONE), 9: ("Neris", _TEN), 10: ("Rbot", _TEN),
    11: ("Rbot", _THREE), 12: ("NSIS.ay", _THREE), 13: ("Virut", _ONE),
}


def _port(value: object) -> float:
    # Ports are decimal or 0x-hex text in binetflow; ICMP rows carry type/code text, which is not a port.
    text = str(value).strip()
    if not text or text in ("-", "nan", "None"):
        return np.nan
    try:
        return float(int(text, 0))
    except ValueError:
        return np.nan


def _num(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series.replace({"-": np.nan, "(empty)": np.nan}), errors="coerce").to_numpy(dtype=np.float64)


def connections(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalise a binetflow or Zeek conn.log frame to CONNECTION_COLUMNS (format detected by its columns)."""
    cols = set(frame.columns)
    if {"StartTime", "SrcAddr", "DstAddr"} <= cols:
        tot_bytes, src_bytes = _num(frame["TotBytes"]), _num(frame["SrcBytes"])
        out = pd.DataFrame({
            "time": epoch_seconds(frame["StartTime"]),
            "src": frame["SrcAddr"].astype(str).str.strip(),
            "sport": [_port(v) for v in frame["Sport"].tolist()],
            "dst": frame["DstAddr"].astype(str).str.strip(),
            "dport": [_port(v) for v in frame["Dport"].tolist()],
            "proto": frame["Proto"].astype(str).str.strip().str.lower(),
            "duration": _num(frame["Dur"]),
            "orig_bytes": src_bytes,
            "resp_bytes": tot_bytes - src_bytes,
            "orig_pkts": np.nan,
            "resp_pkts": np.nan,
            "tot_pkts": _num(frame["TotPkts"]),
            "state": frame["State"].astype(str).str.strip(),
            "label": frame["Label"].astype(str) if "Label" in cols else "",
        })
    elif {"ts", "id.orig_h", "id.resp_h"} <= cols:
        orig_pkts = _num(frame["orig_pkts"]) if "orig_pkts" in cols else np.full(len(frame), np.nan)
        resp_pkts = _num(frame["resp_pkts"]) if "resp_pkts" in cols else np.full(len(frame), np.nan)
        label_col = next((c for c in ("label", "Label", "detailed-label") if c in cols), None)
        out = pd.DataFrame({
            "time": _num(frame["ts"]),
            "src": frame["id.orig_h"].astype(str).str.strip(),
            "sport": [_port(v) for v in frame["id.orig_p"].tolist()],
            "dst": frame["id.resp_h"].astype(str).str.strip(),
            "dport": [_port(v) for v in frame["id.resp_p"].tolist()],
            "proto": frame["proto"].astype(str).str.strip().str.lower(),
            "duration": _num(frame["duration"]) if "duration" in cols else np.nan,
            "orig_bytes": _num(frame["orig_bytes"]) if "orig_bytes" in cols else np.nan,
            "resp_bytes": _num(frame["resp_bytes"]) if "resp_bytes" in cols else np.nan,
            "orig_pkts": orig_pkts,
            "resp_pkts": resp_pkts,
            "tot_pkts": orig_pkts + resp_pkts,
            "state": frame["conn_state"].astype(str).str.strip() if "conn_state" in cols else "",
            "label": frame[label_col].astype(str) if label_col else "",
        })
    elif set(CONNECTION_COLUMNS) - {"label"} <= cols:
        out = frame.loc[:, [c for c in CONNECTION_COLUMNS if c in cols]].copy()
        if "label" not in out.columns:
            out["label"] = ""
    else:
        raise InvariantViolation("frame is neither binetflow, Zeek conn.log nor the normalised connection schema")
    if not np.all(np.isfinite(out["time"].to_numpy(dtype=np.float64))):
        raise InvariantViolation("connection times must be finite")
    for passthrough in ("scenario", "dataset", "network", "split"):
        if passthrough in cols and passthrough not in out.columns:
            out[passthrough] = frame[passthrough].to_numpy()
    return out.reset_index(drop=True)


def infected(scenario: int) -> tuple[str, ...]:
    """Infected addresses of a CTU-13 scenario."""
    if scenario not in INFECTED_HOSTS:
        raise KeyError(f"CTU-13 has scenarios 1 ... 13, not {scenario}")
    return INFECTED_HOSTS[scenario][1]
