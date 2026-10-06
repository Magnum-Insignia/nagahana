"""LANL 2015 "Comprehensive, Multi-Source Cyber-Security Events" (Kent, Los Alamos National Laboratory,
2015): authentication and red-team records, as EULER (King and Huang, NDSS 2022) and the Hopper-style
path detector consume them.

Record formats of the released files (comma-separated, no header; time in seconds from the start of the
collection, the first second being 1):

    auth.txt      time, source user@domain, destination user@domain, source computer,
                  destination computer, authentication type, logon type, authentication orientation,
                  success/failure
    redteam.txt   time, user@domain, source computer, destination computer

A red-team record marks the authentication event with the same time, source user, source computer and
destination computer as a compromise event (King and Huang label anomalous edges this way).

`read_auth` streams the very large auth.txt in chunks and keeps only a time range, so a snapshot range
can be read without loading the whole file.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

AUTH_COLUMNS: tuple[str, ...] = (
    "time", "src_user", "dst_user", "src", "dst", "auth_type", "logon_type", "orientation", "outcome",
)
REDTEAM_COLUMNS: tuple[str, ...] = ("time", "user", "src", "dst")


def read_auth(path: str | Path, *, start: float | None = None, end: float | None = None, success_only: bool = False,
              chunksize: int = 5_000_000) -> pd.DataFrame:
    """Authentication events with start <= time < end, optionally successful ones only."""
    parts: list[pd.DataFrame] = []
    reader = pd.read_csv(path, header=None, names=list(AUTH_COLUMNS), chunksize=chunksize, dtype=str,
                         compression="infer")
    for chunk in reader:
        t = pd.to_numeric(chunk["time"], errors="coerce").to_numpy(dtype=np.float64)
        keep = np.isfinite(t)
        if start is not None:
            keep &= t >= start
        if end is not None:
            keep &= t < end
        if success_only:
            keep &= chunk["outcome"].str.strip().str.lower().eq("success").to_numpy()
        if keep.any():
            part = chunk.loc[keep].copy()
            part["time"] = t[keep]
            parts.append(part)
        if end is not None and t.size and np.nanmin(t) >= end:
            break                                    # the file is sorted by time: nothing later can match
    if not parts:
        return pd.DataFrame({c: pd.Series(dtype=np.float64 if c == "time" else object) for c in AUTH_COLUMNS})
    return pd.concat(parts, ignore_index=True)


def read_redteam(path: str | Path) -> pd.DataFrame:
    """Red-team compromise events."""
    out = pd.read_csv(path, header=None, names=list(REDTEAM_COLUMNS), dtype=str, compression="infer")
    out["time"] = pd.to_numeric(out["time"], errors="raise").astype(np.float64)
    return out


def label_auth(auth: pd.DataFrame, redteam: pd.DataFrame) -> np.ndarray:
    """1 for authentication events matched by a red-team record (time, user, source, destination), else 0."""
    key_cols = ["time", "src_user", "src", "dst"]
    rt = redteam.rename(columns={"user": "src_user"})[key_cols].drop_duplicates()
    rt["_rt"] = 1
    merged = auth[key_cols].merge(rt, on=key_cols, how="left")
    return merged["_rt"].fillna(0).to_numpy(dtype=np.int64)
