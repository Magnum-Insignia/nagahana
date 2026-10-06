"""UNSW-NB15 feature sets (Moustafa and Slay, MilCIS 2015, "UNSW-NB15: a comprehensive data set for
network intrusion detection systems").

Two releases are in use:

    partitioned   UNSW_NB15_training-set.csv / UNSW_NB15_testing-set.csv: 175,341 and 82,332 records
                  (the official partition Vinayakumar et al. 2019 use; baselines-notes.md, A1), with an
                  `id`, 42 features, `attack_cat` and `label`
    full          UNSW-NB15_1.csv ... UNSW-NB15_4.csv: 2,540,044 records with 47 features including the
                  flow identifiers and time stamps, then `attack_cat` and `Label` (headerless files whose
                  column names are given in UNSW-NB15_features.csv)

The column names below are those of the released files. `proto`, `service` and `state` are
categorical. The TTL-based features are `sttl`, `dttl` and `ct_state_ttl` (the last is derived from the
TTL values), removed in Sarhan et al.'s protocol for UNSW-NB15 (AS-536).
"""

from __future__ import annotations

#: Columns of the partitioned training / testing CSV files, in file order.
UNSW_PARTITIONED_COLUMNS: tuple[str, ...] = (
    "id", "dur", "proto", "service", "state", "spkts", "dpkts", "sbytes", "dbytes", "rate", "sttl", "dttl",
    "sload", "dload", "sloss", "dloss", "sinpkt", "dinpkt", "sjit", "djit", "swin", "stcpb", "dtcpb", "dwin",
    "tcprtt", "synack", "ackdat", "smean", "dmean", "trans_depth", "response_body_len", "ct_srv_src",
    "ct_state_ttl", "ct_dst_ltm", "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm", "is_ftp_login",
    "ct_ftp_cmd", "ct_flw_http_mthd", "ct_src_ltm", "ct_srv_dst", "is_sm_ips_ports", "attack_cat", "label",
)
#: Columns of the full four-file release (header from UNSW-NB15_features.csv), in file order.
UNSW_FULL_COLUMNS: tuple[str, ...] = (
    "srcip", "sport", "dstip", "dsport", "proto", "state", "dur", "sbytes", "dbytes", "sttl", "dttl", "sloss",
    "dloss", "service", "Sload", "Dload", "Spkts", "Dpkts", "swin", "dwin", "stcpb", "dtcpb", "smeansz",
    "dmeansz", "trans_depth", "res_bdy_len", "Sjit", "Djit", "Stime", "Ltime", "Sintpkt", "Dintpkt", "tcprtt",
    "synack", "ackdat", "is_sm_ips_ports", "ct_state_ttl", "ct_flw_http_mthd", "is_ftp_login", "ct_ftp_cmd",
    "ct_srv_src", "ct_srv_dst", "ct_dst_ltm", "ct_src_ltm", "ct_src_dport_ltm", "ct_dst_sport_ltm",
    "ct_dst_src_ltm", "attack_cat", "Label",
)
UNSW_CATEGORICAL: tuple[str, ...] = ("proto", "service", "state")
UNSW_IDENTIFIERS: tuple[str, ...] = ("id", "srcip", "sport", "dstip", "dsport", "Stime", "Ltime")
UNSW_TTL_FEATURES: tuple[str, ...] = ("sttl", "dttl", "ct_state_ttl")
UNSW_LABELS: tuple[str, ...] = ("attack_cat", "label", "Label")
#: The ten classes of `attack_cat` ("Normal" first); the full release writes some names with trailing
#: spaces or as "Backdoors", which normalisation of the label text resolves.
UNSW_CLASSES: tuple[str, ...] = (
    "Normal", "Generic", "Exploits", "Fuzzers", "DoS", "Reconnaissance", "Analysis", "Backdoor", "Shellcode", "Worms",
)


def partitioned_features() -> tuple[str, ...]:
    """The 42 features of the partitioned files (no id, no labels)."""
    return tuple(c for c in UNSW_PARTITIONED_COLUMNS if c not in ("id", "attack_cat", "label"))


def full_features(*, drop_identifiers: bool = True, drop_ttl: bool = False) -> tuple[str, ...]:
    """Features of the full release without labels, optionally without identifiers and TTL fields."""
    drop = set(UNSW_LABELS)
    if drop_identifiers:
        drop |= set(UNSW_IDENTIFIERS)
    if drop_ttl:
        drop |= set(UNSW_TTL_FEATURES)
    return tuple(c for c in UNSW_FULL_COLUMNS if c not in drop)
