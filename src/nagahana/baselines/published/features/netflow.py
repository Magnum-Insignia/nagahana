"""NetFlow standard feature sets (Sarhan, Layeghy and Portmann, Mobile Networks and Applications 27(1),
2022, "Towards a Standard Feature Set for Network Intrusion Detection System Datasets", arXiv:2101.11315).

The paper proposes a NetFlow feature set of 12 features (the v1 datasets, NF-*) and an extended set of 43
features (the v2 datasets, NF-*-v2), exported with nProbe from the original captures. The v2 column
names below are the headers of the published NF-*-v2 CSV files; `Label` (0 benign, 1 attack) and
`Attack` (the class name) follow them.

Protocol of the paper's binary experiments (baselines-notes.md, B2): flow identifiers (addresses, ports,
time stamps) are removed, and for the UNSW-NB15 variants the TTL-based features as well; features are
min-max scaled; an extra-trees classifier is trained on random 70 %/30 % splits, averaged over five
splits.
"""

from __future__ import annotations

from collections.abc import Iterable

from nagahana.core.errors import InvariantViolation

#: The 12 features of the NetFlow v1 datasets (NF-UNSW-NB15, NF-CSE-CIC-IDS2018, ...).
NF_V1_FEATURES: tuple[str, ...] = (
    "IPV4_SRC_ADDR", "L4_SRC_PORT", "IPV4_DST_ADDR", "L4_DST_PORT", "PROTOCOL", "L7_PROTO",
    "IN_BYTES", "OUT_BYTES", "IN_PKTS", "OUT_PKTS", "TCP_FLAGS", "FLOW_DURATION_MILLISECONDS",
)

#: The 43 features of the NetFlow v2 datasets (NF-UNSW-NB15-v2, NF-CSE-CIC-IDS2018-v2, ...).
NF_V2_FEATURES: tuple[str, ...] = (
    "IPV4_SRC_ADDR", "L4_SRC_PORT", "IPV4_DST_ADDR", "L4_DST_PORT", "PROTOCOL", "L7_PROTO",
    "IN_BYTES", "IN_PKTS", "OUT_BYTES", "OUT_PKTS", "TCP_FLAGS", "CLIENT_TCP_FLAGS", "SERVER_TCP_FLAGS",
    "FLOW_DURATION_MILLISECONDS", "DURATION_IN", "DURATION_OUT", "MIN_TTL", "MAX_TTL",
    "LONGEST_FLOW_PKT", "SHORTEST_FLOW_PKT", "MIN_IP_PKT_LEN", "MAX_IP_PKT_LEN",
    "SRC_TO_DST_SECOND_BYTES", "DST_TO_SRC_SECOND_BYTES",
    "RETRANSMITTED_IN_BYTES", "RETRANSMITTED_IN_PKTS", "RETRANSMITTED_OUT_BYTES", "RETRANSMITTED_OUT_PKTS",
    "SRC_TO_DST_AVG_THROUGHPUT", "DST_TO_SRC_AVG_THROUGHPUT",
    "NUM_PKTS_UP_TO_128_BYTES", "NUM_PKTS_128_TO_256_BYTES", "NUM_PKTS_256_TO_512_BYTES",
    "NUM_PKTS_512_TO_1024_BYTES", "NUM_PKTS_1024_TO_1514_BYTES",
    "TCP_WIN_MAX_IN", "TCP_WIN_MAX_OUT", "ICMP_TYPE", "ICMP_IPV4_TYPE",
    "DNS_QUERY_ID", "DNS_QUERY_TYPE", "DNS_TTL_ANSWER", "FTP_COMMAND_RET_CODE",
)

#: Flow identifiers removed before training (addresses and ports; the NF files carry no time stamp).
NF_IDENTIFIERS: tuple[str, ...] = ("IPV4_SRC_ADDR", "L4_SRC_PORT", "IPV4_DST_ADDR", "L4_DST_PORT")
#: TTL-based features removed for the UNSW-NB15 variants (the only TTL fields of the v2 set).
NF_TTL_FEATURES: tuple[str, ...] = ("MIN_TTL", "MAX_TTL")
NF_LABEL = "Label"
NF_ATTACK = "Attack"

#: FlowTransformer's field selection for NetFlow v2 data ("unified flow format"): the v2 features
#: without the two addresses and without the DNS and FTP application fields, with nine fields treated
#: as categorical (AS-542; to verify against the official repository, tools/third_party).
FLOWTRANSFORMER_FIELDS: tuple[str, ...] = tuple(
    f for f in NF_V2_FEATURES
    if f not in ("IPV4_SRC_ADDR", "IPV4_DST_ADDR", "DNS_QUERY_ID", "DNS_QUERY_TYPE", "DNS_TTL_ANSWER", "FTP_COMMAND_RET_CODE")
)
FLOWTRANSFORMER_CATEGORICAL: tuple[str, ...] = (
    "CLIENT_TCP_FLAGS", "L4_SRC_PORT", "TCP_FLAGS", "ICMP_IPV4_TYPE", "ICMP_TYPE", "PROTOCOL", "SERVER_TCP_FLAGS",
    "L4_DST_PORT", "L7_PROTO",
)


def standard_features(version: int, *, drop_identifiers: bool = True, drop_ttl: bool = False) -> tuple[str, ...]:
    """The v1 or v2 feature list, without identifiers and/or TTL fields as the paper's protocol requires."""
    if version == 1:
        base = NF_V1_FEATURES
    elif version == 2:
        base = NF_V2_FEATURES
    else:
        raise ValueError("NetFlow feature-set version must be 1 or 2")
    drop: set[str] = set()
    if drop_identifiers:
        drop |= set(NF_IDENTIFIERS)
    if drop_ttl:
        drop |= set(NF_TTL_FEATURES)
    return tuple(f for f in base if f not in drop)


def check_columns(columns: Iterable[str], wanted: Iterable[str]) -> None:
    """Raise when a NetFlow frame lacks some of the wanted features."""
    have = set(columns)
    missing = [f for f in wanted if f not in have]
    if missing:
        raise InvariantViolation(f"NetFlow frame lacks features {missing}")
