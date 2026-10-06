"""CIC-IoT-2023 features and class groupings (Neto, Dadkhah, Ferreira, Zohourian, Lu and Ghorbani,
Sensors 23(13):5941, 2023, "CICIoT2023: A Real-Time Dataset and Benchmark for Large-Scale Attacks in IoT
Environment", DOI 10.3390/s23135941).

The published CSV files carry 46 features computed over windows of packets and a `label` column with 34
classes (33 attacks and benign). The paper reports three tasks: 34 classes, 8 classes (7 attack
categories and benign) and 2 classes (benign against attack). The column list and the 34 class names are
those of the released CSV files; the 34 -> 8 grouping is the grouping of the dataset's example notebook
published with the data (to verify against the files, AS-540).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nagahana.baselines.published.frames import normalise_token

#: The 46 feature columns of the CIC-IoT-2023 CSV files ("Magnitue" is the files' spelling).
CICIOT2023_FEATURES: tuple[str, ...] = (
    "flow_duration", "Header_Length", "Protocol Type", "Duration", "Rate", "Srate", "Drate",
    "fin_flag_number", "syn_flag_number", "rst_flag_number", "psh_flag_number", "ack_flag_number",
    "ece_flag_number", "cwr_flag_number", "ack_count", "syn_count", "fin_count", "urg_count", "rst_count",
    "HTTP", "HTTPS", "DNS", "Telnet", "SMTP", "SSH", "IRC", "TCP", "UDP", "DHCP", "ARP", "ICMP", "IPv", "LLC",
    "Tot sum", "Min", "Max", "AVG", "Std", "Tot size", "IAT", "Number", "Magnitue", "Radius", "Covariance",
    "Variance", "Weight",
)
CICIOT2023_LABEL = "label"
BENIGN_CLASS = "BenignTraffic"

#: The 34 classes of the released files -> the 8-class grouping.
CLASS_GROUP: dict[str, str] = {
    "DDoS-RSTFINFlood": "DDoS", "DDoS-PSHACK_Flood": "DDoS", "DDoS-SYN_Flood": "DDoS", "DDoS-UDP_Flood": "DDoS",
    "DDoS-TCP_Flood": "DDoS", "DDoS-ICMP_Flood": "DDoS", "DDoS-SynonymousIP_Flood": "DDoS",
    "DDoS-ACK_Fragmentation": "DDoS", "DDoS-UDP_Fragmentation": "DDoS", "DDoS-ICMP_Fragmentation": "DDoS",
    "DDoS-SlowLoris": "DDoS", "DDoS-HTTP_Flood": "DDoS",
    "DoS-UDP_Flood": "DoS", "DoS-SYN_Flood": "DoS", "DoS-TCP_Flood": "DoS", "DoS-HTTP_Flood": "DoS",
    "Mirai-greeth_flood": "Mirai", "Mirai-greip_flood": "Mirai", "Mirai-udpplain": "Mirai",
    "Recon-PingSweep": "Recon", "Recon-OSScan": "Recon", "Recon-PortScan": "Recon", "VulnerabilityScan": "Recon",
    "Recon-HostDiscovery": "Recon",
    "DNS_Spoofing": "Spoofing", "MITM-ArpSpoofing": "Spoofing",
    "BrowserHijacking": "Web", "Backdoor_Malware": "Web", "XSS": "Web", "Uploading_Attack": "Web",
    "SqlInjection": "Web", "CommandInjection": "Web",
    "DictionaryBruteForce": "BruteForce",
    "BenignTraffic": "Benign",
}
CLASSES_34: tuple[str, ...] = tuple(CLASS_GROUP)
CLASSES_8: tuple[str, ...] = ("Benign", "DDoS", "DoS", "Mirai", "Recon", "Spoofing", "Web", "BruteForce")
CLASSES_2: tuple[str, ...] = ("Benign", "Attack")
_BY_TOKEN: dict[str, str] = {normalise_token(k): k for k in CLASS_GROUP}


def task_classes(task: str) -> tuple[str, ...]:
    """Class names of a task: "binary", "multiclass-8" or "multiclass-34"."""
    if task == "binary":
        return CLASSES_2
    if task == "multiclass-8":
        return CLASSES_8
    if task == "multiclass-34":
        return CLASSES_34
    raise ValueError(f"unknown CIC-IoT-2023 task {task!r}")


def class_index(labels: pd.Series, task: str) -> np.ndarray:
    """Class code of each raw label for the task; labels outside the 34 known classes map to -1."""
    classes = task_classes(task)
    out = np.empty(len(labels), dtype=np.int64)
    for i, raw in enumerate(labels.tolist()):
        name = _BY_TOKEN.get(normalise_token(raw))
        if name is None:
            out[i] = -1
        elif task == "binary":
            out[i] = 0 if name == BENIGN_CLASS else 1
        elif task == "multiclass-8":
            out[i] = classes.index(CLASS_GROUP[name])
        else:
            out[i] = classes.index(name)
    return out
