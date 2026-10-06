"""Derived fields: values computed from other fields of the same record, as the mapping tables' notes state.

Each function reads contributing values from a draft and writes derived fields through
`mapping.set_field`, which never overwrites a value the source supplied itself and counts refusals.
A derived value inherits the weakest status of its inputs: from an OBSERVED input it is OBSERVED; a
derived value that cannot be complete (TCP flags rebuilt from the letters of a Zeek history, which
does not encode PSH and URG) is LOW_RELIABILITY with a stated reliability.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter

from nagahana.datamodel.status import ObservationStatus
from nagahana.ingest.core import EntityResolver, IngestStats, UpdateDraft
from nagahana.ingest.mapping import set_field

LOW = ObservationStatus.LOW_RELIABILITY

#: Rule identities from (generator, signature) stay exact in a float64 matrix while gid < 2^20.
_GID_LIMIT = 1 << 20
_HASH_BASE = 1 << 52


def rule_signature(gid: int, sid: int) -> int:
    """alert.signature of a numbered rule: gid * 2^32 + sid (AS-687); exact below 2^52."""
    if 0 <= gid < _GID_LIMIT and 0 <= sid < (1 << 32):
        return gid * (1 << 32) + sid
    return name_signature("rule", f"{gid}:{sid}")


def name_signature(namespace: str, name: str) -> int:
    """alert.signature of a named rule (a Zeek notice, a weird, an anomaly event, a CEF class id).

    A SHA-256-derived code in [2^52, 2^53): integers there are exact in float64 and never collide with
    the numbered-rule codes below 2^52 (AS-687).
    """
    h = int.from_bytes(hashlib.sha256(f"{namespace}:{name}".encode()).digest()[:8], "big")
    return _HASH_BASE + (h % _HASH_BASE)


def severity_from_priority(priority: int) -> int:
    """OCSF severity_id of a Snort or Suricata priority: 1 High, 2 Medium, 3 Low, 4 and above Informational
    (AS-694)."""
    return {1: 4, 2: 3, 3: 2}.get(int(priority), 1)


def flow_totals(d: UpdateDraft, stats: IngestStats, source: str) -> None:
    """flow.packets_total, flow.unanswered and flow.bidir_ratio from the per-direction counters."""
    pf, pb = d.value("flow.packets_fwd"), d.value("flow.packets_bwd")
    if pf is not None and pb is not None:
        set_field(d, "flow.packets_total", int(pf) + int(pb), source, stats)
    if pb is not None:
        set_field(d, "flow.unanswered", 1 if int(pb) == 0 else 0, source, stats)
    bf, bb = d.value("flow.bytes_fwd"), d.value("flow.bytes_bwd")
    if bf is not None and bb is not None and int(bf) > 0:
        set_field(d, "flow.bidir_ratio", float(bb) / float(bf), source, stats)


#: Zeek history letters with the TCP flags they imply (Zeek conn.log documentation, `history`).
_HISTORY_FLAGS: dict[str, int] = {"s": 0x02, "h": 0x12, "a": 0x10, "f": 0x01, "r": 0x04}
#: Share of the six flags of flow.tcp_flags that a history can show (SYN, ACK, FIN, RST of six; AS-688).
HISTORY_RELIABILITY = 4.0 / 6.0


def zeek_history_flags(d: UpdateDraft, history: str, stats: IngestStats, source: str) -> None:
    """flow.tcp_flags_fwd / _bwd / flow.tcp_flags from a Zeek history (upper case originator, lower
    case responder), LOW_RELIABILITY because PSH and URG are not encoded (AS-688)."""
    fwd = bwd = 0
    for ch in history:
        bits = _HISTORY_FLAGS.get(ch.lower())
        if bits is None:
            continue
        if ch.isupper():
            fwd |= bits
        else:
            bwd |= bits
    set_field(d, "flow.tcp_flags_fwd", fwd, source, stats, status=LOW, reliability=HISTORY_RELIABILITY)
    set_field(d, "flow.tcp_flags_bwd", bwd, source, stats, status=LOW, reliability=HISTORY_RELIABILITY)
    set_field(d, "flow.tcp_flags", (fwd | bwd) & 0x3F, source, stats, status=LOW, reliability=HISTORY_RELIABILITY)


#: Zeek conn_state -> flow.end_reason: SF closed by FIN from both sides; REJ and RST* reset. Other
#: states do not say how the flow ended (AS-681).
CONN_STATE_END: dict[str, int] = {"SF": 1, "REJ": 2, "RSTO": 2, "RSTR": 2, "RSTOS0": 2, "RSTRH": 2}


def end_reason_from_conn_state(d: UpdateDraft, conn_state: str, stats: IngestStats, source: str) -> None:
    code = CONN_STATE_END.get(conn_state)
    if code is not None:
        set_field(d, "flow.end_reason", code, source, stats)


def shannon_entropy(text: str) -> float:
    """Shannon entropy per character, in bits: H = -sum_c p_c log2 p_c over the characters of `text`."""
    if not text:
        return 0.0
    n = len(text)
    return -sum((k / n) * math.log2(k / n) for k in Counter(text).values())


def dns_name_stats(d: UpdateDraft, stats: IngestStats, source: str) -> None:
    """proto.dns.query_length and query_entropy of the queried name (lower case, final dot removed)."""
    q = d.value("proto.dns.query")
    if q is None:
        return
    name = str(q).rstrip(".").lower()
    set_field(d, "proto.dns.query_length", len(name), source, stats)
    set_field(d, "proto.dns.query_entropy", shannon_entropy(name), source, stats)


def dns_answer_stats(d: UpdateDraft, stats: IngestStats, source: str) -> None:
    """proto.dns.answer_count from the answers list and proto.dns.ttl_min from the TTLs."""
    answers = d.value("proto.dns.answers")
    if answers is not None:
        set_field(d, "proto.dns.answer_count", len(answers), source, stats)
    ttls = d.value("proto.dns.ttls")
    if ttls:
        set_field(d, "proto.dns.ttl_min", float(min(ttls)), source, stats)


def http_uri_length(d: UpdateDraft, stats: IngestStats, source: str) -> None:
    uri = d.value("proto.http.uri")
    if uri is not None:
        set_field(d, "proto.http.uri_length", len(str(uri)), source, stats)


def cert_facts(d: UpdateDraft, stats: IngestStats, source: str) -> None:
    """proto.tls.cert_validity (not after - not before) and cert_self_signed (subject equals issuer)."""
    nb, na = d.value("proto.tls.cert_not_before"), d.value("proto.tls.cert_not_after")
    if nb is not None and na is not None:
        if float(na) >= float(nb):
            set_field(d, "proto.tls.cert_validity", float(na) - float(nb), source, stats)
        else:
            stats.refuse("proto.tls.cert_validity", "not after precedes not before")
    subj, iss = d.value("proto.tls.cert_subject"), d.value("proto.tls.cert_issuer")
    if subj is not None and iss is not None:
        set_field(d, "proto.tls.cert_self_signed", 1 if str(subj).strip() == str(iss).strip() else 0, source, stats)


def alert_identity(d: UpdateDraft, stats: IngestStats, source: str) -> None:
    """alert.signature from generator and signature ids, alert.severity from the priority."""
    gid, sid = d.value("alert.generator_id"), d.value("alert.signature_id")
    if sid is not None:
        set_field(d, "alert.signature", rule_signature(int(gid) if gid is not None else 1, int(sid)), source, stats)
    pri = d.value("alert.priority")
    if pri is not None:
        set_field(d, "alert.severity", severity_from_priority(int(pri)), source, stats)


def flow_entities(d: UpdateDraft, resolver: EntityResolver, *, service: bool = True) -> None:
    """Initiator, responder and (for TCP, UDP, SCTP with a port) the responder's service."""
    a = resolver.address(d.value("flow.src_ip"))
    b = resolver.address(d.value("flow.dst_ip"))
    d.add_entity(a, "initiator")
    d.add_entity(b, "responder")
    if service:
        port, proto = d.value("flow.dst_port"), d.value("flow.protocol")
        d.add_entity(resolver.service(b, None if port is None else int(port), None if proto is None else int(proto)),
                     "service")


def times_and_duration(d: UpdateDraft, stats: IngestStats, source: str) -> None:
    """flow.duration from flow.start_time and flow.end_time when the source gives both."""
    s, e = d.value("flow.start_time"), d.value("flow.end_time")
    if s is not None and e is not None:
        if float(e) >= float(s):
            set_field(d, "flow.duration", float(e) - float(s), source, stats)
        else:
            stats.refuse("flow.duration", "end precedes start")


__all__ = [
    "CONN_STATE_END", "HISTORY_RELIABILITY", "alert_identity", "cert_facts", "dns_answer_stats", "dns_name_stats",
    "end_reason_from_conn_state", "flow_entities", "flow_totals", "http_uri_length", "name_signature",
    "rule_signature", "severity_from_priority", "shannon_entropy", "times_and_duration", "zeek_history_flags",
]
