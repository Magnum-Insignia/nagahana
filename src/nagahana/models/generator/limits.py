"""Hard physical limits on Generator variants: an exact check, and a projection for learned outputs.

Purpose
-------
The owner requires that the Generator "uses the same concept of physics informed concept to not
hallucinate impossible ones" [A-17]. Physics is a boundary on *model outputs*, never a detector (D-18).
This module holds the limits that are **exact facts** of IP/TCP flow accounting and protocol field
widths, so they can be checked strictly (no weight, no tolerance beyond floating-point rounding):

    name                   limit (per update row, only where every field involved contributes, D-41)
    ---------------------  -----------------------------------------------------------------------------
    nonnegative            x ≥ 0 for every COUNT and CONTINUOUS cell (all catalogue magnitudes are ≥ 0)
    integral               COUNT, CATEGORICAL and BITMASK cells are integers
    code_range             protocol-field widths: ports ≤ 65535 (16 bit), IP protocol ≤ 255 (8 bit), …
    packets_min            packets_fwd + packets_bwd ≥ 1 (a flow record exists because a packet was seen)
    flag_le_packets        flag_count_f ≤ packets_fwd + packets_bwd           (a packet carries a flag or not)
    tally_le_packets       ip_df_count, ip_mf_count, retransmissions ≤ packets  (each is a packet tally)
    histogram_le_packets   Σ payload-size bins ≤ packets                        (each binned item is a packet)
    mtu                    bytes_d ≤ packets_d · MTU                            (no IP packet exceeds the MTU)
    min_header             bytes_d ≥ 20 · packets_d                (IPv4 header ≥ 20 B, RFC 791; IPv6 = 40 B)
    iat_mean_le_max        iat_mean ≤ iat_max                                   (a mean never exceeds the max)
    iat_max_le_duration    iat_max ≤ duration                    (every gap lies between first and last packet)
    ttl_range              ttl_mean ≤ 255                                       (TTL is an 8-bit field, RFC 791)
    window_range           tcp_window_init_d ≤ 65535   (16-bit field; never scaled in a SYN, RFC 7323 §2.2)
    link_rate              bytes_d ≤ (link_bps / 8) · duration + MTU     (only if the site link rate is set)

`flow.bytes_*` are IP-layer bytes (datamodel/fields.py), so `mtu` and `min_header` are in the same
layer. The link-rate bound: after the first packet (≤ MTU bytes) every further byte of direction d
must cross the link within the flow's duration, so bytes_d − MTU ≤ rate · duration.

What is deliberately *not* a hard limit here (and why)
- Bounds on `iat_var` / `ttl_var` (Bhatia–Davis, Popoviciu) hold for the *population* variance; whether
  an adapter reports the population or the sample variance is a stage-1 definition (fields.py,
  "definitions recorded per adapter"), and the sample variance can exceed those bounds. Only ≥ 0.
- `flow.bidir_ratio`: its definition is per adapter. Only ≥ 0.

Two uses
--------
1. `check_hard_limits` / `hard_limit_counts`: the strict check every variant passes (acceptance.py).
2. `project_hard_limits`: for *learned* families only. It moves **only free cells** (cells the model
   generated) so that the limits hold — the "hard: by construction" form of physics/constraints.py.
   Real (kept) cells are never edited. A violation that involves only kept cells is left in place and
   the strict check then rejects the variant. The projection order is assumption AS-359 (declared by
   the learned producers that call it).

The limits themselves are not assumptions: each is a protocol or accounting fact cited above, applied
under the decided physics-bounded principle ([A-17], D-18).

Owner sources: [A-08], [A-17], [Q-37]. Decisions: D-18, D-37, D-40, D-41. Assumptions: AS-359.

Invariants
----------
- Excluded cells (NaN) never enter arithmetic: every computation reads `np.where(contrib, values, 0)`.
- The projection never changes a non-free cell and never changes a status.

Extension points: `CODE_WIDTH_BITS` (add a categorical field's width), `SIGNED_FIELDS` (a future field
that can be negative), and new limits appended to `LIMIT_NAMES` with a branch in `check_hard_limits`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from nagahana.core.errors import InvariantViolation
from nagahana.datamodel.columnar import Column
from nagahana.datamodel.fields import Kind

# ----------------------------------------------------------------------------------- field IDs
PACKETS = ("flow.packets_fwd", "flow.packets_bwd")
BYTES = ("flow.bytes_fwd", "flow.bytes_bwd")
FLAGS = tuple(f"flow.flag_count.{f}" for f in ("syn", "ack", "fin", "rst", "psh", "urg"))
TALLIES = ("pkt.ip_df_count", "pkt.ip_mf_count", "pkt.retransmissions")
HISTOGRAM = "pkt.payload_size_hist"
DURATION, IAT_MEAN, IAT_MAX, IAT_VAR = "flow.duration", "flow.iat_mean", "flow.iat_max", "flow.iat_var"
TTL_MEAN = "pkt.ttl_mean"
WINDOWS = ("pkt.tcp_window_init_fwd", "pkt.tcp_window_init_bwd")

#: Protocol facts (not site parameters).
MIN_IP_HEADER_BYTES = 20.0      # RFC 791: IHL ≥ 5 32-bit words; IPv6's fixed header is larger (40 B)
TTL_MAX = 255.0                 # RFC 791: 8-bit TTL / RFC 8200: 8-bit hop limit
WINDOW_MAX = 65535.0            # RFC 9293: 16-bit window; RFC 7323 §2.2: a SYN's window is never scaled

#: Bit width of categorical / bitmask fields whose wire width is certain (an upper bound on the code).
CODE_WIDTH_BITS: dict[str, int] = {
    "flow.src_port": 16, "flow.dst_port": 16,           # TCP/UDP header (RFC 9293, RFC 768)
    "flow.protocol": 8,                                  # IPv4 protocol / IPv6 next header
    "flow.tcp_flags": 16,                                # ≤ FieldEncoderConfig.max_bits (TCP uses 6–9 bits)
    "proto.icmp.type": 8,                                # RFC 792
    "proto.arp.opcode": 16,                              # RFC 826
    "proto.dns.qtype": 16,                               # RFC 1035
    "proto.dns.rcode": 12,                               # RFC 6891 extended RCODE (4 + 8 bits)
    "ot.modbus.function_code": 8, "ot.modbus.unit_id": 8,
    "ot.modbus.register_start": 16, "ot.modbus.register_count": 16,
    "ot.dnp3.function_code": 8, "ot.dnp3.object_group": 8,
    "ot.iec104.type_id": 8, "ot.iec104.cot": 8,
}

#: Catalogue fields that may legitimately be negative (none today).
SIGNED_FIELDS: frozenset[str] = frozenset()

LIMIT_NAMES: tuple[str, ...] = (
    "nonnegative", "integral", "code_range", "packets_min", "flag_le_packets", "tally_le_packets",
    "histogram_le_packets", "mtu", "min_header", "iat_mean_le_max", "iat_max_le_duration", "ttl_range",
    "window_range", "link_rate",
)

_REL_TOL = 1e-9  # floating-point slack only; a real violation is far larger


@dataclass(frozen=True)
class PhysicalSetting:
    """Site physics the limits need. Both values are site-specific, so neither has a default.

    mtu: largest IP packet on the path, bytes (1500 on standard Ethernet, up to 9000 with jumbo frames;
        physics/residuals.MTUBound). link_bps: link rate in bit/s, or None when unknown — the link-rate
        limit is then *not checked*, and reports say so (never silently assumed).
    """

    mtu: float
    link_bps: float | None = None

    def __post_init__(self) -> None:
        if not (self.mtu > 0 and math.isfinite(self.mtu)):
            raise InvariantViolation("mtu must be a positive finite number of bytes")
        if self.link_bps is not None and not (self.link_bps > 0 and math.isfinite(self.link_bps)):
            raise InvariantViolation("link_bps must be positive and finite (or None = not checked)")


class ColumnIndex:
    """Field ID → column positions of a matrix (histograms have one column per bin)."""

    def __init__(self, columns: Sequence[Column]) -> None:
        self.columns = tuple(columns)
        self._by_field: dict[str, list[int]] = {}
        for j, c in enumerate(self.columns):
            self._by_field.setdefault(c.field_id, []).append(j)

    def one(self, field_id: str) -> int | None:
        """The column of a scalar field, or None if the matrix does not have it."""
        cols = self._by_field.get(field_id)
        return cols[0] if cols else None

    def all(self, field_id: str) -> list[int]:
        """Every column of a field (histogram bins), [] if absent."""
        return list(self._by_field.get(field_id, []))

    def has(self, *field_ids: str) -> bool:
        return all(f in self._by_field for f in field_ids)


def _gt(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # a > b beyond floating-point rounding (relative to the larger magnitude).
    return a > b + _REL_TOL * np.maximum(1.0, np.abs(b))


def check_hard_limits(
    values: np.ndarray,
    contributing: np.ndarray,
    columns: Sequence[Column],
    setting: PhysicalSetting,
) -> dict[str, np.ndarray]:
    """Rows violating each limit: name → bool [N]. Only rows where every field involved contributes.

    values: float64 [N, C] (NaN allowed in excluded cells); contributing: bool [N, C].
    """
    n = values.shape[0]
    idx = ColumnIndex(columns)
    x = np.where(contributing, values, 0.0)        # [N, C] — NaN never reaches arithmetic (D-41)
    out = {name: np.zeros(n, dtype=bool) for name in LIMIT_NAMES}

    def col(f: str) -> tuple[np.ndarray, np.ndarray] | None:
        # (values [N], contributing [N]) of a scalar field, or None if the matrix lacks it.
        j = idx.one(f)
        return None if j is None else (x[:, j], contributing[:, j])

    # -- per-cell limits: sign, integrality, code width --------------------------------------------
    for j, c in enumerate(columns):
        m = contributing[:, j]
        v = x[:, j]
        if c.kind in (Kind.COUNT, Kind.CONTINUOUS) and c.field_id not in SIGNED_FIELDS:
            out["nonnegative"] |= m & (v < -_REL_TOL)
        if c.kind in (Kind.COUNT, Kind.CATEGORICAL, Kind.BITMASK, Kind.HISTOGRAM):
            out["integral"] |= m & (np.abs(v - np.round(v)) > 1e-9)
        bits = CODE_WIDTH_BITS.get(c.field_id)
        if bits is not None:
            out["code_range"] |= m & ((v < -_REL_TOL) | (v > float(2**bits - 1) + 1e-9))

    # -- packet totals ------------------------------------------------------------------------------
    pf, pb = col(PACKETS[0]), col(PACKETS[1])
    if pf is not None and pb is not None:
        total, m_tot = pf[0] + pb[0], pf[1] & pb[1]           # [N]
        out["packets_min"] |= m_tot & (total < 1.0 - 1e-9)
        for f in FLAGS:
            fc = col(f)
            if fc is not None:
                out["flag_le_packets"] |= m_tot & fc[1] & _gt(fc[0], total)
        for f in TALLIES:
            tc = col(f)
            if tc is not None:
                out["tally_le_packets"] |= m_tot & tc[1] & _gt(tc[0], total)
        bins = idx.all(HISTOGRAM)
        if bins:
            m_h = contributing[:, bins].all(axis=1)
            out["histogram_le_packets"] |= m_tot & m_h & _gt(x[:, bins].sum(axis=1), total)

    # -- bytes vs packets (MTU and minimum header), per direction --------------------------------
    for fb, fp in zip(BYTES, PACKETS, strict=True):
        b, p = col(fb), col(fp)
        if b is None or p is None:
            continue
        m = b[1] & p[1]
        out["mtu"] |= m & _gt(b[0], p[0] * setting.mtu)
        out["min_header"] |= m & _gt(MIN_IP_HEADER_BYTES * p[0], b[0])
        d = col(DURATION)
        if setting.link_bps is not None and d is not None:
            cap = setting.link_bps / 8.0 * d[0] + setting.mtu      # bytes the link can carry
            out["link_rate"] |= m & d[1] & _gt(b[0], cap)

    # -- timing order ----------------------------------------------------------------------------
    mean, mx, dur = col(IAT_MEAN), col(IAT_MAX), col(DURATION)
    if mean is not None and mx is not None:
        out["iat_mean_le_max"] |= mean[1] & mx[1] & _gt(mean[0], mx[0])
    if mx is not None and dur is not None:
        out["iat_max_le_duration"] |= mx[1] & dur[1] & _gt(mx[0], dur[0])

    # -- protocol field ranges ---------------------------------------------------------------------
    ttl = col(TTL_MEAN)
    if ttl is not None:
        out["ttl_range"] |= ttl[1] & _gt(ttl[0], np.full(n, TTL_MAX))
    for f in WINDOWS:
        w = col(f)
        if w is not None:
            out["window_range"] |= w[1] & _gt(w[0], np.full(n, WINDOW_MAX))
    return out


def hard_limit_counts(
    values: np.ndarray, contributing: np.ndarray, columns: Sequence[Column], setting: PhysicalSetting
) -> dict[str, int]:
    """Number of violating rows per limit (0 everywhere = the variant is inside the boundary)."""
    return {k: int(v.sum()) for k, v in check_hard_limits(values, contributing, columns, setting).items()}


def project_hard_limits(
    values: np.ndarray,
    contributing: np.ndarray,
    columns: Sequence[Column],
    setting: PhysicalSetting,
    free: np.ndarray,
) -> np.ndarray:
    """Move only `free` cells (bool [N, C], generated by a learned family) so the hard limits hold.

    Returns a new float64 matrix. Non-free cells and excluded cells are returned unchanged. The order of
    the passes (AS-359) resolves chains: integrality and ranges first, then packets, then everything that
    is bounded by packets, then timing. Each pair constraint moves a free side only:
    - packet-bounded tallies (flags, DF/MF, retransmissions, histogram bins) are capped when free, else a
      free packet direction is raised to cover them;
    - bytes are clipped into [20·packets, MTU·packets] (and the link rate) when free, else a free packet
      count is moved into [⌈bytes/MTU⌉, ⌊bytes/20⌋] (before the tallies are checked);
    - in the chain iat_mean ≤ iat_max ≤ duration a free upper bound is raised first (raising never breaks
      an earlier link, and a longer duration only relaxes the link-rate bound), else a free lower one is
      lowered.
    The strict `check_hard_limits` decides afterwards; nothing here hides a violation among kept cells.
    """
    if free.shape != values.shape or contributing.shape != values.shape:
        raise InvariantViolation("free / contributing must have the shape of values")
    out = values.astype(np.float64, copy=True)
    fr = free & contributing                         # a free cell must be a contributing cell
    idx = ColumnIndex(columns)

    def j_of(f: str) -> int | None:
        return idx.one(f)

    def set_where(j: int, rows: np.ndarray, new: np.ndarray) -> None:
        # write `new` into column j on `rows` (bool [N]) — only on free cells
        r = rows & fr[:, j]
        out[r, j] = new[r]

    # ---- pass 1: sign, integrality, code width on every free cell ---------------------------------
    for j, c in enumerate(columns):
        r = fr[:, j]
        if not r.any():
            continue
        v = out[:, j]
        if c.kind in (Kind.COUNT, Kind.HISTOGRAM):
            v = np.where(r, np.round(np.maximum(v, 0.0)), v)
        elif c.kind is Kind.CONTINUOUS and c.field_id not in SIGNED_FIELDS:
            v = np.where(r, np.maximum(v, 0.0), v)
        elif c.kind in (Kind.CATEGORICAL, Kind.BITMASK):
            v = np.where(r, np.round(v), v)
        bits = CODE_WIDTH_BITS.get(c.field_id)
        if bits is not None:
            v = np.where(r, np.clip(v, 0.0, float(2**bits - 1)), v)
        out[:, j] = v
    for f, top in ((TTL_MEAN, TTL_MAX), (WINDOWS[0], WINDOW_MAX), (WINDOWS[1], WINDOW_MAX)):
        jr = j_of(f)
        if jr is not None:
            set_where(jr, np.ones(len(out), dtype=bool), np.minimum(out[:, jr], top))

    jpf, jpb = j_of(PACKETS[0]), j_of(PACKETS[1])
    both = jpf is not None and jpb is not None
    jd = j_of(DURATION)

    # ---- pass 2: packets consistent with kept bytes (and a kept duration), then ≥ 1 packet in total ---
    for fb, fp in zip(BYTES, PACKETS, strict=True):
        jb, jp = j_of(fb), j_of(fp)
        if jb is None or jp is None:
            continue
        m = contributing[:, jb] & contributing[:, jp]
        kept_bytes = m & ~fr[:, jb]
        # packets free, bytes kept: ceil(bytes / MTU) ≤ packets ≤ floor(bytes / 20)
        lo = np.ceil(out[:, jb] / setting.mtu)
        hi = np.floor(out[:, jb] / MIN_IP_HEADER_BYTES)
        set_where(jp, kept_bytes, np.clip(out[:, jp], lo, np.maximum(lo, hi)))
    if setting.link_bps is not None and jd is not None:
        # a kept duration bounds the bytes, hence the packets: 20·packets ≤ bytes ≤ rate·duration + MTU
        for fp in PACKETS:
            jp = j_of(fp)
            if jp is None:
                continue
            m = contributing[:, jp] & contributing[:, jd] & ~fr[:, jd]
            cap_p = np.floor((setting.link_bps / 8.0 * out[:, jd] + setting.mtu) / MIN_IP_HEADER_BYTES)
            set_where(jp, m & (out[:, jp] > cap_p), cap_p)
    if both:
        assert jpf is not None and jpb is not None
        m = contributing[:, jpf] & contributing[:, jpb]
        for fb, jp in ((BYTES[0], jpf), (BYTES[1], jpb)):
            # raise a free direction to one packet where its bytes allow a packet (free, or ≥ 20 kept)
            jb = j_of(fb)
            room = np.ones(len(out), dtype=bool) if jb is None else (
                ~contributing[:, jb] | fr[:, jb] | (out[:, jb] >= MIN_IP_HEADER_BYTES))
            zero = m & (out[:, jpf] + out[:, jpb] < 1.0) & room
            set_where(jp, zero, np.ones(len(out)))

    # ---- pass 3: everything bounded by the packet total ------------------------------------------
    if both:
        assert jpf is not None and jpb is not None
        m_tot = contributing[:, jpf] & contributing[:, jpb]

        def packet_cap(fp: str, fb: str) -> np.ndarray:
            # largest packet count direction `fp` may reach: ⌊bytes/20⌋ if its bytes are kept, and the
            # link-rate cap if the duration is kept; +∞ otherwise
            jp, jb = j_of(fp), j_of(fb)
            assert jp is not None
            cap = np.full(len(out), np.inf)
            if jb is not None:
                kept_b = contributing[:, jb] & ~fr[:, jb]
                cap = np.where(kept_b, np.floor(out[:, jb] / MIN_IP_HEADER_BYTES), cap)
            if setting.link_bps is not None and jd is not None:
                kept_d = contributing[:, jd] & ~fr[:, jd]
                link = np.floor((setting.link_bps / 8.0 * out[:, jd] + setting.mtu) / MIN_IP_HEADER_BYTES)
                cap = np.where(kept_d, np.minimum(cap, link), cap)
            return cap

        def raise_total(rows: np.ndarray, deficit: np.ndarray) -> None:
            # cover `deficit` packets by raising free directions, each only up to its cap
            left = np.where(rows, np.maximum(deficit, 0.0), 0.0)
            for fp, fb in zip(PACKETS, BYTES, strict=True):
                jp = j_of(fp)
                assert jp is not None
                room = np.where(fr[:, jp], np.maximum(packet_cap(fp, fb) - out[:, jp], 0.0), 0.0)
                add = np.minimum(left, room)
                out[:, jp] = out[:, jp] + add
                left = left - add

        for f in (*FLAGS, *TALLIES):
            jf = j_of(f)
            if jf is None:
                continue
            m = m_tot & contributing[:, jf]
            # the tally is free → cap it at the total; else raise free packet directions to cover it
            set_where(jf, m & (out[:, jf] > out[:, jpf] + out[:, jpb]), out[:, jpf] + out[:, jpb])
            raise_total(m, out[:, jf] - (out[:, jpf] + out[:, jpb]))
        bins = idx.all(HISTOGRAM)
        if bins:
            total = out[:, jpf] + out[:, jpb]
            m_h = m_tot & contributing[:, bins].all(axis=1)
            over = m_h & (out[:, bins].sum(axis=1) > total)
            if over.any():
                # scale the free bins down proportionally to the room the kept bins leave
                fb_mask = fr[:, bins]                                        # [N, n_bins]
                kept_sum = np.where(fb_mask, 0.0, out[:, bins]).sum(axis=1)
                free_sum = np.where(fb_mask, out[:, bins], 0.0).sum(axis=1)
                room = np.maximum(total - kept_sum, 0.0)
                ratio = np.where(free_sum > 0, np.minimum(1.0, room / np.maximum(free_sum, 1e-12)), 1.0)
                for k, jb in enumerate(bins):
                    new = np.floor(out[:, jb] * ratio)
                    r = over & fb_mask[:, k]
                    out[r, jb] = new[r]
            # kept bins still above the total → raise free packet directions by the deficit
            raise_total(m_h, out[:, bins].sum(axis=1) - (out[:, jpf] + out[:, jpb]))

    # ---- pass 4: bytes inside [20·packets, MTU·packets] (and the link rate) ---------------------
    for fb, fp in zip(BYTES, PACKETS, strict=True):
        jb, jp = j_of(fb), j_of(fp)
        if jb is None or jp is None:
            continue
        m = contributing[:, jb] & contributing[:, jp]
        lo = MIN_IP_HEADER_BYTES * out[:, jp]
        hi = setting.mtu * out[:, jp]
        if setting.link_bps is not None and jd is not None:
            # a free bytes cell also respects the link rate given the (possibly free) duration
            md = contributing[:, jd]
            cap = setting.link_bps / 8.0 * out[:, jd] + setting.mtu
            hi = np.where(md, np.minimum(hi, np.floor(cap)), hi)
        set_where(jb, m, np.clip(out[:, jb], lo, np.maximum(lo, hi)))
        if setting.link_bps is not None and jd is not None:
            # bytes kept but too many for the duration → lengthen a free duration
            md = m & contributing[:, jd]
            need = (out[:, jb] - setting.mtu) * 8.0 / setting.link_bps
            set_where(jd, md & (out[:, jd] < need), need)

    # ---- pass 5: iat_mean ≤ iat_max ≤ duration (raise a free upper bound first; raising never breaks
    #      an earlier link of the chain, and a longer duration only relaxes the link-rate bound)
    jmean, jmax = j_of(IAT_MEAN), j_of(IAT_MAX)
    if jmean is not None and jmax is not None:
        m = contributing[:, jmean] & contributing[:, jmax]
        bad = m & (out[:, jmean] > out[:, jmax])
        set_where(jmax, bad, out[:, jmean])                          # raise a free max …
        bad = m & (out[:, jmean] > out[:, jmax])
        set_where(jmean, bad, out[:, jmax])                          # … else lower a free mean
    if jmax is not None and jd is not None:
        m = contributing[:, jmax] & contributing[:, jd]
        bad = m & (out[:, jmax] > out[:, jd])
        set_where(jd, bad, out[:, jmax])                             # lengthen a free duration …
        bad = m & (out[:, jmax] > out[:, jd])
        set_where(jmax, bad, out[:, jd])                             # … else lower a free max
        if jmean is not None:
            m2 = m & contributing[:, jmean]
            bad = m2 & (out[:, jmean] > out[:, jmax])
            set_where(jmean, bad, out[:, jmax])                      # a lowered max pulls a free mean
    return out
