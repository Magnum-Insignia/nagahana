"""Shannon entropies of traffic distributions per entity and per window, batch and streaming (D-56, D-41).

Purpose
-------
Entropy-based anomaly detection reads the dispersal of traffic features: a port scan spreads one
source over many destination ports (high destination-port entropy at the target), a sweep spreads it
over many peers, a flood concentrates many sources on one destination (low destination entropy).
Lakhina, Crovella and Diot ("Mining anomalies using traffic feature distributions", SIGCOMM 2005)
showed that the entropies of address and port distributions expose such anomalies even when volume
does not change. This module computes those entropies for the network state window of every trigger
(AS-764: the state updates of (tau - Delta, tau]), per entity and over the whole network.

Distributions (AS-765)
----------------------
Per entity (its participations in the state updates of the window; `TrafficConfig.roles` selects
which participations count: any member, the initiator only or the responder only):
    dst_port, src_port, protocol   the categorical field of each update (flow.dst_port, ...)
    flags                          the TCP flag combination of the update (the flow.tcp_flags bitmask code)
    flag_mass                      packets per flag (flow.flag_count.{syn, ack, fin, rst, psh, urg}),
                                   a distribution over the six flags weighted by packet counts
    peer                           the other members of the update (counterpart entities)
Over the network (one event per update):
    dst_port, src_port, protocol, flags, flag_mass   as above
    src_entity, dst_entity         initiator and responder entities (Lakhina's source and destination
                                   address distributions)
Only contributing cells are evidence (D-41: absence is "not supplied", never zero). A distribution
whose column is absent, or a key without any event in the window, has no entropy: NaN, with sample
size 0.

Estimators (AS-766), all in nats, for counts n_k with N = sum_k n_k and m = #{k : n_k > 0}:
    plugin        H = log N - (1 / N) sum_k n_k log n_k                       (maximum likelihood)
    miller_madow  H_MM = H + (m - 1) / (2 N)                                  (first-order bias correction)
    chao_shen     C = 1 - f1 / N (f1 singletons; f1 = N is replaced by N - 1), p_k = C n_k / N,
                  H_CS = -sum_k p_k log p_k / (1 - (1 - p_k)^N)               (coverage-adjusted)
The plug-in estimator is biased downwards by about (m - 1) / (2 N) (Miller, "Note on the bias of
information estimates", in Information Theory in Psychology, Free Press 1955; Paninski, "Estimation
of entropy and mutual information", Neural Computation 15(6):1191, 2003). Chao and Shen
("Nonparametric estimation of Shannon's index of diversity when there are unseen species in sample",
Environmental and Ecological Statistics 10:429, 2003) correct for unseen symbols with the
Horvitz-Thompson form above; it needs integer counts. For weighted counts (flag_mass) N is the
number of packets.

Jensen-Shannon divergence between successive network states (per distribution, plug-in
probabilities P and Q of two windows): JSD(P, Q) = H((P + Q) / 2) - (H(P) + H(Q)) / 2, in [0, log 2],
zero iff P = Q (Lin, "Divergence measures based on the Shannon entropy", IEEE Transactions on
Information Theory 37(1):145, 1991).

Streaming (AS-778)
------------------
`TrafficEntropyTracker` keeps every histogram of the sliding window (tau - Delta, tau] with O(1)
amortised work per state update: each event is added once and evicted once, and the totals
N, sum_k n_k log n_k and m are updated incrementally (a count n -> n + w changes the sum by
(n + w) log(n + w) - n log n). The sum is accumulated with Neumaier's compensated summation
(Neumaier, ZAMM 54:39, 1974; Higham, "Accuracy and Stability of Numerical Algorithms", 2nd ed., SIAM
2002, section 4.3) and recomputed exactly (math.fsum) after `resync_every` changes of a key, so its
error never accumulates over a long run. Reading the plug-in or Miller-Madow entropy of a key is
O(1); Chao-Shen reads the key's histogram (O(support)). Events are committed only up to the reading
time, so a reading at tau never sees an update after tau, whatever was ingested already. The values
equal the batch `traffic_entropies` on the same updates (tested to 1e-12).

Precision (D-54): all entropies and sample sizes are float64.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import torch

from nagahana.datamodel.columnar import STATUS_ORDER
from nagahana.datamodel.status import CONTRIBUTING
from nagahana.models.batch import WindowBatch
from nagahana.statphys.config import ENTITY_DISTRIBUTIONS, ENTROPY_ESTIMATORS, NETWORK_DISTRIBUTIONS, ROLES, TrafficConfig

#: Columns of the categorical distributions (datamodel/fields.py).
FIELD_COLUMNS: dict[str, str] = {"dst_port": "flow.dst_port", "src_port": "flow.src_port", "protocol": "flow.protocol",
                                 "flags": "flow.tcp_flags"}
#: Per-flag packet counts of the flag_mass distribution, in the order of the flag index (symbol).
FLAG_COUNT_COLUMNS: tuple[str, ...] = ("flow.flag_count.syn", "flow.flag_count.ack", "flow.flag_count.fin",
                                       "flow.flag_count.rst", "flow.flag_count.psh", "flow.flag_count.urg")
#: Every column a traffic distribution may read.
TRAFFIC_COLUMNS: tuple[str, ...] = (*FIELD_COLUMNS.values(), *FLAG_COUNT_COLUMNS)
#: Status codes whose cells are evidence (D-41).
CONTRIBUTING_CODES: tuple[int, ...] = tuple(i for i, s in enumerate(STATUS_ORDER) if s in CONTRIBUTING)


@dataclass(frozen=True)
class EntropyEstimate:
    """Entropy estimates (nats) with their sample sizes: value NaN where samples == 0."""

    value: torch.Tensor
    samples: torch.Tensor
    support: torch.Tensor


def _check_estimator(estimator: str) -> None:
    if estimator not in ENTROPY_ESTIMATORS:
        raise ValueError(f"estimator must be one of {ENTROPY_ESTIMATORS}; got {estimator!r}")


def _chao_shen_terms(count: torch.Tensor, n_tot: torch.Tensor, coverage: torch.Tensor) -> torch.Tensor:
    # -p log p / (1 - (1 - p)^N) per symbol, p = C n / N; (1 - p)^N via exp(N log1p(-p)) (no underflow loss).
    p = coverage * count / n_tot.clamp_min(1.0)
    p_safe = torch.where(count > 0, p, torch.full_like(p, 0.5))
    denom = -torch.expm1(n_tot * torch.log1p(-p_safe.clamp(max=1.0 - 1e-300)))
    denom = torch.where(p_safe >= 1.0, torch.ones_like(denom), denom)
    term = -torch.xlogy(p_safe, p_safe) / denom
    return torch.where(count > 0, term, torch.zeros_like(term))


def entropy_from_counts(counts: torch.Tensor, *, estimator: str = "miller_madow") -> EntropyEstimate:
    """Entropy of the histograms on the last axis of `counts` [..., K] (non-negative), module docstring."""
    _check_estimator(estimator)
    n = counts.to(torch.float64)
    if bool((n < 0).any()) or not bool(torch.isfinite(n).all()):
        raise ValueError("counts must be finite and non-negative")
    n_tot = n.sum(-1)
    support = (n > 0).sum(-1)
    ok = n_tot > 0
    safe_n = torch.where(ok, n_tot, torch.ones_like(n_tot))
    if estimator == "chao_shen":
        if not bool((n == torch.round(n)).all()):
            raise ValueError("the Chao-Shen estimator needs integer counts")
        f1 = (n == 1).sum(-1).to(torch.float64)
        f1 = torch.where(f1 >= safe_n, safe_n - 1.0, f1)
        cov = 1.0 - f1 / safe_n
        h = _chao_shen_terms(n, safe_n.unsqueeze(-1), cov.unsqueeze(-1)).sum(-1)
    else:
        h = torch.log(safe_n) - torch.xlogy(n, n).sum(-1) / safe_n
        if estimator == "miller_madow":
            h = h + (support.to(torch.float64) - 1.0) / (2.0 * safe_n)
    return EntropyEstimate(value=torch.where(ok, h, torch.full_like(h, math.nan)), samples=n_tot, support=support)


@dataclass(frozen=True)
class GroupedCounts:
    """Sparse histograms of many groups: one entry per observed (group, symbol) pair with its count."""

    group: torch.Tensor          # long [P]
    count: torch.Tensor          # float64 [P] (> 0)
    n_groups: int


def grouped_counts(group: torch.Tensor, symbol: torch.Tensor, n_groups: int,
                   weight: torch.Tensor | None = None) -> GroupedCounts:
    """Histogram events (group g_e, symbol s_e, weight w_e) into sparse per-group counts.

    Events with g < 0, g >= n_groups or w <= 0 are ignored. Symbols are arbitrary int64 codes.
    """
    g = group.to(torch.long).flatten()
    s = symbol.to(torch.long).flatten()
    w = torch.ones_like(g, dtype=torch.float64) if weight is None else weight.to(torch.float64).flatten()
    if g.shape != s.shape or g.shape != w.shape:
        raise ValueError("group, symbol and weight must have the same number of events")
    keep = (g >= 0) & (g < n_groups) & (w > 0) & torch.isfinite(w)
    g, s, w = g[keep], s[keep], w[keep]
    if g.numel() == 0:
        return GroupedCounts(group=g, count=w, n_groups=n_groups)
    sym, s_inv = torch.unique(s, return_inverse=True)
    key = g * sym.numel() + s_inv                                               # unique (group, symbol) code
    pair, inv = torch.unique(key, return_inverse=True)
    count = torch.zeros(pair.numel(), dtype=torch.float64, device=w.device).index_add_(0, inv, w)
    return GroupedCounts(group=pair // sym.numel(), count=count, n_groups=n_groups)


def entropy_grouped(gc: GroupedCounts, *, estimator: str = "miller_madow") -> EntropyEstimate:
    """Per-group entropy [G] of sparse histograms (equal to `entropy_from_counts` on dense rows; tested)."""
    _check_estimator(estimator)
    g_n = gc.n_groups
    dev = gc.count.device
    n_tot = torch.zeros(g_n, dtype=torch.float64, device=dev).index_add_(0, gc.group, gc.count)
    support = torch.zeros(g_n, dtype=torch.long, device=dev).index_add_(0, gc.group, torch.ones_like(gc.group))
    ok = n_tot > 0
    safe_n = torch.where(ok, n_tot, torch.ones_like(n_tot))
    if estimator == "chao_shen":
        if not bool((gc.count == torch.round(gc.count)).all()):
            raise ValueError("the Chao-Shen estimator needs integer counts")
        f1 = torch.zeros(g_n, dtype=torch.float64, device=dev).index_add_(0, gc.group, (gc.count == 1).to(torch.float64))
        f1 = torch.where(f1 >= safe_n, safe_n - 1.0, f1)
        cov = 1.0 - f1 / safe_n
        terms = _chao_shen_terms(gc.count, safe_n[gc.group], cov[gc.group])
        h = torch.zeros(g_n, dtype=torch.float64, device=dev).index_add_(0, gc.group, terms)
    else:
        sxl = torch.zeros(g_n, dtype=torch.float64, device=dev).index_add_(0, gc.group, torch.xlogy(gc.count, gc.count))
        h = torch.log(safe_n) - sxl / safe_n
        if estimator == "miller_madow":
            h = h + (support.to(torch.float64) - 1.0) / (2.0 * safe_n)
    return EntropyEstimate(value=torch.where(ok, h, torch.full_like(h, math.nan)), samples=n_tot, support=support)


def jensen_shannon(p_counts: torch.Tensor, q_counts: torch.Tensor) -> torch.Tensor:
    """JSD of the plug-in distributions of two histograms on the last axis (nats; NaN if either is empty)."""
    p = p_counts.to(torch.float64)
    q = q_counts.to(torch.float64)
    sp, sq = p.sum(-1, keepdim=True), q.sum(-1, keepdim=True)
    ok = (sp > 0) & (sq > 0)
    pp = p / torch.where(ok, sp, torch.ones_like(sp))
    qq = q / torch.where(ok, sq, torch.ones_like(sq))
    mix = 0.5 * (pp + qq)

    def h(x: torch.Tensor) -> torch.Tensor:
        return -torch.xlogy(x, x).sum(-1)

    out = h(mix) - 0.5 * (h(pp) + h(qq))
    return torch.where(ok.squeeze(-1), out.clamp_min(0.0), torch.full_like(out, math.nan))


def jensen_shannon_counts(a: Mapping[int, float], b: Mapping[int, float]) -> float:
    """JSD (nats) of two sparse histograms {symbol: count}; NaN if either is empty."""
    sa, sb = math.fsum(a.values()), math.fsum(b.values())
    if sa <= 0 or sb <= 0:
        return math.nan
    keys = sorted(set(a) | set(b))
    pa = np.array([a.get(k, 0.0) / sa for k in keys], dtype=np.float64)
    pb = np.array([b.get(k, 0.0) / sb for k in keys], dtype=np.float64)
    return float(jensen_shannon(torch.from_numpy(pa), torch.from_numpy(pb)))


@dataclass(frozen=True)
class UpdateArrays:
    """The state-update fields the traffic distributions read, one row per update (NumPy).

    time float64 [n] (non-decreasing for streaming); members int64 [n, 3] (initiator, responder,
    service; -1 none); columns: column name -> float64 [n], NaN where the cell does not contribute.
    """

    time: np.ndarray
    members: np.ndarray
    columns: Mapping[str, np.ndarray]

    def __post_init__(self) -> None:
        n = self.time.shape[0]
        if self.time.ndim != 1 or self.members.shape != (n, 3):
            raise ValueError("UpdateArrays needs time [n] and members [n, 3]")
        for k, v in self.columns.items():
            if v.shape != (n,):
                raise ValueError(f"column {k!r} must be [n]")


@dataclass(frozen=True)
class Events:
    """Histogram events of one distribution: update row, key (entity id, or 0 over the network), symbol, weight."""

    row: np.ndarray
    key: np.ndarray
    symbol: np.ndarray
    weight: np.ndarray


def _participations(members: np.ndarray, roles: str) -> list[tuple[int, np.ndarray]]:
    # (member column, rows where that member counts for its entity), distinct members only.
    m = members
    distinct = [m[:, 0] >= 0, (m[:, 1] >= 0) & (m[:, 1] != m[:, 0]),
                (m[:, 2] >= 0) & (m[:, 2] != m[:, 0]) & (m[:, 2] != m[:, 1])]
    cols = {"any": (0, 1, 2), "initiator": (0,), "responder": (1,)}[roles]
    return [(k, distinct[k]) for k in cols]


def _concat(parts: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]) -> Events:
    if not parts:
        z = np.zeros(0, dtype=np.int64)
        return Events(row=z, key=z, symbol=z, weight=np.zeros(0, dtype=np.float64))
    return Events(row=np.concatenate([p[0] for p in parts]), key=np.concatenate([p[1] for p in parts]),
                  symbol=np.concatenate([p[2] for p in parts]),
                  weight=np.concatenate([p[3] for p in parts]).astype(np.float64))


def distribution_events(upd: UpdateArrays, name: str, *, level: str, roles: str = "any") -> Events | None:
    """Events of distribution `name` at `level` ("entity" or "network"); None when its columns are absent."""
    if level not in ("entity", "network"):
        raise ValueError("level must be 'entity' or 'network'")
    if roles not in ROLES:
        raise ValueError(f"roles must be one of {ROLES}")
    known = ENTITY_DISTRIBUTIONS if level == "entity" else NETWORK_DISTRIBUTIONS
    if name not in known:
        raise ValueError(f"unknown {level} distribution {name!r}; known: {known}")
    n = upd.time.shape[0]
    rows = np.arange(n, dtype=np.int64)
    m = upd.members.astype(np.int64)
    parts: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    if name in FIELD_COLUMNS:
        col = upd.columns.get(FIELD_COLUMNS[name])
        if col is None:
            return None
        ok = np.isfinite(col)
        sym = np.where(ok, np.rint(np.where(ok, col, 0.0)), 0.0).astype(np.int64)
        if level == "network":
            parts.append((rows[ok], np.zeros(int(ok.sum()), dtype=np.int64), sym[ok], np.ones(int(ok.sum()))))
        else:
            for k, counts in _participations(m, roles):
                sel = ok & counts
                parts.append((rows[sel], m[sel, k], sym[sel], np.ones(int(sel.sum()))))
        return _concat(parts)
    if name == "flag_mass":
        present = [(f, upd.columns[c]) for f, c in enumerate(FLAG_COUNT_COLUMNS) if c in upd.columns]
        if not present:
            return None
        for f, col in present:
            ok = np.isfinite(col) & (np.where(np.isfinite(col), col, 0.0) > 0)
            w = np.where(ok, col, 0.0)
            if level == "network":
                parts.append((rows[ok], np.zeros(int(ok.sum()), dtype=np.int64), np.full(int(ok.sum()), f, dtype=np.int64),
                              w[ok]))
            else:
                for k, counts in _participations(m, roles):
                    sel = ok & counts
                    parts.append((rows[sel], m[sel, k], np.full(int(sel.sum()), f, dtype=np.int64), w[sel]))
        return _concat(parts)
    if name == "peer":
        everyone = _participations(m, "any")
        for k, counts in _participations(m, roles):
            for k2, counts2 in everyone:
                if k2 == k:
                    continue
                sel = counts & counts2
                parts.append((rows[sel], m[sel, k], m[sel, k2], np.ones(int(sel.sum()))))
        return _concat(parts)
    # src_entity / dst_entity (network level)
    k = 0 if name == "src_entity" else 1
    sel = m[:, k] >= 0
    parts.append((rows[sel], np.zeros(int(sel.sum()), dtype=np.int64), m[sel, k], np.ones(int(sel.sum()))))
    return _concat(parts)


def update_arrays_from_window(window: WindowBatch, b: int) -> UpdateArrays:
    """The traffic fields of window b of a batch (real updates only), from its FieldBatch and entity table."""
    fb = window.fields
    real = window.update_mask[b].cpu().numpy().astype(bool)
    contributing = np.isin(fb.status[b].cpu().numpy(), CONTRIBUTING_CODES)
    values = fb.values[b].to(torch.float64).cpu().numpy()
    cols: dict[str, np.ndarray] = {}
    for name in TRAFFIC_COLUMNS:
        if name in fb.column_names:
            j = fb.column_names.index(name)
            cols[name] = np.where(contributing[:, j], values[:, j], np.nan)[real]
    return UpdateArrays(time=window.update_time[b].cpu().numpy().astype(np.float64)[real],
                        members=window.update_entities[b].cpu().numpy().astype(np.int64)[real], columns=cols)


def update_arrays_from_columnar(values: np.ndarray, status: np.ndarray, column_names: Sequence[str], time: np.ndarray,
                                members: np.ndarray) -> UpdateArrays:
    """The traffic fields of rows of a `ColumnarUpdates` log (values/status [n, C], time [n], members [n, 3])."""
    contributing = np.isin(np.asarray(status), CONTRIBUTING_CODES)
    vals = np.asarray(values, dtype=np.float64)
    names = list(column_names)
    cols = {name: np.where(contributing[:, names.index(name)], vals[:, names.index(name)], np.nan)
            for name in TRAFFIC_COLUMNS if name in names}
    return UpdateArrays(time=np.asarray(time, dtype=np.float64), members=np.asarray(members, dtype=np.int64), columns=cols)


@dataclass(frozen=True)
class TrafficEntropy:
    """Traffic entropies of a window batch at its triggers (float64; NaN where there is no sample).

    entity[name]: [B, M, V]; entity_samples[name]: [B, M, V]; network[name]: [B, M];
    network_samples[name]: [B, M]; network_jsd[name]: [B, M] divergence from the previous valid trigger
    of the same window (NaN at the first one).
    """

    entity: dict[str, torch.Tensor]
    entity_samples: dict[str, torch.Tensor]
    network: dict[str, torch.Tensor]
    network_samples: dict[str, torch.Tensor]
    network_jsd: dict[str, torch.Tensor]


def traffic_entropies(window: WindowBatch, *, config: TrafficConfig, window_seconds: float,
                      times: torch.Tensor | None = None, trigger_mask: torch.Tensor | None = None) -> TrafficEntropy:
    """Entropies of the traffic distributions of the state window (tau - window_seconds, tau] at each trigger.

    times float64 [B, M] (relative to each window's origin) and trigger_mask bool [B, M] default to the
    window's own triggers. Entity keys are window entity indices.
    """
    if not (math.isfinite(window_seconds) and window_seconds > 0):
        raise ValueError("window_seconds must be finite and > 0")
    tt = window.triggers.time if times is None else times
    tm = window.triggers.mask if trigger_mask is None else trigger_mask
    b_n, m_n = tt.shape
    v_n = window.entity_mask.shape[1]
    out_e = {d: torch.full((b_n, m_n, v_n), math.nan, dtype=torch.float64) for d in config.entity_distributions}
    out_es = {d: torch.zeros((b_n, m_n, v_n), dtype=torch.float64) for d in config.entity_distributions}
    out_n = {d: torch.full((b_n, m_n), math.nan, dtype=torch.float64) for d in config.network_distributions}
    out_ns = {d: torch.zeros((b_n, m_n), dtype=torch.float64) for d in config.network_distributions}
    out_j = {d: torch.full((b_n, m_n), math.nan, dtype=torch.float64) for d in config.network_distributions}
    for b in range(b_n):
        upd = update_arrays_from_window(window, b)
        tau = tt[b].to(torch.float64).cpu().numpy()
        on = tm[b].cpu().numpy().astype(bool)
        for level, names, k_n in (("entity", config.entity_distributions, v_n), ("network", config.network_distributions, 1)):
            for name in names:
                ev = distribution_events(upd, name, level=level, roles=config.roles)
                if ev is None:
                    continue
                t_ev = upd.time[ev.row]
                inside = (t_ev[:, None] > tau[None, :] - window_seconds) & (t_ev[:, None] <= tau[None, :]) & on[None, :]
                e_idx, m_idx = np.nonzero(inside)                                       # (event, trigger) pairs
                grp = torch.from_numpy(m_idx * k_n + ev.key[e_idx])
                gc = grouped_counts(grp, torch.from_numpy(ev.symbol[e_idx]), m_n * k_n,
                                    torch.from_numpy(ev.weight[e_idx]))
                est = entropy_grouped(gc, estimator=config.estimator)
                if level == "entity":
                    out_e[name][b] = est.value.reshape(m_n, k_n)
                    out_es[name][b] = est.samples.reshape(m_n, k_n)
                    continue
                out_n[name][b] = est.value
                out_ns[name][b] = est.samples
                # Divergence between the distributions of successive valid triggers of this window.
                if gc.group.numel():
                    sym, s_inv = torch.unique(torch.from_numpy(ev.symbol[e_idx]), return_inverse=True)
                    dense = torch.zeros(m_n, sym.numel(), dtype=torch.float64)
                    dense.index_put_((torch.from_numpy(m_idx), s_inv), torch.from_numpy(ev.weight[e_idx]), accumulate=True)
                    valid_m = [mm for mm in range(m_n) if on[mm]]
                    for prev, cur in zip(valid_m, valid_m[1:], strict=False):
                        out_j[name][b, cur] = jensen_shannon(dense[prev], dense[cur])
    return TrafficEntropy(entity=out_e, entity_samples=out_es, network=out_n, network_samples=out_ns, network_jsd=out_j)


@dataclass
class _KeyState:
    # One histogram: symbol -> [weight sum, event count]; totals N, sum n log n (compensated) and changes.
    counts: dict[int, list[float]] = field(default_factory=dict)
    n: float = 0.0
    s: float = 0.0
    c: float = 0.0
    changes: int = 0


class SlidingHistograms:
    """Weighted symbol histograms of several keys over a sliding time window (module docstring, AS-778)."""

    def __init__(self, window_seconds: float, *, resync_every: int) -> None:
        if not (math.isfinite(window_seconds) and window_seconds > 0):
            raise ValueError("window_seconds must be finite and > 0")
        if resync_every < 1:
            raise ValueError("resync_every must be >= 1")
        self.window = float(window_seconds)
        self.resync_every = int(resync_every)
        self._live: deque[tuple[float, int, int, float]] = deque()      # events inside the window, time order
        self._pending: deque[tuple[float, int, int, float]] = deque()   # ingested, after the last reading time
        self._keys: dict[int, _KeyState] = {}
        self._last_time = -math.inf

    def push(self, time: float, key: int, symbol: int, weight: float = 1.0) -> None:
        """Queue one event (times must be non-decreasing); it enters the window when a reading reaches it."""
        if time < self._last_time:
            raise ValueError("events must arrive in non-decreasing time order")
        if not (math.isfinite(weight) and weight > 0):
            return
        self._last_time = float(time)
        self._pending.append((float(time), int(key), int(symbol), float(weight)))

    def _accumulate(self, st: _KeyState, delta: float) -> None:
        # Neumaier compensated summation of the running sum n log n.
        t = st.s + delta
        if abs(st.s) >= abs(delta):
            st.c += (st.s - t) + delta
        else:
            st.c += (delta - t) + st.s
        st.s = t

    def _change(self, key: int, symbol: int, weight: float, events: int) -> None:
        # Add (weight > 0, events = +1) or remove (weight < 0, events = -1) one event of `symbol` under `key`.
        st = self._keys.setdefault(key, _KeyState())
        cell = st.counts.get(symbol)
        old = cell[0] if cell is not None else 0.0
        n_ev = (cell[1] if cell is not None else 0) + events
        new = max(old + weight, 0.0) if n_ev > 0 else 0.0           # an emptied cell is exactly zero
        if n_ev > 0:
            st.counts[symbol] = [new, n_ev]
        else:
            st.counts.pop(symbol, None)
        if not st.counts:
            del self._keys[key]                                       # an empty histogram holds no state
            return
        st.n += new - old
        self._accumulate(st, (new * math.log(new) if new > 0 else 0.0) - (old * math.log(old) if old > 0 else 0.0))
        st.changes += 1
        if st.changes % self.resync_every == 0:
            # Exact recomputation: the compensated running sums never drift over a long run.
            st.s = math.fsum(v[0] * math.log(v[0]) for v in st.counts.values() if v[0] > 0)
            st.c = 0.0
            st.n = math.fsum(v[0] for v in st.counts.values())

    def advance(self, now: float) -> None:
        """Commit queued events with time <= now and evict events with time <= now - window."""
        while self._pending and self._pending[0][0] <= now:
            ev = self._pending.popleft()
            self._live.append(ev)
            self._change(ev[1], ev[2], ev[3], +1)
        cut = now - self.window
        while self._live and self._live[0][0] <= cut:
            ev = self._live.popleft()
            self._change(ev[1], ev[2], -ev[3], -1)

    def active_keys(self) -> list[int]:
        """Keys with at least one event in the window."""
        return sorted(self._keys)

    def counts(self, key: int) -> dict[int, float]:
        """The key's histogram {symbol: weight} (empty if the key has no event in the window)."""
        st = self._keys.get(key)
        return {} if st is None else {s: v[0] for s, v in st.counts.items()}

    def estimate(self, key: int, estimator: str) -> tuple[float, float]:
        """(entropy in nats, sample size) of the key's histogram; (NaN, 0) when it is empty."""
        _check_estimator(estimator)
        st = self._keys.get(key)
        if st is None or st.n <= 0:
            return math.nan, 0.0
        if estimator == "chao_shen":
            counts = torch.tensor([v[0] for v in st.counts.values()], dtype=torch.float64)
            est = entropy_from_counts(counts, estimator="chao_shen")
            return float(est.value), float(est.samples)
        h = math.log(st.n) - (st.s + st.c) / st.n
        if estimator == "miller_madow":
            h += (len(st.counts) - 1) / (2.0 * st.n)
        return h, st.n


@dataclass(frozen=True)
class TrafficReading:
    """Traffic entropies at one reading time (streaming): plain floats, NaN where there is no sample.

    network[name], network_samples[name], network_jsd[name] (divergence from the previous reading);
    entity[key][name] and entity_samples[key][name] for every key with an event in the window.
    """

    time: float
    network: dict[str, float]
    network_samples: dict[str, float]
    network_jsd: dict[str, float]
    entity: dict[int, dict[str, float]]
    entity_samples: dict[int, dict[str, float]]


class TrafficEntropyTracker:
    """Streaming traffic entropies over the sliding state window (module docstring)."""

    def __init__(self, config: TrafficConfig, *, window_seconds: float) -> None:
        self.config = config
        self.window_seconds = float(window_seconds)
        self._hist: dict[tuple[str, str], SlidingHistograms] = {}
        for level, names in (("entity", config.entity_distributions), ("network", config.network_distributions)):
            for name in names:
                self._hist[(level, name)] = SlidingHistograms(window_seconds, resync_every=config.resync_every)
        self._available: dict[tuple[str, str], bool] = {k: False for k in self._hist}
        self._previous: dict[str, dict[int, float]] = {}

    def observe(self, upd: UpdateArrays) -> None:
        """Queue the events of a block of state updates (time-sorted, after every update observed before)."""
        if upd.time.shape[0] and not bool(np.all(np.diff(upd.time) >= 0)):
            raise ValueError("observe needs time-sorted state updates")
        for (level, name), hist in self._hist.items():
            ev = distribution_events(upd, name, level=level, roles=self.config.roles)
            if ev is None:
                continue
            self._available[(level, name)] = True
            order = np.argsort(upd.time[ev.row], kind="stable")
            for i in order.tolist():
                hist.push(float(upd.time[ev.row[i]]), int(ev.key[i]), int(ev.symbol[i]), float(ev.weight[i]))

    def read(self, now: float) -> TrafficReading:
        """Entropies of the state window (now - window, now] (events after `now` stay queued)."""
        est = self.config.estimator
        net, net_n, net_j = {}, {}, {}
        ent: dict[int, dict[str, float]] = {}
        ent_n: dict[int, dict[str, float]] = {}
        for (level, name), hist in self._hist.items():
            hist.advance(now)
            if level == "network":
                h, n = hist.estimate(0, est) if self._available[(level, name)] else (math.nan, 0.0)
                net[name], net_n[name] = h, n
                cur = hist.counts(0)
                prev = self._previous.get(name)
                net_j[name] = jensen_shannon_counts(prev, cur) if prev is not None else math.nan
                self._previous[name] = cur
                continue
            for key in hist.active_keys():
                h, n = hist.estimate(key, est)
                ent.setdefault(key, {})[name] = h
                ent_n.setdefault(key, {})[name] = n
        return TrafficReading(time=float(now), network=net, network_samples=net_n, network_jsd=net_j, entity=ent,
                              entity_samples=ent_n)


__all__ = [
    "CONTRIBUTING_CODES", "FIELD_COLUMNS", "FLAG_COUNT_COLUMNS", "TRAFFIC_COLUMNS", "EntropyEstimate", "Events",
    "GroupedCounts", "SlidingHistograms", "TrafficEntropy", "TrafficEntropyTracker", "TrafficReading", "UpdateArrays",
    "distribution_events", "entropy_from_counts", "entropy_grouped", "grouped_counts", "jensen_shannon",
    "jensen_shannon_counts", "traffic_entropies", "update_arrays_from_columnar", "update_arrays_from_window",
]
