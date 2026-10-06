"""Noise and signal features ν_v per entity, as of each trigger (AS-36, AS-205; held D-26).

Purpose
-------
The owner named "colored noise, white noise, various noise analysis" as a TAAFT lens and is "not
sure yet" whether it stands on its own (D-26, held). AS-14 and AS-36 put it *inside* the information
lens: these features say how informative an entity's timing evidence is, and E_info uses them
(lenses.py). Nothing here is learned; the features are deterministic statistics of the entity's
event times, so they are auditable and reportable as "noise and signal" outputs (architecture §6).

Owner sources: [A-19]. Decisions: D-26 (held; this module is its content inside E_info, not a term
of its own), D-41 (absence is not zero: every feature has a validity flag), D-49 (time, not index).
Assumptions: AS-36 (which features), AS-205 (their exact definitions below).

Events
------
An entity's events are its TSTCT positions (one per update the entity takes part in as initiator
or responder, AS-41) with time ≤ τ. Their times t_1 < … < t_n (window-relative float64 seconds) and
gaps x_k = t_{k+1} − t_k are the inputs.

1. Event-time periodogram (Schuster 1898) and its peak ratio
-----------------------------------------------------------
For the point process of event times, the periodogram at frequency f is

    I(f) = | Σ_{k=1..n} exp(−2πi f t_k) |² / n.

- For a Poisson process and f well above 1/span, I(f) is approximately Exp(1): mean ≈ 1.
- For strictly periodic events with period T, every term at f = 1/T has phase 2π·(t_1/T + k), so
  I(1/T) = n exactly (tests/test_taaft_noise.py checks this closed form).

The peak ratio is ρ = max_f I(f) / mean_f I(f) over a fixed log-spaced grid of periods
[`noise_period_min`, `noise_period_max`]. Large ρ is evidence of periodic (beacon-like) timing under
jitter; BAYWATCH (Hu et al., DSN 2016) uses spectral peaks of event series for the same purpose.
The dominant period 1/argmax_f I(f) is reported with it.

Using event times directly (rather than a binned count series or the gap sequence) needs no bin
width, is exact for irregular times, and is what "periodogram of inter-arrival times" means for a
point process: the spectrum of the process the gaps generate. This reading is AS-205.

2. Aggregated-variance Hurst estimate (Taqqu, Teverovsky & Willinger, Fractals 1995)
-----------------------------------------------------------------------------------
For block sizes m = 1, 2, 4, …, 2^{K−1}, split the gap series into consecutive complete blocks of m
gaps and take the block means X^{(m)}_j. For a self-similar process

    Var(X^{(m)}) ∝ m^{2H − 2},

so the least-squares slope β of log Var(X^{(m)}) against log m gives H = 1 + β/2.
- I.i.d. gaps (a Poisson process): Var(X^{(m)}) = σ²/m exactly, β = −1, H = 0.5.
- Long-range dependent gaps (benign aggregate traffic is self-similar: Leland et al., IEEE/ACM ToN
  1994): H > 0.5.
A scale enters the fit only with at least `noise_min_blocks` complete blocks and a positive
variance; H needs at least two scales. H is clamped to [0, 1] for reporting.

3. Event count
--------------
log(1 + n): how much evidence there is at all.

As-of computation (no future leakage)
-------------------------------------
Every quantity above is a sum over the entity's events ≤ τ (Σ cos, Σ sin, block sums, counts), so it
is a per-entity prefix sum read at the entity's latest position ≤ τ (structure.py). One pass over
positions serves every trigger.

Output
------
values float32 [B, M, V, 4] in the order of `NOISE_FEATURES`, and valid bool [B, M, V, 4]. Invalid
entries hold 0 and must be read together with their flag (D-41); E_info embeds (value·valid, valid).

Extension points
----------------
More statistics (spectral slope β of 1/f^β noise, entropy of gaps, Lomb–Scargle with a floating
mean) can be appended to `NOISE_FEATURES`; E_info's input width follows `len(NOISE_FEATURES)`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from nagahana.governance.assumptions import assume
from nagahana.models.taaft.structure import event_groups, gather_rows, group_sort, segmented_cumsum

#: Feature order of ν (AS-205).
NOISE_FEATURES: tuple[str, ...] = ("log_peak_ratio", "log_period", "hurst", "log_events")


@dataclass(frozen=True)
class NoiseSpec:
    """Settings of the noise features (from `TAAFTConfig`, AS-205)."""

    n_frequencies: int
    period_min: float
    period_max: float
    scales: int
    min_events: int
    min_blocks: int

    def periods(self, device: torch.device | None = None) -> torch.Tensor:
        """Log-spaced periods (seconds), longest first, so the first maximum is the fundamental."""
        lo, hi = math.log10(self.period_min), math.log10(self.period_max)
        return torch.logspace(hi, lo, self.n_frequencies, dtype=torch.float64, device=device)


@dataclass
class NoiseFeatures:
    """ν per (window, trigger, entity). values float32 [B, M, V, F]; valid bool [B, M, V, F]."""

    values: torch.Tensor
    valid: torch.Tensor

    def named(self) -> dict[str, torch.Tensor]:
        """Feature name → [B, M, V] values (0 where invalid)."""
        return {n: self.values[..., i] for i, n in enumerate(NOISE_FEATURES)}


@torch.no_grad()
def noise_features(
    entity: torch.Tensor,
    time: torch.Tensor,
    mask: torch.Tensor,
    entity_latest: torch.Tensor,
    spec: NoiseSpec,
) -> NoiseFeatures:
    """Compute ν for every trigger and entity of a window batch. See the module docstring.

    entity: long [B, P]; time: float64 [B, P] (window-relative, sorted per window); mask: bool [B, P];
    entity_latest: long [B, M, V] (−1 = not yet seen at that trigger).
    """
    assume("AS-36", by=__name__)
    b, p = entity.shape
    v = entity_latest.shape[-1]
    t = time.to(torch.float64)
    groups = event_groups(entity, mask, v)                                     # [B, P]
    real = (groups < v).to(torch.float64)                                      # 1 on real events

    # ---------------------------------------------------------------- 1. periodogram sums
    freqs = 1.0 / spec.periods(entity.device)                                  # [F] (Hz), low → high
    phase = 2.0 * math.pi * torch.remainder(t[..., None] * freqs, 1.0)        # [B, P, F], reduced mod 1 cycle
    cs = segmented_cumsum(torch.cos(phase) * real[..., None], groups)          # [B, P, F]
    sn = segmented_cumsum(torch.sin(phase) * real[..., None], groups)
    n_ev = segmented_cumsum(real, groups)                                      # [B, P] events so far

    c_at = gather_rows(cs, entity_latest)                                      # [B, M, V, F]
    s_at = gather_rows(sn, entity_latest)
    n_at = gather_rows(n_ev, entity_latest)                                    # [B, M, V]
    seen = entity_latest >= 0
    n_at = torch.where(seen, n_at, torch.zeros_like(n_at))
    power = (c_at**2 + s_at**2) / n_at.clamp_min(1.0)[..., None]               # I(f)
    peak, arg = power.max(dim=-1)
    mean = power.mean(dim=-1)
    pg_valid = seen & (n_at >= spec.min_events) & (mean > 0)
    log_ratio = torch.log(peak.clamp_min(1e-12) / mean.clamp_min(1e-12))
    log_period = torch.log(1.0 / freqs[arg])

    # ---------------------------------------------------------------- 2. aggregated variance
    # Gap x at a position = time since the same entity's previous event (0 at its first event).
    # Rank r = number of earlier events of the entity; gap index k = r − 1.
    rank = n_ev - 1.0                                                          # [B, P] (float, exact ints)
    # Σ_{k < r} x_k telescopes to t_r − t_0; t_0 (the group's first event time) is the prefix sum of
    # t·𝟙[rank = 0], which only the first event contributes to.
    t0 = segmented_cumsum(t * real * (rank == 0).to(t.dtype), groups)         # [B, P]
    gap_cum = (t - t0) * real                                                  # C_r = t_r − t_0
    log_m: list[float] = []
    s1s: list[torch.Tensor] = []
    s2s: list[torch.Tensor] = []
    ns: list[torch.Tensor] = []
    for s in range(spec.scales):
        m = 2**s
        # Block j of m gaps ends at rank r = (j + 1)·m; its mean is (C_r − C_{r−m}) / m where
        # C_r = t_r − t_0. Find C_{r−m}: the same entity's position m events back.
        back = _shift_back(gap_cum, groups, m)                                 # [B, P] C at rank r − m
        ends = (real > 0) & (rank >= m) & (torch.remainder(rank, m) == 0)
        blk = torch.where(ends, (gap_cum - back) / m, torch.zeros_like(gap_cum))
        f = ends.to(t.dtype)
        s1s.append(segmented_cumsum(blk, groups))
        s2s.append(segmented_cumsum(blk * blk, groups))
        ns.append(segmented_cumsum(f, groups))
        log_m.append(math.log(m))
    s1 = gather_rows(torch.stack(s1s, dim=-1), entity_latest)                  # [B, M, V, K]
    s2 = gather_rows(torch.stack(s2s, dim=-1), entity_latest)
    nb = gather_rows(torch.stack(ns, dim=-1), entity_latest)
    var = (s2 - s1 * s1 / nb.clamp_min(1.0)) / (nb - 1.0).clamp_min(1.0)       # unbiased block-mean variance
    ok = seen[..., None] & (nb >= max(2, spec.min_blocks)) & (var > 1e-300)
    x = torch.tensor(log_m, dtype=torch.float64, device=entity.device)         # [K]
    y = torch.log(torch.where(ok, var, torch.ones_like(var)))
    w = ok.to(torch.float64)
    sw = w.sum(-1)
    xm = (w * x).sum(-1) / sw.clamp_min(1.0)
    ym = (w * y).sum(-1) / sw.clamp_min(1.0)
    sxx = (w * (x - xm[..., None]) ** 2).sum(-1)
    sxy = (w * (x - xm[..., None]) * (y - ym[..., None])).sum(-1)
    h_valid = (sw >= 2) & (sxx > 0)
    slope = sxy / sxx.clamp_min(1e-12)
    hurst = (1.0 + slope / 2.0).clamp(0.0, 1.0)

    # ---------------------------------------------------------------- 3. assemble (invalid → 0)
    log_events = torch.log1p(n_at)
    vals = torch.stack([log_ratio, log_period, hurst, log_events], dim=-1)    # [B, M, V, 4]
    valid = torch.stack([pg_valid, pg_valid, h_valid, seen], dim=-1)
    vals = torch.where(valid, vals, torch.zeros_like(vals))
    return NoiseFeatures(values=vals.to(torch.float32), valid=valid)


def event_periodogram(times: torch.Tensor, periods: torch.Tensor) -> torch.Tensor:
    """Direct Schuster periodogram I(f) = |Σ_k e^{−2πi f t_k}|² / n of one event list (reference form).

    times: float64 [n]; periods: float64 [F] → I [F]. `noise_features` computes the same sums as
    per-entity prefix sums; this direct form exists for checks and for reporting one entity.
    """
    ph = 2.0 * math.pi * torch.remainder(times.to(torch.float64)[:, None] / periods[None, :], 1.0)
    n = max(1, times.shape[0])
    return (torch.cos(ph).sum(0) ** 2 + torch.sin(ph).sum(0) ** 2) / n


def _shift_back(x: torch.Tensor, groups: torch.Tensor, m: int) -> torch.Tensor:
    """Value of `x` at the same group's position m events earlier (0 where there is none).

    x: [B, P]; groups: long [B, P]. Works in grouped order: positions m apart inside one group.
    """
    order, inverse = group_sort(groups)
    xs = x.gather(1, order)
    gs = groups.gather(1, order)
    out = torch.zeros_like(xs)
    if m < xs.shape[1]:
        same = gs[:, m:] == gs[:, :-m]
        out[:, m:] = torch.where(same, xs[:, :-m], torch.zeros_like(xs[:, :-m]))
    return out.gather(1, inverse)
