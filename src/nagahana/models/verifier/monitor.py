"""The Monitor region's statistics: latent drift, residual CUSUM, systematic gap, poisoning alerts (build-spec section 2.10).

"[The Verifier] computes the memory drift analysis (outcome forecast pair) which will allow us to analyze
the poisoning scenarios" [A-16]. The Verifier is the only writer of the Monitor (D-35; `memory/access.py`,
checked on construction). Every statistic here only raises alerts for human review; nothing feeds back
into the model (D-21: "it would conduct the poisoning again" [A-16]). The feedback workflow also uses a
fresh Monitor per side to check a candidate update's held-out pairs for new drift alerts (evaluation.py).

Assumptions: AS-25 (drift mechanics), AS-258 (Page-Hinkley on the standardised latent deviation, delta /
lambda / warm-up values), AS-259 (systematic-gap threshold).

1. Welford running moments per region (Welford, Technometrics 1962; batch merge of Chan, Golub and
   LeVeque 1979): for a batch of n_b rows with mean xbar_b and centred sum of squares M_b,

       n = n_a + n_b,  d = xbar_b - xbar_a,  xbar = xbar_a + d n_b / n,  M = M_a + M_b + d^2 n_a n_b / n,  var = M / (n - 1)

2. Page-Hinkley alarm (Page, "Continuous inspection schemes", Biometrika 41, 1954; Hinkley, Biometrika 58,
   1971) on the standardised deviation of each new batch before it updates the moments:

       x_t = mean_d (z_(t,d) - mu_d)^2 / var_d            (about 1 while the latent distribution is stable)
       m_t = sum_(i<=t) (x_i - xbar_i - delta),  PH_t = m_t - min_(i<=t) m_i,  alarm when PH_t > lambda

   (xbar_i the running mean of x; only after `ph_warmup` rows, so the moments are trustworthy.)

3. Two-sided CUSUM on residuals e_t = f_t - y_t of scored resolutions (Page 1954):

       S+_t = max(0, S+_(t-1) + e_t - k),   S-_t = max(0, S-_(t-1) - e_t - k),   alarm when S+ or S- > h

   k = `cusum_k` (allowance), h = `cusum_h`. S+ grows when forecasts run high (false alarms), S- when
   outcomes happen that were not forecast (misses: the dangerous direction for a defender).

4. Systematic gap over the last W = `drift_window` scored resolutions: G = |sum (f - y)| / W; an alert when
   G > `gap_threshold`. Responded-to pairs are excluded from 3 and 4 [Q-38].

Precision (D-54): Welford moments are float64 tensors (inputs cast with `.double()` before any arithmetic);
Page-Hinkley and CUSUM accumulate Python floats (IEEE double). Nothing here is float32.

Invariants (tested): CUSUM alarms after a planted shift and not before (fixed seed); responded-to pairs
never move the statistics; the Monitor never changes a model; its statistics are float64.
"""

from __future__ import annotations

from collections import deque

import torch

from nagahana.core.roles import Role
from nagahana.governance.assumptions import assume
from nagahana.memory.access import Op, Region, check
from nagahana.models.config.components import VerifierConfig
from nagahana.models.verifier.reports import DriftReport, PoisoningAlert, RegionDrift
from nagahana.roles.contracts import OutcomeForecastPair


class Welford:
    """Running mean and variance of d-dimensional rows (batch-merged Welford; module docstring, part 1)."""

    def __init__(self, dim: int) -> None:
        self.n = 0
        self.mean = torch.zeros(dim, dtype=torch.float64)
        self.m2 = torch.zeros(dim, dtype=torch.float64)

    def update(self, x: torch.Tensor) -> None:
        """Merge a batch x [n_b, d]."""
        xb = x.detach().double().reshape(-1, self.mean.shape[0])
        nb = xb.shape[0]
        if nb == 0:
            return
        mb = xb.mean(0)
        m2b = ((xb - mb) ** 2).sum(0)
        n = self.n + nb
        delta = mb - self.mean
        self.mean = self.mean + delta * (nb / n)
        self.m2 = self.m2 + m2b + delta ** 2 * (self.n * nb / n)
        self.n = n

    @property
    def var(self) -> torch.Tensor:
        """Unbiased per-dimension variance (zeros until n >= 2)."""
        return self.m2 / (self.n - 1) if self.n > 1 else torch.zeros_like(self.m2)


class PageHinkley:
    """Page-Hinkley test for an increase in the mean of a scalar sequence (module docstring, part 2)."""

    def __init__(self, delta: float, lam: float) -> None:
        self.delta, self.lam = delta, lam
        self.n, self.mean, self.m, self.m_min = 0, 0.0, 0.0, 0.0

    def update(self, x: float) -> bool:
        self.n += 1
        self.mean += (x - self.mean) / self.n
        self.m += x - self.mean - self.delta
        self.m_min = min(self.m_min, self.m)
        return self.statistic > self.lam

    @property
    def statistic(self) -> float:
        return self.m - self.m_min


class Cusum:
    """Two-sided CUSUM on residuals (module docstring, part 3)."""

    def __init__(self, k: float, h: float) -> None:
        self.k, self.h = k, h
        self.up, self.down = 0.0, 0.0

    def update(self, e: float) -> tuple[bool, bool]:
        self.up = max(0.0, self.up + e - self.k)
        self.down = max(0.0, self.down - e - self.k)
        return self.up > self.h, self.down > self.h


class Monitor:
    """The Verifier's Monitor statistics and alert list. See the module docstring."""

    def __init__(self, cfg: VerifierConfig, *, region_dims: dict[str, int]) -> None:
        check(Role.VERIFIER, Region.MONITOR, Op.READ | Op.WRITE)      # the Verifier owns the Monitor (D-35)
        assume("AS-25", by=__name__)
        self.cfg = cfg
        self.welford = {r: Welford(d) for r, d in region_dims.items()}
        self.ph = {r: PageHinkley(cfg.ph_delta, cfg.ph_lambda) for r in region_dims}
        self.ph_alarmed = dict.fromkeys(region_dims, False)
        self.cusum = Cusum(cfg.cusum_k, cfg.cusum_h)
        self.cusum_alarmed = (False, False)
        self.window: deque[float] = deque(maxlen=cfg.drift_window)
        self.gap_alarmed = False
        self.n_scored = 0
        self.n_responded = 0
        self.alerts: list[PoisoningAlert] = []

    def observe_latents(self, region: str, z: torch.Tensor) -> list[PoisoningAlert]:
        """Feed a batch of latents [n, d] of `region` (Environment or Imagination). Returns alerts raised by this batch."""
        w, ph = self.welford[region], self.ph[region]
        new: list[PoisoningAlert] = []
        zz = z.detach().double().reshape(-1, w.mean.shape[0])
        if w.n >= self.cfg.ph_warmup and zz.shape[0]:
            var = w.var.clamp_min(1e-12)
            x = float((((zz - w.mean) ** 2) / var).mean())            # standardised deviation (before update)
            if ph.update(x) and not self.ph_alarmed[region]:
                self.ph_alarmed[region] = True
                new.append(PoisoningAlert("page_hinkley", region, ph.statistic, ph.lam, w.n,
                                          f"latent distribution of {region} drifted (Page-Hinkley {ph.statistic:.3g} > {ph.lam:g})"))
        w.update(zz)
        self.alerts.extend(new)
        return new

    def record_resolution(self, pair: OutcomeForecastPair) -> list[PoisoningAlert]:
        """Feed one resolved outcome-forecast pair. Responded-to pairs are counted and otherwise ignored [Q-38]."""
        if pair.responded_to:
            self.n_responded += 1
            return []
        self.n_scored += 1
        e = pair.predicted - float(pair.occurred)
        new: list[PoisoningAlert] = []
        up, down = self.cusum.update(e)
        if up and not self.cusum_alarmed[0]:
            new.append(PoisoningAlert("cusum_up", "ledger", self.cusum.up, self.cusum.h, self.n_scored,
                                      "forecasts run systematically high (CUSUM S+)"))
        if down and not self.cusum_alarmed[1]:
            new.append(PoisoningAlert("cusum_down", "ledger", self.cusum.down, self.cusum.h, self.n_scored,
                                      "outcomes occur that were not forecast (CUSUM S-)"))
        self.cusum_alarmed = (self.cusum_alarmed[0] or up, self.cusum_alarmed[1] or down)
        self.window.append(e)
        gap = self.systematic_gap()
        if gap is not None and gap > self.cfg.gap_threshold and not self.gap_alarmed:
            self.gap_alarmed = True
            new.append(PoisoningAlert("systematic_gap", "ledger", gap, self.cfg.gap_threshold, self.n_scored,
                                      f"systematic gap {gap:.3g} over the last {len(self.window)} resolutions"))
        self.alerts.extend(new)
        return new

    def systematic_gap(self) -> float | None:
        """|sum (f - y)| / W over the last W scored resolutions (None until the window is full)."""
        if len(self.window) < self.cfg.drift_window:
            return None
        return abs(sum(self.window)) / len(self.window)

    def report(self) -> DriftReport:
        regions = tuple(
            RegionDrift(region=r, n=w.n, mean_norm=float(w.mean.norm()), mean_variance=float(w.var.mean()) if w.n > 1 else 0.0,
                        ph_statistic=self.ph[r].statistic, ph_alarm=self.ph_alarmed[r])
            for r, w in self.welford.items())
        return DriftReport(regions=regions, cusum_up=self.cusum.up, cusum_down=self.cusum.down,
                           cusum_alarm=any(self.cusum_alarmed), systematic_gap=self.systematic_gap(),
                           gap_window_n=len(self.window), n_scored=self.n_scored, n_responded_to=self.n_responded,
                           alerts=tuple(self.alerts))
