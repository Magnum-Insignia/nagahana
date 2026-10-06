"""Verifier heads: the trust value head and the calibration policy head (D-45; build-spec section 2.10).

Sources: [A-16], [A-21]; D-45: "a value head scores how far a forecast or an advice can be trusted
(learned from human feedback), and a policy head proposes the calibration correction". Decisions: D-21 (a
proposal is applied only on a HumanCommand, `gate.py`), D-65 (the heads learn the RLCD objective,
rlcd.py). Assumptions: AS-25.

Trust value head

    P(correct | forecast summary phi_f, Monitor statistics phi_m) = sigma(MLP([phi_f ; phi_m]))

phi_f (`forecast_features`, 12 numbers per trigger, all bounded): P_inf(K), mean P_inf, mean band width,
normalised stage entropy (H / log S, mean over steps), route concentration 1 - distinct/N, mode-route
weight, mean and max hazard, tanh of the mean step value and reward, mean PRM plausibility and a flag
saying whether the PRM was available. phi_m (`monitor_features`, 6 numbers): S+/h, S-/h (tanh), the
systematic gap, log(1 + alerts), the largest Page-Hinkley statistic / lambda (tanh), log(1 + scored)/10.
Trained on resolved decisions ("correct" comes from the outcome or the analyst's verdict): by binary
cross-entropy (`TrustValueHead.loss`) or by the Brier score of the RLCD objective (`rlcd.rlcr_trust_loss`).

Calibration policy head

For each output family, reliability statistics psi (`reliability_features`: per bin the share of pairs,
the mean confidence and the frequency; plus log(1 + n)/10, ECE and Brier) and a family embedding give

    log T = log(T_max) * tanh(MLP([psi ; e_family]))        (T in [1/T_max, T_max] = [0.25, 4] by default)

Training target: the temperature that optimises the family's resolved pairs (the maximum-likelihood
temperature `calibration.ml_temperature`, or the RLCD temperature `rlcd.rlcd_temperature`); loss
(log T - log T_target)^2. `propose` returns a `CalibrationProposal` only for families with at least
`cfg.min_pairs` scored pairs.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch
from torch import nn

from nagahana.evaluation.calibration import ece
from nagahana.models.batch import ForecastOut
from nagahana.models.config.components import VerifierConfig
from nagahana.models.verifier.reports import OUTPUT_FAMILIES, CalibrationProposal, DriftReport
from nagahana.nn.mlp import SwiGLU
from nagahana.nn.norms import RMSNorm

N_FORECAST_FEATURES = 12
N_MONITOR_FEATURES = 6


def forecast_features(out: ForecastOut, plausibility: torch.Tensor | None = None) -> torch.Tensor:
    """phi_f [B, M, 12] (module docstring). plausibility: PRM probabilities [B, M, N, K] or None.

    Precision: phi_f is the float32 input of the trust network (a learned float32 head), so the float64
    forecast outputs (D-54) are narrowed here, at the network boundary, and nowhere upstream (AS-450).
    """
    w = out.route_weight.float()                                                   # [B, M, N]
    s = out.stage.float().clamp_min(1e-12)
    ent = -(s * torch.log(s)).sum(-1).mean(-1) / math.log(s.shape[-1])             # [B, M]
    band = (out.p_inf_band[..., 1] - out.p_inf_band[..., 0]).float().mean(-1)
    hz = (w.unsqueeze(-1) * out.hazard.float()).sum(-2)                            # route-weighted hazard [B, M, K]
    val = torch.tanh((w.unsqueeze(-1) * out.step_value.float()).sum(-2).mean(-1))
    rew = torch.tanh((w.unsqueeze(-1) * out.step_reward.float()).sum(-2).mean(-1))
    conc = 1.0 - out.route_distinct.float() / out.routes_n
    mode_w = w.max(-1).values
    if plausibility is None:
        plaus, flag = torch.full_like(mode_w, 0.5), torch.zeros_like(mode_w)
    else:
        plaus, flag = (w.unsqueeze(-1) * plausibility.float()).sum(-2).mean(-1), torch.ones_like(mode_w)
    feats = [out.p_inf[..., -1].float(), out.p_inf.float().mean(-1), band, ent, conc, mode_w,
             hz.mean(-1), hz.max(-1).values, val, rew, plaus, flag]
    return torch.stack(feats, dim=-1)


def monitor_features(report: DriftReport, *, cusum_h: float, ph_lambda: float) -> torch.Tensor:
    """phi_m [6] from a Monitor drift report (module docstring)."""
    ph = max((r.ph_statistic for r in report.regions), default=0.0)
    return torch.tensor([
        math.tanh(report.cusum_up / cusum_h), math.tanh(report.cusum_down / cusum_h),
        report.systematic_gap if report.systematic_gap is not None else 0.0,
        math.log1p(len(report.alerts)), math.tanh(ph / ph_lambda), math.log1p(report.n_scored) / 10.0,
    ], dtype=torch.float32)


def reliability_features(p: torch.Tensor, y: torch.Tensor, *, bins: int) -> torch.Tensor:
    """psi [3 bins + 3]: per bin (share, mean confidence, frequency; 0 for empty bins), log(1 + n)/10, ECE, Brier.

    Computed in float64 (D-54); returned as float32 because psi is the input of the calibration network.
    """
    pp, yy = p.double().flatten(), y.double().flatten()
    n = pp.numel()
    edges = torch.linspace(0, 1, bins + 1, dtype=torch.float64)
    idx = torch.clamp(torch.bucketize(pp, edges, right=True) - 1, 0, bins - 1)
    share = torch.zeros(bins, dtype=torch.float64)
    conf = torch.zeros(bins, dtype=torch.float64)
    freq = torch.zeros(bins, dtype=torch.float64)
    if n:
        cnt = torch.bincount(idx, minlength=bins).double()
        share = cnt / n
        conf = torch.bincount(idx, weights=pp, minlength=bins) / cnt.clamp_min(1)
        freq = torch.bincount(idx, weights=yy, minlength=bins) / cnt.clamp_min(1)
    e = ece(pp, yy, bins=bins) if n else 0.0
    brier = float(((pp - yy) ** 2).mean()) if n else 0.0
    return torch.cat([share, conf, freq, torch.tensor([math.log1p(n) / 10.0, e, brier], dtype=torch.float64)]).float()


class _MLP(nn.Module):
    """in -> dim -> SwiGLU residual -> out (pre-norm)."""

    def __init__(self, d_in: int, dim: int, hidden: int, d_out: int) -> None:
        super().__init__()
        self.inp = nn.Linear(d_in, dim)
        self.norm = RMSNorm(dim)
        self.mlp = SwiGLU(dim, hidden)
        self.out_norm = RMSNorm(dim)
        self.out = nn.Linear(dim, d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.inp(x)
        h = h + self.mlp(self.norm(h))
        return self.out(self.out_norm(h))


class TrustValueHead(nn.Module):
    """P(forecast or advice is right | phi_f, phi_m) (module docstring)."""

    def __init__(self, cfg: VerifierConfig) -> None:
        super().__init__()
        self.net = _MLP(N_FORECAST_FEATURES + N_MONITOR_FEATURES, cfg.dim, cfg.mlp_hidden, 1)

    def forward(self, forecast_feats: torch.Tensor, monitor_feats: torch.Tensor) -> torch.Tensor:
        """Logit of P(correct): forecast_feats [..., 12], monitor_feats [6] or [..., 6] -> [...]."""
        m = monitor_feats.expand(*forecast_feats.shape[:-1], -1)
        return self.net(torch.cat([forecast_feats.float(), m.float()], dim=-1)).squeeze(-1)

    @staticmethod
    def loss(logit: torch.Tensor, correct: torch.Tensor) -> torch.Tensor:
        return nn.functional.binary_cross_entropy_with_logits(logit.float(), correct.float())


class CalibrationPolicyHead(nn.Module):
    """Proposes log T per output family from reliability statistics (module docstring)."""

    def __init__(self, cfg: VerifierConfig) -> None:
        super().__init__()
        self.cfg = cfg
        if not math.isclose(cfg.temperature_min * cfg.temperature_max, 1.0):
            raise ValueError("the tanh parameterisation needs a symmetric interval in log T (t_min * t_max = 1)")
        d_in = 3 * cfg.reliability_bins + 3
        self.family_emb = nn.Embedding(cfg.n_output_families, cfg.dim)
        self.inp = nn.Linear(d_in, cfg.dim)
        self.net = _MLP(cfg.dim, cfg.dim, cfg.mlp_hidden, 1)

    def forward(self, reliability: torch.Tensor, family: torch.Tensor) -> torch.Tensor:
        """log T: reliability [F, 3 bins + 3], family long [F] -> [F] in [log t_min, log t_max]."""
        h = self.inp(reliability.float()) + self.family_emb(family)
        return math.log(self.cfg.temperature_max) * torch.tanh(self.net(h).squeeze(-1))

    @staticmethod
    def loss(log_t: torch.Tensor, target_t: torch.Tensor) -> torch.Tensor:
        """(log T - log T_target)^2 averaged."""
        return ((log_t - torch.log(target_t.to(log_t.dtype))) ** 2).mean()

    @torch.no_grad()
    def propose(self, pairs: Mapping[str, tuple[torch.Tensor, torch.Tensor]]) -> CalibrationProposal | None:
        """Proposal for families with >= cfg.min_pairs scored pairs (p, y). None if no family qualifies."""
        fams = [f for f, (p, _) in pairs.items() if p.numel() >= self.cfg.min_pairs]
        if not fams:
            return None
        feats = torch.stack([reliability_features(*pairs[f], bins=self.cfg.reliability_bins) for f in fams])
        codes = torch.tensor([OUTPUT_FAMILIES.index(f) for f in fams])
        temps = torch.exp(self(feats, codes)).clamp(self.cfg.temperature_min, self.cfg.temperature_max)
        return CalibrationProposal(temperatures={f: float(t) for f, t in zip(fams, temps, strict=True)}, source="policy-head",
                                   n_pairs={f: int(pairs[f][0].numel()) for f in fams},
                                   t_min=self.cfg.temperature_min, t_max=self.cfg.temperature_max)
