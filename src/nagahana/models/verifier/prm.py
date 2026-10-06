"""Process-reward model: per-step plausibility of imagined routes (build-spec section 2.10).

"both models need to have a process reward model idea but modified for each k states" [Q-24]. The
Verifier's PRM scores every imagined step of every route: P(step k of route n is plausible | context,
steps <= k). It is trained on human step labels (analyst feedback, `FeedbackKind.STEP_LABEL`; for public
datasets the dataset annotations, AS-25): Lightman et al., "Let's Verify Step by Step", arXiv:2305.20050
(process supervision of each step rather than only the outcome).

Decisions: D-21 (its weights change only under a HumanCommand), D-35 (it reads Imagination; it writes only
the Monitor), D-49 (step time t = k * window_seconds). Assumptions: AS-25, AS-415 (step labels).

Architecture: the same layout as the Forecaster's dynamics, with its own weights: `cfg.blocks` SelfBlocks
over [context (G slots + top-C entities) ; step 1 ... step K], causal over steps, time-rotary. Step input:
W_s sg(state_k) + E_tech[technique_k] + W_tgt x_(target_k) (or a learned "no target"). The Forecaster's
outputs and the analysis are read through a stop-gradient: the Verifier never trains the models it checks.

    plausibility_(n,k) = sigma(w . RMSNorm(h_(n,k)))

Loss: binary cross-entropy on labelled steps (label -1 = unknown, masked).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from nagahana.models.batch import AnalysisOut, ForecastOut
from nagahana.models.config.components import VerifierConfig
from nagahana.models.forecaster.context import ContextEncoder, gather_context
from nagahana.models.forecaster.sequence import RouteStack
from nagahana.nn.norms import RMSNorm


class ProcessRewardModel(nn.Module):
    """Per-step plausibility over imagined routes. See the module docstring."""

    def __init__(self, cfg: VerifierConfig, *, d_context: int, d_hyp: int, d_state: int, n_techniques: int,
                 n_stages: int, window_seconds: float) -> None:
        super().__init__()
        self.cfg, self.window_seconds, self.n_techniques = cfg, window_seconds, n_techniques
        dim = cfg.dim
        self.ctx_enc = ContextEncoder(d_context, d_hyp, n_stages, dim)
        self.state_in = nn.Linear(d_state, dim, bias=False)
        self.tech_emb = nn.Embedding(n_techniques, dim)
        self.tgt_in = nn.Linear(dim, dim, bias=False)
        self.null_target = nn.Parameter(torch.randn(dim) * 0.02)
        self.stack = RouteStack(dim, cfg.heads, cfg.blocks, cfg.mlp_hidden, p_min=1.0, p_max=604_800.0)
        self.norm = RMSNorm(dim)
        self.head = nn.Linear(dim, 1)

    def forward(self, analysis: AnalysisOut, forecast: ForecastOut) -> torch.Tensor:
        """Plausibility logits [B, M, N, K] of every imagined step."""
        g = gather_context(analysis, self.cfg.context_entities, read=lambda x: x.detach())
        tokens = self.ctx_enc(g)                                                    # [Bt, T, dim]
        bt, t, dim = tokens.shape
        b, m, n, k = forecast.hazard.shape
        # Steps: imagined state + technique + target (context entity slot of the target, or "no target").
        tech = forecast.route_actions[..., 0].reshape(bt, n, k).clamp(0, self.n_techniques - 1)
        ent = forecast.route_actions[..., 1].reshape(bt, n, k)
        match = (g.entity_index[:, None, None, :] == ent.unsqueeze(-1)) & (ent.unsqueeze(-1) >= 0)   # [Bt, N, K, C]
        has = match.any(-1)
        slot = match.float().argmax(-1)
        ent_tok = tokens[:, g.n_adv:]                                               # [Bt, C, dim]
        tgt = torch.gather(ent_tok.unsqueeze(1).expand(-1, n, -1, -1), 2,
                           slot.unsqueeze(-1).expand(-1, -1, -1, dim))              # [Bt, N, K, dim]
        tgt = torch.where(has.unsqueeze(-1), self.tgt_in(tgt), self.null_target.expand_as(tgt))
        x_steps = self.state_in(forecast.step_state.detach().reshape(bt, n, k, -1).float()) + self.tech_emb(tech) + tgt
        # Sequence per route: [context ; steps], rows Bt * N.
        rows = bt * n
        ctx_rows = tokens.unsqueeze(1).expand(-1, n, -1, -1).reshape(rows, t, dim)
        x = torch.cat([ctx_rows, x_steps.reshape(rows, k, dim)], dim=1)            # [R, T+K, dim]
        step_t = torch.arange(1, k + 1, dtype=torch.float64, device=x.device) * self.window_seconds
        times = torch.cat([torch.zeros(rows, t, dtype=torch.float64, device=x.device), step_t.expand(rows, -1)], dim=1)
        valid = g.valid.unsqueeze(1).expand(-1, n, -1).reshape(rows, t)
        allowed = torch.zeros(rows, 1, t + k, t + k, dtype=torch.bool, device=x.device)
        allowed[:, 0, :t, :t] = valid[:, None, :]
        allowed[:, 0, t:, :t] = valid[:, None, :]
        allowed[:, 0, t:, t:] = torch.ones(k, k, dtype=torch.bool, device=x.device).tril()
        out, _, _ = self.stack.dense(x, times, allowed)
        logits = self.head(self.norm(out[:, t:])).squeeze(-1)                      # [R, K]
        return logits.reshape(b, m, n, k)

    @staticmethod
    def loss(logits: torch.Tensor, step_labels: torch.Tensor) -> torch.Tensor:
        """BCE on labelled steps. step_labels: [B, M, N, K] in {0, 1}, -1 = unknown (masked)."""
        known = step_labels >= 0
        bce = F.binary_cross_entropy_with_logits(logits.float(), step_labels.clamp_min(0).float(), reduction="none")
        return (bce * known).sum() / known.sum().clamp_min(1)
