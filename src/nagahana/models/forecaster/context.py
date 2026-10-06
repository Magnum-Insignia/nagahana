"""Context of an imagined future: which Imagination states the Forecaster reads at a trigger (build-spec §2.8).

Purpose
-------
At each trigger the Forecaster (and the Verifier's process-reward model, which must see what the
Forecaster saw) reads a fixed-size set of context positions from TAAFT's analysis (`AnalysisOut`):

    context = [ G_adv adversary-hypothesis slots ; top-C entity states by compromise belief ]

(16 + 48 = 64 at L). This module selects them (`gather_context`, no parameters), encodes them
(`ContextEncoder`) and pools them into one trigger summary σ (`SummaryPool`).

Owner sources: [A-14] (the Forecaster reads Imagination), D-35 (Forecaster reads Imagination, never
writes the Environment). Decisions: D-49 (no index positions: entity order carries no meaning).
Assumptions: AS-22 (STAGED coupling: the analysis is read through a stop-gradient until the joint
phase), AS-18 (compromise readout drives the selection).

Maths
-----
- Selection: for trigger (b, m), score_v = p_v (compromise readout) for active entities (token_mask),
  −∞ otherwise; the C highest-scoring entities are kept (fewer if fewer are active; the rest padded
  and masked).
- Encoding of context position i (entity or slot):

      x_i = W_c c_i + W_y y_i + W_r [p_i ; π^stage_i] + e_type(i)

  with c the TAAFT thinking-stream context, y the refined hypothesis, p and π^stage the compromise
  and stage readouts (zeros for adversary slots), e_type ∈ {adversary, entity}. No index embedding:
  the set of entity positions is permutation-equivariant (D-49).
- Summary: σ = Attn(q_σ; {x_i}) with a learned query q_σ and the null key of `nn.attention`
  (a trigger without any valid position pools to the null value, never NaN).

Invariants: padded entity positions are masked everywhere; permuting entity tokens of the analysis
permutes nothing observable after selection (selection is by score, ties by index).

Extension points: other selection scores (e.g. malignity, energy) can replace compromise in
`gather_context(score=...)`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import nn

from nagahana.models.batch import AnalysisOut
from nagahana.nn.attention import MultiHeadAttention
from nagahana.nn.norms import RMSNorm


def stage_probabilities(stage: torch.Tensor) -> torch.Tensor:
    """Stage readout as probabilities: kept if it already is a distribution, else softmax of logits. [..., S]."""
    s = stage.float()
    is_prob = bool((s >= 0).all()) and bool(torch.allclose(s.sum(-1), torch.ones_like(s[..., 0]), atol=1e-4))
    return s if is_prob else torch.softmax(s, dim=-1)


@dataclass
class GatheredContext:
    """Raw context features of every trigger row (Bt = B·M), before learned projection.

    context: [Bt, T, d_context]; hyp: [Bt, T, d_hyp]; readout: [Bt, T, 1 + n_stages];
    is_entity: bool [T] (first G positions are adversary slots); valid: bool [Bt, T];
    entity_index: long [Bt, C] window entity index of each entity position (−1 = padding);
    n_adv: G; n_entities: C.
    """

    context: torch.Tensor
    hyp: torch.Tensor
    readout: torch.Tensor
    is_entity: torch.Tensor
    valid: torch.Tensor
    entity_index: torch.Tensor
    n_adv: int
    n_entities: int


def gather_context(analysis: AnalysisOut, n_entities: int, *,
                   read: Callable[[torch.Tensor], torch.Tensor] | None = None) -> GatheredContext:
    """Select [adversary slots ; top-C entities by compromise] for every trigger (see module docstring).

    `read`: how the analysis tensors are read, e.g. `policy_value.head_input` under the coupling
    (AS-22 STAGED: stop-gradient in the first phase). None = read as is.
    """
    rd = read if read is not None else (lambda x: x)
    ctx_all, y_all = rd(analysis.context), rd(analysis.y)            # [B, M, V+G, d], [B, M, V+G, d_y]
    comp = rd(analysis.readouts["compromise"])                        # [B, M, V]
    stage = stage_probabilities(rd(analysis.readouts["stage"]))       # [B, M, V, S]
    b, m, vg, _ = ctx_all.shape
    v = comp.shape[-1]
    g = vg - v
    if g < 0:
        raise ValueError("AnalysisOut has fewer positions than compromise readouts")
    bt = b * m
    ctx_all = ctx_all.reshape(bt, vg, -1)
    y_all = y_all.reshape(bt, vg, -1)
    mask = analysis.token_mask.reshape(bt, vg)
    comp = comp.reshape(bt, v).float()
    stage = stage.reshape(bt, v, -1)

    # Top-C active entities by compromise belief; inactive entities score −∞ and are never chosen.
    c_take = min(n_entities, v)
    score = comp.masked_fill(~mask[:, :v], float("-inf"))            # [Bt, V]
    top_score, top_idx = torch.topk(score, c_take, dim=-1)            # [Bt, c_take]
    ent_valid = torch.isfinite(top_score)
    ent_index = torch.where(ent_valid, top_idx, torch.full_like(top_idx, -1))
    # Pad to C positions when the window has fewer entities than C.
    if c_take < n_entities:
        pad = n_entities - c_take
        top_idx = torch.cat([top_idx, top_idx.new_zeros(bt, pad)], dim=1)
        ent_valid = torch.cat([ent_valid, ent_valid.new_zeros(bt, pad)], dim=1)
        ent_index = torch.cat([ent_index, ent_index.new_full((bt, pad), -1)], dim=1)

    # Entity features gathered at the selected indices.                 # [Bt, C, ·]
    ent_ctx = torch.gather(ctx_all, 1, top_idx.unsqueeze(-1).expand(-1, -1, ctx_all.shape[-1]))
    ent_y = torch.gather(y_all, 1, top_idx.unsqueeze(-1).expand(-1, -1, y_all.shape[-1]))
    ent_read = torch.cat([comp.unsqueeze(-1), stage], dim=-1)        # [Bt, V, 1+S]
    ent_read = torch.gather(ent_read, 1, top_idx.unsqueeze(-1).expand(-1, -1, ent_read.shape[-1]))
    # Padded positions: zero features (they are masked anyway; zeros keep arithmetic clean).
    zero = ~ent_valid.unsqueeze(-1)
    ent_ctx, ent_y, ent_read = ent_ctx.masked_fill(zero, 0.0), ent_y.masked_fill(zero, 0.0), ent_read.masked_fill(zero, 0.0)

    # Adversary slots: positions V … V+G−1 of the analysis.             # [Bt, G, ·]
    adv_ctx, adv_y = ctx_all[:, v:], y_all[:, v:]
    adv_valid = mask[:, v:]
    adv_read = ent_read.new_zeros(bt, g, ent_read.shape[-1])

    is_entity = torch.cat([torch.zeros(g, dtype=torch.bool), torch.ones(n_entities, dtype=torch.bool)]).to(ctx_all.device)
    return GatheredContext(
        context=torch.cat([adv_ctx, ent_ctx], dim=1),
        hyp=torch.cat([adv_y, ent_y], dim=1),
        readout=torch.cat([adv_read, ent_read], dim=1),
        is_entity=is_entity,
        valid=torch.cat([adv_valid, ent_valid], dim=1),
        entity_index=ent_index,
        n_adv=g,
        n_entities=n_entities,
    )


class ContextEncoder(nn.Module):
    """x_i = W_c c_i + W_y y_i + W_r [p_i ; π^stage_i] + e_type(i)  (module docstring)."""

    def __init__(self, d_context: int, d_hyp: int, n_stages: int, dim: int) -> None:
        super().__init__()
        self.ctx_proj = nn.Linear(d_context, dim, bias=False)
        self.hyp_proj = nn.Linear(d_hyp, dim, bias=False)
        self.read_proj = nn.Linear(1 + n_stages, dim, bias=False)
        self.type_emb = nn.Embedding(2, dim)            # 0 adversary slot, 1 entity
        self.norm = RMSNorm(dim)

    def forward(self, g: GatheredContext) -> torch.Tensor:
        """→ [Bt, T, dim] (normalised; padded positions are whatever, they are masked downstream)."""
        x = self.ctx_proj(g.context.float()) + self.hyp_proj(g.hyp.float()) + self.read_proj(g.readout.float())
        x = x + self.type_emb(g.is_entity.long())[None]
        return self.norm(x)


class SummaryPool(nn.Module):
    """σ = Attn(q_σ; {x_i}): one learned query over the valid context positions (null key: never NaN)."""

    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.randn(dim) * 0.02)
        self.attn = MultiHeadAttention(dim, heads)

    def forward(self, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """x [Bt, T, dim], valid [Bt, T] → σ [Bt, dim]."""
        bt = x.shape[0]
        q = self.attn.project_q(self.query.expand(bt, 1, -1))         # [Bt, H, 1, d_h]
        k, v = self.attn.project_kv(x)                                 # [Bt, H, T, d_h]
        out, _ = self.attn.attend(q, k, v, allowed=valid[:, None, None, :])
        return out[:, 0]
