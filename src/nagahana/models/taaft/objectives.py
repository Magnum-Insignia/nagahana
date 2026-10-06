"""Stage-4 objectives: self-supervised pretraining of TAAFT with CVG-AE, Decoder and TSTCT frozen.

Purpose (build-spec §3 stage 4; architecture §5 item 4; [A-24], [I-01])
-----------------------------------------------------------------------
"Self-supervised pretraining of TAAFT with those frozen: the intuition of adversaries, conflict,
cooperation, coordination, strategies, intentions, malignity and benignity." Labels never enter
TAAFT's forward pass; the one soft label used here (malignity) is a training target only.

Objectives
----------
1. **Masked-entity belief** (`masked_entity_loss`). Hide whole entities' states from TAAFT's view
   (`TAAFT.forward(hidden_entities=…)`: the token starts from a learned vector, its own Environment
   reads and its noise features are removed, and no other token reads it as a neighbour). The
   believed current latent of the hidden entity must predict the frozen CVG-AE posterior at its
   latest position: Gaussian NLL on z_c + soft cross-entropy on the categorical groups. What is
   not seen is not assumed absent (partial observability, D-41's spirit at entity level).
2. **Future latent** (`future_latent_loss`). The believed next latent at trigger m must predict
   the posterior at the entity's latest position as of trigger m + 1 (the world-model target
   P(S_{t+1} | S_≤t) read through TAAFT's belief).
3. **Contrastive energy** (`contrastive_energy_loss`). Real windows against corrupted ones:
       L = softplus(E_pos − E_neg) + λ_reg (E_pos² + E_neg²),
   E = E_total at ŷ per trigger. The squared-energy regulariser keeps learned scales from running
   away (Du & Mordatch, "Implicit Generation and Modeling with Energy-Based Models", NeurIPS 2019,
   arXiv:1903.08689, their energy-magnitude regulariser). Corruptions at TAAFT's view (AS-215):
   `entity_swap` (token v reads entity π(v)'s states while keeping v's kind and contacts) and
   `time_shuffle` (each entity's anchor state is a random *earlier* state of itself, so the view is
   out of time order with the contacts, but never reads the future).
4. **Malignity** (`malignity_loss`, build-spec §4b.4). Entity malignity m_v against the soft target
   `LabelBatch.entity_malicious_share` (the malicious share of the entity's updates trailing each
   trigger): binary cross-entropy with a soft target, on entries where the share is defined.

Unrolled descent (AS-16)
------------------------
Train with `TAAFT.forward(create_graph=True, descent_steps=sample_descent_steps(gen))`, S drawn
uniformly from {2, …, 8}: the loss on ŷ_S then reaches the lenses, α and the transformer through
the descent (Gladstone et al. 2025, arXiv:2507.02092). With create_graph=False ŷ is detached from
y₀ and only the readout heads (and the energies at ŷ) receive gradients.

Precision (D-54, AS-452): the contrastive term reads E_total (float64) and the malignity term reads
the float64 malignity readout, so both are float64 (targets are cast to the readout's dtype); the
latent likelihoods read float32 latent readouts and stay float32. `stage4_loss["total"]` is float64
by type promotion; backward casts the gradients to the float32 parameters.

Weights: `Stage4Weights` has no defaults (loss weights are a training choice; nothing is silently
defaulted).

Assumptions: AS-16, AS-215, AS-216. Decisions: D-22 (stages), D-23 (splits).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from nagahana.governance.assumptions import assume
from nagahana.models.batch import AnalysisOut, TriggerBatch, WindowBatch
from nagahana.models.taaft.readouts import gaussian_nll, soft_categorical_ce
from nagahana.models.taaft.structure import gather_rows, group_prev, segmented_cumsum


@dataclass(frozen=True)
class Stage4Weights:
    """Weights of the stage-4 terms (all required; > 0 to include a term, 0 to drop it)."""

    masked_entity: float
    future_latent: float
    contrastive: float
    malignity: float
    energy_reg: float


@dataclass
class Stage4Targets:
    """Targets of stage 4 (from frozen stage-3 outputs and labels; never model inputs).

    latent_mean [B, P, Dc], latent_logits [B, P, G, C]: CVG-AE posterior per position (LatentOut).
    hidden [B, V] bool or None: the entities hidden in the forward pass (masked-entity objective).
    malicious_share [B, V, M] or None: LabelBatch.entity_malicious_share (NaN = undefined).
    """

    latent_mean: torch.Tensor
    latent_logits: torch.Tensor
    hidden: torch.Tensor | None = None
    malicious_share: torch.Tensor | None = None


# ============================================================================ helpers
def sample_descent_steps(generator: torch.Generator | None = None, low: int = 2, high: int = 8) -> int:
    """S ~ Uniform{low, …, high} for unrolled-descent training (AS-16: {2, …, 8})."""
    assume("AS-16", by=__name__)
    return int(torch.randint(low, high + 1, (1,), generator=generator))


def gather_entity_latents(
    latent_mean: torch.Tensor, latent_logits: torch.Tensor, entity_latest: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Posterior at each entity's latest position per trigger: ([B,M,V,Dc], [B,M,V,G,C], valid [B,M,V])."""
    return gather_rows(latent_mean, entity_latest), gather_rows(latent_logits, entity_latest), entity_latest >= 0


def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    w = mask.to(x.dtype)
    return (x * w).sum() / w.sum().clamp_min(1.0)


def _latent_nll(mean: torch.Tensor, logvar: torch.Tensor, logits: torch.Tensor,
                t_mean: torch.Tensor, t_logits: torch.Tensor) -> torch.Tensor:
    # Gaussian NLL of the continuous part + soft cross-entropy of the categorical part (if any).
    nll = gaussian_nll(mean, logvar, t_mean)
    if logits.shape[-2] > 0 and logits.shape[-1] > 0:
        nll = nll + soft_categorical_ce(logits, t_logits)
    return nll


def total_energy(out: AnalysisOut) -> torch.Tensor:
    """E_total at ŷ per trigger [B, M] float64: the sum of the lens terms (D-42), accumulated in float64 (D-54)."""
    return torch.stack([e.double() for e in out.lens_energy.values()], dim=0).sum(0)


# ============================================================================ objectives
def masked_entity_loss(out: AnalysisOut, targets: Stage4Targets, entity_latest: torch.Tensor) -> torch.Tensor:
    """Belief about hidden entities against their CVG-AE posterior (objective 1)."""
    if targets.hidden is None:
        raise ValueError("masked_entity_loss needs the hidden-entity mask used in the forward pass")
    v = entity_latest.shape[-1]
    t_mean, t_logits, valid = gather_entity_latents(targets.latent_mean, targets.latent_logits, entity_latest)
    nll = _latent_nll(out.readouts["latent_mean"], out.readouts["latent_logvar"], out.readouts["latent_logits"],
                      t_mean, t_logits)                                                  # [B, M, V]
    mask = valid & targets.hidden[:, None, :] & out.token_mask[..., :v]
    return _masked_mean(nll, mask)


def future_latent_loss(out: AnalysisOut, targets: Stage4Targets, triggers: TriggerBatch) -> torch.Tensor:
    """Believed next latent at trigger m against the posterior at the entity's latest state at m + 1 (objective 2)."""
    v = triggers.entity_latest.shape[-1]
    if triggers.entity_latest.shape[1] < 2:
        return out.y.sum() * 0.0
    nxt = triggers.entity_latest[:, 1:]                                                  # [B, M−1, V]
    t_mean, t_logits, valid = gather_entity_latents(targets.latent_mean, targets.latent_logits, nxt)
    r = out.readouts
    nll = _latent_nll(r["next_latent_mean"][:, :-1], r["next_latent_logvar"][:, :-1], r["next_latent_logits"][:, :-1],
                      t_mean, t_logits)                                                  # [B, M−1, V]
    mask = valid & out.token_mask[:, :-1, :v] & triggers.mask[:, 1:, None]
    return _masked_mean(nll, mask)


def contrastive_energy_loss(e_pos: torch.Tensor, e_neg: torch.Tensor, mask: torch.Tensor, *, reg: float) -> torch.Tensor:
    """softplus(E_pos − E_neg) + reg·(E_pos² + E_neg²), averaged over valid triggers (objective 3)."""
    loss = F.softplus(e_pos - e_neg) + reg * (e_pos**2 + e_neg**2)
    return _masked_mean(loss, mask)


def malignity_loss(out: AnalysisOut, malicious_share: torch.Tensor) -> torch.Tensor:
    """Soft-target binary cross-entropy of entity malignity (objective 4, build-spec §4b.4).

    malicious_share [B, V, M] (NaN where the entity had no updates: excluded, never treated as 0).
    """
    v = malicious_share.shape[1]
    m = out.readouts["malignity"][..., :v].clamp(1e-6, 1.0 - 1e-6)                      # float64 readout (D-54)
    # Target in the readout's dtype (float64): the cross-entropy is evaluated at the reported precision.
    target = malicious_share.permute(0, 2, 1).to(m.dtype)                                # [B, M, V]
    ok = torch.isfinite(target) & out.token_mask[..., :v]
    t = torch.where(ok, target, torch.zeros_like(target)).clamp(0.0, 1.0)
    bce = -(t * torch.log(m) + (1.0 - t) * torch.log1p(-m))
    return _masked_mean(bce, ok)


def stage4_loss(
    out: AnalysisOut,
    window: WindowBatch,
    targets: Stage4Targets,
    weights: Stage4Weights,
    *,
    out_neg: AnalysisOut | None = None,
) -> dict[str, torch.Tensor]:
    """Combine the stage-4 terms. Returns {term: loss} plus "total" = Σ weight·loss.

    `out` is the forward pass on the real window (with `hidden_entities=targets.hidden` when the
    masked-entity term is on); `out_neg` the forward pass on a corrupted view (entity_swap or
    time_shuffle) for the contrastive term.
    """
    assume("AS-452", by=__name__)                # float64 terms on float64 outputs (D-54)
    losses: dict[str, torch.Tensor] = {}
    trig = window.triggers
    if weights.masked_entity > 0:
        losses["masked_entity"] = masked_entity_loss(out, targets, trig.entity_latest)
    if weights.future_latent > 0:
        losses["future_latent"] = future_latent_loss(out, targets, trig)
    if weights.malignity > 0:
        if targets.malicious_share is None:
            raise ValueError("malignity term needs LabelBatch.entity_malicious_share")
        losses["malignity"] = malignity_loss(out, targets.malicious_share)
    if weights.contrastive > 0:
        if out_neg is None:
            raise ValueError("contrastive term needs the forward pass on a corrupted view (out_neg)")
        losses["contrastive"] = contrastive_energy_loss(total_energy(out), total_energy(out_neg), trig.mask,
                                                        reg=weights.energy_reg)
    w = {"masked_entity": weights.masked_entity, "future_latent": weights.future_latent,
         "malignity": weights.malignity, "contrastive": weights.contrastive}
    total = out.y.sum() * 0.0
    for k, val in losses.items():
        total = total + w[k] * val
    losses["total"] = total
    return losses


# ============================================================================ corruptions and masks (AS-215, AS-216)
def sample_hidden_entities(window: WindowBatch, ratio: float, generator: torch.Generator | None = None) -> torch.Tensor:
    """Hide each real entity seen in the window with probability `ratio`: bool [B, V] (AS-216)."""
    seen = (window.triggers.entity_latest >= 0).any(dim=1) & window.entity_mask
    draw = torch.rand(seen.shape, generator=generator) < ratio
    return seen & draw.to(seen.device)


def _with_latest(window: WindowBatch, latest: torch.Tensor) -> WindowBatch:
    return dataclasses.replace(window, triggers=dataclasses.replace(window.triggers, entity_latest=latest))


def entity_swap(window: WindowBatch, generator: torch.Generator | None = None) -> WindowBatch:
    """Negative view: per window a random permutation π of real entities; token v reads π(v)'s states.

    Kinds, contacts and causal structure stay those of v, so states and structure disagree. The
    permutation is fixed across triggers (consistent in time), and every read is still ≤ τ.
    """
    assume("AS-16", by=__name__)
    lat = window.triggers.entity_latest.clone()                                           # [B, M, V]
    b, _, v = lat.shape
    for bi in range(b):
        real = torch.nonzero(window.entity_mask[bi]).flatten()
        perm = real[torch.randperm(real.numel(), generator=generator)]
        lat[bi][:, real] = window.triggers.entity_latest[bi][:, perm]
    return _with_latest(window, lat)


def time_shuffle(window: WindowBatch, generator: torch.Generator | None = None) -> WindowBatch:
    """Negative view: each (trigger, entity) anchor becomes a uniformly random earlier-or-equal state
    of the same entity, so the view is out of time order with the contacts (AS-215).

    Never reads the future: the new anchor is on the entity's chain back from its latest state ≤ τ.
    """
    pos = window.positions
    lat = window.triggers.entity_latest
    v = lat.shape[-1]
    groups = torch.where(pos.mask & (pos.entity >= 0), pos.entity, torch.full_like(pos.entity, v))
    prev = group_prev(groups)                                                             # [B, P]
    # Rank of every position within its entity (0 for the first state): prefix count − 1.
    rank = (segmented_cumsum(torch.ones_like(pos.time), groups) - 1.0).round().long()
    r_at = gather_rows(rank, lat)                                                         # [B, M, V]
    back = (torch.rand(lat.shape, generator=generator) * (r_at + 1).to(torch.float32)).long().clamp_max(r_at)
    out = lat.clone()
    b = lat.shape[0]
    flat_prev = prev
    while bool((back > 0).any()):
        step = (back > 0) & (out >= 0)
        nxt = flat_prev.gather(1, out.clamp_min(0).reshape(b, -1)).reshape(out.shape)
        out = torch.where(step, nxt, out)
        back = back - step.long()
        if not bool(step.any()):
            break
    return _with_latest(window, torch.where(lat >= 0, out, lat))
