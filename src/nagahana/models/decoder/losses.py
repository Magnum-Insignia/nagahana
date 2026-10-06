"""Stage-3 loss pieces owned by Perception: reconstruction, candidate hyperedges, physics, masking.

Purpose
-------
The parts of L₃ (build-spec §3) that read the Decoder, plus the two input perturbations of stage 3:

    L₃ ⊃ L_rec + λ_edge L_edge + λ_phys Φ_phys(decoded)

- `reconstruction_loss`: weighted mean NLL over contributing cells (AS-04), masked cells up-weighted.
- `edge_loss`: observed hyperedges against member-swap negatives (AS-04, AS-111).
- `physics_loss`: Φ_phys on the decoded fields (D-37; the Decoder has no physics of its own).
- `mask_fields`: 15 % of contributing cells → MASK status (build-spec §3; AS-113).
- `permute_within_uncertainty`: reorder updates only inside their recorded ordering uncertainty
  (build-spec §4b.5; AS-113).
- `gather_position_targets`: aligns each position with the fields of the update behind it.

Owner sources, decisions and assumptions
----------------------------------------
AS-04 (reconstruction target), AS-30 (status weights), D-37 (one physics term for all), new AS-111
(negatives), AS-113 (masking and reordering details).

Maths
-----
    L_rec  = Σ_{n,c} w_{n,c} ℓ_{n,c} / Σ_{n,c} w_{n,c},   w = status weight × (mask_weight if masked)
    L_edge = ½ mean_{e ∈ ℰ⁺} softplus(−s_e) + ½ mean_{e ∈ ℰ⁻} softplus(s_e)      (balanced BCE)
A negative e⁻ copies an observed hyperedge and replaces one uniformly chosen real member by the
latent of a member drawn uniformly from all members of the batch's observed hyperedges ("members
swapped", build-spec §2.4). A swap can by chance produce a real hyperedge (a false negative); at
window scale this is rare and is accepted, not filtered (AS-111).

Masking: each contributing cell is masked independently with probability `ratio`; masked cells get
status MASK and value NaN in the masked copy (the encoder can never read them), the original batch
stays the reconstruction target.

Reordering: t'_i = t_i + δ_i with δ_i ~ U[−r_i, r_i] (float64), r_i the update's
`reorder_uncertainty`; the permutation is the stable argsort of t'. An update with r_i = 0 keeps its
time exactly, so two updates are swapped only if their uncertainty intervals allow it.

Invariants (tests/test_perception_decoder.py)
---------------------------------------------
- Masking touches only contributing cells; the masked batch carries NaN on masked cells.
- `permute_within_uncertainty` never moves a time outside [t − r, t + r]; with r = 0 it is the identity.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from nagahana.core.errors import InvariantViolation
from nagahana.models.batch import FieldBatch, PositionBatch
from nagahana.models.decoder.model import DecodedFields, Decoder
from nagahana.models.inputs.encoder import CONTRIBUTING_STATUS
from nagahana.models.vocab import MASK_STATUS
from nagahana.physics.term import PhysicsTerm


# ================================================================================ perturbations
def mask_fields(fields: FieldBatch, ratio: float, generator: torch.Generator | None = None) -> tuple[FieldBatch, torch.Tensor]:
    """(masked copy, mask [B, U, C] bool): each contributing cell → MASK with probability `ratio`."""
    if not 0.0 <= ratio <= 1.0:
        raise ValueError("ratio must lie in [0, 1]")
    contrib = CONTRIBUTING_STATUS[fields.status.long()]
    draw = torch.rand(fields.status.shape, generator=generator)
    mask = contrib & (draw < ratio)
    status = torch.where(mask, torch.full_like(fields.status, MASK_STATUS), fields.status)
    values = torch.where(mask, torch.full_like(fields.values, float("nan")), fields.values)
    masked = FieldBatch(values=values, status=status, column_kind=fields.column_kind,
                        column_slot=fields.column_slot, column_names=fields.column_names)
    return masked, mask


def permute_within_uncertainty(
    update_time: torch.Tensor,
    reorder_uncertainty: torch.Tensor,
    generator: torch.Generator | None = None,
    *,
    update_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """(perturbed times t' [B, U] float64, permutation [B, U] long = stable argsort of t').

    Padded updates (update_mask False) get t' = +inf so they stay last. Apply the permutation to every
    per-update tensor (and rebuild the window structure) to obtain the reordered window.
    """
    t = update_time.to(torch.float64)
    r = reorder_uncertainty.to(torch.float64)
    if r.shape != t.shape:
        raise InvariantViolation("update_time and reorder_uncertainty must share a shape")
    if bool((r < 0).any()) or not bool(torch.isfinite(r).all()):
        raise InvariantViolation("reorder_uncertainty must be finite and ≥ 0")
    u = torch.rand(t.shape, generator=generator, dtype=torch.float64)
    t_new = t + (2.0 * u - 1.0) * r                                          # δ ~ U[−r, r]
    if update_mask is not None:
        t_new = torch.where(update_mask, t_new, torch.full_like(t_new, float("inf")))
    perm = torch.sort(t_new, dim=-1, stable=True).indices
    return t_new, perm


# ================================================================================ alignment
def gather_position_targets(
    fields: FieldBatch,
    seen_status: torch.Tensor,
    positions: PositionBatch,
    update_planes: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Fields of the update behind each position: values, status (as seen), true_status [B, P, C],
    planes [B, P, n_planes], role [B, P], row_mask [B, P] (AS-04: a position reconstructs its update)."""
    upd = positions.update.clamp_min(0)                                      # [B, P]
    b, p = upd.shape
    c = fields.values.shape[-1]
    idx = upd.unsqueeze(-1).expand(b, p, c)
    out = {
        "values": torch.gather(fields.values, 1, idx),
        "status": torch.gather(seen_status, 1, idx),
        "true_status": torch.gather(fields.status, 1, idx),
        "planes": torch.gather(update_planes, 1, upd.unsqueeze(-1).expand(b, p, update_planes.shape[-1])),
        "role": positions.role,
        "row_mask": positions.mask & (positions.update >= 0),
    }
    return out


# ================================================================================ losses
def reconstruction_loss(
    decoder: Decoder,
    z: torch.Tensor,
    role: torch.Tensor,
    planes: torch.Tensor,
    values: torch.Tensor,
    status: torch.Tensor,
    true_status: torch.Tensor,
    *,
    mask_weight: float = 1.0,
    row_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Weighted mean NLL over contributing cells (see the module docstring).

    Returns 'total' (scalar), 'masked' (mean over masked cells, 0 if none), 'weight' (Σ w).
    """
    nll = decoder.field_nll(z, role, planes, values, status, true_status=true_status, mask_weight=mask_weight)
    w = decoder.cell_weight(status, true_status, mask_weight=mask_weight)
    if row_mask is not None:
        rm = row_mask.unsqueeze(-1)
        nll = torch.where(rm, nll, torch.zeros((), dtype=nll.dtype))
        w = torch.where(rm, w, torch.zeros((), dtype=w.dtype))
    total = nll.sum() / w.sum().clamp_min(1e-8)
    masked_cells = (status == MASK_STATUS) & (w > 0)
    masked = (nll * masked_cells).sum() / (w * masked_cells).sum().clamp_min(1e-8)
    return {"total": total, "masked": masked, "weight": w.sum()}


def edge_loss(
    decoder: Decoder,
    plane: int,
    member_z: torch.Tensor,
    member_mask: torch.Tensor,
    *,
    generator: torch.Generator | None = None,
    negatives: int | None = None,
) -> dict[str, torch.Tensor]:
    """Balanced BCE of observed hyperedges [E, m, dz] (mask [E, m]) against member-swap negatives.

    Returns 'total', 'pos' and 'neg' (mean BCE of each side) and 'auc_proxy' (fraction of
    positive/negative pairs ranked correctly, for monitoring).
    """
    k = decoder.cfg.edge_negatives if negatives is None else negatives
    mask = member_mask.to(torch.bool)
    e, m, _ = member_z.shape
    if e == 0:
        zero = member_z.sum() * 0.0
        return {"total": zero, "pos": zero, "neg": zero, "auc_proxy": zero}
    if bool((mask.sum(1) < 2).any()):
        raise InvariantViolation("every observed hyperedge needs ≥ 2 real members")
    pos = decoder.edge_logits(plane, member_z, mask)                          # [E]
    pool = member_z[mask]                                                    # [M, dz] all real members
    neg_z = member_z.repeat(k, 1, 1)                                         # [k·E, m, dz]
    neg_mask = mask.repeat(k, 1)
    # One real member slot per negative, uniformly among the real members.
    slot_scores = torch.rand(neg_mask.shape, generator=generator).masked_fill(~neg_mask, -1.0)
    slot = slot_scores.argmax(dim=1)                                         # [k·E]
    donor = torch.randint(0, pool.shape[0], (k * e,), generator=generator)
    rows = torch.arange(k * e)
    neg_z = neg_z.index_put((rows, slot), pool[donor])
    neg = decoder.edge_logits(plane, neg_z, neg_mask)                        # [k·E]
    l_pos = F.softplus(-pos).mean()
    l_neg = F.softplus(neg).mean()
    with torch.no_grad():
        auc = (pos.unsqueeze(1) > neg.unsqueeze(0)).float().mean()
    return {"total": 0.5 * (l_pos + l_neg), "pos": l_pos, "neg": l_neg, "auc_proxy": auc}


def physics_loss(decoder: Decoder, decoded: DecodedFields, term: PhysicsTerm) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Φ_phys on the decoded fields (D-37): `term(*decoder.physics_inputs(decoded))`."""
    values, contributing = decoder.physics_inputs(decoded)
    return term(values, contributing)
