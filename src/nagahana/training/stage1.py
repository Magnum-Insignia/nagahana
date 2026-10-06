"""Stage 1: Simulator pretraining of the FieldEncoder, CVG-AE, Decoder and TSTCT (self-supervised).

Purpose
-------
"Self-supervised pretraining of CVG-AE, Decoder and TSTCT: the intuition of network topology and its
dynamics" (architecture section 5, D-22 step 3). Real training segments and accepted Generator
variants of them (D-23, AS-592) are read in stream order with the TSTCT carry (D-51), the carry reset at
each segment start.

Objective (build-spec section 3; per batch)
-------------------------------------------
    L_1 = L_rec + lambda_edge L_edge
          + w_mem KL_bal(q_next || p_mem) + w_ref KL_bal(q_next || p_ref)                (AS-05, AS-159, AS-409)
          + beta_0 max(fb, KL(q_first || N(0, I) x Unif))                                 (AS-408)
          + lambda_phys Phi_phys(decoded) / n_rows + lambda_gate mean(g)                 (AS-15, AS-402, AS-10)

- L_rec: weighted NLL of the contributing cells of the update behind each position, on a masked copy
  of the fields (15 % of contributing cells -> MASK), masked cells weighted `mask_weight` (AS-04,
  AS-113, AS-407). Out-of-order augmentation inside the recorded uncertainty is applied by the stream
  loader (AS-320), so the structure is rebuilt consistently with the reordered times.
- L_edge: balanced BCE of observed hyperedges of the local subgraphs against member-swap negatives
  (AS-111), members represented by the latent of their latest state as of the subgraph's centre (AS-410).
- KL_bal: DreamerV3 balancing with free bits on the posterior at each position's next state of the same
  entity, against the memory-stream prior and the refined (thinking-stream) prior (AS-159); beta_dyn,
  beta_rep and the free bits come from `TrainingConfig` (AS-05).
- First states: an entity's first position in the window, when the entity has no carried state.
- lambda_phys is the one shared physics weight `TAAFTConfig.lambda_phys` (D-37, AS-15): TAAFT's energy
  and every component loss read the same value.
- R: drawn by the engine from the generator shared by all ranks (AS-07: 1 + Poisson(3) clipped to
  [1, 8]; AS-577), or fixed by an ablation; backpropagation through the last 2 thinking passes.

Randomness on the device (AS-588): field masking draws from the rank's CPU generator and moves the draw
to the fields' device, so the same seed gives the same masks on every accelerator.

Decisions: D-22, D-23, D-37, D-41, D-51. Assumptions: AS-04, AS-05, AS-07, AS-10, AS-15, AS-111, AS-113,
AS-159, AS-320, AS-402, AS-407 ... AS-410, AS-588, AS-592.
"""

from __future__ import annotations

from typing import Any

import torch

from nagahana.core.errors import InvariantViolation
from nagahana.governance.assumptions import assume
from nagahana.models.batch import EnvironmentOut, FieldBatch, LatentOut, WindowBatch
from nagahana.models.decoder.losses import edge_loss, gather_position_targets, physics_loss, reconstruction_loss
from nagahana.models.inputs.encoder import CONTRIBUTING_STATUS
from nagahana.models.latent_kl import kl_balanced, kl_standard
from nagahana.models.nagahana import NagaHana
from nagahana.models.taaft.structure import prev_positions
from nagahana.models.vocab import MASK_STATUS
from nagahana.physics.term import PhysicsTerm
from nagahana.training.carry import PreparedBatch, StreamBridge
from nagahana.training.config import Stage1Options
from nagahana.training.engine import Draws, ObjectiveOut, ProgramTrainer, StageProgram, detach_tree

#: Components of each role in stage 1 (pipeline/stages.py).
TRAINED: tuple[str, ...] = ("inputs", "cvgae", "decoder", "tstct")
FROZEN: tuple[str, ...] = ("longterm", "taaft", "forecaster", "advisor", "verifier")


def mask_fields(fields: FieldBatch, ratio: float, generator: torch.Generator | None) -> tuple[FieldBatch, torch.Tensor]:
    """(masked copy, mask [B, U, C] bool): each contributing cell -> MASK with probability `ratio` (AS-113).

    The uniform draw is made on the generator's device and moved to the fields' device (AS-588).
    """
    if not 0.0 <= ratio <= 1.0:
        raise ValueError("ratio must lie in [0, 1]")
    status = fields.status
    contrib = CONTRIBUTING_STATUS.to(status.device)[status.long()]
    dev = generator.device if generator is not None else status.device
    draw = torch.rand(status.shape, generator=generator, device=dev).to(status.device)
    mask = contrib & (draw < ratio)
    new_status = torch.where(mask, torch.full_like(status, MASK_STATUS), status)
    values = torch.where(mask, torch.full_like(fields.values, float("nan")), fields.values)
    masked = FieldBatch(values=values, status=new_status, column_kind=fields.column_kind,
                        column_slot=fields.column_slot, column_names=fields.column_names)
    return masked, mask


def position_lookup(window: WindowBatch) -> torch.Tensor:
    """long [B, U, 2]: the position of (update, role) for role in {initiator, responder} (-1 none)."""
    pos = window.positions
    b_n, u_n = window.update_mask.shape
    out = torch.full((b_n, u_n, 2), -1, dtype=torch.long, device=pos.entity.device)
    real = pos.mask & (pos.update >= 0) & (pos.role >= 0) & (pos.role <= 1)
    bi, pi = torch.nonzero(real, as_tuple=True)
    out[bi, pos.update[bi, pi], pos.role[bi, pi]] = pi
    return out


def hyperedge_rows(window: WindowBatch, plane: str, p_n: int, *, cap: int,
                   gen: torch.Generator | None) -> tuple[torch.Tensor, torch.Tensor]:
    """Member latent rows [E, m] (flat b * P + p, -1 padding) and the owner position [E] of the observed
    hyperedges of one plane's local subgraphs (AS-410).

    A member node is represented by the latent of its latest update as of the subgraph's centre time
    (`GraphBatch.node_update`, `node_role`); service nodes and nodes without an update have no latent
    and are left out. Hyperedges with fewer than 2 represented members are dropped; at most `cap` are
    kept, chosen uniformly with the generator (draw on its device, moved to the batch's).
    """
    g = window.graph
    inc = g.incidence[plane]
    dev = inc.device
    empty = (torch.zeros(0, 2, dtype=torch.long, device=dev), torch.zeros(0, dtype=torch.long, device=dev))
    if inc.numel() == 0:
        return empty
    u_n = window.update_mask.shape[1]
    lookup = position_lookup(window)                                         # [B, U, 2]
    node, edge = inc[0], inc[1]
    nu, nr = g.node_update[node], g.node_role[node]
    ok = (nu >= 0) & (nr >= 0) & (nr <= 1)
    b = torch.where(ok, nu // u_n, torch.zeros_like(nu))
    u = torch.where(ok, nu % u_n, torch.zeros_like(nu))
    p = torch.where(ok, lookup[b, u, nr.clamp(0, 1)], torch.full_like(nu, -1))
    ok = ok & (p >= 0)
    node_pos = torch.where(ok, b * p_n + p, torch.full_like(p, -1))         # flat latent row of each member
    owner_all = g.node_owner[node]                                           # the subgraph's centre position
    node, edge, node_pos, owner_all = node[ok], edge[ok], node_pos[ok], owner_all[ok]
    if edge.numel() == 0:
        return empty
    # Group members by hyperedge: rank inside the hyperedge = index - first index of the hyperedge.
    order = torch.argsort(edge, stable=True)
    edge, node_pos, owner_all = edge[order], node_pos[order], owner_all[order]
    uniq, counts = torch.unique_consecutive(edge, return_counts=True)
    starts = torch.cumsum(counts, 0) - counts
    rank = torch.arange(edge.numel(), device=dev) - torch.repeat_interleave(starts, counts)
    e_idx = torch.repeat_interleave(torch.arange(uniq.numel(), device=dev), counts)
    m = int(counts.max())
    rows = torch.full((uniq.numel(), m), -1, dtype=torch.long, device=dev)
    rows[e_idx, rank] = node_pos
    owner = owner_all[starts]                                                # every member shares the subgraph
    keep = (rows >= 0).sum(1) >= 2
    rows, owner = rows[keep], owner[keep]
    if rows.shape[0] > cap:
        gdev = gen.device if gen is not None else torch.device("cpu")
        pick = torch.randperm(rows.shape[0], generator=gen, device=gdev)[:cap].to(dev)
        rows, owner = rows[pick], owner[pick]
    return rows, owner


def hyperedge_members(window: WindowBatch, plane: str, z_flat: torch.Tensor, p_n: int, *, cap: int,
                      gen: torch.Generator | None) -> tuple[torch.Tensor, torch.Tensor]:
    """Member latents [E, m, dz] and mask [E, m] of the observed hyperedges of one plane (`hyperedge_rows`)."""
    rows, _owner = hyperedge_rows(window, plane, p_n, cap=cap, gen=gen)
    if rows.shape[0] == 0:
        return z_flat.new_zeros(0, 2, z_flat.shape[-1]), torch.zeros(0, 2, dtype=torch.bool, device=z_flat.device)
    return z_flat[rows.clamp_min(0)], rows >= 0


def carried_entities(prep: PreparedBatch) -> torch.Tensor:
    """bool [B, V']: entities with at least one carried slot (their first window state has a predecessor)."""
    w = prep.window
    out = torch.zeros_like(w.entity_mask)
    c = prep.carry
    if c is None or c.entity is None:
        return out
    ok = c.mask & (c.entity >= 0)
    bi, ci = torch.nonzero(ok, as_tuple=True)
    out[bi, c.entity[bi, ci]] = True
    return out


def stage1_loss(model: NagaHana, prep: PreparedBatch, *, options: Stage1Options, physics: PhysicsTerm | None,
                gen: torch.Generator | None, passes: int) -> tuple[torch.Tensor, dict[str, torch.Tensor], LatentOut, EnvironmentOut]:
    """L_1 for one prepared batch (module docstring). Returns (total, parts, LatentOut, EnvironmentOut)."""
    assume("AS-04", by=__name__)
    assume("AS-05", by=__name__)
    cfg, tcfg = model.cfg, model.cfg.training
    window = prep.window
    pos = window.positions
    b_n, p_n = pos.entity.shape
    masked, _cells = mask_fields(window.fields, tcfg.mask_ratio, gen)          # AS-113
    lat, env = model.perceive(window, sample=True, passes=passes, grad_passes=cfg.tstct.grad_passes,
                              carry=prep.carry, generator=gen, fields=masked)
    parts: dict[str, torch.Tensor] = {}
    # Reconstruction of the update behind each position (AS-04).
    tg = gather_position_targets(window.fields, masked.status, pos, window.update_planes)
    rec = reconstruction_loss(model.decoder, lat.z.float(), tg["role"], tg["planes"], tg["values"], tg["status"],
                              tg["true_status"], mask_weight=options.mask_weight, row_mask=tg["row_mask"])
    parts["rec"], parts["rec_masked"] = rec["total"], rec["masked"]
    # Candidate hyperedges per plane (AS-111, AS-410).
    z_flat = lat.z.float().reshape(b_n * p_n, -1)
    edge_terms = []
    for pi, plane in enumerate(cfg.graph.planes):
        mz, mm = hyperedge_members(window, plane, z_flat, p_n, cap=options.edge_cap, gen=gen)
        if mz.shape[0]:
            edge_terms.append(edge_loss(model.decoder, pi, mz, mm, generator=gen)["total"])
    parts["edge"] = torch.stack(edge_terms).mean() if edge_terms else z_flat.sum() * 0.0
    # Transition priors: memory stream and thinking stream (AS-05, AS-159, AS-409).
    nxt = pos.next_index
    has = pos.mask & (nxt >= 0)

    def at_next(x: torch.Tensor) -> torch.Tensor:
        idx = nxt.clamp_min(0).view(b_n, p_n, *([1] * (x.dim() - 2))).expand_as(x)
        return torch.gather(x, 1, idx)

    q_mean, q_logvar, q_logits = at_next(lat.mean), at_next(lat.logvar), at_next(lat.logits)
    w = has.float()

    def kl_term(prior_mean: torch.Tensor, prior_logvar: torch.Tensor, prior_logits: torch.Tensor) -> dict[str, torch.Tensor]:
        k = kl_balanced(q_mean.float(), q_logvar.float(), q_logits.float(), prior_mean.float(), prior_logvar.float(),
                        prior_logits.float(), free_bits=tcfg.free_bits, beta_dyn=tcfg.beta_dyn, beta_rep=tcfg.beta_rep,
                        unimix=cfg.cvgae.unimix)
        return {key: (val * w).sum() / w.sum().clamp_min(1.0) for key, val in k.items()}

    km = kl_term(env.prior.mean, env.prior.logvar, env.prior.logits)
    ref = model.tstct.refined_prior(env, window)
    kr = kl_term(ref.mean, ref.logvar, ref.logits)
    wm, wr = options.prior_weights
    parts["kl_memory"], parts["kl_refined"] = km["total"], kr["total"]
    parts["kl_memory_raw"], parts["kl_refined_raw"] = km["kl"].detach(), kr["kl"].detach()
    # First states of entities without a carried predecessor (AS-408).
    prev = prev_positions(pos.next_index)
    carried = carried_entities(prep)
    is_carried = torch.gather(carried, 1, pos.entity.clamp_min(0))
    first = pos.mask & (prev < 0) & ~is_carried
    k0 = kl_standard(lat.mean.float(), lat.logvar.float(), lat.logits.float(), unimix=cfg.cvgae.unimix)
    k0 = torch.maximum(k0, torch.tensor(float(tcfg.free_bits), device=k0.device, dtype=k0.dtype))
    f = first.float()
    parts["kl_first"] = (k0 * f).sum() / f.sum().clamp_min(1.0)
    # Physics on the decoded fields of every real position (D-37, AS-402); lambda_phys is shared (AS-15).
    if physics is not None:
        rows = tg["row_mask"]
        dec = model.decoder.decode_fields(lat.z.float()[rows], tg["role"][rows], tg["planes"][rows])
        phi, _ = physics_loss(model.decoder, dec, physics)
        parts["physics"] = phi / max(1, int(rows.sum()))
    else:
        parts["physics"] = lat.z.sum() * 0.0
    # Sparse causal gates (AS-10).
    parts["gate_l1"] = model.tstct.gate_l1(env)
    total = (parts["rec"] + tcfg.lambda_edge * parts["edge"] + wm * parts["kl_memory"] + wr * parts["kl_refined"]
             + options.beta_first * parts["kl_first"] + cfg.taaft.lambda_phys * parts["physics"]
             + tcfg.lambda_gate * parts["gate_l1"])
    parts["passes"] = torch.tensor(float(passes))
    if not torch.isfinite(total):
        raise InvariantViolation(f"stage-1 loss is not finite: { {k: float(v) for k, v in parts.items()} }")
    return total, parts, lat, env


def scalar_parts(parts: dict[str, torch.Tensor]) -> dict[str, float]:
    """Loss parts as Python floats (scalars only)."""
    return {k: float(v.detach()) for k, v in parts.items() if isinstance(v, torch.Tensor) and v.numel() == 1}


class Stage1Program(StageProgram):
    """Stage 1 for the engine (training/engine.py): the perceptors train on every batch."""

    stage = 1
    name = "simulator-pretraining"
    needs_trigger = False

    def __init__(self, model: NagaHana, *, options: Stage1Options, physics: PhysicsTerm | None,
                 fixed_passes: int | None = None) -> None:
        super().__init__(model)
        self.options = options
        self.physics = physics
        self.fixed_passes = fixed_passes

    def trained(self) -> list[str]:
        return list(TRAINED)

    def frozen(self) -> list[str]:
        return list(FROZEN)

    def passes(self, draws: Draws, train: bool) -> int:
        """R: fixed by an ablation, else sampled from the shared generator in training, the run-time default otherwise."""
        if self.fixed_passes is not None:
            return self.fixed_passes
        return self.model.tstct.sample_passes(draws.shared) if train else self.model.cfg.tstct.default_passes

    def compute(self, model: NagaHana, prep: PreparedBatch, pre: Any, draws: Draws, step: int, train: bool) -> ObjectiveOut:
        total, parts, _lat, env = stage1_loss(model, prep, options=self.options, physics=self.physics, gen=draws.local,
                                              passes=self.passes(draws, train))
        out = scalar_parts(parts)
        out["total"] = float(total.detach())
        return ObjectiveOut(loss=total, parts=out, aux={"env": detach_tree(env)})

    def post(self, bridge: StreamBridge, prep: PreparedBatch, pre: Any, out: ObjectiveOut | None) -> None:
        if out is None:
            raise InvariantViolation("stage 1 has objective work on every batch")
        bridge.commit(prep, out.aux["env"])


class Stage1Trainer(ProgramTrainer):
    """Single-process stage-1 trainer on prepared batches (training/engine.ProgramTrainer)."""

    def __init__(self, model: NagaHana, *, options: Stage1Options, physics: PhysicsTerm | None, optim: Any, total_steps: int,
                 seed: int, fixed_passes: int | None = None) -> None:
        super().__init__(Stage1Program(model, options=options, physics=physics, fixed_passes=fixed_passes), optim=optim,
                         total_steps=total_steps, seed=seed)


__all__ = ["FROZEN", "Stage1Program", "Stage1Trainer", "TRAINED", "carried_entities", "hyperedge_members", "hyperedge_rows",
           "mask_fields", "position_lookup", "scalar_parts", "stage1_loss"]
