"""Forecaster training (build-spec §3, stage 5): survival NLL, teacher-forced dynamics, behaviour cloning,
TD(λ) values and the process-reward model.

Owner sources: [Q-24] (a process reward for every imagined step), [Q-28] (model-based RL), [A-06].
Decisions: D-22/D-23 (stage 5 full training). Assumptions: AS-17 (adversary reward), AS-18
(infiltration state), AS-22 (STAGED coupling, via the model), AS-25 (dataset annotations are human
truth), AS-253 (λ = 0.95), AS-260 (a malicious update's responder is the action target; a labelled
window with no malicious update is the "no target" action).

Maths
-----
1. **Discrete-time survival NLL with censoring** (e.g. Tutz & Schmid, *Modeling Discrete
   Time-to-Event Data*, Springer 2016, ch. 3). Hazard h_j = P(T = j | T ≥ j). For an event at step e:

       −log L = −Σ_{j<e} log(1 − h_j) − log h_e

   For a trigger censored after c event-free observed steps (c ∈ 0…K):

       −log L = −Σ_{j≤c} log(1 − h_j)

   This is the likelihood of the same P_inf the Forecaster reports (P_inf(k) = 1 − Π_{j≤k}(1 − h_j)),
   so training and the reported curve are one model.

2. **Teacher forcing.** The route transformer is run densely on the *real* action sequence (labels
   where known; a learned "unknown" input otherwise). Because step inputs are actions only (AS-250),
   this is the same computation imagination performs, with labelled instead of sampled actions.

3. **Latent consistency** (self-predictive representations; TD-MPC2, Hansen et al. 2024,
   arXiv:2310.16828; BYOL's normalised form, Grill et al., NeurIPS 2020, arXiv:2006.07733):
       L_cons = Σ_k ‖ σ̂_k/‖σ̂_k‖ − sg(σ_{m+k})/‖sg(σ_{m+k})‖ ‖²
   σ_{m+k} is the Forecaster's own summary of the real analysis k triggers later (stop-gradient).
   The imagined hypothesis ŷ_k of the target is regressed on sg(y_{m+k, target}) the same way (MSE).

4. **Behaviour cloning**: −log π(tech_k | s_{k−1}) where the technique label is known (−1 masked);
   −log π(target_k | s_{k−1}, tech_k) where the target is known and among the context positions.

5. **Reward model** (AS-17): r̂_k regresses r^task_k = (progress(stage_k) − progress(stage_{k−1})) +
   β·𝟙[infiltration at k], where both stage labels are known. The exposure part −κ·exposure_k is not
   learned: it is computed from TAAFT's marginal energy (model docstring, AS-252).

6. **TD(λ) value targets** (Sutton & Barto, *Reinforcement Learning: An Introduction*, 2nd ed. 2018,
   §12.1, λ-return; DreamerV3 uses λ = 0.95):
       G_K = V(s_K),   G_k = r_{k+1} + γ[(1 − λ) V(s_{k+1}) + λ G_{k+1}]
   with r_{k+1} = r^task (label if known, else sg r̂) − κ·exposure, all bootstraps stop-gradient.
   Loss ½(V(s_k) − sg G_k)² for k = 0 … K−1.

Precision (D-54; AS-452)
------------------------
The parts that consume a D-54 float64 output are computed in float64: the survival NLL (on the float64
hazards), behaviour cloning (on the float64 log π of technique and target) and the per-step stage
cross-entropy (softmax of the float64-cast stage logits). They are the likelihoods of exactly the
probabilities the Forecaster reports, so training and reporting share one link function at one
precision. The other parts (consistency, hypothesis, latent, reward, value) regress float32 learned
estimates and stay float32. `total` is therefore float64 (type promotion of the weighted sum); its
backward casts every gradient back to the float32 parameters (exact autograd of `.double()`).

Invariants (tested): hazard NLL equals the hand computation; λ-returns equal the recursion; every
trainable Forecaster parameter receives a gradient from `teacher_forced`; the hazard, stage and BC
parts and the total are float64 and backpropagate.

Extension points: add parts to `ForecasterLossWeights` / the `parts` dict; each part is reported
separately (no hidden weighting).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from nagahana.governance.assumptions import assume
from nagahana.models.batch import AnalysisOut, LabelBatch, WindowBatch
from nagahana.models.forecaster.model import Forecaster, MarginalEnergy
from nagahana.models.vocab import STAGE_PROGRESS
from nagahana.objectives.rewards import adversary_reward

#: Target code for "no target" (a labelled window without malicious activity, AS-260); −1 = unknown.
NO_TARGET = -2


# =========================================================================== survival
def hazard_nll(hazard: torch.Tensor, event_step: torch.Tensor, censored: torch.Tensor, *, eps: float = 1e-6) -> torch.Tensor:
    """Discrete-time survival negative log-likelihood per item (module docstring, part 1).

    hazard: [..., K] in (0, 1); event_step: long [...] — the 1-based event step (1…K) for events, the
    number of event-free observed steps (0…K) for censored items; censored: bool [...].
    Returns float64 [...] (D-54, AS-452: the likelihood of the float64 hazards the Forecaster reports;
    a float32 input is promoted exactly). `eps` only guards log(0) for saturated hazards; it is kept
    at 1e-6 so the gradient region of training is unchanged by the precision change.
    """
    k = hazard.shape[-1]
    h = hazard.double().clamp(eps, 1.0 - eps)
    steps = torch.arange(1, k + 1, device=hazard.device)                     # [K]
    e = event_step.unsqueeze(-1)                                             # [..., 1]
    log_surv = torch.log1p(-h)                                               # log(1 − h_j)
    # Survived steps: j < e for events, j ≤ c for censored items.
    survived = torch.where(censored.unsqueeze(-1), steps <= e, steps < e)    # [..., K]
    nll = -(log_surv * survived).sum(-1)
    # Event term −log h_e (only for events).
    hit = (~censored.unsqueeze(-1)) & (steps == e)
    nll = nll - (torch.log(h) * hit).sum(-1)
    return nll


def survival_targets(trigger_time: torch.Tensor, trigger_mask: torch.Tensor, infiltrated_at: torch.Tensor,
                     entity_internal: torch.Tensor, observed_until: torch.Tensor, *, window_seconds: float,
                     horizon_k: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(event_step, censored, usable) per trigger from the label timeline (AS-18).

    trigger_time float64 [B, M]; trigger_mask bool [B, M]; infiltrated_at float64 [B, V] (+inf never);
    entity_internal bool [B, V]; observed_until float64 [B] (end of the window's label horizon).
    - A trigger at which an internal entity is already infiltrated is not usable (P_inf is 1 there).
    - The next infiltration at time t* gives e = ⌈(t* − τ)/w⌉ (t* ∈ (τ+(e−1)w, τ+ew]); it is an event
      if e ≤ K and t* ≤ observed_until; otherwise the trigger is censored after min(K, ⌊(end − τ)/w⌋)
      event-free steps.
    """
    assume("AS-18", by=__name__)
    tau = trigger_time.to(torch.float64)                                     # [B, M]
    infil = torch.where(entity_internal, infiltrated_at.to(torch.float64), torch.full_like(infiltrated_at, float("inf"), dtype=torch.float64))
    already = (infil.unsqueeze(1) <= tau.unsqueeze(-1)).any(-1)              # [B, M]
    later = torch.where(infil.unsqueeze(1) > tau.unsqueeze(-1), infil.unsqueeze(1), torch.full_like(infil.unsqueeze(1), float("inf")))
    t_next = later.min(-1).values                                            # [B, M]
    end = observed_until.to(torch.float64).unsqueeze(-1)                     # [B, 1]
    gap = (t_next - tau) / window_seconds
    e = torch.ceil(gap.clamp(max=horizon_k + 1.0)).long().clamp(min=1)       # finite even when t_next = inf
    is_event = torch.isfinite(t_next) & (t_next <= end) & (e <= horizon_k)
    obs_steps = torch.floor(((end - tau) / window_seconds).clamp(min=0.0, max=float(horizon_k))).long()
    event_step = torch.where(is_event, e, obs_steps)
    usable = trigger_mask & ~already
    return event_step, ~is_event, usable


# =========================================================================== step labels
def step_labels(window: WindowBatch, labels: LabelBatch, *, window_seconds: float, horizon_k: int
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(technique [B, M, K], target_entity [B, M, K], stage [B, M, K+1]) from the dataset annotations (AS-25, AS-260).

    Step k ≥ 1 covers updates with time in (τ + (k−1)w, τ + kw]; index 0 of `stage` covers (τ − w, τ].
    - stage: the furthest kill-chain stage among labelled updates of the step (−1 if none labelled);
    - target_entity: the responder of the first malicious update of the step; NO_TARGET (−2) if the step
      has labelled updates but none malicious; −1 if unknown;
    - technique: the technique of the first malicious update with a known technique (−1 otherwise).
    """
    assume("AS-25", by=__name__)
    tau = window.triggers.time.to(torch.float64)                             # [B, M]
    t_u = window.update_time.to(torch.float64)                               # [B, U]
    ks = torch.arange(0, horizon_k + 1, dtype=torch.float64, device=tau.device)
    lo = tau.unsqueeze(-1) + (ks - 1.0) * window_seconds                     # [B, M, K+1]
    hi = lo + window_seconds
    t = t_u[:, None, None, :]                                                # [B, 1, 1, U]
    in_win = (t > lo.unsqueeze(-1)) & (t <= hi.unsqueeze(-1)) & window.update_mask[:, None, None, :]   # [B, M, K+1, U]
    st = labels.update_stage[:, None, None, :].expand_as(in_win)
    # STAGE_PROGRESS is increasing in the stage code, so the furthest stage is the largest code.
    stage = torch.where(in_win & (st >= 0), st, torch.full_like(st, -1)).max(-1).values   # [B, M, K+1]
    mal = (labels.update_malicious.nan_to_num(0.0) > 0.5)[:, None, None, :] & in_win
    labelled = in_win & ~torch.isnan(labels.update_malicious)[:, None, None, :]
    u = in_win.shape[-1]
    idx = torch.arange(u, device=tau.device)

    def first(mask: torch.Tensor) -> torch.Tensor:
        # index of the first True along U (U where none)
        return torch.where(mask, idx, torch.full_like(idx, u)).min(-1).values

    f_mal = first(mal)[..., 1:]                                              # [B, M, K]
    any_mal = f_mal < u
    resp = window.update_entities[..., 1]                                    # [B, U]
    b_idx = torch.arange(resp.shape[0], device=tau.device)[:, None, None].expand_as(f_mal)
    target = torch.where(any_mal, resp[b_idx, f_mal.clamp(max=u - 1)], torch.full_like(f_mal, -1))
    no_tgt = ~any_mal & labelled[..., 1:, :].any(-1)
    target = torch.where(no_tgt, torch.full_like(target, NO_TARGET), target)
    tech_lab = labels.update_technique[:, None, None, :].expand_as(in_win)
    f_tech = first(mal & (tech_lab >= 0))[..., 1:]
    technique = torch.where(f_tech < u, labels.update_technique[b_idx, f_tech.clamp(max=u - 1)], torch.full_like(f_tech, -1))
    return technique, target, stage


def latent_targets(window: WindowBatch, z: torch.Tensor, target_entity: torch.Tensor, *, window_seconds: float
                   ) -> tuple[torch.Tensor, torch.Tensor]:
    """Back-projection targets: the posterior latent of the target entity's first state in each step.

    z: [B, P, dz] (LatentOut.z, posterior); target_entity: long [B, M, K]. → (latent [B, M, K, dz], mask [B, M, K]).
    """
    tau = window.triggers.time.to(torch.float64)
    k = target_entity.shape[-1]
    ks = torch.arange(1, k + 1, dtype=torch.float64, device=tau.device)
    lo = tau.unsqueeze(-1) + (ks - 1.0) * window_seconds                     # [B, M, K]
    hi = lo + window_seconds
    pt = window.positions.time.to(torch.float64)[:, None, None, :]           # [B, 1, 1, P]
    pe = window.positions.entity[:, None, None, :]
    hit = (pt > lo.unsqueeze(-1)) & (pt <= hi.unsqueeze(-1)) & (pe == target_entity.unsqueeze(-1)) \
        & (target_entity.unsqueeze(-1) >= 0) & window.positions.mask[:, None, None, :]
    p = hit.shape[-1]
    pos = torch.where(hit, torch.arange(p, device=tau.device), torch.full_like(pe, p)).min(-1).values   # [B, M, K]
    mask = pos < p
    b_idx = torch.arange(z.shape[0], device=z.device)[:, None, None].expand_as(pos)
    lat = z[b_idx, pos.clamp(max=p - 1)] * mask.unsqueeze(-1)
    return lat, mask


# =========================================================================== TD(λ)
def lambda_returns(rewards: torch.Tensor, values: torch.Tensor, *, gamma: float, lam: float) -> torch.Tensor:
    """λ-returns G_0 … G_{K−1}. rewards [..., K] (r_{k+1} at index k), values [..., K+1] (V(s_0) … V(s_K))."""
    k = rewards.shape[-1]
    g = values[..., k]
    out = []
    for i in range(k - 1, -1, -1):
        g = rewards[..., i] + gamma * ((1.0 - lam) * values[..., i + 1] + lam * g)
        out.append(g)
    return torch.stack(out[::-1], dim=-1)


# =========================================================================== teacher-forced loss
@dataclass
class ForecasterTargets:
    """Stage-5 targets for one batch of triggers (all long codes use −1 = unknown).

    technique, target_entity: [B, M, K] (target NO_TARGET = −2 for "no target"); stage: [B, M, K+1];
    event_step, censored, usable: [B, M] (from `survival_targets`); latent, latent_mask: optional
    [B, M, K, dz], [B, M, K] (from `latent_targets`).
    """

    technique: torch.Tensor
    target_entity: torch.Tensor
    stage: torch.Tensor
    event_step: torch.Tensor
    censored: torch.Tensor
    usable: torch.Tensor
    latent: torch.Tensor | None = None
    latent_mask: torch.Tensor | None = None


@dataclass(frozen=True)
class ForecasterLossWeights:
    """Weights of the stage-5 Forecaster parts. Equal by default; every part is also reported unweighted."""

    hazard: float = 1.0
    bc_technique: float = 1.0
    bc_target: float = 1.0
    stage: float = 1.0
    consistency: float = 1.0
    hypothesis: float = 1.0
    latent: float = 1.0
    reward: float = 1.0
    value: float = 1.0


def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.to(x.dtype)
    return (x * m).sum() / m.sum().clamp_min(1.0)


def teacher_forced(model: Forecaster, analysis: AnalysisOut, targets: ForecasterTargets, *,
                   exposure: MarginalEnergy | None = None,
                   weights: ForecasterLossWeights | None = None) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """The stage-5 Forecaster loss on real futures (module docstring). Returns (total float64, parts)."""
    assume("AS-452", by=__name__)                                               # float64 likelihoods (D-54)
    weights = ForecasterLossWeights() if weights is None else weights
    cfg = model.cfg
    b, m = analysis.context.shape[:2]
    k = targets.technique.shape[-1]
    # Map NO_TARGET to the pointer's "no target" option; −1 stays unknown.
    out = model.teacher_forced(analysis, targets.technique, targets.target_entity, exposure=exposure)
    enc, heads, h = out["enc"], out["heads"], out["h"]
    bt = b * m
    trig_ok = enc.gathered.valid.any(-1)                                         # [Bt]
    parts: dict[str, torch.Tensor] = {}

    # 1. survival NLL of the hazards on the real future.
    usable = targets.usable.reshape(bt) & trig_ok
    nll = hazard_nll(heads["hazard"], targets.event_step.reshape(bt), targets.censored.reshape(bt))
    parts["hazard"] = _masked_mean(nll, usable)

    # 2. behaviour cloning: technique (from s_{k−1}) and target pointer (given the labelled technique).
    tech = targets.technique.reshape(bt, k)
    known_t = tech >= 0
    ce_t = -torch.gather(out["log_pi"], -1, tech.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    parts["bc_technique"] = _masked_mean(ce_t, known_t & trig_ok[:, None])
    slot = out["slot"]                                                          # [Bt, K]; C = "no target", −1 unknown
    lp_s = torch.gather(out["pointer"], -1, slot.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    ok_s = (slot >= 0) & trig_ok[:, None] & torch.isfinite(lp_s)                 # a label masked by the context is skipped
    parts["bc_target"] = _masked_mean(torch.where(ok_s, -lp_s, torch.zeros_like(lp_s)), ok_s)

    # 3. per-step stage posterior: cross-entropy of softmax(stage_logits.double()), the reported link (D-54).
    st = targets.stage.reshape(bt, k + 1)
    ce_st = F.cross_entropy(heads["stage_logits"].reshape(bt * k, -1).double(), st[:, 1:].clamp_min(0).reshape(-1),
                            reduction="none").reshape(bt, k)
    parts["stage"] = _masked_mean(ce_st, (st[:, 1:] >= 0) & trig_ok[:, None])

    # 4. latent consistency against the real analysis k·stride triggers later (stop-gradient targets).
    stride = cfg.future_stride
    fut = torch.arange(m, device=h.device)[:, None] + stride * torch.arange(1, k + 1, device=h.device)[None]   # [M, K]
    fut_ok = (fut < m).unsqueeze(0).expand(b, -1, -1)                            # [B, M, K]
    fut_c = fut.clamp(max=m - 1)
    summ = enc.summary.detach().reshape(b, m, -1)
    trig_ok_bm = trig_ok.reshape(b, m)
    tgt_sum = summ[:, fut_c]                                                     # [B, M, K, dim]
    fut_valid = fut_ok & trig_ok_bm[:, fut_c]
    pred = heads["summary_pred"].reshape(b, m, k, -1)
    cons = 2.0 - 2.0 * F.cosine_similarity(pred.float(), tgt_sum.float(), dim=-1)
    parts["consistency"] = _masked_mean(cons, fut_valid & trig_ok_bm[..., None])

    # 5. imagined hypothesis of the target vs the real refined hypothesis at the later trigger.
    tgt_bm = targets.target_entity.clamp_min(-1)
    y_all = analysis.y.detach()                                                  # [B, M, V+G, d_y]
    v_ent = analysis.readouts["compromise"].shape[-1]
    b_idx = torch.arange(b, device=h.device)[:, None, None].expand(b, m, k)
    y_tgt = y_all[b_idx, fut_c[None].expand(b, -1, -1), tgt_bm.clamp(min=0, max=v_ent - 1)]   # [B, M, K, d_y]
    hyp_ok = fut_valid & (tgt_bm >= 0) & trig_ok_bm[..., None]
    mse_y = ((heads["hyp"].reshape(b, m, k, -1).float() - y_tgt.float()) ** 2).mean(-1)
    parts["hypothesis"] = _masked_mean(mse_y, hyp_ok)

    # 6. back-projected latent of the target (when posterior latents are supplied).
    if targets.latent is not None and targets.latent_mask is not None:
        mse_z = ((heads["latent"].reshape(b, m, k, -1).float() - targets.latent.detach().float()) ** 2).mean(-1)
        parts["latent"] = _masked_mean(mse_z, targets.latent_mask & trig_ok_bm[..., None])
    else:
        parts["latent"] = heads["latent"].sum() * 0.0

    # 7. reward model on the label-derived part of AS-17.
    prog = torch.tensor(STAGE_PROGRESS, dtype=torch.float32, device=h.device)
    p_prev, p_next = prog[st[:, :-1].clamp_min(0)], prog[st[:, 1:].clamp_min(0)]
    ev = targets.event_step.reshape(bt)
    infil_k = (~targets.censored.reshape(bt))[:, None] & (torch.arange(1, k + 1, device=h.device)[None] == ev[:, None])
    r_task_t = adversary_reward(p_prev, p_next, infil_k.float(), torch.zeros_like(p_next),
                                beta=cfg.infiltration_bonus, kappa=cfg.exposure_cost)   # [Bt, K]
    r_known = (st[:, :-1] >= 0) & (st[:, 1:] >= 0) & usable[:, None]
    parts["reward"] = _masked_mean((heads["reward_task"].float() - r_task_t) ** 2, r_known)

    # 8. TD(λ) value targets from process rewards (labels where known, else the model's own sg r̂).
    assume("AS-17", by=__name__)
    r_task = torch.where(r_known, r_task_t, heads["reward_task"].detach().float())
    r_total = r_task - cfg.exposure_cost * out["exposure"].float()
    v_all = out["value"].float()                                                 # [Bt, K+1]
    g_lam = lambda_returns(r_total.detach(), v_all.detach(), gamma=cfg.gamma, lam=cfg.td_lambda)   # targets: all sg
    parts["value"] = _masked_mean(0.5 * (v_all[:, :-1] - g_lam) ** 2, trig_ok[:, None].expand(-1, k))

    # Weighted sum: float64 by type promotion (hazard, BC and stage parts are float64, AS-452).
    total = sum(getattr(weights, name) * val for name, val in parts.items())
    assert isinstance(total, torch.Tensor)
    return total, parts
