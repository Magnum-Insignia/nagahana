"""Belief readouts from TAAFT's refined hypotheses y_hat (build-spec sections 2.7, 4b.4; architecture section 6).

Purpose
-------
y_hat is TAAFT's belief, refined by energy descent. The readouts turn it into the quantities defenders
and the Forecaster consume. Every belief readout is a linear functional of y_hat followed by a link
function (AS-213). Keeping them linear has two consequences:
- the hypothesis space carries the belief; the readouts only name directions in it, so a lens that
  moves y_hat moves the readouts in a way the lens shares (D-42) explain directly;
- gradients of the energy and of the readout losses act on the same coordinates.

Readouts (keys of `AnalysisOut.readouts`; shapes per trigger, stacked to [B, M, ...] by the model)
- `compromise` [V]: p_v = phi + (1 - phi) sigmoid(w_c . y_hat_v + b_c), phi = `suspicion_floor` in (0, 1)
  (architecture section 4.7, assume-breach: suspicion is never zero). p_v in [phi, 1] for every input,
  by construction.
- `stage` [V, n_stages]: softmax over the 15 stage classes (AS-19).
- `malignity` [V + G]: m = sigmoid(w . y_hat + b) in [0, 1] for entity tokens and adversary slots, with
  separate heads for the two token types (build-spec section 4b.4: "benign 20 %, malign 80 %").
- `trust` [V]: t_v = sigmoid(w_t . y_hat_v + b_t), the telemetry trust of the entity's evidence. The same
  head enters the belief-and-trust energy (lenses.py), so the trust that is reported is the trust that
  weighted the evidence (one source of truth).
- `slot_weight` [G]: omega_g = softmax_g(w_omega . y_hat_g) over active adversary slots.
- `goal` [n_goals], `type` [n_types]: sum_g omega_g softmax(W y_hat_g): a mixture of per-slot posteriors,
  so several adversary hypotheses can hold different goals at once (P-13, assumed).
- `next_latent_mean` [V, Dc], `next_latent_logvar` [V, Dc], `next_latent_logits` [V, G_z, C]: the
  believed next latent of each entity, in the shared latent space (D-20). Feeds the physics term
  (decoded and checked) and the future-latent objective.
- `latent_mean`, `latent_logvar`, `latent_logits`: the believed current latent; the masked-entity
  objective trains it on entities whose states were hidden (partial observability, stage 4).

Thermodynamic readouts (D-56; statphys/thermo.py for the mathematics)
Each is a Gibbs ensemble at the native temperature T = 1 (`THERMO_TEMPERATURE`): TAAFT's energies are
in nats and exp(-E_total) is the product of the lens experts (D-42), so T = 1 is the temperature at
which the energy is a negative log-density (AS-761). With members i, energies E_i:
log Z = logsumexp_i(-E_i / T), F = -T log Z, U = sum_i p_i E_i, S = log Z + U / T, C = Var_p(E) / T^2.
- `thermo/stage/free_energy`, `.../mean_energy`, `.../entropy`, `.../heat_capacity` [V]: the ensemble
  of the entity's stage classes with energies -logit_s. Its free energy -log sum_s exp(logit_s) is the
  energy-based novelty score of a classifier head (Liu, Wang, Owens and Li, "Energy-based
  Out-of-distribution Detection", NeurIPS 2020, arXiv:2010.03759); its entropy is the entropy of the
  `stage` posterior.
- `thermo/entities/<potential>` and `thermo/slots/<potential>` [] (one value per trigger and window),
  potential in log_partition, free_energy, mean_energy, entropy, heat_capacity, size (int64): the
  ensembles of the active entity tokens and of the active adversary slots with their per-token energies
  (`token_energy`, the sum of the lens per-token maps); `thermo/entities/valid`, `thermo/slots/valid`
  (bool): the ensemble has a member. An empty ensemble reports 0.0 with valid False, so every readout
  stays finite in training graphs (AS-762).
- `thermo/occupation` [V + G]: each token's Boltzmann occupation within its own ensemble (0 if inactive).
- `thermo/total_energy` []: E_total at y_hat, the sum of the lens energies (written by the model).
Across triggers these readouts are the energy, entropy and free-energy trajectories of the analysis;
their growth rates, read across calls and windows, are computed by statphys (trajectory.py, AS-763),
so no readout depends on where a stream is cut into calls (AS-223).

Precision (D-54)
The heads are float32 linear maps (weights stored and served in float32). Their logits are cast to
float64 before the link function, so `compromise` (with the floor phi), `stage`, `malignity`, `trust`,
`slot_weight`, `goal`, `type` and every `thermo/*` readout are float64 tensors. The cast is exact and
differentiable, so the readout losses train the float64 link functions that are reported. The latent
readouts (`latent_*`, `next_latent_*`) are not probabilities: they are float32 network quantities
that feed the Decoder and the latent likelihoods (AS-450).

Invariants (tests/test_taaft_energy.py, tests/test_precision_outputs.py, tests/test_statphys_taaft.py)
- compromise >= phi > 0 for any y_hat, including +-1e6 inputs;
- stage, goal, type and slot_weight rows sum to 1 (slot_weight is uniform when no slot is active);
- logvar in [-10, 10] (smooth bound), so the Gaussian likelihoods cannot collapse to a point;
- the posterior and thermodynamic readouts are float64 and finite.

Assumptions: AS-14 (floor value), AS-213 (linear readouts, slot mixture), AS-761, AS-762. Proposal P-13
(goal/type).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from nagahana.models.config.components import TAAFTConfig
from nagahana.statphys.thermo import GibbsState, gibbs

#: Bound of the readout log-variances (smooth, via tanh).
LOGVAR_BOUND = 10.0
#: Temperature of the thermodynamic readouts: the native temperature of the energy in nats (AS-761).
THERMO_TEMPERATURE = 1.0
#: Potentials of the token ensembles, in the order of their readout keys.
THERMO_POTENTIALS: tuple[str, ...] = ("log_partition", "free_energy", "mean_energy", "entropy", "heat_capacity")


def _bounded_logvar(x: torch.Tensor) -> torch.Tensor:
    return LOGVAR_BOUND * torch.tanh(x / LOGVAR_BOUND)


def _potentials(prefix: str, state: GibbsState) -> dict[str, torch.Tensor]:
    # Readout entries of one Gibbs state: the potentials, the member count and the validity flag.
    out = {f"{prefix}/{name}": getattr(state, name) for name in THERMO_POTENTIALS}
    out[f"{prefix}/size"] = state.size
    out[f"{prefix}/valid"] = state.valid
    return out


class Readouts(nn.Module):
    """Readout heads over y_hat. See the module docstring.

    Parameters
    ----------
    cfg: TAAFT configuration (d_hyp, suspicion_floor, n_goals, n_types).
    n_stages: stage classes (15, AS-19).
    latent_split: (Dc, G_z, C) of the shared latent space; dz = Dc + G_z * C.
    """

    def __init__(self, cfg: TAAFTConfig, *, n_stages: int, latent_split: tuple[int, int, int]) -> None:
        super().__init__()
        if not 0.0 < cfg.suspicion_floor < 1.0:
            raise ValueError("suspicion_floor must lie strictly in (0, 1) (assume-breach floor, architecture section 4.7)")
        d = cfg.d_hyp
        self.floor = float(cfg.suspicion_floor)
        self.dc, self.gz, self.cz = latent_split
        nz = 2 * self.dc + self.gz * self.cz
        self.compromise = nn.Linear(d, 1)
        self.stage = nn.Linear(d, n_stages)
        self.malignity_entity = nn.Linear(d, 1)
        self.malignity_slot = nn.Linear(d, 1)
        self.trust = nn.Linear(d, 1)
        self.slot_weight = nn.Linear(d, 1)
        self.goal_head = nn.Linear(d, cfg.n_goals)
        self.type_head = nn.Linear(d, cfg.n_types)
        self.next_latent_head = nn.Linear(d, nz)
        self.latent_head = nn.Linear(d, nz)

    def trust_logit(self, y_ent: torch.Tensor) -> torch.Tensor:
        """Trust logit per entity token: [..., V, d_y] -> [..., V] (used by E_bt and the readout)."""
        return self.trust(y_ent).squeeze(-1)

    def stage_logits(self, y_ent: torch.Tensor) -> torch.Tensor:
        """Stage logits per entity token in float64: [..., V, d_y] -> [..., V, n_stages] (an exact cast)."""
        return self.stage(y_ent).double()

    def _latent(self, head: nn.Linear, y_ent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out = head(y_ent)                                                     # [..., 2 Dc + G_z C]
        mean = out[..., : self.dc]
        logvar = _bounded_logvar(out[..., self.dc : 2 * self.dc])
        logits = out[..., 2 * self.dc :].reshape(*out.shape[:-1], self.gz, self.cz)
        return mean, logvar, logits

    def next_latent(self, y_ent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Believed next latent per entity: (mean [.., Dc], logvar [.., Dc], logits [.., G_z, C])."""
        return self._latent(self.next_latent_head, y_ent)

    def latent(self, y_ent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Believed current latent per entity (masked-entity objective)."""
        return self._latent(self.latent_head, y_ent)

    def latent_vector(self, mean: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        """z = [mean ; softmax(logits) flattened] in R^dz: the expected latent, differentiable (AS-209)."""
        probs = torch.softmax(logits.float(), dim=-1).to(mean.dtype)
        return torch.cat([mean, probs.flatten(-2)], dim=-1)

    def forward(self, y: torch.Tensor, *, n_entities: int, token_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        """y: [B, N, d_y] (N = V + G); token_mask: bool [B, N]. Returns name -> tensor (shapes above).

        Posterior readouts are float64 (D-54): float32 head logits -> `.double()` -> link function.
        """
        v = n_entities
        ye, ys = y[:, :v], y[:, v:]                                           # [B, V, d], [B, G, d]
        out: dict[str, torch.Tensor] = {}
        # Compromise with the assume-breach floor: p in [phi, 1] whatever the logit (float64, D-54;
        # phi is a Python float, so the floor is applied at float64 precision, not rounded to float32).
        out["compromise"] = self.floor + (1.0 - self.floor) * torch.sigmoid(self.compromise(ye).squeeze(-1).double())
        stage_logits = self.stage_logits(ye)                                  # [B, V, S] float64
        out["stage"] = torch.softmax(stage_logits, dim=-1)
        m_e = torch.sigmoid(self.malignity_entity(ye).squeeze(-1).double())
        m_s = torch.sigmoid(self.malignity_slot(ys).squeeze(-1).double())
        out["malignity"] = torch.cat([m_e, m_s], dim=-1)                       # [B, V+G] float64
        out["trust"] = torch.sigmoid(self.trust_logit(ye).double())            # same head as E_bt, float64 link
        # Slot mixture weights over active slots (uniform if none is active, never NaN).
        slot_mask = token_mask[:, v:]
        logit_w = self.slot_weight(ys).squeeze(-1).double()
        any_active = slot_mask.any(dim=-1, keepdim=True)
        logit_w = torch.where(slot_mask | ~any_active, logit_w, torch.full_like(logit_w, float("-inf")))
        w = torch.softmax(logit_w, dim=-1)                                     # [B, G] float64
        out["slot_weight"] = w
        # Mixtures of per-slot posteriors, all in float64: [B, n_goals], [B, n_types].
        out["goal"] = torch.einsum("bg,bgk->bk", w, torch.softmax(self.goal_head(ys).double(), dim=-1))
        out["type"] = torch.einsum("bg,bgk->bk", w, torch.softmax(self.type_head(ys).double(), dim=-1))
        mean, logvar, logits = self.next_latent(ye)
        out["next_latent_mean"], out["next_latent_logvar"], out["next_latent_logits"] = mean, logvar, logits
        mean, logvar, logits = self.latent(ye)
        out["latent_mean"], out["latent_logvar"], out["latent_logits"] = mean, logvar, logits
        # Stage-class ensemble of every entity (energies -logit_s) at the native temperature (D-56).
        st = gibbs(-stage_logits, temperature=THERMO_TEMPERATURE, fill=0.0, check_finite=False)
        for name in ("free_energy", "mean_energy", "entropy", "heat_capacity"):
            out[f"thermo/stage/{name}"] = getattr(st, name)                    # [B, V] float64
        return out

    def thermodynamics(self, token_energy: torch.Tensor, token_mask: torch.Tensor, *, n_entities: int
                       ) -> dict[str, torch.Tensor]:
        """Gibbs readouts of the entity and slot ensembles at one trigger (module docstring, D-56).

        token_energy: float64 [B, N] per-token energies at y_hat (V entity tokens, then G slots);
        token_mask: bool [B, N]. Returns `thermo/entities/*`, `thermo/slots/*` ([B]) and
        `thermo/occupation` ([B, N]), float64 (size int64, valid bool), differentiable in the energies.
        """
        v = n_entities
        e = token_energy.to(torch.float64)
        m = token_mask.to(torch.bool)
        ent = gibbs(e[:, :v], temperature=THERMO_TEMPERATURE, mask=m[:, :v], fill=0.0, check_finite=False)
        slots = gibbs(e[:, v:], temperature=THERMO_TEMPERATURE, mask=m[:, v:], fill=0.0, check_finite=False)
        out = _potentials("thermo/entities", ent)
        out.update(_potentials("thermo/slots", slots))
        out["thermo/occupation"] = torch.cat([ent.occupation, slots.occupation], dim=-1)   # [B, V + G]
        return out


def gaussian_nll(mean: torch.Tensor, logvar: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """1/2 [(x - mu)^2 exp(-log sigma^2) + log sigma^2 + log 2 pi], averaged over the last dimension -> [...]."""
    return 0.5 * (((target - mean) ** 2) * torch.exp(-logvar) + logvar + 1.8378770664093453).mean(-1)


def soft_categorical_ce(logits: torch.Tensor, target_logits: torch.Tensor) -> torch.Tensor:
    """-sum_c softmax(target)_c log softmax(logits)_c, averaged over groups: [..., G, C] -> [...]."""
    return -(torch.softmax(target_logits.float(), -1) * F.log_softmax(logits.float(), -1)).sum(-1).mean(-1)
