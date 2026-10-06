"""CVG-AE: Complex Variational Graph AutoEncoder, the encoder registry and the reference encoder (D-20, D-39).

The design ([A-01], [A-03], [A-04], [A-11])
-------------------------------------------
- Relation planes are parallel branches, one per plane ("relation-specific parallel branches", P-08).
- Each branch has its own sequence of hidden layers.
- Heterogeneous: node and hyperedge kinds get their own parameters ([Q-43]).
- Coupled: planes exchange information every layer through learned weights omega_pq.
- Variational: a posterior over a hybrid latent (models/latent.py, models/variational.py).
- Bounded by physics: reconstructions are scored by the shared Phi_phys (D-37), added by the caller's
  component loss (`objectives/template.py`).

The layer equation and the reference realisation
------------------------------------------------
    h_v^(l+1, p) = phi_(p, tau(v))( h_v^(l, p),  aggregate over e in E_p(v) of psi_(p, tau(e))({h_u^(l, p)} for u in e),
                                   sum over q != p of omega_pq h_v^(l, q) )

The reference layer (`TypedHypergraphLayer`) uses means:
    m_e  = mean over u in e of W_tau(e) h_u           psi: typed hyperedge message
    a_v  = mean over e containing v of m_e             aggregation over the hyperedges containing v
    c_v  = sum over q != p of omega_pq h_v^(q)         cross-plane coupling (from the previous layer)
    h_v' = h_v + MLP_tau(v)([h_v ; a_v ; c_v])         typed residual update
This is the two-stage node -> hyperedge -> node scheme of HGNN (Feng et al., AAAI 2019) with relation-
specific weights (Schlichtkrull et al., ESWC 2018, R-GCN) and type-dependent parameters (Hu et al.,
WWW 2020, HGT). It is the realisation written in the architecture chapter, kept as a baseline for
ablations (P-17) and permutation tests; the production encoder is `hgnn-attn` (`attn.py`, AS-03: typed
attention on sparse incidence in plain PyTorch scatter operations, so no graph library is required).

The objective (`CVGAEEncoder.loss_terms`)
-----------------------------------------
The evidence lower bound under the Decoder likelihood p_psi (architecture, CVG-AE objective):

    ELBO_v = E_q[ log p_psi(x_T | z_v) ] - beta_c KL( q(z_c | G) || N(0, I) ) - beta_d sum_g KL( q(z_d,g | G) || p(z_d,g) )

- The reconstruction target set T is the held decision D-11a, resolved to the option in force:
  "contributing fields only", "contributing fields + candidate hyperedges" (working option, AS-04), or
  "contributing fields + candidate hyperedges + next state". The likelihood terms themselves are the
  Decoder's (`models/decoder`: `Decoder.field_nll` for fields, `Decoder.edge_logits` for candidate
  hyperedges): this module only assembles the terms the option includes, so there is one implementation of
  each likelihood.
- The KL terms are computed by `models/latent_kl` in closed form: the Gaussian
  1/2 sum_d (sigma_d^2 + mu_d^2 - 1 - log sigma_d^2) and the categorical sum_c q_c (log q_c - log p_c), with
  the categorical prior uniform unless prior logits are given, and the uniform mix applied where each
  categorical is formed (AS-109).
- The Simulator minimises -mean_v ELBO_v; the physics term is added by its component loss.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, cast

import torch
from torch import nn

from nagahana.core.errors import ConfigMissing, InvariantViolation
from nagahana.core.registry import Registry
from nagahana.governance import decisions
from nagahana.graph.hypergraph import HypergraphSnapshot
from nagahana.models.latent import LatentState
from nagahana.models.latent_kl import kl_categorical_logprobs, kl_gaussian, unimix_logits
from nagahana.models.variational import reparameterize, straight_through_categorical

ENCODERS: Registry[Any] = Registry("CVG-AE encoder")

#: D-11a options (values of the decision registry) and the likelihood terms each includes.
RECONSTRUCTION_TERMS: dict[str, tuple[str, ...]] = {
    "contributing fields only": ("fields",),
    "contributing fields + candidate hyperedges": ("fields", "edges"),
    "contributing fields + candidate hyperedges + next state": ("fields", "edges", "next_state"),
}


def reconstruction_terms(configured: str | None = None) -> tuple[str, ...]:
    """The likelihood terms of the D-11a option in force (`configured`, the run's option, or AS-04)."""
    d = decisions.require("reconstruction-target", configured, by=__name__)
    assert d.value is not None
    return RECONSTRUCTION_TERMS[d.value]


class TypedHypergraphLayer(nn.Module):
    """One horizontal layer of one plane's branch (reference). See the module docstring."""

    def __init__(self, dim: int, node_kinds: Sequence[str], edge_kinds: Sequence[str]) -> None:
        super().__init__()
        self.dim = dim
        self.edge_maps = nn.ModuleDict({k: nn.Linear(dim, dim, bias=False) for k in sorted(set(edge_kinds))})
        self.node_updates = nn.ModuleDict(
            {k: nn.Sequential(nn.Linear(3 * dim, dim), nn.GELU(), nn.Linear(dim, dim)) for k in sorted(set(node_kinds))}
        )

    def forward(
        self,
        h: torch.Tensor,                   # [V, D] states of this plane
        incidence: torch.Tensor,           # [V, E] dense 0/1 incidence H_p
        edge_kinds: Sequence[str],         # len E
        node_kinds: Sequence[str],         # len V
        coupling: torch.Tensor,            # [V, D] sum over q != p of omega_pq h^(q)
    ) -> torch.Tensor:
        v, e = incidence.shape
        if len(edge_kinds) != e or len(node_kinds) != v:
            raise InvariantViolation("edge/node kind lists do not match the incidence shape")
        unknown = (set(edge_kinds) - set(self.edge_maps)) | (set(node_kinds) - set(self.node_updates))
        if unknown:
            raise InvariantViolation(f"kinds without parameters: {sorted(unknown)} (add them at construction)")
        deg_e = incidence.sum(dim=0).clamp_min(1.0)                      # |e|
        msg_e = h.new_zeros(e, self.dim)
        for kind, lin in self.edge_maps.items():
            cols = torch.tensor([i for i, k in enumerate(edge_kinds) if k == kind], dtype=torch.long)
            if cols.numel():
                h_k = incidence[:, cols]                                   # [V, E_k]
                msg_e[cols] = (h_k.t() @ lin(h)) / deg_e[cols, None]       # psi: mean over members
        deg_v = incidence.sum(dim=1).clamp_min(1.0)                        # number of hyperedges containing v
        agg = (incidence @ msg_e) / deg_v[:, None]                         # mean over e containing v
        out = h.new_zeros(h.shape)
        for kind, mlp in self.node_updates.items():
            rows = torch.tensor([i for i, k in enumerate(node_kinds) if k == kind], dtype=torch.long)
            if rows.numel():
                out[rows] = h[rows] + mlp(torch.cat([h[rows], agg[rows], coupling[rows]], dim=-1))
        return out


@ENCODERS.register("hgnn-mean-reference", summary="reference baseline: typed mean-aggregation hypergraph layers")
class CVGAEEncoder(nn.Module):
    """Relation-specific parallel branches + coupling + variational head (reference encoder).

    Parameters
    ----------
    planes: plane names, in branch order.
    in_dim: width of the per-node input features (from the field embedding, P-22).
    dim: hidden width of every branch.
    layers: horizontal layers per branch ([A-04]).
    node_kinds: every node kind the encoder must handle.
    edge_kinds: per plane, every hyperedge kind it must handle.
    cont_dim: Dc, width of the continuous latent.
    disc_groups, disc_classes: G and C of the categorical latent.
    latent_space: version tag of the latent space this encoder defines (P-19).
    """

    def __init__(
        self,
        *,
        planes: Sequence[str],
        in_dim: int,
        dim: int,
        layers: int,
        node_kinds: Sequence[str],
        edge_kinds: Mapping[str, Sequence[str]],
        cont_dim: int,
        disc_groups: int,
        disc_classes: int,
        latent_space: str,
    ) -> None:
        super().__init__()
        if set(edge_kinds) != set(planes):
            raise InvariantViolation("edge_kinds must list the hyperedge kinds of every plane")
        self.planes = tuple(planes)
        self.layers = layers
        self.latent_space = latent_space
        self.g, self.c = disc_groups, disc_classes
        self.inp = nn.Linear(in_dim, dim)
        self.branches = nn.ModuleDict(
            {p: nn.ModuleList([TypedHypergraphLayer(dim, node_kinds, edge_kinds[p]) for _ in range(layers)])
             for p in self.planes}
        )
        # omega_pq: cross-plane coupling; starts at 0 (planes independent), learned. Diagonal unused.
        self.omega = nn.Parameter(torch.zeros(len(self.planes), len(self.planes)))
        width = dim * len(self.planes)
        self.head_mean = nn.Linear(width, cont_dim)
        self.head_logvar = nn.Linear(width, cont_dim)
        self.head_logits = nn.Linear(width, disc_groups * disc_classes)

    def forward(
        self,
        snapshot: HypergraphSnapshot,
        x: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> LatentState:
        """Encode one snapshot. `x`: [V, in_dim] node inputs. Returns per-node latents ([V, ...])."""
        if tuple(snapshot.planes) != self.planes:
            raise InvariantViolation(f"snapshot planes {snapshot.planes} differ from encoder planes {self.planes}")
        incid = {p: snapshot.dense_incidence(p, dtype=x.dtype) for p in self.planes}
        ekinds = {p: snapshot.hyperedge_kinds.get(p) for p in self.planes}
        for p, kinds in ekinds.items():
            if kinds is None and snapshot.num_hyperedges(p) > 0:
                raise InvariantViolation(f"plane {p!r}: hyperedge kinds are required by the typed encoder")
        h0 = self.inp(x)
        states = {p: h0 for p in self.planes}
        mask = 1.0 - torch.eye(len(self.planes), dtype=x.dtype)
        for layer in range(self.layers):
            omega = self.omega * mask
            new: dict[str, torch.Tensor] = {}
            for i, p in enumerate(self.planes):
                coupling = sum((omega[i, j] * states[q] for j, q in enumerate(self.planes) if j != i),
                               start=torch.zeros_like(h0))
                branch = cast(nn.ModuleList, self.branches[p])
                new[p] = branch[layer](states[p], incid[p], ekinds[p] or (), snapshot.node_kinds, coupling)
            states = new
        cat = torch.cat([states[p] for p in self.planes], dim=-1)
        mean, logvar = self.head_mean(cat), self.head_logvar(cat)
        logits = self.head_logits(cat).reshape(*cat.shape[:-1], self.g, self.c)
        return LatentState(
            z_cont=reparameterize(mean, logvar, generator=generator),
            z_disc=straight_through_categorical(logits, generator=generator),
            space=self.latent_space,
            mean=mean,
            logvar=logvar,
            logits=logits,
        )

    def loss_terms(
        self,
        latent: LatentState,
        *,
        field_nll: torch.Tensor,
        edge_nll: torch.Tensor | None = None,
        next_state_nll: torch.Tensor | None = None,
        beta_c: float,
        beta_d: float,
        unimix: float = 0.0,
        prior_logits: torch.Tensor | None = None,
        configured: str | None = None,
    ) -> dict[str, torch.Tensor]:
        """The ELBO terms of a forward pass (module docstring), per node and as the loss -mean_v ELBO_v.

        latent: the output of `forward` (posterior parameters required). field_nll [V]: -log p_psi of the
        contributing cells behind each node (`Decoder.field_nll(...).sum(-1)`; exactly 0 on cells that do
        not contribute). edge_nll [V] and next_state_nll [V]: the per-node negative log-likelihoods of the
        candidate hyperedges and of the next state, required when the D-11a option in force includes them.
        beta_c, beta_d > 0: KL weights. unimix: the uniform mix of the categorical posterior (and prior).
        prior_logits [G, C] or [V, G, C]: a learned categorical prior (uniform when None).
        configured: an explicit D-11a option (default: the option in force).

        Returns 'reconstruction', 'kl_continuous', 'kl_categorical', 'elbo' (all [V]) and 'loss' (scalar).
        """
        if latent.mean is None or latent.logvar is None or latent.logits is None:
            raise InvariantViolation("loss_terms needs the posterior parameters (mean, logvar, logits) of the latent")
        if latent.space != self.latent_space:
            raise InvariantViolation(f"latent from space {latent.space!r}, encoder defines {self.latent_space!r}")
        if beta_c <= 0 or beta_d <= 0:
            raise InvariantViolation("beta_c and beta_d must be positive")
        lead_shape = latent.leading_shape
        terms = reconstruction_terms(configured)
        supplied = {"fields": field_nll, "edges": edge_nll, "next_state": next_state_nll}
        rec = torch.zeros(lead_shape, dtype=torch.float32)
        for name in terms:
            t = supplied[name]
            if t is None:
                raise ConfigMissing(f"the D-11a option in force includes '{name}'; pass its per-node negative log-likelihood")
            if t.shape != lead_shape:
                raise InvariantViolation(f"{name} NLL has shape {tuple(t.shape)}, the latent has {tuple(lead_shape)}")
            if not torch.isfinite(t).all():
                raise InvariantViolation(f"{name} NLL is not finite")
            rec = rec + t.float()
        # KL to the fixed prior, in closed form (models/latent_kl): Gaussian to N(0, I), categorical to the prior.
        zeros = torch.zeros_like(latent.mean, dtype=torch.float32)
        kl_c = kl_gaussian(latent.mean, latent.logvar, zeros, zeros)                    # [V]
        log_q = unimix_logits(latent.logits, unimix)                                     # [V, G, C]
        if prior_logits is None:
            log_p = torch.full_like(log_q, -math.log(latent.logits.shape[-1]))           # uniform prior
        else:
            log_p = unimix_logits(prior_logits.expand_as(latent.logits), unimix)
        kl_d = kl_categorical_logprobs(log_q, log_p).sum(dim=-1)                         # sum over groups, [V]
        elbo = -rec - beta_c * kl_c - beta_d * kl_d
        return {"reconstruction": rec, "kl_continuous": kl_c, "kl_categorical": kl_d, "elbo": elbo,
                "loss": -elbo.mean()}
