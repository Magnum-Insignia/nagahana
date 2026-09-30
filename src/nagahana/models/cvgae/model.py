"""CVG-AE: Complex Variational Graph AutoEncoder (D-20, D-39; [A-03], [A-04], [A-11]).

The owner's design
------------------
"the Multi-layer GNN Variational Autoencoder with math topological modeling (Complex Variational
Graph AE or CVG-AE): which encodes the multiple layers of graph … this produces the current state
(observed from whatever is given at input) of the network topology graph (world state) -- and this
also work as the latent space generator" [A-11].

- **Vertical layers = relation planes**, one parallel branch per plane [A-01], [A-03] (proposed
  term: "relation-specific parallel branches", P-08).
- **Horizontal layers**: each branch has its own sequence of hidden layers [A-04].
- **Heterogeneous**: node and hyperedge *kinds* get their own parameters [Q-43].
- **Coupled**: planes exchange information every layer through learned weights ω_pq.
- **Variational**: a posterior over a hybrid latent (models/latent.py, models/variational.py).
- **Guided by physics**: reconstructions are scored by the shared Φ_phys (D-37) [Q-23].

The layer equation (legend 00, diagram 01)
------------------------------------------
    h_v^{(ℓ+1,p)} = φ_{p,τ(v)}( h_v^{(ℓ,p)},  ⊕_{e ∈ ℰ_p(v)} ψ_{p,τ(e)}({h_u^{(ℓ,p)}}_{u ∈ e}),
                                Σ_{q ≠ p} ω_pq · h_v^{(ℓ,q)} )

Reference realisation below (`TypedHypergraphLayer`):
    m_e  = mean_{u ∈ e} W_{τ(e)} h_u               ψ: typed hyperedge message (mean over members)
    a_v  = mean_{e ∋ v} m_e                         ⊕: mean over the hyperedges containing v
    c_v  = Σ_{q ≠ p} ω_pq h_v^{(q)}                 cross-plane coupling (from the previous layer)
    h_v' = h_v + MLP_{τ(v)}([h_v ; a_v ; c_v])      φ: typed residual update

This is the HGNN two-stage node→hyperedge→node scheme (Feng et al., AAAI 2019) with R-GCN-style
relation-specific weights (Schlichtkrull et al., ESWC 2018) and HGT-style type-dependent parameters
(Hu et al., WWW 2020).

What is reference and what is open
----------------------------------
- `hgnn-mean-reference` is a **baseline**: dense incidence matrices, mean aggregation. It exists so
  that the rest of the pipeline, the tests (e.g. permutation equivariance: relabelling nodes
  relabels outputs, ARCH §4.4) and ablations have something real to run. The production layer
  (sparse PyG `HypergraphConv`, attention aggregation, topology-aware features) is a separate
  registry entry.
- The ELBO's reconstruction target is held (D-11a). `CVGAE.loss_terms` raises until it is decided.
- Plane formation is held (D-04). The encoder takes planes as given.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast

import torch
from torch import nn

from nagahana.core.errors import InvariantViolation, NotBuiltYet
from nagahana.core.registry import Registry
from nagahana.governance import decisions
from nagahana.graph.hypergraph import HypergraphSnapshot
from nagahana.models.latent import LatentState
from nagahana.models.variational import reparameterize, straight_through_categorical

ENCODERS: Registry[Any] = Registry("CVG-AE encoder")


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
        coupling: torch.Tensor,            # [V, D] Σ_{q≠p} ω_pq h^(q)
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
                msg_e[cols] = (h_k.t() @ lin(h)) / deg_e[cols, None]       # ψ: mean over members
        deg_v = incidence.sum(dim=1).clamp_min(1.0)                        # number of hyperedges ∋ v
        agg = (incidence @ msg_e) / deg_v[:, None]                         # ⊕: mean over e ∋ v
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
    layers: horizontal layers per branch [A-04].
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
        # ω_pq: cross-plane coupling; starts at 0 (planes independent), learned. Diagonal unused.
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
        """Encode one snapshot. `x`: [V, in_dim] node inputs. Returns per-node latents ([V, …])."""
        if tuple(snapshot.planes) != self.planes:
            raise InvariantViolation(f"snapshot planes {snapshot.planes} ≠ encoder planes {self.planes}")
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

    def loss_terms(self, *_args: object, **_kwargs: object) -> dict[str, torch.Tensor]:
        """ELBO terms. The reconstruction target is held (D-11a); see objectives/template.py."""
        decisions.require("reconstruction-target")
        raise NotBuiltYet("CVG-AE ELBO", waiting_on=("D-11a", "P-20"))


@ENCODERS.register("pyg-hypergraphconv", summary="production path on PyG sparse ops (template)")
def _pyg_encoder(**_kwargs: object) -> nn.Module:
    raise NotBuiltYet("CVG-AE on torch_geometric HypergraphConv / HeteroData", waiting_on=("D-04", "install [graph] extra"))
