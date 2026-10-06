"""CVG-AE production encoder `hgnn-attn`: typed attention on sparse multiplex hypergraphs (build-spec section 2.3).

Purpose
-------
Encode each position's local hypergraph as of its own time (built by `graph.window`) into the posterior of
the hybrid latent z = (z_c in R^Dc, z_d in (Simplex^C)^G): the world state the rest of NagaHana reads
([A-11], [A-13]).

Decisions and assumptions
-------------------------
- [A-01], [A-03], [A-04]: one parallel branch per relation plane, each a stack of layers; D-39
  heterogeneous multiplex hypergraph; D-20 variational hybrid latent; D-49 RWSE, hop bias, kind and role
  embeddings, no index-based encodings.
- AS-03 (typed attention aggregation on sparse incidence), AS-05 (unimix 1 %, posterior mean at
  inference), AS-108 (details below).
- D-04 (plane formation, held; option in force from `graph.planes.formation_policy`): under the working
  option "declared" (AS-01) the branches are the declared planes; under "learned" or "declared + learned"
  the encoder adds K learned planes (AS-720, below).

Mathematics, per branch p and layer l (H heads, d_h = dim / H; tau(e) hyperedge kind, tau(v) node kind)
--------------------------------------------------------------------------------------------------
    node -> hyperedge   alpha_ue = softmax over u in e of ( <a_tau(e), W^K_tau(e) h_u>_h / sqrt(d_h) )
                        m_e      = sum over u in e of alpha_ue * W^V_tau(e) h_u
    hyperedge -> node   beta_ev  = softmax over e containing v of
                                   ( <N(W^Q h_v), N(m_e)>_h / sqrt(d_h) + b_hop[hop(e)] + b_size[floor(log2 |e|)] )
                        g_v      = sum over e containing v of beta_ev * m_e       (0 if v is in no hyperedge of p)
    coupling            c_v      = sum over q != p of omega^l_pq h_v^(l, q)
    typed update        h_v^(l+1, p) = h_v^(l, p) + MLP_(p, l, tau(v))( RMSNorm([h_v ; g_v ; c_v ; rho_v^p]) )

- Two-stage HGNN scheme (Feng et al., "Hypergraph Neural Networks", AAAI 2019, arXiv:1809.09401) with
  attention instead of means, and HGT-style type-dependent parameters (Hu et al., "Heterogeneous Graph
  Transformer", WWW 2020, arXiv:2003.01332).
- N is a per-head RMSNorm of queries and hyperedge messages (QK-norm, AS-32).
- hop(e) = min hop of the members of e from the centre (a bias that depends only on v would cancel in the
  softmax over e containing v, so the hop bias is carried by the hyperedge); |e| counts members in the
  subgraph.
- rho_v^p: the RWSE of v on plane p (`GraphBatch.rwse`).
- Segment softmaxes use scatter with an amax shift (`scatter_reduce`), then `index_add`: stable and exact
  on sparse incidence, with no dense V x E matrix.

Learned planes (D-04 options "learned" and "declared + learned", AS-720)
------------------------------------------------------------------------
A learned plane k is a soft membership s_k(e) in (0, 1) of every relation e of the connectivity plane
(which holds every flow, AS-01), computed once per forward from the layer-0 node inputs:
    d_e   = [ E_kind[tau(e)] ; mean over u in e of h_u^(0) ; log |e| ]
    s(e)  = sigmoid( MLP_gate(d_e) ) in (0, 1)^K
Branch k runs the same layer on the connectivity incidence with the membership in the hyperedge -> node
softmax and a learned null option (logit b_null per head, message 0):
    beta_ev = s_k(e) exp(score_ev) / ( exp(b_null) + sum over e' containing v of s_k(e') exp(score_e'v) )
so a hyperedge outside plane k (s_k -> 0) carries no message and a node whose hyperedges are all outside
the plane receives nothing (the null option takes the mass). This is the soft selection of relation types
of Graph Transformer Networks (Yun et al., NeurIPS 2019, arXiv:1911.06455) carried to hyperedges; the
null option is the learned null key of the attention blocks (AS-32). The learned planes are flagged as
inferred (`branch_specs`), and their memberships are reported in `last_plane_gates`. With K = 0 (the
working option) no learned module exists, so the parameter count of the L preset is unchanged.

Node input (build-spec section 2.3)
-----------------------------------
    h_v^(0) = RMSNorm( W_u u_v + E_role[role_v] + E_kind[tau(v)] + E_hop[hop_v] + Per(log(1 + age_v)) + E_age[b(age_v)] )
with u_v the FieldEncoder vector of the node's latest update as of the centre time, or a learned "unseen"
vector when there is none (node_update = -1). b(age) is the learned log-time bucket of
`nn.positional.LogDeltaBias` (no decay is imposed with age, build-spec section 4b.1); Per is the periodic
embedding of `nn.numeric` (AS-31 form).

Variational head on the centre node (branches concatenated)
-----------------------------------------------------------
    c = RMSNorm([h^(L, p)_centre]_p),   mu = W_mu c,   log sigma^2 = b tanh(W_sigma c / b),   l = W_l c in R^(G x C)
    sample:  z_c = mu + sigma * eps,   z_d,g = one_hot(k) + pi - sg(pi),  k ~ Cat(pi),  pi = (1 - u) softmax(l_g) + u / C
    mean (sample=False):  z_c = mu,  z_d,g = pi
`logits` are returned raw; every consumer forms the categorical with `latent_kl.unimix_logits` (AS-109), so
the mixing is applied exactly once per distribution.

Invariants (tests/test_perception_cvgae.py, tests/test_core_cvgae_planes.py)
---------------------------------------------------------------------------
- Permutation equivariance: relabelling nodes and hyperedges leaves every position's output unchanged;
  permuting positions permutes the outputs (also with learned planes).
- Gradients reach every parameter. Padded positions (center = -1) output zeros.
- The last layer computes the hyperedge -> node stage and the typed MLP on the centre rows only (the head
  reads nothing else); this equals the full layer restricted to those rows (tested). A node kind that is
  never a centre gets no gradient in the last layer's MLP.

Extension points
----------------
- Attention weights for explanations: `forward(..., need_weights=True)` stores them in
  `self.last_attention` (branch name -> layer -> (alpha, beta)) and the learned memberships in
  `self.last_plane_gates`.
"""

from __future__ import annotations

import math
from typing import cast

import torch
import torch.nn.functional as F
from torch import nn

from nagahana.core.errors import InvariantViolation
from nagahana.governance.assumptions import assume
from nagahana.graph.planes import SUBSTRATE_PLANE, PlaneFormation, PlaneSpec, formation_policy, plane_specs
from nagahana.models.batch import GraphBatch
from nagahana.models.config.components import CVGAEConfig, GraphConfig
from nagahana.models.cvgae.model import ENCODERS
from nagahana.models.latent_kl import unimix_logits
from nagahana.models.variational import reparameterize, straight_through_categorical
from nagahana.models.vocab import ROLES
from nagahana.nn.norms import RMSNorm
from nagahana.nn.numeric import PeriodicEmbedding
from nagahana.nn.positional import LogDeltaBias


def segment_softmax(scores: torch.Tensor, segment: torch.Tensor, n_segments: int) -> torch.Tensor:
    """Softmax of `scores` [nnz, H] within groups given by `segment` [nnz] (values in [0, n_segments)).

    Stable: each group is shifted by its maximum (scatter amax). Float32 throughout.
    """
    s = scores.float()
    idx = segment.view(-1, 1).expand_as(s)
    mx = torch.full((n_segments, s.shape[1]), float("-inf"), dtype=s.dtype, device=s.device)
    mx = mx.scatter_reduce(0, idx, s, reduce="amax", include_self=True)     # [n_seg, H]
    ex = torch.exp(s - mx[segment].detach())                                 # the shift does not change the softmax
    den = torch.zeros_like(mx).index_add(0, segment, ex)                     # [n_seg, H]
    return ex / den[segment]


#: Smallest membership entering the log of a learned plane's attention (log(1e-6) is about -13.8).
_GATE_FLOOR = 1e-6


class _PlaneLayer(nn.Module):
    """One horizontal layer of one branch. See the module docstring for the equations.

    `gated` adds the learned null option of a learned plane (b_null per head); the membership itself is
    passed to `forward` as `edge_gate`.
    """

    def __init__(self, cfg: CVGAEConfig, n_edge_kinds: int, n_node_kinds: int, rwse_steps: int, *, gated: bool = False) -> None:
        super().__init__()
        d, h = cfg.dim, cfg.heads
        self.h, self.dh = h, d // h
        self.n_ek, self.n_nk = n_edge_kinds, n_node_kinds
        # Typed node -> hyperedge parameters (HGT-style, one set per hyperedge kind).
        self.w_k = nn.Parameter(torch.randn(n_edge_kinds, d, d) / math.sqrt(d))
        self.w_v = nn.Parameter(torch.randn(n_edge_kinds, d, d) / math.sqrt(d))
        self.a = nn.Parameter(torch.randn(n_edge_kinds, h, self.dh) / math.sqrt(self.dh))
        # Hyperedge -> node: queries, QK-norm, hop and size biases.
        self.w_q = nn.Linear(d, d, bias=False)
        self.q_norm = RMSNorm(self.dh)
        self.m_norm = RMSNorm(self.dh)
        self.b_hop = nn.Parameter(torch.zeros(cfg.max_hops_bias, h))
        self.b_size = nn.Parameter(torch.zeros(cfg.size_buckets, h))
        # Typed residual update, one MLP per node kind.
        width = 3 * d + rwse_steps
        self.norm = RMSNorm(width)
        self.mlps = nn.ModuleList(
            [nn.Sequential(nn.Linear(width, cfg.mlp_mult * d), nn.GELU(), nn.Linear(cfg.mlp_mult * d, d))
             for _ in range(n_node_kinds)]
        )
        self.max_hops_bias, self.size_buckets = cfg.max_hops_bias, cfg.size_buckets
        self.gated = gated
        if gated:
            self.b_null = nn.Parameter(torch.zeros(h))                       # null-option logit per head

    def forward(
        self,
        h: torch.Tensor,             # [N, d] states of this branch
        incidence: torch.Tensor,     # [2, nnz] (node, hyperedge)
        edge_kind: torch.Tensor,     # [E]
        node_kind: torch.Tensor,     # [N]
        node_hop: torch.Tensor,      # [N]
        rwse: torch.Tensor,          # [N, K] RWSE on this branch's plane
        coupling: torch.Tensor,      # [N, d]
        out_rows: torch.Tensor | None = None,
        edge_gate: torch.Tensor | None = None,   # [E] soft membership of a learned plane (gated layers only)
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        """New states [N, d], or only the rows `out_rows` [R, d] when given (the last layer needs only the
        centre nodes: messages still flow from every node into the hyperedges, but the hyperedge -> node
        stage and the MLP run on the R output rows only)."""
        if (edge_gate is not None) != self.gated:
            raise InvariantViolation("a learned-plane layer needs its membership, a declared-plane layer takes none")
        n, d = h.shape
        v_idx, e_idx = incidence[0], incidence[1]
        n_e = int(edge_kind.shape[0])
        # Output rows: all nodes, or the requested subset (map node -> output row, -1 if not an output).
        rows_out = torch.arange(n) if out_rows is None else out_rows
        r_n = int(rows_out.shape[0])
        out_of = torch.full((n,), -1, dtype=torch.long).index_copy(0, rows_out, torch.arange(r_n))
        g = h.new_zeros(r_n, d)
        att: tuple[torch.Tensor, torch.Tensor] | None = None
        if v_idx.numel():
            # Node -> hyperedge: typed keys and values per incidence pair (computed per kind subset).
            ek = edge_kind[e_idx]                                            # [nnz]
            hv = h[v_idx]                                                    # [nnz, d]
            keys = hv.new_zeros(hv.shape)
            vals = hv.new_zeros(hv.shape)
            for k in range(self.n_ek):
                sel = torch.nonzero(ek == k).flatten()
                if sel.numel():
                    keys = keys.index_copy(0, sel, hv[sel] @ self.w_k[k])
                    vals = vals.index_copy(0, sel, hv[sel] @ self.w_v[k])
            keys = keys.view(-1, self.h, self.dh)                            # [nnz, H, d_h]
            vals = vals.view(-1, self.h, self.dh)
            s1 = (keys * self.a[ek]).sum(-1) / math.sqrt(self.dh)            # [nnz, H]
            alpha = segment_softmax(s1, e_idx, n_e)                          # softmax over u in e
            m = torch.zeros(n_e, self.h, self.dh, dtype=vals.dtype).index_add(0, e_idx, alpha.unsqueeze(-1).to(vals.dtype) * vals)
            # Hyperedge features for the biases: hop(e) = min member hop, |e| = member count.
            hop_e = torch.full((n_e,), self.max_hops_bias - 1, dtype=torch.long).scatter_reduce(
                0, e_idx, node_hop.clamp(0, self.max_hops_bias - 1)[v_idx], reduce="amin", include_self=True)
            size_e = torch.bincount(e_idx, minlength=n_e).clamp_min(1)
            size_b = torch.floor(torch.log2(size_e.float())).long().clamp(0, self.size_buckets - 1)
            # Hyperedge -> node attention, on incidence pairs whose node is an output row.
            keep = torch.nonzero(out_of[v_idx] >= 0).flatten()
            vo, eo = out_of[v_idx[keep]], e_idx[keep]                        # output row, hyperedge
            q = self.q_norm(self.w_q(h[rows_out]).view(r_n, self.h, self.dh))   # [R, H, d_h]
            mk = self.m_norm(m)                                              # [E, H, d_h]
            s2 = (q[vo] * mk[eo]).sum(-1) / math.sqrt(self.dh)               # [nnz_R, H]
            s2 = s2 + self.b_hop[hop_e[eo]] + self.b_size[size_b[eo]]
            if edge_gate is None:
                beta = segment_softmax(s2, vo, r_n)                          # softmax over e containing v
            else:
                # Learned plane: membership in the logits, plus one null option per output row (message 0).
                s2 = s2 + torch.log(edge_gate[eo].clamp_min(_GATE_FLOOR)).unsqueeze(-1).to(s2.dtype)
                null = self.b_null.unsqueeze(0).expand(r_n, self.h).to(s2.dtype)
                beta = segment_softmax(torch.cat([s2, null]), torch.cat([vo, torch.arange(r_n)]), r_n)[: s2.shape[0]]
            g = torch.zeros(r_n, self.h, self.dh, dtype=m.dtype).index_add(0, vo, beta.unsqueeze(-1).to(m.dtype) * m[eo])
            g = g.view(r_n, d)
            att = (alpha, beta)
        # Typed residual update on the output rows.
        h_o = h[rows_out]
        x = self.norm(torch.cat([h_o, g, coupling[rows_out], rwse[rows_out].to(h.dtype)], dim=-1))  # [R, 3d + K]
        kind_o = node_kind[rows_out]
        delta = h.new_zeros(r_n, d)
        for k in range(self.n_nk):
            rows = torch.nonzero(kind_o == k).flatten()
            if rows.numel():
                mlp = cast(nn.Sequential, self.mlps[k])
                delta = delta.index_copy(0, rows, mlp(x[rows]))
        return h_o + delta, att


@ENCODERS.register("hgnn-attn", summary="production: typed attention on sparse multiplex hypergraphs (AS-03)")
class CVGAE(nn.Module):
    """Production CVG-AE encoder. See the module docstring.

    Parameters
    ----------
    cfg: `CVGAEConfig` (L: 6 planes x 4 layers x 256, z = 128 + 16 x 32).
    graph: `GraphConfig` (planes, node and hyperedge kinds, rwse_steps, learned_planes).
    d_update: width of the FieldEncoder update vectors.
    formation: the D-04 option; None resolves the option in force (`graph.planes.formation_policy`).
    """

    def __init__(self, cfg: CVGAEConfig, graph: GraphConfig, d_update: int, *, formation: PlaneFormation | None = None) -> None:
        super().__init__()
        assume("AS-03", by=__name__)
        assume("AS-05", by=__name__)
        if cfg.dim % cfg.heads:
            raise ValueError("CVG-AE dim must be divisible by heads")
        if cfg.layers < 1:
            raise ValueError("CVG-AE needs at least one layer")
        self.cfg, self.planes = cfg, tuple(graph.planes)
        self.rwse_steps = graph.rwse_steps
        self.formation = formation if formation is not None else formation_policy()
        #: Branch specifications: declared planes (unless the option is "learned"), then learned planes.
        self.branch_specs: tuple[PlaneSpec, ...] = plane_specs(self.planes, graph.learned_planes, self.formation)
        self.declared = tuple(s.name for s in self.branch_specs if s.name in self.planes)
        self.n_learned = graph.learned_planes
        n_br, d = len(self.branch_specs), cfg.dim
        # Node input.
        self.unseen = nn.Parameter(torch.randn(d_update) * 0.02)
        self.in_update = nn.Linear(d_update, d)
        self.role = nn.Embedding(len(ROLES), d)
        self.kind = nn.Embedding(len(graph.node_kinds), d)
        self.hop = nn.Embedding(cfg.max_hops_bias, d)
        self.age_periodic = PeriodicEmbedding(1, cfg.age_frequencies, d)
        self.age_bucket = LogDeltaBias(d, n_buckets=cfg.age_buckets)         # table [buckets, d] as embedding
        self.in_norm = RMSNorm(d)
        # Declared branches x layers, and per-layer cross-branch coupling omega (starts at 0: branches independent).
        self.branches = nn.ModuleList(
            [nn.ModuleList([_PlaneLayer(cfg, len(graph.hyperedge_kinds), len(graph.node_kinds), graph.rwse_steps)
                            for _ in range(cfg.layers)]) for _ in self.declared]
        )
        self.omega = nn.Parameter(torch.zeros(cfg.layers, n_br, n_br))
        self.register_buffer("_offdiag", 1.0 - torch.eye(n_br), persistent=False)
        self._offdiag: torch.Tensor
        # Variational head on the centre node.
        self.head_norm = RMSNorm(n_br * d)
        self.head_mean = nn.Linear(n_br * d, cfg.cont_dim)
        self.head_logvar = nn.Linear(n_br * d, cfg.cont_dim)
        self.head_logits = nn.Linear(n_br * d, cfg.disc_groups * cfg.disc_classes)
        # Learned planes (D-04 options with learning, AS-720): gated branches and the membership network.
        if self.n_learned:
            assume("AS-01", by=__name__)
            self.learned_branches = nn.ModuleList(
                [nn.ModuleList([_PlaneLayer(cfg, len(graph.hyperedge_kinds), len(graph.node_kinds), graph.rwse_steps,
                                            gated=True) for _ in range(cfg.layers)]) for _ in range(self.n_learned)]
            )
            self.gate_kind = nn.Embedding(len(graph.hyperedge_kinds), d)
            self.gate_mlp = nn.Sequential(nn.Linear(2 * d + 1, d), nn.SiLU(), nn.Linear(d, self.n_learned))
        self.last_attention: dict[str, list[tuple[torch.Tensor, torch.Tensor] | None]] = {}
        self.last_plane_gates: dict[str, torch.Tensor] = {}

    @property
    def latent_dim(self) -> int:
        """dz = Dc + G * C."""
        return self.cfg.cont_dim + self.cfg.disc_groups * self.cfg.disc_classes

    def node_inputs(self, graph: GraphBatch, update_vec_flat: torch.Tensor) -> torch.Tensor:
        """h^(0) [N, dim] from the node fields of `graph` and the FieldEncoder vectors [B*U, d_update]."""
        nu = graph.node_update
        if nu.numel() and int(nu.max()) >= update_vec_flat.shape[0]:
            raise InvariantViolation("node_update points past update_vec_flat")
        seen = (nu >= 0).unsqueeze(-1)
        u = torch.where(seen, update_vec_flat[nu.clamp_min(0)], self.unseen.to(update_vec_flat.dtype))
        age = graph.node_age.float().clamp_min(0.0)
        x = (self.in_update(u) + self.role(graph.node_role) + self.kind(graph.node_kind)
             + self.hop(graph.node_hop.clamp(0, self.cfg.max_hops_bias - 1))
             + self.age_periodic(torch.log1p(age).unsqueeze(-1))[:, 0] + self.age_bucket(age))
        return self.in_norm(x)

    def plane_memberships(self, graph: GraphBatch, h0: torch.Tensor) -> torch.Tensor:
        """s(e) in (0, 1)^K for every hyperedge of the connectivity plane (module docstring): [E_conn, K]."""
        inc = graph.incidence[SUBSTRATE_PLANE]
        kinds = graph.hyperedge_kind[SUBSTRATE_PLANE]
        n_e = int(kinds.shape[0])
        d = h0.shape[1]
        if n_e == 0:
            return h0.new_zeros(0, self.n_learned)
        v_idx, e_idx = inc[0], inc[1]
        size = torch.bincount(e_idx, minlength=n_e).clamp_min(1).to(h0.dtype)            # [E]
        mean = torch.zeros(n_e, d, dtype=h0.dtype).index_add(0, e_idx, h0[v_idx]) / size[:, None]
        desc = torch.cat([self.gate_kind(kinds), mean, torch.log(size).unsqueeze(-1)], dim=-1)   # [E, 2d + 1]
        return torch.sigmoid(self.gate_mlp(desc))

    def forward(
        self,
        graph: GraphBatch,
        update_vec_flat: torch.Tensor,
        n_positions: int,
        *,
        sample: bool,
        generator: torch.Generator | None = None,
        need_weights: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """(z [n_pos, dz], mean [n_pos, Dc], logvar [n_pos, Dc], logits [n_pos, G, C]); logits raw.

        `graph.center` must have `n_positions` entries (-1 = padded position, zero rows).
        sample=True draws the posterior sample (training); False returns the posterior mean (AS-05).
        """
        if graph.center.shape[0] != n_positions:
            raise InvariantViolation(f"graph.center has {graph.center.shape[0]} entries, expected {n_positions}")
        if tuple(graph.incidence) != self.planes:
            raise InvariantViolation(f"graph planes {tuple(graph.incidence)} differ from encoder planes {self.planes}")
        h0 = self.node_inputs(graph, update_vec_flat)                       # [N, d]
        names = [s.name for s in self.branch_specs]
        states = [h0 for _ in names]
        self.last_attention = {nm: [] for nm in names} if need_weights else {}
        self.last_plane_gates = {}
        gates = self.plane_memberships(graph, h0) if self.n_learned else None   # [E_conn, K]
        if gates is not None and need_weights:
            self.last_plane_gates = {names[len(self.declared) + k]: gates[:, k].detach() for k in range(self.n_learned)}
        valid = graph.center >= 0
        ctr = graph.center[valid]                                            # centre node of each real position
        sub = self.planes.index(SUBSTRATE_PLANE) if self.n_learned else -1
        for layer in range(self.cfg.layers):
            stack = torch.stack(states)                                      # [n_br, N, d]
            coupling = torch.einsum("pq,qnd->pnd", self.omega[layer] * self._offdiag, stack)
            last = layer == self.cfg.layers - 1                              # last layer: centre rows only
            new: list[torch.Tensor] = []
            for bi, nm in enumerate(names):
                if bi < len(self.declared):
                    pi = self.planes.index(nm)
                    branch = cast(nn.ModuleList, self.branches[bi])
                    out, att = branch[layer](states[bi], graph.incidence[nm], graph.hyperedge_kind[nm], graph.node_kind,
                                             graph.node_hop, graph.rwse[:, pi, :], coupling[bi],
                                             out_rows=ctr if last else None)
                else:
                    k = bi - len(self.declared)
                    assert gates is not None
                    lbranch = cast(nn.ModuleList, self.learned_branches[k])
                    out, att = lbranch[layer](states[bi], graph.incidence[SUBSTRATE_PLANE],
                                              graph.hyperedge_kind[SUBSTRATE_PLANE], graph.node_kind, graph.node_hop,
                                              graph.rwse[:, sub, :], coupling[bi], out_rows=ctr if last else None,
                                              edge_gate=gates[:, k])
                new.append(out)
                if need_weights:
                    self.last_attention[nm].append(att)
            states = new
        # Variational head on centre nodes (the last layer returned their rows); padded positions stay zero.
        cat = self.head_norm(torch.cat(states, dim=-1))                      # [n_valid, n_br * d]
        g_, c_ = self.cfg.disc_groups, self.cfg.disc_classes
        mean_v = self.head_mean(cat)
        b = self.cfg.logvar_bound
        logvar_v = b * torch.tanh(self.head_logvar(cat) / b)                 # smooth bound (AS-108)
        logits_v = self.head_logits(cat).view(-1, g_, c_)
        mixed = unimix_logits(logits_v, self.cfg.unimix)                     # log pi, 1 % uniform (AS-05)
        if sample:
            z_c = reparameterize(mean_v, logvar_v, generator=generator)
            z_d = straight_through_categorical(mixed, generator=generator)
        else:
            z_c, z_d = mean_v, mixed.exp()
        z_v = torch.cat([z_c, z_d.flatten(1).to(z_c.dtype)], dim=-1)
        # Scatter into position rows (zeros for padded positions).
        rows = torch.nonzero(valid).flatten()
        z = z_v.new_zeros(n_positions, z_v.shape[1]).index_copy(0, rows, z_v)
        mean = mean_v.new_zeros(n_positions, mean_v.shape[1]).index_copy(0, rows, mean_v)
        logvar = logvar_v.new_zeros(n_positions, logvar_v.shape[1]).index_copy(0, rows, logvar_v)
        logits = logits_v.new_zeros(n_positions, g_, c_).index_copy(0, rows, logits_v)
        return z, mean, logvar, logits


def posterior_probs(logits: torch.Tensor, unimix: float) -> torch.Tensor:
    """pi = (1 - u) softmax(l) + u / C from raw CVG-AE logits [..., G, C]."""
    return F.softmax(unimix_logits(logits, unimix), dim=-1)
