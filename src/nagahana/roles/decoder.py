"""The Decoder role: the live, provenance-tagged view of memory ([A-13]; D-05, P-01).

What the role does
------------------
It reads the Environment and Imagination (memory/access.py: the Decoder holds READ on both, and the Monitor
only under proposal P-12), decodes latents with the working Decoder (`models.decoder.model.Decoder.view`: the
field likelihoods, hard-limited decoding, candidate hyperedges and the provenance tags of AS-112), and flattens
the relation planes into one view by the option of the held D-05 in force (`governance.decisions`). Every
element keeps its tag OBSERVED, BELIEVED or FORECAST; a belief or a forecast never renders as observed (P-01).

Flattenings (the admissible options of D-05)
--------------------------------------------
- "typed multigraph" (working option, AS-29): planes kept as edge types; the Decoder's view as it is.
- "supra-graph": the lossless multilayer representation of De Domenico et al., "Mathematical Formulation of
  Multilayer Networks", Physical Review X 3, 041022 (2013): one node per (entity, plane) pair that the view
  touches, every hyperedge on the replicas of its plane, and coupling edges joining the replicas of one entity
  across planes. A coupling edge is OBSERVED only when the entity has an OBSERVED hyperedge on both planes,
  BELIEVED otherwise, FORECAST in a forecast view (AS-726).
- "reducibility merge": De Domenico, Nicosia, Arenas and Latora, "Structural reducibility of multilayer
  networks", Nature Communications 6, 6864 (2015). For each plane a, the clique expansion A_a of its hyperedges
  over the view's entities (weights = the elements' values) gives the density rho_a = L_a / tr(L_a) of its
  Laplacian L_a = D_a - A_a, with von Neumann entropy h(rho) = -sum_i lambda_i log2 lambda_i. Planes are
  clustered hierarchically (Ward's linkage, as in the paper) on the Jensen-Shannon distance
      d(a, b) = sqrt( h((rho_a + rho_b) / 2) - (h(rho_a) + h(rho_b)) / 2 ),
  and every level of the dendrogram is scored by the relative entropy q = 1 - H_bar / h(rho_aggregate), H_bar
  the mean entropy of the level's (merged) planes; the level of largest q is kept. Planes with no hyperedge in
  the view carry no structure and are left as they are. Merged planes are named "a+b"; a hyperedge present on
  several merged planes is kept once, OBSERVED if it was observed on any of them, with its largest value.
- "combination": the reducibility merge, then the supra-graph of the merged planes (AS-726).
"""

from __future__ import annotations

import math
from collections.abc import Collection, Mapping
from dataclasses import dataclass

import numpy as np
import torch

from nagahana.core.errors import InvariantViolation
from nagahana.core.roles import Role
from nagahana.governance import decisions
from nagahana.memory.access import Op, Region, check
from nagahana.models.decoder.model import DecodedView, Decoder, ViewElement
from nagahana.models.latent import LatentState

#: The admissible options of D-05.
FLATTENINGS: tuple[str, ...] = ("typed multigraph", "supra-graph", "reducibility merge", "combination")


@dataclass(frozen=True)
class SupraHyperedge:
    """A hyperedge of the supra-graph: replica node ids on its plane, kind, value and provenance."""

    nodes: tuple[int, ...]
    plane: str
    kind: str
    value: float
    provenance: str


@dataclass(frozen=True)
class SupraGraph:
    """The supra-graph of a view (module docstring): replicas, intra-plane hyperedges, coupling edges, fields."""

    source: str
    latent_space: str
    planes: tuple[str, ...]
    nodes: tuple[tuple[int, str], ...]
    hyperedges: tuple[SupraHyperedge, ...]
    coupling: tuple[tuple[int, int, str], ...]
    fields: tuple[ViewElement, ...]

    def __post_init__(self) -> None:
        for el in (*self.hyperedges, *((None, None, c[2]) for c in self.coupling)):
            prov = el.provenance if isinstance(el, SupraHyperedge) else el[2]
            if prov == "observed" and self.source != "environment":
                raise InvariantViolation("an element of an Imagination view can never carry the OBSERVED tag (P-01)")


@dataclass(frozen=True)
class ReducedView:
    """A reducibility merge (module docstring): the plane groups, the quality of every level, the merged view."""

    groups: tuple[tuple[str, ...], ...]
    quality: tuple[float, ...]
    view: DecodedView


def _region(source: str) -> Region:
    if source == "environment":
        return Region.ENVIRONMENT
    if source in ("belief", "forecast"):
        return Region.IMAGINATION
    raise InvariantViolation(f"unknown view source {source!r}")


def laplacian_density(adjacency: np.ndarray) -> np.ndarray | None:
    """rho = L / tr(L), L = D - A, for a symmetric non-negative adjacency; None without edges."""
    a = np.asarray(adjacency, dtype=np.float64)
    lap = np.diag(a.sum(1)) - a
    tr = float(np.trace(lap))
    return lap / tr if tr > 0.0 else None


def density_entropy(rho: np.ndarray) -> float:
    """-sum lambda log2 lambda over the eigenvalues of a density matrix (0 log 0 = 0)."""
    lam = np.linalg.eigvalsh(rho)
    lam = lam[lam > 1e-15]
    return float(-(lam * np.log2(lam)).sum())


def von_neumann_entropy(adjacency: np.ndarray) -> float:
    """h(rho) of rho = L / tr(L) for a symmetric non-negative adjacency (0 without edges)."""
    rho = laplacian_density(adjacency)
    return 0.0 if rho is None else density_entropy(rho)


def _plane_adjacency(view: DecodedView, plane: str, index: Mapping[int, int]) -> np.ndarray:
    """Clique expansion of the plane's hyperedge elements over the view's entities (weights: element values)."""
    n = len(index)
    a = np.zeros((n, n), dtype=np.float64)
    for el in view.elements:
        if el.kind != "hyperedge" or el.plane != plane:
            continue
        members = [index[e] for e in el.entities]
        for i in members:
            for j in members:
                if i != j:
                    a[i, j] += float(el.value)
    return a


def reducibility_merge(view: DecodedView) -> ReducedView:
    """The reducibility merge of a typed-multigraph view (module docstring)."""
    from scipy.cluster.hierarchy import linkage

    planes = list(dict.fromkeys(el.plane for el in view.elements if el.kind == "hyperedge" and el.plane is not None))
    entities = sorted({e for el in view.elements for e in el.entities})
    index = {e: i for i, e in enumerate(entities)}
    adj = {p: _plane_adjacency(view, p, index) for p in planes}
    active = [p for p in planes if float(adj[p].sum()) > 0.0]
    if len(active) < 2:
        return ReducedView(groups=tuple((p,) for p in planes), quality=(), view=view)
    h = {p: von_neumann_entropy(adj[p]) for p in active}
    m = len(active)
    dist = np.zeros(m * (m - 1) // 2)
    k = 0
    for i in range(m):
        for j in range(i + 1, m):
            # Jensen-Shannon divergence of the two densities: the mixture's entropy minus the mean entropy.
            ra, rb = laplacian_density(adj[active[i]]), laplacian_density(adj[active[j]])
            assert ra is not None and rb is not None                   # active planes have edges
            js = density_entropy(0.5 * (ra + rb)) - 0.5 * (h[active[i]] + h[active[j]])
            dist[k] = math.sqrt(max(js, 0.0))
            k += 1
    tree = linkage(dist, method="ward")
    total = sum(adj[p] for p in active)
    h_all = von_neumann_entropy(total)
    clusters: dict[int, tuple[str, ...]] = {i: (active[i],) for i in range(m)}
    levels: list[tuple[tuple[str, ...], ...]] = [tuple(clusters.values())]
    for step, row in enumerate(tree):
        a_id, b_id = int(row[0]), int(row[1])
        clusters[m + step] = clusters.pop(a_id) + clusters.pop(b_id)
        levels.append(tuple(clusters.values()))

    def quality(groups: tuple[tuple[str, ...], ...]) -> float:
        # q = 1 - mean entropy of the level's planes / entropy of the aggregate (De Domenico et al. 2015).
        if h_all <= 0.0:
            return 0.0
        mean_h = float(np.mean([von_neumann_entropy(sum(adj[p] for p in g)) for g in groups]))
        return 1.0 - mean_h / h_all

    q = tuple(quality(g) for g in levels)
    best = levels[int(np.argmax(q))]
    rename = {p: "+".join(g) for g in best for p in g}
    merged: dict[tuple[str, tuple[int, ...], str], ViewElement] = {}
    others: list[ViewElement] = []
    rank = {"forecast": 0, "believed": 1, "observed": 2}
    for el in view.elements:
        if el.kind != "hyperedge" or el.plane not in rename:
            others.append(el)
            continue
        key = (rename[el.plane], tuple(sorted(el.entities)), el.name)
        new = ViewElement("hyperedge", rename[el.plane], tuple(sorted(el.entities)), el.name, float(el.value), el.provenance)
        old = merged.get(key)
        if old is not None:
            prov = old.provenance if rank[old.provenance] >= rank[new.provenance] else new.provenance
            new = ViewElement("hyperedge", new.plane, new.entities, new.name, max(old.value, new.value), prov)
        merged[key] = new
    groups = (*best, *((p,) for p in planes if p not in active))
    out = DecodedView(source=view.source, latent_space=view.latent_space, elements=(*others, *merged.values()))
    return ReducedView(groups=tuple(groups), quality=q, view=out)


def supra_graph(view: DecodedView) -> SupraGraph:
    """The supra-graph of a view (module docstring)."""
    planes = tuple(dict.fromkeys(el.plane for el in view.elements if el.kind == "hyperedge" and el.plane is not None))
    nodes: dict[tuple[int, str], int] = {}
    hyper: list[SupraHyperedge] = []
    observed_on: dict[int, set[str]] = {}
    for el in view.elements:
        if el.kind != "hyperedge" or el.plane is None:
            continue
        ids = []
        for e in el.entities:
            key = (e, el.plane)
            if key not in nodes:
                nodes[key] = len(nodes)
            ids.append(nodes[key])
            if el.provenance == "observed":
                observed_on.setdefault(e, set()).add(el.plane)
        hyper.append(SupraHyperedge(tuple(ids), el.plane, el.name, float(el.value), el.provenance))
    by_entity: dict[int, list[tuple[str, int]]] = {}
    for (e, p), i in nodes.items():
        by_entity.setdefault(e, []).append((p, i))
    coupling: list[tuple[int, int, str]] = []
    for e, reps in by_entity.items():
        reps.sort()
        for a in range(len(reps)):
            for b in range(a + 1, len(reps)):
                pa, pb = reps[a][0], reps[b][0]
                if view.source == "forecast":
                    prov = "forecast"
                elif view.source == "environment" and {pa, pb} <= observed_on.get(e, set()):
                    prov = "observed"
                else:
                    prov = "believed"
                coupling.append((reps[a][1], reps[b][1], prov))
    fields = tuple(el for el in view.elements if el.kind == "field")
    node_list = tuple(sorted(nodes, key=lambda k: nodes[k]))
    return SupraGraph(source=view.source, latent_space=view.latent_space, planes=planes, nodes=node_list,
                      hyperedges=tuple(hyper), coupling=tuple(coupling), fields=fields)


def flatten(view: DecodedView, option: str) -> DecodedView | SupraGraph | ReducedView:
    """Flatten a typed-multigraph view by a D-05 option (module docstring)."""
    if option not in FLATTENINGS:
        raise InvariantViolation(f"unknown flattening {option!r}; admissible: {FLATTENINGS}")
    if option == "typed multigraph":
        return view
    if option == "supra-graph":
        return supra_graph(view)
    reduced = reducibility_merge(view)
    if option == "reducibility merge":
        return reduced
    return supra_graph(reduced.view)


class DecoderView:
    """The Decoder role over the working Decoder (module docstring).

    Parameters
    ----------
    decoder: the model's Decoder.
    enabled_proposals: proposals enabled for the run (P-12 grants the Monitor read).
    """

    def __init__(self, decoder: Decoder, *, enabled_proposals: Collection[str] = ()) -> None:
        self.decoder = decoder
        self.enabled_proposals = tuple(enabled_proposals)

    def render(
        self,
        latent: LatentState,
        *,
        source: str,
        entity: torch.Tensor,
        role: torch.Tensor,
        planes: torch.Tensor,
        observed_values: torch.Tensor | None = None,
        observed_status: torch.Tensor | None = None,
        candidate_edges: Mapping[str, torch.Tensor] | None = None,
        candidate_kinds: Mapping[str, torch.Tensor] | None = None,
        observed_edges: Mapping[str, torch.Tensor] | None = None,
        edge_threshold: float = 0.5,
        flattening: str | None = None,
    ) -> DecodedView | SupraGraph | ReducedView:
        """The provenance-tagged view of `latent`, flattened by the D-05 option in force (or `flattening`).

        The arguments are those of `Decoder.view`; the role checks its read right on the memory region of the
        source (Environment for "environment", Imagination for "belief" and "forecast") first.
        """
        check(Role.DECODER, _region(source), Op.READ, self.enabled_proposals)
        view = self.decoder.view(latent, source=source, entity=entity, role=role, planes=planes,
                                 observed_values=observed_values, observed_status=observed_status,
                                 candidate_edges=candidate_edges, candidate_kinds=candidate_kinds,
                                 observed_edges=observed_edges, edge_threshold=edge_threshold)
        option = decisions.require("decoder-view-flattening", flattening, by=__name__).value
        assert option is not None
        return flatten(view, option)
