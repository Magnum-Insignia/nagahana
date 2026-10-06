"""Spatial (graph) reports: one traffic graph, the graphs of a corpus and their structural drift.

Traffic graphs of a corpus
--------------------------
Nodes are corpus entities, labelled by their stable identity "network|kind|key" (D-48), so graphs of
different windows align. Every state update with an initiator and a responder adds the directed edge
initiator -> responder with weight 1 (an update count) on each relation plane the update belongs to;
planes follow the declared rules of AS-01 (`graph.planes.declared_update_planes`: connectivity for every
flow, services through a service entity, identity, remote_admin, name_resolution and ot_control by
destination port and transport). An update whose port or protocol is absent (D-41) matches no
port-ruled plane, and is still on connectivity. Graphs are built per network (entities of different
networks are different machines) and per window (the corpus window records, or consecutive slices of
`window_seconds`).

Measures of one graph
---------------------
Degree distributions (in, out, total; CCDF and Hill tail index), betweenness (exact, or sampled
sources), PageRank, eigenvector centrality, k-core numbers, Louvain communities on the symmetrised
graph, degree and kind assortativity, triangles and clustering, the directed triad census (with
edge-switch z-scores when `motif_null_samples` > 0), and the multiplex measures of the planes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from nagahana.analytics.config import SpatialConfig, to_dict
from nagahana.analytics.report import Figure, Report
from nagahana.analytics.spatial import centrality as cen
from nagahana.analytics.spatial import community as com
from nagahana.analytics.spatial import multiplex as mpx
from nagahana.analytics.spatial import spectral as spc
from nagahana.analytics.spatial import structure as stc
from nagahana.analytics.spatial.graph import TrafficGraph

if TYPE_CHECKING:                                                        # the data model is needed by corpus inputs only
    from nagahana.analytics.corpus import Corpus


def graph_measures(g: TrafficGraph, cfg: SpatialConfig) -> tuple[dict[str, object], dict[str, pd.DataFrame]]:
    """(summary, tables) of every measure of one graph (module docstring)."""
    directed = g.adjacency(weighted=False, symmetric=False) if g.directed else g.adjacency(weighted=False, symmetric=True)
    sym_w = g.adjacency(weighted=True, symmetric=True)
    sym = g.simple_undirected()
    n = g.n
    out_deg = np.asarray(directed.sum(axis=1)).ravel()
    in_deg = np.asarray(directed.sum(axis=0)).ravel()
    tot_deg = np.asarray(sym.sum(axis=1)).ravel()
    strength = np.asarray(sym_w.sum(axis=1)).ravel()
    summary: dict[str, object] = {"nodes": n, "edges": g.m, "self_loops_dropped": g.self_loops_dropped,
                                  "density": float(sym.nnz / (n * (n - 1))) if n > 1 else float("nan")}
    tables: dict[str, pd.DataFrame] = {}
    deg_rows = []
    ccdf_rows = []
    for name, seq in (("out", out_deg), ("in", in_deg), ("total", tot_deg)):
        ds = cen.degree_stats(seq)
        deg_rows.append({"degree": name, **{k: v for k, v in ds.summary.items() if k in ("mean", "std", "median", "q099",
                                                                                         "q100", "n_distinct")},
                         "tail_alpha": ds.tail_alpha, "tail_k_star": ds.tail_k_star})
        ccdf_rows.append(pd.DataFrame({"degree": name, "k": ds.ccdf_k, "ccdf": ds.ccdf_p}))
    tables["degree_summary"] = pd.DataFrame(deg_rows)
    tables["degree_ccdf"] = pd.concat(ccdf_rows, ignore_index=True)
    # Centralities.
    bc = cen.betweenness(sym_w if cfg.weighted else sym, weighted=cfg.weighted, undirected=True,
                         sources=cfg.betweenness_sources, batch=cfg.betweenness_batch, seed=cfg.seed)
    pr, pr_it = cen.pagerank(directed if g.directed else sym, alpha=cfg.pagerank_alpha, tol=cfg.pagerank_tol,
                             max_iter=cfg.pagerank_max_iter)
    ev, ev_it = cen.eigenvector_centrality(sym, tol=cfg.eigen_tol, max_iter=cfg.eigen_max_iter)
    core = cen.core_numbers(sym)
    comm = com.louvain(sym_w, resolution=cfg.louvain_resolution, tol=cfg.louvain_tol, max_levels=cfg.louvain_max_levels,
                       max_sweeps=cfg.louvain_max_sweeps, seed=cfg.seed)
    tri, local, avg_clust, trans = stc.clustering(sym)
    nodes = pd.DataFrame({
        "node": np.arange(n), "label": g.node_label if g.node_label is not None else np.arange(n).astype(str),
        "kind": g.node_kind if g.node_kind is not None else "", "out_degree": out_deg, "in_degree": in_deg,
        "degree": tot_deg, "strength": strength, "betweenness": bc, "pagerank": pr, "eigenvector": ev, "core": core,
        "community": comm.membership, "triangles": tri, "clustering": local,
    })
    tables["nodes"] = nodes
    summary.update({
        "pagerank_iterations": pr_it, "eigenvector_iterations": ev_it, "max_core": int(core.max(initial=0)),
        "communities": comm.n_communities, "modularity": comm.modularity, "community_repairs": comm.repaired_splits,
        "triangles": int(tri.sum() // 3), "average_clustering": avg_clust, "transitivity": trans,
        "degree_assortativity": stc.degree_assortativity(sym),
        "degree_assortativity_out_in": stc.degree_assortativity(directed, directed=True, mode="out-in") if g.directed
        else float("nan"),
        "kind_assortativity": stc.attribute_assortativity(sym, g.node_kind) if g.node_kind is not None else float("nan"),
        "betweenness_sampled_sources": cfg.betweenness_sources if cfg.betweenness_sources is not None else 0,
    })
    sizes = np.bincount(comm.membership) if n else np.zeros(0, dtype=np.int64)
    tables["communities"] = pd.DataFrame({"community": np.arange(sizes.size), "size": sizes})
    # Motifs.
    if g.directed:
        if cfg.motif_null_samples > 0:
            z = stc.motif_zscores(directed, samples=cfg.motif_null_samples, seed=cfg.seed)
            tables["triad_census"] = pd.DataFrame([{"triad": k, "count": v[0], "random_mean": v[1], "random_sd": v[2],
                                                     "z": v[3]} for k, v in z.items()])
        else:
            tc = stc.triad_census(directed)
            tables["triad_census"] = pd.DataFrame({"triad": list(tc), "count": list(tc.values())})
    um = stc.undirected_motifs(sym)
    summary.update({f"motif_{k}": v for k, v in um.items()})
    # Multiplex measures of the planes.
    if g.plane is not None and len(g.planes()) >= 1:
        layers = mpx.layer_adjacencies(g)
        overlap, pairs = mpx.edge_overlap(layers)
        part, odeg = mpx.participation(layers)
        red = mpx.reducibility(layers, max_dense=cfg.max_dense_nodes, vectors=cfg.slq_vectors, steps=cfg.slq_steps,
                               seed=cfg.seed)
        summary.update({"planes": g.planes(), "edge_overlap": overlap,
                        "mean_participation": float(np.nanmean(part)) if np.isfinite(part).any() else float("nan"),
                        "reducibility_best_layers": red.best_layers})
        tables["plane_pairs"] = pairs.merge(mpx.interlayer_degree_correlation(layers), on=["layer_a", "layer_b"], how="left")
        tables["plane_entropy"] = pd.DataFrame({"plane": list(layers), "edges": [int(a.nnz // 2) for a in layers.values()],
                                                "von_neumann_entropy_bits": [mpx.von_neumann_entropy(
                                                    a, max_dense=cfg.max_dense_nodes, vectors=cfg.slq_vectors,
                                                    steps=cfg.slq_steps, seed=cfg.seed) for a in layers.values()]})
        tables["reducibility"] = red.steps.assign(groups=red.steps["groups"].astype(str))
        nodes["participation"] = part
    return summary, tables


def graph_report(g: TrafficGraph, cfg: SpatialConfig | None = None, *, title: str = "Traffic graph") -> Report:
    """Report of one graph (module docstring)."""
    cfg = cfg or SpatialConfig()
    summ, tabs = graph_measures(g, cfg)
    rep = Report(kind="spatial_graph", title=title, summary=summ, provenance={"config": to_dict(cfg)})
    for name, frame in tabs.items():
        rep.add_table(name, frame, title=name.replace("_", " "))
    rep.add_figure(Figure(name="degree_ccdf", kind="step", data=tabs["degree_ccdf"], x="k", y="ccdf", series="degree",
                          title="Degree CCDF", x_label="degree k", y_label="P(K >= k)", log_x=True, log_y=True))
    return rep


def corpus_graphs(corpus: Corpus, cfg: SpatialConfig) -> list[tuple[str, str, TrafficGraph]]:
    """(network, window id, graph) for every network and window of the corpus (module docstring)."""
    from nagahana.graph.planes import declared_update_planes
    from nagahana.models.vocab import PLANES

    ents = corpus.entities
    networks = corpus.row_attr("network")
    dport = corpus.values[:, corpus.column_index("flow.dst_port")] if corpus.has_column("flow.dst_port") else np.full(len(corpus), np.nan)
    proto = corpus.values[:, corpus.column_index("flow.protocol")] if corpus.has_column("flow.protocol") else np.full(len(corpus), np.nan)
    planes = declared_update_planes(PLANES, dst_port=dport, protocol=proto, has_service=ents[:, 2] >= 0)   # [n, P]
    if (corpus.window != "").any():
        win = corpus.window.astype(object)
    else:
        t = corpus.time
        base = np.nanmin(t) if np.isfinite(t).any() else 0.0
        b = np.where(np.isfinite(t), np.floor((t - base) / cfg.window_seconds), -1).astype(np.int64)
        win = np.array([f"bin{k}" if k >= 0 else "untimed" for k in b], dtype=object)
    labels_all = np.array([f"{nw}|{k}|{key}" for nw, k, key in zip(corpus.entity_network, corpus.entity_kind,
                                                                    corpus.entity_key, strict=True)], dtype=object)
    out = []
    frame = pd.DataFrame({"net": networks, "win": win, "t": np.nan_to_num(corpus.time, nan=0.0)})
    order = frame.groupby(["net", "win"])["t"].min().sort_values(kind="stable").index
    groups = frame.groupby(["net", "win"]).indices
    for net, w in order:
        rows = np.asarray(groups[(net, w)])
        e = ents[rows, :2]
        ok = (e[:, 0] >= 0) & (e[:, 1] >= 0)
        rows, e = rows[ok], e[ok]
        if rows.size == 0:
            continue
        uniq, inv = np.unique(e, return_inverse=True)
        inv = inv.reshape(-1, 2)
        pl = planes[rows]
        r_idx, p_idx = np.nonzero(pl)
        g = TrafficGraph.from_edges(inv[r_idx, 0], inv[r_idx, 1], n=uniq.size,
                                    plane=np.array(PLANES, dtype=object)[p_idx], node_label=labels_all[uniq],
                                    node_kind=corpus.entity_kind[uniq], directed=True)
        out.append((str(net), str(w), g))
    return out


def corpus_report(corpus: Corpus, cfg: SpatialConfig | None = None) -> Report:
    """Spatial report of a corpus: the whole-network graphs, per-window summaries and structural drift."""
    cfg = cfg or SpatialConfig()
    rep = Report(kind="spatial", title="Spatial (graph) analytics", provenance={"config": to_dict(cfg)})
    graphs = corpus_graphs(corpus, cfg)
    rep.summary["networks"] = sorted({net for net, _, _ in graphs})
    rep.summary["windows"] = len(graphs)
    # Whole-network graphs: union of the window graphs of each network (aligned by node labels).
    for net in rep.summary["networks"]:
        parts = [g for nw, _, g in graphs if nw == net]
        labels = np.unique(np.concatenate([g.node_label for g in parts if g.node_label is not None]))
        pos = {lab: i for i, lab in enumerate(labels.tolist())}
        kinds = np.empty(labels.size, dtype=object)
        src, dst, weights, pl = [], [], [], []
        for g in parts:
            assert g.node_label is not None and g.node_kind is not None and g.plane is not None
            m = np.array([pos[x] for x in g.node_label.tolist()], dtype=np.int64)
            kinds[m] = g.node_kind
            src.append(m[g.src])
            dst.append(m[g.dst])
            weights.append(g.weight)
            pl.append(g.plane)
        whole = TrafficGraph.from_edges(np.concatenate(src), np.concatenate(dst), weight=np.concatenate(weights), n=labels.size,
                                        plane=np.concatenate(pl), node_label=labels, node_kind=kinds, directed=True)
        rep.merge(graph_report(whole, cfg, title=f"Traffic graph of {net}"), prefix=f"network.{net}")
    # Per-window summaries and structural drift.
    rows: list[dict[str, object]] = []
    for net, w, g in graphs:
        sym = g.simple_undirected()
        comm = com.louvain(g.adjacency(weighted=True, symmetric=True), resolution=cfg.louvain_resolution,
                           tol=cfg.louvain_tol, max_levels=cfg.louvain_max_levels, max_sweeps=cfg.louvain_max_sweeps,
                           seed=cfg.seed)
        _, _, avg_c, trans = stc.clustering(sym)
        core = cen.core_numbers(sym)
        rows.append({"network": net, "window": w, "nodes": g.n, "edges": g.m,
                     "density": float(sym.nnz / (g.n * (g.n - 1))) if g.n > 1 else float("nan"),
                     "max_core": int(core.max(initial=0)), "communities": comm.n_communities, "modularity": comm.modularity,
                     "transitivity": trans, "average_clustering": avg_c,
                     "max_out_degree": float(np.asarray((g.adjacency(weighted=False) > 0).sum(axis=1)).max(initial=0))})
    windows = pd.DataFrame(rows)
    rep.add_table("windows", windows, title="Graph summary per window")
    drift_parts = []
    for net in rep.summary["networks"]:
        seq = [(w, g) for nw, w, g in graphs if nw == net]
        if len(seq) >= 2:
            d = spc.structural_drift([g for _, g in seq], [w for w, _ in seq], k=cfg.spectral_k,
                                     times=np.asarray(cfg.heat_times), max_dense=cfg.max_dense_nodes,
                                     vectors=cfg.slq_vectors, steps=cfg.slq_steps, seed=cfg.seed)
            drift_parts.append(d.assign(network=net, step=np.arange(len(d))))
    drift = pd.concat(drift_parts, ignore_index=True) if drift_parts else pd.DataFrame(
        columns=["window", "previous", "nodes", "edges", "spectral_l2", "spectral_w1", "heat", "edge_jaccard_distance",
                 "exact_spectra", "network", "step"])
    rep.add_table("structural_drift", drift, title="Structural drift between consecutive windows",
                  description="Spectral L2 and W1 distances of normalised-Laplacian spectra, heat-trace distance, "
                              "edge-set Jaccard distance over stable entity labels.")
    if len(drift):
        rep.add_figure(Figure(name="structural_drift", kind="line", data=drift, x="step", y="spectral_w1", series="network",
                              title="Spectral drift between consecutive windows"))
    return rep


__all__ = ["corpus_graphs", "corpus_report", "graph_measures", "graph_report"]
