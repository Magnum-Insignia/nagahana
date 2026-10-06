"""Spatial (graph) analytics of traffic: centralities, communities, mixing, motifs, multiplex planes, drift.

Modules
-------
    graph        TrafficGraph: directed weighted multiplex edges from arrays or window contact matrices
    centrality   degree distributions, betweenness (algebraic Brandes), PageRank, eigenvector, k-core
    community    modularity, Louvain with connectivity repair, ARI and NMI
    structure    assortativity, triangles and clustering, the directed triad census, edge-switch z-scores
    multiplex    edge overlap, participation, interlayer degree correlation, von Neumann entropy,
                 structural reducibility of the relation planes
    spectral     normalised-Laplacian spectra (exact or Lanczos quadrature), spectral and heat-trace
                 distances, structural drift between windows
    analysis     reports of one graph and of the per-network, per-window graphs of a corpus
"""

from nagahana.analytics.spatial.analysis import corpus_graphs, corpus_report, graph_measures, graph_report
from nagahana.analytics.spatial.centrality import betweenness, core_numbers, degree_stats, eigenvector_centrality, pagerank
from nagahana.analytics.spatial.community import (
    Communities,
    adjusted_rand_index,
    louvain,
    modularity,
    normalized_mutual_information,
)
from nagahana.analytics.spatial.graph import TrafficGraph
from nagahana.analytics.spatial.multiplex import (
    edge_overlap,
    interlayer_degree_correlation,
    layer_adjacencies,
    participation,
    reducibility,
    von_neumann_entropy,
)
from nagahana.analytics.spatial.spectral import heat_trace, normalized_laplacian, spectrum, structural_drift
from nagahana.analytics.spatial.structure import (
    attribute_assortativity,
    clustering,
    degree_assortativity,
    motif_zscores,
    triad_census,
    triangles,
)

__all__ = [
    "Communities", "TrafficGraph", "adjusted_rand_index", "attribute_assortativity", "betweenness", "clustering",
    "core_numbers", "corpus_graphs", "corpus_report", "degree_assortativity", "degree_stats", "edge_overlap",
    "eigenvector_centrality", "graph_measures", "graph_report", "heat_trace", "interlayer_degree_correlation",
    "layer_adjacencies", "louvain", "modularity", "motif_zscores", "normalized_laplacian", "normalized_mutual_information",
    "pagerank", "participation", "reducibility", "spectrum", "structural_drift", "triad_census", "triangles",
    "von_neumann_entropy",
]
