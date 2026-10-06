"""Topological data analysis implemented in-house: filtrations, persistent homology over Z/2, diagram
distances, vectorisations and sliding-window embeddings.

Modules
-------
    complex      flag filtrations: Vietoris-Rips (sparse neighbourhoods, maximum radius, enclosing radius),
                 weighted-clique and contact filtrations of graphs, maxmin landmarks
    homology     H_0 by union-find; H_k by boundary reduction with clearing (twist) or by coboundary
                 reduction (cohomology), apparent pairs, representatives
    diagrams     persistence diagrams, exact bottleneck and p-Wasserstein distances, summaries
    vectorize    Betti and Euler curves, persistence landscapes, persistence images
    embedding    sliding-window (Takens) embeddings, delay and dimension selection, periodicity score
    analysis     reports for point clouds, series and the contact topology of traffic windows

Closed forms checked by the tests: a noisy circle has one dominant H_1 bar, a sampled 2-sphere one
dominant H_2 bar, k separated clusters k long H_0 bars, the unit square's H_1 is the single bar
[1, sqrt(2)), the octahedron's H_2 the single bar [sqrt(2), 2); the Euler characteristic of the
complex equals the alternating sum of Betti numbers at every scale; homology and cohomology, with and
without clearing and apparent pairs, give identical diagrams; both distances are metrics and agree
with brute-force matching.
"""

from nagahana.analytics.tda.analysis import (
    choose_radius,
    corpus_cloud,
    corpus_report,
    diagram_frame,
    point_cloud_report,
    series_report,
    window_topology,
)
from nagahana.analytics.tda.complex import (
    FilteredComplex,
    contact_complex,
    enclosing_radius,
    flag_complex,
    maxmin_landmarks,
    rips_complex,
    rips_from_distances,
    weighted_clique_complex,
)
from nagahana.analytics.tda.diagrams import Diagram, bottleneck_distance, distance_matrix, summary, wasserstein_distance
from nagahana.analytics.tda.embedding import (
    delay_by_ami,
    dimension_by_fnn,
    periodicity_score,
    sliding_window,
)
from nagahana.analytics.tda.homology import Persistence, persistence
from nagahana.analytics.tda.vectorize import betti_curve, euler_curve, landscape, persistence_image

__all__ = [
    "Diagram", "FilteredComplex", "Persistence", "betti_curve", "bottleneck_distance", "choose_radius", "contact_complex",
    "corpus_cloud", "corpus_report", "delay_by_ami", "diagram_frame", "dimension_by_fnn", "distance_matrix",
    "enclosing_radius", "euler_curve", "flag_complex", "landscape", "maxmin_landmarks", "periodicity_score",
    "persistence", "persistence_image", "point_cloud_report", "rips_complex", "rips_from_distances", "series_report",
    "sliding_window", "summary", "wasserstein_distance", "weighted_clique_complex", "window_topology",
]
