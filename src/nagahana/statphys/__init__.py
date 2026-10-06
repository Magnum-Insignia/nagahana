"""Statistical physics of the network state: thermodynamic readouts, network entropies, early warning (D-56).

The energy view of NagaHana (TAAFT's E_total, D-42) is read as a thermodynamic trajectory, and the
traffic and the multiplex graph of the network state as information-theoretic ones; their
trajectories carry early-warning signs of an approaching transition (critical slowing down).

Modules (import them directly; this package imports nothing at load time, because TAAFT's readouts
import `thermo` and must not pull in the data model or the inference stack):
    config      configuration dataclasses, the single source of truth of conf/statphys/statphys.yaml
    thermo      Gibbs ensembles: log Z, F, U, S, C, susceptibilities, growth rates
    entropy     Shannon entropies of traffic distributions per entity and window, batch and streaming
    spectral    von Neumann and spectral entropies of graphs, quantum Jensen-Shannon divergence,
                multiplex reducibility; exact and stochastic Lanczos quadrature
    graphs      multiplex graph states of the network at a trigger (activity and contact views)
    ews         critical-slowing-down indicators, Kendall trends, surrogate tests, composite alarm
    trajectory  per-trigger readings (StatPhysTracker for the inference engine) and the batch series
    evaluation  the statphys.* names of ModelOutputs.component and the early-warning scores
    io          saved trajectories for offline analysis
    cli         `python -m nagahana statphys ...` (register_cli)
Documentation: docs/statphys.md; assumptions AS-760 to AS-779 (docs/assumptions/statphys.md).
"""
