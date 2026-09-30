"""NagaHana: Adversary Foundation Model for Simulation-based Threat Forecasting (research codebase).

Start with docs/architecture.md (the design), docs/sources/quotes.md (the owner's words; private workspace) and
`python -m nagahana decisions` (what is decided, held or proposed, and where code depends on it).

Package map
-----------
- core: errors, registries, roles, run modes, config, introspection hooks
- governance: the decision registry and its report
- datamodel: observation status, field catalogue, state updates, schema versions, layers
- ingest: passive, source-agnostic adapters (CSV flows, PCAP, Kafka)
- graph: the heterogeneous multiplex hypergraph and its builder
- physics: the shared physics boundary (residuals, hard limits, Φ_phys)
- memory: Environment / Imagination / Monitor, access rules, KV caches, retention, event log
- models: CVG-AE, TSTCT, Decoder, TAAFT, policy/value heads, Generator
- roles: Simulator, Forecaster, Advisor, Verifier, Decoder view, and their output contracts
- objectives: loss template, proper scores, process rewards
- pipeline: the six training stages, splits, freezing
- evaluation: metrics, calibration, forecasting skill, baseline, catalogue
- tracking: experiment logging
- lab: isolated JAX/Equinox research bench
"""

__version__ = "0.0.1"
