"""TAAFT: Topological Anti-Adversary Foundation Transformer (energy core and lenses; build-spec §2.7).

Modules: `model` (TAAFT, registry `ANALYSERS["taaft"]`), `blocks` (belief-recursion block, cross reads of
TSTCT's cache), `structure` (as-of structure), `noise` (noise features inside E_info), `lenses` (energy
terms, registry `LENSES`), `energy` (descent and per-lens shares), `readouts`, `objectives` (stage 4),
`testing` (contract-shaped fakes for tests).
"""
