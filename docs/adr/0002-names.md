# ADR-0002: Role and component names

- **Status:** Accepted (owner, 2026-09-29)
- **Decision IDs:** D-19, D-20
- **Sources:** [A-10], [A-11]

## Decision
- planner → **Advisor**
- renderer → **Forecaster**
- spatio-temporal-causal transformer → **TSTCT** (Topological Spatio-Temporal Causal Transformer)
- energy-based transformer core → **TAAFT** (Topological Anti-Adversary Foundation Transformer)
- the encoder is **CVG-AE** (Complex Variational Graph AutoEncoder); it is variational

## Consequences
`core/roles.py` accepts the legacy names with a DeprecationWarning, so older notes still resolve.
Every new artefact uses the new names.
