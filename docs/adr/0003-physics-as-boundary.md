# ADR-0003: Physics is a boundary against hallucination, used everywhere

- **Status:** Accepted (owner, 2026-09-29)
- **Decision IDs:** D-18, D-37; held: D-25
- **Sources:** [A-08], [Q-05], [Q-41]

## Context
An earlier framing treated physics violations as a detection signal ("impossible: strong signal").
The owner corrected it: "physics informed acts a boundary/guide to not hallucinate … the same way we
can govern the model's understanding to be limited within the possibility boundary" [A-08]. Physics
is also global: one term across all losses [Q-41].

## Decision
One shared term, Φ_phys(x) = Σ_c w_c ‖m_c ⊙ r_c(x)‖² with x ∈ 𝒞. It is applied to **model outputs**:
reconstructions, imagined steps, generated variants and predicted effects of counters. Residuals
count only where their fields are observed. Hard limits are built into outputs where exact. The
Decoder has none of its own.

## Consequences
- `physics/term.py` accepts `Target.MODEL_OUTPUT`. `Target.OBSERVATION` (scoring incoming telemetry)
  raises until D-25 is decided.
- The capability-envelope wording in older documents ("violates physics: strong signal") is
  superseded.
