# ADR-0004: The Verifier changes the model only on a human command

- **Status:** Accepted (owner, 2026-09-29)
- **Decision IDs:** D-07, D-21
- **Sources:** [A-16], [A-21], [Q-14], [Q-38]

## Decision
The Verifier stores human feedback as supplied truth. It computes calibration and memory drift from
outcome–forecast pairs, and it observes online. It never feeds back automatically ("it would conduct
the poisoning again", [A-16]). It adjusts weights only on a human's supervised command [A-21].
Responded-to outcomes are not penalised [Q-38].

## Consequences
- `roles/verifier.py`: `apply_update(fn, command)` is the only model-changing path, and it raises
  `HumanCommandRequired` without a `HumanCommand`. Every command is logged (audit trail).
- Calibration summaries exclude responded-to pairs.
- Open: the RLCD mechanism (Brier reward is proposal P-10) and site adapters (D-13).
