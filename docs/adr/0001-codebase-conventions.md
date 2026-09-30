# ADR-0001: Codebase conventions

- **Status:** Proposed (engineering, 2026-09-29). The owner may change any of these.
- **Decision IDs:** D-09 (stack), D-10 (location, held)
- **Sources:** [Q-39], [Q-45]

## Context
The owner asked for "highly decoupled, transformable further, modular … fully templated so that we
can keep updating with more and more decisions and logic on the go; heavily comment it" [Q-45], and
"please don't just default to any without my permission" [Q-39]. Some engineering conventions are
unavoidable in order to start at all. They are listed here so they can be reviewed as a set.

## Decision (proposed)
- `src/` layout, package `nagahana`, Python ≥ 3.11, hatchling build.
- Governance as code: `governance/decisions.py`, with `require()` / `require_proposal()` gates and a
  generated report.
- Component registries, so implementations are swapped by config name (`core/registry.py`).
- Hydra-format YAML with `???` for every undecided value (readable with PyYAML until Hydra is
  installed).
- Optional-dependency extras per stack area; no version pins until the first environment lock
  (`uv lock` or a pip-tools file) on the machine that trains.
- Lint and types: ruff (E, F, I, UP, B, SIM) and mypy. Tests: pytest + hypothesis.
- Documentation standard: each module docstring states purpose, owner sources (Q-/A-/I- IDs),
  decisions, maths, invariants and extension points.
- Location: `nagahana/` inside this workspace (D-10 is still held; moving is a copy or push).

## Consequences
- Every design choice is visible and reversible in one place.
- Heavy dependencies never block the core tests.
- Pins must be added before the first real training run, for reproducibility (with DVC and MLflow).
