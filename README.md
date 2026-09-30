# NagaHana: research codebase

**Adversary Foundation Model for Simulation-based Threat Forecasting**: an AI world model that learns
the evolving state of a network from passive telemetry, imagines where an intrusion is heading, and
gives defenders calibrated, explainable forecasts and advisory counter-measures, across enterprise and
CII, OT and IT.

Status: **v0.0.1, a template.** The following are real, tested code:
- the structure, contracts and governance;
- the decided invariants;
- the evaluation maths;
- a reference encoder.

Model internals are templates that raise `NotBuiltYet`, naming the decision or stage they wait on.
Nothing here fakes a result.

## Start here
1. `docs/architecture.md`: the current design, as one story.
2. `docs/sources/quotes.md` (private workspace, not published): the owner's own words, which the code cites as `[Q-xx]` / `[A-xx]`.
3. `python -m nagahana decisions`: what is decided, held or proposed, and which code depends on each.

## Three kinds of statement in this code
| Kind | Where recorded | How the code treats it |
|---|---|---|
| **Decided** by the owner | `governance/decisions.py` (DECIDED) | implemented or enforced (e.g. the access matrix, human-gated Verifier updates, real-only zero-shot data) |
| **Held**: the owner has not decided | DECIDED → HELD, config value `???` | `require("…")` raises `DecisionHeld`; nothing is defaulted |
| **Proposal** (engineering), not approved | PROPOSED (P-xx) | runs only when listed in `enabled_proposals` |

## Layout
```
conf/                 Hydra-format configs; every undecided value is ???
docs/                 architecture, glossary, sources, ADRs, generated decisions report
src/nagahana/
  core/               errors, registries, roles (+ legacy names), run modes, config, introspection
  governance/         decision registry and report
  datamodel/          observation status, field catalogue, state updates, versions, layers
  ingest/             passive adapters: CIC/CTU CSV flows, PCAP, Kafka
  graph/              heterogeneous multiplex hypergraph, planes, builder
  physics/            the shared physics boundary Φ_phys
  memory/             Environment / Imagination / Monitor, access rules, KV caches, retention, event log
  models/             CVG-AE, TSTCT, Decoder, TAAFT (energy core + lenses), policy/value heads, Generator
  roles/              Simulator, Forecaster, Advisor, Verifier, Decoder view, output contracts
  objectives/         loss template, proper scores, process rewards
  pipeline/           the six training stages, split rules, freezing proof
  evaluation/         metrics, calibration, forecast skill, C-index, baseline, catalogue
  tracking/           experiment logging (MLflow)
  lab/                isolated JAX/Equinox research bench
tests/                invariants that encode the owner's decisions
```

## Everyday commands
```
python -m pytest                          # tests (needs only torch, numpy, pyyaml, pytest, hypothesis)
python -m ruff check src tests            # lint
python -m mypy                            # types
python -m nagahana decisions              # governance report   (--write-docs updates docs/decisions.md)
python -m nagahana stages                 # the six-stage pipeline
python -m nagahana metrics                # evaluation catalogue
python -m nagahana access                 # memory access matrix
python -m nagahana check-config conf/model/taaft/taaft.yaml   # what is still ???
```
Tests run from the repo root without installing (`pythonpath = src` in pyproject). Install with
`pip install -e ".[dev]"`, and add extras as needed:
`graph`, `rl`, `metrics`, `config`, `track`, `scale`, `jax`, `prob`, `data`, `pcap`, `ui`, `perf`, `docs`.

## How to change things
- **A decision is made.**
  1. Set its entry to DECIDED with the value and source in `governance/decisions.py`.
  2. Replace the matching `???` in `conf/`.
  3. Update `DESIGN_LOG.md`.
  4. Run `python -m nagahana decisions --write-docs`.
- **A new component or variant.**
  1. Implement it.
  2. Register it in its registry with `requires=` and/or `proposal=`.
  3. Select it by name in config.
  4. Add tests for its invariants.
- **A borrowed idea from the literature.** Cite its `refs.md` line, and give it an ablation gate (P-17):
  it must beat its own removal.

## Tech stack
Decision D-09 approves everything below. It is the engineer's reading of "all the tech stack yours and mine
together", pending the owner's confirmation.
- PyTorch family (core, PyG incl. TGN, TorchRL, TorchMetrics), MLflow, DVC, HF Accelerate, DeepSpeed.
- JAX + Flax/Equinox (with Diffrax, Optimistix, jaxtyping, BlackJAX, NumPyro); Pyro; LightZero.
- Hydra; hypothesis; ruff/mypy; MkDocs.
- Arrow/Polars/DuckDB; scipy/statsmodels.
- Triton/FlashAttention; Scapy/PyShark; Streamlit.

The Kafka client is still held (D-27).
