# NagaHana — Adversary Foundation Model

> **⚠️ Model Weights Notice**
> The trained NagaHana (L) weights file is **4.54 GB (fp32)** — too large for GitHub to host.
> Weights are distributed separately. This repository contains the full source, configuration, evaluation framework, and documentation only.

---

**SIH Team ID: 155021**

---

## What NagaHana Is

**NagaHana — Adversary Foundation Model for Simulation-based Threat Forecasting.**
It is an AI world model that learns the evolving state of a computer network from passive telemetry.
It imagines where an intrusion is heading before the kill chain completes, and gives defenders interpretable, calibrated forecasts and advisory counter-measures.

- **Scope:** Enterprise and Critical Information Infrastructure (CII), OT and IT together, with CII as the priority.
- **Posture:** Passive only (taps, NICs, integrations — no host agents). Advisory only (humans decide and act). Fully offline.
- **Two settings of one engine:** Live forecasting, and forensic replay of an uploaded capture.
- **Why "adversary foundation model":** It models conflict, coordination, cooperation, adversarial strategies, patterns, intentions, and the semantics of malignity and benignity.

---

## Repository Layout

```
nagahana/
├── src/nagahana/        # The model and all supporting packages
├── conf/                # YAML configuration files (generated from dataclasses)
├── docs/                # Architecture, sizing, decisions, ADRs, assumptions, glossary
├── scripts/             # Utility scripts (smoke test, etc.)
├── tools/               # Developer tooling (assumption generator, etc.)
├── .github/             # CI workflow
└── pyproject.toml       # Package metadata and dependencies
```

---

## Folder Contents

### `src/nagahana/` — Source packages

| Package | What it contains |
|---|---|
| `analytics/` | Data-quality, drift, leakage, duplication, observability, and EDA analysis tools used before and during training |
| `baselines/` | Logistic-regression and published third-party baseline families used in evaluation comparisons |
| `core/` | Package-wide config, error types, introspection, run-mode declarations, and the role/component registry |
| `data/` | Dataset loading, windowing, stream planning, label handling, and split-assignment utilities |
| `datamodel/` | The superset data model (OCSF/CSTS-compatible): columnar storage, field specs, observation status, versioning |
| `evaluation/` | Full evaluation framework: metrics, calibration, forecasting scoring, survival analysis, ranking, significance, multi-dataset and leave-one-network-out protocols |
| `explain/` | Field attribution (Expected Gradients / SHAP, LIME) reaching back to the input layer |
| `governance/` | Machine-readable decisions (D-xx), assumptions (AS-xx), and assumption types; audit and reporting |
| `graph/` | Heterogeneous multiplex hypergraph builder, plane definitions, hyperedge kinds, local subgraphs, and RWSE |
| `inference/` | Live inference engine, forensic replay engine, streaming buffer, and inline explanation |
| `ingest/` | Source adapters: pcap, NetFlow/sFlow/IPFIX, Zeek, Suricata, Snort, CEF/LEEF, syslog, Wireshark, Windows event logs, dev telemetry, and Kafka |
| `lab/` | Model sizing, compute profiling, information audit, and world-simulator lab entry point |
| `memory/` | Environment store (log-time bucketed KV cache), Imagination store, long-term Titans-style memory, event log, retention schedule, and access-control enforcement |
| `models/` | Top-level model assembly (`nagahana.py`), latent space, vocabulary, batch contracts, and variational utilities |
| `models/advisor/` | Advisor policy/value network — searches D3FEND counter-measure sequences and ranks them by risk reduction |
| `models/config/` | Typed configuration dataclasses for every component (the single source of truth for all shapes) |
| `models/cvgae/` | CVG-AE (Complex Variational Graph AutoEncoder) — the heterogeneous multi-plane hypergraph encoder |
| `models/decoder/` | Decoder — decodes latents back to the readable network graph with observed/believed/forecast provenance |
| `models/forecaster/` | Forecaster — policy/value network, MPC-guided imagination, infiltration-curve computation |
| `models/generator/` | Generator (training only) — observability sliding, signature variation, masked-generative and diffusion augmentation, energy-acceptance gate |
| `models/inputs/` | FieldEncoder — maps each record's value/status matrix into update vectors |
| `models/taaft/` | TAAFT (Topological Anti-Adversary Foundation Transformer) — the adversary-analysis core with energy descent, six lens terms, belief/suspicion readouts |
| `models/verifier/` | Verifier — process-reward model, trust value head, calibration policy head, drift monitoring, conformal thresholds |
| `nn/` | Shared neural primitives: norms, SwiGLU, typed attention with null keys, rotary time encoding, two-stream loop, LoRA |
| `objectives/` | Reward definitions, scoring contracts, and template objectives |
| `physics/` | Physics boundary Φ_phys: hard limits, soft residuals, network constants, protocol constraints, queueing bounds |
| `pipeline/` | Training-pipeline orchestration: stage freezing, dataset splits, stage-level managers |
| `roles/` | High-level role contracts for Simulator, Forecaster, Advisor, Verifier, Decoder |
| `statphys/` | Statistical-physics readouts: Gibbs thermodynamics, traffic/graph entropies, early-warning indicators (critical slowing down), CLI |
| `testing/` | Shared test fixtures and synthetic data builders used by the test suite |
| `tracking/` | Run logging and experiment tracking |
| `training/` | Full training engine: all six stages, optimisers (Muon, AdamW), QK-norm clipping, EMA, distributed training, checkpointing, augmentation, ablation, and smoke tests |
| `worldsim/` | Ground-truth world simulator: topology builder, hidden attacker/defender dynamics, observation layer, scenario library, CLI |

### `conf/` — Configuration files (auto-generated from dataclasses)

| Sub-folder | What it configures |
|---|---|
| `analytics/` | Drift, EDA, information audit, leakage, observability, spatial, TDA and temporal analysis parameters |
| `baselines/lr/` | Logistic-regression baseline family hyperparameters and cross-validation settings |
| `decisions/` | Governance decision manifest |
| `evaluation/` | Evaluation protocol configuration |
| `model/advisor/` | Advisor shape and search parameters |
| `model/forecaster/` | Forecaster shape, MPC budgets (K, N, B) |
| `model/graph/` | Graph builder: planes, hyperedge kinds, subgraph caps |
| `model/inputs/` | FieldEncoder: slot width, hash rows, pooling heads |
| `model/memory/` | Retention schedule, bucket layout, long-term memory size |
| `model/training/` | Training-time shapes and stage switches |
| `model/verifier/` | Verifier shape and calibration settings |
| `site/` | Site-adapter (LoRA) configuration |
| `statphys/` | Statistical-physics engine: ensembles, temperature, EWS series, alarm thresholds |
| `training/L.yaml` | Full L-size training run configuration (1.13 B parameters) |
| `training/tiny.yaml` | Tiny-width training run for CI/smoke testing |
| `worldsim/` | Named scenario library: enterprise APT, ransomware, exfiltration, DoS, OT manipulation, slow recon, benign baselines |

### `docs/` — Documentation

| File / folder | What it covers |
|---|---|
| `architecture.md` | The canonical design brief: processing story, memory layout, training pipeline, outputs, evaluation |
| `build-spec.md` | Engineering specification: philosophy, computation contracts, maths per component, training stages, L preset |
| `sizing.md` | Exact parameter counts (built on meta device), compute profile, memory budget per operating point |
| `baselines-lr.md` | Logistic-regression baseline protocol, usage, solvers |
| `limitations.md` | Known limitations, mitigations, and ongoing work |
| `statphys.md` | Statistical-physics module: thermodynamic readouts, entropies, early-warning system, references |
| `worldsim.md` | Ground-truth world simulator: design, topology, attacker dynamics, observation, outputs |
| `assumptions.md` | Master list of engineering assumptions (AS-xx) |
| `decisions.md` | Generated decisions report (decided, held, proposed) |
| `glossary.md` | Project vocabulary |
| `build-agents.md` | Agent and role descriptions for the build |
| `adr/` | Architecture Decision Records (ADR-0001 through ADR-0009) |
| `assumptions/` | Per-domain assumption files: agents, data, generator, integration, lr-baseline, perception, precision, statphys, taaft, tstct-memory, worldsim |
| `sources/` | Literature index |

### `scripts/`

| File | What it does |
|---|---|
| `smoke_e2e.py` | End-to-end smoke test: runs one mini-batch through all six pipeline stages and checks invariants |

### `tools/`

| File | What it does |
|---|---|
| `gen_assumptions.py` | Generates the `docs/assumptions.md` master list and per-domain assumption files from the source of truth in `governance/assumptions.py` |

### `.github/workflows/`

| File | What it does |
|---|---|
| `ci.yml` | Continuous integration: ruff, mypy, and the full pytest suite at tiny widths |

---

## Module-level Details

### `analytics/`
| Module | Purpose |
|---|---|
| `cli.py` | Command-line entry for all analytics sub-commands |
| `config.py` | Dataclasses for every analytics task |
| `corpus.py` | Builds the analysis corpus from source segments |
| `dependence.py` | Statistical-dependence tests between features |
| `discrimination.py` | Measures class discriminability per field |
| `drift.py` | Detects distributional drift between splits |
| `duplicates.py` | Near-duplicate and exact-duplicate detection |
| `eda.py` | Exploratory data analysis and summary statistics |
| `information.py` | Mutual information and information-theoretic measures |
| `leakage.py` | Temporal and label leakage auditing |
| `observability.py` | Per-field observation-status coverage analysis |
| `report.py` | Renders analysis results to structured reports |
| `robust.py` | Robustness checks under feature perturbation |

### `core/`
| Module | Purpose |
|---|---|
| `config.py` | Top-level `NagaHanaConfig` and run-level settings |
| `errors.py` | Custom exception hierarchy |
| `introspection.py` | Runtime inspection of built modules and decisions |
| `modes.py` | Run-mode enum: train, evaluate, live, forensic, lab |
| `registry.py` | Component registry for dynamic dispatch |
| `roles.py` | Abstract role contracts (Simulator, Forecaster, Advisor, Verifier, Decoder) |

### `data/`
| Module | Purpose |
|---|---|
| `collate.py` | Batching and padding of `ColumnarUpdates` |
| `dataset.py` | PyTorch dataset wrapping source segments |
| `labels.py` | Label tables, infiltration flags, stage labels |
| `sampling.py` | Split assignment, class-balanced window sampling |
| `stream.py` | Stream planning: ordered walk over multi-source corpora |
| `windows.py` | Window builder: trigger ranges, label limits, forecast targets |

### `datamodel/`
| Module | Purpose |
|---|---|
| `columnar.py` | `ColumnarUpdates`: the value/status matrix that carries every record |
| `fields.py` | Field catalogue: name, kind, physics limits, observation-status rules |
| `layers.py` | Observation-layer rules per telemetry source |
| `native.py` | Native Python record representation |
| `records.py` | Record schema and validation |
| `spec.py` | Data model spec and version declaration |
| `status.py` | Observation-status enum and propagation rules |
| `versioning.py` | Schema versioning and migration |

### `evaluation/`
| Module | Purpose |
|---|---|
| `arena.py` | Evaluation arena: runs models and baselines side by side |
| `baseline.py` | Baseline wrapper for the evaluation loop |
| `calibration.py` | Reliability analysis, ECE, Brier, conformal calibration |
| `catalogue.py` | Dataset catalogue and source metadata |
| `cli.py` | Evaluation CLI |
| `components.py` | Per-component output arrays and aggregation |
| `config.py` | Evaluation configuration dataclasses |
| `episodes.py` | Episode-table handling (per-entity compromise timelines) |
| `faithfulness.py` | Faithfulness tests for attributions and explanations |
| `forecasting.py` | Forecast scoring: Brier, log score, CRPS, skill scores |
| `forensics.py` | Forensic replay evaluation: stage onset, patient zero, narrative |
| `generalisation.py` | Leave-one-network-out and cross-dataset evaluation |
| `metrics.py` | F1, precision, recall, FPR, FNR, AUROC, AUPRC |
| `multidataset.py` | Multi-dataset result aggregation |
| `operations.py` | Operational metrics: latency, throughput, memory growth |
| `paths.py` | Attack-path evaluation: precision/recall@N, edit distance, Kendall τ |
| `predictions.py` | Prediction record contracts: `DetectionPredictions`, `ForecastPredictions`, `StagePredictions`, `TimeToEventPredictions`, `StateForecastPredictions` |
| `protocols.py` | Evaluation protocols P1…Pn |
| `ranking.py` | Ranking quality metrics and ordinal safety |
| `registry.py` | Metric and protocol registry |
| `reports.py` | Report rendering |
| `resampling.py` | Bootstrap and jackknife confidence intervals |
| `scorer.py` | Score aggregation across splits |
| `significance.py` | Statistical significance tests |
| `stages.py` | ATT&CK stage evaluation: top-k accuracy, macro-F1, confusion |
| `state.py` | Evaluation state and run bookkeeping |
| `survival.py` | Survival analysis: C-index, time-dependent AUC, integrated Brier score, lead time |

### `governance/`
| Module | Purpose |
|---|---|
| `assumptions.py` | All engineering assumptions (AS-xx) as dataclasses: reasoning, evidence, held decisions they stand in for |
| `assumptions_build.py` | Assumption validation at build time |
| `assumption_types.py` | Assumption type taxonomy |
| `decisions.py` | All design decisions (D-xx): decided, held, proposed; with code dependency annotations |
| `report.py` | Generates `docs/decisions.md` and `docs/assumptions.md` |

### `graph/`
| Module | Purpose |
|---|---|
| `builder.py` | Builds local hypergraph subgraphs per position (BFS, capped by recency) |
| `hypergraph.py` | `GraphBatch`: disjoint-union mini-batch of local subgraphs |
| `planes.py` | Six relation-plane definitions and hyperedge-kind rules |
| `window.py` | As-of discipline enforcement: filters hyperedges and node features by `since ≤ t` |

### `inference/`
| Module | Purpose |
|---|---|
| `buffer.py` | Streaming input buffer with event-time ordering and bounded reordering |
| `engine.py` | Live inference engine: per-state-update processing and per-trigger forecasting |
| `explain.py` | Inline attribution extraction during inference |
| `forensic.py` | Forensic replay engine: reconstructs the timeline from an uploaded capture |

### `ingest/`
| Module | Purpose |
|---|---|
| `capture.py` | Packet capture orchestration |
| `cef_leef.py` | CEF and LEEF log adapter |
| `codes.py` | Port, protocol, and flag code tables |
| `config.py` | Ingest configuration |
| `convert.py` | Record-to-columnar conversion |
| `core.py` | Source-adapter base class and dispatch |
| `derive.py` | Derived field computation (IAT, duration, flag aggregates) |
| `devtelemetry.py` | Developer telemetry adapter |
| `flowexport.py` | NetFlow v5/v9 and IPFIX adapter |
| `kafka.py` | Kafka consumer adapter |
| `mapping.py` | Field mapping from source formats to the superset data model |
| `packets.py` | Packet-level field extraction |
| `pcap.py` | PCAP/pcapng adapter with flow-state emission |
| `ports.py` | Port-to-service-class and port-to-plane mapping |
| `registry.py` | Source adapter registry |
| `sflow.py` | sFlow adapter |
| `snort.py` | Snort alert adapter |
| `suricata.py` | Suricata EVE JSON adapter |
| `syslog.py` | Syslog adapter |
| `timeparse.py` | Timestamp parsing and UTC normalisation |
| `windows.py` | Windows event log adapter |
| `wireshark.py` | Wireshark / tshark JSON adapter |
| `zeek.py` | Zeek log adapter |

### `lab/`
| Module | Purpose |
|---|---|
| `compute.py` | Compute profiler: FLOP counts per component, verified against PyTorch's `FlopCounterMode` |
| `info_audit.py` | Information audit: measures the best achievable accuracy per observation regime |
| `sizing.py` | Parameter counter: builds every component on the meta device and counts exactly |
| `world_sim.py` | Lab entry point for world-simulator runs |

### `memory/`
| Module | Purpose |
|---|---|
| `access.py` | Access-control matrix: enforces which roles may read/write each memory region |
| `environment.py` | Environment store: log-time bucketed KV cache per entity, merge-on-overflow |
| `eventlog.py` | Durable event log (`ColumnarUpdates`); caches are rebuildable views |
| `imagination.py` | Imagination store: TAAFT memory-stream KV for the last M_im triggers plus forecasts |
| `kvcache.py` | Generic KV cache with typed access |
| `longterm.py` | Titans-style long-term neural memory, updated once per trigger |
| `regions.py` | Memory region declarations (Environment, Imagination, Monitor) |
| `retention.py` | Retention schedule: alpha per trigger, flood-invariant bucket updates |

### `models/`
| Module | Purpose |
|---|---|
| `nagahana.py` | Top-level model assembly and the `count_parameters` utility |
| `batch.py` | Tensor-contract dataclasses shared across components |
| `latent.py` | Latent-space utilities (continuous + categorical hybrid) |
| `latent_kl.py` | KL divergence with free bits for the variational objective |
| `variational.py` | Reparameterisation and straight-through-categorical sampling |
| `vocab.py` | Shared vocabulary: 15 ATT&CK stage classes, 700 technique slots, D3FEND action slots |

### `models/taaft/`
| Module | Purpose |
|---|---|
| `blocks.py` | 34-block TAAFT decoder-style transformer (self-attention + cross-attention to Environment + SwiGLU) |
| `imagination.py` | Two-stream loop and Imagination KV write |
| `noise.py` | Per-entity noise statistics (periodogram SNR, Hurst exponent) used by the information lens |
| `objectives.py` | Energy terms: belief-and-trust, game, information, topology, time, cause, physics |
| `readouts.py` | Belief readouts: compromise belief, stage posterior, goal/type posterior, telemetry trust, thermodynamic tensors |
| `structure.py` | Hypothesis space, energy-descent refinement, full TAAFT assembly |
| `testing.py` | Test fixtures for TAAFT at small widths |

### `models/forecaster/`
| Module | Purpose |
|---|---|
| `losses.py` | Survival NLL, hazard targets, censoring |
| `routes.py` | MPC-guided imagination: route sampling, one-step lookahead, route-weight correction, P_inf curve |

### `models/verifier/`
| Module | Purpose |
|---|---|
| `calibration.py` | Calibration policy head, temperature proposal, RLCD reward |
| `feedback_ledger.py` | Human-feedback ledger (supplied truth, responded-to tags) |
| `heads.py` | Process-reward model, trust value head |
| `monitor_gate.py` | Monitor drift detectors: CUSUM, Page–Hinkley, systematic-gap alarm, poisoning alert |

### `nn/`
| Module | Purpose |
|---|---|
| `attention.py` | Typed multi-head attention with null keys, topology bias, and KV return |
| `blocks.py` | Pre-norm residual block (RMSNorm + SwiGLU) |
| `loop.py` | Two-stream weight-tied loop (memory stream + thinking stream with input re-injection) |
| `lora.py` | LoRA rank-16 adapters for site calibration |
| `mlp.py` | MLP and SwiGLU variants |
| `norms.py` | RMSNorm and QK-norm |
| `numeric.py` | Signed log1p and periodic numerical embeddings |
| `positional.py` | Continuous-time rotary positional encoding (no index positions) |

### `physics/`
| Module | Purpose |
|---|---|
| `constants.py` | Physical constants (max packet size, port range, MTU, etc.) |
| `constraints.py` | Hard limits applied by the Decoder (counts ≥ 0, duration ≥ 0, flag counts ≤ packets) |
| `network.py` | Network-level physics: reachability and rate bounds |
| `normalise.py` | Physics-aware normalisation of raw field values |
| `protocol.py` | Protocol-level constraints (TCP flag combinations, header sizes) |
| `queueing.py` | Queueing-theory bounds on inter-arrival times and throughputs |
| `residuals.py` | Soft-residual Φ_phys term: counts violations on observed fields |
| `term.py` | Physics-boundary term assembly for the energy function |

### `statphys/`
| Module | Purpose |
|---|---|
| `cli.py` | `nagahana statphys` CLI: write-config, check-config, show, ews, calibrate, alarm |
| `config.py` | Statistical-physics configuration dataclass |
| `entropy.py` | Shannon traffic entropies (port, protocol, flags, peers); Miller-Madow and Chao-Shen estimators |
| `evaluation.py` | Evaluation helpers: novelty AUROC, alarm lead times, false-alarm rate |
| `ews.py` | Early-warning system: variance, lag-1 AC, skewness, DFA, Kendall τ trends, conformal alarms |
| `graphs.py` | Graph entropy: multiplex activity graph, hierarchical redundancy reduction |
| `io.py` | Trajectory save/load for the statphys module |
| `spectral.py` | Von Neumann entropy, spectral entropy, quantum Jensen-Shannon divergence; stochastic Lanczos |
| `thermo.py` | Gibbs thermodynamics: free energy, mean energy, entropy, heat capacity, susceptibility |
| `trajectory.py` | Streaming EWS, sliding Kendall, traffic-entropy tracker, sliding multiplex |

### `training/`
| Module | Purpose |
|---|---|
| `stage1.py` – `stage6.py` | Six training stages (pre-train CVG-AE/TSTCT, pre-train TAAFT, full training, zero-shot validation) |
| `engine.py` | Training loop orchestration |
| `optim.py` | Optimiser setup (Muon, AdamW, LR schedules) |
| `muon.py` | Muon optimiser implementation |
| `qkclip.py` | QK-norm gradient clipping |
| `ema.py` | Exponential moving average of weights |
| `distributed.py` | Distributed training utilities |
| `checkpoint.py` | Checkpoint saving and loading |
| `data.py` | Training data loader and batch assembly |
| `augment.py` | Generator-variant mixing into training batches |
| `carry.py` | KV-cache carry across micro-batches |
| `feedback.py` | Human-feedback ingestion during supervised phases |
| `manifest.py` | Dataset manifest and split tracking |
| `prep.py` | Data preparation pipeline |
| `config.py` | Training configuration dataclass |
| `common.py` | Shared training utilities |
| `randomness.py` | Seeding and reproducibility |
| `runlog.py` | Run log and metric recording |
| `runs.py` | Run management (resume, compare) |
| `serialization.py` | Model serialisation (versioned, checksummed) |
| `smoke.py` | Smoke tests run at the end of each stage |
| `ablation.py` | Ablation study scaffolding |
| `assumptions.py` | Training-time assumption checks |
| `variants.py` | Variant-mixing strategy |
| `orchestrator.py` | Multi-node training orchestrator |

### `worldsim/`
| Module | Purpose |
|---|---|
| `attack.py` | Attacker stochastic policy over the logical attack graph (ATT&CK-tagged techniques) |
| `benign.py` | Benign traffic model: inhomogeneous Poisson sessions, OT polling, diurnal/weekly profiles |
| `cli.py` | `nagahana worldsim` CLI: list, generate, write-library |
| `config.py` | Scenario configuration dataclasses |
| `dynamics.py` | Ogata thinning: marked point process for the hidden dynamics |
| `emit.py` | Record emitter: turns hidden sessions into `ColumnarUpdates` |
| `observe.py` | Observation layer: sensor fabric, sampling, packet loss, observation status |
| `params.py` | World parameters (entity counts, session rates, attacker pace) |
| `rng.py` | Counter-based Threefry RNG with reproducible key hierarchy |
| `scenarios.py` | Named scenario library: enterprise APT, ransomware, exfiltration, DoS, OT manipulation, slow recon |
| `simulate.py` | `simulate_world`: runs one world, returns `WorldOutput` |
| `state.py` | Hidden state: attacker control level, persistence, knowledge, credentials, per-entity impact |
| `topology.py` | Network topology builder: enterprise estate and OT Purdue model |
| `vocab.py` | Shared vocabulary: campaign styles, stage classes, technique identifiers |

---

## Model: Parameters and Compute

NagaHana is one model, **L** (1.13 billion parameters, Generator excluded).
Weights are in single precision (fp32). Outputs are in float64.

### Parameter Count

| Component | Parameters | Share |
|---|---:|---:|
| Input layer (FieldEncoder) | 9,409,792 | 0.8 % |
| CVG-AE | 117,852,056 | 10.4 % |
| Decoder | 6,309,600 | 0.6 % |
| TSTCT | 214,722,872 | 18.9 % |
| Long-term memory | 7,340,032 | 0.6 % |
| TAAFT | 515,800,696 | 45.5 % |
| Forecaster | 115,942,095 | 10.2 % |
| Advisor | 72,931,585 | 6.4 % |
| Verifier | 73,959,939 | 6.5 % |
| **Total (Generator excluded)** | **1,134,268,667** | 100 % |

Generator (training only): 107,992,101 parameters at the canonical 54-column layout.

### Weights and Memory

| Figure | Value |
|---|---|
| Weights (fp32) | 4.23 GiB (4.54 GB) |
| Training state (mixed-precision Adam) | 16.90 GiB |
| Working Environment cache (fp32) | 256.00 GiB |
| Inference memory in all | 283.26 GiB (304.1 GB) |
| 80 GB accelerators needed for inference | 4 (320 GB total) |

### Compute Per Operating Point

Working context: 4,096 active entities × 512 states per entity.

| Figure | Value |
|---|---|
| Per state update (FieldEncoder + CVG-AE + TSTCT, R=4) | 5.75 GFLOP |
| At 18,400 state updates/s | 105.8 TFLOP/s |
| Per Forecaster trigger (TAAFT + imagination K=12 N=200 + Verifier + long-term memory) | 37.66 TFLOP |
| Server load (streaming + one trigger per 60 s) | 89 % of 1.2 × 10¹⁴ FLOP/s |
| Advisor solve (on demand, W=64, depth 3, 50 rollouts) | 245.13 TFLOP |

### Training

| Figure | Value |
|---|---|
| State updates in the corpus | 3.2 × 10¹⁰ |
| Total training compute | 9.84 × 10²⁰ FLOP |
| GPU time (H100 at 40 % utilisation) | 691 GPU-hours (~3.6 days on 8 accelerators) |
| Training datasets | CIC-IDS2018, CTU-13, CIC-IoT-2023 |
| Splits (full training) | 60 % train (real + generated), 20 % test (real), 20 % validation / zero-shot (real + unseen) |

---

## Results and Evaluation

### Evaluation Methods and Metrics

| Component | Method | Metrics |
|---|---|---|
| Detection / forecast (overall) | Train on known, test on real zero-shot; novel and known reported separately; cross-dataset and leave-one-network-out | F1, precision, recall, FPR, FNR, detection error; AUROC, AUPRC |
| Forecast probability | Proper scoring; reliability analysis | Brier score, log score, CRPS (paths), ECE, reliability diagrams; Brier/CRPS **skill scores** vs persistence and climatology baselines |
| Lead time | Time before stage completion at which P_inf crosses the alert threshold | Median lead time; lead time at a fixed FPR |
| Time-to-event | Survival analysis with censoring | C-index, time-dependent AUC, integrated Brier score |
| ATT&CK stage | Per-step stage prediction | Top-k accuracy, macro-F1 per stage, confusion by stage |
| Attack paths | Set/sequence match of imagined vs realised paths | Path precision/recall@N, edit distance, Kendall τ, ordinal safety |
| CVG-AE | Masked reconstruction on observed fields; latent health | Reconstruction error, edge AUROC/AP, KL, active latent units; OOD AUROC |
| TSTCT | Next-latent prediction; mask audits | Latent prediction error, no-future-leak test, attention-mask audit |
| Decoder | Fidelity and provenance | Decoded-vs-observed error per plane; zero believed-as-observed renderings |
| TAAFT belief/trust | On simulated worlds with known hidden state | Belief Brier/log score; trust AUROC on injected corrupted telemetry |
| Energy (novelty / early warning) | OOD detection, early-warning lead time | OOD AUROC; early-warning lead time; false-alarm rate of energy alerts |
| Game / Advisor | Re-imagination and counterfactual replay | ΔP_inf per counter, disruption cost, feasibility rate, regret vs oracle, analyst acceptance |
| Verifier | Calibration and drift | ECE/Brier before vs after calibration; drift-detection delay at fixed false-alarm rate; injected-poisoning detection rate |
| Generator | Fidelity, diversity, utility, safety | MMD/Wasserstein to real; physics-violation rate; label preservation; train-synthetic-test-real (TSTR); no zero-shot leakage |
| Physics boundary | Hallucination control | Violation rate of model outputs (reconstructions, imagined steps, variants) |
| Operations | Deployment realism | Latency per state update, throughput, memory growth, robustness to telemetry loss |
| Forensics | Replay on labelled incidents | Stage-onset timing error, patient-zero accuracy, narrative step precision/recall |

Baselines: logistic regression trained on the same features (Detection LR, Hazard LR, Stage LR, Ridge next-state forecaster), plus persistence and climatology reference forecasters. All are produced by `nagahana.baselines.lr` through the same evaluation prediction contracts.

### Known Limitations

| # | Limitation | Severity |
|---|---|---|
| L-01 | Encrypted payloads (TLS 1.3, QUIC) shrink the observable surface | Moderate |
| L-02 | Single-vantage segments limit cross-check of sensor reliability | Moderate |
| L-03 | Public training corpora carry label noise and emulated (not production) attacks | Moderate |
| L-04 | Thinner coverage of some industrial protocols (PROFINET, EtherNet/IP CIP, BACnet, ICCP) | Low |
| L-05 | Out-of-order and lossy telemetry can leave the reconstructed timeline incomplete | Low |
| L-06 | Cold start: no established normal behaviour for the first days at a site | High |
| L-07 | Rare legitimate events can resemble low-and-slow attacks | Moderate |
| L-08 | Full L inference needs ~4 × 80 GB accelerators | Moderate |
| L-09 | Per-entity memory grows with host count × retention period at large CII sites | Moderate |
| L-10 | PCAP parsing is a per-packet Python loop; line-rate capture needs flow-level sensors in front | Low |
| L-11 | An adaptive adversary can shift behaviour once it notices defensive responses | Moderate |
| L-12 | Explanations attribute an internal quantity close to, but not identical with, the displayed probability | Moderate |
| L-13 | Threat-actor attribution from network telemetry alone is low-confidence | Low |
| L-14 | Causal attention heads model influence but do not certify identified causal structure | Moderate |
| L-15 | Physical-consistency checks assume wire-level packet sizes; segmentation offload can exceed them | Low |

---

## Architecture Summary

```
ColumnarUpdates ──► FieldEncoder ──► update vectors
                                         │
                    graph builder ──► local hypergraph ──► CVG-AE ──► z (posterior)
                                                                           │
                               TSTCT memory stream ──► Environment KV cache
                               TSTCT thinking stream (R passes) ──► refined state
                                                                           │
trigger ──► TAAFT (entity tokens + adversary slots; reads Environment + Imagination)
        ──► energy descent on E_total ──► belief readouts
        ──► Imagination KV written
Forecaster ──► P_inf(k), stage per step, hazard, attack routes, back-projected states
Verifier ──► trust score, calibration proposal, drift report
Advisor (on demand) ──► D3FEND counter-measure sequences, ranked by ΔP_inf
Decoder (parallel) ──► live network graph: observed / believed / forecast
```

**Memory regions:**
- **Environment** — observed facts and transitions (TSTCT KV cache). Simulator writes, others read.
- **Imagination** — beliefs, suspicion and forecasts (TAAFT analysis cache). Forecaster writes, Advisor/Decoder/Verifier read.
- **Monitor** — deviations, drift, poisoning alerts. Verifier writes.

---

## Quick Start

```bash
# Write default configs
python -m nagahana lr write-config --out conf/baselines/lr
python -m nagahana worldsim write-library --out conf/worldsim

# Run a world simulation
python -m nagahana worldsim generate --scenario enterprise_apt --worlds 4 --out data/sim

# Fit and evaluate the LR baseline
python -m nagahana lr fit --corpus corpora/cic18 --config conf/baselines/lr/lr.yaml --out models/lr-cic18.npz
python -m nagahana lr predict --model models/lr-cic18.npz --corpus corpora/cic18 --protocol P1 --out outputs/lr-cic18.npz

# Parameter and compute profile of L
python -m nagahana.lab.sizing
python -m nagahana.lab.compute

# Statistical physics CLI
python -m nagahana statphys show --file outputs/trigger.npz
```

---

## References

- Hoffmann et al. 2022 (arXiv:2203.15556) — compute-optimal training
- Hafner et al. 2023 (arXiv:2301.04104) — DreamerV3, straight-through categorical
- Behrouz et al. 2024 (arXiv:2501.00663) — Titans long-term neural memory
- Williams et al. 2017 / Hansen et al. 2024 (arXiv:2310.16828) — MPPI / TD-MPC2
- Damani et al. 2025 (arXiv:2507.16806) — RLCD
- Angelopoulos & Bates 2021 (arXiv:2107.07511) — conformal thresholds
- De Domenico & Biamonte 2016 (PRX 6:041062) — spectral graph entropy
- Scheffer et al. 2009 (Nature 461:53) — critical slowing down
- Ubaru, Chen & Saad 2017 (SIAM J. Matrix Anal.) — stochastic Lanczos
- Ou, Govindavajhala & Appel 2005 (USENIX Security) — MulVAL attack graphs
- Erion et al. 2021 (Nature MI, arXiv:1906.10670) — Expected Gradients (SHAP)
