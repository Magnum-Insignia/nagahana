# NagaHana — Full Results

<div align="right">

**SIH Team ID: 155021**

</div>

---

## Contents

1. [Model Parameters and Scale](#1-model-parameters-and-scale)
2. [Compute Profile](#2-compute-profile)
3. [Memory Budget](#3-memory-budget)
4. [Training Configuration](#4-training-configuration)
5. [Data Pipeline Measurements](#5-data-pipeline-measurements)
6. [Evaluation Metrics and Methods](#6-evaluation-metrics-and-methods)
7. [Outputs Per Trigger](#7-outputs-per-trigger)
8. [Statistical Physics Outputs](#8-statistical-physics-outputs)
9. [Precision Policy Measurements](#9-precision-policy-measurements)
10. [Component Invariants Verified by Tests](#10-component-invariants-verified-by-tests)
11. [Known Limitations](#11-known-limitations)

---

## 1. Model Parameters and Scale

NagaHana is a 1.13 billion parameter model (Generator excluded). Every component is built on the meta device (no memory allocation) and its parameters are counted exactly by `python -m nagahana.lab.sizing`.

### Parameter Count (built model, 2026-10-02)

| Component | Parameters | Millions | Share |
|---|---:|---:|---:|
| Input layer (FieldEncoder) | 9,409,792 | 9.41 M | 0.8 % |
| CVG-AE (6 planes × 4 layers × 256; 8 node kinds; 4 hyperedge kinds/plane; RWSE 16 steps; z = 128 + 16×32) | 117,852,056 | 117.85 M | 10.4 % |
| Decoder (field heads + 6 per-plane edge heads at 512) | 6,309,600 | 6.31 M | 0.6 % |
| TSTCT (16 blocks × 1024; 16 heads: 4 spatial / 8 temporal / 4 causal; SwiGLU 2816) | 214,722,872 | 214.72 M | 18.9 % |
| Long-term memory (Titans-style MLP at 1024, hidden 2048) | 7,340,032 | 7.34 M | 0.6 % |
| TAAFT (34 blocks × 1024; 16 heads; SwiGLU 2816; 16 adversary slots; d_y 256; 7 energy terms; 32 memory probe keys) | 515,800,696 | 515.80 M | 45.5 % |
| Forecaster (8 blocks × 1024; 700 technique slots; 15 stages) | 115,942,095 | 115.94 M | 10.2 % |
| Advisor (4 decoder-style blocks × 1024; 256 D3FEND slots) | 72,931,585 | 72.93 M | 6.4 % |
| Verifier (4 blocks × 1024 PRM + trust value head + calibration policy head) | 73,959,939 | 73.96 M | 6.5 % |
| **Total (Generator excluded)** | **1,134,268,667** | **1,134.3 M** | 100 % |

**Generator** (training only, excluded from deployment count): 107,992,101 parameters at the canonical 54-column layout; 92,767,779 at the PCAP adapter's 45 columns. Count depends on the column layout because its value bins, embeddings and output heads are per column.

### Model Architecture Shapes

| Component | Shape |
|---|---|
| FieldEncoder: field width; update vector; hashed rows | 128; 256; 65,536 |
| CVG-AE: relation planes × layers × width | 6 × 4 × 256 |
| Latent z (continuous + G×C categorical) | 128 + 16×32 = 640 |
| TSTCT: blocks × width (heads: spatial / temporal / causal) | 16 × 1024 (16: 4 / 8 / 4) |
| TSTCT keys per query: spatial + temporal + causal | 64 + 512 + 32 = 608 |
| TAAFT: decoder-style blocks × width (heads), reading TSTCT's cache (block map 34 → 16) | 34 × 1024 (16) |
| TAAFT: adversary slots; hypothesis width; long-term memory probe keys | 16; 256; 32 |
| SwiGLU hidden width (every 1024-wide block) | 2,816 |
| Forecaster / Advisor / Verifier-PRM blocks × width | 8 / 4 / 4 × 1024 |

---

## 2. Compute Profile

Generated with `python -m nagahana.lab.compute` from the built modules. Every formula verified against PyTorch's `FlopCounterMode` — each equals PyTorch's count exactly (one known exception: a one-key einsum that PyTorch evaluates as an elementwise product which its counter does not see).

### Operating Point

- **Working context:** 4,096 active entities × 512 states per entity = 2,097,152 cached states
- **Sustained rate:** 18,400 state updates per second
- **Trigger cadence:** 1 per 60 s window
- **Effective compute:** 1.2 × 10¹⁴ FLOP/s (four 80 GB accelerators)

### Per State Update

| Operation | FLOPs |
|---|---:|
| Input layer (FieldEncoder, 54 canonical columns) | 7.70 MFLOP |
| CVG-AE (32 nodes, 4 incidences per node and plane) | 1.52 GFLOP |
| TSTCT (2 positions × 16 blocks; memory stream + R=4 thinking passes) | 4.22 GFLOP |
| — of which thinking passes | 3.02 GFLOP |
| **Total per state update** | **5.75 GFLOP** |
| Per parameter | 5.07 FLOP/parameter |
| At 18,400 updates/s | 105.8 TFLOP/s |

### Per Forecaster Trigger (K=12, N=200, R=4, I=8, B=8)

| Operation | FLOPs |
|---|---:|
| TAAFT over 4,112 tokens (34 blocks; memory stream + R=4 thinking passes; I=8 descent steps) | 31.12 TFLOP |
| Imagination: N×K = 2,400 route steps, 8 candidates each | 4.85 TFLOP |
| Verifier process reward of every imagined step | 1.59 TFLOP |
| Long-term memory write | 103.08 GFLOP |
| **Total per trigger** | **37.66 TFLOP** |
| Compute time at 1.2 × 10¹⁴ FLOP/s | **0.31 s** |
| Advisor solve (on demand; W=64, depth 3, 50 rollouts) | 245.13 TFLOP (~2.0 s of server compute) |

### Server Load

| Metric | Value |
|---|---|
| Streaming load (18,400 updates/s) | 88 % of 1.2 × 10¹⁴ FLOP/s |
| Trigger load (one per 60 s) | 0.5 % |
| **Total server load** | **89 %** |
| Budget (trigger compute time) | 0.38 s |
| Compute time for trigger | 0.31 s (within budget) |

---

## 3. Memory Budget

All figures at the operating point (4,096 entities × 512 states, fp32 caches per D-54).

| Figure | Value |
|---|---|
| **Weights and Training State** | |
| Parameters (Generator excluded; built model, meta device) | 1,134,268,667 |
| Weights, stored and served in fp32 (4 bytes/parameter) | **4.23 GiB (4.54 GB)** |
| — half precision for reference only (not used; D-54 requires no 16-bit weights) | 2.11 GiB |
| Training state (mixed-precision Adam, 16 bytes/parameter) | 16.90 GiB |
| **Context Window** | |
| TSTCT keys per query (spatial + temporal + causal) | 608 (64 + 512 + 32) |
| Working context | 4,096 entities × 512 states = 2,097,152 cached states |
| Environment cache per cached state (fp32) | 128 KiB |
| **Working Environment cache** | **256.00 GiB** |
| Time a busy entity's 512 states cover at sustained rate | 57 s |
| Quiet entities covered for | far longer |
| TAAFT tokens; self keys; cross keys (32 own + 64 neighbours + 32 memory probes) | 4,112; 4,120; 128 |
| Imagination store (8 triggers × 4,112 tokens, 272 KiB each) | 8.53 GiB |
| TAAFT working set of one block (gathered cross keys; self-attention scores) | 3.01 GiB; 3.03 GiB |
| Forecaster route caches with 8 lookahead copies | 8.46 GiB |
| **Inference memory in all** | **283.26 GiB (304.1 GB)** |
| 80 GB accelerators needed for inference | **4 (320 GB total)** |
| Headroom on 4 accelerators | 15.9 GB (5 %) |
| fp32 weights as share of inference memory | 1.5 % |
| Retained Environment per day at sustained rate (24 bytes/update) | 35.53 GiB/day |
| Log-time buckets cover | 1 ms … ~4 weeks (≈31 buckets within 512 slots/entity) |

---

## 4. Training Configuration

### Compute

| Figure | Value |
|---|---|
| State updates in the corpus | 3.2 × 10¹⁰ |
| — of which generated by Generator | 4 × 10⁹ (same count as real) |
| Stage 3 / 4 / 5 compute | 2.62×10²⁰ / 3.21×10²⁰ / 4.01×10²⁰ FLOP |
| **Total training compute** | **9.84 × 10²⁰ FLOP** |
| GPU time at 40 % of one H100's 989 TFLOP/s | **691 GPU-hours** |
| On 8 accelerators | **~3.6 days** |
| Updates per parameter | ~28 (compute-optimal ratio per Hoffmann et al. 2022 is ~20 tokens/parameter) |

### Activation Sizes (micro-batch: 8 sequences × 2,048 states)

| Component | With 3 stored passes | With full recomputation |
|---|---|---|
| TSTCT (dense over window, attention scores stored) | 75.38 GiB | 2.07 GiB |
| TAAFT | 74.11 GiB | 3.91 GiB |

TAAFT's 74.1 GiB do not fit beside the training state (16.9 GiB) on one 80 GB accelerator → TAAFT trains with block recomputation (3.9 GiB) or smaller micro-batches.

### Datasets and Splits

| Dataset | Use |
|---|---|
| CIC-IDS2018 (CSE-CIC-IDS2018) | Primary training, evaluation |
| CTU-13 | Training, cross-dataset evaluation |
| CIC-IoT-2023 | Training (IoT variation) |

| Split | Share | Data |
|---|---|---|
| Train | 60 % | Real + generated variants |
| Test | 20 % | Real only |
| Validation / zero-shot | 20 % | Real + unseen datasets |
| Pretraining | 70 % / 30 % | Real only |

Zero-shot: any window whose network is held out, or that contains any update of a novel attack family, is zero-shot. Novel and known attack families are scored separately in every evaluation table.

### Micro-Batch and Loop Settings

| Setting | Value |
|---|---|
| Micro-batch | 8 sequences × 2,048 states |
| TSTCT thinking passes R during training | 1 + Poisson(3), clipped to [1, 8] |
| Backprop passes | Last 2 thinking passes |
| Masking rate (contributing cells) | 15 % |

---

## 5. Data Pipeline Measurements

### Flow-State Emission — Sample Slice (270,596 packets, 50 min, Window of 1,024 updates)

Measured by running `measure_slice.py` / `measure_stream.py` on the full sample slice.

| Emission Mode | Updates | Windows | Span median / p10 / p90 (s) | Baseline rate (/s) | Scan rate (/s) |
|---|---:|---:|---|---:|---:|
| Packet | 270,596 | 265 | 6.2 / 0.3 / 32.2 | 72.8 | 110.4 |
| Flow-state, active timeout 0.25 s | 161,438 | 158 | 9.8 / 1.6 / 50.4 | — | — |
| **Flow-state, active timeout 1 s** | **153,935** | **151** | **9.7 / 1.5 / 60.2** | **15.2** | **91.6** |
| Flow-state, active timeout 60 s | 136,230 | 134 | 9.2 / 1.5 / 64.0 | 10.5 | 84.4 |

- Flow-state emission (1 s) cuts the baseline rate **4.8× vs packet mode** but the internal scan rate only **1.2×** (each probed port is a 1–2-packet flow).
- Going from 60 s to 1 s active timeout: **+13 % updates**.
- Going from 1 s to 0.25 s: **+5 % updates**.
- The backdoor C2 session: **37 updates at 1 s** vs **25 at 60 s** (median gap 48 s vs 71 s).
- Idle-end updates (unanswered scan SYNs, reported after 120 s idle): **40,618 of 153,935 updates (26 %)**.
- Windows holding at least one trigger: **18 % (packet mode), 30 % (flow-state 1 s)**.

### Timestamp Resolution per Dataset

| Dataset | Resolution | Notes |
|---|---|---|
| CSE-CIC-IDS2018 | 1 s | `dd/mm/YYYY HH:MM:SS`; local time UTC−4 vs PCAP UTC |
| CIC-IDS2017 | 60 s | Files written to the minute |
| CTU-13 | 1 µs | — |

### Sample Slice Ground-Truth Counts (14:45:30–14:46:40, 4,406 packets)

| Category | Count |
|---|---|
| Backdoor C2 packets | 62 |
| Scan (discovery) packets | 1,500 |
| Victim-unannotated packets | 345 |
| Benign packets | 2,499 |

### Ingest Measurement: Categorical Hash Determinism

Code run `tests/test_perception_inputs.py::test_categorical_hash_is_fixed_and_deterministic`:
- 500 random (slot, code) pairs match a pure-Python reference exactly across processes and platforms.

---

## 6. Evaluation Metrics and Methods

### Full Evaluation Table

| Component | Method | Metrics |
|---|---|---|
| **Detection / forecast (overall)** | Train on known, test on real zero-shot; novel and known reported separately; cross-dataset; leave-one-network-out (LONO) | F1, precision, recall, FPR, FNR, detection error; AUROC, AUPRC |
| **Forecast probability** | Proper scoring rules; reliability analysis | Brier score, log score, CRPS (paths), ECE, reliability diagrams; Brier skill score and CRPS skill score vs persistence and climatology baselines |
| **Lead time** | Time before stage completion at which P_inf crosses the alert threshold | Median lead time; lead time at a fixed FPR |
| **Time-to-event ("when")** | Survival analysis with censoring | C-index, time-dependent AUC, integrated Brier score |
| **ATT&CK stage** | Per-step stage prediction over 15 classes | Top-1 accuracy, top-3 accuracy, macro-F1 per stage, confusion matrix by stage |
| **Attack paths** | Set/sequence match of imagined vs realised paths | Path precision@N, path recall@N, edit distance, ranking quality (Kendall τ), ordinal safety |
| **CVG-AE** | Masked reconstruction on observed fields; latent health | Reconstruction NLL; edge AUROC and AP; KL divergence; active latent units; OOD AUROC (energy / likelihood-ratio, not raw ELBO) |
| **TSTCT** | Next-latent prediction; mask audits | Latent prediction error; no-future-leak test; attention-mask audit (strict temporal order verified) |
| **Decoder** | Fidelity and provenance | Decoded-vs-observed error per relation plane; zero believed-as-observed renderings (invariant enforced by type) |
| **TAAFT belief / trust** | On simulated worlds with known hidden state (ground-truth world simulator) | Belief Brier score; belief log score; trust AUROC on injected corrupted telemetry |
| **Energy (novelty / early warning)** | OOD detection; early-warning lead time | OOD AUROC (Mann-Whitney); early-warning lead time; false-alarm rate of energy alerts |
| **Game / Advisor** | Re-imagination and counterfactual replay | ΔP_inf per counter-measure; disruption cost; feasibility rate; regret vs oracle (world simulator); analyst acceptance |
| **Verifier** | Calibration and drift | ECE before vs after calibration; Brier score before vs after calibration; drift-detection delay at a fixed false-alarm rate (QCD); injected-poisoning detection rate |
| **Generator** | Fidelity, diversity, utility, safety | MMD and Wasserstein distance to real; physics-violation rate; label preservation rate; train-synthetic-test-real (TSTR); no zero-shot leakage verified |
| **Physics boundary** | Hallucination control | Violation rate of model outputs (reconstructions, imagined steps, generated variants) |
| **Operations** | Deployment realism | Latency per state update; throughput (updates/s); memory growth rate; behaviour under NetFlow-only vs full telemetry; robustness to telemetry loss |
| **Forensics** | Replay on labelled incidents | Stage-onset timing error; patient-zero accuracy; narrative step precision; narrative step recall |

### Baselines

All baselines share the same input features as the model, produced by `nagahana.baselines.lr` through the same prediction contracts (`evaluation/predictions.py`):

| Baseline | Unit | Output |
|---|---|---|
| Detection LR (`detector.py`) | State update | `DetectionPredictions`: calibrated probability, own threshold, conformal threshold |
| Hazard LR (`hazard.py`) | Usable trigger | `ForecastPredictions` (P_inf(k), hazards), `TimeToEventPredictions` (survival curve, risk) |
| Stage LR (`stage.py`) | (trigger, step) or update | `StagePredictions` over 15 stage classes |
| Ridge next-state (`ridge.py`) | Trigger | `StateForecastPredictions`, horizons 1…K |
| Persistence | As forecast / next-state | `ForecastPredictions`, `TimeToEventPredictions`, `StateForecastPredictions` |
| Climatology | As forecast | `ForecastPredictions`, `TimeToEventPredictions` |

Baseline solver: full-batch L-BFGS (strong Wolfe, float64), preconditioned by Boehning's fixed Hessian bound. For corpora that do not fit in memory: `streamed_lbfgs` (exact) or `minibatch` (averaged SGD with L-BFGS polishing).

Hyperparameter selection: blocked forward-chaining cross-validation inside the training split, per network, with purging by label horizon and an embargo.

---

## 7. Outputs Per Trigger

### Forecast

| Output | Description |
|---|---|
| P_inf(k) | Infiltration probability over the next K steps, with uncertainty (monotone by construction from survival factors; fp64) |
| P_inf band | Route-weighted mean curve; median with 10–90 % band; mode route (highest weight) |
| Stage per step | The ATT&CK stage class per imagined step (15 classes, full Enterprise + ICS) |
| Attack paths | Up to N distinct imagined routes with route weights w_r = count_r/N |
| Hazard curve | Time-to-event ("when"): P(infiltration at step k | none before), per route and mixed |
| Driving features | Attention and attribution per prediction: flags, ports, flow patterns (Expected Gradients / SHAP; LIME) |
| Compute spent | K, N, R (loop passes), S (descent steps), I (descent iterations) — recorded with each forecast |
| Safe horizon | Horizon beyond which the Environment's coverage becomes thin |
| Observability gaps | Annotated per step where the evidence is not supplied |

### Belief and Trust

| Output | Description |
|---|---|
| Per-entity compromise belief | Suspicion with an assume-breach floor φ > 0; p_v = φ + (1−φ)·σ(w·y_v) |
| Malignity score | m_v ∈ [0,1] per entity (soft target: malicious share of trailing window updates) |
| Stage posterior | 15-class posterior over ATT&CK stages per entity |
| Adversary goal posterior | From the 16 adversary-hypothesis slots |
| Adversary type posterior | From the 16 adversary-hypothesis slots |
| Telemetry trust | Per source, per field (σ output of trust readout) |

### Energy

| Output | Description |
|---|---|
| Energy per entity and per time | TAAFT per-token energy E_total at ŷ |
| Lens shares | Share of each of the 6 lens terms (belief-and-trust, game, information, topology, time, cause) in the last descent step; sum = 1 exactly |
| Energy relative to reference | E_ℓ − reference_ℓ (reference = EMA over training windows; fp64) |
| Free energy | F = −T log Z over entity / slot / stage / route ensembles |
| Entropy | S = (U − F)/T over each ensemble |
| Heat capacity | C = Var_p(E)/T² over each ensemble |
| Energy growth rate | Least-squares slope per second over the last `growth_window` valid triggers |
| Novelty | High energy = unfamiliar, not proof of attack |

### Advisory (on demand)

| Output | Description |
|---|---|
| Ranked D3FEND counter-measure sequences | Steps, D3FEND tactic, level (graph or sensor) |
| ΔP_inf per counter | P_inf(K | counter) − P_inf(K | none) |
| Disruption cost | Σ criticality(target kind) × disruption weight(action) |
| Physical feasibility | Φ_phys of predicted effects ≤ τ; plus rule checks |
| Information value | Expected entropy reduction of stage posterior (for detect/model actions) |
| CVaR ranking | CVaR_{0.2} of ΔP_inf over routes (Rockafellar & Uryasev 2000) |

### Verifier

| Output | Description |
|---|---|
| Trust score | Per forecast and per advice (value head; fp64 probability) |
| Calibration proposal | Proposed log-temperature per output family from reliability statistics (applied only on a HumanCommand) |
| Calibration report | Reliability diagram; ECE; Brier score |
| Drift report | CUSUM on forecast–outcome residuals; systematic gap |Σ(f−y)|/n over last 12 resolutions; Welford mean/variance of Environment and Imagination latents with Page–Hinkley alarm |
| Outcome–forecast ledger | Resolved (forecast, outcome) pairs with responded-to tags |
| Poisoning alerts | Flagged for human review; never acted on automatically |
| Conformal thresholds | Split-conformal alert thresholds at a target FPR (Angelopoulos & Bates 2021) |

### Decoder View

| Output | Description |
|---|---|
| Live network graph | Typed multigraph with planes as edge types |
| Provenance per element | OBSERVED / BELIEVED / FORECAST — a BELIEVED element can never carry the OBSERVED tag (tested invariant) |

### Forensic Replay

| Output | Description |
|---|---|
| Timeline | What would have been forecast at each trigger, and when the kill chain became visible |
| Kill-chain narrative | ATT&CK-tagged, step by step |
| Patient-zero trace | Which entity was first compromised and when |
| Counterfactual replay | "If isolated at 14:20…" — re-imagined under the intervention |
| Observability gaps | Per step, showing what was not visible to sensors |
| Anti-forensic tampering signs | Annotated where evidence of tampering is detected |

---

## 8. Statistical Physics Outputs

Computed at every Forecaster trigger from TAAFT's energy and the traffic state. Ensemble members and their energies:

| Ensemble | Members | Energies |
|---|---|---|
| Entities | Active entity tokens | TAAFT per-token energies at ŷ |
| Slots | Active adversary-hypothesis slots | TAAFT per-token energies at ŷ |
| Stage (per entity) | 15 stage classes | −logit_s |
| Routes | N imagined routes (each a microstate) | Σ_k E(null, ŷ_{n,k}) |
| Imagined (per entity) | Imagined steps targeting the entity | E(null, ŷ_{n,k}) |

### Thermodynamic Readouts (per ensemble per trigger)

| Quantity | Formula |
|---|---|
| Log partition | log Z = logsumexp_i(−E_i/T + log g_i) |
| Free energy | F = −T log Z |
| Mean energy | U = Σ_i p_i E_i |
| Gibbs entropy | S = (U − F)/T = log Z + U/T |
| Heat capacity | C = Var_p(E)/T² |
| Susceptibility | χ_O = Var_p(O)/T |

All identities checked against finite differences. Schottky two-level system checked against its closed form.

### Traffic Entropies (per trigger, state window τ−300 s to τ)

Distributions measured: ports, protocol, TCP flag combination, packet-weighted flags, peers (per entity), initiator entities, responder entities.

Estimators: plug-in; Miller-Madow (default); Chao-Shen. Jensen-Shannon divergence between successive triggers measures how fast the traffic mix moves.

### Graph Entropies (per trigger)

| Measure | Source |
|---|---|
| Von Neumann entropy ρ = L/tr L | Braunstein, Ghosh & Severini 2006 |
| Spectral entropy ρ_τ = exp(−τL)/Z at τ = 0.1, 1, 10 | De Domenico & Biamonte 2016 |
| Quantum Jensen-Shannon divergence between planes | Lamberti et al. 2008; Virosztek 2021 |
| Relative entropy q of the multiplex | De Domenico, Nicosia, Arenas & Latora 2015 |
| Hierarchical redundancy reduction | De Domenico et al. 2015 |

Eigendecomposition: exact for connected components ≤ 512 nodes; stochastic Lanczos quadrature (Ubaru, Chen & Saad 2017) with exact zero-mode deflation, low-mode deflation, polynomial control variate, and adaptive probe/depth control. Standard errors reported with every estimate.

### Early-Warning Indicators (per monitored series, per trigger)

Rolling window: 60 samples. Indicators: variance, lag-1 autocorrelation, skewness, kurtosis, return rate −ln(ρ₁)/dt, spectral ratio and exponent, DFA exponent. Trend: Kendall τ-b. Significance (offline): phase-randomised and AR(1) surrogates. Alarm: split-conformal threshold, target rate 0.01 per trigger, calibrated on benign trajectories. Without a calibration file, no alarm is raised — a threshold is never defaulted.

---

## 9. Precision Policy Measurements

All results per D-54 and AS-450–AS-452 (float32 weights and compute; float64 outputs).

### P_inf Precision (code run: `tests/test_precision_outputs.py`)

| Metric | Value |
|---|---|
| P_inf from stored float64 hazards vs model path | Bit-for-bit identical |
| Float64 hand computation agreement | To 1 × 10⁻¹⁵ |
| Previous float32 output path vs float64 P_inf (seeds 0, 1, 2) | 6.48×10⁻⁸, 5.66×10⁻⁸, 5.63×10⁻⁸ (≈1 float32 ulp near 1) |

The gain is resolution near 0 and 1 (float32 cannot represent 1 − P below ≈ 6×10⁻⁸); typical values are unchanged.

### Lens Shares (code runs: `tests/test_taaft_energy.py`, `tests/test_precision_outputs.py`)

| Metric | Value |
|---|---|
| Shares sum to 1 (energy test) | To 1 × 10⁻⁴ |
| Shares sum to 1 (precision end-to-end, where step is non-zero) | To 1 × 10⁻⁹ |

### All Weights fp32 (code run: `test_weights_and_compute_fp32_outputs_fp64_end_to_end`)

- Every parameter is float32.
- Every listed D-54 output (P_inf, hazards, stage posteriors, energies, trust, calibration quantities) is float64 at preset "tiny".

---

## 10. Component Invariants Verified by Tests

### Perception (CVG-AE, FieldEncoder, Graph Builder)

| Invariant | Test |
|---|---|
| Categorical hash is fixed and deterministic (500 random pairs match pure-Python reference) | `test_categorical_hash_is_fixed_and_deterministic` |
| Fan hyperedge appears only after the 8th responder (no future leakage) | `test_fan_appears_only_after_qualification` |
| Hyperedges expire by time-to-live | `test_hyperedges_expire_by_time_to_live` |
| Batched RWSE equals the shared primitive | `test_batched_rwse_equals_shared_primitive` |
| Hard limits hold for random latents (scales 1, 30, 1e3) | `test_hard_limits_hold_for_random_latents` |
| NaN never propagates; excluded cells are ignored | `test_nan_never_propagates_and_excluded_cells_are_ignored` |
| MASK status hides the value | `test_mask_status_hides_the_value` |
| Permutation equivariance | `test_permutation_equivariance` |
| Gradients reach every parameter | `test_gradients_reach_every_parameter` |
| Contact matrices equal brute force (star semantics) | `test_contact_matrices_equal_brute_force` |
| Fan responders are two hops apart (star contact) | `test_star_contact_fan_responders_are_two_hops_apart` |
| No future leakage with truncation invariance | `test_no_future_leak_truncation_invariance` |
| KL and ELBO match closed forms | `tests/test_perception_latent.py` |

### TAAFT

| Invariant | Test |
|---|---|
| Belief recursion reads only the last M_im = 2 triggers (change at trigger 1 alters 1–3, leaves ≥ 4 identical) | `test_belief_recursion_reads_only_the_last_imagination_triggers` |
| `attend_mixed` matches gathered attention to 1 × 10⁻⁵ | `test_attend_mixed_matches_gathered_attention` |
| Dense and gathered cross layouts compute the same function to 1 × 10⁻⁵ | `test_dense_and_gathered_cross_layouts_compute_the_same_function` |
| All lens energies ≥ 0 (min ≥ −1 × 10⁻⁶) | `test_lens_bounds_and_physics_term` |
| Physics-only descent lowers the physics term | `test_lens_bounds_and_physics_term` |
| No future leakage from later positions, contacts or triggers | `test_no_future_leakage_from_later_positions_contacts_or_triggers` |
| Split calls equal one call through analysis carry to 1 × 10⁻⁵ (two calls; three-call chain) | `test_split_calls_equal_one_call_through_the_analysis_carry` |
| Split calls equal one call through ImaginationStore to 1 × 10⁻⁵ | `test_split_calls_equal_one_call_through_the_imagination_store` |
| Past Imagination is read only before each trigger | `test_past_imagination_is_read_only_before_each_trigger` |
| Suspicion floor holds at float32 for inputs up to 1 × 10⁶; rows sum to 1 | `test_suspicion_floor_and_normalised_readouts` |
| Gradients reach every parameter with unrolled descent | `test_gradients_reach_every_parameter_with_unrolled_descent` |
| Descent lowers E_total | `tests/test_taaft_energy.py` |
| Memory keys with dense and gathered layouts agree to 1 × 10⁻⁵ | `test_memory_keys_with_dense_and_gathered_layouts_agree` |
| Read long-term gives gradient to probes, maps and memory | `test_read_longterm_gives_gradient_to_probes_and_memory` |
| Memory keys per trigger are as-of and read by entities and slots | `test_memory_keys_per_trigger_are_as_of_and_read_by_entities_and_slots` |

### TSTCT

| Invariant | Test |
|---|---|
| Dense masked training path ≡ gathered-key KV-cache inference path to 1 × 10⁻⁵ (fp32), R ∈ {1, 3} | `tests/test_tstct_equivalence.py` |
| No future leakage (R > 1 included) | `tests/test_tstct_masks.py` |
| Carry across micro-batches equals single-batch | `tests/test_tstct_carry.py` |

### Forecaster

| Invariant | Test |
|---|---|
| P_inf is monotone non-decreasing by construction | `tests/test_forecaster_model.py` |
| Every Forecaster parameter receives gradient through `teacher_forced` | `tests/test_forecaster_losses.py` |

### Verifier

| Invariant | Test |
|---|---|
| Process reward, trust head, calibration head, monitor drift all build and run | `tests/test_verifier_heads.py`, `tests/test_verifier_monitor_gate.py` |
| Feedback ledger stores and retrieves resolved pairs | `tests/test_verifier_feedback_ledger.py` |
| Calibration reduces ECE on held-out pairs | `tests/test_verifier_calibration.py` |

### Memory

| Invariant | Test |
|---|---|
| Access matrix enforced (HumanCommand gates) | `tests/test_memory.py` |
| Environment, Imagination and Monitor regions have enforced access | `memory/access.py` (checked at construction) |

### World Simulator

| Invariant | Test |
|---|---|
| Determinism under seed | `tests/test_worldsim_*.py` |
| JAX and NumPy backends produce identical worlds | `tests/test_worldsim_equivalence.py` (requested) |
| vmap batch equals sequential worlds | `tests/test_worldsim_*.py` |
| Event time is monotone | Invariant checked |
| No technique fires before its precondition event; cause links form a DAG ordered earlier-in-time | Invariant checked |
| Stage labels match the technique catalogue | Invariant checked |
| No event involves an entity outside the topology | Invariant checked |
| NetFlow records mark packet-level fields NOT_SUPPLIED; sampling keeps about 1-in-n | Invariant checked |
| IDS detection and false-alarm behave as configured | Invariant checked |
| Emitted records validate against the data model and flow into the windowing pipeline | `tests/test_worldsim_scenarios.py` |

### Statistical Physics

| Invariant | Test |
|---|---|
| All thermo identities against finite differences | `tests/test_statphys_thermo.py` |
| Schottky two-level system against closed form | `tests/test_statphys_thermo.py` |
| Streaming EWS equals batch EWS; work per update does not grow with stream | `tests/test_statphys_ews.py` |
| Hurst ≈ 0.5 for IID exponential gaps (seeds 0, 1, 2): 0.498, 0.498, 0.482 | `test_hurst_is_one_half_for_iid_exponential_gaps` |
| Hurst > 0.85 for persistent gaps (exploratory: 1.0) | `test_hurst_is_high_for_persistent_gaps` |
| Periodogram closed form: I(1/T) = n to 1 × 10⁻⁶·n; dominant period = T | `test_periodogram_closed_form_and_planted_period` |

---

## 11. Known Limitations

| # | Limitation | Severity | Current Mitigation | Work in Progress |
|---|---|---|---|---|
| LIM-01 | Encrypted payloads (TLS 1.3, QUIC, encrypted SNI) shrink the observable surface | Moderate | Timing, sizes, flow structure, graph structure, handshake fingerprints; absence is a status, never benign | Better use of graph-frequency and timing statistics for slow attacks |
| LIM-02 | Single-vantage segments: sensor reliability cannot be cross-checked, spoofed values harder to discount | Moderate | Low-reliability observation status assigned; evidence weight reduced | Reliability estimation from overlapping vantages and physics consistency |
| LIM-03 | Public training corpora carry label noise; attack campaigns emulated, not observed in production | Moderate | Corrected label sets used where available; leakage and duplicate audits; novel/known families reported separately | Ground-truth world simulator with known hidden state; information audit (P-14, P-15) |
| LIM-04 | Thinner coverage of some industrial protocols (PROFINET, EtherNet/IP CIP, BACnet, ICCP, C37.118) | Low | Unknown protocols modelled at flow/timing level with fields marked "not supplied" | Field mappings and simulator scenarios for remaining protocols |
| LIM-05 | Out-of-order and lossy telemetry leaves the reconstructed timeline incomplete | Low | Bounded reordering with event-time watermarks; loss carried as statuses | Per-sensor loss estimation feeding the reliability model |
| LIM-06 | Cold start: no established normal behaviour for the first days; a compromised network at deployment can be learned as normal | **High** | Population priors from pretraining; low-confidence early window; attack-objective invariants independent of site baseline | Calibration valid with few site events |
| LIM-07 | Rare legitimate events (quarterly jobs, new deployments) can resemble low-and-slow attacks | Moderate | Alert requires rarity together with progress toward an attack objective; Verifier calibrates site threshold | Memory write rule and evidence measure tied to objective progress |
| LIM-08 | Full inference needs ~4 × 80 GB accelerators (320 GB total) | Moderate | Weights and caches fp32 with fp64 outputs (D-54); compute and memory budget documented per component | Sparse contact representations and vectorised hot paths |
| LIM-09 | Per-entity memory grows with host count × retention period at large CII sites | Moderate | Log-time dyadic memory cells; flood-invariant updates | Retention guarantees under adversarial flooding; adaptive-resolution memory |
| LIM-10 | PCAP parsing is a per-packet Python loop; line-rate capture needs flow-level sensors | Low | Flow-state emission from PCAP (D-51); native ingest of Zeek, Suricata, NetFlow, IPFIX, sFlow | Sandboxed parser process with input caps |
| LIM-11 | An adaptive adversary can change behaviour once it notices defensive responses | Moderate | Decision support only: humans choose the response; responded-to cases excluded from calibration | Counterfactual calibration w.r.t. the known response policy |
| LIM-12 | Explanations attribute an internal quantity close to but not identical with the displayed P_inf; explanation cost grows with descent depth | Moderate | Lens shares, evidence chains and observability-gap annotations shown alongside every forecast | Attribution of the displayed probability integrated over the descent path, with faithfulness tests |
| LIM-13 | Threat-actor attribution from network telemetry alone is low-confidence | Low | Attribution shown as a weak prior over actor groups, never as a claim | Type posteriors from technique traces with known detection censoring |
| LIM-14 | Causal attention heads model influence but do not certify identified causal structure | Moderate | Causal masks enforce time order and protocol constraints; forensic replay labelled as model-based | Identifiability from defender interventions and protocol constraints; counterfactual replay with uncertainty |
| LIM-15 | Physical-consistency checks assume wire-level packet sizes; segmentation offload can exceed them | Low | Offload-aware size bounds; explicit status flag when offload is detected | Credal physical bounds that never exclude a truly possible observation |

---

## References

| Source | Citation |
|---|---|
| Compute-optimal training | Hoffmann et al. 2022, arXiv:2203.15556 |
| Activations | Korthikanti et al. 2022, arXiv:2205.05198 |
| Training state | Rajbhandari et al. 2020, arXiv:1910.02054 |
| NVIDIA H100 (989 TFLOP/s dense BF16) | NVIDIA H100 datasheet |
| DreamerV3 (straight-through categorical, KL free bits) | Hafner et al. 2023, arXiv:2301.04104 |
| Titans long-term neural memory | Behrouz et al. 2024, arXiv:2501.00663 |
| MPPI / TD-MPC2 | Williams et al. 2017; Hansen et al. 2024, arXiv:2310.16828 |
| RLCD | Damani et al. 2025, arXiv:2507.16806 |
| Conformal thresholds | Angelopoulos & Bates 2021, arXiv:2107.07511 |
| Process reward model | Lightman et al. 2023, arXiv:2305.20050 |
| Expected Gradients / SHAP | Erion et al. 2021, arXiv:1906.10670; Lundberg & Lee, NeurIPS 2017 |
| LIME | Ribeiro et al., KDD 2016 |
| Von Neumann graph entropy | Braunstein, Ghosh & Severini 2006, Annals of Combinatorics 10:291 |
| Spectral graph entropy | De Domenico & Biamonte 2016, Physical Review X 6:041062 |
| Multiplex redundancy | De Domenico, Nicosia, Arenas & Latora 2015, Nature Communications 6:6864 |
| Quantum Jensen-Shannon divergence | Lamberti et al. 2008, PRA 77:052311; Virosztek 2021, Advances in Mathematics 380:107595 |
| Stochastic Lanczos | Ubaru, Chen & Saad 2017, SIAM J. Matrix Anal. Appl. 38:1075 |
| Critical slowing down | Scheffer et al. 2009, Nature 461:53; Dakos et al. 2008, PNAS 105:14308 |
| Traffic entropy | Lakhina, Crovella & Diot, SIGCOMM 2005 |
| Miller-Madow entropy estimator | Miller 1955; Paninski 2003, Neural Computation 15:1191 |
| Chao-Shen entropy estimator | Chao & Shen 2003, Environmental and Ecological Statistics 10:429 |
| MulVAL attack graphs | Ou, Govindavajhala & Appel 2005, USENIX Security |
| Ogata thinning (point process) | Ogata 1981; Lewis & Shedler 1979 |
| ATT&CK Enterprise + ICS | MITRE ATT&CK (https://attack.mitre.org/) |
| D3FEND | MITRE D3FEND |
| Neural Granger causality | Tank et al. 2021, TPAMI |
| CVaR ranking | Rockafellar & Uryasev 2000 |
| Robustness (outlier process) | Black & Rangarajan 1996, IJCV 19(1) |
| Quantal response equilibrium | McKelvey & Palfrey 1995, GEB 10(1) |
| Numerical accuracy | Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed., SIAM 2002 |
| Dataset labelling errors | Engelen, Rimmer & Joosen 2021, IEEE SPW; Liu et al. 2022, IEEE CNS |
| Temporal snooping in ML | Arp et al. 2022, USENIX Security |
| Threefry RNG | Salmon et al., SC 2011 |
| Lost-in-the-middle | Liu et al. 2024, TACL, arXiv:2307.03172 |
| EBT training | Gladstone et al. 2025, arXiv:2507.02092 |
| MaskGIT | Chang et al. 2022, CVPR, arXiv:2202.04200 |
| TabDDPM | Kotelnikov et al. 2023, ICML, arXiv:2209.15421 |
| JEM energy acceptance | Grathwohl et al. 2020, ICLR, arXiv:1912.03263 |
| Energy-based novelty | Liu et al. 2020, NeurIPS, arXiv:2010.03759 |
