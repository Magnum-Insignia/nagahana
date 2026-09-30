# NagaHana architecture: canonical brief (2026-09-29)

This is the single reference for the *current* design. The codebase, diagrams, thesis document,
product site and story all follow it. Sources: `ai-mod-arch` (snapshot 2026-09-29 → quote IDs
A-xx), the owner's chat statements (Q-xx), the pipeline image (I-01), `ARCHITECTURE.md`,
`DESIGN_LOG.md`. Quotes are in `docs/sources/quotes.md` (kept in the private workspace, not published). Decision IDs (D-xx) and proposal IDs
(P-xx) are in `src/nagahana/governance/decisions.py`.

---

## 1. What NagaHana is
**NagaHana — Adversary Foundation Model for Simulation-based Threat Forecasting.** It is an AI world
model that learns the evolving state of a computer network from passive telemetry. It imagines where
an intrusion is heading before the kill chain completes, and it gives defenders interpretable,
calibrated forecasts and advisory counter-measures.
- **Scope:** enterprise and Critical Information Infrastructure (CII), **OT and IT together**, with CII as the priority (D-34).
- **Posture:**
  - passive only: taps, NICs and integrations; no host agents (D-32);
  - advisory only: humans decide and act (D-33);
  - fully offline.
- **Two settings of one engine:** live forecasting, and forensic replay of an uploaded capture ([Q-17], ARCH §7).
- **Why "adversary foundation model":** it models "conflict, coordination, cooperation, adversarial strategies, patterns, intentions, semantics of malignity, and benignity" [A-22].
- **Harness:** Wyvon (formerly Wyzre) is the separate deployment harness. It is not part of this model.

## 2. Names (D-19; [A-10], [A-11])
| Role / component | Name | Formerly |
|---|---|---|
| Role that builds and maintains the Environment | **Simulator** | — |
| Role that forecasts into Imagination | **Forecaster** | Renderer |
| Role that proposes counter-measures | **Advisor** | Planner |
| Regulator and calibrator | **Verifier** | — |
| Parallel output (memory → readable graph) | **Decoder** | — |
| Training-only augmentation | **Generator** | — |
| Encoder | **CVG-AE**: Complex Variational Graph AutoEncoder (multi-layer variational hypergraph GNN autoencoder with mathematical topological modelling) | "encoder/autoencoder" |
| Spatio-temporal-causal transformer | **TSTCT**: Topological Spatio-Temporal Causal Transformer | "STC transformer" |
| Energy-based transformer core | **TAAFT**: Topological Anti-Adversary Foundation Transformer | "EBT" |

Vocabulary: state model, state, state update, transition, entity state. **Never "token"** (D-31, [Q-25]).

## 3. The processing story (sequential, even where units run in parallel)
1. **Sense.** Passive sources produce records: pcap, NetFlow/sFlow/IPFIX, Zeek, Suricata, Snort, TAPs, routers/switches, custom hardware. Source adapters map each record into one **superset data model** (datamodel.md; OCSF/CSTS-compatible, D-06). The data model is human-auditable, Kafka-carried, and has temporal ordering resolved before the model [Q-20]. Each field carries an **observation status**. **Absence is "not supplied", never zero** (D-41).
2. **One record = one state update** (event-driven, D-30). Every update carries flow-level features, packet-level features, and further observable-region features such as DNS, Kerberos, TLS/JA4, Modbus, DNP3 and IEC-104 (D-38).
3. **Perceive: CVG-AE (Simulator).** The network is a **heterogeneous multiplex hypergraph**: typed entities, relation planes, hyperedges (D-39).
   - Each plane has its own parallel branch of hypergraph layers ("vertical" layers [A-01], [A-03]), each with its own sequence of hidden layers ("horizontal" [A-04]).
   - Branches exchange information every layer through learned cross-plane coupling.
   - A **variational** head produces the latent state z: hybrid, with continuous factors for rates, entropies and timings, and categorical factors for structure and stage (D-20; ARCH #16).
   - That latent space is shared by every role [A-13], [A-20].
4. **Physics boundary (global).** A shared term Φ_phys keeps every model output inside what is physically possible, so the model **cannot hallucinate impossible states** [A-08]. It covers reconstructions, imagined states, generated variants and predicted effects of counters.
   - It is a *boundary on learning*, **not a detector** of "impossible" attackers (D-18); attackers obey physics too.
   - Hard limits are built into the outputs. Soft residuals are counted only where their fields are observed.
   - It applies across components (D-37). The Decoder has none of its own.
5. **Organise: TSTCT (Simulator) → the Environment.** TSTCT reads the latent states and writes the **spatio-temporal-causal KV cache**. This is the **Environment**: the state model and its updates [A-12]. It uses three head types:
   - spatial: entities at their latest state as of t, biased by hypergraph topology;
   - temporal: an entity's own past, with no future leakage;
   - causal: learned, sparse cause → effect.

   CVG-AE and TSTCT are *perceptors, not analysers* [A-19].
6. **Show: Decoder (parallel output).** It decodes latents back into a readable network graph. This proves the perceptors work, gives a direct view into the Environment and, through the shared latent space, into Imagination [A-13]. Every element is tagged **observed / believed / forecast**. Beliefs never render as facts.
7. **Analyse: TAAFT (Forecaster) → Imagination.** TAAFT reads the Environment cache and writes **the KV cache of the analysis**: this is Imagination [A-14]. It is "the crux of this model being adversary foundation model" [A-19]. Its lenses:
   - **observability trust/distrust management and belief (suspicion) computation** (POMDP/POSG): what cannot be seen is not assumed absent;
   - **the energy-based transformer as its core**: an energy landscape of the network; refinement by descending energy; inference-time scaling;
   - **game theory and mechanism design** (team-level two-sum POSG [Q-12]);
   - **information theory and SNR / noise analysis** (coloured vs white noise, held D-26);
   - **graph and mathematical topology**, with light social-network analysis;
   - **time series and temporal analysis**;
   - **causal inference**.
8. **Forecast: Forecaster policy/value.** Policy and value functions, MPC-guided model-based deep RL [Q-28], plan as the adversary would. They imagine **K future states over N samples** (K and N are chosen at runtime, D-30), with a process reward per imagined step [Q-24]. How TAAFT and the heads couple (joint / separate / staged) is held (D-12). Outputs are listed in §6.
9. **Advise: Advisor.** A second policy/value agent works on **both** caches (D-01, [A-15], [A-20]). It searches for **D3FEND-based counter-measure sequences** at graph or sensor level. They are re-imagined against the Forecaster's adversary and ranked by risk reduction, disruption and feasibility. **Advisory only.** It relates to the Forecaster like a GAN in function only [Q-38]. The Advisor keeps no memory of its own.
10. **Govern: Verifier.** The regulator. It stores human feedback (step labels on imagined steps, outcome labels, "responded-to" tags) as **supplied truth** [A-21]. It calibrates forecasts (RLCD, the calibrated-decisions standard [Q-34]; conformal thresholds). It computes **memory drift from outcome–forecast pairs** in the **Monitor** region, which makes poisoning visible [A-16].
    - It observes online, but **changes weights only on a human's supervised command** (D-21).
    - It never feeds back automatically, because that "would conduct the poisoning again" [A-16].
11. **Augment (training only): Generator.** It makes label-preserving variants of real events by observability sliding, signatures, traces and impacts [A-17], [Q-37]. The variants are physics-bounded ("to not hallucinate impossible ones"). Methods: energy-based SSL, Joint Energy Models, generative training that fills in missing parts, autoregressive and diffusion methods. Which families come first is held (D-14). Variants are mixed into the real training batches.

## 4. Memory (D-35, D-36)
|  | Environment | Imagination | Monitor |
|---|---|---|---|
| holds | observed facts + transition history (TSTCT KV cache) | belief/suspicion + forecasts (TAAFT analysis cache) | deviations, variances, drift |
| Simulator | R W | – | – |
| Forecaster | R | R W | – |
| Advisor (memory-less) | R | R | – |
| Verifier | R | R | R W |
| Decoder | R | R | (R: proposal P-12) |
| Generator | – | – | – |

- Retention follows a **fixed schedule per Forecaster trigger** and never depends on input volume (D-36).
- What fires a trigger is held (D-02).
- Whether the KV cache is the durable Environment or a view rebuilt from an append-only event log is held (D-15; proposals P-02, P-18).

## 5. Training pipeline (D-22, D-23; [I-01], [A-24])
1. **Deep data analysis** of raw open datasets, documented [Q-02].
2. **Preparation:** cleaning, splitting, curation of attacks, normalisation, augmentation with the Generator.
   - Splits: training and validation use **real + generated** data; **zero-shot uses real only**; novel and known attacks are evaluated **separately**.
3. **Self-supervised pretraining of CVG-AE, Decoder and TSTCT:** the intuition of network topology and its dynamics.
4. **Self-supervised pretraining of TAAFT** with those frozen: the intuition of adversaries, conflict, cooperation, coordination, strategies, intentions, malignity and benignity.
5. **Full training** of the whole architecture, including the Forecaster and Advisor policy/value agents. The Verifier is trained on human feedback.
6. **Zero-shot validation with calibration:** novel and known attacks separately. Human-supervised calibration adapters settle the model to each site's normality.

Run modes: train · evaluate · live inference · forensic replay · lab.

## 6. Outputs
**Forecast (per trigger)**
- P_inf(k): infiltration probability over the next K steps, with uncertainty.
- The predicted MITRE ATT&CK stage per step. The problem statement names Reconnaissance, Initial Access, Lateral Movement, Command & Control and Exfiltration; this extends to full Enterprise + ICS.
- The top-N imagined attack paths with probabilities.
- A time-to-event hazard curve ("when").
- Driving features per prediction (attention and attribution): flags, ports, flow patterns.
- The compute spent (inference-time scaling).
- The safe horizon, and observability-gap annotations.

**Belief and trust**
- Per-entity compromise belief (suspicion, with an assume-breach floor).
- Posteriors over the adversary's goal, stage and type.
- Trust and reliability per telemetry source and field.

**Energy**
- Energy per entity and per time.
- Energy and entropy growth as an early warning.
- Novelty (high energy means unfamiliar, not proof of attack).

**Noise and signal**
- Periodicity and beaconing evidence, and self-similarity (Hurst) of traffic (if D-26 is adopted).

**Advisory**
- Ranked D3FEND counter-measure sequences: steps, D3FEND tactic, level (graph or sensor).
- For each: ΔP_inf, disruption cost, physical feasibility, and information value.

**Verifier**
- Calibration report (reliability, ECE, Brier).
- Drift report per memory region.
- Outcome–forecast ledger.
- Poisoning alerts for human review.

**Decoder view**
- A live network graph with observed / believed / forecast provenance.

**Forensic replay (uploaded capture)**
- Timeline of what would have been forecast, and when the kill chain became visible.
- Kill-chain narrative, ATT&CK-tagged.
- Patient-zero trace.
- Counterfactual replay ("if isolated at 14:20…").
- Observability gaps, and signs of anti-forensic tampering.

## 7. Evaluation methods and metrics
**Required by the problem statement:** F1, precision, recall and false-positive rate against a logistic-regression baseline trained on the same features. We also report the false-negative rate and the detection error (FP + FN) in operational settings.

| Component | Method | Metrics |
|---|---|---|
| Detection/forecast (overall) | train on known, test on real zero-shot; novel and known reported separately; cross-dataset and leave-one-network-out (D-16) | F1, precision, recall, FPR, FNR, detection error; AUROC, AUPRC |
| Forecast probability | proper scoring; reliability analysis | Brier score, log score, CRPS (paths), ECE, reliability diagrams; Brier/CRPS **skill scores** vs persistence and climatology baselines |
| Lead time | time before stage completion at which P_inf crosses the alert threshold | median lead time; lead time at a fixed FPR |
| When (time-to-event) | survival analysis with censoring | concordance index (C-index), time-dependent AUC, integrated Brier score |
| ATT&CK stage | per-step stage prediction | top-k accuracy, macro-F1 per stage, confusion by stage |
| Attack paths | set/sequence match of imagined vs realised paths | path precision/recall@N, edit distance, ranking quality (Kendall τ; ordinal safety, ARCH §3.3) |
| CVG-AE | masked reconstruction on observed fields; latent health | reconstruction error, edge AUROC/AP, KL, active latent units; OOD AUROC (energy / likelihood-ratio, not raw ELBO) |
| TSTCT | next-latent prediction; mask audits | latent prediction error, no-future-leak test, attention-mask audit |
| Decoder | fidelity and provenance | decoded-vs-observed error per plane; zero believed-as-observed renderings |
| TAAFT belief/trust | on simulated worlds with known hidden state (P-14) | belief Brier/log score; trust AUROC on injected corrupted telemetry |
| Energy | novelty and early warning | OOD AUROC; early-warning lead time; false-alarm rate of energy alerts |
| Game / Advisor | re-imagination and counterfactual replay | ΔP_inf per counter, disruption cost, feasibility rate, regret vs oracle (simulator), analyst acceptance |
| Verifier | calibration and drift | ECE/Brier before vs after; drift-detection delay at a fixed false-alarm rate (QCD); injected-poisoning detection rate |
| Generator | fidelity, diversity, utility, safety | MMD/Wasserstein to real; physics-violation rate; label preservation; train-synthetic-test-real (TSTR); no zero-shot leakage |
| Physics boundary | hallucination control | violation rate of model outputs (reconstructions, imagined steps, variants) |
| Operations | deployment realism | latency per state update, throughput, memory growth, behaviour under NetFlow-only vs full telemetry, robustness to telemetry loss |
| Forensics | replay on labelled incidents | stage-onset timing error, patient-zero accuracy, narrative step precision/recall |

## 8. Status of decisions (see `governance/decisions.py`)
**Decided:** D-01, D-07, D-09 (the engineer's reading of the owner's words; to confirm), D-17 to D-23, D-30 to D-41.

**Held (never default):**
- D-02 trigger policy
- D-03a–c success criterion, disruption pricing, expected vs worst case
- D-04 plane formation
- D-05 view flattening
- D-06 standards
- D-08 threat-model document
- D-10 repo location
- D-11a–c reconstruction target, energy jobs in v1, adversary objective
- D-12 TAAFT coupling
- D-13 calibration adapters
- D-14 Generator families
- D-15 Environment persistence
- D-16 zero-shot definition
- D-24 mechanism-design placement
- D-25 physics on telemetry
- D-26 noise analysis
- D-27 Kafka client
- D-28 demo interface
- D-29 status weights

**Proposals:** P-01 to P-23.

## 9. Writing rules for every artefact
- Public artefacts never name events, programmes or clients.
- Vocabulary: states, never tokens.
- Physics is a **boundary against hallucination**, never "detects the impossible".
- Human feedback is trusted supplied truth. Telemetry is inside the threat model (D-17).
- **No fabricated results.**
  - Never state a number as measured unless it was measured.
  - Published baselines are cited to their source.
  - Targets are labelled as targets.
- Held decisions are described as configurable or site-level settings, never as a chosen option.
