# Glossary

| Term | Meaning |
|---|---|
| **state model** | The schema of the world state (datamodel/). |
| **state** | An instance of the state model: the world at a time. |
| **state update** | One telemetry record, which updates part of the state (event-driven; D-30). Never "token" [Q-25]. |
| **transition** | The change a state update causes. |
| **observation status** | Evidence status of a field: observed, stale, low reliability, not supplied, not observable (P-03). Absence ≠ zero (D-41). |
| **heterogeneous multiplex hypergraph** | The world as typed entities, several relation planes, and hyperedges joining ≥ 2 entities (D-39). |
| **relation plane** | One relation layer of the hypergraph, e.g. connectivity or identity. Planes are the model's "vertical" parallel branches [A-01]. |
| **relation-specific parallel branches** | Proposed standard term for the owner's "MIMD-type" architecture (P-08). |
| **CVG-AE** | Complex Variational Graph AutoEncoder: the encoder and latent-space generator [A-11]. |
| **TSTCT** | Topological Spatio-Temporal Causal Transformer: writes the Environment [A-12]. |
| **TAAFT** | Topological Anti-Adversary Foundation Transformer: the analysis core that writes Imagination [A-14], [A-19]. |
| **Environment** | Memory of observed facts and transitions (TSTCT's KV cache). Written only by the Simulator. |
| **Imagination** | Memory of belief, suspicion and forecasts (TAAFT's analysis cache). Written only by the Forecaster. |
| **Monitor** | Memory of drift and deviations. Written only by the Verifier. |
| **belief / suspicion** | Belief = the POSG distribution over the hidden state. Suspicion = its adversarial part, never zero (assume-breach floor) [Q-10]. |
| **physics boundary** | The shared term Φ_phys plus hard limits, which keep model outputs physically possible. A guard against hallucination, not a detector (D-18) [A-08]. |
| **K, N, H** | Steps ahead, imagined samples, and the MPC plan length (H ≤ K). All set at runtime. |
| **P_inf(k)** | Probability that the first infiltration happens by step k (cumulative, non-decreasing). |
| **process reward** | A reward for each imagined step, not only the outcome [Q-24]. |
| **RLCD** | Reinforcement Learning for Calibrated Decisions: the calibration standard the Verifier follows [Q-34]. |
| **supplied truth** | Human feedback held by the Verifier. Trusted, and outside the threat model (D-17) [A-21]. |
| **responded-to** | An outcome that changed because people acted on a forecast. Not scored as an error [Q-38]. |
| **D3FEND** | MITRE's defensive-technique framework. The Advisor's vocabulary: Model, Harden, Detect, Isolate, Deceive, Evict, Restore. |
| **held decision** | A design question the owner has not decided. Code refuses to choose for them [Q-39]. |
| **proposal (P-xx)** | An engineering suggestion awaiting approval. It runs only when enabled. |
