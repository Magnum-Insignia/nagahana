# Known limitations and ongoing work

This page lists the known limitations of NagaHana, how each one is mitigated today, and the work in
progress to remove or reduce it. Severity is the operational impact on a deployment: Low, Moderate or
High. Items are revised as the work lands.

## Observation and data

| # | Limitation | Severity | Current mitigation | Work in progress |
|---|---|---|---|---|
| L-01 | Encrypted payloads (TLS 1.3, QUIC) and encrypted SNI hide content-level evidence. The observable surface shrinks as these protocols spread. | Moderate | The model uses the observables that survive encryption: timing, sizes, flow structure, graph structure and handshake fingerprints. Every field carries an observation status, so unobservable is never read as benign. | Better use of graph-frequency and timing statistics for slow attacks. |
| L-02 | Segments seen from only one vantage point: sensor reliability there cannot be cross-checked, so a spoofed value is harder to discount. | Moderate | Such segments are marked with a low-reliability observation status, and their evidence weight is reduced. | Reliability estimation from overlapping vantages and physics consistency, with a bound on the influence of compromised sensors. |
| L-03 | Public training corpora contain known labelling errors, and their attack campaigns are emulated rather than observed in production networks. | Moderate | Corrected label sets are used where they exist. Leakage and duplicate audits run before training, and known and novel attack families are reported separately. | A ground-truth world simulator with known hidden state (P-14) and an information audit that measures the best achievable accuracy per observation regime (P-15). |
| L-04 | Coverage of some industrial protocols (PROFINET, EtherNet/IP CIP, BACnet, ICCP, C37.118) is thinner than for Modbus, DNP3 and IEC 60870-5-104. | Low | Unknown industrial protocols are still modelled at the flow and timing level, with their protocol fields marked "not supplied". | Field mappings and simulator scenarios for the remaining protocols. |
| L-05 | Out-of-order and lossy telemetry: when sensors drop or reorder records, the reconstructed timeline can be incomplete. | Low | Bounded reordering with event-time watermarks; residual ordering uncertainty and loss are carried as statuses rather than silently corrected. | Per-sensor loss estimation feeding the reliability model. |

## Deployment and operations

| # | Limitation | Severity | Current mitigation | Work in progress |
|---|---|---|---|---|
| L-06 | Cold start: for the first days a site has no established normal behaviour, and a network that is already compromised at deployment can be learned as normal. | High | Population priors from pretraining, a low-confidence early window, and checks on attack-objective invariants (exfiltration, lateral reach, command channels) that do not depend on the site's baseline. | Calibration that remains valid with few site events. |
| L-07 | Rare legitimate events (quarterly jobs, new deployments, rare admin actions) can resemble low-and-slow attacks. | Moderate | An alert requires rarity together with progress toward an attack objective, not rarity alone. Human feedback through the Verifier calibrates the site's threshold. | A memory write rule and evidence measure tied to objective progress. |
| L-08 | Hardware: full inference of the L model needs about 4 accelerators with 80 GB each. | Moderate | Weights and caches are fp32 with fp64 outputs (D-54); the compute and memory budget is documented per component (`docs/sizing.md`). | Sparse contact representations and vectorised hot paths to reduce memory and latency. |
| L-09 | Per-entity memory grows with the number of hosts times the retention period, which is heavy at very large critical-infrastructure sites. | Moderate | Log-time dyadic memory cells that bound storage per entity, with flood-invariant updates. | Retention guarantees under adversarial flooding, and an adaptive-resolution memory. |
| L-10 | PCAP parsing is a per-packet Python loop, so line-rate capture needs flow-level sensors (Zeek, NetFlow, IPFIX) in front. | Low | Flow-state emission from PCAP (D-51), and native ingest of Zeek, Suricata, NetFlow, IPFIX and sFlow. | A sandboxed parser process with input caps and bounded parser state. |

## Model and explanation

| # | Limitation | Severity | Current mitigation | Work in progress |
|---|---|---|---|---|
| L-11 | An adaptive adversary can change its behaviour once it notices defensive responses, which shifts the data the forecasts are judged on. | Moderate | Decision support only: humans choose the response. Calibration excludes responded-to cases. | Counterfactual calibration with respect to the known response policy, and response options that do not reveal detection. |
| L-12 | Explanations attribute an internal quantity close to, but not identical with, the displayed infiltration probability, and explanation cost grows with descent depth. | Moderate | Lens shares, evidence chains and observability-gap annotations are shown alongside every forecast. | Attribution of the displayed probability itself, integrated over the whole descent path, with faithfulness tests (D-60). |
| L-13 | Threat-actor attribution from network telemetry alone is low-confidence. | Low | Attribution is shown as a weak prior over actor groups, never as a claim. | Type posteriors from technique traces with known detection censoring. |
| L-14 | The causal attention heads model influence between entities but do not yet certify identified causal structure. | Moderate | Causal masks enforce time order and protocol constraints. Forensic replay is labelled as model-based. | Identifiability from defender interventions and protocol constraints, and counterfactual replay with uncertainty (D-59). |
| L-15 | Physical-consistency checks assume wire-level packet sizes, which segmentation offload on host-side captures can exceed. | Low | Offload-aware size bounds and an explicit status flag when offload is detected. | Credal physical bounds that never exclude a truly possible observation. |

## Reporting a limitation

New limitations found in review or deployment are added here with a severity, the current mitigation
and the planned work, and are linked to the decision (D-xx) or assumption (AS-xx) they affect.
