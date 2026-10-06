# NagaHana L: build specification (2026-10-02)

This is the engineering specification of the full NagaHana model at the L size (1,134,268,667
parameters, 1.13 B, Generator excluded; `docs/sizing.md`). It turns the canonical brief (`docs/architecture.md`) and the ADRs into
something buildable. It runs from philosophy, to logic, to maths, to code contracts.

**How decisions appear here**
- **D-xx:** decided by the owner. Code relies on it.
- **AS-xx:** an engineering assumption the owner authorised for this build (owner, 2026-10-02: "with
  your assumptions of wherever if you think are missing more details"). Each one lives in
  `governance/assumptions.py` and `docs/assumptions.md`, with its reasoning and evidence. The held
  decision it stands in for stays **held** in `governance/decisions.py`: an assumption is this build's
  working choice, never the owner's decision.

---

## 0. Philosophy: what a network is, and what the model must be

1. **A network is a partially observed, adversarial dynamical system.**
   - The defender sees traffic, never intent.
   - The attacker sees part of the network and acts to move through it.
   - So the object to model is not "is this flow bad" but the joint process (network state, adversary
     state) and how it evolves: P(S_{t+1} | S_≤t).
2. **Epistemic separation is a type, not a habit.**
   - What was observed (Environment), what is believed or imagined (Imagination) and how far both
     can be trusted (Monitor) are different memories with enforced access (D-35).
   - A belief can never be read back as an observation. This shows in the dataflow, which is one-way:
     CVG-AE → TSTCT → TAAFT. The only loops are each model looping on itself (D-43).
3. **Absence is not zero (D-41).**
   - Every value carries its observation status. The model learns from "not observable here" as a
     fact about the sensor, never as a value of the traffic.
4. **Physics is a boundary, not a detector (D-18).**
   - Anything the model *produces* (reconstructions, imagined states, generated variants, predicted
     effects of counters) is kept inside the physically possible.
   - This is done by construction where a limit is exact, and by a residual penalty where it is only
     checkable on observed fields.
5. **Thinking is a budget, not a constant.**
   - Four knobs, all set at run time and recorded with each forecast (D-44): loop passes R, energy
     descent steps S, horizon K and routes N.
   - The design must make each knob safe to raise beyond what training used.
6. **Explanations are built in, not bolted on.**
   - The energy is a sum of named lens terms (D-42), so every refinement step decomposes by lens.
   - Attention is typed (spatial, temporal, causal).
   - Field attributions reach back to the field-state embeddings of the input layer.
7. **Nothing an attacker controls may set the model's clock or forgetting.**
   - No index-based positions (D-49), only time.
   - Retention is per trigger, not per update (D-36).
   - Bounded memories merge floods inside their own time bucket instead of evicting history (AS-11).

---

## 1. Logic: the processing story as computation

```
ColumnarUpdates ──► FieldEncoder ──► update vectors u_i
                         │
   graph builder ───────►  local hypergraph as of t (per position) ──► CVG-AE ──► z (posterior)
                                                                                     │
                                         TSTCT memory stream (1 pass) ──► Environment K/V cache
                                         TSTCT thinking stream (R passes, reads memory) ──► refined e
                                         TSTCT prior head ──► p(z_next | Environment)   (world-model transition)
                                                                                     │
   trigger (cadence, AS-12) ──► TAAFT (entity tokens + adversary slots; reads Environment K/V,
                                       Imagination K/V of past triggers; R passes) ──► context c
                                ──► hypothesis y0 ──► energy descent on E_total (S steps) ──► ŷ
                                ──► belief readouts (compromise, stage, goal, type, trust)
                                ──► Imagination K/V written (memory stream)
   Forecaster (dynamics transformer; policy π_A over (technique, target); MPPI routes K × N)
                                ──► P_inf(k), stage per step, hazard, routes, back-projected z → Decoder → Φ_phys
   Verifier (PRM over imagined steps, trust value head, calibration policy head, drift in Monitor)
   Advisor (on demand: D3FEND counters re-imagined against the Forecaster; CVaR ranking)
   Decoder (any z → fields and candidate hyperedges, provenance-tagged view)
```

### 1.1 Units of computation
- **State update** u_i: one record (one row of `ColumnarUpdates`), at event time t_i, touching
  entities (initiator, responder, service).
- **Entity state** (a TSTCT position): created for the initiator and for the responder of every update
  (2 positions per update, as in the compute profile).
  - Its latent z comes from CVG-AE applied to the entity's local hypergraph as of t_i.
  - The service entity, when present, is a graph node but gets no position of its own (AS-41).
- **Trigger**: a moment at which the Forecaster computes. Fixed cadence plus capped priority triggers
  (AS-12, standing in for held D-02).
- **Imagined step** k = 1…K: one window of `window_seconds` into the future, along one route n = 1…N.

---

## 2. Maths and code contracts by component

Notation:
- B batch, U updates, P positions, V entities, d model width;
- H heads, d_h = d/H head width;
- z = (z_c ∈ ℝ^{Dc}, z_d ∈ (Δ^C)^G), with dz = Dc + G·C.

### 2.1 Input layer: `models/inputs` (FieldEncoder)
For each update i and each column c of the value/status matrices:

    f_{i,c} = s_c + σ_{status(i,c)} + 𝟙[c contributes] · v_{kind(c)}(x_{i,c}) + 𝟙[not] · a_{status}

- s_c: field-slot embedding (D-49), keyed by column.
- σ: status embedding over the 5 statuses plus a MASK code used by self-supervised masking.
- a_status: a learned "no value" vector. NaN never enters arithmetic (D-41).
- v_kind, by kind:
  - CONTINUOUS / COUNT: signed log1p, then periodic embedding [sin(2πw·x̃), cos(2πw·x̃)] with learned
    frequencies, then a linear map (Gorishniy et al., NeurIPS 2022, arXiv:2203.05556) (AS-31);
  - CATEGORICAL (ports, protocol): a hashed table row h(c, code) mod R_hash (P-22, AS-33);
  - BITMASK: the sum of per-bit embeddings;
  - HISTOGRAM bins: one column each (columnar form).
- Update vector: u_i = AttnPool_q({f_{i,c}}_c), with a learned query and H_pool heads.
  - The pooling weights are the first explanation layer: which fields this update was "about".
- Clock features (time of day, day of week): built, config switch `clock_features`, off in training
  (D-50).

### 2.2 Graph builder: `graph/` (planes, hyperedges, local subgraphs, RWSE)
- **Planes** (AS-01, standing in for held D-04): six declared planes from protocol and port evidence:
  - `connectivity`: every flow;
  - `services`: L7 sessions through a service entity;
  - `identity`: Kerberos 88, LDAP 389/636, SMB 445 auth, NTLM-bearing;
  - `remote_admin`: SSH 22, RDP 3389, WinRM 5985/5986, VNC 5900, Telnet 23;
  - `name_resolution`: DNS 53, LLMNR 5355, NBT-NS 137, mDNS 5353;
  - `ot_control`: Modbus 502, DNP3 20000, IEC-104 2404, S7 102, EtherNet/IP 44818, BACnet 47808,
    OPC UA 4840.
- **Hyperedge kinds per plane** (AS-02):
  - `session` (TCP);
  - `exchange` (UDP and other);
  - `group` (a multicast entity is a member, D-47);
  - `fan`: one initiator touching ≥ f distinct responders within Δ_fan. This is a scan or sweep,
    built as one hyperedge, time-stamped by its first member.
- **As-of discipline (no future leakage):** every hyperedge carries `since` (the event time of the
  update that created it). A subgraph "as of t" uses only hyperedges with since ≤ t, and node
  features from updates with t_j ≤ t.
- **Local subgraph per position:** BFS from the position's entity, up to `cvg_layers` hops, capped at
  `max_nodes` (32) by recency of contact. It is mini-batched as a disjoint union (`GraphBatch`).
- **RWSE (D-49):** for each plane, the clique expansion's random-walk matrix M = D⁻¹A on the local
  subgraph, and per node the diagonal of M^k for k = 1…k_rw (Dwivedi et al. 2022). This is
  permutation-equivariant by construction.
- **Contact matrices for TSTCT and TAAFT masks:**
  - C¹[u, v]: first time u and v shared a hyperedge on any plane (+∞ if never);
  - C²[u, v] = min_w max(C¹[u, w], C¹[w, v]): the first time a 2-hop path existed (a min–max
    "tropical" product);
  - plane bits Π[u, v]: the planes on which they have shared a hyperedge, as of the first contact
    (kept time-stamped per plane).

### 2.3 CVG-AE: `models/cvgae` (production entry `hgnn-attn`)
Per plane p, per layer ℓ, typed attention hypergraph layer (HGNN two-stage, with HGT-style typed
parameters):

    m_e   = Σ_{u∈e} softmax_u( a_{τ(e)}ᵀ W^K_{τ(e)} h_u ) · W^V_{τ(e)} h_u              node → hyperedge
    g_v   = Σ_{e∋v} softmax_e( (W^Q h_v)ᵀ m_e / √d + b_hop + b_size(|e|) ) · m_e          hyperedge → node
    c_v   = Σ_{q≠p} ω_pq h_v^{(ℓ,q)}                                                       cross-plane coupling
    h_v^{(ℓ+1,p)} = h_v + MLP_{p,τ(v)}( RMSNorm([h_v ; g_v ; c_v ; ρ_v^p]) )                 typed residual update

- ρ_v^p: the RWSE of v on plane p. b_hop: a learned bias by hop distance from the centre (D-49).
- Segment softmaxes use scatter with an `amax` shift, so they are stable on sparse incidence.
- Variational head on the centre node (concatenated planes): μ, log σ² ∈ ℝ^{Dc} and logits ∈ ℝ^{G×C}.
- Posterior sample:
  - Gaussian reparameterisation;
  - straight-through categorical with 1 % uniform mix ("unimix", DreamerV3, Hafner et al. 2023,
    arXiv:2301.04104) (AS-05).
- Input per node:
  - [update vector of the node's latest update as of t (or a learned "unseen" vector); role
    embedding; kind embedding; age encoding log(1 + age)];
  - the centre also gets its own update.

### 2.4 Decoder: `models/decoder` (parallel output)
- p_ψ(x | z, role) per column:
  - continuous/count: heteroscedastic Gaussian in signed-log1p space;
  - bitmask: per-bit Bernoulli;
  - categorical: softmax over service-class buckets (AS-33).
- Only contributing cells enter the likelihood (P-20, assumed in AS-04).
- Hard limits (`physics/constraints.py`), applied when decoding to raw space: counts ≥ 0 (softplus in
  log space), duration ≥ 0, iat_max ≥ iat_mean (ordered pair), flag counts ≤ packets.
- Candidate hyperedges per plane: p(e ∈ ℰ_p | z_members) = σ(MLP_p(DeepSets(z_u : u ∈ e))).
  - Candidates are the observed hyperedges plus sampled negatives (members swapped).
- View (D-05 held, AS-29): a typed multigraph, planes kept as edge types. Every element is tagged
  OBSERVED, BELIEVED or FORECAST. A FORECAST or BELIEVED element can never carry the OBSERVED tag;
  this is a tested invariant.

### 2.5 TSTCT: `models/tstct` (Environment)
- Input: e⁰_j = W_in z_j (the posterior sample during training; the posterior mean at inference, AS-05).
- Blocks: 16 pre-norm blocks (RMSNorm, SwiGLU; AS-32). Each block has H = 16 heads split by type
  (AS-09):
  - spatial (4 heads): keys are the *latest state as of t_i* of entities at hop ≤ 2 as of t_i
    (C¹, C² ≤ t_i), plus the state itself.
    - Bias: b_sp(hop) + b_planes(Π) + b_age(log(1 + t_i − t_j)).
    - Cap: the 64 most recent neighbours.
  - temporal (8 heads): keys are the same entity's states with t_j ≤ t_i, up to the last 512.
    - Continuous-time rotary encoding of q and k by t (D-49), plus a log-Δt bucket bias.
  - causal (4 heads): candidates are states j of *other* entities with t_j < t_i (strict),
    t_i − t_j ≤ Λ_lag, and contact C¹ ≤ t_i.
    - A learned gate g_ij = σ(MLP_g([q_i ; k_j ; φ(Δt)])) enters as an additive log g_ij.
    - The top 32 candidates by g are kept, and an L1 penalty on g makes the pattern sparse.
    - Time rotation as for temporal heads.
    - This is Granger-style lagged influence (Tank et al., TPAMI 2021, "Neural Granger Causality").
      It is **not** a claim of identified causal structure (AS-10).
- **Two-stream weight-tied loop (AS-06; D-43, D-44):**
  - Memory stream, 1 pass: m = Stack(e⁰) with the masks above. Its per-block (K, V) are the
    **Environment cache**.
  - Thinking stream, R passes: h^{r+1} = Stack_read(h^r + e⁰). This is input re-injection; no pass
    embedding (D-49).
    - Every attention reads the memory stream's K/V of allowed keys (fixed across passes), with
      queries from the thinking stream.
    - Diagonal (self) keys also come from the memory stream.
  - Output: refined e = h^R.
  - **Why:** training and inference are *exactly* the same computation. The cache does not depend on R
    (this settles ADR-0008's open note on which pass's K/V is cached). Raising R at run time changes
    only the thinking stream.
- **Transition prior (the world model; AS-05):**
  - A head on the memory stream at position j predicts the latent of the *same entity's next state*:
    p(z_{next(j)} | Environment ≤ t_j) = 𝒩(μ_p, σ_p²) × Cat(ℓ_p).
  - With Δt to the next state as an input, so the prior knows how far ahead it predicts.
  - This is exactly the problem statement's P(S_{t+1} | S_t), learned by the KL of §3.
- **Equivalence test (required):** the dense masked training path and the gathered-key KV-cache
  inference path give the same outputs to 1e-5 (fp32), for R ∈ {1, 3}.

### 2.6 Memory: `memory/` (Environment store, long-term memory, Imagination store)
- **Environment store (AS-11; D-15 held):**
  - Per entity, a log-time bucketed buffer of (K, V) per block.
  - Bucket b covers ages [2^b, 2^{b+1}) · δ, with at most m_b slots per bucket.
  - When a bucket overflows, its two most similar slots are **merged**: the K/V mean, with a count n.
    Attention then adds log n to the merged key's logit.
  - This is exact for identical keys (softmax over n copies = one key + log n), and it never evicts
    older buckets because of volume. A flood compresses inside its own bucket (D-36 spirit).
  - Capacity is the 512 states per entity of the working context.
- **Long-term memory:**
  - A Titans-style neural memory (Behrouz et al. 2024, arXiv:2501.00663):
    M_k = (1 − α_k) M_{k−1} + S_k and S_k = η S_{k−1} − θ ∇ℓ(M_{k−1}; k_k, v_k), updated **once per
    trigger** with α_k from `RetentionSchedule.alpha(k)` (D-36).
  - It is read by TAAFT as extra memory keys.
- **Event log:** `ColumnarUpdates` is the durable record. Caches are rebuildable views keyed to the
  model hash (P-18, assumed in AS-11).
- **Imagination store:**
  - TAAFT memory-stream K/V per (entity or adversary slot) for the last M_im = 8 triggers;
  - plus the trigger's forecasts.
  - Access is checked through `memory/access.py`.

### 2.7 TAAFT: `models/taaft` (Imagination; the adversary-foundation core)
- **Tokens at trigger τ:**
  - one per active entity v (initialised from its latest refined TSTCT state as of τ);
  - plus G_adv = 16 adversary-hypothesis slots (learned slot embeddings, D-49).
- **Blocks:** 24 decoder-style blocks: self-attention, then cross-attention to the Environment, then
  SwiGLU.
  - **Self-attention:** over the current tokens (topology bias from C¹/C², Π as of τ) plus each
    token's own Imagination K/V from previous triggers (time-rotary by trigger time). Belief recursion
    b_τ ← b_{τ−1} *is* this attention.
  - **Cross-attention:** reads **TSTCT's cached K/V directly** (block map b ↦ ⌊b·16/24⌋), with
    TAAFT's own query and output projections (AS-13).
    - An entity token reads its own last 32 states and its neighbours' latest states (≤ 64).
    - Adversary slots read a learned null key only. They see entities through self-attention.
  - Every attention has a learned null key/value (no all-masked rows).
- **Two-stream loop** exactly as TSTCT. The memory stream writes the Imagination K/V.
- **Hypothesis space (AS-14):**
  - y_v ∈ ℝ^{d_y} per entity (d_y = 256) and y_A ∈ ℝ^{G_adv × d_y}.
  - Readouts from y:
    - compromise probability with the assume-breach floor:
      p_v = φ + (1 − φ)·σ(w·y_v), with φ = `suspicion_floor` (> 0, ARCH §4.7);
    - stage posterior over 15 classes (AS-19);
    - next-latent parameters (for the physics term and the Decoder);
    - telemetry trust per source (σ);
    - goal and type posteriors from y_A (P-13, assumed).
- **Energy (D-42):** E_total(c, y) = Σ_ℓ E_ℓ(c, y) + λ_phys Φ_phys(decode(y)), with the lens terms
  (AS-14):
  - **belief-and-trust:**
    E_bt = Σ_v [ −log p(evidence_v | y_v) under trust t_v ] + Σ_v KL-style penalty of y_v away from
    the TAAFT prior.
    - The evidence term is a learned compatibility ψ_bt(c_v, y_v), weighted by the trust readout.
  - **game** (mechanism design inside, D-24 held):
    E_game = −U_A(y_A, {y_v}), a learned adversary utility.
    - Hypotheses in which the adversary's inferred position is consistent with a rational progression
      get low energy.
  - **information** (noise features inside, D-26 held, AS-36):
    E_info = ψ_info(c_v, y_v, ν_v), where ν_v are per-entity SNR/periodicity statistics.
    - ν_v are a periodogram peak ratio of inter-arrival times and an aggregated-variance Hurst
      estimate.
    - The term penalises hypotheses not supported by informative evidence.
  - **topology:** E_top = Σ_{(u,v): C¹ ≤ τ} w_uv φ_top(y_u, y_v). This is a pairwise MRF on the
    contact graph: compromise propagates along edges.
  - **time:** E_time = ‖y_v − f_time(y_v^{prev}, Δτ)‖²_{Σ}, compatibility with the previous trigger's
    hypothesis, read from Imagination.
  - **cause:** E_cause = Σ_{u→v} ḡ_uv φ_cause(y_u, y_v), with ḡ the TSTCT causal gates averaged over
    the window. This term is directed.
  - **physics:** Φ_phys on the decoded next state (hard limits by construction; soft residuals scaled
    by λ_phys, AS-15).
- **Refinement:**
  - y_0 = W_y c (amortised guess);
  - ŷ = `energy_descent(E_total, y_0, steps=S, step_size=α, noise=σ_i)`, with α learned (softplus)
    and σ_i annealed.
  - Per-lens contributions are recorded at every step (D-42 explanation).
- **One network, two readings (P-09, assumed in AS-16):**
  - during training the context is dropped with probability 0.1 and replaced by a learned null
    context;
  - E(∅, y) is the marginal "is this normal" energy, used for novelty and Advisor shaping.

### 2.8 Forecaster: `models/forecaster`
- **Context tokens:** the 16 adversary slots plus the top 48 entities by compromise belief: 64 tokens.
- **Dynamics:** 8 pre-norm blocks over [context ; route steps], with causal masking over route steps.
  - Step k carries s_k plus an action embedding.
  - Time rotation with t = k · window_seconds (D-49).
- **Policy π_A(a | s_k)** factorises: technique a_tech over 700 slots (AS-20) × target a_tgt (a
  pointer over context entity tokens).
- **Per-step heads:**
  - stage posterior (15);
  - hazard h_k = P(infiltration at k | none before) ∈ (0, 1);
  - value V(s_k);
  - process reward r̂_k;
  - back-projection to z of the target entity (decoded and physics-checked).
- **MPC-guided imagination (AS-21):**
  - roll out N routes for K steps by sampling π_A;
  - at each step, re-weight the top B candidate actions by one-step lookahead
    Q = r̂ + γV(s′) (an MPPI-style softmax with temperature η; Williams et al. 2017; TD-MPC2, Hansen
    et al. 2024, arXiv:2310.16828).
- **Route weights (corrected 2026-10-02, AS-251):** routes are sampled from π̃, so each draw
  weighs 1/N. Identical action sequences are merged with their counts summed, w_r = count_r/N: the
  unbiased Monte-Carlo estimate of the route mixture. ≤ N distinct routes come back (D-46). The
  route's own probability Π_k π̃(a_k | s_k) is reported beside it as a likelihood, not used as a
  weight; using it would count the policy twice.
- **Infiltration curve (monotone by construction):**
  - per route, S_n(k) = Π_{j≤k}(1 − h_{n,j});
  - P_inf(k) = Σ_n w_n (1 − S_n(k)).
- **Infiltration state (AS-18; D-03a held):** an internal entity reaches a post-initial-access
  tactic (Execution, Persistence, Privilege Escalation, Defense Evasion, Credential Access, Discovery
  from inside, Lateral Movement, Collection, C2, Exfiltration, Impact). The set is configurable.
- **Adversary reward (AS-17; D-11c held):** r^A_k = Δ(stage progress along the kill-chain order)
  + β·𝟙[infiltration at k] − κ·exposure_k, where exposure is the marginal-energy novelty of the step.
- **Coupling (AS-22; D-12 held):** STAGED. The heads read TAAFT through stop-gradient in the first
  part of stage 5, then joint (`policy_value.head_input`).

### 2.9 Advisor: `models/advisor` (advisory only, D-33)
- 4 decoder-style blocks.
  - Queries: candidate counter tokens.
  - Cross-attention: to Imagination tokens (TAAFT context and adversary slots) and to the Environment
    (gathered as for TAAFT).
- **Defender policy π_D** over 256 D3FEND action slots × target pointer × level {graph, sensor}.
- **Effect model:** a counter is an intervention on the imagined state. Graph level edits contact
  (isolate removes the target's edges; block removes a port class). Sensor level edits observation
  (adds an observable).
  - The Forecaster re-imagines under the intervention.
  - ΔP_inf = P_inf(K | counter) − P_inf(K | none).
- **Search:** beam search of width W over sequences of up to L_seq steps, each candidate evaluated
  with a fixed number of re-imagined routes.
- **Ranking:** CVaR_{0.2} of ΔP_inf over routes (Rockafellar & Uryasev 2000; AS-24, D-03c held),
  then cost, then feasibility. The expected and worst case are reported beside it.
- **Disruption cost (AS-23; D-03b held):** Σ criticality(kind of target) × disruption weight(action),
  with both tables in config. OT devices are the most critical (D-34).
- **Feasibility:** Φ_phys of the predicted effects ≤ τ, plus rule checks (no sensor action where no
  sensor exists).
- **Information value:** the expected entropy reduction of the stage posterior (for detect and model
  actions).

### 2.10 Verifier: `models/verifier` (weights change only on a HumanCommand, D-21)
- **Process-reward model:** 4 pre-norm blocks over [context ; route steps], giving a per-step
  plausibility. It is trained on step labels (Lightman et al. 2023, arXiv:2305.20050).
- **Trust value head:** P(forecast or advice is right | forecast summary, Monitor statistics).
- **Calibration policy head:** proposes log T (temperature) per output family from reliability
  statistics. Its training target is the maximum-likelihood temperature on resolved pairs.
  - The proposal is applied only through `apply_calibration(HumanCommand)`.
- **RLCD reward:** r = −(p − y)², excluding responded-to pairs (P-10, assumed in AS-25; Damani et al.
  2025, arXiv:2507.16806).
- **Drift in Monitor:**
  - CUSUM on forecast−outcome residuals;
  - the systematic gap |Σ(f − y)|/n over the last 12 resolutions;
  - Welford mean and variance of Environment and Imagination latents, with a Page–Hinkley alarm.
  - All of these are poisoning alerts for human review.
- **Conformal thresholds:** split-conformal alert thresholds at a target false-positive rate
  (Angelopoulos & Bates 2021, arXiv:2107.07511).
- **Initial supervised truth (AS-25):** for public datasets, the dataset authors' annotations count as
  human-supplied truth. They are people's labels (D-17: human feedback is outside the threat model).

### 2.11 Generator: `models/generator` (training only, D-40; families AS-27, D-14 held)
1. **Observability sliding** (deterministic, label-preserving by construction):
   - drop packet-level fields (status → NOT_OBSERVABLE);
   - flow-only export;
   - 1-in-n sampling with rescaled counts;
   - sensor hiding;
   - timing jitter within the physics bounds.
2. **Signature variation:** port remapping inside the service class, tool-fingerprint swaps, and
   rate scaling within the physics bounds.
3. **Masked-generative model:** a field-state transformer that fills masked fields (MaskGIT-style
   iterative unmasking, Chang et al. CVPR 2022, arXiv:2202.04200). Run left-to-right over a record
   sequence, it is the autoregressive family.
4. **Diffusion:** a TabDDPM-style denoiser for continuous fields, conditioned on the observed fields
   (Kotelnikov et al., ICML 2023, arXiv:2209.15421).
5. **Energy acceptance (JEM-style):** a variant is kept only if TAAFT's marginal energy E(∅, ·) is
   within the real-data range. Plus the physics gate Φ_phys ≤ τ (P-11, assumed in AS-28).
- **Rules:**
  - trained on the training split only (P-23);
  - zero-shot sets never see variants (D-23).

---

## 3. Training (D-22, D-23; stages 3–6)

**Stage 3: CVG-AE, Decoder, TSTCT.** Self-supervised, on real + generated data.

    L₃ = L_rec + λ_edge L_edge + β_dyn KL(sg q_{t+1} ‖ p_{t+1}) + β_rep KL(q_{t+1} ‖ sg p_{t+1})
         + β_0 KL(q_first ‖ 𝒩(0,I)×Unif) + λ_phys Φ_phys(decoded) + λ_gate ‖g‖₁

- KL with free bits of 1 nat; β_dyn = 0.5, β_rep = 0.1 (DreamerV3) (AS-05).
- Masking: 15 % of contributing cells (status → MASK), with reconstruction scored on all contributing
  cells and up-weighted on masked ones.
- R is sampled per batch: 1 + Poisson(3), clipped to [1, 8]. Backprop goes through the last 2
  thinking passes only (Geiping et al. 2025) (AS-07).

**Stage 4: TAAFT.** Self-supervised, with stage-3 modules frozen (checked by `pipeline/freezing.py`).
- Masked-entity belief: hide whole entities' recent states from the Environment view; ŷ_v must
  predict their latents (partial observability: what is not seen is not assumed absent).
- Future-latent prediction: ŷ's next-latent readout against the posterior at the next trigger.
- EBT training: the loss is taken on ŷ_S after an unrolled descent (create_graph). S is drawn from
  {2,…,8} and α is learned (Gladstone et al. 2025, arXiv:2507.02092).
- Contrastive marginal energy: real windows against time-shuffled or entity-swapped windows,
  L = softplus(E_pos − E_neg).

**Stage 5: full training.**
- Supervised labels: compromise per entity, stage per update and entity, and infiltration hazard as a
  discrete-time survival NLL with censoring.
- Forecaster: behaviour cloning on technique labels where known; latent consistency of route states
  against the TAAFT summaries at later triggers (stop-gradient targets); TD(λ) value; process reward.
- Advisor: model-based policy improvement on −ΔP_inf − κ·cost under re-imagination.
- Verifier: PRM on step labels (dataset annotations, AS-25); trust head on resolved pairs.
- Coupling: STAGED (AS-22).

**Stage 6: zero-shot validation with calibration.**
- Real data only; novel and known families are scored separately.
- Site adapters (AS-26; D-13 held):
  - LoRA rank 16 on the TSTCT and TAAFT attention projections, plus the Verifier temperature;
  - trained on 24 h of the site's unlabelled traffic (stage-3 and stage-4 objectives) plus 50
    analyst-confirmed alerts;
  - applied only on a HumanCommand.

---

## 4. L preset (1,134,268,667 parameters; counted exactly by `models/config` + meta-device build)

| Component | Shape | Parameters |
|---|---|---:|
| FieldEncoder | slots 128, d_field 128, hash rows 65,536, pool 4 heads → d_update 256 | 9,409,792 |
| CVG-AE | 6 planes × 4 layers × 256, 8 node kinds, 4 hyperedge kinds/plane, RWSE 16 steps; z = 128 + 16 × 32 | 117,852,056 |
| Decoder | field heads + 6 per-plane edge heads at 512 | 6,309,600 |
| TSTCT | 16 blocks × 1024, 16 heads (4 spatial / 8 temporal / 4 causal), SwiGLU 2816; keys per query 64 spatial + 512 temporal + 32 causal | 214,722,872 |
| Long-term memory | Titans-style MLP memory at 1024 (hidden 2048) | 7,340,032 |
| TAAFT | 34 decoder-style blocks × 1024 reading TSTCT's cache (block map 34 → 16), 16 heads, SwiGLU 2816; 16 adversary slots; d_y 256; 7 energy terms; cross keys 32 own + 64 neighbours + 32 long-term memory probes | 515,800,696 |
| Forecaster | 8 blocks × 1024; 700 techniques; 15 stages | 115,942,095 |
| Advisor | 4 decoder-style blocks × 1024; 256 D3FEND slots | 72,931,585 |
| Verifier | 4 blocks × 1024 PRM + heads | 73,959,939 |
| **Total (Generator excluded)** | | **1,134,268,667** |

The owner scaled TAAFT from 24 to 34 blocks (2026-10-02: "scale up taaft to 500m+"). The compute
profile of this model is in `docs/sizing.md`. The tests run the same code at small widths.

---

## 4b. From the owner's ai-mod-arch update (read 2026-10-02, 03:56 file)
1. **Lost-in-the-middle** (expert feedback, ai-mod-arch §11c; Liu et al., "Lost in the Middle", TACL
   2024, arXiv:2307.03172).
   - The log-Δt bias is learned per bucket, so it need not decay with distance the way plain RoPE's
     long-range decay does (Su et al. §3.4.3).
   - Environment buckets reserve slots for mid-term ages (§2.6).
   - Required test, the **mid-age recall probe**: a distinctive state planted at a middle age stays
     retrievable by the temporal heads, with weight within a factor of the recent case.
2. **Weeks of memory** (§4 of ai-mod-arch). The log-time buckets cover 1 ms … ≈ 4 weeks in about 31
   buckets, within 512 slots per entity. The Titans memory sits behind them.
3. **SHAP and LIME** (tech stack §6d; the problem statement names SHAP).
   - **Expected Gradients:** the SHAP GradientExplainer form, IG averaged over background baselines
     (Erion et al., Nature Machine Intelligence 2021, arXiv:1906.10670; Lundberg & Lee, NeurIPS 2017).
   - **LIME:** a weighted linear surrogate over field-group masks (Ribeiro et al., KDD 2016).
   - Both are written in-house over the field-state embeddings, so no new dependency.
4. **Malignity score** (§7f: "regressive scoring format (eg: benign 20%, malign 80%)").
   - A TAAFT readout m_v ∈ [0, 1] per entity and per adversary slot, trained in stage 4.
   - Soft target: the malicious share of the entity's updates in the trailing window.
5. **Out-of-order and topology variation** (§5n).
   - Stage-3 masking also permutes updates within their recorded `reorder_uncertainty_s`.
   - The Generator makes topology variants: hyperedge dropout and rewiring, physics-checked.
6. **Class imbalance** (§5e).
   - Class-balanced window sampling.
   - The Generator draws most of its variants from attack events (label-preserving).
7. **Splits** (§7d).
   - Full training: 60 % train (real + generated), 20 % test (real), 20 % validation and zero-shot
     (real + unseen datasets).
   - Pretraining: 70 % / 30 % (real).
   - Datasets: CIC-IDS2018, CTU-13, CIC-IoT-2023 (§7a).
8. **Forecast pathway statistics** (§4 of the first part). The route-weighted mean P_inf curve, the
   median with a 10–90 % band, and the mode route (highest weight).

---

## 5. Engineering contracts
- **PyTorch only** for the model (D-09 stack). No new dependencies without the owner's approval.
- Shared primitives live in `nagahana/nn/`: norms, SwiGLU, attention with typed masks / bias /
  null keys / KV return, rotary time, blocks, the two-stream loop, LoRA.
- **Tensor contracts:** `models/batch.py`. **Configs:** `models/config/`.
- Every module docstring has: purpose, owner sources, decisions (D-), assumptions (AS-), maths,
  invariants, extension points. Every non-trivial block has a comment.
- **Tests** (pytest + hypothesis), small widths:
  - no future leakage (R > 1 included);
  - dense ≡ cache path;
  - permutation equivariance;
  - monotone P_inf;
  - provenance never upgraded;
  - descent lowers E_total;
  - KL and ELBO match closed forms;
  - gradients reach every trainable parameter;
  - physics hard limits hold;
  - access matrix enforced;
  - HumanCommand gates.
- The L preset is built on the meta device and its parameter count is reported.
- ruff and mypy clean.
