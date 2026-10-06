# Assumptions of TAAFT (AS-200 … AS-224), 2026-10-02

Engineer C (TAAFT, the adversary-foundation core). Each entry is an engineering assumption of the L
build. It is **not** an owner decision: the held decision it stands for stays held in
`governance/decisions.py`. These IDs are not yet in the code registry (`governance/assumptions.py` is
outside this engineer's scope); `tools/gen_assumptions.py` turns each section below into a registry
entry. Code refers to them by ID in docstrings and comments.

Findings from code runs are cited as "code run: <test>, <number>". Citations are to sources the author
is sure of; anything else is marked "(citation to verify)".

---

### AS-200: The belief recursion is trigger-sequential self-attention over each token's own Imagination
- **Assumed:** each token's self-attention reads the current trigger's tokens and its own memory-stream
  K/V of the last M_im triggers in one softmax; half the heads rotate by trigger time; ŷ is not fed back.
  - Triggers of a window are processed in time order. At trigger m, a token's self-attention keys are
    the current trigger's tokens *and* its own memory-stream K/V from the last M_im =
    `imagination_triggers` triggers (m−1 … m−M_im), in one softmax (`blocks.attend_mixed`).
  - The past keys carry a learned log-Δτ bucket bias. The last `self_rotary_fraction` (½) of the heads
    rotate q and k by trigger time with the TSTCT rotary periods; the others are time-blind.
  - The refined hypothesis ŷ is **not** fed back into the memory stream.
- **Stands for:** The detail of build-spec §2.7 "belief recursion b_τ ← b_{τ−1} *is* this attention".
- **Why:**
  - Reading only one's *own* past keeps the per-trigger cost O(N·M_im) instead of O(N²·M_im) and is
    exactly "each token's own Imagination K/V" in the spec.
  - Not feeding ŷ back means Imagination depends neither on R nor on S: the cache is the same whatever
    the run-time budget (the two-stream argument of `nn/loop.py`, AS-06, extended to S). The belief is
    still carried forward through the temporal lens (AS-207).
  - Half the heads time-blind, half rotated: some heads should match "the same slot one trigger ago"
    regardless of gap, others should be phase-sensitive (periodic adversary activity).
- **Evidence:** Code run: `test_belief_recursion_reads_only_the_last_imagination_triggers`: with one
  block and no temporal lens, a change at trigger 1 alters triggers 1–3 and leaves triggers ≥ 4
  identical (M_im = 2). Code run: `test_attend_mixed_matches_gathered_attention`: equal to
  `attend_gathered` on concatenated key sets to 1e-5.

### AS-201: Which Environment positions an entity token reads, and how
- **Assumed:** an entity token reads its own last states and the latest states of hop-1-then-hop-2
  neighbours ranked by recency, with kind / log-Δt / plane biases, in a gathered or dense layout.
  - Own states: the chain latest → previous → … of the entity, up to `own_states` (AS-13: 32).
  - Neighbours: entities with a hop-1 or hop-2 contact as of τ, hop-1 first, then by the time of the
    neighbour's latest state (most recent first), up to `neighbour_states` (64); each contributes its
    latest position ≤ τ.
  - Bias per key: kind (own / hop-1 / hop-2) + log-Δt bucket of τ − t_j + per-plane bias for hop-1
    neighbours. Queries of TSTCT's temporal and causal heads (head order spatial, temporal, causal) are
    rotated by τ, so the score depends on τ − t_j only.
  - Two layouts with the same function: "gathered" (default; the inference layout from the Environment
    store) and "dense" (the same key sets as a mask over all P positions).
- **Stands for:** The neighbour definition and ranking left open in AS-13 / AS-37.
- **Why:** TSTCT's spatial heads use hop ≤ 2 as of t (build-spec §2.5); TAAFT uses the same reach
  so its view is not narrower than the Environment's. Recency ranking matches TSTCT's "most recent
  neighbours" cap. The dense layout costs B·H·N·P scores, the gathered layout B·H·N·(own + neighbours)·d
  of gathered memory; which is cheaper depends on P versus N·T_k, so both are kept.
- **Evidence:** Code run: `test_dense_and_gathered_cross_layouts_compute_the_same_function`: contexts
  equal to 1e-5. Code run: `test_cross_attention_block_map`: a TSTCT block outside the map does not
  change the output; the mapped block does.

### AS-202: Belief-and-trust energy is a robust (outlier-process) likelihood with a trust prior
- **Assumed:** E_bt = Σ_v −log(t_v e^{−ψ_v} + (1−t_v) e^{−β₀}) + KL(Bern(t_v) ‖ Bern(t₀(c_v)))
  + Σ_i ½ ‖(y_i − μ(c_i)) / σ(c_i)‖², with ψ a learned compatibility ≥ 0, t_v the trust *readout*, β₀ a
  learned outlier level, t₀ a context trust prior.
- **Stands for:** The form of "−log p(evidence | y) under trust t" (build-spec §2.7; lens design open).
- **Why:** With the trust inside the energy, a plain t·ψ term could be lowered by distrusting
  everything; the mixture form cannot (lowering t costs log-likelihood unless the evidence is worse than
  β₀, and the KL to t₀ costs too). The gradient on y is the responsibility-weighted evidence gradient:
  distrusted telemetry moves the belief less. This is the outlier-process view of robust estimation:
  Black & Rangarajan, "On the unification of line processes, outlier rejection, and robust statistics
  with applications in early vision", IJCV 19(1), 1996.
- **Evidence:** All parts are ≥ 0 (code run: `test_lens_bounds_and_physics_term`, min ≥ −1e-6).

### AS-203: Game energy is the adversary's soft best-response value; mechanism design inside it
- **Assumed:** E_game = −(1/G) Σ_g (1/β) log(e^{βu_∅} + Σ_v e^{β u_gv}), u_gv = κ tanh(⟨W_a y_g,
  W_e y_v⟩/√r) − cost(c_v). The defender-shaped cost and the participation constraint (abstain action
  u_∅) are the mechanism-design content.
- **Stands for:** D-24 (held: mechanism-design placement), D-11c (held: adversary objective) as far as
  TAAFT's own energy needs it; AS-14 already places mechanism design inside E_game.
- **Why:** The logit quantal response (McKelvey & Palfrey, "Quantal Response Equilibria for Normal
  Form Games", Games and Economic Behavior 10(1), 1995) gives a boundedly rational adversary whose value
  is a smooth, bounded function of the hypotheses. Individual rationality (an agent acts only if acting
  beats its outside option) is the standard participation constraint of mechanism design (e.g. Myerson,
  "Optimal Auction Design", Mathematics of Operations Research 6(1), 1981). Nothing here decides whether
  the defender's *design* belongs to the Advisor (D-24 stays held); the `mechanism-design` registry entry
  still requires D-24.
- **Evidence:** Bounded below by −(max(u_∅, κ) + log(V+1)/β)·s (code run:
  `test_lens_bounds_and_physics_term`). Masking after the β-scaling avoids a 0·∞ NaN gradient
  (found by `test_gradients_reach_every_parameter_with_unrolled_descent`, fixed).

### AS-204: Information energy prices departures from the null hypothesis by evidence informativeness
- **Assumed:** E_info = Σ_v ½ κ_v ‖W_i(y_v − y_∅)‖²/r with κ_v = softplus(MLP(c_v, ν_v)) and ν_v
  the noise features of AS-205 with validity flags.
- **Stands for:** The form of ψ_info (build-spec §2.7); D-26 (held: noise analysis) via AS-36.
- **Why:** ½κ‖Δ‖² is the code length of a departure under a Gaussian prior of precision κ (the MDL
  reading: Rissanen, "Modeling by shortest data description", Automatica 14(5), 1978). Uninformative
  evidence → high κ → strong claims are expensive. The term is ≥ 0 and reports κ as the lens's own SNR
  readout.
- **Evidence:** Code run: `test_lens_bounds_and_physics_term` (≥ 0).

### AS-205: Noise features are the event-time periodogram peak ratio, dominant period, aggregated-variance Hurst and event count
- **Assumed:** per entity as of τ: Schuster periodogram peak ratio and dominant period on a log grid of
  periods, aggregated-variance Hurst estimate, log event count, each with a validity flag.
  - Events of an entity = its TSTCT positions ≤ τ.
  - Schuster periodogram I(f) = |Σ_k e^{−2πi f t_k}|²/n on 64 log-spaced periods in [1 s, 3600 s];
    peak ratio = max/mean; dominant period = 1/argmax (grid ordered longest period first).
  - Hurst: aggregated variance of the gap series over block sizes 1 … 128 (8 scales), a scale counted
    with ≥ 4 complete blocks; least-squares slope β; H = 1 + β/2 clamped to [0, 1].
  - Validity: periodogram needs ≥ 8 events; Hurst needs ≥ 2 scales; invalid features are 0 with a
    False flag (D-41).
- **Stands for:** AS-36's "periodogram peak ratio of inter-arrival times and aggregated-variance Hurst
  estimate" (held D-26).
- **Why:** The event-time periodogram needs no bin width and is exact for irregular times (Schuster,
  "On the investigation of hidden periodicities…", Terrestrial Magnetism 3(1), 1898 (citation to
  verify: volume/pages)). The aggregated-variance estimator is the simplest of the classical estimators
  (Taqqu, Teverovsky & Willinger, "Estimators for long-range dependence: an empirical study", Fractals
  3(4), 1995). Benign aggregates are self-similar (Leland et al., IEEE/ACM ToN 1994); beacons are periodic
  under jitter (Hu et al., BAYWATCH, DSN 2016). Every statistic is a per-entity prefix sum, so "as of τ"
  is a single gather (no future leakage by construction).
- **Evidence:** Code runs: `test_hurst_is_one_half_for_iid_exponential_gaps` (|H − 0.5| < 0.06 for
  seeds 0–2; an exploratory run gave 0.498, 0.498, 0.482); `test_hurst_is_high_for_persistent_gaps`
  (H > 0.85; exploratory run 1.0); `test_periodogram_closed_form_and_planted_period` (I(1/T) = n to
  1e-6·n; dominant period = T); `test_features_are_as_of_and_segmented_per_entity`. Note found while
  building: integer event times alias onto the 1 s grid period (exploratory run, period reported 1 s
  for T = 10 s) — a real property of the grid, which the dominant-period readout must be read with.

### AS-206: Topology energy is a robust pairwise MRF on the contact graph as of τ
- **Assumed:** E_top = ½ Σ_{u≠v} w_uv log(1 + ‖A(y_u − y_v)‖²/r), w from hop-1 (with per-plane
  biases) and hop-2 contact as of τ; the null reading uses one learned weight w_∅/(n−1) on every active
  pair.
- **Stands for:** The weights "from plane bits / hop" and φ_top (build-spec §2.7).
- **Why:** The Lorentzian potential lets an edge "break" between a compromised and a benign host
  (Black & Rangarajan 1996). Plane-specific weights let identity/remote-admin/OT contacts carry more.
- **Evidence:** ≥ 0 (code run: `test_lens_bounds_and_physics_term`).

### AS-207: Temporal energy compares with the previous refined belief through a learned transition
- **Assumed:** E_time = Σ_i ½ Σ_d π_d(Δτ)(y_id − f_time(y_i^prev, Δτ)_d)², f_time = y^prev +
  MLP([y^prev; e(Δτ)]), precision from the Δτ bucket embedding; y^prev is the token's refined ŷ at its
  previous active trigger within M_im, detached.
- **Stands for:** f_time and Σ of build-spec §2.7.
- **Why:** A Gaussian transition without its normaliser; Δτ-dependent precision lets beliefs move
  more after long gaps. Detaching y^prev bounds the training graph (no gradient path across triggers
  through the hypotheses; the transformer path through Imagination keys remains).
- **Evidence:** ≥ 0 by construction.

### AS-208: Causal energy is directed and weighted by TSTCT's Granger gates aggregated as of τ
- **Assumed:** ḡ_{u→v}(τ) = Σ_{i: v, t_i ≤ τ} Σ_{j: u, t_j < t_i} g_ij / max(1, #candidates), gates
  averaged over the causal heads; E_cause = γ Σ ḡ_{u→v} log(1 + ‖A y_v − B y_u‖²/r).
- **Stands for:** "ḡ the TSTCT causal gates averaged over the window" (build-spec §2.7), read as of τ.
- **Why:** Granger-style predictive influence (Granger, Econometrica 1969; Tank et al., TPAMI 2021;
  AS-10) — reported as such, never as identified causal structure. The strict t_j < t_i mask is re-applied
  here so even a gate that broke TSTCT's contract cannot leak a later state.
- **Evidence:** Code run: `test_no_future_leakage_from_later_positions_contacts_or_triggers` perturbs gate
  rows and columns of later positions.

### AS-209: The physics term on beliefs uses the expected next latent and is normalised per entity
- **Assumed:** z_v = [μ_next ; softmax(ℓ_next)] (expected latent), decoded with the role and planes
  of the entity's latest update; E_phys = λ_phys log(1 + Φ_b/|R_b|) per trigger; λ_phys = 0.1 (AS-15),
  not learned.
- **Stands for:** "normalise residuals" in AS-15 for TAAFT's use.
- **Why:** Raw-unit residuals (bytes) can be huge; per-entity normalisation and log(1+·) keep the
  zero set and order while bounding the gradient. A learned λ could be trained to zero, removing the
  boundary (D-18).
- **Evidence:** Code run: `test_lens_bounds_and_physics_term`: 0 without decoder; > 0 with violations;
  physics-only descent lowers it.

### AS-210: Every lens has a learned scale and a reference level learned as an EMA over real windows
- **Assumed:** E_ℓ = exp(s_ℓ)·raw_ℓ (not physics). Reference_ℓ = bias-corrected EMA (momentum 0.99)
  of the mean per-trigger E_ℓ over training windows read with their real context; reported
  `energy_rel/ℓ` = E_ℓ − reference_ℓ.
- **Stands for:** ADR-0007's consequence "shares are read relative to each term's own reference level
  (e.g. its value on benign traffic)".
- **Why:** Labels never enter the forward pass, so "benign" cannot be selected inside TAAFT; the EMA
  over real (mostly benign) training traffic is the label-free stand-in. Stage 5 can restrict it to
  benign windows through the objectives.
- **Evidence:** Code run: `test_gradients_reach_every_parameter_with_unrolled_descent` (one update from the
  real pass, none from the null pass).

### AS-211: A lens's share is the projection of its step on the total displacement
- **Assumed:** share_ℓ = ⟨−α∇E_ℓ, d⟩/‖d‖², d = −αΣ∇E_ℓ, at the last step's input point; Σ = 1
  exactly; negative shares allowed; 0 when d = 0; noise excluded.
- **Stands for:** "each lens's share of the last step's displacement (−α∇E_lens / total)".
- **Evidence:** Code runs: `test_lens_shares_closed_form_and_sum_to_one`;
  `test_real_lens_sum_decreases_for_small_steps_and_shares_sum_to_one` (sum = 1 to 1e-4).

### AS-212: Descent noise is annealed linearly and off at inference unless asked for
- **Assumed:** σ_i = σ_0(1 − i/S), σ_0 = 0.01; applied in training, and at inference only when a
  generator is passed.
- **Stands for:** "annealed noise" (build-spec §2.7) without a schedule.
- **Why:** Forensic replay must be reproducible by default; exploratory noise is opt-in.

### AS-213: Readouts are linear functionals of ŷ; goal and type are slot mixtures
- **Assumed:** Every readout is a linear map of ŷ with a link (σ, softmax); goal/type = Σ_g ω_g
  softmax(W ŷ_g) with ω = softmax over active slots; the trust readout is the head E_bt uses; a second
  latent head gives the believed *current* latent (masked-entity objective).
- **Evidence:** Code run: `test_suspicion_floor_and_normalised_readouts` (floor holds at float32 for
  inputs up to 1e6; rows sum to 1).

### AS-214: The null reading removes context and evidence-derived structure
- **Assumed:** Dropped windows (all with `drop_context=True`; probability `context_dropout` per
  window in training) use a learned null context per token type; contacts, causal gates and noise
  features are removed; topology uses the uniform null weight.
- **Stands for:** P-09 / AS-16 ("E(∅, y)") details.
- **Evidence:** Code run: `test_null_context_reading_and_marginal_energy` (`marginal_energy` equals the
  forward's null-reading terms to 1e-4).

### AS-215: Stage-4 negatives are corruptions of TAAFT's view, plus an energy-magnitude regulariser
- **Assumed:** `entity_swap` (token v reads entity π(v)'s states) and `time_shuffle` (anchor =
  random earlier-or-equal state of the same entity); L = softplus(E_pos − E_neg) + λ_reg(E_pos² + E_neg²).
- **Stands for:** "time-shuffled or entity-swapped windows" (build-spec §3) at the level TAAFT can build
  without re-running the frozen perceptors.
- **Why:** A corruption that re-times updates needs the data pipeline and graph builder
  to rebuild positions and contacts (outside this scope); the TAAFT-level corruptions are always as-of
  safe. `time_shuffle` makes anchors staler, so the age embedding could become a shortcut; real-window
  re-timing through the pipeline is the recommended complement. The regulariser follows Du & Mordatch,
  NeurIPS 2019, arXiv:1903.08689.
- **Evidence:** Code run: `test_corruptions_never_read_the_future`.

### AS-216: Masked-entity targets are the frozen CVG-AE posterior at the hidden entity's latest state
- **Assumed:** Hidden entities: token from a learned vector, no own reads, not read as anyone's
  neighbour, noise features invalid, causal gates zeroed; contacts kept. Target: posterior (mean,
  categorical logits) at `entity_latest`.

### AS-217: M_im is duplicated in TAAFTConfig and must equal MemoryConfig.imagination_triggers
- **Assumed:** `TAAFTConfig.imagination_triggers` (L: 8) mirrors `MemoryConfig.imagination_triggers`;
  the integrator should assert equality (the tiny preset currently has memory 3, TAAFT 8: requested
  override in the report).

### AS-218: Entity tokens add kind, internal flag and age-bucket embeddings
- **Assumed:** x₀_v = W_in e_v + emb(kind) + emb(internal) + emb(bucket(τ − t_latest)).
- **Why:** Kinds and time buckets are allowed encodings (D-49: no index-based positions); staleness
  of the latest state matters to a belief.

### AS-219: Pairwise and game subspaces have rank d_hyp / 4
- **Assumed:** r = d_hyp // `lens_rank_divisor` (L: 64).
- **Why:** Pairwise terms cost O(V²·r) per descent step; r = 64 keeps them a small fraction of the
  transformer's cost at L while leaving most of ŷ free for unary terms and readouts.

### AS-220: TAAFT reads the long-term memory through learned probe queries as extra cross keys
- **Assumed:** TAAFT owns X = `memory_probes` (L: 32) probe vectors p_x of the memory's query width and
  two maps D → H·d_h; r_x = M(normalise(W_Q p_x)) by `LongTermMemory.read`, K = RMSNorm_h(W_K^mem r),
  V = W_V^mem r. One K/V set [B, H, X, d_h] is shared by every block; tokens read it in the same softmax
  as the Environment keys, scored with the un-rotated query (memory keys carry no time, D-49), plus a
  learned per-head bias by token type. Replaces the integration workaround AS-400 (recall added to the
  token initialisation).
- **Stands for:** build-spec §2.6 "It is read by TAAFT as extra memory keys" (detail); AS-11.
- **Why:** Titans' memory is read by a forward pass at queries (Behrouz et al. 2024, arXiv:2501.00663).
  A fixed set of learned probes turns the memory's content into a small key set whose size does not
  depend on traffic volume, so reading it costs O(N·X) per block whatever has been written (D-36 spirit).
  Probes do not depend on the current input, so the read cannot be steered by what an attacker sends
  *now*, only by what was written before. One shared K/V set (not one per block) costs 2·D·d parameters
  instead of 2·L·D·d (34× fewer at L); each block still reads it with its own query projection. Scoring
  memory keys with the rotated query would make the score depend on absolute trigger time (D-49).
- **Evidence:** code runs: `test_attend_with_memory_matches_gathered_attention_on_concatenated_keys`
  (equal to `attend_gathered` on the concatenated key sets to 1e-5);
  `test_memory_keys_with_dense_and_gathered_layouts_agree` (1e-5);
  `test_read_longterm_gives_gradient_to_probes_and_memory` (probes, maps, bias, W_Q, W_K and M₀ all get
  a non-zero gradient). L preset: 2,130,016 parameters (meta-device count).

### AS-221: Adversary slots read the long-term memory too
- **Assumed:** adversary slots, which read only the null key in the Environment cross-attention, do
  read the memory keys, with their own learned per-head bias (entity bias and slot bias are separate).
- **Stands for:** build-spec §2.7 "Adversary slots read a learned null key only" (detail of the new
  memory keys; the Environment rule is unchanged).
- **Why:** the memory summarises observations over weeks (AS-401 writes observed memory-stream states
  only, D-35), which is the horizon of the low-and-slow campaigns an adversary hypothesis is about; a
  slot otherwise sees the past only through the entities of the current trigger. The separate bias lets
  training switch the slots' reading off (a large negative bias) if it does not help, without a code
  change. Slots still never read individual Environment states.
- **Evidence:** code run: `test_memory_keys_per_trigger_are_as_of_and_read_by_entities_and_slots`
  (slot contexts change when memory keys are given).

### AS-222: The memory read at a trigger holds only writes of earlier triggers
- **Assumed:** `memory_kv` is either one state for the call, written by triggers strictly before the
  call's first trigger, or one state per trigger ([B, M, H, X, d_h]) where entry m holds the writes of
  triggers strictly before τ_m. A trigger never reads its own write.
- **Stands for:** the no-future-leak contract of build-spec §2.2 applied to the long-term memory.
- **Why:** the write at τ_m uses observations ≤ τ_m, so reading it at τ_m would not leak the future,
  but inference writes after analysing (engine order: analyse, then write), and training must compute
  the same function; "strictly before" is a rule both can follow. The per-trigger form gives training
  exactly the inference function inside a multi-trigger window; the single-state form (state at the
  window start) is cheaper and lags by at most the window.
- **Evidence:** code runs: `test_memory_keys_per_trigger_are_as_of_and_read_by_entities_and_slots`
  (perturbing entries m > m* leaves triggers ≤ m* unchanged; a shared state equals the same state
  repeated per trigger); the split-call tests use per-trigger states built this way.

### AS-223: Carried Imagination uses one slot per trigger, right-aligned, with an as-of guard
- **Assumed:** carried triggers enter `forward` as slots (oldest first, padding at the front with mask
  False) on the receiving call's token axis, with trigger times relative to the receiving window's
  origin; the call's history is the carried slots then its own triggers, and "the last M_im triggers"
  is counted over that history. A carried entry is read at trigger τ only if its time is < τ, whatever
  the caller passed. Token alignment is the caller's (store read with the receiving token ids, or a
  `token_index` for a previous `AnalysisOut`), so `forward` takes no token-id argument.
- **Stands for:** detail of AS-13 / AS-157 (Imagination of the last M_im triggers) across calls.
- **Why:** the store keeps the last M_im *triggers* (AS-157) and the in-call recursion reads the last
  M_im triggers; a per-trigger layout makes both count the same thing, which is what makes split calls
  equal one call. The time guard makes a caller error (an entry stamped at or after τ) unable to leak.
- **Evidence:** code runs: `test_split_calls_equal_one_call_through_the_analysis_carry` (two calls, and
  a chained three-call carry, equal one call to 1e-5, Imagination included);
  `test_split_calls_equal_one_call_through_the_imagination_store` (one trigger per call through a real
  `ImaginationStore`, equal to one call to 1e-5); `test_past_imagination_is_read_only_before_each_trigger`.

### AS-224: The temporal lens reads the latest earlier belief among the last M_im triggers
- **Assumed:** E_time at trigger τ compares ŷ with the token's refined ŷ at its latest trigger among
  the last M_im with time < τ (carried `past_y` included); without `past_y`, carried triggers give
  E_time nothing.
- **Stands for:** detail of AS-207 across calls.
- **Why:** continuity of the belief across calls needs the previous ŷ, not only the K/V; the store does
  not hold ŷ, so the engine keeps the ŷ of the kept triggers (`past_from_store(y=…)`). Choosing "latest
  with time < τ" rather than "latest written" keeps the guard of AS-223 for the hypothesis too.
- **Evidence:** code run: `test_past_imagination_is_read_only_before_each_trigger` (a misplaced carried
  slot does not change trigger 0 and does change trigger 1).
