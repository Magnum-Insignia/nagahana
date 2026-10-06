# Assumptions of the integration (AS-400 … AS-449), 2026-10-02

Engineer G (Integration: `models/nagahana.py`, `training/`, `inference/`). Each entry is an
engineering assumption of the L build. It is **not** an owner decision: the held decision it stands for
stays held in `governance/decisions.py`. Code calls `assume("AS-4NN")` where it relies on an entry.

Findings from code runs are cited as "code run: <test or script>, <number>". Citations are to sources
the author is sure of; anything else is marked "(citation to verify)".

---

### AS-400: TAAFT reads the long-term memory through its token initialisation (superseded by AS-220)
- **Assumed:** superseded on 2026-10-02 by AS-220 (TAAFT reads the long-term memory through learned
  probe queries as extra cross-attention keys, `models/taaft/model.read_longterm`). No code relies on
  AS-400 any more: the read map `longterm_read` and the recall added to the refined states were removed
  from `models/nagahana.py`. Previous content, kept for the record: ẽ_p = e_p + W_lt·M(normalise(W_Q e_p)),
  read as of the window start in training.
- **Stands for:** build-spec §2.6 "It is read by TAAFT as extra memory keys", now implemented by AS-220.
- **Why:** the TAAFT engineer implemented the requested extra-key read; the workaround lagged by up to
  one window in training and touched TAAFT's token initialisation instead of its memory reads. The
  training read is now exact per trigger (AS-222: the state read at τ_m holds only writes of triggers
  before τ_m, `NagaHana.longterm_per_trigger`).
- **Evidence:** code run: `tests/test_integration_model.py::test_longterm_memory_receives_gradient_through_the_read`
  (every long-term parameter and TAAFT's memory maps get a non-zero gradient through `memory_kv`);
  `tests/test_integration_e2e.py::test_engine_reads_memory_before_write_and_earlier_imagination`.

### AS-401: the long-term memory stores observations, written once per trigger (read through AS-220)
- **Assumed:** at trigger τ_k the memory is written with the TSTCT memory-stream states of the
  entities active at τ_k (their latest position ≤ τ_k), one pair per entity, α_k = `retention_alpha`.
- **Stands for:** build-spec §2.6 (what is written is not specified; D-36 fixes when).
- **Why:** the memory sits behind the Environment buckets (build-spec §4b.2) and belongs to the
  Environment's working memory (`memory/longterm.py`). Writing TAAFT beliefs into it would let a belief
  return as an observation, which D-35 forbids. The memory-stream state is the cache-consistent,
  R-independent summary of an observed state (AS-06).
- **Evidence:** D-35 access matrix (`memory/access.py`): the Environment is written only by the
  Simulator role.

### AS-402: the physics term of training
- **Assumed:** Φ_phys over FlagCountBound (6 flags), MTUBound (both directions, site MTU from
  `GeneratorConfig.mtu`, required), IATMaxBound, IATMeanBound, IATVarianceBound and CountWithinPackets
  (DF, MF, retransmissions), each as `RelativeResidual` with weight w_c = 1; λ_phys from
  `TrainingConfig.lambda_phys`; in stage 3 the term is divided by the number of decoded rows.
  `MinHeaderBound` is left out (its site value has no config field).
- **Stands for:** AS-15 (λ_phys on normalised residuals) and AS-114 (residual catalogue); weights w_c
  are required by `PhysicsTerm` and are not otherwise set.
- **Why:** relative residuals are dimensionless, so equal weights compare violations on one scale; the
  per-row mean keeps λ_phys independent of the batch size, which the term's own docstring leaves to the
  component's loss.
- **Evidence:** `physics/normalise.py` (scale invariance tested in `tests/test_perception_decoder.py`).

### AS-403: TAAFT aggregates only the current window's causal gates
- **Assumed:** with a carry, TSTCT's causal gates are [B, H_c, P, C + P]; TAAFT receives the current
  part [..., −P:]. Causes that lie only in earlier windows are not aggregated by TAAFT's cause lens.
- **Stands for:** detail of D-51 and AS-208 (TAAFT's cause lens over window positions).
- **Why:** TAAFT's cause lens aggregates gates per (cause entity, effect entity) over window
  positions; carried slots are not positions. TSTCT still uses the carried causes in its own attention.
  Extending the cause lens to carried keys is a TAAFT change (requested in the build report).
- **Evidence:** `models/taaft/structure.causal_prefix` requires a [B, H_c, P, P] gate.

### AS-404: carried entities enter the window's entity table when they are within two hops
- **Assumed:** before a window with carry, the bridge appends to its entity table every carried entity
  (seen earlier in the segment) that is a ledger neighbour (hop 1) or a neighbour of a neighbour
  (hop 2) of a window entity, hop-1 first, then by recency of the entity's latest state, at most
  `TrainingConfig.max_entities` per window. Appended entities have no positions (inactive at every
  trigger), kind and internal flag from earlier windows of the lane, infiltration time +∞ and malicious
  share NaN (unknown). Carried slots of other entities are dropped before alignment.
- **Stands for:** item 2 of the carry contract (AS-161), which `data/stream.py` does not provide.
- **Why:** TSTCT's spatial heads read neighbours within two hops and its causal heads read hop-1
  sources; a carried entity farther away is masked anyway, so listing it would only cost memory. The
  cap keeps the key axis bounded by the same number as a window's own entity table (a scan touches
  thousands of hosts). Dropping the other carried slots before alignment bounds the carried key axis by
  |table| × slots_per_entity instead of the whole store.
- **Evidence:** code run: `tests/test_integration_carry.py::test_extension_keeps_window_indices_and_reads_carried_neighbours`.

### AS-405: cumulative contacts come from a per-lane ledger of first contacts by stable key
- **Assumed:** a per-lane ledger keeps, for every pair of stable entity keys that shared a hyperedge in
  an earlier window of the segment, the first contact time (any plane and per plane, epoch seconds).
  The window's C¹ and C¹_p become min(window, ledger − origin); C² is recomputed as the min–max product
  over the table and the ledger neighbours of the window entities. Times written to the Environment
  store are clamped to the store clock when the out-of-order augmentation moved a window's first update
  before the previous window's last one (a shift no larger than the recorded reorder uncertainty).
- **Stands for:** item 3 of the carry contract (AS-161), which `data/stream.py` does not provide.
- **Why:** first contact is monotone (D-52, `graph/window.py`), so the minimum over windows is the
  stream's first contact; C² rows of window entities are exact because every intermediate of a 2-hop
  path from a window entity is its neighbour. The store requires time order ([Q-20]); clamping changes a
  slot time by at most the sensor's own ordering uncertainty.
- **Evidence:** code run: `tests/test_integration_carry.py::test_cumulative_contacts_match_brute_force`.

### AS-406: optimiser details
- **Assumed:** AdamW with weight decay on matrices only (dim ≥ 2), linear warm-up over
  `warmup_steps` then a constant learning rate, global-norm clipping at `grad_clip`; a non-finite loss
  refuses the update.
- **Stands for:** detail of build-spec §3 (`TrainingConfig` gives lr, weight decay, warm-up, clip but no
  schedule after warm-up).
- **Why:** decaying gains and biases pulls norms towards zero without regularising capacity (Loshchilov
  & Hutter, "Decoupled Weight Decay Regularization", ICLR 2019, arXiv:1711.05101, decouple decay; the
  matrices-only convention is common practice, citation to verify). A decay schedule after warm-up
  needs the total step budget of a real run, which is not set; constant is the neutral choice.
- **Evidence:** code run: `tests/test_integration_model.py::test_optimiser_groups_and_warmup`.

### AS-407: masked cells weigh twice in the stage-3 reconstruction
- **Assumed:** `mask_weight` = 2.0: L_rec scores every contributing cell, masked cells with weight 2.
- **Stands for:** build-spec §3 "reconstruction scored on all contributing cells and up-weighted on
  masked ones" (the weight is not given).
- **Why:** masked cells carry the self-supervised signal (they cannot be copied through the encoder);
  doubling them keeps them a visible share of the loss at a 15 % mask rate (15 % × 2 = 30 % of the
  weight) without discarding the reconstruction of the others.
- **Evidence:** none beyond the arithmetic; the masked part is logged separately (`stage3/rec_masked`).

### AS-408: the first-state KL
- **Assumed:** positions that are an entity's first state in the window, for entities with no carried
  state, add β₀·max(fb, KL(q ‖ 𝒩(0, I) × Unif)) with β₀ = 0.1 (= β_rep) and the free bits of AS-05.
- **Stands for:** build-spec §3 β₀ (value not given).
- **Why:** the fixed prior plays the role of the learned prior for states without a predecessor, so it
  gets the representation weight of the balanced KL (DreamerV3's β_rep = 0.1, arXiv:2301.04104) and the
  same free bits; states with a carried predecessor have a learned prior in the store and are not
  pulled to the standard prior.
- **Evidence:** none beyond the reasoning.

### AS-409: the two transition-prior KLs weigh one half each
- **Assumed:** w_mem = w_ref = 0.5 for the balanced KL against the memory-stream prior and against the
  refined (thinking-stream) prior.
- **Stands for:** AS-159 ("the weighting between the two KL terms is the trainer's choice").
- **Why:** the posterior receives the representation pressure of both terms; halves keep its total at
  β_rep as AS-05 specifies, while each stream's prior still gets its full gradient direction.
- **Evidence:** none beyond the reasoning.

### AS-410: candidate hyperedges for the edge loss come from the local subgraphs
- **Assumed:** for each plane the observed hyperedges are those of the batch's as-of local subgraphs;
  a member is represented by the latent of its latest state as of the subgraph's centre (via
  `node_update`, `node_role`), service nodes and nodes without an update are left out, hyperedges with
  fewer than 2 represented members are dropped, at most 512 per plane per step are kept (uniformly).
- **Stands for:** detail of AS-04 / AS-111 (which latents represent members is not specified).
- **Why:** the subgraph holds exactly the hyperedges that existed as of the centre time, so no member
  latent comes from the future; the cap bounds the cost of the duplicated copies of one hyperedge
  across subgraphs.
- **Evidence:** code run: `tests/test_integration_model.py::test_hyperedge_members_are_as_of_and_real`.

### AS-411: stage-4 weights and the hidden-entity ratio
- **Assumed:** `Stage4Weights(masked_entity=1, future_latent=1, contrastive=1, malignity=1,
  energy_reg=0.01)`; each seen entity is hidden with probability 0.15 per window; the negative view
  alternates between `entity_swap` and `time_shuffle` from one step to the next.
- **Stands for:** build-spec §3 stage 4 (weights and ratio are not given; `Stage4Weights` has no defaults).
- **Why:** equal weights are the neutral start, every term is logged separately so a reweighting is a
  visible choice later; 0.15 mirrors the stage-3 field mask rate at entity level; a small energy-magnitude
  regulariser only keeps scales bounded (Du & Mordatch, NeurIPS 2019, arXiv:1903.08689 use such a term;
  the value 0.01 is this build's choice). Alternating negatives gives both corruptions equal exposure.
- **Evidence:** none beyond the reasoning; code run of the smoke test logs each term.

### AS-412: loop budgets during stages 4 and 5
- **Assumed:** TAAFT's R is drawn like TSTCT's (1 + Poisson(3), clipped to [1, 8], gradient through
  the last `TAAFTConfig.grad_passes` passes, which TAAFT applies itself); S ~ U{2, …, 8} (AS-16). The
  frozen perception runs TSTCT at its run-time default R (`TSTCTConfig.default_passes`).
- **Stands for:** AS-07 (stated for TSTCT) applied to TAAFT; the R of a frozen perceptor in stage 4/5.
- **Why:** TAAFT is the same weight-tied two-stream loop as TSTCT (AS-06), so the argument for random R
  carries over (Geiping et al. 2025, arXiv:2502.05171). Frozen perception should read the Environment
  the way inference will, i.e. at the run-time R.
- **Evidence:** none beyond the reasoning.

### AS-413: perceptors stay frozen in stage 5; STAGED switch and advisor weight
- **Assumed:** stage 5 trains TAAFT, the long-term memory, the Forecaster, the Advisor and (under a
  HumanCommand) the Verifier; the input layer, CVG-AE, Decoder and TSTCT stay frozen. The Forecaster
  and Advisor read TAAFT through a stop-gradient for the first `joint_after` optimiser steps (a run
  parameter, no default), then jointly. L_A has weight 1.
- **Stands for:** `pipeline/stages.py` stage 5 (trains: taaft, forecaster_heads, advisor_heads,
  verifier; the perceptors are listed neither as trained nor as frozen) and AS-22 (the length of the
  stop-gradient phase is not given).
- **Why:** stage-5 losses are label-driven; letting them reshape the perceptors would change the
  Environment that the stores of a deployed site hold and would let labels leak into the
  self-supervised world model. The stage table's `trains` list is followed literally.
- **Evidence:** `pipeline/stages.py` STAGES[4].trains.

### AS-414: supervision of TAAFT's readouts in stage 5
- **Assumed:** at trigger τ, the compromise readout of an active *internal* entity is trained by BCE
  towards 𝟙[t_infil ≤ τ] (AS-18 infiltration state reached; external entities are not supervised), and
  the stage readout by cross-entropy towards the stage label of the update behind the entity's latest
  position ≤ τ (unknown labels masked). Weight 1.
- **Stands for:** build-spec §3 stage 5 "compromise per entity, stage per update and entity"
  (the exact targets are not given).
- **Why:** the infiltration time is the label pipeline's per-entity truth (`data/labels.py`, AS-18);
  public datasets give no compromise truth for external hosts. The latest update ≤ τ is what the entity
  was doing as of τ, so the stage target never looks ahead. Per-update stages are supervised through
  the Forecaster's per-step stage head (`forecaster/losses.py`).
- **Evidence:** none beyond the reasoning.

### AS-415: the Verifier's step labels and trust labels during stage 5
- **Assumed:** for the PRM, 4 routes are imagined per trigger; imagined step k is plausible when its
  target is the realised target of step k (responder of the first malicious update in the step, or "no
  target" when the step has labelled updates but none malicious), implausible when the realised target
  is known and differs, unknown otherwise. The trust head's label is whether the decision
  P_inf(K) ≥ ½ matched the outcome (an internal entity first infiltrated within K steps, censoring
  respected). The calibration head trains only when ≥ `min_pairs` resolved pairs with both outcomes
  exist. Every Verifier update goes through `gate.train_on_feedback` with a HumanCommand.
- **Stands for:** AS-25 (dataset annotations as initial human truth) details; D-21 (decided: changes only
  on a human command).
- **Why:** the target is the most specific annotated fact of a step; techniques are mostly unlabelled in
  the public datasets. ½ is the decision threshold of a calibrated probability; conformal thresholds
  replace it in deployment (`verifier/calibration.py`).
- **Evidence:** none beyond the reasoning.

### AS-416: the priority trigger
- **Assumed:** every `probe_every` processed updates (half a training window) the engine evaluates the
  mean marginal energy E(∅, ŷ) of the active entity tokens at the last processed update, with the
  run-time budgets. A priority trigger fires there when it exceeds the mean of the cadence triggers'
  values by more than 3 standard deviations, after at least 5 cadence triggers, at most once per
  cadence interval ⌊t / c⌋.
- **Stands for:** AS-12 ("at most one priority trigger per cadence interval when marginal energy jumps";
  the jump test is not specified), held D-02.
- **Why:** the marginal energy is the model's own novelty reading (P-09, AS-16); a z-score against the
  cadence baseline is scale-free; the cap per interval bounds what an attacker can force, so retention
  per trigger (D-36) stays bounded in time. The probe interval is counted in updates processed, which
  only sets how often the test runs, never how much is forgotten.
- **Evidence:** code run: `tests/test_integration_e2e.py::test_priority_trigger_at_most_once_per_interval`.

### AS-417: what the driving features explain
- **Assumed:** the driving features of a forecast are Expected Gradients over the field states of the
  trigger's window, of F = 1 − Π_k(1 − h_k) along the forecast's mode route (actions fixed, teacher
  forcing), computed by the full differentiable chain with the TSTCT carry as of the window start; the
  background is the window with every cell NOT_SUPPLIED; 8 samples; per column the sum over the window's
  updates; categorical and bitmask columns are labelled with the value of their most contributing
  update. The energy-lens shares of the last descent step are reported beside them. F(f) and the
  completeness gap are returned with each explanation.
- **Stands for:** build-spec §4b.3 and the problem statement's "driving features" (the explained
  function is not specified; `imagine` is a sampled mixture and has no gradient).
- **Why:** the mode route is the forecast's single most likely future; holding its actions fixed makes
  the reading differentiable without changing what the model computes. "Nothing supplied" is the
  honest absence baseline (D-41), unlike zero. Reporting F and the gap lets a reader check how close
  the explained reading is to the reported P_inf.
- **Evidence:** Erion et al., Nature Machine Intelligence 2021 (arXiv:1906.10670). Code run:
  `scripts/smoke_e2e.py` (tiny preset, untrained beyond a few steps, 2 samples, after the AS-220/AS-223
  wiring): at 14:45:00 UTC P_inf(4) = 0.935, F(mode route) = 0.948, gap 1.30·10⁻⁴; at 14:46:00 UTC
  P_inf(4) = 0.915, F = 0.958, gap −1.63·10⁻⁴.

### AS-418: the engine's incremental structure and step context
- **Assumed:** the open window (the training plan's rule, AS-317) is rebuilt with
  `data.windows.build_window` each time a chunk of updates is processed (cost O(window) per chunk; old
  positions' graphs are unchanged by the as-of construction); new positions are stepped into the store
  with a context listing, per state, the neighbours within two hops as of its time that have a state
  (the `spatial_keys` most recent, plus every hop-1 neighbour with a state within Λ_lag). At a trigger,
  TAAFT reads the window holding the trigger with its Environment assembled from the stored step outputs
  (keys re-based from the store origin to the window origin), extended like training (AS-404, AS-405).
- **Stands for:** build-spec §1 / §2.2 streaming graph builder (not implemented) and the inference side of
  D-51.
- **Why:** this reuses the training computation exactly (same windows, same as-of graphs, same table
  extension), so a model is evaluated live on the function it was trained on. Neighbours without a state
  have no keys in the store; pre-selecting the most recent ones is what TSTCT's spatial heads keep anyway,
  and causal candidates exist only for neighbours active within Λ_lag.
- **Evidence:** TSTCT's dense ≡ cached path test (`tests/test_tstct_equivalence.py`) and carry test
  (`tests/test_tstct_carry.py`); code run: `tests/test_integration_e2e.py` replay.

### AS-419: alert level of reports and the spread of a compromise belief
- **Assumed:** forensic reports count P_inf(K) ≥ 0.5 and compromise ≥ 0.5 as alerts (patient-zero
  ranking, counterfactual trigger); `BeliefReadout.entity_compromise` carries (p, √(p(1−p))).
- **Stands for:** detail of the forensic outputs (architecture §6) and of `BeliefReadout` (std not defined).
- **Why:** ½ is the decision threshold of a calibrated probability; site deployments replace it with the
  Verifier's conformal threshold. Without an ensemble the only spread the model states for a binary
  belief is its Bernoulli standard deviation; it is labelled as such and never as an epistemic interval.
- **Evidence:** none beyond the reasoning.

### AS-420: site calibration objective
- **Assumed:** a site-calibration step adds the stage-3 loss, the stage-4 loss without the malignity term
  (site traffic is unlabelled) and, at trigger-bearing windows, the BCE of the compromise readout of each
  alerted entity at the first trigger at or after the alert, weight 1. Adapters: LoRA on q, k, v, o of
  TSTCT's and TAAFT's attention (rank, α from `TrainingConfig`). The Verifier temperature of the
  "compromise" family is the ML temperature on the alert pairs. Coverage below 24 h or 50 alerts raises
  unless the run states `allow_short=True`, and the report records the shortfall.
- **Stands for:** AS-26 (adapters and data) details, D-13 held.
- **Why:** the self-supervised terms fit the adapters to the site's normality; the alerts anchor the
  belief scale on confirmed outcomes; the temperature is the one-parameter correction the Verifier can
  propose (D-45). Applying either still needs a human command (D-21).
- **Evidence:** code run: `tests/test_integration_stage6.py`.

### AS-421: the marginal-energy callable of the Generator's acceptance
- **Assumed:** E(∅, ·) of a candidate window (`ColumnarUpdates`) is computed by windowing it with the
  training rule, putting one trigger at the last update of each of the first `max_windows` windows,
  reading the posterior mean without carry, running TAAFT at the given budgets and averaging
  `TAAFT.marginal_energy(ŷ, token_mask)` over the active tokens and windows. Real training segments
  calibrate the accepted range with the same callable (`EnergyAcceptance.calibrate_on`).
- **Stands for:** AS-28 / AS-361 (the energy check needs a window → E(∅, ·) callable; the Generator
  leaves it to TAAFT's owner and the integrator).
- **Why:** variants are short candidate windows that may contain no cadence point; a trigger at the end
  of each window reads the whole window as of its last state, the same as-of discipline as training. No
  carry is used because a candidate is judged on its own content; real and generated windows are read
  the same way, so the range comparison is like for like.
- **Evidence:** code run: `tests/test_integration_variants.py::test_energy_callable_is_finite_on_real_and_variant_windows`.

### AS-422: transition-prior targets stay inside a window
- **Assumed:** the stage-3 balanced KL pairs a position with the same entity's next position *in the same
  window* (`PositionBatch.next_index`). An entity's last state of a window gets no transition target, and
  its first state of the next window gets no first-state KL when the entity has a carried state.
- **Stands for:** detail of D-51 and AS-05 (the next state across a window boundary lies in another batch).
- **Why:** with Transformer-XL recurrence the carried memory is stop-gradient data; the prior of a
  position in window k could only be trained against window k + 1 by keeping window k's graph alive
  (backpropagation through time across windows), which the carry design deliberately avoids. Windows of
  the L preset hold 1,024 updates, so most transitions fall inside a window; at the tiny preset (64
  updates) a larger share is lost, which is a property of the test fixture.
- **Evidence:** code run: `scripts/smoke_e2e.py` logs `kl_memory_raw`, `kl_refined_raw` per step.
