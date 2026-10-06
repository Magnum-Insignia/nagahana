# Assumptions of TSTCT and the memories (AS-150 … AS-161), 2026-10-02

Engineer B (TSTCT + memory). Each entry is an engineering assumption of the L build. It is **not** an
owner decision; the held decision it stands for stays held in `governance/decisions.py`. They are not
yet in the code registry (`governance/assumptions.py` is outside this engineer's scope): the requested
registry entries are in the build report. Code refers to them by ID in docstrings and comments.

Findings from code runs are cited as "code run: <test>, <number>".

---

## AS-150: Environment buckets are dyadic cells of absolute time, with a per-cell quota
- **What is assumed.**
  - Time is cut into dyadic cells of absolute time (relative to a fixed store origin):
    cell (ℓ, n) = [n·2^ℓ, (n+1)·2^ℓ)·δ₀, levels ℓ = 0 … n_buckets − 1.
  - At store clock T a cell is "aged" when ℓ = 0 or it ended at least one own width ago. Each state
    lives in its maximal aged cell.
  - Level-(n_buckets − 1) cells older than 2^{n_buckets}·δ₀ fold into one archive cell.
  - Every cell keeps at most m = `bucket_slots` slots (L: 7). An overflow merges the two most similar
    slots of that cell.
  - The store refuses configurations with m·(2·n_buckets + 2) + 1 > `slots_per_entity`. At L this is
    7·66 + 1 = 463 ≤ 512.
- **Stands for.** The bucket definition in AS-11 (detail of held D-15). It also departs from the brief's
  rule "when an entity exceeds capacity, merge in the most-populated recent bucket". That rule is never
  needed here, because capacity is a theorem (P4 below).
- **Reasoning.**
  - Buckets defined by *age* move as time passes. A flood written today would later share a coarse
    age bucket with older history and use up its quota, so volume would force merges of older states:
    a delayed breach of D-36.
  - Absolute cells with a canonical, bottom-up compaction make a cell's content depend only on the
    states inside it (P3). A flood in [s, T] can then change only cells that meet [s, T]. Cells that
    ended by s are bit-for-bit unchanged, and nothing older than s − (T − s) is touched.
  - The number of cells is bounded by time alone: at most 4 at level 0, 2 per middle level, 1 at the
    top level, plus the archive, so 2·n_buckets + 2 in total. Capacity therefore cannot be exceeded by
    any volume.
  - Resolution scales with age: a slot of age a sits in a cell of width w with a/4 < w ≤ a. That is
    about 2 cells, and so about 2m slots, per octave of age. Mid ages keep reserved slots (§4b.1).
- **Evidence.**
  - Code run: `test_flood_never_touches_history_older_than_its_own_cells`. A 3-day history, then a
    flood at 10× the rate in the last minute: every slot in cells that ended before the flood is
    identical, and the flood store performed more merges.
  - Code run: `test_compaction_is_path_independent`. Contents are identical whether compaction was
    evaluated at every step or only at the end.
  - Code run: `test_max_cells_bound_holds_and_is_tight`. With n_buckets = 6, a dense history kept at
    most 2·6 + 2 = 14 cells.
  - Code run: `test_capacity_holds_under_a_flood_and_bad_configs_are_refused`. 3000 irregular updates
    stayed within capacity.
  - The rationale for keeping mid ages ("lost in the middle"): Liu et al., TACL 2024, arXiv:2307.03172.

## AS-151: causal candidates are pre-capped to the 256 most recent
- **What is assumed.** Before the gate scores them, a query's causal candidates are capped to the
  `causal_candidate_cap` = 256 most recent, ordered by (time, write order). The gate then keeps the
  top 32.
- **Stands for.** A detail of AS-10. The build-spec has no bound on the candidate set.
- **Reasoning.**
  - The gate MLP runs per (query, candidate, head). An unbounded candidate set within Λ_lag = 300 s
    could be quadratic in P under a flood.
  - The cap bounds cost and memory in both paths, in the same way.
  - Recency is volume-sensitive, but only *inside* the 300 s lag window.
- **Evidence.** Code run: `test_masks_match_reference` (with cap 5): the cap matches the brute-force
  definition.

## AS-152: one causal gate map per head, from the block-0 values V⁰ (revised 2026-10-02, D-51)
- **What is assumed.**
  - g_ij^h = σ(w_hᵀ SiLU(A_h u_i + B_h u_j + C_h φ(Δt)) + c_h), with u = V⁰ = W_V⁰ RMSNorm⁰(e⁰), the
    block-0 memory-stream values. This is an MLP on [q^g_i ; k^g_j ; φ(Δt)] whose first layer is split
    by input.
  - One map per causal head, shared by every block and every pass.
  - Hidden width 64. The initial logit is 2.0, so the gates start almost open (σ ≈ 0.88).
  - Hard top-32 selection; the kept gates enter as log g.
  - The L1 term is the mean of g over the candidates.
- **Revision.**
  - The first version projected RMSNorm(e⁰) and stored the gate keys as a per-slot aux vector.
  - `carry_out` (D-51) has only `EnvironmentOut`, which holds no e⁰, and `batch.py` is outside this
    engineer's scope.
  - V⁰ is already cached for every state, and it is a full-rank learned linear map of the same
    normalised input. Using it therefore removes the extra storage, and it works unchanged for stored,
    carried and new states.
  - Because the gate projections are linear, the gate key of a merged slot (count-weighted mean V⁰) is
    exactly the mean of its members' gate keys.
  - Cost: the gate's gradient also reaches W_V⁰. This interaction has not been measured.
- **Stands for.** A detail of AS-10. The spec says "q_i ; k_j" without saying which projections.
- **Reasoning.**
  - `EnvironmentOut.causal_gate` is one map, and TAAFT's E_cause reads one ḡ.
  - Biases must be identical across passes (D-43).
  - Rotated block keys would make an MLP of them depend on absolute time phase.
  - Hard top-k with differentiable kept values is the sparse-gating routing of Shazeer et al.,
    ICLR 2017, arXiv:1701.06538.
- **Known risk.** Unselected candidates get no attention gradient, only the L1 push down, so gate
  collapse is possible. Mitigations to try if seen: noise on the gate logits in training, or a soft
  top-k.
- **Evidence.**
  - Code run: `test_causal_gates_are_sparse_and_on_candidates_only`.
  - Code run: `test_gradients_reach_every_parameter`.
  - Code run: `test_dense_equals_cached_path` still holds after the revision.

## AS-153: the Δt encoding φ
- **What is assumed.** φ(Δt) = Periodic(x̃) + w·x̃/20, with x̃ = log1p(Δt/δ₀), δ₀ = 1 ms, 8 learned
  frequencies, and initial frequency scale σ = 0.5. It is used by the causal gate and by the
  transition prior's horizon input.
- **Stands for.** A detail: the build-spec names φ(Δt) without a form.
- **Reasoning.**
  - The log compresses 1 ms … months to about 0 … 25.
  - Periodic features resolve fine differences (Gorishniy et al., NeurIPS 2022, arXiv:2203.05556,
    the same family as AS-31).
  - The trend term gives a monotone direction for "further ahead".
- **Evidence.** None beyond the cited paper. This is a design default to be checked by ablation.

## AS-154: an exact latest-state register per entity
- **What is assumed.** Each entity keeps one extra slot, outside the cells and never merged, holding an
  exact copy of its newest state. Spatial heads and TAAFT read the latest state from it.
- **Stands for.** A detail of AS-11.
- **Reasoning.**
  - Protecting the newest slot inside the cells would break AS-150's locality, because which slot is
    newest depends on later volume.
  - A copy costs one slot per entity, and it is included in the capacity bound (+1).
- **Evidence.**
  - Code run: `test_order_access_and_compatibility`: `latest_slot` returns a state for t ≥ its time and
    none before it.
  - Code run: the dense ≡ cached equivalence tests read neighbours through it.

## AS-155: Titans memory constants and form
- **What is assumed.**
  - M(x) = W₂ SiLU(W₁ x), with keys and queries ℓ2-normalised.
  - η = 0.9 and θ = 0.05 are fixed scalars.
  - ℓ = (1/N) Σ_i ‖M(k_i) − v_i‖², Titans' per-pair squared error summed over dimensions, so ∇ℓ is
    the **mean** over the trigger's pairs.
  - The gradient is written in closed form.
  - The initial memory M₀ and the projections are slow (learned) weights; the fast state is data.
- **Stands for.** A detail of AS-11. The build-spec gives the update rule only.
- **Reasoning.**
  - Titans (Behrouz et al., arXiv:2501.00663) makes α, η, θ data-dependent per token. D-36 requires
    retention per trigger, independent of volume, so α_k comes from the schedule (k only), and η and
    θ are fixed.
  - The mean over pairs makes the step size independent of how many pairs traffic produced.
  - The closed-form gradient makes the write an ordinary differentiable program, so the slow weights
    are meta-learned with plain backprop.
- **Evidence.**
  - Code run: `test_closed_form_gradient_matches_autograd` (atol 1e-6).
  - Code run: `test_volume_does_not_scale_the_step`: 1 pair or 50 copies gives the same update.
  - Code run: `test_retention_depends_only_on_the_trigger_index`: `alpha` receives k only.
  - Code runs (scratch script; 10 writes of the same 16 pairs, α = 0; loss relative to its start).
    These show why the form and the value were chosen:
    - first form, loss averaged over dimensions too, θ = 0.1: 0.995 at D = 64 and 0.9995 at
      D = 1024. The memory barely learned.
    - Titans' form (summed over dimensions):
      - θ = 0.05: 0.843 at D = 64 and 0.485 at D = 1024, monotone;
      - θ = 0.1: 0.633 at D = 64. At D = 1024 the loss reached 0.088 after 9 writes and then rose to
        0.189 at write 10 (momentum overshoot);
      - θ = 0.3: oscillates at D = 1024.
    - θ = 0.05 was chosen as the largest of these that stayed monotone at L. It depends on the scale
      of the values (learned W_V), so it must be re-checked after training.
  - Code run: `test_write_reduces_associative_loss_on_written_pairs` (default θ).

## AS-156: "as of" means position order; details of the spatial pattern
- **What is assumed.**
  - "State j exists as of state i" means j ≤ i in position order. Positions are sorted by
    (time, update, role), so this implies t_j ≤ t_i.
  - Within one update, the responder's state sees the initiator's state of the same update, but not
    the reverse.
  - The self key of the spatial heads is always kept (hop 0, no plane bits, age 0). The
    `spatial_keys` cap applies to neighbours only.
  - Causal candidates need t_j < t_i strictly, so they never include the same update.
- **Stands for.** A detail of build-spec §2.5 ("latest state as of t_i").
- **Reasoning.**
  - With equal timestamps, "as of t" is ambiguous. Position order is exactly the order in which the
    incremental path writes states. That is what makes dense ≡ cached hold for any split of the
    stream into calls.
  - Hop 0 is its own category: the contact matrices' diagonal is 0, which would otherwise set every
    plane bit on the self key.
- **Evidence.**
  - Code run: `test_masks_match_reference` (brute force).
  - Code run: `test_dense_equals_cached_path` for chunks [1], [2] and [3, 1, 4, 2, 5], R ∈ {1, 3}.
    The figures are in the build report.

## AS-157: Imagination retention counts global triggers
- **What is assumed.** The Imagination store keeps the entries of the last M_im triggers of the store,
  for all tokens at once, not the last M_im writes of each token.
- **Stands for.** A detail of AS-13 under D-36.
- **Reasoning.**
  - Forgetting is a function of the trigger count only.
  - An inactive token's old belief expires on the same schedule as everyone's.
  - No token's activity changes another token's memory.
- **Evidence.** Code run: `test_retention_is_by_trigger_count_for_every_token`.

## AS-158: TSTCT bias tables are shared by all blocks
- **What is assumed.** One hop table, one plane table, one age table and one log-Δt table (per head)
  serve every block and every pass.
- **Stands for.** A detail of build-spec §2.5.
- **Reasoning.**
  - `nn.blocks.AttnContext` carries one bias for the whole stack, which keeps biases identical across
    blocks and passes.
  - T5 shares its relative-position bias across layers in the same way (Raffel et al., JMLR 2020,
    arXiv:1910.10683).
  - Per-block tables would need a per-block context, a change to `nn/`.
- **Evidence.** None beyond the cited precedent. Per-block tables are an ablation to run.

## AS-159: the transition prior is also read from the thinking stream
- **What is assumed.** `refined_prior(out, window)` applies the same prior head to `out.refined`. The
  trainer may put the stage-3 KL on both the memory-stream prior and the refined prior.
- **Stands for.** A detail of AS-05 and AS-07.
- **Reasoning.**
  - The memory-stream prior does not depend on R. Without a second reading, the R thinking passes get
    no stage-3 gradient, are frozen in stage 4, and are trained only in stage 5.
  - The memory-stream prior remains the cache-consistent world model.
  - The refined prior makes "thinking longer predicts the next state better" measurable per R.
  - The weighting between the two KL terms is the trainer's choice, and is not set here.
- **Evidence.** Code run: `test_refined_prior_reads_the_thinking_stream`. With R = 3 it differs from the
  memory prior, and its loss sends gradient into the blocks.

## AS-160: times within 100 ns are simultaneous
- **What is assumed.** `TSTCTConfig.time_tie_s` = 1e-7 s. It applies to:
  - contact "as of t_i" tests, C ≤ t_i + τ;
  - the strict causal order, t_i − t_j > τ;
  - the sortedness check;
  - and identically to the cached path (`StepContext.from_contacts`, `_key_sets`, `lagged_slots`).
- **Stands for.** A detail of AS-10 and AS-156.
- **Reasoning.**
  - Re-basing a time between origins changes it by float64 rounding. For example, (t + 1000) − 1000 is
    not exactly t, an error of about 10⁻¹³ s at these magnitudes and about 10⁻⁹ s for a store that is a
    year old.
  - The two states of one update share t. When the update straddles a window boundary, that rounding
    turned the initiator into a "strictly earlier" causal candidate of its own responder.
  - Code run: the carry test split at position 21 was off by 0.23 before this tolerance and 3.6·10⁻⁷
    after it.
  - A cause → effect lag under 100 ns is not a meaningful network lag.
- **Evidence.** Code run: `test_windows_with_carry_equal_one_dense_window[21-*]`.

## AS-161: the contract for carrying the Environment across windows (D-51)
- **What is assumed.** The data pipeline provides four things:
  1. a stable entity key per window entity (`entity_keys`, long [B, V]), the same for a machine
     across a stream's windows;
  2. every carried entity it wants readable is listed in the next window's entity table, even
     without updates there; carried slots of absent entities are masked;
  3. contact matrices that are cumulative over the stream (first contact ever, relative to the
     window origin, negative before it);
  4. windows in time order.

  On the model side:
  - one `EnvironmentStore` per stream (`CarryPolicy.stores`) bounds the carry and gives it P3 and P4;
  - carried keys are rotated relative to the store origin and re-based to the window origin by one
    rotation;
  - carried slots are stop-gradient.
- **Stands for.** The implementation of D-51 (owner's decision).
- **Reasoning.**
  - Reusing the inference store means training reads exactly what inference reads, with the same
    bounds (Transformer-XL segment recurrence with a stop-gradient memory; Dai et al., ACL 2019,
    arXiv:1901.02860).
- **Evidence.**
  - Code run: `test_windows_with_carry_equal_one_dense_window`, R ∈ {1, 3}, split at an update
    boundary and mid-update, permuted entity table in w2, store origin 1000 s earlier. The maximum
    difference is 9.5·10⁻⁷ on memory and refined.
  - Code run: `test_no_future_leak_across_the_boundary`, `test_carried_slots_never_receive_gradient`,
    `test_bounded_carry_over_a_long_stream_with_merges`.
- **Open issue (outside this engineer's scope).**
  - The learned null key in `nn/attention.py` is appended unrotated, while queries are rotated by
    absolute time. The null logit ⟨R(t_i − o)q, k∅⟩ therefore depends on t_i − origin.
  - Code run (scratch): shifting one whole window's origin by 0.001–64 s changed `memory` by up to
    7.5·10⁻³. With rotary off the change was exactly 0.
  - Until the requested nn fix lands, outputs depend on the window and store origins. The carry
    equivalence holds only when w2 has w1's origin. The two strict-xfail tests
    (`test_carry_is_invariant_to_the_next_window_origin`,
    `test_dense_equals_cached_path_with_a_different_store_origin`) pass with the fix applied in a
    scratch run.

---

## Measured results quoted above
- Mid-age recall probe (`test_mid_age_recall_probe`):
  - setup: L memory configuration, 2 days of background states every 60 s with pairwise cosine
    ≈ 0.9, and one orthogonal state planted 1 day old or 10 s old;
  - attention weight on the planted slot: 0.9647 at 1 day and 0.9647 at 10 s (ratio ≈ 1.000; the
    asserted factor is F = 2);
  - a 512-most-recent FIFO keeps 0.9885 at 10 s and 0 at 1 day (evicted).
  - Caveat: the probe needs the planted state to be distinctive *in key space* relative to the
    background spread. With background pairwise cosine ≈ 0.5 (noise 0.3 per dimension in 8
    dimensions), the planted state was merged into a background slot (code run, first version of the
    probe). Merging chooses the most similar pair, and "distinctive" has to mean distinctive keys.
  - The probe reads unrotated content keys. TSTCT's temporal heads also rotate keys by time, which
    attenuates content matching over long Δt in the fast-frequency pairs. Retrieval at long Δt relies
    on the slow-frequency pairs and the learned log-Δt bias. This is not measured here.
