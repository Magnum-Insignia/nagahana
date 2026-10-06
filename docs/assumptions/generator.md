# Assumptions of the Generator (AS-350 … AS-370), 2026-10-02

Engineer F (Generator: training-only augmentation, outside the ≈ 1 B count). Each entry is an
engineering assumption of the L build. It is **not** an owner decision: the held item it stands for
stays held in `governance/decisions.py` — **D-14** (which Generator families come first) stays HELD;
**P-11** (hard physics gate) and **P-23** (Generator trains on the training split only) stay PROPOSED.

All of these refine the two umbrella assumptions already in the registry:
**AS-27** (Generator families in v1, standing for D-14) and **AS-28** (physics gate enabled, standing
for P-11). The new IDs are not yet in `governance/assumptions.py` (outside this engineer's scope; the
registry entries are requested in the build report). Until then, code declares them through
`models/generator/assumptions.py::use(id)`, which calls `assume()` on the ID if it is registered and
otherwise on its umbrella (AS-27 / AS-28). Strict mode (`strict_mode(True)`) therefore blocks every one.

Findings from code runs are cited as "code run: <test or script>, <number>". Citations are to sources
the author is sure of; anything else is marked "(citation to verify)".

---

## AS-350: Flow-only export keeps the IPFIX biflow basic profile
- **What is assumed.** The "flow-only export" variant keeps exactly: addresses (entity keys), source
  and destination port, IP protocol, TCP flags (OR), bytes and packets per direction, and duration.
  Every other contributing field becomes NOT_SUPPLIED. At packet granularity (PCAP running-state rows)
  only the last row of each flow is kept, since an exporter emits one record per flow; its label is the
  flow's aggregate (malicious if any packet is; benign if all are; otherwise unknown).
- **Stands for.** D-14 (via AS-27): the concrete field set of "flow-only export" (build-spec §2.11 item 1).
- **Reasoning.** This is what a standard exporter supplies: IPFIX Information Elements octetDeltaCount,
  packetDeltaCount, protocolIdentifier, tcpControlBits, sourceTransportPort, destinationTransportPort,
  flowStartMilliseconds / flowEndMilliseconds (RFC 7012), with the reverse direction from RFC 5103
  biflows. IAT statistics, per-flag counts, TTL, windows, histograms are not in that basic profile, so
  pretending they survive would teach a NetFlow-only sensor that it sees packet detail. NOT_SUPPLIED
  (not NOT_OBSERVABLE) is the status the taxonomy gives to "the source cannot provide it (e.g. NetFlow
  has no TTL)" (datamodel/status.py).
- **Evidence.** RFC 7012 (Claise & Trammell, 2013); RFC 5103 (Trammell & Boschi, 2008). Code run:
  `test_flow_only_export_keeps_ipfix_profile_and_one_record_per_flow` (both granularities).
- **Open.** Stage-1 analysis may show exporters that add IAT or per-flag counts (e.g. CICFlowMeter
  CSVs); the profile is a parameter (`FlowOnlyExport(keep=…)`).

## AS-351: Dropping packet-level fields marks them NOT_OBSERVABLE
- **What is assumed.** The "drop packet-level fields" variant turns every contributing cell of a
  `Level.PACKET` field into NOT_OBSERVABLE (value NaN). Cells already excluded keep their own status.
- **Stands for.** D-14 (via AS-27); the brief's "drop packet-level fields → NOT_OBSERVABLE".
- **Reasoning.** It models a vantage point that cannot see packets (an untapped or encrypted segment
  monitored only by flow summaries), i.e. "cannot be seen passively" in the P-03 taxonomy. Derived
  port-scan evidence (`Level.DERIVED`) is left alone by default because it can be computed from flow
  records too; the level set is a parameter.
- **Evidence.** Code run: `test_drop_packet_level_excludes_exactly_packet_fields`.

## AS-352: 1-in-n packet sampling is independent Bernoulli(1/n) sampling with ×n rescaling
- **What is assumed.**
  - Each packet is kept independently with probability 1/n; n is drawn per variant from
    `sampling_rates` = (8, 32, 128) (engineering values spanning light to heavy sampling, not taken from
    a measured deployment).
  - Flow rows: k_d ~ Binomial(N_d, 1/n) per direction; each per-packet tally (flag counts, DF, MF,
    retransmissions) ~ Hypergeometric(N, tally, k); histogram bins ~ multivariate hypergeometric over the
    bins plus an "unbinned" rest; bytes_d = round(B_d·k_d/N_d) (the direction's mean packet size is kept —
    an approximation, since per-packet sizes are unknown in a flow record). Reported = n × sampled.
  - Packet rows (running state): a row survives with probability 1/n; running tallies are rebuilt
    exactly from per-packet increments: c_i = n·Σ_{j≤i, j kept} Δc_j; duration_i = t_i − t_first kept.
  - A flow with no sampled packet disappears. IAT statistics, TTL, initial windows, bidirectional ratio,
    derived scan evidence and application fields become NOT_SUPPLIED (their sampled values depend on
    adapter definitions and cannot be vouched for). Rows whose packet counts are unknown keep only their
    flow-key fields.
- **Stands for.** D-14 (via AS-27): "1-in-n sampling with rescaled counts" (build-spec §2.11 item 1).
- **Reasoning.** Random packet sampling is what sFlow does (RFC 3176), and ×n is the inverse-probability
  (Horvitz–Thompson) estimator of the unsampled counts, the usual basis for estimating flow statistics
  from sampled streams (Duffield, Lund & Thorup, ACM SIGCOMM 2003). Drawing the tallies from the sampled
  packets keeps flag ≤ packets and 20·packets ≤ bytes ≤ MTU·packets exactly before and after rescaling.
- **Evidence.** Horvitz & Thompson, JASA 47(260), 1952. Code runs:
  `test_packet_sampling_rescales_counts_by_n` (all counts multiples of n; IAT never contributing),
  `test_packet_sampling_rebuilds_running_tallies_exactly` (running packets_fwd equals n × surviving
  forward packets so far, every row).

## AS-353: Sensor hiding is an untapped segment of internal hosts
- **What is assumed.** A random share (`sensor_hide_fraction` = 0.2, at least two hosts) of internal
  `host` entities forms a hidden segment; updates whose initiator and responder are both in it vanish.
- **Stands for.** D-14 (via AS-27): "sensor hiding".
- **Reasoning.** A boundary tap does not see intra-segment traffic; traffic crossing the boundary is
  still seen. The variant teaches that silence inside a segment is a sensor gap, not an absence of
  activity (D-41, ARCH observability floor). Data models without a sensor column need this entity form;
  a `sensor` column form is an extension point.
- **Evidence.** Code run: `test_sensor_hiding_removes_only_intra_segment_updates`.

## AS-354: Port remapping stays inside service-alias classes; ephemeral source ports stay ephemeral
- **What is assumed.** Destination ports are permuted (a bijection, so distinct services never merge)
  inside alias classes of one service: http {80, 8080, 8008}; SMB {445, 139}. Other destination ports
  are unchanged. Source ports in 49152–65535 are re-drawn inside that range. Service entity keys
  ("addr:port/proto") follow the remap.
- **Stands for.** D-14 (via AS-27): "port remapping inside the service class" (build-spec §2.11 item 2).
- **Reasoning.** The same protocol on an alternative port is the same act; attackers and administrators
  choose such ports. SMB runs over direct TCP (445) and over the NetBIOS session service (139)
  ([MS-SMB2] §2.1, "Transport"). The class table is deliberately small: a remap between *different*
  services (e.g. SSH ↔ RDP) would change the technique. Port-based plane formation (AS-01) will see
  some remapped flows as off-port services: that is the robustness the variant trains.
- **Evidence.** RFC 6335 §6 (dynamic ports 49152–65535). IANA service-name registry: 8080 is
  "http-alt"; 8008 as "http-alt" (citation to verify against the registry snapshot used in stage 1).
  Code run: `test_port_remap_is_a_bijection_inside_alias_classes`.

## AS-355: Timing jitter is one multiplicative scale per flow
- **What is assumed.** For each flow, s = exp(u), u ~ U[−log(1+ε), log(1+ε)], ε = `jitter_rel` = 0.1;
  duration, iat_mean, iat_max × s and iat_var × s². At packet granularity the flow's packet times move
  with it: t' = t_first + s·(t − t_first).
- **Stands for.** D-14 (via AS-27): "timing jitter within the physics bounds".
- **Reasoning.** A common scale keeps every order and ratio between timing fields exactly (iat_mean ≤
  iat_max ≤ duration still hold; variance scales with the square of the unit), so the jitter cannot
  create an impossible state from a possible one. The only physics it can break is the link-rate bound
  (shorter duration, same bytes); the acceptance gate rejects such variants when a link rate is set.
- **Evidence.** Code run: `test_timing_jitter_keeps_timing_ratios` (ratios equal to 1e-12, |log s| ≤ log 1.1).

## AS-356: Rate scaling is a time dilation of the attack's flows
- **What is assumed.** Start times of flows containing malicious updates are dilated about the first
  malicious time, t' = t_a + ρ(t − t_a), log ρ ~ U[log 0.5, log 2]; each flow's internal timing is kept.
  In a window without malicious updates all flows are dilated.
- **Stands for.** D-14 (via AS-27): "rate scaling within the physics bounds".
- **Reasoning.** The problem statement singles out the "slow reconnaissance scan designed to evade
  flow-based thresholds": the same campaign at another tempo is the same act. Keeping each flow's
  internal timing keeps its per-flow physics.
- **Evidence.** Code run: `test_rate_scaling_dilates_attack_flow_starts` (exact to 1e-12 for ρ = 2).

## AS-357: Re-ordering within the recorded reorder uncertainty
- **What is assumed.** The true event time of an update is uniform in [t − r, t + r], r =
  `reorder_uncertainty_s` (NaN: ordering certain, r = 0); rows are re-sorted by the drawn times. At
  packet granularity a flow's rows keep their order (the drawn times are sorted within the flow).
- **Stands for.** The detail of build-spec §4b.5 ("permutes updates within their recorded
  reorder_uncertainty_s") for the Generator.
- **Reasoning.** Every order consistent with the uncertainty intervals is reachable, and no other; a
  running state cannot go backwards inside its flow. Flow fields are what the sensor measured and are
  not changed.
- **Evidence.** Code run: `test_reorder_moves_times_within_uncertainty_only`.

## AS-358: Topology variation touches benign-only relations, away from the attack
- **What is assumed.** Hyperedge dropout removes each relation whose every update is *known* benign
  with probability `edge_dropout` = 0.1. Rewiring gives each known-benign relation whose endpoints never
  take part in a malicious update a new initiator of the same entity kind (also never malicious) with
  probability `rewire_rate` = 0.1; the service entity stays with the responder. Unknown labels (NaN)
  count as not benign.
- **Stands for.** D-14 (via AS-27): build-spec §4b.5 "hyperedge dropout and rewiring, physics-checked".
- **Reasoning.** Attack updates and the attack's entities keep their exact neighbourhood of attack
  evidence; the background varies, so the model cannot memorise a fixed benign topology. Restricting to
  known-benign relations is what makes the variant label-preserving by construction.
- **Evidence.** Code run: `test_topology_variants_never_touch_attack_updates` (attack rows identical,
  rewired initiators of the same kind).

## AS-359: Projection of learned cells into the hard limits
- **What is assumed.** After a learned family generates cells, `limits.project_hard_limits` moves only
  those cells, in this order: sign / integrality / field width; packets consistent with kept bytes and
  a kept duration, then ≥ 1 packet; packet-bounded tallies (capped if free, else free packet directions
  raised up to their byte and link caps); bytes into [20·packets, MTU·packets] and the link rate; the
  chain iat_mean ≤ iat_max ≤ duration (raise a free upper bound first).
- **Stands for.** D-14 (via AS-27): how "hard limits applied to every variant" is realised for learned
  outputs ("hard: by construction", physics/constraints.py).
- **Reasoning.** Real (kept) cells are evidence and are never edited; a violation among kept cells is
  left for the strict check to reject. The limits themselves are not assumptions: each is a protocol or
  accounting fact (limits.py cites RFC 791, RFC 8200, RFC 9293, RFC 7323 §2.2).
- **Evidence.** Code run: `test_projection_moves_only_free_cells_and_lands_inside_the_boundary`
  (40 random trials in the suite); a 600-trial stress run (random free shares 5–95 %, garbage values
  over six orders of magnitude, MTU 1500, link rate none / 1 Gbit/s / 1 Mbit/s): 0 violations outside
  sources that already violated the chosen link rate.

## AS-360: Gate weights w_c = 1 on raw-unit residuals
- **What is assumed.** Φ_phys in the gate uses the catalogue residuals that apply to one flow record
  (flag ≤ packets ×6, bytes ≤ packets·MTU ×2, iat_max ≤ duration) with w_c = `physics_weight` = 1, on raw
  units; τ = `physics_tau` = 1e-3 (AS-28).
- **Stands for.** P-11 (via AS-28): the weights `PhysicsTerm` requires explicitly.
- **Reasoning.** For a *gate* (not a loss) the weights only decide whether any violation is tolerated.
  With raw units and τ = 1e-3, a violation of one packet, one byte or 0.032 s is rejected, i.e. the gate
  accepts nothing beyond floating-point rounding. Normalised weights (AS-15) matter for the training
  loss, not for a yes/no gate. In strict mode Φ is not computed; enabling P-11 in strict mode raises.
- **Evidence.** Code run: `test_physics_gate_rejects_planted_violation_and_names_the_residual`
  (residual 3 packets → Φ = 9, rejected; clean variant Φ = 0).

## AS-361: Energy acceptance keeps the [1 %, 99 %] quantiles of real energies
- **What is assumed.** With TAAFT's marginal energy E(∅, ·) supplied, a variant is kept iff
  Q_0.01(E_real) ≤ E(variant) ≤ Q_0.99(E_real), calibrated on real *training* windows.
- **Stands for.** D-14 (via AS-27): "a variant is kept only if TAAFT's marginal energy is within the
  real-data range" (build-spec §2.11 item 5).
- **Reasoning.** The min–max range is set by the two most extreme real windows and moves with every new
  outlier; central quantiles are stable. The lower bound rejects over-typical (collapsed) samples, the
  upper bound atypical ones. This is the JEM reading of a classifier/energy as a density score
  (Grathwohl et al., ICLR 2020, arXiv:1912.03263), used for acceptance, not for SGLD sampling.
- **Evidence.** Code run: `test_energy_acceptance_by_real_quantiles_with_a_fake_energy` (0..10 linear
  energies → range (1, 9); 0.5, 9.5 and NaN rejected).

## AS-362: Codes of the masked model
- **What is assumed.** Numeric columns: `value_bins` = 256 uniform bins in signed-log1p space over the
  column's training range, uniform dequantisation inside a bin, counts rounded. Categorical / bitmask
  columns: the `cat_vocab` − 1 = 1023 most frequent training codes, plus an OOV class decoded by drawing
  from the empirical distribution of the rare training codes (disallowed if there are none).
- **Stands for.** D-14 (via AS-27): the value representation of the masked-generative family.
- **Reasoning.** MaskGIT predicts discrete classes and needs a per-cell confidence; a class view gives
  both for every column kind. 256 bins over a ~24-unit log range is ~10 % relative resolution; finer
  detail comes back through dequantisation and is bounded by the projection. Decoded categorical values
  are always values seen in training (a port never seen cannot be invented).
- **Evidence.** Code run: `test_codec_round_trip_oov_and_absence` (continuous values return to their
  bin; rounded counts within one bin; OOV decodes only to rare training codes; absence stays NaN).

## AS-363: MaskGIT schedule and conditioning
- **What is assumed.** Training hides ⌈γ(r)·N⌉ contributing cells with γ(r) = cos(πr/2), r ~ U(0,1);
  generation runs T = `unmask_steps` = 8 iterations, keeping the ⌊γ(t/T)·N⌋ least confident cells hidden,
  with confidences = log p(sampled) + Gumbel noise × τ_choice·(1 − t/T), τ_choice = 1. Each record is
  conditioned on its stage label (15 stages + unknown).
- **Stands for.** D-14 (via AS-27): the masked-generative family's schedule.
- **Reasoning.** The cosine schedule is the one MaskGIT reports as best among the mask-scheduling
  functions it compares (Chang et al., CVPR 2022, arXiv:2202.04200); 8 steps is in the range the paper
  uses. Annealed choice noise is the paper's device against greedy collapse; τ_choice = 1 is an
  engineering value (citation to verify for the paper's own value). Label conditioning is what lets a
  learned variant copy its source label (AS-366).
- **Evidence.** Code run: `test_masked_cell_loss_decreases` — masked-cell cross-entropy 3.542 → 1.65
  after 60 AdamW steps on three synthetic windows (fixed evaluation mask, seed 0).

## AS-364: The autoregressive family is the masked network read left-to-right
- **What is assumed.** No separate network: record ℓ is generated with records > ℓ removed from
  attention. Training mixes `ar_fraction` = 0.5 truncated examples (hide cells of the last kept record
  only) with full MaskGIT examples. `max_records` = 16 records per sequence.
- **Stands for.** D-14 (via AS-27): the owner's "autoregressive" method (build-spec §2.11 item 3:
  "Run left-to-right over a record sequence, it is the autoregressive family").
- **Reasoning.** One network serves both readings, halving Generator parameters; truncation makes the
  left-to-right reading exact (no look-ahead) without a separate causal model.
- **Evidence.** Code run: `test_autoregressive_reading_has_no_look_ahead` (logits of records ≤ ℓ
  unchanged when later records change; they do change under the bidirectional mask).

## AS-365: Diffusion details
- **What is assumed.** Cosine noise schedule (s = 0.008, β ≤ 0.999, ᾱ recomputed from the clipped β),
  T = `diffusion_steps` (100 at L), ε-prediction with L_simple on target cells, ancestral sampling with
  the posterior variance β̃_t, per-column standardisation in signed-log1p space, final samples clipped to
  the column's training range. Targets per training row: a uniform share of the contributing numeric
  cells (≥ 1). Denoiser: linear in, `denoiser_blocks` = 4 residual RMSNorm + SwiGLU blocks of width
  `denoiser_hidden`, embeddings of time step, discrete classes and stage.
- **Stands for.** D-14 (via AS-27): the diffusion family.
- **Reasoning.** TabDDPM models numeric features with Gaussian diffusion and an MLP (Kotelnikov et al.,
  ICML 2023, arXiv:2209.15421); the cosine schedule works with few steps (Nichol & Dhariwal, ICML 2021,
  arXiv:2102.09672); ε-prediction and Algorithm 2 sampling are Ho et al. (NeurIPS 2020, arXiv:2006.11239).
  Categorical cells are conditions rather than diffused (TabDDPM's multinomial part is not needed when
  the categorical context is kept from the real record).
- **Evidence.** Code runs: `test_cosine_schedule_and_closed_form_forward_moments` (Monte-Carlo moments
  within 5 standard errors, 200 000 draws per step), `test_ancestral_step_with_true_noise_is_the_posterior_mean`
  (Ho et al. Eq. 7 to float64 tolerance), `test_ddpm_loss_decreases_on_a_toy` — L_simple 0.901 → 0.522
  after 150 AdamW steps (fixed evaluation draws, seed 0).

## AS-366: Learned variants regenerate 30 % of contributing cells and copy labels
- **What is assumed.** Each contributing modelled cell is regenerated with probability `regen_fraction`
  = 0.3 (≥ 1 per row); statuses are never generated; labels are copied from the source rows.
- **Stands for.** D-14 (via AS-27): how a learned family produces a "variant of the same event".
- **Reasoning.** Regenerating a share keeps the rest of the act real, the generation is conditioned on
  the label, and the variant must pass the physics gate (and the energy range when supplied). Copied
  labels are *not* guaranteed by construction; every learned variant is tagged `label_mode="copied"` so
  evaluation (label preservation, TSTR; ARCH §7) can audit them separately.
- **Evidence.** Code run: `test_learned_producers_keep_statuses_copy_labels_and_obey_limits`.

## AS-367: Generator sources are real training-split samples only
- **What is assumed.** `VariantPipeline.generate`, `fit_masked` and `fit_diffusion` accept only
  `Origin.REAL`, `Split.TRAIN` samples. Zero-shot samples are refused always; validation samples and
  generated samples are refused.
- **Stands for.** P-23 (proposal: "Split before the Generator trains"); the zero-shot refusal follows
  from decided D-23.
- **Reasoning.** A variant of a validation or zero-shot sample would carry held-out content into
  training: the data-snooping pitfall (Arp et al., "Dos and Don'ts of Machine Learning in Computer
  Security", USENIX Security 2022). Variants of variants would break the one-step provenance chain that
  `pipeline/splits.validate` checks.
- **Evidence.** Code runs: `test_only_real_training_samples_feed_the_generator`,
  `test_variants_are_generated_train_samples_and_zero_shot_never_sees_them`.

## AS-368: An attack window's variant keeps at least one malicious update
- **What is assumed.** A variant of a window with malicious updates that loses all of them (e.g. by
  sampling or sensor hiding) is rejected ("attack-erased").
- **Stands for.** D-14 (via AS-27).
- **Reasoning.** Per-update labels would still be correct, but the window's family label would not, and
  the attack-share budget (§4b.6) would silently be spent on benign windows.
- **Evidence.** Code run: `test_duplicates_and_erased_attacks_are_rejected`.

## AS-369: Provenance inside variant tables
- **What is assumed.** Variant tables have zero `raw_hash` rows, `record = −1`, an `origin =
  "generated"` and a `derived_from_seq` column, `adapter = "nagahana-generator:<producer>"` and
  `source_id = "<real source>~<variant id>"`; variant ids are "<real sample id>~gNNNN".
- **Stands for.** The detail of D-23 / P-23 provenance at row level.
- **Reasoning.** A generated row has no raw record; copying the real record's hash would claim a chain
  of custody (ARCH §7, §10) for something never on the wire.
- **Evidence.** Code runs: `test_every_transform_preserves_labels_absence_and_limits`,
  `test_variants_are_generated_train_samples_and_zero_shot_never_sees_them`.

## AS-370: Attack-share budget by largest remainder
- **What is assumed.** With attack and benign windows present, attack windows get round(attack_share·n)
  variants (attack_share = 0.7), split by the largest-remainder method; if one class is absent, the other
  gets the whole budget and a note says the share could not be honoured.
- **Stands for.** The detail of build-spec §4b.6 ("the Generator draws most of its variants from attack
  events").
- **Reasoning.** Exact totals, per-window counts within one of each other, and an explicit note instead
  of a silent change of the share.
- **Evidence.** Code runs: `test_attack_share_budget_is_exact`,
  `test_generate_many_draws_most_variants_from_attack_windows` (14 of 20).

---

## Code-run summary (tiny widths, CPU, synthetic windows; 2026-10-02)
- Parameter count at the L `GeneratorConfig`, PCAP adapter column layout (40 columns), meta device:
  masked-generative / autoregressive (shared) 42,517,504; diffusion 45,252,640; **total 87,770,144**
  (separate from the ≈ 1 B model).
- End-to-end smoke (scratch script, three synthetic attack windows, all 13 producers, budget 90): 90
  accepted; rejections: 9 duplicates, 3 attack-erased, 0 physics or hard-limit rejections. These are
  synthetic windows: the numbers say the machinery works, not how a real corpus will behave.
