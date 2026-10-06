# Perception assumptions (AS-100 … AS-149)

Engineer A (Perception: FieldEncoder, graph builder, CVG-AE, latent KL, Decoder), 2026-10-02.

These are engineering assumptions of the L build, in the sense of `governance/assumptions.py`: each one
is this build's working choice, never the owner's decision. Where an assumption stands in for a held
decision, that decision stays held. The IDs below are **not yet in the code registry**
(`governance/assumptions.py` is outside Perception's scope); the code cites them in docstrings and
comments, and calls `assume()` only for the registered AS-01 … AS-41 it relies on. The registry entries
are requested in the build report.

Format: ID · what is assumed · what it stands for · reasoning · evidence.

---

## AS-100 · FieldEncoder numeric clip and the fixed categorical hash
- **Assumed.** (a) |signed_log1p(x)| is clipped at `max_log_magnitude` = 50 before the periodic
  embedding. (b) The categorical row is h(slot, code) = ((A₂·((A·slot + B·code + C) mod p) + C₂) mod p)
  mod R_hash with p = 2³¹ − 1 and fixed constants (`models/inputs/encoder.py`). (c) Bits of a bitmask
  at or above `max_bits` (16) are ignored. (d) Categorical and bitmask codes are read from the float32
  value matrix, so they are exact up to 2²⁴ (ports, protocols, flag masks and small vocabularies are far
  below).
- **Stands for.** Detail of AS-31 / AS-33 (the hash function was unspecified).
- **Why.** (a) e⁵⁰ ≈ 5·10²¹ exceeds any byte or packet count; the clip only guards against corrupt
  inputs driving the periodic features into aliasing. (b) The hash must be deterministic across runs
  and platforms: Python's `hash()` is salted per process for strings, and int64 overflow is not a
  portable behaviour; the chosen form keeps every intermediate below 2⁶², so it is exact. A member of a
  universal family keeps collisions between (slot, code) pairs near the 1/R_hash rate.
- **Evidence.** Carter & Wegman, "Universal classes of hash functions", JCSS 18(2), 1979. Code run:
  `tests/test_perception_inputs.py::test_categorical_hash_is_fixed_and_deterministic` (500 random pairs
  equal a pure-Python reference).

## AS-101 · Fan hyperedges: episodes and an as-of `since`
- **Assumed.** Per plane and initiator, the initiator's updates are split into episodes at gaps
  > `fan_window_s` (60 s). A fan *qualifies* at the first update where ≥ `fan_min_responders` (8)
  distinct responders were touched within the trailing `fan_window_s`. Its `since` is that qualifying
  update; member y joins at max(qualification, first touch of y in the episode); every later touch of
  the episode is activity. The episode's first touch is kept as the sweep's start time.
- **Stands for.** Detail of AS-02 ("time-stamped by its first member").
- **Why.** Stamping the fan with its first member would let a position between the first touch and the
  8th responder see a sweep that was not yet evident: future leakage. The sweep's start time is kept
  for explanation, but visibility follows the update that made the fan evident (build-spec §2.2,
  "since (the event time of the update that created it)").
- **Evidence.** Code run: `test_fan_appears_only_after_qualification` (the fan is absent from positions
  0–6 and present from position 7, the 8th responder).

## AS-102 · Local subgraph: radius, cap, recency, time-to-live
- **Assumed.** Radius `max_hops` = 2 (new GraphConfig field). BFS over *alive* hyperedges: a hyperedge is
  alive at a position if it exists as of the position and its last activity is within
  `hyperedge_ttl_s` (3600 s, existing field). Each hop's candidates are ordered by recency (the latest
  activity of an alive hyperedge joining them to the frontier; ties by entity index) and the subgraph
  is capped at `max_nodes`. The subgraph keeps every alive hyperedge with ≥ 2 selected members.
  `hyperedge_entities` lists the hyperedge's members as of the position (not only those in the
  subgraph), truncated to `max_members` (8) in join order.
- **Stands for.** build-spec §2.2 says "up to `cvg_layers` hops"; the brief says hop ≤ 2.
- **Why.** Hop ≤ 2 equals the TSTCT spatial reach (C¹, C² ≤ t), and since D-52 the BFS steps only along
  communication (star), so CVG-AE and TSTCT see the same neighbourhood. Time-based expiry follows the builder notes (never count-based, D-36 spirit): a contact
  from hours ago should not crowd out the current neighbourhood. Contact matrices do **not** expire
  (first contact is monotone history; the masks need it).
- **Evidence.** Code run: `test_hyperedges_expire_by_time_to_live`, truncation-invariance tests.

## AS-103 · RWSE on the weighted clique expansion
- **Assumed.** Per plane, A_uv = Σ_{e ∋ u,v; u≠v} 1/(|e| − 1) on the local subgraph (|e| counted inside
  the subgraph); RWSE = diag(Mᵏ), M = D⁻¹A, exactly as `nn.positional.random_walk_se`.
- **Stands for.** Detail of D-49 ("RWSE per plane"; the clique-expansion weights were unspecified).
- **Why.** Unweighted clique expansion lets one large fan dominate every walk (a sweep of 30 hosts adds
  435 edges); the 1/(|e| − 1) weight makes every hyperedge spread one unit of walk mass, which is the
  hypergraph random walk of Zhou et al. without its "stay" step. D-52 keeps this clique expansion for RWSE:
  it encodes the structure a node sits in (one of thirty hosts swept together), not contact.
- **Evidence.** Zhou, Huang & Schölkopf, "Learning with Hypergraphs: Clustering, Classification, and
  Embedding", NeurIPS 2006. Code run: `test_batched_rwse_equals_shared_primitive`.

## AS-104 · Relation identity, hyperedge kinds, plane matching
- **Assumed.** A relation (base hyperedge) is (distinct member set, kind) with kind = group if a member
  is a `multicast` entity, else session if protocol = 6, else exchange (UDP, other and unknown protocol).
  Port-ruled planes match the destination port when the protocol carries ports (TCP 6, UDP 17,
  SCTP 132) or is unknown; an unknown port matches no port rule; `services` = the update goes through a
  service entity. `update_planes` holds the planes of the update itself; an update joining < 2 distinct
  entities has no relation (−1) and no planes.
- **Stands for.** Detail of AS-01 / AS-02 (held D-04).
- **Why.** Kind priority puts the D-47 group semantics first (an LLMNR query to 224.0.0.252 is a group
  act whatever its transport). Ports of non-port protocols (CICFlowMeter writes 0 for ICMP) must not
  create planes. The relation's planes accumulate as of time through `contact_planes`; using the
  relation's final planes on every update would leak later updates' evidence.
- **Evidence.** IANA port and protocol registries (cited in `graph/planes.py`); code run:
  `test_plane_rules_on_known_ports`, `test_relations_kinds_and_update_planes`.

## AS-105 · Visibility order; contact semantics now decided (D-52)
- **Assumed.** Updates are ranked by (time, index). With `pos_update` given, a position sees exactly the
  updates of rank ≤ its own update's rank; without it, every update with t ≤ t_p (ties included).
- **Decided, not assumed (D-52, owner 2026-10-02): star semantics for contact.** Contact means
  communication. In a hyperedge the hubs are every member of a session or exchange (one flow), the
  initiator of a fan, the multicast member(s) of a group; two members are in contact through it iff at
  least one is a hub. A fan therefore joins its initiator with each responder, a group joins each sender
  with the group entity, and co-members of a fan or group are not in contact: they are 2 hops apart and
  meet in C² through the hub. Pairs join at the later of their two joins; diagonals are 0. The local
  subgraph's BFS (hops) follows the same rule, so `node_hop` is contact distance; RWSE keeps the clique
  expansion (AS-103).
- **Stands for.** The visibility order is a detail of build-spec §2.2 (ties were unspecified). The earlier
  clique reading of contact (co-membership) is withdrawn by D-52.
- **Why.** Lab datasets have second-resolution timestamps (many ties); the strict (time, index) order
  matches the position order (time, update, role) of `PositionBatch`. Under clique semantics one /16
  sweep would make every swept host a 1-hop neighbour of every other, flooding spatial attention with
  pairs that never exchanged a packet (D-52 note).
- **Evidence.** D-52 in `governance/decisions.py`. Code run: `test_contact_matrices_equal_brute_force`
  (brute force under star semantics), `test_star_contact_fan_responders_are_two_hops_apart`,
  `test_no_future_leak_truncation_invariance`.

## AS-106 · Service-class bucket tables
- **Assumed.** `models/decoder/buckets.py`: 50 named port classes (planes' ports first, then common IT
  services), four range classes at the end (0 | 1–1023 | 1024–49151 | 49152–65535, RFC 6335 §6), named
  class i kept when i < n − 4; 14 protocol classes + "other"; other categorical columns: code itself for
  0 ≤ code ≤ n − 2, else "other".
- **Stands for.** Detail of AS-33 (the bucketing function).
- **Why.** The world model needs the kind of service, not the exact port; the same function works at
  n = 64 (L) and n = 8 (tiny). Tables are append-only.
- **Evidence.** IANA registries; RFC 6335 §6; code run: `test_service_class_buckets`.

## AS-107 · Pooling attends to every cell; MASK has its own vectors
- **Assumed.** The attention pooling of the FieldEncoder attends to all C field states, absent ones
  included (their state is s_c + σ_status + a_status). MASK has its own σ and a vectors. The pooled
  vector gets a final RMSNorm. No null key (every update has C ≥ 1 states).
- **Stands for.** Detail of build-spec §2.1.
- **Why.** D-41: "not observable here" is a fact about the sensor the model should be able to use;
  excluding absent cells from pooling would hide it. The value is still never read.
- **Evidence.** Code run: `test_nan_never_propagates_and_excluded_cells_are_ignored`,
  `test_mask_status_hides_the_value`.

## AS-108 · CVG-AE details
- **Assumed.** One coupling matrix ω per layer (init 0); hop bias carried by the hyperedge
  (hop(e) = min member hop, buckets `max_hops_bias`); size bias by ⌊log₂|e|⌋ (`size_buckets` = 8); QK-norm
  on queries and hyperedge messages; g_v = 0 for a node in no hyperedge of a plane; typed MLP = Linear →
  GELU → Linear (hidden `mlp_mult`·dim); node input is a sum of embeddings (≡ a linear map of the
  concatenation) plus a periodic log(1 + age) encoding and a learned log-time age bucket (no decay with
  age imposed, build-spec §4b.1); log σ² = 10·tanh(raw/10); the last layer computes only centre rows.
  `logits` are returned raw; the posterior categorical is formed with `unimix_logits`.
- **Stands for.** Details of AS-03 / AS-05.
- **Why.** A bias depending only on the receiving node cancels in the softmax over its hyperedges, so
  the hop bias must sit on the hyperedge. Per-layer ω lets coupling differ between shallow and deep
  layers at negligible cost. Mixing in one place avoids applying unimix twice.
- **Evidence.** Hu et al., HGT, WWW 2020 (arXiv:2003.01332); Feng et al., HGNN, AAAI 2019
  (arXiv:1809.09401). Code run: `test_permutation_equivariance`, `test_gradients_reach_every_parameter`,
  `test_last_layer_restriction_equals_full_layer`.

## AS-109 · KL details
- **Assumed.** Free bits clip the total KL per position (Gaussian + Σ groups), as DreamerV3 clips its
  summed KL; unimix (1 %) is applied to both posterior and prior inside `kl_balanced`, from raw logits;
  `kl_standard` uses N(0, I) × Uniform.
- **Stands for.** Detail of AS-05.
- **Why.** Per-dimension free bits would let many dimensions each hold up to 1 nat for free; the summed
  form matches the reference setting.
- **Evidence.** Hafner et al., DreamerV3, arXiv:2301.04104, §3. Code run:
  `tests/test_perception_latent.py` (closed forms against `torch.distributions`).

## AS-110 · Decoder likelihood details
- **Assumed.** Gaussian in signed-log1p space with a learned, z-dependent log-scale clamped to
  [−5, 3] (Jacobian omitted: it does not depend on ψ); log-space means clipped at ±50; nonnegative
  columns = COUNT, HISTOGRAM and CONTINUOUS columns with a physical unit in the catalogue; decoded value
  = signed_expm1 of the log-space mean (the model's median); status weights of AS-30 in the NLL; decoded
  values all count as contributing for Φ_phys (they are model outputs).
- **Stands for.** Detail of build-spec §2.4 / AS-04.
- **Why.** Relative errors suit magnitudes spanning ten orders; clamping keeps σ away from collapse and
  explosion. Columns without a unit (`flow.bidir_ratio`, port-scan evidence) have source-specific
  definitions and are left unconstrained.
- **Evidence.** Code run: `test_hard_limits_hold_for_random_latents` (z scales 1, 30, 10³),
  `test_nll_closed_forms`.

## AS-111 · Candidate-hyperedge negatives
- **Assumed.** One negative per observed hyperedge (`edge_negatives` = 1): one uniformly chosen real
  member is replaced by the latent of a member drawn uniformly from all observed hyperedges of the
  batch; loss = balanced BCE. Accidental true hyperedges among negatives are not filtered.
- **Stands for.** Detail of build-spec §2.4 ("sampled negatives (members swapped)").
- **Why.** Swapping one member keeps negatives hard (they share the rest of the set), the usual recipe for
  link prediction. Filtering false negatives needs the full window's hyperedge set at loss time; the
  rate is small when the number of entities is much larger than hyperedge sizes.
- **Evidence.** Code run: `test_edge_logits_are_set_functions_and_ignore_padding`.

## AS-112 · Provenance rules of the view
- **Assumed.** Sources: environment, belief, forecast. OBSERVED only for Environment elements backed by
  an observation (a contributing observed cell, an observed hyperedge); every other Environment element
  and every belief element is BELIEVED; every forecast element is FORECAST. Imagination views refuse
  observed inputs, and `DecodedView` refuses an OBSERVED element from Imagination at construction.
  Unobserved candidate hyperedges are shown only when p ≥ threshold.
- **Stands for.** D-05 held (AS-29), P-01.
- **Why.** "Beliefs never render as facts" (architecture §3.6); enforcing it in the type makes the
  invariant independent of callers.
- **Evidence.** Code run: `test_provenance_invariant_imagination_is_never_observed`.

## AS-113 · Stage-3 masking and reordering
- **Assumed.** Each contributing cell is masked independently with probability 0.15 (status → MASK,
  value → NaN in the masked copy). Reordering: t' = t + U[−r, r] per update (float64), permutation =
  stable argsort of t'; updates with r = 0 keep their exact time; padded updates stay last.
- **Stands for.** Detail of build-spec §3 and §4b.5.
- **Why.** Independent masking gives the stated expected ratio without coupling cells; uniform jitter
  inside the recorded uncertainty only produces orders the sensor could not rule out.
- **Evidence.** Code run: `test_mask_fields_touches_only_contributing_cells` (rate in (0.10, 0.20)),
  `test_permute_within_uncertainty_stays_inside_the_interval`.

## AS-114 · Further unconditional residuals and relative normalisation
- **Assumed.** New residuals: DF/MF-flagged and retransmitted packets ≤ packets; iat_mean ≤ duration;
  iat_var ≤ iat_max²/2; bytes_d ≥ packets_d × minimum IP header (site value required, 20 for IPv4).
  Normalised form r̂ = r / (1 + |bound|) with the bound detached (`physics/normalise.py`).
- **Stands for.** AS-15 ("residuals normalised by their field scale"); D-18 catalogue.
- **Why.** Each bound is a fact of IP flow accounting or elementary statistics, so it never penalises
  real traffic. iat_var ≤ M²/2 holds for both the population and the sample variance of gaps in [0, M].
  Relative scaling makes a violation of one packet in a 3-packet flow count like 33 %, not like one
  packet in a million.
- **Evidence.** Bhatia & Davis, "A Better Bound on the Variance", American Mathematical Monthly 107(4),
  2000; RFC 791 (IPv4 header ≥ 20 bytes); RFC 8200 (IPv6 header 40 bytes). Code run:
  `test_new_residuals_and_relative_scaling`.
