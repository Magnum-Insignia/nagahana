# Assumptions of the agents: Forecaster, Advisor, Verifier (AS-250 … AS-299)

Engineer D, 2026-10-02. Each entry: what is assumed, the held decision or unspecified detail it stands
for, the reasoning, and the evidence. None of these is an owner decision; the held decisions they
touch stay held in `governance/decisions.py`. They are not yet in `governance/assumptions.py` (outside
this engineer's scope); the registry entries are requested in the build report, and until they are
added the code names them in comments and docstrings instead of calling `assume()`.

The existing assumptions this work relies on (and calls through `assume()`): AS-17, AS-18, AS-21,
AS-22, AS-23, AS-24, AS-25.

---

### AS-250: imagined steps carry actions; the state is the step's output
- **Assumed:** the input of imagined step k is the action embedding (technique slot + target's context
  encoding); the imagined state s_k is the transformer's output at that step. A predicted state is
  never fed back as an input.
- **Stands for:** detail of build-spec §2.8 ("Step k carries s_k plus an action embedding").
- **Why:** with action-only inputs, teacher forcing on real action sequences and imagination with
  sampled actions are the *same* computation, so there is no train/test mismatch (exposure bias). The
  state is still carried: it is the residual stream that every later step reads through causal
  attention. This keeps the build's principle that training and inference compute the same function
  (AS-06).
- **Evidence:** exposure bias of feeding ground truth in training and predictions at test time: Bengio
  et al., "Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks", NeurIPS 2015,
  arXiv:1506.03099. Code run: `tests/test_forecaster_model.py::test_dense_teacher_forcing_equals_incremental_imagination_path`
  (dense teacher forcing equals the incremental imagination path to 1e-5).

### AS-251: route weights and the merge of identical routes
- **Assumed:** `ForecasterConfig.route_estimator = "monte_carlo"` (lead's decision, see below):
  w_r = count_r / N over distinct routes. Alternative `"probability"`: w_r ∝ q(r) = Π_k π̃(a_k | s_k)
  over *distinct* routes, merged copies weight 0.
- **Stands for:** detail of build-spec §2.8 ("w_n ∝ Π_k π̃(a_k | s_k)"; "identical action sequences are
  merged and their weights summed") and D-46.
- **Why:** identical sequences have identical q(r); summing the copies' q would weight a route by
  count·q ≈ N·q², which over-weights the mode and is not a consistent estimator. Read as written, the
  build-spec formula is the self-normalised distinct-set estimator.
- **Caveat (important):** when the route space is huge (700 techniques × ~48 targets per step, K = 12),
  almost every sampled route is distinct and their q differ by orders of magnitude; the "probability"
  estimator then concentrates on the most probable sampled route (high variance, biased toward the
  mode). The "monte_carlo" estimator is unbiased for E_π̃[F]. The owner should choose; both keep P_inf
  monotone and both are implemented and tested.
- **Evidence:** code run: `tests/test_forecaster_routes.py::test_identical_routes_are_merged_and_distinct_at_most_n`.

**Lead's decision (2026-10-02):** the default is now `monte_carlo`. Routes are drawn from π̃, so
the plain average over draws, count/N per distinct route, is the unbiased estimate of
E_π̃[F]. Weighting distinct draws by π̃ again counts the policy twice. With about 700 × 48 actions
per step, almost every draw is distinct, so that weighting concentrates on the single most probable
draw. The build-spec §2.8 formula was the error; it has been corrected.

### AS-252: the exposure term of the adversary reward
- **Assumed:** exposure_k = max(0, E(∅, ŷ_k) − Ē_0), where ŷ_k is the Forecaster's imagined hypothesis
  of the target (a head into TAAFT's hypothesis space d_y), E(∅, ·) TAAFT's marginal energy (AS-16),
  and Ē_0 the mean marginal energy of the trigger's selected entity hypotheses. Exposure is 0 when no
  marginal-energy callable is supplied. The exposure part is computed, not learned; the reward head
  learns only the label-derived part (Δ progress + β·infiltration).
- **Stands for:** detail of AS-17 (held D-11c) "exposure is the marginal-energy novelty of the step".
- **Why:** energies have arbitrary offsets per term (ADR-0007), so novelty is measured relative to the
  trigger's own baseline; only standing out *more* than now costs the adversary (becoming more normal
  is not rewarded beyond zero cost).
- **Evidence:** none beyond the design; to be checked once TAAFT's energy is trained.

### AS-253: λ of the TD(λ) value targets
- **Assumed:** λ = 0.95 (`ForecasterConfig.td_lambda`), γ = 0.97 (existing config).
- **Stands for:** detail ("TD(λ) value targets", build-spec §3).
- **Why/evidence:** λ = 0.95 is the value used for λ-returns in DreamerV3 (Hafner et al.,
  arXiv:2301.04104); λ-return definition: Sutton & Barto, *Reinforcement Learning: An Introduction*,
  2nd ed., 2018, §12.1. Code run: `tests/test_forecaster_losses.py::test_lambda_returns_match_the_recursion`.

### AS-254: criticality per entity kind
- **Assumed:** host 1.0, service 1.5, account 1.0, ot_device 5.0, external 0.2, subnet 3.0,
  application 1.5, multicast 2.0 (`AdvisorConfig.criticality`, aligned with `vocab.NODE_KINDS`).
- **Stands for:** the criticality table of AS-23 (held D-03b).
- **Why:** only the order is argued: OT devices highest (D-34, availability first in CII); a subnet
  isolates many machines; an external entity is outside the site's operations. Magnitudes are starting
  values for the owner or the site to set; when entity kinds are unknown the Advisor prices every
  target at the most critical kind and says so in its info.
- **Evidence:** none (a pricing choice by definition).

### AS-255: counter effects and the information-value model
- **Assumed:** effects are structural edits of the imagined state: isolate (no targeting + no outbound
  reads), inbound filtering (no targeting), outbound filtering (no outbound reads), plane block
  (techniques of the plane cannot target the entity; needs a technique → plane table, absent today, so
  such actions are reported as "effect not modelled"), observe (sensor level: no change to attacker
  capability). Information value of a sensor action with strength ρ on entity v:
  IV = ρ·(1 − trust_v)·H(stage posterior_v), summed over distinct (action, entity) pairs and capped at
  H per entity. Strengths per action are in `models/advisor/d3fend.py`.
- **Stands for:** detail of build-spec §2.9 (effect model, information value).
- **Why:** structural effects cannot be "learned" by the Advisor to flatter its own objective (no
  reward hacking through the effect model); the IV model is the expected entropy reduction when the
  observation reveals the true stage with probability ρ(1 − t).
- **Caveat:** deterrence and detection-driven attacker behaviour changes are not modelled, so sensor
  actions show ΔP_inf ≈ 0 and are valued by IV alone. D3FEND IDs in the table are unverified (no
  network access); names and tactics are the well-known matrix entries.
- **Evidence:** code run: `tests/test_advisor_model.py::test_cost_information_value_and_rules`.

### AS-256: counters act at the trigger, all at once
- **Assumed:** all steps of a counter sequence take effect at the trigger time, so a plan's effect
  depends on its set of steps; the beam evaluates each set once.
- **Stands for:** detail (build-spec §2.9 "sequences of up to L_seq steps").
- **Why:** staggering counters in time needs per-step interventions inside the imagination loop; the
  structure is ready for it (an intervention per imagined step) but it multiplies search cost.
- **Evidence:** none.

### AS-257: model-based policy improvement for the Advisor
- **Assumed:** MPO-style improvement: target q_c ∝ sg π_D(c)·exp(J_c/η) over the evaluated candidates,
  η = 0.1 (`AdvisorConfig.improvement_temperature`), 8 candidates per trigger; rule-infeasible
  candidates get q = 0; the value head regresses J.
- **Stands for:** detail ("model-based policy improvement on −ΔP_inf − κ·cost", build-spec §3).
- **Why/evidence:** Abdolmaleki et al., "Maximum a Posteriori Policy Optimisation", ICLR 2018,
  arXiv:1806.06920 (E-step re-weighting by exp(Q/η), M-step fitting the policy). η = 0.1 is chosen
  for J in ΔP units (|ΔP| ≤ 1); not tuned.

### AS-258: Page–Hinkley drift alarm on latents
- **Assumed:** per region, the statistic is the standardised squared deviation of each new batch from
  the running Welford moments, x_t = mean_d (z − μ)²/σ² (≈ 1 when stable), fed to Page–Hinkley with
  δ = 0.1, λ = 10, after a warm-up of 30 rows.
- **Stands for:** detail of build-spec §2.10 ("Welford mean and variance … with a Page–Hinkley alarm").
- **Why:** one scalar per batch that grows with both mean and variance shifts of the latent
  distribution. δ and λ trade detection delay against false alarms and should be set from a measured
  false-alarm rate (architecture §7: drift-detection delay at a fixed false-alarm rate).
- **Evidence:** Page, Biometrika 41 (1954); Hinkley, Biometrika 58 (1971). Code run:
  `tests/test_verifier_monitor_gate.py::test_page_hinkley_alarms_on_latent_drift_only`.

### AS-259: systematic-gap alert threshold
- **Assumed:** alert when |Σ(f − y)|/12 > 0.2 over the last 12 scored resolutions (`gap_threshold`).
- **Stands for:** detail of build-spec §2.10 (the window of 12 is specified; the threshold is not).
- **Why:** a mean residual of 0.2 over 12 resolutions is well above the sampling noise of a calibrated
  rare-event forecaster but small enough to catch a slow poisoning; to be set from data.
- **Evidence:** none.

### AS-260: action labels from dataset annotations
- **Assumed:** the target of a malicious update is its responder; a step window with labelled updates
  but no malicious one is the "no target" action (code −2); the technique of a step is the technique
  of its first malicious update with a known technique.
- **Stands for:** detail ("behaviour cloning on technique labels where known", build-spec §3).
- **Caveat:** for C2 and exfiltration the internal victim *initiates*; the responder is then the
  external server. A per-family direction table (with AS-34's label mapping) would fix this.
- **Evidence:** none.

### AS-261: ranking key of counter sequences
- **Assumed:** feasible first; then CVaR_α(ΔP_inf) + κ·cost (κ = `cost_weight`); then cost.
- **Stands for:** the reading of build-spec §2.9 "CVaR, then cost, then feasibility" with the training
  objective −ΔP_inf − κ·cost (AS-23, AS-24; held D-03b, D-03c).
- **Why:** a strictly lexicographic order on a continuous CVaR would make cost matter only on exact
  ties; feasibility is treated as a filter (an infeasible counter is still reported, below the
  feasible ones, with its reasons).
- **Evidence:** code run: `tests/test_advisor_model.py::test_advise_is_always_advisory_and_ranked`.
