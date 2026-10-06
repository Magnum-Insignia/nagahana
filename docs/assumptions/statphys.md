# Assumptions of the statistical-physics module (AS-760 ... AS-779)

Implementation of decision D-56 (statistical-physics readouts and early warning: energy, entropy and
free-energy trajectories of the network state; Shannon and von Neumann entropies of the multiplex graph;
critical-slowing-down indicators; wired into the TAAFT readouts, the inference engine and the
evaluation), with D-42 (the energy is the sum of the lens terms) and D-54 (fp64 outputs). Each entry is
an engineering assumption, not a decision: the decisions it details stay as recorded in
`governance/decisions.py`. Code: `src/nagahana/statphys/`; documentation: `docs/statphys.md`.

Findings from code runs are cited as "code run: <test>, <number>".

---

### AS-760: The Gibbs view is read over five finite ensembles of imagined states and routes
- **Assumed:** at every trigger the Gibbs readouts are formed over (1) the active entity tokens, with
  their per-token energies at the refined hypotheses (`token_energy`, the sum of the lens per-token
  maps); (2) the active adversary-hypothesis slots, same energies; (3) for each entity, its stage classes
  with energies -logit_s; (4) the N imagined routes of the Forecaster; (5) for each entity, the imagined
  route steps that target it.
- **Stands for:** detail | D-56 ("a Gibbs view over imagined states or routes", "per entity and per time")
- **Why:** these are the sets of imagined states the model produces at a trigger whose energies are
  computed exactly, so log Z is exact (log-sum-exp). The continuous hypothesis space y has no tractable
  partition function: a Laplace approximation would need the log-determinant of the energy Hessian at a
  minimum, and the robust (Lorentzian) lens terms do not make E_total convex, so y_hat after S descent
  steps is not guaranteed to be a minimum. The token ensembles read how the energy is spread over
  entities and over competing adversary hypotheses at a time; the stage ensemble gives a per-entity
  novelty (its free energy -log sum_s exp(logit_s) is the energy score of a classifier head); the route
  ensembles read the imagined futures.
- **Evidence:** Liu, Wang, Owens and Li, "Energy-based Out-of-distribution Detection", NeurIPS 2020,
  arXiv:2010.03759 (free energy of logits as a novelty score); Grathwohl et al., ICLR 2020,
  arXiv:1912.03263 (logits as an energy-based model). Code run:
  `test_statphys_trajectory.py::test_reading_ensembles_routes_and_entities` (every ensemble equals the
  direct Gibbs computation to 1e-12).

### AS-761: One Gibbs temperature per trigger, the Verifier temperature of the forecast family
- **Assumed:** every Gibbs reading of the inference engine at a trigger uses one temperature T: the
  Verifier temperature in force for the output family `gibbs.verifier_family` (default `p_inf`), or the
  fixed `gibbs.temperature` (default 1.0) when the family is null. TAAFT's own thermodynamic readouts
  are computed at the native temperature T = 1.
- **Stands for:** detail | D-56 ("T the Verifier temperature (configurable)"), D-21, D-45
- **Why:** the Verifier temperature is the only temperature of the system that changes, and it changes
  only under a human command (D-21), so the readings gain no ungoverned free parameter. A Gibbs
  distribution over -E / T is a softmax of the energies, and temperature scaling of a softmax is the
  Verifier's calibration form. The family `p_inf` is used because the readings accompany P_inf at every
  trigger; the Verifier's `TemperatureState` admits only its four output families, and adding an energy
  family would widen the policy head (`VerifierConfig.n_output_families = 4`) and change the parameter
  count. T = 1 for TAAFT keeps the model's outputs a function of weights and inputs only: TAAFT's
  energies are nats and exp(-E_total) is the product of the lens experts (D-42).
- **Evidence:** Guo, Pleiss, Sun and Weinberger, "On Calibration of Modern Neural Networks", ICML 2017,
  arXiv:1706.04599 (temperature scaling). Code run:
  `test_statphys_engine.py::test_gibbs_temperature_follows_the_verifier_family`.

### AS-762: An empty ensemble is flagged, and finite in training graphs
- **Assumed:** an ensemble without a member has no Gibbs distribution: its `valid` flag is False; the
  engine's readings and the evaluation arrays report NaN for its potentials; TAAFT's readouts report 0.0
  next to `thermo/<ensemble>/valid` = False.
- **Stands for:** detail | D-41
- **Why:** absence is not zero (D-41), so reported values say NaN. Inside the model a NaN readout would
  reach every loss and every gradient that sums readouts (the TAAFT gradient test sums all floating
  readouts); a finite fill with an explicit flag keeps both the training graph and the meaning intact.
- **Evidence:** code run: `test_statphys_taaft.py::test_empty_ensembles_stay_finite_and_differentiable`
  (an invalid trigger: flags False, readouts 0.0, every parameter gradient finite);
  `test_statphys_thermo.py::test_gradients_are_finite_with_empty_ensembles`.

### AS-763: Growth rates are least-squares slopes over five valid points, read across calls
- **Assumed:** the growth rate of a series at a valid point is the least-squares slope (per second) over
  the last `growth_window` = 5 valid points up to it (2 gives the backward difference); it is computed by
  statphys (the engine's tracker across calls, `trajectory.window_series` within windows), never inside
  TAAFT; the engine tracks the energy growth of at most `growth_entities` = 65,536 entities, dropping
  the least recently updated beyond that.
- **Stands for:** detail | D-56 ("growth rates of energy and entropy"), architecture section 6
- **Why:** with noise of variance sigma^2 per point, the slope over w equally spaced points has
  variance 12 sigma^2 / (w (w^2 - 1) dt^2), a factor w (w^2 - 1) / 6 = 20 smaller than the variance
  2 sigma^2 / dt^2 of the two-point difference at w = 5 (standard least squares, Var(slope) = sigma^2 / sum (t - tbar)^2), while five
  cadence triggers are still only five minutes. A slope computed inside one TAAFT call would depend on
  where a stream is cut into calls and would break the one-call-equals-two-calls equivalence of AS-223.
  The bound keeps memory finite on networks with many transient entities.
- **Evidence:** code run: `test_statphys_thermo.py::test_growth_rates_batch_and_streaming` (exact
  slope of a line to 1e-15; streaming equals batch to 1e-12).

### AS-764: The network state at tau is the traffic of (tau - 300 s, tau]
- **Assumed:** the traffic entropies and the activity multiplex of a trigger tau are built from the
  state updates with event time in (tau - `state_window_seconds`, tau], `state_window_seconds` = 300,
  inclusive at tau.
- **Stands for:** detail | D-56 ("per entity and per window"), D-36 (time, not volume)
- **Why:** a time window cannot be stretched or shrunk by traffic volume, which an attacker controls
  (build-spec section 0, item 7). 300 s is five cadence intervals (60 s, AS-12): successive states share
  four fifths of their traffic, so the divergence between successive states measures change rather than
  resampling, and five minutes stays short against the minutes-to-hours span of attack stages. Inclusive
  at tau matches the engine, which processes updates with time equal to tau before the trigger fires.
- **Evidence:** Lakhina, Crovella and Diot, "Mining anomalies using traffic feature distributions",
  SIGCOMM 2005, compute feature entropies over fixed time bins of network-wide traffic (the bin length
  they used: citation to verify). Code run: `test_statphys_engine.py::test_traffic_reading_sees_only_updates_up_to_the_trigger`.

### AS-765: Which traffic distributions are read, and which participations count
- **Assumed:** per entity: destination port, source port, protocol, TCP flag combination (the
  `flow.tcp_flags` code), packet-weighted flag mass (`flow.flag_count.*`) and peers (the other members
  of each update); over the network: the same fields plus the initiator and responder entities. Only
  contributing cells count. Role "any": the initiator, the responder and the service entity of an update
  all count it.
- **Stands for:** detail | D-56 ("ports, protocols, flags, peers"), D-41
- **Why:** Lakhina et al. read source and destination address and port distributions; the problem
  statement names flags and ports as driving features, and protocol completes the flow key. The service
  entity is a member of the update's hyperedge (graph/window.py, AS-104), so it shares the update's
  evidence. Non-contributing cells are not evidence (D-41).
- **Evidence:** Lakhina et al., SIGCOMM 2005. Code run:
  `test_statphys_entropy.py::test_traffic_entropies_match_a_hand_computation`.

### AS-766: Miller-Madow is the default entropy estimator
- **Assumed:** entropies use the Miller-Madow correction H + (m - 1) / (2 N) by default; plug-in and
  Chao-Shen are configurable.
- **Stands for:** detail | D-56 ("with bias correction (Miller-Madow)")
- **Why:** a per-entity window holds tens of updates, where the plug-in bias of about -(m - 1) / (2 N) is
  of the order of the changes being watched. Miller-Madow costs O(1) per streaming read; Chao-Shen needs
  the whole histogram per read and integer counts.
- **Evidence:** Miller, "Note on the bias of information estimates", 1955; Paninski, Neural Computation
  15(6):1191, 2003; Chao and Shen, Environmental and Ecological Statistics 10:429, 2003. Code run:
  `test_statphys_entropy.py::test_chao_shen_closed_form_and_bias_reduction` (uniform law over 40 symbols,
  60 samples, 300 replicates: both corrections less biased than plug-in).

### AS-767: The graph state is the activity multiplex with binary weights and star semantics
- **Assumed:** the graph state is the activity multiplex of the state window: nodes are the members of
  its updates; layer p joins two members of an update on plane p (AS-01) that communicate under the star
  semantics of D-52 (a multicast group member, D-47, is the hub). Edge weights are binary by default. The
  first-contact view (`contact_state`) is the alternative of the batch path.
- **Stands for:** detail | D-56 ("von Neumann entropy of graphs ... over the six CVG-AE planes"), D-47, D-52
- **Why:** star semantics is the contact rule of the whole model (D-52), so the entropy reads the same
  structure TSTCT and TAAFT see. Binary weights make the structural reading invariant to traffic volume;
  volume is read by the traffic entropies. First contacts never expire inside a window, so the contact
  view grows monotonically and is a poor early-warning state.
- **Evidence:** code runs: `test_statphys_graphs.py::test_star_semantics_of_update_pairs`,
  `::test_activity_state_matches_a_hand_construction`.

### AS-768: Density matrices and diffusion times
- **Assumed:** the von Neumann entropy uses rho = L / tr L with the combinatorial Laplacian; the spectral
  entropy uses rho_tau = exp(-tau L) / Z at tau in {0.1, 1, 10} (units of 1 / edge weight); the node set
  includes isolated nodes, which are part of the diffusion ensemble (one zero mode each).
- **Stands for:** detail | D-56
- **Why:** with binary weights the Laplacian spectrum lies in [0, 2 d_max], so tau = 0.1, 1 and 10 read
  the ensemble at fine, intermediate and coarse diffusion scales. A node present in the window but
  isolated on a plane is part of that plane's state; dropping it would change Z and make layers of one
  state incomparable.
- **Evidence:** Braunstein, Ghosh and Severini, Annals of Combinatorics 10:291, 2006; De Domenico and
  Biamonte, Physical Review X 6:041062, 2016, arXiv:1609.01214. Code run:
  `test_statphys_spectral.py::test_spectral_entropy_limits_identities_and_dense_agreement` (tau -> 0 gives
  log n, tau -> inf gives log c, to 1e-6 and 1e-9).

### AS-769: Exact spectra up to 512 nodes per component, stochastic Lanczos quadrature beyond
- **Assumed:** components of at most 512 nodes are diagonalised exactly. Larger ones use stochastic
  Lanczos quadrature with: Rademacher probes in batches of 16 (16 to 256); 32 Lanczos steps, doubled while
  the quadrature error exceeds half the tolerance (up to 128); stopping at 2.576 standard errors plus the
  depth error below max(1e-6, 1e-3 |value|); exact deflation of the zero modes; 64 deflated low modes for
  the diffusion functions; a degree-2 polynomial control variate fitted on a separate pilot batch; seed 0.
- **Stands for:** detail | D-56 ("exact eigendecomposition for small graphs and stochastic Lanczos
  quadrature with error control for large ones")
- **Why:** dense eigendecomposition costs O(n^3); 512^3 is about 1.3e8 operations, under a second.
  Beyond that the Krylov method costs O(probes x steps x edges). Each variance reduction is unbiased by
  construction. Common random numbers (one seed) make successive readings share their probes, so
  estimation noise does not appear as white noise in a trajectory.
- **Evidence:** Ubaru, Chen and Saad, SIAM J. Matrix Anal. Appl. 38(4):1075, 2017; Hutchinson 1989; Avron
  and Toledo, J. ACM 58(2):8, 2011; Meyer, Musco, Musco and Woodruff, SOSA 2021, arXiv:2010.09649. Code
  runs:
  - `test_statphys_spectral.py::test_stochastic_lanczos_estimates_are_within_tolerance_of_exact_values`:
    a 400-node graph; every estimate within max(5 standard errors, 2e-3 relative).
  - A benchmark on random graphs of 600 and 2000 nodes: von Neumann errors 6.8e-5 and 2.6e-5 against
    standard errors 5.6e-5 and 3.9e-5.

### AS-770: Divergence between successive states: laplacian kind over the union of nodes
- **Assumed:** the divergence between successive states is the quantum Jensen-Shannon divergence of
  rho = L / tr L, per plane and for the aggregate. It is computed on the union of the two node sets, an
  entity absent from one state being an isolated node of it. The diffusion kind (rho_tau) is available
  where every component of the union graph can be diagonalised exactly.
- **Stands for:** detail | D-56 ("Jensen-Shannon divergence between successive network states")
- **Why:** the laplacian-kind mixture is itself a graph Laplacian (weights w_a / (2 t_a) + w_b / (2 t_b)),
  so the divergence needs three von Neumann entropies at any size. Isolated nodes leave a von Neumann
  entropy unchanged, so an entity that enters the network registers through its new edges. The
  diffusion mixture has no such form: it needs exp(-tau L) explicitly.
- **Evidence:** Lamberti et al., Physical Review A 77:052311, 2008; Virosztek, Advances in Mathematics
  380:107595, 2021, arXiv:1910.10447. Code run: `test_statphys_spectral.py::test_quantum_jensen_shannon_divergence`
  (both kinds equal the dense density-matrix computation to 1e-9).

### AS-771: Multiplex reduction by average linkage on the Jensen-Shannon distance
- **Assumed:** the relative entropy q is computed over the layers that have an edge. The layers are
  clustered by average linkage (configurable: single, complete, Ward) on sqrt(D_QJS). q is reported at
  every level, and the best partition is the level of maximal q (ties go to the finer level).
- **Stands for:** detail | D-56 ("multiplex extension over the six CVG-AE planes")
- **Why:** a layer without an edge has no density matrix. The relative entropy and the reduction by
  hierarchical clustering on the Jensen-Shannon distance are the method of De Domenico et al.; the
  linkage rule is kept configurable rather than claimed from the paper. Average linkage (UPGMA)
  satisfies the reducibility property, so its dendrogram has no inversions (Murtagh, "A survey of
  recent advances in hierarchical clustering algorithms", The Computer Journal 26(4):354, 1983;
  citation to verify).
- **Evidence:** De Domenico, Nicosia, Arenas and Latora, Nature Communications 6:6864, 2015; Lance and
  Williams, The Computer Journal 9(4):373, 1967. Code run:
  `test_statphys_spectral.py::test_multiplex_relative_entropy_and_reduction`.

### AS-772: Sixty-sample rolling windows, causal Gaussian detrending online
- **Assumed:** the indicators use rolling windows of 60 samples, on full windows only. Online
  detrending is a one-sided Gaussian kernel (sigma = 10 samples, truncated at 4 sigma); the two-sided
  kernel is for offline analysis of a saved trajectory. The online samples are the cadence triggers.
- **Stands for:** detail | D-56 ("Gaussian-kernel detrending")
- **Why:**
  - One hour of 60 s triggers is long enough for stable lag-1 and spectral estimates (30 Fourier
    frequencies).
  - A two-sided smoother uses later samples, which an online reading cannot have.
  - Priority triggers are irregular in time, and the spectral and autocorrelation indicators assume a
    regular grid.
- **Evidence:** Dakos et al., PNAS 105(38):14308, 2008; Dakos et al., PLoS ONE 7(7):e41010, 2012. Code
  run: `test_statphys_ews.py::test_detrending_causality_and_linear_reproduction` (residuals before a
  change at index 200 are unchanged; the two-sided kernel reproduces a line to 1e-9).

### AS-773: Definitions of the indicators
- **Assumed:**
  - variance with 1 / (W - 1);
  - lag-1 autocorrelation as the biased sample autocorrelation;
  - Pearson skewness and kurtosis;
  - return rate -ln(rho_1) / dt (undefined for rho_1 <= 0);
  - spectral ratio as mean Hann-tapered power at f <= 0.05 over power at f >= 0.25 (cycles per sample);
  - spectral exponent as the least-squares slope over all Fourier frequencies;
  - DFA-1 with boxes aligned to absolute sample indices, scales 4, 6, 8 and 12, at least four boxes per
    scale.
- **Stands for:** detail | D-56
- **Why:**
  - The AR(1) coefficient of a process with recovery rate lambda is exp(-lambda dt), so -ln(rho_1) / dt
    estimates lambda itself.
  - Absolute alignment keeps every complete box fixed while the window slides, which makes an exact
    streaming form possible; a box's residual sum of squares does not depend on the profile's offset.
- **Evidence:** Scheffer et al., Nature 461:53, 2009; Held and Kleinen, Geophysical Research Letters
  31:L23207, 2004; Kleinen et al., Ocean Dynamics 53:53, 2003; Peng et al., Physical Review E 49:1685,
  1994; Livina and Lenton, Geophysical Research Letters 34:L03712, 2007. Code runs:
  - `test_statphys_ews.py::test_indicators_equal_direct_window_formulas` (each indicator equals an
    independent per-window computation to 1e-9).
  - `::test_spectral_reddening_and_dfa_respond_to_slowing_down` (white-noise DFA exponent 0.5 +- 0.12).

### AS-774: Kendall tau-b trends, trailing online and whole-series offline
- **Assumed:** the trend of an indicator is Kendall's tau-b against time. Online it is computed over the
  trailing 60 valid indicator values; offline over the whole indicator series. Expected directions:
  rising for variance, ar1, kurtosis, the spectral indicators and DFA; falling for the return rate;
  two-sided for skewness.
- **Stands for:** detail | D-56 ("Kendall tau trend statistics")
- **Why:** tau is invariant to monotone transformations of the indicator and robust to outliers.
  Skewness changes sign depending on the side from which a transition is approached.
- **Evidence:** Kendall, Biometrika 30:81, 1938 and 33:239, 1945; Guttal and Jayaprakash, Ecology Letters
  11:450, 2008; Dakos et al. 2012. Code run: `test_statphys_ews.py::test_kendall_tau_b_against_references`
  (equal to the O(n^2) definition and to SciPy's kendalltau to 1e-12 on tied data).

### AS-775: Surrogate significance from phase-randomised and AR(1) surrogates of the residuals
- **Assumed:** 199 surrogates per family, phase-randomised and Yule-Walker AR(1), are generated from the
  residual series. Each passes through the same rolling indicators and trend, without detrending again.
  The p-value is (1 + #{at least as extreme}) / (1 + n) in the expected direction.
- **Stands for:** detail | D-56 ("significance from surrogates (phase-randomised and fitted AR(1)
  surrogates)")
- **Why:** rolling indicators are strongly autocorrelated, so the classical null law of tau does not
  apply. Both surrogate families keep the residuals' autocorrelation and remove any trend. 199 surrogates
  give p-values on a grid of 0.005.
- **Evidence:** Theiler et al., Physica D 58:77, 1992; North, Curtis and Sham, American Journal of Human
  Genetics 71:439, 2002; Dakos et al. 2012. Code runs:
  - `test_statphys_ews.py::test_rising_ar1_coefficient_shows_critical_slowing_down` (p <= 0.02 with 99
    surrogates for ar1 and return rate).
  - `::test_stationary_ar1_gives_approximately_uniform_p_values` (40 stationary series: mean p in
    (0.3, 0.7) and Kolmogorov-Smirnov distance to uniform < 0.3, for both families).

### AS-776: Composite scores from the variance and lag-1 autocorrelation
- **Assumed:** the level score is the mean signed robust z of the composite indicators (default:
  variance and ar1), against the benign median and 1.4826 MAD. The trend score is the mean signed Kendall
  tau.
- **Stands for:** detail | D-56 ("a composite early-warning score")
- **Why:** rising variance and lag-1 autocorrelation are the two generic signatures of critical slowing
  down. The median and MAD are not moved by the occasional benign burst.
- **Evidence:** Scheffer et al., Nature 2009; Rousseeuw and Croux, JASA 88:1273, 1993. Code run:
  `test_statphys_ews.py::test_alarm_calibration_meets_the_target_false_alarm_rate`.

### AS-777: Alarm thresholds by split conformal calibration on benign data, never defaulted
- **Assumed:**
  - The alarm fires when the level score exceeds a split-conformal threshold calibrated on benign
    trajectories, at a target rate of 0.01 per trigger over all monitored series.
  - The rate is split equally over the series (Bonferroni), so the alarm of any series keeps the target.
  - The baselines and the threshold come from disjoint benign data (by series, or by time within one
    series, fit fraction 0.5).
  - Without a calibration file no alarm is raised.
  - Monitored series by default: energy.total, energy.marginal, entities.free_energy, entities.entropy,
    routes.free_energy, routes.entropy, graph.vn.aggregate, traffic.dst_port.
- **Stands for:** detail | D-56 ("an alarm threshold calibrated to a target false-alarm rate on benign
  data")
- **Why:** a threshold that was never calibrated has no false-alarm meaning, so it is not defaulted.
  Disjoint data keep the conformal guarantee of a fixed score function. Benign scores are serially
  dependent, so the guarantee holds marginally (for a benign trigger chosen independently of the
  calibration data) rather than for runs of triggers.
- **Evidence:** Vovk, Gammerman and Shafer, Algorithmic Learning in a Random World, Springer 2005;
  Angelopoulos and Bates, arXiv:2107.07511, 2021; Dunn, JASA 56:52, 1961. Code run:
  `test_statphys_ews.py::test_alarm_calibration_meets_the_target_false_alarm_rate` (target 0.05: the
  rate on fresh benign series lies in [0.01, 0.12]).

### AS-778: Streaming forms, their exactness and their work per state update
- **Assumed:**
  - Moment indicators use running power sums of shifted residuals: O(1) per sample.
  - Spectral indicators use a sliding DFT of floor(W / 2) coefficients with the Hann taper applied in
    frequency.
  - DFA uses per-box statistics computed once per completed box.
  - Trends use a sorted window: O(log H) comparisons and O(H) moves.
  - Traffic histograms change by O(1) per event, with Neumaier-compensated sums of n log n.
  - Running sums are recomputed exactly every 60 samples; histogram sums every 4096 changes.
  - Each streaming form equals its batch form.
- **Stands for:** detail | D-56 ("streaming implementations with O(1) work per state update that equal
  the batch versions")
- **Why:** none of these costs grows with the length of the stream. Periodic exact recomputation bounds
  the accumulated rounding by the work since the last one.
- **Evidence:** Neumaier, ZAMM 54:39, 1974; Higham, Accuracy and Stability of Numerical Algorithms, 2nd
  ed., SIAM 2002, section 4.3. Code runs:
  - `test_statphys_ews.py::test_streaming_equals_batch_for_every_indicator_and_trend` (1e-9).
  - `test_statphys_entropy.py::test_streaming_tracker_equals_batch` (1e-12).
  - `::test_sliding_histograms_stay_exact_over_a_long_run` (20,000 events, 1e-12).
  - `test_statphys_graphs.py::test_sliding_multiplex_equals_activity_state_at_every_trigger`.

### AS-779: Energies of imagined steps and routes, and the route ensemble's base measure
- **Assumed:**
  - The energy of an imagined step (n, k) is E(null, y_hat_{n,k}). Its hypothesis comes from the
    Forecaster's step heads re-applied to the stored imagined state.
  - The energy of a route is the sum over its K steps.
  - Each of the N drawn routes is one microstate: a distinct route has multiplicity N w_n.
  - The susceptibility observable is the route's infiltration probability at the horizon.
- **Stands for:** detail | D-56 ("a Gibbs view over imagined states or routes"), AS-17, AS-251, AS-252
- **Why:**
  - `ForecastOut` keeps the imagined states but not their hypotheses. The step heads applied to the
    states are the computation imagination ran, and E(null, y) is the Forecaster's own exposure reading.
  - The sum is the action of the path.
  - Counting draws makes S = log Z + U / T the entropy over the drawn futures (0 <= S <= log N).
    Weighting by the policy measure instead would give a relative entropy -KL(p || w).
- **Evidence:** code run: `test_statphys_engine.py::test_every_trigger_carries_a_reading` (route
  ensembles of 1 to 6 members with 0 <= S <= log 6 on the sample capture).
