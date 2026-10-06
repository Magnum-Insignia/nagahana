# Logistic-regression baseline family: assumptions (AS-500 ... AS-529)

Each entry states what the LR family assumes, which decision or open detail it stands for, why, and the
evidence. Code that relies on an entry cites it by ID in its docstring (`src/nagahana/baselines/lr/`).
"(citation to verify)" marks a reference written from memory that has not been checked against the
source. These IDs are not yet in `governance/assumptions.py`; the code cites them in docstrings only.

### AS-500: "The same features" means NagaHana's canonical window tables
- **Assumed:** the baseline reads exactly the value/status matrices of NagaHana's windows in the canonical
  column layout (`data/windows.py` `CANONICAL_COLUMNS`: flow-level, packet-level and observable-region
  fields), the same windows built in stream order with the same trigger ranges and label limits
  (`data/stream.py`, `data/windows.build_window`), and the same labels and forecast targets; it does not
  read NagaHana's learned embeddings or graph structure, which a linear model cannot use.
- **Stands for:** detail of the problem statement's "logistic regression baseline trained on the same
  features" | D-38 | D-51
- **Why:** the comparison isolates what learned dynamics add over a linear model of the same observations;
  sharing windows, triggers and targets makes every unit and every label identical between the two models.
- **Evidence:** evaluation chapter (`latexdocs/full-docs/content/evaluation.tex`, Baselines): "trained on
  exactly the same features ... with the same normalisation and the same splits".

### AS-501: Per-update encoding of the canonical columns
- **Assumed:** a numeric column (continuous, count, histogram bin) becomes clip(sign(x) log(1 + |x|), +-50);
  a bitmask column becomes its bits 0 ... 15; a categorical column becomes one indicator per vocabulary code
  plus "other"; every column gets indicators for the statuses STALE, LOW_RELIABILITY, NOT_SUPPLIED and
  NOT_OBSERVABLE, with OBSERVED as the reference level.
- **Stands for:** detail | AS-31 | AS-33 | AS-100
- **Why:** the numeric transform and clip are the FieldEncoder's (so a value reaches both models alike);
  ports and codes are categories, never magnitudes (`datamodel/fields.py`); bits are what the FieldEncoder
  embeds; dummy coding with a reference level avoids an exact collinearity with the intercept.
- **Evidence:** `models/inputs/encoder.py` (signed log1p, clip `max_log_magnitude` = 50, per-bit
  embeddings); Hastie, Tibshirani and Friedman, The Elements of Statistical Learning, 2nd ed., Springer
  2009, Section 3.2 (dummy coding).

### AS-502: Absent values: the missing-indicator method with a neutral fill
- **Assumed:** a value whose status does not contribute is NaN in the raw design and 0 after
  standardisation (the training location of the observed values), while its status indicator carries the
  absence; the raw value zero is never substituted for an absent value.
- **Stands for:** D-41
- **Why:** "absence is not supplied, never zero" (D-41) forbids encoding absence as a measured zero; the
  evaluation chapter fixes "the standard practice of an indicator column with a neutral fill" for the
  baseline. A neutral fill contributes w * 0 to the logit, so the indicator alone models absence.
- **Evidence:** evaluation chapter, Baselines ("an indicator column with a neutral fill, and this
  treatment is reported"); Groenwold et al., "Missing covariate data in clinical research: when and when not
  to use the missing-indicator method for analysis", CMAJ 184(11), 2012 (citation to verify); Josse et al.,
  "On the consistency of supervised learning with missing values", arXiv:1902.06931 (citation to verify).

### AS-503: Categorical vocabularies from the training updates
- **Assumed:** the vocabulary of a categorical column is the codes seen at least 5 times among the
  contributing cells of the training updates, most frequent first (ties by code), capped at 32 for the
  per-update design and at the first 16 for window aggregates; any other code is "other".
- **Stands for:** detail
- **Why:** a training-only vocabulary keeps the design free of test information; the caps bound the width of
  the design while keeping the frequent services (ports 53, 80, 443, 445 and the like) as their own columns.
- **Evidence:** code run: tests/test_lr_features.py::test_vocabulary_is_training_only_and_deterministic.

### AS-504: Standardisation: robust for numeric columns, z-score for indicators, training rows only
- **Assumed:** numeric columns are centred on the median and scaled by IQR / 1.349 (the "robust location and
  scale" of the methodology chapter), indicator and bit columns by mean and standard deviation; a robust
  scale of zero falls back to the standard deviation; statistics come from the training rows of each fit
  only (each cross-validation fold refits them on its own training part) and are stored with the model.
- **Stands for:** detail of the methodology chapter's normalisation | D-23
- **Why:** "Normalisation statistics (robust location and scale, and logarithmic transforms for
  heavy-tailed counts) are computed on the training split only" (methodology chapter, data preparation); ridge
  penalties need comparable column scales; refitting inside folds keeps validation folds out of the
  preprocessing.
- **Evidence:** methodology chapter, data preparation; Hastie, Tibshirani and Friedman 2009, Sections 3.4.1 and
  7.10.2; code run: tests/test_lr_standardise.py::test_standardiser_never_sees_validation_or_test_rows.

### AS-505: Columns constant on the training rows are dropped
- **Assumed:** a design column whose valid training values are all equal (or that has none) is removed
  before fitting (`drop_constant`), and the removal is recorded with the model.
- **Stands for:** detail
- **Why:** after standardisation such a column is 0 on every training row, so its gradient is 0 and its l2
  coefficient would be exactly 0; dropping it changes no prediction and shortens the design.
- **Evidence:** code run: tests/test_lr_features.py::test_constant_columns_are_dropped_without_changing_predictions.

### AS-506: Window context stays inside the unit's own record
- **Assumed:** a trigger reads the updates of its own record (segment) at or before its time, in the cadence
  windows g, g - 1, ..., g - L; a window before the record is "unavailable" and a window of the record
  without updates is "empty"; the first window of a record is read with its coverage (the share of the
  window inside the record).
- **Stands for:** D-51 | AS-333 | AS-334
- **Why:** NagaHana's carried Environment resets at the start of every segment, so the information set of
  the two models is the same, and no context crosses into another record and therefore into another split.
- **Evidence:** `data/stream.py` (reset at segment starts); code run:
  tests/test_lr_features.py::test_trigger_features_use_no_update_after_the_trigger.

### AS-507: Window states
- **Assumed:** flows = distinct flow keys (one per update for flow-record sources); packets, IP-layer bytes
  and payload bytes = sums of each flow's increments since its previous update; destination hosts =
  distinct responders; destination ports = distinct contributing destination ports; SYN and RST shares =
  shares of updates with a contributing TCP flag field that has the flag; failed share = share of updates
  with flow.unanswered = 1.
- **Stands for:** detail of the evaluation chapter's next-state features
- **Why:** flow-state updates carry running totals (D-51), so summing raw totals would count a flow several
  times; the two byte definitions stay separate as the data model keeps them (AS-303); "failed" follows the
  meaning of an unanswered connection attempt.
- **Evidence:** evaluation chapter, Next-state forecasts (counts of flows, packets and bytes, distinct
  destination hosts and ports, shares of SYN, RST and failed connections); `ingest/pcap.py` (running
  state per update); code run: tests/test_lr_features.py::test_flow_increments_and_window_states.

### AS-508: Empty windows: zero counts, undefined shares
- **Assumed:** in an observed window without updates, the counts are 0 for every quantity the source
  supplies at all, undefined (NaN, with an "undefined" indicator) for quantities it never supplies, and the
  shares are undefined; a sum over a window in which any update lacks the quantity is undefined.
- **Stands for:** D-41
- **Why:** silence in an observed window is a measurement (zero traffic), whereas a ratio over no flow and a
  sum over partly unknown terms are not values (D-41: absence is not zero).
- **Evidence:** code run: tests/test_lr_features.py::test_empty_and_unavailable_windows.

### AS-509: Count states on a log scale
- **Assumed:** the count-type window states enter both as inputs and as next-state targets as
  log(1 + count), then are standardised on the training split; shares are kept on [0, 1].
- **Stands for:** detail of the methodology chapter's normalisation
- **Why:** "logarithmic transforms for heavy-tailed counts" (methodology chapter, data preparation) applies to window
  counts as to field values; on the raw scale a single burst would dominate MAE and RMSE.
- **Evidence:** methodology chapter, data preparation.

### AS-510: Step-stage labels
- **Assumed:** the stage of future step k of a trigger is the furthest stage among the labelled updates of
  its source in (tau + (k - 1) w, tau + k w], counted only when the whole step lies inside the label
  horizon; otherwise, or without a labelled update, it is unknown (-1).
- **Stands for:** detail | AS-25 | AS-334
- **Why:** this is the rule of `forecaster.losses.step_labels`, applied to the whole source stream so that
  steps reaching past a window are labelled, while the label limit keeps the look-ahead inside the split.
- **Evidence:** `models/forecaster/losses.py` (`step_labels`); code run:
  tests/test_lr_corpus.py::test_targets_match_the_forecaster_definitions.

### AS-511: Hyperparameters chosen on held-out folds in time; the validation split calibrates
- **Assumed:** lambda and the class-weight strength are chosen by blocked forward-chaining cross-validation
  inside the training split (purging by label horizon, embargo), and the validation split is reserved for
  calibration and the operating threshold; selection on the validation split itself is the configuration
  `lr-validation-selection.yaml`.
- **Stands for:** detail of the evaluation chapter ("strengths are chosen on the validation split")
- **Why:** validation folds in time are held-out validation data that never follow their training data;
  keeping the validation split for calibration means each fitted quantity (coefficients, hyperparameters,
  calibrator and threshold) is fitted on data the previous step did not use.
- **Evidence:** Lopez de Prado, Advances in Financial Machine Learning, Wiley 2018, Chapter 7; Tashman,
  "Out-of-sample tests of forecasting accuracy", Int. J. Forecasting 16(4), 2000; code run:
  tests/test_lr_temporal_cv.py::test_folds_never_train_after_validation_and_respect_embargo.

### AS-512: Grids and fold layout
- **Assumed:** lambda on a log grid 10^-7 ... 10^0 (15 values) for the logistic models and 10^-4 ... 10^4
  (17 values) for the ridge forecaster; class-weight power in {0, 0.5, 1}; ridge lag counts {0, 1, 2, 3};
  5 time blocks per network, the first 2 for training only, expanding window, an embargo of one cadence
  window.
- **Stands for:** detail
- **Why:** the objective is a mean loss on standardised features, so useful lambdas lie well below 1; the
  power grid spans no weighting to balanced weighting; three validation blocks per network give a mean and
  a standard error for the one-standard-error rule.
- **Evidence:** Hastie, Tibshirani and Friedman 2009, Section 7.10 (grid and one-standard-error rule).

### AS-513: Selection criteria
- **Assumed:** average precision for the detector, the censored survival negative log-likelihood for the
  hazard model, the class-balanced log loss for the stage model, the mean squared error (standardised
  units) for the ridge forecaster.
- **Stands for:** detail
- **Why:** each criterion is comparable across class-weight strengths: average precision ranks and reads
  against the base rate of rare attacks; the survival likelihood is the proper score of the hazard model;
  the balanced log loss weighs rare stages equally, as macro-F1 does.
- **Evidence:** Davis and Goadrich, "The relationship between precision-recall and ROC curves", ICML 2006;
  Saito and Rehmsmeier, PLoS ONE 10(3), 2015; Gneiting and Raftery, JASA 102(477), 2007.

### AS-514: The detector's own operating threshold is max-F1 on validation
- **Assumed:** the detector's own threshold maximises F1 on the calibrated validation probabilities (the
  midpoint between the chosen and the next lower distinct score); the split-conformal threshold at the
  configured false-positive rate (0.10 %) is kept beside it.
- **Stands for:** detail of the evaluation chapter ("each method's own operating threshold")
- **Why:** F1 is the headline metric of the required comparison, so its own threshold tunes the baseline for
  it; the conformal threshold gives the common false-positive-rate comparison of the timeliness table.
- **Evidence:** results chapter (own threshold and fixed-FPR lead times); Angelopoulos and Bates,
  arXiv:2107.07511.

### AS-515: Calibration default: Platt scaling of the logit
- **Assumed:** the detector's probabilities are calibrated on the validation updates by Platt scaling with
  Platt's smoothed targets; isotonic and temperature scaling are configurations.
- **Stands for:** detail
- **Why:** two parameters are stable with few validation positives and undo the intercept shift that class
  weighting introduces, while keeping the ranking.
- **Evidence:** Platt 1999 (Advances in Large Margin Classifiers, MIT Press); Lin, Lin and Weng, Machine
  Learning 68(3), 2007; code run: tests/test_lr_calibration.py::test_calibration_lowers_ece.

### AS-516: Weighted objective normalised by the total weight
- **Assumed:** F = (1 / S) sum_i c_i l_i + (lambda / 2) ||w||^2 with S = sum_i c_i and c_y = (n / (2 n_y)) ** p.
- **Stands for:** detail
- **Why:** normalising by S makes lambda mean the same penalty for every class-weight strength, so the grid
  is shared; p = 1 is the balanced weighting n / (2 n_c).
- **Evidence:** King and Zeng, "Logistic regression in rare events data", Political Analysis 9(2), 2001;
  code run: tests/test_lr_logistic.py::test_sklearn_agreement.

### AS-517: Hazard calibration on the hazard logit
- **Assumed:** the hazard model is calibrated on the validation triggers by eta' = a eta + b on the hazard
  logits, fitted by the censored likelihood.
- **Stands for:** detail
- **Why:** any per-step transform of the hazards keeps P_inf monotone; the censored likelihood is the proper
  score of the model and uses censored triggers correctly.
- **Evidence:** Platt 1999; Tutz and Schmid, Modeling Discrete Time-to-Event Data, Springer 2016.

### AS-518: The stage model predicts the stage of each future step by default
- **Assumed:** the stage model's default unit is (trigger, step k), the per-step stage of NagaHana's
  Forecaster; the per-update stage is a configuration.
- **Stands for:** detail of the evaluation chapter ("per-step stage prediction")
- **Why:** like-for-like units for the stage table of the results chapter.
- **Evidence:** evaluation chapter, Metric catalogue (ATT&CK stage: per-step stage prediction).

### AS-519: Unseen stages have probability zero
- **Assumed:** a stage absent from a fit's training labels (the final fit's, or a cross-validation fold's)
  is left out of that fit's softmax and receives probability 0.
- **Stands for:** detail
- **Why:** with no training label of a class, the likelihood increases without bound as that class's
  intercept goes to minus infinity, so the maximum-likelihood probability is 0 and no finite optimum exists
  with the class kept; any smoothing constant would be invented.
- **Evidence:** code run: tests/test_lr_stage.py::test_unseen_stages_get_zero_probability.

### AS-520: Timeless sources give detection units only
- **Assumed:** a source without times (all event times 0) contributes state updates to detection and to the
  per-update stage model, and no trigger of it is a forecast, step-stage or next-state unit.
- **Stands for:** AS-307
- **Why:** a source without times supplies no temporal information; its single trigger at time 0 would
  produce fabricated censored outcomes.
- **Evidence:** `data/windows.py` (timeless sources, AS-307).

### AS-521: Persistence of the infiltration state
- **Assumed:** `infiltrated_now` of a trigger is true when an internal entity with an update (as initiator or
  responder) in the cadence window ending at the trigger has its first infiltration at or before the
  trigger; the persistence forecast of P(T <= k) is 1 for every k where it is true, else 0.
- **Stands for:** detail of the evaluation chapter's persistence reference and of the prediction contract's
  `ForecastPredictions.infiltrated_now` ("the infiltration state holds in the window ending at the trigger")
- **Why:** "the current state continues" read with the infiltration state of AS-18; the entities are those
  of the window ending at the trigger, so the definition uses no information after the trigger, and it is the
  one the evaluation applies to every forecast record.
- **Evidence:** results chapter (persistence issues probabilities of zero and one; its calibration error
  equals its Brier score); evaluation/forecasting.py (`persistence`); code run: brute-force check of the
  definition on the synthetic corpus, tests/test_lr_corpus.py::test_infiltrated_now_matches_its_definition.

### AS-522: Undefined current states in the persistence forecast
- **Assumed:** where the current window's state is undefined (a share over an empty window), persistence
  issues the climatology value 0 (standardised units), and the number of such entries is reported.
- **Stands for:** detail
- **Why:** the prediction record needs a finite value; the climatology value is the reference that makes
  no claim, and reporting the count keeps the substitution visible.
- **Evidence:** `references.py` (`state_forecast.climatology_filled` component).

### AS-523: Forecast targets are NagaHana's
- **Assumed:** event steps, censoring and usability come from `forecaster.losses.survival_targets` on each
  window's labels; observed_steps = floor((end - tau) / w) clipped to [0, K] and at least the event step;
  the time-to-event is t* - tau for an event, else min(K w, end - tau).
- **Stands for:** AS-18
- **Why:** identical targets for both models; an event observed inside a partly observed step counts as
  observed up to that step.
- **Evidence:** `models/forecaster/losses.py` (`survival_targets`); code run:
  tests/test_lr_corpus.py::test_targets_match_the_forecaster_definitions.

### AS-524: Hazard steps: shared coefficients, shrunk deviations, zero hazard without events
- **Assumed:** by default every step shares the coefficients w and has its own unpenalised intercept; with
  `coefficients: per_horizon`, step j has w + d_j with d_j penalised by the same lambda. A step with no
  training event is held at hazard 0 exactly and listed with the model (`zero_steps`).
- **Stands for:** detail
- **Why:** proportional odds over steps is the standard discrete-time hazard model; per-step deviations
  shrunk towards it stay estimable for steps with few events; without an event in step j the likelihood
  increases without bound as alpha_j goes to minus infinity, so hazard 0 is the maximum-likelihood limit.
- **Evidence:** Singer and Willett, Journal of Educational Statistics 18(2), 1993; Evgeniou and Pontil,
  "Regularized multi-task learning", KDD 2004; code run:
  tests/test_lr_hazard.py::test_steps_without_events_are_held_at_zero_hazard.

### AS-525: Static comparators by default
- **Assumed:** the detector (context_lags 0) and the infiltration forecaster (lags 0) read only the unit's
  own update or window; the ridge forecaster reads lags, chosen from {0, 1, 2, 3}.
- **Stands for:** detail of the evaluation chapter ("a static forecaster"; "ridge regression on the same
  features and their lags")
- **Why:** the comparison asks what temporal dynamics add over a static model of the same observations;
  lagged variants remain available by configuration.
- **Evidence:** evaluation chapter, Baselines.

### AS-526: Time-to-event risk score
- **Assumed:** the risk score of a trigger is minus the restricted mean time to infiltration within the
  horizon, -w sum_{k=0}^{K-1} S(k).
- **Stands for:** detail
- **Why:** it uses the whole survival curve, and a sooner expected event gives a higher risk, as the
  concordance index requires.
- **Evidence:** Royston and Parmar, BMC Medical Research Methodology 13:152, 2013.

### AS-527: Next-state units and observed targets
- **Assumed:** a trigger is a next-state unit when its own window lies inside its source's observed span;
  a target window is observed when it ends inside the trigger's label horizon and the source's span.
- **Stands for:** detail
- **Why:** persistence needs the current window; a window past the end of a capture was not observed and
  must not be scored as empty.
- **Evidence:** code run: tests/test_lr_ridge.py::test_ridge_outputs_validate_and_mask_unobserved_windows.

### AS-528: Streamed solvers
- **Assumed:** corpora that do not fit in memory are fitted by L-BFGS with every evaluation accumulated over
  chunks (exact), or, for the per-update detector, by averaged mini-batch SGD with gain
  eta_0 / (1 + eta_0 lambda t) ** 0.75, eta_0 scaled by the curvature bound of the first chunk, followed by
  streamed L-BFGS polishing; the trigger-level models (one row per cadence window) use the exact solvers.
- **Stands for:** detail
- **Why:** both reach the unique optimum of the strictly convex objective; averaging gives the optimal
  asymptotic rate of stochastic approximation and the polishing removes its residual error.
- **Evidence:** Polyak and Juditsky, SIAM J. Control Optim. 30(4), 1992; Bottou, LNCS 7700, 2012; code run:
  tests/test_lr_logistic.py::test_streamed_and_minibatch_solvers_reach_the_optimum.

### AS-529: Preconditioned L-BFGS and its stopping rules
- **Assumed:** every exact solve runs L-BFGS (history 20) in the variables of Boehning's fixed bound of the
  Hessian, computed on a seeded sample of at most 100,000 rows; it stops when max |grad F| <= 1e-6, or when
  the relative objective change between checks (every 10 iterations) is <= 1e-10, or after 1000 iterations.
- **Stands for:** detail
- **Why:** a standardised window design is strongly collinear, and plain L-BFGS then needs thousands of
  iterations at small lambda and can stop on a slowly falling objective far from the optimum; the change of
  variables leaves the optimum unchanged and makes the iteration count insensitive to collinearity. F is a
  mean loss of order 0.1 to 1, so 1e-6 on its gradient is a hundred times stricter than scikit-learn's
  default (1e-4 on the same objective); the diagnostics record which rule stopped each fit, and the
  scikit-learn cross-check polishes to machine precision before it compares optima.
- **Evidence:** Boehning and Lindsay, Annals of the Institute of Statistical Mathematics 40(4), 1988;
  Boehning, same journal 44(1), 1992; Nocedal and Wright, Numerical Optimization, 2nd ed., Springer 2006,
  Chapters 5 and 7; code run: a collinear logistic problem at lambda = 1e-5 took 2147 plain iterations and 13
  preconditioned ones, to a lower objective; tests/test_lr_logistic.py::test_sklearn_agreement;
  tests/test_lr_hazard.py::test_preconditioning_does_not_move_the_optimum.
