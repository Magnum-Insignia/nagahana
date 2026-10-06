# Assumptions of the precision policy (AS-450 … AS-459), 2026-10-02

Implementation of owner decision **D-54** ("fp32 overall — weights stored and served in fp32; outputs
computed in fp64: P_inf and hazards, stage and other posteriors, energies (E_total, per-lens energies
and shares), and calibration (temperature, ECE, conformal thresholds); training keeps fp32 master
weights"). Each entry below is a precision choice that D-54 does **not** dictate word for word. It is an
engineering assumption, not an owner decision: D-54 itself stays as decided in
`governance/decisions.py`. The owner's follow-up to D-54 (2026-10-02, "Caches fp32 too") stores the
Environment and Imagination K/V caches in fp32 (`memory/kvcache.py` `CACHE_DTYPE`; 4 bytes per element
in `lab/compute.py`); training's bf16 matrix compute with fp32 master weights (AS-39) is unchanged.

Findings from code runs are cited as "code run: <test or script>, <number>". Numerical facts are cited
to Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed., SIAM 2002 (unit roundoff u = 2⁻²⁴
≈ 6.0·10⁻⁸ for IEEE single, 2⁻⁵³ ≈ 1.1·10⁻¹⁶ for double, §2.1; recursive summation error
≤ (n − 1)·u·Σ|xᵢ| to first order, §4.2).

---

### AS-450: which tensors are D-54 outputs (float64) and which stay float32 network quantities
- **Assumed:** a tensor is a D-54 *output* (float64) when it is a probability, a posterior, an energy,
  a share or a calibration quantity that is reported or summed into one; it is computed by casting the
  float32 head logits to float64 *before* the link function (σ, softmax, log-softmax). Network
  quantities (states, values, rewards, latents, hypotheses, features fed to learned heads) stay float32.
- **Lists:**
  - float64, Forecaster: per-step hazard; survival and P_inf; band and median; per-route and mixed stage
    posteriors; log π(technique | s) and log π(target | s, technique); the MPPI log-weights; the route
    log-probability log q(r); route weights.
  - float64, TAAFT: compromise (with the suspicion floor), stage, malignity, trust, slot weight, goal and
    type readouts (energies: AS-451).
  - float64, Verifier: logits, tempered probabilities (`apply_temperature` returns float64 whatever its
    input dtype), the ML temperature, conformal thresholds, reliability, ECE, Brier rewards, Monitor
    statistics; in the engine, the trust probability σ(logit) of the Verifier's trust head.
  - float32: imagined states (`step_state`), values, task rewards, back-projected latents and imagined
    hypotheses of the Forecaster; the MPPI action value Q = r̂ − κ·exposure + γV (a float32 estimate
    that enters the float64 log-weights by exact promotion); TAAFT's context, hypotheses y and latent
    readouts (`latent_*`, `next_latent_*`, which feed the Decoder and the latent likelihoods); the
    Verifier's feature vectors φ_f, φ_m and ψ, which are *inputs* of learned float32 heads and are
    narrowed to float32 at that network boundary only (`verifier/heads.py`).
- **Stands for:** detail of D-54 (the decision names the output families, not each tensor).
- **Why:** D-54 keeps compute in fp32; the precision gain is in the link functions and in what is summed
  after them (products of survival factors, softmax over route log-probabilities, mixtures). The policy
  log-probabilities are included because the route weights are a softmax of their sums: leaving them in
  float32 would put float32 rounding into a reported weight. A float32 → float64 cast is exact and its
  backward casts the gradient back to float32, so training changes only in rounding, and the link that
  is trained is the link that is reported. Feeding float64 features into float32 layers would require
  float64 weights, which D-54 excludes, so narrowing at the network boundary appears to be the
  consistent reading of D-54 (the owner may decide otherwise).
- **Evidence:** code run: `tests/test_precision_outputs.py::test_weights_and_compute_fp32_outputs_fp64_end_to_end`
  (every parameter float32; every listed output float64 at preset "tiny"); code run:
  `tests/test_precision_outputs.py::test_p_inf_identical_from_fp64_hazards_and_model_path` (P_inf from the
  stored float64 hazards and weights equals the model's P_inf bit for bit, and a float64 hand computation
  to 1e-15). Magnitude, measured on three tiny-preset seeds: the previous float32 output path differed
  from the float64 P_inf by at most 6.5·10⁻⁸ (≈ 1 float32 ulp near 1; code run: a script calling
  `_run(seed)` of that test file with seeds 0, 1, 2: 6.48e-08, 5.66e-08, 5.63e-08). The gain is resolution
  near 0 and 1 (float32 cannot represent 1 − P below ≈ 6·10⁻⁸; Higham §2.1), not a change of typical values.

### AS-451: TAAFT energies are reduced in float64; the descent on y stays float32
- **Assumed:** each lens computes its elementwise terms (projections, distances, per-token energies) in
  float32 and reduces them to the per-trigger energy with a float64 accumulation; the learned scale
  exp(log s) is applied in float64; `TotalEnergy` sums the lenses in float64; the per-token energy map
  stays float32 per lens and is summed over lenses in float64 (`readouts["token_energy"]`); the lens
  shares project the float32 per-lens gradients in float64; the energy trace is float64. The descent
  variable y and its update y − α∇E stay float32 (∇_y E arrives in y's dtype through the exact cast).
  The stored reference levels (`energy_reference`, a persistent float32 buffer of the model) stay
  float32; `energy_rel/ℓ` = E_ℓ − reference_ℓ is computed in float64.
- **Stands for:** D-54's open choice ("the descent may stay fp32 unless a lens's numerics need more")
  and the meaning of "accumulate the lens sum in float64".
- **Why:** the reductions are the step where float32 error grows with the number of terms (n entities,
  n² pairs for the topology and causal lenses: Higham §4.2), so they are where float64 helps; the
  elementwise terms have O(u) relative error regardless. No lens of this build needs float64 gradients:
  every term is a smooth function of float32 projections, and the step's float32 rounding (relative
  ≈ 6·10⁻⁸ per coordinate) is far below the step's own scale and, in training, below the annealed
  noise σ_i (AS-212). Keeping y in float32 keeps the readout heads and the Forecaster inputs float32,
  as D-54's "compute fp32" requires. Keeping the reference buffer float32 keeps checkpoints unchanged.
- **Evidence:** code run: `tests/test_taaft_energy.py` (descent lowers E_total; shares sum to 1),
  `tests/test_precision_outputs.py::test_weights_and_compute_fp32_outputs_fp64_end_to_end` (shares sum to
  1 within 1e-9 where the step is non-zero; every energy float64).

### AS-452: losses on float64 outputs are computed in float64
- **Assumed:** a loss that is the likelihood of a D-54 output is evaluated in float64 on that output:
  the Forecaster's survival NLL (float64 hazards), behaviour cloning (float64 log π), the per-step stage
  cross-entropy (softmax of float64-cast logits); TAAFT's readout BCE/CE (stage 5), malignity BCE and
  contrastive energy (stage 4); the stage-6 alert BCE. Targets are cast to the output's dtype. Other
  losses (consistency, hypothesis, latent, reward, value, latent likelihoods) stay float32. Weighted
  totals are float64 by type promotion; `loss.backward()` reaches the float32 parameters with float32
  gradients. The guards ε = 10⁻⁶ of the clamps are unchanged (the trainable region is not moved by the
  precision change). Lists of Python floats turned into tensors for calibration are built with
  `dtype=torch.float64` (stage 5 and stage 6; `torch.tensor(list)` defaults to float32).
- **Stands for:** D-54 ("training keeps fp32 master weights") and the brief's "losses may return fp32
  or fp64 — document".
- **Why:** training and reporting then share one link function at one precision; casting a float64
  output down to float32 before its likelihood would train a different (rounded) quantity than the one
  reported. The cost is negligible: these losses act on per-trigger vectors, not on the transformer.
- **Evidence:** code run: `tests/test_precision_outputs.py::test_float64_losses_backpropagate_into_fp32_parameters`
  (float64 totals; finite float32 gradients reach the readout heads, W_y, the step size and every scaled
  lens; the survival NLL reaches the Forecaster's hazard head); `tests/test_forecaster_losses.py`
  (every Forecaster parameter receives gradient through `teacher_forced`).
