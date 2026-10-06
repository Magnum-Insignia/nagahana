# ADR-0008: Weight-tied looping in TSTCT and TAAFT, with run-time budgets

- **Status:** Accepted (owner, 2026-09-30)
- **Decision IDs:** D-43, D-44, D-46; related: D-30, D-35, D-15, P-18
- **Sources:** owner, 2026-09-30, chat; [Q-21], [Q-31], [A-14]; Dehghani et al., "Universal
  Transformers", ICLR 2019 (arXiv:1807.03819); Lan et al., "ALBERT", ICLR 2020 (arXiv:1909.11942);
  Geiping et al., "Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach",
  2025 (arXiv:2502.05171)

## Context
The owner wants the model to "look deeper and longer before flagging" [Q-21], and inference-time
scaling is already a knob of TAAFT's energy refinement (the descent steps). Depth can also be spent
by applying the same layers again. Prior work shows this is workable: Universal Transformers apply
one shared block recurrently over depth; ALBERT uses "cross-layer parameter sharing", which "prevents
the parameter from growing with the depth of the network"; Geiping et al. iterate a recurrent block,
"unrolling to arbitrary depth at test-time".

Belief lives only in Imagination and the Environment holds observed facts (D-35, [Q-31]). A loop
that fed TAAFT's output back into TSTCT would write beliefs into the Environment.

## Decision
1. **Weight-tied looping (D-43).** TSTCT and TAAFT each loop on themselves: the same block stack is
   applied for R passes with shared weights. More passes give more thinking at no extra parameters.
   There is no loop from TAAFT back into TSTCT, because beliefs must never overwrite observed facts in
   the Environment.
2. **Loop budget (D-44).** R is a budget set at run time, like K, N and the descent steps, and it is
   recorded with each forecast.
3. **Meaning of N (D-46).** N is the maximum number of routes explored in the Forecaster's
   imagination, since futures diverge. The model may return fewer distinct routes than N.

## Consequences
- Run-time thinking now has three separate knobs: R passes (TSTCT and TAAFT), S descent steps on
  E_total (ADR-0007), and N routes over K steps. Compute grows with each; all are reported.
- `conf/model/tstct/tstct.yaml` and `conf/model/taaft/taaft.yaml` each have `loop_passes_R: ???`.
  One R per model is the engineer's reading of "the number of passes R"; please confirm whether one
  shared R is meant (D-44 note).
- `roles/contracts.py`: `ComputeRecord` records K, N and the descent steps but not yet R; a field for
  R (per looped model) must be added when that file is next edited. `ForecastBundle.top_paths` may
  hold fewer than N paths; nothing there assumes exactly N.
- Training: a looped model trained at one fixed R is not known to improve when run with more passes.
  Geiping et al. (§3.3) "randomly sample iteration counts during training, assigning a random number
  of iterations r to every input sequence". How R is chosen in training is not decided (D-43 note).
- Masks: every TSTCT pass must use the same causal and temporal masks, so no pass can see the future.
  The no-future-leak test must also run with R > 1.
- The Environment: which pass's keys and values TSTCT writes into its cache is not decided (D-43
  note). If R can change between runs, cached Environment entries depend on R as well as on the
  model; this matters for D-15 and proposal P-18 (caches keyed to the model that produced them).
- The one-way flow CVG-AE → TSTCT → TAAFT stays as drawn; the only loops are inside TSTCT and inside
  TAAFT.
