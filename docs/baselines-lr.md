# Logistic-regression baseline family: protocol and usage

The problem statement requires benchmark results (F1, precision, recall, false-positive rate) against a
logistic regression trained on the same features. The evaluation chapter extends the comparison to
forecasts, lead times, stages and next-state forecasts, and scores forecasts against persistence and
climatology. `nagahana.baselines.lr` produces every one of these outputs through the prediction contract
of `evaluation/predictions.py`. Assumptions: `docs/assumptions/lr-baseline.md` (AS-500 ... AS-529).

## Members and their outputs

| Member | Unit | Record | Thesis use |
|---|---|---|---|
| Detection LR (`detector.py`) | state update | `DetectionPredictions` (calibrated probability, own threshold; conformal threshold in `component`) | detection table, transfer, leave-one-network-out, site calibration, robustness rows |
| Discrete-time hazard LR (`hazard.py`) | usable trigger | `ForecastPredictions` (P_inf(k), hazards), `TimeToEventPredictions` (survival curve, risk) | forecast quality, lead time, survival columns |
| Multinomial stage LR (`stage.py`) | (trigger, step) or update | `StagePredictions` over the 15 stage classes | stage table (top-1, top-3, macro-F1) |
| Ridge next-state forecaster (`ridge.py`) | trigger | `StateForecastPredictions`, horizons 1 ... K | next-state table (MAE, RMSE, skill vs persistence) |
| Persistence, climatology (`references.py`) | as the forecast and next-state members | `ForecastPredictions`, `TimeToEventPredictions`, `StateForecastPredictions` | skill scores |

The evaluation code computes the reported metrics from these records; this package does not compute or
print any reported number. One record holds every split (meta["split"]): the evaluation fits its
conformal thresholds on the validation units and its climatology on the training triggers of the record it
scores. The model name is "logistic_regression" and the references are "persistence" and "climatology",
the names the evaluation configuration uses. Forecast records carry `infiltrated_now` and the forecaster's
own `alert_threshold` (max-F1 on validation); state forecasts carry `current` and a Gaussian predictive
`variance` from held-out residuals.

## The same features

Rows come from NagaHana's own windows: each source is walked in stream order (`data.stream.plan_stream`),
every window is built by `data.windows.build_window` with its trigger range and label limit, and the
forecast targets are computed by `forecaster.losses.survival_targets` on the window labels. The design
columns are the canonical fields with the FieldEncoder's numeric transform, categorical indicators over a
training vocabulary, bits, and status indicators; absent values get a neutral fill and their own
indicator, never a zero (D-41). Window-level units (triggers) read aggregates and network states of the
cadence windows of their own segment at or before their time. Every column carries a provenance entry
(field, status condition, operation, lag).

## Protocol

1. Records (segments) carry the split roles of the data pipeline's manifest: train, val, test, zero_shot
   (known and novel marked), excluded (purged).
2. The update encoder (vocabularies) and every standardiser are fitted on training rows only; each
   cross-validation fold refits its standardiser on its own training part.
3. Hyperparameters (l2 strength lambda, class-weight strength p; for the ridge forecaster lambda and the lag
   count) are chosen by blocked forward-chaining cross-validation inside the training split, per network,
   with purging by label horizon and an embargo (`lr.yaml`), or on the validation split
   (`lr-validation-selection.yaml`).
4. The final model is refitted on all training units.
5. The validation split calibrates the probabilities (Platt by default; hazard logits for the forecaster)
   and fixes the detector's own operating threshold (max-F1) and its conformal threshold at the configured
   false-positive rate.
6. Test and zero-shot splits are predicted once with the fixed model.

## Usage

```bash
# configuration files (generated from the dataclasses; never edited by hand)
python -m nagahana lr write-config --out conf/baselines/lr

# corpus: sources -> stream segments -> manifest (data.sampling.assign_splits) -> LR corpus
python -m nagahana lr extract --sources sources.yaml --preset L --purge 1 --out corpora/cic18

# fit, inspect the selection, predict
python -m nagahana lr fit --corpus corpora/cic18 --config conf/baselines/lr/lr.yaml --out models/lr-cic18.npz
python -m nagahana lr cross-validate --corpus corpora/cic18 --config conf/baselines/lr/lr.yaml --task detection --out cv-detection.json
python -m nagahana lr predict --model models/lr-cic18.npz --corpus corpora/cic18 --protocol P1 \
    --out outputs/lr-cic18.npz --references outputs/refs        # one record: train, val, test, zero_shot

# site calibration: refit the calibrator and thresholds on a site's labelled calibration records
python -m nagahana lr calibrate --model models/lr-cic18.npz --corpus corpora/site --roles val --out models/lr-site.npz
```

From Python (the entry point of the evaluation protocols):

```python
from nagahana.baselines.lr import build_corpus, load_config, run_protocol

corpus = build_corpus(sources, manifest, cfg)            # PreparedSources, SplitManifest, NagaHanaConfig
run = run_protocol(corpus, load_config("conf/baselines/lr/lr.yaml"), protocol="P1")
detection = run.outputs.detection                         # DetectionPredictions of every split (meta["split"])
skill_refs = run.references                               # {"persistence": ..., "climatology": ...}
run.family.save("models/lr.npz")                          # versioned, checksummed bundle
```

The rows can also be extracted from the batches a NagaHana evaluation loop already holds:
`CorpusBuilder(sources_by_id, cfg, roles=..., label_limits=...).add_batch(window, labels, contexts)`.

## Solvers and scale

Full-batch L-BFGS (strong Wolfe, float64) is the default. Every exact solve is preconditioned by Boehning's
fixed bound of the Hessian (a change of variables that leaves the optimum unchanged), so the iteration
count does not grow with the collinearity of the window design. For corpora that do not fit in memory, the
corpus is stored as per-source shards (memory-mapped value and status matrices), the per-update design is
produced chunk by chunk, the standardiser is fitted exactly in passes (radix selection of order
statistics), and the solver is `streamed_lbfgs` (exact) or `minibatch` (averaged SGD with L-BFGS
polishing). An optional scikit-learn cross-check (`detector.cross_check`, `[baselines]` extra) refits the
detector's objective with `LogisticRegression` and asserts agreement of coefficients and probabilities.
