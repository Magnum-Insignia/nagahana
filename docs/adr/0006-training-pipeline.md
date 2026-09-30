# ADR-0006: Six-stage training pipeline and data splits

- **Status:** Accepted (owner, 2026-09-29)
- **Decision IDs:** D-22, D-23; open: D-16, P-23
- **Sources:** [I-01], [A-24]

## Decision
1. Deep data analysis.
2. Preparation with the Generator.
3. Self-supervised pretraining of CVG-AE, Decoder and TSTCT.
4. Self-supervised pretraining of TAAFT with those three frozen.
5. Full training, including the Forecaster and Advisor agents; the Verifier is trained on human feedback.
6. Zero-shot validation with calibration.

Splits: training and validation use real + generated data. Zero-shot uses real data only, with novel
and known attacks evaluated separately.

## Consequences
- `pipeline/stages.py` holds the plan as data.
- `pipeline/splits.py` rejects manifests that break the split rules.
- `pipeline/freezing.py` proves stage-4 freezing with parameter fingerprints.
- Open: whether "novel" includes unseen networks (D-16); whether the Generator may train only on the
  training split (P-23).
