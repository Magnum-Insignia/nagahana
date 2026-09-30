"""The evaluation catalogue: every method and metric, by component (docs/architecture.md §7).

The owner asked that the new design "should give more evaluations methods & metrics, and more
outputs" (2026-09-29). This catalogue lists them as data, so that:
- the CLI can print it (`python -m nagahana metrics`);
- the thesis evaluation chapter and this code stay in step;
- each entry says whether it is implemented here or still a template, and where.

`status`: "implemented" = real code with tests in this repository; "template" = defined, not yet coded.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MetricSpec:
    """One evaluation entry."""

    component: str
    name: str
    method: str
    definition: str
    status: str
    where: str = ""


CATALOGUE: tuple[MetricSpec, ...] = (
    # Problem statement + operational error
    MetricSpec("overall", "precision / recall / F1 / FPR", "vs logistic regression on the same features",
               "confusion-matrix ratios; NaN when undefined", "implemented", "evaluation/metrics.py, evaluation/baseline.py"),
    MetricSpec("overall", "FNR, detection error, base rate", "operational error accounting",
               "FN/(FN+TP); (FP+FN)/N; positives/N", "implemented", "evaluation/metrics.py"),
    MetricSpec("overall", "AUROC / AUPRC", "threshold-free ranking", "areas under ROC and precision-recall curves", "template"),
    MetricSpec("overall", "zero-shot known vs novel", "real-only zero-shot split, reported separately",
               "all metrics per novelty group", "implemented (split rules)", "pipeline/splits.py"),
    MetricSpec("overall", "cross-dataset / leave-one-network-out", "generalisation protocol (D-16)",
               "train on some datasets or networks, test on held-out ones", "template"),
    # Forecast
    MetricSpec("forecast", "Brier / log score / CRPS", "proper scoring", "see objectives/scoring.py", "implemented", "objectives/scoring.py"),
    MetricSpec("forecast", "ECE + reliability diagram", "calibration", "Σ_b n_b/n·|acc_b − conf_b|", "implemented", "evaluation/calibration.py"),
    MetricSpec("forecast", "skill score", "vs persistence and climatology", "1 − S_model/S_reference", "implemented", "evaluation/forecasting.py"),
    MetricSpec("forecast", "lead time", "time before stage completion at alert threshold (fixed FPR)",
               "t_c − t_alert", "implemented", "evaluation/forecasting.py"),
    MetricSpec("forecast", "C-index / time-dependent AUC / integrated Brier", "survival analysis with censoring",
               "Harrell's C; others template", "implemented (C-index)", "evaluation/forecasting.py"),
    MetricSpec("forecast", "stage top-k, macro-F1 per ATT&CK stage", "per-step stage prediction", "standard", "template"),
    MetricSpec("forecast", "path precision/recall@N, edit distance", "imagined vs realised attack paths", "set/sequence match", "template"),
    MetricSpec("forecast", "ranking agreement (Kendall τ)", "ordinal safety (ARCH §3.3)", "rank correlation of threat orderings", "template"),
    # Components
    MetricSpec("CVG-AE", "masked reconstruction, edge AUROC/AP, KL, active units", "observed fields only (P-20)", "", "template"),
    MetricSpec("CVG-AE", "OOD AUROC (energy / likelihood ratio)", "not raw ELBO (P-21)", "", "template"),
    MetricSpec("TSTCT", "latent prediction error; no-future-leak test", "mask audit", "temporal mask forbids t_j > t_i",
               "implemented (mask test)", "models/tstct/masks.py"),
    MetricSpec("Decoder", "decoded-vs-observed error; believed-as-observed count", "provenance audit", "count must be 0", "template"),
    MetricSpec("TAAFT", "belief Brier/log score; trust AUROC", "simulated worlds with known hidden state (P-14)", "", "template"),
    MetricSpec("TAAFT", "energy OOD AUROC; early-warning lead time", "novelty and trend", "", "template"),
    MetricSpec("Advisor", "ΔP_inf, disruption, feasibility, regret vs oracle", "re-imagination and counterfactual replay", "", "template"),
    MetricSpec("Verifier", "ECE/Brier before vs after; drift-detection delay at fixed false-alarm rate",
               "QCD on memory drift; injected poisoning", "", "template"),
    MetricSpec("Generator", "MMD/Wasserstein; physics-violation rate; label preservation; TSTR; no leakage",
               "fidelity, diversity, utility, safety", "", "implemented (no-leakage rule)", "pipeline/splits.py"),
    MetricSpec("physics", "violation rate of model outputs", "hallucination control (D-18)", "Φ_phys > 0 fraction",
               "implemented (term)", "physics/term.py"),
    MetricSpec("operations", "latency per state update; throughput; memory growth", "deployment realism",
               "incl. NetFlow-only vs full telemetry; telemetry loss", "template"),
    MetricSpec("forensics", "stage-onset timing error; patient-zero accuracy; narrative precision/recall",
               "replay on labelled incidents", "", "template"),
)
