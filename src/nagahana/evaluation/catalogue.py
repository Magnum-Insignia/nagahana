"""The evaluation catalogue: every method and metric, by component, with the code that computes it.

The catalogue follows the evaluation table of docs/architecture.md (section 7) and the metric catalogue of
the thesis evaluation chapter, extended with the statistics, protocols and outputs of this package. Each
entry names the component evaluated, the metric, the method by which it is obtained, its definition and
where it is implemented ("module.function" inside nagahana.evaluation unless a package is named). Every
entry is implemented and covered by the tests tests/test_evaluation_*.py; `python -m nagahana evaluate
catalogue` prints it.
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


_I = "implemented"

CATALOGUE: tuple[MetricSpec, ...] = (
    # Detection and forecast, overall (problem statement and operational error).
    MetricSpec("overall", "precision / recall / F1 / FPR", "against logistic regression on the same features",
               "confusion-matrix ratios at the operating threshold; NaN when undefined", _I,
               "metrics.rates; scorer (detection); baselines.lr"),
    MetricSpec("overall", "FNR, detection error, balanced accuracy, MCC, base rate, alert rate",
               "operational error accounting", "FN/(FN+TP); FP+FN and (FP+FN)/N; (TPR+TNR)/2; Matthews; positives/N",
               _I, "metrics.rates"),
    MetricSpec("overall", "AUROC with DeLong variance", "threshold-free ranking; paired DeLong test",
               "Mann-Whitney area; structural components by midranks (Sun and Xu 2014)", _I,
               "ranking.auroc, ranking.delong_covariance, significance.delong_test"),
    MetricSpec("overall", "AUPRC (average precision)", "threshold-free ranking under class imbalance",
               "sum_g (R_g - R_g-1) P_g; Davis-Goadrich interpolated area as a cross-check", _I,
               "ranking.average_precision, ranking.auprc_interpolated"),
    MetricSpec("overall", "partial AUROC at low FPR", "McClish standardisation", "area over FPR in [0, e], standardised", _I,
               "ranking.partial_auroc"),
    MetricSpec("overall", "recall and precision at fixed FPR; conformal threshold", "matched-FPR and conformal operating points",
               "largest TPR with FPR <= alpha; k-th smallest benign score, k = ceil((n+1)(1-alpha))", _I,
               "ranking.operating_point_at_fpr, ranking.conformal_threshold"),
    MetricSpec("overall", "alerts per day; precision at deployment base rate", "base-rate realism (Sommer and Paxson; Arp et al.)",
               "alerts / observed days; pi TPR / (pi TPR + (1 - pi) FPR)", _I,
               "ranking.alerts_per_day, ranking.precision_at_base_rate"),
    MetricSpec("overall", "zero-shot known vs novel", "real-only zero-shot split, never pooled (D-23, AS-35)",
               "all metrics per novelty group, unseen networks reported apart", _I, "generalisation.novelty_masks"),
    MetricSpec("overall", "cross-dataset / leave-one-network-out", "protocols P2 and P3 (D-16 as a protocol setting)",
               "train on some datasets or networks, test on held-out ones; site-calibrated variant", _I,
               "protocols, generalisation.variant_of, scorer"),
    # Forecast probability.
    MetricSpec("forecast", "Brier / log score / ECE per horizon", "proper scoring with censoring as ForecastPredictions.outcome",
               "means over triggers with a known outcome at k", _I, "forecasting.horizon_scores, calibration"),
    MetricSpec("forecast", "calibration: ECE (width, mass, debiased), ECE sweep, ACE, reliability, Murphy decomposition",
               "reliability analysis", "binned gaps; Kumar-Liang-Ma debiasing; Roelofs sweep; Nixon ACE; "
               "REL - RES + UNC + WBV - WBC", _I,
               "calibration.calibration_error, calibration.ece_sweep, calibration.adaptive_calibration_error, "
               "calibration.reliability_table, calibration.brier_decomposition"),
    MetricSpec("forecast", "temperature effects", "calibration under temperature scaling",
               "metrics over temperatures and at the maximum-likelihood temperature", _I, "calibration.temperature_effects"),
    MetricSpec("forecast", "CRPS of the time to infiltration; fair ensemble CRPS with route weights", "proper scoring",
               "sum_j (F(j) - 1[T <= j])^2 up to k; Ferro's fair estimator with weights", _I,
               "forecasting.crps_cumulative, forecasting.crps_ensemble"),
    MetricSpec("forecast", "skill scores vs persistence and climatology", "references built from labels",
               "1 - S_model / S_ref; Kaplan-Meier climatology from training triggers", _I,
               "forecasting.skill, forecasting.persistence, forecasting.climatology"),
    MetricSpec("forecast", "lead time; time to detect; detection rate", "alerts in the horizon-bounded window of each episode",
               "T_a - A_a(theta) at the own and the conformal threshold; missed episodes rank last", _I,
               "episodes.episode_alerts, episodes.lead_time_metrics"),
    MetricSpec("forecast", "C-index, Uno's C, time-dependent AUC, integrated Brier, D-calibration",
               "survival analysis with right-censoring", "Fenwick-tree concordance; IPCW (Graf; Uno); Haider et al.", _I,
               "survival.harrell_c, survival.uno_c, survival.cumulative_dynamic_auc, survival.integrated_brier, "
               "survival.d_calibration"),
    MetricSpec("next state", "MAE, RMSE, MSESS; energy score; Gaussian CRPS", "forecast of the observable window features",
               "masked errors per horizon and feature; Gneiting-Raftery energy score; closed-form Gaussian CRPS", _I,
               "state.errors_by_horizon, state.msess, state.energy_score, state.gaussian_crps"),
    MetricSpec("forecast", "stage top-k, macro and weighted F1, confusion, ordinal error, RPS", "per-step stage prediction",
               "tie-aware top-k; F1 over the evaluated classes; kill-chain MAE; ranked probability score", _I,
               "stages.stage_metrics, stages.confusion"),
    MetricSpec("forecast", "path precision/recall@N, edit distance, exact and prefix match, hit@k",
               "imagined vs realised attack paths", "normalised Levenshtein distance with tolerance epsilon", _I,
               "paths.path_metrics, paths.levenshtein"),
    MetricSpec("forecast", "ranking agreement (Kendall tau, NDCG) and ordinal safety", "ordinal safety (ARCH section 3.3)",
               "tau-b of route probabilities against true values; share of correctly ordered pairs by horizon", _I,
               "paths.kendall_tau_b, paths.ordinal_safety"),
    # Components.
    MetricSpec("CVG-AE", "masked reconstruction, edge AUROC/AP, KL, active units", "observed fields only (P-20)",
               "MSE over observed cells; ranking of candidate hyperedges; Var_x E[u|x] > 0.01", _I,
               "components.masked_mse, components.active_units, COMPONENT_METRICS"),
    MetricSpec("CVG-AE", "OOD AUROC (energy / likelihood ratio)", "not raw ELBO (P-21)", "AUROC of the novelty scores", _I,
               "components.COMPONENT_METRICS"),
    MetricSpec("TSTCT", "latent prediction error; no-future-leak test; attention-mask audit", "mask audit",
               "MSE of the next latent; largest output change under future perturbation (0); violating mass (0)", _I,
               "components.future_leak_change, components.COMPONENT_METRICS"),
    MetricSpec("Decoder", "decoded-vs-observed error per plane; believed-as-observed count", "provenance audit",
               "lowest and highest plane MSE; count must be 0", _I, "components.COMPONENT_METRICS"),
    MetricSpec("TAAFT", "belief Brier/log score/ECE; trust AUROC", "simulated worlds with known hidden state (P-14), "
               "injected corruption", "proper scores of the compromise belief; AUROC of 1 - trust", _I,
               "components.COMPONENT_METRICS"),
    MetricSpec("TAAFT", "energy OOD AUROC; early-warning lead time; false-alarm rate", "novelty and trend",
               "AUROC of the marginal energy; median completion - first energy alert; alerts on benign units", _I,
               "components.COMPONENT_METRICS"),
    MetricSpec("Advisor", "Delta P_inf, disruption, feasibility, regret vs oracle, analyst acceptance",
               "re-imagination and counterfactual replay", "means over triggers; oracle value - achieved value", _I,
               "components.COMPONENT_METRICS"),
    MetricSpec("Verifier", "ECE/Brier before vs after; drift-detection delay at fixed false-alarm rate; poisoning detection",
               "QCD on memory drift; injected poisoning", "threshold from drift-free run maxima; delay after onset", _I,
               "components.detection_delay_at_far, calibration.calibration_change, components.COMPONENT_METRICS"),
    MetricSpec("Generator", "MMD/Wasserstein; physics-violation rate; label preservation; TSTR; no leakage",
               "fidelity, diversity, utility, safety", "unbiased MMD^2; exact 1-D and sliced Wasserstein; shared ids (0)", _I,
               "components.mmd2_unbiased, components.wasserstein_per_feature, components.sliced_wasserstein, "
               "pipeline.splits"),
    MetricSpec("physics", "violation rate of model outputs; hard-limit violations", "hallucination control (D-18)",
               "share with Phi_phys > 0; count of hard-limit violations (0)", _I, "components.COMPONENT_METRICS"),
    MetricSpec("simulated worlds", "information audit and Fano ceiling", "mutual information per regime (P-15 gated)",
               "I(O; S); largest accuracy allowed by Fano's inequality; gap of the model", _I,
               "components.information_audit, components.fano_accuracy_bound"),
    # Operations and forensics.
    MetricSpec("operations", "latency per state update; throughput; memory growth; trigger time",
               "sustained streams (P8), NetFlow-only vs full telemetry", "quantiles with block-bootstrap intervals; "
               "batch means; OLS slope with Newey-West errors", _I,
               "operations.latency_quantiles, operations.throughput, operations.memory_growth"),
    MetricSpec("operations", "compute and latency profile", "stage timings, operations per update, peak memory",
               "per-stage quantiles with order-statistic intervals", _I, "operations.quantile_interval, scorer (profile)"),
    MetricSpec("forensics", "stage-onset timing error; patient-zero accuracy; narrative precision/recall",
               "replay on labelled incidents (P7)", "median |t_hat - t|; top-1 and top-k; maximum matching of steps", _I,
               "forensics.forensic_metrics"),
    # Explanations.
    MetricSpec("explanations", "deletion and insertion faithfulness, AOPC, gains over random orders",
               "perturbation curves of the attributions", "areas under the curves; Samek's AOPC", _I,
               "faithfulness.faithfulness_metrics, faithfulness.mean_curves"),
    # Arena.
    MetricSpec("arena", "steps to exceed the control; final return; learning curves; generalisation",
               "CyberWheel arena (P-CW), at least three seeds", "IQM with bootstrap intervals over runs; probability of "
               "improvement; performance profiles", _I, "arena.summarise, arena.learning_curve, arena.compare_agents"),
    # Statistics.
    MetricSpec("statistics", "bootstrap intervals: percentile and BCa; iid, stratified, stationary block, cluster",
               "resampling as weight matrices", "Politis-Romano blocks with the Politis-White length; grouped jackknife "
               "acceleration", _I, "resampling.estimate"),
    MetricSpec("statistics", "paired tests: McNemar (exact, mid-p), DeLong, permutation, Diebold-Mariano with HAC and HLN, "
               "Wilcoxon", "paired on the same units", "per matched seed, intersection-union p-value", _I,
               "significance"),
    MetricSpec("statistics", "multiple comparisons: Benjamini-Hochberg, Benjamini-Yekutieli, Holm",
               "families of comparisons", "adjusted p-values", _I, "significance.adjust_pvalues"),
    MetricSpec("statistics", "Friedman and Iman-Davenport tests, Nemenyi critical difference, Wilcoxon-Holm",
               "models over datasets (Demsar 2006)", "average ranks, CD, cliques", _I,
               "multidataset.friedman_nemenyi, multidataset.wilcoxon_holm"),
    MetricSpec("statistics", "seeds", "several seeds per configuration", "seed mean, SD and Student-t interval", _I,
               "significance.aggregate_seeds"),
    MetricSpec("statistics", "pre-registration and deviations", "hypotheses and analysis plan fixed before any run",
               "SHA-256 hashed, time-stamped, hash-chained registry; confirmatory hypothesis family", _I, "registry"),
    MetricSpec("ablations", "tiered ablations: inference-time, single-stage retrains, full retrains",
               "each ablation vs the full model (P-ABL)", "paired differences with BCa intervals and FDR over the table", _I,
               "protocols, reports.ablation_cells"),
)
