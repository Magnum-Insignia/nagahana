"""Evaluation protocols P1 to P8 as objects: what each answers, which units, models, metrics and outputs.

The protocols are those of the thesis evaluation chapter (Table "Evaluation protocols"), with the
evaluation questions EQ1 to EQ8 they answer and the data sources of the evaluation matrix:

    P1  Zero-shot, novel and known   EQ1-EQ5  train on known families; test on real zero-shot data;
                                              novel and known reported separately (D-23; AS-35 for D-16)
    P2  Cross-dataset transfer       EQ5      train on one dataset, test on another through the data model
    P3  Leave-one-network-out        EQ5      hold out each network (or OT plant) in turn; also after site
                                              calibration (AS-26 budget)
    P4  Observability regimes        EQ6      deterministic projections of real test records onto sensor
                                              tiers: full, flow and packet, flow-only, sampled, encrypted
                                              payload, untapped segment
    P5  Telemetry corruption         EQ6      clock skew, reordering, dropped and duplicated records,
                                              spoofed field values, flooding
    P6  Simulated worlds             EQ4, EQ7 known hidden state; belief, trust, ordinal safety and the
                                              information audit per regime
    P7  Forensic backtesting         EQ3, EQ8 replay of labelled incidents: onset, patient zero,
                                              narrative, lead times from the replay timeline
    P8  Operational load             EQ8      sustained streams at increasing state-update rates

Two further protocols complete the evaluation against frontier systems:

    P-ABL  Tiered ablations          EQ7      each variant an L model with one component removed: inference-
                                              time ablations (no retraining), single-stage retrains and full
                                              retrains; every ablation is compared with the full model on the
                                              units of a base protocol (P1 to P7), with BCa intervals, paired
                                              tests and false-discovery-rate control over the ablation family
    P-CW   Cyber-defence arena       EQ7, EQ8 defender agents (NagaHana's Advisor, CyberWorld's world-model agents,
                                              model-free agents) trained and evaluated in a simulated defended
                                              network such as CyberWheel: environment steps to exceed the
                                              control, learning curves, final return and generalisation to a
                                              held-out attacker strategy, larger networks and degraded
                                              observations, over at least three seeds (arena.py)

Every protocol scores NagaHana, the logistic-regression family and the reproduced published baselines
(CyberWorld's RSSM among them, under every forecasting protocol and on the simulated worlds' ground truth)
through the same code (scorer.py); persistence and climatology are built by the evaluation itself; the
ablated variants of NagaHana are scored against the full model. Every protocol run must be pre-registered
(registry.py).
"""

from __future__ import annotations

from dataclasses import dataclass

from nagahana.core.errors import InvariantViolation

MODEL_ROLES: tuple[str, ...] = ("primary", "lr", "published", "reference", "ablation")


@dataclass(frozen=True)
class Protocol:
    """One evaluation protocol of the thesis.

    splits      the meta `split` values whose units are scored (the calibration split is read for conformal
                thresholds and the training split for climatology, never scored)
    models      the model roles compared
    tasks       the prediction records scored
    metrics     the metric families reported
    outputs     the tables (thesis labels) and figure data produced
    variant     the ModelOutputs.config key that distinguishes runs of the protocol ("" when none)
    sources     the data sources of the evaluation matrix on which the protocol runs
    """

    id: str
    name: str
    questions: tuple[str, ...]
    construction: str
    splits: tuple[str, ...]
    models: tuple[str, ...]
    tasks: tuple[str, ...]
    metrics: tuple[str, ...]
    outputs: tuple[str, ...]
    variant: str
    sources: tuple[str, ...]


ALL_MODELS = ("primary", "lr", "published", "reference", "ablation")
DETECTION_METRICS = ("precision", "recall", "f1", "fpr", "fnr", "detection_error_rate", "balanced_accuracy", "mcc",
                     "auroc", "auprc", "pauroc", "recall_at_fpr", "alerts_per_day")
FORECAST_METRICS = ("brier", "log_score", "ece", "bss_persistence", "bss_climatology", "crps", "crpss_climatology")

PROTOCOLS: dict[str, Protocol] = {p.id: p for p in (
    Protocol(
        "P1", "Zero-shot, novel and known", ("EQ1", "EQ2", "EQ3", "EQ4", "EQ5"),
        "Train on known families; test on real zero-shot data; novel and known reported separately. Whether "
        "novel includes unseen networks as well as unseen families is a configuration of the protocol (D-16).",
        ("zero_shot",), ALL_MODELS,
        ("detection", "forecast", "state_forecast", "stage", "paths", "time_to_event", "episodes", "component"),
        (*DETECTION_METRICS, *FORECAST_METRICS, "mae", "rmse", "msess", "median_lead_time", "alerted_before_completion",
         "lead_time_at_fpr", "c_index", "uno_c", "td_auc", "ibs", "top1", "top3", "macro_f1", "precision_at_n",
         "recall_at_n", "median_best_distance", "kendall_tau", "ordinal_safety"),
        ("res-detection", "res-ci", "res-forecast", "res-state", "res-timeliness", "res-stages", "res-components",
         "res-ablations", "reliability", "roc", "pr", "lead_time_distribution", "edit_distance", "td_auc", "dcal",
         "stage_confusion", "skill_by_horizon", "temperature_effects"),
        "", ("cic-ids2017", "cse-cic-ids2018", "ctu-13", "unsw-nb15", "cic-iot-2023", "ot")),
    Protocol(
        "P2", "Cross-dataset transfer", ("EQ5",),
        "Train on one dataset and test on another through the common data model; the diagonal is the "
        "chronological split of the training dataset.",
        ("test", "zero_shot"), ("primary", "lr", "published", "ablation"), ("detection",),
        DETECTION_METRICS, ("res-transfer", "res-transfer-f1", "res-ci", "res-ablations"), "train_datasets",
        ("cic-ids2017", "cse-cic-ids2018", "ctu-13", "unsw-nb15", "lanl", "ot")),
    Protocol(
        "P3", "Leave-one-network-out", ("EQ5",),
        "Configurable protocol: hold out each network (or OT plant) in turn and train on the rest; the same "
        "evaluation after calibration to the held-out site (AS-26 budget).",
        ("test", "zero_shot"), ("primary", "lr", "published"), ("detection",),
        DETECTION_METRICS, ("res-lono", "res-sitecal"), "held_out",
        ("cic-ids2017", "cse-cic-ids2018", "ctu-13", "unsw-nb15", "lanl", "ot")),
    Protocol(
        "P4", "Observability regimes", ("EQ6",),
        "Deterministic projection of real test records onto sensor tiers: full, flow and packet, flow-only, "
        "sampled, encrypted payload, untapped segment.",
        ("test", "zero_shot"), ("primary", "lr", "published"), ("detection", "forecast"),
        ("f1", "auprc", "brier", "degradation_signalled"), ("res-robustness",), "regime",
        ("cic-ids2017", "cse-cic-ids2018", "ctu-13", "unsw-nb15", "lanl", "ot", "simulated")),
    Protocol(
        "P5", "Telemetry corruption", ("EQ6",),
        "Injected clock skew, reordering, dropped and duplicated records, spoofed field values and flooding.",
        ("test", "zero_shot"), ("primary", "lr", "published"), ("detection", "forecast", "component"),
        ("f1", "auprc", "brier", "trust_auroc", "degradation_signalled"), ("res-robustness", "res-components"),
        "corruption", ("cic-ids2017", "cse-cic-ids2018", "ctu-13", "unsw-nb15", "lanl", "ot", "simulated")),
    Protocol(
        "P6", "Simulated worlds", ("EQ4", "EQ7"),
        "Simulated IT and OT networks with known hidden state and explicit observation models; information "
        "audit per regime.",
        ("test", "zero_shot"), ("primary", "lr", "published", "ablation"), ("stage", "paths", "component"),
        ("belief_brier", "trust_auroc", "ordinal_safety", "kendall_tau", "top1", "macro_f1", "accuracy_ceiling"),
        ("res-stages", "res-components", "information_audit", "ordinal_safety_horizon"), "", ("simulated",)),
    Protocol(
        "P7", "Forensic backtesting", ("EQ3", "EQ8"),
        "Replay of labelled incidents from the datasets with red-team or attack schedules.",
        ("test", "zero_shot"), ("primary", "lr", "published"), ("forensics", "forecast", "episodes"),
        ("median_onset_error_s", "patient_zero_top1", "patient_zero_topk", "narrative_precision", "narrative_recall",
         "median_lead_time", "alerted_before_completion", "lead_time_at_fpr"),
        ("res-forensics", "res-timeliness", "res-ci", "lead_time_distribution"), "",
        ("cic-ids2017", "cse-cic-ids2018", "lanl", "ot")),
    Protocol(
        "P8", "Operational load", ("EQ8",), "Sustained streams at increasing state-update rates.",
        ("test", "zero_shot"), ("primary",), ("operations",),
        ("latency_p50_ms", "latency_p99_ms", "throughput_per_s", "memory_gib_per_day", "trigger_median_s",
         "stage_latency_p50_ms", "stage_latency_p99_ms", "flops_per_update", "peak_memory_gib"),
        ("res-operations", "res-profile"), "", ("cse-cic-ids2018", "lanl")),
    Protocol(
        "P-ABL", "Tiered ablations", ("EQ7",),
        "Each variant an L model with one component removed (inference-time, single-stage retrain, full retrain), "
        "scored on the units of a base protocol and compared with the full model.",
        ("test", "zero_shot"), ("primary", "ablation"),
        ("detection", "forecast", "state_forecast", "stage", "paths", "time_to_event", "episodes"),
        (*DETECTION_METRICS, *FORECAST_METRICS, "median_lead_time"), ("res-ablations",), "base",
        ("cic-ids2017", "cse-cic-ids2018", "ctu-13", "unsw-nb15", "cic-iot-2023", "ot", "simulated")),
    Protocol(
        "P-CW", "Cyber-defence arena", ("EQ7", "EQ8"),
        "Defender agents trained and evaluated in a simulated defended network (CyberWheel): steps to exceed the "
        "control, learning curves, final return and generalisation, over at least three seeds.",
        (), ("primary", "published"), ("arena",),
        ("final_iqm", "final_mean", "improvement_over_control", "steps_to_exceed", "steps_to_exceed_sustained",
         "probability_of_improvement"),
        ("res-arena", "learning_curves", "performance_profiles"), "environment", ("simulated",)),
)}

#: Columns of the evaluation matrix (thesis Table "Evaluation matrix"): report-group ids and labels.
MATRIX_SOURCES: tuple[tuple[str, str], ...] = (
    ("cic-ids2017", "CIC-IDS2017"), ("cse-cic-ids2018", "CSE-CIC-IDS2018"), ("ctu-13", "CTU-13"),
    ("unsw-nb15", "UNSW-NB15"), ("cic-iot-2023", "CIC-IoT-2023"), ("lanl", "LANL"), ("ot", "OT"),
    ("simulated", "Simulated"),
)


def get_protocol(pid: str) -> Protocol:
    """The protocol with id P1 ... P8, P-ABL or P-CW."""
    try:
        return PROTOCOLS[pid.upper()]
    except KeyError as exc:
        raise InvariantViolation(f"unknown protocol {pid!r}; known: {sorted(PROTOCOLS)}") from exc
