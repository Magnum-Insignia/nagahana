"""Component diagnostics: does each part of NagaHana do its own job? (docs/architecture.md section 7)

The diagnostics are read from `ModelOutputs.component`, a mapping name -> array, using the keys of
COMPONENT_KEYS. A metric is computed when all of its keys are present; the keys, shapes and meanings
form the contract between the model's diagnostic outputs and this module.

Metrics and their definitions

CVG-AE
  reconstruction error  mean squared error over observed cells only (recon_mask), D-41 and AS-04.
  edge AUROC / AP       ranking of candidate hyperedges (observed against sampled negatives), ranking.py.
  KL                    mean KL divergence of the posterior from the transition prior per position.
  active latent units   share of latent dimensions u with Var_x(E_q[u | x]) > 0.01 (thesis definition;
                        the activity statistic of Burda, Grosse and Salakhutdinov, ICLR 2016, arXiv:1509.00519).
  OOD AUROC             AUROC of the energy and of the likelihood-ratio score for out-of-distribution units
                        (not the raw ELBO, P-21; Ren et al., NeurIPS 2019, arXiv:1906.02845, on likelihood
                        ratios; Liu et al., NeurIPS 2020, arXiv:2010.03759, on energy scores).
TSTCT
  latent prediction error  mean squared error of the predicted next latent.
  no-future-leak test      the largest change of an output at time t when only inputs after t are
                           perturbed (`future_leak_change`); passes when it is 0 up to a tolerance.
  attention-mask audit     the number of attention weights placed on disallowed keys; passes at 0.
Decoder
  decoded-versus-observed error per plane, and the count of items rendered as observed whose true
  provenance is believed or forecast (required 0).
TAAFT
  belief Brier and log score on simulated worlds with known hidden state (P-14); trust AUROC: how well
  1 - trust ranks corrupted sources first; energy OOD AUROC; early-warning lead time of energy alerts
  per episode; false-alarm rate of energy alerts on benign units.
Advisor
  mean Delta P_inf of the top-ranked counter-measure sequence, mean disruption cost, feasibility rate,
  regret against the simulator oracle (oracle value minus achieved value), analyst acceptance rate.
Verifier
  ECE and Brier before and after calibration (calibration.calibration_change); drift-detection delay at
  a fixed false-alarm rate (`detection_delay_at_far`: the alarm threshold is the (1 - alpha) quantile of
  the run maxima of the detection statistic on drift-free runs, so a drift-free run alarms with
  probability alpha; the delay of a drift run is the time from the drift onset to the first crossing, as
  in the quickest-change-detection literature, Lorden, Annals of Mathematical Statistics 42:1897-1908,
  1971); injected-poisoning detection rate.
Generator
  MMD^2 to real data with a Gaussian kernel and the median-distance bandwidth, unbiased estimator
  (Gretton et al., JMLR 13:723-773, 2012); Wasserstein-1 distance per feature (exact in one dimension,
  scipy.stats.wasserstein_distance) and its sliced version over random directions (Bonneel, Rabin,
  Peyre and Pfister, Journal of Mathematical Imaging and Vision 51:22-45, 2015); physics-violation rate
  of the variants; label preservation; train-synthetic test-real F1 (Esteban, Hyland and Ratsch,
  arXiv:1706.02633, 2017); the zero-shot leakage check (no zero-shot sample among the Generator's
  training samples, D-23 and P-23; required 0).
Physics
  violation rate of model outputs: the share of outputs whose soft residual Phi_phys exceeds the
  tolerance, and the hard-limit violation count (required 0, D-18).

Information audit (protocol P6). For each observability regime, the mutual information I(O; S) between
observables and the hidden stage is estimated by lab/info_audit.py (proposal P-15, which must be
enabled), and Fano's inequality H(S | O) <= h(P_e) + P_e log(|S| - 1) bounds the achievable accuracy:
the smallest P_e satisfying it gives the ceiling 1 - P_e (Fano, Transmission of Information, MIT Press
1961; Cover and Thomas, Elements of Information Theory, 2nd ed., Wiley 2006, section 2.10). The model is
judged by its gap to this ceiling.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import stats

from nagahana.core.errors import InvariantViolation
from nagahana.evaluation._arrays import safe_ratio, unbatch, weight_matrix, weighted_quantile
from nagahana.evaluation.calibration import brier_score, calibration_error, log_loss
from nagahana.evaluation.metrics import counts, rates
from nagahana.evaluation.ranking import auroc, average_precision


@dataclass(frozen=True)
class ComponentKey:
    """A documented key of ModelOutputs.component."""

    name: str
    shape: str
    meaning: str


#: The diagnostic arrays this module reads (name -> shape and meaning).
COMPONENT_KEYS: dict[str, ComponentKey] = {k.name: k for k in (
    ComponentKey("cvgae.recon_pred", "[n, F]", "reconstructed field values (standardised)"),
    ComponentKey("cvgae.recon_true", "[n, F]", "observed field values"),
    ComponentKey("cvgae.recon_mask", "[n, F] bool", "True where the field was observed"),
    ComponentKey("cvgae.edge_score", "[e]", "score of each candidate hyperedge"),
    ComponentKey("cvgae.edge_label", "[e] 0/1", "1 for observed hyperedges, 0 for sampled negatives"),
    ComponentKey("cvgae.kl", "[n]", "KL(q || p) per position, nats"),
    ComponentKey("cvgae.latent_mean", "[n, d]", "posterior mean of the continuous latent per position"),
    ComponentKey("ood.energy", "[n]", "energy score, higher = less familiar"),
    ComponentKey("ood.likelihood_ratio", "[n]", "likelihood-ratio score, higher = less familiar"),
    ComponentKey("ood.label", "[n] 0/1", "1 for out-of-distribution units (novel families or networks)"),
    ComponentKey("tstct.latent_pred", "[n, d]", "predicted next latent"),
    ComponentKey("tstct.latent_true", "[n, d]", "posterior latent at the next position"),
    ComponentKey("tstct.leak_change", "[k]", "largest output change at t under perturbation of inputs after t"),
    ComponentKey("tstct.mask_violations", "[k]", "attention weight mass on disallowed keys, per audited batch"),
    ComponentKey("decoder.pred", "[n, F]", "decoded field values"),
    ComponentKey("decoder.true", "[n, F]", "observed field values"),
    ComponentKey("decoder.mask", "[n, F] bool", "True where the field was observed"),
    ComponentKey("decoder.plane", "[n]", "plane code of each decoded row"),
    ComponentKey("decoder.rendered", "[k]", "provenance rendered: 0 observed, 1 believed, 2 forecast"),
    ComponentKey("decoder.provenance", "[k]", "true provenance of each rendered item, same codes"),
    ComponentKey("belief.prob", "[n]", "compromise belief per (entity, trigger)"),
    ComponentKey("belief.true", "[n] 0/1", "true hidden compromise state (simulated worlds)"),
    ComponentKey("trust.score", "[s]", "trust in [0, 1] per telemetry source or field"),
    ComponentKey("trust.corrupted", "[s] 0/1", "1 for sources whose telemetry was corrupted"),
    ComponentKey("energy.score", "[n]", "marginal energy per unit"),
    ComponentKey("energy.novel", "[n] 0/1", "1 for novel units"),
    ComponentKey("energy.alert_time", "[e]", "first energy alert per episode, epoch s (NaN: none)"),
    ComponentKey("energy.completion", "[e]", "completion time of each episode, epoch s"),
    ComponentKey("energy.alert", "[n] 0/1", "energy alert per unit"),
    ComponentKey("energy.benign", "[n] 0/1", "1 for benign units"),
    ComponentKey("advisor.delta_p_inf", "[m]", "Delta P_inf of the top-ranked sequence per trigger"),
    ComponentKey("advisor.cost", "[m]", "disruption cost of the top-ranked sequence"),
    ComponentKey("advisor.feasible", "[m] 0/1", "feasibility of the top-ranked sequence"),
    ComponentKey("advisor.value", "[m]", "value achieved by the top-ranked sequence (simulator)"),
    ComponentKey("advisor.oracle_value", "[m]", "value of the simulator oracle's best sequence"),
    ComponentKey("advisor.accepted", "[k] 0/1", "analyst accepted the ranked advice"),
    ComponentKey("verifier.p_before", "[n]", "forecast probability before calibration"),
    ComponentKey("verifier.p_after", "[n]", "forecast probability after calibration"),
    ComponentKey("verifier.y", "[n] 0/1", "resolved outcome"),
    ComponentKey("verifier.null_stat", "[r0, T]", "drift statistic on drift-free runs"),
    ComponentKey("verifier.drift_stat", "[r1, T]", "drift statistic on runs with injected drift"),
    ComponentKey("verifier.drift_onset", "[r1]", "index of the drift onset in each drift run"),
    ComponentKey("verifier.step_seconds", "[]", "time between successive statistic values"),
    ComponentKey("verifier.poison_injected", "[k] 0/1", "1 for poisoned items"),
    ComponentKey("verifier.poison_flagged", "[k] 0/1", "1 for items flagged for review"),
    ComponentKey("generator.real", "[n1, D]", "real feature vectors"),
    ComponentKey("generator.synthetic", "[n2, D]", "generated feature vectors"),
    ComponentKey("generator.physics_residual", "[n2]", "Phi_phys of each variant"),
    ComponentKey("generator.label_source", "[n2]", "label of the real sample a variant derives from"),
    ComponentKey("generator.label_variant", "[n2]", "label of the variant"),
    ComponentKey("generator.tstr_decision", "[n] 0/1", "decisions on real data of a model trained on variants"),
    ComponentKey("generator.tstr_label", "[n] 0/1", "true labels of those real units"),
    ComponentKey("generator.trained_on", "[k] str", "sample ids the Generator trained on"),
    ComponentKey("generator.zero_shot_ids", "[z] str", "sample ids of the zero-shot split"),
    ComponentKey("physics.residual", "[n]", "Phi_phys of each model output"),
    ComponentKey("physics.hard_violation", "[n] 0/1", "hard-limit violation of each model output"),
    ComponentKey("audit.observation", "[n] int", "discrete (or pre-binned) observable of each unit (P6 information audit)"),
    ComponentKey("audit.hidden", "[n] int", "true hidden stage of each unit (simulated worlds)"),
    ComponentKey("audit.regime", "[n] str", "observability regime of each unit"),
    ComponentKey("audit.correct", "[n] 0/1", "1 where the model's most probable stage equals the hidden stage"),
    ComponentKey("ordinal.model_value", "[m, C, H]", "model value of each candidate at each horizon (ordinal safety)"),
    ComponentKey("ordinal.true_value", "[m, C, H]", "true value of each candidate at each horizon (simulated worlds)"),
    ComponentKey("ordinal.valid", "[m, C, H] bool", "True where the candidate exists at that horizon"),
)}


def _vec(c: Mapping[str, np.ndarray], key: str) -> np.ndarray:
    a = np.asarray(c[key], dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(a)):
        raise InvariantViolation(f"component {key!r} contains NaN or infinite values")
    return a


def masked_mse(pred: Any, true: Any, mask: Any, weights: Any = None) -> Any:
    """Mean squared error over masked cells, rows weighted ([n] or [B, n])."""
    p, t, m = np.asarray(pred, dtype=np.float64), np.asarray(true, dtype=np.float64), np.asarray(mask, dtype=bool)
    if p.shape != t.shape or m.shape != p.shape or p.ndim != 2:
        raise InvariantViolation("pred, true and mask must share the shape [n, F]")
    w, batched = weight_matrix(weights, p.shape[0])
    err = np.where(m, p - t, 0.0) ** 2
    return unbatch(safe_ratio(w @ err.sum(axis=1), w @ m.sum(axis=1).astype(np.float64)), batched)


def active_units(latent_mean: Any, weights: Any = None, *, threshold: float = 0.01) -> Any:
    """Share of latent dimensions whose posterior mean varies across units by more than `threshold`.

    The variance over units uses frequency weights, sum_i w_i (z_i - zbar)^2 / (sum_i w_i - 1), which is
    the sample variance of the resample a weight row represents.
    """
    z = np.asarray(latent_mean, dtype=np.float64)
    if z.ndim != 2 or z.shape[0] < 2:
        raise InvariantViolation("latent_mean must be [n, d] with n >= 2")
    w, batched = weight_matrix(weights, z.shape[0])
    tot = w.sum(axis=1)                                                 # [B]
    mean = safe_ratio(w @ z, tot[:, None])                              # [B, d]
    second = safe_ratio(w @ (z * z), tot[:, None])
    var = (second - mean * mean) * safe_ratio(tot, tot - 1.0)[:, None]
    return unbatch(np.mean(np.nan_to_num(var, nan=0.0) > threshold, axis=1), batched)


def future_leak_change(f: Callable[[np.ndarray], np.ndarray], x: np.ndarray, *, cut: int, trials: int = 8,
                       scale: float = 1.0, rng: np.random.Generator | None = None) -> float:
    """Largest |f(x')[:, :cut] - f(x)[:, :cut]| when only inputs at positions >= cut are perturbed.

    f maps a sequence batch [B, T, ...] to outputs [B, T, ...] (position-aligned). A causal model gives 0.
    """
    gen = rng if rng is not None else np.random.default_rng(0)
    base = np.asarray(f(x))
    worst = 0.0
    for _ in range(trials):
        xp = np.array(x, dtype=np.float64, copy=True)
        xp[:, cut:] = xp[:, cut:] + scale * gen.standard_normal(xp[:, cut:].shape)   # perturb the future only
        worst = max(worst, float(np.max(np.abs(np.asarray(f(xp))[:, :cut] - base[:, :cut]))))
    return worst


def drift_delays(null_stat: Any, drift_stat: Any, onset: Any, *, far: float, step_seconds: float = 1.0
                 ) -> tuple[float, np.ndarray, np.ndarray]:
    """(threshold, delay per drift run, false alarm before onset per run) at false-alarm probability `far`.

    The threshold is the (1 - far) quantile ("higher" order statistic) of the run maxima on drift-free
    runs. A drift run never crossing after its onset has delay +inf (it ranks last).
    """
    if not 0.0 < far < 1.0:
        raise ValueError("far must lie in (0, 1)")
    s0, s1 = np.asarray(null_stat, dtype=np.float64), np.asarray(drift_stat, dtype=np.float64)
    on = np.asarray(onset, dtype=np.int64).reshape(-1)
    if s0.ndim != 2 or s1.ndim != 2 or on.shape != (s1.shape[0],):
        raise InvariantViolation("null_stat [r0, T], drift_stat [r1, T] and onset [r1] are required")
    threshold = float(np.quantile(s0.max(axis=1), 1.0 - far, method="higher"))
    over = s1 > threshold                                               # [r1, T]
    idx = np.arange(s1.shape[1])[None, :]
    after = over & (idx >= on[:, None])
    first_after = np.where(after.any(axis=1), after.argmax(axis=1), -1)
    delays = np.where(first_after >= 0, (first_after - on) * step_seconds, np.inf)
    false_before = (over & (idx < on[:, None])).any(axis=1)
    return threshold, delays.astype(np.float64), false_before


def detection_delay_at_far(null_stat: Any, drift_stat: Any, onset: Any, *, far: float, step_seconds: float = 1.0
                           ) -> dict[str, float]:
    """Median detection delay (missed runs rank last) and detection rate of drift runs at false-alarm probability `far`."""
    threshold, delays, false_before = drift_delays(null_stat, drift_stat, onset, far=far, step_seconds=step_seconds)
    s0 = np.asarray(null_stat, dtype=np.float64)
    return {"threshold": threshold,
            "median_delay": float(np.median(delays)) if delays.size else math.nan,
            "detection_rate": float(np.isfinite(delays).mean()) if delays.size else math.nan,
            "false_alarm_before_onset": float(false_before.mean()) if false_before.size else math.nan,
            "null_false_alarm_rate": float(np.mean(s0.max(axis=1) > threshold))}


def mmd2_unbiased(x: Any, y: Any, *, bandwidth: float | None = None) -> float:
    """Unbiased MMD^2 with the Gaussian kernel exp(-||a - b||^2 / (2 h^2)), h = median pairwise distance by default."""
    a, b = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1] or a.shape[0] < 2 or b.shape[0] < 2:
        raise InvariantViolation("x and y must be [n, D] and [m, D] with n, m >= 2")
    z = np.vstack([a, b])
    sq = np.sum(z * z, axis=1)
    d2 = np.maximum(sq[:, None] + sq[None, :] - 2.0 * z @ z.T, 0.0)
    if bandwidth is None:
        iu = np.triu_indices(z.shape[0], k=1)
        med = float(np.median(np.sqrt(d2[iu])))
        h = med if med > 0 else 1.0
    else:
        h = float(bandwidth)
    k = np.exp(-d2 / (2.0 * h * h))
    n, m = a.shape[0], b.shape[0]
    kxx, kyy, kxy = k[:n, :n], k[n:, n:], k[:n, n:]
    return float((kxx.sum() - np.trace(kxx)) / (n * (n - 1)) + (kyy.sum() - np.trace(kyy)) / (m * (m - 1))
                 - 2.0 * kxy.mean())


def wasserstein_per_feature(x: Any, y: Any) -> np.ndarray:
    """Exact one-dimensional Wasserstein-1 distance of each feature [D]."""
    a, b = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1]:
        raise InvariantViolation("x and y must be [n, D] and [m, D]")
    return np.array([stats.wasserstein_distance(a[:, j], b[:, j]) for j in range(a.shape[1])])


def sliced_wasserstein(x: Any, y: Any, *, directions: int = 128, seed: int = 0) -> float:
    """Mean Wasserstein-1 distance of the projections on `directions` random unit vectors."""
    a, b = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((directions, a.shape[1]))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    pa, pb = a @ v.T, b @ v.T
    return float(np.mean([stats.wasserstein_distance(pa[:, j], pb[:, j]) for j in range(directions)]))


def fano_accuracy_bound(conditional_entropy: float, n_classes: int) -> float:
    """Largest accuracy compatible with H(S | O) (nats) by Fano's inequality."""
    if n_classes < 2:
        return 1.0
    hmax = math.log(n_classes)
    if conditional_entropy <= 0.0:
        return 1.0
    if conditional_entropy >= hmax:
        return 1.0 / n_classes

    def fano(pe: float) -> float:
        h = 0.0 if pe in (0.0, 1.0) else -pe * math.log(pe) - (1.0 - pe) * math.log(1.0 - pe)
        return h + pe * math.log(n_classes - 1)

    lo, hi = 0.0, 1.0 - 1.0 / n_classes                                # fano() increases on [0, 1 - 1/|S|]
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if fano(mid) < conditional_entropy:
            lo = mid
        else:
            hi = mid
    return 1.0 - hi


def information_audit(observation: Any, hidden: Any, regime: Any, correct: Any, *, enabled_proposals: Collection[str],
                      miller_madow: bool = True) -> list[dict[str, float | str]]:
    """Per regime: I(O; S), H(S), the Fano accuracy ceiling, the model's accuracy and its gap (P-15 gated)."""
    from nagahana.lab.info_audit import mutual_information

    obs, hid, reg = np.asarray(observation), np.asarray(hidden), np.asarray(regime).astype(str)
    ok = np.asarray(correct, dtype=np.float64)
    if not (obs.shape == hid.shape == reg.shape == ok.shape):
        raise InvariantViolation("observation, hidden, regime and correct must align")
    rows: list[dict[str, float | str]] = []
    for r in np.unique(reg):
        sel = reg == r
        mi = mutual_information(obs[sel].tolist(), hid[sel].tolist(), enabled_proposals=enabled_proposals,
                                miller_madow=miller_madow)
        _, cnt = np.unique(hid[sel], return_counts=True)
        p = cnt / cnt.sum()
        h_s = float(-(p * np.log(p)).sum())
        ceiling = fano_accuracy_bound(max(h_s - mi, 0.0), int(cnt.size))
        acc = float(ok[sel].mean())
        rows.append({"regime": str(r), "mutual_information": mi, "entropy": h_s, "accuracy_ceiling": ceiling,
                     "model_accuracy": acc, "gap": ceiling - acc, "n": float(sel.sum())})
    return rows


@dataclass(frozen=True)
class ComponentMetric:
    """One component diagnostic: required keys, the unit axis that is resampled and how it is computed."""

    name: str
    component: str
    label: str
    keys: tuple[str, ...]
    direction: str
    required: str
    compute: Callable[[Mapping[str, np.ndarray], Any], Any]
    resample: bool = True


def _decisions_rate(c: Mapping[str, np.ndarray], yk: str, dk: str, rate: str, w: Any) -> Any:
    out = rates(counts(_vec(c, yk).astype(np.int64), _vec(c, dk).astype(np.int64), w))[rate]
    return out


def _decoder_by_plane(c: Mapping[str, np.ndarray], w: Any) -> Any:
    plane = np.asarray(c["decoder.plane"]).astype(np.int64).reshape(-1)
    vals = []
    for p in np.unique(plane):
        sel = plane == p
        ww = None if w is None else np.asarray(w)[..., sel]
        vals.append(np.atleast_1d(masked_mse(np.asarray(c["decoder.pred"])[sel], np.asarray(c["decoder.true"])[sel],
                                             np.asarray(c["decoder.mask"])[sel], ww)))
    arr = np.vstack(vals)                                               # [planes, B]
    return np.column_stack([arr.min(axis=0), arr.max(axis=0)])          # [B, 2]: lowest and highest plane error


def _believed_as_observed(c: Mapping[str, np.ndarray], w: Any) -> Any:
    rendered = np.asarray(c["decoder.rendered"]).astype(np.int64).reshape(-1)
    true = np.asarray(c["decoder.provenance"]).astype(np.int64).reshape(-1)
    bad = ((rendered == 0) & (true != 0)).astype(np.float64)
    wm, batched = weight_matrix(w, bad.size)
    return unbatch(wm @ bad, batched)


def _regret(c: Mapping[str, np.ndarray], w: Any) -> Any:
    gap = _vec(c, "advisor.oracle_value") - _vec(c, "advisor.value")
    wm, batched = weight_matrix(w, gap.size)
    return unbatch(safe_ratio(wm @ gap, wm.sum(axis=1)), batched)


def _mean(key: str) -> Callable[[Mapping[str, np.ndarray], Any], Any]:
    def fn(c: Mapping[str, np.ndarray], w: Any) -> Any:
        v = _vec(c, key)
        wm, batched = weight_matrix(w, v.size)
        return unbatch(safe_ratio(wm @ v, wm.sum(axis=1)), batched)
    return fn


def _energy_lead(c: Mapping[str, np.ndarray], w: Any) -> Any:
    # Median over episodes of completion - first energy alert; an episode without an alert ranks last (-inf).
    alert = np.asarray(c["energy.alert_time"], dtype=np.float64).reshape(-1)
    done = _vec(c, "energy.completion")
    lead = np.where(np.isfinite(alert), done - alert, -np.inf)
    wm, batched = weight_matrix(w, lead.size)
    return unbatch(weighted_quantile(lead, wm, 0.5), batched)


def _energy_far(c: Mapping[str, np.ndarray], w: Any) -> Any:
    # Share of benign units on which an energy alert fired.
    alert, benign = _vec(c, "energy.alert"), _vec(c, "energy.benign")
    wm, batched = weight_matrix(w, alert.size)
    return unbatch(safe_ratio(wm @ (alert * benign), wm @ benign), batched)


def _leak_pass(c: Mapping[str, np.ndarray], _w: Any) -> Any:
    return float(np.max(np.abs(np.asarray(c["tstct.leak_change"], dtype=np.float64)))) <= 1e-6


def _no_leak(c: Mapping[str, np.ndarray], _w: Any) -> Any:
    trained = {str(s) for s in np.asarray(c["generator.trained_on"]).reshape(-1)}
    zero = {str(s) for s in np.asarray(c["generator.zero_shot_ids"]).reshape(-1)}
    return float(len(trained & zero))


#: Every component diagnostic of the catalogue, in table order.
COMPONENT_METRICS: tuple[ComponentMetric, ...] = (
    ComponentMetric("cvgae.reconstruction_error", "CVG-AE", "Reconstruction error on observed fields",
                    ("cvgae.recon_pred", "cvgae.recon_true", "cvgae.recon_mask"), "lower", "",
                    lambda c, w: masked_mse(c["cvgae.recon_pred"], c["cvgae.recon_true"], c["cvgae.recon_mask"], w)),
    ComponentMetric("cvgae.edge_auroc", "CVG-AE", "Edge AUROC on candidate hyperedges", ("cvgae.edge_score", "cvgae.edge_label"),
                    "higher", "", lambda c, w: auroc(_vec(c, "cvgae.edge_score"), _vec(c, "cvgae.edge_label").astype(np.int64), w)),
    ComponentMetric("cvgae.edge_ap", "CVG-AE", "Edge average precision on candidate hyperedges",
                    ("cvgae.edge_score", "cvgae.edge_label"), "higher", "",
                    lambda c, w: average_precision(_vec(c, "cvgae.edge_score"), _vec(c, "cvgae.edge_label").astype(np.int64), w)),
    ComponentMetric("cvgae.kl", "CVG-AE", "KL divergence per position (nats)", ("cvgae.kl",), "lower", "", _mean("cvgae.kl")),
    ComponentMetric("cvgae.active_units", "CVG-AE", "Active latent units (fraction)", ("cvgae.latent_mean",), "higher", "",
                    lambda c, w: active_units(c["cvgae.latent_mean"], w)),
    ComponentMetric("cvgae.ood_energy_auroc", "CVG-AE", "OOD AUROC (energy)", ("ood.energy", "ood.label"), "higher", "",
                    lambda c, w: auroc(_vec(c, "ood.energy"), _vec(c, "ood.label").astype(np.int64), w)),
    ComponentMetric("cvgae.ood_lr_auroc", "CVG-AE", "OOD AUROC (likelihood ratio)", ("ood.likelihood_ratio", "ood.label"),
                    "higher", "", lambda c, w: auroc(_vec(c, "ood.likelihood_ratio"), _vec(c, "ood.label").astype(np.int64), w)),
    ComponentMetric("tstct.latent_error", "TSTCT", "Next-latent prediction error", ("tstct.latent_pred", "tstct.latent_true"),
                    "lower", "", lambda c, w: masked_mse(c["tstct.latent_pred"], c["tstct.latent_true"],
                                                          np.ones(np.shape(c["tstct.latent_pred"]), dtype=bool), w)),
    ComponentMetric("tstct.no_future_leak", "TSTCT", "No-future-leak test", ("tstct.leak_change",), "", "pass",
                    _leak_pass, resample=False),
    ComponentMetric("tstct.mask_audit", "TSTCT", "Attention-mask audit (violating weight mass)", ("tstct.mask_violations",),
                    "", "0", lambda c, w: float(np.sum(np.abs(np.asarray(c["tstct.mask_violations"], dtype=np.float64)))),
                    resample=False),
    ComponentMetric("decoder.plane_error", "Decoder", "Decoded-versus-observed error per plane (lowest, highest)",
                    ("decoder.pred", "decoder.true", "decoder.mask", "decoder.plane"), "lower", "", _decoder_by_plane),
    ComponentMetric("decoder.believed_as_observed", "Decoder", "Believed-as-observed renderings",
                    ("decoder.rendered", "decoder.provenance"), "", "0", _believed_as_observed),
    ComponentMetric("taaft.belief_brier", "TAAFT", "Belief Brier score (simulated worlds)", ("belief.prob", "belief.true"),
                    "lower", "", lambda c, w: brier_score(_vec(c, "belief.prob"), _vec(c, "belief.true").astype(np.int64), w)),
    ComponentMetric("taaft.belief_log", "TAAFT", "Belief log score (simulated worlds)", ("belief.prob", "belief.true"),
                    "lower", "", lambda c, w: log_loss(_vec(c, "belief.prob"), _vec(c, "belief.true").astype(np.int64), weights=w)),
    ComponentMetric("taaft.belief_ece", "TAAFT", "Belief ECE (simulated worlds)", ("belief.prob", "belief.true"),
                    "lower", "", lambda c, w: calibration_error(_vec(c, "belief.prob"), _vec(c, "belief.true").astype(np.int64),
                                                                bins=10, weights=w)),
    ComponentMetric("taaft.trust_auroc", "TAAFT", "Trust AUROC (injected corruption)", ("trust.score", "trust.corrupted"),
                    "higher", "", lambda c, w: auroc(1.0 - _vec(c, "trust.score"), _vec(c, "trust.corrupted").astype(np.int64), w)),
    ComponentMetric("taaft.energy_auroc", "TAAFT", "Energy OOD AUROC", ("energy.score", "energy.novel"), "higher", "",
                    lambda c, w: auroc(_vec(c, "energy.score"), _vec(c, "energy.novel").astype(np.int64), w)),
    ComponentMetric("taaft.energy_lead", "TAAFT", "Early-warning lead time of energy growth (s)",
                    ("energy.alert_time", "energy.completion"), "higher", "", _energy_lead),
    ComponentMetric("taaft.energy_far", "TAAFT", "False-alarm rate of energy alerts", ("energy.alert", "energy.benign"),
                    "lower", "", _energy_far),
    ComponentMetric("advisor.delta_p_inf", "Advisor", "Mean Delta P_inf of top-ranked sequence", ("advisor.delta_p_inf",),
                    "higher", "", _mean("advisor.delta_p_inf")),
    ComponentMetric("advisor.cost", "Advisor", "Mean disruption cost of top-ranked sequence", ("advisor.cost",), "lower", "",
                    _mean("advisor.cost")),
    ComponentMetric("advisor.feasibility", "Advisor", "Feasibility rate", ("advisor.feasible",), "higher", "",
                    _mean("advisor.feasible")),
    ComponentMetric("advisor.regret", "Advisor", "Regret against simulator oracle", ("advisor.value", "advisor.oracle_value"),
                    "lower", "", _regret),
    ComponentMetric("advisor.acceptance", "Advisor", "Analyst acceptance", ("advisor.accepted",), "higher", "",
                    _mean("advisor.accepted")),
    ComponentMetric("verifier.ece_before", "Verifier", "ECE before calibration", ("verifier.p_before", "verifier.y"), "lower", "",
                    lambda c, w: calibration_error(_vec(c, "verifier.p_before"), _vec(c, "verifier.y").astype(np.int64), bins=10, weights=w)),
    ComponentMetric("verifier.ece_after", "Verifier", "ECE after calibration", ("verifier.p_after", "verifier.y"), "lower", "",
                    lambda c, w: calibration_error(_vec(c, "verifier.p_after"), _vec(c, "verifier.y").astype(np.int64), bins=10, weights=w)),
    ComponentMetric("verifier.brier_before", "Verifier", "Brier before calibration", ("verifier.p_before", "verifier.y"),
                    "lower", "", lambda c, w: brier_score(_vec(c, "verifier.p_before"), _vec(c, "verifier.y").astype(np.int64), w)),
    ComponentMetric("verifier.brier_after", "Verifier", "Brier after calibration", ("verifier.p_after", "verifier.y"),
                    "lower", "", lambda c, w: brier_score(_vec(c, "verifier.p_after"), _vec(c, "verifier.y").astype(np.int64), w)),
    ComponentMetric("verifier.poison_detection", "Verifier", "Injected-poisoning detection rate",
                    ("verifier.poison_injected", "verifier.poison_flagged"), "higher", "",
                    lambda c, w: _decisions_rate(c, "verifier.poison_injected", "verifier.poison_flagged", "recall", w)),
    ComponentMetric("generator.physics_violation", "Generator", "Physics-violation rate of variants",
                    ("generator.physics_residual",), "lower", "",
                    lambda c, w: _mean_indicator(_vec(c, "generator.physics_residual") > 1e-3, w)),
    ComponentMetric("generator.label_preservation", "Generator", "Label preservation",
                    ("generator.label_source", "generator.label_variant"), "higher", "",
                    lambda c, w: _mean_indicator(np.asarray(c["generator.label_source"]).reshape(-1)
                                                 == np.asarray(c["generator.label_variant"]).reshape(-1), w)),
    ComponentMetric("generator.tstr_f1", "Generator", "Train-synthetic test-real F1",
                    ("generator.tstr_label", "generator.tstr_decision"), "higher", "",
                    lambda c, w: _decisions_rate(c, "generator.tstr_label", "generator.tstr_decision", "f1", w)),
    ComponentMetric("generator.zero_shot_leakage", "Generator", "Zero-shot leakage check (shared samples)",
                    ("generator.trained_on", "generator.zero_shot_ids"), "", "0", _no_leak, resample=False),
    ComponentMetric("physics.violation_rate", "Physics", "Violation rate of model outputs (soft residual)",
                    ("physics.residual",), "lower", "", lambda c, w: _mean_indicator(_vec(c, "physics.residual") > 0.0, w)),
    ComponentMetric("physics.hard_violations", "Physics", "Hard-limit violations", ("physics.hard_violation",), "", "0",
                    lambda c, w: _count_indicator(_vec(c, "physics.hard_violation") > 0.5, w)),
)


def _mean_indicator(flag: np.ndarray, w: Any) -> Any:
    f = np.asarray(flag, dtype=np.float64).reshape(-1)
    wm, batched = weight_matrix(w, f.size)
    return unbatch(safe_ratio(wm @ f, wm.sum(axis=1)), batched)


def _count_indicator(flag: np.ndarray, w: Any) -> Any:
    f = np.asarray(flag, dtype=np.float64).reshape(-1)
    wm, batched = weight_matrix(w, f.size)
    return unbatch(wm @ f, batched)


def available_metrics(component: Mapping[str, np.ndarray]) -> list[ComponentMetric]:
    """The component metrics whose keys are all present."""
    return [m for m in COMPONENT_METRICS if all(k in component for k in m.keys)]


def unit_count(metric: ComponentMetric, component: Mapping[str, np.ndarray]) -> int:
    """Number of units along the resampled axis of a metric (the length of its first key)."""
    return int(np.asarray(component[metric.keys[0]]).shape[0])


def generator_distances(component: Mapping[str, np.ndarray], *, directions: int = 128, seed: int = 0) -> dict[str, float]:
    """MMD^2 and Wasserstein distances of the variants to real data."""
    if "generator.real" not in component or "generator.synthetic" not in component:
        return {}
    real, syn = component["generator.real"], component["generator.synthetic"]
    return {"generator.mmd2": mmd2_unbiased(real, syn),
            "generator.wasserstein_mean": float(np.mean(wasserstein_per_feature(real, syn))),
            "generator.wasserstein_sliced": sliced_wasserstein(real, syn, directions=directions, seed=seed)}


def generator_distance_intervals(component: Mapping[str, np.ndarray], *, n_resamples: int, confidence: float,
                                 rng: np.random.Generator, directions: int = 128, seed: int = 0
                                 ) -> dict[str, tuple[float, float, float]]:
    """(value, low, high) of each generator distance.

    The real and the generated samples are resampled independently with replacement (a two-sample
    bootstrap) and the interval is the percentile interval of the replicates.
    """
    point = generator_distances(component, directions=directions, seed=seed)
    if not point:
        return {}
    real = np.asarray(component["generator.real"], dtype=np.float64)
    syn = np.asarray(component["generator.synthetic"], dtype=np.float64)
    reps: dict[str, list[float]] = {k: [] for k in point}
    for _ in range(n_resamples):
        rr = real[rng.integers(0, real.shape[0], real.shape[0])]
        ss = syn[rng.integers(0, syn.shape[0], syn.shape[0])]
        for k, v in generator_distances({"generator.real": rr, "generator.synthetic": ss}, directions=directions,
                                        seed=seed).items():
            reps[k].append(v)
    lo_q, hi_q = (1.0 - confidence) / 2.0, (1.0 + confidence) / 2.0
    out: dict[str, tuple[float, float, float]] = {}
    for k, v in point.items():
        arr = np.asarray(reps[k])
        arr = arr[np.isfinite(arr)]
        lo = float(np.quantile(arr, lo_q)) if arr.size else math.nan
        hi = float(np.quantile(arr, hi_q)) if arr.size else math.nan
        out[k] = (v, lo, hi)
    return out
