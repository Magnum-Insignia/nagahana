"""Discrete-time hazard LR: the static infiltration forecaster P_inf(k), k = 1 ... K.

Model (Singer and Willett, "It's about time: using discrete-time survival analysis to study duration and
the timing of events", Journal of Educational Statistics 18(2), 1993; Tutz and Schmid, Modeling Discrete
Time-to-Event Data, Springer 2016, Chapter 3). For a trigger with window features x and step j:

    h_j(x) = P(T = j | T >= j, x) = sigmoid(eta_j),   eta_j = alpha_j + w.x                (shared)
                                                       eta_j = alpha_j + (w + d_j).x        (per_horizon)
    S(k) = prod_{j <= k} (1 - h_j),   P_inf(k) = 1 - S(k)

P_inf is non-decreasing in k by construction: every factor 1 - h_j lies in [0, 1]. It is computed as
-expm1(sum_{j <= k} log1p(-h_j)) in float64.

Person-period likelihood with right censoring. A trigger with its first infiltration in step e contributes
the person-period rows j = 1 ... e with y_j = 1[j = e]; a trigger censored after c event-free observed
steps contributes the rows j = 1 ... c with y_j = 0 (event_step and observed_steps of the trigger table,
which are the targets of NagaHana's Forecaster, forecaster.losses.survival_targets). The negative log-
likelihood of a trigger is then a sum of binary cross-entropies over its rows (Allison, "Discrete-time
methods for the analysis of event histories", Sociological Methodology 13, 1982), so the model is a
logistic regression on the person-period data. The rows are never materialised: with s = w.x, the
logits of all K periods of a trigger are s + alpha (a [n, K] matrix masked by the at-risk set), so the
objective costs O(n K) beyond the product X w.

    F = (1 / S) sum_{i, j at risk} c_ij [log(1 + exp(eta_ij)) - y_ij eta_ij] + (lambda / 2) (||w||^2 + sum_j ||d_j||^2)

with class weights c = (N / (2 N_y)) ** p over the person-period rows (p chosen with lambda) and the
period intercepts alpha unpenalised. "per_horizon" lets every step have its own coefficients w + d_j, with
the deviations d_j shrunk towards the shared w (regularised multi-task learning, Evgeniou and Pontil,
KDD 2004), which stays estimable for steps with few events (AS-524).

Steps without an event. When no training trigger has its event in step j, the likelihood increases
without bound as alpha_j -> -infinity: the maximum-likelihood hazard of that step is 0. Such steps are
held at h_j = 0 exactly (their person-period rows then contribute log(1 - 0) = 0 and leave w unchanged),
listed in `zero_steps`, and the remaining parameters are fitted by the finite optimum (AS-524).

Calibration (AS-517). On the validation triggers, (a, b) in eta'_j = a eta_j + b maximise the same
censored likelihood: Platt's logistic recalibration (Platt 1999) applied to the hazard logit. Any
per-step transform of the hazards keeps P_inf monotone.

Alert threshold. The forecaster's own operating threshold on P_inf(K) is chosen on the validation triggers
whose outcome at K is known (an infiltration within K steps, or K observed steps without one) by the rule
of HazardConfig.threshold (max-F1 by default, as the detector's; AS-514) and reported as the record's
`alert_threshold`, so lead times can be read at each method's own threshold. The record also carries
`infiltrated_now` (corpus.py; the persistence reference of the evaluation, AS-521).

Time to event. The survival curve S(k) on the grid t_k = k w (seconds after the trigger) and the risk
score -RMST = -w sum_{k=0}^{K-1} S(k) with S(0) = 1, minus the restricted mean time to infiltration within
the horizon (Royston and Parmar, BMC Medical Research Methodology 13:152, 2013; AS-526), give the
time-to-event record; "p_inf_k" uses P_inf(K) as the risk score instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from nagahana.core.errors import InvariantViolation
from nagahana.evaluation.predictions import ForecastPredictions, TimeToEventPredictions

from .common import RawRows, ZRows, fit_standardiser, groups_of, solver_rows
from .config import FeatureConfig, HazardConfig, SolverConfig, StandardiserConfig, config_to_dict, from_dict
from .corpus import LRCorpus
from .design import TriggerDesign, design_rows, gather, trigger_units, unit_meta
from .features import FeatureColumn
from .logistic import SolveInfo, lbfgs_minimise, log1pexp
from .precondition import BlockPreconditioner, CholeskyPreconditioner, Preconditioner, sample_mask
from .scoring import LOWER_IS_BETTER, brier_k, survival_nll
from .standardise import Standardiser
from .temporal_cv import SelectionResult, select_hyperparameters, selection_from_dict
from .thresholds import ThresholdChoice, select_threshold


def risk_sets(event_step: np.ndarray, observed_steps: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """(at_risk bool [m, K], y float [m, K]) of the person-period expansion (module docstring)."""
    e = np.asarray(event_step, dtype=np.int64)
    c = np.asarray(observed_steps, dtype=np.int64)
    if np.any((e < 0) | (e > k)) or np.any((c < 0) | (c > k)) or np.any((e > 0) & (e > c)):
        raise InvariantViolation("event_step and observed_steps must lie in 0 ... K, with event_step <= observed_steps")
    steps = np.arange(1, k + 1)[None, :]
    last = np.where(e > 0, e, c)
    at_risk = steps <= last[:, None]
    y = ((e[:, None] > 0) & (steps == e[:, None])).astype(np.float64)
    return at_risk, y


def person_period_weights(at_risk: np.ndarray, y: np.ndarray, power: float) -> np.ndarray:
    """c_ij = (N / (2 N_y)) ** power over the at-risk person-period rows, 0 elsewhere."""
    n = float(at_risk.sum())
    n1 = float(y[at_risk].sum())
    n0 = n - n1
    if n1 <= 0 or n0 <= 0:
        raise InvariantViolation(f"the hazard model needs event and event-free person-periods (events {n1:.0f}, others {n0:.0f})")
    c1, c0 = (n / (2 * n1)) ** power, (n / (2 * n0)) ** power
    return np.where(at_risk, np.where(y > 0.5, c1, c0), 0.0)


def survival_from_hazard(h: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(S [m, K], P_inf [m, K]) from hazards, in float64 (module docstring)."""
    log_s = np.cumsum(np.log1p(-np.clip(np.asarray(h, dtype=np.float64), 0.0, 1.0 - 1e-16)), axis=1)
    return np.exp(log_s), -np.expm1(log_s)


@dataclass
class HazardParams:
    """Fitted parameters: shared slopes w [D'], step intercepts alpha [K], deviations d [K, D'], and the
    steps held at zero hazard (bool [K])."""

    w: np.ndarray
    alpha: np.ndarray
    d: np.ndarray
    zero: np.ndarray

    def eta(self, z: np.ndarray) -> np.ndarray:
        """Hazard logits [n, K] for standardised rows z [n, D'] (-inf on the zero steps)."""
        e = (z @ self.w)[:, None] + self.alpha[None, :] + z @ self.d.T
        return np.where(self.zero[None, :], -np.inf, e)


def hazard_preconditioner(rows: ZRows, cweights: np.ndarray, *, lam: float, d_cols: int, per_horizon: bool,
                          solver: SolverConfig, seed: int = 0) -> Preconditioner:
    """Boehning's bound of the person-period Hessian (precondition.py) on a seeded trigger sample.

    With row weights r_i = sum_j c_ij and S the sampled weight: G_zz = Z^T diag(r) Z / S, G_za = Z^T C / S,
    G_aa = diag(sum_i c_ij) / S. Shared coefficients: B = (1/4) [[G_zz, G_za], [G_za^T, G_aa]] + lambda
    diag(1, 0); a step without weight (held at zero hazard) gets 1 on its inert intercept. Per-horizon
    deviations add the blocks (1/4) Z^T diag(c_:j) Z / S + lambda I, coupled to nothing else in the bound.
    """
    k = cweights.shape[1]
    mask = None
    gzz = np.zeros((d_cols, d_cols))
    gza = np.zeros((d_cols, k))
    gaa = np.zeros(k)
    gdev = [np.zeros((d_cols, d_cols)) for _ in range(k)] if per_horizon else []
    total = 0.0
    for pos, z in rows():
        if mask is None:
            mask = sample_mask(cweights.shape[0], solver.precond_rows, seed)
        keep = mask[pos]
        if not keep.any():
            continue
        zz = z[keep]
        cw = cweights[pos[keep]]                                   # [c, K]
        r = cw.sum(axis=1)
        gzz += (zz * r[:, None]).T @ zz
        gza += zz.T @ cw
        gaa += cw.sum(axis=0)
        for j in range(len(gdev)):
            gdev[j] += (zz * cw[:, j][:, None]).T @ zz
        total += float(r.sum())
    if total <= 0:
        raise InvariantViolation("the hazard preconditioning sample holds no weighted person-period")
    top = np.concatenate([gzz, gza], axis=1)
    bot = np.concatenate([gza.T, np.diag(gaa)], axis=1)
    b = 0.25 * np.concatenate([top, bot], axis=0) / total
    b += lam * np.diag(np.r_[np.ones(d_cols), np.zeros(k)])
    inert = np.flatnonzero(gaa <= 0)
    b[d_cols + inert, d_cols + inert] = 1.0
    main = CholeskyPreconditioner.from_matrix(b)
    if not per_horizon:
        return main
    blocks: list[tuple[slice, Preconditioner]] = [(slice(0, d_cols + k), main)]
    for j in range(k):
        a0 = d_cols + k + j * d_cols
        blocks.append((slice(a0, a0 + d_cols), CholeskyPreconditioner.from_matrix(0.25 * gdev[j] / total + lam * np.eye(d_cols))))
    return BlockPreconditioner(blocks)


def fit_hazard(rows: ZRows, at_risk: np.ndarray, y: np.ndarray, cweights: np.ndarray, *, lam: float, d_cols: int,
               per_horizon: bool, solver: SolverConfig, theta0: torch.Tensor | None = None
               ) -> tuple[HazardParams, SolveInfo, torch.Tensor]:
    """Minimise F (module docstring) over the rows the factory serves; cweights is 0 outside the training rows."""
    k = at_risk.shape[1]
    active_cells = at_risk & (cweights > 0)
    events = (y * active_cells).sum(axis=0)
    zero = events == 0
    cw_np = np.where(zero[None, :], 0.0, cweights)                    # event-free steps leave the objective
    total = float(cw_np.sum())
    if total <= 0:
        raise InvariantViolation("the hazard model has no person-period row with an event-bearing step")
    yt = torch.from_numpy(y)
    ct = torch.from_numpy(cw_np)
    n_par = d_cols + k + (k * d_cols if per_horizon else 0)

    def vg(theta: torch.Tensor) -> tuple[float, torch.Tensor]:
        w = theta[:d_cols]
        al = theta[d_cols:d_cols + k]
        dev = theta[d_cols + k:].view(k, d_cols) if per_horizon else None
        f = 0.0
        g = torch.zeros_like(theta)
        for pos, z in rows():
            zt = torch.from_numpy(z)
            pt = torch.from_numpy(pos)
            eta = (zt @ w)[:, None] + al[None, :]                          # [c, K]
            if dev is not None:
                eta = eta + zt @ dev.T
            cw = ct[pt]
            yy = yt[pt]
            f += float((cw * (log1pexp(eta) - yy * eta)).sum()) / total
            r = cw * (torch.sigmoid(eta) - yy) / total                     # [c, K]
            g[:d_cols] += zt.T @ r.sum(dim=1)
            g[d_cols:d_cols + k] += r.sum(dim=0)
            if dev is not None:
                g[d_cols + k:] += (zt.T @ r).T.reshape(-1)
        f += 0.5 * lam * float(w @ w)
        g[:d_cols] += lam * w
        if dev is not None:
            f += 0.5 * lam * float((dev * dev).sum())
            g[d_cols + k:] += lam * theta[d_cols + k:]
        return f, g

    if theta0 is None:
        th0 = torch.zeros(n_par, dtype=torch.float64)
        # intercepts start at the life-table hazards of the training rows (the optimum without features)
        n_j = active_cells.sum(axis=0)
        h0 = np.clip(np.divide(events, n_j, out=np.full(k, 0.5), where=n_j > 0), 1e-6, 1 - 1e-6)
        th0[d_cols:d_cols + k] = torch.from_numpy(np.where(zero, 0.0, np.log(h0 / (1 - h0))))
    else:
        th0 = theta0.clone()
    precond = (hazard_preconditioner(rows, cw_np, lam=lam, d_cols=d_cols, per_horizon=per_horizon, solver=solver)
               if solver.precondition and lam > 0 else None)
    theta, info = lbfgs_minimise(vg, th0, solver, method=solver.method, precond=precond)
    th = theta.numpy()
    params = HazardParams(w=th[:d_cols].copy(), alpha=th[d_cols:d_cols + k].copy(),
                          d=th[d_cols + k:].reshape(k, d_cols).copy() if per_horizon else np.zeros((k, d_cols)),
                          zero=zero.copy())
    return params, info, theta


@dataclass
class HazardLR:
    """A fitted discrete-time hazard LR (module docstring)."""

    cfg: HazardConfig
    columns: list[FeatureColumn]
    keep: np.ndarray
    standardiser: Standardiser
    params: HazardParams
    lam: float
    power: float
    calib: tuple[float, float]     # (a, b) on the hazard logit
    info: SolveInfo
    selection: SelectionResult | None
    window_seconds: float
    horizon_k: int
    threshold: ThresholdChoice | None = None
    notes: list[str] = field(default_factory=list)

    @classmethod
    def train(cls, corpus: LRCorpus, design: TriggerDesign, cfg: HazardConfig, std_cfg: StandardiserConfig,
              feat_cfg: FeatureConfig) -> HazardLR:
        k, w_s = corpus.horizon_k, corpus.window_seconds
        units = trigger_units(corpus, usable_only=True)
        cols = design.columns_for(cfg.lags)
        raw = RawRows.from_matrix(design.x[design_rows(design, units)][:, cols])
        columns = [design.columns[i] for i in cols.tolist()]
        ev = gather(corpus, units, "t_event_step").astype(np.int64)
        ob = gather(corpus, units, "t_observed_steps").astype(np.int64)
        at_risk, y = risk_sets(ev, ob, k)
        informative = at_risk.any(axis=1)
        tr = np.flatnonzero((units.role == "train") & informative)
        va = np.flatnonzero((units.role == "val") & informative)
        if tr.size == 0 or va.size == 0:
            raise InvariantViolation("the hazard model needs informative training and validation triggers")
        binary = np.asarray([c.binary for c in columns], dtype=bool)
        chunk = cfg.solver.chunk_rows
        per_h = cfg.coefficients == "per_horizon"

        def fit_on(pos: np.ndarray, std: Standardiser, keep: np.ndarray, lam: float, power: float,
                   theta0: torch.Tensor | None) -> tuple[HazardParams, SolveInfo, torch.Tensor]:
            sub_r = np.zeros_like(at_risk)
            sub_r[pos] = at_risk[pos]
            cw = person_period_weights(sub_r, y, power)
            return fit_hazard(solver_rows(raw, pos, std, keep, cfg.solver), sub_r, y, cw, lam=lam, d_cols=int(keep.size),
                              per_horizon=per_h, solver=cfg.solver, theta0=theta0)

        def fit_and_score(p_tr: np.ndarray, p_va: np.ndarray, grid: list[tuple[float, float]]) -> np.ndarray:
            std, keep = fit_standardiser(raw, p_tr, binary, std_cfg, feat_cfg, chunk)
            z_va = std.transform(raw.matrix[p_va], keep)  # type: ignore[index]
            out = np.full(len(grid), np.nan)
            warm: dict[float, torch.Tensor] = {}
            for g_i, (lam, power) in enumerate(grid):
                params, _info, theta = fit_on(p_tr, std, keep, lam, power, warm.get(power))
                warm[power] = theta
                h = 1.0 / (1.0 + np.exp(-params.eta(z_va)))
                out[g_i] = _score(cfg.criterion, h, ev[p_va], ob[p_va])
            return out

        t = gather(corpus, units, "t_time").astype(np.float64)
        end = np.minimum(t + k * w_s, gather(corpus, units, "t_horizon_end").astype(np.float64))
        selection = select_hyperparameters(cfg.selection, tr=tr, va=va, times=t, label_end=end,
                                           groups=groups_of(corpus, units, cfg.selection.group_by),
                                           fit_and_score=fit_and_score, criterion=cfg.criterion,
                                           lower_is_better=LOWER_IS_BETTER[cfg.criterion],
                                           embargo_seconds=cfg.selection.embargo_windows * w_s)
        lam, power = selection.chosen if selection is not None else (float(cfg.selection.fixed_l2 or 0.0),
                                                                      float(cfg.selection.fixed_power or 0.0))
        std, keep = fit_standardiser(raw, tr, binary, std_cfg, feat_cfg, chunk)
        params, info, _theta = fit_on(tr, std, keep, lam, power, None)
        model = cls(cfg=cfg, columns=columns, keep=keep, standardiser=std, params=params, lam=lam, power=power,
                    calib=(1.0, 0.0), info=info, selection=selection, window_seconds=w_s, horizon_k=k)
        if cfg.calibration == "platt":
            eta_va = params.eta(std.transform(raw.matrix[va], keep))  # type: ignore[index]
            model.calib = fit_hazard_calibration(eta_va, ev[va], ob[va], cfg.solver)
        # own alert threshold on P_inf(K), from the validation triggers whose outcome at K is known
        p_k = survival_from_hazard(model.hazard(raw.matrix[va]))[1][:, -1]  # type: ignore[index]
        happened = (ev[va] > 0) & (ev[va] <= k)
        known = happened | (ob[va] >= k)
        try:
            model.threshold = select_threshold(p_k[known], happened[known].astype(np.int64), cfg.threshold.rule,
                                               alpha=cfg.threshold.alpha)
        except InvariantViolation as exc:
            model.notes.append(f"no own alert threshold: {exc}")
        return model

    def hazard(self, raw_x: np.ndarray) -> np.ndarray:
        """Calibrated hazards [n, K] for raw trigger rows with this model's columns."""
        a, b = self.calib
        eta = self.params.eta(self.standardiser.transform(raw_x, self.keep))
        out = np.zeros(eta.shape)
        fin = np.isfinite(eta)
        out[fin] = 1.0 / (1.0 + np.exp(-(a * eta[fin] + b)))
        return out

    def predict(self, corpus: LRCorpus, design: TriggerDesign, roles: tuple[str, ...]) -> tuple[ForecastPredictions, TimeToEventPredictions]:
        """Forecast and time-to-event records for the usable triggers of the given roles."""
        units = trigger_units(corpus, roles, usable_only=True)
        pos = design_rows(design, units)
        cols = design.columns_for(self.cfg.lags)
        if [design.columns[i].name for i in cols.tolist()] != [c.name for c in self.columns]:
            raise InvariantViolation("the trigger design does not carry this model's columns")
        k = self.horizon_k
        h = self.hazard(design.x[pos][:, cols]) if pos.size else np.zeros((0, k))
        surv, p_inf = survival_from_hazard(h)
        meta = unit_meta(corpus, units, triggers=True)
        ev = gather(corpus, units, "t_event_step").astype(np.int64)
        ob = gather(corpus, units, "t_observed_steps").astype(np.int64)
        fc = ForecastPredictions(p_inf=p_inf, window_seconds=self.window_seconds, event_step=ev, observed_steps=ob,
                                 meta=meta, hazard=h,
                                 infiltrated_now=gather(corpus, units, "t_infiltrated_now").astype(bool),
                                 alert_threshold=self.threshold.value if self.threshold is not None else None)
        # minus the restricted mean time to infiltration within the horizon (AS-526), or P_inf(K)
        risk = (-self.window_seconds * (1.0 + surv[:, :-1].sum(axis=1)) if self.cfg.risk == "neg_rmst"
                else p_inf[:, -1].copy())
        tte = TimeToEventPredictions(risk=risk, survival=surv, time_grid=self.window_seconds * np.arange(1, k + 1),
                                     event_time=gather(corpus, units, "t_event_time").astype(np.float64),
                                     event_observed=gather(corpus, units, "t_event_observed").astype(bool), meta=meta.copy())
        return fc, tte

    def to_bundle(self, prefix: str) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
        arrays = {f"{prefix}.keep": self.keep, f"{prefix}.w": self.params.w, f"{prefix}.alpha": self.params.alpha,
                  f"{prefix}.d": self.params.d, f"{prefix}.zero": self.params.zero}
        arrays.update({f"{prefix}.std.{k}": v for k, v in self.standardiser.state().items()})
        header = {"config": config_to_dict(self.cfg), "columns": [c.as_dict() for c in self.columns], "lam": self.lam,
                  "power": self.power, "calib": list(self.calib), "info": self.info.as_dict(),
                  "selection": self.selection.as_dict() if self.selection is not None else None,
                  "window_seconds": self.window_seconds, "horizon_k": self.horizon_k,
                  "threshold": self.threshold.as_dict() if self.threshold is not None else None, "notes": list(self.notes)}
        return header, arrays

    @classmethod
    def from_bundle(cls, prefix: str, header: dict[str, Any], arrays: dict[str, np.ndarray]) -> HazardLR:
        std = Standardiser.from_state({k[len(prefix) + 5:]: v for k, v in arrays.items() if k.startswith(f"{prefix}.std.")})
        params = HazardParams(w=np.asarray(arrays[f"{prefix}.w"], dtype=np.float64),
                              alpha=np.asarray(arrays[f"{prefix}.alpha"], dtype=np.float64),
                              d=np.asarray(arrays[f"{prefix}.d"], dtype=np.float64),
                              zero=np.asarray(arrays[f"{prefix}.zero"], dtype=bool))
        return cls(cfg=from_dict(HazardConfig, header["config"], where="hazard"),
                   columns=[FeatureColumn.from_dict(c) for c in header["columns"]],
                   keep=np.asarray(arrays[f"{prefix}.keep"], dtype=np.int64), standardiser=std, params=params,
                   lam=float(header["lam"]), power=float(header["power"]),
                   calib=(float(header["calib"][0]), float(header["calib"][1])), info=SolveInfo(**header["info"]),
                   selection=selection_from_dict(header["selection"]), window_seconds=float(header["window_seconds"]),
                   horizon_k=int(header["horizon_k"]),
                   threshold=ThresholdChoice(**header["threshold"]) if header.get("threshold") else None,
                   notes=list(header.get("notes", [])))


def _score(criterion: str, h: np.ndarray, ev: np.ndarray, ob: np.ndarray) -> float:
    if criterion == "survival_nll":
        return survival_nll(h, ev, ob)
    if criterion == "brier_k":
        return brier_k(survival_from_hazard(h)[1], ev, ob)
    raise InvariantViolation(f"unknown hazard criterion {criterion!r}")


def fit_hazard_calibration(eta: np.ndarray, event_step: np.ndarray, observed_steps: np.ndarray,
                           solver: SolverConfig) -> tuple[float, float]:
    """(a, b) maximising the censored likelihood of sigmoid(a eta + b) on validation triggers (module docstring).

    Steps held at zero hazard (eta = -inf) stay at zero and do not enter the fit.
    """
    k = eta.shape[1]
    at_risk, y = risk_sets(event_step, observed_steps, k)
    fin = np.isfinite(eta)
    use = at_risk & fin
    if not (y[use] > 0).any() or not (y[use] < 1).any():
        raise InvariantViolation("hazard calibration needs event and event-free person-periods among the validation triggers")
    e = torch.from_numpy(np.where(fin, eta, 0.0))
    m = torch.from_numpy(use.astype(np.float64))
    yt = torch.from_numpy(y)
    n = float(use.sum())

    def vg(theta: torch.Tensor) -> tuple[float, torch.Tensor]:
        z = theta[0] * e + theta[1]
        f = float((m * (log1pexp(z) - yt * z)).sum()) / n
        r = m * (torch.sigmoid(z) - yt) / n
        return f, torch.stack([(r * e).sum(), r.sum()])

    theta, _info = lbfgs_minimise(vg, torch.tensor([1.0, 0.0], dtype=torch.float64), solver)
    return float(theta[0]), float(theta[1])


__all__ = ["HazardLR", "HazardParams", "fit_hazard", "fit_hazard_calibration", "hazard_preconditioner",
           "person_period_weights", "risk_sets", "survival_from_hazard"]
