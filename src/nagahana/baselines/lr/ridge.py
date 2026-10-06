"""Ridge next-state forecaster: the linear comparator for forecasts of the network state itself.

Targets. For a trigger at tau (window g = tau / w) and horizon h = 1 ... K, the state vector y_h of the
cadence window g + h of its source: the nine window states of features.py (counts log-transformed when
`log_count_states`, AS-509), standardised with the mean and standard deviation of the current-window
states of the training triggers (the evaluation chapter scores next-state forecasts in units standardised
on the training split). A target is observed when its window lies inside the trigger's label horizon and
inside the source's observed span; unobserved targets are masked. The units are the triggers whose own
window g lies inside the observed span of their source, so persistence is defined for each of them (AS-527).

Model (Hoerl and Kennard, "Ridge regression: biased estimation for nonorthogonal problems",
Technometrics 12(1), 1970). For horizon h and every group of state features observed on the same rows R,
with standardised inputs z (the trigger design at lags 0 ... L, "the same features and their lags"):

    (beta, b) = argmin sum_{i in R} ||y_ih - b - B^T z_i||^2 + lambda ||B||_F^2
    B = (Zc^T Zc + lambda I)^{-1} Zc^T Yc,    b = ybar - B^T zbar

with Zc and Yc centred over R (the intercept is unpenalised). The system is solved by a Cholesky
factorisation of Zc^T Zc + lambda I (torch.linalg.cholesky and torch.cholesky_solve, float64), once for
all outputs of the group. Because the design columns are ordered by lag, the Gram matrix of L lags is
the leading block of the Gram matrix of the largest lag count, so every (L, lambda) of the grid is scored
from one Gram matrix per fold and group.

Selection of (L, lambda) uses the folds of temporal_cv.py with the mean squared error over every observed
target of the validation units; inputs and targets are standardised inside each fold with its training
rows only.

Predictive variance. The record carries a Gaussian predictive variance per horizon and feature: the mean
squared residual of the final model on the validation triggers (held-out residuals; the training residuals
where a horizon and feature has no observed validation target), so the evaluation can score the linear
comparator with the Gaussian CRPS as well (Gneiting and Raftery, JASA 102(477), 2007, Section 4.2). It
also carries `current`, the standardised states of the window at the forecast origin, which the
persistence reference repeats.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from nagahana.core.errors import InvariantViolation
from nagahana.evaluation.predictions import StateForecastPredictions

from .common import RawRows, fit_standardiser, groups_of
from .config import FeatureConfig, RidgeConfig, SelectionConfig, StandardiserConfig, config_to_dict, from_dict
from .corpus import LRCorpus
from .design import TriggerDesign, Units, design_rows, gather, trigger_units, unit_meta
from .features import STATE_FEATURES, FeatureColumn, bin_of, transform_states
from .standardise import Standardiser
from .temporal_cv import SelectionResult, select_hyperparameters, selection_from_dict

FEATURE_NAMES: tuple[str, ...] = tuple(f"state.{s}" for s in STATE_FEATURES)


def state_units(corpus: LRCorpus, roles: tuple[str, ...] | None = None) -> Units:
    """Triggers whose own window lies inside their source's observed span (module docstring)."""
    units = trigger_units(corpus, roles)
    if not len(units):
        return units
    t = gather(corpus, units, "t_time").astype(np.float64)
    first = np.asarray([corpus.sources[int(i)].t_first for i in units.source])
    return units.select(t - corpus.window_seconds >= first)


def state_targets(corpus: LRCorpus, units: Units, feat_cfg: FeatureConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(current [n, F], future [n, K, F], observed mask [n, K, F]) of transformed window states."""
    k, w = corpus.horizon_k, corpus.window_seconds
    n, f = len(units), len(STATE_FEATURES)
    cur = np.full((n, f), np.nan)
    fut = np.full((n, k, f), np.nan)
    obs = np.zeros((n, k, f), dtype=bool)
    for i in np.unique(units.source):
        s = corpus.sources[int(i)]
        sel = np.flatnonzero(units.source == i)
        tau = np.asarray(s["t_time"], dtype=np.float64)[units.row[sel]]
        end = np.asarray(s["t_horizon_end"], dtype=np.float64)[units.row[sel]]
        g = bin_of(tau, w)
        c, _ok = s.states_at(g, np.full(sel.size, np.inf))
        cur[sel] = transform_states(c, feat_cfg)
        for h in range(1, k + 1):
            st, ok = s.states_at(g + h, end)
            tr = transform_states(st, feat_cfg)
            fut[sel, h - 1] = tr
            obs[sel, h - 1] = ok[:, None] & np.isfinite(tr)
    return cur, fut, obs


@dataclass
class StateScaler:
    """Per-feature mean and standard deviation of the training triggers' current states."""

    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, current: np.ndarray) -> StateScaler:
        ok = np.isfinite(current)
        cnt = ok.sum(axis=0)
        mean = np.divide(np.where(ok, current, 0.0).sum(axis=0), cnt, out=np.zeros(current.shape[1]), where=cnt > 0)
        var = np.divide(np.where(ok, (current - mean) ** 2, 0.0).sum(axis=0), cnt, out=np.zeros(current.shape[1]), where=cnt > 0)
        scale = np.sqrt(var)
        return cls(mean=mean, scale=np.where(scale > 1e-12, scale, 1.0))

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.scale


def _groups(mask_h: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Feature groups with identical row masks: [(feature indices, row mask)]."""
    out: list[tuple[np.ndarray, np.ndarray]] = []
    seen: dict[bytes, int] = {}
    for f in range(mask_h.shape[1]):
        key = np.packbits(mask_h[:, f]).tobytes() + mask_h.shape[0].to_bytes(8, "little")
        if key in seen:
            idx, m = out[seen[key]]
            out[seen[key]] = (np.r_[idx, f], m)
        else:
            seen[key] = len(out)
            out.append((np.asarray([f]), mask_h[:, f].copy()))
    return out


def ridge_solve(z: np.ndarray, y: np.ndarray, lams: list[float]) -> list[tuple[np.ndarray, np.ndarray]]:
    """Closed-form ridge with an unpenalised intercept for each lambda: [(B [D, F], b [F])] (module docstring)."""
    zbar = z.mean(axis=0)
    ybar = y.mean(axis=0)
    zt = torch.from_numpy(z - zbar)
    yt = torch.from_numpy(y - ybar)
    gram = zt.T @ zt
    rhs = zt.T @ yt
    eye = torch.eye(gram.shape[0], dtype=torch.float64)
    out = []
    for lam in lams:
        if lam <= 0:
            raise InvariantViolation("ridge needs lambda > 0")
        chol = torch.linalg.cholesky(gram + lam * eye)
        beta = torch.cholesky_solve(rhs, chol).numpy()
        out.append((beta, ybar - zbar @ beta))
    return out


@dataclass
class RidgeForecaster:
    """A fitted ridge next-state forecaster (module docstring)."""

    cfg: RidgeConfig
    columns: list[FeatureColumn]
    keep: np.ndarray
    standardiser: Standardiser
    scaler: StateScaler
    lags: int
    lam: float
    coef: np.ndarray            # [D', K, F]
    intercept: np.ndarray       # [K, F]
    selection: SelectionResult | None
    horizon_k: int
    feat_cfg: FeatureConfig
    variance: np.ndarray | None = None   # [K, F] predictive variance (standardised units)

    @classmethod
    def train(cls, corpus: LRCorpus, design: TriggerDesign, cfg: RidgeConfig, std_cfg: StandardiserConfig,
              feat_cfg: FeatureConfig) -> RidgeForecaster:
        k = corpus.horizon_k
        units = state_units(corpus)
        l_max = max(cfg.lags)
        cols = design.columns_for(l_max)
        raw = RawRows.from_matrix(design.x[design_rows(design, units)][:, cols])
        all_columns = [design.columns[i] for i in cols.tolist()]
        binary = np.asarray([c.binary for c in all_columns], dtype=bool)
        cur, fut, obs = state_targets(corpus, units, feat_cfg)
        has = obs.reshape(len(units), -1).any(axis=1)
        tr = np.flatnonzero((units.role == "train") & has)
        va = np.flatnonzero((units.role == "val") & has)
        if tr.size == 0:
            raise InvariantViolation("the ridge forecaster needs training triggers with observed future states")
        width = design.block_width
        chunk = 1 << 20

        def fit_grid(p_tr: np.ndarray, lags: list[int], lams: list[float]) -> dict[tuple[int, float], Any]:
            std, keep = fit_standardiser(raw, p_tr, binary, std_cfg, feat_cfg, chunk)
            scaler = StateScaler.fit(cur[p_tr])
            z = std.transform(raw.matrix[p_tr], keep) if raw.matrix is not None else None
            assert z is not None
            y = scaler.transform(fut[p_tr])
            res: dict[tuple[int, float], Any] = {}
            for lag in lags:
                d_l = int(np.searchsorted(keep, (lag + 1) * width))       # kept columns of lags <= lag (prefix)
                coef = np.zeros((len(lams), d_l, k, fut.shape[2]))
                icpt = np.zeros((len(lams), k, fut.shape[2]))
                for h in range(k):
                    for feats, rows in _groups(obs[p_tr, h]):
                        if not rows.any():
                            continue
                        sols = ridge_solve(z[rows][:, :d_l], y[rows][:, h, :][:, feats], lams)
                        for li, (beta, b) in enumerate(sols):
                            coef[li][:, h, feats] = beta
                            icpt[li][h, feats] = b
                for li, lam in enumerate(lams):
                    res[(lag, lam)] = (std, keep[:d_l], scaler, coef[li], icpt[li])
            return res

        sel_cfg: SelectionConfig = cfg.selection
        grid = [(float(lam), float(lag)) for lag in cfg.lags for lam in cfg.l2.values()]

        def fit_and_score(p_tr: np.ndarray, p_va: np.ndarray, grid_: list[tuple[float, float]]) -> np.ndarray:
            lags = sorted({int(g[1]) for g in grid_})
            lams = sorted({float(g[0]) for g in grid_}, reverse=True)
            fits = fit_grid(p_tr, lags, lams)
            out = np.full(len(grid_), np.nan)
            for g_i, (lam, lag) in enumerate(grid_):
                std, keep, scaler, coef, icpt = fits[(int(lag), float(lam))]
                pred = _predict(std.transform(raw.matrix[p_va], keep), coef, icpt)  # type: ignore[index]
                y = scaler.transform(fut[p_va])
                m = obs[p_va]
                out[g_i] = float(np.mean((pred[m] - y[m]) ** 2)) if m.any() else np.nan
            return out

        t = gather(corpus, units, "t_time").astype(np.float64)
        end = t + k * corpus.window_seconds
        selection = select_hyperparameters(sel_cfg, tr=tr, va=va, times=t, label_end=end,
                                           groups=groups_of(corpus, units, sel_cfg.group_by), fit_and_score=fit_and_score,
                                           criterion="mse", lower_is_better=True,
                                           embargo_seconds=sel_cfg.embargo_windows * corpus.window_seconds, grid=grid)
        if selection is not None:
            lam, lag_f = selection.chosen
        else:
            # method "fixed": fixed_l2 and the largest lag count of `lags` (give one lag count to fix it)
            if sel_cfg.fixed_l2 is None or sel_cfg.fixed_l2 <= 0:
                raise InvariantViolation("ridge selection 'fixed' needs fixed_l2 > 0")
            lam, lag_f = float(sel_cfg.fixed_l2), float(max(cfg.lags))
        lag = int(lag_f)
        std, keep, scaler, coef, icpt = fit_grid(tr, [lag], [float(lam)])[(lag, float(lam))]
        columns = [all_columns[i] for i in range((lag + 1) * width)]
        model = cls(cfg=cfg, columns=columns, keep=keep, standardiser=_restrict(std, (lag + 1) * width), scaler=scaler,
                    lags=lag, lam=float(lam), coef=coef, intercept=icpt, selection=selection, horizon_k=k, feat_cfg=feat_cfg)
        # predictive variance per (horizon, feature): held-out residuals, training residuals where none exist
        sel = (lag + 1) * width
        res_va = model.predict_raw(raw.matrix[va][:, :sel]) - scaler.transform(fut[va]) if va.size else None  # type: ignore[index]
        res_tr = model.predict_raw(raw.matrix[tr][:, :sel]) - scaler.transform(fut[tr])  # type: ignore[index]
        model.variance = _residual_variance(res_tr, obs[tr], res_va, obs[va] if va.size else None)
        return model

    def predict_raw(self, raw_x: np.ndarray) -> np.ndarray:
        """Standardised forecasts [n, K, F] for raw trigger rows with this model's columns."""
        return _predict(self.standardiser.transform(raw_x, self.keep), self.coef, self.intercept)

    def predict(self, corpus: LRCorpus, design: TriggerDesign, roles: tuple[str, ...]) -> StateForecastPredictions:
        units = state_units(corpus, roles)
        cols = design.columns_for(self.lags)
        if [design.columns[i].name for i in cols.tolist()] != [c.name for c in self.columns]:
            raise InvariantViolation("the trigger design does not carry this model's columns")
        k, f = self.horizon_k, len(STATE_FEATURES)
        x = design.x[design_rows(design, units)][:, cols]
        pred = self.predict_raw(x) if len(units) else np.zeros((0, k, f))
        cur, fut, obs = state_targets(corpus, units, self.feat_cfg)
        cur_z = self.scaler.transform(cur)
        cur_ok = np.isfinite(cur_z)
        var = (np.broadcast_to(self.variance[None], pred.shape).copy() if self.variance is not None else None)
        return StateForecastPredictions(predicted=pred, observed=np.where(obs, self.scaler.transform(fut), 0.0), mask=obs,
                                        horizons=np.arange(1, k + 1), feature_names=FEATURE_NAMES,
                                        meta=unit_meta(corpus, units, triggers=True), variance=var,
                                        current=np.where(cur_ok, cur_z, 0.0), current_mask=cur_ok)

    def to_bundle(self, prefix: str) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
        arrays = {f"{prefix}.keep": self.keep, f"{prefix}.coef": self.coef, f"{prefix}.intercept": self.intercept,
                  f"{prefix}.scaler.mean": self.scaler.mean, f"{prefix}.scaler.scale": self.scaler.scale}
        if self.variance is not None:
            arrays[f"{prefix}.variance"] = self.variance
        arrays.update({f"{prefix}.std.{k}": v for k, v in self.standardiser.state().items()})
        header = {"config": config_to_dict(self.cfg), "features": config_to_dict(self.feat_cfg),
                  "columns": [c.as_dict() for c in self.columns], "lags": self.lags, "lam": self.lam,
                  "selection": self.selection.as_dict() if self.selection is not None else None, "horizon_k": self.horizon_k}
        return header, arrays

    @classmethod
    def from_bundle(cls, prefix: str, header: dict[str, Any], arrays: dict[str, np.ndarray]) -> RidgeForecaster:
        std = Standardiser.from_state({k[len(prefix) + 5:]: v for k, v in arrays.items() if k.startswith(f"{prefix}.std.")})
        return cls(cfg=from_dict(RidgeConfig, header["config"], where="ridge"),
                   columns=[FeatureColumn.from_dict(c) for c in header["columns"]],
                   keep=np.asarray(arrays[f"{prefix}.keep"], dtype=np.int64), standardiser=std,
                   scaler=StateScaler(np.asarray(arrays[f"{prefix}.scaler.mean"], dtype=np.float64),
                                      np.asarray(arrays[f"{prefix}.scaler.scale"], dtype=np.float64)),
                   lags=int(header["lags"]), lam=float(header["lam"]),
                   coef=np.asarray(arrays[f"{prefix}.coef"], dtype=np.float64),
                   intercept=np.asarray(arrays[f"{prefix}.intercept"], dtype=np.float64),
                   selection=selection_from_dict(header["selection"]), horizon_k=int(header["horizon_k"]),
                   feat_cfg=from_dict(FeatureConfig, header["features"], where="features"),
                   variance=(np.asarray(arrays[f"{prefix}.variance"], dtype=np.float64)
                             if f"{prefix}.variance" in arrays else None))


def _residual_variance(res_tr: np.ndarray, obs_tr: np.ndarray, res_va: np.ndarray | None,
                       obs_va: np.ndarray | None) -> np.ndarray:
    """Mean squared residual [K, F] over observed targets: validation where any, else training, else 1."""
    def msr(res: np.ndarray, obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        cnt = obs.sum(axis=0)
        ss = np.where(obs, res ** 2, 0.0).sum(axis=0)
        return np.divide(ss, cnt, out=np.full(cnt.shape, np.nan), where=cnt > 0), cnt

    v_tr, _c = msr(res_tr, obs_tr)
    if res_va is not None and obs_va is not None:
        v_va, c_va = msr(res_va, obs_va)
        out = np.where(c_va > 0, v_va, v_tr)
    else:
        out = v_tr
    return np.where(np.isfinite(out), out, 1.0)


def _predict(z: np.ndarray, coef: np.ndarray, icpt: np.ndarray) -> np.ndarray:
    return np.einsum("nd,dkf->nkf", z, coef) + icpt[None]


def _restrict(std: Standardiser, n_cols: int) -> Standardiser:
    """The standardiser of the first n_cols columns (the lags kept by the selection)."""
    return Standardiser(location=std.location[:n_cols], scale=std.scale[:n_cols], kind=std.kind[:n_cols],
                        constant=std.constant[:n_cols], fallback=std.fallback[:n_cols], n_valid=std.n_valid[:n_cols],
                        n_rows=std.n_rows, mean=std.mean[:n_cols])


__all__ = ["FEATURE_NAMES", "RidgeForecaster", "StateScaler", "ridge_solve", "state_targets", "state_units"]
