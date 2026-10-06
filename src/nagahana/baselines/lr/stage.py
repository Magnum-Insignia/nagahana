"""Multinomial (softmax) stage LR over the ATT&CK stage vocabulary of models/vocab.py.

Targets (StageConfig.target)

    step     at every trigger, the stage of each future step k = 1 ... K: the furthest stage among the
             labelled updates of the step (the rule of forecaster.losses.step_labels, built by corpus.py),
             -1 where the step has no labelled update (AS-510). This is the per-step stage of NagaHana's
             Forecaster, so the comparison is like for like (AS-518).
    update   the stage of each labelled state update (the detection unit).

Model. With the present classes P (the stage codes that occur in the fit's training labels) and
standardised features z:

    logits_s(i, k) = z_i . W_s + B_{k, s}                  (shared slopes, a per-step intercept)
    logits_s(i, k) = z_i . (W_s + D_{k, s}) + B_{k, s}       (per_horizon: deviations shrunk towards W)
    p(s | i, k) = softmax_s(logits(i, k))

    F = (1 / S) sum_{i, k labelled} c_{y_ik} [-log p(y_ik | i, k)] + (lambda / 2) (||W||^2 + sum_k ||D_k||^2)

with class weights c_s = (N / (P N_s)) ** p (balanced for p = 1), the intercepts unpenalised and the
intercept of the most frequent class fixed at 0 (the softmax is invariant to a common shift of the
logits, so this removes the only flat direction of F). The gradient is dF/dlogits = c (p - onehot(y)) / S;
the solver is the L-BFGS driver of logistic.py in float64.

Classes without training labels. A class absent from a fit's training labels would have its intercept
driven to -infinity (the likelihood increases without bound as its probability goes to 0), so it is left
out of that fit's softmax and receives probability 0: the maximum-likelihood limit, with no smoothing
constant invented (AS-519). Every fit (each cross-validation fold and the final one) determines its own
present classes. The probabilities are reported over all 15 classes (StagePredictions). With the "step"
target, a step whose training labels lack a class present elsewhere shares the class's slopes and
intercept offsets of the other steps.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from nagahana.core.errors import InvariantViolation
from nagahana.evaluation.predictions import StagePredictions, make_meta
from nagahana.models.vocab import N_STAGES, STAGES

from .common import RawRows, ZRows, fit_standardiser, groups_of, solver_rows
from .config import FeatureConfig, SolverConfig, StageConfig, StandardiserConfig, config_to_dict, from_dict
from .corpus import LRCorpus
from .design import Contexts, TriggerDesign, UpdateDesign, design_rows, gather, trigger_units, unit_meta, update_units
from .features import FeatureColumn
from .logistic import SolveInfo, lbfgs_minimise
from .precondition import BlockPreconditioner, DiagonalPreconditioner, Preconditioner, SoftmaxPreconditioner, sample_mask
from .scoring import LOWER_IS_BETTER, multiclass_score
from .standardise import Standardiser
from .temporal_cv import SelectionResult, select_hyperparameters, selection_from_dict

STAGE_NAMES: tuple[str, ...] = tuple(name for name, _ in STAGES)


def softmax_rows(logits: np.ndarray) -> np.ndarray:
    """Row-wise softmax along the last axis, shifted by the maximum for stability."""
    z = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


@dataclass
class StageParams:
    """Fitted parameters over the present classes: W [D', P], B [K, P], D [K, D', P], classes [P]."""

    classes: np.ndarray
    W: np.ndarray
    B: np.ndarray
    D: np.ndarray

    def probs(self, z: np.ndarray) -> np.ndarray:
        """Probabilities [n, K, 15] over every stage class for standardised rows z [n, D']."""
        logits = (z @ self.W)[:, None, :] + self.B[None, :, :] + np.einsum("nd,kdp->nkp", z, self.D)
        out = np.zeros((z.shape[0], self.B.shape[0], N_STAGES))
        out[..., self.classes] = softmax_rows(logits)
        return out


def stage_preconditioner(rows: ZRows, cweights: np.ndarray, *, lam: float, d_cols: int, n_classes: int, ref: int,
                         per_horizon: bool, solver: SolverConfig, seed: int = 0) -> Preconditioner:
    """Boehning's bound of the softmax Hessian (precondition.py) on a seeded unit sample, block-diagonal over
    the slopes W, the step intercepts B and the per-step deviations D_k.

    With row weights r_i = sum_k c_ik and S the sampled weight: W uses (1/2)(I - 1 1^T / P) (x) G + lambda I
    with G = Z^T diag(r) Z / S; the intercept of step k and class s uses the diagonal (1/2)(1 - 1/P) g_k with
    g_k = sum_i c_ik / S (1 for the fixed reference class); D_k uses G_k = Z^T diag(c_:k) Z / S.
    """
    kk = cweights.shape[1]
    p = n_classes
    mask = None
    g = np.zeros((d_cols, d_cols))
    gk = np.zeros(kk)
    gdev = [np.zeros((d_cols, d_cols)) for _ in range(kk)] if per_horizon else []
    total = 0.0
    for pos, z in rows():
        if mask is None:
            mask = sample_mask(cweights.shape[0], solver.precond_rows, seed)
        keep = mask[pos]
        if not keep.any():
            continue
        zz = z[keep]
        cw = cweights[pos[keep]]
        r = cw.sum(axis=1)
        g += (zz * r[:, None]).T @ zz
        gk += cw.sum(axis=0)
        for j in range(len(gdev)):
            gdev[j] += (zz * cw[:, j][:, None]).T @ zz
        total += float(r.sum())
    if total <= 0:
        raise InvariantViolation("the stage preconditioning sample holds no labelled unit")
    n_w, n_b = d_cols * p, kk * p
    diag_b = np.repeat(0.5 * (1.0 - 1.0 / p) * gk / total, p).reshape(kk, p)
    diag_b[:, ref] = 1.0
    blocks: list[tuple[slice, Preconditioner]] = [
        (slice(0, n_w), SoftmaxPreconditioner.build(g / total, lam, p)),
        (slice(n_w, n_w + n_b), DiagonalPreconditioner.from_diagonal(np.where(diag_b > 0, diag_b, 1.0).reshape(-1))),
    ]
    for j in range(len(gdev)):
        a0 = n_w + n_b + j * n_w
        blocks.append((slice(a0, a0 + n_w), SoftmaxPreconditioner.build(gdev[j] / total, lam, p)))
    return BlockPreconditioner(blocks)


def fit_stage(rows: ZRows, labels: np.ndarray, positions: np.ndarray, *, power: float, lam: float, d_cols: int,
              per_horizon: bool, solver: SolverConfig, warm: StageParams | None = None) -> tuple[StageParams, SolveInfo]:
    """Minimise F (module docstring). labels int [N, K] of stage codes (-1 unknown) over all units;
    only `positions` are training rows."""
    lab_tr = labels[positions]
    present = np.unique(lab_tr[lab_tr >= 0])
    if present.size < 2:
        raise InvariantViolation("the stage model needs at least two stages in its training labels")
    index = np.full(N_STAGES, -1, dtype=np.int64)
    index[present] = np.arange(present.size)
    cls_lab = np.where(labels >= 0, index[np.maximum(labels, 0)], -1)              # -1 also for absent classes
    cls_lab_tr = cls_lab[positions]
    counts = np.bincount(cls_lab_tr[cls_lab_tr >= 0], minlength=present.size).astype(np.float64)
    ref = int(np.argmax(counts))
    n_lab = float(counts.sum())
    cw_cls = (n_lab / (present.size * counts)) ** power
    cweights = np.zeros(labels.shape)
    cweights[positions] = np.where(cls_lab_tr >= 0, cw_cls[np.maximum(cls_lab_tr, 0)], 0.0)
    kk, p = labels.shape[1], present.size
    total = float(cweights.sum())
    lab_t = torch.from_numpy(np.maximum(cls_lab, 0))
    cw_t = torch.from_numpy(cweights)
    n_w, n_b = d_cols * p, kk * p
    n_par = n_w + n_b + (kk * d_cols * p if per_horizon else 0)
    ref_mask = torch.ones(kk, p, dtype=torch.float64)
    ref_mask[:, ref] = 0.0

    def vg(theta: torch.Tensor) -> tuple[float, torch.Tensor]:
        w = theta[:n_w].view(d_cols, p)
        b = theta[n_w:n_w + n_b].view(kk, p) * ref_mask
        dev = theta[n_w + n_b:].view(kk, d_cols, p) if per_horizon else None
        f = 0.0
        g = torch.zeros_like(theta)
        for pos, z in rows():
            zt = torch.from_numpy(z)
            pt = torch.from_numpy(pos)
            logits = (zt @ w)[:, None, :] + b[None, :, :]                  # [c, K, P]
            if dev is not None:
                logits = logits + torch.einsum("cd,kdp->ckp", zt, dev)
            y = lab_t[pt]                                                  # [c, K]
            c = cw_t[pt]                                                   # 0 where unlabelled
            logp = torch.log_softmax(logits, dim=-1)
            f -= float((c * torch.gather(logp, -1, y[..., None]).squeeze(-1)).sum()) / total
            r = torch.exp(logp)
            r.scatter_add_(-1, y[..., None], -torch.ones_like(r[..., :1]))
            r = r * (c / total)[..., None]
            g[:n_w] += (zt.T @ r.sum(dim=1)).reshape(-1)
            g[n_w:n_w + n_b] += (r.sum(dim=0) * ref_mask).reshape(-1)
            if dev is not None:
                g[n_w + n_b:] += torch.einsum("cd,ckp->kdp", zt, r).reshape(-1)
        f += 0.5 * lam * float((w * w).sum())
        g[:n_w] += lam * theta[:n_w]
        if dev is not None:
            f += 0.5 * lam * float((dev * dev).sum())
            g[n_w + n_b:] += lam * theta[n_w + n_b:]
        return f, g

    th0 = torch.zeros(n_par, dtype=torch.float64)
    if warm is not None and np.array_equal(warm.classes, present) and warm.W.shape == (d_cols, p) and warm.B.shape == (kk, p):
        th0[:n_w] = torch.from_numpy(warm.W.reshape(-1))
        th0[n_w:n_w + n_b] = torch.from_numpy(warm.B.reshape(-1))
        if per_horizon:
            th0[n_w + n_b:] = torch.from_numpy(warm.D.reshape(-1))
    else:
        # intercepts start at the log class frequencies relative to the reference class
        th0[n_w:n_w + n_b] = torch.from_numpy(np.tile(np.log(counts / counts[ref]), kk))
    precond = (stage_preconditioner(rows, cweights, lam=lam, d_cols=d_cols, n_classes=p, ref=ref,
                                    per_horizon=per_horizon, solver=solver) if solver.precondition and lam > 0 else None)
    theta, info = lbfgs_minimise(vg, th0, solver, method=solver.method, precond=precond)
    th = theta.numpy()
    w = th[:n_w].reshape(d_cols, p).copy()
    b = th[n_w:n_w + n_b].reshape(kk, p).copy()
    b[:, ref] = 0.0
    dev = th[n_w + n_b:].reshape(kk, d_cols, p).copy() if per_horizon else np.zeros((kk, d_cols, p))
    return StageParams(classes=present.astype(np.int64), W=w, B=b, D=dev), info


@dataclass
class StageLR:
    """A fitted multinomial stage LR (module docstring)."""

    cfg: StageConfig
    columns: list[FeatureColumn]
    keep: np.ndarray
    standardiser: Standardiser
    params: StageParams
    lam: float
    power: float
    info: SolveInfo
    selection: SelectionResult | None
    horizon_k: int

    @property
    def classes(self) -> np.ndarray:
        return self.params.classes

    @classmethod
    def train(cls, corpus: LRCorpus, contexts: Contexts, design: TriggerDesign | None, cfg: StageConfig,
              std_cfg: StandardiserConfig, feat_cfg: FeatureConfig) -> StageLR:
        k = corpus.horizon_k
        if cfg.target == "step":
            if design is None:
                raise InvariantViolation("the step target needs the trigger design")
            units = trigger_units(corpus)
            cols = design.columns_for(cfg.lags)
            raw = RawRows.from_matrix(design.x[design_rows(design, units)][:, cols])
            columns = [design.columns[i] for i in cols.tolist()]
            labels = gather(corpus, units, "t_stage_step").astype(np.int64).reshape(len(units), k)
            times = gather(corpus, units, "t_time").astype(np.float64)
            label_end = times + k * corpus.window_seconds
        else:
            udesign = UpdateDesign(corpus, contexts, 0)
            units = update_units(corpus)
            columns = udesign.columns()
            labels = gather(corpus, units, "stage").astype(np.int64)[:, None]
            materialise = np.flatnonzero(np.isin(units.role, ["train", "val"]) & (labels[:, 0] >= 0))
            raw = RawRows.from_design(udesign, units, materialise=materialise if cfg.solver.method == "lbfgs" else None,
                                      chunk_rows=cfg.solver.chunk_rows)
            times = gather(corpus, units, "time").astype(np.float64)
            label_end = times
        labelled = (labels >= 0).any(axis=1)
        tr = np.flatnonzero((units.role == "train") & labelled)
        va = np.flatnonzero((units.role == "val") & labelled)
        if tr.size == 0 or va.size == 0:
            raise InvariantViolation("the stage model needs labelled training and validation units")
        binary = np.asarray([c.binary for c in columns], dtype=bool)
        chunk = cfg.solver.chunk_rows
        per_h = cfg.coefficients == "per_horizon" and cfg.target == "step"

        def fit_and_score(p_tr: np.ndarray, p_va: np.ndarray, grid: list[tuple[float, float]]) -> np.ndarray:
            std, keep = fit_standardiser(raw, p_tr, binary, std_cfg, feat_cfg, chunk)
            rows = solver_rows(raw, p_tr, std, keep, cfg.solver)
            out = np.full(len(grid), np.nan)
            warm: dict[float, StageParams] = {}
            for g_i, (lam, power) in enumerate(grid):
                params, _i = fit_stage(rows, labels, p_tr, power=power, lam=lam, d_cols=int(keep.size), per_horizon=per_h,
                                       solver=cfg.solver, warm=warm.get(power))
                warm[power] = params
                pr = _probs_at(params, raw, p_va, std, keep, chunk)         # [n, K, 15]
                lab = labels[p_va]
                ok = lab >= 0
                out[g_i] = multiclass_score(cfg.criterion, lab[ok], pr[ok])
            return out

        selection = select_hyperparameters(cfg.selection, tr=tr, va=va, times=times, label_end=label_end,
                                           groups=groups_of(corpus, units, cfg.selection.group_by),
                                           fit_and_score=fit_and_score, criterion=cfg.criterion,
                                           lower_is_better=LOWER_IS_BETTER[cfg.criterion],
                                           embargo_seconds=cfg.selection.embargo_windows * corpus.window_seconds)
        lam, power = selection.chosen if selection is not None else (float(cfg.selection.fixed_l2 or 0.0),
                                                                      float(cfg.selection.fixed_power or 0.0))
        std, keep = fit_standardiser(raw, tr, binary, std_cfg, feat_cfg, chunk)
        params, info = fit_stage(solver_rows(raw, tr, std, keep, cfg.solver), labels, tr, power=power, lam=lam,
                                 d_cols=int(keep.size), per_horizon=per_h, solver=cfg.solver)
        return cls(cfg=cfg, columns=columns, keep=keep, standardiser=std, params=params, lam=lam, power=power, info=info,
                   selection=selection, horizon_k=k)

    def probabilities(self, raw_x: np.ndarray) -> np.ndarray:
        """Probabilities [n, K or 1, 15] over all stage classes for raw rows."""
        return self.params.probs(self.standardiser.transform(raw_x, self.keep))

    def predict(self, corpus: LRCorpus, contexts: Contexts, design: TriggerDesign | None, roles: tuple[str, ...]) -> StagePredictions:
        """StagePredictions of every unit of the given roles (step units are (trigger, k) pairs, meta column `step`)."""
        k = self.horizon_k
        if self.cfg.target == "step":
            if design is None:
                raise InvariantViolation("the step target needs the trigger design")
            units = trigger_units(corpus, roles)
            cols = design.columns_for(self.cfg.lags)
            if [design.columns[i].name for i in cols.tolist()] != [c.name for c in self.columns]:
                raise InvariantViolation("the trigger design does not carry this model's columns")
            x = design.x[design_rows(design, units)][:, cols]
            probs = self.probabilities(x) if len(units) else np.zeros((0, k, N_STAGES))
            labels = gather(corpus, units, "t_stage_step").astype(np.int64).reshape(len(units), k)
            base = unit_meta(corpus, units, triggers=True)
            meta = base.loc[base.index.repeat(k)].reset_index(drop=True)
            meta["step"] = np.tile(np.arange(1, k + 1), len(units))
            return StagePredictions(probs=probs.reshape(-1, N_STAGES), label=labels.reshape(-1), stage_names=STAGE_NAMES,
                                    meta=make_meta(len(meta), **{c: meta[c].to_numpy() for c in meta.columns}))
        units = update_units(corpus, roles)
        udesign = UpdateDesign(corpus, contexts, 0)
        if [c.name for c in udesign.columns()] != [c.name for c in self.columns]:
            raise InvariantViolation("the corpus encoder does not produce this model's columns")
        out = np.zeros((len(units), N_STAGES))
        for i in np.unique(units.source):
            sel = np.flatnonzero(units.source == i)
            for a in range(0, sel.size, self.cfg.solver.chunk_rows):
                p = sel[a:a + self.cfg.solver.chunk_rows]
                out[p] = self.probabilities(udesign.chunk(int(i), units.row[p]))[:, 0, :]
        return StagePredictions(probs=out, label=gather(corpus, units, "stage").astype(np.int64), stage_names=STAGE_NAMES,
                                meta=unit_meta(corpus, units, triggers=False))

    def to_bundle(self, prefix: str) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
        arrays = {f"{prefix}.keep": self.keep, f"{prefix}.classes": self.params.classes, f"{prefix}.W": self.params.W,
                  f"{prefix}.B": self.params.B, f"{prefix}.D": self.params.D}
        arrays.update({f"{prefix}.std.{k}": v for k, v in self.standardiser.state().items()})
        header = {"config": config_to_dict(self.cfg), "columns": [c.as_dict() for c in self.columns], "lam": self.lam,
                  "power": self.power, "info": self.info.as_dict(),
                  "selection": self.selection.as_dict() if self.selection is not None else None, "horizon_k": self.horizon_k}
        return header, arrays

    @classmethod
    def from_bundle(cls, prefix: str, header: dict[str, Any], arrays: dict[str, np.ndarray]) -> StageLR:
        std = Standardiser.from_state({k[len(prefix) + 5:]: v for k, v in arrays.items() if k.startswith(f"{prefix}.std.")})
        params = StageParams(classes=np.asarray(arrays[f"{prefix}.classes"], dtype=np.int64),
                             W=np.asarray(arrays[f"{prefix}.W"], dtype=np.float64),
                             B=np.asarray(arrays[f"{prefix}.B"], dtype=np.float64),
                             D=np.asarray(arrays[f"{prefix}.D"], dtype=np.float64))
        return cls(cfg=from_dict(StageConfig, header["config"], where="stage"),
                   columns=[FeatureColumn.from_dict(c) for c in header["columns"]],
                   keep=np.asarray(arrays[f"{prefix}.keep"], dtype=np.int64), standardiser=std, params=params,
                   lam=float(header["lam"]), power=float(header["power"]), info=SolveInfo(**header["info"]),
                   selection=selection_from_dict(header["selection"]), horizon_k=int(header["horizon_k"]))


def _probs_at(params: StageParams, raw: RawRows, positions: np.ndarray, std: Standardiser, keep: np.ndarray,
              chunk: int) -> np.ndarray:
    """Probabilities [n, K, 15] of the given positions, computed chunk by chunk (memory bounded by `chunk`)."""
    positions = np.asarray(positions, dtype=np.int64)
    out = np.empty((positions.size, params.B.shape[0], N_STAGES))
    order = np.argsort(positions, kind="mergesort")
    sp = positions[order]
    for pos, x in raw.factory(positions)(chunk):
        out[order[np.searchsorted(sp, pos)]] = params.probs(std.transform(x, keep))
    return out


__all__ = ["STAGE_NAMES", "StageLR", "StageParams", "fit_stage", "softmax_rows", "stage_preconditioner"]
