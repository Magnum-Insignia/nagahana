"""How well one score separates two classes: AUROC with DeLong intervals, out-of-fold category scores.

AUROC
-----
AUC = P(S+ > S-) + 0.5 P(S+ = S-) for a random positive and a random negative, computed from mid-ranks
(the Mann-Whitney statistic): AUC = (R+ - n+ (n+ + 1) / 2) / (n+ n-), R+ the rank sum of the positives.

Interval (DeLong, DeLong and Clarke-Pearson, Biometrics 44(3):837-845, 1988), with the mid-rank
computation of the placement values of Sun and Xu (IEEE Signal Processing Letters 21(11):1389-1393, 2014):
    V10_i = (R_i - R+_i) / n-        for each positive i  (R+_i: its mid-rank among the positives)
    V01_j = 1 - (R_j - R-_j) / n+    for each negative j
    Var(AUC) = s10^2 / n+ + s01^2 / n-  (sample variances of V10 and V01).
The interval is AUC +- z sqrt(Var), clipped to [0, 1].

Separability
------------
A single feature may separate the classes with either orientation; `separability` = max(AUC, 1 - AUC)
and `direction` says which values point to the positive class.

Out-of-fold category scores
---------------------------
A categorical field (a port, a protocol) is scored by the positive rate of its category. Computing
the rate on the rows being scored is optimistic, so the rate of row i is estimated on the other
folds only (K-fold cross-fitting) with additive smoothing towards the global rate:
    score_i = (pos_c + a * prior) / (count_c + a), counts from the folds that do not contain i.
Folds are stratified by label (positives and negatives are dealt round-robin after a seeded shuffle).
The AUROC of such scores is computed inside each fold and averaged (`out_of_fold_auroc`): within a fold
the scores do not depend on that fold's labels, so each fold AUROC is unbiased for an uninformative
feature (0.5), whereas pooling the folds is not (a category over-represented among a fold's positives
gets a lower training-part rate, which anti-correlates the pooled scores with the labels; Airola,
Pahikkala, Waegeman, De Baets and Salakoski, Computational Statistics and Data Analysis 55(4):1828-1844,
2011, compare pooled and averaged cross-validated AUC). The interval of the average treats the fold
estimates as independent: Var = sum_f Var_f / K^2 with the DeLong variances Var_f. The in-sample rate
(optimistic; the area under the ROC hull of the feature's categories) is `in_sample_rate`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import special, stats


@dataclass(frozen=True)
class AUC:
    """AUROC of a score for a binary label (module docstring)."""

    auc: float
    lower: float
    upper: float
    se: float
    separability: float
    direction: str
    n_pos: int
    n_neg: int


def _binary(scores: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    y = np.asarray(labels, dtype=np.float64).reshape(-1)
    if s.shape != y.shape:
        raise ValueError("scores and labels must have the same length")
    ok = np.isfinite(s) & np.isin(y, (0.0, 1.0))
    return s[ok], y[ok].astype(bool)


def auroc(scores: np.ndarray, labels: np.ndarray, *, level: float = 0.95) -> AUC:
    """AUROC with its DeLong interval; rows with a non-finite score or an unknown label are left out."""
    s, y = _binary(scores, labels)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        nan = float("nan")
        return AUC(nan, nan, nan, nan, nan, "", n_pos, n_neg)
    r = stats.rankdata(s)                                             # mid-ranks over all rows
    auc = float((r[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))
    v10 = (r[y] - stats.rankdata(s[y])) / n_neg
    v01 = 1.0 - (r[~y] - stats.rankdata(s[~y])) / n_pos
    var = (v10.var(ddof=1) if n_pos > 1 else 0.0) / n_pos + (v01.var(ddof=1) if n_neg > 1 else 0.0) / n_neg
    se = float(np.sqrt(max(var, 0.0)))
    z = float(special.ndtri(0.5 + level / 2.0))
    lo, hi = max(0.0, auc - z * se), min(1.0, auc + z * se)
    return AUC(auc, lo, hi, se, max(auc, 1.0 - auc), "higher" if auc >= 0.5 else "lower", n_pos, n_neg)


def _oof(categories: np.ndarray, labels: np.ndarray, folds: int, smoothing: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """(cross-fitted rates [n], fold of each row [n], -1 where the label is unknown)."""
    if folds < 2:
        raise ValueError("folds must be >= 2")
    y = np.asarray(labels, dtype=np.float64).reshape(-1)
    from nagahana.analytics.information import factorize

    codes, k = factorize(np.asarray(categories))
    known = np.isin(y, (0.0, 1.0))
    out = np.full(y.shape[0], np.nan)
    fold_all = np.full(y.shape[0], -1, dtype=np.int64)
    idx = np.flatnonzero(known)
    if idx.size == 0:
        return out, fold_all
    rng = np.random.default_rng(seed)
    fold = np.empty(idx.size, dtype=np.int64)
    for cls in (0.0, 1.0):                                               # stratified: each class dealt round-robin
        members = np.flatnonzero(y[idx] == cls)
        members = members[rng.permutation(members.size)]
        fold[members] = np.arange(members.size) % folds
    fold_all[idx] = fold
    prior = float(y[idx].mean())
    for f in range(folds):
        train, test = idx[fold != f], idx[fold == f]
        if test.size == 0:
            continue
        pos = np.bincount(codes[train], weights=y[train], minlength=k)
        cnt = np.bincount(codes[train], minlength=k).astype(np.float64)
        out[test] = (pos[codes[test]] + smoothing * prior) / (cnt[codes[test]] + smoothing)
    return out, fold_all


def out_of_fold_rate(categories: np.ndarray, labels: np.ndarray, *, folds: int = 5, smoothing: float = 1.0,
                     seed: int = 0) -> np.ndarray:
    """Cross-fitted positive rate of each row's category (module docstring); NaN where the label is unknown.

    categories: any hashable values (NaN is a category of its own); labels: 1 / 0 / NaN.
    """
    return _oof(categories, labels, folds, smoothing, seed)[0]


def out_of_fold_auroc(categories: np.ndarray, labels: np.ndarray, *, folds: int = 5, smoothing: float = 1.0,
                      seed: int = 0, level: float = 0.95) -> AUC:
    """Fold-averaged AUROC of the cross-fitted category rate, with its interval (module docstring)."""
    y = np.asarray(labels, dtype=np.float64).reshape(-1)
    rate, fold = _oof(categories, y, folds, smoothing, seed)
    aucs, variances, n_pos, n_neg = [], [], 0, 0
    for f in range(folds):
        m = fold == f
        a = auroc(rate[m], y[m])
        n_pos += a.n_pos
        n_neg += a.n_neg
        if np.isfinite(a.auc):
            aucs.append(a.auc)
            variances.append(a.se ** 2)
    if not aucs:
        nan = float("nan")
        return AUC(nan, nan, nan, nan, nan, "", n_pos, n_neg)
    mean = float(np.mean(aucs))
    se = float(np.sqrt(np.sum(variances)) / len(aucs))
    z = float(special.ndtri(0.5 + level / 2.0))
    return AUC(mean, max(0.0, mean - z * se), min(1.0, mean + z * se), se, max(mean, 1.0 - mean),
               "higher" if mean >= 0.5 else "lower", n_pos, n_neg)


def in_sample_rate(categories: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Positive rate of each row's category computed on all known rows (optimistic; module docstring)."""
    from nagahana.analytics.information import factorize

    y = np.asarray(labels, dtype=np.float64).reshape(-1)
    codes, k = factorize(np.asarray(categories))
    known = np.isin(y, (0.0, 1.0))
    pos = np.bincount(codes[known], weights=y[known], minlength=k)
    cnt = np.bincount(codes[known], minlength=k).astype(np.float64)
    rate = np.where(cnt > 0, pos / np.maximum(cnt, 1.0), np.nan)
    return np.where(known, rate[codes], np.nan)


__all__ = ["AUC", "auroc", "in_sample_rate", "out_of_fold_auroc", "out_of_fold_rate"]
