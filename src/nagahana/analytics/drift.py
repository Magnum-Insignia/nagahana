"""Drift: covariate shift, label shift and streaming concept-drift detection.

Kernel two-sample test (Gretton, Borgwardt, Rasch, Schoelkopf and Smola, JMLR 13:723-773, 2012)
-------------------------------------------------------------------------------------------
Unbiased MMD^2_u = sum_{i!=j} k(x_i, x_j) / (n(n-1)) + sum_{i!=j} k(y_i, y_j) / (m(m-1)) - 2 sum_ij k(x_i, y_j) / (nm).
Status-aware mixed kernel on records with absent cells (D-41). Numeric fields (continuous, count,
histogram bins) are put on the slog1p scale (AS-31) and divided by the pooled MAD of their column;
categorical and bitmask fields (ports, protocol, flags) are codes and are compared by equality only
(a port is not close to another port, datamodel/fields.py). Absent cells are compared only through
the status pattern:
    k(a, b) = [pattern(a) = pattern(b)] exp(-||z(a) - z(b)||^2 / (2 s^2)) exp(-H(a, b) / h0),
with z the scaled numeric values (absent cells set to 0, harmless inside an equal pattern) and H the
number of categorical fields whose codes differ. The pattern indicator sum_p [a in p][b in p] is
positive semi-definite, the Gaussian factor positive definite, and exp(-H / h0) is the product over
categorical fields of exp(-[a_c != b_c] / h0) = e^(-1/h0) + (1 - e^(-1/h0)) [a_c = b_c], each a
non-negative combination of a constant and a delta kernel; so k is a valid kernel (Schur product
theorem). s^2 and h0 are the medians of ||z(a) - z(b)||^2 and of H over pooled pairs with equal
patterns (the median heuristic; h0 at least 1). Records observed through different sensors
(different status patterns) are different distributions, as they are.
Permutation test: the pooled kernel matrix is computed once; for each of B seeded relabellings the
three block sums are z'Kz, z'K1 and 1'K1 with z the indicator of the first sample (one matrix product
per batch of relabellings); p = (1 + #{MMD^2_perm >= MMD^2_obs}) / (1 + B), exact in finite samples
under exchangeability.

Per-feature tests and multiplicity
----------------------------------
Numeric columns: two-sample Kolmogorov-Smirnov on contributing values (scipy.stats.ks_2samp).
Categorical and bitmask columns: chi-square test of homogeneity of the code frequencies (the most
frequent codes plus "other"). Status: chi-square test of homogeneity of the five statuses. The
family of all tests is corrected by Benjamini-Hochberg (JRSS B 57(1):289-300, 1995):
q_(i) = min_{j >= i} min(1, m p_(j) / j), or Benjamini-Yekutieli (Annals of Statistics 29(4):1165-1188,
2001), which multiplies by sum_{i<=m} 1/i and holds under any dependence.

Population stability index
--------------------------
PSI = sum_b (q_b - p_b) log(q_b / p_b) over the reference deciles of a column plus one bin for absent
cells, with empty-bin shares floored at epsilon. PSI is the symmetrised Kullback-Leibler (Jeffreys)
divergence KL(q || p) + KL(p || q) of the binned distributions. The customary reading thresholds 0.1
and 0.25 come from credit-scoring practice (Siddiqi, "Credit Risk Scorecards", Wiley 2006; citation to
verify) and are reported as such, never as a test.

Label shift: black-box shift estimation (Lipton, Wang and Smola, ICML 2018, arXiv:1802.03916)
------------------------------------------------------------------------------------------
Under label shift p(x | y) is fixed and only p(y) moves. With a fixed classifier f, the joint
confusion C[i, j] = P_s(f(x) = i, y = j) on held-out source data and the target prediction rates
mu[i] = P_t(f(x) = i) satisfy mu = C w with w_j = q(y = j) / p(y = j). w is solved by least squares
(Tikhonov-regularised by `l2` when C is ill-conditioned), clipped at 0, and q = w * p renormalised.
The condition number of C says whether the shift is identifiable from f. A chi-square test of the
predicted-label frequencies (source held-out against target) tests for any shift seen through f, as
Lipton et al. propose. The black box used here, when no predictions are supplied, is a multinomial
logistic regression fitted in-house on the source training part (L2, L-BFGS).

Streaming detectors
-------------------
ADWIN (Bifet and Gavalda, SIAM SDM 2007): the window W is kept as an exponential histogram (at most
`max_buckets` buckets per size 2^i; each bucket holds count, total and squared deviation, merged by
Chan's formula); after each insertion every split W = W0 W1 at a bucket boundary is tested with
    eps_cut = sqrt((2 / m) var_W ln(2 / delta')) + (2 / (3 m)) ln(2 / delta'),   1/m = 1/n0 + 1/n1,
and the oldest bucket is dropped while some split exceeds it (a change). delta' = delta / |W| is the
union bound over all cut points (conservative: only bucket boundaries are tested). Values must lie in
[0, 1]; a series is rescaled by its observed range first.
Page-Hinkley (Page, Biometrika 41(1/2):100-115, 1954): m_T = sum_{t<=T} (x_t - xbar_T - delta),
alarm when m_T - min_{t<=T} m_t > lambda (an increase; the mirrored statistic detects a decrease);
statistics restart after an alarm.
DDM (Gama, Medas, Castillo and Rodrigues, SBIA 2004, LNCS 3171:286-295): for a 0/1 error stream,
p_i the error rate and s_i = sqrt(p_i (1 - p_i) / i); after `min_samples`, with (p_min, s_min) at the
smallest p + s, warning when p_i + s_i > p_min + 2 s_min, drift when > p_min + 3 s_min (then reset).
The inequalities are strict so that an error-free start (p = s = 0, hence s_min = 0) cannot raise an
alarm by equality.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy import optimize, special, stats

from nagahana.analytics.config import DriftConfig, to_dict
from nagahana.analytics.dependence import slog1p
from nagahana.analytics.information import factorize
from nagahana.analytics.report import Figure, Report
from nagahana.analytics.robust import MAD_NORMAL

if TYPE_CHECKING:                                                        # the data model is needed by corpus inputs only
    from nagahana.analytics.corpus import Corpus


@dataclass(frozen=True)
class MMDResult:
    """Kernel two-sample test result (module docstring)."""

    mmd2: float
    p_value: float
    bandwidth2: float
    n: int
    m: int
    permutations: int
    null_mean: float
    null_sd: float


def _status_kernel(values: np.ndarray, status: np.ndarray, numeric: np.ndarray | None = None) -> tuple[np.ndarray, float]:
    """(pooled kernel matrix [N, N], s^2) of the status-aware mixed kernel (module docstring)."""
    raw = np.asarray(values, dtype=np.float64)
    num = np.ones(raw.shape[1], dtype=bool) if numeric is None else np.asarray(numeric, dtype=bool)
    n_rows = raw.shape[0]
    # Numeric part: slog1p values scaled by their pooled MAD (standard deviation, then 1, when the MAD is 0).
    v = slog1p(raw[:, num])
    contrib = np.isfinite(v)
    med = np.array([np.median(v[contrib[:, j], j]) if contrib[:, j].any() else 0.0 for j in range(v.shape[1])])
    mad = np.array([np.median(np.abs(v[contrib[:, j], j] - med[j])) * MAD_NORMAL if contrib[:, j].any() else 1.0
                    for j in range(v.shape[1])])
    sd = np.array([v[contrib[:, j], j].std() if contrib[:, j].any() else 1.0 for j in range(v.shape[1])])
    scale = np.where(mad > 0, mad, np.where(sd > 0, sd, 1.0))
    z = np.where(contrib, (v - med) / scale, 0.0)
    sq = (z * z).sum(axis=1)
    d2 = np.maximum(sq[:, None] + sq[None, :] - 2.0 * (z @ z.T), 0.0)    # [N, N]
    # Categorical part: number of fields whose codes differ (absent cells meet only inside equal patterns).
    ham = np.zeros((n_rows, n_rows))
    for j in np.flatnonzero(~num).tolist():
        codes, _ = factorize(np.where(np.isfinite(raw[:, j]), raw[:, j], -np.inf))
        ham += codes[:, None] != codes[None, :]
    pat, _ = factorize(*[np.asarray(status)[:, j] for j in range(status.shape[1])]) if status.shape[1] else (
        np.zeros(n_rows, dtype=np.int64), 1)
    same = pat[:, None] == pat[None, :]
    iu = np.triu_indices(n_rows, 1)
    cand = d2[iu][same[iu]]
    s2 = float(np.median(cand[cand > 0])) if (cand > 0).any() else 1.0
    hcand = ham[iu][same[iu]]
    h0 = max(float(np.median(hcand)) if hcand.size else 1.0, 1.0)
    return np.exp(-d2 / (2.0 * s2) - ham / h0) * same, s2


def _mmd2_from_blocks(k: np.ndarray, z: np.ndarray, n: int, m: int) -> np.ndarray:
    """MMD^2_u for membership vectors z [B, N] (1 = first sample) from the pooled kernel matrix."""
    kz = z @ k                                                           # [B, N]
    a = (kz * z).sum(axis=1)                                             # z'Kz
    zk1 = kz.sum(axis=1)                                                 # z'K1
    total = k.sum()
    d = np.diag(k)
    dx = z @ d
    dy = d.sum() - dx
    bxy = zk1 - a
    c = total - 2.0 * zk1 + a
    return (a - dx) / (n * (n - 1)) + (c - dy) / (m * (m - 1)) - 2.0 * bxy / (n * m)


def mmd_test(
    x_values: np.ndarray,
    x_status: np.ndarray,
    y_values: np.ndarray,
    y_status: np.ndarray,
    *,
    numeric: np.ndarray | None = None,
    permutations: int = 500,
    seed: int = 0,
    batch: int = 64,
) -> MMDResult:
    """Status-aware kernel MMD two-sample test with a permutation p-value (module docstring).

    numeric: bool [C], True for magnitude fields and False for categorical codes (default: all numeric).
    """
    xv, yv = np.asarray(x_values, dtype=np.float64), np.asarray(y_values, dtype=np.float64)
    n, m = xv.shape[0], yv.shape[0]
    if n < 2 or m < 2:
        raise ValueError("each sample needs at least two records")
    pooled_v = np.vstack([xv, yv])
    pooled_s = np.vstack([np.asarray(x_status), np.asarray(y_status)])
    k, s2 = _status_kernel(pooled_v, pooled_s, numeric)
    nn = n + m
    z0 = np.zeros((1, nn))
    z0[0, :n] = 1.0
    obs = float(_mmd2_from_blocks(k, z0, n, m)[0])
    rng = np.random.default_rng(seed)
    null = []
    for b0 in range(0, permutations, batch):
        bsz = min(batch, permutations - b0)
        z = np.zeros((bsz, nn))
        for r in range(bsz):
            z[r, rng.permutation(nn)[:n]] = 1.0
        null.append(_mmd2_from_blocks(k, z, n, m))
    nul = np.concatenate(null) if null else np.zeros(0)
    p = float((1 + (nul >= obs).sum()) / (1 + nul.size))
    return MMDResult(mmd2=obs, p_value=p, bandwidth2=s2, n=n, m=m, permutations=int(nul.size),
                     null_mean=float(nul.mean()) if nul.size else float("nan"),
                     null_sd=float(nul.std(ddof=1)) if nul.size > 1 else float("nan"))


def fdr_adjust(p: np.ndarray, *, method: str = "bh") -> np.ndarray:
    """Benjamini-Hochberg or Benjamini-Yekutieli adjusted p-values (module docstring); NaN stays NaN."""
    pv = np.asarray(p, dtype=np.float64)
    out = np.full(pv.shape, np.nan)
    ok = np.isfinite(pv)
    m = int(ok.sum())
    if m == 0:
        return out
    q = pv[ok]
    order = np.argsort(q, kind="stable")
    ranked = q[order] * m / np.arange(1, m + 1)
    if method == "by":
        ranked *= float(np.sum(1.0 / np.arange(1, m + 1)))
    elif method != "bh":
        raise ValueError("method must be 'bh' or 'by'")
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adj = np.empty(m)
    adj[order] = np.minimum(ranked, 1.0)
    out[ok] = adj
    return out


def psi(reference: np.ndarray, current: np.ndarray, *, bins: int = 10, epsilon: float = 1e-4,
        ref_absent: int = 0, cur_absent: int = 0) -> float:
    """Population stability index of contributing values plus an absent-cell bin (module docstring)."""
    r = np.asarray(reference, dtype=np.float64)
    c = np.asarray(current, dtype=np.float64)
    r, c = r[np.isfinite(r)], c[np.isfinite(c)]
    nr, nc = r.size + ref_absent, c.size + cur_absent
    if nr == 0 or nc == 0:
        return float("nan")
    edges = np.unique(np.quantile(r, np.linspace(0, 1, bins + 1))) if r.size else np.array([0.0, 1.0])
    inner = edges[1:-1]
    pr = np.bincount(np.searchsorted(inner, r, side="right"), minlength=inner.size + 1) / nr
    pc = np.bincount(np.searchsorted(inner, c, side="right"), minlength=inner.size + 1) / nc
    pr = np.append(pr, ref_absent / nr)
    pc = np.append(pc, cur_absent / nc)
    pr, pc = np.maximum(pr, epsilon), np.maximum(pc, epsilon)
    return float(((pc - pr) * np.log(pc / pr)).sum())


def _chi2_homogeneity(a: np.ndarray, b: np.ndarray, top: int = 50) -> tuple[float, float]:
    """(statistic, p) of the chi-square test of homogeneity of two categorical samples (top codes + other)."""
    codes, k = factorize(np.concatenate([a, b]))
    ca, cb = codes[: a.size], codes[a.size:]
    counts = np.bincount(codes, minlength=k)
    keep = np.argsort(-counts, kind="stable")[:top]
    remap = np.full(k, keep.size)
    remap[keep] = np.arange(keep.size)
    ta = np.bincount(remap[ca], minlength=keep.size + 1)
    tb = np.bincount(remap[cb], minlength=keep.size + 1)
    table = np.stack([ta, tb])
    table = table[:, table.sum(axis=0) > 0]
    if table.shape[1] < 2:
        return 0.0, 1.0
    res = stats.chi2_contingency(table, correction=False)
    return float(res.statistic), float(res.pvalue)


def feature_drift(reference: Corpus, current: Corpus, cfg: DriftConfig) -> pd.DataFrame:
    """Per-column value and status tests, PSI and FDR-adjusted p-values (module docstring)."""
    from nagahana.analytics.corpus import EXCLUDED_CODES

    rows = []
    numeric = reference.numeric_columns()
    rc, cc = reference.contributing(), current.contributing()
    for j, col in enumerate(reference.columns):
        if current.columns[j].name != col.name:
            raise ValueError("reference and current corpora must share the column layout")
        rv, cv = reference.values[rc[:, j], j], current.values[cc[:, j], j]
        row: dict[str, object] = {"column": col.name, "kind": col.kind.value, "reference_n": int(rv.size), "current_n": int(cv.size)}
        if rv.size >= 2 and cv.size >= 2:
            if numeric[j]:
                ks = stats.ks_2samp(rv, cv)
                row.update({"value_test": "ks", "value_statistic": float(ks.statistic), "value_p": float(ks.pvalue)})
            else:
                s, p = _chi2_homogeneity(rv, cv)
                row.update({"value_test": "chi2", "value_statistic": s, "value_p": p})
        st_s, st_p = _chi2_homogeneity(reference.status[:, j], current.status[:, j])
        row.update({"status_statistic": st_s, "status_p": st_p,
                    "psi": psi(rv, cv, bins=cfg.psi_bins, epsilon=cfg.psi_epsilon,
                               ref_absent=int(np.isin(reference.status[:, j], EXCLUDED_CODES).sum()),
                               cur_absent=int(np.isin(current.status[:, j], EXCLUDED_CODES).sum()))})
        rows.append(row)
    frame = pd.DataFrame(rows)
    for c in ("value_test", "value_statistic", "value_p"):
        if c not in frame:
            frame[c] = np.nan
    allp = np.concatenate([frame["value_p"].to_numpy(dtype=float), frame["status_p"].to_numpy(dtype=float)])
    adj = fdr_adjust(allp, method=cfg.fdr)
    frame["value_q"] = adj[: len(frame)]
    frame["status_q"] = adj[len(frame):]
    frame["drifted"] = (frame["value_q"] < cfg.ks_alpha) | (frame["status_q"] < cfg.ks_alpha)
    return frame


def fit_softmax(x: np.ndarray, y: np.ndarray, n_classes: int, *, l2: float = 1e-3, max_iter: int = 500) -> np.ndarray:
    """Multinomial logistic regression weights [d + 1, K] (intercept last) by L-BFGS on the L2-penalised NLL."""
    xa = np.hstack([x, np.ones((x.shape[0], 1))])
    n, d = xa.shape
    onehot = np.zeros((n, n_classes))
    onehot[np.arange(n), y] = 1.0

    def f(wflat: np.ndarray) -> tuple[float, np.ndarray]:
        w = wflat.reshape(d, n_classes)
        logits = xa @ w
        lse = special.logsumexp(logits, axis=1)
        nll = float((lse - (logits * onehot).sum(axis=1)).sum()) / n
        prob = np.exp(logits - lse[:, None])
        grad = xa.T @ (prob - onehot) / n
        reg = l2 * w.copy()
        reg[-1] = 0.0                                                    # intercepts are not penalised
        return nll + 0.5 * l2 * float((w[:-1] ** 2).sum()), (grad + reg).ravel()

    res = optimize.minimize(f, np.zeros(d * n_classes), jac=True, method="L-BFGS-B", options={"maxiter": max_iter})
    return np.asarray(res.x).reshape(d, n_classes)


def predict_softmax(w: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Most probable class of each row."""
    return np.argmax(np.hstack([x, np.ones((x.shape[0], 1))]) @ w, axis=1)


@dataclass(frozen=True)
class LabelShift:
    """BBSE result (module docstring)."""

    classes: tuple[str, ...]
    source_prior: np.ndarray
    weights: np.ndarray
    target_prior: np.ndarray
    confusion: np.ndarray
    condition_number: float
    prediction_shift_p: float
    target_prior_observed: np.ndarray | None = field(default=None)


def bbse(y_source: np.ndarray, pred_source: np.ndarray, pred_target: np.ndarray, n_classes: int, *, l2: float = 0.0,
         y_target: np.ndarray | None = None, classes: tuple[str, ...] | None = None) -> LabelShift:
    """Black-box shift estimation from held-out source labels and predictions (module docstring)."""
    ys, ps, pt = (np.asarray(a, dtype=np.int64) for a in (y_source, pred_source, pred_target))
    n = ys.size
    conf = np.zeros((n_classes, n_classes))
    np.add.at(conf, (ps, ys), 1.0)
    conf /= max(n, 1)
    mu = np.bincount(pt, minlength=n_classes) / max(pt.size, 1)
    src_prior = np.bincount(ys, minlength=n_classes) / max(n, 1)
    a = conf.T @ conf + l2 * np.eye(n_classes)
    w = np.linalg.lstsq(a, conf.T @ mu, rcond=None)[0] if l2 > 0 else np.linalg.lstsq(conf, mu, rcond=None)[0]
    w = np.clip(w, 0.0, None)
    q = w * src_prior
    q = q / q.sum() if q.sum() > 0 else q
    sv = np.linalg.svd(conf, compute_uv=False)
    cond = float(sv.max() / sv.min()) if sv.min() > 0 else float("inf")
    _, p = _chi2_homogeneity(ps, pt)
    obs = None if y_target is None else np.bincount(np.asarray(y_target, dtype=np.int64), minlength=n_classes) / len(y_target)
    return LabelShift(classes=classes or tuple(str(i) for i in range(n_classes)), source_prior=src_prior, weights=w,
                      target_prior=q, confusion=conf, condition_number=cond, prediction_shift_p=p,
                      target_prior_observed=obs)


class ADWIN:
    """ADWIN2 change detector over values in [0, 1] (module docstring)."""

    def __init__(self, delta: float = 0.002, max_buckets: int = 5) -> None:
        if not 0.0 < delta < 1.0 or max_buckets < 2:
            raise ValueError("delta must be in (0, 1) and max_buckets >= 2")
        self.delta = delta
        self.max_buckets = max_buckets
        self.levels: list[list[tuple[int, float, float]]] = []        # level i: buckets (count, total, M2), oldest first
        self.n = 0
        self.total = 0.0
        self.m2 = 0.0

    @staticmethod
    def _merge(a: tuple[int, float, float], b: tuple[int, float, float]) -> tuple[int, float, float]:
        na, ta, va = a
        nb, tb, vb = b
        n = na + nb
        d = ta / na - tb / nb
        return n, ta + tb, va + vb + d * d * na * nb / n

    def _drop_oldest(self) -> None:
        for lvl in range(len(self.levels) - 1, -1, -1):
            if self.levels[lvl]:
                old = self.levels[lvl].pop(0)
                rest_n = self.n - old[0]
                if rest_n > 0:
                    d = old[1] / old[0] - (self.total - old[1]) / rest_n
                    self.m2 -= old[2] + d * d * old[0] * rest_n / self.n
                else:
                    self.m2 = 0.0
                self.n = rest_n
                self.total -= old[1]
                self.m2 = max(self.m2, 0.0)
                return

    def update(self, x: float) -> bool:
        """Insert one value; True when a change was detected (the window shrank)."""
        if not 0.0 <= x <= 1.0:
            raise ValueError("ADWIN values must lie in [0, 1]")
        if self.n:
            mean = self.total / self.n
            self.m2 += (x - mean) ** 2 * self.n / (self.n + 1)
        self.n += 1
        self.total += x
        if not self.levels:
            self.levels.append([])
        self.levels[0].append((1, x, 0.0))
        lvl = 0
        while len(self.levels[lvl]) > self.max_buckets:                # compress: merge the two oldest of a level
            a, b = self.levels[lvl].pop(0), self.levels[lvl].pop(0)
            if lvl + 1 == len(self.levels):
                self.levels.append([])
            self.levels[lvl + 1].append(self._merge(a, b))
            lvl += 1
        changed = False
        while self._cut():
            self._drop_oldest()
            changed = True
        return changed

    def _cut(self) -> bool:
        """True when some split at a bucket boundary exceeds eps_cut (module docstring)."""
        if self.n < 2:
            return False
        var = self.m2 / self.n
        dprime = self.delta / self.n
        n0, t0 = 0, 0.0
        for lvl in range(len(self.levels) - 1, -1, -1):                # oldest buckets sit at the highest levels
            for cnt, tot, _ in self.levels[lvl]:
                n0 += cnt
                t0 += tot
                n1 = self.n - n0
                if n1 <= 0:
                    return False
                m = 1.0 / (1.0 / n0 + 1.0 / n1)
                ln = np.log(2.0 / dprime)
                eps = np.sqrt(2.0 / m * var * ln) + 2.0 / (3.0 * m) * ln
                if abs(t0 / n0 - (self.total - t0) / n1) > eps:
                    return True
        return False

    @property
    def width(self) -> int:
        return self.n

    @property
    def mean(self) -> float:
        return self.total / self.n if self.n else float("nan")


def adwin_changes(x: np.ndarray, *, delta: float = 0.002, max_buckets: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """(indices where ADWIN detected a change, window width after each value) of a series, rescaled to [0, 1]."""
    a = np.asarray(x, dtype=np.float64).reshape(-1)
    lo, hi = float(np.min(a)), float(np.max(a))
    z = (a - lo) / (hi - lo) if hi > lo else np.zeros_like(a)
    det = ADWIN(delta=delta, max_buckets=max_buckets)
    hits, widths = [], np.empty(a.size, dtype=np.int64)
    for i, v in enumerate(z.tolist()):
        if det.update(min(max(v, 0.0), 1.0)):
            hits.append(i)
        widths[i] = det.width
    return np.array(hits, dtype=np.int64), widths


def page_hinkley(x: np.ndarray, *, delta: float = 0.005, threshold: float = 50.0) -> tuple[np.ndarray, np.ndarray]:
    """(alarms for increases, alarms for decreases) of the two-sided Page-Hinkley test (module docstring)."""
    a = np.asarray(x, dtype=np.float64).reshape(-1)
    up, down = [], []
    t, mean = 0, 0.0
    m_up = m_dn = 0.0
    min_up = max_dn = 0.0
    for i, v in enumerate(a.tolist()):
        t += 1
        mean += (v - mean) / t
        m_up += v - mean - delta
        m_dn += v - mean + delta
        min_up = min(min_up, m_up)
        max_dn = max(max_dn, m_dn)
        hit = False
        if m_up - min_up > threshold:
            up.append(i)
            hit = True
        elif max_dn - m_dn > threshold:
            down.append(i)
            hit = True
        if hit:                                                          # restart after an alarm
            t, mean, m_up, m_dn, min_up, max_dn = 0, 0.0, 0.0, 0.0, 0.0, 0.0
    return np.array(up, dtype=np.int64), np.array(down, dtype=np.int64)


def ddm(errors: np.ndarray, *, min_samples: int = 30, warning: float = 2.0, drift: float = 3.0) -> tuple[np.ndarray, np.ndarray]:
    """(warning indices, drift indices) of DDM on a 0/1 error stream (module docstring)."""
    e = np.asarray(errors, dtype=np.float64).reshape(-1)
    if not np.isin(e, (0.0, 1.0)).all():
        raise ValueError("DDM needs a 0/1 error stream")
    warns, drifts = [], []
    i, err = 0, 0.0
    p_min = s_min = np.inf
    for k, v in enumerate(e.tolist()):
        i += 1
        err += v
        p = err / i
        s = np.sqrt(p * (1.0 - p) / i)
        if i < min_samples:
            continue
        if p + s < p_min + s_min:
            p_min, s_min = p, s
        if p + s > p_min + drift * s_min:
            drifts.append(k)
            i, err, p_min, s_min = 0, 0.0, np.inf, np.inf
        elif p + s > p_min + warning * s_min:
            warns.append(k)
    return np.array(warns, dtype=np.int64), np.array(drifts, dtype=np.int64)


def _numeric_design(corpus: Corpus, cols: list[int], med: np.ndarray, scale: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(rows with every column contributing, standardised slog1p design) for the black-box classifier."""
    ci = np.asarray(cols, dtype=np.int64)
    c = corpus.contributing()[:, ci].all(axis=1) if ci.size else np.zeros(len(corpus), dtype=bool)
    z = (slog1p(corpus.values[np.ix_(np.flatnonzero(c), ci)]) - med) / scale
    return np.flatnonzero(c), z


def run(reference: Corpus, current: Corpus, cfg: DriftConfig | None = None, *, label: str = "stage") -> Report:
    """Covariate shift, label shift and streaming detectors between two corpora (module docstring).

    label: "stage" (ATT&CK stage codes) or "malicious" for the label-shift estimate.
    """
    cfg = cfg or DriftConfig()
    rep = Report(kind="drift", title="Drift between a reference and a current corpus", provenance={"config": to_dict(cfg)})
    if [c.name for c in reference.columns] != [c.name for c in current.columns]:
        raise ValueError("reference and current corpora must share the column layout")
    rng = np.random.default_rng(cfg.seed)
    ri = np.sort(rng.choice(len(reference), size=min(cfg.mmd_max_rows, len(reference)), replace=False))
    ci = np.sort(rng.choice(len(current), size=min(cfg.mmd_max_rows, len(current)), replace=False))
    informative_cols = [j for j in range(len(reference.columns))
                   if np.unique(np.concatenate([reference.status[ri, j], current.status[ci, j]])).size > 1
                   or np.isfinite(np.concatenate([reference.values[ri, j], current.values[ci, j]])).any()]
    informative = np.asarray(informative_cols, dtype=np.int64)
    mm = mmd_test(reference.values[np.ix_(ri, informative)], reference.status[np.ix_(ri, informative)],
                  current.values[np.ix_(ci, informative)], current.status[np.ix_(ci, informative)],
                  numeric=reference.numeric_columns()[informative], permutations=cfg.mmd_permutations, seed=cfg.seed)
    rep.summary.update({"mmd2": mm.mmd2, "mmd_p": mm.p_value, "mmd_permutations": mm.permutations,
                        "mmd_rows": [mm.n, mm.m], "mmd_bandwidth2": mm.bandwidth2})
    fd = feature_drift(reference, current, cfg)
    rep.add_table("feature_drift", fd, title="Per-feature drift tests",
                  description=f"KS (numeric) or chi-square (categorical) on values, chi-square on statuses; "
                              f"{cfg.fdr.upper()} adjusted over all tests; PSI with an absent-cell bin.")
    rep.summary["drifted_features"] = fd.loc[fd["drifted"], "column"].tolist()
    rep.add_figure(Figure(name="psi", kind="bar", data=fd[["column", "psi"]], x="column", y="psi",
                          title="Population stability index per feature"))
    # Label shift through a black-box classifier fitted on the reference.
    if label == "stage":
        ys_all, yt_all = reference.stage, current.stage
        known_s = ys_all >= 0
    else:
        ys_all, yt_all = reference.malicious, current.malicious
        known_s = np.isin(ys_all, (0.0, 1.0))
    numeric = reference.numeric_columns()
    share = reference.contributing().mean(axis=0) if len(reference) else np.zeros(len(reference.columns))
    cols = [j for j in range(len(reference.columns)) if numeric[j] and share[j] >= 0.5]
    if cols and known_s.sum() >= 20:
        z_all = slog1p(reference.values[:, cols])
        med = np.nanmedian(z_all, axis=0)
        mad = np.nanmedian(np.abs(z_all - med), axis=0) * MAD_NORMAL
        scale = np.where(mad > 0, mad, 1.0)
        rows_s, xs = _numeric_design(reference, cols, med, scale)
        rows_t, xt = _numeric_design(current, cols, med, scale)
        keep_s = known_s[rows_s]
        rows_s, xs = rows_s[keep_s], xs[keep_s]
        classes, ys = np.unique(ys_all[rows_s], return_inverse=True)
        if classes.size >= 2 and rows_t.size:
            perm = rng.permutation(rows_s.size)
            half = rows_s.size // 2
            tr, ho = perm[:half], perm[half:]
            w = fit_softmax(xs[tr], ys[tr], classes.size, l2=cfg.bbse_l2)
            yt_rows = yt_all[rows_t]
            obs_mask = np.isin(yt_rows, classes)
            y_target = np.searchsorted(classes, yt_rows[obs_mask]) if obs_mask.all() else None
            ls = bbse(ys[ho], predict_softmax(w, xs[ho]), predict_softmax(w, xt), classes.size, l2=0.0,
                      y_target=y_target, classes=tuple(str(c) for c in classes))
            tab = pd.DataFrame({"class": ls.classes, "source_prior": ls.source_prior, "weight": ls.weights,
                                "target_prior_estimated": ls.target_prior})
            if ls.target_prior_observed is not None:
                tab["target_prior_observed"] = ls.target_prior_observed
            rep.add_table("label_shift", tab, title=f"Black-box shift estimation ({label})")
            rep.summary.update({"bbse_condition_number": ls.condition_number, "bbse_prediction_shift_p": ls.prediction_shift_p})
            # DDM over the classifier's errors on the current corpus, in time order (labels known).
            if obs_mask.any():
                order = np.argsort(np.nan_to_num(current.time[rows_t], nan=0.0), kind="stable")
                pred_t = predict_softmax(w, xt)[order]
                truth = yt_rows[order]
                ok = np.isin(truth, classes)
                err = (classes[pred_t[ok]] != truth[ok]).astype(np.float64)
                warns, drifts = ddm(err, min_samples=cfg.ddm_min_samples, warning=cfg.ddm_warning, drift=cfg.ddm_drift)
                rep.summary.update({"ddm_warnings": int(warns.size), "ddm_drifts": drifts.tolist(),
                                    "classifier_error_current": float(err.mean()) if err.size else float("nan")})
    else:
        rep.notes.append("Label shift was not estimated: too few labelled reference rows or no well-covered numeric field.")
    # Streaming detectors over the binned update rate and malicious share of the time-ordered concatenation.
    t = np.concatenate([reference.time, current.time])
    mal = np.concatenate([reference.malicious, current.malicious])
    timed = np.isfinite(t)
    if timed.sum() >= 4:
        tt = t[timed]
        b = np.floor((tt - tt.min()) / cfg.stream_bin_seconds).astype(np.int64)
        counts = np.bincount(b).astype(np.float64)
        known = np.isin(mal[timed], (0.0, 1.0))
        num = np.bincount(b[known], weights=mal[timed][known], minlength=counts.size)
        den = np.bincount(b[known], minlength=counts.size)
        share_series = np.where(den > 0, num / np.maximum(den, 1), np.nan)
        a_hits, widths = adwin_changes(counts, delta=cfg.adwin_delta, max_buckets=cfg.adwin_max_buckets)
        ph_up, ph_dn = page_hinkley(counts, delta=cfg.ph_delta, threshold=cfg.ph_threshold)
        fin = np.isfinite(share_series)
        s_hits = adwin_changes(share_series[fin], delta=cfg.adwin_delta, max_buckets=cfg.adwin_max_buckets)[0] if fin.sum() else np.zeros(0, dtype=np.int64)
        rep.summary.update({"adwin_rate_changes": a_hits.tolist(), "page_hinkley_increases": ph_up.tolist(),
                            "page_hinkley_decreases": ph_dn.tolist(),
                            "adwin_malicious_share_changes": np.flatnonzero(fin)[s_hits].tolist() if s_hits.size else []})
        rep.add_figure(Figure(name="stream", kind="line",
                              data=pd.DataFrame({"bin": np.arange(counts.size), "updates": counts, "adwin_width": widths,
                                                 "malicious_share": share_series}),
                              x="bin", y="updates", title="Update rate with ADWIN window width",
                              description=f"Bins of {cfg.stream_bin_seconds} s over reference then current."))
    return rep


__all__ = ["ADWIN", "LabelShift", "MMDResult", "adwin_changes", "bbse", "ddm", "fdr_adjust", "feature_drift",
           "fit_softmax", "mmd_test", "page_hinkley", "predict_softmax", "psi", "run"]
