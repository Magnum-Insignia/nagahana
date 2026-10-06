"""Exact and near-duplicate state updates: hashing, banded locality-sensitive hashing, label conflicts.

Duplicates matter three ways: they inflate the apparent size of a corpus, a duplicate shared by two
splits leaks the test set into training, and identical inputs with different labels set a floor under
the error of every model that sees only those inputs.

Exact duplicates
----------------
Two updates are exact duplicates when every column has the same observation status and every
contributing cell the same value (absent cells are compared by status only, D-41). Each row is
encoded as 64-bit words (the canonical float64 bits of its values with absent cells and -0.0
normalised, then its status bytes) and hashed with the SplitMix64 finaliser (Steele, Lea and Flood,
"Fast splittable pseudorandom number generators", OOPSLA 2014) chained over the words. Rows are grouped
by hash and every group is verified word by word against its first member, so a hash collision can
never merge two different rows (a colliding group is split exactly). Raw duplicates are grouped by the
SHA-256 digest of the raw record (`ColumnarUpdates.raw_hash`).

Near duplicates
---------------
Similarity of two updates over the C informative columns (those whose status or value varies among
the examined rows; a column that is identical in every row, such as a field one source never
supplies, would make every pair look alike and carries no evidence of duplication):
    s(i, j) = (1/C) sum_c m_c(i, j),
    m_c = [status_ic = status_jc] and (cell absent, or categorical and equal value,
          or numeric and |slog1p(v_ic) - slog1p(v_jc)| <= w)
with w the `tolerance` on the signed-log scale (w = 0.05 is about 5 % relative difference). A pair is
a near duplicate when s >= `threshold`. Candidate pairs come from banded LSH: band b samples r
columns and draws a random shift u_bc ~ U[0, w) per column; a numeric cell is quantised to
floor((slog1p(v) + u_bc) / w), so two values at signed-log distance d share a bin with probability
max(0, 1 - d / w) (the one-dimensional case of Datar, Immorlica, Indyk and Mirrokni, SoCG 2004),
and categorical and absent cells are kept as (status, value) tokens. Rows whose r tokens agree in a
band collide there. With per-column agreement probability p, a band collides with probability p^r and
the pair is proposed with probability 1 - (1 - p^r)^b: the S-curve of banded LSH (Leskovec, Rajaraman
and Ullman, "Mining of Massive Datasets", Cambridge University Press, chapter 3); bit sampling for
Hamming similarity is due to Indyk and Motwani (STOC 1998). Every candidate is verified with the exact
similarity above, so LSH can only miss pairs, never invent them. Exact duplicates are collapsed before
LSH (each distinct row once, with its multiplicity). A bucket larger than `max_bucket` (a flood of
near-identical updates) is verified against `pivots` seeded members instead of all pairs, which bounds
the work and only adds misses. Near-duplicate clusters are the connected components of the verified
pairs (scipy.sparse.csgraph.connected_components).

Label conflicts and the duplicate floor
---------------------------------------
For a group of identical inputs of size g whose known labels have majority count m, any function of
those inputs, deterministic or randomised, errs on at least g - m of its members. The share
sum_groups (g - m) / n_known is therefore an exact lower bound on the training error of every model
restricted to those inputs (it is the empirical Bayes error of the duplicated part).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from nagahana.analytics.dependence import slog1p

_M1 = np.uint64(0xBF58476D1CE4E5B9)
_M2 = np.uint64(0x94D049BB133111EB)
_GOLDEN = np.uint64(0x9E3779B97F4A7C15)


def splitmix64(z: np.ndarray | np.unsignedinteger[Any] | int) -> np.ndarray:
    """SplitMix64 finaliser applied elementwise to uint64 values (bijective mixing)."""
    a: np.ndarray = np.asarray(z, dtype=np.uint64)
    with np.errstate(over="ignore"):
        a = (a ^ (a >> np.uint64(30))) * _M1
        a = (a ^ (a >> np.uint64(27))) * _M2
    return np.asarray(a ^ (a >> np.uint64(31)), dtype=np.uint64)


def hash_words(words: np.ndarray, *, seed: int = 0) -> np.ndarray:
    """64-bit hash of each row of a uint64 word matrix [n, W], chained SplitMix64 over the words."""
    w = np.asarray(words, dtype=np.uint64)
    h = np.full(w.shape[0], splitmix64(np.uint64(seed) + _GOLDEN), dtype=np.uint64)
    with np.errstate(over="ignore"):
        for j in range(w.shape[1]):
            h = splitmix64(h ^ (w[:, j] + _GOLDEN * np.uint64(j + 1)))
    return h


def row_words(values: np.ndarray, status: np.ndarray) -> np.ndarray:
    """Canonical uint64 words of each row: value bits (absent cells and -0.0 normalised), then status bytes."""
    v = np.asarray(values, dtype=np.float64)
    s = np.ascontiguousarray(np.asarray(status, dtype=np.uint8))
    if v.shape != s.shape or v.ndim != 2:
        raise ValueError("values and status must be 2-D arrays of the same shape")
    canon = np.where(np.isnan(v), 0.0, v) + 0.0                       # NaN never carries bits; -0.0 -> 0.0
    vbits = np.ascontiguousarray(canon).view(np.uint64)               # [n, C]
    pad = (-s.shape[1]) % 8
    sp = np.concatenate([s, np.zeros((s.shape[0], pad), dtype=np.uint8)], axis=1) if pad else s
    sbits = np.ascontiguousarray(sp).view(np.uint64)                  # [n, ceil(C/8)]
    return np.concatenate([vbits, sbits], axis=1)


@dataclass(frozen=True)
class DuplicateGroups:
    """Groups of identical rows.

    group: int64 [n] group id, -1 for a row without an identical partner. sizes, representative:
    [G] size and first row of every group. n_duplicate_rows: rows in some group; n_redundant_rows:
    rows beyond the first of their group (what deduplication would remove).
    """

    group: np.ndarray
    sizes: np.ndarray
    representative: np.ndarray
    n_duplicate_rows: int
    n_redundant_rows: int


def _groups_from_codes(codes: np.ndarray) -> DuplicateGroups:
    """DuplicateGroups from exact row codes (equal code = identical row)."""
    n = codes.size
    uniq, first, inv, counts = np.unique(codes, return_index=True, return_inverse=True, return_counts=True)
    inv = inv.reshape(-1)
    multi = counts >= 2
    gid = np.full(uniq.size, -1, dtype=np.int64)
    gid[multi] = np.arange(int(multi.sum()))
    group = gid[inv] if n else np.zeros(0, dtype=np.int64)
    sizes = counts[multi].astype(np.int64)
    return DuplicateGroups(group=group, sizes=sizes, representative=first[multi].astype(np.int64),
                           n_duplicate_rows=int(sizes.sum()), n_redundant_rows=int((sizes - 1).sum()))


def exact_codes(words: np.ndarray) -> np.ndarray:
    """int64 codes, equal exactly when two rows of `words` are identical (hash, then exact verification)."""
    w = np.ascontiguousarray(np.asarray(words, dtype=np.uint64))
    n = w.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    h = hash_words(w)
    _, first, inv = np.unique(h, return_index=True, return_inverse=True)
    inv = inv.reshape(-1)
    codes = inv.astype(np.int64).copy()
    # Verify every row against the first row of its hash group; split groups that collided.
    bad = ~np.all(w == w[first[inv]], axis=1)
    if bad.any():
        next_code = int(codes.max()) + 1
        for g in np.unique(inv[bad]).tolist():
            members = np.flatnonzero(inv == g)
            rows = np.ascontiguousarray(w[members]).view(np.dtype((np.void, w.shape[1] * 8))).reshape(-1)
            _, sub = np.unique(rows, return_inverse=True)
            codes[members] = np.where(sub.reshape(-1) == 0, g, next_code + sub.reshape(-1) - 1)
            next_code += int(sub.max())
    return codes


def exact_duplicates(values: np.ndarray, status: np.ndarray) -> DuplicateGroups:
    """Exact duplicates by value and status (module docstring)."""
    return _groups_from_codes(exact_codes(row_words(values, status)))


def raw_duplicates(raw_hash: np.ndarray) -> DuplicateGroups:
    """Duplicates of the raw record, by its SHA-256 digest (uint8 [n, 32])."""
    r = np.ascontiguousarray(np.asarray(raw_hash, dtype=np.uint8))
    if r.ndim != 2 or r.shape[1] != 32:
        raise ValueError("raw_hash must be uint8 [n, 32]")
    return _groups_from_codes(exact_codes(r.view(np.uint64)))


@dataclass(frozen=True)
class LabelConflicts:
    """Identical inputs with different known labels (module docstring, "duplicate floor").

    n_known: rows with a known label. conflicting_groups: groups with at least two distinct known
    labels. conflicting_rows: rows in those groups. floor_errors: sum over groups of (g - m).
    error_floor: floor_errors / n_known.
    """

    n_known: int
    conflicting_groups: int
    conflicting_rows: int
    floor_errors: int
    error_floor: float


def label_conflicts(group: np.ndarray, labels: np.ndarray) -> LabelConflicts:
    """Label conflicts inside duplicate groups; NaN or None labels count as unknown and are ignored."""
    lab = pd.Series(np.asarray(labels, dtype=object))
    known = lab.notna().to_numpy() & (np.asarray(group) >= 0)
    n_known_all = int(lab.notna().sum())
    if not known.any():
        return LabelConflicts(n_known_all, 0, 0, 0, 0.0)
    frame = pd.DataFrame({"g": np.asarray(group)[known], "y": lab[known].astype(str).to_numpy()})
    counts = frame.groupby(["g", "y"]).size()
    per_group = counts.groupby(level=0)
    size = per_group.sum()
    top = per_group.max()
    distinct = per_group.size()
    conflict = distinct >= 2
    floor = int((size - top)[conflict].sum())
    return LabelConflicts(
        n_known=n_known_all, conflicting_groups=int(conflict.sum()), conflicting_rows=int(size[conflict].sum()),
        floor_errors=floor, error_floor=float(floor / n_known_all) if n_known_all else float("nan"),
    )


@dataclass(frozen=True)
class NearDuplicates:
    """Near-duplicate clusters (module docstring).

    cluster: int64 [n] cluster id, -1 for a row with no near duplicate. sizes: [K] rows per cluster.
    pairs: verified pairs between distinct rows (row_i, row_j, similarity), at most `max_pairs_report`.
    n_candidates, n_verified: distinct candidate pairs proposed by LSH and those that passed.
    n_distinct: distinct rows after collapsing exact duplicates.
    """

    cluster: np.ndarray
    sizes: np.ndarray
    pairs: pd.DataFrame
    n_candidates: int
    n_verified: int
    n_distinct: int
    params: dict[str, float] = field(default_factory=dict)


def _tokens(v: np.ndarray, s: np.ndarray, numeric: np.ndarray, contributing: np.ndarray, shift: np.ndarray,
            tolerance: float, cols: np.ndarray, band_seed: int) -> np.ndarray:
    """uint64 band key of each row from the sampled columns `cols` (module docstring)."""
    key = np.full(v.shape[0], splitmix64(np.uint64(band_seed)), dtype=np.uint64)
    with np.errstate(over="ignore", invalid="ignore"):
        for t, c in enumerate(cols.tolist()):
            vc = v[:, c]
            if numeric[c]:
                q = np.floor((slog1p(vc) + shift[t]) / tolerance)
                val = np.where(contributing[:, c], q, 0.0)
            else:
                val = np.where(contributing[:, c], vc, 0.0)
            vb = (np.ascontiguousarray(val + 0.0).view(np.uint64))
            tok = splitmix64(vb ^ (s[:, c].astype(np.uint64) * _GOLDEN) ^ np.uint64(c + 1))
            key = splitmix64(key ^ tok)
    return key


def _bucket_pairs(keys: np.ndarray, *, max_bucket: int, pivots: int, rng: np.random.Generator) -> np.ndarray:
    """Candidate pairs (int64 [P, 2], i < j) of rows sharing a band key."""
    order = np.argsort(keys, kind="stable")
    k = keys[order]
    starts = np.flatnonzero(np.r_[True, k[1:] != k[:-1]])
    sizes = np.diff(np.r_[starts, k.size])
    out: list[np.ndarray] = []
    for size in np.unique(sizes[sizes >= 2]).tolist():
        runs = starts[sizes == size]
        if size <= max_bucket:
            members = order[runs[:, None] + np.arange(size)[None, :]]  # [R, size]
            iu, ju = np.triu_indices(size, 1)
            a, b = members[:, iu].reshape(-1), members[:, ju].reshape(-1)
        else:
            parts_a, parts_b = [], []
            for r0 in runs.tolist():
                members = order[r0: r0 + size]
                piv = members[rng.choice(size, size=min(pivots, size), replace=False)]
                parts_a.append(np.repeat(piv, size))
                parts_b.append(np.tile(members, piv.size))
            a, b = np.concatenate(parts_a), np.concatenate(parts_b)
        keep = a != b
        lo, hi = np.minimum(a, b)[keep], np.maximum(a, b)[keep]
        out.append(np.stack([lo, hi], axis=1))
    return np.concatenate(out) if out else np.zeros((0, 2), dtype=np.int64)


def similarity(v: np.ndarray, s: np.ndarray, numeric: np.ndarray, pairs: np.ndarray, *, tolerance: float) -> np.ndarray:
    """Exact similarity s(i, j) of the module docstring for each pair (float64 [P])."""
    i, j = pairs[:, 0], pairs[:, 1]
    same_status = s[i] == s[j]                                         # [P, C]
    contributing = ~np.isnan(v[i])
    lv = slog1p(v)
    with np.errstate(invalid="ignore"):
        num_match = np.abs(lv[i] - lv[j]) <= tolerance
        cat_match = v[i] == v[j]
    value_match = np.where(numeric[None, :], num_match, cat_match)
    match = same_status & (~contributing | value_match)
    return match.mean(axis=1)


def near_duplicates(
    values: np.ndarray,
    status: np.ndarray,
    numeric: np.ndarray,
    *,
    tolerance: float = 0.05,
    threshold: float = 0.95,
    bands: int = 24,
    rows_per_band: int = 8,
    max_bucket: int = 64,
    pivots: int = 8,
    seed: int = 0,
    max_pairs_report: int = 10_000,
    pair_chunk: int = 500_000,
) -> NearDuplicates:
    """Near-duplicate clusters by banded LSH plus exact verification (module docstring).

    Parameters
    ----------
    values, status : arrays [n, C]
        Cell values (NaN where absent) and observation status codes.
    numeric : bool array [C]
        True for continuous and count columns (compared within `tolerance` on the slog1p scale);
        False for categorical and bitmask columns (compared exactly).
    """
    v = np.asarray(values, dtype=np.float64)
    s = np.asarray(status, dtype=np.uint8)
    num = np.asarray(numeric, dtype=bool).reshape(-1)
    n, c = v.shape
    if s.shape != v.shape or num.shape != (c,):
        raise ValueError("values, status and numeric must agree in shape")
    if not 0.0 < threshold <= 1.0 or tolerance <= 0:
        raise ValueError("threshold must be in (0, 1] and tolerance > 0")
    params = {"tolerance": tolerance, "threshold": threshold, "bands": float(bands),
              "rows_per_band": float(rows_per_band), "max_bucket": float(max_bucket), "pivots": float(pivots)}
    empty = pd.DataFrame({"row_i": np.zeros(0, dtype=np.int64), "row_j": np.zeros(0, dtype=np.int64),
                          "similarity": np.zeros(0)})
    if n == 0 or c == 0:
        return NearDuplicates(np.full(n, -1, dtype=np.int64), np.zeros(0, dtype=np.int64), empty, 0, 0, 0, params)
    # Collapse exact duplicates: LSH runs over distinct rows, multiplicities are added back at the end.
    codes = exact_codes(row_words(v, s))
    _, rep, inv, mult = np.unique(codes, return_index=True, return_inverse=True, return_counts=True)
    inv = inv.reshape(-1)
    # Informative columns only (module docstring): a column whose (status, value) is the same in every row.
    first_v, first_s = v[0], s[0]
    same_status = (s == first_s).all(axis=0)
    same_value = ((v == first_v) | (np.isnan(v) & np.isnan(first_v))).all(axis=0)
    informative = np.flatnonzero(~(same_status & same_value))
    params["informative_columns"] = float(informative.size)
    if informative.size == 0:
        informative = np.arange(c)                                     # every row identical: exact groups only
    uv, us, num = v[rep][:, informative], s[rep][:, informative], num[informative]
    c = informative.size
    m = uv.shape[0]
    contributing = ~np.isnan(uv)
    rng = np.random.default_rng(seed)
    r = min(rows_per_band, c)
    cand: list[np.ndarray] = []
    for b in range(bands):
        cols = np.sort(rng.choice(c, size=r, replace=False))
        shift = rng.uniform(0.0, tolerance, size=r)
        keys = _tokens(uv, us, num, contributing, shift, tolerance, cols, band_seed=seed * 1_000_003 + b)
        cand.append(_bucket_pairs(keys, max_bucket=max_bucket, pivots=pivots, rng=rng))
    allp = np.concatenate(cand) if cand else np.zeros((0, 2), dtype=np.int64)
    if allp.size:
        packed = np.unique(allp[:, 0].astype(np.int64) * m + allp[:, 1])
        allp = np.stack([packed // m, packed % m], axis=1)
    n_cand = int(allp.shape[0])
    sims = np.concatenate([similarity(uv, us, num, allp[k: k + pair_chunk], tolerance=tolerance)
                           for k in range(0, n_cand, pair_chunk)]) if n_cand else np.zeros(0)
    ok = sims >= threshold
    vp, vs = allp[ok], sims[ok]
    # Clusters over distinct rows: verified pairs, then rows of any exact group of size >= 2.
    graph = coo_matrix((np.ones(vp.shape[0]), (vp[:, 0], vp[:, 1])), shape=(m, m))
    _, comp = connected_components(graph, directed=False)
    comp_rows = np.bincount(comp, weights=mult).astype(np.int64)      # rows per component
    multi = comp_rows >= 2
    cid = np.full(comp_rows.size, -1, dtype=np.int64)
    cid[multi] = np.arange(int(multi.sum()))
    cluster = cid[comp[inv]]
    ri = rep[vp[:max_pairs_report, 0]].astype(np.int64)
    rj = rep[vp[:max_pairs_report, 1]].astype(np.int64)
    pairs = pd.DataFrame({"row_i": np.minimum(ri, rj), "row_j": np.maximum(ri, rj),
                          "similarity": vs[:max_pairs_report]})
    return NearDuplicates(cluster=cluster, sizes=comp_rows[multi], pairs=pairs, n_candidates=n_cand,
                          n_verified=int(ok.sum()), n_distinct=int(m), params=params)


__all__ = [
    "DuplicateGroups", "LabelConflicts", "NearDuplicates", "exact_codes", "exact_duplicates", "hash_words",
    "label_conflicts", "near_duplicates", "raw_duplicates", "row_words", "similarity", "splitmix64",
]
