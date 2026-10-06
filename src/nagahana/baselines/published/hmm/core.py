"""Discrete hidden Markov models (Rabiner, Proceedings of the IEEE 77(2):257-286, 1989, "A Tutorial on
Hidden Markov Models and Selected Applications in Speech Recognition").

Model: hidden states s_t in {0 ... N-1}, symbols o_t in {0 ... M-1},

    P(s_1 = i) = pi_i,   P(s_t = j | s_{t-1} = i) = A_ij,   P(o_t = k | s_t = i) = B_ik.

A missing observation (code -1) contributes the factor 1 for every state, so a sequence with gaps is
filtered without inventing a symbol.

Scaled forward-backward (Rabiner Sec. V-A)
    alpha_1(i) = pi_i B_i(o_1);  c_1 = sum_i alpha_1(i);  alpha^_1 = alpha_1 / c_1
    alpha_t(j) = (sum_i alpha^_{t-1}(i) A_ij) B_j(o_t);  c_t = sum_j alpha_t(j);  alpha^_t = alpha_t / c_t
    log P(o_1:T) = sum_t log c_t;  alpha^_t(i) = P(s_t = i | o_1:t)
    beta^_T = 1;  beta^_t(i) = sum_j A_ij B_j(o_{t+1}) beta^_{t+1}(j) / c_{t+1}
    gamma_t(i) = alpha^_t(i) beta^_t(i) = P(s_t = i | o_1:T)
    xi_t(i, j) = alpha^_t(i) A_ij B_j(o_{t+1}) beta^_{t+1}(j) / c_{t+1} = P(s_t = i, s_{t+1} = j | o_1:T)

Log-space forward-backward: the same recursions with logsumexp in place of sums, exact where
probabilities are zero (log 0 = -inf), used as a cross-check and for models with structural zeros.

Baum-Welch (EM) with Dirichlet pseudo-counts a (MAP estimation; a = 0 is maximum likelihood)
    pi_i ~ sum_s gamma_{s,1}(i) + a_pi
    A_ij ~ sum_s sum_{t<T_s} xi_{s,t}(i, j) + a_A m_ij              m: allowed transitions (topology)
    B_ik ~ sum_s sum_{t: o_t = k} gamma_{s,t}(i) + a_B
Each iteration does not decrease log P(data) + sum a log(theta) (the log posterior under Dirichlet
priors with exponents a); with a = 0 this is the log-likelihood (Dempster, Laird and Rubin, JRSS-B 39,
1977). Both are recorded per iteration.

Viterbi: delta_t(j) = max_i [delta_{t-1}(i) + log A_ij] + log B_j(o_t), with back-pointers.

Prediction from the filtered belief b_t = alpha^_t
    P(s_{t+k} | o_1:t) = b_t A^k;   P(o_{t+k} | o_1:t) = b_t A^k B
First passage into a set I of states (an infiltration stage, AS-548): with N = complement of I,
    P(s_{t+1} not in I, ..., s_{t+k} not in I | o_1:t) = b_t A[:, N] (A[N, N])^{k-1} 1
    P_inf(k) = 1 - that product, non-decreasing in k.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from nagahana.core.errors import InvariantViolation

_TOL = 1e-8


@dataclass
class HMMParams:
    """Initial distribution pi [N], transition matrix A [N, N], emission matrix B [N, M]."""

    pi: np.ndarray
    A: np.ndarray
    B: np.ndarray

    def __post_init__(self) -> None:
        self.pi = np.asarray(self.pi, dtype=np.float64)
        self.A = np.asarray(self.A, dtype=np.float64)
        self.B = np.asarray(self.B, dtype=np.float64)
        self.validate()

    @property
    def n_states(self) -> int:
        return int(self.pi.shape[0])

    @property
    def n_symbols(self) -> int:
        return int(self.B.shape[1])

    def validate(self) -> None:
        """Shapes agree, entries are probabilities, every distribution sums to 1."""
        n = self.pi.shape[0]
        if self.pi.ndim != 1 or self.A.shape != (n, n) or self.B.ndim != 2 or self.B.shape[0] != n:
            raise InvariantViolation(f"HMM shapes disagree: pi {self.pi.shape}, A {self.A.shape}, B {self.B.shape}")
        for name, m in (("pi", self.pi), ("A", self.A), ("B", self.B)):
            if not np.all(np.isfinite(m)) or m.min() < -_TOL:
                raise InvariantViolation(f"{name} must hold finite non-negative probabilities")
        if abs(self.pi.sum() - 1.0) > 1e-6 or np.any(np.abs(self.A.sum(axis=1) - 1.0) > 1e-6) or np.any(
                np.abs(self.B.sum(axis=1) - 1.0) > 1e-6):
            raise InvariantViolation("pi, the rows of A and the rows of B must each sum to 1")

    def log(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Elementwise logarithms with log 0 = -inf."""
        with np.errstate(divide="ignore"):
            return np.log(self.pi), np.log(self.A), np.log(self.B)

    def state(self) -> dict[str, Any]:
        return {"pi": self.pi.tolist(), "A": self.A.tolist(), "B": self.B.tolist()}

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> HMMParams:
        return cls(np.asarray(state["pi"]), np.asarray(state["A"]), np.asarray(state["B"]))

    def copy(self) -> HMMParams:
        return HMMParams(self.pi.copy(), self.A.copy(), self.B.copy())


def _obs(obs: np.ndarray | Sequence[int], n_symbols: int) -> np.ndarray:
    o = np.asarray(obs, dtype=np.int64)
    if o.ndim != 1:
        raise InvariantViolation("an observation sequence must be one-dimensional")
    if o.size and (o.max() >= n_symbols or o.min() < -1):
        raise InvariantViolation(f"observations must lie in -1 ... {n_symbols - 1}")
    return o


def emission_matrix(params: HMMParams, obs: np.ndarray) -> np.ndarray:
    """[T, N] emission factors B_i(o_t), 1 for missing observations (code -1)."""
    o = _obs(obs, params.n_symbols)
    e = np.ones((o.size, params.n_states), dtype=np.float64)
    seen = o >= 0
    e[seen] = params.B[:, o[seen]].T
    return e


def forward(params: HMMParams, obs: np.ndarray | Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    """Scaled forward pass: (alpha^ [T, N], c [T]); log P(o) = sum(log c)."""
    e = emission_matrix(params, np.asarray(obs))
    t_len, n = e.shape
    alpha = np.empty((t_len, n), dtype=np.float64)
    c = np.empty(t_len, dtype=np.float64)
    prev = params.pi
    for t in range(t_len):
        a = (prev if t == 0 else prev @ params.A) * e[t]
        s = a.sum()
        if s <= 0.0:
            raise InvariantViolation(f"observation {t} has probability 0 under the model")
        c[t] = s
        alpha[t] = a / s
        prev = alpha[t]
    return alpha, c


def backward(params: HMMParams, obs: np.ndarray | Sequence[int], c: np.ndarray) -> np.ndarray:
    """Scaled backward pass with the forward scales c: beta^ [T, N]."""
    e = emission_matrix(params, np.asarray(obs))
    t_len, n = e.shape
    beta = np.empty((t_len, n), dtype=np.float64)
    if t_len == 0:
        return beta
    beta[-1] = 1.0
    for t in range(t_len - 2, -1, -1):
        beta[t] = params.A @ (e[t + 1] * beta[t + 1]) / c[t + 1]
    return beta


def loglik(params: HMMParams, obs: np.ndarray | Sequence[int]) -> float:
    """log P(o_1:T) (scaled forward)."""
    _, c = forward(params, obs)
    return float(np.log(c).sum())


def posteriors(params: HMMParams, obs: np.ndarray | Sequence[int]) -> tuple[np.ndarray, np.ndarray, float]:
    """(gamma [T, N], xi summed over t [N, N], log-likelihood) of one sequence."""
    o = np.asarray(obs, dtype=np.int64)
    alpha, c = forward(params, o)
    beta = backward(params, o, c)
    gamma = alpha * beta
    e = emission_matrix(params, o)
    if o.size > 1:
        # xi_t(i, j) = alpha^_t(i) A_ij B_j(o_{t+1}) beta^_{t+1}(j) / c_{t+1}, summed over t.
        right = e[1:] * beta[1:] / c[1:, None]                       # [T-1, N]
        xi_sum = params.A * (alpha[:-1].T @ right)                   # [N, N]
    else:
        xi_sum = np.zeros_like(params.A)
    return gamma, xi_sum, float(np.log(c).sum())


def _logsumexp(x: np.ndarray, axis: int) -> np.ndarray:
    m = np.max(x, axis=axis, keepdims=True)
    m_safe = np.where(np.isfinite(m), m, 0.0)
    with np.errstate(divide="ignore"):
        out = np.log(np.sum(np.exp(x - m_safe), axis=axis, keepdims=True)) + m_safe
    return np.squeeze(out, axis=axis)


def forward_log(params: HMMParams, obs: np.ndarray | Sequence[int]) -> tuple[np.ndarray, float]:
    """Log-space forward pass: (log alpha [T, N] with alpha_t(i) = P(o_1:t, s_t = i), log P(o))."""
    o = _obs(obs, params.n_symbols)
    log_pi, log_a, log_b = params.log()
    t_len, n = o.size, params.n_states
    la = np.empty((t_len, n), dtype=np.float64)
    for t in range(t_len):
        emit = log_b[:, o[t]] if o[t] >= 0 else np.zeros(n)
        la[t] = (log_pi if t == 0 else _logsumexp(la[t - 1][:, None] + log_a, axis=0)) + emit
    total = float(_logsumexp(la[-1], axis=0)) if t_len else 0.0
    return la, total


def backward_log(params: HMMParams, obs: np.ndarray | Sequence[int]) -> np.ndarray:
    """Log-space backward pass: log beta [T, N] with beta_t(i) = P(o_{t+1:T} | s_t = i)."""
    o = _obs(obs, params.n_symbols)
    _, log_a, log_b = params.log()
    t_len, n = o.size, params.n_states
    lb = np.zeros((t_len, n), dtype=np.float64)
    for t in range(t_len - 2, -1, -1):
        emit = log_b[:, o[t + 1]] if o[t + 1] >= 0 else np.zeros(n)
        lb[t] = _logsumexp(log_a + (emit + lb[t + 1])[None, :], axis=1)
    return lb


def viterbi(params: HMMParams, obs: np.ndarray | Sequence[int]) -> tuple[np.ndarray, float]:
    """Most probable state path and its log joint probability log P(s*, o)."""
    o = _obs(obs, params.n_symbols)
    log_pi, log_a, log_b = params.log()
    t_len, n = o.size, params.n_states
    if t_len == 0:
        return np.zeros(0, dtype=np.int64), 0.0
    delta = np.empty((t_len, n), dtype=np.float64)
    back = np.zeros((t_len, n), dtype=np.int64)
    emit0 = log_b[:, o[0]] if o[0] >= 0 else np.zeros(n)
    delta[0] = log_pi + emit0
    for t in range(1, t_len):
        cand = delta[t - 1][:, None] + log_a                          # [N (from), N (to)]
        back[t] = np.argmax(cand, axis=0)
        emit = log_b[:, o[t]] if o[t] >= 0 else np.zeros(n)
        delta[t] = cand[back[t], np.arange(n)] + emit
    path = np.empty(t_len, dtype=np.int64)
    path[-1] = int(np.argmax(delta[-1]))
    for t in range(t_len - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path, float(delta[-1, path[-1]])


def predict_states(params: HMMParams, belief: np.ndarray, k: int) -> np.ndarray:
    """P(s_{t+k} | o_1:t) = b A^k for belief rows b [..., N]."""
    if k < 0:
        raise ValueError("k must be non-negative")
    return np.asarray(belief, dtype=np.float64) @ np.linalg.matrix_power(params.A, k)


def predict_observations(params: HMMParams, belief: np.ndarray, k: int) -> np.ndarray:
    """P(o_{t+k} | o_1:t) = b A^k B for belief rows b [..., N] (k >= 1)."""
    if k < 1:
        raise ValueError("an observation is predicted at k >= 1 steps ahead")
    return predict_states(params, belief, k) @ params.B


def first_passage(params: HMMParams, belief: np.ndarray, targets: np.ndarray, horizon: int) -> np.ndarray:
    """P_inf(k), k = 1 ... horizon: probability of entering a target state within k steps.

    belief [..., N] is P(s_t | o_1:t); targets [N] bool marks the target (infiltration) states. Returns
    [..., horizon], non-decreasing in k.
    """
    tgt = np.asarray(targets, dtype=bool)
    if tgt.shape != (params.n_states,):
        raise InvariantViolation("targets must mark each state")
    b = np.asarray(belief, dtype=np.float64)
    keep = ~tgt
    a_to_n = params.A[:, keep]                                        # [N, |N|]
    a_nn = params.A[np.ix_(keep, keep)]                               # [|N|, |N|]
    survive = b @ a_to_n                                              # P(s_{t+1} in N, ...) [..., |N|]
    out = np.empty((*b.shape[:-1], horizon), dtype=np.float64)
    for k in range(horizon):
        out[..., k] = 1.0 - survive.sum(axis=-1)
        survive = survive @ a_nn
    return np.clip(np.maximum.accumulate(out, axis=-1), 0.0, 1.0)


@dataclass
class EMHistory:
    """Per-iteration log-likelihood and MAP objective of Baum-Welch."""

    loglik: list[float] = field(default_factory=list)
    objective: list[float] = field(default_factory=list)
    converged: bool = False


def _log_prior(params: HMMParams, pseudo_pi: float, pseudo_a: float, pseudo_b: float, mask: np.ndarray) -> float:
    # sum a log(theta) over the parameters that carry pseudo-counts (log 0 terms only where a = 0).
    lp, la, lb = params.log()
    total = 0.0
    if pseudo_pi > 0:
        total += pseudo_pi * float(lp.sum())
    if pseudo_a > 0:
        total += pseudo_a * float(la[mask].sum())
    if pseudo_b > 0:
        total += pseudo_b * float(lb.sum())
    return total


def _normalise_rows(counts: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    # Row-normalise; a row without any mass keeps its previous value.
    s = counts.sum(axis=-1, keepdims=True)
    return np.where(s > 0, counts / np.where(s > 0, s, 1.0), fallback)


def baum_welch(
    sequences: Sequence[np.ndarray],
    init: HMMParams,
    *,
    max_iter: int = 100,
    tol: float = 1e-6,
    pseudo_pi: float = 0.0,
    pseudo_a: float = 0.0,
    pseudo_b: float = 0.0,
    transition_mask: np.ndarray | None = None,
    update: tuple[bool, bool, bool] = (True, True, True),
) -> tuple[HMMParams, EMHistory]:
    """Baum-Welch EM over several sequences (see the module docstring).

    `transition_mask` [N, N] bool fixes the topology: disallowed transitions are 0 throughout. `update`
    selects which of (pi, A, B) are re-estimated. Stops when the objective improves by less than
    tol * (1 + |objective|), or after max_iter iterations. Returns the parameters and the history; the
    history's last entries are those of the returned parameters.
    """
    mask = np.ones_like(init.A, dtype=bool) if transition_mask is None else np.asarray(transition_mask, dtype=bool)
    if mask.shape != init.A.shape or not mask.any(axis=1).all():
        raise InvariantViolation("transition_mask must be [N, N] with at least one allowed transition per row")
    # Restrict the starting transitions to the topology; a row without allowed mass starts uniform over it.
    params = HMMParams(init.pi, _normalise_rows(np.where(mask, init.A, 0.0), mask / mask.sum(axis=1, keepdims=True)), init.B)
    seqs = [np.asarray(s, dtype=np.int64) for s in sequences if len(s) > 0]
    if not seqs:
        raise InvariantViolation("Baum-Welch needs at least one non-empty sequence")
    hist = EMHistory()
    n, m = params.n_states, params.n_symbols
    for it in range(max_iter + 1):
        pi_acc = np.zeros(n)
        a_acc = np.zeros((n, n))
        b_acc = np.zeros((n, m))
        total = 0.0
        for o in seqs:
            gamma, xi_sum, ll = posteriors(params, o)
            total += ll
            pi_acc += gamma[0]
            a_acc += xi_sum
            seen = o >= 0
            np.add.at(b_acc.T, o[seen], gamma[seen])                  # b_acc[i, k] += gamma_t(i) where o_t = k
        obj = total + _log_prior(params, pseudo_pi, pseudo_a, pseudo_b, mask)
        hist.loglik.append(total)
        hist.objective.append(obj)
        if it > 0 and abs(obj - hist.objective[-2]) <= tol * (1.0 + abs(obj)):
            hist.converged = True
            break
        if it == max_iter:
            break
        # M-step (MAP with pseudo-counts).
        new_pi = _normalise_rows(pi_acc + pseudo_pi, params.pi) if update[0] else params.pi
        new_a = _normalise_rows(np.where(mask, a_acc + pseudo_a, 0.0), params.A) if update[1] else params.A
        new_b = _normalise_rows(b_acc + pseudo_b, params.B) if update[2] else params.B
        params = HMMParams(new_pi, new_a, new_b)
    return params, hist


def supervised_estimate(
    sequences: Sequence[np.ndarray],
    states: Sequence[np.ndarray],
    n_states: int,
    n_symbols: int,
    *,
    pseudo: float = 0.0,
    transition_mask: np.ndarray | None = None,
) -> HMMParams:
    """Maximum-likelihood (pseudo = 0) or smoothed estimates from labelled state sequences by counting.

    State code -1 marks an unlabelled step: it contributes no count. A distribution without counts and
    without pseudo-counts is uniform over its allowed entries.
    """
    mask = np.ones((n_states, n_states), dtype=bool) if transition_mask is None else np.asarray(transition_mask, dtype=bool)
    pi_c = np.zeros(n_states)
    a_c = np.zeros((n_states, n_states))
    b_c = np.zeros((n_states, n_symbols))
    for o_raw, s_raw in zip(sequences, states, strict=True):
        o, s = np.asarray(o_raw, dtype=np.int64), np.asarray(s_raw, dtype=np.int64)
        if o.shape != s.shape:
            raise InvariantViolation("each state sequence must match its observation sequence")
        if s.size and s[0] >= 0:
            pi_c[s[0]] += 1
        pair = (s[:-1] >= 0) & (s[1:] >= 0)
        np.add.at(a_c, (s[:-1][pair], s[1:][pair]), 1.0)
        em = (s >= 0) & (o >= 0)
        np.add.at(b_c, (s[em], o[em]), 1.0)
    pi_c = pi_c + pseudo
    a_c = np.where(mask, a_c + pseudo, 0.0)
    b_c = b_c + pseudo
    uniform_a = mask / mask.sum(axis=1, keepdims=True)
    pi = _normalise_rows(pi_c, np.full(n_states, 1.0 / n_states))
    a = _normalise_rows(a_c, uniform_a)
    b = _normalise_rows(b_c, np.full((n_states, n_symbols), 1.0 / n_symbols))
    return HMMParams(pi, a, b)


def initial_params(n_states: int, n_symbols: int, kind: str, rng: np.random.Generator,
                   transition_mask: np.ndarray | None = None) -> HMMParams:
    """Uniform or random (Dirichlet(1)) starting parameters, respecting the transition topology."""
    mask = np.ones((n_states, n_states), dtype=bool) if transition_mask is None else np.asarray(transition_mask, dtype=bool)
    if kind == "uniform":
        pi = np.full(n_states, 1.0 / n_states)
        a = mask / mask.sum(axis=1, keepdims=True)
        b = np.full((n_states, n_symbols), 1.0 / n_symbols)
    elif kind == "random":
        pi = rng.dirichlet(np.ones(n_states))
        a = np.where(mask, rng.dirichlet(np.ones(n_states), size=n_states), 0.0)
        a = a / a.sum(axis=1, keepdims=True)
        b = rng.dirichlet(np.ones(n_symbols), size=n_states)
    else:
        raise ValueError(f"unknown initialisation {kind!r} (uniform or random)")
    return HMMParams(pi, a, b)


def left_to_right_mask(n_states: int, *, max_jump: int | None = None) -> np.ndarray:
    """Upper-triangular topology: a stage can stay or advance (by at most max_jump stages)."""
    i, j = np.indices((n_states, n_states))
    mask = j >= i
    if max_jump is not None:
        mask &= (j - i) <= max_jump
    return mask


def sample(params: HMMParams, length: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Draw (states, observations) of the given length from the model."""
    s = np.empty(length, dtype=np.int64)
    o = np.empty(length, dtype=np.int64)
    for t in range(length):
        p = params.pi if t == 0 else params.A[s[t - 1]]
        s[t] = rng.choice(params.n_states, p=p)
        o[t] = rng.choice(params.n_symbols, p=params.B[s[t]])
    return s, o


def filter_stream(params: HMMParams, obs: np.ndarray | Sequence[int], *, window: int | None = None,
                  chunk: int = 4096) -> np.ndarray:
    """Filtered beliefs [T, N]: row t is P(s_t | o_{t-W+1 ... t}) for a window of W observations.

    Without a window (or W >= T) this is the scaled forward pass, P(s_t | o_1:t). With a window, each row
    restarts from pi at its window's first observation, as a detector that keeps only the last W alerts
    would (Chadza et al. 2020 use W = 150). The windows of all target times are filtered together, `chunk`
    target times at a time.
    """
    o = _obs(obs, params.n_symbols)
    t_len, n = o.size, params.n_states
    if window is None or window >= t_len:
        return forward(params, o)[0] if t_len else np.zeros((0, n))
    if window < 1:
        raise ValueError("window must be >= 1")
    out = np.empty((t_len, n), dtype=np.float64)
    offsets = np.arange(window)[None, :] - (window - 1)                      # [1, W]: t-W+1 ... t relative to t
    for s in range(0, t_len, chunk):
        rows = np.arange(s, min(s + chunk, t_len))
        times = rows[:, None] + offsets                                       # [R, W]
        valid = times >= 0
        sym = np.where(valid, o[np.clip(times, 0, None)], -1)                 # [R, W]
        start = np.argmax(valid, axis=1)                                      # first real column per row
        alpha = np.broadcast_to(params.pi, (rows.size, n)).copy()
        for j in range(window):
            emit = np.ones((rows.size, n))
            seen = sym[:, j] >= 0
            emit[seen] = params.B[:, sym[seen, j]].T
            # Before and at a row's first real observation the prior is pi; afterwards alpha A.
            prior = np.where((j <= start)[:, None], params.pi[None, :], alpha @ params.A)
            a = prior * emit
            tot = a.sum(axis=1, keepdims=True)
            if np.any(tot <= 0):
                raise InvariantViolation("an observation has probability 0 under the model")
            alpha = a / tot
        out[rows] = alpha
    return out
