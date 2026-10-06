"""Differential-evolution training of HMM parameters (AS-550).

Chadza, Kyriakopoulos and Lambotharan (IEEE Access 8, 2020) compare two training techniques, "BW" and
"DE", each with uniform, random and count-based starting points (baselines-notes.md, C2). BW is
Baum-Welch; DE is read here as differential evolution (Storn and Price, Journal of Global Optimization 11,
1997), a population-based maximiser of the log-likelihood that does not follow EM's local monotone path
(citation to verify against the paper).

Parameterisation: unconstrained logits theta = (theta_pi [N], theta_A [N, N], theta_B [N, M]), mapped to
the model by row-wise softmax; transitions outside the topology mask have logit -inf (probability 0).

DE/rand/1/bin, per generation and population member i:
    v = theta_r1 + F (theta_r2 - theta_r3)                      r1, r2, r3 distinct and != i
    u_j = v_j if U_j < CR or j = j_rand, else theta_i,j           binomial crossover
    theta_i <- u if f(u) >= f(theta_i)                           f = log P(training sequences)
The selection is elitist, so the best fitness of the population never decreases.

Starting population: member 0 is the starting model (uniform, random or counted), the others are its
logits plus Gaussian perturbations of scale `spread`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from nagahana.baselines.published.hmm.core import HMMParams, loglik
from nagahana.core.errors import InvariantViolation


def _softmax_rows(z: np.ndarray) -> np.ndarray:
    m = np.max(z, axis=-1, keepdims=True)
    e = np.exp(z - np.where(np.isfinite(m), m, 0.0))
    return e / e.sum(axis=-1, keepdims=True)


@dataclass
class DEResult:
    """Best parameters and the best fitness per generation (non-decreasing)."""

    params: HMMParams
    best_fitness: list[float] = field(default_factory=list)


def _pack(params: HMMParams, floor: float = 1e-6) -> np.ndarray:
    # Logits of a model: log of the (floored) probabilities; masked transitions stay -inf via the mask.
    return np.concatenate([np.log(np.maximum(params.pi, floor)), np.log(np.maximum(params.A, floor)).ravel(),
                           np.log(np.maximum(params.B, floor)).ravel()])


def _unpack(theta: np.ndarray, n: int, m: int, mask: np.ndarray) -> HMMParams:
    pi = _softmax_rows(theta[:n])
    a_logits = np.where(mask, theta[n:n + n * n].reshape(n, n), -np.inf)
    b = _softmax_rows(theta[n + n * n:].reshape(n, m))
    return HMMParams(pi, _softmax_rows(a_logits), b)


def _fitness(theta: np.ndarray, seqs: Sequence[np.ndarray], n: int, m: int, mask: np.ndarray) -> float:
    try:
        p = _unpack(theta, n, m, mask)
        return float(sum(loglik(p, s) for s in seqs))
    except InvariantViolation:
        return -np.inf


def differential_evolution(
    sequences: Sequence[np.ndarray],
    start: HMMParams,
    rng: np.random.Generator,
    *,
    population: int = 20,
    generations: int = 50,
    mutation: float = 0.5,
    crossover: float = 0.9,
    spread: float = 0.5,
    transition_mask: np.ndarray | None = None,
) -> DEResult:
    """Maximise the log-likelihood of `sequences` over HMM parameters by DE/rand/1/bin from `start`."""
    if population < 4:
        raise ValueError("DE/rand/1 needs a population of at least 4")
    if not (0.0 < mutation <= 2.0 and 0.0 <= crossover <= 1.0):
        raise ValueError("mutation must lie in (0, 2] and crossover in [0, 1]")
    seqs = [np.asarray(s, dtype=np.int64) for s in sequences if len(s) > 0]
    if not seqs:
        raise InvariantViolation("differential evolution needs at least one non-empty sequence")
    n, m = start.n_states, start.n_symbols
    mask = np.ones((n, n), dtype=bool) if transition_mask is None else np.asarray(transition_mask, dtype=bool)
    base = _pack(start)
    dim = base.size
    pop = base[None, :] + spread * rng.standard_normal((population, dim))
    pop[0] = base
    fit = np.asarray([_fitness(p, seqs, n, m, mask) for p in pop])
    result = DEResult(params=_unpack(pop[int(np.argmax(fit))], n, m, mask), best_fitness=[float(fit.max())])
    for _gen in range(generations):
        for i in range(population):
            r1, r2, r3 = rng.choice(np.delete(np.arange(population), i), size=3, replace=False)
            v = pop[r1] + mutation * (pop[r2] - pop[r3])
            cross = rng.random(dim) < crossover
            cross[rng.integers(dim)] = True
            u = np.where(cross, v, pop[i])
            fu = _fitness(u, seqs, n, m, mask)
            if fu >= fit[i]:
                pop[i], fit[i] = u, fu
        result.best_fitness.append(float(fit.max()))
    result.params = _unpack(pop[int(np.argmax(fit))], n, m, mask)
    return result
