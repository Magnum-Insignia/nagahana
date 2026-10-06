"""Counter-based random numbers shared by the NumPy and JAX backends (P-14).

A sequential generator makes a world depend on the order draws are consumed in, so a vectorised run
(vmap over worlds, scan over event slots) and a loop over worlds would disagree. A counter-based
generator is a keyed function f_k(c) of a counter c: every draw is addressed by (key, counter) and
does not depend on what else was drawn, on how many worlds run together, or on the backend (AS-801).

Generator: Threefry-2x32 with 20 rounds (Salmon, Moraes, Dror, Shaw, SC 2011,
DOI 10.1145/2063384.2063405), the design behind jax.random. One call maps a 64-bit key (k0, k1) and
a 64-bit counter (c0, c1) to two 32-bit words. All arithmetic is uint32 modulo 2^32; NumPy and XLA
implement add, xor, or and shift on uint32 identically, so the same source under xp = numpy or
xp = jax.numpy yields the same bits. `KNOWN_ANSWERS` are the Random123 reference vectors; `self_test`
checks them.

Key hierarchy (one Threefry call per level; tags give domain separation):
    seed (< 2^64)          seed_key(seed) = (seed mod 2^32, seed // 2^32)
    world w                derive(seed_key, w, TAG_WORLD)
    stream s of a world     derive(world_key, s, TAG_STREAM)
    draw index j, lane l   threefry(stream_key, (j, l)) -> 64 bits

From bits to numbers:
    to_unit(w0, w1)   float64 in [0, 1) with 53 bits, u = (floor(w0/2^5)*2^26 + floor(w1/2^6))*2^-53;
                      every step is exact in double precision, so to_unit is bit-identical.
    to_int63(w0, w1)  integer in [0, 2^63).
    below(w0, w1, n)  to_int63 mod n, in [0, n); modulo bias <= n/2^63 (< 1.1e-10 for n <= 1e9).
    threshold_u32(p)  integer t so that "w0 < t" happens with probability p; a Bernoulli decision is
                      then an integer compare, bit-identical across backends (AS-802).
Floating transforms (log, exp, cos) appear only in magnitudes (`exponential`, `normal`, `lognormal`,
`bounded_pareto`), never in a discrete decision or in event ordering.

JAX callers run under jax.enable_x64(True) so int64 and float64 match NumPy (AS-819).
"""

from __future__ import annotations

import contextlib
from typing import Any

import numpy as np

MASK32 = 0xFFFFFFFF
ROTATIONS: tuple[int, ...] = (13, 15, 26, 6, 17, 29, 16, 24)
PARITY32 = 0x1BD11BDA
ROUNDS = 20

#: Random123 known-answer vectors: (key0, key1, ctr0, ctr1) -> (out0, out1).
KNOWN_ANSWERS: tuple[tuple[tuple[int, int, int, int], tuple[int, int]], ...] = (
    ((0x00000000, 0x00000000, 0x00000000, 0x00000000), (0x6B200159, 0x99BA4EFE)),
    ((0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF), (0x1CB996FC, 0xBB002BE7)),
    ((0x13198A2E, 0x03707344, 0x243F6A88, 0x85A308D3), (0xC4923A9C, 0x483DF7A0)),
)

TAG_WORLD = 0x57524C44
TAG_STREAM = 0x5354524D

_TWO_M53 = 1.0 / 9007199254740992.0            # 2^-53
_POW2_26 = 67108864.0                          # 2^26
_TWO63 = 1 << 63

Array = Any


def _u32(xp: Any, value: Any) -> Array:
    # A uint32 array in the given namespace.
    return xp.asarray(value, dtype=xp.uint32)


def _rotl32(xp: Any, x: Array, r: int) -> Array:
    # Rotate a uint32 word left by r bits (0 < r < 32).
    left = (x << np.uint32(r)) & xp.asarray(MASK32, dtype=xp.uint32)
    right = x >> np.uint32(32 - r)
    return (left | right).astype(xp.uint32)


def threefry(xp: Any, key: tuple[Array, Array], ctr: tuple[Array, Array]) -> tuple[Array, Array]:
    """Threefry-2x32-20 of a counter under a key.

    Parameters
    ----------
    xp:
        numpy or jax.numpy.
    key, ctr:
        pairs of uint32 arrays (broadcastable). key is (k0, k1); ctr is (c0, c1).

    Returns
    -------
    (out0, out1):
        two uint32 arrays, the 64-bit output.
    """
    # Addition wraps modulo 2^32 by design; silence NumPy's scalar-overflow warning for that wrap.
    guard = np.errstate(over="ignore") if xp is np else contextlib.nullcontext()
    with guard:
        k0 = key[0].astype(xp.uint32)
        k1 = key[1].astype(xp.uint32)
        k2 = (k0 ^ k1 ^ xp.asarray(PARITY32, dtype=xp.uint32)).astype(xp.uint32)
        ks = (k0, k1, k2)
        x0 = (ctr[0].astype(xp.uint32) + k0).astype(xp.uint32)
        x1 = (ctr[1].astype(xp.uint32) + k1).astype(xp.uint32)
        injection = 0
        for i in range(ROUNDS):
            x0 = (x0 + x1).astype(xp.uint32)
            x1 = (_rotl32(xp, x1, ROTATIONS[i % 8]) ^ x0).astype(xp.uint32)
            if i % 4 == 3:
                injection += 1
                s = injection
                x0 = (x0 + ks[s % 3]).astype(xp.uint32)
                x1 = (x1 + ks[(s + 1) % 3] + xp.asarray(s, dtype=xp.uint32)).astype(xp.uint32)
        return x0.astype(xp.uint32), x1.astype(xp.uint32)


def seed_key(xp: Any, seed: int) -> tuple[Array, Array]:
    """Split a 64-bit seed into a (k0, k1) key of uint32 scalars."""
    seed &= (1 << 64) - 1
    return _u32(xp, seed & MASK32), _u32(xp, (seed >> 32) & MASK32)


def derive(xp: Any, key: tuple[Array, Array], index: Array | int, tag: int) -> tuple[Array, Array]:
    """A child key of `key` at `index`, separated from other uses by `tag`."""
    ctr = (_u32(xp, index), _u32(xp, tag & MASK32))
    return threefry(xp, key, ctr)


def world_key(xp: Any, seed: int, world: Array | int) -> tuple[Array, Array]:
    """Key of world `world` of a run with the given seed."""
    return derive(xp, seed_key(xp, seed), world, TAG_WORLD)


def stream_key(xp: Any, wkey: tuple[Array, Array], stream: int) -> tuple[Array, Array]:
    """Key of stream `stream` of a world (one stream per purpose; see the stream catalogue)."""
    return derive(xp, wkey, stream, TAG_STREAM)


def bits(xp: Any, key: tuple[Array, Array], index: Array | int, lane: Array | int) -> tuple[Array, Array]:
    """The 64-bit output at draw `index`, lane `lane` of a stream."""
    return threefry(xp, key, (_u32(xp, index), _u32(xp, lane)))


def to_unit(xp: Any, w0: Array, w1: Array) -> Array:
    """A float64 in [0, 1) with 53 random bits (exact on every backend; see the module docstring)."""
    hi = (w0.astype(xp.uint32) >> np.uint32(5)).astype(xp.float64)
    lo = (w1.astype(xp.uint32) >> np.uint32(6)).astype(xp.float64)
    return (hi * _POW2_26 + lo) * _TWO_M53


def to_int63(xp: Any, w0: Array, w1: Array) -> Array:
    """An int64 in [0, 2^63): 31 bits of w0 as the high part, 32 bits of w1 as the low part."""
    hi = (w0.astype(xp.uint32) >> np.uint32(1)).astype(xp.int64)        # 31 bits
    lo = w1.astype(xp.uint32).astype(xp.int64)                          # 32 bits
    return (hi << xp.asarray(32, dtype=xp.int64)) | lo


def below(xp: Any, w0: Array, w1: Array, n: Array | int) -> Array:
    """An int64 in [0, n) from 63 bits (modulo bias <= n / 2^63)."""
    return to_int63(xp, w0, w1) % xp.asarray(n, dtype=xp.int64)


def threshold_u32(p: float) -> int:
    """Integer t in [0, 2^32] so that a uniform uint32 is < t with probability p (clamped to [0, 1])."""
    p = min(max(float(p), 0.0), 1.0)
    return int(np.floor(p * 4294967296.0 + 0.5))


def threshold_lt(xp: Any, w0: Array, threshold: Array | int) -> Array:
    """Compare a uint32 word to an integer threshold in [0, 2^32].

    The comparison is done in int64 so that the "always true" threshold 2^32 is representable (a
    uint32 cannot hold it). The outcome is a bit-identical integer decision across backends (AS-802).
    """
    return w0.astype(xp.int64) < xp.asarray(threshold, dtype=xp.int64)


def unit(xp: Any, key: tuple[Array, Array], index: Array | int, lane: Array | int) -> Array:
    """A float64 uniform in [0, 1) at (index, lane)."""
    w0, w1 = bits(xp, key, index, lane)
    return to_unit(xp, w0, w1)


def randint(xp: Any, key: tuple[Array, Array], index: Array | int, lane: Array | int, n: Array | int) -> Array:
    """An int64 uniform in [0, n) at (index, lane)."""
    w0, w1 = bits(xp, key, index, lane)
    return below(xp, w0, w1, n)


def bernoulli(xp: Any, key: tuple[Array, Array], index: Array | int, lane: Array | int, threshold: int) -> Array:
    """A boolean that is True with the probability `threshold` encodes (`threshold_u32`).

    The comparison is on uint32, so the outcome is identical on every backend (AS-802).
    """
    w0, _ = bits(xp, key, index, lane)
    return threshold_lt(xp, w0, threshold)


def exponential(xp: Any, key: tuple[Array, Array], index: Array | int, lane: Array | int, rate: Array | float) -> Array:
    """An Exponential(rate) sample (seconds): -log(1 - u) / rate, u uniform in [0, 1)."""
    u = unit(xp, key, index, lane)
    rate_a = xp.asarray(rate, dtype=xp.float64)
    return -xp.log1p(-u) / rate_a


def normal(xp: Any, key: tuple[Array, Array], index: Array | int, lane: Array | int) -> Array:
    """A standard normal via the Box-Muller transform (uses lanes `lane` and `lane + 1`)."""
    u1 = unit(xp, key, index, lane)
    u2 = unit(xp, key, index, (xp.asarray(lane, dtype=xp.int64) + 1) if not isinstance(lane, int) else lane + 1)
    # 1 - u1 keeps the logarithm argument away from 0.
    radius = xp.sqrt(-2.0 * xp.log1p(-u1))
    return radius * xp.cos(2.0 * np.pi * u2)


def lognormal(
    xp: Any, key: tuple[Array, Array], index: Array | int, lane: Array | int, mu: float, sigma: float
) -> Array:
    """A lognormal sample exp(mu + sigma * z), z standard normal (heavy-tailed flow sizes, AS-804)."""
    z = normal(xp, key, index, lane)
    return xp.exp(mu + sigma * z)


def bounded_pareto(
    xp: Any, key: tuple[Array, Array], index: Array | int, lane: Array | int,
    alpha: float, low: float, high: float,
) -> Array:
    """A bounded-Pareto sample in [low, high] with tail index alpha (inverse CDF; AS-804, AS-814).

    x = (low^-a - u (low^-a - high^-a))^(-1/a). Heavy-tailed sizes with a finite maximum keep the
    physics byte and packet bounds (physics/residuals.py) satisfiable.
    """
    u = unit(xp, key, index, lane)
    la = low ** (-alpha)
    ha = high ** (-alpha)
    base = la - u * (la - ha)
    return base ** (-1.0 / alpha)


def self_test() -> None:
    """Check the Random123 known-answer vectors (raises AssertionError on a mismatch)."""
    for (k0, k1, c0, c1), (o0, o1) in KNOWN_ANSWERS:
        key = (_u32(np, k0), _u32(np, k1))
        ctr = (_u32(np, c0), _u32(np, c1))
        out0, out1 = threefry(np, key, ctr)
        if (int(out0), int(out1)) != (o0, o1):
            raise AssertionError(
                f"Threefry-2x32-20 mismatch for key=({k0:#x},{k1:#x}) ctr=({c0:#x},{c1:#x}): "
                f"got ({int(out0):#x},{int(out1):#x}), expected ({o0:#x},{o1:#x})"
            )


__all__ = [
    "KNOWN_ANSWERS", "MASK32", "PARITY32", "ROTATIONS", "ROUNDS", "TAG_STREAM", "TAG_WORLD",
    "below", "bernoulli", "bits", "bounded_pareto", "derive", "exponential", "lognormal", "normal",
    "randint", "seed_key", "self_test", "stream_key", "threefry", "threshold_lt", "threshold_u32", "to_int63",
    "to_unit", "unit", "world_key",
]
