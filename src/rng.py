"""Track A random number generation.

The project brief: "our own uniform generator (LCG or Mersenne-style) plus
our own Box-Muller normal and our own Poisson sampler, all explicitly seeded."
No `numpy.random` symbol is used anywhere in this file (enforced mechanically
by tests/test_no_library_solvers.py).

Three layers:
  - `LCG`: 64-bit linear congruential generator, the sole source of raw
    randomness for everything else in this module (see DECISIONS.md D-M1-1).
  - `standard_normal`: Box-Muller transform on top of `LCG` (D-M1-2).
  - `poisson`: Knuth's multiplication algorithm on top of `LCG` (D-M1-3).
  - `derive_seed`: deterministic sub-seed derivation for reproducible,
    independent streams across regions/realisations in later milestones
    (M0 "seeded RNG plumbing").
"""
from __future__ import annotations

import hashlib
import math

import numpy as np

MASK64 = (1 << 64) - 1

# Knuth/MMIX LCG parameters: state_{n+1} = (A * state_n + C) mod 2**64.
_A = 6364136223846793005
_C = 1442695040888963407


class LCG:
    """64-bit linear congruential generator.

    Only the upper bits of the state are exposed as randomness (the low bits
    of any LCG have short periods / visible patterns — see DECISIONS.md
    D-M1-1), so `uniform()` is built from the top 53 bits of the state, which
    matches a float64 mantissa exactly.
    """

    def __init__(self, seed: int):
        if seed < 0:
            raise ValueError("seed must be a non-negative integer")
        # Mix the seed once so that seed=0, seed=1, ... do not start in
        # obviously related states (splitmix64-style avalanche of the seed
        # before it becomes the LCG state).
        self.state = _splitmix64(seed & MASK64)

    def next_uint64(self) -> int:
        self.state = (_A * self.state + _C) & MASK64
        return self.state

    def uniform(self) -> float:
        """One draw, uniform on [0, 1), using the top 53 bits of the state."""
        raw = self.next_uint64()
        return (raw >> 11) * (1.0 / (1 << 53))

    def _uniform_open01(self) -> float:
        """One draw, uniform on the OPEN interval (0, 1) — never exactly 0 or 1.

        Used internally wherever a value feeds into log() (Box-Muller) where
        a literal 0.0 would blow up. See DECISIONS.md D-M1-2.
        """
        raw = self.next_uint64()
        return (raw + 1.0) / (float(MASK64) + 3.0)

    def uniform_array(self, n: int) -> np.ndarray:
        """n iid draws on [0, 1), as a numpy array (storage only, no library RNG)."""
        out = np.empty(n, dtype=np.float64)
        for i in range(n):
            out[i] = self.uniform()
        return out

    def randint_below(self, bound: int) -> int:
        """Uniform integer in [0, bound), via rejection to avoid modulo bias."""
        if bound <= 0:
            raise ValueError("bound must be positive")
        limit = (MASK64 + 1) - ((MASK64 + 1) % bound)
        while True:
            raw = self.next_uint64()
            if raw < limit:
                return raw % bound


def _splitmix64(x: int) -> int:
    """Deterministic 64-bit avalanche mix, used to seed LCG and to derive
    sub-seeds in `derive_seed`. Pure arithmetic, independent of Python's
    salted `hash()`, so it is identical across processes and runs."""
    x = (x + 0x9E3779B97F4A7C15) & MASK64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & MASK64
    x = x ^ (x >> 31)
    return x & MASK64


def derive_seed(root_seed: int, *labels: str) -> int:
    """Deterministically derive a sub-seed from a root seed and string labels.

    Used to give independent-looking, reproducible seeds to different parts
    of a Monte Carlo study (e.g. `derive_seed(42, "region=frontal", "realisation=7")`)
    without any global RNG state. Two calls with the same arguments, in the
    same or a different process, always return the same integer (M0
    acceptance: "Seeded RNG returns identical streams across processes").
    """
    payload = str(root_seed) + "|" + "|".join(labels)
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big") & MASK64


# ---------------------------------------------------------------------------
# Box-Muller normal (DECISIONS.md D-M1-2)
# ---------------------------------------------------------------------------
def standard_normal(rng: LCG, n: int) -> np.ndarray:
    """n iid draws from N(0, 1) via the basic Box-Muller transform.

    Consumes two uniforms per pair of normals:
        u1 ~ Unif(0, 1) (open, to keep log(u1) finite)
        u2 ~ Unif[0, 1)
        r = sqrt(-2 ln u1)
        z0 = r cos(2 pi u2), z1 = r sin(2 pi u2)
    z0, z1 are independent standard normals (standard textbook derivation via
    the Jacobian of the polar transform of two independent N(0,1)'s).
    """
    n_pairs = (n + 1) // 2
    out = np.empty(2 * n_pairs, dtype=np.float64)
    for i in range(n_pairs):
        u1 = rng._uniform_open01()
        u2 = rng.uniform()
        r = math.sqrt(-2.0 * math.log(u1))
        theta = 2.0 * math.pi * u2
        out[2 * i] = r * math.cos(theta)
        out[2 * i + 1] = r * math.sin(theta)
    return out[:n]


def normal(rng: LCG, n: int, mean: float = 0.0, std: float = 1.0) -> np.ndarray:
    """n iid draws from N(mean, std**2), built on `standard_normal`."""
    if std < 0:
        raise ValueError("std must be non-negative")
    z = standard_normal(rng, n)
    return mean + std * z


# ---------------------------------------------------------------------------
# Poisson sampler (DECISIONS.md D-M1-3, D-M1-10)
# ---------------------------------------------------------------------------
# Above this lambda, `poisson_one` switches from exact Knuth to a normal
# approximation (DECISIONS.md D-M1-10). Knuth's algorithm compares a running
# product of uniforms (each in (0,1)) against exp(-lambda); in float64,
# exp(-lambda) underflows to EXACTLY 0.0 for lambda >~ 745, and the loop's
# termination test `p <= L` can then never be satisfied (p stays > 0 forever)
# — an infinite loop, not a slowdown. 30 is chosen well below that point
# because (a) the normal approximation to Poisson is already accurate there
# (skewness 1/sqrt(30) ~ 0.18) and (b) M4's realistic photon-count regime is
# expected to need lambda in the hundreds to thousands, so there is no reason
# to run Knuth's O(lambda) loop anywhere near its hang point.
POISSON_KNUTH_MAX_LAMBDA = 30.0


def poisson_one(rng: LCG, lam: float, *, force_normal_approx: bool = False) -> int:
    """One draw from Poisson(lam).

    For `lam <= POISSON_KNUTH_MAX_LAMBDA`: Knuth's multiplication algorithm,
    exact (not an approximation) — simulates the arrival-time process
    directly from the definition of the Poisson distribution as the count of
    unit-rate Exponential inter-arrivals falling within [0, lam]. Cost is
    O(lam) uniforms per draw on average.

    For `lam > POISSON_KNUTH_MAX_LAMBDA`: a normal approximation,
    `round(max(0, N(lam, lam)))`, built on `standard_normal` (Box-Muller).
    This is an approximation (not exact), acceptable here because lambda this
    large only arises from noise-level calibration in M4 (never from an exact
    small-count regime where the approximation would matter).

    `force_normal_approx=True` bypasses the crossover to use the normal
    branch even for small lambda, purely so tests can compare both branches
    at the same lambda near the crossover (DECISIONS.md D-M1-10).
    """
    if lam < 0:
        raise ValueError("lam must be non-negative")

    if lam > POISSON_KNUTH_MAX_LAMBDA or force_normal_approx:
        z = standard_normal(rng, 1)[0]
        return max(0, int(round(lam + math.sqrt(lam) * z)))

    L = math.exp(-lam)
    k = 0
    p = 1.0
    while True:
        k += 1
        p *= rng.uniform()
        if p <= L:
            return k - 1


def poisson(rng: LCG, lam: float, n: int, *, force_normal_approx: bool = False) -> np.ndarray:
    """n iid draws from Poisson(lam). See `poisson_one` for the two branches."""
    out = np.empty(n, dtype=np.int64)
    for i in range(n):
        out[i] = poisson_one(rng, lam, force_normal_approx=force_normal_approx)
    return out


if __name__ == "__main__":
    rng = LCG(seed=42)
    u = rng.uniform_array(5)
    print("first 5 uniforms, seed=42:", u)
    rng2 = LCG(seed=42)
    z = standard_normal(rng2, 4)
    print("first 4 normals, seed=42:", z)
    rng3 = LCG(seed=42)
    pois = poisson(rng3, lam=5.0, n=10)
    print("first 10 poisson(5), seed=42:", pois)
