"""M1 acceptance tests for src/rng.py (LCG uniform, Box-Muller normal, Poisson).

No `numpy.random` is used anywhere in this file (or anywhere under src/,
mechanically enforced by test_no_library_solvers.py) — every random number
here comes from our own `src.rng.LCG`. `numpy` is used only for elementwise
statistics (mean, var, histogram counts) on the resulting arrays.
"""
from __future__ import annotations

import math

import numpy as np

from src.rng import LCG, POISSON_KNUTH_MAX_LAMBDA, derive_seed, poisson, poisson_one, standard_normal

N = 1_000_000

# Chi-square critical values for the goodness-of-fit test below, 10 bins
# (df = 9), standard textbook table (Chapra & Canale, or any stats
# reference) — not computed via scipy.stats, just looked up constants.
CHI2_DF9_CRIT_05 = 16.919   # P(chi2_9 > x) = 0.05
CHI2_DF9_CRIT_001 = 27.877  # P(chi2_9 > x) = 0.001


def test_uniform_mean_variance_vs_theory():
    rng = LCG(seed=123)
    u = rng.uniform_array(N)

    mean = u.mean()
    var = u.var()

    assert abs(mean - 0.5) < 5e-3, f"uniform mean {mean:.6f}, expected ~0.5"
    assert abs(var - 1.0 / 12.0) < 5e-3, f"uniform var {var:.6f}, expected ~{1/12:.6f}"
    assert u.min() >= 0.0 and u.max() < 1.0


def test_uniform_chi_square_goodness_of_fit():
    rng = LCG(seed=123)
    u = rng.uniform_array(N)

    bins = 10
    counts, _edges = np.histogram(u, bins=bins, range=(0.0, 1.0))
    expected = N / bins
    chi2 = float(np.sum((counts - expected) ** 2 / expected))

    # Reported precisely in handoffs/RUN_M1.md alongside the 0.05 critical
    # value above. We assert against the much more lenient 0.001 critical
    # value to keep this test robust to the ordinary sampling variability of
    # a single run while still catching a badly broken generator (a biased
    # or short-period LCG produces chi2 in the hundreds or thousands, not
    # single/double digits).
    assert chi2 < CHI2_DF9_CRIT_001, f"chi2={chi2:.3f} exceeds 0.001 critical value {CHI2_DF9_CRIT_001}"


def test_uniform_seed_reproducibility_and_seed_sensitivity():
    a1 = LCG(seed=42).uniform_array(100)
    a2 = LCG(seed=42).uniform_array(100)
    b = LCG(seed=43).uniform_array(100)

    assert np.array_equal(a1, a2), "same seed must give identical stream"
    assert not np.array_equal(a1, b), "different seeds must give different streams"


def test_derive_seed_is_deterministic_and_distinct():
    s1 = derive_seed(42, "region=frontal", "realisation=0")
    s2 = derive_seed(42, "region=frontal", "realisation=0")
    s3 = derive_seed(42, "region=temporal", "realisation=0")
    assert s1 == s2
    assert s1 != s3


def test_box_muller_mean_variance_vs_theory():
    rng = LCG(seed=123)
    z = standard_normal(rng, N)

    mean = z.mean()
    var = z.var()

    assert abs(mean) < 5e-3, f"normal mean {mean:.6f}, expected ~0"
    assert abs(var - 1.0) < 5e-3, f"normal var {var:.6f}, expected ~1"


def test_box_muller_skewness_and_kurtosis_close_to_gaussian():
    """A weak higher-moment check: catches a Box-Muller implementation that
    gets the magnitude right but the angle/radius formula wrong (e.g. using
    u1 where u2 belongs), which would still pass a bare mean/var check."""
    rng = LCG(seed=99)
    z = standard_normal(rng, N)
    m2 = np.mean(z**2)
    m3 = np.mean(z**3)
    m4 = np.mean(z**4)
    skew = m3 / m2**1.5
    excess_kurtosis = m4 / m2**2 - 3.0

    assert abs(skew) < 0.02, f"skewness {skew:.4f}, expected ~0"
    assert abs(excess_kurtosis) < 0.05, f"excess kurtosis {excess_kurtosis:.4f}, expected ~0"


def test_box_muller_empirical_cdf_close_to_standard_normal_at_quantiles():
    """Compares empirical quantiles to the standard normal CDF at a few
    points, using a hand-rolled CDF via math.erf (no scipy.stats.norm)."""
    rng = LCG(seed=17)
    z = np.sort(standard_normal(rng, N))

    def std_normal_cdf(x: float) -> float:
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    for x in (-2.0, -1.0, 0.0, 1.0, 2.0):
        empirical = float(np.searchsorted(z, x)) / N
        theoretical = std_normal_cdf(x)
        assert abs(empirical - theoretical) < 0.01, (
            f"at x={x}: empirical CDF {empirical:.4f} vs theoretical {theoretical:.4f}"
        )


def test_poisson_mean_equals_variance_equals_lambda():
    for lam in (4.0, 10.0, 25.0):
        rng = LCG(seed=int(1000 + lam))
        n = 200_000
        p = poisson(rng, lam, n)

        mean = p.mean()
        var = p.var()

        # Monte Carlo standard error of the mean is sqrt(lam/n); use a 6-sigma
        # band so this is not flaky but a mean/var bug (e.g. off-by-one in
        # Knuth's algorithm) still fails it clearly.
        se = math.sqrt(lam / n)
        assert abs(mean - lam) < 6 * se, f"lam={lam}: mean {mean:.4f}, expected ~{lam}"
        assert abs(var - lam) < 12 * se, f"lam={lam}: var {var:.4f}, expected ~{lam}"


def test_poisson_zero_lambda_is_always_zero():
    rng = LCG(seed=5)
    for _ in range(1000):
        assert poisson_one(rng, 0.0) == 0


def test_poisson_large_lambda_uses_normal_approximation_branch_correctly():
    """DECISIONS.md D-M1-10: above POISSON_KNUTH_MAX_LAMBDA=30, poisson_one
    switches to a normal-approximation branch. Verified here at lambda=100
    and lambda=1000 — both are well above the crossover, and lambda=1000 is
    close to where Knuth's algorithm (if used unguarded) silently returns
    wrong, roughly lambda-independent values around 700-800 (measured, not
    the 'hang' originally hypothesised — see D-M1-10)."""
    for lam in (100.0, 1000.0):
        rng = LCG(seed=int(lam) + 7)
        n = 200_000
        p = poisson(rng, lam, n)

        mean = p.mean()
        var = p.var()
        se_mean = math.sqrt(lam / n)
        # Monte Carlo SE of the sample VARIANCE (not the mean): for a
        # Poisson(lam), Var(sample variance) ~= (mu4 - sigma^4)/n with
        # mu4 = lam + 3*lam^2, sigma^2 = lam, giving SE ~= lam*sqrt(2/n) for
        # large lam (the leading term dominates; matches the near-Gaussian
        # regime this large-lambda branch operates in).
        se_var = lam * math.sqrt(2.0 / n)

        assert abs(mean - lam) < 6 * se_mean, f"lam={lam}: mean {mean:.4f}, expected ~{lam}"
        assert abs(var - lam) < 6 * se_var, f"lam={lam}: var {var:.4f}, expected ~{lam}"
        # The specific silent-failure signature of the unguarded Knuth loop:
        # a result clustered around 700-900 regardless of lambda. Confirms
        # we are NOT accidentally hitting that path.
        assert not (650 < mean < 900 and lam >= 1000), (
            f"lam={lam}: mean {mean:.2f} looks like the unguarded-Knuth failure signature"
        )


def test_poisson_knuth_and_normal_approx_branches_agree_near_crossover():
    """Compares the exact Knuth branch (lambda=25, forced) against the
    normal-approximation branch (same lambda=25, force_normal_approx=True):
    both should have mean/variance ~25, agreeing with each other within
    Monte Carlo error — a two-sample check, not a full KS test (DECISIONS.md
    D-M1-10 explicitly keeps this lightweight)."""
    lam = 25.0
    assert lam <= POISSON_KNUTH_MAX_LAMBDA, "test assumes lam is on the exact-Knuth side"
    n = 200_000

    rng_knuth = LCG(seed=501)
    p_knuth = poisson(rng_knuth, lam, n, force_normal_approx=False)

    rng_normal = LCG(seed=502)
    p_normal = poisson(rng_normal, lam, n, force_normal_approx=True)

    se = math.sqrt(lam / n)
    assert abs(p_knuth.mean() - lam) < 6 * se
    assert abs(p_normal.mean() - lam) < 6 * se
    assert abs(p_knuth.mean() - p_normal.mean()) < 8 * se, (
        f"branches disagree on mean: knuth={p_knuth.mean():.4f} vs normal_approx={p_normal.mean():.4f}"
    )
    assert abs(p_knuth.var() - p_normal.var()) < 0.15 * lam, (
        f"branches disagree on variance: knuth={p_knuth.var():.4f} vs normal_approx={p_normal.var():.4f}"
    )


def test_poisson_matches_hand_computed_pmf_at_small_lambda():
    """lambda=1: P(X=0) = e^-1, P(X=1) = e^-1, P(X=2) = e^-1/2 (hand-derived
    from the Poisson pmf) — direct check the sampler's distribution shape is
    right, not just its first two moments."""
    rng = LCG(seed=321)
    n = 500_000
    lam = 1.0
    p = poisson(rng, lam, n)

    for k, expected_p in ((0, math.exp(-1)), (1, math.exp(-1)), (2, math.exp(-1) / 2)):
        empirical_p = np.mean(p == k)
        se = math.sqrt(expected_p * (1 - expected_p) / n)
        assert abs(empirical_p - expected_p) < 6 * se, (
            f"P(X={k}): empirical {empirical_p:.5f} vs theory {expected_p:.5f}"
        )
