"""M1 benchmark: statistical validation of src/rng.py (LCG, Box-Muller, Poisson).

No `numpy.random` anywhere (mechanically enforced under src/, and simply not
used here either — every random draw comes from `src.rng.LCG`). `numpy` and
`math` are used only to compute summary statistics of the generated arrays.

Run: `python3 experiments/m1_rng_benchmark.py`
Writes: results/m1/rng_benchmark.json, results/m1/rng_histograms.png
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments._common import save_json
from src.plotting import save_fig, set_style
from src.rng import LCG, poisson, standard_normal

N = 1_000_000
SEED_UNIFORM = 123
SEED_NORMAL = 123
SEED_POISSON_BASE = 1000

CHI2_DF9_CRIT_05 = 16.919
CHI2_DF9_CRIT_001 = 27.877


def std_normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def uniform_diagnostics():
    rng = LCG(seed=SEED_UNIFORM)
    u = rng.uniform_array(N)

    bins = 10
    counts, edges = np.histogram(u, bins=bins, range=(0.0, 1.0))
    expected = N / bins
    chi2 = float(np.sum((counts - expected) ** 2 / expected))

    return {
        "n": N,
        "seed": SEED_UNIFORM,
        "mean": float(u.mean()),
        "mean_theory": 0.5,
        "variance": float(u.var()),
        "variance_theory": 1.0 / 12.0,
        "min": float(u.min()),
        "max": float(u.max()),
        "chi2_stat_10bins_df9": chi2,
        "chi2_crit_alpha_0.05": CHI2_DF9_CRIT_05,
        "chi2_crit_alpha_0.001": CHI2_DF9_CRIT_001,
        "bin_counts": counts.tolist(),
        "expected_count_per_bin": expected,
    }, u


def normal_diagnostics():
    rng = LCG(seed=SEED_NORMAL)
    z = standard_normal(rng, N)

    m2 = float(np.mean(z**2))
    m3 = float(np.mean(z**3))
    m4 = float(np.mean(z**4))
    skew = m3 / m2**1.5
    excess_kurtosis = m4 / m2**2 - 3.0

    zs = np.sort(z)
    ks_points = np.linspace(-3, 3, 61)
    max_abs_cdf_diff = 0.0
    for x in ks_points:
        empirical = float(np.searchsorted(zs, x)) / N
        theoretical = std_normal_cdf(x)
        max_abs_cdf_diff = max(max_abs_cdf_diff, abs(empirical - theoretical))

    return {
        "n": N,
        "seed": SEED_NORMAL,
        "mean": float(z.mean()),
        "variance": float(z.var()),
        "skewness": skew,
        "excess_kurtosis": excess_kurtosis,
        "max_abs_empirical_cdf_diff_over_grid": max_abs_cdf_diff,
    }, z


def poisson_diagnostics():
    results = {}
    n = 200_000
    # 4, 10, 25: exact Knuth branch. 100, 1000: normal-approximation branch
    # (DECISIONS.md D-M1-10). Both must show mean ~= var ~= lambda.
    for lam in (4.0, 10.0, 25.0, 100.0, 1000.0):
        rng = LCG(seed=SEED_POISSON_BASE + int(lam))
        p = poisson(rng, lam, n)
        se_mean = math.sqrt(lam / n)
        se_var = lam * math.sqrt(2.0 / n)  # see tests/test_rng.py for the derivation
        results[str(lam)] = {
            "n": n,
            "lambda": lam,
            "branch": "knuth" if lam <= 30.0 else "normal_approx",
            "mean": float(p.mean()),
            "variance": float(p.var()),
            "mc_standard_error_mean": se_mean,
            "mc_standard_error_variance": se_var,
        }
    # PMF shape check at lambda=1
    rng = LCG(seed=321)
    n2 = 500_000
    p1 = poisson(rng, 1.0, n2)
    pmf_check = {}
    for k, theory in ((0, math.exp(-1)), (1, math.exp(-1)), (2, math.exp(-1) / 2)):
        empirical = float(np.mean(p1 == k))
        pmf_check[str(k)] = {"empirical": empirical, "theory": theory}
    results["pmf_check_lambda_1"] = {"n": n2, "values": pmf_check}
    return results


def lcg_neave_effect_diagnostics(u: np.ndarray, z: np.ndarray):
    """DECISIONS.md D-M1-12: lag-1..5 autocorrelation of the raw uniform
    stream, plus 2-D scatter data for consecutive uniform pairs and
    consecutive Box-Muller normal pairs, specifically to check for the
    Neave effect (Box-Muller fed by consecutive LCG outputs producing
    normals on visible spirals/lattice planes)."""
    n = u.shape[0]
    ubar = u.mean()
    denom = float(np.sum((u - ubar) ** 2))
    autocorr = {}
    for lag in (1, 2, 3, 4, 5):
        num = float(np.sum((u[: n - lag] - ubar) * (u[lag:] - ubar)))
        autocorr[str(lag)] = num / denom
    # Under the null (iid), autocorrelation at lag k ~ N(0, 1/n) for large n.
    se = 1.0 / math.sqrt(n)
    return {"n": n, "autocorrelation_by_lag": autocorr, "se_under_iid_null": se}


def make_figure(u: np.ndarray, z: np.ndarray):
    import matplotlib.pyplot as plt

    set_style()
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    axes[0, 0].hist(u, bins=50, density=True, alpha=0.75, color="tab:blue")
    axes[0, 0].axhline(1.0, color="k", linestyle="--", linewidth=1, label="theory: Unif(0,1)")
    axes[0, 0].set_title(f"LCG uniform, n={N:,}")
    axes[0, 0].set_xlabel("u")
    axes[0, 0].legend()

    axes[0, 1].hist(z, bins=80, density=True, alpha=0.75, color="tab:orange")
    xs = np.linspace(-4, 4, 400)
    axes[0, 1].plot(xs, np.exp(-0.5 * xs**2) / math.sqrt(2 * math.pi), "k--", linewidth=1, label="theory: N(0,1)")
    axes[0, 1].set_title(f"Box-Muller normal, n={N:,}")
    axes[0, 1].set_xlabel("z")
    axes[0, 1].legend()

    # Neave-effect check (DECISIONS.md D-M1-12): a subset for a legible
    # scatter — banding/spiraling, if present, is visible with a few
    # thousand points and gets *harder* to see, not easier, with more (just
    # a denser fill), so a subset is the right choice here, not a shortcut.
    n_scatter = 5000
    axes[1, 0].scatter(u[:n_scatter], u[1 : n_scatter + 1], s=2, alpha=0.4, color="tab:blue")
    axes[1, 0].set_title(f"consecutive uniform pairs $(u_i, u_{{i+1}})$, n={n_scatter:,}")
    axes[1, 0].set_xlabel("$u_i$")
    axes[1, 0].set_ylabel("$u_{i+1}$")

    axes[1, 1].scatter(z[:n_scatter], z[1 : n_scatter + 1], s=2, alpha=0.4, color="tab:orange")
    axes[1, 1].set_title(f"consecutive Box-Muller pairs $(z_i, z_{{i+1}})$, n={n_scatter:,}")
    axes[1, 1].set_xlabel("$z_i$")
    axes[1, 1].set_ylabel("$z_{i+1}$")

    fig.tight_layout()
    return fig


if __name__ == "__main__":
    print("Uniform diagnostics...")
    uni, u = uniform_diagnostics()
    print("  mean=%.6f (theory 0.5), var=%.6f (theory %.6f)" % (uni["mean"], uni["variance"], 1 / 12))
    print("  chi2(df=9) = %.3f  [crit@0.05=%.3f, crit@0.001=%.3f]" % (
        uni["chi2_stat_10bins_df9"], CHI2_DF9_CRIT_05, CHI2_DF9_CRIT_001))

    print("\nBox-Muller normal diagnostics...")
    nrm, z = normal_diagnostics()
    print("  mean=%.6f, var=%.6f, skew=%.5f, excess_kurtosis=%.5f" % (
        nrm["mean"], nrm["variance"], nrm["skewness"], nrm["excess_kurtosis"]))
    print("  max |empirical CDF - Phi| over grid in [-3,3]: %.5f" % nrm["max_abs_empirical_cdf_diff_over_grid"])

    print("\nPoisson diagnostics...")
    pois = poisson_diagnostics()
    for lam in (4.0, 10.0, 25.0, 100.0, 1000.0):
        r = pois[str(lam)]
        print("  lambda=%6.1f [%14s]: mean=%.4f var=%.4f (se_mean=%.4f, se_var=%.4f)" % (
            lam, r["branch"], r["mean"], r["variance"], r["mc_standard_error_mean"], r["mc_standard_error_variance"]))
    print("  PMF check at lambda=1:", pois["pmf_check_lambda_1"]["values"])

    print("\nNeave-effect diagnostics (lag-1..5 autocorrelation of uniform stream)...")
    neave = lcg_neave_effect_diagnostics(u, z)
    for lag, val in neave["autocorrelation_by_lag"].items():
        n_se = val / neave["se_under_iid_null"]
        print("  lag=%s: autocorr=%+.5f  (%.2f standard errors from 0 under iid null)" % (lag, val, n_se))

    fig = make_figure(u, z)
    fig_path = save_fig(fig, "m1", "rng_histograms")
    print("\nfigure saved:", fig_path)

    out = save_json(
        "m1",
        "rng_benchmark",
        {"uniform": uni, "normal": nrm, "poisson": pois, "neave_effect": neave},
        seed=SEED_UNIFORM,
    )
    print("summary saved:", out)
