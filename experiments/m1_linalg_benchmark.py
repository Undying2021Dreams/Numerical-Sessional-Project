"""M1 benchmark: LU and QR accuracy on 200 random systems + a Hilbert(8) stress test.

Track B usage (clearly marked, per the project brief): `numpy.linalg.solve`
and `numpy.linalg.cond` are reference/diagnostic only, used here to verify
Track A (`src.linalg`, `src.qr`) and to report the condition number of the
Hilbert matrix. Nothing in `src/` calls either.

Run: `python3 experiments/m1_linalg_benchmark.py`
Writes: results/m1/linalg_benchmark.json, results/m1/linalg_residuals.png
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments._common import save_json
from src import linalg, qr
from src.plotting import save_fig, set_style
from src.qr import lstsq as our_lstsq

SEED = 2024
N_SYSTEMS = 200
SIZES = np.linspace(5, 100, N_SYSTEMS).astype(int)
HILBERT_SIZES = (6, 8, 10, 12)


def _random_well_conditioned(n: int, rng: np.random.Generator) -> np.ndarray:
    A = rng.normal(size=(n, n))
    A += n * np.eye(n)
    return A


def run_200_system_sweep():
    rng = np.random.default_rng(SEED)

    sizes = []
    lu_rel_residual = []
    lu_rel_error_vs_numpy = []
    qr_rel_residual = []
    qr_rel_error_vs_numpy = []
    lu_time_per_n = []
    qr_time_per_n = []

    t_lu_total = 0.0
    t_qr_total = 0.0
    t_np_total = 0.0

    for n in SIZES:
        n = int(n)
        A = _random_well_conditioned(n, rng)
        x_true = rng.normal(size=n)
        b = A @ x_true

        t0 = time.perf_counter()
        x_lu = linalg.solve(A, b)
        dt_lu = time.perf_counter() - t0
        t_lu_total += dt_lu

        t0 = time.perf_counter()
        x_qr = qr.solve(A, b)
        dt_qr = time.perf_counter() - t0
        t_qr_total += dt_qr

        t0 = time.perf_counter()
        x_ref = np.linalg.solve(A, b)  # Track B reference
        t_np_total += time.perf_counter() - t0

        sizes.append(n)
        lu_time_per_n.append(dt_lu)
        qr_time_per_n.append(dt_qr)
        lu_rel_residual.append(float(np.linalg.norm(A @ x_lu - b) / np.linalg.norm(b)))
        lu_rel_error_vs_numpy.append(float(np.linalg.norm(x_lu - x_ref) / np.linalg.norm(x_ref)))
        qr_rel_residual.append(float(np.linalg.norm(A @ x_qr - b) / np.linalg.norm(b)))
        qr_rel_error_vs_numpy.append(float(np.linalg.norm(x_qr - x_ref) / np.linalg.norm(x_ref)))

    # DECISIONS.md D-M1-15: reframe timing as operation counts vs MEASURED
    # scaling, not a bare wall-clock claim against LAPACK. LU is ~2n^3/3
    # flops, Householder QR is ~4n^3/3 flops (~2x LU) by construction. Fit
    # log(time) = p*log(n) + log(C) via our own QR least squares (2
    # parameters: p and C, p left free rather than pinned to 3, so the
    # measurement can actually confirm or contradict the n^3 expectation
    # rather than assuming it).
    sizes_arr = np.array(sizes, dtype=np.float64)
    lu_t = np.array(lu_time_per_n)
    qr_t = np.array(qr_time_per_n)
    # Drop the smallest sizes: at n<20 or so, fixed Python call overhead
    # (function calls, array allocation) dominates over the n^3 term, which
    # would bias the fit; restricting to n>=20 isolates the asymptotic
    # regime the operation-count argument is actually about.
    mask = sizes_arr >= 20
    log_n = np.log(sizes_arr[mask])
    design = np.column_stack([log_n, np.ones_like(log_n)])

    coeffs_lu, _ = our_lstsq(design, np.log(lu_t[mask]))
    coeffs_qr, _ = our_lstsq(design, np.log(qr_t[mask]))
    p_lu, C_lu = float(coeffs_lu[0]), float(np.exp(coeffs_lu[1]))
    p_qr, C_qr = float(coeffs_qr[0]), float(np.exp(coeffs_qr[1]))
    # Ratio of *predicted* times at a fixed n=100 (rather than the ratio of
    # C's, which is only meaningful if p_lu == p_qr exactly) — this is the
    # exponent-independent way to state "QR takes ~Rx as long as LU here".
    n_compare = 100.0
    predicted_time_lu_at_100 = C_lu * n_compare**p_lu
    predicted_time_qr_at_100 = C_qr * n_compare**p_qr
    qr_over_lu_predicted_ratio_at_n100 = predicted_time_qr_at_100 / predicted_time_lu_at_100

    summary = {
        "n_systems": N_SYSTEMS,
        "sizes_range": [int(SIZES.min()), int(SIZES.max())],
        "lu": {
            "max_rel_residual": max(lu_rel_residual),
            "max_rel_error_vs_numpy": max(lu_rel_error_vs_numpy),
            "mean_rel_residual": float(np.mean(lu_rel_residual)),
            "total_solve_time_s": t_lu_total,
            "fitted_time_exponent_p": p_lu,
            "fitted_time_constant_C": C_lu,
        },
        "qr": {
            "max_rel_residual": max(qr_rel_residual),
            "max_rel_error_vs_numpy": max(qr_rel_error_vs_numpy),
            "mean_rel_residual": float(np.mean(qr_rel_residual)),
            "total_solve_time_s": t_qr_total,
            "fitted_time_exponent_p": p_qr,
            "fitted_time_constant_C": C_qr,
        },
        "qr_over_lu_predicted_ratio_at_n100": qr_over_lu_predicted_ratio_at_n100,
        "theoretical_flop_ratio_qr_over_lu": (4.0 / 3.0) / (2.0 / 3.0),
        "numpy_reference_total_solve_time_s": t_np_total,
        "sizes": sizes,
        "lu_rel_residual": lu_rel_residual,
        "qr_rel_residual": qr_rel_residual,
        "lu_time_per_n": lu_time_per_n,
        "qr_time_per_n": qr_time_per_n,
    }
    return summary


def run_hilbert_stress_test(n: int):
    A = np.array([[1.0 / (i + j + 1) for j in range(n)] for i in range(n)])
    x_true = np.ones(n)
    b = A @ x_true

    cond = float(np.linalg.cond(A))  # Track B diagnostic

    x_lu = linalg.solve(A, b)
    x_qr_ = qr.solve(A, b)
    x_ref = np.linalg.solve(A, b)

    rel_err_lu = float(np.linalg.norm(x_lu - x_true) / np.linalg.norm(x_true))
    rel_err_qr = float(np.linalg.norm(x_qr_ - x_true) / np.linalg.norm(x_true))
    rel_err_ref = float(np.linalg.norm(x_ref - x_true) / np.linalg.norm(x_true))
    rel_residual_lu = float(np.linalg.norm(A @ x_lu - b) / np.linalg.norm(b))
    rel_residual_qr = float(np.linalg.norm(A @ x_qr_ - b) / np.linalg.norm(b))
    lu_vs_qr = float(np.linalg.norm(x_lu - x_qr_) / np.linalg.norm(x_lu))

    return {
        "n": n,
        "condition_number_2norm": cond,
        "lu": {"rel_error_vs_true": rel_err_lu, "rel_residual": rel_residual_lu},
        "qr": {"rel_error_vs_true": rel_err_qr, "rel_residual": rel_residual_qr},
        "numpy_reference": {"rel_error_vs_true": rel_err_ref},
        "lu_vs_qr_rel_diff": lu_vs_qr,
        "lu_more_accurate_than_qr": rel_err_lu < rel_err_qr,
    }


def run_hilbert_multi_size_sweep():
    """DECISIONS.md D-M1-9: secondary evidence that the M1 Hilbert(8) finding
    (LU slightly more accurate than QR) is round-off noise, not a trend —
    swept over several sizes rather than asserted from one matrix."""
    return {str(n): run_hilbert_stress_test(n) for n in HILBERT_SIZES}


def make_figure(summary: dict):
    import matplotlib.pyplot as plt

    set_style()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.semilogy(summary["sizes"], summary["lu_rel_residual"], "o-", ms=3, label="LU")
    ax.semilogy(summary["sizes"], summary["qr_rel_residual"], "s-", ms=3, label="QR")
    ax.set_xlabel("matrix size n")
    ax.set_ylabel("relative residual $\\|Ax-b\\|/\\|b\\|$")
    ax.set_title(f"LU vs QR relative residual, {summary['n_systems']} random systems")
    ax.legend()
    return fig


if __name__ == "__main__":
    print("Running 200-system sweep (sizes 5..100)...")
    sweep = run_200_system_sweep()
    print("LU  : max rel residual = %.3e, max rel error vs numpy = %.3e" % (
        sweep["lu"]["max_rel_residual"], sweep["lu"]["max_rel_error_vs_numpy"]))
    print("QR  : max rel residual = %.3e, max rel error vs numpy = %.3e" % (
        sweep["qr"]["max_rel_residual"], sweep["qr"]["max_rel_error_vs_numpy"]))
    print("wall-clock totals (s): LU=%.4f QR=%.4f numpy=%.4f (numpy is compiled LAPACK; not a same-implementation comparison)" % (
        sweep["lu"]["total_solve_time_s"], sweep["qr"]["total_solve_time_s"],
        sweep["numpy_reference_total_solve_time_s"]))
    print("fitted time ~ C * n^p (n>=20): LU p=%.3f C=%.3e | QR p=%.3f C=%.3e" % (
        sweep["lu"]["fitted_time_exponent_p"], sweep["lu"]["fitted_time_constant_C"],
        sweep["qr"]["fitted_time_exponent_p"], sweep["qr"]["fitted_time_constant_C"]))
    print("predicted QR/LU time ratio at n=100: %.3f  (theoretical flop ratio 4n^3/3 / 2n^3/3 = %.3f)" % (
        sweep["qr_over_lu_predicted_ratio_at_n100"], sweep["theoretical_flop_ratio_qr_over_lu"]))

    print("\nRunning Hilbert ill-conditioned stress test, sizes", HILBERT_SIZES, "...")
    hilbert_sweep = run_hilbert_multi_size_sweep()
    for n_str, h in hilbert_sweep.items():
        print("n=%2s: cond=%.3e | LU rel_err=%.3e | QR rel_err=%.3e | LU<QR? %s | LU-vs-QR diff=%.3e" % (
            n_str, h["condition_number_2norm"], h["lu"]["rel_error_vs_true"], h["qr"]["rel_error_vs_true"],
            h["lu_more_accurate_than_qr"], h["lu_vs_qr_rel_diff"]))
    n_lu_wins = sum(1 for h in hilbert_sweep.values() if h["lu_more_accurate_than_qr"])
    print(f"LU more accurate than QR in {n_lu_wins}/{len(hilbert_sweep)} Hilbert sizes tested "
          f"(consistent with 'round-off noise, not a trend' if this is neither 0 nor {len(hilbert_sweep)})")

    fig = make_figure(sweep)
    fig_path = save_fig(fig, "m1", "linalg_residuals")
    print("\nfigure saved:", fig_path)

    out = save_json(
        "m1", "linalg_benchmark",
        {"sweep": sweep, "hilbert8": hilbert_sweep["8"], "hilbert_multi_size_sweep": hilbert_sweep},
        seed=SEED,
    )
    print("summary saved:", out)
