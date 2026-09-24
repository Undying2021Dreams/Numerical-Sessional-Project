"""M5: timing and complexity — LU, QR, Simpson, and the IRGNM inner solve,
reported as fitted operation-count scaling (time ~ C * n^p) rather than raw
wall-clock, per remaining_task.md Part 3's timing gotcha (Python/NumPy
dispatch overhead dominates in the tested size range, so a raw-seconds claim
would be misleading).

LU and QR scaling were already measured in `m1_linalg_benchmark.py`; this
script reads that JSON rather than re-measuring (same numbers, no
duplicated randomness/timing noise) and adds the two pieces M1 did not
cover:

- Simpson (`src.quadrature.simpson`) runtime vs number of grid points,
  fitted the same way, plus `scipy.integrate.simpson` (Track B) at the same
  sizes for comparison. Track B usage here is a runtime/accuracy reference
  only, per AGENTS.md section 1 — nothing under `src/` calls it.
- The IRGNM inner solve. The real PET problem has a *fixed* physical size
  (104-dim F, 23 parameters) — it cannot itself be "scaled up" meaningfully.
  What actually varies with problem size is the stacked least-squares solve
  `src.qr.lstsq` performs once per IRGNM iteration (see `irgnm_step`), so
  that call is benchmarked directly on synthetic stacked systems that keep
  the real problem's rows:cols aspect ratio (~4.5:1, from 104 residuals :
  23 parameters). See DECISIONS.md D-M5-2. `numpy.linalg.lstsq` is the
  Track B reference at the same sizes. A single real `run_irgnm` call at
  the actual problem size is also timed, to ground the synthetic curve
  with one concrete iterations/second figure.

Run: `python3 experiments/m5_timing_complexity.py`
Writes: results/m5/timing_complexity.json, results/m5/timing_complexity.png
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.integrate import simpson as scipy_simpson

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments._common import RESULTS_DIR, save_json
from src.config import blood_sample_times_minutes, frame_midtimes_minutes, ground_truth_vector
from src.forward_model import C_WB_from_C_P, arterial_input, parent_plasma_fraction
from src.irgnm import DEFAULT_SCHEDULE, DEFAULT_TAU, run_irgnm
from src.jacobian import forward_operator, unpack
from src.plotting import save_fig, set_style
from src.quadrature import simpson as our_simpson
from src.qr import lstsq as our_lstsq

SEED = 20240501
N_REPEATS = 5  # repeat each timing measurement and take the min (reduces OS jitter)

# --- Simpson runtime vs n --------------------------------------------------
SIMPSON_NS = np.array([101, 201, 401, 801, 1601, 3201, 6401, 12801, 25601])


def _time_call(fn, repeats=N_REPEATS):
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = fn()
        dt = time.perf_counter() - t0
        best = min(best, dt)
    return best, result


def _fit_power_law(ns, ts):
    """time ~ C * n^p, fit via our own QR least squares (Track A), consistent
    with m1_linalg_benchmark.py's convention."""
    log_n = np.log(ns.astype(np.float64))
    log_t = np.log(ts)
    design = np.column_stack([log_n, np.ones_like(log_n)])
    coeffs, _resid = our_lstsq(design, log_t)
    p, C = float(coeffs[0]), float(np.exp(coeffs[1]))
    return p, C


def simpson_scaling():
    f = np.sin
    a, b = 0.0, np.pi
    exact = 2.0

    our_times, scipy_times = [], []
    our_errs, scipy_errs = [], []
    for n in SIMPSON_NS:
        x = np.linspace(a, b, int(n))
        y = f(x)

        dt_ours, val_ours = _time_call(lambda: our_simpson(x, y))
        dt_scipy, val_scipy = _time_call(lambda: scipy_simpson(y, x=x))

        our_times.append(dt_ours)
        scipy_times.append(dt_scipy)
        our_errs.append(abs(val_ours - exact))
        scipy_errs.append(abs(val_scipy - exact))

    our_times = np.array(our_times)
    scipy_times = np.array(scipy_times)
    p_ours, C_ours = _fit_power_law(SIMPSON_NS, our_times)
    p_scipy, C_scipy = _fit_power_law(SIMPSON_NS, scipy_times)

    n_compare = 12801.0
    ratio_at_n = (C_ours * n_compare**p_ours) / (C_scipy * n_compare**p_scipy)

    return {
        "ns": SIMPSON_NS.tolist(),
        "our_time_s": our_times.tolist(),
        "scipy_time_s": scipy_times.tolist(),
        "our_abs_error": our_errs,
        "scipy_abs_error": scipy_errs,
        "our_fitted_exponent_p": p_ours,
        "our_fitted_constant_C": C_ours,
        "scipy_fitted_exponent_p": p_scipy,
        "scipy_fitted_constant_C": C_scipy,
        "ours_over_scipy_predicted_ratio_at_n12801": ratio_at_n,
        "note": "our simpson is a pure-Python loop over quadratic segments "
                "(non-uniform-grid support, DECISIONS.md D-M1-4); scipy's is "
                "a vectorised C implementation, so a slower constant here is "
                "expected and not a correctness concern.",
    }


# --- IRGNM inner solve (src.qr.lstsq) vs synthetic problem size -----------
# Real problem: 104 residuals (25 frames x 4 regions + 4 blood samples), 23
# parameters -> rows:cols ~ 4.52:1. Synthetic sizes keep that ratio while
# scaling cols up, so the *shape* of the linear system IRGNM solves every
# iteration is preserved (DECISIONS.md D-M5-2).
IRGNM_COLS = np.array([23, 46, 92, 184, 368, 736])
ASPECT_RATIO = 104.0 / 23.0


def irgnm_inner_solve_scaling():
    rng = np.random.default_rng(SEED)  # Track B: only used to build synthetic
                                        # benchmark matrices, not in src/.
    our_times, numpy_times = [], []
    for cols in IRGNM_COLS:
        cols = int(cols)
        rows = int(round(cols * ASPECT_RATIO))
        A = rng.normal(size=(rows, cols))
        A += np.eye(rows, cols) * cols  # keep well-conditioned as size grows
        b = rng.normal(size=rows)

        dt_ours, _ = _time_call(lambda: our_lstsq(A, b))
        dt_numpy, _ = _time_call(lambda: np.linalg.lstsq(A, b, rcond=None))

        our_times.append(dt_ours)
        numpy_times.append(dt_numpy)

    our_times = np.array(our_times)
    numpy_times = np.array(numpy_times)
    p_ours, C_ours = _fit_power_law(IRGNM_COLS, our_times)
    p_numpy, C_numpy = _fit_power_law(IRGNM_COLS, numpy_times)

    return {
        "cols": IRGNM_COLS.tolist(),
        "rows": [int(round(c * ASPECT_RATIO)) for c in IRGNM_COLS],
        "our_time_s": our_times.tolist(),
        "numpy_time_s": numpy_times.tolist(),
        "our_fitted_exponent_p": p_ours,
        "our_fitted_constant_C": C_ours,
        "numpy_fitted_exponent_p": p_numpy,
        "numpy_fitted_constant_C": C_numpy,
        "theoretical_exponent": 3.0,
        "note": "synthetic stacked systems at the real problem's rows:cols "
                "aspect ratio (~4.52:1), benchmarking the exact "
                "src.qr.lstsq call irgnm_step makes every iteration. The "
                "real PET problem's own size (104x23) is fixed and not "
                "itself varied.",
    }


def real_irgnm_per_iteration_timing():
    """One concrete measured data point: ms/iteration of run_irgnm on the
    actual fixed-size noiseless PET problem, grounding the synthetic curve
    above in the real pipeline."""
    x_true = ground_truth_vector()
    t_frames = frame_midtimes_minutes()
    s_blood = blood_sample_times_minutes()
    lam, mu, m, K1, k2, k3 = unpack(x_true)
    A, xi1, xi2 = m
    C_WB_data = C_WB_from_C_P(arterial_input(s_blood, lam, mu), parent_plasma_fraction(s_blood, A, xi1, xi2))
    y_true = forward_operator(x_true, t_frames, s_blood, C_WB_data)

    x0 = x_true * 1.1  # small, fixed perturbation; timing only, not a recovery claim
    max_iter = 100
    t0 = time.perf_counter()
    result = run_irgnm(
        x0, y_true, t_frames, s_blood, C_WB_data,
        schedule=DEFAULT_SCHEDULE, tau=DEFAULT_TAU, delta_y=0.0,
        max_iter=max_iter, solver="qr", x_true=x_true,
    )
    dt = time.perf_counter() - t0
    n_iter = result["n_iterations"]
    return {
        "n_iterations": n_iter,
        "total_time_s": dt,
        "ms_per_iteration": 1000.0 * dt / n_iter if n_iter else None,
        "problem_shape": "104 residuals x 23 parameters (fixed by the model)",
    }


def make_figure(simpson_res: dict, irgnm_res: dict):
    import matplotlib.pyplot as plt

    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    ax = axes[0]
    ax.loglog(simpson_res["ns"], simpson_res["our_time_s"], "o-", label=f"our simpson (p={simpson_res['our_fitted_exponent_p']:.2f})")
    ax.loglog(simpson_res["ns"], simpson_res["scipy_time_s"], "s-", label=f"scipy.integrate.simpson (p={simpson_res['scipy_fitted_exponent_p']:.2f})")
    ax.set_xlabel("n grid points")
    ax.set_ylabel("wall time (s)")
    ax.set_title("Simpson runtime vs n")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.loglog(irgnm_res["cols"], irgnm_res["our_time_s"], "o-", label=f"our qr.lstsq (p={irgnm_res['our_fitted_exponent_p']:.2f})")
    ax.loglog(irgnm_res["cols"], irgnm_res["numpy_time_s"], "s-", label=f"numpy.linalg.lstsq (p={irgnm_res['numpy_fitted_exponent_p']:.2f})")
    ax.set_xlabel("n parameters (cols), rows = 4.52x cols")
    ax.set_ylabel("wall time (s)")
    ax.set_title("IRGNM inner solve runtime vs problem size")
    ax.legend(fontsize=8)

    fig.tight_layout()
    return fig


if __name__ == "__main__":
    print("Loading LU/QR scaling from results/m1/linalg_benchmark.json...")
    linalg_path = RESULTS_DIR / "m1" / "linalg_benchmark.json"
    linalg_summary = json.loads(linalg_path.read_text())["sweep"]
    print(f"  LU: p={linalg_summary['lu']['fitted_time_exponent_p']:.3f}  "
          f"QR: p={linalg_summary['qr']['fitted_time_exponent_p']:.3f}  (theoretical: 3.0)")

    print("\nMeasuring Simpson runtime vs n...")
    simpson_res = simpson_scaling()
    print(f"  our simpson:   p={simpson_res['our_fitted_exponent_p']:.3f}")
    print(f"  scipy simpson: p={simpson_res['scipy_fitted_exponent_p']:.3f}")
    print(f"  predicted ratio (ours/scipy) at n=12801: {simpson_res['ours_over_scipy_predicted_ratio_at_n12801']:.2f}x")

    print("\nMeasuring IRGNM inner solve (qr.lstsq) vs synthetic problem size...")
    irgnm_res = irgnm_inner_solve_scaling()
    print(f"  our qr.lstsq:      p={irgnm_res['our_fitted_exponent_p']:.3f}  (theoretical ~3.0)")
    print(f"  numpy.linalg.lstsq: p={irgnm_res['numpy_fitted_exponent_p']:.3f}")

    print("\nTiming one real run_irgnm call at the actual fixed problem size...")
    real_timing = real_irgnm_per_iteration_timing()
    print(f"  {real_timing['n_iterations']} iterations in {real_timing['total_time_s']:.3f}s "
          f"= {real_timing['ms_per_iteration']:.3f} ms/iteration")

    fig = make_figure(simpson_res, irgnm_res)
    fig_path = save_fig(fig, "m5", "timing_complexity")
    print("\nfigure saved:", fig_path)

    out = save_json(
        "m5",
        "timing_complexity",
        {
            "lu_qr_from_m1": {
                "lu_fitted_exponent_p": linalg_summary["lu"]["fitted_time_exponent_p"],
                "qr_fitted_exponent_p": linalg_summary["qr"]["fitted_time_exponent_p"],
                "source": "results/m1/linalg_benchmark.json (not re-measured)",
            },
            "simpson": simpson_res,
            "irgnm_inner_solve": irgnm_res,
            "real_irgnm_per_iteration": real_timing,
        },
        seed=SEED,
    )
    print("summary saved:", out)
