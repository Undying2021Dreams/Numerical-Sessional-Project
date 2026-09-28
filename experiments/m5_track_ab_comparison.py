from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments._common import RESULTS_DIR, save_json
from experiments.m3_irgnm_recovery import S_BLOOD, T_FRAMES, X_TRUE, _cwb_data, perturb_initial_guess
from src.config import METABOLIC_START, N_PARAMS, PROJECTION_EPS
from src.irgnm import DEFAULT_SCHEDULE, DEFAULT_TAU, run_irgnm
from src.jacobian import analytic_jacobian, forward_operator
from src.rng import derive_seed

DELTA_X = 0.1
ROOT_SEED = 20240301
SEED_IDX = 0


def _bounds_D_F():
    lo = np.full(N_PARAMS, -np.inf)
    hi = np.full(N_PARAMS, np.inf)
    lo[8] = 0.0
    hi[9] = 0.0
    hi[10] = 0.0
    lo[METABOLIC_START:] = PROJECTION_EPS
    return lo, hi


def irgnm_vs_scipy_least_squares():
    C_WB_data = _cwb_data(X_TRUE)
    y_true = forward_operator(X_TRUE, T_FRAMES, S_BLOOD, C_WB_data)
    seed = derive_seed(ROOT_SEED, f"delta_x={DELTA_X}", f"seed_idx={SEED_IDX}")
    x0, _x0_raw = perturb_initial_guess(DELTA_X, seed)

    t0 = time.perf_counter()
    ours = run_irgnm(
        x0, y_true, T_FRAMES, S_BLOOD, C_WB_data,
        schedule=DEFAULT_SCHEDULE, tau=DEFAULT_TAU, delta_y=0.0,
        max_iter=300, solver="qr", x_true=X_TRUE,
    )
    t_ours = time.perf_counter() - t0
    ours_final_err = ours["rel_error_total"][-1]

    def fun(x):
        return forward_operator(x, T_FRAMES, S_BLOOD, C_WB_data) - y_true

    def jac(x):
        return analytic_jacobian(x, T_FRAMES, S_BLOOD, C_WB_data)

    lo, hi = _bounds_D_F()
    t0 = time.perf_counter()
    scipy_res = least_squares(fun, x0, jac=jac, bounds=(lo, hi), method="trf",
                               ftol=1e-14, xtol=1e-14, gtol=1e-14, max_nfev=300)
    t_scipy = time.perf_counter() - t0
    scipy_final_err = float(np.linalg.norm(scipy_res.x - X_TRUE) / np.linalg.norm(X_TRUE))

    return {
        "setup": "noiseless, tissue+blood (Setup B-like), delta_x=0.1, seed_idx=0, "
                 "same analytic Jacobian and D(F) box bounds passed to both solvers",
        "ours_irgnm": {
            "final_rel_error": ours_final_err,
            "n_iterations": ours["n_iterations"],
            "diverged": ours["diverged"],
            "time_s": t_ours,
        },
        "scipy_least_squares": {
            "final_rel_error": scipy_final_err,
            "n_iterations": int(scipy_res.nfev),
            "converged": bool(scipy_res.success),
            "status": int(scipy_res.status),
            "time_s": t_scipy,
        },
    }


def load(path):
    return json.loads((RESULTS_DIR / path).read_text())


def build_table():
    linalg = load("m1/linalg_benchmark.json")["sweep"]
    quad = load("m1/quadrature_benchmark.json")
    fwd = load("m2/forward_model_benchmark.json")
    jac = load("m3/jacobian_and_solver.json")

    table = {
        "linear_solve": {
            "routine": "src.linalg.solve (LU, partial pivoting)",
            "reference": "numpy.linalg.solve",
            "max_rel_error_vs_numpy": linalg["lu"]["max_rel_error_vs_numpy"],
            "our_total_time_s": linalg["lu"]["total_solve_time_s"],
            "numpy_total_time_s": linalg["numpy_reference_total_solve_time_s"],
            "n_systems": linalg["n_systems"],
        },
        "least_squares_linear": {
            "routine": "src.qr.lstsq (Householder QR)",
            "reference": "numpy.linalg.solve (square case, from the same 200-system sweep)",
            "max_rel_error_vs_numpy": linalg["qr"]["max_rel_error_vs_numpy"],
            "our_total_time_s": linalg["qr"]["total_solve_time_s"],
            "numpy_total_time_s": linalg["numpy_reference_total_solve_time_s"],
            "n_systems": linalg["n_systems"],
        },
        "quadrature": {
            "routine": "src.quadrature.simpson / trapezoid",
            "reference": "exact integral (scipy comparison in results/m5/timing_complexity.json)",
            "simpson_measured_order": {k: v["simp_order_measured"] for k, v in quad.items() if k != "provenance"},
            "trapezoid_measured_order": {k: v["trap_order_measured"] for k, v in quad.items() if k != "provenance"},
        },
        "ode_forward_model": {
            "routine": "src.forward_model.closed_form_C_T / quadrature_C_T",
            "reference": "scipy.integrate.solve_ivp (method=Radau)",
            "max_rel_diff_closed_vs_radau": fwd["three_way_agreement"]["overall_max"]["closed_vs_ivp_radau"],
            "max_rel_diff_quad_vs_radau": fwd["three_way_agreement"]["overall_max"]["quad_vs_ivp_radau"],
            "solve_ivp_radau_time_s_per_region": {
                r: v["solve_ivp_radau_time_s"] for r, v in fwd["three_way_agreement"]["per_region"].items()
            },
        },
        "eigenvalues": {
            "routine": "src.eigen.power_method / inverse_power_iteration / symmetric_eigendecomposition",
            "reference": "numpy.linalg.eigvalsh / svd",
            "largest_eigenvalue": jac["conditioning_analysis"]["eigenvalue_largest"],
            "smallest_eigenvalue": jac["conditioning_analysis"]["eigenvalue_smallest"],
        },
        "fit_optimise": irgnm_vs_scipy_least_squares(),
    }
    return table


if __name__ == "__main__":
    print("Building Track A vs Track B comparison table...")
    table = build_table()

    print("\nlinear_solve      : max rel err vs numpy = %.3e" % table["linear_solve"]["max_rel_error_vs_numpy"])
    print("least_squares     : max rel err vs numpy = %.3e" % table["least_squares_linear"]["max_rel_error_vs_numpy"])
    print("ode_forward_model : max rel diff vs Radau (closed) = %.3e" % table["ode_forward_model"]["max_rel_diff_closed_vs_radau"])
    fo = table["fit_optimise"]
    print("fit_optimise      : ours final_rel_error=%.3e (%d iters, %.3fs) | "
          "scipy final_rel_error=%.3e (%d nfev, %.3fs)" % (
              fo["ours_irgnm"]["final_rel_error"], fo["ours_irgnm"]["n_iterations"], fo["ours_irgnm"]["time_s"],
              fo["scipy_least_squares"]["final_rel_error"], fo["scipy_least_squares"]["n_iterations"], fo["scipy_least_squares"]["time_s"]))

    out = save_json("m5", "track_ab_comparison", table, seed=ROOT_SEED)
    print("\nsummary saved:", out)
