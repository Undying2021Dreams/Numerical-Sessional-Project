"""M3 experiments, items 1-5: Jacobian verification, conditioning analysis,
the null-space experiment, parameter scaling, and the LU-vs-QR linear
solver comparison on the real IRGNM problem.

Track B usage (clearly marked): `numpy.linalg.eigvalsh`/`svd`/`cond` are
used here as independent references for the conditioning analysis, per
CLAUDE.md section 1 ("Track B ... clearly-marked benchmark scripts").
`src/eigen.py` (our own power method / inverse power iteration) is the
primary Track A computation.

Run: `python3 experiments/m3_jacobian_and_solver.py`
Writes: results/m3/*.json, results/m3/*.png
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments._common import save_json
from src.config import (
    LAMBDA_SLICE,
    METABOLIC_START,
    MU_SLICE,
    M_SLICE,
    N_PARAMS,
    REGION_NAMES,
    UNKNOWN_LAYOUT,
    blood_sample_times_minutes,
    frame_midtimes_minutes,
    ground_truth_vector,
)
from src.eigen import inverse_power_iteration, power_method, symmetric_eigendecomposition
from src.forward_model import C_WB_from_C_P, arterial_input, parent_plasma_fraction
from src.irgnm import DEFAULT_SCHEDULE, DEFAULT_TAU, project, run_irgnm
from src.jacobian import analytic_jacobian, forward_operator, unpack
from src.linalg import solve as lu_solve
from src.plotting import save_fig, set_style
from src.qr import lstsq as qr_lstsq

X_TRUE = ground_truth_vector()
T_FRAMES = frame_midtimes_minutes()
S_BLOOD = blood_sample_times_minutes()
M_COLS = (8, 9, 10)


def _cwb_data(x):
    lam, mu, m, K1, k2, k3 = unpack(x)
    A, xi1, xi2 = m
    return C_WB_from_C_P(arterial_input(S_BLOOD, lam, mu), parent_plasma_fraction(S_BLOOD, A, xi1, xi2))


# ---------------------------------------------------------------------------
# Item 1: analytic Jacobian vs finite differences
# ---------------------------------------------------------------------------
FD_H_REL = 1.5e-5
BLOCKS = {"lambda": range(0, 4), "mu": range(4, 8), "m": range(8, 11), "K": range(11, 23)}


def _fd_column(x, j, C_WB_data, h_rel=FD_H_REL):
    hh = h_rel * max(abs(x[j]), 1.0)
    xp, xm = x.copy(), x.copy()
    xp[j] += hh
    xm[j] -= hh
    return (forward_operator(xp, T_FRAMES, S_BLOOD, C_WB_data) - forward_operator(xm, T_FRAMES, S_BLOOD, C_WB_data)) / (2 * hh)


def jacobian_verification():
    C_WB_data = _cwb_data(X_TRUE)
    J = analytic_jacobian(X_TRUE, T_FRAMES, S_BLOOD, C_WB_data)

    block_errors = {}
    for name, idxs in BLOCKS.items():
        worst = 0.0
        for j in idxs:
            fd_col = _fd_column(X_TRUE, j, C_WB_data)
            floor = max(1e-10, 1e-6 * np.max(np.abs(J[:, j])))
            relerr = float(np.max(np.abs(fd_col - J[:, j]) / np.maximum(np.abs(J[:, j]), floor)))
            worst = max(worst, relerr)
        block_errors[name] = worst

    return {"fd_h_rel": FD_H_REL, "max_rel_error_per_block": block_errors, "jacobian_shape": list(J.shape)}


# ---------------------------------------------------------------------------
# Item 2: conditioning analysis of F' at ground truth
# ---------------------------------------------------------------------------
def conditioning_analysis():
    C_WB_data = _cwb_data(X_TRUE)
    J = analytic_jacobian(X_TRUE, T_FRAMES, S_BLOOD, C_WB_data)
    M = J.T @ J

    t0 = time.perf_counter()
    lam_max, _v, it_max = power_method(M)
    t_power = time.perf_counter() - t0

    t0 = time.perf_counter()
    lam_min, _v, it_min = inverse_power_iteration(M, shift=0.0)
    t_inv = time.perf_counter() - t0

    t0 = time.perf_counter()
    eigvals_ours, _ = symmetric_eigendecomposition(M)
    t_full = time.perf_counter() - t0

    eigvals_np = np.linalg.eigvalsh(M)[::-1]  # Track B reference, descending
    svd_np = np.linalg.svd(J, compute_uv=False)  # Track B reference

    cond_ours = lam_max / lam_min
    cond_np = float(eigvals_np[0] / eigvals_np[-1])

    return {
        "eigenvalue_largest": {"ours_power_method": lam_max, "iterations": it_max, "time_s": t_power, "numpy_ref": float(eigvals_np[0])},
        "eigenvalue_smallest": {"ours_inverse_power": lam_min, "iterations": it_min, "time_s": t_inv, "numpy_ref": float(eigvals_np[-1])},
        "condition_number_of_M": {"ours": cond_ours, "numpy_ref": cond_np},
        "condition_number_of_Fprime": float(np.sqrt(cond_ours)),
        "full_spectrum_ours_deflation": eigvals_ours.tolist(),
        "full_spectrum_numpy": eigvals_np.tolist(),
        "full_spectrum_max_rel_error_top_half": float(
            np.max(np.abs(eigvals_ours[:11] - eigvals_np[:11]) / np.abs(eigvals_np[:11]))
        ),
        "full_spectrum_deflation_time_s": t_full,
        "singular_values_numpy": svd_np.tolist(),
        "numerical_rank_tol_1e-8_times_largest": int(np.sum(svd_np > 1e-8 * svd_np[0])),
    }


# ---------------------------------------------------------------------------
# Item 3: the null-space experiment
# ---------------------------------------------------------------------------
def null_space_experiment():
    C_WB_data = _cwb_data(X_TRUE)
    J = analytic_jacobian(X_TRUE, T_FRAMES, S_BLOOD, C_WB_data)
    F1 = J[: 4 * len(T_FRAMES), :]

    lam, mu, m, K1, k2, k3 = unpack(X_TRUE)
    v_pred = np.zeros(N_PARAMS)
    v_pred[0:4] = -lam
    for i in range(4):
        v_pred[11 + 3 * i] = K1[i]
    v_pred_unit = v_pred / np.sqrt(v_pred @ v_pred)

    direct_check = float(np.sqrt(np.sum((F1 @ v_pred_unit) ** 2)))

    keep = [i for i in range(N_PARAMS) if i not in M_COLS]
    F1r = F1[:, keep]
    M1r = F1r.T @ F1r
    v_pred_r = v_pred_unit[keep]
    v_pred_r = v_pred_r / np.sqrt(v_pred_r @ v_pred_r)

    lam_min_r, v_min_r, it_r = inverse_power_iteration(M1r, shift=0.0)
    cos_align = abs(float(v_pred_r @ v_min_r))

    eigvals_np_r = np.linalg.eigvalsh(M1r)
    ratio_to_next = float(eigvals_np_r[0] / eigvals_np_r[1])

    # deflation on the SAME reduced matrix, for comparison (documents the
    # accuracy gap between deflation-tail and direct inverse iteration —
    # DECISIONS.md D-M3-4)
    evals_defl, evecs_defl = symmetric_eigendecomposition(M1r)
    v_defl_smallest = evecs_defl[:, -1]
    cos_align_deflation = abs(float(v_pred_r @ v_defl_smallest))

    M_full = J.T @ J
    lam_min_full, _v, _it = inverse_power_iteration(M_full, shift=0.0)

    return {
        "direct_F1_v_pred_norm": direct_check,
        "F1_scale_norm": float(np.sqrt(np.sum(F1**2))),
        "reduced_smallest_eigenvalue_inverse_power": lam_min_r,
        "reduced_smallest_eigenvalue_numpy": float(eigvals_np_r[0]),
        "reduced_next_smallest_eigenvalue_numpy": float(eigvals_np_r[1]),
        "ratio_smallest_to_next_eigenvalue": ratio_to_next,
        "ratio_smallest_to_next_singular_value": float(np.sqrt(abs(ratio_to_next))),
        "alignment_cosine_inverse_power": cos_align,
        "alignment_cosine_deflation_tail": cos_align_deflation,
        "iterations_inverse_power": it_r,
        "full_jacobian_smallest_eigenvalue": lam_min_full,
        "increase_factor_F2_added": lam_min_full / max(abs(lam_min_r), 1e-300),
    }


# ---------------------------------------------------------------------------
# Item 3b: invariance at the function level
# ---------------------------------------------------------------------------
def invariance_check():
    from src.forward_model import closed_form_C_T

    lam, mu, m, K1, k2, k3 = unpack(X_TRUE)
    results = {}
    for c in (0.5, 2.0):
        lam_scaled = lam / c
        max_diff = 0.0
        for i, name in enumerate(REGION_NAMES):
            ct_true = closed_form_C_T(T_FRAMES, K1[i], k2[i], k3[i], lam, mu)
            ct_scaled = closed_form_C_T(T_FRAMES, K1[i] * c, k2[i], k3[i], lam_scaled, mu)
            max_diff = max(max_diff, float(np.max(np.abs(ct_true - ct_scaled))))
        results[str(c)] = max_diff
    return results


# ---------------------------------------------------------------------------
# Item 4: parameter scaling
# ---------------------------------------------------------------------------
def parameter_scaling_report():
    blocks = {"lambda": LAMBDA_SLICE, "mu": MU_SLICE, "m": M_SLICE, "K/k2/k3": slice(METABOLIC_START, N_PARAMS)}
    report = {}
    for name, sl in blocks.items():
        vals = np.abs(X_TRUE[sl])
        report[name] = {"min_abs": float(vals.min()), "max_abs": float(vals.max()), "ratio_max_over_min": float(vals.max() / vals.min())}
    all_vals = np.abs(X_TRUE)
    report["all_23_params"] = {"min_abs": float(all_vals.min()), "max_abs": float(all_vals.max()), "ratio_max_over_min": float(all_vals.max() / all_vals.min())}
    return report


# ---------------------------------------------------------------------------
# Item 5: LU (normal equations) vs QR (stacked) linear solver comparison
# ---------------------------------------------------------------------------
def linear_solver_comparison():
    """Uses a REAL, non-trivial point from an actual IRGNM trajectory (not
    x_true itself, where the residual is exactly zero) as the evaluation
    point, per the reviewer's 'on the real problem' instruction."""
    C_WB_data = _cwb_data(X_TRUE)
    y_true = forward_operator(X_TRUE, T_FRAMES, S_BLOOD, C_WB_data)

    from src.rng import LCG, standard_normal

    rng = LCG(seed=2024)
    delta_x = 0.2
    sigma = np.array([1.0 if rng.uniform() < 0.5 else -1.0 for _ in range(N_PARAMS)])
    gamma = delta_x + np.sqrt(delta_x / 4.0) * standard_normal(rng, N_PARAMS)
    x0 = project(X_TRUE * (1 + sigma * gamma))

    result = run_irgnm(x0, y_true, T_FRAMES, S_BLOOD, C_WB_data, schedule=DEFAULT_SCHEDULE, tau=DEFAULT_TAU, delta_y=0.0, max_iter=50, solver="qr", x_true=X_TRUE)
    x_eval = result["x_final"]  # a representative "mid-optimisation" point

    Fp = analytic_jacobian(x_eval, T_FRAMES, S_BLOOD, C_WB_data)
    r = y_true - forward_operator(x_eval, T_FRAMES, S_BLOOD, C_WB_data)

    rows = []
    for i in [0, 5, 10, 20, 30, 50, 75, 100, 150, 200, 250, 299]:
        reg_diag = DEFAULT_SCHEDULE.diag(i)
        M = Fp.T @ Fp + np.diag(reg_diag)
        cond_M = float(np.linalg.cond(M))  # Track B diagnostic

        sq = np.sqrt(reg_diag)
        A_stack = np.vstack([Fp, np.diag(sq)])
        cond_stack = float(np.linalg.cond(A_stack))  # Track B diagnostic

        rhs = Fp.T @ r + reg_diag * (x0 - x_eval)
        t0 = time.perf_counter()
        delta_lu = lu_solve(M, rhs)
        t_lu = time.perf_counter() - t0

        b_stack = np.concatenate([r, sq * (x0 - x_eval)])
        t0 = time.perf_counter()
        delta_qr, _resid = qr_lstsq(A_stack, b_stack)
        t_qr = time.perf_counter() - t0

        rel_diff = float(np.linalg.norm(delta_lu - delta_qr) / max(np.linalg.norm(delta_qr), 1e-300))

        rows.append({
            "iteration_index": i,
            "reg_alpha_metabolic": float(reg_diag[METABOLIC_START]),
            "reg_beta_arterial": float(reg_diag[0]),
            "reg_gamma_f": float(reg_diag[8]),
            "cond_normal_equations": cond_M,
            "cond_stacked_matrix": cond_stack,
            "cond_ratio_normal_over_stacked_sq": cond_M / cond_stack**2 if cond_stack > 0 else float("nan"),
            "rel_diff_lu_vs_qr_step": rel_diff,
            "time_lu_s": t_lu,
            "time_qr_s": t_qr,
        })

    return {"eval_point_source": "delta_x=0.2 seed=2024, 50 IRGNM(qr) iterations from x0", "sweep": rows}


def make_conditioning_figure(cond: dict):
    import matplotlib.pyplot as plt

    set_style()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    idx = np.arange(1, N_PARAMS + 1)
    ax.semilogy(idx, cond["full_spectrum_ours_deflation"], "o-", ms=4, label="ours (power method + deflation)")
    ax.semilogy(idx, cond["full_spectrum_numpy"], "x--", ms=5, label="numpy.linalg.eigvalsh (Track B)")
    ax.axhline(cond["eigenvalue_smallest"]["ours_inverse_power"], color="red", linestyle=":", linewidth=1,
               label=f"our inverse-power smallest = {cond['eigenvalue_smallest']['ours_inverse_power']:.3e}")
    ax.set_xlabel("eigenvalue index (descending)")
    ax.set_ylabel(r"eigenvalue of $F'^T F'$")
    ax.set_title(f"F'^T F' spectrum at ground truth (cond(F') = {cond['condition_number_of_Fprime']:.3e})")
    ax.legend(fontsize=8)
    return fig


def make_solver_comparison_figure(solver_cmp: dict):
    import matplotlib.pyplot as plt

    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    rows = solver_cmp["sweep"]
    it = [r["iteration_index"] for r in rows]
    axes[0].semilogy(it, [r["cond_normal_equations"] for r in rows], "o-", label="cond(F'^T F' + Lambda_i)  [LU path]")
    axes[0].semilogy(it, [r["cond_stacked_matrix"] for r in rows], "s-", label="cond([F'; sqrt(Lambda_i)])  [QR path]")
    axes[0].set_xlabel("IRGNM iteration index i (sets Lambda_i)")
    axes[0].set_ylabel("condition number")
    axes[0].set_title("Normal equations vs stacked system conditioning")
    axes[0].legend(fontsize=8)

    axes[1].semilogy(it, [max(r["rel_diff_lu_vs_qr_step"], 1e-18) for r in rows], "o-", color="tab:red")
    axes[1].set_xlabel("IRGNM iteration index i (sets Lambda_i)")
    axes[1].set_ylabel("relative difference, LU vs QR step")
    axes[1].set_title("Agreement between the two solved steps")
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    print("Item 1: Jacobian verification...")
    jac = jacobian_verification()
    for name, err in jac["max_rel_error_per_block"].items():
        print(f"  block {name:8s}: max rel error = {err:.3e}")

    print("\nItem 2: conditioning analysis...")
    cond = conditioning_analysis()
    print(f"  largest eigenvalue (power method): {cond['eigenvalue_largest']['ours_power_method']:.6e} "
          f"({cond['eigenvalue_largest']['iterations']} iters, numpy: {cond['eigenvalue_largest']['numpy_ref']:.6e})")
    print(f"  smallest eigenvalue (inverse power): {cond['eigenvalue_smallest']['ours_inverse_power']:.6e} "
          f"({cond['eigenvalue_smallest']['iterations']} iters, numpy: {cond['eigenvalue_smallest']['numpy_ref']:.6e})")
    print(f"  cond(F'^T F') = {cond['condition_number_of_M']['ours']:.6e}  cond(F') = {cond['condition_number_of_Fprime']:.6e}")
    print(f"  numerical rank (tol 1e-8*largest singular value): {cond['numerical_rank_tol_1e-8_times_largest']} / 23")
    print(f"  full-spectrum deflation vs numpy, top-half max rel err: {cond['full_spectrum_max_rel_error_top_half']:.3e}")

    print("\nItem 3: null-space experiment...")
    ns = null_space_experiment()
    print(f"  F1 @ v_pred norm: {ns['direct_F1_v_pred_norm']:.3e}  (F1 scale: {ns['F1_scale_norm']:.3e})")
    print(f"  reduced smallest eigenvalue: {ns['reduced_smallest_eigenvalue_inverse_power']:.3e}")
    print(f"  ratio smallest/next eigenvalue: {ns['ratio_smallest_to_next_eigenvalue']:.3e}  (singular value ratio: {ns['ratio_smallest_to_next_singular_value']:.3e})")
    print(f"  alignment cosine (inverse power): {ns['alignment_cosine_inverse_power']:.8f}")
    print(f"  alignment cosine (deflation tail, for comparison): {ns['alignment_cosine_deflation_tail']:.8f}")
    print(f"  full Jacobian (with F2) smallest eigenvalue: {ns['full_jacobian_smallest_eigenvalue']:.6e}  (increase factor: {ns['increase_factor_F2_added']:.3e})")

    print("\nItem 3b: invariance check...")
    inv = invariance_check()
    for c, diff in inv.items():
        print(f"  c={c}: max abs diff in C_T = {diff:.3e}")

    print("\nItem 4: parameter scaling...")
    scaling = parameter_scaling_report()
    for name, r in scaling.items():
        print(f"  {name:14s}: min={r['min_abs']:.4f} max={r['max_abs']:.4f} ratio={r['ratio_max_over_min']:.2f}")

    print("\nItem 5: linear solver comparison (this evaluates a real IRGNM trajectory point)...")
    solver_cmp = linear_solver_comparison()
    for r in solver_cmp["sweep"]:
        print(f"  i={r['iteration_index']:4d}  cond(normal_eq)={r['cond_normal_equations']:.3e}  "
              f"cond(stacked)={r['cond_stacked_matrix']:.3e}  rel_diff={r['rel_diff_lu_vs_qr_step']:.3e}")

    fig1 = make_conditioning_figure(cond)
    p1 = save_fig(fig1, "m3", "conditioning_spectrum")
    print("\nfigure saved:", p1)

    fig2 = make_solver_comparison_figure(solver_cmp)
    p2 = save_fig(fig2, "m3", "solver_comparison")
    print("figure saved:", p2)

    out = save_json(
        "m3",
        "jacobian_and_solver",
        {
            "jacobian_verification": jac,
            "conditioning_analysis": cond,
            "null_space_experiment": ns,
            "invariance_check": inv,
            "parameter_scaling": scaling,
            "linear_solver_comparison": solver_cmp,
        },
    )
    print("summary saved:", out)
