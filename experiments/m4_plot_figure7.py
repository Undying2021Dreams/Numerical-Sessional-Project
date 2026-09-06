"""Figure 7 analogue — 4 panels, one per delta_x.

Matches the paper's Figure 7 layout:
  - Setup C (full, noisy C_WB), normal count
  - 4 panels: delta_x in {0.4, 0.3, 0.2, 0.1}  (right to left in paper; we go left to right)
  - Each panel: 4 curves — K1 error, k2 error, k3 error, K (net influx) error
  - Vertical orange line at the discrepancy-principle stopping iteration
  - Annotations: rho_d (error at stopping), rho_opt (minimum error over the run)

Because our grid run only records block-level errors (not per-parameter), this script
re-runs 4 representative IRGNM solves from scratch (one per delta_x), tracking
individual K1, k2, k3 errors at every iteration.  Each solve takes ~0.2 s.

Usage:
    python experiments/m4_plot_figure7.py
    -> saves results/m4/figure7_analogue.png
"""
from __future__ import annotations

import sys
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    METABOLIC_START,
    N_PARAMS,
    N_REGIONS,
    REGION_NAMES,
    blood_sample_times_minutes,
    frame_midtimes_minutes,
    ground_truth_vector,
)
from src.forward_model import (
    C_WB_from_C_P,
    arterial_input,
    closed_form_C_T,
    parent_plasma_fraction,
)
from src.irgnm import DEFAULT_SCHEDULE, DEFAULT_TAU, project, run_irgnm
from src.jacobian import forward_operator, unpack
from src.montecarlo import (
    GAUSSIAN_SIGMA_REL,
    POISSON_ALPHA,
    ROOT_SEED,
    SETUP_A_MASK,
    X_TRUE,
    T_FRAMES,
    S_BLOOD,
    C_T_CLEAN,
    C_WB_CLEAN,
    _build_noisy_observations,
)
from src.noise import add_poisson_noise, add_gaussian_noise
from src.plotting import save_fig, set_style
from src.rng import LCG, derive_seed, standard_normal

DELTA_X_VALUES = (0.4, 0.3, 0.2, 0.1)   # paper goes right-to-left; we go left-to-right
SETUP = "C"
NOISE_LEVEL = "normal_count"
MAX_ITER = 200  # paper cap is 200 for Figure 7
SEED_IDX = 0    # first converged seed; re-run attempts next seeds if this one diverges


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_x0(delta_x: float, seed_idx: int) -> np.ndarray:
    seed = derive_seed(ROOT_SEED, f"setup={SETUP}", f"dx={delta_x}",
                       f"noise={NOISE_LEVEL}", f"x0", f"k={seed_idx}")
    rng = LCG(seed)
    sigma = np.where(rng.uniform_array(N_PARAMS) < 0.5, -1.0, 1.0)
    gamma = delta_x + np.sqrt(delta_x / 4.0) * standard_normal(rng, N_PARAMS)
    return project(X_TRUE * (1.0 + sigma * gamma))


def run_and_track(delta_x: float) -> dict | None:
    """Run IRGNM for Setup C, normal_count, given delta_x.
    Track K1, k2, k3, and K (net influx) errors at each step.
    Tries up to 5 seed indices; returns None if all diverge."""

    # Build noisy observation for seed_idx=0 (same as montecarlo.py)
    for seed_idx in range(20):
        y_delta, C_WB_data, delta_y_tac = _build_noisy_observations(
            NOISE_LEVEL, seed_idx, SETUP, delta_x
        )
        x0 = _make_x0(delta_x, seed_idx)
        delta_y_stop = delta_y_tac

        # Run with x_true so we get rel_error_K, but we need per-param errors,
        # so we do a custom loop here.
        x_i = project(x0.copy())
        from src.irgnm import irgnm_step

        K1_true = np.array([X_TRUE[METABOLIC_START + 3*i + 0] for i in range(N_REGIONS)])
        k2_true = np.array([X_TRUE[METABOLIC_START + 3*i + 1] for i in range(N_REGIONS)])
        k3_true = np.array([X_TRUE[METABOLIC_START + 3*i + 2] for i in range(N_REGIONS)])
        # Net influx Ki = K1*k3/(k2+k3) per region
        Ki_true = K1_true * k3_true / (k2_true + k3_true)

        err_K1 = []
        err_k2 = []
        err_k3 = []
        err_Ki = []
        residuals = []
        stopping_iter = None
        diverged = False

        # Initial residual
        from src.jacobian import forward_operator as fwd
        Fx0 = fwd(x_i, T_FRAMES, S_BLOOD, C_WB_data, include_blood=True)
        r_norm = float(np.sqrt((y_delta - Fx0) @ (y_delta - Fx0)))

        def _record(x_cur, rn):
            K1_cur = np.array([x_cur[METABOLIC_START + 3*j + 0] for j in range(N_REGIONS)])
            k2_cur = np.array([x_cur[METABOLIC_START + 3*j + 1] for j in range(N_REGIONS)])
            k3_cur = np.array([x_cur[METABOLIC_START + 3*j + 2] for j in range(N_REGIONS)])
            Ki_cur = K1_cur * k3_cur / (k2_cur + k3_cur + 1e-30)
            err_K1.append(float(np.sqrt(np.sum((K1_cur - K1_true)**2)) / np.sqrt(np.sum(K1_true**2))))
            err_k2.append(float(np.sqrt(np.sum((k2_cur - k2_true)**2)) / np.sqrt(np.sum(k2_true**2))))
            err_k3.append(float(np.sqrt(np.sum((k3_cur - k3_true)**2)) / np.sqrt(np.sum(k3_true**2))))
            err_Ki.append(float(np.sqrt(np.sum((Ki_cur - Ki_true)**2)) / np.sqrt(np.sum(Ki_true**2))))
            residuals.append(rn)

        _record(x_i, r_norm)

        with np.errstate(over="ignore", invalid="ignore"):
            for i in range(MAX_ITER):
                if r_norm <= DEFAULT_TAU * delta_y_stop and delta_y_stop > 0.0:
                    stopping_iter = i
                    break
                reg_diag = DEFAULT_SCHEDULE.diag(i)
                try:
                    x_next, _Fx, _Fp = irgnm_step(
                        x_i, x0, y_delta, T_FRAMES, S_BLOOD, C_WB_data,
                        reg_diag, "qr", active_mask=None, include_blood=True,
                    )
                except Exception:
                    diverged = True
                    break
                if not np.all(np.isfinite(x_next)):
                    diverged = True
                    break
                x_i = x_next
                Fx_i = fwd(x_i, T_FRAMES, S_BLOOD, C_WB_data, include_blood=True)
                r_norm = float(np.sqrt((y_delta - Fx_i) @ (y_delta - Fx_i)))
                if not np.isfinite(r_norm):
                    diverged = True
                    break
                _record(x_i, r_norm)

        if diverged:
            continue  # try next seed

        # rho_d: error at stopping; rho_opt: minimum error
        stop_idx = stopping_iter if stopping_iter is not None else len(err_K1) - 1
        # aggregate K error (all three blocks)
        err_K_total = [np.sqrt(a**2 + b**2 + c**2) / np.sqrt(3) for a, b, c in zip(err_K1, err_k2, err_k3)]
        rho_d = err_K_total[stop_idx] if stop_idx < len(err_K_total) else err_K_total[-1]
        rho_opt = min(err_K_total)

        return {
            "seed_idx": seed_idx,
            "delta_x": delta_x,
            "err_K1": err_K1,
            "err_k2": err_k2,
            "err_k3": err_k3,
            "err_Ki": err_Ki,
            "err_K_total": err_K_total,
            "stopping_iter": stopping_iter,
            "n_iters": len(err_K1),
            "rho_d": rho_d,
            "rho_opt": rho_opt,
        }
    return None  # all seeds diverged


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    set_style()

    fig, axes = plt.subplots(1, 4, figsize=(18, 5), sharey=True)
    fig.subplots_adjust(wspace=0.06, left=0.07, right=0.99, top=0.82, bottom=0.13)

    for ax, dx in zip(axes, DELTA_X_VALUES):
        print(f"Running Setup C / {NOISE_LEVEL} / delta_x={dx} ...")
        res = run_and_track(dx)
        if res is None:
            ax.text(0.5, 0.5, "All seeds diverged", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title(f"$\\delta_x = {dx}$\n(no converged run)")
            ax.set_xlabel("Iteration")
            continue

        iters = np.arange(res["n_iters"])
        stop = res["stopping_iter"]

        # Shade the post-stopping region (if discrepancy rule fired)
        if stop is not None:
            ax.axvspan(stop, MAX_ITER, alpha=0.10, color="red", zorder=0)
            ax.axvline(x=stop, color="darkorange", linewidth=1.5, zorder=5)

        # Plot the 4 curves matching paper's colors
        ax.semilogy(iters, res["err_K1"], color="#E6B800", linewidth=1.8, label="$K_1$")
        ax.semilogy(iters, res["err_k2"], color="#4CAF50", linewidth=1.8, label="$k_2$")
        ax.semilogy(iters, res["err_k3"], color="#2196F3", linewidth=1.8, label="$k_3$")
        ax.semilogy(iters, res["err_Ki"], color="#9C27B0", linewidth=1.8, label="$K$")

        # Annotate rho_d and rho_opt (as percentages, matching paper)
        stop_k = stop if stop is not None else res["n_iters"] - 1
        opt_k = int(np.argmin(res["err_K_total"]))
        rho_d_pct = res["rho_d"] * 100
        rho_opt_pct = res["rho_opt"] * 100

        title = (f"$\\delta_x = {dx}$\n"
                 f"$\\rho_d = {rho_d_pct:.1f}\\%$ : $k = {stop_k}$\n"
                 f"$\\rho_{{opt}} = {rho_opt_pct:.1f}\\%$ : $k = {opt_k}$")
        ax.set_title(title, fontsize=10, pad=4)
        ax.set_xlabel("Iteration", fontsize=10)
        ax.set_xlim(0, MAX_ITER)
        ax.set_ylim(5e-3, 3.0)
        ax.grid(True, which="both", linestyle="--", alpha=0.4)

        if dx == DELTA_X_VALUES[-1]:  # rightmost (leftmost in original) gets legend
            ax.legend(fontsize=9, loc="lower right")

    axes[0].set_ylabel(
        r"Relative Deviation  $\|x^i\|^{-1}\|x^i_k - x^i\|_X$",
        fontsize=10
    )

    fig.suptitle(
        "Figure 7 Analogue — Relative error by parameter type\n"
        f"Setup C (noisy $C_{{WB}}$), normal count, varying $\\delta_x$",
        fontsize=11,
    )

    out = save_fig(fig, "m4", "figure7_analogue")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
