"""M3 experiment, items 6-8: IRGNM noiseless recovery, the milestone's
centrepiece acceptance test.

No scipy.optimize anywhere — the solver (src/irgnm.py) is entirely our own
LU/QR + analytic Jacobian. `numpy.random` is not used; all randomness is
`src.rng.LCG` (our own).

Run: `python3 experiments/m3_irgnm_recovery.py`
Writes: results/m3/irgnm_recovery.json, results/m3/noiseless_recovery_*.png
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments._common import save_json
from src.config import N_PARAMS, ground_truth_vector, blood_sample_times_minutes, frame_midtimes_minutes
from src.forward_model import C_WB_from_C_P, arterial_input, parent_plasma_fraction
from src.irgnm import DEFAULT_SCHEDULE, DEFAULT_TAU, project, run_irgnm
from src.jacobian import forward_operator, unpack
from src.plotting import save_fig, set_style
from src.rng import LCG, derive_seed, standard_normal

X_TRUE = ground_truth_vector()
T_FRAMES = frame_midtimes_minutes()
S_BLOOD = blood_sample_times_minutes()
DELTA_X_VALUES = (0.1, 0.2, 0.3, 0.4)
N_SEEDS = 20
MAX_ITER = 300
ROOT_SEED = 20240301


def _cwb_data(x):
    lam, mu, m, K1, k2, k3 = unpack(x)
    A, xi1, xi2 = m
    return C_WB_from_C_P(arterial_input(S_BLOOD, lam, mu), parent_plasma_fraction(S_BLOOD, A, xi1, xi2))


def perturb_initial_guess(delta_x: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Paper eq. (23): x0 = x_dagger * (1 + sigma*gamma), sigma ~ Unif{-1,1},
    gamma ~ N(delta_x, delta_x/4) — the SECOND argument of N(.,.) here is the
    VARIANCE (paper eq. 24), not the std, so gamma's std is sqrt(delta_x/4).
    Returns (x0 clipped into D(F), raw unclipped x0) so callers can detect
    whether clipping was needed."""
    rng = LCG(seed=seed)
    sigma = np.where(rng.uniform_array(N_PARAMS) < 0.5, -1.0, 1.0)
    gamma = delta_x + np.sqrt(delta_x / 4.0) * standard_normal(rng, N_PARAMS)
    x0_raw = X_TRUE * (1.0 + sigma * gamma)
    x0_clipped = project(x0_raw)
    return x0_clipped, x0_raw


def verify_perturbation_sampler_statistics():
    """Empirically confirms E[(sigma*gamma)^2] = delta_x^2 + delta_x/4 (eq. 24)
    before trusting the sampler for the recovery study."""
    rows = {}
    for delta_x in DELTA_X_VALUES:
        rng = LCG(seed=derive_seed(ROOT_SEED, "sampler_check", str(delta_x)))
        n = 200_000
        u = rng.uniform_array(n)
        sigma = np.where(u < 0.5, -1.0, 1.0)
        z = standard_normal(rng, n)
        gamma = delta_x + np.sqrt(delta_x / 4.0) * z
        measured = float(np.mean((sigma * gamma) ** 2))
        predicted = delta_x**2 + delta_x / 4.0
        rows[str(delta_x)] = {"measured": measured, "predicted": predicted, "rel_error": abs(measured - predicted) / predicted}
    return rows


def run_recovery_study(solver: str = "qr"):
    C_WB_data = _cwb_data(X_TRUE)
    y_true = forward_operator(X_TRUE, T_FRAMES, S_BLOOD, C_WB_data)

    results = {}
    failures = []
    for delta_x in DELTA_X_VALUES:
        runs = []
        for seed_idx in range(N_SEEDS):
            seed = derive_seed(ROOT_SEED, f"delta_x={delta_x}", f"seed_idx={seed_idx}")
            x0, x0_raw = perturb_initial_guess(delta_x, seed)
            clip_needed = not np.allclose(x0, x0_raw)
            init_err = float(np.linalg.norm(x0 - X_TRUE) / np.linalg.norm(X_TRUE))

            t0 = time.perf_counter()
            result = run_irgnm(
                x0, y_true, T_FRAMES, S_BLOOD, C_WB_data,
                schedule=DEFAULT_SCHEDULE, tau=DEFAULT_TAU, delta_y=0.0,
                max_iter=MAX_ITER, solver=solver, x_true=X_TRUE,
            )
            dt = time.perf_counter() - t0

            final_err = result["rel_error_total"][-1]
            run_record = {
                "seed": seed,
                "seed_idx": seed_idx,
                "clip_needed": bool(clip_needed),
                "init_rel_error": init_err,
                "final_rel_error": final_err,
                "n_iterations": result["n_iterations"],
                "diverged": result["diverged"],
                "time_s": dt,
                "rel_error_total_trajectory": result["rel_error_total"],
                "rel_error_lambda_trajectory": result["rel_error_lambda"],
                "rel_error_mu_trajectory": result["rel_error_mu"],
                "rel_error_m_trajectory": result["rel_error_m"],
                "rel_error_K_trajectory": result["rel_error_K"],
            }
            runs.append(run_record)
            if result["diverged"] or not np.isfinite(final_err) or final_err > 1e-2:
                failures.append({"delta_x": delta_x, "seed": seed, "seed_idx": seed_idx, "final_rel_error": final_err,
                                  "diverged": result["diverged"], "init_rel_error": init_err})

        n_clipped = sum(r["clip_needed"] for r in runs)
        n_diverged = sum(r["diverged"] for r in runs)
        converged_runs = [r for r in runs if not r["diverged"] and np.isfinite(r["final_rel_error"])]
        final_errs = [r["final_rel_error"] for r in converged_runs]
        results[str(delta_x)] = {
            "n_seeds": N_SEEDS,
            "n_clipped": n_clipped,
            "n_diverged": n_diverged,
            "median_final_rel_error": float(np.median(final_errs)) if final_errs else None,
            "mean_final_rel_error_converged": float(np.mean(final_errs)) if final_errs else None,
            "max_final_rel_error_converged": float(np.max(final_errs)) if final_errs else None,
            "mean_iterations": float(np.mean([r["n_iterations"] for r in runs])),
            "runs": runs,
        }

    return results, failures


def make_convergence_figure(results: dict, solver_label: str):
    import matplotlib.pyplot as plt

    set_style()
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.2), sharey=True)
    for ax, delta_x in zip(axes, DELTA_X_VALUES):
        r = results[str(delta_x)]
        # Plot the run closest to the median final error (representative, not cherry-picked-best)
        finite_runs = [run for run in r["runs"] if not run["diverged"] and np.isfinite(run["final_rel_error"])]
        if finite_runs:
            median_err = np.median([run["final_rel_error"] for run in finite_runs])
            rep_run = min(finite_runs, key=lambda run: abs(run["final_rel_error"] - median_err))
            traj_total = rep_run["rel_error_total_trajectory"]
            iters = np.arange(len(traj_total))
            ax.semilogy(iters, traj_total, "k-", linewidth=2, label="total")
            ax.semilogy(iters, rep_run["rel_error_lambda_trajectory"], label="C_P block ($\\lambda$)")
            ax.semilogy(iters, rep_run["rel_error_mu_trajectory"], label="C_P block ($\\mu$)")
            ax.semilogy(iters, rep_run["rel_error_m_trajectory"], label="f block (m)")
            ax.semilogy(iters, rep_run["rel_error_K_trajectory"], label="metabolic block (K)")
        # also plot all OTHER non-diverged runs faintly, for context
        for run in finite_runs:
            ax.semilogy(np.arange(len(run["rel_error_total_trajectory"])), run["rel_error_total_trajectory"],
                        color="gray", alpha=0.15, linewidth=0.6, zorder=0)
        ax.set_title(f"delta_x = {delta_x}  ({r['n_diverged']}/{r['n_seeds']} diverged)")
        ax.set_xlabel("IRGNM iteration")
        if delta_x == DELTA_X_VALUES[0]:
            ax.set_ylabel(r"$\|x_k - x^\dagger\| / \|x^\dagger\|$")
            ax.legend(fontsize=7)
    fig.suptitle(f"Noiseless recovery convergence, solver={solver_label} (median-representative run per panel; gray = other seeds)")
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    print("Verifying perturbation sampler statistics (eq. 23/24)...")
    sampler_check = verify_perturbation_sampler_statistics()
    for dx, r in sampler_check.items():
        print(f"  delta_x={dx}: measured E[(sigma*gamma)^2]={r['measured']:.5f}  predicted={r['predicted']:.5f}  rel_err={r['rel_error']:.3e}")

    print("\nRunning noiseless recovery study (solver=qr, default)...")
    results_qr, failures_qr = run_recovery_study(solver="qr")
    for dx in DELTA_X_VALUES:
        r = results_qr[str(dx)]
        print(f"  delta_x={dx}: median_final_err={r['median_final_rel_error']:.3e}  "
              f"n_diverged={r['n_diverged']}/{r['n_seeds']}  n_clipped={r['n_clipped']}  "
              f"mean_iters={r['mean_iterations']:.1f}")

    print(f"\n{len(failures_qr)} divergent/failed runs (qr solver):")
    for f in failures_qr:
        print(f"  delta_x={f['delta_x']} seed={f['seed']}: init_err={f['init_rel_error']:.4f} "
              f"final_err={f['final_rel_error']} diverged={f['diverged']}")

    fig = make_convergence_figure(results_qr, "qr")
    p = save_fig(fig, "m3", "noiseless_recovery_qr")
    print("\nfigure saved:", p)

    out = save_json(
        "m3",
        "irgnm_recovery",
        {
            "sampler_check": sampler_check,
            "solver": "qr",
            "n_seeds_per_delta_x": N_SEEDS,
            "max_iter": MAX_ITER,
            "results": results_qr,
            "failures": failures_qr,
        },
        seed=ROOT_SEED,
    )
    print("summary saved:", out)
