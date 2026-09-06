"""M4 noise calibration experiment.

Produces results/m4/noise_calibration.json.

For each of the three noise levels (high_count, normal_count, low_count):
  - Finds alpha (Poisson) and sigma_rel (Gaussian) such that the mean
    delta_y over 20 realisations matches the paper's target (0.003 / 0.011 / 0.07).
  - Reports the measured mean and std of delta_y over the 20 realisations
    used in the calibration sweep.
  - Compares Poisson vs Gaussian at the same nominal level.

Calibrated parameters are printed and saved to JSON for downstream use
by the M4 Monte Carlo harness.

Usage:
    python experiments/m4_noise_calibration.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

# Make sure repo root is on sys.path when run directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments._common import save_json
from src.config import (
    N_FRAMES,
    N_REGIONS,
    blood_sample_times_minutes,
    frame_midtimes_minutes,
    ground_truth_vector,
)
from src.forward_model import closed_form_C_T
from src.jacobian import unpack
from src.noise import (
    DELTA_Y_TARGETS,
    add_gaussian_noise,
    add_poisson_noise,
    calibrate_gaussian,
    calibrate_poisson,
    compute_delta_y,
)
from src.rng import LCG, derive_seed

ROOT_SEED = 20240401
N_CALIB = 20    # realisations used to estimate mean delta_y during calibration
N_VERIFY = 20   # realisations used to verify and report final mean/std

X_TRUE = ground_truth_vector()
T_FRAMES = frame_midtimes_minutes()


def build_clean_tacs() -> np.ndarray:
    """Clean TAC array, shape (N_REGIONS, N_FRAMES)."""
    lam, mu, m, K1, k2, k3 = unpack(X_TRUE)
    C_T = np.empty((N_REGIONS, N_FRAMES))
    for i in range(N_REGIONS):
        C_T[i] = closed_form_C_T(T_FRAMES, K1[i], k2[i], k3[i], lam, mu)
    return C_T


def measure_delta_y_stats(
    C_T_clean: np.ndarray,
    noise_type: str,
    param: float,
    n: int,
    root_seed: int,
    label: str,
) -> tuple[float, float, list[float]]:
    """Return (mean, std, list_of_delta_y) over n realisations."""
    dys = []
    for k in range(n):
        seed = derive_seed(root_seed, label, f"k={k}")
        rng = LCG(seed)
        if noise_type == "poisson":
            noisy = add_poisson_noise(rng, C_T_clean, alpha=param)
        else:
            noisy = add_gaussian_noise(rng, C_T_clean, sigma_rel=param)
        dys.append(compute_delta_y(noisy, C_T_clean))
    arr = np.array(dys)
    return float(arr.mean()), float(arr.std()), dys


def main():
    print("=" * 60)
    print("M4 Noise calibration experiment")
    print("=" * 60)

    C_T_clean = build_clean_tacs()
    print(f"\nClean TAC shape: {C_T_clean.shape}  "
          f"(N_REGIONS={N_REGIONS}, N_FRAMES={N_FRAMES})")
    print(f"TAC value range: [{C_T_clean.min():.4f}, {C_T_clean.max():.4f}]")

    results = {}

    for level_name, target in DELTA_Y_TARGETS.items():
        print(f"\n" + "-"*50)
        print(f"Level: {level_name}  (target delta_y = {target})")
        print("-"*50)
        results[level_name] = {"target_delta_y": target}

        # --- Poisson calibration ---
        t0 = time.time()
        alpha_cal = calibrate_poisson(
            C_T_clean, target,
            n_samples=N_CALIB,
            root_seed=ROOT_SEED,
        )
        t_calib = time.time() - t0

        mean_p, std_p, dys_p = measure_delta_y_stats(
            C_T_clean, "poisson", alpha_cal, N_VERIFY,
            ROOT_SEED, f"verify_poisson_{level_name}"
        )
        print(f"  Poisson: alpha = {alpha_cal:.4g}   "
              f"measured delta_y = {mean_p:.4f} ± {std_p:.4f}   "
              f"(target {target}, calib time {t_calib:.1f}s)")
        results[level_name]["poisson"] = {
            "alpha": alpha_cal,
            "mean_delta_y": mean_p,
            "std_delta_y": std_p,
            "all_delta_y": dys_p,
        }

        # --- Gaussian calibration ---
        t0 = time.time()
        sigma_cal = calibrate_gaussian(
            C_T_clean, target,
            n_samples=N_CALIB,
            root_seed=ROOT_SEED,
        )
        t_calib = time.time() - t0

        mean_g, std_g, dys_g = measure_delta_y_stats(
            C_T_clean, "gaussian", sigma_cal, N_VERIFY,
            ROOT_SEED, f"verify_gaussian_{level_name}"
        )
        print(f"  Gaussian: sigma_rel = {sigma_cal:.4g}   "
              f"measured delta_y = {mean_g:.4f} ± {std_g:.4f}   "
              f"(target {target}, calib time {t_calib:.1f}s)")
        results[level_name]["gaussian"] = {
            "sigma_rel": sigma_cal,
            "mean_delta_y": mean_g,
            "std_delta_y": std_g,
            "all_delta_y": dys_g,
        }

        # --- Poisson vs Gaussian comparison ---
        rel_diff = abs(mean_p - mean_g) / target
        print(f"  Poisson vs Gaussian mean delta_y: {mean_p:.4f} vs {mean_g:.4f}   "
              f"rel diff = {rel_diff:.3f} (w.r.t. target)")
        results[level_name]["poisson_vs_gaussian_rel_diff"] = rel_diff

    # --- Summary table ---
    print(f"\n{'='*60}")
    print("Summary table (target / measured Poisson / measured Gaussian):")
    print(f"{'Level':<15} {'Target':>10} {'Poi mean':>10} {'Poi std':>8} {'Gau mean':>10} {'Gau std':>8}")
    print("-" * 65)
    for level_name, r in results.items():
        target = r["target_delta_y"]
        mp, sp = r["poisson"]["mean_delta_y"], r["poisson"]["std_delta_y"]
        mg, sg = r["gaussian"]["mean_delta_y"], r["gaussian"]["std_delta_y"]
        print(f"{level_name:<15} {target:>10.4f} {mp:>10.4f} {sp:>8.4f} {mg:>10.4f} {sg:>8.4f}")

    # --- Save results ---
    out_path = save_json("m4", "noise_calibration", {
        "root_seed": ROOT_SEED,
        "n_calib_samples": N_CALIB,
        "n_verify_samples": N_VERIFY,
        "results": results,
    }, seed=ROOT_SEED)
    print(f"\nSaved: {out_path}")

    # --- Return calibrated params dict for downstream use ---
    return {
        level_name: {
            "poisson_alpha": r["poisson"]["alpha"],
            "gaussian_sigma_rel": r["gaussian"]["sigma_rel"],
        }
        for level_name, r in results.items()
    }


if __name__ == "__main__":
    main()
