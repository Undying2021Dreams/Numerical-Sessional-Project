"""M4.4 step 6 — does the identifiability signature survive noise?

Repeats the tissue-only fit and its single-C_P counterpart at every noise
level, under BOTH discrepancy-principle conventions (DECISIONS.md D-M4-9):

  rms      — M4.3's convention. `compute_delta_y` returns an RMS but
             `run_irgnm` compares it against a 2-norm, so the rule is
             sqrt(n_obs)=10x too strict and never fires: every run reaches
             max_iter = 300. Kept because M4.3's counts use it.
  morozov  — delta_y passed as the actual noise NORM (delta_y*sqrt(n_obs)),
             making the comparison dimensionally consistent. The rule fires,
             and early stopping acts as the regularisation it is meant to be.

Both are run because they answer different questions, and the difference is
itself a result: the K1-ratio coincidence is an ASYMPTOTIC property of the
fit, so a rule that stops early truncates it before the iterates settle onto
the null manifold.

The single-C_P arm uses frame index 3 only — the noiseless sweep
(`m4_identifiability.py`) measured that sample to be 20-60x better than the
later ones, so there is nothing left to learn from re-sweeping all four here.

    python3 experiments/m4_identifiability_noisy.py

Writes results/m4/identifiability_noisy.json.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments._common import save_json  # noqa: E402
from src.identifiability import (  # noqa: E402
    NOISY_RESIDUAL_FACTOR,
    ROOT_SEED,
    STOPPING_CONVENTIONS,
    run_identifiability_case,
    summarise,
)

NOISE_LEVELS = ("noiseless", "high_count", "normal_count", "low_count")
DELTA_X_VALUES = (0.1, 0.2, 0.3, 0.4)
N_SEEDS = 20
BEST_BLOOD_FRAME = 3


def _fmt(v, spec=".2e"):
    return "n/a" if v is None or not np.isfinite(v) else format(v, spec)


def run_arm(blood_frame_indices: tuple[int, ...]) -> dict:
    out = {}
    for stopping in STOPPING_CONVENTIONS:
        for noise in NOISE_LEVELS:
            for dx in DELTA_X_VALUES:
                runs = [
                    run_identifiability_case(
                        delta_x=dx, seed_idx=k, noise_level=noise,
                        stopping=stopping, blood_frame_indices=blood_frame_indices,
                    )
                    for k in range(N_SEEDS)
                ]
                out[f"stop={stopping}_noise={noise}_dx={dx}"] = {
                    "summary": summarise(runs),
                    "runs": runs,
                }
    return out


def _print_arm(arm: dict, title: str, zeta_col: str) -> None:
    print("\n" + "=" * 106)
    print(title)
    print("=" * 106)
    for stopping in STOPPING_CONVENTIONS:
        print(f"\n--- stopping = {stopping} ---")
        print(f"{'noise':>13} | {'d_x':>4} | {'acc':>3} {'div':>3} {'triv':>4} {'rej':>3} | "
              f"{'spread max':>10} {'spread med':>10} | {zeta_col:>21} | "
              f"{'k2 err':>9} {'k3 err':>9} | {'iters':>6}")
        print("-" * 106)
        for noise in NOISE_LEVELS:
            for dx in DELTA_X_VALUES:
                e = arm[f"stop={stopping}_noise={noise}_dx={dx}"]
                s, runs = e["summary"], e["runs"]
                acc = s.get("n_accepted", 0)
                triv = s.get("n_trivial", 0)
                rej = s["n_runs"] - acc - s["n_diverged"] - triv
                if not acc:
                    print(f"{noise:>13} | {dx:>4} | {0:>3} {s['n_diverged']:>3} {triv:>4} {rej:>3} | "
                          f"{'— no accepted fits —':>60}")
                    continue
                it = int(np.median([r["n_iterations"] for r in runs if r["fit_accepted"]]))
                if zeta_col.startswith("|zeta"):
                    zc = f"{_fmt(s['abs_zeta_minus_1_max'])} / {_fmt(s['abs_zeta_minus_1_median'])}"
                else:
                    zc = f"[{s['zeta_min']:.3f}, {s['zeta_max']:.3f}]"
                print(f"{noise:>13} | {dx:>4} | {acc:>3} {s['n_diverged']:>3} {triv:>4} {rej:>3} | "
                      f"{_fmt(s['spread_K1_max']):>10} {_fmt(s['spread_K1_median']):>10} | "
                      f"{zc:>21} | {_fmt(s['k2_rel_error_max']):>9} "
                      f"{_fmt(s['k3_rel_error_max']):>9} | {it:>6}")


def main() -> None:
    t0 = time.time()
    print("M4.4 step 6 — the signature under noise")
    print(f"root seed {ROOT_SEED}, {N_SEEDS} seeds per cell, "
          f"noisy acceptance {NOISY_RESIDUAL_FACTOR}x the noise floor")

    tissue = run_arm(())
    single = run_arm((BEST_BLOOD_FRAME,))
    elapsed = time.time() - t0

    _print_arm(tissue, "TISSUE ONLY — does the K1/lambda signature survive noise?", "zeta range")
    _print_arm(single, f"ONE C_P MEASUREMENT (frame {BEST_BLOOD_FRAME}) — is it still removed?",
               "|zeta-1| max / med")

    path = save_json(
        "m4",
        "identifiability_noisy",
        {
            "description": (
                "M4.4 step 6: the identifiability signature at every noise level, "
                "under both discrepancy-principle conventions (D-M4-9)."
            ),
            "n_seeds": N_SEEDS,
            "noise_levels": list(NOISE_LEVELS),
            "delta_x_values": list(DELTA_X_VALUES),
            "stopping_conventions": list(STOPPING_CONVENTIONS),
            "best_blood_frame": BEST_BLOOD_FRAME,
            "noisy_residual_factor": NOISY_RESIDUAL_FACTOR,
            "runtime_seconds": elapsed,
            "tissue_only": tissue,
            "single_cp": single,
        },
        seed=ROOT_SEED,
    )
    print(f"\nRuntime {elapsed:.1f} s. Wrote {path}")


if __name__ == "__main__":
    main()
