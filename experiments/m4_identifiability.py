"""M4.4 — the identifiability signature experiment, noiseless stage.

Runs the five noiseless steps of remaining_task.md 4.4 and writes
`results/m4/identifiability_noiseless.json`:

  1-2. Fit with tissue TACs only; report the four K1_est/K1_true ratios and
       their SPREAD (the headline number of the milestone).
  3.   Confirm k2 and k3 are recovered accurately in the same runs.
  4.   Confirm C_P (i.e. lambda) comes out scaled by 1/zeta.
  5.   Add a single C_P measurement and show zeta -> 1, sweeping which of the
       four candidate blood-sample times is used (project rule 6: measure
       rather than assume that the choice does not matter).

Step 6 (repeat under noise) is a separate stage, run once these numbers are
reviewed.

    python3 experiments/m4_identifiability.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments._common import save_json  # noqa: E402
from src.config import BLOOD_SAMPLE_FRAME_INDICES, REGION_NAMES  # noqa: E402
from src.identifiability import (  # noqa: E402
    FIT_RESIDUAL_TOL,
    ROOT_SEED,
    T_FRAMES,
    run_identifiability_case,
    summarise,
)

DELTA_X_VALUES = (0.1, 0.2, 0.3, 0.4)
N_SEEDS = 20


def _fmt(v: float, spec: str = ".3e") -> str:
    return "n/a" if v is None or not np.isfinite(v) else format(v, spec)


def run_tissue_only_stage() -> dict:
    out = {}
    for dx in DELTA_X_VALUES:
        runs = [run_identifiability_case(delta_x=dx, seed_idx=k) for k in range(N_SEEDS)]
        out[f"dx={dx}"] = {"summary": summarise(runs), "runs": runs}
    return out


def run_single_cp_stage() -> dict:
    out = {}
    for fi in BLOOD_SAMPLE_FRAME_INDICES:
        for dx in DELTA_X_VALUES:
            runs = [
                run_identifiability_case(delta_x=dx, seed_idx=k, blood_frame_indices=(fi,))
                for k in range(N_SEEDS)
            ]
            out[f"frame={fi}_dx={dx}"] = {
                "t_minutes": float(T_FRAMES[fi]),
                "summary": summarise(runs),
                "runs": runs,
            }
    return out


def _print_tissue_only(stage: dict) -> None:
    print("\n" + "=" * 92)
    print("STEPS 1-4 — tissue TACs only, no arterial data (noiseless)")
    print("=" * 92)
    print(f"{'delta_x':>8} | {'acc':>5} {'div':>4} {'stall':>5} | "
          f"{'SPREAD max':>11} {'SPREAD med':>11} | {'zeta range':>17} | "
          f"{'K1 err':>9} {'k2 err':>9} {'k3 err':>9}")
    print("-" * 92)
    for dx in DELTA_X_VALUES:
        s = stage[f"dx={dx}"]["summary"]
        if not s.get("n_accepted"):
            print(f"{dx:>8} | {0:>5} {s['n_diverged']:>4} {s['n_stalled']:>5} | all runs rejected")
            continue
        print(f"{dx:>8} | {s['n_accepted']:>5} {s['n_diverged']:>4} {s['n_stalled']:>5} | "
              f"{_fmt(s['spread_K1_max']):>11} {_fmt(s['spread_K1_median']):>11} | "
              f"[{s['zeta_min']:.4f}, {s['zeta_max']:.4f}] | "
              f"{_fmt(s['K1_rel_error_max'], '.2e'):>9} {_fmt(s['k2_rel_error_max'], '.2e'):>9} "
              f"{_fmt(s['k3_rel_error_max'], '.2e'):>9}")
    print("\n'SPREAD' = max(K1 ratio)/min(K1 ratio) - 1 over the four regions, the")
    print("headline number: Proposition 12 says the four ratios must coincide.")
    print("'K1 err' vs 'k2/k3 err' is the second claim: K1 is wrong, k2/k3 are not.")


def _print_one_run_detail(stage: dict) -> None:
    """One fully worked example, so a reader can see the raw four ratios."""
    runs = stage["dx=0.1"]["runs"]
    r = next(x for x in runs if x["fit_accepted"])
    print("\n" + "-" * 92)
    print(f"Worked example — delta_x = 0.1, seed_idx = {r['seed_idx']}")
    print("-" * 92)
    print(f"{'region':>13} | {'K1_est/K1_true':>16} | {'lambda_true/lambda_est':>23}")
    for name, zk, zl in zip(REGION_NAMES, r["zeta_from_K1"], r["zeta_from_lambda"]):
        print(f"{name:>13} | {zk:>16.10f} | {zl:>23.10f}")
    print(f"{'spread':>13} | {r['spread_K1']['max_over_min_minus_1']:>16.3e} | "
          f"{r['spread_lambda']['max_over_min_minus_1']:>23.3e}")
    print(f"\n  zeta (from K1)      = {r['zeta_mean']:.10f}")
    print(f"  zeta (from lambda)  = {r['spread_lambda']['mean']:.10f}")
    print(f"  cross-check rel diff= {_fmt(r['zeta_K1_vs_lambda_rel_diff'])}   <- step 4")
    print(f"  k2 max rel error    = {_fmt(r['k2_rel_error_max'])}   <- step 3")
    print(f"  k3 max rel error    = {_fmt(r['k3_rel_error_max'])}   <- step 3")
    print(f"  relative residual   = {_fmt(r['rel_residual'])}")


def _print_single_cp(stage: dict) -> None:
    print("\n" + "=" * 92)
    print("STEP 5 — one C_P measurement added back (noiseless)")
    print("=" * 92)
    print(f"{'frame':>6} {'t (min)':>9} | {'delta_x':>8} | {'acc':>4} {'rej':>4} | "
          f"{'|zeta-1| max':>13} {'|zeta-1| med':>13} | {'K1 err max':>11}")
    print("-" * 92)
    for fi in BLOOD_SAMPLE_FRAME_INDICES:
        for dx in DELTA_X_VALUES:
            e = stage[f"frame={fi}_dx={dx}"]
            s = e["summary"]
            rej = s["n_runs"] - s.get("n_accepted", 0)
            if not s.get("n_accepted"):
                print(f"{fi:>6} {e['t_minutes']:>9.4f} | {dx:>8} | {0:>4} {rej:>4} | all runs rejected")
                continue
            print(f"{fi:>6} {e['t_minutes']:>9.4f} | {dx:>8} | {s['n_accepted']:>4} {rej:>4} | "
                  f"{_fmt(s['abs_zeta_minus_1_max']):>13} {_fmt(s['abs_zeta_minus_1_median']):>13} | "
                  f"{_fmt(s['K1_rel_error_max'], '.2e'):>11}")
    print("\nCompare |zeta-1| here against the tissue-only zeta range above: that")
    print("collapse is the paper's point — one blood draw removes the ambiguity.")


def main() -> None:
    t0 = time.time()
    print(f"M4.4 identifiability signature — noiseless stage")
    print(f"root seed {ROOT_SEED}, {N_SEEDS} seeds per cell, "
          f"acceptance: rel. residual <= {FIT_RESIDUAL_TOL:g}")

    tissue = run_tissue_only_stage()
    single = run_single_cp_stage()
    elapsed = time.time() - t0

    _print_tissue_only(tissue)
    _print_one_run_detail(tissue)
    _print_single_cp(single)

    path = save_json(
        "m4",
        "identifiability_noiseless",
        {
            "description": (
                "M4.4 steps 1-5: tissue-only fit (K1/lambda ambiguity) and its "
                "removal by a single C_P measurement. Noiseless."
            ),
            "n_seeds": N_SEEDS,
            "delta_x_values": list(DELTA_X_VALUES),
            "blood_frame_indices": list(BLOOD_SAMPLE_FRAME_INDICES),
            "fit_residual_tol": FIT_RESIDUAL_TOL,
            "runtime_seconds": elapsed,
            "tissue_only": tissue,
            "single_cp": single,
        },
        seed=ROOT_SEED,
    )
    print(f"\nRuntime {elapsed:.1f} s. Wrote {path}")


if __name__ == "__main__":
    main()
