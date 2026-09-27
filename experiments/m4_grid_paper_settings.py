"""M4.3 grid rerun under the paper's own settings, to test the explanation of
the Table 1 gap (ours 498/960 divergences vs the paper's 97/960).

Each variant switches on more of the differences listed in DECISIONS.md D-M5-7,
so the gap can be attributed step by step:

  original               M4.3 exactly (rerun here so every variant shares one machine)
  stopping_fix           + stopping rule gets the noise norm, 200-step cap on noisy data
  paper_settings         + per-setup regularisation and tau (paper p. 22),
                           Setup A gets the clean blood block (paper's reduced setup)
  paper_settings_blood25 + blood readings at all 25 frames, as in the paper

Every run is counted two ways: our "diverged" (an iterate became non-finite)
and the paper's criterion (final error not below the initial error).
The original M4.3 results in results/m4/grid_results.json are not touched.

Usage:
    python experiments/m4_grid_paper_settings.py
    python experiments/m4_grid_paper_settings.py --variants original paper_settings
Writes: results/m4/grid_paper_settings.json
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

from experiments._common import save_json
from src.montecarlo import (
    DELTA_X_VALUES, N_SEEDS, NOISE_LEVELS, ROOT_SEED, SETUP_NAMES, cell_key, run_one_cell,
)

VARIANTS: dict[str, dict[str, bool]] = {
    "original": {},
    "stopping_fix": {"fixed_stopping": True, "paper_max_iter": True},
    "paper_settings": {"fixed_stopping": True, "paper_max_iter": True,
                       "paper_hyperparams": True, "setup_a_blood": True},
    "paper_settings_blood25": {"fixed_stopping": True, "paper_max_iter": True,
                               "paper_hyperparams": True, "setup_a_blood": True,
                               "blood_at_all_frames": True},
}

# Paper Table 1 (p. 20): experiments out of 20 dropped for divergence,
# "reduced setup / full setup with noiseless C_WB / full setup with noisy C_WB",
# i.e. our Setups A / B / C. Reference data for comparison only.
PAPER_TABLE1: dict[str, dict[float, tuple[int, int, int]]] = {
    "noiseless":    {0.4: (0, 1, 0), 0.3: (0, 0, 0), 0.2: (0, 0, 0), 0.1: (0, 0, 0)},
    "high_count":   {0.4: (0, 4, 2), 0.3: (0, 5, 3), 0.2: (0, 2, 0), 0.1: (0, 3, 0)},
    "normal_count": {0.4: (2, 5, 3), 0.3: (2, 7, 3), 0.2: (2, 4, 0), 0.1: (2, 5, 0)},
    "low_count":    {0.4: (0, 7, 4), 0.3: (0, 6, 5), 0.2: (0, 5, 3), 0.1: (0, 4, 8)},
}


def paper_counts() -> dict[str, int]:
    return {cell_key(s, n, dx): PAPER_TABLE1[n][dx][i]
            for n in NOISE_LEVELS for dx in DELTA_X_VALUES for i, s in enumerate(SETUP_NAMES)}


def run_variant(name: str, options: dict[str, bool]) -> dict:
    nonfinite: dict[str, int] = {}
    no_improvement: dict[str, int] = {}
    median_final_error: dict[str, float | None] = {}
    t0 = time.perf_counter()
    for setup in SETUP_NAMES:
        for noise in NOISE_LEVELS:
            for dx in DELTA_X_VALUES:
                key = cell_key(setup, noise, dx)
                runs = [run_one_cell(setup, noise, dx, k, **options) for k in range(N_SEEDS)]
                nonfinite[key] = sum(r["diverged"] for r in runs)
                no_improvement[key] = sum(r["no_improvement"] for r in runs)
                kept = [r["rel_error_total"][-1] for r in runs if not r["no_improvement"]]
                median_final_error[key] = float(np.median(kept)) if kept else None
        print(f"  [{name}] setup {setup} done, {time.perf_counter() - t0:.0f}s", flush=True)
    return {
        "options": options,
        "time_s": time.perf_counter() - t0,
        "nonfinite": nonfinite,
        "no_improvement": no_improvement,
        "median_final_rel_error_improved_runs": median_final_error,
    }


def totals(counts: dict[str, int]) -> dict:
    by_noise = {n: sum(counts[cell_key(s, n, dx)] for s in SETUP_NAMES for dx in DELTA_X_VALUES)
                for n in NOISE_LEVELS}
    by_setup = {s: sum(counts[cell_key(s, n, dx)] for n in NOISE_LEVELS for dx in DELTA_X_VALUES)
                for s in SETUP_NAMES}
    return {"total": sum(counts.values()), "by_noise": by_noise, "by_setup": by_setup}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", nargs="*", default=list(VARIANTS))
    args = parser.parse_args()

    paper = paper_counts()
    out = {"paper_table1": {"counts": paper, **totals(paper)}, "variants": {}}
    for name in args.variants:
        print(f"Running variant {name}: {VARIANTS[name] or 'M4.3 defaults'}", flush=True)
        res = run_variant(name, VARIANTS[name])
        res["totals_nonfinite"] = totals(res["nonfinite"])
        res["totals_no_improvement"] = totals(res["no_improvement"])
        out["variants"][name] = res

    p = out["paper_table1"]
    print(f"\n{'variant':<24}{'non-finite':>12}{'paper def.':>12}   by noise (paper def.)  by setup A/B/C (paper def.)")
    print(f"{'paper Table 1':<24}{'':>12}{p['total']:>12}   "
          f"{'/'.join(str(v) for v in p['by_noise'].values()):<22} "
          f"{'/'.join(str(v) for v in p['by_setup'].values())}")
    for name, res in out["variants"].items():
        a, b = res["totals_nonfinite"], res["totals_no_improvement"]
        print(f"{name:<24}{a['total']:>12}{b['total']:>12}   "
              f"{'/'.join(str(v) for v in b['by_noise'].values()):<22} "
              f"{'/'.join(str(v) for v in b['by_setup'].values())}")
    print("(by noise = noiseless/high/normal/low, each out of 240; by setup, each out of 320)")

    path = save_json("m4", "grid_paper_settings", out, seed=ROOT_SEED)
    print("summary saved:", path)


if __name__ == "__main__":
    main()
