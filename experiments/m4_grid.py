"""M4.3 — Full Monte Carlo grid experiment.

Runs: 3 setups × 4 noise levels × 4 delta_x × 20 seeds = 960 cells.

Produces:
  results/m4/grid_results.json    — raw per-cell data
  results/m4/table1_divergence.json  — divergence count table (analogue of paper Table 1)
  results/m4/table2_parameters.json  — parameter recovery table (analogue of paper Table 2)
  results/m4/figure7_error_trajectories.json — error trajectories by param type

Before launching the full grid, the script estimates the per-cell runtime
from 3 probe cells and prints the expected total time.

Usage:
    python experiments/m4_grid.py
    python experiments/m4_grid.py --dry-run   # prints timing estimate only
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

from experiments._common import save_json
from src.config import (
    METABOLIC_START,
    N_REGIONS,
    REGION_NAMES,
    ground_truth_vector,
)
from src.montecarlo import (
    DELTA_X_VALUES,
    N_SEEDS,
    NOISE_LEVELS,
    ROOT_SEED,
    SETUP_NAMES,
    cell_key,
    run_one_cell,
)

X_TRUE = ground_truth_vector()
DRY_RUN = "--dry-run" in sys.argv


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _timing_probe() -> float:
    """Return estimated seconds per cell from 3 probe runs."""
    probes = [
        ("A", "noiseless", 0.1, 0),
        ("B", "normal_count", 0.1, 0),
        ("C", "normal_count", 0.1, 0),
    ]
    times = []
    for setup, noise, dx, k in probes:
        t0 = time.perf_counter()
        run_one_cell(setup, noise, dx, k)
        times.append(time.perf_counter() - t0)
    return float(np.mean(times))


def _print_table1(divergence_table: dict):
    """Print Table 1 analogue: divergence counts."""
    print("\n--- Table 1 analogue: divergence counts (n_diverged / 20) ---")
    # Header
    header = f"{'Setup':<8} {'Noise':<15}"
    for dx in DELTA_X_VALUES:
        header += f" dx={dx}"
    print(header)
    print("-" * (8 + 15 + len(DELTA_X_VALUES) * 8))
    for setup in SETUP_NAMES:
        for noise in NOISE_LEVELS:
            row = f"{setup:<8} {noise:<15}"
            for dx in DELTA_X_VALUES:
                key = cell_key(setup, noise, dx)
                n_div = divergence_table.get(key, "?")
                row += f" {n_div:>6}"
            print(row)


def _print_table2(parameter_table: dict):
    """Print Table 2 analogue: mean ± std of K1, k2, k3 at normal_count, delta_x=0.3."""
    print("\n--- Table 2 analogue: recovered params at normal_count, delta_x=0.3 ---")
    print("(mean +/- std over non-diverged runs; ground truth in parentheses)")
    print()
    from src.config import REGION_KINETICS
    K1_true = [REGION_KINETICS[r][0] for r in REGION_NAMES]
    k2_true = [REGION_KINETICS[r][1] for r in REGION_NAMES]
    k3_true = [REGION_KINETICS[r][2] for r in REGION_NAMES]

    for setup in SETUP_NAMES:
        key = cell_key(setup, "normal_count", 0.3)
        entry = parameter_table.get(key)
        if entry is None:
            print(f"  Setup {setup}: no data")
            continue
        print(f"\n  Setup {setup}: n_converged={entry['n_converged']}/{N_SEEDS}")
        print(f"  {'Region':<15} {'K1 true':>10} {'K1 rec':>17} {'k2 true':>10} {'k2 rec':>17} {'k3 true':>10} {'k3 rec':>17}")
        print("  " + "-" * 90)
        for i, region in enumerate(REGION_NAMES):
            k1r = entry.get("K1_mean", [None]*N_REGIONS)[i]
            k1s = entry.get("K1_std", [None]*N_REGIONS)[i]
            k2r = entry.get("k2_mean", [None]*N_REGIONS)[i]
            k2s = entry.get("k2_std", [None]*N_REGIONS)[i]
            k3r = entry.get("k3_mean", [None]*N_REGIONS)[i]
            k3s = entry.get("k3_std", [None]*N_REGIONS)[i]
            def _fmt(v, s):
                if v is None:
                    return "    N/A"
                return f"{v:.4f}+/-{s:.4f}"
            print(f"  {region:<15} {K1_true[i]:>10.4f} {_fmt(k1r,k1s):>17} "
                  f"{k2_true[i]:>10.4f} {_fmt(k2r,k2s):>17} "
                  f"{k3_true[i]:>10.4f} {_fmt(k3r,k3s):>17}")


# ---------------------------------------------------------------------------
# Main grid runner
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("M4.3 Monte Carlo grid")
    print(f"  {len(SETUP_NAMES)} setups x {len(NOISE_LEVELS)} noise levels x "
          f"{len(DELTA_X_VALUES)} delta_x x {N_SEEDS} seeds = "
          f"{len(SETUP_NAMES)*len(NOISE_LEVELS)*len(DELTA_X_VALUES)*N_SEEDS} cells")
    print("=" * 60)

    # --- Timing estimate ---
    print("\nProbing per-cell runtime (3 cells)...")
    t_per_cell = _timing_probe()
    n_cells = len(SETUP_NAMES) * len(NOISE_LEVELS) * len(DELTA_X_VALUES) * N_SEEDS
    t_total = t_per_cell * n_cells
    print(f"  Estimated: {t_per_cell:.2f}s per cell => "
          f"total ~{t_total/60:.1f} min ({t_total:.0f}s)")

    if DRY_RUN:
        print("\n--dry-run: stopping here.")
        return

    # --- Run the grid ---
    all_cells: list[dict] = []
    divergence_table: dict[str, int] = {}    # cell_key -> n_diverged
    parameter_table: dict[str, dict] = {}   # cell_key -> stats (for Table 2)
    failure_log: list[dict] = []

    t_start = time.perf_counter()
    n_done = 0

    for setup in SETUP_NAMES:
        for noise in NOISE_LEVELS:
            for dx in DELTA_X_VALUES:
                key = cell_key(setup, noise, dx)
                cell_results = []
                n_div = 0

                for k in range(N_SEEDS):
                    cell = run_one_cell(setup, noise, dx, k)
                    cell_results.append(cell)
                    if cell["diverged"]:
                        n_div += 1
                        failure_log.append({
                            "setup": setup,
                            "noise_level": noise,
                            "delta_x": dx,
                            "seed_idx": k,
                            "seed": cell["seed"],
                            "n_iterations": cell["n_iterations"],
                            "failure_reason": cell.get("failure_reason"),
                        })
                    n_done += 1

                all_cells.extend(cell_results)
                divergence_table[key] = n_div

                # Build parameter-recovery stats for this cell
                converged = [c for c in cell_results if not c["diverged"]]
                nc = len(converged)
                if nc > 0:
                    K1_arr = np.array([c["K1_recovered"] for c in converged])
                    k2_arr = np.array([c["k2_recovered"] for c in converged])
                    k3_arr = np.array([c["k3_recovered"] for c in converged])
                    final_errs = [c["rel_error_total"][-1] for c in converged
                                  if c["rel_error_total"]]
                    parameter_table[key] = {
                        "n_converged": nc,
                        "n_diverged": n_div,
                        "K1_mean": K1_arr.mean(axis=0).tolist(),
                        "K1_std":  K1_arr.std(axis=0).tolist(),
                        "k2_mean": k2_arr.mean(axis=0).tolist(),
                        "k2_std":  k2_arr.std(axis=0).tolist(),
                        "k3_mean": k3_arr.mean(axis=0).tolist(),
                        "k3_std":  k3_arr.std(axis=0).tolist(),
                        "median_rel_error_total": float(np.median(final_errs)) if final_errs else None,
                        "mean_rel_error_total": float(np.mean(final_errs)) if final_errs else None,
                    }
                else:
                    parameter_table[key] = {
                        "n_converged": 0,
                        "n_diverged": n_div,
                    }

                elapsed = time.perf_counter() - t_start
                eta = (elapsed / n_done) * (n_cells - n_done) if n_done > 0 else 0
                print(f"  {key:<45} n_div={n_div:>2}/{N_SEEDS}  "
                      f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s")

    t_total_actual = time.perf_counter() - t_start
    print(f"\nGrid complete in {t_total_actual:.1f}s ({t_total_actual/60:.2f} min)")

    # --- Print Table 1 and Table 2 analogues ---
    _print_table1(divergence_table)
    _print_table2(parameter_table)

    # --- Build Figure 7 analogue data ---
    # Error trajectories by parameter type, for normal_count, delta_x=0.3, all three setups
    fig7_data: dict = {}
    for setup in SETUP_NAMES:
        key = cell_key(setup, "normal_count", 0.3)
        converged = [c for c in all_cells
                     if c["setup"] == setup
                     and c["noise_level"] == "normal_count"
                     and c["delta_x"] == 0.3
                     and not c["diverged"]]
        if converged:
            fig7_data[f"setup_{setup}"] = {
                "rel_error_K":      [c["rel_error_K"] for c in converged],
                "rel_error_lambda": [c["rel_error_lambda"] for c in converged],
                "rel_error_mu":     [c["rel_error_mu"] for c in converged],
                "rel_error_m":      [c["rel_error_m"] for c in converged],
                "rel_error_total":  [c["rel_error_total"] for c in converged],
            }

    # --- Save everything ---
    # 1. Full raw grid (large; include per-cell trajectories stripped to save space)
    def _strip(cell):
        """Keep everything except the full trajectory lists (keep just final value)."""
        c = dict(cell)
        for key_ in ("rel_error_total", "rel_error_K", "rel_error_lambda",
                     "rel_error_mu", "rel_error_m"):
            traj = c.get(key_)
            if traj:
                c[key_ + "_final"] = traj[-1]
                c[key_ + "_initial"] = traj[0] if traj else None
            c.pop(key_, None)
        c.pop("x_final", None)  # 23 floats per cell, skip in summary
        return c

    stripped_cells = [_strip(c) for c in all_cells]

    p1 = save_json("m4", "grid_results", {
        "n_cells": n_cells,
        "total_time_s": t_total_actual,
        "cells": stripped_cells,
        "failure_log": failure_log,
    }, seed=ROOT_SEED)
    print(f"\nSaved grid results: {p1}")

    p2 = save_json("m4", "table1_divergence", {
        "description": "Divergence counts per (setup, noise_level, delta_x) cell out of 20 runs",
        "setups": list(SETUP_NAMES),
        "noise_levels": list(NOISE_LEVELS),
        "delta_x_values": list(DELTA_X_VALUES),
        "counts": divergence_table,
    }, seed=ROOT_SEED)
    print(f"Saved Table 1: {p2}")

    p3 = save_json("m4", "table2_parameters", {
        "description": "Parameter recovery stats per cell: mean+/-std over non-diverged runs",
        "region_names": list(REGION_NAMES),
        "K1_true":  [float(X_TRUE[METABOLIC_START + 3*i]) for i in range(N_REGIONS)],
        "k2_true":  [float(X_TRUE[METABOLIC_START + 3*i+1]) for i in range(N_REGIONS)],
        "k3_true":  [float(X_TRUE[METABOLIC_START + 3*i+2]) for i in range(N_REGIONS)],
        "table": parameter_table,
    }, seed=ROOT_SEED)
    print(f"Saved Table 2: {p3}")

    p4 = save_json("m4", "figure7_error_trajectories", {
        "description": "Error trajectories by parameter type at normal_count, delta_x=0.3",
        "noise_level": "normal_count",
        "delta_x": 0.3,
        "data": fig7_data,
    }, seed=ROOT_SEED)
    print(f"Saved Figure 7 data: {p4}")

    # --- Append divergent runs to logs/failures.md ---
    if failure_log:
        _append_failures(failure_log)

    # --- Final trend check ---
    _trend_check(divergence_table, parameter_table)


def _append_failures(failure_log: list[dict]):
    """Append M4.3 divergent run entries to logs/failures.md."""
    failures_path = Path("logs/failures.md")
    lines = [
        "\n\n## M4.3 Monte Carlo grid divergences\n",
        f"Total: {len(failure_log)} divergent cells out of 960\n\n",
        "| setup | noise | delta_x | seed_idx | seed | n_iter | reason |\n",
        "|---|---|---|---|---|---|---|\n",
    ]
    for f in failure_log:
        reason = str(f.get("failure_reason") or "non-finite iterate").replace("|", "/")
        lines.append(
            f"| {f['setup']} | {f['noise_level']} | {f['delta_x']} "
            f"| {f['seed_idx']} | {f['seed']} | {f['n_iterations']} | {reason} |\n"
        )
    with open(failures_path, "a", encoding="utf-8") as fh:
        fh.writelines(lines)
    print(f"\nAppended {len(failure_log)} divergences to logs/failures.md")


def _trend_check(divergence_table: dict, parameter_table: dict):
    """Report the paper's expected qualitative trends with measured verdicts."""
    print("\n--- Trend check (remaining_task.md Part 5) ---")

    # Trend 1: divergence increases from high_count -> normal_count -> low_count
    print("\nTrend 1: More divergences at lower count levels")
    for setup in SETUP_NAMES:
        divs = {}
        for noise in NOISE_LEVELS[1:]:  # skip noiseless
            total = sum(
                divergence_table.get(cell_key(setup, noise, dx), 0)
                for dx in DELTA_X_VALUES
            )
            divs[noise] = total
        trend_ok = divs["high_count"] <= divs["normal_count"] <= divs["low_count"]
        print(f"  Setup {setup}: high={divs['high_count']} normal={divs['normal_count']} "
              f"low={divs['low_count']}  -> {'AGREES' if trend_ok else 'DISAGREES'}")

    # Trend 2: Setup A (known f) should have fewer divergences than C (noisy C_WB)
    print("\nTrend 2: Setup A diverges less than Setup C")
    for noise in NOISE_LEVELS[1:]:
        da = sum(divergence_table.get(cell_key("A", noise, dx), 0) for dx in DELTA_X_VALUES)
        dc = sum(divergence_table.get(cell_key("C", noise, dx), 0) for dx in DELTA_X_VALUES)
        print(f"  {noise}: A_div={da} C_div={dc}  -> {'AGREES' if da <= dc else 'DISAGREES'}")

    # Trend 3: At normal_count, K1 relative error < k3 relative error (K1 easier to recover)
    print("\nTrend 3: K1 recovered more accurately than k3 at normal_count, delta_x=0.3")
    for setup in SETUP_NAMES:
        key = cell_key(setup, "normal_count", 0.3)
        entry = parameter_table.get(key)
        if entry and entry.get("n_converged", 0) > 0:
            K1_true_arr = np.array([X_TRUE[METABOLIC_START + 3*i] for i in range(N_REGIONS)])
            k3_true_arr = np.array([X_TRUE[METABOLIC_START + 3*i + 2] for i in range(N_REGIONS)])
            K1_mean = np.array(entry.get("K1_mean", [float("nan")]*N_REGIONS))
            k3_mean = np.array(entry.get("k3_mean", [float("nan")]*N_REGIONS))
            K1_rel = float(np.mean(np.abs(K1_mean - K1_true_arr) / K1_true_arr))
            k3_rel = float(np.mean(np.abs(k3_mean - k3_true_arr) / k3_true_arr))
            print(f"  Setup {setup}: mean |K1_err/K1|={K1_rel:.4f}  mean |k3_err/k3|={k3_rel:.4f}  "
                  f"-> {'AGREES' if K1_rel <= k3_rel else 'DISAGREES'}")
        else:
            print(f"  Setup {setup}: insufficient converged runs to check")


if __name__ == "__main__":
    main()
