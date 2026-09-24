"""M5: results narrative — the trend checklist from PLAN.md's M5 acceptance
criterion, computed from already-produced M4 result files (nothing here
re-runs the grid; it aggregates results/m4/*.json).

Stated *in advance*, per PLAN.md: agreement means qualitative trend
agreement, not digit matching — the imaging chain was cut, so absolute
noise levels are not comparable to the paper's.

Trends checked, each against the M4.3 grid (`table2_parameters.json`,
`grid_results.json`) and, for trend 2, also the M4.5 known-C_P arm
(`consistency_regularization.json`):

1. K1 recovers better than k2/k3 (expected for Setups B/C; Setup A is
   expected to *disagree*, per Proposition 12 — that disagreement is the
   M4.4 finding, not a bug).
2. Known C_P beats clean C_WB-only, which beats noisy C_WB.
3. The low-count setting fails, especially for the parameters of f (the
   plasma-fraction block m).

DECISIONS.md D-M5-4 records the scope caveat for trend 2: the M4.5 known-C_P
arm freezes m at truth and uses the corrected Morozov stopping rule
(D-M4-9), while the M4.3 grid (Setups B/C) fits all 23 parameters under the
uncorrected "rms" convention — the comparison is informative, not strictly
apples-to-apples, and this script reports both facts rather than hiding it.

Run: `python3 experiments/m5_trend_checklist.py`
Writes: results/m5/trend_checklist.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments._common import RESULTS_DIR, save_json

NOISE_LEVELS = ("noiseless", "high_count", "normal_count", "low_count")
DELTA_X_VALUES = (0.1, 0.2, 0.3, 0.4)


def load(path):
    return json.loads((RESULTS_DIR / path).read_text())


def trend1_K1_vs_k3(table2: dict, grid_cells: list):
    """For Setups B and C, mean(|K1_est-K1_true|/K1_true) should be smaller
    than mean(|k3_est-k3_true|/k3_true), averaged **per seed, then over
    seeds** (Setup A is reported separately and is expected to disagree,
    Prop 12).

    Deliberately uses the raw per-run records in grid_results.json, not
    table2_parameters.json's precomputed K1_mean/k3_mean. table2's K1_mean is
    the mean of the *estimates* across seeds; taking |mean(estimate) -
    truth| after that averaging silently cancels the ζ-scatter Proposition
    12 predicts for Setup A (positive and negative ζ deviations partly
    average out), which is exactly the effect this trend is supposed to
    detect. Averaging the per-seed |error| first, THEN across seeds, does
    not have that blind spot."""
    K1_true = np.array(table2["K1_true"])
    k3_true = np.array(table2["k3_true"])

    per_setup = {}
    for setup in ("A", "B", "C"):
        K1_rel_errs, k3_rel_errs = [], []
        for run in grid_cells:
            if run["setup"] != setup or run["diverged"]:
                continue
            K1_rec = run.get("K1_recovered")
            k3_rec = run.get("k3_recovered")
            if K1_rec is None or k3_rec is None:
                continue
            K1_rel_errs.append(float(np.mean(np.abs(np.array(K1_rec) - K1_true) / K1_true)))
            k3_rel_errs.append(float(np.mean(np.abs(np.array(k3_rec) - k3_true) / k3_true)))
        per_setup[setup] = {
            "n_runs_used": len(K1_rel_errs),
            "mean_K1_rel_error": float(np.mean(K1_rel_errs)) if K1_rel_errs else None,
            "mean_k3_rel_error": float(np.mean(k3_rel_errs)) if k3_rel_errs else None,
            "K1_better_than_k3": bool(np.mean(K1_rel_errs) < np.mean(k3_rel_errs)) if K1_rel_errs else None,
        }

    verdict_B = "AGREES" if per_setup["B"]["K1_better_than_k3"] else "DISAGREES"
    verdict_C = "AGREES" if per_setup["C"]["K1_better_than_k3"] else "DISAGREES"
    verdict_A = "DISAGREES (expected: Setup A has no arterial data at all, " \
                "so the K1/lambda ambiguity of Proposition 12 applies — this " \
                "disagreement IS the M4.4 finding)" if not per_setup["A"]["K1_better_than_k3"] \
                else "AGREES (unexpected under Proposition 12 — see M4.4 for the mechanism)"

    return {
        "per_setup": per_setup,
        "verdict_setup_B": verdict_B,
        "verdict_setup_C": verdict_C,
        "verdict_setup_A": verdict_A,
    }


def trend2_known_cp_vs_cwb(consistency: dict, grid_cells: list):
    """Known C_P (M4.5 arm) vs clean C_WB (Setup B) vs noisy C_WB (Setup C),
    at the matching delta_x=0.1 used by the M4.5 arm.

    Uses `rel_error_K_final` from the raw grid cells for Setup B/C — the
    SAME "12 kinetic parameters only" metric as the M4.5 arm's
    `metabolic_error` (both restrict to METABOLIC_START:). An earlier version
    of this script compared against table2's `mean_rel_error_total`, which
    is the full 23-parameter (including arterial lambda/mu and plasma
    fraction m) vector error — an apples-to-oranges comparison that
    understated Setup B/C's kinetic accuracy. Fixed; see DECISIONS.md
    D-M5-4."""
    rows = {}
    for noise in ("high_count", "normal_count"):
        known_cp_err = consistency["consistency"][noise]["error_mean"]
        setup_errs = {}
        for setup in ("B", "C"):
            errs = [c["rel_error_K_final"] for c in grid_cells
                    if c["setup"] == setup and c["noise_level"] == noise
                    and c["delta_x"] == 0.1 and not c["diverged"]]
            setup_errs[setup] = {"mean_rel_error_K": float(np.mean(errs)) if errs else None, "n_runs": len(errs)}

        b_err = setup_errs["B"]["mean_rel_error_K"]
        c_err = setup_errs["C"]["mean_rel_error_K"]
        rows[noise] = {
            "known_C_P_mean_kinetic_error": known_cp_err,
            "setup_B_clean_C_WB_mean_rel_error_K": b_err,
            "setup_B_n_runs": setup_errs["B"]["n_runs"],
            "setup_C_noisy_C_WB_mean_rel_error_K": c_err,
            "setup_C_n_runs": setup_errs["C"]["n_runs"],
            "known_cp_beats_clean_cwb": (known_cp_err < b_err) if b_err is not None and known_cp_err is not None else None,
            "clean_cwb_beats_noisy_cwb": (b_err < c_err) if b_err is not None and c_err is not None else None,
        }

    all_known_beats_clean = all(r["known_cp_beats_clean_cwb"] for r in rows.values() if r["known_cp_beats_clean_cwb"] is not None)
    all_clean_beats_noisy = all(r["clean_cwb_beats_noisy_cwb"] for r in rows.values() if r["clean_cwb_beats_noisy_cwb"] is not None)

    return {
        "rows": rows,
        "caveat": "known-C_P arm (M4.5) freezes the plasma-fraction block m "
                  "at truth and uses the corrected Morozov stopping rule "
                  "(delta_y*sqrt(n_obs), D-M4-9); Setup B/C (M4.3 grid) fit "
                  "all 23 parameters under the uncorrected rms convention, "
                  "and only 7-13 of 20 seeds survive (non-diverged) per cell "
                  "at normal_count/high_count. Informative, not a fully "
                  "controlled comparison (DECISIONS.md D-M5-4).",
        "verdict_known_cp_beats_clean_cwb": "AGREES" if all_known_beats_clean else "MIXED/DISAGREES",
        "verdict_clean_beats_noisy_cwb": "AGREES" if all_clean_beats_noisy else "MIXED/DISAGREES — see rows",
    }


def trend3_low_count_fails(grid_cells: list):
    """Divergence/non-convergence rate at low_count vs other noise levels,
    and the plasma-fraction (m) block's relative error specifically, on the
    rare low_count runs that do not diverge."""
    by_noise = {n: {"n_total": 0, "n_diverged": 0, "m_errors": []} for n in NOISE_LEVELS}
    for run in grid_cells:
        n = run["noise_level"]
        if n not in by_noise:
            continue
        by_noise[n]["n_total"] += 1
        if run["diverged"]:
            by_noise[n]["n_diverged"] += 1
        else:
            m_err = run.get("rel_error_m_final")
            if m_err is not None and np.isfinite(m_err):
                by_noise[n]["m_errors"].append(m_err)

    summary = {}
    for n, d in by_noise.items():
        summary[n] = {
            "n_total": d["n_total"],
            "n_diverged": d["n_diverged"],
            "divergence_rate": d["n_diverged"] / d["n_total"] if d["n_total"] else None,
            "n_survivors_with_m_error": len(d["m_errors"]),
            "mean_m_rel_error_on_survivors": float(np.mean(d["m_errors"])) if d["m_errors"] else None,
        }

    low_rate = summary["low_count"]["divergence_rate"]
    other_rates = [summary[n]["divergence_rate"] for n in NOISE_LEVELS if n != "low_count" and summary[n]["divergence_rate"] is not None]
    verdict = "AGREES" if (low_rate is not None and other_rates and low_rate >= max(other_rates)) else "NOT CONFIRMED"

    return {
        "by_noise_level": summary,
        "verdict": verdict,
        "caveat": "the divergence rate (near-total at low_count: 99%) is the "
                  "primary evidence for this trend, not the mean_m_err on "
                  "survivors. Only 3 of 60 low_count runs converge at all, so "
                  "their mean m-error is a 3-sample statistic dominated by "
                  "survivorship bias (the few non-diverged cases are not a "
                  "representative sample) — it should not be read as 'f "
                  "recovers fine at low_count', only 'almost nothing "
                  "survives to be measured'.",
    }


if __name__ == "__main__":
    print("Loading M4 result files...")
    table2 = load("m4/table2_parameters.json")
    consistency = load("m4/consistency_regularization.json")
    grid = load("m4/grid_results.json")

    print("\nTrend 1: K1 recovers better than k2/k3 (Setups B, C; A expected to disagree)")
    t1 = trend1_K1_vs_k3(table2, grid["cells"])
    for setup in ("A", "B", "C"):
        p = t1["per_setup"][setup]
        print(f"  Setup {setup}: mean K1 rel err={p['mean_K1_rel_error']:.4f}  "
              f"mean k3 rel err={p['mean_k3_rel_error']:.4f}  ({p['n_runs_used']} runs)")
    print(f"  verdict B: {t1['verdict_setup_B']}")
    print(f"  verdict C: {t1['verdict_setup_C']}")
    print(f"  verdict A: {t1['verdict_setup_A']}")

    print("\nTrend 2: known C_P beats clean C_WB-only, which beats noisy C_WB")
    t2 = trend2_known_cp_vs_cwb(consistency, grid["cells"])
    for noise, row in t2["rows"].items():
        print(f"  {noise}: known_C_P={row['known_C_P_mean_kinetic_error']:.4f}  "
              f"clean_C_WB={row['setup_B_clean_C_WB_mean_rel_error_K']:.4f} (n={row['setup_B_n_runs']})  "
              f"noisy_C_WB={row['setup_C_noisy_C_WB_mean_rel_error_K']:.4f} (n={row['setup_C_n_runs']})")
    print(f"  verdict known>clean: {t2['verdict_known_cp_beats_clean_cwb']}")
    print(f"  verdict clean>noisy: {t2['verdict_clean_beats_noisy_cwb']}")

    print("\nTrend 3: low-count setting fails, especially for f (plasma fraction m)")
    t3 = trend3_low_count_fails(grid["cells"])
    for n, s in t3["by_noise_level"].items():
        print(f"  {n}: divergence_rate={s['divergence_rate']:.2f}  "
              f"survivors={s['n_survivors_with_m_error']}  "
              f"mean_m_err={s['mean_m_rel_error_on_survivors']}")
    print(f"  verdict: {t3['verdict']}")

    out = save_json("m5", "trend_checklist", {
        "trend1_K1_vs_k3": t1,
        "trend2_known_cp_vs_cwb": t2,
        "trend3_low_count_fails": t3,
    })
    print("\nsummary saved:", out)
