"""M2 experiments: forward model (closed form, quadrature, three-way agreement,
near-degeneracy sweep, late-time slope, Figure 2 analogue).

Track B usage (clearly marked): `scipy.integrate.solve_ivp` is the third,
independent code path for the three-way agreement check (PLAN.md M2's
centrepiece acceptance test). Nothing in `src/` imports scipy.

Run: `python3 experiments/m2_forward_model.py`
Writes: results/m2/*.json, results/m2/*.png
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments._common import save_json
from src.config import (
    ARTERIAL_LAMBDA,
    ARTERIAL_MU,
    F_A,
    F_XI1,
    F_XI2,
    REGION_KINETICS,
    REGION_NAMES,
    V_B,
    frame_midtimes_minutes,
)
from src.forward_model import (
    C_PET,
    C_WB_from_C_P,
    GradedGridSpec,
    arterial_input,
    closed_form_C_T,
    closed_form_C_T_derivative,
    parent_plasma_fraction,
    quadrature_C_T,
    _phi1,
)
from src.plotting import save_fig, set_style

LAM = np.array(ARTERIAL_LAMBDA)
MU = np.array(ARTERIAL_MU)
FRAME_T = frame_midtimes_minutes()
GRID = GradedGridSpec(n=1601, q=3.0)


# ---------------------------------------------------------------------------
# D-M2-1: phi1 numerical stability study
# ---------------------------------------------------------------------------
def _phi1_series_reference(x: float, terms: int = 30) -> float:
    total, term, fact = 0.0, 1.0, 1.0
    for k in range(terms):
        fact *= k + 1
        total += term / fact
        term *= x
    return total


def phi1_stability_study():
    # Restricted to |x| <= 1: the 30-term Taylor series reference only
    # converges to full double precision there (at x=10 it still has ~1e-8
    # truncation error of its own, since 10^30/30! is not yet negligible —
    # irrelevant to the actual question anyway, since the naive form's
    # cancellation problem is specific to x near 0, not large x).
    xs = np.array(
        [1.0, 1e-1, 1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12, 1e-13, 1e-14, 1e-15, 1e-16, 1e-20, 1e-100, 1e-300]
    )
    rows = []
    for x in xs:
        ref = _phi1_series_reference(float(x))
        naive = (math.exp(x) - 1.0) / x if x != 0 else float("nan")
        stable = float(_phi1(np.array([x]))[0])
        rel_err_naive = abs(naive - ref) / abs(ref)
        rel_err_stable = abs(stable - ref) / abs(ref)
        rows.append(
            {
                "x": float(x),
                "series_reference": ref,
                "naive_rel_error": rel_err_naive,
                "expm1_stable_rel_error": rel_err_stable,
            }
        )
    # crossover: largest x (rows are listed descending in x) at which the
    # naive form's relative error first exceeds 0.5 (complete breakdown) —
    # and, separately, first exceeds 1e-8 (the point at which it has
    # already lost ~8 of ~16 available significant digits, a meaningful
    # degradation well before total breakdown).
    crossover_x_50pct = None
    crossover_x_1e8 = None
    for row in rows:
        if row["naive_rel_error"] > 0.5 and crossover_x_50pct is None:
            crossover_x_50pct = row["x"]
        if row["naive_rel_error"] > 1e-8 and crossover_x_1e8 is None:
            crossover_x_1e8 = row["x"]
    return {
        "rows": rows,
        "naive_crossover_x_50pct_relative_error": crossover_x_50pct,
        "naive_crossover_x_1e-8_relative_error": crossover_x_1e8,
    }


# ---------------------------------------------------------------------------
# D-M2-3: grid-refinement study
# ---------------------------------------------------------------------------
def grid_refinement_study():
    K1, k2, k3 = REGION_KINETICS["frontal"]
    t_points = {"earliest": FRAME_T[0], "mid": FRAME_T[12], "latest": FRAME_T[-1]}
    ns = [51, 101, 201, 401, 801, 1601, 3201, 6401, 12801]
    qs = [1.0, 2.0, 3.0, 4.0, 5.0]

    results = {}
    for label, t_end in t_points.items():
        t_arr = np.array([t_end])
        exact = float(closed_form_C_T(t_arr, K1, k2, k3, LAM, MU)[0])
        rows = []
        for q in qs:
            for n in ns:
                gs = GradedGridSpec(n=n, q=q)
                val, fb = quadrature_C_T(t_arr, K1, k2, k3, LAM, MU, gs)
                rel = abs(float(val[0]) - exact) / abs(exact)
                rows.append({"q": q, "n": n, "rel_error": rel, "fallback": any(fb)})
        results[label] = {"t": float(t_end), "exact": exact, "rows": rows}
    return results


# ---------------------------------------------------------------------------
# Three-way agreement (centrepiece)
# ---------------------------------------------------------------------------
def _solve_ivp_C_T(t_eval, K1, k2, k3, lam, mu, method):
    a = k2 + k3

    def rhs(t, y):
        C_F, C_B = y
        C_P_t = float(np.sum(lam * np.exp(mu * t)))
        return [K1 * C_P_t - a * C_F, k3 * C_F]

    t_max = float(np.max(t_eval))
    t0 = time.perf_counter()
    sol = solve_ivp(
        rhs, (0.0, t_max), y0=[0.0, 0.0], t_eval=np.sort(t_eval),
        method=method, rtol=1e-12, atol=1e-14, max_step=t_max / 2000.0,
    )
    dt = time.perf_counter() - t0
    assert sol.success, f"solve_ivp[{method}] failed: {sol.message}"
    C_F, C_B = sol.y
    return C_F + C_B, dt


def three_way_agreement():
    per_region = {}
    overall = {"closed_vs_quad": 0.0, "closed_vs_ivp_radau": 0.0, "quad_vs_ivp_radau": 0.0,
               "closed_vs_ivp_rk45": 0.0}
    any_fallback = False

    for name in REGION_NAMES:
        K1, k2, k3 = REGION_KINETICS[name]
        ct_closed = closed_form_C_T(FRAME_T, K1, k2, k3, LAM, MU)
        ct_quad, fb = quadrature_C_T(FRAME_T, K1, k2, k3, LAM, MU, GRID)
        any_fallback = any_fallback or any(fb)
        ct_ivp_radau, t_radau = _solve_ivp_C_T(FRAME_T, K1, k2, k3, LAM, MU, "Radau")
        ct_ivp_rk45, t_rk45 = _solve_ivp_C_T(FRAME_T, K1, k2, k3, LAM, MU, "RK45")

        scale = np.maximum(np.abs(ct_closed), 1e-8)
        d_cq = float(np.max(np.abs(ct_closed - ct_quad) / scale))
        d_ci_radau = float(np.max(np.abs(ct_closed - ct_ivp_radau) / scale))
        d_qi_radau = float(np.max(np.abs(ct_quad - ct_ivp_radau) / scale))
        d_ci_rk45 = float(np.max(np.abs(ct_closed - ct_ivp_rk45) / scale))

        per_region[name] = {
            "closed_vs_quad_max_rel_diff": d_cq,
            "closed_vs_ivp_radau_max_rel_diff": d_ci_radau,
            "quad_vs_ivp_radau_max_rel_diff": d_qi_radau,
            "closed_vs_ivp_rk45_max_rel_diff": d_ci_rk45,
            "solve_ivp_radau_time_s": t_radau,
            "solve_ivp_rk45_time_s": t_rk45,
            "fallback_fired": any(fb),
        }
        overall["closed_vs_quad"] = max(overall["closed_vs_quad"], d_cq)
        overall["closed_vs_ivp_radau"] = max(overall["closed_vs_ivp_radau"], d_ci_radau)
        overall["quad_vs_ivp_radau"] = max(overall["quad_vs_ivp_radau"], d_qi_radau)
        overall["closed_vs_ivp_rk45"] = max(overall["closed_vs_ivp_rk45"], d_ci_rk45)

    return {"per_region": per_region, "overall_max": overall, "any_fallback_fired": any_fallback}


# ---------------------------------------------------------------------------
# Near-degeneracy sweep (for the required plot)
# ---------------------------------------------------------------------------
def near_degeneracy_sweep():
    K1, k2, k3 = REGION_KINETICS["frontal"]
    a = k2 + k3
    t_eval = np.array([10.0, 30.0, 57.5])
    mu5_values = np.linspace(-1.3 * a, -0.7 * a, 601)
    lam5 = 0.5

    closed_vals = np.empty((len(mu5_values), len(t_eval)))
    quad_vals = np.empty((len(mu5_values), len(t_eval)))
    for i, mu5 in enumerate(mu5_values):
        lam = np.concatenate([LAM, [lam5]])
        mu = np.concatenate([MU, [mu5]])
        closed_vals[i] = closed_form_C_T(t_eval, K1, k2, k3, lam, mu)
        qv, fb = quadrature_C_T(t_eval, K1, k2, k3, lam, mu, GRID)
        quad_vals[i] = qv

    max_rel_diff = float(np.max(np.abs(closed_vals - quad_vals) / np.maximum(np.abs(closed_vals), 1e-8)))
    return {
        "a": a,
        "t_eval": t_eval.tolist(),
        "mu5_values": mu5_values.tolist(),
        "closed_vals": closed_vals.tolist(),
        "quad_vals": quad_vals.tolist(),
        "max_rel_diff_closed_vs_quad": max_rel_diff,
    }


# ---------------------------------------------------------------------------
# Late-time slope (D-M2-4), all regions
# ---------------------------------------------------------------------------
def late_time_slope_table():
    mu_slowest = float(np.max(MU))
    t_late = np.array([FRAME_T[-1]])
    t_huge = np.array([800.0])
    rows = {}
    for name in REGION_NAMES:
        K1, k2, k3 = REGION_KINETICS[name]
        Ki_classical = K1 * k3 / (k2 + k3)
        Ki_corrected = K1 * (k3 + mu_slowest) / (k2 + k3 + mu_slowest)

        cp_late = float(arterial_input(t_late, LAM, MU)[0])
        ctp_late = float(closed_form_C_T_derivative(t_late, K1, k2, k3, LAM, MU)[0])
        ratio_late = ctp_late / cp_late

        cp_huge = float(arterial_input(t_huge, LAM, MU)[0])
        ctp_huge = float(closed_form_C_T_derivative(t_huge, K1, k2, k3, LAM, MU)[0])
        ratio_huge = ctp_huge / cp_huge

        rows[name] = {
            "K1": K1, "k2": k2, "k3": k3,
            "Ki_classical_plateau_assumption": Ki_classical,
            "Ki_corrected_formula": Ki_corrected,
            "measured_ratio_at_t_last_frame": ratio_late,
            "measured_ratio_at_t_800": ratio_huge,
            "rel_error_classical_at_t_last_frame": abs(ratio_late - Ki_classical) / Ki_classical,
            "rel_error_corrected_at_t_last_frame": abs(ratio_late - Ki_corrected) / Ki_corrected,
            "rel_error_corrected_at_t_800": abs(ratio_huge - Ki_corrected) / Ki_corrected,
        }
    return {"mu_slowest": mu_slowest, "t_last_frame": float(FRAME_T[-1]), "regions": rows}


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def make_degeneracy_figure(sweep: dict):
    import matplotlib.pyplot as plt

    set_style()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    mu5 = np.array(sweep["mu5_values"])
    a = sweep["a"]
    for j, t in enumerate(sweep["t_eval"]):
        ax.plot(mu5, np.array(sweep["closed_vals"])[:, j], "-", label=f"closed form, t={t:g}")
        ax.plot(mu5, np.array(sweep["quad_vals"])[:, j], "--", label=f"quadrature, t={t:g}")
    ax.axvline(-a, color="k", linestyle=":", linewidth=1, label=f"$\\mu_5 = -(k_2+k_3) = {-a:.3f}$")
    ax.set_xlabel("$\\mu_5$")
    ax.set_ylabel("$C_T(t)$")
    ax.set_title("Near-degeneracy sweep: closed form vs quadrature across $\\mu_5 = -(k_2+k_3)$")
    ax.legend(fontsize=7, ncol=2)
    return fig


def make_figure2_analogue():
    import matplotlib.pyplot as plt

    set_style()
    t_dense = np.concatenate([np.linspace(0.01, 1.0, 300), np.linspace(1.0, 100.0, 700)])
    f_vals = parent_plasma_fraction(t_dense, F_A, F_XI1, F_XI2)
    cp_vals = arterial_input(t_dense, LAM, MU)
    cwb_vals = C_WB_from_C_P(cp_vals, f_vals)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    axes[0, 0].plot(t_dense, f_vals)
    axes[0, 0].set_xscale("log")
    axes[0, 0].set_title("Parent plasma fraction $f(t)$")
    axes[0, 0].set_xlabel("Time (min)")

    axes[0, 1].plot(t_dense, cwb_vals)
    axes[0, 1].set_xscale("log")
    axes[0, 1].set_title("Total arterial blood tracer concentration $C_{WB}$")
    axes[0, 1].set_xlabel("Time (min)")

    axes[1, 0].plot(t_dense, cp_vals)
    axes[1, 0].set_xscale("log")
    axes[1, 0].set_title("Plasma concentration $C_P$")
    axes[1, 0].set_xlabel("Time (min)")

    for name in REGION_NAMES:
        K1, k2, k3 = REGION_KINETICS[name]
        ct = closed_form_C_T(t_dense, K1, k2, k3, LAM, MU)
        cwb_at_t = C_WB_from_C_P(arterial_input(t_dense, LAM, MU), parent_plasma_fraction(t_dense, F_A, F_XI1, F_XI2))
        cpet = C_PET(ct, cwb_at_t, V_B)
        axes[1, 1].plot(t_dense, cpet, label=name)
    axes[1, 1].set_xscale("log")
    axes[1, 1].set_title("Tissue time activity curve $C_{PET}$ (4 regions)")
    axes[1, 1].set_xlabel("Time (min)")
    axes[1, 1].legend(fontsize=8)

    fig.tight_layout()
    return fig


if __name__ == "__main__":
    print("phi1 numerical stability study...")
    phi1_study = phi1_stability_study()
    for row in phi1_study["rows"]:
        print(f"  x={row['x']:.1e}  naive_rel_err={row['naive_rel_error']:.3e}  expm1_rel_err={row['expm1_stable_rel_error']:.3e}")
    print("  naive form crosses 1e-8 relative error at approximately x =", phi1_study["naive_crossover_x_1e-8_relative_error"])
    print("  naive form crosses 50% (complete breakdown) relative error at approximately x =", phi1_study["naive_crossover_x_50pct_relative_error"])

    print("\nGrid refinement study (this takes a little while)...")
    grid_study = grid_refinement_study()
    for label, data in grid_study.items():
        print(f"  --- {label} (t={data['t']:.4f}) ---")
        for row in data["rows"]:
            if row["n"] in (401, 801, 1601, 3201):
                print(f"    q={row['q']:.0f} n={row['n']:6d}: rel_error={row['rel_error']:.3e}")

    print("\nThree-way agreement (closed form / quadrature / scipy.solve_ivp)...")
    agreement = three_way_agreement()
    for name, r in agreement["per_region"].items():
        print(f"  {name:12s}: closed_vs_quad={r['closed_vs_quad_max_rel_diff']:.3e}  "
              f"closed_vs_ivp[Radau]={r['closed_vs_ivp_radau_max_rel_diff']:.3e}  "
              f"closed_vs_ivp[RK45]={r['closed_vs_ivp_rk45_max_rel_diff']:.3e}  "
              f"quad_vs_ivp[Radau]={r['quad_vs_ivp_radau_max_rel_diff']:.3e}")
    print("  overall max:", agreement["overall_max"])
    print("  any simpson fallback fired:", agreement["any_fallback_fired"])

    print("\nNear-degeneracy sweep...")
    sweep = near_degeneracy_sweep()
    print("  max rel diff closed vs quadrature during sweep:", sweep["max_rel_diff_closed_vs_quad"])

    print("\nLate-time slope table...")
    slope_table = late_time_slope_table()
    for name, r in slope_table["regions"].items():
        print(f"  {name:12s}: Ki_classical={r['Ki_classical_plateau_assumption']:.5f}  "
              f"Ki_corrected={r['Ki_corrected_formula']:.5f}  "
              f"measured@t_last={r['measured_ratio_at_t_last_frame']:.5f}  "
              f"rel_err_classical={r['rel_error_classical_at_t_last_frame']:.3e}  "
              f"rel_err_corrected={r['rel_error_corrected_at_t_last_frame']:.3e}")

    fig1 = make_degeneracy_figure(sweep)
    p1 = save_fig(fig1, "m2", "near_degeneracy_sweep")
    print("\nfigure saved:", p1)

    fig2 = make_figure2_analogue()
    p2 = save_fig(fig2, "m2", "figure2_analogue")
    print("figure saved:", p2)

    out = save_json(
        "m2",
        "forward_model_benchmark",
        {
            "phi1_stability_study": phi1_study,
            "grid_refinement_study": grid_study,
            "three_way_agreement": agreement,
            "near_degeneracy_sweep_summary": {
                k: v for k, v in sweep.items() if k not in ("closed_vals", "quad_vals", "mu5_values")
            },
            "late_time_slope_table": slope_table,
        },
    )
    print("summary saved:", out)
