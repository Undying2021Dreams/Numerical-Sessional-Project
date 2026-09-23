"""M4.5: noise consistency and the variance effect of IRGNM regularisation.

Run with a fixed root seed through src.identifiability's seeded observations.
The consistency arm uses the dimensionally correct Morozov noise norm and one
exact arterial sample, so the K1/lambda ambiguity is removed. The variance
arm compares paired noisy observations at a fixed linearisation point and
regularisation iteration. It isolates the variance caused by observation
noise; failed full nonlinear fits are counted separately.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments._common import RESULTS_DIR, save_json  # noqa: E402
from src.config import METABOLIC_START, config_hash  # noqa: E402
from src.identifiability import (  # noqa: E402
    ROOT_SEED, TISSUE_ONLY_MASK, T_FRAMES, X_TRUE, build_observations,
    make_initial_guess, run_identifiability_case,
)
from src.irgnm import DEFAULT_SCHEDULE, DEFAULT_TAU, RegularizationSchedule, irgnm_step, run_irgnm  # noqa: E402

NOISE_LEVELS = ("noiseless", "high_count", "normal_count", "low_count")
NOISY_LEVELS = NOISE_LEVELS[1:]
N_SEEDS = 20
DELTA_X = 0.1
BLOOD_FRAME = 3
REG_ITERATION = 70


def metabolic_error(x: np.ndarray) -> float:
    """Relative Euclidean error of the 12 kinetic parameters."""
    t = X_TRUE[METABOLIC_START:]
    d = x[METABOLIC_START:] - t
    return float(np.sqrt(d @ d) / np.sqrt(t @ t))


def consistency() -> dict:
    out = {}
    for noise in NOISE_LEVELS:
        rows = []
        for k in range(N_SEEDS):
            r = run_identifiability_case(
                delta_x=DELTA_X, seed_idx=k, blood_frame_indices=(BLOOD_FRAME,),
                noise_level=noise, stopping="morozov",
            )
            rows.append({
                "seed_idx": k, "seed": r["seed"], "accepted": r["fit_accepted"],
                "diverged": r["diverged"], "trivial_stop": r["trivial_stop"],
                "iterations": r["n_iterations"], "delta_y": r["delta_y"],
                "kinetic_relative_error": metabolic_error(np.asarray(r["x_final"]))
                if r["fit_accepted"] else None,
            })
        errors = np.asarray([r["kinetic_relative_error"] for r in rows
                             if r["kinetic_relative_error"] is not None])
        out[noise] = {
            "n_accepted": len(errors), "n_diverged": sum(r["diverged"] for r in rows),
            "n_trivial": sum(r["trivial_stop"] for r in rows),
            "delta_y_mean": float(np.mean([r["delta_y"] for r in rows])),
            "error_mean": float(np.mean(errors)) if len(errors) else None,
            "error_std": float(np.std(errors, ddof=1)) if len(errors) > 1 else None,
            "error_median": float(np.median(errors)) if len(errors) else None,
            "rows": rows,
        }
    return out


def variance() -> dict:
    # Identical x0 and Jacobian for all realisations. Only y changes.
    x0 = make_initial_guess(DELTA_X, 0)
    reg_on = DEFAULT_SCHEDULE.diag(REG_ITERATION)
    reg_off = np.zeros_like(reg_on)
    out = {}
    for noise in NOISY_LEVELS:
        estimates = {"on": [], "off": []}
        for k in range(N_SEEDS):
            obs = build_observations(blood_frame_indices=(BLOOD_FRAME,),
                                     noise_level=noise, seed_idx=k)
            for mode, diag in (("on", reg_on), ("off", reg_off)):
                x1, _, _ = irgnm_step(
                    x0, x0, obs["y"], T_FRAMES, obs["s_blood"],
                    obs["C_WB_data"], diag, active_mask=TISSUE_ONLY_MASK,
                    include_blood=True,
                )
                estimates[mode].append(x1[METABOLIC_START:] / X_TRUE[METABOLIC_START:])
        stats = {}
        for mode in ("on", "off"):
            arr = np.asarray(estimates[mode])
            per_parameter = np.var(arr, axis=0, ddof=1)
            stats[mode] = {
                "relative_parameter_variance": per_parameter.tolist(),
                "variance_trace": float(np.sum(per_parameter)),
                "mean_relative_error": float(np.mean(
                    np.sqrt(np.mean((arr - 1.0) ** 2, axis=1)))),
            }
        stats["off_over_on_variance_ratio"] = (
            stats["off"]["variance_trace"] / stats["on"]["variance_trace"])
        out[noise] = stats
    return out


def full_fit_variance() -> dict:
    """Paired nonlinear on/off fits; variance uses only common survivors."""
    zero = RegularizationSchedule(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    out = {}
    for noise in ("high_count", "normal_count"):
        rows = []
        for k in range(N_SEEDS):
            obs = build_observations(blood_frame_indices=(BLOOD_FRAME,),
                                     noise_level=noise, seed_idx=k)
            x0 = make_initial_guess(DELTA_X, k)
            pair = {"seed_idx": k}
            for mode, schedule in (("on", DEFAULT_SCHEDULE), ("off", zero)):
                r = run_irgnm(
                    x0, obs["y"], T_FRAMES, obs["s_blood"], obs["C_WB_data"],
                    schedule=schedule, tau=DEFAULT_TAU,
                    delta_y=obs["delta_y"] * np.sqrt(len(obs["y"])),
                    max_iter=300, x_true=X_TRUE, active_mask=TISSUE_ONLY_MASK,
                    include_blood=True,
                )
                accepted = (not r["diverged"] and r["converged_at"] is not None
                            and r["converged_at"] >= 1)
                pair[mode] = {
                    "accepted": accepted, "diverged": bool(r["diverged"]),
                    "converged_at": r["converged_at"],
                    "relative_kinetics": (
                        (r["x_final"][METABOLIC_START:] /
                         X_TRUE[METABOLIC_START:]).tolist() if accepted else None),
                }
            rows.append(pair)
        common = [r for r in rows if r["on"]["accepted"] and r["off"]["accepted"]]
        stats = {"n_on_accepted": sum(r["on"]["accepted"] for r in rows),
                 "n_off_accepted": sum(r["off"]["accepted"] for r in rows),
                 "n_off_diverged": sum(r["off"]["diverged"] for r in rows),
                 "n_common": len(common), "rows": rows}
        if len(common) > 1:
            for mode in ("on", "off"):
                arr = np.asarray([r[mode]["relative_kinetics"] for r in common])
                stats[f"{mode}_variance_trace_common"] = float(
                    np.sum(np.var(arr, axis=0, ddof=1)))
            stats["off_over_on_variance_ratio_common"] = (
                stats["off_variance_trace_common"] / stats["on_variance_trace_common"])
        out[noise] = stats
    return out


def plot(cons: dict, var: dict) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    labels = ["noiseless", "high", "normal", "low"]
    for i, noise in enumerate(NOISE_LEVELS):
        s = cons[noise]
        if s["n_accepted"]:
            axes[0].errorbar(i, s["error_mean"], yerr=s["error_std"], fmt="o", capsize=4)
        else:
            axes[0].text(i, 0.03, "0 fits\n(20 trivial)", ha="center", va="center", fontsize=8)
    axes[0].set_xticks(range(4), labels)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("kinetic relative error (mean ± SD)")
    axes[0].set_title("Reconstruction error by noise level")

    x = np.arange(3)
    width = 0.35
    axes[1].bar(x - width / 2, [var[n]["on"]["variance_trace"] for n in NOISY_LEVELS],
                width, label="regularised")
    axes[1].bar(x + width / 2, [var[n]["off"]["variance_trace"] for n in NOISY_LEVELS],
                width, label="unregularised")
    axes[1].set_xticks(x, ["high", "normal", "low"])
    axes[1].set_yscale("log")
    axes[1].set_ylabel("sum of relative parameter variances")
    axes[1].set_title("Paired one-step variance, n=20")
    axes[1].legend()
    path = RESULTS_DIR / "m4" / "consistency_regularization.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, metadata={
        "Description": f"M4.5; seed={ROOT_SEED}; config_hash={config_hash()}"
    })
    plt.close(fig)
    return path


def main() -> None:
    import json

    if "--plot-only" in sys.argv:
        data = json.loads((RESULTS_DIR / "m4" / "consistency_regularization.json").read_text())
        print(f"wrote {plot(data['consistency'], data['variance'])}")
        return
    cons = consistency()
    var = variance()
    full = full_fit_variance()
    path = save_json("m4", "consistency_regularization", {
        "delta_x": DELTA_X, "blood_frame": BLOOD_FRAME,
        "n_seeds": N_SEEDS, "reg_iteration": REG_ITERATION,
        "stopping": "morozov", "consistency": cons, "variance": var,
        "full_fit_variance": full,
    }, seed=ROOT_SEED)
    figure = None if "--no-plot" in sys.argv else plot(cons, var)
    for noise in NOISE_LEVELS:
        s = cons[noise]
        print(f"{noise}: accepted={s['n_accepted']}/20, diverged={s['n_diverged']}, "
              f"trivial={s['n_trivial']}, mean error={s['error_mean']}, "
              f"SD={s['error_std']}")
    for noise in NOISY_LEVELS:
        s = var[noise]
        print(f"{noise}: variance on={s['on']['variance_trace']:.6g}, "
              f"off={s['off']['variance_trace']:.6g}, "
              f"off/on={s['off_over_on_variance_ratio']:.1f}")
    for noise, s in full.items():
        print(f"full fit {noise}: common={s['n_common']}, on accepted={s['n_on_accepted']}, "
              f"off accepted={s['n_off_accepted']}, off diverged={s['n_off_diverged']}, "
              f"on variance={s.get('on_variance_trace_common')}, "
              f"off variance={s.get('off_variance_trace_common')}")
    print(f"wrote {path}" + (f" and {figure}" if figure else ""))


if __name__ == "__main__":
    main()
