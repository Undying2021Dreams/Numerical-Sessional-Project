"""M4.4 — the identifiability signature experiment (paper's Proposition 12).

The tissue-data block F^1 depends on K1^i and lambda_j only through their
product: C_T is linear in K1 (eq. 1) and C_P is linear in lambda
(Definition 2).  Scaling every K1^i by zeta while scaling every lambda_j by
1/zeta therefore leaves F^1 *exactly* invariant, for any zeta > 0 — this is
already verified at the function level and in the Jacobian's singular
structure by `tests/test_null_space.py` (M3).

M4.4 makes the much stronger claim: the ambiguity is visible in the
*fitted parameters* of an actual IRGNM run.  Fit with tissue TACs only and

  * every K1^i comes out multiplied by the SAME constant zeta,
  * lambda comes out divided by that same zeta,
  * k2 and k3 are nonetheless recovered accurately,
  * and a single C_P measurement collapses zeta to 1.

Two notes on how this is set up, both forced by the model rather than chosen:

1. The plasma-fraction parameters m = (A, xi1, xi2) are FROZEN at ground
   truth in the tissue-only runs.  F^1 has identically zero dependence on m
   (see `analytic_jacobian`: the m columns of the F^1 rows are never
   written), so leaving them free does not fit them — the regularisation
   term simply snaps them back to x0 on the first step.  Freezing them via
   `active_mask` is the same mechanism M4.2 Setup A already uses.
2. "A single C_P measurement" needs no new forward operator.  The F^2 block
   is `C_WB_data(s)*f_m(s) - C_P(lambda,mu)(s)`; with m frozen at truth,
   f_m is the true f, so F^2 reduces exactly to
   `C_P_true(s) - C_P(lambda,mu)(s)` — a literal measurement of C_P at s.
   So step 5 is just `include_blood=True` with `s_blood` truncated to one
   time point.  See DECISIONS.md D-M4-6.

This module is pure computation (Track A: no scipy, no numpy.linalg solvers,
no numpy.random).  All file I/O lives in `experiments/m4_identifiability.py`.
"""
from __future__ import annotations

import numpy as np

from src.config import (
    LAMBDA_SLICE,
    METABOLIC_START,
    M_SLICE,
    N_PARAMS,
    N_REGIONS,
    frame_midtimes_minutes,
    ground_truth_vector,
)
from src.forward_model import (
    C_WB_from_C_P,
    arterial_input,
    parent_plasma_fraction,
)
from src.irgnm import DEFAULT_SCHEDULE, DEFAULT_TAU, project, run_irgnm
from src.jacobian import forward_operator, unpack
from src.montecarlo import POISSON_ALPHA
from src.noise import add_poisson_noise, compute_delta_y
from src.rng import LCG, derive_seed, standard_normal

# ---------------------------------------------------------------------------
# Fixed experiment data (module level: identical to src/montecarlo.py's, and
# deliberately recomputed here rather than imported so this module stands on
# its own if the MC harness changes).
# ---------------------------------------------------------------------------
X_TRUE: np.ndarray = ground_truth_vector()
T_FRAMES: np.ndarray = frame_midtimes_minutes()

_LAM_TRUE, _MU_TRUE, _M_TRUE, _K1_TRUE, _K2_TRUE, _K3_TRUE = unpack(X_TRUE)

ROOT_SEED: int = 20240401
MAX_ITER: int = 300

#: Acceptance gate for a NOISELESS fit, on the relative residual
#: ||y - F(x_final)|| / ||y||.  `run_irgnm`'s own divergence flag only checks
#: that the iterates stayed finite, which is necessary but not sufficient: at
#: delta_y = 0 the discrepancy principle cannot fire, so a run that stalls far
#: from the data still exits after `max_iter` with `diverged=False`.  Measured
#: over 20 seeds at delta_x = 0.1 (see handoffs/RUN_M4.md, M4.4): 18 runs land
#: at relative residual 1.9e-10 to 2.3e-9, one returns NaN (correctly flagged),
#: and one stalls at 3.05e+03 — a gap of twelve orders of magnitude, so the
#: threshold is not delicate.  1e-6 sits well inside that gap and means "the
#: fit reproduces the data to six significant figures".  This TIGHTENS
#: acceptance (a stalled fit is counted as a failure, not averaged into the
#: statistics); it does not loosen any existing tolerance, and it is local to
#: this module — `src/irgnm.py` and M4.3's published counts are untouched.
#: See DECISIONS.md D-M4-8.
FIT_RESIDUAL_TOL: float = 1e-6

#: Acceptance gate for a NOISY fit under the `stopping="rms"` convention, as a
#: multiple of the noise floor ||noise|| / ||y|| = delta_y*sqrt(n_obs)/||y||.
#: Under that convention the discrepancy principle never fires (see
#: STOPPING_CONVENTIONS below), so there is no stopping-rule-based criterion to
#: defer to. Measured over the 240 noisy tissue-only cells: non-diverged fits
#: cluster at 0.78-0.97 x the noise floor (just below it, as expected once the
#: iterates start fitting noise), while failures sit at 5.6-10.9 x. 2.0 lies in
#: that gap with ~2x headroom on both sides. See DECISIONS.md D-M4-10.
NOISY_RESIDUAL_FACTOR: float = 2.0

#: A `stopping="morozov"` run whose discrepancy rule fires at iteration 0 has
#: taken no IRGNM step at all: `x_final` is bit-identical to the (projected)
#: initial guess, so its "zeta" is the initial guess's zeta and says nothing
#: about identifiability. Measured at low_count, delta_x = 0.1: this happens in
#: 20 of 20 runs, because tau*delta_y*sqrt(n_obs) = 6.8*0.073*10 = 5.0 already
#: exceeds the initial residual. That is the discrepancy principle behaving
#: correctly — "the data is too noisy to improve on your guess" — but it is not
#: a fit, so it is rejected and counted separately. See DECISIONS.md D-M4-11.
#: The two stopping conventions run side by side in M4.4 step 6 (D-M4-9).
#:   "rms"     — pass delta_y through as `compute_delta_y` returns it (an RMS).
#:               This is M4.3's convention, kept so the counts stay comparable.
#:               Because `run_irgnm` tests ||r|| <= tau*delta_y and ||r|| is a
#:               2-norm, the rule is sqrt(n_obs) = 10x too strict and never
#:               fires; every run reaches `max_iter`.
#:   "morozov" — pass delta_y*sqrt(n_obs), i.e. the actual noise NORM, making
#:               the comparison dimensionally consistent. The rule then fires at
#:               iteration ~24-71 and early stopping acts as the regularisation
#:               it is meant to be.
STOPPING_CONVENTIONS: tuple[str, ...] = ("rms", "morozov")

#: Freeze the three plasma-fraction parameters; see note 1 in the module docstring.
TISSUE_ONLY_MASK: np.ndarray = np.ones(N_PARAMS, dtype=bool)
TISSUE_ONLY_MASK[M_SLICE] = False


# ---------------------------------------------------------------------------
# Observation building
# ---------------------------------------------------------------------------
def _blood_data(s_blood: np.ndarray) -> np.ndarray:
    """Ground-truth C_WB at the given sample times (fixed problem data)."""
    return C_WB_from_C_P(
        arterial_input(s_blood, _LAM_TRUE, _MU_TRUE),
        parent_plasma_fraction(s_blood, *_M_TRUE),
    )


def build_observations(
    *,
    blood_frame_indices: tuple[int, ...] = (),
    noise_level: str = "noiseless",
    seed_idx: int = 0,
    root_seed: int = ROOT_SEED,
) -> dict:
    """Assemble the observation vector for one identifiability run.

    Parameters
    ----------
    blood_frame_indices : frame indices (into `frame_midtimes_minutes()`) at
        which a C_P measurement is available.  Empty tuple -> tissue-only
        (the F^2 block is dropped entirely, `include_blood=False`).
    noise_level : 'noiseless', 'high_count', 'normal_count' or 'low_count'.
        Poisson-derived TAC noise, matching M4.3's choice (D-M4-1).  The
        blood measurements are left noiseless here: step 5 of M4.4 asks what
        an *exact* C_P value does to zeta, so adding blood noise would
        confound the two effects.  See DECISIONS.md D-M4-7.

    Returns a dict with keys y, s_blood, C_WB_data, include_blood, delta_y.
    """
    include_blood = len(blood_frame_indices) > 0
    if include_blood:
        s_blood = T_FRAMES[np.array(blood_frame_indices, dtype=int)]
    else:
        s_blood = np.empty(0, dtype=np.float64)
    C_WB_data = _blood_data(s_blood)

    y_clean = forward_operator(X_TRUE, T_FRAMES, s_blood, C_WB_data, include_blood=include_blood)

    if noise_level == "noiseless":
        return {
            "y": y_clean,
            "s_blood": s_blood,
            "C_WB_data": C_WB_data,
            "include_blood": include_blood,
            "delta_y": 0.0,
        }

    if noise_level not in POISSON_ALPHA:
        raise ValueError(f"unknown noise_level {noise_level!r}")

    n_tac = N_REGIONS * len(T_FRAMES)
    C_T_clean = y_clean[:n_tac].reshape(N_REGIONS, len(T_FRAMES))

    tac_seed = derive_seed(
        root_seed, "exp=m44", f"blood={blood_frame_indices}",
        f"noise={noise_level}", "tac", f"k={seed_idx}",
    )
    C_T_noisy = add_poisson_noise(LCG(tac_seed), C_T_clean, alpha=POISSON_ALPHA[noise_level])
    delta_y = compute_delta_y(C_T_noisy, C_T_clean)

    y = y_clean.copy()
    y[:n_tac] = C_T_noisy.ravel()
    return {
        "y": y,
        "s_blood": s_blood,
        "C_WB_data": C_WB_data,
        "include_blood": include_blood,
        "delta_y": float(delta_y),
    }


def initial_guess_seed(
    delta_x: float,
    seed_idx: int,
    *,
    root_seed: int = ROOT_SEED,
    label: str = "exp=m44",
) -> int:
    """The derived seed behind one run's initial guess, so a failed run can be
    reproduced in isolation from `logs/failures.md` (same convention as the
    M4.3 grid)."""
    return derive_seed(root_seed, label, f"dx={delta_x}", "x0", f"k={seed_idx}")


def make_initial_guess(
    delta_x: float,
    seed_idx: int,
    *,
    root_seed: int = ROOT_SEED,
    label: str = "exp=m44",
) -> np.ndarray:
    """Perturbed initial guess, same recipe as `src.montecarlo.run_one_cell`
    (paper Section 6: x0 = x_true * (1 + sigma*gamma), sigma a random sign,
    gamma ~ delta_x + sqrt(delta_x/4) * N(0,1)), with m reset to truth
    because m is frozen in these runs."""
    rng = LCG(initial_guess_seed(delta_x, seed_idx, root_seed=root_seed, label=label))
    sigma = np.where(rng.uniform_array(N_PARAMS) < 0.5, -1.0, 1.0)
    gamma = delta_x + np.sqrt(delta_x / 4.0) * standard_normal(rng, N_PARAMS)
    x0 = project(X_TRUE * (1.0 + sigma * gamma))
    x0[M_SLICE] = X_TRUE[M_SLICE]
    return x0


# ---------------------------------------------------------------------------
# zeta extraction and the spread statistics
# ---------------------------------------------------------------------------
def zeta_from_K1(x_final: np.ndarray) -> np.ndarray:
    """The four ratios K1_est^i / K1_true^i.  Proposition 12 says these must
    coincide; their spread is the headline number of M4.4."""
    K1_est = np.asarray(
        [x_final[METABOLIC_START + 3 * i] for i in range(N_REGIONS)], dtype=np.float64
    )
    return K1_est / _K1_TRUE


def zeta_from_lambda(x_final: np.ndarray) -> np.ndarray:
    """The four ratios lambda_true_j / lambda_est_j.  Proposition 12 says
    lambda_est = lambda_true / zeta, so this is an INDEPENDENT estimate of
    the same zeta — an internal cross-check that the compensation really
    happened in C_P and not somewhere else."""
    lam_est = np.asarray(x_final[LAMBDA_SLICE], dtype=np.float64)
    # A diverged iterate can drive a lambda component to exactly 0. Return NaN
    # for that component rather than raising or emitting a divide-by-zero
    # warning; `spread` already reports NaN for a non-finite ratio set.
    safe = np.where(lam_est == 0.0, np.nan, lam_est)
    return _LAM_TRUE / safe


def spread(ratios: np.ndarray) -> dict:
    """Three ways of quantifying 'do these four numbers coincide?'.

    `max_over_min_minus_1` is the headline (max relative deviation between
    any two of them); `rel_range` and `cv` are reported alongside so the
    number cannot be made to look good by a lucky choice of statistic.
    Returns NaNs rather than raising if the ratios are not finite or change
    sign (a diverged run).
    """
    r = np.asarray(ratios, dtype=np.float64)
    if not np.all(np.isfinite(r)) or np.min(r) <= 0.0:
        nan = float("nan")
        return {"max_over_min_minus_1": nan, "rel_range": nan, "cv": nan,
                "mean": float(np.mean(r)) if np.all(np.isfinite(r)) else nan}
    mean = float(np.mean(r))
    return {
        "max_over_min_minus_1": float(np.max(r) / np.min(r) - 1.0),
        "rel_range": float((np.max(r) - np.min(r)) / mean),
        "cv": float(np.std(r) / mean),
        "mean": mean,
    }


def metabolic_relative_errors(x_final: np.ndarray) -> dict:
    """Per-region relative errors of K1, k2, k3 — the point being that k2 and
    k3 are small while K1 is not."""
    K1 = np.asarray([x_final[METABOLIC_START + 3 * i + 0] for i in range(N_REGIONS)])
    k2 = np.asarray([x_final[METABOLIC_START + 3 * i + 1] for i in range(N_REGIONS)])
    k3 = np.asarray([x_final[METABOLIC_START + 3 * i + 2] for i in range(N_REGIONS)])
    return {
        "K1_rel_error": (np.abs(K1 - _K1_TRUE) / _K1_TRUE).tolist(),
        "k2_rel_error": (np.abs(k2 - _K2_TRUE) / _K2_TRUE).tolist(),
        "k3_rel_error": (np.abs(k3 - _K3_TRUE) / _K3_TRUE).tolist(),
        "K1_rel_error_max": float(np.max(np.abs(K1 - _K1_TRUE) / _K1_TRUE)),
        "k2_rel_error_max": float(np.max(np.abs(k2 - _K2_TRUE) / _K2_TRUE)),
        "k3_rel_error_max": float(np.max(np.abs(k3 - _K3_TRUE) / _K3_TRUE)),
    }


def fit_accepted(
    *,
    diverged: bool,
    rel_residual: float,
    converged_at: int | None,
    noise_floor_rel: float = 0.0,
    stopping: str = "rms",
) -> bool:
    """Was this run a usable fit?

    Noiseless (`noise_floor_rel <= 0`): the discrepancy principle cannot fire,
    so gate on the relative residual at `FIT_RESIDUAL_TOL`.
    Noisy, `stopping="morozov"`: the rule works, so use it — the paper's own
    criterion.
    Noisy, `stopping="rms"`: the rule never fires, so gate on the residual
    reaching the noise floor, within `NOISY_RESIDUAL_FACTOR`.
    """
    if diverged or not np.isfinite(rel_residual):
        return False
    if noise_floor_rel <= 0.0:
        return rel_residual <= FIT_RESIDUAL_TOL
    if stopping == "morozov":
        return converged_at is not None and converged_at >= 1
    return rel_residual <= NOISY_RESIDUAL_FACTOR * noise_floor_rel


# ---------------------------------------------------------------------------
# One run, end to end
# ---------------------------------------------------------------------------
def run_identifiability_case(
    *,
    delta_x: float,
    seed_idx: int,
    blood_frame_indices: tuple[int, ...] = (),
    noise_level: str = "noiseless",
    stopping: str = "rms",
    root_seed: int = ROOT_SEED,
    max_iter: int = MAX_ITER,
    solver: str = "qr",
) -> dict:
    """Fit once and return the fitted parameters together with every
    identifiability statistic derived from them.

    `blood_frame_indices=()` is the tissue-only case (M4.4 steps 1-4, 6);
    a one-element tuple is the single-C_P-measurement case (step 5).
    `stopping` selects the discrepancy-principle convention; see
    `STOPPING_CONVENTIONS`. It has no effect on noiseless runs.
    """
    if stopping not in STOPPING_CONVENTIONS:
        raise ValueError(f"unknown stopping {stopping!r}, expected one of {STOPPING_CONVENTIONS}")
    obs = build_observations(
        blood_frame_indices=blood_frame_indices,
        noise_level=noise_level,
        seed_idx=seed_idx,
        root_seed=root_seed,
    )
    x0 = make_initial_guess(delta_x, seed_idx, root_seed=root_seed)

    n_obs = len(obs["y"])
    delta_y_stop = obs["delta_y"]
    if stopping == "morozov":
        delta_y_stop = delta_y_stop * np.sqrt(n_obs)

    result = run_irgnm(
        x0,
        obs["y"],
        T_FRAMES,
        obs["s_blood"],
        obs["C_WB_data"],
        schedule=DEFAULT_SCHEDULE,
        tau=DEFAULT_TAU,
        delta_y=delta_y_stop,
        max_iter=max_iter,
        solver=solver,
        x_true=X_TRUE,
        active_mask=TISSUE_ONLY_MASK,
        include_blood=obs["include_blood"],
    )

    y = obs["y"]
    y_norm = float(np.sqrt(y @ y))
    final_resid = float(result["residual_norm"][-1])
    rel_residual = final_resid / y_norm if y_norm > 0 else float("nan")
    # ||noise|| / ||y||: the residual a good fit should settle at.
    noise_floor_rel = (
        obs["delta_y"] * float(np.sqrt(n_obs)) / y_norm if y_norm > 0 else 0.0
    )

    x_f = result["x_final"]
    zK1 = zeta_from_K1(x_f)
    zlam = zeta_from_lambda(x_f)
    sK1 = spread(zK1)
    slam = spread(zlam)

    mean_K1, mean_lam = sK1["mean"], slam["mean"]
    if np.isfinite(mean_K1) and np.isfinite(mean_lam) and mean_K1 != 0.0:
        cross_check = float(abs(mean_K1 - mean_lam) / abs(mean_K1))
    else:
        cross_check = float("nan")

    return {
        "delta_x": delta_x,
        "seed_idx": seed_idx,
        "seed": int(initial_guess_seed(delta_x, seed_idx, root_seed=root_seed)),
        "blood_frame_indices": list(blood_frame_indices),
        "noise_level": noise_level,
        "stopping": stopping,
        "diverged": bool(result["diverged"]),
        "n_iterations": int(result["n_iterations"]),
        "converged_at": result["converged_at"],
        "trivial_stop": bool(result["converged_at"] == 0),
        "delta_y": obs["delta_y"],
        "final_residual_norm": final_resid,
        "rel_residual": rel_residual,
        "noise_floor_rel": noise_floor_rel,
        "fit_accepted": fit_accepted(
            diverged=bool(result["diverged"]),
            rel_residual=rel_residual,
            converged_at=result["converged_at"],
            noise_floor_rel=noise_floor_rel,
            stopping=stopping,
        ),
        "x_final": x_f.tolist(),
        "zeta_from_K1": zK1.tolist(),
        "zeta_from_lambda": zlam.tolist(),
        "spread_K1": sK1,
        "spread_lambda": slam,
        "zeta_K1_vs_lambda_rel_diff": cross_check,
        "zeta_mean": mean_K1,
        "abs_zeta_minus_1": float(abs(mean_K1 - 1.0)) if np.isfinite(mean_K1) else float("nan"),
        **metabolic_relative_errors(x_f),
        "failure_reason": result.get("failure_reason", None),
    }


def summarise(runs: list[dict]) -> dict:
    """Aggregate a list of `run_identifiability_case` results, excluding
    diverged runs (which are counted, never silently dropped — AGENTS.md
    rule 3)."""
    ok = [r for r in runs if r["fit_accepted"]]
    n_div = sum(1 for r in runs if r["diverged"])
    n_trivial = sum(1 for r in runs if r.get("trivial_stop"))
    n_stalled = sum(
        1 for r in runs
        if not r["diverged"] and not r.get("trivial_stop") and not r["fit_accepted"]
    )
    if not ok:
        return {"n_runs": len(runs), "n_diverged": n_div, "n_trivial": n_trivial,
                "n_stalled": n_stalled, "n_accepted": 0}

    def col(key):
        return np.asarray([r[key] for r in ok], dtype=np.float64)

    spreads = np.asarray(
        [r["spread_K1"]["max_over_min_minus_1"] for r in ok], dtype=np.float64
    )
    zetas = col("zeta_mean")
    return {
        "n_runs": len(runs),
        "n_diverged": n_div,
        "n_trivial": n_trivial,
        "n_stalled": n_stalled,
        "n_accepted": len(ok),
        "spread_K1_max": float(np.max(spreads)),
        "spread_K1_median": float(np.median(spreads)),
        "zeta_min": float(np.min(zetas)),
        "zeta_max": float(np.max(zetas)),
        "zeta_mean": float(np.mean(zetas)),
        "abs_zeta_minus_1_max": float(np.max(col("abs_zeta_minus_1"))),
        "abs_zeta_minus_1_median": float(np.median(col("abs_zeta_minus_1"))),
        "zeta_K1_vs_lambda_rel_diff_max": float(np.max(col("zeta_K1_vs_lambda_rel_diff"))),
        "K1_rel_error_max": float(np.max(col("K1_rel_error_max"))),
        "k2_rel_error_max": float(np.max(col("k2_rel_error_max"))),
        "k3_rel_error_max": float(np.max(col("k3_rel_error_max"))),
    }
