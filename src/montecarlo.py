"""Monte Carlo harness for M4.3 — one function per cell of the experiment grid.

A "cell" is a (setup, noise_level, delta_x, seed_index) tuple. This module
provides a single `run_one_cell` function that is self-contained, reproducible,
and safe to call in a sweep.  All randomness is seeded via `src.rng.derive_seed`;
every stochastic run can be reproduced in isolation by passing the same
`root_seed` and `seed_idx`.

Setup definitions
-----------------
  A — reduced:      f known (m parameters frozen), C_WB noiseless.
                    active_mask omits M_SLICE; include_blood=False.
  B — full, clean:  all 23 params free, C_WB noiseless.
                    active_mask=None; include_blood=True.
  C — full, noisy:  all 23 params free, C_WB has Gaussian noise.
                    active_mask=None; include_blood=True.

Noise levels (from noise_calibration.json, reproduced here as constants so
the harness is self-contained — see DECISIONS.md D-M4-1):
  noiseless      — no noise added to TAC or C_WB
  high_count     — Poisson alpha=88198.85, Gaussian sigma_rel=0.002969
  normal_count   — Poisson alpha=6255.53,  Gaussian sigma_rel=0.011139
  low_count      — Poisson alpha=160.55,   Gaussian sigma_rel=0.070945

For Setups A and B, TAC noise is Poisson-derived (the more physically motivated
choice for PET; see remaining_task.md M4.1 open decision).  For Setup C,
TAC noise is also Poisson-derived, and the C_WB blood-measurement noise is
Gaussian (blood draws have a different noise character than scanned frames).
This choice is recorded here per AGENTS.md rule: "every modelling choice not
dictated by the paper gets recorded with a one-line justification."
"""
from __future__ import annotations

import math

import numpy as np

from src.config import (
    ARTERIAL_LAMBDA,
    ARTERIAL_MU,
    F_A,
    F_XI1,
    F_XI2,
    M_SLICE,
    METABOLIC_START,
    N_FRAMES,
    N_PARAMS,
    N_REGIONS,
    PROJECTION_EPS,
    REGION_KINETICS,
    REGION_NAMES,
    blood_sample_times_minutes,
    frame_midtimes_minutes,
    ground_truth_vector,
)
from src.forward_model import (
    C_WB_from_C_P,
    arterial_input,
    closed_form_C_T,
    parent_plasma_fraction,
)
from src.irgnm import DEFAULT_SCHEDULE, DEFAULT_TAU, RegularizationSchedule, project, run_irgnm
from src.jacobian import forward_operator, unpack
from src.noise import (
    add_gaussian_noise,
    add_poisson_noise,
    compute_delta_y,
)
from src.rng import LCG, derive_seed, standard_normal

# ---------------------------------------------------------------------------
# Calibrated noise parameters (from results/m4/noise_calibration.json).
# Hardcoded here so this module is self-contained; values chosen by bisection
# calibration in experiments/m4_noise_calibration.py (see DECISIONS.md D-M4-1).
# ---------------------------------------------------------------------------
POISSON_ALPHA: dict[str, float] = {
    "high_count":   88198.85,
    "normal_count":  6255.53,
    "low_count":      160.55,
}
GAUSSIAN_SIGMA_REL: dict[str, float] = {
    "high_count":   0.002969,
    "normal_count": 0.011139,
    "low_count":    0.070945,
}

# C_WB noise: Gaussian sigma_rel for blood measurements in Setup C.
# Same sigma_rel scale used for TAC Gaussian, applied to C_WB.
# Justification: blood-draw measurements in PET are typically assumed Gaussian
# (counting statistics in a very short draw sample); Poisson is not appropriate
# for the low-volume blood sample model we use here.
CWB_GAUSSIAN_SIGMA_REL: dict[str, float] = GAUSSIAN_SIGMA_REL

SETUP_NAMES = ("A", "B", "C")
NOISE_LEVELS = ("noiseless", "high_count", "normal_count", "low_count")
DELTA_X_VALUES = (0.1, 0.2, 0.3, 0.4)
N_SEEDS = 20
MAX_ITER = 300
ROOT_SEED = 20240401


# ---------------------------------------------------------------------------
# Ground-truth data (module-level constants — avoid recomputing per call)
# ---------------------------------------------------------------------------
X_TRUE = ground_truth_vector()
T_FRAMES = frame_midtimes_minutes()
S_BLOOD = blood_sample_times_minutes()

# Clean TACs, shape (N_REGIONS, N_FRAMES)
_lam, _mu, _m, _K1, _k2, _k3 = unpack(X_TRUE)
_A, _xi1, _xi2 = _m

C_T_CLEAN: np.ndarray = np.array([
    closed_form_C_T(T_FRAMES, _K1[i], _k2[i], _k3[i], _lam, _mu)
    for i in range(N_REGIONS)
])  # shape (N_REGIONS, N_FRAMES)

C_WB_CLEAN: np.ndarray = C_WB_from_C_P(
    arterial_input(S_BLOOD, _lam, _mu),
    parent_plasma_fraction(S_BLOOD, _A, _xi1, _xi2),
)  # shape (Q_BLOOD_SAMPLES,)


# ---------------------------------------------------------------------------
# Mask for Setup A: freeze the 3 plasma-fraction (m) parameters
# ---------------------------------------------------------------------------
SETUP_A_MASK: np.ndarray = np.ones(N_PARAMS, dtype=bool)
SETUP_A_MASK[M_SLICE] = False  # freeze m_1, m_2, m_3


# ---------------------------------------------------------------------------
# Paper-settings options for run_one_cell (all off by default; DECISIONS.md
# D-M5-7). Values from the paper's Section 6, p. 22, converted from base 2
# (c * 2^(-i/d)) to the a * exp(-b*i) form: b = ln(2)/d.
# ---------------------------------------------------------------------------
_LN2 = math.log(2.0)
PAPER_SCHEDULES: dict[str, RegularizationSchedule] = {
    # Reduced setup: f is known, so the gamma block is frozen and unused.
    "A": RegularizationSchedule(a_alpha=10.0, b_alpha=_LN2 / 5, a_beta=600.0, b_beta=_LN2 / 7,
                                a_gamma=DEFAULT_SCHEDULE.a_gamma, b_gamma=DEFAULT_SCHEDULE.b_gamma),
    "B": DEFAULT_SCHEDULE,
    "C": RegularizationSchedule(a_alpha=3000.0, b_alpha=_LN2 / 8, a_beta=100.0, b_beta=_LN2 / 8,
                                a_gamma=400.0, b_gamma=_LN2 / 8),
}
PAPER_TAU: dict[str, float] = {"A": 9.2, "B": DEFAULT_TAU, "C": 17.6}
PAPER_MAX_ITER_NOISY = 200  # paper: 300 noiseless, 200 for the count settings

# The paper's C_WB is measured at every frame time t_1..t_T (Algorithm 1 input,
# and p. 19: "corrected for each time t_1, ..., t_T"); ours uses 4 (D-M3-1).
S_BLOOD_ALL_FRAMES: np.ndarray = T_FRAMES.copy()
C_WB_CLEAN_ALL_FRAMES: np.ndarray = C_WB_from_C_P(
    arterial_input(S_BLOOD_ALL_FRAMES, _lam, _mu),
    parent_plasma_fraction(S_BLOOD_ALL_FRAMES, _A, _xi1, _xi2),
)


def _build_noisy_observations(
    noise_level: str,
    seed_idx: int,
    setup: str,
    delta_x: float,
    *,
    s_blood: np.ndarray = S_BLOOD,
    c_wb_clean: np.ndarray = C_WB_CLEAN,
    setup_a_blood: bool = False,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Build the noisy y_delta observation vector and noisy C_WB for one cell.

    Returns (y_delta, C_WB_data, delta_y_tac) where:
      - y_delta   : observation vector passed to run_irgnm
      - C_WB_data : fixed blood data (noisy for Setup C, clean otherwise)
      - delta_y_tac : measured TAC-level discrepancy level (used as delta_y
                      in the discrepancy stopping criterion)

    `setup_a_blood=True` gives Setup A the clean blood block, like the paper's
    reduced setup; the default keeps M4.3's tissue-only Setup A.
    """
    a_tissue_only = setup == "A" and not setup_a_blood
    if noise_level == "noiseless":
        y = forward_operator(X_TRUE, T_FRAMES, s_blood, c_wb_clean,
                             include_blood=not a_tissue_only)
        return y, c_wb_clean.copy(), 0.0

    # --- Noisy case ---
    # TAC noise: Poisson-derived for all setups
    alpha = POISSON_ALPHA[noise_level]
    tac_seed = derive_seed(ROOT_SEED, f"setup={setup}", f"dx={delta_x}",
                           f"noise={noise_level}", f"tac", f"k={seed_idx}")
    tac_rng = LCG(tac_seed)
    C_T_noisy = add_poisson_noise(tac_rng, C_T_CLEAN, alpha=alpha)

    # Measured TAC discrepancy: the delta_y used in the stopping rule.
    # Flatten row-major to match forward_operator's F^1 layout.
    delta_y_tac = compute_delta_y(C_T_noisy, C_T_CLEAN)

    # Build F^1 from noisy TACs (just the flattened noisy values)
    F1_noisy = C_T_noisy.ravel()  # (N_REGIONS * N_FRAMES,) row-major

    if setup in ("A", "B"):
        # C_WB is noiseless for Setups A and B
        C_WB_data = c_wb_clean.copy()
        if a_tissue_only:
            y = F1_noisy
        else:
            # Append F^2 = C_WB_data * f(s) - C_P(s) at ground truth
            F2 = forward_operator(X_TRUE, T_FRAMES, s_blood, c_wb_clean,
                                  include_blood=True)[N_REGIONS * N_FRAMES:]
            y = np.concatenate([F1_noisy, F2])

    else:  # Setup C: noisy C_WB
        sigma_rel = CWB_GAUSSIAN_SIGMA_REL[noise_level]
        cwb_seed = derive_seed(ROOT_SEED, f"setup={setup}", f"dx={delta_x}",
                               f"noise={noise_level}", f"cwb", f"k={seed_idx}")
        cwb_rng = LCG(cwb_seed)
        C_WB_noisy = add_gaussian_noise(cwb_rng, c_wb_clean, sigma_rel=sigma_rel)
        C_WB_data = C_WB_noisy

        # F^2 uses the NOISY C_WB_data in the data constraint
        # F^2 = C_WB_data * f_m(s) - C_P(lam, mu)(s)
        # At ground truth f_m, this is just the noisy version of the constraint
        f_s = parent_plasma_fraction(s_blood, _A, _xi1, _xi2)
        C_P_s = arterial_input(s_blood, _lam, _mu)
        F2_noisy = C_WB_noisy * f_s - C_P_s
        y = np.concatenate([F1_noisy, F2_noisy])

    return y, C_WB_data, delta_y_tac


def run_one_cell(
    setup: str,
    noise_level: str,
    delta_x: float,
    seed_idx: int,
    *,
    paper_hyperparams: bool = False,
    fixed_stopping: bool = False,
    setup_a_blood: bool = False,
    blood_at_all_frames: bool = False,
    paper_max_iter: bool = False,
) -> dict:
    """Run one Monte Carlo cell and return a result dict.

    Parameters
    ----------
    setup       : 'A', 'B', or 'C'
    noise_level : 'noiseless', 'high_count', 'normal_count', or 'low_count'
    delta_x     : initial perturbation level in {0.1, 0.2, 0.3, 0.4}
    seed_idx    : integer in [0, N_SEEDS), used with derive_seed for reproducibility

    Keyword options (all False reproduces M4.3 exactly; DECISIONS.md D-M5-7):
    paper_hyperparams   : the paper's per-setup schedule and tau (PAPER_SCHEDULES, PAPER_TAU)
    fixed_stopping      : pass the noise NORM delta_y*sqrt(n_tac) to the stopping
                          rule instead of the RMS (the D-M4-9 correction)
    setup_a_blood       : give Setup A the clean blood block (paper's reduced setup)
    blood_at_all_frames : blood readings at all 25 frames instead of 4
    paper_max_iter      : 200 iterations for noisy data, as in the paper

    Returns a dict with keys:
      seed, setup, noise_level, delta_x, seed_idx,
      diverged, no_improvement, n_iterations, converged_at,
      x_final, rel_error_total (trajectory),
      rel_error_K, rel_error_lambda, rel_error_mu, rel_error_m,
      delta_y_tac, K1_recovered, k2_recovered, k3_recovered

    `diverged` = an iterate became non-finite (ours); `no_improvement` = the
    final error is not below the initial error (the paper's Table 1 criterion).
    """
    assert setup in SETUP_NAMES, f"Unknown setup {setup!r}"
    assert noise_level in NOISE_LEVELS, f"Unknown noise_level {noise_level!r}"
    assert delta_x in DELTA_X_VALUES, f"Unknown delta_x {delta_x}"
    s_blood = S_BLOOD_ALL_FRAMES if blood_at_all_frames else S_BLOOD
    c_wb_clean = C_WB_CLEAN_ALL_FRAMES if blood_at_all_frames else C_WB_CLEAN

    # --- Initial guess ---
    seed = derive_seed(ROOT_SEED, f"setup={setup}", f"dx={delta_x}",
                       f"noise={noise_level}", f"x0", f"k={seed_idx}")
    rng = LCG(seed)
    sigma = np.where(rng.uniform_array(N_PARAMS) < 0.5, -1.0, 1.0)
    gamma = delta_x + np.sqrt(delta_x / 4.0) * standard_normal(rng, N_PARAMS)
    x0 = project(X_TRUE * (1.0 + sigma * gamma))

    # For Setup A: freeze m parameters at ground truth in x0
    if setup == "A":
        x0[M_SLICE] = X_TRUE[M_SLICE]

    # --- Observations ---
    y_delta, C_WB_data, delta_y_tac = _build_noisy_observations(
        noise_level, seed_idx, setup, delta_x,
        s_blood=s_blood, c_wb_clean=c_wb_clean, setup_a_blood=setup_a_blood,
    )

    # --- Solver configuration ---
    active_mask = SETUP_A_MASK if setup == "A" else None
    include_blood = setup != "A" or setup_a_blood
    noisy = noise_level != "noiseless"
    delta_y_stop = delta_y_tac if noisy else 0.0
    if fixed_stopping:
        delta_y_stop *= math.sqrt(N_REGIONS * N_FRAMES)
    schedule = PAPER_SCHEDULES[setup] if paper_hyperparams else DEFAULT_SCHEDULE
    tau = PAPER_TAU[setup] if paper_hyperparams else DEFAULT_TAU
    max_iter = PAPER_MAX_ITER_NOISY if (paper_max_iter and noisy) else MAX_ITER

    # --- Run IRGNM ---
    result = run_irgnm(
        x0, y_delta, T_FRAMES, s_blood, C_WB_data,
        schedule=schedule,
        tau=tau,
        delta_y=delta_y_stop,
        max_iter=max_iter,
        solver="qr",
        x_true=X_TRUE,
        active_mask=active_mask,
        include_blood=include_blood,
    )
    traj = result["rel_error_total"]
    no_improvement = bool(
        result["diverged"] or not traj or not np.isfinite(traj[-1]) or traj[-1] >= traj[0]
    )

    # --- Extract regional K1, k2, k3 from x_final ---
    x_f = result["x_final"]
    K1_rec = [float(x_f[METABOLIC_START + 3 * i + 0]) for i in range(N_REGIONS)]
    k2_rec = [float(x_f[METABOLIC_START + 3 * i + 1]) for i in range(N_REGIONS)]
    k3_rec = [float(x_f[METABOLIC_START + 3 * i + 2]) for i in range(N_REGIONS)]

    return {
        "seed": int(seed),
        "seed_idx": seed_idx,
        "setup": setup,
        "noise_level": noise_level,
        "delta_x": delta_x,
        "diverged": bool(result["diverged"]),
        "no_improvement": no_improvement,
        "n_iterations": int(result["n_iterations"]),
        "converged_at": result["converged_at"],
        "delta_y_tac": float(delta_y_tac),
        "x_final": result["x_final"].tolist(),
        "rel_error_total": result["rel_error_total"],
        "rel_error_K": result["rel_error_K"],
        "rel_error_lambda": result["rel_error_lambda"],
        "rel_error_mu": result["rel_error_mu"],
        "rel_error_m": result["rel_error_m"],
        "K1_recovered": K1_rec,
        "k2_recovered": k2_rec,
        "k3_recovered": k3_rec,
        "failure_reason": result.get("failure_reason", None),
    }


def cell_key(setup: str, noise_level: str, delta_x: float) -> str:
    return f"setup={setup}_noise={noise_level}_dx={delta_x}"
