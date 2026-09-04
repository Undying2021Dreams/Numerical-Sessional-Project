"""M3 acceptance tests for src/irgnm.py (IRGNM solver).

Lightweight by design: the full 4x20-seed Monte Carlo recovery study lives
in experiments/m3_irgnm_recovery.py (too slow for the default test suite —
~30s). These tests check the solver's building blocks and run a couple of
single-seed recoveries to a reduced iteration budget, fast enough for
routine `pytest` runs.
"""
from __future__ import annotations

import numpy as np

from src.config import (
    LAMBDA_SLICE,
    METABOLIC_START,
    MU_SLICE,
    M_SLICE,
    N_PARAMS,
    PROJECTION_EPS,
    blood_sample_times_minutes,
    frame_midtimes_minutes,
    ground_truth_vector,
)
from src.forward_model import C_WB_from_C_P, arterial_input, parent_plasma_fraction
from src.irgnm import DEFAULT_SCHEDULE, DEFAULT_TAU, RegularizationSchedule, project, run_irgnm
from src.jacobian import forward_operator, unpack
from src.rng import LCG, standard_normal

X_TRUE = ground_truth_vector()
T_FRAMES = frame_midtimes_minutes()
S_BLOOD = blood_sample_times_minutes()


def _cwb_data(x):
    lam, mu, m, K1, k2, k3 = unpack(x)
    A, xi1, xi2 = m
    return C_WB_from_C_P(arterial_input(S_BLOOD, lam, mu), parent_plasma_fraction(S_BLOOD, A, xi1, xi2))


def test_project_enforces_domain_constraints():
    x = X_TRUE.copy()
    x[M_SLICE.start] = -5.0  # A should clip to 0
    x[M_SLICE.start + 1] = 3.0  # xi1 should clip to 0
    x[M_SLICE.start + 2] = 7.0  # xi2 should clip to 0
    x[METABOLIC_START] = -1.0  # K1^1 should clip to PROJECTION_EPS
    xp = project(x)
    assert xp[M_SLICE.start] == 0.0
    assert xp[M_SLICE.start + 1] == 0.0
    assert xp[M_SLICE.start + 2] == 0.0
    assert xp[METABOLIC_START] == PROJECTION_EPS
    # unconstrained blocks (lambda, mu) untouched
    assert np.array_equal(xp[LAMBDA_SLICE], x[LAMBDA_SLICE])
    assert np.array_equal(xp[MU_SLICE], x[MU_SLICE])


def test_project_is_identity_on_ground_truth():
    assert np.array_equal(project(X_TRUE), X_TRUE)


def test_regularization_schedule_decays_and_blocks_differ():
    sched = DEFAULT_SCHEDULE
    d0 = sched.diag(0)
    d100 = sched.diag(100)
    assert np.all(d100 < d0), "regularisation should decay with iteration"
    # three distinct block values at any iteration (unless coincidentally equal)
    alpha0, beta0, gamma0 = d0[METABOLIC_START], d0[0], d0[8]
    assert len({round(alpha0, 6), round(beta0, 6), round(gamma0, 6)}) == 3


def test_irgnm_recovers_ground_truth_from_small_perturbation_qr():
    C_WB_data = _cwb_data(X_TRUE)
    y_true = forward_operator(X_TRUE, T_FRAMES, S_BLOOD, C_WB_data)

    rng = LCG(seed=42)
    delta_x = 0.1
    sigma = np.where(rng.uniform_array(N_PARAMS) < 0.5, -1.0, 1.0)
    gamma = delta_x + np.sqrt(delta_x / 4.0) * standard_normal(rng, N_PARAMS)
    x0 = project(X_TRUE * (1 + sigma * gamma))

    result = run_irgnm(
        x0, y_true, T_FRAMES, S_BLOOD, C_WB_data,
        schedule=DEFAULT_SCHEDULE, tau=DEFAULT_TAU, delta_y=0.0,
        max_iter=300, solver="qr", x_true=X_TRUE,
    )
    assert not result["diverged"]
    assert result["rel_error_total"][-1] < 1e-5, f"final rel error {result['rel_error_total'][-1]:.3e}"
    # error should have decreased monotonically-ish (at least by the end vs the start)
    assert result["rel_error_total"][-1] < result["rel_error_total"][0]


def test_irgnm_lu_and_qr_paths_agree_on_a_well_behaved_run():
    C_WB_data = _cwb_data(X_TRUE)
    y_true = forward_operator(X_TRUE, T_FRAMES, S_BLOOD, C_WB_data)

    rng = LCG(seed=42)
    delta_x = 0.1
    sigma = np.where(rng.uniform_array(N_PARAMS) < 0.5, -1.0, 1.0)
    gamma = delta_x + np.sqrt(delta_x / 4.0) * standard_normal(rng, N_PARAMS)
    x0 = project(X_TRUE * (1 + sigma * gamma))

    result_qr = run_irgnm(x0, y_true, T_FRAMES, S_BLOOD, C_WB_data, max_iter=100, solver="qr", x_true=X_TRUE)
    result_lu = run_irgnm(x0, y_true, T_FRAMES, S_BLOOD, C_WB_data, max_iter=100, solver="lu", x_true=X_TRUE)

    assert not result_qr["diverged"] and not result_lu["diverged"]
    rel_diff = np.linalg.norm(result_qr["x_final"] - result_lu["x_final"]) / np.linalg.norm(result_lu["x_final"])
    assert rel_diff < 1e-4, f"LU and QR paths disagree: {rel_diff:.3e}"


def test_perturbation_sampler_reproduces_eq24_expected_squared_deviation():
    """Paper eq. (23)/(24): E[(sigma*gamma)^2] = delta_x^2 + delta_x/4, where
    gamma ~ N(delta_x, delta_x/4) — the second argument is the VARIANCE."""
    for delta_x in (0.1, 0.2, 0.3, 0.4):
        rng = LCG(seed=int(delta_x * 1000) + 7)
        n = 100_000
        sigma = np.where(rng.uniform_array(n) < 0.5, -1.0, 1.0)
        gamma = delta_x + np.sqrt(delta_x / 4.0) * standard_normal(rng, n)
        measured = float(np.mean((sigma * gamma) ** 2))
        predicted = delta_x**2 + delta_x / 4.0
        assert abs(measured - predicted) / predicted < 0.02, f"delta_x={delta_x}: measured {measured:.5f} vs predicted {predicted:.5f}"
