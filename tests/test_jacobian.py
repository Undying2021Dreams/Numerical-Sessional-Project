"""M3 acceptance tests for src/jacobian.py (forward operator F, analytic F').

No Track B here: everything is checked against our own finite-difference
approximation of the same forward_operator (an independent numerical
computation of the same quantity, not a library call).
"""
from __future__ import annotations

import numpy as np
import pytest

from src.config import (
    N_PARAMS,
    REGION_NAMES,
    UNKNOWN_LAYOUT,
    blood_sample_times_minutes,
    frame_midtimes_minutes,
    ground_truth_vector,
)
from src.forward_model import C_WB_from_C_P, arterial_input, parent_plasma_fraction
from src.jacobian import (
    PHI2_SERIES_THRESHOLD,
    _phi1_prime,
    _phi2,
    _phi2_closed_form,
    _phi2_series,
    analytic_jacobian,
    forward_operator,
    unpack,
)

T_FRAMES = frame_midtimes_minutes()
S_BLOOD = blood_sample_times_minutes()
X_TRUE = ground_truth_vector()

# Optimal FD step, found by sweeping h in {1e-4,...,1e-8} against the mu
# block (the worst-conditioned block — see DECISIONS.md D-M3-3): error is
# U-shaped (truncation-dominated above ~1e-5, cancellation-dominated below),
# minimised around h ~ 1.5e-5.
FD_H_REL = 1.5e-5

BLOCKS = {
    "lambda": range(0, 4),
    "mu": range(4, 8),
    "m": range(8, 11),
    "K": range(11, 23),
}


def _cwb_data_at(x, s):
    lam, mu, m, K1, k2, k3 = unpack(x)
    A, xi1, xi2 = m
    return C_WB_from_C_P(arterial_input(s, lam, mu), parent_plasma_fraction(s, A, xi1, xi2))


def _fd_jacobian_column(x, j, t, s, C_WB_data, h_rel=FD_H_REL):
    hh = h_rel * max(abs(x[j]), 1.0)
    xp, xm = x.copy(), x.copy()
    xp[j] += hh
    xm[j] -= hh
    Fp = forward_operator(xp, t, s, C_WB_data)
    Fm = forward_operator(xm, t, s, C_WB_data)
    return (Fp - Fm) / (2 * hh)


def _max_rel_error(J_col, fd_col):
    floor = max(1e-10, 1e-6 * np.max(np.abs(J_col)))
    return float(np.max(np.abs(fd_col - J_col) / np.maximum(np.abs(J_col), floor)))


# ---------------------------------------------------------------------------
# phi2 / phi1' correctness
# ---------------------------------------------------------------------------
def _phi2_high_precision_series_reference(x: float, terms: int = 40) -> float:
    from math import factorial

    total = 0.0
    for k in range(terms):
        total += x**k / factorial(k + 2)
    return total


def test_phi2_matches_high_precision_reference_across_all_scales():
    for x in (10.0, 1.0, 0.1, PHI2_SERIES_THRESHOLD * 10, PHI2_SERIES_THRESHOLD / 10, 1e-8, 1e-12, 0.0, -1.0, -10.0):
        ours = float(_phi2(np.array([x]))[0])
        ref = _phi2_high_precision_series_reference(x)
        rel = abs(ours - ref) / max(abs(ref), 1e-300)
        assert rel < 1e-10, f"phi2({x}) = {ours}, reference = {ref}, rel err {rel:.3e}"


def test_phi2_series_and_closed_form_agree_near_threshold():
    """The two branches should agree closely right at the switch point (no
    discontinuity introduced by the threshold)."""
    x = np.array([PHI2_SERIES_THRESHOLD])
    series_val = float(_phi2_series(x)[0])
    closed_val = float(_phi2_closed_form(x)[0])
    assert abs(series_val - closed_val) < 1e-10


def test_phi1_prime_matches_central_finite_difference_of_phi1():
    from src.forward_model import _phi1

    h = 1e-5
    for x in (2.0, 0.5, 0.05, 0.0, -0.5, -3.0):
        fd = (float(_phi1(np.array([x + h]))[0]) - float(_phi1(np.array([x - h]))[0])) / (2 * h)
        analytic = float(_phi1_prime(np.array([x]))[0])
        assert abs(fd - analytic) < 1e-6, f"phi1'({x}): fd={fd}, analytic={analytic}"


# ---------------------------------------------------------------------------
# Analytic Jacobian vs finite differences (PLAN.md M3 acceptance criterion)
# ---------------------------------------------------------------------------
def test_jacobian_matches_finite_differences_at_ground_truth_per_block():
    C_WB_data = _cwb_data_at(X_TRUE, S_BLOOD)
    J = analytic_jacobian(X_TRUE, T_FRAMES, S_BLOOD, C_WB_data)
    assert J.shape == (4 * len(T_FRAMES) + len(S_BLOOD), N_PARAMS)

    block_errors = {}
    for name, idxs in BLOCKS.items():
        worst = 0.0
        for j in idxs:
            fd_col = _fd_jacobian_column(X_TRUE, j, T_FRAMES, S_BLOOD, C_WB_data)
            worst = max(worst, _max_rel_error(J[:, j], fd_col))
        block_errors[name] = worst

    for name, err in block_errors.items():
        assert err < 1e-4, f"block '{name}': max relative error {err:.3e}, expected <1e-4"


def test_jacobian_matches_finite_differences_at_random_feasible_points():
    rng = np.random.default_rng(7)
    for trial in range(3):
        xr = X_TRUE * (1.0 + 0.1 * rng.normal(size=N_PARAMS))
        xr[8] = abs(xr[8])  # A >= 0
        xr[9] = -abs(xr[9])  # xi1 <= 0
        xr[10] = -abs(xr[10])  # xi2 <= 0
        xr[11:] = np.maximum(xr[11:], 1e-3)  # K1,k2,k3 >= eps

        C_WB_data = _cwb_data_at(xr, S_BLOOD)
        J = analytic_jacobian(xr, T_FRAMES, S_BLOOD, C_WB_data)

        worst = 0.0
        for j in range(N_PARAMS):
            fd_col = _fd_jacobian_column(xr, j, T_FRAMES, S_BLOOD, C_WB_data)
            worst = max(worst, _max_rel_error(J[:, j], fd_col))
        assert worst < 1e-4, f"trial {trial}: max relative error {worst:.3e}"


def test_forward_operator_F1_matches_closed_form_C_T_directly():
    from src.forward_model import closed_form_C_T

    C_WB_data = _cwb_data_at(X_TRUE, S_BLOOD)
    F = forward_operator(X_TRUE, T_FRAMES, S_BLOOD, C_WB_data)
    lam, mu, m, K1, k2, k3 = unpack(X_TRUE)
    T = len(T_FRAMES)
    for i, name in enumerate(REGION_NAMES):
        expected = closed_form_C_T(T_FRAMES, K1[i], k2[i], k3[i], lam, mu)
        got = F[i * T : (i + 1) * T]
        assert np.allclose(got, expected)


def test_forward_operator_F2_is_zero_at_ground_truth():
    """F^2(x_true) should be exactly (to round-off) zero: C_WB_data was built
    FROM x_true's own C_P and f, so C_WB_data*f_true - C_P_true = 0."""
    C_WB_data = _cwb_data_at(X_TRUE, S_BLOOD)
    F = forward_operator(X_TRUE, T_FRAMES, S_BLOOD, C_WB_data)
    F2 = F[4 * len(T_FRAMES) :]
    assert np.max(np.abs(F2)) < 1e-10, f"F^2(x_true) should vanish, got max abs {np.max(np.abs(F2)):.3e}"
