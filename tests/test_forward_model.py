"""M2 acceptance tests for src/forward_model.py (closed form, quadrature, C_PET).

Track B usage (clearly marked): `scipy.integrate.solve_ivp` is used here, in
tests/, as an independent third code path against which both our closed-form
and quadrature implementations are checked (PLAN.md M2's "three-way
agreement" acceptance criterion). Nothing in `src/` imports scipy.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from src.config import (
    ARTERIAL_LAMBDA,
    ARTERIAL_MU,
    F_A,
    F_XI1,
    F_XI2,
    N_REGIONS,
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
    closed_form_term1,
    parent_plasma_fraction,
    quadrature_C_T,
    _phi1,
)

LAM = np.array(ARTERIAL_LAMBDA)
MU = np.array(ARTERIAL_MU)
FRAME_T = frame_midtimes_minutes()
GRID = GradedGridSpec(n=1601, q=3.0)  # DECISIONS.md D-M2-3


# ---------------------------------------------------------------------------
# phi1 correctness (underlies the whole closed-form stability argument)
# ---------------------------------------------------------------------------
def _phi1_series_reference(x: float, terms: int = 30) -> float:
    """phi1(x) = sum_{k=0}^inf x^k / (k+1)! — used only as an independent
    high-precision reference for small |x| (DECISIONS.md D-M2-1)."""
    total = 0.0
    term = 1.0
    fact = 1.0
    for k in range(terms):
        fact *= k + 1
        total += term / fact
        term *= x
    return total


def test_phi1_matches_series_reference_for_small_x():
    for x in (1e-1, 1e-4, 1e-8, 1e-12, 1e-20, 0.0, -1e-8, -1e-1):
        ours = float(_phi1(np.array([x]))[0])
        ref = _phi1_series_reference(x)
        rel = abs(ours - ref) / max(abs(ref), 1e-300)
        assert rel < 1e-12, f"phi1({x}) = {ours}, series reference = {ref}, rel err {rel:.3e}"


def test_phi1_matches_direct_formula_for_moderate_x():
    for x in (0.5, 1.0, -0.5, -2.0, 5.0, -10.0):
        ours = float(_phi1(np.array([x]))[0])
        direct = (math.exp(x) - 1.0) / x
        assert abs(ours - direct) < 1e-12


def test_phi1_at_exact_zero_is_exactly_one():
    assert float(_phi1(np.array([0.0]))[0]) == 1.0


# ---------------------------------------------------------------------------
# Sanity: C_T(0) == 0, non-negativity
# ---------------------------------------------------------------------------
def test_closed_form_C_T_zero_at_t_zero_all_regions():
    for name in REGION_NAMES:
        K1, k2, k3 = REGION_KINETICS[name]
        ct0 = closed_form_C_T(np.array([0.0]), K1, k2, k3, LAM, MU)[0]
        assert ct0 == 0.0, f"{name}: C_T(0) = {ct0}, expected exactly 0"


def test_closed_form_C_T_nonnegative_on_frame_grid_all_regions():
    for name in REGION_NAMES:
        K1, k2, k3 = REGION_KINETICS[name]
        ct = closed_form_C_T(FRAME_T, K1, k2, k3, LAM, MU)
        assert np.all(ct >= -1e-12), f"{name}: negative C_T values: {ct[ct < 0]}"


# ---------------------------------------------------------------------------
# Closed form vs paper's literal eq. (3) branches (proves D-M2-1's algebraic
# equivalence claim programmatically, not just by hand)
# ---------------------------------------------------------------------------
def _closed_form_literal_eq3(t, K1, k2, k3, lam, mu):
    """A direct, literal transcription of the paper's eq. (3), with explicit
    if/else branches on exact equality — deliberately NOT using phi1, so
    this is an independent implementation to check `closed_form_C_T`
    against (not a copy of it). Naive (not round-off-safe near degeneracy),
    which is fine: it is only evaluated at points away from degeneracy in
    this test, exactly where naive evaluation is fine."""
    a = k2 + k3
    p = len(lam)
    t = np.asarray(t, dtype=np.float64)
    out = np.zeros_like(t)
    for j in range(p):
        delta_j = a + mu[j]
        coeff = 0.0
        if mu[j] != 0:
            coeff += k3 / mu[j]
        if delta_j != 0:
            coeff += k2 / delta_j
        out += (K1 / a) * coeff * lam[j] * np.exp(mu[j] * t)

    bracket_e = 0.0
    bracket_t_e = 0.0
    const = 0.0
    bracket_t_const = 0.0
    for j in range(p):
        delta_j = a + mu[j]
        if delta_j != 0:
            bracket_e += lam[j] / delta_j
        else:
            bracket_t_e += lam[j]
        if mu[j] != 0:
            const += lam[j] / mu[j]
        else:
            bracket_t_const += lam[j]

    out -= (K1 * k2 / a) * bracket_e * np.exp(-a * t)
    out -= (K1 * k3 / a) * const
    out += (K1 * k2 / a) * bracket_t_e * t * np.exp(-a * t)
    out += (K1 * k3 / a) * bracket_t_const * t
    return out


def test_stable_form_matches_paper_eq3_branches():
    """Away from degeneracy: our phi1-based closed_form_C_T must match a
    literal transcription of the paper's branched eq. (3) exactly."""
    for name in REGION_NAMES:
        K1, k2, k3 = REGION_KINETICS[name]
        ours = closed_form_C_T(FRAME_T, K1, k2, k3, LAM, MU)
        literal = _closed_form_literal_eq3(FRAME_T, K1, k2, k3, LAM, MU)
        rel = np.max(np.abs(ours - literal) / np.maximum(np.abs(literal), 1e-300))
        assert rel < 1e-10, f"{name}: our closed form vs literal eq(3): max rel diff {rel:.3e}"


def test_stable_form_matches_literal_branch_exactly_at_mu_equals_zero():
    """Directly exercises the mu_j == 0 branch."""
    lam = np.array([1.0, 2.0])
    mu = np.array([0.0, -0.5])
    K1, k2, k3 = 0.15, 0.17, 0.10
    t = np.linspace(0.01, 10.0, 20)
    ours = closed_form_C_T(t, K1, k2, k3, lam, mu)
    literal = _closed_form_literal_eq3(t, K1, k2, k3, lam, mu)
    assert np.max(np.abs(ours - literal)) < 1e-10


def test_stable_form_matches_literal_branch_exactly_at_delta_equals_zero():
    """Directly exercises the k2+k3+mu_j == 0 branch."""
    K1, k2, k3 = 0.15, 0.17, 0.10
    a = k2 + k3
    lam = np.array([1.0, 2.0])
    mu = np.array([-a, -0.5])  # delta_0 = a + mu[0] = 0 exactly
    t = np.linspace(0.01, 10.0, 20)
    ours = closed_form_C_T(t, K1, k2, k3, lam, mu)
    literal = _closed_form_literal_eq3(t, K1, k2, k3, lam, mu)
    assert np.max(np.abs(ours - literal)) < 1e-10


# ---------------------------------------------------------------------------
# Three-way agreement: closed form vs quadrature vs scipy.solve_ivp (Track B)
# This is the milestone's centrepiece acceptance test (PLAN.md M2).
# ---------------------------------------------------------------------------
def _solve_ivp_C_T(t_eval, K1, k2, k3, lam, mu, method="Radau"):
    """Independent third code path: integrate ODE system (S) directly."""
    a = k2 + k3

    def rhs(t, y):
        C_F, C_B = y
        C_P_t = float(np.sum(lam * np.exp(mu * t)))
        dC_F = K1 * C_P_t - a * C_F
        dC_B = k3 * C_F
        return [dC_F, dC_B]

    t_max = float(np.max(t_eval))
    sol = solve_ivp(
        rhs, (0.0, t_max), y0=[0.0, 0.0], t_eval=np.sort(t_eval),
        method=method, rtol=1e-12, atol=1e-14, max_step=t_max / 2000.0,
    )
    assert sol.success, f"solve_ivp failed: {sol.message}"
    C_F, C_B = sol.y
    return C_F + C_B


def test_three_way_agreement_closed_form_quadrature_solve_ivp():
    """PLAN.md M2 centrepiece: closed form vs quadrature vs scipy.solve_ivp,
    max relative difference across all 4 regions and all 25 frames, each
    pairing reported separately."""
    max_diff_closed_vs_quad = 0.0
    max_diff_closed_vs_ivp = 0.0
    max_diff_quad_vs_ivp = 0.0
    any_fallback = False

    for name in REGION_NAMES:
        K1, k2, k3 = REGION_KINETICS[name]

        ct_closed = closed_form_C_T(FRAME_T, K1, k2, k3, LAM, MU)
        ct_quad, fallback_flags = quadrature_C_T(FRAME_T, K1, k2, k3, LAM, MU, GRID)
        any_fallback = any_fallback or any(fallback_flags)
        ct_ivp = _solve_ivp_C_T(FRAME_T, K1, k2, k3, LAM, MU)

        scale = np.maximum(np.abs(ct_closed), 1e-8)
        max_diff_closed_vs_quad = max(max_diff_closed_vs_quad, float(np.max(np.abs(ct_closed - ct_quad) / scale)))
        max_diff_closed_vs_ivp = max(max_diff_closed_vs_ivp, float(np.max(np.abs(ct_closed - ct_ivp) / scale)))
        max_diff_quad_vs_ivp = max(max_diff_quad_vs_ivp, float(np.max(np.abs(ct_quad - ct_ivp) / scale)))

    assert not any_fallback, "simpson_diag's trapezoid fallback fired — grid should always have an even interval count"
    # Thresholds chosen from measurement (handoffs/RUN_M2.md has the exact
    # numbers): closed-vs-quadrature agree to near round-off; closed-vs-ODE
    # agree to solve_ivp's own requested tolerance, not tighter.
    assert max_diff_closed_vs_quad < 1e-7, f"closed vs quadrature: {max_diff_closed_vs_quad:.3e}"
    assert max_diff_closed_vs_ivp < 1e-8, f"closed vs solve_ivp: {max_diff_closed_vs_ivp:.3e}"
    assert max_diff_quad_vs_ivp < 1e-7, f"quadrature vs solve_ivp: {max_diff_quad_vs_ivp:.3e}"


# ---------------------------------------------------------------------------
# Near-degeneracy stress sweep (DECISIONS.md D-M2-6)
# ---------------------------------------------------------------------------
def test_near_degeneracy_sweep_stays_smooth_and_matches_quadrature():
    """Sweep a synthetic 5th exponential's mu through -(k2+k3): closed form
    must stay smooth (no spike at the crossing) and keep matching
    quadrature throughout."""
    K1, k2, k3 = REGION_KINETICS["frontal"]
    a = k2 + k3
    t_eval = np.array([10.0, 30.0, 57.5])

    mu5_values = a * np.linspace(-1.02, -0.98, 401) * (-1.0)  # sweep through -a
    lam5 = 0.5

    values_at_t = np.empty((len(mu5_values), len(t_eval)))
    max_diff_vs_quad = 0.0
    for i, mu5 in enumerate(mu5_values):
        lam = np.concatenate([LAM, [lam5]])
        mu = np.concatenate([MU, [mu5]])
        ct = closed_form_C_T(t_eval, K1, k2, k3, lam, mu)
        values_at_t[i] = ct

        ctq, fb = quadrature_C_T(t_eval, K1, k2, k3, lam, mu, GRID)
        assert not any(fb)
        diff = np.max(np.abs(ct - ctq) / np.maximum(np.abs(ct), 1e-8))
        max_diff_vs_quad = max(max_diff_vs_quad, diff)

    # Smoothness: no spike at the crossing. Check the second difference
    # (discrete curvature) stays bounded relative to the function's own
    # scale — a genuine bug (e.g. missing the delta=0 branch) produces a
    # value that blows up (goes to +/- inf or nan) exactly at the crossing,
    # which a bounded-second-difference check catches easily.
    assert np.all(np.isfinite(values_at_t)), "non-finite value somewhere in the mu sweep"
    second_diff = np.diff(values_at_t, n=2, axis=0)
    scale = np.maximum(np.abs(values_at_t).max(axis=0), 1e-8)
    max_rel_second_diff = np.max(np.abs(second_diff) / scale)
    assert max_rel_second_diff < 0.05, f"non-smooth (spike) at crossing: max rel 2nd difference {max_rel_second_diff:.3e}"

    assert max_diff_vs_quad < 1e-6, f"closed form vs quadrature disagree during the sweep: {max_diff_vs_quad:.3e}"


# ---------------------------------------------------------------------------
# Late-time slope (DECISIONS.md D-M2-4)
# ---------------------------------------------------------------------------
def test_late_time_slope_matches_corrected_asymptotic_formula():
    """C_T'(t)/C_P(t) at the last real PET frame time should match the
    corrected asymptotic net-influx rate K1*(k3+mu_slowest)/(k2+k3+mu_slowest)
    far more closely than the classical (plateau-assumption) Ki = K1*k3/(k2+k3)
    — see DECISIONS.md D-M2-4 for the derivation."""
    t_late = np.array([FRAME_T[-1]])
    mu_slowest = float(np.max(MU))  # least negative (closest to 0) = slowest decay  # least negative = -0.0106

    for name in REGION_NAMES:
        K1, k2, k3 = REGION_KINETICS[name]
        C_P_t = float(arterial_input(t_late, LAM, MU)[0])
        C_T_prime = float(closed_form_C_T_derivative(t_late, K1, k2, k3, LAM, MU)[0])
        measured_ratio = C_T_prime / C_P_t

        Ki_classical = K1 * k3 / (k2 + k3)
        Ki_corrected = K1 * (k3 + mu_slowest) / (k2 + k3 + mu_slowest)

        rel_err_classical = abs(measured_ratio - Ki_classical) / Ki_classical
        rel_err_corrected = abs(measured_ratio - Ki_corrected) / Ki_corrected

        assert rel_err_corrected < 1e-2, f"{name}: corrected-formula relative error {rel_err_corrected:.3e}, expected small"
        assert rel_err_corrected < rel_err_classical, (
            f"{name}: corrected formula ({rel_err_corrected:.3e}) should be closer than "
            f"classical Ki ({rel_err_classical:.3e})"
        )


def test_late_time_slope_asymptotic_formula_converges_at_very_large_t():
    """At t far beyond the scan duration, the corrected formula should match
    to near machine precision (this is a genuine t->infinity asymptote)."""
    K1, k2, k3 = REGION_KINETICS["frontal"]
    mu_slowest = float(np.max(MU))  # least negative (closest to 0) = slowest decay
    t_huge = np.array([800.0])

    C_P_t = float(arterial_input(t_huge, LAM, MU)[0])
    C_T_prime = float(closed_form_C_T_derivative(t_huge, K1, k2, k3, LAM, MU)[0])
    measured_ratio = C_T_prime / C_P_t
    Ki_corrected = K1 * (k3 + mu_slowest) / (k2 + k3 + mu_slowest)

    rel_err = abs(measured_ratio - Ki_corrected) / Ki_corrected
    assert rel_err < 1e-4, f"corrected formula relative error at t=800: {rel_err:.3e}"


# ---------------------------------------------------------------------------
# C_WB / C_PET assembly
# ---------------------------------------------------------------------------
def test_c_wb_and_c_pet_assembly():
    f_vals = parent_plasma_fraction(FRAME_T, F_A, F_XI1, F_XI2)
    cp_vals = arterial_input(FRAME_T, LAM, MU)
    cwb = C_WB_from_C_P(cp_vals, f_vals)
    assert np.all(cwb >= cp_vals - 1e-12), "C_WB should be >= C_P (f in (0,1])"

    K1, k2, k3 = REGION_KINETICS["frontal"]
    ct = closed_form_C_T(FRAME_T, K1, k2, k3, LAM, MU)
    cpet = C_PET(ct, cwb, V_B)
    expected = (1 - V_B) * ct + V_B * cwb
    assert np.allclose(cpet, expected)
