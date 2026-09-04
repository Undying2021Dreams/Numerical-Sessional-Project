"""M3 acceptance tests: the K1/lambda null-space experiment (PLAN.md M3 item 3),
and its function-level counterpart (item 3b).

The forward model depends on K1^i and lambda_j ONLY through their product (K1
multiplies C_P linearly in eq. 1, and C_P is linear in lambda), so F^1 (the
tissue-data block) must have an exact null direction: scale every K1^i up by
c while scaling every lambda_j down by 1/c, for ANY c > 0 — not just an
infinitesimal one. Item 3b checks this directly at the function level
(cheap, cannot be faked); item 3 checks it shows up correctly in the
Jacobian's singular value structure.
"""
from __future__ import annotations

import numpy as np

from src.config import N_PARAMS, REGION_KINETICS, REGION_NAMES, blood_sample_times_minutes, frame_midtimes_minutes, ground_truth_vector
from src.eigen import inverse_power_iteration
from src.forward_model import C_WB_from_C_P, arterial_input, closed_form_C_T, parent_plasma_fraction
from src.jacobian import analytic_jacobian, unpack

T_FRAMES = frame_midtimes_minutes()
S_BLOOD = blood_sample_times_minutes()
X_TRUE = ground_truth_vector()
M_COLS = (8, 9, 10)


def _cwb_data(x):
    lam, mu, m, K1, k2, k3 = unpack(x)
    A, xi1, xi2 = m
    return C_WB_from_C_P(arterial_input(S_BLOOD, lam, mu), parent_plasma_fraction(S_BLOOD, A, xi1, xi2))


# ---------------------------------------------------------------------------
# 3b: function-level invariance (cannot be faked, independent of the Jacobian)
# ---------------------------------------------------------------------------
def test_C_T_invariant_under_K1_lambda_reciprocal_scaling():
    lam, mu, m, K1, k2, k3 = unpack(X_TRUE)
    for c in (0.5, 2.0):
        lam_scaled = lam / c
        for name in REGION_NAMES:
            K1_i, k2_i, k3_i = REGION_KINETICS[name]
            ct_true = closed_form_C_T(T_FRAMES, K1_i, k2_i, k3_i, lam, mu)
            ct_scaled = closed_form_C_T(T_FRAMES, K1_i * c, k2_i, k3_i, lam_scaled, mu)
            max_abs_diff = float(np.max(np.abs(ct_true - ct_scaled)))
            max_scale = float(np.max(np.abs(ct_true)))
            assert max_abs_diff < 1e-10 * max_scale, (
                f"{name}, c={c}: C_T changed under K1*c, lambda/c scaling — max abs diff {max_abs_diff:.3e}"
            )


# ---------------------------------------------------------------------------
# 3: the null direction in the Jacobian's singular value structure
# ---------------------------------------------------------------------------
def _predicted_null_direction():
    lam, mu, m, K1, k2, k3 = unpack(X_TRUE)
    v = np.zeros(N_PARAMS)
    v[0:4] = -lam
    for i in range(4):
        v[11 + 3 * i] = K1[i]
    return v / np.sqrt(v @ v)


def test_F1_restricted_jacobian_has_exact_null_direction_aligned_with_prediction():
    C_WB_data = _cwb_data(X_TRUE)
    J = analytic_jacobian(X_TRUE, T_FRAMES, S_BLOOD, C_WB_data)
    F1 = J[: 4 * len(T_FRAMES), :]

    v_pred = _predicted_null_direction()
    direct_check = float(np.sqrt(np.sum((F1 @ v_pred) ** 2)))
    scale = float(np.sqrt(np.sum(F1**2)))
    assert direct_check < 1e-10 * scale, (
        f"F1 @ v_pred should vanish (exact invariance): {direct_check:.3e} vs scale {scale:.3e}"
    )

    # Drop the m-block columns (8,9,10): F^1 does not depend on m at all, so
    # they are exact trivial null directions that would otherwise pollute
    # (and, as measured, corrupt) the identification of the K1/lambda
    # direction specifically — see DECISIONS.md D-M3-4.
    keep = [i for i in range(N_PARAMS) if i not in M_COLS]
    F1r = F1[:, keep]
    M1r = F1r.T @ F1r
    v_pred_r = v_pred[keep]
    v_pred_r = v_pred_r / np.sqrt(v_pred_r @ v_pred_r)

    lam_min, v_min, _ = inverse_power_iteration(M1r, shift=0.0)
    cos_align = abs(float(v_pred_r @ v_min))

    assert lam_min < 1e-8, f"F1-restricted smallest eigenvalue {lam_min:.3e}, expected ~0"
    assert cos_align > 1 - 1e-6, f"alignment with predicted direction: {cos_align:.8f}, expected ~1"


def test_adding_F2_block_removes_the_near_null_direction():
    C_WB_data = _cwb_data(X_TRUE)
    J = analytic_jacobian(X_TRUE, T_FRAMES, S_BLOOD, C_WB_data)
    F1 = J[: 4 * len(T_FRAMES), :]

    keep = [i for i in range(N_PARAMS) if i not in M_COLS]
    M1r = (F1[:, keep]).T @ (F1[:, keep])
    lam_min_F1_only, _, _ = inverse_power_iteration(M1r, shift=0.0)

    M_full = J.T @ J
    lam_min_full, _, _ = inverse_power_iteration(M_full, shift=0.0)

    assert abs(lam_min_F1_only) < 1e-8
    assert lam_min_full > 1e-8, f"full Jacobian's smallest eigenvalue {lam_min_full:.3e} should be measurably nonzero"
    # The increase should be dramatic (many orders of magnitude), not marginal.
    assert lam_min_full / max(abs(lam_min_F1_only), 1e-16) > 1e3
