"""M1 acceptance tests for src/qr.py (Householder QR: factor + solve + lstsq).

Track B usage note: `numpy.linalg.solve`, `numpy.linalg.lstsq`, and
`numpy.linalg.cond` are used here, in tests/, purely as an independent
reference. `numpy.random` generates synthetic test problems only.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.qr import lstsq, qr_factor, qr_solve, solve

N_SYSTEMS = 200
SIZES = np.linspace(5, 100, N_SYSTEMS).astype(int)


def _random_well_conditioned(n: int, rng: np.random.Generator) -> np.ndarray:
    A = rng.normal(size=(n, n))
    A += n * np.eye(n)
    return A


def test_qr_solves_200_random_systems_accurately():
    rng = np.random.default_rng(2024)
    max_rel_residual = 0.0
    max_rel_error = 0.0
    for n in SIZES:
        A = _random_well_conditioned(int(n), rng)
        x_true = rng.normal(size=n)
        b = A @ x_true

        x = solve(A, b)

        rel_residual = np.linalg.norm(A @ x - b) / np.linalg.norm(b)
        rel_error = np.linalg.norm(x - x_true) / np.linalg.norm(x_true)
        max_rel_residual = max(max_rel_residual, rel_residual)
        max_rel_error = max(max_rel_error, rel_error)

    assert max_rel_residual < 1e-9, f"max relative residual too large: {max_rel_residual:.3e}"
    assert max_rel_error < 1e-8, f"max relative error too large: {max_rel_error:.3e}"


def test_qr_matches_numpy_solve_on_200_random_systems():
    rng = np.random.default_rng(7)
    max_rel_diff = 0.0
    for n in SIZES:
        A = _random_well_conditioned(int(n), rng)
        b = rng.normal(size=n)

        x_ours = solve(A, b)
        x_ref = np.linalg.solve(A, b)

        rel_diff = np.linalg.norm(x_ours - x_ref) / max(np.linalg.norm(x_ref), 1e-300)
        max_rel_diff = max(max_rel_diff, rel_diff)

    assert max_rel_diff < 1e-8, f"max relative diff vs numpy.linalg.solve: {max_rel_diff:.3e}"


def test_qr_factor_R_is_upper_triangular_and_reproduces_A():
    rng = np.random.default_rng(3)
    A = _random_well_conditioned(15, rng)
    qr = qr_factor(A)

    below_diag = np.tril(qr.R, -1)
    assert np.max(np.abs(below_diag)) < 1e-11, "R has nonzero entries below the diagonal"

    # Reconstruct Q explicitly (m small enough to afford this) and check A = QR.
    from src.qr import apply_q

    Q_cols = [apply_q(qr, e) for e in np.eye(15)]
    Q = np.column_stack(Q_cols)
    residual = np.linalg.norm(Q @ qr.R - A) / np.linalg.norm(A)
    assert residual < 1e-10, f"Q@R != A, relative residual {residual:.3e}"

    orth_residual = np.linalg.norm(Q.T @ Q - np.eye(15))
    assert orth_residual < 1e-9, f"Q not orthogonal, ||Q^T Q - I|| = {orth_residual:.3e}"


def test_qr_least_squares_overdetermined_matches_numpy_lstsq():
    rng = np.random.default_rng(5)
    m, n = 50, 8
    A = rng.normal(size=(m, n))
    x_true = rng.normal(size=n)
    noise = 1e-3 * rng.normal(size=m)
    b = A @ x_true + noise

    x_ours, resid_ours = lstsq(A, b)
    x_ref, res_ref, rank_ref, sv_ref = np.linalg.lstsq(A, b, rcond=None)  # Track B

    rel_diff = np.linalg.norm(x_ours - x_ref) / np.linalg.norm(x_ref)
    assert rel_diff < 1e-8, f"lstsq solution differs from numpy: {rel_diff:.3e}"

    resid_ref = np.linalg.norm(A @ x_ref - b)
    assert abs(resid_ours - resid_ref) / resid_ref < 1e-6


def test_qr_on_hilbert_8x8_ill_conditioned():
    n = 8
    A = np.array([[1.0 / (i + j + 1) for j in range(n)] for i in range(n)])
    x_true = np.ones(n)
    b = A @ x_true

    x = solve(A, b)
    cond = np.linalg.cond(A)
    rel_error = np.linalg.norm(x - x_true) / np.linalg.norm(x_true)

    assert np.isfinite(rel_error)
    assert rel_error < 1.0, f"QR blew up on Hilbert(8): rel_error={rel_error:.3e}, cond={cond:.3e}"


def test_qr_square_solve_matches_lu_on_hilbert_8x8():
    """LU and QR should agree with each other on the same ill-conditioned
    system even if both disagree somewhat with the exact answer — this is
    the 'agreement between the two paths' half of the PLAN.md M1 criterion."""
    from src.linalg import solve as lu_solve_convenience

    n = 8
    A = np.array([[1.0 / (i + j + 1) for j in range(n)] for i in range(n)])
    b = A @ np.ones(n)

    x_lu = lu_solve_convenience(A, b)
    x_qr = solve(A, b)

    rel_diff = np.linalg.norm(x_lu - x_qr) / np.linalg.norm(x_lu)
    assert rel_diff < 1e-4, f"LU and QR disagree substantially on Hilbert(8): {rel_diff:.3e}"
