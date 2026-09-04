"""M1 acceptance tests for src/linalg.py (LU with partial pivoting).

Track B usage note (CLAUDE.md section 1): `numpy.linalg.solve` and
`numpy.linalg.cond` are used here, in tests/, purely as an independent
reference to check Track A correctness. `numpy.random` is used here only to
generate synthetic test matrices, which is not part of the scientific
pipeline itself.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.linalg import SingularMatrixError, lu_factor, lu_solve, solve

# ---------------------------------------------------------------------------
# 200 random well-conditioned systems, sizes 5..100 (PLAN.md M1 acceptance).
# ---------------------------------------------------------------------------
N_SYSTEMS = 200
SIZES = np.linspace(5, 100, N_SYSTEMS).astype(int)


def _random_well_conditioned(n: int, rng: np.random.Generator) -> np.ndarray:
    """A well-conditioned n x n matrix: diagonally dominant, so partial
    pivoting has an easy time and cond(A) stays modest for all n in 5..100."""
    A = rng.normal(size=(n, n))
    A += n * np.eye(n)  # strengthen diagonal dominance
    return A


def test_lu_solves_200_random_systems_accurately():
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

    # Reported in handoffs/RUN_M1.md with the actual measured values; these
    # thresholds are deliberately tight enough to fail on a broken
    # elimination or a permutation-bookkeeping bug, not just "loose enough
    # to pass" (CLAUDE.md section 3).
    assert max_rel_residual < 1e-9, f"max relative residual too large: {max_rel_residual:.3e}"
    assert max_rel_error < 1e-8, f"max relative error too large: {max_rel_error:.3e}"


def test_lu_matches_numpy_solve_on_200_random_systems():
    rng = np.random.default_rng(7)
    max_rel_diff = 0.0
    for n in SIZES:
        A = _random_well_conditioned(int(n), rng)
        b = rng.normal(size=n)

        x_ours = solve(A, b)
        x_ref = np.linalg.solve(A, b)  # Track B reference

        rel_diff = np.linalg.norm(x_ours - x_ref) / max(np.linalg.norm(x_ref), 1e-300)
        max_rel_diff = max(max_rel_diff, rel_diff)

    assert max_rel_diff < 1e-8, f"max relative diff vs numpy.linalg.solve: {max_rel_diff:.3e}"


def test_lu_factor_reproduces_A_up_to_permutation():
    """Direct structural check: P @ A == L @ U exactly (to round-off)."""
    rng = np.random.default_rng(11)
    A = _random_well_conditioned(12, rng)
    LU, piv = lu_factor(A)
    n = A.shape[0]
    L = np.tril(LU, -1) + np.eye(n)
    U = np.triu(LU)
    PA = A[piv]
    residual = np.linalg.norm(PA - L @ U) / np.linalg.norm(A)
    assert residual < 1e-12, f"P@A != L@U, relative residual {residual:.3e}"


def test_lu_on_hilbert_8x8_ill_conditioned():
    """Hilbert 8x8: deliberately ill-conditioned (PLAN.md M1 acceptance).

    We do not assert a tight tolerance here (that would defeat the point of
    the exercise) — we only assert the solve *runs* and *report* the error
    and condition number for handoffs/RUN_M1.md to discuss.
    """
    n = 8
    A = np.array([[1.0 / (i + j + 1) for j in range(n)] for i in range(n)])
    x_true = np.ones(n)
    b = A @ x_true

    x = solve(A, b)
    cond = np.linalg.cond(A)  # Track B, diagnostic only
    rel_error = np.linalg.norm(x - x_true) / np.linalg.norm(x_true)

    # Sanity bounds loose enough to always pass but tight enough to catch a
    # completely broken solver (e.g. NaN, or error >> 1).
    assert np.isfinite(rel_error)
    assert rel_error < 1.0, f"LU blew up on Hilbert(8): rel_error={rel_error:.3e}, cond={cond:.3e}"


def test_lu_detects_exact_singular_matrix():
    A = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [1.0, 1.0, 1.0]])  # row2 = 2*row1
    b = np.array([1.0, 2.0, 3.0])
    with pytest.raises(SingularMatrixError):
        solve(A, b)


def test_lu_partial_pivoting_avoids_zero_pivot_that_naive_lu_hits():
    """A[0,0] == 0: an LU *without* pivoting would divide by zero immediately.
    This is the single most direct test that partial pivoting is actually
    wired in (not just present in the code as dead logic)."""
    A = np.array([[0.0, 1.0], [1.0, 1.0]])
    b = np.array([1.0, 2.0])
    x = solve(A, b)
    assert np.allclose(A @ x, b, atol=1e-12)
