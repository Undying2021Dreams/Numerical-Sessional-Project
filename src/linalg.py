"""Track A dense linear algebra: LU with partial pivoting.

Implements the two Track A linear-solve primitives named in the project brief
section 1 for the LU path: "our own LU with partial pivoting (factor +
solve)". Also exposes `forward_substitute`/`back_substitute`, which
`src/qr.py` reuses for the triangular solves after Householder reduction, so
the two solvers share one tested implementation of triangular solves rather
than duplicating it.

Only `np.ndarray`, elementwise arithmetic, slicing, and `@` (plain matmul,
used here only as a dot product over 1-D slices) are used — no
`numpy.linalg.*` routine of any kind.
"""
from __future__ import annotations

import numpy as np


class SingularMatrixError(RuntimeError):
    """Raised when LU factorisation encounters an (up to round-off) zero pivot."""


def lu_factor(A: np.ndarray, tol: float = 1e-300) -> tuple[np.ndarray, np.ndarray]:
    """Doolittle LU factorisation with partial (row) pivoting.

    Produces P, L, U implicitly such that P @ A = L @ U, where P is the
    permutation with a 1 in row i, column piv[i].

    Returns
    -------
    LU : (n, n) ndarray
        Combined factors: strict lower triangle holds the multipliers of L
        (unit diagonal implied, not stored), upper triangle including the
        diagonal holds U.
    piv : (n,) int ndarray
        Row permutation: piv[i] is the original row of A now in position i.

    Raises
    ------
    SingularMatrixError
        If the largest available pivot in some column is (numerically) zero.
    """
    A = np.asarray(A, dtype=np.float64)
    n, m = A.shape
    if n != m:
        raise ValueError(f"lu_factor requires a square matrix, got shape {A.shape}")

    U = A.copy()
    piv = np.arange(n)

    for k in range(n - 1):
        # Partial pivoting: bring the largest-magnitude entry in column k
        # (at or below row k) onto the diagonal, for numerical stability.
        p = k + int(np.argmax(np.abs(U[k:, k])))
        if p != k:
            U[[k, p], :] = U[[p, k], :]
            piv[[k, p]] = piv[[p, k]]

        pivot = U[k, k]
        if abs(pivot) < tol:
            raise SingularMatrixError(
                f"zero (or near-zero) pivot at column {k}: |pivot|={abs(pivot):.3e}"
            )

        # Eliminate below the pivot, storing multipliers in place of the
        # eliminated entries (Doolittle: L's strict lower triangle).
        multipliers = U[k + 1 :, k] / pivot
        U[k + 1 :, k] = multipliers
        U[k + 1 :, k + 1 :] -= np.outer(multipliers, U[k, k + 1 :])

    if abs(U[n - 1, n - 1]) < tol:
        raise SingularMatrixError(
            f"zero (or near-zero) pivot at column {n - 1}: |pivot|={abs(U[n - 1, n - 1]):.3e}"
        )

    return U, piv


def forward_substitute(L_unit: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve L y = b for y, where L is unit-lower-triangular (diagonal = 1,
    implied, not read from `L_unit`'s diagonal). `L_unit`'s strict lower
    triangle holds the multipliers."""
    n = L_unit.shape[0]
    y = np.empty(n, dtype=np.float64)
    for i in range(n):
        y[i] = b[i] - L_unit[i, :i] @ y[:i]
    return y


def back_substitute(U: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Solve U x = y for x, where U is upper triangular (square, n x n)."""
    n = U.shape[0]
    x = np.empty(n, dtype=np.float64)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - U[i, i + 1 :] @ x[i + 1 :]) / U[i, i]
    return x


def lu_solve(LU: np.ndarray, piv: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve A x = b given the (LU, piv) factorisation from `lu_factor`."""
    b = np.asarray(b, dtype=np.float64)
    pb = b[piv]
    y = forward_substitute(LU, pb)
    x = back_substitute(LU, y)
    return x


def solve(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Convenience wrapper: factor and solve A x = b in one call."""
    LU, piv = lu_factor(A)
    return lu_solve(LU, piv, b)
