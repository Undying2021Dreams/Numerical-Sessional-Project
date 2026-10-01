"""Track A: power method and inverse power iteration for symmetric matrices.

The project brief: "eigenvalue decomposition: power method" is explicitly
named as a graded course topic. Used in M3 for the conditioning analysis of
F'^T F' (largest eigenvalue via power method, smallest via inverse power
iteration using our own LU, full spectrum via repeated deflation).
"""
from __future__ import annotations

import numpy as np

from src.linalg import lu_factor, lu_solve


def _norm(v: np.ndarray) -> float:
    return float(np.sqrt(v @ v))


def power_method(
    M: np.ndarray, v0: np.ndarray | None = None, max_iter: int = 2000, tol: float = 1e-13
) -> tuple[float, np.ndarray, int]:
    """Largest-magnitude eigenvalue/eigenvector of symmetric `M` via power
    iteration. Returns (eigenvalue, unit eigenvector, iterations used)."""
    n = M.shape[0]
    v = np.ones(n) if v0 is None else np.asarray(v0, dtype=np.float64).copy()
    v = v / _norm(v)
    lam_prev = 0.0
    for it in range(1, max_iter + 1):
        w = M @ v
        nw = _norm(w)
        if nw < 1e-300:
            return 0.0, v, it
        v = w / nw
        lam = float(v @ (M @ v))  # Rayleigh quotient
        if abs(lam - lam_prev) < tol * max(abs(lam), 1e-300):
            return lam, v, it
        lam_prev = lam
    return lam_prev, v, max_iter


def deflate(M: np.ndarray, lam: float, v: np.ndarray) -> np.ndarray:
    """Hotelling deflation: removes the (lam, v) eigenpair from symmetric M
    while leaving the other eigenpairs unchanged, so power_method can be
    re-applied to find the next-largest eigenvalue."""
    return M - lam * np.outer(v, v)


def symmetric_eigendecomposition(
    M: np.ndarray, max_iter: int = 2000, tol: float = 1e-13
) -> tuple[np.ndarray, np.ndarray]:
    """Full eigendecomposition of a symmetric matrix via repeated power
    method + Hotelling deflation (the course's power method, applied n
    times). Returns (eigenvalues descending, eigenvectors as columns)."""
    n = M.shape[0]
    Mk = M.copy()
    eigvals = np.empty(n)
    eigvecs = np.empty((n, n))
    for k in range(n):
        lam, v, _ = power_method(Mk, max_iter=max_iter, tol=tol)
        eigvals[k] = lam
        eigvecs[:, k] = v
        Mk = deflate(Mk, lam, v)
    return eigvals, eigvecs


def inverse_power_iteration(
    M: np.ndarray, shift: float = 0.0, v0: np.ndarray | None = None, max_iter: int = 500, tol: float = 1e-13
) -> tuple[float, np.ndarray, int]:
    """Smallest-magnitude eigenvalue of `M` near `shift`, via inverse power
    iteration on `(M - shift*I)` solved with our own LU (factored once,
    reused every iteration). Returns (eigenvalue of M, unit eigenvector,
    iterations used)."""
    n = M.shape[0]
    Ms = M - shift * np.eye(n)
    LU, piv = lu_factor(Ms)
    v = np.ones(n) if v0 is None else np.asarray(v0, dtype=np.float64).copy()
    v = v / _norm(v)
    lam_prev = 0.0
    for it in range(1, max_iter + 1):
        w = lu_solve(LU, piv, v)
        nw = _norm(w)
        v = w / nw
        lam = float(v @ (M @ v))
        if abs(lam - lam_prev) < tol * max(abs(lam), 1e-300):
            return lam, v, it
        lam_prev = lam
    return lam_prev, v, max_iter
