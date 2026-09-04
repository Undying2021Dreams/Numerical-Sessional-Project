"""Track A dense linear algebra: Householder QR (factor + solve + least squares).

CLAUDE.md section 1: "our own Householder QR." Q is never formed explicitly;
instead each reflector is stored as a vector and applied to a right-hand
side on demand (`apply_qt`), which is both the standard efficient approach
and the one that avoids ever materialising an m x m matrix for tall systems.

Reuses `src.linalg.back_substitute` for the final triangular solve so the
LU and QR paths share one tested triangular-solve implementation.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.linalg import back_substitute


@dataclass
class QRFactors:
    """Householder QR factorisation of an (m, n) matrix A, m >= n.

    `R` is the full (m, n) matrix after applying all reflectors (upper
    triangular in its top n x n block, exactly zero below). `vs` holds one
    normalised Householder vector per eliminated column, each tagged with
    the row offset it applies from, in application order.
    """

    vs: list[tuple[int, np.ndarray]]
    R: np.ndarray
    m: int
    n: int


def qr_factor(A: np.ndarray) -> QRFactors:
    """Householder QR factorisation: A = Q R, Q orthogonal (m, m) (implicit),
    R upper triangular (m, n), for m >= n.

    Standard algorithm (see e.g. Golub & Van Loan, or Chapra & Canale ch. 9
    for the course's own treatment): for each column k, build a reflector
    that zeroes everything below the diagonal in that column, choosing the
    reflection sign to move away from the existing entry (avoiding
    catastrophic cancellation when forming `alpha`).
    """
    A = np.asarray(A, dtype=np.float64)
    m, n = A.shape
    if m < n:
        raise ValueError(f"qr_factor requires m >= n, got shape {A.shape}")

    R = A.copy()
    vs: list[tuple[int, np.ndarray]] = []

    for k in range(n):
        x = R[k:, k]
        norm_x = np.sqrt(x @ x)
        if norm_x == 0.0:
            # Column already zero below the diagonal here: no reflection
            # needed, but we still record a zero vector so `apply_qt`'s
            # bookkeeping (one entry per k) stays simple and uniform.
            vs.append((k, np.zeros_like(x)))
            continue

        # Sign choice: move alpha away from x[0] to avoid cancellation in
        # v = x - alpha*e1 when x[0] and alpha would otherwise nearly cancel.
        sign = 1.0 if x[0] >= 0 else -1.0
        alpha = -sign * norm_x

        v = x.copy()
        v[0] -= alpha
        v_norm = np.sqrt(v @ v)
        if v_norm > 0:
            v = v / v_norm
        vs.append((k, v))

        # Apply H_k = I - 2 v v^T to R[k:, k:] in place.
        R[k:, k:] -= 2.0 * np.outer(v, v @ R[k:, k:])

    return QRFactors(vs=vs, R=R, m=m, n=n)


def apply_qt(qr: QRFactors, b: np.ndarray) -> np.ndarray:
    """Compute Q^T b without forming Q, by re-applying the stored reflectors
    in the same order used during factorisation."""
    b = np.asarray(b, dtype=np.float64).copy()
    for k, v in qr.vs:
        if not np.any(v):
            continue
        b[k:] -= 2.0 * v * (v @ b[k:])
    return b


def apply_q(qr: QRFactors, y: np.ndarray) -> np.ndarray:
    """Compute Q y without forming Q, by applying the reflectors in reverse
    order (each Householder reflector is its own inverse/transpose)."""
    y = np.asarray(y, dtype=np.float64).copy()
    for k, v in reversed(qr.vs):
        if not np.any(v):
            continue
        y[k:] -= 2.0 * v * (v @ y[k:])
    return y


def qr_solve(qr: QRFactors, b: np.ndarray) -> np.ndarray:
    """Solve the square system A x = b (m == n) via A = QR."""
    if qr.m != qr.n:
        raise ValueError("qr_solve requires a square factorisation (m == n); use qr_lstsq")
    qtb = apply_qt(qr, b)
    return back_substitute(qr.R[: qr.n, : qr.n], qtb[: qr.n])


def qr_lstsq(qr: QRFactors, b: np.ndarray) -> tuple[np.ndarray, float]:
    """Least-squares solution of A x ~= b (m >= n) via A = QR.

    Standard result: with Q^T b = [c1; c2] (c1 the first n entries), the
    least-squares solution solves R[:n,:n] x = c1, and the minimal residual
    norm is ||c2|| exactly (Q orthogonal preserves norms).

    Returns (x, residual_norm).
    """
    qtb = apply_qt(qr, b)
    x = back_substitute(qr.R[: qr.n, : qr.n], qtb[: qr.n])
    residual_norm = float(np.sqrt(np.sum(qtb[qr.n :] ** 2))) if qr.m > qr.n else 0.0
    return x, residual_norm


def solve(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Convenience wrapper for the square case: factor and solve in one call."""
    qr = qr_factor(A)
    return qr_solve(qr, b)


def lstsq(A: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, float]:
    """Convenience wrapper for the rectangular (m >= n) least-squares case."""
    qr = qr_factor(A)
    return qr_lstsq(qr, b)
