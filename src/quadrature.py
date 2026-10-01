"""Track A numerical integration: trapezoid and Simpson on non-uniform grids.

The project brief: "our own trapezoid, Simpson, and (if used) Romberg."
Non-uniform support is required (not optional) because the forward model
(M2) evaluates PET time-activity curves on the paper's 25 non-uniform frame
midtimes (config.frame_midtimes_minutes), so a Track A integrator that only
handles equal spacing would be useless for the actual pipeline — see
DECISIONS.md D-M1-4.
"""
from __future__ import annotations

import numpy as np


def trapezoid(x: np.ndarray, y: np.ndarray) -> float:
    """Composite trapezoid rule on an arbitrary (non-uniform) grid.

    integral ~= sum_i (x[i+1]-x[i]) * (y[i]+y[i+1]) / 2
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError(f"x and y must have the same shape, got {x.shape} vs {y.shape}")
    if x.shape[0] < 2:
        return 0.0
    h = x[1:] - x[:-1]
    return float(np.sum(h * (y[:-1] + y[1:]) / 2.0))


def _quadratic_segment_integral(
    x0: float, x1: float, x2: float, y0: float, y1: float, y2: float
) -> float:
    """Exact integral over [x0, x2] of the unique quadratic through the three
    points (x0,y0), (x1,y1), (x2,y2), via closed-form integration of the
    Lagrange quadratic basis (reduces to the textbook Simpson 1/3 weights
    (h/3, 4h/3, h/3) when x1-x0 == x2-x1 == h; see tests/test_quadrature.py
    for that equal-spacing check).
    """

    def basis_integral(xa: float, xb: float, xc: float) -> float:
        # L(x) = (x - xb)(x - xc) / ((xa - xb)(xa - xc));
        # (x-xb)(x-xc) = x^2 - (xb+xc)x + xb*xc, antiderivative below.
        denom = (xa - xb) * (xa - xc)

        def antideriv(x: float) -> float:
            return x**3 / 3.0 - (xb + xc) * x**2 / 2.0 + xb * xc * x

        return (antideriv(x2) - antideriv(x0)) / denom

    w0 = basis_integral(x0, x1, x2)
    w1 = basis_integral(x1, x0, x2)
    w2 = basis_integral(x2, x0, x1)
    return w0 * y0 + w1 * y1 + w2 * y2


def simpson(x: np.ndarray, y: np.ndarray) -> float:
    """Composite Simpson's rule on an arbitrary (non-uniform) grid.

    Pairs up consecutive intervals two at a time and integrates the local
    quadratic interpolant exactly (`_quadratic_segment_integral`) — this is
    algebraically identical to the standard composite Simpson 1/3 rule when
    the grid is uniform, and generalises correctly when it is not.

    If the number of intervals (len(x)-1) is odd, one interval is left over;
    it is closed with a single trapezoid step (DECISIONS.md D-M1-4) and the
    fallback is recorded in the returned diagnostics via `simpson_diag`
    (this function returns only the value; see `simpson_diag` for the flag).
    """
    value, _ = simpson_diag(x, y)
    return value


def simpson_diag(x: np.ndarray, y: np.ndarray) -> tuple[float, bool]:
    """Same as `simpson`, but also returns whether the odd-leftover trapezoid
    fallback was used (True = fallback triggered on this call)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError(f"x and y must have the same shape, got {x.shape} vs {y.shape}")
    n = x.shape[0]
    if n < 2:
        return 0.0, False
    if n == 2:
        return trapezoid(x, y), False

    total = 0.0
    i = 0
    while i + 2 <= n - 1:
        total += _quadratic_segment_integral(x[i], x[i + 1], x[i + 2], y[i], y[i + 1], y[i + 2])
        i += 2

    fallback_used = False
    if i < n - 1:
        # One interval left over (odd number of intervals): close it with a
        # trapezoid step rather than leaving it unintegrated.
        total += trapezoid(x[i:], y[i:])
        fallback_used = True

    return float(total), fallback_used
