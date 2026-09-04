"""M1 acceptance tests for src/quadrature.py (trapezoid, Simpson).

Includes the convergence-order check from PLAN.md M1: fit the log-log
error-vs-h slope and assert it lands near the textbook orders (2 for
trapezoid, 4 for Simpson). `numpy.polyfit`-free: the slope is fit with our
own least-squares via `src.qr.lstsq` so this test does not sneak a library
solver in through the back door even though tests/ is a permitted Track B
zone (using our own Track A code here is stronger evidence of correctness,
so we do it here even though it wasn't required).

**Why some integrands measure Simpson order ~6, not ~4 (read before editing
the `> 3.5` assertions below):** composite Simpson's rule has an asymptotic
error expansion (see DECISIONS.md D-M1-8 for the full derivation and the
reviewer-supplied reference form)

    E(h) = (h^4 / 180) * [f'''(a) - f'''(b)]  +  O(h^6)

i.e. the leading h^4 term is a *boundary* quantity — it depends only on the
third derivative at the two endpoints of the interval, not on any interior
behaviour (the interior contributions telescope away in the composite sum).
Consequently: whenever f'''(a) == f'''(b), the h^4 term vanishes identically
and the true convergence order is (at least) 6, not a bug. `runge_0_1` hits
this by an accident of even symmetry (f is even about x=0, so its odd-order
derivatives vanish at x=0, and f'''(1) also happens to be 0 — see D-M1-8).
`endpoint_matched_0_b` below is the *deliberate* version of the same
mechanism: f'''(a) = f'''(b) = -8 by construction (not by both sides
happening to be zero), chosen specifically to turn this into a tested
prediction rather than an unexplained observation (see D-M1-8).
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from src.quadrature import _quadratic_segment_integral, simpson, simpson_diag, trapezoid
from src.qr import lstsq

# f(x) = cos(x) + sin(2x); f'''(x) = sin(x) - 8 cos(2x).
# f'''(0) = -8. b = 3.204133415386284 is a numerically-found second root of
# f'''(x) = -8 in (0, 2*pi) (found once, offline, via bisection on a coarse
# scan + refinement — see DECISIONS.md D-M1-8; deliberately NOT pi, which is
# also a root but gives f odd about the midpoint and an exact-zero integral,
# see D-M1-8 for why that degenerate case was rejected). Exact integral via
# the closed-form antiderivative F(x) = sin(x) - cos(2x)/2, no library calls.
_EPM_A, _EPM_B = 0.0, 3.204133415386284


def _endpoint_matched_f(x: float) -> float:
    return math.cos(x) + math.sin(2 * x)


def _endpoint_matched_fppp(x: float) -> float:
    return math.sin(x) - 8.0 * math.cos(2 * x)


def _endpoint_matched_F(x: float) -> float:
    return math.sin(x) - math.cos(2 * x) / 2.0


_EPM_EXACT = _endpoint_matched_F(_EPM_B) - _endpoint_matched_F(_EPM_A)

# (name, f, F_exact_definite_integral, a, b)
KNOWN_INTEGRALS = [
    ("sin_0_pi", math.sin, 2.0, 0.0, math.pi),
    ("exp_0_1", math.exp, math.e - 1.0, 0.0, 1.0),
    ("runge_0_1", lambda x: 1.0 / (1.0 + x**2), math.pi / 4.0, 0.0, 1.0),
    ("endpoint_matched_0_b", _endpoint_matched_f, _EPM_EXACT, _EPM_A, _EPM_B),
]


def _fit_log_log_slope(hs: np.ndarray, errs: np.ndarray) -> float:
    """Fit log(err) = slope*log(h) + intercept via our own QR least squares."""
    mask = errs > 0
    log_h = np.log(hs[mask])
    log_e = np.log(errs[mask])
    design = np.column_stack([log_h, np.ones_like(log_h)])
    coeffs, _resid = lstsq(design, log_e)
    return float(coeffs[0])


@pytest.mark.parametrize("name,f,exact,a,b", KNOWN_INTEGRALS)
def test_trapezoid_and_simpson_converge_to_known_integral(name, f, exact, a, b):
    # n=129 keeps Simpson's truncation error well above the float64 round-off
    # floor we measured empirically (~1e-10 to 1e-13 depending on the
    # integrand, reached around n=257-513; see DECISIONS.md D-M1-7). At
    # n=2001 the true Simpson error is dominated by round-off, not
    # truncation, and can be *larger* than at n=513 — that U-shaped
    # error-vs-h curve is expected numerical-analysis behaviour, not a bug,
    # but it means a fixed absolute tolerance must be checked at an n where
    # truncation error still dominates.
    n = 129
    x = np.linspace(a, b, n)
    y = np.array([f(xi) for xi in x])

    trap_val = trapezoid(x, y)
    simp_val = simpson(x, y)

    assert abs(trap_val - exact) < 2e-4, f"{name}: trapezoid off by {abs(trap_val-exact):.3e}"
    assert abs(simp_val - exact) < 1e-7, f"{name}: simpson off by {abs(simp_val-exact):.3e}"


@pytest.mark.parametrize("name,f,exact,a,b", KNOWN_INTEGRALS)
def test_convergence_orders_match_theory(name, f, exact, a, b):
    # Restricted to n <= 33: empirically (see handoffs/RUN_M1.md) this is the
    # largest common range where all three integrands' Simpson error is still
    # truncation-dominated (well above the float64 round-off floor), so the
    # fitted log-log slope reflects the algorithm's order, not noise.
    trap_ns = np.array([9, 17, 33, 65, 129])
    simp_ns = np.array([9, 17, 33])

    def errors_at(ns, rule):
        errs = np.empty(len(ns))
        for i, n in enumerate(ns):
            x = np.linspace(a, b, n)
            y = np.array([f(xi) for xi in x])
            errs[i] = abs(rule(x, y) - exact)
        return errs

    trap_hs = (b - a) / (trap_ns - 1)
    simp_hs = (b - a) / (simp_ns - 1)
    trap_errs = errors_at(trap_ns, trapezoid)
    simp_errs = errors_at(simp_ns, simpson)

    # err ~ C * h^p  =>  log(err) = p*log(h) + log(C): the fitted slope IS
    # the order p directly (no sign flip — h -> 0 makes both log(h) and
    # log(err) -> -infinity together for p > 0).
    trap_order = _fit_log_log_slope(trap_hs, trap_errs)
    simp_order = _fit_log_log_slope(simp_hs, simp_errs)

    # Reported precisely (not just pass/fail) in handoffs/RUN_M1.md. Only a
    # lower bound is asserted: `runge_0_1` and `endpoint_matched_0_b` both
    # show genuine order ~6 in this regime because f'''(a) == f'''(b) for
    # both (see the module docstring and DECISIONS.md D-M1-8) — a real,
    # explained, *predicted* numerical finding, not a bug. An upper bound
    # would wrongly fail on it; the dedicated tests above pin the order-6
    # cases down more tightly.
    assert trap_order > 1.8, f"{name}: trapezoid convergence order {trap_order:.3f}, expected ~2"
    assert simp_order > 3.5, f"{name}: simpson convergence order {simp_order:.3f}, expected ~4"


def test_endpoint_matched_integrand_has_equal_third_derivatives_by_construction():
    """Confirms the deliberate construction actually has the claimed property
    (not just 'looks like it does') before we build a superconvergence claim
    on top of it."""
    diff = _endpoint_matched_fppp(_EPM_A) - _endpoint_matched_fppp(_EPM_B)
    assert abs(diff) < 1e-9, f"f'''(a) - f'''(b) = {diff:.3e}, expected ~0 by construction"


def test_endpoint_matched_integrand_shows_genuine_order_six_not_degenerate_exactness():
    """This is the tested prediction from DECISIONS.md D-M1-8: an integrand
    deliberately built so f'''(a) = f'''(b) (but NOT trivially odd/even about
    the domain, unlike a naive construction — see D-M1-8's discussion of why
    the first two attempts at this integrand were rejected) should show
    Simpson order ~6, clearly above the generic ~4 and clearly distinguishable
    from the ~inf ('exact to round-off at every h') degenerate case a
    symmetric construction accidentally produces.
    """
    a, b, f, exact = _EPM_A, _EPM_B, _endpoint_matched_f, _EPM_EXACT
    ns = np.array([9, 17, 33])
    hs = (b - a) / (ns - 1)
    errs = np.empty(len(ns))
    for i, n in enumerate(ns):
        x = np.linspace(a, b, n)
        y = np.array([f(xi) for xi in x])
        errs[i] = abs(simpson(x, y) - exact)

    # Not already at the round-off floor at the coarsest grid (n=9) — this is
    # what distinguishes a genuine order-6 measurement from the degenerate
    # fully-antisymmetric construction discussed in D-M1-8, which is "exact"
    # (~1e-15) even at n=9 for reasons unrelated to the h^4-term argument.
    assert errs[0] > 1e-8, f"error at n=9 is {errs[0]:.3e}: suspiciously small, check for a degenerate construction"

    order = _fit_log_log_slope(hs, errs)
    assert 5.5 < order < 6.5, f"measured order {order:.3f}, expected ~6 (see DECISIONS.md D-M1-8)"


def test_quadratic_segment_integral_reduces_to_uniform_simpson_weights():
    """Equal spacing h: I = h/3*(y0 + 4 y1 + y2) exactly (textbook Simpson 1/3)."""
    h = 0.37
    x0, x1, x2 = 1.0, 1.0 + h, 1.0 + 2 * h
    y0, y1, y2 = 2.3, -1.1, 4.4
    ours = _quadratic_segment_integral(x0, x1, x2, y0, y1, y2)
    textbook = h / 3.0 * (y0 + 4 * y1 + y2)
    assert abs(ours - textbook) < 1e-12


def test_simpson_exact_for_quadratics_on_nonuniform_grid():
    """A quadratic polynomial is integrated exactly by Simpson on ANY grid
    (uniform or not), since the method interpolates a quadratic exactly."""
    rng = np.random.default_rng(0)
    a_coef, b_coef, c_coef = 1.7, -0.4, 2.2

    def f(x):
        return a_coef * x**2 + b_coef * x + c_coef

    def F(x):
        return a_coef * x**3 / 3.0 + b_coef * x**2 / 2.0 + c_coef * x

    x = np.sort(rng.uniform(0, 5, size=9))
    x[0], x[-1] = 0.0, 5.0
    y = f(x)

    exact = F(5.0) - F(0.0)
    got, fallback_used = simpson_diag(x, y)
    assert abs(got - exact) < 1e-9, f"non-uniform simpson not exact for a quadratic: {got} vs {exact}"


def test_simpson_odd_number_of_intervals_uses_documented_trapezoid_fallback():
    """5 points = 4 intervals (even) -> no fallback. 4 points = 3 intervals
    (odd) -> fallback must trigger (DECISIONS.md D-M1-4)."""
    x_even = np.linspace(0, 1, 5)
    y_even = x_even**2
    _, fb_even = simpson_diag(x_even, y_even)
    assert fb_even is False

    x_odd = np.linspace(0, 1, 4)
    y_odd = x_odd**2
    _, fb_odd = simpson_diag(x_odd, y_odd)
    assert fb_odd is True


def test_trapezoid_exact_for_linear_function_on_nonuniform_grid():
    rng = np.random.default_rng(1)
    x = np.sort(rng.uniform(0, 10, size=13))
    y = 3.0 * x - 2.0
    exact = 1.5 * (10.0**2 - 0.0**2) - 2.0 * (x[-1] - x[0])
    # integral of (3x-2) from x[0] to x[-1]:
    exact = 1.5 * (x[-1] ** 2 - x[0] ** 2) - 2.0 * (x[-1] - x[0])
    got = trapezoid(x, y)
    assert abs(got - exact) < 1e-10
