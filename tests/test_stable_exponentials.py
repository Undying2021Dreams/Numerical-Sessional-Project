"""Independent Decimal references and stress cases for stable PET evaluation."""
from decimal import Decimal, localcontext

import numpy as np
import pytest

from src.forward_model import (
    _convolution_moments, _phi1_prime, closed_form_C_T, quadrature_C_T, GradedGridSpec,
)
from src.jacobian import _dCT_block


def decimal_moments(t, a, mu):
    with localcontext() as ctx:
        ctx.prec = 80
        t, a, mu = map(Decimal.from_float, map(float, (t, a, mu)))
        delta = a + mu
        ea, em = (-a*t).exp(), (mu*t).exp()
        if delta == 0:
            return tuple(map(float, (t*ea, t*t*ea/2, -t*t*ea/2)))
        numerator = em - ea
        return tuple(map(float, (numerator/delta,
                                 (t*em*delta-numerator)/delta**2,
                                 (t*ea*delta-numerator)/delta**2)))


@pytest.mark.parametrize("x", [-1e8, -1000., -100., -2., -1.00001, -1., -1e-8, 0., 1e-8, 1., 1.00001, 2., 10.])
def test_phi1_prime_against_80_digit_reference(x):
    with localcontext() as ctx:
        ctx.prec = 80
        z = Decimal.from_float(x)
        ref = float(((z-1)*z.exp()+1)/(z*z)) if x else 0.5
    assert float(_phi1_prime(np.array(x))) == pytest.approx(ref, rel=3e-15, abs=0.0)


@pytest.mark.parametrize("t,a,mu", [
    (0., .3, -.1), (57.5, .3, -.01), (57.5, .3, -13.45),
    (2., .3, -.3), (2., .3, -.3+1e-12), (2., .3, -.3-1e-12),
    (60., 20., -.01), (60., .3, -10000.), (2., .3, .1),
])
def test_convolution_and_derivatives_against_80_digit_reference(t, a, mu):
    values = _convolution_moments(np.array([t]), a, np.array([mu]))
    for value, ref in zip(values, decimal_moments(t, a, mu)):
        assert value[0, 0] == pytest.approx(ref, rel=2e-12, abs=1e-300)


def test_large_decay_forward_and_jacobian_remain_finite_and_match_differences():
    t = np.array([0.0, .1, 1., 60.])
    lam = np.array([1.])
    mu = np.array([-.01])
    K1, k2, k3 = .2, 19., 1.
    with np.errstate(over="raise", invalid="raise"):
        result = closed_form_C_T(t, K1, k2, k3, lam, mu)
        derivatives = _dCT_block(t, K1, k2, k3, lam, mu)
    assert np.all(np.isfinite(result))
    for index, derivative in enumerate(derivatives[:3]):
        params = [K1, k2, k3]
        h = 1e-5 * max(1., params[index])
        params[index] += h
        plus = closed_form_C_T(t, *params, lam, mu)
        params[index] -= 2*h
        minus = closed_form_C_T(t, *params, lam, mu)
        np.testing.assert_allclose(derivative, (plus-minus)/(2*h), rtol=1e-7, atol=1e-10)


def test_quadrature_decay_inside_integral_avoids_intermediate_overflow():
    t = np.array([60.])
    lam, mu = np.array([1.]), np.array([-.01])
    # The sharp endpoint layer requires a fine grid; this test isolates overflow,
    # not a claim that the default graded grid resolves all possible parameters.
    with np.errstate(over="raise", invalid="raise"):
        values, _ = quadrature_C_T(t, .2, 19., 1., lam, mu, GradedGridSpec(25601, 1.))
    exact = closed_form_C_T(t, .2, 19., 1., lam, mu)
    np.testing.assert_allclose(values, exact, rtol=1e-8)


def test_local_simpson_is_now_the_default():
    from src.quadrature import simpson
    x = np.linspace(0., np.pi, 12801)
    assert simpson(x, np.sin(x)) == simpson(x, np.sin(x), local_weights=True)
    assert abs(simpson(x, np.sin(x))-2.) < 1e-12
