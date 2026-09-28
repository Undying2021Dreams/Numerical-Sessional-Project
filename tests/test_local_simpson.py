"""Independent exact-integral checks for the experimental Simpson formulation."""
import numpy as np
import pytest

from src.quadrature import simpson, simpson_diag


@pytest.mark.parametrize("offset", [0.0, 1000.0, 1e6])
def test_shifted_nonuniform_quadratic(offset):
    # Binary-exact nodes isolate weight cancellation from input rounding.
    u = np.array([0.0, 0.125, 0.5, 0.75, 1.0])
    y = 2.0 + 3.0 * u + 4.0 * u**2
    value, fallback = simpson_diag(offset + u, y, local_weights=True)
    assert abs(value - (2.0 + 1.5 + 4.0 / 3.0)) < 1e-13
    assert fallback is False


def test_fine_grid_sine_does_not_have_baseline_cancellation_floor():
    x = np.linspace(0.0, np.pi, 12801)
    assert abs(simpson(x, np.sin(x), local_weights=True) - 2.0) < 1e-12


def test_uniform_cubic_is_exact():
    x = np.linspace(-2.0, 3.0, 101)
    assert abs(simpson(x, x**3, local_weights=True) - 65.0 / 4.0) < 1e-12


def test_fourth_order_before_roundoff():
    errors = []
    for n in (17, 33, 65):
        x = np.linspace(0.0, np.pi, n)
        errors.append(abs(simpson(x, np.sin(x), local_weights=True) - 2.0))
    orders = np.log2(np.array(errors[:-1]) / errors[1:])
    assert np.all((orders > 3.9) & (orders < 4.1))


def test_leftover_interval_retains_trapezoid_policy():
    x = np.array([0.0, 0.25, 0.5, 1.0])
    # Exact quadratic on [0,.5], then the original trapezoid on [.5,1].
    expected = 0.5**3 / 3.0 + 0.5 * (0.5**2 + 1.0) / 2.0
    result, fallback = simpson_diag(x, x**2, local_weights=True)
    assert abs(result - expected) < 1e-15
    assert fallback is True


@pytest.mark.parametrize("n", [0, 1, 2])
def test_short_inputs_retain_baseline_behavior(n):
    x = np.arange(n, dtype=float)
    assert simpson_diag(x, x + 1.0, local_weights=True) == simpson_diag(x, x + 1.0)


def test_pet_forward_path_with_local_weights():
    from src.config import ARTERIAL_LAMBDA, ARTERIAL_MU, REGION_KINETICS, frame_midtimes_minutes
    from src.forward_model import GradedGridSpec, closed_form_C_T, quadrature_C_T

    t = frame_midtimes_minutes()
    for K1, k2, k3 in REGION_KINETICS.values():
        exact = closed_form_C_T(t, K1, k2, k3, ARTERIAL_LAMBDA, ARTERIAL_MU)
        value, flags = quadrature_C_T(
            t, K1, k2, k3, ARTERIAL_LAMBDA, ARTERIAL_MU,
            GradedGridSpec(n=6401, q=3.0), local_weights=True,
        )
        assert not any(flags)
        assert np.max(np.abs((value - exact) / exact)) < 1e-11
