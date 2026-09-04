"""Track A forward model: the irreversible two-tissue compartment model.

Implements:
  - `arterial_input`: C_P(t) = sum_j lambda_j exp(mu_j t) (Definition 2).
  - `parent_plasma_fraction`: f(t), biexponential model (Remark 18).
  - `closed_form_C_T`: eq. (3) of Lemma 6 — the closed-form solution of the
    ODE system (S) when C_P is polyexponential — rewritten in a numerically
    stable form (see the `_phi1` docstring and DECISIONS.md D-M2-1) that
    handles both degenerate branches (mu_j == 0, k2+k3+mu_j == 0) exactly,
    as one smooth expression rather than an if/else on exact equality.
  - `closed_form_C_T_derivative`: dC_T/dt in closed form, obtained by
    differentiating eq. (1) directly (not itself a numbered equation in the
    paper; derived here to test the late-time Patlak-slope claim precisely,
    without finite-difference error — see DECISIONS.md D-M2-4).
  - `quadrature_C_T`: eq. (1) of Lemma 5, evaluated by directly integrating
    the two definite integrals with our own Track A Simpson's rule on a
    dense per-t grid (DECISIONS.md D-M1-13 / D-M2-3), NOT on the 25-frame
    measurement grid.
  - `C_WB_from_C_P`, `C_PET`: the arterial whole-blood concentration and the
    PET measurement equation (Section 2 of the paper).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from src.quadrature import simpson_diag


def _phi1(x: np.ndarray) -> np.ndarray:
    """phi1(x) = (e^x - 1) / x for x != 0, phi1(0) = 1 (its removable
    singularity's exact limit). See DECISIONS.md D-M2-1 for the numerical
    stability analysis: built on `np.expm1`, which is accurate uniformly for
    ALL x (no separate small-x branch/threshold needed — the only special
    case is x == 0.0 exactly, a division-by-zero guard, not a precision
    workaround).
    """
    x = np.asarray(x, dtype=np.float64)
    safe_x = np.where(x == 0.0, 1.0, x)
    return np.where(x == 0.0, 1.0, np.expm1(x) / safe_x)


def arterial_input(t: np.ndarray, lam: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """C_P(t) = sum_j lambda_j exp(mu_j t) — Definition 2 (polyexponential)."""
    t = np.asarray(t, dtype=np.float64)
    lam = np.asarray(lam, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    return np.exp(np.outer(t, mu)) @ lam


def parent_plasma_fraction(t: np.ndarray, A: float, xi1: float, xi2: float) -> np.ndarray:
    """f(t) = A*exp(xi1*t) + (1-A)*exp(xi2*t) — Remark 18, biexponential model."""
    t = np.asarray(t, dtype=np.float64)
    return A * np.exp(xi1 * t) + (1.0 - A) * np.exp(xi2 * t)


def arterial_input_integral(t: np.ndarray, lam: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """int_0^t C_P(s) ds = t * sum_j lambda_j * phi1(mu_j t) (same phi1 stability
    trick as `closed_form_C_T`; exact for mu_j == 0 too). Used for the Patlak
    plot's normalised-time axis (DECISIONS.md D-M2-7): x(t) = this / C_P(t)."""
    t = np.asarray(t, dtype=np.float64)
    lam = np.asarray(lam, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    return t * (_phi1(np.outer(t, mu)) @ lam)


def C_WB_from_C_P(C_P_values: np.ndarray, f_values: np.ndarray) -> np.ndarray:
    """C_WB(t) = C_P(t) / f(t) (Section 2, the relation defining f)."""
    return np.asarray(C_P_values) / np.asarray(f_values)


def C_PET(C_T: np.ndarray, C_WB: np.ndarray, V_B: float) -> np.ndarray:
    """C_PET(t) = (1 - V_B) * C_T(t) + V_B * C_WB(t) (Section 2 / Remark 17)."""
    return (1.0 - V_B) * np.asarray(C_T) + V_B * np.asarray(C_WB)


def closed_form_C_T(
    t: np.ndarray, K1: float, k2: float, k3: float, lam: np.ndarray, mu: np.ndarray
) -> np.ndarray:
    """eq. (3) of Lemma 6: C_T(t) for a single region, C_P polyexponential.

    Derivation of the stable form actually implemented here (DECISIONS.md
    D-M2-1): substituting C_P(s) = sum_j lambda_j exp(mu_j s) into eq. (1)
    and evaluating the two resulting elementary integrals via
    `int_0^t exp(c s) ds = t * phi1(c t)` (exact for all c, including c=0)
    gives, with a = k2+k3, delta_j = a + mu_j:

        C_T(t) = (K1 k2 / a) * exp(-a t) * t * sum_j lambda_j * phi1(delta_j * t)
               +  (K1 k3 / a)             * t * sum_j lambda_j * phi1(mu_j * t)

    This is algebraically identical to the paper's eq. (3) (verified by hand
    and in tests/test_forward_model.py::test_stable_form_matches_paper_eq3_branches):
    expanding phi1 for delta_j != 0 / mu_j != 0 recovers exactly the paper's
    `e^{mu_j t}` and `e^{-(k2+k3)t}` coefficient terms, and phi1(0) = 1
    recovers exactly the paper's separate `t * e^{-(k2+k3)t}` / `t` branches
    for delta_j == 0 / mu_j == 0. Using phi1 throughout means those two
    "degenerate branches" PLAN.md M2 asks for are handled by ONE smooth
    formula rather than an explicit if/else on floating-point equality, and
    near-degenerate (small but nonzero) denominators never appear at all —
    see DECISIONS.md D-M2-1 for why this avoids the catastrophic
    cancellation a naive `(exp(mu_j*t) - exp(-a*t)) / (a+mu_j)` evaluation
    would have.
    """
    t = np.asarray(t, dtype=np.float64)
    lam = np.asarray(lam, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    a = k2 + k3
    delta = a + mu

    term1 = (K1 * k2 / a) * np.exp(-a * t) * t * (_phi1(np.outer(t, delta)) @ lam)
    term2 = (K1 * k3 / a) * t * (_phi1(np.outer(t, mu)) @ lam)
    return term1 + term2


def closed_form_term1(t: np.ndarray, K1: float, k2: float, k3: float, lam: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """The k2-part of `closed_form_C_T` alone: (K1 k2/a) e^{-at} int_0^t e^{as}C_P(s)ds.
    Exposed separately because `closed_form_C_T_derivative` needs it (see D-M2-4)."""
    t = np.asarray(t, dtype=np.float64)
    lam = np.asarray(lam, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    a = k2 + k3
    delta = a + mu
    return (K1 * k2 / a) * np.exp(-a * t) * t * (_phi1(np.outer(t, delta)) @ lam)


def closed_form_C_T_derivative(
    t: np.ndarray, K1: float, k2: float, k3: float, lam: np.ndarray, mu: np.ndarray
) -> np.ndarray:
    """dC_T/dt(t) in closed form (not a numbered paper equation; derived by
    differentiating eq. (1) — see the module docstring and DECISIONS.md
    D-M2-4).

        d/dt term1(t) = -a * term1(t) + (K1 k2 / a) * C_P(t)
        d/dt term2(t) =                 (K1 k3 / a) * C_P(t)
        => C_T'(t) = K1 * C_P(t) - a * term1(t)

    Used only to test the late-time Patlak-slope claim precisely (an exact
    derivative, not a finite-difference approximation, isolates the model's
    own behaviour from numerical differentiation error).
    """
    a = k2 + k3
    C_P_t = arterial_input(t, lam, mu)
    term1 = closed_form_term1(t, K1, k2, k3, lam, mu)
    return K1 * C_P_t - a * term1


# ---------------------------------------------------------------------------
# Quadrature path (eq. 1 of Lemma 5), dense per-t grid (DECISIONS.md D-M1-13)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class GradedGridSpec:
    """Grid used to integrate from 0 to t: t_i = t_end * u_i^q, u uniform in
    [0,1], concentrating points near t=0 where C_P's fast components
    (time constant ~4.5s for the fastest arterial exponential) live. `n` is
    always forced odd so `src.quadrature.simpson_diag`'s trapezoid fallback
    never triggers (an even number of intervals). Chosen by the
    grid-refinement study in experiments/m2_forward_model.py; see
    DECISIONS.md D-M1-13 and D-M2-3 for the numbers."""

    n: int
    q: float

    def build(self, t_end: float) -> np.ndarray:
        n = self.n if self.n % 2 == 1 else self.n + 1
        if t_end <= 0.0:
            return np.array([0.0])
        u = np.linspace(0.0, 1.0, n)
        return t_end * u**self.q


def quadrature_C_T(
    t_values: np.ndarray,
    K1: float,
    k2: float,
    k3: float,
    lam: np.ndarray,
    mu: np.ndarray,
    grid_spec: GradedGridSpec,
) -> tuple[np.ndarray, list[bool]]:
    """eq. (1) of Lemma 5, evaluated by direct Track A quadrature:

        C_T(t) = (K1 k2 / a) e^{-a t} int_0^t e^{a s} C_P(s) ds
               + (K1 k3 / a)          int_0^t C_P(s) ds

    For each requested t, a fresh dense grid from 0 to t is built (per
    DECISIONS.md D-M1-13 — this is NOT evaluated on the 25-frame grid; that
    grid is only ever used as the set of t_values passed in here). Returns
    (C_T values, list of bool — whether Simpson's odd-interval trapezoid
    fallback fired for that t; the caller is expected to assert these are
    all False, since `grid_spec.build` always constructs an odd number of
    points on purpose).
    """
    t_values = np.asarray(t_values, dtype=np.float64)
    a = k2 + k3
    out = np.empty_like(t_values)
    fallback_flags: list[bool] = []

    for i, t_end in enumerate(t_values):
        grid = grid_spec.build(float(t_end))
        C_P_grid = arterial_input(grid, lam, mu)
        integrand1 = np.exp(a * grid) * C_P_grid
        I1, fb1 = simpson_diag(grid, integrand1)
        I2, fb2 = simpson_diag(grid, C_P_grid)
        out[i] = (K1 * k2 / a) * math.exp(-a * float(t_end)) * I1 + (K1 * k3 / a) * I2
        fallback_flags.append(bool(fb1) or bool(fb2))

    return out, fallback_flags
