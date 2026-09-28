"""Track A: the forward operator F and its analytic Jacobian F', M3.

F: D(F) subset R^23 -> R^(n*T + q), n=4 regions, T=25 frames, q=4 blood samples
(paper eq. 18-20):

  F^1(x) in R^(n*T): C_T for each region, at each of the T frame midtimes
    (this is exactly `closed_form_C_T`, looped over the 4 regions).
  F^2(x) in R^q: C_WB_data(s_l) * f_m(s_l) - C_P(lambda,mu)(s_l), for the q
    blood-sample times s_l (paper's "measurements of C_WB" channel).

The analytic Jacobian reuses stable exponential convolution moments and
phi1/phi1' from `src/forward_model.py`. The historical phi2 helper remains
for reference comparisons. See NUMERICAL_CHANGES_README.md for the numerical
reformulations and their independently checked derivatives.
"""
from __future__ import annotations

import numpy as np

from src.config import (
    LAMBDA_SLICE,
    MU_SLICE,
    M_SLICE,
    N_PARAMS,
    N_REGIONS,
    Q_BLOOD_SAMPLES,
    REGION_NAMES,
    region_slice,
)
from src.forward_model import _phi1, _phi1_prime, _convolution_moments, closed_form_C_T

# ---------------------------------------------------------------------------
# phi2 and phi1' (DECISIONS.md D-M3-3)
# ---------------------------------------------------------------------------
PHI2_SERIES_THRESHOLD = 1e-4  # measured; see DECISIONS.md D-M3-3


def _phi2_series(x: np.ndarray, terms: int = 8) -> np.ndarray:
    """phi2(x) = sum_{k=0}^inf x^k / (k+2)! ; `terms=8` is far more than
    needed for |x| < PHI2_SERIES_THRESHOLD (next term ratio ~x/(terms+2),
    i.e. < 1e-4/10, converging well past float64 precision)."""
    x = np.asarray(x, dtype=np.float64)
    total = np.zeros_like(x)
    xp = np.ones_like(x)
    fact = 2  # (0+2)!
    for k in range(terms):
        total = total + xp / fact
        xp = xp * x
        fact = fact * (k + 3)
    return total


def _phi2_closed_form(x: np.ndarray) -> np.ndarray:
    safe_x = np.where(x == 0.0, 1.0, x)
    return (np.expm1(x) - x) / safe_x**2


def _phi2(x: np.ndarray) -> np.ndarray:
    """phi2(x) = (e^x - 1 - x) / x^2 for x != 0, phi2(0) = 1/2.

    Unlike `phi1`, the closed form `(expm1(x)-x)/x^2` DOES lose precision for
    small x: `expm1(x)` is computed accurately, but its own absolute error
    (~machine_epsilon * |expm1(x)| ~ eps*|x| for small x) is no longer
    negligible once we subtract the exact value `x` and divide by `x^2` —
    the true numerator `expm1(x)-x` is O(x^2), so the relative error of the
    subtraction is O(eps/x), the same scaling phi1 has, but starting from a
    smaller absolute error means it becomes significant at a different
    (measured, not assumed) threshold than phi1's. Below
    `PHI2_SERIES_THRESHOLD`, a truncated Taylor series is used instead (no
    cancellation: every term is added directly, no subtraction of
    near-equal quantities). See DECISIONS.md D-M3-3 for the measurement.
    """
    x = np.asarray(x, dtype=np.float64)
    small = np.abs(x) < PHI2_SERIES_THRESHOLD
    return np.where(small, _phi2_series(x), _phi2_closed_form(x))


# ---------------------------------------------------------------------------
# Parameter (un)packing
# ---------------------------------------------------------------------------
def unpack(x: np.ndarray):
    """x (23,) -> (lam(4,), mu(4,), m(3,), K1(4,), k2(4,), k3(4,)) — K1/k2/k3
    ordered by REGION_NAMES."""
    x = np.asarray(x, dtype=np.float64)
    lam = x[LAMBDA_SLICE]
    mu = x[MU_SLICE]
    m = x[M_SLICE]
    K1 = np.empty(N_REGIONS)
    k2 = np.empty(N_REGIONS)
    k3 = np.empty(N_REGIONS)
    for i in range(N_REGIONS):
        K1[i], k2[i], k3[i] = x[region_slice(i)]
    return lam, mu, m, K1, k2, k3


# ---------------------------------------------------------------------------
# Forward operator F
# ---------------------------------------------------------------------------
def forward_operator(
    x: np.ndarray,
    t_frames: np.ndarray,
    s_blood: np.ndarray,
    C_WB_data: np.ndarray,
    *,
    include_blood: bool = True,
) -> np.ndarray:
    """F(x) in R^(n*T+q) or R^(n*T) depending on `include_blood`.

    F^1 (C_T, all regions x all frames, row-major by region) is always
    included.  F^2 (C_WB_data*f_m - C_P, at the q blood times) is
    appended only when `include_blood=True` (the default, matching the
    paper's eq. 18-20).  Set `include_blood=False` to run the tissue-only
    forward operator needed by M4.2 Setup A and the identifiability
    experiment (M4.4).

    `C_WB_data` is FIXED problem data (the ground-truth C_WB at s_blood),
    not a function of x — matches the paper's eq. (20).
    `s_blood` and `C_WB_data` are ignored when `include_blood=False`.
    """
    lam, mu, m, K1, k2, k3 = unpack(x)
    A, xi1, xi2 = m

    T = len(t_frames)
    F1 = np.empty(N_REGIONS * T)
    for i in range(N_REGIONS):
        F1[i * T : (i + 1) * T] = closed_form_C_T(t_frames, K1[i], k2[i], k3[i], lam, mu)

    if not include_blood:
        return F1

    from src.forward_model import arterial_input, parent_plasma_fraction

    f_s = parent_plasma_fraction(s_blood, A, xi1, xi2)
    C_P_s = arterial_input(s_blood, lam, mu)
    F2 = C_WB_data * f_s - C_P_s

    return np.concatenate([F1, F2])


# ---------------------------------------------------------------------------
# Analytic Jacobian
# ---------------------------------------------------------------------------
def _dCT_block(t: np.ndarray, K1: float, k2: float, k3: float, lam: np.ndarray, mu: np.ndarray):
    """Partial derivatives of closed_form_C_T(t; K1,k2,k3,lam,mu) w.r.t.
    K1, k2, k3, lam (4,), mu (4,), for a SINGLE region, vectorised over t.

    Derivation (DECISIONS.md D-M3-3): with a=k2+k3, delta_j=a+mu_j,
        A1(t) = exp(-a t) * t * sum_j lam_j phi1(delta_j t)
        A2(t) = t * sum_j lam_j phi1(mu_j t)
        C_T = (K1 k2/a) A1 + (K1 k3/a) A2
    Then:
        dCT/dK1  = (k2/a) A1 + (k3/a) A2
        dCT/dlam_j = (K1 k2/a) exp(-at) t phi1(delta_j t) + (K1 k3/a) t phi1(mu_j t)
        dCT/dmu_j  = (K1 k2/a) exp(-at) t^2 lam_j phi1'(delta_j t)
                     + (K1 k3/a) t^2 lam_j phi1'(mu_j t)
        dA1/da = -t*A1 + B1,  B1 = exp(-at) t^2 sum_j lam_j phi1'(delta_j t)
        dCT/dk2 = K1*[ (k3/a^2)*(A1-A2) + (k2/a)*dA1/da ]
        dCT/dk3 = K1*[ (k2/a^2)*(A2-A1) + (k2/a)*dA1/da ]

    These identities are evaluated via the convolution moments, without
    forming large exponentials or subtracting -t*A1+B1 numerically.
    """
    t = np.asarray(t, dtype=np.float64)
    a = k2 + k3
    om = np.outer(t, mu)  # (T,p)
    phi1_m = _phi1(om)
    phi1p_m = _phi1_prime(om)
    kernel, kernel_mu, kernel_a = _convolution_moments(t, a, mu)
    S2 = phi1_m @ lam  # (T,)
    A1 = kernel @ lam
    A2 = t * S2

    dCT_dK1 = (k2 / a) * A1 + (k3 / a) * A2

    dCT_dlam = (K1 * k2 / a) * kernel + (K1 * k3 / a) * t[:, None] * phi1_m

    dCT_dmu = lam[None, :] * (
        (K1 * k2 / a) * kernel_mu + (K1 * k3 / a) * (t**2)[:, None] * phi1p_m
    )

    dA1_da = kernel_a @ lam

    dCT_dk2 = K1 * ((k3 / a**2) * (A1 - A2) + (k2 / a) * dA1_da)
    dCT_dk3 = K1 * ((k2 / a**2) * (A2 - A1) + (k2 / a) * dA1_da)

    return dCT_dK1, dCT_dk2, dCT_dk3, dCT_dlam, dCT_dmu


def analytic_jacobian(
    x: np.ndarray,
    t_frames: np.ndarray,
    s_blood: np.ndarray,
    C_WB_data: np.ndarray,
    *,
    include_blood: bool = True,
) -> np.ndarray:
    """F'(x) in R^((n*T+q) x 23) or R^(n*T x 23) depending on `include_blood`.

    See module docstring for the block structure; `_dCT_block` for the
    per-region tissue derivatives.  `C_WB_data` is fixed problem data (see
    `forward_operator`), needed for the m-block of F^2.
    When `include_blood=False`, only the F^1 rows are returned (the F^2
    block is dropped entirely, as required for M4.2 Setup A and M4.4).
    `s_blood` and `C_WB_data` are ignored when `include_blood=False`.
    """
    lam, mu, m, K1, k2, k3 = unpack(x)
    A, xi1, xi2 = m
    T = len(t_frames)
    q = len(s_blood)
    n = N_REGIONS

    n_rows = n * T + (q if include_blood else 0)
    Jac = np.zeros((n_rows, N_PARAMS))

    for i in range(n):
        dK1, dk2, dk3, dlam, dmu = _dCT_block(t_frames, K1[i], k2[i], k3[i], lam, mu)
        rows = slice(i * T, (i + 1) * T)
        Jac[rows, LAMBDA_SLICE] = dlam
        Jac[rows, MU_SLICE] = dmu
        rs = region_slice(i)
        Jac[rows, rs.start] = dK1
        Jac[rows, rs.start + 1] = dk2
        Jac[rows, rs.start + 2] = dk3
        # M block (F^1 does not depend on m at all): stays zero.

    if not include_blood:
        return Jac

    # F^2 block: C_WB_data(s_l)*f_m(s_l) - C_P(lam,mu)(s_l)
    s = np.asarray(s_blood, dtype=np.float64)
    C_WB_data = np.asarray(C_WB_data, dtype=np.float64)
    exp_mu_s = np.exp(np.outer(s, mu))  # (q,p)
    f2_rows = slice(n * T, n * T + q)

    Jac[f2_rows, LAMBDA_SLICE] = -exp_mu_s
    Jac[f2_rows, MU_SLICE] = -lam[None, :] * s[:, None] * exp_mu_s

    exp_xi1_s = np.exp(xi1 * s)
    exp_xi2_s = np.exp(xi2 * s)
    # d/dA[f] = exp(xi1 s) - exp(xi2 s); d/dxi1[f] = A*s*exp(xi1 s);
    # d/dxi2[f] = (1-A)*s*exp(xi2 s).
    Jac[f2_rows, M_SLICE.start] = C_WB_data * (exp_xi1_s - exp_xi2_s)
    Jac[f2_rows, M_SLICE.start + 1] = C_WB_data * A * s * exp_xi1_s
    Jac[f2_rows, M_SLICE.start + 2] = C_WB_data * (1.0 - A) * s * exp_xi2_s

    return Jac
