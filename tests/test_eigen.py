"""M3 acceptance tests for src/eigen.py (power method, inverse power iteration).

Track B usage (clearly marked): `numpy.linalg.eigvalsh` is used here, in
tests/, purely as an independent reference. `src/eigen.py` itself uses only
our own `src.linalg.lu_factor`/`lu_solve` and elementwise numpy ops.
"""
from __future__ import annotations

import numpy as np

from src.eigen import inverse_power_iteration, power_method, symmetric_eigendecomposition


def _random_symmetric_psd(n: int, rng: np.random.Generator) -> np.ndarray:
    A = rng.normal(size=(n, n))
    return A.T @ A + 1e-6 * np.eye(n)  # PSD, tiny ridge to avoid exact singularity


def test_power_method_largest_eigenvalue_matches_numpy():
    rng = np.random.default_rng(1)
    for n in (5, 10, 23, 40):
        M = _random_symmetric_psd(n, rng)
        lam, v, _ = power_method(M)
        ref = np.linalg.eigvalsh(M)[-1]  # Track B reference
        assert abs(lam - ref) / abs(ref) < 1e-8, f"n={n}: power method {lam} vs numpy {ref}"
        # eigenvector check: M v ~= lam v. Looser than the eigenvalue check
        # above on purpose: power iteration's eigenvector convergence rate
        # is governed by the ratio of the two largest eigenvalues (linear),
        # while the Rayleigh-quotient eigenvalue estimate converges
        # quadratically — for some random instances the top two eigenvalues
        # land close together, slowing eigenvector (not eigenvalue)
        # convergence at the default max_iter/tol.
        resid = np.linalg.norm(M @ v - lam * v) / np.linalg.norm(v)
        assert resid < 1e-4


def test_inverse_power_iteration_smallest_eigenvalue_matches_numpy():
    rng = np.random.default_rng(2)
    for n in (5, 10, 23):
        M = _random_symmetric_psd(n, rng)
        lam, v, _ = inverse_power_iteration(M, shift=0.0)
        ref = np.linalg.eigvalsh(M)[0]
        assert abs(lam - ref) / abs(ref) < 1e-6, f"n={n}: inverse power {lam} vs numpy {ref}"


def test_full_spectrum_via_deflation_matches_numpy_for_large_and_medium_eigenvalues():
    rng = np.random.default_rng(3)
    M = _random_symmetric_psd(15, rng)
    evals, evecs = symmetric_eigendecomposition(M)
    ref = np.linalg.eigvalsh(M)[::-1]  # descending, Track B reference

    # The top half of the spectrum should match tightly; deflation
    # accumulates round-off proportional to the largest eigenvalue already
    # removed, so the bottom of the spectrum is expected to be less
    # accurate for ill-conditioned matrices (documented in DECISIONS.md
    # D-M3-4) — this test only asserts the well-conditioned regime.
    half = len(evals) // 2
    rel_err_top_half = np.max(np.abs(evals[:half] - ref[:half]) / np.abs(ref[:half]))
    assert rel_err_top_half < 1e-6, f"top half spectrum rel err {rel_err_top_half:.3e}"

    # orthogonality of the returned eigenvectors (sanity on the deflation itself)
    gram = evecs.T @ evecs
    off_diag_max = np.max(np.abs(gram - np.eye(len(evals))))
    assert off_diag_max < 1e-4


def test_inverse_power_iteration_accurate_on_ill_conditioned_matrix():
    """`inverse_power_iteration` targets the smallest eigenvalue directly
    (via our own LU on the shifted matrix), so its accuracy should not
    depend on how large the largest eigenvalue is — checked here on a
    matrix with condition number ~1e10. This is the method M3's actual
    null-space experiment relies on for identifying the near-null direction
    (DECISIONS.md D-M3-4); see `handoffs/RUN_M3.md` for the measured
    real-problem finding that the *deflation-based* full spectrum's tail
    eigenvector became unreliable (near-zero eigenvalue found, but aligned
    with the wrong direction) on the actual, highly clustered near-zero
    eigenspace of this project's restricted Jacobian F1 — a clustering
    effect specific to that matrix (three exactly-zero directions from the
    m-block plus one near-zero K1/lambda direction) that is not
    straightforward to reproduce with a small well-separated synthetic
    spectrum, so it is reported as a measured finding on the real matrix
    rather than asserted here on a synthetic stand-in."""
    rng = np.random.default_rng(4)
    n = 20
    Q, _ = np.linalg.qr(rng.normal(size=(n, n)))  # Track B, test-fixture construction only
    true_eigs = np.geomspace(1e10, 1.0, n)
    M = Q @ np.diag(true_eigs) @ Q.T
    M = 0.5 * (M + M.T)  # symmetrize away any round-off asymmetry

    lam_min_direct, v_min_direct, _ = inverse_power_iteration(M, shift=0.0)
    rel_err = abs(lam_min_direct - true_eigs[-1]) / true_eigs[-1]
    assert rel_err < 1e-6, f"inverse power iteration rel err {rel_err:.3e} on cond~1e10 matrix"

    true_v_min = Q[:, -1]  # eigenvector for the smallest eigenvalue, by construction
    cos_align = abs(float(true_v_min @ v_min_direct))
    assert cos_align > 1 - 1e-6, f"eigenvector alignment {cos_align:.8f}, expected ~1"
