"""Track A TAC-level noise model for M4.

Two variants, matching the remaining_task.md M4.1 spec:

  - Poisson-derived: scale C_PET to photon counts (via alpha), draw
    from Poisson(alpha * C_PET[t]), divide back by alpha.
  - Gaussian:        add N(0, sigma_rel^2 * C_PET[t]^2) to each TAC
    point (time- and region-dependent standard deviation, proportional
    to the local signal level).

Both variants are calibrated so the resulting discrepancy level
  delta_y = ||y_noisy - y_clean|| / sqrt(n_obs)
(RMS over all 100 tissue-TAC observations) lands near the paper's three
targets: 0.003 (high count), 0.011 (normal), 0.07 (low count).
See DECISIONS.md D-M4-1 for the calibration procedure and chosen
parameter values.

All randomness uses src.rng primitives; no numpy.random anywhere
(enforced by tests/test_no_library_solvers.py).
"""
from __future__ import annotations

import math

import numpy as np

from src.rng import LCG, normal, poisson, standard_normal


# ---------------------------------------------------------------------------
# 1.  Poisson-derived noise
# ---------------------------------------------------------------------------

def add_poisson_noise(
    rng: LCG,
    C_T: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Apply Poisson-derived noise to a TAC array.

    Model: for each entry c = C_T[i,t],
        c_noisy = Poisson(alpha * c) / alpha

    `alpha` plays the role of a photon-count scale: large alpha => many
    counts => low relative noise.  alpha > 0 is required; raises
    ValueError otherwise.

    C_T shape: (n_regions, n_frames) or (n_frames,) for a single region.
    Returns an array of the same shape and dtype float64.

    Note: any C_T entry <= 0 is left at its original value (Poisson is
    not defined for a negative rate — in practice TACs are non-negative by
    construction, but guard here rather than crash).
    """
    if alpha <= 0.0:
        raise ValueError(f"alpha must be positive, got {alpha!r}")
    C_T = np.asarray(C_T, dtype=np.float64)
    flat = C_T.ravel()
    out = flat.copy()
    for i in range(len(flat)):
        lam = alpha * flat[i]
        if lam > 0.0:
            out[i] = poisson(rng, lam, 1)[0] / alpha
        # lam <= 0: leave as-is (non-positive TAC entries are a model artefact)
    return out.reshape(C_T.shape)


# ---------------------------------------------------------------------------
# 2.  Gaussian noise
# ---------------------------------------------------------------------------

def add_gaussian_noise(
    rng: LCG,
    C_T: np.ndarray,
    sigma_rel: float,
) -> np.ndarray:
    """Apply zero-mean Gaussian noise with a relative (proportional) sigma.

    Model: for each entry c = C_T[i,t],
        c_noisy = c + N(0, (sigma_rel * c)^2)

    i.e. the noise standard deviation scales with the local signal level —
    a simple heteroscedastic model appropriate when we are bypassing the
    sinogram/OSEM imaging chain (remaining_task.md section 3, open decision).

    sigma_rel > 0 is required.  Returns same shape as C_T, dtype float64.
    """
    if sigma_rel <= 0.0:
        raise ValueError(f"sigma_rel must be positive, got {sigma_rel!r}")
    C_T = np.asarray(C_T, dtype=np.float64)
    z = standard_normal(rng, C_T.size)
    return C_T + sigma_rel * C_T * z.reshape(C_T.shape)


# ---------------------------------------------------------------------------
# 3.  Discrepancy level delta_y
# ---------------------------------------------------------------------------

def compute_delta_y(
    C_T_noisy: np.ndarray,
    C_T_clean: np.ndarray,
) -> float:
    """Compute the RMS discrepancy between noisy and clean TAC arrays.

        delta_y = ||C_T_noisy - C_T_clean||_F / sqrt(n_obs)

    where n_obs = C_T.size (total number of scalar observations).
    Both arrays must have the same shape.  Returns a plain Python float.
    """
    C_T_noisy = np.asarray(C_T_noisy, dtype=np.float64)
    C_T_clean = np.asarray(C_T_clean, dtype=np.float64)
    if C_T_noisy.shape != C_T_clean.shape:
        raise ValueError(
            f"shape mismatch: {C_T_noisy.shape} vs {C_T_clean.shape}"
        )
    n = C_T_noisy.size
    diff = C_T_noisy - C_T_clean
    return float(math.sqrt(float(np.sum(diff * diff)) / n))


# ---------------------------------------------------------------------------
# 4.  Monte Carlo calibration helpers
# ---------------------------------------------------------------------------

def _mean_delta_y_poisson(
    C_T_clean: np.ndarray,
    alpha: float,
    n_samples: int,
    root_seed: int,
    label: str,
) -> float:
    """Return the mean delta_y over `n_samples` Poisson realisations."""
    from src.rng import derive_seed
    total = 0.0
    for k in range(n_samples):
        seed = derive_seed(root_seed, label, f"k={k}")
        rng = LCG(seed)
        noisy = add_poisson_noise(rng, C_T_clean, alpha)
        total += compute_delta_y(noisy, C_T_clean)
    return total / n_samples


def _mean_delta_y_gaussian(
    C_T_clean: np.ndarray,
    sigma_rel: float,
    n_samples: int,
    root_seed: int,
    label: str,
) -> float:
    """Return the mean delta_y over `n_samples` Gaussian realisations."""
    from src.rng import derive_seed
    total = 0.0
    for k in range(n_samples):
        seed = derive_seed(root_seed, label, f"k={k}")
        rng = LCG(seed)
        noisy = add_gaussian_noise(rng, C_T_clean, sigma_rel)
        total += compute_delta_y(noisy, C_T_clean)
    return total / n_samples


def calibrate_poisson(
    C_T_clean: np.ndarray,
    target_delta_y: float,
    n_samples: int = 20,
    root_seed: int = 20240401,
    *,
    alpha_lo: float = 1.0,
    alpha_hi: float = 1e8,
    tol: float = 1e-4,
    max_iter: int = 80,
) -> float:
    """Find alpha such that mean Poisson delta_y ≈ target_delta_y.

    Uses bisection on log10(alpha) (delta_y is monotone-decreasing in alpha).
    Returns the calibrated alpha value.

    Parameters
    ----------
    C_T_clean   : clean TAC array, shape (n_regions, n_frames)
    target_delta_y : desired RMS discrepancy level
    n_samples   : number of MC realisations to average over per alpha probe
    root_seed   : root seed for derive_seed; individual realisations are
                  derived deterministically from this
    alpha_lo, alpha_hi : search bracket (must bracket the root)
    tol         : stop when |delta_y_measured - target| < tol * target
    max_iter    : maximum bisection steps (safety only; bisection
                  converges in ~log2((hi-lo)/eps) steps)
    """
    log_lo = math.log10(alpha_lo)
    log_hi = math.log10(alpha_hi)
    for _it in range(max_iter):
        log_mid = 0.5 * (log_lo + log_hi)
        alpha_mid = 10.0 ** log_mid
        dy = _mean_delta_y_poisson(
            C_T_clean, alpha_mid, n_samples, root_seed,
            label=f"calib_poisson_alpha={alpha_mid:.6g}"
        )
        if abs(dy - target_delta_y) < tol * target_delta_y:
            return alpha_mid
        # delta_y decreases as alpha increases (more counts = less noise)
        if dy > target_delta_y:
            log_lo = log_mid   # need more counts (higher alpha)
        else:
            log_hi = log_mid   # already too many counts (lower alpha)
    return 10.0 ** (0.5 * (log_lo + log_hi))


def calibrate_gaussian(
    C_T_clean: np.ndarray,
    target_delta_y: float,
    n_samples: int = 20,
    root_seed: int = 20240401,
    *,
    sigma_lo: float = 1e-6,
    sigma_hi: float = 10.0,
    tol: float = 1e-4,
    max_iter: int = 80,
) -> float:
    """Find sigma_rel such that mean Gaussian delta_y ≈ target_delta_y.

    Uses bisection on log10(sigma_rel) (delta_y is monotone-increasing in
    sigma_rel).  Returns the calibrated sigma_rel value.
    """
    log_lo = math.log10(sigma_lo)
    log_hi = math.log10(sigma_hi)
    for _it in range(max_iter):
        log_mid = 0.5 * (log_lo + log_hi)
        sigma_mid = 10.0 ** log_mid
        dy = _mean_delta_y_gaussian(
            C_T_clean, sigma_mid, n_samples, root_seed,
            label=f"calib_gaussian_sigma={sigma_mid:.6g}"
        )
        if abs(dy - target_delta_y) < tol * target_delta_y:
            return sigma_mid
        # delta_y increases as sigma_rel increases
        if dy < target_delta_y:
            log_lo = log_mid   # need more noise (higher sigma)
        else:
            log_hi = log_mid   # already too noisy (lower sigma)
    return 10.0 ** (0.5 * (log_lo + log_hi))


# ---------------------------------------------------------------------------
# 5.  Named noise levels (paper's three targets, Section 6)
# ---------------------------------------------------------------------------

# Paper section 6: delta_y targets for the three count regimes.
# These are the values we calibrate alpha / sigma_rel to hit.
DELTA_Y_TARGETS: dict[str, float] = {
    "high_count":   0.003,
    "normal_count": 0.011,
    "low_count":    0.070,
}
