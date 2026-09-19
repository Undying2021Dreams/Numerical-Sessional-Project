"""M4.4 acceptance tests: the identifiability signature (paper Proposition 12).

Split into three layers so a failure localises:

  * arithmetic   — `spread`, `zeta_from_*`, `fit_accepted` on inputs whose
                   answer is known by hand, no solver involved;
  * construction — `build_observations` / `make_initial_guess` shapes,
                   reproducibility, and the exact `F^2(x_true) == 0` identity;
  * end to end   — an actual IRGNM fit, checking the four claims of
                   Proposition 12 against measured numbers.

The end-to-end thresholds are set ~100x looser than the measured values
(handoffs/RUN_M4.md, M4.4) so they are not brittle, but every one of them is
still many orders of magnitude tighter than what a broken null-space
direction would produce — mutating the K1 or lambda column of the Jacobian
turns the measured spread from ~1e-9 into ~1e-2.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.config import LAMBDA_SLICE, METABOLIC_START, M_SLICE, N_FRAMES, N_REGIONS
from src.identifiability import (
    FIT_RESIDUAL_TOL,
    TISSUE_ONLY_MASK,
    T_FRAMES,
    X_TRUE,
    build_observations,
    fit_accepted,
    make_initial_guess,
    metabolic_relative_errors,
    run_identifiability_case,
    spread,
    summarise,
    zeta_from_K1,
    zeta_from_lambda,
)
from src.jacobian import forward_operator


def _scale_along_null_direction(c: float) -> np.ndarray:
    """x_true with every K1^i multiplied by c and every lambda_j divided by c
    — the exact invariance of F^1 (see tests/test_null_space.py)."""
    x = X_TRUE.copy()
    x[LAMBDA_SLICE] = x[LAMBDA_SLICE] / c
    for i in range(N_REGIONS):
        x[METABOLIC_START + 3 * i] *= c
    return x


# ---------------------------------------------------------------------------
# Layer 1 — arithmetic
# ---------------------------------------------------------------------------
def test_spread_of_identical_values_is_zero():
    s = spread(np.array([1.25, 1.25, 1.25, 1.25]))
    assert s["max_over_min_minus_1"] == 0.0
    assert s["rel_range"] == 0.0
    assert s["cv"] == 0.0
    assert s["mean"] == pytest.approx(1.25)


def test_spread_matches_hand_computed_values():
    s = spread(np.array([1.0, 1.0, 1.0, 1.1]))
    assert s["max_over_min_minus_1"] == pytest.approx(0.1)
    assert s["mean"] == pytest.approx(1.025)
    assert s["rel_range"] == pytest.approx(0.1 / 1.025)


@pytest.mark.parametrize("bad", [
    np.array([1.0, np.nan, 1.0, 1.0]),
    np.array([1.0, np.inf, 1.0, 1.0]),
    np.array([1.0, -1.0, 1.0, 1.0]),  # sign change: ratio statistics meaningless
])
def test_spread_returns_nan_rather_than_a_misleading_number(bad):
    assert np.isnan(spread(bad)["max_over_min_minus_1"])


@pytest.mark.parametrize("c", [0.5, 0.9, 1.0, 1.3, 2.0])
def test_zeta_extractors_recover_the_scaling_exactly(c):
    """If the parameters really are x_true scaled along the null direction by
    c, both extractors must return exactly c, four times over."""
    x = _scale_along_null_direction(c)
    assert zeta_from_K1(x) == pytest.approx(np.full(N_REGIONS, c), rel=1e-14)
    assert zeta_from_lambda(x) == pytest.approx(np.full(N_REGIONS, c), rel=1e-14)
    assert spread(zeta_from_K1(x))["max_over_min_minus_1"] == pytest.approx(0.0, abs=1e-13)


def test_null_direction_leaves_the_tissue_observations_unchanged():
    """The premise of the whole experiment: a scaled parameter vector fits the
    tissue data exactly as well as the truth does. Guards against the
    extractors measuring a direction the forward model does not actually have."""
    obs = build_observations()
    for c in (0.7, 1.4):
        y_scaled = forward_operator(
            _scale_along_null_direction(c), T_FRAMES,
            obs["s_blood"], obs["C_WB_data"], include_blood=False,
        )
        max_abs = float(np.max(np.abs(y_scaled - obs["y"])))
        assert max_abs < 1e-12 * float(np.max(np.abs(obs["y"]))), (
            f"c={c}: tissue observations changed by {max_abs:.3e}"
        )


def test_metabolic_relative_errors_are_zero_at_ground_truth():
    e = metabolic_relative_errors(X_TRUE)
    assert e["K1_rel_error_max"] == pytest.approx(0.0, abs=1e-15)
    assert e["k2_rel_error_max"] == pytest.approx(0.0, abs=1e-15)
    assert e["k3_rel_error_max"] == pytest.approx(0.0, abs=1e-15)


def test_fit_accepted_noiseless_gates_on_relative_residual():
    assert fit_accepted(diverged=False, rel_residual=1e-9, converged_at=None)
    assert not fit_accepted(diverged=False, rel_residual=3e3, converged_at=None)
    assert not fit_accepted(diverged=False, rel_residual=10 * FIT_RESIDUAL_TOL, converged_at=None)


def test_fit_accepted_morozov_gates_on_the_discrepancy_principle():
    kw = dict(noise_floor_rel=1e-2, stopping="morozov")
    assert fit_accepted(diverged=False, rel_residual=9e-3, converged_at=42, **kw)
    assert not fit_accepted(diverged=False, rel_residual=9e-3, converged_at=None, **kw)


def test_fit_accepted_rms_gates_on_reaching_the_noise_floor():
    """Under the rms convention the discrepancy rule never fires, so
    `converged_at is None` must NOT by itself reject a good fit."""
    kw = dict(converged_at=None, noise_floor_rel=1e-2, stopping="rms")
    assert fit_accepted(diverged=False, rel_residual=0.88e-2, **kw)   # measured good band
    assert fit_accepted(diverged=False, rel_residual=1.9e-2, **kw)    # just inside 2.0x
    assert not fit_accepted(diverged=False, rel_residual=2.1e-2, **kw)
    assert not fit_accepted(diverged=False, rel_residual=6.0e-2, **kw)  # measured bad band


def test_fit_accepted_rejects_a_morozov_run_that_took_no_step():
    """converged_at == 0 means the discrepancy rule fired before any IRGNM
    step, so x_final IS the initial guess and its zeta carries no information
    about identifiability. Measured: 20/20 low_count runs at delta_x = 0.1."""
    kw = dict(rel_residual=0.4, noise_floor_rel=0.5, stopping="morozov")
    assert not fit_accepted(diverged=False, converged_at=0, **kw)
    assert fit_accepted(diverged=False, converged_at=1, **kw)


def test_zeta_from_lambda_is_nan_not_an_exception_when_lambda_hits_zero():
    x = X_TRUE.copy()
    x[LAMBDA_SLICE.start] = 0.0
    z = zeta_from_lambda(x)
    assert np.isnan(z[0]) and np.all(np.isfinite(z[1:]))
    assert np.isnan(spread(z)["max_over_min_minus_1"])


def test_fit_accepted_always_rejects_divergence_and_nan():
    assert not fit_accepted(diverged=True, rel_residual=1e-9, converged_at=None)
    assert not fit_accepted(diverged=False, rel_residual=float("nan"), converged_at=None)
    assert not fit_accepted(diverged=True, rel_residual=1e-3, converged_at=7,
                            noise_floor_rel=1e-2, stopping="morozov")


def test_unknown_stopping_convention_is_rejected():
    with pytest.raises(ValueError):
        run_identifiability_case(delta_x=0.1, seed_idx=0, stopping="discrepancy")


def test_the_two_stopping_conventions_differ_under_noise_but_not_without_it():
    """rms never fires (runs to max_iter); morozov fires early. With no noise
    both must give bit-identical results, since delta_y = 0 either way."""
    a = run_identifiability_case(delta_x=0.1, seed_idx=0, stopping="rms")
    b = run_identifiability_case(delta_x=0.1, seed_idx=0, stopping="morozov")
    assert a["x_final"] == b["x_final"]

    a = run_identifiability_case(delta_x=0.1, seed_idx=0, noise_level="high_count",
                                 stopping="rms")
    b = run_identifiability_case(delta_x=0.1, seed_idx=0, noise_level="high_count",
                                 stopping="morozov")
    assert a["converged_at"] is None and a["n_iterations"] == 300
    assert b["converged_at"] is not None and b["n_iterations"] < 300


def test_summarise_counts_diverged_and_stalled_separately():
    def stub(diverged, accepted, zeta=1.0, sp=1e-9):
        return {"diverged": diverged, "fit_accepted": accepted,
                "spread_K1": {"max_over_min_minus_1": sp}, "zeta_mean": zeta,
                "abs_zeta_minus_1": abs(zeta - 1.0),
                "zeta_K1_vs_lambda_rel_diff": 1e-8,
                "K1_rel_error_max": 0.1, "k2_rel_error_max": 1e-7,
                "k3_rel_error_max": 1e-7}
    s = summarise([stub(False, True), stub(True, False), stub(False, False)])
    assert (s["n_runs"], s["n_diverged"], s["n_stalled"], s["n_accepted"]) == (3, 1, 1, 1)


# ---------------------------------------------------------------------------
# Layer 2 — construction
# ---------------------------------------------------------------------------
def test_tissue_only_mask_freezes_exactly_the_plasma_fraction_block():
    assert TISSUE_ONLY_MASK.sum() == 20
    assert not TISSUE_ONLY_MASK[M_SLICE].any()
    assert TISSUE_ONLY_MASK[LAMBDA_SLICE].all()
    assert TISSUE_ONLY_MASK[METABOLIC_START:].all()


def test_observation_shapes():
    assert build_observations()["y"].shape == (N_REGIONS * N_FRAMES,)
    assert build_observations()["include_blood"] is False
    for n_blood in (1, 2, 4):
        obs = build_observations(blood_frame_indices=tuple(range(3, 3 + n_blood)))
        assert obs["y"].shape == (N_REGIONS * N_FRAMES + n_blood,)
        assert obs["include_blood"] is True


def test_blood_block_vanishes_at_ground_truth():
    """F^2 = C_WB_true*f_true - C_P_true == 0 identically, so with m frozen at
    truth the blood row IS a measurement of C_P. This is the justification
    for implementing step 5 with the existing operator (D-M4-6)."""
    n_tac = N_REGIONS * N_FRAMES
    for fi in (3, 10, 17, 24):
        obs = build_observations(blood_frame_indices=(fi,))
        assert abs(float(obs["y"][n_tac])) < 1e-12


def test_initial_guess_is_reproducible_and_freezes_m():
    a = make_initial_guess(0.2, 7)
    b = make_initial_guess(0.2, 7)
    assert np.array_equal(a, b)
    assert np.array_equal(a[M_SLICE], X_TRUE[M_SLICE])
    assert not np.array_equal(a, make_initial_guess(0.2, 8))
    rel = float(np.sqrt((a - X_TRUE) @ (a - X_TRUE)) / np.sqrt(X_TRUE @ X_TRUE))
    assert 0.0 < rel < 1.0


def test_noise_changes_only_the_tac_block_and_is_seed_reproducible():
    n_tac = N_REGIONS * N_FRAMES
    clean = build_observations(blood_frame_indices=(3,))
    noisy = build_observations(blood_frame_indices=(3,), noise_level="normal_count", seed_idx=1)
    again = build_observations(blood_frame_indices=(3,), noise_level="normal_count", seed_idx=1)
    assert np.array_equal(noisy["y"], again["y"])
    assert noisy["y"][n_tac] == clean["y"][n_tac]          # blood row untouched
    assert not np.array_equal(noisy["y"][:n_tac], clean["y"][:n_tac])
    assert noisy["delta_y"] > 0.0


def test_unknown_noise_level_is_rejected():
    with pytest.raises(ValueError):
        build_observations(noise_level="medium_count")


# ---------------------------------------------------------------------------
# Layer 3 — end to end: the four claims of Proposition 12
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def tissue_only_run():
    return run_identifiability_case(delta_x=0.1, seed_idx=0)


def test_tissue_only_fit_is_accepted(tissue_only_run):
    assert tissue_only_run["fit_accepted"], tissue_only_run["failure_reason"]
    assert tissue_only_run["rel_residual"] < FIT_RESIDUAL_TOL


def test_claim1_the_four_K1_ratios_coincide(tissue_only_run):
    """Proposition 12's central claim. Measured 2.0e-09; a broken null
    direction gives ~1e-2, so 1e-6 separates the two cases decisively."""
    assert tissue_only_run["spread_K1"]["max_over_min_minus_1"] < 1e-6


def test_claim2_K1_is_nonetheless_wrong(tissue_only_run):
    """The ratios coinciding would be vacuous if they all coincided AT 1."""
    assert tissue_only_run["K1_rel_error_max"] > 1e-3
    assert abs(tissue_only_run["zeta_mean"] - 1.0) > 1e-3


def test_claim3_k2_and_k3_are_recovered_in_the_same_run(tissue_only_run):
    assert tissue_only_run["k2_rel_error_max"] < 1e-5
    assert tissue_only_run["k3_rel_error_max"] < 1e-5


def test_claim4_lambda_is_scaled_by_one_over_zeta(tissue_only_run):
    """The compensation must land in C_P, and the two independent estimates of
    zeta must agree."""
    assert tissue_only_run["zeta_K1_vs_lambda_rel_diff"] < 1e-5
    assert spread(np.asarray(tissue_only_run["zeta_from_lambda"]))["max_over_min_minus_1"] < 1e-5


@pytest.mark.parametrize("frame_index", [3, 10, 17, 24])
def test_a_single_C_P_measurement_collapses_zeta_to_one(frame_index):
    r = run_identifiability_case(delta_x=0.1, seed_idx=1, blood_frame_indices=(frame_index,))
    assert r["fit_accepted"], r["failure_reason"]
    assert r["abs_zeta_minus_1"] < 1e-4, (
        f"frame {frame_index}: zeta = {r['zeta_mean']!r}, expected ~1"
    )
    assert r["K1_rel_error_max"] < 1e-4
