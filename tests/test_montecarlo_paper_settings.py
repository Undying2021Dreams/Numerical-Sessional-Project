"""Tests for run_one_cell's paper-settings options (DECISIONS.md D-M5-7)."""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from src.config import N_FRAMES, N_REGIONS
from src.montecarlo import (
    PAPER_SCHEDULES,
    PAPER_TAU,
    _build_noisy_observations,
    run_one_cell,
)

warnings.filterwarnings("ignore")
N_TAC = N_REGIONS * N_FRAMES


def test_paper_schedules_match_section_6_values():
    # Reduced: alpha = 10 * 2^(-i/5), beta = 600 * 2^(-i/7)
    d = PAPER_SCHEDULES["A"].diag(5)
    assert d[-1] == pytest.approx(10.0 * 2 ** (-5 / 5), rel=1e-12)
    assert d[0] == pytest.approx(600.0 * 2 ** (-5 / 7), rel=1e-12)
    # Full, noisy C_WB: alpha = 3000 * 2^(-i/8), beta = 100 * 2^(-i/8), gamma = 400 * 2^(-i/8)
    d = PAPER_SCHEDULES["C"].diag(8)
    assert d[-1] == pytest.approx(1500.0, rel=1e-12)
    assert d[0] == pytest.approx(50.0, rel=1e-12)
    assert d[8] == pytest.approx(200.0, rel=1e-12)
    assert PAPER_TAU == {"A": 9.2, "B": 6.8, "C": 17.6}


def test_setup_a_blood_adds_the_blood_block():
    y_tissue, _, _ = _build_noisy_observations("high_count", 0, "A", 0.1)
    y_blood, _, _ = _build_noisy_observations("high_count", 0, "A", 0.1, setup_a_blood=True)
    assert len(y_tissue) == N_TAC
    assert len(y_blood) == N_TAC + 4
    # Same seed, so the tissue part is identical; only the blood block is added.
    np.testing.assert_array_equal(y_tissue, y_blood[:N_TAC])


def test_blood_at_all_frames_gives_one_reading_per_frame():
    r = run_one_cell("B", "noiseless", 0.1, 0, blood_at_all_frames=True)
    assert not r["diverged"]
    assert r["rel_error_total"][-1] < 1e-4


def test_fixed_stopping_makes_the_discrepancy_rule_fire():
    # D-M4-9: with the RMS the rule never fires on noisy data; with the norm it does.
    old = run_one_cell("B", "high_count", 0.1, 0)
    new = run_one_cell("B", "high_count", 0.1, 0, fixed_stopping=True)
    assert old["converged_at"] is None
    assert new["converged_at"] is not None and new["converged_at"] < old["n_iterations"]


def test_paper_max_iter_caps_noisy_runs_at_200():
    r = run_one_cell("B", "normal_count", 0.1, 0, paper_max_iter=True)
    assert r["n_iterations"] <= 200


def test_no_improvement_flag_follows_the_error_trajectory():
    good = run_one_cell("B", "noiseless", 0.1, 0)
    assert good["no_improvement"] is False
    # At low count the corrected rule fires before the first step (D-M4-11): the
    # answer is the starting guess, which the paper's criterion counts as a failure.
    stuck = run_one_cell("B", "low_count", 0.1, 0, fixed_stopping=True)
    assert stuck["converged_at"] == 0
    assert stuck["diverged"] is False
    assert stuck["no_improvement"] is True
