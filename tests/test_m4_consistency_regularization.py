"""Integration check for the M4.5 paired variance comparison."""
from __future__ import annotations

from experiments import m4_consistency_regularization as m45


def test_paired_regularisation_reduces_observation_noise_variance(monkeypatch):
    # Eight real seeded observations are enough to catch accidentally swapping
    # the on/off schedules while keeping the full 20-run study in the script.
    monkeypatch.setattr(m45, "N_SEEDS", 8)
    stats = m45.variance()
    for noise in m45.NOISY_LEVELS:
        assert stats[noise]["on"]["variance_trace"] > 0.0
        assert stats[noise]["off_over_on_variance_ratio"] > 10.0
