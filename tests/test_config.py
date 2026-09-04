"""M0 acceptance: repo scaffolding sanity, incl. seeded-RNG cross-process check."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from src import config

ROOT = Path(__file__).resolve().parent.parent


def test_ground_truth_vector_shape_and_layout():
    x = config.ground_truth_vector()
    assert x.shape == (23,)
    assert len(config.UNKNOWN_LAYOUT) == 23


def test_frame_schedule_has_25_frames():
    assert config.N_FRAMES == 25
    edges = config.frame_edges_minutes()
    mids = config.frame_midtimes_minutes()
    widths = config.frame_widths_minutes()
    assert edges.shape == (26,)
    assert mids.shape == (25,)
    assert widths.shape == (25,)
    assert edges[0] == 0.0
    # total scan duration: 4*5+4*10+4*30+2*60+3*150+6*300+2*600 = 3750 s = 62.5 min
    assert abs(edges[-1] - 3750.0 / 60.0) < 1e-9


def test_config_hash_is_deterministic_within_process():
    assert config.config_hash() == config.config_hash()
    assert len(config.config_hash()) == 12


def test_config_hash_deterministic_across_processes():
    """The config hash (used to stamp every results/ artifact) must not depend
    on process-specific state (PYTHONHASHSEED, object ids, dict ordering)."""
    code = "import sys; sys.path.insert(0, %r); from src import config; print(config.config_hash())" % str(ROOT)
    out1 = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    out2 = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
        env={"PYTHONHASHSEED": "1", **__import__("os").environ},
    )
    assert out1.stdout.strip() == out2.stdout.strip()
    assert out1.stdout.strip() == config.config_hash()


def test_seeded_rng_identical_streams_across_processes():
    """M0 acceptance: 'Seeded RNG returns identical streams across processes.'"""
    code = (
        "import sys; sys.path.insert(0, %r); "
        "from src.rng import LCG; "
        "print(list(LCG(seed=12345).uniform_array(10)))" % str(ROOT)
    )
    out1 = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    out2 = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out1.returncode == 0 and out2.returncode == 0
    assert out1.stdout == out2.stdout
    assert out1.stdout.strip() != ""
