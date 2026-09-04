"""Shared plotting helpers.

Matplotlib is not part of the Track A / Track B distinction (it produces no
numbers used in the scientific pipeline, only pictures of numbers computed
elsewhere), so it is used freely here and under src/. This module only
standardises figure style and the results/ save path convention; it must not
contain any numerical algorithm.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: never depend on a display being present
import matplotlib.pyplot as plt

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def set_style() -> None:
    """Apply one consistent, colour-blind-friendlyish style to all figures."""
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 150,
            "font.size": 10,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "lines.linewidth": 1.6,
        }
    )


def save_fig(fig, milestone: str, name: str) -> Path:
    """Save `fig` under results/<milestone>/<name>.png and return the path.

    Every figure that ends up in a handoff report is saved through this
    function so the on-disk location convention is uniform across milestones.
    """
    out_dir = RESULTS_DIR / milestone
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.savefig(path, bbox_inches="tight")
    return path
