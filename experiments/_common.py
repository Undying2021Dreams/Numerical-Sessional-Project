"""Shared helpers for experiments/ scripts: results/ path + provenance stamping.

Not part of src/ (no Track A/B restriction applies here), but deliberately
tiny and non-numerical: it only handles file I/O and JSON serialisation, per
CLAUDE.md section 2 ("every artifact written to results/ records the seed
and config hash that produced it").
"""
from __future__ import annotations

import json
import hashlib
import platform
from importlib.metadata import version
from pathlib import Path
from typing import Any

from src.config import config_hash

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def _json_default(o: Any):
    import numpy as np

    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(f"not JSON serialisable: {type(o)!r}")


def save_json(milestone: str, name: str, data: dict, *, seed: int | None = None) -> Path:
    """Write results/<milestone>/<name>.json, stamped with config_hash (+ seed)."""
    out_dir = RESULTS_DIR / milestone
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.json"
    payload = {
        "provenance": {
            "config_hash": config_hash(), "seed": seed,
            "python": platform.python_version(), "numpy": version("numpy"),
            "platform": platform.platform(),
            "src_sha256": hashlib.sha256(b"".join(
                p.name.encode() + b"\0" + p.read_bytes()
                for p in sorted((RESULTS_DIR.parent / "src").glob("*.py"))
            )).hexdigest(),
        },
        **data,
    }
    path.write_text(json.dumps(payload, indent=2, default=_json_default))
    return path
