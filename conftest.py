"""Repo-root pytest conftest.

Ensures the repository root is on sys.path so `import src...` and
`import config` resolve the same way regardless of how pytest is invoked
(from repo root, from a subdirectory, or via `python -m pytest`).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
