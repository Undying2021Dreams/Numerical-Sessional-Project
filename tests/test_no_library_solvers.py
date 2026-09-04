"""Track A / Track B guard (CLAUDE.md section 1 — the most important rule here).

Statically checks every .py file under src/ for use of the banned library
routines:

    scipy.optimize, scipy.linalg, scipy.integrate (in fact: any scipy at all),
    numpy.linalg.solve, numpy.linalg.lstsq, numpy.linalg.inv,
    numpy.random (any function in it).

This is done via the AST rather than a plain text grep, deliberately: a plain
substring grep would also flag this file's own docstrings and the
DECISIONS.md-style comments inside src/ modules that *name* the banned
routines to explain why they are not used. The AST walk only looks at actual
Import/ImportFrom/Attribute nodes, i.e. code that would actually execute,
following `import numpy as np`-style aliasing so an aliased call cannot slip
past the guard.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC_DIR = Path(__file__).resolve().parent.parent / "src"

BANNED_TOP_MODULES = {"scipy"}  # entire scipy package banned under src/
BANNED_FULL_NAMES = {
    "numpy.linalg.solve",
    "numpy.linalg.lstsq",
    "numpy.linalg.inv",
}
BANNED_PREFIXES = (
    "numpy.random.",
    "scipy.",
)
BANNED_EXACT_MODULES = {
    "numpy.random",
}


def _iter_src_files():
    return sorted(SRC_DIR.rglob("*.py"))


def _resolve_dotted(node: ast.AST, aliases: dict[str, str]) -> str | None:
    """Reconstruct a dotted attribute/name chain, resolving import aliases.

    e.g. with aliases {"np": "numpy"}, the node for `np.linalg.solve` resolves
    to the string "numpy.linalg.solve".
    """
    parts: list[str] = []
    cur = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
    else:
        return None
    parts.reverse()
    if not parts:
        return None
    head, rest = parts[0], parts[1:]
    resolved_head = aliases.get(head, head)
    return ".".join([resolved_head, *rest]) if rest else resolved_head


def _check_file(path: Path) -> list[str]:
    """Return a list of human-readable violation strings for one file."""
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))

    aliases: dict[str, str] = {}
    violations: list[str] = []
    rel = path.relative_to(SRC_DIR.parent)

    # Pass 1: collect import aliases and flag banned imports directly.
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                real_name = alias.name
                local_name = alias.asname or alias.name.split(".")[0]
                aliases[local_name] = real_name if alias.asname else real_name.split(".")[0]
                # asname aliasing a submodule import, e.g. `import numpy.linalg as la`
                if alias.asname:
                    aliases[alias.asname] = real_name
                top = real_name.split(".")[0]
                if top in BANNED_TOP_MODULES:
                    violations.append(
                        f"{rel}:{node.lineno}: `import {real_name}` "
                        f"(banned top-level module '{top}')"
                    )
                if real_name in BANNED_EXACT_MODULES:
                    violations.append(
                        f"{rel}:{node.lineno}: `import {real_name}` is banned"
                    )
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            top = module.split(".")[0] if module else ""
            if top in BANNED_TOP_MODULES:
                names = ", ".join(a.name for a in node.names)
                violations.append(
                    f"{rel}:{node.lineno}: `from {module} import {names}` "
                    f"(banned top-level module '{top}')"
                )
            if module in BANNED_EXACT_MODULES:
                names = ", ".join(a.name for a in node.names)
                violations.append(
                    f"{rel}:{node.lineno}: `from {module} import {names}` is banned"
                )
            if module == "numpy.linalg":
                for alias in node.names:
                    full = f"numpy.linalg.{alias.name}"
                    if full in BANNED_FULL_NAMES:
                        violations.append(
                            f"{rel}:{node.lineno}: `from numpy.linalg import "
                            f"{alias.name}` is banned"
                        )
            # Track local bindings for later attribute resolution, e.g.
            # `from numpy import linalg` -> aliases['linalg'] = 'numpy.linalg'
            for alias in node.names:
                local_name = alias.asname or alias.name
                if module:
                    aliases[local_name] = f"{module}.{alias.name}"

    # Pass 2: walk attribute-access chains using the alias map from pass 1.
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            dotted = _resolve_dotted(node, aliases)
            if dotted is None:
                continue
            if dotted in BANNED_FULL_NAMES:
                violations.append(f"{rel}:{node.lineno}: use of `{dotted}` is banned")
            elif any(dotted.startswith(p) for p in BANNED_PREFIXES):
                violations.append(f"{rel}:{node.lineno}: use of `{dotted}` is banned")
            elif dotted in BANNED_EXACT_MODULES:
                violations.append(f"{rel}:{node.lineno}: use of `{dotted}` is banned")

    return violations


def test_src_directory_exists():
    assert SRC_DIR.is_dir(), f"expected {SRC_DIR} to exist"


def test_no_banned_library_solvers_under_src():
    files = _iter_src_files()
    assert files, f"no .py files found under {SRC_DIR}; guard would pass vacuously"

    all_violations: list[str] = []
    for path in files:
        all_violations.extend(_check_file(path))

    if all_violations:
        report = "\n".join(all_violations)
        pytest.fail(
            "Track A / Track B violation(s) found under src/ "
            "(see CLAUDE.md section 1):\n" + report
        )


def test_guard_actually_detects_a_violation(tmp_path):
    """Meta-test: prove the guard is not vacuous by feeding it bad code."""
    bad_file = tmp_path / "bad.py"
    bad_file.write_text(
        "import numpy as np\n"
        "def f(A, b):\n"
        "    return np.linalg.solve(A, b)\n"
    )
    tree = ast.parse(bad_file.read_text())
    aliases = {"np": "numpy"}
    found = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            dotted = _resolve_dotted(node, aliases)
            if dotted in BANNED_FULL_NAMES:
                found = True
    assert found, "guard failed to detect np.linalg.solve in a synthetic bad file"
