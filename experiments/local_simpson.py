"""Paired numerical-stability experiment; no library integration/solver used.

Run: python experiments/local_simpson.py
Writes only results/experimental_local_simpson/{comparison.json,convergence.png}.
"""
from __future__ import annotations

import hashlib
import platform
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments._common import save_json
from src.config import ARTERIAL_LAMBDA, ARTERIAL_MU, REGION_KINETICS, frame_midtimes_minutes
from src.forward_model import GradedGridSpec, closed_form_C_T, quadrature_C_T
from src.quadrature import simpson


def main():
    integrals = []
    for name, fun, end, exact in (
        ("sin", np.sin, np.pi, 2.0),
        ("exp", np.exp, 1.0, np.expm1(1.0)),
        ("runge", lambda x: 1.0 / (1.0 + x*x), 1.0, np.pi / 4.0),
    ):
        for n in (17, 33, 65, 129, 401, 1601, 6401, 12801, 25601):
            x = np.linspace(0.0, end, n)
            y = fun(x)
            row = {"integral": name, "n": n, "exact": float(exact)}
            for label, local in (("baseline", False), ("local", True)):
                value = simpson(x, y, local_weights=local)
                row[label + "_value"] = value
                row[label + "_absolute_error"] = abs(value - exact)
            integrals.append(row)

    translations = []
    u = np.array([0.0, 0.125, 0.5, 0.75, 1.0])
    y = 2.0 + 3.0*u + 4.0*u*u
    exact = 2.0 + 1.5 + 4.0/3.0
    for offset in (0.0, 1000.0, 1e6):
        translations.append({
            "offset": offset, "exact": exact,
            **{label + "_absolute_error": abs(simpson(offset + u, y, local_weights=local) - exact)
               for label, local in (("baseline", False), ("local", True))},
        })

    pet = []
    t = frame_midtimes_minutes()
    for n in (401, 1601, 6401, 12801):
        row = {"n": n, "q": 3.0, "regions": {}}
        for region, (K1, k2, k3) in REGION_KINETICS.items():
            exact = closed_form_C_T(t, K1, k2, k3, ARTERIAL_LAMBDA, ARTERIAL_MU)
            errors = {}
            for label, local in (("baseline", False), ("local", True)):
                values, flags = quadrature_C_T(
                    t, K1, k2, k3, ARTERIAL_LAMBDA, ARTERIAL_MU,
                    GradedGridSpec(n=n, q=3.0), local_weights=local,
                )
                assert not any(flags)
                errors[label + "_max_relative_error"] = float(np.max(np.abs((values - exact) / exact)))
            row["regions"][region] = errors
        for label in ("baseline", "local"):
            key = label + "_max_relative_error"
            row[key] = max(r[key] for r in row["regions"].values())
        pet.append(row)
        print(f"PET n={n}: baseline={row['baseline_max_relative_error']:.6e}, "
              f"local={row['local_max_relative_error']:.6e}", flush=True)

    root = Path(__file__).resolve().parent.parent
    hashes = {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
              for name in ("src/quadrature.py", "src/forward_model.py", "experiments/local_simpson.py")}
    path = save_json("experimental_local_simpson", "comparison", {
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "platform": platform.platform()},
        "source_sha256": hashes,
        "method": "Same samples, float64, pair order and sequential summation; only weights differ",
        "stochastic": False,
        "known_integrals": integrals, "translation_check": translations, "pet_forward": pet,
    }, seed=None)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    for label, color in (("baseline", "#c35b35"), ("local", "#176b87")):
        rows = [r for r in integrals if r["integral"] == "sin"]
        # Keep raw zeros in JSON; skip them on a logarithmic axis.
        errs = [r[label + "_absolute_error"] or np.nan for r in rows]
        axes[0].loglog([r["n"] for r in rows], errs, "o-", color=color, label=label)
        axes[1].loglog([r["n"] for r in pet], [r[label + "_max_relative_error"] for r in pet],
                       "o-", color=color, label=label)
    axes[0].set(title="Known integral: sin(x) on [0, pi]", ylabel="Absolute error")
    axes[1].set(title="PET: all 4 regions and 25 frames", ylabel="Maximum relative error")
    for ax in axes:
        ax.set_xlabel("Grid points")
        ax.grid(True, which="both", alpha=0.2)
        ax.legend()
    fig.suptitle("Same Simpson rule, different floating-point evaluation")
    fig.savefig(path.parent / "convergence.png", dpi=180)
    plt.close(fig)
    print(path)


if __name__ == "__main__":
    main()
