"""M1 benchmark: trapezoid/Simpson accuracy and measured convergence order.

No library integrator or library least-squares is used anywhere: the
log-log slope fit uses `src.qr.lstsq` (our own Track A QR), not
`numpy.polyfit`. This is stronger than required (tests/ would be allowed to
use a library fit) but keeps this script's headline numbers 100% Track A.

Run: `python3 experiments/m1_quadrature_benchmark.py`
Writes: results/m1/quadrature_benchmark.json, results/m1/quadrature_convergence.png
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments._common import save_json
from src.plotting import save_fig, set_style
from src.quadrature import simpson, trapezoid
from src.qr import lstsq

KNOWN_INTEGRALS = [
    ("sin_0_pi", math.sin, 2.0, 0.0, math.pi),
    ("exp_0_1", math.exp, math.e - 1.0, 0.0, 1.0),
    ("runge_0_1", lambda x: 1.0 / (1.0 + x**2), math.pi / 4.0, 0.0, 1.0),
]

# Full sweep for the log-log plot (shows the round-off floor honestly).
NS_FULL = np.array([9, 17, 33, 65, 129, 257, 513, 1025, 2001])
# Sub-range used for the *fitted order*, chosen (see DECISIONS.md D-M1-6) to
# stay in the truncation-error-dominated regime for all three integrands.
NS_ORDER_FIT_SIMPSON = np.array([9, 17, 33])
NS_ORDER_FIT_TRAP = np.array([9, 17, 33, 65, 129])


def _fit_log_log_slope(hs: np.ndarray, errs: np.ndarray) -> float:
    mask = errs > 0
    log_h = np.log(hs[mask])
    log_e = np.log(errs[mask])
    design = np.column_stack([log_h, np.ones_like(log_h)])
    coeffs, _resid = lstsq(design, log_e)
    return float(coeffs[0])


def errors_over_ns(f, exact, a, b, ns):
    trap_errs, simp_errs = [], []
    for n in ns:
        x = np.linspace(a, b, n)
        y = np.array([f(xi) for xi in x])
        trap_errs.append(abs(trapezoid(x, y) - exact))
        simp_errs.append(abs(simpson(x, y) - exact))
    return np.array(trap_errs), np.array(simp_errs)


def run_all():
    results = {}
    for name, f, exact, a, b in KNOWN_INTEGRALS:
        trap_full, simp_full = errors_over_ns(f, exact, a, b, NS_FULL)
        hs_full = (b - a) / (NS_FULL - 1)

        trap_fit_errs, _ = errors_over_ns(f, exact, a, b, NS_ORDER_FIT_TRAP)
        hs_trap_fit = (b - a) / (NS_ORDER_FIT_TRAP - 1)
        trap_order = _fit_log_log_slope(hs_trap_fit, trap_fit_errs)

        _, simp_fit_errs = errors_over_ns(f, exact, a, b, NS_ORDER_FIT_SIMPSON)
        hs_simp_fit = (b - a) / (NS_ORDER_FIT_SIMPSON - 1)
        simp_order = _fit_log_log_slope(hs_simp_fit, simp_fit_errs)

        results[name] = {
            "exact": exact,
            "ns_full": NS_FULL.tolist(),
            "hs_full": hs_full.tolist(),
            "trap_errs_full": trap_full.tolist(),
            "simp_errs_full": simp_full.tolist(),
            "trap_order_fit_ns": NS_ORDER_FIT_TRAP.tolist(),
            "trap_order_measured": trap_order,
            "simp_order_fit_ns": NS_ORDER_FIT_SIMPSON.tolist(),
            "simp_order_measured": simp_order,
            "simpson_error_floor_min_over_sweep": float(np.min(simp_full)),
            "simpson_error_floor_reached_at_n": int(NS_FULL[np.argmin(simp_full)]),
        }
    return results


def make_figure(results: dict):
    import matplotlib.pyplot as plt

    set_style()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=False)
    for ax, (name, f, exact, a, b) in zip(axes, KNOWN_INTEGRALS):
        r = results[name]
        hs = np.array(r["hs_full"])
        ax.loglog(hs, r["trap_errs_full"], "o-", ms=4, label=f"trapezoid (order {r['trap_order_measured']:.2f})")
        ax.loglog(hs, r["simp_errs_full"], "s-", ms=4, label=f"simpson (order {r['simp_order_measured']:.2f})")
        ax.set_xlabel("h")
        ax.set_ylabel("absolute error")
        ax.set_title(name)
        ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    results = run_all()
    for name, r in results.items():
        print(f"{name}:")
        print(f"  trapezoid measured order (fit over n={r['trap_order_fit_ns']}): {r['trap_order_measured']:.4f}")
        print(f"  simpson   measured order (fit over n={r['simp_order_fit_ns']}): {r['simp_order_measured']:.4f}")
        print(f"  simpson error floor: {r['simpson_error_floor_min_over_sweep']:.3e} at n={r['simpson_error_floor_reached_at_n']}")

    fig = make_figure(results)
    fig_path = save_fig(fig, "m1", "quadrature_convergence")
    print("\nfigure saved:", fig_path)

    out = save_json("m1", "quadrature_benchmark", results)
    print("summary saved:", out)
