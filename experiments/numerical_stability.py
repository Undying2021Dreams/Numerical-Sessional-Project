"""Numerical reformulations: high-precision checks and paired noiseless fits.

Historical formulas below are comparison controls only. All fits still use
our own IRGNM/QR, identical observations and initial guesses, with no tuning.
Decimal is an independent reference for scalar functions, never a fitter.
"""
from __future__ import annotations

from contextlib import contextmanager
from decimal import Decimal, localcontext
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments._common import save_json
import src.jacobian as jac
import src.montecarlo as mc
from src.forward_model import _phi1, _phi1_prime, _convolution_kernel


def legacy_prime(x):
    return 1.0 + (x - 1.0) * jac._phi2(x)


def legacy_closed(t, K1, k2, k3, lam, mu):
    a = k2 + k3
    return (K1*k2/a)*np.exp(-a*t)*t*(_phi1(np.outer(t, a+mu)) @ lam) + \
           (K1*k3/a)*t*(_phi1(np.outer(t, mu)) @ lam)


def legacy_derivatives(t, K1, k2, k3, lam, mu):
    """Pre-adoption _dCT_block arithmetic, retained as an explicit control."""
    a = k2 + k3
    pd, pm = _phi1(np.outer(t, a+mu)), _phi1(np.outer(t, mu))
    dd, dm = legacy_prime(np.outer(t, a+mu)), legacy_prime(np.outer(t, mu))
    decay = np.exp(-a*t)
    A1, A2 = decay*t*(pd @ lam), t*(pm @ lam)
    dK1 = (k2/a)*A1 + (k3/a)*A2
    dlam = (K1*k2/a)*(decay*t)[:, None]*pd + (K1*k3/a)*t[:, None]*pm
    dmu = lam[None, :]*((K1*k2/a)*(decay*t**2)[:, None]*dd + (K1*k3/a)*(t**2)[:, None]*dm)
    da = -t*A1 + decay*t**2*(dd @ lam)
    dk2 = K1*((k3/a**2)*(A1-A2)+(k2/a)*da)
    dk3 = K1*((k2/a**2)*(A2-A1)+(k2/a)*da)
    return dK1, dk2, dk3, dlam, dmu


@contextmanager
def legacy_evaluation():
    old = jac.closed_form_C_T, jac._dCT_block
    jac.closed_form_C_T, jac._dCT_block = legacy_closed, legacy_derivatives
    try:
        yield
    finally:
        jac.closed_form_C_T, jac._dCT_block = old


def scalar_checks():
    rows = []
    for x in (-1e8, -1000., -100., -2., -1., -1e-8, 0., 1e-8, 1., 2., 10.):
        with localcontext() as ctx:
            ctx.prec = 80
            z = Decimal.from_float(x)
            ref = float(((z-1)*z.exp()+1)/z**2) if x else .5
        values = {"legacy": float(legacy_prime(np.array([x]))[0]),
                  "stable": float(_phi1_prime(np.array([x]))[0])}
        rows.append({"x": x, "reference": ref, **values,
                     **{name+"_relative_error": abs(value-ref)/abs(ref) for name, value in values.items()}})
    t, a, mu = 60., 20., -.01
    with np.errstate(over="ignore", invalid="ignore"):
        old = float((np.exp(-a*t)*t*_phi1(np.array([(a+mu)*t])))[0])
    new = float(_convolution_kernel(np.array([t]), a, np.array([mu]))[0, 0])
    with localcontext() as ctx:
        ctx.prec = 80
        td, ad, md = map(Decimal.from_float, (t, a, mu))
        ref = float(((md*td).exp()-(-ad*td).exp())/(ad+md))
    return {"derivative": rows, "overflow_example": {
        "t": t, "a": a, "mu": mu, "legacy_finite": bool(np.isfinite(old)),
        "legacy_value": old if np.isfinite(old) else None, "stable_value": new,
        "reference": ref, "stable_relative_error": abs(new-ref)/abs(ref)}}


def compact(run):
    err = run["rel_error_total"]
    final = err[-1] if err and np.isfinite(err[-1]) else None
    return {"diverged": run["diverged"], "no_improvement": run["no_improvement"],
            "n_iterations": run["n_iterations"], "initial_error": err[0],
            "final_error": final, "failure_reason": run["failure_reason"]}


def paired_fits():
    rows = []
    elapsed_by_arm = {"legacy": 0.0, "stable": 0.0}
    options = dict(fixed_stopping=True, paper_max_iter=True,
                   paper_hyperparams=True, setup_a_blood=True)
    build = mc._build_noisy_observations
    for setup in mc.SETUP_NAMES:
        for dx in mc.DELTA_X_VALUES:
            for k in range(mc.N_SEEDS):
                # Freeze data before switching evaluation, so neither arm gets
                # an observation vector generated with its own rounding errors.
                data = build("noiseless", k, setup, dx, setup_a_blood=True)
                mc._build_noisy_observations = lambda *args, **kwargs: tuple(
                    value.copy() if isinstance(value, np.ndarray) else value for value in data)
                try:
                    start = time.perf_counter()
                    with legacy_evaluation():
                        baseline = mc.run_one_cell(setup, "noiseless", dx, k, **options)
                    elapsed_by_arm["legacy"] += time.perf_counter() - start
                    start = time.perf_counter()
                    stable = mc.run_one_cell(setup, "noiseless", dx, k, **options)
                    elapsed_by_arm["stable"] += time.perf_counter() - start
                finally:
                    mc._build_noisy_observations = build
                assert baseline["seed"] == stable["seed"]
                assert baseline["rel_error_total"][0] == stable["rel_error_total"][0]
                rows.append({"setup": setup, "delta_x": dx, "seed_idx": k,
                             "seed": baseline["seed"], "legacy": compact(baseline),
                             "stable": compact(stable)})
            print(f"Paired noiseless fits: setup={setup} dx={dx} done", flush=True)
    totals = {arm: {"diverged": sum(r[arm]["diverged"] for r in rows),
                    "no_improvement": sum(r[arm]["no_improvement"] for r in rows)}
              for arm in ("legacy", "stable")}
    transitions = {
        "rescued": sum(r["legacy"]["no_improvement"] and not r["stable"]["no_improvement"] for r in rows),
        "regressed": sum(not r["legacy"]["no_improvement"] and r["stable"]["no_improvement"] for r in rows),
    }
    return {"options": options, "runs_per_arm": len(rows), "totals": totals,
            "elapsed_s_by_arm": elapsed_by_arm,
            "transitions": transitions, "rows": rows}


def main():
    start = time.perf_counter()
    checks = scalar_checks()
    paired = paired_fits()
    path = save_json("numerical_changes", "stability", {
        "scalar_checks": checks, "paired_noiseless": paired,
        "elapsed_s": time.perf_counter()-start,
        "scope": "Same IRGNM, QR, projection, regularisation, observations, guesses; evaluation formulas differ",
    }, seed=mc.ROOT_SEED)
    print(paired["totals"], paired["transitions"])
    print(path)


if __name__ == "__main__":
    main()
