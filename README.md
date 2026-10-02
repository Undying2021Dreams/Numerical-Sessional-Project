# PET Pharmacokinetic Parameter Identification — CSE 402 Project

A BUET CSE 402 (Numerical Analysis, Simulation and Modeling) sessional project that
reimplements, by hand, the numerical core of:

> Holler, Morina, Schramm (2024). *Exact parameter identification in PET
> pharmacokinetic modeling using the irreversible two tissue compartment model.*
> Phys. Med. Biol. 69 165008.

## The problem

In PET imaging, a radioactive tracer moves from the blood into tissue, where part of it
leaks back out and part of it gets trapped. The irreversible two-tissue compartment
model describes this with three rate constants per brain region: `K1` (uptake), `k2`
(wash-out) and `k3` (trapping).

- **Forward problem:** given the rates and the blood curve, predict the tissue curve the
  scanner measures.
- **Inverse problem:** given the measured curves, recover the rates. This is the goal.

The paper shows that, with several brain regions sharing one blood supply plus a few
blood samples, the rates can be identified exactly from noise-free data. This project
implements and tests that claim numerically.

## Numerical methods, all written by hand

Every solver, integrator and random number generator is our own code. No
`scipy.optimize`, `scipy.linalg`, `scipy.integrate`, `numpy.linalg.solve`/`lstsq`/`inv`
or `numpy.random` is used anywhere under `src/`; NumPy is used only for array storage
and elementwise arithmetic. Library routines appear only in `tests/` and in clearly
marked benchmark scripts, as an independent reference to check our implementations
against. `tests/test_no_library_solvers.py` enforces this automatically.

| Module | What it implements |
|---|---|
| `src/linalg.py` | LU factorisation with partial pivoting, forward/back substitution |
| `src/qr.py` | Householder QR: square solves and linear least squares |
| `src/quadrature.py` | Composite trapezoid and Simpson rules on non-uniform grids |
| `src/rng.py` | Seeded 64-bit LCG, Box-Muller normal sampler, Poisson sampler |
| `src/eigen.py` | Power method and inverse power iteration (condition numbers) |
| `src/forward_model.py` | Closed-form solution of the compartment model (paper eq. 3) and a direct-quadrature evaluation (paper eq. 1) |
| `src/jacobian.py` | Forward operator (23 unknowns → 104 data values) and its analytic Jacobian |
| `src/irgnm.py` | Iteratively regularised Gauss-Newton method (IRGNM) |
| `src/noise.py` | Poisson and Gaussian noise on the time-activity curves |
| `src/montecarlo.py` | Monte Carlo study over setups, noise levels and starting guesses |
| `src/identifiability.py` | Identifiability and null-space experiments |
| `src/config.py` | Ground-truth constants from the paper (Section 5.1, Table 2) |

## Results

- **Linear algebra:** LU and QR reach relative residuals of about 1e-16 on 200 random
  systems. On Hilbert matrices both lose about log10(cond) digits, as theory predicts.
- **Integration:** measured convergence orders of 2.00 (trapezoid) and 4.01 (Simpson).
- **Random numbers:** the uniform generator passes a chi-square test (12.7 against a 5%
  critical value of 16.9); the normal samples have mean 0, variance 1, and negligible
  skew, kurtosis and autocorrelation.
- **Forward model:** the closed-form formula, numerical integration and an independent
  ODE solver agree to `3.3e-10` and `1.7e-13` relative difference across all four brain
  regions and all 25 frame times.
- **Inverse problem (noise-free):** the IRGNM recovers all 23 parameters to a median
  relative error of about `4.3e-7` from starts 10% away from the truth. From worse starts
  some runs diverge (20/20 converge at 10%, 17/20 at 20%, 15/20 at 30%, and 11 or 12 of
  20 at 40%, depending on the machine).
- **Identifiability:** without blood samples, `K1` and the blood-curve scale cannot be
  separated (the null-space direction the paper predicts). One arterial sample removes
  this ambiguity.
- **Noisy data:** a 960-run Monte Carlo study (3 measurement setups × 4 noise levels × 4
  starting errors × 20 seeds) reproduces the paper's qualitative trends: `K1` is
  recovered best, and the low-count setting almost always fails. Divergent runs are
  counted and reported, not discarded. Rerun with the paper's own settings and a
  corrected stopping rule, the failure counts match the paper at high and normal
  count (14 vs 19, 18 vs 35).
- **Numerical stability:** Simpson weights are computed in local coordinates and the
  exponential terms are factored to avoid overflow and cancellation; details in
  [NUMERICAL_CHANGES_README.md](NUMERICAL_CHANGES_README.md).

## Scope

The image-reconstruction pipeline of the paper (brain phantom, PET scanner physics,
sinogram generation, OSEM reconstruction) is not reproduced. Instead, noise is added
directly to the regional time-activity curves, calibrated to the paper's high, normal
and low count settings. Results are therefore compared with the paper by trend, not by
exact numbers.

## Repository layout

```
src/            numerical methods and the model
tests/          pytest suite (176 tests)
experiments/    scripts that regenerate every figure and table, with fixed seeds
results/        generated figures and JSON summaries
```

## Running it

```bash
pip install -r requirements.txt
python3 -m pytest tests/ -q                     # full test suite

python3 experiments/m1_linalg_benchmark.py      # LU / QR
python3 experiments/m1_quadrature_benchmark.py  # trapezoid / Simpson
python3 experiments/m1_rng_benchmark.py         # random number generators
python3 experiments/m2_forward_model.py         # forward model
python3 experiments/m3_irgnm_recovery.py        # inverse problem

python3 experiments/run_all.py                  # regenerate everything (18 scripts, ~38 min)
```

Every stochastic run uses an explicit seed, so results are reproducible.
